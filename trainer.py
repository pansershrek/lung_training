import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from eval.froc import calculate_FROC
from utils.tools import *
from torch.utils.tensorboard import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger

#from eval_coco import *
#from eval.cocoapi_evaluator import COCOAPIEvaluator

#from databuilder.abus import ABUSDetectionDataset
from dataset import Tumor, LungDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm
from apex import amp

class Trainer(object):
    def __init__(self, testing_mode, weight_path, checkpoint_save_dir, resume, gpu_id, accumulate, fp_16, writer, logger, crx_fold_num, dataset_name, eval_interval, npy_name):
        #self.data_root = 'datasets/abus'
        #init_seeds(0)
        self.lung_dataset_name = dataset_name
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.accumulate = accumulate
        self.fp_16 = fp_16
        self.writer = writer
        self.logger = logger
        self.weight_path = weight_path
        self.checkpoint_save_dir = checkpoint_save_dir
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train:print('Using multi scales training')
        else:print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.logger.info('augmentation=False, crx_fold_num= {}'.format(crx_fold_num))
        self.testing_mode = testing_mode
        self.crx_fold_num = crx_fold_num
        
        #train_dataset = ABUSDetectionDataset(testing_mode, augmentation=True, crx_fold_num= crx_fold_num, crx_partition= 'train', crx_valid=True, include_fp=False, root=self.data_root,
        #    batch_size=cfg.TRAIN["BATCH_SIZE"])
        dataset = LungDataset.load(dataset_name)
        dataset.get_data(dataset.pids, name=npy_name)
        train_data, validation_data, test_data = dataset.make_kfolds_using_pids(num_k_folds=5, k_folds_seed=123, current_fold=self.crx_fold_num, valid_test_split=True, portion_list=[3,1,1])
        train_dataset = LungDataset.load(dataset_name)
        train_dataset.data = train_data
        
        #validation_dataset =  LungDataset.load(dataset_name)
        #validation_dataset.data = validation_data
        test_dataset =  LungDataset.load(dataset_name)
        if self.testing_mode==0:
            test_dataset.data = validation_data
        elif self.testing_mode==1:
            test_dataset.data = test_data
        elif self.testing_mode == -1:
            test_dataset.data = train_data[:100]
        else:
            raise TypeError(f"Invalid testing mode: {self.testing_mode}. {{1: 'test', 0: 'val', -1: 'train'}}")

        #useless, cache should implement on yolo dataset
        train_dataset.cacher.cache_size = 0
        test_dataset.cacher.cache_size = 0

        if (0): #debug
            train_dataset.data = train_data[:40]
            test_dataset.data = train_data[:40]

        self.train_dataset = YOLO4_3DDataset(train_dataset, classes=[0, 1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"], cache_size=0)
        #self.train_dataset = data.Build_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        self.epochs = cfg.TRAIN["YOLO_EPOCHS"] if cfg.MODEL_TYPE["TYPE"] == 'YOLOv4' else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True, pin_memory=True
                                           )

        if eval_interval==-1:
            step_per_epoch = int( len(self.train_dataset)/cfg.TRAIN["BATCH_SIZE"] )
            eval_per_steps = 1000
            self.eval_interval = int( eval_per_steps/step_per_epoch )
        else:
            self.eval_interval = eval_interval
        self.logger.info('eval_interval = {}'.format(self.eval_interval))
        #test_dataset = ABUSDetectionDataset(testing_mode, augmentation=False, crx_fold_num= crx_fold_num, crx_partition= 'valid', crx_valid=True, include_fp=False, root=self.data_root,
        #    batch_size=cfg.VAL["BATCH_SIZE"])

        self.test_dataset = YOLO4_3DDataset(test_dataset, classes=[0, 1], img_size=cfg.VAL["TEST_IMG_SIZE"], cache_size=0)
        self.test_dataloader = DataLoader(self.test_dataset,
                                            batch_size=cfg.VAL["BATCH_SIZE"],
                                            num_workers=cfg.VAL["NUMBER_WORKERS"],
                                            shuffle=False, pin_memory=True
                                            )
        #sum([p.flatten().size(0) for p in self.model.parameters()])
        self.model = Build_Model(weight_path=weight_path, resume=resume, dims=3).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"], dims=3)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))
        if weight_path:
            if resume: self.__load_resume_weights(weight_path, load_as_pretrained=False)
            if not resume: self.__load_resume_weights(weight_path, load_as_pretrained=True)
        self.logger.info(self.model)
    def __load_resume_weights(self, weight_path, load_as_pretrained):
        last_weight = os.path.join(weight_path)
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        self.logger.info('__load_resume_weights, last_weight= {}, load_as_pretrained:{}'.format(last_weight, load_as_pretrained))
        if not load_as_pretrained:
            if chkpt['epoch'] is not None:
                self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
        del chkpt

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP

        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}

        torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'backup_epoch%g.pt'%epoch))
        torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'lastest_epoch.pt'))
        if epoch==0 or (self.best_mAP == mAP and mAP>0):
            torch.save(chkpt, os.path.join(self.checkpoint_save_dir, "best.pt"))
        #torch.save(chkpt, os.path.join(self.checkpoint_save_dir, "last.pt"))
        del chkpt



    def train(self):
        writer = self.writer
        logger = self.logger
        logger.info("Training start,img size is: {},batchsize is: {:d},work number is {:d}".format(cfg.TRAIN["TRAIN_IMG_SIZE"],cfg.TRAIN["BATCH_SIZE"],cfg.TRAIN["NUMBER_WORKERS"]))
        logger.info("Train datasets number is : {}".format(len(self.train_dataset)))

        if self.fp_16: self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        logger.info("        =======  start  training   ======     ")
        #area_small, area_big, plt = self.evaluate()
        for epoch in range(self.start_epoch, self.epochs+1):
            start = time.time()
            self.model.train()
            self.current_epoch = epoch

            mloss = torch.zeros(5)
            logger.info("===Epoch:[{}/{}]===".format(epoch, self.epochs))
            n_batch = len(self.train_dataloader)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names)  in tqdm(enumerate(self.train_dataloader), total=n_batch):
                if (0): # if you had processed batch without dataloader, and use dataloader with B=1
                    imgs = imgs[0]
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                        label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                    img_names = [_[0] for _ in img_names]
                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.model(imgs)
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                if self.fp_16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # Accumulate gradient for x batches before optimizing
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update running mean of tracked metrics

                conf_data = p_d[0][..., 6:7].detach().cpu().numpy().flatten()
                pr999_p_conf = np.sort(conf_data)[-8]
                loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss, pr999_p_conf])
                mloss = (mloss * i + loss_items) / (i + 1)
                # len(self.train_dataloader) / (cfg.TRAIN["BATCH_SIZE"]) * epoch + iter
                # Print batch results
                if i % 10 == 0:

                    logger.info("  === Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}|train_pr99:{:.4f}".format(
                        epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,mloss[3], mloss[0], mloss[1],mloss[2],self.optimizer.param_groups[0]['lr'],
                        mloss[4]
                    ))
                    if writer:
                        writer.add_scalar('loss_ciou', mloss[0],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('loss_conf', mloss[1],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('loss_cls', mloss[2],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('train_loss', mloss[3],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('train_pr99.9_p_conf', mloss[4],
                                        len(self.train_dataloader) * epoch + i)
                        writer.add_scalar('train_lr', self.optimizer.param_groups[0]["lr"],
                                        len(self.train_dataloader) * epoch + i)
                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(range(10, 20)) * 32

            if epoch % self.eval_interval==0: #tag:Val #20
                if cfg.TRAIN["DATA_TYPE"] == 'VOC' or cfg.TRAIN["DATA_TYPE"] == 'ABUS':
                    area_small, area_big, plt, pr999_p_conf = self.evaluate()
                    logger.info("===== Validate =====".format(epoch, self.epochs))
                    if writer:
                        #writer.add_scalar('AUC_10mm', area_small, epoch)
                        #writer.add_scalar('AUC_15mm', area_big, epoch)
                        writer.add_scalar('AUC', area_big, epoch)
                        writer.add_scalar('EVAL_pr99.9_p_conf', pr999_p_conf, epoch)
                    save_per_epoch = 1
                    if epoch % save_per_epoch==0:
                        self.__save_model_weights(epoch, area_big)
                    logger.info('save weights done')
                    logger.info("  ===test AUC:{:.3f}".format(area_big))

            end = time.time()
            logger.info("  ===cost time:{:.4f}s".format(end - start))
        logger.info("=====Training Finished.   best_test_mAP:{:.3f}%====".format(self.best_mAP))

    def evaluate(self):
        logger = self.logger
        logger.info("Evaluate start,img size is: {},batchsize is: {:d},work number is {:d}".format(cfg.VAL["TEST_IMG_SIZE"], cfg.VAL["BATCH_SIZE"], cfg.VAL["NUMBER_WORKERS"]))
        logger.info("Test datasets number is : {}".format(len(self.test_dataloader)))

        if self.fp_16: self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        logger.info("        =======  start  evaluate   ======     ")
        start = time.time()
        self.model.eval()
        mloss = []
        pred_result_path=os.path.join(self.checkpoint_save_dir, 'evaluate')
        self.evaluator = Evaluator(self.model, showatt=False, pred_result_path=pred_result_path, box_top_k=cfg.VAL["BOX_TOP_K"])
        self.evaluator.clear_predict_file()
        TOP_K = 50
        with torch.no_grad():
            start_time=time.time()
            npy_dir = pred_result_path
            if 0: #for 96
                npy_format = npy_dir + '/{}'
                fold_list_root = '/home/lab402/User/eason_thesis/program_update_v1/5_fold_list/'
                # EASON code
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                fold_num = self.crx_fold_num
                val_path = fold_list_root + 'five_fold_val_'+str(fold_num)+'_separate.txt'
                #val_path = fold_list_root + 'five_fold_train_'+str(fold_num)+'_separate.txt'
                test_path = fold_list_root + 'five_fold_test_'+str(fold_num)+'.txt'
                val_set = open(val_path).readlines()
                #val_set = val_set[:100]
                for line in val_set:
                    line = line.split(',', 4)

                    #true_box = np.array([np.array(list(map(int, box.split(','))))
                    #                    for box in boxes])
                    img_vol = np.load(line[0])
                    img_vol = torch.from_numpy(img_vol)
                    ##img_vol = torch.transpose(img_vol, 0, 2).contiguous() # from xyz to zyx

                    img = img_vol.unsqueeze(dim=0).cuda().float() / 255.0
                    img_name = line[0].replace('/home/lab402/User/eason_thesis/ABUS_data/', '').replace('/','_')
                    img = img.to(self.device)
                    #for img, img_name in zip(imgs, img_names):
                    bboxes_prd, box_raw_data = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
                    pr999_p_conf = np.sort(box_raw_data[:, 6].detach().cpu().numpy().flatten())[-8]

                    mloss.append(pr999_p_conf)
                    if 0:
                        true_boxes = line[-1].split(' ')
                        true_boxes = list(map(lambda box: box.split(','), true_boxes))
                        true_boxes = [list(map(int, box)) for box in true_boxes]
                        box_data = [true_boxes]
                        boxes = [[{
                            'z_bot': box[0],
                            'z_top': box[3],
                            'z_range': box[3] - box[0] + 1,
                            'z_center': (box[0] + box[3]) / 2,
                            'y_bot': box[1],
                            'y_top': box[4],
                            'y_range': box[4] - box[1] + 1,
                            'y_center': (box[1] + box[4]) / 2,
                            'x_bot': box[2],
                            'x_top': box[5],
                            'x_range': box[5] - box[2] + 1,
                            'x_center': (box[2] + box[5]) / 2,
                        } for box in each_box_data if (box[3]*box[4]*box[5])>0] for each_box_data in box_data]
                        scale = [1,1,1]
                        ori_data = img
                        for i in range(int(boxes[0][0]['x_bot']), int(boxes[0][0]['x_top']), 1):
                            #TY Image
                            img = Image.fromarray(((ori_data.detach().squeeze().cpu().numpy() * 255.0).astype('uint8'))[:,:,i], 'L')
                            #img = Image.fromarray(TY_ori_data[i,:,:], 'L')
                            img = img.convert(mode='RGB')
                            draw = ImageDraw.Draw(img)
                            for bx in boxes[0]:
                                z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]
                                if int(x_bot) <= i <= int(x_top):
                                    #z_bot,y_bot = int(z_bot), int(y_bot)
                                    #z_top,y_top = int(z_top), int(y_top)

                                    draw.rectangle(
                                        [(y_bot, z_bot),(y_top, z_top)],
                                        outline ="red", width=2)
                            img.save('debug/infer_' + str(i)+'.png')
                    #if len(bboxes_prd) > 0:
                    #    bboxes_prd[:, :6] = (bboxes_prd[:, :6] / img.size(1)) * cfg.VAL['TEST_IMG_BBOX_ORIGINAL_SIZE'][0]
                    self.evaluator.store_bbox(img_name, bboxes_prd)
            log_txt = ""
            if 1: #for 640
                npy_format = npy_dir + '/{}_test.npy'
                n_batch = len(self.test_dataloader)
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names)  in tqdm(enumerate(self.test_dataloader), total=n_batch):
                    if (0):
                        imgs = imgs[0]
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                            label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                        img_names = [_[0] for _ in img_names]
                    imgs = imgs.to(self.device)
                    for img, img_name in zip(imgs, img_names):
                        bboxes_prd, box_raw_data, sub_log_txt = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
                        log_txt += sub_log_txt
                        pr999_p_conf = np.sort(box_raw_data[:, 6].detach().cpu().numpy().flatten())[-8]
                        mloss.append(pr999_p_conf)
                        if len(bboxes_prd) > 0:
                            bboxes_prd[:, :6] = (bboxes_prd[:, :6] / img.size(1)) * cfg.VAL['TEST_IMG_BBOX_ORIGINAL_SIZE'][0]
                        self.evaluator.store_bbox(img_name+"_test", bboxes_prd)
            
            txt = "Average time cost: {:.2f} sec.".format((time.time() - start_time)/len(self.test_dataloader))
            print(txt)
            annotation_file = "annotation_all.txt"
            area_small, area_big, plt, sub_log_txt = calculate_FROC(annotation_file, npy_dir, npy_format, size_threshold=20, th_step=0.01)
            log_txt += txt + "\n" + sub_log_txt
            plt.savefig(os.path.join(self.checkpoint_save_dir, 'froc_test.png'))
            if hasattr(self, "current_epoch"): # from train3D.py
                out_log_name = os.path.join(self.checkpoint_save_dir, 'evaluate_log_e{}.txt'.format(self.current_epoch))
            else: # from draw_froc.py
                out_log_name = os.path.join(self.checkpoint_save_dir, 'evaluate_log.txt')
            with open(out_log_name, "w") as f:
                f.write(log_txt)
            #print("SAVE IMG TO", os.path.join(self.checkpoint_save_dir, 'froc_test.png'))

        end = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end - start))
        return area_small, area_big, plt, np.percentile(mloss, 50)

    def evaluate_and_logTB(self):
        writer = self.writer
        logger = self.logger
