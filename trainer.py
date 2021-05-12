import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
#from model.loss.yolo_loss_brucetu import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from eval.froc import calculate_FROC, calculate_FROC_randomcrop
from utils.tools import *
from torch.utils.tensorboard import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger
import shutil
import os
from os.path import join as pjoin
import random_crop
from utils_hsz import AnimationViewer
from adabelief_pytorch import AdaBelief
#from eval_coco import *
#from eval.cocoapi_evaluator import COCOAPIEvaluator

#from databuilder.abus import ABUSDetectionDataset
from dataset import Tumor, LungDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm
from apex import amp
from global_variable import USE_LUNA, MASK_SAVED_PATH, NEGATIVE_NPY_SAVED_PATH, ITERATIVE_FP_CROP_PATH

class Trainer(object):
    def __init__(self, testing_mode, weight_path, checkpoint_save_dir, resume, gpu_id, accumulate, fp_16, writer, logger, crx_fold_num,
                dataset_name, eval_interval, npy_name, det_tp_iou_thresh=None, eval_conf_thresh=None):
        #self.data_root = 'datasets/abus'
        #init_seeds(0)
        self.lung_dataset_name = dataset_name
        self.device = gpu.select_device(gpu_id)
        #self.device = torch.device("cpu")
        self.start_epoch = 0
        self.best_cpm = 0.
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

        self.train_random_crop = cfg.TRAIN["USING_RANDOM_CROP_TRAIN"]
        self.eval_random_crop = cfg.VAL["USING_RANDOM_CROP_EVAL"]
        self.random_crop_file_prefix = cfg.TRAIN["RANDOM_CROP_FILE_PREFIX"]
        self.random_crop_ncopy = cfg.TRAIN["RANDOM_CROP_NCOPY"]
        self.batch_1_eval = cfg.VAL["BATCH_1_EVAL"]
        self.may_change_optimizer = False
        self.test_lung_voi = cfg.VAL["TEST_LUNG_VOI"]
        self.test_lung_voi_ignore_invalid_voi = cfg.VAL["TEST_LUNG_VOI_IGNORE_INVALID_VOI"]
        self.det_tp_iou_thresh = cfg.VAL["TP_IOU_THRESH"] if det_tp_iou_thresh==None else det_tp_iou_thresh
        self.eval_conf_thresh = cfg.VAL["CONF_THRESH"] if eval_conf_thresh==None else eval_conf_thresh

        self.use_5mm = cfg.TRAIN["USE_5MM"]
        self.use_2d5mm = cfg.TRAIN["USE_2.5MM"]
        assert not (self.use_5mm and self.use_2d5mm), "Either 5/2.5mm is ok, but not both at the same time"

        self.do_fp_reduction = cfg.TRAIN["DO_FP_REDUCTION"]
        #self.fp_reduction_target = cfg.TRAIN["FP_REDUCTION_TARGET_DATASET"] 
        self.fp_reduction_start_epoch = cfg.TRAIN["FP_REDUCTION_START_EPOCH"]
        self.fp_reduction_interval = cfg.TRAIN["FP_REDUCTION_INTERVAL"]
        if self.do_fp_reduction:
            #assert not self.eval_random_crop, "FP_REDUCTION CAN ONLY BE DONE WHEN USING THE WHOLE IMAGE AS EVAL DATA!"
            #assert self.testing_mode != 1 ## only debug/train/validation allowed
            #assert self.fp_reduction_target.lower() in ["training", "testing"]
            self.empty_dataset = LungDataset.load(dataset_name) # empty means "didn't get_data"
            assert self.fp_reduction_interval > 0
        
        

        
        #train_dataset = ABUSDetectionDataset(testing_mode, augmentation=True, crx_fold_num= crx_fold_num, crx_partition= 'train', crx_valid=True, include_fp=False, root=self.data_root,
        #    batch_size=cfg.TRAIN["BATCH_SIZE"])
        dataset = LungDataset.load(dataset_name)
        dataset.get_data(dataset.pids, name=npy_name)
        train_data, validation_data, test_data = dataset.make_kfolds_using_pids(num_k_folds=5, k_folds_seed=123, current_fold=self.crx_fold_num, valid_test_split=True, portion_list=[3,1,1])
        train_dataset = LungDataset.load(dataset_name)
        train_dataset.data = train_data
        if self.train_random_crop:
            train_dataset.set_random_crop(self.random_crop_file_prefix, self.random_crop_ncopy, True)
        
        #validation_dataset =  LungDataset.load(dataset_name)
        #validation_dataset.data = validation_data
        test_dataset =  LungDataset.load(dataset_name)
        if self.testing_mode==0: # validation
            test_dataset.data = validation_data
        elif self.testing_mode==1: # testing
            test_dataset.data = test_data
        elif self.testing_mode == -1: # train_debug
            test_dataset.data = train_data[:50]
        elif self.testing_mode == -2: # whole train_debug
            test_dataset.data = train_data
        else:
            raise TypeError(f"Invalid testing mode: {self.testing_mode}. {{1: 'test', 0: 'val', -1: 'train_debug', -2: 'whole_train_debug'}}")
        if self.eval_random_crop:
            #test_dataset.set_random_crop(self.random_crop_file_prefix, 1) # no reason to use ncopy > 1 during eval
            test_dataset.set_random_crop(self.random_crop_file_prefix, 1, False) # no reason to use ncopy > 1 during eval
   
        #useless, cache should be implemented on yolo dataset
        train_dataset.cacher.cache_size = 0
        test_dataset.cacher.cache_size = 0

        if (0): #debug
            train_dataset.data = train_data[:40]
            test_dataset.data = train_data[:40]
        if (0): #debug2
            test_dataset.data = test_dataset.data[:3]  # 3 datum only
        if (0): #debug3
            train_dataset.data = train_data[:cfg.TRAIN["BATCH_SIZE"] ]
        if (0): #debug4
            train_dataset.data = train_data[:8]
            test_dataset.data = train_data[:3]

        if self.batch_1_eval: # adjust test_dataset.data to always return original image
            for i in range(len(test_dataset.data)): 
                _, bboxs_ori, pid = test_dataset.data[i]
                datum = (None, bboxs_ori, pid) # force load original img
                test_dataset.data[i] = datum
            test_dataset.set_batch_1_eval(True, cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"])
        
        if self.test_lung_voi:
            assert self.batch_1_eval
            test_dataset.set_lung_voi(True)
            if self.test_lung_voi_ignore_invalid_voi:
                err_file = pjoin(MASK_SAVED_PATH, "error_pid.txt")
                with open(err_file, "r") as f:
                    err_pids = f.read()[1:-1].split(",\n")
                self.err_pids = err_pids
                trimmed_data = []
                for i in range(len(test_dataset.data)): 
                    datum = test_dataset.data[i]
                    npy, bboxs_ori, pid = datum
                    if pid not in err_pids:
                        trimmed_data.append(datum)
                test_dataset.data = trimmed_data

        if self.use_5mm:
            if cfg.VAL["FAST_EVAL_PKL_NAME"] not in (False, None):
                pkl_name = cfg.VAL["FAST_EVAL_PKL_NAME"]
                assert type(pkl_name) == str, "Bad param for FAST_EVAL_PKL_NAME in cfg: '{}'".format(pkl_name)
            else:
                pkl_name = False
            train_dataset.set_5mm(True)
            test_dataset.set_5mm(True, pkl_name)
        elif self.use_2d5mm:
            if cfg.VAL["FAST_EVAL_PKL_NAME"] not in (False, None):
                pkl_name = cfg.VAL["FAST_EVAL_PKL_NAME"]
                assert type(pkl_name) == str, "Bad param for FAST_EVAL_PKL_NAME in cfg: '{}'".format(pkl_name)
            else:
                pkl_name = False
            train_dataset.set_2d5mm(True)
            test_dataset.set_2d5mm(True, pkl_name)
            
        
        if self.do_fp_reduction: # (pre-cropped version)
            assert self.train_random_crop
            train_fp_dataset = LungDataset.load(dataset_name)
            train_fp_dataset.data = train_data
            train_fp_dataset.set_random_crop(cfg.TRAIN["FP_REDUCTION_CROP_PREFIX"], cfg.TRAIN["FP_REDUCTION_CROP_NCOPY"], True, NEGATIVE_NPY_SAVED_PATH, cfg.TRAIN["FP_REDUCTION_MODE"])
            self.train_fp_dataset = YOLO4_3DDataset(train_fp_dataset, classes=[0, 1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"], cache_size=0, batch_1_eval=False, use_zero_conf=cfg.TRAIN["FP_REDUCTION_USE_ZERO_CONF"])
            self.train_fp_data_loader = DataLoader(self.train_fp_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"]//2,
                                           shuffle=True, pin_memory=False
                                           )
            if (0): #debug
                for img, bbox, pid in train_fp_dataset:
                    pass
                    #print("fp pid", pid)
                    #view_img = img.squeeze(-1).numpy()
                    #print("fp bbox:", bbox)
                    #AnimationViewer(view_img, bbox[:,:6], note=str(pid), draw_face=False)


        self.train_dataset = YOLO4_3DDataset(train_dataset, classes=[0, 1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"], cache_size=0, batch_1_eval=False)
        #self.train_dataset = data.Build_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        self.epochs = cfg.TRAIN["YOLO_EPOCHS"] if cfg.MODEL_TYPE["TYPE"] == 'YOLOv4' else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        self.early_stopping_epoch = cfg.TRAIN["EARLY_STOPPING_EPOCH"]
        assert (type(self.early_stopping_epoch)==int and self.early_stopping_epoch>0) or (self.early_stopping_epoch==None), "Bad stop epoch: {}".format(self.early_stopping_epoch)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True, pin_memory=False
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

        self.test_dataset = YOLO4_3DDataset(test_dataset, classes=[0, 1], img_size=cfg.VAL["TEST_IMG_SIZE"], cache_size=0, batch_1_eval=cfg.VAL["BATCH_1_EVAL"])
        self.test_dataloader = DataLoader(self.test_dataset,
                                            batch_size=cfg.VAL["BATCH_SIZE"],
                                            num_workers=cfg.VAL["NUMBER_WORKERS"],
                                            shuffle=False, pin_memory=False
                                            )
        #sum([p.flatten().size(0) for p in self.model.parameters()])
        self.model = Build_Model(weight_path=weight_path, resume=resume, dims=3).to(self.device)

        self.optimizer_type = cfg.TRAIN["OPTIMIZER"]
        assert self.optimizer_type in ("ADAM", "SGD", "ADABELIEF")
        if cfg.TRAIN["USE_SGD_BEFORE_LOSS_LOWER_THAN_THRESH"]:
            self.may_change_optimizer = True

        if self.optimizer_type == 'SGD' or cfg.TRAIN["USE_SGD_BEFORE_LOSS_LOWER_THAN_THRESH"]:
            self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                    momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        elif self.optimizer_type == "ADAM":
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        elif self.optimizer_type == "ADABELIEF":
            self.optimizer = AdaBelief(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"], eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        else:
            raise TypeError("Unrecognized optimizer:", self.optimizer_type)

        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS3D"], strides=cfg.MODEL["STRIDES"],
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
                if "best_cpm" in chkpt:
                    self.best_cpm = chkpt['best_cpm']
                else:
                    assert "best_mAP" in chkpt ## previous version code compatibility (it is cpm, but is named mAP wrongly)
                    self.best_cpm = chkpt['best_mAP']
        del chkpt

    def __save_model_weights(self, epoch, cpm):
        if cpm > self.best_cpm:
            self.best_cpm = cpm

        chkpt = {'epoch': epoch,
                 'best_cpm': self.best_cpm,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}

        torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'backup_epoch%g.pt'%epoch))
        torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'lastest_epoch.pt'))
        if epoch==0 or (self.best_cpm == cpm and cpm>0):
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
        shutil.copyfile("config/yolov4_config.py", os.path.join(self.checkpoint_save_dir, "training_config.txt"))
        #area_small, area_big, plt = self.evaluate()
        self.optimizer.zero_grad() ## ccy
        for epoch in range(self.start_epoch, self.epochs+1):
            start = time.time()
            self.model.train()
            self.current_epoch = epoch

            mloss = torch.zeros(5)
            logger.info("===Epoch:[{}/{}]===".format(epoch, self.epochs))
            n_batch = len(self.train_dataloader)

            if self.do_fp_reduction:
                do_fp_reduction_this_epoch = (self.do_fp_reduction) and (epoch>=self.fp_reduction_start_epoch) and (epoch%self.fp_reduction_interval==0)
                fp_iterator = iter(self.train_fp_data_loader)

            for i, batched_data in tqdm(enumerate(self.train_dataloader), total=n_batch):
                imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names, shapes_before_pad, _ = batched_data
                
                if (0): # if you had processed batch without dataloader, and use dataloader with B=1
                    imgs = imgs[0]
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                        label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                    img_names = [_[0] for _ in img_names]
                

                # fp reduction (batch_version, slow and buggy)
                #if do_fp_reduction_this_epoch:
                #    fp_data = self.get_fp_for_reduction_batch(img_names)
                #    cat = lambda t1,t2: torch.cat([t1,t2], dim=0) if type(t1)==torch.Tensor==type(t2) else t1+t2
                #    merged_data = [cat(t1,t2) for t1,t2 in zip(batched_data, fp_data)]
                #    imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names, shapes_before_pad, _ = merged_data
                #    self.model.train() # turn from evaluating back to training

                # fp reduction (load npy version)
                if self.do_fp_reduction and do_fp_reduction_this_epoch:
                    fp_data = next(fp_iterator)
                    cat = lambda t1,t2: torch.cat([t1,t2], dim=0) if type(t1)==torch.Tensor==type(t2) else t1+t2
                    merged_data = [cat(t1,t2) for t1,t2 in zip(batched_data, fp_data)]
                    imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names, shapes_before_pad, _ = merged_data


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
                loss = loss / self.accumulate  ## ccy
                if self.fp_16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # Accumulate gradient for x batches before optimizing
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if cfg.MODEL["VERBOSE_SHAPE"]:
                    raise EOFError("One batch end")
                # Update running mean of tracked metrics



                conf_data = p_d[0][..., 6:7].detach().cpu().numpy().flatten()
                pr999_p_conf = np.sort(conf_data)[-8]
                loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss, pr999_p_conf])
                mloss = (mloss * i + loss_items) / (i + 1)

                if self.may_change_optimizer and cfg.TRAIN["USE_SGD_BEFORE_LOSS_LOWER_THAN_THRESH"]:
                    if loss_conf < cfg.TRAIN["CHANGE_OPTIMIZER_THRESH"]:
                        print("Change optimizer to {}".format(self.optimizer_type))
                        self.may_change_optimizer = False
                        current_lr = self.scheduler.get_last_lr()
                        if self.optimizer_type=="ADAM":
                            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"]) 
                        elif self.optimizer_type=="ADABELIEF":
                            self.optimizer = AdaBelief(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"], eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

                        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=current_lr,
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))
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

            if (epoch % self.eval_interval==0) or (epoch == self.epochs) or (epoch == self.early_stopping_epoch):# or (do_fp_reduction_this_epoch): #tag:Val #20
                if cfg.TRAIN["DATA_TYPE"] == 'VOC' or cfg.TRAIN["DATA_TYPE"] == 'ABUS' or cfg.TRAIN["DATA_TYPE"] == 'LUNG':
                    area_dist, area_iou, plt, pr999_p_conf, cpm_dist, cpm, max_sens_dist, max_sens_iou, bboxes_pred = self.evaluate(return_box=True)
                    ##TODO: add fp reduction by throwing fp back!
                    logger.info("===== Validate =====".format(epoch, self.epochs))
                    if writer:
                        writer.add_scalar('AUC (IOU)', area_iou, epoch)
                        writer.add_scalar('EVAL_pr99.9_p_conf', pr999_p_conf, epoch)
                        writer.add_scalar('CPM (IOU)', cpm, epoch)
                        writer.add_scalar('AUC (dist)', area_dist, epoch)
                        writer.add_scalar('CPM (dist)', cpm_dist, epoch)
                        writer.add_scalar('Max sens(iou)', max_sens_iou, epoch)
                        writer.add_scalar('Max sens(dist)', max_sens_dist, epoch)
                    save_per_epoch = 1
                    if epoch % save_per_epoch==0:
                        self.__save_model_weights(epoch, cpm_dist)
                    logger.info('save weights done')
                    logger.info("  ===test CPM:{:.3f}".format(cpm_dist))

            end = time.time()
            logger.info("  ===cost time:{:.4f}s".format(end - start))
            if epoch == self.early_stopping_epoch:
                break
        logger.info("=====Training Finished.   best_test_CPM:{:.3f}%====".format(self.best_cpm))

    def evaluate(self, return_box=False):
        logger = self.logger
        logger.info("Evaluate start,img size is: {},batchsize is: {:d},work number is {:d}".format(cfg.VAL["TEST_IMG_SIZE"], cfg.VAL["BATCH_SIZE"], cfg.VAL["NUMBER_WORKERS"]))
        logger.info("Test datasets number is : {}".format(len(self.test_dataloader)))

        if self.fp_16: self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        logger.info("        =======  start  evaluate   ======     ")
        start = time.time()
        self.model.eval()
        mloss = []
        pred_result_path=os.path.join(self.checkpoint_save_dir, 'evaluate')
        self.evaluator = Evaluator(self.model, showatt=False, pred_result_path=pred_result_path, box_top_k=cfg.VAL["BOX_TOP_K"], conf_thresh=self.eval_conf_thresh)
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
            log_txt += f"Using eval_conf_thresh: {self.eval_conf_thresh}\n"
            if 1: #for 640
                npy_format = npy_dir + '/{}_test.npy'
                n_batch = len(self.test_dataloader)
                print("eval n_batch in test_dataloader:", n_batch)
                gt_lut = {}
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names, shapes_before_pad, valid_bboxes)  in tqdm(enumerate(self.test_dataloader), total=n_batch):
                    if (0):
                        imgs = imgs[0]
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                            label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                        img_names = [_[0] for _ in img_names]

                    print("eval imgs input shape:", imgs.shape)
                    if (0) and self.batch_1_eval:
                        vimg = imgs.squeeze(0).squeeze(0).numpy()
                        AnimationViewer(vimg, note="eval:{}".format(img_names[0]))
                    imgs = imgs.to(self.device)
                    for img, img_name, shape_before_pad, valid_bbox in zip(imgs, img_names, shapes_before_pad, valid_bboxes):
                        #print("(Eval) Current img:", img_name)
                        bboxes_prd, box_raw_data, sub_log_txt = self.evaluator.get_bbox(img, multi_test=False, flip_test=False, shape_before_pad=shape_before_pad)
                        #print("bboxes_prd:", bboxes_prd)
                        #print("type of bboxes_prd", type(bboxes_prd))
                        #print("bboxes_prd max", bboxes_prd.max(axis=0))
                        #print("box_raw_data:", box_raw_data)
                        log_txt += sub_log_txt
                        pr999_p_conf = np.sort(box_raw_data[:, 6].detach().cpu().numpy().flatten())[-8]
                        mloss.append(pr999_p_conf)
                        if len(bboxes_prd) > 0 and (not self.batch_1_eval):
                            bboxes_prd[:, :6] = (bboxes_prd[:, :6] / img.size(1)) * cfg.VAL['TEST_IMG_BBOX_ORIGINAL_SIZE'][0]
                        self.evaluator.store_bbox(img_name+"_test", bboxes_prd)
                        valid_bbox = valid_bbox.cpu().tolist()
                        gt_lut[img_name] = valid_bbox
                    
            
            txt = "Average time cost: {:.2f} sec.".format((time.time() - start_time)/len(self.test_dataloader))
            print(txt)
            annotation_file = "annotation_luna.txt" if USE_LUNA else "annotation_chung.txt"
            if self.eval_random_crop:
                area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou = calculate_FROC_randomcrop(annotation_file, npy_dir, npy_format, size_threshold=20, th_step=0.01, ori_dataset=self.test_dataset.ori_dataset, det_tp_iou_thresh=self.det_tp_iou_thresh)
            else:
                #if self.do_fp_reduction:
                #    area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou, fp_bboxes_all_pid = calculate_FROC(gt_lut, npy_dir, npy_format, size_threshold=20, th_step=0.01, det_tp_iou_thresh=self.det_tp_iou_thresh, return_fp_bboxes=True)
                #else:
                area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou, _ = calculate_FROC(gt_lut, npy_dir, npy_format, size_threshold=20, th_step=0.01, det_tp_iou_thresh=self.det_tp_iou_thresh, return_fp_bboxes=False)
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
        if return_box:
            return area_dist, area_iou, plt, np.percentile(mloss, 50), cpm_dist, cpm, max_sens_dist, max_sens_iou, bboxes_prd
        else:
            return area_dist, area_iou, plt, np.percentile(mloss, 50), cpm_dist, cpm, max_sens_dist, max_sens_iou

    def evaluate_and_logTB(self):
        writer = self.writer
        logger = self.logger

    def iterative_update_fp_crops(self):
        assert self.train_random_crop
        new_fp_dataset = LungDataset.load(self.lung_dataset_name)
        pids = [pid for _,_,pid in self.train_dataset.ori_dataset.data]
        new_fp_dataset.get_data(pids)
        new_fp_dataset.set_batch_1_eval(True, cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"])
        new_fp_dataset.set_lung_voi()
        if self.use_5mm or self.use_2d5mm:
            if cfg.VAL["FAST_EVAL_PKL_NAME"] not in (False, None):
                pkl_name = cfg.VAL["FAST_EVAL_PKL_NAME"]
                assert type(pkl_name) == str, "Bad param for FAST_EVAL_PKL_NAME in cfg: '{}'".format(pkl_name)
            else:
                pkl_name = False
            if self.use_5mm:
                new_fp_dataset.set_5mm(True, pkl_name)
            else: #2.5mm
                new_fp_dataset.set_2d5mm(True, pkl_name)
        new_fp_dataset = YOLO4_3DDataset(new_fp_dataset, classes=[0, 1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"], cache_size=0, batch_1_eval=False, use_zero_conf=cfg.TRAIN["FP_REDUCTION_USE_ZERO_CONF"])
        new_fp_data_loader = DataLoader(new_fp_dataset, batch_size=1, num_workers=cfg.VAL["NUMBER_WORKERS"], shuffle=False, pin_memory=False)
        npy_dir = ITERATIVE_FP_CROP_PATH
        new_fp_evaluator = Evaluator(self.model, showatt=False, pred_result_path=npy_dir, box_top_k=cfg.VAL["BOX_TOP_K"], conf_thresh=self.eval_conf_thresh)
        #new_fp_evaluator.clear_predict_file() #not remove crop yet
        #TODO: eval new crops
        

        with torch.no_grad():
            start_time=time.time()
            log_txt = ""
            log_txt += f"Using eval_conf_thresh: {self.eval_conf_thresh}\n"
            if 1: #for 640
                npy_format = npy_dir + '/{}_test.npy'
                n_batch = len(self.test_dataloader)
                print("eval n_batch in new_fp_dataloader:", n_batch)
                gt_lut = {}
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names, shapes_before_pad, valid_bboxes)  in tqdm(enumerate(self.test_dataloader), total=n_batch):
                    if (0):
                        imgs = imgs[0]
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                            label_sbbox[0], label_mbbox[0], label_lbbox[0], sbboxes[0], mbboxes[0], lbboxes[0]
                        img_names = [_[0] for _ in img_names]

                    print("eval imgs input shape:", imgs.shape)
                    if (0) and self.batch_1_eval:
                        vimg = imgs.squeeze(0).squeeze(0).numpy()
                        AnimationViewer(vimg, note="eval:{}".format(img_names[0]))
                    imgs = imgs.to(self.device)
                    for img, img_name, shape_before_pad, valid_bbox in zip(imgs, img_names, shapes_before_pad, valid_bboxes):
                        #print("(Eval) Current img:", img_name)
                        bboxes_prd, box_raw_data, sub_log_txt = self.evaluator.get_bbox(img, multi_test=False, flip_test=False, shape_before_pad=shape_before_pad)
                        #print("bboxes_prd:", bboxes_prd)
                        #print("type of bboxes_prd", type(bboxes_prd))
                        #print("bboxes_prd max", bboxes_prd.max(axis=0))
                        #print("box_raw_data:", box_raw_data)
                        log_txt += sub_log_txt
                        pr999_p_conf = np.sort(box_raw_data[:, 6].detach().cpu().numpy().flatten())[-8]
                        mloss.append(pr999_p_conf)
                        if len(bboxes_prd) > 0 and (not self.batch_1_eval):
                            bboxes_prd[:, :6] = (bboxes_prd[:, :6] / img.size(1)) * cfg.VAL['TEST_IMG_BBOX_ORIGINAL_SIZE'][0]
                        #self.evaluator.store_bbox(img_name+"_test", bboxes_prd)
                        valid_bbox = valid_bbox.cpu().tolist()
                        gt_lut[img_name] = valid_bbox
                    
            
            txt = "Average time cost: {:.2f} sec.".format((time.time() - start_time)/len(self.test_dataloader))
            print(txt)
            annotation_file = "annotation_luna.txt" if USE_LUNA else "annotation_chung.txt"
            if self.eval_random_crop:
                area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou = calculate_FROC_randomcrop(annotation_file, npy_dir, npy_format, size_threshold=20, th_step=0.01, ori_dataset=self.test_dataset.ori_dataset, det_tp_iou_thresh=self.det_tp_iou_thresh)
            else:
                #if self.do_fp_reduction:
                #    area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou, fp_bboxes_all_pid = calculate_FROC(gt_lut, npy_dir, npy_format, size_threshold=20, th_step=0.01, det_tp_iou_thresh=self.det_tp_iou_thresh, return_fp_bboxes=True)
                #else:
                area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou, _ = calculate_FROC(gt_lut, npy_dir, npy_format, size_threshold=20, th_step=0.01, det_tp_iou_thresh=self.det_tp_iou_thresh, return_fp_bboxes=False)
            log_txt += txt + "\n" + sub_log_txt
            plt.savefig(os.path.join(self.checkpoint_save_dir, 'froc_test.png'))
            if hasattr(self, "current_epoch"): # from train3D.py
                out_log_name = os.path.join(self.checkpoint_save_dir, 'evaluate_log_e{}.txt'.format(self.current_epoch))
            else: # from draw_froc.py
                out_log_name = os.path.join(self.checkpoint_save_dir, 'evaluate_log.txt')
            with open(out_log_name, "w") as f:
                f.write(log_txt)
            #print("SAVE IMG TO", os.path.join(self.checkpoint_save_dir, 'froc_test.png'))
        

    def get_fp_for_reduction_batch(self, img_names, return_crop_only=False, topk=None, correct_fp_usage=False): # batch_version
        """
        fp_bboxes_all_pid: a dictionary, key=pid, value=fp_bboxes
        fp_bboxes: shape (X, 8), where 8 = pred_coor(6) + pred_conf(1) + pred_class_idx(1) [X differs due to postprocessing]
        @Argument
            topk: None or int; if assigned, crop all topk FP
        """
        ### IDEAS
        ## Using eval crop (fp) into training?? (WIP)
        ## Using random cropped blank area??
        ## Using other idea?? (GAN?)

        npy_dir = "fp_reduction_tmp"
        fp_dataset = self.empty_dataset
        if self.test_lung_voi:
            fp_dataset.set_lung_voi(True)
            if self.test_lung_voi_ignore_invalid_voi:
                trimmed_names = []
                for pid in img_names:
                    if pid not in self.err_pids:
                        trimmed_names.append(pid)
                fp_dataset.get_data(trimmed_names)
            else:
                fp_dataset.get_data(img_names)
        
        if self.use_5mm:
            if cfg.VAL["FAST_EVAL_PKL_NAME"] not in (False, None):
                pkl_name = cfg.VAL["FAST_EVAL_PKL_NAME"]
                assert type(pkl_name) == str, "Bad param for FAST_EVAL_PKL_NAME in cfg: '{}'".format(pkl_name)
            else:
                pkl_name = False
            fp_dataset.set_5mm(True, pkl_name)
        elif self.use_2d5mm:
            if cfg.VAL["FAST_EVAL_PKL_NAME"] not in (False, None):
                pkl_name = cfg.VAL["FAST_EVAL_PKL_NAME"]
                assert type(pkl_name) == str, "Bad param for FAST_EVAL_PKL_NAME in cfg: '{}'".format(pkl_name)
            else:
                pkl_name = False
            fp_dataset.set_2d5mm(True, pkl_name)
        
        if correct_fp_usage:
            assert return_crop_only # workflow need this
        
        if (0): #debug
            for i, (_,_,pid) in enumerate(fp_dataset.data):
                print(pid)
                if i==4:
                    break
            print("img_names:", img_names[:5]) # img_names look random
            raise EOFError("Fp dataset's data") 

        fp_dataset.set_batch_1_eval(True, cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"])
        fp_dataset = YOLO4_3DDataset(fp_dataset, classes=[0,1], batch_1_eval=True)
        fp_dataloader = DataLoader(fp_dataset, batch_size=1, num_workers=cfg.VAL["NUMBER_WORKERS"], pin_memory=False, shuffle=False)
        fp_evaluator = Evaluator(self.model, showatt=False, pred_result_path=npy_dir, box_top_k=cfg.VAL["BOX_TOP_K"], conf_thresh=self.eval_conf_thresh)
        fp_evaluator.clear_predict_file()

        log_txt = "Enter FP reduction!!\n"
        # (1) run evaluation on 1 batch of samples to generate fp_bboxes
        with torch.no_grad():
            start_time=time.time()
            if 1: 
                npy_format = npy_dir + '/{}_test.npy'
                n_batch = len(fp_dataloader)
                gt_lut = {}
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names, shapes_before_pad, valid_bboxes)  in tqdm(enumerate(fp_dataloader), total=n_batch, desc="fp getting"):
                    imgs = imgs.to(self.device)
                    for img, img_name, shape_before_pad, valid_bbox in zip(imgs, img_names, shapes_before_pad, valid_bboxes):
                        if (0): #debug
                            print("fp(1) pid:", img_name)
                            continue
                        #print("(FP reduction) Current img:", img_name)
                        bboxes_prd, box_raw_data, sub_log_txt = fp_evaluator.get_bbox(img, multi_test=False, flip_test=False, shape_before_pad=shape_before_pad)
                        log_txt += sub_log_txt
                        fp_evaluator.store_bbox(img_name+"_test", bboxes_prd)
                        valid_bbox = valid_bbox.cpu().tolist()
                        gt_lut[img_name] = valid_bbox
                #raise EOFError("End of fp(1)")
            
            #txt = "Average time cost: {:.2f} sec.".format((time.time() - start_time)/len(self.test_dataloader))
            #print(txt)
            #annotation_file = "annotation_luna.txt" if USE_LUNA else "annotation_chung.txt"
            area_dist, area_iou, plt, sub_log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou, fp_bboxes_all_pid = calculate_FROC(gt_lut, npy_dir, npy_format, size_threshold=20, th_step=0.01, det_tp_iou_thresh=self.det_tp_iou_thresh, return_fp_bboxes=True)
            plt.close()
            log_txt += sub_log_txt
            #plt.savefig(os.path.join(self.checkpoint_save_dir, 'froc_test.png'))
            #if hasattr(self, "current_epoch"): # from train3D.py
            #    out_log_name = os.path.join(self.checkpoint_save_dir, 'evaluate_log_e{}.txt'.format(self.current_epoch))
            #else: # from draw_froc.py
            #    out_log_name = os.path.join(self.checkpoint_save_dir, 'evaluate_log.txt')
            #with open(out_log_name, "w") as f:
            #    f.write(log_txt)
            #print("SAVE IMG TO", os.path.join(self.checkpoint_save_dir, 'froc_test.png'))
            end = time.time()
            #logger.info("  ===cost time:{:.4f}s".format(end - start))


        # (2) crop and make label of those fp bboxes
        n_batch = len(fp_dataloader)
        cropped_imgs = []
        cropped_boxes = []
        cropped_names = []
        uncropped_boxes = []
        for i, (imgs, _, _, _, _, _, _, img_names, shapes_before_pad, valid_bboxes)  in tqdm(enumerate(fp_dataloader), total=n_batch, desc="fp cropping"):
            for img, img_name, shape_before_pad, valid_bbox in zip(imgs, img_names, shapes_before_pad, valid_bboxes):
                if (0): #debug
                    print("fp(2) pid:", img_name)
                    continue

                if topk==None:
                    k=5 # top k fp to **choose** in "random_crop_3D"
                else:
                    k=topk
                img = img.squeeze_(-1).to(self.device)
                fp_bboxes = fp_bboxes_all_pid[img_name] # shape (?, 8)
                top_k_hard_fp_idx = fp_bboxes[:, 6].argsort()[-k:][::-1]
                top_k_hard_fp = fp_bboxes[top_k_hard_fp_idx] # shape (k, 8)
                to_crop_fp = torch.tensor(top_k_hard_fp[:,:6], device=self.device, dtype=torch.float32)
                if len(to_crop_fp)==0: #no fp
                    continue
                if topk==None:
                    to_crop_fp = random.choice(to_crop_fp).unsqueeze_(0)

                for j in range(len(to_crop_fp)):
                    try:
                        cropped_img, cropped_box = random_crop.random_crop_3D(img, to_crop_fp[j].unsqueeze(0), cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["TRAIN_IMG_SIZE"])
                    except Exception as e:
                        print("Encounter error in random crop: {}".format(e.__repr__()))
                        continue         
                    cropped_img, cropped_box = cropped_img[0], cropped_box[0] # only remove list, shape (C, new_d, new_h, new_w)
                    if (0): # random crop debug view
                        view_img = cropped_img.squeeze(0).cpu().numpy()
                        view_box = cropped_box
                        AnimationViewer(view_img, view_box, "fp crop")
                    cropped_img = cropped_img.squeeze_(0).unsqueeze_(-1) # (1,Z,Y,X) -> (Z,Y,X,1)
                    n_box = cropped_box.shape[0]
                    ## mode
                    fp_mode = cfg.TRAIN["FP_REDUCTION_MODE"] # 0,0 | 0,1 | 1,0 | 1,1 (cls_index, mix)
                    assert fp_mode in ("0,1", "1,1")
                    conf_cls_label = torch.ones((n_box,2), device=self.device) #  cls == 1 and mix == 1
                    if fp_mode == "0,1": # cls_idx == 0
                        #conf_cls_label[:,1] = 0 #ori
                        conf_cls_label[:,0] = 0

                    cropped_box = torch.cat([cropped_box, conf_cls_label], dim=1)
                    cropped_imgs.append(cropped_img.cpu())
                    cropped_boxes.append(cropped_box.cpu().numpy())
                    cropped_names.append(img_name)
                    uncropped_boxes.append(to_crop_fp[j])
        
        if return_crop_only: # return numpy fp crop and then exit (not for training)
            out_imgs = [crop.squeeze_(-1).numpy() for crop in cropped_imgs]
            out_boxes = [boxes[:,:-2].tolist() for boxes in cropped_boxes]
            out_names = cropped_names
            if correct_fp_usage:
                return out_imgs, out_boxes, out_names, uncropped_boxes
            return out_imgs, out_boxes, out_names


        # the tmp dataset is used for "__creat_label" only
        tmp_dataset = FastImageDataset(cropped_imgs, cropped_boxes, cropped_names)
        tmp_dataset = YOLO4_3DDataset(tmp_dataset, classes=[0,1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"], batch_1_eval=False)
        tmp_dataloader = DataLoader(tmp_dataset, batch_size=1000, #big is okay
                                        num_workers=0,
                                        shuffle=True, pin_memory=False
                                        )
        assert len(tmp_dataloader) == 1
        for i, (fp_imgs, fp_label_sbbox, fp_label_mbbox, fp_label_lbbox, fp_sbboxes, fp_mbboxes, fp_lbboxes, img_names, shapes_before_pad, _)  in enumerate(tmp_dataloader):
            return fp_imgs, fp_label_sbbox, fp_label_mbbox, fp_label_lbbox, fp_sbboxes, fp_mbboxes, fp_lbboxes, img_names, shapes_before_pad, _
            
            
            



                



class FastImageDataset():
    def __init__(self, imgs, boxes, pids):
        self.imgs = imgs
        self.boxes = boxes
        self.pids = pids
    def __getitem__(self, idx):
        out = self.imgs[idx], self.boxes[idx], self.pids[idx]
        return out
    def __len__(self):
        return len(self.imgs)
    


        

