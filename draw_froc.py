

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
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger

from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator

from databuilder.abus import ABUSDetectionDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs,0),targets


class Trainer(object):
    def __init__(self, weight_path, checkpoint_save_dir, resume, gpu_id, accumulate, fp_16):
        init_seeds(0)
        self.fp_16 = fp_16
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.accumulate = accumulate
        self.weight_path = weight_path
        self.checkpoint_save_dir = checkpoint_save_dir
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train:print('Using multi scales training')
        else:print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.epochs = cfg.TRAIN["YOLO_EPOCHS"] if cfg.MODEL_TYPE["TYPE"] == 'YOLOv4' else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]

        train_dataset = ABUSDetectionDataset(augmentation=False, crx_fold_num= 0, crx_partition= 'train', crx_valid=True, include_fp=False, root='datasets/abus')
        self.train_dataset = YOLO4_3DDataset(train_dataset, classes=[0, 1], img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])

        test_dataset = ABUSDetectionDataset(augmentation=False, crx_fold_num= 0, crx_partition= 'valid', crx_valid=True, include_fp=False, root='datasets/abus')
        self.test_dataset = YOLO4_3DDataset(test_dataset, classes=[0, 1], img_size=cfg.VAL["TEST_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True, pin_memory=True
                                           )
        self.test_dataloader = DataLoader(self.test_dataset,
                                            batch_size=cfg.VAL["BATCH_SIZE"],
                                            num_workers=cfg.VAL["NUMBER_WORKERS"],
                                            shuffle=False, pin_memory=True
                                            )
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
        if resume: self.__load_resume_weights(weight_path)

    def __load_resume_weights(self, weight_path):
        last_weight = os.path.join(weight_path)
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
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

        if epoch > 0 and epoch % 20 == 0:
            torch.save(chkpt, os.path.join(self.checkpoint_save_dir, 'backup_epoch%g.pt'%epoch))
        if epoch==0 or (self.best_mAP == mAP and mAP>0):
            torch.save(chkpt, os.path.join(self.checkpoint_save_dir, "best.pt"))
        #torch.save(chkpt, os.path.join(self.checkpoint_save_dir, "last.pt"))
        del chkpt

    def evaluate(self):
        global writer
        logger.info("Evaluate start,img size is: {},batchsize is: {:d},work number is {:d}".format(cfg.VAL["TEST_IMG_SIZE"],cfg.VAL["BATCH_SIZE"],cfg.VAL["NUMBER_WORKERS"]))
        logger.info(self.model)
        logger.info("Test datasets number is : {}".format(len(self.test_dataloader)))

        if self.fp_16: self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)
        logger.info("        =======  start  evaluate   ======     ")
        start = time.time()
        self.model.eval()
        mloss = torch.zeros(4)
        pred_result_path=os.path.join(self.checkpoint_save_dir, 'evaluate')
        self.evaluator = Evaluator(self.model, showatt=False, pred_result_path=pred_result_path, box_top_k=50)
        self.evaluator.clear_predict_file()
        TOP_K = 50
        with torch.no_grad():
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_names)  in tqdm(enumerate(self.train_dataloader)):
                imgs = imgs.to(self.device)
                for img, img_name in zip(imgs, img_names):
                    bboxes_prd = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
                    self.evaluator.store_bbox(img_name, bboxes_prd)
                if 0:
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    sbboxes = sbboxes.to(self.device)
                    mbboxes = mbboxes.to(self.device)
                    lbboxes = lbboxes.to(self.device)
                    p, p_d = self.model(imgs)
                    self.optimizer.zero_grad()
                    loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                        label_lbbox, sbboxes, mbboxes, lbboxes)
                    # Update running mean of tracked metrics
                    loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
                    mloss = (mloss * i + loss_items) / (i + 1)
            root = 'datasets/abus/'
            npy_dir = 'data/pred_result/evaluate/'
            npy_format = npy_dir + '{}_0.npy'
            calculate_FROC(root, npy_dir, npy_format, size_threshold=20)
        if 0:
            mAP = 0.
            with torch.no_grad():
                APs, inference_time = Evaluator(self.model, showatt=False).APs_voc()
                for i in APs:
                    logger.info("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.train_dataset.num_classes
                logger.info("mAP : {}".format(mAP))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                #writer.add_scalar('mAP', mAP, epoch)
                logger.info('save weights done')
            logger.info("  ===test mAP:{:.3f}".format(mAP))
        end = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=None, help='weight file path')#weight/darknet53_448.weights
    parser.add_argument('--gpu_id', type=int, default=-1, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    opt = parser.parse_args()
    weight_path = opt.weight_path

    checkpoint_save_dir = 'checkpoint/' + 'evaluate'
    if not os.path.exists(checkpoint_save_dir):
        os.mkdir(checkpoint_save_dir)

    trainer = Trainer(
            resume=True,
            gpu_id=opt.gpu_id,
            weight_path=weight_path,
            checkpoint_save_dir=checkpoint_save_dir,
            accumulate=1,
            fp_16=False)
    logger = Logger(log_level=logging.DEBUG, logger_name='YOLOv4').get_log()
    trainer.evaluate()