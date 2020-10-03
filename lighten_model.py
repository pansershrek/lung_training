import os
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from tqdm import tqdm

from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import config.yolov4_config as cfg

from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator
from eval.evaluator import Evaluator

import utils.gpu as gpu

class lightenYOLOv4(pl.LightningModule):
    def __init__(self, weight_path, resume, accumulate=None):
        # precision=16 for fp16

        super().__init__()
        self.model = Build_Model(weight_path=weight_path, resume=resume)
        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.evaluator = Evaluator(self.model, showatt=False)

    # how you want your model to do inference/predictions
    def forward(self, img):
        p, p_d = self.model(img)
        return p, p_d

    # the train loop INDEPENDENT of forward.
    def training_step(self, batch, batch_idx):
        img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, _ = batch

        p, p_d = self(img)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

        '''
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('train_loss_ciou', loss_ciou)
        result.log('train_loss_conf', loss_conf)
        result.log('train_loss_cls', loss_cls)
        result.log('train_loss', loss, on_epoch=True)
        '''

        return loss

    def sample_validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)
        return {}

    def validation_epoch_end(self, outputs):
        #         self.coco_evaluator.accumulate()
        #         self.coco_evaluator.summarize()
        #         # coco main metric
        #         metric = self.coco_evaluator.coco_eval['bbox'].stats[0]
        APs = self.evaluator.calc_APs()
        mAP = 0
        for i in APs:
            #logger.info("{} --> mAP : {}".format(i, APs[i]))
            mAP += APs[i]
        mAP = mAP / self.model.getNC()
        #logger.info("mAP : {}".format(mAP))
        #logger.info("inference time: {:.2f} ms".format(inference_time))
        #writer.add_scalar('mAP', mAP, epoch)
        #self.__save_model_weights(epoch, mAP)
        #logger.info('save weights done')

        #tensorboard_logs = {'main_score': metric}
        #return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        return {'val_mAP': mAP}

    def validation_step(self, batch, batch_idx):
        img_batch, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_name = batch

        for idx, img in tqdm(zip(img_name, img_batch)):
            # CHW -> HWC
            img = img.cpu().numpy().transpose(1, 2, 0)
            bboxes_prd = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
            self.evaluator.store_bbox(idx, bboxes_prd)
        '''
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

        self.log('val_loss_ciou', loss_ciou)
        self.log('val_loss_conf', loss_conf)
        self.log('val_loss_cls', loss_cls)
        self.log('val_loss', loss)
        '''

        return 1

    def test_step(self, batch, batch_idx):
        img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = batch

        p, p_d = self(img)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
        #, loss_ciou, loss_conf, loss_cls
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                        momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        return optimizer
