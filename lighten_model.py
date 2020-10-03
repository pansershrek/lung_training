import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import config.yolov4_config as cfg

from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator


class lightenYOLOv4(pl.LightningModule):
    def __init__(self, weight_path, resume, gpu_id, accumulate, fp_16):
        super().__init__()
        self.model = Build_Model(weight_path=weight_path, resume=resume)
        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.optmizer = optim.SGD(self.yolov4.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

    def forward(self, img):
        p, p_d = self.model(img)
    
    def training_step(self, batch, batch_idx):
        img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = batch

        p, p_d = self.model(img)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
        return loss, loss_ciou, loss_conf, loss_cls
    
    def configure_optimizers(self):
        return self.optmizer