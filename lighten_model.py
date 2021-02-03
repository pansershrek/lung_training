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
from utils import cosine_lr_scheduler

from tqdm import tqdm
import time
class lightenYOLOv4(pl.LightningModule):
    def __init__(self, weight_path, resume, exp_name, accumulate=None, dims=2):
        # precision=16 for fp16

        super().__init__()
        self.model = Build_Model(weight_path=weight_path, resume=resume, dims=dims)
        self.criterion = YoloV4Loss(anchors=cfg.MODEL["ANCHORS3D"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"], dims=dims)

        self.evaluator = Evaluator(self.model, showatt=False, exp_name=exp_name)
        self.evaluator.clear_predict_file()
        self.train_step_counter = 0
        self.optimizer = []
    # how you want your model to do inference/predictions
    def forward(self, img):
        p, p_d = self.model(img)
        return p, p_d
    """
    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = 0
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # creating log dictionary
        result = pl.TrainResult()
        result.log('val/loss_epoch', avg_loss)
        return result
    """


    # the train loop INDEPENDENT of forward.
    def training_step(self, batch, batch_idx):
        self.train_step_counter+=1

        opt_g = self.trainer.optimizers[0]
        img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, _ = batch
        avg_dict = {}
        if 1:
            for i in tqdm(range(1000000)):
                t = time.time()
                torch.cuda.synchronize()
                p, p_d = self.model(img)

                loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
                loss.backward()
                torch.cuda.synchronize()
                print("time:{:.4f}".format(time.time() - t))

            for i in tqdm(range(1000000)):

                time_dict = {}
                torch.cuda.synchronize()
                t = time.time()
                # do anything you want
                p, p_d = self(img)
                torch.cuda.synchronize()
                time_dict['forward'] = time.time() - t
                t = time.time()
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                    label_lbbox, sbboxes, mbboxes, lbboxes)
                torch.cuda.synchronize()
                time_dict['compute_loss'] = time.time() - t

                t = time.time()
                loss.backward()
                torch.cuda.synchronize()
                time_dict['backward'] = time.time() - t

                t = time.time()
                # use self.backward which will also handle scaling the loss when using amp
                #self.manual_backward(loss_a, opt_g)
                opt_g.step()
                torch.cuda.synchronize()
                time_dict['optim.step'] = time.time() - t

                t = time.time()
                opt_g.zero_grad()
                torch.cuda.synchronize()
                time_dict['optim.zero_grad'] = time.time() - t
                for k in time_dict.keys():
                    if not k in avg_dict:
                        avg_dict[k] = []
                    avg_dict[k].append(time_dict[k])
                    print('{} avg:{:.4f}'.format(k, np.mean(avg_dict[k])))

        p, p_d = self(img)
        loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
        result = pl.TrainResult(minimize=loss)

        #result.log('train_loss', loss, on_step=True, on_epoch=True)
        #result.log('train_loss_ciou', loss_ciou, on_step=True, on_epoch=True)
        #result.log('train_loss_conf', loss_conf, on_step=True, on_epoch=True)
        #result.log('train_loss_cls', loss_cls, on_step=True, on_epoch=True)
        #result.log('lr', self.optimizer.param_groups[0]["lr"], on_step=True, on_epoch=True)
        return result
        '''
        #https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
        logs={"train_loss": loss,
            "train_loss_ciou":loss_ciou,
            "train_loss_conf":loss_conf,
            "train_loss_cls":loss_cls,}
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            #optional for batch logging purposes
            "log": logs,
        }
        return batch_dictionary

        '''



    def validation_epoch_end(self, outputs):
        #APs = self.evaluator.calc_APs()
        APs = [0, 0]
        self.evaluator.clear_predict_file()
        mAP = 0
        for i in APs:
            mAP += APs[i]
        mAP = mAP / self.model.getNC()
        result = pl.EvalResult()
        result.log('val/mAP_epoch', torch.Tensor([mAP]).cuda())
        #trainer.logger_connector.logged_metrics
        return result

    def validation_step(self, batch, batch_idx):
        img_batch, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, img_name = batch

        for idx, img in tqdm(zip(img_name, img_batch)):
            bboxes_prd = self.evaluator.get_bbox(img, multi_test=False, flip_test=False)
            pass
            #self.evaluator.store_bbox(idx, bboxes_prd)
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

        scheduler = cosine_lr_scheduler.CosineDecayLR(optimizer,
            T_max=cfg.TRAIN['YOLO_EPOCHS'],
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"])
        self.optimizer = optimizer
        return [optimizer], [scheduler]
