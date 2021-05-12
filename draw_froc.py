

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
import numpy as np

#from eval_coco import *
#from eval.cocoapi_evaluator import COCOAPIEvaluator

#from databuilder.abus import ABUSDetectionDataset
from dataset import Tumor, LungDataset
from global_variable import CURRENT_DATASET_PKL_PATH
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm
from trainer import Trainer
import config.yolov4_config as cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=None, help='weight file path')
    parser.add_argument('--gpu_id', type=int, default=0, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    parser.add_argument('--exp_name', type=str, default='debug', help='log experiment name')
    parser.add_argument('--dataset_name', type=str, default=CURRENT_DATASET_PKL_PATH)
    #parser.add_argument('--eval_interval', type=int, default=-1)
    #parser.add_argument('--npy_name', type=str, default="hu+norm_128x128x128.npy")
    parser.add_argument('--npy_name', type=str, default="hu+norm_256x256x256_fp16.npy")
    opt = parser.parse_args()


    for exp_name, fold_num, eval_epochs in [

        ('train_rc_config_5.6.4_resnest_shallower_f0', 0, 187),
        ('train_rc_config_5.6.5_resnest_shallower_f1', 1, 255),
        ('train_rc_config_5.6_resnest+sgd_shallower_f2', 2, 187),
        ('train_rc_config_5.6_resnest+sgd_shallower_f3', 3, 204),
        ('train_rc_config_5.6.2_resnest_shallower_f4', 4, 221)

        # 5mm model inference on 1.25
        #('train_rc_config_5.6.4_resnest_shallower_f0', 0, 187),
        #('train_rc_config_5.6.5_resnest_shallower_f1', 1, 255),
        #('train_rc_config_5.6_resnest+sgd_shallower_f2', 2, 187),
        #('train_rc_config_5.6_resnest+sgd_shallower_f3', 3, 204),
        #('train_rc_config_5.6.2_resnest_shallower_f4', 4, 221),


        #('train_rc_config_5.7.2_SEConv_f2', 2, 272),

        
        #('train_5mm_max_no_fp_reduction_dry_run_f0', 0, list(range(0,220,17))+[220]),
        #('train_5mm_max_no_fp_reduction_dry_run_f1', 1, list(range(0,220,17))+[220]),
        #('train_5mm_max_no_fp_reduction_dry_run_f2', 2, list(range(0,220,17))+[220]),
        #('train_5mm_max_no_fp_reduction_dry_run_f3', 3, list(range(85+17,220,17))+[220]),
        #('train_5mm_max_no_fp_reduction_dry_run_f4', 4, list(range(0,220,17))+[220]),

        
        ]:
        eval_epochs = [eval_epochs] if (not hasattr(eval_epochs, "__len__")) else eval_epochs
        eval_conf_thresh = [0.015]  ## original: 0.015
        for testing_mode in [1]: #[0, 1, -1, -2]:
            #if testing_mode==0 and exp_name=='Fd0_BS2_Stem4_8_128_r2':
            #    continue
            if (1):
                opt.exp_name = exp_name
                #opt.exp_name = exp_name + "_diff_score"
                #opt.exp_name = exp_name
            if cfg.TRAIN["EXTRA_FP_USAGE"] == "eval_only":
                opt.exp_name = opt.exp_name + "_EXTRA_FP"
            logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='YOLOv4').get_log()
            checkpoint_root = 'checkpoint/' #'/home/lab402/p08922003/YOLOv4-pytorch/checkpoint/'
            checkpoint_folder = '{}{}'.format(checkpoint_root, exp_name)
            checkpoints = os.listdir(checkpoint_folder)

            phase = 'VAL'if testing_mode==0 else 'TEST' if testing_mode==1 else 'TRAIN_debug' if testing_mode==-1 else 'TRAIN_whole_debug'
            writer = SummaryWriter(log_dir=opt.log_path + '/{}_'.format(phase) + opt.exp_name)
            eval_conf_thresh_list = eval_conf_thresh if hasattr(eval_conf_thresh, "__iter__") else [eval_conf_thresh]
            for eval_conf_thresh in eval_conf_thresh_list:
                for epoch in eval_epochs: #range(204, 306, 17):#list(range(45, 65+1, 10))+[120]: #[255,425,646]: #range(255, 425+1, 17):
                    weight_path = '{}/backup_epoch{}.pt'.format(checkpoint_folder, str(epoch))
                    if os.path.exists(weight_path):
                        opt.weight_path = weight_path
                        exp_name_folder = opt.exp_name
                        if testing_mode==1:
                            exp_name_folder = opt.exp_name + '_testing'
                        elif testing_mode==-1 or testing_mode==-2:
                            exp_name_folder = opt.exp_name + '_train_debug'
                        checkpoint_save_dir = 'preidction/{}/{}_conf{}'.format(exp_name_folder, str(epoch), eval_conf_thresh)
                        if not os.path.exists('preidction'):
                            os.mkdir('preidction')

                        if not os.path.exists('preidction/{}'.format(exp_name_folder)):
                            os.mkdir('preidction/{}'.format(exp_name_folder))

                        if not os.path.exists(checkpoint_save_dir):
                            os.mkdir(checkpoint_save_dir)

                        weight_path = opt.weight_path
                        #weight_path = 'checkpoint/96_B4_F1/backup_epoch150.pt'
                        trainer = Trainer(testing_mode=testing_mode,
                                weight_path=weight_path,
                                checkpoint_save_dir=checkpoint_save_dir,
                                resume=False,
                                gpu_id=opt.gpu_id,
                                accumulate=1,
                                fp_16=opt.fp_16,
                                writer=None,
                                logger=logger,
                                crx_fold_num=fold_num,
                                dataset_name=opt.dataset_name,
                                eval_interval=None,
                                npy_name=opt.npy_name,
                                eval_conf_thresh=eval_conf_thresh,
                                )

                        area_dist, area_iou, plt, _, cpm_dist, cpm, max_sens_dist, max_sens_iou = trainer.evaluate()
                        writer.add_scalar('AUC (IOU)', area_iou, epoch)
                        writer.add_scalar('CPM (IOU)', cpm, epoch)
                        writer.add_scalar('AUC (dist)', area_dist, epoch)
                        writer.add_scalar('CPM (dist)', cpm_dist, epoch)
                        writer.add_scalar('Max sens(iou)', max_sens_iou, epoch)
                        writer.add_scalar('Max sens(dist)', max_sens_dist, epoch)


