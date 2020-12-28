

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

#from eval_coco import *
#from eval.cocoapi_evaluator import COCOAPIEvaluator

#from databuilder.abus import ABUSDetectionDataset
from dataset import Tumor, LungDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from tqdm import tqdm
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=None, help='weight file path')
    parser.add_argument('--gpu_id', type=int, default=0, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    parser.add_argument('--exp_name', type=str, default='debug', help='log experiment name')
    parser.add_argument('--dataset_name', type=str, default="lung_dataset_20201215.pkl")
    #parser.add_argument('--eval_interval', type=int, default=-1)
    #parser.add_argument('--npy_name', type=str, default="hu+norm_128x128x128.npy")
    parser.add_argument('--npy_name', type=str, default="hu+norm_256x256x256_fp16.npy")
    opt = parser.parse_args()


    for exp_name, fold_num in [
        #('train_128x128x128_2_f3', 3),
        ('train_256_256_256_1', 0),

        #('Fd0_BS1_Stem4_16_256_r2', 0),
        #('Fd1_BS1_Stem4_16_256_r2', 1),
        #('Fd2_BS1_Stem4_16_256_r2', 2),
        #('Fd3_BS1_Stem4_16_256_r2', 3),
        #('Fd4_BS1_Stem4_16_256_r2', 4),
        ]:
        for testing_mode in [0]: #[0, 1]:
            #if testing_mode==0 and exp_name=='Fd0_BS2_Stem4_8_128_r2':
            #    continue

            opt.exp_name = exp_name
            logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='YOLOv4').get_log()
            checkpoint_root = 'checkpoint/' #'/home/lab402/p08922003/YOLOv4-pytorch/checkpoint/'
            checkpoint_folder = '{}{}'.format(checkpoint_root, exp_name)
            checkpoints = os.listdir(checkpoint_folder)

            phase = 'VAL'if testing_mode==0 else 'TEST' if testing_mode==1 else 'TRAIN_debug'
            writer = SummaryWriter(log_dir=opt.log_path + '/{}_'.format(phase) + opt.exp_name)

            for epoch in range(255, 425+1, 17): #[255,425,646]: #range(255, 425+1, 17):
                weight_path = '{}/backup_epoch{}.pt'.format(checkpoint_folder, str(epoch))
                if os.path.exists(weight_path):
                    opt.weight_path = weight_path
                    exp_name_folder = opt.exp_name
                    if testing_mode==1:
                        exp_name_folder = opt.exp_name + '_testing'
                    checkpoint_save_dir = 'preidction/{}/{}'.format(exp_name_folder, str(epoch))
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
                            npy_name=opt.npy_name)

                    area_small, area_big, plt, _ = trainer.evaluate()
                    writer.add_scalar('AUC', area_big, epoch)
                    #writer.add_scalar('AUC_10mm', area_small, epoch)
                    #writer.add_scalar('AUC_15mm', area_big, epoch)


