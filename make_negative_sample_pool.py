import torch
#from torch.utils.data import DataLoader
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join as pjoin
import argparse
from tqdm import tqdm
import pickle
from PIL import Image
import logging


from dataset import Tumor, LungDataset
#from databuilder.yolo4dataset import YOLO4_3DDataset
from trainer import Trainer
#from model.build_model import Build_Model
#import config.yolov4_config as cfg
from utils.log import Logger
from global_variable import CURRENT_DATASET_PKL_PATH, NEGATIVE_NPY_SAVED_PATH
from utils_ccy import linear_normalization


def make_negative_samples(save_crop, ncopy=1):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weight_path', type=str, default=None, help='weight file path')
    parser.add_argument('--gpu_id', type=int, default=0, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    parser.add_argument('--exp_name', type=str, default='debug', help='log experiment name')
    parser.add_argument('--dataset_name', type=str, default=CURRENT_DATASET_PKL_PATH)
    parser.add_argument('--npy_name', type=str, default="None.npy")
    opt = parser.parse_args()

    if save_crop and not os.path.exists(NEGATIVE_NPY_SAVED_PATH):
        os.makedirs(NEGATIVE_NPY_SAVED_PATH, exist_ok=True)

    for exp_name, fold_num in [

        ('train_rc_config_3_f0', 0),
        #('train_rc_config_3_f1', 1),
        #('train_rc_config_3_f2', 2),
        #('train_rc_config_3_f3', 3),
        #('train_rc_config_3_f4', 4),

        ]:

        testing_mode = 0 # use eval here to avoid repeat calculations
        epoch = 323
        fake_batch_size = 32 # need this to prevent accumulating fp crops in RAM
        eval_conf_thresh = 0.015

        opt.exp_name = exp_name
        logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='YOLOv4').get_log()
        phase = 'VAL'if testing_mode==0 else 'TEST' if testing_mode==1 else 'TRAIN_debug'
        #writer = SummaryWriter(log_dir=opt.log_path + '/{}_'.format(phase) + opt.exp_name)
        checkpoint_root = 'checkpoint/'
        checkpoint_folder = '{}{}'.format(checkpoint_root, exp_name)
        weight_path = '{}/backup_epoch{}.pt'.format(checkpoint_folder, str(epoch))
        if os.path.exists(weight_path):
            opt.weight_path = weight_path
            exp_name_folder = opt.exp_name + '_making_negatives'

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

            #area_dist, area_iou, plt, _, cpm_dist, cpm, max_sens_dist, max_sens_iou = trainer.evaluate()
            #target_pids = list({pid for _,_,pid in trainer.test_dataset.ori_dataset.data})
            target_pids = ["25607996"]
            remained_pids = target_pids.copy()
            n_batches = len(target_pids)//fake_batch_size + 1
            tqdm_bar = tqdm(total=n_batches, desc=f"Making fp fold {fold_num}")
            while len(remained_pids)!=0:
                pids = remained_pids[:fake_batch_size]
                remained_pids = remained_pids[fake_batch_size:]
                for c in range(ncopy):
                    out_imgs, out_bboxes, out_names = trainer.get_fp_for_reduction_batch(pids, return_crop_only=True)
                    ## saving crops
                    
                    for img, bboxes, pid in zip(out_imgs, out_bboxes, out_names):
                        to_save = (img, bboxes)
                        folder_name = os.path.join(NEGATIVE_NPY_SAVED_PATH, str(pid))
                        if save_crop and not os.path.exists(folder_name):
                            os.makedirs(folder_name, exist_ok=True)
                        pkl_name = os.path.join(folder_name, f"false_positive_c{c+1}.pkl")
                        if save_crop:
                            with open(pkl_name, "wb") as f:
                                pickle.dump(to_save, f)
                                print("save to", pkl_name)
                            save_slices_png(img, pjoin(folder_name, f"slices_view_c{c+1}"))
                tqdm_bar.update()



def save_slices_png(img, save_dir, z_slices=10):
    """z_slices indicate how many slices should be saved (cut evenly)"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    Z = img.shape[0]
    assert Z>=z_slices
    # (1) convert from [-1,1] to [0,255]
    img = linear_normalization(img, 0, 255)
    target_slices = [int(Z/(z_slices+1)*(i+1)) for i in range(z_slices)]
    for z_idx in target_slices:
        png_name = pjoin(save_dir, f"slice_{z_idx}.jpg")
        png = Image.fromarray( img[z_idx].astype(np.uint8) )
        png.save(png_name)
        

def make_negative_samples_for_luna():
    raise NotImplementedError() # may not need this
    import csv
    from dataset_luna import true2pixel_coordinate
    def read_candidates_V2(fpath):
        labels = {}
        with open(fpath, "r") as f:
            f = csv.reader(f)
            for i, line in enumerate(f):
                if i==0: #title
                    continue 
                excel_r = i+1
                pid, raw_x, raw_y, raw_z, cls_label = line
                if pid in labels:
                    labels[pid].append([excel_r, (raw_x, raw_y, raw_z, cls_label)])
                else:
                    labels[pid] = [[excel_r, (raw_x, raw_y, raw_z, cls_label)]]
        return labels
    annofile = pjoin(LUNA_DIR, "candidates_V2.csv")
    labels = read_candidates_V2(annofile)

    for subset in range(10):
        print("Processing subset:", subset)
        dirpath = pjoin(LUNA_DIR, f"subset{subset}")
        #dirpath = pjoin(LUNA_DIR, "seg-lungs-LUNA16")
        for f in tqdm(os.listdir(dirpath)):
            fpath = pjoin(dirpath, f)
            if ".mhd" in f:
                pid = f.split(".mhd")[0]
                scan, origin, spacing = load_itk(fpath)
                #scan = normalize(scan)
                img_size = scan.shape
                if pid in labels:
                    bboxes = []
                    for label in labels[pid]:
                        # get VOI
                        excel_r, raws = label
                        try:
                            bbox = true2pixel_coordinate(img_size, raws, origin, spacing, extend)
                        except:
                            warnings.warn(f"An error occurred at pid={pid}, excel_r={excel_r}; can't get bbox")
                            #raise
                            continue
                        bboxes.append(bbox)
                        z1, y1, x1, z2, y2, x2 = bbox
                        voi = {"x": (x1,x2), "y":(y1,y2), "z":(z1,z2)}
                        # make dcm for Tumor object to prevent error
                        fake_dcm = FakeDicomReaderForLuna(fpath, pid, spacing, img_size[0])
                        #scan = fake_dcm.get_series()
                        tumor = Tumor(excel_r, pid, fpath, voi, "", fake_dcm, tuple(img_size))
                        # write LungDataset attributes
                        tumors[excel_r] = tumor
                        valid_rows.append(excel_r)
                        if pid not in pid_to_excel_r_relation:
                            pid_to_excel_r_relation[pid] = [excel_r]
                        else:
                            pid_to_excel_r_relation[pid].append(excel_r)
                        pids.append(pid)
                    if len(bboxes)!=0: #write log
                        D,H,W = img_size
                        log = "{},{},{},{},".format(pid, D,H,W)
                        for bbox in bboxes:
                            z1,y1,x1,z2,y2,x2 = bbox
                            bbox_log = "{},{},{},{},{},{},0 ".format(z1,y1,x1,z2,y2,x2)
                            log += bbox_log
                        log = log[:-1]
                        logs += log + "\n"
                    if (0):
                        AnimationViewer(scan, bboxes, note=f"{pid}\n")
                    #break
                else:
                    pass
                    #warnings.warn(f"PID '{pid}' not existed in labels")
        #break
    dataset.tumors = tumors
    dataset.valid_rows = valid_rows
    dataset.pid_to_excel_r_relation = pid_to_excel_r_relation
    dataset.pids = pids
    logs = logs[:-1]
    # save
    with open("annotation_luna.txt", "w") as f:
        f.write(logs)
    dataset.save(to_save_path)


if __name__ == "__main__":
    make_negative_samples(save_crop=True, ncopy=3)

