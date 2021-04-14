import torch
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

from dataset import Tumor, LungDataset
from global_variable import MASK_SAVED_PATH, CURRENT_DATASET_PKL_PATH
from utils_hsz import AnimationViewer
import config.yolov4_config as cfg


def test_lung_voi():
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    #dataset.get_data(["25607996"])
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    dataset.set_lung_voi(True)
    #err_fpath = pjoin(MASK_SAVED_PATH, "error_pid.txt")
    #with open(err_fpath, "r") as f:
    #    err_pids = f.read()[1:-1].split(",\n")
    #for datum in dataset.data.copy():
    #    _, _, pid = datum
    #    if pid in err_pids:
    #        dataset.data.remove(datum)
    max_shape = None
    max_shape_numel = 0
    for i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        img, bboxes, pid = datum
        #AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())
        numel = img.numel()
        if numel > max_shape_numel:
            max_shape = img.shape
            max_shape_numel = numel
        #if i>10:
        #    break

def test_data():
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    if (1): # may error here -> i.e. some crops have invalid bbox_coor
        dataset.set_random_crop(cfg.TRAIN["RANDOM_CROP_FILE_PREFIX"], cfg.TRAIN["RANDOM_CROP_NCOPY"], False)
    elif (0): #fine here
        dataset.set_batch_1_eval(True, (1.25, 0.75, 0.75))
        dataset.set_5mm(True, "fast_test_max_5.0x0.75x0.75.pkl")

    for i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        img, bboxes, pid = datum
        #AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())
        for bbox in bboxes:
            bbox_coor = bbox[:6]
            normal = (bbox_coor[3:] - bbox[:3])>=0
            if (not normal.all()) or (0):
                print("Invalid bbox detected in pid: {}".format(pid))
                print("bbox_coor:", bbox_coor)
                AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())

def find_data(pid:str):
    #21678302 has 4 bad crops
    #6993538 has 3 bad crops
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    """ #fine here
    for excel_r in dataset.pid_to_excel_r_relation[pid]:
        tumor = dataset.tumors[excel_r]
        print("original bbox:", tumor.voi)
        x1,x2 = tumor.voi["x"]
        y1,y2 = tumor.voi["y"]
        z1,z2 = tumor.voi["z"]
        bbox_coor = [z1,y1,x1,z2,y2,x2]
        img = tumor.get_series()
        AnimationViewer(img, [bbox_coor], note="excel_r: {}".format(excel_r))
    """
    dataset.get_data([pid])
    dataset.set_random_crop(cfg.TRAIN["RANDOM_CROP_FILE_PREFIX"], cfg.TRAIN["RANDOM_CROP_NCOPY"], False)
    for i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        img, bboxes, pid = datum
        #AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())
        for bbox in bboxes:
            bbox_coor = bbox[:6]
            normal = (bbox_coor[3:] - bbox[:3])>=0
            if (not normal.all()) or (1):
                print("Invalid bbox detected in pid: {}".format(pid))
                print("bbox_coor:", bbox_coor)
                AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())

if __name__ == "__main__":
    #test_lung_voi()
    test_data()
    #find_data("21678302")
