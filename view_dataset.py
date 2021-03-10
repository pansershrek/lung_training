import torch
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm

from dataset import Tumor, LungDataset
from global_variable import MASK_SAVED_PATH, CURRENT_DATASET_PKL_PATH
from utils_hsz import AnimationViewer


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

if __name__ == "__main__":
    test_lung_voi()
