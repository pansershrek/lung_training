import torch
import numpy as np
import os
from os.path import join as pjoin

from dataset import Tumor, LungDataset
from global_variable import MASK_SAVED_PATH, CURRENT_DATASET_PKL_PATH


def test_lung_voi():
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    dataset.set_lung_voi(True)
    err_fpath = pjoin(MASK_SAVED_PATH, "error_pid.txt")
    with open(err_fpath, "r") as f:
        err_pids = f.read()[1:-1].split(",\n")
    for datum in dataset.data.copy():
        _, _, pid = datum
        if pid in err_pids:
            dataset.data.remove(datum)
    for i, datum in enumerate(dataset):
        ...
        if i>10:
            break

if __name__ == "__main__":
    test_lung_voi()
