import numpy as np
import os
from os.path import join as pjoin
import csv
import warnings
from tqdm import tqdm

from utils_hsz import AnimationViewer, normalize
from global_variable import LUNA_DIR
from luna_dcm import FakeDicomReaderForLuna, load_itk
from dataset import Tumor, LungDataset

def read_annotation(fpath):
    labels = {}
    with open(fpath, "r") as f:
        f = csv.reader(f)
        for i, line in enumerate(f):
            if i==0: #title
                continue 
            excel_r = i+1
            pid, raw_x, raw_y, raw_z, diameter = line
            if pid in labels:
                labels[pid].append([excel_r, (raw_x, raw_y, raw_z, diameter)])
            else:
                labels[pid] = [[excel_r, (raw_x, raw_y, raw_z, diameter)]]
    return labels

def true2pixel_coordinate(img_size, raws, origin, spacing, extend=(0,0,0)):
    """
    Convert coorX, coorY, coorZ, diamters in true cooridinate to appproximate bbox using origin and spacing
    @Argument
        extend: How many pixels to extend on each axis after conversion; in (z,y,x) format
    """
    D, H, W = img_size 
    raw_x, raw_y, raw_z, diameter = map(lambda num: float(num), raws) #(xyz)
    ori_z, ori_y, ori_x = origin #(zyx)
    space_z, space_y, space_x = spacing #(zyx)
    ez, ey, ex = extend #(zyx)
    assert raw_x>=ori_x and raw_y>=ori_y and raw_z>=ori_z, "Bad datum"
    x, y, z = (raw_x-ori_x)/space_x, (raw_y-ori_y)/space_y, (raw_z-ori_z)/space_z
    # approximate x1,x2... by diameter
    d, h, w = diameter/space_z, diameter/space_y, diameter/space_x
    x1, x2 = max(round(x-w/2)-ex, 0), min(round(x+w/2)+ex, W-1)
    y1, y2 = max(round(y-h/2)-ey, 0), min(round(y+h/2)+ey, H-1)
    z1, z2 = max(round(z-d/2)-ez, 0), min(round(z+d/2)+ez, D-1)
    assert 0<=x1<=x2<W, "x1={}, x2={}, W={}".format(x1,x2,W)
    assert 0<=y1<=y2<H, "y1={}, y2={}, H={}".format(y1,y2,H)
    assert 0<=z1<=z2<D, "z1={}, z2={}, D={}".format(z1,z2,D)
    bbox = (z1, y1, x1, z2, y2, x2)
    return bbox

def view_luna_data():
    annofile = pjoin(LUNA_DIR, "annotations.csv")
    labels = read_annotation(annofile)
    print(list(labels.keys())[:3])
    for subset in range(10):
        dirpath = pjoin(LUNA_DIR, f"subset{subset}")
        #dirpath = pjoin(LUNA_DIR, "seg-lungs-LUNA16")
        for f in os.listdir(dirpath):
            fpath = pjoin(dirpath, f)
            if ".mhd" in f:
                pid = f.split(".mhd")[0]
                scan, origin, spacing = load_itk(fpath)
                scan = normalize(scan)
                img_size = scan.shape
                if pid in labels:
                    bboxes = []
                    for label in labels[pid]:
                        excel_r, raws = label
                        bbox = true2pixel_coordinate(img_size, raws, origin, spacing, (1,2,2))
                        bboxes.append(bbox)
                    AnimationViewer(scan, bboxes, note=f"{pid}\n")
                    #break
                else:
                    warnings.warn(f"PID '{pid}' not existed in labels")
        #break

def make_dataset(to_save_path, extend=(0,0,0)):
    dataset = LungDataset.empty()
    tumors = {}
    valid_rows = []
    pid_to_excel_r_relation = {}
    pids = []
    logs = "" # annotations.txt for froc.py!!


    annofile = pjoin(LUNA_DIR, "annotations.csv")
    labels = read_annotation(annofile)
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

def _test_luna_dataset(dataset_path):
    from torch.utils.data import DataLoader
    dataset = LungDataset.load(dataset_path)
    dataset.get_data(dataset.pids)
    for img, bboxes, pid in DataLoader(dataset, shuffle=True, batch_size=1):
        img, bboxes, pid = img[0], bboxes[0], pid[0]
        view_img = img.squeeze(-1).numpy()
        view_box = [bbox[:6] for bbox in bboxes.tolist()]
        AnimationViewer(view_img, bbox=view_box, verbose=False, note=f"{pid}")


if __name__ == "__main__":
    #view_luna_data()
    make_dataset("luna_test_dataset.pkl")
    #_test_luna_dataset("luna_test_dataset.pkl")
    