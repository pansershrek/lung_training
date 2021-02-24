import cc3d
from scipy.ndimage import morphology
import random
import numpy as np
from tqdm import tqdm
import warnings
import os
from os.path import join as pjoin
import gzip

from utils_hsz import AnimationViewer
from dataset import Tumor, LungDataset
from global_variable import CURRENT_DATASET_PKL_PATH, MASK_SAVED_PATH

def binarize(arr, thre):
    """binarize the img with threshold"""
    return (arr >= thre).astype(np.int64)

def get_bbox(arr):
    """get bbox from mask image"""
    zs = np.any(arr, axis=(1,2))  #non zero part
    ys = np.any(arr, axis=(0,2))
    xs = np.any(arr, axis=(0,1))
    
    x1, x2 = np.where(xs)[0][[0, -1]]
    y1, y2 = np.where(ys)[0][[0, -1]]
    z1, z2 = np.where(zs)[0][[0, -1]]

    return (z1,y1,x1,z2,y2,x2)

"""def save_npy_gz(fpath, arr):
    with gzip.GzipFile(fpath, "w") as f:
        f.write(arr)

def load_npy_gz(fpath):
    with gzip.GzipFile(fpath, "r") as f:
        arr = np.load(f)
    return arr"""

def get_lung_part(img, gt_bboxes, debug_view=False):
    """get lung part bbox from img, gt_bboxes are used to check VOI validity only"""

    ## (1) binarize with threshold
    binarized = binarize(img, thre=0) 

    ## (2) use erosion to eliminate artifacts
    kernel = np.ones((1,10,1)).astype(np.bool)
    eroded = morphology.binary_erosion(binarized.astype(np.bool), kernel)

    ## (3) invert the mask
    inv = np.invert(eroded)

    ## (4) use connected component to get two lobes of lung
    connectivity = 6 # 26, 18, 6 for 3D
    labels_out, N = cc3d.connected_components(inv, connectivity=connectivity, return_N=True)
    label_ranker = {}
    for label, image in cc3d.each(labels_out, binary=True, in_place=True):
        s = image.sum()
        label_ranker[label] = s
        if label > 10 :
            break
    descending_label = sorted(label_ranker, key=label_ranker.__getitem__, reverse=True)

    two_lobes_mask = (labels_out==descending_label[1]) + (labels_out==descending_label[2]) # 2nd and 3rd largest black part
    #two_lobes_mask = (labels_out!=descending_label[0])
    bbox = get_bbox(two_lobes_mask)

    ## (5) if VOI is invalid, use larget white part rather than 2nd+3rd largest black part
    if not check_voi(bbox, gt_bboxes, img.shape):
        print("Try brightest VOI")
        connectivity = 6 # 26, 18, 6 for 3D
        labels_out, N = cc3d.connected_components(eroded, connectivity=connectivity, return_N=True)
        label_ranker = {}
        for label, image in cc3d.each(labels_out, binary=True, in_place=True):
            s = image.sum()
            label_ranker[label] = s
            if label > 10 :
                break
        descending_label = sorted(label_ranker, key=label_ranker.__getitem__, reverse=True)
        bright_part_mask = labels_out==descending_label[0]
        bbox = get_bbox(bright_part_mask)

    if debug_view:
        AnimationViewer(img, note="Original")
        AnimationViewer(binarized, note="Binarized")
        AnimationViewer(eroded, note="Eroded")
        AnimationViewer(inv, note="Inverted")
        AnimationViewer(labels_out, note="raw cc3d")
        print("label_ranking:", label_ranker)
        print("descending_label:", descending_label)
        AnimationViewer(two_lobes_mask, note="cc3d filtered")
        #AnimationViewer(eroded2, note="Eroded2")
        #print("bbox:", bbox)
        AnimationViewer(img, bbox=[bbox])
    return bbox, two_lobes_mask



    
def check_voi(voi, bboxes, shape):
    """check if the voi is reasonable"""
    vz1, vy1, vx1, vz2, vy2, vx2 = voi
    Z,Y,X = shape
    if (not 0<=vz1<vz2<Z) or (not 0<=vy1<vy2<Y) or (not 0<=vx1<vx2<X):
        return False
    for bbox in bboxes:
        z1,y1,x1,z2,y2,x2 = bbox[:6]
        if (not vz1<=z1<=z2<=vz2) or (not vy1<=y1<=y2<=vy2) or (not vx1<=x1<=x2<=vx2):
            print("bbox {} be trimmed for voi {}. shape={}".format(bbox[:6], voi, shape))
            return False
    return True

def dataset_preprocessing(save=False, mask_name="mask.npy", debug_view=False, bypass_pids=()):
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    if (1): # process error_pid only
        fname = pjoin(MASK_SAVED_PATH, "error_pid.txt")
        with open(fname, "r") as f:
            err_pids = f.read()[1:-1].split(",\n")
        dataset.get_data(err_pids)
        dataset.get_data(["10378435"])
    else:
        dataset.get_data(dataset.pids)
    ##dataset.data = dataset.data[:10]
    errors = []
    vois = {}
    if not os.path.exists(MASK_SAVED_PATH):
        os.mkdir(MASK_SAVED_PATH)
    assert mask_name.endswith(".npy")
    for img, bboxes, pid in tqdm(dataset):
        img = img.squeeze(-1).numpy()
        lung_voi, two_lobes_mask = get_lung_part(img, bboxes, debug_view=debug_view)
        z1, y1, x1, z2, y2, x2 = lung_voi
        if pid not in bypass_pids:
            if not check_voi(lung_voi, bboxes, img.shape):
                warnings.warn(f"pid: {pid} failed check_voi")
                errors.append(pid)
        else:
            warnings.warn(f"pid: {pid} bypass...")
        lung_voi_shape = (z2-z1, y2-y1, x2-x1)
        #AnimationViewer(img, bbox=[lung_voi])
        #print("pid", pid, img.shape)
        #print("voi coor:", lung_voi)
        #print("voi shape:", lung_voi_shape)
        vois[pid] = lung_voi
        if save:
            os.makedirs(pjoin(MASK_SAVED_PATH, pid), exist_ok=True)
            fname = pjoin(MASK_SAVED_PATH, pid, mask_name)
            np.save(fname, two_lobes_mask) # too large
            #save_npy_gz(fname, two_lobes_mask)
 
    if save:
        fname = pjoin(MASK_SAVED_PATH, "VOI_v3.txt")
        out_txt = ""
        for pid in vois:
            out_txt += "{} {}\n".format(pid, vois[pid])
        out_txt = out_txt[:-1]
        with open(fname, "w") as f:
            f.write(out_txt)
        
        err_txt = "[" + ",\n".join(errors) + "]"
        fname = pjoin(MASK_SAVED_PATH, "error_pid_v3.txt")
        with open(fname, "w") as f:
            f.write(err_txt)
        
    print(  "Invalid: {}/{} ({:.2f}%)".format(len(errors), len(dataset), len(errors)/len(dataset)*100)  )
    print("All Errors:", errors)

def single_volume_preprocessing(arr):
    """ no checking!!! """
    lung_voi = get_lung_part(arr)
    return lung_voi


def _test():
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(["1473397"]) # 1473397, 1149150
    for img, bboxes, pid in tqdm(dataset):
        img = img.squeeze(-1).numpy()
        lung_part = get_lung_part(img, debug_view=True)
        z1, y1, x1, z2, y2, x2 = lung_part
        lung_shape = (z2-z1, y2-y1, x2-x1)
        print("pid", pid, img.shape)
        print("voi coor:", lung_part)
        print("voi shape:", lung_shape)

if __name__ == "__main__":
    bypass_pids = ("10378435",)
    dataset_preprocessing(save=True, mask_name="mask_v2.npy", debug_view=False, bypass_pids=bypass_pids)
    #_test()

