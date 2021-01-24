import numpy as np
import os
from os.path import join as pjoin
import pickle

from dataset import Tumor, LungDataset
import global_variable
from global_variable import CURRENT_DATASET_PKL_PATH
import utils_hsz
import config.yolov4_config as cfg
#import config.yolov4_config as cfg

def fast_evaluate(npy_dir_path, pid, npy_name, exp_name, top_k=3, check_gt=False, batch_1_eval=False, fix_spacing=(0,0,0)):
    """
    evaluate predictiob result based on...
    1. predicted bbox npy
    2. pid of the image
    3. LungDataset
    """
    assert npy_name==None or npy_name.endswith(".npy") or (npy_name.startswith("random_crop") and npy_name.endswith(".pkl"))
    if npy_name == None:
        pass
    elif npy_name.endswith(".npy"):
        img = np.load(pjoin(global_variable.NPY_SAVED_PATH, str(pid), npy_name))
    elif npy_name.endswith(".pkl"):
        with open(pjoin(global_variable.NPY_SAVED_PATH, str(pid), npy_name), "rb") as f:
            img, pkl_gt = pickle.load(f)
        
    pred_boxes = np.load(pjoin(npy_dir_path, f"{pid}_test.npy"))
    boxes={}
    lowest_conf = float("-inf")
    for box in pred_boxes:
        pred_conf = box[-1]
        if pred_conf > lowest_conf:
            """if pred_conf in boxes:
                exist_box = boxes[pred_conf]
                if type(exist_box)==list:
                    boxes[pred_conf].append(box)
                else:
                    boxes[pred_conf] = [exist_box, box]
            else:
                """
            boxes[pred_conf] = box
            if len(boxes)>top_k:
                boxes.pop(lowest_conf)
            lowest_conf = min(boxes)
        elif len(boxes) < top_k:
            boxes[pred_conf] = box
            lowest_conf = min(boxes)
    boxes = [(box[:6], conf) for conf, box in boxes.items()]
    if check_gt:
        if npy_name==None:
            dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
            dataset.get_data(dataset.pids)
            dataset.set_batch_1_eval(batch_1_eval, fix_spacing)
            for i, datum in enumerate(dataset.data):
                if datum[2]==str(pid):
                    key=i
                    break
            img, gt_boxes, _ = dataset[key] # This is the **scaled bbox**, according to shape of the npy!!
            img = img.squeeze_(-1).numpy()
        elif npy_name.endswith(".npy"):
            dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
            dataset.get_data(dataset.pids, name=npy_name)
            for i, datum in enumerate(dataset.data):
                if datum[2]==str(pid):
                    key=i
                    break
            gt_boxes = dataset[key][1] # This is the **scaled bbox**, according to shape of the npy!!
        elif npy_name.endswith(".pkl"):
            gt_boxes = pkl_gt
        print("scaled GT bbox:", gt_boxes)
    print("lower bound of top_k conf:", lowest_conf)
    print("bbox:", boxes)
    if batch_1_eval:
        def trans(x, base=cfg.MODEL["BASE_MULTIPLE"]):
	            return x + base - x%base if x%base else x
        pad_d, pad_h, pad_w = map(trans, img.shape)
        pad_d, pad_h, pad_w = pad_d-img.shape[0], pad_h-img.shape[1], pad_w-img.shape[2]
        pad_img = np.pad(img, [[0,pad_d],[0,pad_h],[0,pad_w]])
        img = pad_img
    utils_hsz.AnimationViewer(img, bbox=[box for box, _ in boxes], verbose=False)

def plot_img_with_bbox(pid = "10755333"):
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)#, name="hu+norm_128x128x128.npy")
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    for i, datum in enumerate(dataset.data):
        if datum[2]==str(pid):
            key=i
            break
    img, gt_boxes, _ = dataset[key]
    boxes = [box[:6] for box in gt_boxes]
    img = img.squeeze(-1).numpy().astype(np.float32)
    utils_hsz.AnimationViewer(img, bbox=boxes)

if __name__ == "__main__":
    #plot_img_with_bbox()
    #raise EOFError
    #pid = "1926851"
    top_k = 10
    fix_spacing = (1.25,0.75,0.75)
    npy_name = "hu+norm_256x256x256.npy"
    exp_name = "train_256_256_256_1"

    #npy_name = "random_crop_128x128x128_1.25x0.75x0.75_c1.pkl"
    batch_1_eval = False
    npy_name = "random_crop_128x128x128_1.25x0.75x0.75_c1.pkl"
    exp_name = "train_rc_luna_f3_try_train_debug"

    ## validation
    #npy_dir_path = pjoin("checkpoint", exp_name, "evaluate")

    ## testing (draw_froc.py)
    epoch = 132
    npy_dir_path = pjoin("preidction", exp_name, str(epoch), "evaluate")

    pids = os.listdir(npy_dir_path)
    #pids = [""]
    for fname in pids:
        pid = fname.split("_test")[0]
        print("pid:", pid)
        fast_evaluate(npy_dir_path, pid, npy_name, exp_name, top_k, check_gt=True, batch_1_eval=batch_1_eval, fix_spacing=fix_spacing)

