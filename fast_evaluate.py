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

def fast_evaluate(npy_dir_path, pid, npy_name, exp_name, top_k=3, check_gt=False, batch_1_eval=False, fix_spacing=(0,0,0), use_lung_voi=False, use_5mm=False, load_5mm_pkl=False, view_raw=False, bold_box=True):
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

    # normal or raw output (no nms)   
    if view_raw:
        pred_boxes = np.load(pjoin(npy_dir_path, f"{pid}_test_raw.npy"))
    else:
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
            dataset.set_lung_voi(use_lung_voi)
            if use_5mm:
                dataset.set_5mm(use_5mm, load_5mm_pkl=load_5mm_pkl)
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
    bbox2 = [b[:6] for b in gt_boxes] if check_gt else None
    print("A (pred)", [box for box, _ in boxes])
    print("B (GT)", bbox2)
    utils_hsz.AnimationViewer(img, bbox=[box for box, _ in boxes], verbose=False, bbox2=bbox2, bold_box=bold_box)

def plot_img_with_bbox(pid = "10755333", extra_bbox=(), use_5mm=False, use_2d5mm=False, add_notFP=False):
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)#, name="hu+norm_128x128x128.npy")
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    if use_5mm:
        dataset.set_5mm(True, "fast_test_max_5.0x0.75x0.75.pkl")
    elif use_2d5mm:
        dataset.set_2d5mm(True, "fast_test_max_2.5x0.75x0.75.pkl")
    if type(pid)!=str: # if pid not assigned, use random pid
        import random
        pid = random.choice(dataset.pids)
    for i, datum in enumerate(dataset.data):
        if datum[2]==str(pid):
            key=i
            break
    img, gt_boxes, _ = dataset[key]
    boxes = [box[:6] for box in gt_boxes]
    if len(extra_bbox)!=0:
        boxes = boxes + list(extra_bbox)
    if add_notFP:
        import pandas as pd
        not_fp_excel_path = "D:/CH/LungDetection/not_fp_1.25mm.xlsx"
        df = pd.read_excel(not_fp_excel_path, sheet_name="Sheet1", converters={'pid':str,'bbox':str, 'isFP':int})
        df = df[df['pid']==pid]
        extra_bbox = [eval(lst) for lst in df["bbox"]]
        boxes = boxes + list(extra_bbox)
    img = img.squeeze(-1).numpy().astype(np.float32)
    utils_hsz.AnimationViewer(img, bbox=boxes)

if __name__ == "__main__":
    from view_dataset import ct_ldct
    #plot_img_with_bbox(use_2d5mm=True, add_notFP=True, pid=None)
    #raise EOFError
    #pid = "1926851"
    top_k = 5
    fix_spacing = (1.25,0.75,0.75)
    npy_name = "hu+norm_256x256x256.npy"
    exp_name = "train_256_256_256_1"

    # using cropped img
    batch_1_eval = False
    npy_name = "random_crop_128x128x128_1.25x0.75x0.75_c1.pkl"
    exp_name = "train_rc_config_1_f0_lung_voi"

    # using original(pad) img
    batch_1_eval = True
    npy_name = None
    use_lung_voi = True
    exp_name = "train_rc_config_5.6_resnest+sgd_shallower" #"train_rc_config_5.6.4_resnest_shallower_f0_fake_1.25_testing"
    fix_spacing = (1.25, 0.75, 0.75)

    # use 5mm or not
    use_5mm = False
    load_5mm_pkl = "fast_test_max_5.0x0.75x0.75.pkl"

    ## validation
    npy_dir_path = pjoin("checkpoint", exp_name, "evaluate")

    ## testing (draw_froc.py)
    exp_name = "train_rc_config_5.15_attnetion2_layer0_f2_EXTRA_FP" #272
    exp_name = "train_rc_config_5.15_attetion2_layer0_for_2d5mm_f2_EXTRA_FP" #170
    #exp_name += "_train_debug"
    exp_name += "_testing"
    epoch = 170
    #npy_dir_path = pjoin("preidction", exp_name, str(epoch), "evaluate")
    npy_dir_path = pjoin("preidction", exp_name, str(epoch)+"_conf0.015", "evaluate")
    use_5mm = False
    load_5mm_pkl = None

    ## choosing pid
    pids = [pid.split("_test")[0] for pid in os.listdir(npy_dir_path)]
    #pids_both = ct_ldct(return_pids=True)
    #pids_ct, pids_ldct = pids_both["CT"], pids_both["LDCT"]
    #pids = [pid for pid in pids if pid in pids_ct]
    
    #pids = ["1034114"]
    #pids = ["13564970"]
    #pids = [ os.listdir(npy_dir_path)[26] ]
    #pids = ['28719691', '23829163', '28753759']
    #pids = ['15179190'] # SA block good but CSP bad

    ## view_raw (mostly no use)
    view_raw = False

    ## bold box (whether to enhance bbox edge when visualization)
    bold_box = True

    for fname in pids:
        pid = fname.split("_test")[0]
        print("pid:", pid)
        fast_evaluate(npy_dir_path, pid, npy_name, exp_name, top_k, check_gt=True, batch_1_eval=batch_1_eval, fix_spacing=fix_spacing, use_lung_voi=use_lung_voi, use_5mm=use_5mm, load_5mm_pkl=load_5mm_pkl, view_raw=view_raw, bold_box=bold_box)

