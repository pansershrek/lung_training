import numpy as np
import os
from os.path import join as pjoin

from dataset import Tumor, LungDataset
import global_variable
import utils_hsz

def fast_evaluate(npy_dir_path, pid, npy_name, exp_name, top_k=3, check_gt=False):
    """
    evaluate predictiob result based on...
    1. predicted bbox npy
    2. pid of the image
    3. LungDataset
    """
    assert npy_name.endswith(".npy")
    img = np.load(pjoin(global_variable.NPY_SAVED_PATH, str(pid), npy_name))
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
        dataset = LungDataset.load("lung_dataset_20201215.pkl")
        dataset.get_data(dataset.pids, name=npy_name)
        for i, datum in enumerate(dataset.data):
            if datum[2]==str(pid):
                key=i
                break
        gt_boxes = dataset[key][1] # This is the **scaled bbox**, according to shape of the npy!!
        print("scaled GT bbox:", gt_boxes)
    print("lower bound of top_k conf:", lowest_conf)
    print("bbox:", boxes)
    utils_hsz.AnimationViewer(img, bbox=[box for box, _ in boxes], verbose=False)

def plot_img_with_bbox(pid = "10755333"):
    dataset = LungDataset.load("lung_dataset_20201215.pkl")
    dataset.get_data(dataset.pids, name="hu+norm_128x128x128.npy")
    for i, datum in enumerate(dataset.data):
        if datum[2]==str(pid):
            key=i
            break
    img, gt_boxes, _ = dataset[key]
    boxes = [box[:6] for box in gt_boxes]
    img = img.squeeze(-1).numpy().astype(np.float32)
    utils_hsz.AnimationViewer(img, bbox=boxes)

if __name__ == "__main__":
    #pid = "1926851"
    npy_name = "hu+norm_256x256x256.npy"
    exp_name = "train_256_256_256_1"
    top_k = 3
    #npy_dir_path = pjoin("checkpoint", exp_name, "evaluate")
    npy_dir_path = pjoin("preidction", exp_name, "425", "evaluate")
    pids = os.listdir(npy_dir_path)
    #pids = [""]
    for fname in pids:
        pid = fname.split("_test")[0]
        print("pid:", pid)
        fast_evaluate(npy_dir_path, pid, npy_name, exp_name, top_k, check_gt=True)

