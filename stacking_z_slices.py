import torch
import numpy as np

from utils_ccy import scale_bbox
from global_variable import CURRENT_DATASET_PKL_PATH
from utils_hsz import AnimationViewer

def stacking1D_average(img, axis, ori_spacing, target_spacing, ori_bbox=None, stack_func="mean"):
    """
    Stacking 3D volumes on 1 specific axis using "average"
    當target_spacing/ori_spacing不整除時，取每int(target_spacing/ori_spacing)做平均
    @Argument
        img: np.ndarray of shape (z,y,x)
        axis: int; the axis to stack
        ori_spacing: tuple of float; original spacing of the img
        target_spacing: tuple of float; target spacing of the img 
        ori_bbox: bbox of form [[z1,y1,x1,z2,y2,x2], [...]]
        stack_func: str, mean | max
    @Return
        stacked_img: np.ndarray
        new_spacing: (float, float, float); pixel spacing of stacked_img
        stacked_bbox: (if ori_bbox != None) bbox of form [[z1,y1,x1,z2,y2,x2,...], [z1,y1,x1,z2,y2,x2,...]]
    @Note
        1. assert target_spacing >= ori_spacing at **axis**
        2. assert target_spacing == ori_spacing at other dimensions except **axis**
    """
    assert target_spacing[axis] >= ori_spacing[axis], "target_spacing: {} should not be less than ori_spacing: {} at axis {}".format(target_spacing, ori_spacing, axis)
    tmp = list(target_spacing)
    tmp[axis] = ori_spacing[axis]
    assert tmp == list(ori_spacing), "Assert target_spacing == ori_spacing except at *axis*"
    original_shape = img.shape
    num_average_slice = int(target_spacing[axis]/ori_spacing[axis])  # how many slices to compute moving average for one target slice
    #print("Using num_average_slice: {}".format(num_average_slice)) #ok
    ori_length = original_shape[axis]
    target_length = ori_length / num_average_slice
    if int(target_length)!=target_length:
        target_length = int(target_length) + 1
    else:
        target_length = int(target_length)
    target_shape = list(original_shape)
    target_shape[axis] = target_length

    stacked_img = []
    slice_idx = 0
    for _ in range(target_length):
        # first, find range in ori img to average for each new slice
        if slice_idx + num_average_slice > ori_length:
            idxes = range(slice_idx, ori_length)
        else:
            idxes = range(slice_idx, slice_idx + num_average_slice)
        slices_to_stack = img.take(indices=idxes, axis=axis) # e.g. shape: (num_average_slice, h, w)
        slice_idx += num_average_slice
        # second, apply mean on each new slice
        if stack_func == "mean":
            s = slices_to_stack.mean(axis=axis) # e,g, shape: (h, w)
        elif stack_func == "max":
            s = slices_to_stack.max(axis=axis)
        else:
            raise TypeError(f"Invalid stack_func: '{stack_func}'. Expect mean|max.")
        stacked_img.append(s)
    # finally, stack all those new slices to form the new image
    stacked_img = np.stack(stacked_img, axis=axis) # e.g. shape: (target_length, h, w)
    assert list(stacked_img.shape) == target_shape, "Alg error: stacked img has shape {}, while target_shape is {}".format(stacked_img.shape, target_shape)
    new_spacing = list(ori_spacing)
    new_spacing[axis] *= num_average_slice

    if type(ori_bbox)!=type(None):
        scale_factor = [1,] * 3 # 3D 
        scale_factor[axis] = num_average_slice
        stacked_bbox_raw = scale_bbox(scale_factor, (1,1,1), ori_bbox)
        #trim bbox if is not within img
        stacked_bbox = []
        for box in stacked_bbox_raw:
            z1, y1, x1, z2, y2, x2 = box[:6]
            z1, y1, x1 = max(0,z1), max(0,y1), max(0,x1)
            z2, y2, x2 = min(z2,target_shape[0]-1), min(y2,target_shape[1]-1), min(x2,target_shape[2]-1)
            box_trimmed = [z1,y1,x1,z2,y2,x2] + box[6:]
            stacked_bbox.append(box_trimmed)
        return stacked_img, new_spacing, stacked_bbox
    else:
        return stacked_img, new_spacing


def _test_average(): # ok, test passed
    dataset_path = CURRENT_DATASET_PKL_PATH
    dataset = LungDataset.load(dataset_path)
    dataset.get_data(dataset.pids)
    dataset.set_lung_voi(True)

    # get pids of different z-thickness
    repeated=set()
    all_thickness = set()
    useful_pids = {}
    for pid in dataset.pids:
        if pid in repeated:
            continue
        repeated.add(pid)
        tumor = dataset.tumors[ dataset.pid_to_excel_r_relation[pid][0] ]
        thickness = tumor.dcm_reader.SliceThickness
        if thickness not in all_thickness:
            all_thickness.add(thickness)
            useful_pids[pid] = thickness

    # averaging slices
    pids = list(useful_pids.keys())
    print("Testing pids:", pids)
    dataset.get_data(pids)
    for img, bbox, pid in dataset:
        thickness = useful_pids[pid]
        img = img.squeeze_(-1).numpy()
        bbox = bbox.tolist()
        tumor = dataset.tumors[dataset.pid_to_excel_r_relation[pid][0]]
        ori_spacing = tumor.dcm_reader.transform
        target_spacing = list(ori_spacing)
        target_spacing[0] = 5.0
        print(f"thickness: {thickness}, pid: {pid}, ori_img_shape: {img.shape}, ori_bbox: {bbox}, ori_spacing: {ori_spacing}")
        new_img, new_spacing, new_bbox = stacking1D_average(img, 0, ori_spacing, target_spacing, ori_bbox=bbox)
        print(f"thickness: {thickness}, pid: {pid}, new_img_shape: {new_img.shape}, new_bbox: {new_bbox}, new_spacing: {new_spacing}")

        AnimationViewer(img, [box[:6] for box in bbox], note=f"Original-- thickness: {thickness}, pid: {pid}")
        AnimationViewer(new_img, [box[:6] for box in new_bbox], note=f"Stacked-- thickness: {thickness}, pid: {pid}")


if __name__ == "__main__":
    from dataset import Tumor, LungDataset
    _test_average()









    