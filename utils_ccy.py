import numpy as np
import scipy.ndimage
from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import torch
from torch import nn
import warnings
import json
from collections import deque


class LRUCache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = deque()
        self.hash_map = dict()

    def is_queue_full(self):
        return len(self.queue) == self.cache_size

    def set(self, key, value):
        if self.cache_size in [0, None]:
            return
        if key not in self.hash_map:
            if self.is_queue_full():
                pop_key = self.queue.pop()
                self.hash_map.pop(pop_key)
                self.queue.appendleft(key)
                self.hash_map[key] = value
            else:
                self.queue.appendleft(key)
                self.hash_map[key] = value

    def get(self, key):
        if key not in self.hash_map:
            return -1, False
        else:
            self.queue.remove(key)
            self.queue.appendleft(key)
            return self.hash_map[key], True
    
    def clear(self):
        self.queue = deque()
        self.hash_map = dict()
        

def get_date_str(date=datetime.today()):
        return "{}{:02}{:02}".format(date.year, date.month, date.day)

def scale_bbox(original_shape, target_shape, bboxs_ori):
    """
    Return the corresponding bbox on target shape 
    bboxs_ori format: list of [z1,y1,x1,z2,y2,x2, ...] 
    shape format: (z,y,x)
    """
    if original_shape == target_shape:
        return bboxs_ori
    else:
        z_ori, y_ori, x_ori = original_shape
        z_new, y_new, x_new = target_shape
        z_scale, y_scale, x_scale = z_new/z_ori, y_new/y_ori, x_new/x_ori
        bboxs_new = []
        for bbox_ori in bboxs_ori:
            z1,y1,x1,z2,y2,x2 = bbox_ori[:6]
            bbox_new = [round(z1*z_scale), round(y1*y_scale), round(x1*x_scale), round(z2*z_scale), round(y2*y_scale), round(x2*x_scale)]
            bbox_new = bbox_new + bbox_ori[6:]
            bboxs_new.append(bbox_new)
        return bboxs_new
    
def iou_3D(boxes, target_box): #zyxzyx format
    """
    boxes: [[z,y,x,z,y,x], [z,y,x,z,y,x], ...]
    target_bbox: [z,y,x,z,y,x]
    """
    assert type(boxes) in [list, np.ndarray]
    if type(boxes[0])!=np.ndarray: # one box only
        boxes = [boxes]
    box2 = np.array(target_box)
    tz1, ty1, tx1, tz2, ty2, tx2 = box2
    box2_area = (tz2-tz1) * (ty2-ty1) * (tx2-tx1) # for not interger input, it should not +1
    IOUs = []
    for box1 in boxes:
        box1 = np.array(box1)
        z1, y1, x1, z2, y2, x2 = box1
        box1_area = (z2-z1) * (y2-y1) * (x2-x1)
        iz1, iy1, ix1, _, _, _ = np.maximum(box1, box2)
        _, _, _, iz2, iy2, ix2 = np.minimum(box1, box2)
        inter_section = np.maximum([iz2-iz1, iy2-iy1, ix2-ix1], 0.0)
        inter_area = np.prod(inter_section)
        union_area = box1_area + box2_area - inter_area
        IOU = 1.0 * inter_area / union_area if union_area!=0 else 0.0 # nan -> 0
        IOUs.append(IOU)
    if len(IOUs)==1:
        IOUs=IOUs[0]
    return IOUs



def crop_voi(arr, voi, extend=[2,2,2]):
    """
    arr: 3D np.array
    extend: a list indicate how many voxels be extended on each axis 順序:(x,y,z)
    """
    x1, x2 = voi["x"]
    y1, y2 = voi["y"]
    z1, z2 = voi["z"]
    z1, z2 = z1-1, z2-1 # np.array index starts from 0. but voi z-slice index starts from 1
    assert x2>=x1 and y2>=y1 and z2>=z1, f"Invalid voi: {voi} detected while cropping"
    ex, ey, ez = extend
    zmax, ymax, xmax = arr.shape #原始arr: (z,y,x)順序
    new = arr.copy()

    ## Avoid out-of-range
    x1 = x1-ex if x1-ex>=0 else 0
    y1 = y1-ey if y1-ey>=0 else 0
    z1 = z1-ez if z1-ez>=0 else 0
    
    x2 = x2+ex+1 if x2+ex+1<=xmax else xmax
    y2 = y2+ey+1 if y2+ey+1<=ymax else ymax
    z2 = z2+ez+1 if z2+ez+1<=zmax else zmax
    
    #print(f"z: {z1}:{z2}, x: {x1}:{x2}, y: {y1}:{y2}")
    cropped = new[z1:z2, y1:y2, x1:x2]
    return cropped # shape 順序: (z,y,x)
    



    
def linear_normalization(arr, newMin=-1, newMax=1, cutoffMax=None, cutoffMin=None, dtype=np.float32):
    """
    Normalize a volume/image to [newMin, newMax] linearly.
    
    If cutoffMax/Min is not None, the value higher/lower than the threshold will be clipped to the edge of cutoffMax/Min.
    The algorithm will perform clipping (if needed) **before** normalization.
    
    See also: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    
    Note that it will convert the volume/image to dtype:float32 by default
    Also note that making a image grayscale in [-1,1] is a little bit different from making it zero centering, and has nothing to do with dataset mean/std

    To do zero-centering and also make value in [-1,1], try the following snippet:
        arr = linear_normalization(arr, newMin=0, newMax=1)
        arr = arr - arr.mean()
    """
    assert newMin<=newMax, "Max should be bigger or equal to Min"
    arr2 = arr.astype(dtype)
    if not (cutoffMax==cutoffMin==None):
        arr2.clip(cutoffMin, cutoffMax, out=arr2)
    oldMin, oldMax = arr2.min(), arr2.max()  
    arr2 = (arr2-oldMin) * (newMax-newMin)/(oldMax-oldMin) + newMin
    new = arr2
    return new

def resize(arr, output_shape, cval=0, silent=False):
    """    
    Deprecated, please use utils.resize_and_pad or utils.resize_without_pad
    Resize array to output_shape, padded with cval
    The non-padded part will always locate at the top-left corner of the output array
    """
    if not silent:
        warnings.warn(f"resize is deprecated, please use utils.resize_and_pad or utils.resize_without_pad")
    ## First, resize while preserving axis-ratio
    init_shape = np.array(arr.shape)
    output_shape = np.array(output_shape)
    assert init_shape.shape == output_shape.shape, f"Dimension mismatch: {init_shape} and {output_shape}"
    dim = len(arr.shape)
    resize_factor = (output_shape/init_shape).min() # use min, e.g 10 20 30 -> 100 200 200, the factor should be 200/30
    resized_arr = scipy.ndimage.interpolation.zoom(arr, resize_factor, mode='nearest')
    
    ## Pad to output_shape
    '''
    #shape_diff = output_shape - np.array(resized_arr.shape) # resize過的array，形狀與目標的差距
    #assert (shape_diff>=0).all()
    #print("shape diff:", shape_diff.astype(list))
    #print("before pad:", resized_arr, resized_arr.shape)
    '''
    new_arr = np.empty(output_shape, dtype=resized_arr.dtype)
    new_arr.fill(cval) # make an "all 0" array of desired shape
    # This part is ok, but it is less intuitive
    for i in np.ndindex(resized_arr.shape) : # iterating array and fill in
        new_arr[i] = resized_arr[i] # always fill in upper-left corner
    
    return new_arr

def resize_without_pad(arr, output_shape, mode="trilinear", device="cpu"):
    """
    Do only resizing using torch.nn.functional.interpolate
    """
    input_shape = arr.shape
    tensor = torch.tensor(arr, device=device)
    tensor = tensor.unsqueeze_(0).unsqueeze_(0)
    #print("ori:", tensor.shape, "; output_shape:", output_shape)
    resized = torch.nn.functional.interpolate(tensor, size=output_shape, mode=mode)
    resized = resized.squeeze_(0).squeeze_(0).cpu().numpy()
    return resized

def resize_and_pad(arr, output_shape, cval=0, resize_before_pad=True, mode="center"):
    """
    Resize (if needed) and then pad the *cloned* array to the desired output shape.

    Argument:
        arr: np.ndarray,
            the input array to (resize and) pad
        output_shape: tuple of integers,
            the desired output shape
        cval: int or float,
            the constant_values used for padding; default 0
        resize_before_pad: bool;
            if True, the input array will first resized to the closest shape to the output_shape while preserving (most of the) aspect ratio of the input before padding
        mode: str,
            valid modes: "center"|"corner"
            "center": the original/resized array will be centered in the padded array
            "corner": the original/resized array will located at the top-left corner of the padded array
        
    """
    Valid_modes = ["center", "corner"]
    assert mode in Valid_modes, f"invalid mode '{mode}'"
    input_shape = arr.shape
    dim = len(input_shape)
    assert len(output_shape) == dim, f"Dimension of the shape mismatched between input {input_shape} and output {output_shape}"
    input_shape, output_shape = np.array(input_shape), np.array(output_shape)
    shape_difference = output_shape - input_shape
    if (not (shape_difference >= 0).all()) and (not resize_before_pad): # must resize !
        msg = f"Force resize the array with shape {input_shape} to fit output shape {output_shape}"
        warnings.warn(msg)
        resize_before_pad = True
    if resize_before_pad:
        #resize arr first
        resize_factor = (output_shape/input_shape).min() # use min, e.g 10 20 30 -> 100 200 200, the factor should be 200/30
        arr = scipy.ndimage.interpolation.zoom(arr, resize_factor, mode='nearest')
        input_shape = arr.shape
        #print("Before pad", input_shape)
    shape_difference = output_shape - input_shape
    assert (shape_difference >= 0).all(), f"Logical Error: shape_difference should be non-negative, not {shape_difference}"
    pad_tuples=[]
    for i in range(dim):
        difference = shape_difference[i]
        rem = difference%2 # 1 for odd, and 0 for even
        if mode=="center":
            half_difference = int(difference/2)
            pad_tuple = (half_difference+rem, half_difference)
        elif mode=="corner":
            pad_tuple = (0, difference)
        pad_tuples.append(pad_tuple)
    pad_tuples = tuple(pad_tuples)
    padded_array = np.pad(arr, pad_tuples, constant_values=cval)
    assert padded_array.shape == tuple(output_shape), f"Logical Error: padded array has shape {padded_array.shape} while desired shape is {tuple(output_shape)}"
    return padded_array

def classification_statistics(dataset, y_pred:dict, binarize_threshold=0.5, n_class=3, return_prediction=False, target=[1,1,1]):
    """
    For 1 epoch usage only
    y_pred: {dict} -> desc -> excel_r -> (er,pr,her2)
    """ 
    if type(binarize_threshold) not in (list, tuple):
        binarize_threshold = [binarize_threshold]*n_class
    statistics={}
    #prediction={} 
    for description in y_pred:
        tp = np.zeros(n_class).astype(int) # (n_class,)
        fp, tn, fn = tp.copy(), tp.copy(), tp.copy()
        #prediction[description]={}
        #if len(y_pred[description]) == 0:
        #    continue
        for excel_r in y_pred[description]:
            labels = dataset.tumors[int(excel_r)].labels
            preds = list(y_pred[description][excel_r]) # (er,pr,her2)
            for n in range(n_class):
                if not target[n]: # skip this genotype
                    continue
                pred = preds.pop(0)
                label = labels[n]
                pred = 1 if pred>=binarize_threshold[n] else 0
                if label:
                    if pred:
                        tp[n]+=1
                    else:
                        fn[n]+=1
                else:
                    if pred:
                        fp[n]+=1
                    else:
                        tn[n]+=1
            #preds = tuple(map(lambda p: 1 if pred>=binarize_threshold else 0, preds))
            #preds = preds
            #prediction[description][excel_r] = preds
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        specificity = tn/(tn+fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        f1_score = 2*recall*precision/(recall+precision)
        statistics[description] = {"recall":recall.tolist(), "precision":precision.tolist(), "specificity":specificity.tolist(), "accuracy":accuracy.tolist(), "f1":f1_score.tolist(), "tp":tp.tolist(), "fp":fp.tolist(), "tn":tn.tolist(), "fn":fn.tolist()}
        #print(f"*****{description}*****")
        #print("tp", tp)
        #print("fp", fp)
        #print("fn", fn)
        #print("tn", tn)
    #print("statistics", statistics)
    #raise EOFError
    #return (statistics, prediction) if return_prediction else statistics
    return (statistics, y_pred) if return_prediction else statistics

def plot_roc_multiclass(y_true_tuples, y_pred_tuples, desc="", save_path="plots", plot=True, save_plot=True, target=[1,1,1]):
    LUT={0:"ER", 1:"PR", 2:"HER2"}
    n_class = len(target)
    roc_aucs = []
    i_shift=0
    for i in range(n_class):
        if not target[i]: #skip the genotype
            roc_aucs.append(float("nan"))
            i_shift -= 1
            continue
        y_pred = [preds[i+i_shift] for preds in y_pred_tuples]
        y_true = [trues[i] for trues in y_true_tuples]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        if plot:
            lw=2
            plt.plot(fpr, tpr,
                lw=lw, label=f'ROC curve {LUT[i]} (area = {roc_auc:0.4f})')
            #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if plot:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC: {desc}')
        plt.legend(loc="lower right")
        if save_plot:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(f"{save_path}/roc_{desc}.png")
        #plt.show()
        plt.close()
    return tuple(roc_aucs)

def plot_one_curve(x=None, y=None, prefix="", desc="", save_path="plots", save_plot=True, xlabel="epoch", ylabel="value", clean=True, legend_label=None, legend_loc="lower right", ylim=None):
    if y==None:
        raise TypeError("You should at least give argument 'y'")
    elif x==None:
        x = range(1, len(y)+1)
    if hasattr(y[0], "__iter__"):
        n_class = len(y[0])
        if legend_label==None: legend_label=[None]*n_class
        for i in range(n_class):
            plt.plot(x,[y_sub[i] for y_sub in y], lw=2, label=legend_label[i])
    else:
        plt.plot(x, y, lw=2, label=legend_label)
    plt.title(prefix+" "+desc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    if legend_label!=None:
        plt.legend(loc=legend_loc)
    if save_plot:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if clean:
            plt.savefig(f"{save_path}/{prefix}_{desc}.png")
            #plt.show()
            plt.close() #if False, no clean --> i.e. can stack curves on same fig
        else:
            pass
 


def scoring_aucs(list_of_roc_aucs, n_class=3):
    """
    Argument:
        list_of_roc_aucs: A list containing tuples of auc [(auc_class1, auc_class2, ...), (auc_class1, auc_class2, ...) ...]
        n_class: int
    Return:
        best_index: list of length n_class+1, where last index indicates setting with the highest mean auc
        best_aucs: list of length n_class+1, where last auc indicates the auc of the setting with the highest mean auc
    """
    best_index = [None]*(n_class+1)
    best_aucs = [float("-inf")]*(n_class+1)
    #print("list of rocaucs", list_of_roc_aucs)
    for i, roc_aucs in enumerate(list_of_roc_aucs):
        if n_class==1 and (not hasattr(roc_aucs, "__iter__")):
            roc_aucs = [roc_aucs]
        mean = sum(roc_aucs)/n_class
        if mean > best_aucs[n_class]:
            best_aucs[n_class] = mean
            best_index[n_class] = i
        for ci, roc_auc in enumerate(roc_aucs):
            if roc_auc > best_aucs[ci]:
                best_aucs[ci] = roc_auc
                best_index[ci] = i
    return best_index, best_aucs
    
def init_weights(mode="he"):
    """
    Valid modes: xavier, he (kaiming), uniform

    **Example usage:
        model = nn.Sequential(...)
        init_fn = init_weights("uniform")
        model.apply(init_fn)
    """
    transform = torch.nn.init.xavier_uniform_ if mode.lower() == "xavier" else torch.nn.init.kaiming_uniform_ if mode.lower() in ["he", "kaiming"] else torch.nn.uniform_ if mode.lower() == "uniform" else None
    if transform==None: raise TypeError(f"Unknown mode: {mode}")
    def fn(m):
        if type(m) == nn.Linear:
            transform(m.weight)
            if m.bias != None:
                m.bias.data.fill_(0.01)
        elif type(m) in [nn.Conv3d, nn.Conv2d]:
            transform(m.weight)
            if m.bias != None:
                m.bias.data.fill_(0.01)
    return fn

def voi_registration(sub_shape, sub_voi, to_crop_shape, voi_expand=[0,0,0], silent=False):
    """
    Deprecated, use registrate_voi_using_dcm instead!
    Calculate and return the voi for the target series, based on their shape and voi on 1 series.
    """
    if not silent:
        warnings.warn("voi_registration is deprecated, use registrate_voi_using_dcm instead!")
    x1, x2 = sub_voi["x"]
    y1, y2 = sub_voi["y"]
    z1, z2 = sub_voi["z"]
    z1, z2 = z1-1, z2-1 # np.array index starts from 0. but voi z-slice index starts from 1
    assert x2>=x1 and y2>=y1 and z2>=z1, f"Invalid voi: {voi} detected while voi registration"
    ez,ey,ex = voi_expand
    z_ratio = to_crop_shape[0]/sub_shape[0] #(z,y,x)
    y_ratio = to_crop_shape[1]/sub_shape[1]
    x_ratio = to_crop_shape[2]/sub_shape[2]
    #Always ensure more information included
    new_x1, new_x2 = int(x1*x_ratio)-ex, x2*x_ratio
    new_y1, new_y2 = int(y1*y_ratio)-ey, y2*y_ratio
    new_z1, new_z2 = int(z1*z_ratio)-ez, z2*z_ratio
    new_x2 = int(new_x2)+ex if int(new_x2)==new_x2 else int(new_x2)+1+ex
    new_y2 = int(new_y2)+ey if int(new_y2)==new_y2 else int(new_y2)+1+ey
    new_z2 = int(new_z2)+ez if int(new_z2)==new_z2 else int(new_z2)+1+ez
    
    if new_x1<0: new_x1=0
    if new_y1<0: new_y1=0
    if new_z1<0: new_z1=0
    if new_x2 >= to_crop_shape[2]: new_x2=to_crop_shape[2]-1
    if new_y2 >= to_crop_shape[1]: new_y2=to_crop_shape[1]-1
    if new_z2 >= to_crop_shape[0]: new_z2=to_crop_shape[0]-1
    new_z1, new_z2 = new_z1+1, new_z2+1  # np.array index starts from 0. but voi z-slice index starts from 1
    new_voi = {"x":(new_x1, new_x2), "y":(new_y1, new_y2), "z":(new_z1, new_z2)}
    return new_voi

def calculate_mean_statistics_from_predictions(predictions_folder, epochs_per_fold=[1,1,1,1,1], thresholds=[0.5,0.5,0.5], n_class=3, dataset=None ,plot_roc=False, desc="", save_plot_path=None):
    valid_test_split=None
    num_k_folds = len(epochs_per_fold)
    valid_preds = {}
    test_preds = {}

    if dataset==None: #use a sample dataset to get ground truth
        global MRIDataset, Tumor
        from dataset import MRIDataset, Tumor
        dataset = MRIDataset.load("mri_dataset_20201028.pkl")

    if type(thresholds) in [int,float]:
        thresholds = [thresholds]*n_class
    elif len(thresholds)!=n_class:
        raise TypeError(f"thresolds has length {len(thresholds)}, while n_class is {n_class}")
    if plot_roc and type(save_plot_path)!=str: raise TypeError("You should specify where to save plots!")
        
    for fold, epoch in enumerate(epochs_per_fold):
        filter_fn = lambda file: file.startswith(f"predictions_e{epoch}_f{fold}_") 
        fnames = list(filter(filter_fn, os.listdir(predictions_folder)))
        assert len(fnames) == 1, f"Logical Error: not unique file: {fnames}"
        fname = pjoin(predictions_folder, fnames[0])
        with open(fname, "r") as f:
            preds = json.load(f)
        if valid_test_split==None:
            valid_test_split = True if len(preds["Validation"])!=0 else False
        valid_preds = {**valid_preds, **preds["Validation"]}
        test_preds = {**test_preds, **preds["Testing"]}
    rows = list(test_preds.keys())

    out={}
    for desc, preds_dict in {"Validation":valid_preds, "Testing":test_preds}.items():
        if len(preds_dict)==0: continue
        tp = np.zeros(n_class).astype(int) # (n_class,)
        fp, tn, fn = tp.copy(), tp.copy(), tp.copy()
        y_true_list = []
        y_pred_list = []
        row_list = []
        #out[desc]={}
        for excel_r in rows:
            labels = dataset.tumors[int(excel_r)].labels #(er,pr,her2)
            #int_keys = list(map(int,preds_dict))
            #print(sorted(int_keys), len(int_keys))
            preds = preds_dict[excel_r]
            y_true_list.append(labels)
            y_pred_list.append(preds)
            row_list.append(int(excel_r))
            for n in range(n_class):
                pred, label = preds[n], labels[n]
                pred = 1 if pred>=thresholds[n] else 0
                if label:
                    if pred:
                        tp[n]+=1
                    else:
                        fn[n]+=1
                else:
                    if pred:
                        fp[n]+=1
                    else:
                        tn[n]+=1
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        f1_score = 2*recall*precision/(recall+precision)
        roc_auc = plot_roc_multiclass(y_true_list, y_pred_list, desc=desc, save_path=save_plot_path, plot=plot_roc)
        out[desc] = {"recall":recall, "precision":precision, "accuracy":accuracy, "f1":f1_score, "tp":tp, "tn":tn, "fp":fp, "fn":fn, "roc_auc":roc_auc}
    return out


def registrate_voi_using_dcm(sub_voi, to_crop_name, tumor, sub_shape, voi_extend=[0,0,0]):
    x1, x2 = sub_voi["x"]
    y1, y2 = sub_voi["y"]
    z1, z2 = sub_voi["z"]
    ##z1, z2 = z1-1, z2-1 # np.array index starts from 0. but voi z-slice index starts from 1
    assert x2>=x1 and y2>=y1 and z2>=z1, f"Invalid voi: {voi} detected while voi registration"
    sub_dcm = tumor.get_dcm("sub")
    to_crop_dcm = tumor.get_dcm(to_crop_name)
    def registrate_xy(sub_dcm, to_crop_dcm, sub_voi, sub_shape, voi_extend):
        x1, x2 = sub_voi["x"]
        y1, y2 = sub_voi["y"]
        ez, ey, ex = voi_extend
        x1, x2 = max(x1-ex,0), min(x2+ex, sub_shape[2]-1)
        y1, y2 = max(y1-ey,0), min(y2+ey, sub_shape[1]-1)
        origin_x_sub, origin_y_sub, _ = sub_dcm.children[0].ImagePositionPatient
        spacing_x_sub, spacing_y_sub = sub_dcm.children[0].PixelSpacing
        # First, find sub_voi in real physical world coordinate
        rx1, rx2 = origin_x_sub+spacing_x_sub*x1, origin_x_sub+spacing_x_sub*x2
        ry1, ry2 = origin_y_sub+spacing_y_sub*y1, origin_y_sub+spacing_y_sub*y2
        # Then find target_voi
        origin_x_crop, origin_y_crop, _ = to_crop_dcm.children[0].ImagePositionPatient
        spacing_x_crop, spacing_y_crop = to_crop_dcm.children[0].PixelSpacing
        nx1, nx2 = (rx1-origin_x_crop)/spacing_x_crop, (rx2-origin_x_crop)/spacing_x_crop
        ny1, ny2 = (ry1-origin_y_crop)/spacing_y_crop, (ry2-origin_y_crop)/spacing_y_crop
        nx1 = int(nx1)
        nx2 = int(nx2)+1 if int(nx2)<nx2 else int(nx2)
        ny1 = int(ny1)
        ny2 = int(ny2)+1 if int(ny2)<ny2 else int(ny2)
        return nx1, nx2, ny1, ny2
    def registrate_z(sub_dcm, to_crop_dcm, sub_voi, tumor, sub_shape, voi_extend):
        z1, z2 = sub_voi["z"]
        ez, ey, ex = voi_extend
        z1, z2 = max(z1-ez,1), min(z2+ez, sub_shape[0])
        #assert sub_dcm.children[z1-1].InstanceNumber.real==z1, f"instance number: {sub_dcm.children[z1-1].InstanceNumber} != z1:{z1}"
        #assert sub_dcm.children[z2-1].InstanceNumber.real==z2, f"instance number: {sub_dcm.children[z2-1].InstanceNumber} != z2:{z2}"
        #SliceLocation是隨著z**遞減**的
        if sub_dcm.children[z1-1].InstanceNumber.real==z1:
            rz1 = float(sub_dcm.children[z1-1].SliceLocation.real)
        #elif tumor.excel_r==136: # sub不連續，特例處理
        #    rz1 = float(sub_dcm.children[97].SliceLocation.real)
        else:
            print(f"z1={z1} instance={sub_dcm.children[z1-1].InstanceNumber.real}")
            raise ValueError("Can't find rz1 in sub series")

        if sub_dcm.children[z2-1].InstanceNumber.real==z2:
            rz2 = float(sub_dcm.children[z2-1].SliceLocation.real)
        else:
            raise ValueError("Can't find rz2 in sub series")
        ##assert rz1 >= rz2, f"Bad order of slices, rz1={rz1}, rz2={rz2}"
        ## SliceLocation 一般都是遞減的
        sub_reverse = False if rz1>=rz2 else True #即sub series SliceLocation次序反轉
        to_crop_reverse = False if to_crop_dcm.children[0].SliceLocation.real >= to_crop_dcm.children[1].SliceLocation.real else True
        if sub_reverse:
            rz1, rz2 = rz2, rz1
        nz1 = nz2 = None
        #print("rz1 =", rz1)
        #print("rz2 =", rz2)
        def less_than(a, b, to_crop_reverse):
            if not to_crop_reverse:
                return a<b
            else:
                return a>b
        current_position = float("inf")
        for i, s in enumerate(to_crop_dcm.children):
            assert current_position==float("inf") or round(abs(current_position-s.SliceLocation),3)==round(float(s.SliceThickness.real),3), f"interval: {current_position-s.SliceLocation}, thick:{s.SliceThickness}"
            current_position = s.SliceLocation # 這個值會越來越小
            #print("curr position", current_position, i)
            if (nz1==None) and less_than(current_position, rz1, to_crop_reverse):
                nz1 = s.InstanceNumber.real
                nz1 = nz1-1 if (nz1>=2 and not to_crop_reverse) else nz1 # -1 means last slice!
                #print("assign nz1 to", nz1)
            if (nz2==None) and less_than(current_position, rz2, to_crop_reverse):
                nz2 = s.InstanceNumber.real
                nz2 = nz2-1 if (nz2>=2 and to_crop_reverse) else nz2 # -1 means last slice!
                #print("assign nz2 to", nz2)
        else: # if touch boundary
            if (nz1==None):
                nz1 = s.InstanceNumber.real # no need -1
                #print("else assign nz1 to", nz1)
            if (nz2==None):
                nz2 = s.InstanceNumber.real
                #print("else assign nz2 to", nz2)
        if (to_crop_reverse and not sub_reverse) or (sub_reverse and not to_crop_reverse): # i.e. XOR
            n = len(to_crop_dcm.children)
            nz1 = n-nz1+1
            nz2 = n-nz2+1
        if sub_reverse:
            nz1, nz2 = nz2, nz1
        if (to_crop_reverse and not sub_reverse):
            # debug
            assert False, f"Debug tumor: {tumor.access}, {tumor.excel_r}"
        return nz1, nz2 # both in real voi, not index
    #print(tumor.access, tumor.excel_r, tumor.voi)
    nx1, nx2, ny1, ny2 = registrate_xy(sub_dcm, to_crop_dcm, sub_voi, sub_shape, voi_extend)
    nz1, nz2 = registrate_z(sub_dcm, to_crop_dcm, sub_voi, tumor, sub_shape, voi_extend)
    assert nx1<=nx2 and ny1<=ny2 and nz1<=nz2
    target_voi = {"x":(nx1,nx2), "y":(ny1,ny2), "z":(nz1,nz2)}           
    #DONE: 1.draw X to see if ImagePositionPatient works, (use target_voi) --> 沒有非常好，但是比最早的直接等比例去找來的好很多
    #TODO: 2. use scipy.interpolate to get hidden value of each voxel (optional)
    return target_voi

def histogram_equalization(volume:np.ndarray):
    assert len(volume.shape)==3 , "Expect input array has 3 dimensions with shape (Z,Y,X)"
    Z,Y,X = volume.shape
    max_val, min_val = volume.max(), volume.min()
    val_range = max_val-min_val
    numel = volume.size
    #assert type(max_val)==type(min_val)==int, "need integer voxel value!, not {} or {}".format(type(max_val), type(min_val))}
    flattened = volume.flatten()
    #print("count:", list(count.values()))
    count, _ = np.histogram(volume, bins=val_range+1, range=[min_val, max_val+1])
    #plt.bar(range(len(count)), count)
    #plt.show()
    accum_count = count.cumsum()
    cdf_m = np.ma.masked_equal(accum_count,0)
    cdf_m = (cdf_m - cdf_m.min())*max_val/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint16') # for mri data: uint16
    assert len(cdf) == val_range+1, f"len(cdf)={len(cdf)}, while n_colors={val_range+1}"
    #print(f"n_color: {len(cdf)}, min_val: {min_val}, max_val: {max_val}")
    volume2 = cdf[volume-min_val]
    return volume2
    


      

def _test_mean_statistics():
    global MRIDataset, Tumor
    from dataset import MRIDataset, Tumor
    from pprint import pprint
    dataset = MRIDataset.load("mri_dataset_20201030.pkl") 
    #epochs = [170,112,156,112,123]
    #fname = r"D:/CH/MRI/trainings/mri_training_20201019/SETTINGS/resnet50/bfl_b50_balanced_sub+tirm_tra_p3_no_scheduler_pad_min/predictions"
    #epochs = [197,144,110,189,177]
    #fname = r"D:/CH/MRI/trainings/mri_training_20201019/SETTINGS/resnet50/bfl_b50_balanced_sub+tirm_tra_p3_no_scheduler_pad_-1/predictions"
    #epochs = [94,193,62,91,171]
    #fname = r"D:/CH/MRI/trainings/mri_training_20201019/SETTINGS/resnet50/bfl_b50_balanced_sub+tirm_tra_p3_no_scheduler/predictions"
    #epochs = [157,101,167,24,150]
    #fname = r"D:/CH/MRI/trainings/mri_training_20201019/SETTINGS/resnet50/b5_100x160x160_pad0/predictions" # 4 hours training, but no better
    # it seems better pad 0>min>-1
    #epochs = [87,198,161,199,68] #from f1
    #epochs = [85,198,97,168,36] #from acc
    #fname = r"D:/CH/MRI/trainings/mri_training_20201019/SETTINGS/resnext50/resnext50_b50_balanced_sub+tirm_tra_p3_no_scheduler_center/predictions"
    epochs = [190,87,59,47,68]
    fname = r"D:/CH/MRI/trainings/mri_training_20201019/SETTINGS/resnext50/testnet_se_ra_b50_balanced_sub+tirm_tra_p3_center/predictions"
    statistics = calculate_mean_statistics_from_predictions(fname, epochs, dataset=dataset)
    pprint(statistics)


        

def _test_put():
    global arr, resized
    arr = np.empty((3,3,3))
    for i in range(arr.size): #arr.size == eval("*".join(str(i) for i in arr.shape)) == 3*3*3 == 27
        arr.put(i, 1)
    resized = resize(arr, (7,7,6))
    print(arr)
    print("-"*30)
    print(resized)

    
if __name__ == "__main__":
    #_test_put()
    _test_mean_statistics()

