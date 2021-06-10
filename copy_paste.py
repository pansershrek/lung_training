from PIL import Image, ImageDraw #version 6.1.0
import PIL
from matplotlib.pyplot import jet #version 1.2.0
import torch
from torch.utils.data import Dataset, DataLoader
import os
#import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
import pickle
#from IPython.display import display
from tqdm import tqdm, trange
import warnings
import pandas as pd
from copy import deepcopy

from utils_hsz import AnimationViewer
import utils_hsz
import utils_ccy
from stacking_z_slices import stacking1D_average
from global_variable import NPY_SAVED_PATH, CURRENT_DATASET_PKL_PATH, EXTRA_FP_EXCEL_PATH
from random_crop import random_crop_3D

def copy_paste_3D(img_1, bbox_1, img_2, bbox_2, target_input_shape, also_crop_bbox_1=None, do_blurring=False, device="cpu"):
    """
    Paste bbox_2 of img_2 on the center of bbox_1 of img_1
    @Argument:
        img_1: 3D tensor ZYX
        bbox_1: np.array of shape (6,) or (8,) [Note that the box was FP]
        img_2: 3D tensor ZYX
        bbox_2: np.array of shape (6,) or (8,) [TP bbox]
        target_input_shape: target input shape of your model (used for random_crop_3D)
        also_crop_bbox_1: np.array of shape (k,8) or None, also bound those bbox on img_1
        do_blurring: blurring the edge of the pasted bbox
    """
    bbox_1, bbox_2 = np.array(bbox_1, dtype=np.float32), np.array(bbox_2, dtype=np.float32)


    # check if the pasted bbox will be cutoff
    center_1 = (bbox_1[3:6] + bbox_1[:3])/2 # (3,)
    bbox_2_dhw = bbox_2[3:6] - bbox_2[:3]
    #print("bbox2_2_dhw", bbox_2_dhw)
    upper_left = center_1-bbox_2_dhw/2
    bbox_pasted = np.concatenate([upper_left, upper_left+bbox_2_dhw], axis=-1) # (3,)+(3,) -> (6,)
    bbox_trimmed = bbox_pasted.copy()
    bbox_trimmed[:3] = np.maximum(bbox_trimmed[:3],0) # cutoff bbox outside img boundary
    bbox_trimmed[3:6] = np.minimum(bbox_trimmed[3:6], img_1.shape) # cutoff bbox outside img boundary
    iou = utils_ccy.iou_3D(bbox_trimmed, bbox_pasted)
    if iou<0.5: # copy paste FAILED
        return False
    # copying
    diff = bbox_pasted - bbox_trimmed
    bbox_trimmed_coor2 = bbox_2[:6] - diff
    bbox_trimmed = bbox_trimmed.astype(np.int64)
    bbox_trimmed_coor2 = bbox_trimmed_coor2.astype(np.int64)

    # +1/-1 adjusting
    shape1 = bbox_trimmed[3:6] - bbox_trimmed[:3]
    shape2 = bbox_trimmed_coor2[3:6] - bbox_trimmed_coor2[:3] 
    diff2 = shape2 - shape1
    assert diff2.max() < 2
    bbox_trimmed_coor2[3:6] -= diff2
    shape2 = bbox_trimmed_coor2[3:6] - bbox_trimmed_coor2[:3]
    assert np.array_equal(shape1, shape2)
    z1,y1,x1,z2,y2,x2 = bbox_trimmed # coordinate at img1
    cz1,cy1,cx1,cz2,cy2,cx2 = bbox_trimmed_coor2 # coordinate at img2

    tp_crop = img_2[cz1:cz2, cy1:cy2, cx1:cx2] # tp crop

    # pasting
    #print("coor1", shape1)
    #print("coor2", shape2)
    if (1): # debug
        img_1_ori = img_1.clone()
    img_1[z1:z2, y1:y2, x1:x2] = tp_crop
    #d1, h1, w1 = z2-z1+1, y2-y1+1, x2-x1+1
    #d2, h2, w2 = tp_crop.shape
    #cz1, cy1, cx1 = int(d2/2-d1/2), int(h2/2-h1/2), int(w2/2-w1/2)
    #cz2, cy2, cx2 = int(d2/2+d1/2), int(h2/2+h1/2), int(w2/2+w1/2)
    #print("b1:", (d1,h1,w1))
    #print("b2:", (d2,h2,w2))
    #print("cbox",(cz1,cy1,cx1,cz2,cy2,cx2))
    #img_1[z1:z2+1, y1:y2+1, x1:x2+1] = tp_crop[cz1:cz2+1, cy1:cy2+1, cx1:cx2+1]  # paste
    if do_blurring:
        ...
    if (0): #debug
        print("iou =", iou)
        print("diff =", (img_1_ori-img_1).abs().sum())
        AnimationViewer(img_1_ori.cpu().numpy(), [bbox_trimmed], note="Before")
        AnimationViewer(img_1.cpu().numpy(), [bbox_trimmed], note="After")
        #1/0
        
    
    ### now, introducing normal random cropping ...
    img_1 = img_1.unsqueeze(0)
    bbox_trimmed = np.concatenate([bbox_trimmed, np.ones(2)], axis=-1) # (8,)
    bbox_trimmed = torch.tensor(bbox_trimmed, dtype=torch.float32, device=device).unsqueeze_(0) # (1,8)
    if type(also_crop_bbox_1) != type(None):
        also_crop_bbox_1 = torch.tensor(also_crop_bbox_1, dtype=torch.float32, device=device)
    cropped_img, cropped_bboxes = random_crop_3D(img_1, bbox_trimmed, min_shape=target_input_shape, max_shape=target_input_shape, 
                                                also_crop_boxes=also_crop_bbox_1, use_all_box=False)
    assert len(cropped_img)==len(cropped_bboxes)==1
    cropped_img, cropped_bbox= cropped_img[0], cropped_bboxes[0]
    cropped_img = cropped_img.squeeze(0)
    cropped_bbox = [cropped_bbox[0].tolist()] if cropped_bbox[0].ndim==1 else cropped_bbox[0].tolist()
    
    # Scaling VOI and bbox using transform
    cropped_shape = cropped_img.shape # Z,Y,X
    if cropped_shape!=target_input_shape:
        msg = "cropped_shape={}, but target_input_shape={}. Start resizing".format(cropped_shape, target_input_shape)
        #raise ValueError(msg)
        warnings.warn(msg)
        cropped_img = torch.nn.functional.interpolate(cropped_img.unsqueeze_(0).unsqueeze_(0), size=target_input_shape, mode="nearest").squeeze_(0).squeeze_(0)
        cropped_bbox = utils_ccy.scale_bbox(cropped_shape, target_input_shape, cropped_bbox) # list of boxes

    if (0): #debug
        #print("Before copy paste")
        #AnimationViewer(cropped_img.cpu().numpy(), [box[:6] for box in cropped_bbox], note="Before cp")
        #print("After copy paste + randomcrop")
        AnimationViewer(cropped_img.cpu().numpy(), [box[:6] for box in cropped_bbox], note="After copy paste + randomcrop")

    return cropped_img, cropped_bbox


def dataset_preprocessing(target_transform=(1.25,0.75,0.75), target_input_shape=(128,128,128), save=False, device="cpu",
                            excel_path=EXTRA_FP_EXCEL_PATH, make_5mm=False, make_2d5mm=False, n_copy=3):
    global Tumor, LungDataset
    from dataset import Tumor, LungDataset
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    dataset.set_lung_voi()

    ## make sure the inputs are original images
    dataset.use_random_crop = False # avoid cropped img

    ## if want to make fake 1.25mm crops from 5mm lung_voi
    if make_5mm:
        assert not make_2d5mm, "Either 5mm or 2.5mm are valid, but not both at the same time"
        dataset.set_5mm(use_5mm=True, load_5mm_pkl="fast_test_max_5.0x0.75x0.75.pkl")
        dataset.set_batch_1_eval(batch_1_eval=True, equal_spacing=[1.25,0.75,0.75])
    elif make_2d5mm:
        assert not make_5mm, "Either 5mm or 2.5mm are valid, but not both at the same time"
        dataset.set_2d5mm(use_2d5mm=True, load_2d5mm_pkl="fast_test_max_2.5x0.75x0.75.pkl")
        dataset.set_batch_1_eval(batch_1_eval=True, equal_spacing=[1.25,0.75,0.75])
    assert all([npy_name==None for npy_name, _, _ in dataset.data]) # avoid loading npy

    #target_transform_text = "x".join(str(i) for i in target_transform)
    #target_input_shape_text = "x".join(str(i) for i in target_input_shape)
    device = torch.device(device)
    target_transform_text = "x".join(str(i) for i in target_transform)
    target_input_shape_text = "x".join(str(i) for i in target_input_shape)
    #if (1): #快進data (less I/O)
    #    pass
        #dataset.data = dataset.data[95+315+17:] #the place where error had occurred before ...
        ###TODO: pid = 20732541 (idx=95+315), seems to have unreasonable transform!! delete data??

        # viewing
        #dataset.data = dataset.data[19+27+21+31+156+2+33+21+215+86+32:]
        #dataset.get_data(["21678302", "6993538"])

        # processing
        #dataset.data = dataset.data[:] #[64:]
        #dataset.data = dataset.data[328:]
        
    df = pd.read_excel(excel_path, sheet_name="Sheet1", converters={'pid':str,'bbox':str, 'isFP':int})

    # use_only_rows capable of copy_paste
    df = df[df["isFP"]==1]

    pids = []
    tmp = set()
    for pid in dataset.pids:
        if pid not in tmp:
            pids.append(pid)
            tmp.add(pid)
    del tmp
    
    if (1): #快進data
        pids = pids[23:]

    invalid_pids = []

    for pid in tqdm(pids, total=len(pids)):
        rows = df[ (df["pid"]==pid) ]
        if len(rows)==0:
            invalid_pids.append(pid)
            msg = "No valid row for pid={}".format(pid)
            warnings.warn(msg)
            continue
        row = rows.sample(n=1) # random choose 1 row; type==DataFrame

        fp_bbox = eval(row.iloc[0]["bbox"])
        fp_bbox_rname = row.iloc[0].name # row name used if pd only
        dataset.get_data([pid])
        assert len(dataset) == 1, "Should only contain 1 big voi img"
        img, bboxes, pid2 = dataset[0]
        assert pid==pid2

        if make_2d5mm or make_5mm:
            transform = [1.25, 0.75, 0.75]
        else:
            dcm_reader = dataset.tumors[ dataset.pid_to_excel_r_relation[pid][0] ].dcm_reader
            transform = [dcm_reader.SliceThickness] + list(dcm_reader.PixelSpacing[::-1])


        also_crop_boxes = None
        #print("pid:",pid)
        also_crop_boxes = bboxes
        #also_crop_boxes[:,6] = 2
        also_crop_boxes = also_crop_boxes.tolist()
        bboxes = [eval(row["bbox"]) for rname, row in rows.iterrows() if rname!=fp_bbox_rname]
        cls_labels = [int(row["isFP"]==0) for rname, row in rows.iterrows() if rname!=fp_bbox_rname] # isfp==0 -> 1 else -> 0
        bboxes = [ box+[cl,1] for box, cl in zip(bboxes, cls_labels) ]

        tp_bbox = random.choice(also_crop_boxes)
        also_crop_boxes = also_crop_boxes + bboxes
        if (0): #debug
            print("bbox1", bboxes[0])
            print("bbox2", also_crop_boxes[0])
            AnimationViewer(img.squeeze(-1).cpu().numpy(), [box[:6] for box in also_crop_boxes], note="watch GT")
            continue
        
        

        img = img.squeeze(-1)
        
        #print("ori img shape", img.shape)
        #norm_img = utils_hsz.normalize(img)
        #AnimationViewer(img.numpy(), bbox=[box[:-2] for box in bboxes], note=f"{pid} Original")
        #img = img.unsqueeze(0).to(device) # -> C=1,Z,Y,X
        img = img.to(device)
        #print("random crop transform: {} -> {}".format(transform, target_transform))
        #bboxes = torch.tensor( [box[:-2] for box in bboxes], dtype=torch.float32, device=device )
        #out = random_crop_preprocessing(img, [box for box in bboxes], transform, target_transform, target_input_shape, n_copy, use_all_box, also_crop_boxes)
        for i in range(n_copy):
            out = copy_paste_3D(img, fp_bbox, img, tp_bbox, target_input_shape, also_crop_boxes, device=device, do_blurring=False)
            if not out: # copy paste FAILED!
                invalid_pids.append(pid)
                msg = "copy paste failed for pid={}, copy={}".format(pid, i)
                warnings.warn(msg)
                break
            out_img, out_bboxes = out
            #print("out_bbox:", out_bboxes)

            if make_5mm:
                name = "copy_paste_{}_{}_fake1.25_from_5mm_max_c{}.pkl".format(target_input_shape_text, target_transform_text, i+1)    
            elif make_2d5mm:
                name = "copy_paste_{}_{}_fake1.25_from_2.5mm_max_c{}.pkl".format(target_input_shape_text, target_transform_text, i+1)   
            else:
                #name = "random_crop_{}_{}_c{}.pkl".format(target_input_shape_text, target_transform_text, j+1)
                name = "copy_paste_{}_{}_c{}.pkl".format(target_input_shape_text, target_transform_text, i+1) 
            #new_img = new_img.unsqueeze(0).cpu().float().numpy() # to float16 (half) here # (Z,Y,X) -> (1,Z,Y,X)
            out_img = out_img.cpu().numpy()
            #new_boxes = [box+[1,1] for box in new_boxes] # [z1,y1,x1,z2,y2,x2] -> [z1,y1,x1,z2,y2,x2,1,1]
            to_save = (out_img, out_bboxes)


            folder_name = os.path.join(NPY_SAVED_PATH, str(pid))
            name = os.path.join(folder_name, name)
            if save: # save
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name, exist_ok=True)
                with open(name, "wb") as f:
                    pickle.dump(to_save, f)
                    #print("save to", name)
            else:
                if (0):
                    AnimationViewer(out_img, [box[:6] for box in out_bboxes], note=pid, draw_face=False)
                    print("Fake saving to", name)
    if save:
        invalid_pids_text = "\n".join(invalid_pids)
        with open("D:/CH/LungDetection/copy_paste_invalid_pids.txt", "w") as f:
            f.write(invalid_pids_text)




if __name__ == "__main__":
    device = "cpu"
    dataset_preprocessing(save=True,   # 5mm
                            device=device,
                            make_5mm=True,
                            make_2d5mm=False,
                            n_copy=3,)

    dataset_preprocessing(save=True,  # 2.5mm
                            device=device,
                            make_5mm=False,
                            make_2d5mm=True,
                            n_copy=3,)

    dataset_preprocessing(save=True,  # 1.25mm
                            device=device,
                            make_5mm=False,
                            make_2d5mm=False,
                            n_copy=3,)