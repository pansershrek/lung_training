from PIL import Image, ImageDraw #version 6.1.0
import PIL
from matplotlib.pyplot import jet #version 1.2.0
import torch
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

def intersect(boxes1, boxes2, dim):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4->6) # n1 == len(boxes1)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4->6) # n2 == len(boxes2)
        dim: boxes is (n,4) if dim==2 else (n,6) if dim==3 ...
        
        # out: an array of shape (n1,n2), where each element in place "ij" indicates intersected area of box i of bboxes1 and box j of bboxes2
        Out: Intersection each of boxes1 with respect to each of boxes2, 
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_zyx =  torch.min(boxes1[:, dim:dim*2].unsqueeze(1).expand(n1, n2, dim),
                        boxes2[:, dim:dim*2].unsqueeze(0).expand(n1, n2, dim))
    
    min_zyx = torch.max(boxes1[:, :dim].unsqueeze(1).expand(n1, n2, dim),
                       boxes2[:, :dim].unsqueeze(0).expand(n1, n2, dim))

    inter = torch.clamp(max_zyx - min_zyx , min=0)  # (n1, n2, dim)
    ##print("inter", inter)
    ##return inter[:, :, 0] * inter[:, :, 1]  #(n1, n2) (2D used only)
    return inter.prod(-1) #general, torch.prod ~= np.multiply.reduce
def find_IoU(boxes1, boxes2, dim):
    '''
        Find IoU between every boxes set of boxes 
        boxes1: a tensor of dimensions (n1, 2*dim) # x1,y1,x2,y2 or z1,y1,x1,z2,y2,x2
        boxes2: a tensor of dimensions (n2, 2*dim)
        
        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of 
             dimensions (n1, n2)
        
        Formula: 
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    '''
    inter = intersect(boxes1, boxes2, dim=dim)
    ##area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # (x2-x1) * (y2-y1), shape==(n1,)
    ##area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # shape==(n2,)
    n1,n2 = len(boxes1), len(boxes2)

    """
    area_boxes1, area_boxes2 = torch.ones(n1, device=device), torch.ones(n2, device=device)
    for d in range(dim):
        area_boxes1 *= (boxes1[:,d+dim] - boxes1[:,d])
        area_boxes2 *= (boxes2[:,d+dim] - boxes2[:,d])
    """
    area_boxes1 = (boxes1[:, dim:dim*2] - boxes1[:, :dim]).prod(-1)
    area_boxes2 = (boxes2[:, dim:dim*2] - boxes2[:, :dim]).prod(-1)
    
    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter) #(n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  #(n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union

def random_crop_3D(image, boxes, min_shape=None, max_shape=None, use_all_box=False, also_crop_boxes=None):
    '''
        image:  tensor of shape (C,Z,Y,X)
        boxes: Bounding boxes, a tensor of dimensions (#objects, 6 or 8) #z1,y1,x1,z2,y2,x2
        min_shape: None or a 3D tuple indicates minimum cropped voi shape; default is original box shape
        max_shape: None or a 3D tuple indicates maximum cropped voi shape; default is 4X orignal box shape

        Out: cropped image , new boxes
    '''
    #if type(image) == PIL.Image.Image:
    #    image = F.to_tensor(image) # For 2D: (Y,X,3) -> (C=3, Y, X); (x,y order approved)
    assert image.device == boxes.device
    device = image.device
    original_d = image.size(-3) # or 1
    original_h = image.size(-2) # or 2
    original_w = image.size(-1) # or 3

    out_vois = []
    out_boxes = []

    boxes_preserve = boxes.clone()
    if type(also_crop_boxes)!=type(None):
        boxes = torch.cat([boxes, also_crop_boxes], dim=0)
    #print("boxes",boxes)
    #print("also_crop_boxes", also_crop_boxes)

    #for box in boxes:
    for i in range(len(boxes_preserve)):
        if use_all_box:
            box = boxes_preserve[i]
        else:
            box = random.choice(boxes_preserve)
        #print("Using bbox:", box)
  
        ori_z1, ori_y1, ori_x1, ori_z2, ori_y2, ori_x2 = box[:6].clone()
        ori_box_d, ori_box_h, ori_box_w = ori_z2-ori_z1, ori_y2-ori_y1, ori_x2-ori_x1
        #Center of bounding boxes
        center_bb = (box[:3] + box[3:6])/2.0 # ( (z1,y1,x1) + (z2,y2,x2) ) / 2, shape==(#obj=1, 3)
        ###print("Current box", box)
        ###print("Current center", center_bb)
        while True:
            NEED_PAD = False
            # Define cropped voi shape (1*ori_box_dwh ~ 4*ori_box_dwh)
            if min_shape==None:
                min_shape = (ori_box_d, ori_box_h, ori_box_w)
            if max_shape==None:
                max_shape = (4.0*ori_box_d, 4.0*ori_box_h, 4.0*ori_box_w)
            new_d = int(random.uniform(min_shape[0], max_shape[0]))
            new_h = int(random.uniform(min_shape[1], max_shape[1]))
            new_w = int(random.uniform(min_shape[2], max_shape[2]))      
            # Define new_z1, new_y1==top, new_x1==left
            # the second part force the cropped img contains the bbox
            #left = max( 0, center_bb[2]-new_w*0.8+1, random.uniform(ori_x1-ori_box_w*2.5, ori_x1+ori_box_w*0.4) )
            #left = max( 0, center_bb[2]-new_w*0.8+1, random.uniform(ori_x1-ori_box_w*2.5, ori_x1) )
            if (0): #bad method, bbox may be cut
                left = max( 0, center_bb[2]-new_w*0.8+1, random.uniform(ori_x1-new_w*0.7, ori_x1-new_w*0.1) )
                right = left + new_w
                top = max( 0, center_bb[1]-new_h*0.8+1, random.uniform(ori_y1-new_h*0.7, ori_y1-new_h*0.1) )
                bottom = top + new_h
                z1 = max( 0, center_bb[0]-new_d*0.8+1, random.uniform(ori_z1-new_d*0.7, ori_z1-new_d*0.1) )
                z2 = z1 + new_d  
            if (0): #ok method, but bbox may be very close to border
                left = random.uniform( max(ori_x1+ori_box_w-new_w,0), min(ori_x1, original_w-1-new_w) )
                right = left + new_w
                top = random.uniform( max(ori_y1+ori_box_h-new_h,0), min(ori_y1, original_h-1-new_h) )
                bottom = top + new_h
                z1 = random.uniform( max(ori_z1+ori_box_d-new_d,0), min(ori_z1, original_d-1-new_d) )
                z2 = z1 + new_d

            if (1): # keep bbox away from border
                away_x = 10 # unit: pixel
                away_y = 10
                away_z = 0
                x_max = original_w-1-new_w if original_w-1-new_w >=0 else 0
                y_max = original_h-1-new_h if original_h-1-new_h >=0 else 0
                z_max = original_d-1-new_d if original_d-1-new_d >=0 else 0
                left = random.uniform( max(min(ori_x1+ori_box_w-new_w+away_x, x_max), 0), min(max(ori_x1-away_x,0), x_max) )
                right = left + new_w
                top =  random.uniform( max(min(ori_y1+ori_box_h-new_h+away_y, y_max), 0), min(max(ori_y1-away_y,0), y_max) )
                bottom = top + new_h
                if new_d*2 >= ori_box_d: # if True, ensure center_bb in crop 
                    z1 = random.uniform( max(min(ori_z1+ori_box_d-new_d+away_z, z_max), 0), min(max(ori_z1-away_z,0), z_max) )
                else:
                    z1 = random.uniform( max(min(ori_z1+ori_box_d/2+away_z, z_max), 0), min(max(ori_z1+ori_box_d/2-new_d/2-away_z,0), z_max) )
                z2 = z1 + new_d
            try:
                assert 0<=left<=right<=original_w
                assert 0<=top<=bottom<=original_h
                assert 0<=z1<=z2<=original_d
            except:
                msg =  f"left: {left}, right: {right}, original_w: {original_w}\n"
                msg += f"top: {top}, bottom: {bottom}, original_h: {original_h}\n"
                msg += f"z1: {z1}, z2: {z2}, original_d: {original_d}\n"
                print(msg)
                ### the error may stem from (original_w < crop_w). i.e. need crop all + padding
                if (original_w < new_w):
                    left, right = 0, original_w-1
                    NEED_PAD = True
                if (original_h < new_h):
                    top, bottom = 0, original_h-1
                    NEED_PAD = True
                if (original_d < new_d):
                    z1, z2 = 0, original_d-1
                    NEED_PAD = True
                if not NEED_PAD: # unknown errors
                    raise 
                

            

            crop = torch.tensor([int(z1), int(top), int(left), int(z2), int(bottom), int(right)], dtype=torch.float32, device=device) # z1,y1,x1,z2,y2,x2
            #print("Cropping :", [int(z1), int(top), int(left), int(z2), int(bottom), int(right)])
            # Calculate IoU  between the crop and the bounding boxes
            overlap = find_IoU(crop.unsqueeze(0), box.unsqueeze(0) , dim=3) #(1, #objects)  # np.expand_dims ~= torch.unsqueeze
            overlap = overlap.squeeze(0)
            # If the crop bounding box doesn't has a IoU of greater than the minimum(mode), try again
            #Crop
            new_image = image[:, int(z1):int(z2), int(top):int(bottom), int(left):int(right)] #(C, new_d, new_h, new_w)
            
            #PAD if needed
            if NEED_PAD:
                warnings.warn("padding image due to 'ori_shape < crop_shape' (NEED_PAD)")
                c, d, w, h = new_image.shape
                assert (d, w, h) != (new_d, new_h, new_w)
                pad_img = torch.zeros((c,new_d,new_h,new_w), dtype=torch.float32)  
                pad_img[:,:d,:h,:w] = new_image
                new_image = pad_img
            #Center of bounding boxes
            center_in_crop = (int(z2) > center_bb[0] > int(z1)) * \
                                (int(bottom) > center_bb[1] > int(top)) * \
                                (int(right) > center_bb[2] > int(left))   #( #objects==1 )
            ###print("center_in_crop", center_in_crop)
            if center_in_crop:
                #center_in_crop = center_in_crop.unsqueeze(0)
                break
            else:
                # untight but acceptable condition
                center_in_crop_2  = (int(z2)+1 > center_bb[0] > int(z1)-1) * \
                                (int(bottom)+1 > center_bb[1] > int(top)-1) * \
                                (int(right)+1 > center_bb[2] > int(left)-1)
                if center_in_crop_2:
                    x = [int(z1), int(top), int(left), int(z2), int(bottom), int(right)]
                    warnings.warn("Border condition in random crop; the random cropped voi doesn't include the original voi center\n ori_center:{}\ncropped:{}".format(
                                    center_bb, x))
                    break
                else:
                    raise TypeError("Algorithm Error: the random cropped voi doesn't include the original voi center\n ori_center:{}\ncropped:{}".format(
                                    center_bb, [int(z1), int(top), int(left), int(z2), int(bottom), int(right)]))
            
        
        #take matching bounding box
        # new_boxes = box.unsqueeze(0) # bug: may contain 2 nodules in 1 voi
        new_boxes = boxes.clone()
        #print("After clone:", new_boxes)
        
        
        #adjust to crop (原點變換，從原圖(0,0)到原圖bbox左上角)
        new_boxes[:, :3] -= crop[:3]
        #print("After subtract crop[:3] 1:", new_boxes)

        #adjust to crop (原點變換，從原圖(0,0)到原圖bbox左上角)
        new_boxes[:, 3:6] -= crop[:3]
        #print("After subtract crop[:3] 2:", new_boxes)
        #new_boxes.squeeze_(0)
        ###print("NEW BOXES",new_boxes)
        
        #Use the box left and top corner or the crop's 
        new_boxes_ori = new_boxes.clone()
        new_boxes[:, :3] = torch.max(new_boxes[:, :3], crop[:3]-crop[:3])
        #print("after torch max:", new_boxes)
                
        new_boxes[:, 3:6] = torch.min(new_boxes[:, 3:6],crop[3:]-crop[:3])
        #print("After torch min:", new_boxes)
        

        if new_boxes.ndim != 1 : # 2 or more bbox
            _, z_lim, y_lim, x_lim = new_image.shape
            tmp = []
            tmp_ori = []
            for box, box_ori in zip(new_boxes, new_boxes_ori):
                if (0<=box[0]<box[3]<=z_lim) and (0<=box[1]<box[4]<=y_lim) and (0<=box[2]<box[5]<=x_lim): # box must be in view
                    tmp.append(box)
                    tmp_ori.append(box_ori)
            if len(tmp)!=0:
                new_boxes = torch.stack(tmp, dim=0)
                new_boxes_ori = torch.stack(tmp_ori, dim=0)
            else: # use all bbox
                pass

        # remove bbox if remain_area < 1/3
        remain_area = (new_boxes[:, 3:6] - new_boxes[:, :3]).prod(-1)
        ori_area = (new_boxes_ori[:, 3:6] - new_boxes_ori[:, :3]).prod(-1)
        mask = (remain_area/ori_area) >= 1/3
        new_boxes_m = new_boxes[mask]
        if len(new_boxes_m)>=1:
            new_boxes = new_boxes_m

        #print("new boxes ori")
        #print(new_boxes_ori)
        #print("new boxes")
        #print(new_boxes)
        
        out_vois.append(new_image)
        out_boxes.append(new_boxes)
        #print("Adding new_box", new_boxes)
        if (0): #debug
            AnimationViewer(new_image.squeeze(0).cpu().numpy(), new_boxes.tolist(), note="debug")

        if not use_all_box:
            return out_vois, out_boxes  ## if want 1 voi per image
    return out_vois, out_boxes
    #return out_vois, out_boxes  ## unreachable



def _test_inter():
    bboxes1 = np.array([[0,1,2,5,5,5],[3,3,3,5,5,5],[3,3,3,5,5,5]]) # --> 5*4*3, 2*2*2
    bboxes2 = np.array([[3,4,4,6,6,7],[3,3,3,5,5,5]]) # --> 2*2*2, 2*2*2
    bboxes1, bboxes2 = torch.tensor(bboxes1), torch.tensor(bboxes2)
    intersected = intersect(bboxes1, bboxes2, dim=3)
    print("intersected", intersected) # an array of shape (# bbox1, # bbox2), where each element in place "ij" indicates intersected area of box i of bboxes1 and box j of bboxes2
    #print("b", b)
def _test_iou():
    bboxes1 = np.array([[0,1,2,5,5,5],[3,3,3,5,5,5], [3,3,3,5,5,5]]) 
    bboxes2 = np.array([[0,0,0,6,6,6],[3,3,3,5,5,5]]) 
    bboxes1, bboxes2 = torch.tensor(bboxes1), torch.tensor(bboxes2)
    iou = find_IoU(bboxes1, bboxes2, dim=3)
    print("iou", iou)

def _test_random_crop():
    global Tumor, LungDataset
    from dataset import Tumor, LungDataset
    dataset = LungDataset.load("lung_dataset_20201229_small.pkl")
    dataset.get_data(dataset.pids, name="hu+norm_256x256x256.npy")
    #dataset.get_data(dataset.pids)

    #img, boxes, pid = dataset[1]
    #print(dataset.pids)
    while True:
        for img, boxes, pid in [dataset[2], dataset[5]]:
            print("pid:", pid)
            print("ori bbox", [box[:6] for box in boxes])
            print("ori img", img.shape)
            print(img.squeeze(-1).shape, boxes[0][:-2])
            #AnimationViewer(img.squeeze(-1).numpy(), bbox=boxes[0][:-2])
            device = torch.device("cuda")
            img = img.permute(3,0,1,2).to(device) # -> C=1,Z,Y,X
            boxes = torch.tensor( [box[:-2] for box in boxes], dtype=torch.float32, device=device)
            new_imgs, new_boxes = random_crop_3D(img, boxes, min_shape=(20,20,20), max_shape=(80,80,80))
            for new_img, new_box in zip(new_imgs, new_boxes):
                print("new bbox", new_box)
                print("new_img", new_img.shape)
                if new_box.ndim != 1: # multiple bbox
                    view_box = [box for box in new_box.cpu().numpy()]
                else:
                    view_box = new_box.cpu().numpy()
                AnimationViewer(new_img.squeeze(0).cpu().numpy(), bbox=view_box)
    

def random_crop_preprocessing(img, bboxes, transform, target_transform, target_input_shape, n_copy, use_all_box=False, also_crop_boxes=None):
    """
    Apply random_crop_3D on the img+bboxes while adjust pixel_spacing with nearest interpolation at the same time
    @Argument
        img: 3D FloatTensor of order (z,y,x)
        bboxes: a list of bboxes in the img; format: [[z1,y1,x1,z2,y2,x2], [z1,y1,x1,z2,y2,x2], ...]
        transform: a tuple indicating the original PixelSpacing (z,y,x)
        target_transform: a tuple indicating the target PixelSpacing (z,y,x)
        target_input_shape: a tuple indicating the returned img shape (z,y,x)
        n_copy: int, how many random_cropped image is needed
    @Returned
        out: a list of length 'n_copy', each of which is a tuple of form (out_img, out_bboxes)
    @Note:
        1. All the operations will proceed within img.device
    """
    device = img.device
    original_shape = img.shape
    img = img.unsqueeze(0) # -> C=1,Z,Y,X

    if use_all_box and n_copy!=1:
        warnings.warn("Ignoring n_copy, due to use_all_box=True")
        n_copy=1
    
    out = []
    for i in range(n_copy):
        fix_shape = np.array(target_input_shape) # the desired voi shape range **AFTER** PixelSpacing norm

        if (0): #crop then resize
            bboxes = torch.tensor( bboxes, dtype=torch.float32, device=device )
            fix_shape_prev = fix_shape *  target_transform / transform
            #print("fix shape for crop3D:", fix_shape_prev)
            fix_shape_prev = tuple(fix_shape_prev)
            new_imgs, new_boxes = random_crop_3D(img, bboxes, fix_shape_prev, fix_shape_prev, use_all_box, also_crop_boxes)
            assert len(new_imgs)==len(new_boxes)==1 or use_all_box

        if (1): #resize then crop
            d, h, w = img.shape[1:]
            d_new, h_new, w_new = round(d*transform[0]/target_transform[0]), round(h*transform[1]/target_transform[1]), round(w*transform[2]/target_transform[2])
            if i==0: #only interpolate once
                #print("Start interpolate, before:", img.shape)
                img = torch.nn.functional.interpolate(img.unsqueeze_(0), size=(d_new,h_new,w_new), mode="nearest").squeeze_(0)
                #print("After interpolate:", img.shape)
                if type(bboxes[0])!=list:
                    bboxes = [bbox.tolist() for bbox in bboxes]
                bboxes= utils_ccy.scale_bbox((d,h,w), (d_new,h_new,w_new), bboxes)
                if type(also_crop_boxes)!=type(None):
                    also_crop_boxes = utils_ccy.scale_bbox((d,h,w), (d_new,h_new,w_new), also_crop_boxes)
                if (0): #debug view: original + equal_spacing
                    AnimationViewer(img.squeeze(0).cpu().numpy().astype(np.float32), bbox=bboxes)
            bboxes = torch.tensor(bboxes, dtype=torch.float32, device=device )
            if type(also_crop_boxes)!=type(None):
                also_crop_boxes = torch.tensor(also_crop_boxes, dtype=torch.float32, device=device )
            new_imgs, new_boxes = random_crop_3D(img, bboxes, tuple(fix_shape), tuple(fix_shape), use_all_box, also_crop_boxes)
            assert new_imgs
            assert len(new_imgs)==len(new_boxes)==1 or use_all_box
        
        #print("shape after crop3D:", new_imgs[0].squeeze(0).shape)
        
        #print("new_boxes", new_boxes)
        if use_all_box:
            new_boxes = [t.tolist() for t in new_boxes] #(N,6)
        else:
            new_boxes = [new_boxes[0].tolist()] if new_boxes[0].ndim==1 else new_boxes[0].tolist() # tensor -> list of bboxes
        #AnimationViewer(new_imgs[0].cpu().numpy().squeeze(0), bbox=new_boxes, note=f"{pid} Before scale")

        #new_img = new_imgs[0].squeeze(0) # cropped (1,Z,Y,X) -> (Z,Y,X)
        for new_img, new_box in zip(new_imgs, new_boxes):
            new_img = new_img.squeeze(0)

            cropped_shape = new_img.shape # Z,Y,X

            # Scaling VOI and bbox using transform
            if cropped_shape!=target_input_shape:
                msg = "cropped_shape={}, but target_input_shape={}. Start resizing".format(cropped_shape, target_input_shape)
                #raise ValueError(msg)
                warnings.warn(msg)
                new_img = torch.nn.functional.interpolate(new_img.unsqueeze_(0).unsqueeze_(0), size=target_input_shape, mode="nearest").squeeze_(0).squeeze_(0)
                new_box = utils_ccy.scale_bbox(cropped_shape, target_input_shape, new_box) # list of boxes

            new_img = new_img.cpu()
            out.append( (new_img, new_box) )


            if (0): # debug view: cropped
                view_box = new_box
                AnimationViewer(new_img.squeeze(0).numpy().astype(np.float32), bbox=view_box, note=f"Crop{i+1}")
                #1/0

            if (0): # save as fp16 (not used code)
                name = f"random_crop_b{i+1}.pkl"
                name = os.path.join(NPY_SAVED_PATH, str(pid), name)
                new_img = new_img.unsqueeze(0).cpu().half().numpy() # to float16 (half) # (Z,Y,X) -> (1,Z,Y,X)
                new_boxes = [box+[1,1] for box in new_boxes] # [z1,y1,x1,z2,y2,x2] -> [z1,y1,x1,z2,y2,x2,1,1]
                to_save = (new_img, new_box)
                with open(name, "wb") as f:
                    pickle.dump(to_save, f)
                    print("save to", name)
        
    return out

                
def _dataset_preprocessing(target_transform=(1.25,0.75,0.75), target_input_shape=(128,128,128), n_copy=10, save=False, device="cuda", stack_before_process=False, load_fast_5mm=False, load_fast_2d5mm=False, use_all_box=False, use_extra_fp=False):
    """ target_input_shape must be multiples of 32, otherwise it will be very complicated!!! """
    global Tumor, LungDataset
    from dataset import Tumor, LungDataset
    #dataset = LungDataset.load("lung_dataset_20201229_small.pkl")
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    #dataset.get_data(dataset.pids, name="hu+norm_256x256x256_fp16.npy")
    dataset.get_data(dataset.pids)

    ## make sure the inputs are original images
    dataset.batch_1_eval = False # avoid resizing
    dataset.use_random_crop = False # avoid cropped img

    if use_extra_fp:
        dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
        dataset.set_lung_voi()

    ## if want to make fake 1.25mm crops from 5mm lung_voi
    if load_fast_5mm:
        assert not load_fast_2d5mm, "Either 5mm or 2.5mm are valid, but not both at the same time"
        dataset.set_5mm(use_5mm=True, load_5mm_pkl="fast_test_max_5.0x0.75x0.75.pkl")
        dataset.set_batch_1_eval(batch_1_eval=True, equal_spacing=[1.25,0.75,0.75])
    elif load_fast_2d5mm:
        assert not load_fast_5mm, "Either 5mm or 2.5mm are valid, but not both at the same time"
        dataset.set_2d5mm(use_2d5mm=True, load_2d5mm_pkl="fast_test_max_2.5x0.75x0.75.pkl")
        dataset.set_batch_1_eval(batch_1_eval=True, equal_spacing=[1.25,0.75,0.75])
    assert all([npy_name==None for npy_name, _, _ in dataset.data]) # avoid loading npy

    target_transform_text = "x".join(str(i) for i in target_transform)
    target_input_shape_text = "x".join(str(i) for i in target_input_shape)
    device = torch.device(device)
    if (1): #快進data (less I/O)
        pass
        #dataset.data = dataset.data[95+315+17:] #the place where error had occurred before ...
        ###TODO: pid = 20732541 (idx=95+315), seems to have unreasonable transform!! delete data??

        # viewing
        #dataset.data = dataset.data[19+27+21+31+156+2+33+21+215+86+32:]
        #dataset.get_data(["21678302", "6993538"])

        # processing
        dataset.data = dataset.data[:] #[64:]
        #dataset.data = dataset.data[328:]
        
    df = pd.read_excel(EXTRA_FP_EXCEL_PATH, sheet_name="Sheet1", converters={'pid':str,'bbox':str, 'isFP':int})

    for i, (img, bboxes, pid) in tqdm(enumerate(dataset), desc="RandomCropPreprocessing",total=len(dataset)):
        #print("pid", pid) 
        if load_fast_2d5mm or load_fast_5mm:
            transform = [1.25, 0.75, 0.75]
        else:
            dcm_reader = dataset.tumors[ dataset.pid_to_excel_r_relation[pid][0] ].dcm_reader
            transform = [dcm_reader.SliceThickness] + list(dcm_reader.PixelSpacing[::-1])
        #print('pid', pid)
        #print("transform", transform)

        also_crop_boxes = None
        if use_extra_fp: # use_extra_fp_instead
            #print("pid:",pid)
            also_crop_boxes = bboxes
            also_crop_boxes[:,6] = 2
            also_crop_boxes = also_crop_boxes.tolist()
            bboxes = [eval(box) for box in df[df["pid"]==pid]["bbox"]]
            cls_labels = [int(isfp==0) for isfp in df[df["pid"]==pid]["isFP"]] # isfp==0 -> 1 else -> 0
            bboxes = [ box+[cl,1] for box, cl in zip(bboxes, cls_labels) ]
            if (0): #debug
                AnimationViewer(img.squeeze(-1).cpu().numpy(), also_crop_boxes[:,:6].tolist(), note="watch GT")
                continue
            

        img = img.squeeze(-1)
        if stack_before_process: #change z-thickness to 5mm before processing (if use preload fast 5/2.5 mm, you should set this False)
            assert (not load_fast_5mm) and (not load_fast_2d5mm), "It looks like you have stack your volumes beforehand; you shouldn't stack them once more."
            img = img.numpy()
            target_spacing = list(transform)
            target_spacing[0] = 5.0
            assert target_transform[0] == 5.0, "Expect using 5.0mm target_transform"

            if type(bboxes)!=list: bboxes = bboxes.tolist()
            img, new_spacing, bboxes = stacking1D_average(img, axis=0, ori_spacing=transform, target_spacing=target_spacing, ori_bbox=bboxes, stack_func="max")
            transform = new_spacing # get new_spacing after stacking!!!
            img = torch.tensor(img, dtype=torch.float32)
            #print("new spacing after 5mm", new_spacing)
        
        


        #print("ori img shape", img.shape)
        #norm_img = utils_hsz.normalize(img)
        #AnimationViewer(img.numpy(), bbox=[box[:-2] for box in bboxes], note=f"{pid} Original")
        #img = img.unsqueeze(0).to(device) # -> C=1,Z,Y,X
        img = img.to(device)
        #print("random crop transform: {} -> {}".format(transform, target_transform))
        #bboxes = torch.tensor( [box[:-2] for box in bboxes], dtype=torch.float32, device=device )
        out = random_crop_preprocessing(img, [box for box in bboxes], transform, target_transform, target_input_shape, n_copy, use_all_box, also_crop_boxes)
        for j, (out_img, out_bboxes) in enumerate(out):
            #print("out_img", out_img.shape)
            #print("out_bboxes", out_bboxes)
            #AnimationViewer(out_img.cpu().numpy(), out_bboxes, note=f"{pid} Crop{i}")
            if len(out)==1:
                j=i # re-indexing


            if load_fast_5mm:
                name = "random_crop_{}_{}_fake1.25_from_5mm_max_c{}.pkl".format(target_input_shape_text, target_transform_text, j+1)    
            elif load_fast_2d5mm:
                name = "random_crop_{}_{}_fake1.25_from_2.5mm_max_c{}.pkl".format(target_input_shape_text, target_transform_text, j+1)   
            else:
                #name = "random_crop_{}_{}_c{}.pkl".format(target_input_shape_text, target_transform_text, j+1)
                name = "another_data_{}_{}_c{}.pkl".format(target_input_shape_text, target_transform_text, j+1) 
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
                if (1):
                    AnimationViewer(out_img, [box[:6] for box in out_bboxes], note=pid, draw_face=False)
                print("Fake saving to", name)

def _test_pkl():
    pid = 1217803
    for i in range(1,6):
        fpath = os.path.join(NPY_SAVED_PATH, str(pid), f"random_crop_128x128x128_1.25x0.75x0.75_c{i}.pkl")
        with open(fpath, "rb") as f:
            img, bbox = pickle.load(f)
        print(f"bbox {i}:", bbox)
        AnimationViewer(img, bbox)


if __name__ == "__main__":
    #_test_inter()
    #_test_iou()
    #_test_random_crop()
    #_test_pixelspacing()
    device = "cuda"
    #_dataset_preprocessing(save=False, n_copy=2, target_transform=(1.25, 0.75, 0.75),
    #                        target_input_shape=(128, 128, 128), device=device, stack_before_process=False,
    #                        load_fast_5mm=False, load_fast_2d5mm=True)
    _dataset_preprocessing(save=True, n_copy=20, target_transform=(1.25, 0.75, 0.75),
                            target_input_shape=(128, 128, 128), device=device, stack_before_process=False,
                            load_fast_5mm=False, load_fast_2d5mm=False, use_all_box=True, use_extra_fp=True)
    #_test_pkl()