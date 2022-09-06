import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append("/workspace/pancreas")

from dataset import Tumor, LungDataset
from databuilder.yolo4dataset import YOLO4_3DDataset
from utils_ABUS.postprocess import centroid_distance, eval_precision_recall_by_dist, eval_precision_recall
from utils_ABUS.misc import draw_full, build_threshold, AUC
import config.yolov4_config as cfg
from global_variable import CURRENT_DATASET_PKL_PATH, GT_LUT_PKL_PATH, NPY_SAVED_PATH, EXTRA_FP_EXCEL_PATH, GT_LUT_PKL_PATH

def iou_3D(boxes, target_box): #zyxzyx format, by ccy
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

def check_boundary(ct):
    y = (ct[1] > 130 or ct[1] < 5)
    z = (ct[0] > 600 or ct[0] < 40)
    x = (ct[2] > 600 or ct[2] < 40)
    return y or (z and x)


def check_size(axis, size):
    return axis[0]*axis[1]*axis[2] > size

def interpolate_FROC_data(froc_x, froc_y, max_fps=(8, 4, 2, 1, 0.5, 0.25, 0.125)):
        assert max_fps[0]==8
        y_interpolate = 0
        take_i = 0
        log_txt=""
        #print("froc_x", froc_x)
        max_fps = sorted(max_fps, reverse=True) # e.g. [8, 4, 2, 1, 0.5, 0.25, 0.125]
        max_fp = max_fps.pop(0)
        take_is = []
        sens_for_cpm = {}
        for i in range(len(froc_x)):
            FP = froc_x[i]
            sen = froc_y[i]
            while FP<=max_fp:
                take_i = i
                take_is.append(take_i)
                x1 = FP
                y1 = froc_y[i]
                if i>0:
                    x2 = froc_x[i-1]
                    y2 = froc_y[i-1]

                    x_interpolate = max_fp
                    if x2-x1 > 0.001:
                        y_interpolate = (y1 * (x2-x_interpolate) + y2 * (x_interpolate-x1)) / (x2-x1)
                    else: #nan or inf
                        y_interpolate = 0
                else:
                    #if no data point for FP > 8
                    #use sensitivity at FP = FP_small
                    y_interpolate = y1
                log_txt += "take i = {}, FP = {}, sen = {}\n".format(i, int(FP*100)/100, sen)
                log_txt += "interpolate sen = {} for FP = {}\n".format(y_interpolate, max_fp)
                sens_for_cpm[max_fp] = y_interpolate
                if len(max_fps)==0:
                    break
                else:
                    max_fp = max_fps.pop(0)
            else:
                log_txt += "skip i = {}, FP = {}, sen = {}\n".format(i, int(FP*100)/100, sen)
            if len(max_fps)==0 and (max_fp in sens_for_cpm):
                break
        else:
            for max_fp in [max_fp]+list(max_fps):
                log_txt += "No datapoint in froc_x < max_fp = {} (i.e. FP so mush after nms)\n".format(max_fp)
                sens_for_cpm[max_fp] = 0  # since no prediction can get this max_fp
        if len(take_is)>0:
            froc_x = froc_x[take_is[0]:]
            froc_y = froc_y[take_is[0]:]

        if not froc_x[0]==8:
            froc_x = np.insert(froc_x, 0, 8)
            froc_y = np.insert(froc_y, 0, sens_for_cpm[8])
        cpm = sum(sens_for_cpm.values())/len(sens_for_cpm)
        log_txt += f"sens_for_cpm: {sens_for_cpm}\n"
        log_txt += f"CPM: {cpm}"
        print(log_txt)
        return froc_x, froc_y, log_txt, cpm, sens_for_cpm

def froc_take_max(froc_x, froc_y):
    froc_x_tmp = []
    froc_y_tmp = []
    for i in range(len(froc_x)):
        if i==0 or froc_x_tmp[-1] > froc_x[i]:
            froc_x_tmp.append(froc_x[i])
            froc_y_tmp.append(froc_y[i])
    froc_x = np.array(froc_x_tmp)
    froc_y = np.array(froc_y_tmp)
    return froc_x, froc_y

def calculate_FROC(gt_lut, npy_dir, npy_format, size_threshold=0, th_step=0.05, eval_input_size=cfg.VAL["TEST_IMG_SIZE"], 
                    dynamic_input_shape=cfg.VAL["BATCH_1_EVAL"], det_tp_iou_thresh=cfg.VAL["TP_IOU_THRESH"], 
                    return_fp_bboxes=False, return_froc_only=False, npy_format_already=False, only_eval_these_pids=None):
    #size_threshold is 20 in thesis
    if type(npy_dir)==str:
        num_npy = os.listdir(npy_dir) # dir is your directory path
    elif hasattr(npy_dir, "__iter__"): # npy_dirs
        num_npy = sum([os.listdir(d) for d in npy_dir], [])
    total_pass = len(num_npy)
    #print("total_pass:", total_pass, npy_dir)
    #1/0
    all_thre=build_threshold(th_step)
    PERF_per_thre=[]
    PERF_per_thre_s=[]
    true_num, true_small_num = 0, 0
    log_txt = ""
    #with open(annotation_file, 'r') as f:
    #    lines = f.read().splitlines()
    box_lists_cacher = {}
    fp_bboxes_all_pid = {} # only used wher score_hit_thre==0.00

    # extra notFP set
    if cfg.TRAIN["EXTRA_FP_USAGE"] == "eval_only":
        extra_not_fp_df = pd.read_excel(EXTRA_FP_EXCEL_PATH, sheet_name="Sheet1", converters={'pid':str,'bbox':str, 'isFP':int})


    for i, score_hit_thre in enumerate(all_thre):
        txt='Use threshold: {:.3f}'.format(score_hit_thre)
        print(txt)
        log_txt += txt + "\n"

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        pred_num, pred_small_num, file_table, iou_table \
        = [], [], [], [], [], [], [], [], [], []
        # , score_table, mean_score_table, std_score_table
        
        if (0): #ABUS
            TP_table_s, FP_table_s, FN_table_s, \
            TP_table_IOU_1_s, FP_table_IOU_1_s, FN_table_IOU_1_s = [], [], [], [], [], []

        if (1): #LUNG
            TP_table_dist, FP_table_dist, FN_table_dist = [], [], []

        current_pass = 0
        #annotation_file = os.path.join(root, 'annotations/rand_all.txt'
        for pid, boxes in gt_lut.items():
            if (type(only_eval_these_pids)!=type(None)) and (pid not in only_eval_these_pids):
                continue
            if not dynamic_input_shape:
                size = eval_input_size
                scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
            else:
                scale = (1., 1., 1.)
            if not npy_format_already:
                pred_npy = npy_format.format(pid)
            else: # total_FROC
                for form in npy_format:
                    pred_npy = form.format(pid)
                    if os.path.exists(pred_npy):
                        break
                else:
                    # it means in total_FROC function, a specific pid wan't found in all 5 folds
                    pass
                    #assert False, "Not existed: {}".format(pred_npy)
 
            if not os.path.exists(pred_npy):
                continue
            else:
                current_pass += 1
                txt = 'Processing {}/{} data...'.format(current_pass, total_pass)
                print(txt, end='\r')
                #log_txt += txt + "\n"
                if current_pass == total_pass:
                    print("\n")
                    log_txt += "\n"

            #boxes = line[-1].split(' ')
            #boxes = list(map(lambda box: box.split(','), boxes))
            #true_box = [list(map(float, box)) for box in boxes]
            #true_box_s = true_box
            true_box = boxes
            true_box_s = true_box

            if (0): #abus original
                true_box_s = []
                # For the npy volume (after interpolation by spacing), 4px = 1mm (ABUS)
                for li in true_box:
                    axis = [0,0,0]
                    axis[0] = (li[3] - li[0]) / 4
                    axis[1] = (li[4] - li[1]) / 4
                    axis[2] = (li[5] - li[2]) / 4
                    if axis[0] < 10 and axis[1] < 10 and axis[2] < 10:
                        true_box_s.append(li)

            if i == 0:
                true_num += len(true_box)
                true_small_num += len(true_box_s)

            file_name = pid
            file_table.append(file_name)

            ##########################################
            out_boxes = []
            if pred_npy not in box_lists_cacher:  ## to reduce I/O
                box_list = np.load(pred_npy)
                box_lists_cacher[pred_npy] = box_list
            else:
                box_list = box_lists_cacher[pred_npy]

            for bx in box_list: #postprocessing, filtering bbox
                axis = [0,0,0]
                axis[0] = (bx[3] - bx[0]) / scale[0] #/ 4
                axis[1] = (bx[4] - bx[1]) / scale[1] #/ 4
                axis[2] = (bx[5] - bx[2]) / scale[2] #/ 4
                ct = [0,0,0]
                ct[0] = (bx[3] + bx[0]) / 2
                ct[1] = (bx[4] + bx[1]) / 2
                ct[2] = (bx[5] + bx[2]) / 2
                #print(bx)
                #pass
                if bx[6] >= score_hit_thre:# and (not check_boundary(ct)) : #and check_size(axis, size_threshold):
                    out_boxes.append(list(bx))

            pred_num.append(len(out_boxes))

            if (0):
                TP, FP, FN, hits_index, hits_iou, hits_score, TP_by_size_15 = eval_precision_recall_by_dist(
                    out_boxes, true_box, 15, scale)

                TP_IOU_1, FP_IOU_1, FN_IOU_1, hits_index_IOU_1, hits_iou_IOU_1, hits_score_IOU_1, TP_by_size_10 = eval_precision_recall_by_dist(
                    out_boxes, true_box, 10, scale)

                if FN_IOU_1 > 0 and i is 0:
                    print("FN = {}: {}".format(FN_IOU_1, line[0]))
            
            if cfg.TRAIN["EXTRA_FP_USAGE"] == "eval_only": # extra_tp/fp
                current_not_fp_df = extra_not_fp_df[(extra_not_fp_df["pid"]==pid) & (extra_not_fp_df["isFP"]==0)]
                extra_tp =  [eval(lst) for lst in current_not_fp_df["bbox"]]
            else:
                extra_tp = None
            
            if (1): #using iou
                TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall(out_boxes, true_box, det_thresh=det_tp_iou_thresh, scale=scale, extra_tp=extra_tp) #det_thresh == IOU thresh
                #print(f"TP:{TP}, FP:{FP}, FN:{FN}, hits_index:{hits_index}, hits_iou:{hits_iou}, hits_score:{hits_score}")
            if (1): # using luna or other distance criteria
                TP_dist, FP_dist, FN_dist, hits_index_dist, hits_dist, hits_score_dist, TP_by_size, fp_bboxes = eval_precision_recall_by_dist(out_boxes, true_box, dist_thresh=None, scale=scale, spacing=cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"], return_fp_bboxes=True, extra_tp=extra_tp)
                if return_fp_bboxes and score_hit_thre == 0.0:
                    fp_bboxes_all_pid[pid] = fp_bboxes

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            if (1):
                TP_table_dist.append(TP_dist)
                FP_table_dist.append(FP_dist)
                FN_table_dist.append(FN_dist)

            ##########################################
            # Small tumor

            # TP_s, FP_s, FN_s, hits_index_s, hits_iou_s, hits_score_s = eval_precision_recall_by_dist(
            #     out_boxes, true_box_s, 15, scale)

            # TP_IOU_1_s, FP_IOU_1_s, FN_IOU_1_s, hits_index_IOU_1_s, hits_iou_IOU_1_s, hits_score_IOU_1_s = eval_precision_recall_by_dist(
            #     out_boxes, true_box_s, 10, scale)

            # TP_table_s.append(TP_s)
            # FP_table_s.append(FP_s)
            # FN_table_s.append(FN_s)

            # TP_table_IOU_1_s.append(TP_IOU_1_s)
            # FP_table_IOU_1_s.append(FP_IOU_1_s)
            # FN_table_IOU_1_s.append(FN_IOU_1_s)

        TP_table_sum = np.array(TP_table)
        FP_table_sum = np.array(FP_table)
        FN_table_sum = np.array(FN_table)

        if (1):
            TP_table_sum_dist = np.array(TP_table_dist)
            FP_table_sum_dist = np.array(FP_table_dist)
            FN_table_sum_dist = np.array(FN_table_dist)

        # TP_table_sum_s = np.array(TP_table_s)
        # FP_table_sum_s = np.array(FP_table_s)
        # FN_table_sum_s = np.array(FN_table_s)

        # TP_table_sum_IOU_1_s = np.array(TP_table_IOU_1_s)
        # FP_table_sum_IOU_1_s = np.array(FP_table_IOU_1_s)
        # FN_table_sum_IOU_1_s = np.array(FN_table_IOU_1_s)

        sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum()
        sensitivity = sum_TP / (sum_TP + sum_FN + 1e-10)
        precision = sum_TP / (sum_TP + sum_FP + 1e-10)

        if (1):
            sum_TP_dist, sum_FP_dist, sum_FN_dist = TP_table_sum_dist.sum(), FP_table_sum_dist.sum(), FN_table_sum_dist.sum()
            sensitivity_dist = sum_TP_dist / (sum_TP_dist + sum_FN_dist + 1e-10)
            precision_dist = sum_TP_dist / (sum_TP_dist + sum_FP_dist + 1e-10)

        # sum_TP_s, sum_FP_s, sum_FN_s = TP_table_sum_s.sum(), FP_table_sum_s.sum(), FN_table_sum_s.sum()
        # sensitivity_s = sum_TP_s/(sum_TP_s+sum_FN_s+1e-10)
        # precision_s = sum_TP_s/(sum_TP_s+sum_FP_s+1e-10)

        # sum_TP_IOU_1_s, sum_FP_IOU_1_s, sum_FN_IOU_1_s = TP_table_sum_IOU_1_s.sum(), FP_table_sum_IOU_1_s.sum(), FN_table_sum_IOU_1_s.sum()
        # sensitivity_IOU_1_s = sum_TP_IOU_1_s/(sum_TP_IOU_1_s+sum_FN_IOU_1_s+1e-10)
        # precision_IOU_1_s = sum_TP_IOU_1_s/(sum_TP_IOU_1_s+sum_FP_IOU_1_s+1e-10)

        if (0) and sensitivity > 0.125:
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity, #small FROC Y
                precision,
                sum_FP/total_pass, #small FROC X
                sensitivity_IOU_1, #all FROC Y
                precision_IOU_1,
                sum_FP_IOU_1/total_pass], #all FROC x
                )
        if (0) and sensitivity > 0.125:
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity, #small FROC Y
                precision,
                sum_FP/total_pass, #small FROC X
                None,
                None,
                None], #all FROC x
                )
        if (1):
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity, # ---using iou---
                precision, # ---using iou---
                sum_FP/total_pass, # ---using iou---
                sensitivity_dist, # ---using dist---
                precision_dist, # ---using dist---
                sum_FP_dist/total_pass],  # ---using dist---
                )

        # if sensitivity_s > 0.125:
        #     PERF_per_thre_s.append([
        #         score_hit_thre,
        #         total_pass,
        #         sensitivity_s,
        #         precision_s,
        #         sum_FP_s/total_pass,
        #         sensitivity_IOU_1_s,
        #         precision_IOU_1_s,
        #         sum_FP_IOU_1_s/total_pass])

        txt = 'Threshold:{:.3f}\n'.format(score_hit_thre)
        txt += 'Using IOU -- Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}\n'.format(sensitivity, precision, sum_FP/total_pass)
        if (1):
            txt += 'Using dist -- Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}\n'.format(sensitivity_dist, precision_dist, sum_FP_dist/total_pass)
        print(txt)
        log_txt += txt + "\n"
        if (0):
            print('Dist of Center < 15mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity, precision, sum_FP/total_pass))
            print('Dist of Center < 10mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity_IOU_1, precision_IOU_1, sum_FP_IOU_1/total_pass))
        #print('\n')

    txt = 'All tumors: {}'.format(true_num)
    print(txt)
    log_txt += txt + "\n"
    if (0):
        print('Small/All tumors: {}/{}'.format(true_small_num, true_num))

    data = np.array(PERF_per_thre)
    max_sens_iou = PERF_per_thre[0][2]
    max_sens_dist = PERF_per_thre[0][5]
    if (0):
        data_s = np.array(PERF_per_thre_s)
    plt.figure()
    plt.rc('font',family='Times New Roman', weight='bold')
    area_iou, area_dist, cpm, cpm_dist = 0, 0, 0 ,0
    if len(data) == 0:
        txt = 'Inference result is empty.'
        print(txt)
        log_txt += txt+"\n"
        area = 0
    else:
        area_dist = 0.0 #prevent error
        if (1):
            froc_x_dist, froc_y_dist, sub_log_txt, cpm_dist, sens_for_cpm_dist = interpolate_FROC_data(data[..., 7], data[..., 5], max_fps=(8, 4, 2, 1, 0.5, 0.25 ,0.125))
            froc_x_dist, froc_y_dist = froc_take_max(froc_x_dist, froc_y_dist)
            draw_full(froc_x_dist, froc_y_dist, '#FF6D6C', 'Dist', '-', 1, True)
            area_dist = AUC(froc_x_dist, froc_y_dist, normalize=True)
            log_txt += sub_log_txt + "\n"

        froc_x, froc_y, sub_log_txt, cpm, sens_for_cpm = interpolate_FROC_data(data[..., 4], data[..., 2], max_fps=(8, 4, 2, 1, 0.5, 0.25 ,0.125))
        log_txt += sub_log_txt + "\n"
        froc_x, froc_y = froc_take_max(froc_x, froc_y)
        #draw_full(froc_x, froc_y, '#FF0000', 'D < 15 mm', '-', 1, True)
        draw_full(froc_x, froc_y, '#FF0000', 'IOU', '-', 1, True)
        area_iou = AUC(froc_x, froc_y, normalize=True)

    # if len(data_s) == 0:
    #     print('Inference result for small is empty.')
    # else:
    #     draw_full(data_s[..., 7], data_s[..., 5], '#6D6CFF', 'Dist < 15', ':', 1)
    #     draw_full(data_s[..., 4], data_s[..., 2], '#0000FF', 'Dist < 10', '-', 1)

    # axes = plt.gca()
    # axes.axis([0, 10, 0.5, 1])
    # axes.set_aspect('auto')
    if return_froc_only:
        return froc_x_dist, froc_y_dist, cpm_dist, log_txt, sens_for_cpm_dist

    plt.xlim(1, 8)
    x_tick = np.arange(0, 10, 2)
    plt.xticks(x_tick)
    #plt.ylim(0.5, 1)
    plt.ylim(0,1)
    y_tick = np.arange(0.5, 1, 0.05)
    y_tick = np.append(y_tick, 0.98)
    y_tick = np.sort(y_tick)
    plt.yticks(y_tick)
    plt.legend(loc='lower right')
    # plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('False Positive Per Pass')
    if return_fp_bboxes:
        #plt.close()
        return area_dist, area_iou, plt, log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou, fp_bboxes_all_pid
    else:
        return area_dist, area_iou, plt, log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou

def calculate_FROC_randomcrop(annotation_file, npy_dir, npy_format, ori_dataset, size_threshold=0, th_step=0.05, 
                                det_tp_iou_thresh=cfg.VAL["TP_IOU_THRESH"]):
    #size_threshold is 20 in thesis
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)
    all_thre=build_threshold(th_step)
    PERF_per_thre=[]
    PERF_per_thre_s=[]
    true_num, true_small_num = 0, 0
    log_txt = ""
    with open(annotation_file, 'r') as f:
        lines = f.read().splitlines()
    box_lists_cacher = {}
    true_boxes_cacher = {}

    for i, score_hit_thre in enumerate(all_thre):
        txt='Use threshold: {:.3f}'.format(score_hit_thre)
        print(txt)
        log_txt += txt + "\n"

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        pred_num, pred_small_num, file_table, iou_table \
        = [], [], [], [], [], [], [], [], [], []
        # , score_table, mean_score_table, std_score_table
        
        if (1): #LUNG
            TP_table_dist, FP_table_dist, FN_table_dist = [], [], []

        current_pass = 0
        #annotation_file = os.path.join(root, 'annotations/rand_all.txt')
        
        for line in lines:
            line = line.split(',', 4)
            # Always use 640,160,640 to compute iou
            #size = eval_input_size
            #scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
            scale = (1., 1., 1.)  ## for random crop
            pid = line[0]#.replace('/', '_')
            pred_npy = npy_format.format(pid)
            if not os.path.exists(pred_npy):
                continue
            else:
                current_pass += 1
                txt = 'Processing {}/{} data...'.format(current_pass, total_pass)
                print(txt, end='\r')
                #log_txt += txt + "\n"
                if current_pass == total_pass:
                    print("\n")
                    log_txt += "\n"

            assert ori_dataset.use_random_crop
            prefix = ori_dataset.random_crop_file_prefix
            fpath = pjoin(NPY_SAVED_PATH, str(pid), "{}_c1.pkl".format(prefix))
            if fpath not in true_boxes_cacher:
                with open(fpath, "rb") as f:
                    _, bboxes = pickle.load(f)
                n = len(bboxes)
                bboxes = np.array(bboxes)
                true_box = np.concatenate( [bboxes, np.ones((n,2))], axis=1 )
                true_boxes_cacher[fpath] = true_box
            else:
                true_box = true_boxes_cacher[fpath]
            
            true_box_s = true_box
            
            if i == 0:
                true_num += len(true_box)
                true_small_num += len(true_box_s)

            file_name = line[0]
            file_table.append(file_name)

            ##########################################
            out_boxes = []
            if pred_npy not in box_lists_cacher:  ## to reduce I/O
                box_list = np.load(pred_npy)
                box_lists_cacher[pred_npy] = box_list
            else:
                box_list = box_lists_cacher[pred_npy]
            #box_list = np.load(pred_npy)
            for bx in box_list: #postprocessing, filtering bbox
                axis = [0,0,0]
                axis[0] = (bx[3] - bx[0]) / 1  # random crop no need scaling
                axis[1] = (bx[4] - bx[1]) / 1
                axis[2] = (bx[5] - bx[2]) / 1
                ct = [0,0,0]
                ct[0] = (bx[3] + bx[0]) / 2
                ct[1] = (bx[4] + bx[1]) / 2
                ct[2] = (bx[5] + bx[2]) / 2
                if bx[6] >= score_hit_thre:# and (not check_boundary(ct)) : #and check_size(axis, size_threshold):
                    out_boxes.append(list(bx))

            pred_num.append(len(out_boxes))

            
            if (1):
                TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall(out_boxes, true_box, det_thresh=det_tp_iou_thresh, scale=scale) #det_thresh == IOU thresh
                #print(f"TP:{TP}, FP:{FP}, FN:{FN}, hits_index:{hits_index}, hits_iou:{hits_iou}, hits_score:{hits_score}")
            if (1): # using luna or other distance criteria
                TP_dist, FP_dist, FN_dist, hits_index_dist, hits_dist, hits_score_dist, TP_by_size = eval_precision_recall_by_dist(out_boxes, true_box, dist_thresh=None, scale=scale, spacing=cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"])

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            if (1):
                TP_table_dist.append(TP_dist)
                FP_table_dist.append(FP_dist)
                FN_table_dist.append(FN_dist)

        TP_table_sum = np.array(TP_table)
        FP_table_sum = np.array(FP_table)
        FN_table_sum = np.array(FN_table)

        if (1):
            TP_table_sum_dist = np.array(TP_table_dist)
            FP_table_sum_dist = np.array(FP_table_dist)
            FN_table_sum_dist = np.array(FN_table_dist)

        sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum()
        sensitivity = sum_TP/(sum_TP+sum_FN+1e-10)
        precision = sum_TP/(sum_TP+sum_FP+1e-10)

        if (1):
            sum_TP_dist, sum_FP_dist, sum_FN_dist = TP_table_sum_dist.sum(), FP_table_sum_dist.sum(), FN_table_sum_dist.sum()
            sensitivity_dist = sum_TP_dist / (sum_TP_dist + sum_FN_dist + 1e-10)
            precision_dist = sum_TP_dist / (sum_TP_dist + sum_FP_dist + 1e-10)

        if (1):
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity, # ---using iou---
                precision, # ---using iou---
                sum_FP/total_pass, # ---using iou---
                sensitivity_dist, # ---using dist---
                precision_dist, # ---using dist---
                sum_FP_dist/total_pass],  # ---using dist---
                )

        txt = 'Threshold:{:.3f}\n'.format(score_hit_thre)
        txt += 'Using IOU -- Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}\n'.format(sensitivity, precision, sum_FP/total_pass)
        if (1):
            txt += 'Using dist -- Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}\n'.format(sensitivity_dist, precision_dist, sum_FP_dist/total_pass)
        print(txt)
        log_txt += txt + "\n"
        if (0):
            print('Dist of Center < 15mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity, precision, sum_FP/total_pass))
            print('Dist of Center < 10mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity_IOU_1, precision_IOU_1, sum_FP_IOU_1/total_pass))
        #print('\n')

    txt = 'All tumors: {}'.format(true_num)
    print(txt)
    log_txt += txt + "\n"

    data = np.array(PERF_per_thre)
    max_sens_iou = PERF_per_thre[0][2]
    max_sens_dist = PERF_per_thre[0][5]

    plt.figure()
    plt.rc('font',family='Times New Roman', weight='bold')
    area_dist, area_iou, cpm, cpm_dist = 0, 0, 0 ,0
    if len(data) == 0:
        txt = 'Inference result is empty.'
        print(txt)
        log_txt += txt+"\n"
        area = 0
    else:
        area_dist = 0.0 #prevent error
        if (1):
            froc_x_dist, froc_y_dist, sub_log_txt, cpm_dist, sens_for_cpm_dist = interpolate_FROC_data(data[..., 7], data[..., 5], max_fps=(8, 4, 2, 1, 0.5, 0.25 ,0.125))
            froc_x_dist, froc_y_dist = froc_take_max(froc_x_dist, froc_y_dist)
            draw_full(froc_x_dist, froc_y_dist, '#FF6D6C', 'Dist', '-', 1, True)
            area_dist = AUC(froc_x_dist, froc_y_dist, normalize=True)
            log_txt += sub_log_txt + "\n"

        froc_x, froc_y, sub_log_txt, cpm, sens_for_cpm = interpolate_FROC_data(data[..., 4], data[..., 2], max_fps=(8, 4, 2, 1, 0.5, 0.25 ,0.125))
        log_txt += sub_log_txt + "\n"
        froc_x, froc_y = froc_take_max(froc_x, froc_y)
        draw_full(froc_x, froc_y, '#FF0000', '', '-', 1, True)
        area_iou = AUC(froc_x, froc_y, normalize=True)


    x_tick = np.arange(0, 10, 2)
    plt.xticks(x_tick)
    #plt.ylim(0.5, 1)
    plt.ylim(0,1)
    y_tick = np.arange(0.5, 1, 0.05)
    y_tick = np.append(y_tick, 0.98)
    y_tick = np.sort(y_tick)
    plt.yticks(y_tick)
    plt.legend(loc='lower right')
    # plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('False Positive Per Pass')
    return area_dist, area_iou, plt, log_txt, cpm_dist, cpm, max_sens_dist, max_sens_iou

def plot_total_FROC(froc_data, save=False, save_path="total_froc.png", gt_lut_pkl_path=GT_LUT_PKL_PATH, only_eval_these_pids=None, calc_5_fold_differently=False):
    #size_threshold is 20 in thesis
    #assert len(npy_dirs) == len(exp_names), "Should of same length"

    with open(gt_lut_pkl_path, "rb") as f:
        gt_lut = pickle.load(f)
    
    #dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    #dataset.get_data(dataset.pids)
    #pid_folds = {}
    ##all_pids = set()
    #for fold in range(5):
    #    _, _, test_data = dataset.make_kfolds_using_pids(num_k_folds=5, k_folds_seed=123, current_fold=fold, valid_test_split=True, portion_list=[3,1,1])
    #    pid_folds[fold] = test_data
        

    size_threshold=20
    th_step=0.01 # not 0.05
    eval_input_size=cfg.VAL["TEST_IMG_SIZE"]
    dynamic_input_shape=cfg.VAL["BATCH_1_EVAL"]
    det_tp_iou_thresh=cfg.VAL["TP_IOU_THRESH"]
    data = {}
    total_log = ""
    if calc_5_fold_differently:
        total_sens = {k:0 for k in [8,4,2,1,0.5,0.25,0.125]}
        total_cpm = 0
    for exp_name, npy_dirs_kfold in froc_data.items():
        npy_dirs = []
        npy_formats = []
        for i, (npy_dir, epoch) in enumerate(npy_dirs_kfold):
            npy_dir = "D:/CH/LungDetection/training/preidction/" + npy_dir + f"/{epoch}_conf0.015/evaluate"
            npy_format = npy_dir + "/{}_test.npy"
            npy_dirs.append(npy_dir)
            npy_formats.append(npy_format)
            if calc_5_fold_differently:
                froc_x, froc_y, cpm_dist, log_txt, sens_for_cpm_dist = calculate_FROC(gt_lut, [npy_dir], [npy_format], size_threshold, th_step, eval_input_size, 
                                                    dynamic_input_shape, det_tp_iou_thresh, return_fp_bboxes=False, 
                                                    return_froc_only=True, npy_format_already=True, only_eval_these_pids=only_eval_these_pids)
                #print("FROC x", froc_x)
                #print("FROC Y", froc_y)
                #print("sens for cpm dist", sens_for_cpm_dist)
                for k in sens_for_cpm_dist:
                    #print(f"k={k}", type(k))
                    #print(total_sens.keys())
                    total_sens[k] += sens_for_cpm_dist[k]
                total_cpm += cpm_dist
                total_log += f"Fold {i}, cpm={cpm_dist}, sens_for_cpm={sens_for_cpm_dist}\n"

                
        if not calc_5_fold_differently:
            froc_x, froc_y, cpm_dist, log_txt, sens_for_cpm_dist = calculate_FROC(gt_lut, npy_dirs, npy_formats, size_threshold, th_step, eval_input_size, 
                                                dynamic_input_shape, det_tp_iou_thresh, return_fp_bboxes=False, 
                                                return_froc_only=True, npy_format_already=True, only_eval_these_pids=only_eval_these_pids)
        else:
            sens_for_cpm_dist = {k:v/len(npy_dirs_kfold) for k,v in total_sens.items()}
            total_cpm /= len(npy_dirs_kfold)
            total_log += f"\nAverage cpm: {total_cpm}, sens: {sens_for_cpm_dist}\n"
        data[exp_name] = (froc_x, froc_y, cpm_dist)
    plt.close()
    print("total log", total_log)
    
    plt.figure()
    plt.rc('font',family='Times New Roman', weight='bold')
    for exp_name, (froc_x, froc_y, cpm_dist) in data.items():
        plt.plot(froc_x, froc_y, label=exp_name+', CPM = %.4f'%cpm_dist, linestyle="-")
    x_tick = np.arange(0, 10, 1)
    plt.xticks(x_tick)
    #plt.ylim(0.5, 1)
    plt.ylim(0,1)
    if (0):
        y_tick = np.arange(0.5, 1, 0.05)
        y_tick = np.append(y_tick, 0.98)
        y_tick = np.sort(y_tick)
        plt.yticks(y_tick)
    plt.legend(loc='lower right')
    # plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('Average Number of False Positive Per Scan')
    plt.title("FROC Performance")
    if save:
        plt.savefig(save_path)
    plt.show()
    return log_txt
    



def get_gt_lut(dataset_path=CURRENT_DATASET_PKL_PATH, save=False, save_path=GT_LUT_PKL_PATH):
    dataset = LungDataset.load(dataset_path)
    pids = dataset.pids
    #pids = pids[:3]
    dataset.get_data(pids)
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    dataset.set_lung_voi(True)
    dataset = YOLO4_3DDataset(dataset, classes=[0, 1], img_size=cfg.VAL["TEST_IMG_SIZE"], cache_size=0, batch_1_eval=cfg.VAL["BATCH_1_EVAL"])
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=cfg.VAL["NUMBER_WORKERS"],
                            shuffle=False, pin_memory=False
                            )
    gt_lut = {}
    for i, (_, _, _, _, _, _, _, img_names, _, valid_bboxes)  in tqdm(enumerate(dataloader), total=len(dataloader)):
        #print("eval imgs input shape:", imgs.shape)
        for img_name, valid_bbox in zip(img_names, valid_bboxes):
            valid_bbox = valid_bbox.cpu().tolist()
            gt_lut[img_name] = valid_bbox
    print("gt_lut:", gt_lut)
    if save:
        with open(save_path, "wb") as f:
            pickle.dump(gt_lut, f)
    return gt_lut


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='Threshold for size filtering.'
    )
    parser.add_argument(
        '--root', '-r', type=str, required=True,
        help='folder path for data/sys_ucc/'
    )
    return parser.parse_args()


if __name__ == '__main__':
    #root = 'datasets/abus'
    #get_gt_lut(save=False)

    # currently no use
    froc_data = {
        #"CSP-SA-YOLOv4 1.25mm":[
        #    ("train_rc_config_5.15_attnetion2_layer0_f0_EXTRA_FP_testing", 272),
        #    ("train_rc_config_5.15_attnetion2_layer0_f1_EXTRA_FP_testing", 170),
        #    ("train_rc_config_5.15_attnetion2_layer0_f2_EXTRA_FP_testing", 272),
        #    ("train_rc_config_5.15.2_attnetion2_layer0_f3_EXTRA_FP_testing", 238),
        #    ("train_rc_config_5.15.2_attnetion2_layer0_f4_EXTRA_FP_testing", 221),
        #],

        #"CSP-SA-YOLOv4 2.5mm":[
        #    ("train_rc_config_5.15_attetion2_layer0_for_2d5mm_f0_EXTRA_FP_testing", 170),
        #    ("train_rc_config_5.15_attetion2_layer0_for_2d5mm_f1_EXTRA_FP_testing", 250),
        #    ("train_rc_config_5.15_attetion2_layer0_for_2d5mm_f2_EXTRA_FP_testing", 170),
        #    ("train_rc_config_5.15_attetion2_layer0_for_2d5mm_f3_EXTRA_FP_testing", 119),
        #    ("train_rc_config_5.15_attetion2_layer0_for_2d5mm_f4_EXTRA_FP_testing", 238),
        #],

        "CSP-SA-YOLOv4 5.0mm":[
            ("train_rc_config_5.15_attetion2_layer0_for_5mm_f0_EXTRA_FP_testing", 153),
            ("train_rc_config_5.15_attetion2_layer0_for_5mm_f1_EXTRA_FP_testing", 136),
            ("train_rc_config_5.15_attetion2_layer0_for_5mm_f2_EXTRA_FP_testing", 272),
            ("train_rc_config_5.15_attetion2_layer0_for_5mm_f3_EXTRA_FP_testing", 204),
            ("train_rc_config_5.15_attetion2_layer0_for_5mm_f4_EXTRA_FP_testing", 238),
        ],

        #"ResNeSt+FPR+iter":[
        #    ("train_rc_config_5.9.1_iterative_fp_update_f0_EXTRA_FP_testing", 272),
        #    ("train_rc_config_5.9.2_iterative_fp_update_f1_EXTRA_FP_testing", 221),
        #    ("train_rc_config_5.9.1_iterative_fp_update_f2_EXTRA_FP_testing", 289),
        #    ("train_rc_config_5.9.1_iterative_fp_update_f3_EXTRA_FP_testing", 238),
        #    ("train_rc_config_5.9.1_iterative_fp_update_f4_EXTRA_FP_testing", 255),
        #],

        #"ResNeSt":[
        #    ("resnest_no_fp_reduction_dry_run_f0_EXTRA_FP_testing", 170),
        #    ("resnest_no_fp_reduction_dry_run_f1_EXTRA_FP_testing", 204),
        #    ("resnest_no_fp_reduction_dry_run_f2_EXTRA_FP_testing", 187),
        #    ("resnest_no_fp_reduction_dry_run_f3_EXTRA_FP_testing", 255),
        #    ("resnest_no_fp_reduction_dry_run_f4_EXTRA_FP_testing", 255),       
        #],
        
    }
    from view_dataset import ct_ldct
    #ldct_dic = {"CT":None, "LDCT":None}
    ldct_dic = ct_ldct(return_pids=True) #672 CT, 94 LDCT
    #print(ldct_dic["LDCT"][:10])
    #1/0
    print("1034114" in ldct_dic["CT"], "13564970" in ldct_dic["CT"])

    calc_5_fold_differently = False
    log_txt = plot_total_FROC(froc_data, save=True, save_path="D:/CH/total_froc_CT.png", gt_lut_pkl_path=GT_LUT_PKL_PATH, only_eval_these_pids=ldct_dic["CT"], calc_5_fold_differently=calc_5_fold_differently)
    #log_txt = plot_total_FROC(froc_data, save=True, save_path="D:/CH/total_froc_LDCT.png", gt_lut_pkl_path=GT_LUT_PKL_PATH, only_eval_these_pids=ldct_dic["LDCT"], calc_5_fold_differently=calc_5_fold_differently)
    print()
    #print("LOG TXT")
    #print(log_txt)

    #npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    #npy_format = npy_dir + '{}'

    #npy_dir = '/data/Hiola/YOLOv4-pytorch/data/pred_result/evaluate/'
    #npy_format = npy_dir + '{}_0.npy'
    #args = _parse_args()
    #root = args.root
    #main(args)
    #area_small, area_big, plt = calculate_FROC(root, npy_dir, npy_format)
