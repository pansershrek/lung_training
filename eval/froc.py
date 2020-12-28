import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils_ABUS.postprocess import centroid_distance, eval_precision_recall_by_dist, eval_precision_recall
from utils_ABUS.misc import draw_full, build_threshold, AUC

import config.yolov4_config as cfg

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

def interpolate_FROC_data(froc_x, froc_y, max_fp):
        y_interpolate = 0
        take_i = 0
        log_txt=""
        #print("froc_x", froc_x)
        for i in range(len(froc_x)):
            FP = froc_x[i]
            sen = froc_y[i]
            if FP<=max_fp:
                take_i = i
                x1 = FP
                y1 = froc_y[i]
                if i>0:
                    x2 = froc_x[i-1]
                    y2 = froc_y[i-1]

                    x_interpolate = max_fp
                    y_interpolate = (y1 * (x2-x_interpolate) + y2 * (x_interpolate-x1)) / (x2-x1)
                else:
                    #if no data point for FP > 8
                    #use sensitivity at FP = FP_small
                    y_interpolate = y1
                log_txt += "take i = {}, FP = {}, sen = {}\n".format(i, int(FP*100)/100, sen)
                log_txt += "interpolate sen = {} for FP = {}\n".format(y_interpolate, max_fp)
                break
            else:
                log_txt += "skip i = {}, FP = {}, sen = {}\n".format(i, int(FP*100)/100, sen)
        else:
            log_txt += "No datapoint in froc_x < max_fp = {} (i.e. FP so mush after nms)".format(max_fp)
        print(log_txt)
        froc_x = froc_x[take_i:]
        froc_y = froc_y[take_i:]

        if not froc_x[0]==8:
            froc_x = np.insert(froc_x, 0, 8)
            froc_y = np.insert(froc_y, 0, y_interpolate)
        return froc_x, froc_y, log_txt

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

def calculate_FROC(annotation_file, npy_dir, npy_format, size_threshold=0, th_step=0.05, eval_input_size=cfg.VAL["TEST_IMG_SIZE"]):
    #size_threshold is 20 in thesis
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)
    all_thre=build_threshold(th_step)
    PERF_per_thre=[]
    PERF_per_thre_s=[]
    true_num, true_small_num = 0, 0
    log_txt = ""

    for i, score_hit_thre in enumerate(all_thre):
        txt='Use threshold: {:.3f}'.format(score_hit_thre)
        print(txt)
        log_txt += txt + "\n"

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        pred_num, pred_small_num, file_table, iou_table \
        = [], [], [], [], [], [], [], [], [], []
        # , score_table, mean_score_table, std_score_table
        
        if (0):
            TP_table_s, FP_table_s, FN_table_s, \
            TP_table_IOU_1_s, FP_table_IOU_1_s, FN_table_IOU_1_s = [], [], [], [], [], []

        current_pass = 0
        #annotation_file = os.path.join(root, 'annotations/rand_all.txt'
        with open(annotation_file, 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            line = line.split(',', 4)
            # Always use 640,160,640 to compute iou
            size = eval_input_size
            scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
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

            boxes = line[-1].split(' ')
            boxes = list(map(lambda box: box.split(','), boxes))
            true_box = [list(map(float, box)) for box in boxes]
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

            file_name = line[0]
            file_table.append(file_name)

            ##########################################
            out_boxes = []
            box_list = np.load(pred_npy)
            for bx in box_list: #postprocessing, filtering bbox
                axis = [0,0,0]
                axis[0] = (bx[3] - bx[0]) / scale[0] #/ 4
                axis[1] = (bx[4] - bx[1]) / scale[1] #/ 4
                axis[2] = (bx[5] - bx[2]) / scale[2] #/ 4
                ct = [0,0,0]
                ct[0] = (bx[3] + bx[0]) / 2
                ct[1] = (bx[4] + bx[1]) / 2
                ct[2] = (bx[5] + bx[2]) / 2
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
            
            if (1):
                TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall(out_boxes, true_box, det_thresh=0.25, scale=scale) #det_thresh == IOU thresh
                #print(f"TP:{TP}, FP:{FP}, FN:{FN}, hits_index:{hits_index}, hits_iou:{hits_iou}, hits_score:{hits_score}")

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            if (0):
                TP_table_IOU_1.append(TP_IOU_1)
                FP_table_IOU_1.append(FP_IOU_1)
                FN_table_IOU_1.append(FN_IOU_1)

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

        if (0):
            TP_table_sum_IOU_1 = np.array(TP_table_IOU_1)
            FP_table_sum_IOU_1 = np.array(FP_table_IOU_1)
            FN_table_sum_IOU_1 = np.array(FN_table_IOU_1)

        # TP_table_sum_s = np.array(TP_table_s)
        # FP_table_sum_s = np.array(FP_table_s)
        # FN_table_sum_s = np.array(FN_table_s)

        # TP_table_sum_IOU_1_s = np.array(TP_table_IOU_1_s)
        # FP_table_sum_IOU_1_s = np.array(FP_table_IOU_1_s)
        # FN_table_sum_IOU_1_s = np.array(FN_table_IOU_1_s)

        sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum()
        sensitivity = sum_TP/(sum_TP+sum_FN+1e-10)
        precision = sum_TP/(sum_TP+sum_FP+1e-10)

        if (0):
            sum_TP_IOU_1, sum_FP_IOU_1, sum_FN_IOU_1 = TP_table_sum_IOU_1.sum(), FP_table_sum_IOU_1.sum(), FN_table_sum_IOU_1.sum()
            sensitivity_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FN_IOU_1+1e-10)
            precision_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FP_IOU_1+1e-10)

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
                sensitivity, #small FROC Y
                precision,
                sum_FP/total_pass, #small FROC X
                None,
                None,
                None], #all FROC x
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
    if (0):
        data_s = np.array(PERF_per_thre_s)
    plt.figure()
    plt.rc('font',family='Times New Roman', weight='bold')
    area_small, area_big = 0, 0
    if len(data) == 0:
        txt = 'Inference result is empty.'
        print(txt)
        log_txt += txt+"\n"
        area = 0
    else:
        area_small = 0.0 #prevent error
        if (0):
            froc_x, froc_y, sub_log_txt = interpolate_FROC_data(data[..., 7], data[..., 5], max_fp=8)
            froc_x, froc_y = froc_take_max(froc_x, froc_y)
            draw_full(froc_x, froc_y, '#FF6D6C', 'D < 10 mm', '-.', 1, True)
            area_small = AUC(froc_x, froc_y, normalize=True)

        froc_x, froc_y, sub_log_txt = interpolate_FROC_data(data[..., 4], data[..., 2], max_fp=8)
        log_txt += sub_log_txt + "\n"
        froc_x, froc_y = froc_take_max(froc_x, froc_y)
        #draw_full(froc_x, froc_y, '#FF0000', 'D < 15 mm', '-', 1, True)
        draw_full(froc_x, froc_y, '#FF0000', '', '-', 1, True)
        area_big = AUC(froc_x, froc_y, normalize=True)



    # if len(data_s) == 0:
    #     print('Inference result for small is empty.')
    # else:
    #     draw_full(data_s[..., 7], data_s[..., 5], '#6D6CFF', 'Dist < 15', ':', 1)
    #     draw_full(data_s[..., 4], data_s[..., 2], '#0000FF', 'Dist < 10', '-', 1)

    # axes = plt.gca()
    # axes.axis([0, 10, 0.5, 1])
    # axes.set_aspect('auto')
    plt.xlim(1, 10)
    x_tick = np.arange(0, 10, 2)
    plt.xticks(x_tick)
    plt.ylim(0.5, 1)
    y_tick = np.arange(0.5, 1, 0.05)
    y_tick = np.append(y_tick, 0.98)
    y_tick = np.sort(y_tick)
    plt.yticks(y_tick)
    plt.legend(loc='lower right')
    # plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('False Positive Per Pass')
    return area_small, area_big, plt, log_txt

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
    root = 'datasets/abus'

    #npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    #npy_format = npy_dir + '{}'

    npy_dir = '/data/Hiola/YOLOv4-pytorch/data/pred_result/evaluate/'
    npy_format = npy_dir + '{}_0.npy'
    args = _parse_args()
    root = args.root
    main(args)
    area_small, area_big, plt = calculate_FROC(root, npy_dir, npy_format)
