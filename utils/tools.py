#coding=utf-8
import sys
sys.path.append("..")
import torch
import numpy as np
import cv2
import random
import config.yolov4_config as cfg
import os
import math

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        print("initing {} ".format(m))
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('BatchNorm2d') != -1:
        print("initing {} ".format(m))

        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2.0
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2.0
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyzxyz2xyzwhd(x):
    # Convert bounding box format from [x1, y1, z1, x2, y2, z2] to [x, y, z, w, h, d]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 3]) / 2.0
    y[:, 1] = (x[:, 1] + x[:, 4]) / 2.0
    y[:, 2] = (x[:, 2] + x[:, 5]) / 2.0
    y[:, 3] = x[:, 3] - x[:, 0]
    y[:, 4] = x[:, 4] - x[:, 1]
    y[:, 5] = x[:, 5] - x[:, 2]
    return y


def xyzwhd2xyzxyz(x):
    # Convert bounding box format from [x, y, z, w, h, d] to [x1, y1, z1, x2, y2, z2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 3] / 2
    y[:, 1] = x[:, 1] - x[:, 4] / 2
    y[:, 2] = x[:, 2] - x[:, 5] / 2
    y[:, 3] = x[:, 0] + x[:, 3] / 2
    y[:, 4] = x[:, 1] + x[:, 4] / 2
    y[:, 5] = x[:, 2] + x[:, 5] / 2
    return y

def wh_iou(box1, box2):
    # box1 shape : [2]
    # box2 shape : [bs*N, 2]
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return (inter_area / union_area)  # iou shape : [bs*N]


def bbox_iou(box1, box2, mode="xyxy"):
    """
    numpy version iou, and use for nms
    """
    # Get the coordinates of bounding boxes

    if mode == "xyxy":
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_area = np.maximum((np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)), 0.0) * \
                 np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0.0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def iou_xywh_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x,y,w,h)，其中(x,y)是bbox的中心坐标
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    if boxes1.shape[-1]==6:
        boxes1_area = boxes1[..., 3] * boxes1[..., 4] * boxes1[..., 5]
        boxes2_area = boxes2[..., 3] * boxes2[..., 4] * boxes2[..., 5]

        # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
        # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
        boxes1 = np.concatenate([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                                boxes1[..., :3] + boxes1[..., 3:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                                boxes2[..., :3] + boxes2[..., 3:] * 0.5], axis=-1)

        # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :3], boxes2[..., :3])
        right_down = np.minimum(boxes1[..., 3:], boxes2[..., 3:])

        # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]

    else:
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
        # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    IOU = np.nan_to_num(IOU, nan=0)
    return IOU


def iou_xyxy_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    if boxes1.shape[-1]==6:
        boxes1_area = np.multiply.reduce(boxes1[..., 3:6] - boxes1[..., 0:3], axis=-1)
        boxes2_area = np.multiply.reduce(boxes2[..., 3:6] - boxes2[..., 0:3], axis=-1)
        # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :3], boxes2[..., :3])
        right_down = np.minimum(boxes1[..., 3:], boxes2[..., 3:])
    else:
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = np.multiply.reduce(inter_section, axis=-1)
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / (union_area+1e-3)
    return IOU


def iou_xyxy_torch(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def iou_xywh_torch(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def GIOU_xywh_torch(boxes1, boxes2):
    """
     https://arxiv.org/abs/1902.09630
    boxes1(boxes2)' shape is [..., (x,y,w,h)].The size is for original image.
    """
    # xywh->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area =  inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
    return GIOU

def CIOU_xyzwhd_torch(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    #print("boxes1:", boxes1)
    #print("boxes2:", boxes2)
    # xywh->xyxy
    boxes1 = torch.cat([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                        boxes1[..., :3] + boxes1[..., 3:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                        boxes2[..., :3] + boxes2[..., 3:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :3], boxes1[..., 3:]),
                        torch.max(boxes1[..., :3], boxes1[..., 3:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :3], boxes2[..., 3:]),
                        torch.max(boxes2[..., :3], boxes2[..., 3:])], dim=-1)

    boxes1_area = (boxes1[..., 3] - boxes1[..., 0]) * (boxes1[..., 4] - boxes1[..., 1]) * (boxes1[..., 5] - boxes1[..., 2])
    boxes2_area = (boxes2[..., 3] - boxes2[..., 0]) * (boxes2[..., 4] - boxes2[..., 1]) * (boxes2[..., 5] - boxes2[..., 2])

    inter_left_up = torch.max(boxes1[..., :3], boxes2[..., :3])
    inter_right_down = torch.min(boxes1[..., 3:], boxes2[..., 3:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = inter_area / union_area

    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :3], boxes2[..., :3])
    outer_right_down = torch.max(boxes1[..., 3:], boxes2[..., 3:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    # outer_diagonal_line = torch.pow(outer[...,0]+outer[...,1])
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2) + torch.pow(outer[..., 2], 2)
    # outer_diagonal_line = torch.sum(torch.pow(outer, 2), axis=-1)

    # cal center distance
    boxes1_center = (boxes1[..., :3] +  boxes1[...,3:]) * 0.5
    boxes2_center = (boxes2[..., :3] +  boxes2[...,3:]) * 0.5
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2) +\
                 torch.pow(boxes1_center[...,2]-boxes2_center[...,2], 2)

    # cal penalty term
    # cal width,height
    boxes1_size = torch.max(boxes1[..., 3:] - boxes1[..., :3], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 3:] - boxes2[..., :3], torch.zeros_like(inter_right_down))
    if 0:
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)
        alpha = v / (1-ious+v)

    cious = ious - center_dis / outer_diagonal_line #( + alpha*v)
    #MY MODIFICATION
    #kek = -1 + inter_area / boxes1_area # FOR yolo_loss.py, line 190
    #cious += kek
    #cious += 1 - inter_area / boxes1_area
    return cious

def IOU_xywh_torch(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # xywh->xyxy
    if boxes1.size(-1) == 6:
        boxes1 = torch.cat([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                        boxes1[..., :3] + boxes1[..., 3:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                            boxes2[..., :3] + boxes2[..., 3:] * 0.5], dim=-1)

        boxes1 = torch.cat([torch.min(boxes1[..., :3], boxes1[..., 3:]),
                            torch.max(boxes1[..., :3], boxes1[..., 3:])], dim=-1)
        boxes2 = torch.cat([torch.min(boxes2[..., :3], boxes2[..., 3:]),
                            torch.max(boxes2[..., :3], boxes2[..., 3:])], dim=-1)

        boxes1_area = (boxes1[..., 3] - boxes1[..., 0]) * (boxes1[..., 4] - boxes1[..., 1]) * (boxes1[..., 5] - boxes1[..., 2])
        boxes2_area = (boxes2[..., 3] - boxes2[..., 0]) * (boxes2[..., 4] - boxes2[..., 1]) * (boxes2[..., 5] - boxes2[..., 2])

        inter_left_up = torch.max(boxes1[..., :3], boxes2[..., :3])
        inter_right_down = torch.min(boxes1[..., 3:], boxes2[..., 3:])
    else:
        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

        boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                            torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
        boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                            torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
        inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    if boxes1.size(-1) == 6:
        inter_area = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
    else:
        inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = inter_area / (union_area+1e-3)
    #box1_nonzero = boxes1.reshape([-1, 6])
    #box1_nonzero = [k for k in box1_nonzero if k[0] > 0]
    #box2_nonzero = boxes2.reshape([-1, 6])
    #box2_nonzero = [k for k in box2_nonzero if k[0] > 0]
    return ious
def CIOU_xywh_torch(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # xywh->xyxy
    if boxes1.size(-1) == 6:
        boxes1 = torch.cat([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                        boxes1[..., :3] + boxes1[..., 3:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                            boxes2[..., :3] + boxes2[..., 3:] * 0.5], dim=-1)

        boxes1 = torch.cat([torch.min(boxes1[..., :3], boxes1[..., 3:]),
                            torch.max(boxes1[..., :3], boxes1[..., 3:])], dim=-1)
        boxes2 = torch.cat([torch.min(boxes2[..., :3], boxes2[..., 3:]),
                            torch.max(boxes2[..., :3], boxes2[..., 3:])], dim=-1)

        boxes1_area = (boxes1[..., 3] - boxes1[..., 0]) * (boxes1[..., 4] - boxes1[..., 1]) * (boxes1[..., 5] - boxes1[..., 2])
        boxes2_area = (boxes2[..., 3] - boxes2[..., 0]) * (boxes2[..., 4] - boxes2[..., 1]) * (boxes2[..., 5] - boxes2[..., 2])

        inter_left_up = torch.max(boxes1[..., :3], boxes2[..., :3])
        inter_right_down = torch.min(boxes1[..., 3:], boxes2[..., 3:])
    else:
        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

        boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                            torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
        boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                            torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
        inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    if boxes1.size(-1) == 6:
        inter_area = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
    else:
        inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / (union_area+1e-3)
    raise NotImplementedError
    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    # outer_diagonal_line = torch.pow(outer[...,0]+outer[...,1])
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)
    # outer_diagonal_line = torch.sum(torch.pow(outer, 2), axis=-1)

    # cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

    # cal penalty term
    # cal width,height
    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))
    v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)
    alpha = v / (1-ious+v)


    #cal ciou
    #ciou dosen't need square root when calculate distance?
    cious = ious - (center_dis / outer_diagonal_line + alpha*v)

    return cious


def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms', box_top_k=50):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    if bboxes.shape[-1]==8:
        classes_in_img = list(set(bboxes[:, 7].astype(np.int32)))
    else:
        classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    top_k_bboxes = []
    log_txt = "{} bboxes before nms\n".format(len(bboxes))

    if (1): # To speed up nms, pass only top 500 boxes into nms actually
        if len(bboxes) > 20:
            for idx in bboxes[:, 6].argsort()[-500:][::-1]:
                best_bbox = bboxes[idx]
                top_k_bboxes.append(best_bbox)

            bboxes = np.array(top_k_bboxes)

    classes_in_img = [_ for _ in classes_in_img if not _==0]
    #print("class in img:", classes_in_img) # only [1]
    best_bboxes = []
    score_top_k_list = []
    for cls in classes_in_img:
        if bboxes.shape[-1]==8:
            cls_mask = (bboxes[:, 7].astype(np.int32) == cls) #3d
        else:
            cls_mask = (bboxes[:, 5].astype(np.int32) == cls) #2d
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            if bboxes.shape[-1]==8:
                max_ind = np.argmax(cls_bboxes[:, 6])
            else:
                max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            #print("best_bbox", best_bbox)
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            if bboxes.shape[-1]==8:
                iou = iou_xyxy_numpy(best_bbox[np.newaxis, :6], cls_bboxes[:, :6])
            else:
                iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            if bboxes.shape[-1]==8:
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > score_threshold
            else:
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
            if len(best_bboxes) >= box_top_k:
                break
    best_bboxes = np.array(best_bboxes)
    top_k_bboxes = []
    log_txt += "{} bboxes after nms\n".format(len(best_bboxes))
    min_conf = -1
    for i in range(min(len(best_bboxes), box_top_k)):
        max_ind = np.argmax(best_bboxes[:, -2])
        best_bbox = best_bboxes[max_ind]
        top_k_bboxes.append(best_bbox)
        best_bboxes = np.concatenate([best_bboxes[: max_ind], best_bboxes[max_ind + 1:]])
        if min_conf==-1 or min_conf>best_bbox[-2]:
            min_conf=best_bbox[-2]
    log_txt += "min confidence in top {} boxes: {}".format(len(top_k_bboxes),  min_conf)
    print(log_txt)
    log_txt += "\n"
    return np.array(top_k_bboxes), log_txt


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_box(bboxes, img, id = None, color=None, line_thickness=None):
    """
    显示图片img和其所有的bboxes
    :param bboxes: [N, 5] 表示N个bbox, 格式仅支持np.array
    :param img: img格式为pytorch, 需要进行转换
    :param color:
    :param line_thickness:
    """

    img = img.permute(0,2,3,1).contiguous()[0].numpy() if isinstance(img, torch.Tensor) else img# [C,H,W] ---> [H,W,C]
    img_size, _, _ = img.shape
    bboxes[:, :4] = xywh2xyxy(bboxes[:, :4])
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, x in enumerate(bboxes):
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        label = cfg.DATA["CLASSES"][int(x[4])]
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    # cv2.imshow("img-bbox", img[:, :, ::-1])
    # cv2.waitKey(0)
    img = cv2.cvtColor(img* 255.0, cv2.COLOR_RGB2BGR).astype(np.float32)
    cv2.imwrite("../data/dataset{}.jpg".format(id), img)
