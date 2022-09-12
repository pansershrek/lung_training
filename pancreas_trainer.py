import os
from copy import deepcopy
import random

from apex import amp
from monai.visualize import plot_2d_or_3d_image
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.pancreas_build_model import BuildModel
from model.loss.yolo_loss import YoloV4Loss
from utils import cosine_lr_scheduler
from utils_ABUS.misc import build_threshold
from utils_ABUS.postprocess import centroid_distance, eval_precision_recall_by_dist, eval_precision_recall
from utils.tools import nms, xyzwhd2xyzxyz


def interpolate_FROC_data(
    froc_x, froc_y, max_fps=(8, 4, 2, 1, 0.5, 0.25, 0.125)
):
    assert max_fps[0] == 8
    y_interpolate = 0
    take_i = 0
    max_fps = sorted(
        max_fps, reverse=True
    )  # e.g. [8, 4, 2, 1, 0.5, 0.25, 0.125]
    max_fp = max_fps.pop(0)
    take_is = []
    sens_for_cpm = {}
    for i in range(len(froc_x)):
        FP = froc_x[i]
        sen = froc_y[i]
        while FP <= max_fp:
            take_i = i
            take_is.append(take_i)
            x1 = FP
            y1 = froc_y[i]
            if i > 0:
                x2 = froc_x[i - 1]
                y2 = froc_y[i - 1]

                x_interpolate = max_fp
                if x2 - x1 > 0.001:
                    y_interpolate = (
                        y1 * (x2 - x_interpolate) + y2 * (x_interpolate - x1)
                    ) / (x2 - x1)
                else:  #nan or inf
                    y_interpolate = 0
            else:
                #if no data point for FP > 8
                #use sensitivity at FP = FP_small
                y_interpolate = y1
            sens_for_cpm[max_fp] = y_interpolate
            if len(max_fps) == 0:
                break
            else:
                max_fp = max_fps.pop(0)
        if len(max_fps) == 0 and (max_fp in sens_for_cpm):
            break
    else:
        for max_fp in [max_fp] + list(max_fps):
            sens_for_cpm[max_fp] = 0  # since no prediction can get this max_fp
    if len(take_is) > 0:
        froc_x = froc_x[take_is[0]:]
        froc_y = froc_y[take_is[0]:]

    if not froc_x[0] == 8:
        froc_x = np.insert(froc_x, 0, 8)
        froc_y = np.insert(froc_y, 0, sens_for_cpm[8])
    cpm = sum(sens_for_cpm.values()) / len(sens_for_cpm)
    return froc_x, froc_y, cpm, sens_for_cpm


def froc_take_max(froc_x, froc_y):
    froc_x_tmp = []
    froc_y_tmp = []
    for i in range(len(froc_x)):
        if i == 0 or froc_x_tmp[-1] > froc_x[i]:
            froc_x_tmp.append(froc_x[i])
            froc_y_tmp.append(froc_y[i])
    froc_x = np.array(froc_x_tmp)
    froc_y = np.array(froc_y_tmp)
    return froc_x, froc_y


def AUC(froc_x, froc_y, normalize=False):
    froc_x = np.array(froc_x)
    froc_y = np.array(froc_y)

    area = np.trapz(froc_y[::-1], x=froc_x[::-1], dx=0.001)

    if normalize:
        return area / np.max(froc_x[::-1])
    else:
        return area


class Trainer:

    def _setup_seed(self, seed=1717):
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def __init__(
        self,
        train_dataset,
        val_dataset,
        inference_dataset,
        checkpoint_save_dir,
        writer,
        logger,
        device,
        epochs,
        batch_size=4,
        opt_level="O0",
        inference_model_path="",
        val_interval=1,
        seed=1717,
    ):
        self._setup_seed(seed)
        self.device = device
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            pin_memory=False
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=False
        )
        self.inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=False
        )
        self.inference_to_store = inference_dataset.labels_dir
        self.scale_function = inference_dataset.scale_bbox
        self.checkpoint_save_dir = checkpoint_save_dir
        self.writer = writer
        self.logger = logger
        self.epochs = epochs
        self.inference_model_path = inference_model_path
        self.val_interval = val_interval
        self.opt_level = opt_level
        self.image_size = val_dataset.image_size

        self.model = BuildModel(weight_path=None, resume=False, dims=3)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)

        anchors = [
            [
                [14 / 8., 27 / 8., 28 / 8.],
                [17 / 8., 37 / 8., 36 / 8.],
                [21 / 8., 51 / 8., 52 / 8.],
            ],
            [
                [26 / 16., 39 / 16., 40 / 16.],
                [33 / 16., 50 / 16., 48 / 16.],
                [34 / 16., 62 / 16., 64 / 16.],
            ],
            [
                [48 / 32., 57 / 32., 58 / 32.],
                [57 / 32., 73 / 32., 72 / 32.],
                [79 / 32., 90 / 32., 91 / 32.],
            ],
        ]
        anchors = np.array(anchors)
        strides = np.array([4, 8, 16])

        self.criterion = YoloV4Loss(
            anchors=anchors, strides=strides, iou_threshold_loss=0.5, dims=3
        )
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=5e-3,
            lr_min=5e-8,
            warmup=5 * len(self.train_dataloader)
        )

    def _save_model_weights(self, epoch):
        chkpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(
            chkpt,
            os.path.join(self.checkpoint_save_dir, f'checkpoint_{epoch}.pt')
        )

    def _load_model_weights(self):
        chkpt = torch.load(self.inference_model_path)
        self.model.load_state_dict(chkpt["model"])

    def train(self):
        self.logger.info("Start to train model")
        self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level=self.opt_level
        )
        for epoch in range(self.epochs):
            self.model.train()
            mloss = torch.zeros(5)
            self.logger.info(f"Train epoch: {epoch}")
            for idx, data in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader) * epoch + idx)

                p, p_d = self.model(data["images"].to(self.device))

                loss, loss_ciou, loss_conf, loss_cls = self.criterion(
                    p, p_d, data["label_sbbox"].to(self.device),
                    data["label_mbbox"].to(self.device),
                    data["label_lbbox"].to(self.device), data["sbboxes"].to(
                        self.device
                    ), data["mbboxes"].to(self.device),
                    data["lbboxes"].to(self.device)
                )

                #loss.backward()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                conf_data = p_d[0][..., 6:7].detach().cpu().numpy().flatten()
                pr999_p_conf = np.sort(conf_data)[-8]
                loss_items = torch.tensor(
                    [loss_ciou, loss_conf, loss_cls, loss, pr999_p_conf]
                )
                mloss = (mloss * idx + loss_items) / (idx + 1)

                if self.writer:
                    self.writer.add_scalar(
                        'train/loss_ciou', mloss[0],
                        len(self.train_dataloader) * epoch + idx
                    )
                    self.writer.add_scalar(
                        'train/loss_conf', mloss[1],
                        len(self.train_dataloader) * epoch + idx
                    )
                    self.writer.add_scalar(
                        'train/loss_cls', mloss[2],
                        len(self.train_dataloader) * epoch + idx
                    )
                    self.writer.add_scalar(
                        'train/train_loss', mloss[3],
                        len(self.train_dataloader) * epoch + idx
                    )
                    self.writer.add_scalar(
                        'train/train_pr99.9_p_conf', mloss[4],
                        len(self.train_dataloader) * epoch + idx
                    )
                    self.writer.add_scalar(
                        'train/train_lr', self.optimizer.param_groups[0]["lr"],
                        len(self.train_dataloader) * epoch + idx
                    )
            self._save_model_weights(epoch)

            if (epoch + 1) % self.val_interval == 0:
                (
                    area_dist, area_iou, cpm_dist, cpm, max_sens_dist,
                    max_sens_iou
                ) = self.validate()
                if self.writer:
                    self.writer.add_scalar('val/AUC_(IOU)', area_iou, epoch)
                    self.writer.add_scalar('val/CPM_(IOU)', cpm, epoch)
                    self.writer.add_scalar('val/AUC_(dist)', area_dist, epoch)
                    self.writer.add_scalar('val/CPM_(dist)', cpm_dist, epoch)
                    self.writer.add_scalar(
                        'val/Max_sens(iou)', max_sens_iou, epoch
                    )
                    self.writer.add_scalar(
                        'val/Max_sens(dist)', max_sens_dist, epoch
                    )

    def validate(self):
        self.logger.info("Start to validate model")

        gt_lut = {}
        pred_lut = {}
        with torch.no_grad():
            for idx, data in enumerate(self.val_dataloader):
                bboxes_prd, box_raw_data, bboxes_prd_no_nms = self._get_bbox(
                    data["images"]
                )

                gt_lut[data["names"][0]] = data["bboxes"]
                pred_lut[data["names"][0]] = bboxes_prd
        (
            area_dist, area_iou, sub_log_txt, cpm_dist, cpm, max_sens_dist,
            max_sens_iou, fp_bboxes_all_pid
        ) = self._calculate_FROC(
            gt_lut,
            pred_lut,
            size_threshold=20,
            th_step=0.01,
            det_tp_iou_thresh=0.3,
            return_fp_bboxes=True
        )
        return (
            area_dist, area_iou, cpm_dist, cpm, max_sens_dist, max_sens_iou
        )

    def inference(self):
        self.logger.info("start to inference model")
        self._load_model_weights()

        with torch.no_grad():
            for idx, data in enumerate(self.inference_dataloader):
                bboxes_prd, box_raw_data, bboxes_prd_no_nms = self._get_bbox(
                    data["images"]
                )
                plot_2d_or_3d_image(
                    data=data["images"][0],
                    step=idx,
                    writer=self.writer,
                    frame_dim=-1,
                    tag="image"
                )
                bbox_original = torch.zeros_like(data["images"][0])
                bbox_original[data["bboxes"][0]:data["bboxes"][3],
                              data["bboxes"][1]:data["bboxes"][4],
                              data["bboxes"][2]:data["bboxes"][5]] = 1
                plot_2d_or_3d_image(
                    data=bbox_original,
                    step=idx,
                    writer=self.writer,
                    frame_dim=-1,
                    tag="original"
                )
                with open(
                    os.path.join(
                        self.inference_to_store, f'pred_{data["names"][0]}'
                    ), "w"
                ) as f:
                    for idx_bbox, bbox in enumerate(bboxes_prd):
                        bbox_tmp = self.scale_function(
                            self.image_size, data["original_size"][0], bbox[:6]
                        )
                        if idx_bbox == 0:
                            bbox_predict = torch.zeros_like(data["images"][0])
                            bbox_predict[bbox_tmp[0]:bbox_tmp[3],
                                         bbox_tmp[1]:bbox_tmp[4],
                                         bbox_tmp[2]:bbox_tmp[5]] = 1
                            plot_2d_or_3d_image(
                                data=bbox_predict,
                                step=idx,
                                writer=self.writer,
                                frame_dim=-1,
                                tag="predict"
                            )
                        print(*bbox_tmp, bbox[6], bbox[7], file=f, flush=True)

    def _get_bbox(self, image):
        bboxes, box_raw_data = self._predict(image)
        boxes_no_nms = deepcopy(bboxes)
        bboxes, _ = nms(
            bboxes, score_threshold=0.015, iou_threshold=0.15, box_top_k=512
        )
        return bboxes, box_raw_data, boxes_no_nms

    def _predict(self, image):
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(image.to(self.device))
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self._convert_pred(pred_bbox)
        return bboxes, p_d.cpu()

    def _calculate_FROC(
        self,
        gt_lut,
        pred_lut,
        size_threshold=20,
        th_step=0.01,
        det_tp_iou_thresh=0.3,
        return_fp_bboxes=True
    ):
        all_thre = build_threshold(th_step)
        PERF_per_thre = []
        PERF_per_thre_s = []
        true_num, true_small_num = 0, 0
        box_lists_cacher = {}
        fp_bboxes_all_pid = {}

        for i, score_hit_thre in enumerate(all_thre):
            (
                TP_table, FP_table, FN_table, TP_table_IOU_1, FP_table_IOU_1,
                FN_table_IOU_1, pred_num, pred_small_num, file_table, iou_table
            ) = [], [], [], [], [], [], [], [], [], []
            TP_table_dist, FP_table_dist, FN_table_dist = [], [], []

            current_pass = 0

            for pid, boxes in gt_lut.items():
                true_box = boxes
                true_box_s = true_box
                if i == 0:
                    true_num += len(true_box)
                    true_small_num += len(true_box_s)
                out_boxes = []

                box_list = pred_lut[pid]
                for bx in box_list:  #postprocessing, filtering bbox
                    axis = [0, 0, 0]
                    axis[0] = (bx[3] - bx[0]) / 1.  #/ 4
                    axis[1] = (bx[4] - bx[1]) / 1.  #/ 4
                    axis[2] = (bx[5] - bx[2]) / 1.  #/ 4
                    ct = [0, 0, 0]
                    ct[0] = (bx[3] + bx[0]) / 2
                    ct[1] = (bx[4] + bx[1]) / 2
                    ct[2] = (bx[5] + bx[2]) / 2
                    #print(bx)
                    #pass
                    if bx[
                        6
                    ] >= score_hit_thre:  # and (not check_boundary(ct)) : #and check_size(axis, size_threshold):
                        out_boxes.append(list(bx))

                pred_num.append(len(out_boxes))

                extra_tp = None
                TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall(
                    out_boxes,
                    true_box,
                    det_thresh=det_tp_iou_thresh,
                    scale=(1., 1., 1.),
                    extra_tp=extra_tp
                )
                TP_dist, FP_dist, FN_dist, hits_index_dist, hits_dist, hits_score_dist, TP_by_size, fp_bboxes = eval_precision_recall_by_dist(
                    out_boxes,
                    true_box,
                    dist_thresh=None,
                    scale=(1., 1., 1.),
                    spacing=(1.25, 0.75, 0.75),
                    return_fp_bboxes=True,
                    extra_tp=extra_tp
                )
                if return_fp_bboxes and score_hit_thre == 0.0:
                    fp_bboxes_all_pid[pid] = fp_bboxes
                TP_table.append(TP)
                FP_table.append(FP)
                FN_table.append(FN)

                TP_table_dist.append(TP_dist)
                FP_table_dist.append(FP_dist)
                FN_table_dist.append(FN_dist)

            TP_table_sum = np.array(TP_table)
            FP_table_sum = np.array(FP_table)
            FN_table_sum = np.array(FN_table)

            TP_table_sum_dist = np.array(TP_table_dist)
            FP_table_sum_dist = np.array(FP_table_dist)
            FN_table_sum_dist = np.array(FN_table_dist)

            sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(
            ), FN_table_sum.sum()
            sensitivity = sum_TP / (sum_TP + sum_FN + 1e-10)
            precision = sum_TP / (sum_TP + sum_FP + 1e-10)

            sum_TP_dist, sum_FP_dist, sum_FN_dist = TP_table_sum_dist.sum(
            ), FP_table_sum_dist.sum(), FN_table_sum_dist.sum()
            sensitivity_dist = sum_TP_dist / (
                sum_TP_dist + sum_FN_dist + 1e-10
            )
            precision_dist = sum_TP_dist / (sum_TP_dist + sum_FP_dist + 1e-10)

            PERF_per_thre.append(
                [
                    score_hit_thre,
                    len(gt_lut),  #total_pass,
                    sensitivity,  # ---using iou---
                    precision,  # ---using iou---
                    #sum_FP / total_pass,  # ---using iou---
                    sum_FP / len(gt_lut),
                    sensitivity_dist,  # ---using dist---
                    precision_dist,  # ---using dist---
                    sum_FP_dist / len(gt_lut)  #sum_FP_dist / total_pass
                ],  # ---using dist---
            )

        data = np.array(PERF_per_thre)
        max_sens_iou = PERF_per_thre[0][2]
        max_sens_dist = PERF_per_thre[0][5]

        area_iou, area_dist, cpm, cpm_dist = 0, 0, 0, 0

        if len(data) == 0:
            area = 0
        else:
            area_dist = 0.0  #prevent error
            froc_x_dist, froc_y_dist, cpm_dist, sens_for_cpm_dist = interpolate_FROC_data(
                data[..., 7],
                data[..., 5],
                max_fps=(8, 4, 2, 1, 0.5, 0.25, 0.125)
            )
            froc_x_dist, froc_y_dist = froc_take_max(froc_x_dist, froc_y_dist)
            area_dist = AUC(froc_x_dist, froc_y_dist, normalize=True)

            froc_x, froc_y, cpm, sens_for_cpm = interpolate_FROC_data(
                data[..., 4],
                data[..., 2],
                max_fps=(8, 4, 2, 1, 0.5, 0.25, 0.125)
            )
            froc_x, froc_y = froc_take_max(froc_x, froc_y)
            area_iou = AUC(froc_x, froc_y, normalize=True)

        if return_fp_bboxes:
            return area_dist, area_iou, "", cpm_dist, cpm, max_sens_dist, max_sens_iou, fp_bboxes_all_pid
        else:
            return area_dist, area_iou, "", cpm_dist, cpm, max_sens_dist, max_sens_iou

    def _convert_pred(self, pred_bbox):
        """
        Filter out the prediction box to remove the unreasonable scale of the box
        """
        pred_coor = xyzwhd2xyzxyz(pred_bbox[:, :6])
        pred_conf = pred_bbox[:, 6]
        pred_prob = pred_bbox[:, 7:]
        org_d, org_h, org_w = self.image_size

        # (2)Crop off the portion of the predicted Bbox that is beyond the original image
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :3], [0, 0, 0]),
                np.minimum(
                    pred_coor[:, 3:], [org_d - 1, org_h - 1, org_w - 1]
                )
            ],
            axis=-1
        )
        # (3)Sets the coor of an invalid bbox to 0
        invalid_mask = np.logical_or(
            (pred_coor[:, 1] > pred_coor[:, 4]),
            (pred_coor[:, 2] > pred_coor[:, 5])
        )
        pred_coor[invalid_mask] = 0
        invalid_mask = (pred_coor[:, 0] > pred_coor[:, 3])
        pred_coor[invalid_mask] = 0

        # (4)Remove bboxes that are not in the valid range
        bboxes_scale = np.multiply.reduce(
            pred_coor[:, 3:6] - pred_coor[:, 0:3], axis=-1
        )
        v_scale_3 = np.power((0, np.inf), 3.0)
        scale_mask = np.logical_and(
            (v_scale_3[0] < bboxes_scale), (bboxes_scale < v_scale_3[1])
        )

        # (5)Remove bboxes whose score is below the score_threshold
        classes = np.argmax(pred_prob, axis=-1)  #predicted class idx
        scores = pred_conf * pred_prob[
            np.arange(len(pred_coor)), classes
        ]  # score = pred_prob_from_class * pred_conf (p.s. it only used sigmoid, no softmax in YoloHead)

        score_mask = scores > 0.015

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate(
            [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1
        )
        return bboxes
