import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

import utils.data_augment as dataAug
import utils.tools as tools
import utils_ccy as utils


class PancreasDataset(Dataset):

    def __init__(
        self,
        images_dir,
        labels_dir,
        validate=False,
        image_size=(128, 128, 128),
        cache_size=40
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.meta_data = {}
        self.classes = set()

        self.validate = validate

        self.cacher = utils.LRUCache(cache_size=cache_size)
        self.image_size = image_size

        for file in os.listdir(self.labels_dir):
            with open(os.path.join(self.labels_dir, file), "r") as f:
                data = [int(x) for x in f.read().strip().split(" ")]
                self.meta_data[len(self.meta_data)] = {
                    "name": file,
                    "class": (
                        data[0]
                    ),  # Background - 0, pancrease - 1
                    "bbox": data[1:]  # BBox format is [z1,y1,x1,z2,y2,x2]
                }
                self.classes.add(data[0])
        for file in os.listdir(self.images_dir):
            name = file.replace("nii.gz", "txt")
            flag = False
            for value in self.meta_data.values():
                if name == value["name"]:
                    flag = True
                    break
            if not flag:
                self.meta_data[len(self.meta_data)] = {
                    "name": name,
                    "class": -1,
                    "bbox": None,
                }

        self.num_classes = 2

    def __len__(self):
        return len(self.meta_data)

    def get_classes(self):
        return sorted(list(self.classes))

    def scale_bbox(self, original_shape, target_shape, bbox):
        if original_shape == target_shape:
            return bbox
        else:
            z_original, y_original, x_original = original_shape
            z_new, y_new, x_new = target_shape
            z_scale, y_scale, x_scale = z_new / z_original, y_new / y_original, x_new / x_original
            z1, y1, x1, z2, y2, x2 = bbox
            return [
                round(float(z1) * float(z_scale)),
                round(float(y1) * float(y_scale)),
                round(float(x1) * float(x_scale)),
                round(float(z2) * float(z_scale)),
                round(float(y2) * float(y_scale)),
                round(float(x2) * float(x_scale))
            ]

    def _zyxzyx2zyxdhw_normalize(self, bbox):
        z = (bbox[3] + bbox[0]) / 2.0
        y = (bbox[4] + bbox[1]) / 2.0
        x = (bbox[5] + bbox[2]) / 2.0
        d = (bbox[3] - bbox[0])
        h = (bbox[4] - bbox[1])
        w = (bbox[5] - bbox[2])
        return [
            z / self.image_size[0],
            y / self.image_size[1],
            x / self.image_size[2],
            d / self.image_size[0],
            h / self.image_size[1],
            w / self.image_size[2],
        ]

    def __getitem__(self, idx):
        output, exist = self.cacher.get(idx)
        if exist:
            image = output
        else:
            image_name = self.meta_data[idx]["name"].replace("txt", "nii.gz")
            image = nib.load(os.path.join(self.images_dir, image_name)).get_fdata()
            self.cacher.set(idx, image)
        original_size = image.shape
        bboxes = None
        if self.meta_data[idx]["bbox"] is not None:
            if not self.validate:
                image, bboxes = self.__data_aug(
                    image,
                    torch.tensor(self.meta_data[idx]["bbox"]).unsqueeze(0)
                )
                bboxes = [x for x in bboxes[0]]
                bboxes = self.scale_bbox(image.shape, self.image_size, bboxes)
            else:
                bboxes = self.scale_bbox(
                    image.shape, self.image_size, self.meta_data[idx]["bbox"]
                )
        image = utils.resize_without_pad(
            image, self.image_size, "trilinear", align_corners=False
        )
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = (
            [], [], [], [], [], []
        )
        eval_flag = 1
        if bboxes is not None:
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self._creat_label(
                [np.array(bboxes + [self.meta_data[idx]["class"], 1, 1])],
                self.image_size
            )
            eval_flag = 0
        else:
            bboxes = []
        image = torch.FloatTensor(image)
        image = image.view(1, *image.shape)
        output = {
            "names": self.meta_data[idx]["name"],
            "images": image,
            "bboxes": torch.FloatTensor(bboxes),
            "classes": self.meta_data[idx]["class"],
            "label_sbbox": torch.FloatTensor(label_sbbox),
            "label_mbbox": torch.FloatTensor(label_mbbox),
            "label_lbbox": torch.FloatTensor(label_lbbox),
            "sbboxes": torch.FloatTensor(sbboxes),
            "mbboxes": torch.FloatTensor(mbboxes),
            "lbboxes": torch.FloatTensor(lbboxes),
            "original_size": torch.FloatTensor(original_size),
            "eval_flag": torch.FloatTensor(eval_flag),
        }

        return output

    def __data_aug(self, img, bboxes):
        img, bboxes = dataAug.RandomHorizontalFlip()(
            np.copy(img), np.copy(bboxes)
        )
        img, bboxes = dataAug.RandomVerticalFlip()(
            np.copy(img), np.copy(bboxes)
        )
        #img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        #img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))

        return img, bboxes

    def _creat_label(self, bboxes, img_size):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "yxhw"; and scale bbox'
           yxhw by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """
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
        anchors_per_scale = 3
        d_xyz = 6
        d_cls = 2
        label = [
            np.zeros(
                (
                    int(img_size[0] / strides[i]), int(
                        img_size[1] / strides[i]
                    ), int(img_size[2] / strides[i]), anchors_per_scale,
                    d_xyz + d_cls + self.num_classes
                )
            ) for i in range(3)
        ]
        for i in range(3):
            label[i][..., 7] = 1.0  # assign all **mix** to 1.0

        bboxes_yxhw = [
            np.zeros((15, 6)) for _ in range(3)
        ]  # Darknet the max_num is 30
        bbox_count = np.zeros((3, ))

        for bbox in bboxes:
            bbox_coor = bbox[:6]
            bbox_class_ind = int(bbox[6])
            if len(bbox) >= 8:
                bbox_mix = bbox[7]
            else:
                bbox_mix = None

            # onehot (on classes)
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "zyxzyx" to "zyxdhw"
            bbox_yxhw = np.concatenate(
                [
                    (bbox_coor[3:] + bbox_coor[:3]) *
                    0.5,  # center_z, center_y, center_x
                    bbox_coor[3:] - bbox_coor[:3]
                ],
                axis=-1
            )  # d, h, w

            # get bbox per scale: (1,6) / (3,1) -> (3,6) == (scale, zyxdhw)
            bbox_yxhw_scaled = 1.0 * bbox_yxhw[
                np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):  # for 3 different scales
                anchors_yxhw = np.zeros((anchors_per_scale, 6))
                anchors_yxhw[:,
                             0:3] = np.floor(bbox_yxhw_scaled[i, 0:3]).astype(
                                 np.int32
                             ) + 0.5  # 0.5 for compensation
                anchors_yxhw[:, 3:6] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(
                    bbox_yxhw_scaled[i][np.newaxis, :], anchors_yxhw
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    zind, yind, xind = np.floor(bbox_yxhw_scaled[i, 0:3]
                                                ).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][zind, yind, xind, iou_mask, 0:6] = bbox_yxhw
                    label[i][zind, yind, xind, iou_mask, 6:7] = 1.0

                    label[i][zind, yind, xind, iou_mask, 7:8] = bbox_mix
                    label[i][zind, yind, xind, iou_mask, 8:] = one_hot_smooth

                    #bbox_ind = int(bbox_count[i] % 15)  # BUG : 150为一个先验值,内存消耗大
                    bbox_ind = int(bbox_count[i])
                    bboxes_yxhw[i][bbox_ind, :6] = bbox_yxhw
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                zind, yind, xind = np.floor(
                    bbox_yxhw_scaled[best_detect, 0:3]
                ).astype(np.int32)

                if zind == label[best_detect].shape[
                    0
                ]:  # if zind >= label.shape[z] (>= or ==都可以) (to avoid error)
                    zind -= 1
                label[best_detect][zind, yind, xind, best_anchor,
                                   0:6] = bbox_yxhw
                label[best_detect][zind, yind, xind, best_anchor, 6:7] = 1.0
                label[best_detect][zind, yind, xind, best_anchor,
                                   7:8] = bbox_mix
                label[best_detect][zind, yind, xind, best_anchor,
                                   8:] = one_hot_smooth

                #bbox_ind = int(bbox_count[best_detect] % 15)
                bbox_ind = int(bbox_count[best_detect])
                bboxes_yxhw[best_detect][bbox_ind, :6] = bbox_yxhw
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_yxhw

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
