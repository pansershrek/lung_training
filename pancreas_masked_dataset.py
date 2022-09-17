import os
import math
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
#from torchvision.ops import masks_to_boxes
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)

import utils.data_augment as dataAug
import utils.tools as tools
import utils_ccy as utils


class PancreasMaskedDataset(Dataset):

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
            self.meta_data[len(self.meta_data)] = {
                "name": file,
                "class": 1,  # Background - 0, pancrease - 1
                #"bbox": data[1:]  # BBox format is [z1,y1,x1,z2,y2,x2]
            }
            self.classes.add(1)

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

    def __getitem__(self, idx):
        output, exist = self.cacher.get(idx)
        if exist:
            return output
            #data_dict = exist
            #image, label = exist
        else:
            image_name = self.meta_data[idx]["name"].replace(
                "pancreas_mask", "image"
            )
            #image = nib.load(os.path.join(self.images_dir, image_name)).get_fdata()
            #label = nib.load(
            #    os.path.join(self.labels_dir, self.meta_data[idx]["name"])
            #).get_fdata()
            data_dict = {
                "image": os.path.join(self.images_dir, image_name),
                "label":
                os.path.join(self.labels_dir, self.meta_data[idx]["name"])
            }
            loader = LoadImaged(keys=("image", "label"), image_only=False)
            data_dict = loader(data_dict)
            #self.cacher.set(idx, data_dict)

        original_size = data_dict["image"].shape

        #if not self.validate:
        #    ensure_channel_first = EnsureChannelFirstd(keys=["image", "label"])
        #    data_dict = {"image": image, label: "label"}
        #    data_dict = ensure_channel_first(data_dict)
        #    data_dict = self.__data_aug_3d(data_dict)
        #    image = data_dict["image"].squeeze(0)
        #    label = data_dict["label"].squeeze(0)
        #else:
        #    image = data_dict["image"]
        #    label = data_dict["label"]
        image = torch.tensor(data_dict["image"])
        label = torch.tensor(data_dict["label"])

        # For mdc
        if not self.validate:
            image = image.permute((2, 0, 1))
            label = label.permute((2, 0, 1))

        bboxes = self._create_bbox(label)

        if bboxes is not None:
            #if not self.validate:
            #    image, bboxes = self.__data_aug(
            #        image,
            #        torch.tensor(bboxes).unsqueeze(0)
            #    )
            #    bboxes = [x for x in bboxes[0]]
            bboxes = self.scale_bbox(image.shape, self.image_size, bboxes)
        image = np.copy(image)
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

        self.cacher.set(idx, output)
        return output

    def _create_bbox(self, label):
        bbox_3d = [
            # Format channel min, height min, width min,
            # channel max, height max, width max.
            math.inf,
            math.inf,
            math.inf,
            -math.inf,
            -math.inf,
            -math.inf
        ]
        for idx, label_slice in enumerate(label):
            label_slice[label_slice != 0] = 1
            label_slice = label_slice.int()
            if 1 in label_slice:
                bbox = self._masks_to_boxes(
                    label_slice.view(
                        1, label_slice.shape[0], label_slice.shape[1]
                    )
                )
                bbox = bbox.int().tolist()[0]
                bbox_3d[0] = min(bbox_3d[0], idx)
                bbox_3d[3] = max(bbox_3d[3], idx)
                bbox_3d[1] = min(bbox_3d[1], bbox[0])
                bbox_3d[2] = min(bbox_3d[2], bbox[1])
                bbox_3d[4] = max(bbox_3d[4], bbox[2])
                bbox_3d[5] = max(bbox_3d[5], bbox[3])
        return bbox_3d

    def _masks_to_boxes(self, masks):
        n = masks.shape[0]

        bounding_boxes = torch.zeros(
            (n, 4), device=masks.device, dtype=torch.float
        )

        for index, mask in enumerate(masks):
            x, y = torch.where(mask != 0)

            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

        return bounding_boxes

    def __data_aug_3d(self, data_dict):
        spatial_size = data_dict["image"].shape[1:]
        if random.random() < 0.5:
            rand_affine = RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=spatial_size,
                translate_range=(0, 0, 0),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            )
            #rand_affine.set_random_state(seed=1717)
            data_dict = rand_affine(data_dict)
        if random.random() < 0.5:
            rand_elastic = Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                spatial_size=spatial_size,
                translate_range=(0, 0, 0),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            )
            #rand_elastic.set_random_state(seed=1717)
            data_dict = rand_elastic(data_dict)

        return data_dict

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
