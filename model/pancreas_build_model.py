import torch.nn as nn
import torch
import numpy as np

from .head.yolo_head import Yolo_head
from .YOLOv4 import YOLOv4


class BuildModel(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """

    def __init__(self, weight_path=None, resume=False, dims=3):
        super().__init__()
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

        self.__anchors = torch.FloatTensor(anchors)
        self.__strides = torch.FloatTensor([4, 8, 16])
        self.__nC = 1
        self.__out_channel = 3 * (self.__nC + 7)

        self.__yolov4 = YOLOv4(
            weight_path=weight_path,
            out_channels=self.__out_channel,
            resume=resume,
            dims=dims
        )

        self.__head_s = Yolo_head(
            nC=self.__nC,
            anchors=self.__anchors[0],
            stride=self.__strides[0],
            dims=dims
        )
        # medium
        self.__head_m = Yolo_head(
            nC=self.__nC,
            anchors=self.__anchors[1],
            stride=self.__strides[1],
            dims=dims
        )
        # large
        self.__head_l = Yolo_head(
            nC=self.__nC,
            anchors=self.__anchors[2],
            stride=self.__strides[2],
            dims=dims
        )

        self.verbose = False

    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.__yolov4(x)
        if self.verbose:
            print(
                "After YOLOv4:\nx_s: {}\nx_m: {}\nx_l: {}".format(
                    x_s.shape, x_m.shape, x_l.shape
                )
            )

        out_s = self.__head_s(x_s)
        out_m = self.__head_m(x_m)
        out_l = self.__head_l(x_l)
        if self.verbose:
            print("After Yolo_heads:")
            print(
                *[m[1].shape for m in [out_s, out_m, out_l]],
                sep="\n",
                end="\n" + "=" * 20 + "\n"
            )
        out.append(out_s)
        out.append(out_m)
        out.append(out_l)

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def getNC(self):
        return self.__nC
