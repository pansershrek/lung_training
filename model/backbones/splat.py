"""
Split-Attention (3D)
modified from: https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv3d, Module, Linear, BatchNorm3d, ReLU
from torch.nn.modules.utils import _pair
import sys
sys.path.append("/workspace/pancreas/detectoRS/mmdet/ops")
from saconv import SAConv3d
__all__ = ['SplAtConv3d']

class SplAtConv3d(Module):
    """Split-Attention Conv2d -> 3d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, 
                 use_SAConv=False,
                 **kwargs):
        super(SplAtConv3d, self).__init__()
        ##padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        default_conv = SAConv3d if use_SAConv else Conv3d
        if self.rectify:
            raise NotImplementedError("Rectified convolution not implemented")
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = default_conv(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv3d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv3d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            raise NotImplementedError("DropBlock not implemented")
            self.dropblock = DropBlock3D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn and gap.shape[0]>1:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


def _test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = SplAtConv3d(10, 3, (3,3,3), norm_layer=BatchNorm3d).to(device)
    inp = torch.randn((2, 10, 128, 128, 128), device=device)
    out = m(inp)
    print(out.shape)

if __name__ == "__main__":
    _test()
