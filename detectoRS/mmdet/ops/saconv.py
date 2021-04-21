import torch
import sys
try:
    from .conv_ws import ConvAWS2d, ConvAWS3d
except:
    sys.path.append("D:/CH/LungDetection/training/detectoRS/mmdet/ops")
    from conv_ws import ConvAWS2d, ConvAWS3d
#from .dcn import deform_conv #original, but support 2D cuda only

sys.path.append("D:/CH/LungDetection/training/detectoRS/D3Dnet/code/dcn/functions")
from deform_conv_func import deform_conv_3d
sys.path.append("D:/CH/LungDetection/training/config")
import yolov4_config as cfg


class SAConv2d(ConvAWS2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=False):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.use_deform = use_deform
        self.switch = torch.nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=stride,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
        if self.use_deform:
            self.offset_s = torch.nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_l = torch.nn.Conv2d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_s.weight.data.fill_(0)
            self.offset_s.bias.data.fill_(0)
            self.offset_l.weight.data.fill_(0)
            self.offset_l.bias.data.fill_(0)

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        if self.use_deform:
            raise NotImplementedError()
            offset = self.offset_s(avg_x)
            out_s = deform_conv(
                    x,
                    offset,
                    weight,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    1)
        else:
            out_s = super()._conv_forward(x, weight)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        if self.use_deform:
            raise NotImplementedError()
            offset = self.offset_l(avg_x)
            out_l = deform_conv(
                    x,
                    offset,
                    weight,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    1)
        else:
            out_l = super()._conv_forward(x, weight)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out



class SAConv3d(ConvAWS3d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 use_deform=cfg.MODEL["SACONV_USE_DEFORM"]):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.use_deform = use_deform
        self.switch = torch.nn.Conv3d(
            self.in_channels,
            1,
            kernel_size=1,
            stride=stride,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.pre_context = torch.nn.Conv3d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)
        if self.use_deform:
            self.offset_s = torch.nn.Conv3d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_l = torch.nn.Conv3d(
                self.in_channels,
                18,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True)
            self.offset_s.weight.data.fill_(0)
            self.offset_s.bias.data.fill_(0)
            self.offset_l.weight.data.fill_(0)
            self.offset_l.bias.data.fill_(0)

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool3d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        # avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2, 2, 2), mode="reflect") # original, but 3d case not supported
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2, 2, 2), mode="circular") # ccy: use circular instead
        avg_x = torch.nn.functional.avg_pool3d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        weight = self._get_weight(self.weight)
        if self.use_deform:
            offset = self.offset_s(avg_x)
            out_s = deform_conv_3d(
                    x,
                    offset,
                    weight,
                    None, # bias==None
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    1,
                    8,) # maybe need batch_size % ctx.im2col_step == 0
        else:
            out_s = super()._conv_forward(x, weight)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        if self.use_deform:
            offset = self.offset_l(avg_x)
            out_l = deform_conv_3d(
                    x,
                    offset,
                    weight,
                    None, #bias == None
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    1,
                    8,) # maybe need batch_size % ctx.im2col_step == 0
        else:
            out_l = super()._conv_forward(x, weight)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool3d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out


if __name__ == "__main__":
    device = "cuda"
    if (1):
        conv = SAConv3d(1, 3, 3, padding=1).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(conv.parameters())
        for i in range(5):
            t = torch.ones((1,1,10,128,128), device=device)
            out = conv(t)
            loss = criterion(out, torch.cat([t]*3, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
    if (0):
        conv = SAConv2d(1, 3, 3, padding=1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(conv.parameters())
        for i in range(5):
            t = torch.randn((2,1,10,10), device=device)
            out = conv(t)
            loss = criterion(out, torch.cat([t]*3, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
