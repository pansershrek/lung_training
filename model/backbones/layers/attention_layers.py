import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################################################################
class SEModule_Conv(nn.Module):
    def __init__(self, channels, reduction=4, dims=2):
        super(SEModule_Conv, self).__init__()
        assert channels >= reduction
        if dims == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc_1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
            self.relu = nn.ReLU(inplace=True)
            self.fc_2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
            self.sigmoid = nn.Sigmoid()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc_1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
            self.relu = nn.ReLU(inplace=True)
            self.fc_2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avg_pool(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return original * x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4, dims=2):
        super(SEModule, self).__init__()
        assert channels >= reduction
        self.dims = dims
        if dims == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc_1 = nn.Linear(channels, channels // reduction)
            self.relu = nn.ReLU(inplace=True)
            self.fc_2 = nn.Linear(channels // reduction, channels)
            self.sigmoid = nn.Sigmoid()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc_1 = nn.Linear(channels, channels // reduction)
            self.relu = nn.ReLU(inplace=True)
            self.fc_2 = nn.Linear(channels // reduction, channels)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avg_pool(x).view(x.shape[0], x.shape[1]) # flatten
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        if self.dims == 3:
            x = self.sigmoid(x).view(x.shape[0], x.shape[1], 1, 1, 1) # (B,C,1,1,1)
        elif self.dims == 2:
            x = self.sigmoid(x).view(x.shape[0], x.shape[1], 1, 1)
        return original * x


###########################################################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_planes = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_planes = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True)if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1) #i.e. keep "batch" dim


def lp_pool3d(tensor, power, window, stride):
    """coded by ccy"""
    if math.isinf(power):
        return F.avg_pool3d(tensor, window, stride)
    else:
        out = F.avg_pool3d(tensor.pow(power), window, stride)
        kw, kh, kd = window
        return (torch.sign(out) * F.relu(torch.abs(out))).mul(kw * kh * kd).pow(1. / power)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], dims=2):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.__dims = dims

    def forward(self, x):
        channel_att_sum = None
        if self.__dims == 2:
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == 'lp':
                    lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size*(3)))
                    channel_att_raw = self.mlp(lp_pool)
                elif pool_type == 'lse':
                    # LSE pool
                    lse_pool = logsumexp_2d(x)
                    channel_att_raw = self.mlp(lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        elif self.__dims == 3:
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == 'lp':
                    lp_pool = lp_pool3d(x, 2, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(lp_pool)
                elif pool_type == 'lse':
                    # LSE pool
                    lse_pool = logsumexp_3d(x)
                    channel_att_raw = self.mlp(lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

def logsumexp_3d(tensor):
    return logsumexp_2d(tensor)  ## ccy: 2D/3D do same things here


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, dims=3):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool() ## ccy: 2D/3D do same things here
        if dims==2:
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        elif dims==3:
            self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, dims=3):
         super(CBAM, self).__init__()
         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, dims=dims)
         self.no_spatial = no_spatial
         if not no_spatial:
             self.SpatialGate = SpatialGate(dims=dims)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out