##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNet variants (3d)
modified from: https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnet.py
"""
import math
import torch
import torch.nn as nn

import sys
sys.path.append("/workspace/pancreas/detectoRS/mmdet/ops")
from saconv import SAConv3d

try:
    from .splat import SplAtConv3d
    from model.layers.attention_layers import SEModule, SEModule_Conv, CBAM
except (ImportError, ModuleNotFoundError):
    from splat import SplAtConv3d
    sys.path.append("/workspace/pancreas")
    from model.layers.attention_layers import SEModule, SEModule_Conv, CBAM
    

__all__ = ['ResNet', 'Bottleneck']

class DropBlock3D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool3d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool3d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False,
                 use_SAConv=False,
                 extra_attention=None):
        self.inplanes = inplanes
        self.planes = planes
        self.cardinality = cardinality
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        default_conv = SAConv3d if use_SAConv else nn.Conv3d # use SAConv3d only for kernel_size = 3
        self.conv1 = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        # extra attention
        self.attention = extra_attention
        out_channels = planes * self.expansion
        if self.attention == 'SEnet':self.attention_module = SEModule(out_channels, dims=3, reduction=16) # lower reduction to avoid OOM during testing
        elif self.attention == 'SEnetConv':self.attention_module = SEModule_Conv(out_channels, dims=3, reduction=16)
        elif self.attention == 'CBAM':self.attention_module = CBAM(out_channels, dims=3)
        else: self.attention = None

        if self.avd:
            self.avd_layer = nn.AvgPool3d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock3D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock3D(dropblock_prob, 3)
            self.dropblock3 = DropBlock3D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv3d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,
                use_SAConv=False) # ccy: True has super low cpm
        elif rectified_conv:
            raise NotImplementedError("Rectified convolution not implemented")
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = default_conv(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv3d(
            group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*self.expansion)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        #print("bottleneck x.shape", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.attention is not None:
            out = self.attention_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm3d, 
                 used_for_yolo=True, # if True, not using adaptivepool/fc and return feature maps of 3 different scales
                 in_channel=3, # original
                 ## no need stem_channel, it's identical to stem_width
                 feature_channels=(128, 256, 512), # original
                 stride_per_layer=(2, 2, 2), # original
                 use_SAConv=False, #ccy
                 extra_attention=None, #ccy
                 use_csp_bottleneck=False, #ccy
                 ):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else stem_width
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        # ccy params
        self.used_for_yolo = used_for_yolo
        self.in_channel = in_channel
        ##self.stem_channel = stem_width
        self.feature_channels = feature_channels
        self.use_SAConv = use_SAConv
        self.extra_attention = extra_attention
        self.use_csp = use_csp_bottleneck

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            raise NotImplementedError("Rectified convolution not implemented")
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        elif use_SAConv:
            conv_layer = SAConv3d
        else:
            conv_layer = nn.Conv3d
        self.conv_layer = conv_layer
        self.use_SAConv = use_SAConv
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(in_channel, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            # 7*7*7 conv no need SAConv3d
            self.conv1 = nn.Conv3d(in_channel, stem_width, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        if self.use_csp:
            stem_width = int(stem_width/2)
            feature_channels = [int(i/2) for i in feature_channels]
            #self.inplanes = int(self.inplanes/2) # don't do this if stem_layer is shared

        self.layer1 = self._make_layer(block, stem_width, layers[0], norm_layer=norm_layer, is_first=False, extra_attention=self.extra_attention)
        self.layer2 = self._make_layer(block, feature_channels[0], layers[1], stride=stride_per_layer[0], norm_layer=norm_layer, extra_attention=self.extra_attention)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, feature_channels[1], layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, feature_channels[2], layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, feature_channels[1], layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, feature_channels[2], layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, feature_channels[1], layers[2], stride=stride_per_layer[1],
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, feature_channels[2], layers[3], stride=stride_per_layer[2],
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        
        if not self.used_for_yolo:
            self.avgpool = GlobalAvgPool3d()
            self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
            self.fc = nn.Linear(feature_channels[2] * block.expansion, num_classes)
        
        if self.use_csp:
            ...
            #stem_width *= 2
            #feature_channels = [i*2 for i in feature_channels]
            # *2 for split recover (no need, processed in csp)
            # another *2 on layer1 inplanes for deep_stem
            double = True # whether to make c2=2*c2 per layer
            if deep_stem:
                stem_out = stem_width*2
            self.layer1 = BottleneckCSPWrapper(stem_out*2, stem_width, self.layer1, expansion=block.expansion, stride=1, isfirst=True, double=double)
            self.layer2 = BottleneckCSPWrapper(stem_width, feature_channels[0], self.layer2, expansion=block.expansion, stride=stride_per_layer[0], double=double)
            self.layer3 = BottleneckCSPWrapper(feature_channels[0], feature_channels[1], self.layer3, expansion=block.expansion, stride=stride_per_layer[1], double=double)
            self.layer4 = BottleneckCSPWrapper(feature_channels[1], feature_channels[2], self.layer4, expansion=block.expansion, stride=stride_per_layer[2], double=double)

        self._initialize_weights()
        #for name, m in self.named_modules():
        #    if isinstance(m, nn.Conv3d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()
        print("channels_per_layer", [stem_width]+list(feature_channels))
        print("self.inplanes", self.inplanes)
        
    def _initialize_weights(self):
        print("**" * 10, "Initing ResNeSt weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))

            if isinstance(m, nn.Conv3d) or isinstance(m, self.conv_layer):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))   

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True, extra_attention=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool3d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool3d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv3d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv3d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,
                                use_SAConv=self.use_SAConv,
                                extra_attention=extra_attention,
                                ))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,
                                use_SAConv=self.use_SAConv,
                                extra_attention=extra_attention,
                                ))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma,
                                use_SAConv=self.use_SAConv,
                                extra_attention=extra_attention,
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print("After maxpool:", x.shape)

  
        x4 = self.layer1(x)
        #print("layer1", x.shape)
        x3 = self.layer2(x4)
        #print("layer2", x3.shape)
        x2 = self.layer3(x3)
        #print("layer3", x2.shape)
        x1 = self.layer4(x2)
        #print("layer4", x1.shape)
 

        if self.used_for_yolo:
            return [x4, x3, x2, x1]
        else:
            x = self.avgpool(x1)
            #x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            if self.drop:
                x = self.drop(x)
            x = self.fc(x)
            return x


class Conv(nn.Conv3d):
    def __init__(self, *args, norm_layer=nn.BatchNorm3d, act=nn.ReLU, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.norm = norm_layer(args[1]) if type(norm_layer)!=type(None) else None
        self.act = act()
    def forward(self, x):
        x = super().forward(x)
        x = self.act(self.norm(x))
        return x


class BottleneckCSPWrapper(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks (Their Proposed)
    def __init__(self, inplane, plane, layer, expansion, stride=1, isfirst=False, double=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPWrapper, self).__init__()
        self.m = layer # in: inplane, out: plane*expansion
        c_in = inplane
        c1_out = inplane
        c_ = plane * expansion  # bottleneck out_channels = plane(feature channels) * expansion
        c_out = plane * expansion

        #if not isfirst and double:
        #    c1 *= expansion
        #    c2 *= 2
        if (not isfirst):
            c_in *= expansion
            c1_out *= expansion
            if double:
                c_in *= 2
                c_out *= 2
        elif double:
            c_out *= 2
            

        self.c_in = c_in # bottleneck in_channel
        self.c1_out = c1_out
        self.c_ = c_
        self.c_out = c_out

        self.c_layer = plane * expansion # bottleneck out_channel
        self.cv1 = Conv(c_in, c1_out, 1, 1 )  # (transition 1) (base_layer)
        self.cv2 = nn.Conv3d(c_in, c_, 1, stride, bias=False)
        self.cv3 = nn.Conv3d(c_, c_, 1, 1, bias=False) # (transition 2)
        self.cv4 = Conv(2 * c_, c_out, 1, 1) # (transition 3)
        self.bn = nn.BatchNorm3d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1)
        

    def forward(self, x):
        #print("Entering CSP with c_in={}, c1_out={}, c_={}, c_out={}, self.m.inplanes={}".format(self.c_in, self.c1_out, self.c_, self.c_out, self.m[0].inplanes))
        #print("csp inp.shape", x.shape, "[CSP config: {}->{}]".format(self.c_in, self.c_layer))
        tmp = self.cv1(x)
        #print("csp conv1", tmp.shape)
        tmp = self.m(tmp)
        #print("csp m",tmp.shape)
        y1 = self.cv3(tmp)
        #print("csp conv3",tmp.shape)
        #y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        out = self.act(self.bn(torch.cat((y1, y2), dim=1)))
        out = self.cv4(out)
        #print("csp out", out.shape)
        return out