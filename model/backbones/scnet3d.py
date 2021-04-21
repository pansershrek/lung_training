##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Jiang-Jiang Liu
## Email: j04.liu@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""3D SCNet variants"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

try:
    from .splat import SplAtConv3d
except (ImportError, ModuleNotFoundError):
    from splat import SplAtConv3d

import sys
sys.path.append("D:/CH/LungDetection/training")
import config.yolov4_config as cfg
from model.layers.attention_layers import SEModule, SEModule_Conv, CBAM

__all__ = ['SCNet', 'scnet50', 'scnet101', 'scnet50_v1d', 'scnet101_v1d']

model_urls = {
    'scnet50': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50-dc6a7e87.pth',
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
    'scnet101': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet101-44c5b751.pth',
    # 'scnet101_v1d': coming soon...
}

class SCConv3d(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv3d, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool3d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck3d(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 2 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None,
                 dropblock_prob=0.0, last_gamma=False,
                 use_splat=False,
                 use_SAConv=False, #ccy: Not Implemented
                 extra_attention=None,
                 ):
        super(SCBottleneck3d, self).__init__()
        assert dropblock_prob==0.
        assert not use_SAConv
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        # extra attention (3d)
        self.attention = extra_attention
        out_channels = planes * self.expansion
        if self.attention == 'SEnet':self.attention_module = SEModule(out_channels, dims=3, reduction=16) # lower reduction to avoid OOM during testing
        elif self.attention == 'SEnetConv':self.attention_module = SEModule_Conv(out_channels, dims=3, reduction=16)
        elif self.attention == 'CBAM':self.attention_module = CBAM(out_channels, dims=3)
        else: self.attention = None

        if self.avd:
            self.avd_layer = nn.AvgPool3d(3, stride, padding=1)
            stride = 1

        if use_splat:
            radix=2
            cardinality=1
            self.k1 = SplAtConv3d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=2, rectify=False,
                rectify_avg=False,
                norm_layer=norm_layer,
                dropblock_prob=0.0,
                use_SAConv=False) # ccy: True has super low cpm
        else:
            self.k1 = nn.Sequential(
                        nn.Conv3d(
                            group_width, group_width, kernel_size=3, stride=stride,
                            padding=dilation, dilation=dilation,
                            groups=cardinality, bias=False),
                        norm_layer(group_width),
                        )

        self.scconv = SCConv3d(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv3d(
            group_width * 2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        #print("Bottleneck input:", x.shape)
        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        #print("out_a:", out_a.shape)
        #print("out_b:", out_b.shape)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.attention is not None:
            out = self.attention_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #print("Bottleneck before residual:", out.shape)
        #print("Residual shape:", residual.shape)

        out += residual
        out = self.relu(out)

        #print("Bottleneck output:", out.shape)

        return out

class SCNet(nn.Module):
    """ SCNet Variants Definations
    Parameters
    ----------
    block : Block
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block.
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained SCNet yielding a stride-8 model.
    deep_stem : bool, default False
        Replace 7x7 conv in input stem with 3 3x3 conv.
    avg_down : bool, default False
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck.
    norm_layer : object
        Normalization layer used (default: :class:`torch.nn.BatchNorm3d`).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    def __init__(self, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm3d,

                 used_for_yolo=True, # if True, not using adaptivepool/fc and return feature maps of 3 different scales
                 in_channel=3, # original
                 ## no need stem_channel, it's identical to stem_width
                 use_splat=False,
                 feature_channels=(128, 256, 512), # original
                 stride_per_layer=(2, 2, 2), # original
                 use_SAConv=False, #ccy: Not Implemented
                 extra_attention=None, #ccy
                 ):
        #raise NotImplementedError("Please don't call SCNet, but call ResNeSt instead")
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd

        #ccy
        self.in_channel = in_channel
        assert not use_SAConv
        self.used_for_yolo = used_for_yolo
        self.use_splat = use_splat
        self.extra_attention = extra_attention

        super(SCNet, self).__init__()
        conv_layer = nn.Conv3d
        self.conv_layer = conv_layer
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(self.in_channel, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(self.in_channel, stem_width, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, stem_width, layers[0], norm_layer=norm_layer, is_first=False, extra_attention=self.extra_attention)
        self.layer2 = self._make_layer(block, feature_channels[0], layers[1], stride=stride_per_layer[0], norm_layer=norm_layer, extra_attention=self.extra_attention)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, feature_channels[1], layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, feature_channels[2], layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation==2:
            self.layer3 = self._make_layer(block, feature_channels[1], layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, feature_channels[2], layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, feature_channels[1], layers[2], stride=stride_per_layer[1],
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, feature_channels[2], layers[3], stride=stride_per_layer[2],
                                           norm_layer=norm_layer)
        
        if not self.used_for_yolo:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            #self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        

        self._initialize_weights()
    
    def _initialize_weights(self):
        print("**" * 10, "Initing SCResNeSt weights", "**" * 10)

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
                    is_first=True, extra_attention=None):
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
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=1, is_first=is_first, 
                                norm_layer=norm_layer,
                                use_splat=self.use_splat,
                                extra_attention=self.extra_attention))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=2, is_first=is_first, 
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation, 
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print("After maxpool:", x.shape)

        x = self.layer1(x)
        #print("layer1", x.shape)
        x3 = self.layer2(x)
        #print("layer2", x3.shape)
        x2 = self.layer3(x3)
        #print("layer3", x2.shape)
        x1 = self.layer4(x2)
        #print("layer4", x1.shape)
        
        if self.used_for_yolo:
            return [x3, x2, x1]
        else:
            x = self.avgpool(x1)
            #x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            if self.drop:
                x = self.drop(x)
            x = self.fc(x)
            return x




def scnet50(pretrained=False, **kwargs):
    """Constructs a SCNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3],
                deep_stem=False, stem_width=32, avg_down=False,
                avd=False, **kwargs)
    if pretrained:
        raise NotImplementedError()
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50']))
    return model

def scnet50_v1d(pretrained=False, **kwargs):
    """Constructs a SCNet-50_v1d model described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.
    `ResNeSt: Split-Attention Networks <https://arxiv.org/pdf/2004.08955.pdf>`_.

    Compared with default SCNet(SCNetv1b), SCNetv1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 3x3 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, **kwargs)
    if pretrained:
        raise NotImplementedError()
        model.load_state_dict(model_zoo.load_url(model_urls['scnet50_v1d']))
    return model

def scnet101(pretrained=False, **kwargs):
    """Constructs a SCNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck3d, [3, 4, 23, 3],
                deep_stem=False, stem_width=64, avg_down=False,
                avd=False, **kwargs)
    if pretrained:
        raise NotImplementedError()
        model.load_state_dict(model_zoo.load_url(model_urls['scnet101']))
    return model

def scnet101_v1d(pretrained=False, **kwargs):
    """Constructs a SCNet-101_v1d model described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.
    `ResNeSt: Split-Attention Networks <https://arxiv.org/pdf/2004.08955.pdf>`_.

    Compared with default SCNet(SCNetv1b), SCNetv1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 3x3 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SCNet(SCBottleneck, [3, 4, 23, 3],
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, **kwargs)
    if pretrained:
        raise NotImplementedError()
        model.load_state_dict(model_zoo.load_url(model_urls['scnet101_v1d']))
    return model


def _Build_SCResNeSt3D(in_channel, weight_path=None, resume=False, used_for_yolo=True, bottleneck_expansion=4, debug=False, debug_device="cuda"):
    #WIP: make model! (need multiistaged feature maps for yolo 3 different scale!!)
    # check Darknet53 for those shape beforehand!
    # ccy: p.s. groups = cardinality 
    # ccy: stem_width = stem_channel (deep_stem should be True)
    # bottleneck_width has no significant use here; just keep it 64

    bottleneck_expansion = cfg.MODEL["RESNEST_EXPANSION"]
    feature_channels = cfg.MODEL["RESNEST_FEATURE_CHANNELS"] # original: (128, 256, 512)
    stem_width = cfg.MODEL["RESNEST_STEM_WIDTH"] # orginal: 32
    blocks_per_stage = cfg.MODEL["RESNEST_BLOCKS_PER_STAGE"] # Resnet50 original: (3, 4, 6, 3)
    stride_per_layer = cfg.MODEL["RESNEST_STRIDE_PER_LAYER"] # original: (2,2,2)
    use_splat = True
    use_SAConv = False
    extra_attention = cfg.MODEL["RESNEST_EXTRA_ATTENTION"]

    if debug:
        bottleneck_expansion = 2 #cfg.MODEL["RESNEST_EXPANSION"]
        feature_channels = (24, 64, 128)#cfg.MODEL["RESNEST_FEATURE_CHANNELS"] # original: (128, 256, 512)
        stem_width = 16 #cfg.MODEL["RESNEST_STEM_WIDTH"] # orginal: 32
        blocks_per_stage = (2,3,3,3) #cfg.MODEL["RESNEST_BLOCKS_PER_STAGE"] # Resnet50 original: (3, 4, 6, 3)
        stride_per_layer = (1, 2, 2)
        use_splat = True
        use_SAConv = False
        extra_attention = "SEnetConv"

    Bottleneck = SCBottleneck3d
    if bottleneck_expansion == 4:  # expansion == 4 (original), it has 78W params, otherwise can try other expansion
        resnet_feature_channels = [i//Bottleneck.expansion for i in feature_channels] 
    elif bottleneck_expansion == 2: # expansion ==2, 250M params
        Bottleneck.expansion = 2
        resnet_feature_channels = [i//Bottleneck.expansion for i in feature_channels] 
    elif bottleneck_expansion == 1: # expansion == 1, this has about 900M params (bad) (smaller expansion, higher # of params)
        Bottleneck.expansion = 1
        resnet_feature_channels = feature_channels
    else:
        raise TypeError("Invalid bottleneck_expansion: {}".format(bottleneck_expansion))
    
    model = SCNet(Bottleneck, blocks_per_stage,
                   groups=1, bottleneck_width=64, #radix=2 by default
                   deep_stem=True, avg_down=True,
                   avd=True,  # for simplicity, don't define avd_first here
                   used_for_yolo=used_for_yolo,
                   in_channel=in_channel, # == 1
                   stem_width=stem_width, # original: 32
                   feature_channels=resnet_feature_channels, # original: (128,256,512)
                   stride_per_layer=stride_per_layer, # original: (2,2,2)
                   use_splat=use_splat,
                   use_SAConv=use_SAConv,
                   extra_attention=extra_attention,
                   )

    if debug: # debug
        import time
        from torchsummary import summary
        device = debug_device
        model =  model.to(device)
        model.eval()

        #t_shape = [[1, 128,128,128]] # not including batch_size
        #summary(model, t_shape, device=device)
        #raise EOFError("End of file")
        t = torch.randn((1,1,256,512,512), device=device)
        #t = torch.randn((1,1,32,128,128), device=device)
        out = model(t)
        print("Output:", *[f.shape for f in out])
        #print("Feature channels:", model.feature_channels[-3:])
        print("Feature channels:", feature_channels)
        time.sleep(5)
    return model, list(feature_channels[-3:])

if __name__ == "__main__":
    #images = torch.rand(1, 3, 224, 224).cuda(0)
    #model = scnet101(pretrained=True)
    #model = model.cuda(0)
    #print(model(images).size())
    from memory_limiting import main as memory_limiting
    device = "cuda"
    if device == "cpu": # NEVER REMOVE THIS LINE, OR THE PC MAY STUCK
        memory_limiting(15*1000)  # 15*1000 means 15GB
    _Build_SCResNeSt3D(1, debug=True, debug_device=device)
    
