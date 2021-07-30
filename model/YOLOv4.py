import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

try:
    import config.yolov4_config as cfg
    from .backbones.CSPDarknet53 import _BuildCSPDarknet53
    from .backbones.mobilenetv2 import _BuildMobilenetV2
    from .backbones.mobilenetv3 import _BuildMobilenetV3
    from .backbones.resnest import _BuildResNeSt3D
    from .backbones.scnet3d import _Build_SCResNeSt3D
except:
    sys.path.append("D:/CH/LungDetection/training")
    import config.yolov4_config as cfg
    from model.backbones.CSPDarknet53 import _BuildCSPDarknet53
    from model.backbones.mobilenetv2 import _BuildMobilenetV2
    from model.backbones.mobilenetv3 import _BuildMobilenetV3
    from model.backbones.resnest import _BuildResNeSt3D
    from model.backbones.scnet3d import _Build_SCResNeSt3D


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dims=2):
        super(Conv, self).__init__()

        if (0): #ccy: avoid error
            padding = 0
        else:
            padding = kernel_size//2

        if dims==3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        try:
            return self.conv(x)
        except: # avoid BN
            x = self.conv[0](x)
            x = self.conv[-1](x)
            return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13], dims=2):
        super(SpatialPyramidPooling, self).__init__()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1]//2, 1, dims=dims),
            Conv(feature_channels[-1]//2, feature_channels[-1], 3, dims=dims),
            Conv(feature_channels[-1], feature_channels[-1]//2, 1, dims=dims),
        )
        if dims==3:
            self.maxpools = nn.ModuleList([nn.MaxPool3d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        else:
            self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x]+features, dim=1)

        return features

    def __initialize_weights(self):
        print("**" * 10, "Initing head_conv weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, dims=2):
        super(Upsample, self).__init__()
        self.conv1x1 = nn.Sequential(
            Conv(in_channels, out_channels, 1, dims=dims),
            #nn.Upsample(scale_factor=scale)
        )

    def forward(self, x, odds=(0,0,0)): # ccy: odds is to handle odd number shape
        x = self.conv1x1(x)
        _, _, Z, Y, X = x.shape
        Z, Y, X = Z*2-odds[0], Y*2-odds[1], X*2-odds[2]
        x = torch.nn.functional.interpolate(x, size=(Z,Y,X))
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, dims=2):
        super(Downsample, self).__init__()
        self.downsample = Conv(in_channels, out_channels, 3, 2, dims=dims)

    def forward(self, x):
        return self.downsample(x)

class PANet(nn.Module):
    def __init__(self, feature_channels, dims=2):
        super(PANet, self).__init__()

        self.feature_transform3 = Conv(feature_channels[0], feature_channels[0]//2, 1, dims=dims)
        self.feature_transform4 = Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims)

        self.resample5_4 = Upsample(feature_channels[2]//2, feature_channels[1]//2, dims=dims)
        self.resample4_3 = Upsample(feature_channels[1]//2, feature_channels[0]//2, dims=dims)
        self.resample3_4 = Downsample(feature_channels[0]//2, feature_channels[1]//2, dims=dims)
        self.resample4_5 = Downsample(feature_channels[1]//2, feature_channels[2]//2, dims=dims)

        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2]*2, feature_channels[2]//2, 1, dims=dims),
            Conv(feature_channels[2]//2, feature_channels[2], 3, dims=dims),
            Conv(feature_channels[2], feature_channels[2]//2, 1, dims=dims)
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims),
            Conv(feature_channels[1]//2, feature_channels[1], 3, dims=dims),
            Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims),
            Conv(feature_channels[1]//2, feature_channels[1], 3, dims=dims),
            Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0]//2, 1, dims=dims),
            Conv(feature_channels[0]//2, feature_channels[0], 3, dims=dims),
            Conv(feature_channels[0], feature_channels[0]//2, 1, dims=dims),
            Conv(feature_channels[0]//2, feature_channels[0], 3, dims=dims),
            Conv(feature_channels[0], feature_channels[0]//2, 1, dims=dims),
        )

        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims),
            Conv(feature_channels[1]//2, feature_channels[1], 3, dims=dims),
            Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims),
            Conv(feature_channels[1]//2, feature_channels[1], 3, dims=dims),
            Conv(feature_channels[1], feature_channels[1]//2, 1, dims=dims),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2]//2, 1, dims=dims),
            Conv(feature_channels[2]//2, feature_channels[2], 3, dims=dims),
            Conv(feature_channels[2], feature_channels[2]//2, 1, dims=dims),
            Conv(feature_channels[2]//2, feature_channels[2], 3, dims=dims),
            Conv(feature_channels[2], feature_channels[2]//2, 1, dims=dims)
        )
        self.__initialize_weights()

    def forward(self, features):
        odds = [tuple(np.array(f.shape[-3:])%2) for f in features]

        features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]
        downstream_feature5 = self.downstream_conv5(features[2])
        #downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5, odds[1])], dim=1))
        #downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4, odds[0])], dim=1))
        downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5)], dim=1))
        downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4)], dim=1))

        upstream_feature4 = self.upstream_conv4(torch.cat([self.resample3_4(downstream_feature3), downstream_feature4], dim=1))
        upstream_feature5 = self.upstream_conv5(torch.cat([self.resample4_5(upstream_feature4), downstream_feature5], dim=1))

        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def __initialize_weights(self):
        print("**" * 10, "Initing PANet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))

class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels, dims=2):
        super(PredictNet, self).__init__()
        nn_conv = nn.Conv3d if dims==3 else nn.Conv2d
        
        self.predict_conv = nn.ModuleList([
            nn.Sequential(
                Conv(feature_channels[i]//2, feature_channels[i], 3, dims=dims),
                nn_conv(feature_channels[i], target_channels, 1)
            ) for i in range(len(feature_channels))
        ])

        self.__initialize_weights()

    def forward(self, features):
        predicts = [predict_conv(feature) for predict_conv, feature in zip(self.predict_conv, features)]

        return predicts

    def __initialize_weights(self):
        print("**" * 10, "Initing PredictNet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))

class MixLayer0Net(nn.Module):
    def __init__(self, feature_channels, layer0_nc, dims=2, mode=None):
        super(MixLayer0Net, self).__init__()
        nn_conv = nn.Conv3d if dims==3 else nn.Conv2d

        self.mode = cfg.MODEL["MIXLAYER0NET_MODE"] if type(mode)==type(None) else mode

        if self.mode=="concat": # executed on concatted features
            self.mix_conv = nn.ModuleList([
                nn.Sequential(
                    Conv(feature_channels[i]//2+layer0_nc, feature_channels[i], 3, dims=dims),
                    nn_conv(feature_channels[i], feature_channels[i]//2, 1)
                ) for i in range(len(feature_channels))
            ])
            self.forward_func = self.forward_concat
        elif self.mode=="attention1": # executed on layer0 feature
            self.mix_conv = nn.ModuleList([
                nn.Sequential(
                    Conv(layer0_nc, feature_channels[i], 3, dims=dims),
                    nn_conv(feature_channels[i], feature_channels[i]//2, 1)
                ) for i in range(len(feature_channels))
            ])
            self.gap = nn.AdaptiveAvgPool3d(1)
            self.sigmoid = nn.Sigmoid()
            self.forward_func = self.forward_attention1
        elif self.mode=="attention2": # executed on layer0 feature (SE alike, use 1x1 conv to mimic linear)
            self.mix_conv = nn.ModuleList([
                nn.Sequential(
                    Conv(layer0_nc, feature_channels[i], 1, dims=dims),
                    nn_conv(feature_channels[i], feature_channels[i]//2, 1)
                ) for i in range(len(feature_channels))
            ])
            self.gap = nn.AdaptiveAvgPool3d(1)
            self.sigmoid = nn.Sigmoid()
            self.forward_func = self.forward_attention2
        elif self.mode=="concat2":
            self.channel_conv = nn.ModuleList([
                    nn_conv(layer0_nc, feature_channels[i]//2, 1) for i in range(len(feature_channels))
            ])
            self.forward_func = self.forward_concat2
        else:
            raise TypeError(f"Invalid mix mode in MixLayer0Net, unknown mode: '{self.mode}'")

        self.__initialize_weights()
    
    def forward_concat(self, features, layer0_feature):
        mixed_total = []
        for mix_conv, feature in zip(self.mix_conv, features):
            feature_shape = feature.shape[-3:] # z,y,x part only, to know the scale
            l0 = nn.functional.adaptive_avg_pool3d(layer0_feature, feature_shape)
            catted = torch.cat([feature, l0], dim=1) # channel-wise
            mixed = mix_conv(catted)
            mixed_total.append(mixed)
        return mixed_total

    def forward_attention1(self, features, layer0_feature):
        mixed_total=[]
        layer0_features = [mix_conv(layer0_feature) for mix_conv in self.mix_conv]
        for l0, feature in zip(layer0_features, features):
            w = self.sigmoid(self.gap(l0))
            mixed = feature * w
            mixed_total.append(mixed)
        return mixed_total 

    def forward_attention2(self, features, layer0_feature):
        #print("Within MixLayer0 attn2")
        #print("layer0_feature", layer0_feature.shape)
        #print("features.shape", [x.shape for x in features])
        #1/0
        layer0_feature = self.gap(layer0_feature)
        l0_w = [self.sigmoid(mix_conv(layer0_feature)) for mix_conv in self.mix_conv]
        mixed_total=[feature*w for feature, w in zip(features, l0_w)]
        return mixed_total 

    def forward_concat2(self, features, layer0_feature):
        mixed_total = []
        for channel_conv, feature in zip(self.channel_conv, features):
            feature_shape = feature.shape[-3:] # z,y,x part only, to know the scale
            l0 = nn.functional.adaptive_max_pool3d(layer0_feature, feature_shape) # 1. downsample
            l0 = channel_conv(l0) # 2. make channel number == that after PANet == feature_channel//2
            catted = torch.cat([feature, l0], dim=1) # 3. concat and that's all; out_channel*=2
            mixed_total.append(catted)
        return mixed_total 


    def forward(self, features, layer0_feature):
        return self.forward_func(features, layer0_feature)

    def __initialize_weights(self):
        print("**" * 10, "Initing MixLayer0Net weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))

class YOLOv4(nn.Module):
    def __init__(self, weight_path=None, out_channels=255, resume=False, dims=2, verbose=cfg.MODEL["VERBOSE_SHAPE"]):
        super(YOLOv4, self).__init__()

        self.use_layer0 = cfg.MODEL["YOLO_USE_LAYER0"]
        self.use_layer0_mode = cfg.MODEL["MIXLAYER0NET_MODE"]

        #self.use_layer0_mode = "concat2"
        

        a = cfg.MODEL_TYPE['TYPE']
        if cfg.MODEL_TYPE['TYPE'] == 'YOLOv4':
            if cfg.MODEL["BACKBONE"] == "CSPDarknet":
                # CSPDarknet53 backbone
                self.backbone, feature_channels = _BuildCSPDarknet53(in_channel=cfg.MODEL_INPUT_CHANNEL, weight_path=weight_path, resume=resume, dims=dims)
                layer0_channel_nC = 8 # resnest
            elif cfg.MODEL["BACKBONE"] == "ResNeSt":
                # ccy: the load weight feature had been handled in trainer.py, so you don't need to care about it here
                self.backbone, feature_channels = _BuildResNeSt3D(in_channel=cfg.MODEL_INPUT_CHANNEL, used_for_yolo=True, bottleneck_expansion=4) 
                layer0_channel_nC = 32 # resnest
            elif cfg.MODEL["BACKBONE"] == "SCResNeSt":
                self.backbone, feature_channels = _Build_SCResNeSt3D(in_channel=cfg.MODEL_INPUT_CHANNEL, used_for_yolo=True, bottleneck_expansion=4) 
            else:
                raise TypeError("Unknown model_type: '{}'".format(cfg.MODEL["BACKBONE"]))
        elif cfg.MODEL_TYPE["TYPE"] == 'Mobilenet-YOLOv4':
            # MobilenetV2 backbone
            self.backbone, feature_channels = _BuildMobilenetV2(in_channel=cfg.MODEL_INPUT_CHANNEL, weight_path=weight_path, resume=resume)
        elif cfg.MODEL_TYPE["TYPE"] == 'Mobilenetv3-YOLOv4':
            # MobilenetV2 backbone
            self.backbone, feature_channels = _BuildMobilenetV3(in_channel=cfg.MODEL_INPUT_CHANNEL, weight_path=weight_path, resume=resume)
        else:
            assert print('model type must be YOLOv4 or Mobilenet-YOLOv4')

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(feature_channels, dims=dims)

        # Path Aggregation Net
        self.panet = PANet(feature_channels, dims=dims)

        # Concat layer0 features before predictnet
        if self.use_layer0:
            self.mixlayer0net = MixLayer0Net(feature_channels, layer0_channel_nC, dims=dims, mode=self.use_layer0_mode)

        # predict
        if self.use_layer0 and self.use_layer0_mode in ["concat2"]:
            self.predict_net = PredictNet([c*2 for c in feature_channels], out_channels, dims=dims)
        else:
            self.predict_net = PredictNet(feature_channels, out_channels, dims=dims)
        
        self.verbose = verbose
        self.feature_channels_txt = str(feature_channels)

    def forward(self, x):
        verbose = self.verbose
        features = self.backbone(x)
        if verbose:
            print("After backbone:", end="")
            print(*[m.shape for m in features], sep="\n", end="\n"+"="*20+"\n")

        if len(features)!=3:
            assert (len(features)==4)
            layer0 = features[0]
            features = features[1:]

        features[-1] = self.spp(features[-1])
        if verbose:
            print("After SPP:", end=" ")
            print(*[m.shape for m in features], sep="\n", end="\n"+"="*20+"\n")
        features = self.panet(features)
        if verbose:
            print("After PAN:", end=" ")
            print(*[m.shape for m in features], sep="\n", end="\n"+"="*20+"\n")
        if self.use_layer0:
            features = self.mixlayer0net(features, layer0)
            if verbose:
                print("After MixLayer0Net:", end=" ")
                print(*[m.shape for m in features], sep="\n", end="\n"+"="*20+"\n")
        predicts = self.predict_net(features)
        if verbose:
            print("After predict_net:", end=" ")
            print(*[m.shape for m in predicts], sep="\n", end="\n"+"="*20+"\n")
        #raise EOFError
        return predicts


if __name__ == '__main__':
    #cuda = torch.cuda.is_available()
    #device = torch.device('cuda:{}'.format(0) if cuda else 'cpu')
    device = "cuda"
    #device = "cpu"
    model = YOLOv4(out_channels=27, dims=3, verbose=True).to(device)
    print("feature channels:", model.feature_channels_txt)

    x = torch.randn(2, 1, 128,128,128).to(device)
    torch.cuda.empty_cache()
    predicts = model(x)

