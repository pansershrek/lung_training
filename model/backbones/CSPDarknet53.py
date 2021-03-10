import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

if __name__ == "__main__":
    import sys
    sys.path.append("D:/CH/LungDetection/training")

from model.layers.attention_layers import SEModule, SEModule_Conv, CBAM
import config.yolov4_config as cfg

if 0:
    #non cuda supported
    class Mish(nn.Module):
        def __init__(self):
            super(Mish, self).__init__()

        def forward(self, x):
            return x * torch.tanh(F.softplus(x))
#from https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/8f006d351bf1ac888239cfeaf6fcd4a31eb866ca
#from mish_cuda import MishCuda as Mish

norm_name = {"bn": nn.BatchNorm2d, "bn3d": nn.BatchNorm3d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    'linear': nn.Identity(),
    "mish": None #Mish()
    }

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, norm='bn', activate='mish', dims=2):
        super(Convolutional, self).__init__()
        # ccy: the "norm" and "activate" argument seems useless here (always used BN3D + ReLU)

        self.norm = norm
        self.activate = activate
        if dims==3:
            self.__conv = nn.Conv3d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size//2, bias=not norm)
        else:
            self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size//2, bias=not norm)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                if dims==3:
                    self.__norm = norm_name["bn3d"](num_features=filters_out)
                else:
                    self.__norm = norm_name[norm](num_features=filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "mish":
                self.__activate = activate_name['relu'](inplace=False)
                #self.__activate = activate_name[activate]

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, residual_activation='linear', dims=2):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.activation = activate_name[residual_activation]
        self.attention = cfg.ATTENTION["TYPE"]

        #if self.attention == "SplAt":
        #    use_splat = True
        #    self.attention = None

        if self.attention == 'SEnet':self.attention_module = SEModule(out_channels, dims=dims)
        elif self.attention == 'SEnetConv':self.attention_module = SEModule_Conv(out_channels, dims=dims)
        elif self.attention == 'CBAM':self.attention_module = CBAM(out_channels, dims=dims)
        else: self.attention = None

 

        self.block = nn.Sequential(
            Convolutional(in_channels, hidden_channels, 1, dims=dims),
            Convolutional(hidden_channels, out_channels, 3, dims=dims)
        )

        
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.attention is not None:
            out = self.attention_module(out)
        out += residual
        return out

class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels, dims=2, downsample_stride=2):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = Convolutional(in_channels, out_channels, 3, stride=downsample_stride, dims=dims)

        self.split_conv0 = Convolutional(out_channels, out_channels, 1, dims=dims)
        self.split_conv1 = Convolutional(out_channels, out_channels, 1, dims=dims)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels, dims=dims),
            Convolutional(out_channels, out_channels, 1, dims=dims)
        )

        self.concat_conv = Convolutional(out_channels*2, out_channels, 1, dims=dims)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, dims=2, downsample_stride=2):
        super(CSPStage, self).__init__()

        self.downsample_conv = Convolutional(in_channels, out_channels, 3, stride=downsample_stride, dims=dims)

        self.split_conv0 = Convolutional(out_channels, out_channels//2, 1, dims=dims)
        self.split_conv1 = Convolutional(out_channels, out_channels//2, 1, dims=dims)

        self.blocks_conv = nn.Sequential(
            *[CSPBlock(out_channels//2, out_channels//2, dims=dims) for _ in range(num_blocks)],
            Convolutional(out_channels//2, out_channels//2, 1, dims=dims)
        )

        self.concat_conv = Convolutional(out_channels, out_channels, 1, dims=dims)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPDarknet53(nn.Module):
    def __init__(self, in_channel, stem_channels=None, feature_channels=None, num_features=3,weight_path=None, resume=False, dims=2):
        super(CSPDarknet53, self).__init__()

        #BS1
        #stem_channels = 4
        #channel_factor = 1/16 * 5

        #for 1080 96 #for 96
        #1080Ti
        #stem_channels = 4
        #channel_factor = 1/16 * 8

        #for 1080 640 #for 640
        #1080Ti
        #stem_channels = 2
        #channel_factor = 1/16 * 2 #like shit

        #for 3090 640 #for 96
        #BS2
        #stem_channels = 4
        #channel_factor = 1/16 * 4

        #today and in those best day

        #4_4_64
        #stem_channels = 4
        #channel_factor = 1/16 * 1

        if (stem_channels == None):
            stem_channels = cfg.MODEL["CSPDARKNET53_STEM_CHANNELS"]

            
        if (feature_channels == None):
            feature_channels=cfg.MODEL["CSPDARKNET53_FEATURE_CHANNELS"]
            channel_factor = 1.0
           
        #blocks_per_stage = [2,8,8,4] #original
        blocks_per_stage = cfg.MODEL["CSPDARKNET53_BLOCKS_PER_STAGE"]

        #4_8_128
        #stem_channels = 4
        #channel_factor = 1/16 * 2

        #4_16_256
        #stem_channels = 4
        #channel_factor = 1/16 * 4



        if (1): # ccy; it runs but uses so much GRAM (edit: use smaller channel per stage can solve the problem)
            feature_channels = [int(_ * (channel_factor)) for _ in feature_channels]
            self.stem_conv = Convolutional(in_channel, stem_channels, 3, dims=dims)
            self.stages = nn.ModuleList([
            CSPFirstStage(stem_channels, feature_channels[0], dims=dims, downsample_stride=2),
            CSPStage(feature_channels[0], feature_channels[1], blocks_per_stage[0], dims=dims, downsample_stride=2),
            CSPStage(feature_channels[1], feature_channels[2], blocks_per_stage[1], dims=dims, downsample_stride=2),
            CSPStage(feature_channels[2], feature_channels[3], blocks_per_stage[2], dims=dims, downsample_stride=2),
            ])
        else:
            feature_channels = [int(_ * (channel_factor)) for _ in feature_channels]
            self.stem_conv = Convolutional(in_channel, stem_channels, 3, dims=dims)
            #self.stem_conv = Convolutional(in_channel, stem_channels, kernel_size=7, stride=2, dims=dims)
            self.stages = nn.ModuleList([
                CSPFirstStage(stem_channels, feature_channels[0], dims=dims),
                CSPStage(feature_channels[0], feature_channels[1], 2, dims=dims),
                CSPStage(feature_channels[1], feature_channels[2], 8, dims=dims),
                CSPStage(feature_channels[2], feature_channels[3], 8, dims=dims), 
                CSPStage(feature_channels[3], feature_channels[4], 4, dims=dims)  
            ])
        
        assert len(feature_channels) == len(self.stages)
        assert len(blocks_per_stage) == len(self.stages) - 1 # first_stage not changable

        self.feature_channels = feature_channels
        self.num_features = num_features

        if weight_path and not resume and not dims==3:
            self.load_CSPdarknet_weights(weight_path, dims=dims)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        #print("In CSP backbone")
        for i,stage in enumerate(self.stages):
            x_ori_shape = x.shape
            x = stage(x)
            #print("stage {}: {} -> {}".format(i, x_ori_shape, x.shape))
            features.append(x)

        return features[-self.num_features:]

    def _initialize_weights(self):
        print("**" * 10, "Initing CSPDarknet53 weights", "**" * 10)

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

            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


    def load_CSPdarknet_weights(self, weight_file, cutoff=52, dims=2):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        assert dims==2, 'load_CSPdarknet_weights with dims==3 not implemented'
        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                # if count == cutoff:
                #     break
                # count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))


def _BuildCSPDarknet53(in_channel, weight_path, resume, dims=2):
    model = CSPDarknet53(in_channel, weight_path=weight_path, resume=resume, dims=dims)
    out = model, model.feature_channels[-3:]
    #print("At _BuildCSPDarknet53:", out[1])
    #raise TypeError
    return out

if __name__ == '__main__':
    def verbose_forward(model, x):
        x = model.stem_conv(x)

        features = []
        print("In CSP backbone")
        for i,stage in enumerate(model.stages):
            x_ori_shape = x.shape
            x = stage(x)
            print("stage {}: {} -> {}".format(i, x_ori_shape, x.shape))
            features.append(x)
        return features[-model.num_features:]


    from torchsummary import summary

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    backbone, num_features = _BuildCSPDarknet53(1, None, False, dims=3)
    x = torch.randn((1, 1, 128, 128, 128), device=device)
    model = backbone.to(device)
    print("num features:", num_features)
    if (1): # torchsummary
        summary(backbone, (1,128,128,128), device=device)
 
        # verbose forward
        verbose_forward(backbone, x)
        # forward 
        ys = model(x)
        print("output:", *[y.shape for y in ys])

