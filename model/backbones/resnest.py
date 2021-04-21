##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models (3d)
modified from: https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnest.py
"""

import torch
try:
    from .resnet import ResNet, Bottleneck
    import config.yolov4_config as cfg
except (ImportError, ModuleNotFoundError):
    from resnet import ResNet, Bottleneck
    import sys
    sys.path.append("D:/CH/LungDetection/training")
    import config.yolov4_config as cfg

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,  # ccy: p.s. groups = cardinality 
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model


def _BuildResNeSt3D(in_channel, weight_path=None, resume=False, used_for_yolo=True, bottleneck_expansion=4, debug=False, debug_device="cuda"):
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
    use_SAConv = cfg.MODEL["USE_SACONV"]
    extra_attention = cfg.MODEL["RESNEST_EXTRA_ATTENTION"]

    if debug:
        bottleneck_expansion = 2 #cfg.MODEL["RESNEST_EXPANSION"]
        feature_channels = (24, 64, 128)#cfg.MODEL["RESNEST_FEATURE_CHANNELS"] # original: (128, 256, 512)
        stem_width = 16 #cfg.MODEL["RESNEST_STEM_WIDTH"] # orginal: 32
        blocks_per_stage = (2,3,3,3) #cfg.MODEL["RESNEST_BLOCKS_PER_STAGE"] # Resnet50 original: (3, 4, 6, 3)
        stride_per_layer = (1, 2, 2)
        use_SAConv = True
        extra_attention = None

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
    
    model = ResNet(Bottleneck, blocks_per_stage,
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=False, 
                   used_for_yolo=used_for_yolo,
                   in_channel=in_channel, # == 1
                   stem_width=stem_width, # original: 32
                   feature_channels=resnet_feature_channels, # original: (128,256,512)
                   stride_per_layer=stride_per_layer, # original: (2,2,2)
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
    from memory_limiting import main as memory_limiting
    device = "cuda"
    if device == "cpu": # NEVER REMOVE THIS LINE, OR THE PC MAY STUCK
        memory_limiting(15*1000)  # 15*1000 means 15GB
    _BuildResNeSt3D(1, debug=True, debug_device=device)

    