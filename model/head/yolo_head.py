try:
    from model.YOLOv4 import YOLOv4
    import sys
except:
    import sys
    sys.path.append(r"D:/CH/LungDetection/training")

import torch.nn as nn
import torch



class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride, dims=2):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride
        self.__dims=dims


    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        #print("p.shape:", p.shape)
        if self.__dims==3:
            p = p.view(bs, self.__nA, 7 + self.__nC, p.shape[-3], p.shape[-2], p.shape[-1]).permute(0, 3, 4, 5, 1, 2)
        else:
            p = p.view(bs, self.__nA, 5 + self.__nC, p.shape[-2], p.shape[-1]).permute(0, 3, 4, 1, 2)
        p_de = self.__decode(p.clone())
        return (p, p_de)


    def __decode(self, p):
        batch_size, output_size = p.shape[0], p.shape[1:-2]
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)
        if self.__dims==3:
            conv_raw_dxdy = p[:, :, :, :, :, 0:3]
            conv_raw_dwdh = p[:, :, :, :, :, 3:6]
            conv_raw_conf = p[:, :, :, :, :, 6:7]
            conv_raw_prob = p[:, :, :, :, :, 7:]

            y = torch.arange(0, output_size[1]).unsqueeze(1).repeat(1, output_size[2])
            x = torch.arange(0, output_size[2]).unsqueeze(0).repeat(output_size[1], 1)
            y = y.unsqueeze(0).repeat(output_size[0], 1, 1)
            x = x.unsqueeze(0).repeat(output_size[0], 1, 1)

            z = torch.arange(0, output_size[0]).unsqueeze(1).repeat(1, output_size[1]).unsqueeze(2).repeat(1, 1, output_size[2])

            grid_xy = torch.stack([z, y, x], dim=-1)
            grid_xy = grid_xy.unsqueeze(0).unsqueeze(4).repeat(batch_size, 1, 1, 1, 3, 1).float().to(device)
        else:
            conv_raw_dxdy = p[:, :, :, :, 0:2]
            conv_raw_dwdh = p[:, :, :, :, 2:4]
            conv_raw_conf = p[:, :, :, :, 4:5]
            conv_raw_prob = p[:, :, :, :, 5:]

            y = torch.arange(0, output_size[0]).unsqueeze(1).repeat(1, output_size[1])
            x = torch.arange(0, output_size[1]).unsqueeze(0).repeat(output_size[0], 1)
            grid_xy = torch.stack([x, y], dim=-1)
            grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        return pred_bbox.view(-1, p.size(-1)) if not self.training else pred_bbox


def _test_model():
    global YOLOv4, cfg
    from model.YOLOv4 import YOLOv4
    import config.yolov4_config as cfg
    from memory_limiting import main as memory_limiting
    anchors = torch.FloatTensor(cfg.MODEL["ANCHORS3D"])
    strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
    dims = 3
    nC = cfg.MODEL["ANCHORS_PER_SCLAE"] * (2 + 7)
    device="cuda"
    if device == "cpu":
        memory_limiting(15*1000)

    yolov4 = YOLOv4(weight_path=None, out_channels=nC, resume=False, dims=dims).to(device)
    head_s = Yolo_head(nC=2, anchors=anchors[0], stride=strides[0], dims=dims).to(device)
    head_m = Yolo_head(nC=2, anchors=anchors[1], stride=strides[1], dims=dims).to(device)
    head_l = Yolo_head(nC=2, anchors=anchors[2], stride=strides[2], dims=dims).to(device)

    yolov4.train()
    head_s.train()
    head_m.train()
    head_l.train()

    # try run
    x = torch.randn(4,1,128,128,128).to(device)
    x_s, x_m, x_l = yolov4(x)
    print("After backbone")
    #print(*[m[1].shape for m in [x_s, x_m, x_l]], sep="\n", end="\n"+"="*20+"\n")
    print(*[m.shape for m in [x_s, x_m, x_l]], sep="\n", end="\n"+"="*20+"\n")
    out = []
    out_s = head_s(x_s)
    out_m = head_m(x_m)
    out_l = head_l(x_l)
    print()
    print("After heads")
    # m[0]==p, m[1]==p_de (p, p_de have same shape)
    print(*[m[1].shape for m in [out_s, out_m, out_l]], sep="\n", end="\n"+"="*20+"\n")
    #print(*[m.shape for m in [out_s, out_m, out_l]], sep="\n", end="\n"+"="*20+"\n")
    out.append(out_s)
    out.append(out_m)
    out.append(out_l)

    if True: # training
        p, p_d = list(zip(*out))
        return p, p_d  # smalll, medium, large
    else:
        p, p_d = list(zip(*out))
        return p, torch.cat(p_d, 0)

if __name__ == "__main__":
    _test_model()