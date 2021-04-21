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
