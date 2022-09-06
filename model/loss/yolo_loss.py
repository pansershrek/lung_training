import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import tools
#import config.yolov4_config as cfg


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)  ## it calc loss based on "p", not "p_d", so it does need extra sigmoid

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

class BinaryFocalLossWithSigmoid(nn.Module):
    """
    自己刻的Focal loss，用於Binary classification (就是不像CELoss, 這個沒有softmax)
    但是對於Batch中的每個sample(根據其y=1 or 0),有類似於focal loss去給...
        weight=alpha_t*(1-p_t)**gamma
            , where "p_t = p if y==1 else (1-p)"
                    "alpha_t = alpha if y==1 else (1-alpha)"

    p_t越大，代表模型分類結果越正確(easy case)，越小則越錯誤(hard case)

    參見:
        https://arxiv.org/pdf/1708.02002.pdf
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    def __init__(self, gamma=2.0, alpha=0.5, pos_weight=None, reduction="mean"):
        """
        weight=alpha_t*(1-p_t)**gamma
            , where "p_t = p if y==1 else (1-p)"
                    "alpha_t = alpha if y==1 else (1-alpha)"

        **Argument
            gamma: scalar, 定義域: [0,inf)
            alpha: scalar, 定義域: [0,1]
            pos_weight: FloatTensor of shape (C,)
            reduction: "none"|"mean"|"sum", 
            
        **Note
            1. 原始paper中，最好的gamma=2.0, alpha=0.25
            2. B=batch_size, C=class(這不是多分類問題，而是你要做幾個不同的二分類(on single head)，比如ER/PR/HER2預測+/-，是個3個不同的二分類問題，C=3)
            3. 由於會呼叫torch.BCEWithLogitsLoss, 輸入x請不要先做sigmoid操作!
            4. weight != pos_weight， weight是(B,C)的張量，每個sample的每個class有自己的weighy。
            5. pos_weight是(C,)的張量，表示加權在**正樣本**上的權重，用來平衡C個二分類問題中, 正負樣本差太多的問題，若 positive:negative = 300:100, 該C的pos_weight應設為1/3
            6. alpha是在改變正負樣本比重，由於已有pos weight，設成0.5
        """
        super(BinaryFocalLossWithSigmoid, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction
 
    def forward(self, input, target):
        """
        x/input: 還沒過sigmoid的 FloatTensor with shape (B,C)
        y/target: 裡面只有1/0的 LongTensor with shape (B,C) 
        **Note
            B=batch_size, C=class(這不是多分類問題，而是你要做幾個不同的二分類，比如ER/PR/HER2預測+/-，是個3個不同的二分類問題，C=3)
            
            e.g. B=1,C=3
                x=Tensor([-1.233, 2.33, 5.322]), y=LongTensor([0,1,1]) #此y對應到該case是ER(-)/PR(+)/HER2(+)
        """
        x = input
        y = target
        ## 計算w, 只是凸顯難易樣本的weighting，(原paper)不需要backprop到它
        p = torch.sigmoid(x) # (B,C)
        num_classes = p.shape[1] # C
        p_t = p*y + (1-p)*(1-y) # p_t = p if y==1 else (1-p) (根據focal loss paper中的定義), (B,C)
        alpha_t = self.alpha*y + (1-self.alpha)*(1-y) # (B,C)
        weight = alpha_t * (1-p_t).pow(self.gamma) # (B,C)
        weight = weight.detach()
        ## 丟進BCEWithLogits
        loss = F.binary_cross_entropy_with_logits(x, y.float(), weight=weight, pos_weight=self.pos_weight, reduction=self.reduction)
        return loss

class YoloV4Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5, dims=2, hard_negative_mining=False):
        super(YoloV4Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides
        self.__dims=dims
        self.hard_negative_mining = hard_negative_mining

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides

        loss_s, loss_s_ciou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        loss_m, loss_m_ciou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, strides[1])
        loss_l, loss_l_ciou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox,
                                                               lbboxes, strides[2])


        loss = loss_l + loss_m + loss_s
        loss_ciou = loss_s_ciou + loss_m_ciou + loss_l_ciou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_ciou, loss_conf, loss_cls



    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        dims = self.__dims
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        #FOCAL = FocalLoss(gamma=2, alpha=0.5, reduction="none")
        if dims==3:
            batch_size, grid = p.shape[0], p.shape[1:4]
            img_size = stride * torch.tensor([_ for _ in grid])
            p_conf = p[..., 6:7]
            p_cls = p[..., 7:]

            p_d_xywh = p_d[..., :6]

            label_xywh = label[..., :6]
            label_obj_mask = label[..., 6:7]
            label_cls = label[..., 8:]
            label_mix = label[..., 7:8]
        else:
            batch_size, grid = p.shape[0], p.shape[1:3]
            img_size = stride * torch.tensor([_ for _ in grid])
            p_conf = p[..., 4:5]
            p_cls = p[..., 5:]

            p_d_xywh = p_d[..., :4]

            label_xywh = label[..., :4]
            label_obj_mask = label[..., 4:5]
            label_cls = label[..., 6:]
            label_mix = label[..., 5:6]
        

        # loss iou
        #print("At yololoss.py:")
        #print("p_d_xywh: {}; label_xywh:{}".format(p_d_xywh.shape, label_xywh.shape))
        diou = tools.CIOU_xyzwhd_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        if (1):
            diou = diou * 1

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        if dims==3:
            bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 3:4] * label_xywh[..., 4:5] * label_xywh[..., 5:6] / (img_size[0] * img_size[1] * img_size[2])
        else:
            bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size[0] * img_size[1])
        loss_ciou = label_obj_mask * bbox_loss_scale * (1.0 - diou) * label_mix

        #set(((1-iou)*label_mix).detach().cpu().numpy().flatten())
        # loss confidence
        #iou = tools.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        if dims==3:
            iou = tools.IOU_xywh_torch(p_d_xywh.unsqueeze(5), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1))
        else:
            iou = tools.IOU_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1))
        #t = torch.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        iou_max = iou.max(-1, keepdim=True)[0]

        if (0): #original (get 0.9 sens, but FP too high)
            """
            ##PSEUDO
            if max(pred_iou) < self.__iou_threshold_loss:
                calc loss using all bboxes' p_conf
            else:
                calc loss using p_conf of bboxes with label_conf==1
            """
            pos_weight = None
            FOCAL = BinaryFocalLossWithSigmoid(gamma=2.0, alpha=0.5, pos_weight=pos_weight, reduction="none")
            label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()
            loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix
        elif (1): # try do fp reduction, bbox with label=0 now has gradient!
            """
            ##PSEUDO
            calc loss using (p_conf of bboxes with label_conf==1) AND (1-p_conf of bboxes with label_conf==0) 
            """
            tot = label_obj_mask.shape.numel()
            pos = label_obj_mask.sum().detach()
            pos_weight = torch.tensor([(tot-pos)/(pos+1e-9)], device=p.device) ## #neg/#pos
            pos_weight = pos_weight.clamp(1.0, 20.0) ## 20.0 is calculated by anchor size
            FOCAL = BinaryFocalLossWithSigmoid(gamma=2.0, alpha=0.5, pos_weight=pos_weight, reduction="none")
            label_noobj_mask = (1.0 - label_obj_mask)
            loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input = 1-p_conf, target=label_noobj_mask)) * label_mix
        else: # WIP: top k hard negative (grid/bbox) mining 
            """
            topk hard negative minin (poor performance)
            """
            # p_conf of shape (B, Grid,Grid,Grid, Anchor, 1)
            top_k = 2  # how many hardest neg grid involved in calculating loss_conf
            pos_weight = None
            FOCAL = BinaryFocalLossWithSigmoid(gamma=2.0, alpha=0.5, pos_weight=pos_weight, reduction="none")
            label_noobj_mask = (1.0 - label_obj_mask) # shape (B, G,G,G, Anc, 1)
            ##p_conf_of_noobjs = p_conf[label_noobj_mask] # shape (#bbox,)
            Anc = p_conf.shape[-2]
            p_conf_of_noobjs = (p_conf * label_noobj_mask).reshape(batch_size, -1, Anc, 1)  # (B, G*G*G, Anc, 1)
            ind = p_conf_of_noobjs.sort(dim=1, descending=True)[1][:, :top_k, :, :] # (B, top_k, Anc, 1)
            p_conf_top_k_hard_negative =  p_conf_of_noobjs.gather(dim=1, index=ind) # (B, top_k, Anc, 1)

            if (1): # v1, bad if using fp_dataset_zero_conf (maybe stemmed from p_conf is not in [0,1] (why?))
                label_top_k_hard_negative = torch.zeros((batch_size, top_k, Anc, 1), dtype=torch.long, device=p.device) # (B, top_k, Anc, 1)
                loss_conf_neg = FOCAL(input = p_conf_top_k_hard_negative, target=label_top_k_hard_negative).sum() # (B, top_k, Anc, 1) -> (1,)
            else: # v2, not tried yet (maybe no need sigmoid within loss func?)
                label_top_k_hard_negative = torch.ones((batch_size, top_k, Anc, 1), dtype=torch.long, device=p.device) # (B, top_k, Anc, 1)
                loss_conf_neg = FOCAL(input = 1-p_conf_top_k_hard_negative, target=label_top_k_hard_negative).sum() # (B, top_k, Anc, 1) -> (1,)
            
            loss_conf_pos = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask)).sum() # (B, G, G, G, Anc, 1) -> (1,)
            loss_conf = loss_conf_pos + loss_conf_neg
            
            n_pos = label_obj_mask.sum().detach().cpu().item() # number of pos bbox/grid used in loss_conf
            n_neg = label_top_k_hard_negative.numel()
            if (0): #debug
                print("n_pos/n_neg = {}/{} = {}".format(n_pos, n_neg, n_pos/n_neg))



        


        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix


        if (self.hard_negative_mining) and (batch_size>=3): #hard example mining on p_conf
            topk = min(batch_size, 3) # top3
            ndim = loss_conf.ndim
            # sum loss over anchors, scales, but not samples
            loss_conf = torch.sum(loss_conf, dim=list(range(ndim))[1:]) # (B,A,A,A,S,1) -> (B)
            loss_conf, _ = torch.sort(loss_conf, dim=0, descending=True) # sort from max to min, (B)
            loss_conf = loss_conf[:topk] # (B,) -> (topk,)
            loss_conf = torch.sum(loss_conf) / loss_conf.shape[0] # (topk,) -> float

        else:
            loss_conf = (torch.sum(loss_conf)) / batch_size

        loss_ciou = (torch.sum(loss_ciou)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_ciou + loss_conf + loss_cls
        #bboxes[0, 0]
        #(p_cls[0, 11, 2, 24, 0] * 1000).long()
        #(iou[0, 11, 2, 24, 0] * 10000).long()
        #(p_d[0, 11, 2, 24, 0] * 1000).long()
        return loss, loss_ciou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.build_model import Yolov4
    net = Yolov4()

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3,  52, 52, 3,26)
    label_mbbox = torch.rand(3,  26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3,26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    #loss, loss_xywh, loss_conf, loss_cls = YoloV4Loss(cfg.MODEL["ANCHORS3D"], cfg.MODEL["STRIDES"])(p, p_d, label_sbbox,
    #                                label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
