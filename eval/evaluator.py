import shutil
from eval import voc_eval
from utils.data_augment import *
from utils.tools import *
from tqdm import tqdm
from utils.visualize import *
from utils.heatmap import imshowAtt
import config.yolov4_config as cfg
import time
import torch.nn.functional as F
import warnings
current_milli_time = lambda: int(round(time.time() * 1000))
class Evaluator(object):
    def __init__(self, model, showatt, pred_result_path, box_top_k, conf_thresh=None):
        if cfg.TRAIN["DATA_TYPE"] == 'VOC':
            self.classes = cfg.VOC_DATA["CLASSES"]
        elif cfg.TRAIN["DATA_TYPE"] == 'COCO':
            self.classes = cfg.COCO_DATA["CLASSES"]
        elif cfg.TRAIN["DATA_TYPE"] == 'ABUS':
            self.classes = cfg.ABUS_DATA["CLASSES"]
        elif cfg.TRAIN["DATA_TYPE"] == 'LUNG':
            self.classes = cfg.LUNG_DATA["CLASSES"]
        else:
            self.classes = cfg.Customer_DATA["CLASSES"]
        self.pred_result_path = pred_result_path
        self.val_data_path = os.path.join(cfg.DATA_PATH, 'VOCtest-2007', 'VOCdevkit', 'VOC2007')

        self.val_shape = cfg.VAL["TEST_IMG_SIZE"]
        self.model = model
        self.device = 'cuda'#next(model.parameters()).device
        self.__visual_imgs = 0
        self.showatt = showatt
        self.inference_time = 0.

        self.conf_thresh = cfg.VAL["CONF_THRESH"] if conf_thresh==None else conf_thresh
        self.nms_thresh = cfg.VAL["NMS_THRESH"]
        self.box_top_k = box_top_k
        self.batch_1_eval = cfg.VAL["BATCH_1_EVAL"]
        assert cfg.VAL["NODULE_RANKING_STRATEGY"] in ("conf_only", "conf+class"), "Unknown ranking strategy '{}'".format(cfg.VAL["NODULE_RANKING_STRATEGY"])
        self.use_conf_x_prob = True if cfg.VAL["NODULE_RANKING_STRATEGY"]=="conf+class" else False

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)

        txtpath = "./output/detection-results/"
        if not os.path.exists(txtpath):
            os.mkdir(txtpath)
        os.mkdir(self.pred_result_path)
        print('val img size is {}'.format(self.val_shape))
        for img_ind in tqdm(img_inds):
            img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind+'.jpg')
            img = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test)

            f = open("./output/detection-results/" + img_ind + ".txt", "w")
            for bbox in bboxes_prd:
                coor = np.array(bbox[:6], dtype=np.int32)
                score = bbox[6]
                class_ind = int(bbox[7])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                zmin, ymin, xmin, zmax, ymax, xmax = map(str, coor)
                s = ' '.join([img_ind, score, zmin, ymin, xmin, zmax, ymax, xmax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as r:
                    r.write(s)
                f.write("%s %s %s %s %s %s\n" % (class_name, score, str(zmin), str(ymin), str(xmin), str(zmax), str(ymax), str(xmax)))
            f.close()
        self.inference_time = 1.0 * self.inference_time / len(img_inds)
        return self.calc_APs(), self.inference_time

    def store_bbox(self, img_ind, bboxes_prd):
        #'/data/bruce/CenterNet_ABUS/results/prediction/new_CASE_SR_Li^Ling_1073_201902211146_1.3.6.1.4.1.47779.1.002.npy'
        boxes = bboxes_prd[..., :7]
        if len(boxes)>0:
            boxes=boxes
        np.save(os.path.join(self.pred_result_path, img_ind), boxes)
        # f = open("./output/detection-results/" + img_ind + ".txt", "w")
        # for bbox in bboxes_prd:
        #     coor = np.array(bbox[:6], dtype=np.int32)
        #     score = bbox[6]
        #     class_ind = int(bbox[7])

        #     class_name = self.classes[class_ind]
        #     score = '%.4f' % score
        #     xmin, ymin, xmax, ymax = map(str, coor)
        #     s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'

        #     with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as r:
        #         r.write(s)
        #     f.write("%s %s %s %s %s %s\n" % (class_name, score, str(xmin), str(ymin), str(xmax), str(ymax)))
        # f.close()

    def get_bbox(self, img, multi_test=False, flip_test=False, shape_before_pad=[0,0,0]):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
            box_raw_data = []
        else:
            bboxes, box_raw_data = self.__predict(img, self.val_shape, (0, np.inf), shape_before_pad)

        bboxes, log_txt = nms(bboxes, score_threshold=self.conf_thresh, iou_threshold=self.nms_thresh, box_top_k=self.box_top_k)

        return bboxes, box_raw_data, log_txt

    def __predict(self, img, test_shape, valid_scale, shape_before_pad=[0,0,0]):
        org_img = img
        #print("test_img shape in evaluator.py:",org_img.shape)
        if len(org_img.size())==4:
            _, org_d, org_h, org_w = org_img.size()
            org_shape = (org_d, org_h, org_w)
            img = img.unsqueeze(0)
            if (test_shape==org_shape) or (self.batch_1_eval):
                pass
            else:
                raise TypeError(f"test img has shape {org_shape} != test_shape = {test_shape}")
                #warnings.warn(f"test img has shape {org_shape} != test_shape = {test_shape}")
                img = F.interpolate(img, size=test_shape, mode='trilinear')
        else:
            raise TypeError(f"2D input detected")
            _, org_h, org_w = org_img.size()
            org_shape = (org_h, org_w)
            img = img.unsqueeze(0)
            if (test_shape==org_shape):
                pass
            else:
                img = F.interpolate(img, size=test_shape, mode='bilinear')
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            if self.showatt: _, p_d, beta = self.model(img)
            else: _, p_d = self.model(img)
            self.inference_time += (current_milli_time() - start_time)
        pred_bbox = p_d.squeeze().cpu().numpy()
        if self.batch_1_eval:
            bboxes = self.__convert_pred(pred_bbox, org_shape, org_shape, valid_scale, shape_before_pad)
        else:
            bboxes = self.__convert_pred(pred_bbox, test_shape, org_shape, valid_scale)
        if self.showatt and len(img):
            self.__show_heatmap(beta[2], np.copy(org_img.cpu().numpy()))
        #return bboxes, p_d # GRAM BURDEN
        return bboxes, p_d.cpu()

    def __show_heatmap(self, beta, img):
        imshowAtt(beta, img)

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale, shape_before_pad=(0,0,0)):
        """
        Filter out the prediction box to remove the unreasonable scale of the box
        """
        if len(org_img_shape)==3:
            pred_coor = xyzwhd2xyzxyz(pred_bbox[:, :6])
            pred_conf = pred_bbox[:, 6]
            pred_prob = pred_bbox[:, 7:]
            # (1)
            # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
            # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
            org_d, org_h, org_w = org_img_shape

            if not self.batch_1_eval:
                resize_ratio = 1.0 * min([test_input_size[i] / org_img_shape[i] for i in range(3)])
                dd = (test_input_size[0] - resize_ratio * org_d) / 2
                dh = (test_input_size[1] - resize_ratio * org_h) / 2
                dw = (test_input_size[2] - resize_ratio * org_w) / 2
            else:
                resize_ratio = 1.0
                dd = dh = dw = 0
            #pred_coor[:, 0::3] = 1.0 * (pred_coor[:, 0::3] - dd) / resize_ratio
            #pred_coor[:, 1::3] = 1.0 * (pred_coor[:, 1::3] - dh) / resize_ratio
            #pred_coor[:, 2::3] = 1.0 * (pred_coor[:, 2::3] - dw) / resize_ratio
            #for i in range(3):
            #    pred_coor[:, i::3] = 1.0 * (pred_coor[:, i::3]) * (org_img_shape[i] / test_input_size[i])


        else:
            pred_coor = xywh2xyxy(pred_bbox[:, :4])
            pred_conf = pred_bbox[:, 4]
            pred_prob = pred_bbox[:, 5:]
            # (1)
            # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
            # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
            org_h, org_w = org_img_shape
            resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
            dw = (test_input_size - resize_ratio * org_w) / 2
            dh = (test_input_size - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        if len(org_img_shape)==3:
            # (2)Crop off the portion of the predicted Bbox that is beyond the original image
            pred_coor = np.concatenate([np.maximum(pred_coor[:, :3], [0, 0, 0]),
                                        np.minimum(pred_coor[:, 3:], [org_d - 1, org_h - 1, org_w - 1])], axis=-1)
            # (3)Sets the coor of an invalid bbox to 0
            invalid_mask = np.logical_or((pred_coor[:, 1] > pred_coor[:, 4]), (pred_coor[:, 2] > pred_coor[:, 5]))
            pred_coor[invalid_mask] = 0
            invalid_mask = (pred_coor[:, 0] > pred_coor[:, 3])
            pred_coor[invalid_mask] = 0

            # (4)Remove bboxes that are not in the valid range
            bboxes_scale = np.multiply.reduce(pred_coor[:, 3:6] - pred_coor[:, 0:3], axis=-1)
            v_scale_3 = np.power(valid_scale, 3.0)
            scale_mask = np.logical_and((v_scale_3[0] < bboxes_scale), (bboxes_scale < v_scale_3[1]))
        else:
            # (2)Crop off the portion of the predicted Bbox that is beyond the original image
            pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
            # (3)Sets the coor of an invalid bbox to 0
            invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
            pred_coor[invalid_mask] = 0

            # (4)Remove bboxes that are not in the valid range
            bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
            scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))


        # (5)Remove bboxes whose score is below the score_threshold
        classes = np.argmax(pred_prob, axis=-1) #predicted class idx
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes] # score = pred_prob_from_class * pred_conf (p.s. it only used sigmoid, no softmax in YoloHead)
        if self.use_conf_x_prob: #only calculated scores for class "1"
            invalid_cls_mask = classes!=1
            s = invalid_cls_mask.sum()
            #print( "cls1:others = {}:{}".format(len(classes)-s, s) )
            scores[invalid_cls_mask] = 0
            #pass

        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        # (6)ccy: Remove bboxes that touch padded area for batch_1_eval
        if self.batch_1_eval:
            shape_before_pad = np.array(shape_before_pad)
            assert shape_before_pad.shape == (3,)
            invalid_mask_z = ((coors[:, 0] + coors[:, 3])/2 > shape_before_pad[0]-1) # (z1+z2)/2 > ori_z
            invalid_mask_y = ((coors[:, 1] + coors[:, 4])/2 > shape_before_pad[1]-1) # (y1+y2)/2 > ori_y
            invalid_mask_x = ((coors[:, 2] + coors[:, 5])/2 > shape_before_pad[2]-1) # (x1+x2)/2 > ori_x
            invalid_mask = invalid_mask_z + invalid_mask_y + invalid_mask_x # the "+" here acts like "logical_or"
            mask = np.invert(invalid_mask)
            coors = coors[mask]
            scores = scores[mask]
            classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        #print("bboxes[0]:", bboxes[0])
        return bboxes
    def clear_predict_file(self):
        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)
    def calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        Calculate ap values for each category
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        raise NotImplementedError()
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        APs = {}
        Recalls = {}
        Precisions = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            Recalls[cls] = R
            Precisions[cls] = P
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs
