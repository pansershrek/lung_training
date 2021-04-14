# coding=utf-8
# project
DATA_PATH = "/home/lab402/p08922003/YOLOv4-pytorch/dataset_link"
PROJECT_PATH = "/home/lab402/p08922003/YOLOv4-pytorch/data"
DETECTION_PATH = "/home/lab402/p08922003/YOLOv4-pytorch"

Customer_DATA = {"NUM": 1, #your dataset number
                 "CLASSES":['aeroplane'],# your dataset class
        }
ABUS_DATA = {"NUM": 2, #your dataset number
                 "CLASSES":['background', 'tumor'],# your dataset class
        }

LUNG_DATA = {"NUM": 2, #your dataset number
                 "CLASSES":['background', 'tumor'],# your dataset class
        }

VOC_DATA = {"NUM": 20, "CLASSES":['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'],
        }

COCO_DATA = {"NUM":80,"CLASSES":['person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'backpack',
'umbrella',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'dining table',
'toilet',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush',]}

# ==========================

MODEL_TYPE = {"TYPE": 'YOLOv4'}  #YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4
MODEL_INPUT_CHANNEL = 1
CONV_TYPE = {"TYPE": 'DO_CONV'}  #conv type:DO_CONV or GENERAL

ATTENTION = {"TYPE": 'SEnet'}  #For CSPDarknet only: attention type:SEnet, SEnetConv, CBAM or NONE

# train
TRAIN = {
         "DATA_TYPE": 'LUNG',  #DATA_TYPE: LUNG, ABUS, VOC ,COCO or Customer
         "TRAIN_IMG_SIZE": (128,128,128), #(128, 128, 128), (16,128,128) (32,128,128), 5MM
         #"AUGMENT": True,
         #for 640
         "BATCH_SIZE": 8, # *8
         #for 96
         #"BATCH_SIZE": 4,
         "MULTI_SCALE_TRAIN": False,
         "IOU_THRESHOLD_LOSS": 0.5, # *0.5, 0.02
         #for 640
         "YOLO_EPOCHS": 300, #8: 425, 500, 800, *300
         "EARLY_STOPPING_EPOCH": None, # None or int
         #for 96
         #"YOLO_EPOCHS": 100,
         #"Mobilenet_YOLO_EPOCHS": 120,
         "OPTIMIZER": "SGD", # SGD / ADAM / ADABELIEF
         "USE_SGD_BEFORE_LOSS_LOWER_THAN_THRESH": False,
         "CHANGE_OPTIMIZER_THRESH": 50,
         "NUMBER_WORKERS": 0,  # *6, 0 # for resnest, workers==0 runs faster
         "MOMENTUM": 0.9,
         "WEIGHT_DECAY": 0.0001, # *0.0005
         "LR_INIT": 5e-5 , #SGD: 1e-4, *5e-5, 5e-6               #ADAM:
         "LR_END": 5e-8, #SGE: 1e-6, 5e-7, *5e-8, 5e-9          #ADAM:
         "CIOU_LOSS_MULTIPLIER": 1.0 , # *1.0, 2.0
         #for 640
         "WARMUP_EPOCHS": 5, #40  # or None
         "USING_RANDOM_CROP_TRAIN": True,
         "RANDOM_CROP_FILE_PREFIX": "random_crop_128x128x128_1.25x0.75x0.75", #"random_crop_128x128x128_1.25x0.75x0.75_fake1.25_from_5mm_max", # 5MM
         "RANDOM_CROP_SPACING": (1.25, 0.75, 0.75), #used in dataset.__getitem__ if using "fresh-cropped", 5MM
         "RANDOM_CROP_NCOPY": 20,
         "USE_5MM": False, # 5MM
         "ESTIMATE_5MM_ANCHOR": False, # if True, use ANCHORS_ORI to estimate ANCHORS_5MM rather than using ANCHORS_5MM directly

         "DO_FP_REDUCTION": True,
         "FP_REDUCTION_CROP_PREFIX": "false_positive", #"false_positive_fake_1.25_from_5mm_max", # 5MM 
         "FP_REDUCTION_CROP_NCOPY": 3, # 3 for original 1.25mm, and 5 for others
         #"FP_REDUCTION_TARGET_DATASET": "training", #WIP
         "FP_REDUCTION_START_EPOCH": 150,  # *150
         "FP_REDUCTION_INTERVAL": 1, # *1
         "FP_REDUCTION_MODE": "0,1", # 0,0 | *0,1 | 1,0 | 1,1 (cls_index, mix)
         "FP_REDUCTION_USE_ZERO_CONF": False, # whether to set conf in __create_label to 0 for fp; *False
         # 2021/3/4 v1: (目前 "0,1" + No_use_zero_conf + "try fp reduction loss" 表現最好)
         # 2021/3/6: "0,1" + use_zero_conf has similar(same) performance as 2021/3/4
         # 2021/3/8: "1,1" + use_zero_conf has super bad result (cpm<0.3)
         #for 96
         #"WARMUP_EPOCHS": 10 #40  # or None
         }


# val
VAL = {
        #"TEST_IMG_SIZE": 416,
        #"TEST_IMG_SIZE": (128,128,128), #(640, 160, 640),#(256, 64, 256), #
        #"TEST_IMG_BBOX_ORIGINAL_SIZE": (128,128,128),
        "TEST_IMG_SIZE": (128,128,128), # (128,128,128), (128,256,256)
        "TEST_IMG_BBOX_ORIGINAL_SIZE": (128,128,128), # (128,128,128), (128,256,256)
        "NUMBER_WORKERS": 4, # *5 or 0 (only use 0 for 5mm input with fast eval pkl)
        "CONF_THRESH": 0.015, #0.005, 0.01, *0.015, 0.03, 0.05, 0.1, 0.15  # score_thresh in utils.tools.nms (discard bbox < thresh)
        "NMS_THRESH": 0.15, #0.15, *0.3, 0.45 # iou_thresh in utils.tools.nms (if two bbox has iou > thresh, discard one of them)
        "BOX_TOP_K": 512, # highest number of bbox after nms # *256, 512
        "TP_IOU_THRESH": 0.15, # iou threshold to view a predicted bbox as TP # *0.15, 0.3

        "BATCH_SIZE": 1, # 1 or 8

        "USING_RANDOM_CROP_EVAL": False, # T/F
        "BATCH_1_EVAL": True, # T/F
        "RANDOM_CROPPED_VOI_FIX_SPACING": (1.25,0.75,0.75), #(z,y,x), 5MM
        "TEST_LUNG_VOI": True, # T/F, whether to use lung voi only during testing 
        "TEST_LUNG_VOI_IGNORE_INVALID_VOI": False, # T/F, whether to ignore pid listed in ${MASK_SAVED_PATH}/error.txt automatocaly
 
        # to load fake_1.25mm test img, do the following settings:
        # (1) let "FAST_EVAL_PKL_NAME"="fast_test_max_5.0x0.75x0.75.pkl"
        # (2) let "RANDOM_CROPPED_VOI_FIX_SPACING"=(1.25,0.75,0.75)
        # (3) let TRAIN["USE_5MM"]=True
        #  *** FAST_EVAL_PKL_NAME has no use, if TRAIN["USE_5MM"]==False ***
        "FAST_EVAL_PKL_NAME": "fast_test_max_5.0x0.75x0.75.pkl", # str or False, *fast_test_max_5.0x0.75x0.75.pkl
        "5MM_STACKING_STRATEGY": "max",     
        #"MULTI_SCALE_VAL": True,
        #"FLIP_VAL": True,
        #"Visual": True
        }

# model
MODEL = {#"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
         #   [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
         #   [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],  # Anchors for big obj(142,110),(192,243),(459,401)
         #"ANCHORS3D":[[[ 1.4375  ,  1.4375  ,  1.4375  ], #yolov4 original, "NOT FOR 彰基"
         #               [ 2.875   ,  2.875   ,  2.875   ],
         #               [ 3.5     ,  3.5     ,  3.5     ]],
         #               
         #               [[ 2.84375 ,  2.84375 ,  2.84375 ],
         #               [ 3.34375 ,  3.34375 ,  3.34375 ],
         #               [ 5.5625  ,  5.5625  ,  5.5625  ]],
         # 
         #               [[ 3.21875 ,  3.21875 ,  3.21875 ],
         #                [ 5.53125 ,  5.53125 ,  5.53125 ],
         #                [10.921875, 10.921875, 10.921875]]], #shape (STRIDES, Anchors_PER_SCALE, 3 element for 3D ZYX Anchor length)

        #See also: https://github.com/argusswift/YOLOv4-pytorch/issues/52

        "ANCHORS3D_ORI":  [ [[14/8., 27/8., 28/8.], [17/8., 37/8., 36/8.], [21/8., 51/8., 52/8.]], #calc from 彰基 at 1.25 mm random cropped img
                        [[26/16., 39/16., 40/16.], [33/16., 50/16., 48/16.], [34/16., 62/16., 64/16.]],
                        [[48/32., 57/32., 58/32.], [57/32., 73/32., 72/32.], [79/32., 90/32., 91/32.]] ],

        "ANCHORS3D_5MM":  [ [[2/4., 25/4., 26/4.], [4/4., 29/4., 31/4.], [5/4., 36/4., 36/4.]], #calc from 彰基 at 5.0 mm random cropped img
                        [[4/8., 40/8., 41/8.], [6/8., 51/8., 52/8.], [8/8., 45/8., 46/8.]],
                        [[10/16., 62/16., 63/16.], [16/16., 81/16., 86/16.], [16/16., 110/16., 119/16.]] ],        


        #"ANCHORS3D":  [ [[3/8., 6/8., 6/8.], [4/8., 7/8., 7/8.], [4/8., 8/8., 8/8.]],  #calc from luna
        #                [[6/16., 9/16., 9/16.], [6/16., 10/16., 10/16.], [7/16., 11/16., 11/16.]],
        #                [[8/32., 14/32., 14/32.], [11/32., 18/32., 18/32.], [16/32., 27/32., 26/32.]] ],
         "ANCHORS_PER_SCLAE":3,

         ## General params
         "BACKBONE": "ResNeSt", # ResNeSt | CSPDarknet
         "STRIDES":[4,8,16], # [4,8,16] for CSPDarknet; [4,8,16] for resnest # the last elements should == base_multiple
         "BASE_MULTIPLE":16, # == 2 ^ (#_stages in CSPDarknet)
         "USE_SACONV": False, ## RESNEST finished, CSPDarknet not implemented yet (i.e. no usage)
         "SACONV_USE_DEFORM": False, # True is buggy due to cpp 

         ## CSPDarknet related parameters
         "CSPDARKNET53_STEM_CHANNELS": 4, 
         "CSPDARKNET53_FEATURE_CHANNELS": [8,16,32,64] ,  #length of this should == (#_stages in CSPDarknet)
         "CSPDARKNET53_BLOCKS_PER_STAGE": [3,3,3] , #length of this should == (#_stages in CSPDarknet) - 1
         "VERBOSE_SHAPE": False,

         ## ResNeSt related parameters
         "RESNEST_STEM_WIDTH": 16, # similar to stem_channel, just a little bit different
         "RESNEST_EXPANSION": 2,  # orginal: 4 (higher -> less param)
         "RESNEST_FEATURE_CHANNELS": (24, 64, 128), # length == #_stages-1 == 3
         "RESNEST_BLOCKS_PER_STAGE": (2, 3, 3, 3), # length == #_stages == 4
         "RESNEST_STRIDE_PER_LAYER": (1, 2, 2),
         "RESNEST_EXTRA_ATTENTION": "SEnetConv", #attention type:SEnet, SEnetConv, CBAM or NONE
         }

def _check():
        if VAL["BATCH_1_EVAL"]:
                assert VAL["BATCH_SIZE"] == 1
                assert VAL["USING_RANDOM_CROP_EVAL"] == False
        if "max" in TRAIN["RANDOM_CROP_FILE_PREFIX"]:
                assert VAL["5MM_STACKING_STRATEGY"] == "max"
        elif "mean" in TRAIN["RANDOM_CROP_FILE_PREFIX"]:
                assert VAL["5MM_STACKING_STRATEGY"] == "mean"
        assert list(TRAIN["RANDOM_CROP_SPACING"]) == list(VAL["RANDOM_CROPPED_VOI_FIX_SPACING"])

def modify():
        # (1) anchor box
        if TRAIN["USE_5MM"] and TRAIN["ESTIMATE_5MM_ANCHOR"]:
                tmp = MODEL["ANCHORS3D_ORI"]
                for scale in tmp:
                        for anc in scale:
                                anc[0] /= 5/1.25
                MODEL["ANCHORS3D"] = tmp
        elif TRAIN["USE_5MM"] and TRAIN['RANDOM_CROP_SPACING'][0] == 5.0: 
                MODEL["ANCHORS3D"] = MODEL["ANCHORS3D_5MM"] 
        else:
                MODEL["ANCHORS3D"] = MODEL["ANCHORS3D_ORI"]

_check()
modify()

"""
UPDATE_NOTE:
1. WIP: add gradient for bbox with label=0 (change loss_conf in yolo_loss.py)
2. Add FP reduction by using the pre-cropped negative sample pool (config 4) (both fp_dataset_0_conf and online hardmining seems not good (v1))
3. Try calculate loss_conf using [ only top k hard negative pred bbox/grid and all pos bbox/grid ] (very bad if use zero_conf + cls==1, but ok if use zero_conf + cls==0)
4. Try ResNeSt (config 5)
5. Try SAConv with ResNeSt (replace all 3x3x3 conv3d to SAConv3d) (Super unstable, and not so good)
6. Try max/mean 5mm slice thickness data (using new dataset file and anchor boxes)
7. Try first version of 5mm image training with fp reduction (stack_func=max), "5_mm_max_config_1" (sens=0.5,0.4 around fp=4,2)
8. Try estimate 5mm anchors using ori_anchors; "train_5mm_max_config_1.2_estimated_anchor"
9. Try run testing on fake_1.25mm(interpolated from 5mm) with model trained from true 1.255_mm crops.
10. Try 5mm mean training (exclude 5mm不清楚+5mm看不到) (worse than max, even exlude both data)
11. Try fake 1.25 mm crops training + fake 1.25 mm testing, generated from 5mm max data (set TRAIN['use_5mm']=True, VAL['FAST_EVAL_PKL_NAME']!=False)
12. Try SE-block in resnest (original 1.25mm) (config 5.7) [SE-Conv/SEnet cause OOM in testing] (edit: may try lower reduction parameter in SE layer!!)


WHAT'S NEW:
** Try SE-block in resnest (original 1.25mm) (config 5.7) [SE-Conv/SEnet cause OOM in testing] (edit: may try lower reduction parameter in SE layer!!)
"""