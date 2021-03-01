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

ATTENTION = {"TYPE": 'SEnet'}  #attention type:SEnet, SEnetConv, CBAM or NONE

# train
TRAIN = {
         "DATA_TYPE": 'LUNG',  #DATA_TYPE: LUNG, ABUS, VOC ,COCO or Customer
         "TRAIN_IMG_SIZE": (128,128,128), #(128, 128, 128), (128,256,256)
         #"AUGMENT": True,
         #for 640
         "BATCH_SIZE": 8, # *8
         #for 96
         #"BATCH_SIZE": 4,
         "MULTI_SCALE_TRAIN": False,
         "IOU_THRESHOLD_LOSS": 0.5, # *0.5, 0.02
         #for 640
         "YOLO_EPOCHS": 400, #8: 425, 500, 800
         #for 96
         #"YOLO_EPOCHS": 100,
         #"Mobilenet_YOLO_EPOCHS": 120,
         "OPTIMIZER": "SGD", 
         "USE_SGD_BEFORE_LOSS_LOWER_THAN_5": False,
         "NUMBER_WORKERS": 6,  # *6
         "MOMENTUM": 0.9,
         "WEIGHT_DECAY": 0.0001, # *0.0005
         "LR_INIT": 5e-5 , #SGD: 1e-4, *5e-5, 5e-6               #ADAM:
         "LR_END": 5e-8, #SGE: 1e-6, *5e-7, 5e-8, 5e-9          #ADAM:
         "CIOU_LOSS_MULTIPLIER": 1.0 , # *1.0, 2.0
         #for 640
         "WARMUP_EPOCHS": 5, #40  # or None
         "USING_RANDOM_CROP_TRAIN": True,
         "RANDOM_CROP_FILE_PREFIX": "random_crop_128x128x128_1.25x0.75x0.75",
         "RANDOM_CROP_SPACING": (1.25, 0.75, 0.75), #used in dataset.__getitem__ 
         "RANDOM_CROP_NCOPY": 20,

         "DO_FP_REDUCTION": True,
         #"FP_REDUCTION_TARGET_DATASET": "training", #WIP
         "FP_REDUCTION_START_EPOCH": 0,
         "FP_REDUCTION_INTERVAL": 1, 
         "FP_REDUCTION_MODE": "0,1", # 0,0 | 0,1 | 1,0 (conf, cls_index)
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
        "NUMBER_WORKERS": 2,
        "CONF_THRESH": 0.015, #0.005, 0.01, *0.015, 0.03, 0.05, 0.1, 0.15  # score_thresh in utils.tools.nms (discard bbox < thresh)
        "NMS_THRESH": 0.15, #0.15, *0.3, 0.45 # iou_thresh in utils.tools.nms (if two bbox has iou > thresh, discard one of them)
        "BOX_TOP_K": 512, # highest number of bbox after nms # *256, 512
        "TP_IOU_THRESH": 0.15, # iou threshold to view a predicted bbox as TP # *0.15

        "BATCH_SIZE": 1, # 1 or 8

        "USING_RANDOM_CROP_EVAL": False, # T/F
        "BATCH_1_EVAL": True, # T/F
        "RANDOM_CROPPED_VOI_FIX_SPACING": (1.25,0.75,0.75), #(z,y,x)
        "TEST_LUNG_VOI": True, # T/F, whether to use lung voi only during testing 
        "TEST_LUNG_VOI_IGNORE_INVALID_VOI": False, # T/F, whether to ignore pid listed in ${MASK_SAVED_PATH}/error.txt automatocaly
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

        "ANCHORS3D":  [ [[14/8., 27/8., 28/8.], [17/8., 37/8., 36/8.], [21/8., 51/8., 52/8.]], #calc from 彰基
                        [[26/16., 39/16., 40/16.], [33/16., 50/16., 48/16.], [34/16., 62/16., 64/16.]],
                        [[48/32., 57/32., 58/32.], [57/32., 73/32., 72/32.], [79/32., 90/32., 91/32.]] ],

        #"ANCHORS3D":  [ [[3/8., 6/8., 6/8.], [4/8., 7/8., 7/8.], [4/8., 8/8., 8/8.]],  #calc from luna
        #                [[6/16., 9/16., 9/16.], [6/16., 10/16., 10/16.], [7/16., 11/16., 11/16.]],
        #                [[8/32., 14/32., 14/32.], [11/32., 18/32., 18/32.], [16/32., 27/32., 26/32.]] ],
         "STRIDES":[4,8,16], # original: [8,16,32] # the last elements should == base_multiple
         "ANCHORS_PER_SCLAE":3,
         "BASE_MULTIPLE":16, # == 2 ^ (#_stages in CSPDarknet)
         "CSPDARKNET53_STEM_CHANNELS": 4, 
         "CSPDARKNET53_FEATURE_CHANNELS": [8,16,32,64] ,  #length of this should == (#_stages in CSPDarknet)
         "CSPDARKNET53_BLOCKS_PER_STAGE": [3,3,3] , #length of this should == (#_stages in CSPDarknet) - 1
         "VERBOSE_SHAPE": False,
         }

def _check():
        if VAL["BATCH_1_EVAL"]:
                assert VAL["BATCH_SIZE"] == 1
                assert VAL["USING_RANDOM_CROP_EVAL"] == False

_check()

"""
UPDATE_NOTE:
1. WIP: add gradient for bbox with label=0 (change loss_conf in yolo_loss.py)
2. Add FP reduction by using the pre-cropped negative sample pool

WHAT'S NEW:
** Add FP reduction by using the pre-cropped negative sample pool
"""