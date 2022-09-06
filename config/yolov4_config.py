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
         "BATCH_SIZE": 8, # *8, 16
         #for 96
         #"BATCH_SIZE": 4,
         "MULTI_SCALE_TRAIN": False,
         "IOU_THRESHOLD_LOSS": 0.5, # *0.5, 0.02
         #for 640
         "YOLO_EPOCHS": 300, #8: 425, 500, 800, *300
         "EARLY_STOPPING_EPOCH": 250, # None or int 200
         #for 96
         #"YOLO_EPOCHS": 100,
         #"Mobilenet_YOLO_EPOCHS": 120,
         "OPTIMIZER": "SGD", # SGD / ADAM / ADABELIEF
         "USE_SGD_BEFORE_LOSS_LOWER_THAN_THRESH": False,
         "CHANGE_OPTIMIZER_THRESH": 50,
         "NUMBER_WORKERS": 4,  # *6, 0, 4 # for resnest, workers==0 runs faster
         "MOMENTUM": 0.9,
         "WEIGHT_DECAY": 0.0001, # *0.0005
         "LR_INIT": 5e-5 , #SGD: 1e-4, *5e-5, 5e-6               #ADAM:
         "LR_END": 5e-8, #SGE: 1e-6, 5e-7, *5e-8, 5e-9          #ADAM:
         "CIOU_LOSS_MULTIPLIER": 1.0 , # *1.0, 2.0
         #for 640
         "WARMUP_EPOCHS": 5, #40  # or None
         "USING_RANDOM_CROP_TRAIN": False,
         "RANDOM_CROP_FILE_PREFIX": "random_crop_128x128x128_1.25x0.75x0.75", #"random_crop_128x128x128_1.25x0.75x0.75_fake1.25_from_5mm_max", # 5MM
         "RANDOM_CROP_SPACING": (1.25, 0.75, 0.75), #used in dataset.__getitem__ if using "fresh-cropped", 5MM
         "RANDOM_CROP_NCOPY": 20,
         "USE_5MM": False, # 5MM
         "USE_2.5MM": False, # 2.5MM (Either 5/2.5mm is ok, but not both at the same time)
         "ESTIMATE_5MM_ANCHOR": False, # if True, use ANCHORS_ORI to estimate ANCHORS_5MM rather than using ANCHORS_5MM directly
        
        #####
         "DO_FP_REDUCTION": False,
         #"FP_REDUCTION_CROP_PREFIX": "another_data_128x128x128_1.25x0.75x0.75", #"false_positive", #"false_positive_fake_1.25_from_5mm_max", # 5MM 
         "FP_REDUCTION_CROP_NCOPY": 5, # 3 for original 1.25mm, and 5 for others
         #"FP_REDUCTION_TARGET_DATASET": "training", #WIP
         #"FP_REDUCTION_START_EPOCH": 150,  # *150, 2, *100
         "FP_REDUCTION_INTERVAL": 1, # *1
         "FP_REDUCTION_MODE": "0,1", # 0,0 | *0,1 | 1,0 | 1,1 (cls_index, mix)
         "FP_REDUCTION_USE_ZERO_CONF": False, # whether to set conf in __create_label to 0 for fp; *False
         
         "EXTRA_FP_USAGE": "eval_only", # "eval_only"/None  # whether to consider extra tp/fp during testing
         "CHANGE_FP_REDUCTION_FOLDER_ROOT": True, # if True, set NPY_SAVED_PATH as root folder (used mainly for "another_data"), else NEGATIVE_NPY_SAVED_PATH
         #####
         "ITERATIVE_FP_UPDATE": False, # whether update fp crops every 2000 steps
         "ITERATIVE_FP_UPDATE_START_EPOCH": 200, # *200 (6/11 update: normally == FP_REDUCTION_START_EPOCH)
        
         "HORIZONTAL_FLIP_RATE": 0.5, # 0.3, 0.5. *0.0 prob to flip crops horizontally during training (only be appllied for random_crops dataset)
         
         "USE_EXTRA_ANNOTATION": False, # whether to use another data
         "EXTRA_ANNOTATION_CROP_PREFIX": "another_data_128x128x128_1.25x0.75x0.75",
         "EXTRA_ANNOTATION_NCOPY": 5,
         "USE_EXTRA_ANNOTATION_START_EPOCH": 100, 

         "USE_COPY_PASTE": False, # if True, then USE_EXTRA_ANNOTATION should also be True (not useful)
         "COPY_PASTE_EXCLUDE_PIDS_TXT": "D:/CH/LungDetection/copy_paste_invalid_pids.txt",
         "COPY_PASTE_CROP_PREFIX": "copy_paste_128x128x128_1.25x0.75x0.75",
         "COPY_PASTE_NCOPY": 3,
        
         "DO_HARD_NEGATIVE_MINING": False, #OHEM
         "HARD_NEGATIVE_MINING_START_EPOCH": 150,


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
        "TP_IOU_THRESH": 0.3, # iou threshold to view a predicted bbox as TP # 0.15, 0.3
        "NODULE_RANKING_STRATEGY": "conf_only", # conf_only|conf+class (not much difference in cpm)

        "BATCH_SIZE": 1, # 1 or 8

        "USING_RANDOM_CROP_EVAL": False, # T/F
        "BATCH_1_EVAL": True, # T/F
        "RANDOM_CROPPED_VOI_FIX_SPACING": (1.25,0.75,0.75), #(z,y,x), 5MM
        "TEST_LUNG_VOI": False, # T/F, whether to use lung voi only during testing 
        "TEST_LUNG_VOI_IGNORE_INVALID_VOI": False, # T/F, whether to ignore pid listed in ${MASK_SAVED_PATH}/error.txt automatocaly
 
        # to load fake_1.25mm (from 5mm or 2.5mm) test img, do the following settings:
        # (1) let "FAST_EVAL_PKL_NAME"="fast_test_max_5.0x0.75x0.75.pkl" (or 2.5mm version)
        # (2) let "RANDOM_CROPPED_VOI_FIX_SPACING"=(1.25,0.75,0.75)
        # (3) let TRAIN["USE_5MM"]=True or TRAIN["USE_2.5MM"]=True (based on your settings)
        #  *** FAST_EVAL_PKL_NAME has no use, if TRAIN["USE_5MM"]==False and TRAIN["USE_2.5MM"]==False***
        "FAST_EVAL_PKL_NAME": "fast_test_max_2.5x0.75x0.75.pkl", # str or False, *fast_test_max_5.0x0.75x0.75.pkl
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
         "BACKBONE": "CSPDarknet", # ResNeSt | CSPDarknet | SCResNeSt
         "STRIDES":[4,8,16], # [4,8,16] for CSPDarknet; [4,8,16] for resnest # the last elements should == base_multiple
         "BASE_MULTIPLE":16, # == 2 ^ (#_stages in CSPDarknet)
         "USE_SACONV": False, ## RESNEST finished, CSPDarknet not implemented yet (i.e. no usage)
         "SACONV_USE_DEFORM": False, # True is buggy due to cpp 

         ## CSPDarknet related parameters
         "CSPDARKNET53_STEM_CHANNELS": 4, 
         "CSPDARKNET53_FEATURE_CHANNELS": [16,24,64,128] ,  #length of this should == (#_stages in CSPDarknet) # (16,24,64,128)
         "CSPDARKNET53_BLOCKS_PER_STAGE": [3,3,3] , #length of this should == (#_stages in CSPDarknet) - 1
         "VERBOSE_SHAPE": False,

         ## ResNeSt related parameters
         "RESNEST_STEM_WIDTH": 16, # similar to stem_channel, just a little bit different
         "RESNEST_EXPANSION": 2,  # orginal: 4 (higher -> less param)
         "RESNEST_FEATURE_CHANNELS": (24, 64, 128), # length == #_stages-1 == 3
         "RESNEST_BLOCKS_PER_STAGE": (2,3,3,3), # length == #_stages == 4 (default: 2,3,3,3)
         "RESNEST_STRIDE_PER_LAYER": (1, 2, 2),
         "RESNEST_EXTRA_ATTENTION": None, #"SEnetConv", #attention type:SEnet, SEnetConv, CBAM or NONE
         "RESNEST_GROUPS": 2, # default: 1
         "RESNEST_USE_CSP": True, # default: False

         ## ADDITONAL
          "YOLO_USE_LAYER0": True, # whether to add skip connection, allowing layer0 feature bypass the neck
          "MIXLAYER0NET_MODE": "attention2", # concat/attention1/*attention2/concat2
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