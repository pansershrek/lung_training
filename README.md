# 3-D YOLOv4 with Split-attention Mechanism and Self-FP Reduction for Different Resolution Lung Nodule Detection in CT Images

A 3-D implementation of YOLOv4 for lung nodule detection in CT images.

## The Brief Folder Hierarchy (Only Often Used/Modified Files are Presented)

```txt
├── README.md
├── __init__.py
├── config
│   └── yolov4_config.py      the main cfg file
├── copy_paste.py             a dirty implementation of copy-paste data augmentation
├── databuilder
│   └── yolo4dataset.py       YOLO4_Dataset that wrap the LungDataset to meet YOLO format
├── dataset.py                define the LungDataset class used in this study 
├── dicom_reader.py           handle basic I/O of DICOM files/folder  
├── draw_froc.py              the main entry script for model testing 
├── eval
│   ├── evaluator.py          define the Evaluator class used in trainer.py
│   └── froc.py               calculate CPM score + FROC performance; also generate output log
├── fast_evaluate.py          fast visulization of the predicted bbox
├── global_variable.py        an extra cfg file that mainly define where the data are stored
├── k_means_3D.py             the k-means algorithm for bbox clustering
├── lung_extraction.py        a traditional algorithm that crop the lung from a intact CT image
├── model
│   ├── backbones
│   │   ├── CSPDarknet53.py   define the Darknet backbone in YOLOv4
│   │   └── resnest.py        define the ResNeSt backbone in YOLOv4
│   ├── build_model.py        build the whole YOLOv4 model with YOLO heads
│   ├── head
│   │   └── yolo_head.py      define the YOLO head
│   └── YOLOv4.py             define the network structure of YOLOv4 without YOLO heads
├── random_crop.py            a 3-D implementation of random crop data augmentation
├── stacking_z_slices.py      image simulation of lower resolution CT images 
├── train3D.py                the main entry script for model training
├── train_5_fold.ps1          a powershell script that execute other Python scripts
├── trainer.py                define the training/testing/FP-reduction procedures 
├── utils
│   └── tools.py              useful functions for calculating CIoU and performing NMS algorithm
├── utils_ccy.py              useful functions including bbox scaling 
├── utils_hsz.py              useful functions including the 3-D slice viewer for visualization
├── view_dataset.py           view dataset statistics given a LungDataset object  
├── view_dicom_tag.py         view dicom tags of the images given a LungDataset object 
├── log                       the folder of logger output used in tensorboard
├── checkpoint                the folder storing trained model weights and validation log
└── preidiction               the folder storing testing bbox and testing log 
```

## Before running any script, please adjust global_variable.py and yolov4_config.py to fit your data location and desired settings to avoid any annoying error.
___

## Training with Validation

1. Adjust some settings in the yolov4_config.py
2. Start training by typing ...

```bash
python train3D.py --exp_name <EXPERIMENT_NAME> [--resume] [--weight_path <PRETRAINED_WEIGHT_FILE>]
```

## Testing

1. Adjust some settings in the yolov4_config.py
2. Adjust the experiment names, fold numbers and epoch numbers in the draw_froc.py
2. Start testing by typing ...

```bash
python draw_froc.py
```


