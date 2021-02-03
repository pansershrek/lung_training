

LUNG_DATA_PATH = r"E:/LungData/1500_DICOM" #r"D:/CH/LungDetection/DATA/1500_DICOM" 
VOI_EXCEL_PATH = r"D:/CH/LungDetection/AI-肺部手術病人資料_trimmed.xlsx" #記錄VOI的那個Excel
EXCLUDE_KEYWORDS = () #若VOI_EXCEL的備註欄裡面有任何這些keyword, 將略過該筆資料
NPY_SAVED_PATH = r"E:/LungData/Preprocessed"
MASK_SAVED_PATH = r"E:/LungData/AutoMask"
LUNG_DATASET_PKL_PATH = r"D:/CH/LungDetection/training/lung_dataset_20210118.pkl"

#REPEATABLE_SERIES = () #MRI資料中，可允許重複出現的series種類

USE_LUNA = False  ##記得要同時改Anchorbox3D
LUNA_DIR = r"E:/Luna16/RAW"
LUNA_DATASET_PKL_PATH = r"D:/CH/LungDetection/training/luna_test_dataset.pkl"


CURRENT_DATASET_PKL_PATH = LUNA_DATASET_PKL_PATH if USE_LUNA else LUNG_DATASET_PKL_PATH

