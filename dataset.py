"""
2020/10/14 預期WorkFlow

原圖 --> normalize/clip to [-1,1] --> more preprocssing? --套合-------> Crop+resize ---> visualize/training
    |--> 套VOI 與 GT -------> 延伸/內縮 VOI -----------------|

"""
import numpy as np
import os
from os.path import join as pjoin
import pickle
import pandas as pd
from tqdm import tqdm, trange
import warnings
from datetime import datetime
#from skimage.transform import resize as sk_resize
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import KFold
import random
from copy import deepcopy
from pydicom.filereader import read_dicomdir, dcmread

#from utils import linear_normalization, crop_voi, resize
import utils_ccy as utils
import utils_hsz
from dicom_reader import DicomReader
#from mri_data_searcher import DataSearcher
from global_variable import LUNG_DATA_PATH, VOI_EXCEL_PATH, EXCLUDE_KEYWORDS, NPY_SAVED_PATH, CURRENT_DATASET_PKL_PATH, MASK_SAVED_PATH
from random_crop import random_crop_preprocessing
import config.yolov4_config as cfg

#raise NotImplementedError("Work in progress")


class Tumor():
    def __init__(self, excel_r, pid, path, voi, comments, dcm_reader, original_shape=None):
        """
        For each tumor/excel_row, Tumor object contains all information needed for training

        **Property
            self.excel_r : excel_row_index of the tumor
            self.pid: accession number of the tumor
            self.path: path to the dicom data directory
            self.voi: {"x":[x1,x2], "y":[y1,y2], "z":[z1,z2]}
            self.comments: comments written in voi_excel
            self.dcm_reader: DicomReader object; it can read the volume pixel value with GDCM(?)
        """
        self.excel_r = excel_r
        self.pid = pid
        self.path = path
        self.voi = voi
        self.comments = comments
        self.dcm_reader = dcm_reader
        #img = self.get_series(False)
        #self.original_shape = img.shape
        n_slice = dcm_reader.check()
        if original_shape==None:
            self.original_shape = (n_slice, 512, 512)
        else:
            self.original_shape = original_shape
        print(pid, self.original_shape)

    def get_series(self, norm_hu=True):
        """
        See also DicomReader.get_series
        """
        #name = self.get_exact_series_name(name)
        #return self.dcm_reader.get_series(name)
        img = self.dcm_reader.get_series(norm_hu=norm_hu)
        return img

    def get_series_with_transform(self, norm_hu=True):
        """
        See also DicomReader.get_series_with_transform
        """
        #name = self.get_exact_series_name(name)
        #return self.dcm_reader.get_series_with_transform(name)
        img, transform = self.dcm_reader.get_series_with_transform(norm_hu=norm_hu)
        return img, transform

    def get_processed_series(self, npy_saved_path, name):
        npy = LungDataset.get_npy(self.pid, name)
        return npy

    @property
    def labels(self):
        """
        """
        raise NotImplementedError()
    
    
 
class LungDataset(Dataset):
    """
    An organized tumor searcher for the whole Lung Detection dataset

    **Argument
        entry:
            "default": load all valid volumes without preprocessing
            "empty": load nothing
        
    **Property
        self.summary: DataSearcher object
        self.voi_excel: pandas.dataframe.DataFrame
        self.data: a list to iter for DataLoader
        self.tumors: a dictionary, key: {excel_row_index}, value: {Tumor object}
        self.valid_rows: a list containing all valid excel rows index
        self.accession_to_rows_relation: a dictionary, key: {accession number}, value: [excel_row_index_1, ...]
    
    """
    #forbidden_data = (("t1",134), ("t1",137)) #忽略這些series
    def __init__(self, voi_excel_path=VOI_EXCEL_PATH, lung_data_path=LUNG_DATA_PATH, entry="default"):
        ## basic parameters
        self.voi_excel = pd.read_excel(voi_excel_path, sheet_name="VOI")
        self.data = []
        self.tumors = {}
        self.valid_rows = []
        self.pid_to_excel_r_relation = {}
        self.pids = []
        self.cacher = utils.LRUCache(cache_size=0)

        ## random crop related paremeters
        self.use_random_crop = False
        self.random_crop_file_prefix = ""
        self.random_crop_ncopy = 0
        self.random_choose_one = False
        self.batch_1_eval = False
        self.equal_spacing = (None,None,None)

        ## lung voi related parameters
        self.use_lung_voi = False
        self.lung_voi_lut = {}

        df = self.voi_excel
        col_top_left_x, col_top_left_y, col_top_left_z = df.columns.get_loc("top_left_x"), df.columns.get_loc("top_left_y"), df.columns.get_loc("top_left_z")
        col_bottom_right_x, col_bottom_right_y, col_bottom_right_z = df.columns.get_loc("bottom_right_x"), df.columns.get_loc("bottom_right_y"), df.columns.get_loc("bottom_right_z")
        col_box_w, col_box_h, col_box_d = df.columns.get_loc("box_width"), df.columns.get_loc("box_height"), df.columns.get_loc("box_depth") # dx, dy, dz, respectively

        col_case = df.columns.get_loc("case")
        col_path = df.columns.get_loc("path")
        col_comment = df.columns.get_loc("備註")
        col_status = col_comment-1
        if entry=="empty":
            return
        nrows, ncols = df.shape #pandas已去除首row
        for r in trange(nrows): #因為pandas的row=0已經是資料了，可直接從r=0開始
            excel_r = r+2 #第一筆資料是 r=0, excel_r=2
            valid=False
            x1 = df.iloc[r, col_top_left_x]
            x2 = df.iloc[r, col_bottom_right_x]
            width = df.iloc[r, col_box_w]
            assert x2 - x1 == width, f"Bad VOI: 'x2-x1 != width' at excel_r={r}"
            status = df.iloc[r, col_status]
            if status==1: # contains coordinate
                comments = df.iloc[r, col_comment]
                if not pd.isnull(comments): #check row validity
                    comments = comments.split("/")
                    for keyword in EXCLUDE_KEYWORDS:
                        if keyword in comments: # BAD row
                            break
                    else: # rows with coordinate and comments not containing exclusive keywords
                        valid=True
                else: # rows with coordinate and no comment
                    comments=[]
                    valid=True
            if valid:
                comments = "/".join(comments)
                y1 = df.iloc[r, col_top_left_y]
                y2 = df.iloc[r, col_bottom_right_y]
                height = df.iloc[r, col_box_h]
                z1 = df.iloc[r, col_top_left_z]
                z2 = df.iloc[r, col_bottom_right_z]
                depth = df.iloc[r, col_box_d]
                assert y2 - y1 == height, f"Bad VOI: 'y2-y1 != height' at excel_r={excel_r}"
                assert z2 - z1 == depth, f"Bad VOI: 'z2-z1 != depth' at excel_r={excel_r}"
                voi = {"x": [x1,x2], "y":[y1,y2], "z":[z1,z2]}
                ##voi = {key: value for key,value in zip("xyz", list(map( lambda i: list(map(int, i.split("-"))), [x,y,z] )))} # {"x": [x1,x2], "y": [y1,y2] ... }
                
                pid = str( df.iloc[r, col_case] )
                path = pjoin(lung_data_path, df.iloc[r, col_path])
                    
                try:
                    dcm_reader = DicomReader(pjoin(LUNG_DATA_PATH, path))
                except:
                    print(f"An error occur while opening {pid} at excel_r={excel_r}")
                    raise
                if (pid != dcm_reader.pid) and (pid.split("_")[0] != dcm_reader.pid):
                    msg = f"Inconsistent pid between voi_excel and dicom metadata in {path}\n '{pid}' != '{dcm_reader.pid}'"
                    warnings.warn(msg)

                tumor = Tumor(excel_r, pid, path, voi, comments, dcm_reader)

                assert excel_r not in self.tumors, f"Logical error: repeating excel row index {excel_r}"
                self.tumors[excel_r] = tumor
                self.valid_rows.append(excel_r)
                self.pids.append(pid)
                if pid not in self.pid_to_excel_r_relation:
                     self.pid_to_excel_r_relation[pid] = [excel_r]
                else:
                    #msg="Repeated pid '{}' detected when forming dataset".format(pid)
                    #warnings.warn(msg)
                    self.pid_to_excel_r_relation[pid].append(excel_r)

    def set_batch_1_eval(self, batch_1_eval, equal_spacing):
        assert batch_1_eval in (True, False)
        assert len(equal_spacing)==3
        self.batch_1_eval = batch_1_eval
        self.equal_spacing = equal_spacing

    def set_random_crop(self, random_crop_file_prefix, ncopy, random_choose_one=False):
        self.use_random_crop = True
        self.random_crop_file_prefix = random_crop_file_prefix
        self.random_crop_ncopy = ncopy
        self.random_choose_one = random_choose_one

    def set_lung_voi(self, use_lung_voi=True):
        assert (use_lung_voi in (True, False))
        self.use_lung_voi = use_lung_voi
        if self.use_lung_voi:
            voi_path = pjoin(MASK_SAVED_PATH, "VOI.txt")
            with open(voi_path, "r") as f:
                voi_txt = f.read()
            lut = {pid:eval(voi) for pid, voi in (line.split(" ",1) for line in voi_txt.split("\n"))}  ## had tested many times
            self.lung_voi_lut = lut



    @classmethod
    def empty(cls):
        """Make an empty MRIDataset object"""
        return cls(entry="empty")

    def __len__(self): # for DataLoader
        if self.use_random_crop:
            if self.random_choose_one:
                return len(self.data)
            return len(self.data)*self.random_crop_ncopy
        else:
            return len(self.data)

    def __getitem__(self, i): # for DataLoader
        if self.use_random_crop:
            if self.random_choose_one:
                c = random.choice(range(self.random_crop_ncopy-1))
                i = i
            else:
                c = i % self.random_crop_ncopy # index for ncopy
                i = i // self.random_crop_ncopy # index for self.data
            if (1): #pre-cropped
                _, _, pid = self.data[i] # ignore npy_name if using random crop!
                fpath = pjoin(NPY_SAVED_PATH, str(pid), "{}_c{}.pkl".format(self.random_crop_file_prefix, c+1)) # c+1 to turn c0~c4 -> c1~c5
                #print(f"i={i}, c={c}, opening {fpath} ...")
                with open(fpath, "rb") as f:
                    img, bboxes = pickle.load(f)
            else: #fresh-cropped (slow)
                _, bboxes, pid = self.data[i]
                img = self.get_series_by_pid(pid)  #已於dicom_reader.py處理過hu
                img = utils_hsz.normalize(img)
                img = torch.tensor(img, dtype=torch.float32)
                dcm_reader = self.tumors[ self.pid_to_excel_r_relation[pid][0] ].dcm_reader
                transform = [dcm_reader.SliceThickness] + list(dcm_reader.PixelSpacing[::-1])
                out = random_crop_preprocessing(img, [box[:-2] for box in bboxes], transform, cfg.TRAIN["RANDOM_CROP_SPACING"], cfg.TRAIN["TRAIN_IMG_SIZE"], n_copy=1)
                img, bboxes = out[0] # get 1 copy only
            img = torch.tensor(img).unsqueeze_(-1) # -> (Z,Y,X,1)
            bboxes = np.array(bboxes)
            assert bboxes.ndim==2
            if bboxes.shape[1] == 6 :
                n_box = bboxes.shape[0]
                bboxes = np.concatenate([bboxes, np.ones((n_box,2))], axis=-1) # zyxzyx -> zyxzyx11
            if (0): #debug
                print("pid={}, copy#={}".format(pid, c+1))
                view_img = img.squeeze(-1).numpy()
                view_box = [bbox[:6] for bbox in bboxes.tolist()]
                utils_hsz.AnimationViewer(view_img, bbox=view_box, verbose=False, note=f"{pid}_{c+1}")
            return img, bboxes, pid
        else:
            npy_name, bboxs_ori, pid = self.data[i]
            tumor = self.tumors[self.pid_to_excel_r_relation[pid][0]]
            original_shape = tumor.original_shape
            bboxes = bboxs_ori
            if npy_name==None: #using raw data
                img, exist = self.cacher.get(pid)
                if not exist: #not in cache
                    img = self.get_series_by_pid(pid)  #已於dicom_reader.py處理過hu
                    img = utils_hsz.normalize(img)
                    if self.use_lung_voi:
                        lung_voi = self.lung_voi_lut[pid]
                        z1,y1,x1,z2,y2,x2 = lung_voi
                        img = img[z1:z2+1, y1:y2+1, x1:x2+1]
                        original_shape = img.shape ## voi shape
                        new_bboxes = []
                        for bbox in bboxes:
                            oz1,oy1,ox1,oz2,oy2,ox2 = bbox[:6]
                            bbox= [oz1-z1, oy1-y1, ox1-x1, oz2-z1, oy2-y1, ox2-x1, 1, 1] # shifting O from (0,0,0) to (z1,y1,x1)
                            new_bboxes.append(bbox)
                        bboxes = new_bboxes
                    if self.batch_1_eval:
                        dcm = tumor.dcm_reader
                        transform = (dcm.SliceThickness, dcm.PixelSpacing[1], dcm.PixelSpacing[0]) #z,y,x
                        target_transform = self.equal_spacing
                        d, h, w = img.shape
                        d_new, h_new, w_new = round(d*transform[0]/target_transform[0]), round(h*transform[1]/target_transform[1]), round(w*transform[2]/target_transform[2])
                        img = utils.resize_without_pad(img, (d_new,h_new,w_new), "nearest")
                    self.cacher.set(pid, img)
            else: #using npys
                img, exist = self.cacher.get(pid)
                if not exist: #not in cache
                    img = self.get_npy(pid, npy_name)
                    self.cacher.set(pid, img)
            target_shape = tuple(img.shape)
            #print("dataset.py getitem:", target_shape)
            img = torch.FloatTensor(img).unsqueeze_(-1) # auto convert to float32
            
            bboxs_scaled = utils.scale_bbox(original_shape, target_shape, bboxes)
            bboxs_scaled = np.array(bboxs_scaled, dtype=np.int64)

            if (0): #debug
                print("pid={}, original_shape='{}', target_shape='{}', bboxs_ori='{}', bboxs_scaled='{}'".format(pid, original_shape, target_shape, bboxs_ori, bboxs_scaled))
                view_img = img.squeeze(-1).numpy()
                view_box = [bbox[:6] for bbox in bboxs_scaled.tolist()]
                utils_hsz.AnimationViewer(view_img, bbox=view_box, verbose=False, note=f"{pid}")
            return img, bboxs_scaled, pid
            
            

    def get_tumor(self, i): # i defined within [ 0,len(self.tumors) )
        excel_r = self.valid_rows[i]
        return self.tumors[excel_r]

    def save(self, saved_path):
        """
        只存 self.tumors, self.valid_rows, self.pid_to_excel_r_relation, self.pids
        """
        to_save = (self.tumors, self.valid_rows, self.pid_to_excel_r_relation, self.pids)
        with open(saved_path, "wb") as f:
            pickle.dump(to_save, f)

    @classmethod
    def load(cls, pkl_to_load):
        with open(pkl_to_load, "rb") as f:
            tumors, valid_rows, p2r, pids = pickle.load(f)
        dataset = cls.empty()
        dataset.tumors = tumors
        dataset.valid_rows = valid_rows
        dataset.pid_to_excel_r_relation = p2r
        dataset.pids = pids
        dataset.get_data(dataset.pids)
        return dataset


    @staticmethod
    def get_npy(pid, name, npy_saved_path=NPY_SAVED_PATH):
        """
        Get single np.ndarray from npy file, given *pid* and name
        """
        name=name.lower()
        if not name.endswith(".npy"):
            name = name + ".npy"
        target = pjoin(npy_saved_path, str(pid), name)
        assert os.path.exists(target), f"{target} not existed!"
        with open(target, "rb") as f:
            npy = np.load(f, allow_pickle=True)
        return npy

    
    def get_npys_by_name(self, name, pids, npy_saved_path=NPY_SAVED_PATH):
        """
        Load and return all npys containing the given series

        **Note:
            This method can be very memory-consuming!
        """
        for pid in pids:
            assert pid in self.pids, f"PID '{pid}' is not present in dataset, but is included in pids argument"
        candidates = pids
        npys = []
        for pid in candidates:
            try:
                npy = self.get_npy(pid, name, npy_saved_path=npy_saved_path)
            except FileNotFoundError:
                print(f"Series {name}.npy not existed for pid={pid}, try making npy...")
                raise NotImplementedError()
            npys.append(npy)
        return npys
    
    def get_series_by_pid(self, pid):
        return self.tumors[self.pid_to_excel_r_relation[pid][0]].get_series()

    def get_data(self, pids, name=None): # this function will loads npy by itself
        #pids = set(pids) # Bug: order random!
        tmp = []
        for pid in pids:
            if pid not in tmp:
                tmp.append(pid)
        pids = tmp

        if name!=None:
            ## Note that this is memory-consuming
            ## npys = self.get_npys_by_name(name, pids=pids) # a list of np.ndarray (same shape, e.g. (50,50,50))
            x = [name]*len(pids)
        else:
            # load raw images
            #x =  [ img for pid in pids for img,_ in dataset.tumors[dataset.pid_to_excel_r_relation[pid][0]].get_series_with_transform() ]
            x = [None]*len(pids)
            #x = [self.get_series_by_pid(pid) for pid in pids]

        y = []
        for pid in pids:
            excel_rs = self.pid_to_excel_r_relation[pid]
            bboxes = []
            for excel_r in excel_rs:
                tumor = self.tumors[excel_r]
                voi = tumor.voi
                x1,x2 = voi["x"]
                y1,y2 = voi["y"]
                z1,z2 = voi["z"]
                z1,z2 = z1-1, z2-1 # starts from 1 -> 0
                bbox = [z1,y1,x1,z2,y2,x2,1,1] #(zyxzyx, class, mixup)
                bboxes.append(bbox)
            y.append(bboxes)
        
        #y = self.__creat_label(y, img_size)
        assert len(x)==len(y)==len(pids)
        self.data = list(zip(x,y,pids))
        
    
    #def collate_fn(self, samples):
    #    #print("TYPE:", type(samples), ", LEN:", len(samples))
    #    #print("samples[0]", type(samples[0]), len(samples[0]))
    #    batch_x = torch.stack([x for x,_ in samples])
    #    batch_y = torch.stack([y for _,y in samples])
    #    return batch_x, batch_y

    def make_kfolds_using_pids(self, num_k_folds, k_folds_seed, current_fold, valid_test_split=True, portion_list=None):
        if portion_list!=None:
            if (not all([type(i)==int for i in portion_list])) or (sum(portion_list)!=num_k_folds) :
                raise ValueError(f"portion list should contain integers only, and its sum should be equal to num_k_folds={num_k_folds}, not {portion_list}")
            if len(portion_list)!=3 and valid_test_split: raise TypeError("For valid_test_split, portion_list should have length 3")
            if len(portion_list)!=2 and (not valid_test_split): raise TypeError("For no valid set settings, portion_list should have length 2")
        else:
            portion_list = [num_k_folds-2, 1, 1] if valid_test_split else [num_k_folds-1, 1]
            if not all([i>0 for i in portion_list]): raise ValueError("num_k_folds should be at least 3 for valid_test_split, and at least 2 otherwise")
        
        all_folds=[[] for i in range(num_k_folds)] # store all K-fold pids
        def get_folds_with_shortest_length(all_folds):
            """Return a list of index, indicating flods with least length currently"""
            idxs = []
            min_length = float("inf")
            for i, fold in enumerate(all_folds):
                length = len(fold)
                if length < min_length:
                    min_length = length
                    idxs = [i]
                elif length == min_length:
                    idxs.append(i)
            idxs.sort()
            return idxs
        random.seed(k_folds_seed) #initiate random seed
        for datum in self.data:
            folds_to_choose = get_folds_with_shortest_length(all_folds)
            fold_index = random.choice(folds_to_choose)
            all_folds[fold_index].append(datum)

        merge_folds=[] # let it be the order: [train, valid, test]
        if current_fold not in range(num_k_folds): raise TypeError(f"invalid current fold: {current_fold}, for num_k_folds: {num_k_folds}")
        i_fold = current_fold # shifting index by current_fold!! (i.e. how k-folds work)
        for portion in portion_list:
            rows=[]
            for _ in range(portion):
                rows = rows + all_folds[i_fold]
                i_fold = (i_fold+1) % num_k_folds # 0~4
            merge_folds.append(rows)

        if len(portion_list)==3:
            train_data, validation_data, test_data = merge_folds[0], merge_folds[1], merge_folds[2]
            self.train_data, self.validation_data, self.test_data = train_data, validation_data, test_data
            
        else:
            train_data, test_data = merge_folds[0], merge_folds[1]
            self.train_data, self.test_data = train_data, test_data   
        return merge_folds

    def make_froc_annotation_file(self, file_save_path="annotation_chung.txt"):
        out_text = ""
        for pid in self.pids:
            line_txt = f"{pid},"
            excel_rs = self.pid_to_excel_r_relation[pid]
            for excel_r in excel_rs:
                tumor = self.tumors[excel_r]
                bbox_txt = ""
                x1,x2 = tumor.voi["x"]
                y1,y2 = tumor.voi["y"]
                z1,z2 = tumor.voi["z"]
                z1,z2 = z1-1, z2-1
                original_shape = tumor.get_series().shape
                ori_z, ori_y, ori_x = original_shape
                bbox_txt = f"{ori_z},{ori_y},{ori_x},{x1},{y1},{z1},{x2},{y2},{z2},0"
                line_txt = line_txt + bbox_txt + " "
            line_txt = line_txt[:-1] # delete trailing space
            out_text = out_text + line_txt + "\n"
        out_text = out_text[:-1]
        with open("annotation_chung.txt", "w") as f:
            f.write(out_text)
        




    def make_balanced_kfolds_and_prepare_tensors(self, num_k_folds, k_folds_seed, current_fold, series_names=["sub"], npy_names=[""], valid_test_split=False, portion_list=None, data_augmentation=True, debug=False):
        """
        """
        raise NotImplementedError("Deprecated, for MRI only")
        if portion_list!=None:
            if (not all([type(i)==int for i in portion_list])) or (sum(portion_list)!=num_k_folds) :
                raise ValueError(f"portion list should contain integers only, and its sum should be equal to num_k_folds={num_k_folds}, not {portion_list}")
            if len(portion_list)!=3 and valid_test_split: raise TypeError("For valid_test_split, portion_list should have length 3")
            if len(portion_list)!=2 and (not valid_test_split): raise TypeError("For no valid set settings, portion_list should have length 2")
        else:
            portion_list = [num_k_folds-2, 1, 1] if valid_test_split else [num_k_folds-1, 1]
            if not all([i>0 for i in portion_list]): raise ValueError("num_k_folds should be at least 3 for valid_test_split, and at least 2 otherwise")
        
        all_folds=[[] for i in range(num_k_folds)] # store all K-fold index
        
        current_valid_rows = set(self.valid_rows) # all rows
        for series_name in series_names:
            rows_with_name = self.get_rows_with_series(series_name)
            #print(series_name, rows_with_name)
            current_valid_rows = current_valid_rows.intersection(rows_with_name)
        current_valid_rows = list(current_valid_rows)
        assert all([type(i)==int for i in current_valid_rows]), f"Logical Error: not-interger excel_r detected in current_valid_rows: {current_valid_rows}"

        ## filtered out tumors that are too big
        def filter_tumors(excel_rs, long_limit=40):
            """long_limit is the longest mm you can tolerate"""
            warnings.warn(f"Start filtering tumors with any length > {long_limit}(mm)")
            out = []
            for excel_r in excel_rs:
                tumor = self.tumors[excel_r]
                voi = tumor.voi
                sub_dcm = tumor.get_dcm("sub")
                xp, yp = sub_dcm.children[0].PixelSpacing
                xp, yp = xp.real, yp.real
                zp = sub_dcm.children[0].SliceThickness.real
                x_len = (tumor.voi["x"][1]-tumor.voi["x"][0]+1)*xp
                y_len = (tumor.voi["y"][1]-tumor.voi["y"][0]+1)*yp
                z_len = (tumor.voi["z"][1]-tumor.voi["z"][0]+1)*zp
                assert x_len>0 and y_len>0 and z_len>0
                print(f"excel_r={excel_r}: x,y,z={round(x_len,2)}, {round(y_len,2)}, {round(z_len,2)}")
                max_len = max([x_len, y_len, z_len])
                #print(f"max_len={max_len}, max_len<={long_limit}: {max_len<=long_limit}")
                if max_len<=long_limit:
                    out.append(excel_r)
                del tumor._dcm
            return out

        #print("before filter:", len(current_valid_rows), current_valid_rows)
        ##current_valid_rows = filter_tumors(current_valid_rows)
        #print("after filter:", len(current_valid_rows), current_valid_rows)


        current_valid_rows.sort()
        #print("current_valid_rows", current_valid_rows)
        voi_statistics = {} # keys: (er,pr,her2); values: [count, [excel_r1, excel_r2, ...]]
        ER, PR, HER2 = [0,0], [0,0], [0,0]
        mapping = {"+":0, "-":1}
        for excel_r in current_valid_rows: # all rows containing series "name"
            tumor = self.tumors[excel_r]
            er, pr, her2 = tumor.er, tumor.pr, tumor.her2
            ER[mapping[er]] += 1
            PR[mapping[pr]] += 1
            HER2[mapping[her2]] += 1
            status = (er,pr,her2)
            if status not in voi_statistics:
                voi_statistics[status] = [1, [excel_r]]
            else:
                voi_statistics[status][0] += 1
                voi_statistics[status][1].append(excel_r)
        
        recommended_pos_weight = (ER[1]/ER[0], PR[1]/PR[0], HER2[1]/HER2[0]) # pos_weight is the weight act on positive samples, it should be (# negative)/(# positive)
        
        def get_folds_with_shortest_length(all_folds):
            """Return a list of index, indicating flods with least length currently"""
            idxs = []
            min_length = float("inf")
            for i, fold in enumerate(all_folds):
                length = len(fold)
                if length < min_length:
                    min_length = length
                    idxs = [i]
                elif length == min_length:
                    idxs.append(i)
            idxs.sort()
            return idxs

        random.seed(k_folds_seed) #initiate random seed
        for status, (count, rows) in voi_statistics.items():
            assert count==len(rows)
            for i, excel_r in enumerate(rows):
                folds_to_choose = get_folds_with_shortest_length(all_folds)
                fold_index = random.choice(folds_to_choose)
                all_folds[fold_index].append(excel_r)
            
        merge_folds=[] # for algorithm simplicity, let it be the order: [train, test, valid]
        if current_fold not in range(num_k_folds): raise TypeError(f"invalid current fold: {current_fold}, for num_k_folds: {num_k_folds}")
        i_fold = current_fold # shifting index by current_fold!! (i.e. how k-folds work)
        for portion in portion_list:
            rows=[]
            for _ in range(portion):
                rows = rows + all_folds[i_fold]
                i_fold = (i_fold+1) % num_k_folds # 0~4
            merge_folds.append(rows)
        
        train_rows, test_rows = merge_folds[0], merge_folds[1]

        
        if len(portion_list)==3:
            validation_rows = merge_folds[2]
            validation_x, validation_y = [], []
            xs = []
            for series_name, npy_name in zip(series_names, npy_names):
                if npy_name.endswith(".npy"): npy_name = npy_name[:-4]
                validation_x_raw, validation_y = self.get_tensors(validation_rows, name=npy_name)
                xs.append(validation_x_raw)
            validation_x = self.cat_or_stack_tensor_lists(xs, 0)
            self.validation_data = list(zip(validation_x, validation_y, validation_rows)) #don't confuse it with self.valid_rows

        #print("train_rows", train_rows) # check different seed/current_fold ok!
        #print("test_rows", test_rows)
        #print("validation_rows", validation_rows)

        xs = []
        for series_name, npy_name in zip(series_names, npy_names):
            if npy_name.endswith(".npy"): npy_name = npy_name[:-4]
            train_x_raw, train_y = self.get_tensors(train_rows, name=npy_name)
            xs.append(train_x_raw)
        train_x = self.cat_or_stack_tensor_lists(xs, 0)

        xs = []
        for series_name, npy_name in zip(series_names, npy_names):
            if npy_name.endswith(".npy"): npy_name = npy_name[:-4]
            test_x_raw, test_y = self.get_tensors(test_rows, name=npy_name)
            xs.append(test_x_raw)
        test_x = self.cat_or_stack_tensor_lists(xs, 0)     

        debug_train_x, debug_train_y = train_x[:40], train_y[:40]
        debug_test_x, debug_test_y = test_x[:10], test_y[:10]

        if data_augmentation:
            extra_x, extra_y, extra_rows = [], [], []
            for x, y, excel_r in zip(train_x, train_y, train_rows):
                #print("x_shape", x.shape) # shape (C,Z,Y,X)
                #print("y_shape", y.shape) # shape (out_C,)
                label = tuple(y.tolist())
                label2 = self.tumors[excel_r].labels
                assert label == label2
                #if label == (0,0,1): # Her2-enriched
                if True:
                    new_x = x.flip(-1) # (C,Z,Y,X)
                    extra_x.append(new_x)
                    extra_y.append(y)
                    extra_rows.append(excel_r)
                    new_x = x.flip(-2) # (C,Z,Y,X)
                    extra_x.append(new_x)
                    extra_y.append(y)
                    extra_rows.append(excel_r)
                    new_x = x.flip([-1, -2]) # (C,Z,Y,X)
                    extra_x.append(new_x)
                    extra_y.append(y)
                    extra_rows.append(excel_r)
            train_x = train_x + extra_x
            train_y = train_y + extra_y
            train_rows = train_rows + extra_rows
            # update pos_weight
            n = len(extra_y)
            #recommended_pos_weight = (  (ER[1]+n)/ER[0], (PR[1]+n)/PR[0], HER2[1]/(HER2[0]+n)  ) #p.s. 0 -> pos, 1 -> neg, pos_weight = (#neg)/(#pos)
            
        self.train_data = list(zip(train_x, train_y, train_rows))
        self.test_data = list(zip(test_x, test_y, test_rows))
        self.debug_train_data = list(zip(debug_train_x, debug_train_y, train_rows[:40]))
        self.debug_test_data = list(zip(debug_test_x, debug_test_y, test_rows[:10]))

        def debug_check(train_rows, test_rows, validation_rows=None): #check balancing of each fold/data
            mapping = {"+":0, "-":1}
            printed = ["Training", "Testing", "Validation"]
            total = len(train_rows) + len(test_rows)
            if validation_rows!=None:
                total += len(validation_rows)
            ERs, PRs, HER2s = [0,0], [0,0], [0,0]
            total_rows_statistics = {}

            for i, excel_rs in enumerate([train_rows, test_rows, validation_rows]):
                if excel_rs==None: continue
                rows_statistics = {}
                ER, PR, HER2 = [0,0], [0,0], [0,0]
                fold_total = 0
                for excel_r in excel_rs:
                    tumor = self.tumors[excel_r]
                    er, pr, her2 = tumor.er, tumor.pr, tumor.her2
                    status = (er,pr,her2)
                    ER[mapping[er]]+=1
                    PR[mapping[pr]]+=1
                    HER2[mapping[her2]]+=1
                    fold_total += 1
                    if status not in rows_statistics:
                        rows_statistics[status] = 1
                    else:
                        rows_statistics[status] += 1
                ERs[0] += ER[0]
                ERs[1] += ER[1]
                PRs[0] += PR[0]
                PRs[1] += PR[1]
                HER2s[0] += HER2[0]
                HER2s[1] += HER2[1]
                print("\n***"+printed[i]+"***")
                print("tumor count in the set: {} ({:.2f}%)".format(fold_total, fold_total/total*100))
                print("{:>6} positive negative pos_ratio".format(""))
                print("ER: {:>8} {:>8} {:>8.2f}".format(*ER, ER[0]/fold_total))
                print("PR: {:>8} {:>8} {:>8.2f}".format(*PR, PR[0]/fold_total))
                print("HER2: {:>6} {:>8} {:>8.2f}".format(*HER2, HER2[0]/fold_total))
                print()
                print("**Detailed distribution: (ER,PR,HER2)**")
                print("{:>3}{:>5}{:>6}".format("ER","PR","HER2"))
                for typ, count in rows_statistics.items():
                    if typ not in total_rows_statistics:
                        total_rows_statistics[typ] = count
                    else:
                        total_rows_statistics[typ] += count
                    print("{}: {}  ({:.2f}%)".format(typ, count, count/fold_total*100))
            print()
            print("***Total***")
            print("All tumors count: {}".format(total))
            print("{:>6} positive negative pos_ratio".format(""))
            print("ER: {:>8} {:>8} {:>8.2f}".format(*ERs, ERs[0]/total))
            print("PR: {:>8} {:>8} {:>8.2f}".format(*PRs, PRs[0]/total))
            print("HER2: {:>6} {:>8} {:>8.2f}".format(*HER2s, HER2s[0]/total))
            print()
            print("**Detailed distribution: (ER,PR,HER2)**")
            print("{:>3}{:>5}{:>6}".format("ER","PR","HER2"))
            for typ, count in total_rows_statistics.items():
                print("{}: {}  ({:.2f}%)".format(typ, count, count/total*100))
            print("train_rows", sorted(train_rows))
            print()
            print("test_rows", sorted(test_rows))
            if len(portion_list)==3:
                print()
                print("validation_rows", sorted(validation_rows))
            print("x shape", self.train_data[0][0].shape)
            print("y shape", self.test_data[0][1].shape)
    
        if debug:
            print("Recommended pos weight", recommended_pos_weight)
            debug_check(train_rows, test_rows, validation_rows) if len(portion_list)==3 else debug_check(train_rows, test_rows)
            raise EOFError("End of debug report")

        return recommended_pos_weight
        
                



    


def dataset_preprocessing(dataset, npy_saved_path, model_input_shape, output_name="debug.npy", pad_mode="center", pad_cval=0, resize_before_pad=True, overwrite=False, debug=False):
    """
    Make all npy files and save them into npy_saved_path
    """
    debug_start = 100
    debug_end = 103
    debug_i = 0
    
    if npy_saved_path is not None:
        os.makedirs(npy_saved_path, exist_ok=True)
    
    if not output_name.endswith(".npy"):
        raise TypeError(f"Invalid npy name '{output_name}'")
    
    tmp=[]
    for pid in dataset.pids:
        if pid not in tmp:
            tmp.append(pid)
    pids = tmp
    #pids = set(dataset.pids)
    for pid in tqdm(pids, desc="Preprocessing:", total=len(pids)):
        if debug:
            if debug_i < debug_start:
                debug_i += 1
                continue
            if debug_i > debug_end:
                print("break", debug_i)
                break
            print("Debugging:", debug_i, "pid =", pid)
            debug_i += 1


        ##Check existed npy and try loading it
        target = pjoin(npy_saved_path, str(pid), output_name)
        if os.path.exists(target) and (not overwrite):
            with open(target, "rb") as f:
                arr = np.load(f, allow_pickle=True)
                assert model_input_shape==arr.shape or model_input_shape==None, f"Tumor on row {tumor.excel_r} has invalid shape {sub.shape}, while desired model input shape is {model_input_shape}"
        else: # else, do preprocessing
            tumor = dataset.tumors[dataset.pid_to_excel_r_relation[pid][0]]
            img, transform = tumor.get_series_with_transform() #shape: z,y,x
            transform = np.array(transform)
            
            ori_shape = img.shape

            #print(f"original shape: {sub.shape}, {sub.dtype}")
   
            #norm_img = utils.linear_normalization(img, newMin=0, newMax=1, dtype=np.float64) # normalization
            #norm_img = norm_img - norm_img.mean() # zero-centering
            ##USE hsz
            norm_img = utils_hsz.normalize(img)
            out_img = norm_img
            """
            #May not need this
            equal_spacing_img, new_spacing = utils_hsz.resample( norm_img, transform, new_spacing=[1,1,1] )
            
            #resized = utils.resize(equal_spacing_cropped_voi, model_input_shape) #deprecated
            """
            equal_spacing_img = norm_img # if not using hsz_resize
            if model_input_shape!=None:
                if pad_cval=="min":
                    min_val = equal_spacing_img.min()
                    #resized_img = utils.resize_and_pad(equal_spacing_img, model_input_shape, mode=pad_mode, cval=min_val, resize_before_pad=resize_before_pad)
                    resized_img = utils.resize_without_pad(equal_spacing_img, model_input_shape, device="cuda")
                else:
                    #resized_img = utils.resize_and_pad(equal_spacing_img, model_input_shape, mode=pad_mode, cval=pad_cval, resize_before_pad=resize_before_pad)
                    resized_img = utils.resize_without_pad(equal_spacing_img, model_input_shape, device="cuda")
                out_img = resized_img
            out_img = out_img.astype(np.float32) # double -> single to reduce disk usage
            #Debug
            if debug:
                print("VOI:", tumor.voi)
                (x1,x2), (y1,y2), (z1,z2) = tumor.voi["x"], tumor.voi["y"], tumor.voi["z"]
                z1,z2 = z1-1, z2-1
                voi = [z1,y1,x1,z2,y2,x2] # zyxzyx format
                print("Converted hu")
                utils_hsz.AnimationViewer(img, voi)
                print("Normalized")
                utils_hsz.AnimationViewer(norm_img, voi)
                #print("try crop voi")
                #cropped = utils.crop_voi(norm_img, voi, [0,0,0])
                #utils_hsz.AnimationViewer(cropped)
                if model_input_shape!=None:
                    print("Resized:")
                    utils_hsz.AnimationViewer(resized_img)
                print("output_image shape:", out_img.shape)
                """
                print("after hsz resample", equal_spacing_img.shape)
                utils_hsz.AnimationViewer(equal_spacing_img)
                print("after resize", resized_img.shape)
                utils_hsz.AnimationViewer(resized_img) #target series, the npy to save
                """
                
                continue

    
            if npy_saved_path is not None:
                subf = pjoin(npy_saved_path, str(pid)) #存放在名為excel_r的子資料夾
                os.makedirs(subf, exist_ok=True)
                saved_name = pjoin(subf, output_name)
                if overwrite or (not os.path.exists(saved_name)):
                    print(f"save {saved_name}")
                    np.save(saved_name, out_img)
                    #with open(saved_name, "wb") as f:
                    #    out_img.dump(f)
            #sub = resized_img
            

        
             
        
def single_volume_preprocessing(to_process_excel_r, overwrite_npy, force_reconstruct, model_input_shape, output_name, voi_extend=[5,5,5], npy_saved_path=NPY_SAVED_PATH, pad_mode="center", pad_cval=0, resize_before_pad=True, to_crop_series="sub", debug=False):
    """
    Preprocess, crop and resize the given volume to model_input_shape, based on its sub-series shape and voi
    """
    raise NotImplementedError()
    def get_date_str(date=datetime.today()):
        return "{}{:02}{:02}".format(date.year, date.month, date.day)
        #today_str = get_date_str()
    today_str = "20201118"
    try:
        if force_reconstruct:
            raise TypeError
        print("Load dataset...")
        dataset = MRIDataset.load("mri_dataset_{}.pkl".format(today_str))
    except:
        print("Load dataset failed! Reconstructing dataset...")
        dataset = construct_dataset()
    print("Start preprocessing")
    tumor = dataset.tumors[to_process_excel_r]
    sub, transform = tumor.get_series_with_transform("sub") #shape: z,y,x
    to_crop, transform2 = tumor.get_series_with_transform(to_crop_series)
    sub_shape = sub.shape
    to_crop_shape = to_crop.shape
    target_voi = utils.registrate_voi_using_dcm(tumor.voi, to_crop_series, tumor, sub_shape, voi_extend)
    # try adjust z-thickness!
    transform2[0] = (tumor.voi["z"][1] - tumor.voi["z"][0] + 1 + voi_extend[0]) / (target_voi["z"][1] - target_voi["z"][0] + 1 + voi_extend[0])
    to_crop = utils.linear_normalization(to_crop, newMin=0, newMax=1, dtype=np.float64) # normalization
    to_crop = to_crop - to_crop.mean() # zero-centering
    try:
        cropped_voi = utils.crop_voi(to_crop, target_voi, extend=voi_extend) # cropped
    except:
        print(f"An error occur while '{to_crop_series}' series preprocessing; tumor on excel_r={tumor.excel_r}")
        raise
    if debug:
        print(f"after crop: {cropped_voi.shape}, {cropped_voi.dtype}") 
        #AnimationViewer(cropped_voi)

    equal_spacing_cropped_voi, new_spacing = utils_hsz.resample( cropped_voi, transform2, new_spacing=[1,1,1] )
    if debug:
        print(f"after hsz_resample: {equal_spacing_cropped_voi.shape}")
        #AnimationViewer(equal_spacing_cropped_voi)

    #resized = utils.resize(equal_spacing_cropped_voi, model_input_shape) #deprecated
    if pad_cval=="min":
        min_val = equal_spacing_cropped_voi.min()
        resized = utils.resize_and_pad(equal_spacing_cropped_voi, model_input_shape, mode=pad_mode, cval=min_val, resize_before_pad=resize_before_pad)
    else:
        resized = utils.resize_and_pad(equal_spacing_cropped_voi, model_input_shape, mode=pad_mode, cval=pad_cval, resize_before_pad=resize_before_pad)
    if debug:
        print(f"after pad: {resized.shape}, {resized.dtype}") 
        AnimationViewer(resized)
        arr = utils.linear_normalization(sub, newMin=0, newMax=1, dtype=np.float64)
        arr = arr - arr.mean()
        arr = utils.crop_voi(arr, tumor.voi, extend=voi_extend)
        print("sub after crop", arr.shape)
        arr, _ = utils_hsz.resample( arr, transform, new_spacing=[1,1,1] )
        print("sub after hsz resample", arr.shape)
        arr = utils.resize_and_pad(arr, model_input_shape, mode=pad_mode, cval=pad_cval, resize_before_pad=resize_before_pad)
        print("sub after resize and pad", arr.shape)
        utils_hsz.AnimationViewer(arr) #sub series, reference usage
        return 

    if npy_saved_path is not None:
        subf = pjoin(npy_saved_path, str(tumor.excel_r)) #存放在名為excel_r的子資料夾
        os.makedirs(subf, exist_ok=True)
        saved_name = pjoin(subf, output_name)
        if overwrite_npy or (not os.path.exists(saved_name)):
            with open(saved_name, "wb") as f:
                resized.dump(f)
    

           
        



###################################
def _test():
    single_volume_preprocessing(to_process_excel_r=281,
                                overwrite_npy=True, 
                                force_reconstruct=False, 
                                model_input_shape=(50,50,50), 
                                voi_extend=[5,5,5], 
                                to_crop_series="sub",
                                output_name="sub.npy", 
                                pad_mode="center", 
                                pad_cval=0, 
                                debug=False)

def _test_kfold(data_augmentation):
    dataset = MRIDataset.load("mri_dataset_20201118.pkl")
    #series_name = ["sub", "tirm_tra_p3"]
    #npy_names = ["sub_center.npy", "tirm_tra_p3_center.npy"]
    series_name = ["sub", "t1", "t2"]
    npy_names = ["sub.npy", "t1.npy", "t2.npy"]
    debug = True
    portion_list = [3,1,1]
    dataset.make_balanced_kfolds_and_prepare_tensors(num_k_folds=5, k_folds_seed=123, current_fold=0, series_names=series_name, npy_names=npy_names, valid_test_split=True, portion_list=portion_list, data_augmentation=data_augmentation, debug=debug)

    

def construct_dataset():
    dataset = LungDataset()
    def get_date_str(date=datetime.today()):
        return "{}{:02}{:02}".format(date.year, date.month, date.day)
    today_str = get_date_str()
    dataset.save("lung_dataset_{}.pkl".format(today_str))
    return dataset
    
def preprocessed_all(overwrite_npy, force_reconstruct, model_input_shape, output_name, pad_mode="center", pad_cval=0, resize_before_pad=True, debug=False):
    def get_date_str(date=datetime.today()):
        return "{}{:02}{:02}".format(date.year, date.month, date.day)
    #today_str = get_date_str()
    today_str = "20210118"
    try:
        if force_reconstruct:
            raise TypeError
        print("Load dataset...")
        DATA = LungDataset.load("lung_dataset_{}.pkl".format(today_str))
    except:
        print("Load dataset failed! Reconstructing dataset...")
        DATA = construct_dataset()
    print("Start preprocessing")
    dataset_preprocessing(DATA, NPY_SAVED_PATH, model_input_shape, overwrite=overwrite_npy, output_name=output_name, pad_mode=pad_mode, pad_cval=pad_cval, debug=debug, resize_before_pad=resize_before_pad)
    
    
if __name__ == "__main__":
    #_test()
    construct_dataset()
    raise EOFError
    #dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    preprocessed_all(overwrite_npy=True, 
                    force_reconstruct=False, 
                    model_input_shape=(256,256,256),
                    output_name="hu+norm_256x256x256.npy",  
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=False,
                    debug=False)

    """preprocessed_all(overwrite_npy=True, 
                    force_reconstruct=False, 
                    model_input_shape=(160,160,160), 
                    voi_extend=[5,5,5],
                    output_name="t1_no_resize.npy", 
                    to_crop_series="t1", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=False,
                    debug=False)

    preprocessed_all(overwrite_npy=True, 
                    force_reconstruct=False, 
                    model_input_shape=(160,160,160), 
                    voi_extend=[5,5,5],
                    output_name="t2_no_resize.npy", 
                    to_crop_series="t2", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=False,
                    debug=False)

    preprocessed_all(overwrite_npy=True, 
                    force_reconstruct=False, 
                    model_input_shape=(160,160,160), 
                    voi_extend=[5,5,5],
                    output_name="sub_no_resize.npy", 
                    to_crop_series="sub", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=False,
                    debug=False)"""
    
    
    """preprocessed_all(overwrite_npy=False, 
                    force_reconstruct=False, 
                    model_input_shape=(50,50,50), 
                    voi_extend=[5,5,5],
                    output_name="sub_no_pad.npy", 
                    to_crop_series="sub", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=True,
                    debug=False)

    preprocessed_all(overwrite_npy=False, 
                    force_reconstruct=False, 
                    model_input_shape=(50,50,50), 
                    voi_extend=[5,5,5],
                    output_name="t1_no_pad.npy", 
                    to_crop_series="t1", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=True,
                    debug=False)
    
    preprocessed_all(overwrite_npy=False, 
                    force_reconstruct=False, 
                    model_input_shape=(50,50,50), 
                    voi_extend=[5,5,5],
                    output_name="t2_no_pad.npy", 
                    to_crop_series="t2", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=True,
                    debug=False)"""
    
    #_test_kfold(False)
    raise EOFError("End of script")
    """preprocessed_all(overwrite_npy=True, 
                    force_reconstruct=False, 
                    model_input_shape=(100,160,160), 
                    output_name="sub_center_no_resize_100x160x160.npy", 
                    to_crop_series="sub", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=False,
                    debug=False)
    
    preprocessed_all(overwrite_npy=True, 
                    force_reconstruct=False, 
                    model_input_shape=(100,160,160), 
                    output_name="tirm_tra_p3_center_no_resize_100x160x160.npy", 
                    to_crop_series="tirm_tra_p3", 
                    pad_mode="center", 
                    pad_cval = 0,
                    resize_before_pad=False,
                    debug=False)"""


