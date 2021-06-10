import torch
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import pydicom

from dataset import Tumor, LungDataset
from global_variable import MASK_SAVED_PATH, CURRENT_DATASET_PKL_PATH
from utils_hsz import AnimationViewer
import config.yolov4_config as cfg


def test_lung_voi():
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    #dataset.get_data(["25607996"])
    dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
    dataset.set_lung_voi(True)
    #err_fpath = pjoin(MASK_SAVED_PATH, "error_pid.txt")
    #with open(err_fpath, "r") as f:
    #    err_pids = f.read()[1:-1].split(",\n")
    #for datum in dataset.data.copy():
    #    _, _, pid = datum
    #    if pid in err_pids:
    #        dataset.data.remove(datum)
    max_shape = None
    max_shape_numel = 0
    for i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        img, bboxes, pid = datum
        #AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())
        numel = img.numel()
        if numel > max_shape_numel:
            max_shape = img.shape
            max_shape_numel = numel
        #if i>10:
        #    break

def test_data():
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    if (1): # may error here -> i.e. some crops have invalid bbox_coor
        dataset.set_random_crop(cfg.TRAIN["RANDOM_CROP_FILE_PREFIX"], cfg.TRAIN["RANDOM_CROP_NCOPY"], False)
    elif (0): #fine here
        dataset.set_batch_1_eval(True, (1.25, 0.75, 0.75))
        dataset.set_5mm(True, "fast_test_max_5.0x0.75x0.75.pkl")

    for i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        img, bboxes, pid = datum
        #AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())
        for bbox in bboxes:
            bbox_coor = bbox[:6]
            normal = (bbox_coor[3:] - bbox[:3])>=0
            if (not normal.all()) or (0):
                print("Invalid bbox detected in pid: {}".format(pid))
                print("bbox_coor:", bbox_coor)
                AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())

def find_data(pid:str):
    #21678302 has 4 bad crops
    #6993538 has 3 bad crops
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    """ #fine here
    for excel_r in dataset.pid_to_excel_r_relation[pid]:
        tumor = dataset.tumors[excel_r]
        print("original bbox:", tumor.voi)
        x1,x2 = tumor.voi["x"]
        y1,y2 = tumor.voi["y"]
        z1,z2 = tumor.voi["z"]
        bbox_coor = [z1,y1,x1,z2,y2,x2]
        img = tumor.get_series()
        AnimationViewer(img, [bbox_coor], note="excel_r: {}".format(excel_r))
    """
    dataset.get_data([pid])
    dataset.set_random_crop(cfg.TRAIN["RANDOM_CROP_FILE_PREFIX"], cfg.TRAIN["RANDOM_CROP_NCOPY"], False)
    for i, datum in tqdm(enumerate(dataset), total=len(dataset)):
        img, bboxes, pid = datum
        #AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())
        for bbox in bboxes:
            bbox_coor = bbox[:6]
            normal = (bbox_coor[3:] - bbox[:3])>=0
            if ((not normal.all()) or (1) and 0):
                print("Invalid bbox detected in pid: {}".format(pid))
                print("bbox_coor:", bbox_coor)
                AnimationViewer(img.squeeze(-1).numpy(), bboxes[:,:6].tolist())

def show_metadata(metadatas=("SliceThickness",), return_dic=False):
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    if type(metadatas)==str:
        metadatas = [metadatas]
    assert type(metadatas) in (list, tuple)
    out = {metadata:{} for metadata in metadatas}
    pids = set(dataset.pids) # avoid repeated pid
    for pid in tqdm(pids, desc="Reading metadata", total=len(pids)):
        tumor = dataset.tumors[dataset.pid_to_excel_r_relation[pid][0]]
        dpath = tumor.dcm_reader.path
        for s in os.listdir(dpath):
            if "." not in s:
                assert s.isnumeric()
                s0 = pydicom.read_file(pjoin(dpath, s))
                break
        for metadata in metadatas:
            if hasattr(s0, metadata):
                md = s0.__getattr__(metadata)
                if type(md) not in (str, int, float, bool, None):
                    md = md.__str__()      
                if md not in out[metadata]:
                    out[metadata][md] = [pid]
                else:
                    out[metadata][md].append(pid)
            else:
                blank = "BLANK"
                if blank not in out[metadata]:
                    out[metadata][blank] = [pid]
                else:
                    out[metadata][blank].append(pid)
                pass
                #print("pid={} has no attr '{}'".format(pid, metadata))

    if return_dic:
        return out
    
    for metadata in metadatas:
        print("="*15)
        print("For '{}'".format(metadata))
        print("-"*15)
        for md, pids in out[metadata].items():
            count = len(pids)
            print("{}: {}".format(md, count))


def ct_ldct():
    metadatas = show_metadata(["XRayTubeCurrent","ContrastBolusAgent", "ContrastBolusTotalDose"], return_dic=True)
    currents = metadatas["XRayTubeCurrent"]
    assert "BLANK" not in currents
    big_x = sum([pids for cur,pids in currents.items() if float(cur)>=100], [])
    small_x = sum([pids for cur,pids in currents.items() if float(cur)<100], [])
    print("Using XRayCurrent: CT={}, LDCT={}".format(len(big_x), len(small_x)))

    agents = metadatas["ContrastBolusAgent"]
    doses = metadatas["ContrastBolusTotalDose"]
    have_dose = sum([pids for dose, pids in doses.items() if dose!="BLANK" and float(dose)>0], [])
    no_dose = sum([pids for dose, pids in doses.items() if dose=="BLANK" or float(dose)==0], [])
    have_agent = sum([pids for agent, pids in agents.items() if agent not in ("BLANK", "None")], [])
    if (0): #debug:
        for pid in have_dose:
            for agent, pids in agents.items():
                if pid in pids:
                    print("pid={} have dose of {}".format(pid, agent))
        assert len(have_agent)==286, "len(have_agent)=={}".format(len(have_agent))
    print("Using ContrastAgent: CT={}, LDCT={}".format(len(have_dose), len(no_dose)))


if __name__ == "__main__":
    #test_lung_voi()
    #test_data()
    #find_data("21678302")
    #show_metadata(["Manufacturer", "ManufacturerModelName", "SliceThickness"])
    #show_metadata(["XRayTubeCurrent"])
    #show_metadata(["ContrastBolusAgent", "ContrastBolusTotalDose"])
    ct_ldct()