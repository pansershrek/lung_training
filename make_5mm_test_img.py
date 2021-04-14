from torch.utils.data import Dataset, DataLoader
import os
import pickle
from tqdm import tqdm

from dataset import Tumor, LungDataset
from global_variable import CURRENT_DATASET_PKL_PATH, NPY_SAVED_PATH
from utils_hsz import AnimationViewer
import config.yolov4_config as cfg


def main(save, num_workers=0, force_overwrite=False, strategy_5mm=cfg.VAL["5MM_STACKING_STRATEGY"]):
    assert strategy_5mm in ("max", "mean")
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.get_data(dataset.pids)
    dataset.set_5mm(True, load_5mm_pkl=False) # make sure not loading other npys
    dataset.set_lung_voi(True)
    dataset.set_batch_1_eval(True, cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    for img, bbox, pid in tqdm(dataloader, total=len(dataloader)):
        img = img.squeeze(-1).squeeze(0).numpy()
        bbox = bbox.numpy()[0][:,:6]
        pid = pid[0]
        space_text = "x".join([str(space) for space in cfg.VAL["RANDOM_CROPPED_VOI_FIX_SPACING"]])  
        name = "fast_test_{}_{}.pkl".format(strategy_5mm, space_text)
        if (0): #debug view
            print("img:", img.shape, type(img))
            print("box:", bbox)
            print("pid:", pid)
            AnimationViewer(img, bbox , note=pid, verbose=False, draw_face=False)

        folder_path = os.path.join(NPY_SAVED_PATH, pid)
        pkl_path = os.path.join(folder_path, name)
        if save:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            to_save = (img, bbox)
            with open(pkl_path, "wb") as f:
                pickle.dump(to_save, f)
        else:
            print("Fake saving to", pkl_path)

if __name__ == "__main__":
    #main(save=False, num_workers=0, strategy_5mm="mean")
    main(save=True, num_workers=6, strategy_5mm="mean")
