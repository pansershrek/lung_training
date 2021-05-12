import numpy as np
import math
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import argparse
import logging
from tqdm import tqdm
import warnings

from utils_hsz import AnimationViewer
from utils_ccy import iou_3D
from utils.log import Logger
from dataset import Tumor, LungDataset
from global_variable import CURRENT_DATASET_PKL_PATH
from trainer import Trainer

class FPViewer(AnimationViewer):
    def __init__(self, *args, debug=False):
        self.debug = debug
        self.isfp=1 # assert it is a FP by default
        super(FPViewer, self).__init__(*args)
    
    def fp_button_click_event(self, event): # change FP -> not a FP
        self.isfp=0
        self.update_fp_status()

    def reset_button_click_event(self, event):
        self.isfp=1
        self.update_fp_status()
    
    def bad_fp_button_click_event(self, event):
        self.isfp=2
        self.update_fp_status()

    def update_fp_status(self):
        if self.isfp==1:
            self.text_label.set_text("FP")
            self.text_label.set_position((30,200))
            self.text_label.set_color("black")
        elif self.isfp==0: # i.e. TP
            self.text_label.set_text("Not a FP")
            self.text_label.set_position((-20,200))
            self.text_label.set_color("r")
        else: # isfp==2
            self.text_label.set_text("Bad FP")
            self.text_label.set_position((-10,200))
            self.text_label.set_color("grey")

    def __call__(self):
        return self.isfp

    def multi_slice_viewer(self):
        fig, ax = plt.subplots()
        ax.volume = self.image
        ax.index = self.image.shape[0] // 2 if len(self.bbox)==0 else int( (self.bbox[0][0]+self.bbox[0][3])/2 )
        ax.imshow(self.image[ax.index], cmap="gray", vmin=self.pixel_min, vmax=self.pixel_max)
        self.add_bbox(ax)
        ax.set_title(f"{self.note}shape = {self.volume_shape}, slice_idx= {ax.index}")
        fig.canvas.mpl_connect("key_press_event", self.process_key)
        fig.canvas.mpl_connect("scroll_event", self.process_key)
        y, x = self.image.shape[1:]

        #Additional button part
        assert self.debug or len(self.bbox)==1 # exact 1 bbox
        ax_fpbut = plt.axes([0.85, 0.55, 0.1, 0.075])
        fp_button = Button(ax_fpbut, 'Not a FP')
        fp_button.on_clicked(self.fp_button_click_event)

        ax_resetbut = plt.axes([0.85, 0.35, 0.1, 0.075])
        reset_button = Button(ax_resetbut, 'Reset')
        reset_button.on_clicked(self.reset_button_click_event)

        ax_badfpbut = plt.axes([0.85, 0.25, 0.1, 0.075])
        badfp_button = Button(ax_badfpbut, 'Bad FP')
        badfp_button.on_clicked(self.bad_fp_button_click_event)

        self.text_label = plt.text(30, 200, "FP", c="black", fontsize=15)

        plt.xlim((0, x))
        plt.ylim((y, 0))
        plt.show()

def _draw_sphere(r=100):
    shape = r # assume x==y==z
    Z,Y,X = shape,shape,shape
    vol = np.zeros((Z,Y,X))
    r = int(shape/2)-1
    center = round(shape/2)
    #print("R =",r)
    for z in range(Z):
        # formula: x^2 + y^2 = r^2 - z^2
        for y in range(Y):
            x_square  = r**2 - (z-center)**2 - (y-center)**2
            if x_square >= 0:
                x = math.sqrt(x_square)
                vol[z,y,round(x)+center] = 1
                vol[z,y,-round(x)+center] = 1
                ## for more precise only (assume shape_x == shape_y  
                vol[z,round(x)+center,y] = 1
                vol[z,-round(x)+center,y] = 1
    return vol

def _test_fpviewer():
    sphere = _draw_sphere()
    #xls_path="C:/Users/Panko/Desktop/tmpp.xlsx"
    status = FPViewer(sphere, [[40,40,40,60,60,60]], debug=False)()
    print("final fp status:", status)

class ExcelHandler():
    def __init__(self, excel_path=None):
        """
        excel_path is where to load data from,
        while save_path in self.save is where to save data to
        """
        self.excel_path = excel_path
        if excel_path!=None and os.path.exists(excel_path):
            self.prev_df = pd.read_excel(excel_path, sheet_name="Sheet1", converters={'pid':str,'bbox':str, 'isFP':int})
        else:
            self.prev_df = None
        self.new_data = []

    def add_new(self, pid, bbox, isFP):
        ## NOTE: All bbox should use the same coordinate system as standard testing image (i.e. VOI or fast_eval image)
        assert len(bbox)==6 #z1y1x1z2y2x2
        pid = str(pid)
        if type(self.prev_df)!=type(None):
            df = self.prev_df[self.prev_df["pid"]==pid]
            ## For practical usage, data checking should be used before eval to avoid sql injection
            existed_bbox = [eval(lst) for lst in df["bbox"]] #this is a bad practice, but it works here
            #print("existed_bbox", existed_bbox)
            if len(existed_bbox)>0:
                ious = iou_3D(existed_bbox, bbox)
                if len(existed_bbox)==1:
                    ious = [ious]
                if any(iou>0.5 for iou in ious):
                    print("Skip duplicated: pid={}, bbox={} with max_iou={}".format(pid, list(map(round, bbox)), max(ious)))
                    return # no add duplicated bbox    

        self.new_data.append({"pid":pid, "bbox":bbox, "isFP":isFP})
    
    def save(self, save_path):
        if type(self.prev_df)!=type(None): # need concat
            df = pd.DataFrame(self.new_data, columns=["pid", "bbox", "isFP"])
            df = pd.concat([self.prev_df, df])
        else:
            df = pd.DataFrame(self.new_data, columns=["pid", "bbox", "isFP"])
        df.to_excel(save_path, sheet_name="Sheet1", index=False)
    

def _test_df():
    datahandler = ExcelHandler("C:/Users/Panko/Desktop/tmp.xlsx")
    for i in range(3):
        pid = int(random.randint(100000,200000))
        bbox = (np.random.sample(6)*100).tolist()
        isfp = random.randint(0,2)
        datahandler.add_new(pid, bbox, isfp)
    datahandler.add_new(170033, [20,20,20,30,29,28.], 1)
    datahandler.save("C:/Users/Panko/Desktop/tmp.xlsx")

def correcting_fp(not_fp_excel_path, save=False, debug_view=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    parser.add_argument('--exp_name', type=str, default='debug', help='log experiment name')
    parser.add_argument('--dataset_name', type=str, default=CURRENT_DATASET_PKL_PATH)
    parser.add_argument('--npy_name', type=str, default="None.npy")
    opt = parser.parse_args()
    for exp_name, fold_num, epoch in [
        
        #('train_rc_config_5.6.4_resnest_shallower_f0', 0, 187),
        #('train_rc_config_5.6.5_resnest_shallower_f1', 1, 255),
        #('train_rc_config_5.6_resnest+sgd_shallower_f2', 2, 187),
        #('train_rc_config_5.6_resnest+sgd_shallower_f3', 3, 204),
        ('train_rc_config_5.6.2_resnest_shallower_f4', 4, 221)
        

   
        ]:

        testing_mode = 1 # use eval here to avoid repeat calculations

        fake_batch_size = 10 # need this to prevent accumulating fp crops in RAM
        eval_conf_thresh = 0.015

        opt.exp_name = exp_name
        logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='YOLOv4').get_log()
        phase = 'VAL'if testing_mode==0 else 'TEST' if testing_mode==1 else 'TRAIN_debug'
        #writer = SummaryWriter(log_dir=opt.log_path + '/{}_'.format(phase) + opt.exp_name)
        checkpoint_root = 'checkpoint/'
        checkpoint_folder = '{}{}'.format(checkpoint_root, exp_name)
        weight_path = '{}/backup_epoch{}.pt'.format(checkpoint_folder, str(epoch))
        if os.path.exists(weight_path):
            opt.weight_path = weight_path
            exp_name_folder = opt.exp_name + '_making_negatives'

            checkpoint_save_dir = 'preidction/{}/{}_conf{}'.format(exp_name_folder, str(epoch), eval_conf_thresh)
            #if not os.path.exists('preidction'):
            #    os.mkdir('preidction')

            #if not os.path.exists('preidction/{}'.format(exp_name_folder)):
            #    os.mkdir('preidction/{}'.format(exp_name_folder))

            #if not os.path.exists(checkpoint_save_dir):
            #    os.mkdir(checkpoint_save_dir)

            weight_path = opt.weight_path
            #weight_path = 'checkpoint/96_B4_F1/backup_epoch150.pt'
            trainer = Trainer(testing_mode=testing_mode,
                    weight_path=weight_path,
                    checkpoint_save_dir=checkpoint_save_dir,
                    resume=False,
                    gpu_id=opt.gpu_id,
                    accumulate=1,
                    fp_16=opt.fp_16,
                    writer=None,
                    logger=logger,
                    crx_fold_num=fold_num,
                    dataset_name=opt.dataset_name,
                    eval_interval=None,
                    npy_name=opt.npy_name,
                    eval_conf_thresh=eval_conf_thresh,
                    )

            # force do fp reduction
            #area_dist, area_iou, plt, _, cpm_dist, cpm, max_sens_dist, max_sens_iou = trainer.evaluate()
            if (1): #smaller pid set
                #target_pids = ["25607996"]
                start_from = 0
                target_pids = list(pid for _,_,pid in trainer.test_dataset.ori_dataset.data)
                target_pids = target_pids[start_from:]
                #target_pids = target_pids[120:121]
            else: # the whole pid set
                target_pids = list(pid for _,_,pid in trainer.test_dataset.ori_dataset.data)
            
            if (0): #debug
                ori_pids = []
                for i, (_,_,pid) in enumerate(trainer.test_dataset.ori_dataset.data):
                    ori_pids.append(pid)
                    if i==4:
                        break
                print("origin_pids", ori_pids[:5])  
                print("target_pids", target_pids[:5])
                raise EOFError
            
            remained_pids = target_pids.copy()
            n_batches = len(target_pids)//fake_batch_size + 1
            tqdm_bar = tqdm(total=n_batches, desc=f"Making fp fold {fold_num}")

            datahandler = ExcelHandler(not_fp_excel_path)
            if debug_view:
                dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH) # debug view usage only
                dataset.set_batch_1_eval(True, (1.25,0.75,0.75))
                dataset.set_lung_voi(True)
            processed_pid_count = 0
            all_pid_count = len(remained_pids)
            while len(remained_pids)!=0:
                pids = remained_pids[:fake_batch_size]
                remained_pids = remained_pids[fake_batch_size:]
                processed_pid_count += len(pids)
                #for c in range(ncopy):
                if (1):
                    ncopy = 5
                    out_imgs, out_bboxes, out_names, uncropped_bboxes = trainer.get_fp_for_reduction_batch(pids, return_crop_only=True, topk=ncopy, correct_fp_usage=True)
                    ## saving crops
                    for c_raw, (img, bboxes, pid, ori_bboxes) in enumerate(zip(out_imgs, out_bboxes, out_names, uncropped_bboxes)):
                        to_save = (img, bboxes)
                        assert len(bboxes)==1, "bboxes has length {}, bboxes={}".format(len(bboxes), bboxes)

                        c = c_raw%ncopy + 1
                        if c==1:
                            current_pid = pid
                        elif pid != current_pid:
                            warnings.warn(f"Alg Error: pid={current_pid} has only {c-1} copy, whike ncopy={ncopy}")
                            current_pid = pid
                            c=1

                        if (0) or debug_view: #debug view (whole img version)
                            dataset.get_data([pid])
                            debug_img, _, _ = dataset[0]
                            debug_img = debug_img.squeeze_(-1).numpy()
                            print("ori bboxes", ori_bboxes)
                            debug_bbox = [ori_bboxes.tolist()]
                            AnimationViewer(debug_img, debug_bbox, note=f"{pid} big debug view")

                        isfp = FPViewer(img, bboxes)()
                        if isfp==0: # i.e. TP
                            bbox = ori_bboxes.tolist()
                            print("Adding pid={}, bbox={} to notFP set".format(pid, bbox))
                            datahandler.add_new(pid, bbox, isfp)
                        elif isfp==2: # Bad FP
                            bbox = ori_bboxes.tolist()
                            print("Adding pid={}, bbox={} to BadFP set".format(pid, bbox))
                            datahandler.add_new(pid, bbox, isfp)
                        else: #isfp==1 # FP
                            assert isfp==1
                            bbox = ori_bboxes.tolist()
                            datahandler.add_new(pid, bbox, isfp)
                tqdm_bar.update()
                #print("datahandler.new_data", datahandler.new_data)
                if save:
                    try:
                        datahandler.save(not_fp_excel_path)
                    except: # save failed, maybe read-only or invalid-path
                        if os.path.exists(not_fp_excel_path): #read-only
                            s = not_fp_excel_path[::-1].index(".")
                            if s!=-1:
                                new_path = not_fp_excel_path[:len(not_fp_excel_path)-1-s] # e.g. ".../fp_recent_tmp.xlsx" -> ".../fp_recent_tmp"
                                ext = not_fp_excel_path[len(not_fp_excel_path)-1-s:] # e.g. ".xlsx"
                                s = new_path[::-1].index("_")
                                if s!=-1:
                                    suffix = new_path[len(new_path)-1-s:] # e.g. "_tmp" or "_2"
                                    others = new_path[:len(new_path)-1-s] # e.g. ".../fp_recent" 
                                    if suffix[1:].isnumeric():
                                        n = int(suffix[1:])
                                        new_path = others + f"_{n}" + ext
                                        while os.path.exists(new_path):
                                            n += 1
                                            new_path = others + f"_{n}" + ext
                                        datahandler.save(new_path)
                                else: # no "n" yet, first file
                                    n=2
                                    new_path = new_path + f"_{n}" + ext
                                    while os.path.exists(new_path):
                                        n += 1
                                        new_path = others + f"_{n}" + ext
                                    datahandler.save(new_path)
                    finally: # a temporary path
                        default_path = "D:/CH/LungDetection/fp_recent_tmp.xlsx"
                        datahandler.save(default_path)
                print("\n********** Progress: {}/{} ***********\n".format(processed_pid_count, all_pid_count))
                        

if __name__ == "__main__":
    #_test_fpviewer()
    #_test_df()
    """
    isfp==1 : regular fp, supporting copy-pastE (need bbox size exclusion)
    isfp==0 : TP
    isfp==2 : bad fp, not supporting copy-paste, but is also eligible for fp-reduction
    """
    not_fp_excel_path = "D:/CH/LungDetection/not_fp_1.25mm.xlsx"
    correcting_fp(not_fp_excel_path, save=False)
    