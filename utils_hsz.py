import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.ndimage
import torch

#from global_variable import MRI_DATA_PATH


class AnimationViewer(object):
    """
    Visualizing 3D np.array in slices
    Modified to support scrolling and adding bbox
    bbox is of format [z1,y1,x1,z2,y2,x2]
    """
    def __init__(self, volume, bbox=None, verbose=True, note="", draw_face=True):
        self.image = volume
        self.draw_face = draw_face
        self.bbox=bbox  if type(bbox)!=type(None) else ()
        self.pixel_min = np.min(volume)
        self.pixel_max = np.max(volume)
        self.volume_shape = volume.shape
        self.note = f"{note}: " if note !="" else ""
        print(f'shape={self.image.shape} maxvalue={self.pixel_max} minvalue={self.pixel_min}')
        if verbose:
            print(f"bbox (viewer): {bbox}")
        self.multi_slice_viewer()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]
        #print(ax.index)
        ax.images[0].set_array(volume[ax.index])
        self.add_bbox(ax)
        ax.set_title(f"{self.note}shape = {self.volume_shape}, z_index= {ax.index}")

    def next_slice(self, ax):
        volume = ax.volume

        ax.index = (ax.index + 1) % volume.shape[0]
        #print(ax.index)
        ax.images[0].set_array(volume[ax.index])
        self.add_bbox(ax)
        ax.set_title(f"{self.note}shape = {self.volume_shape}, z_index= {ax.index}")

    def remove_all_patch(self, ax):
        for p in ax.patches:
            p.remove()

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        move = event.key if event.name=="key_press_event" else event.button if event.name=="scroll_event" else None
        if move == "up" :
            self.remove_all_patch(ax)
            self.previous_slice(ax)

        elif move == "down":
            self.remove_all_patch(ax)
            self.next_slice(ax)

        fig.canvas.draw()

    def add_bbox(self, ax):
        if type(self.bbox)!=np.ndarray and self.bbox == None:
            return
        #elif type(self.bbox)==np.ndarray:
        #    boxes = [self.bbox]
        else:
            boxes = self.bbox
        for box in boxes:
            z1,y1,x1,z2,y2,x2 = box
            if z1 <= ax.index <= z2:
                w, h = x2-x1, y2-y1
                if self.draw_face and (ax.index in [z1,z2]):
                    rect = Rectangle((x1,y1), w, h, linewidth=1,edgecolor='r',facecolor='r')
                else:
                    rect = Rectangle((x1,y1), w, h, linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)

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

        plt.xlim((0, x))
        plt.ylim((y, 0))
        plt.show()

def resample(image, transform, new_spacing=[1,1,1]):
    """
    把每個Volume的pixel間隔(x,y,z軸)，統一成 new_spacing
    """
    # Determine current pixel spacing
    spacing = transform

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    #  Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    #transform = slices[0].PixelSpacing
    #transform.insert(0, slices[0].SliceThickness)
    #transform = np.array(transform, dtype=np.float32)

    return np.array(image, dtype=np.int16)#, transform

def normalize(volume, MIN_BOUND = -1150, MAX_BOUND = 350): # WL=-400, WW=1500
    data = volume
    data = np.array(data, dtype = np.float32)
    data = (data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    data[data>1] = 1.0
    data[data<0] = 0.0
    data = data - np.mean(data) ########## holp mean value closed to zero
    return data
    

def resample_torch(image, transform, new_spacing=(1,1,1), mode="nearest"):
    """
    把每個Volume的pixel間隔(x,y,z軸)，統一成 new_spacing
    只要image, transform, new_spacing彼此的順序一樣即可 (e.g. all (x,y,z) or all (z,y,x))
    """
    # Determine current pixel spacing
    spacing = np.array(transform)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    t = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    #print("REAL RESIZE FACTOR", real_resize_factor)
    new_shape = new_shape.astype(np.int32).tolist()
    #print("new shape", new_shape)
    t = torch.nn.functional.interpolate(t, size=new_shape, mode=mode) # trilinear no better
    #print("After interpolate", t.shape)
    image = t.squeeze(0).squeeze(0)
    
    return image, new_spacing

def _test_resample():
    """
    Note: 
    tumor 96 的 sub反序 (已debug，仍可使用這筆資料)
    tumor 134 的 t1 不連續 (無法使用這張t1, 缺關鍵部分)
    tumor 136 的 sub 不連續 (由於剛好卡邊上，決定放棄此資料)
    tumor 137 的 t1 不連續 (無法使用這張t1, 缺關鍵部分)
    tumor 281 在voi_extend=[5,5,5]時，無法獲得sub/t1/t2 (用voi_extend=[0,0,0]處理這筆資料)
    """
    global MRIDataset, Tumor
    from dataset import Tumor, MRIDataset
    import utils
    from pydicom.filereader import read_dicomdir, dcmread
    from os.path import join as pjoin
    model_input_shape= (50,50,50)
    pad_mode="center"
    pad_cval=0
    voi_extend = [5,5,5]
    to_crop_series="sub"
    resize_before_pad = True
    dataset = MRIDataset.load("mri_dataset_20201118.pkl")
    tumor = dataset.tumors[52] # 96 has reverse order!
    #tumor = dataset.get_tumor(102)
    #print(tumor.access, tumor.excel_r)
    #print("voi", tumor.voi)
    print(tumor.path)
    sub, transform = tumor.get_series_with_transform("sub") #shape: z,y,x
    to_crop, transform2 = tumor.get_series_with_transform(to_crop_series)
    sub_shape = sub.shape
    to_crop_shape = to_crop.shape
    print("Accession:", tumor.access)
    print("sub.shape", sub_shape)
    print("to_crop.shape", to_crop_shape)
    print("transform1", transform)
    print("transform2", transform2)
    sub_norm = np.array(sub_shape)*np.array(transform)
    to_crop_norm = np.array(to_crop_shape)*np.array(transform)
    print("sub norm shape", sub_norm)
    print("to_crop norm shape", to_crop_norm)
    target_voi = utils.voi_registration(sub_shape, tumor.voi, to_crop_shape, voi_extend) ## TODO: z-slice interpolation? (too much empty z-slice in npy)
    print("ori target_voi:", target_voi)
    target_voi = utils.registrate_voi_using_dcm(tumor.voi, to_crop_series, tumor, sub_shape, voi_extend)
    print("new target_voi:", target_voi)
    #raise EOFError
    # try adjust z-thickness!
    transform2[0] = (tumor.voi["z"][1] - tumor.voi["z"][0] + 1 + voi_extend[0]) / (target_voi["z"][1] - target_voi["z"][0] + 1 + voi_extend[0])

    print(tumor.series_list)
    """dcm = dcmread(pjoin(MRI_DATA_PATH, tumor.path, "DICOMDIR"))
    study = dcm.patient_records[0].children[0]
    out_t2 = ""
    out_sub = ""
    if to_crop_series=="t2":
        fast_tuple = ("tirm_tra_p3", "t2_tirm_tra", "tirm_tra")
    elif to_crop_series=="t1":
        fast_tuple = ("tse_t1_tra_p4", "t1_tse_tra", "tse_t1_tra_p3")
    else:
        fast_tuple = (to_crop_series)
    for series in study.children:
        series_name = series.SeriesDescription.lower()
        for s in fast_tuple:
            if s in series_name:
                for i, x in enumerate(series.children):
                    out_t2 += x.__str__() + "\n" + "@"*50 +"\n"
                    #print(f"slice {i}: loc={x.SliceLocation}, instance={x.InstanceNumber}")
                #print("total slice:", i+1)
                break
        if series_name == "sub":
            for i,x in enumerate(series.children):
                out_sub += x.__str__() + "\n" + "@"*50 +"\n"
                print(f"slice {i}: loc={x.SliceLocation}, instance={x.InstanceNumber}")

    with open("tmp3.txt", "w") as f:
        f.write(out_sub)
    with open("tmp4.txt", "w") as f:
        f.write(out_t2)
    #raise EOFError"""
    #print(f"original shape: {sub.shape}, {sub.dtype}")
    to_crop = utils.linear_normalization(to_crop, newMin=0, newMax=1, dtype=np.float64) # normalization
    to_crop = to_crop - to_crop.mean() # zero-centering
    try:
        cropped_voi = utils.crop_voi(to_crop, target_voi, extend=voi_extend) # cropped
    except:
        print(f"An error occur while '{to_crop_series}' series preprocessing; tumor on excel_r={tumor.excel_r}")
        raise
    print(f"after crop: {cropped_voi.shape}, {cropped_voi.dtype}") 
    AnimationViewer(cropped_voi)
    ### This part seems useless
    equal_spacing_cropped_voi, new_spacing = resample( cropped_voi, transform2, new_spacing=[1,1,1] )
    print("new_spacing")
    print(f"after hsz_resample: {equal_spacing_cropped_voi.shape}")
    AnimationViewer(equal_spacing_cropped_voi)

    """equal_spacing_cropped_voi, new_spacing = resample_torch( cropped_voi, transform2, new_spacing=[1,1,1] )
    print("new_spacing2")
    print(f"after hsz_resample2: {equal_spacing_cropped_voi.shape}")
    AnimationViewer(equal_spacing_cropped_voi)"""
    ##### equal_spacing_cropped_voi = cropped_voi
    ### resize to input shape, what is input shape??
    #minval = equal_spacing_cropped_voi.min()
    """ # sk_resize, have some bugs(?)
    resized = sk_resize(equal_spacing_cropped_voi, model_input_shape, order=3, mode="constant", cval=0, preserve_range=True)  #resize to same input shape
    print(f"after resize: {resized.shape}, {resized.dtype}")
    utils_hsz.AnimationViewer(resized)
        """      
    #resized = utils.resize(equal_spacing_cropped_voi, model_input_shape) #deprecated
    if pad_cval=="min":
        min_val = equal_spacing_cropped_voi.min()
        resized = utils.resize_and_pad(equal_spacing_cropped_voi, model_input_shape, mode=pad_mode, cval=min_val, resize_before_pad=resize_before_pad)
    else:
        resized = utils.resize_and_pad(equal_spacing_cropped_voi, model_input_shape, mode=pad_mode, cval=pad_cval, resize_before_pad=resize_before_pad)
    print(f"after pad: {resized.shape}, {resized.dtype}") 
    AnimationViewer(resized)


if __name__ == "__main__":
    _test_resample()
