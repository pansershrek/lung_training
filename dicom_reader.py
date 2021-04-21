"""
2020/10/14 預期WorkFlow

原圖 --> normalize/clip to [-1,1] --> more preprocssing? --套合-------> Crop+resize ---> visualize/training
    |--> 套VOI 與 GT -------> 延伸/內縮 VOI -----------------|

"""
import PIL
import os
from os.path import join as pjoin
import pydicom
from pydicom.filereader import read_dicomdir, dcmread
import numpy as np
#import zipfile
import time

#from global_variable import VOI_EXCEL_PATH 
import utils_hsz


class DicomdirManager:
    """
    不需要GDCM(?)
    """
    def __init__(self, path, check=True):
        """
        For LungDetection, each path should have exact 1 series (of exactly 1 study from exactly 1 patient).

        self.path: path to the dicom directory
        self.uid: SeriesInstanceUID used in ground truth excel
        self.pid: patient's id of this dicom directory
        """
        self.path = path
        raw_slices = []
        for f in os.listdir(path):
            if "." not in f:
                assert f.isnumeric(), f"Bad dicom file name: {f} in {path}"
                raw_slices.append(f)
        s0 = pydicom.read_file(pjoin(self.path, raw_slices[0])) #sample

        self.pid = s0.PatientID
        if "StudyDescription" in dir(s0):
            self.study_description = s0.StudyDescription
        self.uid = s0.SeriesInstanceUID #used in ground truth excel
        self.PixelSpacing = [v.real for v in s0.PixelSpacing]
        self.SliceThickness = s0.SliceThickness.real
        self.raw_slices = raw_slices
        if check:
            self.check()
    
    @property
    def transform(self):
        return (self.SliceThickness, self.PixelSpacing[1], self.PixelSpacing[0]) #z,y,x

    def check(self):
        """
        check if there is exact one series in that directory
        Also, construct self.slices, and return # of slices

        self.slices: A dictionary
        """
        def isContinuous(lst:list):
            """Check if a list contains only continuous integers"""
            lst = sorted(lst)
            diff_lst = [lst[i]-lst[i-1] for i in range(1,len(lst))]
            return all(val==1 for val in diff_lst)

        slice_idxes = {}
        for sname in self.raw_slices:
            fname = pjoin(self.path, sname)
            s = pydicom.read_file(fname)
            pid = s.PatientID
            uid = s.SeriesInstanceUID
            slice_idx = s.InstanceNumber.real
            assert pid==self.pid, f"Pid mismatch for slice '{sname}' in dir '{self.path}'\n'{pid} != {self.pid}'"
            assert uid==self.uid, f"SeriesInstanceUID mismatch for slice '{sname}' in dir '{self.path}'\n'{uid} != {self.uid}'"
            if slice_idx not in slice_idxes:
                slice_idxes[slice_idx] = sname
            else:
                raise ValueError(f"Repeated slice_idx '{slice_idx}' detected in {self.path}")
        assert isContinuous(slice_idxes), f"Incontinuous indices detected in '{self.path}'"
        self.slices = slice_idxes
        s0 = min(slice_idxes)
        self.InstanceNumber = s0
        s0 = slice_idxes[ s0 ] # min slice
        s0 = pydicom.read_file(  pjoin(self.path, s0) )
        self.ImagePositionPatient = [val.real for val in s0.ImagePositionPatient]
        self.SliceLocation = s0.SliceLocation.real
        assert self.SliceLocation == self.ImagePositionPatient[-1]
        n_slice = len(slice_idxes)
        return n_slice

    def __len__(self):
        return len(self.raw_slices)

    def getFirstSliceDCM(self):
        s0 = self.slices[self.InstanceNumber]
        s0 = pydicom.read_file(  pjoin(self.path, s0) )
        return s0
    

        



class DicomReader(DicomdirManager):
    """
    Need GDCM(?), 用來真正打開影像
    """
    def __init__(self, data_path, check=True):
        super(DicomReader, self).__init__(data_path, check=check)
        self.data_path = data_path       


    def get_series(self, norm_hu=True):
        if "slices" not in dir(self):
            self.check()
        lst = sorted(self.slices.keys())
        slices = []
        for s in lst:
            sname = self.slices[s]
            s = pydicom.read_file(pjoin(self.path, sname))
            slices.append(s)
        if norm_hu:
            volume = utils_hsz.get_pixels_hu(slices)
        else:
            volume = [s.pixel_array for s in slices]
        volume = np.array(volume, dtype=np.int16)
        return volume
            
    def get_series_with_transform(self, norm_hu=True):
        """
        回傳對應volume的np.array, dtype=uint16
        transform是用於utils_hsz.resample，將x,y,z的間隔統一成[1,1,1]所使用的np.array, 順序為[z,y,x]!!
        """
        volume = self.get_series(norm_hu=norm_hu) # shape: z,y,x
        transform = [self.SliceThickness, self.PixelSpacing[1], self.PixelSpacing[0]] # z,y,x
        return volume, transform


   

###  BELOW IS TESTING REGION
###
###
    
#def main():
#    global dcm
#    DATA_PATH = r"D://Lab/研究/彰基breast MRI/DATA"
#    cases=["1.2.528.1.1001.200.10.10073.12729.2500766177.20190830045933496"]
#    for case in cases:
#        if "DICOMDIR" in os.listdir(os.path.join(DATA_PATH, case)):
#            dcm = dcmread(os.path.join(DATA_PATH, case, "DICOMDIR"), force=True)
#        else:
#            raise Value(f"No DICOMDIR for {case}")

def _test_dicomdir():
    global dcm_man
    DATA_PATH = r"D:/CH/LungDetection/DATA/1500_DICOM"
    cases=["1873390"]
    dcm_man = pydicom.read_file(pjoin(DATA_PATH, cases[0]))
    print(dcm_man)

def _test_dicomreader():
    global dcm_man
    """try:
        import gdcm
    except:
        print("Need GDCM environment, e.g. gdcm in conda-forge")
        raise"""
    DATA_PATH = r"D:/CH/LungDetection/DATA/1500_DICOM"
    cases=[r"1873390\20052606\18010001"]
    dcm_man = DicomReader( pjoin(DATA_PATH, cases[0]) )
    volume, transform = dcm_man.get_series_with_transform()
    print("transform:", transform)
    print("volume", volume.shape, volume.dtype)
    #volume = dcm_man.get_series("sub")
    #print(volume.shape)
    #print(volume.dtype)
    

def _test_gdcm():
    PATH=r"D://Lab/MRI_fast_entrance/1.2.528.1.1001.200.10.10073.12729.2500766177.20190829050127731/SDY00000/SRS00007"
    slices = [pydicom.read_file(pjoin(PATH,s)) for s in os.listdir(PATH)]
    arr = slices[0].pixel_array
    print(arr)
    print(type(arr))
    print(arr.dtype)
    arr2 = arr.astype(np.float32)
    assert np.array_equal(arr, arr2), "inconsistent array after dtype conversion!"
    print(arr2.dtype)
    print(arr2)
    print("shape", arr2.shape)
    
if __name__ == "__main__":
    #_test_gdcm()
    #_test_dicomdir()
    _test_dicomreader()
        
