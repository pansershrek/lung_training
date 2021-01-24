from dataset import Tumor, LungDataset
from collections import OrderedDict
import matplotlib.pyplot as plt
from global_variable import CURRENT_DATASET_PKL_PATH

def get_transform(dataset):
    repeated=set()
    all_spacings = []
    all_thickness = []
    all_zs = []
    all_hws = []
    norm_spacing = (1.25, 0.75, 0.75)
    for pid in dataset.pids:
        if pid in repeated:
            continue
        repeated.add(pid)
        tumor = dataset.tumors[ dataset.pid_to_excel_r_relation[pid][0] ]
        spacing = tumor.dcm_reader.PixelSpacing
        assert spacing[0]==spacing[1], "x,y has different spacing: {} != {}".format(spacing[0], spacing[1])
        all_spacings.append(spacing[0])
        thickness = tumor.dcm_reader.SliceThickness
        all_thickness.append(thickness)
        shape = tumor.original_shape
        norm_shape = shape[0]*thickness/norm_spacing[0],  shape[1]*spacing[1]/norm_spacing[1], shape[2]*spacing[0]/norm_spacing[2]
        assert norm_shape[1]==norm_shape[2]
        all_zs.append(norm_shape[0])
        all_hws.append(norm_shape[1])
    return all_spacings, all_thickness, all_zs, all_hws

class Stats():
    def __init__(self, lst:list):
        statistics = OrderedDict()
        for i in lst:
            if i not in statistics:
                statistics[i]=1
            else:
                statistics[i]+=1
        #plt.bar(statistics.keys(), statistics.values())
        self.statistics = statistics
    def filter_show(self, count=0):
        view=self.statistics.copy()
        for key in self.statistics:
            if self.statistics[key]<count:
                view.pop(key)
        return view
    def __repr__(self):
        return str(self.filter_show(count=10))


def main(dataset_path):
    dataset = LungDataset.load(dataset_path)
    dataset.get_data(dataset.pids)
    all_spacings, all_thickness, all_zs, all_hws = get_transform(dataset)
    spacing_stat = Stats(all_spacings)
    thickness_stat = Stats(all_thickness)
    z_stat = Stats(all_zs)
    hw_stat = Stats(all_hws)
    print(spacing_stat)
    plt.hist(all_spacings, bins=100)
    plt.show()
    print(thickness_stat)
    plt.hist(all_thickness, bins=100)
    plt.show()
    print(z_stat)
    plt.hist(all_zs, bins=100)
    plt.show()
    print(hw_stat)
    plt.hist(all_hws, bins=100)
    plt.show()
    print("average spacing", sum(all_spacings)/len(all_spacings))
    print("average thickness", sum(all_thickness)/len(all_thickness))
    print("average z", sum(all_zs)/len(all_zs))
    print("average hw", sum(all_hws)/len(all_hws))
    ## averagely and techniquely speacking, target_transform (z,y,x) = (1.25, 0.75, 0.75)

if __name__ == "__main__":
    print("Using:", CURRENT_DATASET_PKL_PATH)
    main(CURRENT_DATASET_PKL_PATH)