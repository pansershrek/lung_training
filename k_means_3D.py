"""
Original source for 2D bbox by lars76 on github: 
https://github.com/lars76/kmeans-anchor-boxes/blob/master/kmeans.py
"""
import numpy as np
import random

from dataset import Tumor, LungDataset
from global_variable import CURRENT_DATASET_PKL_PATH

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i.e. d, h, w)
    :param clusters: numpy array of shape (k, 3) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    z = np.minimum(clusters[:, 0], box[0]) #shape (k,)
    y = np.minimum(clusters[:, 1], box[1])
    x = np.minimum(clusters[:, 2], box[2])

    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0 or np.count_nonzero(z == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y * z
    box_area = box[0] * box[1] * box[2]
    cluster_area = clusters[:, 0] * clusters[:, 1] * clusters[:, 2]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_ # shape (k,)


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows (r = N = # of bbox)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])



def kmeans(boxes, k, dist=np.median, seed=None):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 3), where r is the number of rows
    :param k: int, number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 3)
    """
    rows = boxes.shape[0] # r == N == (# of bbox)

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    

    # the Forgy method will fail if the whole array contains the same rows
    # BUG: 同bbox可能被多次選中, "[0,rows)裡面做k次random.choice" 可能重複
    # np.random.seed(seed=seed)
    # clusters = boxes[np.random.choice(rows, k, replace=False)] # 在[0,rows)裡面做k次random.choice, 再依據這些index從boxes中slice下來; (rows,)->(k,)

    random.seed(seed)
    chosen_idx = []
    rows_to_choose = list(range(rows))
    for _ in range(k):
        idx = random.choice(rows_to_choose)
        rows_to_choose.remove(idx)
        chosen_idx.append(idx)
    clusters = boxes[chosen_idx] # subset of boxes as initial cluster; shape (k,3)

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters) #(k,)

        nearest_clusters = np.argmin(distances, axis=1) #index of the nearest cluster; value_range: [0,k), shape: (rows,)

        if (last_clusters == nearest_clusters).all(): # if no update
            break

        for cluster in range(k):
             # 把boxes裡面, nearest_cluster == (current) cluster的box找出來，形成subset，再找subset的中間位置 
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0) # shape: (3,)

        last_clusters = nearest_clusters

    return clusters

def get_anchors(random_crop_file_prefix, k, seed=None):
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.set_random_crop(random_crop_file_prefix, 5, True) # only one copy is needed to calculate anchor
    print("dataset.random_crop_ncopy:", dataset.random_crop_ncopy)
    boxes = []
    for _, bboxes, pid in dataset:
        bboxes = bboxes.tolist()
        for bbox in bboxes:
            bbox = bbox[:6]
            dwh = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
            boxes.append(dwh)
    boxes = np.array(boxes)
    print("start kmeans, boxes shape:", boxes.shape)
    clusters = kmeans(boxes, k, seed=seed)
    print("clusters: ")
    print(clusters)
        


def _test():
    np.random.seed(123) # to contruct same boxes only
    boxes = (np.random.sample((100,3)) * 100 + 1).astype(np.int64)
    np.random.seed()
    clusters = kmeans(boxes, k=3)
    print(clusters)

if __name__ == "__main__":
    #_test()
    for i in range(1):
        get_anchors("random_crop_128x128x128_1.25x0.75x0.75", k=9)