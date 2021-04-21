import numpy as np
import argparse

from dataset import Tumor, LungDataset
from global_variable import CURRENT_DATASET_PKL_PATH


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1] * boxes[:, 2]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1] * clusters[:, 2]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

        box_d_matrix = np.reshape(boxes[:, 2].repeat(k), (n, k))
        cluster_d_matrix = np.reshape(np.tile(clusters[:, 2], (1, n)), (n, k))
        min_d_matrix = np.minimum(cluster_d_matrix, box_d_matrix)

        inter_area = np.multiply(min_w_matrix, min_h_matrix)
        inter_area = np.multiply(inter_area, min_d_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data, save_dir):
        f = open(save_dir, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d,%d" % (data[i][0], data[i][1], data[i][2])
            else:
                x_y = ", %d,%d,%d" % (data[i][0], data[i][1], data[i][2])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(',')
            width = int(float(infos[3])) - \
                int(float(infos[0]))
            height = int(float(infos[4])) - \
                int(float(infos[1]))
            depth = int(float(infos[5])) - \
                int(float(infos[2]))
            dataSet.append([width, height, depth])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self, boxes):
        all_boxes = boxes
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        #self.result2txt(result, save_dir)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cluster_number', '-c', type=int, default=3, help='the wanted number of anchors, default: 6'
    )
    parser.add_argument(
        '--GT_file', '-g', type=str,
        help='the path of ground truth file'
    )

    parser.add_argument(
        '--save_dir', '-s', type=str, default='kmeans_anchors.txt',
        help='the path of ground truth file, default :\'kmeans_anchors.txt\''
    )
    dataset = LungDataset.load(CURRENT_DATASET_PKL_PATH)
    dataset.set_random_crop("random_crop_16x128x128_5.0x0.75x0.75", 20, True) # only one copy is needed to calculate anchor
    print("dataset.random_crop_ncopy:", dataset.random_crop_ncopy)
    boxes = []
    for _, bboxes, pid in dataset:
        bboxes = bboxes.tolist()
        for bbox in bboxes:
            bbox = bbox[:6]
            dwh = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
            boxes.append(dwh)
    boxes = np.array(boxes)
    kmeans_parameter = parser.parse_args()
    cluster_number = kmeans_parameter.cluster_number
    filename = kmeans_parameter.GT_file
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters(boxes)
