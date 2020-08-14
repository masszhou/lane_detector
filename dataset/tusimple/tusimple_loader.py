#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from typing import List, Optional
from copy import deepcopy

from dataset.types import Sample, TuSimpleLabel, BatchSample


class DatasetTusimple(Dataset):
    """ Data loader class """
    def __init__(self, root_path, json_files, transform=None):
        """ initialize (load data set from url) """
        self.root_path = root_path
        if isinstance(json_files, str):
            self.json_files = [json_files]
        else:
            self.json_files = json_files
        self.transform = transform

        # load training set
        self.label_data: List[TuSimpleLabel] = []

        for each_json in self.json_files:
            with open(root_path + each_json) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    json_string = json.loads(line)
                    self.label_data.append(json_string)

    def __len__(self):
        return len(self.label_data)  # 3626 for tusimple

    def __getitem__(self, idx: int) -> Sample:
        """return readable sample data

        :param idx:
        :return:
        """
        sample_label = self.label_data[idx]
        img_path = self.root_path + sample_label["raw_file"]
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # bgr to rgb
        rows, cols = img.shape[:2]
        lane_list = []
        for id_pos, each_lane in enumerate(sample_label["lanes"]):
            lane_pts = []
            for x, y in zip(each_lane, sample_label["h_samples"]):
                if x > 0:
                    lane_pts.append((x, y))
            if len(lane_pts) > 0:
                lane_list.append(np.array(lane_pts, dtype=float))

        sample: Sample = {"image_path": img_path,
                          "image": img,
                          "id": idx,
                          "lane_list": lane_list,
                          "original_size": (rows, cols)
                          }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch: List[Sample]) -> BatchSample:
        """convert readable sample data as aligned ground truth data

        Ground truth data is used for loss calculation

        :param batch: List[Sample]
        :return:
        """
        grid_y = 32
        grid_x = 64
        resize_ratio = 8  # resize ratio = 8, resize [256, 512] to [32, 64]
        img = np.stack([b['image'] for b in batch], axis=0)
        image_id_list = [each["id"] for each in batch]

        n_samples = len(batch)
        # build detection ground truth
        detection_gt = np.zeros((n_samples, 3, grid_y, grid_x))  # [3, 1, 32, 64]
        for i_smaple, sample in enumerate(batch):
            for i_lane, lane_pts in enumerate(sample['lane_list']):
                for i_pt, xy_pt in enumerate(lane_pts):
                    x_index = int(xy_pt[0] / resize_ratio)  # resize ratio = 8, resize [256, 512] to [32, 64]
                    y_index = int(xy_pt[1] / resize_ratio)
                    detection_gt[i_smaple][0][y_index][x_index] = 1.0  # confidence
                    detection_gt[i_smaple][1][y_index][x_index] = (xy_pt[0] * 1.0 / resize_ratio) - x_index  # offset x
                    detection_gt[i_smaple][2][y_index][x_index] = (xy_pt[1] * 1.0 / resize_ratio) - y_index  # offset y

        # build instance ground truth, inefficient code but better reading
        instance_gt = np.zeros((n_samples, 1, grid_y * grid_x, grid_y * grid_x))  # [8, 1, 2048, 2048]
        for i_smaple, sample in enumerate(batch):
            temp = np.zeros((1, grid_y, grid_x))  # e.g. [1, 32, 64]
            lane_cluster = 1
            for i_lane, lane_pts in enumerate(sample['lane_list']):
                previous_x_index = 0
                previous_y_index = 0
                for i_pt, xy_pt in enumerate(lane_pts):
                    x_index = int(xy_pt[0] / resize_ratio)  # resize ratio = 8, resize [256, 512] to [32, 64]
                    y_index = int(xy_pt[1] / resize_ratio)
                    temp[0][y_index][x_index] = lane_cluster

                    if previous_x_index != 0 or previous_y_index != 0:  # interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while True:
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster
                            if temp_x < x_index:
                                temp[0][temp_y][temp_x + 1] = lane_cluster
                                delta_x = 1
                            elif temp_x > x_index:
                                temp[0][temp_y][temp_x - 1] = lane_cluster
                                delta_x = -1
                            if temp_y < y_index:
                                temp[0][temp_y + 1][temp_x] = lane_cluster
                                delta_y = 1
                            elif temp_y > y_index:
                                temp[0][temp_y - 1][temp_x] = lane_cluster
                                delta_y = -1
                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break

                    previous_x_index = x_index
                    previous_y_index = y_index

                lane_cluster += 1

            for i_sim in range(grid_y * grid_x):  # make gt
                temp = temp[temp > -1]
                gt_one = deepcopy(temp)
                if temp[i_sim] > 0:
                    gt_one[temp == temp[i_sim]] = 1  # same instance
                    if temp[i_sim] == 0:
                        gt_one[temp != temp[i_sim]] = 3  # different instance, different class
                    else:
                        gt_one[temp != temp[i_sim]] = 2  # different instance, same class
                        gt_one[temp == 0] = 3  # different instance, different class
                    instance_gt[i_smaple][0][i_sim] += gt_one

        samples: BatchSample = {'image': img,
                                "image_id": image_id_list,  # help tracing source image
                                'detection_gt': detection_gt,
                                'instance_gt': instance_gt,
                                }
        return samples


