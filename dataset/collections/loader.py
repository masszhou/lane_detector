import random
from torch.utils.data import Dataset
from typing import List
from copy import deepcopy
import numpy as np

from dataset.types import Sample, BatchSample
from dataset.bdd100k import DatasetBDD100K
from dataset.culane import DatasetCULane
from dataset.tusimple import DatasetTusimple
from configs.PINet import ParamsBDD100K
from configs.PINet import ParamsCuLane
from configs.PINet import ParamsTuSimple


class DatasetCollections(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        params_tusimple = ParamsTuSimple()
        params_culane = ParamsCuLane()
        params_bdd100k = ParamsBDD100K()

        # 3626 for tusimple, autobahn
        print("loading tusimple... ")
        self.train_dataset_tusimple = DatasetTusimple(params_tusimple.train_root_url,
                                                      params_tusimple.train_json_file)
        # 88880 CULane train samples, Beijing city
        print("loading culane... ")
        self.train_dataset_culane = DatasetCULane(params_culane.train_root_url,
                                                  params_culane.train_json_file)
        # 69863 bdd100k, 69863, US city
        print("loading bdd100k... ")
        self.train_dataset_bdd100k = DatasetBDD100K(params_bdd100k.train_root_url,
                                                    params_bdd100k.train_json_file)

        tusimple_size = len(self.train_dataset_tusimple)
        tusimple_sequence = [(0, i) for i in range(tusimple_size)]

        culane_size = len(self.train_dataset_culane)
        culane_sequence = [(1, i) for i in range(culane_size)]

        bdd100k_size = len(self.train_dataset_bdd100k)
        bdd100k_sequence = [(2, i) for i in range(bdd100k_size)]

        # proportion
        # tusimple x5 -> 10.25%
        # culane x1 -> 50.25%
        # bdd100k x10 -> 39.5%
        self.random_sequence = tusimple_sequence * 5 + culane_sequence + bdd100k_sequence
        random.shuffle(self.random_sequence)

    def __len__(self):
        return len(self.random_sequence)  # 88880 CULane train samples

    def __getitem__(self, idx: int) -> Sample:
        set_id, sample_id = self.random_sequence[idx]
        if set_id == 0:
            sample = self.train_dataset_tusimple[sample_id]
        elif set_id == 1:
            sample = self.train_dataset_culane[sample_id]
        elif set_id == 2:
            sample = self.train_dataset_bdd100k[sample_id]
        else:
            raise
        sample["set_id"] = set_id

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
        set_id_list = [each["set_id"] for each in batch]

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
                                "set_id": set_id_list,
                                'detection_gt': detection_gt,
                                'instance_gt': instance_gt,
                                }
        return samples
