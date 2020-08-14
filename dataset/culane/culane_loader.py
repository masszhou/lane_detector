#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
from typing import List
import cv2
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy

from dataset.types import Sample, BatchSample


class DatasetCULane(Dataset):
    def __init__(self,
                 root_path="/media/zzhou/data-culane/",
                 index_file="list/train.txt",
                 transform=None):
        self.root_path = Path(root_path)
        self.index_file = Path(index_file)
        self.transform = transform

        # mask for car hood
        carhood_mask = self.get_car_hood_mask()  # key=y, value=(xmin, xmax)

        # read index file
        self.train_files_path = []
        with open(self.root_path / self.index_file) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.train_files_path.append(line[1:-1])  # remove "/" at beginning, "\n" at end

        # read labels
        self.label_list: List[Sample] = []  # READ ONLY after initialized
        pbar = tqdm(total=len(self.train_files_path))
        for each in self.train_files_path:
            image_path = self.root_path / each
            label_path = image_path.with_suffix(".lines.txt")
            lines = list(self._read_lines(label_path, mask=carhood_mask))
            self.label_list.append({"image_path": image_path.absolute().as_posix(),
                                    "label_path": label_path.absolute().as_posix(),
                                    "lane_list": lines})
            pbar.update()
        pbar.close()

    def __len__(self):
        return len(self.label_list)  # 88880 CULane train samples

    def __getitem__(self, idx: int) -> Sample:
        """return readable sample data

        :param idx:
        :return:
        """
        label = self.label_list[idx]
        img = cv2.imread(label["image_path"])  # (590, 1640, 3)
        img = img[:, :, ::-1]  # bgr to rgb
        rows, cols = img.shape[:2]

        sample: Sample = {"image_path": label["image_path"],
                          "image": img,
                          "id": idx,
                          "lane_list": deepcopy(label["lane_list"]),
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

        batch_sample = {"image": img,
                        "image_id": image_id_list,  # help tracing source image
                        "detection_gt": detection_gt,
                        "instance_gt": instance_gt,
                        }
        return batch_sample

    @staticmethod
    def get_car_hood_mask():
        mask_y = list(range(430, 600, 10))
        mask_x_min = [718,  660,  610,  567,  531,  503,  478,  453,  434,  412,  386,  361,  342,  321,  295,  272,  249]
        mask_x_max = [1076, 1134, 1175, 1220, 1258, 1307, 1343, 1387, 1425, 1473, 1512, 1555, 1594, 1639, 1639, 1639, 1639]
        carhood_mask = {}  # key=y, value=(xmin, xmax)
        for i in range(len(mask_y)):
            x_min = mask_x_min[i] - 5  # add 5 pixels for margin
            x_max = mask_x_max[i] + 5 if mask_x_max[i] + 5 < 1639 else 1639  # add 5 pixels for margin
            carhood_mask[mask_y[i]] = (x_min, x_max)
        return carhood_mask

    @staticmethod
    def _read_lines(file_path, mask=None):
        with open(file_path) as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            for row in reader:
                row = row[:-1]  # remove a ghost space char
                if mask is None:
                    yield np.array([float(i) for i in row], dtype=float).reshape([-1, 2])
                else:
                    x_list = row[::2]
                    y_list = row[1::2]
                    filtered_list = []
                    for i in range(len(x_list)):
                        y = float(y_list[i])
                        x = float(x_list[i])
                        if y >= 430:
                            if mask[y][0] < x < mask[y][1]:
                                continue
                        filtered_list.append((x, y))
                    if len(filtered_list) > 0:  # skip empty lane list after filtering
                        yield np.array(filtered_list, dtype=float)