#########################################################################
#  2020
#  Author: Zhiliang Zhou
# pipeline template
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#########################################################################

import cv2
import copy
import numpy as np
import json
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import random
from copy import deepcopy

from dataset.types import BDD100KLabel, Sample, BatchSample


def one_bezier_curve(a, b, t):
    return (1 - t) * a + t * b


def n_bezier_curve(xs, n, k, t):
    if n == 1:
        return one_bezier_curve(xs[k], xs[k + 1], t)
    else:
        return (1 - t) * n_bezier_curve(xs, n - 1, k, t) + t * n_bezier_curve(xs, n - 1, k + 1, t)


def bezier_curve(xs, ys, num):
    bezier_pts = []
    n = len(xs) - 1
    t_step = 1.0 / (num - 1)
    t = np.arange(0.0, 1 + np.finfo(np.float32).eps, t_step)
    for each in t:
        x = n_bezier_curve(xs, n, 0, each)
        y = n_bezier_curve(ys, n, 0, each)
        bezier_pts.append((x, y))
    return np.array(bezier_pts)


# def test_bezier_curve():
#     xs = [0, 2, 5, 10, 15, 20]
#     ys = [0, 6, 10, 0, 5, 5]
#     num = 10
#     pts = bezier_curve(xs, ys, num)
#     plt.plot(pts[:, 0], pts[:, 1])
#     plt.plot(xs, ys)
#     plt.show()


class DatasetBDD100K(Dataset):
    """
    # Definitions
    # * laneType
    #   * 0: unknown
    #   * 1: road curb            157401 in pixel?  Bordstein auf deutsch
    #   * 2: double white         8222
    #   * 2: double yellow        53426
    #   * 4: double other         37
    #   * 5: single white         353288
    #   * 6: single yellow        28993
    #   * 7: single other         394
    #   * 8: crosswalk            154108  ---ignored !---
    # * laneDirection
    #   * 0: unknown
    #   * 1: parallel             562157
    #   * 2: vertical             193712
    # * LaneStyle
    #   * 0: unknown
    #   * 1: solid                 755869
    #   * 2: dashed               91626
    """
    def __init__(self, root_path, json_files, transform=None):
        self.root_path = root_path
        self.seg_label_path = root_path + "seg/color_labels/train/"
        self.img_root_path = root_path + "images/100k/train/"
        self.transform = transform
        if isinstance(json_files, str):
            self.json_files = [json_files]
        else:
            self.json_files = json_files
        self.transform = transform

        for each_json in self.json_files:
            with open(root_path + each_json, "r") as train_file:
                self.train_data: List[Dict] = json.load(train_file)  # "name", "attributes", "timestamp", "labels"

        self.use_random_sequence = False
        self.random_sequence = list(range(len(self.train_data)))  # need random.shuffle(self.random_sequence)

    def __len__(self):
        return len(self.train_data)  # e.g. 69863

    def __getitem__(self, idx: int) -> Sample:
        """return readable sample data

        :param idx:
        :return:
        """
        if self.use_random_sequence is True:
            use_idx = self.random_sequence[idx]
        else:
            use_idx = idx
        label = self.train_data[use_idx]
        img_path = self.img_root_path + label["name"]
        img_bgr = cv2.imread(img_path)
        rows, cols = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1]
        lane_label = self.get_lane_label(label)
        sample: Sample = {"image_path": img_path,
                          "image": img_rgb,
                          "id": idx,
                          "lane_list": lane_label["lane_list"],
                          "lane_type": lane_label["lane_type"],
                          "lane_direction": lane_label["lane_direction"],
                          "lane_style": lane_label["lane_style"],
                          "original_size": (rows, cols)}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def shuffle(self):
        random.shuffle(self.random_sequence)

    @staticmethod
    def collate_fn(batch: List[Sample]) -> BatchSample:
        """convert readable sample data as aligned ground truth data

        Ground truth data is used for loss calculation

        :param batch: List[TrainingSample]
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

        samples: BatchSample = {"image": img,
                                "image_id": image_id_list,  # help tracing source image
                                "detection_gt": detection_gt,
                                "instance_gt": instance_gt, }
        return samples

    def get_lane_label(self, label_row, bezier_seg_size=20) -> BDD100KLabel:
        labels = label_row["labels"]
        lanes = self.get_lanes(labels)
        filtered_lanes = []
        duplicated_lanes = []

        for i in range(len(lanes)):
            lane_a = lanes[i]
            if lane_a in duplicated_lanes:  # if lane is used, then skip
                continue

            if lane_a['attributes']['laneDirection'] == 'vertical':  # if lane is vertical, then skip
                continue

            # else check if this lane has duplicate one
            lane_merge = None
            for j in range(i + 1, len(lanes)):
                lane_b = lanes[j]
                if lane_a['attributes']['laneStyle'] != lane_b['attributes']['laneStyle']:
                    continue
                elif lane_a['attributes']['laneType'] != lane_b['attributes']['laneType']:
                    continue
                elif lane_a['attributes']['laneType'] == "double yellow":
                    continue
                elif lane_a['attributes']['laneType'] == "double white":
                    continue
                elif lane_a['attributes']['laneDirection'] == 'vertical':
                    continue
                elif len(lane_a['poly2d'][0]['vertices']) != len(lane_b['poly2d'][0]['vertices']):
                    continue
                if lane_merge is not None:
                    continue
                bottom_x, bottom_y, top_x, top_y = self.get_dist_of_lanes(lane_a, lane_b)
                if bottom_x < 112 and bottom_y < 30:
                    # merge lanes when found the same lane
                    lane_merge = self.merge_lanes(lane_a, lane_b)
                    duplicated_lanes.append(lane_b)  # mark b as used
            if lane_merge is None:
                filtered_lanes.append(lane_a)
            else:
                filtered_lanes.append(lane_merge)

        lane_list = []
        lane_type = []
        lane_direction = []
        lane_style = []
        for each_label in filtered_lanes:
            pts_list = each_label["poly2d"][0]['vertices']
            # compare the first and last element, and start lane always from the bottom
            # can not use sort, since lane can be not monotonic
            if pts_list[0][1] < pts_list[-1][1]:
                pts_list = pts_list[::-1]
            x_list, y_list = map(list, zip(*pts_list))
            dist = self.get_dist_of_pts(pts_list[0], pts_list[-1])
            pts_num = int(dist / bezier_seg_size) if int(dist / bezier_seg_size) > 2 else 2
            bezier_pts = bezier_curve(x_list, y_list, num=pts_num)

            lane_list.append(bezier_pts)
            lane_type.append(self.get_lane_type(each_label["attributes"]["laneType"]))
            lane_direction.append(self.get_lane_direction(each_label["attributes"]["laneDirection"]))
            lane_style.append(self.get_lane_style(each_label["attributes"]["laneStyle"]))

        lane_label_bdd100k: BDD100KLabel = {"raw_file": label_row["name"],
                                            "lane_list": lane_list,
                                            "lane_type": lane_type,
                                            "lane_direction": lane_direction,
                                            "lane_style": lane_style}
        return lane_label_bdd100k

    @staticmethod
    def get_lanes(label_list):
        return [label for label in label_list if 'poly2d' in label and label['category'][:4] == 'lane']

    @staticmethod
    def get_dist_of_lanes(lane_a, lane_b):
        pt1 = lane_a['poly2d'][0]['vertices'][0]  # begin pt
        pt2 = lane_a['poly2d'][0]['vertices'][-1]  # end pt
        if pt1[1] > pt2[1]:
            a_bottom_x, a_top_x = pt1[0], pt2[0]
            a_bottom_y, a_top_y = pt1[1], pt2[1]
        else:
            a_bottom_x, a_top_x = pt2[0], pt1[0]
            a_bottom_y, a_top_y = pt2[1], pt1[1]

        pt3 = lane_b['poly2d'][0]['vertices'][0]  # begin pt
        pt4 = lane_b['poly2d'][0]['vertices'][-1]  # end pt
        if pt3[1] > pt4[1]:
            b_bottom_x, b_top_x = pt3[0], pt4[0]
            b_bottom_y, b_top_y = pt3[1], pt4[1]
        else:
            b_bottom_x, b_top_x = pt4[0], pt3[0]
            b_bottom_y, b_top_y = pt4[1], pt3[1]

        return abs(a_bottom_x - b_bottom_x), abs(a_bottom_y - b_bottom_y), abs(a_top_x - b_top_x), abs(
            a_top_y - b_top_y)

    @staticmethod
    def get_dist_of_pts(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

    @staticmethod
    def merge_lanes(lane_a, lane_b):
        """
        only merge two lane with same number of key points
        :param lane_a:
        :param lane_b:
        :return:
        """
        pts_a = lane_a['poly2d'][0]['vertices']  # list of points
        pts_b = lane_b['poly2d'][0]['vertices']

        # align lane_a and lane_b
        dist_1 = (pts_a[0][0] - pts_b[0][0]) ** 2 + (pts_a[0][1] - pts_b[0][1]) ** 2
        dist_2 = (pts_a[0][0] - pts_b[-1][0]) ** 2 + (pts_a[0][1] - pts_b[-1][1]) ** 2
        if dist_1 > dist_2:
            pts_b = pts_b[::-1]

        pts_merge = [[(pt_a[0] + pt_b[0]) / 2, (pt_a[1] + pt_b[1]) / 2] for pt_a, pt_b in zip(pts_a, pts_b)]
        lane_c = copy.deepcopy(lane_a)  # must deep copy
        lane_c['poly2d'][0]['vertices'] = pts_merge
        return lane_c

    @staticmethod
    def get_lane_type(input_str: str) -> Optional[int]:
        type_dict = {
            "unknown": 0,
            "road curb": 1,  # 157401 in pixel?  Bordstein auf deutsch
            "double white": 2,  # 8222
            "double yellow": 3,  # 53426
            "double other": 4,  # 37
            "single white": 5,  # 353288
            "single yellow": 6,  # 28993
            "single other": 7,  # 394
            "crosswalk": 8,  # 154108
        }
        return type_dict.get(input_str)

    @staticmethod
    def get_lane_direction(input_str: str) -> Optional[int]:
        direction_dict = {
            "unknown": 0,
            "parallel": 1,  # 562157
            "vertical": 2,  # 193712
        }
        return direction_dict.get(input_str)

    @staticmethod
    def get_lane_style(input_str: str) -> Optional[int]:
        style_dict = {
            "unknown": 0,
            "solid": 1,  # 755869
            "dashed": 2,  # 91626
        }
        return style_dict.get(input_str)

