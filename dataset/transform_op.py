#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
import cv2
from typing import Tuple

from dataset.types import Sample


class Resize:
    def __init__(self, rows, cols):
        self.shape = (rows, cols)

    def __call__(self, sample: Sample) -> Sample:
        old_rows, old_cols, old_chs = sample["image"].shape
        sample["image"] = cv2.resize(sample["image"], self.shape[::-1])
        x_ratio = self.shape[1] / old_cols
        y_ratio = self.shape[0] / old_rows

        n_lanes = len(sample["lane_list"])
        for i in range(n_lanes):
            xy_array = sample["lane_list"][i]
            xy_array[:, 0] = xy_array[:, 0] * x_ratio
            xy_array[:, 1] = xy_array[:, 1] * y_ratio
            mask = (xy_array[:, 0] < 0) | (xy_array[:, 0] >= self.shape[1]) | (xy_array[:, 1] < 0) | (xy_array[:, 1] >= self.shape[0])
            sample["lane_list"][i] = xy_array[~mask]  # remove transformed point if outside image boundary
        return sample


class TransposeNumpyArray:
    def __init__(self, order: Tuple[int, int, int]):
        self.order = order

    def __call__(self, sample: Sample) -> Sample:
        sample["image"] = sample["image"].transpose(self.order)
        return sample


class NormalizeInstensity:
    def __init__(self, old_max=255.0, new_max=1.0):
        self.old_max = old_max
        self.new_max = new_max

    def __call__(self, sample: Sample) -> Sample:
        sample["image"] = sample["image"].astype(float) * self.new_max / self.old_max
        return sample
