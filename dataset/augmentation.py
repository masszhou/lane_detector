#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################
import numpy as np
import cv2

from dataset.types import Sample


class Flip:
    def __init__(self, prob_to_apply=0.5):
        self.prob = prob_to_apply

    def __call__(self, sample: Sample) -> Sample:
        if np.random.uniform() < self.prob:
            sample["image"] = cv2.flip(sample["image"], 1)
            rows, cols, chs = sample["image"].shape
            for xy_array in sample["lane_list"]:
                xy_array[:, 0] = cols - xy_array[:, 0]
        return sample


class Translate:
    def __init__(self, prob_to_apply=0.5, tx_min=-50, tx_max=50, ty_min=-30, ty_max=30):
        self.prob = prob_to_apply
        self.tx = (tx_min, tx_max)
        self.ty = (ty_min, ty_max)

    def __call__(self, sample: Sample) -> Sample:
        if np.random.uniform() < self.prob:
            tx = np.random.randint(*self.tx)
            ty = np.random.randint(*self.ty)
            rows, cols, chs = sample["image"].shape
            sample["image"] = cv2.warpAffine(sample["image"],
                                             np.float32([[1, 0, tx], [0, 1, ty]]),
                                             (cols, rows))

            n_lanes = len(sample["lane_list"])
            for i in range(n_lanes):
                xy_array = sample["lane_list"][i]
                xy_array[:, 0] = xy_array[:, 0] + tx
                xy_array[:, 1] = xy_array[:, 1] + ty
                mask = (xy_array[:, 0] < 0) | (xy_array[:, 0] >= cols) | (xy_array[:, 1] < 0) | (xy_array[:, 1] >= rows)
                sample["lane_list"][i] = xy_array[~mask]  # remove transformed point if outside image boundary

        return sample


class Rotate:
    def __init__(self, prob_to_apply=0.5, angle_min=-10, angle_max=10):
        """
        :param prob_to_apply: from [0.0, 1.0)
        :param angle_min: [deg]
        :param angle_max: [deg]
        """
        self.prob = prob_to_apply
        self.angle = (angle_min, angle_max)

    def __call__(self, sample: Sample) -> Sample:
        if np.random.uniform() < self.prob:
            rows, cols, chs = sample["image"].shape
            angle = np.random.randint(*self.angle)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            sample["image"] = cv2.warpAffine(sample["image"], M, (cols, rows))

            n_lanes = len(sample["lane_list"])
            for i in range(n_lanes):
                xy_array = sample["lane_list"][i]
                xy_array = np.dot(xy_array, M.T[:2, :2]) + M.T[-1, :]
                mask = (xy_array[:, 0] < 0) | (xy_array[:, 0] >= cols) | (xy_array[:, 1] < 0) | (xy_array[:, 1] >= rows)
                sample["lane_list"][i] = xy_array[~mask]  # remove transformed point if outside image boundary
        return sample


class AddGaussianNoise:
    def __init__(self, prob_to_apply=0.5):
        self.prob = prob_to_apply
        self.mean = (0, 0, 0)
        self.stddev = (20, 20, 20)

    def __call__(self, sample: Sample) -> Sample:
        if np.random.uniform() < self.prob:
            noise = np.zeros_like(sample["image"], dtype=np.uint8)
            cv2.randn(noise, self.mean, self.stddev)
            sample["image"] = sample["image"] + noise
        return sample


class ChangeIntensity:
    def __init__(self, prob_to_apply=0.5):
        self.prob = prob_to_apply
        self.range = (-60.0, 60.0)

    def __call__(self, sample: Sample) -> Sample:
        if np.random.uniform() < self.prob:
            hsv = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(np.random.uniform(*self.range))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1 * value
                v[v < lim] = 0
                v[v >= lim] -= lim
            final_hsv = cv2.merge((h, s, v))
            sample["image"] = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return sample


# class AddShadow:
#     def __init__(self, prob_to_apply=0.5, alpha_min=0.5, alpha_max=0.75):
#         self.prob = prob_to_apply
#         self.alpha = (alpha_min, alpha_max)
#
#     def __call__(self, sample: SampleBDD100K) -> SampleBDD100K:
#         if np.random.uniform() < self.prob:
#             rows, cols, chs = sample["image"].shape
#             coin = np.random.randint(2)
#             top_x, bottom_x = np.random.randint(0, 512, 2)
#             shadow_img = sample["image"].copy()
#             if coin == 0:
#                 rand = np.random.randint(2)
#                 vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
#                 if rand == 0:
#                     vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
#                 elif rand == 1:
#                     vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
#                 mask = sample["image"].copy()
#                 channel_count = sample["image"].shape[2]  # i.e. 3 or 4 depending on your image
#                 ignore_mask_color = (0,) * channel_count
#                 cv2.fillPoly(mask, [vertices], ignore_mask_color)
#                 rand_alpha = np.random.uniform(*self.alpha)
#                 cv2.addWeighted(mask, rand_alpha, sample["image"], 1 - rand_alpha, 0., shadow_img)
#                 sample["image"] = shadow_img
#         return sample
