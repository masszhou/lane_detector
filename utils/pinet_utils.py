from typing import List

import cv2
from copy import deepcopy
import numpy as np


from configs.PINet.network import NetworkParameters

p = NetworkParameters()
grid_location = np.zeros((p.grid_y, p.grid_x, 2))  # anchor template
for y in range(p.grid_y):
    for x in range(p.grid_x):
        grid_location[y][x][0] = x
        grid_location[y][x][1] = y


###############################################################
##
## visualize
##
###############################################################
def visualize_points(image, x, y):
    image = image
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    for k in range(len(y)):
        for i, j in zip(x[k], y[k]):
            if i > 0:
                image = cv2.circle(image, (int(i), int(j)), 5, p.color[1], -1)

    cv2.imshow("test2", image)
    cv2.waitKey(0)


def visualize_points_origin_size(x, y, test_image, ratio_w, ratio_h):
    color = 0
    image = deepcopy(test_image)
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    image = cv2.resize(image, (int(p.x_size / ratio_w), int(p.y_size / ratio_h)))

    for i, j in zip(x, y):
        color += 1
        for index in range(len(i)):
            cv2.circle(image, (int(i[index]), int(j[index])), 10, p.color[color], -1)
    cv2.imshow("test2", image)
    cv2.waitKey(0)

    return test_image


def visualize_gt(gt_point, gt_instance, ground_angle, image):
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    for y in range(p.grid_y):
        for x in range(p.grid_x):
            if gt_point[0][y][x] > 0:
                xx = int(gt_point[1][y][x] * p.resize_ratio + p.resize_ratio * x)
                yy = int(gt_point[2][y][x] * p.resize_ratio + p.resize_ratio * y)
                image = cv2.circle(image, (xx, yy), 10, p.color[1], -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)


def visualize_regression(image, gt):
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    for i in gt:
        for j in range(p.regression_size):  # gt
            y_value = p.y_size - (p.regression_size - j) * (220 / p.regression_size)
            if i[j] > 0:
                x_value = int(i[j] * p.x_size)
                image = cv2.circle(image, (x_value, y_value), 5, p.color[1], -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 5, p.color[color_index], -1)

    return image


###############################################################
##
## calculate
##
###############################################################
def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        out_x.append((np.array(i) / ratio_w).tolist())
        out_y.append((np.array(j) / ratio_h).tolist())

    return out_x, out_y


def write_result_json(result_data, x, y, testset_index):
    for i in x:
        result_data[testset_index]['lanes'].append(i)
        result_data[testset_index]['run_time'] = 1
    return result_data


############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size / ratio_w
    y_size = p.y_size / ratio_h
    for i, j in zip(x, y):
        min_y = min(j)
        max_y = max(j)
        temp_x = []
        temp_y = []
        for h in target_h:
            temp_y.append(h)
            if h < min_y:
                temp_x.append(-2)
            elif min_y <= h and h <= max_y:
                for k in range(len(j) - 1):
                    if j[k] >= h and h >= j[k + 1]:
                        # linear regression
                        if i[k] < i[k + 1]:
                            temp_x.append(int(i[k + 1] - float(abs(j[k + 1] - h)) * abs(i[k + 1] - i[k]) / abs(
                                j[k + 1] + 0.0001 - j[k])))
                        else:
                            temp_x.append(int(i[k + 1] + float(abs(j[k + 1] - h)) * abs(i[k + 1] - i[k]) / abs(
                                j[k + 1] + 0.0001 - j[k])))
                        break
            else:
                if i[0] < i[1]:
                    l = int(i[1] - float(-j[1] + h) * abs(i[1] - i[0]) / abs(j[1] + 0.0001 - j[0]))
                    if l > x_size or l < 0:
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
                else:
                    l = int(i[1] + float(-j[1] + h) * abs(i[1] - i[0]) / abs(j[1] + 0.0001 - j[0]))
                    if l > x_size or l < 0:
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
        out_x.append(temp_x)
        out_y.append(temp_y)

    return out_x, out_y
