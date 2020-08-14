from typing import List, Tuple

import cv2
import torch
from copy import deepcopy
import numpy as np
import math

from configs.PINet.network import NetworkParameters


color_palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                 (255, 255, 255), (100, 255, 0), (100, 0, 255), (255, 100, 0), (0, 100, 255), (255, 0, 100),
                 (0, 255, 100)]
net_params = NetworkParameters()
grid_location = np.zeros((net_params.grid_y, net_params.grid_x, 2))  # anchor template
for y in range(net_params.grid_y):
    for x in range(net_params.grid_x):
        grid_location[y][x][0] = x
        grid_location[y][x][1] = y


def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 3, color_palette[color_index])

    return image


def get_num_along_point(x, y, point1, point2):  # point1 : source
    x = np.array(x)
    y = np.array(y)

    x = x[y < point1[1]]
    y = y[y < point1[1]]

    dis = np.sqrt((x - point1[0]) ** 2 + (y - point1[1]) ** 2)

    count = 0
    shortest = 1000
    target_angle = get_angle_two_points(point1, point2)
    for i in range(len(dis)):
        angle = get_angle_two_points(point1, (x[i], y[i]))
        diff_angle = abs(angle - target_angle)
        distance = dis[i] * math.sin(diff_angle * math.pi * 2)
        if distance <= 12:
            count += 1
            if distance < shortest:
                shortest = distance

    return count, shortest


def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y < point[1]]
    y = y[y < point[1]]

    dis = (x - point[0]) ** 2 + (y - point[1]) ** 2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i, j))

    return points


def get_angle_two_points(p1, p2):
    del_x = p2[0] - p1[0]
    del_y = p2[1] - p1[1] + 0.000001
    if p2[0] >= p1[0] and p2[1] > p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta /= 360.0
    elif p2[0] > p1[0] and p2[1] <= p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta += 180
        theta /= 360.0
    elif p2[0] <= p1[0] and p2[1] < p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta += 180
        theta /= 360.0
    elif p2[0] < p1[0] and p2[1] >= p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta += 360
        theta /= 360.0

    return theta


def get_angle_on_lane(x, y):
    sorted_x = None
    sorted_y = None
    angle = []

    # sort
    ind = np.argsort(y, axis=0)
    sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
    sorted_y = np.take_along_axis(y, ind[::-1], axis=0)

    # calculate angle
    length = len(x)
    theta = -2
    for i in range(length - 1):
        if sorted_x[i] < 0:
            angle.append(-2)
        else:
            p1 = (sorted_x[i], sorted_y[i])
            for index, j in enumerate(sorted_x[i + 1:]):
                if j > 0:
                    p2 = (sorted_x[i + 1 + index], sorted_y[i + 1 + index])
                    break
                else:
                    p2 = (-2, -2)
            if p2[0] < 0:
                angle.append(theta)
                continue
            theta = get_angle_two_points(p1, p2)
            angle.append(theta)
    angle.append(theta)

    return angle


def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())

    return out_x, out_y


def sort_along_x(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(i, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())

    return out_x, out_y


def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)

    return out_x, out_y


############################################################################
#
# post processing for eliminating outliers
#
############################################################################
def eliminate_out(sorted_x, sorted_y):
    out_x = []
    out_y = []

    for lane_x, lane_y in zip(sorted_x, sorted_y):

        lane_x_along_y = np.array(deepcopy(lane_x))
        lane_y_along_y = np.array(deepcopy(lane_y))

        ind = np.argsort(lane_x_along_y, axis=0)
        lane_x_along_x = np.take_along_axis(lane_x_along_y, ind, axis=0)
        lane_y_along_x = np.take_along_axis(lane_y_along_y, ind, axis=0)

        if lane_y_along_x[0] > lane_y_along_x[-1]:  # if y of left-end point is higher than right-end
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]),
                               (lane_x_along_y[2], lane_y_along_y[2]),
                               (lane_x_along_x[0], lane_y_along_x[0]), (lane_x_along_x[1], lane_y_along_x[1]),
                               (lane_x_along_x[2], lane_y_along_x[2])]  # some low y, some left/right x
        else:
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]),
                               (lane_x_along_y[2], lane_y_along_y[2]),
                               (lane_x_along_x[-1], lane_y_along_x[-1]), (lane_x_along_x[-2], lane_y_along_x[-2]),
                               (lane_x_along_x[-3], lane_y_along_x[-3])]  # some low y, some left/right x

        temp_x = []
        temp_y = []
        for start_point in starting_points:
            temp_lane_x, temp_lane_y = generate_cluster(start_point, lane_x, lane_y)
            temp_x.append(temp_lane_x)
            temp_y.append(temp_lane_y)

        max_lenght_x = None
        max_lenght_y = None
        max_lenght = 0
        for i, j in zip(temp_x, temp_y):
            if len(i) > max_lenght:
                max_lenght = len(i)
                max_lenght_x = i
                max_lenght_y = j
        out_x.append(max_lenght_x)
        out_y.append(max_lenght_y)

    return out_x, out_y


############################################################################
## generate cluster
############################################################################
def generate_cluster(start_point, lane_x, lane_y):
    cluster_x = [start_point[0]]
    cluster_y = [start_point[1]]

    point = start_point
    while True:
        points = get_closest_upper_point(lane_x, lane_y, point, 3)

        max_num = -1
        max_point = None

        if len(points) == 0:
            break
        if len(points) < 3:
            for i in points:
                cluster_x.append(i[0])
                cluster_y.append(i[1])
            break
        for i in points:
            num, shortest = get_num_along_point(lane_x, lane_y, point, i)
            if max_num < num:
                max_num = num
                max_point = i

        total_remain = len(np.array(lane_y)[np.array(lane_y) < point[1]])
        cluster_x.append(max_point[0])
        cluster_y.append(max_point[1])
        point = max_point

        if len(points) == 1 or max_num < total_remain / 5:
            break

    return cluster_x, cluster_y


############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i) > 2:
            out_x.append(i)
            out_y.append(j)
    return out_x, out_y


@torch.jit.script
def eliminate_fewer_points_jit(ixy: torch.Tensor, min_pts_threshold: int) -> torch.Tensor:
    # eliminate fewer points
    # for each_id in torch.unique(ixy[:, 0]):  # error for jit, Unknown builtin op: aten::unique.
    max_id = torch.max(ixy[:, 0])
    for each_id in torch.range(0, max_id, dtype=torch.int32):
        mask = (ixy[:, 0] == each_id)
        n_pts = mask.sum()
        if n_pts < min_pts_threshold:
            ixy[:, 0:3][mask] = torch.tensor(0, dtype=torch.int32)  # set i, x, y to 0
    return ixy


############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets, instance, threshold_confidence):
    mask = confidance > threshold_confidence  # [32, 64]

    # grid_location -> [32, 64, 2]
    grid = grid_location[mask]  # e.g. [107,2]
    offset = offsets[mask]        # e.g. [107,2]
    feature = instance[mask]      # e.g. [107,4]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        point_x = int((offset[i][0] + grid[i][0]) * net_params.resize_ratio)
        point_y = int((offset[i][1] + grid[i][1]) * net_params.resize_ratio)
        if point_x > net_params.x_size or point_x < 0 or point_y > net_params.y_size or point_y < 0:
            continue
        if len(lane_feature) == 0:
            lane_feature.append(feature[i])  # list[np.array shape=(4)]
            x.append([])
            x[0].append(point_x)
            y.append([])
            y[0].append(point_y)
        else:
            flag = 0  # flag = 0, if lane is not counted yet
            index = 0  # lane index
            for feature_idx, j in enumerate(lane_feature):
                index += 1
                if index >= 12:
                    index = 12
                if np.linalg.norm((feature[i] - j) ** 2) <= 0.22:  # p.threshold_instance
                    lane_feature[feature_idx] = (j * len(x[index - 1]) + feature[i]) / (len(x[index - 1]) + 1)
                    x[index - 1].append(point_x)
                    y[index - 1].append(point_y)
                    flag = 1
                    break
            if flag == 0:
                lane_feature.append(feature[i])
                x.append([])
                x[index].append(point_x)
                y.append([])
                y[index].append(point_y)

    return x, y


def lane_index(ls):
    return ls[0]


@torch.jit.script
def generate_result_jit(confidance: torch.Tensor,
                        offsets: torch.Tensor,
                        instance: torch.Tensor,
                        grid_template: torch.Tensor):
    """
    Note: torch.jit.script compiler need to know exactly data type, some types can not be automatically deduced,
    thus MUST have type hints
    """
    # Fixme, now MUST define here due to jit
    x_size = 512
    y_size = 256
    resize_ratio = 8
    threshold_point = 0.81
    threshold_instance = 0.22
    max_output_pts = 400

    mask = confidance > threshold_point
    lane_feature: List[torch.Tensor] = []
    x: List[List[int]] = []  # MUST use type hint for compiling torch.script
    y: List[List[int]] = []
    ixy = torch.zeros([max_output_pts, 3], dtype=torch.int32)

    # moved to inference instance
    # x_array = torch.arange(grid_x, dtype=torch.float, device="cpu").view([1, grid_x])
    # x_array = x_array.expand(grid_y, grid_x)
    # y_array = torch.arange(grid_y, dtype=torch.float, device="cpu").view([grid_y, 1])
    # y_array = y_array.expand(grid_y, grid_x)
    # grid = torch.stack([x_array, y_array])
    # grid = grid.permute([1, 2, 0])
    grid = grid_template[mask]

    offset = offsets[mask]
    feature = instance[mask]
    counter = 0
    for i in range(grid.shape[0]):
        point_x = (offset[i][0] + grid[i][0]) * resize_ratio
        point_y = (offset[i][1] + grid[i][1]) * resize_ratio
        point_x = int(point_x.item())
        point_y = int(point_y.item())

        # ---------------- >>>>>>>>>>>>>>>>>>>
        # this code block slow down jit model in cpp inference significantly
        # each IF statement will double the cpp inference time from jit model
        if point_x > x_size or point_x < 0 or point_y > y_size or point_y < 0:
            continue
        if len(lane_feature) == 0:
            lane_feature.append(feature[i])
            x_list: List[int] = []
            x.append(x_list)
            x[0].append(point_x)
            y_list: List[int] = []
            y.append(y_list)
            y[0].append(point_y)
            ixy[counter] = torch.tensor([0, point_x, point_y], dtype=torch.int32)
            counter += 1
        else:
            flag = 0
            index = 0
            for feature_idx, j in enumerate(lane_feature):
                index += 1
                if index >= 12:
                    index = 12
                if torch.norm(feature[i] - j) <= threshold_instance:
                    # update mean feature value for each cluster
                    lane_feature[feature_idx] = (j * len(x[index - 1]) + feature[i]) / (len(x[index - 1]) + 1)
                    x[index - 1].append(point_x)
                    y[index - 1].append(point_y)
                    if counter < max_output_pts:
                        ixy[counter] = torch.tensor([index - 1, point_x, point_y], dtype=torch.int32)
                        counter += 1
                    flag = 1
                    break
            if flag == 0:
                lane_feature.append(feature[i])
                x_list: List[int] = []
                x.append(x_list)
                x[index].append(point_x)
                y_list: List[int] = []
                y.append(y_list)
                y[index].append(point_y)
                if counter < max_output_pts:
                    ixy[counter] = torch.tensor([index, point_x, point_y], dtype=torch.int32)
                    counter += 1
        # <<<<<<<<<<<<<<<<<<< ---------------------

    return ixy[ixy[:, 0].sort()[1]]

