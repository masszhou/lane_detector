#########################################################################
#  2020
#  Author: Zhiliang Zhou
#########################################################################

import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
from typing import List, Tuple
import numpy as np


class Sample(TypedDict, total=False):
    image_path: str
    label_path: str
    image: np.ndarray
    id: int  # global index of image in dataset
    set_id: int
    lane_list: List[np.ndarray]   # np.ndarray shape = [n, 2]  ((x,y),...)
    lane_type: List[int]
    lane_direction: List[int]
    lane_style: List[int]
    original_size: Tuple[int, int]  # rows, cols


class BatchSample(TypedDict, total=False):
    image: np.ndarray
    image_id: List[int]
    set_id: List[int]
    detection_gt: np.ndarray
    instance_gt: np.ndarray


class TuSimpleLabel(TypedDict, total=True):
    lanes: List[List[int]]  # in column
    h_samples: List[int]    # in row
    raw_file: str

# e.g
# {
#   "lanes": [  // e.g. 4 x lanes
#         [-2, -2, -2, -2, 632, 625, 617, ... , 330, 322, 314, 307, 299],  // Column pos for marks, -2 means no mark
#         [-2, -2, -2, -2, 719, 734, 748, ... , -2,   -2,  -2,  -2,  -2],
#         [-2, -2, -2, -2, -2, 532, 503,  ... , -2,   -2,  -2,  -2,  -2],
#         [-2, -2, -2, 781, 822, 862,     ... , -2,   -2,  -2,  -2,  -2]
#        ],
#   "h_samples": [240, 250, 260, 270, ... , 630, 640, 650, 660, 670, 680, 690, 700, 710],  // row pos for marks
#   "raw_file": "path_to_clip"
# }


class BDD100KLabel(TypedDict, total=True):
    raw_file: str
    lane_list: List[np.ndarray]  # np.ndarray shape = [n, 2]  ((x,y),...)
    lane_type: List[int]
    lane_direction: List[int]
    lane_style: List[int]
