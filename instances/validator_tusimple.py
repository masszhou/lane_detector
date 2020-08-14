import fire
import torch
import json
from copy import deepcopy
from typing import List
from tqdm import tqdm
from torchvision import transforms

from model import PINet
from instances.trainer import TrainerLaneDetector
from dataset.tusimple import DatasetTusimple
from dataset.transform_op import Resize, TransposeNumpyArray, NormalizeInstensity
from dataset.types import TuSimpleLabel
from utils import convert_to_original_size
from utils import find_target
from utils import write_result_json

from evaluations import LaneEval


def validate_fn(dataset, net, validate_file_name=None, logger=None):
    if validate_file_name is None:
        validate_file_name = "./tmp/validate_result.json"

    result_data: List[TuSimpleLabel] = deepcopy(dataset.label_data)
    # Fixme: hard code
    target_h = list(range(160, 720, 10))
    x_size = 512.0
    y_size = 256.0

    net.evaluate_mode()

    pbar = tqdm(total=len(result_data))
    for testset_index, sample in enumerate(dataset):
        ratio_w = x_size / sample["original_size"][1]
        ratio_h = y_size / sample["original_size"][0]
        x, y, img = net.test_on_image(sample["image"], threshold_confidence=0.81)
        x, y = convert_to_original_size(x[0], y[0], ratio_w, ratio_h)
        x, y = find_target(x, y, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x, y, testset_index)
        pbar.set_description(f'test image id {testset_index}')
        pbar.update()
    pbar.close()

    with open(validate_file_name, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

    metrics = LaneEval.bench_one_submit(validate_file_name, "./data/test_label.json")

    if logger is not None:
        logger.log_metric("validate accuracy", metrics[0]["value"])
        logger.log_metric("validate FP", metrics[1]["value"])
        logger.log_metric("validate FN", metrics[2]["value"])

    return metrics

