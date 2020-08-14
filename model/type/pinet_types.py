from typing import NamedTuple  # since TypedDict only available >3.8
import torch


class TypeHourglassOut(NamedTuple):
    confidence: torch.Tensor  # [n, 1, 32, 64] 32, 64 are depends on grid setup
    offset: torch.Tensor  # [n, 2, 32, 64] 2 is offset_x and offset_y
    instance: torch.Tensor  # [n, 4, 32, 64], similarity matrix use dim=4 feature vector, more in Similarity Group Proposal Network
