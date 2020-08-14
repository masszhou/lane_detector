#########################################################################
## Structure of network.
#########################################################################
import torch
import torch.nn as nn
from model.block import ResizeLayer
from model.block import HourglassPINet


class PINet(nn.Module):
    def __init__(self):
        super(PINet, self).__init__()

        # resizing = ResizeLayer(3, 128)
        # self.resizing = torch.jit.trace(resizing, (torch.rand(64, 3, 7, 7),))
        self.resizing = ResizeLayer(3, 128)
        self.layer1 = HourglassPINet(128, 128)
        self.layer2 = HourglassPINet(128, 128)

    def forward(self, inputs: torch.Tensor):
        # series connect 2x hourglass blocks,
        # each block has an independent output for intermediate supervision during training phase
        # output of last hourglass block is network output
        out = self.resizing(inputs)
        result1, out = self.layer1(out)
        result2, out = self.layer2(out)
        return [result1, result2]


class PINetJIT(nn.Module):
    """
    1. parameters in PINetJIT are identical to PINet.
    2. the only difference is output only for inference, no intermediate loss.
    3. PINetJIT is compatible to torch.jit compiler
    """
    def __init__(self):
        super(PINetJIT, self).__init__()

        # resizing = ResizeLayer(3, 128)
        # self.resizing = torch.jit.trace(resizing, (torch.rand(64, 3, 7, 7),))
        self.resizing = ResizeLayer(3, 128)
        self.layer1 = HourglassPINet(128, 128)
        self.layer2 = HourglassPINet(128, 128)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: NCHW = [batch_size, 3,  256, 512]
        :return: Tuple[Tensor], (out_confidence, out_offset, out_instance)
            out_confidence -> [#batch, 1, 32, 64]
            out_offset -> [#batch, 2, 32, 64]
            out_instance -> [#batch, 4, 32, 64]
        """
        # series connect 2x hourglass blocks,
        # each block has an independent output for intermediate supervision during training phase
        # output of last hourglass block is network output
        out = self.resizing(inputs)
        result1, out = self.layer1(out)
        result2, out = self.layer2(out)

        # Only tensors, lists and tuples of tensors can be output from traced functions
        # return [result1, result2]
        return result2