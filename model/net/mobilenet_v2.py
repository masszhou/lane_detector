""" Mobilenet V2
Author: Zhiliang Zhou

paper:
    https://arxiv.org/pdf/1801.04381.pdf

blog:
    https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html

official tf implementation:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py

reference pytorch implementation:
    https://github.com/Randl/MobileNetV2-pytorch/blob/master/model.py
    https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InvertedResidual(nn.Module):
    """
    Builds a composite convolution that has the following structure
    expansion (1x1) -> depthwise (kernel_size) -> compress/projection (1x1)

    compared with traditional residual block, which is
    compress (1x1) -> conv2d (3x3) -> expansion (1x1)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_factor: float,
                 stride: int,
                 padding=1):
        super(InvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(in_channels*expansion_factor)  # int

        # why in tonylins' implementation, when expansion_factor=1, there is no this 1x1 layer ? not the same with paper
        # tonylins claims https://github.com/tonylins/pytorch-mobilenet-v2/issues/28
        # but I don't agree, after reading the official tf implementation
        # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/conv_blocks.py

        # BUT, due to the model in torchvision.models.mobilenet_v2(), tonylins is CORRECT
        # when expansion_factor=1 there is no first pointwise expansion

        # pointwise expansion
        if expansion_factor != 1:
            self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)

        # depth-wise, groups=hidden_dim
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # pointwise projection/compression
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x: "torch.Tensor, rank=4, order=NCHW") -> "[torch.Tensor]":
        residual = x

        if self.expansion_factor == 1:
            out = self.conv2(x)
            out = self.bn2(out)
            out = self.activation(out)

            out = self.conv3(out)
            out = self.bn3(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.activation(out)

            out = self.conv3(out)
            out = self.bn3(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            return residual + out
        else:
            return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class MobileNetV2(nn.Module):
    """
    feature layers
    | id | Input      | Operator   | t | c    | n | s |
    | -- | ---------  |------------|---|------|---|---|
    | 0  | 224x224x3  | conv2d     | - | 32   | 1 | 2 |
    | 1  | 112x112x32 | bottleneck | 1 | 16   | 1 | 1 |
    | 2  | 112x112x16 | bottleneck | 6 | 24   | 2 | 2 |
    | 3  | 56x56x24   | bottleneck | 6 | 32   | 3 | 2 |
    | 4  | 28x28x32   | bottleneck | 6 | 64   | 4 | 2 |
    | 5  | 14x14x64   | bottleneck | 6 | 96   | 3 | 1 |
    | 6  | 14x14x96   | bottleneck | 6 | 160  | 3 | 2 |
    | 7  | 7x7x160    | bottleneck | 6 | 320  | 1 | 1 |
    | 8  | 7x7x320    | conv2d 1x1 | - | 1280 | 1 | 1 |

    classification layers
    | 9  | 7x7x1280   | avgpool 7x7| - | -    | 1 | - |
    | 10 | 1x1x1280   | conv2d 1x1 | - | k    | - | - |

    t -> expansion factor
    c -> output channels
    n -> repeat n times
    s -> stride
    c needs to be divisible by 8,
    """
    def __init__(self, n_class):
        super(MobileNetV2, self).__init__()
        # assert input_size % 32 == 0

        self.bottlenet_setting = [
            # t, c, n, s
            [1, 16,  1, 1],
            [6, 24,  2, 2],
            [6, 32,  3, 2],
            [6, 64,  4, 2],
            [6, 96,  3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # build features layers
        self.layer0 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 3, 3, 1, bias=False)),
            ("bn1", nn.BatchNorm2d(32)),
            ("relu6", nn.ReLU6(inplace=True))
        ]))

        self.layer1 = self._build_bottleneck(32,  16,  1, 1, 1, name="Bottleneck1")
        self.layer2 = self._build_bottleneck(16,  24,  6, 2, 2, name="Bottleneck2")
        self.layer3 = self._build_bottleneck(24,  32,  6, 3, 2, name="Bottleneck3")
        self.layer4 = self._build_bottleneck(32,  64,  6, 4, 2, name="Bottleneck4")
        self.layer5 = self._build_bottleneck(64,  96,  6, 3, 1, name="Bottleneck5")
        self.layer6 = self._build_bottleneck(96,  160, 6, 3, 2, name="Bottleneck6")
        self.layer7 = self._build_bottleneck(160, 320, 6, 1, 1, name="Bottleneck7")

        self.layer8 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(320, 1280, 1, 1, 0, bias=False)),
            ("bn1", nn.BatchNorm2d(1280)),
            ("relu6", nn.ReLU6(inplace=True))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d(1)),  # target output size of 1x1
            ("dropout", nn.Dropout(0.5)),
            ("flatten", Flatten()),
            ("linear", nn.Linear(1280, n_class)),
        ]))

        self.apply(init_weight)

    @staticmethod
    def _build_bottleneck(in_c, out_c, t, n, s, name=None):
        modules = OrderedDict()
        if name is None:
            name = "Bottleneck"

        for i in range(n):
            if i == 0:
                block = InvertedResidual(in_channels=in_c, out_channels=out_c, expansion_factor=t, stride=s)
            else:
                block = InvertedResidual(in_channels=out_c, out_channels=out_c, expansion_factor=t, stride=1)
            modules[name+"_{}".format(i)] = block

        return nn.Sequential(modules)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = self.classifier(out)

        return out


def load_imagenet_weights(model_state_dict, device):
    """
    model trained by Evgeniy Zheltonozhskiy, github user: Randl
    https://github.com/Randl/MobileNetV2-pytorch
    """
    ckpt = torch.load("./saved_weights/mobilenet_v2_1.0_224/model_best.pth.tar", map_location=device)

    weights = ckpt["state_dict"]
    # there are 267 items in weights,
    # in BN there is no num_batches_tracked
    # in pytorch 1.0 every BN item has num_batches_tracked, and there is 319 items totally
    # be careful the difference, 53 num_batches_tracked items
    counter = 0
    keyname_list = list(weights.keys())
    for i, keyname in enumerate(model_state_dict.keys()):
        if keyname.split(".")[-1] == "num_batches_tracked":
            continue
        model_state_dict[keyname].copy_(weights[keyname_list[counter]])
        counter += 1


if __name__ == "__main__":
    from torchsummary import summary

    mobilenetv2 = MobileNetV2(n_class=1000)
    # load_imagenet_weights(mobilenetv2.state_dict(), device="cpu")
    # print(mobilenetv2.state_dict()["layer7.Bottleneck7_0.conv2.weight"][1, 0, 1, 1])  # should be -0.0084
    #print(mobilenetv2)
    summary(mobilenetv2, (3, 224, 224), batch_size=1, device="cpu")