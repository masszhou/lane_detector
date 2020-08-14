#########################################################################
## Some utility for training, data processing, and network.
#########################################################################
import torch
import torch.nn as nn


def backward_hook(self, grad_input, grad_output):
    print('grad_input norm:', grad_input[0].data.norm())


def cross_entropy2d(inputs, target, weight=None, size_average=True):
    loss = torch.nn.CrossEntropyLoss()

    n, c, h, w = inputs.size()
    prediction = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    gt = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1)

    return loss(prediction, gt)


######################################################################
## Convolution layer modules
######################################################################
class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size,
                                                    padding=padding, stride=stride, bias=bias),
                                          nn.BatchNorm2d(n_filters),
                                          nn.ReLU(inplace=True), )
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class BottleneckSame(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True):
        super(BottleneckSame, self).__init__()
        self.acti = acti
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1, acti=self.acti)

        self.residual = Conv2D_BatchNorm_Relu(in_channels, out_channels, 1, 0, 1)

    def forward(self, x):
        re = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if not self.acti:
            return out
        re = self.residual(x)
        out = out + re

        return out


class BottleneckDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckDown, self).__init__()
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 2)
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1)

        self.residual = Conv2D_BatchNorm_Relu(in_channels, out_channels, 3, 1, 2)

    def forward(self, x):
        re = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        re = self.residual(x)
        out = out + re

        return out


class BottleneckUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckUp, self).__init__()
        temp_channels = in_channels // 4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(temp_channels, temp_channels, 3, 2, 1, 1),
                                   nn.BatchNorm2d(temp_channels),
                                   nn.ReLU())
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1)

        self.residual = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        re = self.residual(re)

        out = out + re

        return out


class OutputLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(OutputLayer, self).__init__()
        self.conv = BottleneckSame(in_size, out_size, acti=False)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs


class HourglassBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBase, self).__init__()
        self.down1 = BottleneckDown(in_channels, out_channels)
        self.down2 = BottleneckDown(out_channels, out_channels)
        self.down3 = BottleneckDown(out_channels, out_channels)
        self.down4 = BottleneckDown(out_channels, out_channels)

        self.same1 = BottleneckSame(out_channels, out_channels)
        self.same2 = BottleneckSame(out_channels, out_channels)

        self.up2 = BottleneckUp(out_channels, out_channels)
        self.up3 = BottleneckUp(out_channels, out_channels)
        self.up4 = BottleneckUp(out_channels, out_channels)
        self.up5 = BottleneckUp(out_channels, out_channels)

        self.residual1 = BottleneckDown(in_channels, out_channels)
        self.residual2 = BottleneckDown(out_channels, out_channels)
        self.residual3 = BottleneckDown(out_channels, out_channels)
        self.residual4 = BottleneckDown(out_channels, out_channels)

    def forward(self, inputs):
        outputs1 = self.down1(inputs)  # 512*256 -> 256*128
        outputs2 = self.down2(outputs1)  # 256*128 -> 128*64
        outputs3 = self.down3(outputs2)  # 128*64 -> 64*32
        outputs4 = self.down4(outputs3)  # 64*32 -> 32*16

        outputs = self.same1(outputs4)  # 16*8 -> 16*8
        outputs = self.same2(outputs)  # 16*8 -> 16*8

        outputs = self.up2(outputs + self.residual4(outputs3))  # 32*16 -> 64*32
        outputs = self.up3(outputs + self.residual3(outputs2))  # 64*32 -> 128*64
        outputs = self.up4(outputs + self.residual2(outputs1))  # 128*64 -> 256*128
        outputs = self.up5(outputs + self.residual1(inputs))  # 256*128 -> 512*256

        return outputs


class ResizeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True):
        super(ResizeLayer, self).__init__()
        self.conv = Conv2D_BatchNorm_Relu(
            in_channels=in_channels,
            n_filters=out_channels // 2,
            k_size=7,
            padding=3,
            stride=2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.re1 = BottleneckSame(out_channels // 2, out_channels // 2)
        self.re2 = BottleneckSame(out_channels // 2, out_channels // 2)
        self.re3 = BottleneckSame(out_channels // 2, out_channels)

    def forward(self, inputs):
        """
        :param inputs: NCHW = [batch_size, 3,  256, 512]
        :return:
        """
        outputs = self.conv(inputs)  #
        outputs = self.re1(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.re2(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.re3(outputs)

        return outputs


class HourglassPINet(nn.Module):
    def __init__(self, in_channels, out_channels, instance_feature_size=4, input_re=True):
        super(HourglassPINet, self).__init__()
        self.layer1 = HourglassBase(in_channels, out_channels)
        self.re1 = BottleneckSame(out_channels, out_channels)
        self.re2 = BottleneckSame(out_channels, out_channels)
        self.re3 = BottleneckSame(1, out_channels)

        self.out_confidence = OutputLayer(out_channels, 1)
        self.out_offset = OutputLayer(out_channels, 2)
        self.out_instance = OutputLayer(out_channels, instance_feature_size)
        self.input_re = input_re

    def forward(self, inputs):
        outputs = self.layer1(inputs)
        outputs = self.re1(outputs)

        out_confidence = self.out_confidence(outputs)
        out_offset = self.out_offset(outputs)
        out_instance = self.out_instance(outputs)

        out = out_confidence

        outputs = self.re2(outputs)
        out = self.re3(out)

        if self.input_re:
            outputs = outputs + out + inputs
        else:
            outputs = outputs + out

        return [out_confidence, out_offset, out_instance], outputs
