import torch
import torch.nn as nn

from model.utils import init_weight

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ENet(nn.Module):
    def __init__(self, nclass):
        super(ENet, self).__init__()

        # see https://arxiv.org/pdf/1606.02147.pdf
        # suppose input size is [#batch, 512, 512]
        self.initial = InitialBlock(13)  # out [#batch, 16, 256, 256]

        self.bottleneck1_0 = Bottleneck(16, 64, mod="downsampling", dropout_rate=0.01)  # out [#batch, 64, 128, 128]
        self.bottleneck1_1 = Bottleneck(64, 64, dropout_rate=0.01)
        self.bottleneck1_2 = Bottleneck(64, 64, dropout_rate=0.01)
        self.bottleneck1_3 = Bottleneck(64, 64, dropout_rate=0.01)
        self.bottleneck1_4 = Bottleneck(64, 64, dropout_rate=0.01)
        # out [#batch, 64, 128, 128]

        self.bottleneck2_0 = Bottleneck(64, 128, mod="downsampling")  # out [#batch, 128, 64, 64]
        self.bottleneck2_1 = Bottleneck(128, 128)
        self.bottleneck2_2 = Bottleneck(128, 128, mod="dilated", dilation_rate=2)
        self.bottleneck2_3 = Bottleneck(128, 128, mod="asymmetric5")
        self.bottleneck2_4 = Bottleneck(128, 128, mod="dilated", dilation_rate=4)
        self.bottleneck2_5 = Bottleneck(128, 128)
        self.bottleneck2_6 = Bottleneck(128, 128, mod="dilated", dilation_rate=8)
        self.bottleneck2_7 = Bottleneck(128, 128, mod="asymmetric5")
        self.bottleneck2_8 = Bottleneck(128, 128, mod="dilated", dilation_rate=16)
        # out [#batch, 128, 64, 64]

        self.bottleneck3_1 = Bottleneck(128, 128)
        self.bottleneck3_2 = Bottleneck(128, 128, mod="dilated", dilation_rate=2)
        self.bottleneck3_3 = Bottleneck(128, 128, mod="asymmetric5")
        self.bottleneck3_4 = Bottleneck(128, 128, mod="dilated", dilation_rate=4)
        self.bottleneck3_5 = Bottleneck(128, 128)
        self.bottleneck3_6 = Bottleneck(128, 128, mod="dilated", dilation_rate=8)
        self.bottleneck3_7 = Bottleneck(128, 128, mod="asymmetric5")
        self.bottleneck3_8 = Bottleneck(128, 128, mod="dilated", dilation_rate=16)
        # out [#batch, 128, 64, 64]

        self.bottleneck4_0 = Bottleneck(128, 64, mod="upsampling")
        self.bottleneck4_1 = Bottleneck(64, 64)
        self.bottleneck4_2 = Bottleneck(64, 64)
        # out [#batch, 64, 256, 256]

        self.bottleneck5_0 = Bottleneck(64, 16, mod="upsampling")
        self.bottleneck5_1 = Bottleneck(16, 16)
        # out [#batch, 16, 512, 512]

        self.fullconv = nn.ConvTranspose2d(16, nclass, 2, 2, bias=False)

        # initialize weights
        self.apply(init_weight)

    def forward(self, x):
        x = self.initial(x)  # out [16,

        x, max_indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        x, max_indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)

        x = self.bottleneck4_0(x, max_indices2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x, max_indices1)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x)
        return x

    def transfer_weights(self, weights):
        # https: // github.com / e - lab / ENet - training
        pass


class InitialBlock(nn.Module):
    def __init__(self, out_depth):

        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, out_depth, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_depth + 3)
        self.activation = nn.PReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.maxpool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_depth,
                 out_depth,
                 dropout_rate=0.1,
                 compress_scale=4,
                 mod="regular",
                 dilation_rate=1):

        super(Bottleneck, self).__init__()

        # see https://arxiv.org/pdf/1606.02147.pdf
        assert (mod in ["regular", "downsampling", "dilated", "asymmetric5", "upsampling"])

        self.mod = mod
        self.in_depth = in_depth
        self.out_depth = out_depth

        # compressed_depth means use conv1 1x1 to compress tensor depth with a scale factor before core conv
        # and use conv3 1x1 to restore the depth after core conv
        reduced_depth = out_depth // compress_scale

        # -----------------
        # residual branch
        # -----------------
        # conv1, 1x1
        # BN
        # PReLU
        # conv2, 3x3 or asymmetric or dilated or ConvTransposed
        # BN
        # PReLU
        # conv3, 1x1
        # BN
        # PReLU
        # Regularizer
        if mod in ["regular", "dilated"]:
            self.residual_block = nn.Sequential(
                nn.Conv2d(in_depth, reduced_depth, kernel_size=1, stride=1, bias=False),  # downsampling, if
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, reduced_depth, kernel_size=3, dilation=dilation_rate, padding=dilation_rate, stride=1, bias=False),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, out_depth, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.Dropout2d(p=dropout_rate)
            )
        elif mod == "downsampling":
            # -- notes
            # original repo by Prof. Eugenio Culurciello https://github.com/e-lab/ENet-training links to a
            # keras implementation, which down-sample at 1x1 conv
            # but some other repo down-sample at 3x3 conv

            # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
            self.residual_block = nn.Sequential(
                nn.Conv2d(in_depth, reduced_depth, kernel_size=2, stride=2, bias=False),  # downsampling, if
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, reduced_depth, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, out_depth, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.Dropout2d(p=dropout_rate)
            )
        elif mod == "asymmetric5":
            self.residual_block = nn.Sequential(
                nn.Conv2d(in_depth, reduced_depth, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, reduced_depth, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(reduced_depth, reduced_depth, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, out_depth, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.Dropout2d(p=dropout_rate)
            )

        elif mod == "upsampling":
            self.residual_block = nn.Sequential(
                nn.Conv2d(in_depth, reduced_depth, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.ConvTranspose2d(reduced_depth, reduced_depth, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),  # upsampling, output_padding=0 or 1
                nn.BatchNorm2d(reduced_depth),
                nn.PReLU(),
                nn.Conv2d(reduced_depth, out_depth, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_depth),
                nn.Dropout2d(p=dropout_rate)
            )
        else:
            print("error")

        # -----------------
        # main branch
        # -----------------
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) if mod == "downsampling" else None
        self.upsampling = nn.MaxUnpool2d(2) if mod == "upsampling" else None
        self.activation = nn.PReLU()
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_depth, out_depth, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_depth)
        ) if mod == "upsampling" else None

    def forward(self, x, *args):
        batch_size = x.shape[0]
        x_copy = x
        max_indices = None

        # -----------------
        # main branch
        # -----------------
        # if downsample
        #   maxpool
        # zeropadding ( or conv4 when upsampling )

        if self.mod == "downsampling":
            x_copy, max_indices = self.maxpool(x_copy)

        if self.mod == "upsampling":
            assert (len(args) > 0)
            # conv to fit channels
            x_copy = self.conv4(x_copy)
            x_copy = self.upsampling(x_copy, args[0])
        else:
            # zero padding to fit channels
            if self.in_depth != self.out_depth:
                padding_depth = self.out_depth - self.in_depth  # for encode, out_depth >= in_depth
                height = x_copy.shape[2]
                width = x_copy.shape[3]
                zero_padding = torch.zeros([batch_size, padding_depth, height, width]).to(device)  # NCHW
                x_copy = torch.cat([x_copy, zero_padding], dim=1)

        # -----------------
        # residual branch
        # -----------------
        res = self.residual_block(x)

        # -----------------
        # merge two branches
        # -----------------
        # add both branch, not concat, recall residual structure
        # out = activation(main + residual)

        out = res + x_copy
        out = self.activation(out)

        if self.mod == "downsampling":
            return out, max_indices
        else:
            return out


if __name__ == "__main__":
    from torchsummary import summary

    enet = ENet(nclass=12).to(device)

    summary(enet, (3, 512, 512), batch_size=1)