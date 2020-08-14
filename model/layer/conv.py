from torch import nn


class Conv2D(nn.Module):
    """
    Convolution -> Batch Normalization -> Relu: denote as "cbr unit"
    """
    def __init__(self,
                 inp_dim,
                 out_dim,
                 ksize,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 bn=False,
                 relu=True):

        super(Conv2D, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, ksize, stride, padding, dilation, groups, bias, padding_mode)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

