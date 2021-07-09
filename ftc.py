"""
Fast-Text-Classifier model.
The model is based on MobileNet.
The model modifies MobileNetv2 source code from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.functional as F
import torch.nn as nn
import math

__all__ = ['ftc']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FTC(nn.Module):
    def __init__(self,
                 width_mult=1.,
                 dropout=False
                 ):
        super(FTC, self).__init__()

        self.dropout = dropout

        self.cfgsPool1 = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
        ]

        self.cfgsPool2 = [
            # t, c, n, s
            [6, 320, 1, 1],
        ]

        # custom for pooling
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layersPool1 = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgsPool1:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layersPool1.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.featuresPool1 = nn.Sequential(*layersPool1)

        layersPool2 = []
        for t, c, n, s in self.cfgsPool2:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layersPool2.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.featuresPool2 = nn.Sequential(*layersPool2)

        self.maxpool_3 = nn.AdaptiveMaxPool2d(3, return_indices=False)
        self.maxpool_5 = nn.AdaptiveMaxPool2d(5, return_indices=False)

        if self.dropout:
            self.block_classifiers = nn.Sequential(
                nn.Linear(480, 1000),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1000, 2)
            )
        else:
            self.block_classifiers = nn.Sequential(
                nn.Linear(480, 1000),
                nn.Linear(1000, 2)
            )

        self._initialize_weights()


    def forward(self, x):

        features1 = self.featuresPool1(x)
        features2 = self.featuresPool2(features1)

        # merge features
        # 3x3, 5x5 blocks
        f1 = self.maxpool_3(features1)
        f2 = self.maxpool_3(features2)
        block3 = torch.cat((f1, f2), 1)

        f1 = self.maxpool_5(features1)
        f2 = self.maxpool_5(features2)
        block5 = torch.cat((f1, f2), 1)

        block = block3.permute(0, 2, 3, 1)
        y3 = self.block_classifiers(block)
        out3 = self._magicCombine(y3, 1, 3)

        block = block5.permute(0, 2, 3, 1)
        y5 = self.block_classifiers(block)
        out5 = self._magicCombine(y5, 1, 3)

        output = torch.cat((out3, out5), 1)

        return output

    def _magicCombine(self, x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def ftc(**kwargs):
    """
    Constructs FTC model
    """
    return FTC(**kwargs)

