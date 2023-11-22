import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_utils import misc
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, gain=1, use_wscale=False, lrmul=1, activation=None, conv_clamp=None):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.activation = activation
        
        he_std = gain / np.sqrt(in_channels * kernel_size ** 2)  # He initialization std
        # Equalized learning rate and custom learning rate multiplier.
        init_std = 1.0 / lrmul
        self.scale = he_std * lrmul
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * init_std)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bias_gain = lrmul
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight * self.scale
        b = self.bias
        if b is not None:
            b = b * self.bias_gain
        x = F.conv2d(x, w, b, padding=self.padding, stride=self.stride)
        if self.activation:
            x = F.leaky_relu(x, 0.2)
        if self.conv_clamp:
            x = x.clamp(-self.conv_clamp, self.conv_clamp)
        return x

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, tmp_channels, out_channels, resolution, img_channels, first_layer_idx, architecture='resnet', activation='lrelu', conv_clamp=None, use_fp16=False, fp16_channels_last=False, freeze_layers=0):
        super().__init__()
        self.resolution = resolution
        self.architecture = architecture
        self.channels_last = fp16_channels_last

        # FromRGB layer if needed
        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, bias=True, activation=activation, conv_clamp=conv_clamp)

        # Main layers
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, bias=True, activation=activation, conv_clamp=conv_clamp)
        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, bias=True, activation=activation, conv_clamp=conv_clamp)

        # Skip layer for the 'resnet' architecture
        if architecture == 'resnet':
            self.skip = nn.Conv2d(tmp_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)

    def forward(self, x, img):

        # FromRGB
        if self.in_channels == 0 or self.architecture == 'skip':
            img = self.fromrgb(img)
            x = x + img if x is not None else img

        # Main convolutions
        if self.architecture == 'resnet' and x is not None:
            y = F.avg_pool2d(x, kernel_size=2, stride=2)  # Downsample
            y = self.skip(y)
            x = self.conv0(x)
            x = self.conv1(x)
            x = (x + y) / np.sqrt(2)  # Scale the residuals
        else:
            x = self.conv0(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)  # Downsample
            x = self.conv1(x)

        return x

class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 in_features,                # Number of input features.
                 out_features,               # Number of output features.
                 bias=True,                  # Apply additive bias before the activation function?
                 activation='linear',        # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,            # Learning rate multiplier.
                 bias_init=0,                # Initial value for the additive bias.
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.lr_multiplier = lr_multiplier

        # Initialize weights and biases
        he_std = np.sqrt(2) / np.sqrt(in_features)  # He initialization std
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * he_std * lr_multiplier)
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), bias_init * lr_multiplier))
        else:
            self.bias = None

    def forward(self, x):
        # Apply the linear transformation
        x = F.linear(x, self.weight, self.bias)

        # Apply activation function
        if self.activation == 'lrelu':
            x = F.leaky_relu(x, negative_slope=0.2)
        elif self.activation == 'relu':
            x = F.relu(x)

        return x

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, activation={self.activation}'