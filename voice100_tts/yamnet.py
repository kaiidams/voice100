import torch
from torch import nn

class YamNetConv(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, params
    ):
        super().__init__()
        #self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, bias=False,
            padding=2)
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=params.batchnorm_epsilon)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class YamNetSeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size, stride, params
    ):
        super().__init__()
        
        if stride == 2:
            pass #self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        else:
            pass #self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        self.pad = None
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride, bias=False,
            padding=(2, 2))
        self.depthwise_bn = nn.BatchNorm2d(
            in_channels,
            eps=params.batchnorm_epsilon)
        self.depthwise_relu = nn.ReLU()
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, [1, 1],
            stride=1, bias=False,
            padding=0)
        self.pointwise_bn = nn.BatchNorm2d(
            out_channels,
            eps=params.batchnorm_epsilon)
        self.pointwise_relu = nn.ReLU()
        
    def forward(self, x):
        #print(x.shape)
        if self.pad is not None:
            x = self.pad(x)
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_relu(x)
        return x

def _conv(name, kernel, stride, prev_filters, filters, params):
    return YamNetConv(prev_filters, filters, kernel, stride, params)

def _separable_conv(name, kernel, stride, prev_filters, filters, params):
    return YamNetSeparableConv(prev_filters, filters, kernel, stride, params)

_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]

_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,           [5, 5], (2, 1),   32 // 8),
    (_separable_conv, [5, 5], (1, 1),   64 // 8),
    (_separable_conv, [5, 5], (2, 1),  128 // 8),
    (_separable_conv, [5, 5], (1, 1),  128 // 8),
    (_separable_conv, [5, 5], (2, 1),  256 // 8),
    (_separable_conv, [5, 5], (1, 1),  256 // 8),
    (_separable_conv, [5, 5], (2, 1),  512 // 8),
    (_separable_conv, [5, 5], (1, 1),  512 // 8)
]

class YamNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_filters = 1
        for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
            name = 'layer{}'.format(i + 1)
            layer = layer_fun(name, kernel, stride, prev_filters, filters, params)
            self.layers.append(layer)
            prev_filters = filters
        self.dense = nn.Linear(4 * prev_filters, params.num_classes, bias=True)
        self.activation = nn.Sigmoid()
    
    def forward(self, features):
        #net = torch.reshape(features, (-1, 1, params.patch_frames, params.patch_bands))
        net = torch.transpose(features, 1, 2)
        net = net[:, None, :, :]
        for layer in self.layers:
            net = layer(net)
        #embeddings = torch.mean(net, dim=[2, 3])
        embeddings = torch.transpose(net, 1, 3)
        embeddings = torch.flatten(embeddings, 2)
        logits = self.dense(embeddings)
        #predictions = self.activation(logits)
        return logits, embeddings


def create_model():

    import argparse
    params = argparse.Namespace(
        sample_rate=16000.0, stft_window_seconds=0.025, stft_hop_seconds=0.01, mel_bands=64, mel_min_hz=125.0,
        mel_max_hz=7500.0, log_offset=0.001, patch_window_seconds=0.96, patch_hop_seconds=0.48, num_classes=521,
        conv_padding='same', batchnorm_center=True, batchnorm_scale=False, batchnorm_epsilon=0.0001,
        classifier_activation='sigmoid', tflite_compatible=False)

    params.patch_frames = int(round(params.patch_window_seconds / params.stft_hop_seconds))
    params.patch_bands = params.mel_bands
    params.num_classes = 27

    return YamNet(params)