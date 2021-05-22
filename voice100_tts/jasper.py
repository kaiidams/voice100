import torch
from torch import nn

class JasperBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, repeat, residual):
        super(JasperBlock, self).__init__()
        layers = []
        c = in_channels
        for i in range(repeat):
            padding = kernel_size // 2
            layer = nn.Conv1d(c, c, kernel_size=kernel_size, stride=stride, padding=padding, groups=c, bias=False)
            layers.append(layer)
            layer = nn.Conv1d(c, out_channels, kernel_size=1, bias=False)
            c = out_channels
            layers.append(layer)
            layer = nn.BatchNorm1d(out_channels, eps=0.001)
            layers.append(layer)
            if i < repeat - 1:
                layer = nn.ReLU()
                layers.append(layer)
                layer = nn.Dropout(0.1)
                layers.append(layer)
        self.conv = nn.Sequential(*layers)
        if residual:
            layers = []
            layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            layers.append(layer)
            self.res = nn.Sequential(*layers)
        else:
            self.res = None
        layers = []
        layer = nn.ReLU()
        layers.append(layer)
        layer = nn.Dropout(0.1)
        layers.append(layer)
        self.out = nn.Sequential(*layers)
        
    def forward(self, x):
        y = self.conv(x)
        if self.res:
            y += self.res(x)
        y = self.out(y)
        return y

class QuartzNet(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(QuartzNet, self).__init__()
        self.layer = nn.Sequential(
            JasperBlock(input_dim, 256, kernel_size=33, stride=2, repeat=1, residual=False),

            JasperBlock(256, 512, kernel_size=63, stride=1, repeat=3, residual=True),
            JasperBlock(512, 512, kernel_size=63, stride=1, repeat=3, residual=True),
            JasperBlock(512, 512, kernel_size=75, stride=1, repeat=3, residual=True),
            JasperBlock(512, 512, kernel_size=75, stride=1, repeat=3, residual=True),
            JasperBlock(512, 512, kernel_size=75, stride=1, repeat=3, residual=True),

            JasperBlock(512, 512, kernel_size=87, stride=1, repeat=1, residual=False),
            JasperBlock(512, 1024, kernel_size=1, stride=1, repeat=1, residual=False),
        )
        self.dense = nn.Linear(1024, num_classes)

    def forward(self, spec):
        x = self.layer(spec)
        x = torch.transpose(x, 1, 2)
        x = self.dense(x)
        return x

if False:
    import pytorch_lightning as pl
    class EncoderCTC(pl.LightningModule):
        def __init__(self):
            super(EncoderCTC, self).__init__()
            self.encoder = QuartzNet()
        def forward(self, spec):
            return self.encoder(spec)
        
    module = EncoderCTC()
    module.summarize()