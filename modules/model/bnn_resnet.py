from torch import nn
# from torch.nn import functional as F
from blitz.modules import BayesianConv2d, BayesianLinear, BayesianModule
from blitz.utils import variational_estimator

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, drop_rate=0, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.conv1 = BayesianConv2d(in_channel, out_channel, (3,3), padding=1)
        self.drop = nn.Dropout(drop_rate)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.conv2 = BayesianConv2d(out_channel, out_channel, (3,3), padding=1, stride=stride)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut.add_module('shortconv',
                BayesianConv2d(in_channel, out_channel, kernel_size=(1,1), stride=stride)
            )

    def forward(self, x):
        out = self.drop(self.conv1(self.relu1(self.bn1(x))))
        out = self.conv2(self.relu2(self.bn2(out)))
        out += self.shortcut(x)

        return out

@variational_estimator
class BNNWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, in_channel, dropout_rate=0):
        super().__init__()
        self.in_channel = in_channel
        self.start_channel = 16
        self.num_classes = num_classes

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'

        n = (depth-4)/6
        k = widen_factor
        n_stages = [16, 16*k, 32*k, 64*k]

        self.layer1 = BayesianConv2d(self.in_channel, n_stages[0], (3,3), padding=1)
        self.layer2 = self.__wide_layer(n_stages[1], n, dropout_rate, stride=1)
        self.layer3 = self.__wide_layer(n_stages[2], n, dropout_rate, stride=2)
        self.layer4 = self.__wide_layer(n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.relu1 = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten()
        self.linear = BayesianLinear(n_stages[3], num_classes)

    def __wide_layer(self, channel, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(BasicBlock(self.start_channel, channel,
                                    drop_rate=dropout_rate, stride=stride))
            self.start_channel = channel

        return nn.Sequential(*layers)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, BayesianModule):
                m.resample()
        
        for module in self.children():
            x = module(x)
        return x