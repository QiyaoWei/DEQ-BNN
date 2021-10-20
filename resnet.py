from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader

from tqdm import tqdm

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10, prune=False):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=54 else BasicBlock
        # ========== according to the GraSP code, we double the #filter here ============
        self.ratio = get_ratio(prune)
        self.inplanes = int(32 * next(self.ratio))
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, n)
        self.layer2 = self._make_layer(block, 64, n, stride=2)
        self.layer3 = self._make_layer(block, 128, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(self.inplanes, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        ratio = next(self.ratio)
        temp_planes = int(planes*ratio)
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, temp_planes,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(temp_planes),
        )
        layers = []
        layers.append(block(self.inplanes, temp_planes, stride, downsample))
        self.inplanes = temp_planes
        for _ in range(1, blocks):
            ratio = next(self.ratio)
            temp_planes = int(planes*ratio)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, temp_planes,
                        kernel_size=1, bias=False),
                nn.BatchNorm2d(temp_planes),
            )
            layers.append(block(self.inplanes, temp_planes, downsample=downsample))
            self.inplanes = temp_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    

        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x) 

        x = F.avg_pool2d(x,x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def get_ratio(prune=True):
    x = 1
    while True:
        if prune:
            yield 1/(x**1.65)
            x += 0.1
        else:
            yield 1

def main():
    data_dir = './data'
    batchsize = 500
    lr = 0.1
    epochs = 160
    momentum = 0.9
    weight_decay = 1e-4
    gamma = 0.1
    prune = True
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    traindata = DataLoader(trainset, batchsize, True)
    testdata = DataLoader(testset, batchsize, False)
    model = resnet(depth=32, prune=prune).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_sched = lr_scheduler.MultiStepLR(optimizer, [80, 120], gamma)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        train_correct = 0
        for x, y in tqdm(traindata):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += torch.sum(torch.argmax(y_pred, 1) == y).item()
        train_loss = train_loss/50000
        train_acc = train_correct/50000
        lr_sched.step()
        test_loss = 0
        test_correct = 0
        for x, y in tqdm(testdata):
            x, y = x.cuda(), y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            test_correct += torch.sum(torch.argmax(y_pred, 1) == y).item()
        test_loss = test_loss/10000
        test_acc = test_correct/10000
        print(f"Epoch:{epoch} Training Acc:{train_acc} Test Acc:{test_acc}")

if __name__ == '__main__':
    # print(resnet(depth=32, prune=True))
    # print(count_total_parameters(resnet(depth=32,prune=True)))
    main()