# ref: https://github.com/facebookresearch/odin
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


################################################################
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Ref: https://github.com/xternalz/WideResNet-pytorch
    """
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, input_std=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # Normalization: 
        self.input_std = input_std
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        if self.input_std:
            size = list(x.shape)
            d= x.view(-1, size[1]*size[2]*size[3]).float()
            m = torch.mean(d, -1).view(-1, 1, 1, 1)
            s = torch.std(d, -1).view(-1, 1, 1, 1)
            x = (x-m)/(s+1e-10)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    
    
    
    
    
    
    
    
############################################################################################   
    
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, activate_before_residual):
        super(ResBlock, self).__init__()
        self.activate_before_residual = activate_before_residual
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool1 = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        if self.activate_before_residual:
            x = self.bn1(x)
            x = self.relu1(x)
            orig_x = x
        else:
            orig_x = x
            x = self.bn1(x)
            x = self.relu1(x)
        out = self.conv1(x)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        if self.in_planes != self.out_planes:
            orig_x = self.avgpool1(orig_x)
            orig_x = F.pad(orig_x, (0, 0, 0, 0, (self.out_planes -
                                                 self.in_planes)//2, (self.out_planes-self.in_planes)//2, 0, 0))
        out += orig_x
        return out
    
class AggResBlock(nn.Module):
    def __init__(self, block, in_planes, out_planes, stride, activate_before_residual):
        super(AggResBlock, self).__init__()
        layers=  []
        for i in range(1, 5):
            layers.append(block(in_planes, out_planes, stride, activate_before_residual))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet_madry(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_madry, self).__init__()
        filters = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_0 = ResBlock(
            filters[0], filters[1], strides[0], activate_before_residual[0])
        self.block1_1 = AggResBlock(ResBlock, filters[1], filters[1], 1, False)
        self.block2_0 = ResBlock(
            filters[1], filters[2], strides[1], activate_before_residual[1])
        self.block2_1 = AggResBlock(ResBlock, filters[2], filters[2], 1, False)
        self.block3_0 = ResBlock(
            filters[2], filters[3], strides[2], activate_before_residual[2])
        self.block3_1 = AggResBlock(ResBlock, filters[3], filters[3], 1, False)
        self.bn = nn.BatchNorm2d(filters[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fc = nn.Linear(filters[3], num_classes)
        self.nfilters = filters[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # input standardization
        size = list(x.shape)
        d= x.view(-1, size[1]*size[2]*size[3]).float()
        m = torch.mean(d, -1).view(-1, 1, 1, 1)
        s = torch.std(d, -1).view(-1, 1, 1, 1)
        x = (x-m)/(s+1e-10)
        out = self.conv1(x)
        out = self.block1_0(out)
        out = self.block1_1(out)
        out = self.block2_0(out)
        out = self.block2_1(out)
        out = self.block3_0(out)
        out = self.block3_1(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nfilters)
        return self.fc(out)
