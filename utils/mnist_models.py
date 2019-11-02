import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_3l(nn.Module):
    def __init__(self, n_classes=10):
        super(cnn_3l, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class cnn_3l_large(nn.Module):
    def __init__(self, n_classes=10):
        super(cnn_3l_large, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, 5, 1)
        self.conv2 = nn.Conv2d(80, 200, 5, 1)
        self.fc1 = nn.Linear(4*4*200, 2000)
        self.fc2 = nn.Linear(2000, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*200)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)