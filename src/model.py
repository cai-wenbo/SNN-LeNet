import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer



# SNN Lenet model
class SNN_LeNet(nn.Module):
    def __init__(self):
        super(SNN_LeNet, self).__init__()
        self.conv1 = layer.Conv2d(1, 6, 5, padding = 2)
        self.IF1   = neuron.IFNode(surrogate_function = surrogate.ATan())
        self.norm1 = layer.BatchNorm2d(num_features = 6)
        self.pool1 = layer.AvgPool2d(2)
        self.conv2 = layer.Conv2d(6, 16, 5)
        self.IF2   = neuron.IFNode(surrogate_function = surrogate.ATan())
        self.norm2 = layer.BatchNorm2d(num_features = 16)
        self.pool2 = layer.AvgPool2d(2)
        self.fc1   = layer.Linear(400, 120)
        self.IF3   = neuron.IFNode(surrogate_function = surrogate.ATan())
        self.norm3 = layer.BatchNorm1d(num_features=1)
        self.fc2   = layer.Linear(120, 84)
        self.IF4   = neuron.IFNode(surrogate_function = surrogate.ATan())
        self.norm4 = layer.BatchNorm1d(num_features=1)
        self.fc3   = nn.Linear(84, 10)
        self.IF5   = neuron.IFNode(surrogate_function = surrogate.ATan())
            


    def forward(self, x):
        y = self.conv1(x)
        y = self.IF1(y)
        y = self.norm1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.IF2(y)
        y = self.norm2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], 1, -1)
        y = self.fc1(y)
        y = self.IF3(y)
        y = self.norm3(y)
        y = self.fc2(y)
        y = self.IF4(y)
        y = self.norm4(y)
        y = y.view(y.shape[0], -1)
        y = self.fc3(y)
        y = self.IF5(y)
        return y
