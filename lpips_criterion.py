import torch.nn as nn
from collections import namedtuple
import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Alexnet, self).__init__()
        alex_pretrained_features = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), alex_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alex_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alex_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alex_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alex_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alex_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alex_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class Lpips(nn.Module):
    def __init__(self, net, device):
        super(Lpips, self).__init__()
        self.net = net
        self.vgg = Vgg16(requires_grad=False).to(device)
        self.alex = Alexnet(requires_grad=False).to(device)
        self.criterion = nn.MSELoss()

    def forward(self, rec, label):
        if self.net == 'vgg16':
            features_rec = self.vgg(rec)
            features_label = self.vgg(label)
            lpips_loss = self.criterion(features_rec.relu2_2, features_label.relu2_2) + \
                         self.criterion(features_rec.relu4_3, features_label.relu4_3)

        if self.net == 'alex':
            features_rec = self.alex(rec)
            features_label = self.alex(label)
            lpips_loss = self.criterion(features_rec.relu2, features_label.relu2) + \
                         self.criterion(features_rec.relu4, features_label.relu4)

        return lpips_loss
