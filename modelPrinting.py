import torchxrayvision as xrv
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import skimage

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding = "same")
        self.relu1 = F.relu
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p = 0.3)

        self.conv2 = nn.Conv2d(32, 64, 3, padding = "same")
        self.relu2 = F.relu
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p = 0.3)

        self.conv3 = nn.Conv2d(64, 128, 3, padding = "same")
        self.relu3 = F.relu
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p = 0.4)

        self.conv4 = nn.Conv2d(128, 256, 3, padding = "same")
        self.relu4 = F.relu
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p = 0.4)


        self.conv5 = nn.Conv2d(256, 512, 3, padding = "same")
        self.relu5 = F.relu
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dropout5 = nn.Dropout(p = 0.4)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(512, 80)
        self.relu6 = F.relu
        self.dropout6 = nn.Dropout(p = 0.3)

        self.linear2 = nn.Linear(80, 20)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.dropout5(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.relu6(x)
        x = self.dropout6(x)

        x = self.linear2(x)
        
        return x


# Prepare the image:
img = skimage.io.imread("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\european-rabbit-in-australia.jpg")
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img.mean(2)[None, ...] # Make single color channel

transformImageNet = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
imgImageNet = transformImageNet(img)
imgImageNet = torch.from_numpy(imgImageNet)
imgImageNet = torch.unsqueeze(imgImageNet, 0)
imgImageNet = torch.cat([imgImageNet,imgImageNet,imgImageNet], dim = 1)

transformCifar10 = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(32)])
imgCifar10 = transformCifar10(img)
imgCifar10 = torch.from_numpy(imgCifar10)
imgCifar10 = torch.unsqueeze(imgCifar10, 0)
imgCifar10 = torch.cat([imgCifar10,imgCifar10,imgCifar10], dim = 1)



imageNetModel = torchvision.models.resnet18(pretrained = True)
cifar10Network = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Simple Model.pt")

writer = SummaryWriter("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\Cifar 10 Network\\")
writer.add_graph(cifar10Network, imgCifar10)
writer.close()

writer = SummaryWriter("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\ImageNet Model\\")
writer.add_graph(imageNetModel, imgImageNet)
writer.close()

