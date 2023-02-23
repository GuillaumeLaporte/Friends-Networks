import torchxrayvision as xrv
import torchvision
import torch

from torch.utils.tensorboard import SummaryWriter
import skimage


# Prepare the image:
img = skimage.io.imread("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\european-rabbit-in-australia.jpg")
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img.mean(2)[None, ...] # Make single color channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])

img = transform(img)
img = torch.from_numpy(img)
img = torch.unsqueeze(img, 0)

xRayVisionModel = xrv.models.ResNet(weights="resnet50-res512-all")


writer = SummaryWriter("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\")
writer.add_graph(xRayVisionModel, img)
writer.close()



