import torchxrayvision as xrv
import torchvision
import torch

from torch.utils.tensorboard import SummaryWriter
import skimage


# Prepare the image:
img = skimage.io.imread("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\european-rabbit-in-australia.jpg")
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img.mean(2)[None, ...] # Make single color channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

img = transform(img)
img = torch.from_numpy(img)
img = torch.unsqueeze(img, 0)

xRayVisionModel = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
imageNetModel = torchvision.models.resnet18(pretrained = True)


writer = SummaryWriter("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\X-Ray Vision Model\\")
writer.add_graph(xRayVisionModel, img)
writer.close()

writer = SummaryWriter("D:\\School\\2023 Winter\\Term Project\\Model Visualization\\ImageNet Model\\")

imageNetImage = torch.cat([img,img,img],dim = 1)
writer.add_graph(imageNetModel, imageNetImage)
writer.close()

