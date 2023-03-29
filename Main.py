import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#The following list contains the lower layers of the ImageNet model at different level. 
#Each element of the list is a succession of layers up to a different level in the ImageNet model.
ImageNetModelsLowerLayers = []

imageNetModel = models.resnet18(pretrained = True)
partialImageNetModel1 = torch.nn.Sequential(*list(imageNetModel.children())[:-5])
partialImageNetModel2 = torch.nn.Sequential(*list(imageNetModel.children())[:-4])
partialImageNetModel3 = torch.nn.Sequential(*list(imageNetModel.children())[:-3])
partialImageNetModel4 = torch.nn.Sequential(*list(imageNetModel.children())[:-2])
partialImageNetModels = [partialImageNetModel1, partialImageNetModel2, partialImageNetModel3, partialImageNetModel4]

for model in partialImageNetModels:
  for param in model.parameters():
    param.requires_grad = False

def outputSizeOfLowerLayers(lowerLayers, testImage):
    return lowerLayers(testImage).size()

class MergedNetwork(torch.nn.Module):

    def __init__(self, lowerLayersOfImageNet, lowerLayersOfSimpleModel, upperLayersOfSimpleModel, layerOfSimpleModelToBeReplaced, specificationsOfLayerToBeReplaced, batchSize):
        super(MergedNetwork, self).__init__()
        self.lowerLayersOfImageNet = lowerLayersOfImageNet
        self.lowerLayersOfSimpleModel = lowerLayersOfSimpleModel
        self.upperLayersOfSimpleModel = upperLayersOfSimpleModel

        #Cifar10 Images are 32 by 32 with 3 channels (I think (the color channels))
        self.outputSizeValuesOfLowerLayersOfSimpleModel = outputSizeOfLowerLayers(lowerLayersOfSimpleModel, torch.zeros((batchSize,3,32,32)))


        #the ImageNet network is ResNet18 which takes as input 224 by 244 images with 3 channels
        self.outputSizeValuesOfLowerLayersOfImageNetModel = outputSizeOfLowerLayers(lowerLayersOfImageNet, torch.zeros((batchSize,3,224,224)))
        
        #In the following, the index value of (1) should be the number of channels.
        numberOfOutputChannelsOfLowerLayersOfImageNetModel = self.outputSizeValuesOfLowerLayersOfImageNetModel[1]
        self.numberOfInputChannelsInNewMergingLayer = self.outputSizeValuesOfLowerLayersOfSimpleModel[1] + numberOfOutputChannelsOfLowerLayersOfImageNetModel

        ###########################################################################################
        #We construct below the new convolution layer
        outNumberOfChannels = specificationsOfLayerToBeReplaced["outChannels"]
        kernelSize = specificationsOfLayerToBeReplaced["kernelSize"]
        padding = specificationsOfLayerToBeReplaced["padding"]
        self.newConvolutionLayer = torch.nn.Conv2d(self.numberOfInputChannelsInNewMergingLayer, outNumberOfChannels, kernelSize, padding = padding)

        listOfParameters = []
        for param in self.newConvolutionLayer.parameters():
          param.requires_grad = False
          listOfParameters.append(param)

        listOfParameters[0][:,:,:,:] = torch.zeros((outNumberOfChannels, self.numberOfInputChannelsInNewMergingLayer, kernelSize, kernelSize))
        listOfParameters[0][:, numberOfOutputChannelsOfLowerLayersOfImageNetModel:,:,:] = layerOfSimpleModelToBeReplaced.state_dict()["weight"]
        listOfParameters[1][:] = layerOfSimpleModelToBeReplaced.state_dict()["bias"]

        for param in self.newConvolutionLayer.parameters():
          param.requires_grad = True

        ###########################################################################################
        #Becareful here! We don't want to interpolate in the channel dimension!
        targetFeatureMapWidth = self.outputSizeValuesOfLowerLayersOfSimpleModel[2]
        targetFeatureMapHeight = self.outputSizeValuesOfLowerLayersOfSimpleModel[3]
        self.targetShapeOfImageNetOutput = (targetFeatureMapWidth, targetFeatureMapHeight)


    def forward(self, x):
        #The ImageNet model (ResNet18) takes as input 224 by 224 images (with 3 channels).
        #Cifar10 Images are 32 by 32 and are color images, so have 3 channels (I think).
        inputToImageNet = F.interpolate(x, size = (224, 224), mode = "bilinear")
        lowerLayersImageNetOutput = self.lowerLayersOfImageNet(inputToImageNet)

        #Becareful here! We don't want to interpolate in the channel dimension!
        lowerLayersImageNetOutput = F.interpolate(lowerLayersImageNetOutput, size = self.targetShapeOfImageNetOutput, mode = "bilinear")

        lowerLayersSimpleModelOutput = self.lowerLayersOfSimpleModel(x)

        #Below is the right order in which to do the concatenation:
        inputTensorToTheNewLayer = torch.cat([lowerLayersImageNetOutput, lowerLayersSimpleModelOutput], dim = 1)

        outputOfNewLayer = self.newConvolutionLayer(inputTensorToTheNewLayer)

        return self.upperLayersOfSimpleModel(outputOfNewLayer)

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


simpleModel = Net()

numberOfChildren = len(list(simpleModel.children()))

partsOfSimpleModellist = []


for layerToBeReplacedNumber in range(1,5):

  lowerLayers = []
  upperLayers = []
  
  skipChild = False
  convolutionLayerNumberOrPlusOne = 0
  for (childNumber, child) in enumerate(simpleModel.children()):

    if(isinstance(child, nn.Conv2d) and (childNumber > 0)):
      convolutionLayerNumberOrPlusOne = convolutionLayerNumberOrPlusOne + 1

      if(convolutionLayerNumberOrPlusOne == layerToBeReplacedNumber):
        layerToBeReplaced = child
        skipChild = True
        convolutionLayerNumberOrPlusOne = convolutionLayerNumberOrPlusOne + 1

    if(convolutionLayerNumberOrPlusOne > layerToBeReplacedNumber):
      
      if(skipChild == False):
        upperLayers.append(child)

      if((isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)) and (childNumber < numberOfChildren - 1)):
        upperLayers.append(nn.ReLU())

    elif((convolutionLayerNumberOrPlusOne < layerToBeReplacedNumber) and (skipChild == False)):
      lowerLayers.append(child)

      if((isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)) and (childNumber < numberOfChildren - 1)):
        lowerLayers.append(nn.ReLU())

    skipChild = False
    
    

  partsOfSimpleModellist.append({})
  partsOfSimpleModellist[layerToBeReplacedNumber - 1]["Lower Layers"] = torch.nn.Sequential(*lowerLayers)
  partsOfSimpleModellist[layerToBeReplacedNumber - 1]["Upper Layers"] = torch.nn.Sequential(*upperLayers)
  partsOfSimpleModellist[layerToBeReplacedNumber - 1]["Layer to be replaced"] = layerToBeReplaced

for partsOfSimpleModelIndex in range(4):
  for child in partsOfSimpleModellist[partsOfSimpleModelIndex]["Lower Layers"]:
    for param in child.parameters():
      param.requires_grad = False

  for child in partsOfSimpleModellist[partsOfSimpleModelIndex]["Upper Layers"]:
    for param in child.parameters():
      param.requires_grad = False

specificationsOfLayerToBeReplaced = {}
specificationsOfLayerToBeReplaced["outChannels"] = 256
specificationsOfLayerToBeReplaced["kernelSize"] = 3
specificationsOfLayerToBeReplaced["padding"] = "same"

someMergedNetwork = MergedNetwork(partialImageNetModels[2], partsOfSimpleModellist[2]["Lower Layers"], partsOfSimpleModellist[2]["Upper Layers"], partsOfSimpleModellist[2]["Layer to be replaced"], specificationsOfLayerToBeReplaced, 4)
someMergedNetwork(torch.ones(4, 3, 32, 32))

#print(simpleModel)
#print(partsOfSimpleModellist[1]["Lower Layers"])
#print(partsOfSimpleModellist[1]["Upper Layers"])
#print(partsOfSimpleModellist[1]["Layer to be replaced"])