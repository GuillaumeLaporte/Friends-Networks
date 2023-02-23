import torch 
import torch.nn.functional as F
import torchxrayvision as xrv
from torchvision import models

#Some image coming from the X-Ray Vision dataset
testImage = 1


#The following lists contain the X-ray model split at different levels.
#Each elements of these lists is a succession of layers. 
#The index is the position of the split.
#We will take each of these successions of layers to construct the new networks to be analysed.
#A new layer will have to be added at the split position.
#The split position layer is not included in either (xRayModelsLowerLayers) or (xRayModelsUpperLayers).
#The layer to be replaced in the X-Ray Vision model is in the list (xRayModelsLayerToBeReplaced).
xRayModelsLowerLayers = []
xRayModelsUpperLayers = []
xRayModelsLayerToBeReplaced = []

#The following list contains the lower layers of the ImageNet model at different level. 
#Each element of the list is a succession of layers up to a different level in the ImageNet model.
ImageNetModelsLowerLayers = []

#The models that we are going to use:
imageNetModel = models.resnet18(pretrained = True)
xRayVisionModel = xrv.models.ResNet(weights="resnet50-res512-all")

def outputSizeOfLowerLayersOfXRayModel(lowerLayers, testImage):
    return lowerLayers(testImage).size()

def outputSizeOfLowerLayersOfImageNetModel(lowerLayers, testImage):
    return lowerLayers(testImage).size()

class MergedNetwork(torch.nn.Module):

    def __init__(self, lowerLayersOfImageNet, lowerLayersOfXRayModel, upperLayersOfXRayModel, layerOfXRayModelToBeReplaced, batchSize):
        self.lowerLayersOfImageNet = lowerLayersOfImageNet
        self.lowerLayersOfXRayModel = lowerLayersOfXRayModel
        self.upperLayersOfXRayModel = upperLayersOfXRayModel

        self.outputSizeValuesOfLowerLayersOfXRayModel = outputSizeOfLowerLayersOfXRayModel(lowerLayersOfXRayModel, testImage)
        self.outputSizeValuesOfLowerLayersOfXRayModel[0] = batchSize

        self.outputSizeValuesOfLowerLayersOfImageNetModel = outputSizeOfLowerLayersOfImageNetModel(lowerLayersOfImageNet, testImage)
        
        #In the following, the index value of (1) should be the number of channels.
        self.numberOfInputChannelsInNewMergingLayer = outputSizeValuesOfLowerLayersOfXRayModel[1] + outputSizeValuesOfLowerLayersOfImageNetModel[1]

    def forward(self, x):
        lowerLayersImageNetOutput = self.lowerLayersOfImageNet(x)
        lowerLayersImageNetOutput = F.interpolate(lowerLayersImageNetOutput, size = outputSizeValuesOfLowerLayersOfXRayModel, mode = "bilinear")

        lowerLayersXRayModelOutput = self.lowerXRayModelBranch(x)


#asdf = 1
#testObject = MergedNetwork(asdf, asdf,asdf)
print("asdf")