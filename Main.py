import torch 

#The following lists contain the X-ray model split at different levels.
#Each elements of these lists is a succession of layers. 
#The index is the position of the split. 
#A new layer will have to be added at the split position.
xRayModelsLowerLayers = []
xRayModelsUpperLayes = []

#The following list contains the lower layers of the ImageNet model at different level. 
#Each element of the list is a succession of layers up to a different level in the ImageNet model..
ImageNetModelsLowerLayers = []

def outputSizeOfLowerLayersOfXRayModel(lowerLayers, testImage):
    return lowerLayers(testImage).size()

 class MergedNetwork(nn.Module):

     def __init__(self, lowerImageNetBranch, lowerXRayModelBranch, upperXRayModelBranch):
         self.lowerImageNetBranch = lowerImageNetBranch
         self.lowerXRayModelBranch = lowerXRayModelBranch
         self.upperXRayModelBranch = upperXRayModelBranch

         def forward(self, x):
             lowerLayersImageNetOuput = self.lowerImageNetBranch(x)
             lowerLayersXRayModelOutput = self.lowerXRayModelBranch(x)

