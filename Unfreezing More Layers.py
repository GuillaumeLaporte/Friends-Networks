# Disclaimer: we did not have time to make the code efficient. So we are redefining everything just to select a specific
# network out of all of the defined ones. We are then looping over this to recreate networks that will have each different layers unfreezed.. Sorry.
# The layers that are unfreezed are only the ones of the Cifar10 network. We don't unfreezed any layers of the ImageNet network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

trainSimpleModel = False
everythingAlreadyTrained = False
accuraciesAlreadyComputed = False

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

#The following model architecture is taken from  https://github.com/Xinyi6/CIFAR10-CNN-by-Keras/blob/master/lic/model2_3.ipynb

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


#The training loops, the code for loading the Cifar10 dataset and computing the accuracy is taken from the following Pytorch tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#We load below the Cifar10 dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='D:\\School\\2023 Winter\\Term Project\\Program Output', train=True,
                                        download=False, transform=transform)

device = torch.device("cuda")

#trainset = (torch.from_numpy(trainset.data)).to(device)
#dataInTrainSet = []
#for image in trainset.data:
#    dataInTrainSet.append(torch.tensor(image).to(device))
    
#trainset.data = dataInTrainSet


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)


testset = torchvision.datasets.CIFAR10(root='D:\\School\\2023 Winter\\Term Project\\Program Output', train=False,
                                       download=False, transform=transform)

#dataInTestSet = []
#for image in testset.data:
#    dataInTestSet.append(torch.tensor(image).to(device))

#testset.data = dataInTestSet



testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# The merging location that we start from is at level (2) in the ImageNet network (starting from (0)), and at level (1) in the
# Cifar10 network (starting from (0)). Level (1) in the Cifar10 network corresponds to the 3th convolutional layer
listOfUnfreezingPossibilities = [[2],[2,3],[0],[0,2],[0,2,3]]
# We start below the big (for) loop that is going to train each possibilities of unfreezing various layers of the Cifar10 network under and above the merging
# layer (G). In this case, the merging layer (G) replaces the 3th convolutional layer in the Cifar10 network

accuraciesForUnfreezingPossibilities = torch.zeros((5))

for (unfreezingPossibilityNumber, unfreezingPossibility) in enumerate(listOfUnfreezingPossibilities):
    #We train below the simple model that specializes on classifiying Cifar10 images. We made a test on Google Colab
    #and we saw that there is no improvement in training the network over 8 epochs. The accuracy on a validation set does
    #not increase over 64% (in (.train()) mode (I don't think testing using the (.train()) mode is a problem here, isn't it? Since the only goal 
    #is seeing if things are improving)) after two more epochs of training

    if trainSimpleModel == True:
        simpleModel = Net()


        #simpleModel.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(simpleModel.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(8):  # loop over the dataset multiple times

            print("Time now: ", time.perf_counter())

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = simpleModel(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                #print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training the Simple Model')
        torch.save(simpleModel,"D:\\School\\2023 Winter\\Term Project\\Program Output\\Simple Model.pt")

    else:
        #We are reloading the Cifar10 network (simpleModel) each time to be sure we are starting from the same initial model
        # for each posibilities of unfreezing the layers. We don't want to train a model for a certain possibility; and then continue training it for 
        # a different possibility: this is not the goal. The goal is testing each possibilities independently. 
        simpleModel = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Simple Model.pt")


    # Below we freeze and unfreeze the relevant biases weights and kernel weights of the Cifar10 network (simpleModel)
    # We first freeze all parameters:
    for parameter in simpleModel.parameters():
        parameter.requires_grad = False
    
    # Below we find the indices numbers in the parameters list of the Cifar10 network (simpleModel)
    unfreezingPossibilityParametersIndices = []
    for unfreezingPossibilityLevel in unfreezingPossibility:
      unfreezingPossibilityParametersIndices.append(2*unfreezingPossibilityLevel + 2) #This possibility is for kernel weights
      unfreezingPossibilityParametersIndices.append(2*unfreezingPossibilityLevel + 3) #This possibility is for the bias weights

    # We than unfreeze the relevant parameters
    for (parameterNumber, parameter) in enumerate(simpleModel.parameters()):
      if (parameterNumber in unfreezingPossibilityParametersIndices):
        parameter.requires_grad = True #We unfreeze the layers that needs to be trained

    #Below we are selecting the lower layers of the ImageNet models and the simple model; as well as the upper layers of the simple model
    #We are also selecting the layer that is going to be replaced; whose parameters are going to be copied in the new layer (G)

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
  
      

    #Below is the definition of some data that has to be passed to the MergedNetwork's constructor (the __init__ function), 
    #to specify the number of output channels, the kernel sizes, and the padding type of the new merging layer (G)

    specificationsOfLayerToBeReplaced0 = {}
    specificationsOfLayerToBeReplaced0["outChannels"] = 64
    specificationsOfLayerToBeReplaced0["kernelSize"] = 3
    specificationsOfLayerToBeReplaced0["padding"] = "same"

    specificationsOfLayerToBeReplaced1 = {}
    specificationsOfLayerToBeReplaced1["outChannels"] = 128
    specificationsOfLayerToBeReplaced1["kernelSize"] = 3
    specificationsOfLayerToBeReplaced1["padding"] = "same"

    specificationsOfLayerToBeReplaced2 = {}
    specificationsOfLayerToBeReplaced2["outChannels"] = 256
    specificationsOfLayerToBeReplaced2["kernelSize"] = 3
    specificationsOfLayerToBeReplaced2["padding"] = "same"

    specificationsOfLayerToBeReplaced3 = {}
    specificationsOfLayerToBeReplaced3["outChannels"] = 512
    specificationsOfLayerToBeReplaced3["kernelSize"] = 3
    specificationsOfLayerToBeReplaced3["padding"] = "same"

    specificationsOfLayersToBeReplaced = [specificationsOfLayerToBeReplaced0, specificationsOfLayerToBeReplaced1, specificationsOfLayerToBeReplaced2, specificationsOfLayerToBeReplaced3]

    mergedNetworks = []
    accuraciesForMergedNetworks = []
    for indexOfImageNetLayer in range(4):
      accuraciesForMergedNetworks.append([])
      mergedNetworks.append([])


    #Below we construct the merged networks from the lower layers of the ImageNet model and the layers of the simple model
    for indexOfImageNetLayer in range(4):
      for indexOfSimpleNetworkLayer in range(4):
        mergedNetworks[indexOfImageNetLayer].append(MergedNetwork(partialImageNetModels[indexOfImageNetLayer], partsOfSimpleModellist[indexOfSimpleNetworkLayer]["Lower Layers"], partsOfSimpleModellist[indexOfSimpleNetworkLayer]["Upper Layers"], partsOfSimpleModellist[indexOfSimpleNetworkLayer]["Layer to be replaced"], specificationsOfLayersToBeReplaced[indexOfSimpleNetworkLayer], 4))


    #Below we train the merged network for specific unfreezed layers

    if everythingAlreadyTrained == False:
            
            #This is the specific choice of the location of the merging between the ImageNet network and the Cifar10 network
            indexOfImageNetLayer = 2
            indexOfSimpleNetworkLayer = 1

            mergedNetwork = mergedNetworks[indexOfImageNetLayer][indexOfSimpleNetworkLayer]
            criterion = nn.CrossEntropyLoss()
            optimizerForMergedNetwork = optim.SGD(mergedNetwork.parameters(), lr=0.001, momentum=0.9)


            for epoch in range(2):  # loop over the dataset multiple times 

                print("Time now: ", time.perf_counter())

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizerForMergedNetwork.zero_grad()

                    # forward + backward + optimize
                    outputs = mergedNetwork(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizerForMergedNetwork.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:    # print every 500 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                        running_loss = 0.0
    
            torch.save(mergedNetwork,"D:\\School\\2023 Winter\\Term Project\\Program Output\\Unfreezed Model Possibility Number {unfreezingPossibilityNumberPrime}.pt".format(unfreezingPossibilityNumberPrime = unfreezingPossibilityNumber))
            print('Finished Training')

    #Below we compute all of the accuracies of the trained merged networks on a test set

    if accuraciesAlreadyComputed == False:


            if everythingAlreadyTrained == False:
                pass
            else:
                mergedNetwork = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Unfreezed Model Possibility Number {unfreezingPossibilityNumberPrime}.pt".format(unfreezingPossibilityNumberPrime = unfreezingPossibilityNumber))
  
            mergedNetwork.eval()

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = mergedNetwork(images)
                    # the class with the highest energy is what we choose as prediction

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

            accuraciesForUnfreezingPossibilities[unfreezingPossibilityNumber] = 100 * correct / total

    else:
        accuraciesForUnfreezingPossibilities = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Accuracies for Unfreezing Possibilities.pt")

torch.save(accuraciesForUnfreezingPossibilities, "D:\\School\\2023 Winter\\Term Project\\Program Output\\Accuracies for Unfreezing Possibilities.pt")

for (unfreezingPossibilityNumber, unfreezingPossibility) in enumerate(listOfUnfreezingPossibilities):
    print("Unfreezing Possibility Combination: ", unfreezingPossibility)
    print("Accuracy:", accuraciesForUnfreezingPossibilities[unfreezingPossibilityNumber])

