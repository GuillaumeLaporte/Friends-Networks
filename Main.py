#In this code, (simpleModel) is the Cifar10 network that we are talking about 
#in the report. (imageNetModel) is the ImageNet network that we are also talking about
#in the report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

#Put the following to (True) is you want to retrain the Cifar10 network (simpleModel)
trainSimpleModel = False

#Put the following to (False) if you want to train (or have not trained) all of the different ways
#of merging the Cifar10 and ImageNet networt together
everythingAlreadyTrained = True

#Put the following to (False) if the accuracies have not yet been computed
#and you want to compute them
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

        #The following two lines should deepcopy the weight parameters of the layer to be replaced
        #; to put them as parameters of the layer (G)
        listOfParameters[0][:, numberOfOutputChannelsOfLowerLayersOfImageNetModel:,:,:] = layerOfSimpleModelToBeReplaced.state_dict()["weight"]
        listOfParameters[1][:] = layerOfSimpleModelToBeReplaced.state_dict()["bias"]

        #Since we want to train layer (G), we unfreeze its parameters
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
        #This interpolation is made because the ImageNet network does not necessarily
        #output feature maps of the same sizes as the Cifar10 network (at various
        #intermediate layers)
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


#We train below the simple model that specializes on classifiying Cifar10 images. We made a test on Google Colab
#and we saw that there is no improvement in training the network over 8 epochs. The accuracy on a validation set does
#not increase over 64% (in (.train()) mode (I don't think testing using the (.train()) mode is a problem here, isn't it? Since the only goal 
#is seeing if things are improving)) after two more epochs of training.

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
    simpleModel = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Simple Model.pt")

#Below we are selecting the lower layers of the ImageNet models and the simple model; as well as the upper layers of the simple model
#We are also selecting the layer that is going to be replaced; whose parameters are going to be copied in the new layer (G)
#The logic is complicated because we have to check (in the case of the Cifar10 network)
#if a layer is a convolutional layer, and we also have to add by hand the activation functions
#(nn.ReLU()) when reconstructing the lower and upper parts of the Cifar10 network, 
#and store the layer to be replaced, and skip the layer to be replaced when we are selecting
#only the lower and upper parts of the Cifar10 network.

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
  

#Below we are freezing the layers that should not get updated during training; because we only want to train the layer (G)
#We are probably freezing and re-freezing multiple times the same layers since we don't deepcopy the layers when we select
#them.. But this works anyway..

for partsOfSimpleModelIndex in range(4):
  for child in partsOfSimpleModellist[partsOfSimpleModelIndex]["Lower Layers"]:
    for param in child.parameters():
      param.requires_grad = False

  for child in partsOfSimpleModellist[partsOfSimpleModelIndex]["Upper Layers"]:
    for param in child.parameters():
      param.requires_grad = False
      

#Below is the definition of some data that has to be passed to the MergedNetwork's constructor (the __init__ function), 
#to specify the number of output channels, the kernel sizes, and the padding type of the new merging layer (G)
#The characteristic of the layer (G) are based on the ones of the layer that it is
#replacing

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


#Below we train all of the merged networks

if everythingAlreadyTrained == False:
    for indexOfImageNetLayer in range(4):
      for indexOfSimpleNetworkLayer in range(4):

        mergedNetwork = mergedNetworks[indexOfImageNetLayer][indexOfSimpleNetworkLayer]
        criterion = nn.CrossEntropyLoss()
        optimizerForMergedNetwork = optim.SGD(mergedNetwork.parameters(), lr=0.001, momentum=0.9)

        #As we said in the report, we only train for 2 epochs
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
    
        torch.save(mergedNetwork,"D:\\School\\2023 Winter\\Term Project\\Program Output\\Merged Model ImageNet level {imageNetLevel} and Simple Model level {simpleModelLevel}.pt".format(imageNetLevel = indexOfImageNetLayer, simpleModelLevel = indexOfSimpleNetworkLayer))
        print('Finished Training')

#Below we compute all of the accuracies of the trained merged networks on a test set

if accuraciesAlreadyComputed == False:
    accuraciesForMergedNetworkTensor = torch.zeros((4,4))

    for indexOfImageNetLayer in range(4):
      for indexOfSimpleNetworkLayer in range(4):


        if everythingAlreadyTrained == False:
            mergedNetwork = mergedNetworks[indexOfImageNetLayer][indexOfSimpleNetworkLayer]
        else:
            mergedNetwork = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Merged Model ImageNet level {imageNetLevel} and Simple Model level {simpleModelLevel}.pt".format(imageNetLevel = indexOfImageNetLayer, simpleModelLevel = indexOfSimpleNetworkLayer))
  
        #We put the networks in (.eval()) mode to get the accuracies
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
        accuraciesForMergedNetworks[indexOfImageNetLayer].append( 100 * correct / total )

        accuraciesForMergedNetworkTensor[indexOfImageNetLayer, indexOfSimpleNetworkLayer] = accuraciesForMergedNetworks[indexOfImageNetLayer][indexOfSimpleNetworkLayer]

    torch.save(accuraciesForMergedNetworkTensor, "D:\\School\\2023 Winter\\Term Project\\Program Output\\Accuracies for Different Mergings.pt")

else:
    accuraciesForMergedNetworkTensor = torch.load("D:\\School\\2023 Winter\\Term Project\\Program Output\\Accuracies for Different Mergings.pt")

#Below we just print the merged models' accuracies to see them
for indexOfImageNetLayer in range(4):
    for indexOfSimpleNetworkLayer in range(4):

        print("Merged Model ImageNet level {imageNetLevel} and Simple Model level {simpleModelLevel}.pt".format(imageNetLevel = indexOfImageNetLayer, simpleModelLevel = indexOfSimpleNetworkLayer))
        print("Corresponding accuracy: ", accuraciesForMergedNetworkTensor[indexOfImageNetLayer, indexOfSimpleNetworkLayer])
