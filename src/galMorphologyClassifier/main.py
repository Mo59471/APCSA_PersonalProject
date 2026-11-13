# Mo Spiegel | APCSA | Period 4B

from pathlib import Path
import numpy as np
from PIL import Image
from TrainCNN import TrainCNN
import matplotlib.pyplot as plt

training = False # Run training

inputFeatures = [] # Features of the input data: Shape = (galaxy, x-coord, y-coord)
inputClasses = [] # Clasification of the input data that the program guesses: Shape = (galaxy)
inputProbs = [] # Probabaility for each classifiacation of the input data that the program guesses: Shape = (galaxy, two_probabilities)

galFeatures = [[[]]] # Features of the training data: Shape = (galaxy, x-coord, y-coord)
galLabels = [] # Labels of the training data: Shape = (galaxy)

weights_class = [[]] # Weights between classifier layers: Shape = (layer, receiving_neuron)
biases_class = [[]] # Biases at classifier layers: Shape = (layer, receiivng_neuron)

weights_conv = [[[[[]]]]] # Weights for each kernel in the convolutional layers: Shape = (layer, kernel, z-coord, x-coord, y-coord) 
biases_conv = [[]] # Biases for each channel in the convolutional layers: Shape = (layer, channel)

convLayers = [] # Convolutional layers
poolLayers = [] # Pooling layers
classLayers = [] # Classification layers

def crop(img): # crop the image to remove background artifcats
    delIndex1 = []
    delIndex2 = []

    cropSize = (200,200) # Dimensions of the crop
    for i in range(len(img)): 
        if i < len(img)/2-cropSize[0]/2 or i >= len(img)/2+cropSize[0]/2:
            delIndex1.append(i)
        for j in range(len(img[i])):
            if j < len(img[i])/2-cropSize[1]/2 or j >= len(img[i])/2+cropSize[1]/2:
                delIndex2.append(j)

    return np.delete(np.delete(img, delIndex1, axis = 0), delIndex2, axis=1)

def populateLayers(): # Populate the layers randomly with weights and biases
    pass

def getData(): # load training data from Galaxy Zoo
    currentDir = Path(__file__).resolve().parent # access the current directory
    imgDir = currentDir.parent / "images" # access the sibling images directory

    # get image features
    for i in np.concatenate((np.arange(3,10,1), np.arange(11,20,1))):
        img = Image.open(imgDir / f"{i}.jpg").convert('L') # .convert('L') shifts to grayscale
        inputFeatures.append(crop(np.asarray(img).tolist())/255) # /255 to convert 8 bit values into 0-1
        print(inputFeatures[len(inputFeatures)-1].shape) 
    
    # get image labels
    

def readParams():
    # Read the trained weights and biases, append to weights, biases, kWeights, cBiases
    pass

def train(bSize, lRate, epochNum):
    getData()
    trainCNN = TrainCNN.TrainCNN(bSize, lRate, epochNum, galFeatures, galLabels) 
    # Call TrainCNN, call getData(), pass in the data

def instantiateLayers():
    # Append new layer instantiations to each list
    pass

def main():
    if training:
        train()
    
    # Run the CNN
    readParams()
    instantiateLayers()

    # Logic for running CNN

# Temporary method calls 
getData()
plt.imshow(inputFeatures[8], cmap='gray')
plt.show()
