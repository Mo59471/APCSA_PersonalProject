# Mo Spiegel | APCSA | Period 4B

from TrainCNN import TrainCNN

training = False # Run training

inputFeatures = [[[]]] # Features of the input data: Shape = (galaxy, x-coord, y-coord)
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

def getData():
    # Extract data from galaxy zoo files for training
    pass

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
