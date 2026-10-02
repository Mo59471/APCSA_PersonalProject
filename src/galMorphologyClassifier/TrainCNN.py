class TrainCNN:
    # Support file that trains the CNN
    
    def __init__(self, bSize, lRate, epochNum, galFeatures, galLabels):
        self.bSize = bSize # Batch size for training
        self.lRate = lRate # Learning rate for parameter adjustment
        self.epochNum = epochNum # Number of epochs to run
        self.galFeatures = galFeatures # Features of the training data: Shape = (galaxy, x-coord, y-coord)
        self.galLabels = galLabels # Labels of the training data: Shape = (galaxy)
        self.wGradVector_class = [[[]]] # Gradient vector for updating weights in the classification layers: Shape = (layer, starting_neuron, ending_neuron)
        self.bGradVector_class = [[]] # Gradient vector for updating biases in the classification layers: Shape = (layer, nueron)
        self.wGradVector_conv = [[[[[]]]]] # Gradient vector for updating weights in the kernels in the convolutional layers: Shape = (layer, kernel, z-coord, x-coord, y-coord) 
        self.bGradVector_conv = [[]] # Gradient vector for updating the biases in the channels in the convolutional layers: Shape = (layer, channel)
        self.convLayers = [] # Convolutional layers
        self.poolLayers = [] # Pooling layers
        self.classLayers = [] # Classification layers
        
    def instantiateLayers(self):
        # Append new layer instantiations to each list
        # Instantiate random weights and biases
        pass

    def forward(self):
        self.instantiateLayers()
        # Calculate the loss for a set of data in the training set, forward propogate
        # return the loss
        loss = None 
        return loss
        
    def backprop(self, loss):
        # Backpropogate through all layers, find gradient vectors analytically
        pass
    
    def trainCNN(self):
       loss = self.forward()
       self.backprop(loss)
       # Run the training procedure
       # Write to files 
