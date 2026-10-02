class ClassifierLayer:
    
    def __init__(self, layerNum, neuronNum, weights, biases, weightedSums, afTog, input):
        self.layerNum = layerNum # Which layer this specific instantiation correpsonds to 
        self.neuronNum = neuronNum # Number of neurons in layer
        self.weights = weights # Weights: Shape = (neuron, weights)
        self.biases = biases # Biases: Shape = (neuron)
        self.weightedSums = weightedSums # Weighted sum: Shape = (neuron)
        self.afTog = afTog # Toggle activation functions
        self.input = input # Inputs for the next layer: Shape = (number_of_inputs), same for all neurons
        self.output = [] # Outputs (weighted sums passed through AF): Shape = (neuron)
        
    def runLayer(self):
        # Run the layer, perform the weighted sum calculation, perform the activation function calculation
        # Modify self.output
        return self.output
