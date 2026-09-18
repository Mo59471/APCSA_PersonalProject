import numpy as np

class Neuron:
    
    def __init__(self, features, weights, bias):
        self.features = features # inputs taken from previous layer
        self.weights = weights # weight for each input
        self.bias = bias # bias (additive shift)
        self.weightedSum = 0 # weighted sum of features to be passed into activation funciton
        for i in range(len(features)):
            self.weightedSum += features[i]*weights[i] # Calculate weighted sum: Sum = weight * feature + bias (for N features)
        self.weightedSum += self.bias
    
    def getOutput(self): # output the weighted sum after passing through the logistic activation function
        if np.isinf(1+np.e**(-1*self.weightedSum)): # edge case where denominator blows up to infinity (output is NaN, should be near 0)
            return(10**-100)
        elif np.log10(1/(1+np.e**(-1*self.weightedSum))) == 0: # edge case where the denominator = 1 because e**-weightedSum is super small
            return(0.99999999)  
        else:
            return(1/(1+np.e**(-1*self.weightedSum))) # Return the logistic function of the weighted sum