"""
Neural Network for basic 6x4 pixel image calssification
Mo Spiegel | 4B

- 24 neurons in input layer (1 for each pixel)
- 18 neuron hidden layer (set to around 2/3 * 26, although this could vary)
- 2 neuron output layer (probability of elliptical, probability of spiral)

Trained on two rough images: one of a circular blob ("elliptical galaxy")
and one of an oblate bar/disk ("spiral galaxy")

Takes the third input image and tries to classify it

"""

import numpy as np
from Neuron import Neuron

"""
Rough  outline of how I thought about approcahing this

Outline: 

Create a neural network with randomly generated weights
- 24 neurons in input layer (1 for each pixel)
- 18 neuron hidden layer (set to around 2/3 * 26, although this could vary)
- 2 neuron output layer (probability of elliptical, probability of spiral)

Activation function: Logistic

Train the network:
- Give it either the elliptical galaxy or the spiral galaxy as an input at each iteration
- Find the log loss in the resulting values for each weight
- Backpropogate: Calculate the partial differential equation numerically for each loss function, 
  get the gradient vector by calculating loss for each weight, set a training rate, adjust weights by this
    - Apply the same principle to biases

468 total weights (connections), 20 biases (input nodes don't have biases)

"""

tRate = 0.1 # Training rate

n = 100 # number of training epochs (the network will train 100 times on both the elliptical and spiral images -> 200 total training iterations)

# pixel image of elliptical galaxy/blob: 1 = bright pixel
elliptical = [
    0,0,1,1,0,0,
    0,1,1,1,1,0,
    0,1,1,1,1,0,
    0,0,1,1,0,0
]

# spiral galaxy
spiral = [
    0,0,0,0,0,0,
    0,0,1,1,1,1,
    1,1,1,1,0,0,
    0,0,0,0,0,0
]

# # input galaaxy (should be recognized as a spiral galaxy)
# inputGal = [
#     0,0,0,0,0,0,
#     0,0,1,1,1,1,
#     1,1,1,1,1,0,
#     0,0,0,0,0,0
# ]

# Define neural network structure

# Layer 1: input nodes
neuronsL1 = np.random.choice([0,1])  # set the input neurons to the pixels in one of the galaxy images to begin with
if neuronsL1 == 0:
    neuronsL1 = elliptical
elif neuronsL1 == 1:
    neuronsL1 = spiral

# Layer 1 connections/weights 
weightsL1 =[] # 2D array: 1st dimension = neuron in the next layer that the weights are being passed to, second dimension = weights being passed into neuron

#Layer 2: hidden layer of neurons
neuronsL2 = []

# Later 2 connections/weights
weightsL2 = []

#Layer 3: outputs
neuronsL3 = [] # two output nodes, one for the probability that the galaxy is a spiral, the other for ellptical

# Get the logarithmic loss for a given neural network
def getLoss(weightsL1_temp, weightsL2_temp, biases): #biases is 2D list: 0 index = biases for hidden layer, 1 index = biases for output layer
    neuronsL2_temp = []
    neuronsL3_temp = []
    
    # Instantiate the neural network based on passed in weights and biases
    for i in range(18): 
        neuronsL2_temp.append(Neuron(neuronsL1,weightsL1_temp[i],biases[0][i]))

    for i in range(2):
        features = [j.getOutput() for j in neuronsL2_temp] #pass in outputs of hidden layer into output layer (outputs become features)
        neuronsL3_temp.append(Neuron(features,weightsL2_temp[i],biases[1][i]))
    
    # based on the current data being trained on, define the true labels that the loss is calculated from
    if neuronsL1 == elliptical:
        labels = (1,0)
    elif neuronsL1 == spiral:
        labels = (0,1)
    else:
        labels = (0,0) # these values don't matter here, since the loss is not relevant when predicting a data point with unknown labels
        
    # Calculate the logarithmic loss function: Average log loss for both output neurons relative to true classification values
    logLoss = (np.abs(labels[0] * np.log10(neuronsL3_temp[0].getOutput()) + (1-labels[0]) * np.log10(1-neuronsL3_temp[0].getOutput())) + #Average of loss from the true results
        np.abs(labels[1] * np.log10(neuronsL3_temp[1].getOutput()) + (1-labels[1]) * np.log10(1-neuronsL3_temp[1].getOutput())))/2

    return(logLoss,neuronsL3_temp[0].getOutput(),neuronsL3_temp[1].getOutput())
        

"""Training procedure """

print("Trainng Neural network: The loss/innacuraccy is printed for each training epoch, should get closer to 0 as the network learns.")
# Create neural network with random weights and biases

for i in range(18):
    for j in range(24):
        if j == 0:
            weightsL1.append([])
        weightsL1[i].append(np.random.uniform(-5,5))

for i in range(18):
    neuronsL2.append(Neuron(neuronsL1,weightsL1[i],np.random.uniform(-5,5)))

for i in range(2):
    for j in range(18):
        if j == 0:
            weightsL2.append([])
        weightsL2[i].append(np.random.uniform(-5,5))

for i in range(2):
    features = [j.getOutput() for j in neuronsL2]
    neuronsL3.append(Neuron(features,weightsL2[i],np.random.uniform(-5,5)))

# Calculate log loss for this starting, random neural network
if neuronsL1== elliptical:
    loss = (
        np.abs(1 * np.log10(neuronsL3[0].getOutput()) + (1-1) * np.log10(1-neuronsL3[0].getOutput())) +
        np.abs(0 * np.log10(neuronsL3[1].getOutput()) + (1-0) * np.log10(1-neuronsL3[1].getOutput()))
        )/2
else:
    loss = (
    np.abs(0 * np.log10(neuronsL3[0].getOutput()) + (1-0) * np.log10(1-neuronsL3[0].getOutput())) +
    np.abs(1 * np.log10(neuronsL3[1].getOutput()) + (1-1) * np.log10(1-neuronsL3[1].getOutput()))
    )/2

"""
Loop to train the Neural Network (backpropagation loop)

Outline:
- Calculate the log loss for the final two neurons
- For each weight, calculate the partial derivative numerically
    * calculate the loss shifting that weight a small bit forward
    * subtract the current loss from this
    * Divide by the magnitude of the shift
    * This will give numerical approximation for the partial deriavitve
- Adjust each weight by -1 * learning rate * partial derivative
- Do the same thing for each bias
- Recalculate the loss from forward propogation
 
"""
# train over n epochs of two galaxies (2*n total iterations)
for i in range(n*2):
    if neuronsL1 == elliptical:
        neuronsL1 = spiral
    else:
        neuronsL1 = elliptical
        
    loss = getLoss(weightsL1,weightsL2, [[N.bias for N in neuronsL2],[K.bias for K in neuronsL3]])[0] # loss is recalculated for the new image
        
    biases = ([N.bias for N in neuronsL2],[K.bias for K in neuronsL3]) # get the biases of the current neural network
    # Gradient vector that will determine how weights and biases are modified
    gradientVector = [[],[],[],[]] #2D vector: index 0 = modification to weights in L1, index 1 = modification to weights in L2, index 2 = modification to biases in L2, index 3 = modification to biases in L3
    
    # update weights in first layer
    for j in range(len(weightsL1)):
        for k in range(len(weightsL1[j])):
            weightsL1_temp = weightsL1[0:j] + [weightsL1[j][0:k] + [weightsL1[j][k] + 0.01] + weightsL1[j][k+1:]] + (weightsL1[j+1:]) # add a small shift in the weight currently being modified to get the small shift in loss, so as to approximate dJ/dw
            dJ = getLoss(weightsL1_temp,weightsL2,biases)[0]-loss # calculate small change in loss due to small change in weight
            if k == 0:
                gradientVector[0].append([tRate*(dJ/0.01)]) # Calculate dJ/dw: Partial derivative of the loss function with respect to this specific weight
            else:
                gradientVector[0][j].append(tRate*(dJ/0.01))
        
    # update biases in hidden layer
    for j in range(len(biases[0])):
        biases_temp = [biases[0][0:j] + [biases[0][j] + 0.01] + (biases[0][j+1:]), biases[1]]
        dJ = getLoss(weightsL1,weightsL2,biases_temp)[0]-loss
        gradientVector[2].append(tRate*(dJ/0.01))
    
    # update weights in second layer
    for j in range(len(weightsL2)):
        for k in range(len(weightsL2[j])):
            weightsL2_temp = weightsL2[0:j] + [weightsL2[j][0:k] + [weightsL2[j][k] + 0.01] + weightsL2[j][k+1:]] + (weightsL2[j+1:])       
            dJ = getLoss(weightsL1,weightsL2_temp,biases)[0]-loss 
            if k == 0:
                gradientVector[1].append([tRate*(dJ/0.01)])
            else:
                gradientVector[1][j].append(tRate*(dJ/0.01))
    
    # Updater biases in output layer
    for j in range(len(biases[1])):
        biases_temp = [biases[0],biases[1][0:j] + [biases[1][j] + 0.01] + (biases[1][j+1:])]
        dJ = getLoss(weightsL1,weightsL2,biases_temp)[0]-loss
        gradientVector[3].append(tRate*(dJ/0.01))
    
    # Update the weights and the biases by subtracting the gradient vector
    for j in range(len(weightsL1)):
        for k in range(len(weightsL1[j])):
            weightsL1[j][k] -= gradientVector[0][j][k]
    for j in range(len(weightsL2)):
        for k in range(len(weightsL2[j])):
            weightsL2[j][k] -= gradientVector[1][j][k]

    for j in range(len(neuronsL2)):
        neuronsL2[j].bias -= gradientVector[2][j]
    for j in range(len(neuronsL3)):
        neuronsL3[j].bias -= gradientVector[3][j]
        
    # Recalculate the loss
    loss = getLoss(weightsL1,weightsL2, [[N.bias for N in neuronsL2],[K.bias for K in neuronsL3]])[0]
    print(loss) #loss progressively gets closer to 0 as neural network trains: alternates between loss for elliptical data vs. loss for spiral data

print(f"training Complete.") 

# To test whether training is valid, take an input galaxy from the user and try to predict what it is
userRunning = True
while userRunning:
    print("To use the neural network for classification: Input a galaxy image pixel grid to classify (0 = dark pixel, 1 = bright pixel):")
    r1 = input(" Input the first row of a 6x4 image (ex. 001100):")
    r2 = input(" Input the second row of a 6x4 image (ex. 011110):")
    r3 = input(" Input the third row of a 6x4 image (ex. 011110):")
    r4 = input(" Input the fourth row of a 6x4 image (ex. 001100):")
    inputGal = [int(i) for i in r1] + [int(i) for i in r2] + [int(i) for i in r3] + [int(i) for i in r4]
    neuronsL1 = inputGal
    print(f"Elliptical Probability: {getLoss(weightsL1,weightsL2,[[N.bias for N in neuronsL2],[K.bias for K in neuronsL3]])[1]}")
    print(f"Spiral Probability: {getLoss(weightsL1,weightsL2,[[N.bias for N in neuronsL2],[K.bias for K in neuronsL3]])[2]}")
    if  getLoss(weightsL1,weightsL2,[[N.bias for N in neuronsL2],[K.bias for K in neuronsL3]])[1] > getLoss(weightsL1,weightsL2,[[N.bias for N in neuronsL2],[K.bias for K in neuronsL3]])[2] :
        print("Galaxy is most likely elliptical according to the neural network.")
    else:
        print("Galaxy is most likely spiral according to the neural network.")
    if input("Would you like to input another galaxy? type 'y'").lower() != "y":
        userRunning = False

