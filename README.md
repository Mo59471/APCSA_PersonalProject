# AP Computer Science A Personal Project
## Mo Spiegel moshespieg@gmail.com
### Documentation Guide
* See planning for documentation detailing the planning process
  * See brainstorming.md for the three initial ideas
### First Commit: Basic Neural Network for Pixel Image CLassification (basicClassifier.py)
I implemented a neural network with three layers (input layer, one hidden layer, and output layer), with 34 neurons, and 488 parameters (468 weights/connections, 20 biases). The neural network utilizes a logistic activation function, and predicts the probability that a passed in 6X4 pixel image is either an elliptical galaxy or a spiral galaxy (blob-like or disk-like). The neural network is trained using backpropagation, where weights and biases are modified layer by layer to minimize a logarithmic loss function. Weight modification involves shifting weights proportional to the partial derivative of the loss function/surface with respect to that weight (gradient descent). The partial derivatives are approximated numerically. This simple neural network proves that a pixel image classification model can be trained in Python without the use of external ML libraries. This lays a groundwork for the remainder of the project, which will focus on developing a large scale version of this, trained on real datasets.
