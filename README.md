# AP Computer Science A Personal Project
## Mo Spiegel moshespieg@gmail.com
### Documentation Roadmap
* See **planning** for documentation detailing the planning process
  * See **brainstorming.md** for the three initial ideas
***
### First Commit: Basic Neural Network for Pixel Image CLassification (basicClassifier.py)
I implemented a neural network with three layers (input layer, one hidden layer, and output layer), with 34 neurons, and 488 parameters (468 weights/connections, 20 biases). The neural network utilizes a logistic activation function, and predicts the probability that a passed in 6X4 pixel image is either an elliptical galaxy or a spiral galaxy (blob-like or disk-like). The neural network is trained using backpropagation, where weights and biases are modified layer by layer to minimize a logarithmic loss function. Weight modification involves shifting weights proportional to the partial derivative of the loss function/surface with respect to that weight (gradient descent). The partial derivatives are approximated numerically. This simple neural network proves that a pixel image classification model can be trained in Python without the use of external ML libraries. This lays a groundwork for the remainder of the project, which will focus on developing a large scale version of this, trained on real datasets, that incorporates additional rigor and a more robust ML framework.
#### Class Details:
The Neuron class represents a node in a classification neural network. It is essential to this proposed project, which hopes to leverage neural networks for galaxy morphology classification, because such neurons act as the building blocks of the large interconnected, classifier structure. This test proves that a basic implementation of a pixel-image classifier using individual Neuron instantiations and rudimentary backpropogation can succeed without the use of external ML libraries, laying the groundwork for future development.
* **Dependencies:** NumPy
* **Location:** See src/basicClassfier for the entry file and Neuron class source
***
### UML Class Diagram
![UML Class Diagram](https://github.com/Mo59471/APCSA_PersonalProject/blob/main/plannning/system-design/GalaxyMorphologyClassificationCNN_ClassDiagram.png?raw=True)
This artifact will help guide the creation/coding of the basic application framework (setting up main file, support class files, initializing data, writing placeholder methods) and eventually direct the implementation of class-main file interaction, class instantiation, support class method calls ,etc.
