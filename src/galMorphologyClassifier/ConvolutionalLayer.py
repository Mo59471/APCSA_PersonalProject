class ConvolutionalLayer:

    def __init__(self, layerNum, stride, padding, kCount, kWidth, kHeight, kDepth, kWeights, cBiases, afTog, input):
        self.layerNum = layerNum # Which layer this specific instantiation correpsonds to 
        self.stride = stride # Kernel stride
        self.padding = padding # Kernel padding
        self.kCount = kCount # Number of kernels
        self.kWidth = kWidth # Kernel width
        self.kHeight = kHeight # Kernel height
        self.kDepth = kDepth # Kernel depth
        self.kWeights = kWeights # Kernel weights: Shape = (kernel, z-coord, x-coord, y-coord)
        self.cBiases = cBiases # Channel biases: Shape = (channel)
        self.afTog = afTog # Toggle activation functions
        self.input = input # Inputted feature maps: Shape = (depth, width, height)
        self.output = [[[]]] # Outputted feature maps: Shape = (depth, width, height)

    def runConvolution(self):
        # Run the convolutional process
        return self.output # Output the feature maps
    
