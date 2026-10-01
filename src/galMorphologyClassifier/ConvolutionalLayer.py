class ConvolutionalLayer:

    def __init__(self, layerNum, stride, padding, kCount, kWidth, kHeight, kDepth, kWeights, cBiases, afTog, input):
        self.layerNum = layerNum # Which layer this specific instantiation correpsonds to 
        self.stride = stride # Kernel stride
        self.padding = padding # Kernel padding
        self.kCount = kCount # Number of kernels
        self.kWidth = kWidth # Kernel width
        self.kHeight = kHeight # Kernel height
        self.kDepth = kDepth # Kernel depth
        self.kWeights = kWeights # Kernel weights: Shape = (kernel, x-coord, y-coord, z-coord)
        self.cBiases = cBiases # Channel biases: Shape = (channel)
        self.afTog = afTog # Toggle activation functions
        self.input = input # Inputted feature maps: Shape = (width, height, depth)
        self.output = [[[]]] # Outputted feature maps: Shape = (width, height, depth)

    def runConvolution(self):
        # Run the convolutional process
        return self.output