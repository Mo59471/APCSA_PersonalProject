class PoolingLayer:
    
    def __init__(self, wWidth, wHeight, poolTog, input):
        self.wWidth = wWidth # Pooling window width
        self.wHeight = wHeight # Pooling window height
        self.poolTog = poolTog # Toggle aggregation/pooling function (ex. max pooling, average pooling)
        self.input = [[[]]] # Inputted feature maps: Shape = (depth, width, height)
        self.output = [[[]]] # Outputted, pooled feature maps: Shape = (depth, width, height)
        
    def runPooling(self):
        # Run the pooling operation
        return self.output
