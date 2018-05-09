from FeatureMap import *
import random
from scipy import signal
class CNNLayer(object):
    def __init__(self,numFeatureMaps=0,numPrevLayerFeatureMaps=0,inputSize=0,kernelSize=0,poolingType=PoolingType.AVGPOOLING,activationType=ActivationFunction.ReLU):
        convOutputSize=inputSize-kernelSize+1
        self.ConvolSums=[np.zeros((convOutputSize,convOutputSize)) for _ in range(numFeatureMaps)]
        self.kernelSize=kernelSize
        self.numFeatureMaps=numFeatureMaps
        self.numPrevLayerFeatureMaps=numPrevLayerFeatureMaps
        self.ConvolResults=[[np.zeros((convOutputSize,convOutputSize)) for _ in range(numFeatureMaps)] for _ in range(numPrevLayerFeatureMaps)]
        self.Kernels=[[np.zeros((kernelSize,kernelSize)) for _ in range(numFeatureMaps)] for _ in range(numPrevLayerFeatureMaps)]
        self.KernelGrads=[[np.zeros((kernelSize,kernelSize)) for _ in range(numFeatureMaps)] for _ in range(numPrevLayerFeatureMaps)]        
        self.FeatureMapList=[FeatureMap(convOutputSize,poolingType,activationType) for i in range(numFeatureMaps)]
        self.InitializeKernels()

    def InitializeKernels(self):
        for i in range(len(self.Kernels)):
            for j in range(len(self.Kernels[0])):
                self.InitializeKernel(self.Kernels[i][j])
        pass
    
    def InitializeKernel(self,kernel):
        assert isinstance(kernel,np.ndarray)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                kernel[i,j]=random.random()*0.1 if random.random()<0.5 else random.random()*-0.1
        pass

    def Evaluate(self,PrevLayerOutputList):
        assert isinstance(PrevLayerOutputList,list)
        
        # inputs come from outputs of previous layer
        # do convolutions with outputs of feature maps from previous layer
        for p in range(self.numPrevLayerFeatureMaps):
            for q in range(self.numFeatureMaps):
                self.ConvolResults[p][q]=signal.convolve2d(PrevLayerOutputList[p],self.Kernels[p][q],'valid')
        # add convolution results
        for q in range(len(self.FeatureMapList)):
            #clear
            self.ConvolSums[q]=np.zeros(self.ConvolSums[q].shape)
            for p in range(len(PrevLayerOutputList)):
                self.ConvolSums[q]+=self.ConvolResults[p][q]
        # evaluate each feature map i.e., perform activation after adding bias
        for i in range(len(self.FeatureMapList)):
            self.FeatureMapList[i].Evaluate(self.ConvolSums[i])
        pass


