import numpy as np
from MyEnum import *
import random as r
import scipy.signal as s
class FeatureMap(object):
    """Define a feature map of a CNN layer"""
    def __init__(self, inputDataSize,poolingType,activationType):
        assert isinstance(inputDataSize,int)
        assert isinstance(poolingType,PoolingType)
        assert isinstance(activationType,ActivationFunction)
        self.inputDataSize = inputDataSize
        self.poolingType = poolingType
        self.activationType = activationType
        self.DeltaSS=np.zeros((0,0))    # subsampling deltas
        self.DeltaCV=np.zeros((0,0))    # convol. layer deltas
        self.OutPutSS=np.zeros((0,0))   # subsampling layer output (half size as ActCV)
        self.ActCV =np.zeros((0,0))  # Activation function after convol
        self.APrime =np.zeros((0,0))  # Aprime (same size as ActCV)
        self.Sum =np.zeros((0,0))  # result after convol, then +  bias (same size as ActCV)
        self.Bias=r.random()*0.1 if r.random()<0.5 else r.random()*-0.1  # one bias for the feature map
        self.BiasGrad=0.0  # one bias for the feature map
    

    def Evaluate(self,inputData):
        assert isinstance(inputData,np.ndarray)
        c2Size=inputData.shape[0]
        Res=np.zeros((c2Size,c2Size))
        self.Sum=inputData+self.Bias
        if self.activationType==ActivationFunction.SIGMOID:
            self.ActCV=1/(1+np.exp(-self.Sum))
            sigmoidprime=np.vectorize(lambda x: x*(1-x))
            self.APrime= sigmoidprime(self.ActCV)
        elif self.activationType==ActivationFunction.TANH:
            self.ActCV= np.tanh(self.Sum)
            self.APrime=1-np.square(self.ActCV)
        elif self.activationType==ActivationFunction.ReLU:
            self.ActCV=np.maximum(0,self.Sum)
        if self.poolingType==PoolingType.AVGPOOLING:
            Res=self.AvgPool(self.ActCV)
        self.OutPutSS=Res
        return Res
        pass

    def AvgPool(self,M):
        assert isinstance(M,np.ndarray)
        res=np.zeros((M.shape[0]//2,M.shape[1]//2))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i][j] = (M[i*2][j*2] + M[i*2][j*2+1] + M[i * 2+1][j * 2] + M[i * 2+1][j * 2+1])/4.0
        return res

