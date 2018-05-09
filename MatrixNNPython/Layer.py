import numpy as np
from MyEnum import *
import random
from math import *
class Layer(object):
    def __init__(self,numNeurons,activationFunc=ActivationFunction.SIGMOID,annealingAlgo=AnnealingAlgorithm.NOT_APPLIED):
        self.activationFunc=activationFunc
        self.annealAlgo=annealingAlgo
        self.numNeurons=numNeurons
        self.weights=None
        self.bias=None
        self.delta=None
        self.fprime=None
        self.gradW=None
        self.gradB=None
        self.a=None
        self.s=None
        #for Annealing algorithm
        if self.annealAlgo==AnnealingAlgorithm.MOMENTUM:
            self.v_w=None 
            self.v_b=None
            self.gamma=0.9
        elif self.annealAlgo==AnnealingAlgorithm.ADAGRAD:
            self.G_w=None
            self.G_b=None
            self.epsilon=1e-8
        elif self.annealAlgo==AnnealingAlgorithm.ADAM:
            self.beta1=0.9
            self.beta2=0.999
            self.epsilon=1e-8

    def Construct(self,numInputs):
        """Construct the layer based on the number of inputs"""
        self.numInputs=numInputs
        self.weights=np.array([[random.random() if random.random()>0.5 else -1* random.random() for _ in range(numInputs)] for _ in range(self.numNeurons)])
        #self.weights=0.01 * np.random.randn(self.numNeurons,numInputs)
        self.bias=np.array([random.random() if random.random()>0.5 else -1* random.random() for _ in range(self.numNeurons)]).reshape((self.numNeurons,1))
        #self.bias=np.zeros((self.numNeurons,1))
        self.delta=np.zeros((self.numNeurons,1))
        self.fprime=np.zeros((self.numNeurons,1))
        self.gradW=np.zeros((self.numNeurons,numInputs))
        self.gradB=np.zeros((self.numNeurons,1))
        self.a=np.zeros((self.numNeurons,1))
        self.s=np.zeros((self.numNeurons,1))
        if self.annealAlgo==AnnealingAlgorithm.MOMENTUM:
            self.v_w=np.zeros((self.numNeurons,self.numInputs))
            self.v_b=np.zeros((self.numNeurons,1))
        elif self.annealAlgo==AnnealingAlgorithm.ADAGRAD:
            self.G_w=np.zeros((self.numNeurons,self.numInputs))
            self.G_b=np.zeros((self.numNeurons,1))
        elif self.annealAlgo==AnnealingAlgorithm.ADAM:
            self.m_w=np.zeros((self.numNeurons,self.numInputs))
            self.m_b=np.zeros((self.numNeurons,1))
            self.v_w=np.zeros((self.numNeurons,self.numInputs))
            self.v_b=np.zeros((self.numNeurons,1))

    def ComputeSum(self,inputs):
        """Compute the sum. 
        inputs: vector of floating type inputs"""
        #code for batch
        #self.s=np.dot(inputs,self.weights.T)
        #self.s+=self.bias.T
        self.s=self.weights*inputs
        
        self.s+=self.bias
        return self.s

    def Evaluate(self,inputs,sums=0):
        """Compute the sum; apply and return the output vector of layer. 
        inputs: matrix of floats
        sums: the summs of exp(s_j) for softmax
        """
        self.ComputeSum(inputs)
        #if softmax
        if self.activationFunc==ActivationFunction.SOFTMAX:
            self.a=np.exp(self.s)/sums
            #no computing the function derivative

        elif self.activationFunc==ActivationFunction.ReLU:
            self.a=np.maximum(0,self.s)

        elif self.activationFunc==ActivationFunction.TANH:
            self.a = np.tanh(self.s) 
            self.ComputeActivationFunctionDerivative()
        else:
            #if sigmoid
            #sigmoid=np.vectorize(lambda x: 1 / (1 + exp(-x)))
            self.a = 1/(1+np.exp(-self.s))
            self.ComputeActivationFunctionDerivative()
        return self.a

    def ComputeActivationFunctionDerivative(self):
        """Compute and return the activation function derivative """
        if self.activationFunc==ActivationFunction.SIGMOID:
            sigmoidprime=np.vectorize(lambda x: x*(1-x))
            self.fprime=sigmoidprime(self.a)
        elif self.activationFunc==ActivationFunction.TANH:
            self.fprime=1-np.square(self.a)
        return self.fprime

    def resetGradient(self):
        """Set the gradient matrices to 0s"""
        self.gradW=np.zeros((self.numNeurons,self.numInputs))
        self.gradB=np.zeros((self.numNeurons,1))
        if self.annealAlgo==AnnealingAlgorithm.MOMENTUM:
            #self.v_w=np.zeros((self.numNeurons,self.numInputs))
            #self.v_b=np.zeros((self.numNeurons,1))
            pass
        elif self.annealAlgo==AnnealingAlgorithm.ADAGRAD:
            pass
            #there is no reset of accumulation matrix G
            #self.G_w=np.zeros((self.numNeurons,self.numInputs))
            #self.G_b=np.zeros((self.numNeurons,1))

    def clone(self):
        l=Layer(self.numNeurons,self.activationFunc,self.annealAlgo)
        l.weights=np.array( self.weights)
        l.numInputs=self.numInputs
        l.bias=np.array(self.bias)
        
        return l

class Layer_CNN:
    """NN layer for CNN implementation"""
    def __init__(self, numNeurons=0,inputSize=0,activationType=ActivationFunction.ReLU):
        self.numNeurons=numNeurons
        self.W=np.array([[random.random()*0.1 if random.random()<.5 else random.random()*-0.1 for _ in range(inputSize)] for _ in range(numNeurons)])
        self.B=np.array([random.random()*0.1 if random.random()<.5 else random.random()*-0.1 for _ in range(numNeurons)]).reshape((numNeurons,1))
        self.Delta=np.zeros((numNeurons,1))
        self.GradW=np.zeros((numNeurons,inputSize))
        self.GradB=np.zeros((numNeurons,1))
        self.Sum=np.zeros((numNeurons,1))
        self.A=np.zeros((numNeurons,1))
        self.APrime=np.zeros((numNeurons,1))
        self.activationType=activationType

    def Evaluate(self,inputData):
        assert isinstance(inputData,np.ndarray)
        self.Sum=np.dot(self.W, inputData)+ self.B
        if self.activationType==ActivationFunction.SIGMOID:
            self.A=1/(1+np.exp(-self.Sum))
            self.APrime=np.multiply( self.A, 1-self.A)
        elif self.activationType==ActivationFunction.ReLU:
            self.A=np.maximum(0,self.Sum)
        elif self.activationType==ActivationFunction.TANH:
            self.A=np.tanh(self.Sum)
            self.APrime=1-np.square(self.Sum)
        elif self.activationType==ActivationFunction.SOFTMAX:
            sums=np.sum(np.exp(self.Sum))
            self.A=np.exp(self.Sum)/sums
        return self.A