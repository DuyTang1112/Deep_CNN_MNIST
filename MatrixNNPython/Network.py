import numpy as np
import math
from Layer import *
class Network(object):
    """A faster version of NN"""
    def __init__(self,x=np.array([]),Layers=[Layer(0)],y=np.array([]),x_val=np.array([]),y_val=np.array([])):
        #dimension of x and y should be (#examples,data dimension)
        self.x=x
        self.Layers=Layers
        self.y=y
        self.x_val=x_val
        self.y_val=y_val
        self.validation_error=float('inf')
        self.training_error=0
        layernum=0
        for lr in self.Layers:
            #initialize the weights and bias matrix for each layer
            if layernum==0:#first layer
                lr.Construct(x.shape[1])
            else:
                """ other layers. 2nd dimmension of weight matrix depends on number
                of neurons in previous layer"""
                size=self.Layers[layernum-1].numNeurons
                lr.Construct(size)
            layernum+=1

    def TrainByBackProp(self,numEpochs,learningRate=0.1,LAMBDA=0.00002,gradDescType=GradDescType.STOCHASTIC,batchSize=1,tolerance=100):
        """Train the network with Back propagation Algorithm
        """
        m=self.x.shape[0]
        self.validation_error=float('inf')
        #for early stopping, indicating the number epochs to keep looking for global optimum
        tolerance_counter=0
        prevConfig=[l.clone() for l in self.Layers]
        self.learnRate=learningRate
        if gradDescType==GradDescType.BATCH:
            batchSize=m
        elif gradDescType==GradDescType.STOCHASTIC:
            batchSize=1
        if gradDescType!=GradDescType.BATCH:
            self.shuffleData()
        for j in range(numEpochs):
            error=0
            currBatchsize=0
            #shuffle data if it is not Batch
            if gradDescType!=GradDescType.BATCH:
                self.shuffleData()
            for i in range(0,m,batchSize):
                x=self.x[i:i+batchSize if i+batchSize<=m else m] #fetching whole batch of inputs as a matrix
                y=self.y[i:i+batchSize if i+batchSize<=m else m]
                currBatchsize=x.shape[0]
                #----------------do forward pass----------------
                res=self.DoForwardPass(x)
                #--------------compute error for all outputs---------
                if self.Layers[-1].activationFunc!=ActivationFunction.SOFTMAX: #if isn't softmax
                    error+=np.square(y-res[range(currBatchsize),y.flatten()]).sum()
                else:
                    #only consider the neuron that correspond to the classification
                    temp=res[range(currBatchsize),y.flatten()]
                    error+=-np.log(res[range(currBatchsize),y.flatten()]).sum()

                #--------------- compute deltas, grads on the output layer--------------
                outLayer=self.Layers[-1]
                if outLayer.activationFunc==ActivationFunction.SOFTMAX:
                    outLayer.delta=np.array( outLayer.a)
                    outLayer.delta[range(currBatchsize),y.flatten()]-=1
                    #outLayer.delta/=batchSize
                else:
                    outLayer.a[range(currBatchsize),y.flatten()]-=1
                    outLayer.delta=np.multiply(outLayer.a,outLayer.fprime)
                self.computeGradients(outLayer,self.Layers[len(self.Layers)-2],currBatchsize)#gradients

                # ------------compute deltas, grads on next layers, stop before input layer
                for layernum in range(len(self.Layers)-2,0,-1):
                    currLayer, prevLayer=self.Layers[layernum],self.Layers[layernum+1]
                    if currLayer.activationFunc!=ActivationFunction.ReLU:
                        currLayer.delta=np.multiply( prevLayer.delta.dot(prevLayer.weights),currLayer.fprime)
                    else:
                        currLayer.delta=prevLayer.delta.dot(prevLayer.weights)
                        currLayer.delta[currLayer.a<=0]=0
                    #compute gradients
                    self.computeGradients(currLayer,self.Layers[layernum-1],currBatchsize)
                    pass
                #------------- compute deltas, grads on first layer, it is connected to inputs
                    #delta
                firstlayer=self.Layers[0]
                prevLayer=self.Layers[1]
                if firstlayer.activationFunc!=ActivationFunction.ReLU:#if not ReLU
                    firstlayer.delta=np.multiply( prevLayer.delta.dot(prevLayer.weights),firstlayer.fprime)
                else:
                    firstlayer.delta=prevLayer.delta.dot(prevLayer.weights)
                    firstlayer.delta[firstlayer.a<=0]=0
                    #gradients
                temp=Layer(0)
                temp.a=x
                self.computeGradients(firstlayer,temp,currBatchsize)

                #update weights and biases
                self.UpdateWeightsAndBiases(learningRate,LAMBDA,currBatchsize)
                self.ClearGradients()
                pass

            #display error
            if j%100==0 or j==numEpochs-1:
                if self.Layers[-1].activationFunc==ActivationFunction.SOFTMAX:
                    error/=m
                    sumweights=0
                    for l in self.Layers:
                        sumweights+=np.sum(l.weights*l.weights)
                    reg_loss=.5*LAMBDA*sumweights
                    error+=reg_loss
                print("Epochs = {} Training error = {}".format(j,error),end='\t')

            #------------------computing validation error-------------------
            #----------------do forward pass----------------
            res=self.DoForwardPass(self.x_val)
            error=0
            #--------------compute error for all outputs---------
            if self.Layers[-1].activationFunc!=ActivationFunction.SOFTMAX: #if isn't softmax
                error=np.square(self.y_val-res[range(self.x_val.shape[0]),self.y_val.flatten()]).sum()
            else:
                #only consider the neuron that correspond to the classification
                temp=res[range(self.x_val.shape[0]),self.y_val.flatten()]
                error=-np.log(res[range(self.x_val.shape[0]),self.y_val.flatten()]).sum()
                error/=self.x_val.shape[0]
                sumweights=0
                for l in self.Layers:
                    sumweights+=np.sum(l.weights*l.weights)
                reg_loss=.5*LAMBDA*sumweights
                error+=reg_loss
            if j%100==0 or j==numEpochs-1:
                print("Validation error = {}".format(error))
            if error<self.validation_error:
                tolerance_counter=0
                self.validation_error=error
                prevConfig=[l.clone() for l in self.Layers]
            elif error>self.validation_error:
                tolerance_counter+=1
                if tolerance_counter>tolerance:
                    print('Early stopping ....\nFinal model error = %f'%(self.validation_error))
                    break
        self.Layers=prevConfig
        pass
                

    def shuffleData(self):
        for i in range(len(self.x)):
            j=random.randint(0,len(self.x)-1)
            self.x[[i,j]]=self.x[[j,i]]
            self.y[[i,j]]=self.y[[j,i]]

    def DoForwardPass(self,data=np.array([])):
        """             
        based on the input data, the netwok is evalauted and the
        output from the network is returned
        data: vector of inputs
        return: vector of floats which are the outputs of the network"""
        layernum=0
        for l in self.Layers:
            sums=None
            #compute the sum of exp(s_j) for the outer layer
            if l.activationFunc==ActivationFunction.SOFTMAX:
                sums=np.exp(l.ComputeSum(data)).sum(axis=1,keepdims=True)
            data=l.Evaluate(data,sums)
        return data

    def computeGradients(self,thisLayer,leftLayer,batchSize):
        
        """Compute gradients for weights and biases on thisLayer
        thisLayer: the layer to compute the gradient
        leftLayer: the layer that produce the inputs for thisLayer"""
        #accumulate gradients as we may need it for Batch and miniBatch
        # gradient is delta*outputs from the left layer
        a=leftLayer.a
        delta=thisLayer.delta
        gradW=np.dot(delta.T,a)
        gradB=np.sum(delta,axis=0,keepdims=True).T
        if thisLayer.annealAlgo==AnnealingAlgorithm.MOMENTUM:
            #thisLayer.v_w= thisLayer.gamma* thisLayer.v_w + (self.learnRate/batchSize) *gradW
            #thisLayer.v_b= thisLayer.gamma* thisLayer.v_b + (self.learnRate/batchSize) *gradB
            pass
        if thisLayer.annealAlgo==AnnealingAlgorithm.ADAGRAD:
            thisLayer.G_w+=np.dot(np.square(delta.T),np.square(a))
            thisLayer.G_b+=np.sum(np.square(delta),axis=0,keepdims=True).T
            #thisLayer.G_w/=(batchSize**2)
            #thisLayer.G_b/=(batchSize**2)
        if thisLayer.annealAlgo==AnnealingAlgorithm.ADAM:
            l=thisLayer
            l.m_w=l.beta1*l.m_w + ((1-l.beta1))*gradW
            l.m_b=l.beta1*l.m_b + ((1-l.beta1))*gradB
            gradWSq=np.dot(np.square(delta.T),np.square(a))
            gradBSq=np.sum(np.square(delta),axis=0,keepdims=True).T
            l.v_w=l.beta2*l.v_w + ((1-l.beta2))*gradWSq
            l.v_b=l.beta2*l.v_b + ((1-l.beta2))*gradBSq
        thisLayer.gradW+= gradW
        thisLayer.gradB+= gradB

    def UpdateWeightsAndBiases(self,learningRate,LAMBDA,batchSize):
        for l in self.Layers:
            #l.gradW+=LAMBDA*l.weights
            if l.annealAlgo==AnnealingAlgorithm.MOMENTUM:
                l.v_w = l.gamma * l.v_w + (learningRate/batchSize) * l.gradW
                l.v_b = l.gamma * l.v_b + (learningRate/batchSize) * l.gradB
                l.weights=l.weights-l.v_w
                l.bias=l.bias-l.v_b
            elif l.annealAlgo==AnnealingAlgorithm.ADAGRAD:
                l.weights=l.weights- np.multiply( (learningRate/batchSize)/ np.sqrt(l.G_w+l.epsilon),l.gradW) 
                l.bias=l.bias- np.multiply( (learningRate/batchSize)/ np.sqrt(l.G_b+l.epsilon),l.gradB)
            elif l.annealAlgo==AnnealingAlgorithm.ADAM:
                """
                l.m_w=l.beta1*l.m_w + ((1-l.beta1)/batchSize)*l.gradW
                l.m_b=l.beta1*l.m_b + ((1-l.beta1)/batchSize)*l.gradB
                l.v_w=l.beta2*l.v_w + ((1-l.beta2)/batchSize)*np.square(l.gradW)
                l.v_b=l.beta2*l.v_b + ((1-l.beta2)/batchSize)*np.square(l.gradB)
                #"""
                m_w_hat=l.m_w/(1-l.beta1)
                m_b_hat=l.m_b/(1-l.beta1)
                v_w_hat=l.v_w/(1-l.beta2)
                v_b_hat=l.v_b/(1-l.beta2)
                l.weights=l.weights-((learningRate/batchSize)*m_w_hat)/(np.sqrt(v_w_hat)+l.epsilon)
                l.bias=l.bias-((learningRate/batchSize)*m_b_hat)/(np.sqrt(v_b_hat)+l.epsilon)
                pass
            else:
                #no annealing algorithm
                l.weights=l.weights-(learningRate/batchSize)*l.gradW - (learningRate*LAMBDA)*l.weights
                l.bias=l.bias-(learningRate/batchSize)*l.gradB
        
    def ClearGradients(self):
        for l in self.Layers:
            l.resetGradient()

class Network_CNN:
    """NN used for CNN implementation"""
    def __init__(self, layerList=[Layer_CNN(0,0,ActivationFunction.ReLU)],inputDataList=np.zeros(1),outputLabels=np.zeros(1)):
        assert isinstance(inputDataList,np.ndarray)
        assert isinstance(outputLabels,np.ndarray)
        self.LayerList = layerList
        self.InputDataList = inputDataList
        self.OutputLabels = outputLabels
        pass
    def Evaluate(self,inputData):
        assert isinstance(inputData,np.ndarray)
        count=0
        res=None
        for l in self.LayerList:
            res=l.Evaluate(inputData) if count==0 else l.Evaluate(res)
            count+=1
        return res
    def Train(self,numEpochs,learningRate,batchSize):
        #TO-DO:implement this
        pass
    
    pass
