from CNNLayer import *
from FeatureMap import *
from Layer import *
import scipy.signal as sc
class DeepCNN(object):
    
    def __init__(self,cnnLayerList=[CNNLayer()],layerList=[Layer_CNN()],inputDataList=np.array(0),outputLabels=np.array(0)):
        self.CNNLayerList = cnnLayerList;
        self.LayerList = layerList
        self.InputDataList = inputDataList
        self.OutputLabels = outputLabels
        
        self.Flatten=np.zeros(1)

    def Evaluate(self,inputData):
        """Forward pass of input data in CNN"""
        #--in CNN
        for i in range(len(self.CNNLayerList)):
            PrevLayerOutputList=[]
            if i==0:
                PrevLayerOutputList.append(inputData)
            else:
                PrevLayerOutputList.clear()
                #get the outputs from previous layer (in each feature maps)
                for fmp in self.CNNLayerList[i-1].FeatureMapList:
                    PrevLayerOutputList.append(fmp.OutPutSS)
            self.CNNLayerList[i].Evaluate(PrevLayerOutputList)
        #-- in NN
        # flatten each feature map in the CNN layer and assemble
        # all maps into an nx1 vector
        
        outputSSSize=self.CNNLayerList[-1].FeatureMapList[0].OutPutSS.shape[0]
        
        #flatten size=outputArea*numOfFeatureMaps in last Cnn layer
        flattenSize=outputSSSize*self.CNNLayerList[-1].FeatureMapList[0].OutPutSS.shape[1]*len(self.CNNLayerList[-1].FeatureMapList)

        self.Flatten=np.zeros((flattenSize,1))#flatten array
        index=0
        for fmp in self.CNNLayerList[-1].FeatureMapList:
            size=fmp.OutPutSS.shape[0]*fmp.OutPutSS.shape[1]
            ss=fmp.OutPutSS.flatten()
            for i in range(ss.shape[0]):
                self.Flatten[index][0]=ss[i]
                index+=1
        #-----regular NN
        res=self.Flatten
        for l in self.LayerList:
            res=l.Evaluate(res)
        return res
            
    def Train(self,numEpochs,learningRate,batchSize):
        """Train the CNN+NN network"""
        trainingError=0
        for i in range(numEpochs):
            trainingError=0
            self.RandomizeInputs()
            #iterate thorugh each sample
            for j in range(self.InputDataList.shape[0]):
                #do forward pass
                #print("Forward passing")
                data=self.InputDataList[j]
                res=self.Evaluate(data)
                #error is computed as (a-y)^2
                
                trainingError += np.sum(np.square(res - self.OutputLabels[j]) )
                #print("Back prop in NN")
                # ----------Back prop in NN layers--------------
                for count in range(len(self.LayerList)-1,-1,-1):
                    layer=self.LayerList[count]
                    if count==len(self.LayerList)-1:#last layer
                        layer.Delta=layer.A-self.OutputLabels[j] #Softmax by default
                        if layer.activationType==ActivationFunction.SIGMOID or layer.activationType==ActivationFunction.TANH :
                            layer.Delta=np.multiply(layer.Delta,layer.APrime)
                        elif layer.activationType==ActivationFunction.ReLU:
                            layer.Delta[layer.Sum<=0]=0
                            pass
                    else:#not the last layer
                        #(W^T*Deltas)_prevlayer * APRime_thisLayer
                        layer.Delta=np.dot(self.LayerList[count+1].W.T , self.LayerList[count+1].Delta)
                        if layer.activationType==ActivationFunction.SIGMOID or layer.activationType==ActivationFunction.TANH :
                            layer.Delta=np.multiply(layer.Delta,layer.APrime)
                        elif layer.activationType==ActivationFunction.ReLU:
                            layer.Delta[layer.Sum<=0]=0
                    #compute gradient of weights
                    layer.GradB+=layer.Delta
                    #compute gradient of weights
                    if count==0: #first layer
                        layer.GradW+= np.dot(layer.Delta, self.Flatten.T)
                    else:
                        layer.GradW+= np.dot(layer.Delta, self.LayerList[count-1].A.T)
                # compute delta on the output of SS layer of all feature maps
                deltaSSFlat=np.dot(self.LayerList[0].W.T , self.LayerList[0].Delta)
                #print("Back prop in CNN")
                #----------Back prop in CNN  layers------------------
                # do reverse flattening and distribute the deltas on
                # each feature map's SS
                index=0
                for fmp in self.CNNLayerList[-1].FeatureMapList:
                    fmp.DeltaSS=np.zeros((fmp.OutPutSS.shape[0],fmp.OutPutSS.shape[1]))
                    for m in range(fmp.OutPutSS.shape[0]):
                        for n in range(fmp.OutPutSS.shape[1]):
                            fmp.DeltaSS[m,n]=deltaSSFlat[index,0]
                            index+=1
                    pass
                
                #----iterate each CNN layers in reverse order
                for cnnCount in range(len(self.CNNLayerList)-1,-1,-1):
                    #compute deltas on C layers - distribute deltas from SS layer
                    #then multiply by activation function
                    
                    #------reverse subsampling, compute delta*fprime
                    # (2Nx2N) <-----(NxN)
                    for fmp in self.CNNLayerList[cnnCount].FeatureMapList:
                        indexm,indexn=0,0
                        fmp.DeltaCV=np.zeros((fmp.OutPutSS.shape[0]*2,fmp.OutPutSS.shape[1]*2))
                        for m in range(fmp.DeltaSS.shape[0]):
                            indexn=0
                            for n in range(fmp.DeltaSS.shape[1]):
                                if fmp.activationType==ActivationFunction.SIGMOID or fmp.activationType==ActivationFunction.TANH:
                                    fmp.DeltaCV[indexm, indexn] = (1 / 4.0) * fmp.DeltaSS[m, n] * fmp.APrime[indexm, indexn] 
                                    fmp.DeltaCV[indexm, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[m, n] * fmp.APrime[indexm, indexn + 1]
                                    fmp.DeltaCV[indexm + 1, indexn] = (1 / 4.0) * fmp.DeltaSS[m, n] * fmp.APrime[indexm + 1, indexn]
                                    fmp.DeltaCV[indexm + 1, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[m, n] * fmp.APrime[indexm + 1, indexn + 1]
                                    pass
                                elif fmp.activationType==ActivationFunction.ReLU:
                                    fmp.DeltaCV[indexm, indexn] = (1 / 4.0) * fmp.DeltaSS[m, n]  if fmp.Sum[indexm,indexn]>0 else 0
                                    fmp.DeltaCV[indexm, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[m, n] if fmp.Sum[indexm,indexn+1]>0 else 0
                                    fmp.DeltaCV[indexm + 1, indexn] = (1 / 4.0) * fmp.DeltaSS[m, n] if fmp.Sum[indexm+1,indexn]>0 else 0
                                    fmp.DeltaCV[indexm + 1, indexn + 1] = (1 / 4.0) * fmp.DeltaSS[m, n] if fmp.Sum[indexm+1,indexn+1]>0 else 0
                                    pass
                                indexn+=2
                            indexm+=2
                            pass
                        pass
                    
                    #-------compute BiasGrad in current CNN Layer
                    for fmp in self.CNNLayerList[cnnCount].FeatureMapList:
                        fmp.BiasGrad+=np.sum(fmp.DeltaCV)
                    #----compute gradients for pxq kernels in current CNN layer
                    if cnnCount>0:# not first layer
                        for p in range(len(self.CNNLayerList[cnnCount-1].FeatureMapList)):
                            for q in range(len(self.CNNLayerList[cnnCount].FeatureMapList)):
                                inputRot180=np.rot90(self.CNNLayerList[cnnCount-1].FeatureMapList[p].OutPutSS,2,(1,0))
                                deltaCV=self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV
                                self.CNNLayerList[cnnCount].KernelGrads[p][q]+= sc.convolve2d(inputRot180,deltaCV,"valid")
                                pass
                        pass
                        #back propagate to previous CNN layer
                        for p in range(len(self.CNNLayerList[cnnCount-1].FeatureMapList)):
                            size=self.CNNLayerList[cnnCount-1].FeatureMapList[p].OutPutSS.shape[0]
                            "TO-DO: make this work with rectangular matrix"
                            self.CNNLayerList[cnnCount-1].FeatureMapList[p].DeltaSS=np.zeros((size,size))
                            for q in range(len(self.CNNLayerList[cnnCount].FeatureMapList)):
                                kernelRot180=np.rot90(self.CNNLayerList[cnnCount].Kernels[p][q],2,(1,0))
                                deltaCV=self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV
                                #full convol of delta with kernel rotated 180
                                self.CNNLayerList[cnnCount-1].FeatureMapList[p].DeltaSS+=sc.convolve2d(deltaCV,kernelRot180) 
                                pass
                            pass
                    else: 
                        #first layer connected to output
                        for p in range(1):
                            for q in range(len(self.CNNLayerList[cnnCount].FeatureMapList)):
                                inputRot180=np.rot90(self.InputDataList[j],2,(1,0))
                                deltaCV=self.CNNLayerList[cnnCount].FeatureMapList[q].DeltaCV
                                self.CNNLayerList[cnnCount].KernelGrads[p][q]+= sc.convolve2d(inputRot180,deltaCV,"valid")
                        pass
                    pass
                if j%batchSize==0:
                    self.UpdateKernelsWeightsBiases(learningRate,batchSize)
                    self.ClearGradients()
                pass
                
            if i%10==0:
                learningRate/=2
            print("epochs: %d train error: %f"%(i,trainingError))
            pass
        pass
    
    def ClearGradients(self):
        for cnnCount in range (len(self.CNNLayerList)):
            if cnnCount==0:#first layer
                for p in range(1):
                    for q in range(len(self.CNNLayerList[cnnCount].FeatureMapList)):
                        self.CNNLayerList[cnnCount].KernelGrads[p][q]=np.zeros(self.CNNLayerList[cnnCount].KernelGrads[p][q].shape)
                pass
            else:
                for p in range(len(self.CNNLayerList[cnnCount-1].FeatureMapList)):
                    for q in range(len(self.CNNLayerList[cnnCount].FeatureMapList)):
                        self.CNNLayerList[cnnCount].KernelGrads[p][q]=np.zeros(self.CNNLayerList[cnnCount].KernelGrads[p][q].shape)
                pass
            
            for fmp in self.CNNLayerList[cnnCount].FeatureMapList:
                fmp.BiasGrad=0
            pass
        for l in self.LayerList:
            l.GradW=np.zeros(l.GradW.shape)
            l.GradB=np.zeros(l.GradB.shape)
        pass

    def UpdateKernelsWeightsBiases(self,learningRate,batchSize):
        #update kernes and weights in Cnn
        for cnnCount in range(len(self.CNNLayerList)):
            #update kernels
            if cnnCount==0:#last layer
                for p in range(1):
                    for q in range(len(self.CNNLayerList[0].FeatureMapList)):
                        self.CNNLayerList[cnnCount].Kernels[p][q]-= (learningRate/batchSize)*self.CNNLayerList[cnnCount].KernelGrads[p][q]
                pass
            else:
                for p in range(len(self.CNNLayerList[cnnCount-1].FeatureMapList)):
                    for q in range(len(self.CNNLayerList[cnnCount].FeatureMapList)):
                        self.CNNLayerList[cnnCount].Kernels[p][q]-= (learningRate/batchSize)*self.CNNLayerList[cnnCount].KernelGrads[p][q]
            
            #update bias
            for fmp in self.CNNLayerList[cnnCount].FeatureMapList:
                fmp.Bias-= fmp.BiasGrad*(learningRate/batchSize)
            pass
        #update weights and bias in NN
        for l in self.LayerList:
            l.W-=l.GradW*(learningRate/batchSize)
            l.B-=l.GradB*(learningRate/batchSize)
        pass

    def RandomizeInputs(self):
        for i in range(self.InputDataList.shape[0]):
            j=random.randint(0,self.InputDataList.shape[0]-1)
            self.InputDataList[[i,j]]=self.InputDataList[[j,i]]
            self.OutputLabels[[i,j]]=self.OutputLabels[[j,i]]

