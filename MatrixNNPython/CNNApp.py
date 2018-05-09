
from tkinter import filedialog,messagebox
from tkinter import *
import threading
from DeepCNN import *
import numpy as np
import os
from PIL import Image, ImageTk
import pickle
class CNNApp():
    """windows application of training a NN"""
    def __init__(self):
        self.learnrate=.01
        self.trained=False
        self.dcnn=None
        self.nn=None
        self.win=Tk()
        frame=Frame(self.win)
        frame.pack(pady=10)
        testFrame=Frame(self.win)
        testFrame.pack(side=TOP,pady=5)
        testFrame.pack(side=TOP,pady=5)
        loadSaveFrame=Frame(self.win)
        loadSaveFrame.pack(side=TOP,pady=5)
        statusFrame=Frame(self.win)
        statusFrame.pack(side=TOP,pady=5)

        self.trainbutton=Button(frame,text="Train a new CNN network",command=self.start)
        self.trainbutton.pack()
        
        self.testBtn=Button(testFrame, text="Test Digit",command=self.TestDigitAfterTrain)
        self.testBtn.pack(side=LEFT,padx=10)
        self.imgLabel=Label(testFrame)
        self.imgLabel.pack(pady=20)
        self.status=Label(statusFrame,text="",fg="blue")
        self.status.pack()

        self.loadButton=Button(loadSaveFrame,text="Load CNN",command=self.loadCNN)
        self.saveButton=Button(loadSaveFrame,text="Save CNN",command=self.saveCNN)
        self.loadButton.pack(side=LEFT,padx=10)
        self.saveButton.pack()
        self.win.resizable(False,False)
        self.win.geometry("360x200")
        self.win.title("Train MNIST Data")
        self.isTraining=False
        self.win.mainloop()
        
    def resetMsg(self,msg):
        self.status["text"]=msg
    def start(self):
        if self.isTraining:
            self.status["text"]="Already training!"
            threading.Timer(5,self.resetMsg,(["Training...."])).start()
            return
        threading.Thread(target=self.train).start()
        

    def train(self):
        try:
            self.status["text"]="Training...."
            self.isTraining=True
            #------------CNN configuration---------------
            input,output=self.ReadMNISTTrainingData()
            numFeatureMapsLayer1=6
            numFeatureMapsLayer2=12
            imgSize=28
            kernelSize=5
            imgSizeC2=(imgSize-kernelSize+1)//2 #12
            C1=CNNLayer(numFeatureMapsLayer1,1,imgSize,kernelSize,PoolingType.AVGPOOLING,ActivationFunction.ReLU)
            C2=CNNLayer(numFeatureMapsLayer2,numFeatureMapsLayer1,imgSizeC2,kernelSize,PoolingType.AVGPOOLING,ActivationFunction.ReLU)
            CNNList=[C1,C2]
            #-----NN configuration--------
            NNinputsSize=(((imgSizeC2-kernelSize+1)//2)**2)*numFeatureMapsLayer2
            classNum=10
            hidden_layer_neurons=50
            #numNeurons,inputsize,Activation function
            l1=Layer_CNN(hidden_layer_neurons,NNinputsSize,ActivationFunction.ReLU)
            l2=Layer_CNN(classNum,hidden_layer_neurons,ActivationFunction.SOFTMAX)
            NNLayerList=[l1,l2]
            self.dcnn=DeepCNN(CNNList,NNLayerList,np.array(input),np.array(output))
            self.dcnn.Train(22,0.5,10)
            accuracy=self.ComputeAccuracy()
            messagebox.showinfo("Accuracy","Accuracy: %.2f "%(accuracy*100))
            #--------------------------------------------
            self.status["text"]="Training finished!"
            self.trained=True
            self.isTraining=False
        except Exception as e:
            self.status["text"]=e
            threading.Timer(3,self.resetMsg,([""])).start()
            self.isTraining=False
        pass

    def ReadMNISTTrainingData(self):
        #trainDir = "D:\\Schoolwork\\Spring 2018\\Deep Learning\\Assignment 5\\data\\train"
        trainDir=filedialog.askdirectory(initialdir = "D:\\Schoolwork\\Spring 2018\\Deep Learning\\Assignment 5\\data\\train",
                                         title = "Select trainning directory")
        if trainDir=="":
            raise NotADirectoryError("Please select training directory")
        inputList=[]
        outputLabel=[]
        for fname in os.listdir(trainDir):
            #img=[[0 for _ in range(28)] for _ in range(28)]
            img=self.ReadOneImage(fname,trainDir)
            inputList.append(img)
            classLabel=int(fname[0]) #index 0 of name file gives the classification
            outLabel=[[0] if i!=classLabel else [1] for i in range(10)]
            outputLabel.append(outLabel)
            pass
        return (inputList,outputLabel)
        pass

    def ReadOneImage(self,fname,trainDir="D:\\Schoolwork\\Spring 2018\\Deep Learning\\Assignment 5\\data\\train"):
        image=Image.open(trainDir+"\\"+fname)
        img=[[0 for _ in range(image.height)] for _ in range(image.width)]
        if (not self.is_grey_scale(image)):
            image=image.convert("L")
        for i in range(image.width):
            for j in range(image.height):
                img[i][j]= image.getpixel((i,j))/255 #0 is getting R
        return img

    def TestDigitAfterTrain(self):
        if self.dcnn==None:
            self.status["text"]="Train or load the network first!"
            threading.Timer(2,self.resetMsg,([""])).start()
            return
        if self.isTraining:
            self.status["text"]="Network is being trained...."
            return
        filename=filedialog.askopenfilename(initialdir = "D:\\Schoolwork\\Spring 2018\\Deep Learning\\Assignment 5\\data\\test",title = "Select file")
        if filename!="":
            classLabel=int(filename[filename.rfind('/')+1:filename.rfind('/')+2])
            imgDp=ImageTk.PhotoImage(Image.open(filename))
            self.imgLabel.config(image = imgDp)
            self.imgLabel.image=imgDp
            npIMG=self.ReadOneImage(filename[filename.rfind('/')+1:],filename[:filename.rfind('/')])
            res=self.dcnn.Evaluate(np.array(npIMG))
            actualLabel=np.argmax(res)
            print("actual:",actualLabel)
            if classLabel==actualLabel:
                messagebox.showinfo("Matched!","It's a match. Label=%d"%(actualLabel))
            else:
                messagebox.showinfo("No matched","No matched. Actual result=%d Expected=%d"%(actualLabel,classLabel))
            pass

    def is_grey_scale(self,img):
        im=img.convert("RGB")
        w,h = im.size
        for i in range(w):
            for j in range(h):
                r,g,b = im.getpixel((i,j))
                if r != g != b: 
                    return False
        return True
    

    def ComputeAccuracy(self):
        #testDir= "D:\\Schoolwork\\Spring 2018\\Deep Learning\\Assignment 5\\data\\TestAll10000"
        testDir=filedialog.askdirectory(initialdir = "D:\\Schoolwork\\Spring 2018\\Deep Learning\\Assignment 5\\data\\TestAll10000",title = "Select test directory")
        if testDir=="":
            raise NotADirectoryError("Did not select a test directory")
        accuracycount=0
        self.status["text"]="Computing accuracy...."
        total=0
        for filename in os.listdir(testDir):
            total+=1
            Img=self.ReadOneImage(filename,testDir)
            classLabel=int(filename[0])
            res=None
            res=self.dcnn.Evaluate(np.array(Img))
            index=np.argmax(res)
            if index==classLabel:
                accuracycount+=1
        return accuracycount/total

    def loadCNN(self):
        if self.isTraining:
            self.status["text"]="Network is being trained...."
            return
        if self.dcnn!=None:
            if messagebox.askyesno("Network Existed","A network is already existed. Do you want to load the saved version?"):
                if os.path.exists(os.getcwd()+"\\CNN"):
                    with open("CNN","rb") as f:
                        self.dcnn=pickle.load(f)
                        self.status["text"]="Load network successfully"
                        threading.Timer(2,self.resetMsg,([""])).start()
                else:
                    self.status["text"]="No network found"
                    threading.Timer(2,self.resetMsg,([""])).start()
        else:
            if os.path.exists(os.getcwd()+"\\CNN"):
                    with open("CNN","rb") as f:
                        self.dcnn=pickle.load(f)
                        self.status["text"]="Load network successfully"
                        threading.Timer(2,self.resetMsg,([""])).start()
            else:
                self.status["text"]="No network found"
                threading.Timer(2,self.resetMsg,([""])).start()
        pass

    def saveCNN(self):
        if self.dcnn==None:
            self.status["text"]="Train or load the network first!"
            threading.Timer(2,self.resetMsg,([""])).start()
            return
        if self.isTraining:
            self.status["text"]="Network is being trained...."
            threading.Timer(2,self.resetMsg,(["Training...."])).start()
            return
        if os.path.exists(os.getcwd()+"\\CNN"):
            CNN=None
            with open("CNN","rb") as f:
                CNN=pickle.load(f)
            if CNN!=None:
                if not messagebox.askokcancel("Save the network?","An already existing network is found. Save the network?"):
                    return
                pass
            pass
        
        with open("CNN","wb") as f:
            pickle.dump(self.dcnn,f)
            self.status["text"]="Save network successfully"
            threading.Timer(2,self.resetMsg,([""])).start()
            pass
        pass
            
            