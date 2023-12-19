import random
import numpy as np
import copy
import math


class Ann:

    def __init__(self,X=None,Y=None,errorThresh = 0.01,learningRate=0.02,epoch=200):
        """"
        NOTE : provide X and Y dataset with learning rate and errorthreshold 
        """""
        self.LayerArr = []
        self.X =X
        self.Y = Y
        self.errorThresh=errorThresh
        self.learningRate = learningRate
        self.epoch = epoch
             
    def bind(self,layer1=None,layer2=None):
        "sets the initial weights of the ANN"

        layer2.setInitialWeights(layer1)

    def forward(self,layer1=None,layer2=None): 
        "performs forward passing for all layers in network with activation function"

        if len(layer2.neuronvals)==0:

            for i in range(layer2.neuronCount):
                activfn = layer2.activationfn()
                layer2.neuronvals.append(activfn(np.dot(layer1.neuronvals,layer2.weights[i])+layer2.bias[i]))
                
            
            if(layer2.position==-1):
                temp =copy.deepcopy(layer2.neuronvals)
                layer2.batchNeuronVals.append(temp)

        else:
            for i in range(layer2.neuronCount):
                activfn = layer2.activationfn()
                layer2.neuronvals[i]=activfn(np.dot(layer1.neuronvals,layer2.weights[i])+layer2.bias[i])

            if(layer2.position==-1):
                temp =copy.deepcopy(layer2.neuronvals)
                layer2.batchNeuronVals.append(temp)
            
    def backPropagate(self,currentLayer=None,AfterLayer=None,CurrentDataPoint=-1):
        """""
        performs back propagation by

        1. seting nodeDelta (a function in lasyer class)
        2.Update werights and biases (a function in layer class)
        
        """""
        currentLayer.setNodeDelta(AfterLayer,CurrentDataPoint=CurrentDataPoint)
        currentLayer.updateWeightsandBias()
        
    def compile(self):
        """""
        calls bind function for every layer : initilizes weights in Ann
        CALL THIS AFTER ADDING ALL LAYER WITH ANN.add()
        """""
        
        for i in range(len(self.LayerArr)-1):

            self.bind(layer1=self.LayerArr[i],layer2=self.LayerArr[i+1])

    def BatchError(self):
        """""
        find the error after 1 epoch
        """""
        error = 0
        
        for i in range(len(self.LayerArr[-1].batchNeuronVals)):     
            for j in range(len(self.Y[i])):
                
                error=error+0.5*(self.LayerArr[-1].batchNeuronVals[i][j]-self.Y[i][j])**2
        return error
      
    def Train(self):
        "Trains the network for specified epoch"
        errorrate = 9999999 
        layercount = len(self.LayerArr)
        eph =1
        while (eph < self.epoch+1):
            print("EPOCH.....",eph)
            #forward propagation
            for N in range(len(self.X)):
                
                self.LayerArr[0].neuronvals = self.X[N]      

                for i in range(layercount-1):
                    self.forward(layer1=self.LayerArr[i],layer2=self.LayerArr[i+1])
                
                #backward propagation
                currentLayeridx = layercount-1  
                for i in range(layercount-1):
                   
                    if self.LayerArr[currentLayeridx].position ==-1:# handling output layer 
                        self.backPropagate(currentLayer=self.LayerArr[currentLayeridx],CurrentDataPoint=N)
                        currentLayeridx=currentLayeridx-1
                    else:
                        self.backPropagate(currentLayer=self.LayerArr[currentLayeridx],AfterLayer=self.LayerArr[currentLayeridx+1])
                        currentLayeridx=currentLayeridx-1
                    

            eph=eph+1       
                
            errorrate = self.BatchError()
            print("error_rate",errorrate)
            
            if(errorrate < self.errorThresh):
                print("optimised")
                print(self.LayerArr[-1].batchNeuronVals)
                break
            self.LayerArr[-1].batchNeuronVals = []
    
    def predict(self,inputvals):
        self.LayerArr[0].neuronvals = inputvals
        layercount = len(self.LayerArr)
        for i in range(layercount-1):
            self.forward(layer1=self.LayerArr[i],layer2=self.LayerArr[i+1])
        print(self.LayerArr[-1].neuronvals)
        
    def add(self,layer=None):
        """""
        use this function to add layers into network in order such that
        add input layer first and output layer last
        """""
        layer.X = self.X
        layer.Y = self.Y
        layer.errorThresh = self.errorThresh
        layer.learningRate = self.learningRate
        self.LayerArr.append(layer)



class Layer:
       
    def __init__(self,neuronCount=None,position=None,neuronvals=None,activation="linear"):
        
        self.X = None
        self.Y = None
        self.activation = activation
        self.neuronCount = neuronCount
        self.position = position
        self.neuronvals = []
        self.weights = []
        self.bias = []
        self.NodeDeltas=[] 
        self.batchNeuronVals=[]
        self.errorThresh=None
        self.learningRate = None
        
        #set by setintialweights and setnodedelta
        self.previousLayer =None
        self.AfterLayer =None

        for i in range(self.neuronCount):

            self.bias.append(random.uniform(0.0001,1))
            # self.bias.append(1)

        if position==1:
            self.neuronvals = neuronvals
    
    def activationfn(self):
        funcs ={
            "linear": lambda x :x,
            "relu" : lambda x : max(0,x),
            "sigmoid":lambda x : 1 / (1 + math.exp(-x))
        }
        return funcs[self.activation]

    def setInitialWeights(self,previousLayer):

        self.previousLayer = previousLayer

        for i in range(self.neuronCount):
            temp = []
            for j in range(previousLayer.neuronCount):
                temp.append(random.uniform(0.0001,1))


            self.weights.append(temp)
  
    def setNodeDelta(self,AfterLayer,CurrentDataPoint=-1):
        """""
        set nodedelta value for layers
        """""
        self.AfterLayer = AfterLayer
        if self.position == -1:
            # outputlayer
            if len(self.NodeDeltas)==0:
                for i in range(self.neuronCount):

                    nodedelta = self.neuronvals[i]-self.Y[CurrentDataPoint][i] 
                    self.NodeDeltas.append(nodedelta)
            else:
                for i in range(self.neuronCount):

                    nodedelta = self.neuronvals[i]-self.Y[CurrentDataPoint][i]
                    self.NodeDeltas[i]=nodedelta
        else:
            # hidden layers
            if len(self.NodeDeltas)==0:
                
                for i in range(self.neuronCount):
                    nodedelta= 0
                    for j in range(AfterLayer.neuronCount):
                        
                        nodedelta = nodedelta+AfterLayer.NodeDeltas[j]*AfterLayer.weights[j][i]
                        
                    self.NodeDeltas.append(nodedelta)

            else:
                    for i in range(self.neuronCount):
                        nodedelta= 0
                        for j in range(AfterLayer.neuronCount):
                            
                            nodedelta = nodedelta+AfterLayer.NodeDeltas[j]*AfterLayer.weights[j][i]
                        self.NodeDeltas[i]=nodedelta
    
    def updateWeightsandBias(self):

        len_weight = len(self.weights[0])# based on previous layer of no(neurons) lenof weights is set
        
        
        for i in range(self.neuronCount):

            for j in range(len_weight):
                
                new_Weight = self.weights[i][j] - self.learningRate * self.NodeDeltas[i] * self.previousLayer.neuronvals[j]
                self.weights[i][j] = new_Weight
            
            new_Bias = self.bias[i] - self.learningRate * self.NodeDeltas[i]
            self.bias[i]=new_Bias  
            