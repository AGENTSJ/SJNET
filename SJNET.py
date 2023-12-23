import random
import numpy as np
import copy
import math
import json


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
        self.gradArr=[]
        
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
        "uses uniform distribution"

        self.previousLayer = previousLayer 
        fan_in = previousLayer.neuronCount
        upperBound =1/fan_in
        lowerBound = -upperBound
        for i in range(self.neuronCount):
            temp = []
            for j in range(previousLayer.neuronCount):
                
                temp.append(random.uniform(lowerBound,upperBound))

            self.weights.append(temp)

        for i in range(self.neuronCount):

            self.bias.append(random.uniform(lowerBound,upperBound))
            
    def loadLayer(self,weights,bias):
        "loads weights and biases into layer when using loadnetwork"
        self.weights = weights
        self.bias = bias

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
                    if self.activation=="sigmoid":nodedelta*self.neuronvals[i]*(1-self.neuronvals[i]) #handling if sigmoid is used
                    self.NodeDeltas.append(nodedelta)
            else:
                for i in range(self.neuronCount):

                    nodedelta = self.neuronvals[i]-self.Y[CurrentDataPoint][i]
                    if self.activation=="sigmoid":nodedelta*self.neuronvals[i]*(1-self.neuronvals[i]) #handling if sigmoid is used
                    self.NodeDeltas[i]=nodedelta
        else:
            # hidden layers
            if len(self.NodeDeltas)==0:
                
                for i in range(self.neuronCount):
                    nodedelta= 0
                    for j in range(AfterLayer.neuronCount):
                        
                        nodedelta = nodedelta+AfterLayer.NodeDeltas[j]*AfterLayer.weights[j][i]
                        if self.activation=="sigmoid":nodedelta*self.neuronvals[i]*(1-self.neuronvals[i]) #handling if sigmoid is used
                        
                    self.NodeDeltas.append(nodedelta)

            else:
                    for i in range(self.neuronCount):
                        nodedelta= 0
                        for j in range(AfterLayer.neuronCount):
                            
                            nodedelta = nodedelta+AfterLayer.NodeDeltas[j]*AfterLayer.weights[j][i]
                            if self.activation=="sigmoid":nodedelta*self.neuronvals[i]*(1-self.neuronvals[i]) #handling if sigmoid is used
                        self.NodeDeltas[i]=nodedelta
    
    def updateWeightsandBias(self):

        len_weight = len(self.weights[0])# based on previous layer of no(neurons) lenof weights is set
        
        #setting gradient array
        if len(self.gradArr)==0:

            for i in range(self.neuronCount):
                tempGrad=[]
                for j in range(len_weight):
                    
                   tempGrad.append (   self.NodeDeltas[i] * self.previousLayer.neuronvals[j] )
                
                self.gradArr.append(np.array(tempGrad))
        else:

            for i in range(self.neuronCount):
                tempGrad=[]
                for j in range(len_weight):
                    
                    tempGrad.append(  self.NodeDeltas[i] * self.previousLayer.neuronvals[j] )
                
                self.gradArr[i]=np.array(tempGrad)

        # performing l2 normalisation // handling exploding gradient 
        for i in range(self.neuronCount):

            for j in range(len_weight):

               
                l2norm = np.linalg.norm(self.gradArr[i][j])
                
                if l2norm>1.0:
                    self.gradArr[i][j]= self.gradArr[i][j]/l2norm
                
                new_Weight = self.weights[i][j] - self.learningRate *  self.gradArr[i][j] 
                self.weights[i][j] = new_Weight
            
            new_Bias = self.bias[i] - self.learningRate *  self.gradArr[i][j] 
            self.bias[i]=new_Bias
   
class Network:

    def __init__(self,X=None,Y=None,errorThresh = 0.001,learningRate=0.02,epoch=200):
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
                
                error=error+(1/len(self.Y))*(self.LayerArr[-1].batchNeuronVals[i][j]-self.Y[i][j])**2
        return error
      
    def Train(self):
        "Trains the network for specified epoch"
        errorrate = 9999999
        least_Error= 9999999
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

            if errorrate < least_Error:
                self.save(name="BestError")
                least_Error = errorrate
                
            
            print("error_rate",errorrate)
            
            if(errorrate < self.errorThresh):
                print("optimised")
                print(self.LayerArr[-1].batchNeuronVals)
                break
            self.LayerArr[-1].batchNeuronVals = []
    
    def predict(self,inputvals):
        "returns predicted output by the network"
        self.LayerArr[0].neuronvals = inputvals
        layercount = len(self.LayerArr)
        for i in range(layercount-1):
            self.forward(layer1=self.LayerArr[i],layer2=self.LayerArr[i+1])
        return self.LayerArr[-1].neuronvals

    def save(self,name="model"):
        net = {
            "layerCount":len(self.LayerArr),
            "neuronDistribution":[],
            "weights":[],
            "biases":[],
            "actvFns":[],
            "nodeDeltas":[]
        }
        for i in range(len(self.LayerArr)):

            net["neuronDistribution"].append(self.LayerArr[i].neuronCount)
            net["actvFns"].append(self.LayerArr[i].activation)
            if i!=0:
                net["weights"].append(self.LayerArr[i].weights)
                net["biases"].append(self.LayerArr[i].bias)
                net["nodeDeltas"].append(self.LayerArr[i].NodeDeltas)
        # print("saving.....")
        with open(f'{name}.json', 'w') as file:
            json.dump(net, file)
        # print("saved.....")

    def loadNetwork(self,network={}):  
        
        for i in range(network["layerCount"]):
            neuronCount = network["neuronDistribution"][i]
            position = i+1
            activation= network["actvFns"][i]
            weights=[]
            bias=[]

            if i==network["layerCount"]-2 : position = -1
            layer = Layer(neuronCount=neuronCount,position=position,activation=activation)

            if i!=0:
                weights = network["weights"][i-1]
                bias = network["biases"][i-1]
                layer.loadLayer(weights=weights,bias=bias)

            self.LayerArr.append(layer)

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