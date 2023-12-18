import random
import numpy as np

dataset=[10,11]

class Layer:
    
    def __init__(self,neuronCount,init,neuronvals):
        
        self.neuronCount = neuronCount
        self.init = init
        self.neuronvals = []
        self.weights = []
        self.bias = []
        self.NodeDeltas=[] 

        for i in range(self.neuronCount):

            self.bias.append(random.uniform(0.0001,1))

        if init==1:
            self.neuronvals = neuronvals
    
    def setInitialWeights(self,previousLayer):
        
        test =[
            [1,2,3,4]
        ]
        for i in range(self.neuronCount):
            temp = []

            for j in range(previousLayer.neuronCount):
                # temp.append(random.uniform(0.0001,1))
                temp.append(test[j])

            self.weights.append(temp)

    def setNodeDelta(self,AfterLayer):
        """""
        set nodedelta value for layers
        """""
        if self.init == -1:
            # outputlayer
            for i in range(self.neuronCount):

                nodedelta = self.neuronvals[i]-dataset[i] #exepectd to change dimenston
                self.NodeDeltas.append(nodedelta)
        else:
            # hidden layers
            for i in range(self.neuronCount):

                nodedelta= 0

                for j in range(AfterLayer.neuronCount):
                    
                    nodedelta = nodedelta+AfterLayer.NodeDeltas[j]*self.weights[i][j]


                self.NodeDeltas.append(nodedelta)


    
class Ann:

    def __init__(self,layer1,layer2):

        self.layer1 = layer1
        self.layer2 = layer2

    def strap(layer1,layer2):

        layer2.setInitialWeights(layer1)


    def forward(layer1,layer2):  
        
        for i in range(layer2.neuronCount):
            
            layer2.neuronvals.append(np.dot(layer1.neuronvals,layer2.weights[i]))

    def backPropagate(self,currentLayer,AfterLayer):

        currentLayer.setNodeDelta(AfterLayer)

        pass








inputl = Layer(2,1,[2,3])
hidden = Layer(2,0,[])
output = Layer(2,-1,[])

Ann.strap(inputl,hidden)
Ann.strap(hidden,output)

print(hidden.weights)



        
