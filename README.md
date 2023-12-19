SJNET
## This repository contains SJNET.py which contains Network and Layer class 
## These can be used to create and train custom neural network

### A new Network can be initialised by 

```python
# provide the dataset X and Y 
network = Network(X=X,Y=Y,learningRate=0.0002,epoch=1000,errorThresh=3)
```
### Declare or initilise layers with its coresponding arguments
- neuronCount :number of neurons required in the layer

- position    :position of layer in network

  - Note: (1 for input layer and -1 for output layer)

- activation  :required activation function , avilable("linear","relu")

```python
#Note: (1 for input layer and -1 for output layer)
inputl = Layer(neuronCount=2,position=1)
hidden = Layer(neuronCount=10,position=2,activation="linear")
hidden2 = Layer(neuronCount=4,position=3,activation="linear")
output = Layer(neuronCount=1,position=-1,activation="linear")
```

