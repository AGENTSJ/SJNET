<div align="center">
  <h1 style="font-size: 3em;">SJNET</h1>
</div>

## This repository includes SJNET.py, which contains the Network and Layer classes.
### These can be used to create and train custom neural network

### Import SJNET 

```python
# download SJNET.py then continue
from SJNET import Network ,Layer
```

### A new Network can be initialized by 

```python
# provide the dataset X and Y
# each element is Y must be an array [] , Y = [[y1..y1n],[y2..y2n]...[yN..yNn]]

network = Network(X=X,Y=Y,learningRate=0.0002,epoch=1000,errorThresh=3)
```
### Declare or initialize layers with its coresponding arguments 

- neuronCount : number of neurons required in the layer

- position    : position of layer in network

  - Note : (1 for input layer and -1 for output layer)

- activation  : required activation function , avilable("linear","relu")

```python
#Note: position must be (1 for input layer and -1 for output layer)
inputLayer = Layer(neuronCount=2,position=1)
hidden = Layer(neuronCount=10,position=2,activation="linear")
hidden2 = Layer(neuronCount=4,position=3,activation="linear")
output = Layer(neuronCount=1,position=-1,activation="linear")
```
### Add declared layers into the network 

```python
#add them in order (inputLayer->first , outputLayer->last)
network.add(layer=inputLayer)
network.add(layer=hidden)
network.add(layer=hidden2)
network.add(layer=output)
```
### Compile or initilise the network setup

```python
network.compile()
```
### Train the model

```python
network.Train()
```
### Save the model

```python
# saved as json
network.save(name="testModel")
```
### Load the model

```python
Model = {}
with open('./savedmodels/testModel.json', 'r') as file:
    Model = json.load(file)

#Loads the  network topology weights and biases 
network.loadNetwork(network=Model)
```

### predict with model

```python
pred_val = network.predict(inputvals=[3.1, 2.5])
```
### How to train and save a model -> Training_and_Saving_Model.py 
### How to load and use the model -> Loading_Pre_Trained_Model.py
#### It uses stochastic gradient descent, just in case you are curious.


