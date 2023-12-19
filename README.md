<div align="center">
  <h1 style="font-size: 3em;">SJNET</h1>
</div>

## This repository includes SJNET.py, which contains the Network and Layer classes.
### These can be used to create and train custom neural network

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
#Note: position msut be (1 for input layer and -1 for output layer)
inputl = Layer(neuronCount=2,position=1)
hidden = Layer(neuronCount=10,position=2,activation="linear")
hidden2 = Layer(neuronCount=4,position=3,activation="linear")
output = Layer(neuronCount=1,position=-1,activation="linear")
```
### Add declared layers into the network 

```python
#add them in order (inputLayer->first , outputLayer->last)
network.add(layer=inputl)
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
### save the model (Deprecated: Not fully implemented)

```python
Network.save(name="firstModel")
```
### predict with model

```python
network.predict(inputvals=[3.1, 2.5])
```
## These functions are used and can be used as reference in Ann_Stochastic_Descent2.ipynb and test.py


