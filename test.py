from SJNET import Network ,Layer
import pandas as pd

data = pd.read_csv("data.csv",header=None,delimiter=",")

X=[]
Y=[]
for idx,row in enumerate(data.values):
    x=[]
    for i in range(2):
        x.append(row[i])
    X.append(x)
    Y.append([row[2]])

print(len(X))
print(Y)


inputl = Layer(neuronCount=2,position=1)
hidden = Layer(neuronCount=10,position=2,activation="linear")
hidden2 = Layer(neuronCount=4,position=3,activation="linear")
output = Layer(neuronCount=1,position=-1,activation="linear")

network = Network(X=X,Y=Y,learningRate=0.0002,epoch=1000,errorThresh=3)

network.add(layer=inputl)
network.add(layer=hidden)
network.add(layer=hidden2)
network.add(layer=output)
network.compile()
network.Train()