# import sys
# sys.path.append("D:/ACtive/My_MachineLearning_Library/ANN")
from ANN import Ann ,Layer
import pandas as pd

data = pd.read_csv("csv.csv",header=None,delimiter=",")

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

Network = Ann(X=X,Y=Y,learningRate=0.0002,epoch=1000,errorThresh=3)

Network.add(layer=inputl)
Network.add(layer=hidden)
Network.add(layer=hidden2)
Network.add(layer=output)
Network.compile()
Network.Train()