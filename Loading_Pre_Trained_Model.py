from SJNET import Network ,Layer
import json

network = Network()

Model = {}
with open('testModel.json', 'r') as file:
    Model = json.load(file)

#Loads the  network topology weights and biases 
network.loadNetwork(network=Model)

predicted_Val = network.predict(inputvals=[1.6, 4.3])
#expected output : 5.1 from data.csv
print(predicted_Val)