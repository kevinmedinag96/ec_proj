"""
This module describes the machine learning model to be used in our evolutionary problem

"""
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Model:
    def __init__(self,input_shape,weight_units_per_layer,loss_fn) -> None:
        self.model = Sequential()
        self.weight_units_per_layer = weight_units_per_layer
        self.input_shape = input_shape
        self.loss_fn = loss_fn

    def setNetwork(self,hidden_activation = "relu",
                   output_activation = "sigmoid"):
        
        self.model.add(Dense(self.weight_units_per_layer[0], input_shape=(self.input_shape,), activation=hidden_activation))

        for i in range(1,len(self.weight_units_per_layer) - 1):
            self.model.add(Dense(self.weight_units_per_layer[i], activation=hidden_activation))  


        self.model.add(Dense(self.weight_units_per_layer[-1], activation=output_activation)) 

    def compute_prediction(self,X_train):
        return self.model.predict(X_train,verbose=0)

    def compute_loss(self,y_true,y_pred):
        return self.loss_fn(y_true,y_pred)


    def updateNetworkWeights(self,weights,biases):
        """
        update model's weights and biases based on an individual decision space
        """
        #reconstruct list of weights-biases
        weights_biases = [] #layer 1 weights, layer 1 biases , ... layer k weights, layer k biases
        for id_layer in range(len(weights)):
            weights_biases.append(weights[id_layer])
            weights_biases.append(biases[id_layer])

        #load parameters in model
        self.model.set_weights(weights_biases)




if __name__ == "__main__":
    

    num_weights_per_layer = [14,3]
    nn = Model(num_weights_per_layer)
    nn.setNetwork(input_shape=5,units_per_layer=num_weights_per_layer)
    x = 10
    
    