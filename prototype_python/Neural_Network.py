
# Things that are needed for a neural network

import numpy as np
import neural_math

# Creating model
# input size
# hidden layers
# output size
# hidden units
# batch size
# learning rate
class neural_network:
    def __init__(self,training_set,input_size):
        self.input_size = input_size
        self.w = np.zeros(input_size)
        self.b = 0.5

    def predict(self,x):
        dot_product_sum =  np.sum(x*self.w)
        self.z =   dot_product_sum + self.b
        y_hat = neural_math.sigmoid(self.z)

    def fit

# Output simgmoid layer is base on ReLu becaue other shit sucks as fuck

# Loss function



# Training loop
# Forward propagation: compute the predicted y and calcualte the current loss
# Backward propagation: After each epoch we set the gradients to zero before starting to do backpropagation
# Gradient decent: Will update model paramters by calling optimizer.step() fuction

