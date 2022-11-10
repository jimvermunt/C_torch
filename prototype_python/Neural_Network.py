# Things that are needed for a neural network

import numpy as np

# Creating model
# input size
# hidden layers
# output size
# hidden units
# batch size

def simgoid(x):
    z = 1/(1+np.exp(-x))
    return z

class neural_network:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w = np.random.randn(self.input_size,self.hidden_size)
        self.b = np.zeros((1,self.hidden_size))

    def fit(self,x,y,epoch):
        for i in range(epoch):
            for j in range(len(x)):
                # Forward propagation
                z = np.dot(x[j],self.w) + self.b
                a = simgoid(z)
                # Backpropagation
                dz = a-y[j]
                dw = np.dot(x[j].T,dz)
                db = np.sum(dz,axis=0,keepdims=True)
                # Update weights
                self.w -= dw
                self.b -= db
    
    def predict(self,x):
        z = np.dot(x,self.w) + self.b
        a = simgoid(z)
        return a

# # learning rate
# class neural_network:
#     def __init__(self,input_size,learning_rate = 0.01,hidden_layers = 1):
#         self.input_size = input_size
#         self.w = np.zeros(input_size) + 0.5
#         self.b = 0.5
#         self.learning_rate = learning_rate
#         self.hidden_layers = hidden_layers

#     def gradient_cost_function_with_respect_to_w(y_hat,y,z,x):
#         return (2/y.size) * np.sum((y_hat - y)) * 1/(1+np.exp(z)) * (1-1/(1+np.exp(-z))) * x

#     def gradient_cost_function_with_respect_to_b(y_hat,y,z):
#         return (2/y.size) * np.sum((y - y_hat)) * 1/(1+np.exp(z)) * (1-1/(1+np.exp(-z)))

#     def update_weights(self,x,y,y_hat):
#         z = np.sum(x*self.w) + self.b
#         self.w = self.w - self.learning_rate * self.gradient_cost_function_with_respect_to_w(y_hat,y,z)
#         self.b = self.b - self.learning_rate * self.gradient_cost_function_with_respect_to_b(y_hat,y,z)

#     def fit(self,x,y,epochs=10):
#         for i in range(epochs):
#             y_hat = self.predict(x)
#             self.update_weights(x,y,y_hat)

#     def predict(self,x):
#         dot_product_sum =  np.sum(x*self.w)
#         z =   dot_product_sum + self.b
#         y_hat = 1/(1+np.exp(-z))
#         return y_hat

# Output simgmoid layer is base on ReLu becaue other shit sucks as fuck

# Loss function



# Training loop
# Forward propagation: compute the predicted y and calcualte the current loss
# Backward propagation: After each epoch we set the gradients to zero before starting to do backpropagation
# Gradient decent: Will update model paramters by calling optimizer.step() fuction

