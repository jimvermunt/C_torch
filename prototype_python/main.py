import first_module
import Neural_Network as nrl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://towardsdatascience.com/introduction-to-math-behind-neural-networks-e8b60dbbdeba

# Import data to test the classifier
# https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/
dataset = pd.read_excel("C:/Users/jimve/Documents/C_torch/prototype_python/HousePricePrediction.xlsx")
print(dataset.head(3))

first_module.starting_message()

neural = nrl.neural_network(1,1,1)

# Weigth of mouses
x = np.array([40, 60, 80, 90, 100, 20, 30, 34, 20, 60, 90, 20, 10, 15, 99 ],dtype=np.float32)
# Obese or not
y = [0,   1, 1,   1,   1,   0, 0, 0,  0,   1,  1,  0,  0,  0,  1]

neural.fit(x,y,100)
print(neural.predict(60))

plt.scatter(x,y)

line = np.linspace(0,100,100)

z = np.zeros(len(line))
for i in range(len(line)):
    z[i] = np.dot(line[i],neural.w) + neural.b

sigmoid = 1/(1+np.exp(-z))
plt.plot(line,sigmoid)
plt.show()