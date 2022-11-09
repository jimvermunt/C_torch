import first_module
import Neural_Network as nrl
import pandas as pd

# https://towardsdatascience.com/introduction-to-math-behind-neural-networks-e8b60dbbdeba

# Import data to test the classifier
# https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/
dataset = pd.read_excel("C:/Users/jimve/Documents/C_torch/prototype_python/HousePricePrediction.xlsx")
print(dataset.head(3))

first_module.starting_message()

neural = nrl.neural_network(1,10)
print(neural.input_size)
print(neural.w)
print(neural.b.size)