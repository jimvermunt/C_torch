import numpy as np

def simgoid(x):
    z = 1/(1+np.exp(-x))
    return z