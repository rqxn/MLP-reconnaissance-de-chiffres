import numpy as np

#Fonctions d'activation que l'on va utiliser

def sigmoide(x):
            return 1.0 / (1.0 + np.exp(-x))
def deriv_sigmoide(x):
            return sigmoide(x) * (1 - sigmoide(x))

def tanh(x):
    return np.tanh(x)
def deriv_tanh(x):
    return 1 - (np.tanh(x))**2

def relu(x):
    return np.maximum(0, x)
def deriv_relu(x):
    return int(x > 0)