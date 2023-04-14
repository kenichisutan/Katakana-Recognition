import numpy as np


# Sigmoid functions
def sigmoidEstimation(x):
    return 1 / (1 + np.e ** (-x))


def sigmoidDerivative(x):
    return x * (1 - x)


