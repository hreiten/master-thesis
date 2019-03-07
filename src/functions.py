import numpy as np

"""
Script that defines helper functions that should be globally available to all notebooks. 
"""

def RMSE(targets, predictions):
    """
    Calculates the mean absolute error between vectors/matrices
    
    :param targets: a matrix/vector of true targets
    :param predictions: a matrix/vector of predictions
    """
    return np.sqrt(np.mean(np.square(targets - predictions)))


def MAE(targets, predictions, vector=False):
    """
    Calculates the mean absolute error between vectors/matrices
    
    :param targets: a matrix/vector of true targets
    :param predictions: a matrix/vector of predictions
    :param vector: boolean stating if a vector of MAEs should be returned in the case where the targets/predictions are matrices.
    """
    if vector:
        return np.mean(np.abs(targets - predictions), axis=0)
    else:
        return np.mean(np.abs(targets - predictions))
    