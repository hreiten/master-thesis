import numpy as np

"""
Script that defines helper functions that should be globally available to all notebooks. 
"""

def RMSE(targets, predictions):
    """
    Calculates the root mean squared error between numpy vectors/matrices
    
    Args: 
        targets (np.ndarray): A matrix/vector of true targets
        predictions (np.ndarray): A numpy array of vector predictions
    
    Returns: 
        float: The RMSE score
    """
    return np.sqrt(np.mean(np.square(targets - predictions)))


def MAE(targets, predictions, vector=False):
    """
    Calculates the mean absolute error between vectors/matrices
    
    Args: 
        targets (np.ndarray): A matrix/vector of true targets
        predictions (np.ndarray): A numpy array of vector predictions
        vector (bool=False): boolean stating if a vector of MAEs should be returned
    """
    if vector:
        return np.mean(np.abs(targets - predictions), axis=0)
    else:
        return np.mean(np.abs(targets - predictions))
   