import numpy as np
import pandas as pd

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
    
    
def get_stats_properties(data, include_nans=True):
    """
    Get the statistical properties of a pandas dataframe
    
    Args: 
        data (pandas.DataFrame): Data to analyse properties of
    
    Returns:
        pandas.DataFrame: Dataframe with statistical properties of the input data
    """
    dfmeans = data.mean().tolist()
    dfmedians = data.median().tolist()
    dfstds = data.std().tolist()
    dfmaxs = data.max().tolist()
    dfmins = data.min().tolist()
    dfdiff_maxmin = np.subtract(dfmaxs,dfmins)
    dfquantiles = data.quantile([.25,.75])
    dfq1 = dfquantiles.iloc[0,:].tolist()
    dfq3 = dfquantiles.iloc[1,:].tolist()
    
    stats_matr = [dfmeans,dfmedians,dfstds,dfmaxs,dfmins,dfq1,dfq3]
    stat_cols = ["Mean", "Median", "Std", "Max", "Min", "1st Qu.", "3rd Qu."]
    
    if include_nans:
        nas = data.isna().sum().tolist()
        stats_matr.append(nas)
        stat_cols.append("NAs")
        
    return pd.DataFrame(np.array(stats_matr).T, index=data.columns, columns=stat_cols)