import sys, os
import numpy as np
import pandas as pd
import pickle

"""
Script that defines helper functions that should be globally available to all notebooks. 
"""

def MSE(targets, predictions, vector=True):
    if vector: 
         return np.mean(np.square(targets-predictions), axis=0)
            
    return np.mean(np.square(targets-predictions))

def RMSE(targets, predictions):
    """
    Calculates the root mean squared error between numpy vectors/matrices
    
    Args: 
        targets (np.ndarray): A matrix/vector of true targets
        predictions (np.ndarray): A numpy array of vector predictions
        mse (bool): If the resutls should be root or not
    
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

def split_dataset(data, target_idxs=range(3), delay=None):
    """
    Splits a numpy dataset into features and targets.
    
    Args: 
        data (np.ndarray): Full dataset including targets and features
        target_idxs (list=[1,2,3]): Indices of targets
        delay (int=None): Number of time steps to lag the data
        
    Returns: 
        (np.ndarray, np.ndarray): Tuple with X and y.
    """
    X = np.delete(data,target_idxs,axis=1)
    y = data[:,target_idxs]
    
    if delay: 
        X = X[:-delay]
        y = y[delay:]
        
    return X, y

def load_data(dummy_data=False, dummy_obs=5000):
    root_path = os.path.abspath(".").split("src")[0]
    
    if not dummy_data:
        path = root_path + "data/dataframes/"
        df_train = pd.read_pickle(path + "df_selected_train.pkl")
        df_valid = pd.read_pickle(path + "df_selected_valid.pkl")
        df_test = pd.read_pickle(path + "df_selected_test.pkl")
    else: 
        path = root_path + "data/dummy/"
        df_train = pd.read_pickle(path + "dummy_train_{0}.pkl".format(dummy_obs))
        df_valid = pd.read_pickle(path + "dummy_valid_{0}.pkl".format(dummy_obs))
        df_test = pd.read_pickle(path + "dummy_test_{0}.pkl".format(dummy_obs))
    
    return df_train, df_valid, df_test
    
def load_metadata():
    root_path = os.path.abspath(".").split("src")[0]
    path = root_path + "data/metadata/"
    
    stats = pd.read_csv(path+"stats_selected.csv", index_col=0)
    ts = np.load(path + "timestamps/dtimestamps.npy")
    ts_train = np.load(path + "timestamps/ts_train.npy")
    ts_valid = np.load(path + "timestamps/ts_valid.npy")
    ts_test = np.load(path + "timestamps/ts_test.npy")
    
    return stats, ts, ts_train, ts_valid, ts_test


def save_pickle(obj, fpath ):
    if not ".pkl" in fpath:
        fpath += ".pkl"
    
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(fpath ):
    if not ".pkl" in fpath:
        fpath += ".pkl"
        
    with open(fpath, 'rb') as f:
        return pickle.load(f)