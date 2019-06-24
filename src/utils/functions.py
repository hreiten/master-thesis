import sys, os
import numpy as np
import pandas as pd
import pickle

def MSE(targets, predictions, vector=True):
    """
    Calculate the Mean Squared Error (MSE) between numpy vectors/matrices
    
    Args: 
        targets (np.ndarray): A matrix/vector of true targets
        predictions (np.ndarray): A numpy array of vector predictions
        vector (bool=False): boolean stating if a vector of MAEs should be returned
    
    Returns: 
        float: The mean squeared error.
        
    """
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
        target_idxs (list=[0,1,2]): Indices of targets
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
    """
    Load training, validation and test dataframes
    
    Args: 
        dummy_data (bool=False): If dummy datasets should be returned
        dummy_obs (int=5000): The size of the dummy dataset
    
    Returns: 
        (pd.DataFrame, pd.Dataframe, pd.Dataframe): training, validation and test dataframes. 
    """
    
    
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
        df_anomaly = []
    
    return df_train, df_valid, df_test
    
def load_metadata(return_ts=False):
    """
    Load metadata
    
    Args: 
        return_ts (bool=False): If timestamps should be returned
        
    Returns 
        Scaling stats, target tags, feature tags and potentially timestamps. 
    """
    root_path = os.path.abspath(".").split("src")[0]
    path = root_path + "data/metadata/"
    
    scaling_stats = pd.read_csv(path+"scaling_stats_selected.csv", index_col=0)
    tags_targets = pd.read_csv(path+"tags/selected_tags_targets.csv")["Name"].values
    tags_features = pd.read_csv(path+"tags/selected_tags_features.csv")["Name"].values
    
    if return_ts: 
        ts = np.load(path + "timestamps/dtimestamps.npy")
        ts_train = np.load(path + "timestamps/ts_train.npy")
        ts_valid = np.load(path + "timestamps/ts_valid.npy")
        ts_test = np.load(path + "timestamps/ts_test.npy")
    
        return scaling_stats, tags_targets, tags_features, ts, ts_train, ts_valid, ts_test
    
    return scaling_stats, tags_targets, tags_features


def save_pickle(obj, fpath):
    """
    Save any object as a .pkl file. 
    
    Args: 
        obj (Any): Any python object
        fpath (string): Path to file
    
    Returns: 
        None
    """
    if not ".pkl" in fpath:
        fpath += ".pkl"
    
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(fpath ):
    """
    Load a .pkl file
    
    Args: 
        fpath (string): Path to file
    
    Returns: 
        the loaded pickle
    """
    if not ".pkl" in fpath:
        fpath += ".pkl"
        
    with open(fpath, 'rb') as f:
        return pickle.load(f)
    
def color_palette():
    """
    Defines the color palettes used in the thesis. 
    """
    c_blue_med = "#053B89"
    c_blue_dark = "#072159"
    c_blue_light = "#1261A0"
    c_red = "#BE0209"
    c_red_light = "#e2514a"
    c_orange = "#E96F36"
    c_orange_light = "#fca55d"
    c_gray = "#282739"
    
    colors = {"blue_dark": "#072159",
              "blue_med": "#053B89",
              "blue_light": "#1261A0",
              "red": "#BE0209",
              "red_light": "#e2514a",
              "orange": "#E96F36",
              "orange_light": "#fca55d",
              "gray": "#282739"}
    palette = [c_blue_dark, c_blue_med, c_blue_light, c_red, c_red_light, c_orange, c_orange_light, c_gray]
    return colors, palette

def latexify(df):
    """
    Returns the latex table (string) of a pandas dataframe.
    
    Args: 
        df (pandas.DataFrame): The dataframe 
    
    Return: 
        string: Latex tabulate of the dataframe. 
    """
    
    multirow = type(df.index) == pd.core.indexes.multi.MultiIndex
    multicolumn = type(df.columns) == pd.core.indexes.multi.MultiIndex
    
    col_format = "ll" if multirow else "l"
    col_format += "c"*len(df.columns.levels[1])*2 if multicolumn else "c"*len(df.columns)
    
    tex = df.to_latex(column_format=col_format,
                      multicolumn=multicolumn, 
                      multicolumn_format='c',
                      multirow=multirow,
                      bold_rows=True)
    
    return tex