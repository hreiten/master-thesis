import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import keras

from functions import MAE

def plot_history(history, savepath=None):    
    """
    Plots the training and validation loss history of the model in the training phase.
    """
    
    epochs = history.epoch

    train_mae = history.history['loss']
    val_mae = history.history['val_loss']

    plt.figure()
    plt.plot(epochs, train_mae, marker='o', markersize='3.0', label=r'Training loss', color="darkred")
    plt.plot(epochs, val_mae, marker='o', markersize='3.0', label=r'Validation loss', color="darkblue")  
    plt.xlabel(r'Epoch')
    plt.ylabel(r'MAE')
    plt.legend(frameon=True)
    if savepath is not None: 
        plt.savefig(savepath)
    plt.show()

    

def predict_with_model(model, x_data, y_data, n_predictions=50):
    """
    Makes multiple predicitons with a model and returns mean and std of each prediction
    """
    
    x = x_data
    if len(x_data.shape) < 3:
        x = np.expand_dims(x_data, axis=0)
    
    n_obs, n_targets = y_data.shape
    preds_matr = np.zeros((n_predictions,n_obs,n_targets))  # (n_pred, predictions, num_targets)
    err_matr = np.zeros((n_predictions,n_targets))  # (n_pred, num_targets) - one error per target per run

    for it in range(n_predictions):
        preds = model.predict(x)[0]
        errs = MAE(y_data, preds, vector=True)
        
        preds_matr[it] = preds
        err_matr[it] = errs
    
    mean_predictions = np.array([np.mean(preds_matr[:,:,i], axis=0) for i in range(n_targets)]).T
    std_of_predictions = np.array([np.std(preds_matr[:,:,i], axis=0) for i in range(n_targets)]).T
    
    return mean_predictions, std_of_predictions, {'pred_matr': preds_matr, 'loss_matr': err_matr}

def load_keras_model(model_folder):
    """
    Loads a model
    """
    model = keras.models.load_model(model_folder + "model.h5")
    model.load_weights(model_folder + "weights.h5")
    
    return model

def get_model_maes(model, x_data, y_data, target_stds, n_predictions=50):
    
    
    mean_preds, std_preds, pred_dict = predict_with_model(model, x_data, y_data, n_predictions)
    expected_mean = np.mean(mean_preds, axis=0)
    expected_std = np.mean(std_preds, axis=0) 
    maes = np.mean(pred_dict['loss_matr'], axis=0)
    maes_unstd = maes * target_stds
    
    # summarize in dataframe
    indexes = ["FT", "TT", "PT"]
    cols = ["MAE (std)", "MAE (unstd)", "Expect. Mean", "Expect. Stdev"]
    data = np.column_stack([maes, maes_unstd, expected_mean, expected_std])
    df = pd.DataFrame(data, index=indexes, columns=cols)
    df.loc["Avg"] = df.mean()
    
    # make a string representation of the dataframe
    str_table = tabulate(df, headers='keys', tablefmt='psql', floatfmt='.5f')
    
    return_dict = {
        'df': df,
        'str_table': str_table,
        'pred_matr': pred_dict['pred_matr'], # matrix of predictions for each run
        'loss_matr': pred_dict['loss_matr']  # matrix of losses for each run
    }
    
    return return_dict

def plot_multiple_predictions(model, x_data, y_data, time_vec, target_tags, 
                              start_idx=0, n_obs=200, n_predictions=50, plotCI=False):
    """
    Plots multiple predictions of a RNN
    """
    
    mean_preds, std_preds, pred_dict = predict_with_model(model, x_data, y_data, n_predictions)
    preds_matr = pred_dict['pred_matr']
    
    n_targets = y_data.shape[-1]
    n_iterations = preds_matr.shape[0]
    
    start_idx = start_idx if start_idx < len(y_data) else max(0,len(y_data)-n_obs) 
    end_idx = min(len(y_data),start_idx+n_obs)
    interval = range(start_idx,end_idx)
    
    time = time_vec[interval]
    
    for signal in range(n_targets):
        plt.figure()
        if not plotCI: # then plot individual predictions 
            for run in range(n_iterations):
                preds = preds_matr[run, interval, signal]
                plt.plot_date(time, preds, alpha=0.3, color="gray", markersize=0, linestyle="-")
        else: 
            # calculate upper and lower bounds
            z = 1.96 #95% CI
            CI_low = np.subtract(mean_preds,std_preds*z)
            CI_high = np.add(mean_preds,std_preds*z)
            
            # plot it
            plt.fill_between(time,
                             CI_low[interval,signal], 
                             CI_high[interval,signal], 
                             color="gray", alpha=0.5, label="95% CI")
    
        y_pred_mean = mean_preds[interval, signal]
        y_signal_true = y_data[interval, signal]
        
        plt.plot_date(time, y_pred_mean, color="darkblue", 
                      linewidth=1.5, linestyle="-", markersize=0, label="Mean prediction")
        plt.plot_date(time, y_signal_true, color="darkred", 
                      linewidth=1.5, linestyle="-", markersize=0, label="True")
        plt.ylabel(target_tags[signal])
        plt.legend(frameon=True)
        plt.show()
    
