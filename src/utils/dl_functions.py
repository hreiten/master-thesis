import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
from tabulate import tabulate
import keras

from functions import MAE, color_palette

# get color codes
c, p = color_palette()

def plot_history(history, savepath=None):    
    """
    Plots the training and validation loss history of the model in the training phase.
    
    Args: 
        history (dict): history obtained by 
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

    

def predict_with_model(model, x_data, y_data, n_predictions=50, input_dim=3):
    """
    Makes multiple predicitons with a model and returns mean and std of each prediction
    """
    
    x = x_data
    if len(x_data.shape) < 3 and input_dim==3:
        x = np.expand_dims(x_data, axis=0)
    
    n_obs, n_targets = y_data.shape
    preds_matr = np.zeros((n_predictions,n_obs,n_targets))  # (n_pred, predictions, num_targets)
    err_matr = np.zeros((n_predictions,n_targets))  # (n_pred, num_targets) - one error per target per run

    for it in range(n_predictions):
        preds = model.predict(x)
        if len(preds.shape) == 3 and input_dim == 3:
            preds = preds[0]
        errs = MAE(y_data, preds, vector=True)
        
        preds_matr[it] = preds
        err_matr[it] = errs
    
    mean_predictions = np.array([np.mean(preds_matr[:,:,i], axis=0) for i in range(n_targets)]).T
    std_of_predictions = np.array([np.std(preds_matr[:,:,i], axis=0) for i in range(n_targets)]).T
    
    return mean_predictions, std_of_predictions, {'pred_matr': preds_matr, 'loss_matr': err_matr}

def predict_with_ensemble(linear_model, lstm_model, mlp_model, X, y, n_pred=50, 
                          return_predictions_matrix=False,return_stds=False,model_mses=[]):
    """
    Predict with the ensemble model, given a linear model, lstm model and MLP model.
    The linear model takes as inputs the predictions of the LSTM and MLP model. 
    This method will first make predictions with the LSTM and MLP model, then feed it to 
    the linear model that makes the final prediciton.
    """
    
    # Predict B times with the LSTM model and extract the predictions matrix
    lstm_means, lstm_stds, lstm_dict = predict_with_model(lstm_model, X, y, 
                                                          n_predictions=n_pred, input_dim=3)
    lstm_preds = lstm_dict['pred_matr']

    # Predict B times with the MLP model and extract the predictions matrix
    mlp_means, mlp_stds, mlp_dict = predict_with_model(mlp_model, X, y, 
                                                       n_predictions=n_pred, input_dim=2)
    mlp_preds = mlp_dict['pred_matr']
    
    # If uncertainty estimation is wanted
    lm_preds_matrix = np.zeros(shape=lstm_preds.shape)
    pred_unc = np.zeros(shape=lstm_preds[0].shape)
    pred_means = np.zeros(shape=lstm_preds[0].shape)
    pred_stds = np.zeros(shape=lstm_preds[0].shape)
    
    if len(model_mses)==0:
        # set the MSEs to previously calculated validation scores
        model_mses = [0.59492895, 0.09677962, 0.27542952] 
    
    # Make a prediction with the linear model for each of the rows in X. 
    # To find uncertainty-measurements, we predict with the linear model B times for each input. 
    # The standard deviance of the resulting predictions vector is then extracted empirically. 
    for t in range(len(X)):
        row_x = np.concatenate((lstm_preds[:,t,:], mlp_preds[:,t,:]), axis=1)
        pred = linear_model.predict(row_x)

        lm_preds_matrix[:,t,:] = pred
        pred_means[t,:] = np.mean(pred, axis=0)
        stds = np.std(pred, axis=0)
        pred_unc[t,:] = np.sqrt(stds**2 + model_mses)
        pred_stds[t,:] = stds
    
    if return_predictions_matrix:
        return pred_means, pred_unc, lm_preds_matrix
    
    if return_stds: 
        return pred_means, pred_unc, pred_stds, model_mses
    
    return pred_means, pred_unc

def load_keras_model(model_folder):
    """
    Loads a model
    """
    model = keras.models.load_model(model_folder + "model.h5")
    model.load_weights(model_folder + "weights.h5")
    
    return model

def get_model_maes(model, x_data, y_data, target_stds, n_predictions=50, input_dim=3):
    
    mean_preds, std_preds, pred_dict = predict_with_model(model, x_data, y_data, n_predictions, input_dim)
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
                              start_idx=0, n_obs=200, n_predictions=50, plotCI=False, input_dim=3):
    """
    Plots multiple predictions of a RNN
    """
    
    mean_preds, std_preds, pred_dict = predict_with_model(model, x_data, y_data, n_predictions, input_dim)
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
        
def plot_pred_matrix(preds_matr, preds_uncertainties, y_data, time_vec, target_tags, 
                     start_idx=12000, n_obs=200, plotCI=False, z=1.645):
        
    n_targets = y_data.shape[-1]
    n_iterations = preds_matr.shape[0]
    
    mean_preds = np.array([np.mean(preds_matr[:,:,i], axis=0) for i in range(n_targets)]).T
    std_preds = preds_uncertainties
    
    start_idx = start_idx if start_idx < len(y_data) else max(0,len(y_data)-n_obs) 
    end_idx = min(len(y_data),start_idx+n_obs)
    interval = range(start_idx,end_idx)
    
    time = time_vec[interval]
    
    for signal in range(n_targets):
        
        fig, ax = plt.subplots(1,1,figsize=(12,4), dpi=200)
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
        
        if not plotCI: # then plot individual predictions 
            #for run in range(n_iterations):
            for run in range(min(80,n_iterations)):
                preds = preds_matr[run, interval, signal]
                ax.plot(time, preds, alpha=0.25, color="gray", markersize=0, linestyle="-")
        else: 
            # calculate upper and lower bounds
            CI_low = np.subtract(mean_preds,std_preds*z)
            CI_high = np.add(mean_preds,std_preds*z)
            
            # plot it
            ax.fill_between(time,
                             CI_low[interval,signal], 
                             CI_high[interval,signal], 
                             color="gray", alpha=0.3, label="CI (z={0})".format(z))
            
            ax.plot(time,CI_low[interval,signal], c="darkgray")
            ax.plot(time,CI_high[interval,signal], c="darkgray")
        
        ax.plot(time, mean_preds[interval, signal], c=c["red"], lw=2, ls="-", ms=0, 
                 label="Mean prediction")
        ax.plot(time, y_data[interval, signal], c=c["blue_med"], lw=2, ls="-", ms=0, 
                 label="Actual")
        #plt.ylabel(target_tags[signal])
        #plt.legend(frameon=True, loc="upper right")
        #ax.legend(frameon=True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=5, frameon=False)
        
        fig.show()
    
        
def get_df_from_dicts(dicts, columns, index, texpath=None, round_digits=4):
    """
    Will make a dataframe with error metrics for each target tag out of a collection of dictionaries 
    as obtained by evaluate_model(). The model names will be collected as indexes in the dataframe, and the 
    target errors in the columns. 
    """
    
    val_maes = []
    test_maes = []
    for d in dicts:
        tmp_mae_val = [round(float(digit),round_digits) for digit in d['validation']['df']['MAE (std)'].tolist()]
        tmp_mae_test = [round(float(digit),round_digits) for digit in d['test']['df']['MAE (std)'].tolist()]

        val_maes.append(tmp_mae_val)
        test_maes.append(tmp_mae_test)

    # make df
    df_val = pd.DataFrame(np.vstack(val_maes), index=index, columns=columns)
    df_test = pd.DataFrame(np.vstack(test_maes), index=index, columns=columns)
    df_summary = pd.concat([df_val, df_test], axis=1, keys=["Validation", "Test"])

    tex = df_summary.to_latex(column_format="l" + "c"*(len(columns)*2),
                              multicolumn=True, 
                              multicolumn_format='c', 
                              bold_rows=True)
    if texpath is not None: 
        with open(texpath) as f:
            f.write(tex)

    return df_summary, tex

def get_uncertainty_df_from_dicts(dicts, columns, index, levels, texpath=None, round_digits=4):
    """
    Will make a dataframe with uncertainty and error metrics for each target tag out of a collection of dictionaries 
    as obtained by evaluate_model().
    """
    
    dataframes = []
    for d in dicts:
        df = d['validation']
        avg_maes = [round(float(digit),round_digits) for digit in df['df']['MAE (std)'].tolist()]
        exp_means = [round(float(digit),round_digits) for digit in df['df']['Expect. Mean'].tolist()]
        exp_stds = [round(float(digit),round_digits) for digit in df['df']['Expect. Stdev'].tolist()]
        df_1 = pd.DataFrame(np.column_stack([avg_maes, exp_means, exp_stds]), index = levels, columns = columns)

        df = d['test']
        avg_maes = [round(float(digit),round_digits) for digit in df['df']['MAE (std)'].tolist()]
        exp_means = [round(float(digit),round_digits) for digit in df['df']['Expect. Mean'].tolist()]
        exp_stds = [round(float(digit),round_digits) for digit in df['df']['Expect. Stdev'].tolist()]
        df_2 = pd.DataFrame(np.column_stack([avg_maes, exp_means, exp_stds]), index = levels, columns = columns)
        df_concat = pd.concat([df_1, df_2], axis=1, keys=["Validation", "Test"])
        dataframes.append(df_concat)
    
    summary_df = pd.concat(dataframes, axis=0, keys=index)
    
    tex = summary_df.to_latex(column_format="ll" + "c"*(len(columns)*2),
                              multicolumn=True, 
                              multicolumn_format='c',
                              multirow=True,
                              bold_rows=True)

    if texpath is not None: # save the file
        with open(texpath, 'w+') as f:
            f.write(tex)
    
    return summary_df, tex

def evaluate_trained_model(model, valid_tup, test_tup, target_stds, n_pred=300, input_dim=3):
    val_dict = get_model_maes(model, valid_tup[0], valid_tup[1], 
                              target_stds, n_predictions=n_pred, input_dim=input_dim)
    test_dict = get_model_maes(model, test_tup[0], test_tup[1], 
                               target_stds, n_predictions=n_pred, input_dim=input_dim)
    
    return_dict = {
        'validation': val_dict,
        'test': test_dict
    }
    
    return return_dict
