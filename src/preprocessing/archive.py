def impute_nan(data, tau=TAU):
    """
    Set observations with abs(Z(x))>tau to NaN
    
    Args: 
        data (pandas.DataFrame): data to impute
        tau (float): threshold value
    
    Returns:
        pandas.DataFrame: the imputed dataset
    """
    
    data_imputed = data.copy()
    data_means = data.mean().values
    data_stds = data.std().values

    for col in (range(data.shape[-1])):       
        # compute z_values for each observation
        z_values = (data.iloc[:,col] - data_means[col])/data_stds[col]
        
        # find idxs exceeding threshold
        idxs = np.where(np.abs(z_values) > tau)[0]
        
        # impute with np.NaN
        data_imputed.iloc[idxs,col] = np.nan
    
    return data_imputed

def impute_cap(data, tau=TAU):
    """
    Set observations with abs(Z(x))>tau to capped value (x_new = mu +- std*tau)
    
    Args: 
        data (pandas.DataFrame): data to impute
        tau (float): threshold value
        
    Returns: 
        pandas.DataFrame: the imputed dataset
    """
    data_imputed = data.copy()
    data_means = data.mean().values
    data_stds = data.std().values
    
    cap_low, cap_high = data_means - data_stds*tau, data_means + data_stds*tau
    
    for col in (range(data.shape[-1])):       
        # compute z_values for each observation
        z_values = (data.iloc[:,col] - data_means[col])/data_stds[col]
        
        # find idxs exceeding threshold
        idxs_high = np.where(z_values > tau)[0]
        idxs_low = np.where(z_values < -tau)[0]
        
        # impute by capping the values
        data_imputed.iloc[idxs_high,col] = cap_high[col]
        data_imputed.iloc[idxs_low,col] = cap_low[col]
    
    return data_imputed


# ===================================== #
# Outlier removal #

"""
---
# (3) Handling exisiting anomalies and outliers from the dataset
<a class="anchor" id="outlier-handling"></a>

---
Faulty data can stem from multiple sources:
- Error/Noise in the sensor readings
- Actual faulty data

Faulty data is a problem for mainly these reasons: 
- We want to train a model on the normal behaviour of the compressor. Existing faulty behaviour must therefore be handled in some say, so that our model doesn't learn anomalous behaviour as normal behaviour.
- Outliers and extreme values has a large impact on how the data is scaled because it largely affect data parameters such as mean and variance. 

In effect, the faulty data can drastically bias the precision of our models. In order to avoid this we need to identify and act on the existing anomalies. 

As we're dealing with unsupervised anomaly detection, we do not have any labeled regions of faulty or healthy data. As a result, we must find alternative ways to spot the outliers in the dataset. Here, we set a fixed threshold ${\tau}$ that labels a datapoint as an extreme value or not. The procedure goes as follows: 
- We define a threshold $\tau$
- Any observation $x$ such that $\left\lvert Z(x) \right\rvert = \frac{x - \mu}{\sigma} \gt \tau$ is considered an extreme value or an anomaly. These values are set to undefined (or "NaN").
- The values are then imputed using the R-package Amelia. 
- If Amelia for some reason doesn't impute _all_ the NAs, then forward and backwards- filling is used. 

"""

def plot_tau_lims(data, tau=TAU):
    """
    Plots the input data with Z thresholds given by tau. 
    
    Args: 
        data (pandas.DataFrame): The data to plot
        tau (float): The threshold of Z 
        
    Returns: 
        nothing, but shows a plt.subplot
    """
    
    data_means = data.mean().values
    data_stds = data.std().values

    plot_from, plot_length=0, 100000
    plot_to = min(plot_from+plot_length,len(data))
    
    fig, axs = plt.subplots(nrows=data_targets.shape[-1], ncols=1, sharex=True, figsize = (15,15))
    for col in (range(data_targets.shape[-1])):
        tmp_data = data.iloc[plot_from:plot_to, col].values
        z_values = (tmp_data - data_means[col])/data_stds[col]
        idxs_outside = np.where(np.abs(z_values) > tau)[0]
        idxs_within = np.where(np.abs(z_values) <= tau)[0]

        data_within_lims = tmp_data[idxs_within]
        data_outside_lims = tmp_data[idxs_outside]
        
        ax = axs[col]
        ax.plot(idxs_within, data_within_lims, color="darkblue", marker="o", ms=1.0, lw=0)
        ax.plot(idxs_outside, data_outside_lims, color="darkorange", marker="o", ms=2.0, lw=0)
        ax.hlines(y=data_means[col]-data_stds[col]*tau, xmin=0, 
                   xmax=len(data), linestyle="dotted", color="red")
        ax.hlines(y=data_means[col]+data_stds[col]*tau, xmin=0, 
                   xmax=len(data), linestyle="dotted", color="red")
        ax.set_title("{0} with tau={1}. [{2}-{3}]".format(TAGS_TARGETS[col], tau, plot_from, plot_to))
    fig.show()

plot_tau_lims(data=data_amelia, tau=TAU)


# ===== #

def impute_nan(data, tau=TAU):
    """
    Set observations with abs(Z(x))>tau to NaN
    
    Args: 
        data (pandas.DataFrame): data to impute
        tau (afloat): threshold value
    
    Returns:
        pandas.DataFrame: the imputed dataset
    """
    
    data_imputed = data.copy()
    data_means = data.mean().values
    data_stds = data.std().values

    for col in (range(data.shape[-1])):       
        # compute z_values for each observation
        z_values = (data.iloc[:,col] - data_means[col])/data_stds[col]
        
        # find idxs exceeding threshold
        idxs = np.where(np.abs(z_values) > tau)[0]
        
        # impute with np.NaN
        data_imputed.iloc[idxs,col] = np.nan
    
    return data_imputed

data_amelia_nan = impute_nan(data=data_amelia, tau=5)
print("# NA in data_amelia:      ", np.sum(data_amelia.isna().sum()))
print("# NA in data_amelia_nan:  ", np.sum(data_amelia_nan.isna().sum()))

# ==== #
%%R -i data_amelia_nan -w 5 -h 5 --units in -r 200

# install.packages(c("Amelia", "stringr"))  ## uncomment if you need to install the libraries
library(Amelia)
library(stringr)

print(sum(is.na(data_amelia_nan)))
amelia.data <- amelia(data_amelia_nan, m = 1, parallel = "multicore")

imput <- 1
am.data <- amelia.data$imputations[[imput]]
print(sum(is.na(am.data)))
percentage_na <- round(sum(is.na(am.data)) / nrow(am.data),2)
cat(str_glue('Percentage NAs: {percentage_na}%'), '\n')

# write to file
write.table(
    am.data, 
    "../../data/amelia_data.csv",
    row.names = FALSE,
    sep = ","
)

# === #



# read the data
path=ROOT_PATH+"data/amelia_data.csv"
data_amelia_nan = pd.read_csv(path, sep=",")

data_amelia_nan.columns = data_amelia.columns

# if not all columns have 0 NA, use forward filling
if not np.all(data_amelia_nan.isna().sum() == 0):
    data_amelia = forward_fill(data_amelia_nan)
assert np.all(data_amelia_nan.isna().sum() == 0)

# delete the data file
os.remove(path)

plot_tau_lims(data_amelia_nan)


# ==== #