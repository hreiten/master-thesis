import os
import numpy as np
import pandas as pd
from cognite.client import CogniteClient

class DataPreprocessor:
    
    def __init__(self, verbose=0):
        # initialize cognite client
        self.client = CogniteClient(api_key=os.environ['COGNITE_API_SECRET'])
        
        # read scaling stats and tags 
        
        root_path = os.path.abspath(".").split("src")[0]
        path = root_path + "data/metadata/"
        
        self.scaling_stats = pd.read_csv(path+"scaling_stats_selected.csv", index_col=0)
        self.target_tags = pd.read_csv(path+"tags/selected_tags_targets.csv")["Name"].values
        self.feature_tags = pd.read_csv(path+"tags/selected_tags_features.csv")["Name"].values
        self.all_tags = np.append(self.target_tags, self.feature_tags)
        
        self.verbose=verbose
    
    def read_data(self, tags, start, end, granularity="1m", aggregates=["average"], timestamps=None):
        """
        Extract time series from the Cognite API for a list of tag names.

        Args: 
            tags ([String]): A list of tags to extract time series of.
            start (datetime): A datetime object specifying the start of the interval.
            end (datetime): A datetime object specifying the end of the interval.
            granularity (String): Delta time between two samples (e.g. '1m' or '10s').
            aggregates ([String]): Data properties to extract.
            timestamps ([Datetime]): Array of timestamps. None if default behavior is wanted. 

        Returns: 
            pd.DataFrame: The dataframe of tags data.
        """

        # if no predefined timestamps array
        if timestamps is None: 
            data = self.client.datapoints.get_datapoints_frame(list(tags), 
                                                               start=start, 
                                                               end=end, 
                                                               granularity=granularity, 
                                                               aggregates=aggregates)

            columns = [name.split("|")[0] for name in data.columns]
            data.columns=columns
            
            data.index = pd.to_datetime(data.timestamp, unit="ms")
            data = data.drop(columns=["timestamp"])

            return data

        else: 
            # Cognite API is limited to extracting 100 tags at a time, so extract partly with intervals
            data = pd.DataFrame(timestamps, columns=["timestamp"])
            intervals = np.arange(0,len(tags),100)
            if intervals[-1] != len(tags):
                intervals = np.append(intervals, len(tags))

            for i, j in enumerate(intervals[1:]):
                from_int = intervals[i]
                to_int = j

                tmp_tags = list(tags[from_int:to_int])
                tmp_data = self.read_data(tmp_tags, start, end, granularity, aggregates, timestamps=None)      

                data_ts = data_ts.merge(tmp_data)
                if verbose > 0: print("✓ Completed tags {0} to {1}".format(from_int, to_int))

            return data_ts
    
    def impute_linear(self, df): 
        tmp_df = df.copy()
        for col in range(tmp_df.shape[-1]):
            idxs = np.where(tmp_df.iloc[:,col].isna())[0]
            for idx in idxs: 
                # if the index is the first or last, continue
                if idx == 0 or idx == len(tmp_df)-1:
                    continue

                curr_val = df.iloc[idx, col]
                next_val = df.iloc[idx+1, col]
                last_val = df.iloc[idx-1, col]

                # if value exists before and after, set value to average of the two
                if np.isnan(curr_val) and not np.isnan(last_val) and not np.isnan(next_val):
                    tmp_df.iloc[idx,col] = np.average([last_val, next_val])

        return tmp_df

    def impute_ffill(self, df): 
        tmp_df = df.copy()
        tmp_df = tmp_df.fillna(method="ffill")
        tmp_df = tmp_df.fillna(method="backfill")
        
        return tmp_df

    def scale_data(self, df, scaling_means, scaling_stds):        
        return (df - scaling_means) / scaling_stds
    
    def unscale_tags(self, df, scaling_means, scaling_stds): 
        return df * scaling_stds + scaling_means
        
    
    def read_preprocess_data(self, from_date, to_date, 
                             tags=None,
                             granularity="1m", 
                             aggregate=["average"], 
                             timestamps=None):
        """
        Function to read and do any preprocessing steps to the data
        """
        if tags is None: 
            tags = self.all_tags
            
        if self.verbose > 0: print("Extracting data for",len(tags),"tags from",from_date,"to",to_date)
        
        # read data
        if self.verbose > 1: print("Reading raw data ...", end=" ")
        raw_data = self.read_data(tags, from_date, to_date, granularity, aggregate, timestamps=timestamps)
        if self.verbose > 1: print("\t\t✓", raw_data.shape)
        
        # handle missing values
        if self.verbose > 1: print("Handling missing values ...", end=" ")
        data = self.impute_linear(raw_data)
        data = self.impute_ffill(data)
        if self.verbose > 1: print ("\t✓", sum(data.isna().sum()),"NAs")
        
        # scale data
        if self.verbose > 1: print ("Scaling data ...", end=" ")
        data_scaled = self.scale_data(data, 
                                      self.scaling_stats.loc[tags, "Mean"], 
                                      self.scaling_stats.loc[tags, "Std"]) 
        if self.verbose > 1: print("\t\t✓")
        
        return data_scaled