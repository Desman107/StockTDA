# -*-coding:utf-8 -*-

"""
# File       : BinaryClassification.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Framework for classification models in stock topology data analysis
"""


from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from StockTDA import config

class BinaryClassificationModel(metaclass=ABCMeta):
    def __init__(self):pass

    def get_feature(self,df:pd.DataFrame,features:list):
        """
        get specific TDA features from df 
        """
        feature_list = []
        for col in features:
            feature_list.append(np.array(df[col].to_list()))
        return np.concatenate(feature_list,axis = 1)
    
    @abstractmethod
    def run_classification(self, X_train, y_train, X_test, y_test):
        """
        Perform binary classification and return predicted labels for the test set.

        Parameters:
        - X_train: Training feature matrix.
        - y_train: Training labels.
        - X_test: Testing feature matrix.
        - y_test: Testing labels.

        Returns:
        - y_pred: Predicted labels for X_test, same length as y_test.
        """
        return ...

    def rolling_predict(self, all_df: pd.DataFrame, features: List[str]):
        pred_list = []
        # We use a generator for date_range
        date_iter = iter(config.date_range)
        prev_date = next(date_iter)  # Initialize the first date
        
        for next_date in date_iter:
            # get train data
            train_df = all_df[config.start_date:prev_date]
            X_train = self.get_feature(train_df, features)
            y_train = (train_df[['return_t+1']] > 0).astype(int)

            # get test data
            test_df = all_df[prev_date:next_date]
            X_test = self.get_feature(test_df, features)
            y_test = (test_df[['return_t+1']] > 0).astype(int)

            # train and predict
            y_pred = self.run_classification(X_train, y_train, X_test, y_test)

            # save predict result
            test_df['pred'] = y_pred
            pred_list.append(test_df)

            # Update prev_date to the current next_date for the next iteration
            prev_date = next_date

        result_df = pd.concat(pred_list)
        result_df = result_df[~result_df.index.duplicated(keep='first')]
        return result_df

