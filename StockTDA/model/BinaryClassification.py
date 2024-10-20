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
        return ...

    def rolling_predict(self, all_df : pd.DataFrame, features : List[str]):
        pred_list = []
        for i, date in tqdm(enumerate(config.date_range),total=len(config.date_range)):
            if i + 1 > len(config.date_range) - 1: break
            train_df = all_df[config.start_date:date]
            X_train = self.get_feature(train_df,features)
            y_train = (train_df[['return_t+1']] > 0).astype(int)
            test_df = all_df[date:config.date_range[i + 1]]
            X_test = self.get_feature(test_df,features)
            y_test = (test_df[['return_t+1']] > 0).astype(int)
            y_pred = self.run_classification(X_train, y_train, X_test, y_test)
            test_df['pred'] = y_pred
            pred_list.append(test_df)
        result_df = pd.concat(pred_list)
        result_df = result_df[~result_df.index.duplicated(keep='first')]
        return result_df