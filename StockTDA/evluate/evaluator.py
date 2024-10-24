
from StockTDA.model.BinaryClassification import BinaryClassificationModel
from StockTDA.TDA.Cloud.TDACloud import StockTDACloud
from StockTDA.utils.mlflow import record

from typing import List, Union, Optional, Tuple, Type
from itertools import combinations
import pandas as pd
import logging
from sklearn.metrics import classification_report

class StockTDAClassificationEvaluator():
    def __init__(self, TDAModel : StockTDACloud, ClassificationModel : BinaryClassificationModel):
        self.TDAModel = TDAModel
        self.ClassificationModel = ClassificationModel

    
    @property
    def features_combination(self):
        features = self.TDAModel.features_list
        for r in range(1, len(features) + 1):
            for combo in combinations(features, r):
                yield list(combo)

    def evaluateTDAFeatures(self, features: List[str]):

        # predict 
        result_df = self.ClassificationModel.rolling_predict(self.TDAModel.all_df, features)

        # evaluate predict result
        result_df['preds'] = (result_df['pred'] > 0.5).astype(int)
        result_df['label'] = (result_df['return_t+1'] > 0).astype(int)
        report = classification_report(result_df['label'], result_df['preds'])
        logging.log(logging.INFO,f'\nClassification Report:\n{report}')
        
        # construct long-short strategy
        result_df['long_short'] = 0
        result_df.loc[result_df['preds'] == 1, 'long_short'] = 1
        result_df.loc[result_df['preds'] == 0, 'long_short'] = -1
        result_df['return'] = result_df['long_short'] * result_df['return_t+1']
        result_df.sort_index(inplace=True)
        result_df.index = pd.to_datetime(result_df.index)

        # record result by mlflow
        record(result_df,self.ClassificationModel,self.TDAModel,features)
    
    def evaluate_all_combinations(self):
        for feature_set in self.features_combination:
            feature_set = [str(feature_) for feature_ in feature_set]
            self.evaluateTDAFeatures(feature_set)