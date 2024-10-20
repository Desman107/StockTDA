
from StockTDA.model.BinaryClassification import BinaryClassificationModel
from StockTDA.TDA.TDAFrame import StockTDAFrame
from StockTDA.utils.mlflow import record

from typing import List, Union, Optional, Tuple, Type
from itertools import combinations
import pandas as pd
import logging
from sklearn.metrics import classification_report

class StockTDAClassificationEvaluator():
    def __init__(self, TDAModel : StockTDAFrame, ClassificationModel : BinaryClassificationModel):
        self.TDAModel = TDAModel
        self.ClassificationModel = ClassificationModel

    
    @property
    def features_combination(self):
        features = ['betti', 'entropy', 'l2_norm']
        for r in range(1, len(features) + 1):
            for combo in combinations(features, r):
                yield list(combo)

    def evaluateTDAFeatures(self, features: List[str]):
        result_df = self.ClassificationModel.rolling_predict(self.TDAModel.all_df, features)
        result_df['preds'] = (result_df['pred'] > 0.5).astype(int)
        result_df['label'] = (result_df['return_t+1'] > 0).astype(int)
        report = classification_report(result_df['label'], result_df['preds'])
        logging.log(logging.INFO,f'\nClassification Report:\n{report}')
        result_df['long_short'] = 0
        result_df['long_short'][result_df['preds'] == 1] = 1
        result_df['long_short'][result_df['preds'] == 0] = -1
        result_df['return'] = result_df['long_short'] * result_df['return_t+1']
        result_df.sort_index(inplace=True)
        result_df.index = pd.to_datetime(result_df.index)
        record(result_df,self.ClassificationModel,self.TDAModel,features)
    
    def evaluate_all_combinations(self):
        for feature_set in self.features_combination:
            self.evaluateTDAFeatures(feature_set)