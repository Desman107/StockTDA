# -*-coding:utf-8 -*-

"""
# File       : XGB.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Stock topology data analysis combines with XGBoost
"""



from StockTDA.model.BinaryClassification import BinaryClassificationModel
import xgboost as xgb




class TDAXGBoost(BinaryClassificationModel):
    def __init__(self, objective = 'binary:logistic', eta = 0.3, eval_metric = 'logloss'):
        super().__init__()
        self.params = {
            'objective': objective,  
            'eta': eta,
            'eval_metric': eval_metric
        }


    def run_classification(self, X_train, y_train, X_test, y_test):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_rounds = 1000
        bst = xgb.train(self.params, dtrain, num_rounds)
        y_pred = bst.predict(dtest)
        return y_pred