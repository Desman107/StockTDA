# -*-coding:utf-8 -*-

"""
# File       : XGB.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Stock topology data analysis combines with LightGBM
"""



from StockTDA.model.BinaryClassification import BinaryClassificationModel
import lightgbm as lgb



class TDALightGBM(BinaryClassificationModel):
    def __init__(self, objective = 'binary:logistic', learning_rate = 0.3, eval_metric = 'logloss'):
        super().__init__()
        self.params = {
            'objective': objective,  
            'learning_rate': learning_rate,
            'eval_metric': eval_metric
        }


    def run_classification(self, X_train, y_train, X_test, y_test):
        # LightGBM Dataset creation
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        # Training the model
        num_rounds = 1000
        bst = lgb.train(self.params, dtrain, num_boost_round=num_rounds)

        # Prediction
        y_pred = bst.predict(X_test)
        return y_pred