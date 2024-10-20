
from StockTDA.model.XGB import TDAXGBoost
from StockTDA.TDA.TDAReturnSeriesCloud import StockTDAReturnSeriesCloud
from StockTDA.evluate.evaluator import StockTDAClassificationEvaluator
from StockTDA.data.data_prepare import prepare_formulaic_factor
from StockTDA import config
import os
import inspect
import joblib
prepare_formulaic_factor() # for  the first time, place run this
ClassificationModel = TDAXGBoost()
TDAModel = StockTDAReturnSeriesCloud()
TDAModel.all_Features()
all_df = TDAModel.all_df


Evaluator = StockTDAClassificationEvaluator(TDAModel,ClassificationModel)

Evaluator.evaluate_all_combinations()