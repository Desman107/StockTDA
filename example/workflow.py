from StockTDA.model.XGB import TDAXGBoost
from StockTDA.TDA.TDAReturnSeriesCloud import StockTDAReturnSeriesCloud
from StockTDA.evluate.evaluator import StockTDAClassificationEvaluator
from StockTDA.data.data_prepare import prepare_formulaic_factor
from StockTDA import config
import os
import inspect

# prepare_formulaic_factor() # for  the first time, place run this
ClassificationModel = TDAXGBoost()
TDAModel = StockTDAReturnSeriesCloud()
# print(ClassificationModel.__class__.__name__)
# print(TDAModel.__class__.__name__)
# TDA_code = inspect.getsource(TDAModel.__class__)
# with open(os.path.join(config.temp_file_path,"TDA_code.py"), "w") as f:
#     f.write(TDA_code)
Evaluator = StockTDAClassificationEvaluator(TDAModel,ClassificationModel)
Evaluator.evaluate_all_combinations()