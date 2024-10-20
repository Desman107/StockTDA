import logging
import os
import pandas as pd
import subprocess
import webbrowser
import time
import inspect
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import joblib
from typing import List

from StockTDA import config
from StockTDA.model.BinaryClassification import BinaryClassificationModel
from StockTDA.TDA.TDAFrame import StockTDAFrame


def record(result_df:pd.DataFrame, model_obj : BinaryClassificationModel, TDA_obj : StockTDAFrame,features : List[str]):
    name  = '&'.join(features)
    model = model_obj.__class__.__name__
    cloud_type = TDA_obj.__class__.__name__



    path = config.mlflow_path.replace("\\","/")
    if not os.path.exists(path):os.makedirs(path)
    mlflow_path = "file:///"+path
    record_dict = {
        'precision_0' : precision_score(result_df['label'], result_df['preds'],pos_label=0),
        'precision_1' : precision_score(result_df['label'], result_df['preds'],pos_label=1),
        'recall_0' : recall_score(result_df['label'], result_df['preds'],pos_label=0),
        'recall_1' : recall_score(result_df['label'], result_df['preds'],pos_label=1),
        'f1_score_0' : f1_score(result_df['label'], result_df['preds'],pos_label=0),
        'f1_score_1' : f1_score(result_df['label'], result_df['preds'],pos_label=1),
        'accuracy' : accuracy_score(result_df['label'], result_df['preds']),
        'cum_return': result_df['return'].cumsum().iloc[-1],
        'max_drawdown': (((result_df['return'] + 1).cumprod().sub((result_df['return'] + 1).cumprod().cummax())) / (result_df['return'] + 1).cumprod().cummax()).min()
    }
    param_dict = {
        'betti' : 0,
        'entropy' : 0,
        'l2_norm' : 0
    }
    for feature in features:
        param_dict[feature] = 1

    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment("TDA")

    with mlflow.start_run(run_name=f'{name}|{model}'):

        mlflow.log_param("model_name", model)
        mlflow.log_param('cloud_type',cloud_type)

        for metric, value in record_dict.items():
            logging.log(logging.INFO,f'metric : {metric}, value : {value}')
            mlflow.log_metric(metric, value)
        
        for param, value in param_dict.items():
            logging.log(logging.INFO,f'param : {param}, value : {value}')
            mlflow.log_param(param, value)


        joblib.dump(result_df,f'temp_log/{name}')
        mlflow.log_artifact(f'temp_log/{name}')
        current_code = inspect.getsource(model_obj.__class__)
        with open(os.path.join(config.temp_file_path,"experiment_code.py"), "w") as f:
            f.write(current_code)
        mlflow.log_artifact(os.path.join(config.temp_file_path,"experiment_code.py"))

        TDA_code = inspect.getsource(TDA_obj.__class__)
        with open(os.path.join(config.temp_file_path,"TDA_code.py"), "w") as f:
            f.write(TDA_code)
        mlflow.log_artifact(os.path.join(config.temp_file_path,"TDA_code.py"))

def ui():
    path = config.mlflow_path.replace("\\","/")
    
    mlflow_path = "file:///"+path
    command = f'mlflow ui --backend-store-uri {mlflow_path} --host 127.0.0.1 --port 5002'
    logging.log(logging.INFO,f'{command}')
    subprocess.Popen(command, shell=True)
    time.sleep(5)  
    webbrowser.open('http://127.0.0.1:5002')
