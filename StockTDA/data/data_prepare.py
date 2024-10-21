# -*-coding:utf-8 -*-

"""
# File       : data_prepare.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Preparing stock data for TDA
"""

import sys

import joblib
import os

import pandas as pd
import numpy as np
from StockTDA import config
from StockTDA.utils import ygo


class Info_data():
    def __init__(self):
        TradingDay = joblib.load(os.path.join(config.data_path,'calendar','BIG_TABLE')) # load Trading day calendar
        TradingDay = np.array(TradingDay[TradingDay['IfTradingDay'] == 1].index)
        self.TradingDay = TradingDay[(TradingDay > config.start_date) & (TradingDay < config.end_date)] # set date range

        self.StockMain = joblib.load(os.path.join(config.data_path,'info','STOCK-MAIN'))

INFO = Info_data()


def get_index_comp(code : str,date : str) -> pd.DataFrame:
    """
    Query the constituent stocks of an index on the given date.

    Parameters:
    code : str
        The index code.
    date : str
        The date in 'YYYY-MM-DD' format. This function will retrieve the constituent stocks of the index 
        as of the most recent date prior to or on the specified date.
    
    Returns:
    pd.DataFrame
        The constituent stocks of the index on the closest date before or equal to the input date.
    """


    index_comp_path = os.path.join(config.data_path,'index','INDEX-COMP',code)
    cl = np.array(os.listdir(index_comp_path))
    d = cl[cl < date][-1] # the last day
    return joblib.load(os.path.join(index_comp_path,d))


def prepare_formulaic_factor():
    """
    Prepares formulaic factor data.

    This function processes and selects formulaic factor data for the constituent stocks 
    of a given index. The data is saved day by day into the specified directory 
    [config.factor_data_save_path].

    Process overview:
    1. Loads available formulaic factor data for all stocks.
    2. For each trading day, saves the constituent stock data in a temporary directory.
    3. Filters the data by index constituents and secures it for the given date.
    4. Finally, cleans and stores the factor data by trading day in the designated directory.

    The process is parallelized to improve performance using `ygo.pool()`.

    Returns:
    None
    """
    path = os.path.join(config.data_path,'factor','all_avail_df')
    df = joblib.load(path)
    def dump(date):
        temp_df = df.loc[date]
        joblib.dump(temp_df,os.path.join(config.temp_save_path,str(date)))
    with ygo.ThreadPool() as pool:
        for date in INFO.TradingDay:
            pool.submit(dump,date)
        pool.do()
    is_map = INFO.StockMain
    def get_data(date):
        df = joblib.load(os.path.join(config.temp_save_path,date))
        ic = get_index_comp(config.index_code, date) 
        df = df[(df.index.isin(is_map.loc[ic.index]['SecuCode']))]
        df['date'] = date
        return df
    with ygo.ThreadPool() as pool:
        for date in INFO.TradingDay:
            pool.submit(get_data,date)
        result = pool.do(description='loading')
    df = pd.concat(result)
    df.reset_index(inplace=True)
    df.set_index(['date','SecuCode'],inplace=True)
    df.drop(columns = 'Lv1',inplace=True)
    def dump(date):
        temp_df = df.loc[date]
        temp_df.dropna(inplace = True)
        joblib.dump(temp_df,os.path.join(config.factor_data_save_path,str(date)))
    with ygo.ThreadPool() as pool:
        for date in INFO.TradingDay:
            pool.submit(dump,date)
        pool.do()


def get_quote_df():
    """
    Retrieve and process index quote data for specified trading days.
    
    This function gathers index quote data for a given index code from a local file storage. The data is retrieved 
    for each trading day in INFO.TradingDay, then various return metrics are computed for the index.
    """
    
    index_path = os.path.join(config.data_path,'Index','INDEX-QUOTE')
    def get_data_csi(date):
        path = os.path.join(index_path,date)
        if os.path.exists(path):
            df = joblib.load(os.path.join(index_path,date)).loc[config.index_code]
            df.name = date
            return df
    with ygo.pool() as pool:
        for date in INFO.TradingDay:
            pool.submit(get_data_csi,date)
        csi_result = pool.do(description='loading index quote')
    quote_df = pd.concat(csi_result,axis=1).T
    quote_df['return'] = (quote_df['close'] - quote_df['prev_close']) / quote_df['prev_close']
    quote_df['return_t+1'] = quote_df['return'].shift(-1)
    quote_df.dropna(inplace=True)
    quote_df['return_t-5'] = quote_df['return'].shift(5)
    quote_df['return_t-20'] = quote_df['return'].shift(20)
    quote_df['return_t-60'] = quote_df['return'].shift(60)
    return quote_df
