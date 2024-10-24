# -*-coding:utf-8 -*-

"""
# File       : TDAFrame.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Framework for stock topology data analysis
"""

from tqdm.auto import tqdm
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type
import pandas as pd
import numpy as np
import warnings

from StockTDA.TDA.Features.TDAFeatures import persistent_entropy, landscape, betti_sequence
from StockTDA.data.data_prepare import INFO, prepare_formulaic_factor, get_quote_df
from StockTDA.utils import ygo
from StockTDA.TDA.Features.TDAFeatures import TDAFeatures
class StockTDACloud(metaclass=ABCMeta):
    def __init__(self,features_list : List[TDAFeatures]):
        self.features_list = features_list
        self.quote_df = get_quote_df()


    @abstractmethod
    def compute_persistence(self, date : str) -> List[Tuple[int,Tuple[float,float]]]:
        """
            Abstract method to compute the persistence diagram for a given date.

            Parameters:
            date: str
                The date for which the persistence diagram is being computed, typically in 'YYYY-MM-DD' format.
            
            Returns:
            List
                A list representing the persistence diagram of topological features.
                
                Each entry in the list corresponds to a topological feature and contains:
                - 'dimension': The dimension of the topological feature (e.g., 0 for connected components, 1 for loops, 2 for voids, etc.).
                - 'persistence': A tuple representing the birth and death times (birth, death) of the topological feature, which indicates when the feature appears and disappears in the filtration process.
                
                The persistence diagram captures the shape characteristics of the data at different scales.
                
                Example:
                
                [   ['dimension',  'persistence'] ]
                [
                [   0           ,   (0.0121, 0.2324)],  # 0D feature: connected component
                [   0           ,   (0.0135, 0.3134)],  # 0D feature: connected component
                [   0           ,   (0.0141, 0.3325)],  # 0D feature: connected component
                [   1           ,   (0.0159, 0.4386)],  # 1D feature: loop
                [   1           ,   (0.5421, 0.6529)],  # 1D feature: loop
                ]
        """
        return ...
    
    def compute_TDA_Features(self, persistence : List[Tuple[int,Tuple[float,float]]], date : str) -> Tuple[Union[str,List[float]]]:
        result = [feature.compute_TDAFeatures_all_dim(persistence) for feature in self.features_list]
        result.insert(0, date)
        return result


    def compute_persistence_and_features(self, date : str) ->  Tuple[Union[str,List[float]]]:
        persistence = self.compute_persistence(date)
        if persistence is None:
            return
        features_list = self.compute_TDA_Features(persistence,date)
        return features_list
    
    
    def all_Features(self):
        with ygo.pool() as paral:
            for date in INFO.TradingDay:
                paral.submit(self.compute_persistence_and_features,date)
            result = paral.do(description=f'Computing TDA features, TDAModel = {self.__class__.__name__}')
        col_name = [str(feature) for feature in self.features_list]
        col_name.insert(0, 'date')
        result = [i for i in result if i is not None]
        TDA_df = pd.DataFrame(result,columns = col_name)
        TDA_df.set_index('date',inplace=True)
        self.all_df = pd.merge(self.quote_df,TDA_df,how='left',left_index=True,right_index=True)
        self.all_df.dropna(inplace=True)
        