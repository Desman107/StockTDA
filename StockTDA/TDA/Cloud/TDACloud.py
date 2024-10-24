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

class StockTDACloud(metaclass=ABCMeta):
    def __init__(self):
        # prepare_formulaic_factor()
        self.quote_df = get_quote_df()


    @abstractmethod
    def compute_persistence(self, date : str) -> pd.DataFrame:
        """
            Abstract method to compute the persistence diagram for a given date.

            Parameters:
            date: str
                The date for which the persistence diagram is being computed, typically in 'YYYY-MM-DD' format.
            
            Returns:
            pd.DataFrame
                A DataFrame called 'persistence_df' with the following two columns:
                - 'dimension': int, representing the dimension of the topological feature (e.g., 1 for connected components, 2 for loops, etc.).
                - 'persistence': tuple of (float, float), representing the birth and death times of each topological feature.
                
                The structure of 'persistence_df' should be as follows:
                ------------------------------------------------
                |   'dimension' |   'persistence'              |
                ------------------------------------------------
                |   0           |   (birth_time, death_time)   |
                |   1           |   (birth_time, death_time)   |
                |   ...         |   ...                        |
                ------------------------------------------------
        """
        return ...
    
    def compute_TDA_Features(self, persistence_df : pd.DataFrame, date : str) -> Tuple[str, List[int], List[float], List[float]]:
        """
        Compute a variety of topological data analysis (TDA) features based on the provided persistence diagram.
        
        This function calculates the Betti numbers, persistent entropy, and L2-norm of the persistence landscape 
        for different dimensions of topological features (e.g., connected components, loops, voids) on a given date.

        Parameters:
        persistence_df: pd.DataFrame
            A DataFrame that contains the persistence diagram with two columns:
            - 'dimension': The dimension of the topological feature (e.g., 1 for connected components, 2 for loops, etc.).
            - 'persistence': A tuple representing the birth and death times of each topological feature.
            
            Example:
            -------------------------------------
            |   'dimension' |   'persistence'   |
            -------------------------------------
            |   0           |   (0.0121, 0.2324)|
            |   0           |   (0.0135, 0.3134)|
            |   0           |   (0.0141, 0.3325)|
            |   1           |   (0.0159, 0.4386)|
            |   1           |   (0.5421, 0.6529)|
            -------------------------------------
        
        date: str
            The date for which the TDA features are being computed, typically in the format 'YYYY-MM-DD'.
        
        Returns:
        List[str, List[int], List[float], List[float]]
            A list containing:
            - date: The date of the computation (str).
            - betti_list: A list of Betti numbers for each dimension (List[int]).
            - entropy_list: A list of persistent entropy values for dimensions 1 to 3 (List[float]).
            - l2_norm_list: A list of L2-norms of the persistence landscapes for dimensions 2 and 3 (List[float]).

        Process:
        1. The function first extracts the persistence data from the DataFrame and computes the Betti number sequence 
        using the `betti_sequence()` function.
        2. Then, it computes the persistent entropy for dimensions 1 to 3 using the `persistent_entropy()` function.
        3. Finally, the function calculates the L2-norm of the persistence landscape for dimensions 2 and 3 using 
        the `landscape()` function and computes the L2-norm of the resulting landscapes.
        """
        # persistence_list = persistence_df['persistence'].to_list()
        # betti_list = betti_sequence(persistence_list)
        for i in [0,1,2,3]:
            persistence_list = persistence_df[persistence_df['dimension'] == i]['persistence'].to_list()
            betti_list = betti_sequence(persistence_list)
            
            betti_list = betti_list * 2 ** ((i-1)*2)

        entropy_list = []
        for i in range(1,4):
            persistence_list = persistence_df[persistence_df['dimension'] == i]['persistence'].to_list()
            entropy_list.append(persistent_entropy(persistence_list))

        l2_norm_list = []
        for i in [2,3]:
            persistence_list = persistence_df[persistence_df['dimension'] == i]['persistence'].to_list()
            persistent_landscape = landscape(persistence_df[persistence_df['dimension'] == i]['persistence'].to_list())
            l2_norm = np.sqrt(np.sum((persistent_landscape ** 2)))
            l2_norm_list.append(l2_norm)
        return [date,betti_list,entropy_list,l2_norm_list]


    def compute_persistence_and_features(self, date : str) -> Tuple[str, List[int], List[float], List[float]]:
        persistence_df = self.compute_persistence(date)
        if persistence_df is None or persistence_df.empty:
            # warnings.warn(f"Persistence data for {date} is missing or empty.", UserWarning)
            return
        features_list = self.compute_TDA_Features(persistence_df,date)
        return features_list
    
    
    def all_Features(self):
        with ygo.pool() as paral:
            for date in INFO.TradingDay:
                paral.submit(self.compute_persistence_and_features,date)
            result = paral.do(description=f'Computing TDA features, TDAModel = {self.__class__.__name__}')
        # result = []
        # for date in tqdm(INFO.TradingDay):
        #     result.append(self.compute_persistence_and_features(date))
        result = [i for i in result if i is not None] # filter empty Features
        TDA_df = pd.DataFrame(result,columns=['date','betti','entropy','l2_norm'])
        TDA_df.set_index('date',inplace=True)

        self.all_df = pd.merge(self.quote_df,TDA_df,how='left',left_index=True,right_index=True)
        self.all_df.dropna(inplace=True)
        