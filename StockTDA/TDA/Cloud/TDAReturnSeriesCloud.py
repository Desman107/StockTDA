
from StockTDA.TDA.TDAFrame import StockTDAFrame
from StockTDA import config

import os
import joblib
import pandas as pd
import gudhi as gd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from typing import List, Union, Optional, Tuple, Type

class StockTDAReturnSeriesCloud(StockTDAFrame):
    def __init__(self):
        super().__init__()
    
    
    def compute_persistence(self, date: str) -> Union[pd.DataFrame,None]:
        """
        Computes the persistence diagram for a given date based on the return series of stock constituent data.

        This method overrides the abstract method `compute_persistence` defined in the parent class `StockTDAFrame`.
        It performs the following steps:
        
        1. Loads the most recent 120 trading days of stock return data up to the specified date from `quote_df`.
        2. Filters the data to include only selected return metrics ('return', 'return_t-5', 'return_t-20', 'return_t-60').
        3. If any NaN values are found in the filtered data, the function returns `None` (indicating that the persistence cannot be computed).
        4. Standardizes the data using `StandardScaler` to ensure each feature has zero mean and unit variance.
        5. Constructs an Alpha Complex from the standardized data and computes the persistence diagram using GUDHI.
        6. Converts the persistence diagram into a pandas DataFrame with two columns:
            - 'dimension': Represents the dimension of the topological feature (e.g., 1 for connected components, 2 for loops, etc.).
            - 'persistence': A tuple containing the birth and death times of the topological features.

        Parameters:
        date: str
            The date for which the persistence diagram is being computed. The date is typically in 'YYYY-MM-DD' format.
        
        Returns:
        pd.DataFrame or None:
            A DataFrame containing the persistence diagram with the following columns:
            - 'dimension': The dimension of the topological feature.
            - 'persistence': The birth and death times of each topological feature, stored as tuples (birth, death).
            
            If the return series data contains missing values (NaN), the function returns `None`.

        Example of the returned DataFrame:
        -------------------------------------
        |   'dimension' |   'persistence'   |
        -------------------------------------
        |   0           |   (0.0121, 0.2324)|
        |   0           |   (0.0135, 0.3134)|
        |   0           |   (0.0141, 0.3325)|
        |   1           |   (0.0159, 0.4386)|
        |   1           |   (0.5421, 0.6529)|
        -------------------------------------
        """
        df = self.quote_df.loc[:date].tail(120)
        df = df[['return','return_t-5','return_t-20','return_t-60']]
        if df.isna().any().any():return
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        alpha_complex = gd.AlphaComplex(df_scaled)
        simplex_tree = alpha_complex.create_simplex_tree()
        persistence = simplex_tree.persistence()
        persistence_df = pd.DataFrame(persistence, columns=['dimension', 'persistence'])
        return persistence_df