from StockTDA.TDA.TDAFrame import StockTDAFrame
from StockTDA import config
from StockTDA.data.data_prepare import INFO
from StockTDA.utils import ygo

import os
import joblib
import pandas as pd
import numpy as np
import gudhi as gd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from collections import deque
from typing import List, Union, Optional, Tuple, Type

class StockTDACorrMDSCloud(StockTDAFrame):
    def __init__(self, window_size = 50):
        """
        Initializes the StockTDACorrMDSCloud instance.

        Parameters:
        window_size: int, default 50
            The size of the sliding window used to compute the correlation matrix for Multi-Dimensional Scaling (MDS).
            This sliding window maintains the most recent `window_size` trading days of data for analysis.
        
        Attributes:
        sliding_window: deque
            A deque (double-ended queue) to store a sliding window of factor data for the most recent `window_size` days.
        """
        self.window_size = window_size
        # self.sliding_window = deque(maxlen=window_size)
        self.prepare_data()
        super().__init__()
    
    
    def compute_persistence(self, date: str) -> Union[pd.DataFrame,None]:
        """
        Computes the persistence diagram for a given date based on the sliding window of stock constituent factor data.

        This method inherits from the abstract method `compute_persistence` defined in the parent class `StockTDAFrame`.
        It performs the following steps:

        1. Load stock constituent factor data for the specified date.
        2. Add the factor data to the sliding window, which maintains data for the most recent `window_size` trading days.
        3. If the sliding window is not yet full, exit without computing persistence.
        4. Concatenate all data in the sliding window into a single DataFrame and clean the data:
            - Drop columns with more than 30% missing values.
        5. Compute the correlation matrix of the cleaned data, then convert it into a distance matrix.
        6. Apply Multi-Dimensional Scaling (MDS) to reduce the dimensionality of the distance matrix to 4 components.
        7. Construct an Alpha Complex from the MDS coordinates and compute its persistence diagram using GUDHI.
        8. Convert the persistence diagram into a pandas DataFrame.

        Parameters:
        date: str
            The date for which the persistence diagram is being computed. The date is typically in 'YYYY-MM-DD' format.

        Returns:
        pd.DataFrame or None
            A DataFrame containing the persistence diagram with columns:
            - 'dimension': The dimension of the topological feature (e.g., 1 for connected components, 2 for loops, etc.).
            - 'persistence': The birth and death times of each topological feature, stored as tuples (birth, death).
            If the sliding window does not contain enough data, returns `None`.

        Example of returned DataFrame:
        -------------------------------------
        |   'dimension' |   'persistence'   |
        -------------------------------------
        |   0           |   (0.0121, 0.2324)|
        |   0           |   (0.0135, 0.3134)|
        |   1           |   (0.0159, 0.4386)|
        |   1           |   (0.5421, 0.6529)|
        -------------------------------------
        """

        date_loc = np.searchsorted(INFO.TradingDay, date, side='right')
        if date_loc < self.window_size:return
        loc_20 = date_loc - self.window_size
        date_20 = INFO.TradingDay[loc_20]
        win_df = self.csi300.loc[date_20:date]
        win_df = win_df.unstack()

        # Cleaning data by removing columns with more than 30% missing values
        missing_ratio = win_df.isna().sum() / len(win_df)
        win_df_cleaned = win_df.loc[:, missing_ratio <= 0.3]
        Cij = win_df_cleaned.corr()
        distance_matrix = np.sqrt(2 * (1 - Cij))

        # Step 5: Compute the correlation matrix and convert it into a distance matrix
        mds = MDS(n_components=4, dissimilarity="precomputed", random_state=42, normalized_stress='auto')

        # Step 7: Construct an Alpha Complex from the MDS coordinates and compute the persistence diagram
        coords = mds.fit_transform(distance_matrix)
        alpha_complex = gd.AlphaComplex(coords)
        simplex_tree = alpha_complex.create_simplex_tree()

        # Step 8: Compute persistence diagram
        persistence = simplex_tree.persistence()
        persistence_df = pd.DataFrame(persistence, columns=['dimension', 'persistence'])

        return persistence_df
    

    def prepare_data(self):
        def load(date):
            path = os.path.join(config.factor_data_save_path, date)
            df = joblib.load(path)
            df['date'] = date
            return df[['past_1D','date']]
        with ygo.pool() as pool:
            for date in INFO.TradingDay:
                pool.submit(load,date)
            result = pool.do()
        result_df : pd.DataFrame = pd.concat(result)
        result_df.reset_index(inplace=True)
        result_df.set_index(['date','SecuCode'],inplace=True)
        result_df.sort_index(inplace=True)
        self.csi300 = result_df
