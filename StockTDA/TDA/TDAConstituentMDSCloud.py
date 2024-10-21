from StockTDA.TDA.TDAFrame import StockTDAFrame
from StockTDA import config

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
        self.sliding_window = deque(maxlen=window_size)
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
        # Step 1: Load factor data for the given date
        df = joblib.load(os.path.join(config.factor_data_save_path, date))
        df['date'] = date
        df = df[['past_1D','date']].copy()

        # Step 2: Add the data to the sliding window
        self.sliding_window.append(df)

        # Step 3: If the sliding window is not yet full, return None
        if len(self.sliding_window < self.window_size): return

        # Step 4: Concatenate all data in the sliding window into a single DataFrame
        win_df = pd.concat(self.sliding_window)
        win_df = pd.concat(self.sliding_window)
        win_df.reset_index(inplace=True)
        win_df.set_index(['date','SecuCode'],inplace=True)
        win_df = win_df.unstack()

        # Cleaning data by removing columns with more than 30% missing values
        missing_ratio = win_df.isna().sum() / len(win_df)
        win_df_cleaned = win_df.loc[:, missing_ratio <= 0.3]
        Cij = win_df_cleaned.corr()
        distance_matrix = np.sqrt(2 * (1 - Cij))

        # Step 5: Compute the correlation matrix and convert it into a distance matrix
        mds = MDS(n_components=4, dissimilarity="precomputed", random_state=42)

        # Step 7: Construct an Alpha Complex from the MDS coordinates and compute the persistence diagram
        coords = mds.fit_transform(distance_matrix)
        alpha_complex = gd.AlphaComplex(coords)
        simplex_tree = alpha_complex.create_simplex_tree()

        # Step 8: Compute persistence diagram
        persistence = simplex_tree.persistence()
        persistence_df = pd.DataFrame(persistence, columns=['dimension', 'persistence'])

        return persistence_df