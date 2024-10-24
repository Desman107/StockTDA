from StockTDA.TDA.Features.TDAFeatures import TDAFeatures
from StockTDA.TDA.Cloud.TDACloud import StockTDACloud
from StockTDA import config

import os
import joblib
import pandas as pd
import gudhi as gd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from typing import List, Union, Optional, Tuple, Type

class StockTDAConstituentCloud(StockTDACloud):
    def __init__(self,features_list : List[TDAFeatures]):
        super().__init__(features_list)
    
    
    def compute_persistence(self, date: str) -> pd.DataFrame:
        """
        Computes the persistence diagram for a given date based on stock constituent factor data.

        This method inherits from the abstract method `compute_persistence` defined in the parent class `StockTDACloud`.
        It performs the following steps:
        
        1. Load stock constituent factor data for the specified date.
        2. Standardize the data using `StandardScaler` to ensure each feature has zero mean and unit variance.
        3. Apply Kernel PCA (Principal Component Analysis with a radial basis function kernel) to reduce the dimensionality of the data to 4 components.
        4. Construct an Alpha Complex using the reduced data and compute its persistence diagram using GUDHI.
        5. Convert the persistence diagram into a pandas DataFrame with two columns:
            - 'dimension': Represents the dimension of the topological feature (e.g., 1 for connected components, 2 for loops, etc.).
            - 'persistence': A tuple containing the birth and death times of the topological features.

        Parameters:
        date: str
            The date for which the persistence diagram is being computed. The date is typically in 'YYYY-MM-DD' format.
        
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
        # Step 1: Load factor data for the given date
        df = joblib.load(os.path.join(config.factor_data_save_path, date))
        df = df.drop(columns = '1D')  # Exclude the first non-feature column)

        # Step 2: Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        
        # Step 3: Apply Kernel PCA to reduce dimensions to 4 components
        kpca = KernelPCA(n_components=4, kernel='rbf')
        kpca_results = kpca.fit_transform(df_scaled)

        # Step 4: Convert the Kernel PCA results into a DataFrame
        kpca_df = pd.DataFrame(kpca_results, columns=['kPCA_1', 'kPCA_2', 'kPCA_3', 'kPCA_4'])
        
        # Step 5: Create an Alpha Complex from the Kernel PCA results and compute the persistence diagram
        alpha_complex = gd.AlphaComplex(kpca_df.values)
        simplex_tree = alpha_complex.create_simplex_tree()
        persistence = simplex_tree.persistence()
        

        return persistence