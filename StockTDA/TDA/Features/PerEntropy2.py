

from StockTDA import config
from .TDAFeatures import TDAFeatures
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type
import numpy as np
from gtda.diagrams import PersistenceEntropy
class PerEntropy2(TDAFeatures):
    def __init__(self):
        super().__init__()


    def compute_TDAFeatures(self, persistence : List[Tuple[int, Tuple[float, float]]]) -> np.ndarray:
        """
        Compute the persistent entropy of a persistence diagram.

        Parameters:
        persistence_list: List[Tuple[float, float]]
            A list of tuples representing the birth and death times of topological features
            from a persistence diagram, where each tuple is (birth, death).

        Returns:
        float
            The persistent entropy value, which quantifies the distribution of lifetimes
            of topological features in the persistence diagram. This value provides a measure
            of the complexity of the underlying data by evaluating the spread of feature lifetimes.
        """
        diagrams  = [[[item[1][0], item[1][1], item[0]] for item in persistence if item[1][1] != np.inf]]
        ep = PersistenceEntropy()
        PE = ep.fit_transform(diagrams)
        return PE
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim  : List[Tuple[int, Tuple[float, float]]]):
        vectoralize_features = []
        PE = self.compute_TDAFeatures(persistence_all_dim)
        vectoralize_features = PE.reshape((-1,))
        return vectoralize_features