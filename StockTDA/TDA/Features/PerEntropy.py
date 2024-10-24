

from StockTDA import config
from .TDAFeatures import TDAFeatures
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type
import numpy as np

class PerEntropy(TDAFeatures):
    def __init__(self):
        super().__init__()


    def compute_TDAFeatures(self, persistence : List[Tuple[float, float]]):
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

        life_time = np.array(persistence)
        l = life_time[:,1] - life_time[:,0]
        p = l / np.sum(l)
        return -np.sum(l*np.log(l))
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim  : List[Tuple[int, Tuple[float, float]]]):
        vectoralize_features = []
        for dim in range(config.max_dim + 1):
            persistence = [(item[1][0], item[1][1]) for item in persistence_all_dim if (item[0] == dim and item[1][1] != np.inf)]
            norm = self.compute_TDAFeatures(persistence)
            vectoralize_features.append(norm)
        return vectoralize_features