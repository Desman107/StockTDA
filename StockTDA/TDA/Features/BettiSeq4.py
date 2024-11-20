

from StockTDA import config
from .TDAFeatures import TDAFeatures 
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type
from gtda.diagrams import  BettiCurve

class BettiSeq4(TDAFeatures):
    def __init__(self):
        super().__init__()
    
    def compute_TDAFeatures(self, persistence : List[Tuple[float, float]], n_bins) -> np.ndarray:
        """
        Compute the Betti number sequence.
        
        Parameters:
        persistence: List of tuples representing the birth and death times of topological features 
                    (typically from a persistence diagram).

        Returns:
        List or np.ndarray: The Betti number sequence over the given filtration.
        """
        diagrams = [[[item[1][0], item[1][1], item[0]] for item in persistence if item[1][1] != np.inf]]
        bc = BettiCurve(n_bins = n_bins)
        bt_c = bc.fit_transform(diagrams)
        return bt_c
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim):
        
        n_bins = 25
        betti_curve = self.compute_TDAFeatures(persistence=persistence_all_dim,n_bins=n_bins)
        vectoralize_features = betti_curve.reshape((-1,))
        return vectoralize_features