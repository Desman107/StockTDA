

from StockTDA import config
from .TDAFeatures import TDAFeatures
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type
import numpy as np
from gtda.diagrams import PersistenceLandscape


class Landscape(TDAFeatures,metaclass=ABCMeta):
    def __init__(self, n_layers = 50, n_bins = 1000):
        super().__init__()
        self.n_layers = n_layers
        self.n_bins = n_bins

    def compute_TDAFeatures(self, persistence):
        diagrams = np.expand_dims(persistence,axis = 0)
        landscape = PersistenceLandscape(n_layers=self.n_layers, n_bins=self.n_bins)
        landscape_values = landscape.fit_transform(diagrams)
        norm = self.compute_norm(landscape_values)
        return norm

    @abstractmethod
    def compute_norm(self, landscape_values):
        return ...
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim  : List[Tuple[int, Tuple[float, float]]]):
        vectoralize_features = []
        for dim in range(config.max_dim + 1):
            persistence = np.array([[item[1][0], item[1][1], item[0]] for item in persistence_all_dim if(item[0] == dim and item[1][1] != np.inf) ] )
            norm = self.compute_TDAFeatures(persistence)
            vectoralize_features.append(norm)
        return vectoralize_features
    
class landscapeL2Norm(Landscape):
    def __init__(self):
        super().__init__()
    
    def compute_norm(self, landscape_values):
        return sum(sum((np.sqrt((np.sum(landscape_values**2,axis = 2) / 100) ))))