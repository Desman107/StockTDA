


from .TDAFeatures import TDAFeatures
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type






class Landscape(TDAFeatures,metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def compute_TDAFeatures(self, persistence):
        return super().compute_TDAFeatures(persistence)
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim):
        return super().compute_TDAFeatures_all_dim(persistence_all_dim)