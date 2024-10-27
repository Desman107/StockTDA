

from StockTDA import config
from .TDAFeatures import TDAFeatures 
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type


class BettiSeq3(TDAFeatures):
    def __init__(self):
        super().__init__()
    
    def compute_TDAFeatures(self, persistence : List[Tuple[float, float]], r , n_bins):
        """
        Compute the Betti number sequence.
        
        Parameters:
        persistence: List of tuples representing the birth and death times of topological features 
                    (typically from a persistence diagram).

        Returns:
        List or np.ndarray: The Betti number sequence over the given filtration.
        """

        persistence=np.array(persistence) 
        idx = 0  
        a,low = [],[]
        mp=dict() 
        for i in persistence:
            a.append((float(i[0]),float(i[1]))) 
            low.append(float(i[1])) 
        low.sort()
        mp[-1e18]=1 
        idx=1 
        for i in low:
            mp[i]=idx+1
            idx+=1  
        a.sort()  


        l=0
        
        step= r / n_bins 
        
        tr = [0 for i in range(idx+3)] 
        def lowbit(x): 
            return (x&-x) 
        def add(x,c): 
            i=x 
            while i <= idx:
                tr[i]+=c
                i+=lowbit(i) 
        def sums(x): 
            i=x
            su = 0 
            while i>0:
                su+=tr[i]
                i-=lowbit(i) 
            return su 
        zz = l 
        hs = -1
        ans=[]

        def f(z): 
            if low[0] > z or low[len(low)-1] < z: return 0 
            L,R=0,len(low)-1 
            while L<R:
                mid = (L+R)//2 
                if low[mid]>=z:
                    R=mid 
                else: 
                    L=mid+1 
            return sums(idx)-sums(mp[low[L]]-1) 
        while zz <= r: 
            while hs + 1 < len(a) and a[hs+1][0] <= zz:
                hs+=1 
                add(mp[a[hs][1]],1) 
            ans.append(f(zz))  
            zz+=step 
        return np.array(ans[:n_bins])
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim):
        vectoralize_features = []
        r = max([item[1][1] for item in persistence_all_dim if ( item[1][1] != np.inf)])
        n_bins = 100
        for dim in range(config.max_dim + 1):
            persistence = [(item[1][0], item[1][1]) for item in persistence_all_dim if (item[0] == dim and item[1][1] != np.inf)]
            bettiseq = self.compute_TDAFeatures(persistence, r, n_bins=n_bins)
            bettiseq = bettiseq * (2 ** dim)
            for betti in bettiseq:
                vectoralize_features.append(betti)
        vectoralize_features = np.array(vectoralize_features)
        vectoralize_features = vectoralize_features.reshape((-1,n_bins))
        vectoralize_features = np.sum(vectoralize_features, axis = 0)
        return vectoralize_features