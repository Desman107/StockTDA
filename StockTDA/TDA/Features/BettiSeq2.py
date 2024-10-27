from StockTDA import config
from .TDAFeatures import TDAFeatures 
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Tuple, Type

class BettiSeq2(TDAFeatures):
    def __init__(self):
        super().__init__()

    def compute_TDAFeatures(self, persistence : List[Tuple[float, float]], right_boundary: float,n_bins: int) ->np.ndarray:
        """
        Compute the Betti number sequence.
        
        Parameters:
        persistence: List of tuples representing the birth and death times of topological features 
                    (typically from a persistence diagram).

        Returns:
        List or np.ndarray: The Betti number sequence over the given filtration.
        """
        
        # 修改该位置
        def modify_tree(tr: List[int],pos: int,ed: int,c: int): 
            i = pos 
            while i <= ed: 
                tr[i] += c 
                i += (i&-i)
        # 查询  
        def query_tree(tr: List[int],pos: int) -> int:  
            i, res = pos, 0 
            while i > 0: 
                res += tr[i] 
                i -= (i&-i) 
            return res 
        step = right_boundary/n_bins 
        Len = right_boundary/n_bins # 浮标区间的长度
        L, point_list, mp, map_val = [], [], {}, 1 
        for birth, death in persistence:
            L.append((birth, death))
            point_list.append(death)
            point_list.append(birth) 
        L.sort()
        point_list.sort() 
        mp[-1e18] = 1 
        temp_i, poo = 0, [-1e18]
        while temp_i < len(point_list): 
            temp_j = temp_i 
            while temp_j + 1 < len(point_list) and point_list[temp_j+1] == point_list[temp_j]: temp_j += 1 
            mp[point_list[temp_i]]=map_val+1 
            poo.append(point_list[temp_i]) # 去重，减少重复情况较多的情况下二分的复杂度 
            map_val+=1 
            temp_i=temp_j+1
        mp[1e18]=map_val+1
        map_val+=1 
        poo.append(1e18) 
        # tr1[i]: 左树，表示<=i的左边界所有右边界情况。
        # tr2[i]: 右树，表示>i的左边界所有右边界情况。
        tr1, tr2 = [0 for i in range(map_val+1)], [0 for i in range(map_val+1)] 
        for seg in L: 
            modify_tree(tr2,mp[seg[0]],map_val,1) 
        buoy, ans, ml = 0, [], -1 
        
        def Calc(Left: int,Right: int) ->int: 
            if Left>Right: return 0 
            l, r, i1, i2 = 0, len(poo)-1, -1, -1  
            while l < r: 
                mid = (l + r) // 2 
                if poo[mid] >= Left: r = mid 
                else: l = mid + 1   
            if poo[l] >= Left: i1 = mp[poo[l]] 
            l, r = 0, len(poo)-1 
            while l < r: 
                mid = (l + r + 1) // 2 
                if poo[mid] <= Right: l = mid 
                else: r = mid - 1 
            if poo[l] <= Right: i2 = mp[poo[l]] 
            res = 0 
            if i1 != -1: res += query_tree(tr1,map_val)-query_tree(tr1,i1-1) 
            if i2 != -1: res += query_tree(tr2,i2)   
            return res 
        
        while buoy + Len <= right_boundary: 
            l, r = buoy, buoy + Len 
            while ml+1 < len(L) and L[ml+1][0]<=buoy:
                ml+=1 
                modify_tree(tr2,mp[L[ml][0]],map_val,-1) 
                modify_tree(tr1,mp[L[ml][1]],map_val,1) 
            ans.append(Calc(l,r))   
            buoy += step
        return np.array(ans) 
    
    
    def compute_TDAFeatures_all_dim(self, persistence_all_dim):
        vectoralize_features = []
        r = max([item[1][1] for item in persistence_all_dim if ( item[1][1] != np.inf)])
        n_bins = 26
        for dim in range(config.max_dim + 1):
            persistence = [(item[1][0], item[1][1]) for item in persistence_all_dim if (item[0] == dim and item[1][1] != np.inf)]
            bettiseq = self.compute_TDAFeatures(persistence, r, n_bins=n_bins)
            for betti in bettiseq[:n_bins-1]:
                vectoralize_features.append(betti)
        return vectoralize_features