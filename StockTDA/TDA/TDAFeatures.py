# -*-coding:utf-8 -*-

"""
# File       : TDAFeatures.py
# Time       : 2024/10/20 17:02
# Author     : JiaYi Chen
# Email      : code_ccc178@outlook.com
# Description: Tools for TDA features computing
"""

import numpy as np
from typing import List, Tuple



def betti_sequence(persistence : List[Tuple[float, float]]):  
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
    r=a[-1][1] 
    step=0.01*r  
    
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
    return ans[2:100]


def persistent_entropy(persistence_list : List[Tuple[float, float]]):
    """
    计算持续熵
    """

    life_time = np.array(persistence_list)
    l = life_time[:,1] - life_time[:,0]
    p = l / np.sum(l)
    return -np.sum(l*np.log(l))




def landscape(persistence : List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute the persistence landscape.
    
    Parameters:
    persistence: List of tuples representing the birth and death times of topological features (e.g., from a persistence diagram).

    Returns:
    np.ndarray: The integral of the persistence landscape for each layer.
    """

    def cal(l:float,r:float):
        if l>r: return 0.0
        le = r - l  
        # print(le*le/4) 
        return le*le/4.0
    persistence = np.array(persistence)
    a = []
    for i in persistence:
        a.append((float(i[0]),float(i[1])))  
    a=sorted(a) 
    un = [] 
    for i in range(len(a)): 
        for j in range(i+1,len(a)): 
            if a[i][1] >= a[j][0]: 
                un.append((a[j][0],a[i][1])) 
            else: break 
    for i in un: a.append(i) 
    a=sorted(a)  

    ans = [] 
    while len(a) > 0: 
        b,u = [],[]
        i = 0 
        while i < len(a): 
            j = i 
            while j + 1 < len(a) and a[j+1][1] <= a[i][1]: 
                j += 1
                u.append(a[j])  
            b.append(a[i])
            i = j + 1
        s,now = 0,0 
        for i in range(len(b)): 
            s+=cal(b[i][0],b[i][1])  
            if i > 0: 
                s-=cal(b[i][0],b[i-1][1]) 
        ans.append(s) 
        a=u
    return np.array(ans )