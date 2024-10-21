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

    life_time = np.array(persistence_list)
    l = life_time[:,1] - life_time[:,0]
    p = l / np.sum(l)
    return -np.sum(p*np.log(p))




def landscape(persistence : List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute the persistence landscape.
    
    Parameters:
    persistence: List of tuples representing the birth and death times of topological features (e.g., from a persistence diagram).

    Returns:
    np.ndarray: The integral of the persistence landscape for each layer.
    """

    mp=dict()
    idx=0 
    low=sorted(i[1] for i in persistence) 
    persistence.sort()
    mp[-1e18]=1 
    idx=1 
    for i in low:
        mp[i]=idx+1
        idx+=1 
    mp[1e18]=idx+1
    idx+=1 
    b=[]
    for i in range(len(persistence)): 
        for j in range(i+1,len(persistence)): 
            if persistence[i][1] > persistence[j][0]: 
                b.append((persistence[j][0], persistence[i][1])) 
            else: break 
    persistence.extend(b) 
    persistence.sort() 
    tl,tr,tv=[0 for i in range(4*(idx+10))],[0 for i in range(4*(idx+10))],[0 for i in range(4*(idx+10))]
    def push_up(u): 
        tv[u]=max(tv[u*2],tv[u*2+1]) 
    def build(u,l,r):
        tl[u],tr[u],tv[u]=l,r,0
        if l==r: return 
        mid=(l+r)//2 
        build(u*2,l,mid) 
        build(u*2+1,mid+1,r) 
    def change(u,p,v): 
        if tl[u]==tr[u] and tl[u]==p:
            tv[u]=v 
            return 
        mid = (tl[u]+tr[u])//2 
        if p<=mid: change(u*2,p,v) 
        else: change(u*2+1,p,v) 
        push_up(u)
    def query(u,l,r):
        if tl[u]>=l and tr[u]<=r:
            return tv[u] 
        mx = 0
        mid = (tl[u]+tr[u])//2 
        if l<=mid:
            mx=max(mx,query(u*2,l,r)) 
        if r>mid:
            mx=max(mx,query(u*2+1,l,r)) 
        # print(mx) 
        return mx 
    def cal(l:float,r:float):
        if l>r: return 0.0
        le = r - l  
        # print(le*le/4) 
        return le*le/4.0
    build(1,1,idx)
    coach = [[] for i in range(len(persistence))]
    id=[0 for i in range(len(persistence))] 
    for i in range(len(persistence)): 
        mx=query(1,mp[persistence[i][1]],idx) 
        id[i]=mx+1 
        change(1,mp[persistence[i][1]],id[i])
        coach[id[i]].append((persistence[i][0],persistence[i][1])) 
    ans = []
    for i in coach:
        if len(i)==0: continue 
        # print(i)
        s = 0 
        for j in range(len(i)): 
            s+=cal(i[j][0],i[j][1])  
            if j > 0: 
                s-=cal(i[j][0],i[j-1][1])
        ans.append(s)  
    return np.array(ans )