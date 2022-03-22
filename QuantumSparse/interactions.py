# -*- coding: utf-8 -*-
import numpy as np
from .functions import magnetic_moment_operator

#%%
def Ising(Ops,couplings=1.0,nn=1):
    H = 0
    N = len(Ops)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    if hasattr(couplings,'__len__') == False :
        Js = np.full(N,couplings)
    else :
        Js = couplings
        
    for i,j,J in zip(index_I,index_J,Js):
        H +=J * Ops[i]@Ops[j]
        
    return H

#%%
def Heisenberg(Sx,Sy,Sz,couplings=1.0,nn=1):
    N = len(Sx)
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
    
    return Ising(Sx,Js[:,0],nn) +\
           Ising(Sy,Js[:,1],nn) +\
           Ising(Sz,Js[:,2],nn)

#%%
def DM(Sx,Sy,Sz,couplings=1.0,nn=1):
    H = 0
    N = len(Sx)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
        
    for i,j,J in zip(index_I,index_J,Js):
        H += J[0] * (Sy[i]@Sz[j]-Sz[i]@Sy[j]) 
        H += J[1] * (Sz[i]@Sx[j]-Sx[i]@Sz[j]) 
        H += J[2] * (Sx[i]@Sy[j]-Sy[i]@Sx[j]) 
        
    return H

#%%
def anisotropy(Ops,couplings):
    return Ising(Ops,couplings,nn=0)

def rombicity(Sx,Sy,couplings):
    return Ising(Sx,couplings,nn=0) - Ising(Sy,couplings,nn=0)

#%%
def Zeeman(Sx,Sy,Sz,B):
    B = np.asarray(B)
    Mx,My,Mz = magnetic_moment_operator(Sx,Sy,Sz)    
    return - ( Mx*B[0] + My*B[1] + Mz*B[2] )
