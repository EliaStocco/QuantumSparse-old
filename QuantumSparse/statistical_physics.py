# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:37:22 2022

@author: Elia Stocco
"""
#%%
import numpy as np
from scipy import sparse

#%%
def partition_function(T,E,beta=None):
    global kB 
    if beta is None:
        beta = 1.0/(kB*T)
    return np.exp(-np.tensordot(beta,E-min(E),axes=0)).sum(axis=1)

def classical_thermal_average_value(T,E,Obs):
    global kB 
    beta = 1.0/(kB*T)
    Z = partition_function(T=None,E=E,beta=beta)
    return (np.exp(-np.tensordot(beta,E-min(E),axes=0))*Obs).sum(axis=1)/Z

def quantum_thermal_average_value(T,E,Op,Psi):
    V  = sparse.csr_matrix(Psi)
    Vc = V.conjugate(True)
    Obs = ((Op @ V).multiply(Vc)).toarray().real.sum(axis=0)
    return classical_thermal_average_value(T,E,Obs)

def susceptibility(T,E,OpAs,OpBs,Psi,meanA=None,meanB=None):
    
    #
    global kB 
    NT = len(T)
    
    #
    if meanA is None :
        meanA =  np.zeros((len(OpAs),NT))       
        for n,Op in enumerate(OpAs):
            meanA[n] = quantum_thermal_average_value(T,E,Op,Psi)
            
    if meanB is None :
        meanB =  np.zeros((len(OpBs),NT))
        for n,Op in enumerate(OpBs):
            meanB[n] = quantum_thermal_average_value(T,E,Op,Psi)
            
         
    #
    square =  np.zeros((len(OpAs),len(OpBs),NT))
    for n1,OpA in enumerate(OpAs):
        for n2,OpB in enumerate(OpBs):
        
            if n2 < n1 :
                square[n1,n2] =  square[n2,n1]
                continue
            
            square[n1,n2] = quantum_thermal_average_value(T,E,OpA@OpB,Psi)
            
    #
    Chi =  np.zeros((3,3,NT))    
    beta  = 1.0/(kB*T)
    for n1,OpA in enumerate(OpAs):
        for n2,OpB in enumerate(OpBs):
            
            Chi[n1,n2] = (square[n1,n2] - meanA[n1]*meanB[n2])
                
    return 10000 * Chi*beta # cm^{-3}

