# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:39:13 2022

@author: Elia Stocco
"""
#%%
from scipy import sparse
from scipy.sparse import linalg
import numpy as np

#%% 
def diagonalize_Hamiltonian(H,NLanczos=100,tol=1E-8,MaxDim=100):
    
    print("\n\t\"diagonalize_Hamiltonian\" function")
    
    dimension = H.shape[0]
    
    #print("\n\t\tHamilonian matrix of dimension %sx%s"%(dimension,dimension))
    
    print("\t\t{:>40s}\t:\t{:<10d}x{:<10d}".format("Hamilonian matrix of dimension",dimension,dimension))
    print("\t\t{:>40s}\t:\t{:<10.2E}".format("Lanczos tolerance",tol))
    print("\t\t{:>40s}\t:\t{:<10d}".format("Lanczos n. of eigenvalues",NLanczos))
    print("\t\t{:>40s}\t:\t{:<10d}".format("Apply full diagonalization up to dimension",MaxDim))
    
    print("")
    NLanczos= min ( NLanczos , H.shape[0]-2)
    
    if dimension >= MaxDim :
        print("\t\tusing Lanczos method")
        print("\t\t{:>40s}\t:\t{:<10d}".format("n. eigenvalues in Lanczos method",NLanczos))
        E,Psi = sparse.linalg.eigsh(H,k=NLanczos,tol=1E-8,which="SA")
    else :
        print("\t\tusing a full diagonalization method")
        #print("\t\t{:>40s}".format("full diagonalization"))
        E,Psi = np.linalg.eig(H.todense())
        
    # sort
    index = np.argsort(E)
    E = E[index]
    Psi = Psi[index,:]
        
    return E,Psi
