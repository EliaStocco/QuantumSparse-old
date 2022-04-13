# some function recalling statistical physics results
import numpy as np
from .physical_constants import kB,muB
from .quantum_mechanics import expectation_value

#%%
def partition_function(T,E,beta=None):
    if beta is None:
        beta = 1.0/(kB*T)
    return np.exp(-np.tensordot(beta,E-min(E),axes=0)).sum(axis=1)

def classical_thermal_average_value(T,E,Obs):
    beta = 1.0/(kB*T)
    Z = partition_function(T=None,E=E,beta=beta)
    return (np.exp(-np.tensordot(beta,E-min(E),axes=0))*Obs).sum(axis=1)/Z

def quantum_thermal_average_value(T,E,Op,Psi):
    Obs = expectation_value(Op,Psi)
    return classical_thermal_average_value(T,E,Obs)

def correlation_function(T,E,OpAs,OpBs,Psi):    
    NT = len(T)    
    meanA =  np.zeros((len(OpAs),NT))       
    for n,Op in enumerate(OpAs):
        meanA[n] = quantum_thermal_average_value(T,E,Op,Psi)
    #    
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
    for n1,OpA in enumerate(OpAs):
        for n2,OpB in enumerate(OpBs):            
            Chi[n1,n2] = (square[n1,n2] - meanA[n1]*meanB[n2])
    
    return Chi

def susceptibility(T,E,OpAs,OpBs,Psi):#,meanA=None,meanB=None):    
    beta  = 1.0/(kB*T)
    Chi = correlation_function(T,E,OpAs,OpBs,Psi) 
    NA =  6.02214076 # E+23 1/mol
    eV =  1.602176634 #E-19 J 
    return beta * Chi * NA * eV * 1E3           
    # return 1.602176634*6.02214076*Chi*beta*1E4 # cm^{3}/mol
    # new_muB = 9.274009994 #E-27 erg/G
    # new_kB  = 1.380649    #E-16 erg/K
    # NA      = 6.02214076  #E+23 1/mol
    # beta  = 1.0/(new_kB*T)
    # return (4*np.pi)*NA * beta * Chi * 1E-15

