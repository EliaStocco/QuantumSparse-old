# some function recalling statistical physics results
import numpy as np
from scipy import sparse
from .quantum_mechanics import expectation_value

#%% correlation functions
def correlation_function(OpA,OpB,Psi):

    meanA =  expectation_value(OpA,Psi)
    meanB =  expectation_value(OpB,Psi)
    square = expectation_value(OpA@OpB,Psi)
    
    return square - meanA*meanB


