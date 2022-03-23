# compute identity operators
import numpy as np
from scipy import sparse

#%%
def compute_identity_operator(NSpin,SpinValues,Degeneracies):
    
    
    iden = np.zeros(NSpin,dtype=object)
    
    for i,s,deg in zip(range(NSpin),SpinValues,Degeneracies):
        print("\t",i+1,"/",NSpin,end="\r")
        
        iden[i] = sparse.diags(np.full(deg,1,dtype=int),dtype=int)    
        
    return iden