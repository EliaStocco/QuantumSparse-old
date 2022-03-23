# compute Sz,S+,S- operators
import numpy as np
from scipy import sparse

#%%
def compute_Szpm_operators(NSpin,SpinValues,Degeneracies):
    sz = np.zeros(NSpin,dtype=object) # s z
    sp = np.zeros(NSpin,dtype=object) # s plus
    sm = np.zeros(NSpin,dtype=object) # s minus
    
    for i,s,deg in zip(range(NSpin),SpinValues,Degeneracies):
    
        print("\t\t",i+1,"/",NSpin,end="\r")
        
        m = np.linspace(s,-s,deg)
        sz[i] = sparse.diags(m,dtype=float)          
        
        vp = np.sqrt( (s-m)*(s+m+1) )[1:]
        vm = np.sqrt( (s+m)*(s-m+1) )[0:-1]
        sp[i] = sparse.diags(vp,offsets=1 )
        sm[i] = sparse.diags(vm,offsets=-1)

    return sz,sp,sm
