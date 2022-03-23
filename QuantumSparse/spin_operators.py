# the core of QuantumSparse code: a module defining spin operators via Kronecker (tensor) product
import numpy as np
from .identity import compute_identity_operator
from .Sxy import compute_Sxy_operators
from .Szpm import compute_Szpm_operators

#%%
def compute_spin_operators(SpinValues,opts=None):
    
    SpinValues = np.asarray(SpinValues)
    
    if opts is None:
        opts = {}
    if "compute_xy" not in opts:
        opts["compute_xy"] = False
        
    from_list_to_str = lambda x :  '[ '+ ' '.join([str(i)+" ," for i in x ])[0:-1]+' ]'
        
    print("\n\t\"compute_spin_operators\" function")
    print("\n\t\tinput parameters:")
    print("\t\t{:>20s}\t:\t{:<60s}".format("spin values",from_list_to_str(SpinValues)))
        
    NSpin        = len(SpinValues)     
    print("\t\t{:>20s}\t:\t{:<60d}".format("N spins",NSpin))
    
    Degeneracies = (2*SpinValues+1).astype(int)
    print("\t\t{:>20s}\t:\t{:<60s}".format("degeneracies",from_list_to_str(Degeneracies)))
   
    print("\n\t\tallocating identity Id operators...")
    iden = compute_identity_operator(NSpin,SpinValues,Degeneracies)
    print("\t\tdone   ")
        
    print("\n\t\tallocating single Sz,S+,S- operators (on the single-spin Hilbert space) ... ")
    sz,sp,sm = compute_Szpm_operators(NSpin,SpinValues,Degeneracies)
    print("\t\tdone   ")    
    
    print("\n\t\tallocating single Sz,S+,S- operators (on the system Hilbert space) ... ")  
    Sx,Sy,Sz = compute_Sxy_operators(NSpin,iden,sz,sp,sm)
    print("\t\tdone   ")    
    
    return Sx,Sy,Sz    
    