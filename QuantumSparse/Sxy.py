# compute Sx,Sy,Sz operators
import numpy as np
from scipy import sparse

#%%
def compute_sx(p,m):
    return 1.0/2.0*(p+m)

def compute_sy(p,m):
    return -1.j/2.0*(p-m) 

#%%
def compute_Sxy_operators(NSpin,iden,sz,sp,sm):
    Sz = np.zeros(NSpin,dtype=object) # S z
    Sx = np.zeros(NSpin,dtype=object) # S x
    Sy = np.zeros(NSpin,dtype=object) # S y
    
    for i in range(NSpin):

        print("\t",i+1,"/",NSpin,end="\r")
        
        if i!=0: #i!=0
            mz = iden[0].copy() # matrix z
            mp = iden[0].copy() # matrix plus
            mm = iden[0].copy() # matrix minus
            
            for j in range(1,i):
                mz = sparse.kron(mz,iden[j])
                mp = sparse.kron(mp,iden[j])
                mm = sparse.kron(mm,iden[j])
                
            mz = sparse.kron(mz,sz[i])
            mp = sparse.kron(mp,sp[i])
            mm = sparse.kron(mm,sm[i])
            
            for j in range(i+1,NSpin):
                mz = sparse.kron(mz,iden[j])
                mp = sparse.kron(mp,iden[j])
                mm = sparse.kron(mm,iden[j])
            
        else : #i==0    
        
            mz = sz[0].copy()
            mp = sp[0].copy()
            mm = sm[0].copy()      
            
            for j in range(1,NSpin):
                mz = sparse.kron(mz,iden[j])
                mp = sparse.kron(mp,iden[j])
                mm = sparse.kron(mm,iden[j])
        #
        mx = compute_sx(mp,mm)
        my = compute_sy(mp,mm)   

        Sz[i] = mz.copy()       
        Sx[i] = mx.copy()
        Sy[i] = my.copy()
    
    return Sx,Sy,Sz
