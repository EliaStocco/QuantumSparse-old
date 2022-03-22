# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse

#%%
def compute_sx(p,m):
    return 1.0/2.0*(p+m)

def compute_sy(p,m):
    return -1.j/2.0*(p-m)   

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
    iden = np.zeros(NSpin,dtype=object)
    Iden = None
    
    for i,s,deg in zip(range(NSpin),SpinValues,Degeneracies):
        print("\t",i+1,"/",NSpin,end="\r")
        
        iden[i] = sparse.diags(np.full(deg,1,dtype=int),dtype=int)    
        if Iden is None :
            Iden = iden[i].copy()
        else :
            Iden = sparse.kron(Iden,iden[i])
    
    print("\t\tdone   ")
        
    print("\n\t\tallocating single Sz,S+,S- operators (on the single-spin Hilbert space) ... ")
    sz = np.zeros(NSpin,dtype=object) # s z
    sp = np.zeros(NSpin,dtype=object) # s plus
    sm = np.zeros(NSpin,dtype=object) # s minus
    if opts["compute_xy"] :
        sx = np.zeros(NSpin,dtype=object) # s x
        sy = np.zeros(NSpin,dtype=object) # s y
    
    for i,s,deg in zip(range(NSpin),SpinValues,Degeneracies):
    
        print("\t\t",i+1,"/",NSpin,end="\r")
        
        m = np.linspace(s,-s,deg)
        sz[i] = sparse.diags(m,dtype=float)          
        
        vp = np.sqrt( (s-m)*(s+m+1) )[1:]
        vm = np.sqrt( (s+m)*(s-m+1) )[0:-1]
        sp[i] = sparse.diags(vp,offsets=1 )
        sm[i] = sparse.diags(vm,offsets=-1)
        
        if opts["compute_xy"] :
            sx[i] = compute_sx(sp[i],sm[i])
            sy[i] = compute_sy(sp[i],sm[i])
            
    print("\t\tdone   ")    
    print("\n\t\tallocating single Sz,S+,S- operators (on the system Hilbert space) ... ")  
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
        
    print("\t\tdone   ")    
    return Sx,Sy,Sz
    
#%%
    
    