# the most adopted interactions in Spin Hamiltonians
import numpy as np
from .functions import magnetic_moment_operator

#%%
def Row_by_Col_mult(A,B,opts=None):
    """
    Row by Columns multiplication
    """
    if opts is None :
        opts = {}
    if "sympy" in opts and opts["sympy"] == True :
        opts["function"] = lambda a,b : a*b
    elif "function" not in opts:
        opts["function"] = lambda a,b : a@b
        
    return opts["function"](A,B)
        

#%%
def Ising(Ops,couplings=1.0,nn=1,opts=None):
    H = 0
    N = len(Ops)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    if hasattr(couplings,'__len__') == False :
        Js = np.full(N,couplings)
    else :
        Js = couplings
        
    for i,j,J in zip(index_I,index_J,Js):
        H +=J * Row_by_Col_mult(Ops[i],Ops[j],opts=opts)
        
    return H

#%%
def Heisenberg(Sx,Sy,Sz,couplings=1.0,nn=1,opts=None):
    N = len(Sx)
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
    
    return Ising(Sx,Js[:,0],nn,opts=opts) +\
           Ising(Sy,Js[:,1],nn,opts=opts) +\
           Ising(Sz,Js[:,2],nn,opts=opts)

#%%
def DM(Sx,Sy,Sz,couplings=1.0,nn=1,opts=None):
    H = 0
    N = len(Sx)
    index_I = np.arange(0,N)
    index_J = np.asarray([ j+nn if j+nn < N else j+nn-N for j in index_I ])
    Js = np.asarray(couplings)
    if len(Js.shape) != 2 : 
        Js = np.full((N,3),couplings)
        
    RbC = lambda a,b : Row_by_Col_mult(a,b,opts=opts)
        
    for i,j,J in zip(index_I,index_J,Js):
        H += J[0] * ( RbC(Sy[i],Sz[j]) - RbC(Sz[i],Sy[j])) 
        H += J[1] * ( RbC(Sz[i],Sx[j]) - RbC(Sx[i],Sz[j]))
        H += J[2] * ( RbC(Sx[i],Sy[j]) - RbC(Sy[i],Sx[j])) 
        
    return H

#%%
def anisotropy(Ops,couplings,opts=None):
    return Ising(Ops,couplings,nn=0,opts=opts)

def rombicity(Sx,Sy,couplings,opts=None):
    return Ising(Sx,couplings,nn=0,opts=opts) - Ising(Sy,couplings,nn=0,opts=opts)

#%%
def Zeeman(Sx,Sy,Sz,B,opts=None):
    B = np.asarray(B)
    Mx,My,Mz = magnetic_moment_operator(Sx,Sy,Sz,opts)    
    return - ( Mx*B[0] + My*B[1] + Mz*B[2] )
