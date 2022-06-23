# quantum_system class
from .functions import prepare_opts
from scipy import sparse
import numpy as np
#%%
class quantum_system:
    
    #%%
    def __init__(self,H):
        self.Hamiltonian = H
        self.eigenstates = None
        self.eigenvalues = None
        
    #%%
    def ground_state(self,H=None,tol=1E-8,MaxDim=100,opts=None):
        return self.diagonalize(H,1,tol,MaxDim,opts)
        
    #%%
    def diagonalize(self,H=None,NLanczos=100,tol=1E-8,MaxDim=100,opts=None):
        if H is None :
            H = self.Hamiltonian
            
        opts = prepare_opts(opts)    
        print("\n\t\"diagonalize_Hamiltonian\" function")    
        dimension = H.shape[0]   
        print("\t\t{:>40s}\t:{:>6d} x {:<10d}".format("Hamilonian matrix of dimension",dimension,dimension))
        print("\t\t{:>40s}\t:{:>6.2E}".format("Lanczos tolerance",tol))
        print("\t\t{:>40s}\t:{:>6d}".format("Lanczos n. of eigenvalues",NLanczos))
        print("\t\t{:>40s}\t:{:>6d}".format("Apply full diagonalization up to dimension",MaxDim))
        
        print("")
        NLanczos= min ( NLanczos , H.shape[0]-1)
        
        if dimension >= MaxDim :
            print("\t\tusing Lanczos method")
            print("\t\t{:>40s}\t:\t{:<10d}".format("n. eigenvalues in Lanczos method",NLanczos))
                    
            E,Psi = sparse.linalg.eigsh(H,k=NLanczos,tol=tol,which="SA")
            # SA : Smallest Algebraic
        else :
            print("\t\tusing a full diagonalization method")
            E,Psi = np.linalg.eigh(H.todense())
            
        # check that eigevectors are orthogonalized: 
        # np.sqrt(np.square(np.absolute(Psi)).sum(axis=0)) = [1,1,1,1...]
            
        # sort
        if opts["sort"] :
            index = np.argsort(E)
            E = E[index]
            Psi = Psi[:,index]
            
        if opts["inplace"] :
            self.eigenvalues,self.eigenstates = E,Psi   
            
        return E,Psi