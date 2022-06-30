# "operator" class
from scipy import sparse
import numpy as np

#%%
class operators(object):
    
    #%%
    def __init__(self,*args,**kwargs):
        # https://realpython.com/python-super/
        super().__init__(*args,**kwargs)
        
    #%%
    
    @staticmethod
    def identity(dimensions):
        """
        Parameters
        ----------
        dimensions : numpy.array
            numpy.array of integer numbers representing the Hilbert space dimension of each site 

        Returns
        -------
        iden : numpy.array of scipy.sparse
            array of the identity operator for each site, represented with sparse matrices,
            acting on the local (only one site) Hilbert space
        """
        if not hasattr(dimensions, '__len__'):
            iden = operators.identity([dimensions])[0]
            return iden
        else :            
            N = len(dimensions)
            iden = np.zeros(N,dtype=object)
            for i,dim in zip(range(N),dimensions):
                #print("\t",i+1,"/",N,end="\r")        
                #iden[i] = sparse.diags(np.full(dim,1,dtype=int),dtype=int)  
                iden[i] = sparse.identity(dim,dtype=int)  
            return iden
    
    @staticmethod
    def sum(Ops):
        """
        Parameters
        ----------
        Ops : np.array of scipy.sparse
            array of operator to be summed,
            each acting on the system Hilbert space
        
        Returns
        -------
        tot : scipy.sparse
            sum of given operator
        """
        dims = [ Op.shape for Op in Ops ]
        boolean = [ dim == dims[0] for dim in dims ]
        if not np.all(boolean) :
            print("\t\terror in \"sum\" function: not all operator with the same (matrix representation) dimensions")
            raise()
        tot = 0 
        for Op in Ops:
            tot += Op
        return tot
    
    #%%
    @staticmethod
    def commutator(A,B):
        C = A @ B - B @ A 
        return C
    
    #%%
    @staticmethod
    def anticommutator(A,B):
        C = A @ B + B @ A 
        return C
    
#%%
# class quantum_operator(sparse.spmatrix):
    
#     #%%
#     def __init__(self, *args):
#         sparse.spmatrix.__init__(self,args)
        
#     #%%
#     def is_diagonal(self):
#         #if sparse :
#         A = self.tolil()
#         A.setdiag(np.zeros(A.shape[0]))
#         return A.count_nonzero() == 0
#         # else :
#         #     print("not yet implemented")
#         #     raise()   
