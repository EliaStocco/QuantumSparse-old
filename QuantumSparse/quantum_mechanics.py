# quantum mechanis
from scipy import sparse

#%%
def expectation_value(Op,Psi):
    V  = sparse.csr_matrix(Psi)
    Vc = V.conjugate(True)
    return ((Op @ V).multiply(Vc)).toarray().real.sum(axis=0)
