import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
# from . import dtype
from ..tools.functional import output

dtype = csr_matrix

class get_class(type):
    def __new__(cls, name, bases, attrs):
        # Iterate through the attributes of the class
        if issubclass(dtype, sparse.spmatrix) :
            attrs["module"] = sparse
        else :
            raise ValueError("not implemented yet")
        
        bases = bases + (dtype,)

        # Create the class using the modified attributes
        return super().__new__(cls, name, bases, attrs)
    

class matrix(metaclass=get_class):
# class matrix(csr_matrix):
    """class to handle matrices in different form, i.e. dense, sparse, with 'numpy', 'scipy', or 'torch'"""

    module = sparse

    def __init__(self,*argc,**argv):
        # https://realpython.com/python-super/
        super().__init__(*argc,**argv)
        pass

    #@staticmethod
    @classmethod
    def diags(cls,*argc,**argv):
        """diagonal matrix"""
        return cls(matrix.module.diags(*argc,**argv))

    @classmethod
    def kron(cls,*argc,**argv):
        """kronecker product"""
        return cls(matrix.module.kron(*argc,**argv))

    @classmethod
    def identity(cls,*argc,**argv):
        """identity operator"""
        return cls(matrix.module.identity(*argc,**argv))

    # def __len__(self):

    