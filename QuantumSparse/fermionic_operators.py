# "fermionic_operators" class
from .operators import operators
from .functions import prepare_opts
import numpy as np
import pandas as pd
from scipy import sparse

#%%
class fermionic_operators(operators):
    
    #%%
    def __init__(self,N=1,deg_names=["spin"],deg_keys=[["up","dw"]],opts=None):
        
        print("\n\tconstructor of \"fermionic_operators\" class")   
        opts = prepare_opts(opts)
        
        self.QuantumNumbers = deg_names
        Nd = len(deg_names)
        
        if Nd != len(deg_keys) :
            print("\t\terror: array of different lenghts")
            raise()
        if not all(isinstance(x, str) for x in deg_names):
            print("\t\terror: each element of \"deg_names\" must be a string")
            raise()
        # for i in deg_keys:
        #     if not all(isinstance(x, str) for x in i):
        #         print("\t\terror: each element of \"deg_keys\" must be list of strings")
        #         raise()
        
        self.quantum_numbers_keys = {}
        for n,i in enumerate(deg_names):
            self.quantum_numbers_keys[i] = deg_keys[n]
            
        if N > 1 :
           self.quantum_numbers = ["site"] +  self.quantum_numbers            
          
        columns = pd.MultiIndex.from_product(deg_keys, names=deg_names)
        index = np.arange(0,N)
        self.columns = columns
        self.index = index
        self.operators_template = pd.DataFrame(columns=columns,index=index,dtype=object)
        self.constructor_operators = self.operators_template.copy()
        self.destructor_operators  = self.operators_template.copy()
                
        c_dag = fermionic_operators.constructor_fermionic_operator()
        c     = fermionic_operators.destructor_fermionic_operator()
        iden  = fermionic_operators.identity(2)
        
        all_index = list(pd.MultiIndex.from_product([list(self.index)]+deg_keys, names=["site"]+deg_names))
        Iden = np.full(len(all_index),iden)
        C    = Iden.copy()
        Cdag = Iden.copy()
        
        for i in self.index: #sites
            for j in self.columns: #quantum numbers
                k = all_index.index(tuple([i]+list(j)))
                C[k] = c
                Cdag [k] = c_dag
                
                for In,Out in zip([Cdag,C],[self.constructor_operators,self.destructor_operators]):
                    Out.at[i,j] = In[0]
                    for n in range(1,len(In)):
                        Out.at[i,j] = Out.at[i,j] @ In[n]
                
                C[k] = iden
                Cdag[k] = iden
            
        return        
        
    #%%
    def n_quantum_numbers(self):
        N = len(self.QuantumNumbers)
        return N
    
    #%%
    def index(self,key=None):
        if key is None:
            out = self.index
        elif key not in self.index.keys():
            print("\n\terror in \"index\" method: key \"",key,"\" does not exists")
            raise()
        else :
            out = self.index[key]
        return out
    
    #%%
    def number_operators(self,sum_over=None):
        N = self.operators_template.copy()
        for i in self.index: #sites
            for j in self.columns: #quantum numbers
                N.at[i,j] = self.constructor_operators.at[i,j] @ self.destructor_operators.at[i,j]
        return N        
    
    #%%
    @staticmethod
    def constructor_fermionic_operator():
        c_dag = sparse.diags([1],offsets= 1).todense()
        return c_dag
            
    #%%
    @staticmethod
    def destructor_fermionic_operator():
        c = sparse.diags([1],offsets=-1).todense()
        return c       