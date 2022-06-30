# the core of QuantumSparse code: a module defining spin operators via Kronecker (tensor) product
import numpy as np
from scipy import sparse
#from .identity import compute_identity_operator
#from .Sxy import compute_Sxy_operators
#from .Szpm import compute_Szpm_operators
from .operators import operators
from .functions import prepare_opts

#%%
class spin_operators(operators,object):
    
    def __init__(self,N=1,S=0.5,spin_values=None,opts=None,*args,**kwargs):
        """
        Parameters
        ----------
        N : int, optional
            number of spin sites
        S : float, optional
            (site-independent) spin value: it must be integer of semi-integer
        spin_values : numpy.array, optional
            numpy.array containing the spin values for each site: they can be integer of semi-integer
            
        Returns
        -------
        None
        """
        
        print("\n\tconstructor of \"SpinSystem\" class")     
        opts = prepare_opts(opts)
        if spin_values is not None:
            self.SpinValues = spin_values
        else :
            self.SpinValues = np.full(N,S)
     
        check = [ not (i%1) for i in self.SpinValues*2 ]
        if not np.all(check):
            print("\n\terror: not all spin values are integer or semi-integer: ",self.SpinValues)
            raise() 
            
        Sx,Sy,Sz = self.compute_spin_operators(self.SpinValues,opts)
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz  
        
        # https://realpython.com/python-super/
        super().__init__(*args, **kwargs)
        
        return
        

    #%%
    @staticmethod
    def compute_spin_operators(SpinValues,opts=None):
        """
        Parameters
        ----------
        SpinValues : numpy.array
            numpy.array containing the spin values for each site: 
            they can be integer of semi-integer
        opts : dict, optional
            dict containing different options

        Returns
        -------
        Sx : np.array of scipy.sparse
            array of the Sx operators for each spin site, 
            represented with sparse matrices,
            acting on the system (all sites) Hilbert space
        Sy : np.array of scipy.sparse
            array of the Sy operators for each spin site, 
            represented with sparse matrices,
            acting on the system (all sites) Hilbert space
        Sz : np.array of scipy.sparse
            array of the Sz operators for each spin site, 
            represented with sparse matrices,
            acting on the system (all sites) Hilbert space
        """
        
        opts = prepare_opts(opts)
        SpinValues = np.asarray(SpinValues)    
        from_list_to_str = lambda x :  '[ '+ ' '.join([str(i)+" ," for i in x ])[0:-1]+' ]'
            
        print("\n\t\"compute_spin_operators\" function",file=opts["print"])
        print("\n\t\tinput parameters:",file=opts["print"])
        print("\t\t{:>20s}\t:\t{:<60s}".format("spin values",from_list_to_str(SpinValues)),file=opts["print"])
            
        NSpin        = len(SpinValues)     
        print("\t\t{:>20s}\t:\t{:<60d}".format("N spins",NSpin),file=opts["print"])
        
        dimensions = spin_operators.dimensions(SpinValues)#(2*SpinValues+1).astype(int)
        print("\t\t{:>20s}\t:\t{:<60s}".format("dimensions",from_list_to_str(dimensions)),file=opts["print"])
       
        print("\n\t\tallocating single Sz,S+,S- operators (on the single-spin Hilbert space) ... ",file=opts["print"])
        sz,sp,sm = spin_operators.compute_Szpm_operators(SpinValues)
        print("\t\tdone   ",file=opts["print"])    
        
        print("\n\t\tallocating the Sx,Sy,Sz operators (on the system Hilbert space) ... ",file=opts["print"])  
        Sx,Sy,Sz = spin_operators.compute_Sxy_operators(dimensions,sz,sp,sm)
        print("\t\tdone   ",file=opts["print"])    
        
        return Sx,Sy,Sz 
    
    #%%
    @staticmethod
    def dimensions(SpinValues):
        """
        Parameters
        ----------
        SpinValues : numpy.array
            numpy.array containing the spin values for each site: 
                they can be integer of semi-integer

        Returns
        -------
        deg : numpy.array
            array of int representing the Hilbert space dimension for each site
        """
        deg = (2*SpinValues+1).astype(int) 
        return deg
    
    #%%
    @staticmethod
    def compute_S2(Sx,Sy,Sz,opts=None):
        """        
        Parameters
        ----------
        Sx : scipy.sparse
            Sx operators of a single spin site, represented with a sparse matrice.
            This operator can be computed by the function "compute_spin_operators"
        Sy : scipy.sparse
            Sy operators of a single spin site, represented with a sparse matrice.
            This operator can be computed by the function "compute_spin_operators"
        Sz : scipy.sparse
            Sz operators of a single spin site, represented with a sparse matrice.
            This operator can be computed by the function "compute_spin_operators"
        opts : dict, optional
            dict containing different options

        Returns
        -------
        S2 : scipy.sparse
            spin square operator 
        """
        S2 = Sx@Sx +   Sy@Sy +  Sz@Sz
        return S2
    
    #%%
    @staticmethod
    def compute_total_S2(Sx,Sy,Sz,opts=None):
        """        
        Parameters
        ----------
        Sx : np.array of scipy.sparse
            array of the Sx operators for each spin site, represented with sparse matrices.
            This array of operators can be computed by the function "compute_spin_operators"
        Sy : np.array of scipy.sparse
            array of the Sy operators for each spin site, represented with sparse matrices.
            This array of operators can be computed by the function "compute_spin_operators"
        Sz : np.array of scipy.sparse
            array of the Sz operators for each spin site, represented with sparse matrices.
            This array of operators can be computed by the function "compute_spin_operators"
        opts : dict, optional
            dict containing different options

        Returns
        -------
        S2 : scipy.sparse
            total spin square operator 
        """
        SxTot= spin_operators.sum(Sx)
        SyTot= spin_operators.sum(Sy)
        SzTot= spin_operators.sum(Sz)
        S2 = SxTot@SxTot +   SyTot@SyTot +  SzTot@SzTot
        return S2
            
    #%%
    @staticmethod
    def from_S2_to_S(S2):
        """
        Parameters
        ----------
        S2 : float
            eigenvalue the S2 operator

        Returns
        -------
        S : float
            spin value (integer of semi-integer value) such that S(S+1) = S2
        """
        S = (-1 + np.sqrt(1+ 4*S2))/2.0
        return S

    #%%
    @staticmethod
    def compute_sx(p,m):
        """
        Parameters
        ----------
        p : scipy.sparse
            spin raising operator S+ = Sx + i Sy
        m : scipy.sparse
            spin lowering operator S- = Sx - i Sy

        Returns
        -------
        Sx : scipy.sparse
            Sx operator, computed fromthe inversion of the S+ and S- expressions
        """
        Sx = 1.0/2.0*(p+m)
        return Sx

    #%%
    @staticmethod
    def compute_sy(p,m):
        """
        Parameters
        ----------
        p : scipy.sparse
            spin raising operator S+ = Sx + i Sy
        m : scipy.sparse
            spin lowering operator S- = Sx - i Sy

        Returns
        -------
        Sy : scipy.sparse
            Sy operator, computed fromthe inversion of the S+ and S- expressions
        """
        Sy = -1.j/2.0*(p-m) 
        return Sy
    
    #%%
    @staticmethod
    def compute_Szpm_operators(SpinValues):
        """
        Parameters
        ----------
        SpinValues : numpy.array
            numpy.array containing the spin values for each site: they can be integer of semi-integer

        Returns
        -------
        Sz : numpy.array of scipy.sparse
            array of Sz operators for each site,
            acting on the local (only one site) Hilbert space
        Sp : numpy.array of scipy.sparse
            array of the raising S+ operators for each site,
            acting on the local (only one site) Hilbert space
        Sm : numpy.array of scipy.sparse
            array of lowering Sz operators for each site,
            acting on the local (only one site) Hilbert space
        """
        NSpin = len(SpinValues)
        Sz = np.zeros(NSpin,dtype=object) # s z
        Sp = np.zeros(NSpin,dtype=object) # s plus
        Sm = np.zeros(NSpin,dtype=object) # s minus
        dimensions = spin_operators.dimensions(SpinValues)  
        
        for i,s,deg in zip(range(NSpin),SpinValues,dimensions):
            
            print("\t\t",i+1,"/",NSpin,end="\r")
            
            m = np.linspace(s,-s,deg)
            Sz[i] = sparse.diags(m,dtype=float)          
            
            vp = np.sqrt( (s-m)*(s+m+1) )[1:]
            vm = np.sqrt( (s+m)*(s-m+1) )[0:-1]
            Sp[i] = sparse.diags(vp,offsets= 1)
            Sm[i] = sparse.diags(vm,offsets=-1)
    
        return Sz,Sp,Sm

    #%%
    @staticmethod
    def compute_Sxy_operators(dimensions,sz,sp,sm):
        """
        Parameters
        ----------
        dimensions : numpy.array
            numpy.array of integer numbers representing the Hilbert space dimension of each site 
        sz : numpy.array of scipy.sparse
            array of Sz operators for each site,
            acting on the local (only one site) Hilbert space
        sp : numpy.array of scipy.sparse
            array of the raising S+ operators for each site,
            acting on the local (only one site) Hilbert space
        sm : numpy.array of scipy.sparse
            array of lowering S- operators for each site,
            acting on the local (only one site) Hilbert space

        Returns
        -------
        Sx : numpy.array of scipy.sparse
            array of Sx operators for each site,
            acting on the system Hilbert space
        Sy : numpy.array of scipy.sparse
            array of Sy operators for each site,
            acting on the system Hilbert space
        Sz : numpy.array of scipy.sparse
            array of Sz operators for each site,
            acting on the system Hilbert space
        """
        NSpin= len(sz)
        if NSpin != len(sp) or NSpin != len(sm) or NSpin != len(dimensions):
            print("\t\terror in \"compute_Sxy_operators\" function: arrays of different lenght")
            raise()
        Sz = np.zeros(NSpin,dtype=object) # S z
        Sx = np.zeros(NSpin,dtype=object) # S x
        Sy = np.zeros(NSpin,dtype=object) # S y
        Sp = np.zeros(NSpin,dtype=object) # S y
        Sm = np.zeros(NSpin,dtype=object) # S y
        iden = spin_operators.identity(dimensions)
        
        for zpm,out in zip([sz,sp,sm],[Sz,Sp,Sm]):
            for i in range(NSpin):
                Ops = iden.copy()
                Ops[i] = zpm[i]
                out[i] = Ops[0]
                for j in range(1,NSpin):
                    out[i] = sparse.kron(out[i],Ops[j]) 
                    
        for i in range(NSpin):
            Sx[i] = spin_operators.compute_sx(Sp[i],Sm[i])
            Sy[i] = spin_operators.compute_sy(Sp[i],Sm[i])
            
        return Sx,Sy,Sz             
        
    
        # for i in range(NSpin):
    
        #     print("\t",i+1,"/",NSpin,end="\r")
            
        #     if i!=0: #i!=0
        #         mz = iden[0].copy() # matrix z
        #         mp = iden[0].copy() # matrix plus
        #         mm = iden[0].copy() # matrix minus
                
        #         for j in range(1,i):
        #             mz = sparse.kron(mz,iden[j])
        #             mp = sparse.kron(mp,iden[j])
        #             mm = sparse.kron(mm,iden[j])
                    
        #         mz = sparse.kron(mz,sz[i])
        #         mp = sparse.kron(mp,sp[i])
        #         mm = sparse.kron(mm,sm[i])
                
        #         for j in range(i+1,NSpin):
        #             mz = sparse.kron(mz,iden[j])
        #             mp = sparse.kron(mp,iden[j])
        #             mm = sparse.kron(mm,iden[j])
                
        #     else : #i==0    
            
        #         mz = sz[0].copy()
        #         mp = sp[0].copy()
        #         mm = sm[0].copy()      
                
        #         for j in range(1,NSpin):
        #             mz = sparse.kron(mz,iden[j])
        #             mp = sparse.kron(mp,iden[j])
        #             mm = sparse.kron(mm,iden[j])
        #     #
        #     mx = spin_operators.compute_sx(mp,mm)
        #     my = spin_operators.compute_sy(mp,mm)   
    
        #     Sz[i] = mz.copy()       
        #     Sx[i] = mx.copy()
        #     Sy[i] = my.copy()
        
        return Sx,Sy,Sz
