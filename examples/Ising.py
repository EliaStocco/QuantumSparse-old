from QuantumSparse.spin.spin_operators import spin_operators
import numpy as np
from QuantumSparse import operator
from QuantumSparse.spin.interactions import Heisenberg, Ising
from scipy import sparse

def main():
   
    S     = 1./2
    NSpin = 4
    SpinValues = np.full(NSpin,S)

    spins = spin_operators(SpinValues)

    H = Heisenberg(spins=spins)
    print(H.shape)
    print(H.sparsity())
    # print(H.todense())

    eigenvalues, eigenstates = H.diagonalize()

    # H = Ising(spins.Sz)
   
    print("\n\tJob done :)\n")


if __name__ == "__main__":
   main()