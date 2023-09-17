from QuantumSparse.spin.spin_operators import spin_operators
import numpy as np
from QuantumSparse.operator.operator import operator
from scipy import sparse

def main():
   
    S     = 0.5
    NSpin = 2
    SpinValues = np.full(NSpin,S)

    spins = spin_operators(SpinValues)

    a = sparse.csr_matrix(spins.Sx[0])
   
    print("\n\tJob done :)\n")


if __name__ == "__main__":
   main()