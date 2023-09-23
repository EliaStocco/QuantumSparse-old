from QuantumSparse.spin.spin_operators import spin_operators
import numpy as np
from QuantumSparse import operator
from QuantumSparse.spin.interactions import Ising
from scipy import sparse

def main():
   
    S     = 0.5
    NSpin = 2
    SpinValues = np.full(NSpin,S)

    spins = spin_operators(SpinValues)

    H = Ising(spins.Sz)
   
    print("\n\tJob done :)\n")


if __name__ == "__main__":
   main()