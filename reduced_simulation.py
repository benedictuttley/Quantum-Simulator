import sys
import os
import datetime
import numpy as np
from scipy import optimize
from scipy import sparse
import scipy as sp
import GPyOpt as gp
import pygraphviz as pgv
import matplotlib.pyplot as plt

# Set number of spins
Nspin = 2

# Construct the connectivity matrix:
# Conn = [[0, 1, 0, 0, 0],
#		[1, 0, 1, 0, 0],
#		[0, 1, 0, 1, 0],
#		[0, 0, 1, 0, 1],
#		[0, 0, 0, 1, 0]]

Conn = [[0, 1],
		[1, 0]] 


# Pauli Matrices
X = [[0,  1 ], [1,   0]]
Y = [[0, -1j], [1j,  0]]
Z = [[1,  0 ], [0,  -1]]

# Construct the full system Hamiltonian:
H = []
for k in range(0,Nspin):
	for l in range(k+1,Nspin):
		HM = np.zeros((pow(2,Nspin),pow(2,Nspin)))
		HM = Conn[k][l]*( np.kron( np.kron(np.kron(np.eye(pow(2,k)),X),
                                    np.eye(pow(2,l-k-1))),
                            np.kron(X,np.eye(pow(2,Nspin-l-1))) )
                 + np.kron( np.kron(np.kron(np.eye(pow(2,k)),Y),
                                    np.eye(pow(2,l-k-1))),
                            np.kron(Y,np.eye(pow(2,Nspin-l-1))) ) )
		H.append(HM)

print(H)

# Find U(t)
T = 0.05
U = sp.linalg.expm(-1j*T*H[0])
print(U)

# Find psi
# What would psi0 be set to?
psi0 = [0, 1, 0, 0]
print(psi0)
psi = psi0*U
print(psi)
