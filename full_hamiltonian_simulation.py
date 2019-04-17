# *** Full space simulator ***
# *** TODO: Visulaise the density matrix ***
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

# Get the number of spin particles in the network
Nspin = abs(int(sys.argv[1]))

# Get the initial state
with open('PSI(0).txt', 'r') as f:
    psi_0 = [[int(num) for num in line.split(' ')] for line in f]

# Constuct list of all possibe states:
list_of_states = []
for state in range(0,pow(2,Nspin)):
	list_of_states.append("{0:b}".format(state).zfill(Nspin))

# Get the connectivity matrix describing the network topology:
with open('network.txt', 'r') as f:
    Conn = [[int(num) for num in line.split(' ')] for line in f]

# Pauli Matrices
X = [[0,  1 ], [1,   0]]
Y = [[0, -1j], [1j,  0]]
Z = [[1,  0 ], [0,  -1]]

# Construct the full system Hamiltonian:
H = np.zeros((pow(2,Nspin),pow(2,Nspin)))
for k in range(0,Nspin):
	for l in range(k+1,Nspin):
		HM = np.zeros((pow(2,Nspin),pow(2,Nspin)))
		HM = Conn[k][l]*( np.kron( np.kron(np.kron(np.eye(pow(2,k)),X),
                                    np.eye(pow(2,l-k-1))),
                            np.kron(X,np.eye(pow(2,Nspin-l-1))) )
                 + np.kron( np.kron(np.kron(np.eye(pow(2,k)),Y),
                                    np.eye(pow(2,l-k-1))),
                            np.kron(Y,np.eye(pow(2,Nspin-l-1))) ) )
		H=H+HM

# Reference: Time optimal information transfer in spintronics networks

# Calculate the unitary operator U(t):
T = 2.5
U = sp.linalg.expm(-1j*T*H) 

# Find the state at time t:
psi_t = U.dot(np.array(psi_0))

# Create the density matrix: |psi(t)> <psi(t)|
rho = psi_t.dot(psi_t.conj().transpose())

# Probability of each state:
print("Final Probability Distribution")
print("------------------------------")
total_prob = 0
Probability_Distribution = np.diag(abs(rho))
for state in range(0, pow(2,Nspin)):
	print("Probability of state: |" + str(list_of_states[state]) + ">" + " is: " + str(round(Probability_Distribution[state], 3)))
	total_prob += Probability_Distribution[state]
# Sanity check:	
print("[CHECK] Sum of probabilities: " + str(round(total_prob,3)))