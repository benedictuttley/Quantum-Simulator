# *** Multi excitation case ***

import sys
import os
import datetime
import numpy as np
from scipy import optimize
from scipy import sparse
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import subprocess

# Encapsulate this in a function:
def Visualize(rho, iterator, T, Nspin):
	print(T)
	size = pow(2, Nspin)
	fig11 = plt.figure(figsize=(size, size), constrained_layout=False)
	# gridspec inside gridspec
	outer_grid = fig11.add_gridspec(size, size, wspace=0.0, hspace=0.0)

	for i in range(size*size):
		print("part 1")
		inner_grid = outer_grid[i].subgridspec(1, 1, wspace=0.0, hspace=0.0)
		ax = plt.Subplot(fig11, inner_grid[0])
		circle1 = plt.Circle((0.5, 0.5), 0.25, color='white')
		style="Simple,tail_width=0.5,head_width=3,head_length=2"
		kw = dict(arrowstyle=style, color="k")
		style2="Simple,tail_width=0.5,head_width=0.5, head_length=8"
		kw2 = dict(arrowstyle=style2, color="white")


		# Phase:
		theta = np.angle(rho[int(np.floor(i/size))][i%size], deg=False)
		x = 0.5 + 0.25*np.cos(theta)
		y = 0.5 + 0.25*np.sin(theta)

		a1 = patches.FancyArrowPatch((0.5,0.5), (x,y), **kw)
		a2 = patches.FancyArrowPatch((0.48,0.5), (0.75,0.5), **kw2)
		circ=[]
		circ.append(patches.Circle ((0.5,0.5), 0.25, color='white'))
		coll=collections.PatchCollection(circ, zorder=-1)
		
		# Phase annotations for coherence (on non-diagonal entries)
		if(i%size != int(np.floor(i/size))):
			ax.add_collection(coll)
			ax.add_patch(a2)
			ax.add_patch(a1)
		
		ax.patch.set_facecolor('#228B22')
		ax.patch.set_alpha(abs(rho[int(np.floor(i/size))][i%size]))
		ax.set_xticks([])
		ax.set_yticks([])
		if(i < size):
			ax.set_xticks([0.5])
		if(i%size == 0):
			ax.set_yticks([0.5])
		x_ticks_labels = list_of_states
		y_ticks_labels = list_of_states
		fig11.add_subplot(ax)

		ax.set_xticklabels(["|" + x_ticks_labels[i%size] + ">"], rotation='horizontal', fontsize=10)
		ax.xaxis.tick_top()
		ax.set_yticklabels(["<" + y_ticks_labels[int(np.floor(i/size))] + "|"], rotation='horizontal', fontsize=8)
		if(i%size == int(np.floor(i/size))):
			ax.text(0.5, 0.5, 'matplotlib', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,text='P = ' + str(round(abs(rho[int(np.floor(i/size))][i%size]),2)) ,  size="medium",
		 bbox=dict(boxstyle="square,pad=0.1",
                              ec="none", facecolor='red', alpha=0.7))
		else:
			ax.patch.set_facecolor('k')
			ax.patch.set_alpha(abs(rho[int(np.floor(i/size))][i%size]))

	all_axes = fig11.get_axes()
	fig11.suptitle("SYSTEM STATE AT T = " + str(T), fontsize=10, horizontalalignment='center')

	# Save image for animation:
	fig11.savefig("file%02d.png" % iterator)
	plt.close(fig11)


# Get the number of spin particles in the network
Nspin = abs(int(sys.argv[1]))
# Get time step information
timeInterval_begin = abs(int(sys.argv[2]))
timeInterval_end = abs(int(sys.argv[3]))
increment = abs(int(sys.argv[4]))

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

# *** DYNAMIC ***

# Reference: Time optimal information transfer in spintronics networks
list_of_density_matrices = []
time_steps = []

for T in range(timeInterval_begin, timeInterval_end, increment):
	print(T)
	time_steps.append(T)
	# Calculate the unitary operator U(t):
	U = sp.linalg.expm(-1j*T*H) 

	# Find the state at time t:
	psi_t = U.dot(np.array(psi_0))

	# Create the density matrix: |psi(t)> <psi(t)|
	rho = psi_t.dot(psi_t.conj().transpose())

	# Add to the list:
	list_of_density_matrices.append(rho)
	
	# Probability of each state: - Diagonal elements of the density matrix equal probability of system being in the associated
	# basis states

	print("Final Probability Distribution")
	print("------------------------------")
	total_prob = 0
	Probability_Distribution = np.diag(abs(rho))
	
	for state in range(0, pow(2,Nspin)):
		print("Probability of state: |" + str(list_of_states[state]) + ">" + " is: " + str(round(Probability_Distribution[state], 3)))
		total_prob += Probability_Distribution[state]
	# Sanity check:	
	print("[CHECK] Sum of probabilities: " + str(round(total_prob,3)))


iterator = 0
print("Plotting Density Matrices...")
for density_matrix in list_of_density_matrices:
	Visualize(density_matrix, iterator, time_steps[iterator], Nspin)
	iterator+=1

subprocess.call([
	'ffmpeg', '-framerate', '1', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv444p',
	'Simulation.mp4'
	])