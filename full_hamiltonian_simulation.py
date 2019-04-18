# *** Full space simulator ***
# *** TODO: Visulaise the density matrix ***
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


# fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)

# # gridspec inside gridspec
# outer_grid = fig11.add_gridspec(8, 8, wspace=0.0, hspace=0.0)

# for i in range(64):
#     inner_grid = outer_grid[i].subgridspec(1, 1, wspace=0.0, hspace=0.0)
#     #for j, (c, d) in enumerate(product(range(1, 2), repeat=2)):
#     ax = plt.Subplot(fig11, inner_grid[0])
#         #ax.plot(*squiggle_xy(a, b, c, d))
#     #i = np.arange(0.0, 2*np.pi, 0.1)
#         #ax.plot( np.sin(i*1)*np.cos(i*2), np.sin(i*1)*np.cos(i*1))
#     circle1 = plt.Circle((0.5, 0.5), 0.25, color='white')
#     style="Simple,tail_width=0.5,head_width=4,head_length=8"
#     kw = dict(arrowstyle=style, color="k")
#     #a1 = patches.FancyArrowPatch((0.1,0.2), (0,0.6),**kw )
#     a1 = patches.FancyArrowPatch((0.3,0.25), (0.75,0.3),connectionstyle="arc3,rad=.5", **kw)
#     #ax.Circle((0, 0), 0.2, color='r')
#     circ=[]
#     circ.append(patches.Circle ((0.5,0.5), 0.25, color='white'))
#     coll=collections.PatchCollection(circ, zorder=-1)
#     ax.add_collection(coll)

#     #ax.add_artist(circle1)

#     ax.add_patch(a1)
#     ax.patch.set_facecolor('#228B22')
#     ax.patch.set_alpha(abs(rho[int(np.floor(i/8))][i%8]))
#     ax.set_xticks([])
#     ax.set_yticks([])

#     if(i < 8):
#     	ax.set_xticks([0.5])
#     if(i%8 == 0):
#     	ax.set_yticks([0.5])
#     x_ticks_labels = list_of_states
#     y_ticks_labels = list_of_states
#     fig11.add_subplot(ax)
#     #ax.set_xticks(1)
#     # Set ticks labels for x-axis

#     ax.set_xticklabels(["|" + x_ticks_labels[i%8] + ">"], rotation='horizontal', fontsize=11)
#     ax.xaxis.tick_top()
#     ax.set_yticklabels(["|" + y_ticks_labels[int(np.floor(i/8))] + ">"], rotation='horizontal', fontsize=11)
#     if(i%8 == int(np.floor(i/8))):
#     	ax.text(0.0, 0.0, 'P = ' + str(round(abs(rho[int(np.floor(i/8))][i%8]),3)) , fontsize=8)
#     else:
#     	ax.patch.set_facecolor('red')
#     	ax.patch.set_alpha(1)
# all_axes = fig11.get_axes()

# fig11.suptitle('Density matrix and transfer state probabilities', fontsize=16)
# plt.show()


# # Attempt to cascasde images to form animation:
# fig11.savefig("test.png")

# # subprocess.call([
# # 	'ffmpeg', '-framerate', '8', '-i', 'test.png', '-r', '30', '-pix_fmt', 'yuv420p',
# # 	'video_name.mp4'
# # 	])



# Encapsulate this in a function:
def Visualize(rho, iterator, T):
	fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)
	# gridspec inside gridspec
	outer_grid = fig11.add_gridspec(8, 8, wspace=0.0, hspace=0.0)

	for i in range(64):
		inner_grid = outer_grid[i].subgridspec(1, 1, wspace=0.0, hspace=0.0)
		ax = plt.Subplot(fig11, inner_grid[0])
		circle1 = plt.Circle((0.5, 0.5), 0.25, color='white')
		style="Simple,tail_width=0.5,head_width=4,head_length=8"
		kw = dict(arrowstyle=style, color="k")
		a1 = patches.FancyArrowPatch((0.3,0.25), (0.75,0.3),connectionstyle="arc3,rad=.5", **kw)
		circ=[]
		circ.append(patches.Circle ((0.5,0.5), 0.25, color='white'))
		coll=collections.PatchCollection(circ, zorder=-1)
		
		# Phase annotations
		#ax.add_collection(coll)
		#ax.add_patch(a1)
		
		ax.patch.set_facecolor('#228B22')
		ax.patch.set_alpha(abs(rho[int(np.floor(i/8))][i%8]))
		ax.set_xticks([])
		ax.set_yticks([])
		if(i < 8):
			ax.set_xticks([0.5])
		if(i%8 == 0):
			ax.set_yticks([0.5])
		x_ticks_labels = list_of_states
		y_ticks_labels = list_of_states
		fig11.add_subplot(ax)

		ax.set_xticklabels(["|" + x_ticks_labels[i%8] + ">"], rotation='horizontal', fontsize=11)
		ax.xaxis.tick_top()
		ax.set_yticklabels(["|" + y_ticks_labels[int(np.floor(i/8))] + ">"], rotation='horizontal', fontsize=11)
		if(i%8 == int(np.floor(i/8))):
			ax.text(0.1, 0.0, 'P = ' + str(round(abs(rho[int(np.floor(i/8))][i%8]),3)) , fontsize=10)
		else:
			ax.patch.set_facecolor('#8a7b70')
			ax.patch.set_alpha(1)
	all_axes = fig11.get_axes()
	fig11.suptitle("SYSTEM STATE AT T = " + str(T) + " s", fontsize=16, horizontalalignment='center')
	#plt.show()

	# Attempt to cascasde images to form animation:
	fig11.savefig("file%02d.png" % iterator)


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
list_of_density_matrices = []
time_steps = [0.0, 0.5, 1.0, 1.5, 2.0]
for T in time_steps:
	# Calculate the unitary operator U(t):
	U = sp.linalg.expm(-1j*T*H) 

	# Find the state at time t:
	psi_t = U.dot(np.array(psi_0))

	# Create the density matrix: |psi(t)> <psi(t)|
	rho = psi_t.dot(psi_t.conj().transpose())

	# Add to the list:
	list_of_density_matrices.append(rho)
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


iterator = 0
print("Plotting Density Matrices...")
for density_matrix in list_of_density_matrices:
	Visualize(density_matrix, iterator, time_steps[iterator])
	iterator+=1

subprocess.call([
	'ffmpeg', '-framerate', '1', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv444p',
	'Simulation.mp4'
	])