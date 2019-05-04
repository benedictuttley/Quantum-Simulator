# Single Excitation Simulator
# ---------------------------

# Models the information transfer in the network with one bit of information being transferred.

# File Input:
# -----------

# [1] network_SES.txt encoding the topology of the network
# [2] PSI(0)_SES.txt encoding the initial system state

# Arguments:
# -----------

# [1] Network size (number of spins in the network)
# [2] Initial time to simulate
# [3] Final time to simulate
# [4] Time step increment

# Reference: Langbein, F.C., Schirmer, S. and Jonckheere, E., 2015, December. Time optimal information transfer in spintronics networks.


import sys
import os
import datetime
import numpy as np
from scipy import optimize
from scipy import sparse
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Get the number of spin particles in the network
Nspin = abs(int(sys.argv[1]))
# Get time step information
timeInterval_begin = abs(int(sys.argv[2]))
timeInterval_end = abs(int(sys.argv[3]))
increment = abs(int(sys.argv[4]))

list_of_spins = []
for spin in range(1, Nspin+1):
	list_of_spins.append("Spin: " + str(spin))

# Construct the hamiltonian
H = np.zeros((Nspin, Nspin))*1j
Conn = np.zeros((Nspin, Nspin))

# Get the connectivity matrix describing the network topology:
with open('network_SES.txt', 'r') as f:
	i = 0
	for line in f:
		j = 0
		for num in line.split(' '):
			Conn[i][j] = int(num)
			j = j+1
		i = i+1

H = Conn + 0j 

# Get the initial state
with open('PSI(0)_SES.txt', 'r') as f:
    psi_0 = [[int(num) for num in line.split(' ')] for line in f]


# Simulate system for each time period
time_steps = []
fidelities = []
info = [[] for x in range(Nspin)] 
for T in range(timeInterval_begin, timeInterval_end, increment):
	time_steps.append(T)
	# Calculate the unitary operator U(t):
	U = sp.linalg.expm(-1j*T*H) 
	# Find the new state at time t:
	psi_t = U.dot(np.array(psi_0))
	total = 0
	prob_dist = []
	for i in psi_t:
		prob = abs(pow(i, 2))
		total += prob[0]
		prob_dist.append(prob[0])
	print(" ***** Time = " + str(T) + " *****")
	i = 0
	for spin in list_of_spins:
		print(spin)
		print("probability is: " + str(round(prob_dist[i], 3)))
		info[i].append(prob_dist[i])
		i+=1
	print("Total probability = " + str(round(total, 3)))
	print("")



# Visualize transfer fidelity change over time steps
print("Vizualising...")
x = time_steps
fig = plt.figure(figsize=(20, 15), dpi=80)
fig.suptitle("Probability of Information Transfer From Spin 1 To Connected Spins")
fig.canvas.set_window_title('Simulation Vizualisation')

gs = gridspec.GridSpec(int(np.ceil((Nspin)/2)), int(np.ceil((Nspin)/2)))
axs = []

i = 0
for spin in list_of_spins:
	y = info[i]
	axs.append(fig.add_subplot(gs[i], title="*** Fidelity For Spin: 1 -> " + spin + " ***",
		ylim=(-0.05, 1.05), yticks=[0.0, 0.25, 0.50, 0.75, 1.0], xlabel="Time(s)", 
    ylabel="Transfer Fidelity"))
	axs[-1].spines['right'].set_visible(False)
	axs[-1].spines['top'].set_visible(False)
	axs[-1].set_facecolor("#d3d3d3")
	axs[-1].set_autoscaley_on(False)
	axs[-1].plot(x,y,'gx-', linewidth=2)
	i+=1

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.show()