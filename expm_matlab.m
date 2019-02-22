 dlmwrite('/home/c1673666/expm_Cuda/cuda/Quantum-Simulator/read.txt', double(rand(10,10)), 'delimiter', ' ')

M = dlmread('read.txt'); 
tic; C = expm(M); toc;
dlmwrite('/home/c1673666/expm_Cuda/cuda/Quantum-Simulator/myFile2.txt', C, 'delimiter', ' ');