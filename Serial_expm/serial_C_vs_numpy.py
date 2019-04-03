# Comparison of C expmwith CUDA expm:

# Dummy script to wrap Cuda function (expm) to be callable from python (Bayesian optimiser)
import numpy
import ctypes
import scipy as sp
import sys
import os
import datetime
import numpy as np
from scipy import optimize
import time
dimensions = 2048;

indata = numpy.ones((dimensions,dimensions), dtype=numpy.cdouble) # Create the input matrix
outdata = numpy.zeros((dimensions,dimensions), dtype=numpy.cdouble) # Create the input matrix

# Run numpy expm:
start_time = time.time()
Python_result = sp.linalg.expm(indata)
python_time = time.time() - start_time

# Link to C library containig expm program:
lib = ctypes.cdll.LoadLibrary("./ctestcuda.a")
expm_initialization = lib.expm_initialization

# Run C expm:
start_time_c = time.time()
expm_initialization(ctypes.c_void_p(indata.ctypes.data),
ctypes.c_void_p(outdata.ctypes.data), ctypes.c_int(dimensions))
c_time = time.time() - start_time_c
C_result = outdata

# Print results
print("Python expm: ")
print ('***: \n %s' % Python_result)
print("Sequential C expm: ")
print('***: \n %s' % C_result)

# Print runtimes
print("Python time: %s seconds" % python_time)
print("C time: %s seconds" % c_time)
