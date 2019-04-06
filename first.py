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

dimensions = int(sys.argv[1])
results_file = (sys.argv[2])

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

# Write results to file:
file = open(results_file, "a+")
file.write("*** New Entry ***\n")
file.write(" Matrix Size:" +  str(dimensions) + "\n")
file.write(" Numpy expm runtime: " + str(python_time) + "\n")
file.write(" C expm runtime: " + str(c_time) + "\n")
if(np.isnan(C_result[0][0].imag) and np.isnan(Python_result[0][0].imag)):
	file.write(" Both contain NaN entries \n\n")

# Compare two matrices are element-wise equal within a tolerance
else: 
	file.write(" Agreement(1e-12 tolerance): " + str(np.allclose(C_result, Python_result,
	rtol=1e-12, atol=1e-12, equal_nan=True)) + "\n\n")
	file.close()