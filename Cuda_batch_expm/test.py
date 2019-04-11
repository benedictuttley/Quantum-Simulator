# Comparison of CUDA batch expm with numpy expm:
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
batchSize = int(sys.argv[3])
inputBatch = np.ndarray(shape=(batchSize, dimensions, dimensions), dtype=numpy.cdouble, order='C')
outputBatchNumpy = np.ndarray(shape=(batchSize, dimensions, dimensions), dtype=numpy.cdouble, order='C')
outputBatchCuda = np.ndarray(shape=(batchSize, dimensions, dimensions), dtype=numpy.cdouble, order='C')

# Populate the batch:
for matrix in range(0, batchSize):
    inputBatch[matrix] = np.random.rand(dimensions,2*dimensions).view(np.complex128)

# Run numpy batch (loop) expm:
start_time = time.time()
for matrix in range(0, batchSize):
    outputBatchNumpy[matrix] = sp.linalg.expm(inputBatch[matrix])
python_time = time.time() - start_time


# Link to C library containig expm program:
lib = ctypes.cdll.LoadLibrary("./ctestcuda.a")
expm_initialization = lib.expm_initialization

# Run Batch CUDA expm:
start_time_c = time.time()
expm_initialization(ctypes.c_void_p(inputBatch.ctypes.data),
ctypes.c_void_p(outputBatchCuda.ctypes.data), ctypes.c_int(dimensions), ctypes.c_int(batchSize))
c_time = time.time() - start_time_c

# Print runtimes
#print("Python time: %s seconds" % python_time)
#print("CUDA time: %s seconds" % c_time)

# Write results to file:
file = open(results_file, "a+")
file.write("*** New Entry ***\n")
file.write(" (Batch Size:" +  str(batchSize) + ", Matrix Size:" + str(dimensions) + ") \n")
file.write(" Numpy expm runtime: " + str(python_time) + "seconds.\n")
file.write(" CUDA expm runtime: " + str(c_time) + " seconds.\n")

nan_present = False
precision_poor = False
for matrix in range(0, batchSize):
    if(np.isnan(outputBatchCuda[matrix][0][0].imag) and np.isnan(outputBatchNumpy[matrix][0][0].imag)):
	    nan_present = True
    
    # Compare two matrices are element-wise equal within a tolerance
    elif(np.allclose(outputBatchCuda[matrix], outputBatchNumpy[matrix], rtol=1e-5, atol=1e-5, equal_nan=True) == False):
        precision_poor = True

    
if nan_present:
    file.write(" Both contain NaN entries \n\n")
elif precision_poor == False:
        file.write(" Agreement(1e-5 tolerance): True \n\n")
else:
    file.write(" Agreement(1e-5 tolerance): False \n\n")

file.close()

