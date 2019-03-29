# Dummy script to wrap Cuda function (expm) to be callable from python (Bayesian optimiser)
import numpy
import ctypes
indata = numpy.ones((5,5), dtype=numpy.cdouble)
outdata = numpy.zeros((5,5), dtype=numpy.cdouble)
lib = ctypes.cdll.LoadLibrary("./ctestcuda.a")
expm_initialization = lib.expm_initialization
expm_initialization(ctypes.c_void_p(indata.ctypes.data),
ctypes.c_void_p(outdata.ctypes.data), ctypes.c_int(5))

print ('indata: %s' % indata)
print ('outdata: %s' % outdata)