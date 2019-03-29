#include <stdio.h>
#include <cuComplex.h>

// Dummy C function that takes the input matrix:
extern "C" void expm_initialization(const void * input, void * output, int dim) {
	const cuDoubleComplex * A = (cuDoubleComplex *) input; // Cast void pointers to complex doubles
    cuDoubleComplex * mat_expm = (cuDoubleComplex *) output; // Cast void pointers to complex doubles
    
    for (int i = 0; i < dim * dim; ++i) {
        mat_expm[i] = make_cuDoubleComplex(i, 100); // Peform the expm algorithm
    }
}