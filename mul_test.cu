// ----------------------------------------------------------------------------------------
// Generates 2 matrices (A and B) filled with reandom values
// C = A * B

// C/C++ implementation also present to ensure correctness of the result

//1) C/C++ assumes matrices are in row-major order but CUDA assumes column-major order
//2) Matrix B allocated as being 640 rows but only 320 rows are actually used in the calc

// When a matrix is passed to CUDA, the memory layout remains the same but CUDA assumes that the
// matrix is layed out in column-major order.

// NVIDIA ASSUMPTION: As the user, you want to calculate C = A*B, your matrices rae in row-major
// order however and you want the product matrix C to be also in row-major order.

// Passing the matrices in reverse order, you get B'*A' which is C' however when the matrices are
// taken into C, there is an implicit transpose such that you actually get C

// -- CODE PARAMETERS --

 // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA,
 // matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C,
 // matrix_size.uiWA);


////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <iostream>
#include <cuComplex.h>

// <--- BEGIN FIND DIMENSIONS OF INPUT MATRIX --->
int getMatrixDimensions() { 

    int i = 0;
    FILE* fp = fopen("/home/c1673666/expm_Cuda/read.txt", "r");
    int ch;

    while (!feof(fp)) {
      ch = fgetc(fp);
      if (ch == '\n') {
        i++;
      }
    }
    return i;
  }
// <--- END FIND DIMENSIONS OF INPUT MATRIX --->


// <--- BEGIN READ INPUT MATRIX --->
void loadMatrix(cuDoubleComplex* Z, int n) {

  int i = 0;
  FILE* fp = fopen("/home/c1673666/expm_Cuda/read.txt", "r");
  char* buf = (char*)malloc(3*sizeof(char));
  while( fscanf(fp, "%s", buf) != EOF ){
    Z[i].x = (buf[0]-'0');
    Z[i].y = (buf[2]-'0'); 
    i++;
  }
}
// <--- END READ INPUT MATRIX --->



void matrix_Print_New(float *A, int n) {
    printf("\n");
    for (int l = 0; l < n; l++) {
        for (int i = 0; i < n; i++) {
            printf("|%20.15e|   ", A[(n*l) + i]);
        }
        printf("\n");
    }
}


void matrix_Print_New_Comp(cuDoubleComplex* Z, int n){
  printf("THE COMPLEX MATRIX IS: \n");
    for (int l = 0; l < n; l++) {
        for (int i = 0; i < n; i++) {
            // printf("|%20.15e|   ", Z[(n*l) + i]); // format printing of complex entry...
              printf("%3.1f + %3.1fi ", Z[(n*l) + i].x, Z[(n*l) + i].y);
        }
        printf("\n");
      }
    }



typedef struct _matrixSize      // CAN THIS BE REMOVED?
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;


void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}


void matrixMultiply(cuDoubleComplex* h_A, cuDoubleComplex* h_B, cuDoubleComplex* h_C, cublasHandle_t handle, int n)
{
   
  // <--- BEGIN DEVICE MEMORY ALLOCATION --->
  clock_t begin = clock();
  int mem_size = (n * n * sizeof(cuDoubleComplex));
  cuDoubleComplex *d_A, *d_B, *d_C;
  
  cudaMalloc((void **) &d_A, mem_size);
  cudaMalloc((void **) &d_B, mem_size);
  cudaMalloc((void **) &d_C, mem_size);

  cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);


  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("TIME FOR DEVICE MEMORY ALLOCATION: %f\n", time_spent);
   // <--- END DEVICE MEMORY ALLOCATION --->

    // execute the kernel
 
    {
      cuDoubleComplex alpha_COMP, beta_COMP;
      alpha_COMP.x = 1; alpha_COMP.y = 0;
      beta_COMP.x = 0; beta_COMP.y = 0;


        // <-- BEGIN MATRIX MULTIPLICATION -->
        clock_t begin = clock();
     
        cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha_COMP, d_B, n, d_A, n, &beta_COMP, d_C, n);
        cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);
        
        clock_t end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("TIME TO PERFORM MATRIX MULTIPLICATION: %f\n", time_spent);
        // <-- BEGIN MATRIX MULTIPLICATION -->
    
        // <--- BEGIN FREE UP COMPLEX MEMORY -->
        begin = clock();
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        end = clock();

        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("TIME FOR DEVICE MEMORY DEALLOCATION %f\n", time_spent);
        // <--- END FREE UP COMPLEX MEMORY -->
      }
    }



int main(int argc, char **argv)
{

    // CONSTANT OVERHEAD
    cudaFree(0); // Trigger the creation of the CUDA context:
    cublasHandle_t handle;
    cublasCreate(&handle); // Creation of the cuBLAS library context:

    // HOST MATRICES
    int n = getMatrixDimensions();
    int mem_size = (n * n * sizeof(cuDoubleComplex));

    // allocate host memory for matrices A and B - COMPLEX [A]
    cuDoubleComplex* h_A = (cuDoubleComplex*) malloc(mem_size);
    cuDoubleComplex* h_B = (cuDoubleComplex*) malloc(mem_size);
    cuDoubleComplex* h_C = (cuDoubleComplex*) malloc(mem_size);

    loadMatrix(h_A, n);
    loadMatrix(h_B, n);
    
    clock_t begin = clock();
    
    matrixMultiply(h_A, h_B, h_C, handle, n);
    
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("TOTAL TIME: %f\n", time_spent);
    
    matrix_Print_New_Comp(h_C, n);

    free(h_A);
    free(h_B);
    free(h_C);

    cublasDestroy(handle);
    cudaDeviceReset(); 

    return 0;
}