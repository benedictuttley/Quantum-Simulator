//25/03/2019 ***
// Attempt at a batched implementation of expm for smaller matrices that can be computed in parallel (as much as is possible)
// This can be used to find the propegator for each time step that are then suummeed to find the complete system propegator
// to describe the whole evolution of the system over the specified time frame.

// Batched operations -> perfrom the operation such as (matrix-matrix multiplication) on a bacth of matrices, note that these
// On certain problem sizes, it might be advantageous to make multiple calls to cublas<t>gemm in different CUDA streams,
// rather than use this API. -> such as on the larger matrix expm 
// matices must all have the same dimensions.

// Available Batch functions:
// gemmBatched() -> multiplication
// getrsBatched() -> LU factorization for inverse
// getrfBatched() -> System solver for inverse
// matinvBatched() -> Inverse shortcut only for matrices with n < 32
// gemmBatched() -> Possible use for matrix scaling
// may need own functions for matrix addition and subtraction
// Other option is to use streams and for loop to compute the additions/subtractions
// Not all library functions have a batch equivelant and as such CUDA streams may then be consdidered

// Note the parallelism for batch is not observed on profiler

#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdio.h>  
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "expm.h"
#include <stdbool.h>

#define BLOCK_SIZE 32

// *** CURRENT AIM: REMOVE DEPENDENCIES SO HOST ARRRAYS CAN BE REMOVED ***

__global__ void identity_kernel(cuDoubleComplex* identity, int dim){
    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y; 

    if(tid_x < dim && tid_y < dim){ // Check the problem bounds
        
        identity[(dim*tid_x) + tid_y].y = 0;                
        
        if(tid_x == tid_y) // Set the identity matrix:
            identity[(dim*tid_x) + tid_y].x = 1;
        else
            identity[(dim*tid_x) + tid_y].x = 0;
    } 
}


__global__ void absolute_kernel(cuDoubleComplex* A, int dim){

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y;


    A[(dim*tid_y) + tid_x].x = cuCabs((A[(dim*tid_y) + tid_x]));
            A[(dim*tid_y) + tid_x].y = 0;


}

// This version only works for small matrices that can fit into shared memory:
__global__ void norm_1_kernel_small(cuDoubleComplex* A, int dim, double* res){
    extern __shared__ double s[];   // Shared memory array to store column sums, size set in <<<>>>

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;

    double sum = 0; // Private variable to hold column sum

    for (int i = 0; i < dim; ++i)   // Calculate column sums, one column per thread
        {
            sum += cuCabs(A[(i*dim) + tid_x]);
        }
        s[tid_x] = sum;

        __syncthreads(); // sum contains the column sums

    if (tid_x == 0) // Calculate the max column sum using thread 0
    {
        for (int i = 0; i < 10; i++)
        {
            if(res[0] < s[i])
                res[0] = s[i];
        }
    }
}

__global__ void norm_1_kernel_large(cuDoubleComplex* A, int dim, double* res){
    extern __device__ double s[];   // Shared memory array to store column sums, size set in <<<>>>

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;

    double sum = 0; // Private variable to hold column sum

    for (int i = 0; i < dim; ++i)   // Calculate column sums, one column per thread
        {
            sum += cuCabs(A[(i*dim) + tid_x]);
        }
        s[tid_x] = sum;

        __syncthreads(); // sum contains the column sums

    if (tid_x == 0) // Calculate the max column sum using thread 0
    {
        for (int i = 0; i < 10; i++)
        {
            if(res[0] < s[i])
                res[0] = s[i];
        }
    }
}

void matrix_complex_print(cuDoubleComplex* A, int network_size){
  for (int j = 0; j < network_size; j++){
    printf("[");
    for (int k = 0; k < network_size; k++){
      printf(" %lf ", A[(j*network_size) + k].x );
      printf("+");
      printf(" %lfi ", A[(j*network_size) + k].y );
    }
    printf("]");
    printf("\n");
  }
}


void write_input_matrix(cuDoubleComplex *A, int n) {
    FILE *f;
    f = fopen("/home/c1673666/expm_Cuda/cuda/Quantum-Simulator/CUDA_INPUT.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {

            if (i == n - 1) {
                if (A[(n * j) + i].x == INFINITY) {
                    fprintf(f, "Inf");
                } else {
                    fprintf(f, "%lf", A[(j*n) + i].x );
                    fprintf(f, "+");
                    fprintf(f, "%lfi ", A[(j*n) + i].y );
                }
            } else {
                if (A[(n * j) + i].x == INFINITY) {
                    fprintf(f, "Inf ");
                } else {
                        fprintf(f, "%lf", A[(j*n) + i].x );
                        fprintf(f, "+");
                        fprintf(f, "%lfi ", A[(j*n) + i].y );;
                    }
                }
            }
            fprintf(f, "\n");
        }
    }

void set_Identity(cuDoubleComplex* A, int dim){
    int dimensions = (int) ceil((float)(BLOCK_SIZE/dim));
    dim3 dimGrid(dimensions, dimensions, 1); // Set a grid of 2*2 blocks
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1); // Set each block to be 2*2 threads
    identity_kernel<<<dimGrid, dimBlock>>>(A, dim);
    cudaDeviceSynchronize();

}

// Scale a matrix
void scale_tester(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n){

    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, NULL, n, d_C, n);
}

// Scale first matrix and then add the second matrix to the result:
void scale_and_add(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n){
    
    const cuDoubleComplex bet = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);
}
// Scale first matrix and then subtract the second matrix from result:
void scale_and_subtract(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n){
    
    const cuDoubleComplex bet = make_cuDoubleComplex(-1, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);
}

// Scale both the first and second matrix by there respective complex factors and add the results to eachother:
void scale_and_add_complete(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, const cuDoubleComplex bet, int n){
    
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);
}

// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// <<<>>> in here place [1] The number of thread blocks in the grid, [2] The number of threads per thread block
// Works for small matrices where the simension does not exceed the block size (due to use of shared memory);
double matrix_1_norm(cuDoubleComplex* d_A, cudaStream_t my_stream, int dim){
    int dimensions = ceil((float) dim/BLOCK_SIZE);
    dim3 dimGrid(dimensions, 1, 1); // Set a grid of 2*2 blocks
    dim3 dimBlock(BLOCK_SIZE, 1, 1); // Set each block to be 2*2 threads
    printf("THE NUMBER OF BLOCKS IS: (%d, %d)\n", dimensions, dimensions);
    printf("THE NUMBER OF THREADS PER BLOCK IS: (%d, %d)\n",2, 2);

    double* res;
    double* d_res;
    cudaMalloc(&d_res,sizeof(double));
    res = (double*)malloc(sizeof(double));
    // Selct the norm kernel to use based on matrix size:
    if(dim <= BLOCK_SIZE)
        norm_1_kernel_small<<<dimGrid, dimBlock, dim*sizeof(double), my_stream>>>(d_A, dim, d_res); // Uses shared memory
    else
        norm_1_kernel_large<<<dimGrid, dimBlock, dim*sizeof(double), my_stream>>>(d_A, dim, d_res); // Uses global memory
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("ONE NORM IS: %lf\n", res[0]);
    return res[0];
 }


void Inverse_Batched(cublasHandle_t handle, cuDoubleComplex** d_A, cuDoubleComplex** inverse, int dim, int batch_count){

    cublasHandle_t my_handle;
    int* dLUPivots_ALT;
    int* dLUInfo_ALT;

    // Create a cublas status object
    cublasStatus_t status;
    status = cublasCreate(&my_handle);
  
    cudaMalloc(&dLUPivots_ALT, dim * sizeof(int)), "Failed to allocate dLUPivots!";
    cudaMalloc(&dLUInfo_ALT, sizeof(int)), "Failed to allocate dLUInfo!";

    // Perform the LU factorization for each matrix in the batch:
    status = cublasZgetrfBatched(handle, dim, d_A, dim, dLUPivots_ALT, dLUInfo_ALT, batch_count);
    cudaDeviceSynchronize();
    if(status != CUBLAS_STATUS_SUCCESS)
        printf("BATCH LU DECOMPOSITION WAS NOT SUCCESSFUL!\n");
    else
        printf("BATCH LU DECOMPOSITION WAS SUCCESSFUL!\n");

    // Solve linear system to get inverse [(LU)^-1]
    // Note there is no need to create the identity when using getri
    status = cublasZgetriBatched(handle, dim,  (const cuDoubleComplex**)d_A, dim, (const int*) dLUPivots_ALT, inverse, dim, dLUInfo_ALT, batch_count);
    cudaDeviceSynchronize();
    if(status != CUBLAS_STATUS_SUCCESS){
        printf("BATCH LU DECOMPOSITION WAS NOT SUCCESSFUL!\n");
        printf("%d\n", status);
    }
    else
        printf("BATCH LU DECOMPOSITION WAS SUCCESSFUL!\n");
}

void Inverse_Batched_Small(){}



// Attempt with Dzasum:
double calculate_one_norm_New_complex(const cuDoubleComplex *A, int n) {
    double max = -DBL_MAX;
  double count;
    for (int i = 0; i < n; i++) {
        count = 0;
        for (int j = 0; j < n; j++) {
            count += cuCabs((A[(n * j) + i]));
        }
        if (count > max) {;
            max = count;
        };
    }
    return max;
}


void get_pade_coefficients(double *buf, int m) {

    double coefficients[5][14] = {
            {120, 60, 12, 1},
            {30240, 15120, 3360, 420, 30, 1},
            {17297280, 8648640, 1995840, 277200, 25200, 1512, 56 ,1},
            {17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1},
            {64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600, 670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1}
        };

    switch (m) {

        case 3  : {
            buf = coefficients[0];

        }
        case 5  : {
            buf = coefficients[1];
        }
        case 7  : {
            buf = coefficients[2];
        }

        case 9  : {
            buf = coefficients[3];
        }
        case 13  : {
            for (int i = 0; i < sizeof(coefficients[4]) / sizeof(double); i++) {
                buf[i] = coefficients[4][i];
            }
        }
        default:
            break;
    }
}

void matrix_Absolute_New(cuDoubleComplex *a, cuDoubleComplex *b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[(n * i) + j].x = cuCabs((a[(n * i) + j]));
            b[(n * i) + j].y = 0;
        }
    }
}

// Calulate the absolute values of the entries of a complex matrix: 
void absolute(cuDoubleComplex* d_A, int dim){
    int dimensions = ceil((float) dim/BLOCK_SIZE);
    dim3 dimGrid(dimensions, dimensions, 1); // Set a grid of 2*2 blocks
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1); // Set each block to be 2*2 threads
    printf("THE NUMBER OF BLOCKS IS: (%d, %d)\n", dimensions, dimensions);
    printf("THE NUMBER OF THREADS PER BLOCK IS: (%d, %d)\n",BLOCK_SIZE, BLOCK_SIZE);
    absolute_kernel<<<dimGrid, dimBlock>>>(d_A, dim);
    cudaDeviceSynchronize();
}


void matrix_Scale_New(cuDoubleComplex *a, cuDoubleComplex *scaled, cuDoubleComplex scale, int dim) {

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            scaled[(dim * i) + j] = cuCmul(a[(dim * i) + j],scale); // Complex multiplication
        }
    }
}


/////////////////////// *** current work *** ////////////////////////////
double ell(cublasHandle_t handle, cuDoubleComplex* d_A, double coeff, int m_val, int dim) {

    double norm_one, norm_two, p, alpha, output;
    cuDoubleComplex* mine;
    
    cudaMalloc(&mine, dim*dim*sizeof(cuDoubleComplex));
    cudaMemcpy(mine, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    absolute(mine, dim);
    printf("m is %d\n", m_val);
    p = pow(coeff, (1.0 / (2 * m_val + 1)));

    scale_tester(handle, mine, mine, make_cuDoubleComplex(p, 0), dim);
    norm_one = matrix_1_norm(mine, 0, dim);
    printf("NORM ONE IS: %lf \n", norm_one);
    norm_two = matrix_1_norm(d_A, 0, dim);
    printf("NORM TWO IS: %lf \n", norm_two);
    
    alpha = norm_one / norm_two;
    printf("ALPHA IS: %lf \n", alpha);
    output = fmax(ceil(log2((2 * alpha) / 2.220446049250313e-16) / (2 * m_val)), 0);
    return output;
  }

void matrixAdd_New(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n) { // PARALLEL CANDIDATE

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = cuCadd(a[(n * i) + j], b[(n * i) + j]); // Complex addition
        }
    }
}

void matrix_Subtract_New(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n) { // PARALLEL CANDIDATE

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = cuCsub(a[(n * i) + j], b[(n * i) + j]); // Complex subtraction
        }
    }
}


void set_Identity_New(cuDoubleComplex *i_matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                i_matrix[(n * i) + j].x = 1;
            } else {
                i_matrix[(n * i) + j].x = 0;
            }
        }
    }
}

int main(int argc, char* argv[])
{


/////////////////////////////////////////////////////////////////////////////////////////////////// SETUP START /////////////////////////////////////////////////////////////////////////////////////////////
	int dim = 2;
	int batch_count = 9;
 
    // Allocate host array A to construct input matrix:
    cuDoubleComplex **A = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
        for(int i=0; i<batch_count; i++) {
            A[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        }
 
    // INITIALIZE BATCHES WITH DUMMY DATA:
    for (int i = 0; i< batch_count; i++) {
        for(int j = 0; j< dim; j++){
            for (int k = 0; k < dim; k++)
            {
                A[i][(dim*j) + k] = make_cuDoubleComplex(i, i);
            }
        }
    }
    

    // WRITE THE 5th INPUT MATRIX FOR COMPARISON WITH MATLAB:
    write_input_matrix(A[4], dim);


    // Create cublas instance
    cublasHandle_t handle;
    cublasCreate(&handle);
 
     // *** CURRENT: CLEAN AND REDUCE THESE ALLOCATIONS:
    // Create host pointer array to device matrix storage
    cuDoubleComplex **d_T1, **d_T2, **d_T4, **d_T6, **d_T8, **d_T10, **h_d_T1, **h_d_T2, **h_d_T4, **h_d_T6, **h_d_T8, **h_d_T10;
    cuDoubleComplex **d_A, **d_B, **d_C, **h_d_A, **h_d_B, **h_d_C;
    h_d_T1 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T2 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T4 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T6 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T8 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T10 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_A = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_B = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_C = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
 
    for(int i=0; i<batch_count; i++) {
        cudaMalloc((void**)&h_d_T1[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T2[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T4[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T6[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T8[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T10[i], dim*dim*sizeof(cuDoubleComplex));
    }
    
    // Copy the host array of device pointers to the device
    cudaMalloc((void**)&d_T1, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T2, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T4, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T6, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T8, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T10, batch_count*sizeof(cuDoubleComplex*));
    cudaMemcpy(d_T1, h_d_T1, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T2, h_d_T2, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T4, h_d_T4, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T6, h_d_T6, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T8, h_d_T8, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T10, h_d_T10, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    for(int i=0; i<batch_count; i++) {
        cudaMalloc((void**)&h_d_A[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_B[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_C[i], dim*dim*sizeof(cuDoubleComplex));
    }

    // Copy the host array of device pointers to the device
    cudaMalloc((void**)&d_A, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_B, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_C, batch_count*sizeof(cuDoubleComplex*));
    cudaMemcpy(d_A, h_d_A, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_d_B, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_d_C, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);


    // Copy host batch to device memory:
    for(int i=0; i<batch_count; i++) {
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), A[i], dim, h_d_A[i], dim);   // Copy input array to device A
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), A[i], dim, h_d_T1[i], dim); // Copy input array to device T1
    }

    // Alpha and beta coeficients set for zgemm:
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;


/////////////////////////////////////////////////////////////////////////////////////////////////// SETUP END /////////////////////////////////////////////////////////////////////////////////////////////

    // [PART 1] TPOWERS CALULATED USING BATCH DGEMM
    // TODO: Launch each DGEMM operation in own CUDA stream
 
    // Calulate T2:

      cudaDeviceSynchronize();

  	  cublasZgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim,
                       alpha,
                       (const cuDoubleComplex**)d_A, dim,
                       (const cuDoubleComplex**)d_A, dim,
                       beta,
                       d_T2, dim,
                       batch_count);

  	  cudaDeviceSynchronize();
  	
  	// Calculate T4:
  	
  	 cublasZgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim,
                       alpha,
                       (const cuDoubleComplex**)d_T2, dim,
                       (const cuDoubleComplex**)d_T2, dim,
                       beta,
                       d_T4, dim,
                       batch_count);

  	// Calculate T6:
  	
  	cudaDeviceSynchronize();

  	cublasZgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim,
                       alpha,
                       (const cuDoubleComplex**)d_T4, dim,
                       (const cuDoubleComplex**)d_T2, dim,
                       beta,
                       d_T6, dim,
                       batch_count);

    cudaDeviceSynchronize();

    // Calculate T8:

    cublasZgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim,
                       alpha,
                       (const cuDoubleComplex**)d_T4, dim,
                       (const cuDoubleComplex**)d_T4, dim,
                       beta,
                       d_T8, dim,
                       batch_count);
 
    // No synchronization needed as T10 calc independent of T8

    cublasZgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim,
                       alpha,
                       (const cuDoubleComplex**)d_T8, dim,
                       (const cuDoubleComplex**)d_T2, dim,
                       beta,
                       d_T10, dim,
                       batch_count);
 
    cudaDeviceSynchronize();
    

  // [PART 2] CALCULATE (d4,d6,d8, d10)
  double* d4 = (double*) malloc(batch_count*sizeof(double));
  double* d6 = (double*) malloc(batch_count*sizeof(double));
  double* d8 = (double*) malloc(batch_count*sizeof(double));
  double* d10 = (double*) malloc(batch_count*sizeof(double));


//////////////// STREAMS FOR BATCH ////////////////////////  
  cudaStream_t streams[batch_count];
  for (int i = 0; i < batch_count; i++) {
    cudaStreamCreate(&streams[i]);
}

//////////////// STREAMS FOR BATCH ////////////////////////

  for (int i = 0; i < batch_count; i++) // Calculated on the host currently
  {
    d4[i] = pow(matrix_1_norm(h_d_T4[i], streams[i], dim), (1.0 / 4));
    d6[i] = pow(matrix_1_norm(h_d_T6[i], streams[i], dim), (1.0 / 6));
    d8[i] = pow(matrix_1_norm(h_d_T8[i], streams[i], dim), (1.0 / 8));
    d10[i] = pow(matrix_1_norm(h_d_T10[i], streams[i], dim), (1.0 / 10));
    
  }


  // PRINT A SAMPLE
  printf("\n");
  printf("%lf", d4[1]);
  printf("\n");
  printf("%lf", d6[1]);
  printf("\n");
  printf("%lf", d8[1]);
  printf("\n");
  printf("%lf", d10[1]);


  // [PART 3] CALCULATE (eta1, eta3, eta4, eta5)
  double* eta1 = (double*) malloc(batch_count*sizeof(double));
  double* eta3 = (double*) malloc(batch_count*sizeof(double));
  double* eta4 = (double*) malloc(batch_count*sizeof(double));
  double* eta5 = (double*) malloc(batch_count*sizeof(double));
  int* m_val = (int*) malloc(batch_count*sizeof(int));

  for (int i = 0; i < batch_count; i++)
  {
    eta1[i] = fmax(d4[i], d6[i]);
    eta3[i] = fmax(d6[i], d8[i]);
    eta4[i] = fmax(d8[i], d10[i]);
    eta5[i] = fmax(eta3[i], eta4[i]); 
  }

  // PRINT A SAMPLE
  printf("\n");
  printf("%lf", eta1[1]);
  printf("\n");
  printf("%lf", eta3[1]);
  printf("\n");


  // [PART 4] CALULATE (m_val: 3, 5, 7, 9)

  double theta[5] = {
                      1.495585217958292e-002, 2.539398330063230e-001,
                       9.504178996162932e-001, 2.097847961257068e+000,
                       5.371920351148152e+000
                     };

  double error_coefficients[5] = {
                                    1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
                                    1 / 113250775606021113483283660800000000.0,
                                    1 / 113250775606021113483283660800000000.0
                                  };


  for (int i = 0; i < batch_count; i++)
  {
    if(eta1[i] <= theta[1] && ell(handle, h_d_A[i], error_coefficients[1], 3, dim) == 0); // Check for m_val = 3
      m_val[i] = 3;
    if(eta1[i] <= theta[2] && ell(handle, h_d_A[i], error_coefficients[2], 5, dim) == 0); // Check for m_val = 5
      m_val[i] = 5;
    if(eta3[i] <= theta[3] && ell(handle, h_d_A[i], error_coefficients[3], 7, dim) == 0); // Check for m_val = 7
      m_val[i] = 7;
    if(eta3[i] <= theta[4] && ell(handle, h_d_A[i], error_coefficients[4], 9, dim) == 0); // Check for m_val = 9
      m_val[i] = 9;
  }

  
  // PRINT A SAMPLE
  printf("\n");
  printf("%d", m_val[1]);
  printf("\n");
  printf("%d", m_val[2]);
  printf("\n");
  printf("%d", m_val[3]);
  printf("\n");
  printf("%d\n", m_val[4]);



// [PART 5] CALULATE s
  double* s = (double*) malloc(batch_count*sizeof(double)); 
  double max = 0;

  for (int i = 0; i < batch_count; i++)
  {
    s[i] = fmax(ceil(log2(eta5[i]/theta[4])), 0);
    printf("--->%lf\n", s[4]);
    scale_tester(handle, h_d_A[i], h_d_A[i], make_cuDoubleComplex(1/pow(2, s[i]), 0), dim);

    s[i] = s[i] + ell(handle, h_d_A[i], error_coefficients[4], 13, dim);
    if(s[i] > max)
        max = s[i];
}
printf("%lf\n", s[4] );

// [PART 6] S CHECK AND M CHECK - [TODO]

for (int i = 0; i < batch_count; i++)
{
  if (isinf(s[i]))
  {
    printf("S/M CHECK HAS BEEN HIT\n");
    exit(0);
  } else{
    m_val[i] = 13;
  }
}


// [PART 7] RESCALE THE POWERS ARRAYS IF S NOT 0
for (int i = 0; i < batch_count; i++)
{
  if (s[i]!=0)
  {
    scale_tester(handle, h_d_T1[i], h_d_T1[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 1)), 0), dim);
    scale_tester(handle, h_d_T2[i], h_d_T2[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 2)), 0), dim);
    scale_tester(handle, h_d_T4[i], h_d_T4[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 4)), 0), dim);
    scale_tester(handle, h_d_T6[i], h_d_T6[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 6)), 0), dim);
    }
}


// [PART 7.5] GET THE PADE COEFFICIENTS FOR EACH BATCH

double** c = (double**) malloc(batch_count*sizeof(double*)); 

for (int i = 0; i < batch_count; i++)
{ 
  c[i] = (double*) malloc(15*sizeof(double));
  get_pade_coefficients(c[i], m_val[i]);
}


for (int i = 0; i < batch_count; i++)
{
    if(m_val[i] != 13){
        printf("DIFFERENCE IS SEEN!\n");
        exit(0);
    }
}

//if (m_val == 13) // Will need to seperate matrices that are not satisfied for batching to commence

  // [PART 8] CALCULATE U

  for (int i = 0; i < batch_count; i++)
  {
    cudaMemset(h_d_C[i], 0, dim*dim*sizeof(cuDoubleComplex));
    scale_and_add(handle, h_d_T6[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][13], 0), dim);
    scale_and_add(handle, h_d_T4[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][11], 0), dim);
    scale_and_add(handle, h_d_T2[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][9], 0), dim);
}

// Perform batch matrix multiplication
  cublasZgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_B, dim,
                      (const cuDoubleComplex**)d_T6, dim,
                      beta,
                      d_C, dim,
                      batch_count);

  cudaDeviceSynchronize();

  for (int i = 0; i < batch_count; i++)
  {

    scale_and_add(handle, h_d_T6[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][7], 0), dim);
    scale_and_add(handle, h_d_T4[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][5], 0), dim);
    scale_and_add(handle, h_d_T2[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][3], 0), dim);
    set_Identity(h_d_B[i], dim);
    scale_and_add(handle, h_d_B[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][1], 0), dim);
    scale_and_add(handle, h_d_C[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim);

  }

  // BATCH MATRIX MULTIPLY:
  cublasZgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_B, dim,
                      (const cuDoubleComplex**)d_T1, dim,
                      beta,
                      d_C, dim,
                      batch_count);

  cudaDeviceSynchronize();
    



  // [PART 9] CALCULATE V
  for (int i = 0; i < batch_count; i++)
  {
    scale_and_add_complete(handle, h_d_T6[i], h_d_T4[i], h_d_B[i], make_cuDoubleComplex(c[i][12], 0), make_cuDoubleComplex(c[i][10], 0), dim);
    scale_and_add(handle, h_d_T2[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][8], 0), dim);
}

   // BATCH MATRIX MULTIPLY:

    cublasZgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_B, dim,
                      (const cuDoubleComplex**)d_T6, dim,
                      beta,
                      d_A, dim,
                      batch_count);

  cudaDeviceSynchronize();
      // Copy each device result to the host
 // Copy each device result to the host


for (int i = 0; i < batch_count; i++)
  {
    cudaMemset(h_d_B[i], 0, dim*dim*sizeof(cuDoubleComplex));
    scale_and_add(handle, h_d_T6[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][6], 0), dim);
    scale_and_add(handle, h_d_T4[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][4], 0), dim);
    scale_and_add(handle, h_d_T2[i], h_d_A[i], h_d_A[i], make_cuDoubleComplex(c[i][2], 0), dim);
    set_Identity(h_d_T2[i], dim);
    scale_and_add(handle, h_d_T2[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][0], 0), dim);
    scale_and_add(handle, h_d_A[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim);





    // CALCULATE (V-U):
    scale_and_subtract(handle, h_d_B[i], h_d_C[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim);
        if(i == 4){
    cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_B[i], dim, A[i], dim); // Output batch stored in A
    matrix_complex_print(A[4], dim);
}
}


// [PART 11] CALCULATE F:

// BATCH MATRIX INVERSE ON (V-U)
Inverse_Batched(handle, d_B, d_A, dim, batch_count);

// SCALE BATCH U BY 2
for (int i = 0; i < batch_count; i++){
    scale_tester(handle, h_d_C[i], h_d_C[i], make_cuDoubleComplex(2, 0), dim);
}


  // BATCH MATRIX MULTIPLICATION:
  cublasZgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_A, dim,
                      (const cuDoubleComplex**)d_C, dim,
                      beta,
                      d_B, dim,
                      batch_count);

  cudaDeviceSynchronize();
  

  for(int i=0; i<batch_count; i++) {
      set_Identity(h_d_C[i], dim);
      scale_and_add(handle, h_d_B[i], h_d_C[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim);

    }


  // SQUARING PHASE:
  for (int k = 0; k < max; k++) {
    printf("max is %lf", max);
  // PERFORM BATCH MATRIX MULTIPLICATION
  cublasZgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_B, dim,
                      (const cuDoubleComplex**)d_B, dim,
                      beta,
                      d_C, dim,
                      batch_count);

  cudaDeviceSynchronize();

for(int i=0; i<batch_count; i++) {
    
    if (k<s[i]-1){
       printf("s is: %lf \n", s[i]);
    if(i == 4){
    cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, A[i], dim); // Output batch stored in A
    matrix_complex_print(A[i], dim);
     printf("--->%lf\n", s[i] );
    }
     printf("%d\n", k );
        printf("%lf\n", s[i] );
    cudaMemcpy(h_d_B[i], h_d_C[i], dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
}
  }

  // Copy each device result to the host
  for(int i=0; i<batch_count; i++) {
    cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, A[i], dim); // Output batch stored in A
  }

  printf("EXPM RESULT FOR 5TH IN BATCH IS: \n");
  matrix_complex_print(A[4], dim);// Clean up resources

 
    for(int i=0; i<batch_count; i++) {
        free(A[i]);
        cudaFree(h_d_A[i]);
        cudaFree(h_d_B[i]);
        cudaFree(h_d_C[i]);
    }
 
    free(A);
    free(h_d_A);
    free(h_d_B);
    free(h_d_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
 
    return 0;
}