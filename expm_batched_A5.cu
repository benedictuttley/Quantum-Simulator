// *** 29/03/2019 *** --> BATCHED WITH STREAMS

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


__global__ void ell_kernel(cuDoubleComplex* A, cuDoubleComplex* B, int dim, int* m_val, int k, int val){
 extern __device__ double s_nrm_one[];   // Device memory array to store column sums 
 extern __device__ double s_nrm_two[];   // Device memory array to store column sums 

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;

     double sum_nrm_one = 0; // Private variable to hold column sum
     double sum_nrm_two = 0; // Private variable to hold column sum

    for (int i = 0; i < dim; i++)   // Calculate column sums, one column per thread
        {
            sum_nrm_one += cuCabs(A[(i*dim) + tid_x]);
            sum_nrm_two += cuCabs(B[(i*dim) + tid_x]);
        }
        s_nrm_one[tid_x] = sum_nrm_one;
        s_nrm_two[tid_x] = sum_nrm_two;

        __syncthreads(); // sum contains the column sums

    if(tid_x == 1){
        double second_norm = 0;
        double first_norm = 0;
        double alpha = 0;
        double output = 0;
        for (int i = 0; i < dim; i++)
        {
            if(first_norm < s_nrm_one[i])
                first_norm = s_nrm_one[i];
                

        }
           for (int i = 0; i < dim; i++)
        {   
            if(second_norm < s_nrm_two[i])
                second_norm = s_nrm_two[i];
        }
        printf("---> FIRST NORM: %lf\n", first_norm );
        printf("---> SECOND NORM: %lf\n", second_norm );
        alpha = first_norm / second_norm;
        printf("---> ALPHA: %lf\n", alpha);
        double a = first_norm/second_norm;
        printf("A IS: %lf \n", a );
        output = ceil(log2((2 * alpha) / 2.220446049250313e-16) / (2 * val));
        printf("OUTPUT IS: %lf", output);
        if(output <= 0.0){
        m_val[k] = 0.0;
        printf("-----> %ld\n", m_val );
        }
        if(val == 13){
             m_val[k] = output;
             printf("SHOULD BE DONE\n");
             printf("-->%lf\n", output );
        }
    }
}

__global__ void identity_kernel(cuDoubleComplex* identity, int dim){
    printf("->HERE\n");
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
        for (int i = 0; i < dim; i++)
        {
            if(res[0] < s[i])
                res[0] = s[i];
        }
    }
}



__global__ void get_one_norm( cuDoubleComplex* A,  double* res, int k, int dim){
    extern __device__ double s[];   // Device memory array to store column sums 
    res[k] = 0;
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
        for (int i = 0; i < dim; i++)
        {
            if(res[k] < s[i])
                res[k] = s[i];
        }
    }
    printf("%lf", res[k]);
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

void set_Identity(cuDoubleComplex* A, int dim, cudaStream_t stream){
    int dimensions = (int) ceil((float)(dim)/BLOCK_SIZE);
    dim3 dimGrid(dimensions, dimensions, 1); // Set a grid of 2*2 blocks
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1); // Set each block to be 2*2 threads
    printf("%d", dimensions);
    identity_kernel<<<dimGrid, dimBlock, 0, stream>>>(A, dim);
}

// Scale a matrix
void scale_tester(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n){

    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, NULL, n, d_C, n);
}

// Scale first matrix and then add the second matrix to the result:
void scale_and_add(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n, cudaStream_t my_stream){
    
    const cuDoubleComplex bet = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;
    cublasSetStream(handle, my_stream);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);
}


// Scale first matrix and then add the second matrix to the result:
void scale_and_add_alt(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n){
    
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


// Calulate the absolute values of the entries of a complex matrix: 

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

int main(int argc, char* argv[])
{

/////////////////////////////////////////////////////////////////////////////////////////////////// SETUP START /////////////////////////////////////////////////////////////////////////////////////////////
	int dim = 5;
	int batch_count = 100;
 
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
                A[i][(dim*j) + k] = make_cuDoubleComplex(i,1);
            }
        }
    }
    

    // WRITE THE 5th INPUT MATRIX FOR COMPARISON WITH MATLAB:
    write_input_matrix(A[5], dim);


    // Create cublas instance
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasHandle_t handle2;
    cublasCreate(&handle2);
 
     // *** CURRENT: CONVERT THESE TO PINNED MEMORY
    // Create host pointer array to device matrix storage
    cuDoubleComplex **d_T1, **d_T2, **d_T4, **d_T6, **d_T8, **d_T10, **d_identity,
     **h_d_T1, **h_d_T2, **h_d_T4, **h_d_T6, **h_d_T8, **h_d_T10, **h_d_identity;
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
    h_d_identity = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
 
    for(int i=0; i<batch_count; i++) {
        cudaMalloc((void**)&h_d_T1[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T2[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T4[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T6[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T8[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_T10[i], dim*dim*sizeof(cuDoubleComplex));
        cudaMalloc((void**)&h_d_identity[i], dim*dim*sizeof(cuDoubleComplex));

    }
    
    // Copy the host array of device pointers to the device
    cudaMalloc((void**)&d_T1, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T2, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T4, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T6, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T8, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_T10, batch_count*sizeof(cuDoubleComplex*));
    cudaMalloc((void**)&d_identity, batch_count*sizeof(cuDoubleComplex*));
    cudaMemcpy(d_T1, h_d_T1, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T2, h_d_T2, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T4, h_d_T4, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T6, h_d_T6, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T8, h_d_T8, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T10, h_d_T10, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_identity, h_d_identity, batch_count*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
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
  double* d4;
  double* d6;
  double* d8;
  double* d10;

  cudaMallocHost((void**)&d4, batch_count*sizeof(double));
  cudaMallocHost((void**)&d6, batch_count*sizeof(double));
  cudaMallocHost((void**)&d8, batch_count*sizeof(double));
  cudaMallocHost((void**)&d10, batch_count*sizeof(double));


//////////////// STREAMS FOR BATCH ////////////////////////  
  cudaStream_t streams[5];
  int n_streams = 5;
  for (int i = 0; i < 5; i++) {
    cudaStreamCreate(&streams[i]);
}

//////////////// STREAMS FOR BATCH ////////////////////////


    int dimensions = ceil((float) dim/BLOCK_SIZE);
    dim3 dimGrid(dimensions, 1, 1); // Set a grid of 2*2 blocks
    dim3 dimBlock(BLOCK_SIZE, 1, 1); // Set each block to be 2*2 threads
    printf("THE NUMBER OF BLOCKS IS: (%d, %d)\n", dimensions, dimensions);

    // double* res;
    double* d_res;
    double* d_res2;
    double* d_res3;
    double* d_res4;
    cudaMalloc((void**)&d_res, sizeof(double)*batch_count);
    cudaMalloc((void**)&d_res2, sizeof(double)*batch_count);
    cudaMalloc((void**)&d_res3, sizeof(double)*batch_count);
    cudaMalloc((void**)&d_res4, sizeof(double)*batch_count);


for (int i = 0; i < batch_count; i++) // 1-norm needs work
{
  //norm_1_kernel_large_alt<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_T4[i], dim, d_res[0]); // Uses global memory
  get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[1]>>>(h_d_T4[i], d_res, i, dim); // Uses global memory
  get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[2]>>>(h_d_T6[i], d_res2, i, dim); // Uses global memory
  get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[3]>>>(h_d_T8[i], d_res3, i, dim); // Uses global memory
  get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[4]>>>(h_d_T10[i], d_res4, i, dim);
}
cudaMemcpyAsync(d4, d_res, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[1]);
cudaMemcpyAsync(d6, d_res2, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[2]);
cudaMemcpyAsync(d8, d_res3, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[3]);
cudaMemcpyAsync(d10, d_res4, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[4]);

cudaDeviceSynchronize();

  for (int i = 0; i < batch_count; i++) // 1-norm needs work
  {
     printf("%lf\n", d4[i]);
    d4[i] = pow(d4[i], (1.0 / 4));

    d6[i] = pow(d6[i], (1.0 / 6));

    d8[i] = pow(d8[i], (1.0 / 8));

    d10[i] = pow(d10[i], (1.0 / 10));
}



  // [PART 3] CALCULATE (eta1, eta3, eta4, eta5)
  double* eta1 = (double*) malloc(batch_count*sizeof(double));
  double* eta3 = (double*) malloc(batch_count*sizeof(double));
  double* eta4 = (double*) malloc(batch_count*sizeof(double));
  double* eta5 = (double*) malloc(batch_count*sizeof(double));
  int* m_val = (int*) malloc(batch_count*sizeof(int));
  memset(m_val, 0, batch_count*sizeof(int));

  for (int i = 0; i < batch_count; i++)
  {
    eta1[i] = fmax(d4[i], d6[i]);
    eta3[i] = fmax(d6[i], d8[i]);
    eta4[i] = fmax(d8[i], d10[i]);
    eta5[i] = fmin(eta3[i], eta4[i]); 

  }



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



///////////////////////////////////////////// ALTERNATIVE ELL PART A ////////////////////////////////////////////////////
// Try for m_val = 3:
int dimensions_alt = ceil((float) dim/BLOCK_SIZE);
dim3 dimGrid_alt(dimensions_alt, dimensions_alt, 1); // Set a grid of 2*2 blocks
dim3 dimBlock_alt(BLOCK_SIZE,BLOCK_SIZE,1); // Set each block to be 2*2 threads

int* d_m_val;
cudaMalloc(&d_m_val, sizeof(double)*batch_count);

// CHECK FOR m_val = 3:
for (int i = 0; i < batch_count; i++) // 1-norm needs work
    {

        if(eta1[i] <=theta[1]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock, 0, streams[i%3]>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[1], (1.0 / (2 * 3 + 1)));
            cublasSetStream(handle, streams[i]);
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), streams[i%3]>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 3);
        }
    }
cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(double)*batch_count, cudaMemcpyDeviceToHost);

// CHECK FOR m_val = 5:
for (int i = 0; i < batch_count; i++) // 1-norm needs work
    {
        if(eta1[i] <=theta[2]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[2], (1.0 / (2 * 5 + 1)));
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 5);
        }
    }
cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(double)*batch_count, cudaMemcpyDeviceToHost);

// CHECK FOR m_val = 7:
for (int i = 0; i < batch_count; i++) // 1-norm needs work
    {
        if(eta3[i] <=theta[3]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[3], (1.0 / (2 * 7 + 1)));
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 7);
        }
    }
cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(double)*batch_count, cudaMemcpyDeviceToHost);

// CHECK FOR m_val = 9:
printf("-----------------------------------------------\n");
for (int i = 0; i < batch_count; i++) // 1-norm needs work
    {
        if(eta3[i] <= theta[4]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[4], (1.0 / (2 * 9 + 1)));
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 9);
        }
    }
cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(double)*batch_count, cudaMemcpyDeviceToHost);

///////////////////////////////////////////// ALTERNATIVE ELL PART A ////////////////////////////////////////////////////


  // PRINT A SAMPLE
  printf("\n");
  printf("%d", m_val[1]);
  printf("\n");
  printf("%d", m_val[2]);
  printf("\n");
  printf("%d", m_val[3]);
  printf("\n");
  printf("%d\n", m_val[4]);


///////////////////////////////////////////////////////////////////////////////////////////////////////
// [PART 5] CALULATE s
  double* s = (double*) malloc(batch_count*sizeof(double)); 
  double max = 0;



//////////////////////////////////// ALTERNATIVE ELL PART B /////////////////////////////////////////////
  ///////////// CURRENT WORK: GET KERNEL OVERLAP //////////////////////////////////////
int* temp_array;
temp_array = (int*) malloc(sizeof(int)*batch_count);
  for (int i = 0; i < batch_count; i++)
  {
    s[i] = fmax(ceil(log2(eta5[i]/theta[4])), 0);
    //Perform scale:
    cublasSetStream(handle, streams[i%n_streams]);
    scale_tester(handle, h_d_A[i], h_d_A[i], make_cuDoubleComplex(1/pow(2, s[i]), 0), dim);
    //Perform ell:
    cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice, streams[i%n_streams]);
    absolute_kernel<<<dimGrid, dimBlock, 0, streams[i%n_streams]>>>(h_d_B[i], dim);
    scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(pow(error_coefficients[4], (1.0 / (2 * 13 + 1))), 0), dim);
    ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), streams[i%n_streams]>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 13);
}

cudaMemcpy(temp_array, d_m_val, sizeof(int)*batch_count, cudaMemcpyDeviceToHost);

  for (int i = 0; i < batch_count; i++)
  {
    printf("S BEFORE IS: %lf\n", s[i] );
    s[i] = s[i] + temp_array[i];
    printf("Temp Array is: %d\n", temp_array[i] );
    printf("S AFTER IS: %lf\n", s[i] );
    if(s[i] > max)
        max = s[i];
}



//////////////////////////////////// ALTERNATIVE ELL PART B /////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////

printf("S IS: %lf\n", s[0] );


    printf("eta3 IS: %lf \n", eta3[1]);
    printf("eta4 IS: %lf \n", eta4[1]);
    printf("eta5 IS: %lf \n", eta5[1]);

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

  cudaMemset(h_d_C, 0, dim*dim*sizeof(cuDoubleComplex)*batch_count);

                                                  // >>> IDENTITY SET
    cublasSetStream(handle2, streams[2]);
    set_Identity(h_d_identity[0], dim, streams[1]);   

   for (int i = 0; i < batch_count; i++)
   {
    scale_and_add_alt(handle2, h_d_T6[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][13], 0), dim); // >>> SCALING
    scale_and_add_alt(handle2, h_d_T4[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][11], 0), dim);
    scale_and_add_alt(handle2, h_d_T2[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][9], 0), dim);
}

cudaDeviceSynchronize();

// Perform batch matrix multiplication
  cublasZgemmBatched(handle,                                                    // >>> BATCH MATRIX MULTILICATION
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_C, dim,
                      (const cuDoubleComplex**)d_T6, dim,
                      beta,
                      d_C, dim,
                      batch_count);

  cudaDeviceSynchronize();
  for (int i = 0; i < batch_count; i++)
  {

    scale_and_add_alt(handle, h_d_T6[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][7], 0), dim); // >>> SCALING
    scale_and_add_alt(handle, h_d_T4[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][5], 0), dim);
    scale_and_add_alt(handle, h_d_T2[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][3], 0), dim);
    scale_and_add_alt(handle, h_d_identity[0], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][1], 0), dim);
    scale_and_add(handle, h_d_C[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim, 0);

  }



  // BATCH MATRIX MULTIPLY:
  cublasZgemmBatched(handle,                                                                    // >>> BATCH MATRIX MULTILICATION
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
    scale_and_add_complete(handle, h_d_T6[i], h_d_T4[i], h_d_B[i], make_cuDoubleComplex(c[i][12], 0), make_cuDoubleComplex(c[i][10], 0), dim); // >>> SCALING
    scale_and_add(handle, h_d_T2[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][8], 0), dim, 0);
}

   // BATCH MATRIX MULTIPLY:

    cublasZgemmBatched(handle,                                                      // >>> BATCH MATRIX MULTILICATION
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_B, dim,
                      (const cuDoubleComplex**)d_T6, dim,
                      beta,
                      d_A, dim,
                      batch_count);

  cudaDeviceSynchronize();
                                        
cudaMemset(h_d_B, 0, dim*dim*sizeof(cuDoubleComplex)*batch_count);
for (int i = 0; i < batch_count; i++)
  {  
    scale_and_add(handle, h_d_T6[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][6], 0), dim, 0); // >>> SCALING
    scale_and_add(handle, h_d_T4[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][4], 0), dim, 0);
    scale_and_add(handle, h_d_T2[i], h_d_A[i], h_d_A[i], make_cuDoubleComplex(c[i][2], 0), dim, 0);
    scale_and_add(handle, h_d_identity[0], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][0], 0), dim, 0);
    scale_and_add(handle, h_d_A[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim, 0);

    // CALCULATE (V-U):
    scale_and_subtract(handle, h_d_B[i], h_d_C[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim); // THIS WOULD BE THE SYNCHRONIZATION POINT
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
                      (const cuDoubleComplex**)d_C, dim,
                      (const cuDoubleComplex**)d_A, dim,
                      beta,
                      d_B, dim,
                      batch_count);

  cudaDeviceSynchronize();

  for(int i=0; i<batch_count; i++) {
      scale_and_add(handle, h_d_B[i], h_d_identity[0], h_d_B[i], make_cuDoubleComplex(1, 0), dim, 0);
  }


  // SQUARING PHASE:

  printf("MAX IS: %lf\n", max);
  printf("WE WANT: %lf\n", s[5]);
  
  for (int k = 0; k < max; k++) { 
  
    // PERFORM BATCH MATRIX MULTIPLICATION -> Try to remove the intermediate memory copies:
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
        if(i==k)
        {
            cudaMemcpy(A[i], h_d_C[i], dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        }
    }
    
    d_B = d_C; // Set factor matrix for next iteration to the product matrix of the last iteration
}

 printf("ANS IS: \n");
 matrix_complex_print(A[5], dim);
 return 0;
}


// Scale operations are compute bound and no advantage being gained from batch parallelism