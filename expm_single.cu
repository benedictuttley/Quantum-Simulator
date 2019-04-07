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


__global__ void ell_kernel(cuDoubleComplex* A, cuDoubleComplex* B, int dim, int* m_val, int val){
 extern __device__ double s_nrm_one[];   // Device memory array to store column sums 
 extern __device__ double s_nrm_two[];   // Device memory array to store column sums 

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid_x < dim){
     double sum_nrm_one = 0; // Private variable to hold column sum
     double sum_nrm_two = 0; // Private variable to hold column sum

    for (int i = 0; i < dim; i++)   // Calculate column sums, one column per thread
        {
            sum_nrm_one += cuCabs(A[(i*dim) + tid_x]);
            sum_nrm_two += cuCabs(B[(i*dim) + tid_x]);
        }
        s_nrm_one[tid_x] = sum_nrm_one;
        s_nrm_two[tid_x] = sum_nrm_two;
    }
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
        m_val[0] = 0.0;
        printf("-----> %ld\n", m_val );
        }
        if(val == 13){
             m_val[0] = output;
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

    if(tid_x < dim && tid_y < dim){ // Check the problem bounds
    A[(dim*tid_y) + tid_x].x = cuCabs((A[(dim*tid_y) + tid_x]));
            A[(dim*tid_y) + tid_x].y = 0;
    }

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



__global__ void get_one_norm( cuDoubleComplex* A,  double* res, int dim){
    extern __device__ double s[];   // Device memory array to store column sums 
    res[0] = 0;
    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;

     double sum = 0; // Private variable to hold column sum
     if(tid_x < dim){ // Check the problem bounds
    for (int i = 0; i < dim; ++i)   // Calculate column sums, one column per thread
        {
            sum += cuCabs(A[(i*dim) + tid_x]);
        }
        s[tid_x] = sum;
        }
        __syncthreads(); // sum contains the column sums

    if (tid_x == 0) // Calculate the max column sum using thread 0
    {
        for (int i = 0; i < dim; i++)
        {
            if(res[0] < s[i])
                res[0] = s[i];
        }
    }
    printf("%lf", res[0]);
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


void Inverse(cuDoubleComplex* L, cuDoubleComplex* inverse, int n, cuDoubleComplex* b){
   
    cusolverStatus_t  status;   // Link to the cusolver context
    cusolverDnHandle_t handler;
    status = cusolverDnCreate(&handler);

    cuDoubleComplex* A;
    int* dLUPivots_ALT;
    int* dLUInfo_ALT;
    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    int h_info = 0;
    cuDoubleComplex *x;

    cudaMalloc(&A, sizeof(cuDoubleComplex)*n*n), "Failed to allocate A!";
    cudaMalloc(&x, n * n*sizeof(cuDoubleComplex)), "Failed to allocate x!";
     
    cudaMalloc(&dLUPivots_ALT, n * sizeof(int)), "Failed to allocate dLUPivots!";
    cudaMalloc(&dLUInfo_ALT, sizeof(int)), "Failed to allocate dLUInfo!";
    cudaMemcpy(A, L, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "Failed to copy to adL!";
    cudaMemcpy(x, b, sizeof(cuDoubleComplex)*n*n, cudaMemcpyHostToDevice);

    cusolverDnZgetrf_bufferSize(handler, n, n, (cuDoubleComplex*)A, n, &bufferSize);
    cudaMalloc(&buffer, sizeof(cuDoubleComplex)*bufferSize);
  
    status = cusolverDnZgetrf(handler, n, n, A, n, buffer, dLUPivots_ALT, dLUInfo_ALT);
    if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } 

    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
 
    if ( h_info != 0 ){
        fprintf(stderr, "Error: LU factorization failed\n");
        printf("%d\n", h_info );
    }
      
    cusolverDnZgetrs(handler, CUBLAS_OP_N, n, n, A, n, dLUPivots_ALT, x, n, dLUInfo_ALT);
    cudaDeviceSynchronize();
     if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } 
    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
        if ( h_info != 0 ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }
    cudaMemcpy(inverse, x, sizeof(cuDoubleComplex) * n * n, cudaMemcpyDeviceToHost), "Failed to copy to res!";

    // Free device memory:
    cudaFree(dLUPivots_ALT);
    cudaFree(dLUInfo_ALT);
    cudaFree(A);
    cudaFree(x);
    cudaFree(buffer);
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
	int dim = 2;
	int batch_count = 100;
 
    // Allocate host array A to construct input matrix:
    cuDoubleComplex *A = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
    // INITIALIZE BATCHES WITH DUMMY DATA:
    for(int j = 0; j< dim; j++){
        for (int k = 0; k < dim; k++)
        {
            A[(dim*j) + k] = make_cuDoubleComplex(j,0);
        }
    }
    
    

    // WRITE THE 5th INPUT MATRIX FOR COMPARISON WITH MATLAB:
    write_input_matrix(A, dim);


    // Create cublas instance
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasHandle_t handle2;
    cublasCreate(&handle2);
 
     // *** CURRENT: CONVERT THESE TO PINNED MEMORY
    // Create host pointer array to device matrix storage
    cuDoubleComplex *d_T1, *d_T2, *d_T4, *d_T6, *d_T8, *d_T10, *d_identity,
     *h_d_T1, *h_d_T2, *h_d_T4, *h_d_T6, *h_d_T8, *h_d_T10, *h_d_identity;
    cuDoubleComplex *d_A, *d_B, *d_C, *h_d_A, *h_d_B, *h_d_C;


    cudaMalloc((void**)&h_d_T1, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_T2, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_T4, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_T6, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_T8, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_T10, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_identity, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_A, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_B, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&h_d_C, dim*dim*sizeof(cuDoubleComplex));

    
    
    // Copy the host array of device pointers to the device
    cudaMalloc((void**)&d_T1, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_T2, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_T4, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_T6, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_T8, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_T10,dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_identity,dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_A, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_B, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_C, dim*dim*sizeof(cuDoubleComplex));
    // Copy the host array of device pointers to the device
    cudaMemcpy(d_T1, h_d_T1, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T2, h_d_T2, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T4, h_d_T4, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T6, h_d_T6, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T8, h_d_T8, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T10, h_d_T10, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_identity, h_d_identity, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_d_B, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_d_C, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Copy input matrix to device memory:
    cudaMemcpy(h_d_A, A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);// Copy input array to device A
    cudaMemcpy(h_d_T1, A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); // Copy input array to device T1


    // Alpha and beta coeficients set for zgemm:
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;


/////////////////////////////////////////////////////////////////////////////////////////////////// SETUP END /////////////////////////////////////////////////////////////////////////////////////////////

    // [PART 1] TPOWERS CALULATED USING BATCH DGEMM
    // TODO: Launch each DGEMM operation in own CUDA stream
 
    // Calulate T2:
    cublasZgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_A, dim,
                d_A, dim,
                beta,
                d_T2, dim);

  	  cudaDeviceSynchronize();
  	
  	// Calculate T4:
  	
  	cublasZgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_T2, dim,
                d_T2, dim,
                beta,
                d_T4, dim);

  	// Calculate T6:
  	
  	cudaDeviceSynchronize();

  	cublasZgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_T4, dim,
                d_T2, dim,
                beta,
                d_T6, dim);

    cudaDeviceSynchronize();

    // Calculate T8:

    cublasZgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_T4, dim,
                d_T4, dim,
                beta,
                d_T8, dim);
 
    // No synchronization needed as T10 calc independent of T8

    cublasZgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_T8, dim,
                d_T2, dim,
                beta,
                d_T10, dim);
 
    cudaDeviceSynchronize();
    

  // [PART 2] CALCULATE (d4,d6,d8, d10)
  double* d4 = (double*) malloc(sizeof(double));
  double* d6 = (double*) malloc(sizeof(double));
  double* d8 = (double*) malloc(sizeof(double));
  double* d10 = (double*) malloc(sizeof(double));



// //////////////// STREAMS FOR SINGLE ////////////////////////  
  cudaStream_t streams[5];
  for (int i = 0; i < 5; i++) {
    cudaStreamCreate(&streams[i]);
}

// //////////////// STREAMS FOR SINGLE ////////////////////////


    int dimensions = ceil((float) dim/BLOCK_SIZE);
    dim3 dimGrid(dimensions, 1, 1); // Set a grid of 2*2 blocks
    dim3 dimBlock(BLOCK_SIZE, 1, 1); // Set each block to be 2*2 threads
    printf("THE NUMBER OF BLOCKS IS: (%d, %d)\n", dimensions, dimensions);

    // double* res;
    double* d_res = (double*) malloc(sizeof(double));
    double* d_res2 = (double*) malloc(sizeof(double));
    double* d_res3 = (double*) malloc(sizeof(double));
    double* d_res4 = (double*) malloc(sizeof(double));


  get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[1]>>>(h_d_T4, d_res, dim); 
  get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[2]>>>(h_d_T6, d_res2, dim); 
  get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[3]>>>(h_d_T8, d_res3, dim); 
  get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[4]>>>(h_d_T10, d_res4, dim);


cudaMemcpyAsync(d4, d_res, sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
cudaMemcpyAsync(d6, d_res2, sizeof(double), cudaMemcpyDeviceToHost, streams[2]);
cudaMemcpyAsync(d8, d_res3, sizeof(double), cudaMemcpyDeviceToHost, streams[3]);
cudaMemcpyAsync(d10, d_res4, sizeof(double), cudaMemcpyDeviceToHost, streams[4]);

cudaDeviceSynchronize();

d4[0] = pow(d4[0], (1.0 / 4));

d6[0] = pow(d6[0], (1.0 / 6));

d8[0] = pow(d8[0], (1.0 / 8));

d10[0] = pow(d10[0], (1.0 / 10));




  // [PART 3] CALCULATE (eta1, eta3, eta4, eta5)
  double* eta1 = (double*) malloc(sizeof(double));
  double* eta3 = (double*) malloc(sizeof(double));
  double* eta4 = (double*) malloc(sizeof(double));
  double* eta5 = (double*) malloc(sizeof(double));
  int* m_val = (int*) malloc(sizeof(int));
  memset(m_val, 0, batch_count*sizeof(int));

  for (int i = 0; i < batch_count; i++)
  {
    eta1[0] = fmax(d4[0], d6[0]);
    eta3[0] = fmax(d6[0], d8[0]);
    eta4[0] = fmax(d8[0], d10[0]);
    eta5[0] = fmin(eta3[0], eta4[0]); 

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
cudaMalloc(&d_m_val, sizeof(int)*batch_count);

// CHECK FOR m_val = 3:

if(eta1[0] <=theta[1]){
    cudaMemcpyAsync(h_d_B, h_d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B, dim);
    double p = pow(error_coefficients[1], (1.0 / (2 * 3 + 1)));
    cublasSetStream(handle, streams[0]);
    scale_tester(handle, h_d_B, h_d_B, make_cuDoubleComplex(p, 0), dim);
    ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A, h_d_B, dim, d_m_val, 3);
}
    

cudaDeviceSynchronize();

cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);


// CHECK FOR m_val = 5:
if(eta1[0] <=theta[2]){
    cudaMemcpyAsync(h_d_B, h_d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B, dim);
    double p = pow(error_coefficients[2], (1.0 / (2 * 5 + 1)));
    scale_tester(handle, h_d_B, h_d_B, make_cuDoubleComplex(p, 0), dim);
    ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A, h_d_B, dim, d_m_val, 5);
}

cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

// CHECK FOR m_val = 7:
if(eta3[0] <=theta[3]){
    cudaMemcpyAsync(h_d_B, h_d_A, dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
    absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B, dim);
    double p = pow(error_coefficients[3], (1.0 / (2 * 7 + 1)));
    scale_tester(handle, h_d_B, h_d_B, make_cuDoubleComplex(p, 0), dim);
    ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A,h_d_B, dim, d_m_val, 7);
        }
    
cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

// CHECK FOR m_val = 9:
printf("-----------------------------------------------\n");

        if(eta3[0] <= theta[4]){
            cudaMemcpyAsync(h_d_B, h_d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B, dim);
            double p = pow(error_coefficients[4], (1.0 / (2 * 9 + 1)));
            scale_tester(handle, h_d_B, h_d_B, make_cuDoubleComplex(p, 0), dim);
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A,h_d_B, dim, d_m_val, 9);
        }
    
cudaDeviceSynchronize();
cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);
/////////////////////////////////////////// ALTERNATIVE ELL PART A ////////////////////////////////////////////////////


  // PRINT A SAMPLE
  // printf("\n");
  // printf("%d", m_val[1]);
  // printf("\n");
  // printf("%d", m_val[2]);
  // printf("\n");
  // printf("%d", m_val[3]);
  // printf("\n");
  // printf("%d\n", m_val[4]);


///////////////////////////////////////////////////////////////////////////////////////////////////////
// [PART 5] CALULATE s
  double* s = (double*) malloc(sizeof(double)); 
  double max = 0;



//////////////////////////////////// ALTERNATIVE ELL PART B /////////////////////////////////////////////
  ///////////// CURRENT WORK: GET KERNEL OVERLAP //////////////////////////////////////
int* temp_array;
temp_array = (int*) malloc(sizeof(int));
s[0] = fmax(ceil(log2(eta5[0]/theta[4])), 0);
//Perform scale:
cublasSetStream(handle, streams[0]);
scale_tester(handle, h_d_A, h_d_A, make_cuDoubleComplex(1/pow(2, s[0]), 0), dim);
//Perform ell:
cudaMemcpyAsync(h_d_B, h_d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, streams[0]);
absolute_kernel<<<dimGrid, dimBlock, 0, streams[0]>>>(h_d_B, dim);
scale_tester(handle, h_d_B, h_d_B, make_cuDoubleComplex(pow(error_coefficients[4], (1.0 / (2 * 13 + 1))), 0), dim);
ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), streams[0]>>>(h_d_A,h_d_B, dim, d_m_val, 13);
cudaMemcpy(temp_array, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

printf("S BEFORE IS: %lf\n", s[0] );
s[0] = s[0] + temp_array[0];
printf("Temp Array is: %d\n", temp_array[0] );
printf("S AFTER IS: %lf\n", s[0] );
if(s[0] > max)
max = s[0];




//////////////////////////////////// ALTERNATIVE ELL PART B /////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////



// [PART 6] S CHECK AND M CHECK - [TODO]


  if (isinf(s[0]))
  {
    printf("S/M CHECK HAS BEEN HIT\n");
    exit(0);
  } else{
    m_val[0] = 13;
  }





// [PART 7] RESCALE THE POWERS ARRAYS IF S NOT 0
  if (s[0]!=0)
  {
    scale_tester(handle, h_d_T1, h_d_T1, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 1)), 0), dim);
    scale_tester(handle, h_d_T2, h_d_T2, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 2)), 0), dim);
    scale_tester(handle, h_d_T4, h_d_T4, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 4)), 0), dim);
    scale_tester(handle, h_d_T6, h_d_T6, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 6)), 0), dim);
}



// [PART 7.5] GET THE PADE COEFFICIENTS FOR EACH BATCH

double* c = (double*) malloc(sizeof(double)); 
get_pade_coefficients(c, m_val[0]);

if(m_val[0] != 13){
    printf("DIFFERENCE IS SEEN!\n");
    exit(0);
    }



//if (m_val == 13) // Will need to seperate matrices that are not satisfied for batching to commence
    // [PART 8] CALCULATE U
    set_Identity(h_d_identity, dim, 0);   // >>> IDENTITY SET
    scale_and_add_alt(handle2, h_d_T6, h_d_C, h_d_C, make_cuDoubleComplex(c[13], 0), dim); // >>> SCALING
    scale_and_add_alt(handle2, h_d_T4, h_d_C, h_d_C, make_cuDoubleComplex(c[11], 0), dim);
    scale_and_add_alt(handle2, h_d_T2, h_d_C, h_d_C, make_cuDoubleComplex(c[9], 0), dim);
    cudaDeviceSynchronize();


// Perform batch matrix multiplication
    cublasZgemm(handle,                       // >>> BATCH MATRIX MULTILICATION
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_C, dim,
                d_T6, dim,
                beta,
                d_C, dim);

  cudaDeviceSynchronize();

  scale_and_add_alt(handle, h_d_T6, h_d_C, h_d_C, make_cuDoubleComplex(c[7], 0), dim); // >>> SCALING
  scale_and_add_alt(handle, h_d_T4, h_d_C, h_d_C, make_cuDoubleComplex(c[5], 0), dim);
  scale_and_add_alt(handle, h_d_T2, h_d_C, h_d_C, make_cuDoubleComplex(c[3], 0), dim);
  scale_and_add_alt(handle, h_d_identity, h_d_C, h_d_C, make_cuDoubleComplex(c[1], 0), dim);
  scale_and_add(handle, h_d_C, h_d_B, h_d_B, make_cuDoubleComplex(1, 0), dim, 0);


  // BATCH MATRIX MULTIPLY:
  cublasZgemm(handle,                    // >>> BATCH MATRIX MULTILICATION
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim, dim, dim,
            alpha,
            d_B, dim,
            d_T1, dim,
            beta,
            d_C, dim);

  cudaDeviceSynchronize();


  // [PART 9] CALCULATE V
  scale_and_add_complete(handle, h_d_T6, h_d_T4, h_d_B, make_cuDoubleComplex(c[12], 0), make_cuDoubleComplex(c[10], 0), dim); // >>> SCALING
  scale_and_add(handle, h_d_T2, h_d_B, h_d_B, make_cuDoubleComplex(c[8], 0), dim, 0);


   // BATCH MATRIX MULTIPLY:

    cublasZgemm(handle,                // >>> BATCH MATRIX MULTILICATION
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_B, dim,
                d_T6, dim,
                beta,
                d_A, dim);

  cudaDeviceSynchronize();

  scale_and_add(handle, h_d_T6, h_d_B, h_d_B, make_cuDoubleComplex(c[6], 0), dim, 0); // >>> SCALING
  scale_and_add(handle, h_d_T4, h_d_B, h_d_B, make_cuDoubleComplex(c[4], 0), dim, 0);
  scale_and_add(handle, h_d_T2, h_d_A, h_d_A, make_cuDoubleComplex(c[2], 0), dim, 0);
  scale_and_add(handle, h_d_identity, h_d_B, h_d_B, make_cuDoubleComplex(c[0], 0), dim, 0);
  scale_and_add(handle, h_d_A, h_d_B, h_d_B, make_cuDoubleComplex(1, 0), dim, 0);

  // CALCULATE (V-U):
  scale_and_subtract(handle, h_d_B, h_d_C, h_d_B, make_cuDoubleComplex(1, 0), dim); // THIS WOULD BE THE SYNCHRONIZATION POINT

  // [PART 11] CALCULATE F:

  // BATCH MATRIX INVERSE ON (V-U)
  Inverse(d_B, d_A, dim, d_A);

  // SCALE BATCH U BY 2
  scale_tester(handle, h_d_C, h_d_C, make_cuDoubleComplex(2, 0), dim);

  // BATCH MATRIX MULTIPLICATION:
  cublasZgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              dim, dim, dim,
              alpha,
              d_C, dim,
              d_A, dim,
              beta,
              d_B, dim);

  cudaDeviceSynchronize();

  scale_and_add(handle, h_d_B, h_d_identity, h_d_B, make_cuDoubleComplex(1, 0), dim, 0);
  


  // SQUARING PHASE:

  printf("MAX IS: %lf\n", max);
  printf("WE WANT: %lf\n", s[0]);
  
  for (int i = 0; i < max; i++) { 
  
    // PERFORM BATCH MATRIX MULTIPLICATION -> Try to remove the intermediate memory copies:
    cublasZgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_B, dim,
                d_B, dim,
                beta,
                d_C, dim);

    cudaDeviceSynchronize();
    d_B = d_C; // Set factor matrix for next iteration to the product matrix of the last iteration
}
printf("ANS IS: \n");
 matrix_complex_print(A, dim);

// Scale operations are compute bound and no advantage being gained from batch parallelism

// free Memory:
cudaFree(h_d_T1);
cudaFree(h_d_T2);
cudaFree(h_d_T4);
cudaFree(h_d_T6);
cudaFree(h_d_T8);
cudaFree(h_d_T10);
cudaFree(h_d_A);
cudaFree(h_d_B);
cudaFree(h_d_C);
cudaFree(h_d_identity);


 return 0;
}