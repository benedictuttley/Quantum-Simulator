// Cuda with single matrix input - 08/04/2019
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

// One option maybe is to create a pool of arrays
// Only works for dimensions up to 128

#define BLOCK_SIZE 32 // Block width and height for all kernels

__global__ void ell_kernel(cuDoubleComplex* A, cuDoubleComplex* B, int dim, int* m_val, int val){
    extern __device__ double s_nrm_one[];   // Device memory array to store column sums 
    extern __device__ double s_nrm_two[];   // Device memory array to store column sums 

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y;
    
    if(tid_x < dim){
        double sum_nrm_one = 0; // Private variable to hold column sum
        double sum_nrm_two = 0; // Private variable to hold column sum

        for (int i = 0; i < dim; i++){   // Calculate column sums, one column per thread
            sum_nrm_one += cuCabs(A[(i*dim) + tid_x]);
            sum_nrm_two += cuCabs(B[(i*dim) + tid_x]);
        }

        s_nrm_one[tid_x] = sum_nrm_one;
        s_nrm_two[tid_x] = sum_nrm_two;
    }
    
    __syncthreads(); // sum contains the column sums

    if(tid_x == 1 && tid_y == 1){
        double second_norm = 0;
        double first_norm = 0;
        
        for (int i = 0; i < dim; i++){
            if(first_norm < s_nrm_one[i])
                first_norm = s_nrm_one[i];
            if(second_norm < s_nrm_two[i])
                second_norm = s_nrm_two[i];
        }

        double alpha = second_norm/ first_norm;
        double output = ceil(log2((2 * alpha) / 2.220446049250313e-16) / (2 * val));

        if(output <= 0.0)
            m_val[0] = 0.0;
        if(val == 13)
            m_val[0] = output;
    }
}


__global__ void identity_kernel(cuDoubleComplex* identity, int dim){
    
    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y; 

    if(tid_x < dim && tid_y < dim){              
        
        if(tid_x == tid_y) {
            identity[(dim*tid_x) + tid_y].x = 1;
            identity[(dim*tid_x) + tid_y].y = 0;
        }
        else {
            identity[(dim*tid_x) + tid_y].x = 0;
            identity[(dim*tid_x) + tid_y].y = 0;
        }
    } 
}


__global__ void absolute_kernel(cuDoubleComplex* A, int dim){

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y;

    if(tid_x < dim && tid_y < dim){ 
        A[(dim*tid_y) + tid_x].x = cuCabs((A[(dim*tid_y) + tid_x]));
        A[(dim*tid_y) + tid_x].y = 0;
    }
}



__global__ void get_one_norm( cuDoubleComplex* A,  double* output, int dim){


    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    extern __device__ double s[];
    
    output[0] = 0;

    double sum = 0; // Private variable to hold column sum
    if(tid_x < dim){ // Check the problem bounds
        for (int i = 0; i < dim; ++i){   // Calculate column sums, one column per thread
            sum += cuCabs(A[(i*dim) + tid_x]);
        }
        s[tid_x] = sum;
    }

    __syncthreads(); // sum contains the column sums


    if (tid_x == 0){ // Find the maximum of the column sums using thread 0
        for (int i = 0; i < dim; i++){
            if(output[0] < s[i])
                output[0] = s[i];
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
                }
                else {
                    fprintf(f, "%lf", A[(j*n) + i].x );
                    fprintf(f, "+");
                    fprintf(f, "%lfi ", A[(j*n) + i].y );
                }
            } else {
                if (A[(n * j) + i].x == INFINITY) {
                    fprintf(f, "Inf ");
                } 
                else {
                    fprintf(f, "%lf", A[(j*n) + i].x );
                    fprintf(f, "+");
                    fprintf(f, "%lfi ", A[(j*n) + i].y );;
                }
            }
        }
        fprintf(f, "\n");
    }
}

// TODO: Remove this
void set_Identity(cuDoubleComplex* A, int dim, cudaStream_t stream){
    int dimensions = (int) ceil((float)(dim)/BLOCK_SIZE);
    dim3 dimGrid(dimensions, dimensions, 1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1); 
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

// Calculate matrix inverse through LU factorisation:
void InverseOfMatrix_Alternative_Two(cuDoubleComplex* L, cuDoubleComplex* inverse, int n, cuDoubleComplex* b){ 
    
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
    if(status!=CUSOLVER_STATUS_SUCCESS)
        printf("ERROR!!\n");

    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
 
    if (h_info != 0){
        fprintf(stderr, "Error: LU factorization failed\n");
        printf("%d\n", h_info );
    }
      
    cusolverDnZgetrs(handler, CUBLAS_OP_N, n, n, A, n, dLUPivots_ALT, x, n, dLUInfo_ALT);
    cudaDeviceSynchronize();
    
    if(status!=CUSOLVER_STATUS_SUCCESS)
        printf("ERROR!!\n");
    
    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_info != 0)
        fprintf(stderr, "Error: LU factorization failed\n");
    
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


extern "C" void expm_initialization(const void * input, void * output, int dim) {

    /* *** Intial setup ***
     --------------------
    */

    // Size of input matrix:
	//int dim = 16;

    // Allocate host array A to construct input matrix:
    cuDoubleComplex *A = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
    A = (cuDoubleComplex*) input;

    // // Populate matrix:
    // for (int j = 0; j< dim; j++){
    //     for (int k = 0; k < dim; k++)
    //     {
    //         A[(dim*j) + k] = make_cuDoubleComplex(j, 0);
    //     }
    // }
    
    // Write input matrix for comparison:
    write_input_matrix(A, dim);

    // Create cublas instance
    cublasHandle_t handle, handle2, handle3;
    cublasCreate(&handle);
    cublasCreate(&handle2);
    cublasCreate(&handle3);

    // Create a handful of streams:
    cudaStream_t streams[5];
    for (int i = 0; i < 5; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Bind streams to a cuda handle:
    cublasSetStream(handle, streams[1]);
    cublasSetStream(handle2, streams[2]);
    cublasSetStream(handle3, streams[3]);

    // Create device pointers for device arrays
    cuDoubleComplex *d_T1, *d_T2, *d_T4, *d_T6, *d_T8, *d_T10, *d_identity, *d_A, *d_B, *d_C;


    // Allocate memory for device arrays:
    cudaMalloc(&d_T1, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_T2, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_T4, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_T6, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_T8, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_T10,dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_identity,dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_A, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_B, dim*dim*sizeof(cuDoubleComplex));
    cudaMalloc(&d_C, dim*dim*sizeof(cuDoubleComplex));

    // Copy input matrix to device memory:
    cudaMemcpy(d_A, A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);// Copy input array to device A
    cudaMemcpy(d_T1, A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); // Copy input array to device T1


    // Alpha and beta coeficients set for zgemm:
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;


    // *** Powers of input matrix calculated using ZGEMM
 
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
    cudaDeviceSynchronize();
  	

    // Calculate T6:
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
    cublasZgemm(handle2,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_T4, dim,
                d_T4, dim,
                beta,
                d_T8, dim);
 
   
    // Calculate T10:
    cublasZgemm(handle3,                   
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                alpha,
                d_T6, dim,
                d_T4, dim,
                beta,
                d_T10, dim);
    cudaDeviceSynchronize();


    double* d4 = (double*) malloc(sizeof(double));
    double* d6 = (double*) malloc(sizeof(double));
    double* d8 = (double*) malloc(sizeof(double));
    double* d10 = (double*) malloc(sizeof(double));


    // Cuda Grid setup:
    int dimensions = ceil((float) dim/BLOCK_SIZE);
    dim3 dimGrid(dimensions, 1, 1); 
    dim3 dimBlock(BLOCK_SIZE, 1, 1); 

    printf("dimensions: %d \n", dimensions);


    double* d_res;
    double* d_res2;
    double* d_res3;
    double* d_res4;
    cudaMalloc(&d_res, sizeof(double));
    cudaMalloc(&d_res2, sizeof(double));
    cudaMalloc(&d_res3, sizeof(double));
    cudaMalloc(&d_res4, sizeof(double));
    
    // Calculate matrix norms for each powers of A:
    get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[1]>>>(d_T4, d_res, dim); 
    get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[2]>>>(d_T6, d_res2, dim); 
    get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[3]>>>(d_T8, d_res3, dim); 
    get_one_norm<<<dimGrid, dimBlock, sizeof(double), streams[4]>>>(d_T10, d_res4, dim);

    cudaMemcpyAsync(d4, d_res, sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(d6, d_res2, sizeof(double), cudaMemcpyDeviceToHost, streams[2]);
    cudaMemcpyAsync(d8, d_res3, sizeof(double), cudaMemcpyDeviceToHost, streams[3]);
    cudaMemcpyAsync(d10, d_res4, sizeof(double), cudaMemcpyDeviceToHost, streams[4]);

    cudaDeviceSynchronize();

    d4[0] = pow(d4[0], (1.0 / 4));
    d6[0] = pow(d6[0], (1.0 / 6));
    d8[0] = pow(d8[0], (1.0 / 8));
    d10[0] = pow(d10[0], (1.0 / 10));

    printf("d4 is: %lf\n", d4[0]);
    printf("d6 is: %lf\n", d6[0]);
    printf("d8 is: %lf\n", d8[0]);
    printf("d10 is: %lf\n", d10[0]);


    double* eta1 = (double*) malloc(sizeof(double));
    double* eta3 = (double*) malloc(sizeof(double));
    double* eta4 = (double*) malloc(sizeof(double));
    double* eta5 = (double*) malloc(sizeof(double));
    int* m_val = (int*) malloc(sizeof(int));
 
    eta1[0] = fmax(d4[0], d6[0]);
    eta3[0] = fmax(d6[0], d8[0]);
    eta4[0] = fmax(d8[0], d10[0]);
    eta5[0] = fmin(eta3[0], eta4[0]);

    printf("eta1 is: %lf\n", eta1[0]); 
    printf("eta3 is: %lf\n", eta3[0]);
    printf("eta4 is: %lf\n", eta4[0]);
    printf("eta5 is: %lf\n", eta5[0]);

    // *** Find value of m_val from set {3, 5, 7, 9, 13}

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


    /* *** Test for m_val equal to (3) ***
    --------------------------------------
    */

    int* d_m_val;
    cudaMalloc(&d_m_val, sizeof(int));

    if(eta1[0] <= theta[1]){
        cudaMemcpyAsync(d_B, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        absolute_kernel<<<dimGrid, dimBlock>>>(d_B, dim);
        double p = pow(error_coefficients[1], (1.0 / (2 * 3 + 1)));
        cublasSetStream(handle, streams[0]);
        scale_tester(handle, d_B, d_B, make_cuDoubleComplex(p, 0), dim);
        ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(d_A, d_B, dim, d_m_val, 3);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);


    /* *** Test for m_val equal to (5) ***
    --------------------------------------
    */

    if(eta1[0] <= theta[2]){
        cudaMemcpyAsync(d_B, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        absolute_kernel<<<dimGrid, dimBlock>>>(d_B, dim);
        double p = pow(error_coefficients[2], (1.0 / (2 * 5 + 1)));
        scale_tester(handle, d_B, d_B, make_cuDoubleComplex(p, 0), dim);
        ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(d_A, d_B, dim, d_m_val, 5);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

    /* *** Test for m_val equal to (7) ***
    --------------------------------------
    */

    if(eta3[0] <= theta[3]){
        cudaMemcpyAsync(d_B, d_A, dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
        absolute_kernel<<<dimGrid, dimBlock>>>(d_B, dim);
        double p = pow(error_coefficients[3], (1.0 / (2 * 7 + 1)));
        scale_tester(handle, d_B, d_B, make_cuDoubleComplex(p, 0), dim);
        ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(d_A,d_B, dim, d_m_val, 7);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

    /* *** Test for m_val equal to (9) ***
    --------------------------------------
    */

    if(eta3[0] <= theta[4]){
        cudaMemcpyAsync(d_B, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        absolute_kernel<<<dimGrid, dimBlock>>>(d_B, dim);
        double p = pow(error_coefficients[4], (1.0 / (2 * 9 + 1)));
        scale_tester(handle, d_B, d_B, make_cuDoubleComplex(p, 0), dim);
        ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(d_A,d_B, dim, d_m_val, 9);
    }
    
    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

    /* *** Find the value of s indicating the number of squares that will be performed ***
    --------------------------------------------------------------------------------------
    */

    double* s = (double*) malloc(sizeof(double)); 
    
    // Cuda Grid setup:
    int dimensions_ell= ceil((float) dim/BLOCK_SIZE);
    dim3 dimBlock_ell(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid_ell(dimensions_ell, dimensions_ell); 

    int* temp_array;
    temp_array = (int*) malloc(sizeof(int));
    
    s[0] = fmax(ceil(log2(eta5[0]/theta[4])), 0);
    printf("s before is %lf\n", s[0]);
    //Perform scale:
    cublasSetStream(handle, streams[0]);
    scale_tester(handle, d_A, d_A, make_cuDoubleComplex(1/pow(2, s[0]), 0), dim);
    cudaMemcpy(A, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    //Perform ell:
    cudaMemcpy(d_B, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    absolute_kernel<<<dimGrid_ell, dimBlock_ell, 0, streams[1]>>>(d_B, dim);
    cudaDeviceSynchronize();
    scale_tester(handle, d_B, d_B, make_cuDoubleComplex(pow(error_coefficients[4], (1.0 / (2 * 13 + 1))), 0), dim);
    ell_kernel<<<dimGrid_ell, dimBlock_ell, dim*sizeof(double), streams[0]>>>(d_A,d_B, dim, d_m_val, 13);
    cudaMemcpy(temp_array, d_m_val, sizeof(int), cudaMemcpyDeviceToHost);

    // Final value of s:
    s[0] = s[0] + temp_array[0];

    // check s for infinity
    if (isinf(s[0]))
    {
        printf("S/M CHECK HAS BEEN HIT\n");
        exit(0);
    } else{
        m_val[0] = 13;
    }

    /* *** Rescale powers of A according to value of s ***
    -------------------------------------------------------
    */
    
    printf("s after is: %lf\n", s[0]);
    if (s[0]!=0)
    {
        scale_tester(handle, d_T1, d_T1, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 1)), 0), dim);
        scale_tester(handle, d_T2, d_T2, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 2)), 0), dim);
        scale_tester(handle, d_T4, d_T4, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 4)), 0), dim);
        scale_tester(handle, d_T6, d_T6, make_cuDoubleComplex(1.0 / pow(2, (s[0] * 6)), 0), dim);
    }


    /* Get pade coefficients 
    ------------------------
    */

    double* c = (double*) malloc(15*sizeof(double)); 
    get_pade_coefficients(c, m_val[0]);

    if(m_val[0] != 13){
        printf("DIFFERENCE IS SEEN!\n");
        exit(0);
    }

    /* Calculate U
    --------------
    */
    
    set_Identity(d_identity, dim, 0);   // Create an identity matrix
    scale_and_add_alt(handle2, d_T6, d_C, d_C, make_cuDoubleComplex(c[13], 0), dim); 
    scale_and_add_alt(handle2, d_T4, d_C, d_C, make_cuDoubleComplex(c[11], 0), dim);
    scale_and_add_alt(handle2, d_T2, d_C, d_C, make_cuDoubleComplex(c[9], 0), dim);
    cudaDeviceSynchronize();

    cublasZgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        d_C, dim,
        d_T6, dim,
        beta,
        d_C, dim
        );

    cudaDeviceSynchronize();

    scale_and_add_alt(handle, d_T6, d_C, d_C, make_cuDoubleComplex(c[7], 0), dim); 
    scale_and_add_alt(handle, d_T4, d_C, d_C, make_cuDoubleComplex(c[5], 0), dim);
    scale_and_add_alt(handle, d_T2, d_C, d_C, make_cuDoubleComplex(c[3], 0), dim);
    scale_and_add_alt(handle, d_identity, d_C, d_C, make_cuDoubleComplex(c[1], 0), dim);
    scale_and_add(handle, d_C, d_B, d_B, make_cuDoubleComplex(1, 0), dim, 0);

    cublasZgemm(
        handle,                    
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        d_B, dim,
        d_T1, dim,
        beta,
        d_C, dim
        );

    cudaDeviceSynchronize();

    /* Calculate V
    --------------
    */
    
    scale_and_add_complete(handle, d_T6, d_T4, d_B, make_cuDoubleComplex(c[12], 0), make_cuDoubleComplex(c[10], 0), dim);
    scale_and_add(handle, d_T2, d_B, d_B, make_cuDoubleComplex(c[8], 0), dim, 0);

    cublasZgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        d_B, dim,
        d_T6, dim,
        beta,
        d_A, 
        dim
        );

    cudaDeviceSynchronize();

    scale_and_add(handle, d_T6, d_B, d_B, make_cuDoubleComplex(c[6], 0), dim, 0);
    scale_and_add(handle, d_T4, d_B, d_B, make_cuDoubleComplex(c[4], 0), dim, 0);
    scale_and_add(handle, d_T2, d_A, d_A, make_cuDoubleComplex(c[2], 0), dim, 0);
    scale_and_add(handle, d_identity, d_B, d_B, make_cuDoubleComplex(c[0], 0), dim, 0);
    scale_and_add(handle, d_A, d_B, d_B, make_cuDoubleComplex(1, 0), dim, 0);


    /* Calculate V-U
    ---------------
    */
  
    scale_and_subtract(handle, d_B, d_C, d_B, make_cuDoubleComplex(1, 0), dim);


    /* *** Calculate F = (V-U)/(2*U) + I ***
    ----------------------------------------
    */

    // Find inverse of (V-U) through LU decomposition:
    cudaMemcpy(A, d_B, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    //matrix_complex_print(A, dim);
    //printf("IDENTITY\n");
    set_Identity(d_A, dim, 0);
    cudaMemcpy(A, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    //matrix_complex_print(A, dim);
    //exit(0);
    InverseOfMatrix_Alternative_Two(d_B, d_A, dim, d_A);
    cudaMemcpy(A, d_A, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    matrix_complex_print(A, dim);
    //exit(0);

    // Scale U by 2:
    scale_tester(handle, d_C, d_C, make_cuDoubleComplex(2, 0), dim);

    cublasZgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        d_C, dim,
        d_A, dim,
        beta,
        d_B, dim
        );

    cudaDeviceSynchronize();

    scale_and_add(handle, d_B, d_identity, d_B, make_cuDoubleComplex(1, 0), dim, 0);



    /* *** Square F a total of s times: ***
    ---------------------------------------
    */
    
    for (int i = 0; i <s[0]; i++) { 
        cublasZgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        d_B, dim,
        d_B, dim,
        beta,
        d_C, dim
        );

        cudaDeviceSynchronize();

        d_B = d_C; // Switch pointers to avoid memory copies
    }

    // Result stored in device array d_C:
    // printf("Matrix exponential: \n");
    cudaMemcpy(output, d_C, dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // matrix_complex_print(A, dim);

    // Free memory:

    cudaFree(d_T1);
    cudaFree(d_T2);
    cudaFree(d_T4);
    cudaFree(d_T6);
    cudaFree(d_T8);
    cudaFree(d_T10);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_identity);
}
