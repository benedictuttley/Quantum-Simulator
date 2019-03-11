#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdio.h>  
#include <stdlib.h>
#include <float.h>
#include <memory.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cblas.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "expm.h"

// *** Matrix exponential program acting on matrices of type : [Double Complex] - 10/03/2019 ***

// Converting synchronous program to be more asynchronous and thus allow for better utilization of GPU resources
// and GPU memory copies as to reduce overall runtime
// Approaches for improvement:
// [1] Overlap of memory copies -> The hardware allows for concurrent memory copies that are performed in opposing
// directions.
// [2] Overlap of GPU kernels (computation) -> THe hardware allows for some concurrency of kernel exeution however
// the amount of overlap witnessed depends on how many GPU resources the kernels use, once there are suffcient resources
// released from one kernel, the other kernels execution may begin.
// [3] Need to be aware of data dependencies, this involves ensuring that dependent (synchronous in nature) kernels are
// assigned to the same stream or are seperated by a synchronisation point as to ensure that correct results are achieved.
// [4] Reducing memory copies by keeping data on the device as much as possible, this involves converting current CPU functions
// into GPU functions that will give the same of better runtime (when run on a better GPU) such that less data transfer is
// required.
// [5] Not yet looked at, but another possibility to reduce the number of memcopies but stillm copy the same total amount
// of data is to look at combining arrays representing matrices into a larger memory space and then use offsets to refer to 
// the matrix of interest (perhaps better banwidth utilisation but perhaps less readable).


// TIME BREAKDOWN BY COMPUTE OPERATION - FROM PROFILER:
// 98.4% of runtime is spent in the multiply kernel
// 1.6% of runtime is spent in the inverse kernels

// Memory copies take small relative to operation time to complete for larger matrices
// Small overlap between dgemm kernels due to using large number of available resources so only a small
// amount of kernel concurrency possible
// For smaller matrices (e.g. 50 * 50) more concurrency is witnissed (50% overlap) - 'A more concurrent
// execution.'


void matrix_complex_print(cuDoubleComplex* A, int network_size){
	for (int j = 0; j < network_size; j++){
		printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %.15lf ", A[(j*network_size) + k].x );
			printf("+");
			printf(" %.15lfi ", A[(j*network_size) + k].y );
		}
		printf("]");
		printf("\n");
	}
}

void matrix_Square_Reduced(cublasHandle_t &handle, cuDoubleComplex *d_A, cuDoubleComplex *d_C, int n, double* multiply_total_time, int s){
    
    clock_t multiply_begin = clock();

    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;
    
    for (int k = 0; k < s; k++) {
        cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_A, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication
        cudaMemcpy(d_A, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }
   
    clock_t multiply_end = clock();
    double time_spent = (double)(multiply_end - multiply_begin) / CLOCKS_PER_SEC;
    multiply_total_time[0] = multiply_total_time[0] + time_spent;


}



void matrix_Square(cublasHandle_t &handle, cuDoubleComplex *A, cuDoubleComplex *C, cuDoubleComplex *d_A, cuDoubleComplex *d_C, int n, double* multiply_total_time, int s){
	
    clock_t multiply_begin = clock();

	const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;
	
	cudaMemcpy(d_A, A, n * n * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);	// Copy first operand to the device (only one copy is needed for the squaring phase)

	for (int k = 0; k < s; k++) {
		cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_A, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication
        cudaMemcpy(d_A, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(C, d_C, n * n * sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);	// Copy product back to host
 	
 	clock_t multiply_end = clock();
    double time_spent = (double)(multiply_end - multiply_begin) / CLOCKS_PER_SEC;
    multiply_total_time[0] = multiply_total_time[0] + time_spent;

}



void matrix_Multiply_Reduced(cublasHandle_t &handle, cuDoubleComplex *d_A, cuDoubleComplex* d_B, cuDoubleComplex *d_C, int n, double* multiply_total_time){

    clock_t multiply_begin = clock();

    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_B, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication

    clock_t multiply_end = clock();
    double time_spent = (double)(multiply_end - multiply_begin) / CLOCKS_PER_SEC;
    multiply_total_time[0] = multiply_total_time[0] + time_spent;


}


void matrix_Multiply(cublasHandle_t &handle, cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C, cuDoubleComplex *d_A, cuDoubleComplex* d_B, cuDoubleComplex *d_C, int n, double* multiply_total_time){

    clock_t multiply_begin = clock();

    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cudaMemcpy(d_A, A, n * n * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);	// Copy first operand to the device
    cudaMemcpy(d_B, B, n * n * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice); // Copy second operand to the device
    
    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_B, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication

    cudaMemcpy(C, d_C, n * n * sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);	// Copy product back to host
    
    clock_t multiply_end = clock();
    double time_spent = (double)(multiply_end - multiply_begin) / CLOCKS_PER_SEC;
    multiply_total_time[0] = multiply_total_time[0] + time_spent;
}


void matrix_Multiply_With_Streams(cublasHandle_t &handle, cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C, cuDoubleComplex *d_A, cuDoubleComplex* d_B, cuDoubleComplex *d_C, int n, double* multiply_total_time, cudaStream_t stream){
 clock_t multiply_begin = clock();

    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cudaMemcpyAsync(d_A, A, n * n * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice, stream); // Copy first operand to the device
    cudaMemcpyAsync(d_B, B, n * n * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice, stream); // Copy second operand to the device

    cublasSetStream(handle, stream);
    cublasZgemm3m(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_B, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication

    cudaMemcpyAsync(C, d_C, n * n * sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost, stream); // Copy product back to host
    
    clock_t multiply_end = clock();
    double time_spent = (double)(multiply_end - multiply_begin) / CLOCKS_PER_SEC;
    multiply_total_time[0] = multiply_total_time[0] + time_spent;

}


cuDoubleComplex **get_Matrix_Powers_New(cuDoubleComplex *A, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, cublasHandle_t handle, int n, double* multiply_total_time) {

    cuDoubleComplex **Tpowers = (cuDoubleComplex **) malloc(11 * sizeof(cuDoubleComplex *));

    for (int i = 0; i < 11; i++) {
        Tpowers[i] = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
    }

    memcpy(Tpowers[1], A, n * n * sizeof(cuDoubleComplex));
    

    // To calculate Tpoers[2] & Tpowers[4] this is sequential
    matrix_Multiply(handle, Tpowers[1], Tpowers[1], Tpowers[2], d_A, d_B, d_C, n, multiply_total_time);
     
    matrix_Multiply(handle, Tpowers[2], Tpowers[2], Tpowers[4], d_A, d_B, d_C, n, multiply_total_time);



    matrix_Multiply(handle, Tpowers[4], Tpowers[2], Tpowers[6], d_A, d_B, d_C, n, multiply_total_time);
   
    matrix_Multiply(handle, Tpowers[4], Tpowers[4], Tpowers[8], d_A, d_B, d_C, n, multiply_total_time);

    return Tpowers;
}



// *** CURRENT WORK ***
// Need description of the method being used
// Note that this memthod calls the DGEMM multiplication kernel and as so the functions suffers from the poor FP64
// performance see for the dtandard CUBLAS DGEMM function when ran on the currenctly used GEFORCE GTX 960

/* Calculate matrix inverse through LU factorisation -> AX = I for an input matrix A, the identity matrix I and the inverse matrix X
 [1]
 * The LU factorization of the input double complex matrix is computed using the CUSOLVER function cusolverDnZgetrf
 * This factors a matrix as the product of a lower triangular and upper triangular marix -> [A = LU] 
 
 [2]
 * Then the resultant linear system is solved:
 * A^-1 = U^-1 * L^-1, hence need to invert each of the two matrices
 * As A*A^-1 = I, then L*U*A^-1 = I and we know L, U and I
 * We can the solve a set of linear equations to find the inverse using cusolverDnZgetrs

 * SOURCE: https://math.stackexchange.com/questions/1009916/easy-way-to-calculate-inverse-of-an-lu-decomposition
*/
void InverseOfMatrix_Alternative_Two(cuDoubleComplex* d_in, cuDoubleComplex* d_out, int n){ 
    
    cusolverStatus_t status;	// Link to the cusolver context
    cusolverDnHandle_t handler;
    status = cusolverDnCreate(&handler);

    int* dLUPivots_ALT;
    int* dLUInfo_ALT;
    cuDoubleComplex *buffer = NULL;
    int bufferSize = 0;
    int h_info = 0;
     
    cudaMalloc(&dLUPivots_ALT, n * sizeof(int)), "Failed to allocate dLUPivots!";
    cudaMalloc(&dLUInfo_ALT, sizeof(int)), "Failed to allocate dLUInfo!";

    cusolverDnZgetrf_bufferSize(handler, n, n, (cuDoubleComplex*)d_in, n, &bufferSize);
    cudaMalloc(&buffer, sizeof(cuDoubleComplex)*bufferSize);
  
    status = cusolverDnZgetrf(handler, n, n, d_in, n, buffer, dLUPivots_ALT, dLUInfo_ALT);
    if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } 

    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
 
    if ( h_info != 0 ){
        fprintf(stderr, "Error: LU factorization failed\n");
        printf("%d\n", h_info );
    }
      
    status = cusolverDnZgetrs(handler, CUBLAS_OP_N, n, n, d_in, n, dLUPivots_ALT, d_out, n, dLUInfo_ALT);
    cudaDeviceSynchronize();
     if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } 
    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
        if ( h_info != 0 ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    // Free device memory:
    cudaFree(dLUPivots_ALT);
    cudaFree(dLUInfo_ALT);
    cudaFree(buffer);
}

void matrix_Subtract_New(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n) { // PARALLEL CANDIDATE

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = cuCsub(a[(n * i) + j], b[(n * i) + j]); // Complex subtraction
        }
    }
}


void matrixAdd_New(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n) { // PARALLEL CANDIDATE

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = cuCadd(a[(n * i) + j], b[(n * i) + j]); // Complex addition
        }
    }
}

void matrix_add_Tester(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, int n){
    
    //const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    //cublasZdgmm(handle, mode, n,n, d_A, n, d_X, n, d_C, n);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);
}

void matrix_subtract_Tester(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, int n){
    
    //const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(-1, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    //cublasZdgmm(handle, mode, n,n, d_A, n, d_X, n, d_C, n);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);
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



void scale_tester(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n, double* scale_total_time ){

    clock_t scale_begin = clock();
    //const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    //cublasZdgmm(handle, mode, n,n, d_A, n, d_X, n, d_C, n);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);

    clock_t scale_end = clock();
    double time_spent = (double)(scale_end - scale_begin) / CLOCKS_PER_SEC;
    scale_total_time[0] = scale_total_time[0] + time_spent;
}



void scale_tester_alt(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n, double* scale_total_time ){

    clock_t scale_begin = clock();
    //const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    //cublasZdgmm(handle, mode, n,n, d_A, n, d_X, n, d_C, n);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, d_B, n, d_C, n);

    clock_t scale_end = clock();
    double time_spent = (double)(scale_end - scale_begin) / CLOCKS_PER_SEC;
    scale_total_time[0] = scale_total_time[0] + time_spent;
}


void matrix_scale_kernel(cublasHandle_t &handle, cuDoubleComplex* A, cuDoubleComplex scale, int n){

	const cuDoubleComplex alf = make_cuDoubleComplex(scale.x, scale.y);
  
    const cuDoubleComplex *alpha = &alf;

	cublasZscal(handle,n*n,alpha,A,1);
}

void matrix_Scale_New(cuDoubleComplex *a, cuDoubleComplex *scaled, cuDoubleComplex scale, int n, double* scale_total_time ) {
    clock_t scale_begin = clock();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //scaled[(n * i) + j] = a[(n * i) + j] * scale;
            scaled[(n * i) + j] = cuCmul(a[(n * i) + j],scale); // Complex multiplication
        }
    }

    clock_t scale_end = clock();
    double time_spent = (double)(scale_end - scale_begin) / CLOCKS_PER_SEC;
    scale_total_time[0] = scale_total_time[0] + time_spent;
}


void matrix_Absolute_New(cuDoubleComplex *a, cuDoubleComplex *b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[(n * i) + j].x = cuCabs((a[(n * i) + j]));
            b[(n * i) + j].y = 0;
        }
    }
}

// NEED TO FIND A GPU EQUIVELANT -> NOT WORTHIT DUE TO ONE OP PER DATA ITEM AND LARGE MEMCOPY COSTS.
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


// COMPUTING OPTIMAL PARAMETERS
double ell(cuDoubleComplex *A, cuDoubleComplex *temp_new, cuDoubleComplex *d_A, double coeff, int m_val, int n, double* scale_total_time, cublasHandle_t &handle, cudaStream_t stream) {

    double norm_one, norm_two, p, alpha, output;
    memcpy(A, temp_new, n * n * sizeof(cuDoubleComplex));
    
    matrix_Absolute_New(A, temp_new, n);

    p = pow(coeff, (1.0 / (2 * m_val + 1)));
    
    cudaMemcpyAsync(d_A, temp_new, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    scale_tester(handle, d_A, NULL, d_A, make_cuDoubleComplex(p, 0), n, scale_total_time);
    cudaMemcpyAsync(temp_new, d_A, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    
    norm_one = calculate_one_norm_New_complex(A, n); // Overlap GPU & CPU WORK WITH ASYNC CALLS

    cudaDeviceSynchronize();
    norm_two = calculate_one_norm_New_complex(temp_new, n);
    
    alpha = norm_two / norm_one;

    output = fmax(ceil(log2((2 * alpha) / 2.220446049250313e-16) / (2 * m_val)), 0);

    return output;
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

int main(){

    cuDoubleComplex* A;
    int n = 1024;

    // Allocate the pinned memory:
    cudaMallocHost((void**)&A, n*n*sizeof(cuDoubleComplex));


    // Initialize the pinned memory:
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; ++j)
        {
           A[(n*i) + j].x = 0.01;
           A[(n*i) + j].y = 0.0035;
        }
    }
    

    clock_t setup_begin = clock();
    // VARIABLES TO HOLD TOTAL COMPONENT TIMES:
    double* scale_total_time = (double *) malloc(1 * sizeof(double));
    double* multiply_total_time = (double *) malloc(1 * sizeof(double));

    clock_t begin = clock();  // Begin recording expm execution

    // CUBLAS HANDLE:
    cublasHandle_t handle, handle2, handle3, handle4;
    cublasCreate(&handle);
    cublasCreate(&handle2);
    cublasCreate(&handle3);
    cublasCreate(&handle4);

    // Allocate 3 arrays on GPU
    cuDoubleComplex *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(cuDoubleComplex));
    cudaMalloc(&d_B, n * n * sizeof(cuDoubleComplex));
    cudaMalloc(&d_C, n * n * sizeof(cuDoubleComplex));


    double theta[5] = {1.495585217958292e-002, 2.539398330063230e-001,
                       9.504178996162932e-001, 2.097847961257068e+000,
                       5.371920351148152e+000};

    double error_coefficients[5] = {1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
                                    1 / 113250775606021113483283660800000000.0,
                                    1 / 113250775606021113483283660800000000.0};

    // Allocate temporary arrays to hold temporary matrices used at various stages in the calculation
    cuDoubleComplex *identity_new;
    cuDoubleComplex *U_new;
    cuDoubleComplex *V_new;
    cuDoubleComplex *temp_new;
    cuDoubleComplex *temp_2_new;
    cudaMallocHost((void**)&identity_new, n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&U_new, n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&V_new, n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&temp_new, n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&temp_2_new, n * n * sizeof(cuDoubleComplex));
    

    // Create cuda streams to enable asynchronous behaviour:
    cudaStream_t stream2, stream3, stream4;
    
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    float* A_d;
    cudaMalloc(&A_d, n*n*sizeof(cuDoubleComplex));

    double d4, d6, d8, d10, eta1, eta3, eta4, eta5, s;

    cuDoubleComplex **Tpowers;
    cudaMallocHost((void**)&Tpowers, 11 * sizeof(cuDoubleComplex*));

    for (int i = 0; i < 11; i++) {
        cudaMallocHost((void**)&Tpowers[i], n *n* sizeof(cuDoubleComplex));
    }

    clock_t setup_end = clock();
    double total_setup_time =  (double)(setup_end - setup_begin) / CLOCKS_PER_SEC;

    memcpy(Tpowers[1], A, n * n * sizeof(cuDoubleComplex));

    // To calculate Tpowers[2] & Tpowers[4] this is sequential
    matrix_Multiply(handle, Tpowers[1], Tpowers[1], Tpowers[2], d_A, d_B, d_C, n, multiply_total_time);
    matrix_Multiply(handle, Tpowers[2], Tpowers[2], Tpowers[4], d_A, d_B, d_C, n, multiply_total_time);

    // Attempt to have concurrency:
    //matrix_Multiply_With_Streams(handle, Tpowers[4], Tpowers[2], Tpowers[6], d_A, d_B, d_C, n, multiply_total_time, stream1);
    //matrix_Multiply_With_Streams(handle2, Tpowers[4], Tpowers[4], Tpowers[8], d_A, d_B, d_C, n, multiply_total_time, stream2);
    matrix_Multiply(handle, Tpowers[4], Tpowers[2], Tpowers[6], d_A, d_B, d_C, n, multiply_total_time);
    matrix_Multiply(handle, Tpowers[4], Tpowers[4], Tpowers[8], d_A, d_B, d_C, n, multiply_total_time);


    // ISSUE:
    clock_t norm_begin = clock();
    d4 = pow(calculate_one_norm_New_complex(Tpowers[4], n), (1.0 / 4));
    d6 = pow(calculate_one_norm_New_complex(Tpowers[6], n), (1.0 / 6));
    d8 = pow(calculate_one_norm_New_complex(Tpowers[8], n), (1.0 / 8));
    d10 = pow(calculate_one_norm_New_complex(Tpowers[10], n), (1.0 / 10));
    clock_t norm_end = clock();
    double total_norm_time =  (double)(norm_end - norm_begin) / CLOCKS_PER_SEC;
    eta1 = fmax(d4, d6);


    int m_val = 0;
    cublasSetStream(handle2, stream2);

    // We know that we need to calculate ell:


    if (eta1 <= theta[1] && ell(A, temp_new, d_A, error_coefficients[1], 3, n, scale_total_time, handle2, stream2) == 0.0) {
        m_val = 3;

    }
    if (eta1 <= theta[2] && ell(A, temp_new, d_A, error_coefficients[2], 5, n, scale_total_time, handle2, stream2) == 0.0) {
        m_val = 5;

    }

    eta3 = fmax(d6, d8);

    if (eta3 <= theta[3] && ell(A, temp_new, d_A, error_coefficients[3], 7, n, scale_total_time, handle2, stream2) == 0.0) {
        m_val = 7;
    }

    if (eta3 <= theta[4] && ell(A, temp_new, d_A, error_coefficients[4], 0, n, scale_total_time, handle2, stream2) == 0.0) {
        m_val = 9;
    }


    eta4 = fmax(d8, d10);
    eta5 = fmin(eta3, eta4);
    
    s = fmax(ceil(log2(eta5 / theta[4])), 0);
    
    cudaMemset(d_A, 0, n*n*sizeof(cuDoubleComplex));
    cudaMemcpy(d_A, A, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Kernel is very fast, mem copies lead to longer runtime then using CPU cblas matrix scalar
 
    cudaMemcpy(d_A, A, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    scale_tester(handle, d_A, NULL, d_C, make_cuDoubleComplex(1 / pow(2, s), 0), n, scale_total_time);
    cudaMemcpy(A, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    s = s + ell(A, temp_new, d_A, error_coefficients[4], 13, n, scale_total_time,  handle2, stream2);

    // Attempt to calculate s without leaving the GPU

    if (isinf(s)) { // Revert to old estimate
       	int exp;
        double t = frexp(calculate_one_norm_New_complex(A, n) / theta[4], &exp);
        s = s - (t == 0.5);
    } else {
        m_val = 13;
    }

    cublasSetStream(handle2, stream2);
    cublasSetStream(handle3, stream3);
    cublasSetStream(handle4, stream4);
    
    if ((int) s != 0) {	// Rescale the matrix powers array
    	      
        // Independent data, Work is streamified:
        cudaMemcpyAsync(d_A, Tpowers[1], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester(handle2, d_A, NULL, d_A, make_cuDoubleComplex(1.0 / pow(2, (s * 1)), 0), n, scale_total_time);
        cudaMemcpyAsync(Tpowers[1], d_A, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream2);
    
        cudaMemcpyAsync(d_B, Tpowers[2], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        scale_tester(handle3, d_B, NULL, d_B, make_cuDoubleComplex(1.0 / pow(2, (s * 2)), 0) , n, scale_total_time);
        cudaMemcpyAsync(Tpowers[2], d_B, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream3);

        cudaMemcpyAsync(d_C, Tpowers[4], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream4);
        scale_tester(handle4, d_C, NULL, d_C, make_cuDoubleComplex(1.0 / pow(2, (s * 4)), 0), n, scale_total_time);
        cudaMemcpyAsync(Tpowers[4], d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream4);

        matrix_Scale_New(Tpowers[6], Tpowers[6], make_cuDoubleComplex(1.0 / pow(2, (s * 6)), 0), n, scale_total_time);

        cudaDeviceSynchronize();

    }

   	// PADE APPROXIMATION:
    
    double c[15] = {1};
    
    get_pade_coefficients(c, m_val);
    
    set_Identity_New(identity_new, n);
    if (m_val == 3 || m_val == 5 || m_val == 7 || m_val == 9) {

    	int strt = sizeof(Tpowers) + 2;
        for (int k = strt; k < m_val - 1; k += 2) {
            matrix_Multiply(handle, Tpowers[2], Tpowers[k-2], Tpowers[k], d_A, d_B, d_C, n, multiply_total_time);

        }

        matrix_Scale_New(identity_new, U_new, make_cuDoubleComplex (c[1], 0), n, scale_total_time);
        matrix_Scale_New(identity_new, V_new, make_cuDoubleComplex (c[0], 0), n, scale_total_time);

        for (int j = m_val; j > n; j -= 2) {

            matrix_Scale_New(Tpowers[j - 1], temp_new, make_cuDoubleComplex(c[j + 1], 0), n, scale_total_time);
            matrixAdd_New(U_new, temp_new, U_new, n);

            matrix_Scale_New(Tpowers[j - 1], temp_new, make_cuDoubleComplex(c[j], 0), n, scale_total_time);
            matrixAdd_New(V_new, temp_new, V_new, n);
        }

         matrix_Multiply(handle, U_new, A, temp_new, d_A, d_B, d_C, n, multiply_total_time);
         memcpy(U_new, temp_new, n * n * sizeof(cuDoubleComplex));
     }


    // TODO: Look at storing Tpowers on the GPU
    // Remove extra cudaDeviceSync
    // Keep result on GPU without copy after the multiplication

    if (m_val == 13) {
        
        // Bind the CUDA streams to CUBLAS handles for asynchronous work:
        cublasSetStream(handle2, stream2);
        cublasSetStream(handle3, stream3);

        // CALCULATE U:
        cudaMemcpyAsync(d_A, Tpowers[6], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        scale_tester(handle3, d_A, NULL, d_C, make_cuDoubleComplex(c[13], 0), n, scale_total_time); // GPU

        cudaMemcpyAsync(d_B, Tpowers[4], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester_alt(handle2, d_B, d_C, d_C,  make_cuDoubleComplex(c[11], 0), n, scale_total_time);

        cudaMemcpyAsync(d_A, Tpowers[2], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        scale_tester_alt(handle3, d_A, d_C, d_C,  make_cuDoubleComplex(c[9], 0), n, scale_total_time);
        
        // SYNCHRONIZATION POINT:
        cudaMemcpy(temp_2_new, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);   
        matrix_Multiply(handle, Tpowers[6], temp_2_new, temp_new, d_A, d_B, d_C, n, multiply_total_time);

        cudaMemcpyAsync(d_A, Tpowers[6], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester_alt(handle2, d_A, d_C, d_C,  make_cuDoubleComplex(c[7], 0), n, scale_total_time);
        
        cudaMemcpyAsync(d_B, Tpowers[4], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        scale_tester_alt(handle3, d_B, d_C, d_C,  make_cuDoubleComplex(c[5], 0), n, scale_total_time);

        cudaMemcpyAsync(d_A, Tpowers[2], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester_alt(handle2, d_A, d_C, d_C,  make_cuDoubleComplex(c[3], 0), n, scale_total_time);
    
        // SYNCHRONIZATION POINT:
        set_Identity_New(identity_new, n); 
        cudaMemcpy(d_A, identity_new, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        scale_tester_alt(handle, d_A, d_C, d_C,  make_cuDoubleComplex(c[1], 0), n, scale_total_time);
        cudaMemcpyAsync(d_B, Tpowers[1], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        matrix_Multiply_Reduced(handle, d_C, d_B, d_A, n, multiply_total_time);
        cudaMemcpy(U_new, d_A, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
        // CALCULATE V:
        cudaMemcpyAsync(d_A, Tpowers[6], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_B, Tpowers[4], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        scale_tester(handle2, d_A, NULL, d_A,  make_cuDoubleComplex(c[12], 0), n, scale_total_time);
        scale_tester(handle3, d_B, NULL, d_B,  make_cuDoubleComplex(c[10], 0), n, scale_total_time);
        cudaDeviceSynchronize();
        matrix_add_Tester(handle, d_A, d_B, d_B, n);
        
        cudaMemcpyAsync(d_C, Tpowers[2], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester_alt(handle2, d_C, d_B, d_C,  make_cuDoubleComplex(c[8], 0), n, scale_total_time);

        // SYNCHRONIZATION POINT:
        cudaMemcpyAsync(d_A, Tpowers[6], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        cudaMemcpy(d_B, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        matrix_Multiply_Reduced(handle, d_A, d_B, d_C, n, multiply_total_time);
        cudaMemcpy(temp_new, d_B, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        cudaMemcpyAsync(d_A, Tpowers[6], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester_alt(handle2, d_A, d_C, d_C,  make_cuDoubleComplex(c[6], 0), n, scale_total_time);

        cudaMemcpyAsync(d_B, Tpowers[4], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream3);
        scale_tester_alt(handle3, d_B, d_C, d_C,  make_cuDoubleComplex(c[4], 0), n, scale_total_time);
        
        cudaMemcpyAsync(d_A, Tpowers[2], n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream2);
        scale_tester_alt(handle2, d_A, d_C, d_C,  make_cuDoubleComplex(c[2], 0), n, scale_total_time);
        
        // SYNCHRONIZATION POINT:
        set_Identity_New(identity_new, n);
        cudaMemcpy(d_A, identity_new, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);  
        scale_tester_alt(handle, d_A, d_C, d_C,  make_cuDoubleComplex(c[0], 0), n, scale_total_time);
        cudaMemcpy(V_new, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        
        // CALCULATE V-U:
        cudaMemcpy(d_A, U_new, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); 
        matrix_subtract_Tester(handle, d_C, d_A, d_B, n);
        cudaMemcpy(V_new, d_B, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        
        // CALCULATE F:
        scale_tester(handle, d_A, NULL, d_A,  make_cuDoubleComplex(c[0], 0), n, scale_total_time);
        cudaMemcpy(U_new, d_A, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        clock_t inverse_begin = clock();
        cudaMemcpy(d_C, identity_new, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        InverseOfMatrix_Alternative_Two(d_B, d_C, n);
        cudaMemcpy(temp_2_new, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        clock_t inverse_end = clock();
        double inverse_total_time = (double)(inverse_end - inverse_begin) / CLOCKS_PER_SEC;

        matrix_Multiply_Reduced(handle, d_C, d_A, d_B, n, multiply_total_time);
        cudaMemcpy(d_A, identity_new, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        matrix_add_Tester(handle, d_B, d_A, d_C, n);  
        cudaMemcpy(temp_new, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
 
        // SQUARE MATRIX F, S TIMES:
        clock_t square_begin = clock();
        matrix_Square_Reduced(handle, d_C, d_A, n, multiply_total_time, s);
        cudaMemcpy(temp_2_new, d_A, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        clock_t square_end = clock();
        double square_total_time = (double)(square_end - square_begin) / CLOCKS_PER_SEC;

        // PERFORMANCE OUTPUT:
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;	// End recording expm execution

        printf("******************\n");
        printf("%.15lf\n", temp_2_new[22].x);
      
        printf("\n----------------------- MATRIX OPERATIONS PERCENTAGE BREAKDOWN -----------------------\n");
        printf("\n TOTAL TIME ELAPSED: %lf seconds \n", time_spent);
        printf("\n SETUP: %lf%% \n", (total_setup_time/time_spent)*100);
        printf("\n NORM [CPU]: %lf%% \n", (total_norm_time/time_spent)*100);
        printf("\n INVERSE: %lf%% \n", (inverse_total_time/time_spent)*100);
        printf("\n SCALE: %lf%% \n", (scale_total_time[0]/time_spent)*100);
        printf("\n MULTIPLY: %lf%% \n", (multiply_total_time[0]/time_spent)*100);
        printf("\n SQUARE: %lf%% \n\n", (square_total_time/time_spent)*100);

    }

    // Free host memory
    cudaFreeHost(identity_new);
    cudaFreeHost(U_new);
    cudaFreeHost(V_new);
    cudaFreeHost(temp_new);
    cudaFreeHost(temp_2_new);
    free(scale_total_time);
    free(multiply_total_time);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// Resources
// http://www.netlib.org/utk/people/JackDongarra/PAPERS/Factor_Inversion_Million_Matrices-iccs17.pdf
// http://mathforcollege.com/nm/mws/che/04sle/mws_che_sle_spe_luinverse.pdf