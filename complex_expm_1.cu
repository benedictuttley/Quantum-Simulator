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

// *** Matrix exponential program acting on matrices of type : [Double Complex] - 01/03/2019 ***


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

void matrix_Square(cublasHandle_t &handle, cuDoubleComplex *A, cuDoubleComplex *C, cuDoubleComplex *d_A, cuDoubleComplex *d_C, int n, double* multiply_total_time, int s){
	clock_t multiply_begin = clock();

	const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;
	
	cudaMemcpy(d_A, A, n * n * sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);	// Copy first operand to the device (only one copy is needed for the squaring phase)

	for (int k = 0; k < s; k++) {
		cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_A, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication
        cudaMemcpy(d_A, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(C, d_C, n * n * sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);	// Copy product back to host
 	
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

    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_B, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication

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
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_B, n, d_A, n, beta, d_C, n); // Perform the cublas matrix multiplication

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


void InverseOfMatrix_Alternative_Two(cuDoubleComplex* L, cuDoubleComplex* inverse, int n, cuDoubleComplex* b){ // Calculate matrix inverse through LU factorisation
    
    cusolverStatus_t  status;	// Link to the cusolver context
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


// *** CURRENT WORK ***
void scale_tester(cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, const cuDoubleComplex alf, int n){

    //const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    //cublasZdgmm(handle, mode, n,n, d_A, n, d_X, n, d_C, n);
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, alpha, d_A, n, beta, NULL, n, d_C, n);
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
double ell(cuDoubleComplex *A, cuDoubleComplex *temp_new, double coeff, int m_val, int n, double* scale_total_time) {

    double norm_one, norm_two, p, alpha, output;
    memcpy(A, temp_new, n * n * sizeof(cuDoubleComplex));
    
    matrix_Absolute_New(A, temp_new, n);

    p = pow(coeff, (1.0 / (2 * m_val + 1)));
    
    matrix_Scale_New(temp_new, temp_new, make_cuDoubleComplex(p, 0), n, scale_total_time);
    
    norm_one = calculate_one_norm_New_complex(A, n);
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
    int n = 4000;

    // Allocate the pinned memory:
    cudaMallocHost((void**)&A, n*n*sizeof(cuDoubleComplex));


    // Initialize the pinned memory:
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; ++j)
        {
           A[(n*i) + j].x = (n*i) + j;
           A[(n*i) + j].y = (n*i) + j;
        }
    }
    

    // VARIABLES TO HOLD TOTAL COMPONENT TIMES:
    double* scale_total_time = (double *) malloc(1 * sizeof(double));
    double* multiply_total_time = (double *) malloc(1 * sizeof(double));

    clock_t begin = clock();  // Begin recording expm execution

    // CUBLAS HANDLE:
    cublasHandle_t handle, handle2;
    cublasCreate(&handle);
    cublasCreate(&handle2);

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

    //identity_new = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&identity_new, n * n * sizeof(cuDoubleComplex));
    
    //U_new = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&U_new, n * n * sizeof(cuDoubleComplex));

    //V_new = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&V_new, n * n * sizeof(cuDoubleComplex));

    //temp_new = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&temp_new, n * n * sizeof(cuDoubleComplex));

    //temp_2_new = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&temp_2_new, n * n * sizeof(cuDoubleComplex));
    // LARGE MEMORY ALLOCATION PHASE


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    float* A_d;
    cudaMalloc(&A_d, n*n*sizeof(cuDoubleComplex));

    double d4, d6, d8, d10, eta1, eta3, eta4, eta5, s;

    ///////////////////////////////////////////////////////////////////////////////////////////////



    cuDoubleComplex **Tpowers;
    // = (cuDoubleComplex **) malloc(11 * sizeof(cuDoubleComplex *));
    cudaMallocHost((void**)&Tpowers, 11 * sizeof(cuDoubleComplex*));

    for (int i = 0; i < 11; i++) {
       // Tpowers[i] = (cuDoubleComplex *) malloc(n * n * sizeof(cuDoubleComplex));
        cudaMallocHost((void**)&Tpowers[i], n *n* sizeof(cuDoubleComplex));

    }

    memcpy(Tpowers[1], A, n * n * sizeof(cuDoubleComplex));

    // To calculate Tpowers[2] & Tpowers[4] this is sequential
    matrix_Multiply(handle, Tpowers[1], Tpowers[1], Tpowers[2], d_A, d_B, d_C, n, multiply_total_time);
     
    matrix_Multiply(handle, Tpowers[2], Tpowers[2], Tpowers[4], d_A, d_B, d_C, n, multiply_total_time);

    // Attemp to have concurrency:

    matrix_Multiply_With_Streams(handle, Tpowers[4], Tpowers[2], Tpowers[6], d_A, d_B, d_C, n, multiply_total_time, stream1);

    matrix_Multiply_With_Streams(handle2, Tpowers[4], Tpowers[4], Tpowers[8], d_A, d_B, d_C, n, multiply_total_time, stream2);
    
   


    //matrix_Multiply_With_Streams(handle2, Tpowers[4], Tpowers[4], Tpowers[8], d_A, d_B, d_C, n, multiply_total_time, stream2);
    ///////////////////////////////////////////////////////////////////////////////////////////////


    //cuDoubleComplex **Tpowers = get_Matrix_Powers_New(A, d_A, d_B, d_C, handle, n, multiply_total_time);
    
  
    // We know we need to calculate d4, d6 d8, d10 - they operate on independent data:



    cudaStreamSynchronize(stream2);
    //matrix_complex_print(Tpowers[8], n);
    printf("-----------> %lf\n", Tpowers[8][0].x );
    


    // ISSUE:
    clock_t norm_begin = clock();
    d4 = pow(calculate_one_norm_New_complex(Tpowers[4], n), (1.0 / 4));
    d6 = pow(calculate_one_norm_New_complex(Tpowers[6], n), (1.0 / 6));
    d8 = pow(calculate_one_norm_New_complex(Tpowers[8], n), (1.0 / 8));
    d10 = pow(calculate_one_norm_New_complex(Tpowers[10], n), (1.0 / 10));
    clock_t norm_end = clock();
    double total_norm_time =  (double)(norm_end - norm_begin) / CLOCKS_PER_SEC;
    printf("TOTAL NORM TIME IS: %lf\n", total_norm_time);
    eta1 = fmax(d4, d6);


    int m_val = 0;

    // We know that we need to calculate ell:


    if (eta1 <= theta[1] && ell(A, temp_new, error_coefficients[1], 3, n, scale_total_time) == 0.0) {
        m_val = 3;

    }
    if (eta1 <= theta[2] && ell(A, temp_new, error_coefficients[2], 5, n, scale_total_time) == 0.0) {
        m_val = 5;

    }


    eta3 = fmax(d6, d8);

    if (eta3 <= theta[3] && ell(A, temp_new, error_coefficients[3], 7, n, scale_total_time) == 0.0) {
        m_val = 7;
    }

    if (eta3 <= theta[4] && ell(A, temp_new, error_coefficients[4], 0, n, scale_total_time) == 0.0) {
        m_val = 9;
    }


    

    eta4 = fmax(d8, d10);
    eta5 = fmin(eta3, eta4);
    
    s = fmax(ceil(log2(eta5 / theta[4])), 0);



    
    cudaMemset(d_A, 0, n*n*sizeof(cuDoubleComplex));
    cudaMemcpy(d_A, A, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    


    // Kernel is very fast, mem copies lead to longer runtime then using CPU blas matrix scalar
    clock_t scale_begin = clock();
    cudaMemcpy(d_A, A, n*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    scale_tester(handle, d_A, NULL, d_C, make_cuDoubleComplex(1 / pow(2, s), 0), n); // GPU
    cudaMemcpy(A, d_C, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    //matrix_Scale_New(A, A, make_cuDoubleComplex(1 / pow(2, s), 0), n, scale_total_time); // CPU
    clock_t scale_end = clock();
    double scale_time = (double)(scale_end - scale_begin) / CLOCKS_PER_SEC;
    
    printf("TIME TO SCALE: %lf\n", scale_time);
    
    //cublasHandle_t handle, cuDoubleComplex* d_A, cuDoubleComplex* d_B, cuDoubleComplex* d_C, int n
    
    printf("the scale is: \n ");
    //matrix_complex_print(A, n);
    exit(0);
    

    s = s + ell(A, temp_new, error_coefficients[4], 13, n, scale_total_time);

    if (isinf(s)) { // Revert to old estimate
       	int exp;
        double t = frexp(calculate_one_norm_New_complex(A, n) / theta[4], &exp);
        s = s - (t == 0.5);
    } else {
        m_val = 13;
    }

    printf("s is: %lf", s);
    if ((int) s != 0) {	// Rescale the matrix powers array
        printf("HAVE TO RESCALE!\n");
    	
    	cuDoubleComplex multiplier = make_cuDoubleComplex(0, 0);

      	multiplier.x = 1.0 / pow(2, (s * 1));
        matrix_Scale_New(Tpowers[1], Tpowers[1], multiplier, n, scale_total_time);
    
        multiplier.x = 1.0 / pow(2, (s * 2));
        matrix_Scale_New(Tpowers[2], Tpowers[2], multiplier, n, scale_total_time);

        multiplier.x = 1.0 / pow(2, (s * 4));
        matrix_Scale_New(Tpowers[4], Tpowers[4], multiplier, n, scale_total_time);

        multiplier.x = 1.0 / pow(2, (s * 6));
        matrix_Scale_New(Tpowers[6], Tpowers[6], multiplier, n, scale_total_time);
     
    }
printf("HERE\n");


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


    if (m_val == 13) {

        // ------------------ TODO --------------
        // Compute more of the below on the GPU
        // Asses any speed-up for scale & one-norm
        // Attempt to overlap

        // CALCULATE U:

        // overlap: scale 6, scale 4, scale 2, add on gpu, scale 6, scale 4, scale 2, add on gpu

        matrix_Scale_New(Tpowers[6], temp_new, make_cuDoubleComplex(c[13], 0), n, scale_total_time);
   
        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));

        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);
    
        matrix_Scale_New(Tpowers[4], temp_new, make_cuDoubleComplex(c[11], 0), n, scale_total_time);
        
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[2], temp_new, make_cuDoubleComplex(c[9], 0), n, scale_total_time);

        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(cuDoubleComplex));
 
        matrix_Multiply(handle, Tpowers[6], temp_2_new, temp_new, d_A, d_B, d_C, n, multiply_total_time);  

        matrix_Scale_New(Tpowers[6], temp_2_new, make_cuDoubleComplex(c[7], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        matrix_Scale_New(Tpowers[4], temp_2_new, make_cuDoubleComplex(c[5], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        matrix_Scale_New(Tpowers[2], temp_2_new, make_cuDoubleComplex(c[3], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        set_Identity_New(identity_new, n);
     
        matrix_Scale_New(identity_new, temp_2_new, make_cuDoubleComplex(c[1], 0), n, scale_total_time);

        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(U_new, 0, n * n * sizeof(cuDoubleComplex));	// IS THIS NEEDED?

        matrix_Multiply(handle, temp_new, Tpowers[1], U_new, d_A, d_B, d_C, n, multiply_total_time);



        
        // CALCULATE V:

        memset(temp_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[6], temp_new, make_cuDoubleComplex(c[12], 0), n, scale_total_time);

        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[4], temp_2_new, make_cuDoubleComplex(c[10], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[2], temp_new, make_cuDoubleComplex(c[8], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(cuDoubleComplex));

        matrix_Multiply(handle, temp_2_new, Tpowers[6], temp_new, d_A, d_B, d_C, n, multiply_total_time);

        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[6], temp_2_new, make_cuDoubleComplex(c[6], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[4], temp_2_new, make_cuDoubleComplex(c[4], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));
        matrix_Scale_New(Tpowers[2], temp_2_new, make_cuDoubleComplex(c[2], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));
        set_Identity_New(identity_new, n);
        matrix_Scale_New(identity_new, temp_2_new, make_cuDoubleComplex(c[0], 0), n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, V_new, n);

        



        // Must wait for above (synchronization point):
        // CALCULATE V-U

        matrix_Subtract_New(V_new, U_new, V_new, n);
       
        matrix_Scale_New(U_new, U_new, make_cuDoubleComplex(2,0), n, scale_total_time);
        //matrix_scale_kernel(handle, U_new, make_cuDoubleComplex(2,0), n);
        memset(temp_2_new, 0, n * n * sizeof(cuDoubleComplex));

        clock_t inverse_begin = clock();
    
        InverseOfMatrix_Alternative_Two(V_new, temp_2_new, n, identity_new);
       
        clock_t inverse_end = clock();
        double inverse_total_time = (double)(inverse_end - inverse_begin) / CLOCKS_PER_SEC;

        memset(temp_new, 0, n * n * sizeof(cuDoubleComplex));

        matrix_Multiply(handle, temp_2_new, U_new, temp_new, d_A, d_B, d_C, n, multiply_total_time);

       	// CALCULATE F:
        matrixAdd_New(temp_new, identity_new, temp_new, n);
        
        // SQUARE THE MATRIX S TIMES - Sequential:
        matrix_Square(handle, temp_new, temp_2_new, d_A, d_C, n, multiply_total_time, s);

        // PERFORMANCE OUTPUT:
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;	// End recording expm execution
        
        printf("\n----------------------- MATRIX OPERATIONS PERCENTAGE BREAKDOWN -----------------------\n");
        printf("\n TOTAL TIME ELAPSED: %lf seconds \n", time_spent);
        printf("\n INVERSE: %lf%% \n", (inverse_total_time/time_spent)*100);
        printf("\n SCALE: %lf%% \n", (scale_total_time[0]/time_spent)*100);
        printf("\n MULTIPLY: %lf%% \n\n", (multiply_total_time[0]/time_spent)*100);

     }


     printf("----> %lf\n", temp_2_new[0].x );
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