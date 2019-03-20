// *** 17/03/2019 ***
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

void InverseOfMatrix_Alternative_Two(cuDoubleComplex* L, cuDoubleComplex* inverse, int n, cuDoubleComplex* b){ // Calculate matrix inverse through LU factorisation
    
    cusolverStatus_t  status; // Link to the cusolver context
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
    matrix_complex_print(L, n);
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

    printf("INVERSE ACTIVATED\n");
    cudaMemcpy(inverse, x, sizeof(cuDoubleComplex) * n * n, cudaMemcpyDeviceToHost), "Failed to copy to res!";
    matrix_complex_print(inverse, n);
    // Free device memory:
    cudaFree(dLUPivots_ALT);
    cudaFree(dLUInfo_ALT);
    cudaFree(A);
    cudaFree(x);
    cudaFree(buffer);
}


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

void matrix_Scale_New(cuDoubleComplex *a, cuDoubleComplex *scaled, cuDoubleComplex scale, int dim) {

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            scaled[(dim * i) + j] = cuCmul(a[(dim * i) + j],scale); // Complex multiplication
        }
    }
}

double ell(cuDoubleComplex *A, cuDoubleComplex *temp_new, double coeff, int m_val, int dim) {

    double norm_one, norm_two, p, alpha, output;
    memcpy(temp_new, A, dim * dim* sizeof(cuDoubleComplex));
    
    matrix_Absolute_New(temp_new, temp_new, dim);

    p = pow(coeff, (1.0 / (2 * m_val + 1)));
    matrix_Scale_New(temp_new, temp_new, make_cuDoubleComplex(p, 0), dim);
    norm_one = calculate_one_norm_New_complex(temp_new, dim); // Overlap GPU & CPU WORK WITH ASYNC CALLS
    norm_two = calculate_one_norm_New_complex(A, dim);
    
    alpha = norm_one / norm_two;
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


// Note that currently we are having to allocate alot of memory:
// Time to perfrom a scale is fairly insiginficant
void batch_scale(cublasHandle_t handle, cuDoubleComplex** d_A, cuDoubleComplex* scalars, int n, int batch_count){
    // Need array of matrices
    // Need array of scalars
    // Need array of results
  
    //const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    //const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;

    cudaStream_t *streams = (cudaStream_t *) malloc(batch_count*sizeof(cudaStream_t));
    // for(int i=0; i<batch_count; i++)
    //   cudaStreamCreate(&streams[i]);

    for (int i = 0; i < batch_count; i++)
    {
      //cublasSetStream(handle, streams[i]);
    // Switch through the streams
   
    cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &scalars[i], d_A[i], n, beta, NULL, n, d_A[i], n);
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


double* batch_ell(cublasHandle_t handle, cuDoubleComplex** A, cuDoubleComplex **d_A, cuDoubleComplex** o_A,
 bool* to_compute, double coeff, int m_val, int dim, int batch_count, double* output, bool add) {

    // need to allocate device space for these variables:
    double* norm_one;
    double* norm_two;
    double* alpha;
    //double* output;
    cuDoubleComplex* scalars = (cuDoubleComplex*) malloc(batch_count* sizeof(cuDoubleComplex));
    norm_one = (double*) malloc(batch_count*sizeof(double));
    norm_two = (double*) malloc(batch_count*sizeof(double));
    alpha = (double*) malloc(batch_count*sizeof(double));
    //output = (double*) malloc(batch_count*sizeof(double));

    
    
    printf("%lf", pow(coeff, (1.0 / (2 * m_val + 1))));
    for (int i = 0; i < batch_count; i++)
    {
      scalars[i] = make_cuDoubleComplex(pow(coeff, (1.0 / (2 * m_val + 1))) , 0); // Create scalar for each batch item
    }
    

 
    // Attempt to perform a batch scale:
  batch_scale(handle, d_A, scalars, dim, batch_count);

  // Copy to temp arrays:
  for(int i=0; i<batch_count; i++) {
    cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), d_A[i], dim, o_A[i], dim);
    }



  printf("SCALE HAS FINISHED: \n");
   matrix_complex_print(o_A[0], dim);
   
   for (int i = 0; i < batch_count; i++)
   {
    //matrix_Absolute_New(A[i], temp_new, n);  // Need kernel to do this on the device
    norm_one[i] = calculate_one_norm_New_complex(o_A[i], dim);
    norm_two[i] = calculate_one_norm_New_complex(o_A[i], dim);
    alpha[i] = norm_two[i] / norm_one[i];
    if(add == true){
    output[i] = fmax(ceil(log2((2 * alpha[i]) / 2.220446049250313e-16) / (2 * m_val)), 0);
    } else{
      output[i] = output[i]+ fmax(ceil(log2((2 * alpha[i]) / 2.220446049250313e-16) / (2 * m_val)), 0);
    }
  }


    return output;
}

 
int main(int argc, char* argv[])
{

	int dim = 2;
	int batch_count = 3;
 
    // Allocate host memory of A,B and C matrices and temp arrays
    cuDoubleComplex **A, **B, **C, **temp_A, **temp_B, **temp_C;
    A = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    B = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    C = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    temp_A = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*)); // Temporary arrays used in exchange stages.
    temp_B = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*)); // Temporary arrays used in exchange stages.
    temp_C = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*)); // Temporary arrays used in exchange stages.
    
    for(int i=0; i<batch_count; i++) {
        A[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        B[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        C[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        temp_A[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        temp_B[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        temp_C[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
    }
 

    // Create host pointer array to device matrix storage
    cuDoubleComplex **d_A, **d_B, **d_C, **h_d_A, **h_d_B, **h_d_C;
    h_d_A = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_B = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_C = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
 
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
 
    // INITIALIZE BATCHES WITH DUMMY DATA:
	for (int i = 0; i< batch_count; i++) {
 		for(int j = 0; j< dim; j++){
 			for (int k = 0; k < dim; k++)
 			{
 				A[i][(dim*j) + k] = make_cuDoubleComplex(1, 1);
 				B[i][(dim*j) + k] = make_cuDoubleComplex(1, 1);
 				C[i][(dim*j) + k] = make_cuDoubleComplex(0, 0);

        temp_A[i][(dim*j) + k] = make_cuDoubleComplex(0, 0); 
        temp_B[i][(dim*j) + k] = make_cuDoubleComplex(0, 0); 
        temp_C[i][(dim*j) + k] = make_cuDoubleComplex(0, 0); 
 			}
 		}
 	}


  matrix_complex_print(A[0], dim);
  //exit(0);
 
    // Create cublas instance
    cublasHandle_t handle;
    cublasCreate(&handle);
 
    // Copy host matrices to device memory
    for(int i=0; i<batch_count; i++) {
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), A[i], dim, h_d_A[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), B[i], dim, h_d_B[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), C[i], dim, h_d_C[i], dim);
    }


	// Need to store Tpowers arrays for each matrix
    cuDoubleComplex **T1, **T2, **T4, **T6, **T8, **T10;
    T1 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    T2 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    T4 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    T6 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    T8 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    T10 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));

    for(int i=0; i<batch_count; i++) {
        T1[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        T2[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        T4[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        T6[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        T8[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
        T10[i] = (cuDoubleComplex*)malloc(dim*dim*sizeof(cuDoubleComplex));
    }


        // Create host pointer array to device matrix storage
    cuDoubleComplex **d_T1, **d_T2, **d_T4, **d_T6, **d_T8, **d_T10, **h_d_T1, **h_d_T2, **h_d_T4, **h_d_T6, **h_d_T8, **h_d_T10;
    h_d_T1 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T2 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T4 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T6 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T8 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
    h_d_T10 = (cuDoubleComplex**)malloc(batch_count*sizeof(cuDoubleComplex*));
 
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



    // Allocate device memory of A,B and C matrices
   
    // Copy host matrices to device memory
    for(int i=0; i<batch_count; i++) {
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T1[i], dim, h_d_T1[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T2[i], dim, h_d_T2[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T4[i], dim, h_d_T4[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T6[i], dim, h_d_T6[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T8[i], dim, h_d_T8[i], dim);
        cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T10[i], dim, h_d_T10[i], dim);
    }

    // Alpha and beta coeficcients set for zgemm:
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;


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

    // Calculate T8:

    cudaDeviceSynchronize();

    cublasZgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       dim, dim, dim,
                       alpha,
                       (const cuDoubleComplex**)d_T4, dim,
                       (const cuDoubleComplex**)d_T4, dim,
                       beta,
                       d_T8, dim,
                       batch_count);
 
    cudaDeviceSynchronize();

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
    
    // Copy each device result to the host
    for(int i=0; i<batch_count; i++) {
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_T2[i], dim, T2[i], dim);
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_T4[i], dim, T4[i], dim);
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_T6[i], dim, T6[i], dim);
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_T8[i], dim, T8[i], dim);
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_T10[i], dim, T10[i], dim);
      memcpy(T1[i], A[i], dim*dim*sizeof(cuDoubleComplex));
    }

  // PRINT A SAMPLE
  printf("HERE:\n");
	matrix_complex_print(A[1], dim);
	printf("\n");
	matrix_complex_print(A[1], dim);
	printf("T6 IS: \n");
	matrix_complex_print(T6[1], dim);
  
  

  // [PART 2] CALCULATE (d4,d6,d8, d10)
  double* d4 = (double*) malloc(batch_count*sizeof(double));
  double* d6 = (double*) malloc(batch_count*sizeof(double));
  double* d8 = (double*) malloc(batch_count*sizeof(double));
  double* d10 = (double*) malloc(batch_count*sizeof(double));

  for (int i = 0; i < batch_count; i++) // Calculated on the host currently
  {
    cudaMemcpy(temp_C[i], h_d_T4[i], (dim*dim)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    d4[i] = pow(calculate_one_norm_New_complex(temp_C[i], dim), (1.0 / 4));
    cudaMemcpy(temp_C[i], h_d_T6[i], (dim*dim)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    d6[i] = pow(calculate_one_norm_New_complex(temp_C[i], dim), (1.0 / 6));
    cudaMemcpy(temp_C[i], h_d_T8[i], (dim*dim)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    d8[i] = pow(calculate_one_norm_New_complex(temp_C[i], dim), (1.0 / 8));
    cudaMemcpy(temp_C[i], h_d_T10[i], (dim*dim)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    d10[i] = pow(calculate_one_norm_New_complex(temp_C[i], dim), (1.0 / 10));
    
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
    if(eta1[i] <= theta[1] && ell(A[i], temp_A[i], error_coefficients[1], 3, dim) == 0); // Check for m_val = 3
      m_val[i] = 3;
    if(eta1[i] <= theta[2] && ell(A[i], temp_A[i], error_coefficients[2], 5, dim) == 0); // Check for m_val = 5
      m_val[i] = 5;
    if(eta3[i] <= theta[3] && ell(A[i], temp_A[i], error_coefficients[3], 7, dim) == 0); // Check for m_val = 7
      m_val[i] = 7;
    if(eta3[i] <= theta[4] && ell(A[i], temp_A[i], error_coefficients[4], 9, dim) == 0); // Check for m_val = 9
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
  matrix_Scale_New(A[i], temp_A[i], make_cuDoubleComplex(1/pow(2, s[i]), 0), dim);
  s[i] = s[i] + ell(A[i], temp_A[i], error_coefficients[4], 13, dim);
  if(s[i] > max)
    max = s[i];
}




// [PART 6] S CHECK AND M CHECK

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
    printf("HAVE TO RESCALE\n");
    matrix_Scale_New(T1[i],T1[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 1)), 0), dim);
    matrix_Scale_New(T2[i],T2[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 2)), 0), dim);
    matrix_Scale_New(T4[i],T4[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 4)), 0), dim);
    matrix_Scale_New(T6[i],T6[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 6)), 0), dim);
  }
}


// [PART 7.5] GET THE PADE COEFFICIENTS FOR EACH BATCH

double** c = (double**) malloc(batch_count*sizeof(double*)); 

for (int i = 0; i < batch_count; i++)
{ 
  c[i] = (double*) malloc(15*sizeof(double));
  get_pade_coefficients(c[i], m_val[i]);
}




//if (m_val == 13) // Will need to seperate matrices that are not satisfied for batching to commence

  // [PART 8] CALCULATE U

  for (int i = 0; i < batch_count; i++)
  {
    memset(temp_A[i], 0, dim*dim*sizeof(cuDoubleComplex));
    matrix_Scale_New(T6[i],temp_A[i], make_cuDoubleComplex(c[i][13], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    matrix_Scale_New(T4[i],temp_A[i], make_cuDoubleComplex(c[i][11], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    matrix_Scale_New(T2[i],temp_A[i], make_cuDoubleComplex(c[i][9], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
  }

// Perform batch matrix multiplication

  // Copy altered Tpowers to device
    for(int i=0; i<batch_count; i++) {
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_B[i], dim, h_d_B[i], dim);
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T6[i], dim, h_d_T6[i], dim);
    }


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
    
      // Copy each device result to the host
  for(int i=0; i<batch_count; i++) {
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, temp_C[i], dim);
    }

    matrix_complex_print(temp_C[0], dim);
    


  
  for (int i = 0; i < batch_count; i++)
  {
    memset(temp_A[i], 0, dim*dim*sizeof(cuDoubleComplex));
    
    matrix_Scale_New(T6[i],temp_A[i],  make_cuDoubleComplex(c[i][7], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    
    matrix_Scale_New(T4[i],temp_A[i], make_cuDoubleComplex(c[i][5], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);

    matrix_Scale_New(T2[i],temp_A[i], make_cuDoubleComplex(c[i][3], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    
    memset(temp_A[i], 0, dim*dim*sizeof(cuDoubleComplex));
    
    set_Identity_New(temp_A[i], dim);
    
    matrix_Scale_New(temp_A[i],temp_C[i], make_cuDoubleComplex(c[i][1], 0), dim);
    matrixAdd_New(temp_C[i], temp_B[i], temp_B[i], dim);

    // COPY THE SCALED TPOWERS TO THE DEVICE
    cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_B[i], dim, h_d_B[i], dim);

  }


  // Copy each device result to the host
  for(int i=0; i<batch_count; i++) {
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, temp_C[i], dim);
      matrixAdd_New(temp_C[i], temp_B[i], temp_B[i], dim);
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_B[i], dim, h_d_B[i], dim);
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T1[i], dim, h_d_A[i], dim);
    }


  cublasZgemmBatched(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      dim, dim, dim,
                      alpha,
                      (const cuDoubleComplex**)d_B, dim,
                      (const cuDoubleComplex**)d_A, dim,
                      beta,
                      d_C, dim,
                      batch_count);

  cudaDeviceSynchronize();
    
    // Copy each device result to the host
  for(int i=0; i<batch_count; i++) {
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, temp_C[i], dim);
    }

  // PRINT SAMPLE (U)
  printf("U IS: \n");
  matrix_complex_print(temp_C[1], dim);
 
  // [PART 9] CALCULATE V
  for (int i = 0; i < batch_count; i++)
  {
    memset(temp_A[i], 0, dim*dim*sizeof(cuDoubleComplex));
    matrix_Scale_New(T6[i],temp_A[i],  make_cuDoubleComplex(c[i][12], 0), dim);
    matrix_Scale_New(T4[i],temp_B[i], make_cuDoubleComplex(c[i][10], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    matrix_Scale_New(T2[i],temp_B[i], make_cuDoubleComplex(c[i][8], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
  }

  // Perform batch matrix multiplication

  // Copy altered Tpowers to device
    for(int i=0; i<batch_count; i++) {
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_B[i], dim, h_d_B[i], dim);
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), T6[i], dim, h_d_T6[i], dim);
    }


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



for (int i = 0; i < batch_count; i++)
  {
    memset(temp_A[i], 0, dim*dim*sizeof(cuDoubleComplex));
    matrix_Scale_New(T6[i],temp_A[i],  make_cuDoubleComplex(c[i][6], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    matrix_Scale_New(T4[i],temp_A[i], make_cuDoubleComplex(c[i][4], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    matrix_Scale_New(T2[i],temp_A[i], make_cuDoubleComplex(c[i][2], 0), dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
    memset(temp_C[i], 0, dim*dim*sizeof(cuDoubleComplex));
    set_Identity_New(temp_C[i], dim);
    matrix_Scale_New(temp_C[i],temp_C[i], make_cuDoubleComplex(c[i][0], 0), dim);
    matrixAdd_New(temp_B[i], temp_C[i], temp_B[i], dim);
    
    // Copy each device result to the host
    cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_A[i], dim, temp_A[i], dim);
    matrixAdd_New(temp_A[i], temp_B[i], temp_B[i], dim);
  }


// V IS STORED IN temp_B array
// U IS STORED IN temp_C array

for (int i = 0; i < batch_count; i++)
  { 

   //cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_A[i], dim, temp_A[i], dim); // V entry
   cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, temp_C[i], dim); // U entry
   matrix_Subtract_New(temp_B[i], temp_C[i], temp_B[i], dim);

   memset(temp_A[i], 0, sizeof(cuDoubleComplex)*dim*dim);
   set_Identity_New(temp_A[i], dim);

   memset(temp_C[i], 0, dim*dim*sizeof(cuDoubleComplex));
   InverseOfMatrix_Alternative_Two(temp_B[i], temp_C[i], dim, temp_A[i]); // Calulate each inverse
 }


    // SCALE MATRIX U BY 2
    for(int i=0; i<batch_count; i++) {
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, temp_A[i], dim);
      matrix_Scale_New(temp_A[i],temp_A[i], make_cuDoubleComplex(2, 0), dim);
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_A[i], dim, h_d_A[i], dim);
      cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_C[i], dim, h_d_C[i], dim);
    }


  // [PART 11] CALCULATE F
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
  
  // Copy each device result to the host
  for(int i=0; i<batch_count; i++) {
      memset(temp_C[i], 0, sizeof(cuDoubleComplex)*dim*dim);
      set_Identity_New(temp_C[i], dim);
      cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_B[i], dim, temp_B[i], dim);
      matrixAdd_New(temp_B[i], temp_C[i], temp_B[i], dim);
    }

  // PRINT SAMPLE (F)
  printf("F IS: \n");
  matrix_complex_print(temp_B[1], dim);
  for(int i=0; i<batch_count; i++) {
    cublasSetMatrix(dim, dim, sizeof(cuDoubleComplex), temp_B[i], dim, h_d_B[i], dim);
  }

  // SQUARING PHASE:
  for (int k = 0; k < max; k++) {

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
    if (k<=s[i]){
    cudaMemcpy(h_d_B[i], h_d_C[i], dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
  }
    }
  }

  // Copy each device result to the host
  for(int i=0; i<batch_count; i++) {
    cublasGetMatrix(dim, dim, sizeof(cuDoubleComplex), h_d_C[i], dim, temp_C[i], dim);
  }

  printf("EXPM RESULT IS: \n");
  matrix_complex_print(temp_C[1], dim);
   // Clean up resources
 
    for(int i=0; i<batch_count; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        cudaFree(h_d_A[i]);
        cudaFree(h_d_B[i]);
        cudaFree(h_d_C[i]);
    }
 
    free(A);
    free(B);
    free(C);
    free(T2);
    free(T4);
    free(T6);
    free(h_d_A);
    free(h_d_B);
    free(h_d_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
 
    return 0;
}