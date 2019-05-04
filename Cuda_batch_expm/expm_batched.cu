// Batched cuEXPM:

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <cublas_v2.h> 
#include <cuComplex.h>  // cuComplex contains cuDoubleComplex datatype used to represent complex numbers of double precision 

#define BLOCK_SIZE 32

__global__ void ell_kernel(cuDoubleComplex* A, cuDoubleComplex* B, int dim, int* m_val, int k, int val){
    
    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    extern __device__ double s_nrm_one[]; 
    extern __device__ double s_nrm_two[]; 

    if(tid_x < dim) {
        double sum_nrm_one = 0; // Private variable to hold column sum
        double sum_nrm_two = 0; // Private variable to hold column sum

    for (int i = 0; i < dim; i++) {
        sum_nrm_one += cuCabs(A[(i*dim) + tid_x]);
        sum_nrm_two += cuCabs(B[(i*dim) + tid_x]);
    }
        s_nrm_one[tid_x] = sum_nrm_one;
        s_nrm_two[tid_x] = sum_nrm_two;
    }
    
    __syncthreads(); 

    if(tid_x == 1) {
        double second_norm = 0;
        double first_norm = 0;

        for (int i = 0; i < dim; i++) {
            if(first_norm < s_nrm_one[i])
                first_norm = s_nrm_one[i];
        }

        for (int i = 0; i < dim; i++) {   
            if(second_norm < s_nrm_two[i])
                second_norm = s_nrm_two[i];
        }

        double alpha = second_norm/first_norm;
        double output = ceil(log2((2 * alpha) / 2.220446049250313e-16) / (2 * val));

        if(output <= 0.0)
            m_val[k] = 0.0;

        if(val == 13)
             m_val[k] = output;
     }
}

__global__ void identity_kernel(cuDoubleComplex* identity, int dim){
    
    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y; 

    if(tid_x < dim && tid_y < dim) { 
        identity[(dim*tid_x) + tid_y].y = 0;                
        
        if(tid_x == tid_y)
            identity[(dim*tid_x) + tid_y].x = 1;
        else
            identity[(dim*tid_x) + tid_y].x = 0;
    } 
}


__global__ void absolute_kernel(cuDoubleComplex* A, int dim){

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y*blockIdx.y + threadIdx.y;

    if(tid_x < dim && tid_y < dim) { 
        A[(dim*tid_y) + tid_x].x = cuCabs((A[(dim*tid_y) + tid_x]));
        A[(dim*tid_y) + tid_x].y = 0;
    }
}


__global__ void get_one_norm( cuDoubleComplex* A, double* res, int k, int dim){

    const int tid_x = blockDim.x*blockIdx.x + threadIdx.x;
    extern __device__ double s[];  
    double sum = 0; 

    res[k] = 0;
    
    if(tid_x < dim){ 
        for (int i = 0; i < dim; ++i){
            sum += cuCabs(A[(i*dim) + tid_x]);
        }
        s[tid_x] = sum;
    }
    
    __syncthreads(); 

    if (tid_x == 0) {
        for (int i = 0; i < dim; i++){
            if(res[k] < s[i])
                res[k] = s[i];
        }
    }
}


void matrix_complex_print(cuDoubleComplex* A, int network_size){
    for (int j = 0; j < network_size; j++){
        printf("[");
        for (int k = 0; k < network_size; k++){
            printf(" %.25lf ", A[(j*network_size) + k].x );
            printf("+");
            printf(" %.25lfi ", A[(j*network_size) + k].y );
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


void Inverse_Batched(cublasHandle_t handle, cuDoubleComplex** d_A, cuDoubleComplex** inverse, int dim, int batch_count){

    cublasHandle_t my_handle;
    int* PIVOTS;
    int* INFO;

    // Create a cublas status object
    cublasStatus_t status;
    status = cublasCreate(&my_handle);

    cudaMalloc(&PIVOTS, dim * batch_count* sizeof(int)), "Failed to allocate pivots!";
    cudaMalloc(&INFO, batch_count*sizeof(int)), "Failed to allocate info!";

    // Perform the LU factorization for each matrix in the batch:
    status = cublasZgetrfBatched(handle, dim, d_A, dim, PIVOTS, INFO, batch_count);
    cudaDeviceSynchronize();
    
    if(status!=CUBLAS_STATUS_SUCCESS)
        printf("ERROR!!\n");

    int INFOh[batch_count];
    cudaMemcpy(INFOh, INFO, batch_count*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_count; i++){
        if(INFOh[i] != 0) 
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
    }

    status = cublasZgetriBatched(handle, dim,  (const cuDoubleComplex**)d_A, dim, (const int*) PIVOTS, inverse, dim, INFO, batch_count);
    cudaDeviceSynchronize();
        
    if(status!=CUBLAS_STATUS_SUCCESS)
        printf("ERROR!!\n");

    cudaMemcpy(INFOh, INFO, batch_count*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_count; i++){
        if(INFOh[i] != 0) 
            fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
    }
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



extern "C" void expm_initialization(void* input, cuDoubleComplex* output, int dim, int batch_count) {

    // Note: Commented code is for testing on dummy matrices without linkage with Python
    // Size of matrix input:
	// int dim = 4;
    // Number of matrices in the batch:
	// int batch_count = 10;
 
    // Allocate host memory for input array:
    cuDoubleComplex *A = (cuDoubleComplex*)malloc(batch_count*sizeof(cuDoubleComplex*));
    A = (cuDoubleComplex*)input;
 
    // INITIALIZE BATCHES WITH DUMMY DATA:
    // for (int i = 0; i< batch_count; i++) {
    //     for(int j = 0; j< dim; j++){
    //         for (int k = 0; k < dim; k++)
    //         {
    //             A[i][(dim*j) + k] = make_cuDoubleComplex(i,i);
    //         }
    //     }
    // }
    // Write input matrix for comparison:
    //write_input_matrix(A[5], dim);

    // Create cublas instance that points to the cuBLAS library context
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasHandle_t handle2;
    cublasCreate(&handle2);
    // Note: need to call destroy() 

    // Create new cublas streams
    cudaStream_t streams[5];
    int n_streams = 5;
    for (int i = 0; i < 5; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Set stream that will be used to execute all subsequent calls to the cuBLAS library functions:
    cublasSetStream(handle2, streams[2]);


    // Create host pointer array to device matrix storage
    
    //Note: Host To Device (h_d) pointers allow for specidying indexes of the double pointer from the host which is not possible with the Device To Device pointers
    cuDoubleComplex **d_T1, **d_T2, **d_T4, **d_T6, **d_T8, **d_T10, **d_A, **d_B, **d_C, **d_identity; // Device pointers to device arrays
    cuDoubleComplex **h_d_T1, **h_d_T2, **h_d_T4, **h_d_T6, **h_d_T8, **h_d_T10, **h_d_A, **h_d_B, **h_d_C, **h_d_identity; // Host pointers to device arrays
    

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
        cudaMemcpy(h_d_A[i], A + i*(dim*dim), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(h_d_T1[i], A + i*(dim*dim), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }


    // Alpha and beta coeficients set for zgemm :
    const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
    const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
    const cuDoubleComplex *alpha = &alf;
    const cuDoubleComplex *beta = &bet;


/////////////////////////////////////////////////////////////////////////////////////////////////// SETUP END /////////////////////////////////////////////////////////////////////////////////////////////

    // *** Powers of input matrices calculated using Batch ZGEMM (complex batch matrix multiplication) **
    
    // Calulate A^2 = A*A:
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

    // Calculate A^4 = A^2*A^2:
    cublasZgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        (const cuDoubleComplex**)d_T2, dim,
        (const cuDoubleComplex**)d_T2, dim,
        beta,
        d_T4, dim,
        batch_count);
    cudaDeviceSynchronize();

    // Calculate A^6 = A^2*A^4:
    cublasZgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        (const cuDoubleComplex**)d_T2, dim,
        (const cuDoubleComplex**)d_T4, dim,
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

    // Calculate T10:
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


    // purpose?
    double* d4;
    double* d6;
    double* d8;
    double* d10;
    cudaMallocHost((void**)&d4, batch_count*sizeof(double));
    cudaMallocHost((void**)&d6, batch_count*sizeof(double));
    cudaMallocHost((void**)&d8, batch_count*sizeof(double));
    cudaMallocHost((void**)&d10, batch_count*sizeof(double));


    // purpose?
    double* d_res;
    double* d_res2;
    double* d_res3;
    double* d_res4;
    cudaMalloc((void**)&d_res, sizeof(double)*batch_count);
    cudaMalloc((void**)&d_res2, sizeof(double)*batch_count);
    cudaMalloc((void**)&d_res3, sizeof(double)*batch_count);
    cudaMalloc((void**)&d_res4, sizeof(double)*batch_count);


    // Cuda device Grid setup:
    int dimensions = ceil((float) dim/BLOCK_SIZE);
    dim3 dimGrid(dimensions, 1, 1); 
    dim3 dimBlock(BLOCK_SIZE, 1, 1); 



    // Get one norm for each power of A for each matrix in the batch:
    for (int i = 0; i < batch_count; i++) {
        get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[1]>>>(h_d_T4[i], d_res, i, dim); 
        get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[2]>>>(h_d_T6[i], d_res2, i, dim); 
        get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[3]>>>(h_d_T8[i], d_res3, i, dim); 
        get_one_norm<<<dimGrid, dimBlock, dim*sizeof(double), streams[4]>>>(h_d_T10[i], d_res4, i, dim);
    }

    cudaMemcpyAsync(d4, d_res, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(d6, d_res2, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[2]);
    cudaMemcpyAsync(d8, d_res3, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[3]);
    cudaMemcpyAsync(d10, d_res4, sizeof(double)*batch_count, cudaMemcpyDeviceToHost, streams[4]);

    cudaDeviceSynchronize();

    for (int i = 0; i < batch_count; i++) {
        d4[i] = pow(d4[i], (1.0 / 4));      // d4 = ||A^4||^1/4
        d6[i] = pow(d6[i], (1.0 / 6));      // d6 = ||A^2||^1/3
        d8[i] = pow(d8[i], (1.0 / 8));      // d8 = ||A^4||^1/8
        d10[i] = pow(d10[i], (1.0 / 10));   // d10 = ||A^10||^1/10
    }

    double* eta1 = (double*) malloc(batch_count*sizeof(double));
    double* eta3 = (double*) malloc(batch_count*sizeof(double));
    double* eta4 = (double*) malloc(batch_count*sizeof(double));
    double* eta5 = (double*) malloc(batch_count*sizeof(double));
    int* m_val = (int*) malloc(batch_count*sizeof(int));

    // Algorithm refers to eta as n
    for (int i = 0; i < batch_count; i++) {
        eta1[i] = fmax(d4[i], d6[i]);       // n2 = max(d4, d6)
        eta3[i] = fmax(d6[i], d8[i]);       // n3 = max(d6, d8)
        eta4[i] = fmax(d8[i], d10[i]);      // n4 = max(d8, d10)
        eta5[i] = fmin(eta3[i], eta4[i]);   // n5 = min(n3,n4)
    }

    // *** This section finds the degree m of pade approximant to use where m can take the value [3,5,7,9,13] ***

    // Array containing values of theta for each value degree m of pade approximant
   double theta[5] = {
        1.495585217958292e-002, // theta3
        2.539398330063230e-001, // theta5
        9.504178996162932e-001, // theta7
        2.097847961257068e+000, // theta9
        5.371920351148152e+000  // theta13
    };

    double error_coefficients[5] = {
        1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
        1 / 113250775606021113483283660800000000.0,
        1 / 113250775606021113483283660800000000.0
    };


    int* d_m_val;
    cudaMalloc(&d_m_val, sizeof(int)*batch_count);



    // [1] Condition: n1 <= theta3:
    for (int i = 0; i < batch_count; i++) {
        if(eta1[i] <= theta[1]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock, 0, streams[i%3]>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[1], (1.0 / (2 * 3 + 1)));
            cublasSetStream(handle, streams[i]);
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            
            // Condition: ell(A,3) is 0:
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), streams[i%3]>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 3); // Then degree of Pade approximant is set to 3
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int)*batch_count, cudaMemcpyDeviceToHost);


    // [2] Condition: n2 <= theta5:
    for (int i = 0; i < batch_count; i++) {
        if(eta1[i] <=theta[2]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[2], (1.0 / (2 * 5 + 1)));
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);

            // Condition: ell(A, 5) is 0:
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 5); // Then degree of Pade approximant is set to 5
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int)*batch_count, cudaMemcpyDeviceToHost);


    // [3] Condition: n3 <= theta7:
    for (int i = 0; i < batch_count; i++) {
        if(eta3[i] <=theta[3]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[3], (1.0 / (2 * 7 + 1)));
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            // Condition: ell(A, 7) is 0:
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 7); // Then degree of Pade approximant is set to 7
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int)*batch_count, cudaMemcpyDeviceToHost);
  

    // [4] Condition: n3 <= theta9:
    for (int i = 0; i < batch_count; i++) {
        if(eta3[i] <= theta[4]){
            cudaMemcpyAsync(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);
            absolute_kernel<<<dimGrid, dimBlock>>>(h_d_B[i], dim);
            double p = pow(error_coefficients[4], (1.0 / (2 * 9 + 1)));
            scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(p, 0), dim);
            // Condition: ell(A, 9) is 0:
            ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), 0>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 9); // Then degree of Pade approximant is set to 9
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(m_val, d_m_val, sizeof(int)*batch_count, cudaMemcpyDeviceToHost);




    double* s = (double*) malloc(batch_count*sizeof(double)); // Variable s will hold the number of squaring resquired for stage 3 of 3
    double max = 0;

    // Cuda Grid setup:
    int dimensions_ell= ceil((float) dim/BLOCK_SIZE);
    dim3 dimBlock_ell(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid_ell(dimensions_ell, dimensions_ell); 

    int* temp_array;
    temp_array = (int*) malloc(sizeof(int)*batch_count);

    for (int i = 0; i < batch_count; i++) {
        s[i] = fmax(ceil(log2(eta5[i]/theta[4])), 0); // s = max(log2(n5,theta13), 0)
       
        cublasSetStream(handle, streams[i%n_streams]);
        scale_tester(handle, h_d_A[i], h_d_A[i], make_cuDoubleComplex(1/pow(2, s[i]), 0), dim);

        cudaMemcpy(h_d_B[i], h_d_A[i], dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        absolute_kernel<<<dimGrid_ell, dimBlock_ell, 0, streams[i%n_streams]>>>(h_d_B[i], dim);
        cudaDeviceSynchronize();
        scale_tester(handle, h_d_B[i], h_d_B[i], make_cuDoubleComplex(pow(error_coefficients[4], (1.0 / (2 * 13 + 1))), 0), dim);
        // Find ell(2^-s*A, 13)
        ell_kernel<<<dimGrid, dimBlock, dim*sizeof(double), streams[i%n_streams]>>>(h_d_A[i],h_d_B[i], dim, d_m_val, i, 13);
    }

    cudaMemcpy(temp_array, d_m_val, sizeof(int)*batch_count, cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_count; i++) {
        s[i] = s[i] + temp_array[i]; // Final value of s = s + ell(2^-s*A, 13)
        if(s[i] > max)
            max = s[i];
    }

    for (int i = 0; i < batch_count; i++) {
        // check s for infinity:
        if (isinf(s[i])) {
            printf("S/M CHECK HAS BEEN HIT\n");
            exit(0);
        } 
        else{
            m_val[i] = 13;
        }
    }


    // *** Rescale powers of A according to value of s *** MAPS TO PART 1 OF 3 (SCALING STAGE) IN ALGORITHM *** 
    // --------------------------------------------------------------------------------------------------------

    for (int i = 0; i < batch_count; i++) {
        if (s[i]!= 0) {
            scale_tester(handle, h_d_T1[i], h_d_T1[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 1)), 0), dim); // A = 2^-s*A 
            scale_tester(handle, h_d_T2[i], h_d_T2[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 2)), 0), dim); // A^2 = 2^-2s*A^2
            scale_tester(handle, h_d_T4[i], h_d_T4[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 4)), 0), dim); // A^4 = 2^-4s*A^4
            scale_tester(handle, h_d_T6[i], h_d_T6[i], make_cuDoubleComplex(1.0 / pow(2, (s[i] * 6)), 0), dim); // A^6 = 2^-6s*A^6
        }
    }


    // *** Evaluate p13(A) and q13(A) polynomials needed for finding the pade approximant r13(A) ***


    /* Get pade coefficients 
    ------------------------
    */

    double** c = (double**) malloc(batch_count*sizeof(double*)); 
    for (int i = 0; i < batch_count; i++) { 
        c[i] = (double*) malloc(15*sizeof(double));
        get_pade_coefficients(c[i], m_val[i]);
    }


    for (int i = 0; i < batch_count; i++) {
        if(m_val[i] != 13){
            printf("DIFFERENCE IS SEEN!\n");
            exit(0);
        }
    }


    /* Calculate U13 for each matrix in batch
    -----------------------------------------
    */

    set_Identity(h_d_identity[0], dim, streams[1]);   
    for (int i = 0; i < batch_count; i++) {
        scale_and_add_alt(handle2, h_d_T6[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][13], 0), dim); 
        scale_and_add_alt(handle2, h_d_T4[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][11], 0), dim);
        scale_and_add_alt(handle2, h_d_T2[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][9], 0), dim);
    }

    cudaDeviceSynchronize();

    cublasZgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        alpha,
        (const cuDoubleComplex**) d_C, dim,
        (const cuDoubleComplex**) d_T6, dim,
        beta,
        d_C, dim,
        batch_count);

    cudaDeviceSynchronize();

    for (int i = 0; i < batch_count; i++) {
        scale_and_add_alt(handle, h_d_T6[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][7], 0), dim);
        scale_and_add_alt(handle, h_d_T4[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][5], 0), dim);
        scale_and_add_alt(handle, h_d_T2[i], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][3], 0), dim);
        scale_and_add_alt(handle, h_d_identity[0], h_d_C[i], h_d_C[i], make_cuDoubleComplex(c[i][1], 0), dim);
        scale_and_add(handle, h_d_C[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim, 0);
    }

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


    /* Calculate V13 for each matrix in batch
    ---------------------------------------
    */

    for (int i = 0; i < batch_count; i++) {
        scale_and_add_complete(handle, h_d_T6[i], h_d_T4[i], h_d_B[i], make_cuDoubleComplex(c[i][12], 0), make_cuDoubleComplex(c[i][10], 0), dim);
        scale_and_add(handle, h_d_T2[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][8], 0), dim, 0);
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
                                        
  
    for (int i = 0; i < batch_count; i++) {  
        scale_and_add(handle, h_d_T6[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][6], 0), dim, 0); 
        scale_and_add(handle, h_d_T4[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][4], 0), dim, 0);
        scale_and_add(handle, h_d_T2[i], h_d_A[i], h_d_A[i], make_cuDoubleComplex(c[i][2], 0), dim, 0);
        scale_and_add(handle, h_d_identity[0], h_d_B[i], h_d_B[i], make_cuDoubleComplex(c[i][0], 0), dim, 0);
        scale_and_add(handle, h_d_A[i], h_d_B[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim, 0);

        /* Calculate q13(A) = V-U
        -------------------------
        */

        scale_and_subtract(handle, h_d_B[i], h_d_C[i], h_d_B[i], make_cuDoubleComplex(1, 0), dim); // THIS WOULD BE THE SYNCHRONIZATION POINT
    }


    // Calulate the pade approximant rm(A) form equation (3.6) --> (-U13 + V13)r13(A) = (U13 + V13)(A) ** MAPS TO PART 2 OF 3 (PADE APPROXIMANT STAGE) IN ALGORITHM ***
    // -----------------------------------------------------------------------------------------------------------------------------------------------------------------

    /* *** Calculate F = (V-U)/(2*U) + I *** // CAN I DO IT THE OTHER WAY?
    ----------------------------------------
    */

    // Find inverse of each (V-U) in batch through LU decomposition:
    Inverse_Batched(handle, d_B, d_A, dim, batch_count); 

    // Scale U by 2:
    for (int i = 0; i < batch_count; i++) {
        scale_tester(handle, h_d_C[i], h_d_C[i], make_cuDoubleComplex(2, 0), dim);
    }

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



    // *** Square r13(A) s times through repeated squaring (X = r13(A)^2^s) *** MAPS TO PART 3 OF 3 (SQUARING STAGE) IN ALGORITHM *** 
    // -----------------------------------------------------------------------------------------------------------------------------

    for (int k = 0; k < 2; k++) { 
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

        for(int i=0; i< batch_count; i++) {
            
            // If matrix i in batch has been squared s[i] times then copy matrix i back to host: 
            if(s[i] == (k + 1))
                cudaMemcpy(output + i*(dim*dim), h_d_C[i], dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                cudaMemcpy(A + i*(dim*dim), h_d_C[i], dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        }

        // Switch pointers to avoid memory copies after every squaring
        d_B = d_C; 
    }

    // printf("Matrix Exponential: \n");
    // matrix_complex_print(A, dim);

    
    // free Memory:
    for (int i = 0; i < batch_count; i++) {
        cudaFree(h_d_T1[i]);
        cudaFree(h_d_T2[i]);
        cudaFree(h_d_T4[i]);
        cudaFree(h_d_T6[i]);
        cudaFree(h_d_T8[i]);
        cudaFree(h_d_T10[i]);
        cudaFree(h_d_A[i]);
        cudaFree(h_d_B[i]);
        cudaFree(h_d_C[i]);
        cudaFree(h_d_identity[i]);
    }

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