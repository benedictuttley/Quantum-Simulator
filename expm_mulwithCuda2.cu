#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdio.h>  // CUDA TRANSFORM 1, [CHANGE: MULTIPLY WITH CUBLAS]
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

#define CUDA_CALL(res, str) { if (res != cudaSuccess) { printf("CUDA Error : %s : %s %d : ERR %s\n", str, __FILE__, __LINE__, cudaGetErrorName(res)); } }
#define CUBLAS_CALL(res, str) { if (res != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error : %s : %s %d : ERR %d\n", str, __FILE__, __LINE__, int(res)); } }


#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)





void matrix_Write_New(double *A, int n) {
    FILE *f;
    f = fopen("/home/c1673666/expm_Cuda/cuda/Quantum-Simulator/myFile.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    /* print integers and doubles */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {

            if (i == n - 1) {
                if (A[(n * j) + i] == INFINITY) {
                    fprintf(f, "Inf");
                } else {
                    fprintf(f, "%.5g", A[(n * j) + i]);
                }
            } else {
                if (A[(n * j) + i] == INFINITY) {
                    fprintf(f, "Inf ");
                } else {
                    fprintf(f, "%.5g ", A[(n * j) + i]);
                }
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
}


void matrix_Print_New(double *A, int n) {
   printf("\n");
   for (int l = 0; l < n; l++) {
       for (int i = 0; i < n; i++) {
           printf("|%20.15e|   ", A[(n * l) + i]);
       }
       printf("\n");
   }
}


void copy_To_Device_For_Mult(double *h_A, double *h_B, double *h_C, double* d_A, double* d_B, double* d_C, int n){

    cudaMemcpy(d_A,h_A, n * n * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B, n * n * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C, n * n * sizeof(double),cudaMemcpyHostToDevice);

}

void copy_From_Device_For_Mult(double *h_A, double *h_B, double *h_C, double* d_A, double* d_B, double* d_C, int n){

    cudaMemcpy(d_A,h_A, n * n * sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(d_B,h_B, n * n * sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(d_C,h_C, n * n * sizeof(double),cudaMemcpyDeviceToHost);

}


void matrix_Multiply(cublasHandle_t &handle, double *A, double *B, double *C, double *d_A, double* d_B, double *d_C, int n, double* multiply_total_time){

    clock_t multiply_begin = clock();

    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    cudaMemcpy(d_A, A, n * n * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(double),cudaMemcpyHostToDevice);

    //printf("--------------------------\n");
    // printf("__%lf\n", A[0] );
    // printf("%lf\n", B[0] );
    // printf("%lf\n", C[0] );
  
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_B, n, d_A, n, beta, d_C, n);

    cudaMemcpy(C, d_C, n * n * sizeof(double),cudaMemcpyDeviceToHost);
    // printf("__%lf\n", A[0] );
    // printf("%lf\n", B[0] );
    // printf("%lf\n", C[0] );
   // exit(0);
    clock_t multiply_end = clock();
    double time_spent = (double)(multiply_end - multiply_begin) / CLOCKS_PER_SEC;
    multiply_total_time[0] = multiply_total_time[0] + time_spent;
    //printf("\n TIME ELAPSED FOR MULTIPLY: %lf seconds \n", time_spent);
}


void matrix_Copy_New(double *destination, const double *source, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            destination[(n * i) + j] = source[(n * i) + j];
        }
    }
}


double **get_Matrix_Powers_New(double *A, double* d_A, double* d_B, double* d_C, cublasHandle_t handle, int n, double* multiply_total_time) {

    double **Tpowers = (double **) malloc(11 * sizeof(double *));

    for (int i = 0; i < 11; i++) {
        Tpowers[i] = (double *) malloc(n * n * sizeof(double));
    }

    matrix_Copy_New(Tpowers[1], A, n);
   
    matrix_Multiply(handle, Tpowers[1], Tpowers[1], Tpowers[2], d_A, d_B, d_C, n, multiply_total_time);
    
    matrix_Multiply(handle, Tpowers[2], Tpowers[2], Tpowers[4], d_A, d_B, d_C, n, multiply_total_time);
    
    matrix_Multiply(handle, Tpowers[4], Tpowers[2], Tpowers[6], d_A, d_B, d_C, n, multiply_total_time);
    
    matrix_Multiply(handle, Tpowers[4], Tpowers[4], Tpowers[8], d_A, d_B, d_C, n, multiply_total_time);

    return Tpowers;
}


// FIND THE INVERSE OF A MATRIX - NEEDED FOR MATRIX DIVISION, SOURCE:  http://www.sourcecodesworld.com/source/show.asp?ScriptID=1086.
void InverseOfMatrix_New(double *A, double *inv_New, int n) {   // !!PARALLEL CANDIDATE!!
    
    int i, j, k;
    double temp;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (i == j)
                inv_New[(n * i) + j] = 1;
            else
                inv_New[(n * i) + j] = 0;

    for (k = 0; k < n; k++) {
        temp = A[(n * k) + k];

        for (j = 0; j < n; j++) {
            A[(n * k) + j] /= temp;
            inv_New[(n * k) + j] /= temp;
        }
        for (i = 0; i < n; i++) {
            temp = A[(n * i) + k];
            for (j = 0; j < n; j++) {
                if (i == k)
                    break;
                A[(n * i) + j] -= A[(n * k) + j] * temp;
                inv_New[(n * i) + j] -= inv_New[(n * k) + j] * temp;
            }
        }
    }
 
}

// HERE: ATTEMPT TO USE ON LARGE MATRICES AND TEST ACCURACY + REPLACE BATCH

void InverseOfMatrix_Alternative_Two(double* L, double* res2, int n, double* b){
    cusolverStatus_t  status;
    cusolverDnHandle_t handler;
    status=cusolverDnCreate(&handler);

    double* A;
    int* dLUPivots_ALT;
    int* dLUInfo_ALT;
    double *buffer = NULL;
    int bufferSize = 0;
    int h_info = 0;
    double *x;
    


    CUDA_CALL(cudaMalloc(&A, sizeof(double)*n*n), "Failed to allocate A!");
    CUDA_CALL(cudaMalloc(&x, n * n*sizeof(double)), "Failed to allocate x!");
     
    CUDA_CALL(cudaMalloc(&dLUPivots_ALT, n * sizeof(int)), "Failed to allocate dLUPivots!");
    CUDA_CALL(cudaMalloc(&dLUInfo_ALT, sizeof(int)), "Failed to allocate dLUInfo!");
    CUDA_CALL(cudaMemcpy(A, L, n*n*sizeof(double), cudaMemcpyHostToDevice), "Failed to copy to adL!");
    cudaMemcpy(x, b, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    cusolverDnDgetrf_bufferSize(handler, n, n, (double*)A, n, &bufferSize);
    cudaMalloc(&buffer, sizeof(double)*bufferSize);
  
    status=cusolverDnDgetrf(handler, n, n, A, n, buffer, dLUPivots_ALT, dLUInfo_ALT);
    if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } 

    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
 
    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
        printf("%d\n", h_info );
    }
    
  
    cusolverDnDgetrs(handler, CUBLAS_OP_N, n, n, A, n, dLUPivots_ALT, x, n, dLUInfo_ALT);
    cudaDeviceSynchronize();
     if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } 
    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("HERE WE ARE \n");
        if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
        
    }

   // double* res2 = (double*)malloc(n * n * sizeof(double));

    CUDA_CALL(cudaMemcpy(res2, x, sizeof(double) * n * n, cudaMemcpyDeviceToHost), "Failed to copy to res!");

    // for(int i = 0; i < n; i++){
    //     for (int j = 0; j < n; j++)
    //     {
    //         printf("%f ", res2[(n*i) + j] );
    //     }
    //     printf("\n");
    // }
}
void InverseOfMatrix_Alternative(cublasHandle_t handle, double* input, double* inv, int n){

    //cublasHandle_t handle;
    //cublascall(cublasCreate_v2(&handle));


////////////////////////
    double* src_d, *dst_d;
    cudacall(cudaMalloc<double>(&src_d,n * n * sizeof(double)));
    cudacall(cudaMemcpy(src_d,input,n * n * sizeof(double),cudaMemcpyHostToDevice));

    cudacall(cudaMalloc<double>(&dst_d,n * n * sizeof(double)));

////////////////////////



    int batchSize = 1;

    int *P, *INFO;

    cudacall(cudaMalloc<int>(&P,n * batchSize * sizeof(int)));
    cudacall(cudaMalloc<int>(&INFO,batchSize * sizeof(int)));

    int lda = n;

    double *A[] = { src_d };
    double** A_d;
    cudacall(cudaMalloc<double*>(&A_d,sizeof(A)));
    cudacall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublascall(cublasDgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    printf("%d\n", INFOh );
    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));
    //printf("HERE\n");

    if(INFOh == n)
    {
        printf("IN CONDITIONAL\n");
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

   // printf("HERE TOO\n");
    double* C[] = { dst_d };
    //printf("HERE THREE\n");
    double** C_d;
    cudacall(cudaMalloc<double*>(&C_d,sizeof(C)));
    cudacall(cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice));
    //printf("HERE FOUR\n");

    cublascall(cublasDgetriBatched(handle, n, (const double**) A_d,lda,P,C_d,lda,INFO,batchSize));
    //printf("HERE FIVE\n");

    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));
    //printf("HERE SIX\n");

    if(INFOh != 0)
    {
        //printf("IN CONDITIONAL 2\n");
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    
    //printf("HERE ALSO ");
    cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);

    cudacall(cudaMemcpy(inv,dst_d,n * n * sizeof(double),cudaMemcpyDeviceToHost));
    //printf("%lf\n", inv[20]);
    //matrix_Print_New(inv, n);
    cudaFree(src_d), cudaFree(dst_d);


}

///////////////////////////////////////////////////////////////////////////////////


void matrix_Subtract_New(const double *a, const double *b, double *c, int n) {
     clock_t subtract_begin = clock();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = a[(n * i) + j] - b[(n * i) + j];
        }
    }
    clock_t subtract_end = clock();
    double time_spent = (double)(subtract_end - subtract_begin) / CLOCKS_PER_SEC;
    //printf("\n TIME ELAPSED FOR SUBTRACTION: %lf seconds \n", time_spent);
}


void matrixAdd_New(const double *a, const double *b, double *c, int n) {
    clock_t add_begin = clock();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = a[(n * i) + j] + b[(n * i) + j];
        }
    }

    clock_t add_end = clock();
    double time_spent = (double)(add_end - add_begin) / CLOCKS_PER_SEC;
    //printf("\n TIME ELAPSED FOR ADDITION: %lf seconds \n", time_spent);
}



void set_Identity_New(double *i_matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                i_matrix[(n * i) + j] = 1;
            } else {
                i_matrix[(n * i) + j] = 0;
            }
        }
    }
}


void matrix_Scale_New(const double *a, double *scaled, double scale, int n, double* scale_total_time ) {
    clock_t scale_begin = clock();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scaled[(n * i) + j] = a[(n * i) + j] * scale;
        }
    }

    clock_t scale_end = clock();
    double time_spent = (double)(scale_end - scale_begin) / CLOCKS_PER_SEC;
    scale_total_time[0] = scale_total_time[0] + time_spent;
    //printf("\n TIME ELAPSED FOR SCALE: %lf seconds \n", time_spent);
}


void matrix_Absolute_New(double *a, double *b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[(n * i) + j] = fabs((a[(n * i) + j]));
        }
    }
}


double calculate_one_norm_New(const double *A, int n) {
    double max = -DBL_MAX;
    double count;
    for (int i = 0; i < n; i++) {
        count = 0;
        for (int j = 0; j < n; j++) {
            count += A[(n * j) + i];
        }
        if (count > max) {
            max = count;
        };
    }
   // printf("The one-norm of the matrix is: %7.1f \n", max);
    return max;
}


// COMPUTING OPTIMAL PARAMETERS
double ell(double *A, double *temp_new, double coeff, int m_val, int n, double* scale_total_time) {

    double alpha, p, norm_one, norm_two, output;

    matrix_Copy_New(temp_new, A, n);
    matrix_Absolute_New(A, temp_new, n);

    p = pow(coeff, (1.0 / (2 * m_val + 1)));

    matrix_Scale_New(temp_new, temp_new, p, n, scale_total_time);

    norm_one = calculate_one_norm_New(A, n);
    norm_two = calculate_one_norm_New(temp_new, n);
    alpha = norm_two / norm_one;
    //printf("ALPHA IS: |%12.5e|\n", alpha);
    output = fmax(ceil(log2(2 * alpha / nextafterf(1.0, INFINITY) / (2 * m_val))), 0);
    //printf("OUTPUT |%12.5e|\n", output);
    return output;
}


int getNumRows(int n) {
    n = 0;
    FILE *file;
    int ch;
    file = fopen("/home/c1673666/expm_Cuda/cuda/Quantum-Simulator/read.txt", "r");
    while (!feof(file)) {
        ch = fgetc(file);
        if (ch == '\n') {
            n++;
        }
    }

    printf("\n SIZE OF MATRIX IS: %d  * %d \n", n, n);
    return n;
}


// READ INPUT MATRIX FROM A TEXT FILE
void loadMatrix_New(double *a, int n) {
    
    FILE *file;
    
    file = fopen("/home/c1673666/expm_Cuda/cuda/Quantum-Simulator/read.txt", "r");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            if (!fscanf(file, "%lf", &a[(n * i) + j])) // NOLINT(cert-err34-c)
                break;
        }
    }

    fclose(file);
}


void get_pade_coefficients(double *buf, int m) {

    double test[5][14] = {
            {120,               60,                12,         1},

            {30240,             15120,             3360,       420,              30,              1},

            {17297280,          8648640,           1995840,    277200,           25200,           1512,           56,     1},

            {17643225600,       8821612800,        2075673600, 302702400,        30270240,
                                                                                                  2162160,        110880, 3960,        90,         1},

            {64764752532480000, 32382376266240000, 7771770303897600,
                                                               1187353796428800, 129060195264000, 10559470521600, 670442572800,
                                                                                                                          33522128640, 1323241920, 40840800, 960960, 16380, 182, 1}

    };

    switch (m) {

        case 3  : {
            buf = test[0];

        }
        case 5  : {
            buf = test[1];
        }
        case 7  : {
            buf = test[2];
        }

        case 9  : {
            buf = test[3];
        }
        case 13  : {
            for (int i = 0; i < sizeof(test[4]) / sizeof(double); i++) {
                buf[i] = test[4][i];
            }
        }
        default:
            break;
    }
}

int main() {

    printf("\n");
    printf("\n");

    // VARIABLES TO HOLD TOTAL COMPONENT TIMES:
    double addition_total_time = 0;
    double subtraction_total_time = 0;
    double inverse_total_time = 0;
    double* scale_total_time = (double *) malloc(1 * sizeof(double));
    double* multiply_total_time = (double *) malloc(1 * sizeof(double));


    /* here, do your time-consuming job */

    int n = 0;
    double *A;
  
    n = getNumRows(n);

    A = (double *) malloc(n * n * sizeof(double));
    
    loadMatrix_New(A, n);
   

    clock_t begin = clock();    // BEGIN TIMINGS


    // CUBLAS HANDLE:
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate 3 arrays on GPU
     double *d_A, *d_B, *d_C;
     cudaMalloc(&d_A, n * n * sizeof(double));
     cudaMalloc(&d_B, n * n * sizeof(double));
     cudaMalloc(&d_C, n * n * sizeof(double));



    // theta for m=1 -> m=13
    double theta[5] = {1.495585217958292e-002, 2.539398330063230e-001,
                       9.504178996162932e-001, 2.097847961257068e+000,
                       5.371920351148152e+000};

    double error_coefficients[5] = {1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
                                    1 / 113250775606021113483283660800000000.0,
                                    1 / 113250775606021113483283660800000000.0};


    double *identity_new;
    double *U_new;
    double *V_new;
    double *temp_new;
    double *temp_2_new;

    identity_new = (double *) malloc(n * n * sizeof(double));
    U_new = (double *) malloc(n * n * sizeof(double));
    V_new = (double *) malloc(n * n * sizeof(double));
    temp_new = (double *) malloc(n * n * sizeof(double));
    temp_2_new = (double *) malloc(n * n * sizeof(double));


    double **Tpowers = get_Matrix_Powers_New(A, d_A, d_B, d_C, handle, n, multiply_total_time);


    //printf("Norm for A4 \n");
    double d4 = pow(calculate_one_norm_New(Tpowers[4], n), (1.0 / 4));
    //printf("%f", d4);

   // printf("Norm for A6 \n");
    double d6 = pow(calculate_one_norm_New(Tpowers[6], n), (1.0 / 6));
   // printf("%f", d6);
    double eta1 = fmax(d4, d6);


    int m_val = 0;

    if (eta1 <= theta[1] && ell(A, temp_new, error_coefficients[1], 3, n, scale_total_time) == 0.0) {
        m_val = 3;

    }
    if (eta1 <= theta[2] && ell(A, temp_new, error_coefficients[2], 5, n, scale_total_time) == 0.0) {
        m_val = 5;

    }


    //COMPUTE MATRIX POWER EXPLICITLY - NEED TO LOOK AT normAM
   // printf("Norm for A8 \n");
    double d8 = pow(calculate_one_norm_New(Tpowers[8], n), (1.0 / 8));

    double eta3 = fmax(d6, d8);

    if (eta3 <= theta[3] && ell(A, temp_new, error_coefficients[3], 7, n, scale_total_time) == 0.0) {
        m_val = 7;
    }

    if (eta3 <= theta[4] && ell(A, temp_new, error_coefficients[4], 0, n, scale_total_time) == 0.0) {
        m_val = 9;
    }

    // COMPUTE MATRIX POWER EXPLICITLY - NEED TO LOOK AT normAM
    //printf("Norm for A10 \n"); //- DO I EVEN NEED THIS???
    double d10 = pow(calculate_one_norm_New(Tpowers[10], n), (1.0 / 10));

    double eta4 = fmax(d8, d10);
    double eta5 = fmin(eta3, eta4);
    double s = fmax(ceil(log2(eta5 / theta[4])), 0);


    matrix_Scale_New(Tpowers[1], Tpowers[1], 1 / pow(2, s), n, scale_total_time);
    matrix_Scale_New(A, A, 1 / pow(2, s), n, scale_total_time);
    s = s + ell(A, temp_new, error_coefficients[4], 13, n, scale_total_time);
    //printf("S IS:  |%12.5e|\n ", s);


    if (isinf(s)) {
        // Revert to old estimate
        int exp;
        double t = frexp(calculate_one_norm_New(A, n) / theta[4], &exp);
        s = s - (t == 0.5);
    } else {
        m_val = 13;
    }

    double arr[15] = {1};
    int m = 13;
    get_pade_coefficients(arr, m);
//    //printf("\nThe Pade coefficients for m = %d: \n", m);
//    for (int j = 0; j < 14; j++) {
//        printf("%7.1f\n", arr[j]);
//    }


    if ((int) s != 0) {
        double multiplier;

        multiplier = 1.0 / pow(2, (s * 2));
        matrix_Scale_New(Tpowers[2], Tpowers[2], multiplier, n, scale_total_time);

        multiplier = 1.0 / pow(2, (s * 4));
        matrix_Scale_New(Tpowers[4], Tpowers[4], multiplier, n, scale_total_time);

        multiplier = 1.0 / pow(2, (s * 6));
        matrix_Scale_New(Tpowers[6], Tpowers[6], multiplier, n, scale_total_time);
    }


    // PADE APPROXIMATION:
    double c[15] = {1};
    get_pade_coefficients(c, m_val);
    set_Identity_New(identity_new, n);


    if (m_val == 3 || m_val == 5 || m_val == 7 || m_val == 9) {
        printf("UNSEEN HAS BEEN ACTIVATED");
        int strt = sizeof(Tpowers) + 2;
        for (int k = strt; k < m_val - 1; k += 2) {
            //matrix_Multiply(Tpowers[2], Tpowers[k-2], Tpowers[k], n, multiply_total_time);
            matrix_Multiply(handle, Tpowers[2], Tpowers[k-2], Tpowers[k], d_A, d_B, d_C, n, multiply_total_time);

        }

        matrix_Scale_New(identity_new, U_new, c[1], n, scale_total_time);
        matrix_Scale_New(identity_new, V_new, c[0], n, scale_total_time);

        for (int j = m_val; j > n; j -= 2) {

            matrix_Scale_New(Tpowers[j - 1], temp_new, c[j + 1], n, scale_total_time);
            matrixAdd_New(U_new, temp_new, U_new, n);

            matrix_Scale_New(Tpowers[j - 1], temp_new, c[j], n, scale_total_time);
            matrixAdd_New(V_new, temp_new, V_new, n);
        }
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, U_new, n, 1, temp_new, n);
        //matrix_Multiply(U_new, A, temp_new, n, multiply_total_time);
         matrix_Multiply(handle, U_new, A, temp_new, d_A, d_B, d_C, n, multiply_total_time);
         memcpy(U_new, temp_new, n * n * sizeof(double));
    }

    if (m_val == 13) {

        // CALCUATE U PSEUDOCODE:

        matrix_Scale_New(Tpowers[6], temp_new, c[13], n, scale_total_time);

        memset(temp_2_new, 0, n * n * sizeof(double));

        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        matrix_Scale_New(Tpowers[4], temp_new, c[11], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[2], temp_new, c[9], n, scale_total_time);

        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);


        memset(temp_new, 0, n * n * sizeof(double));
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, temp_2_new, n, Tpowers[6], n, 1, temp_new, n);
        //matrix_Multiply(Tpowers[6], temp_2_new, temp_new, n, multiply_total_time);
        matrix_Multiply(handle, Tpowers[6], temp_2_new, temp_new, d_A, d_B, d_C, n, multiply_total_time);

    

        matrix_Scale_New(Tpowers[6], temp_2_new, c[7], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        matrix_Scale_New(Tpowers[4], temp_2_new, c[5], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        matrix_Scale_New(Tpowers[2], temp_2_new, c[3], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        set_Identity_New(identity_new, n);

        matrix_Scale_New(identity_new, temp_2_new, c[1], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(U_new, 0, n * n * sizeof(double));
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, temp_new, n, 1, U_new, n);
        //matrix_Multiply(temp_new, A, U_new, n, multiply_total_time);
        matrix_Multiply(handle, temp_new, A, U_new, d_A, d_B, d_C, n, multiply_total_time);





        // CALCULATE V PSEUDOCODE:
        memset(temp_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[6], temp_new, c[12], n, scale_total_time);

        memset(temp_2_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[4], temp_2_new, c[10], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[2], temp_new, c[8], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n * n * sizeof(double));
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[6], n, temp_2_new, n, 1, temp_new, n);
        //matrix_Multiply(temp_2_new, Tpowers[6], temp_new, n, multiply_total_time);
        matrix_Multiply(handle, temp_2_new, Tpowers[6], temp_new, d_A, d_B, d_C, n, multiply_total_time);

        memset(temp_2_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[6], temp_2_new, c[6], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[4], temp_2_new, c[4], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n * n * sizeof(double));
        matrix_Scale_New(Tpowers[2], temp_2_new, c[2], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n * n * sizeof(double));
        set_Identity_New(identity_new, n);
        matrix_Scale_New(identity_new, temp_2_new, c[0], n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, V_new, n);



        // CALCULATE F:
        matrix_Subtract_New(V_new, U_new, V_new, n);
        matrix_Scale_New(U_new, U_new, 2, n, scale_total_time);
        memset(temp_2_new, 0, n * n * sizeof(double));

        clock_t inverse_begin = clock();
        //InverseOfMatrix_New(V_new, temp_2_new, n);
        //InverseOfMatrix_Alternative(handle, V_new, temp_2_new, n);
        InverseOfMatrix_Alternative_Two(V_new, temp_2_new, n, identity_new);
       
        clock_t inverse_end = clock();
        double time_spent = (double)(inverse_end - inverse_begin) / CLOCKS_PER_SEC;
        double inverse_total_time = inverse_total_time + time_spent;
        //printf("\n TIME ELAPSED FOR INVERSE: %lf seconds \n", time_spent);

        memset(temp_new, 0, n * n * sizeof(double));
        //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, U_new, n, temp_2_new, n, 1, temp_new, n);
        //matrix_Multiply(temp_2_new, U_new, temp_new, n, multiply_total_time);
        matrix_Multiply(handle, temp_2_new, U_new, temp_new, d_A, d_B, d_C, n, multiply_total_time);

        matrixAdd_New(temp_new, identity_new, temp_new, n);

        // NOW PERFROM THE SQUARING PHASE



        for (int k = 0; k < s; k++) { // MORE EFFICIENT WAY TO DO SQUARING --> THE SLOWEST PART
            //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, temp_new, n, temp_new, n, 1, temp_2_new, n);
            //matrix_Multiply(temp_new, temp_new, temp_2_new, n, multiply_total_time);
            matrix_Multiply(handle, temp_new, temp_new, temp_2_new, d_A, d_B, d_C, n, multiply_total_time);
            memcpy(temp_new, temp_2_new, n * n * sizeof(double));
            memset(temp_2_new, 0, n * n * sizeof(double));
        }

        // WRITE THE MATRIX EXPO TO A TEXT FILE
        //printf("DONE \n");

        clock_t end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\n TOTAL TIME ELAPSED: %lf seconds \n", time_spent);

        printf("\n----------------------- MATRIX OPERATIONS PERCENTAGE BREAKDOWN -----------------------\n");

        printf("\n INVERSE: %lf%% \n", (inverse_total_time/time_spent)*100);
        printf("\n SCALE: %lf%% \n", (scale_total_time[0]/time_spent)*100);
        printf("\n MULTIPLY: %lf%% \n\n", (multiply_total_time[0]/time_spent)*100);
        matrix_Write_New(temp_new, n);
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    return 0;
}

// http://www.netlib.org/utk/people/JackDongarra/PAPERS/Factor_Inversion_Million_Matrices-iccs17.pdf
// http://mathforcollege.com/nm/mws/che/04sle/mws_che_sle_spe_luinverse.pdf