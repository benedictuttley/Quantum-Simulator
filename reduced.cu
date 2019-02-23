#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <time.h>

#define CUDA_CALL(res, str) { if (res != cudaSuccess) { printf("CUDA Error : %s : %s %d : ERR %s\n", str, __FILE__, __LINE__, cudaGetErrorName(res)); } }
#define CUBLAS_CALL(res, str) { if (res != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS Error : %s : %s %d : ERR %d\n", str, __FILE__, __LINE__, int(res)); } }

static cudaEvent_t cu_TimerStart;
static cudaEvent_t cu_TimerStop;

void d_CUDATimerStart(void)
{
    CUDA_CALL(cudaEventCreate(&cu_TimerStart), "Failed to create start event!");
    CUDA_CALL(cudaEventCreate(&cu_TimerStop), "Failed to create stop event!");

    CUDA_CALL(cudaEventRecord(cu_TimerStart), "Failed to record start event!");
}

float d_CUDATimerStop(void)
{
    CUDA_CALL(cudaEventRecord(cu_TimerStop), "Failed to record stop event!");

    CUDA_CALL(cudaEventSynchronize(cu_TimerStop), "Failed to synch stop event!");

    float ms;

    CUDA_CALL(cudaEventElapsedTime(&ms, cu_TimerStart, cu_TimerStop), "Failed to elapse events!");

    CUDA_CALL(cudaEventDestroy(cu_TimerStart), "Failed to destroy start event!");
    CUDA_CALL(cudaEventDestroy(cu_TimerStop), "Failed to destroy stop event!");

    return ms;
}

float* d_GetInv(float* L, int n, float* b)
{
    cusolverStatus_t  status;
    cusolverDnHandle_t handler;
    status=cusolverDnCreate(&handler);

    float* A;
    int* dLUPivots_ALT;
    int* dLUInfo_ALT;
    float *buffer = NULL;
    int bufferSize = 0;
    int h_info = 0;
    float *x;
    


    CUDA_CALL(cudaMalloc(&A, sizeof(float)*n*n), "Failed to allocate A!");
    CUDA_CALL(cudaMalloc(&x, n * n*sizeof(float)), "Failed to allocate x!");
     
    CUDA_CALL(cudaMalloc(&dLUPivots_ALT, n * sizeof(int)), "Failed to allocate dLUPivots!");
    CUDA_CALL(cudaMalloc(&dLUInfo_ALT, sizeof(int)), "Failed to allocate dLUInfo!");
    CUDA_CALL(cudaMemcpy(A, L, n*n*sizeof(float), cudaMemcpyHostToDevice), "Failed to copy to adL!");
    cudaMemcpy(x, b, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    cusolverDnSgetrf_bufferSize(handler, n, n, (float*)A, n, &bufferSize);
    cudaMalloc(&buffer, sizeof(float)*bufferSize);
  
    status=cusolverDnSgetrf(handler, n, n, A, n, buffer, dLUPivots_ALT, dLUInfo_ALT);
    if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } else{
        printf("SUCCESS!!\n");
    }

    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
 
    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
        printf("%d\n", h_info );
    }
    
  
    cusolverDnSgetrs(handler, CUBLAS_OP_N, n, n, A, n, dLUPivots_ALT, x, n, dLUInfo_ALT);
    cudaDeviceSynchronize();
     if(status!=CUSOLVER_STATUS_SUCCESS){
        printf("ERROR!!\n");
    } else{
        printf("SUCCESS!!\n");
    }

    cudaMemcpy(&h_info, dLUInfo_ALT, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("HERE WE ARE \n");
        if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
        
    }

    float* res2 = (float*)malloc(n * n * sizeof(float));

    CUDA_CALL(cudaMemcpy(res2, x, sizeof(float) * n * n, cudaMemcpyDeviceToHost), "Failed to copy to res!");

    // for(int i = 0; i < n; i++){
    //     for (int j = 0; j < n; j++)
    //     {
    //         printf("%f ", res2[(n*i) + j] );
    //     }
    //     printf("\n");
    // }

    return res2;
}

int main()
{
    int n = 1000;
    float* L = (float*)malloc(n * n * sizeof(float));
    float* I = (float*)malloc(n * n * sizeof(float));
    for(int i = 0; i < n * n; i++){
        L[i] = ((float)rand()/(float)(RAND_MAX));

    }


    // IDENTITY MATRIX
     for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++){
            if(i == j){
                I[(n*i) + j] = 1;
            } else{
                I[(n*i) + j] = 0;
            }
        }
    }


    clock_t start = clock();
    float* inv = d_GetInv(L, n, I);
    clock_t end = clock();
    
    printf("%lf \n", (double)(end-start)/1000);
    return 0;
}