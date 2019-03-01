#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas.h>
#include <cblas.h>
void resuse(char *str);
 
double timeDiff( struct timespec *t1, struct timespec *t2)
{
  double T1, T2;
  T2 = (double)t2->tv_sec + (double)t2->tv_nsec / 1.0e9;
  T1 = (double)t1->tv_sec - (double)t1->tv_nsec / 1.0e9;
  return(T2 - T1);
}
 
int main(int argc, char *argv[])
{
  int dim = atoi(argv[1]);
  printf("%d\n", dim);
  int i,j;
  int status;
 
  double *psa, *psb, *psc, *psc_GPU;
  double *sap, *sbp, *scp;
  double *pda, *pdb, *pdc;
 
  double alpha   = 1.0;
  double beta    = 0.0;
  float  deltaT  = 0.0;
  struct timespec t1;
  struct timespec t2;
 
  int ptime();
 
  pda = NULL;
  pdb = NULL;
  pdc = NULL;
  psa = (double *) malloc(dim * dim * sizeof(*psa) );
  psb = (double *) malloc(dim * dim * sizeof(*psb) );
  psc = (double *) malloc(dim * dim * sizeof(*psc) );
  //psc_GPU = (double *) malloc(dim * dim * sizeof(*psc) );
 cudaHostAlloc((void**) &psc_GPU, dim * dim * sizeof(*psc), cudaHostAllocMapped);
 
  size_t mem_tot_0 = 0;
  size_t mem_free_0 = 0;
  cudaMemGetInfo  (&mem_free_0, & mem_tot_0);
  printf("DEVICE MEMORY TOTAL: %zu MB\n", mem_tot_0/1000000);
  printf("DEVICE AVAILABLE: %zu MB\n", mem_free_0/1000000);
  clock_gettime(CLOCK_MONOTONIC, &t1); 
  sap = psa;
  sbp = psb;
  scp = psc;
  for (i = 0; i < dim; i++)
    for (j = 0; j < dim; j++) {
        sap[(dim*i) + j] = ((dim*i) + j);
        sbp[(dim*i) + j] = ((dim*i) + j);
        *scp++ = 0.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1); 
  	/* Performs operation using blas */
  	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, psa, dim, psb, dim, beta, psc, dim);
  	clock_gettime(CLOCK_MONOTONIC, &t2); 
  	deltaT = timeDiff(&t1, &t2);
  	printf(" *** Elapsed Time [BLAS] = %6.4f secs *** \n", deltaT);
    
 
  /* Initialize CUDA */
  status = cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! CUBLAS initialization error\n");
      return EXIT_FAILURE;
  }
 
  /* Re-initialize the matrices */
  clock_gettime(CLOCK_MONOTONIC, &t1); 
  sap = psa;
  sbp = psb;
  scp = psc_GPU;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      sap[(dim*i) + j] = ((dim*i) + j);
      sbp[(dim*i) + j] = ((dim*i) + j);
      *scp++ = 0.0;
    }
  }
  
  clock_gettime(CLOCK_MONOTONIC, &t2); 
  deltaT = timeDiff(&t1, &t2);;
  fflush(stdout);
 
  /* Allocate device memory for the matrices */

  status = cublasAlloc(dim*dim, sizeof(*pda), (void**) &pda);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
  }
 
  status = cudaHostAlloc((void**) &pdb, (dim * dim * sizeof(*psb)), cudaHostAllocDefault);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (B)\n");
      return EXIT_FAILURE;
  }

   status = cudaHostAlloc((void**) &pdc, dim * dim * sizeof(*psc), cudaHostAllocMapped);
if (status != cudaSuccess)
  printf("Error allocating pinned host memory\n");
 


  /* Initialize the device matrices with the host matrices */
  
  status = cublasSetVector(dim*dim, sizeof(*psa), psa, 1, pda, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device access error (write A)\n");
      return EXIT_FAILURE;
  }
 

cudaHostGetDevicePointer(&pdc, psc_GPU, 0);


  
 
  /* Clear last error */
  cublasGetError();
  clock_gettime(CLOCK_MONOTONIC, &t1); 
  status = cudaMemcpy(pdb, psb, dim * dim * sizeof(*psb), cudaMemcpyHostToDevice);
  status = cudaMemcpy(pda, psa, dim * dim * sizeof(*psa), cudaMemcpyHostToDevice);

if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device access error (write B)\n");
      printf("%d\n",status );
      return EXIT_FAILURE;
  }

  /* Performs operation using cublas */
  cublasDgemm('n', 'n', dim, dim, dim, alpha, pda, dim, pdb, dim, beta, pdc, dim);

  status = cudaMemcpy(psc_GPU, pdc, dim * dim * sizeof(*pdc), cudaMemcpyDeviceToHost);



  clock_gettime(CLOCK_MONOTONIC, &t2); 
  deltaT = timeDiff(&t1, &t2);
  printf(" *** Elapsed Time [CUBLAS] = %6.4f secs *** \n", deltaT);
  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
  }

for (int i = 0; i < dim; ++i)
 {
 for (int j = 0; j < dim; j++)
    {
    	//printf("%lf",psc[(dim*i) + j] );
    	//printf("---- %lf\n",psc_GPU[(dim*i) + j] );
       if (psc[(dim*i) + j] != psc_GPU[(dim*i) + j])
       {
           printf("ERROR!!!\n");
           exit(1);
         }
   }   
 }
 printf("\nOUTPUT IS THE SAME\n");

}


