// Header files for interface:
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <memory.h>
#include <math.h>
#include <stdbool.h>
#include "sys/time.h"
#include <time.h>
#include <complex.h>
#include <lapacke.h>


// *** Serial Matrix exponential program acting on matrices of type : [Double Complex] - 02/04/2019 ***

// Extern -> Function is assumed to be available somewhere else (in the openblas library)
extern void zgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);


void matrix_complex_write(double complex *A, int n) {
   FILE *f;
    f = fopen("/home/benedict/Desktop/Quantum-Simulator/CUDA_INPUT.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {

            if (i == n - 1) {
                if (__real__(A[(n * j) + i]) == INFINITY) {
                    fprintf(f, "Inf");
                } else {
                    fprintf(f, "%lf", __real__(A[(j*n) + i]));
                    fprintf(f, "+");
                    fprintf(f, "%lfi ", __imag__(A[(j*n) + i]));
                }
            } else {
                if (__real__(A[(n * j) + i]) == INFINITY) {
                    fprintf(f, "Inf ");
                } else {
                        fprintf(f, "%lf", __real__(A[(j*n) + i]));
                        fprintf(f, "+");
                        fprintf(f, "%lfi ", __imag__(A[(j*n) + i]));
                    }
                }
            }
            fprintf(f, "\n");
        }
    }


void matrix_complex_print(double complex* A, int network_size){
	for (int j = 0; j < network_size; j++){
		printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %.15lf ", __real__(A[(j*network_size) + k]));
			printf("+");
			printf(" %.15lfi ", __imag__(A[(j*network_size) + k]));
		}
		printf("]");
		printf("\n");
	}
}

void matrix_Square(double complex *A, double complex *C, int n, int s){

    char ta = 'N';
    double complex alpha = 1;
    double complex beta = 0;

	for (int k = 0; k < s; k++) {
        zgemm_(&ta,&ta , &n, &n, &n, (double*)&alpha, (double*) A, &n, (double*) A, &n,(double*) &beta, (double*) C, &n);
        memcpy(A, C, n*n*sizeof(complex double));
    }
}


void matrix_Multiply_serial(double complex *A, double complex *B, double complex *C,  int n, double* multiply_total_time){

    char ta = 'N';
    double complex alpha = 1;
    double complex beta = 0;
    zgemm_(&ta,&ta , &n, &n, &n, (double*)&alpha, (double*) B, &n, (double*) A, &n,(double*) &beta, (double*) C, &n);

}



void InverseOfMatrix_Alternative_Two(complex double* L, complex double* inverse, int n, complex double* b){ // Calculate matrix inverse through LU factorisation
    
    int *IPIV = malloc(n*n*sizeof(int));
    int LWORK = n*n;
    double complex* WORK = malloc(n*n*sizeof(double complex));// = new double[LWORK];
    int INFO;
    zgetrf_(&n,&n,L,&n,IPIV,&INFO);
    zgetri_(&n,L,&n,IPIV,WORK,&LWORK,&INFO);
}


void matrix_Subtract_New(double complex *a, double complex *b, double complex *c, int n) { // PARALLEL CANDIDATE

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = a[(n * i) + j] - b[(n * i) + j]; // Complex subtraction
        }
    }
}


void matrixAdd_New(double complex *a, double complex *b, double complex *c, int n) { // PARALLEL CANDIDATE

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = a[(n * i) + j] + b[(n * i) + j]; // Complex addition
        }
    }
}


void set_Identity_New(double complex *i_matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                __real__(i_matrix[(n * i) + j]) = 1;
            } else {
                __imag__(i_matrix[(n * i) + j]) = 0;
            }
        }
    }
}



void matrix_Scale_New(double complex *a, double complex *scaled, double complex scale, int n, double* scale_total_time ) {
    clock_t scale_begin = clock();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scaled[(n * i) + j] = a[(n * i) + j]*scale; // Complex multiplication
        }
    }

    clock_t scale_end = clock();
    double time_spent = (double)(scale_end - scale_begin) / CLOCKS_PER_SEC;
    scale_total_time[0] = scale_total_time[0] + time_spent;
}


void matrix_Absolute_New(double complex *a, double complex *b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __real__(b[(n * i) + j]) = cabs((a[(n * i) + j]));
        }
    }
}


double calculate_one_norm_New_complex(double complex *A, int n) {
    double max = -DBL_MAX;
	double count;
    for (int i = 0; i < n; i++) {
        count = 0;
        for (int j = 0; j < n; j++) {
            count += cabs((A[(n * j) + i]));
        }
        if (count > max) {;
            max = count;
        };
    }
    return max;
}


// // COMPUTING OPTIMAL PARAMETERS
double ell(double complex *A, double complex *temp_new, double coeff, int m_val, int n, double* scale_total_time) {

    double norm_one, norm_two, p, alpha, output;
    memcpy(temp_new, A, n * n * sizeof(double complex));
    
    matrix_Absolute_New(A, temp_new, n);

    p = pow(coeff, (1.0 / (2 * m_val + 1)));
    
    matrix_Scale_New(temp_new, temp_new, p + 0I, n, scale_total_time);
    
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

// Dummy C function that takes the input matrix:
void expm_initialization(const void * input, void * output, int n) {
    
    clock_t begin = clock();  // Begin recording expm execution

    double complex* A;// = (double complex*) input; // Cast void pointers to complex doubles
    A = (double complex*) malloc(n*n*sizeof(double complex));
    A = (double complex*) input;

    // TOTAL COMPONENT TIME STRUCTURES:
    double* scale_total_time = (double *) malloc(1 * sizeof(double));
    double* multiply_total_time = (double *) malloc(1 * sizeof(double));

    double complex *B, *C;

    B = (double complex*) malloc(n * n * sizeof(double complex));
    C = (double complex*) malloc(n * n * sizeof(double complex));


    double theta[5] = {1.495585217958292e-002, 2.539398330063230e-001,
                       9.504178996162932e-001, 2.097847961257068e+000,
                       5.371920351148152e+000};

    double error_coefficients[5] = {1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
                                    1 / 113250775606021113483283660800000000.0,
                                    1 / 113250775606021113483283660800000000.0};

    
    struct timeval start_mem,finish_mem;
    gettimeofday(&start_mem, NULL); // Start recording                            
    // Allocate temporary arrays to hold temporary matrices used at various stages in the calculation
    double complex *identity_new;
    double complex *U_new;
    double complex *V_new;
    double complex *temp_new;
    double complex *temp_2_new;

    identity_new = (double complex *) malloc(n * n * sizeof(double complex));
    U_new = (double complex *) malloc(n * n * sizeof(double complex));
    V_new = (double complex *) malloc(n * n * sizeof(double complex));
    temp_new = (double complex *) malloc(n * n * sizeof(double complex));
    temp_2_new = (double complex *) malloc(n * n * sizeof(double complex));
    // LARGE MEMORY ALLOCATION PHASE


    double d4, d6, d8, d10, eta1, eta3, eta4, eta5, s;
    double complex **Tpowers = (double complex **) malloc(11 * sizeof(double complex *));
     for (int i = 0; i < 11; i++) {
        Tpowers[i] = (double complex *) malloc(n * n * sizeof(double complex));
    }

    memcpy(Tpowers[1], A, n * n * sizeof(double complex));

    gettimeofday(&finish_mem, NULL); // Finish recording 
    double duration_mem = ((double)(finish_mem.tv_sec-start_mem.tv_sec)*1000000 + (double)(finish_mem.tv_usec-start_mem.tv_usec)) / 1000000;
    printf("Total time: %lf", duration_mem);

    matrix_Multiply_serial(Tpowers[1], Tpowers[1], Tpowers[2], n, multiply_total_time);
    matrix_Multiply_serial(Tpowers[2], Tpowers[2], Tpowers[4], n, multiply_total_time);
    matrix_Multiply_serial(Tpowers[4], Tpowers[2], Tpowers[6], n, multiply_total_time);
    matrix_Multiply_serial(Tpowers[4], Tpowers[4], Tpowers[8], n, multiply_total_time);
    matrix_Multiply_serial(Tpowers[8], Tpowers[2], Tpowers[10], n, multiply_total_time);


    clock_t norm_begin = clock();
    d4 = pow(calculate_one_norm_New_complex(Tpowers[4], n), (1.0 / 4));
    d6 = pow(calculate_one_norm_New_complex(Tpowers[6], n), (1.0 / 6));
    d8 = pow(calculate_one_norm_New_complex(Tpowers[8], n), (1.0 / 8));
    d10 = pow(calculate_one_norm_New_complex(Tpowers[10], n), (1.0 / 10));
    clock_t norm_end = clock();
    double total_norm_time =  (double)(norm_end - norm_begin) / CLOCKS_PER_SEC;
    
    eta1 = fmax(d4, d6);
    int m_val = 0;

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
    matrix_Scale_New(A, A, (1 / pow(2, s)) + 0I , n, scale_total_time);
    s = s + ell(A, temp_new, error_coefficients[4], 13, n, scale_total_time);

    if (isinf(s)) { // Revert to old estimate
       	int exp;
        double t = frexp(calculate_one_norm_New_complex(A, n) / theta[4], &exp);
        s = s - (t == 0.5);
    } else {
        m_val = 13;
    }

    if ((int) s != 0) {	// Rescale the matrix powers array
    	
    	double complex multiplier = 0 + 0I;

      	multiplier = (1.0 / pow(2, (s * 1))) + 0I;
        matrix_Scale_New(Tpowers[1], Tpowers[1], multiplier, n, scale_total_time);
    
        multiplier = (1.0 / pow(2, (s * 2))) + 0I;
        matrix_Scale_New(Tpowers[2], Tpowers[2], multiplier, n, scale_total_time);

        multiplier = (1.0 / pow(2, (s * 4))) + 0I;
        matrix_Scale_New(Tpowers[4], Tpowers[4], multiplier, n, scale_total_time);

        multiplier = (1.0 / pow(2, (s * 6))) + 0I;
        matrix_Scale_New(Tpowers[6], Tpowers[6], multiplier, n, scale_total_time);
     
    }


   	// PADE APPROXIMATION:
    
    double c[15] = {1};
    
    get_pade_coefficients(c, m_val);
    
    set_Identity_New(identity_new, n);


    if (m_val == 3 || m_val == 5 || m_val == 7 || m_val == 9) {

    	int strt = sizeof(Tpowers) + 2;
        for (int k = strt; k < m_val - 1; k += 2) {
            matrix_Multiply_serial(Tpowers[2], Tpowers[k-2], Tpowers[k], n, multiply_total_time);

        }
        matrix_Scale_New(identity_new, U_new, c[1] + 0I, n, scale_total_time);
        matrix_Scale_New(identity_new, V_new, c[0] + 0I, n, scale_total_time);

        for (int j = m_val; j > n; j -= 2) {

            matrix_Scale_New(Tpowers[j - 1], temp_new, c[j + 1] + 0I, n, scale_total_time);
            matrixAdd_New(U_new, temp_new, U_new, n);

            matrix_Scale_New(Tpowers[j - 1], temp_new, c[j] + 0I, n, scale_total_time);
            matrixAdd_New(V_new, temp_new, V_new, n);
        }

         matrix_Multiply_serial(U_new, A, temp_new, n, multiply_total_time);
         memcpy(U_new, temp_new, n * n * sizeof(double complex));
     }

if (m_val == 13) {

        struct timeval start,finish;
        gettimeofday(&start, NULL); // Start recording

        // CALCULATE U:

        matrix_Scale_New(Tpowers[6], temp_new, c[13] + 0I, n, scale_total_time);
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);
        matrix_Scale_New(Tpowers[4], temp_new, c[11] + 0I, n, scale_total_time);        
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);
        memset(temp_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[2], temp_new, c[9] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);
        memset(temp_new, 0, n * n * sizeof(double complex));
        matrix_Multiply_serial(Tpowers[6], temp_2_new, temp_new, n, multiply_total_time);  
        matrix_Scale_New(Tpowers[6], temp_2_new, c[7] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        matrix_Scale_New(Tpowers[4], temp_2_new, c[5] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        matrix_Scale_New(Tpowers[2], temp_2_new, c[3] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        memset(identity_new, 0, n * n * sizeof(double complex));
        set_Identity_New(identity_new, n);
        matrix_Scale_New(identity_new, temp_2_new, c[1] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        memset(U_new, 0, n * n * sizeof(double complex));	// IS THIS NEEDED?
        matrix_Multiply_serial(temp_new, Tpowers[1], U_new, n, multiply_total_time);


        // CALCULATE V:

        memset(temp_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[6], temp_new, c[12] + 0I, n, scale_total_time);
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[4], temp_2_new, c[10] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);
        memset(temp_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[2], temp_new, c[8] +  0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);
        memset(temp_new, 0, n * n * sizeof(double complex));
        matrix_Multiply_serial(temp_2_new, Tpowers[6], temp_new, n, multiply_total_time);
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[6], temp_2_new,c[6] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[4], temp_2_new, c[4] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        matrix_Scale_New(Tpowers[2], temp_2_new, c[2] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        set_Identity_New(identity_new, n);
        matrix_Scale_New(identity_new, temp_2_new, c[0] + 0I, n, scale_total_time);
        matrixAdd_New(temp_new, temp_2_new, V_new, n);

        // CALCULATE V-U

        matrix_Subtract_New(V_new, U_new, V_new, n);       
        matrix_Scale_New(U_new, U_new, 2 + 0I, n, scale_total_time);  
        memset(temp_2_new, 0, n * n * sizeof(double complex));
        clock_t inverse_begin = clock();    
        InverseOfMatrix_Alternative_Two(V_new, temp_2_new, n, identity_new);
        clock_t inverse_end = clock();
        double inverse_total_time = (double)(inverse_end - inverse_begin) / CLOCKS_PER_SEC;
        memset(temp_new, 0, n * n * sizeof(double complex));
        matrix_Multiply_serial(V_new, U_new, temp_new, n, multiply_total_time);
        
       	// CALCULATE F:
        matrixAdd_New(temp_new, identity_new, temp_new, n);

       
        // SQUARE THE MATRIX S TIMES - Sequential:
        memset(output, 0, n*n*sizeof(double complex));
        if(s!= 0){
            matrix_Square(temp_new, temp_2_new, n, s);
            memcpy(output, temp_2_new, n*n*sizeof(double complex));
        }
        else
            memcpy(output, temp_new, n*n*sizeof(double complex));

        gettimeofday(&finish, NULL); // Finish recording
        double duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
        printf("Total time: %lf", duration);
     }


}

// Resources
// http://www.netlib.org/utk/people/JackDongarra/PAPERS/Factor_Inversion_Million_Matrices-iccs17.pdf
// http://mathforcollege.com/nm/mws/che/04sle/mws_che_sle_spe_luinverse.pdf
