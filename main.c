#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <memory.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <cblas.h>


void matrix_Write_New(double* A, int n ){
    printf("HERE");
    FILE *f;
    f = fopen("/home/benedict/myFile2.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    /* print integers and floats */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {

            if(i == n-1){
                if(A[(n*j) + i] == INFINITY){
                    fprintf(f, "Inf");
                } else {
                    fprintf(f, "%.5g", A[(n*j) + i]);
                }
            }

            else {
                if(A[(n*j) + i] == INFINITY){
                    fprintf(f, "Inf ");
                } else {
                    fprintf(f, "%.5g ", A[(n*j) + i]);
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
            printf("|%20.15e|   ", A[(n*l) + i]);
        }
        printf("\n");
    }
}


void matrix_Copy_New(double *destination, double *source, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            destination[(n*i) + j] = source[(n*i) + j];
        }
    }
}


void matrix_Powers_New(double *A, double *C, double*D, double* E, int num_squares, int n){ // CONVERT TO SQUARING METHOD

    matrix_Copy_New(D, A, n);
    matrix_Copy_New(E, A, n);

    for (int p = 0; p < num_squares - 1; p++) {

    memset(C, 0, n *n* sizeof(double));


        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[(n*i) + j] += E[(n*i) + k] * D[(n*k) + j];
                }
            }

        }

        memcpy(E, C, n *n* sizeof(double)); // COPY THE INPUT MATRIX
    }

}



double ** get_Matrix_Powers_New(double* A, double* B, double* C, int n) {       // CURRENT

    double **Tpowers = (double **) malloc(11 * sizeof(double *));

    for (int i = 0; i < 11; i++) {
        Tpowers[i] = malloc(n*n* sizeof(double));
    }

    matrix_Copy_New(Tpowers[1], A, n);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[1], n, Tpowers[1], n, 1, Tpowers[2], n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[2], n, Tpowers[2], n, 1, Tpowers[4], n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[2], n, Tpowers[4], n, 1, Tpowers[6], n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[4], n, Tpowers[4], n, 1, Tpowers[8], n);

    return Tpowers;
}


// FIND THE INVERSE OF A MATRIX - NEEDED FOR MATRIX DIVISION, SOURCE:  http://www.sourcecodesworld.com/source/show.asp?ScriptID=1086.
void InverseOfMatrix_New(double *A, double *inv_New, int n) {
    int i, j, k;
    double temp;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (i == j)
                inv_New[(n*i)+j] = 1;
            else
                inv_New[(n*i)+j] = 0;

    for (k = 0; k < n; k++) {
        temp = A[(n*k) + k];

        for (j = 0; j < n; j++) {
            A[(n*k)+j] /= temp;
            inv_New[(n*k)+j] /= temp;
        }
        for (i = 0; i < n; i++) {
            temp = A[(n*i) + k];
            for (j = 0; j < n; j++) {
                if (i == k)
                    break;
                A[(n*i) + j] -= A[(n*k) + j] * temp;
                inv_New[(n*i) + j] -= inv_New[(n*k) + j] * temp;
            }
        }
    }
}


void matrix_Subtract_New(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n*i) + j] = a[(n*i) + j] - b[(n*i) + j];
        }
    }
}


void matrixAdd_New(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n*i) + j] = a[(n*i) + j] + b[(n*i) + j];
        }
    }
}


void set_Identity_New(double *i_matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                i_matrix[(n*i) + j] = 1;
            } else {
                i_matrix[(n*i) + j] = 0;
            }
        }
    }
}


void matrix_Scale_New(double *a, double *scaled, double scale, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scaled[(n*i) + j] = a[(n*i) + j] * scale;
        }
    }
}


void matrix_Absolute_New(double *a, double *b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[(n*i) + j] = fabs((a[(n*i) + j]));
        }
    }
}



double calculate_one_norm_New(double *A, int n) {
    printf("%d", n);
    double max = -DBL_MAX;
    double count;
    for (int i = 0; i < n; i++) {
        count = 0;
        for (int j = 0; j < n; j++) {
            count += A[(n*j) + i];
        }
        if (count > max) {
            max = count;
        };
    }
    printf("The one-norm of the matrix is: %7.1f \n", max);
    return max;
}


// COMPUTING OPTIMAL PARAMETERS
double ell(double *A, double* temp_new, double* temp_2, double* temp_3, double coeff, int m_val, int n) {

    matrix_Copy_New(temp_new, A, n);
    double alpha, o, p, norm_one, norm_two, output;

    matrix_Absolute_New(A, temp_new, n);

    o = (1.0 / (2 * m_val + 1));
    p = pow(coeff, o);

    matrix_Scale_New(temp_new, temp_new, p, n);

    norm_one = calculate_one_norm_New(A, n);
    norm_two = calculate_one_norm_New(temp_new, n);
    alpha = norm_two / norm_one;
    printf("ALPHA IS: |%12.5e|\n", alpha);
    output = fmax(ceil(log2(2 * alpha / nextafterf(1.0, INFINITY) / (2 * m_val))), 0);
    printf("OUTPUT |%12.5e|\n", output);
    return output;
}






int getNumRows(int n) {
    n = 0;
    FILE *file;
    int ch;

    file = fopen("/home/benedict/read.txt", "r");

    while (!feof(file)) {
        ch = fgetc(file);
        if (ch == '\n') {
            n++;
        }
    }

    printf("SIZE OF MATRIX IS: %d  * %d \n", n, n);
    return n;
}


// READ INPUT MATRIX FROM A TEXT FILE
void loadMatrix_New(double *a, int n) {

    FILE *file;

    file = fopen("/home/benedict/read.txt", "r");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!fscanf(file, "%lf", &a[(n*i) + j]))
                break;
            //printf("%7.1f", a[i][j]); // PRINT THE MATRIX ELEMENT
        }
        //printf("\n");
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
            return;
        }
        case 5  : {
            buf = test[1];
            return;
        }
        case 7  : {
            buf = test[2];
            return;
        }

        case 9  : {
            buf = test[3];
            return;
        }
        case 13  : {
            for (int i = 0; i < sizeof(test[4]) / sizeof(double); i++) {
                buf[i] = test[4][i];
            }
            return;
        }
        default:
            break;
    }
}

int main() {

    /* here, do your time-consuming job */

    int i, n;
    double **a;
    double *A;

    n = getNumRows(n);
    a = calloc(n, sizeof(double *));
    for (i = 0; i < n; ++i) {
        a[i] = calloc(n, sizeof(double));
    }

    A = (double *) malloc( n*n* sizeof(double) );

    loadMatrix_New(A, n);

    // theta for m=1 -> m=13
    double theta[5] = {1.495585217958292e-002, 2.539398330063230e-001,
                       9.504178996162932e-001, 2.097847961257068e+000,
                       5.371920351148152e+000};

    double error_coefficients[5] = {1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
                                    1 / 113250775606021113483283660800000000.0,
                                    1 / 113250775606021113483283660800000000.0};


    double* identity_new;
    double* U_new;
    double* V_new;
    double* temp_new;
    double* temp_2_new;
    double* temp_3_new;

    identity_new = (double *) malloc( n*n* sizeof(double) );
    U_new = (double *) malloc( n*n* sizeof(double) );
    V_new = (double *) malloc( n*n* sizeof(double) );
    temp_new = (double *) malloc( n*n* sizeof(double) );
    temp_2_new = (double *) malloc( n*n* sizeof(double) );
    temp_3_new = (double *) malloc( n*n* sizeof(double) );


    double** Tpowers = get_Matrix_Powers_New(A, temp_new, temp_2_new, n);

    clock_t begin = clock();


    printf("Norm for A2 \n");
    double d2 = pow(calculate_one_norm_New(Tpowers[2], n), (1.0 / 2));

    printf("Norm for A4 \n");
    double d4 = pow(calculate_one_norm_New(Tpowers[4], n), (1.0 / 4));
    printf("%f", d4);

    printf("Norm for A6 \n");
    double d6 = pow(calculate_one_norm_New(Tpowers[6], n), (1.0 / 6));
    printf("%f", d6);
    double eta1 = fmax(d4, d6);







    int m_val = 0;

    if (eta1 <= theta[1] && ell(A, temp_new, temp_2_new, temp_3_new, error_coefficients[1], 3, n) == 0.0) {
        m_val = 3;

    }
    if (eta1 <= theta[2] && ell(A, temp_new, temp_2_new,temp_3_new, error_coefficients[2], 5, n) == 0.0) {
        m_val = 5;

    }


    //COMPUTE MATRIX POWER EXPLICITLY - NEED TO LOOK AT normAM
    printf("Norm for A8 \n");
    double d8 = pow(calculate_one_norm_New(Tpowers[8], n), (1.0 / 8));

    double eta3 = fmax(d6, d8);

    if (eta3 <= theta[3] && ell(A, temp_new, temp_2_new, temp_3_new, error_coefficients[3], 7, n) == 0.0) {
        m_val = 7;
    }

    if (eta3 <= theta[4] && ell(A,temp_new, temp_2_new, temp_3_new, error_coefficients[4], 0, n) == 0.0) {
        m_val = 9;
    }

    // COMPUTE MATRIX POWER EXPLICITLY - NEED TO LOOK AT normAM
    printf("Norm for A10 \n"); //- DO I EVEN NEED THIS???
    double d10 = pow(calculate_one_norm_New(Tpowers[10], n), (1.0 / 10));

    double eta4 = fmax(d8, d10);
    double eta5 = fmin(eta3, eta4);
    double s = fmax(ceil(log2(eta5 / theta[4])), 0);


    matrix_Scale_New(Tpowers[1], Tpowers[1], 1 / pow(2, s), n);;
    matrix_Scale_New(A, A, 1 / pow(2, s), n);
    printf("uyugihoj\n");
    s = s + ell(A,temp_new, temp_2_new, temp_3_new, error_coefficients[4], 13, n);
    printf("S IS:  |%12.5e|\n ", s);


    if (isinf(s)) {
        // Revert to old estimate
        int exp;
        double t = frexp(calculate_one_norm_New(A, n) / theta[4], &exp);
        s = s - (t == 0.5);
    } else {
        m_val = 13;
    }

    printf("HERE");


    double arr[15] = {1};
    int m = 13;
    get_pade_coefficients(arr, m);
    printf("\nThe Pade coefficients for m = %d: \n", m);
    for (int j = 0; j < 14; j++) {
        printf("%7.1f\n", arr[j]);
    }


    if ((int) s != 0) {
        double multiplier;

        multiplier = 1.0 / pow(2, (s * 2));
        matrix_Scale_New(Tpowers[2], Tpowers[2], multiplier, n);

        multiplier = 1.0 / pow(2, (s * 4));
        matrix_Scale_New(Tpowers[4], Tpowers[4], multiplier, n);

        multiplier = 1.0 / pow(2, (s * 6));
        matrix_Scale_New(Tpowers[6], Tpowers[6], multiplier, n);
    }


    // PADE APPROXIMATION:
    double c[15] = {1};
    get_pade_coefficients(c, m_val);
    set_Identity_New(identity_new, n);


    if (m_val == 3 || m_val == 5 || m_val == 7 || m_val == 9) {
        printf("UNSEEN HAS BEEN ACTIVATED");
        int strt = sizeof(Tpowers) + 2;
        for (int k = strt; k < m_val - 1; k += 2) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[k-2], n, Tpowers[2], n, 1, Tpowers[k], n);

        }

        matrix_Scale_New(identity_new, U_new, c[1], n);
        matrix_Scale_New(identity_new, V_new, c[0], n);

        for (int j = m_val; j > n; j -= 2) {

            matrix_Scale_New(Tpowers[j - 1], temp_new, c[j + 1], n);
            matrixAdd_New(U_new, temp_new, U_new, n);

            matrix_Scale_New(Tpowers[j - 1], temp_new, c[j], n);
            matrixAdd_New(V_new, temp_new, V_new, n);
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, U_new, n, 1, temp_new, n);
        memcpy(U_new, temp_new, n * n * sizeof(double));
    }

    if (m_val == 13) {

        // CALCUATE U PSEUDOCODE:

        matrix_Scale_New(Tpowers[6], temp_new, c[13], n);

        memset(temp_2_new, 0, n *n*sizeof(double));

        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        matrix_Scale_New(Tpowers[4], temp_new, c[11], n);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[2], temp_new, c[9], n);

        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);


        memset(temp_new, 0, n *n*sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, temp_2_new, n, Tpowers[6], n, 1, temp_new, n);

        matrix_Scale_New(Tpowers[6], temp_2_new, c[7], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        matrix_Scale_New(Tpowers[4], temp_2_new, c[5], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        matrix_Scale_New(Tpowers[2], temp_2_new, c[3], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        set_Identity_New(identity_new, n);

        matrix_Scale_New(identity_new, temp_2_new, c[1], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(U_new, 0, n *n*sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, temp_new, n, 1, U_new, n);





        // CALCULATE V PSEUDOCODE:
        memset(temp_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[6], temp_new, c[12], n);

        memset(temp_2_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[4], temp_2_new, c[10], n);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[2], temp_new, c[8], n);
        matrixAdd_New(temp_new, temp_2_new, temp_2_new, n);

        memset(temp_new, 0, n *n*sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Tpowers[6], n, temp_2_new, n, 1, temp_new, n);

        memset(temp_2_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[6], temp_2_new, c[6], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[4], temp_2_new, c[4], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n *n*sizeof(double));
        matrix_Scale_New(Tpowers[2], temp_2_new, c[2], n);
        matrixAdd_New(temp_new, temp_2_new, temp_new, n);

        memset(temp_2_new, 0, n *n*sizeof(double));
        set_Identity_New(identity_new, n);
        matrix_Scale_New(identity_new, temp_2_new, c[0], n);
        matrixAdd_New(temp_new, temp_2_new, V_new, n);



        // CALCULATE F:
        matrix_Subtract_New(V_new, U_new, V_new, n);
        matrix_Scale_New(U_new, U_new, 2, n);
        memset(temp_2_new, 0, n *n*sizeof(double));
        InverseOfMatrix_New(V_new, temp_2_new, n);

        memset(temp_new, 0, n *n*sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, U_new, n, temp_2_new, n, 1, temp_new, n);

        matrixAdd_New(temp_new, identity_new, temp_new, n);

        // NOW PERFROM THE SQUARING PHASE

        for (int k = 0; k < s; k++) { // MORE EFFICIENT WAY TO DO SQUARING --> THE SLOWEST PART
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, temp_new, n, temp_new, n, 1, temp_2_new, n);
            memcpy(temp_new, temp_2_new, n * n*sizeof(double));
            memset(temp_2_new, 0, n *n*sizeof(double));
        }

        // WRITE THE MATRIX EXPO TO A TEXT FILE
        printf("DONE \n");
        matrix_Write_New(temp_new, n);
 }

    return 0;
}