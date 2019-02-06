#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <memory.h>
#include <math.h>
#include <stdbool.h>


void matrix_Print(double **A, int n) {
    printf("\n");
    for (int l = 0; l < n; l++) {
        for (int i = 0; i < n; i++) {
            printf("|%12.5e|   ", A[l][i]);
        }
        printf("\n");
    }

}


void matrix_Powers(double **a, double **matC, int num_squares, int n) {


    double **matD = calloc(n, sizeof(double *));
    double **matE = calloc(n, sizeof(double *));

    for (int i = 0; i < n; i++) {
        matD[i] = calloc(n, sizeof(double));
        matE[i] = calloc(n, sizeof(double));
        memcpy(matE[i], a[i], n * sizeof(double *)); // COPY THE INPUT MATRIX
        memcpy(matD[i], a[i], n * sizeof(double *)); // COPY THE INPUT MATRIX
    }

    for (int p = 0; p < num_squares - 1; p++) {
        for (int l = 0; l < n; l++) {
            memset(matC[l], 0, n * sizeof(double *)); // COPY PRODUCT TO OUTPUT MATRIX
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    matC[i][j] += matE[i][k] * matD[k][j];
                }
            }
        }


        for (int i = 0; i < n; i++) {
            memcpy(matE[i], matC[i], n * sizeof(double *)); // COPY THE INPUT MATRIX
        }
    }

    free(matE);
    free(matD);

}

double ***get_Matrix_Powers(double **a, int n) {
    double ***Tpowers = (double ***) malloc(11 * sizeof(double **));

    if (Tpowers == NULL) {
        fprintf(stderr, "Out of memory");
        exit(0);
    }

    for (int i = 0; i < 11; i++) {
        Tpowers[i] = (double **) malloc(n * sizeof(double *));
        if (Tpowers[i] == NULL) {
            fprintf(stderr, "Out of memory");
            exit(0);
        }

        for (int j = 0; j < n; j++) {
            Tpowers[i][j] = (double *) malloc(n * sizeof(double));
            if (Tpowers[i][j] == NULL) {
                fprintf(stderr, "Out of memory");
                exit(0);
            }
        }
    }


    // assign values to allocated memory
    for (int i = 0; i < 11; i += 2) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                Tpowers[i][j][k] = a[j][k];  // Set power matrices to original matrix
            }
        }
        matrix_Powers(Tpowers[i], Tpowers[i], i, n);
    }


    // print the 3D array
    for (int i = 0; i < 11; i += 2) {
        printf("T_POWERS[%d] \n", i);
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++)
                printf("%fl ", Tpowers[i][j][k]);

            printf("\n");
        }
        printf("\n");
    }

    // deallocate memory
//    for (int i = 0; i < 11; i++)
//    {
//        for (int j = 0; j < n; j++)
//            free(Tpowers[i][j]);
//
//        free(Tpowers[i]);
//    }
//    free(Tpowers);


    return Tpowers;
}


// FIND THE INVERSE OF A MATRIX - NEEDED FOR MATRIX DIVISION, SOURCE:  http://www.sourcecodesworld.com/source/show.asp?ScriptID=1086.
void InverseOfMatrix(double **A, double **I, int n) {
    int i, j, k;
    double temp;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (i == j)
                I[i][j] = 1;
            else
                I[i][j] = 0;

            for (k = 0; k < n; k++)
    {
        temp = A[k][k];

        for (j = 0; j < n; j++)
        {
            A[k][j] /= temp;
            I[k][j] /= temp;
        }
        for (i = 0; i < n; i++)
        {
            temp = A[i][k];
            for (j = 0; j < n; j++)
            {
                if (i == k)
                    break;
                A[i][j] -= A[k][j] * temp;
                I[i][j] -= I[k][j] * temp;
            }
        }
    }
}

void matrix_Subtract(double **a, double **b, double **c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
}


void matrixAdd(double **a, double **b, double **c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}


void multiplyMatrices(double **first, double **second, double **product_matrix, int n) {
    int i, j, k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            product_matrix[i][j] = 0;
        }
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            for (k = 0; k < n; ++k) {
                product_matrix[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}


//FUNCTION TO CONSTRUCT AN IDENTITY MATRIX:
void set_Identity(double **i_matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                i_matrix[i][j] = 1;
            } else {
                i_matrix[i][j] = 0;
            }
        }
    }
}


// FUNCTION TO SCALE A MATRIX
void matrix_Scale(double **a, double **scaled, double scale, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scaled[i][j] = a[i][j] * scale;
        }
    }
}


void matrix_Copy(double **destmat, double **srcmat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            destmat[i][j] = srcmat[i][j];
        }
    }
}


// COMPUTE THE ABSOLUTE OF A MATRIX
void matrix_Absolute(double **a, double **b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[i][j] = fabs(a[i][j]);
        }
    }
}


// FUNCTION TO FIND ONE-NORM --> [||A||(1)] OF INPUT MATRIX
double calculate_one_norm(double **a, int n) {
    double max = -DBL_MAX;
    double count;
    for (int i = 0; i < n; i++) {
        count = 0;
        for (int j = 0; j < n; j++) {
            count += a[j][i];
        }
        if (count > max) {
            max = count;
        };
    }
    printf("The one-norm of the matrix is: %7.1f \n", max);
    return max;
}


// COMPUTING OPTIMAL PARAMETERS
double ell(double **a, double coeff, int m_val, int n) {
    double alpha, o, p, norm_one, norm_two, output;

    double **f = calloc(n, sizeof(double *));

    for (int i = 0; i < n; i++) {
        f[i] = calloc(n, sizeof(double));
    }

    matrix_Absolute(a, f, n);
    o = (1.0 / (2 * m_val + 1));
    p = pow(coeff, o);
    matrix_Scale(f, f, p, n);
    matrix_Powers(f, f, (2 * m_val + 1), n);

    norm_one = calculate_one_norm(a, n);
    norm_two = calculate_one_norm(f, n);
    alpha = norm_two / norm_one;
    output = fmax(ceil(log2(2 * alpha / nextafterf(1.0, INFINITY) / (2 * m_val))), 0);

    printf("OUTPUT |%12.5e|\n", output);
    return output;
}



void getNumRows(int n){
    FILE *file;
    int ch;

    file = fopen("/home/benedict/read.txt", "r");

    while(!feof(file))
    {
        ch = fgetc(file);
        if(ch == '\n')
        {
            n++;
        }
    }
    printf("%d", n);
}


// READ INPUT MATRIX FROM A TEXT FILE
void loadMatrix(double **a, int n) {

    FILE *file;

    file = fopen("/home/benedict/read.txt", "r");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!fscanf(file, "%lf", &a[i][j]))
                break;
            printf("%7.1f", a[i][j]); // PRINT THE MATRIX ELEMENT
        }
        printf("\n");
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
    int i, n;
    double **a;

    n = 3;
    a = calloc(n, sizeof(double *));
    for (i = 0; i < n; ++i) {
        a[i] = calloc(n, sizeof(double));
    }

    getNumRows(n);
    loadMatrix(a, n);
    calculate_one_norm(a, n);

    // theta for m=1 -> m=13
    double theta[5] = {1.495585217958292e-002, 2.539398330063230e-001,
                       9.504178996162932e-001, 2.097847961257068e+000,
                       5.371920351148152e+000};

    double error_coefficients[5] = {1 / 100800.0, 1 / 10059033600.0, 1 / 4487938430976000.0,
                                    1 / 113250775606021113483283660800000000.0,
                                    1 / 113250775606021113483283660800000000.0};

//    // IF STATEMENT

    double **identity;
    double **U;
    double **V;
    double **temp;
    double **temp_2;
    double **L;

    identity = calloc(n, sizeof(double *));
    U = calloc(n, sizeof(double *));
    V = calloc(n, sizeof(double *));
    temp = calloc(n, sizeof(double *));
    temp_2 = calloc(n, sizeof(double *));
    L = calloc(n, sizeof(double *));

    for (int g = 0; g < n; ++g) {

        identity[g] = calloc(n, sizeof(double));
        U[g] = calloc(n, sizeof(double));
        V[g] = calloc(n, sizeof(double));
        temp[g] = calloc(n, sizeof(double));
        temp_2[g] = calloc(n, sizeof(double));
        L[g] = calloc(n, sizeof(double));
    }


    // --------------------------------------------------------

    double ***Tpowers = get_Matrix_Powers(a, n);

    // ----------------------------------------------------------

    printf("Norm for A2 \n");
    double d2 = pow(calculate_one_norm(Tpowers[2], n), (1.0 / 2));

    printf("Norm for A4 \n");
    double d4 = pow(calculate_one_norm(Tpowers[4], n), (1.0 / 4));

    printf("Norm for A6 \n");
    double d6 = pow(calculate_one_norm(Tpowers[6], n), (1.0 / 6));
    double eta1 = fmax(d4, d6);

    int m_val;

    if (eta1 <= theta[1] && ell(a, error_coefficients[1], 3, n) == 0.0) {
        m_val = 3;

    }
    if (eta1 <= theta[2] && ell(a, error_coefficients[2], 5, n) == 0.0) {
        m_val = 5;

    }

    //COMPUTE MATRIX POWER EXPLICITLY - NEED TO LOOK AT normAM
    printf("Norm for A8 \n");
    double d8 = pow(calculate_one_norm(Tpowers[8], n), (1.0 / 8));

    double eta3 = fmax(d6, d8);

    if (eta3 <= theta[3] && ell(a, error_coefficients[3], 7, n) == 0.0) {
        m_val = 7;
    }

    if (eta3 <= theta[4] && ell(a, error_coefficients[4], 0, n) == 0.0) {
        m_val = 9;
    }

    // COMPUTE MATRIX POWER EXPLICITLY - NEED TO LOOK AT normAM
    printf("Norm for A10 \n");
    double d10 = pow(calculate_one_norm(Tpowers[10], n), (1.0 / 10));

    double eta4 = fmax(d8, d10);
    double eta5 = fmin(eta3, eta4);

    double s = fmax(ceil(log2(eta5 / theta[4])), 0);

    printf("s is: %f", s);
    printf("%f", pow(2, s));
    matrix_Scale(a, a, 1 / pow(2, s), n);
    matrix_Scale(Tpowers[1], Tpowers[1], 1 / pow(2, s), n);
    s = s + ell(a, error_coefficients[4], 13, n);
    printf("S IS:  |%12.5e|\n ", s);


    if (isinf(s)) {
        // Revert to old estimate
        int exp;
        double t = frexp(calculate_one_norm(a, n) / theta[4], &exp);
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
        matrix_Scale(Tpowers[2], Tpowers[2], multiplier, n);

        multiplier = 1.0 / pow(2, (s * 4));
        matrix_Scale(Tpowers[4], Tpowers[4], multiplier, n);

        multiplier = 1.0 / pow(2, (s * 6));
        matrix_Scale(Tpowers[6], Tpowers[6], multiplier, n);
    }

    set_Identity(identity, n);






    // PADE APPROXIMATION:



    double c[15] = {1};
    get_pade_coefficients(c, m_val);
    set_Identity(identity, n);


    if (m_val == 3 || m_val == 5 || m_val == 7 || m_val == 9) {

        int strt = sizeof(Tpowers) + 2;
        for (int k = strt; k < m_val - 1; k += 2) {
            multiplyMatrices(Tpowers[k - 2], Tpowers[2], Tpowers[k], n);
        }

        matrix_Scale(identity, U, c[1], n);
        matrix_Scale(identity, V, c[0], n);

        for (int j = m_val; j > 3; j -= 2) {

            matrix_Scale(Tpowers[j - 1], temp, c[j + 1], n);
            matrixAdd(U, temp, U, n);

            matrix_Scale(Tpowers[j - 1], temp, c[j], n);
            matrixAdd(V, temp, V, n);
        }
        multiplyMatrices(a, U, temp, n);
        memcpy(U, temp, n * n * sizeof(double));
    }

    if (m_val == 13) {

        printf("THIS IS ACIVATED");


        // CALCUATE U PSEUDOCODE:

        matrix_Scale(Tpowers[6], temp, c[13], n);
        matrixAdd(temp, temp_2, temp_2, n);

        printf("PRINTING U");
        matrix_Print(temp_2, n);

        matrix_Scale(Tpowers[4], temp, c[11], n);
        matrixAdd(temp, temp_2, temp_2, n);

        printf("PRINTING U");
        matrix_Print(temp_2, n);


        matrix_Scale(Tpowers[2], temp, c[9], n);
        matrixAdd(temp, temp_2, temp_2, n);

        printf("PRINTING U");
        matrix_Print(temp_2, n);


        multiplyMatrices(temp_2, Tpowers[6], temp, n);

        printf("PRINTING U");
        matrix_Print(temp, n);



        // ------------------------------------------

        matrix_Scale(Tpowers[6], temp_2, c[7], n);
        matrixAdd(temp, temp_2, temp, n);

        printf("PRINTING U");
        matrix_Print(temp, n);

        matrix_Scale(Tpowers[4], temp_2, c[5], n);
        matrixAdd(temp, temp_2, temp, n);
        printf("PRINTING U");
        matrix_Print(temp, n);

        matrix_Scale(Tpowers[2], temp_2, c[3], n);
        matrixAdd(temp, temp_2, temp, n);
        printf("PRINTING U");
        matrix_Print(temp, n);

        matrix_Scale(identity, temp_2, c[1], n);
        matrixAdd(temp, temp_2, temp, n);
        printf("PRINTING U");
        matrix_Print(temp, n);

        matrix_Print(a, n);

        multiplyMatrices(a, temp, U, n);
        printf("PRINTING U");
        matrix_Print(U, n);





        // CALCULATE V PSEUDOCODE:

        matrix_Scale(Tpowers[6], temp, c[12], n);
        printf("%f", c[12]);
        printf("PRINTING V");
        matrix_Print(temp, n);


        matrix_Scale(Tpowers[4], temp_2, c[10], n);
        matrixAdd(temp, temp_2, temp_2, n);
        printf("PRINTING V");
        matrix_Print(temp_2, n);

        matrix_Scale(Tpowers[2], temp, c[8], n);
        matrixAdd(temp, temp_2, temp_2, n);
        printf("PRINTING V");
        matrix_Print(temp_2, n);

        multiplyMatrices(temp_2, Tpowers[6], temp, n);

        printf("PRINTING V");
        matrix_Print(temp, n);
        // ------------------------------------------


        matrix_Scale(Tpowers[6], temp_2, c[6], n);
        matrixAdd(temp, temp_2, temp, n);

        matrix_Scale(Tpowers[4], temp_2, c[4], n);
        matrixAdd(temp, temp_2, temp, n);

        matrix_Scale(Tpowers[2], temp_2, c[2], n);
        matrixAdd(temp, temp_2, temp, n);

        matrix_Scale(identity, temp_2, c[0], n);
        matrixAdd(temp, temp_2, V, n);
        printf("PRINTING V");
        matrix_Print(V, n);


        // CALCULATE F:

        matrix_Print(temp_2, n);
        matrix_Subtract(V, U, V, n);
        matrix_Print(V, n);
        matrix_Scale(U, U, 2, n);
        printf("2*U");
        matrix_Print(U, n);

        printf("MATRIX INVERSE");
        InverseOfMatrix(V, temp_2, n);
        matrix_Print(temp_2, n);

        printf("F");
        multiplyMatrices(U, temp_2, temp, n);
        matrixAdd(temp, identity, temp, n);
        matrix_Print(temp, n);




        // NOW PERFROM THE SQUARING PHASE
        printf("%f", s);
        for (int k = 0; k < s; k++) {
            matrix_Powers(temp, temp, 2, n); // s is previously computed matrix power. --> OUTPUT FOR MATRIX EXPONENTIAL
        }


        matrix_Print(temp, n);


    }

    return 0;
}

// Calculate the one norm of a matrix
// Take the MAX of the column sums












