#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <memory.h>
#include <cblas.h>
#include <cuComplex.h>
#include "expm.h"


// Find the system hamiltonian for a single excitation subspace with XX-coupling only

// XX couplings explained:

// Between a pair of qubits that are coupled, there is a tensor product between two standard X pauli matrices
// These terms are sumed over every possible pair of coupled qubits, that can have different strengths of the coupling
// For example in a spin chain with qubits (1,...,n) and nearest neignour coupli9ng such that all pairs i and i+1 are
// coupled, the XX coupling becomes: ∑i=1N−1JiI⊗(i−1)⊗X⊗X⊗I⊗(n−i−1) for real parameters Ji and the YY coupling would be
// the same but replacing the X pauli gate with the Y pauli gate.


// If we have a single excitation subspace:
// [1] Generate the system hamiltonian for the single excitation subspace with XX-couplings


// [2] Generate the control hamiltonians


// If we have more than one excitation subspace (ess):

// [1] Construct the Pauli matrices 
// [2] Generate the full system hamiltonian with XX-couplings
// --> [a] Create sparse matrix of (2^n * 2^n) where n = network size
// --> [b] Compute the X operator:
// --> X[k]X[l] = I⊗I(k times)⊗X⊗I⊗I(l-k-1 times) ⊗ X⊗I⊗I(n-l-1 times)
// --> Y[k]Y[l] = I⊗I(k times)⊗Y⊗I⊗I(l-k-1 times) ⊗ Y⊗I⊗I(n-l-1 times)
// Add them
// [3] Generate the bias control hamiltonians, one for each spin
// --> CM = I⊗I(k times)⊗Z⊗I⊗I(N-K-1 times)
// --> CM = (I⊗I(n times) - CM)/2 
// [4] Reduce the Hamiltonians to the relevant excitation sub spaces
// --> Filter out subspaces that are not relevnt to the present excitations
// [5] Sub-index the system hamiltonian
// --> 
// [6] sub-index the control hamiltionian
// --> 
// [6] Remove the full Hamiltonians to save memory

// -------------------------------------SIGNIFICANT POINT---------------------------------------------------------

// Construct the drift hamiltonian from the system and control hamiltonians
// Hd = H0 + ∑Hn*Cn






void calculate_system_propegator(cuDoubleComplex* drift_hamil, cuDoubleComplex* U, double T, int network_size){
	
	// Scale the drift hamiltonian by the time step:
	const int num_elements = network_size * network_size;
    const void *data; 
    const void *data_2;
    const int x = 1;
    const cuDoubleComplex a = make_cuDoubleComplex(0, -1);
    data = &T;
    data_2 = &a;
	cblas_zscal(num_elements, data, drift_hamil, x);
	cblas_zscal(num_elements, data_2, drift_hamil, x);
	printf(" THE PRODUCT IS: \n");
	for (int j = 0; j < network_size; j++){
		printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %.15lf ", drift_hamil[(j*network_size) + k].x );
			printf("+");
			printf(" %.15lfi ", drift_hamil[(j*network_size) + k].y );
		}
		printf("]");
		printf("\n");
	}
	
	
	// U = expm(-i*Hd*T)
	// Calculate TH 
	// Call cuda expm() here:

	expm_new(drift_hamil, U, network_size);

	printf("THE SYSTEM PROPEGATOR IS:\n");
	for (int j = 0; j < network_size; j++){
		printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %0.15lf ", U[(j*network_size) + k].x );
			printf("+");
			printf(" %0.15lfi ", U[(j*network_size) + k].y );

		}
		printf("]");
		printf("\n");
	}

	exit(0);

}



void matrixAdd_News(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c, int n) {
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            //c[(n * i) + j] = a[(n * i) + j] + b[(n * i) + j];
            c[(n * i) + j] = cuCadd(a[(n * i) + j], b[(n * i) + j]); // Complex addition
        }
    }
    
}



//CURRENT WORK
int construct_drift_hamiltonian(cuDoubleComplex** control_hamil, cuDoubleComplex* system_hamil, cuDoubleComplex* drift_hamil, double *biases, int network_size){
//memcpy(drift_hamil, system_hamil, network_size*network_size*network_size*sizeof(cuDoubleComplex*)); --> The initial hamiltonian is dependent on the user-defined topology of the network.
const int num_elements = network_size * network_size;
for(int i =0; i< network_size; i++){
	//const void *TY = malloc(1*sizeof(void));
	// Scale the contorl matrix by their respective biases 
	 //void* a = malloc(sizeof(void*));
	const int x = 1;
	const void * data;
    double b = biases[i];
    data = &b;
	//memcpy(a, (void)5, 1);
	//a[0] = 6;
	cblas_zscal(num_elements, data, control_hamil[i], x);
	matrixAdd_News(drift_hamil, control_hamil[i], drift_hamil, network_size);
	printf(" THE PRODUCT IS: \n");
	   for (int j = 0; j < network_size; j++){
	   	printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %0.2lf ", control_hamil[i][(j*network_size) + k].x );
			printf("+");
			printf(" %0.2lfi ", control_hamil[i][(j*network_size) + k].y );
		}
		printf("]");
		printf("\n");
	}
	
	// add sum to drift hamiltonian
	}
	
	printf(" THE ADDITION IS: \n");
	for (int j = 0; j < network_size; j++){
		printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %0.2lf ", drift_hamil[(j*network_size) + k].x );
			printf("+");
			printf(" %0.2lfi ", drift_hamil[(j*network_size) + k].y );
		}
		printf("]");
		printf("\n");
	}

	return 0;
}


// AIM: U^n = expm(-ihH^nT)
// Calculate the value of U for every time step n
// The unitary propegator U = ∑ U^n + U^n-1 + ... U^0



// Single excitation subspace example:
int main(){
	

	int network_size = 5;
	//int num_excitation_subspaces = 1;
	
	// Get number of hamiltonians needed for memory allocation
	int num_hamils = 0;
	for (int c = network_size - 1; c > 0; c--)
   		num_hamils = num_hamils + c;
 

	// Generate the system Hamiltonians and add to array
	// int** HH = malloc(network_size*network_size*network_size*network_size*sizeof(int));

	cuDoubleComplex ** array = (cuDoubleComplex **)malloc(network_size*network_size*network_size*sizeof(cuDoubleComplex*));
 		for (int i = 0; i< num_hamils; i++) {
 			array[i] = (cuDoubleComplex *) malloc(network_size*network_size*sizeof(cuDoubleComplex ));
 		}


     int n = 0;
     for (int k = 0; k < num_hamils; k++)
     {
        	for (int l = k+1; l < network_size; l++)
        	{
        		cuDoubleComplex my_array[network_size*network_size];
        		memset(my_array, 0, network_size*network_size*sizeof(cuDoubleComplex));
        		my_array[(network_size*l) + k].x = 1;
        		my_array[(network_size*k) + l].x = 1;
        		memcpy(array[n], my_array, network_size*network_size*sizeof(cuDoubleComplex));
        		n = n + 1;
        	}
        }

    // Print the System Hamiltonians
    printf("\nTHE SYSTEM HAMILTONIANS [SEP]: \n");
	for (int n = 0;  n< num_hamils; n++){
		printf("\n");
	    for (int j = 0; j < network_size; j++){
	    	printf("[");
			for (int k = 0; k < network_size; k++){
				printf(" %0.2lf ", array[n][(j*network_size) + k].x );
				printf("+");
				printf(" %0.2lfi ", array[n][(j*network_size) + k].y );
			}
			printf("]");
			printf("\n");
			}
		printf("\n");
	}


	// Condense matrix adding subspaces corresponding to connetions in the network

	cuDoubleComplex * network_hamil = (cuDoubleComplex *)malloc(network_size*network_size*sizeof(cuDoubleComplex));
	bool dummy_connectivity[] = {1,1,0,1,0,1,0,1,0,0}; 
	for(int l =0; l < num_hamils; l++){
		if(dummy_connectivity[l] == 1){
			matrixAdd_News(network_hamil, array[l], network_hamil, network_size);
		}
	}


	// Print the network hamiltonian
 	printf("\nTHE NETWORK HAMILTONIANS [SEP]: \n");

	    for (int j = 0; j < network_size; j++){
	    	printf("[");
			for (int k = 0; k < network_size; k++){
				printf(" %0.2lf ", network_hamil[(j*network_size) + k].x );
				printf("+");
				printf(" %0.2lfi ", network_hamil[(j*network_size) + k].y );
			}
			printf("]");
			printf("\n");
			}



	// Generate the control Hamiltonians and add to array

	cuDoubleComplex ** CC = (cuDoubleComplex **)malloc(network_size*network_size*network_size*sizeof(cuDoubleComplex*));
 		for (int i = 0; i< network_size; i++) {
 			CC[i] = (cuDoubleComplex *) malloc(network_size*network_size*sizeof(cuDoubleComplex ));
 		}


 	for (int k = 0; k < network_size; k++)
 	{
 		cuDoubleComplex my_array[network_size*network_size];
        memset(my_array, 0, network_size*network_size*sizeof(cuDoubleComplex));
        my_array[(network_size*k) + k].x =  1;
        memcpy(CC[k], my_array, network_size*network_size*sizeof(cuDoubleComplex));
 	}


 	// Print the Control Hamiltonians
 	printf("\nTHE CONTROL HAMILTONIANS [SEP]: \n");
	for (int n = 0;  n< network_size; n++){
		printf("\n");
	    for (int j = 0; j < network_size; j++){
	    	printf("[");
			for (int k = 0; k < network_size; k++){
				printf(" %0.2lf ", CC[n][(j*network_size) + k].x );
				printf("+");
				printf(" %0.2lfi ", CC[n][(j*network_size) + k].y );
			}
			printf("]");
			printf("\n");
			}
		printf("\n");
	}


	// Allocate mmemory for the drift hamiltionian
	cuDoubleComplex* drift_hamil = (cuDoubleComplex *)malloc(network_size * network_size * network_size * sizeof(cuDoubleComplex));
	memcpy(drift_hamil, network_hamil, network_size*network_size*sizeof(cuDoubleComplex));


 	// Create random control biases that will be selected by the bayesian optimiser
 	double* biases = (double *)malloc((network_size + 1) * sizeof(double));
 	printf("THE CONTROL BIASES ARE: \n");
 	printf("[");
 	for (int i = 0; i < network_size+1; i++)
 	{
 		//biases[i] = ((double)rand()/(double)(RAND_MAX)) * 10;
 		// TEMP FOR MATLAB TESTS:
 		biases[i] = ((double)i/(double)(RAND_MAX)) * 10;
 		printf(" %0.2lf ", biases[i]);
 	}
 	printf("] \n");
 	
 	
 	
 	construct_drift_hamiltonian(CC, drift_hamil, drift_hamil, biases, network_size);
 	
 	// Allocate memory for the system propegator
 	cuDoubleComplex* U = (cuDoubleComplex *)malloc(network_size * network_size * network_size * sizeof(cuDoubleComplex));
 	double T = 4.83645;

 	
 	
 	calculate_system_propegator(drift_hamil, U, T, network_size);






	// Pauli Matrices
	// cuDoubleComplex X[] = {0, 1, 1, 0};
	// cuDoubleComplex complex Y[] = {0 + 0*I, 0 - 1*I, 0 + 1*I, 0 + 0*I};
	// cuDoubleComplex Z[] = {1, 0, 0 ,-1};
	
	for (int i = 0; i < network_size; i++)
	{
		for (int l = i; l < network_size; l++)
		{
			
		}
	}

	return 0;
}









