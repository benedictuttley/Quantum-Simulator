#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <memory.h>
#include <cblas.h>
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






void calculate_system_propegator(float* drift_hamil, float* U, float T, int network_size){

	// U = expm(-i*Hd*T)
	// Call cuda expm() here:
}








int matrixAdd_New(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[(n * i) + j] = a[(n * i) + j] + b[(n * i) + j];
        }
    }
	return 0;
}






int construct_drift_hamiltonian(float** control_hamil, float** system_hamil, float* drift_hamil, float*biases, int network_size){
//memcpy(drift_hamil, system_hamil, network_size*network_size*network_size*sizeof(float*)); --> The initial hamiltonian is dependent on the user-defined topology of the network.
for(int i =0; i< network_size; i++){
	
	// Scale the contorl matrix by their respective biases 
	cblas_sscal( network_size * network_size, biases[i], control_hamil[i], 1);
	matrixAdd_New(drift_hamil, control_hamil[i], drift_hamil, network_size);
	printf(" THE PRODUCT IS: \n");
	   for (int j = 0; j < network_size; j++){
	   	printf("[");
		for (int k = 0; k < network_size; k++){
			printf(" %f ", control_hamil[i][(j*network_size) + k] );
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
			printf(" %f ", drift_hamil[(j*network_size) + k] );
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
	printstructure();
	int network_size = 5;
	int num_excitation_subspaces = 1;
	
	// Get number of hamiltonians needed for memory allocation
	int num_hamils = 0;
	for (int c = network_size - 1; c > 0; c--)
   		num_hamils = num_hamils + c;
 

	// Generate the system Hamiltonians and add to array
	// int** HH = malloc(network_size*network_size*network_size*network_size*sizeof(int));

	float ** array = (float **)malloc(network_size*network_size*network_size*sizeof(float*));
 		for (int i = 0; i< num_hamils; i++) {
 			array[i] = (float *) malloc(network_size*network_size*sizeof(float ));
 		}


     int n = 0;
     for (int k = 0; k < num_hamils; k++)
     {
        	for (int l = k+1; l < network_size; l++)
        	{
        		float my_array[network_size*network_size];
        		memset(my_array, 0, network_size*network_size*sizeof(float));
        		my_array[(network_size*l) + k] =  1;
        		my_array[(network_size*k) + l] = 1;
        		memcpy(array[n], my_array, network_size*network_size*sizeof(float));
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
				printf(" %f ", array[n][(j*network_size) + k] );
			}
			printf("]");
			printf("\n");
			}
		printf("\n");
	}


	// Generate the control Hamiltonians and add to array

	float ** CC = (float **)malloc(network_size*network_size*network_size*sizeof(float*));
 		for (int i = 0; i< network_size; i++) {
 			CC[i] = (float *) malloc(network_size*network_size*sizeof(float ));
 		}


 	for (int k = 0; k < network_size; k++)
 	{
 		float my_array[network_size*network_size];
        memset(my_array, 0, network_size*network_size*sizeof(float));
        my_array[(network_size*k) + k] =  1;
        memcpy(CC[k], my_array, network_size*network_size*sizeof(float));
 	}


 	// Print the Control Hamiltonians
 	printf("\nTHE CONTROL HAMILTONIANS [SEP]: \n");
	for (int n = 0;  n< network_size; n++){
		printf("\n");
	    for (int j = 0; j < network_size; j++){
	    	printf("[");
			for (int k = 0; k < network_size; k++){
				printf(" %f ", CC[n][(j*network_size) + k] );
			}
			printf("]");
			printf("\n");
			}
		printf("\n");
	}


	// Allocate mmemory for the drift hamiltionian
	float * drift_hamil = (float *)malloc(network_size*network_size*network_size*sizeof(float));


 	// Create random control biases that will be selected by the bayesian optimiser
 	float* biases = malloc((network_size + 1) * sizeof(float));
 	printf("THE CONTROL BIASES ARE: \n");
 	printf("[");
 	for (int i = 0; i < network_size+1; i++)
 	{
 		biases[i] = ((float)rand()/(float)(RAND_MAX)) * 10;
 		printf(" %f ", biases[i]);
 	}
 	printf("] \n");
 	
 	construct_drift_hamiltonian(CC, array, drift_hamil, biases, network_size);




	// Pauli Matrices
	double X[] = {0, 1, 1, 0};
	double complex Y[] = {0 + 0*I, 0 - 1*I, 0 + 1*I, 0 + 0*I};
	double Z[] = {1, 0, 0 ,-1};
	
	for (int i = 0; i < network_size; i++)
	{
		for (int l = i; l < network_size; l++)
		{
			
		}
	}

	return 0;
}









