/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * Code prepared to be used with the Tablon on-line judge.
 * The current Parallel Computing course includes contests using:
 * OpenMP, MPI, and CUDA.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the CUDA assignment versions */
#include<cuda.h>

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}


/* Use fopen function in local tests. The Tablon online judge software
   substitutes it by a different function to run in its sandbox */
#ifdef CP_TABLON
#include "cputilstablon.h"
#else
#define    cp_open_file(name) fopen(name,"r")
#endif

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


#define THRESHOLD    0.001f
#define RAD 1
#define TPB 256
#define RAND_MIN -1.0

/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;



/* Kernel Function to update energy due to particle bombardment on cells */
__global__ void kernelupdate( float *d_layer,float *h_layer, int layer_size, Storm storms) {

//global thread index
    int k = blockIdx.x * blockDim.x + threadIdx.x;

//local thread index for local shared array
	const int s_idx = threadIdx.x + RAD;
  
//dynamic shared memory
	extern __shared__ float s_in[];

    if (k >= layer_size) return;



	float energy_value = 0.0f ;
	for (int i=0; i<storms.size; i++)
	{
 /* 1. Compute the absolute value of the distance between the impact position and the k-th position of the layer */
		float energy = storms.posval[i*2+1] * 1000;
		int distance = abs(storms.posval[i*2] - k) + 1;
		float attenuation = sqrtf(static_cast<float>(distance));
     /*  Compute attenuated energy */
		float energy_k = energy / layer_size / attenuation;


		if (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size)
		{
			energy_value +=energy_k;
	    }
	}

  /*  Add intermediate impacts energies before relaxation to  cells */
	h_layer[k]=h_layer[k]+energy_value;
	__syncthreads();

//copying data from global to local shared array
	s_in[s_idx] = h_layer[k];
	if (threadIdx.x < RAD) {
		if (k==0){
			s_in[s_idx - RAD] = h_layer[k];
			}
		else {
				s_in[s_idx - RAD] = h_layer[k - RAD];
				}
		s_in[s_idx + blockDim.x] = h_layer[k + blockDim.x];
	}
	__syncthreads();

   /*  Energy relaxation between storms, preserving first and last element */
    if ( k == 0 || k == layer_size-1 ) {
		d_layer[k]=h_layer[k];
    }
	else {
		d_layer[k] = (s_in[s_idx-1]+ s_in[s_idx] + s_in[s_idx+1])/3;
	}
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer_size <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer_size; k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters.
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
                printf("x");
            else
                printf("o");

            /* If the cell is the maximum of any storm, print the storm mark */
            for (i=0; i<num_storms; i++)
                if ( positions[i] == k ) printf(" M%d", i );

            /* Line feed */
            printf("\n");
        }
    }
}

/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file( char *fname ) {
    FILE *fstorm = cp_open_file( fname );
    if ( fstorm == NULL ) {
        fprintf(stderr,"Error: Opening storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    Storm storm;
    int ok = fscanf(fstorm, "%d", &(storm.size) );
    if ( ok != 1 ) {
        fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
    if ( storm.posval == NULL ) {
        fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
        exit( EXIT_FAILURE );
    }

    int elem;
    for ( elem=0; elem<storm.size; elem++ ) {
        ok = fscanf(fstorm, "%d %d\n",
                    &(storm.posval[elem*2]),
                    &(storm.posval[elem*2+1]) );
        if ( ok != 2 ) {
            fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
            exit( EXIT_FAILURE );
        }
    }
    fclose( fstorm );

    return storm;
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,k;

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    int numBlocks = (atoi( argv[1] ) + TPB - 1) / TPB; // Number of blocks needed

    Storm storms[ num_storms ];

    /* 1.2. Read storms information */
    for( i=2; i<argc; i++ )
        storms[i-2] = read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* 2. Begin time measurement */
	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */

    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );

    if ( layer == NULL  ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;


    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {

 // Load storm to device memory
	    Storm storm;
	    storm.size= storms[i].size;
	    size_t size1=2*storms[i].size* sizeof(int);
	    cudaMalloc(&storm.posval, size1);
        cudaMemcpy(storm.posval, storms[i].posval, size1,
               cudaMemcpyHostToDevice);

// Allocate layer and intermediate layer (h_layer) before relaxation in device memory
	    float *d_layer;
		cudaMalloc( &d_layer,sizeof(float) * layer_size );

	    float *h_layer;
		cudaMalloc( &h_layer,sizeof(float) * layer_size );

// Copy layer from host CPU to device GPU
		cudaMemcpy(h_layer, layer, layer_size * sizeof(float), cudaMemcpyHostToDevice);

 
        dim3 blockDim(TPB);
        dim3 gridDim(numBlocks);

//Size for shared memory array        
        const size_t smemSize = (blockDim.x + 2 * RAD) * sizeof(float);

// Invoke kernel
		kernelupdate<<<gridDim, blockDim,smemSize>>>(d_layer, h_layer, layer_size, storm);
        cudaDeviceSynchronize() ;

// Read layer from device memory
		size_t size=sizeof(float) * layer_size;
	    cudaMemcpy(layer, d_layer, size,
               cudaMemcpyDeviceToHost);

// Free device memory
    cudaFree(h_layer);
		cudaFree(d_layer);

        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }
    }

    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
    ttotal = cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG
    debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (i=0; i<num_storms; i++)
        printf(" %d %f", positions[i], maximum[i] );
    printf("\n");

    /* 8. Free resources */
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
    return 0;
}
