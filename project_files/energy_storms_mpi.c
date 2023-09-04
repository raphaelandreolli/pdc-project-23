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
#include<math.h>
#include<sys/time.h>

/* Headers for the MPI assignment versions */
#include<mpi.h>

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

/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
void update( float *layer, int layer_size, int glob_pos_shift, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
	// K is now the local position and must be shiften with glob_pos_shift
    int distance = pos - k - glob_pos_shift;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[k] = layer[k] + energy_k;
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
    int i,j,k;

    int rank, MPIsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
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
    MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */

	/* 3. Allocate memory for the layer and initialize to zero */
	float *layer = (float *)malloc( sizeof(float) * layer_size );
	if ( layer == NULL){
		fprintf(stderr,"Error: Allocating the layer memory\n");
		exit( EXIT_FAILURE );
	}
	for( k=0; k<layer_size; k++ ){
		layer[k] = 0.0f;
		}
	// Calculate the average layer size for each MPI process
	int avg_loc_layer_size = layer_size/MPIsize;
	// Calculate the remaining bins which can't be eavenly distributed 
	int loc_glob_size_diff = layer_size - avg_loc_layer_size*MPIsize;
	
	

	
	// Create array with correctec number of bins for each process (used in MPI_Gatherv(...))
	int *rcounts = (int *)malloc(MPIsize*sizeof(int)); 
	for (j=0;j<MPIsize;j++)
	{
		rcounts[j] = avg_loc_layer_size;
	}
	for (k=0;k<loc_glob_size_diff;k++)
	{
		rcounts[k+1] += 1;
	}
	// Create array with global shift of local array (used in MPI_Gatherv(...))
	int *displs = (int *)malloc(MPIsize*sizeof(int)); 
	displs[0]=0;
	for (j=1;j<MPIsize;j++)
	{
		displs[j] = displs[j-1]+rcounts[j-1];
	}
	int loc_layer_size = rcounts[rank];
	int glob_pos_shift = displs[rank];
	
    /* 3. Allocate memory for the local layer and initialize to zero */
    float *loc_layer = (float *)malloc( sizeof(float) * loc_layer_size );
    float *loc_layer_copy = (float *)malloc( sizeof(float) * loc_layer_size );
    if ( loc_layer == NULL || loc_layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the local layer memory\n");
        exit( EXIT_FAILURE );
    }
    for( k=0; k<loc_layer_size; k++ ) loc_layer[k] = 0.0f;
    for( k=0; k<loc_layer_size; k++ ) loc_layer_copy[k] = 0.0f;
    
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) 
	{
		MPI_Barrier(MPI_COMM_WORLD);
        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for( j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];
            /* For each cell in the layer */
            for( k=0; k<loc_layer_size; k++) {
                /* Update the energy value for the cell */ 
				// Also include the global position shift for the local arrays                
				update( loc_layer, layer_size, glob_pos_shift, k, position, energy );
            }
        }

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        for( k=0; k<loc_layer_size; k++ )
		{
            loc_layer_copy[k] = loc_layer[k];
		}
        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */ 
		MPI_Barrier(MPI_COMM_WORLD);
		// If not first of last process, send first and last index to previous/next process
        if ((rank != 0) && (rank != MPIsize-1))
		{
			for( k=1; k<loc_layer_size-1; k++ )
			{
				loc_layer[k] = ( loc_layer_copy[k-1] + loc_layer_copy[k] + loc_layer_copy[k+1] ) / 3;
			}
			MPI_Send( &loc_layer_copy[0], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
			MPI_Send( &loc_layer_copy[loc_layer_size-1], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
			float prev_index;
			float next_index;
			MPI_Recv( &prev_index, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
			MPI_Recv( &next_index, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
			loc_layer[0] = ( prev_index + loc_layer_copy[0] + loc_layer_copy[1] ) / 3;
			loc_layer[loc_layer_size-1] = ( next_index + loc_layer_copy[loc_layer_size-2] + loc_layer_copy[loc_layer_size-1] ) / 3;
		}
		// If first local layer, only send last index to next process
		if (rank==0)
		{
			for( k=1; k<loc_layer_size-1; k++ ){
				loc_layer[k] = ( loc_layer_copy[k-1] + loc_layer_copy[k] + loc_layer_copy[k+1] ) / 3;
			}	
			MPI_Send( &loc_layer_copy[loc_layer_size-1], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
			float next_index;
			MPI_Recv( &next_index, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE  );
			loc_layer[loc_layer_size-1] = ( next_index + loc_layer_copy[loc_layer_size-2] + loc_layer_copy[loc_layer_size-1] ) / 3;
		}
		// If last local layer, only send first index to previous process
		if(rank==MPIsize-1)
		{
			for( k=1; k<loc_layer_size-1; k++ ){
				loc_layer[k] = ( loc_layer_copy[k-1] + loc_layer_copy[k] + loc_layer_copy[k+1] ) / 3;
			}	
			MPI_Send( &loc_layer_copy[0], 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
			float prev_index;
			MPI_Recv( &prev_index, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE  );
			loc_layer[0] = ( prev_index + loc_layer_copy[0] + loc_layer_copy[1] ) / 3;
		}
			
		// Create struct to hold maximum value of local array and its global index
		struct { 
			float loc_max_val; 
			int   glob_index; 
			} in={0,0}, out={0,0}; 
		
		/* 4.3. Locate the maximum value in the layer, and its position */
        for( k=0; k<loc_layer_size; k++ ) {
			//Skip first value in first process and last value in last process
			if ((rank==0 && k==0) || (rank==MPIsize-1 && k==loc_layer_size-1)){
				continue;
			}
			//***********************************************************************************
			/*if (rank==MPIsize-1 && k == loc_layer_size-1)
			{
				continue;
			}*/
            		// **********************************************************************************
			/* Check it only if it is a local maximum */
            if ( loc_layer[k] > loc_layer[k-1] && loc_layer[k] > loc_layer[k+1] ) {
                if ( loc_layer[k] > in.loc_max_val ) {
                    in.loc_max_val = loc_layer[k];
                    in.glob_index = k + glob_pos_shift;
                }
            }
        }
		MPI_Barrier(MPI_COMM_WORLD);
		// Find maximum of all processes using MPI_Recude(..,MPI_MAXLOC,..)
		MPI_Reduce(&in,&out,1,MPI_FLOAT_INT,MPI_MAXLOC, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// Use root to assign maximum value and global position
		if (rank==0)
		{
			maximum[i] = out.loc_max_val;
			positions[i] = out.glob_index;
		}
		// Same as in beginning of i-loop
		MPI_Barrier(MPI_COMM_WORLD);

    }
	MPI_Barrier(MPI_COMM_WORLD);

	//Rebuild full layer form processes using MPI_Gatherv
	MPI_Gatherv( loc_layer, loc_layer_size, MPI_FLOAT, layer, rcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	/* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    if ( rank == 0 ) {

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG
    debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
	fflush(stdout);

    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
	fflush(stdout);
    /* 7.2. Print the maximum levels */
    printf("Result:");
	fflush(stdout);
    for (i=0; i<num_storms; i++)
        printf(" %d %f", positions[i], maximum[i] );
		fflush(stdout);
    printf("\n");
	fflush(stdout);
    }

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    /* 9. Program ended successfully */
	// Finalize MPI
    MPI_Finalize();
    return 0;
}


