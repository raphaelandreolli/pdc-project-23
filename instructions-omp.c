// ! PDC project 2023 - Run tests for OpenMP case
// ! -----------------------------------------------------------------------------------------------------
	
// Compile programs
make all // change compiler setting in Makefile depending on if the program is run on local machine or Dardel

// Allocate resources (1 node)
salloc -t 00:20:00 -A edu23.summer -N 1 -p main 

// Run jobs in compute node, not login node.

// ! test_01
// ! -----------------------------------------------------------------------------------------------------
	// Basic general debugging test, for small arrays of 35 positions. 

// ? Local machine
// serial
./energy_storms_seq 35 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p8_w4

// omp
./energy_storms_omp 35 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p8_w4


// ? Dardel (use on full node)
// serial
srun -n 1 -u ./energy_storms_seq 35 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p8_w4

// omp
export OMP_NUM_THREADS=4 // before running the program in terminal (not required inside program)
srun -u ./energy_storms_omp 35 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p8_w4

// ! test_02
// ! -----------------------------------------------------------------------------------------------------
	// Small workload test, for arrays of 30.000 positions. 
	// Each wave has 20.000 random particles.

// ? Local machine
// serial
./energy_storms_seq 30000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

// omp
./energy_storms_omp 30000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

// ? Dardel (use on full node)
// serial
srun -n 1 -u ./energy_storms_seq 30000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

// omp
export OMP_NUM_THREADS=128 // before running the program in terminal (not required inside program)
srun -u ./energy_storms_omp 30000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6

// ! test_03 to test_06	
// ! -----------------------------------------------------------------------------------------------------
	// Test race conditions and proper communication in borders of partitions. 
	// Execute with arrays of 20 positions, with 2 and 4 threads/processes,
	// only one file at a time.

// ? Local machine
// serial
./energy_storms_seq 20 test_files/test_03_a20_p4_w1
./energy_storms_seq 20 test_files/test_04_a20_p4_w1
./energy_storms_seq 20 test_files/test_05_a20_p4_w1
./energy_storms_seq 20 test_files/test_06_a20_p4_w1
// omp
./energy_storms_omp 20 test_files/test_03_a20_p4_w1
./energy_storms_omp 20 test_files/test_04_a20_p4_w1
./energy_storms_omp 20 test_files/test_05_a20_p4_w1
./energy_storms_omp 20 test_files/test_06_a20_p4_w1


// ? Dardel (use on full node)
// serial - 03
srun -n 1 -u ./energy_storms_seq 20 test_files/test_03_a20_p4_w1
// serial - 04
srun -n 1 -u ./energy_storms_seq 20 test_files/test_04_a20_p4_w1
// serial - 05
srun -n 1 -u ./energy_storms_seq 20 test_files/test_05_a20_p4_w1
// serial - 06
srun -n 1 -u ./energy_storms_seq 20 test_files/test_06_a20_p4_w1

// omp
export OMP_NUM_THREADS=2 // before running the program in terminal (not required inside program)
// omp - 03
srun -u ./energy_storms_omp 20 test_files/test_03_a20_p4_w1
// omp - 04
srun -u ./energy_storms_omp 20 test_files/test_04_a20_p4_w1
// omp - 05
srun -u ./energy_storms_omp 20 test_files/test_05_a20_p4_w1
// omp - 06
srun -u ./energy_storms_omp 20 test_files/test_06_a20_p4_w1


export OMP_NUM_THREADS=4 // before running the program in terminal (not required inside program)
// omp - 03
srun -u ./energy_storms_omp 20 test_files/test_03_a20_p4_w1
// omp - 04
srun -u ./energy_storms_omp 20 test_files/test_04_a20_p4_w1
// omp - 05
srun -u ./energy_storms_omp 20 test_files/test_05_a20_p4_w1
// omp - 06
srun -u ./energy_storms_omp 20 test_files/test_06_a20_p4_w1


// ! test_07
// ! -----------------------------------------------------------------------------------------------------
	// Optimizations test, for arrays of 1 million positions. Each wave has 5.000 random particles.

// ? Local machine
// serial
./energy_storms_seq 1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4

// omp
./energy_storms_omp 1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4

// ? Dardel (use on full node)
// serial
srun -n 1 -u ./energy_storms_seq 1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4

// omp
export OMP_NUM_THREADS=128 // before running the program in terminal (not required inside program)
srun -u ./energy_storms_omp 1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4

// ! test_08
// ! -----------------------------------------------------------------------------------------------------
	// Reduction efficiency test, for arrays with 100 million positions, only 1 particle per wave.

// ? Local machine
// serial
./energy_storms_seq 100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3

// omp
./energy_storms_omp 100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3

// ? Dardel (use on full node)
// serial
srun -n 1 -u ./energy_storms_seq 100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3

// omp
export OMP_NUM_THREADS=128 // before running the program in terminal (not required inside program)
srun -u ./energy_storms_omp 100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3

// ! test_09
// ! -----------------------------------------------------------------------------------------------------
	// Execute with both 16 and 17 array positions to see the difference. 
	// Use it to test that the behaviour of the sequential program in the array extremes has been preserved after your transformations.

// ? Local machine
// serial
./energy_storms_seq 16 test_files/test_09_a16-17_p3_w1
./energy_storms_seq 17 test_files/test_09_a16-17_p3_w1

// omp
./energy_storms_omp 16 test_files/test_09_a16-17_p3_w1
./energy_storms_omp 17 test_files/test_09_a16-17_p3_w1

// ? Dardel (use on full node)
// serial
// 16
srun -n 1 -u ./energy_storms_seq 16 test_files/test_09_a16-17_p3_w1
// 17
srun -n 1 -u ./energy_storms_seq 17 test_files/test_09_a16-17_p3_w1

// omp
// 16
export OMP_NUM_THREADS=4 // before running the program in terminal (not required inside program)
srun -u ./energy_storms_omp 16 test_files/test_09_a16-17_p3_w1
// 17
export OMP_NUM_THREADS=4 // before running the program in terminal (not required inside program)
srun -u ./energy_storms_omp 17 test_files/test_09_a16-17_p3_w1
