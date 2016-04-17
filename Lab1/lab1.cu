#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<sys/time.h>

void usage(int exitStatus, char* programName);
int sumArray(int* array, int arraySize);
void getSeqPrimes(int* array, int arraySize);

__host__ __device__ int  isPrime(int value);

__global__ void getPrimes(int* d_array, int N){
    int threadId = 0;
	threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int thisValue = 0;
	thisValue = (threadId * 2) + 1;

    if(threadId < 1)
    	return;

    if(thisValue < N){
		d_array[thisValue] = isPrime(thisValue);
	}
}

__host__ __device__ int isPrime(int value){

	int limit = 0;
	limit = (int) sqrt( (float) value ) + 1;
	int j = 0;
	for(j = 2; j < limit; j++){
		if(value % j == 0){
			return 0;
		}
	}
	return 1;
}


int main(int argc, char** argv){
	if(argc != 3)
		usage(1, argv[0]);

	int N = 0;
	N = (int) atoi(argv[1]);
	int blockSize = 0;
	blockSize = (int) atoi(argv[2]);

	if(!(N | blockSize))
		usage(2, argv[0]);

	int arraySizeInBytes = 0;
	arraySizeInBytes = 0;
	arraySizeInBytes = sizeof(int) * (N + 1);

	// index 0 : start time,  index 1: end time
	struct timeval sequentialTimes[2] = {{0,0},{0,0}};
	struct timeval parallelTimes[2]   = {{0,0},{0,0}};


	// allocate our arrays
	int* h_array = NULL;
	int* d_array = NULL;

	int* seqArray = NULL;
	h_array  = (int*) malloc(arraySizeInBytes);
	seqArray = (int*) calloc(sizeof(int), N + 1);

	// caculate the grid size
	int gridSize = 0;
	gridSize = (int)ceil((N + 1) / 2.0 / blockSize);

	// start parallel timer
	gettimeofday( &(parallelTimes[0]), NULL);

		// allocate device memory for the array
		cudaMalloc(&d_array, arraySizeInBytes);

		// zero the memory in cuda
		cudaMemset(d_array, 0, arraySizeInBytes);

		// run the kernel
		getPrimes<<<gridSize, blockSize>>>(d_array, N);

		// copy the results back to the host array
		cudaMemcpy(h_array, d_array, arraySizeInBytes, cudaMemcpyDeviceToHost);

		// release the device array
		cudaFree(d_array);

	// stop parallel timer
	gettimeofday( &(parallelTimes[1])  , NULL);

	// start sequential timer
	gettimeofday( &(sequentialTimes[0]), NULL);

		// run the sequential version
		getSeqPrimes(seqArray, N + 1);

	// stop sequential timer
	gettimeofday( &(sequentialTimes[1]), NULL);

	// calculated time values
	double parallelSeconds[2] = {0.0, 0.0};
	parallelSeconds[0] = parallelTimes[0].tv_sec + ((double)parallelTimes[0].tv_usec / 1000000);
	parallelSeconds[1] = parallelTimes[1].tv_sec + ((double)parallelTimes[1].tv_usec / 1000000);

	double sequentialSeconds[2] = {0.0, 0.0};
	sequentialSeconds[0] = sequentialTimes[0].tv_sec + ((double)sequentialTimes[0].tv_usec / 1000000);
	sequentialSeconds[1] = sequentialTimes[1].tv_sec + ((double)sequentialTimes[1].tv_usec / 1000000);

	double parallelCost   = 0;
	parallelCost = parallelSeconds[1] - parallelSeconds[0];
	double sequentialCost = 0;
	sequentialCost = sequentialSeconds[1] - sequentialSeconds[0];
	double speedup = 0;
	speedup = sequentialCost / parallelCost;


	int seqSum = 0;
	seqSum = sumArray(seqArray, N + 1);
	
	int parSum = 0;
	parSum = sumArray(h_array, N + 1);

	printf("                     N: %d\n",  N);
	printf("             blockSize: %d\n",  blockSize);
	printf("              gridSize: %d\n",  gridSize);
	printf("sequential prime count: %d\n",  seqSum);
	printf("  parallel prime count: %d\n",  parSum);
	printf("    parallel time cost: %lf\n", parallelCost);
	printf("  sequential time cost: %lf\n", sequentialCost);
	printf("               speedup: %lf\n", speedup);

	free(h_array);
	free(seqArray);

	return 0;
}

void getSeqPrimes(int* array, int arraySize){
	int thisValue = 0;
	for(thisValue = 3; thisValue < arraySize; thisValue += 2){
		array[thisValue] = isPrime(thisValue);
	}
}

int sumArray(int* array, int arraySize){
	int sum = 0;
	int index = 0;
	for(; index < arraySize; ++index){
		sum += array[index];
	}
	return sum;
}

void usage(int exitStatus, char* programName){
	fprintf(stderr, "usage: %s N blockSize\n", programName);
	exit(exitStatus);
}
