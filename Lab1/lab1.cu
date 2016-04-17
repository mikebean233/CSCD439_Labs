#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<sys/time.h>

void usage(int exitStatus, char* programName);
int sumArray(int* array, int arraySize);
void getSeqPrimes(int* array, int arraySize);

__host__ __device__ int  isPrime(int value);

__global__ void getPrimes(int* d_array, int N){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int thisValue = (threadId * 2) + 1;

    if(threadId < 1)
    	return;

    if(thisValue < N){
		d_array[thisValue] = isPrime(thisValue);
	}
}

__host__ __device__ int isPrime(int value){

	int limit = (int) sqrt( (float) value ) + 1;
	int j;
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

	int N = (int) atoi(argv[1]);
	int blockSize = atoi(argv[2]);

	if(!(N | blockSize))
		usage(2, argv[0]);

	int arraySizeInBytes = sizeof(int) * (N + 1);

	// index 0 : start time,  index 1: end time
	struct timeval sequentialTimes[2];
	struct timeval parallelTimes[2];


	// allocate our arrays
	int* h_array;
	int* d_array;

	int* seqArray;
	h_array  = (int*) malloc(arraySizeInBytes);
	seqArray = (int*) calloc(sizeof(int), N + 1);

	// caculate the grid size
	int gridSize = (int)ceil((N + 1) / 2.0 / blockSize);

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

	// stop parallel timer
	gettimeofday( &(sequentialTimes[1]), NULL);

	// calculated time values
	unsigned long  parCostInMicroseconds = (unsigned long) (parallelTimes[1].tv_usec) - (unsigned long) (parallelTimes[0].tv_usec);
	unsigned long seqCostInMicroseconds = (unsigned long) (sequentialTimes[1].tv_usec) -(unsigned long) (sequentialTimes[0].tv_usec);
	//double speedup = parallelElapsedSeconds / sequentialElapsedSeconds;


	int seqSum = sumArray(seqArray, N + 1);
	int parSum = sumArray(h_array, N + 1);

	printf("                     N: %d\n", N);
	printf("             blockSize: %d\n", blockSize);
	printf("              gridSize: %d\n", gridSize);
	printf("sequential prime count: %d\n", seqSum);
	printf("   parallel prim count: %d\n", parSum);
	printf("    parallel time cost: %lu\n", parCostInMicroseconds);
	printf("  sequential time cost: %lu\n", seqCostInMicroseconds);
	//printf("              speedup: %lf\n", speedup);

	free(h_array);
	free(seqArray);

	return 0;
}

void getSeqPrimes(int* array, int arraySize){
	int thisValue;
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
