#include<stdio.h>
#include<math.h>
#include<stdlib.h>

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

	// allocate our arrays
	int* h_array;
	int* d_array;

	int* seqArray;
	h_array  = (int*) malloc(arraySizeInBytes);
	seqArray = (int*) calloc(sizeof(int), N + 1);
	printf("h_array: %p\n", h_array);
	printf("seqArray: %p\n", seqArray);

	free(h_array);
	free(seqArray);
	return 0;
	cudaMalloc(&d_array, arraySizeInBytes);

	// zero the memory in cuda
	cudaMemset(d_array, 0, arraySizeInBytes);

	// caculate the grid size
	int gridSize = (int)ceil((N + 1) / 2.0 / blockSize);

	// run the kernel
	getPrimes<<<gridSize, blockSize>>>(d_array, N);

	// copy the results back to the host array
	cudaMemcpy(h_array, d_array, arraySizeInBytes, cudaMemcpyDeviceToHost);

	// release the device array
	cudaFree(d_array);

	// run the sequential version
	getSeqPrimes(seqArray, arraySizeInBytes);

	int seqSum = sumArray(seqArray, N + 1);
	int parSum = sumArray(h_array, N + 1);

	printf("N: %d\n", N);
	printf("blockSize: %d\n", blockSize);
	printf("gridSize: %d\n", gridSize);
	printf("sequential prime count: %d\n", seqSum);
	printf("paralell prim count: %d\n", parSum);

	printf("h_array: %p\n", h_array);
	printf("seqArray: %p\n", seqArray);

	free(seqArray);
	free(h_array);

	return 0;
}

void getSeqPrimes(int* array, int arraySize){
	if(arraySize >= 3)
		array[2] = 1;

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