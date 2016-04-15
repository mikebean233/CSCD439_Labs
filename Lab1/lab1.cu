#include<stdio.h>
#include<math.h>

void usage(int exitStatus, char* programName);
long long int sumArray(int* array, long long int arraySize);
void getSeqPrimes(int* array, long long int arraySize);
__device__ int  isPrime2(long long int value);

__global__ void isPrime(int* d_array, long long int N){
    long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    long long int thisValue = (threadId * 2) + 1;

    if(threadId < 1)
    	return;

    if(thisValue < N){
		d_array[thisValue] = isPrime2(thisValue);
	}
}

__device__ int isPrime2(long long int value){

long long int limit = (long long int) sqrt( (float) value ) + 1;
long long int j;
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

	long long int N = (long long int) atoi(argv[1]);
	int blockSize = atoi(argv[2]);

	if(!(N | blockSize))
		usage(2, argv[0]);

	int arraySizeInBytes = sizeof(int) * (N + 1);

	// allocate our arrays
	int* h_array;
	int* d_array;

	int* seqArray;
	h_array  = (int*) malloc(arraySizeInBytes);
//	seqArray = (int*) calloc(sizeof(int), N + 1);
	cudaMalloc(&d_array, arraySizeInBytes);

	// zero the memory in cuda
	cudaMemset(d_array, 0, arraySizeInBytes);

	// caculate the grid size
	int gridSize = (int)ceil((N + 1) / 2.0 / blockSize);

	int currentTime();
	// run the kernel
	isPrime<<<gridSize, blockSize>>>(d_array, N);

	// copy the results back to the host array
	cudaMemcpy(h_array, d_array, arraySizeInBytes, cudaMemcpyDeviceToHost);

	// release the device array
	cudaFree(d_array);

	// run the sequential version
//	getSeqPrimes(seqArray, arraySizeInBytes);

//	int seqSum = sumArray(seqArray, N + 1);
	int parSum = sumArray(h_array, N + 1);

	printf("N: %lld\n", N);
	printf("blockSize: %d\n", blockSize);
	printf("gridSize: %d\n", gridSize);
	//printf("sequential prime count: %d\n", seqSum);
	printf("paralell prim count: %d\n", parSum);
    
    //free(seqArray);
    free(h_array);

	return 0;
}

void getSeqPrimes(int* array, long long int arraySize){
	
    long long int thisValue;
    for(thisValue = 2; thisValue < arraySize; thisValue += 2){
    	if(thisValue == 2){
    		array[thisValue] = 1;
	    	continue;
	    }

	    long long int j;
	    for(j = 2; j*j < thisValue; j++){
	        if(thisValue % j == 0){
	            array[thisValue] = 0;
	        }
	    }
	    array[thisValue] = 1;
	}
}

long long int sumArray(int* array, long long int arraySize){
	long long int sum = 0;
	long long index = 0;
	for(; index < arraySize; ++index){
		sum += array[index];
	}
	return sum;
}

void usage(int exitStatus, char* programName){
	fprintf(stderr, "usage: %s N blockSize\n", programName);
	exit(exitStatus);
}




