#include<stdio.h>
#include<math.h>

void usage(int exitStatus, char* programName);
long long int sumArray(int* array, long long int arraySize);
void* getSeqPrimes(int* array, long long int arraySize);

__global__ void isPrime(int* d_array, long long int N){
    long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    long long int thisValue = (threadId * 2) + 1;

    if(thisValue < 1)
    	return;

    if(thisValue == 2){
        d_array[thisValue] = 1;
        return;
    }
    
    if(thisValue % 2 == 0){
        d_array[thisValue] = 0;
        return;
    }

    long long int j;
    for(j = 3; j*j < thisValue; j += 2){
        if(thisValue % j== 0){
            d_array[thisValue] = 0;
            return;
        }
    }

    d_array[thisValue] = 1;
}

int main(int argc, char** argv){
	if(argc != 3)
		usage(1, argv[0]);

	int N = (long long int) atoi(argv[1]);
	int blockSize = atoi(argv[2]);

	if(!(N | blockSize))
		usage(2, argv[0]);

	int arraySizeInBytes = sizeof(long long int) * (N + 1);

	// allocate our arrays
	int* h_array;
	int* d_array;
	int* seqArray;
	h_array = (int*) malloc(arraySizeInBytes);
	seqArray = (int*) calloc(arraySizeInBytes);
	cudaMalloc(&d_array, arraySizeInBytes);

	// zero the memory in cuda
	cudaMemset(d_array, 0, arraySizeInBytes);

	// caculate the grid size
	int gridSize = ceil((N + 1) / 2.0 / blockSize);

	int currentTime();
	// run the kernel
	isPrime<<<gridSize, blockSize>>>(d_array, N);

	// copy the results back to the host array
	cudaMemcpy(h_array, d_array, arraySizeInBytes, cudaMemcpyDeviceToHost);

	// release the device array
	cudaFree(d_array);

	// run the sequential version
	getSeqPrimes(seqArray, arraySizeInBytes);

	int seqSum = sumArray(seqArray);
	int parSum = sumArray(h_array);

	printf("N: %lld\n", N);
	printf("blockSize: %d\n", blockSize);
	printf("gridSize: %d\n", gridSize);
	printf("sequential prime count: %d\n", seqSum);
	printf("paralell prim count: %d\n", parSum);


	// Print our results
	int i = 0;
	for(; i < N; ++i){
		printf("h_array[%d] = %d\n", i, h_array[i]);
	}

	// free the host memory
	free(h_array);

	return 0;
}

void getSeqPrimes(int* array, long long int arraySize){
	if(thisValue < 1)
    	return;

    if(thisValue == 2){
        d_array[thisValue] = 1;
        return;
    }
    if(arraySize < 3)
    	return;

    long long int thisValue;
    for(thisValue = 3; thisValue < arraySize; thisValue += 2){}

	  
	    long long int j;
	    for(j = 3; j*j < thisValue; j += 2){
	        if(thisValue % j== 0){
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
	fprintf(stderr, "usage: %s N blockSize", programName);
	exit(exitStatus);
}




