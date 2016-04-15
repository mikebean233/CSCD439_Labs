#include<stdio.h>

void usage(int exitStatus, char* programName);
long long int sumArray(int* array);

__global__ void isPrime(int* d_array, long long int N){
    long long int theadId = blockIdx.x * blockDim.x + threadIdx.x;
    long long ing thisValue = (threadId * 2) + 1;

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
	h_array = (long long int*) malloc(arraySizeInBytes);
	cudaMalloc(&d_array, arraySizeInBytes);

	// zero the memory in cuda
	cudaMemset(d_array, 0, arraySizeInBytes);

	// caculate the grid size
	int gridSize ((N + 1) / 2.0 / blockSize);


	// run the kernel
	isPrime<<<gridSize, blockSize>>>(d_array, N);

	// copy the results back to the host array
	cudaMemcpy(h_array, d_array, arraySizeInBytes, cudaMemcpyDeviceToHost);

	// release the device array
	cudaFree(d_array);

	// Print our results
	int i = 0;
	for(; i < N; ++i){
		printf("h_array[%d] = %lld\n", i, h_array[i]);
	}

	// free the host memory
	free(h_array);

	return 0;
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




