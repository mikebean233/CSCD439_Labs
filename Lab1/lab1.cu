#include<stdio.h>

void usage(int exitStatus, char* programName);

__global__ void isPrime(int* d_array, int N){
        int thisValue = blockIdx.x * blockDim.x + threadIdx.x;

    if(thisValue == 0){
        d_array[thisValue] = 0;
                return;
    }

    if(thisValue == 1){
        d_array[thisValue] = 0;
            return;
        }

    if(thisValue == 2){
        d_array[thisValue] = 1;
        return;
    }
    
    if(thisValue % 2 == 0){
        d_array[thisValue] = 0;
        return;
    }

    int j = 3;
    for(; j*j < N; j += 2){
        if(j % thisValue == 0){
            d_array[thisValue] = 0;
            return;
        }
    }

    d_array[thisValue] = 1;
}

int main(int argc, char** argv){
	if(argc != 3)
		usage(1, argv[0]);

	int N = atoi(argv[1]);
	int blockSize = atoi(argv[2]);

	if(!(N | blockSize))
		usage(2, argv[0]);

	int arraySizeInBytes = sizeof(long long int) * N;

	// allocate our arrays
	long long int* h_array;
	long long int* d_array;
	h_array = (long long int*) malloc(arraySizeInBytes);
	cudaMalloc(&d_array, arraySizeInBytes);

	// caculate the grid size
	int gridSize;
	if(N % blockSize == 0)
		gridSize = N / blockSize;
	else
		gridSize = (N / blockSize) + 1;


	// run the kernel
	isPrime<<<gridSize, blockSize>>>(d_array, N);

	// copy the results back to the host array
	cudaMemcpy(h_array, d_array, arraySizeInBytes, cudaMemcpyDeviceToHost);

	// release the device array
	cudaFree(d_array);

	// Print our results
	int i = 0;
	for(; i < N; ++i){
		printf("h_array[%d] = %d\n", i, h_array[i]);
	}

	// free the host memory
	free(h_array);

	return 0;
}

void usage(int exitStatus, char* programName){
	fprintf(stderr, "usage: %s N blockSize", programName);
	exit(exitStatus);
}




