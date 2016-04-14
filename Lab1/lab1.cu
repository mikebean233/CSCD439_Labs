#include<stdio.h>

__device__ void isPrime(int* d_values, int N){
	int thisValue = blockIdx.x * blockDim.x + threadIdx.x;

	if(thisValue == 0){
		d_values[thisValue] = 0;
		return;	
	}

	if(thisValue == 1){
		d_values[thisValue] = 0;
		return;
	}

	if(thisValue == 2){
		d_values[thisValue] = 1;
		return;
	}
	if(thisValue % 2 == 0){
		d_values[thisValue] = 0;
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
	int blockSize |= atoi(argv[2]);

	if(!(n | blockSize))
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
	doubleIndexValue<<<gridSize, blockSize>>>(d_array, N);

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

/*
// Gpu kernel code
__global__ void doubleIndexValue(int* a, int n){
	int threadId = blockIdx.x*blockDim.x + threadIdx.x;
	a[threadId] = 2 * threadId;
}


int main(){
	// length of array
	int n = 100;

	//	size of array in bytes
	int arraySize = n * sizeof(int);

	// allocate the host array
	int* h_array = (int*) malloc(arraySize);

	// allocate the gpu array
	int* d_array;
	cudaMalloc(&d_array, arraySize);

	// copy host array to gpu (not neccessary in this test)
	// ...


	int blockSize = 10;
	int gridSize  = 15;

	// run the actual gpu kernel
	doubleIndexValue<<<gridSize, blockSize>>>(d_array, n);


	// copy the results back to the host array
	cudaMemcpy( h_array, d_array, arraySize, cudaMemcpyDeviceToHost );

	// free the gpu memory
	cudaFree(d_array);


	// Print our results
	int i;
	for(i = 0; i < n; ++i){
		printf("h_array[%d] = %d\n", i, h_array[i]);
	}

	// free the host memory
	free(h_array);

	return 0;
}

*/



