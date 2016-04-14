#include<stdio.h>

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
	int* h_array = (void*) malloc(arraySize);

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





