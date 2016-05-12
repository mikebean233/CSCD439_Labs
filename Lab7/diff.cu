#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCKSIZE 4 // Number of threads in each thread block
 
// CUDA kernel. Each thread takes care of one element of a 
__global__ void diffKernel( float *in, float *out, int n )
{
    // Wrtie the kernel to implement the diff operation on an array Using shared memory
    extern __shared__ int s_data[];
    
    unsigned int global_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Make sure this thread is within bounds
    if(global_index >= n - 2)
        return;
    
    // Copy global data to shared memory
    s_data[threadIdx.x] = in[global_index];

    if(threadIdx.x == blockDim.x - 1)
        s_data[threadIdx.x + 1] = in[global_index + 1];

    // Wait for all of the threads to reach the barrier
    __syncthreads();

    // Perform the actuall diff
    out[global_index] = s_data[threadIdx.x + 1] - s_data[threadIdx.x];
}  
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int i;
    float input[] = {4, 5, 6, 7, 19, 10, 0, 4, 2, 3, 1, 7, 9, 11, 45, 23, 99, 29};
    int n = sizeof(input) / sizeof(float); //careful, this usage only works with statically allocated arrays, NOT dynamic arrays

    // Host input vectors
    float *h_in = input;
    //Host output vector
    float *h_out = (float *) malloc((n - 1) * sizeof(float));
 
    // Device input vectors
    float *d_in;;
    //Device output vector
    float *d_out;
 
    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(float);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
 
    // Copy host data to device
    cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // TODO: setup the blocksize and gridsize and launch the kernel below.
    // Number of threads in each thread block
     int blocksize = 5;
    // Number of thread blocks in grid
    int gridsize = ceil((float) (n - 1) / blocksize);
    // Execute the kernel
    diffKernel<<<blocksize, gridsize, blocksize + 1>>>(d_in, d_out, n); 
 
    // Copy array back to host
    cudaMemcpy( h_out, d_out, bytes, cudaMemcpyDeviceToHost );
 
    // Show the result
    printf("The original array is: ");
    for(i = 0; i < n; i ++)
        printf("%4.0f,", h_in[i] );    
    
    printf("\n\nThe diff     array is:     ");
    for(i = 0; i < n - 1; i++)
        printf("%4.0f,", h_out[i] );    
    puts("");
    
    // Release device memory
    cudaFree(d_in);
    cudaFree(d_out);
 
    // Release host memory
    free(h_out);
 
    return 0;
}
