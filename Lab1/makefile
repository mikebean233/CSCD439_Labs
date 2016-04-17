all: lab1.cu
	nvcc -arch=sm_35 -o cscd439lab1 lab1.cu

jetson: lab1.cu
	nvcc -G -g -O0 -ccbin arm-linux-gnueabihf-g++-4.6 -gencode arch=compute_30, code=sm_30 -gencode arch=compute_32,code=sm_32 --target-cpu-architecture ARM -o cscd439lab1 lab1.cu

clean: cscd439lab1
	rm cscd439lab1
