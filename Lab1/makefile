all: lab1.cu
	nvcc -arch=sm_35 -o cscd439lab1 lab1.cu

jetson_tk1: lab1.cu
	nvcc -arch=sm_32 -o cscd439lab1 lab1.cu

clean: cscd439lab1
	rm cscd439lab1
