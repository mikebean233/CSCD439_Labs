all: lab1.cu
	nvcc -arch=sm_35 -o cscd439lab1 lab1.cu

jetson: lab1.cu
	nvcc -m32 -o cscd439lab1 lab1.cu

clean: cscd439lab1
	rm cscd439lab1