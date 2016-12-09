#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <time.h>

//#include <malloc.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#include <helper_timer.h>

#define ARRAY_ELEMENTS_MAX 0x7fffffff
//262144
#define THREADS_IN_BLOCK_MAX 128

#define min(x,y)  ((y) > (x) ? (x) : (y))

__global__ void exchEven(unsigned int *A, const unsigned int n);
__global__ void exchOdd(unsigned int *A, const unsigned int n);

int main(const int argc, const char * argv[]) {
	int elemCount = 0;

	printf("sizeof(unsigned int) = %d, sizeof(long) = %d.\n",sizeof(unsigned int), sizeof(long));

	if (argc != 2)
		return EXIT_FAILURE;

	elemCount = atoi(argv[1]);

	if (!(elemCount > 0 && elemCount <= ARRAY_ELEMENTS_MAX)) {
		printf("Supplied number of elements is out of bound %d.\n", ARRAY_ELEMENTS_MAX);
		return EXIT_FAILURE;
	}
	printf("%u elements.\n\n");

	// device(0) : GTX 1080
	// device(1) : GTX 750Ti
	cudaSetDevice(0);
	
	unsigned int *A;
	A = (unsigned int*) malloc(sizeof(unsigned int) * elemCount );

	// setup dummy input 
	srand(time(NULL));
	for (unsigned int i = 0; i < elemCount; i++) {
		A[i] = rand() % 10000;
	}

	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < 100 || i == elemCount - 1) {
			printf("%4u ", i);
		}
		else if (i == elemCount - 2) {
			printf("... ");
		}
	}
	printf("\n");
	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < 100 || i == elemCount - 1) {
			printf("%4u ", A[i]);
		}
		else if (i == elemCount - 2) {
			printf("... ");
		}
	}
	printf("\n\n");
	
	// setup input copy on device mem
	unsigned int *devA;
	const unsigned devACapa = ((elemCount + 31) / 32) * 32;
	cudaMalloc((void**)&devA, sizeof(unsigned int) * devACapa);
	cudaMemcpy(devA, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	const unsigned int blockSize = min(THREADS_IN_BLOCK_MAX, devACapa/2);
	const unsigned int gridSize = (devACapa/2+(blockSize -1))/blockSize;
	printf("Going to use %d blocks, %d threas per block with capacity %d.\n\n", gridSize, blockSize, devACapa);
	dim3 grids(gridSize), blocks(blockSize);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (unsigned int i = 0; i < (elemCount>>1); i++) {
		exchEven << < grids, blocks >> > (devA, elemCount);
		exchOdd << < grids, blocks >> > (devA, elemCount); // possibly the last call is redundant
	}
	
	cudaMemcpy(A, devA, sizeof(unsigned int) * devACapa, cudaMemcpyDeviceToHost);

	sdkStopTimer(&timer);

	int flag = 0;
	unsigned int firstFailure = elemCount;
	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < elemCount - 1) {
			if (A[i] > A[i + 1]) {
				flag = 1;
				firstFailure = i;
			}
		}
		if (i < 100 || i == elemCount - 1) {
			printf("%4u ", A[i]);
		}
		else if (i == elemCount - 2) {
			printf("... ");
		}

	}
	printf("\n");
	if (flag != 0) {
		printf("!!!Sort failure deteced at A[%d] = %d and A[%d] = %d!!!\n", 
			firstFailure, A[firstFailure], firstFailure+1, A[firstFailure+1]);
	}
	printf("[%u] = %u\n", elemCount - 1, A[elemCount - 1]);

	printf("Elapsed time %f msec.\n", (float)((int)(sdkGetTimerValue(&timer)*1000))/1000 );
	sdkDeleteTimer(&timer);

	cudaFree(devA);
	free(A);

	cudaDeviceReset();

}


__global__ void exchEven(unsigned int *a, const unsigned int n) {
	unsigned int thix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int left = thix << 1;
	unsigned int tmp;

	if ( left + 1 < n ) {
		if (a[left] > a[left+1]) {
			tmp = a[left];
			a[left] = a[left+1];
			a[left+1] = tmp;
		}
	}
	__syncthreads();
}

__global__ void exchOdd(unsigned int *a, const unsigned int n) {
	unsigned int thix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int left = (thix << 1) + 1;
	unsigned int tmp;

	if (left + 1 < n) {
		if (a[left] > a[left+1]) {
			tmp = a[left];
			a[left] = a[left+1];
			a[left+1] = tmp;
		}
	}
	__syncthreads();
}
