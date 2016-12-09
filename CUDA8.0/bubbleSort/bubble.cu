#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#define ARRAY_ELEMENTS_MAX 0x7fffffff
//262144
#define THREADS_IN_BLOCK_MAX 1024

#define min(x,y)  ((y) > (x) ? (x) : (y))
#define align(val,base)  ( (((val)/(base)) + ((val)%(base) != 0) )*(base) )

__global__ void exchEven(unsigned int *A, const unsigned int n);
__global__ void exchOdd(unsigned int *A, const unsigned int n);

int main(const int argc, const char * argv[]) {
	int elemCount = 0;
	int blockSize = THREADS_IN_BLOCK_MAX;

	if (argc == 2) {
		elemCount = atoi(argv[1]);
	} else if ( argc == 3 ) {
		elemCount = atoi(argv[1]);
		blockSize = atoi(argv[2]);
	} else {
		printf("incorrect argument number.n");
		return EXIT_FAILURE;
	}

	if (!(elemCount > 0 && elemCount <= ARRAY_ELEMENTS_MAX)) {
		printf("Supplied number of elements is out of bound %d.\n", ARRAY_ELEMENTS_MAX);
		return EXIT_FAILURE;
	}

	// device(0) : GTX 1080
	// device(1) : GTX 750Ti
	cudaSetDevice(0);
	
	unsigned int *A;
	A = (unsigned int*) malloc(sizeof(unsigned int) * elemCount );
	if (A == NULL) {
		printf("malloc failed.\n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
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
	printf("\n");
	printf("generated %u elements.\n\n", elemCount);
	fflush(stdout);

	// setup input copy on device mem
	unsigned int *devA;
	unsigned devACapa = align(elemCount,32);
	blockSize = min(blockSize, devACapa / 2);
	unsigned int gridSize = align(devACapa / 2, blockSize)/blockSize;
	cudaMalloc((void**)&devA, sizeof(unsigned int) * devACapa);
	cudaMemcpy(devA, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	printf("Going to use %d blocks, %d threas per block for array capacity %d.\n\n", gridSize, blockSize, devACapa);
	fflush(stdout);

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

	int firstFailure = elemCount;
	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < elemCount - 1) {
			if (A[i] > A[i + 1]) {
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
	if (firstFailure < elemCount) {
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
