#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#include "cuutils.h"

#define ARRAY_ELEMENTS_MAX 0x7fffffff
//262144

////////////////////////////////////////////////////////////////////////////////
// Inline PTX call to return index of highest non-zero bit in a word
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ unsigned int bfind32(unsigned int ui32)
{
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(ui32));
	return ret;
}

__global__ void qsflo(unsigned int * x);

__global__ void exchEven(int *A, const unsigned int n);
__global__ void exchOdd(int *A, const unsigned int n);

__global__ void exch64Even(int *A, const unsigned int n);
__global__ void exch64Odd(int *A, const unsigned int n);

int main(const int argc, const char * argv[]) {
	unsigned int elemCount = 0;

	if (argc == 2) {
		elemCount = atoi(argv[1]);
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
	
	int *A;
	A = (int*) malloc(sizeof(unsigned int) * elemCount );
	if (A == NULL) {
		printf("malloc failed.\n");
		fflush(stdout);
		return EXIT_FAILURE;
	}
	// setup dummy input 
	srand(time(NULL));
	for (unsigned int i = 0; i < elemCount; i++) {
		A[i] = rand() % 10;
	}

	if (elemCount <= 16) {
		for (unsigned int i = 0; i < elemCount; i++) {
			if (i < 100 || i == elemCount - 1) {
				printf("%4u ", i);
			}
			else if (i == elemCount - 2) {
				printf("... ");
			}
		}
		printf("\n");
	}
	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < 100 || i == elemCount - 1) {
			printf("%4u ", A[i]);
		}
		else if (i == elemCount - 2) {
			printf("... ");
		}
	}
	printf("\n");
	unsigned int * tp;
	cudaMalloc((void**)&tp, sizeof(unsigned int));
	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < 100 || i == elemCount - 1) {
			unsigned int t = A[i];
			cudaMemcpy(tp, &A[i], sizeof(unsigned int), cudaMemcpyHostToDevice);
			qsflo<<<1,1>>>(tp);
			cudaMemcpy(&t, tp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			printf("%1u %1d,", t, nlz32_IEEE(A[i]));
		}
		else if (i == elemCount - 2) {
			printf("... ");
		}
	}
	printf("\n");
	printf("generated %u elements.\n\n", elemCount);

	fflush(stdout);

	// setup input copy on device mem
	int *devArray;
	unsigned int blockSize = 32;
	unsigned devACapa = 64 * MAX(cdiv32(elemCount,64),1);
	unsigned int gridSize = (devACapa >> 1) / blockSize;
	cudaMalloc((void**)&devArray, sizeof(unsigned int) * devACapa);
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	printf("Going to use %d blocks of %d threads for array capacity %d.\n\n", gridSize, blockSize, devACapa);
	fflush(stdout);

	dim3 grids(gridSize), blocks(blockSize);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (unsigned int i = 0; i < (elemCount>>1); i++) {
		exchEven << < grids, blocks >> > (devArray, elemCount);
		exchOdd << < grids, blocks >> > (devArray, elemCount); // possibly the last call is redundant
	}
	
	sdkStopTimer(&timer);
	printf("Elapsed time %f msec.\n\n", (float)((int)(sdkGetTimerValue(&timer) * 1000)) / 1000);


	printf("exch64\n");
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (unsigned int i = 0; i < (elemCount >> 1); i++) {
		exch64Even << < grids, blocks >> > (devArray, elemCount);
		exch64Odd << < grids, blocks >> > (devArray, elemCount); // possibly the last call is redundant
	}

	sdkStopTimer(&timer);
	printf("Elapsed time %f msec.\n\n", (float)((int)(sdkGetTimerValue(&timer) * 1000)) / 1000);
	sdkDeleteTimer(&timer);

	cudaMemcpy(A, devArray, sizeof(unsigned int) * devACapa, cudaMemcpyDeviceToHost);

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


	cudaFree(devArray);
	free(A);

	cudaDeviceReset();

}

__global__ void qsflo(unsigned int * x) {
	*x = bfind32(*x);
}

__global__ void exchEven(int *a, const unsigned int n) {
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

__global__ void exchOdd(int *a, const unsigned int n) {
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


__global__ void exch64Even(int *a, const unsigned int n) {
	__shared__ int sa[64];
	unsigned int bid = blockIdx.x;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int left = tid << 1;
	unsigned int tmp;

	if (left + 1 < n) {
		if (a[left] > a[left + 1]) {
			tmp = a[left];
			a[left] = a[left + 1];
			a[left + 1] = tmp;
		}
	}
	__syncthreads();
}

__global__ void exch64Odd(int *a, const unsigned int n) {
	__shared__ int sa[64];
	unsigned int bid = blockIdx.x;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int left = (tid << 1) + 1;
	unsigned int tmp;

	if (left + 1 < n) {
		if (a[left] > a[left + 1]) {
			tmp = a[left];
			a[left] = a[left + 1];
			a[left + 1] = tmp;
		}
	}
	__syncthreads();
}
