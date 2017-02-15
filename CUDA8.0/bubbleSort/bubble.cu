#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#include "cu_utils.h"

#define ARRAY_ELEMENTS_MAX 0x7fffffff
//262144

__global__ void exchEven(int *A, const unsigned int n);
__global__ void exchOdd(int *A, const unsigned int n);

__global__ void exch128(int *A, const unsigned int n, const unsigned int offset);

int comp_int(const void *a, const void *b) {
	return *(int*)a - *(int*)b;
}

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
		A[i] = (rand() % 10000);
	}

	if (elemCount <= 16) {
		for (unsigned int i = 0; i < elemCount; i++) {
			if (i < 70 || i == elemCount - 1) {
				printf("%4u ", i);
			}
			else if (i == elemCount - 2) {
				printf("... ");
			}
		}
		printf("\n");
	}
	for (unsigned int i = 0; i < elemCount; i++) {
		if (i < 70 || i == elemCount - 1) {
			printf("%4d ", A[i]);
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
	unsigned int threadsperblock = 192;
	unsigned devACapa = 192 * MAX(CDIV(elemCount,192),1);
	unsigned int blockspergrid = CDIV(devACapa, threadsperblock *2);
	cudaMalloc((void**)&devArray, sizeof(unsigned int) * devACapa);
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	printf("Going to use %d blocks of %d threads for array capacity %d.\n\n", blockspergrid, threadsperblock, devACapa);
	fflush(stdout);


	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (unsigned int i = 0; i < (elemCount>>1); i++) {
		exchEven << < blockspergrid, threadsperblock >> > (devArray, elemCount);
		exchOdd << < blockspergrid, threadsperblock >> > (devArray, elemCount); // possibly the last call is redundant
	}

	cudaDeviceSynchronize();

	sdkStopTimer(&timer);
	printf("Elapsed time %f msec.\n\n", (float)((int)(sdkGetTimerValue(&timer) * 1000)) / 1000);


	printf("Sort by exch64...\n");
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	threadsperblock = 64;
	blockspergrid = CDIV(devACapa, threadsperblock * 2);
	printf("Going to use %d blocks of %d threads for array capacity %d.\n\n", blockspergrid, threadsperblock, devACapa);
	fflush(stdout);

	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	for (unsigned int i = 0; i < blockspergrid << 1; i++) {
		//printf("exch128 %d (%d)\n", (i & 1) * threadsperblock, i);
		exch128<<< blockspergrid, threadsperblock >>>(devArray, elemCount, (i & 1) * threadsperblock);
	}

	sdkStopTimer(&timer);
	printf("Elapsed time %f msec.\n\n", (float)((int)(sdkGetTimerValue(&timer) * 1000)) / 1000);

	printf("Going to sort array of %d int's by qsort in stdlib.\n\n", elemCount);
	fflush(stdout);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	qsort(A, elemCount, sizeof(int), comp_int);

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
		if (i < 70 || i == elemCount - 1) {
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

	cudaFree(devArray);
	free(A);

	cudaDeviceReset();

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


__global__ void exch128(int *a, const unsigned int n, const unsigned int offset) {
	__shared__ int sa[128];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int left = (tid << 1) + offset;
	unsigned int sleft = threadIdx.x << 1;
	unsigned int tmp;

	if (left < n) {
		sa[sleft] = a[left];
	}
	if (left + 1 < n) {
		sa[sleft + 1] = a[left + 1];
	}

	__syncthreads();
	for (int i = 0; i < 64; i++) {
		if (left + 1 < n) {
			if (sa[sleft] > sa[sleft + 1]) {
				tmp = sa[sleft];
				sa[sleft] = sa[sleft + 1];
				sa[sleft + 1] = tmp;
			}
		}
		__syncthreads();
		if (left + 2 < n && sleft + 2 < 128) {
			if (sa[sleft + 1] > sa[sleft + 2]) {
				tmp = sa[sleft + 1];
				sa[sleft + 1] = sa[sleft + 2];
				sa[sleft + 2] = tmp;
			}
		}
		__syncthreads();
	}

	if (left < n) {
		a[left] = sa[sleft];
	}
	if (left + 1 < n) {
		a[left + 1] = sa[sleft + 1];
	}
}
