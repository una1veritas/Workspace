/*
 */
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#include "cu_utils.h"

#include "oddevensort.h"

void oddevensort_gmem(int *devArray, const unsigned int nsize) {
	unsigned int threadsperblock = 192;
	unsigned devACapa = 192 * MAX(CDIV(nsize, 192), 1);
	unsigned int blockspergrid = CDIV(devACapa, threadsperblock * 2);

	for (unsigned int i = 0; i < (nsize >> 1); i++) {
		exchEven << < blockspergrid, threadsperblock >> > (devArray, nsize);
		exchOdd << < blockspergrid, threadsperblock >> > (devArray, nsize); // possibly the last call is redundant
	}
	//cudaDeviceSynchronize();
}

void oddevensort_smem(int *devArray, const unsigned int nsize) {
	unsigned devACapa = 256 * MAX(CDIV(nsize, 256), 1);
	for (unsigned int i = 0; i < CDIV(nsize, 128); i++) {
		//printf("exch64 %d (%d)\n", (i & 1) * 64, i);
		exch256 << < CDIV(devACapa, 256), 128 >> > (devArray, nsize, (i & 1) * 128);
	}
	//cudaDeviceSynchronize();
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


__global__ void exch256(int *a, const unsigned int n, const unsigned int offset) {
	__shared__ int sa[256];
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

	for (int i = 0; i < 128; i++) {
		if (left + 1 < n) {
			if (sa[sleft] > sa[sleft + 1]) {
				tmp = sa[sleft];
				sa[sleft] = sa[sleft + 1];
				sa[sleft + 1] = tmp;
			}
		}
		__syncthreads();
		if (left + 2 < n && sleft + 2 < 256) {
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
