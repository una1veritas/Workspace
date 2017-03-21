/*
 */
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#include "cu_utils.h"

#include "cdpHeapsort.h"

__device__ uint32 c2pow32dev(uint32 x) {
	return (x != 0) * (1 << (32 - __clz(x - 1)));

}

__device__ uint32 clog32dev(uint32 x) {
	return (x != 0) * (32 - __clz(x - 1));
}


void cu_makeheap(int *devArray, const unsigned int nsize) {
	int *t = new int[nsize];
	unsigned int maxparid = (nsize >> 1) - 1;
	printf("size = %d, maxparid = %d\n", nsize, maxparid);
	for (int i = 0; i < clog32(nsize); ++i) {
		printf("iteration %d\n", i);
		dev_makeheap << <1, maxparid + 1 >> > (devArray, nsize, i);
		cudaMemcpy(&t,&devArray, nsize*sizeof(int), cudaMemcpyDeviceToHost);
		for (int j = 0; j < nsize; ++j) {
			printf("%d, ", t[j]);
		}
		printf("\n\n");
	}
	delete[] t;
}

__global__ void dev_makeheap(int *a, const unsigned int n, const unsigned int parity) {
	const unsigned int thrid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int level = clog32dev(thrid + 2) - 1;
	unsigned int chid;
	int t;

	if (((level^parity) & 1) == 0) {
		printf("a[%d] @ level %d -> a[%d], a[%d]\n", thrid, level, ((thrid + 1) << 1) - 1, ((thrid + 1) << 1));
		if (a[((thrid + 1) << 1) - 1] > a[((thrid + 1) << 1)])
			chid = ((thrid + 1) << 1) - 1;
		else
			chid = ((thrid + 1) << 1);
		if (a[chid] > a[thrid]) {
			t = a[thrid];
			a[thrid] = a[chid];
			a[chid] = t;
		}
	}
	__syncthreads();
}

