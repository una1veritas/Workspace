/*
 */
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#include "cu_utils.h"

#include "cdpHeapsort.h"

__global__ void checkHeap(int * a, const unsigned int n) {
	bool result = true;
	printf("heap size = %d\n",n);
	for (int i = 0; i < (n >> 1); ++i) {
		if (a[i] < a[(i << 1) + 1]) {
			printf("check heap failed: a[%d] = %d (lv. %d) < a[%d] = %d (left)\n", i, a[i], clog32dev(i + 2) - 1, (i << 1) + 1, a[(i << 1) + 1]);
			result = false;
		}
		if ( (i << 1) + 2 < n && a[i] < a[(i << 1) + 2]) {
			printf("check heap failed: a[%d] = %d (lv. %d) < a[%d] = %d (right)\n", i, a[i], clog32dev(i + 2) - 1, (i << 1) + 2, a[(i << 1) + 2]);
			result = false;
		}
	}
	if (result) {
		printf("check heap passed.\n");
	}
	//return result;
}

__global__ void showHeap(int * a, const unsigned int heapsize, const unsigned int n) {
	for (int i = 0; i < n; ++i) {
		printf("%d",a[i]);
		if (i + 1 == heapsize) {
			printf("; ");
		}
		else if ( i + 1 < n ) {
				printf(", ");
		}
		else {
			printf(".");
		}
		if (i + 1 == n)
			printf("\n\n");
	}
}

void cu_heapsort(int * devArray, const unsigned int nsize) {
	unsigned int heapsize = nsize;
	unsigned int blksize = 255;
	unsigned int blkcount;

	cu_makeheap(devArray, heapsize);
	while (heapsize > 1) {
		//checkHeap<<<1,1>>>(devArray, heapsize);
		//showHeap << <1, 1 >> >(devArray, heapsize, nsize);
		dev_swapheaptop<<<1,1>>>(devArray,heapsize);
		--heapsize;
		//showHeap << <1, 1 >> >(devArray, heapsize, nsize);
		blkcount = CDIV((heapsize >> 1), blksize);
		dev_shakeheap << <blkcount, blksize >> > (devArray, heapsize, 0);
		dev_shakeheap << <blkcount, blksize >> > (devArray, heapsize, 1);
	}
}

__global__ void dev_swapheaptop(int * a, const unsigned int heapsize) {
	//printf("swap a[%d] = %d <-> a[%d] = %d\n", 0, a[0], heapsize - 1, a[heapsize - 1]);
	int t = a[0];
	a[0] = a[heapsize - 1];
	a[heapsize - 1] = t;
}

void cu_makeheap(int *devArray, const unsigned int nsize) {
//	int *t = new int[nsize];
	unsigned int parents = (nsize >> 1);
	unsigned int blksize = 255;
	unsigned int blkcount = CDIV(parents,blksize);
	//printf("make heap size = %d, parents = %d, block size = %d, num of blocks = %d\n", nsize, parents, blksize, blkcount);
	for (unsigned int i = 0; i < clog32(nsize) ; ++i) {
		//printf("iteration %d\n", i);
		dev_shakeheap <<<blkcount, blksize >> > (devArray, nsize, i+1);
		/*
		cudaMemcpy(t,devArray, nsize*sizeof(int), cudaMemcpyDeviceToHost);
		for (int j = 0; j < nsize; ++j) {
			printf("%d, ", t[j]);
		}
		printf("\n\n");
		*/
		dev_shakeheap << <blkcount, blksize >> > (devArray, nsize, i);
	}
//	delete[] t;
}

__global__ void dev_shakeheap(int *a, const unsigned int n, const unsigned int parity) {
	const unsigned int thrid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int level = clog32dev(thrid + 2) - 1;
	unsigned int chid = (thrid << 1) + 1; // firstly, chid is the left child
	int t;

	if ( ((level^parity) & 1) == 0 ) { 
		if ( chid < n ) {
			// if a[thrid] has, at least, the left child
			if ( chid + 1 < n) {
				if ( a[chid + 1] > a[chid]) {
					// if a[thrid] has the right child and the right child is greater than the left 
					chid = chid + 1;
				}
			} 
			if ( a[chid] > a[thrid] ) {
				t = a[thrid];
				a[thrid] = a[chid];
				a[chid] = t;
			}
		}
	}
	__syncthreads();
}

__global__ void dev_shakeheap255(int *a, const unsigned int n, const unsigned int parity) {
	const unsigned int thrid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int level = clog32dev(thrid + 2) - 1;
	unsigned int chid = (thrid << 1) + 1; // firstly, chid is the left child
	int t;

	if (((level^parity) & 1) == 0) {
		if (chid < n) {
			// if a[thrid] has, at least, the left child
			if (chid + 1 < n) {
				if (a[chid + 1] > a[chid]) {
					// if a[thrid] has the right child and the right child is greater than the left 
					chid = chid + 1;
				}
			}
			if (a[chid] > a[thrid]) {
				t = a[thrid];
				a[thrid] = a[chid];
				a[chid] = t;
			}
		}
	}
	__syncthreads();
}

