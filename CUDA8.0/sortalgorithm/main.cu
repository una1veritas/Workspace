#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <helper_timer.h>

#include "cu_utils.h"

#include "oddevensort.h"
#include "cdpQuicksort.h"
#include "cdpHeapsort.h"

#define ARRAY_ELEMENTS_MAX 0x7fffffff
//262144

int comp_int(const void *a, const void *b) {
	return *(int*)a - *(int*)b;
}

bool checkHeap(int * a, const unsigned int n);

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
	
	int * A = new int[elemCount];
	//	A = (int*) malloc(sizeof(unsigned int) * elemCount );
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
	unsigned devACapa = 128 * MAX(CDIV(elemCount,128),1);
	cudaMalloc((void**)&devArray, sizeof(unsigned int) * devACapa);
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	cuStopWatch sw;
	
	printf("Sort by oddevensort_gmem..\n");

	sw.reset();
	sw.start();
	
	cu_oddevensort_gmem(devArray, elemCount);

	cudaDeviceSynchronize();

	sw.stop();
	printf("Elapsed time %f msec.\n\n", (float)((int)(sw.timerValue() * 1000)) / 1000);


	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	printf("Sort by oddevensort_smem...\n");

	sw.reset();
	sw.start();

	cu_oddevensort(devArray, elemCount);

	sw.stop();
	printf("Elapsed time %f msec.\n\n", (float)((int)(sw.timerValue() * 1000)) / 1000);

	printf("Sort by qsort in stdlib...\n");

	sw.reset();
	sw.start();

	qsort(A, elemCount, sizeof(int), comp_int);

	sw.stop();
	printf("Elapsed time %f msec.\n\n", (float)((int)(sw.timerValue() * 1000)) / 1000);

	printf("Sort by cdp_qsort...\n");
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	sw.reset();
	sw.start();

	cdp_qsort(devArray, elemCount);

	sw.stop();
	printf("Elapsed time %f msec.\n\n", (float)((int)(sw.timerValue() * 1000)) / 1000);

	
	printf("Sort by cdp_heapsort...\n");
	cudaMemcpy(devArray, A, sizeof(unsigned int) * devACapa, cudaMemcpyHostToDevice);

	sw.reset();
	sw.start();

	cu_heapsort(devArray, elemCount);

	sw.stop();
	printf("Elapsed time %f msec.\n\n", (float)((int)(sw.timerValue() * 1000)) / 1000);
	
	cudaMemcpy(A, devArray, sizeof(unsigned int) * devACapa, cudaMemcpyDeviceToHost);

	unsigned int firstFailure = elemCount;
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
	else {
		printf("sort succeeded.\n");
	}

	cudaFree(devArray);
	
	delete [] A;

	cudaDeviceReset();

}
