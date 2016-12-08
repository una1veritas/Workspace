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

#define ARRAY_SIZE 512
//262144
#define BLOCK_SIZE 1024

__global__ void bubSortEven(unsigned int *A, unsigned int *B);
__global__ void bubSortOdd(unsigned int *A, unsigned int *B);

int main(void) {

	if (((ARRAY_SIZE - 1) & ARRAY_SIZE) != 0) {
		printf("ARRAY_SIZE --- exponent of 2\n");
		return 0;
	}

	// device(0) : GTX 1080
	// device(1) : GTX 750Ti
	cudaSetDevice(0);

	unsigned int  arraySize = sizeof(unsigned int) * ARRAY_SIZE;
	
	unsigned int *hA, *hB;
	hA = (unsigned int*)malloc(arraySize);
	hB = (unsigned int*)malloc(arraySize);

	unsigned int i;
	
	srand(time(NULL));

	for (i = 0; i < ARRAY_SIZE; i++) {
		hA[i] = rand() % 1024;

	}

	for (i = 0; i < 24 /*100*/; i++)
		printf("%d, ", hA[i]);
	printf("\n");
			
	unsigned int *dA, *dB;
	cudaMalloc((void**)&dA, arraySize);
	cudaMalloc((void**)&dB, arraySize);

	cudaMemcpy(dA, hA, arraySize, cudaMemcpyHostToDevice);

	dim3 grid = ARRAY_SIZE / (2 * BLOCK_SIZE);
	dim3 block = BLOCK_SIZE;

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	//for (i = 0; i < (ARRAY_SIZE / 2); i++) {
		bubSortEven << < grid, block >> > (dA, dB);
		cudaThreadSynchronize();
		bubSortOdd << < grid, block >> > (dB, dA);
		cudaThreadSynchronize();
	//}
	
	cudaMemcpy(hA, dA, arraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(hB, dB, arraySize, cudaMemcpyDeviceToHost);

	sdkStopTimer(&timer);

	for (i = 0; i < 24 /*ARRAY_SIZE*/; i++) {
		//printf("[%d] = %u\n", i, hA[i]);
		printf("%d, ", hB[i]);
	}
	printf("\n");
	printf("[%d] = %u\n", ARRAY_SIZE - 1, hA[ARRAY_SIZE - 1]);

	printf("[TIME] =%f(ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	free(hA);
	free(hB);
	cudaFree(dA);
	cudaFree(dB);

	cudaThreadExit();

}

__global__ void bubSortEven(unsigned int *bef, unsigned int *aft) {

	unsigned int fro_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	unsigned int beh_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

	if (bef[fro_idx] > bef[beh_idx]) {
		aft[fro_idx] = bef[beh_idx];
		aft[beh_idx] = bef[fro_idx];
	}
	else {
		aft[fro_idx] = bef[fro_idx];
		aft[beh_idx] = bef[beh_idx];
	}
	__syncthreads();
}

__global__ void bubSortOdd(unsigned int *bef, unsigned int *aft) {

	unsigned int fro_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;
	unsigned int beh_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 2;

	if (bef[fro_idx] > bef[beh_idx] && beh_idx != ARRAY_SIZE) {
		aft[fro_idx] = bef[beh_idx];
		aft[beh_idx] = bef[fro_idx];
	}
	else if (beh_idx != ARRAY_SIZE){
		aft[fro_idx] = bef[fro_idx];
		aft[beh_idx] = bef[beh_idx];
	}

	aft[0] = bef[0];
	aft[ARRAY_SIZE - 1] = bef[ARRAY_SIZE - 1];

	__syncthreads();
}