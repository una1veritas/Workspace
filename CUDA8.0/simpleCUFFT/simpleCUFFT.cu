#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define NX      256
#define BATCH   10

int main(int argc, char *argv[])
{
	cufftHandle plan;
	cufftComplex *devPtr;
	cufftComplex data[NX*BATCH];
	int i;

	
	printf("[simpleCUFFT] is starting...\n");
	findCudaDevice(argc, (const char **)argv);
	
	if (argc != 3)
		return EXIT_FAILURE;

	/* å≥ÉfÅ[É^çÏê¨ */
	for (i = 0; i < NX*BATCH; i++) {
		data[i].x = 1.0f;
		data[i].y = 1.0f;
	}

	/* GPUópÉÅÉÇÉääÑÇËìñÇƒ */
	cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*BATCH);

	/* GPUópÉÅÉÇÉäÇ…ì]ëó */
	cudaMemcpy(devPtr, data, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);

	/* 1D FFT plançÏê¨ */
	cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);

	/* FFTèàóùé¿é{ */
	cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD);

	/* FFTèàóùé¿é{(ãtïœä∑) */
	/*
	cufftExecC2C(plan, devPtr, devPtr, CUFFT_INVERSE);
	*/

	/* åvéZåãâ ÇGPUÉÅÉÇÉäÇ©ÇÁì]ëó */
	cudaMemcpy(data, devPtr, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyDeviceToHost);

	/* CUFFT plançÌèú */
	cufftDestroy(plan);

	/* GPUópÉÅÉÇÉääJï˙ */
	cudaFree(devPtr);

	for (i = 0; i < NX*BATCH; i++) {
		printf("data[%d] %f %f\n", i, data[i].x, data[i].y);
	}

	return 0;
}