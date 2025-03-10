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

	/* 元データ作成 */
	for (i = 0; i < NX*BATCH; i++) {
		data[i].x = 1.0f;
		data[i].y = 1.0f;
	}

	/* GPU用メモリ割り当て */
	cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*BATCH);

	/* GPU用メモリに転送 */
	cudaMemcpy(devPtr, data, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);

	/* 1D FFT plan作成 */
	cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);

	/* FFT処理実施 */
	cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD);

	/* FFT処理実施(逆変換) */
	/*
	cufftExecC2C(plan, devPtr, devPtr, CUFFT_INVERSE);
	*/

	/* 計算結果をGPUメモリから転送 */
	cudaMemcpy(data, devPtr, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyDeviceToHost);

	/* CUFFT plan削除 */
	cufftDestroy(plan);

	/* GPU用メモリ開放 */
	cudaFree(devPtr);

	for (i = 0; i < NX*BATCH; i++) {
		printf("data[%d] %f %f\n", i, data[i].x, data[i].y);
	}

	return 0;
}