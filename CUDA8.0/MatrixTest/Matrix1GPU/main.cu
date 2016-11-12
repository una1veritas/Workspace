
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
/*
#include <time.h>
#include <cutil_inline.h>
*/

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_timer.h>

#define MATRIX_SIZE 1024/*行列１辺の数*/
#define BLOCK_SIZE 16

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int main(int argc, char** argv) {
	cudaError_t cudaStatus;
	StopWatchInterface *timer = NULL;

	unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;

	int* hMatrixA;
	int* hMatrixB;
	int* hMatrixC;
	hMatrixA = (int*)malloc(matrixSize);
	hMatrixB = (int*)malloc(matrixSize);

	/*初期値設定*/
	unsigned int col_idx, row_idx;
	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
			hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
		}
	}

	/*デバイス側の変数設定*/
	int* dMatrixA;
	int* dMatrixB;
	int* dMatrixC;

	/*デバイスメモリ領域の確保*/
	cudaMalloc((void**)&dMatrixA, matrixSize);
	cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dMatrixB, matrixSize);
	cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dMatrixC, matrixSize);

	/*ブロックサイズとグリッドサイズの設定*/
	dim3 block(1, 1);
	dim3 grid(1, 1);

	/*タイマーを作成して計測開始*/
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	/*カーネルの起動*/
	matrixMul << < grid, block >> >(dMatrixA, dMatrixB, dMatrixC);
	cudaThreadSynchronize();

	/*結果の領域確保とデバイス側からのメモリ転送*/
	hMatrixC = (int*)malloc(matrixSize);
	cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost);

	/*タイマーを停止しかかった時間を表示*/
	sdkStopTimer(&timer);
	printf("計算時間 =%f(ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return EXIT_FAILURE;
	}

	/*ホスト・デバイスメモリの開放*/
	free(hMatrixA);
	free(hMatrixB);
	free(hMatrixC);
	cudaFree(dMatrixA);
	cudaFree(dMatrixB);
	cudaFree(dMatrixC);

	return EXIT_SUCCESS;
}

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC) {
	unsigned int col_idx;
	unsigned int row_idx;
	unsigned int scan_idx;

	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
				inMatrixC[col_idx * MATRIX_SIZE + row_idx] += inMatrixA[col_idx * MATRIX_SIZE + scan_idx] *
					inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
			}
		}
	}
}