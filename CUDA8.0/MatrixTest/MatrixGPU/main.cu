
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

#define MATRIX_SIZE 1024/*�s��P�ӂ̐�*/
#define BLOCK_SIZE 16

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int main(int argc, char** argv) {
	unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;

	int* hMatrixA;
	int* hMatrixB;
	int* hMatrixC;
	hMatrixA = (int*)malloc(matrixSize);
	hMatrixB = (int*)malloc(matrixSize);

	/*�����l�ݒ�*/
	unsigned int col_idx, row_idx;
	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
			hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
		}
	}

	/*�f�o�C�X���̕ϐ��ݒ�*/
	int* dMatrixA;
	int* dMatrixB;
	int* dMatrixC;

	/*�f�o�C�X�������̈�̊m��*/
	cudaMalloc((void**)&dMatrixA, matrixSize);
	cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dMatrixB, matrixSize);
	cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dMatrixC, matrixSize);

	/*�u���b�N�T�C�Y�ƃO���b�h�T�C�Y�̐ݒ�*/
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);

	/*�^�C�}�[���쐬���Čv���J�n*/
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);

	/*�J�[�l���̋N��*/
	matrixMul <<< grid, block >>>(dMatrixA, dMatrixB, dMatrixC);
	cudaThreadSynchronize();

	/*���ʂ̗̈�m�ۂƃf�o�C�X������̃������]��*/
	hMatrixC = (int*)malloc(matrixSize);
	cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost);

	/*�^�C�}�[���~�������������Ԃ�\��*/
	sdkStopTimer(&timer);
	printf("�v�Z���� =%f(ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	/*�z�X�g�E�f�o�C�X�������̊J��*/
	free(hMatrixA);
	free(hMatrixB);
	free(hMatrixC);
	cudaFree(dMatrixA);
	cudaFree(dMatrixB);
	cudaFree(dMatrixC);

	/*�I������*/
	// Wait for the stop event to complete
	cudaThreadExit();
	/*
	cutilExit(argc, argv);
	*/
}

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC) {
	unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int scan_idx;
	unsigned int target = 0;

	/*�s��̉��Z���s��*/
	for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
		target += inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
		__syncthreads();
	}
	inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}