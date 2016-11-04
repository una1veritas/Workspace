#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

/*n正方行列のサイズを定義*/
#define MATRIX_SIZE 1024

int main(int argc, char** argv) {
	unsigned int col_idx, row_idx, scan_idx;
	int* matA;
	int* matB;
	int* matC;

	/*タイマー作成*/
	time_t Start, Stop;

	/*int型のn×n領域をメモリに確保*/
	matA = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	matB = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	matC = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			matA[col_idx * MATRIX_SIZE + row_idx] = rand() % (MATRIX_SIZE * MATRIX_SIZE);
			matB[col_idx * MATRIX_SIZE + row_idx] = rand() % (MATRIX_SIZE * MATRIX_SIZE);
			matC[col_idx * MATRIX_SIZE + row_idx] = 0;
		}
	}

	time(&Start);

	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++) {
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++) {
			for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
				matC[col_idx * MATRIX_SIZE + row_idx] += matA[col_idx * MATRIX_SIZE + scan_idx] *
					matB[scan_idx * MATRIX_SIZE + row_idx];
			}
		}
	}

	time(&Stop);
	printf("Processing time: %d (sec)\n", Stop - Start);

	/*メモリを解放*/
	free(matA);
	free(matB);
	free(matC);

	return 0;
}
