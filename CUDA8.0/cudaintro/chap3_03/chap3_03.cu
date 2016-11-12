
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define N 256

__global__ void matrix_vector_multi(float *A_d, float * B_d, float *C_d) {
	int i, j;
	
	A_d[threadIdx.x] = 0.0F;

	for (i = 0; i < N; i++) {
		A_d[threadIdx.x] = A_d[threadIdx.x] + B_d[threadIdx.x*N + i] * C_d[i];
	}

}

int main(void) {
	int i, j;
	float A[N], B[N*N], C[N];
	float *A_d, *B_d, *C_d;

	dim3 blocks(1, 1, 1);
	dim3 threads(256, 1, 1);

	for (j = 0; j < N; j++) {
		for (i = 0; i < N; i++) {
			B[j*N + i] = ((float)j) / 256.0;
		}
	}

	for (j = 0; j < N; j++)
		C[j] = 1.0F;

	cudaMalloc((void**)& A_d, N * sizeof(float));
	cudaMalloc((void**)& B_d, N*N * sizeof(float));
	cudaMalloc((void**)& C_d, N * sizeof(float));

	cudaMemcpy(B_d, B, N*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_d, C, N * sizeof(float), cudaMemcpyHostToDevice);

	matrix_vector_multi <<< blocks, threads >>> (A_d, B_d, C_d);

	cudaMemcpy(A, A_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (j = 0; j < N; j++) {
		printf("A[%d]=%f \n", j, A[j]);
	}

	printf("N = %d\n", N);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

/*
N = 256
==8672== Profiling application: .\Debug\chap3_03.exe
==8672== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
88.48%  369.76us         1  369.76us  369.76us  369.76us  matrix_vector_multi(float*, float*, float*)
10.97%  45.825us         2  22.912us  1.1840us  44.641us  [CUDA memcpy HtoD]
0.55%  2.3040us         1  2.3040us  2.3040us  2.3040us  [CUDA memcpy DtoH]

==8672== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
98.14%  120.12ms         3  40.039ms  6.9990us  119.75ms  cudaMalloc
0.63%  775.83us         3  258.61us  106.38us  510.92us  cudaMemcpy
0.52%  633.40us        91  6.9600us       0ns  279.26us  cuDeviceGetAttribute
0.48%  586.86us         3  195.62us  26.945us  304.80us  cudaFree
0.16%  198.07us         1  198.07us  198.07us  198.07us  cuDeviceGetName
0.05%  55.641us         1  55.641us  55.641us  55.641us  cudaLaunch
0.01%  8.3990us         1  8.3990us  8.3990us  8.3990us  cudaConfigureCall
0.01%  7.6990us         1  7.6990us  7.6990us  7.6990us  cuDeviceTotalMem
0.00%  3.4990us         3  1.1660us     350ns  1.7490us  cuDeviceGetCount
0.00%  3.1490us         3  1.0490us     700ns  1.7490us  cuDeviceGet
0.00%  2.1000us         3     700ns     350ns  1.0500us  cudaSetupArgument
*/