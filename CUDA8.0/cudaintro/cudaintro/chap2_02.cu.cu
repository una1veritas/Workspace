
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define N 256

__global__ void matrix_vector_multi(float *A_d, float * B_d, float *C_d) {
	int i, j;

	for (j = 0; i < N; j++) {
		A_d[j] = 0.0F;

		for (i = 0; i < N; i++) {
			A_d[j] = A_d[j] + B_d[j*N + i] * C_d[i];
		}
	}
		
}

int main(void) {
	int i, j;
	float A[N], B[N*N], C[N];
	float *A_d, *B_d, *C_d;

	dim3 blocks(1, 1, 1);
	dim3 threads(1, 1, 1);

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

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}
/*
==11856== Profiling application: .\Debug\chap2_02.exe
==11856== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
73.21%  133.63us         1  133.63us  133.63us  133.63us  matrix_vector_multi(float*, float*, float*)
25.46%  46.465us         2  23.232us  1.1840us  45.281us  [CUDA memcpy HtoD]
1.33%  2.4320us         1  2.4320us  2.4320us  2.4320us  [CUDA memcpy DtoH]

==11856== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
98.47%  132.94ms         3  44.313ms  13.998us  132.47ms  cudaMalloc
0.47%  633.40us        91  6.9600us       0ns  278.91us  cuDeviceGetAttribute
0.46%  621.85us         3  207.28us  29.745us  321.25us  cudaFree
0.40%  539.97us         3  179.99us  106.38us  270.86us  cudaMemcpy
0.13%  181.97us         1  181.97us  181.97us  181.97us  cuDeviceGetName
0.04%  56.341us         1  56.341us  56.341us  56.341us  cudaLaunch
0.01%  11.198us         1  11.198us  11.198us  11.198us  cuDeviceTotalMem
0.01%  9.0990us         1  9.0990us  9.0990us  9.0990us  cudaConfigureCall
0.00%  3.1500us         3  1.0500us     350ns  1.7500us  cuDeviceGetCount
0.00%  3.1490us         3  1.0490us     350ns  2.0990us  cuDeviceGet
0.00%  1.7490us         3     583ns     350ns     700ns  cudaSetupArgument
*/