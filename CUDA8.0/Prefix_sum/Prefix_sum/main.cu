
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


static __device__ __forceinline__ unsigned int bfind32_cuda(unsigned int x)
{
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(x));
	return 31 - ret;
}

__device__ __host__ unsigned int nlz32_IEEE(unsigned int x)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., 5.3, p. 104 -- */
	double d = x;
	d += 0.5;
	unsigned int *p = ((unsigned int*)&d) + 1;
	return 0x41e - (*p >> 20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}

__device__ __host__ unsigned int ceil2pow32(unsigned int x) {
	return (-(x != 0)) & (1 << (32 - nlz32_IEEE(x - 1)));
}

cudaError_t prefixScan(int *a, unsigned const int nsize);

__global__ void prefscan(int *a, const int width)
{
    int thidx = threadIdx.x;
	if ( !(thidx < (width>>1)) ) {
		a[thidx] = a[thidx - (width >> 1)] + a[thidx];	
	}
}

int main()
{
    int a[] = { 11, 21, 13, 24, 8, -3, 15, 31 };
	const unsigned int arraySize = sizeof(a)/sizeof(int);

	printf("{ ");
	for (int i = 0; i < arraySize; i++) {
		printf("%3d, ", a[i]);
	}
	printf("} = \n");
	// Add vectors in parallel.
    cudaError_t cudaStatus = prefixScan(a, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "prefscan failed!");
        return 1;
    }

	
	printf("{ ");
	for (int i = 0; i < arraySize; i++) {
		printf("%3d, ", a[i]);
	}
	printf("}\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t prefixScan(int *a, unsigned const int nsize) 
{
    int *dev_a = 0;
	unsigned int arraySize = ceil2pow32(nsize);
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for vectors a and b.
	cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, nsize * sizeof(int), cudaMemcpyHostToDevice);
	if ( nsize < arraySize)
		cudaStatus = cudaMemset(dev_a + nsize, 0, arraySize - nsize);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy/Memset failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	for(int width = 2; width <= arraySize; width <<= 1)
	    prefscan<<<1, arraySize>>>(dev_a, width);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "prefixScanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_a, nsize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    
    return cudaStatus;
}
