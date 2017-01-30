
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

static __device__ __forceinline__ unsigned int ntz32_cuda(unsigned int x)
{
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(x));
	return 31 - ret;
}

__device__ unsigned int ceil2pow32(unsigned int x) {
	if (x == 0)
		return 0;
	return 1 << (32 - ntz32_cuda(x - 1));

}

cudaError_t prefixScan(int *a, unsigned const int nsize);

__global__ void prefixKernel(int *a, const int width, const int nsize)
{
    int tidx = threadIdx.x;
	int pow2 = ceil2pow32(width); // 2^k - 1
	if (tidx == 0) {
		printf("pow2 = %d\n", pow2);
	}
	if (tidx < nsize) {
		a[tidx] = a[tidx];
		if ( ((tidx+1) & (pow2 - 1)) == 0) {
			a[tidx] = a[tidx - (pow2 >> 1)] + a[tidx];
		}
		
	}
	else {
		printf("I'm %d, stymied.\n", tidx);
	}
}

int main()
{
    int a[] = { 11, 21, 13, 24, 5 };
	const unsigned int arraySize = sizeof(a)/sizeof(int);

    // Add vectors in parallel.
    cudaError_t cudaStatus = prefixScan(a, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "prefscan failed!");
        return 1;
    }

    printf("{ 11, 21, 13, 24, 5 } = {%d,%d,%d,%d,%d}\n",
        a[0], a[1], a[2], a[3], a[4]);

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
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for vectors a and b.
	cudaStatus = cudaMalloc((void**)&dev_a, nsize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, nsize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	for(int width = 2; width < nsize; width <<= 1)
	    prefixKernel<<<1, nsize>>>(dev_a, width, nsize);

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
