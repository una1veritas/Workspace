
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "kernel.h"
#include "cuutils.h"

int main()
{

	cu_test_kernel <<<1, 10>>>();

	CUCHECK(cudaGetLastError());

	cudaDeviceReset();
    
    return 0;
}
