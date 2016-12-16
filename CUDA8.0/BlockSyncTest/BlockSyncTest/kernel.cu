
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void cu_test_kernel(void) {
    int i = threadIdx.x;
	printf("I am %d.\n", i);
}

