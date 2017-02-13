#pragma once
#ifndef __CU_UTILS_H__
#define __CU_UTILS_H__

/* */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define cuCheckErrors(call)										\
{															\
	const cudaError_t cuerror = call;						\
	if ( cuerror != cudaSuccess )							\
	{														\
		printf("Error: %s: %d, ", __FILE__, __LINE__);		\
		printf("code %d, %s\n", cuerror, cudaGetErrorString(cuerror));	\
		exit(1);											\
	}														\
}

#define CDIV(x, y) ( ((x) != 0)*(1 + (((x) - 1) / (y))) )

/* difference or zero */
#define DOZ(x, y) (((x)-(y))&-((x)>=(y)))

__device__ __forceinline__ unsigned int bfind32(unsigned int x)
{
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(x));
	return ret;
}

#define NLZ32_DEV(x) __clz(x)
#define NLZ32(x) nlz32(x)

unsigned int nlz32(unsigned int x);

#endif /* __CU_UTILS_H__ */
