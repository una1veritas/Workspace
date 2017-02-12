#pragma once
#ifndef __CU_UTILS_H__
#define __CU_UTILS_H__

/* */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/* --------------------- */
#define IEEE_FLOATING_POINT
#define INTEL_SSE42
#define COMPILER_NVCC
#define CUDA_DEVICE

#ifdef IEEE_FLOATING_POINT
#define nlz32 nlz32_IEEE
#endif

#if  defined(INTEL_SSE42) && defined(COMPILER_GCC)
#define pop32 pop32_SSE42
#define ntz32 ntz32_
#endif

#define pop32 pop32_
#define ntz32 ntz32_

/* --------------------- */

#include <stdint.h>

typedef uint32_t 	uint32;
typedef int32_t 	int32;
typedef uint64_t 	uint64;
typedef int64_t		int64;

/* popularity (1 bits) count */
int pop32_(uint32 bits);
int pop32_SSE42(uint32 bits);

/* count leading zero */
uint32 nlz32_(uint32 x);
uint32 nlz32_IEEE(uint32 x);

/* count trailing zeros */
uint32 ntz32_(uint32 x);

/* smallest 2 to the nth power that is no less than x */
uint32 c2pow32(uint32 x);	// ceiling of 2 to the power of x within uint32

uint32 flog32(uint32 x);	// floor of log_2 
uint32 clog32(uint32 x);	// ceiling of log_2
uint32 bitsize32(int32 x);	// the number of bits to describe x

							/* difference or zero */
#define DOZ(x, y) (((x)-(y))&-((x)>=(y)))
#define MIN(x, y) ((x) - DOZ(x,y))
#define MAX(x, y) ((y) + DOZ(x,y))

int32 doz32(int32 x, int32 y);
int32 min32(int32 x, int32 y);
int32 max32(int32 x, int32 y);

/* swap values in two variables */
#define SWAP(x, y) { x ^= y; y ^= x; x ^= y; }

void swap32_(int32 * x, int32 * y);

uint32 cdiv32(uint32 x, uint32 y);

namespace cu {

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

}

#endif /* __CU_UTILS_H__ */
