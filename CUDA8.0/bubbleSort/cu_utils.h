#pragma once
#ifndef __CU_UTILS_H__
#define __CU_UTILS_H__

#include <stdint.h>

typedef uint32_t 	uint32;
typedef int32_t 	int32;
typedef uint64_t 	uint64;
typedef int64_t		int64;

/* */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h> // helper functions for SDK examples


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

#define NLZ32DEV(x) __clz(x)
#define NLZ64DEV(x) __clzll(x)
#define POPC32DEV(x) __popc(x)
#define POPC64DEV(x) __popcll(x)

unsigned int nlz32_IEEEFP(unsigned int x);

#define NLZ32(x)  nlz32_IEEEFP(x)

/* popularity (1 bits) count */
uint32 pop32_SSE42(uint32 bits); /* int pop32_SSE42(uint32 bits); */
uint32 pop32_HD(uint32 bits);

/* smallest 2 to the nth power that is no less than x */
uint32 c2pow32(uint32 x);

uint32 flog32(uint32 x);
uint32 clog32(uint32 x);

/* the number of digits including the single leading 0 at msb. */
uint32 bitsize32(int32 x);


struct cuStopWatch {
	StopWatchInterface *timer = NULL;

	cuStopWatch(void) {
		timer = NULL;
		sdkCreateTimer(&timer);
	}

	~cuStopWatch(void) {
		sdkDeleteTimer(&timer);
	}

	void reset(void) {
		sdkResetTimer(&timer);
	}

	void start(void) {
		sdkStartTimer(&timer);
	}

	void stop(void) {
		sdkStopTimer(&timer);
	}

	float timerValue(void) {
		return sdkGetTimerValue(&timer);
	}
	
	uint32 millis(void) {
		return (uint32)(1000 * sdkGetTimerValue(&timer));
	}

	uint32 micros(void) {
		return (uint32)(1000000 * sdkGetTimerValue(&timer));
	}
};

#endif /* __CU_UTILS_H__ */
