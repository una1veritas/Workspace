#pragma once
#ifndef __CU_UTILS_H__
#define __CU_UTILS_H__

/* */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cu {
#ifndef min
#define min(x, y)   ((x) > (y)? (y) : (x))
#endif
#ifndef max
#define max(x, y)   ((x) < (y)? (y) : (x))
#endif
#ifndef abs
#define abs(x)  ((x) < 0? (-(x)) : (x))
#endif

#define ceildiv(n,d)		( ((n)+(d)-1)/(d) )
#define align(val, base)    ( ceildiv(val,base)*(base))

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

	__host__ __device__
	long pow2(const long val);
};
#endif /* __CU_UTILS_H__ */
