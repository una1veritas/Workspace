#pragma once
#ifndef _CUDAUTILS_H_
#define _CUDAUTILS_H_

#define align(base, val)    ((((val)+(base)-1)/(base))*(base))
#define ceildiv(n,d)		( (n)%(d) > 0 ? (n)/(d)+1 : (n)/(d) )

#define CUCHECK(call)		\
{						\
	const cudaError_t cuerror = call;							\
	if ( cuerror != cudaSuccess )								\
	{															\
		printf("Error: %s: %d, ", __FILE__, __LINE__);			\
		printf("code %d, %s\n", cuerror, cudaGetErrorString(cuerror));	\
		exit(1);												\
	}															\
}


#endif
