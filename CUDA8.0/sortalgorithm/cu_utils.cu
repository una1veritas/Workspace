#include "cu_utils.h"

/*
#define NLZ32DEV(x) __clz(x)
#define NLZ64DEV(x) __clzll(x)
#define POPC32DEV(x) __popc(x)
#define POPC64DEV(x) __popcll(x)
*/

__device__ uint32 c2pow32dev(uint32 x) {
	return (x == 0) ? 0 : (1 << (32 - __clz(x - 1)));
}

__device__ uint32 clog32dev(uint32 x) {
	return (x == 0) ? 0 : (32 - __clz(x - 1));
}

__device__ uint32 bitsize32dev(int32 x) {
	x = x ^ (x >> 31);
	return 33 - __clz(x);
}

unsigned int nlz32_IEEEFP(unsigned int x)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., 5.3, p. 104 -- */
	double d = (double)x + 0.5;
	unsigned int *p = ((unsigned int*)&d) + 1;
	return 0x41e - (*p >> 20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}

uint32 c2pow32(uint32 x) {
	return (x != 0) * (1 << (32 - NLZ32(x - 1)));

}

uint32 f2pow32(uint32 x) {
	return (x != 0) * (1 << (31 - NLZ32(x)));

}

uint32 flog32(uint32 x) {
	return 31 - NLZ32(x);

}

uint32 clog32(uint32 x) {
	return 32 - NLZ32(x - 1);
}

uint32 bitsize32(int32 x) {
	x = x ^ (x >> 31);
	return 33 - NLZ32(x);
}
