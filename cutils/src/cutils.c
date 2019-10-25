/*
 ============================================================================
 Name        : cutils.c
 Author      : Sin Shimozono
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#include "cutils.h"


	inline static uint64 & rotl64(uint64 & x) {
		return x = (x<<1) | (x>>(-1&63));
	}

	inline static uint64 & rotl64(uint64_t & x, const unsigned int & n) {
		return x = (x<<n) | (x>>(-n & 63));
	}

	inline static uint64 & rotr64(uint64 & x) {
		return x = (x>>1) | (x<<(-1&63));
	}

	inline static uint64 & rotr64(uint64_t & x, const unsigned int & n) {
		return x = (x>>n) | (x<<(-n & 63));
	}


uint32 pop32_SSE42(uint32 xu32)
{
    int ret;
    __asm__ volatile ("popcntl %1, %0" : "=r"(ret) : "r"(xu32) );
    return ret;
}

uint32 pop32_HD(uint32 bits)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., p. 81 -- */
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);
}


uint32 nlz32_IEEEFP(uint32 x)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., 5.3, p. 104 -- */
	double d = (double)x + 0.5;
	uint32 *p = ((uint32*) &d) + 1;
	return 0x41e - (*p>>20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}

uint32 nlz32_ABM(uint32 x)
{
    int ret;
    __asm__ volatile ("lzcnt %1, %0" : "=r" (ret) : "r" (x) );
    return ret;
}

uint32 nlz32_Harley(uint32 x) {
	static char table[64] = {
			32, 31, -1, 16, -1, 30,  3, -1,  15, -1, -1, -1, 29, 10, 2, -1,
			-1, -1, 12, 14, 21, -1, 19, -1,  -1, 28, -1, 25, -1,  9, 1, -1,
			17, -1,  4, -1, -1, -1, 11, -1,  13, 22, 20, -1, 26, -1, -1, 18,
			 5, -1, -1, 23, -1, 27, -1,  6,  -1, 24,  7, -1,  8, -1,  0, -1,
	};
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	x = x*0x06EB14F9;
	return table[x>>26];
}


uint32 ntz32_pop32(uint32 x) {
	return 32 - POP32( x | -x);
}

uint32 ntz32_nlz32(uint32 x) {
	return 32 - NLZ32((~x) & (x-1));
}


uint32 ntz32_BMI1(uint32 x) {
    int ret;
    __asm__ volatile ("tzcnt %1, %0" : "=r" (ret) : "r" (x));
    return ret;
}

uint32 ntz32_HD(uint32 x) {
	/* Hacker's Delight 2nd by H. S. Warren Jr. */
	uint32 y, bz, b4, b3, b2, b1, b0;
	y = x & -x;
	bz = y ? 0 : 1;
	b4 = (y & 0x0000ffff) ? 0 : 16;
	b3 = (y & 0x00ff00ff) ? 0 : 8;
	b2 = (y & 0x0f0f0f0f) ? 0 : 4;
	b1 = (y & 0x33333333) ? 0 : 2;
	b0 = (y & 0x55555555) ? 0 : 1;
	return bz + b4 + b3 + b2 + b1 + b0;
}


uint32 c2pow32(uint32 x) {
	return (x != 0) * ( 1<<(32 - NLZ32(x-1)) );

}

uint32 f2pow32(uint32 x) {
	return (x != 0) * ( 1<<(31 - NLZ32(x)) );

}

uint32 flog32(uint32 x) {
	return 31 - NLZ32(x);

}

uint32 clog32(uint32 x) {
	return 32 - NLZ32(x - 1);
}

uint32 bitsize32(int32 x) {
	x = x^(x>>31);
	return 33 - NLZ32(x);
}

int32 doz32(int32 x, int32 y) {
	return (x-y)&-(x>=y);
}

int32 min32(int32 x, int32 y) {
	return ((x^y)&-(x<=y))^y;
}

int32 max32(int32 x, int32 y) {
	return ((x^y)&-(x>=y))^y;
}

void swap32_(int32 * x, int32 * y) {
	int32 t = *x;
	*x = *y;
	*y = t;
}
