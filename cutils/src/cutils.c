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
#include <time.h>

#include "cutils.h"

int pop32_SSE42(uint32 bits)
{
    int ret;
    __asm__ ("popcntl %[input], %[output]" : [output] "=r"(ret) : [input] "r"(bits));
    return ret;
}

int pop32_(uint32 bits)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., p. 81 -- */
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);
}


uint32 nlz32_IEEE(uint32 x)
{
	/* Hacker's Delight 2nd by H. S. Warren Jr., 5.3, p. 104 -- */
	double d = x;
	d += 0.5;
	uint32 *p = ((uint32*) &d) + 1;
	return 0x41e - (*p>>20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}

uint32 nlz32_(uint32 x) {
	int count;
	uint32_t bit = 0x80000000;
	for(count = 0; bit != 0; ++count, bit >>= 1) {
		if ( (x & bit) != 0 )
			break;
	}
	return count;
}

uint32 ntz32_(uint32 x) {
	int32 y = (int32) x;
	if ( (x & 0x800000) != 0 )
		return 32;
	return pop32((y & (-y))-1);
}

uint32 c2pow32(uint32 x) {
	return (x != 0) * ( 1<<(32 - nlz32(x-1)) );

}

uint32 flog32(uint32 x) {
	return 31 - nlz32(x);

}

uint32 clog32(uint32 x) {
	return 32 - nlz32(x - 1);
}

uint32 bitsize32(int32 x) {
	x = x^(x>>31);
	return 33 - nlz32(x);
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
