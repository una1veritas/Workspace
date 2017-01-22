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

int popc0(uint32 bits)
{
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >>16 & 0x0000ffff);
}

int popc(uint32 bits)
{
    int ret;
    __asm__ ("popcntl %[input], %[output]" : [output] "=r"(ret) : [input] "r"(bits));
    return ret;
}

uint32 clz(uint32 x) {
	double d = x;
	uint32 *p = ((uint32*) &d) + 1;
	if ( x == 0 )
		return 32;
	return 0x41e - (*p>>20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}

uint32 clz0(uint32 x) {
	int count;
	uint32_t bit = 0x80000000;
	for(count = 0; bit != 0; ++count, bit >>= 1) {
		if ( (x & bit) != 0 )
			break;
	}
	return count;
}

uint32 ceil2pow(uint32 x) {
	if (x == 0)
		return 0;
	return 1<<(32 - clz(x-1));

}

uint32 ctz(uint32 x) {
	int32 y = (int32) x;
	if ( (x & 0x800000) != 0 )
		return 32;
	return popc((y & (-y))-1);
}
