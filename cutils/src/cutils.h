/*
 * cutils.h
 *
 *  Created on: 2017/01/22
 *      Author: sin
 */

#ifndef CUTILS_H_
#define CUTILS_H_

/* --------------------- */
#define INTEL_SSE42

#ifdef INTEL_SSE42
#define pop32 pop32_SSE42
#define nlz32 nlz32_IEEE
#define ntz32 ntz32_
#else
#define pop32 pop32_
#define nlz32 nlz32_IEEE
#define ntz32 ntz32_
#endif
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
uint32 ceil2pow32(uint32 x);

uint32 floorlog32(uint32 x);
uint32 ceillog32(uint32 x);

/* difference or zero */
#define DOZ(x, y) (((x)-(y))&-((x)>=(y)))
#define MIN(x, y) ((x) - DOZ(x,y))
#define MAX(x, y) ((y) + DOZ(x,y))

int32 doz32(int32 x, int32 y);
int32 min32(int32 x, int32 y);
int32 max32(int32 x, int32 y);

/* swap values in two variables */
#define SWAP(x, y) { x ^= y; y ^= x; x ^= y; }

void swap32(int32 * x, int32 * y);

#endif /* CUTILS_H_ */
