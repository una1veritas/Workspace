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
#define COMPILER_GCC
#define HARDWARE_IEEE_FP

/* --------------------- */

#include <stdint.h>

typedef uint32_t 	uint32;
typedef int32_t 	int32;
typedef uint64_t 	uint64;
typedef int64_t		int64;

/* popularity (1 bits) count */
uint32 pop32_SSE42(uint32 bits); /* int pop32_SSE42(uint32 bits); */
uint32 pop32_HD(uint32 bits);

#define POP32(x) pop32_SSE42(x)

/* count leading zero */
uint32 nlz32_IEEEFP(uint32 x); /* uint32 nlz32_IEEE(uint32 x); */
uint32 nlz32_ABM(uint32 x);
uint32 nlz32_Harley(uint32 x);

#define NLZ32(x) nlz32_ABM(x)

/* count trailing zeros */
uint32 ntz32_pop32(uint32 x); /* uint32 ntz32_pop32(uint32 x); */
uint32 ntz32_nlz32(uint32 x);
uint32 ntz32_HD(uint32 x);
uint32 ntz32_BMI1(uint32 x);

#define NTZ32(x) ntz32_BMI1(x)

/* smallest 2 to the nth power that is no less than x */
uint32 c2pow32(uint32 x);

uint32 flog32(uint32 x);
uint32 clog32(uint32 x);

/* the number of digits including the single leading 0 at msb. */
uint32 bitsize32(int32 x);

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

/* ceiling of integer division */
#define CDIV(x, y) ( ((x) != 0)*(1 + (((x) - 1) / (y));) )

#endif /* CUTILS_H_ */
