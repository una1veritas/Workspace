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

/* popularity (1 bits) count */
uint32_t pop32_SSE42(uint32_t bits); /* int pop32_SSE42(uint32_t bits); */
uint32_t pop32_HD(uint32_t bits);

#define POP32(x) pop32_SSE42(x)

/* count leading zero */
uint32_t nlz32_IEEEFP(uint32_t x); /* uint32_t nlz32_IEEE(uint32_tx); */
uint32_t nlz32_ABM(uint32_t x);
uint32_t nlz32_Harley(uint32_t x);

int nlz64_lzcnt_u64(uint64_t x);

#define NLZ32(x) nlz32_ABM(x)

/* count trailing zeros */
uint32_t ntz32_pop32(uint32_t x); /* uint32_t ntz32_pop32(uint32_t x); */
uint32_t ntz32_nlz32(uint32_t x);
uint32_t ntz32_HD(uint32_t x);
uint32_t ntz32_BMI1(uint32_t x);

#define NTZ32(x) ntz32_BMI1(x)

/* smallest 2 to the nth power that is no less than x */
uint32_t c2pow32(uint32_t x);

uint32_t flog32(uint32_t x);
uint32_t clog32(uint32_t x);

/* the number of digits including the single leading 0 at msb. */
uint32_t bitsize32(int32_t x);

/* difference or zero */
#define DOZ(x, y) (((x)-(y))&-((x)>=(y)))
#define MIN(x, y) ((x) - DOZ(x,y))
#define MAX(x, y) ((y) + DOZ(x,y))

int32_t doz32(int32_t x, int32_t y);
int32_t min32(int32_t x, int32_t y);
int32_t max32(int32_t x, int32_t y);

/* swap values in two variables */
#define SWAP(x, y) { x ^= y; y ^= x; x ^= y; }
void swap32_(int32_t * x, int32_t * y);

/* ceiling of integer division */
#define CDIV(x, y) ( ((x) != 0)*(1 + (((x) - 1) / (y));) )

#endif /* CUTILS_H_ */
