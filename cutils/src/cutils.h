/*
 * cutils.h
 *
 *  Created on: 2017/01/22
 *      Author: sin
 */

#ifndef CUTILS_H_
#define CUTILS_H_

#include <stdint.h>

typedef uint32_t 	uint32;
typedef int32_t 	int32;
typedef uint64_t 	uint64;
typedef int64_t		int64;

/* count leading zero */
uint32 clz0(uint32 x);
uint32 clz(uint32 x);

/* smallest 2 to the nth power that is no less than x */
uint32 ceil2pow(uint32 x);

/* popularity (1 bits) count */
int popc0(uint32 bits);
int popc(uint32 bits);

/* count trailing zeros */
uint32 ctz(uint32 x);

#endif /* CUTILS_H_ */
