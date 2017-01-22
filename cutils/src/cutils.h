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
uint32 clz(uint32 x);

/* smallest 2 to the nth power that is no less than x */
uint32 ceil2pow(uint32 x);

int popc_s(uint32 bits);

int popc_h(uint32 bits);

#endif /* CUTILS_H_ */
