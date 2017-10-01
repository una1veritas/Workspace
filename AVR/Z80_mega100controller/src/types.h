/*
 * types.h
 *
 *  Created on: 2017/09/30
 *      Author: sin
 */

#ifndef SRC_TYPES_H_
#define SRC_TYPES_H_

#include <stdio.h>
#include <stdlib.h>

#define DDR(port) 		(*((&port)-1))
#define PIN(port) 		(*((&port)-2))
#define BSET(port, bit)  	(port) |= (bit)
#define BCLR(port, bit)  	(port) &= ~(bit);
#define BTEST(port, bit) 	((port) & (bit))

#define INPUTMODE(port, bit) 	(DDR(port) &= ~bit)
#define OUTPUTMODE(port, bit)	(DDR(port) |= bit)


typedef uint8_t  uint8;
typedef uint8_t  BYTE;
typedef uint16_t uint16;
typedef uint16_t DWORD;
typedef uint32_t uint32;
typedef uint32_t QWORD;

typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;


#endif /* SRC_TYPES_H_ */
