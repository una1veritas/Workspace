/*
 * ringbuffer.h
 *
 *  Created on: 2012/10/23
 *      Author: sin
 */

#ifndef RINGBUFFER_H_
#define RINGBUFFER_H_

#include <stdint.h>

#define RINGBUFFER_SIZE 	128
typedef struct {
	uint16_t buffer[RINGBUFFER_SIZE];
	uint16_t head, tail;
	uint16_t count;
} Ring;

void init(Ring * r);
uint16_t ringin(Ring * r, uint16_t c);
uint16_t ringout(Ring * r);
uint16_t peek(Ring * r);
uint8_t is_full(Ring * r);
uint8_t is_empty(Ring * r);
uint16_t remains(Ring * r);


#endif /* RINGBUFFER_H_ */
