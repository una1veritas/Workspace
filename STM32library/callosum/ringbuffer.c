/*
 * ringbuffer.c
 *
 *  Created on: 2012/10/23
 *      Author: sin
 */

#include "ringbuffer.h"

	void init(Ring * r) {
		r->head = 0;
		r->tail = 0;
		r->count = 0;
	}

	uint16_t ringin(Ring * r, uint16_t c) {
		if ( is_full(r) ) {
			r->head++;  r->head %= RINGBUFFER_SIZE;
			// buffer over run occurred!
		} else
			r->count++;
		r->buffer[r->tail++] = c;
		r->tail %= RINGBUFFER_SIZE;
		return c;
	}

	uint16_t ringout(Ring * r) {
		if ( is_empty(r) )
			return 0;
		uint16_t c = r->buffer[r->head++];
		r->head %= RINGBUFFER_SIZE;
		r->count--;
		return c;
	}

	uint16_t peek(Ring * r) {
		if ( is_empty(r) )
			return 0;
		return r->buffer[r->head];
	}

	uint8_t is_full(Ring * r) {
		return (r->count > 0) && r->head == r->tail;
	}

	uint8_t is_empty(Ring * r) {
		return (r->count == 0) && r->head == r->tail;
	}

	uint16_t remains(Ring * r) {
		return r->count;
	}



