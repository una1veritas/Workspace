/*
 * swatch.h
 *
 *  Created on: 2017/01/20
 *      Author: sin
 */

#ifndef SWATCH_H_
#define SWATCH_H_

#include <time.h>

enum {
	PRI = 0,
	INIT,
	ENCODE,
	SET_TREE,
	SEARCH,
	MOD_BUF,
	OUTPUT,
	ALLOC,
	NUM_SWATCHES,
};

void sw_reset(int counter);
void sw_clear(int counter);
void sw_accumlate(int counter);
void sw_clear_all();
clock_t sw_value(int counter);

#endif /* SWATCH_H_ */
