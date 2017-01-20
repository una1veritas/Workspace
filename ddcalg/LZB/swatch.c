/*
 * swatch.c
 *
 *  Created on: 2017/01/20
 *      Author: sin
 */

#include "swatch.h"

clock_t swatch[NUM_SWATCHES], laps[NUM_SWATCHES];

void sw_reset(int counter) {
	laps[counter] = clock();
}

void sw_clear(int counter) {
	swatch[counter] = 0;
}

void sw_clear_all(void) {
	for(int i = 0; i < NUM_SWATCHES; i++)
		swatch[i] = 0;
}

void sw_accumlate(int counter) {
	swatch[counter] += clock() - laps[counter];
}

clock_t sw_value(int counter) {
	return swatch[counter];
}
