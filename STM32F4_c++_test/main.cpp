/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include "favorites.h"
#include "gpio_digital.h"
#include "delay.h"

int main(void) {

	// init();
	// setup();
	pinMode(PD12, OUTPUT);

	uint32 ticks = 0;

	for(;;) {
		ticks++;
		if ( ticks & 1) {
			digitalWrite(PD12, HIGH);
		} else {
			digitalWrite(PD12, LOW);
		}
		_delay_ms(1000);
	}

	return 0;
}

