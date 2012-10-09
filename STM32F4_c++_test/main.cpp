/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include "favorites.h"
#include "gpio_digital.h"
#include "systick.h"
#include "init.h"

int main(void) {

	init();
	// setup();
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);
	pinMode(PD14, OUTPUT);
	pinMode(PD15, OUTPUT);

	uint32 ticks = 0;

	for(;;) {
		ticks++;
		digitalWrite(PD12, LOW);
		digitalWrite(PD13, LOW);
		digitalWrite(PD14, LOW);
		digitalWrite(PD15, LOW);
		if ( (ticks % 8 == 0) || (ticks % 8 == 7) ) {
			digitalWrite(PD12, HIGH);
		}
		if ( (ticks % 8 == 1) || (ticks % 8 == 6) ) {
			digitalWrite(PD13, HIGH);
		}
		if ( (ticks % 8 == 2) || (ticks % 8 == 5) ) {
			digitalWrite(PD14, HIGH);
		}
		if ( (ticks % 8 == 3) || (ticks % 8 == 4) ) {
			digitalWrite(PD15, HIGH);
		}
		delay(125);
	}

	return 0;
}

