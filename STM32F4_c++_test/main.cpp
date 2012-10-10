/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>
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

	for (;;) {
		ticks++;
		digitalWrite(PD12, LOW);
		digitalWrite(PD13, LOW);
		digitalWrite(PD14, LOW);
		digitalWrite(PD15, LOW);
		switch (ticks % 16) {
		case 0:
		case 4:
		case 11:
		case 15:
			digitalWrite(PD12, HIGH);
			break;
		case 1:
		case 5:
		case 10:
		case 14:
			digitalWrite(PD13, HIGH);
			break;
		case 2:
		case 6:
		case 9:
		case 13:
			digitalWrite(PD14, HIGH);
			break;
		case 3:
		case 7:
		case 8:
		case 12:
			digitalWrite(PD15, HIGH);
			break;
		}
		delay(100);
	}

	return 0;
}

