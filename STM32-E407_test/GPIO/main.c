/**
 *     GPIO/main.c
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

// USE_STDPERIPH_DRIVER is defined in both C and C++ at makefile(?)
#include "stm32f4xx.h"

#include "gpio_digital.h"
#include "stm32f4xx_it.h"
#include "delay.h"

int main(void) {

	uint8_t c = 0;

	pinMode(PD9, OUTPUT);
	pinMode(PD10, OUTPUT);
	pinMode(PD11, OUTPUT);
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);
	pinMode(PD14, OUTPUT);
	pinMode(PD15, OUTPUT);

	pinMode(PA0, INPUT);

	SysTick_init(1000);

	while (1) {
		if (digitalRead(PA0) == HIGH) {
			c++;
		} else {
			c += 4;
		}
		switch ((c) % 5) {
		case 0:
			digitalWrite(PD11, LOW);
			digitalWrite(PD12, HIGH);
			digitalWrite(PD13, HIGH);
			digitalWrite(PD14, HIGH);
			digitalWrite(PD15, HIGH);
			break;
		case 1:
			digitalWrite(PD11, HIGH);
			digitalWrite(PD12, LOW);
			digitalWrite(PD13, HIGH);
			digitalWrite(PD14, HIGH);
			digitalWrite(PD15, HIGH);
			break;
		case 2:
			digitalWrite(PD11, HIGH);
			digitalWrite(PD12, HIGH);
			digitalWrite(PD13, LOW);
			digitalWrite(PD14, HIGH);
			digitalWrite(PD15, HIGH);
			break;
		case 3:
			digitalWrite(PD11, HIGH);
			digitalWrite(PD12, HIGH);
			digitalWrite(PD13, HIGH);
			digitalWrite(PD14, LOW);
			digitalWrite(PD15, HIGH);
			break;
		case 4:
			digitalWrite(PD11, HIGH);
			digitalWrite(PD12, HIGH);
			digitalWrite(PD13, HIGH);
			digitalWrite(PD14, HIGH);
			digitalWrite(PD15, LOW);
			break;
		}
		digitalWrite(PD9, HIGH);
		digitalWrite(PD10, LOW);
		SysTick_delay(500);
		digitalWrite(PD9, LOW);
		digitalWrite(PD10, HIGH);
		SysTick_delay(500);
	}

	return 0;
}

