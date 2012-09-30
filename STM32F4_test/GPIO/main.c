/**
 *     GPIO/main.c
 */
/*
 #include <stdio.h>
 #include <stdint.h>
 #include <stddef.h>
 */
// USE_STDPERIPH_DRIVER is defined in both C and C++ at makefile(?)
#include "stm32f4xx.h"

#include "gpio_digital.h"
#include "stm32f4xx_it.h"
#include "delay.h"

int main(void) {

	uint8_t c = 0;

	pinMode(PC13, OUTPUT);
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);

	SysTick_init(1000);

	while (1) {
		switch ((++c) % 3) {
		case 0:
			digitalWrite(PC13, LOW);
			digitalWrite(PD12, HIGH);
			digitalWrite(PD13, HIGH);
			break;
		case 1:
			digitalWrite(PC13, HIGH);
			digitalWrite(PD12, LOW);
			digitalWrite(PD13, HIGH);
			break;
		default:
			digitalWrite(PC13, HIGH);
			digitalWrite(PD12, HIGH);
			digitalWrite(PD13, LOW);
			break;
		}
		SysTick_delay(500);
	}

	return 0;
}

