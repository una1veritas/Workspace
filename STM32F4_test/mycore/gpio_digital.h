/*
 * gpio_digital.h
 *
 *  Created on: 2012/09/30
 *      Author: sin
 */

#ifndef _GPIO_DIGITAL_H_
#define _GPIO_DIGITAL_H_

#ifdef __cplusplus
extern "C" {
#endif

//#include <stdio.h>
//#include <stdint.h>
//#include <stddef.h>

//#include "stm32f4xx_gpio.h"
//#include "stm32f4xx_rcc.h"
//#include "stm32f4xx.h"

#include "Arduino.h"
//#include "mytypes.h"

enum DigitalPin {
	PA0 = 1 << 5,
	PB0 = 2 << 5,
	PC0 = 3 << 5,
	PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15,
	PD0 = 4 << 5,
	PD1,
	PD2,
	PD3,
	PD4,
	PD5,
	PD6,
	PD7,
	PD8,
	PD9,
	PD10,
	PD11,
	PD12,
	PD13,
	PD14,
	PD15,
	PE0 = 5 << 5,
	PF0 = 6 << 5,
	PG0 = 7 << 5,
};

void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t val);
int digitalRead(uint8_t pin);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _GPIO_DIGITAL_H_ */
