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


#include "favorites.h"

#ifndef USE_STDPERIPH_DRIVER
#define USE_STDPERIPH_DRIVER 1
#endif

#include "stm32f4xx_gpio.h"
#include "stm32f4xx_rcc.h"
//#include "stm32f4xx.h"

typedef enum {
	PA0 = 1 << 5,
	PB0 = 2 << 5,
	PC0 = 3 << 5,
	PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, PC11, PC12, PC13, PC14, PC15,
	PD0 = 4 << 5,
	PD1, PD2, PD3, PD4, PD5, PD6, PD7, PD8, PD9, PD10, PD11, PD12, PD13, PD14, PD15,
	PE0 = 5 << 5,
	PF0 = 6 << 5,
	PG0 = 7 << 5, PG1, PG2, PG3, PG4, PG5, PG6, PG7, PG8, PG9, PG10, PG11, PG12, PG13, PG14, PG15,
} Pin;


void pinMode(uint8 pin, uint8 mode);
uint16 digitalRead(uint8 pin);
void digitalWrite(uint8 pin, uint8 val);

uint8 digitalPin(const GPIO_TypeDef * port, uint16 pin);

void GPIOMode(GPIO_TypeDef * port, uint16 pin, GPIOMode_TypeDef mode, GPIOPuPd_TypeDef pupd = GPIO_PuPd_NOPULL);
void GPIOWrite(GPIO_TypeDef * port, uint16 value);



#ifdef __cplusplus
} // extern "C"
#endif


#endif /* _GPIO_DIGITAL_H_ */
