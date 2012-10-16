/*
 * gpio_digital.h
 *
 *  Created on: 2012/09/30
 *      Author: sin
 */

#ifndef _GPIO_DIGITAL_H_
#define _GPIO_DIGITAL_H_

#include "stm32f4xx.h"
#include "stm32f4xx_gpio.h"
#include "stm32f4xx_rcc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	PA0 = (1 << 16) | (1<<0),
	PB0 = (2 << 16) | (1<<0),
	PC0 = (3 << 16) | (1<<0),
	PC1 = (3 << 16) | (1<<1),
	PD0 = (4<<16) | (1<<0),
	PD1 = (4<<16) | (1<<1),
	PD2 = (4<<16) | (1<<2),
	PD3 = (4<<16) | (1<<3),
	PD4 = (4<<16) | (1<<4),
	PD5 = (4<<16) | (1<<5),
	PD6 = (4<<16) | (1<<6),
	PD7 = (4<<16) | (1<<7),
	PD8 = (4<<16) | (1<<8),
	PD9 = (4<<16) | (1<<9),
	PD10 = (4<<16) | (1<<10),
	PD11 = (4<<16) | (1<<11),
	PD12 = (4<<16) | (1<<12),
	PD13 = (4<<16) | (1<<13),
	PD14 = (4<<16) | (1<<14),
	PD15 = (4<<16) | (1<<15),
	PE0 = (5<<16) | (1 <<0),
	PF0 = (6 <<16) | (1<<0),
	PG0 = (7<<16) | (1<<0),
} PortPin;

//void pinMode(uint8_t pin,	GPIOMode_TypeDef mode,GPIOSpeed_TypeDef clk, GPIOPuPd_TypeDef pupd);
void digitalWrite(GPIO_TypeDef * port, uint16_t pins, uint8_t val);
void portWrite(GPIO_TypeDef * port, uint16_t bits);
uint8_t digitalRead(GPIO_TypeDef * port, uint16_t pin);

void GPIOMode(uint32_t periph, GPIO_TypeDef * port, uint16_t pins, GPIOMode_TypeDef mode,
		GPIOSpeed_TypeDef speed, GPIOOType_TypeDef otype, GPIOPuPd_TypeDef pull);
// void GPIOWrite(GPIO_TypeDef * port, uint16 value);

#ifdef __cplusplus
}
#endif

#endif /* _GPIO_DIGITAL_H_ */
