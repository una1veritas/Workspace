/*
 * armduino.h
 *
 *  Created on: 2012/10/17
 *      Author: sin
 */

#ifndef ARMDUINO_H_
#define ARMDUINO_H_

#include "stm32f4xx.h"
#include "stm32f4xx_gpio.h"
#include "stm32f4xx_rcc.h"

#include "delay.h"
#include "portpin.h"

// declarations for C++

#ifdef __cplusplus
extern "C" {
#endif

void pinMode(uint32_t pin, GPIOMode_TypeDef mode);
void digitalWrite(uint32_t portpin, uint8_t bit);
void portWrite(GPIO_TypeDef * port, uint16_t bits);
uint8_t digitalRead(GPIO_TypeDef * port, uint16_t pin);

void GPIOMode(uint32_t periph, GPIO_TypeDef * port, uint16_t pins, GPIOMode_TypeDef mode,
		GPIOSpeed_TypeDef speed, GPIOOType_TypeDef otype, GPIOPuPd_TypeDef pull);
// void GPIOWrite(GPIO_TypeDef * port, uint16 value);

#ifdef __cplusplus
}
#endif


#endif /* ARMDUINO_H_ */
