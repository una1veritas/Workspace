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

#include "portpin.h"


//#define digitalPinHasPWM(p)         ((p) == 4 || (p) == 5 || (p) == 6 || (p) == 7 || (p) == 9 || (p) == 10)

#define NUM_DIGITAL_PINS            32
//#define NUM_ANALOG_INPUTS           8
//#define analogInputToDigitalPin(p)  ((p < 8) ? (p) + 24 : -1)

void pinMode(uint32_t portpin, GPIOMode_TypeDef mode);
void digitalWrite(uint32_t portpin, uint8_t bit);
void portWrite(GPIO_TypeDef * port, uint16_t bits);
uint8_t digitalRead(GPIO_TypeDef * port, uint16_t pin);

void GPIOMode(uint32_t periph, GPIO_TypeDef * port, uint16_t pins, GPIOMode_TypeDef mode,
		GPIOSpeed_TypeDef speed, GPIOOType_TypeDef otype, GPIOPuPd_TypeDef pull);
// void GPIOWrite(GPIO_TypeDef * port, uint16 value);

#endif /* _GPIO_DIGITAL_H_ */
