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


enum PortNameDef {
	NotAPort = 0,
	PortA = 1,
	PortB,
	PortC,
	PortD,
	PortE,
	PortF,
	PortG,
	PortH,
	PortI
};

#define PinBit(n)  (1<<(n))
enum GPIOPin {
	PA0 = (PortA <<16) | PinBit(0),
	PB0 = (PortB <<16) | PinBit(0),
	PB10 = (PortB <<16) | PinBit(10),
	PB11 = (PortB <<16) | PinBit(11),
	PC0 = (PortC <<16) | PinBit(0),
	PD0 = (PortD <<16) | PinBit(0),
	PD1 = (PortD <<16) | PinBit(1),
	PD2 = (PortD <<16) | PinBit(2),
	PD3 = (PortD <<16) | PinBit(3),
	PD4 = (PortD <<16) | PinBit(4),
	PD5 = (PortD <<16) | PinBit(5),
	PD6 = (PortD <<16) | PinBit(6),
	PD7 = (PortD <<16) | PinBit(7),
	PD8 = (PortD <<16) | PinBit(8),
	PD9 = (PortD <<16) | PinBit(9),
	PD10 = (PortD <<16) | PinBit(10),
	PD11 = (PortD <<16) | PinBit(11),
	PD12 = (PortD <<16) | PinBit(12),
	PD13 = (PortD <<16) | PinBit(13),
	PD14 = (PortD <<16) | PinBit(14),
	PD15 = (PortD <<16) | PinBit(15),
	PE0 = (PortE <<16) | PinBit(0),
	PF0 = (PortF <<16) | PinBit(0),
	PG0 = (PortG <<16) | PinBit(0),
	PH0 = (PortH <<16) | PinBit(0),
	PI0 = (PortI <<16) | PinBit(0)
};

//#define digitalPinHasPWM(p)         ((p) == 4 || (p) == 5 || (p) == 6 || (p) == 7 || (p) == 9 || (p) == 10)

#define NUM_DIGITAL_PINS            32
//#define NUM_ANALOG_INPUTS           8
//#define analogInputToDigitalPin(p)  ((p < 8) ? (p) + 24 : -1)

void pinMode(uint32_t portpin, GPIOMode_TypeDef mode);
void digitalWrite(uint32_t portpin, uint8_t bit);
void portWrite(GPIO_TypeDef * port, uint16_t bits);
uint8_t digitalRead(GPIO_TypeDef * port, uint16_t pin);

void GPIOMode(uint32_t portpins, GPIOMode_TypeDef mode,
		GPIOSpeed_TypeDef speed, GPIOOType_TypeDef otype, GPIOPuPd_TypeDef pull);
// void GPIOWrite(GPIO_TypeDef * port, uint16 value);

#endif /* _GPIO_DIGITAL_H_ */
