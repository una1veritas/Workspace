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
typedef enum GPIOPin {
	PA0 = (PortA <<16) | PinBit(0),
	PA1 = (PortA <<16) | PinBit(1),
	PA2 = (PortA <<16) | PinBit(2),
	PA3 = (PortA <<16) | PinBit(3),

	PA8 = (PortA <<16) | PinBit(8),
	PA9 = (PortA <<16) | PinBit(9),
	PA10 = (PortA <<16) | PinBit(10),

	PB0 = (PortB <<16) | PinBit(0),
	PB6 = (PortB <<16) | PinBit(6),
	PB7 = (PortB <<16) | PinBit(7),
	PB8 = (PortB <<16) | PinBit(8),
	PB9 = (PortB <<16) | PinBit(9),

	PB10 = (PortB <<16) | PinBit(10),
	PB11 = (PortB <<16) | PinBit(11),

	PC0 = (PortC <<16) | PinBit(0),
	PC9 = (PortC <<16) | PinBit(9),
	PC10 = (PortC <<16) | PinBit(10),
	PC11 = (PortC <<16) | PinBit(11),
	PC12 = (PortC <<16) | PinBit(12),
	PC13 = (PortC <<16) | PinBit(13),

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
	PE1 = (PortE <<16) | PinBit(1),
	PE2 = (PortE <<16) | PinBit(2),
	PF0 = (PortF <<16) | PinBit(0),
	PG0 = (PortG <<16) | PinBit(0),
	PH0 = (PortH <<16) | PinBit(0),
	PI0 = (PortI <<16) | PinBit(0)
} GPIOPin_Type;

//#define digitalPinHasPWM(p)         ((p) == 4 || (p) == 5 || (p) == 6 || (p) == 7 || (p) == 9 || (p) == 10)

#define NUM_DIGITAL_PINS            32
//#define NUM_ANALOG_INPUTS           8
//#define analogInputToDigitalPin(p)  ((p < 8) ? (p) + 24 : -1)

#define OUTPUT		GPIO_Mode_OUT
#define INPUT		GPIO_Mode_IN
#define ALTFUNC		GPIO_Mode_AF
#define NOPULL		GPIO_PuPd_NOPULL
#define PULLUP		GPIO_PuPd_UP
#define PULLDOWN	GPIO_PuPd_DOWN
#define HIGH		SET
#define LOW			RESET

void pinMode(uint32_t portpin, GPIOMode_TypeDef mode);
void digitalWrite(uint32_t portpin, uint8_t bit);
void portWrite(GPIO_TypeDef * port, uint16_t bits);
uint8_t digitalRead(GPIO_TypeDef * port, uint16_t pin);

void GPIOMode(uint32_t portpins, GPIOMode_TypeDef mode,
		GPIOSpeed_TypeDef speed, GPIOOType_TypeDef otype, GPIOPuPd_TypeDef pull);
// void GPIOWrite(GPIO_TypeDef * port, uint16 value);

uint8_t pinsrc(uint32_t pin);

#ifdef __cplusplus
}
#endif

#endif /* _GPIO_DIGITAL_H_ */
