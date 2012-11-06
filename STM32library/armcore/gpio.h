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

typedef enum _GPIOPin {
	PA0 = (PortA <<8) | ((uint16_t)0),
	PA1 = (PortA <<8) | ((uint16_t)1),
	PA2 = (PortA <<8) | ((uint16_t)2),
	PA3 = (PortA <<8) | ((uint16_t)3),

	PA8 = (PortA <<8) | ((uint16_t)8),
	PA9 = (PortA <<8) | ((uint16_t)9),
	PA10 = (PortA <<8) | ((uint16_t)10),

	PB0 = (PortB <<8) | ((uint16_t)0),
	PB6 = (PortB <<8) | ((uint16_t)6),
	PB7 = (PortB <<8) | ((uint16_t)7),
	PB8 = (PortB <<8) | ((uint16_t)8),
	PB9 = (PortB <<8) | ((uint16_t)9),

	PB10 = (PortB <<8) | ((uint16_t)10),
	PB11 = (PortB <<8) | ((uint16_t)11),

	PC0 = (PortC <<8) | ((uint16_t)0),
	PC9 = (PortC <<8) | ((uint16_t)9),
	PC10 = (PortC <<8) | ((uint16_t)10),
	PC11 = (PortC <<8) | ((uint16_t)11),
	PC12 = (PortC <<8) | ((uint16_t)12),
	PC13 = (PortC <<8) | ((uint16_t)13),

	PD0 = (PortD <<8) | ((uint16_t)0),
	PD1 = (PortD <<8) | ((uint16_t)1),
	PD2 = (PortD <<8) | ((uint16_t)2),
	PD3 = (PortD <<8) | ((uint16_t)3),
	PD4 = (PortD <<8) | ((uint16_t)4),
	PD5 = (PortD <<8) | ((uint16_t)5),
	PD6 = (PortD <<8) | ((uint16_t)6),
	PD7 = (PortD <<8) | ((uint16_t)7),
	PD8 = (PortD <<8) | ((uint16_t)8),
	PD9 = (PortD <<8) | ((uint16_t)9),
	PD10 = (PortD <<8) | ((uint16_t)10),
	PD11 = (PortD <<8) | ((uint16_t)11),
	PD12 = (PortD <<8) | ((uint16_t)12),
	PD13 = (PortD <<8) | ((uint16_t)13),
	PD14 = (PortD <<8) | ((uint16_t)14),
	PD15 = (PortD <<8) | ((uint16_t)15),

	PE0 = (PortE <<8) | ((uint16_t)0),
	PE1 = (PortE <<8) | ((uint16_t)1),
	PE2 = (PortE <<8) | ((uint16_t)2),
	PF0 = (PortF <<8) | ((uint16_t)0),
	PG0 = (PortG <<8) | ((uint16_t)0),
	PH0 = (PortH <<8) | ((uint16_t)0),
	PI0 = (PortI <<8) | ((uint16_t)0)
} GPIOPin;

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

void pinMode(GPIOPin portpin, GPIOMode_TypeDef mode);
void digitalWrite(GPIOPin portpin, uint8_t bit);
uint8_t digitalRead(GPIOPin portpin);

GPIO_TypeDef * pinPort(GPIOPin portpin);
uint16_t pinBit(GPIOPin portpin);
uint8_t pinSource(GPIOPin portpin);

void GPIOWrite(GPIO_TypeDef * port, uint16_t pinbits);
void GPIOMode(GPIO_TypeDef * port, uint16_t pinbits, GPIOMode_TypeDef mode,
              GPIOSpeed_TypeDef clk, GPIOOType_TypeDef otype, GPIOPuPd_TypeDef pupd);
// void GPIOWrite(GPIO_TypeDef * port, uint16 value);

#ifdef __cplusplus
}
#endif

#endif /* _GPIO_DIGITAL_H_ */
