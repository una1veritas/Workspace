/*
 * gpio_constants.h
 *
 *  Created on: 2012/09/30
 *      Author: sin
 */

#ifndef _GPIO_CONSTANTS_H_
#define _GPIO_CONSTANTS_H_

#ifdef __cplusplus
extern "C" {
#endif

//#include <stdio.h>
//#include <stdint.h>
//#include <stddef.h>

#include "stm32f4xx_gpio.h"
#include "stm32f4xx_rcc.h"
//#include "stm32f4xx.h"

#include "Arduino.h"
//#include "mytypes.h"

const uint32_t GPIOPeripheral[] = {
		0, // Arduino Headers
		RCC_AHB1Periph_GPIOA, RCC_AHB1Periph_GPIOB, RCC_AHB1Periph_GPIOC,
		RCC_AHB1Periph_GPIOD, RCC_AHB1Periph_GPIOE, RCC_AHB1Periph_GPIOF,
		RCC_AHB1Periph_GPIOG, };

const GPIO_TypeDef * GPIOPort[] =
		{ 0, GPIOA, GPIOB, GPIOC, GPIOD, GPIOE, GPIOF, GPIOG, };

const uint16_t GPIOPin[] = { GPIO_Pin_0, GPIO_Pin_1, GPIO_Pin_2, GPIO_Pin_3,
		GPIO_Pin_4, GPIO_Pin_5, GPIO_Pin_6, GPIO_Pin_7, GPIO_Pin_8, GPIO_Pin_9,
		GPIO_Pin_10, GPIO_Pin_11, GPIO_Pin_12, GPIO_Pin_13, GPIO_Pin_14,
		GPIO_Pin_15, GPIO_Pin_All };


#define digitalPinToAHB1Port(p) 	(GPIOPeripheral[ (p) >> 5 & 0x07])
#define digitalPinToGPIOPort(p) 	(GPIOPort[ (p) >> 5 & 0x07])
#define digitalPinToGPIOPin(p) 		(GPIOPin[ (p) & 0x1f])

//#define digitalPinHasPWM(p)         ((p) == 4 || (p) == 5 || (p) == 6 || (p) == 7 || (p) == 9 || (p) == 10)

#define NUM_DIGITAL_PINS            32
//#define NUM_ANALOG_INPUTS           8
//#define analogInputToDigitalPin(p)  ((p < 8) ? (p) + 24 : -1)

/*
 static const uint8_t SDA = 16; // PC1
 static const uint8_t SCL = 17; // PC0

 // Map SPI port
 static const uint8_t SS   = 10; // PB4
 static const uint8_t MOSI = 11;
 static const uint8_t MISO = 12;
 static const uint8_t SCK  = 13; // PB7
 static const uint8_t LED_BUILTIN = 14;
 */

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _GPIO_DIGITAL_H_ */
