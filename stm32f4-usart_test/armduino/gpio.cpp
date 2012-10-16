/*
 * gpio_digital.cpp
 *
 *  Created on: 2012/09/30
 *      Author: sin
 */

#include "gpio.h"


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

void GPIOMode(uint32_t periph, GPIO_TypeDef * port, uint16_t pins,
		GPIOMode_TypeDef mode,
		GPIOSpeed_TypeDef clk,
		GPIOOType_TypeDef otype,
		GPIOPuPd_TypeDef pupd) {

	GPIO_InitTypeDef GPIO_InitStructure;

	// enum GPIOMode_TypeDef =
	//	  GPIO_Mode_IN   = 0x00, /*!< GPIO Input Mode */
	//	  GPIO_Mode_OUT  = 0x01, /*!< GPIO Output Mode */
	//	  GPIO_Mode_AF   = 0x02, /*!< GPIO Alternate function Mode */
	//	  GPIO_Mode_AN   = 0x03  /*!< GPIO Analog Mode */

	//typedef enum {
	//  GPIO_OType_PP = 0x00, // push-pull
	//  GPIO_OType_OD = 0x01 // open drain ?
	// }GPIOOType_TypeDef;

	//typedef enum {
	//  GPIO_PuPd_NOPULL = 0x00,
	//  GPIO_PuPd_UP     = 0x01,
	//  GPIO_PuPd_DOWN   = 0x02
	// }GPIOPuPd_TypeDef;

		// wake up the port
		RCC_AHB1PeriphClockCmd(periph, ENABLE);
		//
		GPIO_InitStructure.GPIO_Pin   = pins;
		GPIO_InitStructure.GPIO_Mode  = mode;
		GPIO_InitStructure.GPIO_OType = otype;
		GPIO_InitStructure.GPIO_PuPd  = pupd;
		GPIO_InitStructure.GPIO_Speed = clk;
		//
		GPIO_Init(port, &GPIO_InitStructure);
}

/*
 static void turnOffPWM(uint8_t timer) {
 }
 */
void digitalWrite(GPIO_TypeDef * port, uint16_t pins, uint8_t val) {
	if ( val ) {
		//? Bit_SET : Bit_RESET ));
		GPIO_SetBits(port, pins);
	} else {
		GPIO_ResetBits(port, pins);
	}
}

void portWrite(GPIO_TypeDef * port, uint16_t bits) {
	GPIO_Write(port, bits);
}

uint8_t digitalRead(GPIO_TypeDef * port, uint16_t pin) {
	uint8_t mode = (port->MODER) >> (pin * 2);
	if ( mode == GPIO_Mode_OUT )
		return (GPIO_ReadOutputDataBit(port, pin) ?
				SET : RESET );
	return (GPIO_ReadInputDataBit(port, pin) ?
			SET : RESET );
}

/*
 void togglePin(uint8 pin) {
 if (pin >= BOARD_NR_GPIO_PINS) {
 return;
 }

 GPIO_ToggleBit(PIN_MAP[pin].gpio_device, PIN_MAP[pin].gpio_bit);
 }
 */

/*
 void ConfigPin(GPIO_TypeDef *myGPIO, uint32_t PIN, uint32_t MODE, uint32_t SPEED, uint32_t PUPD) {

//         myGPIO: The GPIOx port for the selected pin
//         MODE: 0 = INPUT .... 1 = OUTPUT .... 2 = ALTERNATE FUNCTION .... 3 = ANALOG
//         SPEED: 0 = 2MHz (Low Speed) .... 1 = 25MHz (Med. Speed) .... 2 = 50MHz (Fast Speed) .... 3 = 100MHz/80MHz (High Speed)(100MHz(30pf) - 80MHz(15pf))
//         PUPD: 0 = No Pull-Up / No Pull-Down .... 1 = Pull-Up Enabled .... 2 = Pull-Down Enabled .... 3 = Reserved

myGPIO->MODER |= (MODE << (PIN * 2));//OUTPUT
myGPIO->OSPEEDR |= (SPEED << (PIN * 2));//50MHz
myGPIO->PUPDR |= (PUPD << (PIN * 2)); //Set it for NO PUPD
}
 */
