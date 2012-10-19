/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

//#include <stdlib.h>

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>

#include "armduino.h"
#include "usart.h"

int main(void) {

	USARTSerial usart3;
	usart3.begin(19200);
	usart3.println("Hi.");
	usart3.println();

	pinMode(PD12 | PD13 | PD14 | PD15, GPIO_Mode_OUT);
		//GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	uint16_t count = 0;

	while (1) {
		digitalWrite(PD12, SET);
		digitalWrite(PD13 | PD14 | PD15, RESET);
		_delay_ms(50);
		digitalWrite(PD13, SET);
		digitalWrite(PD12 | PD14 | PD15, RESET);
		_delay_ms(50);
		digitalWrite(PD14, SET);
		digitalWrite(PD12 | PD13 | PD15, RESET);
		_delay_ms(50);
		digitalWrite(PD15, SET);
		digitalWrite(PD12 | PD13 | PD14, RESET);
		_delay_ms(50);
		usart3.println((uint32_t) count++ / 100.0f, 3);
	}

	return 0;
}

