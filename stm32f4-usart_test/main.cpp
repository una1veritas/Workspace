/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>

#include "armduino.h"
#include "usart.h"

int main(void) {

	usart_begin(19200);
	usart_print("Hi.\n");

	pinMode(PD12 | PD13 | PD14 | PD15, GPIO_Mode_OUT);
		//GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	uint16_t count = 0;

	while (1) {
		digitalWrite(PD12, SET);
		digitalWrite(PD13 | PD14 | PD15, RESET);
		_delay_ms(125);
		digitalWrite(PD13, SET);
		digitalWrite(PD12 | PD14 | PD15, RESET);
		_delay_ms(125);
		digitalWrite(PD12, RESET);
		digitalWrite(PD13, RESET);
		digitalWrite(PD14, SET);
		digitalWrite(PD15, RESET);
		_delay_ms(125);
		digitalWrite(PD12, RESET);
		digitalWrite(PD13, RESET);
		digitalWrite(PD14, RESET);
		digitalWrite(PD15, SET);
		_delay_ms(125);
		usart_printNumber(count++);
		usart_print("\n");
	}

	return 0;
}

