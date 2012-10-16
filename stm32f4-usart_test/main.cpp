/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include "delay.h"
#include "gpio.h"

int main(void) {

	GPIOMode(RCC_AHB1Periph_GPIOD, GPIOD,
			GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15,
			GPIO_Mode_OUT, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	while (1) {
		digitalWrite(GPIOD, GPIO_Pin_12, SET);
		digitalWrite(GPIOD, GPIO_Pin_13, RESET);
		digitalWrite(GPIOD, GPIO_Pin_14, RESET);
		digitalWrite(GPIOD, GPIO_Pin_15, RESET);
		_delay_ms(500);
		digitalWrite(GPIOD, GPIO_Pin_12, RESET);
		digitalWrite(GPIOD, GPIO_Pin_13, SET);
		digitalWrite(GPIOD, GPIO_Pin_14, RESET);
		digitalWrite(GPIOD, GPIO_Pin_15, RESET);
		_delay_ms(500);
		digitalWrite(GPIOD, GPIO_Pin_12, RESET);
		digitalWrite(GPIOD, GPIO_Pin_13, RESET);
		digitalWrite(GPIOD, GPIO_Pin_14, SET);
		digitalWrite(GPIOD, GPIO_Pin_15, RESET);
		_delay_ms(500);
		digitalWrite(GPIOD, GPIO_Pin_12, RESET);
		digitalWrite(GPIOD, GPIO_Pin_13, RESET);
		digitalWrite(GPIOD, GPIO_Pin_14, RESET);
		digitalWrite(GPIOD, GPIO_Pin_15, SET);
		_delay_ms(500);
	}

	return 0;
}

