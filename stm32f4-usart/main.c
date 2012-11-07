/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stm32f4xx.h>

//#include "stm32f4xx_it.h"

#include "gpio.h"
#include "systick.h"
#include "usart.h"

int main(void) {
	uint16_t count = 0;
	uint32_t dval = 900;
	uint32_t intval = 24;
	char tmp[92];
	
	SysTick_Start();

	usart_begin(USART2Serial, PA3, PA2, 19200);
	usart_print(USART2Serial, "Happy are those who know they are spiritually poor; \n");
	usart_print(USART2Serial, "The kingdom of heaven belongs to them!\n");
//	sprintf(tmp, "port = %d\n", Serial3.port());
//	usart3_print(tmp);
	usart_print(USART2Serial, "How many eyes does Mississipi river have?\n");
	usart_print(USART2Serial, "Quick brown fox jumped over the lazy dog!\n");
	usart_flush(USART2Serial);

	GPIOMode(GPIOD, (GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15),
			GPIO_Mode_OUT, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);
	/*
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);
	pinMode(PD14, OUTPUT);
	pinMode(PD15, OUTPUT);
	| GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15),
			GPIO_Mode_OUT, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);
*/
	while (1) {
		digitalWrite(PD12, SET);
		digitalWrite(PD13, RESET);
		digitalWrite(PD14, RESET);
		digitalWrite(PD15, RESET);
		SysTick_delay(intval);
		digitalWrite(PD12, RESET);
		digitalWrite(PD13, SET);
		digitalWrite(PD14, RESET);
		digitalWrite(PD15, RESET);
		SysTick_delay(intval);
		digitalWrite(PD12, RESET);
		digitalWrite(PD13, RESET);
		digitalWrite(PD14, SET);
		digitalWrite(PD15, RESET);
		SysTick_delay(intval);
		digitalWrite(PD12, RESET);
		digitalWrite(PD13, RESET);
		digitalWrite(PD14, RESET);
		digitalWrite(PD15, SET);
		SysTick_delay(intval);
		//
		digitalWrite(PD12, SET);
		digitalWrite(PD13, RESET);
		digitalWrite(PD14, RESET);
		digitalWrite(PD15, RESET);
		SysTick_delay(dval);
/*
		usart3.print((float)(count++ / 32.0f), 3);
		*/
		count++;
//		uint16_t h, t;
//		h = tx_head();
//		t = tx_tail();
		sprintf(tmp, /*"head =% 4d, tail =% 4d,*/ "%04X\n", SysTick_count());
		usart_print(USART2Serial, tmp);
		/*
		dval = (uint32) (100.0f + 64*sinf( (count % (uint32)(3.14159 * 2 * 32))/32.0f));
		usart3.println(dval);
		*/
		uint16_t i = 0;
		if ( usart_available(USART2Serial) > 0 ) {
			while ( usart_available(USART2Serial) > 0 && i < 92 ) {
				tmp[i++] = (char) usart_read(USART2Serial);
			}
			tmp[i] = 0;
			usart_print(USART2Serial, "> ");
			usart_print(USART2Serial, tmp);
			usart_print(USART2Serial, "\n");
		}
	}
	return 0;
}


