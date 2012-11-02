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
//#include "delay.h"
#include "systick.h"
#include "usart.h"

int main(void) {
	uint16_t count = 0;
	uint32_t dval = 436;
	uint32_t intval = 16;
	char tmp[92];
	
	SysTick_Start();

	usart_begin(USART3, 19200);
	usart_print(USART3, "Happy are those who know they are spiritually poor; \n");
	usart_print(USART3, "The kingdom of heaven belongs to them!\n");
//	sprintf(tmp, "port = %d\n", Serial3.port());
//	usart3_print(tmp);
	usart_print(USART3, "How many eyes does Mississipi river have?\n");
	usart_flush(USART3);

	pinMode(PD12 | PD13 | PD14 | PD15, GPIO_Mode_OUT);
		//GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	while (1) {
		digitalWrite(PD15, SET);
		digitalWrite(PD12 | PD13 | PD14, RESET);
		SysTick_delay(intval);
		digitalWrite(PD12, SET);
		digitalWrite(PD13 | PD14 | PD15, RESET);
		SysTick_delay(intval);
		digitalWrite(PD13, SET);
		digitalWrite(PD12 | PD14 | PD15, RESET);
		SysTick_delay(intval);
		digitalWrite(PD14, SET);
		digitalWrite(PD12 | PD13 | PD15, RESET);
		SysTick_delay(intval);
		//
		digitalWrite(PD15, SET);
		digitalWrite(PD12 | PD13 | PD14, RESET);
		SysTick_delay(dval);
/*
		usart3.print((float)(count++ / 32.0f), 3);
		*/
		count++;
//		uint16_t h, t;
//		h = tx_head();
//		t = tx_tail();
		sprintf(tmp, /*"head =% 4d, tail =% 4d,*/ "%04X\n", count);
		usart_print(USART3, tmp);
		/*
		dval = (uint32) (100.0f + 64*sinf( (count % (uint32)(3.14159 * 2 * 32))/32.0f));
		usart3.println(dval);
		*/
		uint16_t i;
		if ( usart_available(USART3) > 0 ) {
			usart_write(USART3, usart_peek(USART3));
			i = 0;
			while ( usart_available(USART3) > 0 ) {
				tmp[i++] = (char) usart_read(USART3);
			}
			tmp[i] = 0;
			usart_print(USART3, ": read: ");
			usart_print(USART3, tmp);
			usart_print(USART3, "\n");
		}

	}
	return 0;
}


