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
#include "usart.h"
#include "delay.h"
#include "systick.h"

int main(void) {

	SysTick_Config(SystemCoreClock/1000);

	usart_begin(&usart3, USART3, 19200);
	usart_print(&usart3, "Hi!\n\n");

	pinMode(PD12 | PD13 | PD14 | PD15, GPIO_Mode_OUT);
		//GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	uint16_t count = 0;
	uint32_t dval = 400;
	uint32_t intval = 20;
	char tmp[32];
	uint16_t i;

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
		sprintf(tmp, "%X,\n", count);
		usart_print(&usart3, tmp);
		/*
		dval = (uint32) (100.0f + 64*sinf( (count % (uint32)(3.14159 * 2 * 32))/32.0f));
		usart3.println(dval);
		*/
		if ( usart_available(&usart3) > 0 ) {
			i = 0;
			while ( usart_available(&usart3) > 0 ) {
				tmp[i++] = (char) usart_read(&usart3);
			}
			tmp[i] = 0;
			usart_print(&usart3, tmp);
			usart_print(&usart3, "\n");
		}

	}
	return 0;
}


