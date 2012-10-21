/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <math.h>

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>

#include "armduino.h"
#include "USARTSerial.h"

int main(void) {

	usart3.begin(38400);
	usart3.println("Hi.");
	usart3.println();

	pinMode(PD12 | PD13 | PD14 | PD15, GPIO_Mode_OUT);
		//GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	uint16_t count = 0;
	uint32_t dval = 50;

	while (1) {
		digitalWrite(PD12, SET);
		digitalWrite(PD13 | PD14 | PD15, RESET);
		_delay_ms(dval);
		digitalWrite(PD13, SET);
		digitalWrite(PD12 | PD14 | PD15, RESET);
		_delay_ms(dval);
		digitalWrite(PD14, SET);
		digitalWrite(PD12 | PD13 | PD15, RESET);
		_delay_ms(dval);
		digitalWrite(PD15, SET);
		digitalWrite(PD12 | PD13 | PD14, RESET);
		_delay_ms(dval);

		usart3.print((float)(count++ / 32.0f), 3);
		usart3.print(", ");
		dval = (uint32) (92.0f + 36*sinf( (count % (uint32)(3.14159 * 2 * 32))/32.0f));
		usart3.println(dval);
		if ( usart3.available() > 0 ) {
			while ( usart3.available() > 0 ) {
				usart3.print((char) usart3.read());
				usart3.print(' ');
			}
			usart3.println();
		}

	}

	return 0;
}

