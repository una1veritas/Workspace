/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <math.h>

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>

#include "armcore/portpin.h"
//#include "usart.h"
#include "armcore/delay.h"

int main(void) {
/*
	begin(& usart3, 38400);
	print(&usart3, "Hi!\n\n");
*/
	pinMode(PD12 | PD13 | PD14 | PD15, GPIO_Mode_OUT);
		//GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	uint16_t count = 0;
	uint32_t dval = 900;
	uint32_t intval = 24;

	while (1) {
		digitalWrite(PD15, SET);
		digitalWrite(PD12 | PD13 | PD14, RESET);
		_delay_ms(intval);
		digitalWrite(PD12, SET);
		digitalWrite(PD13 | PD14 | PD15, RESET);
		_delay_ms(intval);
		digitalWrite(PD13, SET);
		digitalWrite(PD12 | PD14 | PD15, RESET);
		_delay_ms(intval);
		digitalWrite(PD14, SET);
		digitalWrite(PD12 | PD13 | PD15, RESET);
		_delay_ms(intval);
		//
		digitalWrite(PD15, SET);
		digitalWrite(PD12 | PD13 | PD14, RESET);
		_delay_ms(450);
/*
		usart3.print((float)(count++ / 32.0f), 3);
		usart3.print(", ");
		dval = (uint32) (100.0f + 64*sinf( (count % (uint32)(3.14159 * 2 * 32))/32.0f));
		usart3.println(dval);
		if ( usart3.available() > 0 ) {
			while ( usart3.available() > 0 ) {
				usart3.print((char) usart3.read());
				usart3.print(' ');
			}
			usart3.println();
		}
*/
	}
	return 0;
}

