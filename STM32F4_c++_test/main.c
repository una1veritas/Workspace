/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>
#include <math.h>

#include "favorites.h"
#include "gpio_digital.h"
#include "systick.h"
#include "usart.h"


//#define MAX_STRLEN 12 // this is the maximum string length of our string in characters
//volatile char received_string[MAX_STRLEN + 1]; // this will hold the recieved string

// this is the interrupt request handler (IRQ) for ALL USART1 interrupts
/*
void USART3_IRQHandler(void) {

	// check if the USART1 receive interrupt flag was set
	if (USART_GetITStatus(serial.port(), USART_IT_RXNE )) {

		static uint8_t cnt = 0; // this counter is used to determine the string length
		char t = serial.read(); //USART3 ->DR; // the character from the USART1 data register is saved in t

		if ((t != '\n') && (cnt < MAX_STRLEN)) {
			received_string[cnt] = t;
			cnt++;
		} else { // otherwise reset the character counter and print the received string
			cnt = 0;
			serial.puts(received_string);
		}
	}
}
*/

void wink(Pin pin) {
	digitalWrite(pin, HIGH);
	delay(800);
	digitalWrite(pin, LOW);
	delay(200);
}

int main(void) {

	SysTick_Start();

	// setup();
	pinMode(PD12, OUTPUT, FASTSPEED, NOPULL);
	pinMode(PD13, OUTPUT, FASTSPEED, NOPULL);
	pinMode(PD14, OUTPUT, FASTSPEED, NOPULL);
	pinMode(PD15, OUTPUT, FASTSPEED, NOPULL);

//	SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));

	wink(PD12);
	usart_begin(19200);
	wink(PD12);

	usart_print("Hello World!\n"); // just send a message to indicate that it works
//	serial.print(19200);

	uint32 ticks = 0;

	for (;;) {
		uint32 count = millis();
		digitalWrite(PD15, HIGH);
		delay(count);
		digitalWrite(PD15, LOW);
		delay(1000-count);
		ticks++;

//		usart_print("passed ");
//		usart_printFloat(millis()/1000.0f, 2);
//		usart_print(".\n");
	}

	return 0;
}

