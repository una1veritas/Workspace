/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <stm32f4xx.h>
#include <stm32f4xx_conf.h>
#include "favorites.h"
#include "gpio_digital.h"
#include "systick.h"
#include "USARTSerial.h"


#define MAX_STRLEN 12 // this is the maximum string length of our string in characters
volatile char received_string[MAX_STRLEN + 1]; // this will hold the recieved string

USARTSerial serial(3);

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
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);
	pinMode(PD14, OUTPUT);
	pinMode(PD15, OUTPUT);

	wink(PD12);
	serial.begin(19200);
	wink(PD12);

	serial.print("Hello World!\n"); // just send a message to indicate that it works
//	serial.print(19200);

	uint32 ticks = 0;

	for (;;) {
		ticks++;
		digitalWrite(PD12, LOW);
		digitalWrite(PD13, LOW);
		digitalWrite(PD14, LOW);
		digitalWrite(PD15, LOW);
		switch (ticks % 16) {
		case 0:
		case 4:
		case 11:
		case 15:
			digitalWrite(PD12, HIGH);
			break;
		case 1:
		case 5:
		case 10:
		case 14:
			digitalWrite(PD13, HIGH);
			break;
		case 2:
		case 6:
		case 9:
		case 13:
			digitalWrite(PD14, HIGH);
			break;
		case 3:
		case 7:
		case 8:
		case 12:
			digitalWrite(PD15, HIGH);
			break;
		}
		delay(125);
	}

	return 0;
}

