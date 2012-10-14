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


USARTSerial serial(3);

void wink(Pin pin) {
	digitalWrite(pin, HIGH);
	delay(800);
	digitalWrite(pin, LOW);
	delay(200);
}

int main(void) {

//	SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));  /* set CP10 and CP11 Full Access */

	SysTick_Start();
	// setup();
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);
	pinMode(PD14, OUTPUT);
	pinMode(PD15, OUTPUT);

	wink(PD12);
	serial.begin(19200);
	wink(PD12);

	serial.print("Aha?\n");
	serial.println("Hello, at last!");
	serial.print(19200);
	serial.println();
	serial.print((uint32)19200, HEX);
	serial.println();
	serial.print(-19200);
	serial.println();
	serial.print(3.141592);
	serial.println();
//	serial.print("Hello World!\n"); // just send a message to indicate that it works
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
			serial.print(millis());
			serial.println();
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

