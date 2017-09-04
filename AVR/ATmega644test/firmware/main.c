/* Name: main.c
 * Author: <insert your name here>
 * Copyright: <insert your copyright message here>
 * License: <insert your license reference here>
 */

#include <avr/io.h>
#include <util/delay.h>

#define bitSet(port, bit)		((port) |= 0x01 << (bit))
#define Set(port)				((port) |= 0xff)
#define bitClear(port, bit)		((port) &= ~(0x01<< (bit)))
#define Clear(port)				((port) &= 0x00)
#define bitTest(port, bit)		((port) & (0x01 << (bit)))

int main(void) {
//	char sw = 1, sw_pushed = 0, sw_released = 0;
	char cnt = 1;
	
//	DDRB = 0b00000111;
	Clear(DDRB);
	bitSet(DDRB,0);
	bitSet(DDRB,1);
	bitSet(DDRB,2);
	
	bitSet(PORTD,5); // pull-up PORTD5
	/* insert your hardware initialization here */
//	PORTB = 0;
	Clear(PORTB);
		
	while (1) {
		/* insert your main loop code here */
		cnt++;
		cnt %= 8;
		PORTB &= 0b11111000;
		PORTB |= (0b00000111 & cnt);
		_delay_ms(500);
	}
	return 0;   /* never reached */
}
