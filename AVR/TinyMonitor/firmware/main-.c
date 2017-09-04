/* Name: main.c
 * Author: <insert your name here>
 * Copyright: <insert your copyright message here>
 * License: <insert your license reference here>
 */

#include <avr/interrupt.h>
#include <avr/io.h>
#include <util/delay.h>

#define bitSet(port, bit)		((port) |= 0x01 << (bit))
#define Set(port)				((port) |= 0xff)
#define bitClear(port, bit)		((port) &= ~(0x01<< (bit)))
#define Clear(port)				((port) &= 0x00)
#define bitTest(port, bit)		((port) & (0x01 << (bit)))

typedef unsigned char byte;

byte counter;

int main(void) {
	DDRB = (1<<PB0) | (1<<PB1) | (1<<PB2) | (1<<PB3);
	DDRD = 0x00;
	PORTD |= (1<<PD3);  // weak pull-up

	TCCR1B |= (1 << WGM12); // Configure timer 1 for CTC mode 
	TIMSK |= (1 << OCIE1A); // Enable CTC interrupt 
	
//	OCR1A   = 15624; // Set CTC compare value to 1Hz at 1MHz AVR clock, with a prescaler of 64 
	OCR1A	= 3906;
//	TCCR1B |= ((1 << CS10) | (1 << CS11)); // Start timer at Fcpu/64 
	TCCR1B |= (1 << CS12) | (0 << CS11) | (0 << CS10);  
	
	MCUCR  |= (1<<ISC11);
	GIMSK  |= 1<<INT1;
	
	sei(); //  Enable global interrupts 
	
	for (;;) {
/*
		if (TIFR & (1 << OCF1A)) { 
			PORTB ^= (1 << 0); // Toggle the LED 
			TIFR = (1 << OCF1A); // clear the CTC flag (writing a logic one to the set flag clears it) 
		} 
 */
	}
	/* never reached */
	return 0;
}

ISR(INT1_vect) { 
	PORTB ^= (1<<3);
}

ISR(TIMER1_COMPA_vect) { 
	PORTB &= ~0x07;
	PORTB |= 0x07 & counter++;
//	PORTB ^= (1 << 0); // Toggle the LED 
}
