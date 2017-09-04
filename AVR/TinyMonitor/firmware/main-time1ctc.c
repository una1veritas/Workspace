/* Name: main.c
 * Author: <insert your name here>
 * Copyright: <insert your copyright message here>
 * License: <insert your license reference here>
 */

#include <avr/interrupt.h>
#include <avr/io.h>
#include <util/delay.h>

typedef unsigned char byte;

byte x;

int main(void) {
	DDRB = (1<<PB0) | (1<<PB1) | (1<<PB2);

	TCCR1B |= (1 << WGM12); // Configure timer 1 for CTC mode 
	TIMSK |= (1 << OCIE1A); // Enable CTC interrupt 
	sei(); //  Enable global interrupts 
	
	OCR1A   = 15624; // Set CTC compare value to 1Hz at 1MHz AVR clock, with a prescaler of 64 
	TCCR1B |= ((1 << CS10) | (1 << CS11)); // Start timer at Fcpu/64 
	
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


ISR(TIMER1_COMPA_vect) { 
	PORTB = x++;
//	PORTB ^= (1 << 0); // Toggle the LED 
}
