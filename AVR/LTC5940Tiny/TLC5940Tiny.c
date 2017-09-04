#include <avr/io.h>
#include <avr/interrupt.h>

#include "wiring.h"

#define PB5 5
#define PB3 3



ISR(TIMER1_COMPA_vect) {
	bitSet(PORTB,PB3);
	asm("nop");
	bitClear(PORTB,PB3);
}

int main() {
	//init
	DDRD = 0x7f; // there is no pd7
	DDRB = 0xff;

	TCCR0A |= 0b01<<COM0A0;
	bitSet(TCCR0A,WGM01);
	TCCR0B = 0b001<<CS00;
	OCR0A = 4;

//	TCCR1A = (TCCR1A & ~(0b11<<COM1A0)) | (0b01 << COM1A0);
	TCCR1B |= (1<<WGM12);
	TCCR1B |= (0b010<<CS10);
	
	OCR1AH = highByte(4096);
	OCR1AL = lowByte(4096);

	bitSet(TIMSK,OCIE1A);

	sei();
	for (;;) {
		asm("nop");
	}
}
