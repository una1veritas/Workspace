#include <avr/io.h>

#include <avr/interrupt.h>

volatile unsigned char sec = 0;

ISR (TIMER0_OVF_vect) { 
	sec++;
//	PORTD = ~PORTD;
}

int main() {

	//init
	DDRD=0xff;

	TIMSK0 |= (1<<TOIE0);
	TCNT0 = 0;
	TCCR0B = (1<<CS01) | (1<<CS00);

	sei();

	while(1) {
		if ( (sec & 0b10) != 0 )
			PORTD = 1;
		else 
			PORTD = 0;
	}
}
