#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "wiring.h"

void init() {
	// Timer0 normal count
	TCCR0A |= 0b10<<COM0A0;
	TCCR0A |= 0b10<<COM0B0;
	TCCR0A |= 0b11<<WGM00;
	TCCR0B |= 0b001<<CS00;
	bitSet(TIMSK, TOIE0);
	OCR0A = 0;
	OCR0B = 0;
	bitSet(DDRD, PD5);
	bitSet(DDRB, PB2);
	
	// Timer1 in CTC mode w/ Output Comapre Match Interrupt
	//TCCR1A = 0:
	TCCR1B |= 0b01 << WGM12;
	OCR1AH = highByte(1875);
	OCR1AL = lowByte(1875);
	TCCR1B |= 0b100 << CS10;
	bitSet(TIMSK, OCIE1A);
	
}

volatile char t1count = 0;
volatile char seconds = 0;

ISR(TIMER0_OVF_vect) {
	OCR0B = abs(t1count-12)*20;
	OCR0A = abs((seconds%20) - 10)*10;
}


ISR(TIMER1_COMPA_vect) {
	t1count++;
	if (t1count > 25 - 1) {
		t1count = 0;
		seconds++;
	}
}

int main(void) {
	int prevsec = 0;
	
	init();

	sei();
    for(;;){
		if (prevsec != seconds) {
			seconds %= 60;
			prevsec = seconds;
		}
	}
    return 0;               /* never reached */
}
