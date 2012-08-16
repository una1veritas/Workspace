#include <stdio.h>
#include <avr/io.h>
#include <util/delay.h>

#include "uartprint.h"

#define bitset(r,b) (r |= (1<<(b)))
#define bitclear(r,b) (r &= ~(1<<(b)))

typedef char boolean;
#define true  1
#define false 0
typedef unsigned char byte;
typedef unsigned int  word;

int main(int argc, char * argv[]) {

	bitset(DDRB, 5);
	boolean up = true;
	long wt = 50;
	long full = 160;

	uart_init();
	printf("Hello, world.");
	_delay_ms(100);

loop:
	bitset(PORTB, 5);
	_delay_us(100*wt);
	bitclear(PORTB, 5);
	_delay_us(100*(full-wt));
	if ( up ) wt++; else wt--;
	if ( up && wt >= full ) {
		up = false;
	} else if ( !up && wt <= 0 ) {
		up = true;
	}
	goto loop;
	
	return 0;
}

