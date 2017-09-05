#include <stdio.h>
#include <avr/io.h>
#include <util/delay.h>

#include "Arduino.h"
#include "HardwareSerial.h"

#define bitset(r,b) (r |= (1<<(b)))
#define bitclear(r,b) (r &= ~(1<<(b)))

int main(int argc, char * argv[]) {

  init();

	bitset(DDRB, 5);
	boolean up = true;
	long wt = 50;
	long full = 160;

	_delay_ms(100);

  Serial.begin(9600);
  Serial.println("Hi friends!");
  Serial.println(analogRead(0));

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

