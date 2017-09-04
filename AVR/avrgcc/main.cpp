#include <stdio.h>
#include <avr/io.h>
#include <util/delay.h>

//#include "uartprint.h"
#include "Arduino.h"

#define bitset(r,b) (r |= (1<<(b)))
#define bitclear(r,b) (r &= ~(1<<(b)))

/*
typedef char boolean;
#define true  1
#define false 0
typedef unsigned char byte;
typedef unsigned int  word;
*/

struct IOPort {
  volatile byte & reg;
  byte bpos;

  IOPort(volatile byte & regname, byte pos) : reg(regname), bpos(1<<pos) {}

  void dwrite(boolean val) {
    if ( val ) {
      reg |= bpos;
    } else {
      reg &= ~bpos;
    }
  }
};

int main(int argc, char * argv[]) {

    init();
	// program body
	boolean up = true;
	long wt = 50;
	long full = 160;

  IOPort d13mode(DDRB, 5);
  IOPort d13value(PORTB, 5);
  // ports initialized
  //  bitset(DDRB, D13);
  d13mode.dwrite(true);

  //uart_init();
  //printf("Hello, world.");
  Serial.begin(9600);
  Serial.println("Hi friends!");
	_delay_ms(100);

loop:
	d13value.dwrite(true);
	_delay_us(64*wt);
	d13value.dwrite(false);
	_delay_us(64*(full-wt));
	if ( up ) wt++; else wt--;
	if ( up && wt >= full ) {
	  up = false;
	} else if ( !up && wt <= 0 ) {
	  up = true;
	}
	goto loop;
	
	return 0;
}

