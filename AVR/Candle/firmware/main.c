/*
    Electronic Candle
*/

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#include "wdefs.h"

#define FOSC 8000000// Clock Speed

#define GREEN 1
#define BLUE 3	
#define RED 0

word counter;
byte PORTval;

struct {
	byte red, green, blue;
} color, led;


inline void lightControl(byte clk) {
	PORTval = 0;
	if ( clk < (led.red) ) {
		PORTval |= 1<<RED;
	}
	if ( clk < (led.green) ) {
		PORTval |= 1<<GREEN; //green on
	}
	if ( clk < (led.blue) ) {
		PORTval |= 1<<BLUE;
	}
	PORTB = PORTval;
}

ISR(TIM0_OVF_vect){ 
    //code
	TCNT0 = 5;
	counter++;
	lightControl(0x7f & counter);
}

void colorUpdate() {
	if (led.red < color.red) {
		led.red++;
	} else if (led.red > color.red) {
		led.red--;
	}
	if (led.green < color.green) {
		led.green++;
	} else if (led.green > color.green) {
		led.green--;
	}
	if (led.blue < color.blue) {
		led.blue++;
	} else if (led.blue > color.blue) {
		led.blue--;
	}
}

void setColor(byte r, byte g, byte b) {
	//	//bluish white
	//	maxLevel.red = 116;
	//	maxLevel.green = 128;
	//	maxLevel.blue = 102;
	color.red = r; // red with shallow yellow
	color.green = g; // yellowish green
	color.blue = b; // blue slightly purplish
}

// Read the AD conversion result
word read_adc(byte pin) {
	byte high, low;
	ADMUX &= 0xf0;
	ADMUX |= (0x03 & pin);
	
	// Start the AD conversion
	bitSet(ADCSRA, ADSC); //|=0x40;
	
	// Wait for the AD conversion to complete
	while ( bitRead(ADCSRA, ADIF) == 0 );
	bitSet(ADCSRA,ADIF); // clear ADIF
	
	low = ADCL;
	high = ADCH;
	return (high<<8) | low;
}

int main (void) {
	word cycle = 0;
	word d, base = 0;
	int yb, rg, avx, avy, red, green, blue;

	// init
	PORTB = 0;
	DDRB = (1<<GREEN)|(1<<BLUE)|(1<<RED);
	
	//setup timer0
	bitClear(TCCR0B, CS01);
	bitSet(TCCR0B, CS00);
	bitSet(TIMSK, TOIE0);

	//init adc
	ADCSRA |= bit(ADPS2) | bit(ADPS1) | bit(ADPS0);  // Set ADC prescalar to 128 - 125KHz sample rate @ 16MHz 
	bitClear(ADCSRA, ADIE); // deactivate ADC interrupt
	ADMUX &= ~(bit(REFS1) | bit(REFS0));	 /* vref = Vcc */
	bitClear(ADMUX,ADLAR); // right adjust
	
	ADCSRB &= ~(bit(ADTS2) | bit(ADTS1) | bit(ADTS0)); // free running mode
	bitSet(ADCSRA,ADEN);
	
	
	sei();
	/*
	for (d = 0; d < 10; d++) {
		avx = (avx+read_adc(1))/2;
		avy = (avy+read_adc(2))/2;
	} */
	setColor(0,0,0);
	counter = 0;
	
	d = 1023;
	while(1) {
		/*
		if (cycle % 235 == 0) {
			base = 55;
		} else if (cycle % 237 == 0) {
			base = 48;
		} else if ( cycle % 3041 == 0) {
			base = 13;
		}*/
		switch(cycle %2) {
			case 0:
				rg = read_adc(1);
				break;
			case 1:
				yb = read_adc(2);
				break;
		}
		red = min(127, (max(rg-512,0)>>1));
		green = min(127, max(512-rg,0)>>1);
		blue = min(127, max(yb - 512,0)>>1);
		red = min(127, red+5*(max(512-yb,0)>>1)/8);
		green = min(127, green+5*(max(512-yb,0)>>1)/4);
		setColor(red,green,blue);
		_delay_us(d);
		colorUpdate();
//		lightControl();
		cycle++;
	}
	
    
}

