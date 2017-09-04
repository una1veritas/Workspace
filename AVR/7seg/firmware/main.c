#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>


#define PORTBMode(bits)	(DDRB = ((unsigned char) (bits)))
#define PORTDMode(bits)	(DDRD = ((unsigned char) (bits)))

#define DIGIT4	(PD6+8)
#define DIGIT3	PB0
#define DIGIT2	PB4
#define DIGIT1	PB6
#define COLON	PB2

const unsigned char DIGIT[] = {
	DIGIT1, DIGIT2, DIGIT3, DIGIT4, COLON
};

#define SEGA	(PD3+8)
#define SEGB	(PD5+8)
#define SEGC	(PD2+8)
#define	SEGD	PB1
#define SEGE	PB3
#define SEGF	PB7
//(PD1+8)
#define SEGG	(PD4+8)
#define SEGP	PB5

const unsigned char SEGMENT[] = {
	SEGA,
	SEGB,
	SEGC,
	SEGD,
	SEGE,
	SEGF,
	SEGG,
	SEGP
};

#define SetBit(port, bit)		((port) |= 0x01 << (bit))
#define Set(port, onebyte)		((port) |= (onebyte))
#define ClearBit(port, bit)		((port) &= ~(0x01<< (bit)))
#define Clear(port, onebyte)	((port) &= ~(onebyte))
#define TestBit(port, bit)		((port) & (0x01 << (bit)))

#define abs(x)	((x)>0?(x):-(x))

void SetBits(unsigned int b) {
	Set(PORTB, b & 0xff);
	Set(PORTD, (b>>8) & 0xff);
//	Clear(PORTB, b & 0xff);
//	Clear(PORTD, (b>>8) & 0xff);
}

void ClearBits(unsigned int b) {
	Clear(PORTB, b & 0xff);
	Clear(PORTD, (b>>8) & 0xff);
//	Set(PORTB, b & 0xff);
//	Set(PORTD, (b>>8) & 0xff);
}
#define DIGITALL  ( (1<<DIGIT1)|(1<<DIGIT2)|(1<<COLON)|(1<<DIGIT3)|(1<<DIGIT4) )

unsigned int  LEDState[5];
unsigned int  counter = 0;
void display(unsigned int val, unsigned char base, unsigned char dots);

int main(void) {
	
	SetBit(TCCR1B, WGM12); // Configure timer 1 for CTC mode 
	SetBit(TIMSK, OCIE1A); // Enable CTC interrupt 

	Set(TCCR1B, (0 << CS12) | (0 << CS11) | (1 << CS10) ); // timer at Fcpu/1
	OCR1A   = 15625-208; // Set CTC compare value 
	//
	
//	PORTBMode(0xff & ~(1<<PB7)); // Set as output except SCL
	PORTBMode(0xff);
	PORTDMode(0xff & ~((1<<PD0)|(1<<PD1))); // Set as output except RXD and TXD

	int val = 60*20;
	unsigned int prev = counter;
	sei(); //  Enable global interrupts 
	
	display( (val/60)*100+(val%60),10,0b00000011);

    for(;;){
		if ((counter>>9) != prev) {
			val--;
			prev = (counter>>9);
		//	cli(); // inhibit port change during setting values
			display( (abs(val)/60)*100+(abs(val)%60),10,0b00000011);
		//	sei();
		}
		
		//_delay_ms(100);  /* max is 262.14 ms / F_CPU in MHz */		
		//v--;
		//v -= watch/100;
    }
    return 0;               /* never reached */
}


ISR(TIMER1_COMPA_vect) {
	counter++;
	ClearBits(DIGITALL);
	ClearBits(0xffff);
	SetBits(LEDState[counter%5]);
	SetBits( 0x01<<DIGIT[counter%5] );
}

void display(unsigned int val, unsigned char base, unsigned char dots) {
	unsigned int i, bits;
	unsigned char patt, j;
	
	for (j = 0; j < 5; j++) {
		if (j < 4) {
			switch ( val % base ) {
				case 0:
					patt = 0b00111111;
					break;
				case 1:
					patt = 0b00000110;
					break;
				case 2:
					patt = 0b01011011;
					break;
				case 3:
					patt = 0b01001111;
					break;
				case 4:
					patt = 0b01100110;
					break;
				case 5:
					patt = 0b01101101;
					break;
				case 6:
					patt = 0b01111101;
					break;
				case 7:
					patt = 0b00000111;
					break;
				case 8:
					patt = 0b01111111;
					break;
				case 9:
					patt = 0b01101111;
					break;
				case 0xa:
					patt = 0b01110111;
					break;
				case 0xb:
					patt = 0b01111100;
					break;
				case 0xc:
					patt = 0b01011000;
					break;
				case 0xd:
					patt = 0b01011110;
					break;
				case 0xe:
					patt = 0b01111001;
					break;
				case 0xf:
					patt = 0b01110001;
					break;
				default:
					patt = 0;
					break;
			}
			patt |= ((dots>>j)&0b00010000)<<3 ;
		} else {
			patt = dots & 0x0f;
		}
		bits = 0;
		for (i = 0; i < 8; i++) {
			if ((patt>>i) & 0x01)
				continue;
			SetBit(bits, SEGMENT[i]);
		}
		LEDState[j] = bits;
		val /= base;
	}
}
