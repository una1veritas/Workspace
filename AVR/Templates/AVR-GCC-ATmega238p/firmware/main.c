/* Name: main.c
 * Author: <insert your name here>
 * Copyright: <insert your copyright message here>
 * License: <insert your license reference here>
 */

#include <avr/io.h>
#include <util/delay.h>


#define FOSC F_CPU // Clock Speed
#define BAUD 9600
#define MYUBRR FOSC/16/BAUD-1

void USART_Init(unsigned int ubrr) {
	/*Set baud rate */
	UBRR0H = (unsigned char)(ubrr>>8);
	UBRR0L = (unsigned char)ubrr;
	/* Enable receiver and transmitter */
	UCSR0B = (1<<RXEN0)|(1<<TXEN0);
	/* Set frame format: 8data, 1stop bit */
	UCSR0C = (0<<USBS0)|(3<<UCSZ00);
}

void USART_Transmit( unsigned char data ) {
	/* Wait for empty transmit buffer */
	while ( !( UCSR0A & (1<<UDRE0)) ) ;
	/* Put data into buffer, sends the data */
	UDR0 = data;
}

void printuint(unsigned int i) {
	unsigned int dval;
	char c;
	for (dval = 1000; dval > 0; dval /= 10) {
		c = (i / dval) % 10;
		USART_Transmit('0'+c);
	}
	USART_Transmit(0x0d);
	USART_Transmit(0x0a);	
}

// Read the AD conversion result
#define ADC_VREF_TYPE 0x40;
unsigned int read_adc(unsigned char pin) {
	ADMUX = pin | ADC_VREF_TYPE; // AVcc
	// Start the AD conversion
	ADCSRA|=0x40;
	// Wait for the AD conversion to complete
	while ((ADCSRA & 0x10)==0);
	ADCSRA|=0x10;
	return ADCW;
}

int main(void) {
	char dir = 1;
	unsigned char dot = 0x04;
	unsigned int  waitval = 0;
	
	DDRD = 0b11111111;
	PORTD = dot;
	DDRB = 0; // the default values on reset
	PORTB = 0xff; // set internal pull-ups
	
	USART_Init(MYUBRR);
	
	//adc
	ADMUX=ADC_VREF_TYPE;
	ADCSRA=0x85;
	
    for(;;){
		waitval = read_adc(0);
		if ( (dot & 0x84) != 0 )
			waitval <<= 1;
		_delay_ms(waitval);
		if ( (PINB & 1<<PORTB4) == 0 ) {
			printuint(read_adc(0));
			continue;
		}

		if ( dir > 0 ) {
			dot <<= 1;
			if ( dot == 0x80 ) {
				dir = -dir;
			}
		} else {
			dot >>= 1;
			if ( dot == 0x04 ) {
				dir = -dir;
			}
		}
		PORTD = dot;
    }
    return 0;   /* never reached */
}
