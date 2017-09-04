/*
 *  serial.c
 *  megaclock
 *
 *  Created by ‰º‰’ ^ˆê on 10/05/30.
 *  Copyright 2010 ‹ãBH‹Æ‘åŠwî•ñHŠw•”. All rights reserved.
 *
 */

#include <avr/io.h>
#include "wiring.h"

#include "serial.h"

void serialBegin(long baudrate) {
	unsigned int ubrr = F_CPU/16/baudrate-1;
	/*Set baud rate */
	UBRR0H = highByte(ubrr);
	UBRR0L = lowByte(ubrr);
	/* Enable receiver and transmitter */
	//	UCSR0B = (1<<RXEN0)|(1<<TXEN0);
	UCSR0B = (0<<RXEN0)|(1<<TXEN0);
	/* Set frame format: 8data, 2stop bit */
	UCSR0C = (0<<USBS0)|(3<<UCSZ00);
}

void serialWrite(unsigned char data ){
	/* Wait for empty transmit buffer */
	while ( !( UCSR0A & (1<<UDRE0)) ) ;
	/* Put data into buffer, sends the data */
	UDR0 = data;
}

void serialPrint(char * msg) {
	for (;*msg != '\0';msg++) {
		serialWrite((unsigned char) *msg);
	}
}
