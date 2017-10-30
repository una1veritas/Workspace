/*------------------------------------------------*/
/* UART functions                                 */


#include <avr/io.h>
#include <avr/interrupt.h>
#include "uart.h"

#define	SYSCLK		20000000UL
#define	BAUD		38400


/* Initialize UART */

void uart_init()
{

	/* Enable USART0 (N81) */
	UBRR0L = SYSCLK/BAUD/16-1;
	UCSR0B = _BV(RXEN0) |_BV(TXEN0);
}


/* Get a received character */

uint8_t uart_test ()
{

	if( (UCSR0A & _BV(RXC0)) ) return 0xFF;
	return 0;

}


uint8_t uart_get ()
{
	return UDR0;
}


/* Put a character to transmit */

void uart_put (uint8_t d)
{

	while ( !(UCSR0A & _BV(UDRE0)) );
	UDR0 = d;

}


