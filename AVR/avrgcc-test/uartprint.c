#include <avr/io.h>
#include <stdio.h>
#include "uartprint.h"

static FILE mystdout = FDEV_SETUP_STREAM(uart_putchar, NULL, _FDEV_SETUP_WRITE);

void uart_init (void)
{
    //1 = output, 0 = input
	DDRD &= 0b11111110; //PORTD (RX on PD0)
    DDRD |= 0b10;

    //USART Baud rate: 9600
    UBRR0H = (FOSC/16/BAUD-1) >> 8;
    UBRR0L = (FOSC/16/BAUD-1);
    UCSR0B = (1<<RXEN0)|(1<<TXEN0);
    
    stdout = &mystdout; //Required for printf init
}

int uart_putchar(char c, FILE *stream)
{
    if (c == '\n') uart_putchar('\r', stream);
  
    loop_until_bit_is_set(UCSR0A, UDRE0);
    UDR0 = c;
    
    return 0;
}

uint8_t uart_getchar(void)
{
    while( !(UCSR0A & (1<<RXC0)) );
    return(UDR0);
}
