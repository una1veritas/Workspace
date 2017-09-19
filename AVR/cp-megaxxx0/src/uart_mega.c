
#include <avr/io.h>
#include <avr/interrupt.h>

#include "uart.h"

#ifndef USART0_RX_vect
/* for atmega xx8 */
#define USART0_RX_vect USART_RX_vect
#endif

#define	UART_BUFF_SIZE		(1<<6)

typedef struct {
	uint16_t	wi, ri, ct;
	uint8_t buff[UART_BUFF_SIZE];
} FIFO;
static
volatile FIFO TxFifo, RxFifo;


void uart_init(unsigned long baud) {

	UCSR0B = 0;

	RxFifo.ct = 0; RxFifo.ri = 0; RxFifo.wi = 0;
	TxFifo.ct = 0; TxFifo.ri = 0; TxFifo.wi = 0;

    // Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet

    UBRR0 = (((F_CPU / (baud * 16UL))) - 1);         // Set the baud rate prescale rate register
    UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));  // Set frame format: 8data, 1 stop bit. See Table 22-7 for details

    UCSR0B = ((1<<RXEN0)|(1<<TXEN0) | (1<<RXCIE0));  // Enable receiver and transmitter

}

/*
unsigned char uart_tx(unsigned char data) {
    //while the transmit buffer is not empty loop
    while(!(UCSR0A & (1<<UDRE0)));
    //when the buffer is empty write data to the transmitted
    UDR0 = data;
    return data;
}

unsigned char uart_rx(void) {
	// Wait for data to be received
	while (!(UCSR0A & (1<<RXC0)));
	// Get and return received data from buffer
	return UDR0;
}
*/

unsigned int uart_available (void) {
	return RxFifo.ct;
}

uint8_t uart_rx (void) {
	uint8_t d, i;

	while (RxFifo.ct == 0) ;

	i = RxFifo.ri;
	d = RxFifo.buff[i];
	cli();
	RxFifo.ct--;
	sei();
	RxFifo.ri = (i + 1) & (UART_BUFF_SIZE-1);

	return d;
}

void uart_tx (uint8_t d) {
	uint8_t i;

	while (TxFifo.ct >= UART_BUFF_SIZE ) ;

	i = TxFifo.wi;
	TxFifo.buff[i] = d;
	cli();
	TxFifo.ct++;
	UCSR0B |= (1<<UDRIE0);
	sei();
	TxFifo.wi = (i + 1) & (UART_BUFF_SIZE-1);
}


ISR(USART0_RX_vect)
{
	uint8_t d, n, i;

	d = UDR0;
	n = RxFifo.ct;
	if (n < UART_BUFF_SIZE) {
		RxFifo.ct = ++n;
		i = RxFifo.wi;
		RxFifo.buff[i] = d;
		RxFifo.wi = (i + 1) & (UART_BUFF_SIZE-1);
	}
}

ISR(USART0_UDRE_vect)
{
	uint8_t n, i;


	n = TxFifo.ct;
	if (n) {
		TxFifo.ct = --n;
		i = TxFifo.ri;
		UDR0 = TxFifo.buff[i];
		TxFifo.ri = (i + 1) & (UART_BUFF_SIZE-1);
	}
	if (n == 0)
		UCSR0B &= ~(1<<UDRIE0);
}


inline void uart_putchar(unsigned char c) {
	uart_tx(c);
}

int uart_getchar() {
	if ( uart_available() == 0 ) // empty
		return -1;
	return uart_rx();
}

int uart_peek() {
	if ( uart_available() == 0 ) // empty
		return -1;
	return RxFifo.buff[RxFifo.ri];
}

