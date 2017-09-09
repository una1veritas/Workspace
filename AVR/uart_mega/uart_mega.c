
#include <avr/io.h>
#include <avr/interrupt.h>

#include "uart.h"

static unsigned char rxfifo[FIFO_SIZE];
static unsigned char rxenq = 0;
static unsigned char rxdeq = 0;

#ifdef USE_INTERRUPT_TX
static USART_FIFO txfifo = { "", 0, 0 };
#endif


int debug_rxenq() {
	return rxenq;
}
int debug_rxdeq() {
	return rxdeq;
}
unsigned char * debug_rxfifo() {
	return rxfifo;
}


void uart_init(unsigned long baud) {
	//duint32_t baud = 19200;
    cli();
    // Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet

    UBRR0 = (((F_CPU / (baud * 16UL))) - 1);         // Set the baud rate prescale rate register
    UCSR0B = ((1<<RXEN0)|(1<<TXEN0));  // Enable receiver and transmitter and Rx interrupt
    UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));  // Set frame format: 8data, 1 stop bit. See Table 22-7 for details

    UCSR0B |= (1 << RXCIE0);
#ifdef USE_INTERRUPT_TX
    UCSR0B |= (1 << TXCIE0);
#endif
    sei();
}

unsigned char uart_tx(unsigned char data) {
    //while the transmit buffer is not empty loop
    while(!(UCSR0A & (1<<UDRE0)));
    //when the buffer is empty write data to the transmitted
    UDR0 = data;
    return data;
}

/* this function is no use */
unsigned char uart_rx(void) {
	/* Wait for data to be received */
	while (!(UCSR0A & (1<<RXC0)));
	/* Get and return received data from buffer */
	return UDR0;
}


ISR(USARTn_RX_vect)
{
	rxfifo[rxenq++] = UDR0;
	rxenq &= (FIFO_SIZE-1);
	if ( rxenq == rxdeq ) {
		// waste the oldest data
		rxdeq++;
		rxdeq &= (FIFO_SIZE-1);
	}
}

void uart_putchar(unsigned char c) {
	uart_tx(c);
}

int uart_getchar() {
	if ( rxdeq == rxenq ) // empty
		return -1;
	char c = rxfifo[rxdeq++];
	rxdeq &= (FIFO_SIZE-1);
	return c;
}

int uart_peek() {
	if ( rxenq == rxdeq )
		return -1;
	return rxfifo[rxdeq];
}

unsigned int uart_available() {
	return (rxenq + FIFO_SIZE - rxdeq) & (FIFO_SIZE-1);
}
