/*
 * uart_serial.c
 *
 *  Created on: 2017/09/03
 *      Author: sin
 */

#include <avr/io.h>
#include <util/delay.h>

#include <stdio.h>
#include <stdlib.h>

#include "uart_serial.h"
#include <avr/interrupt.h>


void uart_init(uint32_t baud) {
    cli();
    // Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet

    UBRR0 = (((F_CPU / (baud * 16UL))) - 1);                 // Set the baud rate prescale rate register
    UCSR0B = ((1<<RXEN0)|(1<<TXEN0)|(1 << RXCIE0));       // Enable receiver and transmitter and Rx interrupt
    UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));  // Set frame format: 8data, 1 stop bit. See Table 22-7 for details
    sei();
}
/*
void serial0_init(uint32_t baud) {
	// Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet
	UBRR0 = (((F_CPU / (baud * 16UL))) - 1);         // Set the baud rate prescale rate register

	UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));   // Set frame format: 8data, 1 stop bit. See Table 22-7 for details
	UCSR0B = ((1<<RXEN0)|(1<<TXEN0));       // Enable receiver and transmitter
}
*/

void uart_tx(uint8_t data) {
    //while the transmit buffer is not empty loop
    while(!(UCSR0A & (1<<UDRE0)));

    //when the buffer is empty write data to the transmitted
    UDR0 = data;
}

uint8_t uart_rx(void) {
	/* Wait for data to be received */
	while (!(UCSR0A & (1<<RXC0)));
	/* Get and return received data from buffer */
	return UDR0;
}

/*
void serial0_puts(char* StringPtr) {
// sends the characters from the string one at a time to the USART
    while(*StringPtr != 0x00) {
        serial0_tx(*StringPtr);
        StringPtr++;
    }
}
*/

