/*
 * Copyright (c) 2016, Takashi TOYOSHIMA <toyoshim@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * - Neither the name of the authors nor the names of its contributors may be
 *   used to endorse or promote products derived from this software with out
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUE
 * NTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#include <avr/io.h>
#include <avr/interrupt.h>

#include "uart.h"

static void reset_int (void) {
	UCSR0B |= (1<<RXEN0);
}

static void enable_int (void) {
	UCSR0B |= (1<<RXEN0);
}

static void disable_int (void) {
	UCSR0B &= ~(1<<RXEN0);
}

static unsigned char
fifo[8];

static unsigned char
fifo_rdptr = 0;

static unsigned char
fifo_wrptr = 0;

static void uart_push(unsigned char c) {
  if (fifo_rdptr != (fifo_wrptr + 1)) {
    fifo[fifo_wrptr++] = c;
    fifo_wrptr &= 7;
  }
}

void uart_init(unsigned long baud) {
	//duint32_t baud = 19200;
    cli();
    // Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet

    UBRR0 = (((F_CPU / (baud * 16UL))) - 1);         // Set the baud rate prescale rate register
    UCSR0B = ((1<<RXEN0)|(1<<TXEN0)|(1 << RXCIE0));  // Enable receiver and transmitter and Rx interrupt
    UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));  // Set frame format: 8data, 1 stop bit. See Table 22-7 for details
    sei();
}

unsigned char uart_tx(unsigned char data) {
    //while the transmit buffer is not empty loop
    while(!(UCSR0A & (1<<UDRE0)));

    //when the buffer is empty write data to the transmitted
    UDR0 = data;
    return data;
}

unsigned char uart_rx(void) {
	/* Wait for data to be received */
	while (!(UCSR0A & (1<<RXC0)));
	/* Get and return received data from buffer */
	return UDR0;
}

ISR(USART0_RX_vect)
{
	uart_push(UDR0);
}


void uart_putchar(unsigned char c){
//  unsigned short rc = uart_tx(c);
//  if (0 != rc) uart_push(rc);
	uart_tx(c);
}

int
uart_getchar
(void)
{
  if (fifo_rdptr == fifo_wrptr) return -1;
  cli();
  int rc = fifo[fifo_rdptr++];
  fifo_rdptr &= 7;
  sei();
  return rc;
}

int uart_peek(void) {
  return (fifo_wrptr - fifo_rdptr) & 7;
}
