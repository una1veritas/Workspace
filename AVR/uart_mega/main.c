/*
 * main.c
 *
 *  Created on: 2017/08/28
 *      Author: sin
 */

#include <avr/io.h>
#include <util/delay.h>

#include "uart.h"

int main(void) {

//	char buf[128];
	int c;

	DDRB |= (1<<PB5);

	uart_init(19200);

	uart_putstr("Hello there.\n");
	uart_putstr("Now going to try to receive...\n");

	for(;;) {/*
		if ( uart_rxavailable() > 0 ) {
			uart_putnum_u16(debug_rxenq(), 2);
			uart_putstr("[");
			uart_putnum_u16(debug_rxdeq(), 2);
			uart_putstr(", ");
			uart_putnum_u16(debug_rxenq(), 2);
			uart_putstr(")");
			uart_putstr(" received:\n");
			do {
				uart_putchar((char)c);
			} while ( (c = uart_getchar()) != -1 );
			uart_putstr("\n");
		}*/
		for(int i = 0; i < FIFO_SIZE; i++) {
			uart_puthex(debug_rxfifo()[i]);
			uart_putchar(' ');
		}
		if ( uart_available() ) {
			while ( (c = uart_getchar()) != -1 ) {
				if ( isprint(c) )
					uart_putchar((char)c);
				else {
					uart_putchar('(');
					uart_puthex(c);
					uart_putchar(')');
				}
			}
			uart_putstr("\n");
		}
		uart_putchar('\n');
		PORTB |= (1<<PB5);
		_delay_ms(500);
		PORTB &= ~(1<<PB5);
		_delay_ms(500);
	}
	return 0;
}
