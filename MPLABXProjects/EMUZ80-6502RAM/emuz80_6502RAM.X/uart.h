/* 
 * File:   uart.h
 * Author: sin
 *
 * Created on December 3, 2025, 12:06 AM
 */

#ifndef UART_H
#define	UART_H

#ifdef	__cplusplus
extern "C" {
#endif

void uart_tx(char c);
int uart_rx(void);
void devio_init(void);


#ifdef	__cplusplus
}
#endif

#endif	/* UART_H */

