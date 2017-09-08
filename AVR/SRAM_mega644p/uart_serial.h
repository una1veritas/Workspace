/*
 * uart_serial.h
 *
 *  Created on: 2017/09/03
 *      Author: sin
 */

#ifndef UART_SERIAL_H_
#define UART_SERIAL_H_

void uart_init(uint32_t baud);
void uart_tx(uint8_t data);
uint8_t uart_rx(void);

#endif /* UART_SERIAL_H_ */
