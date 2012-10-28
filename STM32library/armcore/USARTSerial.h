/*
 * USARTSerial.h
 *
 *  Created on: 2012/10/27
 *      Author: sin
 */

#ifndef USARTSERIAL_H_
#define USARTSERIAL_H_

#include <stm32f4xx_usart.h>
#include "gpio.h"
#include "usart3.h"

class USARTSerial {
	const USART_TypeDef * uport;
	const USARTBuffer * rx, * tx;

public:
	uint32_t port(void) { return (uint32_t) uport; }

	USARTSerial(USART_TypeDef * usartx, USARTBuffer * rxbuf, USARTBuffer * txbuf) 
		: uport(usartx), rx(rxbuf), tx(txbuf) { }

	void begin(const uint32_t baud) {
		usart3_begin(baud);
	}
	void write(const uint16_t w) {
		usart3_write(w);
	}
	void print(const char * s) {
		usart3_print(s);
	}
	uint16_t read() {
		return usart3_read();
	}
	uint16_t available() {
		return usart3_available();
	}
	void flush() {
		usart3_flush();
	}
	uint16_t peek() {
		return usart3_peek();
	}

};

//extern USARTSerial Serial3;

#endif /* USARTSERIAL_H_ */
