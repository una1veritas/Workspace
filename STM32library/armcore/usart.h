/*
 * usart.h
 *
 *  Created on: 2012/10/24
 *      Author: sin
 */

#ifndef USART_H_
#define USART_H_

#include <stdio.h>
#include <stdlib.h>

#include <stm32f4xx_usart.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "gpio.h"

#define USART_BUFFER_SIZE 128
typedef struct {
	USART_TypeDef * usartx;
	uint16_t rx_buf[USART_BUFFER_SIZE], tx_buf[USART_BUFFER_SIZE];
	int16_t rx_head, rx_tail, tx_head, tx_tail;
	uint16_t rx_count, tx_count;
} USARTBuffer;

USARTBuffer usart3, usart2, usart1;

void usart_begin(USART_TypeDef * USARTx, USARTBuffer * usartb, uint32_t baud);
//void usart3_begin(uint32_t baud);
//void usart3_write(uint16_t w);
//void usart3_print(char * s);
uint16_t usart_read(USARTBuffer * b);
void usart_write(USARTBuffer * usartb, uint16_t w);
void usart_print(USARTBuffer * usartb, char * s);
//uint16_t usart3_read();
uint8_t usart_available(USARTBuffer * usartb);

void USART3_IRQHandler(void);
uint16_t usart3_irq_read();
uint8_t usart3_rxne();

#ifdef __cplusplus
}
#endif

#endif /* USART_H_ */
