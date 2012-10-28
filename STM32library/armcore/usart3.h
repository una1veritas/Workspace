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

#include "gpio.h"

enum USARTPortNumber {
	USART_1 = 1,
	USART_2,
	USART_3
};

#ifdef __cplusplus
extern "C" {
#endif

struct USARTBuffer;
extern struct USARTBuffer rx3, tx3;

void usart3_begin(const uint32_t baud);
void usart3_write(const uint16_t w);
void usart3_print(const char * s);
uint16_t usart3_read(void);
uint16_t usart3_available(void);
void usart3_flush(void);
uint16_t usart3_peek(void);

uint16_t tx_head(void);
uint16_t tx_tail(void);

void USART3_IRQHandler(void);
uint16_t usart3_irq_read(void);
uint8_t usart3_rxne(void);

#ifdef __cplusplus
}
#endif

#endif /* USART_H_ */
