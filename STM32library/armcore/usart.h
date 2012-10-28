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

#ifdef __cplusplus
extern "C" {
#endif

#define USART_BUFFER_SIZE 128
typedef struct {
	uint16_t buf[USART_BUFFER_SIZE];
	int16_t head, tail;
	uint16_t count;
} USARTRing;

extern USARTRing rxring[3], txring[3];

void usart_begin(USART_TypeDef * uport, const uint32_t baud);
void usart_write(USART_TypeDef * uport, const uint16_t w);
void usart_print(USART_TypeDef * uport, const char * s);
uint16_t usart_read(USART_TypeDef * uport);
uint16_t usart_available(USART_TypeDef * uport);
void usart_flush(USART_TypeDef * uport);
uint16_t usart_peek(USART_TypeDef * uport);


void USART1_IRQHandler(void);
void USART2_IRQHandler(void);
void USART3_IRQHandler(void);

#ifdef __cplusplus
}
#endif

#endif /* USART_H_ */
