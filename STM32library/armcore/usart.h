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

void usart_begin(USART_TypeDef * usart, uint32_t baud);
void usart3_begin(uint32_t baud);
void usart3_write(uint16_t w);
void usart3_print(char * s);
uint16_t usart3_read();
uint8_t usart3_available();

void USART3_IRQHandler(void);
uint16_t usart3_irq_read();
uint8_t usart3_rxne();

#ifdef __cplusplus
}
#endif

#endif /* USART_H_ */
