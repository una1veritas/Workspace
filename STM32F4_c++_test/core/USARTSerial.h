/*
 * USARTSerial.h
 *
 *  Created on: 2012/10/10
 *      Author: sin
 */

#ifndef USARTSERIAL_H_
#define USARTSERIAL_H_


#include <stm32f4xx.h>
#include <misc.h>			 // I recommend you have a look at these in the ST firmware folder
#include <stm32f4xx_usart.h> // under Libraries/STM32F4xx_StdPeriph_Driver/inc and src

#define MAX_STRLEN 12 // this is the maximum string length of our string in characters
volatile char received_string[MAX_STRLEN+1]; // this will hold the recieved string

void init_USART1(uint32_t baudrate);
void USART_puts(USART_TypeDef* USARTx, volatile char *s);
void USART_putch(USART_TypeDef* USARTx, int ch);
void USART1_IRQHandler(void);

#endif /* USARTSERIAL_H_ */
