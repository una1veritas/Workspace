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

typedef enum _USARTSerial {
	USART1Serial = 0,
	USART2Serial,
	USART3Serial,
	UART4Serial,
	UART5Serial,
	USART6Serial
} USARTSerial;


/*
PB6 			USART1_TX
PB7			USART1_RX
PA9 			USART1_TX
PA10 		USART1_RX
PD5 			USART2_TX
PD6 			USART2_RX
PA2			USART2_TX
PA3			USART2_RX
PB10 		USART3_TX
PB11		USART3_RX
PD8 			USART3_TX
PD9 			USART3_RX
PC10 		USART3_TX
PC11		USART3_RX
PA0-WKUP	USART4_TX
PA1			USART4_RX
PC12		UART5_TX
PD2 			UART5_RX
PG9			USART6_RX
PG14 		USART6_TX
PC6 			USART6_TX
PC7 			USART6_RX
*/

void usart_begin(USART_TypeDef * /*USARTSerial*/ usx, GPIOPin rx, GPIOPin tx, const uint32_t baud);
void usart_write(USART_TypeDef * /*USARTSerial*/ usx, const uint16_t w);
void usart_print(USART_TypeDef * /*USARTSerial*/ usx, const char * s);
uint16_t usart_read(USART_TypeDef * /*USARTSerial*/ usx);
uint16_t usart_available(USART_TypeDef * /*USARTSerial*/ usx);
void usart_flush(USART_TypeDef * /*USARTSerial*/ usx);
uint16_t usart_peek(USART_TypeDef * /*USARTSerial*/ usx);

void USART1_IRQHandler(void);
void USART2_IRQHandler(void);
void USART3_IRQHandler(void);
void UART4_IRQHandler(void);
void UART5_IRQHandler(void);
void USART6_IRQHandler(void);

#ifdef __cplusplus
}
#endif

#endif /* USART_H_ */
