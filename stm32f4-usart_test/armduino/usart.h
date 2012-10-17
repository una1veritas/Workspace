/*
 * usart.h
 *
 *  Created on: 2012/10/14
 *      Author: sin
 */

#ifndef USART_H_
#define USART_H_

#include <stdint.h>

void usart_begin(uint32_t baud);
uint16_t usart_write(uint8_t ch);
uint16_t usart_print(const char * s);
//size_t usart_printInt(uint32 val, uint8 base);
uint16_t usart_printNumber(uint32_t val);
uint16_t usart_printFloat(float val, uint8_t prec);


#endif /* USART_H_ */
