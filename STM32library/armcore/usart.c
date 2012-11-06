/*
 * usart.c
 *
 *  Created on: 2012/10/24
 *      Author: sin
 */

#include <stm32f4xx_gpio.h>
#include <stm32f4xx_rcc.h>
#include <stm32f4xx_usart.h>
#include <misc.h>
#include <stm32f4xx.h>

#include "gpio.h"
#include "usart.h"

#define USART_BUFFER_SIZE 128
typedef struct {
	uint16_t buf[USART_BUFFER_SIZE];
	int16_t head, tail;
	uint16_t count;
} USARTRing;

USART_TypeDef * usartx[] = {
		USART1, USART2, USART3, UART4, UART5, USART6
};

//extern USARTRing rxring[3], txring[3];
USARTRing rxring[3], txring[3];

void buffer_clear(USARTRing * r) {
	r->head = 0;
	r->tail = 0;
	r->count = 0;
}

uint16_t buffer_count(USARTRing * r) {
	return r->count;
}

uint8_t buffer_is_full(USARTRing * r) {
	if ( (r->head == r->tail) && (r->count > 0) ) {
		return 1;
	}
	return 0;
}

uint16_t buffer_enque(USARTRing * r, uint16_t w) {
	if ( buffer_is_full(r) )
		return 0xffff;
	r->buf[r->head++] = w;
	r->count++;
	r->head %= USART_BUFFER_SIZE;
	return w;
}

uint16_t buffer_deque(USARTRing * r) {
	uint16_t w;
	if ( buffer_count(r) == 0 )
		return 0xffff;
	w = r->buf[r->tail++];
	r->count--;
	r->tail %= USART_BUFFER_SIZE;
	return w;
}
/*
uint8_t USART_id(USART_TypeDef * USARTx) {
	if ( USARTx == USART1 )
		return USART_1;
	else if ( USARTx == USART2 )
		return USART_2;
	else if ( USARTx == USART3 )
		return USART_3;
	else if ( USARTx == UART4 )
		return UART_4;
	else if ( USARTx == UART5 )
		return UART_5;
	else if ( USARTx == USART6 )
		return USART_6;
	return USART_1;
}
*/
void usart_begin(USARTSerial usx, GPIOPin rx, GPIOPin tx, const uint32_t baud) {
	//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

	uint8_t af = GPIO_AF_USART1;
	IRQn_Type irq = USART1_IRQn;

	switch(usx) {
	case USART1Serial:
		RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
		af = GPIO_AF_USART1;
		irq = USART1_IRQn;
	break;
	case USART2Serial:
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE);
		af = GPIO_AF_USART2;
		irq = USART2_IRQn;
	break;
	case USART3Serial:
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, ENABLE);
		af = GPIO_AF_USART3;
		irq = USART3_IRQn;
	break;
	case UART4Serial:
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_UART4, ENABLE);
		af = GPIO_AF_UART4;
		irq = UART4_IRQn;
	break;
	case UART5Serial:
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_UART5, ENABLE);
		af = GPIO_AF_UART5;
		irq = UART5_IRQn;
		break;
	case USART6Serial:
	default:
		usx = USART6Serial;
		RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART6, ENABLE);
		af = GPIO_AF_USART6;
		irq = USART6_IRQn;
		break;
	}
	GPIOMode(pinPort(rx), pinBit(rx), GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);
	GPIOMode(pinPort(tx), pinBit(tx), GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	GPIO_PinAFConfig(pinPort(rx), pinSource(rx), af );
	GPIO_PinAFConfig(pinPort(tx), pinSource(tx), af );

	USART_InitStruct.USART_BaudRate = baud;	// the baudrate is set to the value we passed into this init function
	USART_InitStruct.USART_WordLength = USART_WordLength_8b;// we want the data frame size to be 8 bits (standard)
	USART_InitStruct.USART_StopBits = USART_StopBits_1;	// we want 1 stop bit (standard)
	USART_InitStruct.USART_Parity = USART_Parity_No;// we don't want a parity bit (standard)
	USART_InitStruct.USART_HardwareFlowControl = USART_HardwareFlowControl_None; // we don't want flow control (standard)
	USART_InitStruct.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // we want to enable the transmitter and the receiver

	USART_Init(usartx[usx], &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting

	USART_ITConfig(usartx[usx], USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	USART_ITConfig(usartx[usx], USART_IT_TXE, (FunctionalState) DISABLE);

	NVIC_InitStructure.NVIC_IRQChannel = irq;
	// we want to configure the USART3 interrupts
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0; // this sets the priority group of the USART3 interrupts
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0; // this sets the subpriority inside the group
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART3 interrupts are globally enabled
	NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff

	buffer_clear(&rxring[usx]);
	buffer_clear(&txring[usx]);

	// finally this enables the complete USART3 peripheral
	USART_Cmd(usartx[usx], ENABLE);
}

void usart_bare_write(USART_TypeDef * USARTx, const uint16_t w) {
	while (USART_GetFlagStatus(USARTx, USART_FLAG_TXE ) == RESET)
		;
	USART_SendData(USARTx, w);
//	while (USART_GetFlagStatus(USART3, USART_FLAG_TC ) == RESET);
}

void usart_write(USART_TypeDef * USARTx, const uint16_t w) {
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);
	buffer_enque(&txring[USART_id(USARTx)], w);
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) ENABLE);
}

void usart_print(USART_TypeDef * USARTx, const char * s) {
	while (*s)
		usart_write(USARTx, (uint16_t) *s++);
}

uint16_t usart_bare_read(USART_TypeDef * USARTx) {
	return USART_ReceiveData(USARTx );
}

uint16_t usart_read(USART_TypeDef * USARTx) {
	uint16_t w = buffer_deque(&rxring[USART_id(USARTx)]);
	if ( w == 0xffff ) return 0; // buffer is empty
	return w;
}

void usart_flush(USART_TypeDef * USARTx) {
	USART_ITConfig(USARTx, USART_IT_RXNE, (FunctionalState) DISABLE); // enable the USART3 receive interrupt
	buffer_clear(&rxring[USART_id(USARTx)]);
	USART_ClearITPendingBit(USARTx, USART_IT_RXNE );
	USART_ITConfig(USARTx, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);
	while ( buffer_count(&txring[USART_id(USARTx)]) > 0 ) {
		while (USART_GetFlagStatus(USARTx, USART_FLAG_TXE ) == RESET);
		USART_SendData(USARTx, buffer_deque(&txring[USART_id(USARTx)]));
		while (USART_GetFlagStatus(USARTx, USART_FLAG_TC ) == RESET);
	}
	USART_ClearITPendingBit(USARTx, USART_IT_TXE );
	buffer_clear(&txring[USART_id(USARTx)]);
}

uint16_t usart_peek(USART_TypeDef * USARTx) {
	if ( ! buffer_count(&rxring[USART_id(USARTx)]) == 0 )
		return rxring[USART_id(USARTx)].buf[rxring[USART_id(USARTx)].tail];
	return 0xffff;
}

uint16_t usart_available(USART_TypeDef * USARTx) {
	return buffer_count(&rxring[USART_id(USARTx)]);
}

// this is the interrupt request handler (IRQ) for ALL USART3 interrupts

void USART1_IRQHandler(void) {
	if (USART_GetITStatus(USART1, USART_IT_RXNE )) {
		buffer_enque(&rxring[USART_1], USART_ReceiveData(USART1) );
	}

	if (USART_GetITStatus(USART1, USART_IT_TXE )) {
		if (txring[USART_1].count == 0) {
			USART_ITConfig(USART1, USART_IT_TXE, (FunctionalState) DISABLE);
			USART_ClearITPendingBit(USART1, USART_IT_TXE );
		} else {
			USART_SendData(USART1, buffer_deque(&txring[USART_1]));
		}
	}
}

void USART3_IRQHandler(void) {
	if (USART_GetITStatus(USART3, USART_IT_RXNE )) {
		buffer_enque(&rxring[USART_3], USART_ReceiveData(USART3) );
	}

	if (USART_GetITStatus(USART3, USART_IT_TXE )) {
		if (txring[USART_3].count == 0) {
			USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
			USART_ClearITPendingBit(USART3, USART_IT_TXE );
		} else {
			USART_SendData(USART3, buffer_deque(&txring[USART_3]));
		}
	}
}
