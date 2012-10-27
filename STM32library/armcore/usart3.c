/*
 * usart.c
 *
 *  Created on: 2012/10/24
 *      Author: sin
 */

#include <stm32f4xx_gpio.h>
#include <stm32f4xx_usart.h>
#include <misc.h>

#include "usart3.h"

#define USART_BUFFER_SIZE 128
typedef struct {
	uint16_t buf[USART_BUFFER_SIZE];
	int16_t head, tail;
	uint16_t count;
} USARTBuffer;
USARTBuffer rx3, tx3;

void buffer_clear(USARTBuffer * b) {
	b->head = 0;
	b->tail = 0;
	b->count = 0;
}

uint16_t buffer_remainder(USARTBuffer * b) {
	return b->count;
}

uint8_t buffer_is_full(USARTBuffer * b) {
	if ( (b->head == b->tail) && (b->count > 0) ) {
		return 1;
	}
	return 0;
}

uint8_t buffer_is_empty(USARTBuffer * b) {
	if ( (b->count == 0) /*&& (b->head == b->tail) */) {
		return 1;
	}
	return 0;
}

uint16_t buffer_enq(USARTBuffer * b, volatile uint16_t w) {
	if ( buffer_is_full(b) )
		return 0xffff;
	b->buf[b->head++] = w;
	b->count++;
	b->head %= USART_BUFFER_SIZE;
	return w;
}

uint16_t buffer_deq(USARTBuffer * b) {
	if ( buffer_is_empty(b) )
		return 0xffff;
	uint16_t w = b->buf[b->tail++];
	b->count--;
	b->tail %= USART_BUFFER_SIZE;
	return w;
}


void usart3_begin(uint32_t baud) {
	//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

	//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, (FunctionalState) ENABLE);
	GPIOMode(PB10 | PB11, GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP,
			GPIO_PuPd_UP);
	/* USART3 clock enable */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, ENABLE);

	GPIO_PinAFConfig(GPIOB, GPIO_PinSource10, GPIO_AF_USART3 ); // TX -- PB10
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource11, GPIO_AF_USART3 ); // RX -- PB11

	USART_InitStruct.USART_BaudRate = baud;	// the baudrate is set to the value we passed into this init function
	USART_InitStruct.USART_WordLength = USART_WordLength_8b;// we want the data frame size to be 8 bits (standard)
	USART_InitStruct.USART_StopBits = USART_StopBits_1;	// we want 1 stop bit (standard)
	USART_InitStruct.USART_Parity = USART_Parity_No;// we don't want a parity bit (standard)
	USART_InitStruct.USART_HardwareFlowControl = USART_HardwareFlowControl_None; // we don't want flow control (standard)
	USART_InitStruct.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // we want to enable the transmitter and the receiver

	USART_Init(USART3, &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting

	USART_ITConfig(USART3, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);

	NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
	// we want to configure the USART3 interrupts
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0; // this sets the priority group of the USART3 interrupts
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0; // this sets the subpriority inside the group
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART3 interrupts are globally enabled
	NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff

	buffer_clear(&rx3);
	buffer_clear(&tx3);

	// finally this enables the complete USART3 peripheral
	USART_Cmd(USART3, (FunctionalState) ENABLE);
}

void usart3_bare_write(uint16_t w) {
	while (USART_GetFlagStatus(USART3, USART_FLAG_TXE ) == RESET)
		;
	USART_SendData(USART3, w);
//	while (USART_GetFlagStatus(USART3, USART_FLAG_TC ) == RESET);
}

void usart3_write(uint16_t w) {
	USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
	buffer_enq(&tx3, w);
	USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) ENABLE);
}

void usart3_print(char * s) {
	while (*s)
		usart3_write((uint16_t) *s++);
}

uint16_t usart3_bare_read() {
	return USART_ReceiveData(USART3 );
}

uint16_t usart3_read() {
	uint16_t w = buffer_deq(&rx3);
	if ( w == 0xffff ) return 0; // buffer is empty
	return w;
}

void usart3_flush() {
	USART_ITConfig(USART3, USART_IT_RXNE, (FunctionalState) DISABLE); // enable the USART3 receive interrupt
	buffer_clear(&rx3);
	USART_ITConfig(USART3, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	USART_ClearITPendingBit(USART3, USART_IT_RXNE );
	USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
	while ( !buffer_is_empty(&tx3) ) {
		while (USART_GetFlagStatus(USART3, USART_FLAG_TXE ) == RESET);
		USART_SendData(USART3, buffer_deq(&tx3));
		while (USART_GetFlagStatus(USART3, USART_FLAG_TC ) == RESET);
	}
	USART_ClearITPendingBit(USART3, USART_IT_TXE );
	buffer_clear(&tx3);
}

uint16_t usart3_peek() {
	if ( !buffer_is_empty(&rx3) )
		return rx3.buf[rx3.tail];
	return 0xffff;
}

uint16_t usart3_available() {
	return buffer_remainder(&rx3);
}

uint16_t tx_head() {
	return tx3.head;
}

uint16_t tx_tail() {
	return tx3.tail;
}
// this is the interrupt request handler (IRQ) for ALL USART3 interrupts

void USART3_IRQHandler(void) {
	if (USART_GetITStatus(USART3, USART_IT_RXNE )) {
		buffer_enq(&rx3, USART_ReceiveData(USART3) );
	}

	if (USART_GetITStatus(USART3, USART_IT_TXE )) {
		if (tx3.count == 0) {
			USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
			USART_ClearITPendingBit(USART3, USART_IT_TXE );
		} else {
			USART_SendData(USART3, buffer_deq(&tx3));
		}
	}
}
