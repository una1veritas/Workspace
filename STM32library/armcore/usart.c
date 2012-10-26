/*
 * usart.c
 *
 *  Created on: 2012/10/24
 *      Author: sin
 */

#include <stm32f4xx_gpio.h>
#include <stm32f4xx_usart.h>
#include <misc.h>

#include "usart.h"

#define USART_BUFFER_SIZE 128
struct USARTBuffer {
	uint16_t buf[USART_BUFFER_SIZE];
	int16_t head, tail;
	uint16_t count;
} rx[3], tx[3], rx3, tx3;

void buffer_init(struct USARTBuffer * x) {
	x->head = 0;
	x->tail = 0;
	x->count = 0;
}

enum PortNumber {
	USART_PORT1 = 0,
	USART_PORT2,
	USART_PORT3,
};

uint8_t USART_id(USART_TypeDef * USARTx) {
	if (USARTx == USART2 ) {
		return USART_PORT2;
	} else if ( USARTx == USART3 ){
		return USART_PORT3;
	} else {
		return USART_PORT1;
	}
}


void usart_begin(USART_TypeDef * USARTx, uint32_t baud) {
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

	USART_Init(USARTx, &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting

	 USART_ITConfig(USARTx, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	 USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);

	 NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
	 // we want to configure the USART3 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;// this sets the priority group of the USART3 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;// this sets the subpriority inside the group
	 NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART3 interrupts are globally enabled
	 NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff

	 buffer_init(&rx[USART_id(USARTx)]);
	 buffer_init(&tx[USART_id(USARTx)]);

	// finally this enables the complete USART3 peripheral
	USART_Cmd(USARTx, (FunctionalState) ENABLE);
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
	 NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;// this sets the priority group of the USART3 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;// this sets the subpriority inside the group
	 NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART3 interrupts are globally enabled
	 NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff

	 buffer_init(&rx3);
	 buffer_init(&tx3);

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
	if ( (tx3.head == tx3.tail) && tx3.count > 0 ) // queue is full
	{
		while (USART_GetFlagStatus(USART3, USART_FLAG_TC ) == RESET);
	}
	USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
	tx3.buf[tx3.head++] = w;
	tx3.count++;
	USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) ENABLE);
}

void usart_write(USART_TypeDef * USARTx, uint16_t w) {
	if ( (tx[USART_id(USARTx)].head == tx[USART_id(USARTx)].tail) && tx[USART_id(USARTx)].count > 0 ) // queue is full
	{
		while (USART_GetFlagStatus(USARTx, USART_FLAG_TC ) == RESET);
	}
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);
	tx[USART_id(USARTx)].buf[tx[USART_id(USARTx)].head++] = w;
	tx[USART_id(USARTx)].count++;
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) ENABLE);
}

void usart3_print(char * s) {
	while (*s)
		usart3_write((uint16_t) *s++);
}

void usart_print(USART_TypeDef * USARTx, char * s) {
	while (*s)
		usart_write(USARTx, (uint16_t) *s++);
}

uint16_t usart3_bare_read() {
	return USART_ReceiveData(USART3 );
}

uint16_t usart3_read() {
	if ( (rx3.head != rx3.tail) || rx3.count > 0) {
		uint16_t c = rx3.buf[rx3.tail++];
		rx3.tail %= USART_BUFFER_SIZE;
		rx3.count--;
		return c;
	} else
		return 0; // buffer is empty
}

uint8_t usart3_rxne() {
	return USART_GetFlagStatus(USART3, USART_FLAG_RXNE ) == SET;
}

uint8_t usart3_available() {
	return rx3.count;
}

// this is the interrupt request handler (IRQ) for ALL USART3 interrupts

void USART3_IRQHandler(void) {
	if (USART_GetITStatus(USART3, USART_IT_RXNE )) {
		rx3.buf[rx3.head++] = USART_ReceiveData(USART3 );
		rx3.head %= USART_BUFFER_SIZE;
		rx3.count++;
	}

	if (USART_GetITStatus(USART3, USART_IT_TXE )) {
		if (tx[USART_PORT3].count == 0) {
			USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
			USART_ClearITPendingBit(USART3, USART_IT_TXE );
		} else {
			USART_SendData(USART3, tx[USART_PORT3].buf[tx[USART_PORT3].tail++]);
			tx[USART_PORT3].tail %= USART_BUFFER_SIZE;
			tx[USART_PORT3].count--;
		}
	}
}
