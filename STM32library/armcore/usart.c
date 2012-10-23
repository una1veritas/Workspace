/*
 * usart.c
 *
 *  Created on: 2012/10/24
 *      Author: sin
 */

#include "usart.h"

void usart3_begin(uint32_t baud) {
	//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
//	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

	//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, (FunctionalState) ENABLE);
	GPIOMode(PB10 | PB11, GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_UP);
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
			/*
			 USART_ITConfig(USART3, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
			 USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);

			 NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
			 // we want to configure the USART3 interrupts
			 NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;// this sets the priority group of the USART3 interrupts
			 NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;// this sets the subpriority inside the group
			 NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART3 interrupts are globally enabled
			 NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff
			 */
	// finally this enables the complete USART3 peripheral
	USART_Cmd(USART3, (FunctionalState) ENABLE);
}

void usart3_write(uint16_t w) {
	while (USART_GetFlagStatus(USART3, USART_FLAG_TXE ) == RESET);
	USART_SendData(USART3, w);
//	while (USART_GetFlagStatus(USART3, USART_FLAG_TC ) == RESET);
}

void usart3_print(char * s) {
	while (*s)
		usart3_write( (uint16_t) *s++);
}

uint16_t usart3_read() {
	return USART_ReceiveData(USART3);
}

uint8_t usart3_available() {
	return USART_GetFlagStatus(USART3, USART_FLAG_RXNE) == SET;
}
