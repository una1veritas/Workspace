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



USARTBuffer usart3, usart2, usart1;

void usart_buffer_init(USARTBuffer * x, USART_TypeDef * USARTx) {
	x->usartx = USARTx;
	x->rx_head = 0;
	x->rx_tail = 0;
	x->rx_count = 0;
	x->tx_head = 0;
	x->tx_tail = 0;
	x->tx_count = 0;
}

/*
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
*/

void usart_begin(USART_TypeDef * USARTx, USARTBuffer * usartb, uint32_t baud) {
	//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

	usart_buffer_init(usartb, USARTx);

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

	USART_Init(usartb->usartx, &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting

	 USART_ITConfig(usartb->usartx, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	 USART_ITConfig(usartb->usartx, USART_IT_TXE, (FunctionalState) DISABLE);

	 NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
	 // we want to configure the USART3 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;// this sets the priority group of the USART3 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;// this sets the subpriority inside the group
	 NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART3 interrupts are globally enabled
	 NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff

	// finally this enables the complete USART3 peripheral
	USART_Cmd(usartb->usartx, (FunctionalState) ENABLE);
}

/*
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
*/

void usart_write(USARTBuffer * usartb, uint16_t w) {
	if ( (usartb->tx_head == usartb->tx_tail) && usartb->tx_count > 0 ) // queue is full
	{
		while (USART_GetFlagStatus(usartb->usartx, USART_FLAG_TC ) == RESET);
	}
	USART_ITConfig(usartb->usartx, USART_IT_TXE, (FunctionalState) DISABLE);
	usartb->tx_buf[usartb->tx_head++] = w;
	usartb->tx_count++;
	USART_ITConfig(usartb->usartx, USART_IT_TXE, (FunctionalState) ENABLE);
}
/*
void usart3_print(char * s) {
	while (*s)
		usart3_write((uint16_t) *s++);
}
*/
void usart_print(USARTBuffer * usartb, char * s) {
	while (*s)
		usart_write(usartb, (uint16_t) *s++);
}

uint16_t usart3_bare_read() {
	return USART_ReceiveData(USART3 );
}

uint16_t usart_read(USARTBuffer * usartb) {
	if ( (usartb->rx_head != usartb->rx_tail) || usartb->rx_count > 0) {
		uint16_t c = usartb->rx_buf[usartb->rx_tail++];
		usartb->rx_tail %= USART_BUFFER_SIZE;
		usartb->rx_count--;
		return c;
	} else
		return 0; // buffer is empty
}
/*
uint8_t usart3_rxne() {
	return USART_GetFlagStatus(USART3, USART_FLAG_RXNE ) == SET;
}
*/

uint8_t usart_available(USARTBuffer * usartb) {
	return usartb->rx_count;
}

// this is the interrupt request handler (IRQ) for ALL USART3 interrupts

void USART3_IRQHandler(void) {
	if (USART_GetITStatus(USART3, USART_IT_RXNE )) {
		usart3.rx_buf[usart3.rx_head++] = USART_ReceiveData(USART3 );
		usart3.rx_head %= USART_BUFFER_SIZE;
		usart3.rx_count++;
	}

	if (USART_GetITStatus(USART3, USART_IT_TXE )) {
		if (usart3.tx_count == 0) {
			USART_ITConfig(USART3, USART_IT_TXE, (FunctionalState) DISABLE);
			USART_ClearITPendingBit(USART3, USART_IT_TXE );
		} else {
			USART_SendData(USART3, usart3.tx_buf[usart3.tx_tail++]);
			usart3.tx_tail %= USART_BUFFER_SIZE;
			usart3.tx_count--;
		}
	}
}
