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

#include "usart.h"

enum USARTPortNumber {
	USART_1 = 0,
	USART_2,
	USART_3,
	UART_4,
	UART_5,
	USART_6
};

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
static USART_TypeDef * const USARTPort[6] =
{
  USART1,
  USART2,
  USART3,
  UART4,
  UART5,
  USART6,
};
*/

uint8_t usart_id(USART_TypeDef * USARTx) {
	if ( USARTx == USART1 )
			return USART_1;
	if ( USARTx == USART2 )
			return USART_2;
	if ( USARTx == USART3 )
			return USART_3;
	if ( USARTx == USART6 )
			return USART_6;
	return 0;
}

struct {
//	USART_TypeDef * port;
	GPIOPin_Type rxpin, txpin;
	uint32_t usart_periph;
	GPIO_TypeDef * afgpio;
	uint8_t rxsource, txsource;
	uint8_t afmapping;
	IRQn_Type  irq_channel;
} uPortInfo[] = {
		{ PA10, PA9,
				((uint32_t) RCC_APB2Periph_USART1), GPIOA, GPIO_PinSource10, GPIO_PinSource9, GPIO_AF_USART1,
				USART1_IRQn
		},
		{ PA3, PA2,
	    	  ((uint32_t) RCC_APB1Periph_USART2), GPIOA, GPIO_PinSource3, GPIO_PinSource2, GPIO_AF_USART2,
	    			  USART2_IRQn
		},
		{ PB11, PB10,
				((uint32_t) RCC_APB1Periph_USART3), GPIOB, GPIO_PinSource11, GPIO_PinSource10, GPIO_AF_USART3,
				USART3_IRQn
		},
		{ PA1, PA0,
				((uint32_t) RCC_APB1Periph_USART3), GPIOA, GPIO_PinSource1, GPIO_PinSource0, GPIO_AF_USART3,
				USART3_IRQn
		}
};

void usart_begin(USART_TypeDef * USARTx, const uint32_t baud) {
	//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

	uint8_t portid = usart_id(USARTx);
	if ( portid == 6 )
		while(1);

	//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, (FunctionalState) ENABLE);
	GPIOMode(uPortInfo[portid].rxpin | uPortInfo[portid].txpin, GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP,
			GPIO_PuPd_UP);
	/* USART3 clock enable */
	RCC_APB1PeriphClockCmd(uPortInfo[portid].usart_periph, (FunctionalState) ENABLE);

	GPIO_PinAFConfig(uPortInfo[portid].afgpio, uPortInfo[portid].txsource, uPortInfo[portid].afmapping ); // TX -- PB10
	GPIO_PinAFConfig(uPortInfo[portid].afgpio, uPortInfo[portid].rxsource,  uPortInfo[portid].afmapping ); // RX -- PB11

	USART_InitStruct.USART_BaudRate = baud;	// the baudrate is set to the value we passed into this init function
	USART_InitStruct.USART_WordLength = USART_WordLength_8b;// we want the data frame size to be 8 bits (standard)
	USART_InitStruct.USART_StopBits = USART_StopBits_1;	// we want 1 stop bit (standard)
	USART_InitStruct.USART_Parity = USART_Parity_No;// we don't want a parity bit (standard)
	USART_InitStruct.USART_HardwareFlowControl = USART_HardwareFlowControl_None; // we don't want flow control (standard)
	USART_InitStruct.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // we want to enable the transmitter and the receiver

	USART_Init(USARTx, &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting

	USART_ITConfig(USARTx, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);

	NVIC_InitStructure.NVIC_IRQChannel = uPortInfo[portid].irq_channel;
	// we want to configure the USART3 interrupts
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0; // this sets the priority group of the USART3 interrupts
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0; // this sets the subpriority inside the group
	NVIC_InitStructure.NVIC_IRQChannelCmd = (FunctionalState) ENABLE;	// the USART3 interrupts are globally enabled
	NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff

	buffer_clear(&rxring[portid]);
	buffer_clear(&txring[portid]);

	// finally this enables the complete USART3 peripheral
	USART_Cmd(USARTx, (FunctionalState) ENABLE);
}

void usart_bare_write(USART_TypeDef * USARTx, const uint16_t w) {
	while (USART_GetFlagStatus(USARTx, USART_FLAG_TXE ) == RESET)
		;
	USART_SendData(USARTx, w);
//	while (USART_GetFlagStatus(USART3, USART_FLAG_TC ) == RESET);
}

void usart_write(USART_TypeDef * USARTx, const uint16_t w) {
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);
	buffer_enque(&txring[USART_3], w);
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
	uint16_t w = buffer_deque(&rxring[USART_3]);
	if ( w == 0xffff ) return 0; // buffer is empty
	return w;
}

void usart_flush(USART_TypeDef * USARTx) {
	USART_ITConfig(USARTx, USART_IT_RXNE, (FunctionalState) DISABLE); // enable the USART3 receive interrupt
	buffer_clear(&rxring[USART_3]);
	USART_ClearITPendingBit(USARTx, USART_IT_RXNE );
	USART_ITConfig(USARTx, USART_IT_RXNE, (FunctionalState) ENABLE); // enable the USART3 receive interrupt
	USART_ITConfig(USARTx, USART_IT_TXE, (FunctionalState) DISABLE);
	while ( buffer_count(&txring[USART_3]) > 0 ) {
		while (USART_GetFlagStatus(USARTx, USART_FLAG_TXE ) == RESET);
		USART_SendData(USARTx, buffer_deque(&txring[USART_3]));
		while (USART_GetFlagStatus(USARTx, USART_FLAG_TC ) == RESET);
	}
	USART_ClearITPendingBit(USARTx, USART_IT_TXE );
	buffer_clear(&txring[USART_3]);
}

uint16_t usart_peek(USART_TypeDef * uport) {
	if ( ! buffer_count(&rxring[USART_3]) == 0 )
		return rxring[USART_3].buf[rxring[USART_3].tail];
	return 0xffff;
}

uint16_t usart_available(USART_TypeDef * uport) {
	return buffer_count(&rxring[USART_3]);
}

// this is the interrupt request handler (IRQ) for ALL USART3 interrupts

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
