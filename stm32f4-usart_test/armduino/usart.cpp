/*
 * usart.c
 *
 *  Created on: 2012/10/14
 *      Author: sin
 */

#include <stm32f4xx.h>
#include <stm32f4xx_gpio.h>
#include <stm32f4xx_rcc.h>
#include <stm32f4xx_usart.h> // under Libraries/STM32F4xx_StdPeriph_Driver/inc and src

#include "usart.h"

void USARTSerial::begin(uint32_t baud) {
//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
//	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

//	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, (FunctionalState) ENABLE);
	GPIOMode(RCC_AHB1Periph_GPIOB, GPIOB, GPIO_Pin_10 | GPIO_Pin_11,
			GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_UP);
	/* USART3 clock enable */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, (FunctionalState) ENABLE);

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
	 USART_ITConfig(USART3, USART_IT_RXNE, ENABLE); // enable the USART1 receive interrupt
USART_IT	説明
USART_IT_PE	Parity Error interrupt
USART_IT_TXE	Transmit Data Register empty interrupt
USART_IT_TC	Transmission complete interrupt
USART_IT_RXNE	Receive Data register not empty interrupt
USART_IT_IDLE	Idle line detection interrupt
USART_IT_LBD	LIN break detection interrupt
USART_IT_CTS	CTS change interrupt (not available for UART4 and UART5)
USART_IT_ERR	Error interrupt (Frame error, noise error, overrun error)

NewState


NewState	説明
ENABLE	有効にします
DISABLE	無効にします

	 NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
	 // we want to configure the USART1 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;// this sets the priority group of the USART1 interrupts
	 NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;// this sets the subpriority inside the group
	 NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART1 interrupts are globally enabled
	 NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff
	 */

	// finally this enables the complete USART1 peripheral
	USART_Cmd(USART3, (FunctionalState) ENABLE);

}

/* This function is used to transmit a string of characters via
 * the USART specified in USARTx.
 *
 * It takes two arguments: USARTx --> can be any of the USARTs e.g. USART1, USART2 etc.
 * 						   (volatile) char *s is the string you want to send
 *
 * Note: The string has to be passed to the function as a pointer because
 * 		 the compiler doesn't know the 'string' data type. In standard
 * 		 C a string is just an array of characters
 *
 * Note 2: At the moment it takes a volatile char because the received_string variable
 * 		   declared as volatile char --> otherwise the compiler will spit out warnings
 * */

uint16_t USARTSerial::write(uint8_t ch) {
	while (!(USART3->SR & 0x00000040))
	;
	USART_SendData(USART3, (uint16_t) ch);
	/* Loop until the end of transmission */
	while (USART_GetFlagStatus(USART3, USART_FLAG_TC )
			== RESET) {
	}
	return 1;
}

uint16_t USARTSerial::write(uint8_t * p, uint16_t length) {
	uint16_t n = 0;
	while (n++ < length)
		write(*p++);
	return n;
}


uint16_t USARTSerial::print(const char * s) {
	uint16_t n = 0;
	while ( *s ) {
		write(*s);
		s++;
		n++;
	}
	return n;
}


uint16_t USARTSerial::printNumber(uint32_t val, const uint8_t base) {
	uint16_t n = 0;
	bool msd = false;
	uint32_t divider = ( base == 2 ? 1<<31 :
		( base == 16 ? 0x10000000L : 1000000000L ) );

	uint8_t digit;
	while ( divider > 0 ) {
		digit = (val / divider) % base;
		if ( digit || msd || (!msd && (divider%base)) ) {
			write('0' + ( digit > 9 ? digit + 7 : digit));
			msd = true;
			n++;
		}
		divider /= base;
	}
	return n;
}

uint16_t USARTSerial::printFloat(float val, uint8_t prec) {
	uint16_t n = 0;
	if ( val < 0 ) {
		write('-');
		val = -val;
		n++;
	}
	uint32_t intpart = (uint32_t) val;
	val -= intpart;
	n += printNumber(intpart, DEC);
	int i;
	if ( val > 0 ) {
		write('.');
		n++;
		for(i = 0; i < prec; i++) {
			val *= 10;
			printNumber((uint32_t)val, DEC);
			val -= (uint32_t)val;
			n++;
		}
	}
	return n;
}

/**
 * @brief  Waits for then gets a char from the USART.
 * @param  none
 * @retval char
 */
/*
int usart::usart_getch() {
	int ch;
	while (USART_GetFlagStatus(USART3, USART_FLAG_RXNE )
			== RESET) {
	}
	ch = USART_ReceiveData(USART3);
	//uartPutch(ch);
	return ch;
}
*/
