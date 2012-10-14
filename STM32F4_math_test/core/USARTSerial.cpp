/*
 * USARTSerial.cpp
 *
 *  Created on: 2012/10/10
 *      Author: sin
 */

#include "USARTSerial.h"

/* This funcion initializes the USART1 peripheral
 *
 * Arguments: baudrate --> the baudrate at which the USART is
 * 						   supposed to operate
 */
void USARTSerial::begin(uint32 baud) {
//	GPIO_InitTypeDef GPIO_InitStruct; // this is for the GPIO pins used as TX and RX
	USART_InitTypeDef USART_InitStruct; // this is for the USART1 initilization
//	NVIC_InitTypeDef NVIC_InitStructure; // this is used to configure the NVIC (nested vector interrupt controller)

	switch (3) {
	/*
	case 1:
		USARTx = USART1;

		digitalWrite(PD13, HIGH);
		digitalWrite(PD15, HIGH);
		delay(500);

		RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
		//RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
		// PB6 TX, PB7 RX

		GPIOMode(GPIOB,
				(GPIO_Pin_6 | GPIO_Pin_7 ), GPIO_Mode_AF, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_UP)
				;

		GPIO_PinAFConfig(GPIOB, GPIO_PinSource6, GPIO_AF_USART1 ); //
		GPIO_PinAFConfig(GPIOB, GPIO_PinSource7, GPIO_AF_USART1 ); //

		USART_InitStruct.USART_BaudRate = baud;	// the baudrate is set to the value we passed into this init function
		USART_InitStruct.USART_WordLength = USART_WordLength_8b;// we want the data frame size to be 8 bits (standard)
		USART_InitStruct.USART_StopBits = USART_StopBits_1;	// we want 1 stop bit (standard)
		USART_InitStruct.USART_Parity = USART_Parity_No;// we don't want a parity bit (standard)
		USART_InitStruct.USART_HardwareFlowControl =
				USART_HardwareFlowControl_None; // we don't want flow control (standard)
		USART_InitStruct.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // we want to enable the transmitter and the receiver
		USART_Init(USART1, &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting

		USART_Cmd(USART1, ENABLE);

		digitalWrite(PD13, LOW);
		digitalWrite(PD15, LOW);

		break;

	case 2:
		USARTx = USART2;

		digitalWrite(PD14, HIGH);
		digitalWrite(PD15, HIGH);
		delay(1000);
		digitalWrite(PD14, LOW);
		digitalWrite(PD15, LOW);

		// Enable GPIOA clock for USARTs.
		RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE); //rcc_peripheral_enable_clock(&RCC_AHB1ENR, RCC_AHB1ENR_IOPAEN);
		// Enable clocks for USART2.
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE); //rcc_peripheral_enable_clock(&RCC_APB1ENR, RCC_APB1ENR_USART2EN);
		// Setup GPIO pins for USART2 transmit.
		GPIOMode(GPIOA, GPIO_Pin_2 | GPIO_Pin_3, GPIO_Mode_AF, GPIO_Speed_50MHz,
				GPIO_OType_PP, GPIO_PuPd_UP);
		//ALTFUNC, CLK_FAST, NOPULL);	// gpio_mode_setup(GPIOA, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO2);
		// Setup USART2 TX pin as alternate function.
		GPIO_PinAFConfig(GPIOA, GPIO_PinSource2, GPIO_AF_USART2 ); // PA2 TX
		GPIO_PinAFConfig(GPIOA, GPIO_PinSource3, GPIO_AF_USART2 ); // PA3 RX
		//gpio_set_af(GPIOA, GPIO_AF7, GPIO2);

		// Setup USART2 parameters.
		USART_InitStruct.USART_BaudRate = baud;	// the baudrate is set to the value we passed into this init function
		USART_InitStruct.USART_WordLength = USART_WordLength_8b;// we want the data frame size to be 8 bits (standard)
		USART_InitStruct.USART_StopBits = USART_StopBits_1;	// we want 1 stop bit (standard)
		USART_InitStruct.USART_Parity = USART_Parity_No;// we don't want a parity bit (standard)
		USART_InitStruct.USART_HardwareFlowControl =
				USART_HardwareFlowControl_None; // we don't want flow control (standard)
		USART_InitStruct.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // we want to enable the transmitter and the receiver

		// Finally enable the USART.
		USART_Init(USART2, &USART_InitStruct);
		//usart_enable(USART2);
		USART_Cmd(USART2, ENABLE);
		break;
*/
	case 3:
		USARTx = USART3;

		digitalWrite(PD12, HIGH);
		digitalWrite(PD15, HIGH);
		delay(500);
		digitalWrite(PD12, LOW);
		delay(250);
		digitalWrite(PD12, HIGH);
		delay(500);

		/* USART3 clock enable */
		RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, (FunctionalState)ENABLE);
		RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, (FunctionalState)ENABLE);

		GPIOMode(GPIOB, GPIO_Pin_10 | GPIO_Pin_11, GPIO_Mode_AF,
				GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_UP);

		digitalWrite(PD12, LOW);
		digitalWrite(PD15, LOW);

		GPIO_PinAFConfig(GPIOB, GPIO_PinSource10, GPIO_AF_USART3 ); // TX -- PB10
		GPIO_PinAFConfig(GPIOB, GPIO_PinSource11, GPIO_AF_USART3 ); // RX -- PB11

		USART_InitStruct.USART_BaudRate = baud;	// the baudrate is set to the value we passed into this init function
		USART_InitStruct.USART_WordLength = USART_WordLength_8b;// we want the data frame size to be 8 bits (standard)
		USART_InitStruct.USART_StopBits = USART_StopBits_1;	// we want 1 stop bit (standard)
		USART_InitStruct.USART_Parity = USART_Parity_No;// we don't want a parity bit (standard)
		USART_InitStruct.USART_HardwareFlowControl =
				USART_HardwareFlowControl_None; // we don't want flow control (standard)
		USART_InitStruct.USART_Mode = USART_Mode_Tx | USART_Mode_Rx; // we want to enable the transmitter and the receiver

		USART_Init(USART3, &USART_InitStruct); // again all the properties are passed to the USART_Init function which takes care of all the bit setting
		/*
		 USART_ITConfig(USART3, USART_IT_RXNE, ENABLE); // enable the USART1 receive interrupt

		 NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
		 // we want to configure the USART1 interrupts
		 NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;// this sets the priority group of the USART1 interrupts
		 NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;// this sets the subpriority inside the group
		 NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;	// the USART1 interrupts are globally enabled
		 NVIC_Init(&NVIC_InitStructure);	// the properties are passed to the NVIC_Init function which takes care of the low level stuff
		 */

		// finally this enables the complete USART1 peripheral
		USART_Cmd(USART3, (FunctionalState)ENABLE);

		break;
	default:
		digitalWrite(PD12, HIGH);
		digitalWrite(PD13, HIGH);
		digitalWrite(PD14, HIGH);
		digitalWrite(PD15, HIGH);
		delay(1000);
		digitalWrite(PD12, LOW);
		digitalWrite(PD13, LOW);
		digitalWrite(PD14, LOW);
		digitalWrite(PD15, LOW);
		break;
	}
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

size_t USARTSerial::write(uint8 ch) {
	while (!(USARTx->SR & 0x00000040))
		;
	USART_SendData((USART_TypeDef *) USARTx, (uint16) ch);
	/* Loop until the end of transmission */
	while (USART_GetFlagStatus((USART_TypeDef *) USARTx, USART_FLAG_TC )
			== RESET) {
	}
	return 1;
}

size_t USARTSerial::write(const uint8 * a, const uint16 len) {
	size_t n = 0;
	while ( n++ < len ) {
		write(*a++);
	}
	return n;
}

size_t USARTSerial::print(const char * s) {
	size_t n = 0;
	while ( *s ) {
		write(*s++);
		n++;
	}
	return n;
}

size_t USARTSerial::printNumber(uint32 num, uint8 base) {
	uint32 divider = (base == DEC ? 1000000000 :
			(base == HEX ? 0x10000000 :
					(base == BIN ? 0b1000000000000000000000000000000 : 040000000)));
	int digit;
	uint8 count = 0;
	boolean msdappeared = false;
	while ( divider > 0 ) {
		digit = num / divider % base;
		if ( digit != 0 )
			msdappeared = true;
		if ( msdappeared || divider == base ) {
			print((char)('0'+ (digit <= 9 ? digit : digit + 7) ));
			count++;
		}
		divider /= base;
	}
	return count;
}

size_t USARTSerial::print(int32 num) {
	size_t count = 0;
	if ( num < 0 ) {
		count = print('-');
	}
	count += printNumber(abs(num), DEC);
	return count;
}

size_t USARTSerial::printFloat(double fp, uint8 prec) {
	size_t count = 0;
	if ( fp < 0 ) {
		print('-');
		count++;
	}

	fp = abs(fp);
	count += printNumber(uint32(fp), DEC);
	fp -= uint32(fp);
	if ( uint32(fp) == 0 )
		return count;
	count += print('.');
	return count;
	for(int i = 0; i < prec; i++) {
		fp *= 10;
		count += printNumber( uint32(fp), DEC );
	}

	return count;
}


/**
 * @brief  Waits for then gets a char from the USART.
 * @param  none
 * @retval char
 */
int USARTSerial::getch() {
	int ch;
	while (USART_GetFlagStatus((USART_TypeDef *) USARTx, USART_FLAG_RXNE )
			== RESET) {
	}
	ch = USART_ReceiveData((USART_TypeDef *) USARTx);
	//uartPutch(ch);
	return ch;
}

