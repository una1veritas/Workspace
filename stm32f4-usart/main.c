/*
 * main.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stm32f4xx.h>

//#include "stm32f4xx_it.h"

#include "gpio.h"
#include "systick.h"
#include "delay5.h"
#include "delay.h"
#include "usart.h"

void NVIC_Configuration();

int main(void) {
	uint16_t count = 0;
	uint16_t bits;
	uint32_t dval = 900;
	uint32_t intval = 24;
	char tmp[92];

	NVIC_Configuration();

	delay5_start();
	SysTick_Start();

	usart_begin(USART2Serial, PA3, PA2, 19200);
	usart_print(USART2Serial, "Happy are those who know they are spiritually poor; \n");
	usart_print(USART2Serial, "The kingdom of heaven belongs to them!\n");
//	sprintf(tmp, "port = %d\n", Serial3.port());
//	usart3_print(tmp);
	usart_print(USART2Serial, "How many eyes does Mississipi river have?\n");
	usart_print(USART2Serial, "Quick brown fox jumped over the lazy dog!\n");
	usart_flush(USART2Serial);

	GPIOMode(GPIOD, (GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15),
			GPIO_Mode_OUT, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);
	/*
	pinMode(PD12, OUTPUT);
	pinMode(PD13, OUTPUT);
	pinMode(PD14, OUTPUT);
	pinMode(PD15, OUTPUT);
	| GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15),
			GPIO_Mode_OUT, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);
*/
	while (1) {
		bits = GPIO_ReadOutputData(GPIOD);
		bits = 1<<12 | (bits & 0x0fff);
		GPIO_Write(GPIOD, bits);
		SysTick_delay(intval);

		bits = 1<<13 | (bits & 0x0fff);
		GPIO_Write(GPIOD, bits);
		SysTick_delay(intval);

		bits = 1<<14 | (bits & 0x0fff);
		GPIO_Write(GPIOD, bits);
		SysTick_delay(intval);

		bits = 1<<15 | (bits & 0x0fff);
		GPIO_Write(GPIOD, bits);
		SysTick_delay(intval);
		//
		bits = 1<<12 | (bits & 0x0fff);
		GPIO_Write(GPIOD, bits);
		SysTick_delay(dval);
/*
		usart3.print((float)(count++ / 32.0f), 3);
		*/
		count++;
//		uint16_t h, t;
//		h = tx_head();
//		t = tx_tail();
		sprintf(tmp, /*"head =% 4d, tail =% 4d,*/ "%04ld\n", millis());
		usart_print(USART2Serial, tmp);
		/*
		dval = (uint32) (100.0f + 64*sinf( (count % (uint32)(3.14159 * 2 * 32))/32.0f));
		usart3.println(dval);
		*/
		uint16_t i = 0;
		if ( usart_available(USART2Serial) > 0 ) {
			while ( usart_available(USART2Serial) > 0 && i < 92 ) {
				tmp[i++] = (char) usart_read(USART2Serial);
			}
			tmp[i] = 0;
			usart_print(USART2Serial, "> ");
			usart_print(USART2Serial, tmp);
			usart_print(USART2Serial, "\n");
		}
	}
	return 0;
}

/**
  * @brief  Configures the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
    NVIC_InitTypeDef NVIC_InitStructure;
    /* Enable the TIM2 gloabal Interrupt */
    NVIC_InitStructure.NVIC_IRQChannel = TIM5_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;

    NVIC_Init(&NVIC_InitStructure);
}

