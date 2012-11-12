/**
 ******************************************************************************
 * @file    i2c_st7032i_lib/main.c
 * @author  Yasuo Kawachi
 * @version V1.0.0
 * @date    04/15/2009
 * @brief   Main program body
 ******************************************************************************
 * @copy
 *
 * Copyright 2008-2009 Yasuo Kawachi All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  1. Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY YASUO KAWACHI "AS IS" AND ANY EXPRESS OR IMPLIE  D
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL YASUO KAWACHI OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * This software way contain the part of STMicroelectronics firmware.
 * Below notice is applied to the part, but the above BSD license is not.
 *
 * THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
 * WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE
 * TIME. AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY
 * DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING
 * FROM THE CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE
 * CODING INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
 *
 * COPYRIGHT 2009 STMicroelectronics
 */

/* Includes ------------------------------------------------------------------*/
#include <ctype.h>

#include "stm32f4xx.h"
//#include "platform_config.h"
//#include "com_config.h"
#include "st7032i.h"

#include "armcore.h"
#include "systick.h"
#include "delay.h"
#include "usart.h"
#include "i2c.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/**
 * @brief  Main program.
 * @param  None
 * @retval : None
 */
int main(void) {
	int16_t i = 0;
	const int rxbufsize = 64;
	char rxbuf[rxbufsize];
	char printbuf[64];
	uint32_t tmp32, rtctime = 0;

	SysTick_Start();

	usart_begin(USART3Serial, PD9, PD8, 19200);
	usart_print(USART3Serial, "Welcome to my serial.\r\n");

	i2c_begin(100000);
	ST7032i_Init();

	ST7032i_Print_String((const int8_t *)"Welcome to lcd.");
	delay_ms(1000);
	ST7032i_Command_Write(0x01);
	delay_ms(10);

	//Receive character from COM and put it on LCD
	while (1) {
		if (usart_available(USART3Serial) > 0) {
			i = 0;
			while ( usart_available(USART3Serial) > 0 && i < rxbufsize )
				rxbuf[i++] = (char) usart_read(USART3Serial);
			rxbuf[i] = 0;
			for (i = 0; rxbuf[i]; i++) {
				if ( isprint(rxbuf[i]) ) {
					usart_write(USART3Serial, rxbuf[i]);
			//		ST7032i_Data_Write(rxbuf[i]);
				} else {
					sprintf(printbuf, "cmd %x\n", rxbuf[i]);
					usart_print(USART3Serial, printbuf);
			//		ST7032i_Command_Write(rxbuf[i]);
				}
			}
		} else {
			i2c_requestFrom(0b1101000, 0, (uint8_t *) &tmp32, 3);
			if ( rtctime != (tmp32 & 0xffffff) ) {
				rtctime = tmp32 & 0xffffff;
				sprintf(printbuf, "%02x:%02x:%02x", UINT8(rtctime>>16), UINT8(rtctime>>8), UINT8(rtctime));
				//usart_print(USART3Serial, printbuf);
			  	ST7032i_Set_DDRAM( ((0 * 0x40) % 0x6c) + 0);
				ST7032i_Print_String((int8_t *)printbuf);
				if ( (rtctime & 0xff) == 0 ) {
				i2c_requestFrom(0b1101000, 3, (uint8_t *) &tmp32, 4);
				sprintf(printbuf, "20%02x %02x/%02x (%x)", UINT8(tmp32>>24), UINT8(tmp32>>16), UINT8(tmp32>>8), UINT8(tmp32));
				//sprintf(printbuf, "cal %lx.", tmp32);
//				usart_print(USART3Serial, printbuf);
//				usart_print(USART3Serial, "\r\n");
			  	ST7032i_Set_DDRAM( ((1 * 0x40) % 0x6c) + 0);
				ST7032i_Print_String((int8_t *)printbuf);
				}
			}
		}
		delay_ms(100);
	}
}

