/**
  ******************************************************************************
  * @file    i2c_st7032i_test/main.c
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
#include <stm32f4xx.h>
#include <stm32f4xx_i2c.h>
#include <stm32f4xx_rcc.h>
//#include "platform_config.h"
//#include "com_config.h"
#include "usart.h"
#include "delay.h"
#include "i2c.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
//#define I2C_CLOCK                  100000
//#define ST7032I_ADDR               0b0111110
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nWelcomet to the Cortex-M3/STM32 World!\r\n";


/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{

  // Configure board specific setting
//  BoardInit();
  // Setting up COM port for Print function
//  COM_Configuration();
	usart_begin(USART3, 19200);

  //Send welcome messages
  usart_print(USART3, (char *) Welcome_Message);

  delay_ms(20);

  I2C_Configuration();

  //Function Set
  ST7032i_Command_Write( 0x38 ); //0b00111000 );

  delay_us(27);

  //Function Set
  ST7032i_Command_Write( 0x39 ); //0b00111001);

  delay_us(27);

  //Bias and OSC frequency
  ST7032i_Command_Write( 0x14 ); //0b00010100);

  delay_us(27);

  //Contrast set
  ST7032i_Command_Write( 0x70 ); //0b01110000);

  delay_us(27);

  //Power/Icon/Contrast control
  ST7032i_Command_Write( 0x56 ); //0b01010110);

  delay_us(27);

  //Follower control
  ST7032i_Command_Write( 0x6c ); //0b01101100);

  delay_ms(200);

  //Function Set
  ST7032i_Command_Write( 0x38 ); //0b00111000);

  //Display control : on
  ST7032i_Command_Write( 0x0c ); //0b00001100);

  delay_us(27);

  //Clear
  ST7032i_Command_Write( 0x01 );//0b00000001);

  delay_ms(2);

  ST7032i_Data_Write('H');
  ST7032i_Data_Write('e');
  ST7032i_Data_Write('l');
  ST7032i_Data_Write('l');
  ST7032i_Data_Write('o');

  ST7032i_Command_Write( /* 0b10000000 */ 0x80 | 0x40);

  ST7032i_Data_Write('W');
  ST7032i_Data_Write('o');
  ST7032i_Data_Write('r');
  ST7032i_Data_Write('l');
  ST7032i_Data_Write('d');
  ST7032i_Data_Write('!');

  usart_print(USART3, "Done! Confirm a message is on LCD.");

  while(1){}
}

