/**
  ******************************************************************************
  * @file    usart_synchronous_shift_register/main.c
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
#include "stm32f10x.h"
#include "platform_config.h"
#include "com_config.h"
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "You can find numeric numbers are on 7 segment LEDs.\r\n";
const uint8_t Digit[] =
  {0b00111111,0b00000110,0b01011011,0b01001111,0b01100110,
   0b01101101,0b01111101,0b00100111,0b01111111,0b01101111};
/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void USRT_Configuration(void);
void Show_Digits(uint8_t Digit3, uint8_t Digit2, uint8_t Digit1, uint8_t Digit0);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{

  // Configure board specific setting
  BoardInit();
  GPIO_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();

  USRT_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  while(1)
    {
      Show_Digits(Digit[0], Digit[1], Digit[2], Digit[3]);
      delay_ms(500);
      Show_Digits(Digit[4], Digit[5], Digit[6], Digit[7]);
      delay_ms(500);
      Show_Digits(Digit[8], Digit[9], 0, 255);
      delay_ms(500);
    }

}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIOY_0: output push-pull */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the USART3.
  * @param  None
  * @retval : None
  */
void USRT_Configuration(void)
{

  /* Supply APB1 clock */
  RCC_APB1PeriphClockCmd(USART3_RCC , ENABLE);
  RCC_APB2PeriphClockCmd(USART3_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;
  /* Configure USART3 Tx as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = USART3_TX_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_Init(USART3_PORT, &GPIO_InitStructure);
  /* Configure USART3 CK as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = USART3_CK_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_Init(USART3_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_USART3)
PartialRemap_USART3_Configuration();
#elif defined (FULL_REMAP_USART3)
FullRemap_USART3_Configuration();
#endif

  /* USART3 configuration ------------------------------------------------------*/
    /* USART3 configured as follow:
        - BaudRate = 1000000 baud
        - Word Length = 8 Bits
        - One Stop Bit
        - No parity
        - Hardware flow control disabled (RTS and CTS signals)
        - Receive and transmit enabled
        - USART Clock Enabled
        - USART CPOL: Clock is active Low
        - USART CPHA: Data is captured on the first edge
        - USART LastBit: The clock pulse of the last data bit is output to
                         the SCLK pin
    */
  USART_ClockInitTypeDef USART_ClockInitStructure;
  USART_ClockInitStructure.USART_Clock = USART_Clock_Enable;
  USART_ClockInitStructure.USART_CPOL = USART_CPOL_Low;
  USART_ClockInitStructure.USART_CPHA = USART_CPHA_1Edge;
  USART_ClockInitStructure.USART_LastBit = USART_LastBit_Enable;
  USART_ClockInit(USART3, &USART_ClockInitStructure);

  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = 1000000;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Tx;
  USART_Init(USART3, &USART_InitStructure);
  /* Enable the USART3 */
  USART_Cmd(USART3, ENABLE);
}

/**
  * @brief  Send Byte data via USART3
  * @param  Byte: Byte data to be send
  * @retval : None
  */
void Show_Digits(uint8_t Digit3, uint8_t Digit2, uint8_t Digit1, uint8_t Digit0)
{
  USART_SendData(USART3, Digit0);
  while(USART_GetFlagStatus(USART3, USART_FLAG_TC) == RESET);
  USART_SendData(USART3, Digit1);
  while(USART_GetFlagStatus(USART3, USART_FLAG_TC) == RESET);
  USART_SendData(USART3, Digit2);
  while(USART_GetFlagStatus(USART3, USART_FLAG_TC) == RESET);
  USART_SendData(USART3, Digit3);
  while(USART_GetFlagStatus(USART3, USART_FLAG_TC) == RESET);
  GPIO_SetBits  (GPIOY_0_PORT, GPIOY_0_PIN);
  delay_us(10);
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
  delay_us(10);
}
