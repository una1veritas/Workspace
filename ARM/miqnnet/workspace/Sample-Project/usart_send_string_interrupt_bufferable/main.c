/**
  ******************************************************************************
  * @file    usart_send_string_interrupt_bufferable/main.c
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
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define TX_POINTER_BUFFER      32
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "Sending message through USART2 using TXE interrupt.\r\n\r\n";
const int8_t Message_A[] = "A: This is message A.\r\n";
const int8_t Message_B[] = "B: This is message B.\r\n";
const int8_t Message_C[] = "C: This is message C.\r\n";
const int8_t Message_D[] = "D: This is message D.\r\n";
const int8_t Message_E[] = "E: This is message E.\r\n";
uint8_t RxData;
int8_t *TxData[TX_POINTER_BUFFER];
uint8_t subscript_in_progress = 0, subscript_to_add = 0;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void USART2_Configuration(void);
void USART2_Send_String(const int8_t String[]);
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
  NVIC_Configuration();

  USART2_Configuration();

  //Send welcome messages
  USART2_Send_String(Welcome_Message);

  while (1)
    {
      //Send message successively
      USART2_Send_String(Message_A);
      USART2_Send_String(Message_B);
      USART2_Send_String(Message_C);
      USART2_Send_String(Message_D);
      USART2_Send_String(Message_E);
      USART2_Send_String("\r\n");
      //wait 1000ms
      delay_ms(1000);
    }
}

/**
  * @brief  Configures the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable the USARTy Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = USART2_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configure the USART2.
  * @param  None
  * @retval : None
  */
void USART2_Configuration(void)
{
  RCC_APB2PeriphClockCmd(USART2_GPIO_RCC, ENABLE);

  /* Supply APB1 clock */
  RCC_APB1PeriphClockCmd(USART2_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure USART2 Tx as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = USART2_TX_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_Init(USART2_PORT, &GPIO_InitStructure);

  /* Configure USART2 Rx as input floating */
  GPIO_InitStructure.GPIO_Pin = USART2_RX_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(USART2_PORT, &GPIO_InitStructure);

#if defined (REMAP_USART2)
Remap_USART2_Configuration();
#endif

  /* USART2 configuration ------------------------------------------------------*/
    /* USART2 configured as follow:
          - BaudRate = 115200 baud
          - Word Length = 8 Bits
          - One Stop Bit
          - No parity
          - Hardware flow control disables
          - Receive and transmit enabled
    */
  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = 115200;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
  USART_Init(USART2, &USART_InitStructure);
  /* Enable the USART2 */
  USART_Cmd(USART2, ENABLE);
}

/**
  * @brief  Send Strings via USART2
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void USART2_Send_String(const int8_t String[])
{
  TxData[subscript_to_add++] = (int8_t *)String;
  //if subscript reaches end make to go back to front
  if (subscript_to_add == TX_POINTER_BUFFER)
    {
      subscript_to_add = 0;
    }
  //Enable USART2 TX empty interrupt
  USART_ITConfig(USART2, USART_IT_TXE, ENABLE);
  //if subscript comes around and get to one in progress, then wait.
  if (subscript_to_add == subscript_in_progress)
    {
      while(subscript_to_add != subscript_in_progress){}
    }
}

/**
  * @brief  This function handles USARTy global interrupt request.
  * @param  None
  * @retval None
  */
void USART2_IRQHandler(void)
{
  if(USART_GetITStatus(USART2, USART_IT_TXE) != RESET)
    {
      // if pointer reaches the end of processing string
      if(*TxData[subscript_in_progress] == '\0')
        {
          // move to next string
          subscript_in_progress++;
          // if pointer reached the end of pointer buffer, make it go to front
          if (subscript_in_progress == TX_POINTER_BUFFER)
            {
              subscript_in_progress = 0;
            }
          // if pointer processing reaches to pointer the next message to be add
          // stops generate interrupt and stop sending.
          if (subscript_in_progress == subscript_to_add)
            {
              /* Disable the USARTy Transmit interrupt */
              USART_ITConfig(USART2, USART_IT_TXE, DISABLE);
            }
        }
      else
        {
          /* Write one byte to the transmit data register */
          USART_SendData(USART2, *TxData[subscript_in_progress]++);
        }
    }
}