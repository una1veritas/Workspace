/**
  ******************************************************************************
  * @file    usart_xbee_test_terminal/main.c
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
#include <string.h>
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define TX_POINTER_BUFFER 16
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "XBee test terminal.\r\nThis terminal is connected to USART1 or 2.\r\n";
int8_t *TxData[TX_POINTER_BUFFER];
uint8_t subscript_in_progress = 0, subscript_to_add = 0;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void XBee_Port_Configuration(uint32_t baudrate);
void FullRemap_XBee_Port_Configuration(void);
void XBee_Port_Send_String(const int8_t String[]);
void XBee_Port_DMA_Configuration(uint32_t Memory_Address, uint16_t Buffer_Size);
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

  // Do reset XBee
  GPIO_Configuration();
  GPIO_ResetBits(GPIOY_0_PORT,GPIOY_0_PIN);
  delay_ms(100);
  GPIO_SetBits(GPIOY_0_PORT,GPIOY_0_PIN);

  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  // Wait XBee module to wake up
  delay_ms(1500);

  // set USART3 speed to XBee default baudrate(9600bps)
  XBee_Port_Configuration(9600);
  // Enter into command mode
  XBee_Port_Send_String("+++");
  cprintf("+++ is send.\r\n");
  // 1000 ms is required for guard time and additional 500 ms is for getting response
  delay_ms(1500);
  XBee_Port_Send_String("ATCH15,ID7777,DH0,DL1,MY2,BD7,CN\r");
  cprintf("\r\nATCH15,ID7777,DH0,DL1,MY2,BD7,CN<cr> is send.\r\n");
  // 500 ms is enough for single command but 1500 ms is required baudrate change
  delay_ms(1500);
  XBee_Port_Configuration(115200);
  XBee_Port_Send_String("+++");
  cprintf("\r\n+++ is send.\r\n");
  delay_ms(1500);
  // Getting version number to confirm communication is established.
  XBee_Port_Send_String("ATVR,CN\r");
  cprintf("\r\nATVR,CN<cr> is send.\r\n");
  delay_ms(500);
  cprintf("\r\nXBee configuration complete.\r\n\r\n");
  XBee_Port_Send_String("\r\nAn message will be send from STM32+XBee consecutively.\r\n\r\n");
  while(1)
    {
      XBee_Port_Send_String("I am alive...   ");
      delay_ms(2000);
    }
}

/**
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable DMA1 channel7 IRQ Channel */
  NVIC_InitStructure.NVIC_IRQChannel = DMA1_Channel2_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

  /* Enable the USART3 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = USART3_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
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

  /* Configure GPIOY_0: output push-pull */
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the USART3.
  * @param  None
  * @retval : None
  */
void XBee_Port_Configuration(uint32_t baudrate)
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
  /* Configure USART3 Rx as input floating */
  GPIO_InitStructure.GPIO_Pin = USART3_RX_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(USART3_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_USART3)
PartialRemap_USART3_Configuration();
#elif defined (FULL_REMAP_USART3)
FullRemap_USART3_Configuration();
#endif

  /* Disable the USART3 */
  USART_Cmd(USART3, DISABLE);

  /* USART3 configuration ------------------------------------------------------*/
    /* USART3 configured as follow:
        - BaudRate = set by argument
        - Word Length = 8 Bits
        - One Stop Bit
        - No parity
        - Hardware flow control disabled (RTS and CTS signals)
        - Receive and transmit enabled
    */
  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = baudrate;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
  USART_Init(USART3, &USART_InitStructure);

  /* Enable USART3 DMA TX request */
  USART_DMACmd(USART3, USART_DMAReq_Tx, ENABLE);

  //Enable USART3 RX empty interrupt
  USART_ITConfig(USART3, USART_IT_RXNE, ENABLE);

  /* Enable the USART3 */
  USART_Cmd(USART3, ENABLE);
}

/**
  * @brief  Configures the DMA.
  * @param  None
  * @retval None
  */
void DMA_Configuration(uint32_t Memory_Address, uint16_t Buffer_Size)
{
  DMA_InitTypeDef DMA_InitStructure;

  /* DMA clock enable */
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);

  /* DMA channel for USART3 (DMA1_Channel2) Config */
  DMA_DeInit(DMA1_Channel2);
  DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&USART3->DR;
  DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)Memory_Address;
  DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralDST;
  DMA_InitStructure.DMA_BufferSize = Buffer_Size;
  DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
  DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
  DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
  DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
  DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
  DMA_InitStructure.DMA_Priority = DMA_Priority_VeryHigh;
  DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
  DMA_Init(DMA1_Channel2, &DMA_InitStructure);

  /* Enable DMA1 Channel2 Transfer Complete interrupt */
  DMA_ITConfig(DMA1_Channel2, DMA_IT_TC, ENABLE);
}

/**
  * @brief  Send Strings via USART2
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void XBee_Port_Send_String(const int8_t String[])
{
  uint8_t idle_flag = 0;

  // check the needs to enable and start DMA transfer
  if (subscript_in_progress == subscript_to_add) idle_flag = 1;

  //if subscript comes around and get to one in progress, then wait.
  if (subscript_in_progress == subscript_to_add + 1 || ( subscript_to_add == TX_POINTER_BUFFER - 1 && subscript_in_progress == 0) )
    {
      while(subscript_in_progress == subscript_to_add + 1 || ( subscript_to_add == TX_POINTER_BUFFER - 1 && subscript_in_progress == 0)){}
    }

  // set string pointer to buffer and increment subscript
  TxData[subscript_to_add++] = (int8_t *)String;

  //if subscript reaches end make to go back to front
  if (subscript_to_add == TX_POINTER_BUFFER)
    {
      subscript_to_add = 0;
    }

  // enable and start DMA transfer
  if (idle_flag != 0)
    {
      DMA_Configuration((uint32_t)TxData[subscript_in_progress], strlen(TxData[subscript_in_progress]));
      /* Enable USARTy DMA TX Channel */
      DMA_Cmd(DMA1_Channel2, ENABLE);
    }
}


/**
  * @brief  This function handles DMA1 Channel 2 interrupt request.
  * @param  None
  * @retval None
  */
void DMA1_Channel2_IRQHandler(void)
{
  /* Test on DMA1 Channel2 Transfer Complete interrupt */
  if(DMA_GetITStatus(DMA1_IT_TC2))
  {
    /* Disable USARTy DMA TX Channel */
    DMA_Cmd(DMA1_Channel2, DISABLE);

    // move to next string
    subscript_in_progress++;
    // if pointer reached the end of pointer buffer, make it go to front
    if (subscript_in_progress == TX_POINTER_BUFFER)
      {
        subscript_in_progress = 0;
      }

    // if pointer processing reaches to pointer the next message to be add
    // stops generate interrupt and stop sending.
    if (subscript_in_progress != subscript_to_add)
      {
        DMA_Configuration((uint32_t)TxData[subscript_in_progress], strlen(TxData[subscript_in_progress]));
        /* Enable USARTy DMA TX Channel */
        DMA_Cmd(DMA1_Channel2, ENABLE);
      }
    /* Clear DMA1 Channel2 Transfer Complete interrupt pending bits */
    DMA_ClearITPendingBit(DMA1_IT_TC2);
  }
}

/**
  * @brief  This function handles USART3 global interrupt request.
  * @param  None
  * @retval None
  */
void USART3_IRQHandler(void)
{
  if(USART_GetITStatus(USART3, USART_IT_RXNE) != RESET)
    {
      cputchar(USART_ReceiveData(USART3));
    }
}
