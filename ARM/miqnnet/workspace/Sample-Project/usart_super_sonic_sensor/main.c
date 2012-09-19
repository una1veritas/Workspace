/**
  ******************************************************************************
  * @file    usart_super_sonic_sensor/main.c
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
#define RX_BUFFER 5
#define SAMPLE 5
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "Sending message through USART2 using DMA.\r\n\r\n";
int8_t RxData[RX_BUFFER + 1];
uint16_t DistSonic[SAMPLE];
uint16_t DistLast = 0;
uint8_t Sub_DistSonic = 0;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void DMA_USART1_Configuration(void);
void BubbleSort(uint16_t array[], uint8_t array_num);
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

  // NVIC configuration for DMA RX interrupt
  NVIC_Configuration();

  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  DMA_USART1_Configuration();

  while (1)
    {
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

  /* Enable DMA1 channel5 IRQ Channel */
  NVIC_InitStructure.NVIC_IRQChannel = DMA1_Channel5_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configures the DMA.
  * @param  None
  * @retval None
  */
void DMA_USART1_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART1_RCC | USART1_GPIO_RCC, ENABLE);
  /* DMA clock enable */
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);


  GPIO_InitTypeDef GPIO_InitStructure;
  /* Configure USART1 Tx as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = USART1_TX_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
  GPIO_Init(USART1_PORT, &GPIO_InitStructure);

  /* Configure USART1 Rx as input floating */
  GPIO_InitStructure.GPIO_Pin = USART1_RX_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(USART1_PORT, &GPIO_InitStructure);

  // keep sonic sensor reset until usart1 setup is done
  GPIO_ResetBits(USART1_PORT, USART1_TX_PIN);

#if defined (REMAP_USART1)
Remap_USART1_Configuration();
#endif

  /* DMA channel for USART1 (DMA1_Channel5) Config */
  DMA_DeInit(DMA1_Channel5);
  DMA_InitTypeDef DMA_InitStructure;
  DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&USART1->DR;
  DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)RxData;
  DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;
  DMA_InitStructure.DMA_BufferSize = 5;
  DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
  DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
  DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
  DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
  DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
  DMA_InitStructure.DMA_Priority = DMA_Priority_VeryHigh;
  DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
  DMA_Init(DMA1_Channel5, &DMA_InitStructure);
  /* Enable DMA1 Channel5 Transfer Complete interrupt */
  DMA_ITConfig(DMA1_Channel5, DMA_IT_TC, ENABLE);
  /* Enable USART1 DMA RX Channel */
  DMA_Cmd(DMA1_Channel5, ENABLE);

  /* USART1 configuration ------------------------------------------------------*/
    /* USART1 configured as follow:
          - BaudRate = 9600 baud
          - Word Length = 8 Bits
          - One Stop Bit
          - No parity
          - Hardware flow control disables
          - only receive and enabled
    */
  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = 9600;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Rx;
  USART_Init(USART1, &USART_InitStructure);

  /* Enable USART1 DMA TX request */
  USART_DMACmd(USART1, USART_DMAReq_Rx, ENABLE);

  /* Enable the USART1 */
  USART_Cmd(USART1, ENABLE);

  // wakeup sonic sensor
  GPIO_SetBits(USART1_PORT, USART1_TX_PIN);

}

/**
  * @brief  This function handles DMA1 Channel 5 interrupt request.
  * @param  None
  * @retval None
  */
void DMA1_Channel5_IRQHandler(void)
{
  /* Test on DMA1 Channel5 Transfer Complete interrupt */
  if(DMA_GetITStatus(DMA1_IT_TC5))
    {
    DistSonic [Sub_DistSonic] = (RxData[1] - 0x30) * 100 + (RxData[2] - 0x30) * 10 + (RxData[3] - 0x30);
    Sub_DistSonic++;
    if (Sub_DistSonic>SAMPLE-1)
      {
      Sub_DistSonic = 0;
      BubbleSort (DistSonic, SAMPLE);
      if (DistLast != DistSonic[SAMPLE/2])
        {
        cprintf("%dinches\r\n",DistSonic[SAMPLE/2]);
        DistLast = DistSonic[SAMPLE/2];
        }
      }

    /* Clear DMA1 Channel5 Transfer Complete interrupt pending bits */
    DMA_ClearITPendingBit(DMA1_IT_TC5);
    }
}

/**
  * @brief  sort array by bubble sort
  * @param  array: array to be sorted
  * @param  array_num:number of array elements
  * @retval None
  */
void BubbleSort(uint16_t array[], uint8_t array_num)
{
    int i, j, temp;

    for (i = 0; i < array_num - 1; i++) {
        for (j = array_num - 1; j > i; j--) {
            if (array[j - 1] > array[j])
              {
                temp = array[j];
                array[j] = array[j - 1];
                array[j - 1]= temp;
            }
        }
    }
}
