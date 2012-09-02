/**
  ******************************************************************************
  * @file    tim1_external_triger_5_count/main.c
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
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define UPDATE 4
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
int8_t RxData;
const int8_t Welcome_Message1[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";
const int8_t Welcome_Message2[] =
  "I send you a message when you push a switch connected to TI2.\r\n"
  "I decrement 1 from the counter value of TIM1 when you push space bar.\r\n"
  "I tell you the current counter value of TIM1 when you type the other key.\r\n";
/* Private function prototypes -----------------------------------------------*/
void RCC_Configuration(void);
void NVIC_Configuration(void);
void TIM1_Configuration(void);
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
  /* System Clocks re-configuration **********************************************/
  RCC_Configuration();
  /* NVIC Configuration */
  NVIC_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();
  /* TIM1 Configuration*/
  TIM1_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message1);
  cprintf(Welcome_Message2);

  while (1)
    {
      if(RX_BUFFER_IS_NOT_EMPTY)
        {
          RxData = (int8_t)RECEIVE_DATA;
          if (RxData == ' ')
            {
              TIM_SetCounter(TIM1, TIM_GetCounter(TIM1)-1);
              cprintf("Decrement Requested: The current value of TIM1 is; \r\n");
              cprintf("%u\r\n", TIM_GetCounter(TIM1));
            }
          else
            {
              cprintf("Report Requested: The current value of TIM1 is; \r\n");
              cprintf("%u\r\n", TIM_GetCounter(TIM1));
            }
         }
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
  /* Enable the TIM1 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM1_UP_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configures the different system clocks.
  * @param  None
  * @retval None
  */
void RCC_Configuration(void)
{
  /* HCLK = SYSCLK/4 */
  RCC_HCLKConfig(RCC_SYSCLK_Div4);

  /* PCLK1 = HCLK/8 */
  RCC_PCLK1Config(RCC_HCLK_Div8);
}

/**
  * @brief  Send message via serial
  * @param  None
  * @retval None
  */
void Notify(void)
{
  cprintf("Switch is pressed for 5 times.\r\n");
}

/**
  * @brief  Configure TIM1
  * @param  None
  * @retval : None
  */
void TIM1_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;

  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM1_RCC | TIM1_GPIO_RCC , ENABLE);

  /* GPIO Configuration:TIM1 Channel as alternate function input floating */
  GPIO_InitStructure.GPIO_Pin = TIM1_CH2_PIN ;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPD;
  GPIO_Init(TIM1_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM1)
PartialRemap_TIM1_Configuration();
#elif defined (FULL_REMAP_TIM1)
FullRemap_TIM1_Configuration();
#endif

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = UPDATE;
  TIM_TimeBaseStructure.TIM_Prescaler = 0;
  TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV4;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);

  //Select TI2 as trigger for external clock mode 1
  TIM_TIxExternalClockConfig(TIM1, TIM_TIxExternalCLK1Source_TI2, TIM_ICPolarity_Rising, 0xF);

  /* Clear TIM1 update interrupt pending bit*/
  TIM_ClearITPendingBit(TIM1, TIM_IT_Update);

  /* TIM IT enable */
  TIM_ITConfig(TIM1, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM1, ENABLE);
}
