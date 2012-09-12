/**
  ******************************************************************************
  * @file    tim3_encoder_interface/main.c
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
#include "main.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHello Cortex-M3/STM32 World!\r\n"
                            "Turn rotary encoder clock wise (CW) or counter clock wise (CCW).\r\n"
                            "CW increments the counter and CCW decrements the counter.\r\n"
                            "I tell you the turn direction and the counter value. Or type anything to request value.\r\n";
int8_t Direction, RxData;
uint8_t Counter = 100;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void TIM3_Configuration(void);
void Turn_CW(void);
void Turn_CCW(void);
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
  /* NVIC Configuration */
  NVIC_Configuration();

  /* TIM3 Configuration*/
  TIM3_Configuration();

  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  while (1)
    {
      if(RX_BUFFER_IS_NOT_EMPTY)
        {
          RxData = (int8_t)RECEIVE_DATA;
          cprintf("Value requested: Current counter value is: ");
          cprintf("%u\r\n", Counter);
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
  /* Enable the TIM3 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configure TIM3
  * @param  None
  * @retval : None
  */
void TIM3_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_OCInitTypeDef  TIM_OCInitStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC , ENABLE);
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(TIM3_CH12_GPIO_RCC , ENABLE);

  /* Configure GPIO for TIM3_CH1/2*/
  GPIO_InitStructure.GPIO_Pin = TIM3_CH1_PIN | TIM3_CH2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPD;
  GPIO_Init(TIM3_CH12_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM3)
PartialRemap_TIM3_Configuration();
#elif defined (FULL_REMAP_TIM3)
FullRemap_TIM3_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM3 Configuration: Encoder Interface Mode:
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 1;
  TIM_TimeBaseStructure.TIM_Prescaler = 0;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  TIM_EncoderInterfaceConfig (TIM3, TIM_EncoderMode_TI12, TIM_ICPolarity_Rising, TIM_ICPolarity_Rising);

  /* Output Compare Inactive Mode configuration: Channel3 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Timing;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Disable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC3Init(TIM3, &TIM_OCInitStructure);
  TIM_OC3PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* Output Compare Toggle Mode configuration: Channel4 */
  TIM_OCInitStructure.TIM_Pulse = 1;
  TIM_OC4Init(TIM3, &TIM_OCInitStructure);
  TIM_OC4PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);

  /* TIM IT enable */
  TIM_ITConfig(TIM3, TIM_IT_CC3 | TIM_IT_CC4 , ENABLE);

}

/**
  * @brief  Describes the action when encoder is turned
  * @param  None
  * @retval : None
  */
void Encoder_Turned(void)
{
  if (GPIO_ReadInputDataBit(TIM3_PORT, TIM3_CH1_PIN) == 0 &&
      GPIO_ReadInputDataBit(TIM3_PORT, TIM3_CH2_PIN) == 0)
    {
      if(TIM3->CR1 & 0b0010000)
        {
          Turn_CCW();
        }
      else
        {
          Turn_CW();
        }
    }
}

/**
  * @brief  Action when encoder turned CW
  * @param  None
  * @retval : None
  */
void Turn_CW(void)
{
  Counter++;
  cprintf("Turned CW!  Counter value is : ");
  cprintf("%u\r\n", Counter);
}

/**
  * @brief  Action when encoder turned CCW
  * @param  None
  * @retval : None
  */
void Turn_CCW(void)
{
  Counter--;
  cprintf("Turned CCW! Counter value is : ");
  cprintf("%u\r\n", Counter);
}


