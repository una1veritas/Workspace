/**
  ******************************************************************************
  * @file    pwr_sleep_wfi/main.c
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
#define SCB_SysCtrl   ((uint32_t)0xE000ED10)
#define SysCtrl_SLEEPDEEP_Set   ((uint32_t)0x00000004)
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Press switch connected to EXTI5 to enter SLEEP mode.\r\n";
int8_t wait_flag;
/* Private function prototypes -----------------------------------------------*/
void EXTI_Interrupt_Configuration(void);
void RCC_Configuration(void);
void TIM3_Configuration(void);
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
  // Setting up COM port for Print function
  COM_Configuration();

  /* TIM3 Configuration*/
  TIM3_Configuration();

  EXTI_Interrupt_Configuration();

  /* Enable PWR and BKP clock */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_PWR | RCC_APB1Periph_BKP, ENABLE);

  //Send welcome messages
  cprintf(Welcome_Message);

  while (1)
    {
      wait_flag = 1;
      while (wait_flag){}
      //wait until chattering on switch is end
      delay_ms(200);

      cprintf("Entering into SLEEP mode.\r\n");
      cprintf("You can recover by pressing it again.\r\r\n\n");
      delay_ms(50);
      TIM_SetAutoreload (TIM3, 499);

      // Reset SLEEPDEEP bit of Cortex System Control Register */
      *(__IO uint32_t *) SCB_SysCtrl &= !(SysCtrl_SLEEPDEEP_Set);
      // Request Wait For Interrupt
      __WFI();

      TIM_SetAutoreload(TIM3, 99);
      TIM_SetCounter(TIM3, 0);

      cprintf("Recovered from SLEEP mode.\r\r\n\n");
      //wait until chattering on switch is end
      delay_ms(200);
    }
}

/**
  * @brief  Configure the Onboard switch as EXTI interrupt
  * @param  None
  * @retval : None
  */
void EXTI_Interrupt_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  EXTI_InitTypeDef EXTI_InitStructure;
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable Button GPIO clock */
  RCC_APB2PeriphClockCmd(GPIOX_RCC | RCC_APB2Periph_AFIO, ENABLE);

  /* Configure Button pin as input floating */
  GPIO_InitStructure.GPIO_Pin = GPIOX_5_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Connect Button EXTI Line to Button GPIO Pin */
  GPIO_EXTILineConfig(GPIOX_PORTSOURCE , GPIOX_5_PINSOURCE);

  /* Configure Button EXTI line */
  EXTI_InitStructure.EXTI_Line = EXTI_Line5;
  EXTI_InitStructure.EXTI_Mode = EXTI_Mode_Interrupt;
  EXTI_InitStructure.EXTI_Trigger = EXTI_Trigger_Rising;
  EXTI_InitStructure.EXTI_LineCmd = ENABLE;
  EXTI_Init(&EXTI_InitStructure);

  /* Enable and set Button EXTI Interrupt*/
  NVIC_InitStructure.NVIC_IRQChannel = EXTI9_5_IRQn;
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
  /* PCLK1 = HCLK/4 */
  RCC_PCLK1Config(RCC_HCLK_Div4);
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

  /* GPIO Configuration:TIM3 Channel1, 2 and 3 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM3_CH1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(TIM3_CH12_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM3)
PartialRemap_TIM3_Configuration();
#elif defined (FULL_REMAP_TIM3)
FullRemap_TIM3_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM3 Configuration: Output Compare Toggle Mode:
    TIM3CLK = 36 MHz, Prescaler = 36000, TIM3 counter clock = 1kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 99;
  TIM_TimeBaseStructure.TIM_Prescaler = 35999;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Toggle;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
  TIM_OC1Init(TIM3, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);
}
