/**
  ******************************************************************************
  * @file    systick_ob_led_toggle_by_interrupt_only/main.c
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
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
static __IO bool LEDFlag = FALSE;

/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void Toggle_LED(void);
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
  /* System Clocks Configuration **********************************************/
  GPIO_Configuration();

  /* Setup SysTick Timer for 1000 msec interrupts  */
  //SystemCoreClock is defined insystem_stm32f10x.c
  //by default SystemCoreClock is 72,000,000 and is to be divided by 8(SysTick_CLKSourceConfig).
  //System clock frequency is 9MHz. By counting, 9,000,000(SystemCoreClock/8), 1000 ms passes.
  if (SysTick_Config(SystemCoreClock/8))
  {
    /* Capture error */
    while (1);
  }
  //Set system clock be divided by 8
  SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8);

  while (1);

}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void GPIO_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(OB_LED_GPIO_RCC | OB_SW_GPIO_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure OB_LED: output push-pull */
  GPIO_InitStructure.GPIO_Pin = OB_LED_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(OB_LED_PORT, &GPIO_InitStructure);

  /* Configure OB_SW: input floating */
  GPIO_InitStructure.GPIO_Pin = OB_SW_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(OB_SW_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Decrements the TimingDelay variable.
  * @param  None
  * @retval : None
  */
void Toggle_LED(void)
{
  if (LEDFlag == FALSE)
    {
      /* Turn on LED */
      GPIO_SetBits(OB_LED_PORT, OB_LED_PIN);
      LEDFlag = TRUE;
    }
  else
    {
      /* Turn off LED */
      GPIO_ResetBits(OB_LED_PORT, OB_LED_PIN);
      LEDFlag = FALSE;
    }
}

