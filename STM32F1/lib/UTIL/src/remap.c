/**
  ******************************************************************************
  * @file    lib/UTIL/inc/remap.C
  * @author  Yasuo Kawachi
  * @version  V1.0.0
  * @date  04/15/2009
  * @brief  Remap functions
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
#include "remap.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Remap USART1
  * @param  None
  * @retval : None
  */
void Remap_USART1_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PB6-7 to USART1 */
  GPIO_PinRemapConfig(GPIO_Remap_USART1, ENABLE);
}

/**
  * @brief  Remap USART2
  * @param  None
  * @retval : None
  */
void Remap_USART2_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PD3-6 to USART2 */
  GPIO_PinRemapConfig(GPIO_Remap_USART2, ENABLE);
}

/**
  * @brief  Full remap USART3
  * @param  None
  * @retval : None
  */
void FullRemap_USART3_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PD8-11 to USART3 */
  GPIO_PinRemapConfig(GPIO_FullRemap_USART3 , ENABLE);
}

/**
  * @brief  Full remap USART3
  * @param  None
  * @retval : None
  */
void PartialRemap_USART3_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PD8-11 to USART3 */
  GPIO_PinRemapConfig(GPIO_PartialRemap_USART3 , ENABLE);
}

/**
  * @brief  Full remap TIM1
  * @param  None
  * @retval : None
  */
void FullRemap_TIM1_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_FullRemap_TIM1, ENABLE);
}

/**
  * @brief  Partial remap TIM1
  * @param  None
  * @retval : None
  */
void PartialRemap_TIM1_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_PartialRemap_TIM1, ENABLE);
}

/**
  * @brief  Full remap TIM2
  * @param  None
  * @retval : None
  */
void FullRemap_TIM2_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_FullRemap_TIM2, ENABLE);
}

/**
  * @brief  Partial remap 1 TIM2
  * @param  None
  * @retval : None
  */
void PartialRemap1_TIM2_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_PartialRemap1_TIM2, ENABLE);
}

/**
  * @brief  Partial remap 2 TIM2
  * @param  None
  * @retval : None
  */
void PartialRemap2_TIM2_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_PartialRemap2_TIM2, ENABLE);
}

/**
  * @brief  Partial remap TIM3
  * @param  None
  * @retval : None
  */
void FullRemap_TIM3_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_FullRemap_TIM3, ENABLE);
}

/**
  * @brief  Partial remap TIM3
  * @param  None
  * @retval : None
  */
void PartialRemap_TIM3_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_PartialRemap_TIM3, ENABLE);
}

/**
  * @brief  Remap TIM4
  * @param  None
  * @retval : None
  */
void Remap_TIM4_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_Remap_TIM4, ENABLE);
}

/**
  * @brief  Remap TIM5
  * @param  None
  * @retval : None
  */
void Remap_TIM5_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PE7-15 to TIM1 */
  GPIO_PinRemapConfig(GPIO_Remap_TIM5CH4_LSI, ENABLE);
}

/**
  * @brief  Remap I2C1
  * @param  None
  * @retval : None
  */
void Remap_I2C1_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO , ENABLE);
  /* Remap PB8-9 to I2C1 */
  GPIO_PinRemapConfig(GPIO_Remap_I2C1, ENABLE);
}
