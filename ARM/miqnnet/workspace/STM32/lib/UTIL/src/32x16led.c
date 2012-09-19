/**
  ******************************************************************************
  * @file    lib_std/UTIL/src/32x16.c
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
#include "32x16led.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define TIM3_ARR 2
/* Private variables ---------------------------------------------------------*/
TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
TIM_OCInitTypeDef  TIM_OCInitStructure;
static __IO uint32_t TimingDelay;
__IO uint8_t line = 0;
__IO uint32_t vram[16] = {0x0000000,0x0000000,0x0000000,0x0000000,
                          0x0000000,0x0000000,0x0000000,0x0000000};
/* Private function prototypes -----------------------------------------------*/


/* Private functions ---------------------------------------------------------*/

void Scroll_Upward(void)
{
  uint8_t i;
  uint32_t tmp;
  tmp = vram[0];
  for (i=0;i<=14;i++)
    {
      vram[i] = vram[i+1];
    }
  vram[15] = tmp;
}
void Scroll_Downward(void)
{
  uint8_t i;
  uint32_t tmp;
  tmp = vram[15];
  for (i=15;i>0;i--)
    {
      vram[i] = vram[i-1];
    }
  vram[0] = tmp;
}
void Scroll_Leftward(void)
{
  uint8_t i;
  uint32_t tmp;
  for (i=0;i<=31;i++)
    {
      tmp = vram[i] >> 31;
      vram[i] = (vram[i] << 1) | tmp;
    }
}
void Scroll_Rightward(void)
{
  uint8_t i;
  uint32_t tmp;
  for (i=0;i<=31;i++)
    {
      tmp = vram[i] << 31;
      vram[i] = (vram[i] >> 1) | tmp;
    }
}

void Set_VRAM(const __IO uint32_t* Pattern)
{
  uint32_t i;

  for (i=0;i<=15;i++)
    {
      vram[i] = Pattern[i];
    }
}

void LED_Configuraion(void)
{
  /* PCLK1 = HCLK/4 */
  RCC_PCLK1Config(RCC_HCLK_Div4);

  LED_NVIC_Configuration();
  /* TIM Configuration*/
  LED_TIM3_Configuration();
  /* GPIO Configuration*/
  LED_GPIO_Configuration();

  //Set CLOCK low
  GPIO_ResetBits(GPIOX_PORT, GPIOX_3_PIN);
  //Set LATCH high
  GPIO_SetBits(GPIOX_PORT, GPIOX_4_PIN);
  //Set STROBE low
  GPIO_ResetBits(GPIOX_PORT, GPIOX_5_PIN);
}

/**
  * @brief  show next line of 8x8 matrix LED
  * @param  None
  * @retval : None
  */
void show_next_line(void)
{
  //Output disable
  GPIO_SetBits(GPIOX_PORT, GPIOX_5_PIN);
  // Senddata to shift register
  Send_3para_2byte_To_SR(0x0001 << line, (vram[line] & 0xFFFF0000) >> 16, vram[line] & 0x0000FFFFF );
  line = (line+1) & 0x0F;
  //Output enable
  GPIO_ResetBits(GPIOX_PORT, GPIOX_5_PIN);
}

/**
  * @brief  Set specified bit
  * @param  Bitval: bit data to be set. 0 is L, the other is H.
  * @retval : None
  */
void GPIO_ControlBit(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, uint32_t Bitval)
{
  if (Bitval)
    {
      GPIOx->BSRR = GPIO_Pin;
    }
  else
    {
      GPIOx->BRR = GPIO_Pin;
    }
}

/**
  * @brief  Send byte data to shift register and output at a moment
  * @param  byte: data to be output
  * @retval : None
  */
void Send_3para_2byte_To_SR(uint16_t data1, uint16_t data2, uint16_t data3)
{
  uint8_t i;

  for (i=0; i<=15; i++)
    {
      GPIO_ControlBit(GPIOX_PORT, GPIOX_0_PIN, data1 & ( 0x0001 << i));
      GPIO_ControlBit(GPIOX_PORT, GPIOX_1_PIN, data2 & ( 0x0001 << i));
      GPIO_ControlBit(GPIOX_PORT, GPIOX_2_PIN, data3 & ( 0x0001 << i));
      // Send Clock
      GPIO_SetBits(GPIOX_PORT, GPIOX_3_PIN);
      GPIO_ResetBits(GPIOX_PORT, GPIOX_3_PIN);
    }
  //On and Off Latch
  GPIO_ResetBits(GPIOX_PORT, GPIOX_4_PIN);
  GPIO_SetBits(GPIOX_PORT, GPIOX_4_PIN);
}


/**
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void LED_NVIC_Configuration(void)
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
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void LED_GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOX_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOX_0:SIN1 */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOX_1:SIN2 */
  GPIO_InitStructure.GPIO_Pin = GPIOX_1_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOX_2:SIN3 */
  GPIO_InitStructure.GPIO_Pin = GPIOX_2_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOX_3:CLOCK */
  GPIO_InitStructure.GPIO_Pin = GPIOX_3_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOX_4:LATCH */
  GPIO_InitStructure.GPIO_Pin = GPIOX_4_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOX_4:STROBE */
  GPIO_InitStructure.GPIO_Pin = GPIOX_5_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure TIM3
  * @param  None
  * @retval : None
  */
void LED_TIM3_Configuration(void)
{
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC , ENABLE);

  /* ---------------------------------------------------------------------------
    TIM3 Configuration: Output Compare Toggle Mode:
    TIM3CLK = 36 MHz, Prescaler = 3600, TIM3 counter clock = 10kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = TIM3_ARR;
  TIM_TimeBaseStructure.TIM_Prescaler = 3600;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  /* Output Compare Inactive Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Active;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Disable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC1Init(TIM3, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* TIM IT enable */
  TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);
}

/**
  * @brief  Inserts a delay time.
  * @param nTime: specifies the delay time length, in milliseconds.
  * @retval : None
  */
void Delay(__IO uint32_t nTime)
{
  TimingDelay = nTime;

  while(TimingDelay != 0);
}

/**
  * @brief  Decrements the TimingDelay variable.
  * @param  None
  * @retval : None
  */
void TimingDelay_Decrement(void)
{
  if (TimingDelay != 0x00)
  {
    TimingDelay--;
  }
}
