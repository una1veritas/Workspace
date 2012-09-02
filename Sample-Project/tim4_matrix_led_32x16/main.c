/**
  ******************************************************************************
  * @file    tim4_matrix_led_32x16/main.c
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
#define PICTURE_NUMBER 3
#define TIM4_ARR 10
#define PATTTERN_NUM 2
/* Private variables ---------------------------------------------------------*/
__IO uint32_t Count;
__IO uint16_t i,j;
__IO uint32_t tmp;
__IO uint8_t line = 0;
__IO uint32_t vram[16] = {0x0000000,0x0000000,0x0000000,0x0000000,
                          0x0000000,0x0000000,0x0000000,0x0000000};
__I uint32_t Pattern[PICTURE_NUMBER][16] = {
{0b00000000000000000000000000000000, // Pattern 0
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000},
{0b11111111111111111111111111111111, // Pattern 1
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111,
 0b11111111111111111111111111111111},
{0b01110000000001110000000101010100, // Pattern 2
 0b10001000000010001000000101010000,
 0b10000000000010001010110010100100,
 0b10111001110010001011001000000000,
 0b10001010001010001010001001001000,
 0b10001010001010001010001011101110,
 0b01111001110001110010001001001010,
 0b00000000000000000000000000000000,
 0b01111011111010001011111001110000,
 0b10000000100011011000010010001000,
 0b10000000100010101000100000001000,
 0b01110000100010001000010000010000,
 0b00001000100010001000001000100000,
 0b00001000100010001010001001000000,
 0b11110000100010001001110011111000,
 0b00000000000000000000000000000000}
};

/* Private function prototypes -----------------------------------------------*/
void RCC_Configuration(void);
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void GPIO_ControlBit(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, uint32_t Bitval);
void Send_3para_2byte_To_SR(uint16_t data1, uint16_t data2, uint16_t data3);
void TIM4_Configuration(void);
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

  NVIC_Configuration();

  /* TIM Configuration*/
  TIM4_Configuration();
  /* GPIO Configuration*/
  GPIO_Configuration();

  //Set CLOCK low
  GPIO_ResetBits(GPIOY_3_PORT, GPIOY_3_PIN);
  //Set LATCH high
  GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  //Set STROBE low
  GPIO_ResetBits(GPIOY_5_PORT, GPIOY_5_PIN);

  //set picture data to vram
  for (i=0;i<=15;i++)
    {
      vram[i] = Pattern[PATTTERN_NUM][i];
    }

  while (1)
  {
    //scroll upward
    for(i=0;i<=15;i++)
      {
        tmp = vram[0];
        for (j=0;j<=14;j++)
          {
            vram[j] = vram[j+1];
          }
        vram[15] = tmp;
        delay_ms(80);
      }

    //scroll leftward
    for (i=0;i<=31;i++)
      {
        for (j=0;j<=31;j++)
          {
            tmp = vram[j] >> 31;
            vram[j] = (vram[j] << 1) | tmp;
          }
        delay_ms(40);
      }
  }
}

/**
  * @brief  show next line of 8x8 matrix LED
  * @param  None
  * @retval : None
  */
void show_next_line(void)
{
  //Output disable
  GPIO_SetBits(GPIOY_5_PORT, GPIOY_5_PIN);
  // Senddata to shift register
  Send_3para_2byte_To_SR(0x0001 << line, (vram[line] & 0xFFFF0000) >> 16, vram[line] & 0x0000FFFFF );
  line = (line+1) & 0x0F;
  //Output enable
  GPIO_ResetBits(GPIOY_5_PORT, GPIOY_5_PIN);
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
      GPIO_ControlBit(GPIOY_0_PORT, GPIOY_0_PIN, data1 & ( 0x0001 << i));
      GPIO_ControlBit(GPIOY_1_PORT, GPIOY_1_PIN, data2 & ( 0x0001 << i));
      GPIO_ControlBit(GPIOY_2_PORT, GPIOY_2_PIN, data3 & ( 0x0001 << i));
      // Send Clock
      GPIO_SetBits(GPIOY_3_PORT, GPIOY_3_PIN);
      GPIO_ResetBits(GPIOY_3_PORT, GPIOY_3_PIN);
    }
  //On and Off Latch
  GPIO_ResetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);
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
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable the TIM4 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
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
void GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC | GPIOY_1_RCC | GPIOY_2_RCC |
                         GPIOY_3_RCC | GPIOY_4_RCC | GPIOY_5_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_0:SIN1 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_1:SIN2 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_2:SIN3 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_2_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_2_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_3:CLOCK */
  GPIO_InitStructure.GPIO_Pin = GPIOY_3_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_3_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_4:LATCH */
  GPIO_InitStructure.GPIO_Pin = GPIOY_4_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_3_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_4:STROBE */
  GPIO_InitStructure.GPIO_Pin = GPIOY_5_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_3_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure TIM4
  * @param  None
  * @retval : None
  */
void TIM4_Configuration(void)
{
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM4_RCC , ENABLE);

  /* ---------------------------------------------------------------------------
    TIM4 Configuration: Output Compare Toggle Mode:
    TIM4CLK = 36 MHz, Prescaler = 3600, TIM4 counter clock = 10kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = TIM4_ARR;
  TIM_TimeBaseStructure.TIM_Prescaler = 3600;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);

  /* TIM IT enable */
  TIM_ITConfig(TIM4, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM4, ENABLE);
}

