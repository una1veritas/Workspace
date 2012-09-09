/**
  ******************************************************************************
  * @file    tim13_matrix_led_8x8/main.c
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
#define PICTURE_DURATION 0x001B
#define LINE_DURATION 0x1000
#define PICTURE_NUMBER 16
#define TIM1_ARR 1000
#define TIM3_ARR 150
/* Private variables ---------------------------------------------------------*/
TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
TIM_OCInitTypeDef  TIM_OCInitStructure;
__IO uint8_t line = 0;
__I uint8_t *Present_Picture_Pointer;
__IO uint8_t Present_Picture_Number = 0;
__I uint8_t  Pattern[ PICTURE_NUMBER][8] = {
                              {0b00000000, // Pattern 0
                               0b00000000,
                               0b00000000,
                               0b00000000,
                               0b00000000,
                               0b00000000,
                               0b00000000,
                               0b00000000},
                              {0b10000000, // Pattern 1
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000},
                              {0b11111111, // Pattern 2
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000,
                               0b10000000},
                              {0b11111111, // Pattern 3
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001},
                              {0b11111111, // Pattern 4
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b10000001,
                               0b11111111},
                              {0b11111111, // Pattern 5
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11111111},
                              {0b11111111, // Pattern 6
                               0b11111111,
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11000001,
                               0b11111111},
                              {0b11111111, // Pattern 7
                               0b11111111,
                               0b11000011,
                               0b11000011,
                               0b11000011,
                               0b11000011,
                               0b11000011,
                               0b11111111},
                              {0b11111111, // Pattern 8
                               0b11111111,
                               0b11000011,
                               0b11000011,
                               0b11000011,
                               0b11000011,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 9
                               0b11111111,
                               0b11100011,
                               0b11100011,
                               0b11100011,
                               0b11100011,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 10
                               0b11111111,
                               0b11111111,
                               0b11100011,
                               0b11100011,
                               0b11100011,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 11
                               0b11111111,
                               0b11111111,
                               0b11100111,
                               0b11100111,
                               0b11100111,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 12
                               0b11111111,
                               0b11111111,
                               0b11100111,
                               0b11100111,
                               0b11111111,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 13
                               0b11111111,
                               0b11111111,
                               0b11110111,
                               0b11110111,
                               0b11111111,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 14
                               0b11111111,
                               0b11111111,
                               0b11111111,
                               0b11110111,
                               0b11111111,
                               0b11111111,
                               0b11111111},
                              {0b11111111, // Pattern 15
                               0b11111111,
                               0b11111111,
                               0b11111111,
                               0b11111111,
                               0b11111111,
                               0b11111111,
                               0b11111111}
                             };
/* Private function prototypes -----------------------------------------------*/
void RCC_Configuration(void);
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void display_8x8(__I uint8_t* Pattern);
void Send2ByteToShiftRegister(uint16_t data);
void TIM1_Configuration(void);
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
  /* System Clocks Configuration **********************************************/
  RCC_Configuration();

  NVIC_Configuration();

  Present_Picture_Pointer = Pattern[Present_Picture_Number];

  /* TIM Configuration*/
  TIM1_Configuration();
  TIM3_Configuration();
  /* GPIO Configuration*/
  GPIO_Configuration();

  while (1)
  {
  }
}

/**
  * @brief  show next line of 8x8 matrix LED
  * @param  None
  * @retval : None
  */
void show_next_line(void)
{
  line++;
  if (line>7)
    {
      line = 0;
    }

  Send2ByteToShiftRegister(((*(Present_Picture_Pointer+line))<<8)| ( 0b00000001 << line));

}

/**
  * @brief  show next picture on matrix LED
  * @param  None
  * @retval : None
  */
void show_next_pic(void)
{
  Present_Picture_Number++;
  if (Present_Picture_Number >=  PICTURE_NUMBER)
    {
      Present_Picture_Number = 0;
    }
  Present_Picture_Pointer = Pattern[Present_Picture_Number];
}

/**
  * @brief  set shift-register1(ST:GPIOY_2_PORT) bit based on send byte data
  * @param  data : 2 byte data to be set on shift-register
  * @retval : None
  */
void Send2ByteToShiftRegister(uint16_t data)
{
  uint8_t i;

  for (i=0; i<=15; i++)
    {
      if ((data >> i) & 0b00000001)
        {
          // Turn on SI
          GPIO_SetBits(GPIOY_1_PORT, GPIOY_1_PIN);
        }
      else
        {
          GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
        }
      // Send Clock
      GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);
      GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
    }

  //Set ST
  GPIO_SetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  //Reset ST
  GPIO_ResetBits(GPIOY_2_PORT, GPIOY_2_PIN);

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

  /* PCLK2 = HCLK/4 */
  RCC_PCLK2Config(RCC_HCLK_Div4);
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

  /* Enable the TIM3 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
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
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC |
                         GPIOY_1_RCC |
                         GPIOY_2_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_0:CK */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_1:SI */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_2:ST */
  GPIO_InitStructure.GPIO_Pin = GPIOY_2_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_2_PORT, &GPIO_InitStructure);


}

/**
  * @brief  Configure TIM1
  * @param  None
  * @retval : None
  */
void TIM1_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM1_RCC , ENABLE);

#if defined (PARTIAL_REMAP_TIM1)
PartialRemap_TIM1_Configuration();
#elif defined (FULL_REMAP_TIM1)
FullRemap_TIM1_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM1 Configuration: Output Compare Toggle Mode:
    TIM1CLK = 36 MHz, Prescaler = 36, TIM1 counter clock = 1MHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = TIM1_ARR;
  TIM_TimeBaseStructure.TIM_Prescaler = 36;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);

  /* TIM IT enable */
  TIM_ITConfig(TIM1, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM1, ENABLE);

}

/**
  * @brief  Configure TIM3
  * @param  None
  * @retval : None
  */
void TIM3_Configuration(void)
{
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC , ENABLE);

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
  TIM_TimeBaseStructure.TIM_Period = TIM3_ARR;
  TIM_TimeBaseStructure.TIM_Prescaler = 36000;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  /* TIM IT enable */
  TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);

}
