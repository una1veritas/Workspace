/**
  ******************************************************************************
  * @file    gpio_matrix_led_8x8/main.c
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
#define PICTURE_DURATION 6
#define LINE_DURATION 2000
/* Private variables ---------------------------------------------------------*/
__I uint8_t  Pattern[16][8] = {
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
void GPIO_Configuration(void);
void Display8x8(__I uint8_t* Pattern);
void Send2ByteToShiftRegister(uint16_t data);
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  uint16_t i;


  // Configure board specific setting
  BoardInit();
  /* GPIO Configuration*/
  GPIO_Configuration();

  while (1)
  {
    //Loop for changing picture
    for (i=0; i<=15 ; i++)
      {
        Display8x8(Pattern[i]);
      }
    for (i=14; i>0 ; i--)
      {
        Display8x8(Pattern[i]);
      }
  }
}

/**
  * @brief  Display a single picture for PICTURE_DURATION on 8x8 matrix
  * @param  Pattern: array of bytes, which contain a picture
  * @retval : None
  */
void Display8x8(__I uint8_t* Pattern)
{
  uint8_t j,l;
  //Loop for keeping picture
  for (j=0; j<PICTURE_DURATION; j++)
    {
      //Loop for shifting lines
      for (l=0; l<=7; l++)
        {
          Send2ByteToShiftRegister((Pattern[l]<<8)| ( 1 << l));
          /* Insert delay */
          delay_us(LINE_DURATION);
        }
    }
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
