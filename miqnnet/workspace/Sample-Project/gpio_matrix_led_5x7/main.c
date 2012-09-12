/**
  ******************************************************************************
  * @file    gpio_matrix_led_5x7/main.c
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
#define PICTURE_DURATION 60
#define LINE_DURATION 2000
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint8_t  Pattern[6][7] = {
                           {0b11111,
                            0b00001,
                            0b00101,
                            0b00110,
                            0b00100,
                            0b00100,
                            0b01000},
                           {0b00000,
                            0b01110,
                            0b00000,
                            0b00000,
                            0b00000,
                            0b00000,
                            0b11111},
                           {0b00001,
                            0b00001,
                            0b01101,
                            0b00010,
                            0b00010,
                            0b00101,
                            0b11000},
                           {0b00111,
                            0b11010,
                            0b00101,
                            0b00100,
                            0b00100,
                            0b00010,
                            0b00001},
                           {0b00010,
                            0b11111,
                            0b00110,
                            0b01010,
                            0b00110,
                            0b00010,
                            0b00100},
                           {0b00100,
                            0b00100,
                            0b00100,
                            0b00100,
                            0b00000,
                            0b00100,
                            0b00100}
                        };
/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void Send2ByteToShiftRegister(uint16_t data);
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  uint16_t i,j,l;


  // Configure board specific setting
  BoardInit();
  /* GPIO Configuration*/
  GPIO_Configuration();

  while (1)
  {
    //Loop for changing picture
    for (i=0; i<=5 ; i++)
      {
        //Loop for keeping picture
        for (j=0; j<PICTURE_DURATION; j++)
          {
            //Loop for shifting lines
            for (l=0; l<=6; l++)
              {
                Send2ByteToShiftRegister(( 1 << (l+8)) | Pattern[i][l]);
                /* Insert delay */
                delay_us(LINE_DURATION);
              }
          }
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
