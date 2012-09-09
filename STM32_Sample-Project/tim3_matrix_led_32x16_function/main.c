/**
  ******************************************************************************
  * @file    tim3_matrix_led_32x16_function/main.c
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
#include "32x16led.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define PICTURE_NUMBER 3
#define PATTTERN_NUM 2
/* Private variables ---------------------------------------------------------*/
__IO uint32_t tmp;
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
extern __IO uint32_t vram[16];
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/


/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  __IO uint16_t i;

  // Configure board specific setting
  BoardInit();

  LED_Configuraion();

  //set picture data to vram
  Set_VRAM(Pattern[PATTTERN_NUM]);

  while (1)
  {
    //scroll upward
    for(i=0;i<=7;i++)
      {
        Scroll_Upward();
        Delay(80);
      }

    //scroll downward
    for(i=0;i<=7;i++)
      {
        Scroll_Downward();
        Delay(80);
      }

    //scroll leftward
    for (i=0;i<=15;i++)
      {
        Scroll_Leftward();
        Delay(40);
      }

    //scroll rightward
    for (i=0;i<=15;i++)
      {
        Scroll_Rightward();
        Delay(40);
      }
  }
}
