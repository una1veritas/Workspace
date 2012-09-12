/**
  ******************************************************************************
  * @file    systick_7seg_6digi_and_obled/main.c
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
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
#define FLASH_TIME 300
#define FRAME_TIME 150
#define BLINK_TIME 150
/* Private variables ---------------------------------------------------------*/
static __IO uint32_t TimingDelay;
__IO uint8_t LEDOut[6] = {0,0,0,0,0,0};
__IO uint8_t Column = 0;
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "LEDs flash and line flows. 012345 is shown while OB LED blinking.\r\n";
uint8_t Digit[] = {0b00111111,0b00000110,0b01011011,0b01001111,0b01100110,
                   0b01101101,0b01111101,0b00100111,0b01111111,0b01101111};
// Pattern            dgfedcba   dgfedcba   dgfedcba   dgfedcba   dgfedcba
//                           a    g     a    g  d  a    g  d          d
uint16_t Pattern[] = {0b00000001,0b01000001,0b01001001,0b01001000,0b00001000};
uint32_t LED_Pin[8] =  {GPIOY_0_PIN,
                        GPIOY_1_PIN,
                        GPIOY_2_PIN,
                        GPIOY_3_PIN,
                        GPIOY_4_PIN,
                        GPIOY_5_PIN,
                        GPIOY_6_PIN,
                        GPIOY_7_PIN};
GPIO_TypeDef *LED_Port[8] = {GPIOY_0_PORT,
                             GPIOY_1_PORT,
                             GPIOY_2_PORT,
                             GPIOY_3_PORT,
                             GPIOY_4_PORT,
                             GPIOY_5_PORT,
                             GPIOY_6_PORT,
                             GPIOY_7_PORT};
/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void Delay(__IO uint32_t nCount);
void TimingDelay_Decrement(void);
void Digits_Show(uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t);
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  uint8_t i;


  // Configure board specific setting
  BoardInit();
  /* System Clocks Configuration **********************************************/
  GPIO_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();

  /* Setup SysTick Timer for 1 msec interrupts  */
  //SystemCoreClock is defined insystem_stm32f10x.c
  //by default SystemCoreClock is 72,000,000. SystemCoreClock/1000 is  72,000
  //System clock frequency is 72MHz. After counting 72,000 at 72MHz, 1ms passes.
  if (SysTick_Config(SystemCoreClock / 1000))
  {
    /* Capture error */
    while (1);
  }

  //Send welcome messages
  cprintf(Welcome_Message);

  //Flash 8 and 0 : 3 times
  for (i = 0 ; i <3 ; i++)
    {
      Digits_Show(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
      Delay(FLASH_TIME);
      Digits_Show(0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
      Delay(FLASH_TIME);
    }

  //Line flows rightward : 3 times
  for (i = 0 ; i <3 ; i++)
    {
      Digits_Show(Pattern[4], Pattern[3], Pattern[2], Pattern[1], Pattern[0],          0);
      Delay(FRAME_TIME);
      Digits_Show(         0, Pattern[4], Pattern[3], Pattern[2], Pattern[1], Pattern[0]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[0],          0, Pattern[4], Pattern[3], Pattern[2], Pattern[1]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[1], Pattern[0],          0, Pattern[4], Pattern[3], Pattern[2]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[2], Pattern[1], Pattern[0],          0, Pattern[4], Pattern[3]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[3], Pattern[2], Pattern[1], Pattern[0],          0, Pattern[4]);
      Delay(FRAME_TIME);
    }

  //Line flows leftward : 3 times
  for (i = 0 ; i <3 ; i++)
    {
      Digits_Show(Pattern[0], Pattern[1], Pattern[2], Pattern[3], Pattern[4],          0);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[1], Pattern[2], Pattern[3], Pattern[4],          0, Pattern[0]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[2], Pattern[3], Pattern[4],          0, Pattern[0], Pattern[1]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[3], Pattern[4],          0, Pattern[0], Pattern[1], Pattern[2]);
      Delay(FRAME_TIME);
      Digits_Show(Pattern[4],          0, Pattern[0], Pattern[1], Pattern[2], Pattern[3]);
      Delay(FRAME_TIME);
      Digits_Show(         0, Pattern[0], Pattern[1], Pattern[2], Pattern[3], Pattern[4]);
      Delay(FRAME_TIME);
    }


  //Show numbers while ob LED brinking
  Digits_Show(Digit[0], Digit[1], Digit[2], Digit[3], Digit[4], Digit[5]);

  while (1)
    {
      /* Turn on LED */
      GPIO_SetBits(OB_LED_PORT, OB_LED_PIN);
      /* Insert delay */
      Delay(BLINK_TIME);
      /* Turn off LED */
      GPIO_ResetBits(OB_LED_PORT, OB_LED_PIN);
      /* Insert delay */
      Delay(BLINK_TIME);
    }
}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOX_RCC |
                         GPIOY_0_RCC |
                         GPIOY_1_RCC |
                         GPIOY_2_RCC |
                         GPIOY_3_RCC |
                         GPIOY_4_RCC |
                         GPIOY_5_RCC |
                         OB_LED_GPIO_RCC
                         , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIO_X:Each Segment of LED */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN | GPIOX_1_PIN | GPIOX_2_PIN |
                                GPIOX_3_PIN | GPIOX_4_PIN | GPIOX_5_PIN |
                                GPIOX_6_PIN | GPIOX_7_PIN ;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_0:Segment 1 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_1:Segment 2 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_2:Segment 3 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_2_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_2_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_3:Segment 4 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_3_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_3_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_4:Segment 5 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_4_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_4_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_5:Segment 6 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_5_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_5_PORT, &GPIO_InitStructure);

  /* Configure OB_LED: output push-pull */
  GPIO_InitStructure.GPIO_Pin = OB_LED_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(OB_LED_PORT, &GPIO_InitStructure);
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

/**
  * @brief  Show numbers on digits
  * @param  Time:Repetition times of dynamic lighting
  * @param  DigitX:pattern data to show. X is column number starting from 1 and right.
  * @retval : None
  */
void Digits_Show(uint8_t Digit6, uint8_t Digit5, uint8_t Digit4,
                 uint8_t Digit3, uint8_t Digit2, uint8_t Digit1)
{
  LEDOut[0] = Digit1;
  LEDOut[1] = Digit2;
  LEDOut[2] = Digit3;
  LEDOut[3] = Digit4;
  LEDOut[4] = Digit5;
  LEDOut[5] = Digit6;
}

/**
  * @brief  put off lighting digit and put next digit. called by interrupt
  * @param  None
  * @retval : None
  */
void Put_Next_Digit(void)
{
  GPIO_Write(GPIOX_PORT, GPIO_ReadOutputData(GPIOX_PORT) & 0xFF00);
  GPIO_ResetBits(LED_Port[Column],LED_Pin[Column]);
  if ( ++Column >= 6)
    {
      Column = 0;
    }
  GPIO_SetBits(LED_Port[Column],LED_Pin[Column]);
  GPIO_Write(GPIOX_PORT, (GPIO_ReadOutputData(GPIOX_PORT) & 0xFF00) | LEDOut[Column]);
}
