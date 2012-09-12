/**
  ******************************************************************************
  * @file    tim4_scan_tip_switch_display/main.c
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
#include "32x16led.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define PICTURE_NUMBER 3
#define PATTTERN_NUM 2
#define SAMPLE_TIMES 25
#define SAMPLE_INTERVAL 20
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHello Cortex-M3/STM32 World!\r\n"
                            "Expand your creativity and enjoy making.\r\n\r\n"
                            "I revert arrow when sensor is tilt.\r\n";
int8_t SW0_previous_state = 0, SW0_present_state = 0,
       SW0_scan_count = 0, SW0_flag = RESET;
__I uint32_t Pattern[PICTURE_NUMBER][16] = {
{0b00000000001000000000000000000000, // Pattern 0
 0b00000000011100000000000000000000,
 0b00000000111110000000000000000000,
 0b00000001111111000000000000000000,
 0b00000011111111100000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000},
{0b00000000000000000000000000000000, // Pattern 1
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000000000000000000000,
 0b00000000000000100000000000000000,
 0b00000000000000110000000000000000,
 0b00000000000000111000000000000000,
 0b00000000000000111100000000000000,
 0b00000000001111111110000000000000,
 0b00000000001000111100000000000000,
 0b00000000001000111000000000000000,
 0b00000000001000110000000000000000,
 0b00000000001000100000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000,
 0b00000000001000000000000000000000},
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
void RCC_Configuration(void);
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void TIM4_Configuration(void);
void SW0_is_released(void);
void SW0_is_pressed(void);

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

  //Configures TIM4 for scanning
  GPIO_Configuration();
  TIM4_Configuration();

  //Configures matrix display
  LED_Configuraion();

  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  //set picture data to vram
  Set_VRAM(Pattern[2]);
  Delay(2000);
  Set_VRAM(Pattern[0]);

  while (1)
  {
  }
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
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;


  /* Configure GPIO for GPIOY_0 */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

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
    TIM4CLK = 36 MHz, Prescaler = 36000, TIM4 counter clock = 1kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = SAMPLE_INTERVAL;
  TIM_TimeBaseStructure.TIM_Prescaler = 36000;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);

  /* TIM IT enable */
  TIM_ITConfig(TIM4, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM4, ENABLE);

}

/**
  * @brief  Scan state of switches
  * @param  None
  * @retval : None
  */
void Scan_Switch(void)
{
  SW0_present_state = GPIO_ReadInputDataBit(GPIOY_0_PORT,GPIOY_0_PIN);
  if (SW0_present_state != SW0_previous_state)
    {
      SW0_scan_count++;
      if (SW0_scan_count > SAMPLE_TIMES)
        {
          SW0_previous_state = SW0_present_state;
          SW0_flag = SET;
        }
    }
  else
    {
      SW0_scan_count = 0;
    }

  if (SW0_flag != RESET && SW0_present_state == 0)
    {
      SW0_flag = RESET;
      SW0_is_released();
    }
  else if (SW0_flag != RESET && SW0_present_state != 0)
    {
      SW0_flag = RESET;
      SW0_is_pressed();
    }
}

/**
  * @brief  Action when SW0 is released
  * @param  None
  * @retval : None
  */
void SW0_is_released(void)
{
  Set_VRAM(Pattern[1]);
}

/**
  * @brief  Action when SW0 is pressed
  * @param  None
  * @retval : None
  */
void SW0_is_pressed(void)
{
  Set_VRAM(Pattern[0]);
}


