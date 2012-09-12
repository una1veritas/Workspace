/**
  ******************************************************************************
  * @file    tim4_scan_rotary_encoder/main.c
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
#define SAMPLE_TIMES 3
#define SAMPLE_INTERVAL 10
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHello Cortex-M3/STM32 World!\r\n"
                            "Expand your creativity and enjoy making.\r\n\r\n"
                            "Turn rotary encoder clock wise (CW) or counter clock wise (CCW).\r\n"
                            "CW increments the counter and CCW decrements the counter.\r\n"
                            "I tell you the turn direction and the counter value.\r\n";
int8_t Direction;
uint8_t SW0_previous_state = 0, SW0_present_state = 0,
       SW0_scan_count = 0, SW0_flag = RESET, Counter = 100;
int8_t encoder_lookup[4][4] = { {0,1,-1,0}, {-1,0,0,1}, {1,0,0,-1}, {0,-1,1,0} };
/* Private function prototypes -----------------------------------------------*/
void RCC_Configuration(void);
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void TIM4_Configuration(void);
void Turn_CW(void);
void Turn_CCW(void);
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
  /* NVIC Configuration */
  NVIC_Configuration();

  /* TIM4 Configuration*/
  TIM4_Configuration();
  /* GPIO Configuration*/
  GPIO_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();

  SW0_previous_state = GPIO_ReadInputDataBit(GPIOY_0_PORT,GPIOY_0_PIN) |
                      (GPIO_ReadInputDataBit(GPIOY_1_PORT,GPIOY_1_PIN) << 1);

  //Send welcome messages
  cprintf(Welcome_Message);

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
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC |
                         GPIOY_1_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_0 : Pin A*/
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPD;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_1 : Pin B*/
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPD;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

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
  TIM_TimeBaseStructure.TIM_Prescaler = 360;
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
  SW0_present_state = GPIO_ReadInputDataBit(GPIOY_0_PORT,GPIOY_0_PIN) |
                      (GPIO_ReadInputDataBit(GPIOY_1_PORT,GPIOY_1_PIN) << 1);
  if (SW0_present_state != SW0_previous_state)
    {
      SW0_scan_count++;
      if (SW0_scan_count > SAMPLE_TIMES)
        {
          SW0_scan_count = 0;
          Direction = encoder_lookup[SW0_previous_state][SW0_present_state];
          SW0_previous_state = SW0_present_state;
          if (Direction == 1 && SW0_present_state == 0)
            {
              Turn_CW();
            }
          else if (Direction == -1 && SW0_present_state == 0)
            {
              Turn_CCW();
            }

        }
    }
}

/**
  * @brief  Action when encoder turned CW
  * @param  None
  * @retval : None
  */
void Turn_CW(void)
{
  Counter++;
  cprintf("Turned CW!  Counter value is : %u\r\n", Counter);

}

/**
  * @brief  Action when encoder turned CCW
  * @param  None
  * @retval : None
  */
void Turn_CCW(void)
{
  Counter--;
  cprintf("Turned CCW!  Counter value is : %u\r\n", Counter);
}

/**
  * @}
  */
