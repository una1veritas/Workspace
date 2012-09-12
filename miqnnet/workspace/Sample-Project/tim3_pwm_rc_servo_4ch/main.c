/**
  ******************************************************************************
  * @file    tim3_pwm_rc_servo_4ch/main.c
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
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
__IO int8_t RxData;
__IO int16_t Duty = 1350;
const int8_t Welcome_Message[] = "\r\nHello Cortex-M3/STM32 World!\r\n"
                            "Expand your creativity and enjoy making.\r\n\r\n"
                            "You can switch between serial control mode and continues move mode\r\n"
                            "by pressing \"m\" at  any time.\r\n\r\n";
/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
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
  // Setting up COM port for Print function
  COM_Configuration();

  /* TIM3 Configuration*/
  TIM3_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  TIM_SetCompare1(TIM3,Duty);
  TIM_SetCompare2(TIM3,Duty-600);
  TIM_SetCompare3(TIM3,Duty);
  TIM_SetCompare4(TIM3,Duty+600);

  while (1)
    {

      cprintf("Serial Control Mode: \r\n");
      cprintf("You can move ch1 servo, +10 degree by pushing \"U\", \r\n");
      cprintf("-10 degree by pushing \"D\". \r\n");
      cprintf("While, ch2-4 servo is set to -60,+-0,+60 degree position.\r\n");
      while(1)
        {
          if(RX_BUFFER_IS_NOT_EMPTY)
            {
              RxData = (int8_t)RECEIVE_DATA;
              if (RxData == 'u' || RxData == 'U')
                {
                  Duty += 100;
                  cprintf("Duty count is: %u\r\n", Duty);
                }
              if ((RxData == 'd' || RxData == 'D'))
                {
                  Duty -= 100;
                  cprintf("Duty count is: %u\r\n", Duty);
                }
              if ((RxData == 'm' || RxData == 'M'))
                {
                  break;
                }
              TIM_SetCompare1(TIM3,Duty);
            }
        }

      cprintf("Continues Move Mode: \r\n");
      cprintf("ch2-4 servo moves -60 to +60 degee position every 0.5 second\r\n");
      cprintf("While, ch1 servo keeps +-0 position.\r\n");
      while(1)
        {
          TIM_SetCompare1(TIM3,1350);
          TIM_SetCompare2(TIM3,750);
          TIM_SetCompare3(TIM3,1350);
          TIM_SetCompare4(TIM3,1950);
          delay_ms(500);
          TIM_SetCompare2(TIM3,1350);
          TIM_SetCompare3(TIM3,1950);
          TIM_SetCompare4(TIM3,750);
          delay_ms(500);
          TIM_SetCompare2(TIM3,1950);
          TIM_SetCompare3(TIM3,750);
          TIM_SetCompare4(TIM3,1350);
          delay_ms(500);
          if(RX_BUFFER_IS_NOT_EMPTY)
            {
              RxData = (int8_t)RECEIVE_DATA;
              if ((RxData == 'm' || RxData == 'M'))
                {
                  break;
                }
            }
        }
    }
}

/**
  * @brief  Configure TIM3
  * @param  None
  * @retval : None
  */
void TIM3_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_OCInitTypeDef  TIM_OCInitStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC , ENABLE);
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM3_CH12_GPIO_RCC | TIM3_CH34_GPIO_RCC , ENABLE);

  /* GPIO Configuration:TIM3 Channel1 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM3_CH1_PIN | TIM3_CH2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(TIM3_CH12_PORT, &GPIO_InitStructure);
  GPIO_InitStructure.GPIO_Pin = TIM3_CH3_PIN | TIM3_CH4_PIN ;
  GPIO_Init(TIM3_CH34_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM3)
PartialRemap_TIM3_Configuration();
#elif defined (FULL_REMAP_TIM3)
FullRemap_TIM3_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM3 Configuration: Output Compare Toggle Mode:
    TIM3CLK = 72 MHz, Prescaler = 60, TIM3 counter clock = 1.2MHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  // 2500us cycle
  TIM_TimeBaseStructure.TIM_Period = 17999;
  // 2000us cycle
  //TIM_TimeBaseStructure.TIM_Period = 14399;
  TIM_TimeBaseStructure.TIM_Prescaler = 79;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
//  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC1Init(TIM3, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* Output Compare Toggle Mode configuration: Channel2 */
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC2Init(TIM3, &TIM_OCInitStructure);
  TIM_OC2PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* Output Compare Toggle Mode configuration: Channel3 */
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC3Init(TIM3, &TIM_OCInitStructure);
  TIM_OC3PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* Output Compare Toggle Mode configuration: Channel4 */
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC4Init(TIM3, &TIM_OCInitStructure);
  TIM_OC4PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);
}
