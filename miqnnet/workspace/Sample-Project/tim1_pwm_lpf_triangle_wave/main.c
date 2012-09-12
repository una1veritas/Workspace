/**
  ******************************************************************************
  * @file    tim1_pwm_lpf_triangle_wave/main.c
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
#define PWM_COUNT 300
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
__IO uint16_t CC1_Value;
__IO uint16_t Freq = 400;
__IO uint16_t Sample_Rate = 32000;
__IO uint16_t Period;
__IO uint16_t Counter = 0;
__IO uint8_t RxData;
const int8_t Welcome_Message[] =
  "\r\nHello Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Monitor voltage at TIM1_CH1 filtered by LPF. \r\n"
  "You can make the voltage up and down by pressing U or D. \r\n";
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void TIM1_Configuration(void);
void TIM1_UP_IRQHandler(void);
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
  /* NVIC Configuration */
  NVIC_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();
  /* TIM1 Configuration*/
  TIM1_Configuration();

  //Compute period length corresponding to frequency
  Period = (uint16_t)(Sample_Rate / Freq);

  //Send welcome messages
  cprintf(Welcome_Message);

  while(1)
    {
      if(RX_BUFFER_IS_NOT_EMPTY)
        {
          RxData = (int8_t)RECEIVE_DATA;
          if (RxData == 'u' || RxData == 'U')
            {
              if (Freq < 12800)
                {
                  Freq = Freq * 2;
                }

              cprintf("Frequency is : %u Hz\r\n", Freq);
            }
          if ((RxData == 'd' || RxData == 'D'))
            {
              if (Freq > 25)
                {
                  Freq = Freq / 2;
                }

              cprintf("Frequency is : %u Hz\r\n", Freq);
            }
          //Compute period length corresponding to frequency
          Period = (uint16_t)(Sample_Rate / Freq);
        }
    }

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
}

/**
  * @brief  Configure TIM3
  * @param  None
  * @retval : None
  */
void TIM1_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_OCInitTypeDef  TIM_OCInitStructure;

  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM1_RCC | TIM1_GPIO_RCC , ENABLE);

  /* GPIO Configuration:TIM1 Channel1 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM1_CH1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(TIM1_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM1)
PartialRemap_TIM1_Configuration();
#elif defined (FULL_REMAP_TIM1)
FullRemap_TIM1_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM1 Configuration: Output Compare Toggle Mode:
    TIM1CLK = 72 MHz, Prescaler = 0, TIM1 counter clock = 72MHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = PWM_COUNT;
  TIM_TimeBaseStructure.TIM_Prescaler = 0;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = ((SystemCoreClock / 1) / Sample_Rate) / PWM_COUNT;
  TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
  TIM_OCInitStructure.TIM_OCIdleState = TIM_OCIdleState_Reset;
  TIM_OCInitStructure.TIM_OutputNState = TIM_OutputNState_Disable;
  TIM_OCInitStructure.TIM_OCNPolarity = TIM_OCNPolarity_High;
  TIM_OCInitStructure.TIM_OCNIdleState = TIM_OCIdleState_Reset;

  TIM_OC1Init(TIM1, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM1, TIM_OCPreload_Disable);

  /* TIM IT enable */
  TIM_ITConfig(TIM1, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM1, ENABLE);

  /* Main Output Enable */
  TIM_CtrlPWMOutputs(TIM1, ENABLE);
}


/**
  * @brief  This function handles TIM1 update interrupt request.
  * @param  None
  * @retval None
  */
void TIM1_UP_IRQHandler(void)
{
  /* Clear TIM1 update interrupt pending bit*/
  TIM_ClearITPendingBit(TIM1, TIM_IT_Update);

  if (Counter <= (Period /2))
    {
      CC1_Value = ((uint32_t)Counter)*2 * 0xFF / Period;
    }
  else
    {
      CC1_Value = 0xFF - (((uint32_t)Counter - (Period /2))*2 * 0xFF / Period);
    }

  TIM_SetCompare1(TIM1,CC1_Value);

  Counter++;
  if (Counter >= Period)
    {
      Counter = 0;
    }
}


