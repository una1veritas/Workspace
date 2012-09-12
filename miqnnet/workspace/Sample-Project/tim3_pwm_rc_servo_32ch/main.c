/**
  ******************************************************************************
  * @file    tim3_pwm_rc_servo_32ch/main.c
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
#include "main.h"
#include "com_config.h"
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
__IO int8_t RxData;
__IO int16_t Duty = 1350;
__IO int16_t Position[32];
__IO uint16_t Decoder_Select = 0;
const int8_t Welcome_Message[] = "\r\nHello Cortex-M3/STM32 World!\r\n"
                            "Expand your creativity and enjoy making.\r\n\r\n"
                            "32 servos swings continuasly.\r\n\r\n";
/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void NVIC_Configuration(void);
void Set_Select(uint16_t Decoder_Select);
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
  NVIC_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();

  /* TIM3 Configuration*/
  TIM3_Configuration();
  /* GPIO Configuration*/
  GPIO_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  TIM_SetCompare1(TIM3, NEUTRAL);
  TIM_SetCompare2(TIM3, NEUTRAL);
  TIM_SetCompare3(TIM3, NEUTRAL);
  TIM_SetCompare4(TIM3, NEUTRAL);

  while (1)
    {

      Position[ 0] = -600;
      Position[ 1] =  600;
      Position[ 2] = -600;
      Position[ 3] =  600;
      Position[ 4] = -600;
      Position[ 5] =  600;
      Position[ 6] = -600;
      Position[ 7] =  600;

      Position[ 8] = -600;
      Position[ 9] =  600;
      Position[10] = -600;
      Position[11] =  600;
      Position[12] = -600;
      Position[13] =  600;
      Position[14] = -600;
      Position[15] =  600;

      Position[16] = -600;
      Position[17] =  600;
      Position[18] = -600;
      Position[19] =  600;
      Position[20] = -600;
      Position[21] =  600;
      Position[22] = -600;
      Position[23] =  600;

      Position[24] = -600;
      Position[25] =  600;
      Position[26] = -600;
      Position[27] =  600;
      Position[28] = -600;
      Position[29] =  600;
      Position[30] = -600;
      Position[31] =  600;

      delay_ms(1000);

      Position[ 0] =  600;
      Position[ 1] = -600;
      Position[ 2] =  600;
      Position[ 3] = -600;
      Position[ 4] =  600;
      Position[ 5] = -600;
      Position[ 6] =  600;
      Position[ 7] = -600;

      Position[ 8] =  600;
      Position[ 9] = -600;
      Position[10] =  600;
      Position[11] = -600;
      Position[12] =  600;
      Position[13] = -600;
      Position[14] =  600;
      Position[15] = -600;

      Position[16] =  600;
      Position[17] = -600;
      Position[18] =  600;
      Position[19] = -600;
      Position[20] =  600;
      Position[21] = -600;
      Position[22] =  600;
      Position[23] = -600;

      Position[24] =  600;
      Position[25] = -600;
      Position[26] =  600;
      Position[27] = -600;
      Position[28] =  600;
      Position[29] = -600;
      Position[30] =  600;
      Position[31] = -600;

      delay_ms(1000);

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
  /* Enable the TIM3 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;
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
  GPIO_InitTypeDef GPIO_InitStructure;

  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(GPIOX_RCC , ENABLE);

  /* GPIO Configuration: output push-pull */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN | GPIOX_1_PIN | GPIOX_2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);
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
  TIM_TimeBaseStructure.TIM_Period = PWM_CYCLE;
  TIM_TimeBaseStructure.TIM_Prescaler = 79;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM2;
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


  /* TIM IT enable */
  TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);
}

/**
  * @brief  Set select pin of decoder
  * @param  None
  * @retval None
  */
void Set_Select(uint16_t Decoder_Select)
{
  GPIO_Write(GPIOX_PORT, (GPIO_ReadOutputData(GPIOX_PORT) & 0x1111111111111000) | Decoder_Select);
}
