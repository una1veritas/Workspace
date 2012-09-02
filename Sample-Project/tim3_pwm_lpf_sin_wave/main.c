/**
  ******************************************************************************
  * @file    tim3_pwm_lpf_sin_wave/main.c
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
  "Monitor voltage at TIM3_CH1 filtered by LPF. \r\n"
  "You can make the voltage up and down by pressing U or D. \r\n";
__I uint16_t sin_16bit_360deg[] = {
    32768,33339,33911,34482,35053,35623,36193,36761,37328,37894,
    38458,39020,39580,40139,40695,41248,41800,42348,42893,43436,
    43975,44511,45043,45571,46095,46616,47132,47644,48151,48654,
    49152,49644,50132,50614,51091,51562,52028,52488,52941,53389,
    53830,54265,54694,55115,55530,55938,56339,56732,57119,57498,
    57869,58233,58589,58937,59277,59609,59933,60249,60556,60855,
    61145,61427,61700,61964,62219,62465,62703,62931,63149,63359,
    63559,63750,63932,64104,64266,64419,64562,64696,64819,64933,
    65038,65132,65217,65291,65356,65411,65456,65491,65516,65531,
    65535,65531,65516,65491,65456,65411,65356,65291,65217,65132,
    65038,64933,64819,64696,64562,64419,64266,64104,63932,63750,
    63559,63359,63149,62931,62703,62465,62219,61964,61700,61427,
    61145,60855,60556,60249,59933,59609,59277,58937,58589,58233,
    57869,57498,57119,56732,56339,55938,55530,55115,54694,54265,
    53830,53389,52941,52488,52028,51562,51091,50614,50132,49644,
    49152,48654,48151,47644,47132,46616,46095,45571,45043,44511,
    43975,43436,42893,42348,41800,41248,40695,40139,39580,39020,
    38458,37894,37328,36761,36193,35623,35053,34482,33911,33339,
    32768,32197,31625,31054,30483,29913,29343,28775,28208,27642,
    27078,26516,25956,25397,24841,24288,23736,23188,22643,22100,
    21561,21025,20493,19965,19441,18920,18404,17892,17385,16882,
    16384,15892,15404,14922,14445,13974,13508,13048,12595,12147,
    11706,11271,10842,10421,10006,9598,9197,8804,8417,8038,7667,
    7303,6947,6599,6259,5927,5603,5287,4980,4681,4391,4109,3836,
    3572,3317,3071,2833,2605,2387,2177,1977,1786,1604,1432,1270,
    1117,974,840,717,603,498,404,319,245,180,125,80,45,20,5,0,5,
    20,45,80,125,180,245,319,404,498,603,717,840,974,1117,1270,
    1432,1604,1786,1977,2177,2387,2605,2833,3071,3317,3572,3836,
    4109,4391,4681,4980,5287,5603,5927,6259,6599,6947,7303,7667,
    8038,8417,8804,9197,9598,10006,10421,10842,11271,11706,
    12147,12595,13048,13508,13974,14445,14922,15404,15892,16384,
    16882,17385,17892,18404,18920,19441,19965,20493,21025,21561,
    22100,22643,23188,23736,24288,24841,25397,25956,26516,27078,
    27642,28208,28775,29343,29913,30483,31054,31625,32197,32768};
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void TIM3_Configuration(void);
void TIM4_Configuration(void);
void TIM4_IRQHandler(void);
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

  /* TIM3 Configuration*/
  TIM3_Configuration();
  /* TIM4 Configuration*/
  TIM4_Configuration();

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
  /* Enable the TIM4 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
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
void TIM3_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_OCInitTypeDef  TIM_OCInitStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC , ENABLE);
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM3_CH12_GPIO_RCC , ENABLE);

  /* GPIO Configuration:TIM3 Channel1 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM3_CH1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(TIM3_CH12_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM3)
PartialRemap_TIM3_Configuration();
#elif defined (FULL_REMAP_TIM3)
FullRemap_TIM3_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM3 Configuration: Output Compare Toggle Mode:
    TIM3CLK = 72 MHz, Prescaler = 1, TIM3 counter clock = 72MHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = PWM_COUNT;
  TIM_TimeBaseStructure.TIM_Prescaler = 1;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC1Init(TIM3, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM3, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);
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
    TIM4CLK = 72 MHz, Prescaler = 2250, TIM4 counter clock = 32kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = (uint16_t)((SystemCoreClock /2 ) * 2 / Sample_Rate);
  TIM_TimeBaseStructure.TIM_Prescaler = 0;
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
  * @brief  This function handles TIM4 update interrupt request.
  * @param  None
  * @retval None
  */
void TIM4_IRQHandler(void)
{
  if (TIM_GetITStatus(TIM4, TIM_IT_Update) != RESET)
  {
    /* Clear TIM4 update interrupt pending bit*/
    TIM_ClearITPendingBit(TIM4, TIM_IT_Update);

    CC1_Value = sin_16bit_360deg[((uint32_t)Counter)*360 / Period] >> 8;

    TIM_SetCompare1(TIM3,CC1_Value);

    Counter++;
    if (Counter >= Period)
      {
        Counter = 0;
      }
  }
}
