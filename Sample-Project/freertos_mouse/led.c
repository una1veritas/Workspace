/**
  ******************************************************************************
  * @file    freertos_mouse/led.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   motor functions
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
#include "led.h"
#include "com.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint8_t Red_Luminance = 0, Green_Luminance = 0, Blue_Luminance = 0;
const LEDPattern_Type LED_PatternInitiating[] =
  {
      {0,0,0,0,30,0,200},
      {0,30,0,0,0,0,200},
      {0,0,0,0,0,0,0}
  };
const LEDPattern_Type LED_PatternConnected[] =
  {
      {0,0,0,0,30,0,600},
      {0,30,0,0,30,0,400},
      {0,30,0,0,0,0,600},
      {0,0,0,0,0,0,0}
  };

/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for LED
 * @param  None
 * @retval None
 */
void LED_Configuration(void)
{
  // Configuration TIM1 for LED
  TIM1_Configuration();
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_LED_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  static LEDQueue_Type Received_Inst;
  portBASE_TYPE xStatus;
  uint16_t TimeCount = 0, PatternCount = 0;

  while (1)
    {
      //Receive data from queue
      xStatus = xQueueReceive( xLEDQueue, &Received_Inst, 0);
      if (xStatus != pdPASS)
        {

        }
      else
        {
          TimeCount = 0;
          PatternCount = 0;
        }

      switch(Received_Inst.Instruction)
      {
        case GRADUAL:
          if (TimeCount <= Received_Inst.Pattern_Type[PatternCount].Period)
            {
              Red_Luminance =
                Received_Inst.Pattern_Type[PatternCount].Red_From +
                (((Received_Inst.Pattern_Type[PatternCount].Red_To -
                   Received_Inst.Pattern_Type[PatternCount].Red_From) *
                    TimeCount ) / Received_Inst.Pattern_Type[PatternCount].Period);
              Green_Luminance =
                Received_Inst.Pattern_Type[PatternCount].Green_From +
                (((Received_Inst.Pattern_Type[PatternCount].Green_To -
                    Received_Inst.Pattern_Type[PatternCount].Green_From) *
                    TimeCount ) / Received_Inst.Pattern_Type[PatternCount].Period);
              Blue_Luminance =
                Received_Inst.Pattern_Type[PatternCount].Blue_From +
                (((Received_Inst.Pattern_Type[PatternCount].Blue_To -
                    Received_Inst.Pattern_Type[PatternCount].Blue_From) *
                    TimeCount ) / Received_Inst.Pattern_Type[PatternCount].Period);

              TimeCount += 10;
              if(TimeCount >= Received_Inst.Pattern_Type[PatternCount].Period)
                {
                  TimeCount = 0;
                  PatternCount++;
                  if (Received_Inst.Pattern_Type[PatternCount].Period == 0)
                    {
                      PatternCount = 0;
                    }
                }
              TIM_SetCompare1(TIM1, Green_Luminance);
              TIM_SetCompare2(TIM1, Blue_Luminance);
              TIM_SetCompare3(TIM1, Red_Luminance);
            }
          break;
        case LED_OFF:
          break;
        case LED_ON:
          break;
      }
      vTaskDelayUntil(&xLastWakeTime, 10 / portTICK_RATE_MS);
    }
}

/**
 * @brief  Configure TIM3
 * @param  None
 * @retval : None
 */
void TIM1_Configuration(void)
{
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM1_RCC | TIM1_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;
  /* GPIO Configuration:TIM1 Channel1, 2 and 3 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM1_CH1_PIN | TIM1_CH2_PIN | TIM1_CH3_PIN ;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_Init(TIM1_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM1)
PartialRemap_TIM1_Configuration();
#elif defined (FULL_REMAP_TIM1)
FullRemap_TIM1_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM1 Configuration: Output Compare Toggle Mode:
    TIM1CLK = 72 MHz, Prescaler = 1440, TIM1 counter clock = 50kHz
  ----------------------------------------------------------------------------*/
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 99;
  TIM_TimeBaseStructure.TIM_Prescaler = 1439;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);

  TIM_OCInitTypeDef  TIM_OCInitStructure;
  /* PWM1 Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
  TIM_OCInitStructure.TIM_OCIdleState = TIM_OCIdleState_Reset;
  TIM_OCInitStructure.TIM_OutputNState = TIM_OutputNState_Disable;
  TIM_OCInitStructure.TIM_OCNPolarity = TIM_OCNPolarity_High;
  TIM_OCInitStructure.TIM_OCNIdleState = TIM_OCIdleState_Reset;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC1Init(TIM1, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM1, TIM_OCPreload_Enable);

  /* PWM1 Mode configuration: Channel2 */
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC2Init(TIM1, &TIM_OCInitStructure);
  TIM_OC2PreloadConfig(TIM1, TIM_OCPreload_Enable);

  /* PWM1 Mode configuration: Channel3 */
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC3Init(TIM1, &TIM_OCInitStructure);
  TIM_OC3PreloadConfig(TIM1, TIM_OCPreload_Enable);

  /* TIM enable counter */
  TIM_Cmd(TIM1, ENABLE);

  /* Main Output Enable */
  TIM_CtrlPWMOutputs(TIM1, ENABLE);



}

