/**
  ******************************************************************************
  * @file    freertos_controller/buzzer.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   buzzer functions
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
#include "buzzer.h"
#include "com.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define POWER_DEFAULT 50
#define RESTRICTION_RATE 100
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint8_t BuzzerPower;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for Buzzer
 * @param  None
 * @retval None
 */
void Buzzer_Configuration(void)
{
  // Configuration TIM3 for buzzer
  Buzzer_TIM3_Configuration();
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_Buzzer_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  BuzzerQueue_Type Received_Inst;
  portBASE_TYPE xStatus;
  uint16_t RepetitionCount = 0, PatternCount = 0, TimeCount = 0;

  while (1)
    {
      //Receive data from queue
      xStatus = xQueueReceive( xBuzzerQueue, &Received_Inst, 0);
      if (xStatus != pdPASS)
        {

        }
      else
        {
          PatternCount = 0;
          TimeCount = 0;
          RepetitionCount = Received_Inst.Repetition_Number;
        }

      switch(Received_Inst.Instruction)
      {
        case BUZZER_OFF:
          TIM_SetCompare3(TIM3,  0);
          break;
        case BUZZER_ON:
          TIM_SetCompare3(TIM3,  (POWER_DEFAULT * RESTRICTION_RATE) / 100);
          break;
        case BUZZER_PATTERN:
          if (RepetitionCount > 0)
            {
              if (TimeCount == 0)
                {
                  TIM_SetCompare3(TIM3,  (Received_Inst.Pattern_Type[PatternCount].Power * RESTRICTION_RATE) / 100);
                }
              TimeCount += 10;
              if (TimeCount >= Received_Inst.Pattern_Type[PatternCount].Period )
                {
                  TimeCount = 0;
                  PatternCount++;
                  if (PatternCount >= Received_Inst.Pattern_Number)
                    {
                      PatternCount = 0;
                      if(Received_Inst.Repetition_Number != 0)
                        {
                          RepetitionCount--;
                        }
                    }
                }
            }
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
void Buzzer_TIM3_Configuration(void)
{
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC, ENABLE);
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM3_CH34_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;
  /* GPIO Configuration:TIM3 Channel3 */
  GPIO_InitStructure.GPIO_Pin = TIM3_CH3_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_Init(TIM3_CH34_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM3)
PartialRemap_TIM3_Configuration();
#elif defined (FULL_REMAP_TIM3)
FullRemap_TIM3_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM3 Configuration: Output Compare Toggle Mode:
    TIM3CLK = 72 MHz, Prescaler = 7200, TIM3 counter clock = 10kHz
  ----------------------------------------------------------------------------*/
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 99;
  TIM_TimeBaseStructure.TIM_Prescaler = 7199;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  TIM_OCInitTypeDef  TIM_OCInitStructure;
  /* PWM1 Mode configuration: Channel3 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC3Init(TIM3, &TIM_OCInitStructure);
  TIM_OC3PreloadConfig(TIM3, TIM_OCPreload_Enable);

  /* TIM enable counter */
  TIM_Cmd(TIM3, ENABLE);
}
