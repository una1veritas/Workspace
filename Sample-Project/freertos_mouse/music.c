/**
  ******************************************************************************
  * @file    freertos_mouse/music.c
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
#include "music.h"
#include "com.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define NOTE_RATIO 95
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const uint16_t Note_Freq[] =
  {
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    45867,43293,40863,38569,36405,34361,32433,30613,28894,27273,25742,24297,
    22934,21646,20431,19285,18202,17181,16216,15306,14447,13636,12871,12149,
    11467,10823,10216,9642,9101,8590,8108,7653,7224,6818,6436,6074,
    5733,5412,5108,4821,4551,4295,4054,3827,3612,3409,3218,3037,
    2867,2706,2554,2411,2275,2148,2027,1913,1806,1705,1609,1519,
    1433,1353,1277,1205,1138,1074,1014,957,903,804,804,759
  };
const uint8_t Note_Conv[] = { 9,11,0,2,4,5,7 };
const uint8_t Period_Conv[] = { 0,32,16,1,8,0,2,0,4,0 };

int8_t MusicHetro[] = "T240CDCD";
int8_t MusicInitial[] = "T240CDEF2";
int8_t MusicAuto[] = "T120O5CCGGAAG2FFEEDDC2T240O6CCGGAAG2FFEEDDC2";
//int8_t TestMusic[] = "C2E";
//int8_t TestMusic[] = "T180O5C4R4G4R4";
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for music
 * @param  None
 * @retval None
 */
void Music_Configuration(void)
{
  // Configuration TIM4 for Music
  TIM4_Configuration();
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_Music_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  TIM_OCInitTypeDef  TIM_OCInitStructure;

  portBASE_TYPE xStatus;
  MusicQueue_Type Received_Inst;
  uint16_t Tempo = 120, PointerCount = 0 , Octave = 5 , NoteSub, PeriodSub;
  uint32_t Period = 0;
  uint8_t MusicStatus;

  while (1)
    {
      //Receive data from queue
      xStatus = xQueueReceive( xMusicQueue, &Received_Inst, 0);
      if (xStatus != pdPASS)
        {

        }
      else
        {
          MusicStatus = Received_Inst.Instruction;
          switch(MusicStatus)
          {
            case MUSIC_PLAY:
              /* Output Compare Inactive Mode configuration: Channel3 */
              TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Toggle;
              TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
              TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
              TIM_OCInitStructure.TIM_Pulse = 0;
              TIM_OC3Init(TIM4, &TIM_OCInitStructure);
              PointerCount = 0;
              break;
            case MUSIC_OFF:
              /* Output Compare Inactive Mode configuration: Channel3 */
              TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Toggle;
              TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
              TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
              TIM_OCInitStructure.TIM_Pulse = 0;
              TIM_OC3Init(TIM4, &TIM_OCInitStructure);
              break;
          }

        }

      switch(MusicStatus)
      {
        case MUSIC_PLAY:

          // tempo
          if (*(Received_Inst.MML+PointerCount) == 'T')
            {
              Tempo =
                (*(Received_Inst.MML+PointerCount + 1) - 0x30 ) * 100 +
                (*(Received_Inst.MML+PointerCount + 2) - 0x30 ) * 10 +
                (*(Received_Inst.MML+PointerCount + 3) - 0x30 ) * 1      ;
              PointerCount += 4;
            }

          // octave
          if (*(Received_Inst.MML+PointerCount) == 'O')
            {
              Octave = *(Received_Inst.MML+PointerCount + 1) - 0x30;
              PointerCount += 2;
            }

          // rest
          if (*(Received_Inst.MML+PointerCount) == 'R')
            {
              NoteSub = 0;
            }
          // note
          else if(*(Received_Inst.MML+PointerCount) >= 'A' && *(Received_Inst.MML+PointerCount) <= 'G')
            {
              NoteSub = (Octave * 12) + Note_Conv[(uint8_t)(*(Received_Inst.MML+PointerCount)-0x41)];

              // sharp and flat
              if (*(Received_Inst.MML+PointerCount+1) == '+')
                {
                  NoteSub++;
                  PointerCount++;
                }
              if (*(Received_Inst.MML+PointerCount+1) == '-')
                {
                  NoteSub--;
                  PointerCount++;
                }
            }

          // note and rest length
          if (*(Received_Inst.MML+PointerCount+1) >= '0' && *(Received_Inst.MML+PointerCount+1) <= '9')
            {
              PeriodSub = (uint16_t)(*(Received_Inst.MML+PointerCount+1) - 0x30);
              PointerCount++;
            }
          else
            {
              PeriodSub = 4;
            }

          Period = 60 * 1000 * Period_Conv[PeriodSub] / 8 / Tempo;

          TIM_SetAutoreload(TIM4, Note_Freq[NoteSub] - 1);

          vTaskDelayUntil(&xLastWakeTime, (Period * NOTE_RATIO) / 100 / portTICK_RATE_MS);

          TIM_SetAutoreload(TIM4, 0);
          vTaskDelayUntil(&xLastWakeTime, (Period * (100- NOTE_RATIO)) / 100 / portTICK_RATE_MS);

          PointerCount++;
          if (*(Received_Inst.MML+PointerCount) == '\0')
            {
              PointerCount = 0;
              if (Received_Inst.Repeat == 0)
                {
                  MusicStatus = MUSIC_OFF;
                  /* Output Compare Inactive Mode configuration: Channel3 */
                  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Toggle;
                  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Disable;
                  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
                  TIM_OCInitStructure.TIM_Pulse = 0;
                  TIM_OC3Init(TIM4, &TIM_OCInitStructure);
                }
            }
          break;
        case MUSIC_OFF:
          vTaskDelayUntil(&xLastWakeTime, 100 / portTICK_RATE_MS);

          break;
      }

    }
}

/**
 * @brief  Configure TIM4
 * @param  None
 * @retval : None
 */
void TIM4_Configuration(void)
{
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM4_RCC , ENABLE);

  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(TIM4_GPIO_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;
  /* Configure GPIO for TIM4_CH3 */
  GPIO_InitStructure.GPIO_Pin = TIM4_CH3_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_Init(TIM4_PORT, &GPIO_InitStructure);

  /* ---------------------------------------------------------------------------
    TIM4 Configuration: Output Compare Toggle Mode:
    TIM4CLK = 72 MHz, Prescaler = 5, TIM4 counter clock = 12MHz
  ----------------------------------------------------------------------------*/

  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 0;
  TIM_TimeBaseStructure.TIM_Prescaler = 4;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);

  TIM_OCInitTypeDef  TIM_OCInitStructure;
  /* Output Compare Inactive Mode configuration: Channel3 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_Toggle;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OC3Init(TIM4, &TIM_OCInitStructure);
  TIM_OC3PreloadConfig(TIM4, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM4, ENABLE);
}

