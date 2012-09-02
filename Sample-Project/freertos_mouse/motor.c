/**
  ******************************************************************************
  * @file    freertos_mouse/motor.c
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
#include "motor.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
int8_t LSpeed = 0, RSpeed = 0;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for motors
 * @param  None
 * @retval None
 */
void Motor_Configuration(void)
{
  // GPIO for low side driver
  Low_Side_Configuration();
  // Timer for high side
  TIM3_Configuration();
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_Motor_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();
  MotorQueue_Type Received_Inst;
  portBASE_TYPE xStatus;
  int8_t LSpeed_From = 0, RSpeed_From = 0, LSpeed_To = 0, RSpeed_To = 0;
  uint16_t Period = 0, TimeCount = 0;
  MotorStatus_Type MotorStatus = STOP;

  while (1)
    {
      //Receive data from queue
      xStatus = xQueueReceive( xMotorQueue, &Received_Inst, 0);
      if (xStatus != pdPASS)
        {

        }
      else
        {
          LSpeed_From = LSpeed;
          RSpeed_From = RSpeed;
          MotorStatus = Received_Inst.Instruction;
          LSpeed_To = Received_Inst.LSpeed_To;
          RSpeed_To = Received_Inst.RSpeed_To;
          Period = Received_Inst.Period;
          TimeCount = 0;
        }

      switch(MotorStatus)
      {
        case ACCEL:
          if (TimeCount <= Period)
            {
              LSpeed = LSpeed_From + (((LSpeed_To - LSpeed_From) * TimeCount) / Period);
              RSpeed = RSpeed_From + (((RSpeed_To - RSpeed_From) * TimeCount) / Period);

              TimeCount += 10;
              SetMotorSpeed(LSpeed, RSpeed);
            }
          break;
        case BRAKE:
          break;
        case CLEAR:
          break;
        case STAY:
          break;
        case STOP:
          break;
      }
      vTaskDelayUntil(&xLastWakeTime, 10 / portTICK_RATE_MS);
    }
}

/**
 * @brief  Configure GPIO for low side driver
 * @param  None
 * @retval : None
 */
void Low_Side_Configuration(void)
{
  RCC_APB2PeriphClockCmd(GPIOY_4_RCC | GPIOY_5_RCC | GPIOY_6_RCC | GPIOY_7_RCC,
      ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_4:M1+ */
  GPIO_InitStructure.GPIO_Pin = GPIOY_4_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_4_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_5:M1- */
  GPIO_InitStructure.GPIO_Pin = GPIOY_5_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_5_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_6:M2+ */
  GPIO_InitStructure.GPIO_Pin = GPIOY_6_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_6_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_7:M2- */
  GPIO_InitStructure.GPIO_Pin = GPIOY_7_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_7_PORT, &GPIO_InitStructure);

  GPIO_ResetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  GPIO_ResetBits(GPIOY_5_PORT, GPIOY_5_PIN);
  GPIO_ResetBits(GPIOY_6_PORT, GPIOY_6_PIN);
  GPIO_ResetBits(GPIOY_7_PORT, GPIOY_7_PIN);

}

/**
 * @brief  Configure TIM3
 * @param  None
 * @retval : None
 */
void TIM3_Configuration(void)
{
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM3_RCC, ENABLE);
  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM3_CH12_GPIO_RCC | TIM3_CH34_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;
  /* GPIO Configuration:TIM3 Channel1/2 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM3_CH1_PIN | TIM3_CH2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_Init(TIM3_CH12_PORT, &GPIO_InitStructure);
  /* GPIO Configuration:TIM3 Channel1 as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = TIM3_CH3_PIN | TIM3_CH4_PIN;
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
   TIM3CLK = 72 MHz, Prescaler = 7200, TIM3 counter clock = 10KHz
   ----------------------------------------------------------------------------*/

  TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = 99;
  TIM_TimeBaseStructure.TIM_Prescaler = 7199;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);

  TIM_OCInitTypeDef TIM_OCInitStructure;
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

void SetMotorSpeed(int8_t MotorL_Speed, int8_t MotorR_Speed)
{
  if (MotorL_Speed >= 0)
    {
      //Run Motor-L Forward
      //Assure OFF of Motor-L +Low and -High
      TIM_SetCompare2(TIM3, 0);
      GPIO_ResetBits(GPIOY_4_PORT, GPIOY_4_PIN);
      //Set ON of Motor-L +High and -Low
      TIM_SetCompare1(TIM3, MotorL_Speed);
      GPIO_SetBits(GPIOY_5_PORT, GPIOY_5_PIN);
    }
  else
    {
      //Run Motor-L Backward
      //Assure OFF of Motor-L +High and -Low
      TIM_SetCompare1(TIM3, 0);
      GPIO_ResetBits(GPIOY_5_PORT, GPIOY_5_PIN);
      //Set ON of Motor-L +Low and -High
      TIM_SetCompare2(TIM3, -MotorL_Speed);
      GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);
    }
  if (MotorR_Speed >= 0)
    {
      //Run Motor-R Forward
      //Assure OFF of Motor-R +High and -Low
      TIM_SetCompare3(TIM3, 0);
      GPIO_ResetBits(GPIOY_7_PORT, GPIOY_7_PIN);
      //Set ON of Motor-R +Low and -High
      TIM_SetCompare4(TIM3, MotorR_Speed);
      GPIO_SetBits(GPIOY_6_PORT, GPIOY_6_PIN);
    }
  else
    {
      //Run Motor-R Backward
      //Assure OFF of Motor-R +Low and -High
      TIM_SetCompare4(TIM3, 0);
      GPIO_ResetBits(GPIOY_6_PORT, GPIOY_6_PIN);
      //Set ON of Motor-R +High and -Low
      TIM_SetCompare3(TIM3, -MotorR_Speed);
      GPIO_SetBits(GPIOY_7_PORT, GPIOY_7_PIN);
    }
}

