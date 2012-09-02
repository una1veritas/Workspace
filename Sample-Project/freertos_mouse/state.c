/**
  ******************************************************************************
  * @file    freertos_mouse/state.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   state functions
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
#include "state.h"
#include "motor.h"
#include "com.h"
#include "button.h"
#include "led.h"
#include "psd.h"
#include "sonic.h"
#include "music.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define STACK_SIZE 200
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint16_t PresentState;
uint16_t TaskEventTable[][3]  =
  {
      {0,1,1}, // 0
      {1,2,1}, // 1
      {2,1,2}, // 2
  };
StateTaskType StateTaskArray[] =
  {
      {State_Initial,     (signed portCHAR *)"State:Initial", 0}, // 0
      {State_Hetro,       (signed portCHAR *)"State:Hetero",  0}, // 1
      {State_Auto,        (signed portCHAR *)"State:Auto",    0}, // 2
  };

const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 and FreeRTOS World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_State_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;

  portBASE_TYPE xStatus;

  PresentState = 0;
  xTaskCreate(StateTaskArray[PresentState].StateTask, StateTaskArray[PresentState].TaskName , STACK_SIZE, NULL, 1, &StateTaskArray[PresentState].TaskHandle);

  EventQueue_Type Received_Inst;

  while (1)
    {
      //Receive data from queue
      xStatus = xQueueReceive( xEventQueue, &Received_Inst, portMAX_DELAY);
      if (xStatus != pdPASS)
        {
          // do nothing
        }
      else
        {
          // delete current task first to reduce memory consumption
          vTaskDelete(StateTaskArray[PresentState].TaskHandle);
          PresentState = TaskEventTable[PresentState][Received_Inst.Event];
          while(1)
            {
              // sometimes it fails to create task due to out of memory
              // free rtos free memory while idle task is running
              // if it fails to create, it will be possible in due course
              xStatus = xTaskCreate(StateTaskArray[PresentState].StateTask, StateTaskArray[PresentState].TaskName , STACK_SIZE, NULL, 1, &StateTaskArray[PresentState].TaskHandle);
              if (xStatus != pdPASS)
                {
                  vTaskDelay(10 / portTICK_RATE_MS);
                }
              else
                {
                  break;
                }
            }
        }
    }
}


/**
 * @brief  function to change status sending queue
 * @param  event : event occurred
 * @retval None
 */
void State_Event(uint16_t event)
{
  EventQueue_Type eventoccured;
  eventoccured.Event = event;
  xQueueSendToBack(xEventQueue, &eventoccured, 0);
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_Initial(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  COM_Wired_PrintString(Welcome_Message);

  COM_Wired_PrintString("Initial task called\r\n");

  MusicQueue_Type Music_Inst;
  Music_Inst.Instruction = MUSIC_PLAY;
  Music_Inst.MML = MusicInitial;
  Music_Inst.Repeat = 0;
  xQueueSendToBack(xMusicQueue, &Music_Inst, 0);


  while(1)
    {
      vTaskDelayUntil(&xLastWakeTime, 500 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_Hetro(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  portBASE_TYPE xStatus;

  COMQueue_Type Received_Inst;
  uint16_t instruction;
  int8_t lspeed, rspeed;

  MusicQueue_Type Music_Inst;
  Music_Inst.Instruction = MUSIC_PLAY;
  Music_Inst.MML = MusicHetro;
  Music_Inst.Repeat = 0;
  xQueueSendToBack(xMusicQueue, &Music_Inst, 0);

  MotorQueue_Type Motor_Inst;
  Motor_Inst.Instruction = ACCEL;
  Motor_Inst.LSpeed_To = 0;
  Motor_Inst.RSpeed_To = 0;
  Motor_Inst.Period = 10;
  xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);

  COM_Wired_PrintString("Radio control / Heteronomy mode\r\n");

  while(1)
    {
      //Receive instruction from controller
      xStatus = xQueueReceive( xCOMQueue, &Received_Inst, 0);
      if (xStatus != pdPASS)
        {

        }
      else
        {
          instruction = (uint16_t)(Received_Inst.Parameter>>16);
          lspeed = (int8_t)(Received_Inst.Parameter>>0);
          rspeed = (int8_t)(Received_Inst.Parameter>>8);

          if (instruction == 1)
            {
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = lspeed;
              Motor_Inst.RSpeed_To = rspeed;
              Motor_Inst.Period = 100;
              xStatus = xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
            }

        }

      // send sensor information to controller
      COM_XBee_PrintFormatted("#%8X\r",0x00050000 | DistSonic);
      COM_XBee_PrintFormatted("#%8X\r",0x00060000 | DistPSDR);
      COM_XBee_PrintFormatted("#%8X\r",0x00070000 | DistPSDL);

      vTaskDelayUntil(&xLastWakeTime, 100 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_Auto(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  uint8_t autostatus = 0, laststatus = 20;

  MotorQueue_Type Motor_Inst;

  COM_Wired_PrintString("Motor test mode\r\n");

  MusicQueue_Type Music_Inst;
  Music_Inst.Instruction = MUSIC_PLAY;
  Music_Inst.MML = MusicAuto;
  Music_Inst.Repeat = 1;
  xQueueSendToBack(xMusicQueue, &Music_Inst, 0);

  typedef enum {
    AUTO_CLEAR = 0,
    AUTO_NEAR = 1,
    AUTO_R_CLOSE = 2,
    AUTO_L_CLOSE = 3,
    AUTO_F_CLOSE = 4,
    AUTO_RL_CLOSE = 5,
    AUTO_RLF_CLOSE = 6
  } Auto_Status_Type;


  while(1)
    {
      autostatus = laststatus;

      if(DistSonic < 9 && DistPSDR > 15 && DistPSDL > 15)
        {
          autostatus = AUTO_RLF_CLOSE;
        }
      else if (DistPSDR > 15 && DistPSDL > 15)
        {
          autostatus = AUTO_RL_CLOSE;
        }
      else if (DistPSDL > 15)
        {
           autostatus = AUTO_L_CLOSE;
        }
      else if (DistPSDR > 15)
        {
           autostatus = AUTO_R_CLOSE;
        }
      else if (DistSonic < 20)
        {
           autostatus = AUTO_F_CLOSE;
        }
      else if (DistSonic < 20 || DistPSDR > 8 || DistPSDL > 8)
        {
           autostatus = AUTO_NEAR;
        }
      else
        {
          autostatus = AUTO_CLEAR;
        }

      if (autostatus != laststatus)
        {
          switch(autostatus)
          {
            case AUTO_CLEAR:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = 50;
              Motor_Inst.RSpeed_To = 50;
              Motor_Inst.Period = 1000;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
            case AUTO_NEAR:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = 20;
              Motor_Inst.RSpeed_To = 20;
              Motor_Inst.Period = 1000;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
            case AUTO_F_CLOSE:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = 20;
              Motor_Inst.RSpeed_To = -20;
              Motor_Inst.Period = 100;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
            case AUTO_L_CLOSE:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = 20;
              Motor_Inst.RSpeed_To = -10;
              Motor_Inst.Period = 100;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
            case AUTO_R_CLOSE:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = -10;
              Motor_Inst.RSpeed_To = 20;
              Motor_Inst.Period = 100;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
            case AUTO_RL_CLOSE:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = 20;
              Motor_Inst.RSpeed_To = 20;
              Motor_Inst.Period = 100;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
            case AUTO_RLF_CLOSE:
              Motor_Inst.Instruction = ACCEL;
              Motor_Inst.LSpeed_To = -20;
              Motor_Inst.RSpeed_To = 10;
              Motor_Inst.Period = 100;
              xQueueSendToBack(xMotorQueue, &Motor_Inst, 0);
              break;
          }
        }

      vTaskDelay(100 / portTICK_RATE_MS);

    }
}

