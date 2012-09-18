/**
  ******************************************************************************
  * @file    freertos_controller/state.c
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
#include "com.h"
#include "lcd.h"
#include "shift.h"
#include "buzzer.h"
#include "joystick.h"
#include "statusled.h"
#include "vibrator.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define STACK_SIZE 200
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint16_t PresentState;
uint16_t TaskEventTable[][5]  =
  {
      {0,1,7,1,1}, // 0
      {1,2,7,2,2}, // 1
      {2,3,1,3,3}, // 2
      {3,4,2,3,3}, // 3
      {4,5,3,3,3}, // 4
      {5,6,4,3,3}, // 5
      {6,7,5,3,3}, // 6
      {7,1,6,3,3}  // 7
  };
StateTaskType StateTaskArray[] =
  {
      {State_Initial,       (signed portCHAR *)"State:Initial",     0}, // 0
      {State_MouseTSControl,(signed portCHAR *)"State:Mouse TS",    0}, // 1
      {State_BuzzerTest,    (signed portCHAR *)"State:Buzzer te",   0}, // 2
      {State_JoystickTest,  (signed portCHAR *)"State:Joystick ",   0}, // 3
      {State_StatusLEDTest, (signed portCHAR *)"State:Status LE",   0}, // 4
      {State_LCDTest,       (signed portCHAR *)"State:LCD test",    0}, // 5
      {State_VibratorTest,  (signed portCHAR *)"State:Vibrator ",   0}, // 6
      {State_WiredCOMTest,  (signed portCHAR *)"State:Wired COM",   0}  // 7

  };

const BuzzerPattern_Type BuzzerPattern1[] =
  {
      {50,10},
      {0,400}
  };

const VibratorPattern_Type VibratorPattern1[] =
  {
      {30,200},
      {0,400},
      {0,0}
  };
const VibratorPattern_Type VibratorPattern2[] =
  {
      {60,200},
      {0,400},
      {0,0}
  };

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
          vTaskDelete(StateTaskArray[PresentState].TaskHandle);
          PresentState = TaskEventTable[PresentState][Received_Inst.Event];
          while(1)
            {
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

  const int8_t Welcome_Message[] =
    "\r\nHellow Cortex-M3/STM32 and FreeRTOS World!\r\n"
    "Expand your creativity and enjoy making.\r\n\r\n";
  COM_Wired_PrintString(Welcome_Message);
  COM_Wired_PrintString("Initial task called\r\n");
  LCD_CursorOFF();
  LCD_Clear();
  LCD_PrintString(0x0, "Mouse controller");
  LCD_PrintString(0x40,"C/D:mode change ");

  while(1)
    {
      vTaskDelay(200 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_MouseTSControl(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  COMQueue_Type COM_Info;
  VibratorQueue_Type Vibrator_Inst;

  uint8_t Pre_L_X, Pre_L_Y, Pre_R_X, Pre_R_Y, vibratorflag = 1;
  int16_t rpower, lpower , psdl = 0, psdr = 0, sonic = 100;

  LCD_Clear();
  LCD_PrintString(0x0, "Mouse Control    ");
  LCD_PrintString(0x40,"Throttle/Steering");
  COM_Wired_PrintString("Mouse control:Throttle/Steering control mode\r\n");

  uint32_t instruction;

  Pre_L_X = Joystick_L_X;
  Pre_L_Y = Joystick_L_Y;
  Pre_R_X = Joystick_R_X;
  Pre_R_Y = Joystick_R_Y;

  while(1)
    {
      if(
          Joystick_L_X != 0  ||
          Joystick_L_Y != 0  ||
          Joystick_R_X != 0  ||
          Joystick_R_Y != 0
          )
        {
          LCD_Clear();
          LCD_PrintString(0x00,"S:    L:   R:  ");
          break;
        }
    }

  while(1)
    {
      while (xQueueReceive( xCOMQueue, &COM_Info, 0) == pdPASS)
        {
          switch(COM_Info.Parameter >> 16)
          {
            case 5:
              LCD_PrintDecimal(0x2, COM_Info.Parameter & 0x0000FFFF, 3, 0 );
              sonic =  COM_Info.Parameter & 0xFFFF;
              break;
            case 6:
              LCD_PrintDecimal(0x8, COM_Info.Parameter & 0x0000FFFF, 2, 0 );
              psdl = COM_Info.Parameter & 0xFFFF;
              break;
            case 7:
              LCD_PrintDecimal(0xd, COM_Info.Parameter & 0x0000FFFF, 2, 0 );
              psdr = COM_Info.Parameter & 0xFFFF;
              break;
          }
        }

      if ((sonic < 10 || psdl > 12 || psdr > 12) && vibratorflag == 0)
        {
          Vibrator_Inst.Instruction = VIBRATOR_PATTERN;
          Vibrator_Inst.Pattern_Number = 2;
          Vibrator_Inst.Repetition_Number = 3;
          Vibrator_Inst.Pattern_Type = (VibratorPattern_Type*)VibratorPattern1;
          xQueueSendToBack(xVibratorQueue, &Vibrator_Inst, 0);

          vibratorflag = 1;
        }
      if ((sonic >= 10 && psdl <= 12 && psdr <= 12) && vibratorflag == 1)
        {
          vibratorflag = 0;
        }

      if (
          Pre_L_X != Joystick_L_X ||
          Pre_L_Y != Joystick_L_Y ||
          Pre_R_X != Joystick_R_X ||
          Pre_R_Y != Joystick_R_Y
          )
        {
          rpower = -Joystick_L_Y;
          lpower = -Joystick_L_Y;
          if (Joystick_R_X > 0)
            {
              rpower = (rpower * (100 - Joystick_R_X)) / 100;
            }
          else if (Joystick_R_X < 0)
            {
              lpower = (lpower * (100 + Joystick_R_X)) / 100;
            }
          instruction = 0x00010000 | ((((uint32_t)rpower) << 8) & 0x0000FF00) | ((((uint32_t)lpower) << 0) & 0x000000FF);
          COM_XBee_PrintFormatted("#%8X\r",instruction);
        }

      Pre_L_X = Joystick_L_X;
      Pre_L_Y = Joystick_L_Y;
      Pre_R_X = Joystick_R_X;
      Pre_R_Y = Joystick_R_Y;

      vTaskDelay(100 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_BuzzerTest(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();
  portBASE_TYPE xStatus;
  ButtonQueue_Type ButtonChangeInfo;
  BuzzerQueue_Type Buzzer_Inst;

  LCD_Clear();
  LCD_PrintString(0x0, "Buzzer test mode");
  LCD_PrintString(0x40,"7:ON 8:OFF A:bep");
  COM_Wired_PrintString("Buzzer test mode\r\n");

  while(1)
    {
      xStatus = xQueueReceive( xButtonQueue, &ButtonChangeInfo, portMAX_DELAY);
      if (xStatus != pdPASS)
        {

        }
      else
        {
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x7)
            {
              Buzzer_Inst.Instruction = BUZZER_ON;
              xStatus = xQueueSendToBack(xBuzzerQueue, &Buzzer_Inst, 0);
            }

          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x8)
            {
              Buzzer_Inst.Instruction = BUZZER_OFF;
              xStatus = xQueueSendToBack(xBuzzerQueue, &Buzzer_Inst, 0);
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xA)
            {
              Buzzer_Inst.Instruction = BUZZER_PATTERN;
              Buzzer_Inst.Pattern_Number = 2;
              Buzzer_Inst.Repetition_Number = 2;
              Buzzer_Inst.Pattern_Type = BuzzerPattern1;
              xStatus = xQueueSendToBack(xBuzzerQueue, &Buzzer_Inst, 0);
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xB)
            {
              Buzzer_Inst.Instruction = BUZZER_PATTERN;
              Buzzer_Inst.Pattern_Number = 2;
              Buzzer_Inst.Repetition_Number = 100;
              Buzzer_Inst.Pattern_Type = BuzzerPattern1;
              xStatus = xQueueSendToBack(xBuzzerQueue, &Buzzer_Inst, 0);
            }

          }
      vTaskDelay(100 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_JoystickTest(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  uint8_t Pre_L_X, Pre_L_Y, Pre_R_X, Pre_R_Y;

  LCD_Clear();
  LCD_PrintString(0x0, "Joystick test :");
  LCD_PrintString(0x40,"Tilt joysticks.  ");
  COM_Wired_PrintString("Joystick test mode\r\n");

  Pre_L_X = Joystick_L_X;
  Pre_L_Y = Joystick_L_Y;
  Pre_R_X = Joystick_R_X;
  Pre_R_Y = Joystick_R_Y;

  while(1)
    {
      if(
          Joystick_L_X != 0  ||
          Joystick_L_Y != 0  ||
          Joystick_R_X != 0  ||
          Joystick_R_Y != 0
          )
        {
          LCD_Clear();
          break;
        }
    }

  while(1)
    {
      if (
          Pre_L_X != Joystick_L_X ||
          Pre_L_Y != Joystick_L_Y ||
          Pre_R_X != Joystick_R_X ||
          Pre_R_Y != Joystick_R_Y
          )
        {
          LCD_PrintString(0x0, "LX:");
          LCD_PrintDecimal(0x3, Joystick_L_X, 3, 1);
          LCD_PrintString(0x40,"LY:");
          LCD_PrintDecimal(0x43, Joystick_L_Y, 3, 1);
          LCD_PrintString(0x9, "RX:");
          LCD_PrintDecimal(0xC, Joystick_R_X, 3, 1);
          LCD_PrintString(0x49,"RY:");
          LCD_PrintDecimal(0x4C, Joystick_R_Y, 3, 1);
        }

      Pre_L_X = Joystick_L_X;
      Pre_L_Y = Joystick_L_Y;
      Pre_R_X = Joystick_R_X;
      Pre_R_Y = Joystick_R_Y;

      vTaskDelay(200 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_StatusLEDTest(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  ButtonQueue_Type ButtonChangeInfo;
  portBASE_TYPE xStatus;

  StatusLEDQueue_Type StatusLED_Inst;
  uint8_t LEDStatus[3];
  LEDStatus[0] = 0;
  LEDStatus[1] = 0;
  LEDStatus[2] = 0;

  LCD_Clear();
  LCD_PrintString(0x0, "LED test 7:LED-1");
  LCD_PrintString(0x40,"8:LED-2 A:LED-3");
  COM_Wired_PrintString("LED Test mode\r\n");

  while(1)
    {
      xStatus = xQueueReceive( xButtonQueue, &ButtonChangeInfo, portMAX_DELAY);
      if (xStatus != pdPASS)
        {
          //do nothing
        }
      else
        {
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED)
            {
              if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x7)
                {
                  StatusLED_Inst.LEDNumber = 0;
                  LEDStatus[0]++;
                  if (LEDStatus[0] > STATUSLED_FLASH) LEDStatus[0] = STATUSLED_OFF;
                  StatusLED_Inst.LEDStatus = LEDStatus[0];
                  xStatus = xQueueSendToBack(xStatusLEDQueue, &StatusLED_Inst, 0);
                }
              if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x8)
                {
                  StatusLED_Inst.LEDNumber = 1;
                  LEDStatus[1]++;
                  if (LEDStatus[1] > STATUSLED_FLASH) LEDStatus[1] = STATUSLED_OFF;
                  StatusLED_Inst.LEDStatus = LEDStatus[1];
                  xStatus = xQueueSendToBack(xStatusLEDQueue, &StatusLED_Inst, 0);
                }
              if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xA)
                {
                  StatusLED_Inst.LEDNumber = 2;
                  LEDStatus[2]++;
                  if (LEDStatus[2] > STATUSLED_FLASH) LEDStatus[2] = STATUSLED_OFF;
                  StatusLED_Inst.LEDStatus = LEDStatus[2];
                  xStatus = xQueueSendToBack(xStatusLEDQueue, &StatusLED_Inst, 0);
                }
            }
        }
      vTaskDelay(100 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_LCDTest(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  ButtonQueue_Type ButtonChangeInfo;
  portBASE_TYPE xStatus;

  LCD_Clear();
  LCD_PrintString(0x0, "LCD test 7:clear");
  LCD_PrintString(0x40,"8:Of A:ON B:Blk");
  COM_Wired_PrintString("LCD test mode\r\n");

  while(1)
    {
      xStatus = xQueueReceive( xButtonQueue, &ButtonChangeInfo, portMAX_DELAY);

      if (xStatus != pdPASS)
        {
          //do nothing
        }
      else
        {
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x7)
            {
              LCD_Clear();
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x8)
            {
              LCD_CursorOFF();
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xA)
            {
              LCD_CursorON();
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xB)
            {
              LCD_CursorBlink();
            }
        }
      vTaskDelay(200 / portTICK_RATE_MS);
    }
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_VibratorTest(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  ButtonQueue_Type ButtonChangeInfo;
  portBASE_TYPE xStatus;

  VibratorQueue_Type Vibrator_Inst;

  LCD_Clear();
  LCD_PrintString(0x0, "Vib test 7:lo");
  LCD_PrintString(0x40,"8:hi A:Off B:on");
  COM_Wired_PrintString("Vibrator test mode\r\n");

  while(1)
    {
      xStatus = xQueueReceive( xButtonQueue, &ButtonChangeInfo, portMAX_DELAY);
      if (xStatus != pdPASS)
        {
          //do nothing
        }
      else
        {
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x7)
            {
              Vibrator_Inst.Instruction = VIBRATOR_PATTERN;
              Vibrator_Inst.Pattern_Number = 2;
              Vibrator_Inst.Repetition_Number = 10;
              Vibrator_Inst.Pattern_Type = (VibratorPattern_Type*)VibratorPattern1;
              xStatus = xQueueSendToBack(xVibratorQueue, &Vibrator_Inst, 0);
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0x8)
            {

              Vibrator_Inst.Instruction = VIBRATOR_PATTERN;
              Vibrator_Inst.Pattern_Number = 2;
              Vibrator_Inst.Repetition_Number = 2;
              Vibrator_Inst.Pattern_Type = (VibratorPattern_Type*)VibratorPattern2;
              xStatus = xQueueSendToBack(xVibratorQueue, &Vibrator_Inst, 0);
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xA)
            {
              Vibrator_Inst.Instruction = VIBRATOR_OFF;
              xStatus = xQueueSendToBack(xVibratorQueue, &Vibrator_Inst, 0);
            }
          if (ButtonChangeInfo.ButtonChange == BUTTON_PRESSED && ButtonChangeInfo.ButtonNumber == 0xB)
            {
              Vibrator_Inst.Instruction = VIBRATOR_ON;
              xStatus = xQueueSendToBack(xVibratorQueue, &Vibrator_Inst, 0);
            }
        }
      vTaskDelay(100 / portTICK_RATE_MS);
    }
}


/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void State_WiredCOMTest(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  COMQueue_Type COMInfo;
  portBASE_TYPE xStatus;

  LCD_Clear();
  LCD_PrintString(0x0, "Wired COM test ");
  LCD_PrintString(0x40,"Send formatted.");
  COM_Wired_PrintString("Wired COM port test mode\r\n");

  while(1)
    {
      xStatus = xQueueReceive(xCOMQueue, &COMInfo, portMAX_DELAY);
      if (xStatus != pdPASS)
        {
          //do nothing
        }
      else
        {
          LCD_Clear();
          LCD_PrintString(0x0, "Received param :");
          LCD_PrintDecimal(0x40,COMInfo.Parameter,10,1);
          COM_Wired_PrintDecimal(COMInfo.Parameter,10,1);
          COM_Wired_PrintString("\r\n");
        }
      vTaskDelay(100 / portTICK_RATE_MS);
    }
}
