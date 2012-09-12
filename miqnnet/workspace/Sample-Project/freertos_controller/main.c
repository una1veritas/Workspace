/**
 ******************************************************************************
 * @file    freertos_controller/main.c
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
#include "com.h"
#include "joystick.h"
#include "shift.h"
#include "vibrator.h"
#include "statusled.h"
#include "buzzer.h"
#include "lcd.h"
#include "state.h"
#include "xbee.h"
/* Scheduler includes --------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define STACK_SIZE 200
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
void prvSetupHardware(void);
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Main program.
 * @param  None
 * @retval : None
 */
int main(void)
{
  //configure and setting up system and peripherals
  prvSetupHardware();

  //Create queue
  xCOMQueue = xQueueCreate(16, sizeof(COMQueue_Type));
  xXBeeQueue = xQueueCreate(16, sizeof(XBeeQueue_Type));
  xVibratorQueue = xQueueCreate(5, sizeof(VibratorQueue_Type));
  xBuzzerQueue = xQueueCreate(5, sizeof(BuzzerQueue_Type));
  xButtonQueue = xQueueCreate(32, sizeof(ButtonQueue_Type));
  xStatusLEDQueue = xQueueCreate(5, sizeof(StatusLEDQueue_Type));
  xLCDQueue = xQueueCreate(64, sizeof(LCDQueue_Type));
  xEventQueue = xQueueCreate(10, sizeof(EventQueue_Type));

  if (
      xCOMQueue != NULL &&
      xVibratorQueue != NULL &&
      xBuzzerQueue != NULL &&
      xButtonQueue != NULL &&
      xStatusLEDQueue != NULL &&
      xLCDQueue != NULL &&
      xEventQueue != NULL
      )
    {
      //Create tasks.
      xTaskCreate(prvTask_XBee_Control, (signed portCHAR *) "XBee Control", STACK_SIZE, NULL, 1, NULL);
      xTaskCreate(prvTask_Buzzer_Control, (signed portCHAR *) "Buzzer Control", STACK_SIZE, NULL, 1, NULL);
      xTaskCreate(prvTask_Vibrator_Control, (signed portCHAR *) "Vibrator Control", STACK_SIZE, NULL, 1, NULL);
      xTaskCreate(prvTask_ShiftRegister_Control, (signed portCHAR *) "Shift register", STACK_SIZE, NULL, 1, NULL);
      xTaskCreate(prvTask_StatusLED_Control, (signed portCHAR *) "StatusLED Control", STACK_SIZE, NULL, 1, NULL);
      xTaskCreate(prvTask_LCD_Control, (signed portCHAR *) "LCD Control", STACK_SIZE, NULL, 1, NULL);
      xTaskCreate(prvTask_State_Control, (signed portCHAR *) "State Control", STACK_SIZE, NULL, 2, NULL);

      /* Start the scheduler. */
      vTaskStartScheduler();
    }
  else
    {
      COM_Wired_PrintString("Failed to create queue.\r\n");
    }

  //get here if there was not enough heap space to create the idle task.
  while (1){}
}

/**
 * @brief  configure and setting up system and peripherals
 * @param  None
 * @retval : None
 */
void prvSetupHardware(void)
{

  // Configure board specific setting
  BoardInit();

  // Setting up COM port for Print function
  COM_Configuration();

  // Configuration for status LED (TIM1)
  StatusLED_Configuration();
  COM_Wired_PrintString("Status LED configuration is done.\r\n");

  //Configuration for shift register (GPIO)
  ShiftRegister_Configuration();
  COM_Wired_PrintString("Shift registers for buttons configuration is done.\r\n");

  // Configuration for vibrator (TIM4)
  Vibrator_Configuration();
  COM_Wired_PrintString("Vibrator configuration is done.\r\n");

  //Configuration for buzzer (TIM3)
  Buzzer_Configuration();
  COM_Wired_PrintString("Buzzer configuration is done.\r\n");

  // Configuration for Joystick (ADC ch2/3/8/9)
  Joystick_Configuration();
  COM_Wired_PrintString("Joystick configuration is done.\r\n");

  // Configuration for LCD (GPIO)
  LCD_GPIO_Configuration();
  COM_Wired_PrintString("LCD configuration is done.\r\n");

}

