/**
 ******************************************************************************
 * @file    freertos_mouse/main.c
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
#include "motor.h"
#include "led.h"
#include "music.h"
#include "com.h"
#include "sonic.h"
#include "psd.h"
#include "state.h"
#include "button.h"
#include "xbee.h"
/* Scheduler includes --------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
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
  xXBeeQueue = xQueueCreate(20, sizeof(XBeeQueue_Type));
  xMotorQueue = xQueueCreate(10, sizeof(MotorQueue_Type));
  xLEDQueue = xQueueCreate(10, sizeof(LEDQueue_Type));
  xMusicQueue = xQueueCreate(10, sizeof(MusicQueue_Type));
  xButtonQueue = xQueueCreate(10, sizeof(ButtonQueue_Type));
  xCOMQueue = xQueueCreate(10, sizeof(COMQueue_Type));
  xEventQueue = xQueueCreate(10, sizeof(EventQueue_Type));

  if (xMotorQueue != NULL && xLEDQueue != NULL && xMusicQueue != NULL)
    {
      //Create tasks.
      xTaskCreate(prvTask_XBee_Control, (signed portCHAR *) "XBee Control", 200, NULL, 1, NULL);
      xTaskCreate(prvTask_Motor_Control, (signed portCHAR *) "Motor Control", 200, NULL, 1, NULL);
      xTaskCreate(prvTask_LED_Control, (signed portCHAR *) "LED Control", 200, NULL, 1, NULL);
      xTaskCreate(prvTask_Music_Control, (signed portCHAR *) "Music Control", 200, NULL, 1, NULL);
      xTaskCreate(prvTask_Button_Control, (signed portCHAR *) "Button Control", 200, NULL, 1, NULL);
      xTaskCreate(prvTask_State_Control, (signed portCHAR *) "Controller", 200, NULL, 2, NULL);

      /* Start the scheduler. */
      vTaskStartScheduler();
    }
  else
    {
      COM_Wired_PrintFormatted("Failed to create queue.\r\n");
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
  COM_Wired_PrintFormatted("COM port setting is done.\r\n");

  // Configuration for motor (TIM3 and GPIOs)
  Motor_Configuration();
  COM_Wired_PrintFormatted("Motors configuration is done.\r\n");

  //Configuration for buttons (GPIOs)
  Button_Configuration();
  COM_Wired_PrintFormatted("Buttons configuration is done.\r\n");

  // Configuration for LED (TIM1)
  LED_Configuration();
  COM_Wired_PrintFormatted("LED configuration is done.\r\n");

  // Configuration for Music (TIM4)
  Music_Configuration();
  COM_Wired_PrintFormatted("Music configuration is done.\r\n");

  // Configuration for supersonic sensor (USART1)
  Sonic_Configuration();
  COM_Wired_PrintFormatted("Supersonic sensor configuration is done.\r\n");

  // Configuration for PSD sensor (ADC12)
  PSD_Configuration();
  COM_Wired_PrintFormatted("PSD sensor configuration is done.\r\n");

}
