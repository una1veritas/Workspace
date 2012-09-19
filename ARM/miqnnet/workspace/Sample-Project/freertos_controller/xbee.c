/**
  ******************************************************************************
  * @file    freertos_controller/xbee.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   xbee functions
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
#include "xbee.h"
#include "statusled.h"
#include "com.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define TIMES_TO_TRY_KEYWORD_MATCH 23
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint8_t XBeeStatus = XBEE_DISCONNECTED;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_XBee_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  StatusLEDQueue_Type StatusLED_Inst;

  XBeeQueue_Type XBeeInfo;
  portBASE_TYPE xStatus;

  uint32_t keynumber = 0x66666663; // $12340004
  static uint8_t timesfailed = 0;

  while (1)
    {
      // proceed key number to generate original key
      keynumber++;

      switch(XBeeStatus)
      {
        case XBEE_DISCONNECTED:

          StatusLED_Inst.LEDNumber = 0;
          StatusLED_Inst.LEDStatus = STATUSLED_FAST;
          xQueueSendToBack(xStatusLEDQueue, &StatusLED_Inst, 0);

          COM_Wired_PrintString("XBee configuring...\r\n");

          XBeeStatus = XBEE_INITIATING;

          GPIO_ResetBits(GPIOY_0_PORT,GPIOY_0_PIN);
          vTaskDelay(100 / portTICK_RATE_MS);
          GPIO_SetBits(GPIOY_0_PORT,GPIOY_0_PIN);

          // Wait XBee module to wake up
          vTaskDelay(1500 / portTICK_RATE_MS);

          // set XBee-USART speed to XBee default baudrate(9600bps)
          COM_XBee_Configuration(9600);
          // Enter into command mode
          COM_XBee_PrintString("+++");

          // 1000 ms is required for guard time and additional 500 ms is for getting response
          vTaskDelay(1500 / portTICK_RATE_MS);
          COM_XBee_PrintString("ATCH15,ID7777,DH0,DL1,MY2,BD7,CN\r");
          // 500 ms is enough for single command but 1500 ms is required baudrate change
          vTaskDelay(1500 / portTICK_RATE_MS);
          COM_XBee_Configuration(115200);
          vTaskDelay(1500 / portTICK_RATE_MS);

          XBeeStatus = XBEE_INITIATED;

          COM_Wired_PrintString("XBee configured.\r\n");

          break;
        case XBEE_INITIATED:
        case XBEE_CONNECTED:
          // clear queue
          while (xQueueReceive(xXBeeQueue, &XBeeInfo, 0) == pdPASS)
            {
              // do nothing
            }
          // send key number
          COM_XBee_PrintFormatted("$%8X\r",keynumber);
          vTaskDelay(100 / portTICK_RATE_MS);
          xStatus = xQueueReceive(xXBeeQueue, &XBeeInfo, 0);
          // received no data within deadline
          if (xStatus != pdPASS)
            {
              timesfailed++;
            }
          // received data within deadline
          else
            {
              if (XBeeInfo.Parameter == keynumber)
                {
                  if(XBeeStatus == XBEE_INITIATED)
                    {
                      StatusLED_Inst.LEDNumber = 0;
                      StatusLED_Inst.LEDStatus = STATUSLED_ON;
                      xQueueSendToBack(xStatusLEDQueue, &StatusLED_Inst, 0);

                      COM_Wired_PrintString("XBee connected.\r\n");
                    }

                  XBeeStatus = XBEE_CONNECTED;
                  timesfailed = 0;
                }
              else
                {
                  timesfailed++;
                }
            }
          if (timesfailed > TIMES_TO_TRY_KEYWORD_MATCH)
            {
              COM_Wired_PrintString("XBee disconnected.\r\n");
              XBeeStatus = XBEE_DISCONNECTED;
              timesfailed = 0;
            }
          break;

      }
      vTaskDelay(1000 / portTICK_RATE_MS);
    }
}

