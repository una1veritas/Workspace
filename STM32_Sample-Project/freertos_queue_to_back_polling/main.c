/**
  ******************************************************************************
  * @file    freertos_queue_to back_polling/main.c
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
/* Scheduler includes --------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 and FreeRTOS World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";
xQueueHandle xQueue;
/* Private function prototypes -----------------------------------------------*/
void prvSetupHardware(void);
void prvTask_USARTRX_Queue_Send(void *pvParameters);
void prvTask_USARTTX_Queue_Receive(void *pvParameters);
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

  //Print welcome message
  cprintf(Welcome_Message);

  //Create queue
  xQueue = xQueueCreate(64, sizeof (int8_t));

  if (xQueue != NULL)
    {
      //Create tasks.
      xTaskCreate(prvTask_USARTRX_Queue_Send,    (signed portCHAR *)"Sender",   200, NULL, 1, NULL);
      xTaskCreate(prvTask_USARTTX_Queue_Receive, (signed portCHAR *)"Receiver", 200, NULL, 1, NULL);

      /* Start the scheduler. */
      vTaskStartScheduler();
    }
  else
    {
      cprintf("Failed to create queue.\r\n");
    }

  //get here if there was not enough heap space to create the idle task.
  while(1);
}

/**
  * @brief  configure and setting up system and peripherals
  * @param  None
  * @retval : None
  */
void prvSetupHardware( void )
{

  // Configure board specific setting
  BoardInit();
  // Setting up COM port for Print function
  COM_Configuration();
}

/**
  * @brief  FreeRTOS Task
  * @param  pvParameters : parameter passed from xTaskCreate
  * @retval None
  */
void prvTask_USARTRX_Queue_Send(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *)pvParameters;
  int8_t RxData;
  portBASE_TYPE xStatus;
  while(1)
    {
      //Receive data from COM port
      while(RX_BUFFER_IS_EMPTY);
      RxData = RECEIVE_DATA;

      //Send COM data to queue
      xStatus = xQueueSendToBack(xQueue, &RxData, 0);

      //Failure Check
      if ( xStatus != pdPASS)
        {
          cprintf("Failed to send to queue.");
        }

      //Terminate before the end of time slice
      taskYIELD();
    }
}

/**
  * @brief  FreeRTOS Task
  * @param  pvParameters : parameter passed from xTaskCreate
  * @retval None
  */
void prvTask_USARTTX_Queue_Receive(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *)pvParameters;
  portBASE_TYPE xStatus;
  int8_t Acquired_Data;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();
  while(1)
    {
      //Checks a data is in queue
      while(uxQueueMessagesWaiting(xQueue))
        {
          //Receive data from queue
          xStatus = xQueueReceive( xQueue, &Acquired_Data, 0);
          //Send data through COM port
          cputchar(Acquired_Data);
        }
      vTaskDelayUntil(&xLastWakeTime, 1000 / portTICK_RATE_MS );
    }
}

