/**
  ******************************************************************************
  * @file    freertos_queue_send_task_number/main.c
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
int8_t Task1_number = 1;
int8_t Task2_number = 2;
int8_t Task3_number = 3;
int8_t Task4_number = 4;
int8_t Task5_number = 5;
xQueueHandle xQueue;
/* Private function prototypes -----------------------------------------------*/
void prvSetupHardware(void);
void prvTask_Queue_Sender(void *pvParameters);
void prvTask_Queue_Receiver(void *pvParameters);
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
  xQueue = xQueueCreate(6, sizeof (int8_t));

  if (xQueue != NULL)
    {
      //Create tasks.
      xTaskCreate(prvTask_Queue_Sender,    (signed portCHAR *)"Sender1",  192, &Task1_number, 1, NULL);
      xTaskCreate(prvTask_Queue_Sender,    (signed portCHAR *)"Sender2",  192, &Task2_number, 1, NULL);
      xTaskCreate(prvTask_Queue_Sender,    (signed portCHAR *)"Sender3",  192, &Task3_number, 1, NULL);
      xTaskCreate(prvTask_Queue_Sender,    (signed portCHAR *)"Sender4",  192, &Task4_number, 1, NULL);
      xTaskCreate(prvTask_Queue_Sender,    (signed portCHAR *)"Sender5",  192, &Task5_number, 1, NULL);
      xTaskCreate(prvTask_Queue_Receiver,  (signed portCHAR *)"Receiver", 192, NULL, 1, NULL);

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
void prvTask_Queue_Sender(void *pvParameters)
{
  int8_t pcTaskNumber;
  pcTaskNumber = *(int8_t *)pvParameters;
  portBASE_TYPE xStatus;
  while(1)
    {
      //Send task number to queue
      xStatus = xQueueSendToBack(xQueue, &pcTaskNumber, 0);

      //Failure Check
      if ( xStatus != pdPASS)
        {
          cprintf("Failed to send to queue.");
        }

      //Wait around 8000 - 2000 ms
      vTaskDelay((500 / portTICK_RATE_MS) + (pcTaskNumber * 300 / portTICK_RATE_MS));
    }
}

/**
  * @brief  FreeRTOS Task
  * @param  pvParameters : parameter passed from xTaskCreate
  * @retval None
  */
void prvTask_Queue_Receiver(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *)pvParameters;
  portBASE_TYPE xStatus;
  int8_t Acquired_Data;
  while(1)
    {
      //Checks a data is in queue
      while(uxQueueMessagesWaiting(xQueue))
        {
          //Receive data from queue
          xStatus = xQueueReceive( xQueue, &Acquired_Data, 0);
          //Send data through COM port
          cprintf("Received a queue data from Task\r\n");
          cprintf("%u\r\n",Acquired_Data);
        }
    }
}


