/**
  ******************************************************************************
  * @file    freertos_stack_overflow_detection/main.c
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
#include "semphr.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 and FreeRTOS World!\r\n"
  "Expand your creativity and enjoy making.\r\r\n\n";
static xSemaphoreHandle xBinarySemaphore;
xTaskHandle xHandlerTaskHandle;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void TIM4_Configuration(void);
void prvSetupHardware(void);
void vHandlerTask(void *pvParameters);
void vApplicationStackOverflowHook(xTaskHandle *pxTask, signed portCHAR *pcTaskName);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  //Create Binary Semaphore : Have to be called before NVIC is configured
  vSemaphoreCreateBinary(xBinarySemaphore);

  //configure and setting up system and peripherals
  prvSetupHardware();

  //Print welcome message
  cprintf(Welcome_Message);

  if( xBinarySemaphore != NULL )
    {
      // 192: stack overflow will not occur
      xTaskCreate(vHandlerTask, (signed portCHAR *)"HandlerTask", 192, NULL, 3, xHandlerTaskHandle);
      // 128: stack overflow will occur. Sometimes could be detect by kernel and sometimes not.
//      xTaskCreate(vHandlerTask, (signed portCHAR *)"HandlerTask", 128, NULL, 3, xHandlerTaskHandle);
      // 64: stack overflow could be detected by kernel
//      xTaskCreate(vHandlerTask, (signed portCHAR *)"HandlerTask", 64, NULL, 3, xHandlerTaskHandle);

      vTaskStartScheduler();
    }
  else
    {
      cprintf("Failed to create semaphore.\r\n");
    }

  //get here if there was not enough heap space to create the idle task.
  while(1);
}

/**
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable the TIM4 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 12;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configure TIM4
  * @param  None
  * @retval : None
  */
void TIM4_Configuration(void)
{
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM4_RCC , ENABLE);

  /* ---------------------------------------------------------------------------
    TIM4 Configuration: Output Compare Toggle Mode:
    TIM4CLK = 72 MHz, Prescaler = 36000, TIM4 counter clock = 2kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
//  TIM_TimeBaseStructure.TIM_Period = 2000;
  TIM_TimeBaseStructure.TIM_Period = 200;
  TIM_TimeBaseStructure.TIM_Prescaler = 36000;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM4, &TIM_TimeBaseStructure);

  /* TIM IT enable */
  TIM_ITConfig(TIM4, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM4, ENABLE);
}

/**
  * @brief  This function handles TIM4 update interrupt request.
  * @param  None
  * @retval None
  */
void TIM4_IRQHandler(void)
{
  static portBASE_TYPE xHigherPriorityTaskWoken;
  xHigherPriorityTaskWoken = pdFALSE;

  if (TIM_GetITStatus(TIM4, TIM_IT_Update) != RESET)
    {
      // give semaphore to unblock handler task
      xSemaphoreGiveFromISR( xBinarySemaphore, &xHigherPriorityTaskWoken );

      // Clear TIM4 update interrupt pending bit
      TIM_ClearITPendingBit(TIM4, TIM_IT_Update);

      // context switch : Set PendSV bit
      portEND_SWITCHING_ISR( xHigherPriorityTaskWoken );

    }
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
  // Set up NVIC
  NVIC_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();
  // TIM4 configuration for 1ms update interrupt
  TIM4_Configuration();
}

/**
  * @brief  FreeRTOS Task
  * @param  pvParameters : parameter passed from xTaskCreate
  * @retval None
  */
void vHandlerTask(void *pvParameters)
{
  int8_t *pcTaskNumber;
  int32_t RemainingStack;
  pcTaskNumber = (int8_t *)pvParameters;
  while(1)
    {
      // Take semaphore to wait for interrupt
      xSemaphoreTake (xBinarySemaphore, portMAX_DELAY);

      RemainingStack = uxTaskGetStackHighWaterMark(NULL);

      //Entering Critical Section to prevent to be interrupted by ISR
      taskENTER_CRITICAL();
      //Print message through COM
      cprintf("Handler task called.\r\n");
      cprintf("%u stack spaces available.\r\n", RemainingStack);
      //Exit from critical section
      taskEXIT_CRITICAL();
    }
}

/**
  * @brief  FreeRTOS Task
  * @param  pvParameters : parameter passed from xTaskCreate
  * @retval None
  */
void vApplicationStackOverflowHook(xTaskHandle *pxTask, signed portCHAR *pcTaskName)
{
  cprintf("\r\nStack Overflowed.\r\n");

  cprintf("The name of the task make its stack overflowed is :");
  cprintf((int8_t *)pcTaskName);
  cprintf("\r\n");

  if (pxTask == xHandlerTaskHandle)
    {
      cprintf ("The name of the task make its stack overflowed is : HandlerTask\r\n");
    }


}


