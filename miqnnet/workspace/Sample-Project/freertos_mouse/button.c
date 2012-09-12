/**
  ******************************************************************************
  * @file    freertos_mouse/button.c
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
#include "button.h"
#include "com.h"
#include "state.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define TIMES_TO_CONFIRM  4
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
uint16_t ButtonStatus = 0b1111111111111111;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for shift register
 * @param  None
 * @retval None
 */
void Button_Configuration(void)
{
  Button_GPIO_Configuration();
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_Button_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();
  portBASE_TYPE xStatus;
  uint16_t buttonbits = 0;
  const uint8_t MaskTimesToConfirm = TIMES_TO_CONFIRM - 1;
  static uint8_t ButtonBufferSub = 0;
  uint16_t ButtonBuffer[TIMES_TO_CONFIRM];
  uint16_t BufferTemp;
  uint8_t i;


  while (1)
    {
      buttonbits =
        0 |
        GPIO_ReadInputDataBit(GPIOY_1_PORT, GPIOY_1_PIN) << 1 |
        GPIO_ReadInputDataBit(GPIOY_3_PORT, GPIOY_3_PIN) << 0 ;

      // save value to last array
      ButtonBuffer[ButtonBufferSub] = buttonbits;
      ButtonBufferSub = (ButtonBufferSub + 1) & MaskTimesToConfirm;

      // routine to detect button release (a bit set to 1)

      // detect a bit of which the value set to 1 successively
      BufferTemp = 0b1111111111111111;
      for(i=0;i<TIMES_TO_CONFIRM;i++)
        {
          BufferTemp = BufferTemp & ButtonBuffer[i];
        }

      // detect the bit of ButtonStatus is 0 and the bit of BufferTemp is 1
      BufferTemp = (~ButtonStatus) & BufferTemp;

      // report the difference of button status
      for(i=0;i<16;i++)
        {
          if (BufferTemp & (0b00000001 << i))
            {
              ButtonQueue_Inst.ButtonChange = BUTTON_RELEASED;
              ButtonQueue_Inst.ButtonNumber = i;
              xStatus = xQueueSendToBack(xButtonQueue, &ButtonQueue_Inst, 0);
            }
        }

      // update ButtonStatus
      ButtonStatus = ButtonStatus | BufferTemp;

      // end of : routine to detect button release (a bit set to 1)

      // routine to detect button push (a bit set to 0)

      // detect a bit of which the value set to 0 successively
      BufferTemp = 0;
      for(i=0;i<TIMES_TO_CONFIRM;i++)
        {
          BufferTemp = BufferTemp | ButtonBuffer[i];
        }

      // detect the bit of ButtonStatus is 1 and the bit of BufferTemp is 0
      BufferTemp = ButtonStatus & (~BufferTemp);

      // report the difference of button status
      for(i=0;i<16;i++)
        {
          if (BufferTemp & (0b00000001 << i))
            {
              ButtonQueue_Inst.ButtonChange = BUTTON_PRESSED;
              ButtonQueue_Inst.ButtonNumber = i;
              xStatus = xQueueSendToBack(xButtonQueue, &ButtonQueue_Inst, 0);
            }
        }

      // Generate state change directly
      if (BufferTemp & 0b0000000000000001)
        {
          State_Event(EVENT_STATE_CHANGE_NEXT);
        }

      // update ButtonStatus
      ButtonStatus = ButtonStatus & (~BufferTemp);

      // end of :routine to detect button push (a bit set to 0)

      vTaskDelayUntil(&xLastWakeTime, 5 / portTICK_RATE_MS);

    }
}

/**
 * @brief  Configure GPIO for shift register
 * @param  None
 * @retval : None
 */
void Button_GPIO_Configuration(void)
{
  RCC_APB2PeriphClockCmd(GPIOY_1_RCC | GPIOY_3_RCC ,ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_1:Switch-L */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_3:Switch-R */
  GPIO_InitStructure.GPIO_Pin = GPIOY_3_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
  GPIO_Init(GPIOY_3_PORT, &GPIO_InitStructure);

}
