/**
  ******************************************************************************
  * @file    freertos_mouse/com.h
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __COM_H
#define __COM_H
/* Includes ------------------------------------------------------------------*/
/* Scheduler includes --------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
/* Exported types ------------------------------------------------------------*/
/* Exported variables --------------------------------------------------------*/
xQueueHandle xCOMQueue;
typedef struct
{
  uint32_t Parameter;
} COMQueue_Type;
COMQueue_Type COMQueue_Inst;
/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */
void COM_Configuration(void);
void COM_NVIC_Configuration(void);
void prvTask_COM_RX(void *pvParameters);
void COM_Wired_Configuration(uint32_t baudrate);
void Remap_USART1_Configuration(void);
void Remap_USART2_Configuration(void);
void FullRemap_USART3_Configuration(void);
void COM_Wired_PrintChar(int8_t charvalue);
void COM_Wired_PrintString(const int8_t string[]);
void COM_Wired_PrintDecimal(int32_t intvalue, uint32_t width, uint8_t plussign);
void COM_Wired_PrintHexaecimal(int32_t intvalue, uint32_t width, uint8_t smallcase);
void COM_Wired_PrintBinary(int32_t intvalue, uint32_t width);
void COM_Wired_PrintFormatted(int8_t* string, ...);
void XBee_GPIO_Configuration(void);
void COM_XBee_Configuration(uint32_t baudrate);
void COM_XBee_PrintChar(int8_t charvalue);
void COM_XBee_PrintString(const int8_t string[]);
void COM_XBee_PrintDecimal(int32_t intvalue, uint32_t width, uint8_t plussign);
void COM_XBee_PrintHexaecimal(int32_t intvalue, uint32_t width, uint8_t smallcase);
void COM_XBee_PrintBinary(int32_t intvalue, uint32_t width);
void COM_XBee_PrintFormatted(int8_t* string, ...);
void COM_PrintString(const int8_t string[]);
void COM_PrintDecimal(uint32_t intdata);

#endif /* __COM_H */
