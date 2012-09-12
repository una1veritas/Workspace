/**
  ******************************************************************************
  * @file    freertos_controller/lcd.h
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   lcd header
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
#ifndef __LCD_H
#define __LCD_H
/* Includes ------------------------------------------------------------------*/
/* Scheduler includes --------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
/* Exported types ------------------------------------------------------------*/
/* Exported variables --------------------------------------------------------*/
extern int8_t DDRAM[1][1];
xQueueHandle xLCDQueue;
typedef struct
{
  uint8_t Instruction;
  uint8_t Address;
  int8_t Char;
} LCDQueue_Type;
typedef enum {
  LCD_OFF = 0,
  LCD_ON = 1,
  LCD_CLEAR = 2,
  LCD_ADDRESS = 3,
  LCD_CHAR = 4,
  LCD_CURSOR_OFF = 5,
  LCD_CURSOR_ON = 6,
  LCD_CURSOR_BLINK = 7
    } LCDStatus_Type;
/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */
void LCD_Configuration(void);
void prvTask_LCD_Control(void *pvParameters);
void LCD_GPIO_Configuration(void);
void LCD_Write_Core(uint8_t WriteData);
void LCD_Write_Inst(uint8_t WriteData);
void LCD_Write_Data(uint8_t WriteData);
void LCD_PrintString(uint8_t address, int8_t* string);
void LCD_PrintDecimal(uint8_t address, int32_t intvalue, uint32_t width, uint8_t plussign);
void LCD_Clear(void);
void LCD_CursorOFF(void);
void LCD_CursorON(void);
void LCD_CursorBlink(void);

#endif /* __LCD_H */
