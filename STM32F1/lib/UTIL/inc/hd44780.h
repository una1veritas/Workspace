/**
  ******************************************************************************
  * @file    UTIL/inc/toascii.h
  * @author  Yasuo Kawachi
  * @version  V1.0.0
  * @date  04/15/2009
  * @brief  convert integral to string expressed in ASCII
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
#ifndef __HD44780_H
#define __HD44780_H

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include "platform_config.h"

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
#define Display_ON  0b100
#define Display_OFF 0b000
#define Cursor_ON   0b010
#define Cursor_OFF  0b000
#define Brink_ON    0b001
#define Brink_OFF   0b000
/* Exported define -----------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */
void LCD_Print_String(const int8_t String[]);
void LCD_GPIO_Configuration(void);
void GPIO_In_Configuration(void);
void GPIO_Out_Configuration(void);
void LCD_Init(void);
void LCD_Write_Core(uint8_t WriteData);
void LCD_Write_Init_Inst(uint8_t WriteData);
void LCD_Write_Inst(uint8_t WriteData);
void LCD_Write_Data(uint8_t WriteData);
uint8_t LCD_Read_Core(void);
uint8_t LCD_Read_Inst(void);
uint8_t LCD_Read_Data(void);
uint8_t LCD_Put_Char(int8_t Char);
void LCD_Clear(void);
void LCD_Home(void);
void LCD_Display(uint8_t setting);
/* External variables --------------------------------------------------------*/

#endif  /*__HD44780_H*/
