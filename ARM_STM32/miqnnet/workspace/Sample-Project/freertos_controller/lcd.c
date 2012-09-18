/**
  ******************************************************************************
  * @file    freertos_controller/lcd.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   lcd functions
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
#include "lcd.h"
#include "toascii.h"
#include "com.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define WAIT_3_CLOCK  __asm__("mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t")
#define WAIT_18_CLOCK __asm__("mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t")
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
int8_t DDRAM[1][1];
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for LCD
 * @param  None
 * @retval None
 */
void LCD_Configuration(void)
{
  // Configuration GPIO for LCD
  LCD_GPIO_Configuration();
}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_LCD_Control(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  LCDQueue_Type Received_Inst;
  portBASE_TYPE xStatus;

  //RS=0:Instruction
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  //E=0:Disable
  GPIO_ResetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  // Wait more than 15ms
  vTaskDelayUntil(&xLastWakeTime, 15 / portTICK_RATE_MS);
  //Initial Set : (00),Initial Set(1), 8 bit(1)
  LCD_Write_Inst(0b00110000);
  // Wait more than 4.1ms
  vTaskDelayUntil(&xLastWakeTime, 5 / portTICK_RATE_MS);
  //Initial Set : (00),Initial Set(1), 8 bit(1)
  LCD_Write_Inst(0b00110000);
  // Wait more than 100us
  vTaskDelayUntil(&xLastWakeTime, 1 / portTICK_RATE_MS);
  //Initial Set : (00),Initial Set(1), 8 bit(1)
  LCD_Write_Inst(0b00110000);
  // Wait more than ???us
  vTaskDelayUntil(&xLastWakeTime, 200 / portTICK_RATE_MS);
  //Initial Set : (00),Initial Set(1), 8 bit(1), 2 lines(1), 5x7 dot font(0), (00)
  LCD_Write_Inst(0b00111000);
  vTaskDelayUntil(&xLastWakeTime, 1 / portTICK_RATE_MS);
  //Display OFF : (0000), Display mode (1), Display off(0), Cursor off(0), Brink off(0)
  LCD_Write_Inst(0b00001000);
  vTaskDelayUntil(&xLastWakeTime, 1 / portTICK_RATE_MS);
  //Display Clear : (0000000), Display Clear(1)
  LCD_Write_Inst(0b00000001);
  vTaskDelayUntil(&xLastWakeTime, 2 / portTICK_RATE_MS);
  //Entry mode : (00000), Entry mode(1), Increment(1), Shift off(0)
  LCD_Write_Inst(0b00000110);
  vTaskDelayUntil(&xLastWakeTime, 1 / portTICK_RATE_MS);
  //Display ON : (0000), Display mode (1), Display on(1), Cursor on(1), Brink off(1)
  LCD_Write_Inst(0b00001111);
  vTaskDelayUntil(&xLastWakeTime, 1 / portTICK_RATE_MS);

  while (1)
    {
      //Receive data from queue
      xStatus = xQueueReceive( xLCDQueue, &Received_Inst, portMAX_DELAY);
      if (xStatus != pdPASS)
        {

        }
      else
        {
        }

      switch(Received_Inst.Instruction)
      {
        case LCD_ADDRESS:
          // set lcd address
          LCD_Write_Inst(Received_Inst.Address | 0b10000000);
          vTaskDelay(1 / portTICK_RATE_MS);
          break;
        case LCD_CHAR:
          LCD_Write_Data(Received_Inst.Char);
          vTaskDelay(1 / portTICK_RATE_MS);
          break;
        case LCD_CLEAR:
          // send clear
          LCD_Write_Inst(0b00000001);
          vTaskDelay(2/ portTICK_RATE_MS);
          break;
        case LCD_OFF:
          // send control
          LCD_Write_Inst(0b00001000);
          vTaskDelay(1 / portTICK_RATE_MS);
          break;
        case LCD_CURSOR_OFF:
          // send control
          LCD_Write_Inst(0b00001100);
          vTaskDelay(1 / portTICK_RATE_MS);
          break;
        case LCD_CURSOR_ON:
          // send control
          LCD_Write_Inst(0b00001110);
          vTaskDelay(1 / portTICK_RATE_MS);
          break;
        case LCD_CURSOR_BLINK:
          // send control
          LCD_Write_Inst(0b00001111);
          vTaskDelay(1 / portTICK_RATE_MS);
          break;
      }
    }
}

/**
 * @brief  Configure GPIO
 * @param  None
 * @retval : None
 */
void LCD_GPIO_Configuration(void)
{
  RCC_APB2PeriphClockCmd(GPIOX_RCC |
                         GPIOY_0_RCC |
                         GPIOY_1_RCC |
                         GPIOY_2_RCC |
                         OB_LED_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_1:RS */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_4:E */
  GPIO_InitStructure.GPIO_Pin = GPIOY_4_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_4_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIO_X */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN | GPIOX_1_PIN | GPIOX_2_PIN |
                                GPIOX_3_PIN | GPIOX_4_PIN | GPIOX_5_PIN |
                                GPIOX_6_PIN | GPIOX_7_PIN ;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

}

/**
  * @brief  Writing process common in instruction and data
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Core(uint8_t WriteData)
{
  //Wait: Setup time: tAS(40) : 40ns : 3 clocks
  WAIT_3_CLOCK;
  //Output Data
  GPIO_Write(GPIOX_PORT, (GPIO_ReadOutputData(GPIOX_PORT) & 0xFF00) | (uint16_t)WriteData);
  //E=1: Enable
  GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  //Wait: Pulse time and holdtime: PWEH(220) + tH(10) : 230ns : 18 clocks
  WAIT_18_CLOCK;
  //E=0: Disable
  GPIO_ResetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  //Wait: tC(500) - PWEH(220) - tH(10) - tAS(40) : 230ns : 18 clocks
  WAIT_18_CLOCK;
}

/**
  * @brief  Writing instruction with busy flag check
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Inst(uint8_t WriteData)
{
  //RS=0: Instruction
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);

  LCD_Write_Core(WriteData);
}

/**
  * @brief  Writing data
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Data(uint8_t WriteData)
{
  //RS=1: Data
  GPIO_SetBits(GPIOY_1_PORT, GPIOY_1_PIN);

  LCD_Write_Core(WriteData);
}

/**
  * @brief  print string on lcd
  * @param  Address : address to be written
  * @param  String : string to be printed
  * @retval : None
  */
void LCD_PrintString(uint8_t address, int8_t* string)
{
  LCDQueue_Type LCD_Inst;
  LCD_Inst.Instruction = LCD_ADDRESS;
  LCD_Inst.Address = address;
  xQueueSendToBack(xLCDQueue, &LCD_Inst, 0);
  while(*string != '\0')
    {
      LCD_Inst.Instruction = LCD_CHAR;
      LCD_Inst.Address = address;
      LCD_Inst.Char = *string;
      xQueueSendToBack(xLCDQueue, &LCD_Inst, 0);
      string++;
    }
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void LCD_PrintDecimal(uint8_t address, int32_t intvalue, uint32_t width, uint8_t plussign)
{
  int8_t buffer[12];
  if (width == 0 && intvalue > 0 && plussign == 0)
    {
      Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');
    }
  else if (width == 0 && intvalue > 0 && plussign == 1)
    {
      buffer[0] = '+';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else if (width == 0 && intvalue < 0)
    {
      buffer[0] = '-';
      Uint32_tToDecimal(-intvalue, &buffer[1], width, ' ');
    }
  else if (width == 0 && intvalue == 0)
    {
      Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');
    }
  else if (plussign != 0 && intvalue > 0)
    {
      buffer[0] = '+';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else if ((plussign == 0 && intvalue > 0) || intvalue == 0)
    {
      buffer[0] = ' ';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else
    {
      buffer[0] = '-';
      Uint32_tToDecimal(-intvalue, &buffer[1], width, ' ');
    }
  LCD_PrintString(address, buffer);
}


/**
  * @brief  Clear LCD
  * @param   : none
  * @retval : None
  */
void LCD_Clear(void)
{
  LCDQueue_Type LCD_Inst;
  LCD_Inst.Instruction = LCD_CLEAR;
  xQueueSendToBack(xLCDQueue, &LCD_Inst, 0);
}

/**
  * @brief  cursor off
  * @param   : none
  * @retval : None
  */
void LCD_CursorOFF(void)
{
  LCDQueue_Type LCD_Inst;
  LCD_Inst.Instruction = LCD_CURSOR_OFF;
  xQueueSendToBack(xLCDQueue, &LCD_Inst, 0);
}

/**
  * @brief  cursor on
  * @param   : none
  * @retval : None
  */
void LCD_CursorON(void)
{
  LCDQueue_Type LCD_Inst;
  LCD_Inst.Instruction = LCD_CURSOR_ON;
  xQueueSendToBack(xLCDQueue, &LCD_Inst, 0);
}

/**
  * @brief  cursor on
  * @param   : none
  * @retval : None
  */
void LCD_CursorBlink(void)
{
  LCDQueue_Type LCD_Inst;
  LCD_Inst.Instruction = LCD_CURSOR_BLINK;
  xQueueSendToBack(xLCDQueue, &LCD_Inst, 0);
}
