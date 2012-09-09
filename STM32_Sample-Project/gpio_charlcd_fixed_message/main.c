/**
  ******************************************************************************
  * @file    gpio_charlcd_fixed_message/main.c
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
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
#define WAIT_3_CLOCK  __asm__("mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t")
#define WAIT_18_CLOCK __asm__("mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t"\
                              "mov r8,r8\n\tmov r8,r8\n\tmov r8,r8\n\t")

/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHello Cortex-M3/STM32 World!\r\n"
                            "Expand your creativity and enjoy making.\r\n\r\n"
                            "You can find the message on LCD.\r\n";
int8_t LCD_Message1[] = "Hello Cortex-M3";
int8_t LCD_Message2[] = "World on LCD!!!";
/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void GPIO_In_Configuration(void);
void GPIO_Out_Configuration(void);
void Init_LCD(void);
void LCD_Write_Core(uint8_t WriteData);
void LCD_Write_Init_Inst(uint8_t WriteData);
void LCD_Write_Inst(uint8_t WriteData);
void LCD_Write_Data(uint8_t WriteData);
uint8_t LCD_Read_Core(void);
uint8_t LCD_Read_Inst(void);
uint8_t LCD_Read_Data(void);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  // Configure board specific setting
  BoardInit();
  /* Configure the GPIO ports */
  GPIO_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();

  GPIO_In_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  //Initialize LCD
  Init_LCD();

  //Show message on first line
  uint8_t i = 0;
  while(LCD_Message1[i]!='\0')
    {
      LCD_Write_Data(LCD_Message1[i]);
      i++;
    }

  //Show message on second line
  LCD_Write_Inst(0x40 | 0b10000000);
  i = 0;
  while(LCD_Message2[i]!='\0')
    {
      LCD_Write_Data(LCD_Message2[i]);
      i++;
    }

  while(1);
}

/**
  * @brief  Initialize HD44780 compatible LCD
  * @param  None
  * @retval : None
  */
void Init_LCD(void)
{
  GPIO_In_Configuration();
  //RS=0:Instruction
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
  //RW=0:Write (Output to LCD)
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  //E=0:Disable
  GPIO_ResetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  // Wait more than 15ms
  delay_ms(15);
  //Initial Set : (00),Initial Set(1), 8 bit(1)
  LCD_Write_Init_Inst(0b00110000);
  // Wait more than 4.1ms
  delay_ms(5);
  //Initial Set : (00),Initial Set(1), 8 bit(1)
  LCD_Write_Init_Inst(0b00110000);
  // Wait more than 100us
  delay_us(100);
  //Initial Set : (00),Initial Set(1), 8 bit(1)
  LCD_Write_Init_Inst(0b00110000);
  delay_us(100);
  // BF can be checked from this point
  //Initial Set : (00),Initial Set(1), 8 bit(1), 2 lines(1), 5x7 dot font(0), (00)
  LCD_Write_Inst(0b00111000);
  //Display OFF : (0000), Display mode (1), Display off(0), Cursor off(0), Blink off(0)
  LCD_Write_Inst(0b00001000);
  //Display Clear : (0000000), Display Clear(1)
  LCD_Write_Inst(0b00000001);
  //Entry mode : (00000), Entry mode(1), Increment(1), Shift off(0)
  LCD_Write_Inst(0b00000110);
  //Display ON : (0000), Display mode (1), Display on(1), Cursor on(1), Blink off(1)
  LCD_Write_Inst(0b00001111);
}

/**
  * @brief  Writing process common in instruction and data
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Core(uint8_t WriteData)
{
  //RW=0: Write (Output to LCD)
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  //Wait: Setup time: tAS(40) : 40ns : 3 clocks
  WAIT_3_CLOCK;
  //Set GPIO as output
  GPIO_Out_Configuration();
  //Output Data
  GPIO_Write(GPIOX_PORT, (GPIO_ReadOutputData(GPIOX_PORT) & 0xFF00) | (uint16_t)WriteData);
  //E=1: Enable
  GPIO_SetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  //Wait: Pulse time and holdtime: PWEH(220) + tH(10) : 230ns : 18 clocks
  WAIT_18_CLOCK;
  //E=0: Disable
  GPIO_ResetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  //Wait: tC(500) - PWEH(220) - tH(10) - tAS(40) : 230ns : 18 clocks
  WAIT_18_CLOCK;
}

/**
  * @brief  Writing instruction without busy flag check
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Init_Inst(uint8_t WriteData)
{
  //RS=0: Instruction
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);

  LCD_Write_Core(WriteData);
}


/**
  * @brief  Writing instruction with busy flag check
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Inst(uint8_t WriteData)
{

  while( LCD_Read_Inst() & 0b10000000);
  //RS=0: Instruction
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);

  LCD_Write_Core(WriteData);
}

/**
  * @brief  Writing data
  * @param  WriteData: Data to be written
  * @retval : None
  */
void LCD_Write_Data(uint8_t WriteData)
{

  while(LCD_Read_Inst() & 0b10000000);
  //RS=1: Data
  GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);

  LCD_Write_Core(WriteData);
}

/**
  * @brief  Read process common in instruction and data
  * @param  : None
  * @retval Data read from LCD
  */
uint8_t LCD_Read_Core(void)
{
  uint8_t ReadData;

  //RW=1: Read (Output from LCD)
  GPIO_SetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  //Wait: Setup time: tAS(40) : 40ns : 3 clocks
  WAIT_3_CLOCK;
  //Set GPIO as input
  GPIO_In_Configuration();
  //E=1:Enable
  GPIO_SetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  //Wait: Pulse time: PWEH(220) : 220ns : 18 clocks
  WAIT_18_CLOCK;
  //Read data from GPIO
  ReadData =(uint8_t)GPIO_ReadInputData(GPIOX_PORT);
  //E=0:Disable
  GPIO_ResetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  //Wait: Pulse holdtime: tH(10) : 10ns
  //should be passed during commands
  //RW=0: Write (Output to LCD)
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  //Wait: tC(500) - PWEH(220) - tH(10) - tAS(40) : 230ns : 18 clocks
  WAIT_18_CLOCK;
  return(ReadData);
}

/**
  * @brief  Read instruction from LCD
  * @param  : None
  * @retval Data read from LCD
  */
uint8_t LCD_Read_Inst(void)
{
  //RS=0: Instruction
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);

  return(LCD_Read_Core());
}

/**
  * @brief  Read data from LCD
  * @param  : None
  * @retval Data read from LCD
  */
uint8_t LCD_Read_Data(void)
{
  //RS=1: Data
  GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);

  return(LCD_Read_Core());
}

/**
  * @brief  Configure the GPIO Pins for instruction.
  * @param  None
  * @retval : None
  */
void GPIO_Configuration(void)
{
  RCC_APB2PeriphClockCmd(GPIOX_RCC |
                         GPIOY_0_RCC |
                         GPIOY_1_RCC |
                         GPIOY_2_RCC |
                         OB_LED_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIOY_0:RS */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_1:RW */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIO for GPIOY_2:E */
  GPIO_InitStructure.GPIO_Pin = GPIOY_2_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOY_2_PORT, &GPIO_InitStructure);

  /* Configure GPIO for OB_LED */
  GPIO_InitStructure.GPIO_Pin = OB_LED_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(OB_LED_PORT, &GPIO_InitStructure);

}

/**
  * @brief  Configure the GPIO Pins for data as output
  * @param  None
  * @retval : None
  */
void GPIO_Out_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIO_X */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN | GPIOX_1_PIN | GPIOX_2_PIN |
                                GPIOX_3_PIN | GPIOX_4_PIN | GPIOX_5_PIN |
                                GPIOX_6_PIN | GPIOX_7_PIN ;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the GPIO Pins for data as input
  * @param  None
  * @retval : None
  */
void GPIO_In_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIO for GPIO_X */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN | GPIOX_1_PIN | GPIOX_2_PIN |
                                GPIOX_3_PIN | GPIOX_4_PIN | GPIOX_5_PIN |
                                GPIOX_6_PIN | GPIOX_7_PIN ;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);
}
