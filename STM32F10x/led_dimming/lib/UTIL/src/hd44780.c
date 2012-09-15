/**
  ******************************************************************************
  * @file    UTIL/src/hd44780.c
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

/* Includes ------------------------------------------------------------------*/
#include "hd44780.h"

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
  __IO uint16_t delay;
/* Extern variables ----------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Clear LCD
  * @param  : None
  * @retval : None
  */
void LCD_Clear(void)
{
  //Wait until busy flag is reset
  while( LCD_Read_Inst() & 0b10000000);
  LCD_Write_Inst(0b00000001);
}

/**
  * @brief  Go home on LCD
  * @param  : None
  * @retval : None
  */
void LCD_Home(void)
{
  //Wait until busy flag is reset
  while( LCD_Read_Inst() & 0b10000000);
  LCD_Write_Inst(0b00000010);
}

/**
  * @brief  Configure LCD display settings
  * @param  : None
  * @retval : None
  */
void LCD_Display(uint8_t setting)
{
  //Wait until busy flag is reset
  while( LCD_Read_Inst() & 0b10000000);
  LCD_Write_Inst(0b00001000 | setting);
}

/**
  * @brief  print sting on LCD
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void LCD_Print_String(const int8_t String[])
{
  uint8_t i = 0;
  while (String[i] != '\0')
    {
      LCD_Put_Char(String[i]);
      i++;
    }
}

/**
  * @brief  print single character on LCD
  * @param  Char: byte to be sent
  * @retval : None
  */
uint8_t LCD_Put_Char(int8_t Char)
{
  uint8_t Address, i;
  int8_t Buffer;  //!< Temporally  buffer for scrolling
  LCD_Write_Data(Char);

  //Wait until busy flag is reset
  while( LCD_Read_Inst() & 0b10000000);

  //Read address from LCD. Need bit7 to be masked cause it is busy flag
  Address = LCD_Read_Inst() & 0b01111111;

  if (Address == 0x10 || ( Char == '\n' && Address < 0x10) )
  {
    LCD_Write_Inst(0x40 | 0b10000000);
    WAIT_18_CLOCK;
    WAIT_18_CLOCK;
  }
  else if (Address == 0x50 || ( Char == '\n' && Address > 0x10))
    {
      LCD_Write_Inst(0x40 | 0b10000000) ;
      for (i =0 ;i<16;i++)
      {
        //Write address of lower line
        LCD_Write_Inst(0b10000000 | (0x40+i));
        //Wait until busy flag is reset
        while( LCD_Read_Inst() & 0b10000000);
        //Read data from lower line
        Buffer = LCD_Read_Data() ;
        //Write address of upper line
        LCD_Write_Inst(0b10000000 | (0x00+i));
        LCD_Write_Data(Buffer);
      }
      //Clear lower line
      LCD_Write_Inst(0b10000000 | 0x40 );
      for (i=0;i<16;i++)
      {
              LCD_Write_Data(0x20);
      }
      //Back home to lower line
      LCD_Write_Inst(0x40 | 0b10000000);

    }

  return(Address);
}

/**
  * @brief  Initialize HD44780 compatible LCD
  * @param  None
  * @retval : None
  */
void LCD_Init(void)
{
  for (delay=0;delay<150;delay++){WAIT_18_CLOCK;}

  LCD_GPIO_Configuration();
  GPIO_In_Configuration();
  //RS=0:Instruction
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
  //RW=0:Write (Output to LCD)
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  //E=0:Disable
  GPIO_ResetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  WAIT_18_CLOCK;
  WAIT_18_CLOCK;
  LCD_Write_Init_Inst(0b00110000);
  WAIT_18_CLOCK;
  WAIT_18_CLOCK;
  LCD_Write_Init_Inst(0b00110000);
  WAIT_18_CLOCK;
  WAIT_18_CLOCK;
  LCD_Write_Init_Inst(0b00110000);
  for (delay=0;delay<150;delay++){WAIT_18_CLOCK;}
  LCD_Write_Init_Inst(0b00111000);
  LCD_Write_Inst(0b00111000);
  //Display OFF
  LCD_Write_Inst(0b00001000);
  //Display Clear
  LCD_Write_Inst(0b00000001);
  //Display Entry mode
  LCD_Write_Inst(0b00000110);
  //Display ON
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
  //Wait until busy flag is reset
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
void LCD_GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOX_RCC |
                         GPIOY_0_RCC |
                         GPIOY_1_RCC |
                         GPIOY_2_RCC , ENABLE);

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
