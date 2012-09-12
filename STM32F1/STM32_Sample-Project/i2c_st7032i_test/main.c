/**
  ******************************************************************************
  * @file    i2c_st7032i_test/main.c
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
#define I2C_CLOCK                  100000
#define ST7032I_ADDR               0b0111110
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Initialize and put character on ST7032i LCD.\r\n\r\n";

/* Private function prototypes -----------------------------------------------*/
void I2C_Configuration(void);
void ST7032i_Command_Write(uint8_t Data);
void ST7032i_Data_Write(uint8_t Data);

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
  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  delay_ms(40);

  I2C_Configuration();

  //Function Set
  ST7032i_Command_Write(0b00111000);

  delay_us(27);

  //Function Set
  ST7032i_Command_Write(0b00111001);

  delay_us(27);

  //Bias and OSC frequency
  ST7032i_Command_Write(0b00010100);

  delay_us(27);

  //Contrast set
  ST7032i_Command_Write(0b01110000);

  delay_us(27);

  //Power/Icon/Contrast control
  ST7032i_Command_Write(0b01010110);

  delay_us(27);

  //Follower control
  ST7032i_Command_Write(0b01101100);

  delay_ms(200);

  //Function Set
  ST7032i_Command_Write(0b00111000);

  //Display control : on
  ST7032i_Command_Write(0b00001100);

  delay_us(27);

  //Clear
  ST7032i_Command_Write(0b00000001);

  delay_ms(2);

  ST7032i_Data_Write('H');
  ST7032i_Data_Write('e');
  ST7032i_Data_Write('l');
  ST7032i_Data_Write('l');
  ST7032i_Data_Write('o');

  ST7032i_Command_Write(0b10000000 | 0x40);

  ST7032i_Data_Write('W');
  ST7032i_Data_Write('o');
  ST7032i_Data_Write('r');
  ST7032i_Data_Write('l');
  ST7032i_Data_Write('d');
  ST7032i_Data_Write('!');

  cprintf("Done! Confirm a message is on LCD.");

  while(1){}
}

/**
  * @brief  I2C Configuration
  * @param  None
  * @retval None
  */
void I2C_Configuration(void)
{
  GPIO_InitTypeDef  GPIO_InitStructure;
  I2C_InitTypeDef  I2C_InitStructure;

  /* I2C Periph clock enable */
  RCC_APB1PeriphClockCmd(I2C1_RCC, ENABLE);
  /* GPIO Periph clock enable */
  RCC_APB2PeriphClockCmd(I2C1_GPIO_RCC, ENABLE);

  /* Configure I2C pins: SCL and SDA */
  GPIO_InitStructure.GPIO_Pin =  I2C1_SCL_PIN | I2C1_SDA_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_OD;
  GPIO_Init(I2C1_PORT, &GPIO_InitStructure);

#if defined (REMAP_I2C1)
Remap_I2C1_Configuration();
#endif

  /* I2C configuration */
  I2C_InitStructure.I2C_Mode = I2C_Mode_I2C;
  I2C_InitStructure.I2C_DutyCycle = I2C_DutyCycle_2;
//  I2C_InitStructure.I2C_OwnAddress1 = I2C_SLAVE_ADDRESS7;
  I2C_InitStructure.I2C_Ack = I2C_Ack_Enable;
  I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit;
  I2C_InitStructure.I2C_ClockSpeed = I2C_CLOCK;

  /* I2C Peripheral Enable */
  I2C_Cmd(I2C1, ENABLE);
  /* Apply I2C configuration after enabling it */
  I2C_Init(I2C1, &I2C_InitStructure);
}

/**
  * @brief  Write Command to ST7032i
  * @param  Data : Command Data
  * @retval None
  */
void ST7032i_Command_Write(uint8_t Data)
{

  /* Send STRAT condition */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, ST7032I_ADDR << 1, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C1, 0b00000000);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C1, Data);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);
}

/**
  * @brief  Write Data to ST7032i
  * @param  Data : "Data" Data
  * @retval None
  */
void ST7032i_Data_Write(uint8_t Data)
{

  /* Send STRAT condition */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, ST7032I_ADDR << 1, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C1, 0b01000000);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C1, Data);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);
}
