/**
  ******************************************************************************
  * @file    i2c_eeprom_memorize_message/main.c
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
#include "scanf.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define EEPROM_SLAVE_ADDRESS       0xA0
#define ROM_Address                0x0000
#define LAST_Address               101
#define I2C_CLOCK                  50000
#define WRITE_DATA                 0b01000001
#define MAXCOUNT 100
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Here I show you EEPROM can memorize message over power off period \r\n\r\n";
/* Private function prototypes -----------------------------------------------*/
void I2C_Configuration(void);
void I2C_EEPROM_Write(uint8_t Data, uint16_t Address);
void I2C_EEPROM_Poll(void);
uint8_t I2C_EEPROM_Read(uint16_t Address);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  int8_t RxBuffer[MAXCOUNT];
  uint8_t RxData;
  uint8_t Counter;


  // Configure board specific setting
  BoardInit();
  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  I2C_Configuration();

  // white NULL for unerased EEPROM
  I2C_EEPROM_Write('\0', LAST_Address);
  I2C_EEPROM_Poll();

  //Read Message from EEPROM
  cprintf("Message written in EEPROM is : \r\n");
  Counter = 0;
  while(1)
    {
      RxData = I2C_EEPROM_Read(ROM_Address + Counter);
      cputchar(RxData);
      if(RxData == '\0')
      {
        break;
      }
      Counter++;
    }

  //Show Welcome message
  cprintf(
      "\r\nType a message to memorize in EEPROM. \r\n"
      "Type Enter when complete\r\n"
      "Message have to be less than 100 characters.\r\n");

  //Get Message to write in EEPROM
  COM_Char_Scanf(RxBuffer, MAXCOUNT);
  cprintf("\r\nWhat you type is \r\n");
  cprintf(RxBuffer);
  cprintf("\r\n");

  //Write Message in EEPROM
  cprintf("Writing start \r\n");
  Counter = 0;
  while(1)
    {
      I2C_EEPROM_Write(RxBuffer[Counter], ROM_Address + Counter);
      I2C_EEPROM_Poll();

      cputchar(RxBuffer[Counter]);
      cprintf("..");
      if(RxBuffer[Counter] == '\0')
      {
        break;
      }
      Counter++;
    }
  cprintf("Write Complete. Reset and confirm your message is memorized.\r\n");

  while (1)
    {
    }

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
  I2C_InitStructure.I2C_Ack = I2C_Ack_Enable;
  I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit;
  I2C_InitStructure.I2C_ClockSpeed = I2C_CLOCK;

  /* I2C Peripheral Enable */
  I2C_Cmd(I2C1, ENABLE);
  /* Apply I2C configuration after enabling it */
  I2C_Init(I2C1, &I2C_InitStructure);
}

void I2C_EEPROM_Write(uint8_t Data, uint16_t Address)
{

  /* Send STRAT condition */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C1, (uint8_t)((Address & 0xFF00) >> 8));
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the EEPROM's internal address to write to : LSB of the address */
  I2C_SendData(I2C1, (uint8_t)(Address & 0x00FF));
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the byte to be written */
  I2C_SendData(I2C1, Data);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);
}

void I2C_EEPROM_Poll(void)
{
  __IO uint16_t SR1_Tmp = 0;

  do
  {
    /* Send START condition */
    I2C_GenerateSTART(I2C1, ENABLE);

    /* Read I2C_EE SR1 register to clear pending flags */
    SR1_Tmp = I2C_ReadRegister(I2C1, I2C_Register_SR1);

    /* Send EEPROM address for write */
    I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Transmitter);

  }while(!(I2C_ReadRegister(I2C1, I2C_Register_SR1) & 0x0002));

  /* Clear AF flag */
  I2C_ClearFlag(I2C1, I2C_FLAG_AF);

  /* STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);
}

uint8_t I2C_EEPROM_Read(uint16_t Address)
{
  uint8_t RxData;
  /* While the bus is busy */
  while(I2C_GetFlagStatus(I2C1, I2C_FLAG_BUSY));
  /* Send START condition */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED)){}
  /* Send the EEPROM's internal address to read from: MSB of the address first */
  I2C_SendData(I2C1, (uint8_t)((Address & 0xFF00) >> 8));
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the EEPROM's internal address to read from: LSB of the address */
  I2C_SendData(I2C1, (uint8_t)(Address & 0x00FF));
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STRAT condition a second time */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for read */
  I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Receiver);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED));
  /* Disable Acknowledgement */
  I2C_AcknowledgeConfig(I2C1, DISABLE);
  /* Send STOP Condition */
  I2C_GenerateSTOP(I2C1, ENABLE);
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED)){}
  /* Read a byte from the EEPROM */
  RxData = I2C_ReceiveData(I2C1);
  /* Enable Acknowledgement to be ready for another reception */
  I2C_AcknowledgeConfig(I2C1, ENABLE);

  return(RxData);

}
