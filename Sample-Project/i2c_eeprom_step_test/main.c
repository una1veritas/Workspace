/**
  ******************************************************************************
  * @file    i2c_eeprom_step_test/main.c
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
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define EEPROM_SLAVE_ADDRESS       0xA0
#define WriteAddress               0x0001
#define I2C_CLOCK                  50000
#define WRITE_DATA                 0b01000001
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Here I show you step by step flow of writing and reading I2C EEPROM\r\n\r\n";
/* Private function prototypes -----------------------------------------------*/
void I2C_Configuration(void);
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

  I2C_Configuration();

  //====================
  //    Write Phase
  //====================

  cprintf("***Write Phase Started***\r\n");

  /* Send STRAT condition */
  I2C_GenerateSTART(I2C1, ENABLE);

  cprintf("START Condition Generated\r\n");

  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

  cprintf("I2C_EVENT_MASTER_MODE_SELECT occurred\r\n");

  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Transmitter);

  cprintf("EEPROM address and direction bit(Master Transmitter) send\r\n");

  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));

  cprintf("I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED occurred\r\n");

  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C1, (uint8_t)((WriteAddress & 0xFF00) >> 8));

  cprintf("Upper EEPROM address send\r\n");

  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

  cprintf("I2C_EVENT_MASTER_BYTE_TRANSMITTED occurred\r\n");

  /* Send the EEPROM's internal address to write to : LSB of the address */
  I2C_SendData(I2C1, (uint8_t)(WriteAddress & 0x00FF));

  cprintf("Lower EEPROM address send\r\n");

  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

  cprintf("I2C_EVENT_MASTER_BYTE_TRANSMITTED occurred\r\n");

  /* Send the byte to be written */
  I2C_SendData(I2C1, WRITE_DATA);

  cprintf("%b is send to EEPROM as write data\r\n", WRITE_DATA);

  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

  cprintf("I2C_EVENT_MASTER_BYTE_TRANSMITTED occurred\r\n");

  /* Send STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);

  cprintf("STOP Condition Generated\r\n");

  //====================
  //   Polling Phase
  //====================

  cprintf("***Polling Phase Started***\r\n");

  __IO uint16_t SR1_Tmp = 0;
  __IO uint32_t polling_count = 0;

  do
  {
    /* Send START condition */
    I2C_GenerateSTART(I2C1, ENABLE);

    cputchar('.');

    /* Read I2C_EE SR1 register to clear pending flags */
    SR1_Tmp = I2C_ReadRegister(I2C1, I2C_Register_SR1);

    /* Send EEPROM address for write */
    I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Transmitter);

    polling_count++;

  }while(!(I2C_ReadRegister(I2C1, I2C_Register_SR1) & 0x0002));

  /* Clear AF flag */
  I2C_ClearFlag(I2C1, I2C_FLAG_AF);

  /* STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);

  cprintf("Polled %u times. Polling cleared.\r\n", polling_count);

  //====================
  //    Read Phase
  //====================

  cprintf("***Read Phase Started***\r\n");

  /* While the bus is busy */
  while(I2C_GetFlagStatus(I2C1, I2C_FLAG_BUSY));

  cprintf("Waited while bus is busy\r\n");

  /* Send START condition */
  I2C_GenerateSTART(I2C1, ENABLE);

  cprintf("START condition generated\r\n");

  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

  cprintf("I2C_EVENT_MASTER_MODE_SELECT occurred\r\n");

  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Transmitter);

  cprintf("EEPROM address and direction bit(Master Transmitter) send\r\n");

  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED)){}

  cprintf("I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED occurred\r\n");

  cprintf("EV6 cleared\r\n");

  /* Send the EEPROM's internal address to read from: MSB of the address first */
  I2C_SendData(I2C1, (uint8_t)((WriteAddress & 0xFF00) >> 8));

  cprintf("Upper EEPROM address send\r\n");

  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

  cprintf("I2C_EVENT_MASTER_BYTE_TRANSMITTED occurred\r\n");

  /* Send the EEPROM's internal address to read from: LSB of the address */
  I2C_SendData(I2C1, (uint8_t)(WriteAddress & 0x00FF));

  cprintf("Lower EEPROM address send\r\n");

  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

  cprintf("I2C_EVENT_MASTER_BYTE_TRANSMITTED occurred\r\n");

  /* Send STRAT condition a second time */
  I2C_GenerateSTART(I2C1, ENABLE);

  cprintf("*RE*START condition generated\r\n");

  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

  cprintf("I2C_EVENT_MASTER_MODE_SELECT occurred\r\n");

  /* Send EEPROM address for read */
  I2C_Send7bitAddress(I2C1, EEPROM_SLAVE_ADDRESS, I2C_Direction_Receiver);

  cprintf("EEPROM address and direction bit(Master Receiver) send\r\n");

  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED));

//  cprintf("I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED occurred\r\n");

  /* Disable Acknowledgement */
  I2C_AcknowledgeConfig(I2C1, DISABLE);

  /* Send STOP Condition */
  I2C_GenerateSTOP(I2C1, ENABLE);

//  cprintf("Disabled Acknowledgement and prepare STOP for following data receive\r\n");

  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED)){}

  cprintf("I2C_EVENT_MASTER_BYTE_RECEIVED occurred\r\n");

  /* Read a byte from the EEPROM */
  cprintf("Data read from EEPROM is %b", I2C_ReceiveData(I2C1));

  /* Enable Acknowledgement to be ready for another reception */
  I2C_AcknowledgeConfig(I2C1, ENABLE);

  cprintf("\r\nAcknowledge enabled. All done.\r\n");

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

  /* Apply I2C configuration */
  I2C_Init(I2C1, &I2C_InitStructure);

  /* I2C Peripheral Enable */
  I2C_Cmd(I2C1, ENABLE);
}
