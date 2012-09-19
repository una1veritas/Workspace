/**
  ******************************************************************************
  * @file    spi_eeprom_read_write_test/main.c
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
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHello Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Here I show you communication with SPI EEPROM.\r\n"
  "I make EEPROM state write enable , write and read data.\r\n\r\n";
/* Private function prototypes -----------------------------------------------*/
void SPI_Configuration(void);
uint8_t SPI_Send_Receive(uint8_t SendData);
void Select_SPI1_CS(void);
void Deselect_SPI1_CS(void);
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  uint16_t ReceivedData =0;

  // Configure board specific setting
  BoardInit();

  SPI_Configuration();
  // Setting up COM port for Print function
  COM_Configuration();
  cprintf(Welcome_Message);

  cprintf("Making EEPROM Write enable...\r\n");
  // Set Write Enable
  //Select SPI1
  Select_SPI1_CS();
  //Send Write Enable Command
  ReceivedData = SPI_Send_Receive(0b00000110);
  //DeSelect SPI1
  Deselect_SPI1_CS();

  cprintf("Receive Status Register\r\n");
  // Read SPI1 Status Register
  //Select SPI1
  Select_SPI1_CS();
  //Send Read Status Register Command
  ReceivedData = SPI_Send_Receive(0b00000101);
  //Send null data and receive response
  ReceivedData = SPI_Send_Receive(0b00000000);
  //DeSelect SPI1
  Deselect_SPI1_CS();

  cprintf("Result is %b\r\n", ReceivedData);

  cprintf("Send Write Byte command, address:00000000, data:0x43:'C'\r\n");
  // Byte Write
  // Read SPI1 Status Register
  //Select SPI1
  Select_SPI1_CS();
  /* Send SPI1 data Write Command*/
  ReceivedData = SPI_Send_Receive(0b00000010);
  /* Send SPI1 data address*/
  ReceivedData = SPI_Send_Receive(0b00000000);
  /* Send SPI1 data */
  ReceivedData = SPI_Send_Receive(0x43);
  //DeSelect SPI1
  Deselect_SPI1_CS();

  while(1)
    {
      // Read SPI1 Status Register
      //Select SPI1
      Select_SPI1_CS();
      //Send Read Status Register Command
      ReceivedData = SPI_Send_Receive(0b00000101);
      //Send null data and receive response
      ReceivedData = SPI_Send_Receive(0b00000000);
      //DeSelect SPI1
      Deselect_SPI1_CS();

      if ((ReceivedData & 0b00000001) ==0)
        {
          cprintf("Write complete!\r\n");
          break;
        }
      cprintf(".");
    }

  cprintf("Read data from address:00000000\r\n");
  // Byte Read
  // Read SPI1 Status Register
  //Select SPI1
  Select_SPI1_CS();
  /* Send SPI1 data Read Command*/
  ReceivedData = SPI_Send_Receive(0b00000011);
  /* Send SPI1 data address*/
  ReceivedData = SPI_Send_Receive(0b00000000);
  //Send null data and receive response
  ReceivedData = SPI_Send_Receive(0b00000000);
  //DeSelect SPI1
  Deselect_SPI1_CS();
  cprintf("What is read from address:00000000 is ");
  cputchar(ReceivedData);
  cprintf("\r\n");

  while(1);

}

/**
  * @brief  Select SPI1_NSS pin
  * @param  None
  * @retval : None
  */
void Select_SPI1_CS(void)
{
  GPIO_ResetBits(SPI1_PORT, SPI1_NSS_PIN);
}

/**
  * @brief  Deselect SPI1_NSS pin
  * @param  None
  * @retval : None
  */
void Deselect_SPI1_CS(void)
{
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_BSY) == SET);
  GPIO_SetBits(SPI1_PORT, SPI1_NSS_PIN);
}

/**
  * @brief  Send receive byte data via SPI
  * @param  Data to be send via SPI
  * @retval : Received data from SPI
  */
uint8_t SPI_Send_Receive(uint8_t SendData)
{
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send SPI1 data Write Command*/
  SPI_I2S_SendData(SPI1, SendData);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  return SPI_I2S_ReceiveData(SPI1);
}

/**
  * @brief  Configure the SPI
  * @param  None
  * @retval : None
  */
void SPI_Configuration(void)
{
  SPI_InitTypeDef  SPI_InitStructure;
  GPIO_InitTypeDef GPIO_InitStructure;

  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(SPI1_RCC | SPI1_GPIO_RCC , ENABLE);

  /* Configure SPI1 pins: SCK, MISO and MOSI */
  GPIO_InitStructure.GPIO_Pin = SPI1_SCK_PIN | SPI1_MISO_PIN | SPI1_MOSI_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(SPI1_PORT, &GPIO_InitStructure);

  /* Configure I/O for Flash Chip select */
  GPIO_InitStructure.GPIO_Pin = SPI1_NSS_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(SPI1_PORT, &GPIO_InitStructure);

  /* Deselect the FLASH: Chip Select high */
  GPIO_SetBits(SPI1_PORT, SPI1_NSS_PIN);

  /* SPI1 configuration */
  SPI_InitStructure.SPI_Direction = SPI_Direction_2Lines_FullDuplex;
  SPI_InitStructure.SPI_Mode = SPI_Mode_Master;
  SPI_InitStructure.SPI_DataSize = SPI_DataSize_8b;
  SPI_InitStructure.SPI_CPOL = SPI_CPOL_Low;
  SPI_InitStructure.SPI_CPHA = SPI_CPHA_1Edge;
  SPI_InitStructure.SPI_NSS = SPI_NSS_Soft;
  SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_256;
  SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;
  SPI_InitStructure.SPI_CRCPolynomial = 7;
  SPI_Init(SPI1, &SPI_InitStructure);

  /* Enable SPI1  */
  SPI_Cmd(SPI1, ENABLE);
}


