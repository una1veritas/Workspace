/**
  ******************************************************************************
  * @file    spi_accelerometer_monitor/main.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   SD test terminal
  ******************************************************************************
  * @copy
  *
  * This code is made by Yasuo Kawachi.
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
  *  3. Neither the name of the copyright holders nor the names of contributors
  *  may be used to endorse or promote products derived from this software
  *  without specific prior written permission.
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

  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include "platform_config.h"
#include "com_config.h"
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Tilt Accelerometer to change values.\r\n"
  "You may shake it to get over 1G value.\r\n";
int16_t Axes_Value[3];
/* Private function prototypes -----------------------------------------------*/
void SPI_Configuration(void);
void Select_CS(void);
void Deselect_CS(void);
uint8_t Read_MMA745xL(uint8_t address);
void Write_MMA745xL(uint8_t address, uint8_t data);
void Set_Offset_MMA745xL(int16_t xoff, int16_t yoff, int16_t zoff);
void Get_Axes_MMA745xL(int16_t* axes);
/* Public functions -- -------------------------------------------------------*/
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

  //Initialize GPIO and SPI
  SPI_Configuration();

  cprintf(Welcome_Message);

  //set CS high
  Deselect_CS();

  /*
  $16: Mode Control Register (Read/Write)
  MODE [1:0]
  00: Standby Mode
  01: Measurement Mode
  10: Level Detection Mode
  11: Pulse Detection Mode
  GLVL [3:2]
  00: 8g is selected for measurement range.
  10: 4g is selected for measurement range.
  01: 2g is selected for measurement range.
  STON [4:4]
  0: Self-test is not enabled
  1: Self-test is enabled
  SPI3W [5:5]
  0: SPI is 4 wire mode
  1: SPI is 3 wire mode
  DRPD [6:6]
  0: Data ready status is output to INT1/DRDY PIN
  1: Data ready status is not output to INT1/DRDY PIN
  */
  // set 2G Measurement mode
  Write_MMA745xL(0x16, 0b01010101);

  // set offset to make neutral value correct
  Set_Offset_MMA745xL(+80, +84, -147);

  while(1)
    {
      Get_Axes_MMA745xL(Axes_Value);
      cprintf("X:%2D Y:%2D Z:%2D\r\n", Axes_Value[0], Axes_Value[1], Axes_Value[2]);
      delay_ms(300);
    }
}

/**
  * @brief  Configure the SPI1
  * @param  None
  * @retval : None
  */
void SPI_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(SPI1_RCC | SPI1_GPIO_RCC , ENABLE);

  SPI_InitTypeDef  SPI_InitStructure;
  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure SPI1 pins: SCK, MISO and MOSI */
  GPIO_InitStructure.GPIO_Pin = SPI1_SCK_PIN | SPI1_MISO_PIN | SPI1_MOSI_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(SPI1_PORT, &GPIO_InitStructure);

  /* Configure SPI1_NSS: xCS : output push-pull */
  GPIO_InitStructure.GPIO_Pin = SPI1_NSS_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_Init(SPI1_PORT, &GPIO_InitStructure);

  /* SPI1 configuration */
  SPI_InitStructure.SPI_Direction = SPI_Direction_2Lines_FullDuplex;
  SPI_InitStructure.SPI_Mode = SPI_Mode_Master;
  SPI_InitStructure.SPI_DataSize = SPI_DataSize_8b;
  SPI_InitStructure.SPI_CPOL = SPI_CPOL_Low;
  SPI_InitStructure.SPI_CPHA = SPI_CPHA_1Edge;
  SPI_InitStructure.SPI_NSS = SPI_NSS_Soft;
  SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_16;
  SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;
  SPI_Init(SPI1, &SPI_InitStructure);

  /* Enable SPI1  */
  SPI_Cmd(SPI1, ENABLE);
}

/**
  * @brief  select CS
  * @param   None
  * @retval  None
  */
void Select_CS(void)
{
  /* Select the CS: Chip Select low */
  GPIO_ResetBits(SPI1_PORT, SPI1_NSS_PIN);
}

/**
  * @brief  deselect CS
  * @param   None
  * @retval  None
  */
void Deselect_CS(void)
{
  /* Deselect the CS: Chip Select high */
  GPIO_SetBits(SPI1_PORT, SPI1_NSS_PIN);
}

/**
  * @brief   Read the specified register of MMA745xL
  * @param   address:register number
  * @retval  data read from MMA745xL
  */
uint8_t Read_MMA745xL(uint8_t address)
{
  Select_CS();

  uint16_t data_read;

  //Send Read instruction and register address
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
    /* Send SPI1 : read instruction*/
  SPI_I2S_SendData(SPI1, 0b00000000 | (address << 1) );
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  // dummy read
  data_read = SPI_I2S_ReceiveData(SPI1);

  //Read data from MMA745xL
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send Null data*/
  SPI_I2S_SendData(SPI1, 0x0);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  // dummy read
  data_read = SPI_I2S_ReceiveData(SPI1);

  Deselect_CS();

  return (uint8_t)data_read;
}

/**
  * @brief   Write the specified register of MMA745xL
  * @param   address:register number
  * @param   data:data to be written
  * @retval  none
  */
void Write_MMA745xL(uint8_t address, uint8_t data)
{
  Select_CS();

  uint16_t data_read;

  //Send Write instruction and register address
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
    /* Send SPI1 : write instruction*/
  SPI_I2S_SendData(SPI1, 0b10000000 | ( address << 1) );
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  // dummy read
  data_read = SPI_I2S_ReceiveData(SPI1);

  //Write data to MMA745xL
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send data to be written*/
  SPI_I2S_SendData(SPI1, data);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  // dummy read
  data_read = SPI_I2S_ReceiveData(SPI1);

  Deselect_CS();
}

/**
  * @brief   Set offset of MMA745xL  for adjusting neutral value
  * @param   xoff: adjusting value for X axis
  * @param   yoff: adjusting value for Y axis
  * @param   zoff: adjusting value for Z axis
  * @retval  none
  */
void Set_Offset_MMA745xL(int16_t xoff, int16_t yoff, int16_t zoff)
{

  Write_MMA745xL(0x10, xoff);
  Write_MMA745xL(0x11, (xoff & 0b0000001100000000) >> 8 | (xoff & 0b1000000000000000) >> 13);
  Write_MMA745xL(0x12, yoff);
  Write_MMA745xL(0x13, (yoff & 0b0000001100000000) >> 8 | (yoff & 0b1000000000000000) >> 13);
  Write_MMA745xL(0x14, zoff);
  Write_MMA745xL(0x15, (zoff & 0b0000001100000000) >> 8 | (zoff & 0b1000000000000000) >> 13);
}

/**
  * @brief   Get Acceleration value for each axis
  * @param   axes : array to contain value
  * @retval  none
  */
void Get_Axes_MMA745xL(int16_t* axes)
{
  uint16_t temp;

  axes[0] = Read_MMA745xL(0x00);
  temp = Read_MMA745xL(0x01);
  axes[0] |= (temp & 0b00000001 ) << 8;
  if (temp & 0b00000010 )
    {
      axes[0] |= 0b1111111000000000;
    }
  else
    {
      axes[0] &= 0b0000000111111111;
    }

  axes[1] = Read_MMA745xL(0x02);
  temp = Read_MMA745xL(0x03);
  axes[1] |= (temp & 0b00000001 ) << 8;
  axes[1] |= (temp & 0b00000010 ) << 14;
  if (temp & 0b00000010 )
    {
      axes[1] |= 0b1111111000000000;
    }
  else
    {
      axes[1] &= 0b0000000111111111;
    }

  axes[2] = Read_MMA745xL(0x04);
  temp = Read_MMA745xL(0x05);
  axes[2] |= (temp & 0b00000001 ) << 8;
  axes[2] |= (temp & 0b00000010 ) << 14;
  if (temp & 0b00000010 )
    {
      axes[2] |= 0b1111111000000000;
    }
  else
    {
      axes[2] &= 0b0000000111111111;
    }
}


