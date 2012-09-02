/**
  ******************************************************************************
  * @file    i2c_control_ar1000/main.c
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
#define MODULE_SLAVE_ADDRESS       0b0010000
#define I2C_CLOCK                  50000

#define R0_DEFAULT              0xFFFC
#define R1_DEFAULT              0x5B05
#define R2_DEFAULT              0xF4B9
#define R3_DEFAULT              0x9842
#define R14_DEFAULT             0xFC2D

#define R0_ENABLE               0x0001
#define R0_XO_EN                0x0002
#define R1_HMUTE                0x0002
#define R2_CHAN                 0x01FF
#define R2_TUNE                 0x0200
#define R3_VOLUMN_MASK          0xF78F
#define R14_VOLUME2_MASK        0x0FFF

#define REGISTERED_STATIONS     5
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Press n(next) or p(previous) to change stations.\r\n"
  "Press u(up) or d(down) to change volume.\r\n\r\n";
uint8_t i;
int8_t RxData;
struct FM_Stations {
        int8_t name[30];  //name of station
        uint16_t freq; // radio station frequency
};
struct FM_Stations Station[REGISTERED_STATIONS] = {
    {"NHK FM", 881 },
    {"FM Osaka", 851 },
    {"FM802", 802 },
    {"FM Co-Co-Lo", 765 },
    {"UmedaFM BeHappy!789", 789 }
};
uint8_t Station_Number = 0;

uint8_t Volume_Step[22][2] = {
    {0x0, 0xF },
    {0xC, 0xF },
    {0xD, 0xF },
    {0xE, 0xF },
    {0xF, 0xF },
    {0xE, 0xE },
    {0xF, 0xE },
    {0xE, 0xD },
    {0xF, 0xD },
    {0xF, 0xB },
    {0xF, 0xA },
    {0xF, 0x9 },
    {0xF, 0x7 },
    {0xE, 0x6 },
    {0xF, 0x6 },
    {0xE, 0x5 },
    {0xF, 0x5 },
    {0xE, 0x3 },
    {0xF, 0x3 },
    {0xF, 0x2 },
    {0xF, 0x1 },
    {0xF, 0x0 }
    };
uint8_t Volume = 10;

/* Private function prototypes -----------------------------------------------*/
void I2C_Configuration(void);
void AR1000_Write (uint8_t reg, uint16_t data);
uint16_t AR1000_Read(uint8_t reg);
void Print_Station_Info(uint8_t station_num);
void Set_AR1000_Frequency(uint16_t freq);
void Set_AR1000_Volume(uint8_t volume);
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

  delay_ms(1000);

  I2C_Configuration();

  AR1000_Write(0, R0_DEFAULT | R0_XO_EN);
  AR1000_Write(1, R1_DEFAULT | R1_HMUTE);
  AR1000_Write(2, R2_DEFAULT);
  AR1000_Write(3, R3_DEFAULT);
  AR1000_Write(4, 0x0400);
  AR1000_Write(5, 0x28AA);
  AR1000_Write(6, 0x4400);
  AR1000_Write(7, 0x1EE7);
  AR1000_Write(8, 0x7141);
  AR1000_Write(9, 0x007D);
  AR1000_Write(10, 0x82CE);
  AR1000_Write(11, 0x4F55);
  AR1000_Write(12, 0x970C);
  AR1000_Write(13, 0xB845);
  AR1000_Write(14, 0xFC2D);
  AR1000_Write(15, 0x8097);
  AR1000_Write(16, 0x04A1);
  AR1000_Write(17, 0xDF6A);

  AR1000_Write(0, R0_DEFAULT | R0_XO_EN | R0_ENABLE);

  while(!(AR1000_Read(19) & 0x20))
    {
      delay_ms(1);
    }


  while(!(AR1000_Read(19) & 0x20))
    {
      delay_ms(1);
    }

  AR1000_Write(1, R1_DEFAULT);

  Print_Station_Info(Station_Number);
  Set_AR1000_Volume(Volume);

  while (1)
    {
      while(RX_BUFFER_IS_EMPTY){}
      RxData = (int8_t)RECEIVE_DATA;
      switch (RxData)
      {
        case 'n' :
          Station_Number++;
          if (Station_Number == REGISTERED_STATIONS )
            {
              Station_Number = 0;
            }
          Set_AR1000_Frequency(Station_Number);
          Print_Station_Info(Station_Number);
          break;
        case 'p' :
          Station_Number--;
          if (Station_Number > REGISTERED_STATIONS )
            {
              Station_Number = REGISTERED_STATIONS - 1;
            }
          Set_AR1000_Frequency(Station_Number);
          Print_Station_Info(Station_Number);
          break;
        case 'u' :
          if (Volume < 21 )
            {
              Volume++;
            }
          Set_AR1000_Volume(Volume);
          break;

        case 'd' :
          if (Volume != 0 )
            {
              Volume--;
            }
          Set_AR1000_Volume(Volume);
          break;

      }
    }
}

/**
  * @brief  Set frequency on AR1000
  * @param  freq : frequency (MHz*10)
  * @retval None
  */
void Set_AR1000_Frequency(uint16_t freq)
{
  AR1000_Write(2, (R2_DEFAULT & ~(R2_TUNE | R2_CHAN)) | (freq - 690));
  AR1000_Write(2, (R2_DEFAULT & ~(R2_TUNE | R2_CHAN)) | (freq - 690) | R2_TUNE);
}

/**
  * @brief  Set volume on AR1000
  * @param  volume : volume (0-21)
  * @retval None
  */
void Set_AR1000_Volume(uint8_t volume)
{
  AR1000_Write(3, (R3_DEFAULT & R3_VOLUMN_MASK) | (Volume_Step[volume][1] << 7) );
  AR1000_Write(14, (R14_DEFAULT & R14_VOLUME2_MASK) | (Volume_Step[volume][0] << 12) );
  cprintf("Volume: %u\r\n", volume);
}

/**
  * @brief  Print station information
  * @param  station_num : subscripts of Station variable
  * @retval None
  */
void Print_Station_Info(uint8_t station_num)
{
  cprintf("Station tuned : ");
  cprintf(Station[station_num].name);
  cprintf("\r\n");

  cprintf("Frequency : %u.%u MHz\r\n", Station[station_num].freq / 10, Station[station_num].freq % 10);

  cprintf("Signal Level :%u\r\n\r\n", (AR1000_Read(18)&0b1111111000000000)>>9);
}

/**
  * @brief  Write to AR1000 registers
  * @param  reg : register number
  * @param  data : data to be written
  * @retval None
  */
void AR1000_Write(uint8_t reg, uint16_t data)
{
  /* Send STRAT condition */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C1, MODULE_SLAVE_ADDRESS << 1, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));
  /* Send the register number*/
  I2C_SendData(I2C1, (uint8_t)reg);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the upper byte*/
  I2C_SendData(I2C1, (uint8_t)(data>>8));
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the lower byte */
  I2C_SendData(I2C1, (uint8_t)(0x00FF &data));
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STOP condition */
  I2C_GenerateSTOP(I2C1, ENABLE);
}

/**
  * @brief  Read from AR1000 registers
  * @param  reg : register number
  * @retval Register value
  */
uint16_t AR1000_Read(uint8_t reg)
{
  uint16_t return_value = 0;

  /* While the bus is busy */
  while(I2C_GetFlagStatus(I2C1, I2C_FLAG_BUSY));

  /* Send START condition */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

  /* Send module address and behave as transmitter */
  I2C_Send7bitAddress(I2C1, MODULE_SLAVE_ADDRESS << 1, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED)){}

  /* Send module internal register to be read*/
  I2C_SendData(I2C1, (uint8_t)reg);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_TRANSMITTED));

  /* Send STRAT condition a second time */
  I2C_GenerateSTART(I2C1, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_MODE_SELECT));

  /* Send module address and behave as receiver */
  I2C_Send7bitAddress(I2C1, MODULE_SLAVE_ADDRESS << 1, I2C_Direction_Receiver);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED));

  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED)){}
  /* Read a byte from the module */
  return_value = I2C_ReceiveData(I2C1) << 8;

  /* Disable Acknowledgment */
  I2C_AcknowledgeConfig(I2C1, DISABLE);
  /* Send STOP Condition */
  I2C_GenerateSTOP(I2C1, ENABLE);

  while(!I2C_CheckEvent(I2C1, I2C_EVENT_MASTER_BYTE_RECEIVED)){}
  /* Read a byte from the module */
  return_value |= I2C_ReceiveData(I2C1);

  /* Enable Acknowledgement to be ready for another reception */
  I2C_AcknowledgeConfig(I2C1, ENABLE);

  return return_value;
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
