/**
  ******************************************************************************
  * @file    spi_accelerometer_servo/main.c
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
#define NEUTRAL   1350
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Tilt Accelerometer to tilt a avatar on servo.\r\n";
int16_t Axes_Value[8][3],Axes_ave[3];
/* Private function prototypes -----------------------------------------------*/
void SPI_Configuration(void);
void Select_CS(void);
void Deselect_CS(void);
uint8_t Read_MMA745xL(uint8_t address);
void Write_MMA745xL(uint8_t address, uint8_t data);
void Set_Offset_MMA745xL(int16_t xoff, int16_t yoff, int16_t zoff);
void Get_Axes_MMA745xL(int16_t* axes);
void TIM1_Configuration(void);
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

  /* TIM1 Configuration*/
  TIM1_Configuration();

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

  // set servos position neutral
  TIM_SetCompare1(TIM1,NEUTRAL);
  TIM_SetCompare2(TIM1,NEUTRAL);

  uint8_t i,j;
  int16_t sum = 0;
  while(1)
    {
      //get acceleration value and compute average
      for (i=0;i<8;i++)
        {
          Get_Axes_MMA745xL(Axes_Value[i]);
          delay_ms(10);
        }

      for (j=0;j<3;j++)
        {
          sum=0;
          for (i=0;i<8;i++)
            {
              sum += Axes_Value[i][j];
            }
          Axes_ave[j] = sum/8;
        }

      //make X Y value within -64 to +64
      if (Axes_ave[0] < -64) Axes_ave[0] = -64;
      if (Axes_ave[0] > +64) Axes_ave[0] = +64;
      if (Axes_ave[1] < -64) Axes_ave[1] = -64;
      if (Axes_ave[1] > +64) Axes_ave[1] = +64;

      // set servos position neutral
      TIM_SetCompare1(TIM1,NEUTRAL - Axes_ave[0]*9);
      TIM_SetCompare2(TIM1,NEUTRAL - Axes_ave[1]*9);

      delay_ms(10);
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

  axes[0] = (int8_t)Read_MMA745xL(0x06);
  axes[1] = (int8_t)Read_MMA745xL(0x07);
  axes[2] = (int8_t)Read_MMA745xL(0x08);
}

/**
  * @brief  Configure TIM1
  * @param  None
  * @retval : None
  */
void TIM1_Configuration(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
  TIM_OCInitTypeDef  TIM_OCInitStructure;

  //Supply APB2 Clock
  RCC_APB2PeriphClockCmd(TIM1_RCC | TIM1_GPIO_RCC , ENABLE);

  GPIO_InitStructure.GPIO_Pin = TIM1_CH1_PIN | TIM1_CH2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(TIM1_PORT, &GPIO_InitStructure);

#if defined (PARTIAL_REMAP_TIM1)
PartialRemap_TIM1_Configuration();
#elif defined (FULL_REMAP_TIM1)
FullRemap_TIM1_Configuration();
#endif

  /* ---------------------------------------------------------------------------
    TIM1 Configuration: Output Compare Toggle Mode:
    TIM1CLK = 72 MHz, Prescaler = 60, TIM1 counter clock = 1.2MHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  // 2500us cycle
  TIM_TimeBaseStructure.TIM_Period = 17999;
  // 2000us cycle
  //TIM_TimeBaseStructure.TIM_Period = 14400;
  TIM_TimeBaseStructure.TIM_Prescaler = 79;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);

  /* Output Compare Toggle Mode configuration: Channel1 */
  TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
  TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
  TIM_OCInitStructure.TIM_Pulse = 0;
  TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
  TIM_OCInitStructure.TIM_OCIdleState = TIM_OCIdleState_Reset;
  TIM_OCInitStructure.TIM_OutputNState = TIM_OutputNState_Disable;
  TIM_OCInitStructure.TIM_OCNPolarity = TIM_OCNPolarity_High;
  TIM_OCInitStructure.TIM_OCNIdleState = TIM_OCIdleState_Reset;
  TIM_OC1Init(TIM1, &TIM_OCInitStructure);
  TIM_OC1PreloadConfig(TIM1, TIM_OCPreload_Disable);

  /* Output Compare Toggle Mode configuration: Channel2 */
  TIM_OC2Init(TIM1, &TIM_OCInitStructure);
  TIM_OC2PreloadConfig(TIM1, TIM_OCPreload_Disable);

  /* TIM enable counter */
  TIM_Cmd(TIM1, ENABLE);

  /* Main Output Enable */
  TIM_CtrlPWMOutputs(TIM1, ENABLE);
}
