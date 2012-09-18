/**
  ******************************************************************************
  * @file    lib/util/st7735.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   library for ST7735
  ******************************************************************************
  * @copy
  *
  * This code is made by Yasuo Kawachi
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
#include "st7735.h"
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Public functions -- -------------------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void ST7735_GPIO_Write_Configuration(void)
{
  /* GPIOX Periph clock enable */
  RCC_APB2PeriphClockCmd(GPIOX_RCC , ENABLE);
  // GPIO Pin Initialization / D0:D7
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin =
    GPIO_Pin_0 | GPIO_Pin_1 | GPIO_Pin_2 | GPIO_Pin_3 |
    GPIO_Pin_4 | GPIO_Pin_5 | GPIO_Pin_6 | GPIO_Pin_7 ;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void ST7735_GPIO_Read_Configuration(void)
{
  /* GPIOX Periph clock enable */
  RCC_APB2PeriphClockCmd(GPIOX_RCC , ENABLE);
  // GPIO Pin Initialization / D0:D7
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin =
    GPIO_Pin_0 | GPIO_Pin_1 | GPIO_Pin_2 | GPIO_Pin_3 |
    GPIO_Pin_4 | GPIO_Pin_5 | GPIO_Pin_6 | GPIO_Pin_7 ;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the GPIO data Pins for out.
  * @param  None
  * @retval : None
  */
void ST7735_GPIO_CTRL_Configuration(void)
{
  /* GPIOY Periph clock enable */
  RCC_APB2PeriphClockCmd(
      GPIOY_0_RCC |
      GPIOY_1_RCC |
      GPIOY_2_RCC |
      GPIOY_3_RCC |
      GPIOY_4_RCC , ENABLE);
  // GPIO Pin Initialization / CS
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin =  GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);
  // GPIO Pin Initialization / CD
  GPIO_InitStructure.GPIO_Pin =  GPIOY_1_PIN;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);
  // GPIO Pin Initialization / RD
  GPIO_InitStructure.GPIO_Pin =  GPIOY_2_PIN;
  GPIO_Init(GPIOY_2_PORT, &GPIO_InitStructure);
  // GPIO Pin Initialization / WR
  GPIO_InitStructure.GPIO_Pin =  GPIOY_3_PIN;
  GPIO_Init(GPIOY_3_PORT, &GPIO_InitStructure);
  // GPIO Pin Initialization / RST
  GPIO_InitStructure.GPIO_Pin =  GPIOY_4_PIN;
  GPIO_Init(GPIOY_4_PORT, &GPIO_InitStructure);

}

/**
  * @brief  Write command to ST7735 LCD
  * @param  command : command to write
  * @retval : None
  */
void ST7735_Write_Command(uint8_t command)
{
  // Set CS  to 0
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
  // Set CD  to 0
  GPIO_ResetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  // Set command to D0:D7
  GPIO_Write(GPIOX_PORT, (GPIO_ReadOutputData(GPIOX_PORT) & 0xFF00) | command);
  // Set WR  to 0
  GPIO_ResetBits(GPIOY_3_PORT, GPIOY_3_PIN);
  // Delay one clock
//  WAIT_1_CLOCK;
  // Set WR  to 1
  GPIO_SetBits(GPIOY_3_PORT, GPIOY_3_PIN);
  // Set CS  to 1
  GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);
}

/**
  * @brief  Write data to ST7735 LCD
  * @param  data : data to write
  * @retval : None
  */
void ST7735_Write_Data(uint8_t data)
{
  // Set CS  to 0
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
  // Set CD  to 1
  GPIO_SetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  // Set WR  to 0
  GPIO_ResetBits(GPIOY_3_PORT, GPIOY_3_PIN);
  // Set command to D0:D7
  GPIO_Write(GPIOX_PORT, (GPIO_ReadOutputData(GPIOX_PORT) & 0xFF00) | data);
  // Set WR  to 1
  GPIO_SetBits(GPIOY_3_PORT, GPIOY_3_PIN);
  // Delay one clock
//  WAIT_1_CLOCK;
  // Set CS  to 1
  GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);
}

/**
  * @brief  Initialise ST7735 LCD
  * @param  : None
  * @retval : None
  */
void ST7735_Init(void)
{
  /* GPIO Configuration*/
  ST7735_GPIO_CTRL_Configuration();

  // Set CS  to 1
  GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);
  // Set CD  to 1
  GPIO_SetBits(GPIOY_1_PORT, GPIOY_1_PIN);
  // Set RD  to 1
  GPIO_SetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  // Set WR  to 1
  GPIO_SetBits(GPIOY_3_PORT, GPIOY_3_PIN);
  // Set RST to 1
  GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);

  // Reset
  GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  delay_ms(100);
  GPIO_ResetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  delay_ms(100);
  GPIO_SetBits(GPIOY_4_PORT, GPIOY_4_PIN);
  delay_ms(100);

  ST7735_GPIO_Write_Configuration();

  ST7735_Write_Command(0x01);
  delay_ms(50);

  ST7735_Write_Command(0x11);//SLEEP OUT
  delay_ms(200);

  ST7735_Write_Command(0xFF);
  ST7735_Write_Data(0x40);
  ST7735_Write_Data(0x03);
  ST7735_Write_Data(0x1A);

  ST7735_Write_Command(0xd9);
  ST7735_Write_Data(0x60);
  ST7735_Write_Command(0xc7);
  ST7735_Write_Data(0x90);
  delay_ms(200);

  ST7735_Write_Command(0xB1);
  ST7735_Write_Data(0x04);
  ST7735_Write_Data(0x25);
  ST7735_Write_Data(0x18);

  ST7735_Write_Command(0xB2);
  ST7735_Write_Data(0x04);
  ST7735_Write_Data(0x25);
  ST7735_Write_Data(0x18);

  ST7735_Write_Command(0xB3);
  ST7735_Write_Data(0x04);
  ST7735_Write_Data(0x25);
  ST7735_Write_Data(0x18);
  ST7735_Write_Data(0x04);
  ST7735_Write_Data(0x25);
  ST7735_Write_Data(0x18);

  ST7735_Write_Command(0xB4);
  ST7735_Write_Data(0x03);

  ST7735_Write_Command(0xB6);
  ST7735_Write_Data(0x15);
  ST7735_Write_Data(0x02);

  ST7735_Write_Command(0xC0);// POWER CONTROL 1 GVDD&VCI1
  ST7735_Write_Data(0x02);
  ST7735_Write_Data(0x70);

  ST7735_Write_Command(0xC1);// POWER CONTROL 2 GVDD&VCI1
  ST7735_Write_Data(0x07);

  ST7735_Write_Command(0xC2);// POWER CONTROL 3 GVDD&VCI1
  ST7735_Write_Data(0x01);
  ST7735_Write_Data(0x01);

  ST7735_Write_Command(0xC3);// POWER CONTROL 4 GVDD&VCI1
  ST7735_Write_Data(0x02);
  ST7735_Write_Data(0x07);

  ST7735_Write_Command(0xC4);// POWER CONTROL 5 GVDD&VCI1
  ST7735_Write_Data(0x02);
  ST7735_Write_Data(0x04);

  ST7735_Write_Command(0xFC);// POWER CONTROL 6 GVDD&VCI1
  ST7735_Write_Data(0x11);
  ST7735_Write_Data(0x17);

  ST7735_Write_Command(0xC5);//VCOMH&VCOML
  ST7735_Write_Data(0x3c);
  ST7735_Write_Data(0x4f);

  ST7735_Write_Command(0x36);//MV,MX,MY,RGB
  ST7735_Write_Data(0b11001000);

  ST7735_Write_Command(0x3a);//GAMMA SET BY REGISTER
  ST7735_Write_Data(0x05);

  //***********************GAMMA*************************
  ST7735_Write_Command(0xE0);
  ST7735_Write_Data(0x06);
  ST7735_Write_Data(0x0E);
  ST7735_Write_Data(0x05);
  ST7735_Write_Data(0x20);
  ST7735_Write_Data(0x27);
  ST7735_Write_Data(0x23);
  ST7735_Write_Data(0x1C);
  ST7735_Write_Data(0x21);
  ST7735_Write_Data(0x20);
  ST7735_Write_Data(0x1C);
  ST7735_Write_Data(0x26);
  ST7735_Write_Data(0x2F);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x03);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x24);

  ST7735_Write_Command(0xE1);
  ST7735_Write_Data(0x06);
  ST7735_Write_Data(0x10);
  ST7735_Write_Data(0x05);
  ST7735_Write_Data(0x21);
  ST7735_Write_Data(0x27);
  ST7735_Write_Data(0x22);
  ST7735_Write_Data(0x1C);
  ST7735_Write_Data(0x21);
  ST7735_Write_Data(0x1F);
  ST7735_Write_Data(0x1D);
  ST7735_Write_Data(0x27);
  ST7735_Write_Data(0x2F);
  ST7735_Write_Data(0x05);
  ST7735_Write_Data(0x03);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x3F);

  //***************************RAM ADDRESS*******************
  ST7735_Write_Command(0x2A);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x02);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x81);

  ST7735_Write_Command(0x2B);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x03);
  ST7735_Write_Data(0x00);
  ST7735_Write_Data(0x82);

  ST7735_Write_Command(0x29);
  delay_ms(100);
}


/**
  * @brief  Write dot data in RGB format
  * @param  red   : 5bit color depth
  * @param  green : 6bit color depth
  * @param  blue  : 5bit color depth
  * @retval : None
  */
void ST7735_Write_RGB(uint8_t red, uint8_t green, uint8_t blue)
{
  ST7735_Write_Data( (red<<3) | (green>>3));
  ST7735_Write_Data( (green<<5) | (blue));
}
