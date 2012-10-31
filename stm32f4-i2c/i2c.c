/*
 * i2c.c
 *
 *  Created on: 2012/10/30
 *      Author: sin
 */

#include <stm32f4xx_gpio.h>
#include <stm32f4xx_rcc.h>
#include <stm32f4xx_i2c.h>
#include "i2c.h"

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
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C3, ENABLE);
  //Reset the Peripheral
//  RCC_APB1PeriphResetCmd(RCC_APB1Periph_I2C3, ENABLE);
//  RCC_APB1PeriphResetCmd(RCC_APB1Periph_I2C3, DISABLE);

  /* GPIO Periph clock enable */
  RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA, ENABLE);
  RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);

  /* Configure I2C pins: SCL PA8 GPIO_Pin_8 and SDA PC9 GPIO_Pin_9 */
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
  GPIO_InitStructure.GPIO_PuPd  = GPIO_PuPd_UP;
  GPIO_InitStructure.GPIO_Pin =  GPIO_Pin_8;
  GPIO_Init(GPIOA, &GPIO_InitStructure);
  GPIO_InitStructure.GPIO_Pin =  GPIO_Pin_9;
  GPIO_Init(GPIOC, &GPIO_InitStructure);

  //Connect GPIO pins to peripheral
  GPIO_PinAFConfig(GPIOA, GPIO_PinSource8, GPIO_AF_I2C3);
  GPIO_PinAFConfig(GPIOC, GPIO_PinSource9, GPIO_AF_I2C3);

  /* I2C configuration */
  I2C_InitStructure.I2C_Mode = I2C_Mode_I2C;
  I2C_InitStructure.I2C_DutyCycle = I2C_DutyCycle_2;
  I2C_InitStructure.I2C_OwnAddress1 = 0x00; //We are the master. We don't need this
  I2C_InitStructure.I2C_Ack = I2C_Ack_Enable;
  I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit;
  I2C_InitStructure.I2C_ClockSpeed = 100000; //I2C_CLOCK;

  /* Apply I2C configuration after enabling it */
  I2C_Init(I2C3, &I2C_InitStructure);
  /* I2C Peripheral Enable */
  I2C_Cmd(I2C3, ENABLE);
}

/**
  * @brief  Write Command to ST7032i
  * @param  Data : Command Data
  * @retval None
  */
void ST7032i_Command_Write(uint8_t Data)
{

  /* Send STRAT condition */
  I2C_GenerateSTART(I2C3, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C3, ST7032I_ADDR << 1, I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C3, 0x00); /* 0b00000000); */
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C3, Data);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STOP condition */
  I2C_GenerateSTOP(I2C3, ENABLE);
}

/**
  * @brief  Write Data to ST7032i
  * @param  Data : "Data" Data
  * @retval None
  */
void ST7032i_Data_Write(uint8_t Data)
{

  /* Send STRAT condition */
  I2C_GenerateSTART(I2C3, ENABLE);
  /* Test on EV5 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_MODE_SELECT));
  /* Send EEPROM address for write */
  I2C_Send7bitAddress(I2C3, (ST7032I_ADDR << 1), I2C_Direction_Transmitter);
  /* Test on EV6 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C3, 0x40); /*0b01000000 ); */
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send the EEPROM's internal address to write to : MSB of the address first */
  I2C_SendData(I2C3, Data);
  /* Test on EV8 and clear it */
  while(!I2C_CheckEvent(I2C3, I2C_EVENT_MASTER_BYTE_TRANSMITTED));
  /* Send STOP condition */
  I2C_GenerateSTOP(I2C3, ENABLE);
}


