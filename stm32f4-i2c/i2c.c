/*
 * i2c.c
 *
 *  Created on: 2012/11/01
 *      Author: sin
 */

#include <stm32f4xx_rcc.h>
#include <stm32f4xx_syscfg.h>
#include "i2c.h"

/**
  * @brief  Initializes the GPIO pins used by the IO expander.
  * @param  None
  * @retval None
  */
void I2C1_GPIO_Config(void)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  /* Enable IOE_I2C and IOE_I2C_GPIO_PORT & Alternate Function clocks */
  RCC_APB1PeriphClockCmd(I2C1_CLK, ENABLE);
  RCC_AHB1PeriphClockCmd(I2C1_SCL_GPIO_CLK | I2C1_SDA_GPIO_CLK | IOE_IT_GPIO_CLK, ENABLE);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);

  /* Reset IOE_I2C IP */
  RCC_APB1PeriphResetCmd(I2C1_CLK, ENABLE);
  /* Release reset signal of IOE_I2C IP */
  RCC_APB1PeriphResetCmd(I2C1_CLK, DISABLE);

  /* Connect PXx to I2C_SCL*/
  GPIO_PinAFConfig(I2C1_SCL_GPIO_PORT, I2C1_SCL_SOURCE, I2C1_SCL_AF);
  /* Connect PXx to I2C_SDA*/
  GPIO_PinAFConfig(I2C1_SDA_GPIO_PORT, I2C1_SDA_SOURCE, I2C1_SDA_AF);

  /* IOE_I2C SCL and SDA pins configuration */
  GPIO_InitStructure.GPIO_Pin =  I2C1_SCL_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_OType = GPIO_OType_OD;
  GPIO_InitStructure.GPIO_PuPd  = GPIO_PuPd_NOPULL;
  GPIO_Init(I2C1_SCL_GPIO_PORT, &GPIO_InitStructure);

  GPIO_InitStructure.GPIO_Pin = I2C1_SDA_PIN;
  GPIO_Init(I2C1_SDA_GPIO_PORT, &GPIO_InitStructure);

  /* Set EXTI pin as Input PullUp - IO_Expander_INT */
  GPIO_InitStructure.GPIO_Pin = IOE_IT_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN;
  GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
  GPIO_Init(IOE_IT_GPIO_PORT, &GPIO_InitStructure);

//  /* Connect Button EXTI Line to Button GPIO Pin */
//  SYSCFG_EXTILineConfig(IOE_IT_EXTI_PORT_SOURCE, IOE_IT_EXTI_PIN_SOURCE);
}

void I2C1_Config(void)
{
  I2C_InitTypeDef I2C_InitStructure;

  /* IOE_I2C configuration */
  I2C_InitStructure.I2C_Mode = I2C_Mode_I2C;
  I2C_InitStructure.I2C_DutyCycle = I2C_DutyCycle_2;
  I2C_InitStructure.I2C_OwnAddress1 = 0x00;
  I2C_InitStructure.I2C_Ack = I2C_Ack_Enable;
  I2C_InitStructure.I2C_AcknowledgedAddress = I2C_AcknowledgedAddress_7bit;
  I2C_InitStructure.I2C_ClockSpeed = I2C1_SPEED;

  I2C_Init(I2C1, &I2C_InitStructure);
}

uint8_t IOE_Reset(uint8_t DeviceAddr)
{
  /* Power Down the IO_Expander */
  I2C_WriteDeviceRegister(DeviceAddr, IOE_REG_SYS_CTRL1, 0x02);

  /* wait for a delay to insure registers erasing */
  _delay_(2);

  /* Power On the Codec after the power off => all registers are reinitialized*/
  I2C_WriteDeviceRegister(DeviceAddr, IOE_REG_SYS_CTRL1, 0x00);

  /* If all OK return IOE_OK */
  return IOE_OK;
}

/**
  * @brief  Writes a value in a register of the device through I2C.
  * @param  DeviceAddr: The address of the IOExpander, could be : IOE_1_ADDR
  *         or IOE_2_ADDR.
  * @param  RegisterAddr: The target register address
  * @param  RegisterValue: The target register value to be written
  * @retval IOE_OK: if all operations are OK. Other value if error.
  */
uint8_t I2C_WriteDeviceRegister(uint8_t DeviceAddr, uint8_t RegisterAddr, uint8_t RegisterValue)
{
  uint32_t read_verif = 0;
  uint8_t IOE_BufferTX = 0;

  /* Get Value to be written */
  IOE_BufferTX = RegisterValue;

  /* Configure DMA Peripheral */
  IOE_DMA_Config(IOE_DMA_TX, (uint8_t*)(&IOE_BufferTX));

  /* Enable the I2C peripheral */
  I2C_GenerateSTART(IOE_I2C, ENABLE);

  /* Test on SB Flag */
  IOE_TimeOut = TIMEOUT_MAX;
  while (I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_SB) == RESET)
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Transmit the slave address and enable writing operation */
  I2C_Send7bitAddress(IOE_I2C, DeviceAddr, I2C_Direction_Transmitter);

  /* Test on ADDR Flag */
  IOE_TimeOut = TIMEOUT_MAX;
  while (!I2C_CheckEvent(IOE_I2C, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Transmit the first address for r/w operations */
  I2C_SendData(IOE_I2C, RegisterAddr);

  /* Test on TXE FLag (data dent) */
  IOE_TimeOut = TIMEOUT_MAX;
  while ((!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_TXE)) && (!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_BTF)))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Enable I2C DMA request */
  I2C_DMACmd(IOE_I2C,ENABLE);

  /* Enable DMA TX Channel */
  DMA_Cmd(IOE_DMA_TX_STREAM, ENABLE);

  /* Wait until DMA Transfer Complete */
  IOE_TimeOut = TIMEOUT_MAX;
  while (!DMA_GetFlagStatus(IOE_DMA_TX_STREAM,IOE_DMA_TX_TCFLAG))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Wait until BTF Flag is set before generating STOP */
  IOE_TimeOut = 2 * TIMEOUT_MAX;
  while ((!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_BTF)))
  {
  }

  /* Send STOP Condition */
  I2C_GenerateSTOP(IOE_I2C, ENABLE);

  /* Disable DMA TX Channel */
  DMA_Cmd(IOE_DMA_TX_STREAM, DISABLE);

  /* Disable I2C DMA request */
  I2C_DMACmd(IOE_I2C,DISABLE);

  /* Clear DMA TX Transfer Complete Flag */
  DMA_ClearFlag(IOE_DMA_TX_STREAM,IOE_DMA_TX_TCFLAG);

#ifdef VERIFY_WRITTENDATA
  /* Verify (if needed) that the loaded data is correct  */

  /* Read the just written register*/
  read_verif = I2C_ReadDeviceRegister(DeviceAddr, RegisterAddr);
  /* Load the register and verify its value  */
  if (read_verif != RegisterValue)
  {
    /* Control data wrongly transferred */
    read_verif = IOE_FAILURE;
  }
  else
  {
    /* Control data correctly transferred */
    read_verif = 0;
  }
#endif

  /* Return the verifying value: 0 (Passed) or 1 (Failed) */
  return read_verif;
}


/**
  * @brief  Reads a register of the device through I2C.
  * @param  DeviceAddr: The address of the device, could be : IOE_1_ADDR
  *         or IOE_2_ADDR.
  * @param  RegisterAddr: The target register address (between 00x and 0x24)
  * @retval The value of the read register (0xAA if Timeout occurred)
  */
uint8_t I2C_ReadDeviceRegister(uint8_t DeviceAddr, uint8_t RegisterAddr)
{
  uint8_t IOE_BufferRX[2] = {0x00, 0x00};

  /* Configure DMA Peripheral */
  IOE_DMA_Config(IOE_DMA_RX, (uint8_t*)IOE_BufferRX);

  /* Enable DMA NACK automatic generation */
  I2C_DMALastTransferCmd(IOE_I2C, ENABLE);

  /* Enable the I2C peripheral */
  I2C_GenerateSTART(IOE_I2C, ENABLE);

  /* Test on SB Flag */
  IOE_TimeOut = TIMEOUT_MAX;
  while (!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_SB))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Send device address for write */
  I2C_Send7bitAddress(IOE_I2C, DeviceAddr, I2C_Direction_Transmitter);

  /* Test on ADDR Flag */
  IOE_TimeOut = TIMEOUT_MAX;
  while (!I2C_CheckEvent(IOE_I2C, I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Send the device's internal address to write to */
  I2C_SendData(IOE_I2C, RegisterAddr);

  /* Test on TXE FLag (data dent) */
  IOE_TimeOut = TIMEOUT_MAX;
  while ((!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_TXE)) && (!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_BTF)))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Send START condition a second time */
  I2C_GenerateSTART(IOE_I2C, ENABLE);

  /* Test on SB Flag */
  IOE_TimeOut = TIMEOUT_MAX;
  while (!I2C_GetFlagStatus(IOE_I2C,I2C_FLAG_SB))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Send IOExpander address for read */
  I2C_Send7bitAddress(IOE_I2C, DeviceAddr, I2C_Direction_Receiver);

  /* Test on ADDR Flag */
  IOE_TimeOut = TIMEOUT_MAX;
  while (!I2C_CheckEvent(IOE_I2C, I2C_EVENT_MASTER_RECEIVER_MODE_SELECTED))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Enable I2C DMA request */
  I2C_DMACmd(IOE_I2C,ENABLE);

  /* Enable DMA RX Channel */
  DMA_Cmd(IOE_DMA_RX_STREAM, ENABLE);

  /* Wait until DMA Transfer Complete */
  IOE_TimeOut = 2 * TIMEOUT_MAX;
  while (!DMA_GetFlagStatus(IOE_DMA_RX_STREAM,IOE_DMA_RX_TCFLAG))
  {
    if (IOE_TimeOut-- == 0) return(IOE_TimeoutUserCallback());
  }

  /* Send STOP Condition */
  I2C_GenerateSTOP(IOE_I2C, ENABLE);

  /* Disable DMA RX Channel */
  DMA_Cmd(IOE_DMA_RX_STREAM, DISABLE);

  /* Disable I2C DMA request */
  I2C_DMACmd(IOE_I2C,DISABLE);

  /* Clear DMA RX Transfer Complete Flag */
 DMA_ClearFlag(IOE_DMA_RX_STREAM,IOE_DMA_RX_TCFLAG);

  /* return a pointer to the IOE_Buffer */
  return (uint8_t)IOE_BufferRX[0];
}

