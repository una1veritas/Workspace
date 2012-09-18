
/*************************************************************************
 *
 *    Used with ICCARM and AARM.
 *
 *    (c) Copyright IAR Systems 2008
 *
 *    File name   : sd_ll_spi1.h
 *    Description : SD/MMC low level SPI1 driver
 *
 *    History :
 *    1. Date        : April 10, 2008
 *       Author      : Stanimir Bonev
 *       Description : Create
 *
 *    $Revision: #2 $
 **************************************************************************/
#include "includes.h"
#define _SSP_FIFO_SIZE 8

/*************************************************************************
 * Function Name: SdPowerOn
 * Parameters: none
 * Return: none
 *
 * Description: Set power off state
 *
 *************************************************************************/
void SdPowerOn (void)
{
  SdDly_1ms(1);
}
/*************************************************************************
 * Function Name: SdPowerOff
 * Parameters: none
 * Return: none
 *
 * Description: Set power off state
 *
 *************************************************************************/
void SdPowerOff (void)
{
  SdDly_1ms(1);
}
/*************************************************************************
 * Function Name: SdChipSelect
 * Parameters: Boolean Select
 * Return: none
 *
 * Description: SD/MMC Chip select control
 * Select = true  - Chip is enable
 * Select = false - Chip is disable
 *
 *************************************************************************/
void SdChipSelect (Boolean Select)
{
GPIO_InitTypeDef GPIO_InitStructure;
  if (Select)
  {

      GPIO_ResetBits(MMC_CS_PORT, MMC_CS_PIN);
      /* Configure the MMC_CS pin as output*/
      GPIO_InitStructure.GPIO_Pin = MMC_CS_PIN;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
      GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

      GPIO_Init(MMC_CS_PORT, &GPIO_InitStructure);
  }
  else
  {

      /* Configure the MMC_CS pin as output*/
      GPIO_InitStructure.GPIO_Pin = MMC_CS_PIN;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN;
      GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

      GPIO_Init(MMC_CS_PORT, &GPIO_InitStructure);
      // Synchronization
      while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_TXE) == RESET);
      SPI_I2S_SendData(MMC_SPI, 0xFF);
      // Wait until tx fifo and tx shift bufer are empty
/* the busy flag doesn't go to LOW every time as it should do */
//      while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_BSY) == SET);
      /* Wait for SPIz data reception */
      while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_RXNE) == RESET);
      do
      {
        SPI_I2S_ReceiveData(MMC_SPI);
      }
      while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_RXNE) == SET);
  }
}
/*************************************************************************
 * Function Name: SdPresent
 * Parameters: none
 * Return: Boolean - true cart present
 *                 - false cart no present
 *
 * Description: SD/MMC precent check
 *
 *************************************************************************/
Boolean SdPresent (void)
{
  return(Bit_SET == GPIO_ReadInputDataBit(MMC_CS_PORT,MMC_CS_PIN));
}

/*************************************************************************
 * Function Name: SdWriteProtect
 * Parameters: none
 * Return: Boolean - true cart is protected
 *                 - false cart no protected
 *
 * Description: SD/MMC Write protect check
 *
 *************************************************************************/
Boolean SdWriteProtect (void)
{
  return(0);
}

/*************************************************************************
 * Function Name: SdSetClockFreq
 * Parameters: Int32U Frequency
 * Return: Int32U
 *
 * Description: Set SPI ckl frequency
 *
 *************************************************************************/
Int32U SdSetClockFreq (Int32U Frequency)
{
RCC_ClocksTypeDef RCC_Clocks;
Int32U Div;

  RCC_GetClocksFreq(&RCC_Clocks);

  for(Div = 0; Div < 8; Div++)
  {
    if(Frequency * (2<<Div) > RCC_Clocks.PCLK1_Frequency)
    {
      break;
    }
  }

  if(8 <= Div) return (-1UL);

  Int32U tmpreg;
  tmpreg = MMC_SPI->CR1;
  tmpreg &= ~SPI_BaudRatePrescaler_256;
  tmpreg |= Div<<3;
  MMC_SPI->CR1 = tmpreg;
  // Return real frequency
  return(RCC_Clocks.PCLK1_Frequency/(2<<Div));
}

/*************************************************************************
 * Function Name: SdInit
 * Parameters: none
 * Return: none
 *
 * Description: Init SPI, Cart Present, Write Protect and Chip select pins
 *
 *************************************************************************/
void SdInit (void)
{
  SPI_InitTypeDef   SPI_InitStructure;
  GPIO_InitTypeDef  GPIO_InitStructure;

  // SPI and GPIO enable clock
  RCC_AHB1PeriphClockCmd(MMC_CS_CLK | MMC_SPI_CLK | MMC_SPI_GPIO_CLK, ENABLE);
  RCC_APB1PeriphClockCmd(MMC_SPI_CLK, ENABLE);

  GPIO_PinAFConfig(MMC_SPI_GPIO, MMC_PIN_SCK_SOURCE, GPIO_AF_SPI3);
  GPIO_PinAFConfig(MMC_SPI_GPIO, MMC_PIN_MISO_SOURCE, GPIO_AF_SPI3);
  GPIO_PinAFConfig(MMC_SPI_GPIO, MMC_PIN_MOSI_SOURCE, GPIO_AF_SPI3);

  // Init CS pin
  GPIO_InitStructure.GPIO_Pin = MMC_CS_PIN;
  GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN; // input to enable card detect
  GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(MMC_CS_PORT, &GPIO_InitStructure);

  // init SCK, MISO and MOSI pins
  GPIO_InitStructure.GPIO_Pin = MMC_PIN_SCK | MMC_PIN_MOSI | MMC_PIN_MISO;
  GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
  GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_UP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(MMC_SPI_GPIO, &GPIO_InitStructure);

  SPI_InitStructure.SPI_Direction = SPI_Direction_2Lines_FullDuplex;
  SPI_InitStructure.SPI_Mode = SPI_Mode_Master;
  SPI_InitStructure.SPI_DataSize = SPI_DataSize_8b;
  SPI_InitStructure.SPI_CPOL = SPI_CPOL_Low;
  SPI_InitStructure.SPI_CPHA = SPI_CPHA_1Edge;
  SPI_InitStructure.SPI_NSS = SPI_NSS_Soft;
  SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_256;
  SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;
  SPI_InitStructure.SPI_CRCPolynomial = 7;
  SPI_Init(MMC_SPI, &SPI_InitStructure);

  // Clock Freq. Identification Mode < 400kHz
  SdSetClockFreq(IdentificationModeClock);
  /* Enable SPIy */
  SPI_Cmd(MMC_SPI, ENABLE);

}

/*************************************************************************
 * Function Name: SdTranserByte
 * Parameters: Int8U ch
 * Return: Int8U
 *
 * Description: Read byte from SPI
 *
 *************************************************************************/
Int8U SdTranserByte (Int8U ch)
{
  /* Wait for SPIy Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_TXE) == RESET);
  SPI_I2S_SendData(MMC_SPI, ch);
  while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_RXNE) == RESET);
  return(SPI_I2S_ReceiveData(MMC_SPI));
}

/*************************************************************************
 * Function Name: SdSendBlock
 * Parameters: pInt8U pData, Int32U Size
 *
 * Return: void
 *
 * Description: Read byte from SPI
 *
 *************************************************************************/
void SdSendBlock (pInt8U pData, Int32U Size)
{
Int32U OutCount = Size;
  while (OutCount)
  {
    if(SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_TXE) == SET);
    {
      SPI_I2S_SendData(MMC_SPI, *pData++);
      --OutCount;
    }
  }

  while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_BSY) == SET);

  while (SPI_I2S_GetFlagStatus(MMC_SPI, SPI_I2S_FLAG_RXNE) == SET);
  {
    SPI_I2S_ReceiveData(MMC_SPI);
  }
}

/*************************************************************************
 * Function Name: SdReceiveBlock
 * Parameters: pInt8U pData, Int32U Size
 *
 * Return: void
 *
 * Description: Read byte from SPI
 *
 *************************************************************************/
void SdReceiveBlock (pInt8U pData, Int32U Size)
{
//Int32U Delta = 0;
  while (Size)
  {
    *pData++ = SdTranserByte(0xFF);
  }
}

/*************************************************************************
 * Function Name: SdDly_1ms
 * Parameters: Int32U Delay
 * Return: none
 *
 * Description: Delay [msec]
 *
 *************************************************************************/
void SdDly_1ms (Int32U Delay)
{
volatile Int32U i;
  for(;Delay;--Delay)
  {
    for(i = SD_DLY_1MSEC;i;--i);
  }
}
