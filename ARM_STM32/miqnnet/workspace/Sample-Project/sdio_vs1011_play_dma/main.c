/**
  ******************************************************************************
  * @file    sdio_vs1011_play_dma/main.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   main program
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
#include "sdcard.h"
#include "ff.h"
#include "diskio.h"
#include "comm.h"
#include "rtc.h"
/* Private typedef -----------------------------------------------------------*/
typedef enum {FAILED = 0, PASSED = !FAILED} TestStatus;
/* Private define ------------------------------------------------------------*/
#define SPI1_Tx_DMA_Channel     DMA1_Channel3
#define SPI1_DR_Base            0x4001300C
#define SPI1_Tx_DMA_FLAG        DMA1_FLAG_TC3
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Play mp3 using VS1011.\r\n"
  "Type any key to stop playing and close a file.\r\n";
FRESULT fsresult;
FATFS myfs;
FIL fmp3;
int8_t filename[] = "/pre.mp3";
uint8_t data_buffer[32];
UINT data_read;
uint8_t ReceivedData, i, RxData;;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void GPIO_Configuration(void);
void SPI_Configuration(void);
void DMA_Configuration(void);
void Select_xCS(void);
void Deselect_xCS(void);
void Select_xDCS(void);
void Deselect_xDCS(void);
void Do_xRESET(void);
void Wait_DREQ_High(void);
uint16_t Read_SCI(uint8_t address);
void Write_SCI(uint8_t address, uint16_t data);
void Write_SDI_DMA(void);
void Set_SPI_PSC_16(void);
TestStatus Initialize_SDIO_SD(void);
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

  // Configure NVIC
  NVIC_Configuration();

  //Initialize GPIO ,SPI and DMA
  GPIO_Configuration();
  SPI_Configuration();
  DMA_Configuration();

  // Setting up COM port for Print function
  COM_Configuration();
  //Print welcome message
  cprintf(Welcome_Message);

  //Initialize SDIO
  if(Initialize_SDIO_SD() == FAILED)
    {
      cprintf("SDIO initialization failed.");
      while(1);
    }
  //set both xCS and xDCS high
  Deselect_xCS();
  Deselect_xDCS();

  //reset VS1011 to wake up
  Do_xRESET();

  //Wait to VS1011 request data
  Wait_DREQ_High();

  //Set VS1002 native mode : allow test
  Write_SCI(0x0, 0b0000100000100000);

  //Set clock for 14.318MHz
  Write_SCI(0x3, 0b1000000000000000 | 7159);

  //Set sample rate for 44.1khz
  Write_SCI(0x5, 0xAC45);

  //wait clock to be doubled
  delay_ms(10);

  //After setting clock rate, we can set SPI speed faster.
  Set_SPI_PSC_16();

  //Wait to VS1011 request data
  Wait_DREQ_High();

  //mount FAT file system
  fsresult = f_mount(0, &myfs);
  if (fsresult != FR_OK)
    {
      cprintf("FAT file system mounting failed.\r\n");
      while(1);
    }

  //open a mp3 file
  fsresult = f_open(&fmp3, filename, FA_READ);
  if (fsresult != FR_OK)
    {
      cprintf("File open failed. FRESULT Error code: %d.  See FATfs/ff.h for FRESULT code meaning.\r\n");
      while(1);
    }

  //read mp3 data and send VS1011
  while(1)
    {
      fsresult = f_read(&fmp3, data_buffer, 32, &data_read);
      if (fsresult != FR_OK)
        {
          cprintf("File read failed.\r\n");
          while(1);
        }

      //Wait to VS1011 request data
      Wait_DREQ_High();

      Write_SDI_DMA();

      if (data_read<32)
        {
          break;
        }
      if(RX_BUFFER_IS_NOT_EMPTY)
        {
          break;
         }
    }
  //close a mp3 file
  fsresult = f_close(&fmp3);
  if (fsresult != FR_OK)
    {
      cprintf("File close failed.\r\n");
      while(1);
    }
  else
    {
      cprintf("Play end or stopped.\r\n");
    }
  ;
}

/**
  * @brief  Configures SDIO IRQ channel.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  NVIC_InitStructure.NVIC_IRQChannel = SDIO_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC | GPIOY_1_RCC | GPIOY_2_RCC , ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIOY_0: xDCS : output push-pull */
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);

  /* Configure GPIOY_1: DREQ : input floating */
  GPIO_InitStructure.GPIO_Pin = GPIOY_1_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(GPIOY_1_PORT, &GPIO_InitStructure);

  /* Configure GPIOY_2: xRESET : output push-pull */
  GPIO_InitStructure.GPIO_Pin = GPIOY_2_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOY_2_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the SPI
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

  /* Configure I/O for xCS */
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
  SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_128;
  SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;
  SPI_InitStructure.SPI_CRCPolynomial = 7;
  SPI_Init(SPI1, &SPI_InitStructure);

  /* Enable SPI1  */
  SPI_Cmd(SPI1, ENABLE);
}

/**
  * @brief  Configure the DMA
  * @param  None
  * @retval : None
  */
void DMA_Configuration(void)
{
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);

  DMA_InitTypeDef  DMA_InitStructure;

  DMA_DeInit(SPI1_Tx_DMA_Channel);
  DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)SPI1_DR_Base;
  DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)data_buffer;
  DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralDST;
  DMA_InitStructure.DMA_BufferSize = 32;
  DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
  DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
  DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
  DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
  DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
  DMA_InitStructure.DMA_Priority = DMA_Priority_VeryHigh;
  DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
  DMA_Init(SPI1_Tx_DMA_Channel, &DMA_InitStructure);

  SPI_I2S_DMACmd(SPI1, SPI_I2S_DMAReq_Tx, ENABLE);
}

/**
  * @brief  select xCS
  * @param   None
  * @retval  None
  */
void Select_xCS(void)
{
  /* Select the FLASH: Chip Select low */
  GPIO_ResetBits(SPI1_PORT, SPI1_NSS_PIN);
}

/**
  * @brief  deselect xCS
  * @param   None
  * @retval  None
  */
void Deselect_xCS(void)
{
  /* Deselect the FLASH: Chip Select high */
  GPIO_SetBits(SPI1_PORT, SPI1_NSS_PIN);
}

/**
  * @brief  select xDCS
  * @param   None
  * @retval  None
  */

void Select_xDCS(void)
{
  /* Select the FLASH: Chip Select low */
  GPIO_ResetBits(GPIOY_0_PORT, GPIOY_0_PIN);
}

/**
  * @brief  select xDCS
  * @param   None
  * @retval  None
  */
void Deselect_xDCS(void)
{
  /* Deselect the FLASH: Chip Select high */
  GPIO_SetBits(GPIOY_0_PORT, GPIOY_0_PIN);
}

/**
  * @brief   Do reset
  * @param   None
  * @retval  None
  */
void Do_xRESET(void)
{
  GPIO_ResetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  delay_ms(10);
  GPIO_SetBits(GPIOY_2_PORT, GPIOY_2_PIN);
  //wait until VS1011 to be stable.
  //without this wait, VS1011 is locked after MCU nreset
  delay_ms(10);
}

/**
  * @brief   Wait until DREQ to be high
  * @param   None
  * @retval  None
  */
void Wait_DREQ_High(void)
{
  while(GPIO_ReadInputDataBit(GPIOY_1_PORT, GPIOY_1_PIN) == RESET){};
}

/**
  * @brief   Read SCI
  * @param   address:register number
  * @retval  data read from VS1011
  */
uint16_t Read_SCI(uint8_t address)
{
  Select_xCS();

  uint16_t data_read;

  //Send Write instruction
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
    /* Send SPI1 : read instruction*/
  SPI_I2S_SendData(SPI1, 0b00000011);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  ReceivedData = SPI_I2S_ReceiveData(SPI1);

  //Send Register address
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send SPI1 VS1001 register address*/
  SPI_I2S_SendData(SPI1, address);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  ReceivedData = SPI_I2S_ReceiveData(SPI1);

  //Read data from VS1001 : upper 8 bit
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send Null data*/
  SPI_I2S_SendData(SPI1, 0x0);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  data_read = SPI_I2S_ReceiveData(SPI1) << 8;

  //Read data from VS1001 : lower 8 bit
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send Null data*/
  SPI_I2S_SendData(SPI1, 0x0);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  data_read |= (uint8_t)SPI_I2S_ReceiveData(SPI1);

  Deselect_xCS();

  return data_read;
}

/**
  * @brief   Write SCI
  * @param   address:register number
  * @param   data:data to be written
  * @retval  none
  */
void Write_SCI(uint8_t address, uint16_t data)
{
  Select_xCS();

  //Send Write instruction
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
    /* Send SPI1 : write instruction*/
  SPI_I2S_SendData(SPI1, 0b00000010);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  ReceivedData = SPI_I2S_ReceiveData(SPI1);

  //Send Register address
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send SPI1 VS1001 register address*/
  SPI_I2S_SendData(SPI1, address);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  ReceivedData = SPI_I2S_ReceiveData(SPI1);

  //Write data to VS1001 : upper 8 bit
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send data to be written*/
  SPI_I2S_SendData(SPI1, (uint8_t)(data >> 8));
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  ReceivedData = SPI_I2S_ReceiveData(SPI1);

  //Write data to VS1001 : lower 8 bit
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 Tx buffer empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_TXE) == RESET);
  /* Send data to be written*/
  SPI_I2S_SendData(SPI1, (uint8_t)data);
  // Read and Clear SPI1 Buffer
  /* Wait for SPI1 RX buffer not-empty */
  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_RXNE) == RESET);
  /* Receive SPI1 data*/
  ReceivedData = SPI_I2S_ReceiveData(SPI1);

  Deselect_xCS();

}

/**
  * @brief   Write SDI using DMA-SPI
  * @param   data:data to be written
  * @retval  none
  */
void Write_SDI_DMA(void)
{
  DMA_Cmd(SPI1_Tx_DMA_Channel, DISABLE);

  SPI1_Tx_DMA_Channel->CNDTR = 32;

  Select_xDCS();

  DMA_Cmd(SPI1_Tx_DMA_Channel, ENABLE);

  while(!DMA_GetFlagStatus(SPI1_Tx_DMA_FLAG)){}

  DMA_ClearFlag(SPI1_Tx_DMA_FLAG);

  while (SPI_I2S_GetFlagStatus(SPI1, SPI_I2S_FLAG_BSY) != RESET){}

  Deselect_xDCS();
}

/**
  * @brief   Change SPI speed to 4.5MHz
  * @param   data:data to be written
  * @retval  none
  */
void Set_SPI_PSC_16(void)
{
  SPI_InitTypeDef  SPI_InitStructure;

  SPI_Cmd(SPI1, DISABLE);

  /* SPI1 configuration */
  SPI_InitStructure.SPI_Direction = SPI_Direction_2Lines_FullDuplex;
  SPI_InitStructure.SPI_Mode = SPI_Mode_Master;
  SPI_InitStructure.SPI_DataSize = SPI_DataSize_8b;
  SPI_InitStructure.SPI_CPOL = SPI_CPOL_Low;
  SPI_InitStructure.SPI_CPHA = SPI_CPHA_1Edge;
  SPI_InitStructure.SPI_NSS = SPI_NSS_Soft;
  SPI_InitStructure.SPI_BaudRatePrescaler = SPI_BaudRatePrescaler_16;
  SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;
  SPI_InitStructure.SPI_CRCPolynomial = 7;
  SPI_Init(SPI1, &SPI_InitStructure);

  /* Enable SPI1  */
  SPI_Cmd(SPI1, ENABLE);
}

/**
  * @brief   Initialize SDIO
  * @param   none
  * @retval  returns 1 if failed. 0 is ok.
  */
TestStatus Initialize_SDIO_SD(void)
{
  SD_CardInfo SDCardInfo;
  SD_Error Status = SD_OK;

  rtc_init();

  /*-------------------------- SD Init ----------------------------- */
  Status = SD_Init();

  if (Status == SD_OK)
  {
   /*----------------- Read CSD/CID MSD registers ------------------*/
    Status = SD_GetCardInfo(&SDCardInfo);
  }
  else
  {
    return FAILED;
  }

  if (Status == SD_OK)
  {
    /*----------------- Select Card --------------------------------*/
    Status = SD_SelectDeselect((uint32_t) (SDCardInfo.RCA << 16));
  }
  else
  {
    return FAILED;
  }

  if (Status == SD_OK)
  {
   /*----------------- Enable Wide Bus Operation --------------------------------*/
    Status = SD_EnableWideBusOperation(SDIO_BusWide_4b);
  }
  else
  {
    return FAILED;
  }

  if (Status == SD_OK)
    {
    }
  else
  {
    return FAILED;
  }
  return PASSED;
}


