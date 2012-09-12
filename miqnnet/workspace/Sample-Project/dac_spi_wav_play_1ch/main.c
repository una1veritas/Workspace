/**
  ******************************************************************************
  * @file    dac_spi_wav_play_1ch/main.c
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
#include "spi_sd_fat.h"
#include <string.h>
/* Private typedef -----------------------------------------------------------*/
//typedef enum {FAILED = 0, PASSED = !FAILED} TestStatus;
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";
FRESULT fsresult;
FATFS myfs;
FIL fhandle;
int8_t filename[] = "/dies_m.wav";
uint16_t data_buffer;
UINT data_read;
uint8_t ReceivedData, RxData;
uint32_t i;
__IO uint32_t Counter = 0;
uint16_t DAC_Value = 0;
typedef struct
{
  int8_t Header_RIFF[4];
  uint32_t File_Size;
  int8_t Header_WAV[4];
} WAV_Header_Type;
WAV_Header_Type WAV_Header;

typedef struct
{
  int8_t Chank_Name[4];
  uint32_t Chank_Length;
} Chank_Type;
Chank_Type Chank;

typedef struct
{
  uint16_t Format_ID;
  uint16_t Number_of_Channel;
  uint32_t Sample_Rate;
  uint32_t Data_Rate;
  uint16_t Block_Size;
  uint16_t Sample_Bits;
  uint16_t extended_data1;
  uint16_t extended_data2;
  uint16_t extended_data3;
} WAV_PCM_Format_Type;
WAV_PCM_Format_Type WAV_PCM_Format;

/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
void DAC_Configuration(void);
void TIM2_Configuration(uint32_t sample_rate);
void TIM2_IRQHandler(void);
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

  /* DAC Configuration*/
  DAC_Configuration();

  // Setting up COM port for Print function
  COM_Configuration();
  //Print welcome message
  cprintf(Welcome_Message);

  // Setting up SD card interface via SPI
  SPI_SD_Init();

  //mount FAT file system
  fsresult = f_mount(0, &myfs);
  if (fsresult != FR_OK)
    {
      cprintf("FAT file system mounting failed.\r\n");
      while(1);
    }

  //open a wav file
  fsresult = f_open(&fhandle, filename, FA_READ);
  if (fsresult != FR_OK)
    {
      cprintf("File open failed. \r\n");
      while(1);
    }

  fsresult = f_read(&fhandle, &WAV_Header, sizeof WAV_Header, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }

  if(strncmp(WAV_Header.Header_RIFF, "RIFF", 4))
    {
      cprintf("RIFF Header not found.\r\n");
      while(1);
    }
  else
    {
      cprintf("RIFF Header found.\r\n");
    }

  cprintf("File size written in RIFF header is :%ubytes. \r\n", WAV_Header.File_Size);

  if(strncmp(WAV_Header.Header_WAV, "WAVE", 4))
    {
      cprintf("RIFF format is not WAVE.\r\n");
      while(1);
    }
  else
    {
      cprintf("RIFF format is WAVE.\r\n");
    }

  fsresult = f_read(&fhandle, &Chank, sizeof Chank, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }

  if(strncmp(Chank.Chank_Name, "fmt ", 4))
    {
      cprintf("fmt chank not found.\r\n");
      while(1);
    }
  else
    {
      cprintf("fmt chank found.\r\n");
    }

  cprintf("fmt chank size is :%ubytes. \r\n", Chank.Chank_Length);

  fsresult = f_read(&fhandle, &WAV_PCM_Format, Chank.Chank_Length, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }

  cprintf("Format ID is :%u", WAV_PCM_Format.Format_ID);
  cprintf("\r\nNumber of Channel is :%u", WAV_PCM_Format.Number_of_Channel);
  cprintf("\r\nSampling rate is :%u", WAV_PCM_Format.Sample_Rate);
  cprintf("\r\nData rate is :%u", WAV_PCM_Format.Data_Rate);
  cprintf("\r\nBlock Size is :%u", WAV_PCM_Format.Block_Size);
  cprintf("\r\nSample Format is :%ubit\r\n", WAV_PCM_Format.Sample_Bits);

  while(1)
    {
      fsresult = f_read(&fhandle, &Chank, sizeof Chank, &data_read);
      if (fsresult != FR_OK)
        {
          cprintf("File read failed.\r\n");
          while(1);
        }

      if(strncmp(Chank.Chank_Name, "data", 4))
        {
          cprintf("Next chank is not data chank. Skipping this chank.\r\n");
          for(i=0;i<Chank.Chank_Length;i++ )
            {
              fsresult = f_read(&fhandle, &data_buffer, 1, &data_read);
            }
        }
      else
        {
          cprintf("data chank found.\r\n");
          break;
        }
    }
  cprintf("data chank size is :%ubytes. \r\n", Chank.Chank_Length);

  /* TIM2 Configuration*/
  TIM2_Configuration(WAV_PCM_Format.Sample_Rate);

  //read wav data
  while(1){}
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
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

  /* Enable the TIM2 Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configure the DAC Pins.
  * @param  None
  * @retval : None
  */
void DAC_Configuration(void)
{
  /* GPIOA Periph clock enable */
  RCC_APB2PeriphClockCmd(DAC_GPIO_RCC, ENABLE);
  /* DAC Periph clock enable */
  RCC_APB1PeriphClockCmd(DAC_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin =  GPIO_Pin_4;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
  GPIO_Init(GPIOA, &GPIO_InitStructure);

  DAC_InitTypeDef            DAC_InitStructure;
  /* DAC channel1 Configuration */
  DAC_InitStructure.DAC_Trigger = DAC_Trigger_None;
  DAC_InitStructure.DAC_WaveGeneration = DAC_WaveGeneration_None;
  DAC_InitStructure.DAC_OutputBuffer = DAC_OutputBuffer_Enable;
  DAC_Init(DAC_Channel_1, &DAC_InitStructure);

  /* Enable DAC Channel1: Once the DAC channel1 is enabled, PA.04 is
     automatically connected to the DAC converter. */
  DAC_Cmd(DAC_Channel_1, ENABLE);
}

/**
  * @brief  Configure TIM2
  * @param  None
  * @retval : None
  */
void TIM2_Configuration(uint32_t sample_rate)
{
  TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(TIM2_RCC , ENABLE);

  /* ---------------------------------------------------------------------------
    TIM2 Configuration: Output Compare Toggle Mode:
    TIM2CLK = 72 MHz, Prescaler = 2250, TIM2 counter clock = 32kHz
  ----------------------------------------------------------------------------*/

  /* Time base configuration */
  TIM_TimeBaseStructure.TIM_Period = (uint16_t)((SystemCoreClock / 2) * 2 / sample_rate);
  TIM_TimeBaseStructure.TIM_Prescaler = 0;
  TIM_TimeBaseStructure.TIM_ClockDivision = 0;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseStructure.TIM_RepetitionCounter = 0;
  TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);

  /* TIM IT enable */
  TIM_ITConfig(TIM2, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM2, ENABLE);

}

/**
  * @brief  This function handles TIM2 update interrupt request.
  * @param  None
  * @retval None
  */
void TIM2_IRQHandler(void)
{
  if (TIM_GetITStatus(TIM2, TIM_IT_Update) != RESET)
  {
    /* Clear TIM2 update interrupt pending bit*/
    TIM_ClearITPendingBit(TIM2, TIM_IT_Update);

    fsresult = f_read(&fhandle, &DAC_Value, 2, &data_read);
    DAC_Value = 0x8000 - DAC_Value;
    if (fsresult != FR_OK)
      {
        cprintf("File read failed.\r\n");
        while(1);
      }

    /* Set DAC Channel1 DAC_DHR12L1 register */
    DAC_SetChannel1Data(DAC_Align_12b_L, DAC_Value);

    if (data_read<2)
      {

        TIM_ITConfig(TIM2, TIM_IT_Update, DISABLE);

        TIM_Cmd(TIM2, DISABLE);

        //close a WAV file
        fsresult = f_close(&fhandle);
        if (fsresult != FR_OK)
          {
            cprintf("File close failed.\r\n");
            while(1);
          }
        else
          {
            cprintf("Play end or stopped.\r\n");
          }
        while(1){}
      }
  }
}


