/**
  ******************************************************************************
  * @file    sdio_sd_modify_text/main.c
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
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Correct a phrase in hello.txt.\r\n\r\n";
FRESULT fsresult;
FATFS myfs;
FIL ftxt;
int8_t filename[] = "/hello.txt";
int8_t data_to_be_written[1];
uint8_t data_buffer[512];
UINT data_read, data_written;
uint8_t i;
/* Private function prototypes -----------------------------------------------*/
void NVIC_Configuration(void);
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

  //mount FAT file system
  fsresult = f_mount(0, &myfs);
  if (fsresult != FR_OK)
    {
      cprintf("FAT file system mounting failed.\r\n");
      while(1);
    }

  //open a text file
  fsresult = f_open(&ftxt, filename, FA_READ | FA_WRITE);
  if (fsresult != FR_OK)
    {
      cprintf("File open failed.\r\n");
      while(1);
    }

  fsresult = f_read(&ftxt, data_buffer, 512, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }

  cprintf("Phrase in hello.txt is:\r\n");

  for(i=0;i<data_read;i++)
    {
      cputchar(data_buffer[i]);
    }

  fsresult = f_lseek(&ftxt, 7);
  if (fsresult != FR_OK)
    {
      cprintf("File seek failed.\r\n");
      while(1);
    }

  cprintf("Pointer is moved to 7.\r\n");

  data_to_be_written[0] = 'D';

  fsresult = f_write(&ftxt, data_to_be_written, 1, &data_written);
  if (fsresult != FR_OK)
    {
      cprintf("File Write failed.\r\n");
      while(1);
    }

  cprintf("D is written.\r\n");

  fsresult = f_lseek(&ftxt, 9);
  if (fsresult != FR_OK)
    {
      cprintf("File seek failed.\r\n");
      while(1);
    }

  cprintf("Pointer is moved to 9.\r\n");

  data_to_be_written[0] = 'C';

  fsresult = f_write(&ftxt, data_to_be_written, 1, &data_written);
  if (fsresult != FR_OK)
    {
      cprintf("File Write failed.\r\n");
      while(1);
    }

  cprintf("C is written.\r\n");

  fsresult = f_lseek(&ftxt, 0);
  if (fsresult != FR_OK)
    {
      cprintf("File seek failed.\r\n");
      while(1);
    }

  cprintf("Pointer is moved to the first address of the file.\r\n");

  fsresult = f_sync(&ftxt);
  if (fsresult != FR_OK)
    {
      cprintf("File seek failed.\r\n");
      while(1);
    }

  cprintf("File is synchronized.\r\n");

  cprintf("Phrase in hello.txt is:\r\n");

  fsresult = f_read(&ftxt, data_buffer, 512, &data_read);
  if (fsresult != FR_OK)
    {
      cprintf("File read failed.\r\n");
      while(1);
    }

  for(i=0;i<data_read;i++)
    {
      cputchar(data_buffer[i]);
    }

  //close a mp3 file
  fsresult = f_close(&ftxt);
  if (fsresult != FR_OK)
    {
      cprintf("File close failed.\r\n");
      while(1);
    }

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
