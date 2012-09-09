/**
  ******************************************************************************
  * @file    sdio_sd_test_terminal/main.c
  * @author  Martin Thomas, Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   SD test terminal
  ******************************************************************************
  * @copy
  *
  * This code is made by Martin Thomas. Yasuo Kawachi made small
  * modification to it.
  *
  * Copyright 2008-2009 Martin Thomas and Yasuo Kawachi All rights reserved.
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
#include "sdcard.h"
#include "ff.h"
#include "diskio.h"
#include "ff_test_term.h"
#include "term_io.h"
#include "comm.h"
#include "rtc.h"
#include "com_config.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
//USART_InitTypeDef USART_InitStructure;
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n"
  "Test SDIO interface accessing to SD card with FAT\r\n"
  "Big thanks to Martin Thomas, J. Shaler and ChaN.\r\r\n\n";
/* Private function prototypes -----------------------------------------------*/
void RCC_Configuration(void);
void NVIC_Configuration(void);
/* Private typedef -----------------------------------------------------------*/
typedef enum {FAILED = 0, PASSED = !FAILED} TestStatus;
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
SD_CardInfo SDCardInfo;
SD_Error Status = SD_OK;

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

  /* System Clocks Configuration */
  RCC_Configuration();

  // Configure NVIC
  NVIC_Configuration();

  // Setting up COM port for Print function
  COM_Configuration();

  /* Setup SysTick Timer for 1 millisec interrupts, also enables Systick and Systick-Interrupt */
  if (SysTick_Config(SystemCoreClock / 1000))
    {
      /* Capture error */
      while (1);
    }

  //Print welcome message
  cprintf(Welcome_Message);
  rtc_init();

  /*-------------------------- SD Init ----------------------------- */
  Status = SD_Init();

  if (Status == SD_OK)
  {
    cprintf("SD Card initialized ok.\r\n");
   /*----------------- Read CSD/CID MSD registers ------------------*/
    Status = SD_GetCardInfo(&SDCardInfo);
  }
  else
  {
    cprintf("SD Card did not initialize, check that a card is inserted. SD_Error code: %d.  See sdcard.h for SD_Error code meaning.\r\n");
    while(1);  //infinite loop
  }

  if (Status == SD_OK)
  {
    cprintf("SD Card information retrieved ok.\r\n");
    /*----------------- Select Card --------------------------------*/
    Status = SD_SelectDeselect((uint32_t) (SDCardInfo.RCA << 16));
  }
  else
  {
    cprintf("Could not get SD Card information. SD_Error code: %d.  See sdcard.h for SD_Error code meaning.\r\n");
    while(1);  //infinite loop
  }

  if (Status == SD_OK)
  {
    cprintf("SD Card selected ok.\r\n");
   /*----------------- Enable Wide Bus Operation --------------------------------*/
    Status = SD_EnableWideBusOperation(SDIO_BusWide_4b);
  }
  else
  {
    cprintf("SD Card selection failed. SD_Error code: %d.  See sdcard.h for SD_Error code meaning.\r\n");
    while(1);  //infinite loop
  }

  if (Status == SD_OK)
     cprintf("SD Card 4-bit Wide Bus operation successfully enabled.\r\n");
  else
  {
    cprintf("Could not enable SD Card 4-bit Wide Bus operation, will revert to 1-bit operation.\nSD_Error code: %d.  See sdcard.h for SD_Error code meaning.\r\n");
  }

  FRESULT fsresult;               //return code for file related operations
  FATFS myfs;                     //FAT file system structure, see ff.h for details
  fsresult = f_mount(0, &myfs);
  if (fsresult == FR_OK)
    cprintf("FAT file system mounted ok.\r\n");
  else
    cprintf("FAT file system mounting failed. FRESULT Error code: %d.  See FATfs/ff.h for FRESULT code meaning.\r\n");

  //Run FatFs test terminal
  ff_test_term();
}

/**
  * @brief  Start and supply clocks
  * @param  None
  * @retval : None
  */
void RCC_Configuration(void)
{
  /* DMA1 clock enable */
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
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

