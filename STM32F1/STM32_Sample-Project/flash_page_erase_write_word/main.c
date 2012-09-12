/**
  ******************************************************************************
  * @file    flash_page_erase_write_word/main.c
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
/* Private typedef -----------------------------------------------------------*/
typedef enum {FAILED = 0, PASSED = !FAILED} TestStatus;
/* Private define ------------------------------------------------------------*/
#if defined STM32F10X_MD
  #define FLASH_PAGE_SIZE    ((uint16_t)0x400)
#elif defined STM32F10X_HD
  #define FLASH_PAGE_SIZE    ((uint16_t)0x800)
#endif
#define FlashBaseAddr  ((uint32_t)0x08000000)
#define StartPage      10
#define PagesToWrite    3
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";

uint32_t EraseCounter = 0x00, Address = 0x00 , EndAddress;
uint32_t Data;
__IO uint32_t NbrOfPage = 0x00;
volatile FLASH_Status FLASHStatus;
volatile TestStatus MemoryProgramStatus;

/* Private function prototypes -----------------------------------------------*/
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
  //Print welcome message
  cprintf(Welcome_Message);

  FLASHStatus = FLASH_COMPLETE;
  MemoryProgramStatus = PASSED;
  Data = 0x01234567;

  cprintf("Unlocking Flash.\r\n");
  /* Unlock the Flash Program Erase controller */
  FLASH_Unlock();

  /* Define the number of page to be erased */
  NbrOfPage = PagesToWrite;

  cprintf("Clearing pending flags.\r\n");
  /* Clear All pending flags */
  FLASH_ClearFlag(FLASH_FLAG_BSY | FLASH_FLAG_EOP | FLASH_FLAG_PGERR | FLASH_FLAG_WRPRTERR);

  cprintf("Erasing Flash Pages.\r\n");
  /* Erase the FLASH pages */
  for(EraseCounter = 0; (EraseCounter < PagesToWrite) && (FLASHStatus == FLASH_COMPLETE); EraseCounter++)
    {
      FLASHStatus = FLASH_ErasePage(FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE) + (FLASH_PAGE_SIZE * EraseCounter));
      cprintf(".");
    }
  cprintf("\r\n");

  /*  FLASH Word program at addresses defined by StartAddr and EndAddr*/
  Address = FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE);
  EndAddress = FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE) + (FLASH_PAGE_SIZE * PagesToWrite);

  cprintf("Writing 0x01234567 from StartPage for PagesToWrite.\r\n");
  while((Address < EndAddress) && (FLASHStatus == FLASH_COMPLETE))
    {
      FLASHStatus = FLASH_ProgramWord(Address, Data);
      Address = Address + 4;
      cprintf(".");
    }
  cprintf("\r\nWriting finished.\r\n");

  /* Check the correctness of written data */
  Address = FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE);

  cprintf("Checking the correctness from StartAddr to EndAddr.\r\n");
  while((Address < EndAddress) && (MemoryProgramStatus != FAILED))
    {
      if((*(__IO uint32_t*) Address) != Data)
      {
        MemoryProgramStatus = FAILED;
        cprintf("Written memory does not match expected data.\r\n");
        while(1){}
      }
      Address += 4;
      cprintf(".");
    }
  cprintf("\r\nAll check passed.\r\n");
  while(1){}

}
