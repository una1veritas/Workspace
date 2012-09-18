/**
  ******************************************************************************
  * @file    flash_memorize_message/main.c
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
#include "scanf.h"
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
#define MAGIC_WORD     0xabcdefab
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";
uint32_t Address ;
volatile FLASH_Status FLASHStatus;
volatile TestStatus MemoryProgramStatus;
int8_t  RxBuffer[FLASH_PAGE_SIZE];
int16_t i;
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

  if (*(uint32_t*)(FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE)) == MAGIC_WORD)
    {
      cprintf("Message found. The message stored in flash is \r\n");
      cprintf((int8_t*)(FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE)) + 4);
      cprintf("\r\n");
    }
  else
    {
      cprintf("No message found. It looks it is the first time to run this program. \r\n");
    }

  //Get Message to write in Backup Register
  i = 0;
  cprintf(
      "Type what you want to memorize in Backup registers. \r\n"
      "Maximum number to type in is :");
  cprintf("%u\r\n",FLASH_PAGE_SIZE - 2);
  COM_Char_Scanf(RxBuffer, FLASH_PAGE_SIZE - 2);
  cprintf("\r\nWhat you type is \r\n");
  cprintf(RxBuffer);
  cprintf("\r\n");

  FLASHStatus = FLASH_COMPLETE;
  MemoryProgramStatus = PASSED;

  cprintf("Unlocking Flash.\r\n");
  /* Unlock the Flash Program Erase controller */
  FLASH_Unlock();

  cprintf("Clearing pending flags.\r\n");
  /* Clear All pending flags */
  FLASH_ClearFlag(FLASH_FLAG_BSY | FLASH_FLAG_EOP | FLASH_FLAG_PGERR | FLASH_FLAG_WRPRTERR);

  cprintf("Erasing Flash Pages.\r\n");
  /* Erase the FLASH pages */
  FLASHStatus = FLASH_ErasePage(FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE));

  cprintf("Writing Magic word to Flash.\r\n");
  FLASH_ProgramWord(FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE), MAGIC_WORD);
  Address = 0x0;
  cprintf("Writing your message to Flash.\r\n");
  while((Address < FLASH_PAGE_SIZE) && (FLASHStatus == FLASH_COMPLETE))
    {
      FLASHStatus =
        FLASH_ProgramWord(FlashBaseAddr + (StartPage * FLASH_PAGE_SIZE) + Address + 4,
            (RxBuffer[Address + 0] <<  0 ) +
            (RxBuffer[Address + 1] <<  8 ) +
            (RxBuffer[Address + 2] << 16 ) +
            (RxBuffer[Address + 3] << 24 ) );
      Address = Address + 4;
      cprintf(".");
    }
  cprintf("\r\nWriting finished.\r\n");

  while(1){}

}
