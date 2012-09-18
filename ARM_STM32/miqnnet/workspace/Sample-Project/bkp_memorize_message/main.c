/**
  ******************************************************************************
  * @file    bkp_memorize_message/main.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   Main program body
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
  *
  * This software way contain the part of STMicroelectronics firmware.
  * Below notice is applied to the part, but the above BSD license is not.
  *
  * THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
  * WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE
  * TIME. AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY
  * DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING
  * FROM THE CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE
  * CODING INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
  *
  * COPYRIGHT 2009 STMicroelectronics
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x.h"
#include "platform_config.h"
#include "com_config.h"
#include "scanf.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#if defined (USE_STM32_P103) || defined (USE_STM32_H103) || (USE_CQ_STARM)
#define BKP_DR_NUMBER              10
#elif defined (USE_CQ_ST103Z) || defined (USE_STM3210E_EVAL) || defined (USE_STBEE)
#define BKP_DR_NUMBER              42
#endif
#define OPEN_SESAME                0x594B
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "Here I show you I can store data while main power is not supplied.\r\n";
uint8_t RxData;
int8_t RxBuffer[BKP_DR_NUMBER * 2 +1];

#if defined (USE_STM32_P103) || defined (USE_STM32_H103) || (USE_CQ_STARM)
uint16_t BKPDataReg[BKP_DR_NUMBER] =
  {
    BKP_DR1, BKP_DR2, BKP_DR3, BKP_DR4, BKP_DR5, BKP_DR6, BKP_DR7, BKP_DR8,
    BKP_DR9, BKP_DR10
  };
#elif defined (USE_CQ_ST103Z) || defined (USE_STM3210E_EVAL) || defined (USE_STBEE)
uint16_t BKPDataReg[BKP_DR_NUMBER] =
  {
    BKP_DR1, BKP_DR2, BKP_DR3, BKP_DR4, BKP_DR5, BKP_DR6, BKP_DR7, BKP_DR8,
    BKP_DR9, BKP_DR10, BKP_DR11, BKP_DR12, BKP_DR13, BKP_DR14, BKP_DR15, BKP_DR16,
    BKP_DR17, BKP_DR18, BKP_DR19, BKP_DR20, BKP_DR21, BKP_DR22, BKP_DR23, BKP_DR24,
    BKP_DR25, BKP_DR26, BKP_DR27, BKP_DR28, BKP_DR29, BKP_DR30, BKP_DR31, BKP_DR32,
    BKP_DR33, BKP_DR34, BKP_DR35, BKP_DR36, BKP_DR37, BKP_DR38, BKP_DR39, BKP_DR40,
    BKP_DR41, BKP_DR42
  };
#endif

/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  uint8_t i;


  // Configure board specific setting
  BoardInit();
  // Setting up COM port for Print function
  COM_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message);

  /* Enable PWR and BKP clock */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_PWR | RCC_APB1Periph_BKP, ENABLE);
  /* Enable write access to Backup domain */
  PWR_BackupAccessCmd(ENABLE);
  /* Clear Tamper pin Event(TE) pending flag */
  BKP_ClearFlag();

  /* Check if the Power On Reset flag is set */
  if(RCC_GetFlagStatus(RCC_FLAG_PORRST) != RESET)
  {

    cprintf("Power-on Reset occurred.\r\n");
    /* Clear reset flags */
    RCC_ClearFlag();

    /* Check if Backup data registers are programmed */
    if( BKP_ReadBackupRegister(BKPDataReg[0]) == OPEN_SESAME )
      {
        cprintf("Found Backup register is programmed.\r\n");
        cprintf("Memorized message is :\r\n");
        for (i=1;i < BKP_DR_NUMBER ; i++)
          {
            cputchar((int8_t)(BKP_ReadBackupRegister(BKPDataReg[i]) >> 8));
            cputchar((int8_t)(BKP_ReadBackupRegister(BKPDataReg[i]) & 0x00FF));
          }
        cprintf("\r\r\n\n");
      }
    else
      {
        cprintf("It seems Backup register is not programmed.\r\r\n\n");
      }

  }

  //Get Message to write in Backup Register
  i = 0;
  cprintf(
      "Type what you want to memorize in Backup registers. \r\n"
      "Maximum number to type in is :");
  cprintf("%u\r\n",(BKP_DR_NUMBER-1)*2);
  COM_Char_Scanf(RxBuffer, (BKP_DR_NUMBER-1)*2);
  cprintf("\r\nWhat you type is \r\n %s\r\n", RxBuffer);

  //Write Message in Backup register
  i = 0;
  cprintf("Writing in Backup registers. \r\n");
  while(RxBuffer[i*2] != '\0'  && RxBuffer[(i*2)+1] != '\0')
    {
      BKP_WriteBackupRegister(BKPDataReg[i+1], (((uint16_t)RxBuffer[i*2])<< 8) | ((uint16_t) RxBuffer[(i*2)+1]) );

      cprintf("...");
      cputchar(RxBuffer[i*2]);
      cprintf("...");
      cputchar(RxBuffer[(i*2)+1]);

      i++;
    }
  BKP_WriteBackupRegister(BKPDataReg[0], OPEN_SESAME);
  cprintf("...and OPEN SESAME !");
  cprintf(
      "\r\nWrite Complete!\r\n"
      "Reset main power and confirm your message is memorized.\r\n"
      "Don't forget to supply VBAT.\r\n");

  while (1){}
}
