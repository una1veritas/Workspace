/**
  ******************************************************************************
  * @file    pwr_standby_rtc_alarm/main.c
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
#include "delay.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
const int8_t Welcome_Message[] = "\r\nHellow Cortex-M3/STM32 World!\r\n"
                "Expand your creativity and enjoy making.\r\n\r\n"
                "I will enter into Standby Mode after 3 second.\r\n";
/* Private function prototypes -----------------------------------------------*/
void RTC_Configuration(void);
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program.
  * @param  None
  * @retval : None
  */
int main(void)
{
  int8_t i;


  // Configure board specific setting
  BoardInit();
  // Setting up COM port for Print function
  COM_Configuration();

  //Print welcome message
  cprintf(Welcome_Message);

  /* Enable PWR and BKP clock */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_PWR | RCC_APB1Periph_BKP, ENABLE);

  /* Allow access to BKP Domain */
  PWR_BackupAccessCmd(ENABLE);

  /* Configure RTC clock source and prescaler */
  RTC_Configuration();

  if (PWR_GetFlagStatus(PWR_FLAG_SB) != RESET)
    {
      PWR_ClearFlag(PWR_FLAG_SB);
      //Send recover message
      cprintf("\r\nRecovered from Standby Mode.\r\n");
    }

  for (i=3;i>0;i--)
    {
      cprintf("%u...", i);
      //wait to stabilize COM output
      delay_ms(1000);
    }

  cprintf("Entering into Standby Mode");

  delay_ms(100);

  /* Wait till RTC Second event occurs */
  RTC_ClearFlag(RTC_FLAG_SEC);
  while(RTC_GetFlagStatus(RTC_FLAG_SEC) == RESET);

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();

  /* Set the RTC Alarm after 3s */
  RTC_SetAlarm(RTC_GetCounter()+ 3);

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();

  /* Request to enter STANDBY mode (Wake Up flag is cleared in PWR_EnterSTANDBYMode function) */
  PWR_EnterSTANDBYMode();
}

/**
  * @brief  Configures RTC clock source and prescaler.
  * @param  None
  * @retval None
  */
void RTC_Configuration(void)
{
  /* Check if the StandBy flag is set */
  if(PWR_GetFlagStatus(PWR_FLAG_SB) != RESET)
    {
      /* Wait for RTC APB registers synchronisation */
      RTC_WaitForSynchro();
      /* No need to configure the RTC as the RTC configuration(clock source, enable,
         prescaler,...) is kept after wake-up from STANDBY */
    }
  else
    {
      /* RTC clock source configuration ----------------------------------------*/
      /* Reset Backup Domain */
      BKP_DeInit();

      /* Enable LSE OSC */
      RCC_LSEConfig(RCC_LSE_ON);
      /* Wait till LSE is ready */
      while(RCC_GetFlagStatus(RCC_FLAG_LSERDY) == RESET)
        {
        }

      /* Select the RTC Clock Source */
      RCC_RTCCLKConfig(RCC_RTCCLKSource_LSE);

      /* Enable the RTC Clock */
      RCC_RTCCLKCmd(ENABLE);

      /* RTC configuration -----------------------------------------------------*/
      /* Wait for RTC APB registers synchronisation */
      RTC_WaitForSynchro();

      /* Set the RTC time base to 1s */
      RTC_SetPrescaler(32767);
      /* Wait until last write operation on RTC registers has finished */
      RTC_WaitForLastTask();
    }
}


