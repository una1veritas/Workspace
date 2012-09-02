/**
  ******************************************************************************
  * @file    rtc_clock_serial/main.c
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
#define BKP_MAGIC_WORD         0x1234
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
__IO uint32_t TimeDisplay = 0;
const int8_t Welcome_Message1[] =
  "\r\nHellow Cortex-M3/STM32 World!\r\n"
  "Expand your creativity and enjoy making.\r\n\r\n";
const int8_t Welcome_Message2[] =
  "Input what time now is. I count time using RTC and tell you it every second.\r\n";

/* Private function prototypes -----------------------------------------------*/
void GPIO_Configuration(void);
void RTC_Configuration(void);
void NVIC_Configuration(void);
uint32_t Time_Regulate(void);
void Time_Adjust(void);
void Time_Display(uint32_t TimeVar);
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Main program.
  * @param  None
  * @retval None
  */
int main(void)
{
  // Configure board specific setting
  BoardInit();
  // Setting up COM port for Print function
  COM_Configuration();
  // Configure GPIO
  GPIO_Configuration();

  //Send welcome messages
  cprintf(Welcome_Message1);

  /* NVIC configuration */
  NVIC_Configuration();

  if (BKP_ReadBackupRegister(BKP_DR1) != BKP_MAGIC_WORD)
    {
      /* Backup data register value is not correct or not yet programmed (when
         the first time the program is executed) */

      cprintf("RTC not yet configured. Configuring.....\r\n");

      /* RTC Configuration */
      RTC_Configuration();

      cprintf("RTC configured.\r\n");

      //Send welcome messages
      cprintf(Welcome_Message2);

      /* Adjust time by values entered by the user on the hyperterminal */
      Time_Adjust();

      BKP_WriteBackupRegister(BKP_DR1, BKP_MAGIC_WORD);
    }
  else
    {
      /* Check if the Power On Reset flag is set */
      if (RCC_GetFlagStatus(RCC_FLAG_PORRST) != RESET)
        {
          cprintf("Power On Reset occurred. \r\n");
        }
      /* Check if the Pin Reset flag is set */
      else if (RCC_GetFlagStatus(RCC_FLAG_PINRST) != RESET)
        {
          cprintf("External Reset occurred. \r\n");
        }
      cprintf("No need to configure RTC. \r\n");
      /* Wait for RTC registers synchronization */
      RTC_WaitForSynchro();

      /* Enable the RTC Second */
      RTC_ITConfig(RTC_IT_SEC, ENABLE);
      /* Wait until last write operation on RTC registers has finished */
      RTC_WaitForLastTask();
    }

  /* Clear reset flags */
  RCC_ClearFlag();

  /* Infinite loop */
  while (1)
    {
      /* If 1s has passed */
      if (TimeDisplay == 1)
        {
          /* Display current time */
          cprintf("Time now is ");
          Time_Display(RTC_GetCounter());
          cprintf("  RTC counter is %u\r\n", RTC_GetCounter());
          TimeDisplay = 0;
        }
    }
}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOX_RCC | OB_LED_GPIO_RCC, ENABLE);

  GPIO_InitTypeDef GPIO_InitStructure;

  /* Configure GPIOX_0: output push-pull */
  GPIO_InitStructure.GPIO_Pin = GPIOX_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(GPIOX_PORT, &GPIO_InitStructure);

  /* Configure OB_LED: output push-pull */
  GPIO_InitStructure.GPIO_Pin = OB_LED_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_Init(OB_LED_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configures the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable the RTC Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = RTC_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
  * @brief  Configures the RTC.
  * @param  None
  * @retval None
  */
void RTC_Configuration(void)
{
  /* Enable PWR and BKP clocks */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_PWR | RCC_APB1Periph_BKP, ENABLE);

  /* Allow access to BKP Domain */
  PWR_BackupAccessCmd(ENABLE);

  /* Reset Backup Domain */
  BKP_DeInit();

  /* Enable LSE */
  RCC_LSEConfig(RCC_LSE_ON);
  /* Wait till LSE is ready */
  while (RCC_GetFlagStatus(RCC_FLAG_LSERDY) == RESET)
  {}

  /* Select LSE as RTC Clock Source */
  RCC_RTCCLKConfig(RCC_RTCCLKSource_LSE);

  /* Enable RTC Clock */
  RCC_RTCCLKCmd(ENABLE);

  /* Wait for RTC registers synchronization */
  RTC_WaitForSynchro();

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();

  /* Enable the RTC Second */
  RTC_ITConfig(RTC_IT_SEC, ENABLE);

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();

  /* Set RTC prescaler: set RTC period to 1sec */
  RTC_SetPrescaler(32767); /* RTC period = RTCCLK/RTC_PR = (32.768 KHz)/(32767+1) */

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();
}

/**
  * @brief  Returns the time entered by user, using Hyperterminal.
  * @param  None
  * @retval Current time RTC counter value
  */
uint32_t Time_Regulate(void)
{
  uint32_t Tmp_HH = 0xFF, Tmp_MM = 0xFF, Tmp_SS = 0xFF;

  cprintf("==Input Time==\r\n");
  cprintf("Please Set Hours\r\n");

  while (Tmp_HH == 0xFF)
    {
      Tmp_HH = COM_Num_Scanf();
      if  (Tmp_HH > 23)
        {
          cprintf("\r\nEnter valid Hour\r\n");
          Tmp_HH = 0xFF;
          continue;
        }
    }

  cprintf("\r\n");
  cprintf("Please Set Minutes\r\n");

  while (Tmp_MM == 0xFF)
    {
      Tmp_MM = COM_Num_Scanf();
      if  (Tmp_MM > 59)
        {
          cprintf("\r\nEnter valid Minutes\r\n");
          Tmp_MM = 0xFF;
          continue;
        }
    }


  cprintf("\r\n");
  cprintf("Please Set Seconds\r\n");
  while (Tmp_SS == 0xFF)
    {
      Tmp_SS = COM_Num_Scanf();
      if  (Tmp_SS > 59)
        {
          cprintf("\r\nEnter valid Second\r\n");
          Tmp_SS = 0xFF;
          continue;
        }
    }

  cprintf("\r\n");

  /* Return the value to store in RTC counter register */
  return((Tmp_HH*3600 + Tmp_MM*60 + Tmp_SS));
}

/**
  * @brief  Adjusts time.
  * @param  None
  * @retval None
  */
void Time_Adjust(void)
{
  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();
  /* Change the current time */
  RTC_SetCounter(Time_Regulate());
  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForLastTask();
}

/**
  * @brief  Displays the current time.
  * @param  TimeVar: RTC counter value.
  * @retval None
  */
void Time_Display(uint32_t TimeVar)
{
  uint32_t THH = 0, TMM = 0, TSS = 0;

  /* Compute  hours */
  THH = TimeVar / 3600;
  /* Compute minutes */
  TMM = (TimeVar % 3600) / 60;
  /* Compute seconds */
  TSS = (TimeVar % 3600) % 60;

  cprintf("%u:%u:%u", THH , TMM, TSS);
}
