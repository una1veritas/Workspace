/**
  ******************************************************************************
  * @file    lib/platform_config.h
  * @author  Yasuo Kawachi
  * @version  V1.0.0
  * @date  04/15/2009
  * @brief  Evaluation board specific configuration file.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __PLATFORM_CONFIG_H
#define __PLATFORM_CONFIG_H
/* Macros ------------------------------------------------------------------*/

/* Uncomment the mecro below if you build binary for DFU*/
//#define USE_DFU

/* Uncomment the mecro below if you want to stop just after COM port configuration */
//#define STOP_AT_STARTUP

/* Uncomment the line corresponding to the STMicroelectronics evaluation board
   used to run the example */
#if !defined (USE_STM32_P103) && \
    !defined (USE_STM32_H103) && \
    !defined (USE_CQ_STARM) && \
    !defined (USE_CQ_ST103Z) && \
    !defined (USE_STM3210E_EVAL) && \
    !defined (USE_STBEE) && \
    !defined (USE_STBEE_MINI)

//#define USE_STM32_P103
//#define USE_STM32_H103
//#define USE_CQ_STARM
//#define USE_CQ_ST103Z
//#define USE_STM3210E_EVAL
#define USE_STBEE
//#define USE_STBEE_MINI

#endif

/* Uncomment the mecro below interface you to use as cprintf() macro*/
#if !defined (USE_USART1) && \
    !defined (USE_USART2) && \
    !defined (USE_USART3) && \
    !defined (USE_VCP)

#define USE_USART1
//#define USE_USART2
//#define USE_USART3
//#define USE_VCP

#endif

//Define cprintf of actual interface function
#ifdef USE_USART1
#define  USARTX_Configuration()          USART_Configuration(USART1, 115200)
#define  RECEIVE_DATA                    USART_ReceiveData(USART1)
#define  cputchar(arg)                   USART_PrintChar(USART1, arg)
#define  cprintf(...)                    USART_PrintFormatted(USART1, __VA_ARGS__)
#define  RX_BUFFER_IS_NOT_EMPTY          USART_GetFlagStatus(USART1, USART_FLAG_RXNE) != RESET
#define  RX_BUFFER_IS_EMPTY              USART_GetFlagStatus(USART1, USART_FLAG_RXNE) == RESET
#define  TX_BUFFER_IS_NOT_EMPTY          USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET
#elif defined USE_USART2
#define  USARTX_Configuration()          USART_Configuration(USART2, 115200)
#define  RECEIVE_DATA                    USART_ReceiveData(USART2)
#define  cputchar(arg)                   USART_PrintChar(USART2, arg)
#define  cprintf(...)                    USART_PrintFormatted(USART2, __VA_ARGS__)
#define  RX_BUFFER_IS_NOT_EMPTY          USART_GetFlagStatus(USART2, USART_FLAG_RXNE) != RESET
#define  RX_BUFFER_IS_EMPTY              USART_GetFlagStatus(USART2, USART_FLAG_RXNE) == RESET
#define  TX_BUFFER_IS_NOT_EMPTY          USART_GetFlagStatus(USART2, USART_FLAG_TXE) == RESET
#elif defined USE_USART3
#define  USARTX_Configuration()          USART_Configuration(USART3, 115200)
#define  RECEIVE_DATA                    USART_ReceiveData(USART3)
#define  cputchar(arg)                   USART_PrintChar(USART3, arg)
#define  cprintf(...)                    USART_PrintFormatted(USART3, __VA_ARGS__)
#define  RX_BUFFER_IS_NOT_EMPTY          USART_GetFlagStatus(USART3, USART_FLAG_RXNE) != RESET
#define  RX_BUFFER_IS_EMPTY              USART_GetFlagStatus(USART3, USART_FLAG_RXNE) == RESET
#define  TX_BUFFER_IS_NOT_EMPTY          USART_GetFlagStatus(USART3, USART_FLAG_TXE) == RESET
#elif defined USE_VCP
#define  USARTX_Configuration()          {}
#define  RECEIVE_DATA                    VCP_ReceiveData()
#define  cputchar(arg)                   VCP_PrintChar(arg)
#define  cprintf(...)                    VCP_PrintFormatted(__VA_ARGS__)
#define  RX_BUFFER_IS_NOT_EMPTY          (count_out != 0) && (bDeviceState == CONFIGURED)
#define  RX_BUFFER_IS_EMPTY              (count_out == 0) && (bDeviceState == CONFIGURED)
#define  TX_BUFFER_IS_NOT_EMPTY          RESET
#endif

//Define the location of vector table
#ifdef USE_DFU
#define  VECTOR_OFFSET                  0x3000
#else
#define  VECTOR_OFFSET                  0x0
#endif

/* Define the STM32F10x hardware depending on the used evaluation board */
#ifdef USE_STM32_P103
#include "STM32_P103.h"
#elif defined USE_STM32_H103
#include "STM32_H103.h"
#elif defined USE_CQ_STARM
#include "CQ_STARM.h"
#elif defined USE_CQ_ST103Z
#include "CQ_ST103Z.h"
#elif defined USE_STM3210E_EVAL
#include "STM3210E_EVAL.h"
#elif defined USE_STBEE
#include "STBee.h"
#elif defined USE_STBEE_MINI
#include "STBee_Mini.h"
#endif

/* Uncomment the line corresponding to the STMicroelectronics evaluation board
   used to run the example */
#if !defined (JTAG_SWD_Enabled) && \
    !defined (JTAG_SWD_Enabled_without_NJTRST) && \
    !defined (JTAG_Disabled_SWD_Enabled) && \
    !defined (JTAG_SWD_Disabled)

#define JTAG_SWD_Enabled
//#define JTAG_SWD_Enabled_without_NJTRST
//#define JTAG_Disabled_SWD_Enabled
//#define JTAG_SWD_Disabled

#endif
/* Private function prototypes -----------------------------------------------*/
void BoardInit(void);

#endif /* __PLATFORM_CONFIG_H */

