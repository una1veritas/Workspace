/**
  ******************************************************************************
  * @file    USART/src/usart_config.c
  * @author  Yasuo Kawachi
  * @version  V1.0.0
  * @date  04/15/2009
  * @brief  USART utility functions
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
#include "usart_config.h"
#include "remap.h"
#include "toascii.h"
#include <stdarg.h>
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Extern variables ----------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

uint32_t aaa = __MPU_PRESENT  ;

/**
  * @brief  Configure the USART
  * @param  usartused : specify USART used
  * @param  baudrate : baudrate to be used
  * @retval None
  */
void USART_Configuration(USART_TypeDef* usartused, uint32_t baudrate)
{
  GPIO_InitTypeDef GPIO_InitStructure;

  if (usartused == USART1)
    {
      RCC_APB2PeriphClockCmd(USART1_GPIO_RCC | USART1_RCC, ENABLE);

      /* Configure USART1 Tx as alternate function push-pull */
      GPIO_InitStructure.GPIO_Pin = USART1_TX_PIN;
      GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
      GPIO_Init(USART1_PORT, &GPIO_InitStructure);

      /* Configure USART1 Rx as input floating */
      GPIO_InitStructure.GPIO_Pin = USART1_RX_PIN;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
      GPIO_Init(USART1_PORT, &GPIO_InitStructure);

      #if defined (REMAP_USART1)
      Remap_USART1_Configuration();
      #endif
    }
  else if (usartused == USART2)
    {
      RCC_APB2PeriphClockCmd(USART2_GPIO_RCC, ENABLE);
      RCC_APB1PeriphClockCmd(USART2_RCC , ENABLE);

      /* Configure USART2 Tx as alternate function push-pull */
      GPIO_InitStructure.GPIO_Pin = USART2_TX_PIN;
      GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
      GPIO_Init(USART2_PORT, &GPIO_InitStructure);

      /* Configure USART2 Rx as input floating */
      GPIO_InitStructure.GPIO_Pin = USART2_RX_PIN;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
      GPIO_Init(USART2_PORT, &GPIO_InitStructure);

      #if defined (REMAP_USART2)
      Remap_USART2_Configuration();
      #endif
    }
  else if (usartused == USART3)
    {
      RCC_APB2PeriphClockCmd(USART3_GPIO_RCC, ENABLE);
      RCC_APB1PeriphClockCmd(USART3_RCC , ENABLE);

      /* Configure USART3 Tx as alternate function push-pull */
      GPIO_InitStructure.GPIO_Pin = USART3_TX_PIN;
      GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
      GPIO_Init(USART3_PORT, &GPIO_InitStructure);

      /* Configure USART3 Rx as input floating */
      GPIO_InitStructure.GPIO_Pin = USART3_RX_PIN;
      GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
      GPIO_Init(USART3_PORT, &GPIO_InitStructure);

      #if defined (PARTIAL_REMAP_USART3)
      PartialRemap_USART3_Configuration();
      #elif defined (FULL_REMAP_USART3)
      FullRemap_USART3_Configuration();
      #endif

    }

  /* USART configuration ------------------------------------------------------*/
  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = baudrate;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
  USART_Init(usartused, &USART_InitStructure);
  /* Enable the USART */
  USART_Cmd(usartused, ENABLE);

}

/**
  * @brief  send a byte data
  * @param  usartused : specify USART used
  * @param  charvalue : byte data to be used
  * @retval None
  */
void USART_PrintChar(USART_TypeDef* usartused, int8_t charvalue)
{
  while(USART_GetFlagStatus(usartused, USART_FLAG_TXE) == RESET);
  USART_SendData(usartused, charvalue);
}

/**
  * @brief  send string
  * @param  usartused : specify USART used
  * @param  string : pointer to stirng
  * @retval None
  */
void USART_PrintString(USART_TypeDef* usartused, const int8_t* string)
{
  while(*string != '\0')
    {
      USART_PrintChar(usartused, *string);
      string++;
    }
}

/**
  * @brief  Send decimal value by strings via USART
  * @param  usartused : specify USART used
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  plussign : set to print '+' for plus value
  * @retval : None
  */
void USART_PrintDecimal(USART_TypeDef* usartused, int32_t intvalue, uint32_t width, uint8_t plussign)
{
  int8_t buffer[12];
  if (width == 0 && intvalue > 0 && plussign == 0)
    {
      Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');
    }
  else if (width == 0 && intvalue > 0 && plussign == 1)
    {
      buffer[0] = '+';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else if (width == 0 && intvalue < 0)
    {
      buffer[0] = '-';
      Uint32_tToDecimal(-intvalue, &buffer[1], width, ' ');
    }
  else if (width == 0 && intvalue == 0)
    {
      Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');
    }
  else if (plussign != 0 && intvalue > 0)
    {
      buffer[0] = '+';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else if ((plussign == 0 && intvalue > 0) || intvalue == 0)
    {
      buffer[0] = ' ';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else
    {
      buffer[0] = '-';
      Uint32_tToDecimal(-intvalue, &buffer[1], width, ' ');
    }
  USART_PrintString(usartused, buffer);
}

/**
  * @brief  Send decimal value by strings via USART
  * @param  usartused : specify USART used
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  plussign : set to print '+' for plus value
  * @retval : None
  */
void USART_PrintUnsignedDecimal(USART_TypeDef* usartused, int32_t intvalue, uint32_t width)
{
  int8_t buffer[11];

  Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');

  USART_PrintString(usartused, buffer);
}


/**
  * @brief  Send Hexadecimal strings via USART
  * @param  usartused : specify USART used
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  smallcase: 1 to small(90abc), 0 to large(09ABC)
  * @retval : None
  */
void USART_PrintHexaecimal(USART_TypeDef* usartused, int32_t intvalue, uint32_t width, uint8_t smallcase)
{
  int8_t buffer[9];

  Uint32_tToHexadecimal(intvalue, buffer, width, smallcase);

  USART_PrintString(usartused, buffer);
}

/**
  * @brief  Send binary strings via USART
  * @param  usartused : specify USART used
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @retval : None
  */
void USART_PrintBinary(USART_TypeDef* usartused, int32_t intvalue, uint32_t width)
{
  int8_t buffer[33];

  Uint32_tToBinary(intvalue, buffer, width);

  USART_PrintString(usartused, buffer);
}

/**
  * @brief  Send formatted string via USART
  * @param  usartused : specify USART used
  * @param  string : string to be send
  * @param  ...: set arguments for identifier in string
  * @retval : None
  */
void USART_PrintFormatted(USART_TypeDef* usartused, const int8_t* string, ...)
{
  va_list arg;
  uint8_t width;

  va_start(arg, string);

  while (*string != '\0')
    {
      if(*string == '%')
        {
          width = 0;
          string++;

          // acquire width as long as number lasts
          while (*string >= '0' && *string <= '9')
            {
              width = (width * 10) + (*string - '0');
              string++;
            }

          // detect identifier
          switch(*string)
          {
            // signed decimal without plus sign for plus value
            case 'd':
              USART_PrintDecimal(usartused, va_arg(arg, int32_t) , width, 0);
              string++;
              break;
            // signed decimal with plus sign for plus value
            case 'D':
              USART_PrintDecimal(usartused, va_arg(arg, int32_t) , width, 1);
              string++;
              break;
            // signed decimal with plus sign for plus value
            case 'u':
              USART_PrintUnsignedDecimal(usartused, va_arg(arg, int32_t) , width);
              string++;
              break;
            // hexadecimal with small case
            case 'x':
              USART_PrintHexaecimal(usartused, va_arg(arg, uint32_t) , width, 1);
              string++;
              break;
            // hexadecimal with large case
            case 'X':
              USART_PrintHexaecimal(usartused, va_arg(arg, uint32_t) , width, 0);
              string++;
              break;
            // binary
            case 'b':
              USART_PrintBinary(usartused, va_arg(arg, uint32_t) , width);
              string++;
              break;
            // one character
            case 'c':
              USART_PrintChar(usartused, (int8_t)va_arg(arg, int32_t));
              string++;
              break;
            // string
            case 's':
              USART_PrintString(usartused, va_arg(arg, int8_t*));
              string++;
              break;
            default:
              USART_PrintChar(usartused, *string);
              string++;
              break;
          }
        }
      else
        {
          USART_PrintChar(usartused, *string);
          string++;
        }
    }

  va_end(arg);
}
