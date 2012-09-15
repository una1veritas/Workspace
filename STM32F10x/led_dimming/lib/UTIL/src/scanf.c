/**
  ******************************************************************************
  * @file    UTIL/src/scanf.c
  * @author  Yasuo Kawachi
  * @version  V1.0.0
  * @date  04/15/2009
  * @brief  convert integral to string expressed in ASCII
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
  */

/* Includes ------------------------------------------------------------------*/
#include "scanf.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Extern variables ----------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Gets numeric values from COM port
  * @param  None
  * @retval numeric value converted from COM input
  */
uint32_t COM_Num_Scanf(void)
{
  __IO int8_t   index = 0;
  __IO uint8_t  RxData, i, tmp[10];
  __IO uint32_t result = 0, mul = 1;

  while (index < 10)
    {
      while (1)
        {
          if (RX_BUFFER_IS_NOT_EMPTY)
            {
              RxData = (int8_t)RECEIVE_DATA;

              if ((RxData >= 0x30 && RxData <= 0x39) || RxData == 0x0D || RxData == 0x08)
                {
                  break;
                }
              else
                {
                  continue;
                }
            }
        }

      if (RxData == 0x0D)
        {
          break;
        }

      if (RxData == 0x08)
        {
          if (index != 0)
            {
              cputchar(0x08);
              cputchar(0x20);
              cputchar(0x08);
              index--;
            }
          continue;
        }

      cputchar(RxData);
      tmp[(uint8_t)index] = RxData - 0x30;
      index++;
    }

  for (i=index;i>0;i--)
    {
      result = result + ( tmp[i-1] * mul);
      mul = mul * 10;
    }

  return result;
}

/**
  * @brief  Gets numeric values from COM port
  * @param  None
  * @retval numeric value converted from COM input
  */
void COM_Char_Scanf(int8_t String[], uint16_t max_len)
{
  __IO uint16_t index = 0;
  __IO uint8_t  RxData;

  while (index < max_len)
    {
      while (1)
        {
          if (RX_BUFFER_IS_NOT_EMPTY)
            {
              RxData = (int8_t)RECEIVE_DATA;

              if ((RxData >= 0x20 && RxData <= 0x7E) || RxData == 0x0D || RxData == 0x08)
                {
                  break;
                }
              else
                {
                  continue;
                }
            }
        }

      if (RxData == 0x0D)
        {
          break;
        }

      if (RxData == 0x08)
        {
          if (index != 0)
            {
              cputchar(0x08);
              cputchar(0x20);
              cputchar(0x08);
              index--;
            }
          continue;
        }

      cputchar(RxData);
      String[index] = RxData;
      index++;
    }
  String[index] = '\0';

}
