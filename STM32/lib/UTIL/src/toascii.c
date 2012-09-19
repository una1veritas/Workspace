/**
  ******************************************************************************
  * @file    UTIL/src/toascii.c
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
#include "toascii.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Extern variables ----------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/**
  * @brief  convert integral to binary ASCII string
  * @param  Byte: data to be converted
  * @param  String: array to conrain converted ASCII string
  * @retval : None
  */
void To_Binary(uint8_t Byte, int8_t String[])
{
  uint8_t i,j;
  j = 7;
  for(i = 0; i<8; i++)
    {
      String[i] = ((Byte >> j) & 0b00000001) + 0x30 ;
      j--;
    }
  String[8] = '\0';
}

/**
  * @brief  convert integral to hexadecimal ASCII string
  * @param  Byte: data to be converted
  * @param  String: array to conrain converted ASCII string
  * @retval : None
  */
void To_Hexadecimal(uint8_t Byte, int8_t String[])
{
  uint8_t i,j,l4bit;
  j = 4;
  for(i = 0; i<2; i++)
    {
      l4bit = (Byte >> j) & 0b00001111;
      if (l4bit <= 9)
        {
          String[i] = l4bit + 0x30 ;
        }
      else
        {
          String[i] = l4bit + 0x37 ;
        }
      j = j - 4;
    }
  String[2] = '\0';
}

/**
  * @brief  convert integral to decimal ASCII string
  * @param  Intdata: data to be converted
  * @param  String: array to conrain converted ASCII string
  * @retval : None
  */
void To_Decimal(uint32_t Intdata, int8_t String[])
{
  uint8_t i,column = 0;
  uint32_t quotient,remainder;
  uint32_t div = 1000000000;
  FlagStatus zero_flag = RESET;
  remainder = Intdata;
  if (remainder != 0)
    {
      for(i = 0; i<10; i++)
        {
          quotient = remainder / div;
          remainder = remainder % div;
          if (quotient != 0)
            {
              String[column] = quotient + 0x30 ;
              column++;
              zero_flag = SET;
            }
          else if (quotient == 0 && zero_flag != RESET)
            {
              String[column] = quotient + 0x30 ;
              column++;
            }
          div = div /10;
        }
      String[column] = '\0';
    }
  else
    {
      String[0] = 0x30;
      String[1] = '\0';
    }
}

/**
  * @brief  change integral to string
  * @param  intvalue : value to be changed
  * @param  string : pointer to buffer to save converted
  * @param  width : maximum width to change, set 0 not to restrict
  * @param  filler : when width is set and value is less than width, left column is filled this character
  * @retval : None
  */
void Uint32_tToDecimal(uint32_t intvalue, int8_t* string, uint8_t width, uint8_t filler)
{
  // divider , also acts as  and set criteria
  uint32_t div = 1000000000;
  // loop counter (intvalue numerical column in progress)
  // logically needless, but implemented for lack of log function
  uint8_t column = 10;
  // subscript of string buffer in progress
  uint8_t sub = 0;
  // set when division is able
  uint8_t flag = 0;

  // main loop: process for 10 times
  while (column > 0)
    {
      // decide a need to set number to string
      // case : there is no value for this column
      if(intvalue < div && flag == 0 && !(column == 1 && intvalue ==0))
        {
          // width : not designated
          if (width == 0)
            {
              // do not change subscript
              // sub++;
              // no needs to process intvalue
              // intvalue = intvalue % div
              // divide div for next column
              div = div / 10;
              column--;
            }
          // width : designated and width is not less than column
          else if (width >= column)
            {
              // fill column by filler character
              string[sub] = filler;
              // proceed subscript
              sub++;
              // no needs to process intvalue
              // intvalue = intvalue % div
              // divide div for next column
              div = div / 10;
              column--;
            }
          // width : designated and width is less than column
          else
            {
              // do not change subscript
              // sub++;
              // no needs to process intvalue
              // intvalue = intvalue % div
              // divide div for next column
              div = div / 10;
              column--;
            }
        }
      else
        {
          flag = 1;
          // width : not designated
          if (width == 0)
            {
              // fill column by character of divided number
              string[sub] = (intvalue / div) + 0x30;
              // proceed subscript
              sub++;
              // set remainder to intvalue for next column
              intvalue = intvalue % div;
              // divide div for next column
              div = div / 10;
              column--;
            }
          // width : designated and width is not less than column
          else if (width >= column)
            {
              // fill column by character of divided number
              string[sub] = (intvalue / div) + 0x30;
              // proceed subscript
              sub++;
              // set remainder to intvalue for next column
              intvalue =  intvalue % div;
              // divide div for next column
              div = div / 10;
              column--;
            }
          // width : designated and width is less than column
          else
            {
              // do not fill column
              // string[sub] = (intvalue / div) + 0x30;
              // do not change subscript
              // sub++;
              // set remainder to intvalue for next column
              intvalue =  intvalue % div;
              // divide div for next column
              div = div / 10;
              column--;
            }
        }
    }
  // set null character at the end of string
  string[sub] = '\0';
}

/**
  * @brief  convert integral to hexadecimal ASCII string
  * @param  intvalue: data to be converted
  * @param  string: array to conrain converted ASCII string
  * @param  width: maximum width to change, if set 0 , 1 byte is printed
  * @param  smallcase: 1 to small(90abc), 0 to large(09ABC)
  * @retval : None
  */
void Uint32_tToHexadecimal(uint32_t intvalue, int8_t* string, uint8_t width , uint8_t smallcase)
{
  if (width  == 0)
    {
      width = 2;
    }

  while(width > 0)
    {
      // numbers
      if(((intvalue >> ((width-1) * 4)) & 0x0000000F) < 10)
        {
          *string = ((intvalue >> ((width-1) * 4)) & 0x0000000F) + 0x30;
        }
      // alphabets
      else
        {
          *string = ((intvalue >> ((width-1) * 4)) & 0x0000000F) - 9 + 0x40  + (smallcase * 0x20);
        }
      width--;
      string++;
    }
  *string = '\0';
}

/**
  * @brief  convert integral to binary ASCII string
  * @param  intvalue: data to be converted
  * @param  string: array to conrain converted ASCII string
  * @param  width: maximum width to change, if set 0, 1 byte is printed
  * @retval : None
  */
void Uint32_tToBinary(uint32_t intvalue, int8_t* string, uint8_t width )
{
  if (width  == 0)
    {
      width = 8;
    }

  while(width > 0)
    {
      *string = ((intvalue >> (width -1)) & 0b1) + 0x30;
      width--;
      string++;
    }
  *string = '\0';
}

