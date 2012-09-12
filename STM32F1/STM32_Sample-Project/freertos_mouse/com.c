/**
  ******************************************************************************
  * @file    freertos_mouse/com.c
  * @author  Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   motor functions
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
#include "com.h"
#include "xbee.h"
#include "toascii.h"
#include <stdarg.h>
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define WIRED_TX_BUFFER_SIZE  256
#define WIRED_RX_BUFFER_SIZE  256
#define WIRED_NVIC_CHANNEL    USART2_IRQn
#define WIRED_USE_USART2
#define WIRED_USART           USART2
#define WIRED_USART_TX_PIN    USART2_TX_PIN
#define WIRED_USART_RX_PIN    USART2_RX_PIN
#define WIRED_USART_PORT      USART2_PORT
#define WIRED_IRQHANLDER      USART2_IRQHandler
#define XBEE_TX_BUFFER_SIZE       256
#define XBEE_RX_BUFFER_SIZE       16
#define XBEE_NVIC_CHANNEL     USART3_IRQn
#define XBEE_USE_USART3
#define XBEE_USART            USART3
#define XBEE_USART_TX_PIN     USART3_TX_PIN
#define XBEE_USART_RX_PIN     USART3_RX_PIN
#define XBEE_USART_PORT       USART3_PORT
#define XBEE_IRQHANLDER       USART3_IRQHandler
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
int8_t Wired_TxBuffer[WIRED_TX_BUFFER_SIZE];
uint16_t Wired_SubscriptInProgress = 0, Wired_SubscriptToAdd = 0;
int8_t Wired_RxBuffer[WIRED_RX_BUFFER_SIZE];
int8_t XBee_TxBuffer[XBEE_TX_BUFFER_SIZE];
uint16_t XBee_SubscriptInProgress = 0, XBee_SubscriptToAdd = 0;
int8_t XBee_RxBuffer[XBEE_RX_BUFFER_SIZE];
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
 * @brief  Configure peripherals for music
 * @param  None
 * @retval None
 */
void COM_Configuration(void)
{
  // Configure NVIC for COM port DMA
  COM_NVIC_Configuration();
  // Configuration USART for wired communication
  COM_Wired_Configuration(115200);
  // Configuration USART for XBee communication
  XBee_GPIO_Configuration();
  COM_XBee_Configuration(115200);
}

/**
  * @brief  Configure the nested vectored interrupt controller.
  * @param  None
  * @retval None
  */
void COM_NVIC_Configuration(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  /* Enable the USART for wired Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = WIRED_NVIC_CHANNEL;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 13;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

  /* Enable the USART for wired Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = XBEE_NVIC_CHANNEL;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 13;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

}

/**
 * @brief  FreeRTOS Task
 * @param  pvParameters : parameter passed from xTaskCreate
 * @retval None
 */
void prvTask_COM_RX(void *pvParameters)
{
  int8_t* pcTaskName;
  pcTaskName = (int8_t *) pvParameters;
  portTickType xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();


  while (1)
    {

    }
}

/**
  * @brief  Configure the USART for wired communication.
  * @param  None
  * @retval : None
  */
void COM_Wired_Configuration(uint32_t baudrate)
{
  GPIO_InitTypeDef GPIO_InitStructure;

#if defined(WIRED_USE_USART1)
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART1_RCC | USART1_GPIO_RCC, ENABLE);
#elif defined(WIRED_USE_USART2)
  /* Supply APB1 clock */
  RCC_APB1PeriphClockCmd(USART2_RCC, ENABLE);
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART2_GPIO_RCC, ENABLE);
#elif defined(WIRED_USE_USART3)
  /* Supply APB1 clock */
  RCC_APB1PeriphClockCmd(USART3_RCC, ENABLE);
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART3_GPIO_RCC, ENABLE);
#endif

  /* Configure USART1 Tx as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = WIRED_USART_TX_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_Init(WIRED_USART_PORT, &GPIO_InitStructure);

  /* Configure USART1 Rx as input floating */
  GPIO_InitStructure.GPIO_Pin = WIRED_USART_RX_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(WIRED_USART_PORT, &GPIO_InitStructure);

#if defined(WIRED_USE_USART1) && defined (REMAP_USART1)
Remap_USART1_Configuration();
#endif

#if defined(WIRED_USE_USART2) && defined (REMAP_USART2)
Remap_USART2_Configuration();
#endif

#if defined(WIRED_USE_USART3) && defined (PARTIAL_REMAP_USART3)
PartialRemap_USART3_Configuration();
#elif defined(WIRED_USE_USART3) && defined (FULL_REMAP_USART3)
FullRemap_USART3_Configuration();
#endif

  /* USART configuration ------------------------------------------------------*/
  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = baudrate;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
  USART_Init(WIRED_USART, &USART_InitStructure);

  //Enable USART RX not empty interrupt
  USART_ITConfig(WIRED_USART, USART_IT_RXNE, ENABLE);

  /* Enable the USART */
  USART_Cmd(WIRED_USART, ENABLE);
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_Wired_PrintChar(int8_t charvalue)
{
  //if it comes to a buffer point just before one processing, wait until the next to be send
  while((Wired_SubscriptToAdd + 1 == Wired_SubscriptInProgress) || (Wired_SubscriptToAdd == WIRED_TX_BUFFER_SIZE -1 && Wired_SubscriptInProgress == 0)){}

  Wired_TxBuffer[Wired_SubscriptToAdd++] = charvalue;
  //if subscript reaches end make to go back to front
  if (Wired_SubscriptToAdd == WIRED_TX_BUFFER_SIZE)
    {
      Wired_SubscriptToAdd = 0;
    }
  //Enable USART TX empty interrupt
  USART_ITConfig(WIRED_USART, USART_IT_TXE, ENABLE);

}

/**
  * @brief  This function handles USARTy global interrupt request.
  * @param  None
  * @retval None
  */
void WIRED_IRQHANLDER(void)
{
  portBASE_TYPE xHigherPriorityTaskWoken;
  xHigherPriorityTaskWoken = pdFALSE;

  static int8_t controlcommand = '\0';
  int8_t charbuffer;
  static uint32_t tempvalue;
  static uint8_t columncounter;
  COMQueue_Type COM_Inst;
  XBeeQueue_Type XBee_Inst;

  if(USART_GetITStatus(WIRED_USART, USART_IT_TXE) != RESET)
    {

      /* Write one byte to the transmit data register */
      USART_SendData(WIRED_USART, Wired_TxBuffer[Wired_SubscriptInProgress++]);

      // if subscript reached the end of buffer, make it go to front
      if (Wired_SubscriptInProgress == WIRED_TX_BUFFER_SIZE)
        {
          Wired_SubscriptInProgress = 0;
        }

      // if subscript processing reaches to subscript the next byte to be add
      // stops generating interrupt and stop sending.
      if (Wired_SubscriptInProgress == Wired_SubscriptToAdd)
        {
          /* Disable the USARTy Transmit interrupt */
          USART_ITConfig(WIRED_USART, USART_IT_TXE, DISABLE);
        }
    }

  // making this routine as independent task is more appropriate because queue could be full
  if(USART_GetITStatus(WIRED_USART, USART_IT_RXNE) != RESET)
    {
      charbuffer = USART_ReceiveData(WIRED_USART);
      COM_Wired_PrintChar(charbuffer);
      if (charbuffer == '#' || charbuffer == '$' || charbuffer == '&')
        {
          controlcommand = charbuffer;
          tempvalue = 0;
          columncounter = 0;
        }
      else if (charbuffer >= '0' && charbuffer <='9' &&
               (controlcommand == '#' || controlcommand == '$' || controlcommand == '&'))
        {
          tempvalue = (tempvalue * 0x10) + charbuffer - 0x30;
          columncounter++;
          if (columncounter > 8)
            {
              controlcommand = '\0';
            }
        }
      else if (charbuffer == '\r' && controlcommand == '#')
        {
          COM_Inst.Parameter = tempvalue;
          controlcommand = '\0';
          xQueueSendToBackFromISR(xCOMQueue, &COM_Inst, &xHigherPriorityTaskWoken );
        }
      else if (charbuffer == '\r' && controlcommand == '$')
        {
          COM_XBee_PrintFormatted("&%8x\r",tempvalue);
          controlcommand = '\0';
        }
      else if (charbuffer == '\r' && controlcommand == '&')
        {
          XBee_Inst.Parameter = tempvalue;
          controlcommand = '\0';
          xQueueSendToBackFromISR(xXBeeQueue, &XBee_Inst, &xHigherPriorityTaskWoken );
        }
      else
        {
          controlcommand = '\0';
        }
    }

  // context switch : Set PendSV bit
  portEND_SWITCHING_ISR( xHigherPriorityTaskWoken );

}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_Wired_PrintString(const int8_t string[])
{
  while(*string != '\0')
    {
      COM_Wired_PrintChar(*string);
      string++;
    }
}

/**
  * @brief  Send Strings via USART
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  plussign : set to print '+' for plus value
  * @retval : None
  */
void COM_Wired_PrintDecimal(int32_t intvalue, uint32_t width, uint8_t plussign)
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
  COM_Wired_PrintString(buffer);
}


/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @param  smallcase: 1 to small(90abc), 0 to large(09ABC)
  * @retval : None
  */
void COM_Wired_PrintHexaecimal(int32_t intvalue, uint32_t width, uint8_t smallcase)
{
  int8_t buffer[9];

  Uint32_tToHexadecimal(intvalue, buffer, width, smallcase);

  COM_Wired_PrintString(buffer);
}


/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_Wired_PrintBinary(int32_t intvalue, uint32_t width)
{
  int8_t buffer[33];

  Uint32_tToBinary(intvalue, buffer, width);

  COM_Wired_PrintString(buffer);
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_Wired_PrintFormatted(int8_t* string, ...)
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
              COM_Wired_PrintDecimal(va_arg(arg, int32_t) , width, 0);
              string++;
              break;
            // signed decimal with plus sign for plus value
            case 'D':
              COM_Wired_PrintDecimal(va_arg(arg, int32_t) , width, 1);
              string++;
              break;
            // hexadecimal with small case
            case 'x':
              COM_Wired_PrintHexaecimal(va_arg(arg, uint32_t) , width, 1);
              string++;
              break;
            // hexadecimal with large case
            case 'X':
              COM_Wired_PrintHexaecimal(va_arg(arg, uint32_t) , width, 0);
              string++;
              break;
            // binary
            case 'b':
              COM_Wired_PrintBinary(va_arg(arg, uint32_t) , width);
              string++;
              break;
            // string
            case 's':
              COM_Wired_PrintString(va_arg(arg, int8_t*));
              string++;
              break;
            default:
              COM_Wired_PrintChar(*string);
              string++;
              break;
          }
        }
      else
        {
          COM_Wired_PrintChar(*string);
          string++;
        }
    }

  va_end(arg);
}

/**
  * @brief  Send Strings via USART and XBee
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_PrintString(const int8_t string[])
{
  COM_Wired_PrintString(string);
  COM_XBee_PrintString(string);
}

/**
  * @brief  Send decimal data expressed in 0-9 ascii string
  * @param  Byte: data to be send
  * @retval : None
  */
void COM_PrintDecimal(uint32_t intvalue)
{
  COM_Wired_PrintDecimal(intvalue, 0, 0);
  COM_XBee_PrintDecimal(intvalue, 0, 0);
}

/**
  * @brief  Configure the GPIO Pins.
  * @param  None
  * @retval : None
  */
void XBee_GPIO_Configuration(void)
{
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(GPIOY_0_RCC, ENABLE);

  /* Configure GPIOY_0: output push-pull */
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin = GPIOY_0_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;
  GPIO_Init(GPIOY_0_PORT, &GPIO_InitStructure);
}

/**
  * @brief  Configure the USART for Xbee communication.
  * @param  None
  * @retval : None
  */
void COM_XBee_Configuration(uint32_t baudrate)
{
  GPIO_InitTypeDef GPIO_InitStructure;

#if defined(XBEE_USE_USART1)
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART1_RCC | USART1_GPIO_RCC, ENABLE);
#elif defined(XBEE_USE_USART2)
  /* Supply APB1 clock */
  RCC_APB1PeriphClockCmd(USART2_RCC, ENABLE);
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART2_GPIO_RCC, ENABLE);
#elif defined(XBEE_USE_USART3)
  /* Supply APB1 clock */
  RCC_APB1PeriphClockCmd(USART3_RCC, ENABLE);
  /* Supply APB2 clock */
  RCC_APB2PeriphClockCmd(USART3_GPIO_RCC, ENABLE);
#endif

  /* Configure USART1 Tx as alternate function push-pull */
  GPIO_InitStructure.GPIO_Pin = XBEE_USART_TX_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
  GPIO_Init(XBEE_USART_PORT, &GPIO_InitStructure);

  /* Configure USART1 Rx as input floating */
  GPIO_InitStructure.GPIO_Pin = XBEE_USART_RX_PIN;
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
  GPIO_Init(XBEE_USART_PORT, &GPIO_InitStructure);

#if defined(XBEE_USE_USART1) && defined (REMAP_USART1)
Remap_USART1_Configuration();
#endif

#if defined(XBEE_USE_USART2) && defined (REMAP_USART2)
Remap_USART2_Configuration();
#endif

#if defined(XBEE_USE_USART3) && defined (PARTIAL_REMAP_USART3)
PartialRemap_USART3_Configuration();
#elif defined(XBEE_USE_USART3) && defined (FULL_REMAP_USART3)
FullRemap_USART3_Configuration();
#endif

  /* USART configuration ------------------------------------------------------*/
  USART_InitTypeDef USART_InitStructure;
  USART_InitStructure.USART_BaudRate = baudrate;
  USART_InitStructure.USART_WordLength = USART_WordLength_8b;
  USART_InitStructure.USART_StopBits = USART_StopBits_1;
  USART_InitStructure.USART_Parity = USART_Parity_No ;
  USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
  USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
  USART_Init(XBEE_USART, &USART_InitStructure);

  //Enable USART RX not empty interrupt
  USART_ITConfig(XBEE_USART, USART_IT_RXNE, ENABLE);

  /* Enable the USART */
  USART_Cmd(XBEE_USART, ENABLE);
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_XBee_PrintChar(int8_t charvalue)
{
  //if it comes to a buffer point just before one processing, wait until the next to be send
  while((XBee_SubscriptToAdd + 1 == XBee_SubscriptInProgress) || (XBee_SubscriptToAdd == XBEE_TX_BUFFER_SIZE -1 && XBee_SubscriptInProgress == 0)){}

  XBee_TxBuffer[XBee_SubscriptToAdd++] = charvalue;
  //if subscript reaches end make to go back to front
  if (XBee_SubscriptToAdd == XBEE_TX_BUFFER_SIZE)
    {
      XBee_SubscriptToAdd = 0;
    }
  //Enable USART TX empty interrupt
  USART_ITConfig(XBEE_USART, USART_IT_TXE, ENABLE);

}

/**
  * @brief  This function handles USARTy global interrupt request.
  * @param  None
  * @retval None
  */
void XBEE_IRQHANLDER(void)
{
  portBASE_TYPE xHigherPriorityTaskWoken;
  xHigherPriorityTaskWoken = pdFALSE;

  static int8_t controlcommand = '\0';
  int8_t charbuffer;
  static uint32_t tempvalue;
  static uint8_t columncounter;
  COMQueue_Type COM_Inst;
  XBeeQueue_Type XBee_Inst;

  if(USART_GetITStatus(XBEE_USART, USART_IT_TXE) != RESET)
    {

      /* Write one byte to the transmit data register */
      USART_SendData(XBEE_USART, XBee_TxBuffer[XBee_SubscriptInProgress++]);

      // if subscript reached the end of buffer, make it go to front
      if (XBee_SubscriptInProgress == XBEE_TX_BUFFER_SIZE)
        {
          XBee_SubscriptInProgress = 0;
        }

      // if subscript processing reaches to subscript the next byte to be add
      // stops generating interrupt and stop sending.
      if (XBee_SubscriptInProgress == XBee_SubscriptToAdd)
        {
          /* Disable the USARTy Transmit interrupt */
          USART_ITConfig(XBEE_USART, USART_IT_TXE, DISABLE);
        }
    }

  // making this routine as independent task is more appropriate because queue could be full
  if(USART_GetITStatus(XBEE_USART, USART_IT_RXNE) != RESET)
    {
      charbuffer = USART_ReceiveData(XBEE_USART);
      if (charbuffer == '#' || charbuffer == '$' || charbuffer == '&')
        {
          controlcommand = charbuffer;
          tempvalue = 0;
          columncounter = 0;
        }
      else if (
          ((charbuffer >= '0' && charbuffer <='9') ||
           (charbuffer >= 'A' && charbuffer <='F')) &&
          (controlcommand == '#' || controlcommand == '$' || controlcommand == '&'))
        {
          if (charbuffer >= '0' && charbuffer <='9')
            {
              tempvalue = (tempvalue * 0x10) + charbuffer - 0x30;
            }
          else
            {
              tempvalue = (tempvalue * 0x10) + charbuffer - 0x41 + 0xa;
            }
          columncounter++;
          if (columncounter > 8)
            {
              controlcommand = '\0';
            }
        }
      else if (charbuffer == '\r' && controlcommand == '#')
        {
          COM_Inst.Parameter = tempvalue;
          controlcommand = '\0';
          xQueueSendToBackFromISR(xCOMQueue, &COM_Inst, &xHigherPriorityTaskWoken );
        }
      else if (charbuffer == '\r' && controlcommand == '$')
        {
          COM_XBee_PrintFormatted("&%8X\r",tempvalue);
          controlcommand = '\0';
        }
      else if (charbuffer == '\r' && controlcommand == '&')
        {
          XBee_Inst.Parameter = tempvalue;
          controlcommand = '\0';
          xQueueSendToBackFromISR(xXBeeQueue, &XBee_Inst, &xHigherPriorityTaskWoken );
        }
      else
        {
          controlcommand = '\0';
        }
    }

  // context switch : Set PendSV bit
  portEND_SWITCHING_ISR( xHigherPriorityTaskWoken );

}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_XBee_PrintString(const int8_t string[])
{
  while(*string != '\0')
    {
      COM_XBee_PrintChar(*string);
      string++;
    }
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_XBee_PrintDecimal(int32_t intvalue, uint32_t width, uint8_t plussign)
{
  int8_t buffer[12];
  if (plussign != 0 && intvalue > 0)
    {
      buffer[0] = '+';
      Uint32_tToDecimal(intvalue, &buffer[1], width, ' ');
    }
  else if ((plussign == 0 && intvalue > 0) || intvalue == 0)
    {
      Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');
    }
  else
    {
      buffer[0] = '-';
      Uint32_tToDecimal(-intvalue, &buffer[1], width, ' ');
    }
  COM_XBee_PrintString(buffer);
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @param  smallcase: 1 to small(90abc), 0 to large(09ABC)
  * @retval : None
  */
void COM_XBee_PrintHexaecimal(int32_t intvalue, uint32_t width, uint8_t smallcase)
{
  int8_t buffer[9];

  Uint32_tToHexadecimal(intvalue, buffer, width, smallcase);

  COM_XBee_PrintString(buffer);
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_XBee_PrintBinary(int32_t intvalue, uint32_t width)
{
  int8_t buffer[33];

  Uint32_tToBinary(intvalue, buffer, width);

  COM_XBee_PrintString(buffer);
}

/**
  * @brief  Send Strings via USART
  * @param  String: Array containing string to be sent
  * @retval : None
  */
void COM_XBee_PrintFormatted(int8_t* string, ...)
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
              COM_XBee_PrintDecimal(va_arg(arg, int32_t) , width, 0);
              string++;
              break;
            // signed decimal with plus sign for plus value
            case 'D':
              COM_XBee_PrintDecimal(va_arg(arg, int32_t) , width, 1);
              string++;
              break;
            // hexadecimal with small case
            case 'x':
              COM_XBee_PrintHexaecimal(va_arg(arg, uint32_t) , width, 1);
              string++;
              break;
            // hexadecimal with large case
            case 'X':
              COM_XBee_PrintHexaecimal(va_arg(arg, uint32_t) , width, 0);
              string++;
              break;
            // binary
            case 'b':
              COM_XBee_PrintBinary(va_arg(arg, uint32_t) , width);
              string++;
              break;
            // string
            case 's':
              COM_XBee_PrintString(va_arg(arg, int8_t*));
              string++;
              break;
            default:
              COM_XBee_PrintChar(*string);
              string++;
              break;
          }
        }
      else
        {
          COM_XBee_PrintChar(*string);
          string++;
        }
    }

  va_end(arg);
}

