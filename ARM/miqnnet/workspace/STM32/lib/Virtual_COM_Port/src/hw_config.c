/******************** (C) COPYRIGHT 2010 STMicroelectronics ********************
* File Name          : hw_config.c
* Author             : MCD Application Team
* Version            : V3.1.1
* Date               : 04/07/2010
* Description        : Hardware Configuration & Setup
********************************************************************************
* THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
* WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE TIME.
* AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY DIRECT,
* INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING FROM THE
* CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE CODING
* INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
*******************************************************************************/

/* Includes ------------------------------------------------------------------*/
#include "stm32f10x_it.h"
#include "usb_lib.h"
#include "usb_prop.h"
#include "usb_desc.h"
#include "hw_config.h"
#include "platform_config.h"
#include "usb_pwr.h"
//#include "stm32_eval.h"
#include "toascii.h"
#include <stdarg.h>
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
ErrorStatus HSEStartUpStatus;
USART_InitTypeDef USART_InitStructure;
uint8_t buffer_in[VIRTUAL_COM_PORT_DATA_SIZE];
__IO uint8_t buffer_counter = 0;

/* Extern variables ----------------------------------------------------------*/
extern uint32_t count_in;
extern LINE_CODING linecoding;

extern __IO uint32_t count_out;
extern uint8_t buffer_out[VIRTUAL_COM_PORT_DATA_SIZE];

/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/*******************************************************************************
* Function Name  : Set_System
* Description    : Configures Main system clocks & power
* Input          : None.
* Return         : None.
*******************************************************************************/
void Set_System(void)
{
#ifndef USE_STM3210C_EVAL
  GPIO_InitTypeDef GPIO_InitStructure;
#endif /* USE_STM3210C_EVAL */

  /* SYSCLK, HCLK, PCLK2 and PCLK1 configuration -----------------------------*/
  /* deleted 20100513 yasuokawachi -----------------------------*/


#ifndef USE_STM3210C_EVAL
  /* Enable USB_DISCONNECT GPIO clock */
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIO_DISCONNECT, ENABLE);

  /* Configure USB pull-up pin */
  GPIO_InitStructure.GPIO_Pin = USB_DISCONNECT_PIN;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
#ifndef USB_DISCONNECT_POLARITY_REVERSE
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
#else
  GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
#endif
  GPIO_Init(USB_DISCONNECT, &GPIO_InitStructure);
#endif /* USE_STM3210C_EVAL */
}

/*******************************************************************************
* Function Name  : Set_USBClock
* Description    : Configures USB Clock input (48MHz)
* Input          : None.
* Return         : None.
*******************************************************************************/
void Set_USBClock(void)
{
#ifdef STM32F10X_CL
  /* Select USBCLK source */
  RCC_OTGFSCLKConfig(RCC_OTGFSCLKSource_PLLVCO_Div3);

  /* Enable the USB clock */ 
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_OTG_FS, ENABLE) ;
#else 
  /* Select USBCLK source */
  RCC_USBCLKConfig(RCC_USBCLKSource_PLLCLK_1Div5);
  
  /* Enable the USB clock */
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_USB, ENABLE);
#endif /* STM32F10X_CL */
}

/*******************************************************************************
* Function Name  : Enter_LowPowerMode
* Description    : Power-off system clocks and power while entering suspend mode
* Input          : None.
* Return         : None.
*******************************************************************************/
void Enter_LowPowerMode(void)
{
  /* Set the device state to suspend */
  bDeviceState = SUSPENDED;
}

/*******************************************************************************
* Function Name  : Leave_LowPowerMode
* Description    : Restores system clocks and power while exiting suspend mode
* Input          : None.
* Return         : None.
*******************************************************************************/
void Leave_LowPowerMode(void)
{
  DEVICE_INFO *pInfo = &Device_Info;

  /* Set the device state to the correct state */
  if (pInfo->Current_Configuration != 0)
  {
    /* Device configured */
    bDeviceState = CONFIGURED;
  }
  else
  {
    bDeviceState = ATTACHED;
  }
}

/*******************************************************************************
* Function Name  : USB_Interrupts_Config
* Description    : Configures the USB interrupts
* Input          : None.
* Return         : None.
*******************************************************************************/
void USB_Interrupts_Config(void)
{
  NVIC_InitTypeDef NVIC_InitStructure;

  NVIC_PriorityGroupConfig(NVIC_PriorityGroup_1);

#ifdef STM32F10X_CL 
  /* Enable the USB Interrupts */
  NVIC_InitStructure.NVIC_IRQChannel = OTG_FS_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
#else
  NVIC_InitStructure.NVIC_IRQChannel = USB_LP_CAN1_RX0_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
#endif /* STM32F10X_CL */

  /* Enable USART Interrupt */
  NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
  NVIC_Init(&NVIC_InitStructure);
}

/*******************************************************************************
* Function Name  : USB_Cable_Config
* Description    : Software Connection/Disconnection of USB Cable
* Input          : None.
* Return         : Status
*******************************************************************************/
void USB_Cable_Config (FunctionalState NewState)
{
#ifdef USE_STM3210C_EVAL  
  if (NewState != DISABLE)
  {
    USB_DevConnect();
  }
  else
  {
    USB_DevDisconnect();
  }
#else /* USE_STM3210B_EVAL or USE_STM3210E_EVAL */
  if (NewState != DISABLE)
  {
#ifndef USB_DISCONNECT_POLARITY_REVERSE
    GPIO_ResetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#else
    GPIO_SetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#endif
  }
  else
  {
#ifndef USB_DISCONNECT_POLARITY_REVERSE
    GPIO_SetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#else
    GPIO_ResetBits(USB_DISCONNECT, USB_DISCONNECT_PIN);
#endif
  }
#endif /* USE_STM3210C_EVAL */
}

/*******************************************************************************
* Function Name  :  USART_Config_Default.
* Description    :  configure the EVAL_COM1 with default values.
* Input          :  None.
* Return         :  None.
*******************************************************************************/
/* deleted 20100513 yasuokawachi -----------------------------*/

/*******************************************************************************
* Function Name  :  USART_Config.
* Description    :  Configure the EVAL_COM1 according to the linecoding structure.
* Input          :  None.
* Return         :  Configuration status
                    TRUE : configuration done with success
                    FALSE : configuration aborted.
*******************************************************************************/
/* deleted 20100513 yasuokawachi -----------------------------*/

/*******************************************************************************
* Function Name  : USB_To_USART_Send_Data.
* Description    : send the received data from USB to the UART 0.
* Input          : data_buffer: data address.
                   Nb_bytes: number of bytes to send.
* Return         : none.
*******************************************************************************/
/* deleted 20100513 yasuokawachi -----------------------------*/
/*void USB_To_USART_Send_Data(uint8_t* data_buffer, uint8_t Nb_bytes)
{
  uint32_t i;

  for (i = 0; i < Nb_bytes; i++)
  {
//	  cputchar(*(data_buffer + i));
//    while(USART_GetFlagStatus(EVAL_COM1, USART_FLAG_TXE) == RESET);
  }
}
*/

/*******************************************************************************
* Function Name  : UART_To_USB_Send_Data.
* Description    : send the received data from UART 0 to USB.
* Input          : None.
* Return         : none.
*******************************************************************************/
/* deleted 20100513 yasuokawachi -----------------------------*/
/*
void USART_To_USB_Send_Data(void)
{
  if (linecoding.datatype == 7)
  {
    buffer_in[count_in] = USART_ReceiveData(USART1) & 0x7F;
  }
  else if (linecoding.datatype == 8)
  {
    buffer_in[count_in] = USART_ReceiveData(USART1);
  }
  count_in++;

//   Write the data to the USB endpoint
  USB_SIL_Write(EP1_IN, buffer_in, count_in);
  
#ifndef STM32F10X_CL
  SetEPTxValid(ENDP1);
#endif // STM32F10X_CL
}
*/

/*******************************************************************************
* Function Name  : Get_SerialNum.
* Description    : Create the serial number string descriptor.
* Input          : None.
* Output         : None.
* Return         : None.
*******************************************************************************/
void Get_SerialNum(void)
{
  uint32_t Device_Serial0, Device_Serial1, Device_Serial2;

  Device_Serial0 = *(__IO uint32_t*)(0x1FFFF7E8);
  Device_Serial1 = *(__IO uint32_t*)(0x1FFFF7EC);
  Device_Serial2 = *(__IO uint32_t*)(0x1FFFF7F0);

  if (Device_Serial0 != 0)
  {
    Virtual_Com_Port_StringSerial[2] = (uint8_t)(Device_Serial0 & 0x000000FF);
    Virtual_Com_Port_StringSerial[4] = (uint8_t)((Device_Serial0 & 0x0000FF00) >> 8);
    Virtual_Com_Port_StringSerial[6] = (uint8_t)((Device_Serial0 & 0x00FF0000) >> 16);
    Virtual_Com_Port_StringSerial[8] = (uint8_t)((Device_Serial0 & 0xFF000000) >> 24);

    Virtual_Com_Port_StringSerial[10] = (uint8_t)(Device_Serial1 & 0x000000FF);
    Virtual_Com_Port_StringSerial[12] = (uint8_t)((Device_Serial1 & 0x0000FF00) >> 8);
    Virtual_Com_Port_StringSerial[14] = (uint8_t)((Device_Serial1 & 0x00FF0000) >> 16);
    Virtual_Com_Port_StringSerial[16] = (uint8_t)((Device_Serial1 & 0xFF000000) >> 24);

    Virtual_Com_Port_StringSerial[18] = (uint8_t)(Device_Serial2 & 0x000000FF);
    Virtual_Com_Port_StringSerial[20] = (uint8_t)((Device_Serial2 & 0x0000FF00) >> 8);
    Virtual_Com_Port_StringSerial[22] = (uint8_t)((Device_Serial2 & 0x00FF0000) >> 16);
    Virtual_Com_Port_StringSerial[24] = (uint8_t)((Device_Serial2 & 0xFF000000) >> 24);
  }
}

/*******************************************************************************
* Function Name  : USB_Send_Char.
* Description    : send the char data to USB.
* Input          : char data to be send
* Return         : none.
*******************************************************************************/
void VCP_PrintChar(uint8_t send_char)
{
  while(GetEPTxStatus(ENDP1) != EP_TX_NAK){}
  buffer_in[count_in] = send_char;
  count_in++;

//   Write the data to the USB endpoint
  USB_SIL_Write(EP1_IN, buffer_in, count_in);

#ifndef STM32F10X_CL
  SetEPTxValid(ENDP1);
#endif // STM32F10X_CL

}

/**
  * @brief  Send Strings via VCP
  * @param  string: Array containing string to be sent
  * @retval : None
  */
void VCP_PrintString(const int8_t string[])
{
  while(*string != '\0')
    {
      VCP_PrintChar(*string);
      string++;
    }
}

/**
  * @brief  Send decimal value by strings via VCP
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  plussign : set to print '+' for plus value
  * @retval : None
  */
void VCP_PrintDecimal(int32_t intvalue, uint32_t width, uint8_t plussign)
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
  VCP_PrintString(buffer);
}

/**
  * @brief  Send decimal value by strings via VCP
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  plussign : set to print '+' for plus value
  * @retval : None
  */
void VCP_PrintUnsignedDecimal(int32_t intvalue, uint32_t width)
{
  int8_t buffer[11];

  Uint32_tToDecimal(intvalue, &buffer[0], width, ' ');

  VCP_PrintString(buffer);
}


/**
  * @brief  Send Hexadecimal strings via VCP
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @param  smallcase: 1 to small(90abc), 0 to large(09ABC)
  * @retval : None
  */
void VCP_PrintHexaecimal(int32_t intvalue, uint32_t width, uint8_t smallcase)
{
  int8_t buffer[9];

  Uint32_tToHexadecimal(intvalue, buffer, width, smallcase);

  VCP_PrintString(buffer);
}

/**
  * @brief  Send binary strings via VCP
  * @param  intvalue : integral value to be send
  * @param  width: width to restrict output string
  * @retval : None
  */
void VCP_PrintBinary(int32_t intvalue, uint32_t width)
{
  int8_t buffer[33];

  Uint32_tToBinary(intvalue, buffer, width);

  VCP_PrintString(buffer);
}

/**
  * @brief  Send formatted string via VCP
  * @param  string : string to be send
  * @param  ...: set arguments for identifier in string
  * @retval : None
  */
void VCP_PrintFormatted(const int8_t* string, ...)
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
              VCP_PrintDecimal(va_arg(arg, int32_t) , width, 0);
              string++;
              break;
            // signed decimal with plus sign for plus value
            case 'D':
              VCP_PrintDecimal(va_arg(arg, int32_t) , width, 1);
              string++;
              break;
            // signed decimal with plus sign for plus value
            case 'u':
              VCP_PrintUnsignedDecimal(va_arg(arg, int32_t) , width);
              string++;
              break;
            // hexadecimal with small case
            case 'x':
              VCP_PrintHexaecimal(va_arg(arg, uint32_t) , width, 1);
              string++;
              break;
            // hexadecimal with large case
            case 'X':
              VCP_PrintHexaecimal(va_arg(arg, uint32_t) , width, 0);
              string++;
              break;
            // binary
            case 'b':
              VCP_PrintBinary(va_arg(arg, uint32_t) , width);
              string++;
              break;
            // one character
            case 'c':
              VCP_PrintChar((int8_t)va_arg(arg, int32_t));
              string++;
              break;
            // string
            case 's':
              VCP_PrintString(va_arg(arg, int8_t*));
              string++;
              break;
            default:
              VCP_PrintChar(*string);
              string++;
              break;
          }
        }
      else
        {
          VCP_PrintChar(*string);
          string++;
        }
    }

  va_end(arg);
}

/*******************************************************************************
* Function Name  : VCP_ReceiveData
* Description    : Receive one byte from VCP
* Input          : none.
* Return         : Received data from virtual COM port
*******************************************************************************/
uint8_t VCP_ReceiveData(void)
{
  __IO uint8_t RxData;
  RxData = buffer_out[buffer_counter];
  buffer_counter++;

  if (!(buffer_counter < count_out))
    {
      count_out = 0;
      buffer_counter = 0;
    }
  return(RxData);
}
