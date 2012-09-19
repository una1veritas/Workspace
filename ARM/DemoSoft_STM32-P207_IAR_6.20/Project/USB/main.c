/**
******************************************************************************
* @file    Project/STM32F2xx_StdPeriph_Template/main.c
* @author  MCD Application Team
* @version V0.0.3
* @date    10/15/2010
* @brief   Main program body
******************************************************************************
* @copy
*
* THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
* WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE
* TIME. AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY
* DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING
* FROM THE CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE
* CODING INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
*
* <h2><center>&copy; COPYRIGHT 2010 STMicroelectronics</center></h2>
*/

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <yfuns.h>
#include "includes.h"

#define DLY_100US  1000

Int32U CriticalSecCntr;
/** @addtogroup Template_Project
* @{
*/

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
extern FontType_t Terminal_6_8_6;
extern FontType_t Terminal_9_12_6;
extern FontType_t Terminal_18_24_12;

USART_InitTypeDef USART_InitStructure;
static __IO uint32_t TimingDelay;
RCC_ClocksTypeDef RCC_Clocks;

void SysTickStart(uint32_t Tick)
{
	RCC_ClocksTypeDef Clocks;
	volatile uint32_t dummy;

	RCC_GetClocksFreq(&Clocks);

	dummy = SysTick->CTRL;
	SysTick->LOAD = (Clocks.HCLK_Frequency/8)/Tick;

	SysTick->CTRL = 1;
}

void SysTickStop(void)
{
	SysTick->CTRL = 0;
}

/* Private function prototypes -----------------------------------------------*/
static int MyLowLevelGetchar(void);
size_t __write(int Handle, const unsigned char * Buf, size_t Bufsize);
size_t __read(int handle, unsigned char * buffer, size_t size);

/* Private functions ---------------------------------------------------------*/
#define MOUSE_DELTA           4
extern void USBDev_IRQHandler(void);

void USB_Mouse(void)
{
	Int8S X = 0, Y = 0;
	Int8U Buttons = 0, ButtonsHold = 0;

	ENTR_CRT_SECTION();
	// SysTick end of count event each 0.01s
	/*SysTick to times per second*/
	SysTickStart(100);

	// HID USB
	HidInit();
	EXT_CRT_SECTION();

	// Soft connection enable
	USB_ConnectRes(TRUE);
	/*Init Buttons*/
	STM_EVAL_PBInit(BUTTON_WAKEUP, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_TAMPER, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_RIGHT, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_LEFT, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_UP, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_DOWN, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_SEL, BUTTON_MODE_GPIO);

	while('\x1B' !=  getchar())
	{
		if( SysTick->CTRL & (1<<16))
		{
			if(UsbCoreReq(UsbCoreReqDevState) == UsbDevStatusConfigured)
			{
				/**/
				if(Bit_SET == STM_EVAL_PBGetState(BUTTON_RIGHT))
				{
					X += MOUSE_DELTA;
				}
				else if(Bit_SET == STM_EVAL_PBGetState(BUTTON_LEFT))
				{
					X -= MOUSE_DELTA;
				}

				if(Bit_SET == STM_EVAL_PBGetState(BUTTON_DOWN))
				{
					Y += MOUSE_DELTA;
				}
				else if(Bit_SET == STM_EVAL_PBGetState(BUTTON_UP))
				{
					Y -= MOUSE_DELTA;
				}

				if((Bit_SET == STM_EVAL_PBGetState(BUTTON_SEL)) || (Bit_RESET == STM_EVAL_PBGetState(BUTTON_TAMPER)))
				{
					Buttons |= 1;
				}
				else
				{
					Buttons &= ~1;
				}

				if(Bit_SET == STM_EVAL_PBGetState(BUTTON_WAKEUP))
				{
					Buttons |= 2;
				}
				else
				{
					Buttons &=~2;
				}

				if(Y || X  || Buttons || (ButtonsHold ^ Buttons))
				{
					// Send report
					HidMouseSendReport(X,Y,Buttons);
					ButtonsHold = Buttons;
					Y = X = Buttons = 0;
				}
			}
		}
	}

	USB_ConnectRes(FALSE);

	//SysTickStop();

}

int USBOTG_Test(void)
{
	printf("Connect USB OTG port to PC\n\r");
	printf("Press Esc for exit\n\r");

	usb_module_select = USB_OTG_HS;

	USB_Mouse();

	return 0;
}

int USBHOST_Test(void)
{

	printf("Connect USB HOST port to PC \n\r");
	printf("Press Esc for exit\n\r");

	usb_module_select = USB_OTG_FS;

	USB_Mouse();

	return 0;
}

/**
* @brief  Main program.
* @param  None
* @retval None
*/
int main(void)
{
	/*!< At this stage the microcontroller clock setting is already configured,
this is done through SystemInit() function which is called from startup
file (startup_stm32f2xx.s) before to branch to application main.
To reconfigure the default setting of SystemInit() function, refer to
system_stm32f2xx.c file
*/
	/* Setup STM32 system (clock, PLL and Flash configuration) */
	SystemInit();

	/* USARTx configured as follow:
	- BaudRate = 115200 baud
	- Word Length = 8 Bits
	- One Stop Bit
	- No parity
	- Hardware flow control disabled (RTS and CTS signals)
	- Receive and transmit enabled
*/
	USART_InitStructure.USART_BaudRate = 115200;
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;

	STM_EVAL_COMInit(COM1, &USART_InitStructure);

	printf("\n\r");
	printf("**************************************************\n\r");
	printf("            OLIMEX STM32-P207\n\r");
	printf("**************************************************\n\r");

	printf("\n\r");
	printf("**************************************************\n\r");
	printf("Test Started\n\r");

	USBOTG_Test();
	
	STM_EVAL_COMInit(COM1, &USART_InitStructure);

	printf("  TEST END\n\r");
	printf("**************************************************\n\r");
}


/*************************************************************************
* Function Name: DelayResolution100us
* Parameters: Int32U Dly
*
* Return: none
*
* Description: Delay ~ (arg * 100us)
*
*************************************************************************/
void DelayResolution100us(Int32U Dly)
{
	for(; Dly; Dly--)
	{
		for(volatile Int32U j = DLY_100US; j; j--)
		{
		}
	}
}

#ifdef  USE_FULL_ASSERT

/**
* @brief  Reports the name of the source file and the source line number
*   where the assert_param error has occurred.
* @param  file: pointer to the source file name
* @param  line: assert_param error line source number
* @retval None
*/
void assert_failed(uint8_t* file, uint32_t line)
{
	/* User can add his own implementation to report the file name and line number,
ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

	/* Infinite loop */
	while (1)
	{
	}
}
#endif

/**
* @}
*/

/**
* @}
*/

/*************************************************************************
* Function Name: __write
* Parameters: Low Level cahracter output
*
* Return:
*
* Description:
*
*************************************************************************/
size_t __write(int Handle, const unsigned char * Buf, size_t Bufsize)
{
	size_t nChars = 0;

	for (/*Empty */; Bufsize > 0; --Bufsize)
	{
		/* Loop until the end of transmission */
		while (USART_GetFlagStatus(EVAL_COM1, USART_FLAG_TXE) == RESET);
		USART_SendData(EVAL_COM1, * Buf++);
		++nChars;
	}
	return nChars;
}
/*************************************************************************
* Function Name: __read
* Parameters: Low Level cahracter input
*
* Return:
*
* Description:
*
*************************************************************************/
size_t __read(int handle, unsigned char * buffer, size_t size)
{
	int nChars = 0;

	/* This template only reads from "standard in", for all other file
* handles it returns failure. */
	if (handle != _LLIO_STDIN)
	{
		return _LLIO_ERROR;
	}

	for (/* Empty */; size > 0; --size)
	{
		int c = MyLowLevelGetchar();
		if (c < 0)
		break;

		*buffer++ = c;
		++nChars;
	}

	return nChars;
}

static int MyLowLevelGetchar(void)
{
	int ch;
	unsigned int status = EVAL_COM1->SR;

	if(status & USART_FLAG_RXNE)
	{
		ch = USART_ReceiveData(EVAL_COM1);
		if(status & (USART_FLAG_ORE | USART_FLAG_PE | USART_FLAG_FE) )
		{
			return (ch | 0x10000000);
		}
		return (ch & 0xff );
	}
	return -1;
}
/******************* (C) COPYRIGHT 2010 STMicroelectronics *****END OF FILE****/
