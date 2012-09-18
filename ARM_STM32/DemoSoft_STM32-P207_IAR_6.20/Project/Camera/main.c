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
int LCD_Test(void)
{
	int cntr = 0;
	char Backlight = 80;
	char BacklightStep = 1;
	char cameraInitSuccessFlag = 0;

	printf("Press Esc for exit\n\r");

	STM_EVAL_LEDInit(LED1);
	STM_EVAL_LEDInit(LED2);
	STM_EVAL_LEDInit(LED3);
//	STM_EVAL_LEDInit(LED4); multiplexed with camera power pin
	
	STM_EVAL_PBInit(BUTTON_WAKEUP, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_UP, BUTTON_MODE_GPIO);
	STM_EVAL_PBInit(BUTTON_DOWN, BUTTON_MODE_GPIO);
	
	pInt8U pOutputData = (pInt8U)(EXT_SRAM_BASE_ADDRESS + CAMERA_HOR_RES * CAMERA_VER_RES * 2);
	pInt8U pInData;

	if( ExtSRAM_Initialize() )
	{
		printf("SRAM Init fault\n\r");
		return 1;
	} else {
		printf("SRAM Testing...");
		switch( ExtSRAM_Test() ) {
		case 0:
			printf("OK\n\r");		
			break;
		case -1:
			printf("SRAM address bus error!\n\r");
			return 1;
		case -2:
			printf("SRAM data bus error!\n\r");
			return 1;
		default:
			printf("SRAM unknown error!\n\r");
			return 1;
		}
	}

	// initialize display
	GLCD_PowerUpInit(NULL);
	GLCD_Backlight(BACKLIGHT_ON);
	
	if( E700_Camera_Initialize() ) {
		E700_Camera_Deinitialize();
		printf("Camera init error(maybe I2C line is bugged)!\n\r");

		GLCD_SetFont(&Terminal_9_12_6,0x000F00,0x00FF0);
		GLCD_SetWindow(10,116,131,131);
		GLCD_TextSetPos(0,0);
		GLCD_print("\fNO CAMERA!!\r");
	} else {
		cameraInitSuccessFlag = 1;
	}
	
	
	/*Systick start*/
	SysTickStart(10);

	while(Bit_SET != STM_EVAL_PBGetState(BUTTON_WAKEUP))
	{

		if( SysTick->CTRL & (1<<16)) {
			STM_EVAL_LEDOff(LED1);
			STM_EVAL_LEDOff(LED2);
			STM_EVAL_LEDOff(LED3);
			//      STM_EVAL_LEDOff(LED4);
			STM_EVAL_LEDOn((Led_TypeDef)((cntr++)&0x3));
	
			// update display with a new photo
			if(cameraInitSuccessFlag) {
				pInData = E700_Camera_CaptureImage();
				E700_Camera_ProcessImage(pInData, pOutputData, CAMERA_HOR_RES, CAMERA_VER_RES);
				GLCD_UpdateMemory(pOutputData);	
			}
	
			// adjust backlight level
			if(Bit_SET == STM_EVAL_PBGetState(BUTTON_UP))
			{
				if(Backlight < 96) {
					Backlight += BacklightStep;
					GLCD_Backlight(Backlight);
				}
			}
			else if(Bit_SET == STM_EVAL_PBGetState(BUTTON_DOWN))
			{
				if(Backlight > 64) {
					Backlight -= BacklightStep;
					GLCD_Backlight(Backlight);
				}
			}
		}
	}
	
	//SysTickStop();

	/**/
	printf("                                              \r");

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


	SysTickStop();
	
	printf("\n\r");
	printf("**************************************************\n\r");
	printf("Test Started\n\r");
	
	LCD_Test();

	ExtSRAM_Deinitialize();
	E700_Camera_Deinitialize();

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
