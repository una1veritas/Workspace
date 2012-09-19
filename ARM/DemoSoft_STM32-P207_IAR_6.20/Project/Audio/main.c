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
#define PI (float)3.14159
int Audio_Test()
{
	I2S_InitTypeDef I2S_InitStructure;
	GPIO_InitTypeDef GPIO_InitStructure;

	int cntr = 0;
	int index = 0;
	int data;
	int volume_r, volume_l;

	/*PLLI2S configure*/
	RCC_PLLI2SConfig(271,2);
	/*Enable PLLI2S*/
	RCC_PLLI2SCmd(ENABLE);
	/*Wait PLLI2S Lock*/
	while(RESET == RCC_GetFlagStatus(RCC_FLAG_PLLI2SRDY));
	/*PLLI2S is I2S clock source*/
	RCC_I2SCLKConfig(RCC_I2S2CLKSource_PLLI2S);
	/* Enable I2S3 APB1 clock */
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_SPI3, ENABLE);
	/* Deinitialize SPI2 peripheral */
	SPI_I2S_DeInit(SPI3);

	/* I2S2 peripheral configuration */
	I2S_InitStructure.I2S_Mode = I2S_Mode_MasterTx;
	I2S_InitStructure.I2S_Standard = I2S_Standard_Phillips;
	I2S_InitStructure.I2S_DataFormat = I2S_DataFormat_24b;
	I2S_InitStructure.I2S_MCLKOutput = I2S_MCLKOutput_Enable;
	I2S_InitStructure.I2S_AudioFreq = I2S_AudioFreq_44k;
	I2S_InitStructure.I2S_CPOL = I2S_CPOL_Low;
	I2S_Init(SPI3, &I2S_InitStructure);

	/* Disable the I2S2 TXE Interrupt */
	SPI_I2S_ITConfig(SPI3, SPI_I2S_IT_TXE, DISABLE);

	/*Pin Configure*/
	/* Enable GPIOs clocks */
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA |RCC_AHB1Periph_GPIOB | RCC_AHB1Periph_GPIOC, ENABLE);

	/* Configure PA15*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_15;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	/* Configure PA15*/
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource15, GPIO_AF_SPI3);

	/* Configure PB3 and PB5*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3 | GPIO_Pin_5;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	/* Connect PB3 and PB5 I2S3 module*/
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource3, GPIO_AF_SPI3);
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource5, GPIO_AF_SPI3);
	/* Configure PC7 */
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOC, &GPIO_InitStructure);

	/* Connect PC7 I2S3 module*/
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource7, GPIO_AF_SPI3);
	/* Enable the SPI2/I2S2 peripheral */
	I2S_Cmd(SPI3, ENABLE);
	/*Systick start*/
	SysTickStart(20);

	while('\x1B' != getchar())
	{
		if(SysTick->CTRL & (1<<16))
		{
			cntr++;
			if(20 > cntr)
			{
				volume_r = ((1<<13)-1);
				volume_l = 0;
			}
			else
			{
				if(25 > cntr)
				{
					volume_l = 0;
					volume_r = 0;
				}
				else
				{
					if(45 > cntr)
					{
						volume_l = ((1<<13)-1);
						volume_r = 0;
					}
					else
					{
						if(50 > cntr)
						{
							volume_l = 0;
							volume_r = 0;
						}
						else cntr = 0;
					}
				}

			}
		}
		index+= SAMPLES_NUM * 1000/I2S_AudioFreq_44k;
		data = Sin_Table[(SAMPLES_NUM - 1) & index]*volume_l;
		while(RESET == SPI_I2S_GetFlagStatus(SPI3,SPI_I2S_FLAG_TXE));
		SPI_I2S_SendData(SPI3,(data>>8) & 0xFFFF);
		while(RESET == SPI_I2S_GetFlagStatus(SPI3,SPI_I2S_FLAG_TXE));
		SPI_I2S_SendData(SPI3,(data<<8) & 0xFFFF);
		data = Sin_Table[(SAMPLES_NUM - 1) & (index+SAMPLES_NUM/4)]*volume_r;
		while(RESET == SPI_I2S_GetFlagStatus(SPI3,SPI_I2S_FLAG_TXE));
		SPI_I2S_SendData(SPI3,(data>>8) & 0xFFFF);
		while(RESET == SPI_I2S_GetFlagStatus(SPI3,SPI_I2S_FLAG_TXE));
		SPI_I2S_SendData(SPI3,(data<<8) & 0xFFFF);
	}

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
	
	Audio_Test();

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
