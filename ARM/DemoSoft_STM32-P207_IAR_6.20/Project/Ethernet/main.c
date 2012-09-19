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
int Ethernet_Test()
{
	GPIO_InitTypeDef GPIO_InitStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	ETH_InitTypeDef ETH_InitStructure;

	printf("connect LAN cable and press Enter\n\r");

	while('\r' != getchar());

	printf("Press Esc for exit\n\r");


	/* Enable ETHERNET clocks  */
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_ETH_MAC | RCC_AHB1Periph_ETH_MAC_Tx |
	RCC_AHB1Periph_ETH_MAC_Rx | RCC_AHB1Periph_ETH_MAC_PTP, ENABLE);


	/* Enable GPIOs clocks */
	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOA |	RCC_AHB1Periph_GPIOB |
	RCC_AHB1Periph_GPIOC | RCC_AHB1Periph_GPIOG, ENABLE);

	/* Enable SYSCFG clock */
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_SYSCFG, ENABLE);
	/*Select RMII Interface*/
	SYSCFG_ETH_MediaInterfaceConfig(SYSCFG_ETH_MediaInterface_RMII);

	/* ETHERNET pins configuration */
	/* PA
ETH_RMII_REF_CLK: PA1
ETH_RMII_MDIO: PA2
ETH_RMII_MDINT: PA3
ETH_RMII_CRS_DV: PA7
*/

	/* Configure PA1, PA2, PA3 and PA7*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1 | GPIO_Pin_2 | GPIO_Pin_3 | GPIO_Pin_7;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	/* Connect PA1, PA2, PA3 and PA7 to ethernet module*/
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource1, GPIO_AF_ETH);
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource2, GPIO_AF_ETH);
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource3, GPIO_AF_ETH);
	GPIO_PinAFConfig(GPIOA, GPIO_PinSource7, GPIO_AF_ETH);

	/* PB
ETH_RMII_TX_EN: PB11
*/

	/* Configure PB11*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);

	/* Connect PB11 to ethernet module*/
	GPIO_PinAFConfig(GPIOB, GPIO_PinSource11, GPIO_AF_ETH);

	/* PC
ETH_RMII_MDC: PC1
ETH_RMII_RXD0: PC4
ETH_RMII_RXD1: PC5
*/

	/* Configure PC1, PC4 and PC5*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1 | GPIO_Pin_4 | GPIO_Pin_5;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOC, &GPIO_InitStructure);

	/* Connect PC1, PC4 and PC5 to ethernet module*/
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource1, GPIO_AF_ETH);
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource4, GPIO_AF_ETH);
	GPIO_PinAFConfig(GPIOC, GPIO_PinSource5, GPIO_AF_ETH);

	/* PG
ETH_RMII_TXD0: PG13
ETH_RMII_TXD1: PG14
*/

	/* Configure PG14 and PG15*/
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13 | GPIO_Pin_14;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOG, &GPIO_InitStructure);

	/* Connect PG13 and PG15 to ethernet module*/
	GPIO_PinAFConfig(GPIOG, GPIO_PinSource13, GPIO_AF_ETH);
	GPIO_PinAFConfig(GPIOG, GPIO_PinSource14, GPIO_AF_ETH);

	/* Reset ETHERNET on AHB Bus */
	ETH_DeInit();

	/* Software reset */
	ETH_SoftwareReset();

	/* Wait for software reset */
	while(ETH_GetSoftwareResetStatus()==SET);

	/* ETHERNET Configuration ------------------------------------------------------*/
	/* Call ETH_StructInit if you don't like to configure all ETH_InitStructure parameter */
	ETH_StructInit(&ETH_InitStructure);

	/* Fill ETH_InitStructure parametrs */
	/*------------------------   MAC   -----------------------------------*/
	ETH_InitStructure.ETH_AutoNegotiation = ETH_AutoNegotiation_Disable  ;
	//ETH_InitStructure.ETH_Speed = ETH_Speed_100M;
	ETH_InitStructure.ETH_LoopbackMode = ETH_LoopbackMode_Disable;
	//ETH_InitStructure.ETH_Mode = ETH_Mode_FullDuplex;
	ETH_InitStructure.ETH_RetryTransmission = ETH_RetryTransmission_Disable;
	ETH_InitStructure.ETH_AutomaticPadCRCStrip = ETH_AutomaticPadCRCStrip_Disable;
	ETH_InitStructure.ETH_ReceiveAll = ETH_ReceiveAll_Enable;
	ETH_InitStructure.ETH_BroadcastFramesReception = ETH_BroadcastFramesReception_Disable;
	ETH_InitStructure.ETH_PromiscuousMode = ETH_PromiscuousMode_Disable;
	ETH_InitStructure.ETH_MulticastFramesFilter = ETH_MulticastFramesFilter_Perfect;
	ETH_InitStructure.ETH_UnicastFramesFilter = ETH_UnicastFramesFilter_Perfect;
	ETH_InitStructure.ETH_Mode = ETH_Mode_FullDuplex;
	ETH_InitStructure.ETH_Speed = ETH_Speed_100M;

	unsigned int PhyAddr;

	// read the ID for match
	for(PhyAddr = 1; 32 >= PhyAddr; PhyAddr++)
	{
		if((0x0022 == ETH_ReadPHYRegister(PhyAddr,2))
				&& (0x1619 == (ETH_ReadPHYRegister(PhyAddr,3)))) break;
	}

	if(32 < PhyAddr)
	{
		printf("Ethernet Phy Not Found\n\r");
		return 1;
	}
	/* Configure Ethernet */
	if(0 == ETH_Init(&ETH_InitStructure, PhyAddr))
	{
		printf("Ethernet Initialization Failed\n\r");
		return 1;
	}

	printf("Check LAN LEDs\n\r");

	/* uIP stack main loop */
	uIPMain();

	TIM_DeInit(TIM2);

	/* Enable the TIM2 Interrupt */
	NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelCmd = DISABLE;
	NVIC_Init(&NVIC_InitStructure);
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
	
	Ethernet_Test();
		
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
