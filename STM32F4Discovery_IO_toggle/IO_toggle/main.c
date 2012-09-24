/**
 ******************************************************************************
 * @file    IO_Toggle/main.c
 * @author  MCD Application Team
 * @version V1.0.0
 * @date    19-September-2011
 * @brief   Main program body
 ******************************************************************************
 * @attention
 *
 * THE PRESENT FIRMWARE WHICH IS FOR GUIDANCE ONLY AIMS AT PROVIDING CUSTOMERS
 * WITH CODING INFORMATION REGARDING THEIR PRODUCTS IN ORDER FOR THEM TO SAVE
 * TIME. AS A RESULT, STMICROELECTRONICS SHALL NOT BE HELD LIABLE FOR ANY
 * DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES WITH RESPECT TO ANY CLAIMS ARISING
 * FROM THE CONTENT OF SUCH FIRMWARE AND/OR THE USE MADE BY CUSTOMERS OF THE
 * CODING INFORMATION CONTAINED HEREIN IN CONNECTION WITH THEIR PRODUCTS.
 *
 * <h2><center>&copy; COPYRIGHT 2011 STMicroelectronics</center></h2>
 ******************************************************************************
 */
/* Includes ------------------------------------------------------------------*/
#include "stm32f4_discovery.h"
#include "stm32f4xx_conf.h"

GPIO_InitTypeDef GPIO_InitStructure;

//void Delay(__IO uint32_t nCount);
static __IO uint32_t TimingDelay;
void TimingDelay_Decrement(void);
void Delay(__IO uint32_t nTime);

int main(void) {

	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOD, ENABLE);

	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14
			| GPIO_Pin_15;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_OUT;
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_100MHz;
	GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_NOPULL;
	GPIO_Init(GPIOD, &GPIO_InitStructure);

	  /* Setup SysTick Timer for 1 msec interrupts.
	     ------------------------------------------
	    1. The SysTick_Config() function is a CMSIS function which configure:
	       - The SysTick Reload register with value passed as function parameter.
	       - Configure the SysTick IRQ priority to the lowest value (0x0F).
	       - Reset the SysTick Counter register.
	       - Configure the SysTick Counter clock source to be Core Clock Source (HCLK).
	       - Enable the SysTick Interrupt.
	       - Start the SysTick Counter.

	    2. You can change the SysTick Clock source to be HCLK_Div8 by calling the
	       SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8) just after the
	       SysTick_Config() function call. The SysTick_CLKSourceConfig() is defined
	       inside the misc.c file.

	    3. You can change the SysTick IRQ priority by calling the
	       NVIC_SetPriority(SysTick_IRQn,...) just after the SysTick_Config() function
	       call. The NVIC_SetPriority() is defined inside the core_cm4.h file.

	    4. To adjust the SysTick time base, use the following formula:

	         Reload Value = SysTick Counter Clock (Hz) x  Desired Time base (s)

	       - Reload Value is the parameter to be passed for SysTick_Config() function
	       - Reload Value should not exceed 0xFFFFFF
	   */

//	RCC_ClocksTypeDef RCC_Clocks;
//	RCC_GetClocksFreq(&RCC_Clocks);
	  if (SysTick_Config(SystemCoreClock / 1000))
	  {
	    /* Capture error */
	    while (1);
	  }

	while (1) {
		/* PD12 to be toggled */
		GPIO_SetBits(GPIOD, GPIO_Pin_12 );

		/* Insert delay */
		Delay(0x3FFFFF);

		/* PD13 to be toggled */
		GPIO_SetBits(GPIOD, GPIO_Pin_13 );

		/* Insert delay */
		Delay(0x3FFFFF);

		/* PD14 to be toggled */
		GPIO_SetBits(GPIOD, GPIO_Pin_14 );

		/* Insert delay */
		Delay(0x3FFFFF);

		/* PD15 to be toggled */
		GPIO_SetBits(GPIOD, GPIO_Pin_15 );

		/* Insert delay */
		Delay(0x7FFFFF);

		GPIO_ResetBits(GPIOD,
				GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15 );

		/* Insert delay */
		Delay(0xFFFFFF);
	}
}

/**
void Delay(__IO uint32_t nCount) {
	while (nCount--)
		;
}
 */

void Delay(__IO uint32_t nTime)
{
  TimingDelay = nTime;

  while(TimingDelay != 0);
}

/**
  * @brief  Decrements the TimingDelay variable.
  * @param  None
  * @retval None
  */
void TimingDelay_Decrement(void)
{
  if (TimingDelay != 0x00)
  {
    TimingDelay--;
  }
}

