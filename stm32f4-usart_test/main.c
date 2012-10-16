/**
 * Modified by Wolfgang Wieser to enable the FPU and compute the Mandelbrot
 * set on the FPU as well as a sqrtf() benchmark.
 *
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
//#include "stm32f4_discovery.h"
#include "stm32f4xx_gpio.h"
#include "stm32f4xx_rcc.h"

#include "favorites.h"
#include "gpio_digital.h"
#include "systick.h"
#include "usart.h"

int main(void) {

	SysTick_Init(3360);

	GPIOMode(GPIOD, GPIO_Pin_12 | GPIO_Pin_13 | GPIO_Pin_14 | GPIO_Pin_15,
			OUTPUT, FASTSPEED, PUSHPULL, NOPULL);

	usart_begin(19200);
	usart_print("System core clock: \n\n");
	usart_printNumber(SystemCoreClock);
	usart_print(".\n\n");

	// NOTE: Important: Enable full access to FPU:
	SCB ->CPACR |= ((3UL << 10 * 2) | (3UL << 11 * 2)); /* set CP10 and CP11 Full Access */

	uint8 ledstate = RESET;
	uint32 ticks = 0;

	while (1) {
		digitalWrite(PD14, SET);
		delay(1000);
		digitalWrite(PD14, RESET);
		delay(1000);
		usart_printNumber(millis());
		usart_print(".\n\n");
	}
}

/******************* (C) COPYRIGHT 2011 STMicroelectronics *****END OF FILE****/
