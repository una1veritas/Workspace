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
//#include "mandelbrot.h"

#include "gpio_digital.h"
#include "systick.h"
#include "usart.h"

GPIO_InitTypeDef  GPIO_InitStructure;


#define nop()       __asm__ __volatile__("nop")


int main(void) {

	SysTick_Start();

	GPIOMode(GPIOD, GPIO_Pin_12 | GPIO_Pin_13| GPIO_Pin_14| GPIO_Pin_15,
			GPIO_Mode_OUT, GPIO_Speed_50MHz, GPIO_OType_PP, GPIO_PuPd_NOPULL);

	usart_begin(19200);
	usart_print("Hi.\n");

  // NOTE: Important: Enable full access to FPU: 
  SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2));  /* set CP10 and CP11 Full Access */

  while (1)
  {
		digitalWrite(PD12, LOW);
		digitalWrite(PD13, LOW);
		digitalWrite(PD14, LOW);
		digitalWrite(PD15, LOW);
		switch (millis() % 16) {
		case 0:
		case 4:
		case 11:
		case 15:
			digitalWrite(PD12, HIGH);
			break;
		case 1:
		case 5:
		case 10:
		case 14:
			digitalWrite(PD13, HIGH);
			break;
		case 2:
		case 6:
		case 9:
		case 13:
			digitalWrite(PD14, HIGH);
			break;
		case 3:
		case 7:
		case 8:
		case 12:
			digitalWrite(PD15, HIGH);
			break;
		}
		usart_print("passed ");
		usart_printFloat(millis()/1000.0f, 2);
		usart_print(".\n");
		delay(125);
	}
}


#ifdef  USE_FULL_ASSERT

/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
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

/******************* (C) COPYRIGHT 2011 STMicroelectronics *****END OF FILE****/
