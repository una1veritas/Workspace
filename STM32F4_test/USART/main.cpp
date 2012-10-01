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

#include "stm32f4xx.h"

#include "gpio_digital.h"
#include "stm32f4xx_it.h"
#include "delay.h"

#include "Olimex_stm32f217ze_sk.h"

//#include <yfuns.h>
//#include "includes.h"

#define DLY_100US  1000

USART_InitTypeDef USART_InitStructure;
//static __IO uint32_t TimingDelay;
RCC_ClocksTypeDef RCC_Clocks;

static int MyLowLevelGetchar(void);
size_t __write(int Handle, const unsigned char * Buf, size_t Bufsize);
size_t __read(int handle, unsigned char * buffer, size_t size);

/* Private functions ---------------------------------------------------------*/
int Button_LED_Demo (void) {
        int x=4;
        pinMode(digitalPin(LED1_GPIO_PORT, LED1_PIN), OUTPUT);//   STM_EVAL_LEDInit (LED1);
        pinMode(digitalPin(WAKEUP_BUTTON_GPIO_PORT, WAKEUP_BUTTON_PIN), INPUT); // this button is pulled down.
        //  STM_EVAL_PBInit (BUTTON_WAKEUP, BUTTON_MODE_GPIO);
        printf ("Demo LED: \n\r");
        printf ("LED should be blinking 3 times and then turns off!\n\r");

        while (x--) {
          STM_EVAL_LEDOn (LED1);
          DelayResolution100us (1000);
          STM_EVAL_LEDOff (LED1);
          DelayResolution100us (1000);
        }
        printf ("Demo Button:\n\r");
        printf ("Press the button to blink the LED again\n\r");

        while (!STM_EVAL_PBGetState (BUTTON_WAKEUP));
        while (STM_EVAL_PBGetState (BUTTON_WAKEUP))
        {
            STM_EVAL_LEDOn (LED1);
            DelayResolution100us (250);
            STM_EVAL_LEDOff (LED1);
            DelayResolution100us (750);
        }
        STM_EVAL_LEDOn (LED1);
        return 0;
}

int RTC_Demo(void) {
RTC_TimeTypeDef time = {
		0,
		0,
		0,
		0
};

  /*SysTick to times per second*/
  SysTickStart(1);

  /* Enable LSE */
  RCC_APB1PeriphClockCmd(  RCC_APB1Periph_PWR, ENABLE);
  PWR_BackupAccessCmd(ENABLE);
  /* */
  RCC_LSEConfig(RCC_LSE_ON);
  /* Wait till LSE is ready */
  int Tick = 0;

  while (RCC_GetFlagStatus(RCC_FLAG_LSERDY) == RESET)
  {
    /*Check Ticks*/
    if( SysTick->CTRL & (1<<16))
    {
      if(6 < ++Tick)
      {
        return 1;
      }
    }
  }
  /* Select LSE as RTC Clock Source */
  RCC_RTCCLKConfig(RCC_RTCCLKSource_LSE);

  /* Enable RTC Clock */
  RCC_RTCCLKCmd(ENABLE);

  /* Wait for RTC registers synchronization */
  //RTC_WaitForSynchro();

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForSynchro();

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForSynchro();

  /* Set RTC prescaler: set RTC period to 1sec */
 // RTC_SetPrescaler(32767); /* RTC period = RTCCLK/RTC_PR = (32.768 KHz)/(32767+1) */

  /* Wait until last write operation on RTC registers has finished */
  RTC_WaitForSynchro();
  /*Set time*/

  printf ("Please wait...");
  DelayResolution100us (30);
  printf ("RTC intialized!\n\r");

  RTC_SetTime(RTC_Format_BIN,&time);
  /*Convert RTC to hour min and sec*/
  RTC_GetTime(RTC_Format_BIN,&time);

//  printf("Time %.2d:%.2d:%.2d\r",time.RTC_Hours,time.RTC_Minutes,time.RTC_Seconds);
  printf ("Press space to exit this demo!\n\r");

  Tick = 0;

  while(' ' !=  getchar())
  {
    /*Check Ticks*/
    if( SysTick->CTRL & (1<<16))
    {
      RTC_GetTime(RTC_Format_BIN,&time);
      printf("Time %.2d:%.2d:%.2d\r",time.RTC_Hours,time.RTC_Minutes,time.RTC_Seconds);
    }
  }

  printf("Time %.2d:%.2d:%.2d\n\r",time.RTC_Hours,time.RTC_Minutes,time.RTC_Seconds);
  //SysTickStop();

  return 0;
}

#define UPDATE_SHOW_DLY       ((Int32U)(0.5 * 2))


void My_GPIOReset (void)
{
    assert_param(IS_GPIO_ALL_PERIPH(GPIOx));

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOA, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOA, DISABLE);

    //RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOB, ENABLE);
    //RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOB, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOC, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOC, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOD, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOD, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOE, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOE, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOF, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOF, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOG, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOG, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOH, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOH, DISABLE);

    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOI, ENABLE);
    RCC_AHB1PeriphResetCmd(RCC_AHB1Periph_GPIOI, DISABLE);
}

void Blink (int x)
{
        STM_EVAL_LEDInit (LED1);
        while (x--)
        {
          STM_EVAL_LEDOn (LED1);
          DelayResolution100us (2000);
          STM_EVAL_LEDOff (LED1);
          DelayResolution100us (2000);
        }
}

/**
  * @brief  Main program.
  * @param  None
  * @retval None
  */
void main(void)
{
  char c, Flag=0;
  /*!< At this stage the microcontroller clock setting is already configured,
       this is done through SystemInit() function which is called from startup
       file (startup_stm32f2xx.s) before to branch to application main.
       To reconfigure the default setting of SystemInit() function, refer to
       system_stm32f2xx.c file
     */
  /* Setup STM32 system (clock, PLL and Flash configuration) */
  SystemInit();
  STM_EVAL_PBInit (BUTTON_WAKEUP, BUTTON_MODE_GPIO);

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
  printf("            Olimex-STM32E407\n\r");
  printf("             Demo  MODE!!\n\r");
  printf("**************************************************\n\r");


  printf("Demos are finished! The green LED will blink now.\n\r");
  // End of the demo
  while (1)
  {
    Blink (1);
  }
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
void DelayResolution100us(uint32_t Dly)
{
  for(; Dly; Dly--)
  {
    for(volatile uint32_t j = DLY_100US; j; j--)
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
