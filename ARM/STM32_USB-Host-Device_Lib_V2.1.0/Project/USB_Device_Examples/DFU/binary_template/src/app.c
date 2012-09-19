/**
  ******************************************************************************
  * @file    app.c
  * @author  MCD Application Team
  * @version V1.1.0
  * @date    19-March-2012
  * @brief   This file provides all the Application firmware functions.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT 2012 STMicroelectronics</center></h2>
  *
  * Licensed under MCD-ST Liberty SW License Agreement V2, (the "License");
  * You may not use this file except in compliance with the License.
  * You may obtain a copy of the License at:
  *
  *        http://www.st.com/software_license_agreement_liberty_v2
  *
  * Unless required by applicable law or agreed to in writing, software 
  * distributed under the License is distributed on an "AS IS" BASIS, 
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  *
  ******************************************************************************
  */ 

/* Includes ------------------------------------------------------------------*/ 
#include  "app.h"
#include  "lcd_log.h"

/** @addtogroup STM32_USB_OTG_DEVICE_LIBRARY
  * @{
  */


/** @defgroup APP_DFU 
  * @brief Mass storage application module
  * @{
  */ 

/** @defgroup APP_DFU_Private_TypesDefinitions
  * @{
  */ 

/**
  * @}
  */ 


/** @defgroup APP_DFU_Private_Defines
  * @{
  */ 

/**
  * @}
  */ 


/** @defgroup APP_DFU_Private_Macros
  * @{
  */ 
/**
  * @}
  */ 


/** @defgroup APP_DFU_Private_Variables
  * @{
  */ 
static __IO uint32_t TimingDelay;

/**
  * @}
  */ 


/** @defgroup APP_DFU_Private_FunctionPrototypes
  * @{
  */ 
void Delay(__IO uint32_t nTime);
/**
  * @}
  */ 


/** @defgroup APP_DFU_Private_Functions
  * @{
  */ 

/**
  * @brief  Program entry point
  * @param  None
  * @retval None
*/
int main(void)
{
  /*!< At this stage the microcontroller clock setting is already configured, 
  this is done through SystemInit() function which is called from startup
  file (startup_stm32fxxx_xx.s) before to branch to application main.
  To reconfigure the default setting of SystemInit() function, refer to
  system_stm32fxxx.c file
  */  

  /* Initialize the LCD */
#if defined (USE_STM322xG_EVAL)
  STM322xG_LCD_Init();
#elif defined(USE_STM324xG_EVAL)
  STM324xG_LCD_Init();
#elif defined (USE_STM3210C_EVAL)
  STM3210C_LCD_Init();
#else
 #error "Missing define: Evaluation board (ie. USE_STM322xG_EVAL)"
#endif

  LCD_LOG_Init();
  
  LCD_LOG_SetHeader(" Binary image template ");
  LCD_UsrLog("> Systick template example started.\n"); 
  LCD_LOG_SetFooter ("     Binary image template " ); 
  
  /* Configure the LEDs */
  STM_EVAL_LEDInit(LED1);
  STM_EVAL_LEDInit(LED2);
  STM_EVAL_LEDInit(LED3);
  STM_EVAL_LEDInit(LED4);  

   /* Set the Vector Table base location at the application start address
      (this is already done in system_stm32fxxx.c file) */   
//#ifdef STM32F2XX
//  /* Set the Vector Table base location at 0xC000 */ 
//  NVIC_SetVectorTable(NVIC_VectTab_FLASH, 0xC000);
//#elif defined(STM32F4XX)
//  /* Set the Vector Table base location at 0xC000 */ 
//  NVIC_SetVectorTable(NVIC_VectTab_FLASH, 0xC000);
//#elif defined(STM32F10X_CL)
//  /* Set the Vector Table base location at 0x8000 */ 
//  NVIC_SetVectorTable(NVIC_VectTab_FLASH, 0x8000);
//#endif /* STM32F2XX */
  
  /* Setup SysTick Timer for 1 msec interrupts  */
  if (SysTick_Config(SystemCoreClock / 1000))
  { 
    /* Capture error */ 
    while (1);
  }
  
  while (1)
  {
    /* Toggle all leds */
    STM_EVAL_LEDToggle(LED1);
    STM_EVAL_LEDToggle(LED2);
    STM_EVAL_LEDToggle(LED3);
    STM_EVAL_LEDToggle(LED4);
    
    /* Insert 200 ms delay */
    Delay(200);
    
    /* Toggle all leds */
    STM_EVAL_LEDToggle(LED1);
    STM_EVAL_LEDToggle(LED2);
    STM_EVAL_LEDToggle(LED3);
    STM_EVAL_LEDToggle(LED4);
    
    /* Insert 200 ms delay */
    Delay(200);
  }
} 

/**
  * @brief  Inserts a delay time.
  * @param  nTime: specifies the delay time length, in milliseconds.
  * @retval None
*/
void Delay(uint32_t nTime)
{
  TimingDelay = nTime;

  while(TimingDelay != 0);
}

/**
  * @brief  Decrements the TimingDelay variable.
  * @param  nTime: specifies the delay time length, in milliseconds.
  * @retval None
*/
void TimingDelay_Decrement(void)
{
  if (TimingDelay != 0x00)
  { 
    TimingDelay--;
  }
}

#ifdef USE_FULL_ASSERT
/**
* @brief  assert_failed
*         Reports the name of the source file and the source line number
*         where the assert_param error has occurred.
* @param  File: pointer to the source file name
* @param  Line: assert_param error line source number
* @retval None
*/
void assert_failed(uint8_t* file, uint32_t line)
{
  /* User can add his own implementation to report the file name and line number,
  ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  
  /* Infinite loop */
  while (1)
  {}
}
#endif

/**
  * @}
  */ 


/**
  * @}
  */ 


/**
  * @}
  */ 

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
