/*
 * delay5.c
 *
 *  Created on: 2012/11/15
 *      Author: sin
 */


/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_rcc.h"
#include "stm32f4xx_tim.h"
#include "delay5.h"
//#include "platform_config.h"
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
//#define TIM_NUM  TIM5
//#define TIM_RCC  RCC_APB1Periph_TIM5 //TIM5_RCC
/* Private variables ---------------------------------------------------------*/
// TIM_TimeBaseInitTypeDef's order is {uint16_t TIM_Prescaler, uint16_t TIM_CounterMode, uint16_t TIM_Period, uint16_t TIM_ClockDivision, uint8_t TIM_RepetitionCounter}
TIM_TimeBaseInitTypeDef  TimeBaseStructureMilli = {42000,TIM_CounterMode_Up,0,0,0};
TIM_TimeBaseInitTypeDef  TimeBaseStructureMicro = {42,TIM_CounterMode_Up,0,0,0};
//  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;

volatile uint32_t __counter_millis;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  millisecond
  * @param  none
  * @retval None
  */
void delay5_start(void) {
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM5 , ENABLE);

  /* Time base configuration */
  TimeBaseStructureMilli.TIM_Period = 0xffff; //((1000+1) * 1)-1;
  TIM_TimeBaseInit(TIM5, &TimeBaseStructureMilli);

//  TIM_SelectOnePulseMode(TIM5, TIM_OPMode_Single);

  TIM_SetCounter(TIM5,0);

  TIM_ITConfig(TIM5, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM5, ENABLE);

//  while (TIM_GetCounter(TIM_NUM)){};

  /* TIM enable counter */
//  TIM_Cmd(TIM_NUM, DISABLE);
  __counter_millis = 0;
}

uint32_t micros(void) {
	return TIM_GetCounter(TIM5);
}

uint32_t millis(void) {
	return __counter_millis + TIM_GetCounter(TIM5);
}

void delay(uint32_t w) {
	uint32_t wtill = millis() + w;
	while ( millis() >= wtill);
}

void TIM5_IRQHandler(void) {
    if( TIM_GetITStatus( TIM5, TIM_IT_Update) != RESET) {
        TIM_ClearITPendingBit(TIM5, TIM_IT_Update);
        __counter_millis += 0x10000;
    }
}
