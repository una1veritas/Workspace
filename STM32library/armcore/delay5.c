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
TIM_TimeBaseInitTypeDef  TimeBaseStructureMicro = {4200,TIM_CounterMode_Up,0,0,0};
//  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
uint32_t micro_count;
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  micro second
  * @param  none
  * @retval None
  */
void delay5_start(void) {
  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM5 , ENABLE);

  /* Time base configuration */
  TimeBaseStructureMicro.TIM_Period = ((1000+1) * 1)-1;
  TIM_TimeBaseInit(TIM5, &TimeBaseStructureMicro);

//  TIM_SelectOnePulseMode(TIM5, TIM_OPMode_Single);

  TIM_SetCounter(TIM5,2);

  /* TIM enable counter */
  TIM_Cmd(TIM5, ENABLE);

//  while (TIM_GetCounter(TIM_NUM)){};

  /* TIM enable counter */
//  TIM_Cmd(TIM_NUM, DISABLE);
  micro_count = 0;
}

uint32_t microsec(void) {
	micro_count++;
	return TIM_GetCounter(TIM5);
}


