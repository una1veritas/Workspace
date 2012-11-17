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

volatile uint32_t __counter_micros;
volatile uint32_t __counter_millis;

/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
  * @brief  millisecond
  * @param  none
  * @retval None
  */
void TIM5_timer_start(void) {
	// TIM_TimeBaseInitTypeDef's order is {uint16_t TIM_Prescaler, uint16_t TIM_CounterMode, uint16_t TIM_Period, uint16_t TIM_ClockDivision, uint8_t TIM_RepetitionCounter}
	TIM_TimeBaseInitTypeDef  TimeBaseStructure = {
			84,
			TIM_CounterMode_Up,
			1000-1,
			TIM_CKD_DIV1,
			0
	};

  //Supply APB1 Clock
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM5 , ENABLE);

  /* Time base configuration */
  TIM_TimeBaseInit(TIM5, &TimeBaseStructure);
//  TIM_SelectOnePulseMode(TIM5, TIM_OPMode_Repetitive);
//  TIM_SetCounter(TIM5,0);
  TIM_ITConfig(TIM5, TIM_IT_Update, ENABLE);

  /* TIM enable counter */
  TIM_Cmd(TIM5, ENABLE);

  __counter_micros = 0;
  __counter_millis = 0;
}

uint32_t micros(void) {
	return __counter_micros + TIM_GetCounter(TIM5);
}

uint32_t millis(void) {
	return __counter_millis;
}

void delay_millis(uint32_t w) {
	uint32_t wtill = millis() + w;
	while ( millis() < wtill);
}

void delay_micros(uint32_t w) {
	uint32_t wtill = micros() + w;
	while ( micros() < wtill);
}

void TIM5_IRQHandler(void) {
    if( TIM_GetITStatus( TIM5, TIM_IT_Update) != RESET) {
        TIM_ClearITPendingBit(TIM5, TIM_IT_Update);
        __counter_micros += 1000;
        __counter_millis += 1;
    }
}
