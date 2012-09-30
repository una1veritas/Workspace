/*
 * delay.cpp
 *
 *  Created on: 2012/09/29
 *      Author: sin
 */

#include <stdint.h>
#include <stm32f4xx.h>

#include "delay.h"

/**
 * @brief  Inserts a delay time.
 * @param  nTime: specifies the delay time length, in milliseconds.
 * @retval None
 */
void SysTick_delay(__IO uint32_t nTime) {
	SysTick_counter = nTime;

	while (SysTick_counter != 0)
		;
}

/**
 * @brief  Decrements the TimingDelay variable.
 * @param  None
 * @retval None
 */
void SysTick_decrement(void) {
	if (SysTick_counter != 0x00) {
		SysTick_counter--;
	}
}

uint8_t SysTick_init(uint32_t coreClockPerTick) {
	return (SysTick_Config(SystemCoreClock / coreClockPerTick));
}

void SysTick_Handler(void) {
	SysTick_decrement();
}

