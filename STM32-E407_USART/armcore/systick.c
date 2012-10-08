/*
 * systick.c
 *
 *  Created on: 2012/09/29
 *      Author: sin
 */

#include "systick.h"

/**
 * @brief  Inserts a delay time.
 * @param  nTime: specifies the delay time length, in milliseconds.
 * @retval None
 */
/*
void SysTick_delay(__IO uint32_t nTime) {
	TimingDelay = nTime;

	while (TimingDelay != 0)
		;
}
*/

/**
 * @brief  Decrements the TimingDelay variable.
 * @param  None
 * @retval None
 */
void SysTick_increment(void) {
	SysTickMillis++;
}

uint8 SysTick_init(uint32 clksPerTick) {
	return (SysTick_Config(SystemCoreClock / clksPerTick));
}


void SysTick_stop(void) {
	SysTick ->CTRL = 0;
}
/*
void SysTick_Handler(void) {
	SysTick_decrement();
}
*/

uint32 SysTick_millis() {
	return SysTickMillis;
}

