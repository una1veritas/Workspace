/*
 * systick.c
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#include "stm32f4xx.h"
#include "systick.h"

volatile uint32_t _systick_counter;

void SysTick_Handler(void) {
	_systick_counter++; /* increment timeTicks counter */
}

void SysTick_delay(const uint32_t dlyTicks) {
	uint32_t currTicks = _systick_counter;

	while ((_systick_counter - currTicks) < dlyTicks)
		;
}

void SysTick_Start(const uint32_t ticks) {
	if ( SysTick_Config(SystemCoreClock / ticks) ) {
		/* Setup SysTick for 1 msec interrupts */
		/* Handle Error */
		while (1)
			;
	}
}

uint32_t SysTick_count() {
	return _systick_counter;
}
