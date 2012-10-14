/*
 * systick.cpp
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */
//#include "stm32f4xx_it.h"
#include "systick.h"

void SysTick_Handler(void) {
	msTicks++; /* increment timeTicks counter */
}

void delay(const uint32_t dlyTicks) {
	uint32_t currTicks = msTicks;

	while ((msTicks - currTicks) < dlyTicks)
		;
}

void SysTick_Start() {
	if (SysTick_Config(SystemCoreClock / 1000)) {
		/* Setup SysTick for 1 msec interrupts */
		/* Handle Error */
		while (1)
			;
	}
}

uint32 millis() {
	return msTicks;
}

