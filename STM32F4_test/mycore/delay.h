/*
 * delay.h
 *
 *  Created on: 2012/09/29
 *      Author: sin
 */

#ifndef DELAY_H_
#define DELAY_H_

#include <stdint.h>

#define __delay()						\
do {							\
  volatile unsigned int i;				\
  for (i = 0; i < 1000000; ++i)				\
    __asm__ __volatile__ ("nop\n\t":::"memory");	\
} while (0)

static __IO uint32_t TimingDelay;

void SysTick_decrement(void);
void SysTick_delay(__IO uint32_t nTime);
uint8_t SysTick_init(uint32_t coreClockPerTick);
void SysTick_start(uint32_t Tick);
void SysTick_stop();


#endif /* DELAY_H_ */
