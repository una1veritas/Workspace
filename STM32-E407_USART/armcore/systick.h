/*
 * systick.h
 *
 *  Created on: 2012/09/29
 *      Author: sin
 */

#ifndef SYSTICK_H_
#define SYSTICK_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "armcore.h"
#include "stm32f4xx.h"
//#include "system_stm32f4xx.h"
#include <stm32f4xx_rcc.h>

/*
#define __delay()						\
do {							\
  volatile unsigned int i;				\
  for (i = 0; i < 1000000; ++i)				\
    __asm__ __volatile__ ("nop\n\t":::"memory");	\
} while (0)
*/

static __IO uint32 SysTickMillis;

void SysTick_increment(void);
//void SysTick_delay(__IO uint32_t nTime);
uint8 SysTick_init(uint32 clksPerTick);
//void SysTick_start(uint32_t Tick);
void SysTick_stop();
uint32 SysTick_millis();

#ifdef __cplusplus
}
#endif

#endif /* SYSTICK_H_ */
