/*
 * delay.h
 *
 *  Created on: 2012/09/29
 *      Author: sin
 */

#ifndef DELAY_H_
#define DELAY_H_

#define __delay()						\
do {							\
  volatile unsigned int i;				\
  for (i = 0; i < 1000000; ++i)				\
    __asm__ __volatile__ ("nop\n\t":::"memory");	\
} while (0)

static __IO uint32_t TimingDelay;

void TimingDelay_Decrement(void);
void Delay(__IO uint32_t nTime);


#endif /* DELAY_H_ */
