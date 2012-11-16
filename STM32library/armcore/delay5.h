/*
 * delay5.h
 *
 *  Created on: 2012/11/15
 *      Author: sin
 */

#ifndef DELAY5_H_
#define DELAY5_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"


/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */
uint32_t micros(void);
uint32_t millis(void);
void delay(uint32_t);
void delay5_start(void);

void TIM5_IRQHandler(void);

#ifdef __cplusplus
}
#endif


#endif /* DELAY5_H_ */
