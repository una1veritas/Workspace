/*
 * init.h
 *
 *  Created on: 2012/10/08
 *      Author: sin
 */

#ifndef INIT_H_
#define INIT_H_

#include "systick.h"

void init(void) {
	SysTick_Start();
}


#endif /* INIT_H_ */
