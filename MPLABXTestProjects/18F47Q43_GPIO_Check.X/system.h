#ifndef SYSTEM_H
#define	SYSTEM_H

#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef _XTAL_FREQ
#define _XTAL_FREQ 64000000U
#endif

#include "uart3.h"
#include "interrupt.h"

void system_init(void);
void pins_default(void);
//void PIN_MANAGER_IOC(void);
void port_init(void);
void NCO1_init(void);
//bool NCO1_GetOutputStatus(void);
void INTERRUPT_Initialize (void);

#endif	/* SYSTEM_H */

