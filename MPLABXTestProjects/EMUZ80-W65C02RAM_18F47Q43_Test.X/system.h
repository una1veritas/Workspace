#ifndef SYSTEM_H
#define	SYSTEM_H

#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef _XTAL_FREQ
#define _XTAL_FREQ 64000000UL
#endif

#define CLK_6502_FREQ 4000000UL

#include "uart3.h"

#define GlobalInterruptHigh     (INTCON0bits.GIE)
#define GlobalInterrupt         (INTCON0bits.GIE)
#define GlobalInterruptLow      (INTCON0bits.GIEL)

void system_init(void);
void pins_default(void);
//void PIN_MANAGER_IOC(void);
void W65C02_interface_init(void);
void NCO1_init(void);
//bool NCO1_GetOutputStatus(void);
void Interrupt_init(void);
void CLC_init(void);

#endif	/* SYSTEM_H */

