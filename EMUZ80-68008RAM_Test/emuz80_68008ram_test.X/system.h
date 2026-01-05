/* 
 * File:   system.h
 * Author: sin
 *
 * Created on December 3, 2025, 9:16 PM
 */

#ifndef SYSTEM_H
#define	SYSTEM_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <xc.h>
#include <stdint.h>
#include <stdbool.h>

//#ifndef _XTAL_FREQ
/* cppcheck-suppress misra-c2012-21.1 */
#define _XTAL_FREQ 64000000U
//#endif

#define CLK_68008_FREQ   8000UL

void CLOCK_Initialize(void);



void PIN_MANAGER_Initialize (void);

//void PIN_MANAGER_IOC(void);

void NCO1_Initialize(void);

#define INTERRUPT_GlobalInterruptEnable() (INTCON0bits.GIE = 1)
#define INTERRUPT_GlobalInterruptDisable() (INTCON0bits.GIE = 0)
#define INTERRUPT_GlobalInterruptStatus() (INTCON0bits.GIE)

void INTERRUPT_Initialize (void);
void SYSTEM_Initialize(void);

#ifdef	__cplusplus
}
#endif

#endif	/* SYSTEM_H */

