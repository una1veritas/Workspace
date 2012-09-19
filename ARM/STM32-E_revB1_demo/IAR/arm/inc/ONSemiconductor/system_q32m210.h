/* ----------------------------------------------------------------------------
 * Copyright (c) 2011 - 2012 Semiconductor Components Industries, LLC (d/b/a        
 * ON Semiconductor), All Rights Reserved
 * 
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * system_Q32M210.h
 * - CMSIS Cortex-M3 Device Peripheral Access Layer Header File for the 
 *   Q32M210 
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef SYSTEM_Q32M210_H
#define SYSTEM_Q32M210_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

extern uint32_t SystemCoreClock;     /* System Clock Frequency (Core Clock) */

/* ----------------------------------------------------------------------------
 * System Initialization
 * ------------------------------------------------------------------------- */
extern void SystemInit (void);

/* ----------------------------------------------------------------------------
 * System Core Clock and Clock Configuration
 * ----------------------------------------------------------------------------
 * Updates the SystemCoreClock with current core Clock 
 * retrieved from cpu registers.  Note:  For Q32M210, this function
 * does nothing.  Developers should call Sys_Analog_Set_RCFreq to
 * set the RC oscillator based on the manufacturing calibration data
 * stored in flash.  This in turn will update SystemCoreClock.
 * ------------------------------------------------------------------------- */
extern void SystemCoreClockUpdate (void);

#ifdef __cplusplus
}
#endif

#endif /* SYSTEM_Q32M210_H */
