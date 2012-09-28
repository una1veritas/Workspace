/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_watchdog.h
 * - Watchdog timer hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_WATCHDOG_H
#define Q32_WATCHDOG_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Watchdog timer support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Watchdog_Refresh()
 * ----------------------------------------------------------------------------
 * Description   : Refresh the watchdog timer count
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Watchdog_Refresh()
{
    WATCHDOG->REFRESH_CTRL = WATCHDOG_REFRESH;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Watchdog_Set_Timeout( uint32_t timeout )
 * ----------------------------------------------------------------------------
 * Description   : Set the watchdog timeout period
 * Inputs        : timeout - Timeout value for watchdog; use WATCHDOG_TIMEOUT_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Watchdog_Set_Timeout( uint32_t timeout )
{
    WATCHDOG->CTRL = timeout;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_WATCHDOG_H */
