/* ----------------------------------------------------------------------------
 * Copyright (c) 2008-2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_systick.h
 * - SYSTICK hardware support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_SYSTICK_H
#define Q32_SYSTICK_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif
    
/* ----------------------------------------------------------------------------
 * SYSTICK support defines
 * ------------------------------------------------------------------------- */
    
#define NVIC_CLKSOURCE_EXTREF_CLK       ((uint32_t)(0x0))
#define NVIC_CLKSOURCE_CORE_CLK         ((uint32_t)(0x1))
    
/* ----------------------------------------------------------------------------
 * SYSTICK support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_Enable()
 * ----------------------------------------------------------------------------
 * Description   : Enable the SYSTICK timer
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_Enable()
{
    SysTick->CTRL = (SysTick->CTRL | (0x1UL << SysTick_CTRL_ENABLE_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_Disable()
 * ----------------------------------------------------------------------------
 * Description   : Disable the SYSTICK timer
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_Disable()
{
    SysTick->CTRL = (SysTick->CTRL & ~(SysTick_CTRL_ENABLE_Msk));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_Set_ClkSource(uint32_t  clksource)
 * ----------------------------------------------------------------------------
 * Description   : Set the clock source for the SYSTICK timer
 * Inputs        : clksource - SYSTICK timer clock source. Use NVIC_CLKSOURCE_*
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_Set_ClkSource(uint32_t clksource)
{
    SysTick->CTRL = (SysTick->CTRL
                         & ~(1U << SysTick_CTRL_CLKSOURCE_Pos))
                         | (clksource << SysTick_CTRL_CLKSOURCE_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_ClkSource()
 * ----------------------------------------------------------------------------
 * Description   : Get the clock source setting for the SYSTICK timer
 * Inputs        : None
 * Outputs       : return value - SYSTICK timer clock source; value loaded from
 *                                SysTick->CTRL
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_ClkSource()
{
    return (SysTick->CTRL >> SysTick_CTRL_CLKSOURCE_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_CountFlag()
 * ----------------------------------------------------------------------------
 * Description   : Get the COUNTFLAG indicating if the SYSTICK timer has
 *                 reached zero since the last time this flag was read.
 * Inputs        : None
 * Outputs       : return value - SYSTICK timer count flag; value loaded from
 *                                SysTick->CTRL
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_CountFlag()
{
    return (SysTick->CTRL >> SysTick_CTRL_COUNTFLAG_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_NoRefFlag()
 * ----------------------------------------------------------------------------
 * Description   : Get the NOREF flag indicating if the SYSTICK timer has an
 *                 external reference clock or not.
 * Inputs        : None
 * Outputs       : return value - SYSTICK timer NOREF flag; value loaded from
 *                                SysTick->CALIB
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_NoRefFlag()
{
    return (SysTick->CALIB >> SysTick_CALIB_NOREF_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_SkewFlag()
 * ----------------------------------------------------------------------------
 * Description   : Get the SKEW flag indicating if the SYSTICK timer TENMS
 *                 calibration value is accurate or not.
 * Inputs        : None
 * Outputs       : return value - SYSTICK timer SKEW flag; Value loaded from
 *                                SysTick->CALIB
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_SkewFlag()
{
    return (SysTick->CALIB >> SysTick_CALIB_SKEW_Pos) & 0x1;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_Set_ReloadCount(uint32_t  count)
 * ----------------------------------------------------------------------------
 * Description   : Set the value to be reloaded when the SYSTICK timer's counter
 *                 reaches zero.
 * Inputs        : count - Counter reload value
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_Set_ReloadCount(uint32_t  count)
{
    SysTick->LOAD = count & SysTick_LOAD_RELOAD_Msk;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_ReloadCount()
 * ----------------------------------------------------------------------------
 * Description   : Get the current value to be reloaded when the SYSTICK timer's
 *                 counter reaches zero.
 * Inputs        : None
 * Outputs       : return value - Counter reload value from NVIC_SYSTICK_RELOAD
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_ReloadCount()
{
    return SysTick->LOAD;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_CurrentCount()
 * ----------------------------------------------------------------------------
 * Description   : Get the SYSTICK timer's current count
 * Inputs        : None
 * Outputs       : return value - Current counter value from
 *                                NVIC_SYSTICK_CURRENT
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_CurrentCount()
{
    return SysTick->VAL;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_ClearCount()
 * ----------------------------------------------------------------------------
 * Description   : Clear the SYSTICK timer's current count.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_ClearCount()
{
    SysTick->VAL = 0;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t  Sys_SYSTICK_Get_CalCount()
 * ----------------------------------------------------------------------------
 * Description   : Get the SYSTICK timer's calibration count (TENMS count).
 *                 This count value corresponds to an elapsed time of 10 ms for
 *                 a SYSTICK frequency of 1 MHz.
 * Inputs        : None
 * Outputs       : return value - Ten millisecond calibration count value; value
 *                                loaded from SysTick->CALIB
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t  Sys_SYSTICK_Get_CalCount()
{
    return (SysTick->CALIB & SysTick_CALIB_TENMS_Pos);
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_EnableInt()
 * ----------------------------------------------------------------------------
 * Description   : Enable the generation of the SYSTICK interrupt
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_EnableInt()
{
    SysTick->CTRL = (SysTick->CTRL | (0x1UL << SysTick_CTRL_TICKINT_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_SYSTICK_DisableInt()
 * ----------------------------------------------------------------------------
 * Description   : Disable the generation of the SYSTICK interrupt
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_SYSTICK_DisableInt()
{
    SysTick->CTRL = (SysTick->CTRL & ~(SysTick_CTRL_TICKINT_Msk));
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_SYSTICK_H */
