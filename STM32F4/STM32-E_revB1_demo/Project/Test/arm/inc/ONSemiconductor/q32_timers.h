/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_timers.h
 * - General-purpose system timer hardware support functions and macros
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_TIMERS_H
#define Q32_TIMERS_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Timer selection defines
 * ------------------------------------------------------------------------- */
#define SELECT_TIMER0                   (1U << \
                                         TIMER_CTRL_STATUS_TIMER0_STATUS_Pos)
#define SELECT_TIMER1                   (1U << \
                                         TIMER_CTRL_STATUS_TIMER1_STATUS_Pos)
#define SELECT_TIMER2                   (1U << \
                                         TIMER_CTRL_STATUS_TIMER2_STATUS_Pos)
#define SELECT_TIMER3                   (1U << \
                                         TIMER_CTRL_STATUS_TIMER3_STATUS_Pos)

#define SELECT_ALL_TIMERS               (SELECT_TIMER0 | SELECT_TIMER1 \
                                         | SELECT_TIMER2 | SELECT_TIMER3)
#define SELECT_NO_TIMERS                (0)

/* ----------------------------------------------------------------------------
 * General-purpose system timer support function prototypes
 * ------------------------------------------------------------------------- */

extern void Sys_Timers_Start(uint32_t config);
extern void Sys_Timers_Stop(uint32_t config);

/* ----------------------------------------------------------------------------
 * General-purpose system timer support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Timer0_Set_Control( uint32_t config )
 * ----------------------------------------------------------------------------
 * Description   : Set up general-purpose system timer 0
 * Inputs        : config - Control configuration for timer 0; use
 *                          TIMER0_COUNT_*,
 *                          TIMER0_FIXED_COUNT_RUN/TIMER0_FREE_RUN,
 *                          TIMER0_PRESCALE_* (or a value shifted by
 *                          TIMER0_CTRL_PRESCALE_Pos) and a timeout count
 *                          setting
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Timer0_Set_Control( uint32_t config )
{
    TIMER->TIMER0_CTRL = (config & (TIMER0_CTRL_MULTI_COUNT_Mask
                                 | (1U << TIMER0_CTRL_MODE_Pos)
                                 | TIMER0_CTRL_PRESCALE_Mask
                                 | TIMER0_CTRL_TIMEOUT_VALUE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Timer1_Set_Control( uint32_t config )
 * ----------------------------------------------------------------------------
 * Description   : Set up general-purpose system timer 1
 * Inputs        : config - Control configuration for timer 1; use
 *                          TIMER1_COUNT_*,
 *                          TIMER1_FIXED_COUNT_RUN/TIMER1_FREE_RUN,
 *                          TIMER1_PRESCALE_* (or a value shifted by
 *                          TIMER1_CTRL_PRESCALE_Pos) and a timeout count
 *                          setting
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Timer1_Set_Control( uint32_t config )
{
    TIMER->TIMER1_CTRL = (config & (TIMER1_CTRL_MULTI_COUNT_Mask
                                 | (1U << TIMER1_CTRL_MODE_Pos)
                                 | TIMER1_CTRL_PRESCALE_Mask
                                 | TIMER1_CTRL_TIMEOUT_VALUE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Timer2_Set_Control( uint32_t config )
 * ----------------------------------------------------------------------------
 * Description   : Set up general-purpose system timer 2
 * Inputs        : config - Control configuration for timer 2; use
 *                          TIMER2_COUNT_*,
 *                          TIMER2_FIXED_COUNT_RUN/TIMER2_FREE_RUN,
 *                          TIMER2_PRESCALE_* (or a value shifted by
 *                          TIMER2_CTRL_PRESCALE_Pos) and a timeout count
 *                          setting
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Timer2_Set_Control( uint32_t config )
{
    TIMER->TIMER2_CTRL = (config & (TIMER2_CTRL_MULTI_COUNT_Mask
                                 | (1U << TIMER2_CTRL_MODE_Pos)
                                 | TIMER2_CTRL_PRESCALE_Mask
                                 | TIMER2_CTRL_TIMEOUT_VALUE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Timer3_Set_Control( uint32_t config )
 * ----------------------------------------------------------------------------
 * Description   : Set up general-purpose system timer 3
 * Inputs        : config - Control configuration for timer 3; use
 *                          TIMER3_COUNT_*,
 *                          TIMER3_FIXED_COUNT_RUN/TIMER3_FREE_RUN,
 *                          TIMER3_PRESCALE_* (or a value shifted by
 *                          TIMER3_CTRL_PRESCALE_Pos) and a timeout count
 *                          setting
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Timer3_Set_Control( uint32_t config )
{
    TIMER->TIMER3_CTRL = (config & (TIMER3_CTRL_MULTI_COUNT_Mask
                                 | (1U << TIMER3_CTRL_MODE_Pos)
                                 | TIMER3_CTRL_PRESCALE_Mask
                                 | TIMER3_CTRL_TIMEOUT_VALUE_Mask));
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Timers_Set_Status(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Set the running or stopped status of general-purpose system
 *                 timers 0 to 3
 * Inputs        : config - The new timer 0 to 3 status; use
 *                          TIMER*_START|TIMER*_STOP
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Timers_Set_Status(uint32_t config)
{
    TIMER->CTRL_STATUS = (config
                         & ((1U << TIMER_CTRL_STATUS_TIMER3_STATUS_Pos)
                            | (1U << TIMER_CTRL_STATUS_TIMER2_STATUS_Pos)
                            | (1U << TIMER_CTRL_STATUS_TIMER1_STATUS_Pos)
                            | (1U << TIMER_CTRL_STATUS_TIMER0_STATUS_Pos)));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Timers_Get_Status()
 * ----------------------------------------------------------------------------
 * Description   : Return the current running or stopped status of
 *                 general-purpose system timers 0 to 3
 * Inputs        : None
 * Outputs       : return value - The current timer 0 to 3 status; value loaded
 *                                from TIMER_CTRL_STATUS
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Timers_Get_Status()
{
    return TIMER->CTRL_STATUS;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_TIMERS_H */
