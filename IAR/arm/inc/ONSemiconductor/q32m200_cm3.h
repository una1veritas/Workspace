/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m200_cm3.h
 * - Support function prototypes and macros directly supporting the ARM Cortex-
 *   M3 processor
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M200_CM3_H
#define Q32M200_CM3_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 *   ARM Cortex-M3 processor support macros
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Macro         : SYS_WAIT_FOR_EVENT
 * ----------------------------------------------------------------------------
 * Description   : Hold the ARM Cortex-M3 core waiting for an event, interrupt
 *                 request, abort or debug entry request (ARM Thumb-2 WFE
 *                 instruction)
 * Inputs        : None
 * Outputs       : None
 * ------------------------------------------------------------------------- */
#define SYS_WAIT_FOR_EVENT              __WFE

/* ----------------------------------------------------------------------------
 * Macro         : SYS_WAIT_FOR_INTERRUPT
 * ----------------------------------------------------------------------------
 * Description   : Hold the ARM Cortex-M3 core waiting for an interrupt
 *                 request, abort or debug entry request (ARM Thumb-2 WFI
 *                 instruction)
 * Inputs        : None
 * Outputs       : None
 * ------------------------------------------------------------------------- */
#define SYS_WAIT_FOR_INTERRUPT          __WFI

/* ----------------------------------------------------------------------------
 *   Program ROM macros
 * ------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------
 * Macro         : SYS_BOOTROM_STARTAPP_RETURN
 * ----------------------------------------------------------------------------
 * Description   : Read the start application return code from the application
 *                 stack for the current application
 * Inputs        : None
 * Outputs       : return   - The value stored on the top of the stack; compare
 *                            against BOOTROM_ERR_*
 * ------------------------------------------------------------------------- */
#define SYS_BOOTROM_STARTAPP_RETURN \
                                *(*((uint32_t **)NVIC_VTABLE_OFFSET) - 1)

/* ----------------------------------------------------------------------------
 *   ARM Cortex-M3 processor support function prototypes
 * ------------------------------------------------------------------------- */
extern void Sys_CM3_SoftReset(void);
extern void Sys_CM3_CoreReset(void);
extern void Sys_CM3_SystemReset(void);

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32M200_CM3_H */
