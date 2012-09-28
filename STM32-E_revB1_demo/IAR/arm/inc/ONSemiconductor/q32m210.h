/* ----------------------------------------------------------------------------
 * Copyright (c) 2010 - 2012 Semiconductor Components Industries, LLC (d/b/a
 * ON Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m210.h
 * - Top-level CMSIS compatible header file for Q32M210
 *
 *   Provides the required <Device>.h implementation for CMSIS compatibility
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_H
#define Q32M210_H

#ifdef __cplusplus
 extern "C" {
#endif 

/* ----------------------------------------------------------------------------
 * Interrupt and Exception Vector Definitions
 * ------------------------------------------------------------------------- */
#include <q32m210_vectors.h>

/* ----------------------------------------------------------------------------
 * Core CMSIS Configuration
 * ------------------------------------------------------------------------- */
#define __CM3_REV                 0x0201    /* Core Revision r2p1 */
#define __NVIC_PRIO_BITS          4         /* Number of Bits used for Priority 
                                             * Levels */
#define __Vendor_SysTickConfig    0         /* Set to 1 if different SysTick 
                                             * Config is used */
#define __MPU_PRESENT             0         /* MPU present or not */

#include <core_cm3.h>                       /* Cortex-M3 processor and core 
                                             * peripherals */
#include "system_Q32M210.h"                 /* Q32M210 System include file */

/* ----------------------------------------------------------------------------
 * Peripheral Register Definitions
 * ------------------------------------------------------------------------- */

#if defined ( __CC_ARM   )
#pragma anon_unions
#endif

#include <q32m210_hw.h>

#if defined ( __CC_ARM )
#pragma no_anon_unions
#endif

/* ----------------------------------------------------------------------------
 * Memory Map and Product-Specific Definitions
 * ------------------------------------------------------------------------- */
#include <q32m210_map.h>
#include <q32m210_gpio_index.h>

/* ROM Vectors */
#include <q32m210_romvect.h>

/* ----------------------------------------------------------------------------
 * Additional Peripheral Support Definitions, Macros and Inline Functions
 * ------------------------------------------------------------------------- */

/* Include legacy peripheral register definitions if desired */
#ifdef COMPATIBILITY_MODE
#include <q32m210_compat.h>
#endif

/* Peripherals */
#include <q32_clock.h>
#include <q32_crc.h>
#include <q32_dma.h>
#include <q32_flash.h>
#include <q32m200_nvic.h>
#include <q32_systick.h>
#include <q32_timers.h>
#include <q32_watchdog.h>

/* Interfaces */
#include <q32_analog.h>
#include <q32_i2c.h>
#include <q32_gpio.h>
#include <q32_lcd.h>
#include <q32_pcm.h>
#include <q32_spi.h>
#include <q32_uart.h>
#include <q32_usb.h>

/* Support Definitions and Functions */
#include <q32m200_cm3.h>
#include <q32_operatingmode.h>
#include <q32_sys.h>

#ifdef __cplusplus
}
#endif

#endif  /* Q32M210_H */
