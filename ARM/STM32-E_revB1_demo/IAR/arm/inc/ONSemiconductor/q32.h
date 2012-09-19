/* -------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a
 * ON Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * -------------------------------------------------------------------------
 * q32.h
 * - Top-level include file for the system firmware library
 * -------------------------------------------------------------------------
 * $Revision: 53433 $
 * $Date: 2012-05-20 19:04:19 +0200 (sö, 20 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_H
#define Q32_H

/* ----------------------------------------------------------------------------
 * Firmware Version
 * ------------------------------------------------------------------------- */
#define SYS_FW_VER_MAJOR                0x03
#define SYS_FW_VER_MINOR                0x00
#define SYS_FW_VER_REVISION             0x01

#define SYS_FW_VER                      ((SYS_FW_VER_MAJOR << 12) | \
                                         (SYS_FW_VER_MINOR << 8)  | \
                                         (SYS_FW_VER_REVISION) )

extern const short q32_SysLib_Version;

/* ----------------------------------------------------------------------------
 * Translate the chip ID to the product name
 * ------------------------------------------------------------------------- */
#if ((Q32_CID == 0x050101) || (Q32_CID == 0x050102) || (Q32_CID == 0x050103))
    #define Q32M210
#endif

/* ----------------------------------------------------------------------------
 * System Includes for a product
 * ------------------------------------------------------------------------- */

#ifdef Q32M210
    #include <q32m210.h>
#endif


#endif /* Q32_H */

