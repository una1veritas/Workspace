/* ----------------------------------------------------------------------------
 * Copyright (c) 2010 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m210_info_map.h
 * - Memory map for the information page
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_INFO_MAP_H
#define Q32M210_INFO_MAP_H

#include <q32m210_map.h>

/* ----------------------------------------------------------------------------
 * Information Page Memory Map Version
 * ------------------------------------------------------------------------- */
#define INFO_MAP_VER_MAJOR          0x1
#define INFO_MAP_VER_MINOR          0x1
#define INFO_MAP_VER_REVISION       0x0

#define INFO_MAP_VERSION            ((INFO_MAP_VER_MAJOR << 12) | \
                                     (INFO_MAP_VER_MINOR << 8)  | \
                                     (INFO_MAP_VER_REVISION))

/* ----------------------------------------------------------------------------
 * Data structures used by the below blocks
 * ----------------------------------------------------------------------------
 * Data structure containing the register address and data to write to that
 * address for data in the system initialization table. */
struct __Sys_Init_Data
{
    uint32_t *addr;
    uint32_t data;
};

/* ----------------------------------------------------------------------------
 * Manufacturing Data Block
 * ------------------------------------------------------------------------- */
#define Q32M210_MFTR_BLOCK_BASE       (Q32M210_FLASH_INFO_BASE)

/* Information page version */
#define Q32M210_MFTR_INFO_MAP_VERSION (Q32M210_MFTR_BLOCK_BASE + 0x00)

/* Manufacturing block pointers */
#define Q32M210_MFTR_BLOCK_END_Pos    (0x04)
#define Q32M210_MFTR_BLOCK_END_PTR    (*(uint32_t const **) \
                                      (Q32M210_MFTR_BLOCK_BASE + \
                                       Q32M210_MFTR_BLOCK_END_Pos))
#define Q32M210_MFTR_BLOCK_EXT_Pos    (0x08)
#define Q32M210_MFTR_BLOCK_EXT_PTR    (*(uint32_t const **) \
                                      (Q32M210_MFTR_BLOCK_BASE + \
                                       Q32M210_MFTR_BLOCK_EXT_Pos))

/* Manufacturing information */
#define Q32M210_MFTR_WAFER_LOT_NUM    (Q32M210_MFTR_BLOCK_BASE + 0x10)
#define Q32M210_MFTR_WAFER_NUM        (Q32M210_MFTR_BLOCK_BASE + 0x14)
#define Q32M210_MFTR_WAFER_X_COORD    (Q32M210_MFTR_BLOCK_BASE + 0x18)
#define Q32M210_MFTR_WAFER_Y_COORD    (Q32M210_MFTR_BLOCK_BASE + 0x1C)

/* Test information */
#define Q32M210_MFTR_TEST_TIMESTAMP   (Q32M210_MFTR_BLOCK_BASE + 0x20)
#define Q32M210_MFTR_TEST_VERSION1    (Q32M210_MFTR_BLOCK_BASE + 0x24)
#define Q32M210_MFTR_TEST_TESTER1     (Q32M210_MFTR_BLOCK_BASE + 0x28)
#define Q32M210_MFTR_TEST_VERSION2    (Q32M210_MFTR_BLOCK_BASE + 0x2C)
#define Q32M210_MFTR_TEST_TESTER2     (Q32M210_MFTR_BLOCK_BASE + 0x30)
#define Q32M210_MFTR_TEST_VERSION3    (Q32M210_MFTR_BLOCK_BASE + 0x34)
#define Q32M210_MFTR_TEST_TESTER3     (Q32M210_MFTR_BLOCK_BASE + 0x38)

/* CRC Value */
#define Q32M210_MFTR_BLOCK_CRC        (Q32M210_MFTR_BLOCK_BASE + 0xFC)


/* ----------------------------------------------------------------------------
 * Calibration Data Block
 * ------------------------------------------------------------------------- */
#define Q32M210_CAL_BLOCK_BASE        (Q32M210_FLASH_INFO_BASE + 0x100)

/* Invalid data setting; matches the data at Q32M210_CAL_*_INFO_Pos for any
 * uninitialized or invalid calibration data */
#define Q32M210_CAL_INFO_INVALID      0xFF

/* Calibration block pointers */
#define Q32M210_CAL_BLOCK_END_Pos     (0x000)
#define Q32M210_CAL_BLOCK_END_PTR     (*(uint32_t **) \
                                      (Q32M210_CAL_BLOCK_BASE + \
                                       Q32M210_CAL_BLOCK_END_Pos))
#define Q32M210_CAL_BLOCK_EXT_Pos     (0x004)
#define Q32M210_CAL_BLOCK_EXT_PTR     (*(uint32_t **) \
                                      (Q32M210_CAL_BLOCK_BASE + \
                                       Q32M210_CAL_BLOCK_EXT_Pos))


/* Analog power component calibration */
#define Q32M210_CAL_REF_CTRL_BASE             (Q32M210_CAL_BLOCK_BASE + 0x010)
#define Q32M210_CAL_REF_CTRL_INFO_Pos         24

#define Q32M210_CAL_PSU_CTRL0_BASE            (Q32M210_CAL_BLOCK_BASE + 0x020)
#define Q32M210_CAL_PSU_CTRL0_INFO_Pos        24

#define Q32M210_CAL_DOUT_VTS_CTRL_BASE        (Q32M210_CAL_BLOCK_BASE + 0x030)
#define Q32M210_CAL_DOUT_VTS_CTRL_INFO_Pos    24

/* Analog clock component calibration */
#define Q32M210_CAL_RC_CCR_CTRL_BASE          (Q32M210_CAL_BLOCK_BASE + 0x080)
#define Q32M210_CAL_RC_CCR_CTRL_INFO_Pos      24

/* Flash component calibration */
#define Q32M210_CAL_SE_CTRL0_BASE             (Q32M210_CAL_BLOCK_BASE + 0x100)
#define Q32M210_CAL_SE_CTRL0_INFO_Pos         24

/* Calibrated VTS measurements */
#define Q32M210_CAL_VTS_ADC_MEAS_BASE         (Q32M210_CAL_BLOCK_BASE + 0x110)
#define Q32M210_CAL_VTS_ADC_MEAS_INFO_Pos     0

/* CRC Value */
#define Q32M210_CAL_BLOCK_CRC         (Q32M210_CAL_BLOCK_BASE + 0x1FC)


/* ----------------------------------------------------------------------------
 * System Data Block
 * ------------------------------------------------------------------------- */
#define Q32M210_SYS_BLOCK_BASE        (Q32M210_FLASH_INFO_BASE + 0x300)

/* System block pointers */
#define Q32M210_SYS_BLOCK_END_Pos     (0x00)
#define Q32M210_SYS_BLOCK_END_PTR     (*(uint32_t **) \
                                      (Q32M210_SYS_BLOCK_BASE + \
                                       Q32M210_SYS_BLOCK_END_Pos))
#define Q32M210_SYS_BLOCK_EXT_Pos     (0x04)
#define Q32M210_SYS_BLOCK_EXT_PTR     (*(uint32_t **) \
                                      (Q32M210_SYS_BLOCK_BASE + \
                                       Q32M210_SYS_BLOCK_EXT_Pos))
#define Q32M210_SYS_CONST_BASE_Pos    (0x08)
#define Q32M210_SYS_CONST_BASE_PTR    (*(uint32_t **) \
                                      (Q32M210_SYS_BLOCK_BASE + \
                                       Q32M210_SYS_CONST_BASE_Pos))
#define Q32M210_SYS_INIT_BASE_Pos     (0x0C)
#define Q32M210_SYS_INIT_BASE_PTR     (*(uint32_t **) \
                                      (Q32M210_SYS_BLOCK_BASE + \
                                       Q32M210_SYS_INIT_BASE_Pos))

/* System data constants */
#define Q32M210_SYS_CONST_BASE        (Q32M210_SYS_BLOCK_BASE + 0x10)

#define Q32M210_SYS_STARTUP_DELAY_Pos (0x00)
#define Q32M210_SYS_STARTUP_DELAY     (*(uint32_t **) \
                                      (Q32M210_SYS_CONST_BASE + \
                                       Q32M210_SYS_STARTUP_DELAY_Pos))
#define Q32M210_SYS_TS_REF_OFFSET_Pos (0x04)
#define Q32M210_SYS_TS_REF_OFFSET     (*(uint32_t **) \
                                      (Q32M210_SYS_CONST_BASE + \
                                       Q32M210_SYS_TS_REF_OFFSET_Pos))
#define Q32M210_SYS_TS_REF_SLOPE_Pos  (0x08)
#define Q32M210_SYS_TS_REF_SLOPE      (*(uint32_t **) \
                                      (Q32M210_SYS_CONST_BASE + \
                                       Q32M210_SYS_TS_REF_SLOPE_Pos))

/* System data initialization table */
#define Q32M210_SYS_INIT_BASE         (Q32M210_SYS_BLOCK_BASE + 0x40)

#define Q32M210_SYS_INIT_COUNT_Pos    (0x00)
#define Q32M210_SYS_INIT_COUNT        (*(uint32_t *) \
                                      (Q32M210_SYS_INIT_BASE + \
                                       Q32M210_SYS_INIT_COUNT_Pos))
#define Q32M210_SYS_INIT_DATA0_Pos    (0x04)
#define Q32M210_SYS_INIT_DATA0_PTR    (*(struct __Sys_Init_Data **) \
                                      (Q32M210_SYS_INIT_BASE + \
                                       Q32M210_SYS_INIT_ADDR0_Pos)

/* CRC Value */
#define Q32M210_SYS_BLOCK_CRC         (Q32M210_SYS_BLOCK_BASE + 0xFC)


#endif /* Q32M210_INFO_MAP_H */


