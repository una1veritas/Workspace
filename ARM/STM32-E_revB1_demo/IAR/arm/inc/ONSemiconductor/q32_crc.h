/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_crc.h
 * - Cyclic-Redundancy Check (CRC) peripheral support functions and macros
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_CRC_H
#define Q32_CRC_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Cyclic-Redundancy Check (CRC) peripheral support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : void Sys_CRC_Initialize()
 * ----------------------------------------------------------------------------
 * Description   : Re-initialize the CRC generator hardware.
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_CRC_Initialize()
{
    CRC->DATA = CRC_INIT_VALUE;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_CRC_Config(uint32_t config)
 * ----------------------------------------------------------------------------
 * Description   : Configure the CRC generator to allow data to be added to the
 *                 generator in little-endian or big-endian byte order
 * Inputs        : config   - Select the endianess of bytes added to the
 *                            calculated CRC; use
 *                            CRC_BIG_ENDIAN/CRC_LITTLE_ENDIAN
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_CRC_Config(uint32_t config)
{
    CRC->CTRL = (config & (1U << CRC_CTRL_BYTE_ORDER_Pos));
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_CRC_GetCurrentCRC()
 * ----------------------------------------------------------------------------
 * Description   : Return the current calculated CRC value
 * Inputs        : None
 * Outputs       : return value - Current calculated value stored in CRC_DATA
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_CRC_GetCurrentCRC()
{
    return CRC->DATA;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_CRC_AddChar(uint8_t data)
 * ----------------------------------------------------------------------------
 * Description   : Add an unsigned char to the current CRC calculation
 * Inputs        : data - Unsigned char to add to the current CRC calculation
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_CRC_AddChar(uint8_t data)
{
    CRC->ADD_8 = data;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_CRC_AddShort(uint16_t data)
 * ----------------------------------------------------------------------------
 * Description   : Add an unsigned short to the current CRC calculation
 * Inputs        : data - Unsigned short to add to the current CRC calculation
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_CRC_AddShort(uint16_t data)
{
    CRC->ADD_16 = data;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_CRC_AddInt(uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Add an unsigned integer to the current CRC calculation
 * Inputs        : data - Unsigned integer to add to the current CRC calculation
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_CRC_AddInt(uint32_t data)
{
    CRC->ADD_32 = data;
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_CRC_H */
