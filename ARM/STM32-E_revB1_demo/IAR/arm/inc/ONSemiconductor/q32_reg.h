/* -------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a
 * ON Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * -------------------------------------------------------------------------
 * q32_reg.h
 * - Register support macros
 * -------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_REG_H
#define Q32_REG_H

/* ----------------------------------------------------------------------------
 * Register Cast Macros
 * ----------------------------------------------------------------------------
 * ----------------------------------------------------------------------------
 * Macro         : REG8_POINTER(addr)
 * ----------------------------------------------------------------------------
 * Description   : Set up a de-referenced pointer to an 8-bit register that has
 *                 at least one writable bit
 * Inputs        : addr     - Address of the register
 * Outputs       : return   - A de-referenced pointer to the unsigned 8-bit
 *                            value at addr
 * ------------------------------------------------------------------------- */
#define REG8_POINTER(addr)              (*((__IO uint8_t *) (addr)))

/* ----------------------------------------------------------------------------
 * Macro         : READONLY_REG8_POINTER(addr)
 * ----------------------------------------------------------------------------
 * Description   : Set up a de-referenced pointer to an 8-bit register that has
 *                 no writable bits
 * Inputs        : addr     - Address of the register
 * Outputs       : return   - A de-referenced pointer to the constant unsigned
 *                            8-bit value at addr
 * ------------------------------------------------------------------------- */
#define READONLY_REG8_POINTER(addr)     (*((__I uint8_t *) (addr)))

/* ----------------------------------------------------------------------------
 * Macro         : REG16_POINTER(addr)
 * ----------------------------------------------------------------------------
 * Description   : Set up a de-referenced pointer to a 16-bit register that has
 *                 at least one writable bit
 * Inputs        : addr     - Address of the register
 * Outputs       : return   - A de-referenced pointer to the unsigned 16-bit
 *                            value at addr
 * ------------------------------------------------------------------------- */
#define REG16_POINTER(addr)             (*((__IO uint16_t *) (addr)))

/* ----------------------------------------------------------------------------
 * Macro         : READONLY_REG16_POINTER(addr)
 * ----------------------------------------------------------------------------
 * Description   : Set up a de-referenced pointer to a 16-bit register that has
 *                 no writable bits
 * Inputs        : addr     - Address of the register
 * Outputs       : return   - A de-referenced pointer to the constant unsigned
 *                            16-bit value at addr
 * ------------------------------------------------------------------------- */
#define READONLY_REG16_POINTER(addr)    (*((__I uint16_t *) (addr)))

/* ----------------------------------------------------------------------------
 * Macro         : REG32_POINTER(addr)
 * ----------------------------------------------------------------------------
 * Description   : Set up a de-referenced pointer to a 32-bit register that has
 *                 at least one writable bit
 * Inputs        : addr     - Address of the register
 * Outputs       : return   - A de-referenced pointer to the unsigned 32-bit
 *                            value at addr
 * ------------------------------------------------------------------------- */
#define REG32_POINTER(addr)             (*((__IO uint32_t *) (addr)))

/* ----------------------------------------------------------------------------
 * Macro         : READONLY_REG32_POINTER(addr)
 * ----------------------------------------------------------------------------
 * Description   : Set up a de-referenced pointer to a 32-bit register that has
 *                 no writable bits
 * Inputs        : addr     - Address of the register
 * Outputs       : return   - A de-referenced pointer to the constant unsigned
 *                            32-bit value at addr
 * ------------------------------------------------------------------------- */
#define READONLY_REG32_POINTER(addr)    (*((__I uint32_t *) (addr)))

/* ----------------------------------------------------------------------------
 * Macro         : SYS_CALC_BITBAND(addr, pos)
 * ----------------------------------------------------------------------------
 * Description   : Calculate the bit-band location for a given address and bit
 *                 position
 * Inputs        : - addr   - The bit-band address for which the bit-band alias
 *                            is being calculated
 *                 - pos    - The bit for which the bit-band alias is being
 *                            calculated
 * Outputs       : return   - The bit-band alias address corresponding to this
 *                            address and bit position
 * ------------------------------------------------------------------------- */
#define SYS_CALC_BITBAND(addr, pos)     ((addr & 0xF0000000) \
                                         + 0x2000000 \
                                         + ((addr & 0xFFFFF) << 5) \
                                         + (pos << 2))

#endif /* Q32_REG_H */

