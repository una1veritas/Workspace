/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_flash.h
 * - System support functions and macros for configuring flash behavior and
 *   writing to the flash memory
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_FLASH_H
#define Q32_FLASH_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

/* ----------------------------------------------------------------------------
 * Flash ECC support inline functions
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Flash_Get_ErrorDetectStatus()
 * ----------------------------------------------------------------------------
 * Description   : Check if an ECC error has been detected but not corrected on
 *                 a read from flash memory
 * Inputs        : None
 * Outputs       : status - Value of the FLASH_ECC_STATUS_ERROR_DETECT bit;
 *                 compare with ECC_ERROR_NOT_DETECTED_BITBAND/
 *                 ECC_ERROR_DETECTED_BITBAND
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Flash_Get_ErrorDetectStatus()
{
    return FLASH_ECC_STATUS->ERROR_DETECT_ALIAS;
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_Flash_Get_ErrorCorrectStatus()
 * ----------------------------------------------------------------------------
 * Description   : Check if an ECC error has been detected and corrected on a
 *                 read from flash memory
 * Inputs        : None
 * Outputs       : status   - Value of the FLASH_ECC_STATUS_ERROR_CORRECT bit;
 *                 compare with ECC_ERROR_NOT_CORRECTED_BITBAND/
 *                 ECC_ERROR_CORRECTED_BITBAND
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE uint32_t Sys_Flash_Get_ErrorCorrectStatus()
{
    return FLASH_ECC_STATUS->ERROR_CORRECT_ALIAS;
}

/* ----------------------------------------------------------------------------
 * Function      : void Sys_Flash_Clear_ErrorStatus()
 * ----------------------------------------------------------------------------
 * Description   : Clear the flash memory ECC error status bits
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_Flash_Clear_ErrorStatus()
{
    FLASH->ECC_STATUS = (ECC_ERROR_DETECT_CLEAR | ECC_ERROR_CORRECT_CLEAR);
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32_FLASH_H */
