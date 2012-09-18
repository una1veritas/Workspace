/* -------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a
 * ON Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * -------------------------------------------------------------------------
 * q32m210_romvect.h
 * - Vectors for functions embedded in the Program ROM
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * -------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_ROMVECT_H
#define Q32M210_ROMVECT_H

/* ----------------------------------------------------------------------------
 * If building with a C++ compiler, make all of the definitions in this header
 * have a C binding.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------------
 * Error codes used by the functions in the ROM vector table
 * ----------------------------------------------------------------------------
 * Boot application error codes */

typedef enum
{
    BOOTROM_ERR_NONE = 0x0,
    BOOTROM_ERR_BAD_ALIGN = 0x1,
    BOOTROM_ERR_BAD_SP = 0x2,
    BOOTROM_ERR_BAD_RESET_VECT = 0x3,
    BOOTROM_ERR_FAILED_ECC_CHECK = 0x4,
    BOOTROM_ERR_CORRECTED_ECC_CHECK = 0x5,
    BOOTROM_ERR_FAILED_START_APP = 0x6,
    BOOTROM_ERR_BAD_CRC = 0x7
} BootROMStatus;

/* System initialization error codes */

typedef enum
{
    SYS_INIT_ERR_NONE = 0x0,
    SYS_INIT_ERR_INVALID_BLOCK = 0x8,
    SYS_INIT_ERR_BAD_CRC = 0x9,
    SYS_INIT_ERR_INVALID_COUNT = 0xA
} SysInitStatus;

/* Flash error codes */

typedef enum
{
    FLASH_ERR_NONE = 0x0,
    FLASH_ERR_GENERAL_FAILURE = 0x1,
    FLASH_ERR_WRITE_NOT_ENABLED = 0x2,
    FLASH_ERR_BAD_ADDRESS = 0x3
} FlashStatus;

/* ----------------------------------------------------------------------------
 * ROM Vector (Function Table) Base
 * ------------------------------------------------------------------------- */
#define ROMVECT_BASEADDR                0x00000020

/* ----------------------------------------------------------------------------
 * Soft Reset Vector
 * ------------------------------------------------------------------------- */
#define ROMVECT_RESET                   (ROMVECT_BASEADDR + 0x00)

/* ----------------------------------------------------------------------------
 * Flash Write Support Vectors
 * ------------------------------------------------------------------------- */
#define ROMVECT_FLASH_WRITE_WORD        (ROMVECT_BASEADDR + 0x04)
#define ROMVECT_FLASH_WRITE_BUFFER      (ROMVECT_BASEADDR + 0x08)
#define ROMVECT_FLASH_ERASE_PAGE        (ROMVECT_BASEADDR + 0x0C)
#define ROMVECT_FLASH_ERASE_ALL         (ROMVECT_BASEADDR + 0x10)

/* ----------------------------------------------------------------------------
 * Application Boot Support Vectors
 * ------------------------------------------------------------------------- */
#define ROMVECT_BOOTROM_VALIDATE_APP    (ROMVECT_BASEADDR + 0x14)
#define ROMVECT_BOOTROM_START_APP       (ROMVECT_BASEADDR + 0x18)

/* ----------------------------------------------------------------------------
 * ROM based System Delay
 * ------------------------------------------------------------------------- */
#define ROMVECT_PROGRAMROM_SYS_DELAY       (ROMVECT_BASEADDR + 0x1C)

/* ----------------------------------------------------------------------------
 * Functions
 * ------------------------------------------------------------------------- */

extern BootROMStatus Sys_BootROM_StrictStartApp(uint32_t* vect_table);

extern void Sys_Delay_ProgramROM(uint32_t cycles);

/* ----------------------------------------------------------------------------
 * Function      : void Sys_BootROM_Reset()
 * ----------------------------------------------------------------------------
 * Description   : Reset the system by executing the reset vector in the Boot
 *                 ROM
 * Inputs        : None
 * Outputs       : None
 * Assumptions   : None
 * ------------------------------------------------------------------------- */
static __INLINE void Sys_BootROM_Reset()
{
    /* Call the reset vector */
    (*((void (**)(void)) ROMVECT_RESET))();
}

/* ----------------------------------------------------------------------------
 * Function      : FlashStatus Sys_Flash_WriteWord(uint32_t freq_in_hz,
 *                                              uint32_t addr,
 *                                              uint32_t data)
 * ----------------------------------------------------------------------------
 * Description   : Write a single word of flash memory at the specified address
 * Inputs        : - freq_in_hz     - System clock frequency in hertz
 *                 - addr           - Address to write in the flash
 *                 - data           - Data to write to the specified address in
 *                                    flash
 * Outputs       : return value     - Status code indicating whether the
 *                                    requested flash operation succeeded;
 *                                    compare with FLASH_ERR_*
 * Assumptions   : - The calling application has unlocked the flash for write
 *                 - The area of flash memory to be written was previously
 *                   erased if necessary
 * ------------------------------------------------------------------------- */
static __INLINE FlashStatus Sys_Flash_WriteWord(uint32_t freq_in_hz,
                                             uint32_t addr, 
                                             uint32_t data)
{
    return (*((FlashStatus (**)(uint32_t, uint32_t, uint32_t))
              ROMVECT_FLASH_WRITE_WORD))(freq_in_hz, addr, data);
}

/* ----------------------------------------------------------------------------
 * Function      : FlashStatus Sys_Flash_WriteBuffer(uint32_t freq_in_hz,
 *                                                uint32_t start_addr,
 *                                                uint32_t length,
 *                                                uint32_t* data)
 * ----------------------------------------------------------------------------
 * Description   : Write a buffer of memory of the specified length, starting at
 *                 the specified address, to flash memory
 * Inputs        : - freq_in_hz - System clock frequency in hertz
 *                 - start_addr - Start address for the write to flash memory
 *                 - length     - Number of words to write to flash memory
 *                 - data       - Pointer to the data to write to flash memory
 * Outputs       : return value - Status code indicating whether the requested
 *                                flash operation succeeded; compare with
 *                                FLASH_ERR_*
 * Assumptions   : - The calling application has unlocked the flash memory for
 *                   write
 *                 - The area of flash memory to be written has been previously
 *                   erased if necessary
 *                 - "data" points to a buffer of at least "length" words
 * ------------------------------------------------------------------------- */
static __INLINE FlashStatus Sys_Flash_WriteBuffer(uint32_t freq_in_hz,
                                               uint32_t start_addr,
                                               uint32_t length, 
                                               uint32_t* data)
{
    return (*((FlashStatus (**)(uint32_t, uint32_t, uint32_t, uint32_t*))
              ROMVECT_FLASH_WRITE_BUFFER))(freq_in_hz, start_addr, length,
                                           data);
}

/* ----------------------------------------------------------------------------
 * Function      : FlashStatus Sys_Flash_ErasePage(uint32_t freq_in_hz,
 *                                              uint32_t addr)
 * ----------------------------------------------------------------------------
 * Description   : Erase the specified page in either the main block or
 *                 information block of the flash memory
 * Inputs        : - freq_in_hz     - System clock frequency in hertz
 *                 - addr           - Any address within the page to be erased
 * Outputs       : return value     - Status code indicating whether the
 *                                    requested flash operation succeeded;
 *                                    compare with FLASH_ERR_*
 * Assumptions   : The calling application has unlocked the flash memory for
 *                 write
 * ------------------------------------------------------------------------- */
static __INLINE FlashStatus Sys_Flash_ErasePage(uint32_t freq_in_hz, 
                                            uint32_t addr)
{
    return (*((FlashStatus (**)(uint32_t, uint32_t))
              ROMVECT_FLASH_ERASE_PAGE))(freq_in_hz, addr);
}

/* ----------------------------------------------------------------------------
 * Function      : FlashStatus Sys_Flash_EraseAll(uint32_t freq_in_hz)
 * ----------------------------------------------------------------------------
 * Description   : Erase all of the pages in the main flash memory
 * Inputs        : freq_in_hz       - System clock frequency in hertz
 * Outputs       : return value     - Status code indicating whether the
 *                                    requested flash operation succeeded;
 *                                    compare with FLASH_ERR_*
 * Assumptions   : The calling application has unlocked the flash memory for
 *                 write
 * ------------------------------------------------------------------------- */
static __INLINE FlashStatus Sys_Flash_EraseAll(uint32_t freq_in_hz)
{
    return (*((FlashStatus (**)(uint32_t))
              ROMVECT_FLASH_ERASE_ALL))(freq_in_hz);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t
 *                 Sys_BootROM_ValidateApp(uint32_t* vect_table)
 * ----------------------------------------------------------------------------
 * Description   : Validate an application using the Boot ROM application
 *                 checks.
 * Inputs        : vect_table   - Pointer to the vector table at the start of an
 *                                application that will be validated.
 * Outputs       : return value - Status code indicating whether a validation
 *                                error occurred or not; compare against
 *                                BOOTROM_ERR_*
 * Assumptions   : A bus fault handler is installed that can be used to handle
 *                 any detected uncorrectable ECC errors that are found during
 *                 validation of the application.
 * ------------------------------------------------------------------------- */
static __INLINE BootROMStatus Sys_BootROM_ValidateApp(uint32_t* vect_table)
{
    return (*((BootROMStatus (**)(uint32_t*))
              ROMVECT_BOOTROM_VALIDATE_APP))(vect_table);
}

/* ----------------------------------------------------------------------------
 * Function      : uint32_t Sys_BootROM_StartApp(uint32_t* vect_table)
 * ----------------------------------------------------------------------------
 * Description   : Validate and start up an application using the Boot ROM.
 * Inputs        : vect_table   - Pointer to the vector table at the start of an
 *                                application that will be validated and then
 *                                run.
 * Outputs       : return value - Status code indicating application validation
 *                                error if application cannot be started. If not
 *                                returning, the status code is written to the
 *                                top of the started application's stack to
 *                                capture non-fatal validation issues.
 * Assumptions   : A bus fault handler is installed that can be used to handle
 *                 any detected uncorrectable ECC errors that are found during
 *                 validation of the application.
 * ------------------------------------------------------------------------- */
static __INLINE BootROMStatus Sys_BootROM_StartApp(uint32_t* vect_table)
{
    return (*((BootROMStatus (**)(uint32_t*))
              ROMVECT_BOOTROM_START_APP))(vect_table);
}

/* ----------------------------------------------------------------------------
 * Close the 'extern "C"' block
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
}
#endif

#endif /* Q32M210_ROMVECT_H */

