/* ----------------------------------------------------------------------------
 * Copyright (c) 2009 - 2012 Semiconductor Components Industries, LLC
 * (d/b/a ON Semiconductor). All Rights Reserved.
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor. The
 * terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32_eeprom.h
 * - Header file for EEPROM library
 * - This library supports EEPROMs that comply with the Catalyst CAT25256
 *   instruction set. EEPROM addresses are 16-bit.
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32_EEPROM_H
#define Q32_EEPROM_H

/* ----------------------------------------------------------------------------
 * Firmware Version
 * ------------------------------------------------------------------------- */
#define EEPROM_FW_VER_MAJOR                0x02
#define EEPROM_FW_VER_MINOR                0x00
#define EEPROM_FW_VER_REVISION             0x00

#define EEPROM_FW_VER                      ((EEPROM_FW_VER_MAJOR << 12) | \
                                            (EEPROM_FW_VER_MINOR << 8)  | \
                                            (EEPROM_FW_VER_REVISION))

extern const uint16_t q32_EEPROMLib_Version;

/* ----------------------------------------------------------------------------
 * EEPROM Status Register
 * ------------------------------------------------------------------------- */

/* EEPROM_STATUS register bit positions */
#define EEPROM_STATUS_WRITEPROTECT_Pos      7
#define EEPROM_STATUS_BLOCKPROTECT_Pos      2
#define EEPROM_STATUS_WRITE_Pos             1
#define EEPROM_STATUS_STATUS_Pos            0

/* EEPROM_STATUS masks */
#define EEPROM_STATUS_WRITEPROTECT_Mask     ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_WRITEPROTECT_Pos))
#define EEPROM_STATUS_BLOCKPROTECT_Mask     ((uint32_t)(0x3U << \
                                             EEPROM_STATUS_BLOCKPROTECT_Pos))
#define EEPROM_STATUS_WRITE_Mask            ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_WRITE_Pos))
#define EEPROM_STATUS_STATUS_Mask           ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_STATUS_Pos))

/* EEPROM_STATUS settings */
#define EEPROM_WRITEPROTECT_DISABLE         ((uint32_t)(0x0U << \
                                             EEPROM_STATUS_WRITEPROTECT_Pos))
#define EEPROM_WRITEPROTECT_ENABLE          ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_WRITEPROTECT_Pos))

#define EEPROM_BLOCKPROTECT_NONE            ((uint32_t)(0x0U << \
                                             EEPROM_STATUS_BLOCKPROTECT_Pos))
#define EEPROM_BLOCKPROTECT_QUARTER         ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_BLOCKPROTECT_Pos))
#define EEPROM_BLOCKPROTECT_HALF            ((uint32_t)(0x2U << \
                                             EEPROM_STATUS_BLOCKPROTECT_Pos))
#define EEPROM_BLOCKPROTECT_ALL             ((uint32_t)(0x3U << \
                                             EEPROM_STATUS_BLOCKPROTECT_Pos))

#define EEPROM_WRITE_DISABLE                ((uint32_t)(0x0U << \
                                             EEPROM_STATUS_WRITE_Pos))
#define EEPROM_WRITE_ENABLE                 ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_WRITE_Pos))

#define EEPROM_STATUS_READY                 ((uint32_t)(0x0U << \
                                             EEPROM_STATUS_STATUS_Pos))
#define EEPROM_STATUS_BUSY                  ((uint32_t)(0x1U << \
                                             EEPROM_STATUS_STATUS_Pos))

/* ----------------------------------------------------------------------------
 * EEPROM Library Defines
 * ------------------------------------------------------------------------- */

/* Number of SPI interfaces available to the library */
#define EEPROM_SPI_NUM_INTERFACE    2

#define EEPROM_PAGE_LENGTH          0x40
#define EEPROM_PAGE_ADDRESS_Mask    ~(EEPROM_PAGE_LENGTH - 0x1)

/* Write commit delay in milliseconds */
#define DEFAULT_COMMIT_DELAY        10
/* Number of iterations to perform a write commit delay */
#define DEFAULT_COMMIT_TIMEOUT      3

/* EEPROM operation codes (opcodes) */
#define EEPROM_OPCODE_NOP                  0x00  /* Invalid opcode */
#define EEPROM_OPCODE_WRITE_STATUS         0x01  /* Write status register */
#define EEPROM_OPCODE_WRITE                0x02  /* Write data to memory 
                                                  * array */
#define EEPROM_OPCODE_READ                 0x03  /* Read data from memory 
                                                  * array */
#define EEPROM_OPCODE_WRITE_DISABLE        0x04  /* Reset write enable latch */
#define EEPROM_OPCODE_READ_STATUS          0x05  /* Read status register */
#define EEPROM_OPCODE_WRITE_ENABLE         0x06  /* Set write enable latch */

/* SPI configuration
 * The bit-fields used by the EEPROM library are identical for both SPI
 * interfaces. */
#define EEPROM_SPI_CFG             (SPI_SELECT_SPI | \
                                    SPI_OVERRUN_INT_DISABLE | \
                                    SPI_UNDERRUN_INT_DISABLE | \
                                    SPI_CONTROLLER_CM3 | \
                                    SPI_SERI_PULLUP_DISABLE | \
                                    SPI_CLK_POLARITY_INVERSE | \
                                    SPI_MODE_SELECT_MANUAL | \
                                    SPI_ENABLE)

#define EEPROM_SPI_READ_BYTE       (SPI_START | SPI_READ_DATA | \
                                    SPI_CS_0 | SPI_WORD_SIZE_8)
#define EEPROM_SPI_READ_WORD       (SPI_START | SPI_READ_DATA | \
                                    SPI_CS_0 | SPI_WORD_SIZE_32)

#define EEPROM_SPI_WRITE_BYTE      (SPI_START | SPI_WRITE_DATA | \
                                    SPI_CS_0 | SPI_WORD_SIZE_8)
#define EEPROM_SPI_WRITE_SHORT     (SPI_START | SPI_WRITE_DATA | \
                                    SPI_CS_0 | SPI_WORD_SIZE_16)
#define EEPROM_SPI_WRITE_WORD      (SPI_START | SPI_WRITE_DATA | \
                                    SPI_CS_0 | SPI_WORD_SIZE_32)

#define EEPROM_SPI_SET_IDLE        (SPI_IDLE | SPI_READ_DATA | \
                                    SPI_CS_1 | SPI_WORD_SIZE_1)

/* ----------------------------------------------------------------------------
 * EEPROM Library Prototypes
 * ------------------------------------------------------------------------- */

/* Initialization and finalization */
extern void EEPROM_Init(uint32_t spi_interface,
                        uint32_t spi_prescalar,
                        uint32_t freq_in_hertz);
extern void EEPROM_Close(uint32_t spi_interface);

/* Configuration support */
extern uint8_t EEPROM_Read_StatusReg(uint32_t spi_interface);
extern void EEPROM_Write_StatusReg(uint32_t spi_interface,
                                   uint8_t status);

extern void EEPROM_Set_WriteCommitConfig(uint32_t spi_interface,
                                         uint32_t milliseconds,
                                         uint32_t timeout);

extern uint8_t EEPROM_Get_WriteEnableOpcode(uint32_t spi_interface);
extern void EEPROM_Set_WriteEnableOpcode(uint32_t spi_interface,
                                         uint8_t opcode);

extern void EEPROM_Write_Enable(uint32_t spi_interface);
extern void EEPROM_Write_Disable(uint32_t spi_interface);

/* Low-level data transfer */
extern void EEPROM_Read_Init(uint32_t spi_interface,
                             uint16_t address);
extern void EEPROM_Write_Init(uint32_t spi_interface,
                              uint16_t address);

extern uint8_t EEPROM_Read_Byte(uint32_t spi_interface);
extern void EEPROM_Write_Byte(uint32_t spi_interface, uint8_t data);

extern uint32_t EEPROM_Read_Word(uint32_t spi_interface);
extern void EEPROM_Write_Word(uint32_t spi_interface, uint32_t data);

extern void EEPROM_Read_Done(uint32_t spi_interface);
extern void EEPROM_Write_Done(uint32_t spi_interface);

/* High-level data transfer */
extern void EEPROM_Read(uint32_t spi_interface,
                        uint16_t address,
                        uint8_t* buffer,
                        uint32_t byte_count);
extern void EEPROM_Write(uint32_t spi_interface,
                         uint16_t address,
                         uint8_t* buffer,
                         uint32_t byte_count);

#endif /* Q32_EEPROM_H */
