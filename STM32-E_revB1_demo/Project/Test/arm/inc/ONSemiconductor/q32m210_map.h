/* ----------------------------------------------------------------------------
 * Copyright (c) 2008 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m210_map.h
 * - Memory map
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead. 
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_MAP_H
#define Q32M210_MAP_H

/* ----------------------------------------------------------------------------
 * Instruction and Data Bus Memory Structures
 * ----------------------------------------------------------------------------
 * ROM */
#define Q32M210_ROM_BASE              0x00000000
#define Q32M210_ROM_TOP               0x00001FFF
#define Q32M210_ROM_SIZE              (Q32M210_ROM_TOP - Q32M210_ROM_BASE + 1)

/* Flash Memory */
#define Q32M210_FLASH_BASE            0x00040000
#define Q32M210_FLASH_TOP             0x0007FFFF
#define Q32M210_FLASH_SIZE            (Q32M210_FLASH_TOP - \
                                       Q32M210_FLASH_BASE + 1)

#define Q32M210_FLASH_INFO_BASE       0x10000000
#define Q32M210_FLASH_INFO_TOP        0x100007FF
#define Q32M210_FLASH_INFO_SIZE       (Q32M210_FLASH_INFO_TOP - \
                                       Q32M210_FLASH_INFO_BASE + 1)


/* ----------------------------------------------------------------------------
 * System Bus Memory Structures
 * ----------------------------------------------------------------------------
 * RAM (SRAM) */
#define Q32M210_RAM_BASE                0x20000000
#define Q32M210_RAM_TOP                 0x2000BFFF
#define Q32M210_RAM_SIZE                (Q32M210_RAM_TOP - \
                                         Q32M210_RAM_BASE + 1)

#define Q32M210_RAM_BITBAND_BASE        0x22000000

/* RAM (USB SRAM Endpoint Buffers) */
#define USB_BULK_IN0_BUF_BASE           0x40001700
#define USB_BULK_OUT0_BUF_BASE          0x400016C0
#define USB_BULK_IN2_BUF_BASE           0x40001600
#define USB_BULK_IN3_BUF_BASE           0x40001580
#define USB_BULK_OUT4_BUF_BASE          0x400014C0
#define USB_BULK_OUT5_BUF_BASE          0x40001440

/* ----------------------------------------------------------------------------
 * Peripheral Bus Memory-Mapped Control Registers
 * ------------------------------------------------------------------------- */
#define Q32M210_PERIPHERAL_BASE         0x40000000
#define Q32M210_PERIPHERAL_BITBAND_BASE 0x42000000

#define Q32M210_PERIPHERAL_AFE_BASE      (Q32M210_PERIPHERAL_BASE + 0x0000)
#define Q32M210_PERIPHERAL_DMA_BASE      (Q32M210_PERIPHERAL_BASE + 0x0200)
#define Q32M210_PERIPHERAL_UART0_BASE    (Q32M210_PERIPHERAL_BASE + 0x0300)
#define Q32M210_PERIPHERAL_UART1_BASE    (Q32M210_PERIPHERAL_BASE + 0x0400)
#define Q32M210_PERIPHERAL_GPIO_LCD_BASE (Q32M210_PERIPHERAL_BASE + 0x0500)
#define Q32M210_PERIPHERAL_SPI0_BASE     (Q32M210_PERIPHERAL_BASE + 0x0600)
#define Q32M210_PERIPHERAL_SPI1_BASE     (Q32M210_PERIPHERAL_BASE + 0x0700)
#define Q32M210_PERIPHERAL_PCM_BASE      (Q32M210_PERIPHERAL_BASE + 0x0800)
#define Q32M210_PERIPHERAL_I2C_BASE      (Q32M210_PERIPHERAL_BASE + 0x0900)
#define Q32M210_PERIPHERAL_CRC_BASE      (Q32M210_PERIPHERAL_BASE + 0x0A00)
#define Q32M210_PERIPHERAL_USB_CTRL_BASE (Q32M210_PERIPHERAL_BASE + 0x0B00)
#define Q32M210_PERIPHERAL_TIMER_BASE    (Q32M210_PERIPHERAL_BASE + 0x0C00)
#define Q32M210_PERIPHERAL_WATCHDOG_BASE (Q32M210_PERIPHERAL_BASE + 0x0D00)
#define Q32M210_PERIPHERAL_CLK_CTRL_BASE (Q32M210_PERIPHERAL_BASE + 0x0E00)
#define Q32M210_PERIPHERAL_FLASH_BASE    (Q32M210_PERIPHERAL_BASE + 0x0F00)
#define Q32M210_PERIPHERAL_USB_BASE      (Q32M210_PERIPHERAL_BASE + 0x1340)

/* ----------------------------------------------------------------------------
 * Private Peripheral Bus Internal Memory-Mapped Control Registers
 * ------------------------------------------------------------------------- */
#define Q32M210_PRIVATE_PERIPHERAL_BASE     0xE0000000

#define Q32M210_PRIVATE_PERIPHERAL_SYS_BASE (Q32M210_PRIVATE_PERIPHERAL_BASE + \
                                             0xE000)


/* ----------------------------------------------------------------------------
 * System Variables
 * ------------------------------------------------------------------------- */
#define Q32M210_VAR_SYSTEM_ID_BASE    (0x1FFFFFFC)
#define Q32M210_VAR_SYSTEM_ID         (*((uint32_t const *) \
                                          Q32M210_VAR_SYSTEM_ID_BASE))

#define Q32M210_VAR_SYSTEM_ID_FAMILY  (*((uint8_t const *) \
                                         (Q32M210_VAR_SYSTEM_ID_BASE + 3)))
#define Q32M210_VAR_SYSTEM_ID_VERSION (*((uint8_t const *) \
                                         (Q32M210_VAR_SYSTEM_ID_BASE + 2)))
#define Q32M210_VAR_SYSTEM_ID_REVISION (*((uint16_t const *) \
                                           Q32M210_VAR_SYSTEM_ID_BASE))
#define Q32M210_VAR_SYSTEM_ID_REVISION_MAJOR (*((uint8_t const *) \
                                                (Q32M210_VAR_SYSTEM_ID_BASE + \
                                                 1)))
#define Q32M210_VAR_SYSTEM_ID_REVISION_MINOR (*((uint8_t const *) \
                                                 Q32M210_VAR_SYSTEM_ID_BASE))

#define Q32M210_VAR_BOOTROM_ERROR     (*((uint32_t *) Q32M210_RAM_BASE))

#endif /* Q32M210_MAP_H */

