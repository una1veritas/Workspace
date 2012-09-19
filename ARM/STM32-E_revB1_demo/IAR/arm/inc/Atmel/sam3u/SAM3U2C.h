/* ---------------------------------------------------------------------------- */
/*                  Atmel Microcontroller Software Support                      */
/* ---------------------------------------------------------------------------- */
/* Copyright (c) 2011, Atmel Corporation                                        */
/*                                                                              */
/* All rights reserved.                                                         */
/*                                                                              */
/* Redistribution and use in source and binary forms, with or without           */
/* modification, are permitted provided that the following condition is met:    */
/*                                                                              */
/* - Redistributions of source code must retain the above copyright notice,     */
/* this list of conditions and the disclaimer below.                            */
/*                                                                              */
/* Atmel's name may not be used to endorse or promote products derived from     */
/* this software without specific prior written permission.                     */
/*                                                                              */
/* DISCLAIMER:  THIS SOFTWARE IS PROVIDED BY ATMEL "AS IS" AND ANY EXPRESS OR   */
/* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT ARE   */
/* DISCLAIMED. IN NO EVENT SHALL ATMEL BE LIABLE FOR ANY DIRECT, INDIRECT,      */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT */
/* LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,  */
/* OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    */
/* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING         */
/* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, */
/* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                           */
/* ---------------------------------------------------------------------------- */

#ifndef SAM3U2C_H
#define SAM3U2C_H

/** \addtogroup SAM3U2C_definitions SAM3U2C definitions
  This file defines all structures and symbols for SAM3U2C:
    - registers and bitfields
    - peripheral base address
    - peripheral ID
    - PIO definitions
*/
/*@{*/

#ifdef __cplusplus
 extern "C" {
#endif 

#ifndef __IAR_SYSTEMS_ASM__
#include <stdint.h>
#ifndef __cplusplus
typedef volatile const uint32_t RoReg; /**< Read only 32-bit register (volatile const unsigned int) */
#else
typedef volatile       uint32_t RoReg; /**< Read only 32-bit register (volatile const unsigned int) */
#endif
typedef volatile       uint32_t WoReg; /**< Write only 32-bit register (volatile unsigned int) */
typedef volatile       uint32_t RwReg; /**< Read-Write 32-bit register (volatile unsigned int) */
#define CAST(type, value) ((type *)(value))
#define REG_ACCESS(type, address) (*(type*)(address)) /**< C code: Register value */
#else
#define CAST(type, value) (value) 
#define REG_ACCESS(type, address) (address) /**< Assembly code: Register address */
#endif

/* ************************************************************************** */
/*   CMSIS DEFINITIONS FOR SAM3U2C */
/* ************************************************************************** */
/** \addtogroup SAM3U2C_cmsis CMSIS Definitions */
/*@{*/

/**< Interrupt Number Definition */
typedef enum IRQn
{
/******  Cortex-M3 Processor Exceptions Numbers ******************************/
  NonMaskableInt_IRQn   = -14, /**<  2 Non Maskable Interrupt                */
  MemoryManagement_IRQn = -12, /**<  4 Cortex-M3 Memory Management Interrupt */
  BusFault_IRQn         = -11, /**<  5 Cortex-M3 Bus Fault Interrupt         */
  UsageFault_IRQn       = -10, /**<  6 Cortex-M3 Usage Fault Interrupt       */
  SVCall_IRQn           = -5,  /**< 11 Cortex-M3 SV Call Interrupt           */
  DebugMonitor_IRQn     = -4,  /**< 12 Cortex-M3 Debug Monitor Interrupt     */
  PendSV_IRQn           = -2,  /**< 14 Cortex-M3 Pend SV Interrupt           */
  SysTick_IRQn          = -1,  /**< 15 Cortex-M3 System Tick Interrupt       */
/******  SAM3U2C specific Interrupt Numbers *********************************/
  
  SUPC_IRQn            =  0, /**<  0 SAM3U2C Supply Controller (SUPC) */
  RSTC_IRQn            =  1, /**<  1 SAM3U2C Reset Controller (RSTC) */
  RTC_IRQn             =  2, /**<  2 SAM3U2C Real Time Clock (RTC) */
  RTT_IRQn             =  3, /**<  3 SAM3U2C Real Time Timer (RTT) */
  WDT_IRQn             =  4, /**<  4 SAM3U2C Watchdog Timer (WDT) */
  PMC_IRQn             =  5, /**<  5 SAM3U2C Power Management Controller (PMC) */
  EFC0_IRQn            =  6, /**<  6 SAM3U2C Enhanced Embedded Flash Controller 0 (EFC0) */
  EFC1_IRQn            =  7, /**<  7 SAM3U2C Enhanced Embedded Flash Controller 1 (EFC1) */
  UART_IRQn            =  8, /**<  8 SAM3U2C Universal Asynchronous Receiver Transmitter (UART) */
  SMC_IRQn             =  9, /**<  9 SAM3U2C Static Memory Controller (SMC) */
  PIOA_IRQn            = 10, /**< 10 SAM3U2C Parallel I/O Controller A, (PIOA) */
  PIOB_IRQn            = 11, /**< 11 SAM3U2C Parallel I/O Controller B (PIOB) */
  PIOC_IRQn            = 12, /**< 12 SAM3U2C Parallel I/O Controller C (PIOC) */
  USART0_IRQn          = 13, /**< 13 SAM3U2C USART 0 (USART0) */
  USART1_IRQn          = 14, /**< 14 SAM3U2C USART 1 (USART1) */
  USART2_IRQn          = 15, /**< 15 SAM3U2C USART 2 (USART2) */
  USART3_IRQn          = 16, /**< 16 SAM3U2C USART 3 (USART3) */
  HSMCI_IRQn           = 17, /**< 17 SAM3U2C High Speed Multimedia Card Interface (HSMCI) */
  TWI0_IRQn            = 18, /**< 18 SAM3U2C Two-Wire Interface 0 (TWI0) */
  TWI1_IRQn            = 19, /**< 19 SAM3U2C Two-Wire Interface 1 (TWI1) */
  SPI_IRQn             = 20, /**< 20 SAM3U2C Serial Peripheral Interface (SPI) */
  SSC_IRQn             = 21, /**< 21 SAM3U2C Synchronous Serial Controller (SSC) */
  TC0_IRQn             = 22, /**< 22 SAM3U2C Timer Counter 0 (TC0) */
  TC1_IRQn             = 23, /**< 23 SAM3U2C Timer Counter 1 (TC1) */
  TC2_IRQn             = 24, /**< 24 SAM3U2C Timer Counter 2 (TC2) */
  PWM_IRQn             = 25, /**< 25 SAM3U2C Pulse Width Modulation Controller (PWM) */
  ADC12B_IRQn          = 26, /**< 26 SAM3U2C 12-bit ADC Controller (ADC12B) */
  ADC_IRQn             = 27, /**< 27 SAM3U2C 10-bit ADC Controller (ADC) */
  DMAC_IRQn            = 28, /**< 28 SAM3U2C DMA Controller (DMAC) */
  UDPHS_IRQn           = 29  /**< 29 SAM3U2C USB Device High Speed (UDPHS) */
} IRQn_Type;

/**
 * \brief Configuration of the Cortex-M3 Processor and Core Peripherals 
 */

#define __MPU_PRESENT          1 /**< SAM3U2C does provide a MPU */
#define __NVIC_PRIO_BITS       4 /**< SAM3U2C uses 4 Bits for the Priority Levels */
#define __Vendor_SysTickConfig 0 /**< Set to 1 if different SysTick Config is used */

/*
 * \brief CMSIS includes
 */

#include <core_cm3.h>

/*@}*/

/* ************************************************************************** */
/**  SOFTWARE PERIPHERAL API DEFINITION FOR SAM3U2C */
/* ************************************************************************** */
/** \addtogroup SAM3U2C_api Peripheral Software API */
/*@{*/

#include "component/ADC.h"
#include "component/ADC12B.h"
#include "component/CHIPID.h"
#include "component/DMAC.h"
#include "component/EFC.h"
#include "component/GPBR.h"
#include "component/HSMCI.h"
#include "component/MATRIX.h"
#include "component/PDC.h"
#include "component/PIO.h"
#include "component/PMC.h"
#include "component/PWM.h"
#include "component/RSTC.h"
#include "component/RTC.h"
#include "component/RTT.h"
#include "component/SMC.h"
#include "component/SPI.h"
#include "component/SSC.h"
#include "component/SUPC.h"
#include "component/TC.h"
#include "component/TWI.h"
#include "component/UART.h"
#include "component/UDPHS.h"
#include "component/USART.h"
#include "component/WDT.h"
/*@}*/

/* ************************************************************************** */
/*   REGISTER ACCESS DEFINITIONS FOR SAM3U2C */
/* ************************************************************************** */
/** \addtogroup SAM3U2C_reg Registers Access Definitions */
/*@{*/

#include "instance/HSMCI.h"
#include "instance/SSC.h"
#include "instance/SPI.h"
#include "instance/TC0.h"
#include "instance/TWI0.h"
#include "instance/TWI1.h"
#include "instance/PWM.h"
#include "instance/USART0.h"
#include "instance/USART1.h"
#include "instance/USART2.h"
#include "instance/UDPHS.h"
#include "instance/ADC12B.h"
#include "instance/ADC.h"
#include "instance/DMAC.h"
#include "instance/SMC.h"
#include "instance/MATRIX.h"
#include "instance/PMC.h"
#include "instance/UART.h"
#include "instance/CHIPID.h"
#include "instance/EFC0.h"
#include "instance/EFC1.h"
#include "instance/PIOA.h"
#include "instance/PIOB.h"
#include "instance/RSTC.h"
#include "instance/SUPC.h"
#include "instance/RTT.h"
#include "instance/WDT.h"
#include "instance/RTC.h"
#include "instance/GPBR.h"
/*@}*/

/* ************************************************************************** */
/*   PERIPHERAL ID DEFINITIONS FOR SAM3U2C */
/* ************************************************************************** */
/** \addtogroup SAM3U2C_id Peripheral Ids Definitions */
/*@{*/

#define ID_SUPC   ( 0) /**< \brief Supply Controller (SUPC) */
#define ID_RSTC   ( 1) /**< \brief Reset Controller (RSTC) */
#define ID_RTC    ( 2) /**< \brief Real Time Clock (RTC) */
#define ID_RTT    ( 3) /**< \brief Real Time Timer (RTT) */
#define ID_WDT    ( 4) /**< \brief Watchdog Timer (WDT) */
#define ID_PMC    ( 5) /**< \brief Power Management Controller (PMC) */
#define ID_EFC0   ( 6) /**< \brief Enhanced Embedded Flash Controller 0 (EFC0) */
#define ID_EFC1   ( 7) /**< \brief Enhanced Embedded Flash Controller 1 (EFC1) */
#define ID_UART   ( 8) /**< \brief Universal Asynchronous Receiver Transmitter (UART) */
#define ID_SMC    ( 9) /**< \brief Static Memory Controller (SMC) */
#define ID_PIOA   (10) /**< \brief Parallel I/O Controller A, (PIOA) */
#define ID_PIOB   (11) /**< \brief Parallel I/O Controller B (PIOB) */
#define ID_PIOC   (12) /**< \brief Parallel I/O Controller C (PIOC) */
#define ID_USART0 (13) /**< \brief USART 0 (USART0) */
#define ID_USART1 (14) /**< \brief USART 1 (USART1) */
#define ID_USART2 (15) /**< \brief USART 2 (USART2) */
#define ID_USART3 (16) /**< \brief USART 3 (USART3) */
#define ID_HSMCI  (17) /**< \brief High Speed Multimedia Card Interface (HSMCI) */
#define ID_TWI0   (18) /**< \brief Two-Wire Interface 0 (TWI0) */
#define ID_TWI1   (19) /**< \brief Two-Wire Interface 1 (TWI1) */
#define ID_SPI    (20) /**< \brief Serial Peripheral Interface (SPI) */
#define ID_SSC    (21) /**< \brief Synchronous Serial Controller (SSC) */
#define ID_TC0    (22) /**< \brief Timer Counter 0 (TC0) */
#define ID_TC1    (23) /**< \brief Timer Counter 1 (TC1) */
#define ID_TC2    (24) /**< \brief Timer Counter 2 (TC2) */
#define ID_PWM    (25) /**< \brief Pulse Width Modulation Controller (PWM) */
#define ID_ADC12B (26) /**< \brief 12-bit ADC Controller (ADC12B) */
#define ID_ADC    (27) /**< \brief 10-bit ADC Controller (ADC) */
#define ID_DMAC   (28) /**< \brief DMA Controller (DMAC) */
#define ID_UDPHS  (29) /**< \brief USB Device High Speed (UDPHS) */
/*@}*/

/* ************************************************************************** */
/*   BASE ADDRESS DEFINITIONS FOR SAM3U2C */
/* ************************************************************************** */
/** \addtogroup SAM3U2C_base Peripheral Base Address Definitions */
/*@{*/

#define HSMCI      CAST(Hsmci     , 0x40000000U) /**< \brief (HSMCI     ) Base Address */
#define SSC        CAST(Ssc       , 0x40004000U) /**< \brief (SSC       ) Base Address */
#define SPI        CAST(Spi       , 0x40008000U) /**< \brief (SPI       ) Base Address */
#define TC0        CAST(Tc        , 0x40080000U) /**< \brief (TC0       ) Base Address */
#define TWI0       CAST(Twi       , 0x40084000U) /**< \brief (TWI0      ) Base Address */
#define PDC_TWI0   CAST(Pdc       , 0x40084100U) /**< \brief (PDC_TWI0  ) Base Address */
#define TWI1       CAST(Twi       , 0x40088000U) /**< \brief (TWI1      ) Base Address */
#define PDC_TWI1   CAST(Pdc       , 0x40088100U) /**< \brief (PDC_TWI1  ) Base Address */
#define PWM        CAST(Pwm       , 0x4008C000U) /**< \brief (PWM       ) Base Address */
#define PDC_PWM    CAST(Pdc       , 0x4008C100U) /**< \brief (PDC_PWM   ) Base Address */
#define USART0     CAST(Usart     , 0x40090000U) /**< \brief (USART0    ) Base Address */
#define PDC_USART0 CAST(Pdc       , 0x40090100U) /**< \brief (PDC_USART0) Base Address */
#define USART1     CAST(Usart     , 0x40094000U) /**< \brief (USART1    ) Base Address */
#define PDC_USART1 CAST(Pdc       , 0x40094100U) /**< \brief (PDC_USART1) Base Address */
#define USART2     CAST(Usart     , 0x40098000U) /**< \brief (USART2    ) Base Address */
#define PDC_USART2 CAST(Pdc       , 0x40098100U) /**< \brief (PDC_USART2) Base Address */
#define UDPHS      CAST(Udphs     , 0x400A4000U) /**< \brief (UDPHS     ) Base Address */
#define ADC12B     CAST(Adc12b    , 0x400A8000U) /**< \brief (ADC12B    ) Base Address */
#define PDC_ADC12B CAST(Pdc       , 0x400A8100U) /**< \brief (PDC_ADC12B) Base Address */
#define ADC        CAST(Adc       , 0x400AC000U) /**< \brief (ADC       ) Base Address */
#define PDC_ADC    CAST(Pdc       , 0x400AC100U) /**< \brief (PDC_ADC   ) Base Address */
#define DMAC       CAST(Dmac      , 0x400B0000U) /**< \brief (DMAC      ) Base Address */
#define SMC        CAST(Smc       , 0x400E0000U) /**< \brief (SMC       ) Base Address */
#define MATRIX     CAST(Matrix    , 0x400E0200U) /**< \brief (MATRIX    ) Base Address */
#define PMC        CAST(Pmc       , 0x400E0400U) /**< \brief (PMC       ) Base Address */
#define UART       CAST(Uart      , 0x400E0600U) /**< \brief (UART      ) Base Address */
#define PDC_UART   CAST(Pdc       , 0x400E0700U) /**< \brief (PDC_UART  ) Base Address */
#define CHIPID     CAST(Chipid    , 0x400E0740U) /**< \brief (CHIPID    ) Base Address */
#define EFC0       CAST(Efc       , 0x400E0800U) /**< \brief (EFC0      ) Base Address */
#define EFC1       CAST(Efc       , 0x400E0A00U) /**< \brief (EFC1      ) Base Address */
#define PIOA       CAST(Pio       , 0x400E0C00U) /**< \brief (PIOA      ) Base Address */
#define PIOB       CAST(Pio       , 0x400E0E00U) /**< \brief (PIOB      ) Base Address */
#define RSTC       CAST(Rstc      , 0x400E1200U) /**< \brief (RSTC      ) Base Address */
#define SUPC       CAST(Supc      , 0x400E1210U) /**< \brief (SUPC      ) Base Address */
#define RTT        CAST(Rtt       , 0x400E1230U) /**< \brief (RTT       ) Base Address */
#define WDT        CAST(Wdt       , 0x400E1250U) /**< \brief (WDT       ) Base Address */
#define RTC        CAST(Rtc       , 0x400E1260U) /**< \brief (RTC       ) Base Address */
#define GPBR       CAST(Gpbr      , 0x400E1290U) /**< \brief (GPBR      ) Base Address */
/*@}*/

/* ************************************************************************** */
/*   PIO DEFINITIONS FOR SAM3U2C */
/* ************************************************************************** */
/** \addtogroup SAM3U2C_pio Peripheral Pio Definitions */
/*@{*/

#include "pio/SAM3U2C.h"
/*@}*/

/* ************************************************************************** */
/*   MEMORY MAPPING DEFINITIONS FOR SAM3U2C */
/* ************************************************************************** */

#define IFLASH_SIZE 0x20000
#define IRAM_SIZE   0x9000

#define IFLASH0_ADDR   (0x00080000u) /**< Internal Flash 0 base address */
#if defined IFLASH0_SIZE
#define IFLASH1_ADDR   (IFLASH0_ADDR+IFLASH0_SIZE) /**< Internal Flash 1 base address */
#endif
#define IROM_ADDR      (0x00180000u) /**< Internal ROM base address */
#define IRAM0_ADDR     (0x20000000u) /**< Internal RAM 0 base address */
#define IRAM1_ADDR     (0x20080000u) /**< Internal RAM 1 base address */
#define NFC_RAM_ADDR   (0x20100000u) /**< NAND Flash Controller RAM base address */
#define UDPHS_RAM_ADDR (0x20180000u) /**< USB High Speed Device Port RAM base address */

#ifdef __cplusplus
}
#endif

/*@}*/

#endif /* SAM3U2C_H */
