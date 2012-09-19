/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Fujitsu MB9AF132K
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 49339 $
 **
 ***************************************************************************/

#ifndef __IOMB9AF132K_H
#define __IOMB9AF132K_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   MB9AF132K SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/


/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 0
#error This file should only be compiled in little endian mode
#endif

/* Interrupt Controller Type Register */
typedef struct {
  __REG32  INTLINESNUM    : 5;
  __REG32                 :27;
} __nvic_bits;

/* SysTick Control and Status Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  TICKINT        : 1;
  __REG32  CLKSOURCE      : 1;
  __REG32                 :13;
  __REG32  COUNTFLAG      : 1;
  __REG32                 :15;
} __systickcsr_bits;

/* SysTick Reload Value Register */
typedef struct {
  __REG32  RELOAD         :24;
  __REG32                 : 8;
} __systickrvr_bits;

/* SysTick Current Value Register */
typedef struct {
  __REG32  CURRENT        :24;
  __REG32                 : 8;
} __systickcvr_bits;

/* SysTick Calibration Value Register */
typedef struct {
  __REG32  TENMS          :24;
  __REG32                 : 6;
  __REG32  SKEW           : 1;
  __REG32  NOREF          : 1;
} __systickcalvr_bits;

/* Interrupt Set-Enable Registers 0-31 */
typedef struct {
  __REG32  SETENA0        : 1;
  __REG32  SETENA1        : 1;
  __REG32  SETENA2        : 1;
  __REG32  SETENA3        : 1;
  __REG32  SETENA4        : 1;
  __REG32  SETENA5        : 1;
  __REG32  SETENA6        : 1;
  __REG32  SETENA7        : 1;
  __REG32  SETENA8        : 1;
  __REG32  SETENA9        : 1;
  __REG32  SETENA10       : 1;
  __REG32  SETENA11       : 1;
  __REG32  SETENA12       : 1;
  __REG32  SETENA13       : 1;
  __REG32  SETENA14       : 1;
  __REG32  SETENA15       : 1;
  __REG32  SETENA16       : 1;
  __REG32  SETENA17       : 1;
  __REG32  SETENA18       : 1;
  __REG32  SETENA19       : 1;
  __REG32  SETENA20       : 1;
  __REG32  SETENA21       : 1;
  __REG32  SETENA22       : 1;
  __REG32  SETENA23       : 1;
  __REG32  SETENA24       : 1;
  __REG32  SETENA25       : 1;
  __REG32  SETENA26       : 1;
  __REG32  SETENA27       : 1;
  __REG32  SETENA28       : 1;
  __REG32  SETENA29       : 1;
  __REG32  SETENA30       : 1;
  __REG32  SETENA31       : 1;
} __setena0_bits;

/* Interrupt Clear-Enable Registers 0-31 */
typedef struct {
  __REG32  CLRENA0        : 1;
  __REG32  CLRENA1        : 1;
  __REG32  CLRENA2        : 1;
  __REG32  CLRENA3        : 1;
  __REG32  CLRENA4        : 1;
  __REG32  CLRENA5        : 1;
  __REG32  CLRENA6        : 1;
  __REG32  CLRENA7        : 1;
  __REG32  CLRENA8        : 1;
  __REG32  CLRENA9        : 1;
  __REG32  CLRENA10       : 1;
  __REG32  CLRENA11       : 1;
  __REG32  CLRENA12       : 1;
  __REG32  CLRENA13       : 1;
  __REG32  CLRENA14       : 1;
  __REG32  CLRENA15       : 1;
  __REG32  CLRENA16       : 1;
  __REG32  CLRENA17       : 1;
  __REG32  CLRENA18       : 1;
  __REG32  CLRENA19       : 1;
  __REG32  CLRENA20       : 1;
  __REG32  CLRENA21       : 1;
  __REG32  CLRENA22       : 1;
  __REG32  CLRENA23       : 1;
  __REG32  CLRENA24       : 1;
  __REG32  CLRENA25       : 1;
  __REG32  CLRENA26       : 1;
  __REG32  CLRENA27       : 1;
  __REG32  CLRENA28       : 1;
  __REG32  CLRENA29       : 1;
  __REG32  CLRENA30       : 1;
  __REG32  CLRENA31       : 1;
} __clrena0_bits;

/* Interrupt Set-Pending Register 0-31 */
typedef struct {
  __REG32  SETPEND0       : 1;
  __REG32  SETPEND1       : 1;
  __REG32  SETPEND2       : 1;
  __REG32  SETPEND3       : 1;
  __REG32  SETPEND4       : 1;
  __REG32  SETPEND5       : 1;
  __REG32  SETPEND6       : 1;
  __REG32  SETPEND7       : 1;
  __REG32  SETPEND8       : 1;
  __REG32  SETPEND9       : 1;
  __REG32  SETPEND10      : 1;
  __REG32  SETPEND11      : 1;
  __REG32  SETPEND12      : 1;
  __REG32  SETPEND13      : 1;
  __REG32  SETPEND14      : 1;
  __REG32  SETPEND15      : 1;
  __REG32  SETPEND16      : 1;
  __REG32  SETPEND17      : 1;
  __REG32  SETPEND18      : 1;
  __REG32  SETPEND19      : 1;
  __REG32  SETPEND20      : 1;
  __REG32  SETPEND21      : 1;
  __REG32  SETPEND22      : 1;
  __REG32  SETPEND23      : 1;
  __REG32  SETPEND24      : 1;
  __REG32  SETPEND25      : 1;
  __REG32  SETPEND26      : 1;
  __REG32  SETPEND27      : 1;
  __REG32  SETPEND28      : 1;
  __REG32  SETPEND29      : 1;
  __REG32  SETPEND30      : 1;
  __REG32  SETPEND31      : 1;
} __setpend0_bits;

/* Interrupt Clear-Pending Register 0-31 */
typedef struct {
  __REG32  CLRPEND0       : 1;
  __REG32  CLRPEND1       : 1;
  __REG32  CLRPEND2       : 1;
  __REG32  CLRPEND3       : 1;
  __REG32  CLRPEND4       : 1;
  __REG32  CLRPEND5       : 1;
  __REG32  CLRPEND6       : 1;
  __REG32  CLRPEND7       : 1;
  __REG32  CLRPEND8       : 1;
  __REG32  CLRPEND9       : 1;
  __REG32  CLRPEND10      : 1;
  __REG32  CLRPEND11      : 1;
  __REG32  CLRPEND12      : 1;
  __REG32  CLRPEND13      : 1;
  __REG32  CLRPEND14      : 1;
  __REG32  CLRPEND15      : 1;
  __REG32  CLRPEND16      : 1;
  __REG32  CLRPEND17      : 1;
  __REG32  CLRPEND18      : 1;
  __REG32  CLRPEND19      : 1;
  __REG32  CLRPEND20      : 1;
  __REG32  CLRPEND21      : 1;
  __REG32  CLRPEND22      : 1;
  __REG32  CLRPEND23      : 1;
  __REG32  CLRPEND24      : 1;
  __REG32  CLRPEND25      : 1;
  __REG32  CLRPEND26      : 1;
  __REG32  CLRPEND27      : 1;
  __REG32  CLRPEND28      : 1;
  __REG32  CLRPEND29      : 1;
  __REG32  CLRPEND30      : 1;
  __REG32  CLRPEND31      : 1;
} __clrpend0_bits;

/* Active Bit Register 0-31 */
typedef struct {
  __REG32  ACTIVE0        : 1;
  __REG32  ACTIVE1        : 1;
  __REG32  ACTIVE2        : 1;
  __REG32  ACTIVE3        : 1;
  __REG32  ACTIVE4        : 1;
  __REG32  ACTIVE5        : 1;
  __REG32  ACTIVE6        : 1;
  __REG32  ACTIVE7        : 1;
  __REG32  ACTIVE8        : 1;
  __REG32  ACTIVE9        : 1;
  __REG32  ACTIVE10       : 1;
  __REG32  ACTIVE11       : 1;
  __REG32  ACTIVE12       : 1;
  __REG32  ACTIVE13       : 1;
  __REG32  ACTIVE14       : 1;
  __REG32  ACTIVE15       : 1;
  __REG32  ACTIVE16       : 1;
  __REG32  ACTIVE17       : 1;
  __REG32  ACTIVE18       : 1;
  __REG32  ACTIVE19       : 1;
  __REG32  ACTIVE20       : 1;
  __REG32  ACTIVE21       : 1;
  __REG32  ACTIVE22       : 1;
  __REG32  ACTIVE23       : 1;
  __REG32  ACTIVE24       : 1;
  __REG32  ACTIVE25       : 1;
  __REG32  ACTIVE26       : 1;
  __REG32  ACTIVE27       : 1;
  __REG32  ACTIVE28       : 1;
  __REG32  ACTIVE29       : 1;
  __REG32  ACTIVE30       : 1;
  __REG32  ACTIVE31       : 1;
} __active0_bits;

/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32  PRI_0          : 8;
  __REG32  PRI_1          : 8;
  __REG32  PRI_2          : 8;
  __REG32  PRI_3          : 8;
} __pri0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32  PRI_4          : 8;
  __REG32  PRI_5          : 8;
  __REG32  PRI_6          : 8;
  __REG32  PRI_7          : 8;
} __pri1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32  PRI_8          : 8;
  __REG32  PRI_9          : 8;
  __REG32  PRI_10         : 8;
  __REG32  PRI_11         : 8;
} __pri2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32  PRI_12         : 8;
  __REG32  PRI_13         : 8;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __pri3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32  PRI_16         : 8;
  __REG32  PRI_17         : 8;
  __REG32  PRI_18         : 8;
  __REG32  PRI_19         : 8;
} __pri4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32  PRI_20         : 8;
  __REG32  PRI_21         : 8;
  __REG32  PRI_22         : 8;
  __REG32  PRI_23         : 8;
} __pri5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32  PRI_24         : 8;
  __REG32  PRI_25         : 8;
  __REG32  PRI_26         : 8;
  __REG32  PRI_27         : 8;
} __pri6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32  PRI_28         : 8;
  __REG32  PRI_29         : 8;
  __REG32  PRI_30         : 8;
  __REG32  PRI_31         : 8;
} __pri7_bits;

/* CPU ID Base Register */
typedef struct {
  __REG32  REVISION       : 4;
  __REG32  PARTNO         :12;
  __REG32                 : 4;
  __REG32  VARIANT        : 4;
  __REG32  IMPLEMENTER    : 8;
} __cpuidbr_bits;

/* Interrupt Control State Register */
typedef struct {
  __REG32  VECTACTIVE     :10;
  __REG32                 : 1;
  __REG32  RETTOBASE      : 1;
  __REG32  VECTPENDING    :10;
  __REG32  ISRPENDING     : 1;
  __REG32  ISRPREEMPT     : 1;
  __REG32                 : 1;
  __REG32  PENDSTCLR      : 1;
  __REG32  PENDSTSET      : 1;
  __REG32  PENDSVCLR      : 1;
  __REG32  PENDSVSET      : 1;
  __REG32                 : 2;
  __REG32  NMIPENDSET     : 1;
} __icsr_bits;

/* Vector Table Offset Register */
typedef struct {
  __REG32                 : 7;
  __REG32  TBLOFF         :22;
  __REG32  TBLBASE        : 1;
  __REG32                 : 2;
} __vtor_bits;

/* Application Interrupt and Reset Control Register */
typedef struct {
  __REG32  VECTRESET      : 1;
  __REG32  VECTCLRACTIVE  : 1;
  __REG32  SYSRESETREQ    : 1;
  __REG32                 : 5;
  __REG32  PRIGROUP       : 3;
  __REG32                 : 4;
  __REG32  ENDIANESS      : 1;
  __REG32  VECTKEY        :16;
} __aircr_bits;

/* System Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32  SLEEPONEXIT    : 1;
  __REG32  SLEEPDEEP      : 1;
  __REG32                 : 1;
  __REG32  SEVONPEND      : 1;
  __REG32                 :27;
} __scr_bits;

/* Configuration Control Register */
typedef struct {
  __REG32  NONEBASETHRDENA: 1;
  __REG32  USERSETMPEND   : 1;
  __REG32                 : 1;
  __REG32  UNALIGN_TRP    : 1;
  __REG32  DIV_0_TRP      : 1;
  __REG32                 : 3;
  __REG32  BFHFNMIGN      : 1;
  __REG32  STKALIGN       : 1;
  __REG32                 :22;
} __ccr_bits;

/* System Handler Control and State Register */
typedef struct {
  __REG32  MEMFAULTACT    : 1;
  __REG32  BUSFAULTACT    : 1;
  __REG32                 : 1;
  __REG32  USGFAULTACT    : 1;
  __REG32                 : 3;
  __REG32  SVCALLACT      : 1;
  __REG32  MONITORACT     : 1;
  __REG32                 : 1;
  __REG32  PENDSVACT      : 1;
  __REG32  SYSTICKACT     : 1;
  __REG32                 : 1;
  __REG32  MEMFAULTPENDED : 1;
  __REG32  BUSFAULTPENDED : 1;
  __REG32  SVCALLPENDED   : 1;
  __REG32  MEMFAULTENA    : 1;
  __REG32  BUSFAULTENA    : 1;
  __REG32  USGFAULTENA    : 1;
  __REG32                 :13;
} __shcsr_bits;

/* Configurable Fault Status Registers */
typedef struct {
  __REG32  IACCVIOL       : 1;
  __REG32  DACCVIOL       : 1;
  __REG32                 : 1;
  __REG32  MUNSTKERR      : 1;
  __REG32  MSTKERR        : 1;
  __REG32                 : 2;
  __REG32  MMARVALID      : 1;
  __REG32  IBUSERR        : 1;
  __REG32  PRECISERR      : 1;
  __REG32  IMPRECISERR    : 1;
  __REG32  UNSTKERR       : 1;
  __REG32  STKERR         : 1;
  __REG32                 : 2;
  __REG32  BFARVALID      : 1;
  __REG32  UNDEFINSTR     : 1;
  __REG32  INVSTATE       : 1;
  __REG32  INVPC          : 1;
  __REG32  NOCP           : 1;
  __REG32                 : 4;
  __REG32  UNALIGNED      : 1;
  __REG32  DIVBYZERO      : 1;
  __REG32                 : 6;
} __cfsr_bits;

/* Hard Fault Status Register */
typedef struct {
  __REG32                 : 1;
  __REG32  VECTTBL        : 1;
  __REG32                 :28;
  __REG32  FORCED         : 1;
  __REG32  DEBUGEVT       : 1;
} __hfsr_bits;

/* Debug Fault Status Register */
typedef struct {
  __REG32  HALTED         : 1;
  __REG32  BKPT           : 1;
  __REG32  DWTTRAP        : 1;
  __REG32  VCATCH         : 1;
  __REG32  EXTERNAL       : 1;
  __REG32                 :27;
} __dfsr_bits;

/* Software Trigger Interrupt Register */
typedef struct {
  __REG32  INTID          : 9;
  __REG32                 :23;
} __stir_bits;

/* FASZR (Flash Access Size Register) */
typedef struct {
  __REG32  ASZ            : 2;
  __REG32                 :30;
} __faszr_bits;

/* FSTR (Flash Status Register) */
typedef struct {
  __REG32  RDY            : 1;
  __REG32  HNG            : 1;
  __REG32                 :30;
} __fstr_bits;

/* FSYNDN (Flash Sync Down Register) */
typedef struct {
  __REG32  SD             : 3;
  __REG32                 :29;
} __fsyndn_bits;

/* CRTRMM (CR Trimming Data Mirror Register) */
typedef struct {
  __REG32  TRMM           :10;
  __REG32                 :22;
} __crtrmm_bits;

/* System Clock Mode Control Register (SCM_CTL) */
typedef struct {
  __REG32                 : 1;
  __REG32  MOSCE          : 1;
  __REG32                 : 1;
  __REG32  SOSCE          : 1;
  __REG32  PLLE           : 1;
  __REG32  RCS            : 3;
  __REG32                 :24;
} __scm_ctl_bits;

/* System Clock Mode Status Register (SCM_STR) */
typedef struct {
  __REG32                 : 1;
  __REG32  MORDY          : 1;
  __REG32                 : 1;
  __REG32  SORDY          : 1;
  __REG32  PLRDY          : 1;
  __REG32  RCM            : 3;
  __REG32                 :24;
} __scm_str_bits;

/* Standby Mode Control Register (STB_CTL) */
typedef struct {
  __REG32  STM            : 2;
  __REG32  DSTM           : 1;
  __REG32                 : 1;
  __REG32  SPL            : 1;
  __REG32                 :11;
  __REG32  KEY            :16;
} __stb_ctl_bits;

/* Reset Cause Register (RST_STR) */
typedef struct {
  __REG32  PONR           : 1;
  __REG32  INITX          : 1;
  __REG32                 : 1;
  __REG32  LVDH           : 1;
  __REG32  SWDT           : 1;
  __REG32  HWDT           : 1;
  __REG32  CSVR           : 1;
  __REG32  FCSR           : 1;
  __REG32  SRST           : 1;
  __REG32                 :23;
} __rst_str_bits;

/* Base Clock Prescaler Register (BSC_PSR) */
typedef struct {
  __REG32  BSR            : 3;
  __REG32                 :29;
} __bsc_psr_bits;

/* APB0 Prescaler Register (APBC0_PSR) */
typedef struct {
  __REG32  APBC0          : 2;
  __REG32                 :30;
} __apbc0_psr_bits;

/* APB1 Prescaler Register (APBC1_PSR) */
typedef struct {
  __REG32  APBC1          : 2;
  __REG32                 : 2;
  __REG32  APBC1RST       : 1;
  __REG32                 : 2;
  __REG32  APBC1EN        : 1;
  __REG32                 :24;
} __apbc1_psr_bits;

/* APB2 Prescaler Register (APBC2_PSR) */
typedef struct {
  __REG32  APBC2          : 2;
  __REG32                 : 2;
  __REG32  APBC2RST       : 1;
  __REG32                 : 2;
  __REG32  APBC2EN        : 1;
  __REG32                 :24;
} __apbc2_psr_bits;

/* Software Watchdog Clock Prescaler Register (SWC_PSR) */
typedef struct {
  __REG32  SWDS           : 2;
  __REG32                 : 5;
  __REG32  TESTB          : 1;
  __REG32                 :24;
} __swc_psr_bits;

/* Clock Stabilization Wait Time Register (CSW_TMR) */
typedef struct {
  __REG32  MOWT           : 4;
  __REG32  SOWT           : 3;
  __REG32                 :25;
} __csw_tmr_bits;

/* PLL Clock Stabilization Wait Time Setup Register (PSW_TMR) */
typedef struct {
  __REG32  POWT           : 3;
  __REG32                 : 1;
  __REG32  PINC           : 1;
  __REG32                 :27;
} __psw_tmr_bits;

/* PLL Control Register 1 (PLL_CTL1) */
typedef struct {
  __REG32  PLLM           : 4;
  __REG32  PLLK           : 4;
  __REG32                 :24;
} __pll_ctl1_bits;

/* PLL Control Register 2 (PLL_CTL2) */
typedef struct {
  __REG32  PLLN           : 6;
  __REG32                 :26;
} __pll_ctl2_bits;

/* CSV control register (CSV_CTL) */
typedef struct {
  __REG32  MCSVE          : 1;
  __REG32  SCSVE          : 1;
  __REG32                 : 6;
  __REG32  FCSDE          : 1;
  __REG32  FCSRE          : 1;
  __REG32                 : 2;
  __REG32  FCD            : 3;
  __REG32                 :17;
} __csv_ctl_bits;

/* CSV status register (CSV_STR) */
typedef struct {
  __REG32  MCMF           : 1;
  __REG32  SCMF           : 1;
  __REG32                 :30;
} __csv_str_bits;

/* Frequency detection window setting register (Upper)(FCSWH_CTL) */
typedef struct {
  __REG32  FWH            :16;
  __REG32                 :16;
} __fcswh_ctl_bits;

/* Frequency detection window setting register (Lower)(FCSWL_CTL) */
typedef struct {
  __REG32  FWL            :16;
  __REG32                 :16;
} __fcswl_ctl_bits;

/* Frequency detection counter register (FCSWD_CTL) */
typedef struct {
  __REG32  FWD            :16;
  __REG32                 :16;
} __fcswd_ctl_bits;

/* Debug Break Watchdog Timer Control Register (DBWDT_CTL) */
typedef struct {
  __REG32                 : 5;
  __REG32  DPSWBE         : 1;
  __REG32                 : 1;
  __REG32  DPHWBE         : 1;
  __REG32                 :24;
} __dbwdt_ctl_bits;

/* Interrupt Enable Register (INT_ENR) */
typedef struct {
  __REG32  MCSE           : 1;
  __REG32  SCSE           : 1;
  __REG32  PCSE           : 1;
  __REG32                 : 2;
  __REG32  FCSE           : 1;
  __REG32                 :26;
} __int_enr_bits;

/* Interrupt Status Register (INT_STR) */
typedef struct {
  __REG32  MCSI           : 1;
  __REG32  SCSI           : 1;
  __REG32  PCSI           : 1;
  __REG32                 : 2;
  __REG32  FCSI           : 1;
  __REG32                 :26;
} __int_str_bits;

/* Interrupt Clear Register (INT_CLR) */
typedef struct {
  __REG32  MCSC           : 1;
  __REG32  SCSC           : 1;
  __REG32  PCSC           : 1;
  __REG32                 : 2;
  __REG32  FCSC           : 1;
  __REG32                 :26;
} __int_clr_bits; 

/* Software Watchdog Timer Control Register (WdogControl) */
/* Hardware Watchdog Timer Control Register (WDG_CTL) */
typedef struct {
  __REG32  INTEN          : 1;
  __REG32  RESEN          : 1;
  __REG32                 :30;
} __wdg_ctl_bits;

/* Software Watchdog Timer Interrupt Status Register (WdogRIS) */
/* Hardware Watchdog Timer Interrupt Status Register (WDG_RIS) */
typedef struct {
  __REG32  RIS            : 1;
  __REG32                 :31;
} __wdg_ris_bits;

/* OCSA10 (OCU Control Register A OCU ch1 and OCU ch0) */
typedef struct {
  __REG8   CST0           : 1;
  __REG8   CST1           : 1;
  __REG8   BDIS0          : 1;
  __REG8   BDIS1          : 1;
  __REG8   IOE0           : 1;
  __REG8   IOE1           : 1;
  __REG8   IOP0           : 1;
  __REG8   IOP1           : 1;
} __mft_ocsa10_bits;

/* OCSB10 (OCU Control Register B OCU ch1 and OCU ch0) */
typedef struct {
  __REG8   OTD0           : 1;
  __REG8   OTD1           : 1;
  __REG8                  : 2;
  __REG8   CMOD           : 1;
  __REG8   BTS0           : 1;
  __REG8   BTS1           : 1;
  __REG8                  : 1;
} __mft_ocsb10_bits;

/* OCSA32 (OCU Control Register A OCU ch3 and OCU ch2) */
typedef struct {
  __REG8   CST2           : 1;
  __REG8   CST3           : 1;
  __REG8   BDIS2          : 1;
  __REG8   BDIS3          : 1;
  __REG8   IOE2           : 1;
  __REG8   IOE3           : 1;
  __REG8   IOP2           : 1;
  __REG8   IOP3           : 1;
} __mft_ocsa32_bits;

/* OCSB32 (OCU Control Register B OCU ch3 and OCU ch2) */
typedef struct {
  __REG8   OTD2           : 1;
  __REG8   OTD3           : 1;
  __REG8                  : 2;
  __REG8   CMOD           : 1;
  __REG8   BTS2           : 1;
  __REG8   BTS3           : 1;
  __REG8                  : 1;
} __mft_ocsb32_bits;

/* OCSA54 (OCU Control Register A OCU ch5 and OCU ch4) */
typedef struct {
  __REG8   CST4           : 1;
  __REG8   CST5           : 1;
  __REG8   BDIS4          : 1;
  __REG8   BDIS5          : 1;
  __REG8   IOE4           : 1;
  __REG8   IOE5           : 1;
  __REG8   IOP4           : 1;
  __REG8   IOP5           : 1;
} __mft_ocsa54_bits;

/* OCSB54 (OCU Control Register B OCU ch5 and OCU ch4) */
typedef struct {
  __REG8   OTD4           : 1;
  __REG8   OTD5           : 1;
  __REG8                  : 2;
  __REG8   CMOD           : 1;
  __REG8   BTS4           : 1;
  __REG8   BTS5           : 1;
  __REG8                  : 1;
} __mft_ocsb54_bits;

/* OCSC (OCU Control Register C) */
typedef struct {
  __REG8   MOD0           : 1;
  __REG8   MOD1           : 1;
  __REG8   MOD2           : 1;
  __REG8   MOD3           : 1;
  __REG8   MOD4           : 1;
  __REG8   MOD5           : 1;
  __REG8                  : 2;
} __mft_ocsc_bits;

/* TCSA0, TCSA1, TCSA2 (FRT Control Register A) */
typedef struct {
  __REG16  CLK            : 4;
  __REG16  SCLR           : 1;
  __REG16  MODE           : 1;
  __REG16  STOP           : 1;
  __REG16  BFE            : 1;
  __REG16  ICRE           : 1;
  __REG16  ICLR           : 1;
  __REG16                 : 3;
  __REG16  IRQZE          : 1;
  __REG16  IRQZF          : 1;
  __REG16  ECKE           : 1;
} __mft_tcsax_bits;

/* TCSB0, TCSB1, TCSB2 (FRT Control Register B) */
typedef struct {
  __REG16  AD0E           : 1;
  __REG16  AD1E           : 1;
  __REG16                 :14;
} __mft_tcsbx_bits;

/* OCFS10 (OCU Connecting FRT Select Register ch1 and OCU ch0) */
typedef struct {
  __REG8   FSO0           : 4;
  __REG8   FSO1           : 4;
} __mft_ocfs10_bits;

/* OCFS32 (OCU Connecting FRT Select Register ch3 and OCU ch2) */
typedef struct {
  __REG8   FSO2           : 4;
  __REG8   FSO3           : 4;
} __mft_ocfs32_bits;

/* OCFS54 (OCU Connecting FRT Select Register ch5 and OCU ch4) */
typedef struct {
  __REG8   FSO4           : 4;
  __REG8   FSO5           : 4;
} __mft_ocfs54_bits;

/* ICFS10 (ICU Connecting FRT Select Register ch1 and ICU ch0) */
typedef struct {
  __REG8   FSI0           : 4;
  __REG8   FSI1           : 4;
} __mft_icfs10_bits;

/* ICFS32 (ICU Connecting FRT Select Register ch3 and ICU ch2) */
typedef struct {
  __REG8   FSI2           : 4;
  __REG8   FSI3           : 4;
} __mft_icfs32_bits;

/* ICSA10 (ICU Control Register A ch1 and ICU ch0) */
typedef struct {
  __REG8   EG0            : 2;
  __REG8   EG1            : 2;
  __REG8   ICE0           : 1;
  __REG8   ICE1           : 1;
  __REG8   ICP0           : 1;
  __REG8   ICP1           : 1;
} __mft_icsa10_bits;

/* ICSB10 (ICU Control Register B ICU ch1 and ICU ch0) */
typedef struct {
  __REG8   IEI0           : 1;
  __REG8   IEI1           : 1;
  __REG8                  : 6;
} __mft_icsb10_bits;

/* ICSA32 (ICU Control Register A ch3 and ICU ch2) */
typedef struct {
  __REG8   EG2            : 2;
  __REG8   EG3            : 2;
  __REG8   ICE2           : 1;
  __REG8   ICE3           : 1;
  __REG8   ICP2           : 1;
  __REG8   ICP3           : 1;
} __mft_icsa32_bits;

/* ICSB32 (ICU Control Register B ICU ch3 and ICU ch2) */
typedef struct {
  __REG8   IEI2           : 1;
  __REG8   IEI3           : 1;
  __REG8                  : 6;
} __mft_icsb32_bits;

/* WFSA10, WFSA32, WFSA54 (WFG Control Register A) */
typedef struct {
  __REG16  DCK            : 3;
  __REG16  TMD            : 3;
  __REG16  GTEN           : 2;
  __REG16  PSEL           : 2;
  __REG16  PGEN           : 2;
  __REG16  DMOD           : 1;
  __REG16                 : 3;
} __mft_wfsa_bits;

/* WFIR (WFG Interrupt Control Register) */
typedef struct {
  __REG16  DTIF           : 1;
  __REG16  DTIC           : 1;
  __REG16                 : 2;
  __REG16  TMIF10         : 1;
  __REG16  TMIC10         : 1;
  __REG16  TMIE10         : 1;
  __REG16  TMIS10         : 1;
  __REG16  TMIF32         : 1;
  __REG16  TMIC32         : 1;
  __REG16  TMIE32         : 1;
  __REG16  TMIS32         : 1;
  __REG16  TMIF54         : 1;
  __REG16  TMIC54         : 1;
  __REG16  TMIE54         : 1;
  __REG16  TMIS54         : 1;
} __mft_wfir_bits;

/* NZCL (NZCL Control Register) */
typedef struct {
  __REG16  DTIE           : 1;
  __REG16  NWS            : 3;
  __REG16  SDTI           : 1;
  __REG16                 :11;
} __mft_nzcl_bits;

/* ACSB (ADCMP Control Register B) */
typedef struct {
  __REG16  BDIS0          : 1;
  __REG16  BDIS1          : 1;
  __REG16                 : 2;
  __REG16  BTS0           : 1;
  __REG16  BTS1           : 1;
  __REG16                 :10;
} __mft_acsb_bits;

/* ACSB (ADCMP Control Register B) */
typedef struct {
  __REG16  CE0            : 2;
  __REG16  CE1            : 2;
  __REG16                 : 4;
  __REG16  SEL0           : 2;
  __REG16  SEL1           : 2;
  __REG16                 : 4;
} __mft_acsa_bits;

/* ATSA (ADC Start Trigger Select Register) */
typedef struct {
  __REG16  AD0S           : 2;
  __REG16  AD1S           : 2;
  __REG16                 : 4;
  __REG16  AD0P           : 2;
  __REG16  AD1P           : 2;
  __REG16                 : 4;
} __mft_atsa_bits; 

/* PPG Start Trigger Control Register 0 (TTCR0) */
typedef struct {
  __REG8   STR0           : 1;
  __REG8   MONI0          : 1;
  __REG8   CS0            : 2;
  __REG8   TRG0O          : 1;
  __REG8   TRG2O          : 1;
  __REG8   TRG4O          : 1;
  __REG8   TRG6O          : 1;
} __ppg_ttcr0_bits;

/* PPG Start Register (TRG) */
typedef struct {
  __REG16  PEN00          : 1;
  __REG16  PEN01          : 1;
  __REG16  PEN02          : 1;
  __REG16  PEN03          : 1;
  __REG16  PEN04          : 1;
  __REG16  PEN05          : 1;
  __REG16  PEN06          : 1;
  __REG16  PEN07          : 1;
  __REG16                 : 8;
} __ppg_trg_bits;

/* Output Reverse Register (REVC) */
typedef struct {
  __REG16  REV00          : 1;
  __REG16  REV01          : 1;
  __REG16  REV02          : 1;
  __REG16  REV03          : 1;
  __REG16  REV04          : 1;
  __REG16  REV05          : 1;
  __REG16  REV06          : 1;
  __REG16  REV07          : 1;
  __REG16                 : 8;
} __ppg_revc_bits;

/* PPG Operation Mode Control Register (PPGC) */
typedef struct {
  __REG8   TTRG           : 1;
  __REG8   MD             : 2;
  __REG8   PCS            : 2;
  __REG8   INTM           : 1;
  __REG8   PUF            : 1;
  __REG8   PIE            : 1;
} __ppg_ppgcx_bits;

/* PPG Reload Registers (PRLH, PRLL) */
typedef union {
  /*PPG_PRLx*/
  struct {
    __REG16   PRLL        : 8;
    __REG16   PRLH        : 8;
  };                    
  struct {              
    __REG8    __byte0 ;
    __REG8    __byte1 ;
  };
}__ppg_prlx_bits;

/* PPG Gate Function Control Registers (GATEC0) */
typedef struct {
  __REG8   EDGE0          : 1;
  __REG8   STRG0          : 1;
  __REG8                  : 2;
  __REG8   EDGE2          : 1;
  __REG8   STRG2          : 1;
  __REG8                  : 2;
} __ppg_gatec0_bits;

/* PPG Gate Function Control Registers (GATEC4) */
typedef struct {
  __REG8   EDGE4          : 1;
  __REG8   STRG4          : 1;
  __REG8                  : 2;
  __REG8   EDGE6          : 1;
  __REG8   STRG6          : 1;
  __REG8                  : 2;
} __ppg_gatec4_bits;

/* Timer Control Registers (BTxTMCR) */
typedef union {
    /* BTyx_PPG_TMCR */
    /* BTyx_PWM_TMCR */
    struct {
      __REG16  STRG           : 1;
      __REG16  CTEN           : 1;
      __REG16  MDSE           : 1;
      __REG16  OSEL           : 1;
      __REG16  FMD            : 3;
      __REG16                 : 1;
      __REG16  EGS            : 2;
      __REG16  PMSK           : 1;
      __REG16  RTGEN          : 1;
      __REG16  CKS0           : 1;
      __REG16  CKS1           : 1;
      __REG16  CKS2           : 1;
      __REG16                 : 1;
    };
    /* BTyx_RT_TMCR */
    struct {
      __REG16  STRG           : 1;
      __REG16  CTEN           : 1;
      __REG16  MDSE           : 1;
      __REG16  OSEL           : 1;
      __REG16  FMD            : 3;
      __REG16  T32            : 1;
      __REG16  EGS            : 2;
      __REG16                 : 2;
      __REG16  CKS0           : 1;
      __REG16  CKS1           : 1;
      __REG16  CKS2           : 1;
      __REG16                 : 1;
    } RT;
    /* BTyx_PWC_TMCR */    
    struct {
      __REG16                 : 1;
      __REG16  CTEN           : 1;
      __REG16  MDSE           : 1;
      __REG16                 : 1;
      __REG16  FMD            : 3;
      __REG16  T32            : 1;
      __REG16  EGS            : 3;
      __REG16                 : 1;
      __REG16  CKS0           : 1;
      __REG16  CKS1           : 1;
      __REG16  CKS2           : 1;
      __REG16                 : 1;
    } PWC;
} __btxtmcr_bits;

/* Status Control Register (STC) */
typedef union {
    /* BTyx_PPG_STC */
    /* BTyx_RT_STC */
    struct {
      __REG8   UDIR           : 1;
      __REG8                  : 1;
      __REG8   TGIR           : 1;
      __REG8                  : 1;
      __REG8   UDIE           : 1;
      __REG8                  : 1;
      __REG8   TGIE           : 1;
      __REG8                  : 1;
    };
    /* BTyx_PWM_STC */
    struct {
      __REG8   UDIR           : 1;
      __REG8   DTIR           : 1;
      __REG8   TGIR           : 1;
      __REG8                  : 1;
      __REG8   UDIE           : 1;
      __REG8   DTIE           : 1;
      __REG8   TGIE           : 1;
      __REG8                  : 1;
    } PWM;
    /* BTyx_PWC_STC */
    struct {
      __REG8   OVIR           : 1;
      __REG8                  : 1;
      __REG8   EDIR           : 1;
      __REG8                  : 1;
      __REG8   OVIE           : 1;
      __REG8                  : 1;
      __REG8   EDIE           : 1;
      __REG8   ERR            : 1;
    } PWC;
} __btxstc_bits;

/* Timer Control Register 2 (TMCR2) */
typedef struct {
  __REG8   CKS3           : 1;
  __REG8                  : 7;
} __btxtmcr2_bits;

/* I/O Select Register (BTSEL0123) */
typedef struct {
  __REG8   SEL01          : 4;
  __REG8   SEL23          : 4;
} __btsel0123_bits;

/* I/O Select Register (BTSEL4567) */
typedef struct {
  __REG8   SEL45          : 4;
  __REG8   SEL67          : 4;
} __btsel4567_bits;

/* Software-based Simultaneous Startup Register (BTSSSR) */
typedef struct {
  __REG16  SSSR0          : 1;
  __REG16  SSSR1          : 1;
  __REG16  SSSR2          : 1;
  __REG16  SSSR3          : 1;
  __REG16  SSSR4          : 1;
  __REG16  SSSR5          : 1;
  __REG16  SSSR6          : 1;
  __REG16  SSSR7          : 1;
  __REG16  SSSR8          : 1;
  __REG16  SSSR9          : 1;
  __REG16  SSSR10         : 1;
  __REG16  SSSR11         : 1;
  __REG16  SSSR12         : 1;
  __REG16  SSSR13         : 1;
  __REG16  SSSR14         : 1;
  __REG16  SSSR15         : 1;
} __btsssr_bits;

/* A/D Status Register (ADSR) */
typedef struct {
  __REG8   SCS            : 1;
  __REG8   PCS            : 1;
  __REG8   PCNS           : 1;
  __REG8                  : 3;
  __REG8   FDAS           : 1;
  __REG8   ADSTP          : 1;
} __adc_adsr_bits;

/* A/D Control Register (ADCR) */
typedef struct {
  __REG8   OVRIE          : 1;
  __REG8   CMPIE          : 1;
  __REG8   PCIE           : 1;
  __REG8   SCIE           : 1;
  __REG8                  : 1;
  __REG8   CMPIF          : 1;
  __REG8   PCIF           : 1;
  __REG8   SCIF           : 1;
} __adc_adcr_bits;

/* Scan Conversion FIFO Stage Count Setup Register (SFNS) */
typedef struct {
  __REG8   SFS            : 4;
  __REG8                  : 4;
} __adc_sfns_bits;

/* Scan Conversion Control Register (SCCR) */
typedef struct {
  __REG8   SSTR           : 1;
  __REG8   SHEN           : 1;
  __REG8   RPT            : 1;
  __REG8                  : 1;
  __REG8   SFCLR          : 1;
  __REG8   SOVR           : 1;
  __REG8   SFUL           : 1;
  __REG8   SEMP           : 1;
} __adc_sccr_bits;

/* Scan Conversion FIFO Data Register (SCFD) */
typedef struct {
  __REG32  SC             : 5;
  __REG32                 : 3;
  __REG32  RS             : 2;
  __REG32                 : 2;
  __REG32  INVL           : 1;
  __REG32                 : 3;
  __REG32  SD             :16;
} __adc_scfd_bits;

/* Scan Conversion Input Selection Register 0 (SCIS0) */
typedef struct {
  __REG8   AN0            : 1;
  __REG8   AN1            : 1;
  __REG8   AN2            : 1;
  __REG8   AN3            : 1;
  __REG8   AN4            : 1;
  __REG8   AN5            : 1;
  __REG8                  : 2;
} __adc_scis0_bits;

/* Priority Conversion FIFO Stage Count Setup Register (PFNS) */
typedef struct {
  __REG8   PFS            : 2;
  __REG8                  : 2;
  __REG8   TEST           : 2;
  __REG8                  : 2;
} __adc_pfns_bits;

/* Priority Conversion Control Register (PCCR) */
typedef struct {
  __REG8   PSTR           : 1;
  __REG8   PHEN           : 1;
  __REG8   PEEN           : 1;
  __REG8   ESCE           : 1;
  __REG8   PFCLR          : 1;
  __REG8   POVR           : 1;
  __REG8   PFUL           : 1;
  __REG8   PEMP           : 1;
} __adc_pccr_bits;

/* Priority Conversion FIFO Data Register (PCFD) */
typedef struct {
  __REG32  PC             : 5;
  __REG32                 : 3;
  __REG32  RS             : 3;
  __REG32                 : 1;
  __REG32  INVL           : 1;
  __REG32                 : 3;
  __REG32  PD             :16;
} __adc_pcfd_bits;

/* Priority Conversion Input Selection Register (PCIS) */
typedef struct {
  __REG8   P1A            : 3;
  __REG8   P2A            : 5;
} __adc_pcis_bits;

/* A/D Comparison Control Register (CMPCR) */
typedef struct {
  __REG8   CCH            : 5;
  __REG8   CMD0           : 1;
  __REG8   CMD1           : 1;
  __REG8   CMPEN          : 1;
} __adc_cmpcr_bits;

/* A/D Comparison Control Register (CMPD) */
typedef struct {
  __REG16                 : 4;
  __REG16  CMAD           :12;
} __adc_cmpd_bits;

/* Sampling Time Selection Register 0 (ADSS0) */
typedef struct {
  __REG8   TS0            : 1;
  __REG8   TS1            : 1;
  __REG8   TS2            : 1;
  __REG8   TS3            : 1;
  __REG8   TS4            : 1;
  __REG8   TS5            : 1;
  __REG8                  : 1;
  __REG8   TS7            : 1;
} __adc_adss0_bits;

/* Sampling Time Selection Register 1 (ADSS1) */
typedef struct {
  __REG8   TS8            : 1;
  __REG8                  : 7;
} __adc_adss1_bits;

/* Sampling Time Setup Register 0 (ADST0) */
typedef struct {
  __REG8   ST0            : 5;
  __REG8   STX0           : 3;
} __adc_adst0_bits;

/* Sampling Time Setup Register 0 (ADST0) */
typedef struct {
  __REG8   ST1            : 5;
  __REG8   STX1           : 3;
} __adc_adst1_bits;

/* Comparison Time Setup Register (ADCT) */
typedef struct {
  __REG8   CT   : 8;
} __adc_adct_bits;

/* A/D Operation Enable Setup Register (ADCEN) */
typedef struct {
  __REG16  ENBL           : 1;
  __REG16  READY          : 1;
  __REG16                 : 6;
  __REG16  ENBLTIME       : 8;
} __adc_adcen_bits;

/* High-speed CR oscillation Frequency Division Setup Register (MCR_PSR) */
typedef struct {
  __REG8   CSR            : 3;
  __REG8                  : 5;
} __mcr_psr_bits;

/* High-speed CR oscillation Frequency Trimming Register (MCR_FTRM) */
typedef struct {
  __REG16  TRD            :10;
  __REG16                 : 6;
} __mcr_ftrm_bits;

/* Enable Interrupt Request Register [ENIR] */
typedef struct {
  __REG16  EN0            : 1;
  __REG16  EN1            : 1;
  __REG16  EN2            : 1;
  __REG16  EN3            : 1;
  __REG16                 : 2;
  __REG16  EN6            : 1;
  __REG16                 : 8;
  __REG16  EN15           : 1;
} __enir_bits;

/* External Interrupt Request Register [EIRR] */
typedef struct {
  __REG16  ER0            : 1;
  __REG16  ER1            : 1;
  __REG16  ER2            : 1;
  __REG16  ER3            : 1;
  __REG16                 : 2;
  __REG16  ER6            : 1;
  __REG16                 : 8;
  __REG16  ER15           : 1;
} __eirr_bits;

/* External Interrupt Clear Register [EICL] */
typedef struct {
  __REG16  ECL0           : 1;
  __REG16  ECL1           : 1;
  __REG16  ECL2           : 1;
  __REG16  ECL3           : 1;
  __REG16                 : 2;
  __REG16  ECL6           : 1;
  __REG16                 : 8;
  __REG16  ECL15          : 1;
} __eicl_bits;

/* External Interrupt Level Register [ELVR] */
typedef struct {
  __REG32  LA0            : 1;
  __REG32  LB0            : 1;
  __REG32  LA1            : 1;
  __REG32  LB1            : 1;
  __REG32  LA2            : 1;
  __REG32  LB2            : 1;
  __REG32  LA3            : 1;
  __REG32  LB3            : 1;
  __REG32                 : 4;
  __REG32  LA6            : 1;
  __REG32  LB6            : 1;
  __REG32                 :16;
  __REG32  LA15           : 1;
  __REG32  LB15           : 1;
} __elvr_bits;

/* Non Maskable Interrupt Request Register [NMIRR] */
typedef struct {
  __REG16  NR0            : 1;
  __REG16                 :15;
} __nmirr_bits;

/* Non Maskable Interrupt Clear Register [NMICL] */
typedef struct {
  __REG16  NCL0           : 1;
  __REG16                 :15;
} __nmicl_bits;

/* EXC02 Batch Read Register (EXC02MON) EXC02MON indicates */
typedef struct {
  __REG32  NMI            : 1;
  __REG32  HWINT          : 1;
  __REG32                 :30;
} __exc02mon_bits;

/* IRQ00 Batch Read Register (IRQ00MON) */
typedef struct {
  __REG32  FCSINT         : 1;
  __REG32                 :31;
} __irqmon0_bits;

/* IRQ01 Batch Read Register (IRQ01MON) */
typedef struct {
  __REG32  SWWDTINT       : 1;
  __REG32                 :31;
} __irqmon1_bits;

/* IRQ02 Batch Read Register (IRQ02MON) */
typedef struct {
  __REG32  LVDINT         : 1;
  __REG32                 :31;
} __irqmon2_bits;

/* IRQ03 Batch Read Register (IRQ03MON) */
typedef struct {
  __REG32  WAVE0INT       : 4;
  __REG32                 :28;
} __irqmon3_bits;

/* IRQ04 Batch Read Register (IRQ04MON) */
typedef struct {
  __REG32  EXTINT00       : 1;
  __REG32  EXTINT01       : 1;
  __REG32  EXTINT02       : 1;
  __REG32  EXTINT03       : 1;
  __REG32                 : 2;
  __REG32  EXTINT06       : 1;
  __REG32                 :25;
} __irqmon4_bits;

/* IRQ05 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32                 : 7;
  __REG32  EXTINT15       : 1;
  __REG32                 :24;
} __irqmon5_bits;

/* IRQ06/IRQ08/10/12/14/16/18/20 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  MFSINT         : 1;
  __REG32                 :31;
} __irqmon6_bits;

/* IRQ07/09/11/13/15/17/19/21 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  MFSINT         : 2;
  __REG32                 :30;
} __irqmon7_bits;

/* IRQ22 Batch Read Register (IRQ22MON) */
typedef struct {
  __REG32  PPG0INT        : 1;
  __REG32  PPG2INT        : 1;
  __REG32  PPG4INT        : 1;
  __REG32                 :29;
} __irqmon22_bits;

/* IRQ23 Batch Read Register (IRQ23MON) */
typedef struct {
  __REG32  MOSCINT        : 1;
  __REG32  SOSCINT        : 1;
  __REG32  MPLLINT        : 1;
  __REG32                 : 2;
  __REG32  RTCINT         : 1;
  __REG32                 :26;
} __irqmon23_bits;

/* IRQ24 Batch Read Register (IRQ24MON) */
typedef struct {
  __REG32  ADCINT         : 4;
  __REG32                 :28;
} __irqmon24_bits;

/* IRQ25 Read Register (IRQ25MON) */
typedef struct {
  __REG32  FRT0INT        : 6;
  __REG32                 :26;
} __irqmon25_bits;

/* IRQ26 Read Register (IRQ26MON) */
typedef struct {
  __REG32  ICU0INT        : 4;
  __REG32                 :28;
} __irqmon26_bits;

/* IRQ27 Read Register (IRQ27MON) */
typedef struct {
  __REG32  OCU0INT        : 6;
  __REG32                 :26;
} __irqmon27_bits;

/* IRQ28 Batch Read Register (IRQ28MON) */
typedef struct {
  __REG32  BT0INT0        : 1;
  __REG32  BT0INT1        : 1;
  __REG32  BT1INT0        : 1;
  __REG32  BT1INT1        : 1;
  __REG32  BT2INT0        : 1;
  __REG32  BT2INT1        : 1;
  __REG32  BT3INT0        : 1;
  __REG32  BT3INT1        : 1;
  __REG32  BT4INT0        : 1;
  __REG32  BT4INT1        : 1;
  __REG32  BT5INT0        : 1;
  __REG32  BT5INT1        : 1;
  __REG32  BT6INT0        : 1;
  __REG32  BT6INT1        : 1;
  __REG32  BT7INT0        : 1;
  __REG32  BT7INT1        : 1;
  __REG32                 :16;
} __irqmon28_bits;

/* Port0 Function Setting Register (PFR0) */
/* Port0 Pull-up Setting Register (PCR0) */
/* Port0 input/output Direction Setting Register (DDR0) */
/* Port0 Input Data Register (PDIR0) */
/* Port0 Output Data Register (PDOR0) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32                :10;
  __REG32  PF            : 1;
  __REG32                :16;
} __port0_bits;

/* Port1 Function Setting Register (PFR1) */
/* Port1 Pull-up Setting Register (PCR1) */
/* Port1 input/output Direction Setting Register (DDR1) */
/* Port1 Input Data Register (PDIR1) */
/* Port1 Output Data Register (PDOR1) */
typedef struct {
  __REG32  P0           : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32  P4           : 1;
  __REG32  P5           : 1;
  __REG32               :26;
} __port1_bits;

/* Port2 Function Setting Register (PFR2) */
/* Port2 Pull-up Setting Register (PCR2) */
/* Port2 input/output Direction Setting Register (DDR2) */
/* Port2 Input Data Register (PDIR2) */
/* Port2 Output Data Register (PDOR2) */
typedef struct {
  __REG32               : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32               :28;
} __port2_bits;

/* Port3 Function Setting Register (PFR3) */
/* Port3 Pull-up Setting Register (PCR3) */
/* Port3 input/output Direction Setting Register (DDR3) */
/* Port3 Input Data Register (PDIR3) */
/* Port3 Output Data Register (PDOR3) */
typedef struct {
  __REG32               : 9;
  __REG32  P9           : 1;
  __REG32  PA           : 1;
  __REG32  PB           : 1;
  __REG32  PC           : 1;
  __REG32  PD           : 1;
  __REG32  PE           : 1;
  __REG32  PF           : 1;
  __REG32               :16;
} __port3_bits;

/* Port4 Function Setting Register (PFR4) */
/* Port4 Pull-up Setting Register (PCR4) */
/* Port4 input/output Direction Setting Register (DDR4) */
/* Port4 Input Data Register (PDIR4) */
/* Port4 Output Data Register (PDOR4) */
typedef struct {
  __REG32                : 6;
  __REG32  P6            : 1;
  __REG32  P7            : 1;
  __REG32                : 1;
  __REG32  P9            : 1;
  __REG32  PA            : 1;
  __REG32                :21;
} __port4_bits;

/* Port5 Function Setting Register (PFR5) */
/* Port5 Pull-up Setting Register (PCR5) */
/* Port5 input/output Direction Setting Register (DDR5) */
/* Port5 Input Data Register (PDIR5) */
/* Port5 Output Data Register (PDOR5) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32                :29;
} __port5_bits;

/* Port6 Function Setting Register (PFR6) */
/* Port6 Pull-up Setting Register (PCR6) */
/* Port6 input/output Direction Setting Register (DDR6) */
/* Port6 Input Data Register (PDIR6) */
/* Port6 Output Data Register (PDOR6) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32                :30;
} __port6_bits;

/* Port8 Function Setting Register (PFR8) */
/* Port8 input/output Direction Setting Register (DDR8) */
/* Port8 Input Data Register (PDIR8) */
/* Port8 Output Data Register (PDOR8) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32                :29;
} __port8_bits;

/* PortE Function Setting Register (PFRE) */
/* PortE input/output Direction Setting Register (DDRE) */
/* PortE Input Data Register (PDIRE) */
/* PortE Output Data Register (PDORE) */
typedef struct {
  __REG32  P0            : 1;
  __REG32                : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32                :28;
} __porte_bits;

/* Analog Input Setting Register (ADE) */
typedef struct {
  __REG32  AN00           : 1;
  __REG32  AN01           : 1;
  __REG32  AN02           : 1;
  __REG32  AN03           : 1;
  __REG32  AN04           : 1;
  __REG32  AN05           : 1;
  __REG32                 :26;
} __ade_bits;

/* Special Port Setting Register (SPSR) */
typedef struct {
  __REG32  SUBXC          : 1;
  __REG32                 : 1;
  __REG32  MAINXC         : 1;
  __REG32                 :29;
} __spsr_bits;

/* Extended Pin Function Setting Register 00 (EPFR00) */
typedef struct {
  __REG32  NMIS           : 1;
  __REG32  CROUTE         : 2;
  __REG32                 : 1;
  __REG32  RTCCOE         : 2;
  __REG32  SUBOUTE        : 2;
  __REG32                 : 8;
  __REG32  JTAGEN0B       : 1;
  __REG32  JTAGEN1S       : 1;
  __REG32                 :14;
} __epfr00_bits;

/* Extended Pin Function Setting Register 01 (EPFR01) */
typedef struct {
  __REG32  RTO00E         : 2;
  __REG32  RTO01E         : 2;
  __REG32  RTO02E         : 2;
  __REG32  RTO03E         : 2;
  __REG32  RTO04E         : 2;
  __REG32  RTO05E         : 2;
  __REG32  DTTI0C         : 1;
  __REG32                 : 3;
  __REG32  DTTI0S         : 2;
  __REG32  FRCK0S         : 2;
  __REG32  IC00S          : 3;
  __REG32  IC01S          : 3;
  __REG32  IC02S          : 3;
  __REG32  IC03S          : 3;
} __epfr01_bits;

/* Extended Pin Function Setting Register 04 (EPFR04) */
typedef struct {
  __REG32                 : 2;
  __REG32  TIOA0E         : 2;
  __REG32  TIOB0S         : 2;
  __REG32                 : 2;
  __REG32  TIOA1S         : 2;
  __REG32  TIOA1E         : 2;
  __REG32  TIOB1S         : 2;
  __REG32                 : 4;
  __REG32  TIOA2E         : 2;
  __REG32  TIOB2S         : 2;
  __REG32                 : 2;
  __REG32  TIOA3S         : 2;
  __REG32  TIOA3E         : 2;
  __REG32  TIOB3S         : 2;
  __REG32                 : 2;
} __epfr04_bits;

/* Extended Pin Function Setting Register 05 (EPFR05) */
typedef struct {
  __REG32                 : 2;
  __REG32  TIOA4E         : 2;
  __REG32  TIOB4S         : 2;
  __REG32                 : 2;
  __REG32  TIOA5S         : 2;
  __REG32  TIOA5E         : 2;
  __REG32  TIOB5S         : 2;
  __REG32                 : 4;
  __REG32  TIOA6E         : 2;
  __REG32  TIOB6S         : 2;
  __REG32                 : 2;
  __REG32  TIOA7S         : 2;
  __REG32  TIOA7E         : 2;
  __REG32  TIOB7S         : 2;
  __REG32                 : 2;
} __epfr05_bits;

/* Extended Pin Function Setting Register 06 (EPFR06) */
typedef struct {
  __REG32  EINT00S        : 2;
  __REG32  EINT01S        : 2;
  __REG32  EINT02S        : 2;
  __REG32  EINT03S        : 2;
  __REG32                 : 4;
  __REG32  EINT06S        : 2;
  __REG32                 :16;
  __REG32  EINT15S        : 2;
} __epfr06_bits;

/* Extended Pin Function Setting Register 07 (EPFR07) */
typedef struct {
  __REG32                 : 4;
  __REG32  SIN0S          : 2;
  __REG32  SOT0B          : 2;
  __REG32  SCK0B          : 2;
  __REG32  SIN1S          : 2;
  __REG32  SOT1B          : 2;
  __REG32  SCK1B          : 2;
  __REG32  SIN2S          : 2;
  __REG32  SOT2B          : 2;
  __REG32  SCK2B          : 2;
  __REG32  SIN3S          : 2;
  __REG32  SOT3B          : 2;
  __REG32  SCK3B          : 2;
  __REG32                 : 4;
} __epfr07_bits;

/* Extended Pin Function Setting Register 08 (EPFR08) */
typedef struct {
  __REG32                 : 4;
  __REG32  SIN4S          : 2;
  __REG32  SOT4B          : 2;
  __REG32  SCK4B          : 2;
  __REG32  SIN5S          : 2;
  __REG32  SOT5B          : 2;
  __REG32  SCK5B          : 2;
  __REG32  SIN6S          : 2;
  __REG32  SOT6B          : 2;
  __REG32  SCK6B          : 2;
  __REG32  SIN7S          : 2;
  __REG32  SOT7B          : 2;
  __REG32  SCK7B          : 2;
  __REG32                 : 4;
} __epfr08_bits;

/* Extended Pin Function Setting Register 09 (EPFR09) */
typedef struct {
  __REG32                 :12;
  __REG32  ADTRG0S        : 4;
  __REG32                 :16;
} __epfr09_bits;

/* Low-voltage Detection Voltage Control Register (LVD_CTL) */
typedef struct {
  __REG16                 : 1;
  __REG16  LVDIM          : 1;
  __REG16  SVHI           : 4;
  __REG16                 : 1;
  __REG16  LVDIE          : 1;
  __REG16                 : 2;
  __REG16  SVHR           : 4;
  __REG16                 : 1;
  __REG16  LVDRE          : 1;
} __lvd_ctl_bits;

/* Low-voltage Detection Interrupt Register (LVD_STR) */
typedef struct {
  __REG8                  : 7;
  __REG8   LVDIR          : 1;
} __lvd_str_bits;

/* Low-voltage Detection Interrupt Clear Register (LVD_CLR) */
typedef struct {
  __REG8                  : 7;
  __REG8   LVDCL          : 1;
} __lvd_clr_bits;

/* Low-voltage Detection Circuit Status Register (LVD_STR2) */
typedef struct {
  __REG8                  : 6;
  __REG8   LVDRRDY        : 1;
  __REG8   LVDIRDY        : 1;
} __lvd_str2_bits;

/* RTC Mode Control Register (PMD_CTL) */
typedef struct {
  __REG8   RTCE           : 1;
  __REG8                  : 7;
} __pmd_ctl_bits;

/* Deep Standby Return Cause Register 1 (WRFSR) */
typedef struct {
  __REG8   WINITX         : 1;
  __REG8   WLVDH          : 1;
  __REG8                  : 6;
} __wrfsr_bits;

/* Deep Standby Return Cause Register 2 (WIFSR) */
typedef struct {
  __REG8   WRTCI          : 1;
  __REG8   WLVDI          : 1;
  __REG8   WUI0           : 1;
  __REG8   WUI1           : 1;
  __REG8   WUI2           : 1;
  __REG8   WUI3           : 1;
  __REG8                  : 2;
} __wifsr_bits;

/* Deep Standby Return Enable Register (WIER) */
typedef struct {
  __REG8   WRTCE          : 1;
  __REG8   WLVDE          : 1;
  __REG8                  : 1;
  __REG8   WUI1E          : 1;
  __REG8   WUI2E          : 1;
  __REG8   WUI3E          : 1;
  __REG8                  : 2;
} __wier_bits;

/* WKUP Pin Input Level Register (WILVR) */
typedef struct {
  __REG8   WUI1LV         : 1;
  __REG8   WUI2LV         : 1;
  __REG8   WUI3LV         : 1;
  __REG8                  : 5;
} __wilvr_bits;

/* Serial Mode Register (SMR) */
typedef union {
  /*UARTx_SMR*/
  struct {
    __REG8   SOE            : 1;
    __REG8                  : 1;
    __REG8   BDS            : 1;
    __REG8   SBL            : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  };
  /*CSIOx_SMR*/
  struct {
    __REG8   SOE            : 1;
    __REG8   SCKE           : 1;
    __REG8   BDS            : 1;
    __REG8   SCINV          : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  } CSIO;
  /*LINx_SMR*/
  struct {
    __REG8   SOE            : 1;
    __REG8                  : 2;
    __REG8   SBL            : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  } LIN;
  /*I2Cx_SMR*/
  struct {
    __REG8                  : 2;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  } I2C;
} __mfsx_smr_bits;

/* Serial Control Register (SCR)
  I2C Bus Control Register (IBCR) */
typedef union {
  /*UARTx_SCR*/
  struct {
    __REG8   TXE            : 1;
    __REG8   RXE            : 1;
    __REG8   TBIE           : 1;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8                  : 2;
    __REG8   UPCL           : 1;
  };
  /*CSIOx_SCR*/
  struct {
    __REG8   TXE            : 1;
    __REG8   RXE            : 1;
    __REG8   TBIE           : 1;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8   SPI            : 1;
    __REG8   MS             : 1;
    __REG8   UPCL           : 1;
  } CSIO;
  /*LINx_SCR*/
  struct {
    __REG8   TXE            : 1;
    __REG8   RXE            : 1;
    __REG8   TBIE           : 1;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8   LBR            : 1;
    __REG8   MS             : 1;
    __REG8   UPCL           : 1;
  } LIN;
  /*I2Cx_IBCR*/
  struct {
    __REG8   INT            : 1;
    __REG8   BER            : 1;
    __REG8   INTE           : 1;
    __REG8   CNDE           : 1;
    __REG8   WSEL           : 1;
    __REG8   ACKE           : 1;
    __REG8   ACT_SCC        : 1;
    __REG8   MSS            : 1;
  } I2C;
} __mfsx_scr_bits;

/* Extended Communication Control Register (ESCR)
   I2C Bus Status Register (IBSR) */
typedef union {
  /*UARTx_ESCR*/
  struct {
    __REG8   L              : 3;
    __REG8   P              : 1;
    __REG8   PEN            : 1;
    __REG8   INV            : 1;
    __REG8   ESBL           : 1;
    __REG8   FLWEN          : 1;
  };
  /*CSIOx_ESCR*/
  struct {
    __REG8   L              : 3;
    __REG8   WT             : 2;
    __REG8                  : 2;
    __REG8   SOP            : 1;
  } CSIO;
  /*LINx_ESCR*/
  struct {
    __REG8   DEL            : 2;
    __REG8   LBL            : 2;
    __REG8   LBIE           : 1;
    __REG8                  : 1;
    __REG8   ESBL           : 1;
    __REG8                  : 1;
  } LIN;
  /*I2Cx_IBSR*/
  struct {
    __REG8   BB             : 1;
    __REG8   SPC            : 1;
    __REG8   RSC            : 1;
    __REG8   AL             : 1;
    __REG8   TRX            : 1;
    __REG8   RSA            : 1;
    __REG8   RACK           : 1;
    __REG8   FBT            : 1;
  } I2C;
} __mfsx_escr_bits;

/* Serial Status Register (SSR) */
typedef union {
  /*UARTx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8   FRE            : 1;
    __REG8   PE             : 1;
    __REG8                  : 1;
    __REG8   REC            : 1;
  };
  /*CSIOx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8                  : 3;
    __REG8   REC            : 1;
  } CSIO;
  /*LINx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8   FRE            : 1;
    __REG8   LBD            : 1;
    __REG8                  : 1;
    __REG8   REC            : 1;
  } LIN;
  /*I2Cx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8   TBIE           : 1;
    __REG8   DMA            : 1;
    __REG8   TSET           : 1;
    __REG8   REC            : 1;
  } I2C;
} __mfsx_ssr_bits;

/* Serial Status Register (SSR) */
typedef union {
  /*UARTx_RDR*/
  /*UARTx_TDR*/
  struct {
    __REG16  D              : 9;
    __REG16                 : 7;
  };
  /*CSIOx_RDR*/
  /*CSIOx_TDR*/
  struct {
    __REG16  D              : 9;
    __REG16                 : 7;
  } CSIO;
  /*LINx_RDR*/
  /*LINx_TDR*/
  struct {
    __REG16  D              : 8;
    __REG16                 : 8;
  } LIN;
  /*I2Cx_RDR*/
  /*I2Cx_TDR*/
  struct {
    __REG16  D              : 8;
    __REG16                 : 8;
  } I2C;
} __mfsx_rdr_tdr_bits;

/* Baud Rate Generator Registers BGR */
typedef union {
  /*UARTx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16  EXT            : 1;
  };
  /*CSIOx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16                 : 1;
  } CSIO;
  /*LINx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16  EXT            : 1;
  } LIN;
  /*I2Cx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16                 : 1;
  } I2C;
} __mfsx_bgr_bits;

/* 7-bit Slave Address Register (ISBA) */
typedef  struct {
  __REG8   SA             : 7;
  __REG8   SAEN           : 1;
} __mfsx_isba_bits;

/* 7-bit Slave  Address Mask Register (ISMK) */
typedef struct {
  __REG8   SM             : 7;
  __REG8   EN             : 1;
} __mfsx_ismk_bits;   

/* FIFO Control Register (FCR) */
typedef union {
  /*UARTx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } ;
  /*CSIOx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } CSIO;
  /*LINx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } LIN;
  /*I2Cx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } I2C;
} __mfsx_fcr_bits;

/* Watch Counter Read Register (WCRD) */
typedef struct {
    __REG8   CTR            : 6;
    __REG8                  : 2;
} __wcrd_bits;

/* Watch Counter Reload Register (WCRL) */
typedef struct {
    __REG8   RLC            : 6;
    __REG8                  : 2;
} __wcrl_bits;

/* Watch Counter Control Register (WCCR) */
typedef struct {
    __REG8   WCIF           : 1;
    __REG8   WCIE           : 1;
    __REG8   CS             : 2;
    __REG8                  : 2;
    __REG8   WCOP           : 1;
    __REG8   WCEN           : 1;
} __wccr_bits;

/* Clock Selection Register (CLK_SEL) */
typedef struct {
    __REG16  SEL_IN         : 1;
    __REG16                 : 7;
    __REG16  SEL_OUT        : 1;
    __REG16                 : 7;
} __clk_sel_bits;

/* Division Clock Enable Register (CLK_EN) */
typedef struct {
    __REG8   CLK_EN         : 1;
    __REG8   CLK_EN_R       : 1;
    __REG8                  : 6;
} __clk_en_bits;

/* RTC Control Register 1 (WTCR1) */
typedef struct {
  __REG32  ST         : 1;
  __REG32             : 1;
  __REG32  RUN        : 1;
  __REG32  SRST       : 1;
  __REG32  SCST       : 1;
  __REG32  SCRST      : 1;
  __REG32  BUSY       : 1;
  __REG32             : 1;
  __REG32  MIEN       : 1;
  __REG32  HEN        : 1;
  __REG32  DEN        : 1;
  __REG32  MOEN       : 1;
  __REG32  YEN        : 1;
  __REG32             : 3;
  __REG32  INTSSI     : 1;
  __REG32  INTSI      : 1;
  __REG32  INTMI      : 1;
  __REG32  INTHI      : 1;
  __REG32  INTTMI     : 1;
  __REG32  INTALI     : 1;
  __REG32  INTERI     : 1;
  __REG32  INTCRI     : 1;
  __REG32  INTSSIE    : 1;
  __REG32  INTSIE     : 1;
  __REG32  INTMIE     : 1;
  __REG32  INTHIE     : 1;
  __REG32  INTTMIE    : 1;
  __REG32  INTALIE    : 1;
  __REG32  INTERIE    : 1;
  __REG32  INTCRIE    : 1;
} __wtcr1_bits;

/* RTC Control Register 2 (WTCR2) */
typedef struct {
  __REG32  CREAD      : 1;
  __REG32             : 7;
  __REG32  TMST       : 1;
  __REG32  TMEN       : 1;
  __REG32  TMRUN      : 1;
  __REG32             :21;
} __wtcr2_bits;

/* RTC Counter Cycle Setting Register (WTBR) */
typedef struct {
  __REG32  BR         :24;
  __REG32             : 8;
} __wtbr_bits;

/* RTC Second Register (WTSR) */
typedef struct {
  __REG8   S          : 4;
  __REG8   TS         : 3;
  __REG8              : 1;
} __wtsr_bits;

/* RTC Minute Register (WTMIR) */
typedef struct {
  __REG8   MI         : 4;
  __REG8   TMI        : 3;
  __REG8              : 1;
} __wtmir_bits;

/* RTC Hour register (WTHR) */
typedef struct {
  __REG8   H          : 4;
  __REG8   TH         : 2;
  __REG8              : 2;
} __wthr_bits;

/* RTC Date Register (WTDR) */
typedef struct {
  __REG8   D          : 4;
  __REG8   TD         : 2;
  __REG8              : 2;
} __wtdr_bits;

/* RTC Day of the Week Register (WTDW) */
typedef struct {
  __REG8   DW         : 3;
  __REG8              : 5;
} __wtdw_bits;

/* RTC Month Register (WTMOR) */
typedef struct {
  __REG8   MO         : 4;
  __REG8   TMO        : 1;
  __REG8              : 3;
} __wtmor_bits;

/* RTC Year Register (WTYR) */
typedef struct {
  __REG8   Y          : 4;
  __REG8   TY         : 4;
} __wtyr_bits;

/* RTC Alarm Minute Register (ALMIR) */
typedef struct {
  __REG8   AMI        : 4;
  __REG8   TAMI       : 3;
  __REG8              : 1;
} __almir_bits;

/* RTC Alarm Hour Register (ALHR) */
typedef struct {
  __REG8   AH         : 4;
  __REG8   TAH        : 2;
  __REG8              : 2;
} __alhr_bits;

/* RTC Alarm Date Register (ALDR) */
typedef struct {
  __REG8   AD         : 4;
  __REG8   TAD        : 2;
  __REG8              : 2;
} __aldr_bits;

/* RTC Alarm Month Register (ALMOR) */
typedef struct {
  __REG8   AMO        : 4;
  __REG8   TAMO       : 1;
  __REG8              : 3;
} __almor_bits;

/* RTC Alarm Years Register (ALYR) */
typedef struct {
  __REG8   AY         : 4;
  __REG8   TAY        : 4;
} __alyr_bits;

/* RTC Timer Setting Register (WTTR) */
typedef struct {
  __REG32  TM         :18;
  __REG32             :14;
} __wttr_bits;

/* RTC Clock Selection Register (WTCLKS) */
typedef struct {
  __REG8   WTCLKS     : 1;
  __REG8              : 7;
} __wtclks_bits;

/* RTC Selection Clock Status Register (WTCLKM) */
typedef struct {
  __REG8   WTCLKM     : 2;
  __REG8              : 6;
} __wtclkm_bits;

/* RTC Frequency Correction Value Setting Register (WTCAL) */
typedef struct {
  __REG8   WTCAL      : 7;
  __REG8              : 1;
} __wtcal_bits;

/* RTC Frequency Correction Enable Register (WTCALEN) */
typedef struct {
  __REG8   WTCALEN    : 1;
  __REG8              : 7;
} __wtcalen_bits;

/* RTC Divider Ratio Setting Register (WTDIV) */
typedef struct {
  __REG8   WTDIV      : 4;
  __REG8              : 4;
} __wtdiv_bits;

/* RTC Divider Output Enable Register (WTDIVEN) */
typedef struct {
  __REG8   WTDIVEN    : 1;
  __REG8   WTDIVRDY   : 1;
  __REG8              : 6;
} __wtdiven_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,              0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,        0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,        0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,        0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,      0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,           0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(CLRENA0,           0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(SETPEND0,          0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(CLRPEND0,          0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(ACTIVE0,           0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(IP0,               0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,               0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,               0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,               0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,               0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,               0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,               0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,               0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(CPUIDBR,           0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,              0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,             0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,               0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,               0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,             0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,             0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,             0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,              0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,              0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,              0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,             0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,              0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,              0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** FLASH IF
 **
 ***************************************************************************/
__IO_REG32_BIT(FASZR,             0x40000000,__READ_WRITE ,__faszr_bits);
__IO_REG32_BIT(FSTR,              0x40000008,__READ_WRITE ,__fstr_bits);
__IO_REG32_BIT(FSYNDN,            0x40000010,__READ_WRITE ,__fsyndn_bits);
__IO_REG32_BIT(CRTRMM,            0x40000100,__READ       ,__crtrmm_bits);

/***************************************************************************
 **
 ** Clock/Reset
 **
 ***************************************************************************/
__IO_REG32_BIT(SCM_CTL,           0x40010000,__READ_WRITE ,__scm_ctl_bits);
__IO_REG32_BIT(SCM_STR,           0x40010004,__READ       ,__scm_str_bits);
__IO_REG32_BIT(STB_CTL,           0x40010008,__READ_WRITE ,__stb_ctl_bits);
__IO_REG32_BIT(RST_STR,           0x4001000C,__READ       ,__rst_str_bits);
__IO_REG32_BIT(BSC_PSR,           0x40010010,__READ_WRITE ,__bsc_psr_bits);
__IO_REG32_BIT(APBC0_PSR,         0x40010014,__READ_WRITE ,__apbc0_psr_bits);
__IO_REG32_BIT(APBC1_PSR,         0x40010018,__READ_WRITE ,__apbc1_psr_bits);
__IO_REG32_BIT(APBC2_PSR,         0x4001001C,__READ_WRITE ,__apbc2_psr_bits);
__IO_REG32_BIT(SWC_PSR,           0x40010020,__READ_WRITE ,__swc_psr_bits);
__IO_REG32_BIT(CSW_TMR,           0x40010030,__READ_WRITE ,__csw_tmr_bits);
__IO_REG32_BIT(PSW_TMR,           0x40010034,__READ_WRITE ,__psw_tmr_bits);
__IO_REG32_BIT(PLL_CTL1,          0x40010038,__READ_WRITE ,__pll_ctl1_bits);
__IO_REG32_BIT(PLL_CTL2,          0x4001003C,__READ_WRITE ,__pll_ctl2_bits);
__IO_REG32_BIT(CSV_CTL,           0x40010040,__READ_WRITE ,__csv_ctl_bits);
__IO_REG32_BIT(CSV_STR,           0x40010044,__READ       ,__csv_str_bits);
__IO_REG32_BIT(FCSWH_CTL,         0x40010048,__READ_WRITE ,__fcswh_ctl_bits);
__IO_REG32_BIT(FCSWL_CTL,         0x4001004C,__READ_WRITE ,__fcswl_ctl_bits);
__IO_REG32_BIT(FCSWD_CTL,         0x40010050,__READ       ,__fcswd_ctl_bits);
__IO_REG32_BIT(DBWDT_CTL,         0x40010054,__READ_WRITE ,__dbwdt_ctl_bits);
__IO_REG32_BIT(INT_ENR,           0x40010060,__READ_WRITE ,__int_enr_bits);
__IO_REG32_BIT(INT_STR,           0x40010064,__READ       ,__int_str_bits);
__IO_REG32_BIT(INT_CLR,           0x40010068,__WRITE      ,__int_clr_bits); 

/***************************************************************************
 **
 ** HW WDT
 **
 ***************************************************************************/
__IO_REG32(    WDG_LDR,           0x40011000,__READ_WRITE );
__IO_REG32(    WDG_VLR,           0x40011004,__READ       );
__IO_REG32_BIT(WDG_CTL,           0x40011008,__READ_WRITE ,__wdg_ctl_bits);
__IO_REG8(     WDG_ICL,           0x4001100C,__READ_WRITE );
__IO_REG32_BIT(WDG_RIS,           0x40011010,__READ       ,__wdg_ris_bits);
__IO_REG32(    WDG_LCK,           0x40011C00,__READ_WRITE );               

/***************************************************************************
 **
 ** SW WDT
 **
 ***************************************************************************/
__IO_REG32(    WdogLoad,          0x40012000,__READ_WRITE );
__IO_REG32(    WdogValue,         0x40012004,__READ       );
__IO_REG32_BIT(WdogControl,       0x40012008,__READ_WRITE ,__wdg_ctl_bits);
__IO_REG32(    WdogIntClr,        0x4001200C,__READ_WRITE );
__IO_REG32_BIT(WdogRIS,           0x40012010,__READ_WRITE ,__wdg_ris_bits);
__IO_REG32(    WdogLock,          0x40012C00,__READ_WRITE );               

/***************************************************************************
 **
 ** MFT0
 **
 ***************************************************************************/
__IO_REG16(    MFT0_OCCP0,        0x40020000,__READ_WRITE );
__IO_REG16(    MFT0_OCCP1,        0x40020004,__READ_WRITE );
__IO_REG16(    MFT0_OCCP2,        0x40020008,__READ_WRITE );
__IO_REG16(    MFT0_OCCP3,        0x4002000C,__READ_WRITE );
__IO_REG16(    MFT0_OCCP4,        0x40020010,__READ_WRITE );
__IO_REG16(    MFT0_OCCP5,        0x40020014,__READ_WRITE );
__IO_REG8_BIT( MFT0_OCSA10,       0x40020018,__READ_WRITE ,__mft_ocsa10_bits);
__IO_REG8_BIT( MFT0_OCSB10,       0x40020019,__READ_WRITE ,__mft_ocsb10_bits);
__IO_REG8_BIT( MFT0_OCSA32,       0x4002001C,__READ_WRITE ,__mft_ocsa32_bits);
__IO_REG8_BIT( MFT0_OCSB32,       0x4002001D,__READ_WRITE ,__mft_ocsb32_bits);
__IO_REG8_BIT( MFT0_OCSA54,       0x40020020,__READ_WRITE ,__mft_ocsa54_bits);
__IO_REG8_BIT( MFT0_OCSB54,       0x40020021,__READ_WRITE ,__mft_ocsb54_bits);
__IO_REG8_BIT( MFT0_OCSC,         0x40020025,__READ_WRITE ,__mft_ocsc_bits);
__IO_REG16(    MFT0_TCCP0,        0x40020028,__READ_WRITE );
__IO_REG16(    MFT0_TCDT0,        0x4002002C,__READ_WRITE );
__IO_REG16_BIT(MFT0_TCSA0,        0x40020030,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT0_TCSB0,        0x40020034,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT0_TCCP1,        0x40020038,__READ_WRITE );
__IO_REG16(    MFT0_TCDT1,        0x4002003C,__READ_WRITE );
__IO_REG16_BIT(MFT0_TCSA1,        0x40020040,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT0_TCSB1,        0x40020044,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT0_TCCP2,        0x40020048,__READ_WRITE );
__IO_REG16(    MFT0_TCDT2,        0x4002004C,__READ_WRITE );
__IO_REG16_BIT(MFT0_TCSA2,        0x40020050,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT0_TCSB2,        0x40020054,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG8_BIT( MFT0_OCFS10,       0x40020058,__READ_WRITE ,__mft_ocfs10_bits);
__IO_REG8_BIT( MFT0_OCFS32,       0x40020059,__READ_WRITE ,__mft_ocfs32_bits);
__IO_REG8_BIT( MFT0_OCFS54,       0x4002005C,__READ_WRITE ,__mft_ocfs54_bits);
__IO_REG8_BIT( MFT0_ICFS10,       0x40020060,__READ_WRITE ,__mft_icfs10_bits);
__IO_REG8_BIT( MFT0_ICFS32,       0x40020061,__READ_WRITE ,__mft_icfs32_bits);
__IO_REG16(    MFT0_ICCP0,        0x40020068,__READ       );
__IO_REG16(    MFT0_ICCP1,        0x4002006C,__READ       );
__IO_REG16(    MFT0_ICCP2,        0x40020070,__READ       );
__IO_REG16(    MFT0_ICCP3,        0x40020074,__READ       );
__IO_REG8_BIT( MFT0_ICSA10,       0x40020078,__READ_WRITE ,__mft_icsa10_bits);
__IO_REG8_BIT( MFT0_ICSB10,       0x40020079,__READ       ,__mft_icsb10_bits);
__IO_REG8_BIT( MFT0_ICSA32,       0x4002007C,__READ_WRITE ,__mft_icsa32_bits);
__IO_REG8_BIT( MFT0_ICSB32,       0x4002007D,__READ       ,__mft_icsb32_bits);
__IO_REG16(    MFT0_WFTM10,       0x40020080,__READ_WRITE );
__IO_REG16(    MFT0_WFTM32,       0x40020084,__READ_WRITE );
__IO_REG16(    MFT0_WFTM54,       0x40020088,__READ_WRITE );
__IO_REG16_BIT(MFT0_WFSA10,       0x4002008C,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT0_WFSA32,       0x40020090,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT0_WFSA54,       0x40020094,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT0_WFIR,         0x40020098,__READ_WRITE ,__mft_wfir_bits);
__IO_REG16_BIT(MFT0_NZCL,         0x4002009C,__READ_WRITE ,__mft_nzcl_bits);
__IO_REG16(    MFT0_ACCP0,        0x400200A0,__READ_WRITE );
__IO_REG16(    MFT0_ACCPDN0,      0x400200A4,__READ_WRITE );
__IO_REG16(    MFT0_ACCP1,        0x400200A8,__READ_WRITE );
__IO_REG16(    MFT0_ACCPDN1,      0x400200AC,__READ_WRITE );
__IO_REG16_BIT(MFT0_ACSB,         0x400200B8,__READ_WRITE ,__mft_acsb_bits);
__IO_REG16_BIT(MFT0_ACSA,         0x400200BC,__READ_WRITE ,__mft_acsa_bits);
__IO_REG16_BIT(MFT0_ATSA,         0x400200C0,__READ_WRITE ,__mft_atsa_bits); 

/***************************************************************************
 **
 ** PPG
 **
 ***************************************************************************/
__IO_REG8_BIT( PPG_TTCR0,         0x40024001,__READ_WRITE ,__ppg_ttcr0_bits);
__IO_REG8(     PPG_COMP0,         0x40024009,__READ_WRITE );
__IO_REG8(     PPG_COMP2,         0x4002400C,__READ_WRITE );
__IO_REG8(     PPG_COMP4,         0x40024011,__READ_WRITE );
__IO_REG8(     PPG_COMP6,         0x40024014,__READ_WRITE );
__IO_REG16_BIT(PPG_TRG,           0x40024100,__READ_WRITE ,__ppg_trg_bits);
__IO_REG16_BIT(PPG_REVC,          0x40024104,__READ_WRITE ,__ppg_revc_bits);
__IO_REG8_BIT( PPG_PPGC1,         0x40024200,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC0,         0x40024201,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC3,         0x40024204,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC2,         0x40024205,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL0,          0x40024208,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL0 PPG_PRL0_bit.__byte0
#define PPG_PRLH0 PPG_PRL0_bit.__byte1
__IO_REG16_BIT(PPG_PRL1,          0x4002420C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL1 PPG_PRL1_bit.__byte0
#define PPG_PRLH1 PPG_PRL1_bit.__byte1
__IO_REG16_BIT(PPG_PRL2,          0x40024210,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL2 PPG_PRL2_bit.__byte0
#define PPG_PRLH2 PPG_PRL2_bit.__byte1
__IO_REG16_BIT(PPG_PRL3,          0x40024214,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL3 PPG_PRL3_bit.__byte0
#define PPG_PRLH3 PPG_PRL3_bit.__byte1
__IO_REG8_BIT( PPG_GATEC0,        0x40024218,__READ_WRITE ,__ppg_gatec0_bits);
__IO_REG8_BIT( PPG_PPGC5,         0x40024240,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC4,         0x40024241,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC7,         0x40024244,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC6,         0x40024245,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL4,          0x40024248,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL4 PPG_PRL4_bit.__byte0
#define PPG_PRLH4 PPG_PRL4_bit.__byte1
__IO_REG16_BIT(PPG_PRL5,          0x4002424C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL5 PPG_PRL5_bit.__byte0
#define PPG_PRLH5 PPG_PRL5_bit.__byte1
__IO_REG16_BIT(PPG_PRL6,          0x40024250,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL6 PPG_PRL6_bit.__byte0
#define PPG_PRLH6 PPG_PRL6_bit.__byte1
__IO_REG16_BIT(PPG_PRL7,          0x40024254,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL7 PPG_PRL7_bit.__byte0
#define PPG_PRLH7 PPG_PRL7_bit.__byte1
__IO_REG8_BIT( PPG_GATEC4,        0x40024258,__READ_WRITE ,__ppg_gatec4_bits);

/***************************************************************************
 **
 ** BT0_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT0_PPG_PRLL,      0x40025000,__READ_WRITE );
__IO_REG16(    BT0_PPG_PRLH,      0x40025004,__READ_WRITE );
__IO_REG16(    BT0_PPG_TMR,       0x40025008,__READ       );
__IO_REG16_BIT(BT0_PPG_TMCR,      0x4002500C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT0_PPG_STC,       0x40025010,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT0_PPG_TMCR2,     0x40025011,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT1_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT1_PPG_PRLL,      0x40025040,__READ_WRITE );
__IO_REG16(    BT1_PPG_PRLH,      0x40025044,__READ_WRITE );
__IO_REG16(    BT1_PPG_TMR,       0x40025048,__READ       );
__IO_REG16_BIT(BT1_PPG_TMCR,      0x4002504C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT1_PPG_STC,       0x40025050,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT1_PPG_TMCR2,     0x40025051,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT2_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT2_PPG_PRLL,      0x40025080,__READ_WRITE );
__IO_REG16(    BT2_PPG_PRLH,      0x40025084,__READ_WRITE );
__IO_REG16(    BT2_PPG_TMR,       0x40025088,__READ       );
__IO_REG16_BIT(BT2_PPG_TMCR,      0x4002508C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT2_PPG_STC,       0x40025090,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT2_PPG_TMCR2,     0x40025091,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT3_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT3_PPG_PRLL,      0x400250C0,__READ_WRITE );
__IO_REG16(    BT3_PPG_PRLH,      0x400250C4,__READ_WRITE );
__IO_REG16(    BT3_PPG_TMR,       0x400250C8,__READ       );
__IO_REG16_BIT(BT3_PPG_TMCR,      0x400250CC,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT3_PPG_STC,       0x400250D0,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT3_PPG_TMCR2,     0x400250D1,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT4_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT4_PPG_PRLL,      0x40025200,__READ_WRITE );
__IO_REG16(    BT4_PPG_PRLH,      0x40025204,__READ_WRITE );
__IO_REG16(    BT4_PPG_TMR,       0x40025208,__READ       );
__IO_REG16_BIT(BT4_PPG_TMCR,      0x4002520C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT4_PPG_STC,       0x40025210,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT4_PPG_TMCR2,     0x40025211,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT5_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT5_PPG_PRLL,      0x40025240,__READ_WRITE );
__IO_REG16(    BT5_PPG_PRLH,      0x40025244,__READ_WRITE );
__IO_REG16(    BT5_PPG_TMR,       0x40025248,__READ       );
__IO_REG16_BIT(BT5_PPG_TMCR,      0x4002524C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT5_PPG_STC,       0x40025250,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT5_PPG_TMCR2,     0x40025251,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT6_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT6_PPG_PRLL,      0x40025280,__READ_WRITE );
__IO_REG16(    BT6_PPG_PRLH,      0x40025284,__READ_WRITE );
__IO_REG16(    BT6_PPG_TMR,       0x40025288,__READ       );
__IO_REG16_BIT(BT6_PPG_TMCR,      0x4002528C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT6_PPG_STC,       0x40025290,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT6_PPG_TMCR2,     0x40025291,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT7_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT7_PPG_PRLL,      0x400252C0,__READ_WRITE );
__IO_REG16(    BT7_PPG_PRLH,      0x400252C4,__READ_WRITE );
__IO_REG16(    BT7_PPG_TMR,       0x400252C8,__READ       );
__IO_REG16_BIT(BT7_PPG_TMCR,      0x400252CC,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT7_PPG_STC,       0x400252D0,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT7_PPG_TMCR2,     0x400252D1,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT0_PWM
 **
 ***************************************************************************/
#define BT0_PWM_PCSR        BT0_PPG_PRLL
#define BT0_PWM_PDUT        BT0_PPG_PRLH
#define BT0_PWM_TMR         BT0_PPG_TMR
#define BT0_PWM_TMCR        BT0_PPG_TMCR
#define BT0_PWM_TMCR_bit    BT0_PPG_TMCR_bit
#define BT0_PWM_STC         BT0_PPG_STC
#define BT0_PWM_STC_bit     BT0_PPG_STC_bit.PWM
#define BT0_PWM_TMCR2       BT0_PPG_TMCR2
#define BT0_PWM_TMCR2_bit   BT0_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT1_PWM
 **
 ***************************************************************************/
#define BT1_PWM_PCSR        BT1_PPG_PRLL
#define BT1_PWM_PDUT        BT1_PPG_PRLH
#define BT1_PWM_TMR         BT1_PPG_TMR
#define BT1_PWM_TMCR        BT1_PPG_TMCR
#define BT1_PWM_TMCR_bit    BT1_PPG_TMCR_bit
#define BT1_PWM_STC         BT1_PPG_STC
#define BT1_PWM_STC_bit     BT1_PPG_STC_bit.PWM
#define BT1_PWM_TMCR2       BT1_PPG_TMCR2
#define BT1_PWM_TMCR2_bit   BT1_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT2_PWM
 **
 ***************************************************************************/
#define BT2_PWM_PCSR        BT2_PPG_PRLL
#define BT2_PWM_PDUT        BT2_PPG_PRLH
#define BT2_PWM_TMR         BT2_PPG_TMR
#define BT2_PWM_TMCR        BT2_PPG_TMCR
#define BT2_PWM_TMCR_bit    BT2_PPG_TMCR_bit
#define BT2_PWM_STC         BT2_PPG_STC
#define BT2_PWM_STC_bit     BT2_PPG_STC_bit.PWM
#define BT2_PWM_TMCR2       BT2_PPG_TMCR2
#define BT2_PWM_TMCR2_bit   BT2_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT3_PWM
 **
 ***************************************************************************/
#define BT3_PWM_PCSR        BT3_PPG_PRLL
#define BT3_PWM_PDUT        BT3_PPG_PRLH
#define BT3_PWM_TMR         BT3_PPG_TMR
#define BT3_PWM_TMCR        BT3_PPG_TMCR
#define BT3_PWM_TMCR_bit    BT3_PPG_TMCR_bit
#define BT3_PWM_STC         BT3_PPG_STC
#define BT3_PWM_STC_bit     BT3_PPG_STC_bit.PWM
#define BT3_PWM_TMCR2       BT3_PPG_TMCR2
#define BT3_PWM_TMCR2_bit   BT3_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT4_PWM
 **
 ***************************************************************************/
#define BT4_PWM_PCSR        BT4_PPG_PRLL
#define BT4_PWM_PDUT        BT4_PPG_PRLH
#define BT4_PWM_TMR         BT4_PPG_TMR
#define BT4_PWM_TMCR        BT4_PPG_TMCR
#define BT4_PWM_TMCR_bit    BT4_PPG_TMCR_bit
#define BT4_PWM_STC         BT4_PPG_STC
#define BT4_PWM_STC_bit     BT4_PPG_STC_bit.PWM
#define BT4_PWM_TMCR2       BT4_PPG_TMCR2
#define BT4_PWM_TMCR2_bit   BT4_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT5_PWM
 **
 ***************************************************************************/
#define BT5_PWM_PCSR        BT5_PPG_PRLL
#define BT5_PWM_PDUT        BT5_PPG_PRLH
#define BT5_PWM_TMR         BT5_PPG_TMR
#define BT5_PWM_TMCR        BT5_PPG_TMCR
#define BT5_PWM_TMCR_bit    BT5_PPG_TMCR_bit
#define BT5_PWM_STC         BT5_PPG_STC
#define BT5_PWM_STC_bit     BT5_PPG_STC_bit.PWM
#define BT5_PWM_TMCR2       BT5_PPG_TMCR2
#define BT5_PWM_TMCR2_bit   BT5_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT6_PWM
 **
 ***************************************************************************/
#define BT6_PWM_PCSR        BT6_PPG_PRLL
#define BT6_PWM_PDUT        BT6_PPG_PRLH
#define BT6_PWM_TMR         BT6_PPG_TMR
#define BT6_PWM_TMCR        BT6_PPG_TMCR
#define BT6_PWM_TMCR_bit    BT6_PPG_TMCR_bit
#define BT6_PWM_STC         BT6_PPG_STC
#define BT6_PWM_STC_bit     BT6_PPG_STC_bit.PWM
#define BT6_PWM_TMCR2       BT6_PPG_TMCR2
#define BT6_PWM_TMCR2_bit   BT6_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT7_PWM
 **
 ***************************************************************************/
#define BT7_PWM_PCSR        BT7_PPG_PRLL
#define BT7_PWM_PDUT        BT7_PPG_PRLH
#define BT7_PWM_TMR         BT7_PPG_TMR
#define BT7_PWM_TMCR        BT7_PPG_TMCR
#define BT7_PWM_TMCR_bit    BT7_PPG_TMCR_bit
#define BT7_PWM_STC         BT7_PPG_STC
#define BT7_PWM_STC_bit     BT7_PPG_STC_bit.PWM
#define BT7_PWM_TMCR2       BT7_PPG_TMCR2
#define BT7_PWM_TMCR2_bit   BT7_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT0_RT
 **
 ***************************************************************************/
#define BT0_RT_PCSR         BT0_PPG_PRLL
#define BT0_RT_TMR          BT0_PPG_TMR
#define BT0_RT_TMCR         BT0_PPG_TMCR
#define BT0_RT_TMCR_bit     BT0_PPG_TMCR_bit.RT
#define BT0_RT_STC          BT0_PPG_STC
#define BT0_RT_STC_bit      BT0_PPG_STC_bit
#define BT0_RT_TMCR2        BT0_PPG_TMCR2
#define BT0_RT_TMCR2_bit    BT0_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT1_RT
 **
 ***************************************************************************/
#define BT1_RT_PCSR         BT1_PPG_PRLL
#define BT1_RT_TMR          BT1_PPG_TMR
#define BT1_RT_TMCR         BT1_PPG_TMCR
#define BT1_RT_TMCR_bit     BT1_PPG_TMCR_bit.RT
#define BT1_RT_STC          BT1_PPG_STC
#define BT1_RT_STC_bit      BT1_PPG_STC_bit
#define BT1_RT_TMCR2        BT1_PPG_TMCR2
#define BT1_RT_TMCR2_bit    BT1_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT2_RT
 **
 ***************************************************************************/
#define BT2_RT_PCSR         BT2_PPG_PRLL
#define BT2_RT_TMR          BT2_PPG_TMR
#define BT2_RT_TMCR         BT2_PPG_TMCR
#define BT2_RT_TMCR_bit     BT2_PPG_TMCR_bit.RT
#define BT2_RT_STC          BT2_PPG_STC
#define BT2_RT_STC_bit      BT2_PPG_STC_bit
#define BT2_RT_TMCR2        BT2_PPG_TMCR2
#define BT2_RT_TMCR2_bit    BT2_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT3_RT
 **
 ***************************************************************************/
#define BT3_RT_PCSR         BT3_PPG_PRLL
#define BT3_RT_TMR          BT3_PPG_TMR
#define BT3_RT_TMCR         BT3_PPG_TMCR
#define BT3_RT_TMCR_bit     BT3_PPG_TMCR_bit.RT
#define BT3_RT_STC          BT3_PPG_STC
#define BT3_RT_STC_bit      BT3_PPG_STC_bit
#define BT3_RT_TMCR2        BT3_PPG_TMCR2
#define BT3_RT_TMCR2_bit    BT3_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT4_RT
 **
 ***************************************************************************/
#define BT4_RT_PCSR         BT4_PPG_PRLL
#define BT4_RT_TMR          BT4_PPG_TMR
#define BT4_RT_TMCR         BT4_PPG_TMCR
#define BT4_RT_TMCR_bit     BT4_PPG_TMCR_bit.RT
#define BT4_RT_STC          BT4_PPG_STC
#define BT4_RT_STC_bit      BT4_PPG_STC_bit
#define BT4_RT_TMCR2        BT4_PPG_TMCR2
#define BT4_RT_TMCR2_bit    BT4_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT5_RT
 **
 ***************************************************************************/
#define BT5_RT_PCSR         BT5_PPG_PRLL
#define BT5_RT_TMR          BT5_PPG_TMR
#define BT5_RT_TMCR         BT5_PPG_TMCR
#define BT5_RT_TMCR_bit     BT5_PPG_TMCR_bit.RT
#define BT5_RT_STC          BT5_PPG_STC
#define BT5_RT_STC_bit      BT5_PPG_STC_bit
#define BT5_RT_TMCR2        BT5_PPG_TMCR2
#define BT5_RT_TMCR2_bit    BT5_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT6_RT
 **
 ***************************************************************************/
#define BT6_RT_PCSR         BT6_PPG_PRLL
#define BT6_RT_TMR          BT6_PPG_TMR
#define BT6_RT_TMCR         BT6_PPG_TMCR
#define BT6_RT_TMCR_bit     BT6_PPG_TMCR_bit.RT
#define BT6_RT_STC          BT6_PPG_STC
#define BT6_RT_STC_bit      BT6_PPG_STC_bit
#define BT6_RT_TMCR2        BT6_PPG_TMCR2
#define BT6_RT_TMCR2_bit    BT6_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT7_RT
 **
 ***************************************************************************/
#define BT7_RT_PCSR         BT7_PPG_PRLL
#define BT7_RT_TMR          BT7_PPG_TMR
#define BT7_RT_TMCR         BT7_PPG_TMCR
#define BT7_RT_TMCR_bit     BT7_PPG_TMCR_bit.RT
#define BT7_RT_STC          BT7_PPG_STC
#define BT7_RT_STC_bit      BT7_PPG_STC_bit
#define BT7_RT_TMCR2        BT7_PPG_TMCR2
#define BT7_RT_TMCR2_bit    BT7_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT0_PWC
 **
 ***************************************************************************/
#define BT0_PWC_DTBF        BT0_PPG_PRLH
#define BT0_PWC_TMCR        BT0_PPG_TMCR
#define BT0_PWC_TMCR_bit    BT0_PPG_TMCR_bit.PWC
#define BT0_PWC_STC         BT0_PPG_STC
#define BT0_PWC_STC_bit     BT0_PPG_STC_bit.PWC
#define BT0_PWC_TMCR2       BT0_PPG_TMCR2
#define BT0_PWC_TMCR2_bit   BT0_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT1_PWC
 **
 ***************************************************************************/
#define BT1_PWC_DTBF        BT1_PPG_PRLH
#define BT1_PWC_TMCR        BT1_PPG_TMCR
#define BT1_PWC_TMCR_bit    BT1_PPG_TMCR_bit.PWC
#define BT1_PWC_STC         BT1_PPG_STC
#define BT1_PWC_STC_bit     BT1_PPG_STC_bit.PWC
#define BT1_PWC_TMCR2       BT1_PPG_TMCR2
#define BT1_PWC_TMCR2_bit   BT1_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT2_PWC
 **
 ***************************************************************************/
#define BT2_PWC_DTBF        BT2_PPG_PRLH
#define BT2_PWC_TMCR        BT2_PPG_TMCR
#define BT2_PWC_TMCR_bit    BT2_PPG_TMCR_bit.PWC
#define BT2_PWC_STC         BT2_PPG_STC
#define BT2_PWC_STC_bit     BT2_PPG_STC_bit.PWC
#define BT2_PWC_TMCR2       BT2_PPG_TMCR2
#define BT2_PWC_TMCR2_bit   BT2_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT3_PWC
 **
 ***************************************************************************/
#define BT3_PWC_DTBF        BT3_PPG_PRLH
#define BT3_PWC_TMCR        BT3_PPG_TMCR
#define BT3_PWC_TMCR_bit    BT3_PPG_TMCR_bit.PWC
#define BT3_PWC_STC         BT3_PPG_STC
#define BT3_PWC_STC_bit     BT3_PPG_STC_bit.PWC
#define BT3_PWC_TMCR2       BT3_PPG_TMCR2
#define BT3_PWC_TMCR2_bit   BT3_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT4_PWC
 **
 ***************************************************************************/
#define BT4_PWC_DTBF        BT4_PPG_PRLH
#define BT4_PWC_TMCR        BT4_PPG_TMCR
#define BT4_PWC_TMCR_bit    BT4_PPG_TMCR_bit.PWC
#define BT4_PWC_STC         BT4_PPG_STC
#define BT4_PWC_STC_bit     BT4_PPG_STC_bit.PWC
#define BT4_PWC_TMCR2       BT4_PPG_TMCR2
#define BT4_PWC_TMCR2_bit   BT4_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT5_PWC
 **
 ***************************************************************************/
#define BT5_PWC_DTBF        BT5_PPG_PRLH
#define BT5_PWC_TMCR        BT5_PPG_TMCR
#define BT5_PWC_TMCR_bit    BT5_PPG_TMCR_bit.PWC
#define BT5_PWC_STC         BT5_PPG_STC
#define BT5_PWC_STC_bit     BT5_PPG_STC_bit.PWC
#define BT5_PWC_TMCR2       BT5_PPG_TMCR2
#define BT5_PWC_TMCR2_bit   BT5_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT6_PWC
 **
 ***************************************************************************/
#define BT6_PWC_DTBF        BT6_PPG_PRLH
#define BT6_PWC_TMCR        BT6_PPG_TMCR
#define BT6_PWC_TMCR_bit    BT6_PPG_TMCR_bit.PWC
#define BT6_PWC_STC         BT6_PPG_STC
#define BT6_PWC_STC_bit     BT6_PPG_STC_bit.PWC
#define BT6_PWC_TMCR2       BT6_PPG_TMCR2
#define BT6_PWC_TMCR2_bit   BT6_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT7_PWC
 **
 ***************************************************************************/
#define BT7_PWC_DTBF        BT7_PPG_PRLH
#define BT7_PWC_TMCR        BT7_PPG_TMCR
#define BT7_PWC_TMCR_bit    BT7_PPG_TMCR_bit.PWC
#define BT7_PWC_STC         BT7_PPG_STC
#define BT7_PWC_STC_bit     BT7_PPG_STC_bit.PWC
#define BT7_PWC_TMCR2       BT7_PPG_TMCR2
#define BT7_PWC_TMCR2_bit   BT7_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT I/O Select
 **
 ***************************************************************************/
__IO_REG8_BIT( BTSEL0123,         0x40025101,__READ_WRITE ,__btsel0123_bits);
__IO_REG8_BIT( BTSEL4567,         0x40025301,__READ_WRITE ,__btsel4567_bits);
__IO_REG16_BIT(BTSSSR,            0x40025FFC,__WRITE      ,__btsssr_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG8_BIT( ADC0_ADSR,         0x40027000,__READ_WRITE ,__adc_adsr_bits);
__IO_REG8_BIT( ADC0_ADCR,         0x40027001,__READ_WRITE ,__adc_adcr_bits);
__IO_REG8_BIT( ADC0_SFNS,         0x40027008,__READ_WRITE ,__adc_sfns_bits);
__IO_REG8_BIT( ADC0_SCCR,         0x40027009,__READ_WRITE ,__adc_sccr_bits);
__IO_REG32_BIT(ADC0_SCFD,         0x4002700C,__READ       ,__adc_scfd_bits);
__IO_REG8_BIT( ADC0_SCIS0,        0x40027014,__READ_WRITE ,__adc_scis0_bits);
__IO_REG8_BIT( ADC0_PFNS,         0x40027018,__READ_WRITE ,__adc_pfns_bits);
__IO_REG8_BIT( ADC0_PCCR,         0x40027019,__READ_WRITE ,__adc_pccr_bits);
__IO_REG32_BIT(ADC0_PCFD,         0x4002701C,__READ       ,__adc_pcfd_bits);
__IO_REG8_BIT( ADC0_PCIS,         0x40027020,__READ_WRITE ,__adc_pcis_bits);
__IO_REG8_BIT( ADC0_CMPCR,        0x40027024,__READ_WRITE ,__adc_cmpcr_bits);
__IO_REG16_BIT(ADC0_CMPD,         0x40027026,__READ_WRITE ,__adc_cmpd_bits);
__IO_REG8_BIT( ADC0_ADSS0,        0x4002702C,__READ_WRITE ,__adc_adss0_bits);
__IO_REG8_BIT( ADC0_ADSS1,        0x4002702D,__READ_WRITE ,__adc_adss1_bits);
__IO_REG8_BIT( ADC0_ADST1,        0x40027030,__READ_WRITE ,__adc_adst1_bits);
__IO_REG8_BIT( ADC0_ADST0,        0x40027031,__READ_WRITE ,__adc_adst0_bits);
__IO_REG8_BIT( ADC0_ADCT,         0x40027034,__READ_WRITE ,__adc_adct_bits);
__IO_REG16_BIT(ADC0_ADCEN,        0x4002703C,__READ_WRITE ,__adc_adcen_bits);

/***************************************************************************
 **
 ** CR Trim
 **
 ***************************************************************************/
__IO_REG8_BIT( MCR_PSR,           0x4002E000,__READ_WRITE ,__mcr_psr_bits);
__IO_REG16_BIT(MCR_FTRM,          0x4002E004,__READ_WRITE ,__mcr_ftrm_bits);
__IO_REG32(    MCR_RLR,           0x4002E00C,__READ_WRITE );

/***************************************************************************
 **
 ** EXTI
 **
 ***************************************************************************/
__IO_REG16_BIT(ENIR,              0x40030000,__READ_WRITE ,__enir_bits);
__IO_REG16_BIT(EIRR,              0x40030004,__READ       ,__eirr_bits);
__IO_REG16_BIT(EICL,              0x40030008,__READ_WRITE ,__eicl_bits);
__IO_REG32_BIT(ELVR,              0x4003000C,__READ_WRITE ,__elvr_bits);
__IO_REG16_BIT(NMIRR,             0x40030014,__READ       ,__nmirr_bits);
__IO_REG16_BIT(NMICL,             0x40030018,__READ_WRITE ,__nmicl_bits);

/***************************************************************************
 **
 ** INT Req Read
 **
 ***************************************************************************/
__IO_REG32_BIT(EXC02MON,          0x40031010,__READ       ,__exc02mon_bits);
__IO_REG32_BIT(IRQ00MON,          0x40031014,__READ       ,__irqmon0_bits);
__IO_REG32_BIT(IRQ01MON,          0x40031018,__READ       ,__irqmon1_bits);
__IO_REG32_BIT(IRQ02MON,          0x4003101C,__READ       ,__irqmon2_bits);
__IO_REG32_BIT(IRQ03MON,          0x40031020,__READ       ,__irqmon3_bits);
__IO_REG32_BIT(IRQ04MON,          0x40031024,__READ       ,__irqmon4_bits);
__IO_REG32_BIT(IRQ05MON,          0x40031028,__READ       ,__irqmon5_bits);
__IO_REG32_BIT(IRQ06MON,          0x4003102C,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ07MON,          0x40031030,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ08MON,          0x40031034,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ09MON,          0x40031038,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ10MON,          0x4003103C,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ11MON,          0x40031040,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ12MON,          0x40031044,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ13MON,          0x40031048,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ14MON,          0x4003104C,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ15MON,          0x40031050,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ16MON,          0x40031054,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ17MON,          0x40031058,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ18MON,          0x4003105C,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ19MON,          0x40031060,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ20MON,          0x40031064,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ21MON,          0x40031068,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ22MON,          0x4003106C,__READ       ,__irqmon22_bits);
__IO_REG32_BIT(IRQ23MON,          0x40031070,__READ       ,__irqmon23_bits);
__IO_REG32_BIT(IRQ24MON,          0x40031074,__READ       ,__irqmon24_bits);
__IO_REG32_BIT(IRQ25MON,          0x40031078,__READ       ,__irqmon25_bits);
__IO_REG32_BIT(IRQ26MON,          0x4003107C,__READ       ,__irqmon26_bits);
__IO_REG32_BIT(IRQ27MON,          0x40031080,__READ       ,__irqmon27_bits);
__IO_REG32_BIT(IRQ28MON,          0x40031084,__READ       ,__irqmon28_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(PFR0,              0x40033000,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PFR1,              0x40033004,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PFR2,              0x40033008,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PFR3,              0x4003300C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PFR4,              0x40033010,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PFR5,              0x40033014,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PFR6,              0x40033018,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PFR8,              0x40033020,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PFRE,              0x40033038,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PCR0,              0x40033100,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PCR1,              0x40033104,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PCR2,              0x40033108,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PCR3,              0x4003310C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PCR4,              0x40033110,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PCR5,              0x40033114,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PCR6,              0x40033118,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PCRE,              0x40033138,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(DDR0,              0x40033200,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(DDR1,              0x40033204,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(DDR2,              0x40033208,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(DDR3,              0x4003320C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(DDR4,              0x40033210,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(DDR5,              0x40033214,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(DDR6,              0x40033218,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(DDR8,              0x40033220,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(DDRE,              0x40033238,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PDIR0,             0x40033300,__READ       ,__port0_bits);
__IO_REG32_BIT(PDIR1,             0x40033304,__READ       ,__port1_bits);
__IO_REG32_BIT(PDIR2,             0x40033308,__READ       ,__port2_bits);
__IO_REG32_BIT(PDIR3,             0x4003330C,__READ       ,__port3_bits);
__IO_REG32_BIT(PDIR4,             0x40033310,__READ       ,__port4_bits);
__IO_REG32_BIT(PDIR5,             0x40033314,__READ       ,__port5_bits);
__IO_REG32_BIT(PDIR6,             0x40033318,__READ       ,__port6_bits);
__IO_REG32_BIT(PDIR8,             0x40033320,__READ       ,__port8_bits);
__IO_REG32_BIT(PDIRE,             0x40033338,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PDOR0,             0x40033400,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PDOR1,             0x40033404,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PDOR2,             0x40033408,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PDOR3,             0x4003340C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PDOR4,             0x40033410,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PDOR5,             0x40033414,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PDOR6,             0x40033418,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PDOR8,             0x40033420,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PDORE,             0x40033438,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(ADE,               0x40033500,__READ_WRITE ,__ade_bits);
__IO_REG32_BIT(SPSR,              0x40033580,__READ_WRITE ,__spsr_bits);
__IO_REG32_BIT(EPFR00,            0x40033600,__READ_WRITE ,__epfr00_bits);
__IO_REG32_BIT(EPFR01,            0x40033604,__READ_WRITE ,__epfr01_bits);
__IO_REG32_BIT(EPFR04,            0x40033610,__READ_WRITE ,__epfr04_bits);
__IO_REG32_BIT(EPFR05,            0x40033614,__READ_WRITE ,__epfr05_bits);
__IO_REG32_BIT(EPFR06,            0x40033618,__READ_WRITE ,__epfr06_bits);
__IO_REG32_BIT(EPFR07,            0x4003361C,__READ_WRITE ,__epfr07_bits);
__IO_REG32_BIT(EPFR08,            0x40033620,__READ_WRITE ,__epfr08_bits);
__IO_REG32_BIT(EPFR09,            0x40033624,__READ_WRITE ,__epfr09_bits);
__IO_REG32_BIT(PZR0,              0x40033700,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PZR1,              0x40033704,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PZR2,              0x40033708,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PZR3,              0x4003370C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PZR4,              0x40033710,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PZR5,              0x40033714,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PZR6,              0x40033718,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PZR8,              0x40033720,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PZRE,              0x40033738,__READ_WRITE ,__porte_bits);

/***************************************************************************
 **
 ** LVD
 **
 ***************************************************************************/
__IO_REG16_BIT(LVD_CTL,           0x40035000,__READ_WRITE ,__lvd_ctl_bits);
__IO_REG8_BIT( LVD_STR,           0x40035004,__READ       ,__lvd_str_bits);
__IO_REG8_BIT( LVD_CLR,           0x40035008,__READ_WRITE ,__lvd_clr_bits);
__IO_REG32(    LVD_RLR,           0x4003500C,__READ_WRITE );
__IO_REG8_BIT( LVD_STR2,          0x40035010,__READ       ,__lvd_str2_bits);

/***************************************************************************
 **
 ** DS Mode
 **
 ***************************************************************************/
__IO_REG8_BIT( PMD_CTL,           0x40035800,__READ_WRITE ,__pmd_ctl_bits);
__IO_REG8_BIT( WRFSR,             0x40035804,__READ_WRITE ,__wrfsr_bits);
__IO_REG8_BIT( WIFSR,             0x40035808,__READ       ,__wifsr_bits);
__IO_REG8_BIT( WIER,              0x4003580C,__READ_WRITE ,__wier_bits);
__IO_REG8_BIT( WILVR,             0x40035810,__READ_WRITE ,__wilvr_bits);
__IO_REG8(     BUR01,             0x40035900,__READ_WRITE );
__IO_REG8(     BUR02,             0x40035901,__READ_WRITE );
__IO_REG8(     BUR03,             0x40035902,__READ_WRITE );
__IO_REG8(     BUR04,             0x40035903,__READ_WRITE );
__IO_REG8(     BUR05,             0x40035904,__READ_WRITE );
__IO_REG8(     BUR06,             0x40035905,__READ_WRITE );
__IO_REG8(     BUR07,             0x40035906,__READ_WRITE );
__IO_REG8(     BUR08,             0x40035907,__READ_WRITE );
__IO_REG8(     BUR09,             0x40035908,__READ_WRITE );
__IO_REG8(     BUR10,             0x40035909,__READ_WRITE );
__IO_REG8(     BUR11,             0x4003590A,__READ_WRITE );
__IO_REG8(     BUR12,             0x4003590B,__READ_WRITE );
__IO_REG8(     BUR13,             0x4003590C,__READ_WRITE );
__IO_REG8(     BUR14,             0x4003590D,__READ_WRITE );
__IO_REG8(     BUR15,             0x4003590E,__READ_WRITE );
__IO_REG8(     BUR16,             0x4003590F,__READ_WRITE );

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG8_BIT( UART0_SMR,         0x40038000,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART0_SCR,         0x40038001,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART0_ESCR,        0x40038004,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART0_SSR,         0x40038005,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART0_RDR,         0x40038008,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART0_TDR     UART0_RDR
#define UART0_TDR_bit UART0_RDR_bit
__IO_REG16_BIT(UART0_BGR,         0x4003800C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG8_BIT( UART1_SMR,         0x40038100,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART1_SCR,         0x40038101,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART1_ESCR,        0x40038104,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART1_SSR,         0x40038105,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART1_RDR,         0x40038108,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART1_TDR     UART1_RDR
#define UART1_TDR_bit UART1_RDR_bit
__IO_REG16_BIT(UART1_BGR,         0x4003810C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
__IO_REG8_BIT( UART3_SMR,         0x40038300,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART3_SCR,         0x40038301,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART3_ESCR,        0x40038304,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART3_SSR,         0x40038305,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART3_RDR,         0x40038308,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART3_TDR     UART3_RDR
#define UART3_TDR_bit UART3_RDR_bit
__IO_REG16_BIT(UART3_BGR,         0x4003830C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART5
 **
 ***************************************************************************/
__IO_REG8_BIT( UART5_SMR,         0x40038500,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART5_SCR,         0x40038501,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART5_ESCR,        0x40038504,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART5_SSR,         0x40038505,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART5_RDR,         0x40038508,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART5_TDR     UART5_RDR
#define UART5_TDR_bit UART5_RDR_bit
__IO_REG16_BIT(UART5_BGR,         0x4003850C,__READ_WRITE ,__mfsx_bgr_bits);
__IO_REG16_BIT(UART5_FCR,         0x40038514,__READ_WRITE ,__mfsx_fcr_bits);
__IO_REG8(     UART5_FBYTE1,      0x40038518,__READ_WRITE );
__IO_REG8(     UART5_FBYTE2,      0x40038519,__READ_WRITE );

/***************************************************************************
 **
 ** CSIO0
 **
 ***************************************************************************/
#define CSIO0_SMR       UART0_SMR
#define CSIO0_SMR_bit   UART0_SMR_bit.CSIO
#define CSIO0_SCR       UART0_SCR
#define CSIO0_SCR_bit   UART0_SCR_bit.CSIO
#define CSIO0_ESCR      UART0_ESCR
#define CSIO0_ESCR_bit  UART0_ESCR_bit.CSIO
#define CSIO0_SSR       UART0_SSR
#define CSIO0_SSR_bit   UART0_SSR_bit.CSIO
#define CSIO0_RDR       UART0_RDR
#define CSIO0_RDR_bit   UART0_RDR_bit.CSIO
#define CSIO0_TDR       UART0_RDR
#define CSIO0_TDR_bit   UART0_RDR_bit.CSIO
#define CSIO0_BGR       UART0_BGR
#define CSIO0_BGR_bit   UART0_BGR_bit.CSIO

/***************************************************************************
 **
 ** CSIO1
 **
 ***************************************************************************/
#define CSIO1_SMR       UART1_SMR
#define CSIO1_SMR_bit   UART1_SMR_bit.CSIO
#define CSIO1_SCR       UART1_SCR
#define CSIO1_SCR_bit   UART1_SCR_bit.CSIO
#define CSIO1_ESCR      UART1_ESCR
#define CSIO1_ESCR_bit  UART1_ESCR_bit.CSIO
#define CSIO1_SSR       UART1_SSR
#define CSIO1_SSR_bit   UART1_SSR_bit.CSIO
#define CSIO1_RDR       UART1_RDR
#define CSIO1_RDR_bit   UART1_RDR_bit.CSIO
#define CSIO1_TDR       UART1_RDR
#define CSIO1_TDR_bit   UART1_RDR_bit.CSIO
#define CSIO1_BGR       UART1_BGR
#define CSIO1_BGR_bit   UART1_BGR_bit.CSIO

/***************************************************************************
 **
 ** CSIO3
 **
 ***************************************************************************/
#define CSIO3_SMR       UART3_SMR
#define CSIO3_SMR_bit   UART3_SMR_bit.CSIO
#define CSIO3_SCR       UART3_SCR
#define CSIO3_SCR_bit   UART3_SCR_bit.CSIO
#define CSIO3_ESCR      UART3_ESCR
#define CSIO3_ESCR_bit  UART3_ESCR_bit.CSIO
#define CSIO3_SSR       UART3_SSR
#define CSIO3_SSR_bit   UART3_SSR_bit.CSIO
#define CSIO3_RDR       UART3_RDR
#define CSIO3_RDR_bit   UART3_RDR_bit.CSIO
#define CSIO3_TDR       UART3_RDR
#define CSIO3_TDR_bit   UART3_RDR_bit.CSIO
#define CSIO3_BGR       UART3_BGR
#define CSIO3_BGR_bit   UART3_BGR_bit.CSIO

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
#define I2C0_SMR        UART0_SMR
#define I2C0_SMR_bit    UART0_SMR_bit.I2C
#define I2C0_IBCR       UART0_SCR
#define I2C0_IBCR_bit   UART0_SCR_bit.I2C
#define I2C0_IBSR       UART0_ESCR
#define I2C0_IBSR_bit   UART0_ESCR_bit.I2C
#define I2C0_SSR        UART0_SSR
#define I2C0_SSR_bit    UART0_SSR_bit.I2C
#define I2C0_RDR        UART0_RDR
#define I2C0_RDR_bit    UART0_RDR_bit.I2C
#define I2C0_TDR        UART0_RDR
#define I2C0_TDR_bit    UART0_RDR_bit.I2C
#define I2C0_BGR        UART0_BGR
#define I2C0_BGR_bit    UART0_BGR_bit.I2C
__IO_REG8_BIT( I2C0_ISBA,         0x40038010,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C0_ISMK,         0x40038011,__READ_WRITE ,__mfsx_ismk_bits);    

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
#define I2C1_SMR        UART1_SMR
#define I2C1_SMR_bit    UART1_SMR_bit.I2C
#define I2C1_IBCR       UART1_SCR
#define I2C1_IBCR_bit   UART1_SCR_bit.I2C
#define I2C1_IBSR       UART1_ESCR
#define I2C1_IBSR_bit   UART1_ESCR_bit.I2C
#define I2C1_SSR        UART1_SSR
#define I2C1_SSR_bit    UART1_SSR_bit.I2C
#define I2C1_RDR        UART1_RDR
#define I2C1_RDR_bit    UART1_RDR_bit.I2C
#define I2C1_TDR        UART1_RDR
#define I2C1_TDR_bit    UART1_RDR_bit.I2C
#define I2C1_BGR        UART1_BGR
#define I2C1_BGR_bit    UART1_BGR_bit.I2C
__IO_REG8_BIT( I2C1_ISBA,        0x40038110,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C1_ISMK,        0x40038111,__READ_WRITE ,__mfsx_ismk_bits);   

/***************************************************************************
 **
 ** I2C3
 **
 ***************************************************************************/
#define I2C3_SMR        UART3_SMR
#define I2C3_SMR_bit    UART3_SMR_bit.I2C
#define I2C3_IBCR       UART3_SCR
#define I2C3_IBCR_bit   UART3_SCR_bit.I2C
#define I2C3_IBSR       UART3_ESCR
#define I2C3_IBSR_bit   UART3_ESCR_bit.I2C
#define I2C3_SSR        UART3_SSR
#define I2C3_SSR_bit    UART3_SSR_bit.I2C
#define I2C3_RDR        UART3_RDR
#define I2C3_RDR_bit    UART3_RDR_bit.I2C
#define I2C3_TDR        UART3_RDR
#define I2C3_TDR_bit    UART3_RDR_bit.I2C
#define I2C3_BGR        UART3_BGR
#define I2C3_BGR_bit    UART3_BGR_bit.I2C
__IO_REG8_BIT( I2C3_ISBA,        0x40038310,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C3_ISMK,        0x40038311,__READ_WRITE ,__mfsx_ismk_bits);   


/***************************************************************************
 **
 ** Watch Counter
 **
 ***************************************************************************/
__IO_REG8_BIT( WCRD,              0x4003A000,__READ       ,__wcrd_bits);
__IO_REG8_BIT( WCRL,              0x4003A001,__READ_WRITE ,__wcrl_bits);
__IO_REG8_BIT( WCCR,              0x4003A002,__READ_WRITE ,__wccr_bits);
__IO_REG16_BIT(CLK_SEL,           0x4003A010,__READ_WRITE ,__clk_sel_bits);
__IO_REG8_BIT( CLK_EN,            0x4003A014,__READ_WRITE ,__clk_en_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(WTCR1,             0x4003B000,__READ_WRITE ,__wtcr1_bits);
__IO_REG32_BIT(WTCR2,             0x4003B004,__READ_WRITE ,__wtcr2_bits);
__IO_REG32_BIT(WTBR,              0x4003B008,__READ_WRITE ,__wtbr_bits);
__IO_REG8_BIT( WTSR,              0x4003B00C,__READ_WRITE ,__wtsr_bits);
__IO_REG8_BIT( WTMIR,             0x4003B00D,__READ_WRITE ,__wtmir_bits);
__IO_REG8_BIT( WTHR,              0x4003B00E,__READ_WRITE ,__wthr_bits);
__IO_REG8_BIT( WTDR,              0x4003B00F,__READ_WRITE ,__wtdr_bits);
__IO_REG8_BIT( WTDW,              0x4003B010,__READ_WRITE ,__wtdw_bits);
__IO_REG8_BIT( WTMOR,             0x4003B011,__READ_WRITE ,__wtmor_bits);
__IO_REG8_BIT( WTYR,              0x4003B012,__READ_WRITE ,__wtyr_bits);
__IO_REG8_BIT( ALMIR,             0x4003B015,__READ_WRITE ,__almir_bits);
__IO_REG8_BIT( ALHR,              0x4003B016,__READ_WRITE ,__alhr_bits);
__IO_REG8_BIT( ALDR,              0x4003B017,__READ_WRITE ,__aldr_bits);
__IO_REG8_BIT( ALMOR,             0x4003B019,__READ_WRITE ,__almor_bits);
__IO_REG8_BIT( ALYR,              0x4003B01A,__READ_WRITE ,__alyr_bits);
__IO_REG32_BIT(WTTR,              0x4003B01C,__READ_WRITE ,__wttr_bits);
__IO_REG8_BIT( WTCLKS,            0x4003B020,__READ_WRITE ,__wtclks_bits);
__IO_REG8_BIT( WTCLKM,            0x4003B021,__READ       ,__wtclkm_bits);
__IO_REG8_BIT( WTCAL,             0x4003B024,__READ_WRITE ,__wtcal_bits);
__IO_REG8_BIT( WTCALEN,           0x4003B025,__READ_WRITE ,__wtcalen_bits);
__IO_REG8_BIT( WTDIV,             0x4003B028,__READ_WRITE ,__wtdiv_bits);
__IO_REG8_BIT( WTDIVEN,           0x4003B029,__READ_WRITE ,__wtdiven_bits);

/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  MB9AF132K Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK             0          /* Main Stack                   */
#define RESETI                 1          /* Reset                        */
#define NMII                   2          /* Non-maskable Interrupt       */
#define HFI                    3          /* Hard Fault                   */
#define MMI                    4          /* Memory Management            */
#define BFI                    5          /* Bus Fault                    */
#define UFI                    6          /* Usage Fault                  */
#define SVCI                  11          /* SVCall                       */
#define DMI                   12          /* Debug Monitor                */
#define PSI                   14          /* PendSV                       */
#define STI                   15          /* SysTick                      */
#define NVIC_CSV              16          /* Anomalous Frequency Detection by Clock Supervisor (FCS)                                             */
#define NVIC_SWDT             17          /* Software Watchdog Timer                                                                             */
#define NVIC_LVD              18          /* Low Voltage Detector (LVD)                                                                          */
#define NVIC_WFG              19          /* Wave Form Generator unit0,                                                                          */
#define NVIC_EXTI0_6          20          /* External Interrupt Request ch.0 to ch.6                                                             */
#define NVIC_EXTI15           21          /* External Interrupt Request ch.15                                                                    */
#define NVIC_MFSI0RX          22          /* Reception Interrupt Request of Multi-Function Serial Interface ch.0                                 */
#define NVIC_MFSI0TX          23          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.0 */ 
#define NVIC_MFSI1RX          24          /* Reception Interrupt Request of Multi-Function Serial Interface ch.1                                 */
#define NVIC_MFSI1TX          25          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.1 */ 
#define NVIC_MFSI3RX          28          /* Reception Interrupt Request of Multi-Function Serial Interface ch.3                                 */
#define NVIC_MFSI3TX          29          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.3 */ 
#define NVIC_MFSI5RX          32          /* Reception Interrupt Request of Multi-Function Serial Interface ch.5                                 */
#define NVIC_MFSI5TX          33          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.5 */ 
#define NVIC_PPG              38          /* PPG unit0                                                                         */
#define NVIC_OSC_PLL_RTC      39          /* External main OSC/external sub OSC/main PLL/RTC interrupt request                 */
#define NVIC_ADC0             40          /* A/D Converter unit0                                                               */
#define NVIC_FRTIM            41          /* Free-run Timer unit0                                                              */
#define NVIC_INCAP            42          /* Input Capture unit0                                                               */
#define NVIC_OUTCOMP          43          /* Output Compare unit0                                                              */
#define NVIC_BTIM             44          /* Base Timer ch.0 to ch.7                                                           */

#endif    /* __IOMB9AF132K_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = MemManage      0x10
Interrupt3   = BusFault       0x14
Interrupt4   = UsageFault     0x18
Interrupt5   = SVC            0x2C
Interrupt6   = DebugMon       0x30
Interrupt7   = PendSV         0x38
Interrupt8   = SysTick        0x3C
Interrupt9   = CSV            0x40
Interrupt10  = SWDT           0x44
Interrupt11  = LVD            0x48
Interrupt12  = WFG            0x4C
Interrupt13  = EXTI0_6        0x50
Interrupt14  = EXTI15         0x54
Interrupt15  = MFSI0RX        0x58
Interrupt16  = MFSI0TX        0x5C
Interrupt17  = MFSI1RX        0x60
Interrupt18  = MFSI1TX        0x64
Interrupt19  = MFSI3RX        0x70
Interrupt20  = MFSI3TX        0x74
Interrupt21  = MFSI5RX        0x80
Interrupt22  = MFSI5TX        0x84
Interrupt23  = PPG            0x98
Interrupt24  = OSC_PLL_RTC    0x9C
Interrupt25  = ADC0           0xA0
Interrupt26  = FRTIM          0xA4
Interrupt27  = INCAP          0xA8
Interrupt28  = OUTCOMP        0xAC
Interrupt29  = BTIM           0xB0
 
###DDF-INTERRUPT-END###*/
