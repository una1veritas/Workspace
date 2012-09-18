/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC1315
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2012
 **
 **    $Revision: 49779 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC1315_H
#define __IOLPC1315_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC1315 SPECIAL FUNCTION REGISTERS
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

/* System memory remap register */
typedef struct {
  __REG32  MAP            : 2;
  __REG32                 :30;
} __sysmemremap_bits;

/* Peripheral reset control register */
typedef struct {
  __REG32  SSP0_RST_N     : 1;
  __REG32  I2C_RST_N      : 1;
  __REG32  SSP1_RST_N     : 1;
  __REG32                 :29;
} __presetctrl_bits;

/* System PLL control register */
/* USB PLL control register */
typedef struct {
  __REG32  MSEL           : 5;
  __REG32  PSEL           : 2;
  __REG32                 :25;
} __pllctrl_bits;

/* System PLL status register */
/* USB PLL status register */
typedef struct {
  __REG32  LOCK           : 1;
  __REG32                 :31;
} __pllstat_bits;

/* System oscillator control register */
typedef struct {
  __REG32  BYPASS         : 1;
  __REG32  FREQRANGE      : 1;
  __REG32                 :30;
} __sysoscctrl_bits;

/* WatchDog oscillator control register */
typedef struct {
  __REG32  DIVSEL         : 5;
  __REG32  FREQSEL        : 4;
  __REG32                 :23;
} __wdtoscctrl_bits;

/* System reset status register */
typedef struct {
  __REG32  POR            : 1;
  __REG32  EXTRST         : 1;
  __REG32  WDT            : 1;
  __REG32  BOD            : 1;
  __REG32  SYSRST         : 1;
  __REG32                 :27;
} __sysresstat_bits;

/* System PLL clock source select register */
/* Main clock source select register */
/* WDT clock source select register */
/* CLKOUT clock source select register */
typedef struct {
  __REG32  SEL            : 2;
  __REG32                 :30;
} __clksel_bits;

/* System AHB clock divider register */
/* SSP0 clock divider register */
/* SSP1 clock divider register */
/* UART clock divider register */
/* TRACECLKDIV clock divider register */
/* SYSTIC clock divider register */
/* WDT clock divider register */
/* CLKOUT clock divider register */
typedef struct {
  __REG32  DIV            : 8;
  __REG32                 :24;
} __clkdiv_bits;

/* System AHB clock control register */
typedef struct {
  __REG32  SYS            : 1;
  __REG32  ROM            : 1;
  __REG32  RAM            : 1;
  __REG32  FLASHREG       : 1;
  __REG32  FLASHARRAY     : 1;
  __REG32  I2C            : 1;
  __REG32  GPIO           : 1;
  __REG32  CT16B0         : 1;
  __REG32  CT16B1         : 1;
  __REG32  CT32B0         : 1;
  __REG32  CT32B1         : 1;
  __REG32  SSP0           : 1;
  __REG32  USART          : 1;
  __REG32  ADC            : 1;
  __REG32  USB            : 1;
  __REG32  WWDT           : 1;
  __REG32  IOCON          : 1;
  __REG32                 : 1;
  __REG32  SSP1           : 1;
  __REG32  PINT           : 1;
  __REG32                 : 3;
  __REG32  GROUP0INT      : 1;
  __REG32  GROUP1INT      : 1;
  __REG32                 : 7;
} __sysahbclkctrl_bits;

/* POR captured PIO status register 0 */
typedef struct {
  __REG32  PIOSTAT0_0      : 1;
  __REG32  PIOSTAT0_1      : 1;
  __REG32  PIOSTAT0_2      : 1;
  __REG32  PIOSTAT0_3      : 1;
  __REG32  PIOSTAT0_4      : 1;
  __REG32  PIOSTAT0_5      : 1;
  __REG32  PIOSTAT0_6      : 1;
  __REG32  PIOSTAT0_7      : 1;
  __REG32  PIOSTAT0_8      : 1;
  __REG32  PIOSTAT0_9      : 1;
  __REG32  PIOSTAT0_10     : 1;
  __REG32  PIOSTAT0_11     : 1;
  __REG32  PIOSTAT0_12     : 1;
  __REG32  PIOSTAT0_13     : 1;
  __REG32  PIOSTAT0_14     : 1;
  __REG32  PIOSTAT0_15     : 1;
  __REG32  PIOSTAT0_16     : 1;
  __REG32  PIOSTAT0_17     : 1;
  __REG32  PIOSTAT0_18     : 1;
  __REG32  PIOSTAT0_19     : 1;
  __REG32  PIOSTAT0_20     : 1;
  __REG32  PIOSTAT0_21     : 1;
  __REG32  PIOSTAT0_22     : 1;
  __REG32  PIOSTAT0_23     : 1;
  __REG32                  : 8;
} __pioporcap0_bits;

/* POR captured PIO status register 1 */
typedef struct {
  __REG32  PIOSTAT1_0      : 1;
  __REG32  PIOSTAT1_1      : 1;
  __REG32  PIOSTAT1_2      : 1;
  __REG32  PIOSTAT1_3      : 1;
  __REG32  PIOSTAT1_4      : 1;
  __REG32  PIOSTAT1_5      : 1;
  __REG32                  : 1;
  __REG32  PIOSTAT1_7      : 1;
  __REG32  PIOSTAT1_8      : 1;
  __REG32                  : 1;
  __REG32  PIOSTAT1_10     : 1;
  __REG32  PIOSTAT1_11     : 1;
  __REG32                  : 1;
  __REG32  PIOSTAT1_13     : 1;
  __REG32  PIOSTAT1_14     : 1;
  __REG32  PIOSTAT1_15     : 1;
  __REG32  PIOSTAT1_16     : 1;
  __REG32  PIOSTAT1_17     : 1;
  __REG32  PIOSTAT1_18     : 1;
  __REG32  PIOSTAT1_19     : 1;
  __REG32  PIOSTAT1_20     : 1;
  __REG32  PIOSTAT1_21     : 1;
  __REG32  PIOSTAT1_22     : 1;
  __REG32  PIOSTAT1_23     : 1;
  __REG32  PIOSTAT1_24     : 1;
  __REG32  PIOSTAT1_25     : 1;
  __REG32  PIOSTAT1_26     : 1;
  __REG32  PIOSTAT1_27     : 1;
  __REG32  PIOSTAT1_28     : 1;
  __REG32  PIOSTAT1_29     : 1;
  __REG32                  : 1;
  __REG32  PIOSTAT1_31     : 1;
} __pioporcap1_bits;

/* BOD control register */
typedef struct {
  __REG32  BODRSTLEV      : 2;
  __REG32  BODINTVAL      : 2;
  __REG32  BODRSTENA      : 1;
  __REG32                 :27;
} __bodctrl_bits;

/* System tick timer calibration register */
typedef struct {
  __REG32  CAL            :26;
  __REG32                 : 6;
} __systckcal_bits;

/* IQR delay register (IRQLATENCY) */
typedef struct {
  __REG32  LATENCY        : 8;
  __REG32                 :24;
} __irqlatency_bits;

/* NMI Source Control register (NMISRC) */
typedef struct {
  __REG32  IRQNO          : 5;
  __REG32                 :26;
  __REG32  NMIEN          : 1;
} __nmisrc_bits;

/* GPIO Pin Interrupt Select register (PINTSEL) */
typedef struct {
  __REG32  INTPIN         : 5;
  __REG32  PORTSEL        : 1;
  __REG32                 :26;
} __pintsel_bits;

/* Start logic 0 interrupt wake-up enable register 0 (STARTERP0)*/
typedef struct {
  __REG32  PINT0      : 1;
  __REG32  PINT1      : 1;
  __REG32  PINT2      : 1;
  __REG32  PINT3      : 1;
  __REG32  PINT4      : 1;
  __REG32  PINT5      : 1;
  __REG32  PINT6      : 1;
  __REG32  PINT7      : 1;
  __REG32             :24;
} __start_er_p0_bits;

/* Start logic 1 interrupt wake-up enable register (STARTERP1) */
typedef struct {
  __REG32                 :12;
  __REG32  WWDTINT        : 1;
  __REG32  BODINT         : 1;
  __REG32                 : 6;
  __REG32  GPIOINT0       : 1;
  __REG32  GPIOINT1       : 1;
  __REG32                 :10;
} __start_er_p1_bits;

/* Deep-sleep configuration register */
typedef struct {
  __REG32                 : 3;
  __REG32  BOD_PD         : 1;
  __REG32                 : 2;
  __REG32  WDTOSC_PD      : 1;
  __REG32                 :25;
} __pdsleepcfg_bits;

/* Wakeup configuration register */
typedef struct {
  __REG32  IRCOUT_PD      : 1;
  __REG32  IRC_PD         : 1;
  __REG32  FLASH_PD       : 1;
  __REG32  BOD_PD         : 1;
  __REG32  ADC_PD         : 1;
  __REG32  SYSOSC_PD      : 1;
  __REG32  WDTOSC_PD      : 1;
  __REG32  SYSPLL_PD      : 1;
  __REG32                 :24;
} __pdawakecfg_bits;

/* Power-down configuration register */
typedef struct {
  __REG32  IRCOUT_PD      : 1;
  __REG32  IRC_PD         : 1;
  __REG32  FLASH_PD       : 1;
  __REG32  BOD_PD         : 1;
  __REG32  ADC_PD         : 1;
  __REG32  SYSOSC_PD      : 1;
  __REG32  WDTOSC_PD      : 1;
  __REG32  SYSPLL_PD      : 1;
  __REG32                 :24;
} __pdruncfg_bits;

/* Power control register */
typedef struct {
  __REG32  PM             : 3;
  __REG32  NODPD          : 1;
  __REG32                 : 4;
  __REG32  SLEEPFLAG      : 1;
  __REG32                 : 2;
  __REG32  DPDFLAG        : 1;
  __REG32                 :20;
} __pcon_bits;

/* GPREGx */
typedef struct {
  __REG32  GPDATA         :32;
} __gpregx_bits;

/* GPREG4 */
typedef struct {
  __REG32                 :10;
  __REG32  WAKEUPHYS      : 1;
  __REG32  GPDATA         :21;
} __gpreg4_bits;

/* IOCON_x registers */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32  INV            : 1;
  __REG32                 : 3;
  __REG32  OD             : 1;
  __REG32                 :21;
} __iocon_bits;

/* IOCON_PIO0_4 register */
/* IOCON_PIO0_5 register */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32                 : 5;
  __REG32  I2CMODE        : 2; 
  __REG32                 :22;
} __iocon_pio0_4_bits;

/* I/O configuration for pin TDI/PIO0_11 */
/* I/O configuration for pin TMS/PIO0_12 */
/* I/O configuration for pin TDO/PIO0_13 */
/* I/O configuration for pin TRST/PIO0_14 */
/* I/O configuration for pin SWDIO/PIO0_15 */
/* I/O configuration for pin PIO0_16 */
/* I/O configuration for pin TDO/PIO0_22 */
/* I/O configuration for pin TDO/PIO0_23 */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32  INV            : 1;
  __REG32  ADMODE         : 1;
  __REG32  FILTR          : 1;
  __REG32                 : 1;
  __REG32  OD             : 1;
  __REG32                 :21;
} __iocon_pio0_11_bits;

/* Pin interrupt mode register */
typedef struct {
  __REG32  PMODE0             : 1;
  __REG32  PMODE1             : 1;
  __REG32  PMODE2             : 1;
  __REG32  PMODE3             : 1;
  __REG32  PMODE4             : 1;
  __REG32  PMODE5             : 1;
  __REG32  PMODE6             : 1;
  __REG32  PMODE7             : 1;
  __REG32                     :24;
} __gpio_isel_bits;

/* Pin interrupt level enable register */
typedef struct {
  __REG32  ENRL0              : 1;
  __REG32  ENRL1              : 1;
  __REG32  ENRL2              : 1;
  __REG32  ENRL3              : 1;
  __REG32  ENRL4              : 1;
  __REG32  ENRL5              : 1;
  __REG32  ENRL6              : 1;
  __REG32  ENRL7              : 1;
  __REG32                     :24;
} __gpio_ienr_bits;

/* Pin interrupt level set register */

#define GPIO_SIENR_SETENRL         (0x000000FFUL)

/* Pin interrupt level clear register */

#define GPIO_CIENR_CENRL           (0x000000FFUL)

/* Pin interrupt active level register */
typedef struct {
  __REG32  ENAF               : 8;
  __REG32                     :24;
} __gpio_ienf_bits;

/* Pin interrupt active level set register */

#define GPIO_SIENF_SETENAF         (0x000000FFUL)

/* Pin interrupt active level clear register */

#define GPIO_CIENF_CENAF           (0x000000FFUL)

/* Pin interrupt rising edge register */
typedef struct {
  __REG32  RDET0              : 1;
  __REG32  RDET1              : 1;
  __REG32  RDET2              : 1;
  __REG32  RDET3              : 1;
  __REG32  RDET4              : 1;
  __REG32  RDET5              : 1;
  __REG32  RDET6              : 1;
  __REG32  RDET7              : 1;
  __REG32                     :24;
} __gpio_rise_bits;

/* Pin interrupt falling edge register */
typedef struct {
  __REG32  FDET0              : 1;
  __REG32  FDET1              : 1;
  __REG32  FDET2              : 1;
  __REG32  FDET3              : 1;
  __REG32  FDET4              : 1;
  __REG32  FDET5              : 1;
  __REG32  FDET6              : 1;
  __REG32  FDET7              : 1;
  __REG32                     :24;
} __gpio_fall_bits;

/* Pin interrupt status register */
typedef struct {
  __REG32  PSTAT0             : 1;
  __REG32  PSTAT1             : 1;
  __REG32  PSTAT2             : 1;
  __REG32  PSTAT3             : 1;
  __REG32  PSTAT4             : 1;
  __REG32  PSTAT5             : 1;
  __REG32  PSTAT6             : 1;
  __REG32  PSTAT7             : 1;
  __REG32                     :24;
} __gpio_ist_bits;

/* Grouped interrupt control register */
typedef struct {
  __REG32  INT                : 1;
  __REG32  COMB               : 1;
  __REG32  TRIG               : 1;
  __REG32                     :29;
} __gpiogx_ctrl_bits;

/* GPIO0 pins */
typedef struct {
  __REG32 P0_0   : 1;
  __REG32 P0_1   : 1;
  __REG32 P0_2   : 1;
  __REG32 P0_3   : 1;
  __REG32 P0_4   : 1;
  __REG32 P0_5   : 1;
  __REG32 P0_6   : 1;
  __REG32 P0_7   : 1;
  __REG32 P0_8   : 1;
  __REG32 P0_9   : 1;
  __REG32 P0_10  : 1;
  __REG32 P0_11  : 1;
  __REG32 P0_12  : 1;
  __REG32 P0_13  : 1;
  __REG32 P0_14  : 1;
  __REG32 P0_15  : 1;
  __REG32 P0_16  : 1;
  __REG32 P0_17  : 1;
  __REG32 P0_18  : 1;
  __REG32 P0_19  : 1;
  __REG32 P0_20  : 1;
  __REG32 P0_21  : 1;
  __REG32 P0_22  : 1;
  __REG32 P0_23  : 1;
  __REG32        : 8;
} __gpio0_bits;

/* GPIO1 pins */
typedef struct {
  __REG32 P1_0   : 1;
  __REG32 P1_1   : 1;
  __REG32 P1_2   : 1;
  __REG32 P1_3   : 1;
  __REG32 P1_4   : 1;
  __REG32 P1_5   : 1;
  __REG32        : 1;
  __REG32 P1_7   : 1;
  __REG32 P1_8   : 1;
  __REG32        : 1;
  __REG32 P1_10  : 1;
  __REG32 P1_11  : 1;
  __REG32        : 1;
  __REG32 P1_13  : 1;
  __REG32 P1_14  : 1;
  __REG32 P1_15  : 1;
  __REG32 P1_16  : 1;
  __REG32 P1_17  : 1;
  __REG32 P1_18  : 1;
  __REG32 P1_19  : 1;
  __REG32 P1_20  : 1;
  __REG32 P1_21  : 1;
  __REG32 P1_22  : 1;
  __REG32 P1_23  : 1;
  __REG32 P1_24  : 1;
  __REG32 P1_25  : 1;
  __REG32 P1_26  : 1;
  __REG32 P1_27  : 1;
  __REG32 P1_28  : 1;
  __REG32 P1_29  : 1;
  __REG32        : 1;
  __REG32 P1_31  : 1;
} __gpio1_bits;

/* UART Receive Buffer Register (RBR) */
/* UART Transmit Holding Register (THR) */
/* UART Divisor Latch Register  Low (DLL) */
typedef union {
  /*UxRBR*/
  struct {
    __REG32 RBR           : 8;
    __REG32               :24;
  } ;
  /*UxTHR*/
  struct {
    __REG32 THR           : 8;
    __REG32               :24;
  } ;
  /*UxDLL*/
  struct {
    __REG32 DLL           : 8;
    __REG32               :24;
  } ;
} __uartrbr_bits;

/* UART Interrupt Enable Register (IER) */
/* UART Divisor Latch MSB Register (DLM) */
typedef union {
  /*UxIER*/
  struct {
    __REG32 RDAIE         : 1;
    __REG32 THREIE        : 1;
    __REG32 RXLSIE        : 1;
    __REG32 RXMSIE        : 1;
    __REG32               : 4;
    __REG32 ABTOINTEN     : 1;
    __REG32 ABEOINTEN     : 1;
    __REG32               :22;
  } ;
  /*UxDLM*/
  struct {
    __REG32 DLM           : 8;
    __REG32               :24;
  } ;
} __uartier_bits;

/* UART Interrupt Identification Register (IIR) */
/* UART FIFO Control Register (FCR) */
typedef union {
  /*UxIIR*/
  struct {
    __REG32 IP            : 1;
    __REG32 IID           : 3;
    __REG32               : 2;
    __REG32 IIRFE         : 2;
    __REG32 ABEOINT       : 1;
    __REG32 ABTOINT       : 1;
    __REG32               :22;
  };
  /*UxFCR*/
  struct {
    __REG32 FCRFE         : 1;
    __REG32 RFR           : 1;
    __REG32 TFR           : 1;
    __REG32               : 3;
    __REG32 RTLS          : 2;
    __REG32               :24;
  };
} __uartiir_bits;

/* UART Line Control Register (LCR) */
typedef struct {
  __REG32 WLS             : 2;
  __REG32 SBS             : 1;
  __REG32 PE              : 1;
  __REG32 PS              : 2;
  __REG32 BC              : 1;
  __REG32 DLAB            : 1;
  __REG32                 :24;
} __uartlcr_bits;

/* UART modem control register */
typedef struct{
  __REG32  DTR   : 1;
  __REG32  RTS   : 1;
  __REG32        : 2;
  __REG32  LMS   : 1;
  __REG32        : 1;
  __REG32  RTSEN : 1;
  __REG32  CTSEN : 1;
  __REG32        :24;
} __uartmcr_bits;

/* UART Line Status Register (LSR) */
typedef struct {
  __REG32 DR              : 1;
  __REG32 OE              : 1;
  __REG32 PE              : 1;
  __REG32 FE              : 1;
  __REG32 BI              : 1;
  __REG32 THRE            : 1;
  __REG32 TEMT            : 1;
  __REG32 RXFE            : 1;
  __REG32 TXERR           : 1;
  __REG32                 :23;
} __uartlsr_bits;

/* UART modem status register */
typedef union{
  /*UxMSR*/
  struct {
    __REG32  DCTS  : 1;
    __REG32  DDSR  : 1;
    __REG32  TERI  : 1;
    __REG32  DDCD  : 1;
    __REG32  CTS   : 1;
    __REG32  DSR   : 1;
    __REG32  RI    : 1;
    __REG32  DCD   : 1;
    __REG32        :24;
  };
  /*UxMSR*/
  struct {
    __REG32  MSR0  : 1;
    __REG32  MSR1  : 1;
    __REG32  MSR2  : 1;
    __REG32  MSR3  : 1;
    __REG32  MSR4  : 1;
    __REG32  MSR5  : 1;
    __REG32  MSR6  : 1;
    __REG32  MSR7  : 1;
    __REG32        :24;
  };
} __uartmsr_bits;

/* UART Auto-baud Control Register */
typedef struct{
  __REG32 START        : 1;
  __REG32 MODE         : 1;
  __REG32 AUTORESTART  : 1;
  __REG32              : 5;
  __REG32 ABEOINTCLR   : 1;
  __REG32 ABTOINTCLR   : 1;
  __REG32              :22;
} __uartacr_bits;

/* IrDA Control Register */
typedef struct{
  __REG32 IRDAEN       : 1;
  __REG32 IRDAINV      : 1;
  __REG32 FIXPULSEEN   : 1;
  __REG32 PULSEDIV     : 3;
  __REG32              :26;
} __uarticr_bits;

/* UART Fractional Divider Register */
typedef struct{
  __REG32 DIVADDVAL  : 4;
  __REG32 MULVAL     : 4;
  __REG32            :24;
} __uartfdr_bits;

/* USART Oversampling Register */
typedef struct{
  __REG32             : 1;
  __REG32 OSFRAC      : 3;
  __REG32 OSINT       : 4;
  __REG32 FDINT       : 7;
  __REG32             :17;
} __uartosr_bits;

/* Transmit Enable Register */
typedef struct{
  __REG32                 : 7;
  __REG32 TXEN            : 1;
  __REG32                 :24;
} __uartter_bits;

/* UART Half-duplex enable register */
typedef struct{
  __REG32 HDEN       : 1;
  __REG32            :31;
} __uarthden_bits;

/* UART Smart card interface control register */
typedef struct{
  __REG32 SCIEN      : 1;
  __REG32 NACKDIS    : 1;
  __REG32 PROTSEL    : 1;
  __REG32            : 2;
  __REG32 TXRETRY    : 3;
  __REG32 GUARDTIME  : 8;
  __REG32            :16;
} __uartscictrl_bits;

/* RS485 Control register */
typedef struct{
  __REG32 NMMEN           : 1;
  __REG32 RXDIS           : 1;
  __REG32 AADEN           : 1;
  __REG32 SEL             : 1;
  __REG32 DCTRL           : 1;
  __REG32 OINV            : 1;
  __REG32                 :26;
} __uartrs485ctrl_bits;

/* UART Synchronous mode control register */
typedef struct{
  __REG32 SYNC              : 1;
  __REG32 CSRC              : 1;
  __REG32 FES               : 1;
  __REG32 TSBYPASS          : 1;
  __REG32 CSCEN             : 1;
  __REG32 SSSDIS            : 1;
  __REG32 CCCLR             : 1;
  __REG32                   :25;
} __uartsyncctrl_bits;

/* SSP Control Register 0 */
typedef struct{
__REG32 DSS  : 4;
__REG32 FRF  : 2;
__REG32 CPOL : 1;
__REG32 CPHA : 1;
__REG32 SCR  : 8;
__REG32      :16;
} __sspcr0_bits;

/* SSP Control Register 1 */
typedef struct{
__REG32 LBM  : 1;
__REG32 SSE  : 1;
__REG32 MS   : 1;
__REG32 SOD  : 1;
__REG32      :28;
} __sspcr1_bits;

/* SSP Data Register */
typedef struct{
__REG32 DATA :16;
__REG32      :16;
} __sspdr_bits;

/* SSP Status Register */
typedef struct{
__REG32 TFE  : 1;
__REG32 TNF  : 1;
__REG32 RNE  : 1;
__REG32 RFF  : 1;
__REG32 BSY  : 1;
__REG32      :27;
} __sspsr_bits;

/* SSP Clock Prescale Register */
typedef struct{
__REG32 CPSDVSR : 8;
__REG32         :24;
} __sspcpsr_bits;

/* SSP Interrupt Mask Set/Clear Register */
typedef struct{
__REG32 RORIM  : 1;
__REG32 RTIM   : 1;
__REG32 RXIM   : 1;
__REG32 TXIM   : 1;
__REG32        :28;
} __sspimsc_bits;

/* SSP Raw Interrupt Status Register */
typedef struct{
__REG32 RORRIS  : 1;
__REG32 RTRIS   : 1;
__REG32 RXRIS   : 1;
__REG32 TXRIS   : 1;
__REG32         :28;
} __sspris_bits;

/* SSP Masked Interrupt Status Register */
typedef struct{
__REG32 RORMIS  : 1;
__REG32 RTMIS   : 1;
__REG32 RXMIS   : 1;
__REG32 TXMIS   : 1;
__REG32         :28;
} __sspmis_bits;

/* SSP Interrupt Clear Register */
typedef struct{
__REG32 RORIC  : 1;
__REG32 RTIC   : 1;
__REG32        :30;
} __sspicr_bits;

/* I2C control set register */
typedef struct{
__REG32       : 2;
__REG32 AA    : 1;
__REG32 SI    : 1;
__REG32 STO   : 1;
__REG32 STA   : 1;
__REG32 I2EN  : 1;
__REG32       :25;
} __i2conset_bits;

/* I2C control clear register */
typedef struct{
__REG32        : 2;
__REG32 AAC    : 1;
__REG32 SIC    : 1;
__REG32        : 1;
__REG32 STAC   : 1;
__REG32 I2ENC  : 1;
__REG32        :25;
} __i2conclr_bits;

/* I2C status register */
typedef struct{
__REG32 STATUS  : 8;
__REG32         :24;
} __i2stat_bits;

/* I2C data register */
typedef struct{
__REG32 DATA  : 8;
__REG32       :24;
} __i2dat_bits;

/* I2C Monitor mode control register */
typedef struct{
__REG32 MM_ENA    : 1;
__REG32 ENA_SCL   : 1;
__REG32 MATCH_ALL : 1;
__REG32           :29;
} __i2cmmctrl_bits;

/* I2C slave address register */
typedef struct{
__REG32 GC    : 1;
__REG32 ADDR  : 7;
__REG32       :24;
} __i2adr_bits;

/* I2C Mask registers */
typedef struct{
__REG32       : 1;
__REG32 MASK  : 7;
__REG32       :24;
} __i2cmask_bits;

/* I2C SCL High Duty Cycle register */
typedef struct{
__REG32 SCLH   :16;
__REG32        :16;
} __i2sch_bits;

/* I2C scl duty cycle register */
typedef struct{
__REG32 SCLL   :16;
__REG32        :16;
} __i2scl_bits;

/* Interrupt Register (TMR16B0IR and TMR16B1IR) */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32         :27;
} __tmr16ir_bits;

/* Interrupt Register (TMR32B0IR and TMR32B1IR) */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32 CR1INT  : 1;
__REG32         :26;
} __tmr32ir_bits;

/* Timer Control Register (TMR16B0TCR and TMR16B1TCR) */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tmr16tcr_bits;

/* Timer Control Register (TMR32B0TCR and TMR32B1TCR) */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tmr32tcr_bits;

/* Count Control Register (TMR16B0CTCR and TMR16B1CTCR) */
typedef struct{
__REG32 CTM   : 2;     /*Counter/Timer Mode*/
__REG32 CIS   : 2;     /*Count Input Select*/
__REG32 ENCC  : 1;     
__REG32 SELCC : 3;     
__REG32       :24;
} __tmr16ctcr_bits;

/* Count Control Register (TMR32B0CTCR and TMR32B1CTCR) */
typedef struct{
__REG32 CTM   : 2;     /*Counter/Timer Mode*/
__REG32 CIS   : 2;     /*Count Input Select*/
__REG32 ENCC  : 1;     
__REG32 SELCC : 3;     
__REG32       :24;
} __tmr32ctcr_bits;

/* Match Control Register (TMR16B0MCR and TMR16B1MCR) */
typedef struct{
__REG32 MR0I     : 1;
__REG32 MR0R     : 1;
__REG32 MR0S     : 1;
__REG32 MR1I     : 1;
__REG32 MR1R     : 1;
__REG32 MR1S     : 1;
__REG32 MR2I     : 1;
__REG32 MR2R     : 1;
__REG32 MR2S     : 1;
__REG32 MR3I     : 1;
__REG32 MR3R     : 1;
__REG32 MR3S     : 1;
__REG32          :20;
} __tmr16mcr_bits;

/* Match Control Register (TMR32B0MCR and TMR32B1MCR) */
typedef struct{
__REG32 MR0I     : 1;
__REG32 MR0R     : 1;
__REG32 MR0S     : 1;
__REG32 MR1I     : 1;
__REG32 MR1R     : 1;
__REG32 MR1S     : 1;
__REG32 MR2I     : 1;
__REG32 MR2R     : 1;
__REG32 MR2S     : 1;
__REG32 MR3I     : 1;
__REG32 MR3R     : 1;
__REG32 MR3S     : 1;
__REG32          :20;
} __tmr32mcr_bits;

/* Capture Control Register (TMR16B0CCR and TMR16B1CCR) */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32          :29;
} __tmr16tccr_bits;

/* Capture Control Register (TMR32B0CCR and TMR32B1CCR) */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32 CAP1RE   : 1;
__REG32 CAP1FE   : 1;
__REG32 CAP1I    : 1;
__REG32          :26;
} __tmr32tccr_bits;

/* External Match Register (TMR16B0EMR and TMR16B1EMR) */
typedef struct{
__REG32 EM0   : 1;
__REG32 EM1   : 1;
__REG32 EM2   : 1;
__REG32 EM3   : 1;
__REG32 EMC0  : 2;
__REG32 EMC1  : 2;
__REG32 EMC2  : 2;
__REG32 EMC3  : 2;
__REG32       :20;
} __tmr16emr_bits;

/* External Match Register (TMR32B0EMR and TMR32B1EMR) */
typedef struct{
__REG32 EM0   : 1;
__REG32 EM1   : 1;
__REG32 EM2   : 1;
__REG32 EM3   : 1;
__REG32 EMC0  : 2;
__REG32 EMC1  : 2;
__REG32 EMC2  : 2;
__REG32 EMC3  : 2;
__REG32       :20;
} __tmr32emr_bits;

/* PWM Control register (TMR16B0PWMC and TMR16B1PWMC) */
typedef struct{
__REG32 PWM0ENA  : 1;
__REG32 PWM1ENA  : 1;
__REG32 PWM2ENA  : 1;
__REG32 PWM3ENA  : 1;
__REG32          :28;
} __tmr16pwmc_bits;

/* PWM Control register (TMR32B0PWMC and TMR32B1PWMC) */
typedef struct{
__REG32 PWM0ENA  : 1;
__REG32 PWM1ENA  : 1;
__REG32 PWM2ENA  : 1;
__REG32 PWM3ENA  : 1;
__REG32          :28;
} __tmr32pwmc_bits;

/* A/D Control Register */
typedef union {
   /* ADxCR */
  struct {
    __REG32 SEL0                 : 1;
    __REG32 SEL1                 : 1;
    __REG32 SEL2                 : 1;
    __REG32 SEL3                 : 1;
    __REG32 SEL4                 : 1;
    __REG32 SEL5                 : 1;
    __REG32 SEL6                 : 1;
    __REG32 SEL7                 : 1;
    __REG32 CLKDIV               : 8;
    __REG32 BURST                : 1;
    __REG32                      : 4;
    __REG32 PDN                  : 1;
    __REG32 LPWRMODE             : 1;
    __REG32 _10BITMODE           : 1;
    __REG32 START                : 3;
    __REG32 EDGE                 : 1;
    __REG32                      : 4;
  };
  /* ADxCR */
  struct {
    __REG32 SEL                  :  8;
    __REG32                      : 24;
  };
} __adcr_bits;

/* A/D Global Data Register */
typedef struct{
__REG32         : 4;
__REG32 V_VREF  :12;
__REG32         : 8;
__REG32 CHN     : 3;
__REG32         : 3;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __adgdr_bits;

/* A/D Status Register */
typedef struct{
__REG32 DONE0     : 1;
__REG32 DONE1     : 1;
__REG32 DONE2     : 1;
__REG32 DONE3     : 1;
__REG32 DONE4     : 1;
__REG32 DONE5     : 1;
__REG32 DONE6     : 1;
__REG32 DONE7     : 1;
__REG32 OVERRUN0  : 1;
__REG32 OVERRUN1  : 1;
__REG32 OVERRUN2  : 1;
__REG32 OVERRUN3  : 1;
__REG32 OVERRUN4  : 1;
__REG32 OVERRUN5  : 1;
__REG32 OVERRUN6  : 1;
__REG32 OVERRUN7  : 1;
__REG32 ADINT     : 1;
__REG32           :15;
} __adstat_bits;

/* A/D Intrrupt Enable Register */
typedef struct{
__REG32 ADINTEN0  : 1;
__REG32 ADINTEN1  : 1;
__REG32 ADINTEN2  : 1;
__REG32 ADINTEN3  : 1;
__REG32 ADINTEN4  : 1;
__REG32 ADINTEN5  : 1;
__REG32 ADINTEN6  : 1;
__REG32 ADINTEN7  : 1;
__REG32 ADGINTEN  : 1;
__REG32           :23;
} __adinten_bits;

/* A/D Data Register */
typedef struct{
__REG32         : 4;
__REG32 V_VREF  :12;
__REG32         :14;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __addr_bits;

/* A/D Trim register (TRM) */
typedef struct{
__REG32         : 4;
__REG32 ADCOFFS : 4;
__REG32 TRIM    : 4;
__REG32         :20;
} __adtrm_bits;

/* A/D input select register (INSEL) */
typedef struct{
__REG32         :10;
__REG32 AD5SEL  : 2;
__REG32 AD6SEL  : 2;
__REG32 AD7SEL  : 2;
__REG32         :16;
} __adinsel_bits;

/* Watchdog mode register */
typedef struct{
__REG32 WDEN      : 1;
__REG32 WDRESET   : 1;
__REG32 WDTOF     : 1;
__REG32 WDINT     : 1;
__REG32 WDPROTECT : 1;
__REG32 LOCK      : 1;
__REG32           :26;
} __wdmod_bits;

/* Watchdog Timer Constant register */
typedef struct{
__REG32 COUNT     :24;
__REG32           : 8;
} __wdtc_bits;

/* Watchdog feed register */
typedef struct{
__REG32 FEED  : 8;
__REG32       :24;
} __wdfeed_bits;

/* Watchdog Clock Select register */
typedef struct {
__REG32 CLKSEL    : 1;
__REG32           :30;
__REG32 LOCK      : 1;
} __wdclksel_bits;

/* Watchdog Timer Warning Interrupt register */
typedef struct {
__REG32 WARNINT   :10;
__REG32           :22;
} __wdwarnint_bits;

/* Watchdog Timer Window register */
typedef struct {
__REG32 WINDOW    :24;
__REG32           : 8;
} __wdwindow_bits;

/* Flash configuration register */
typedef struct{
__REG32 FLASHTIM  : 2;
__REG32           :30;
} __flashcfg_bits;

/* Flash module signature start register */
typedef struct{
__REG32 START     :17;
__REG32           :15;
} __fmsstart_bits;

/* Flash module signature stop register */
typedef struct{
__REG32 STOP      :17;
__REG32 SIG_START : 1;
__REG32           :14;
} __fmsstop_bits;

/* Flash module status register */
typedef struct{
__REG32           : 2;
__REG32 SIG_DONE  : 1;
__REG32           :29;
} __fmstat_bits;

#define FMSTATCLR_SIG_DONE_CLR  (0x00000004UL)

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                  0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,          0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,               0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(ACTIVE0,               0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(IP0,                   0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                   0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                   0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                   0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                   0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                   0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                   0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                   0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(CPUIDBR,               0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,                  0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,                 0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,                 0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,                 0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                  0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                  0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                  0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,                 0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                  0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,                  0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** System control block
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSMEMREMAP,           0x40048000,__READ_WRITE ,__sysmemremap_bits);
__IO_REG32_BIT(PRESETCTRL,            0x40048004,__READ_WRITE ,__presetctrl_bits);
__IO_REG32_BIT(SYSPLLCTRL,            0x40048008,__READ_WRITE ,__pllctrl_bits);
__IO_REG32_BIT(SYSPLLSTAT,            0x4004800C,__READ       ,__pllstat_bits);
__IO_REG32_BIT(SYSOSCCTRL,            0x40048020,__READ_WRITE ,__sysoscctrl_bits);
__IO_REG32_BIT(WDTOSCCTRL,            0x40048024,__READ_WRITE ,__wdtoscctrl_bits);
__IO_REG32_BIT(SYSRESSTAT,            0x40048030,__READ       ,__sysresstat_bits);
__IO_REG32_BIT(SYSPLLCLKSEL,          0x40048040,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(MAINCLKSEL,            0x40048070,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(SYSAHBCLKDIV,          0x40048078,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(SYSAHBCLKCTRL,         0x40048080,__READ_WRITE ,__sysahbclkctrl_bits);
__IO_REG32_BIT(SSP0CLKDIV,            0x40048094,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(UARTCLKDIV,            0x40048098,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(SSP1CLKDIV,            0x4004809C,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(TRACECLKDIV,           0x400480AC,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(SYSTICKCLKDIV,         0x400480B0,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(CLKOUTSEL,             0x400480E0,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(CLKOUTDIV,             0x400480E8,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(PIOPORCAP0,            0x40048100,__READ       ,__pioporcap0_bits);
__IO_REG32_BIT(PIOPORCAP1,            0x40048104,__READ       ,__pioporcap1_bits);
__IO_REG32_BIT(BODCTRL,               0x40048150,__READ_WRITE ,__bodctrl_bits);
__IO_REG32_BIT(SYSTCKCAL,             0x40048158,__READ_WRITE ,__systckcal_bits);
__IO_REG32_BIT(IRQLATENCY,            0x40048170,__READ_WRITE ,__irqlatency_bits);
__IO_REG32_BIT(NMISRC,                0x40048174,__READ_WRITE ,__nmisrc_bits);
__IO_REG32_BIT(PINTSEL,               0x40048178,__READ_WRITE ,__pintsel_bits);
__IO_REG32_BIT(STARTERP0,             0x40048204,__READ_WRITE ,__start_er_p0_bits);
__IO_REG32_BIT(STARTERP1,             0x40048214,__READ_WRITE ,__start_er_p1_bits);
__IO_REG32_BIT(PDSLEEPCFG,            0x40048230,__READ_WRITE ,__pdsleepcfg_bits);
__IO_REG32_BIT(PDAWAKECFG,            0x40048234,__READ_WRITE ,__pdawakecfg_bits);
__IO_REG32_BIT(PDRUNCFG,              0x40048238,__READ_WRITE ,__pdruncfg_bits);
__IO_REG32(    DEVICE_ID,             0x400483F4,__READ       );

/***************************************************************************
 **
 ** PMU
 **
 ***************************************************************************/
__IO_REG32_BIT(PCON,                  0x40038000,__READ_WRITE ,__pcon_bits);
__IO_REG32_BIT(GPREG0,                0x40038004,__READ_WRITE ,__gpregx_bits);
__IO_REG32_BIT(GPREG1,                0x40038008,__READ_WRITE ,__gpregx_bits);
__IO_REG32_BIT(GPREG2,                0x4003800C,__READ_WRITE ,__gpregx_bits);
__IO_REG32_BIT(GPREG3,                0x40038010,__READ_WRITE ,__gpregx_bits);
__IO_REG32_BIT(GPREG4,                0x40038014,__READ_WRITE ,__gpreg4_bits);

/***************************************************************************
 **
 ** I/O configuration
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCON_RESET_PIO0_0,    0x40044000,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_1,          0x40044004,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_2,          0x40044008,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_3,          0x4004400C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_4,          0x40044010,__READ_WRITE ,__iocon_pio0_4_bits);
__IO_REG32_BIT(IOCON_PIO0_5,          0x40044014,__READ_WRITE ,__iocon_pio0_4_bits);
__IO_REG32_BIT(IOCON_PIO0_6,          0x40044018,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_7,          0x4004401C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_8,          0x40044020,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_9,          0x40044024,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_SWCLK_PIO0_10,   0x40044028,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_TDI_PIO0_11,     0x4004402C,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_TMS_PIO0_12,     0x40044030,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_TDO_PIO0_13,     0x40044034,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_TRST_PIO0_14,    0x40044038,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_SWDIO_PIO0_15,   0x4004403C,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_PIO0_16,         0x40044040,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_PIO0_17,         0x40044044,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_18,         0x40044048,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_19,         0x4004404C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_20,         0x40044050,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_21,         0x40044054,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_22,         0x40044058,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_PIO0_23,         0x4004405C,__READ_WRITE ,__iocon_pio0_11_bits);
__IO_REG32_BIT(IOCON_PIO1_0,          0x40044060,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_1,          0x40044064,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_2,          0x40044068,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_3,          0x4004406C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_4,          0x40044070,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_5,          0x40044074,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_7,          0x4004407C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_8,          0x40044080,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_10,         0x40044088,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_11,         0x4004408C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_13,         0x40044094,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_14,         0x40044098,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_15,         0x4004409C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_16,         0x400440A0,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_17,         0x400440A4,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_18,         0x400440A8,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_19,         0x400440AC,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_20,         0x400440B0,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_21,         0x400440B4,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_22,         0x400440B8,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_23,         0x400440BC,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_24,         0x400440C0,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_25,         0x400440C4,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_26,         0x400440C8,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_27,         0x400440CC,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_28,         0x400440D0,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_29,         0x400440D4,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_31,         0x400440DC,__READ_WRITE ,__iocon_bits);

/***************************************************************************
 **
 ** GPIO INT
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO_ISEL,         0x4004C000,__READ_WRITE ,__gpio_isel_bits);
__IO_REG32_BIT(GPIO_IENR,         0x4004C004,__READ_WRITE ,__gpio_ienr_bits);
__IO_REG32(    GPIO_SIENR,        0x4004C008,__WRITE      );
__IO_REG32(    GPIO_CIENR,        0x4004C00C,__WRITE      );
__IO_REG32_BIT(GPIO_IENF,         0x4004C010,__READ_WRITE ,__gpio_ienf_bits);
__IO_REG32(    GPIO_SIENF,        0x4004C014,__WRITE      );
__IO_REG32(    GPIO_CIENF,        0x4004C018,__WRITE      );
__IO_REG32_BIT(GPIO_RISE,         0x4004C01C,__READ_WRITE ,__gpio_rise_bits);
__IO_REG32_BIT(GPIO_FALL,         0x4004C020,__READ_WRITE ,__gpio_fall_bits);
__IO_REG32_BIT(GPIO_IST,          0x4004C024,__READ_WRITE ,__gpio_ist_bits);

/***************************************************************************
 **
 ** GPIO GROUP0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOG0_CTRL,       0x4005C000,__READ_WRITE ,__gpiogx_ctrl_bits);
__IO_REG32_BIT(GPIOG0_PORT_POL0,  0x4005C020,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIOG0_PORT_POL1,  0x4005C024,__READ_WRITE ,__gpio1_bits);
__IO_REG32_BIT(GPIOG0_PORT_ENA0,  0x4005C040,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIOG0_PORT_ENA1,  0x4005C044,__READ_WRITE ,__gpio1_bits);

/***************************************************************************
 **
 ** GPIO GROUP1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOG1_CTRL,       0x40060000,__READ_WRITE ,__gpiogx_ctrl_bits);
__IO_REG32_BIT(GPIOG1_PORT_POL0,  0x40060020,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIOG1_PORT_POL1,  0x40060024,__READ_WRITE ,__gpio1_bits);
__IO_REG32_BIT(GPIOG1_PORT_ENA0,  0x40060040,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIOG1_PORT_ENA1,  0x40060044,__READ_WRITE ,__gpio1_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG8(     GPIO_B0,           0x50000000,__READ_WRITE );
__IO_REG8(     GPIO_B1,           0x50000001,__READ_WRITE );
__IO_REG8(     GPIO_B2,           0x50000002,__READ_WRITE );
__IO_REG8(     GPIO_B3,           0x50000003,__READ_WRITE );
__IO_REG8(     GPIO_B4,           0x50000004,__READ_WRITE );
__IO_REG8(     GPIO_B5,           0x50000005,__READ_WRITE );
__IO_REG8(     GPIO_B6,           0x50000006,__READ_WRITE );
__IO_REG8(     GPIO_B7,           0x50000007,__READ_WRITE );
__IO_REG8(     GPIO_B8,           0x50000008,__READ_WRITE );
__IO_REG8(     GPIO_B9,           0x50000009,__READ_WRITE );
__IO_REG8(     GPIO_B10,          0x5000000A,__READ_WRITE );
__IO_REG8(     GPIO_B11,          0x5000000B,__READ_WRITE );
__IO_REG8(     GPIO_B12,          0x5000000C,__READ_WRITE );
__IO_REG8(     GPIO_B13,          0x5000000D,__READ_WRITE );
__IO_REG8(     GPIO_B14,          0x5000000E,__READ_WRITE );
__IO_REG8(     GPIO_B15,          0x5000000F,__READ_WRITE );
__IO_REG8(     GPIO_B16,          0x50000010,__READ_WRITE );
__IO_REG8(     GPIO_B17,          0x50000011,__READ_WRITE );
__IO_REG8(     GPIO_B18,          0x50000012,__READ_WRITE );
__IO_REG8(     GPIO_B19,          0x50000013,__READ_WRITE );
__IO_REG8(     GPIO_B20,          0x50000014,__READ_WRITE );
__IO_REG8(     GPIO_B21,          0x50000015,__READ_WRITE );
__IO_REG8(     GPIO_B22,          0x50000016,__READ_WRITE );
__IO_REG8(     GPIO_B23,          0x50000017,__READ_WRITE );
__IO_REG8(     GPIO_B32,          0x50000020,__READ_WRITE );
__IO_REG8(     GPIO_B33,          0x50000021,__READ_WRITE );
__IO_REG8(     GPIO_B34,          0x50000022,__READ_WRITE );
__IO_REG8(     GPIO_B35,          0x50000023,__READ_WRITE );
__IO_REG8(     GPIO_B36,          0x50000024,__READ_WRITE );
__IO_REG8(     GPIO_B37,          0x50000025,__READ_WRITE );
__IO_REG8(     GPIO_B39,          0x50000027,__READ_WRITE );
__IO_REG8(     GPIO_B40,          0x50000028,__READ_WRITE );
__IO_REG8(     GPIO_B42,          0x5000002A,__READ_WRITE );
__IO_REG8(     GPIO_B43,          0x5000002B,__READ_WRITE );
__IO_REG8(     GPIO_B45,          0x5000002D,__READ_WRITE );
__IO_REG8(     GPIO_B46,          0x5000002E,__READ_WRITE );
__IO_REG8(     GPIO_B47,          0x5000002F,__READ_WRITE );
__IO_REG8(     GPIO_B48,          0x50000030,__READ_WRITE );
__IO_REG8(     GPIO_B49,          0x50000031,__READ_WRITE );
__IO_REG8(     GPIO_B50,          0x50000032,__READ_WRITE );
__IO_REG8(     GPIO_B51,          0x50000033,__READ_WRITE );
__IO_REG8(     GPIO_B52,          0x50000034,__READ_WRITE );
__IO_REG8(     GPIO_B53,          0x50000035,__READ_WRITE );
__IO_REG8(     GPIO_B54,          0x50000036,__READ_WRITE );
__IO_REG8(     GPIO_B55,          0x50000037,__READ_WRITE );
__IO_REG8(     GPIO_B56,          0x50000038,__READ_WRITE );
__IO_REG8(     GPIO_B57,          0x50000039,__READ_WRITE );
__IO_REG8(     GPIO_B58,          0x5000003A,__READ_WRITE );
__IO_REG8(     GPIO_B59,          0x5000003B,__READ_WRITE );
__IO_REG8(     GPIO_B60,          0x5000003C,__READ_WRITE );
__IO_REG8(     GPIO_B61,          0x5000003D,__READ_WRITE );
__IO_REG8(     GPIO_B63,          0x5000003F,__READ_WRITE );
__IO_REG32(    GPIO_W0,           0x50001000,__READ_WRITE );
__IO_REG32(    GPIO_W1,           0x50001004,__READ_WRITE );
__IO_REG32(    GPIO_W2,           0x50001008,__READ_WRITE );
__IO_REG32(    GPIO_W3,           0x5000100C,__READ_WRITE );
__IO_REG32(    GPIO_W4,           0x50001010,__READ_WRITE );
__IO_REG32(    GPIO_W5,           0x50001014,__READ_WRITE );
__IO_REG32(    GPIO_W6,           0x50001018,__READ_WRITE );
__IO_REG32(    GPIO_W7,           0x5000101C,__READ_WRITE );
__IO_REG32(    GPIO_W8,           0x50001020,__READ_WRITE );
__IO_REG32(    GPIO_W9,           0x50001024,__READ_WRITE );
__IO_REG32(    GPIO_W10,          0x50001028,__READ_WRITE );
__IO_REG32(    GPIO_W11,          0x5000102C,__READ_WRITE );
__IO_REG32(    GPIO_W12,          0x50001030,__READ_WRITE );
__IO_REG32(    GPIO_W13,          0x50001034,__READ_WRITE );
__IO_REG32(    GPIO_W14,          0x50001038,__READ_WRITE );
__IO_REG32(    GPIO_W15,          0x5000103C,__READ_WRITE );
__IO_REG32(    GPIO_W16,          0x50001040,__READ_WRITE );
__IO_REG32(    GPIO_W17,          0x50001044,__READ_WRITE );
__IO_REG32(    GPIO_W18,          0x50001048,__READ_WRITE );
__IO_REG32(    GPIO_W19,          0x5000104C,__READ_WRITE );
__IO_REG32(    GPIO_W20,          0x50001050,__READ_WRITE );
__IO_REG32(    GPIO_W21,          0x50001054,__READ_WRITE );
__IO_REG32(    GPIO_W22,          0x50001058,__READ_WRITE );
__IO_REG32(    GPIO_W23,          0x5000105C,__READ_WRITE );
__IO_REG32(    GPIO_W32,          0x50001080,__READ_WRITE );
__IO_REG32(    GPIO_W33,          0x50001084,__READ_WRITE );
__IO_REG32(    GPIO_W34,          0x50001088,__READ_WRITE );
__IO_REG32(    GPIO_W35,          0x5000108C,__READ_WRITE );
__IO_REG32(    GPIO_W36,          0x50001090,__READ_WRITE );
__IO_REG32(    GPIO_W37,          0x50001094,__READ_WRITE );
__IO_REG32(    GPIO_W39,          0x5000109C,__READ_WRITE );
__IO_REG32(    GPIO_W40,          0x500010A0,__READ_WRITE );
__IO_REG32(    GPIO_W42,          0x500010A8,__READ_WRITE );
__IO_REG32(    GPIO_W43,          0x500010AC,__READ_WRITE );
__IO_REG32(    GPIO_W45,          0x500010B4,__READ_WRITE );
__IO_REG32(    GPIO_W46,          0x500010B8,__READ_WRITE );
__IO_REG32(    GPIO_W47,          0x500010BC,__READ_WRITE );
__IO_REG32(    GPIO_W48,          0x500010C0,__READ_WRITE );
__IO_REG32(    GPIO_W49,          0x500010C4,__READ_WRITE );
__IO_REG32(    GPIO_W50,          0x500010C8,__READ_WRITE );
__IO_REG32(    GPIO_W51,          0x500010CC,__READ_WRITE );
__IO_REG32(    GPIO_W52,          0x500010D0,__READ_WRITE );
__IO_REG32(    GPIO_W53,          0x500010D4,__READ_WRITE );
__IO_REG32(    GPIO_W54,          0x500010D8,__READ_WRITE );
__IO_REG32(    GPIO_W55,          0x500010DC,__READ_WRITE );
__IO_REG32(    GPIO_W56,          0x500010E0,__READ_WRITE );
__IO_REG32(    GPIO_W57,          0x500010E4,__READ_WRITE );
__IO_REG32(    GPIO_W58,          0x500010E8,__READ_WRITE );
__IO_REG32(    GPIO_W59,          0x500010EC,__READ_WRITE );
__IO_REG32(    GPIO_W60,          0x500010F0,__READ_WRITE );
__IO_REG32(    GPIO_W61,          0x500010F4,__READ_WRITE );
__IO_REG32(    GPIO_W63,          0x500010FC,__READ_WRITE );
__IO_REG32_BIT(GPIO_DIR0,         0x50002000,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIO_DIR1,         0x50002004,__READ_WRITE ,__gpio1_bits);
__IO_REG32_BIT(GPIO_MASK0,        0x50002080,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIO_MASK1,        0x50002084,__READ_WRITE ,__gpio1_bits);
__IO_REG32_BIT(GPIO_PIN0,         0x50002100,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIO_PIN1,         0x50002104,__READ_WRITE ,__gpio1_bits);
__IO_REG32_BIT(GPIO_MPIN0,        0x50002180,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIO_MPIN1,        0x50002184,__READ_WRITE ,__gpio1_bits);
__IO_REG32_BIT(GPIO_SET0,         0x50002200,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(GPIO_SET1,         0x50002204,__READ_WRITE ,__gpio1_bits);
__IO_REG32(    GPIO_CLR0,         0x50002280,__WRITE );
__IO_REG32(    GPIO_CLR1,         0x50002284,__WRITE );
__IO_REG32(    GPIO_NOT0,         0x50002300,__WRITE );
__IO_REG32(    GPIO_NOT1,         0x50002304,__WRITE );

/***************************************************************************
 **
 **  USART
 **
 ***************************************************************************/
/* U0DLL, U0RBR and U0THR share the same address */
__IO_REG32_BIT(U0RBR,                 0x40008000,__READ_WRITE  ,__uartrbr_bits);
#define U0THR       U0RBR
#define U0THR_bit   U0RBR_bit
#define U0DLL       U0RBR
#define U0DLL_bit   U0RBR_bit

/* U0DLM and U0IER share the same address */
__IO_REG32_BIT(U0IER,                 0x40008004,__READ_WRITE ,__uartier_bits);
#define U0DLM       U0IER
#define U0DLM_bit   U0IER_bit
/* U0FCR and U0IIR share the same address */
__IO_REG32_BIT(U0IIR,                 0x40008008,__READ_WRITE ,__uartiir_bits);
#define U0FCR       U0IIR
#define U0FCR_bit   U0IIR_bit

__IO_REG32_BIT(U0LCR,                 0x4000800C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U0MCR,                 0x40008010,__READ_WRITE ,__uartmcr_bits);
__IO_REG32_BIT(U0LSR,                 0x40008014,__READ       ,__uartlsr_bits);
__IO_REG32_BIT(U0MSR,                 0x40008018,__READ       ,__uartmsr_bits);
__IO_REG8(     U0SCR,                 0x4000801C,__READ_WRITE);
__IO_REG32_BIT(U0ACR,                 0x40008020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U0ICR,                 0x40008024,__READ_WRITE ,__uarticr_bits);
__IO_REG32_BIT(U0FDR,                 0x40008028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U0OSR,                 0x4000802C,__READ_WRITE ,__uartosr_bits);
__IO_REG32_BIT(U0TER,                 0x40008030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U0HDEN,                0x40008040,__READ_WRITE ,__uarthden_bits);
__IO_REG32_BIT(U0SCICTRL,             0x40008048,__READ_WRITE ,__uartscictrl_bits);
__IO_REG32_BIT(U0RS485CTRL,           0x4000804C,__READ_WRITE ,__uartrs485ctrl_bits);
__IO_REG8(     U0RS485ADRMATCH,       0x40008050,__READ_WRITE );
__IO_REG8(     U0RS485DLY,            0x40008054,__READ_WRITE );
__IO_REG32_BIT(U0SYNCCTRL,            0x40008058,__READ_WRITE ,__uartsyncctrl_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,               0x40040000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,               0x40040004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,                0x40040008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,                0x4004000C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,              0x40040010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,              0x40040014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,               0x40040018,__READ_WRITE ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,               0x4004001C,__READ_WRITE ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,               0x40040020,__WRITE      ,__sspicr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,               0x40058000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,               0x40058004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,                0x40058008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,                0x4005800C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,              0x40058010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,              0x40058014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,               0x40058018,__READ_WRITE ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,               0x4005801C,__READ_WRITE ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,               0x40058020,__WRITE      ,__sspicr_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CONSET,            0x40000000,__READ_WRITE ,__i2conset_bits);
__IO_REG32_BIT(I2C0STAT,              0x40000004,__READ       ,__i2stat_bits);
__IO_REG32_BIT(I2C0DAT,               0x40000008,__READ_WRITE ,__i2dat_bits);
__IO_REG32_BIT(I2C0ADR,               0x4000000C,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0SCLH,              0x40000010,__READ_WRITE ,__i2sch_bits);
__IO_REG32_BIT(I2C0SCLL,              0x40000014,__READ_WRITE ,__i2scl_bits);
__IO_REG32_BIT(I2C0CONCLR,            0x40000018,__WRITE      ,__i2conclr_bits);
__IO_REG32_BIT(I2C0MMCTRL,            0x4000001C,__READ_WRITE ,__i2cmmctrl_bits);
__IO_REG32_BIT(I2C0ADR1,              0x40000020,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR2,              0x40000024,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR3,              0x40000028,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0DATABUFFER,        0x4000002C,__READ       ,__i2dat_bits);
__IO_REG32_BIT(I2C0MASK0,             0x40000030,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK1,             0x40000034,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK2,             0x40000038,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK3,             0x4000003C,__READ_WRITE ,__i2cmask_bits);

/***************************************************************************
 **
 ** CT16B0
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR16B0IR,             0x4000C000,__READ_WRITE ,__tmr16ir_bits);
__IO_REG32_BIT(TMR16B0TCR,            0x4000C004,__READ_WRITE ,__tmr16tcr_bits);
__IO_REG16(    TMR16B0TC,             0x4000C008,__READ_WRITE);
__IO_REG16(    TMR16B0PR,             0x4000C00C,__READ_WRITE);
__IO_REG16(    TMR16B0PC,             0x4000C010,__READ_WRITE);
__IO_REG32_BIT(TMR16B0MCR,            0x4000C014,__READ_WRITE ,__tmr16mcr_bits);
__IO_REG16(    TMR16B0MR0,            0x4000C018,__READ_WRITE);
__IO_REG16(    TMR16B0MR1,            0x4000C01C,__READ_WRITE);
__IO_REG16(    TMR16B0MR2,            0x4000C020,__READ_WRITE);
__IO_REG16(    TMR16B0MR3,            0x4000C024,__READ_WRITE);
__IO_REG32_BIT(TMR16B0CCR,            0x4000C028,__READ_WRITE ,__tmr16tccr_bits);
__IO_REG16(    TMR16B0CR0,            0x4000C02C,__READ);
__IO_REG32_BIT(TMR16B0EMR,            0x4000C03C,__READ_WRITE ,__tmr16emr_bits);
__IO_REG32_BIT(TMR16B0CTCR,           0x4000C070,__READ_WRITE ,__tmr16ctcr_bits);
__IO_REG32_BIT(TMR16B0PWMC,           0x4000C074,__READ_WRITE ,__tmr16pwmc_bits);

/***************************************************************************
 **
 ** CT16B1
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR16B1IR,             0x40010000,__READ_WRITE ,__tmr16ir_bits);
__IO_REG32_BIT(TMR16B1TCR,            0x40010004,__READ_WRITE ,__tmr16tcr_bits);
__IO_REG16(    TMR16B1TC,             0x40010008,__READ_WRITE);
__IO_REG16(    TMR16B1PR,             0x4001000C,__READ_WRITE);
__IO_REG16(    TMR16B1PC,             0x40010010,__READ_WRITE);
__IO_REG32_BIT(TMR16B1MCR,            0x40010014,__READ_WRITE ,__tmr16mcr_bits);
__IO_REG16(    TMR16B1MR0,            0x40010018,__READ_WRITE);
__IO_REG16(    TMR16B1MR1,            0x4001001C,__READ_WRITE);
__IO_REG16(    TMR16B1MR2,            0x40010020,__READ_WRITE);
__IO_REG16(    TMR16B1MR3,            0x40010024,__READ_WRITE);
__IO_REG32_BIT(TMR16B1CCR,            0x40010028,__READ_WRITE ,__tmr16tccr_bits);
__IO_REG16(    TMR16B1CR0,            0x4001002C,__READ);
__IO_REG32_BIT(TMR16B1EMR,            0x4001003C,__READ_WRITE ,__tmr16emr_bits);
__IO_REG32_BIT(TMR16B1CTCR,           0x40010070,__READ_WRITE ,__tmr16ctcr_bits);
__IO_REG32_BIT(TMR16B1PWMC,           0x40010074,__READ_WRITE ,__tmr16pwmc_bits);

/***************************************************************************
 **
 ** CT32B0
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR32B0IR,             0x40014000,__READ_WRITE ,__tmr32ir_bits);
__IO_REG32_BIT(TMR32B0TCR,            0x40014004,__READ_WRITE ,__tmr32tcr_bits);
__IO_REG32(    TMR32B0TC,             0x40014008,__READ_WRITE);
__IO_REG32(    TMR32B0PR,             0x4001400C,__READ_WRITE);
__IO_REG32(    TMR32B0PC,             0x40014010,__READ_WRITE);
__IO_REG32_BIT(TMR32B0MCR,            0x40014014,__READ_WRITE ,__tmr32mcr_bits);
__IO_REG32(    TMR32B0MR0,            0x40014018,__READ_WRITE);
__IO_REG32(    TMR32B0MR1,            0x4001401C,__READ_WRITE);
__IO_REG32(    TMR32B0MR2,            0x40014020,__READ_WRITE);
__IO_REG32(    TMR32B0MR3,            0x40014024,__READ_WRITE);
__IO_REG32_BIT(TMR32B0CCR,            0x40014028,__READ_WRITE ,__tmr32tccr_bits);
__IO_REG32(    TMR32B0CR0,            0x4001402C,__READ);      
__IO_REG32_BIT(TMR32B0EMR,            0x4001403C,__READ_WRITE ,__tmr32emr_bits);
__IO_REG32_BIT(TMR32B0CTCR,           0x40014070,__READ_WRITE ,__tmr32ctcr_bits);
__IO_REG32_BIT(TMR32B0PWMC,           0x40014074,__READ_WRITE ,__tmr32pwmc_bits);

/***************************************************************************
 **
 ** CT32B1
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR32B1IR,             0x40018000,__READ_WRITE ,__tmr32ir_bits);
__IO_REG32_BIT(TMR32B1TCR,            0x40018004,__READ_WRITE ,__tmr32tcr_bits);
__IO_REG32(    TMR32B1TC,             0x40018008,__READ_WRITE);
__IO_REG32(    TMR32B1PR,             0x4001800C,__READ_WRITE);
__IO_REG32(    TMR32B1PC,             0x40018010,__READ_WRITE);
__IO_REG32_BIT(TMR32B1MCR,            0x40018014,__READ_WRITE ,__tmr32mcr_bits);
__IO_REG32(    TMR32B1MR0,            0x40018018,__READ_WRITE);
__IO_REG32(    TMR32B1MR1,            0x4001801C,__READ_WRITE);
__IO_REG32(    TMR32B1MR2,            0x40018020,__READ_WRITE);
__IO_REG32(    TMR32B1MR3,            0x40018024,__READ_WRITE);
__IO_REG32_BIT(TMR32B1CCR,            0x40018028,__READ_WRITE ,__tmr32tccr_bits);
__IO_REG32(    TMR32B1CR0,            0x4001802C,__READ);      
__IO_REG32_BIT(TMR32B1EMR,            0x4001803C,__READ_WRITE ,__tmr32emr_bits);
__IO_REG32_BIT(TMR32B1CTCR,           0x40018070,__READ_WRITE ,__tmr32ctcr_bits);
__IO_REG32_BIT(TMR32B1PWMC,           0x40018074,__READ_WRITE ,__tmr32pwmc_bits);

/***************************************************************************
 **
 ** A/D Converters
 **
 ***************************************************************************/
__IO_REG32_BIT(AD0CR,                 0x4001C000,__READ_WRITE ,__adcr_bits);
__IO_REG32_BIT(AD0GDR,                0x4001C004,__READ_WRITE ,__adgdr_bits);
__IO_REG32_BIT(AD0INTEN,              0x4001C00C,__READ_WRITE ,__adinten_bits);
__IO_REG32_BIT(AD0DR0,                0x4001C010,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR1,                0x4001C014,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR2,                0x4001C018,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR3,                0x4001C01C,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR4,                0x4001C020,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR5,                0x4001C024,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR6,                0x4001C028,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR7,                0x4001C02C,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0STAT,               0x4001C030,__READ       ,__adstat_bits);
__IO_REG32_BIT(AD0TRM,                0x4001C034,__READ_WRITE ,__adtrm_bits);
__IO_REG32_BIT(AD0INSEL,              0x4001C038,__READ_WRITE ,__adinsel_bits);

/***************************************************************************
 **
 ** Watchdog
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,                 0x40004000,__READ_WRITE ,__wdmod_bits);
__IO_REG32_BIT(WDTC,                  0x40004004,__READ_WRITE ,__wdtc_bits);
__IO_REG32_BIT(WDFEED,                0x40004008,__WRITE      ,__wdfeed_bits);
__IO_REG32_BIT(WDTV,                  0x4000400C,__READ       ,__wdtc_bits);
__IO_REG32_BIT(WDCLKSEL,              0x40004010,__READ_WRITE ,__wdclksel_bits);
__IO_REG32_BIT(WDWARNINT,             0x40004014,__READ_WRITE ,__wdwarnint_bits);
__IO_REG32_BIT(WDWINDOW,              0x40004018,__READ_WRITE ,__wdwindow_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASHCFG,              0x4003C010,__READ_WRITE ,__flashcfg_bits);
__IO_REG32_BIT(FMSSTART,              0x4003C020,__READ_WRITE ,__fmsstart_bits);
__IO_REG32_BIT(FMSSTOP,               0x4003C024,__READ_WRITE ,__fmsstop_bits);
__IO_REG32(    FMSW0,                 0x4003C02C,__READ       );
__IO_REG32(    FMSW1,                 0x4003C030,__READ       );
__IO_REG32(    FMSW2,                 0x4003C034,__READ       );
__IO_REG32(    FMSW3,                 0x4003C038,__READ       );
__IO_REG32_BIT(FMSTAT,                0x4003CFE0,__READ       ,__fmstat_bits);
__IO_REG32(    FMSTATCLR,             0x4003CFE8,__WRITE      );

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  NVIC Interrupt channels
 **
 ***************************************************************************/
#define MAIN_STACK             0  /* Main Stack                                             */
#define RESETI                 1  /* Reset                                                  */
#define NMII                   2  /* Non-maskable Interrupt                                 */
#define HFI                    3  /* Hard Fault                                             */
#define MMI                    4  /* Memory Management                                      */
#define BFI                    5  /* Bus Fault                                              */
#define UFI                    6  /* Usage Fault                                            */
#define SVCI                  11  /* SVCall                                                 */
#define DMI                   12  /* Debug Monitor                                          */
#define PSI                   14  /* PendSV                                                 */
#define STI                   15  /* SysTick                                                */
#define NVIC_PIN_INT0         16  /* GPIO pin interrupt 0                                   */
#define NVIC_PIN_INT1         17  /* GPIO pin interrupt 1                                   */
#define NVIC_PIN_INT2         18  /* GPIO pin interrupt 2                                   */
#define NVIC_PIN_INT3         19  /* GPIO pin interrupt 3                                   */
#define NVIC_PIN_INT4         20  /* GPIO pin interrupt 4                                   */
#define NVIC_PIN_INT5         21  /* GPIO pin interrupt 5                                   */
#define NVIC_PIN_INT6         22  /* GPIO pin interrupt 6                                   */
#define NVIC_PIN_INT7         23  /* GPIO pin interrupt 7                                   */
#define NVIC_GINT0            24  /* GPIO GROUP0 interrupt                                  */
#define NVIC_GINT1            25  /* GPIO GROUP1 interrupt                                  */
#define NVIC_SSP1             30  /* SSP1 interrupt                                         */
#define NVIC_I2C              31  /* I2C interrupt                                          */
#define NVIC_CT16B0           32  /* CT16B0 Match 0-3, Capture 0                            */
#define NVIC_CT16B1           33  /* CT16B1 Match 0-3, Capture 0                            */
#define NVIC_CT32B0           34  /* CT32B0 Match 0-3, Capture 0                            */
#define NVIC_CT32B1           35  /* CT32B1 Match 0-3, Capture 0                            */
#define NVIC_SSP0             36  /* SSP0 interrupt                                         */
#define NVIC_USART            37  /* USART interrupt                                        */
#define NVIC_ADC              40  /* ADC interrupt                                          */
#define NVIC_WWDT             41  /* WWDT interrupt                                         */
#define NVIC_BOD              42  /* BOD interrupt                                          */

#endif    /* __IOLPC1315_H */

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
Interrupt9   = PIN_INT0       0x40
Interrupt10  = PIN_INT1       0x44
Interrupt11  = PIN_INT2       0x48
Interrupt12  = PIN_INT3       0x4C
Interrupt13  = PIN_INT4       0x50
Interrupt14  = PIN_INT5       0x54
Interrupt15  = PIN_INT6       0x58
Interrupt16  = PIN_INT7       0x5C
Interrupt17  = GINT0          0x60
Interrupt18  = GINT1          0x64
Interrupt19  = SSP1           0x78
Interrupt20  = I2C            0x7C
Interrupt21  = CT16B0         0x80
Interrupt22  = CT16B1         0x84
Interrupt23  = CT32B0         0x88
Interrupt24  = CT32B1         0x8C
Interrupt25  = SSP0           0x90
Interrupt26  = USART          0x94
Interrupt27  = ADC            0xA0
Interrupt28  = WWDT           0xA4
Interrupt29  = BOD            0xA8
###DDF-INTERRUPT-END###*/
