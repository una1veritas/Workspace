/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC1113
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2009
 **
 **    $Revision: 49779 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC1113_H
#define __IOLPC1113_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC1113 SPECIAL FUNCTION REGISTERS
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
  __REG32  CONSTANT       : 4;
  __REG32  VARIANT        : 4;
  __REG32  IMPLEMENTER    : 8;
} __cpuidbr_bits;

/* Interrupt Control State Register */
typedef struct {
  __REG32  VECTACTIVE     : 6;
  __REG32                 : 6;
  __REG32  VECTPENDING    : 6;
  __REG32                 : 4;
  __REG32  ISRPENDING     : 1;
  __REG32                 : 2;
  __REG32  PENDSTCLR      : 1;
  __REG32  PENDSTSET      : 1;
  __REG32  PENDSVCLR      : 1;
  __REG32  PENDSVSET      : 1;
  __REG32                 : 2;
  __REG32  NMIPENDSET     : 1;
} __icsr_bits;

/* Application Interrupt and Reset Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32  VECTCLRACTIVE  : 1;
  __REG32  SYSRESETREQ    : 1;
  __REG32                 :12;
  __REG32  ENDIANESS      : 1;
  __REG32  VECTKEY        :16;
} __aircr_bits;

/* System Control Register  */
typedef struct {
  __REG32                 : 1;
  __REG32  SLEEPONEXIT    : 1;
  __REG32  SLEEPDEEP      : 1;
  __REG32                 : 1;
  __REG32  SEVONPEND      : 1;
  __REG32                 :27;
} __scr_bits;

/* Configuration and Control Register */
typedef struct {
  __REG32                 : 3;
  __REG32  UNALIGN_TRP    : 1;
  __REG32                 : 4;
  __REG32  STKALIGN       : 1;
  __REG32                 :23;
} __ccr_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32                 :24;
  __REG32  PRI_11         : 8;
} __shpr2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32                 :16;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __shpr3_bits;

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
typedef struct {
  __REG32  MSEL           : 5;
  __REG32  PSEL           : 2;
  __REG32                 :25;
} __pllctrl_bits;

/* System PLL status register */
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

/* Internal resonant crystal control register */
typedef struct {
  __REG32  TRIM           : 8;
  __REG32                 :24;
} __ircctrl_bits;

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

/* System PLL clock source update enable register */
/* Main clock source update enable register */
/* WDT clock source update enable register */
/* CLKOUT clock source update enable register */
typedef struct {
  __REG32  ENA            : 1;
  __REG32                 :31;
} __clkuen_bits;

/* System AHB clock divider register */
/* SSP clock divider register */
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
  __REG32  UART           : 1;
  __REG32  ADC            : 1;
  __REG32                 : 1;
  __REG32  WDT            : 1;
  __REG32  IOCON          : 1;
  __REG32                 : 1;
  __REG32  SSP1           : 1;
  __REG32                 :13;
} __sysahbclkctrl_bits;

/* POR captured PIO status register 0 */
typedef struct {
  __REG32  CAPPIO0_0      : 1;
  __REG32  CAPPIO0_1      : 1;
  __REG32  CAPPIO0_2      : 1;
  __REG32  CAPPIO0_3      : 1;
  __REG32  CAPPIO0_4      : 1;
  __REG32  CAPPIO0_5      : 1;
  __REG32  CAPPIO0_6      : 1;
  __REG32  CAPPIO0_7      : 1;
  __REG32  CAPPIO0_8      : 1;
  __REG32  CAPPIO0_9      : 1;
  __REG32  CAPPIO0_10     : 1;
  __REG32  CAPPIO0_11     : 1;
  __REG32  CAPPIO1_0      : 1;
  __REG32  CAPPIO1_1      : 1;
  __REG32  CAPPIO1_2      : 1;
  __REG32  CAPPIO1_3      : 1;
  __REG32  CAPPIO1_4      : 1;
  __REG32  CAPPIO1_5      : 1;
  __REG32  CAPPIO1_6      : 1;
  __REG32  CAPPIO1_7      : 1;
  __REG32  CAPPIO1_8      : 1;
  __REG32  CAPPIO1_9      : 1;
  __REG32  CAPPIO1_10     : 1;
  __REG32  CAPPIO1_11     : 1;
  __REG32  CAPPIO2_0      : 1;
  __REG32  CAPPIO2_1      : 1;
  __REG32  CAPPIO2_2      : 1;
  __REG32  CAPPIO2_3      : 1;
  __REG32  CAPPIO2_4      : 1;
  __REG32  CAPPIO2_5      : 1;
  __REG32  CAPPIO2_6      : 1;
  __REG32  CAPPIO2_7      : 1;
} __pioporcap0_bits;

/* POR captured PIO status register 1 */
typedef struct {
  __REG32  CAPPIO2_8      : 1;
  __REG32  CAPPIO2_9      : 1;
  __REG32  CAPPIO2_10     : 1;
  __REG32  CAPPIO2_11     : 1;
  __REG32  CAPPIO3_0      : 1;
  __REG32  CAPPIO3_1      : 1;
  __REG32  CAPPIO3_2      : 1;
  __REG32  CAPPIO3_3      : 1;
  __REG32  CAPPIO3_4      : 1;
  __REG32  CAPPIO3_5      : 1;
  __REG32                 :22;
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

/* Start logic edge control register 0 */
typedef struct {
  __REG32  APRPIO0_0      : 1;
  __REG32  APRPIO0_1      : 1;
  __REG32  APRPIO0_2      : 1;
  __REG32  APRPIO0_3      : 1;
  __REG32  APRPIO0_4      : 1;
  __REG32  APRPIO0_5      : 1;
  __REG32  APRPIO0_6      : 1;
  __REG32  APRPIO0_7      : 1;
  __REG32  APRPIO0_8      : 1;
  __REG32  APRPIO0_9      : 1;
  __REG32  APRPIO0_10     : 1;
  __REG32  APRPIO0_11     : 1;
  __REG32  APRPIO1_0      : 1;
  __REG32                 :19;
} __start_apr_p0_bits;

/* Start logic signal enable register 0 */
typedef struct {
  __REG32  ERPIO0_0       : 1;
  __REG32  ERPIO0_1       : 1;
  __REG32  ERPIO0_2       : 1;
  __REG32  ERPIO0_3       : 1;
  __REG32  ERPIO0_4       : 1;
  __REG32  ERPIO0_5       : 1;
  __REG32  ERPIO0_6       : 1;
  __REG32  ERPIO0_7       : 1;
  __REG32  ERPIO0_8       : 1;
  __REG32  ERPIO0_9       : 1;
  __REG32  ERPIO0_10      : 1;
  __REG32  ERPIO0_11      : 1;
  __REG32  ERPIO1_0       : 1;
  __REG32                 :19;
} __start_er_p0_bits;

/* Start logic reset register 0 */
typedef struct {
  __REG32  RSRPIO0_0      : 1;
  __REG32  RSRPIO0_1      : 1;
  __REG32  RSRPIO0_2      : 1;
  __REG32  RSRPIO0_3      : 1;
  __REG32  RSRPIO0_4      : 1;
  __REG32  RSRPIO0_5      : 1;
  __REG32  RSRPIO0_6      : 1;
  __REG32  RSRPIO0_7      : 1;
  __REG32  RSRPIO0_8      : 1;
  __REG32  RSRPIO0_9      : 1;
  __REG32  RSRPIO0_10     : 1;
  __REG32  RSRPIO0_11     : 1;
  __REG32  RSRPIO1_0      : 1;
  __REG32                 :19;
} __start_rsr_p0_clr_bits;


/* Start logic status register 0 */
typedef struct {
  __REG32  SRPIO0_0       : 1;
  __REG32  SRPIO0_1       : 1;
  __REG32  SRPIO0_2       : 1;
  __REG32  SRPIO0_3       : 1;
  __REG32  SRPIO0_4       : 1;
  __REG32  SRPIO0_5       : 1;
  __REG32  SRPIO0_6       : 1;
  __REG32  SRPIO0_7       : 1;
  __REG32  SRPIO0_8       : 1;
  __REG32  SRPIO0_9       : 1;
  __REG32  SRPIO0_10      : 1;
  __REG32  SRPIO0_11      : 1;
  __REG32  SRPIO1_0       : 1;
  __REG32                 :19;
} __start_sr_p0_bits;

/* Deep-sleep configuration register */
typedef struct {
  __REG32  				        : 3;
  __REG32  BOD_PD         : 1;
  __REG32  								: 2;
  __REG32  WDTOSC_PD      : 1;
  __REG32                 :25;
} __pdsleepcfg_bits;

/* Power-down configuration register */
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

/* Power control register */
typedef struct {
  __REG32                 : 1;
  __REG32  DPDEN          : 1;
  __REG32                 : 6;
  __REG32  SLEEPFLAG      : 1;
  __REG32                 : 2;
  __REG32  DPDFLAG        : 1;
  __REG32                 :20;
} __pcon_bits;

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
  __REG32                 :26;
} __iocon_bits;

/* IOCON_PIO0_4 register */
/* IOCON_PIO0_5 register */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32                 : 5;
  __REG32  I2CMODE        : 2;
  __REG32                 :22;
} __iocon_pio0_4_bits;

/* IOCON_PIO1_10 register */
/* IOCON_R_PIO0_11 register */
/* IOCON_R_PIO1_0 register */
/* IOCON_R_PIO1_1 register */
/* IOCON_R_PIO1_2 register */
/* IOCON_SWDIO_PIO1_3 register */
/* IOCON_PIO1_4 register */
/* IOCON_PIO1_11 register */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32                 : 1;
  __REG32  ADMODE         : 1;
  __REG32                 :24;
} __iocon_pio1_10_bits;

/* IOCON SCK location register*/
typedef struct {
  __REG32  SCKLOC         : 2;
  __REG32                 :30;
} __iocon_sckloc_bits;

/* IOCON DSR location register*/
typedef struct {
  __REG32  DSRLOC         : 2;
  __REG32                 :30;
} __iocon_dsrloc_bits;

/* IOCON DCD location register*/
typedef struct {
  __REG32  DCDLOC         : 2;
  __REG32                 :30;
} __iocon_dcdloc_bits;

/* IOCON RI location register*/
typedef struct {
  __REG32  RILOC          : 2;
  __REG32                 :30;
} __iocon_riloc_bits;

typedef union{
  /*GPIO0DATA*/
  /*GPIO0DIR*/
  /*GPIO0IS*/
  /*GPIO0IBE*/
  /*GPIO0IEV*/
  /*GPIO0IE*/
  /*GPIO0RIS*/
  /*GPIO0MIS*/
  /*GPIO0IC*/
  struct {
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
    __REG32        :20;
  };

  struct
  {
    union
    {
      /*GPIO0DATA0*/
      /*GPIO0DIR0*/
      /*GPIO0IS0*/
      /*GPIO0IBE0*/
      /*GPIO0IEV0*/
      /*GPIO0IE0*/
      /*GPIO0RIS0*/
      /*GPIO0MIS0*/
      /*GPIO0IC0*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8  P0_4   : 1;
        __REG8  P0_5   : 1;
        __REG8  P0_6   : 1;
        __REG8  P0_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*GPIO0DATA1*/
      /*GPIO0DIR1*/
      /*GPIO0IS1*/
      /*GPIO0IBE1*/
      /*GPIO0IEV1*/
      /*GPIO0IE1*/
      /*GPIO0RIS1*/
      /*GPIO0MIS1*/
      /*GPIO0IC1*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8         : 4;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8 __byte2;
    __REG8 __byte3;
  };

  struct
  {
    union
    {
      /*GPIO0DATAL*/
      /*GPIO0DIRL*/
      /*GPIO0ISL*/
      /*GPIO0IBEL*/
      /*GPIO0IEVL*/
      /*GPIO0IEL*/
      /*GPIO0RISL*/
      /*GPIO0MISL*/
      /*GPIO0ICL*/
      struct{
        __REG16 P0_0   : 1;
        __REG16 P0_1   : 1;
        __REG16 P0_2   : 1;
        __REG16 P0_3   : 1;
        __REG16 P0_4   : 1;
        __REG16 P0_5   : 1;
        __REG16 P0_6   : 1;
        __REG16 P0_7   : 1;
        __REG16 P0_8   : 1;
        __REG16 P0_9   : 1;
        __REG16 P0_10  : 1;
        __REG16 P0_11  : 1;
        __REG16        : 4;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16 __shortu;
  };
} __gpio0_bits;

typedef union{
  /*GPIO1DATA*/
  /*GPIO1DIR*/
  /*GPIO1IS*/
  /*GPIO1IBE*/
  /*GPIO1IEV*/
  /*GPIO1IE*/
  /*GPIO1RIS*/
  /*GPIO1MIS*/
  /*GPIO1IC*/
  struct {
    __REG32 P1_0   : 1;
    __REG32 P1_1   : 1;
    __REG32 P1_2   : 1;
    __REG32 P1_3   : 1;
    __REG32 P1_4   : 1;
    __REG32 P1_5   : 1;
    __REG32 P1_6   : 1;
    __REG32 P1_7   : 1;
    __REG32 P1_8   : 1;
    __REG32 P1_9   : 1;
    __REG32 P1_10  : 1;
    __REG32 P1_11  : 1;
    __REG32        :20;
  };

  struct
  {
    union
    {
      /*GPIO1DATA0*/
      /*GPIO1DIR0*/
      /*GPIO1IS0*/
      /*GPIO1IBE0*/
      /*GPIO1IEV0*/
      /*GPIO1IE0*/
      /*GPIO1RIS0*/
      /*GPIO1MIS0*/
      /*GPIO1IC0*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8  P1_4   : 1;
        __REG8  P1_5   : 1;
        __REG8  P1_6   : 1;
        __REG8  P1_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*GPIO1DATA1*/
      /*GPIO1DIR1*/
      /*GPIO1IS1*/
      /*GPIO1IBE1*/
      /*GPIO1IEV1*/
      /*GPIO1IE1*/
      /*GPIO1RIS1*/
      /*GPIO1MIS1*/
      /*GPIO1IC1*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8         : 4;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8 __byte2;
    __REG8 __byte3;
  };

  struct
  {
    union
    {
      /*GPIO1DATAL*/
      /*GPIO1DIRL*/
      /*GPIO1ISL*/
      /*GPIO1IBEL*/
      /*GPIO1IEVL*/
      /*GPIO1IEL*/
      /*GPIO1RISL*/
      /*GPIO1MISL*/
      /*GPIO1ICL*/
      struct{
        __REG16 P1_0   : 1;
        __REG16 P1_1   : 1;
        __REG16 P1_2   : 1;
        __REG16 P1_3   : 1;
        __REG16 P1_4   : 1;
        __REG16 P1_5   : 1;
        __REG16 P1_6   : 1;
        __REG16 P1_7   : 1;
        __REG16 P1_8   : 1;
        __REG16 P1_9   : 1;
        __REG16 P1_10  : 1;
        __REG16 P1_11  : 1;
        __REG16        : 4;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16 __shortu;
  };
} __gpio1_bits;

typedef union{
  /*GPIO2DATA*/
  /*GPIO2DIR*/
  /*GPIO2IS*/
  /*GPIO2IBE*/
  /*GPIO2IEV*/
  /*GPIO2IE*/
  /*GPIO2RIS*/
  /*GPIO2MIS*/
  /*GPIO2IC*/
  struct {
    __REG32 P2_0   : 1;
    __REG32 P2_1   : 1;
    __REG32 P2_2   : 1;
    __REG32 P2_3   : 1;
    __REG32 P2_4   : 1;
    __REG32 P2_5   : 1;
    __REG32 P2_6   : 1;
    __REG32 P2_7   : 1;
    __REG32 P2_8   : 1;
    __REG32 P2_9   : 1;
    __REG32 P2_10  : 1;
    __REG32 P2_11  : 1;
    __REG32        :20;
  };

  struct
  {
    union
    {
      /*GPIO2DATA0*/
      /*GPIO2DIR0*/
      /*GPIO2IS0*/
      /*GPIO2IBE0*/
      /*GPIO2IEV0*/
      /*GPIO2IE0*/
      /*GPIO2RIS0*/
      /*GPIO2MIS0*/
      /*GPIO2IC0*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8  P2_4   : 1;
        __REG8  P2_5   : 1;
        __REG8  P2_6   : 1;
        __REG8  P2_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*GPIO2DATA1*/
      /*GPIO2DIR1*/
      /*GPIO2IS1*/
      /*GPIO2IBE1*/
      /*GPIO2IEV1*/
      /*GPIO2IE1*/
      /*GPIO2RIS1*/
      /*GPIO2MIS1*/
      /*GPIO2IC1*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8         : 4;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8 __byte2;
    __REG8 __byte3;
  };

  struct
  {
    union
    {
      /*GPIO2DATAL*/
      /*GPIO2DIRL*/
      /*GPIO2ISL*/
      /*GPIO2IBEL*/
      /*GPIO2IEVL*/
      /*GPIO2IEL*/
      /*GPIO2RISL*/
      /*GPIO2MISL*/
      /*GPIO2ICL*/
      struct{
        __REG16 P2_0   : 1;
        __REG16 P2_1   : 1;
        __REG16 P2_2   : 1;
        __REG16 P2_3   : 1;
        __REG16 P2_4   : 1;
        __REG16 P2_5   : 1;
        __REG16 P2_6   : 1;
        __REG16 P2_7   : 1;
        __REG16 P2_8   : 1;
        __REG16 P2_9   : 1;
        __REG16 P2_10  : 1;
        __REG16 P2_11  : 1;
        __REG16        : 4;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16 __shortu;
  };
} __gpio2_bits;

typedef union{
  /*GPIO3DATA*/
  /*GPIO3DIR*/
  /*GPIO3IS*/
  /*GPIO3IBE*/
  /*GPIO3IEV*/
  /*GPIO3IE*/
  /*GPIO3RIS*/
  /*GPIO3MIS*/
  /*GPIO3IC*/
  struct {
    __REG32 P3_0   : 1;
    __REG32 P3_1   : 1;
    __REG32 P3_2   : 1;
    __REG32 P3_3   : 1;
    __REG32 P3_4   : 1;
    __REG32 P3_5   : 1;
    __REG32        :26;
  };
  
  struct
  {
    union
    {
      /*GPIO3DATA0*/
      /*GPIO3DIR0*/
      /*GPIO3IS0*/
      /*GPIO3IBE0*/
      /*GPIO3IEV0*/
      /*GPIO3IE0*/
      /*GPIO3RIS0*/
      /*GPIO3MIS0*/
      /*GPIO3IC0*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8         : 2;
      } __byte0_bit;
      __REG8 __byte0;
    };

    __REG8 __byte1;
    __REG8 __byte2;
    __REG8 __byte3;
  };

  struct
  {
    union
    {
      /*GPIO3DATAL*/
      /*GPIO3DIRL*/
      /*GPIO3ISL*/
      /*GPIO3IBEL*/
      /*GPIO3IEVL*/
      /*GPIO3IEL*/
      /*GPIO3RISL*/
      /*GPIO3MISL*/
      /*GPIO3ICL*/
      struct{
        __REG16 P3_0   : 1;
        __REG16 P3_1   : 1;
        __REG16 P3_2   : 1;
        __REG16 P3_3   : 1;
        __REG16 P3_4   : 1;
        __REG16 P3_5   : 1;
        __REG16        :10;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16 __shortu;
  };
} __gpio3_bits;

/* UART interrupt enable register */
typedef struct{
__REG32 RDAIE     : 1;
__REG32 THREIE    : 1;
__REG32 RXLSIE    : 1;
__REG32           : 5;
__REG32 ABEOINTEN : 1;
__REG32 ABTOINTEN : 1;
__REG32           :22;
} __uartier_bits;

/* UART Transmit Enable Register */
typedef struct{
__REG8        : 7;
__REG8  TXEN  : 1;
} __uartter_bits;

/* UART line status register */
typedef struct{
__REG8  DR   : 1;
__REG8  OE    : 1;
__REG8  PE    : 1;
__REG8  FE    : 1;
__REG8  BI    : 1;
__REG8  THRE  : 1;
__REG8  TEMT  : 1;
__REG8  RXFE  : 1;
} __uartlsr_bits;

/* UART line control register */
typedef struct{
__REG8  WLS   : 2;
__REG8  SBS   : 1;
__REG8  PE    : 1;
__REG8  PS    : 2;
__REG8  BC    : 1;
__REG8  DLAB  : 1;
} __uartlcr_bits;

/* UART modem control register */
typedef struct{
__REG8  DTR   : 1;
__REG8  RTS   : 1;
__REG8        : 2;
__REG8  LMS   : 1;
__REG8        : 1;
__REG8  RTSEN : 1;
__REG8  CTSEN : 1;
} __uartmcr_bits;

/* UART modem status register */
typedef union{
  /*UxMSR*/
  struct {
__REG8  DCTS  : 1;
__REG8  DDSR  : 1;
__REG8  TERI  : 1;
__REG8  DDCD  : 1;
__REG8  CTS   : 1;
__REG8  DSR   : 1;
__REG8  RI    : 1;
__REG8  DCD   : 1;
  };
  /*UxMSR*/
  struct {
__REG8  MSR0  : 1;
__REG8  MSR1  : 1;
__REG8  MSR2  : 1;
__REG8  MSR3  : 1;
__REG8  MSR4  : 1;
__REG8  MSR5  : 1;
__REG8  MSR6  : 1;
__REG8  MSR7  : 1;
  };
} __uartmsr_bits;

/* UART interrupt identification register and fifo control register */
typedef union {
  /*UxIIR*/
  struct {
__REG32 IP     : 1;
__REG32 IID    : 3;
__REG32        : 2;
__REG32 IIRFE  : 2;
__REG32 ABEOINT: 1;
__REG32 ABTOINT: 1;
__REG32        :22;
  };
  /*UxFCR*/
  struct {
__REG32 FCRFE  : 1;
__REG32 RFR    : 1;
__REG32 TFR    : 1;
__REG32        : 3;
__REG32 RTLS   : 2;
__REG32        :24;
  };
} __uartfcriir_bits;

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

/* UART Fractional Divider Register */
typedef struct{
__REG32 DIVADDVAL  : 4;
__REG32 MULVAL     : 4;
__REG32            :24;
} __uartfdr_bits;

/* UART RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32 SEL        : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __u1rs485ctrl_bits;

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
/* Interrupt Register (TMR32B0IR and TMR32B1IR) */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32         :27;
} __ir_bits;

/* Timer Control Register (TMR16B0TCR and TMR16B1TCR) */
/* Timer Control Register (TMR32B0TCR and TMR32B1TCR) */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tcr_bits;

/* Count Control Register (TMR16B0CTCR and TMR16B1CTCR) */
/* Count Control Register (TMR32B0CTCR and TMR32B1CTCR) */
typedef struct{
__REG32 CTM : 2;     /*Counter/Timer Mode*/
__REG32 CIS : 2;     /*Count Input Select*/
__REG32     :28;
} __ctcr_bits;

/* Match Control Register (TMR16B0MCR and TMR16B1MCR) */
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
} __mcr_bits;

/* Capture Control Register (TMR16B0CCR and TMR16B1CCR) */
/* Capture Control Register (TMR32B0CCR and TMR32B1CCR) */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32          :29;
} __tccr_bits;

/* External Match Register (TMR16B0EMR and TMR16B1EMR) */
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
} __emr_bits;

/* PWM Control register (TMR16B0PWMC and TMR16B1PWMC) */
/* PWM Control register (TMR32B0PWMC and TMR32B1PWMC) */
typedef struct{
__REG32 PWM0ENA  : 1;
__REG32 PWM1ENA  : 1;
__REG32 PWM2ENA  : 1;
__REG32 PWM3ENA  : 1;
__REG32          :28;
} __pwmc_bits;

/* A/D Control Register */
typedef struct{
__REG32 SEL     : 8;
__REG32 CLKDIV  : 8;
__REG32 BURST   : 1;
__REG32 CLKS    : 3;
__REG32         : 4;
__REG32 START   : 3;
__REG32 EDGE    : 1;
__REG32         : 4;
} __adcr_bits;

/* A/D Global Data Register */
typedef struct{
__REG32         : 6;
__REG32 RESULT  :10;
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
__REG32         : 6;
__REG32 RESULT  :10;
__REG32         :14;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __addr_bits;

/* Watchdog mode register */
typedef struct{
__REG32 WDEN     : 1;
__REG32 WDRESET  : 1;
__REG32 WDTOF    : 1;
__REG32 WDINT    : 1;
__REG32          :28;
} __wdmod_bits;

/* Watchdog feed register */
typedef struct{
__REG32 FEED  : 8;
__REG32       :24;
} __wdfeed_bits;

/* Flash configuration register */
typedef struct{
__REG32 FLASHTIM  : 2;
__REG32           :30;
} __flashcfg_bits;

/* Flash Module Signature Start register */
typedef struct{
__REG32 START       :17;
__REG32             :15;
} __fmsstart_bits;

/* Flash Module Signature Stop register */
typedef struct{
__REG32 STOP        :17;
__REG32 SIG_START   : 1;
__REG32             :14;
} __fmsstop_bits;

/* Flash Module Status register */
typedef struct{
__REG32             : 2;
__REG32 SIG_DONE    : 1;
__REG32             :29;
} __fmstat_bits;

/* Flash Module Status Clear register */
typedef struct{
__REG32               : 2;
__REG32 SIG_DONE_CLR  : 1;
__REG32               :29;
} __fmstatclr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,          0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,               0xE000E100,__READ_WRITE ,__setena0_bits);
#define ISER        SETENA0
#define ISER_bit    SETENA0_bit
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
#define ICER        CLRENA0
#define ICER_bit    CLRENA0_bit
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
#define ISPR        SETPEND0
#define ISPR_bit    SETPEND0_bit
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
#define ICPR        CLRPEND0
#define ICPR_bit    CLRPEND0_bit
__IO_REG32_BIT(IP0,                   0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                   0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                   0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                   0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                   0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                   0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                   0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                   0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(CPUIDBR,               0xE000ED00,__READ       ,__cpuidbr_bits);
#define CPUID         CPUIDBR
#define CPUID_bit     CPUIDBR_bit
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED1C,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHPR3,                 0xE000ED20,__READ_WRITE ,__shpr3_bits);

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
__IO_REG32_BIT(IRCCTRL,               0x40048028,__READ_WRITE ,__ircctrl_bits);
__IO_REG32_BIT(SYSRESSTAT,            0x40048030,__READ       ,__sysresstat_bits);
__IO_REG32_BIT(SYSPLLCLKSEL,          0x40048040,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(SYSPLLCLKUEN,          0x40048044,__READ_WRITE ,__clkuen_bits);
__IO_REG32_BIT(MAINCLKSEL,            0x40048070,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(MAINCLKUEN,            0x40048074,__READ_WRITE ,__clkuen_bits);
__IO_REG32_BIT(SYSAHBCLKDIV,          0x40048078,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(SYSAHBCLKCTRL,         0x40048080,__READ_WRITE ,__sysahbclkctrl_bits);
__IO_REG32_BIT(SSP0CLKDIV,            0x40048094,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(UARTCLKDIV,            0x40048098,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(SSP1CLKDIV,            0x4004809C,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(WDTCLKSEL,             0x400480D0,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(WDTCLKUEN,             0x400480D4,__READ_WRITE ,__clkuen_bits);
__IO_REG32_BIT(WDTCLKDIV,             0x400480D8,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(CLKOUTCLKSEL,          0x400480E0,__READ_WRITE ,__clksel_bits);
__IO_REG32_BIT(CLKOUTUEN,             0x400480E4,__READ_WRITE ,__clkuen_bits);
__IO_REG32_BIT(CLKOUTDIV,             0x400480E8,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(PIOPORCAP0,            0x40048100,__READ       ,__pioporcap0_bits);
__IO_REG32_BIT(PIOPORCAP1,            0x40048104,__READ       ,__pioporcap1_bits);
__IO_REG32_BIT(BODCTRL,               0x40048150,__READ_WRITE ,__bodctrl_bits);
__IO_REG32_BIT(SYSTCKCAL,             0x40048154,__READ_WRITE ,__systckcal_bits);
__IO_REG32_BIT(STARTAPRP0,            0x40048200,__READ_WRITE ,__start_apr_p0_bits);
__IO_REG32_BIT(STARTERP0,             0x40048204,__READ_WRITE ,__start_er_p0_bits);
__IO_REG32_BIT(STARTRSRP0CLR,         0x40048208,__WRITE      ,__start_rsr_p0_clr_bits);
__IO_REG32_BIT(STARTSRP0,             0x4004820C,__READ       ,__start_sr_p0_bits);
__IO_REG32_BIT(PDSLEEPCFG,            0x40048230,__READ_WRITE ,__pdsleepcfg_bits);
__IO_REG32_BIT(PDAWAKECFG,            0x40048234,__READ_WRITE ,__pdawakecfg_bits);
__IO_REG32_BIT(PDRUNCFG,              0x40048238,__READ_WRITE ,__pdawakecfg_bits);
__IO_REG32(    DEVICE_ID,             0x400483F4,__READ       );

/***************************************************************************
 **
 ** PMU
 **
 ***************************************************************************/
__IO_REG32_BIT(PCON,                  0x40038000,__READ_WRITE ,__pcon_bits);
__IO_REG32(		 GPREG0,                0x40038004,__READ_WRITE );
__IO_REG32(		 GPREG1,                0x40038008,__READ_WRITE );
__IO_REG32(		 GPREG2,                0x4003800C,__READ_WRITE );
__IO_REG32(		 GPREG3,                0x40038010,__READ_WRITE );
__IO_REG32_BIT(GPREG4,                0x40038014,__READ_WRITE ,__gpreg4_bits);

/***************************************************************************
 **
 ** I/O configuration
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCON_PIO2_6,          0x40044000,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_0,          0x40044008,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_RESET_PIO0_0,    0x4004400C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_1,          0x40044010,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_8,          0x40044014,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_2,          0x4004401C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_7,          0x40044020,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_8,          0x40044024,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_1,          0x40044028,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_3,          0x4004402C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_4,          0x40044030,__READ_WRITE ,__iocon_pio0_4_bits);
__IO_REG32_BIT(IOCON_PIO0_5,          0x40044034,__READ_WRITE ,__iocon_pio0_4_bits);
__IO_REG32_BIT(IOCON_PIO1_9,          0x40044038,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO3_4,          0x4004403C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_4,          0x40044040,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_5,          0x40044044,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO3_5,          0x40044048,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_6,          0x4004404C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_7,          0x40044050,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_9,          0x40044054,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_10,         0x40044058,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_2,          0x4004405C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_8,          0x40044060,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO0_9,          0x40044064,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_SWCLK_PIO0_10,   0x40044068,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_10,         0x4004406C,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_PIO2_11,         0x40044070,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_R_PIO0_11,       0x40044074,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_R_PIO1_0,        0x40044078,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_R_PIO1_1,        0x4004407C,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_R_PIO1_2,        0x40044080,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_PIO3_0,          0x40044084,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO3_1,          0x40044088,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO2_3,          0x4004408C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_SWDIO_PIO1_3,    0x40044090,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_PIO1_4,          0x40044094,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_PIO1_11,         0x40044098,__READ_WRITE ,__iocon_pio1_10_bits);
__IO_REG32_BIT(IOCON_PIO3_2,          0x4004409C,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_5,          0x400440A0,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_6,          0x400440A4,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO1_7,          0x400440A8,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_PIO3_3,          0x400440AC,__READ_WRITE ,__iocon_bits);
__IO_REG32_BIT(IOCON_SCK_LOC,         0x400440B0,__READ_WRITE ,__iocon_sckloc_bits);
__IO_REG32_BIT(IOCON_DSR_LOC,         0x400440B4,__READ_WRITE ,__iocon_dsrloc_bits);
__IO_REG32_BIT(IOCON_DCD_LOC,         0x400440B8,__READ_WRITE ,__iocon_dcdloc_bits);
__IO_REG32_BIT(IOCON_RI_LOC,          0x400440BC,__READ_WRITE ,__iocon_riloc_bits);

/***************************************************************************
 **
 ** CPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO0DATA,             0x50003FFC,__READ_WRITE ,__gpio0_bits);
#define GPIO0DATA0          GPIO0DATA_bit.__byte0
#define GPIO0DATA0_bit      GPIO0DATA_bit.__byte0_bit
#define GPIO0DATA1          GPIO0DATA_bit.__byte1
#define GPIO0DATA1_bit      GPIO0DATA_bit.__byte1_bit
#define GPIO0DATAL          GPIO0DATA_bit.__shortl
#define GPIO0DATAL_bit      GPIO0DATA_bit.__shortl_bit
__IO_REG32_BIT(GPIO0DIR,              0x50008000,__READ_WRITE ,__gpio0_bits);
#define GPIO0DIR0           GPIO0DIR_bit.__byte0
#define GPIO0DIR0_bit       GPIO0DIR_bit.__byte0_bit
#define GPIO0DIR1           GPIO0DIR_bit.__byte1
#define GPIO0DIR1_bit       GPIO0DIR_bit.__byte1_bit
#define GPIO0DIRL           GPIO0DIR_bit.__shortl
#define GPIO0DIRL_bit       GPIO0DIR_bit.__shortl_bit
__IO_REG32_BIT(GPIO0IS,               0x50008004,__READ_WRITE ,__gpio0_bits);
#define GPIO0IS0            GPIO0IS_bit.__byte0
#define GPIO0IS0_bit        GPIO0IS_bit.__byte0_bit
#define GPIO0IS1            GPIO0IS_bit.__byte1
#define GPIO0IS1_bit        GPIO0IS_bit.__byte1_bit
#define GPIO0ISL            GPIO0IS_bit.__shortl
#define GPIO0ISL_bit        GPIO0IS_bit.__shortl_bit
__IO_REG32_BIT(GPIO0IBE,              0x50008008,__READ_WRITE ,__gpio0_bits);
#define GPIO0IBE0           GPIO0IBE_bit.__byte0
#define GPIO0IBE0_bit       GPIO0IBE_bit.__byte0_bit
#define GPIO0IBE1           GPIO0IBE_bit.__byte1
#define GPIO0IBE1_bit       GPIO0IBE_bit.__byte1_bit
#define GPIO0IBEL           GPIO0IBE_bit.__shortl
#define GPIO0IBEL_bit       GPIO0IBE_bit.__shortl_bit
__IO_REG32_BIT(GPIO0IEV,              0x5000800C,__READ_WRITE ,__gpio0_bits);
#define GPIO0IEV0           GPIO0IEV_bit.__byte0
#define GPIO0IEV0_bit       GPIO0IEV_bit.__byte0_bit
#define GPIO0IEV1           GPIO0IEV_bit.__byte1
#define GPIO0IEV1_bit       GPIO0IEV_bit.__byte1_bit
#define GPIO0IEVL           GPIO0IEV_bit.__shortl
#define GPIO0IEVL_bit       GPIO0IEV_bit.__shortl_bit
__IO_REG32_BIT(GPIO0IE,               0x50008010,__READ_WRITE ,__gpio0_bits);
#define GPIO0IE0            GPIO0IE_bit.__byte0
#define GPIO0IE0_bit        GPIO0IE_bit.__byte0_bit
#define GPIO0IE1            GPIO0IE_bit.__byte1
#define GPIO0IE1_bit        GPIO0IE_bit.__byte1_bit
#define GPIO0IEL            GPIO0IE_bit.__shortl
#define GPIO0IEL_bit        GPIO0IE_bit.__shortl_bit
__IO_REG32_BIT(GPIO0RIS,              0x50008014,__READ       ,__gpio0_bits);
#define GPIO0RIS0           GPIO0RIS_bit.__byte0
#define GPIO0RIS0_bit       GPIO0RIS_bit.__byte0_bit
#define GPIO0RIS1           GPIO0RIS_bit.__byte1
#define GPIO0RIS1_bit       GPIO0RIS_bit.__byte1_bit
#define GPIO0RISL           GPIO0RIS_bit.__shortl
#define GPIO0RISL_bit       GPIO0RIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO0MIS,              0x50008018,__READ       ,__gpio0_bits);
#define GPIO0MIS0           GPIO0MIS_bit.__byte0
#define GPIO0MIS0_bit       GPIO0MIS_bit.__byte0_bit
#define GPIO0MIS1           GPIO0MIS_bit.__byte1
#define GPIO0MIS1_bit       GPIO0MIS_bit.__byte1_bit
#define GPIO0MISL           GPIO0MIS_bit.__shortl
#define GPIO0MISL_bit       GPIO0MIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO0IC,               0x5000801C,__WRITE      ,__gpio0_bits);
#define GPIO0IC0            GPIO0IC_bit.__byte0
#define GPIO0IC0_bit        GPIO0IC_bit.__byte0_bit
#define GPIO0IC1            GPIO0IC_bit.__byte1
#define GPIO0IC1_bit        GPIO0IC_bit.__byte1_bit
#define GPIO0ICL            GPIO0IC_bit.__shortl
#define GPIO0ICL_bit        GPIO0IC_bit.__shortl_bit

__IO_REG32_BIT(GPIO1DATA,             0x50013FFC,__READ_WRITE ,__gpio1_bits);
#define GPIO1DATA0          GPIO1DATA_bit.__byte0
#define GPIO1DATA0_bit      GPIO1DATA_bit.__byte0_bit
#define GPIO1DATA1          GPIO1DATA_bit.__byte1
#define GPIO1DATA1_bit      GPIO1DATA_bit.__byte1_bit
#define GPIO1DATAL          GPIO1DATA_bit.__shortl
#define GPIO1DATAL_bit      GPIO1DATA_bit.__shortl_bit
__IO_REG32_BIT(GPIO1DIR,              0x50018000,__READ_WRITE ,__gpio1_bits);
#define GPIO1DIR0           GPIO1DIR_bit.__byte0
#define GPIO1DIR0_bit       GPIO1DIR_bit.__byte0_bit
#define GPIO1DIR1           GPIO1DIR_bit.__byte1
#define GPIO1DIR1_bit       GPIO1DIR_bit.__byte1_bit
#define GPIO1DIRL           GPIO1DIR_bit.__shortl
#define GPIO1DIRL_bit       GPIO1DIR_bit.__shortl_bit
__IO_REG32_BIT(GPIO1IS,               0x50018004,__READ_WRITE ,__gpio1_bits);
#define GPIO1IS0            GPIO1IS_bit.__byte0
#define GPIO1IS0_bit        GPIO1IS_bit.__byte0_bit
#define GPIO1IS1            GPIO1IS_bit.__byte1
#define GPIO1IS1_bit        GPIO1IS_bit.__byte1_bit
#define GPIO1ISL            GPIO1IS_bit.__shortl
#define GPIO1ISL_bit        GPIO1IS_bit.__shortl_bit
__IO_REG32_BIT(GPIO1IBE,              0x50018008,__READ_WRITE ,__gpio1_bits);
#define GPIO1IBE0           GPIO1IBE_bit.__byte0
#define GPIO1IBE0_bit       GPIO1IBE_bit.__byte0_bit
#define GPIO1IBE1           GPIO1IBE_bit.__byte1
#define GPIO1IBE1_bit       GPIO1IBE_bit.__byte1_bit
#define GPIO1IBEL           GPIO1IBE_bit.__shortl
#define GPIO1IBEL_bit       GPIO1IBE_bit.__shortl_bit
__IO_REG32_BIT(GPIO1IEV,              0x5001800C,__READ_WRITE ,__gpio1_bits);
#define GPIO1IEV0           GPIO1IEV_bit.__byte0
#define GPIO1IEV0_bit       GPIO1IEV_bit.__byte0_bit
#define GPIO1IEV1           GPIO1IEV_bit.__byte1
#define GPIO1IEV1_bit       GPIO1IEV_bit.__byte1_bit
#define GPIO1IEVL           GPIO1IEV_bit.__shortl
#define GPIO1IEVL_bit       GPIO1IEV_bit.__shortl_bit
__IO_REG32_BIT(GPIO1IE,               0x50018010,__READ_WRITE ,__gpio1_bits);
#define GPIO1IE0            GPIO1IE_bit.__byte0
#define GPIO1IE0_bit        GPIO1IE_bit.__byte0_bit
#define GPIO1IE1            GPIO1IE_bit.__byte1
#define GPIO1IE1_bit        GPIO1IE_bit.__byte1_bit
#define GPIO1IEL            GPIO1IE_bit.__shortl
#define GPIO1IEL_bit        GPIO1IE_bit.__shortl_bit
__IO_REG32_BIT(GPIO1RIS,              0x50018014,__READ       ,__gpio1_bits);
#define GPIO1RIS0           GPIO1RIS_bit.__byte0
#define GPIO1RIS0_bit       GPIO1RIS_bit.__byte0_bit
#define GPIO1RIS1           GPIO1RIS_bit.__byte1
#define GPIO1RIS1_bit       GPIO1RIS_bit.__byte1_bit
#define GPIO1RISL           GPIO1RIS_bit.__shortl
#define GPIO1RISL_bit       GPIO1RIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO1MIS,              0x50018018,__READ       ,__gpio1_bits);
#define GPIO1MIS0           GPIO1MIS_bit.__byte0
#define GPIO1MIS0_bit       GPIO1MIS_bit.__byte0_bit
#define GPIO1MIS1           GPIO1MIS_bit.__byte1
#define GPIO1MIS1_bit       GPIO1MIS_bit.__byte1_bit
#define GPIO1MISL           GPIO1MIS_bit.__shortl
#define GPIO1MISL_bit       GPIO1MIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO1IC,               0x5001801C,__WRITE      ,__gpio1_bits);
#define GPIO1IC0            GPIO1IC_bit.__byte0
#define GPIO1IC0_bit        GPIO1IC_bit.__byte0_bit
#define GPIO1IC1            GPIO1IC_bit.__byte1
#define GPIO1IC1_bit        GPIO1IC_bit.__byte1_bit
#define GPIO1ICL            GPIO1IC_bit.__shortl
#define GPIO1ICL_bit        GPIO1IC_bit.__shortl_bit
 
__IO_REG32_BIT(GPIO2DATA,             0x50023FFC,__READ_WRITE ,__gpio2_bits);
#define GPIO2DATA0          GPIO2DATA_bit.__byte0
#define GPIO2DATA0_bit      GPIO2DATA_bit.__byte0_bit
#define GPIO2DATA1          GPIO2DATA_bit.__byte1
#define GPIO2DATA1_bit      GPIO2DATA_bit.__byte1_bit
#define GPIO2DATAL          GPIO2DATA_bit.__shortl
#define GPIO2DATAL_bit      GPIO2DATA_bit.__shortl_bit
__IO_REG32_BIT(GPIO2DIR,              0x50028000,__READ_WRITE ,__gpio2_bits);
#define GPIO2DIR0           GPIO2DIR_bit.__byte0
#define GPIO2DIR0_bit       GPIO2DIR_bit.__byte0_bit
#define GPIO2DIR1           GPIO2DIR_bit.__byte1
#define GPIO2DIR1_bit       GPIO2DIR_bit.__byte1_bit
#define GPIO2DIRL           GPIO2DIR_bit.__shortl
#define GPIO2DIRL_bit       GPIO2DIR_bit.__shortl_bit
__IO_REG32_BIT(GPIO2IS,               0x50028004,__READ_WRITE ,__gpio2_bits);
#define GPIO2IS0            GPIO2IS_bit.__byte0
#define GPIO2IS0_bit        GPIO2IS_bit.__byte0_bit
#define GPIO2IS1            GPIO2IS_bit.__byte1
#define GPIO2IS1_bit        GPIO2IS_bit.__byte1_bit
#define GPIO2ISL            GPIO2IS_bit.__shortl
#define GPIO2ISL_bit        GPIO2IS_bit.__shortl_bit
__IO_REG32_BIT(GPIO2IBE,              0x50028008,__READ_WRITE ,__gpio2_bits);
#define GPIO2IBE0           GPIO2IBE_bit.__byte0
#define GPIO2IBE0_bit       GPIO2IBE_bit.__byte0_bit
#define GPIO2IBE1           GPIO2IBE_bit.__byte1
#define GPIO2IBE1_bit       GPIO2IBE_bit.__byte1_bit
#define GPIO2IBEL           GPIO2IBE_bit.__shortl
#define GPIO2IBEL_bit       GPIO2IBE_bit.__shortl_bit
__IO_REG32_BIT(GPIO2IEV,              0x5002800C,__READ_WRITE ,__gpio2_bits);
#define GPIO2IEV0           GPIO2IEV_bit.__byte0
#define GPIO2IEV0_bit       GPIO2IEV_bit.__byte0_bit
#define GPIO2IEV1           GPIO2IEV_bit.__byte1
#define GPIO2IEV1_bit       GPIO2IEV_bit.__byte1_bit
#define GPIO2IEVL           GPIO2IEV_bit.__shortl
#define GPIO2IEVL_bit       GPIO2IEV_bit.__shortl_bit
__IO_REG32_BIT(GPIO2IE,               0x50028010,__READ_WRITE ,__gpio2_bits);
#define GPIO2IE0            GPIO2IE_bit.__byte0
#define GPIO2IE0_bit        GPIO2IE_bit.__byte0_bit
#define GPIO2IE1            GPIO2IE_bit.__byte1
#define GPIO2IE1_bit        GPIO2IE_bit.__byte1_bit
#define GPIO2IEL            GPIO2IE_bit.__shortl
#define GPIO2IEL_bit        GPIO2IE_bit.__shortl_bit
__IO_REG32_BIT(GPIO2RIS,              0x50028014,__READ       ,__gpio2_bits);
#define GPIO2RIS0           GPIO2RIS_bit.__byte0
#define GPIO2RIS0_bit       GPIO2RIS_bit.__byte0_bit
#define GPIO2RIS1           GPIO2RIS_bit.__byte1
#define GPIO2RIS1_bit       GPIO2RIS_bit.__byte1_bit
#define GPIO2RISL           GPIO2RIS_bit.__shortl
#define GPIO2RISL_bit       GPIO2RIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO2MIS,              0x50028018,__READ       ,__gpio2_bits);
#define GPIO2MIS0           GPIO2MIS_bit.__byte0
#define GPIO2MIS0_bit       GPIO2MIS_bit.__byte0_bit
#define GPIO2MIS1           GPIO2MIS_bit.__byte1
#define GPIO2MIS1_bit       GPIO2MIS_bit.__byte1_bit
#define GPIO2MISL           GPIO2MIS_bit.__shortl
#define GPIO2MISL_bit       GPIO2MIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO2IC,               0x5002801C,__WRITE      ,__gpio2_bits);
#define GPIO2IC0            GPIO2IC_bit.__byte0
#define GPIO2IC0_bit        GPIO2IC_bit.__byte0_bit
#define GPIO2IC1            GPIO2IC_bit.__byte1
#define GPIO2IC1_bit        GPIO2IC_bit.__byte1_bit
#define GPIO2ICL            GPIO2IC_bit.__shortl
#define GPIO2ICL_bit        GPIO2IC_bit.__shortl_bit

__IO_REG32_BIT(GPIO3DATA,             0x50033FFC,__READ_WRITE ,__gpio3_bits);
#define GPIO3DATA0          GPIO3DATA_bit.__byte0
#define GPIO3DATA0_bit      GPIO3DATA_bit.__byte0_bit
#define GPIO3DATAL          GPIO3DATA_bit.__shortl
#define GPIO3DATAL_bit      GPIO3DATA_bit.__shortl_bit
__IO_REG32_BIT(GPIO3DIR,              0x50038000,__READ_WRITE ,__gpio3_bits);
#define GPIO3DIR0           GPIO3DIR_bit.__byte0
#define GPIO3DIR0_bit       GPIO3DIR_bit.__byte0_bit
#define GPIO3DIRL           GPIO3DIR_bit.__shortl
#define GPIO3DIRL_bit       GPIO3DIR_bit.__shortl_bit
__IO_REG32_BIT(GPIO3IS,               0x50038004,__READ_WRITE ,__gpio3_bits);
#define GPIO3IS0            GPIO3IS_bit.__byte0
#define GPIO3IS0_bit        GPIO3IS_bit.__byte0_bit
#define GPIO3ISL            GPIO3IS_bit.__shortl
#define GPIO3ISL_bit        GPIO3IS_bit.__shortl_bit
__IO_REG32_BIT(GPIO3IBE,              0x50038008,__READ_WRITE ,__gpio3_bits);
#define GPIO3IBE0           GPIO3IBE_bit.__byte0
#define GPIO3IBE0_bit       GPIO3IBE_bit.__byte0_bit
#define GPIO3IBEL           GPIO3IBE_bit.__shortl
#define GPIO3IBEL_bit       GPIO3IBE_bit.__shortl_bit
__IO_REG32_BIT(GPIO3IEV,              0x5003800C,__READ_WRITE ,__gpio3_bits);
#define GPIO3IEV0           GPIO3IEV_bit.__byte0
#define GPIO3IEV0_bit       GPIO3IEV_bit.__byte0_bit
#define GPIO3IEVL           GPIO3IEV_bit.__shortl
#define GPIO3IEVL_bit       GPIO3IEV_bit.__shortl_bit
__IO_REG32_BIT(GPIO3IE,               0x50038010,__READ_WRITE ,__gpio3_bits);
#define GPIO3IE0            GPIO3IE_bit.__byte0
#define GPIO3IE0_bit        GPIO3IE_bit.__byte0_bit
#define GPIO3IEL            GPIO3IE_bit.__shortl
#define GPIO3IEL_bit        GPIO3IE_bit.__shortl_bit
__IO_REG32_BIT(GPIO3RIS,              0x50038014,__READ       ,__gpio3_bits);
#define GPIO3RIS0           GPIO3RIS_bit.__byte0
#define GPIO3RIS0_bit       GPIO3RIS_bit.__byte0_bit
#define GPIO3RISL           GPIO3RIS_bit.__shortl
#define GPIO3RISL_bit       GPIO3RIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO3MIS,              0x50038018,__READ       ,__gpio3_bits);
#define GPIO3MIS0           GPIO3MIS_bit.__byte0
#define GPIO3MIS0_bit       GPIO3MIS_bit.__byte0_bit
#define GPIO3MISL           GPIO3MIS_bit.__shortl
#define GPIO3MISL_bit       GPIO3MIS_bit.__shortl_bit
__IO_REG32_BIT(GPIO3IC,               0x5003801C,__WRITE      ,__gpio3_bits);
#define GPIO3IC0            GPIO3IC_bit.__byte0
#define GPIO3IC0_bit        GPIO3IC_bit.__byte0_bit
#define GPIO3ICL            GPIO3IC_bit.__shortl
#define GPIO3ICL_bit        GPIO3IC_bit.__shortl_bit

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG8(     U0RBRTHR,              0x40008000,__READ_WRITE);
#define U0DLL U0RBRTHR
#define U0RBR U0RBRTHR
#define U0THR U0RBRTHR

/* U0DLM and U0IER share the same address */
__IO_REG32_BIT(U0IER,                 0x40008004,__READ_WRITE ,__uartier_bits);
#define U0DLM      U0IER

/* U0FCR and U0IIR share the same address */
__IO_REG32_BIT(U0FCR,                 0x40008008,__READ_WRITE ,__uartfcriir_bits);
#define U0IIR      U0FCR
#define U0IIR_bit  U0FCR_bit

__IO_REG8_BIT( U0LCR,                 0x4000800C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U0MCR,                 0x40008010,__READ_WRITE ,__uartmcr_bits);
__IO_REG8_BIT( U0LSR,                 0x40008014,__READ       ,__uartlsr_bits);
__IO_REG8_BIT( U0MSR,                 0x40008018,__READ       ,__uartmsr_bits);
__IO_REG8(     U0SCR,                 0x4000801C,__READ_WRITE);
__IO_REG32_BIT(U0ACR,                 0x40008020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U0FDR,                 0x40008028,__READ_WRITE ,__uartfdr_bits);
__IO_REG8_BIT( U0TER,                 0x40008030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U0RS485CTRL,           0x4000804C,__READ_WRITE ,__u1rs485ctrl_bits);
__IO_REG8(     U0ADRMATCH,            0x40008050,__READ_WRITE );
__IO_REG8(     U0RS485DLY,            0x40008054,__READ_WRITE );

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
__IO_REG32_BIT(SSP0ICR,               0x40040020,__READ_WRITE ,__sspicr_bits);

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
__IO_REG32_BIT(SSP1ICR,               0x40058020,__READ_WRITE ,__sspicr_bits);

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
__IO_REG32_BIT(TMR16B0IR,             0x4000C000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(TMR16B0TCR,            0x4000C004,__READ_WRITE ,__tcr_bits);
__IO_REG16(    TMR16B0TC,             0x4000C008,__READ_WRITE);
__IO_REG16(    TMR16B0PR,             0x4000C00C,__READ_WRITE);
__IO_REG16(    TMR16B0PC,             0x4000C010,__READ_WRITE);
__IO_REG32_BIT(TMR16B0MCR,            0x4000C014,__READ_WRITE ,__mcr_bits);
__IO_REG16(    TMR16B0MR0,            0x4000C018,__READ_WRITE);
__IO_REG16(    TMR16B0MR1,            0x4000C01C,__READ_WRITE);
__IO_REG16(    TMR16B0MR2,            0x4000C020,__READ_WRITE);
__IO_REG16(    TMR16B0MR3,            0x4000C024,__READ_WRITE);
__IO_REG32_BIT(TMR16B0CCR,            0x4000C028,__READ_WRITE ,__tccr_bits);
__IO_REG16(    TMR16B0CR0,            0x4000C02C,__READ);
__IO_REG32_BIT(TMR16B0EMR,            0x4000C03C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(TMR16B0CTCR,           0x4000C070,__READ_WRITE ,__ctcr_bits);
__IO_REG32_BIT(TMR16B0PWMC,           0x4000C074,__READ_WRITE ,__pwmc_bits);

/***************************************************************************
 **
 ** CT16B1
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR16B1IR,             0x40010000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(TMR16B1TCR,            0x40010004,__READ_WRITE ,__tcr_bits);
__IO_REG16(    TMR16B1TC,             0x40010008,__READ_WRITE);
__IO_REG16(    TMR16B1PR,             0x4001000C,__READ_WRITE);
__IO_REG16(    TMR16B1PC,             0x40010010,__READ_WRITE);
__IO_REG32_BIT(TMR16B1MCR,            0x40010014,__READ_WRITE ,__mcr_bits);
__IO_REG16(    TMR16B1MR0,            0x40010018,__READ_WRITE);
__IO_REG16(    TMR16B1MR1,            0x4001001C,__READ_WRITE);
__IO_REG16(    TMR16B1MR2,            0x40010020,__READ_WRITE);
__IO_REG16(    TMR16B1MR3,            0x40010024,__READ_WRITE);
__IO_REG32_BIT(TMR16B1CCR,            0x40010028,__READ_WRITE ,__tccr_bits);
__IO_REG16(    TMR16B1CR0,            0x4001002C,__READ);
__IO_REG32_BIT(TMR16B1EMR,            0x4001003C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(TMR16B1CTCR,           0x40010070,__READ_WRITE ,__ctcr_bits);
__IO_REG32_BIT(TMR16B1PWMC,           0x40010074,__READ_WRITE ,__pwmc_bits);

/***************************************************************************
 **
 ** CT32B0
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR32B0IR,             0x40014000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(TMR32B0TCR,            0x40014004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    TMR32B0TC,             0x40014008,__READ_WRITE);
__IO_REG32(    TMR32B0PR,             0x4001400C,__READ_WRITE);
__IO_REG32(    TMR32B0PC,             0x40014010,__READ_WRITE);
__IO_REG32_BIT(TMR32B0MCR,            0x40014014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    TMR32B0MR0,            0x40014018,__READ_WRITE);
__IO_REG32(    TMR32B0MR1,            0x4001401C,__READ_WRITE);
__IO_REG32(    TMR32B0MR2,            0x40014020,__READ_WRITE);
__IO_REG32(    TMR32B0MR3,            0x40014024,__READ_WRITE);
__IO_REG32_BIT(TMR32B0CCR,            0x40014028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    TMR32B0CR0,            0x4001402C,__READ);
__IO_REG32_BIT(TMR32B0EMR,            0x4001403C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(TMR32B0CTCR,           0x40014070,__READ_WRITE ,__ctcr_bits);
__IO_REG32_BIT(TMR32B0PWMC,           0x40014074,__READ_WRITE ,__pwmc_bits);

/***************************************************************************
 **
 ** CT32B1
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR32B1IR,             0x40018000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(TMR32B1TCR,            0x40018004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    TMR32B1TC,             0x40018008,__READ_WRITE);
__IO_REG32(    TMR32B1PR,             0x4001800C,__READ_WRITE);
__IO_REG32(    TMR32B1PC,             0x40018010,__READ_WRITE);
__IO_REG32_BIT(TMR32B1MCR,            0x40018014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    TMR32B1MR0,            0x40018018,__READ_WRITE);
__IO_REG32(    TMR32B1MR1,            0x4001801C,__READ_WRITE);
__IO_REG32(    TMR32B1MR2,            0x40018020,__READ_WRITE);
__IO_REG32(    TMR32B1MR3,            0x40018024,__READ_WRITE);
__IO_REG32_BIT(TMR32B1CCR,            0x40018028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    TMR32B1CR0,            0x4001802C,__READ);
__IO_REG32_BIT(TMR32B1EMR,            0x4001803C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(TMR32B1CTCR,           0x40018070,__READ_WRITE ,__ctcr_bits);
__IO_REG32_BIT(TMR32B1PWMC,           0x40018074,__READ_WRITE ,__pwmc_bits);

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

/***************************************************************************
 **
 ** Watchdog
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,                 0x40004000,__READ_WRITE ,__wdmod_bits);
__IO_REG32(    WDTC,                  0x40004004,__READ_WRITE);
__IO_REG32_BIT(WDFEED,                0x40004008,__WRITE      ,__wdfeed_bits);
__IO_REG32(    WDTV,                  0x4000400C,__READ);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASHCFG,              0x4003C010,__READ_WRITE ,__flashcfg_bits);

/***************************************************************************
 **
 ** Flash signature generation
 **
 ***************************************************************************/
__IO_REG32_BIT(FMSSTART,              0x4003C020,__READ_WRITE ,__fmsstart_bits);
__IO_REG32_BIT(FMSSTOP,               0x4003C024,__READ_WRITE ,__fmsstop_bits);
__IO_REG32(    FMSW0,                 0x4003C02C,__READ       );
__IO_REG32(    FMSW1,                 0x4003C030,__READ       );
__IO_REG32(    FMSW2,                 0x4003C034,__READ       );
__IO_REG32(    FMSW3,                 0x4003C038,__READ       );
__IO_REG32_BIT(FMSTAT,                0x4003CFE0,__READ       ,__fmstat_bits);
__IO_REG32_BIT(FMSTATCLR,             0x4003CFE8,__WRITE      ,__fmstatclr_bits);

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
#define NVIC_WAKE_UP0         16  /* Wake Up Interrupt 0                                    */
#define NVIC_WAKE_UP1         17  /* Wake Up Interrupt 1                                    */
#define NVIC_WAKE_UP2         18  /* Wake Up Interrupt 2                                    */
#define NVIC_WAKE_UP3         19  /* Wake Up Interrupt 3                                    */
#define NVIC_WAKE_UP4         20  /* Wake Up Interrupt 4                                    */                                    */
#define NVIC_WAKE_UP5         21  /* Wake Up Interrupt 5                                    */
#define NVIC_WAKE_UP6         22  /* Wake Up Interrupt 6                                    */
#define NVIC_WAKE_UP7         23  /* Wake Up Interrupt 7                                    */
#define NVIC_WAKE_UP8         24  /* Wake Up Interrupt 8                                    */
#define NVIC_WAKE_UP9         25  /* Wake Up Interrupt 9                                    */
#define NVIC_WAKE_UP10        26  /* Wake Up Interrupt 10                                   */
#define NVIC_WAKE_UP11        27  /* Wake Up Interrupt 11                                   */
#define NVIC_WAKE_UP12        28  /* Wake Up Interrupt 12                                   */
#define NVIC_SSP1             30  /* SSP1 Tx FIFO, Rx FIFO,Rx Timeout,Rx Overrun            */
#define NVIC_I2C0             31  /* I2C0 SI (state change) Interrupt                       */
#define NVIC_CT16B0           32  /* CT16B0 Match 0-3, Capture 0                            */
#define NVIC_CT16B1           33  /* CT16B1 Match 0-3, Capture 0                            */
#define NVIC_CT32B0           34  /* CT32B0 Match 0-3, Capture 0                            */
#define NVIC_CT32B1           35  /* CT32B1 Match 0-3, Capture 0                            */
#define NVIC_SSP0             36  /* SSP0 Tx FIFO, Rx FIFO,Rx Timeout,Rx Overrun            */
#define NVIC_UART0            37  /* UART0 RLS,THRE, RDA, CTI                               */
#define NVIC_ADC              40  /* A/D Converter end of conversion                        */
#define NVIC_WDT              41  /* WDT                                                    */
#define NVIC_BOD              42  /* BOD                                                    */
#define NVIC_PIO_3            44  /* PIO_3                                                  */
#define NVIC_PIO_2            45  /* PIO_2                                                  */
#define NVIC_PIO_1            46  /* PIO_1                                                  */
#define NVIC_PIO_0            47  /* PIO_0                                                  */

#endif    /* __IOLPC1113_H */

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
Interrupt9   = WAKE_UP0       0x40
Interrupt10  = WAKE_UP1       0x44
Interrupt11  = WAKE_UP2       0x48
Interrupt12  = WAKE_UP3       0x4C
Interrupt13  = WAKE_UP4       0x50
Interrupt14  = WAKE_UP5       0x54
Interrupt15  = WAKE_UP6       0x58
Interrupt16  = WAKE_UP7       0x5C
Interrupt17  = WAKE_UP8       0x60
Interrupt18  = WAKE_UP9       0x64
Interrupt19  = WAKE_UP10      0x68
Interrupt20  = WAKE_UP11      0x6C
Interrupt21  = WAKE_UP12      0x70
Interrupt22  = SSP1           0x78
Interrupt23  = I2C0           0x7C
Interrupt24  = CT16B0         0x80
Interrupt25  = CT16B1         0x84
Interrupt26  = CT32B0         0x88
Interrupt27  = CT32B1         0x8C
Interrupt28  = SSP0           0x90
Interrupt29  = UART0          0x94
Interrupt30  = ADC            0xA0
Interrupt31  = WDT            0xA4
Interrupt32  = BOD            0xA8
Interrupt33  = PIO_3          0xB0
Interrupt34  = PIO_2          0xB4
Interrupt35  = PIO_1          0xB8
Interrupt36  = PIO_0          0xBC
###DDF-INTERRUPT-END###*/
