/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Samsung S3FN21
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 50783 $
 **
 ***************************************************************************/

#ifndef __S3FN21_H
#define __S3FN21_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    S3FN21 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C specific declarations  ************************************************/

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

/* ADC ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __adc_idr_bits;

/* ADC Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __adc_srr_bits;

/* ADC Control Set Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32  START          : 1;
  __REG32                 :30;
} __adc_csr_bits;

/* ADC Control Clear Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32                 :31;
} __adc_ccr_bits;

/* ADC Control Divider Register */
typedef struct {
  __REG32  CDIV           : 5;
  __REG32                 :27;
} __adc_cdr_bits;

/* ADC Mode Register */
typedef struct {
  __REG32  CCSEL          : 4;
  __REG32                 : 1;
  __REG32  TRIG           : 3;
  __REG32                 :24;
} __adc_mr_bits;

/* ADC Status Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32  BUSY           : 1;
  __REG32                 :30;
} __adc_sr_bits;

/* ADC Interrupt Mask Set/Clear Register */
/* ADC Raw Interrupt Status Register */
/* ADC Masked Interrupt Status Register */
/* ADC Interrupt Clear Register */
typedef struct {
  __REG32  EOC            : 1;
  __REG32  OVR            : 1;
  __REG32                 :30;
} __adc_imscr_bits;

/* ADC Conversion Result Register */
typedef struct {
  __REG32  DATA           :10;
  __REG32                 :22;
} __adc_crr_bits;

/* ADC Conversion Result Register */
typedef struct {
  __REG32  DMAE           : 1;
  __REG32                 :31;
} __adc_dmacr_bits;

/* CM ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __cm_idr_bits;

/* CM Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __cm_srr_bits;

/* CM Control Set Register */
/* CM Control Clear Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32                 : 1;
  __REG32  FWAKE          : 1;
  __REG32  USBPLL         : 1;
  __REG32  PLL            : 1;
  __REG32  STCLK          : 1;
  __REG32  PCLK           : 1;
  __REG32  ISCLKS         : 1;
  __REG32  IDLEW          : 1;
  __REG32                 : 7;
  __REG32  IDLESP         : 1;
  __REG32  ESCMRST        : 1;
  __REG32  ESCM           : 1;
  __REG32  EMCMRST        : 1;
  __REG32  EMCM           : 1;
  __REG32                 : 8;
} __cm_csr_bits;

/* CM Peripheral Clock Set Register */
/* CM Peripheral Clock Clear Register */
/* CM Peripheral Clock Status Register */
typedef struct {
  __REG32  SFMCLK         : 1;
  __REG32                 : 1;
  __REG32  WDTCLK         : 1;
  __REG32  FRTCLK         : 1;
  __REG32  PWM0CLK        : 1;
  __REG32  PWM1CLK        : 1;
  __REG32  WTCLK          : 1;
  __REG32                 : 1;
  __REG32  TC0CLK         : 1;
  __REG32  TC1CLK         : 1;
  __REG32  TC2CLK         : 1;
  __REG32  TC3CLK         : 1;
  __REG32  TC4CLK         : 1;
  __REG32  TC5CLK         : 1;
  __REG32  TC6CLK         : 1;
  __REG32  TC7CLK         : 1;
  __REG32  USART0CLK      : 1;
  __REG32  USART1CLK      : 1;
  __REG32  USART2CLK      : 1;
  __REG32                 : 3;
  __REG32  ADCCLK         : 1;
  __REG32  LCDCLK         : 1;
  __REG32  SPI0CLK        : 1;
  __REG32  SPI1CLK        : 1;
  __REG32  I2C0CLK        : 1;
  __REG32  I2C1CLK        : 1;
  __REG32                 : 1;
  __REG32  PFCCLK         : 1;
  __REG32  IOCLK          : 1;
  __REG32                 : 1;
} __cm_pcsr_bits;

/* CM Mode Register 0 */
typedef struct {
  __REG32  LVDRL          : 3;
  __REG32  LVDRSTEN       : 1;
  __REG32  LVDIL          : 3;
  __REG32  LVDINTEN       : 1;
  __REG32                 : 1;
  __REG32  RXEV           : 1;
  __REG32  STCLKEN        : 1;
  __REG32  LVDPD          : 1;
  __REG32  CLKOUT         : 3;
  __REG32                 :17;
} __cm_mr0_bits;

/* CM Mode Register 1 */
typedef struct {
  __REG32  SYSCLK         : 3;
  __REG32                 : 1;
  __REG32  WDTCLK         : 3;
  __REG32                 : 1;
  __REG32  FRTCLK         : 3;
  __REG32                 : 5;
  __REG32  LCDCLK         : 3;
  __REG32                 :13;
} __cm_mr1_bits;

/* CM Interrupt Mask Set/Clear Register */
/* CM Masked Interrupt Status Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32  STABLE         : 1;
  __REG32                 : 1;
  __REG32  USBPLL         : 1;
  __REG32  PLL            : 1;
  __REG32                 : 4;
  __REG32  ESCKFAIL_END   : 1;
  __REG32  ESCKFAIL       : 1;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32                 : 1;
  __REG32  CMDERR         : 1;
  __REG32                 :13;
} __cm_imscr_bits;

/* CM RAW Interrupt Status Register */
/* CM Interrupt Clear Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32  STABLE         : 1;
  __REG32                 : 1;
  __REG32  USBPLL         : 1;
  __REG32  PLL            : 1;
  __REG32                 : 4;
  __REG32  ESCKFAIL_END   : 1;
  __REG32  ESCKFAIL       : 1;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32  LVDRS          : 1;
  __REG32  CMDERR         : 1;
  __REG32                 :13;
} __cm_risr_bits;

/* CM Status Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32  STABLE         : 1;
  __REG32  FWAKE          : 1;
  __REG32  USBPLL         : 1;
  __REG32  PLL            : 1;
  __REG32  STCLK          : 1;
  __REG32  PCLK           : 1;
  __REG32  ISCLKS         : 1;
  __REG32  IDLEW          : 1;
  __REG32  ESCKFAIL_END   : 1;
  __REG32  ESCKFAIL       : 1;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32  LVDRS          : 1;
  __REG32  CMDERR         : 1;
  __REG32  IDLESP         : 1;
  __REG32  ESCMRST        : 1;
  __REG32  ESCM           : 1;
  __REG32  EMCMRST        : 1;
  __REG32  EMCM           : 1;
  __REG32  SWRSTS         : 1;
  __REG32  NRSTS          : 1;
  __REG32  LVDRSTS        : 1;
  __REG32  WDTRSTS        : 1;
  __REG32  PORRSTS        : 1;
  __REG32  ESCMRSTS       : 1;
  __REG32  EMCMRSTS       : 1;
  __REG32  SYSRSTS        : 1;
} __cm_sr_bits;

/* CM System Clock Divider Register */
typedef struct {
  __REG32  SDIV           : 3;
  __REG32                 :13;
  __REG32  SDIVKEY        :16;
} __cm_scdr_bits;

/* CM Peripheral Clock Divider Register */
typedef struct {
  __REG32  PDIV           : 4;
  __REG32                 :12;
  __REG32  PDIVKEY        :16;
} __cm_pcdr_bits;

/* CM FRT Clock Divider Register */
typedef struct {
  __REG32  NDIV           : 4;
  __REG32  MDIV           : 3;
  __REG32                 : 9;
  __REG32  FDIVKEY        :16;
} __cm_fcdr_bits;

/* CM LCD Clock Divider Register */
typedef struct {
  __REG32  KDIV           : 4;
  __REG32  JDIV           : 3;
  __REG32                 : 9;
  __REG32  LDIVKEY        :16;
} __cm_lcdr_bits;

/* CM PLL Stabilization Time Register */
typedef struct {
  __REG32  PST            :11;
  __REG32                 : 5;
  __REG32  PLLSKEY        :16;
} __cm_pstr_bits;

/* CM PLL Divider Parameters Register */
typedef struct {
  __REG32  PLLMUL         : 8;
  __REG32  PLLPRE         : 6;
  __REG32                 : 2;
  __REG32  PLLPOST        : 2;
  __REG32                 : 5;
  __REG32  LFPASS         : 1;
  __REG32  PLLKEY         : 8;
} __cm_pdpr_bits;

/* CLKMNGR USB PLL Stabilization Time Register */
typedef struct {
  __REG32  UPST           :11;
  __REG32                 : 5;
  __REG32  UPLLSKEY       :16;
} __cm_upstr_bits;

/* CLKMNGR USB PLL Divider Parameters Register */
typedef struct {
  __REG32  UPLLMUL        : 8;
  __REG32  UPLLPRE        : 6;
  __REG32                 : 2;
  __REG32  UPLLPOST       : 2;
  __REG32                 : 5;
  __REG32  LFPASS         : 1;
  __REG32  UPLLKEY        : 8;
} __cm_updpr_bits;

/* CM External Main Clock Stabilization Time Register */
typedef struct {
  __REG32  EMST           :16;
  __REG32  EMSKEY         :16;
} __cm_emstr_bits;

/* CM External Sub Clock Stabilization Time Register */
typedef struct {
  __REG32  ESST           :16;
  __REG32  ESSKEY         :16;
} __cm_esstr_bits;

/* CM Basic Timer Clock Divider Register */
typedef struct {
  __REG32  BTCDIV         : 4;
  __REG32                 :12;
  __REG32  BTCDKEY        :16;
} __cm_btcdr_bits;

/* CM Basic Timer Register */
typedef struct {
  __REG32  BTCV           :16;
  __REG32                 :16;
} __cm_btr_bits;

/* CM Wakeup Control Register 0 */
typedef struct {
  __REG32  WSRC0          : 5;
  __REG32                 : 1;
  __REG32  EDGE0          : 1;
  __REG32  WEN0           : 1;
  __REG32  WSRC1          : 5;
  __REG32                 : 1;
  __REG32  EDGE1          : 1;
  __REG32  WEN1           : 1;
  __REG32  WSRC2          : 5;
  __REG32                 : 1;
  __REG32  EDGE2          : 1;
  __REG32  WEN2           : 1;
  __REG32  WSRC3          : 5;
  __REG32                 : 1;
  __REG32  EDGE3          : 1;
  __REG32  WEN3           : 1;
} __cm_wcr0_bits;

/* CM Wakeup Control Register 1 */
typedef struct {
  __REG32  WSRC4          : 5;
  __REG32                 : 1;
  __REG32  EDGE4          : 1;
  __REG32  WEN4           : 1;
  __REG32  WSRC5          : 5;
  __REG32                 : 1;
  __REG32  EDGE5          : 1;
  __REG32  WEN5           : 1;
  __REG32  WSRC6          : 5;
  __REG32                 : 1;
  __REG32  EDGE6          : 1;
  __REG32  WEN6           : 1;
  __REG32  WSRC7          : 5;
  __REG32                 : 1;
  __REG32  EDGE7          : 1;
  __REG32  WEN7           : 1;
} __cm_wcr1_bits;

/* CM Wakeup Interrupt Mask Set/Clear Register */
/* CM Wakeup Raw Interrupt Status Register */
/* CM Wakeup Masked Interrupt Status Register */
/* CM Wakeup Interrupt Clear Register */
typedef struct {
  __REG32  WI0            : 1;
  __REG32  WI1            : 1;
  __REG32  WI2            : 1;
  __REG32  WI3            : 1;
  __REG32  WI4            : 1;
  __REG32  WI5            : 1;
  __REG32  WI6            : 1;
  __REG32  WI7            : 1;
  __REG32                 :24;
} __cm_wimscr_bits;

/* CM Wakeup Interrupt Clear Register */
typedef struct {
  __REG32  NVIC0          : 1;
  __REG32  NVIC1          : 1;
  __REG32  NVIC2          : 1;
  __REG32  NVIC3          : 1;
  __REG32  NVIC4          : 1;
  __REG32  NVIC5          : 1;
  __REG32  NVIC6          : 1;
  __REG32  NVIC7          : 1;
  __REG32  NVIC8          : 1;
  __REG32  NVIC9          : 1;
  __REG32  NVIC10         : 1;
  __REG32  NVIC11         : 1;
  __REG32  NVIC12         : 1;
  __REG32  NVIC13         : 1;
  __REG32  NVIC14         : 1;
  __REG32  NVIC15         : 1;
  __REG32  NVIC16         : 1;
  __REG32  NVIC17         : 1;
  __REG32  NVIC18         : 1;
  __REG32  NVIC19         : 1;
  __REG32  NVIC20         : 1;
  __REG32  NVIC21         : 1;
  __REG32  NVIC22         : 1;
  __REG32  NVIC23         : 1;
  __REG32  NVIC24         : 1;
  __REG32  NVIC25         : 1;
  __REG32  NVIC26         : 1;
  __REG32  NVIC27         : 1;
  __REG32  NVIC28         : 1;
  __REG32  NVIC29         : 1;
  __REG32  NVIC30         : 1;
  __REG32  NVIC31         : 1;
} __cm_nisr_bits;

/* CM Power Status Register */
typedef struct {
  __REG32  SUBIVC         : 1;
  __REG32  NORIVC         : 1;
  __REG32                 :29;
  __REG32  VUSBDET        : 1;
} __cm_psr_bits;

/* DMA Channel x Initial Source Control Register */
/* DMA Channel x Initial Destination Control Register */
typedef struct {
  __REG32  LINC                  : 1;
  __REG32  HINC                  : 1;
  __REG32                        :30;
} __dma_iscr_bits;

/* DMA Channel x Control Register */
typedef struct {
  __REG32  LTC                   :12;
  __REG32  HTC                   :12;
  __REG32  DSIZE                 : 2;
  __REG32  RELOAD                : 1;
  __REG32  SMODE                 : 1;
  __REG32  TSIZE                 : 1;
  __REG32  LTCINT                : 1;
  __REG32  TCINT                 : 1;
  __REG32                        : 1;
} __dma_cr_bits;

/* DMA Channel x Status Register */
typedef struct {
  __REG32  CURR_LTC              :12;
  __REG32  CURR_HTC              :12;
  __REG32                        : 7;
  __REG32  LTCST                 : 1;
} __dma_sr_bits;

/* DMA Channel x Mask Trigger Register */
typedef struct {
  __REG32  SWTRIG                : 1;
  __REG32  CHEN                  : 1;
  __REG32  STOP                  : 1;
  __REG32                        :29;
} __dma_mtr_bits;

/* DMA Channel x Request Selection Register */
typedef struct {
  __REG32  REQ                   : 1;
  __REG32  HWSRC                 : 5;
  __REG32                        :26;
} __dma_rsr_bits;

/* DMA ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __dma_idr_bits;

/* DMA Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __dma_srr_bits;

/* DMA Channel Enable Status Register */
typedef struct {
  __REG32  CH0EN                 : 1;
  __REG32  CH1EN                 : 1;
  __REG32  CH2EN                 : 1;
  __REG32  CH3EN                 : 1;
  __REG32  CH4EN                 : 1;
  __REG32  CH5EN                 : 1;
  __REG32                        :26;
} __dma_cesr_bits;

/* DMA Interrupt Status Register */
typedef struct {
  __REG32  CH0_LTCIT             : 1;
  __REG32  CH1_LTCIT             : 1;
  __REG32  CH2_LTCIT             : 1;
  __REG32  CH3_LTCIT             : 1;
  __REG32  CH4_LTCIT             : 1;
  __REG32  CH5_LTCIT             : 1;
  __REG32                        :10;
  __REG32  CH0_TCIT              : 1;
  __REG32  CH1_TCIT              : 1;
  __REG32  CH2_TCIT              : 1;
  __REG32  CH3_TCIT              : 1;
  __REG32  CH4_TCIT              : 1;
  __REG32  CH5_TCIT              : 1;
  __REG32                        :10;
} __dma_isr_bits;

/* DMA Interrupt Status Register */
typedef struct {
  __REG32  CH0_IT                : 1;
  __REG32  CH1_IT                : 1;
  __REG32  CH2_IT                : 1;
  __REG32  CH3_IT                : 1;
  __REG32  CH4_IT                : 1;
  __REG32  CH5_IT                : 1;
  __REG32                        :26;
} __dma_icr_bits;

/* FRT ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __frt_idr_bits;

/* Clock Enable/Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __frt_cedr_bits;

/* Timer A Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __frt_srr_bits;

/* Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 7;
  __REG32  FRTSIZE               : 5;
  __REG32                        : 3;
  __REG32  CKDIV                 :16;
} __frt_cr_bits;

/* Interrupt Enable Disable Register */
/* Raw Interrupt Status Register */
/* Masked Interrupt Status Register */
/* Interrupt Clear Register */
typedef struct {
  __REG32  OVF                   : 1;
  __REG32                        : 1;
  __REG32  MATCH                 : 1;
  __REG32                        :29;
} __frt_imscr_bits;

/* GPIO ID-Code Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __gpio_idr_bits;

/* GPIO Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :31;
} __gpio_cedr_bits;

/* GPIO Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __gpio_srr_bits;

/* GPIO Interrupt Mask Set/Clear Register */
/* GPIO Raw Interrupt Status Register */
/* GPIO Masked Interrupt Status Register */
/* GPIO Interrupt Clear Register */
/* GPIO Output Enable Register */
/* GPIO Output Status Register */
/* GPIO Write Output Data Register */
/* GPIO Set Output Data Register */
/* GPIO Clear Output Data Register */
/* GPIO Output Data Status Register */
/* GPIO Pin Data Status Register */
typedef struct {
  __REG32  P0             : 1;
  __REG32  P1             : 1;
  __REG32  P2             : 1;
  __REG32  P3             : 1;
  __REG32  P4             : 1;
  __REG32  P5             : 1;
  __REG32  P6             : 1;
  __REG32  P7             : 1;
  __REG32  P8             : 1;
  __REG32  P9             : 1;
  __REG32  P10            : 1;
  __REG32  P11            : 1;
  __REG32  P12            : 1;
  __REG32  P13            : 1;
  __REG32  P14            : 1;
  __REG32  P15            : 1;
  __REG32  P16            : 1;
  __REG32  P17            : 1;
  __REG32  P18            : 1;
  __REG32  P19            : 1;
  __REG32  P20            : 1;
  __REG32  P21            : 1;
  __REG32  P22            : 1;
  __REG32  P23            : 1;
  __REG32  P24            : 1;
  __REG32  P25            : 1;
  __REG32  P26            : 1;
  __REG32  P27            : 1;
  __REG32  P28            : 1;
  __REG32  P29            : 1;
  __REG32  P30            : 1;
  __REG32  P31            : 1;
} __gpio_imscr_bits;

/* I2C ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __i2c_idr_bits;

/* I2C Clock Enable Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :31;
} __i2c_cedr_bits;

/* I2C Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __i2c_srr_bits;

/* I2C Control Register */
typedef struct {
  __REG32                        : 1;
  __REG32  AA                    : 1;
  __REG32  STO                   : 1;
  __REG32  STA                   : 1;
  __REG32  SI                    : 1;
  __REG32                        : 3;
  __REG32  ENA                   : 1;
  __REG32                        :23;
} __i2c_cr_bits;

/* I2C Mode Register */
typedef struct {
  __REG32  PRV                   :12;
  __REG32  FAST                  : 1;
  __REG32                        :19;
} __i2c_mr_bits;

/* I2C Status Register */
typedef struct {
  __REG32                        : 3;
  __REG32  SR                    : 5;
  __REG32                        :24;
} __i2c_sr_bits;

/* I2C Interrupt Mask Set and Clear Register */
/* I2C Raw Interrupt Status Register */
/* I2C Masked Interrupt Status Register */
/* I2C Clear Interrupt Status Register */
typedef struct {
  __REG32                        : 4;
  __REG32  SI                    : 1;
  __REG32                        :27;
} __i2c_imscr_bits;

/* I2C Serial Data Register */
typedef struct {
  __REG32  DAT                   : 8;
  __REG32                        :24;
} __i2c_sdr_bits;

/* I2C Serial Slave Address Register */
typedef struct {
  __REG32  GC                    : 1;
  __REG32  ADR                   : 7;
  __REG32                        :24;
} __i2c_ssar_bits;

/* I2C Hold/Setup Delay Register */
typedef struct {
  __REG32  DL                    : 8;
  __REG32                        :24;
} __i2c_hsdr_bits;

/* I2C DMA Control register */
typedef struct {
  __REG32  RXDMAE                : 1;
  __REG32  TXDMAE                : 1;
  __REG32                        :30;
} __i2c_dmacr_bits;

/* Flash ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __ipf_idr_bits;

/* Flash Clock Enable/Disable Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __ipf_srr_bits;

/* Flash Software Reset Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :31;
} __ipf_cedr_bits;

/* Flash Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 3;
  __REG32  CMD                   : 3;
  __REG32                        :25;
} __ipf_cr_bits;

/* Flash Mode Register */
typedef struct {
  __REG32  BACEN                 : 1;
  __REG32                        : 6;
  __REG32  FSMODE                : 1;
  __REG32                        :24;
} __ipf_mr_bits;

/* Flash Interrupt Mask Set and Clear Register */
/* Flash Raw Interrupt Status Register */
/* Flash Masked Interrupt Status Register */
/* Flash Interrupt Clear Register */
typedef struct {
  __REG32  END                   : 1;
  __REG32                        : 7;
  __REG32  ERR0                  : 1;
  __REG32  ERR1                  : 1;
  __REG32  ERR2                  : 1;
  __REG32                        :21;
} __ipf_imscr_bits;

/* Flash Status Register */
typedef struct {
  __REG32  BUSY                  : 1;
  __REG32                        :31;
} __ipf_sr_bits;

/* Smart Option Protection Status Register */
typedef struct {
  __REG32                        : 4;
  __REG32  nHWPA0                : 1;
  __REG32  nHWPA1                : 1;
  __REG32  nHWPA2                : 1;
  __REG32  nHWPA3                : 1;
  __REG32  nJTAGP                : 1;
  __REG32                        : 3;
  __REG32  nHWPA4                : 1;
  __REG32  nHWPA5                : 1;
  __REG32  nHWPA6                : 1;
  __REG32  nHWPA7                : 1;
  __REG32                        : 1;
  __REG32  nHWP                  : 1;
  __REG32                        : 2;
  __REG32  nHWPA8                : 1;
  __REG32  nHWPA9                : 1;
  __REG32  nHWPA10               : 1;
  __REG32  nHWPA11               : 1;
  __REG32                        : 3;
  __REG32  nSRP                  : 1;
  __REG32  nHWPA12               : 1;
  __REG32  nHWPA13               : 1;
  __REG32  nHWPA14               : 1;
  __REG32  nHWPA15               : 1;
} __so_psr_bits;

/* Smart Option Configuration Status Register */
typedef struct {
  __REG32  POCCS                 : 2;
  __REG32                        : 4;
  __REG32  IMSEL                 : 2;
  __REG32                        : 4;
  __REG32  BTDIV                 : 4;
  __REG32                        :16;
} __so_csr_bits;

/* Internal OSC Trimming Register */
typedef struct {
  __REG32  OSC0                  : 7;
  __REG32                        : 1;
  __REG32  OSC1                  : 7;
  __REG32                        : 1;
  __REG32  OSC2                  : 6;
  __REG32                        : 2;
  __REG32  IOTKEY                : 8;
} __pf_iotr_bits;

/* IO Mode Low Register x */
typedef struct {
  __REG32  IOx_0_FSEL            : 2;
  __REG32  IOx_1_FSEL            : 2;
  __REG32  IOx_2_FSEL            : 2;
  __REG32  IOx_3_FSEL            : 2;
  __REG32  IOx_4_FSEL            : 2;
  __REG32  IOx_5_FSEL            : 2;
  __REG32  IOx_6_FSEL            : 2;
  __REG32  IOx_7_FSEL            : 2;
  __REG32  IOx_8_FSEL            : 2;
  __REG32  IOx_9_FSEL            : 2;
  __REG32  IOx_10_FSEL           : 2;
  __REG32  IOx_11_FSEL           : 2;
  __REG32  IOx_12_FSEL           : 2;
  __REG32  IOx_13_FSEL           : 2;
  __REG32  IOx_14_FSEL           : 2;
  __REG32  IOx_15_FSEL           : 2;
} __ioconf_mlr_bits;

/* IO Mode High Register x */
typedef struct {
  __REG32  IOx_16_FSEL           : 2;
  __REG32  IOx_17_FSEL           : 2;
  __REG32  IOx_18_FSEL           : 2;
  __REG32  IOx_19_FSEL           : 2;
  __REG32  IOx_20_FSEL           : 2;
  __REG32  IOx_21_FSEL           : 2;
  __REG32  IOx_22_FSEL           : 2;
  __REG32  IOx_23_FSEL           : 2;
  __REG32  IOx_24_FSEL           : 2;
  __REG32  IOx_25_FSEL           : 2;
  __REG32  IOx_26_FSEL           : 2;
  __REG32  IOx_27_FSEL           : 2;
  __REG32  IOx_28_FSEL           : 2;
  __REG32  IOx_29_FSEL           : 2;
  __REG32  IOx_30_FSEL           : 2;
  __REG32  IOx_31_FSEL           : 2;
} __ioconf_mhr_bits;

/* IO Pull-Up Configuration Register x */
typedef struct {
  __REG32  IOx_0_PUEN            : 1;
  __REG32  IOx_1_PUEN            : 1;
  __REG32  IOx_2_PUEN            : 1;
  __REG32  IOx_3_PUEN            : 1;
  __REG32  IOx_4_PUEN            : 1;
  __REG32  IOx_5_PUEN            : 1;
  __REG32  IOx_6_PUEN            : 1;
  __REG32  IOx_7_PUEN            : 1;
  __REG32  IOx_8_PUEN            : 1;
  __REG32  IOx_9_PUEN            : 1;
  __REG32  IOx_10_PUEN           : 1;
  __REG32  IOx_11_PUEN           : 1;
  __REG32  IOx_12_PUEN           : 1;
  __REG32  IOx_13_PUEN           : 1;
  __REG32  IOx_14_PUEN           : 1;
  __REG32  IOx_15_PUEN           : 1;
  __REG32  IOx_16_PUEN           : 1;
  __REG32  IOx_17_PUEN           : 1;
  __REG32  IOx_18_PUEN           : 1;
  __REG32  IOx_19_PUEN           : 1;
  __REG32  IOx_20_PUEN           : 1;
  __REG32  IOx_21_PUEN           : 1;
  __REG32  IOx_22_PUEN           : 1;
  __REG32  IOx_23_PUEN           : 1;
  __REG32  IOx_24_PUEN           : 1;
  __REG32  IOx_25_PUEN           : 1;
  __REG32  IOx_26_PUEN           : 1;
  __REG32  IOx_27_PUEN           : 1;
  __REG32  IOx_28_PUEN           : 1;
  __REG32  IOx_29_PUEN           : 1;
  __REG32  IOx_30_PUEN           : 1;
  __REG32  IOx_31_PUEN           : 1;
} __ioconf_pucr_bits;

/* IO Pull-Up Configuration Register x */
typedef struct {
  __REG32  IOx_0_ODEN            : 1;
  __REG32  IOx_1_ODEN            : 1;
  __REG32  IOx_2_ODEN            : 1;
  __REG32  IOx_3_ODEN            : 1;
  __REG32  IOx_4_ODEN            : 1;
  __REG32  IOx_5_ODEN            : 1;
  __REG32  IOx_6_ODEN            : 1;
  __REG32  IOx_7_ODEN            : 1;
  __REG32  IOx_8_ODEN            : 1;
  __REG32  IOx_9_ODEN            : 1;
  __REG32  IOx_10_ODEN           : 1;
  __REG32  IOx_11_ODEN           : 1;
  __REG32  IOx_12_ODEN           : 1;
  __REG32  IOx_13_ODEN           : 1;
  __REG32  IOx_14_ODEN           : 1;
  __REG32  IOx_15_ODEN           : 1;
  __REG32  IOx_16_ODEN           : 1;
  __REG32  IOx_17_ODEN           : 1;
  __REG32  IOx_18_ODEN           : 1;
  __REG32  IOx_19_ODEN           : 1;
  __REG32  IOx_20_ODEN           : 1;
  __REG32  IOx_21_ODEN           : 1;
  __REG32  IOx_22_ODEN           : 1;
  __REG32  IOx_23_ODEN           : 1;
  __REG32  IOx_24_ODEN           : 1;
  __REG32  IOx_25_ODEN           : 1;
  __REG32  IOx_26_ODEN           : 1;
  __REG32  IOx_27_ODEN           : 1;
  __REG32  IOx_28_ODEN           : 1;
  __REG32  IOx_29_ODEN           : 1;
  __REG32  IOx_30_ODEN           : 1;
  __REG32  IOx_31_ODEN           : 1;
} __ioconf_odcr_bits;

/* LCD ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __lcd_idr_bits;

/* LCD Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :31;
} __lcd_cedr_bits;

/* LCD Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __lcd_srr_bits;

/* LCD Control Register */
typedef struct {
  __REG32  LCDEN          : 1;
  __REG32  DISC           : 2;
  __REG32                 : 1;
  __REG32  BTSEL          : 2;
  __REG32                 : 2;
  __REG32  DBSEL          : 4;
  __REG32                 :20;
} __lcd_cr_bits;

/* LCD Clock Divide Register */
typedef struct {
  __REG32  CDIV           : 3;
  __REG32                 : 4;
  __REG32  CDC            : 1;
  __REG32  CPRE           :16;
  __REG32                 : 8;
} __lcd_cdr_bits;

/* LCD Contrast Control Register */
typedef struct {
  __REG32  DIVRSEL        : 1;
  __REG32  CONTRAST       : 1;
  __REG32                 : 2;
  __REG32  CLEVEL         : 4;
  __REG32                 :24;
} __lcd_ccr_bits;

/* PWM ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __pwm_idr_bits;

/* PWM Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :30;
  __REG32  DBGEN          : 1;
} __pwm_cedr_bits;

/* PWM Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __pwm_srr_bits;

/* PWM Control Set Register */
typedef struct {
  __REG32  START          : 1;
  __REG32  UPDATE         : 1;
  __REG32                 : 6;
  __REG32  IDLESL         : 1;
  __REG32  OUTSL          : 1;
  __REG32  KEEP           : 1;
  __REG32  PWMIM          : 1;
  __REG32                 :12;
  __REG32  PWMEX0         : 1;
  __REG32  PWMEX1         : 1;
  __REG32  PWMEX2         : 1;
  __REG32  PWMEX3         : 1;
  __REG32  PWMEX4         : 1;
  __REG32  PWMEX5         : 1;
  __REG32                 : 2;
} __pwm_csr_bits;

/* PWM Control Clear Register */
/* PWM Status Register */
typedef struct {
  __REG32  START          : 1;
  __REG32                 : 7;
  __REG32  IDLESL         : 1;
  __REG32  OUTSL          : 1;
  __REG32  KEEP           : 1;
  __REG32  PWMIM          : 1;
  __REG32                 :12;
  __REG32  PWMEX0         : 1;
  __REG32  PWMEX1         : 1;
  __REG32  PWMEX2         : 1;
  __REG32  PWMEX3         : 1;
  __REG32  PWMEX4         : 1;
  __REG32  PWMEX5         : 1;
  __REG32                 : 2;
} __pwm_ccr_bits;

/* PWM Interrupt Mask Set/Clear Register */
/* PWM Raw Interrupt Status Register */
/* PWM Masked Interrupt Status Register */
/* PWM Interrupt Clear Register */
typedef struct {
  __REG32  PWMSTART       : 1;
  __REG32  PWMSTOP        : 1;
  __REG32  PSTART         : 1;
  __REG32  PEND           : 1;
  __REG32  PMATCH         : 1;
  __REG32                 :27;
} __pwm_imscr_bits;

/* PWM Clock Divider Register */
typedef struct {
  __REG32  DIVN           : 4;
  __REG32  DIVM           :12;
  __REG32                 :16;
} __pwm_cdr_bits;

/* PWM Period Register */
/* PWM Current Period Register */
typedef struct {
  __REG32  PERIOD         :16;
  __REG32                 :16;
} __pwm_prdr_bits;

/* PWM Pulse Register */
/* PWM Current Pulse Register */
typedef struct {
  __REG32  PULSE          :16;
  __REG32                 :16;
} __pwm_pulr_bits;

/* PWM Current Clock Divider Register */
typedef struct {
  __REG32  DIVN           : 4;
  __REG32  DIVM           :12;
  __REG32                 :16;
} __pwm_ccdr_bits;

/* SPIx Control Register 0 */
typedef struct {
  __REG32  DSS                   : 4;
  __REG32  FRF                   : 2;
  __REG32  SPO                   : 1;
  __REG32  SPH                   : 1;
  __REG32  SCR                   : 8;
  __REG32                        :16;
} __spi_cr0_bits;

/* SPIx Control Register 1 */
typedef struct {
  __REG32  LBM                   : 1;
  __REG32  SSE                   : 1;
  __REG32  MS                    : 1;
  __REG32  SOD                   : 1;
  __REG32  RXIFLSEL              : 3;
  __REG32                        :25;
} __spi_cr1_bits;

/* SPIx status register */
typedef struct {
  __REG32  TFE                   : 1;
  __REG32  TNF                   : 1;
  __REG32  RNE                   : 1;
  __REG32  RFF                   : 1;
  __REG32  BSY                   : 1;
  __REG32                        :27;
} __spi_sr_bits;

/* SPIx Clock prescaler register */
typedef struct {
  __REG32  CPSDVSR               : 8;
  __REG32                        :24;
} __spi_cpsr_bits;

/* SPIx interrupt mask set or clear register */
typedef struct {
  __REG32  RORIM                 : 1;
  __REG32  RTIM                  : 1;
  __REG32  RXIM                  : 1;
  __REG32  TXIM                  : 1;
  __REG32                        :28;
} __spi_imsc_bits;

/* SPIx raw interrupt status register */
typedef struct {
  __REG32  RORRIS                : 1;
  __REG32  RTRIS                 : 1;
  __REG32  RXRIS                 : 1;
  __REG32  TXRIS                 : 1;
  __REG32                        :28;
} __spi_ris_bits;

/* SPIx interrupt clear register */
typedef struct {
  __REG32  RORIC                 : 1;
  __REG32  RTIC                  : 1;
  __REG32                        :30;
} __spi_icr_bits;

/* SPIx DMA control register */
typedef struct {
  __REG32  RXDMAE                : 1;
  __REG32  TXDMAE                : 1;
  __REG32                        :30;
} __spi_dmacr_bits;

/* Timer/Counter ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __tc_idr_bits;

/* Timer/Counter Clock Source Selection Register */
typedef struct {
  __REG32  CLKSRC                : 1;
  __REG32                        :31;
} __tc_cssr_bits;

/* Timer/Counter Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __tc_cedr_bits;

/* Timer/Counter Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __tc_srr_bits;

/* Timer/Counter Control Set Register */
/* Timer/Counter Control Clear Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  UPDATE                : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32  ADCTRG                : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc_csr_bits;

/* Timer/Counter Status Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32  ADCTRG                : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc_sr_bits;

/* Timer/Counter Interrupt Mask Set/Clear Register */
/* Timer/Counter Raw Interrupt Status Register */
/* Timer/Counter Masked Interrupt Status Register */
/* Timer/Counter Interrupt Clear Register */
typedef struct {
  __REG32  STARTI                : 1;
  __REG32  STOPI                 : 1;
  __REG32  PSTARTI               : 1;
  __REG32  PENDI                 : 1;
  __REG32  MATI                  : 1;
  __REG32  OVFI                  : 1;
  __REG32  CAPTI                 : 1;
  __REG32                        :25;
} __tc_imscr_bits;

/* Timer/Counter Clock Divider Register */
/* Timer/Counter Current Clock Divider Register */
typedef struct {
  __REG32  DIVN                  : 4;
  __REG32  DIVM                  :28;
} __tc_cdr_bits;

/* Timer/Counter Counter Size Mask Register */
typedef struct {
  __REG32  SIZE                  : 4;
  __REG32                        :28;
} __tc_csmr_bits;

/* Timer/Counter Control Clear Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  UPDATE                : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32                        : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc32_csr_bits;

/* Timer/Counter Status Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32                        : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc32_sr_bits;

/* Timer/Counter Interrupt Mask Set/Clear Register */
/* Timer/Counter Raw Interrupt Status Register */
/* Timer/Counter Masked Interrupt Status Register */
/* Timer/Counter Interrupt Clear Register */
typedef struct {
  __REG32  STARTI                : 1;
  __REG32  STOPI                 : 1;
  __REG32  PSTARTI               : 1;
  __REG32  PENDI                 : 1;
  __REG32  MATCHI                : 1;
  __REG32  OVFI                  : 1;
  __REG32  CAPTI                 : 1;
  __REG32                        :25;
} __tc32_imscr_bits;

/* Timer/Counter Counter Size Mask Register */
typedef struct {
  __REG32  SIZE                  : 5;
  __REG32                        :27;
} __tc32_csmr_bits;

/* USART ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __us_idr_bits;

/* USART Clock Enable Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __us_cedr_bits;

/* USART Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __us_srr_bits;

/* USART Control Register */
typedef struct {
  __REG32                        : 2;
  __REG32  RSTRX                 : 1;
  __REG32  RSTTX                 : 1;
  __REG32  RXEN                  : 1;
  __REG32  RXDIS                 : 1;
  __REG32  TXEN                  : 1;
  __REG32  TXDIS                 : 1;
  __REG32                        : 1;
  __REG32  STTBRK                : 1;
  __REG32  STPBRK                : 1;
  __REG32  STTTO                 : 1;
  __REG32  SENDA                 : 1;
  __REG32                        : 3;
  __REG32  STHEADER              : 1;
  __REG32  STREPS                : 1;
  __REG32  STMESSAGE             : 1;
  __REG32  RSTLIN                : 1;
  __REG32                        :12;
} __us_cr_bits;

/* USART Mode Register */
typedef struct {
  __REG32  LIN                   : 1;
  __REG32  SENDTIME              : 3;
  __REG32  CLKS                  : 2;
  __REG32  CHRL                  : 2;
  __REG32  SYNC                  : 1;
  __REG32  PAR                   : 3;
  __REG32  NBSTOP                : 2;
  __REG32  CHMODE                : 2;
  __REG32  SMCARDPT              : 1;
  __REG32  MODE9                 : 1;
  __REG32  CLKO                  : 1;
  __REG32  LIN2_0                : 1;
  __REG32  DSB                   : 1;
  __REG32                        :11;
} __us_mr_bits;

/* USART Interrupt Mask Set and Clear Register */
/* USART Raw Interrupt Status Register */
/* USART Masked Interrupt Status Register */
typedef struct {
  __REG32  RXRDY                 : 1;
  __REG32  TXRDY                 : 1;
  __REG32  RXBRK                 : 1;
  __REG32  RXEND                 : 1;
  __REG32  TXEND                 : 1;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32                        :13;
  __REG32  ENDHEADER             : 1;
  __REG32  ENDMESS               : 1;
  __REG32  NOTRESP               : 1;
  __REG32  BITERROR              : 1;
  __REG32  IPERROR               : 1;
  __REG32  CHECKSUM              : 1;
  __REG32  WAKEUP                : 1;
  __REG32                        : 1;
} __us_imscr_bits;

/* USART Interrupt Clear Register */
typedef struct {
  __REG32                        : 2;
  __REG32  RXBRK                 : 1;
  __REG32  RXEND                 : 1;
  __REG32  TXEND                 : 1;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32                        :13;
  __REG32  ENDHEADER             : 1;
  __REG32  ENDMESS               : 1;
  __REG32  NOTRESP               : 1;
  __REG32  BITERROR              : 1;
  __REG32  IPERROR               : 1;
  __REG32  CHECKSUM              : 1;
  __REG32  WAKEUP                : 1;
  __REG32                        : 1;
} __us_icr_bits;

/* USART Status Register */
typedef struct {
  __REG32  RXRDY                 : 1;
  __REG32  TXRDY                 : 1;
  __REG32  RXBRK                 : 1;
  __REG32  RXEND                 : 1;
  __REG32  TXEND                 : 1;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32  IDLEFLAG              : 1;
  __REG32                        :12;
  __REG32  ENDHEADER             : 1;
  __REG32  ENDMESS               : 1;
  __REG32  NOTRESP               : 1;
  __REG32  BITERROR              : 1;
  __REG32  IPERROR               : 1;
  __REG32  CHECKSUM              : 1;
  __REG32  WAKEUP                : 1;
  __REG32  LINBUSY               : 1;
} __us_sr_bits;

/* USART Receiver Holding Register */
typedef struct {
  __REG32  RXCHR                 : 9;
  __REG32                        :23;
} __us_rhr_bits;

/* USART Transmit Holding Register */
typedef struct {
  __REG32  TXCHR                 : 9;
  __REG32                        :23;
} __us_thr_bits;

/* USART Baud Rate Generator Register */
typedef struct {
  __REG32  CD                    :16;
  __REG32                        :16;
} __us_brgr_bits;

/* USART Receiver Time-Out Register */
typedef struct {
  __REG32  TO                    :16;
  __REG32                        :16;
} __us_rtor_bits;

/* USART Transmit Time-Guard Register */
typedef struct {
  __REG32  TG                    : 8;
  __REG32                        :24;
} __us_ttgr_bits;

/* USART LIN Identifier Register */
typedef struct {
  __REG32  IDENTIFIER            : 6;
  __REG32  NDATA                 : 3;
  __REG32  CHK_SEL               : 1;
  __REG32                        : 6;
  __REG32  WAKE_UP_TIME          :14;
  __REG32                        : 2;
} __us_lir_bits;

/* USART Data Field Write 0 Register */
/* USART Data Field Read 0 Register */
typedef struct {
  __REG32  DATA0                 : 8;
  __REG32  DATA1                 : 8;
  __REG32  DATA2                 : 8;
  __REG32  DATA3                 : 8;
} __us_dfwr0_bits;

/* USART Data Field Write 1 Register */
/* USART Data Field Read 1 Register */
typedef struct {
  __REG32  DATA4                 : 8;
  __REG32  DATA5                 : 8;
  __REG32  DATA6                 : 8;
  __REG32  DATA7                 : 8;
} __us_dfwr1_bits;

/* USART Synchronous Break Length Register */
typedef struct {
  __REG32  SYNC_BRK              : 5;
  __REG32                        :27;
} __us_sblr_bits;

/* USART Synchronous Break Length Register 1 */
typedef struct {
  __REG32  LCP0                  : 8;
  __REG32  LCP1                  : 8;
  __REG32  LCP2                  : 8;
  __REG32  LCP3                  : 8;
} __us_lcp1_bits;

/* USART Synchronous Break Length Register 2 */
typedef struct {
  __REG32  LCP4                  : 8;
  __REG32  LCP5                  : 8;
  __REG32  LCP6                  : 8;
  __REG32  LCP7                  : 8;
} __us_lcp2_bits;

/* USART DMA Control Register */
typedef struct {
  __REG32  RXDMAE                : 1;
  __REG32  TXDMAE                : 1;
  __REG32                        :30;
} __us_dmacr_bits;

/* USB function address register */
typedef struct {
  __REG32  USBFAF                : 7;
  __REG32  USBAUP                : 1;
  __REG32                        :24;
} __usbfa_bits;

/* USB power management register */
typedef struct {
  __REG32  SUSE                  : 1;
  __REG32  SUSM                  : 1;
  __REG32  RU                    : 1;
  __REG32  USB_RST               : 1;
  __REG32                        : 3;
  __REG32  USB_ISO               : 1;
  __REG32                        :24;
} __usbpm_bits;

/* USB interrupt register */
typedef struct {
  __REG32  EP0I                  : 1;
  __REG32  EP1I                  : 1;
  __REG32  EP2I                  : 1;
  __REG32  EP3I                  : 1;
  __REG32  EP4I                  : 1;
  __REG32                        : 3;
  __REG32  SUSI                  : 1;
  __REG32  RESI                  : 1;
  __REG32  RSTI                  : 1;
  __REG32                        :21;
} __usbintmon_bits;

/* USB interrupt control register */
typedef struct {
  __REG32  EP0IEN                : 1;
  __REG32  EP1IEN                : 1;
  __REG32  EP2IEN                : 1;
  __REG32  EP3IEN                : 1;
  __REG32  EP4IEN                : 1;
  __REG32                        : 3;
  __REG32  SUSIEN                : 1;
  __REG32                        : 1;
  __REG32  RSTIEN                : 1;
  __REG32                        :21;
} __usbintcon_bits;

/* USB Frame Number register */
typedef struct {
  __REG32  FN                    :11;
  __REG32                        :21;
} __usbfn_bits;

/* USB endpoint logical number register */
typedef struct {
  __REG32  LNUMEP1               : 4;
  __REG32  LNUMEP2               : 4;
  __REG32  LNUMEP3               : 4;
  __REG32  LNUMEP4               : 4;
  __REG32                        :16;
} __usbeplnum_bits;

/* USB Endpoint 0 Common Status Register */
typedef struct {
  __REG32  MAXP                  : 2;
  __REG32                        : 5;
  __REG32  MAXPSET               : 1;
  __REG32                        :16;
  __REG32  ORDY                  : 1;
  __REG32  INRDY                 : 1;
  __REG32  STSTALL               : 1;
  __REG32  DEND                  : 1;
  __REG32  SETEND                : 1;
  __REG32  SDSTALL               : 1;
  __REG32  SVORDY                : 1;
  __REG32  SVSET                 : 1;
} __usbep0csr_bits;

/* USB Endpoint n Common Status Register */
typedef struct {
  __REG32  MAXP                  : 4;
  __REG32                        : 3;
  __REG32  MAXPSET               : 1;
  __REG32  OISO                  : 1;
  __REG32  OATCLR                : 1;
  __REG32                        : 1;
  __REG32  DMA_IN_PKT            : 1;
  __REG32  DMA_MODE              : 1;
  __REG32  MODE                  : 1;
  __REG32  IISO                  : 1;
  __REG32  IATSET                : 1;
  __REG32  OORDY                 : 1;
  __REG32  OFFULL                : 1;
  __REG32  OOVER                 : 1;
  __REG32  ODERR                 : 1;
  __REG32  OFFLUSH               : 1;
  __REG32  OSDSTALL              : 1;
  __REG32  OSTSTALL              : 1;
  __REG32  OCLTOG                : 1;
  __REG32  IINRDY                : 1;
  __REG32  INEMP                 : 1;
  __REG32  IUNDER                : 1;
  __REG32  IFFLUSH               : 1;
  __REG32  ISDSTALL              : 1;
  __REG32  ISTSTALL              : 1;
  __REG32  ICLTOG                : 1;
  __REG32                        : 1;
} __usbepcsr_bits;

/* USB Write Count for Endpoint 0 Register */
typedef struct {
  __REG32  WRTCNT                : 5;
  __REG32                        :27;
} __usbep0wc_bits;

/* USB Write Count for Endpoint n Register */
typedef struct {
  __REG32  WRTCNT0               : 8;
  __REG32                        : 8;
  __REG32  WRTCNT1               : 8;
  __REG32                        : 8;
} __usbepwc_bits;

/* USB NAK Control 1 */
typedef struct {
  __REG32  NAKEP6                : 4;
  __REG32  NAKEP5                : 4;
  __REG32  NAKEP4                : 4;
  __REG32  NAKEP3                : 4;
  __REG32  NAKEP2                : 4;
  __REG32  NAKEP1                : 4;
  __REG32                        : 7;
  __REG32  NAK_ENA               : 1;
} __usbnakcon1_bits;

/* USB NAK Control 2 */
typedef struct {
  __REG32  NAKEP12               : 4;
  __REG32  NAKEP11               : 4;
  __REG32  NAKEP10               : 4;
  __REG32  NAKEP9                : 4;
  __REG32  NAKEP7                : 4;
  __REG32  NAKEP6                : 4;
  __REG32                        : 7;
  __REG32  NAK_ENA               : 1;
} __usbnakcon2_bits;

/* USB EPn FIFO */
typedef struct {
  __REG32  EPFIFO                : 8;
  __REG32                        :24;
} __usbep_bits;

/* USB configuration register */
typedef struct {
  __REG32  DN                    : 1;
  __REG32  DP                    : 1;
  __REG32  DD                    : 1;
  __REG32  CIO                   : 1;
  __REG32                        : 1;
  __REG32  SUSC                  : 1;
  __REG32  WKP                   : 1;
  __REG32  TCLK                  : 1;
  __REG32  NAKC                  : 1;
  __REG32  SOFINT                : 2;
  __REG32                        :21;
} __progreg_bits;

/* USB FS Pull-up control register */
typedef struct {
  __REG32  FSPU                  : 1;
  __REG32                        :31;
} __fspullup_bits;

/* WDT ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __wdt_idr_bits;

/* WDT Control Register */
typedef struct {
  __REG32  RSTKEY                :16;
  __REG32                        :15;
  __REG32  DBGEN                 : 1;
} __wdt_cr_bits;

/* WDT Mode Register */
typedef struct {
  __REG32  WDTPDIV               : 3;
  __REG32                        : 5;
  __REG32  PCV                   :16;
  __REG32  CKEY                  : 8;
} __wdt_mr_bits;

/* WDT Overflow Mode Register */
typedef struct {
  __REG32  WDTEN                 : 1;
  __REG32  RSTEN                 : 1;
  __REG32  LOCKRSTEN             : 1;
  __REG32                        : 1;
  __REG32  OKEY                  :12;
  __REG32                        :16;
} __wdt_omr_bits;

/* WDT Status Register */
typedef struct {
  __REG32                        : 8;
  __REG32  PENDING               : 1;
  __REG32  CLEAR_STATUS          : 1;
  __REG32                        :22;
} __wdt_sr_bits;

/* WDT Interrupt Mask Set and Clear Register */
/* WDT Interrupt Raw Interrupt Status Register */
/* WDT Interrupt Masked Interrupt Status Register */
/* WDT Interrupt Clear Register */
typedef struct {
  __REG32  WDTPEND               : 1;
  __REG32  WDTOVF                : 1;
  __REG32                        :30;
} __wdt_imscr_bits;

/* WDT Pending Windows Register */
typedef struct {
  __REG32  RSTALW                : 1;
  __REG32                        : 7;
  __REG32  PWL                   :16;
  __REG32  PWKEY                 : 8;
} __wdt_pwr_bits;

/* WDT Counter Test Register */
typedef struct {
  __REG32  COUNT                 :16;
  __REG32                        :16;
} __wdt_ctr_bits;

/* Watch Timer ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __wt_idr_bits;

/* Watch Timer Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :31;
} __wt_cedr_bits;

/* Watch Timer Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __wt_srr_bits;

/* Watch Timer Control Register */
typedef struct {
  __REG32  WTEN                  : 1;
  __REG32                        : 3;
  __REG32  BCS                   : 2;
  __REG32  ITS                   : 2;
  __REG32                        :24;
} __wt_cr_bits;

/* Watch Timer Interrupt Mask Set and Clear Register */
/* Watch Timer Interrupt Raw Interrupt Status Register */
/* WDT Interrupt Masked Interrupt Status Register */
/* Watch Timer Interrupt Clear Register */
typedef struct {
  __REG32  OVF                   : 1;
  __REG32                        :31;
} __wt_imscr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSTICKCSR,      0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,      0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,      0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,    0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(ISER,            0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(ICER,            0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(ISPR,            0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(ICPR,            0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(IP0,             0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,             0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,             0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,             0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,             0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,             0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,             0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,             0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(CPUID,           0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,            0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(AIRCR,           0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,             0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,             0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR2,           0xE000ED1C,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHPR3,           0xE000ED20,__READ_WRITE ,__shpr3_bits);

/***************************************************************************
 **
 **  ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_IDR,         0x40040000,__READ       ,__adc_idr_bits);
__IO_REG32_BIT(ADC_SRR,         0x40040008,__WRITE      ,__adc_srr_bits);
__IO_REG32_BIT(ADC_CSR,         0x4004000C,__WRITE      ,__adc_csr_bits);
__IO_REG32_BIT(ADC_CCR,         0x40040010,__WRITE      ,__adc_ccr_bits);
__IO_REG32_BIT(ADC_CDR,         0x40040014,__READ_WRITE ,__adc_cdr_bits);
__IO_REG32_BIT(ADC_MR,          0x40040018,__READ_WRITE ,__adc_mr_bits);
__IO_REG32_BIT(ADC_SR,          0x4004001C,__READ       ,__adc_sr_bits);
__IO_REG32_BIT(ADC_IMSCR,       0x40040020,__READ_WRITE ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_RISR,        0x40040024,__READ       ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_MISR,        0x40040028,__READ       ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_ICR,         0x4004002C,__WRITE      ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_CRR,         0x40040030,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_DMACR,       0x4004003C,__READ_WRITE ,__adc_dmacr_bits);

/***************************************************************************
 **
 **  System
 **
 ***************************************************************************/
__IO_REG32_BIT(CM_IDR,          0x40020000,__READ       ,__cm_idr_bits);
__IO_REG32_BIT(CM_SRR,          0x40020004,__WRITE      ,__cm_srr_bits);
__IO_REG32_BIT(CM_CSR,          0x40020008,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_CCR,          0x4002000C,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_PCSR0,        0x40020010,__WRITE      ,__cm_pcsr_bits);
__IO_REG32_BIT(CM_PCCR0,        0x40020018,__WRITE      ,__cm_pcsr_bits);
__IO_REG32_BIT(CM_PCKSR0,       0x40020020,__READ       ,__cm_pcsr_bits);
__IO_REG32_BIT(CM_MR0,          0x40020028,__READ_WRITE ,__cm_mr0_bits);
__IO_REG32_BIT(CM_MR1,          0x4002002C,__READ_WRITE ,__cm_mr1_bits);
__IO_REG32_BIT(CM_IMSCR,        0x40020030,__WRITE      ,__cm_imscr_bits);
__IO_REG32_BIT(CM_RISR,         0x40020034,__READ       ,__cm_risr_bits);
__IO_REG32_BIT(CM_MISR,         0x40020038,__READ       ,__cm_imscr_bits);
__IO_REG32_BIT(CM_ICR,          0x4002003C,__WRITE      ,__cm_risr_bits);
__IO_REG32_BIT(CM_SR,           0x40020040,__READ_WRITE ,__cm_sr_bits);
__IO_REG32_BIT(CM_SCDR,         0x40020044,__READ_WRITE ,__cm_scdr_bits);
__IO_REG32_BIT(CM_PCDR,         0x40020048,__READ_WRITE ,__cm_pcdr_bits);
__IO_REG32_BIT(CM_FCDR,         0x4002004C,__READ_WRITE ,__cm_fcdr_bits);
__IO_REG32_BIT(CM_LCDR,         0x40020054,__READ_WRITE ,__cm_lcdr_bits);
__IO_REG32_BIT(CM_PSTR,         0x40020058,__READ_WRITE ,__cm_pstr_bits);
__IO_REG32_BIT(CM_PDPR,         0x4002005C,__READ_WRITE ,__cm_pdpr_bits);
__IO_REG32_BIT(CM_UPSTR,        0x40020060,__READ_WRITE ,__cm_upstr_bits);
__IO_REG32_BIT(CM_UPDPR,        0x40020064,__READ_WRITE ,__cm_updpr_bits);
__IO_REG32_BIT(CM_EMSTR,        0x40020068,__READ_WRITE ,__cm_emstr_bits);
__IO_REG32_BIT(CM_ESSTR,        0x4002006C,__READ_WRITE ,__cm_esstr_bits);
__IO_REG32_BIT(CM_BTCDR,        0x40020070,__READ_WRITE ,__cm_btcdr_bits);
__IO_REG32_BIT(CM_BTR,          0x40020074,__READ_WRITE ,__cm_btr_bits);
__IO_REG32_BIT(CM_WCR0,         0x40020078,__READ_WRITE ,__cm_wcr0_bits);
__IO_REG32_BIT(CM_WCR1,         0x4002007C,__READ_WRITE ,__cm_wcr1_bits);
__IO_REG32_BIT(CM_WIMSCR,       0x40020088,__READ_WRITE ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_WRISR,        0x4002008C,__READ       ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_WMISR,        0x40020090,__READ       ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_WICR,         0x40020094,__WRITE      ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_NISR,         0x40020098,__READ_WRITE ,__cm_nisr_bits);
__IO_REG32_BIT(CM_PSR,          0x400200A4,__READ       ,__cm_psr_bits);

/***************************************************************************
 **
 **  DMA
 **
 ***************************************************************************/
__IO_REG32(    DMA_ISR0,        0x40100000,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR0,       0x40100004,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR0,        0x40100008,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR0,       0x4010000C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR0,         0x40100010,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR0,         0x40100014,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR0,        0x40100018,__READ       );
__IO_REG32(    DMA_CDR0,        0x4010001C,__READ       );
__IO_REG32_BIT(DMA_MTR0,        0x40100020,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR0,        0x40100024,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR1,        0x40100080,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR1,       0x40100084,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR1,        0x40100088,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR1,       0x4010008C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR1,         0x40100090,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR1,         0x40100094,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR1,        0x40100098,__READ       );
__IO_REG32(    DMA_CDR1,        0x4010009C,__READ       );
__IO_REG32_BIT(DMA_MTR1,        0x401000A0,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR1,        0x401000A4,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR2,        0x40100100,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR2,       0x40100104,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR2,        0x40100108,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR2,       0x4010010C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR2,         0x40100110,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR2,         0x40100114,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR2,        0x40100118,__READ       );
__IO_REG32(    DMA_CDR2,        0x4010011C,__READ       );
__IO_REG32_BIT(DMA_MTR2,        0x40100120,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR2,        0x40100124,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR3,        0x40100180,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR3,       0x40100184,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR3,        0x40100188,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR3,       0x4010018C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR3,         0x40100190,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR3,         0x40100194,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR3,        0x40100198,__READ       );
__IO_REG32(    DMA_CDR3,        0x4010019C,__READ       );
__IO_REG32_BIT(DMA_MTR3,        0x401001A0,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR3,        0x401001A4,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR4,        0x40100200,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR4,       0x40100204,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR4,        0x40100208,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR4,       0x4010020C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR4,         0x40100210,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR4,         0x40100214,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR4,        0x40100218,__READ       );
__IO_REG32(    DMA_CDR4,        0x4010021C,__READ       );
__IO_REG32_BIT(DMA_MTR4,        0x40100220,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR4,        0x40100224,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR5,        0x40100280,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR5,       0x40100284,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR5,        0x40100288,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR5,       0x4010028C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR5,         0x40100290,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR5,         0x40100294,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR5,        0x40100298,__READ       );
__IO_REG32(    DMA_CDR5,        0x4010029C,__READ       );
__IO_REG32_BIT(DMA_MTR5,        0x401002A0,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR5,        0x401002A4,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32_BIT(DMA_IDR,         0x40100500,__READ       ,__dma_idr_bits);
__IO_REG32_BIT(DMA_SRR,         0x40100504,__WRITE      ,__dma_srr_bits);
__IO_REG32_BIT(DMA_CESR,        0x40100508,__READ       ,__dma_cesr_bits);
__IO_REG32_BIT(DMA_ISR,         0x4010050C,__READ       ,__dma_isr_bits);
__IO_REG32_BIT(DMA_ICR,         0x40100510,__WRITE      ,__dma_icr_bits);

/***************************************************************************
 **
 **  Free Running Timer
 **
 ***************************************************************************/
__IO_REG32_BIT(FRT_IDR,         0x40031000,__READ       ,__frt_idr_bits);
__IO_REG32_BIT(FRT_CEDR,        0x40031004,__READ_WRITE ,__frt_cedr_bits);
__IO_REG32_BIT(FRT_SRR,         0x40031008,__WRITE      ,__frt_srr_bits);
__IO_REG32_BIT(FRT_CR,          0x4003100C,__READ_WRITE ,__frt_cr_bits);
__IO_REG32_BIT(FRT_SR,          0x40031010,__READ       ,__frt_srr_bits);
__IO_REG32_BIT(FRT_IMSCR,       0x40031014,__READ_WRITE ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_RISR,        0x40031018,__READ       ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_MISR,        0x4003101C,__READ       ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_ICR,         0x40031020,__WRITE      ,__frt_imscr_bits);
__IO_REG32(    FRT_DR,          0x40031024,__READ_WRITE );
__IO_REG32(    FRT_DBR,         0x40031028,__READ       );
__IO_REG32(    FRT_CVR,         0x4003102C,__READ       );

/***************************************************************************
 **
 **  Port 0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO0_IDR,       0x40050000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO0_CEDR,      0x40050004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO0_SRR,       0x40050008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO0_IMSCR,     0x4005000C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_RISR,      0x40050010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_MISR,      0x40050014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_ICR,       0x40050018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_OER,       0x4005001C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_ODR,       0x40050020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_OSR,       0x40050024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_WODR,      0x40050028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_SODR,      0x4005002C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_CODR,      0x40050030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_ODSR,      0x40050034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_PDSR,      0x40050038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_IDR,       0x40051000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO1_CEDR,      0x40051004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO1_SRR,       0x40051008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO1_IMSCR,     0x4005100C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_RISR,      0x40051010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_MISR,      0x40051014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_ICR,       0x40051018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_OER,       0x4005101C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_ODR,       0x40051020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_OSR,       0x40051024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_WODR,      0x40051028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_SODR,      0x4005102C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_CODR,      0x40051030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_ODSR,      0x40051034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_PDSR,      0x40051038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_IDR,       0x40052000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO2_CEDR,      0x40052004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO2_SRR,       0x40052008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO2_IMSCR,     0x4005200C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_RISR,      0x40052010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_MISR,      0x40052014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_ICR,       0x40052018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_OER,       0x4005201C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_ODR,       0x40052020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_OSR,       0x40052024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_WODR,      0x40052028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_SODR,      0x4005202C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_CODR,      0x40052030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_ODSR,      0x40052034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO2_PDSR,      0x40052038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_IDR,       0x40053000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO3_CEDR,      0x40053004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO3_SRR,       0x40053008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO3_IMSCR,     0x4005300C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_RISR,      0x40053010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_MISR,      0x40053014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_ICR,       0x40053018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_OER,       0x4005301C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_ODR,       0x40053020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_OSR,       0x40053024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_WODR,      0x40053028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_SODR,      0x4005302C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_CODR,      0x40053030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_ODSR,      0x40053034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO3_PDSR,      0x40053038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO4_IDR,       0x40054000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO4_CEDR,      0x40054004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO4_SRR,       0x40054008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO4_IMSCR,     0x4005400C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_RISR,      0x40054010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_MISR,      0x40054014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_ICR,       0x40054018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_OER,       0x4005401C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_ODR,       0x40054020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_OSR,       0x40054024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_WODR,      0x40054028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_SODR,      0x4005402C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_CODR,      0x40054030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_ODSR,      0x40054034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO4_PDSR,      0x40054038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 5
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO5_IDR,       0x40055000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO5_CEDR,      0x40055004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO5_SRR,       0x40055008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO5_IMSCR,     0x4005500C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_RISR,      0x40055010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_MISR,      0x40055014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_ICR,       0x40055018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_OER,       0x4005501C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_ODR,       0x40055020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_OSR,       0x40055024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_WODR,      0x40055028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_SODR,      0x4005502C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_CODR,      0x40055030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_ODSR,      0x40055034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO5_PDSR,      0x40055038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_IDR,        0x400A0000,__READ       ,__i2c_idr_bits);
__IO_REG32_BIT(I2C0_CEDR,       0x400A0004,__READ_WRITE ,__i2c_cedr_bits);
__IO_REG32_BIT(I2C0_SRR,        0x400A0008,__WRITE      ,__i2c_srr_bits);
__IO_REG32_BIT(I2C0_CR,         0x400A000C,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C0_MR,         0x400A0010,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C0_SR,         0x400A0014,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C0_IMSCR,      0x400A0018,__READ_WRITE ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_RISR,       0x400A001C,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_MISR,       0x400A0020,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_ICR,        0x400A0024,__WRITE      ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_SDR,        0x400A0028,__READ_WRITE ,__i2c_sdr_bits);
__IO_REG32_BIT(I2C0_SSAR,       0x400A002C,__READ_WRITE ,__i2c_ssar_bits);
__IO_REG32_BIT(I2C0_HSDR,       0x400A0030,__READ_WRITE ,__i2c_hsdr_bits);
__IO_REG32_BIT(I2C0_DMACR,      0x400A0034,__READ_WRITE ,__i2c_dmacr_bits);

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_IDR,        0x400A1000,__READ       ,__i2c_idr_bits);
__IO_REG32_BIT(I2C1_CEDR,       0x400A1004,__READ_WRITE ,__i2c_cedr_bits);
__IO_REG32_BIT(I2C1_SRR,        0x400A1008,__WRITE      ,__i2c_srr_bits);
__IO_REG32_BIT(I2C1_CR,         0x400A100C,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C1_MR,         0x400A1010,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C1_SR,         0x400A1014,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C1_IMSCR,      0x400A1018,__READ_WRITE ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_RISR,       0x400A101C,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_MISR,       0x400A1020,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_ICR,        0x400A1024,__WRITE      ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_SDR,        0x400A1028,__READ_WRITE ,__i2c_sdr_bits);
__IO_REG32_BIT(I2C1_SSAR,       0x400A102C,__READ_WRITE ,__i2c_ssar_bits);
__IO_REG32_BIT(I2C1_HSDR,       0x400A1030,__READ_WRITE ,__i2c_hsdr_bits);
__IO_REG32_BIT(I2C1_DMACR,      0x400A1034,__READ_WRITE ,__i2c_dmacr_bits);

/***************************************************************************
 **
 **  FLASH
 **
 ***************************************************************************/
__IO_REG32_BIT(IFC_IDR,         0x40010000,__READ       ,__ipf_idr_bits);
__IO_REG32_BIT(IFC_CEDR,        0x40010004,__READ_WRITE ,__ipf_cedr_bits);
__IO_REG32_BIT(IFC_SRR,         0x40010008,__WRITE      ,__ipf_srr_bits);
__IO_REG32_BIT(IFC_CR,          0x4001000C,__READ_WRITE ,__ipf_cr_bits);
__IO_REG32_BIT(IFC_MR,          0x40010010,__READ_WRITE ,__ipf_mr_bits);
__IO_REG32_BIT(IFC_IMSCR,       0x40010014,__READ_WRITE ,__ipf_imscr_bits);
__IO_REG32_BIT(IFC_RISR,        0x40010018,__READ       ,__ipf_imscr_bits);
__IO_REG32_BIT(IFC_MISR,        0x4001001C,__READ       ,__ipf_imscr_bits);
__IO_REG32_BIT(IFC_ICR,         0x40010020,__WRITE      ,__ipf_imscr_bits);
__IO_REG32_BIT(IFC_SR,          0x40010024,__READ       ,__ipf_sr_bits);
__IO_REG32(    IFC_AR,          0x40010028,__READ_WRITE );
__IO_REG32(    IFC_DR,          0x4001002C,__READ_WRITE );
__IO_REG32(    IFC_KR,          0x40010030,__WRITE      );
__IO_REG32_BIT(SO_PSR,          0x40010034,__READ       ,__so_psr_bits);
__IO_REG32_BIT(SO_CSR,          0x40010038,__READ       ,__so_csr_bits);
__IO_REG32_BIT(PF_IOTR,         0x4001003C,__READ_WRITE ,__pf_iotr_bits);

/***************************************************************************
 **
 **  IOCONF
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCONF_MLR0,     0x40058000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR0,     0x40058004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR0,    0x40058008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR0,    0x4005800C,__READ_WRITE ,__ioconf_odcr_bits);
__IO_REG32_BIT(IOCONF_MLR1,     0x40059000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR1,     0x40059004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR1,    0x40059008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR1,    0x4005900C,__READ_WRITE ,__ioconf_odcr_bits);
__IO_REG32_BIT(IOCONF_MLR2,     0x4005A000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR2,     0x4005A004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR2,    0x4005A008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR2,    0x4005A00C,__READ_WRITE ,__ioconf_odcr_bits);
__IO_REG32_BIT(IOCONF_MLR3,     0x4005B000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR3,     0x4005B004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR3,    0x4005B008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR3,    0x4005B00C,__READ_WRITE ,__ioconf_odcr_bits);
__IO_REG32_BIT(IOCONF_MLR4,     0x4005C000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR4,     0x4005C004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR4,    0x4005C008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR4,    0x4005C00C,__READ_WRITE ,__ioconf_odcr_bits);
__IO_REG32_BIT(IOCONF_MLR5,     0x4005D000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR5,     0x4005D004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR5,    0x4005D008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR5,    0x4005D00C,__READ_WRITE ,__ioconf_odcr_bits);

/***************************************************************************
 **
 **  LCD
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_IDR,         0x400D0000,__READ       ,__lcd_idr_bits);
__IO_REG32_BIT(LCD_CEDR,        0x400D0004,__READ_WRITE ,__lcd_cedr_bits);
__IO_REG32_BIT(LCD_SRR,         0x400D0008,__WRITE      ,__lcd_srr_bits);
__IO_REG32_BIT(LCD_CR,          0x400D000C,__READ_WRITE ,__lcd_cr_bits);
__IO_REG32_BIT(LCD_CDR,         0x400D0010,__READ_WRITE ,__lcd_cdr_bits);
__IO_REG32_BIT(LCD_CCR,         0x400D0014,__READ_WRITE ,__lcd_ccr_bits);
__IO_REG32(    LCD_DMR_BASE,    0x400D0400,__READ_WRITE );

/***************************************************************************
 **
 **  PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0_IDR,        0x40070000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM0_CEDR,       0x40070004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM0_SRR,        0x40070008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM0_CSR,        0x4007000C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM0_CCR,        0x40070010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM0_SR,         0x40070014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM0_IMSCR,      0x40070018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_RISR,       0x4007001C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_MISR,       0x40070020,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_ICR,        0x40070024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_CDR,        0x40070028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM0_PRDR,       0x4007002C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM0_PULR,       0x40070030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM0_CCDR,       0x40070034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM0_CPRDR,      0x40070038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM0_CPULR,      0x4007003C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_IDR,        0x40071000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM1_CEDR,       0x40071004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM1_SRR,        0x40071008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM1_CSR,        0x4007100C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM1_CCR,        0x40071010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM1_SR,         0x40071014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM1_IMSCR,      0x40071018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_RISR,       0x4007101C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_MISR,       0x40071020,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_ICR,        0x40071024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_CDR,        0x40071028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM1_PRDR,       0x4007102C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM1_PULR,       0x40071030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM1_CCDR,       0x40071034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM1_CPRDR,      0x40071038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM1_CPULR,      0x4007103C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CR0,        0x40090000,__READ_WRITE ,__spi_cr0_bits);
__IO_REG32_BIT(SPI0_CR1,        0x40090004,__READ_WRITE ,__spi_cr1_bits);
__IO_REG16(    SPI0_DR,         0x40090008,__READ_WRITE );
__IO_REG32_BIT(SPI0_SR,         0x4009000C,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI0_CPSR,       0x40090010,__READ_WRITE ,__spi_cpsr_bits);
__IO_REG32_BIT(SPI0_IMSC,       0x40090014,__READ_WRITE ,__spi_imsc_bits);
__IO_REG32_BIT(SPI0_RISR,       0x40090018,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI0_MISR,       0x4009001C,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI0_ICR,        0x40090020,__WRITE      ,__spi_icr_bits);
__IO_REG32_BIT(SPI0_DMACR,      0x40090024,__READ_WRITE ,__spi_dmacr_bits);

/***************************************************************************
 **
 **  SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CR0,        0x40091000,__READ_WRITE ,__spi_cr0_bits);
__IO_REG32_BIT(SPI1_CR1,        0x40091004,__READ_WRITE ,__spi_cr1_bits);
__IO_REG16(    SPI1_DR,         0x40091008,__READ_WRITE );
__IO_REG32_BIT(SPI1_SR,         0x4009100C,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI1_CPSR,       0x40091010,__READ_WRITE ,__spi_cpsr_bits);
__IO_REG32_BIT(SPI1_IMSC,       0x40091014,__READ_WRITE ,__spi_imsc_bits);
__IO_REG32_BIT(SPI1_RISR,       0x40091018,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI1_MISR,       0x4009101C,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI1_ICR,        0x40091020,__WRITE      ,__spi_icr_bits);
__IO_REG32_BIT(SPI1_DMACR,      0x40091024,__READ_WRITE ,__spi_dmacr_bits);

/***************************************************************************
 **
 **  TC0
 **
 ***************************************************************************/
__IO_REG32_BIT(TC0_IDR,         0x40060000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC0_CSSR,        0x40060004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC0_CEDR,        0x40060008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC0_SRR,         0x4006000C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC0_CSR,         0x40060010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC0_CCR,         0x40060014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC0_SR,          0x40060018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC0_IMSCR,       0x4006001C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_RISR,        0x40060020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_MISR,        0x40060024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_ICR,         0x40060028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_CDR,         0x4006002C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC0_CSMR,        0x40060030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC0_PRDR,        0x40060034,__READ_WRITE );
__IO_REG16(    TC0_PULR,        0x40060038,__READ_WRITE );
__IO_REG32_BIT(TC0_CCDR,        0x4006003C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC0_CCSMR,       0x40060040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC0_CPRDR,       0x40060044,__READ       );
__IO_REG16(    TC0_CPULR,       0x40060048,__READ       );
__IO_REG16(    TC0_CUCR,        0x4006004C,__READ       );
__IO_REG16(    TC0_CDCR,        0x40060050,__READ       );
__IO_REG16(    TC0_CVR,         0x40060054,__READ       );

/***************************************************************************
 **
 **  TC1
 **
 ***************************************************************************/
__IO_REG32_BIT(TC1_IDR,         0x40061000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC1_CSSR,        0x40061004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC1_CEDR,        0x40061008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC1_SRR,         0x4006100C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC1_CSR,         0x40061010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC1_CCR,         0x40061014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC1_SR,          0x40061018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC1_IMSCR,       0x4006101C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_RISR,        0x40061020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_MISR,        0x40061024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_ICR,         0x40061028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_CDR,         0x4006102C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC1_CSMR,        0x40061030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC1_PRDR,        0x40061034,__READ_WRITE );
__IO_REG16(    TC1_PULR,        0x40061038,__READ_WRITE );
__IO_REG32_BIT(TC1_CCDR,        0x4006103C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC1_CCSMR,       0x40061040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC1_CPRDR,       0x40061044,__READ       );
__IO_REG16(    TC1_CPULR,       0x40061048,__READ       );
__IO_REG16(    TC1_CUCR,        0x4006104C,__READ       );
__IO_REG16(    TC1_CDCR,        0x40061050,__READ       );
__IO_REG16(    TC1_CVR,         0x40061054,__READ       );

/***************************************************************************
 **
 **  TC2
 **
 ***************************************************************************/
__IO_REG32_BIT(TC2_IDR,         0x40062000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC2_CSSR,        0x40062004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC2_CEDR,        0x40062008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC2_SRR,         0x4006200C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC2_CSR,         0x40062010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC2_CCR,         0x40062014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC2_SR,          0x40062018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC2_IMSCR,       0x4006201C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_RISR,        0x40062020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_MISR,        0x40062024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_ICR,         0x40062028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_CDR,         0x4006202C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC2_CSMR,        0x40062030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC2_PRDR,        0x40062034,__READ_WRITE );
__IO_REG16(    TC2_PULR,        0x40062038,__READ_WRITE );
__IO_REG32_BIT(TC2_CCDR,        0x4006203C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC2_CCSMR,       0x40062040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC2_CPRDR,       0x40062044,__READ       );
__IO_REG16(    TC2_CPULR,       0x40062048,__READ       );
__IO_REG16(    TC2_CUCR,        0x4006204C,__READ       );
__IO_REG16(    TC2_CDCR,        0x40062050,__READ       );
__IO_REG16(    TC2_CVR,         0x40062054,__READ       );

/***************************************************************************
 **
 **  TC3
 **
 ***************************************************************************/
__IO_REG32_BIT(TC3_IDR,         0x40063000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC3_CSSR,        0x40063004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC3_CEDR,        0x40063008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC3_SRR,         0x4006300C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC3_CSR,         0x40063010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC3_CCR,         0x40063014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC3_SR,          0x40063018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC3_IMSCR,       0x4006301C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_RISR,        0x40063020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_MISR,        0x40063024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_ICR,         0x40063028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_CDR,         0x4006302C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC3_CSMR,        0x40063030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC3_PRDR,        0x40063034,__READ_WRITE );
__IO_REG16(    TC3_PULR,        0x40063038,__READ_WRITE );
__IO_REG32_BIT(TC3_CCDR,        0x4006303C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC3_CCSMR,       0x40063040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC3_CPRDR,       0x40063044,__READ       );
__IO_REG16(    TC3_CPULR,       0x40063048,__READ       );
__IO_REG16(    TC3_CUCR,        0x4006304C,__READ       );
__IO_REG16(    TC3_CDCR,        0x40063050,__READ       );
__IO_REG16(    TC3_CVR,         0x40063054,__READ       );

/***************************************************************************
 **
 **  TC4
 **
 ***************************************************************************/
__IO_REG32_BIT(TC4_IDR,         0x40064000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC4_CSSR,        0x40064004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC4_CEDR,        0x40064008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC4_SRR,         0x4006400C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC4_CSR,         0x40064010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC4_CCR,         0x40064014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC4_SR,          0x40064018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC4_IMSCR,       0x4006401C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_RISR,        0x40064020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_MISR,        0x40064024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_ICR,         0x40064028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_CDR,         0x4006402C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC4_CSMR,        0x40064030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC4_PRDR,        0x40064034,__READ_WRITE );
__IO_REG16(    TC4_PULR,        0x40064038,__READ_WRITE );
__IO_REG32_BIT(TC4_CCDR,        0x4006403C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC4_CCSMR,       0x40064040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC4_CPRDR,       0x40064044,__READ       );
__IO_REG16(    TC4_CPULR,       0x40064048,__READ       );
__IO_REG16(    TC4_CUCR,        0x4006404C,__READ       );
__IO_REG16(    TC4_CDCR,        0x40064050,__READ       );
__IO_REG16(    TC4_CVR,         0x40064054,__READ       );

/***************************************************************************
 **
 **  TC5
 **
 ***************************************************************************/
__IO_REG32_BIT(TC5_IDR,         0x40065000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC5_CSSR,        0x40065004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC5_CEDR,        0x40065008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC5_SRR,         0x4006500C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC5_CSR,         0x40065010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC5_CCR,         0x40065014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC5_SR,          0x40065018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC5_IMSCR,       0x4006501C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_RISR,        0x40065020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_MISR,        0x40065024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_ICR,         0x40065028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_CDR,         0x4006502C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC5_CSMR,        0x40065030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC5_PRDR,        0x40065034,__READ_WRITE );
__IO_REG16(    TC5_PULR,        0x40065038,__READ_WRITE );
__IO_REG32_BIT(TC5_CCDR,        0x4006503C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC5_CCSMR,       0x40065040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC5_CPRDR,       0x40065044,__READ       );
__IO_REG16(    TC5_CPULR,       0x40065048,__READ       );
__IO_REG16(    TC5_CUCR,        0x4006504C,__READ       );
__IO_REG16(    TC5_CDCR,        0x40065050,__READ       );
__IO_REG16(    TC5_CVR,         0x40065054,__READ       );

/***************************************************************************
 **
 **  TC6
 **
 ***************************************************************************/
__IO_REG32_BIT(TC6_IDR,         0x40066000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC6_CSSR,        0x40066004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC6_CEDR,        0x40066008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC6_SRR,         0x4006600C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC6_CSR,         0x40066010,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC6_CCR,         0x40066014,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC6_SR,          0x40066018,__READ       ,__tc32_sr_bits);
__IO_REG32_BIT(TC6_IMSCR,       0x4006601C,__READ_WRITE ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_RISR,        0x40066020,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_MISR,        0x40066024,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_ICR,         0x40066028,__WRITE      ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_CDR,         0x4006602C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC6_CSMR,        0x40066030,__READ_WRITE ,__tc32_csmr_bits);
__IO_REG32(    TC6_PRDR,        0x40066034,__READ_WRITE );
__IO_REG32(    TC6_PULR,        0x40066038,__READ_WRITE );
__IO_REG32_BIT(TC6_CCDR,        0x4006603C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC6_CCSMR,       0x40066040,__READ       ,__tc32_csmr_bits);
__IO_REG32(    TC6_CPRDR,       0x40066044,__READ       );
__IO_REG32(    TC6_CPULR,       0x40066048,__READ       );
__IO_REG32(    TC6_CUCR,        0x4006604C,__READ       );
__IO_REG32(    TC6_CDCR,        0x40066050,__READ       );
__IO_REG32(    TC6_CVR,         0x40066054,__READ       );

/***************************************************************************
 **
 **  TC7
 **
 ***************************************************************************/
__IO_REG32_BIT(TC7_IDR,         0x40067000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC7_CSSR,        0x40067004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC7_CEDR,        0x40067008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC7_SRR,         0x4006700C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC7_CSR,         0x40067010,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC7_CCR,         0x40067014,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC7_SR,          0x40067018,__READ       ,__tc32_sr_bits);
__IO_REG32_BIT(TC7_IMSCR,       0x4006701C,__READ_WRITE ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_RISR,        0x40067020,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_MISR,        0x40067024,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_ICR,         0x40067028,__WRITE      ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_CDR,         0x4006702C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC7_CSMR,        0x40067030,__READ_WRITE ,__tc32_csmr_bits);
__IO_REG32(    TC7_PRDR,        0x40067034,__READ_WRITE );
__IO_REG32(    TC7_PULR,        0x40067038,__READ_WRITE );
__IO_REG32_BIT(TC7_CCDR,        0x4006703C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC7_CCSMR,       0x40067040,__READ       ,__tc32_csmr_bits);
__IO_REG32(    TC7_CPRDR,       0x40067044,__READ       );
__IO_REG32(    TC7_CPULR,       0x40067048,__READ       );
__IO_REG32(    TC7_CUCR,        0x4006704C,__READ       );
__IO_REG32(    TC7_CDCR,        0x40067050,__READ       );
__IO_REG32(    TC7_CVR,         0x40067054,__READ       );

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
__IO_REG32_BIT(US0_IDR,         0x40080000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US0_CEDR,        0x40080004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US0_SRR,         0x40080008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US0_CR,          0x4008000C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US0_MR,          0x40080010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US0_IMSCR,       0x40080014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US0_RISR,        0x40080018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US0_MISR,        0x4008001C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US0_ICR,         0x40080020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US0_SR,          0x40080024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US0_RHR,         0x40080028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US0_THR,         0x4008002C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US0_BRGR,        0x40080030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US0_RTOR,        0x40080034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US0_TTGR,        0x40080038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US0_LIR,         0x4008003C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US0_DFWR0,       0x40080040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US0_DFWR1,       0x40080044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US0_DFRR0,       0x40080048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US0_DFRR1,       0x4008004C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US0_SBLR,        0x40080050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US0_LCP1,        0x40080054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US0_LCP2,        0x40080058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US0_DMACR,       0x4008005C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(US1_IDR,         0x40081000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US1_CEDR,        0x40081004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US1_SRR,         0x40081008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US1_CR,          0x4008100C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US1_MR,          0x40081010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US1_IMSCR,       0x40081014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US1_RISR,        0x40081018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US1_MISR,        0x4008101C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US1_ICR,         0x40081020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US1_SR,          0x40081024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US1_RHR,         0x40081028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US1_THR,         0x4008102C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US1_BRGR,        0x40081030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US1_RTOR,        0x40081034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US1_TTGR,        0x40081038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US1_LIR,         0x4008103C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US1_DFWR0,       0x40081040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US1_DFWR1,       0x40081044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US1_DFRR0,       0x40081048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US1_DFRR1,       0x4008104C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US1_SBLR,        0x40081050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US1_LCP1,        0x40081054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US1_LCP2,        0x40081058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US1_DMACR,       0x4008105C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(US2_IDR,         0x40082000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US2_CEDR,        0x40082004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US2_SRR,         0x40082008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US2_CR,          0x4008200C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US2_MR,          0x40082010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US2_IMSCR,       0x40082014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US2_RISR,        0x40082018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US2_MISR,        0x4008201C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US2_ICR,         0x40082020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US2_SR,          0x40082024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US2_RHR,         0x40082028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US2_THR,         0x4008202C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US2_BRGR,        0x40082030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US2_RTOR,        0x40082034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US2_TTGR,        0x40082038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US2_LIR,         0x4008203C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US2_DFWR0,       0x40082040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US2_DFWR1,       0x40082044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US2_DFRR0,       0x40082048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US2_DFRR1,       0x4008204C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US2_SBLR,        0x40082050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US2_LCP1,        0x40082054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US2_LCP2,        0x40082058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US2_DMACR,       0x4008205C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USBFA,           0x40110000,__READ_WRITE ,__usbfa_bits);
__IO_REG32_BIT(USBPM,           0x40110004,__READ_WRITE ,__usbpm_bits);
__IO_REG32_BIT(USBINTMON,       0x40110008,__READ_WRITE ,__usbintmon_bits);
__IO_REG32_BIT(USBINTCON,       0x4011000C,__READ_WRITE ,__usbintcon_bits);
__IO_REG32_BIT(USBFN,           0x40110010,__READ       ,__usbfn_bits);
__IO_REG32_BIT(USBEPLNUM,       0x40110014,__READ_WRITE ,__usbeplnum_bits);
__IO_REG32_BIT(USBEP0CSR,       0x40110020,__READ_WRITE ,__usbep0csr_bits);
__IO_REG32_BIT(USBEP1CSR,       0x40110024,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP2CSR,       0x40110028,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP3CSR,       0x4011002C,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP4CSR,       0x40110030,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP0WC,        0x40110040,__READ_WRITE ,__usbep0wc_bits);
__IO_REG32_BIT(USBEP1WC,        0x40110044,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBEP2WC,        0x40110048,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBEP3WC,        0x4011004C,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBEP4WC,        0x40110050,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBNAKCON1,      0x40110060,__READ_WRITE ,__usbnakcon1_bits);
__IO_REG32_BIT(USBNAKCON2,      0x40110064,__READ_WRITE ,__usbnakcon2_bits);
__IO_REG32_BIT(USBEP0,          0x40110080,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP1,          0x40110084,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP2,          0x40110088,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP3,          0x4011008C,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP4,          0x40110090,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(PROGREG,         0x401100A0,__READ_WRITE ,__progreg_bits);
__IO_REG32_BIT(FSPULLUP,        0x401100B4,__READ_WRITE ,__fspullup_bits);

/***************************************************************************
 **
 **  WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_IDR,         0x40030000,__READ       ,__wdt_idr_bits);
__IO_REG32_BIT(WDT_CR,          0x40030004,__WRITE      ,__wdt_cr_bits);
__IO_REG32_BIT(WDT_MR,          0x40030008,__READ_WRITE ,__wdt_mr_bits);
__IO_REG32_BIT(WDT_OMR,         0x4003000C,__READ_WRITE ,__wdt_omr_bits);
__IO_REG32_BIT(WDT_SR,          0x40030010,__READ       ,__wdt_sr_bits);
__IO_REG32_BIT(WDT_IMSCR,       0x40030014,__READ_WRITE ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_RISR,        0x40030018,__READ       ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_MISR,        0x4003001C,__READ       ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_ICR,         0x40030020,__WRITE      ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_PWR,         0x40030024,__READ_WRITE ,__wdt_pwr_bits);
__IO_REG32_BIT(WDT_CTR,         0x40030028,__READ       ,__wdt_ctr_bits);

/***************************************************************************
 **
 **  WT
 **
 ***************************************************************************/
__IO_REG32_BIT(WT_IDR,          0x400C0000,__READ       ,__wt_idr_bits);
__IO_REG32_BIT(WT_CEDR,         0x400C0004,__READ_WRITE ,__wt_cedr_bits);
__IO_REG32_BIT(WT_SRR,          0x400C0008,__WRITE      ,__wt_srr_bits);
__IO_REG32_BIT(WT_CR,           0x400C000C,__READ_WRITE ,__wt_cr_bits);
__IO_REG32_BIT(WT_IMSCR,        0x400C0010,__READ_WRITE ,__wt_imscr_bits);
__IO_REG32_BIT(WT_RISR,         0x400C0014,__READ       ,__wt_imscr_bits);
__IO_REG32_BIT(WT_MISC,         0x400C0018,__READ       ,__wt_imscr_bits);
__IO_REG32_BIT(WT_ICR,          0x400C001C,__WRITE      ,__wt_imscr_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  S3FN21 DMA channels number
 **
 ***************************************************************************/
#define UART0_TX_DMA      0x00
#define UART0_RX_DMA      0x01
#define UART1_TX_DMA      0x02
#define UART1_RX_DMA      0x03
#define UART2_TX_DMA      0x04
#define UART2_RX_DMA      0x05
#define SPI0_TX_DMA       0x0A
#define SPI0_RX_DMA       0x0B
#define SPI1_TX_DMA       0x0C
#define SPI1_RX_DMA       0x0D
#define I2C0_TX_DMA       0x12
#define I2C0_RX_DMA       0x13
#define I2C1_TX_DMA       0x14
#define I2C1_RX_DMA       0x15
#define ADC_DMA           0x16
#define USB_EP1_DMA       0x1A
#define USB_EP2_DMA       0x1B
#define USB_EP3_DMA       0x1C
#define USB_EP4_DMA       0x1D

/***************************************************************************
 **
 **  S3FN21 interrupt source number
 **
 ***************************************************************************/
#define WDTINT            0x00    /* Watch-dog Timer Interrupt */
#define FRTINT            0x01    /* Free-running Timer Interrupt */
#define CMINT             0x02    /* Clock Manager Interrupt */
#define IFCINT            0x03    /* Internal Flash Controller Interrupt */
#define DMAINT            0x04    /* DMA Controller Interrupt */
#define ADCINT            0x05    /* ADC Interrupt */
#define WSRC0INT          0x06    /* Wakeup source 0 */
#define TC0INT            0x07    /* Timer/Counter0 Interrupt */
#define WTINT             0x08    /* Watch Timer Interrupt  */
#define TC1INT            0x0A    /* Timer/Counter1 Interrupt */
#define USART0INT         0x0b    /* USART0 Interrupt */
#define PWM0INT           0x0c    /* PWM Interrupt */
#define SPI0INT           0x0d    /* SPI Interrupt */
#define I2C0INT           0x0e    /* I2C Interrupt */
#define USART1INT         0x0f    /* USART1 Interrupt */
#define TC2INT            0x10    /* Timer/Counter2 Interrupt */
#define USART2INT         0x11    /* USART2 Interrupt */
#define TC3INT            0x12    /* Timer/Counter3 Interrupt */
#define SPI1INT           0x13    /* SPI Interrupt */
#define I2C1INT           0x14    /* I2C Interrupt */
#define PWM1INT           0x15    /* PWM Interrupt */
#define TC4INT            0x16    /* Timer/Counter4 Interrupt */
#define TC5INT            0x17    /* Timer/Counter5 Interrupt */
#define TC6INT            0x18    /* Timer/Counter6 Interrupt */
#define TC7INT            0x19    /* Timer/Counter7 Interrupt */
#define USBINT            0x1a    /* USB Interrupt */
#define GPIOINT           0x1e    /* GPIO 0~5 Interrupt */
#define WSRCxINT          0x1f    /* Wakeup source 1~15 */

#endif    /* __S3FN21_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = SVC            0x2C
Interrupt3   = PendSV         0x38
Interrupt4   = SysTick        0x3C
Interrupt5   = WDTINT         0x40
Interrupt6   = FRTINT         0x44
Interrupt7   = CMINT          0x48
Interrupt8   = IFCINT         0x4C
Interrupt9   = DMAINT         0x50
Interrupt10  = ADCINT         0x54
Interrupt11  = WSRC0INT       0x58
Interrupt12  = TC0INT         0x5C
Interrupt13  = WTINT          0x60
Interrupt14  = TC1INT         0x68
Interrupt15  = USART0INT      0x6C
Interrupt16  = PWM0INT        0x70
Interrupt17  = SPI0INT        0x74
Interrupt18  = I2C0INT        0x78
Interrupt19  = USART1INT      0x7C
Interrupt20  = TC2INT         0x80
Interrupt21  = USART2INT      0x84
Interrupt22  = TC3INT         0x88
Interrupt23  = SPI1INT        0x8C
Interrupt24  = I2C1INT        0x90
Interrupt25  = PWM1INT        0x94
Interrupt26  = TC4INT         0x98
Interrupt27  = TC5INT         0x9C
Interrupt28  = TC6INT         0xA0
Interrupt29  = TC7INT         0xA4
Interrupt30  = USBINT         0xA8
Interrupt31  = GPIOINT        0xB8
Interrupt32  = WSRCxINT       0xBC
###DDF-INTERRUPT-END###*/
