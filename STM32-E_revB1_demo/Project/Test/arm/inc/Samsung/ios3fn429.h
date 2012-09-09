/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Samsung S3FN429
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 49286 $
 **
 ***************************************************************************/

#ifndef __S3FN429_H
#define __S3FN429_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    S3FN429 SPECIAL FUNCTION REGISTERS
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

/* ADC Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :30;
  __REG32  DBGEN          : 1;
} __adc_cedr_bits;

/* ADC Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __adc_srr_bits;

/* ADC Control Set Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32  START          : 1;
  __REG32  CCSTOP         : 1;
  __REG32                 :29;
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
  __REG32  TRIG           : 3;
  __REG32                 : 4;
  __REG32  CMODE          : 1;
  __REG32  CCNT           : 4;
  __REG32                 :12;
  __REG32  CALEN          : 1;
  __REG32  ICRV           : 1;
  __REG32  EICR           : 1;
  __REG32                 : 5;
} __adc_mr_bits;

/* ADC_CCSR0 */
typedef struct {
  __REG32  ICNUM0         : 4;
  __REG32  ICNUM1         : 4;
  __REG32  ICNUM2         : 4;
  __REG32  ICNUM3         : 4;
  __REG32  ICNUM4         : 4;
  __REG32  ICNUM5         : 4;
  __REG32  ICNUM6         : 4;
  __REG32  ICNUM7         : 4;
} __adc_ccsr0_bits;

/* ADC_CCSR1 */
typedef struct {
  __REG32  ICNUM8         : 4;
  __REG32  ICNUM9         : 4;
  __REG32  ICNUM10        : 4;
  __REG32                 :20;
} __adc_ccsr1_bits;

/* ADC_SSR */
typedef struct {
  __REG32                 :24;
  __REG32  CCCV           : 4;
  __REG32                 : 4;
} __adc_ssr_bits;

/* ADC Status Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32  BUSY           : 1;
  __REG32                 : 5;
  __REG32  CMODE          : 1;
  __REG32                 : 8;
  __REG32  OVR0          	: 1;
  __REG32  OVR1          	: 1;
  __REG32  OVR2          	: 1;
  __REG32  OVR3          	: 1;
  __REG32  OVR4          	: 1;
  __REG32  OVR5          	: 1;
  __REG32  OVR6          	: 1;
  __REG32  OVR7          	: 1;
  __REG32  OVR8          	: 1;
  __REG32  OVR9          	: 1;
  __REG32  OVR10         	: 1;
  __REG32                 : 5;
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
  __REG32  DATA           :12;
  __REG32                 :20;
} __adc_crr_bits;

/* ADC Gain Calibration Register */
typedef struct {
  __REG32  GCC_FRAC       :14;
  __REG32  GCC_INT        : 1;
  __REG32                 :17;
} __adc_gcr_bits;

/* ADC Gain Calibration Register */
typedef struct {
  __REG32  ADCOCC         :14;
  __REG32                 :18;
} __adc_ocr_bits;

/* CM ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __cm_idr_bits;

/* CM Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :15;
  __REG32  SWRSTKEY       :16;
} __cm_srr_bits;

/* CM Control Set Register */
/* CM Control Clear Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32                 : 3;
  __REG32  FWAKE          : 1;
  __REG32  				        : 1;
  __REG32  PLL            : 1;
  __REG32  STCLK          : 1;
  __REG32  PCLK           : 1;
  __REG32  			          : 1;
  __REG32  IDLEW          : 1;
  __REG32                 :10;
  __REG32  EMCMRST        : 1;
  __REG32  EMCM           : 1;
  __REG32                 : 8;
} __cm_csr_bits;

/* CM Peripheral Clock Set Register */
/* CM Peripheral Clock Clear Register */
/* CM Peripheral Clock Status Register */
typedef struct {
  __REG32  OPACLK         : 1;
  __REG32  WDTCLK         : 1;
  __REG32  PWM0CLK        : 1;
  __REG32  PWM1CLK        : 1;
  __REG32  PWM2CLK        : 1;
  __REG32  PWM3CLK        : 1;
  __REG32  PPDCLK         : 1;
  __REG32  IMCCLK         : 1;
  __REG32  TC0CLK         : 1;
  __REG32  TC1CLK         : 1;
  __REG32  TC2CLK         : 1;
  __REG32  			          : 5;
  __REG32  USART0CLK      : 1;
  __REG32  					      : 5;
  __REG32  ADCCLK         : 1;
  __REG32  COMPCLK        : 1;
  __REG32  SPI0CLK	      : 1;
  __REG32                 : 4;
  __REG32  IFCCLK         : 1;
  __REG32  IOCLK          : 1;
  __REG32  				        : 1;
} __cm_pcsr0_bits;

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
  __REG32                 :20;
} __cm_mr0_bits;

/* CM Mode Register 1 */
typedef struct {
  __REG32  SYSCLK         : 2;
  __REG32                 : 2;
  __REG32  WDTCLK         : 2;
  __REG32                 :26;
} __cm_mr1_bits;

/* CM Interrupt Mask Set/Clear Register */
/* CM Masked Interrupt Status Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  			          : 1;
  __REG32  WDTCLKS        : 1;
  __REG32  STABLE         : 1;
  __REG32                 : 2;
  __REG32  PLL            : 1;
  __REG32                 : 6;
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
  __REG32  			          : 1;
  __REG32  WDTCLKS        : 1;
  __REG32  STABLE         : 1;
  __REG32                 : 2;
  __REG32  PLL            : 1;
  __REG32                 : 6;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32  			          : 1;
  __REG32  CMDERR         : 1;
  __REG32                 :13;
} __cm_risr_bits;

/* CM Status Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  			          : 1;
  __REG32  WDTCLKS        : 1;
  __REG32  STABLE         : 1;
  __REG32  FWAKE          : 1;
  __REG32  			          : 1;
  __REG32  PLL            : 1;
  __REG32  STCLK          : 1;
  __REG32  PCLK           : 1;
  __REG32  			          : 1;
  __REG32  IDLEW          : 1;
  __REG32  			          : 2;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32  LVDRS          : 1;
  __REG32  CMDERR         : 1;
  __REG32  				        : 3;
  __REG32  EMCMRST        : 1;
  __REG32  EMCM           : 1;
  __REG32  SWRSTS         : 1;
  __REG32  NRSTS          : 1;
  __REG32  LVDRSTS        : 1;
  __REG32  WDTRSTS        : 1;
  __REG32  PORRSTS        : 1;
  __REG32  				        : 1;
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
  __REG32  EDGE0          : 2;
  __REG32  WEN0           : 1;
  __REG32  WSRC1          : 5;
  __REG32  EDGE1          : 2;
  __REG32  WEN1           : 1;
  __REG32  WSRC2          : 5;
  __REG32  EDGE2          : 2;
  __REG32  WEN2           : 1;
  __REG32  WSRC3          : 5;
  __REG32  EDGE3          : 2;
  __REG32  WEN3           : 1;
} __cm_wcr0_bits;

/* CM Wakeup Control Register 1 */
typedef struct {
  __REG32  WSRC4          : 5;
  __REG32  EDGE4          : 2;
  __REG32  WEN4           : 1;
  __REG32  WSRC5          : 5;
  __REG32                 : 1;
  __REG32  EDGE5          : 1;
  __REG32  WEN5           : 1;
  __REG32  WSRC6          : 5;
  __REG32  EDGE6          : 2;
  __REG32  WEN6           : 1;
  __REG32  WSRC7          : 5;
  __REG32  EDGE7          : 2;
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
  __REG32  			          : 1;
  __REG32  NORIVC         : 1;
  __REG32                 :30;
} __cm_psr_bits;

/* PPD ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __ppd_idr_bits;

/* PPD Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __ppd_cedr_bits;

/* PPD Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __ppd_srr_bits;

/* PPD Control Register 0 */
typedef struct {
  __REG32  PPDCLKSEL             : 3;
  __REG32  PPDEN                 : 1;
  __REG32  PPDTYPE               : 2;
  __REG32  SCDCTRL               : 1;
  __REG32  					             : 1;
  __REG32  PHASEA                : 3;
  __REG32  ESELA                 : 2;
  __REG32  PHASEB                : 3;
  __REG32  ESELB                 : 2;
  __REG32  PHASEZ                : 3;
  __REG32  ESELZ                 : 2;
  __REG32                        : 1;
  __REG32  PPDFILTER             : 3;
  __REG32                        : 1;
  __REG32  HOLDTRIG              : 4;
} __ppd_cr0_bits;

/* PPD Control Register 1 */
typedef struct {
  __REG32  PCTEN                 : 1;
  __REG32  PCTCL                 : 1;
  __REG32  PCCL                  : 1;
  __REG32  PCSYNCHCL             : 1;
  __REG32                        : 4;
  __REG32  PCTPRESCALE           : 4;
  __REG32                        : 4;
  __REG32  SCTEN                 : 1;
  __REG32  SCTCL                 : 1;
  __REG32  SCCL			             : 1;
  __REG32  SCSYNCHCL			       : 1;
  __REG32                        : 4;
  __REG32  SCTPRESCALE			     : 4;
  __REG32                        : 3;
  __REG32  PZCL							     : 1;
} __ppd_cr1_bits;

/* PPD Status Register */
typedef struct {
  __REG32  DIRECTION             : 1;
  __REG32  GLITCH                : 1;
  __REG32  PBSTAT                : 1;
  __REG32  PASTAT                : 1;
  __REG32                        :28;
} __ppd_sr_bits;

/* PPD Interrupt Mask Set and Clear Register */
/* PPD Raw Interrupt Status Register */
/* PPD Masked Interrupt Status Register */
/* PPD Interrupt Clear Register */
typedef struct {
  __REG32  PCMAT                 : 1;
  __REG32  PCSOVF                : 1;
  __REG32  PCSUNF                : 1;
  __REG32  PCAPT                 : 1;
  __REG32  PCTOVF                : 1;
  __REG32  PCCOVF                : 1;
  __REG32  PCCUNF                : 1;
  __REG32                        : 1;
  __REG32  SCMAT                 : 1;
  __REG32  SCSOVF                : 1;
  __REG32  SCSUNF                : 1;
  __REG32  SCAPT                 : 1;
  __REG32  SCTOVF                : 1;
  __REG32  SCCOVF                : 1;
  __REG32  SCCUNF                : 1;
  __REG32                        : 1;
  __REG32  PHASEZ                : 1;
  __REG32                        :15;
} __ppd_imscr_bits;

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

/* Flash ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __pf_idr_bits;

/* Flash Software Reset Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :31;
} __pf_cedr_bits;

/* Flash Clock Enable/Disable Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __pf_srr_bits;

/* Flash Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 3;
  __REG32  CMD                   : 3;
  __REG32                        :25;
} __pf_cr_bits;

/* Flash Mode Register */
typedef struct {
  __REG32  BACEN                 : 1;
  __REG32                        : 6;
  __REG32  FSMODE                : 1;
  __REG32                        :24;
} __pf_mr_bits;

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
} __pf_imscr_bits;

/* Flash Status Register */
typedef struct {
  __REG32  BUSY                  : 1;
  __REG32                        :31;
} __pf_sr_bits;

/* Smart Option Protection Status Register */
typedef struct {
  __REG32                        : 4;
  __REG32  nHWPA0                : 1;
  __REG32  nHWPA1                : 1;
  __REG32  nHWPA2                : 1;
  __REG32  nHWPA3                : 1;
  __REG32  nSWDP                 : 1;
  __REG32                        : 8;
  __REG32  nHWP                  : 1;
  __REG32                        : 9;
  __REG32  nSRP                  : 1;
  __REG32                        : 4;
} __so_psr_bits;

/* Smart Option Configuration Status Register */
typedef struct {
  __REG32  POCCS                 : 1;
  __REG32                        : 1;
  __REG32  XIN                   : 1;
  __REG32  XOUT                  : 1;
  __REG32                        : 8;
  __REG32  BTDIV                 : 4;
  __REG32                        :16;
} __so_csr_bits;

/* Internal OSC Trimming Register */
typedef struct {
  __REG32  OSC                   : 7;
  __REG32                        :17;
  __REG32  IOTKEY                : 8;
} __pf_iotr_bits;

/* IMC ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __imc_idr_bits;

/* IMC Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __imc_cedr_bits;

/* IMC Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __imc_srr_bits;

/* IMC Control Register 0 */
typedef struct {
  __REG32  IMEN                  : 1;
  __REG32  IMMODE                : 1;
  __REG32  WMODE                 : 1;
  __REG32  PWMSWAP               : 1;
  __REG32  PWMPOLU               : 1;
  __REG32  PWMPOLD               : 1;
  __REG32  ESELPWMOFF            : 2;
  __REG32  IMFILTER              : 3;
  __REG32                        : 1;
  __REG32  PWMOFFEN              : 1;
  __REG32  PWMOUTOFFEN           : 1;
  __REG32  PWMOUTEN              : 1;
  __REG32  PWMOUTOFFENBYCOMP     : 1;
  __REG32  IMCLKSEL              : 3;
  __REG32                        : 1;
  __REG32  NUMSKIP               : 5;
  __REG32                        : 1;
  __REG32  SYNCSEL               : 2;
  __REG32  PACRWM                : 1;
  __REG32  PBCRWM                : 1;
  __REG32  PCCRWM                : 1;
  __REG32                        : 1;
} __imc_cr0_bits;

/* IMC Control Register 1 */
typedef struct {
  __REG32  PWMxD2EN              : 1;
  __REG32  PWMxD1EN              : 1;
  __REG32  PWMxD0EN              : 1;
  __REG32  PWMxU2EN              : 1;
  __REG32  PWMxU1EN              : 1;
  __REG32  PWMxU0EN              : 1;
  __REG32                        : 2;
  __REG32  PWMxD2LEVEL           : 1;
  __REG32  PWMxD1LEVEL           : 1;
  __REG32  PWMxD0LEVEL           : 1;
  __REG32  PWMxU2LEVEL           : 1;
  __REG32  PWMxU1LEVEL           : 1;
  __REG32  PWMxU0LEVEL           : 1;
  __REG32                        : 2;
  __REG32  PWMxD2DT              : 1;
  __REG32  PWMxD1DT              : 1;
  __REG32  PWMxD0DT              : 1;
  __REG32  PWMxU2DT              : 1;
  __REG32  PWMxU1DT              : 1;
  __REG32  PWMxU0DT              : 1;
  __REG32                        :10;
} __imc_cr1_bits;

/* IMC Status Register */
typedef struct {
  __REG32  FAULTSTAT             : 1;
  __REG32  UPDOWN                : 1;
  __REG32  COMPEDGEDET           : 1;
  __REG32                        :29;
} __imc_sr_bits;

/* IMC Interrupt Mask Set and Clear Register */
/* IMC Raw Interrupt Status Register */
/* IMC Masked Interrupt Status Register */
/* IMC Interrupt Clear Register */
typedef struct {
  __REG32  FAULT                 : 1;
  __REG32                        : 5;
  __REG32  ZERO                  : 1;
  __REG32  TOP                   : 1;
  __REG32  ADCRM0                : 1;
  __REG32  ADCFM0                : 1;
  __REG32  ADCRM1                : 1;
  __REG32  ADCFM1                : 1;
  __REG32  ADCRM2                : 1;
  __REG32  ADCFM2                : 1;
  __REG32                        :18;
} __imc_imscr_bits;

/* IMC ADC Start Signal Select Register */
typedef struct {
  __REG32  TOPCMPSEL             : 1;
  __REG32  _0SEL                 : 1;
  __REG32  ADCMPR0SEL            : 1;
  __REG32  ADCMPF0SEL            : 1;
  __REG32  ADCMPR1SEL            : 1;
  __REG32  ADCMPF1SEL            : 1;
  __REG32  ADCMPR2SEL            : 1;
  __REG32  ADCMPF2SEL            : 1;
  __REG32                        :24;
} __imc_astsr_bits;

/* IO Mode Low Register x */
typedef struct {
  __REG32  IO0_0_FSEL            : 2;
  __REG32  IO0_1_FSEL            : 2;
  __REG32  IO0_2_FSEL            : 2;
  __REG32  IO0_3_FSEL            : 2;
  __REG32  IO0_4_FSEL            : 2;
  __REG32  IO0_5_FSEL            : 2;
  __REG32  IO0_6_FSEL            : 2;
  __REG32  IO0_7_FSEL            : 2;
  __REG32  IO0_8_FSEL            : 2;
  __REG32  IO0_9_FSEL            : 2;
  __REG32  IO0_10_FSEL           : 2;
  __REG32  IO0_11_FSEL           : 2;
  __REG32  IO0_12_FSEL           : 2;
  __REG32  IO0_13_FSEL           : 2;
  __REG32  IO0_14_FSEL           : 2;
  __REG32  IO0_15_FSEL           : 2;
} __ioconf_mlr_bits;

/* IO Mode High Register x */
typedef struct {
  __REG32  IO0_16_FSEL           : 2;
  __REG32  IO0_17_FSEL           : 2;
  __REG32  IO0_18_FSEL           : 2;
  __REG32  IO0_19_FSEL           : 2;
  __REG32  IO0_20_FSEL           : 2;
  __REG32  IO0_21_FSEL           : 2;
  __REG32  IO0_22_FSEL           : 2;
  __REG32  IO0_23_FSEL           : 2;
  __REG32  IO0_24_FSEL           : 2;
  __REG32  IO0_25_FSEL           : 2;
  __REG32  IO0_26_FSEL           : 2;
  __REG32  IO0_27_FSEL           : 2;
  __REG32  IO0_28_FSEL           : 2;
  __REG32  IO0_29_FSEL           : 2;
  __REG32  IO0_30_FSEL           : 2;
  __REG32  IO0_31_FSEL           : 2;
} __ioconf_mhr_bits;

/* IO Pull-Up Configuration Register x */
typedef struct {
  __REG32  IO0_0_PUEN            : 1;
  __REG32  IO0_1_PUEN            : 1;
  __REG32  IO0_2_PUEN            : 1;
  __REG32  IO0_3_PUEN            : 1;
  __REG32  IO0_4_PUEN            : 1;
  __REG32  IO0_5_PUEN            : 1;
  __REG32  IO0_6_PUEN            : 1;
  __REG32  IO0_7_PUEN            : 1;
  __REG32  IO0_8_PUEN            : 1;
  __REG32  IO0_9_PUEN            : 1;
  __REG32  IO0_10_PUEN           : 1;
  __REG32  IO0_11_PUEN           : 1;
  __REG32  IO0_12_PUEN           : 1;
  __REG32  IO0_13_PUEN           : 1;
  __REG32  IO0_14_PUEN           : 1;
  __REG32  IO0_15_PUEN           : 1;
  __REG32  IO0_16_PUEN           : 1;
  __REG32  IO0_17_PUEN           : 1;
  __REG32  IO0_18_PUEN           : 1;
  __REG32  IO0_19_PUEN           : 1;
  __REG32  IO0_20_PUEN           : 1;
  __REG32  IO0_21_PUEN           : 1;
  __REG32  IO0_22_PUEN           : 1;
  __REG32  IO0_23_PUEN           : 1;
  __REG32  IO0_24_PUEN           : 1;
  __REG32  IO0_25_PUEN           : 1;
  __REG32  IO0_26_PUEN           : 1;
  __REG32  IO0_27_PUEN           : 1;
  __REG32  IO0_28_PUEN           : 1;
  __REG32  IO0_29_PUEN           : 1;
  __REG32  IO0_30_PUEN           : 1;
  __REG32  IO0_31_PUEN           : 1;
} __ioconf_pucr_bits;

/* IO Pull-Up Configuration Register x */
typedef struct {
  __REG32  IO0_0_ODEN            : 1;
  __REG32  IO0_1_ODEN            : 1;
  __REG32  IO0_2_ODEN            : 1;
  __REG32  IO0_3_ODEN            : 1;
  __REG32  IO0_4_ODEN            : 1;
  __REG32  IO0_5_ODEN            : 1;
  __REG32  IO0_6_ODEN            : 1;
  __REG32  IO0_7_ODEN            : 1;
  __REG32  IO0_8_ODEN            : 1;
  __REG32  IO0_9_ODEN            : 1;
  __REG32  IO0_10_ODEN           : 1;
  __REG32  IO0_11_ODEN           : 1;
  __REG32  IO0_12_ODEN           : 1;
  __REG32  IO0_13_ODEN           : 1;
  __REG32  IO0_14_ODEN           : 1;
  __REG32  IO0_15_ODEN           : 1;
  __REG32  IO0_16_ODEN           : 1;
  __REG32  IO0_17_ODEN           : 1;
  __REG32  IO0_18_ODEN           : 1;
  __REG32  IO0_19_ODEN           : 1;
  __REG32  IO0_20_ODEN           : 1;
  __REG32  IO0_21_ODEN           : 1;
  __REG32  IO0_22_ODEN           : 1;
  __REG32  IO0_23_ODEN           : 1;
  __REG32  IO0_24_ODEN           : 1;
  __REG32  IO0_25_ODEN           : 1;
  __REG32  IO0_26_ODEN           : 1;
  __REG32  IO0_27_ODEN           : 1;
  __REG32  IO0_28_ODEN           : 1;
  __REG32  IO0_29_ODEN           : 1;
  __REG32  IO0_30_ODEN           : 1;
  __REG32  IO0_31_ODEN           : 1;
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
  __REG32  BTSEL          : 1;
  __REG32                 : 3;
  __REG32  DBSEL          : 3;
  __REG32                 :21;
} __lcd_cr_bits;

/* LCD Clock Divide Register */
typedef struct {
  __REG32  CDIV           : 3;
  __REG32                 : 4;
  __REG32  CDC            : 1;
  __REG32  CPRE           :16;
  __REG32                 : 8;
} __lcd_cdr_bits;

/* OPAMP ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __opa_idr_bits;

/* OPAMP Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :31;
} __opa_cedr_bits;

/* OPAMP Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __opa_srr_bits;

/* OPAMP Control Register */
typedef struct {
  __REG32  OPA0           : 1;
  __REG32                 : 7;
  __REG32  OPAM0          : 1;
  __REG32                 :23;
} __opa_cr_bits;

/* OPAMP Gain Control Register */
typedef struct {
  __REG32  GV0            : 4;
  __REG32                 : 3;
  __REG32  GCT0           : 1;
  __REG32                 :24;
} __opa_gcr_bits;

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
  __REG32  DIVM           :11;
  __REG32                 :17;
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
  __REG32  DIVM           :11;
  __REG32                 :17;
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
/* Timer/Counter Status Register */
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
  __REG32  			                 : 1;
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
  __REG32  DIVM                  :11;
  __REG32                        :17;
} __tc_cdr_bits;

/* Timer/Counter Counter Size Mask Register */
typedef struct {
  __REG32  SIZE                  : 4;
  __REG32                        :28;
} __tc_csmr_bits;

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

/* Timer/Counter Clock Divider Register */
typedef struct {
  __REG32  DIVN                  : 4;
  __REG32  DIVM                  :28;
} __tc32_cdr_bits;

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
  __REG32                        :19;
} __us_cr_bits;

/* USART Mode Register */
typedef struct {
  __REG32  		                   : 1;
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
  __REG32  			                 : 1;
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
  __REG32                        : 2;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32                        :21;
} __us_imscr_bits;

/* USART Interrupt Clear Register */
typedef struct {
  __REG32                        : 2;
  __REG32  RXBRK                 : 1;
  __REG32                        : 2;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  				               : 1;
  __REG32  IDLE                  : 1;
  __REG32                        :21;
} __us_icr_bits;

/* USART Status Register */
typedef struct {
  __REG32  RXRDY                 : 1;
  __REG32  TXRDY                 : 1;
  __REG32  RXBRK                 : 1;
  __REG32                        : 2;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32  IDLEFLAG              : 1;
  __REG32                        :20;
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
  __REG32  FRACTION              : 4;
  __REG32  CD                    :12;
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
  __REG32                        :21;
  __REG32  DBGEN                 : 1;
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

/* COMP_IDR */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __comp_idr_bits;

/* COMP_CEDR */
typedef struct {
  __REG32  CLKEN	               : 1;
  __REG32                        :30;
  __REG32  DBGEN	               : 1;
} __comp_cedr_bits;

/* COMP_SRR */
typedef struct {
  __REG32  SWRST	               : 1;
  __REG32                        :31;
} __comp_srr_bits;

/* COMP_CR0 */
typedef struct {
  __REG32  COMP0EN               : 1;
  __REG32  COMP0PINSEL           : 1;
  __REG32  COMP0NINSEL           : 1;
  __REG32  COMP0EDGESEL          : 2;
  __REG32                        : 3;
  __REG32  COMP1EN               : 1;
  __REG32  COMP1PINSEL           : 1;
  __REG32  COMP1NINSEL           : 1;
  __REG32  COMP1EDGESEL          : 2;
  __REG32                        : 3;
  __REG32  COMP2EN               : 1;
  __REG32  COMP2PINSEL           : 1;
  __REG32  COMP2NINSEL           : 1;
  __REG32  COMP2EDGESEL          : 2;
  __REG32                        : 3;
  __REG32  COMP3EN               : 1;
  __REG32  COMP3PINSEL           : 1;
  __REG32  COMP3NINSEL           : 1;
  __REG32  COMP3EDGESEL          : 2;
  __REG32                        : 3;
} __comp_cr0_bits;

/* COMP_CR1 */
typedef struct {
  __REG32  INTREFSEL0            : 3;
  __REG32  						           : 1;
  __REG32  COMP0FILTER           : 3;
  __REG32  						           : 1;
  __REG32  INTREFSEL1            : 3;
  __REG32  						           : 1;
  __REG32  COMP1FILTER           : 3;
  __REG32  						           : 1;
  __REG32  INTREFSEL2            : 3;
  __REG32  						           : 1;
  __REG32  COMP2FILTER           : 3;
  __REG32  						           : 1;
  __REG32  INTREFSEL3            : 3;
  __REG32  						           : 1;
  __REG32  COMP3FILTER           : 3;
  __REG32  						           : 1;
} __comp_cr1_bits;

/* COMP_CR2 */
typedef struct {
  __REG32  CHKSRCSEL0            : 3;
  __REG32  COMP0IMCEN            : 1;
  __REG32  COMP0PPDEN            : 3;
  __REG32  						           : 1;
  __REG32  CHKSRCSEL1            : 3;
  __REG32  COMP1IMCEN            : 1;
  __REG32  COMP1PPDEN            : 3;
  __REG32  						           : 1;
  __REG32  CHKSRCSEL2            : 3;
  __REG32  COMP2IMCEN            : 1;
  __REG32  COMP2PPDEN            : 3;
  __REG32  						           : 1;
  __REG32  CHKSRCSEL3            : 3;
  __REG32  COMP3IMCEN            : 1;
  __REG32  COMP3PPDEN            : 3;
  __REG32  						           : 1;
} __comp_cr2_bits;

/* COMP_SR */
typedef struct {
  __REG32  EDGEEDTSTATUS0        : 1;
  __REG32  EDGEEDTSTATUS1        : 1;
  __REG32  EDGEEDTSTATUS2        : 1;
  __REG32  EDGEEDTSTATUS3        : 1;
  __REG32  						           :12;
  __REG32  COMPSTATUS0           : 1;
  __REG32  COMPSTATUS1           : 1;
  __REG32  COMPSTATUS2           : 1;
  __REG32  COMPSTATUS3           : 1;
  __REG32  						           :12;
} __comp_sr_bits;

/* COMP_IMSCR */
/* COMP_RISR */
/* COMP_MISR */
/* COMP_ICR */
typedef struct {
  __REG32  EDGEDET0			         : 1;
  __REG32  EDGEDET1			         : 1;
  __REG32  EDGEDET2			         : 1;
  __REG32  EDGEDET3			         : 1;
  __REG32  						           :28;
} __comp_imscr_bits;

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
__IO_REG32_BIT(ADC_CEDR,        0x40040004,__READ_WRITE ,__adc_cedr_bits);
__IO_REG32_BIT(ADC_SRR,         0x40040008,__WRITE      ,__adc_srr_bits);
__IO_REG32_BIT(ADC_CSR,         0x4004000C,__WRITE      ,__adc_csr_bits);
__IO_REG32_BIT(ADC_CCR,         0x40040010,__WRITE      ,__adc_ccr_bits);
__IO_REG32_BIT(ADC_CDR,         0x40040014,__READ_WRITE ,__adc_cdr_bits);
__IO_REG32_BIT(ADC_MR,          0x40040018,__READ_WRITE ,__adc_mr_bits);
__IO_REG32_BIT(ADC_CCSR0,       0x40040040,__READ_WRITE ,__adc_ccsr0_bits);
__IO_REG32_BIT(ADC_CCSR1,       0x40040044,__READ_WRITE ,__adc_ccsr1_bits);
__IO_REG32_BIT(ADC_SSR,         0x40040048,__READ				,__adc_ssr_bits);
__IO_REG32_BIT(ADC_SR,          0x40040060,__READ       ,__adc_sr_bits);
__IO_REG32_BIT(ADC_IMSCR,       0x40040064,__READ_WRITE ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_RISR,        0x40040068,__READ       ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_MISR,        0x4004006C,__READ       ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_ICR,         0x40040070,__WRITE      ,__adc_imscr_bits);
__IO_REG32_BIT(ADC_CRR0,        0x40040080,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR1,        0x40040084,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR2,        0x40040088,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR3,        0x4004008C,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR4,        0x40040090,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR5,        0x40040094,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR6,        0x40040098,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR7,        0x4004009C,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR8,        0x400400A0,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR9,        0x400400A4,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CRR10,       0x400400A8,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_GCR,         0x400400AC,__READ_WRITE ,__adc_gcr_bits);
__IO_REG32_BIT(ADC_OCR,         0x400400B0,__READ_WRITE ,__adc_ocr_bits);
__IO_REG32_BIT(ADC_CBR0,       	0x40040100,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR1,       	0x40040104,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR2,       	0x40040108,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR3,       	0x4004010C,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR4,       	0x40040110,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR5,       	0x40040114,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR6,       	0x40040118,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR7,       	0x4004011C,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR8,       	0x40040120,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR9,       	0x40040124,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC_CBR10,       0x40040128,__READ       ,__adc_crr_bits);

/***************************************************************************
 **
 **  System
 **
 ***************************************************************************/
__IO_REG32_BIT(CM_IDR,          0x40020000,__READ       ,__cm_idr_bits);
__IO_REG32_BIT(CM_SRR,          0x40020004,__WRITE      ,__cm_srr_bits);
__IO_REG32_BIT(CM_CSR,          0x40020008,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_CCR,          0x4002000C,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_PCSR0,        0x40020010,__WRITE      ,__cm_pcsr0_bits);
__IO_REG32_BIT(CM_PCCR0,        0x40020018,__WRITE      ,__cm_pcsr0_bits);
__IO_REG32_BIT(CM_PCKSR0,       0x40020020,__READ       ,__cm_pcsr0_bits);
__IO_REG32_BIT(CM_MR0,          0x40020028,__READ_WRITE ,__cm_mr0_bits);
__IO_REG32_BIT(CM_MR1,          0x4002002C,__READ_WRITE ,__cm_mr1_bits);
__IO_REG32_BIT(CM_IMSCR,        0x40020030,__WRITE      ,__cm_imscr_bits);
__IO_REG32_BIT(CM_RISR,         0x40020034,__READ       ,__cm_risr_bits);
__IO_REG32_BIT(CM_MISR,         0x40020038,__READ       ,__cm_imscr_bits);
__IO_REG32_BIT(CM_ICR,          0x4002003C,__WRITE      ,__cm_risr_bits);
__IO_REG32_BIT(CM_SR,           0x40020040,__READ_WRITE ,__cm_sr_bits);
__IO_REG32_BIT(CM_SCDR,         0x40020044,__READ_WRITE ,__cm_scdr_bits);
__IO_REG32_BIT(CM_PCDR,         0x40020048,__READ_WRITE ,__cm_pcdr_bits);
__IO_REG32_BIT(CM_PSTR,         0x40020058,__READ_WRITE ,__cm_pstr_bits);
__IO_REG32_BIT(CM_PDPR,         0x4002005C,__READ_WRITE ,__cm_pdpr_bits);
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
 **  PPD
 **
 ***************************************************************************/
__IO_REG32_BIT(PPD_IDR,         0x400C0000,__READ       ,__ppd_idr_bits);
__IO_REG32_BIT(PPD_CEDR,        0x400C0004,__READ_WRITE ,__ppd_cedr_bits);
__IO_REG32_BIT(PPD_SRR,         0x400C0008,__WRITE      ,__ppd_srr_bits);
__IO_REG32_BIT(PPD_CR0,         0x400C000C,__READ_WRITE ,__ppd_cr0_bits);
__IO_REG32_BIT(PPD_CR1,         0x400C0010,__READ_WRITE ,__ppd_cr1_bits);
__IO_REG32_BIT(PPD_SR,          0x400C0014,__READ_WRITE ,__ppd_sr_bits);
__IO_REG32_BIT(PPD_IMSCR,       0x400C0018,__READ_WRITE ,__ppd_imscr_bits);
__IO_REG32_BIT(PPD_RISR,        0x400C001C,__READ       ,__ppd_imscr_bits);
__IO_REG32_BIT(PPD_MISR,        0x400C0020,__READ       ,__ppd_imscr_bits);
__IO_REG32_BIT(PPD_ICR,         0x400C0024,__WRITE      ,__ppd_imscr_bits);
__IO_REG16(    PPD_PCR,         0x400C0028,__READ_WRITE );
__IO_REG16(    PPD_PCRR,        0x400C002C,__READ_WRITE );
__IO_REG16(    PPD_PCTR,        0x400C0030,__READ_WRITE );
__IO_REG16(    PPD_PCTVR,       0x400C0034,__READ_WRITE );
__IO_REG16(    PPD_SCR,       	0x400C0038,__READ_WRITE );
__IO_REG16(    PPD_SCRR,       	0x400C003C,__READ_WRITE );
__IO_REG16(    PPD_SCTR,        0x400C0040,__READ_WRITE );
__IO_REG16(    PPD_SCTVR,       0x400C0044,__READ_WRITE );
__IO_REG16(    PPD_PCHR,        0x400C0048,__READ_WRITE );
__IO_REG16(    PPD_PCTHR,       0x400C004C,__READ_WRITE );
__IO_REG16(    PPD_PCTVHR,      0x400C0050,__READ_WRITE );
__IO_REG16(    PPD_SCHR,       	0x400C0054,__READ_WRITE );
__IO_REG16(    PPD_SCTHR,       0x400C0058,__READ_WRITE );
__IO_REG16(    PPD_SCTVHR,      0x400C005C,__READ_WRITE );

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
 **  FLASH
 **
 ***************************************************************************/
__IO_REG32_BIT(PF_IDR,          0x40010000,__READ       ,__pf_idr_bits);
__IO_REG32_BIT(PF_CEDR,         0x40010004,__READ_WRITE ,__pf_cedr_bits);
__IO_REG32_BIT(PF_SRR,          0x40010008,__WRITE      ,__pf_srr_bits);
__IO_REG32_BIT(PF_CR,           0x4001000C,__READ_WRITE ,__pf_cr_bits);
__IO_REG32_BIT(PF_MR,           0x40010010,__READ_WRITE ,__pf_mr_bits);
__IO_REG32_BIT(PF_IMSCR,        0x40010014,__READ_WRITE ,__pf_imscr_bits);
__IO_REG32_BIT(PF_RISR,         0x40010018,__READ       ,__pf_imscr_bits);
__IO_REG32_BIT(PF_MISR,         0x4001001C,__READ       ,__pf_imscr_bits);
__IO_REG32_BIT(PF_ICR,          0x40010020,__WRITE      ,__pf_imscr_bits);
__IO_REG32_BIT(PF_SR,           0x40010024,__READ       ,__pf_sr_bits);
__IO_REG32(    PF_AR,           0x40010028,__READ_WRITE );
__IO_REG32(    PF_DR,           0x4001002C,__READ_WRITE );
__IO_REG32(    PF_KR,           0x40010030,__WRITE      );
__IO_REG32_BIT(SO_PSR,          0x40010034,__READ       ,__so_psr_bits);
__IO_REG32_BIT(SO_CSR,          0x40010038,__READ       ,__so_csr_bits);
__IO_REG32_BIT(PF_IOTR,         0x4001003C,__READ_WRITE ,__pf_iotr_bits);

/***************************************************************************
 **
 **  IMC
 **
 ***************************************************************************/
__IO_REG32_BIT(IMC_IDR,         0x400B0000,__READ       ,__imc_idr_bits);
__IO_REG32_BIT(IMC_CEDR,        0x400B0004,__READ_WRITE ,__imc_cedr_bits);
__IO_REG32_BIT(IMC_SRR,         0x400B0008,__WRITE      ,__imc_srr_bits);
__IO_REG32_BIT(IMC_CR0,         0x400B000C,__READ_WRITE ,__imc_cr0_bits);
__IO_REG32_BIT(IMC_CR1,         0x400B0010,__READ_WRITE ,__imc_cr1_bits);
__IO_REG16(    IMC_CNTR,        0x400B0014,__READ       );
__IO_REG32_BIT(IMC_SR,          0x400B0018,__READ_WRITE ,__imc_sr_bits);
__IO_REG32_BIT(IMC_IMSCR,       0x400B001C,__READ_WRITE ,__imc_imscr_bits);
__IO_REG32_BIT(IMC_RISR,        0x400B0020,__READ       ,__imc_imscr_bits);
__IO_REG32_BIT(IMC_MISR,        0x400B0024,__READ       ,__imc_imscr_bits);
__IO_REG32_BIT(IMC_ICR,         0x400B0028,__WRITE      ,__imc_imscr_bits);
__IO_REG16(    IMC_TCR,         0x400B002C,__READ_WRITE );
__IO_REG16(    IMC_DTCR,        0x400B0030,__READ_WRITE );
__IO_REG16(    IMC_PACRR,       0x400B0034,__READ_WRITE );
__IO_REG16(    IMC_PBCRR,       0x400B0038,__READ_WRITE );
__IO_REG16(    IMC_PCCRR,       0x400B003C,__READ_WRITE );
__IO_REG16(    IMC_PACFR,       0x400B0040,__READ_WRITE );
__IO_REG16(    IMC_PBCFR,       0x400B0044,__READ_WRITE );
__IO_REG16(    IMC_PCCFR,       0x400B0048,__READ_WRITE );
__IO_REG32_BIT(IMC_ASTSR,       0x400B004C,__READ_WRITE ,__imc_astsr_bits);
__IO_REG16(    IMC_ASCRR0,      0x400B0050,__READ_WRITE );
__IO_REG16(    IMC_ASCRR1,      0x400B0054,__READ_WRITE );
__IO_REG16(    IMC_ASCRR2,      0x400B0058,__READ_WRITE );
__IO_REG16(    IMC_ASCFR0,      0x400B005C,__READ_WRITE );
__IO_REG16(    IMC_ASCFR1,      0x400B0060,__READ_WRITE );
__IO_REG16(    IMC_ASCFR2,      0x400B0064,__READ_WRITE );

/***************************************************************************
 **
 **  IOCONF
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCONF_MLR0,     0x40058000,__READ_WRITE ,__ioconf_mlr_bits);
__IO_REG32_BIT(IOCONF_MHR0,     0x40058004,__READ_WRITE ,__ioconf_mhr_bits);
__IO_REG32_BIT(IOCONF_PUCR0,    0x40058008,__READ_WRITE ,__ioconf_pucr_bits);
__IO_REG32_BIT(IOCONF_ODCR0,    0x4005800C,__READ_WRITE ,__ioconf_odcr_bits);

/***************************************************************************
 **
 **  OPA
 **
 ***************************************************************************/
__IO_REG32_BIT(OPA_IDR,         0x40041000,__READ       ,__opa_idr_bits);
__IO_REG32_BIT(OPA_CEDR,        0x40041004,__READ_WRITE ,__opa_cedr_bits);
__IO_REG32_BIT(OPA_SRR,         0x40041008,__WRITE      ,__opa_srr_bits);
__IO_REG32_BIT(OPA_CR,          0x4004100C,__READ_WRITE ,__opa_cr_bits);
__IO_REG32_BIT(OPA_GCR,         0x40041010,__READ_WRITE ,__opa_gcr_bits);

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
__IO_REG32_BIT(PWM0_MISR,       0x40070020,__READ				,__pwm_imscr_bits);
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
__IO_REG32_BIT(PWM1_MISR,       0x40071020,__READ				,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_ICR,        0x40071024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_CDR,        0x40071028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM1_PRDR,       0x4007102C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM1_PULR,       0x40071030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM1_CCDR,       0x40071034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM1_CPRDR,      0x40071038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM1_CPULR,      0x4007103C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2_IDR,        0x40072000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM2_CEDR,       0x40072004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM2_SRR,        0x40072008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM2_CSR,        0x4007200C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM2_CCR,        0x40072010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM2_SR,         0x40072014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM2_IMSCR,      0x40072018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_RISR,       0x4007201C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_MISR,       0x40072020,__READ				,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_ICR,        0x40072024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_CDR,        0x40072028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM2_PRDR,       0x4007202C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM2_PULR,       0x40072030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM2_CCDR,       0x40072034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM2_CPRDR,      0x40072038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM2_CPULR,      0x4007203C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM3
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM3_IDR,        0x40073000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM3_CEDR,       0x40073004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM3_SRR,        0x40073008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM3_CSR,        0x4007300C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM3_CCR,        0x40073010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM3_SR,         0x40073014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM3_IMSCR,      0x40073018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_RISR,       0x4007301C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_MISR,       0x40073020,__READ				,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_ICR,        0x40073024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_CDR,        0x40073028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM3_PRDR,       0x4007302C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM3_PULR,       0x40073030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM3_CCDR,       0x40073034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM3_CPRDR,      0x40073038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM3_CPULR,      0x4007303C,__READ       ,__pwm_pulr_bits);

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
__IO_REG32_BIT(TC0_SR,          0x40060018,__READ       ,__tc_csr_bits);
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
__IO_REG32_BIT(TC1_SR,          0x40061018,__READ       ,__tc_csr_bits);
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
__IO_REG32_BIT(TC2_SR,          0x40062018,__READ       ,__tc_csr_bits);
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
 **  COMP
 **
 ***************************************************************************/
__IO_REG32_BIT(COMP_IDR,        0x40042000,__READ       ,__comp_idr_bits);
__IO_REG32_BIT(COMP_CEDR,       0x40042004,__READ_WRITE ,__comp_cedr_bits);
__IO_REG32_BIT(COMP_SRR,        0x40042008,__WRITE 			,__comp_srr_bits);
__IO_REG32_BIT(COMP_CR0,        0x4004200C,__READ_WRITE ,__comp_cr0_bits);
__IO_REG32_BIT(COMP_CR1,        0x40042010,__READ_WRITE ,__comp_cr1_bits);
__IO_REG32_BIT(COMP_CR2,        0x40042014,__READ_WRITE ,__comp_cr2_bits);
__IO_REG32_BIT(COMP_SR,         0x40042018,__READ				,__comp_sr_bits);
__IO_REG32_BIT(COMP_IMSCR,      0x4004201C,__READ_WRITE ,__comp_imscr_bits);
__IO_REG32_BIT(COMP_RISR,      	0x40042020,__READ				,__comp_imscr_bits);
__IO_REG32_BIT(COMP_MISR,       0x40042024,__READ				,__comp_imscr_bits);
__IO_REG32_BIT(COMP_ICR,        0x40042028,__WRITE			,__comp_imscr_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  S3FN429 interrupt source number
 **
 ***************************************************************************/
#define WDTINT            0x00    /* Watch-dog Timer Interrupt */
#define IMC0_FAULTINT     0x01    /* Inverter Motor Controller (IMC)0 Fault Interrupt */
#define EDGEDET0INT       0x02    /* Comparator0 interrupt */
#define EDGEDET1INT       0x03    /* Comparator1 interrupt */
#define EDGEDET2INT       0x04    /* Comparator2 interrupt */
#define EDGEDET3INT       0x05    /* Comparator3 interrupt */
#define IMC0_ZEROINT      0x06    /* IMC 0 Counter Zero Match, Top Compare Match Interrupt 0 */
#define IMC0_ADCRM0INT    0x07    /* IMC 0 Analog to Digital Converter (ADC) Compare Match in Rising/Falling Time Interrupt 0 */
#define IMC0_ADCRM1INT    0x08    /* IMC 0 Analog to Digital Converter (ADC) Compare Match in Rising/Falling Time Interrupt 1 */
#define IMC0_ADCRM2INT    0x09    /* IMC 0 Analog to Digital Converter (ADC) Compare Match in Rising/Falling Time Interrupt 2 */
#define ADCINT            0x0a    /* ADC Interrupt */
#define PPD_PINT         	0x0b    /* Pulse Position Detector (PPD) Position Counter/Capture and PHASEZ Interrupt */
#define PPD_SINT          0x0c    /* PPD Speed Counter/Capture Interrupt */
#define USART_RXRDYINT    0x0d    /* Universal Synchronous/Asynchronous Receiver/Transmitter (USART) RXRDY Interrupt */
#define CMINT         		0x0e    /* Clock Manager Interrupt */
#define IFCINT         		0x0f    /* Internal Flash Controller Interrupt */
#define WSIAINT         	0x10    /* Clock Manager Wakeup Source Interrupt 0 */
#define WSIBINT         	0x11    /* Clock Manager Wakeup Source Interrupt 1 */
#define WSICINT         	0x12    /* Clock Manager Wakeup Source Interrupt 2 */
#define WSIDINT         	0x13    /* Clock Manager Wakeup Source Interrupt 3 */
#define WSIEINT         	0x14    /* Clock Manager Wakeup Source Interrupt 4,5 */
#define WSIFINT         	0x15    /* Clock Manager Wakeup Source Interrupt 6,7 */
#define TC0INT            0x16    /* Timer/Counter0 Interrupt */
#define TC1INT            0x17    /* Timer/Counter1 Interrupt */
#define TC2INT            0x18    /* Timer/Counter2 Interrupt */
#define PWM0INT           0x19    /* Pulse Width Modulation (PWM)0 Interrupt */
#define PWM1INT           0x1A    /* Pulse Width Modulation (PWM)1 Interrupt */
#define PWM2INT           0x1B    /* Pulse Width Modulation (PWM)2 Interrupt */
#define PWM3INT           0x1C    /* Pulse Width Modulation (PWM)3 Interrupt */
#define SPI0INT           0x1D    /* Serial Peripheral Interface (SPI) Interrupt */
#define USART0INT         0x1E    /* USART0 Interrupt */
#define GPIO0INT          0x1F    /* General Purpose IO (GPIO) Interrupt */

#endif    /* __S3FN429_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            		0x08
Interrupt1   = HardFault      		0x0C
Interrupt2   = SVC            		0x2C
Interrupt3   = PendSV         		0x38
Interrupt4   = SysTick        		0x3C
Interrupt5   = WDTINT             0x40
Interrupt6   = IMC0_FAULTINT      0x44
Interrupt7   = EDGEDET0INT        0x48
Interrupt8   = EDGEDET1INT        0x4C
Interrupt9   = EDGEDET2INT        0x50
Interrupt10  = EDGEDET3INT        0x54
Interrupt11  = IMC0_ZEROINT       0x58
Interrupt12  = IMC0_ADCRM0INT     0x5C
Interrupt13  = IMC0_ADCRM1INT     0x60
Interrupt14  = IMC0_ADCRM2INT     0x64
Interrupt15  = ADCINT             0x68
Interrupt16  = PPD_PINT         	0x6C
Interrupt17  = PPD_SINT           0x70
Interrupt18  = USART_RXRDYINT     0x74
Interrupt19  = CMINT         		  0x78
Interrupt20  = IFCINT         		0x7C
Interrupt21  = WSIAINT         	  0x80
Interrupt22  = WSIBINT         	  0x84
Interrupt23  = WSICINT         	  0x88
Interrupt24  = WSIDINT         	  0x8C
Interrupt25  = WSIEINT         	  0x90
Interrupt26  = WSIFINT         	  0x94
Interrupt27  = TC0INT             0x98
Interrupt28  = TC1INT             0x9C
Interrupt29  = TC2INT             0xA0
Interrupt30  = PWM0INT            0xA4
Interrupt31  = PWM1INT            0xA8
Interrupt32  = PWM2INT            0xAC
Interrupt33  = PWM3INT            0xB0
Interrupt34  = SPI0INT            0xB4
Interrupt35  = USART0INT          0xB8
Interrupt36  = GPIO0INT           0xBC

###DDF-INTERRUPT-END###*/
