/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Samsung S3FN60D
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2005
 **
 **    $Revision: 45631 $
 **
 ***************************************************************************/

#ifndef __S3FN60D_H
#define __S3FN60D_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    S3FN60D SPECIAL FUNCTION REGISTERS
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

/* System Clock Control register */
typedef struct {
  __REG32  PLL_EN         : 1;
  __REG32  CLK_SEL        : 1;
  __REG32  LFPASS         : 1;
  __REG32  			          : 1;
  __REG32  CDIV           : 4;
  __REG32  SYSTICK_EN     : 1;
  __REG32                 :23;
} __clkcon0_bits;

/* Internal Ring Oscillator Control register */
typedef struct {
  __REG32  LDIV           : 2;
  __REG32  LCLK_EN        : 1;
  __REG32  LCLK_SEL       : 1;
  __REG32                 : 4;
  __REG32  IDLE_PWR       : 1;
  __REG32                 : 7;
  __REG32  LCLK_STAT      : 1;
  __REG32  PLLCLK_STAT    : 1;
  __REG32  MCLK_STAT      : 1;
  __REG32                 :13;
} __clkcon1_bits;

/* PLL PMS value Register */
typedef struct {
  __REG32  SDIV           : 2;
  __REG32                 : 6;
  __REG32  MDIV           : 8;
  __REG32  PDIV           : 6;
  __REG32                 :10;
} __pllpms_bits;

/* PLL lock count register */
typedef struct {
  __REG32  LOCK_CNT       :13;
  __REG32                 :19;
} __plllcnt_bits;

/* PLL lock status register */
typedef struct {
  __REG32  PLL_LOCK       : 1;
  __REG32                 :31;
} __plllock_bits;

/* Chip reset by s/w */
typedef struct {
  __REG32  SWRST          : 8;
  __REG32                 :24;
} __swrst_bits;

/* Power Source Change register */
typedef struct {
  __REG32  PWRCHG         : 1;
  __REG32  VUSB_DET       : 1;
  __REG32                 :30;
} __pwrchg_bits;

/* Reset Source Indicating Register */
typedef struct {
  __REG32  POR_BIT        : 1;
  __REG32  LVD_BIT        : 1;
  __REG32  WDT_BIT        : 1;
  __REG32                 : 1;
  __REG32  NRESET_BIT     : 1;
  __REG32  SW_RESET       : 1;
  __REG32                 :26;
} __resetid_bits;

/* Low Voltage Detect Control Register */
typedef struct {
  __REG32  LVD_FLAG       : 1;
  __REG32                 :31;
} __lvdcon_bits;

/* Low Voltage Detect Flag Selection Register */
typedef struct {
  __REG32  LVD_FLAG       : 2;
  __REG32                 :30;
} __lvdsel_bits;

/* Port 0 register */
typedef struct {
  __REG32  P00            : 1;
  __REG32  P01            : 1;
  __REG32  P02            : 1;
  __REG32  P03            : 1;
  __REG32  P04            : 1;
  __REG32  P05            : 1;
  __REG32  P06            : 1;
  __REG32  P07            : 1;
  __REG32                 :24;
} __p0_bits;

/* Port 1 register */
typedef struct {
  __REG32  P10            : 1;
  __REG32  P11            : 1;
  __REG32  P12            : 1;
  __REG32  P13            : 1;
  __REG32  P14            : 1;
  __REG32  P15            : 1;
  __REG32  P16            : 1;
  __REG32  P17            : 1;
  __REG32                 :24;
} __p1_bits;

/* Port 2 register */
typedef struct {
  __REG32  P20            : 1;
  __REG32  P21            : 1;
  __REG32  P22            : 1;
  __REG32  P23            : 1;
  __REG32  P24            : 1;
  __REG32  P25            : 1;
  __REG32  P26            : 1;
  __REG32  P27            : 1;
  __REG32                 :24;
} __p2_bits;

/* Port 3 register */
typedef struct {
  __REG32  P30            : 1;
  __REG32  P31            : 1;
  __REG32  P32            : 1;
  __REG32  P33            : 1;
  __REG32  P34            : 1;
  __REG32  P35            : 1;
  __REG32  P36            : 1;
  __REG32                 :25;
} __p3_bits;

/* Port 4 register */
typedef struct {
  __REG32  P40            : 1;
  __REG32  P41            : 1;
  __REG32  P42            : 1;
  __REG32  P43            : 1;
  __REG32  P44            : 1;
  __REG32  P45            : 1;
  __REG32  P46            : 1;
  __REG32  P47            : 1;
  __REG32                 :24;
} __p4_bits;

/* Port 5 register */
typedef struct {
  __REG32  P50            : 1;
  __REG32  P51            : 1;
  __REG32  P52            : 1;
  __REG32  P53            : 1;
  __REG32  P54            : 1;
  __REG32  P55            : 1;
  __REG32  P56            : 1;
  __REG32  P57            : 1;
  __REG32                 :24;
} __p5_bits;

/* Port 6 register */
typedef struct {
  __REG32  P60            : 1;
  __REG32  P61            : 1;
  __REG32  P62            : 1;
  __REG32                 :29;
} __p6_bits;

/* Port 0 control register high */
typedef struct {
  __REG32  P04_SET        : 2;
  __REG32  P05_SET        : 2;
  __REG32  P06_SET        : 2;
  __REG32  P07_SET        : 2;
  __REG32                 :24;
} __p0conh_bits;

/* Port 0 control register low */
typedef struct {
  __REG32  P00_SET        : 2;
  __REG32  P01_SET        : 2;
  __REG32  P02_SET        : 2;
  __REG32  P03_SET        : 2;
  __REG32                 :24;
} __p0conl_bits;

/* Port 0 interrupt EDGE control register */
typedef struct {
  __REG32  P00_EDGE       : 1;
  __REG32  P01_EDGE       : 1;
  __REG32  P02_EDGE       : 1;
  __REG32  P03_EDGE       : 1;
  __REG32  P04_EDGE       : 1;
  __REG32  P05_EDGE       : 1;
  __REG32  P06_EDGE       : 1;
  __REG32  P07_EDGE       : 1;
  __REG32                 :24;
} __p0edge_bits;

/* Port 0 interrupt control register */
typedef struct {
  __REG32  P00_INT        : 1;
  __REG32  P01_INT        : 1;
  __REG32  P02_INT        : 1;
  __REG32  P03_INT        : 1;
  __REG32  P04_INT        : 1;
  __REG32  P05_INT        : 1;
  __REG32  P06_INT        : 1;
  __REG32  P07_INT        : 1;
  __REG32                 :24;
} __p0int_bits;

/* Port 0 interrupt pending register*/
typedef struct {
  __REG32  P00_PND        : 1;
  __REG32  P01_PND        : 1;
  __REG32  P02_PND        : 1;
  __REG32  P03_PND        : 1;
  __REG32  P04_PND        : 1;
  __REG32  P05_PND        : 1;
  __REG32  P06_PND        : 1;
  __REG32  P07_PND        : 1;
  __REG32                 :24;
} __p0pnd_bits;

/* Port 0 interrupt pending register*/
typedef struct {
  __REG32  P00_PUR        : 1;
  __REG32  P01_PUR        : 1;
  __REG32  P02_PUR        : 1;
  __REG32  P03_PUR        : 1;
  __REG32  P04_PUR        : 1;
  __REG32  P05_PUR        : 1;
  __REG32  P06_PUR        : 1;
  __REG32  P07_PUR        : 1;
  __REG32                 :24;
} __p0pur_bits;

/* Port 1 control register high */
typedef struct {
  __REG32  P14_SET        : 2;
  __REG32  P15_SET        : 2;
  __REG32  P16_SET        : 2;
  __REG32  P17_SET        : 2;
  __REG32                 :24;
} __p1conh_bits;

/* Port 1 control register low */
typedef struct {
  __REG32  P10_SET        : 2;
  __REG32  P11_SET        : 2;
  __REG32  P12_SET        : 2;
  __REG32  P13_SET        : 2;
  __REG32                 :24;
} __p1conl_bits;

/* Port 1 interrupt EDGE control register */
typedef struct {
  __REG32  P10_EDGE       : 1;
  __REG32  P11_EDGE       : 1;
  __REG32  P12_EDGE       : 1;
  __REG32  P13_EDGE       : 1;
  __REG32  P14_EDGE       : 1;
  __REG32  P15_EDGE       : 1;
  __REG32  P16_EDGE       : 1;
  __REG32  P17_EDGE       : 1;
  __REG32                 :24;
} __p1edge_bits;

/* Port 1 interrupt control register */
typedef struct {
  __REG32  P10_INT        : 1;
  __REG32  P11_INT        : 1;
  __REG32  P12_INT        : 1;
  __REG32  P13_INT        : 1;
  __REG32  P14_INT        : 1;
  __REG32  P15_INT        : 1;
  __REG32  P16_INT        : 1;
  __REG32  P17_INT        : 1;
  __REG32                 :24;
} __p1int_bits;

/* Port 1 interrupt pending register*/
typedef struct {
  __REG32  P10_PND        : 1;
  __REG32  P11_PND        : 1;
  __REG32  P12_PND        : 1;
  __REG32  P13_PND        : 1;
  __REG32  P14_PND        : 1;
  __REG32  P15_PND        : 1;
  __REG32  P16_PND        : 1;
  __REG32  P17_PND        : 1;
  __REG32                 :24;
} __p1pnd_bits;

/* Port 1 interrupt pending register*/
typedef struct {
  __REG32  P10_PUR        : 1;
  __REG32  P11_PUR        : 1;
  __REG32  P12_PUR        : 1;
  __REG32  P13_PUR        : 1;
  __REG32  P14_PUR        : 1;
  __REG32  P15_PUR        : 1;
  __REG32  P16_PUR        : 1;
  __REG32  P17_PUR        : 1;
  __REG32                 :24;
} __p1pur_bits;

/* Port 2 control register high */
typedef struct {
  __REG32  P24_SET        : 2;
  __REG32  P25_SET        : 2;
  __REG32  P26_SET        : 2;
  __REG32  P27_SET        : 2;
  __REG32                 :24;
} __p2conh_bits;

/* Port 2 control register low */
typedef struct {
  __REG32  P20_SET        : 2;
  __REG32  P21_SET        : 2;
  __REG32  P22_SET        : 2;
  __REG32  P23_SET        : 2;
  __REG32                 :24;
} __p2conl_bits;

/* Port 2 interrupt EDGE control register */
typedef struct {
  __REG32  P20_EDGE       : 1;
  __REG32  P21_EDGE       : 1;
  __REG32  P22_EDGE       : 1;
  __REG32  P23_EDGE       : 1;
  __REG32  P24_EDGE       : 1;
  __REG32  P25_EDGE       : 1;
  __REG32  P26_EDGE       : 1;
  __REG32  P27_EDGE       : 1;
  __REG32                 :24;
} __p2edge_bits;

/* Port 2 interrupt control register */
typedef struct {
  __REG32  P20_INT        : 1;
  __REG32  P21_INT        : 1;
  __REG32  P22_INT        : 1;
  __REG32  P23_INT        : 1;
  __REG32  P24_INT        : 1;
  __REG32  P25_INT        : 1;
  __REG32  P26_INT        : 1;
  __REG32  P27_INT        : 1;
  __REG32                 :24;
} __p2int_bits;

/* Port 2 interrupt pending register*/
typedef struct {
  __REG32  P20_PND        : 1;
  __REG32  P21_PND        : 1;
  __REG32  P22_PND        : 1;
  __REG32  P23_PND        : 1;
  __REG32  P24_PND        : 1;
  __REG32  P25_PND        : 1;
  __REG32  P26_PND        : 1;
  __REG32  P27_PND        : 1;
  __REG32                 :24;
} __p2pnd_bits;

/* Port 2 interrupt pending register*/
typedef struct {
  __REG32  P20_PUR        : 1;
  __REG32  P21_PUR        : 1;
  __REG32  P22_PUR        : 1;
  __REG32  P23_PUR        : 1;
  __REG32  P24_PUR        : 1;
  __REG32  P25_PUR        : 1;
  __REG32  P26_PUR        : 1;
  __REG32  P27_PUR        : 1;
  __REG32                 :24;
} __p2pur_bits;

/* Port 3 control register high */
typedef struct {
  __REG32  P34_SET        : 2;
  __REG32  P35_SET        : 2;
  __REG32  P36_SET        : 2;
  __REG32                 :26;
} __p3conh_bits;

/* Port 3 control register low */
typedef struct {
  __REG32  P30_SET        : 2;
  __REG32  P31_SET        : 2;
  __REG32  P32_SET        : 2;
  __REG32  P33_SET        : 2;
  __REG32                 :24;
} __p3conl_bits;

/* Port 3 interrupt pending register*/
typedef struct {
  __REG32  P30_PUR        : 1;
  __REG32  P31_PUR        : 1;
  __REG32  P32_PUR        : 1;
  __REG32  P33_PUR        : 1;
  __REG32  P34_PUR        : 1;
  __REG32  P35_PUR        : 1;
  __REG32  P36_PUR        : 1;
  __REG32                 :25;
} __p3pur_bits;

/* Port 4 control register high */
typedef struct {
  __REG32  P44_SET        : 2;
  __REG32  P45_SET        : 2;
  __REG32  P46_SET        : 2;
  __REG32  P47_SET        : 2;
  __REG32                 :24;
} __p4conh_bits;

/* Port 4 control register low */
typedef struct {
  __REG32  P40_SET        : 2;
  __REG32  P41_SET        : 2;
  __REG32  P42_SET        : 2;
  __REG32  P43_SET        : 2;
  __REG32                 :24;
} __p4conl_bits;

/* Port 4 interrupt EDGE control register */
typedef struct {
  __REG32  P40_EDGE       : 1;
  __REG32  P41_EDGE       : 1;
  __REG32  P42_EDGE       : 1;
  __REG32  P43_EDGE       : 1;
  __REG32  P44_EDGE       : 1;
  __REG32  P45_EDGE       : 1;
  __REG32  P46_EDGE       : 1;
  __REG32  P47_EDGE       : 1;
  __REG32                 :24;
} __p4edge_bits;

/* Port 4 interrupt control register */
typedef struct {
  __REG32  P40_INT        : 1;
  __REG32  P41_INT        : 1;
  __REG32  P42_INT        : 1;
  __REG32  P43_INT        : 1;
  __REG32  P44_INT        : 1;
  __REG32  P45_INT        : 1;
  __REG32  P46_INT        : 1;
  __REG32  P47_INT        : 1;
  __REG32                 :24;
} __p4int_bits;

/* Port 4 interrupt pending register*/
typedef struct {
  __REG32  P40_PND        : 1;
  __REG32  P41_PND        : 1;
  __REG32  P42_PND        : 1;
  __REG32  P43_PND        : 1;
  __REG32  P44_PND        : 1;
  __REG32  P45_PND        : 1;
  __REG32  P46_PND        : 1;
  __REG32  P47_PND        : 1;
  __REG32                 :24;
} __p4pnd_bits;

/* Port 4 interrupt pending register*/
typedef struct {
  __REG32  P40_PUR        : 1;
  __REG32  P41_PUR        : 1;
  __REG32  P42_PUR        : 1;
  __REG32  P43_PUR        : 1;
  __REG32  P44_PUR        : 1;
  __REG32  P45_PUR        : 1;
  __REG32  P46_PUR        : 1;
  __REG32  P47_PUR        : 1;
  __REG32                 :24;
} __p4pur_bits;

/* Port 5 control register high */
typedef struct {
  __REG32  P54_SET        : 2;
  __REG32  P55_SET        : 2;
  __REG32  P56_SET        : 2;
  __REG32  P57_SET        : 2;
  __REG32                 :24;
} __p5conh_bits;

/* Port 5 control register low */
typedef struct {
  __REG32  P50_SET        : 2;
  __REG32  P51_SET        : 2;
  __REG32  P52_SET        : 2;
  __REG32  P53_SET        : 2;
  __REG32                 :24;
} __p5conl_bits;

/* Port 5 interrupt pending register */
typedef struct {
  __REG32  P50_PUR        : 1;
  __REG32  P51_PUR        : 1;
  __REG32  P52_PUR        : 1;
  __REG32  P53_PUR        : 1;
  __REG32  P54_PUR        : 1;
  __REG32  P55_PUR        : 1;
  __REG32  P56_PUR        : 1;
  __REG32  P57_PUR        : 1;
  __REG32                 :24;
} __p5pur_bits;

/* Port 5 interrupt pending register */
typedef struct {
  __REG32  P50_MODE       : 1;
  __REG32  P51_MODE       : 1;
  __REG32                 : 2;
  __REG32  P54_MODE       : 1;
  __REG32  P55_MODE       : 1;
  __REG32                 :26;
} __p5mode_bits;

/* Port 56control register low */
typedef struct {
  __REG32  P60_SET        : 2;
  __REG32  P61_SET        : 2;
  __REG32  P62_SET        : 2;
  __REG32                 :26;
} __p6conl_bits;

/* Port 6 interrupt pending register */
typedef struct {
  __REG32  P60_PUR        : 1;
  __REG32  P61_PUR        : 1;
  __REG32  P62_PUR        : 1;
  __REG32                 :29;
} __p6pur_bits;

/* Basic timer control register */
typedef struct {
  __REG32  WDT_CLR        : 1;
  __REG32  BT_CLR         : 1;
  __REG32  BTCLK_SEL      : 3;
  __REG32  WDT_EN         : 3;
  __REG32  BT_INT_EN      : 1;
  __REG32                 :23;
} __btcon_bits;

/* Basic timer counter value */
typedef struct {
  __REG32  BTCNT          : 8;
  __REG32                 :24;
} __btcnt_bits;

/* Watchdog timer counter value */
typedef struct {
  __REG32  WDTCNT         : 8;
  __REG32                 :24;
} __wdtcnt_bits;

/* Counter A control register 0 */
typedef struct {
  __REG32  CA_OSP         : 1;
  __REG32  CA_MODE        : 1;
  __REG32  CA_START       : 1;
  __REG32  CA_STOP        : 1;
  __REG32  CA_INTEN       : 1;
  __REG32  CA_INTTIME     : 2;
  __REG32  CA_CLK         : 2;
  __REG32                 :23;
} __cacon0_bits;

/* Counter A control register 1 */
typedef struct {
  __REG32  CA_SW_STROBE_DATA     : 1;
  __REG32  CA_HW_STROBE_DATA     : 1;
  __REG32  CA_TAMATCH_REM_ONOFF  : 1;
  __REG32  REM_STAT              : 1;
  __REG32  TA_ENVELOPE           : 1;
  __REG32  CARRIER_CON           : 1;
  __REG32                        :26;
} __cacon1_bits;

/* Counter A DATAH register */
typedef struct {
  __REG32  CADATAH               :16;
  __REG32                        :16;
} __cadatah_bits;

/* Counter A DATAL register */
typedef struct {
  __REG32  CADATAL               :16;
  __REG32                        :16;
} __cadatal_bits;

/* Clock Source Selection Register */
typedef struct {
  __REG32  CKSRC                 : 2;
  __REG32                        :30;
} __ta_cssr_bits;

/* Clock Source Selection Register */
typedef struct {
  __REG32  CKSRC                 : 1;
  __REG32                        :31;
} __t1_cssr_bits;

/* Clock Enable/Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __ta_cedr_bits;

/* Timer A Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __ta_srr_bits;

/* Timer A Control Set Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  UPDATE                : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  I_LEVEL               : 1;
  __REG32  O_LEVEL               : 1;
  __REG32  KEEPSTATE             : 1;
  __REG32  PWM_INT               : 1;
  __REG32  PWM_EN                : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVERFLOW              : 1;
  __REG32                        : 1;
  __REG32  CAPTURE               : 1;
  __REG32  CAPTURE_TYPE0         : 1;
  __REG32  CAPTURE_TYPE1         : 1;
  __REG32                        : 5;
  __REG32  EXTENSION             : 6;
  __REG32                        : 2;
} __ta_csr_bits;

/* Timer A Control Clear Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  I_LEVEL               : 1;
  __REG32  O_LEVEL               : 1;
  __REG32  KEEPSTATE             : 1;
  __REG32  PWM_INT               : 1;
  __REG32  PWM_EN                : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVERFLOW              : 1;
  __REG32                        : 1;
  __REG32  CAPTURE               : 1;
  __REG32  CAPTURE_TYPE0         : 1;
  __REG32  CAPTURE_TYPE1         : 1;
  __REG32                        : 5;
  __REG32  EXTENSION             : 6;
  __REG32                        : 2;
} __ta_ccr_bits;

/* Interrupt Enable Disable Register */
/* Raw Interrupt Status Register */
/* Masked Interrupt Status Register */
/* Interrupt Clear Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  STOP                  : 1;
  __REG32  P_START               : 1;
  __REG32  P_END                 : 1;
  __REG32  MATCH                 : 1;
  __REG32  OVERFLOW              : 1;
  __REG32  CAPTURE               : 1;
  __REG32                        :25;
} __ta_imscr_bits;

/* Status Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  I_LEVEL               : 1;
  __REG32  O_LEVEL               : 1;
  __REG32  KEEPSTATE             : 1;
  __REG32  PWM_INT               : 1;
  __REG32  PWN_EN                : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVERFLOW              : 1;
  __REG32                        : 1;
  __REG32  CAPTURE               : 1;
  __REG32  CAPTURE_TYPE0         : 1;
  __REG32  CAPTURE_TYPE1         : 1;
  __REG32                        : 5;
  __REG32  EXTENSION             : 6;
  __REG32                        : 2;
} __ta_sr_bits;

/* Clock Divider Register */
typedef struct {
  __REG32  DIVN                  : 4;
  __REG32  DIVM                  :11;
  __REG32                        :17;
} __ta_cdr_bits;

/* Clock Divider Register */
typedef struct {
  __REG32  DIVN                  : 4;
  __REG32  DIVM                  : 8;
  __REG32                        :20;
} __t1_cdr_bits;

/* Counter Size Mask Register */
typedef struct {
  __REG32  SIZE                  : 5;
  __REG32                        :27;
} __ta_csmr_bits;

/* Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        :15;
  __REG32  CKDIV                 : 7;
  __REG32                        : 9;
} __frt_cr_bits;

/* Interrupt Enable Disable Register */
/* Raw Interrupt Status Register */
/* Masked Interrupt Status Register */
/* Interrupt Clear Register */
typedef struct {
  __REG32  OVF32                 : 1;
  __REG32  OVF16                 : 1;
  __REG32  MATCH                 : 1;
  __REG32                        :29;
} __frt_imscr_bits;

/* Status Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __frt_sr_bits;

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

/* UART line control register */
typedef struct {
  __REG32  WL                    : 2;
  __REG32  SB                    : 1;
  __REG32  PMD                   : 3;
  __REG32  IRM                   : 1;
  __REG32                        :25;
} __ulcon_bits;

/* UART control register */
typedef struct {
  __REG32  RM                    : 2;
  __REG32  RSIE                  : 1;
  __REG32  TM                    : 2;
  __REG32                        : 1;
  __REG32  SBS                   : 1;
  __REG32  LBM                   : 1;
  __REG32                        :24;
} __ucon_bits;

/* UART status register */
typedef struct {
  __REG32  OE                    : 1;
  __REG32  PE                    : 1;
  __REG32  FE                    : 1;
  __REG32  BD                    : 1;
  __REG32                        : 1;
  __REG32  RDDR                  : 1;
  __REG32  TBE                   : 1;
  __REG32  TSE                   : 1;
  __REG32                        :24;
} __ustat_bits;

/* UART DMA control register */
typedef struct {
  __REG32  URXDMAE               : 1;
  __REG32  UTXDMAE               : 1;
  __REG32                        :30;
} __udmacr_bits;

/* I2C ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __i2c_idr_bits;

/* I2C Clock Enable Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :31;
} __i2c_cken_bits;

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
  __REG32                        : 4;
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
  __REG32  RST                   : 1;
  __REG32                        : 3;
  __REG32  ISOU                  : 1;
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
  __REG32  D_DIR                 : 1;
  __REG32  CRYSAL_ENA            : 1;
  __REG32                        : 1;
  __REG32  SUSPEND               : 1;
  __REG32  WAKEUP                : 1;
  __REG32  CLK_SEL               : 1;
  __REG32  NAK_CTRL              : 1;
  __REG32  SOF_CTRL              : 2;
  __REG32                        :21;
} __progreg_bits;

/* USB FS Pull-up control register */
typedef struct {
  __REG32  PULLUP                : 1;
  __REG32                        :31;
} __fspullup_bits;

/* A/D converter control register */
typedef struct {
  __REG32  ADC_EN                : 1;
  __REG32  ADC_CLK               : 5;
  __REG32  ADC_EOC               : 1;
  __REG32  ADC_CH_SEL            : 3;
  __REG32  ADCCK_MASK            : 1;
  __REG32  ADC_INT_EN            : 1;
  __REG32  ADC_EN_ADC            : 1;
  __REG32  ADC_STBY              : 1;
  __REG32  ADC_EN_BGR            : 1;
  __REG32                        :17;
} __adccon_bits;

/* A/D converter data register */
typedef struct {
  __REG32  ADDATA                :10;
  __REG32                        :22;
} __addata_bits;

/* A/D DMA control register */
typedef struct {
  __REG32  ADC_DMAE              : 1;
  __REG32                        :31;
} __adc_dmacr_bits;

/* DMA Initial Source Register */
typedef struct {
  __REG32  S_ADDR                :31;
  __REG32                        : 1;
} __disrc_bits;

/* DMA Initial Source Control Register */
typedef struct {
  __REG32  INC                   : 1;
  __REG32                        :31;
} __disrcc_bits;

/* DMA Initial Destination Register */
typedef struct {
  __REG32  D_ADDR                :31;
  __REG32                        : 1;
} __didst_bits;

/* DMA Initial Destination Control Register */
typedef struct {
  __REG32  INC                   : 1;
  __REG32                        :31;
} __didstc_bits;

/* DMA Control Register */
typedef struct {
  __REG32  TC                    :20;
  __REG32  DSZ                   : 2;
  __REG32  RELOAD                : 1;
  __REG32                        : 4;
  __REG32  SERVMODE              : 1;
  __REG32  TSZ                   : 1;
  __REG32  INT                   : 1;
  __REG32                        : 2;
} __dmacon_bits;

/* DMA Count Register */
typedef struct {
  __REG32  CURR_TC               :20;
  __REG32  STAT                  : 2;
  __REG32                        : 2;
  __REG32  MPU_ABORT             : 1;
  __REG32                        : 7;
} __dstat_bits;

/* DMA Current Source Register */
typedef struct {
  __REG32  CURR_SRC              :31;
  __REG32                        : 1;
} __dcsrc_bits;

/* DMA Current Destination Register */
typedef struct {
  __REG32  CURR_DST              :31;
  __REG32                        : 1;
} __dcdst_bits;

/* DMA Mask Trigger Register */
typedef struct {
  __REG32  SW_TRIG               : 1;
  __REG32  ON_OFF                : 1;
  __REG32  STOP                  : 1;
  __REG32                        :29;
} __dmasktrig_bits;

/* DMA Request Selection Register */
typedef struct {
  __REG32  SWHW_SEL              : 1;
  __REG32  HWSRCSEL              : 5;
  __REG32                        :26;
} __dmareqsel_bits;

/* FLASH memory control register */
typedef struct {
  __REG32  ERASE                 : 1;
  __REG32                        : 3;
  __REG32  MODE                  : 4;
  __REG32                        :24;
} __fmcon_bits;

/* FLASH memory control register */
typedef struct {
  __REG32  FMKEY                 : 8;
  __REG32                        :24;
} __fmkey_bits;

/* MPU control register */
typedef struct {
  __REG32  MPUCON                : 1;
  __REG32                        :31;
} __mpucon_bits;

/* MPU Region0 start address */
/* MPU Region0 end address &
   Region1 start address register */
/* MPU Region0 end address &
   Region1 start address register */
/* MPU Region0 end address &
   Region1 start address register */
/* MPU Region3 end address */
/* MPU Region4 start address */
/* MPU Region4 end address */
typedef struct {
  __REG32  REG_ADDR              :14;
  __REG32                        :18;
} __mpustart_r0_bits;

/* MPU control register */
typedef struct {
  __REG32  								       : 1;
  __REG32  SYS_WRITE_ABORT       : 1;
  __REG32  HIGH_TO_LOW_ABORT     : 1;
  __REG32  REGION_WRITE_ABORT    : 1;
  __REG32  MPUSFR_WRITE_ABORT    : 1;
  __REG32  DMA_INVAL_ABORT       : 1;
  __REG32                        :26;
} __mpu_irq_mon_bits;

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
 **  System
 **
 ***************************************************************************/
__IO_REG32_BIT(CLKCON0,         0x40020000,__READ_WRITE ,__clkcon0_bits);
__IO_REG32_BIT(CLKCON1,         0x40020004,__READ_WRITE ,__clkcon1_bits);
__IO_REG32_BIT(PLLPMS,          0x40020008,__READ_WRITE ,__pllpms_bits);
__IO_REG32_BIT(PLLLCNT,         0x4002000C,__READ_WRITE ,__plllcnt_bits);
__IO_REG32_BIT(PLLLOCK,         0x40020010,__READ       ,__plllock_bits);
__IO_REG32_BIT(SWRST,           0x40020014,__READ_WRITE ,__swrst_bits);
__IO_REG32_BIT(PWRCHG,          0x40020018,__READ_WRITE ,__pwrchg_bits);
__IO_REG32_BIT(RESETID,         0x40020020,__READ_WRITE ,__resetid_bits);
__IO_REG32_BIT(LVDCON,          0x40020030,__READ_WRITE ,__lvdcon_bits);
__IO_REG32_BIT(LVDSEL,          0x40020034,__READ_WRITE ,__lvdsel_bits);

/***************************************************************************
 **
 **  Port 0
 **
 ***************************************************************************/
__IO_REG32_BIT(P0CONH,          0x40070000,__READ_WRITE ,__p0conh_bits);
__IO_REG32_BIT(P0CONL,          0x40070004,__READ_WRITE ,__p0conl_bits);
__IO_REG32_BIT(P0EDGE,          0x40070008,__READ_WRITE ,__p0edge_bits);
__IO_REG32_BIT(P0INT,           0x4007000C,__READ_WRITE ,__p0int_bits);
__IO_REG32_BIT(P0PND,           0x40070010,__READ_WRITE ,__p0pnd_bits);
__IO_REG32_BIT(P0PUR,           0x40070014,__READ_WRITE ,__p0pur_bits);
__IO_REG32_BIT(P0DATA,          0x4007001C,__READ_WRITE ,__p0_bits);

/***************************************************************************
 **
 **  Port 1
 **
 ***************************************************************************/
__IO_REG32_BIT(P1CONH,          0x40070100,__READ_WRITE ,__p1conh_bits);
__IO_REG32_BIT(P1CONL,          0x40070104,__READ_WRITE ,__p1conl_bits);
__IO_REG32_BIT(P1EDGE,          0x40070108,__READ_WRITE ,__p1edge_bits);
__IO_REG32_BIT(P1INT,           0x4007010C,__READ_WRITE ,__p1int_bits);
__IO_REG32_BIT(P1PND,           0x40070110,__READ_WRITE ,__p1pnd_bits);
__IO_REG32_BIT(P1PUR,           0x40070114,__READ_WRITE ,__p1pur_bits);
__IO_REG32_BIT(P1DATA,          0x4007011C,__READ_WRITE ,__p1_bits);

/***************************************************************************
 **
 **  Port 2
 **
 ***************************************************************************/
__IO_REG32_BIT(P2CONH,          0x40070200,__READ_WRITE ,__p2conh_bits);
__IO_REG32_BIT(P2CONL,          0x40070204,__READ_WRITE ,__p2conl_bits);
__IO_REG32_BIT(P2EDGE,          0x40070208,__READ_WRITE ,__p2edge_bits);
__IO_REG32_BIT(P2INT,           0x4007020C,__READ_WRITE ,__p2int_bits);
__IO_REG32_BIT(P2PND,           0x40070210,__READ_WRITE ,__p2pnd_bits);
__IO_REG32_BIT(P2PUR,           0x40070214,__READ_WRITE ,__p2pur_bits);
__IO_REG32_BIT(P2DATA,          0x4007021C,__READ_WRITE ,__p2_bits);

/***************************************************************************
 **
 **  Port 3
 **
 ***************************************************************************/
__IO_REG32_BIT(P3CONH,          0x40070300,__READ_WRITE ,__p3conh_bits);
__IO_REG32_BIT(P3CONL,          0x40070304,__READ_WRITE ,__p3conl_bits);
__IO_REG32_BIT(P3PUR,           0x40070314,__READ_WRITE ,__p3pur_bits);
__IO_REG32_BIT(P3DATA,          0x4007031C,__READ_WRITE ,__p3_bits);

/***************************************************************************
 **
 **  Port 4
 **
 ***************************************************************************/
__IO_REG32_BIT(P4CONH,          0x40070400,__READ_WRITE ,__p4conh_bits);
__IO_REG32_BIT(P4CONL,          0x40070404,__READ_WRITE ,__p4conl_bits);
__IO_REG32_BIT(P4EDGE,          0x40070408,__READ_WRITE ,__p4edge_bits);
__IO_REG32_BIT(P4INT,           0x4007040C,__READ_WRITE ,__p4int_bits);
__IO_REG32_BIT(P4PND,           0x40070410,__READ_WRITE ,__p4pnd_bits);
__IO_REG32_BIT(P4PUR,           0x40070414,__READ_WRITE ,__p4pur_bits);
__IO_REG32_BIT(P4DATA,          0x4007041C,__READ_WRITE ,__p4_bits);

/***************************************************************************
 **
 **  Port 5
 **
 ***************************************************************************/
__IO_REG32_BIT(P5CONH,          0x40070500,__READ_WRITE ,__p5conh_bits);
__IO_REG32_BIT(P5CONL,          0x40070504,__READ_WRITE ,__p5conl_bits);
__IO_REG32_BIT(P5PUR,           0x40070514,__READ_WRITE ,__p5pur_bits);
__IO_REG32_BIT(P5MODE,          0x40070518,__READ_WRITE ,__p5mode_bits);
__IO_REG32_BIT(P5DATA,          0x4007051C,__READ_WRITE ,__p5_bits);

/***************************************************************************
 **
 **  Port 6
 **
 ***************************************************************************/
__IO_REG32_BIT(P6CONL,          0x40070604,__READ_WRITE ,__p6conl_bits);
__IO_REG32_BIT(P6PUR,           0x40070614,__READ_WRITE ,__p6pur_bits);
__IO_REG32_BIT(P6DATA,          0x4007061C,__READ_WRITE ,__p6_bits);

/***************************************************************************
 **
 **  BT/WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(BTCON,           0x40050000,__READ_WRITE ,__btcon_bits);
__IO_REG32_BIT(BTCNT,           0x40050004,__READ       ,__btcnt_bits);
__IO_REG32_BIT(WDTCNT,          0x40050008,__READ       ,__wdtcnt_bits);

/***************************************************************************
 **
 **  Counter A
 **
 ***************************************************************************/
__IO_REG32_BIT(CADATAH,         0x400C0000,__READ_WRITE ,__cadatah_bits);
__IO_REG32_BIT(CADATAL,         0x400C0004,__READ_WRITE ,__cadatal_bits);
__IO_REG32_BIT(CACON1,          0x400C0008,__READ_WRITE ,__cacon1_bits);
__IO_REG32_BIT(CACON0,          0x400C000C,__READ_WRITE ,__cacon0_bits);

/***************************************************************************
 **
 **  Timer A
 **
 ***************************************************************************/
__IO_REG32(    TA_IDR,          0x400B0000,__READ       );
__IO_REG32_BIT(TA_CSSR,         0x400B0004,__READ_WRITE ,__ta_cssr_bits);
__IO_REG32_BIT(TA_CEDR,         0x400B0008,__READ_WRITE ,__ta_cedr_bits);
__IO_REG32_BIT(TA_SRR,          0x400B000C,__WRITE      ,__ta_srr_bits);
__IO_REG32_BIT(TA_CSR,          0x400B0010,__WRITE      ,__ta_csr_bits);
__IO_REG32_BIT(TA_CCR,          0x400B0014,__WRITE      ,__ta_ccr_bits);
__IO_REG32_BIT(TA_SR,           0x400B0018,__READ       ,__ta_sr_bits);
__IO_REG32_BIT(TA_IMSCR,        0x400B001C,__READ_WRITE ,__ta_imscr_bits);
__IO_REG32_BIT(TA_RISR,         0x400B0020,__READ       ,__ta_imscr_bits);
__IO_REG32_BIT(TA_MISR,         0x400B0024,__READ       ,__ta_imscr_bits);
__IO_REG32_BIT(TA_ICR,          0x400B0028,__WRITE      ,__ta_imscr_bits);
__IO_REG32_BIT(TA_CDR,          0x400B002C,__READ_WRITE ,__ta_cdr_bits);
__IO_REG32_BIT(TA_CSMR,         0x400B0030,__READ_WRITE ,__ta_csmr_bits);
__IO_REG32(    TA_PRDR,         0x400B0034,__READ_WRITE );
__IO_REG32(    TA_PULR,         0x400B0038,__READ_WRITE );
__IO_REG32_BIT(TA_UCDR,         0x400B003C,__READ       ,__ta_cdr_bits);
__IO_REG32(    TA_UCSMR,        0x400B0040,__READ       );
__IO_REG32(    TA_UPRDR,        0x400B0044,__READ       );
__IO_REG32(    TA_UPULR,        0x400B0048,__READ       );
__IO_REG32(    TA_CUCR,         0x400B004C,__READ       );
__IO_REG32(    TA_CDCR,         0x400B0050,__READ       );
__IO_REG32(    TA_CVR,          0x400B0054,__READ       );

/***************************************************************************
 **
 **  Timer 1
 **
 ***************************************************************************/
__IO_REG32(    T1_IDR,          0x40080000,__READ       );
__IO_REG32_BIT(T1_CSSR,         0x40080004,__READ_WRITE ,__t1_cssr_bits);
__IO_REG32_BIT(T1_CEDR,         0x40080008,__READ_WRITE ,__ta_cedr_bits);
__IO_REG32_BIT(T1_SRR,          0x4008000C,__WRITE      ,__ta_srr_bits);
__IO_REG32_BIT(T1_CSR,          0x40080010,__WRITE      ,__ta_csr_bits);
__IO_REG32_BIT(T1_CCR,          0x40080014,__WRITE      ,__ta_ccr_bits);
__IO_REG32_BIT(T1_SR,           0x40080018,__READ       ,__ta_sr_bits);
__IO_REG32_BIT(T1_IMSCR,        0x4008001C,__READ_WRITE ,__ta_imscr_bits);
__IO_REG32_BIT(T1_RISR,         0x40080020,__READ       ,__ta_imscr_bits);
__IO_REG32_BIT(T1_MISR,         0x40080024,__READ       ,__ta_imscr_bits);
__IO_REG32_BIT(T1_ICR,          0x40080028,__WRITE      ,__ta_imscr_bits);
__IO_REG32_BIT(T1_CDR,          0x4008002C,__READ_WRITE ,__t1_cdr_bits);
__IO_REG32_BIT(T1_CSMR,         0x40080030,__READ_WRITE ,__ta_csmr_bits);
__IO_REG32(    T1_PRDR,         0x40080034,__READ_WRITE );
__IO_REG32(    T1_PULR,         0x40080038,__READ_WRITE );
__IO_REG32_BIT(T1_UCDR,         0x4008003C,__READ       ,__ta_cdr_bits);
__IO_REG32(    T1_UCSMR,        0x40080040,__READ       );
__IO_REG32(    T1_UPRDR,        0x40080044,__READ       );
__IO_REG32(    T1_UPULR,        0x40080048,__READ       );
__IO_REG32(    T1_CUCR,         0x4008004C,__READ       );
__IO_REG32(    T1_CDCR,         0x40080050,__READ       );
__IO_REG32(    T1_CVR,          0x40080054,__READ       );

/***************************************************************************
 **
 **  Timer 2
 **
 ***************************************************************************/
__IO_REG32(    T2_IDR,          0x40090000,__READ       );
__IO_REG32_BIT(T2_CSSR,         0x40090004,__READ_WRITE ,__t1_cssr_bits);
__IO_REG32_BIT(T2_CEDR,         0x40090008,__READ_WRITE ,__ta_cedr_bits);
__IO_REG32_BIT(T2_SRR,          0x4009000C,__WRITE      ,__ta_srr_bits);
__IO_REG32_BIT(T2_CSR,          0x40090010,__WRITE      ,__ta_csr_bits);
__IO_REG32_BIT(T2_CCR,          0x40090014,__WRITE      ,__ta_ccr_bits);
__IO_REG32_BIT(T2_SR,           0x40090018,__READ       ,__ta_sr_bits);
__IO_REG32_BIT(T2_IMSCR,        0x4009001C,__READ_WRITE ,__ta_imscr_bits);
__IO_REG32_BIT(T2_RISR,         0x40090020,__READ       ,__ta_imscr_bits);
__IO_REG32_BIT(T2_MISR,         0x40090024,__READ       ,__ta_imscr_bits);
__IO_REG32_BIT(T2_ICR,          0x40090028,__WRITE      ,__ta_imscr_bits);
__IO_REG32_BIT(T2_CDR,          0x4009002C,__READ_WRITE ,__t1_cdr_bits);
__IO_REG32_BIT(T2_CSMR,         0x40090030,__READ_WRITE ,__ta_csmr_bits);
__IO_REG32(    T2_PRDR,         0x40090034,__READ_WRITE );
__IO_REG32(    T2_PULR,         0x40090038,__READ_WRITE );
__IO_REG32_BIT(T2_UCDR,         0x4009003C,__READ       ,__ta_cdr_bits);
__IO_REG32(    T2_UCSMR,        0x40090040,__READ       );
__IO_REG32(    T2_UPRDR,        0x40090044,__READ       );
__IO_REG32(    T2_UPULR,        0x40090048,__READ       );
__IO_REG32(    T2_CUCR,         0x4009004C,__READ       );
__IO_REG32(    T2_CDCR,         0x40090050,__READ       );
__IO_REG32(    T2_CVR,          0x40090054,__READ       );

/***************************************************************************
 **
 **  Free Running Timer
 **
 ***************************************************************************/
__IO_REG32(    FRT_IDR,         0x400A0000,__READ       );
__IO_REG32_BIT(FRT_CEDR,        0x400A0004,__READ_WRITE ,__ta_cedr_bits);
__IO_REG32_BIT(FRT_SRR,         0x400A0008,__WRITE      ,__ta_srr_bits);
__IO_REG32_BIT(FRT_CR,          0x400A000C,__READ_WRITE ,__frt_cr_bits);
__IO_REG32_BIT(FRT_SR,          0x400A0010,__READ       ,__frt_sr_bits);
__IO_REG32_BIT(FRT_IMSCR,       0x400A0014,__READ_WRITE ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_RISR,        0x400A0018,__READ       ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_MISR,        0x400A001C,__READ       ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_ICR,         0x400A0020,__WRITE      ,__frt_imscr_bits);
__IO_REG32(    FRT_DR,          0x400A0024,__READ_WRITE );
__IO_REG32(    FRT_DBR,         0x400A0028,__READ       );
__IO_REG32(    FRT_CVR,         0x400A002C,__READ       );

/***************************************************************************
 **
 **  SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CR0,        0x400E0000,__READ_WRITE ,__spi_cr0_bits);
__IO_REG32_BIT(SPI0_CR1,        0x400E0004,__READ_WRITE ,__spi_cr1_bits);
__IO_REG16(    SPI0_DR,         0x400E0008,__READ_WRITE );
__IO_REG32_BIT(SPI0_SR,         0x400E000C,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI0_CPSR,       0x400E0010,__READ_WRITE ,__spi_cpsr_bits);
__IO_REG32_BIT(SPI0_IMSC,       0x400E0014,__READ_WRITE ,__spi_imsc_bits);
__IO_REG32_BIT(SPI0_RISR,       0x400E0018,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI0_MISR,       0x400E001C,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI0_ICR,        0x400E0020,__WRITE      ,__spi_icr_bits);
__IO_REG32_BIT(SPI0_DMACR,      0x400E0024,__READ_WRITE ,__spi_dmacr_bits);

/***************************************************************************
 **
 **  SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CR0,        0x400E1000,__READ_WRITE ,__spi_cr0_bits);
__IO_REG32_BIT(SPI1_CR1,        0x400E1004,__READ_WRITE ,__spi_cr1_bits);
__IO_REG16(    SPI1_DR,         0x400E1008,__READ_WRITE );
__IO_REG32_BIT(SPI1_SR,         0x400E100C,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI1_CPSR,       0x400E1010,__READ_WRITE ,__spi_cpsr_bits);
__IO_REG32_BIT(SPI1_IMSC,       0x400E1014,__READ_WRITE ,__spi_imsc_bits);
__IO_REG32_BIT(SPI1_RISR,       0x400E1018,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI1_MISR,       0x400E101C,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI1_ICR,        0x400E1020,__WRITE      ,__spi_icr_bits);
__IO_REG32_BIT(SPI1_DMACR,      0x400E1024,__READ_WRITE ,__spi_dmacr_bits);

/***************************************************************************
 **
 **  UART
 **
 ***************************************************************************/
__IO_REG32_BIT(ULCON,           0x400D0000,__READ_WRITE ,__ulcon_bits);
__IO_REG32_BIT(UCON,            0x400D0004,__READ_WRITE ,__ucon_bits);
__IO_REG32_BIT(USTAT,           0x400D0008,__READ       ,__ustat_bits);
__IO_REG8(     UTXH,            0x400D000C,__WRITE      );
__IO_REG8(     URXH,            0x400D0010,__READ       );
__IO_REG16(    UBRDIV,          0x400D0014,__READ_WRITE );
__IO_REG32_BIT(UDMACR,          0x400D0020,__READ_WRITE ,__udmacr_bits);

/***************************************************************************
 **
 **  I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_IDR,        0x400F0000,__READ       ,__i2c_idr_bits);
__IO_REG32_BIT(I2C0_CKEN,       0x400F0004,__READ_WRITE ,__i2c_cken_bits);
__IO_REG32_BIT(I2C0_SRR,        0x400F0008,__WRITE      ,__i2c_srr_bits);
__IO_REG32_BIT(I2C0_CR,         0x400F000C,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C0_MR,         0x400F0010,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C0_SR,         0x400F0014,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C0_IMSCR,      0x400F0018,__READ_WRITE ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_RISR,       0x400F001C,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_MISR,       0x400F0020,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_ICR,        0x400F0024,__WRITE      ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_SDR,        0x400F0028,__READ_WRITE ,__i2c_sdr_bits);
__IO_REG32_BIT(I2C0_SSAR,       0x400F002C,__READ_WRITE ,__i2c_ssar_bits);
__IO_REG32_BIT(I2C0_HSDR,       0x400F0030,__READ_WRITE ,__i2c_hsdr_bits);
__IO_REG32_BIT(I2C0_DMACR,      0x400F0034,__READ_WRITE ,__i2c_dmacr_bits);

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_IDR,        0x400F1000,__READ       ,__i2c_idr_bits);
__IO_REG32_BIT(I2C1_CKEN,       0x400F1004,__READ_WRITE ,__i2c_cken_bits);
__IO_REG32_BIT(I2C1_SRR,        0x400F1008,__WRITE      ,__i2c_srr_bits);
__IO_REG32_BIT(I2C1_CR,         0x400F100C,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C1_MR,         0x400F1010,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C1_SR,         0x400F1014,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C1_IMSCR,      0x400F1018,__READ_WRITE ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_RISR,       0x400F101C,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_MISR,       0x400F1020,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_ICR,        0x400F1024,__WRITE      ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_SDR,        0x400F1028,__READ_WRITE ,__i2c_sdr_bits);
__IO_REG32_BIT(I2C1_SSAR,       0x400F102C,__READ_WRITE ,__i2c_ssar_bits);
__IO_REG32_BIT(I2C1_HSDR,       0x400F1030,__READ_WRITE ,__i2c_hsdr_bits);
__IO_REG32_BIT(I2C1_DMACR,      0x400F1034,__READ_WRITE ,__i2c_dmacr_bits);

/***************************************************************************
 **
 **  USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USBFA,           0x40100000,__READ_WRITE ,__usbfa_bits);
__IO_REG32_BIT(USBPM,           0x40100004,__READ_WRITE ,__usbpm_bits);
__IO_REG32_BIT(USBINTMON,       0x40100008,__READ_WRITE ,__usbintmon_bits);
__IO_REG32_BIT(USBINTCON,       0x4010000C,__READ_WRITE ,__usbintcon_bits);
__IO_REG32_BIT(USBFN,           0x40100010,__READ       ,__usbfn_bits);
__IO_REG32_BIT(USBEPLNUM,       0x40100014,__READ_WRITE ,__usbeplnum_bits);
__IO_REG32_BIT(USBEP0CSR,       0x40100020,__READ_WRITE ,__usbep0csr_bits);
__IO_REG32_BIT(USBEP1CSR,       0x40100024,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP2CSR,       0x40100028,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP3CSR,       0x4010002C,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP4CSR,       0x40100030,__READ_WRITE ,__usbepcsr_bits);
__IO_REG32_BIT(USBEP0WC,        0x40100040,__READ_WRITE ,__usbep0wc_bits);
__IO_REG32_BIT(USBEP1WC,        0x40100044,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBEP2WC,        0x40100048,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBEP3WC,        0x4010004C,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBEP4WC,        0x40100050,__READ_WRITE ,__usbepwc_bits);
__IO_REG32_BIT(USBNAKCON1,      0x40100060,__READ_WRITE ,__usbnakcon1_bits);
__IO_REG32_BIT(USBNAKCON2,      0x40100064,__READ_WRITE ,__usbnakcon2_bits);
__IO_REG32_BIT(USBEP0,          0x40100080,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP1,          0x40100084,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP2,          0x40100088,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP3,          0x4010008C,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(USBEP4,          0x40100090,__READ_WRITE ,__usbep_bits);
__IO_REG32_BIT(PROGREG,         0x401000A0,__READ_WRITE ,__progreg_bits);
__IO_REG32_BIT(FSPULLUP,        0x401000B4,__READ_WRITE ,__fspullup_bits);

/***************************************************************************
 **
 **  ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADCCON,          0x40060000,__READ_WRITE ,__adccon_bits);
__IO_REG32_BIT(ADDATA,          0x40060004,__READ       ,__addata_bits);
__IO_REG32_BIT(ADC_DMACR,       0x40060008,__READ_WRITE ,__adc_dmacr_bits);

/***************************************************************************
 **
 **  DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DISRC,           0x40030000,__READ_WRITE ,__disrc_bits);
__IO_REG32_BIT(DISRCC,          0x40030004,__READ_WRITE ,__disrcc_bits);
__IO_REG32_BIT(DIDST,           0x40030008,__READ_WRITE ,__didst_bits);
__IO_REG32_BIT(DIDSTC,          0x4003000C,__READ_WRITE ,__didstc_bits);
__IO_REG32_BIT(DMACON,          0x40030010,__READ_WRITE ,__dmacon_bits);
__IO_REG32_BIT(DSTAT,           0x40030014,__READ       ,__dstat_bits);
__IO_REG32_BIT(DCSRC,           0x40030018,__READ       ,__dcsrc_bits);
__IO_REG32_BIT(DCDST,           0x4003001C,__READ       ,__dcdst_bits);
__IO_REG32_BIT(DMASKTRIG,       0x40030020,__READ_WRITE ,__dmasktrig_bits);
__IO_REG32_BIT(DMAREQSEL,       0x40030024,__READ_WRITE ,__dmareqsel_bits);

/***************************************************************************
 **
 **  FLASH
 **
 ***************************************************************************/
__IO_REG32_BIT(FMCON,           0x40010000,__READ_WRITE ,__fmcon_bits);
__IO_REG32_BIT(FMKEY,           0x40010004,__READ_WRITE ,__fmkey_bits);
__IO_REG32(    FMADDR,          0x40010008,__READ_WRITE );

/***************************************************************************
 **
 **  MPU
 **
 ***************************************************************************/
__IO_REG32_BIT(MPUCON,          0x40040000,__READ_WRITE ,__mpucon_bits);
__IO_REG32_BIT(MPUSTART_R0,     0x40040010,__READ       ,__mpustart_r0_bits);
__IO_REG32_BIT(MPUEND_R0,       0x40040014,__READ_WRITE ,__mpustart_r0_bits);
__IO_REG32_BIT(MPUEND_R1,       0x40040018,__READ_WRITE ,__mpustart_r0_bits);
__IO_REG32_BIT(MPUEND_R2,       0x4004001C,__READ_WRITE ,__mpustart_r0_bits);
__IO_REG32_BIT(MPUEND_R3,       0x40040020,__READ_WRITE ,__mpustart_r0_bits);
__IO_REG32_BIT(MPUSTART_R4,     0x40040024,__READ       ,__mpustart_r0_bits);
__IO_REG32_BIT(MPUEND_R4,       0x40040028,__READ_WRITE ,__mpustart_r0_bits);
__IO_REG32_BIT(MPU_IRQ_MON,     0x400400A0,__READ_WRITE ,__mpu_irq_mon_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  S3FN60D DMA channels number
 **
 ***************************************************************************/
#define UART_TX_DMA       0x00
#define UART_RX_DMA       0x01
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
 **  S3FN60D interrupt source number
 **
 ***************************************************************************/
#define MPUINT            0x00    /* MPU abort                                  */
#define TAINT             0x01    /* Timer A P_END / MATCH / OVERFLOW / CAPTURE */
#define T1INT             0x02    /* Timer 1 P_END / MATCH / OVERFLOW / CAPTURE */
#define SPI0INT           0x03    /* SPI0 interrupt                             */
#define SPI1INT           0x04    /* SPI1 interrupt                             */
#define I2C0INT           0x05    /* I2C0 interrupt                             */
#define I2C1INT           0x06    /* I2C0 interrupt                             */
#define BTOVF             0x07    /* Basic timer overflow                       */
#define ADC_INT           0x08    /* ADC interrupt                              */
#define UARTTXINT         0x09    /* UART data transmit interrupt               */
#define UARTRXINT         0x0a    /* UART data receive interrupt                */
#define UARTERRINT        0x0b    /* UART data error interrupt                  */
#define T2INT             0x0c    /* Timer 2 P_END / MATCH / OVERFLOW / CAPTURE */
#define FRTINT            0x0d    /* FRT MATCH / OVF16 / OVF32                  */
#define DMAINT            0x0e    /* DMA interrupt                              */
#define VUSB_DETINTR      0x0f    /* VBUS detect                                */
#define INT0              0x10    /* External pin interrupt P0.0 / P0.1         */
#define INT1              0x11    /* External pin interrupt P0.2 / P0.3         */
#define INT2              0x12    /* External pin interrupt P0.4 / P0.5         */
#define INT3              0x13    /* External pin interrupt P0.6 / P0.7         */
#define INT4              0x14    /* External pin interrupt P2.0 / P2.1         */
#define INT5              0x15    /* External pin interrupt P2.2 / P2.3         */
#define INT6              0x16    /* External pin interrupt P2.4 / P2.5         */
#define INT7              0x17    /* External pin interrupt P2.6 / P2.7         */
#define INT8              0x18    /* External pin interrupt P1.0 / P1.1 / P1.2 / P1.3 */
#define INT9              0x19    /* External pin interrupt P1.4 / P1.4 / P1.6 / P1.7 */
#define INT10             0x1a    /* External pin interrupt P4.0 / P4.1 / P4.2 / P4.3 */
#define INT11             0x1b    /* External pin interrupt P4.4 / P4.5 / P4.6 / P4.7 */
#define CAINT             0x1c    /* Counter A interrupt                        */
#define USBINT            0x1d    /* USB Resume/Suspend/Reset interrupt         */
#define USBSOF            0x1e    /* USB SOF interrupt                          */
#define USBEP             0x1f    /* USB Endpoint 0 / 1 / 2 / 3 /4 interrupt    */

#endif    /* __S3FN60D_H */

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
Interrupt9   = MPUINT         0x40
Interrupt10  = TAINT          0x44
Interrupt11  = T1INT          0x48
Interrupt12  = SPI0INT        0x4C
Interrupt13  = SPI1INT        0x50
Interrupt14  = I2C0INT        0x54
Interrupt15  = I2C1INT        0x58
Interrupt16  = BTOVF          0x5C
Interrupt17  = ADC_INT        0x60
Interrupt18  = UARTTXINT      0x64
Interrupt19  = UARTRXINT      0x68
Interrupt20  = UARTERRINT     0x6C
Interrupt21  = T2INT          0x70
Interrupt22  = FRTINT         0x74
Interrupt23  = DMAINT         0x78
Interrupt23  = VUSB_DETINTR   0x7C
Interrupt24  = INT0           0x80
Interrupt25  = INT1           0x84
Interrupt26  = INT2           0x88
Interrupt27  = INT3           0x8C
Interrupt28  = INT4           0x90
Interrupt29  = INT5           0x94
Interrupt30  = INT6           0x98
Interrupt31  = INT7           0x9C
Interrupt32  = INT8           0xA0
Interrupt33  = INT9           0xA4
Interrupt34  = INT10          0xA8
Interrupt35  = INT11          0xAC
Interrupt37  = CAINT          0xB0
Interrupt38  = USBINT         0xB4
Interrupt39  = USBSOF         0xB8
Interrupt40  = USBEP          0xBC
###DDF-INTERRUPT-END###*/
