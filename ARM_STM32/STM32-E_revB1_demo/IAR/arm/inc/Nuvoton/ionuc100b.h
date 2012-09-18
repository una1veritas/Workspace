/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton NUC100 Advance Line Low Density devices
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2012
 **
 **    $Revision: 52497 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/
#ifndef __IONUC100B_H
#define __IONUC100B_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   NUC100B SPECIAL FUNCTION REGISTERS
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

/* System Control Register  */
typedef struct {
  __REG32                 : 1;
  __REG32  SLEEPONEXIT    : 1;
  __REG32  SLEEPDEEP      : 1;
  __REG32                 : 1;
  __REG32  SEVONPEND      : 1;
  __REG32                 :27;
} __scr_bits;

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

/* System Reset Source Register (RSTSRC) */
typedef struct{
  __REG32 RSTS_POR        : 1;
  __REG32 RSTS_PAD        : 1;
  __REG32 RSTS_WDG        : 1;
  __REG32 RSTS_LVR        : 1;
  __REG32 RSTS_BOD        : 1;
  __REG32 RSTS_SYS        : 1;
  __REG32                 : 1;
  __REG32 RSTS_CPU        : 1;
  __REG32                 :24;
} __rstsrc_bits;

/* IP Reset Control Register1 (IPRSTC1) */
typedef struct{
  __REG32 CHIP_RST        : 1;
  __REG32 CPU_RST         : 1;
  __REG32 PDMA_RST        : 1;
  __REG32                 :29;
} __iprstc1_bits;

/* IP Reset Control Register2 (IPRSTC2) */
typedef struct{
  __REG32                 : 1;
  __REG32 GPIO_RST        : 1;
  __REG32 TMR0_RST        : 1;
  __REG32 TMR1_RST        : 1;
  __REG32 TMR2_RST        : 1;
  __REG32 TMR3_RST        : 1;
  __REG32                 : 2;
  __REG32 I2C0_RST        : 1;
  __REG32 I2C1_RST        : 1;
  __REG32                 : 2;
  __REG32 SPI0_RST        : 1;
  __REG32 SPI1_RST        : 1;
  __REG32                 : 2;
  __REG32 UART0_RST       : 1;
  __REG32 UART1_RST       : 1;
  __REG32                 : 2;
  __REG32 PWM_RST03       : 1;
  __REG32                 : 1;
  __REG32 ACMP_RST        : 1;
  __REG32 PS2_RST         : 1;
  __REG32                 : 4;
  __REG32 ADC_RST         : 1;
  __REG32 I2S_RST         : 1;
  __REG32                 : 2;
} __iprstc2_bits;

/* Brown-Out Detector Control Register (BODCR) */
typedef struct{
  __REG32 BOD_EN          : 1;
  __REG32 BOD_VL          : 2;
  __REG32 BOD_RSTEN       : 1;
  __REG32 BOD_INTF        : 1;
  __REG32 BOD_LPM         : 1;
  __REG32 BOD_OUT         : 1;
  __REG32 LVR_EN          : 1;
  __REG32                 :24;
} __bodcr_bits;

/* Temperature Sensor Control Register (TEMPCR) */
typedef struct{
  __REG32 VTEMP_EN        : 1;
  __REG32                 :31;
} __tempcr_bits;

/* Power-On-Reset Control Register (PORCR) */
typedef struct{
  __REG32 POR_DIS_CODE    :16;
  __REG32                 :16;
} __porcr_bits;

/* Multiple Function Pin GPIOA Control Register (GPA_MFP) */
typedef struct{
  __REG32 GPA_MFP0        : 1;
  __REG32 GPA_MFP1        : 1;
  __REG32 GPA_MFP2        : 1;
  __REG32 GPA_MFP3        : 1;
  __REG32 GPA_MFP4        : 1;
  __REG32 GPA_MFP5        : 1;
  __REG32 GPA_MFP6        : 1;
  __REG32 GPA_MFP7        : 1;
  __REG32 GPA_MFP8        : 1;
  __REG32 GPA_MFP9        : 1;
  __REG32 GPA_MFP10       : 1;
  __REG32 GPA_MFP11       : 1;
  __REG32 GPA_MFP12       : 1;
  __REG32 GPA_MFP13       : 1;
  __REG32 GPA_MFP14       : 1;
  __REG32 GPA_MFP15       : 1;
  __REG32 GPA_TYPE0       : 1;
  __REG32 GPA_TYPE1       : 1;
  __REG32 GPA_TYPE2       : 1;
  __REG32 GPA_TYPE3       : 1;
  __REG32 GPA_TYPE4       : 1;
  __REG32 GPA_TYPE5       : 1;
  __REG32 GPA_TYPE6       : 1;
  __REG32 GPA_TYPE7       : 1;
  __REG32 GPA_TYPE8       : 1;
  __REG32 GPA_TYPE9       : 1;
  __REG32 GPA_TYPE10      : 1;
  __REG32 GPA_TYPE11      : 1;
  __REG32 GPA_TYPE12      : 1;
  __REG32 GPA_TYPE13      : 1;
  __REG32 GPA_TYPE14      : 1;
  __REG32 GPA_TYPE15      : 1;
} __gpa_mfp_bits;

/* Multiple Function Pin GPIOB Control Register (GPB_MFP) */
typedef struct{
  __REG32 GPB_MFP0        : 1;
  __REG32 GPB_MFP1        : 1;
  __REG32 GPB_MFP2        : 1;
  __REG32 GPB_MFP3        : 1;
  __REG32 GPB_MFP4        : 1;
  __REG32 GPB_MFP5        : 1;
  __REG32 GPB_MFP6        : 1;
  __REG32 GPB_MFP7        : 1;
  __REG32 GPB_MFP8        : 1;
  __REG32 GPB_MFP9        : 1;
  __REG32 GPB_MFP10       : 1;
  __REG32 GPB_MFP11       : 1;
  __REG32 GPB_MFP12       : 1;
  __REG32 GPB_MFP13       : 1;
  __REG32 GPB_MFP14       : 1;
  __REG32 GPB_MFP15       : 1;
  __REG32 GPB_TYPE0       : 1;
  __REG32 GPB_TYPE1       : 1;
  __REG32 GPB_TYPE2       : 1;
  __REG32 GPB_TYPE3       : 1;
  __REG32 GPB_TYPE4       : 1;
  __REG32 GPB_TYPE5       : 1;
  __REG32 GPB_TYPE6       : 1;
  __REG32 GPB_TYPE7       : 1;
  __REG32 GPB_TYPE8       : 1;
  __REG32 GPB_TYPE9       : 1;
  __REG32 GPB_TYPE10      : 1;
  __REG32 GPB_TYPE11      : 1;
  __REG32 GPB_TYPE12      : 1;
  __REG32 GPB_TYPE13      : 1;
  __REG32 GPB_TYPE14      : 1;
  __REG32 GPB_TYPE15      : 1;
} __gpb_mfp_bits;

/* Multiple Function Pin GPIOC Control Register (GPA_MFP) */
typedef struct{
  __REG32 GPC_MFP0        : 1;
  __REG32 GPC_MFP1        : 1;
  __REG32 GPC_MFP2        : 1;
  __REG32 GPC_MFP3        : 1;
  __REG32 GPC_MFP4        : 1;
  __REG32 GPC_MFP5        : 1;
  __REG32 GPC_MFP6        : 1;
  __REG32 GPC_MFP7        : 1;
  __REG32 GPC_MFP8        : 1;
  __REG32 GPC_MFP9        : 1;
  __REG32 GPC_MFP10       : 1;
  __REG32 GPC_MFP11       : 1;
  __REG32 GPC_MFP12       : 1;
  __REG32 GPC_MFP13       : 1;
  __REG32 GPC_MFP14       : 1;
  __REG32 GPC_MFP15       : 1;
  __REG32 GPC_TYPE0       : 1;
  __REG32 GPC_TYPE1       : 1;
  __REG32 GPC_TYPE2       : 1;
  __REG32 GPC_TYPE3       : 1;
  __REG32 GPC_TYPE4       : 1;
  __REG32 GPC_TYPE5       : 1;
  __REG32 GPC_TYPE6       : 1;
  __REG32 GPC_TYPE7       : 1;
  __REG32 GPC_TYPE8       : 1;
  __REG32 GPC_TYPE9       : 1;
  __REG32 GPC_TYPE10      : 1;
  __REG32 GPC_TYPE11      : 1;
  __REG32 GPC_TYPE12      : 1;
  __REG32 GPC_TYPE13      : 1;
  __REG32 GPC_TYPE14      : 1;
  __REG32 GPC_TYPE15      : 1;
} __gpc_mfp_bits;

/* Multiple Function Pin GPIOD Control Register (GPD_MFP) */
typedef struct{
  __REG32 GPD_MFP0        : 1;
  __REG32 GPD_MFP1        : 1;
  __REG32 GPD_MFP2        : 1;
  __REG32 GPD_MFP3        : 1;
  __REG32 GPD_MFP4        : 1;
  __REG32 GPD_MFP5        : 1;
  __REG32 GPD_MFP6        : 1;
  __REG32 GPD_MFP7        : 1;
  __REG32 GPD_MFP8        : 1;
  __REG32 GPD_MFP9        : 1;
  __REG32 GPD_MFP10       : 1;
  __REG32 GPD_MFP11       : 1;
  __REG32 GPD_MFP12       : 1;
  __REG32 GPD_MFP13       : 1;
  __REG32 GPD_MFP14       : 1;
  __REG32 GPD_MFP15       : 1;
  __REG32 GPD_TYPE0       : 1;
  __REG32 GPD_TYPE1       : 1;
  __REG32 GPD_TYPE2       : 1;
  __REG32 GPD_TYPE3       : 1;
  __REG32 GPD_TYPE4       : 1;
  __REG32 GPD_TYPE5       : 1;
  __REG32 GPD_TYPE6       : 1;
  __REG32 GPD_TYPE7       : 1;
  __REG32 GPD_TYPE8       : 1;
  __REG32 GPD_TYPE9       : 1;
  __REG32 GPD_TYPE10      : 1;
  __REG32 GPD_TYPE11      : 1;
  __REG32 GPD_TYPE12      : 1;
  __REG32 GPD_TYPE13      : 1;
  __REG32 GPD_TYPE14      : 1;
  __REG32 GPD_TYPE15      : 1;
} __gpd_mfp_bits;

/* Multiple Function Pin GPIOE Control Register (GPE_MFP) */
typedef struct{
  __REG32 GPE_MFP0        : 1;
  __REG32 GPE_MFP1        : 1;
  __REG32                 : 3;
  __REG32 GPE_MFP5        : 1;
  __REG32                 :10;
  __REG32 GPE_TYPE0       : 1;
  __REG32 GPE_TYPE1       : 1;
  __REG32 GPE_TYPE2       : 1;
  __REG32 GPE_TYPE3       : 1;
  __REG32 GPE_TYPE4       : 1;
  __REG32 GPE_TYPE5       : 1;
  __REG32 GPE_TYPE6       : 1;
  __REG32 GPE_TYPE7       : 1;
  __REG32 GPE_TYPE8       : 1;
  __REG32 GPE_TYPE9       : 1;
  __REG32 GPE_TYPE10      : 1;
  __REG32 GPE_TYPE11      : 1;
  __REG32 GPE_TYPE12      : 1;
  __REG32 GPE_TYPE13      : 1;
  __REG32 GPE_TYPE14      : 1;
  __REG32 GPE_TYPE15      : 1;
} __gpe_mfp_bits;

/* Alternative Multiple Function Pin Control Register (ALT_MFP) */
typedef struct{
  __REG32 PB10_S01        : 1;
  __REG32 PB9_S11         : 1;
  __REG32 PA7_S21         : 1;
  __REG32 PB14_S31        : 1;
  __REG32 PB11_PWM4       : 1;
  __REG32 PC0_I2SLRCLK    : 1;
  __REG32 PC1_I2SBCLK     : 1;
  __REG32 PC2_I2SDI       : 1;
  __REG32 PC3_I2SDO       : 1;
  __REG32 PA15_I2SMCLK    : 1;
  __REG32 PB12_CLKO       : 1;
  __REG32                 :21;
} __alt_mfp_bits;

/* Register Lock Key Address Register (RegLockAddr) */
typedef struct{
  __REG32 RegUnLock       : 1;
  __REG32                 :31;
} __reglockaddr_bits;

/* RC Adjustment Control Register (RCADJ) */
typedef struct{
  __REG32 RCADJ           : 6;
  __REG32                 :26;
} __rcadj_bits;

/* Interrupt Source Identify Register (IRQn_SRC) */
typedef struct{
  __REG32 INT_SRC0        : 1;
  __REG32 INT_SRC1        : 1;
  __REG32 INT_SRC2        : 1;
  __REG32                 :29;
} __irqx_src_bits;

/* Interrupt Source Identify Register (IRQn_SRC) */
typedef struct{
  __REG32 INT_SRC0        : 1;
  __REG32 INT_SRC1        : 1;
  __REG32 INT_SRC2        : 1;
  __REG32 INT_SRC3        : 1;
  __REG32                 :28;
} __irq6_src_bits;

/* NMI Interrupt Source Select Control Register (NMI_SEL) */
typedef struct{
  __REG32 NMI_SEL         : 5;
  __REG32                 :27;
} __nmi_sel_bits;

/* MCU Interrupt Request Source Register (MCU_IRQ) */
typedef struct{
  __REG32 MCU_IRQ0        : 1;
  __REG32 MCU_IRQ1        : 1;
  __REG32 MCU_IRQ2        : 1;
  __REG32 MCU_IRQ3        : 1;
  __REG32 MCU_IRQ4        : 1;
  __REG32 MCU_IRQ5        : 1;
  __REG32 MCU_IRQ6        : 1;
  __REG32 MCU_IRQ7        : 1;
  __REG32 MCU_IRQ8        : 1;
  __REG32 MCU_IRQ9        : 1;
  __REG32 MCU_IRQ10       : 1;
  __REG32 MCU_IRQ11       : 1;
  __REG32 MCU_IRQ12       : 1;
  __REG32 MCU_IRQ13       : 1;
  __REG32 MCU_IRQ14       : 1;
  __REG32 MCU_IRQ15       : 1;
  __REG32 MCU_IRQ16       : 1;
  __REG32 MCU_IRQ17       : 1;
  __REG32 MCU_IRQ18       : 1;
  __REG32 MCU_IRQ19       : 1;
  __REG32 MCU_IRQ20       : 1;
  __REG32 MCU_IRQ21       : 1;
  __REG32 MCU_IRQ22       : 1;
  __REG32 MCU_IRQ23       : 1;
  __REG32 MCU_IRQ24       : 1;
  __REG32 MCU_IRQ25       : 1;
  __REG32 MCU_IRQ26       : 1;
  __REG32 MCU_IRQ27       : 1;
  __REG32 MCU_IRQ28       : 1;
  __REG32 MCU_IRQ29       : 1;
  __REG32 MCU_IRQ30       : 1;
  __REG32 MCU_IRQ31       : 1;
} __mcu_irq_bits;

/* Power Down Control Register (PWRCON) */
typedef struct{
__REG32 XTL12M_EN       : 1;
__REG32 XTL32K_EN       : 1;
__REG32 OSC22M_EN       : 1;
__REG32 OSC10K_EN       : 1;
__REG32 WU_DLY          : 1;
__REG32 PD_WU_INT_EN    : 1;
__REG32 PD_WU_STS       : 1;
__REG32 PWR_DOWN_EN     : 1;
__REG32 PD_WAIT_CPU     : 1;
__REG32                 :23;
} __pwrcon_bits;

/* AHB Devices Clock Enable Control Register (AHBCLK) */
typedef struct{
__REG32                 : 1;
__REG32 PDMA_EN         : 1;
__REG32 ISP_EN          : 1;
__REG32                 :29;
} __ahbclk_bits;

/* APB Devices Clock Enable Control Register (APBCLK) */
typedef struct{
__REG32 WDCLK_EN        : 1;
__REG32 RTC_EN          : 1;
__REG32 TMR0_EN         : 1;
__REG32 TMR1_EN         : 1;
__REG32 TMR2_EN         : 1;
__REG32 TMR3_EN         : 1;
__REG32 FDIV_EN         : 1;
__REG32                 : 1;
__REG32 I2C0_EN         : 1;
__REG32 I2C1_EN         : 1;
__REG32                 : 2;
__REG32 SPI0_EN         : 1;
__REG32 SPI1_EN         : 1;
__REG32                 : 2;
__REG32 UART0_EN        : 1;
__REG32 UART1_EN        : 1;
__REG32                 : 2;
__REG32 PWM01_EN        : 1;
__REG32 PWM23_EN        : 1;
__REG32                 : 6;
__REG32 ADC_EN          : 1;
__REG32 I2S_EN          : 1;
__REG32 ACMP_EN         : 1;
__REG32 PS2_EN          : 1;
} __apbclk_bits;

/* Clock Source Select Control Register 0 (CLKSEL0) */
typedef struct{
__REG32 HCLK_S          : 3;
__REG32 STCLK_S         : 3;
__REG32                 :26;
} __clksel0_bits;

/* Clock Source Select Control Register  1 (CLKSEL1) */
typedef struct{
__REG32 WDG_S           : 2;
__REG32 ADC_S           : 2;
__REG32                 : 4;
__REG32 TMR0_S          : 3;
__REG32                 : 1;
__REG32 TMR1_S          : 3;
__REG32                 : 1;
__REG32 TMR2_S          : 3;
__REG32                 : 1;
__REG32 TMR3_S          : 3;
__REG32                 : 1;
__REG32 UART_S          : 2;
__REG32                 : 2;
__REG32 PWM01_S         : 2;
__REG32 PWM23_S         : 2;
} __clksel1_bits;

/* Clock Source Select Control Register (CLKSEL2) */
typedef struct{
__REG32 I2S_S           : 2;
__REG32 FRQDIV_S        : 2;
__REG32                 :28;
} __clksel2_bits;

/* Clock Divider Register (CLKDIV) */
typedef struct{
__REG32 HCLK_N          : 4;
__REG32                 : 4;
__REG32 UART_N          : 4;
__REG32                 : 4;
__REG32 ADC_N           : 8;
__REG32                 : 8;
} __clkdiv_bits;

/* PLL Control Register (PLLCON) */
typedef struct{
__REG32 FB_DV           : 9;
__REG32 IN_DV           : 5;
__REG32 OUT_DV          : 2;
__REG32 PD              : 1;
__REG32 BP              : 1;
__REG32 OE              : 1;
__REG32 PLL_SRC         : 1;
__REG32                 :12;
} __pllcon_bits;

/* Frequency Divider Control Register (FRQDIV) */
typedef struct{
__REG32 FSEL            : 4;
__REG32 FDIV_EN         : 1;
__REG32                 :27;
} __frqdiv_bits;

/* GPIO Port [A/B/C/D/E] Bit Mode Control (GPIOx_PMD) */
typedef struct{
__REG32 PMD0            : 2;
__REG32 PMD1            : 2;
__REG32 PMD2            : 2;
__REG32 PMD3            : 2;
__REG32 PMD4            : 2;
__REG32 PMD5            : 2;
__REG32 PMD6            : 2;
__REG32 PMD7            : 2;
__REG32 PMD8            : 2;
__REG32 PMD9            : 2;
__REG32 PMD10           : 2;
__REG32 PMD11           : 2;
__REG32 PMD12           : 2;
__REG32 PMD13           : 2;
__REG32 PMD14           : 2;
__REG32 PMD15           : 2;
} __gpiox_pmd_bits;

/* GPIO Port [A/B/C/D/E] Bit OFF Digital Resistor Enable (GPIOx_OFFD) */
typedef struct{
__REG32                 :16;
__REG32 OFFD0           : 1;
__REG32 OFFD1           : 1;
__REG32 OFFD2           : 1;
__REG32 OFFD3           : 1;
__REG32 OFFD4           : 1;
__REG32 OFFD5           : 1;
__REG32 OFFD6           : 1;
__REG32 OFFD7           : 1;
__REG32 OFFD8           : 1;
__REG32 OFFD9           : 1;
__REG32 OFFD10          : 1;
__REG32 OFFD11          : 1;
__REG32 OFFD12          : 1;
__REG32 OFFD13          : 1;
__REG32 OFFD14          : 1;
__REG32 OFFD15          : 1;
} __gpiox_offd_bits;

/* GPIO Port [A/B/C/D/E] Data Output Value (GPIOx_DOUT) */
typedef struct{
__REG32 DOUT0           : 1;
__REG32 DOUT1           : 1;
__REG32 DOUT2           : 1;
__REG32 DOUT3           : 1;
__REG32 DOUT4           : 1;
__REG32 DOUT5           : 1;
__REG32 DOUT6           : 1;
__REG32 DOUT7           : 1;
__REG32 DOUT8           : 1;
__REG32 DOUT9           : 1;
__REG32 DOUT10          : 1;
__REG32 DOUT11          : 1;
__REG32 DOUT12          : 1;
__REG32 DOUT13          : 1;
__REG32 DOUT14          : 1;
__REG32 DOUT15          : 1;
__REG32                 :16;
} __gpiox_dout_bits;

/* GPIO Port [A/B/C/D/E] Data Output Write Mask (GPIOx _DMASK) */
typedef struct{
__REG32 DMASK0          : 1;
__REG32 DMASK1          : 1;
__REG32 DMASK2          : 1;
__REG32 DMASK3          : 1;
__REG32 DMASK4          : 1;
__REG32 DMASK5          : 1;
__REG32 DMASK6          : 1;
__REG32 DMASK7          : 1;
__REG32 DMASK8          : 1;
__REG32 DMASK9          : 1;
__REG32 DMASK10         : 1;
__REG32 DMASK11         : 1;
__REG32 DMASK12         : 1;
__REG32 DMASK13         : 1;
__REG32 DMASK14         : 1;
__REG32 DMASK15         : 1;
__REG32                 :16;
} __gpiox_dmask_bits;

/* GPIO Port [A/B/C/D/E] Pin Value (GPIOx _PIN) */
typedef struct{
__REG32 PIN0            : 1;
__REG32 PIN1            : 1;
__REG32 PIN2            : 1;
__REG32 PIN3            : 1;
__REG32 PIN4            : 1;
__REG32 PIN5            : 1;
__REG32 PIN6            : 1;
__REG32 PIN7            : 1;
__REG32 PIN8            : 1;
__REG32 PIN9            : 1;
__REG32 PIN10           : 1;
__REG32 PIN11           : 1;
__REG32 PIN12           : 1;
__REG32 PIN13           : 1;
__REG32 PIN14           : 1;
__REG32 PIN15           : 1;
__REG32                 :16;
} __gpiox_pin_bits;

/* GPIO Port [A/B/C/D/E] De-bounce Enable (GPIOx _DBEN) */
typedef struct{
__REG32 DBEN0           : 1;
__REG32 DBEN1           : 1;
__REG32 DBEN2           : 1;
__REG32 DBEN3           : 1;
__REG32 DBEN4           : 1;
__REG32 DBEN5           : 1;
__REG32 DBEN6           : 1;
__REG32 DBEN7           : 1;
__REG32 DBEN8           : 1;
__REG32 DBEN9           : 1;
__REG32 DBEN10          : 1;
__REG32 DBEN11          : 1;
__REG32 DBEN12          : 1;
__REG32 DBEN13          : 1;
__REG32 DBEN14          : 1;
__REG32 DBEN15          : 1;
__REG32                 :16;
} __gpiox_dben_bits;

/* GPIO Port [A/B/C/D/E] Interrupt Mode Control (GPIOx _IMD) */
typedef struct{
__REG32 IMD0            : 1;
__REG32 IMD1            : 1;
__REG32 IMD2            : 1;
__REG32 IMD3            : 1;
__REG32 IMD4            : 1;
__REG32 IMD5            : 1;
__REG32 IMD6            : 1;
__REG32 IMD7            : 1;
__REG32 IMD8            : 1;
__REG32 IMD9            : 1;
__REG32 IMD10           : 1;
__REG32 IMD11           : 1;
__REG32 IMD12           : 1;
__REG32 IMD13           : 1;
__REG32 IMD14           : 1;
__REG32 IMD15           : 1;
__REG32                 :16;
} __gpiox_imd_bits;

/* GPIO Port [A/B/C/D] Interrupt Enable Control (GPIOx _IEN) */
typedef struct{
__REG32 IF_EN0          : 1;
__REG32 IF_EN1          : 1;
__REG32 IF_EN2          : 1;
__REG32 IF_EN3          : 1;
__REG32 IF_EN4          : 1;
__REG32 IF_EN5          : 1;
__REG32 IF_EN6          : 1;
__REG32 IF_EN7          : 1;
__REG32 IF_EN8          : 1;
__REG32 IF_EN9          : 1;
__REG32 IF_EN10         : 1;
__REG32 IF_EN11         : 1;
__REG32 IF_EN12         : 1;
__REG32 IF_EN13         : 1;
__REG32 IF_EN14         : 1;
__REG32 IF_EN15         : 1;
__REG32 IR_EN0          : 1;
__REG32 IR_EN1          : 1;
__REG32 IR_EN2          : 1;
__REG32 IR_EN3          : 1;
__REG32 IR_EN4          : 1;
__REG32 IR_EN5          : 1;
__REG32 IR_EN6          : 1;
__REG32 IR_EN7          : 1;
__REG32 IR_EN8          : 1;
__REG32 IR_EN9          : 1;
__REG32 IR_EN10         : 1;
__REG32 IR_EN11         : 1;
__REG32 IR_EN12         : 1;
__REG32 IR_EN13         : 1;
__REG32 IR_EN14         : 1;
__REG32 IR_EN15         : 1;
} __gpiox_ien_bits;

/* GPIO Port [A/B/C/D/E] Interrupt Trigger Source (GPIOx _ISRC) */
typedef struct{
__REG32 ISRC0           : 1;
__REG32 ISRC1           : 1;
__REG32 ISRC2           : 1;
__REG32 ISRC3           : 1;
__REG32 ISRC4           : 1;
__REG32 ISRC5           : 1;
__REG32 ISRC6           : 1;
__REG32 ISRC7           : 1;
__REG32 ISRC8           : 1;
__REG32 ISRC9           : 1;
__REG32 ISRC10          : 1;
__REG32 ISRC11          : 1;
__REG32 ISRC12          : 1;
__REG32 ISRC13          : 1;
__REG32 ISRC14          : 1;
__REG32 ISRC15          : 1;
__REG32                 :16;
} __gpiox_isrc_bits;

/* Interrupt De-bounce Cycle Control (DBNCECON) */
typedef struct{
__REG32 DBCLKSEL        : 4;
__REG32 DBCLKSRC        : 1;
__REG32 ICLK_ON         : 1;
__REG32                 :26;
} __dbncecon_bits;

/* I2Cx CONTROL REGISTER (I2CxCON) */
typedef struct{
__REG32                 : 2;
__REG32 AA              : 1;
__REG32 SI              : 1;
__REG32 STO             : 1;
__REG32 STA             : 1;
__REG32 ENSI            : 1;
__REG32 EI              : 1;
__REG32                 :24;
} __i2cxcon_bits;

/* I2Cx DATA REGISTE (I2CxDAT) */
typedef struct{
__REG32 I2DAT           : 8;
__REG32                 :24;
} __i2cxdat_bits;

/* I2Cx STATUS REGISTER (I2CxSTATUS) */
typedef struct{
__REG32 I2STATUS        : 8;
__REG32                 :24;
} __i2cxstatus_bits;

/* I2Cx BAUD RATE CONTROL REGISTER (I2CxCLK) */
typedef struct{
__REG32 I2CLK           : 8;
__REG32                 :24;
} __i2cxclk_bits;

/* I2Cx TIME-OUT COUNTER REGISTER (I2CxTOC) */
typedef struct{
__REG32 TIF             : 1;
__REG32 DIV4            : 1;
__REG32 ENTI            : 1;
__REG32                 :29;
} __i2cxtoc_bits;

/* I2Cx SLAVE ADDRESS REGISTER (I2CxADDRy) */
typedef struct{
__REG32 GC              : 1;
__REG32 I2ADDR          : 7;
__REG32                 :24;
} __i2cxaddry_bits;

/* I2Cx SLAVE ADDRESS MASK REGISTER (I2CxADMy) */
typedef struct{
__REG32 GC              : 1;
__REG32 I2ADM           : 7;
__REG32                 :24;
} __i2cxadmy_bits;

/* PWM Pre-Scale Register (PPR03) */
typedef struct{
__REG32 CP01            : 8;
__REG32 CP23            : 8;
__REG32 DZI01           : 8;
__REG32 DZI23           : 8;
} __pwm_ppr03_bits;

/* PWM Pre-Scale Register (PPR47) */
typedef struct{
__REG32 CP45            : 8;
__REG32 CP67            : 8;
__REG32 DZI45           : 8;
__REG32 DZI67           : 8;
} __pwm_ppr47_bits;

/* PWM Clock Selector Register (CSR03) */
typedef struct{
__REG32 CSR0            : 3;
__REG32                 : 1;
__REG32 CSR1            : 3;
__REG32                 : 1;
__REG32 CSR2            : 3;
__REG32                 : 1;
__REG32 CSR3            : 3;
__REG32                 :17;
} __pwm_csr03_bits;

/* PWM Clock Selector Register (CSR47) */
typedef struct{
__REG32 CSR4            : 3;
__REG32                 : 1;
__REG32 CSR5            : 3;
__REG32                 : 1;
__REG32 CSR6            : 3;
__REG32                 : 1;
__REG32 CSR7            : 3;
__REG32                 :17;
} __pwm_csr47_bits;

/* PWM Control Register (PCR03) */
typedef struct{
__REG32 CH0EN           : 1;
__REG32                 : 1;
__REG32 CH0INV          : 1;
__REG32 CH0MOD          : 1;
__REG32 DZEN01          : 1;
__REG32 DZEN23          : 1;
__REG32                 : 2;
__REG32 CH1EN           : 1;
__REG32                 : 1;
__REG32 CH1INV          : 1;
__REG32 CH1MOD          : 1;
__REG32                 : 4;
__REG32 CH2EN           : 1;
__REG32                 : 1;
__REG32 CH2INV          : 1;
__REG32 CH2MOD          : 1;
__REG32                 : 4;
__REG32 CH3EN           : 1;
__REG32                 : 1;
__REG32 CH3INV          : 1;
__REG32 CH3MOD          : 1;
__REG32                 : 4;
} __pwm_pcr03_bits;

/* PWM Control Register (PCR47) */
typedef struct{
__REG32 CH4EN           : 1;
__REG32                 : 1;
__REG32 CH4INV          : 1;
__REG32 CH4MOD          : 1;
__REG32 DZEN45          : 1;
__REG32 DZEN67          : 1;
__REG32                 : 2;
__REG32 CH5EN           : 1;
__REG32                 : 1;
__REG32 CH5INV          : 1;
__REG32 CH5MOD          : 1;
__REG32                 : 4;
__REG32 CH6EN           : 1;
__REG32                 : 1;
__REG32 CH6INV          : 1;
__REG32 CH6MOD          : 1;
__REG32                 : 4;
__REG32 CH7EN           : 1;
__REG32                 : 1;
__REG32 CH7INV          : 1;
__REG32 CH7MOD          : 1;
__REG32                 : 4;
} __pwm_pcr47_bits;

/* PWM Counter Register 7-0 (CNR7-0) */
typedef struct{
__REG32 CNR             :16;
__REG32                 :16;
} __pwm_cnr_bits;

/* PWM Comparator Register 7-0 (CMR7-0) */
typedef struct{
__REG32 CMR             :16;
__REG32                 :16;
} __pwm_cmr_bits;

/* PWM Data Register 7-0 (PDR 7-0) */
typedef struct{
__REG32 PDR             :16;
__REG32                 :16;
} __pwm_pdr_bits;

/* PWM Interrupt Enable Register (PIER03) */
typedef struct{
__REG32 PWMIE0          : 1;
__REG32 PWMIE1          : 1;
__REG32 PWMIE2          : 1;
__REG32 PWMIE3          : 1;
__REG32                 :28;
} __pwm_pier03_bits;

/* PWM Interrupt Indication Register (PIIR03) */
typedef struct{
__REG32 PWMIF0          : 1;
__REG32 PWMIF1          : 1;
__REG32 PWMIF2          : 1;
__REG32 PWMIF3          : 1;
__REG32                 :28;
} __pwm_piir03_bits;

/* PWM Capture Control Register (CCR0) */
typedef struct{
__REG32 INV0            : 1;
__REG32 CRL_IE0         : 1;
__REG32 CFL_IE0         : 1;
__REG32 CAPCH0EN        : 1;
__REG32 CAPIF0          : 1;
__REG32                 : 1;
__REG32 CRLRI0          : 1;
__REG32 CFLRI0          : 1;
__REG32                 : 8;
__REG32 INV1            : 1;
__REG32 CRL_IE1         : 1;
__REG32 CFL_IE1         : 1;
__REG32 CAPCH1EN        : 1;
__REG32 CAPIF1          : 1;
__REG32                 : 1;
__REG32 CRLRI1          : 1;
__REG32 CFLRI1          : 1;
__REG32                 : 8;
} __pwm_ccr0_bits;

/* PWM Capture Control Register (CCR2) */
typedef struct{
__REG32 INV2            : 1;
__REG32 CRL_IE2         : 1;
__REG32 CFL_IE2         : 1;
__REG32 CAPCH2EN        : 1;
__REG32 CAPIF2          : 1;
__REG32                 : 1;
__REG32 CRLRI2          : 1;
__REG32 CFLRI2          : 1;
__REG32                 : 8;
__REG32 INV3            : 1;
__REG32 CRL_IE3         : 1;
__REG32 CFL_IE3         : 1;
__REG32 CAPCH3EN        : 1;
__REG32 CAPIF3          : 1;
__REG32                 : 1;
__REG32 CRLRI3          : 1;
__REG32 CFLRI3          : 1;
__REG32                 : 8;
} __pwm_ccr2_bits;

/* PWM Capture Rising Latch Register3-0 (CRLR3-0) */
typedef struct{
__REG32 CRLR            :16;
__REG32                 :16;
} __pwm_crlr_bits;

/* PWM Capture Falling Latch Register3-0 (CFLR3-0) */
typedef struct{
__REG32 CFLR            :16;
__REG32                 :16;
} __pwm_cflr_bits;

/* PWM Capture Input Enable Register (CAPENR03) */
typedef struct{
__REG32 CAPENR0         : 1;
__REG32 CAPENR1         : 1;
__REG32 CAPENR2         : 1;
__REG32 CAPENR3         : 1;
__REG32                 :28;
} __pwm_capenr03_bits;

/* PWM Output Enable Register (POE03) */
typedef struct{
__REG32 PWM0            : 1;
__REG32 PWM1            : 1;
__REG32 PWM2            : 1;
__REG32 PWM3            : 1;
__REG32                 :28;
} __pwm_poe03_bits;

/* RTC Initiation Register (INIR) */
typedef struct{
__REG32 Active          : 1;
__REG32 INIR            :31;
} __rtc_inir_bits;

/* RTC Access Enable Register (AER) */
typedef struct{
__REG32 AER             :16;
__REG32 ENF             : 1;
__REG32                 :15;
} __rtc_aer_bits;

/* RTC Frequency Compensation Register (FCR) */
typedef struct{
__REG32 FRACTION        : 6;
__REG32                 : 2;
__REG32 INTEGER         : 4;
__REG32                 :20;
} __rtc_fcr_bits;

/* RTC Time Loading Register (TLR) */
typedef struct{
__REG32 _1SEC           : 4;
__REG32 _10SEC          : 3;
__REG32                 : 1;
__REG32 _1MIN           : 4;
__REG32 _10MIN          : 3;
__REG32                 : 1;
__REG32 _1HR            : 4;
__REG32 _10HR           : 2;
__REG32                 :10;
} __rtc_tlr_bits;

/* RTC Calendar Loading Register (CLR) */
typedef struct{
__REG32 _1DAY           : 4;
__REG32 _10DAY          : 2;
__REG32                 : 2;
__REG32 _1MON           : 4;
__REG32 _10MON          : 1;
__REG32                 : 3;
__REG32 _1YEAR          : 4;
__REG32 _10YEAR         : 4;
__REG32                 : 8;
} __rtc_clr_bits;

/* RTC Time Scale Selection Register (TSSR) */
typedef struct{
__REG32 _24HOUR         : 1;
__REG32                 :31;
} __rtc_tssr_bits;

/* RTC Day of the Week Register (DWR) */
typedef struct{
__REG32 DWR             : 3;
__REG32                 :29;
} __rtc_dwr_bits;

/* RTC Leap year Indication Register (LIR) */
typedef struct{
__REG32 LIR             : 1;
__REG32                 :31;
} __rtc_lir_bits;

/* RTC Interrupt Enable Register (RIER) */
typedef struct{
__REG32 AIER            : 1;
__REG32 TIER            : 1;
__REG32                 :30;
} __rtc_rier_bits;

/* RTC Interrupt Indication Register (RIIR) */
typedef struct{
__REG32 AIF             : 1;
__REG32 TIF             : 1;
__REG32                 :30;
} __rtc_riir_bits;

/* RTC Time Tick Register (TTR) */
typedef struct{
__REG32 TTR             : 3;
__REG32 TWKE            : 1;
__REG32                 :28;
} __rtc_ttr_bits;

/* SPIx Control and Status Register (SPIx_CNTRL) */
typedef struct{
__REG32 GO_BUSY         : 1;
__REG32 Rx_NEG          : 1;
__REG32 Tx_NEG          : 1;
__REG32 Tx_BIT_LEN      : 5;
__REG32 Tx_NUM          : 2;
__REG32 LSB             : 1;
__REG32 CLKP            : 1;
__REG32 SLEEP           : 4;
__REG32 IF              : 1;
__REG32 IE              : 1;
__REG32 SLAVE           : 1;
__REG32 BYTE_SLEEP      : 1;
__REG32 BYTE_ENDIAN     : 1;
__REG32 FIFO            : 1;
__REG32                 : 2;
__REG32 Rx_EMPTY        : 1;
__REG32 Rx_FULL         : 1;
__REG32 Tx_EMPTY        : 1;
__REG32 Tx_FULL         : 1;
__REG32                 : 4;
} __spi_cntrl_bits;

/* SPIx Divider Register (SPIx_DIVIDER) */
typedef struct{
__REG32 DIVIDER         :16;
__REG32                 :16;
} __spi_divider_bits;

/* SPIx Slave Select Register (SPIx_SSR) */
typedef struct{
__REG32 SSR             : 2;
__REG32 SS_LVL          : 1;
__REG32 ASS             : 1;
__REG32 SS_LTRIG        : 1;
__REG32 LTRIG_FLAG      : 1;
__REG32                 :26;
} __spi_ssr_bits;

/* SPIx DMA Control Register (SPIx_DMA) */
typedef struct{
__REG32 Tx_DMA_GO       : 1;
__REG32 Rx_DMA_GO       : 1;
__REG32                 :30;
} __spi_dma_bits;

/* Timer x Control Register (TCSRx) */
typedef struct{
__REG32 PRESCALE        : 8;
__REG32                 : 8;
__REG32 TDR_EN          : 1;
__REG32                 : 8;
__REG32 CACT            : 1;
__REG32 CRST            : 1;
__REG32 MODE            : 2;
__REG32 IE              : 1;
__REG32 CEN             : 1;
__REG32                 : 1;
} __tcsr_bits;

/* Timer x Compare Register (TCMPRx) */
typedef struct{
__REG32 TCMP            :24;
__REG32                 : 8;
} __tcmpr_bits;

/* Timer x Interrupt Status Register (TISRx) */
typedef struct{
__REG32 TIF             : 1;
__REG32                 :31;
} __tisr_bits;

/* Timer x Data Register (TDRx) */
typedef struct{
__REG32 TDR             :24;
__REG32                 : 8;
} __tdr_bits;

/* Watchdog Timer Control Register (WTCR) */
typedef struct{
__REG32 CLRWTR          : 1;
__REG32 WTRE            : 1;
__REG32 WTRF            : 1;
__REG32 WTIF            : 1;
__REG32 WTWKE           : 1;
__REG32 WTWKF           : 1;
__REG32 WTIE            : 1;
__REG32 WTE             : 1;
__REG32 WTIS            : 3;
__REG32                 :21;
} __wtcr_bits;

/* Receive Buffer Register (UA_RBR) */
/* Transmit Holding Register (UA_THR) */
typedef struct {
  __REG32 DATA              : 8;
  __REG32                   :24;
} __uart_rbr_bits;

/* Interrupt Enable Register (UAx_IER) */
typedef struct {
  __REG32 RDA_IEN           : 1;
  __REG32 THRE_IEN          : 1;
  __REG32 RLS_IEN           : 1;
  __REG32 Modem_IEN         : 1;
  __REG32 RTO_IEN           : 1;
  __REG32 BUF_ERR_IEN       : 1;
  __REG32 Wake_EN           : 1;
  __REG32                   : 1;
  __REG32 LIN_RX_BRK_IEN    : 1;
  __REG32                   : 2;
  __REG32 Time_Out_EN       : 1;
  __REG32 Auto_RTS_EN       : 1;
  __REG32 Auto_CTS_EN       : 1;
  __REG32 DMA_Tx_EN         : 1;
  __REG32 DMA_Rx_EN         : 1;
  __REG32                   :16;
} __uart_ier_bits;

/* FIFO Control Register (UAx_FCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 Rx_RST            : 1;
  __REG32 Tx_RST            : 1;
  __REG32                   : 1;
  __REG32 RFITL             : 4;
  __REG32                   : 8;
  __REG32 RTS_Tri_Lev       : 4;
  __REG32                   :12;
} __uart_fcr_bits;

/* UART Line Control Register (UAx_LCR) */
typedef struct {
  __REG32 WLS               : 2;
  __REG32 NSB               : 1;
  __REG32 PBE               : 1;
  __REG32 EPE               : 1;
  __REG32 SPE               : 1;
  __REG32 BCB               : 1;
  __REG32                   :25;
} __uart_lcr_bits;

/* UART MODEM Control Register (UAx_MCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 RTS               : 1;
  __REG32                   : 7;
  __REG32 Lev_RTS           : 1;
  __REG32                   : 3;
  __REG32 RTS_St            : 1;
  __REG32                   :18;
} __uart_mcr_bits;

/* UART Modem Status Register (UAx_MSR) */
typedef struct {
  __REG32 DCTSF             : 1;
  __REG32                   : 3;
  __REG32 CTS_St            : 1;
  __REG32                   : 3;
  __REG32 Lev_CTS           : 1;
  __REG32                   :23;
} __uart_msr_bits;

/* UART FIFO Status Register (UAx_FSR) */
typedef struct {
  __REG32 Rx_Over_IF        : 1;
  __REG32                   : 3;
  __REG32 PEF               : 1;
  __REG32 FEF               : 1;
  __REG32 BIF               : 1;
  __REG32                   : 1;
  __REG32 Rx_Pointer        : 6;
  __REG32 Rx_Empty          : 1;
  __REG32 Rx_Full           : 1;
  __REG32 Tx_Pointer        : 6;
  __REG32 Tx_Empty          : 1;
  __REG32 Tx_Full           : 1;
  __REG32 Tx_Over_IF        : 1;
  __REG32                   : 3;
  __REG32 TE_Flag           : 1;
  __REG32                   : 3;
} __uart_fsr_bits;

/* UART Interrupt Status Control Register (UAx_ISR) */
typedef struct {
  __REG32 RDA_IF              : 1;
  __REG32 THRE_IF             : 1;
  __REG32 RLS_IF              : 1;
  __REG32 Modem_IF            : 1;
  __REG32 Tout_IF             : 1;
  __REG32 Buf_Err_IF          : 1;
  __REG32                     : 1;
  __REG32 LIN_Rx_Break_IF     : 1;
  __REG32 RDA_INT             : 1;
  __REG32 THRE_INT            : 1;
  __REG32 RLS_INT             : 1;
  __REG32 Modem_INT           : 1;
  __REG32 Tout_INT            : 1;
  __REG32 Buf_Err_INT         : 1;
  __REG32                     : 1;
  __REG32 LIN_Rx_Break_INT    : 1;
  __REG32                     : 2;
  __REG32 HW_RLS_IF           : 1;
  __REG32 HW_Modem_IF         : 1;
  __REG32 HW_Tout_IF          : 1;
  __REG32 HW_Buf_Err_IF       : 1;
  __REG32                     : 1;
  __REG32 HW_LIN_Rx_Break_IF  : 1;
  __REG32                     : 2;
  __REG32 HW_RLS_INT          : 1;
  __REG32 HW_Modem_INT        : 1;
  __REG32 HW_Tout_INT         : 1;
  __REG32 HW_Buf_Err_INT      : 1;
  __REG32                     : 1;
  __REG32 HW_LIN_Rx_Break_INT : 1;
} __uart_isr_bits;

/* UART Time out Register (UAx_TOR) */
typedef struct {
  __REG32 TOIC              : 7;
  __REG32                   :25;
} __uart_tor_bits;

/* Baud Rate Divider Register (UAx_BAUD) */
typedef struct {
  __REG32 BRD               :16;
  __REG32                   : 8;
  __REG32 Divider_X         : 4;
  __REG32 DIV_X_ONE         : 1;
  __REG32 DIV_X_EN          : 1;
  __REG32                   : 2;
} __uart_baud_bits;

/* IrDA Control Register (UAx_IRCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 Tx_SELECT         : 1;
  __REG32                   : 3;
  __REG32 INV_Tx            : 1;
  __REG32 INV_Rx            : 1;
  __REG32                   :25;
} __uart_ircr_bits;

/* UART LIN Break Field Count Register (UAx_LIN_BCNT) */
typedef struct {
  __REG32 UA_LIN_BKFL       : 4;
  __REG32                   : 2;
  __REG32 LIN_Rx_EN         : 1;
  __REG32 LIN_Tx_EN         : 1;
  __REG32                   :24;
} __uart_lin_bcnt_bits;

/* UART Function Select Register (UAx_FUN_SEL) */
typedef struct {
  __REG32 LIN_EN            : 1;
  __REG32 IrDA_EN           : 1;
  __REG32                   :30;
} __uart_fun_sel_bits;

/* PS2 Control Register (PS2CON) */
typedef struct {
  __REG32 PS2EN             : 1;
  __REG32 TXINTEN           : 1;
  __REG32 RXINTEN           : 1;
  __REG32 TXFIFO_DEPTH      : 4;
  __REG32 ACK               : 1;
  __REG32 CLRFIFO           : 1;
  __REG32 OVERRIDE          : 1;
  __REG32 FPS2CLK           : 1;
  __REG32 FPS2DAT           : 1;
  __REG32                   :20;
} __ps2con_bits;

/* PS2 Receiver DATA Register (PS2RXDATA ) */
typedef struct {
  __REG32 RXDATA            : 8;
  __REG32                   :24;
} __ps2rxdata_bits;

/* PS2 Status Register (PS2STATUS) */
typedef struct {
  __REG32 PS2CLK            : 1;
  __REG32 PS2DATA           : 1;
  __REG32 FRAMERR           : 1;
  __REG32 RXPARITY          : 1;
  __REG32 RXBUSY            : 1;
  __REG32 TXBUSY            : 1;
  __REG32 RXOVF             : 1;
  __REG32 TXEMPTY           : 1;
  __REG32 BYTEIDX           : 4;
  __REG32                   :20;
} __ps2status_bits;

/* PS2 Interrupt Identification Register (PS2INTID) */
typedef struct {
  __REG32 RXINT             : 1;
  __REG32 TXINT             : 1;
  __REG32                   :30;
} __ps2intid_bits;

/* I2S Control Register (I2S_CON) */
typedef struct {
  __REG32 I2SEN             : 1;
  __REG32 TXEN              : 1;
  __REG32 RXEN              : 1;
  __REG32 MUTE              : 1;
  __REG32 WORDWIDTH         : 2;
  __REG32 MONO              : 1;
  __REG32 FORMAT            : 1;
  __REG32 SLAVE             : 1;
  __REG32 TXTH              : 3;
  __REG32 RXTH              : 3;
  __REG32 MCLKEN            : 1;
  __REG32 RCHZCEN           : 1;
  __REG32 LCHZCEN           : 1;
  __REG32 CLR_TXFIFO        : 1;
  __REG32 CLR_RXFIFO        : 1;
  __REG32 TXDMA             : 1;
  __REG32 RXDMA             : 1;
  __REG32                   :10;
} __i2s_con_bits;

/* I2S Clock Divider (I2S_CLKDIV) */
typedef struct {
  __REG32 MCLK_DIV          : 3;
  __REG32                   : 5;
  __REG32 BCLK_DIV          : 8;
  __REG32                   :16;
} __i2s_clkdiv_bits;

/* I2S Interrupt Enable Register (I2S_IE) */
typedef struct {
  __REG32 RXUDFIE           : 1;
  __REG32 RXOVFIE           : 1;
  __REG32 RXTHIE            : 1;
  __REG32                   : 5;
  __REG32 TXUDFIE           : 1;
  __REG32 TXOVFIE           : 1;
  __REG32 TXTHIE            : 1;
  __REG32 RZCIE             : 1;
  __REG32 LZCIE             : 1;
  __REG32                   :19;
} __i2s_ie_bits;

/* I2S Status Register (I2S_STATUS) */
typedef struct {
  __REG32 I2SINT            : 1;
  __REG32 I2SRXINT          : 1;
  __REG32 I2STXINT          : 1;
  __REG32 RIGHT             : 1;
  __REG32                   : 4;
  __REG32 RXUDF             : 1;
  __REG32 RXOVF             : 1;
  __REG32 RXTHF             : 1;
  __REG32 RXFULL            : 1;
  __REG32 RXEMPTY           : 1;
  __REG32                   : 3;
  __REG32 TXUDF             : 1;
  __REG32 TXOVF             : 1;
  __REG32 TXTHF             : 1;
  __REG32 TXFULL            : 1;
  __REG32 TXEMPTY           : 1;
  __REG32 TXBUSY            : 1;
  __REG32 RZCF              : 1;
  __REG32 LZCF              : 1;
  __REG32 RX_LEVEL          : 4;
  __REG32 TX_LEVEL          : 4;
} __i2s_status_bits;

/* A/D Data Registers (ADDR0 ~ ADDR7) */
typedef struct {
  __REG32 RSLT              :12;
  __REG32                   : 4;
  __REG32 OVERRUN           : 1;
  __REG32 VALID             : 1;
  __REG32                   :14;
} __addr_bits;

/* A/D Control Register (ADCR) */
typedef struct {
  __REG32 ADEN              : 1;
  __REG32 ADIE              : 1;
  __REG32 ADMD              : 2;
  __REG32 TRGS              : 2;
  __REG32 TRGCOND           : 2;
  __REG32 TRGEN             : 1;
  __REG32 PTEN              : 1;
  __REG32 DIFFEN            : 1;
  __REG32 ADST              : 1;
  __REG32                   :20;
} __adcr_bits;

/* A/D Channel Enable Register (ADCHER) */
typedef struct {
  __REG32 CHEN0             : 1;
  __REG32 CHEN1             : 1;
  __REG32 CHEN2             : 1;
  __REG32 CHEN3             : 1;
  __REG32 CHEN4             : 1;
  __REG32 CHEN5             : 1;
  __REG32 CHEN6             : 1;
  __REG32 CHEN7             : 1;
  __REG32 PRESEL            : 2;
  __REG32                   :22;
} __adcher_bits;

/* A/D Compare Register 0/1 (ADCMPR0/1) */
typedef struct {
  __REG32 CPMEN             : 1;
  __REG32 CMPIE             : 1;
  __REG32 CMPCOND           : 1;
  __REG32 CMPCH             : 3;
  __REG32                   : 2;
  __REG32 CMPMATCNT         : 4;
  __REG32                   : 4;
  __REG32 CMPD              :12;
  __REG32                   : 4;
} __adcmprx_bits;

/* A/D Status Register (ADSR) */
typedef struct {
  __REG32 ADF               : 1;
  __REG32 CMPF0             : 1;
  __REG32 CMPF1             : 1;
  __REG32 BUSY              : 1;
  __REG32 CHANNEL           : 3;
  __REG32                   : 1;
  __REG32 VALID             : 8;
  __REG32 OVERRUN           : 8;
  __REG32                   : 8;
} __adsr_bits;

/* A/D Calibration Register (ADCALR) */
typedef struct {
  __REG32 CALEN             : 1;
  __REG32 CALDONE           : 1;
  __REG32                   :30;
} __adcalr_bits;

/* A/D PDMA current transfer data Register (ADPDMA) */
typedef struct {
  __REG32 AD_PDMA           :12;
  __REG32                   :20;
} __adpdma_bits;

/* CMP0 Control Register (CMP0CR) */
typedef struct {
  __REG32 CMP0EN            : 1;
  __REG32 CMP0IE            : 1;
  __REG32 CMP0_HYSEN        : 1;
  __REG32                   : 1;
  __REG32 CN0               : 1;
  __REG32                   :27;
} __cmp0cr_bits;

/* CMP1 Control Register (CMP1CR) */
typedef struct {
  __REG32 CMP1EN            : 1;
  __REG32 CMP1IE            : 1;
  __REG32 CMP1_HYSEN        : 1;
  __REG32                   : 1;
  __REG32 CN1               : 1;
  __REG32                   :27;
} __cmp1cr_bits;

/* CMP Status Register (CMPSR) */
typedef struct {
  __REG32 CMPF0             : 1;
  __REG32 CMPF1             : 1;
  __REG32 CO0               : 1;
  __REG32 CO1               : 1;
  __REG32                   :28;
} __cmpsr_bits;

/* PDMA Control and Status Register (PDMA_CSRx) */
typedef struct {
  __REG32 PDMACEN           : 1;
  __REG32 SW_RST            : 1;
  __REG32 MODE_SEL          : 2;
  __REG32 SAD_SEL           : 2;
  __REG32 DAD_SEL           : 2;
  __REG32                   :11;
  __REG32 APB_TWS           : 2;
  __REG32                   : 2;
  __REG32 Trig_EN           : 1;
  __REG32                   : 8;
} __pdma_csrx_bits;

/* PDMA Transfer Byte Count Register (PDMA_BCRx) */
typedef struct {
  __REG32 PDMA_BCR          :16;
  __REG32                   :16;
} __pdma_bcrx_bits;

/* PDMA Internal Buffer Pointer Register (PDMA_POINTx) */
typedef struct {
  __REG32 PDMA_POINT        : 2;
  __REG32                   :30;
} __pdma_pointx_bits;

/* PDMA Current Byte Count Register (PDMA_CBCRx) */
typedef struct {
  __REG32 PDMA_CBCR         :24;
  __REG32                   : 8;
} __pdma_cbcrx_bits;

/* PDMA Interrupt Enable Control Register (PDMA_IERx) */
typedef struct {
  __REG32 TABORT_IE         : 1;
  __REG32 BLKD_IE           : 1;
  __REG32 WAR_IE            : 1;
  __REG32                   :29;
} __pdma_ierx_bits;

/* PDMA Interrupt Status Register (PDMA_ISRx) */
typedef struct {
  __REG32 TABORT_IF         : 1;
  __REG32 BLKD_IF           : 1;
  __REG32                   : 6;
  __REG32 WAR_BCR_IF        : 4;
  __REG32                   :19;
  __REG32 INTR              : 1;
} __pdma_isrx_bits;

/* PDMA Global Control and Status Register (PDMA_GCRCSR) */
typedef struct {
  __REG32 PDMA_RST          : 1;
  __REG32                   : 7;
  __REG32 HCLK0_EN          : 1;
  __REG32 HCLK1_EN          : 1;
  __REG32 HCLK2_EN          : 1;
  __REG32 HCLK3_EN          : 1;
  __REG32 HCLK4_EN          : 1;
  __REG32 HCLK5_EN          : 1;
  __REG32 HCLK6_EN          : 1;
  __REG32 HCLK7_EN          : 1;
  __REG32 HCLK8_EN          : 1;
  __REG32                   :15;
} __pdma_gcrcsr_bits;

/* PDMA Service Selection Control Register 0 (PDSSR0) */
typedef struct {
  __REG32 SPI0_RXSEL        : 4;
  __REG32 SPI0_TXSEL        : 4;
  __REG32 SPI1_RXSEL        : 4;
  __REG32 SPI1_TXSEL        : 4;
  __REG32                   :16;
} __pdma_pdssr0_bits;

/* PDMA Service Selection Control Register 1 (PDSSR1) */
typedef struct {
  __REG32 UART0_RXSEL       : 4;
  __REG32 UART0_TXSEL       : 4;
  __REG32 UART1_RXSEL       : 4;
  __REG32 UART1_TXSEL       : 4;
  __REG32                   : 8;
  __REG32 ADC_RXSEL         : 4;
  __REG32 ADC_TXSEL         : 4;
} __pdma_pdssr1_bits;

/* PDMA Global Interrupt Status Register (PDMA_GCRISR) */
typedef struct {
  __REG32 INTR0             : 1;
  __REG32 INTR1             : 1;
  __REG32 INTR2             : 1;
  __REG32 INTR3             : 1;
  __REG32 INTR4             : 1;
  __REG32 INTR5             : 1;
  __REG32 INTR6             : 1;
  __REG32 INTR7             : 1;
  __REG32 INTR8             : 1;
  __REG32                   :22;
  __REG32 INTR              : 1;
} __pdma_gcrisr_bits;

/* PDMA Service Selection Control Register 2 (PDSSR2) */
typedef struct {
  __REG32 I2S_RXSEL         : 4;
  __REG32 I2S_TXSEL         : 4;
  __REG32                   :24;
} __pdma_pdssr2_bits;

/* ISP Control Register (ISPCON) */
typedef struct {
  __REG32 ISPEN             : 1;
  __REG32 BS                : 1;
  __REG32                   : 2;
  __REG32 CFGUEN            : 1;
  __REG32 LDUEN             : 1;
  __REG32 ISPFF             : 1;
  __REG32 SWRST             : 1;
  __REG32 PT                : 3;
  __REG32                   : 1;
  __REG32 ET                : 3;
  __REG32                   :17;
} __ispcon_bits;

/* ISP Command (ISPCMD) */
typedef struct {
  __REG32 FCTRL0            : 1;
  __REG32 FCTRL1            : 1;
  __REG32 FCTRL2            : 1;
  __REG32 FCTRL3            : 1;
  __REG32 FCEN              : 1;
  __REG32 FOEN              : 1;
  __REG32                   :26;
} __ispcmd_bits;

/* ISP Trigger Control Register (ISPTRG) */
typedef struct {
  __REG32 ISPGO             : 1;
  __REG32                   :31;
} __isptrg_bits;

/* Flash Access Time Control Register (FATCON) */
typedef struct {
  __REG32 FPSEN             : 1;
  __REG32 FATS              : 3;
  __REG32                   :28;
} __fatcon_bits;

/* External Bus Interface Control Register (EBICON) */
typedef struct {
  __REG32 EXTEN             : 1;
  __REG32 EXTB16            : 1;
  __REG32                   : 6;
  __REG32 MCLKDIV           : 3;
  __REG32                   : 5;
  __REG32 EXTTALE           : 3;
  __REG32                   :13;
} __ebicon_bits;

/* External Bus Interface Timing Control Register (EXTIME) */
typedef struct {
  __REG32                   : 3;
  __REG32 EXTTACC           : 5;
  __REG32 EXTTAHD           : 3;
  __REG32                   : 1;
  __REG32 EXTIW2X           : 4;
  __REG32                   : 8;
  __REG32 EXTIR2X           : 4;
  __REG32                   : 4;
} __extime_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
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
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED1C,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHPR3,                 0xE000ED20,__READ_WRITE ,__shpr3_bits);

/***************************************************************************
 **
 ** GCR (System Manager Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(RSTSRC,            0x50000004,__READ_WRITE ,__rstsrc_bits);
__IO_REG32_BIT(IPRSTC1,           0x50000008,__READ_WRITE ,__iprstc1_bits);
__IO_REG32_BIT(IPRSTC2,           0x5000000C,__READ_WRITE ,__iprstc2_bits);
__IO_REG32_BIT(BODCR,             0x50000018,__READ_WRITE ,__bodcr_bits);
__IO_REG32_BIT(TEMPCR,            0x5000001C,__READ_WRITE ,__tempcr_bits);
__IO_REG32_BIT(PORCR,             0x50000024,__READ_WRITE ,__porcr_bits);
__IO_REG32_BIT(GPA_MFP,           0x50000030,__READ_WRITE ,__gpa_mfp_bits);
__IO_REG32_BIT(GPB_MFP,           0x50000034,__READ_WRITE ,__gpb_mfp_bits);
__IO_REG32_BIT(GPC_MFP,           0x50000038,__READ_WRITE ,__gpc_mfp_bits);
__IO_REG32_BIT(GPD_MFP,           0x5000003C,__READ_WRITE ,__gpd_mfp_bits);
__IO_REG32_BIT(GPE_MFP,           0x50000040,__READ_WRITE ,__gpe_mfp_bits);
__IO_REG32_BIT(ALT_MFP,           0x50000050,__READ_WRITE ,__alt_mfp_bits);
__IO_REG32_BIT(RegLockAddr,       0x50000100,__READ_WRITE ,__reglockaddr_bits);
__IO_REG32_BIT(RCADJ,             0x50000110,__READ_WRITE ,__rcadj_bits);

/***************************************************************************
 **
 ** Interrupt Source Control
 **
 ***************************************************************************/
__IO_REG32_BIT(IRQ0_SRC,           0x50000300,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ1_SRC,           0x50000304,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ2_SRC,           0x50000308,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ3_SRC,           0x5000030C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ4_SRC,           0x50000310,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ5_SRC,           0x50000314,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ6_SRC,           0x50000318,__READ       ,__irq6_src_bits);
__IO_REG32_BIT(IRQ7_SRC,           0x5000031C,__READ       ,__irq6_src_bits);
__IO_REG32_BIT(IRQ8_SRC,           0x50000320,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ9_SRC,           0x50000324,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ10_SRC,          0x50000328,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ11_SRC,          0x5000032C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ12_SRC,          0x50000330,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ13_SRC,          0x50000334,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ14_SRC,          0x50000338,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ15_SRC,          0x5000033C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ16_SRC,          0x50000340,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ17_SRC,          0x50000344,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ18_SRC,          0x50000348,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ19_SRC,          0x5000034C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ20_SRC,          0x50000350,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ21_SRC,          0x50000354,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ22_SRC,          0x50000358,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ23_SRC,          0x5000035C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ24_SRC,          0x50000360,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ25_SRC,          0x50000364,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ26_SRC,          0x50000368,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ27_SRC,          0x5000036C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ28_SRC,          0x50000370,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ29_SRC,          0x50000374,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ30_SRC,          0x50000378,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(IRQ31_SRC,          0x5000037C,__READ       ,__irqx_src_bits);
__IO_REG32_BIT(NMI_SEL,            0x50000380,__READ_WRITE ,__nmi_sel_bits);
__IO_REG32_BIT(MCU_IRQ,            0x50000384,__READ_WRITE ,__mcu_irq_bits);

/***************************************************************************
 **
 ** CLK (Clock Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PWRCON,                0x50000200,__READ_WRITE ,__pwrcon_bits);
__IO_REG32_BIT(AHBCLK,                0x50000204,__READ_WRITE ,__ahbclk_bits);
__IO_REG32_BIT(APBCLK,                0x50000208,__READ_WRITE ,__apbclk_bits);
__IO_REG32_BIT(CLKSEL0,               0x50000210,__READ_WRITE ,__clksel0_bits);
__IO_REG32_BIT(CLKSEL1,               0x50000214,__READ_WRITE ,__clksel1_bits);
__IO_REG32_BIT(CLKSEL2,               0x5000021C,__READ_WRITE ,__clksel2_bits);
__IO_REG32_BIT(CLKDIV ,               0x50000218,__READ_WRITE ,__clkdiv_bits);
__IO_REG32_BIT(PLLCON,                0x50000220,__READ_WRITE ,__pllcon_bits);
__IO_REG32_BIT(FRQDIV ,               0x50000224,__READ_WRITE ,__frqdiv_bits);

/***************************************************************************
 **
 ** GPIO (General Purpose I/O)
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOA_PMD,             0x50004000,__READ_WRITE ,__gpiox_pmd_bits);
__IO_REG32_BIT(GPIOA_OFFD,            0x50004004,__READ_WRITE ,__gpiox_offd_bits);
__IO_REG32_BIT(GPIOA_DOUT,            0x50004008,__READ_WRITE ,__gpiox_dout_bits);
__IO_REG32_BIT(GPIOA_DMASK,           0x5000400C,__READ_WRITE ,__gpiox_dmask_bits);
__IO_REG32_BIT(GPIOA_PIN,             0x50004010,__READ       ,__gpiox_pin_bits);
__IO_REG32_BIT(GPIOA_DBEN,            0x50004014,__READ_WRITE ,__gpiox_dben_bits);
__IO_REG32_BIT(GPIOA_IMD,             0x50004018,__READ_WRITE ,__gpiox_imd_bits);
__IO_REG32_BIT(GPIOA_IEN,             0x5000401C,__READ_WRITE ,__gpiox_ien_bits);
__IO_REG32_BIT(GPIOA_ISRC,            0x50004020,__READ_WRITE ,__gpiox_isrc_bits);
__IO_REG32_BIT(GPIOB_PMD,             0x50004040,__READ_WRITE ,__gpiox_pmd_bits);
__IO_REG32_BIT(GPIOB_OFFD,            0x50004044,__READ_WRITE ,__gpiox_offd_bits);
__IO_REG32_BIT(GPIOB_DOUT,            0x50004048,__READ_WRITE ,__gpiox_dout_bits);
__IO_REG32_BIT(GPIOB_DMASK,           0x5000404C,__READ_WRITE ,__gpiox_dmask_bits);
__IO_REG32_BIT(GPIOB_PIN,             0x50004050,__READ       ,__gpiox_pin_bits);
__IO_REG32_BIT(GPIOB_DBEN,            0x50004054,__READ_WRITE ,__gpiox_dben_bits);
__IO_REG32_BIT(GPIOB_IMD,             0x50004058,__READ_WRITE ,__gpiox_imd_bits);
__IO_REG32_BIT(GPIOB_IEN,             0x5000405C,__READ_WRITE ,__gpiox_ien_bits);
__IO_REG32_BIT(GPIOB_ISRC,            0x50004060,__READ_WRITE ,__gpiox_isrc_bits);
__IO_REG32_BIT(GPIOC_PMD,             0x50004080,__READ_WRITE ,__gpiox_pmd_bits);
__IO_REG32_BIT(GPIOC_OFFD,            0x50004084,__READ_WRITE ,__gpiox_offd_bits);
__IO_REG32_BIT(GPIOC_DOUT,            0x50004088,__READ_WRITE ,__gpiox_dout_bits);
__IO_REG32_BIT(GPIOC_DMASK,           0x5000408C,__READ_WRITE ,__gpiox_dmask_bits);
__IO_REG32_BIT(GPIOC_PIN,             0x50004090,__READ       ,__gpiox_pin_bits);
__IO_REG32_BIT(GPIOC_DBEN,            0x50004094,__READ_WRITE ,__gpiox_dben_bits);
__IO_REG32_BIT(GPIOC_IMD,             0x50004098,__READ_WRITE ,__gpiox_imd_bits);
__IO_REG32_BIT(GPIOC_IEN,             0x5000409C,__READ_WRITE ,__gpiox_ien_bits);
__IO_REG32_BIT(GPIOC_ISRC,            0x500040A0,__READ_WRITE ,__gpiox_isrc_bits);
__IO_REG32_BIT(GPIOD_PMD,             0x500040C0,__READ_WRITE ,__gpiox_pmd_bits);
__IO_REG32_BIT(GPIOD_OFFD,            0x500040C4,__READ_WRITE ,__gpiox_offd_bits);
__IO_REG32_BIT(GPIOD_DOUT,            0x500040C8,__READ_WRITE ,__gpiox_dout_bits);
__IO_REG32_BIT(GPIOD_DMASK,           0x500040CC,__READ_WRITE ,__gpiox_dmask_bits);
__IO_REG32_BIT(GPIOD_PIN,             0x500040D0,__READ       ,__gpiox_pin_bits);
__IO_REG32_BIT(GPIOD_DBEN,            0x500040D4,__READ_WRITE ,__gpiox_dben_bits);
__IO_REG32_BIT(GPIOD_IMD,             0x500040D8,__READ_WRITE ,__gpiox_imd_bits);
__IO_REG32_BIT(GPIOD_IEN,             0x500040DC,__READ_WRITE ,__gpiox_ien_bits);
__IO_REG32_BIT(GPIOD_ISRC,            0x500040E0,__READ_WRITE ,__gpiox_isrc_bits);
__IO_REG32_BIT(GPIOE_PMD,             0x50004100,__READ_WRITE ,__gpiox_pmd_bits);
__IO_REG32_BIT(GPIOE_OFFD,            0x50004104,__READ_WRITE ,__gpiox_offd_bits);
__IO_REG32_BIT(GPIOE_DOUT,            0x50004108,__READ_WRITE ,__gpiox_dout_bits);
__IO_REG32_BIT(GPIOE_DMASK,           0x5000410C,__READ_WRITE ,__gpiox_dmask_bits);
__IO_REG32_BIT(GPIOE_PIN,             0x50004110,__READ       ,__gpiox_pin_bits);
__IO_REG32_BIT(GPIOE_DBEN,            0x50004114,__READ_WRITE ,__gpiox_dben_bits);
__IO_REG32_BIT(GPIOE_IMD,             0x50004118,__READ_WRITE ,__gpiox_imd_bits);
__IO_REG32_BIT(GPIOE_IEN,             0x5000411C,__READ_WRITE ,__gpiox_ien_bits);
__IO_REG32_BIT(GPIOE_ISRC,            0x50004120,__READ_WRITE ,__gpiox_isrc_bits);
__IO_REG32_BIT(DBNCECON,              0x50004180,__READ_WRITE ,__dbncecon_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CON,              0x40020000,__READ_WRITE ,__i2cxcon_bits);
__IO_REG32_BIT(I2C0ADDR0,            0x40020004,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C0DAT,              0x40020008,__READ_WRITE ,__i2cxdat_bits);
__IO_REG32_BIT(I2C0STATUS,           0x4002000C,__READ       ,__i2cxstatus_bits);
__IO_REG32_BIT(I2C0CLK,              0x40020010,__READ_WRITE ,__i2cxclk_bits);
__IO_REG32_BIT(I2C0TOC,              0x40020014,__READ_WRITE ,__i2cxtoc_bits);
__IO_REG32_BIT(I2C0ADDR1,            0x40020018,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C0ADDR2,            0x4002001C,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C0ADDR3,            0x40020020,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C0ADM0,             0x40020024,__READ_WRITE ,__i2cxadmy_bits);
__IO_REG32_BIT(I2C0ADM1,             0x40020028,__READ_WRITE ,__i2cxadmy_bits);
__IO_REG32_BIT(I2C0ADM2,             0x4002002C,__READ_WRITE ,__i2cxadmy_bits);
__IO_REG32_BIT(I2C0ADM3,             0x40020030,__READ_WRITE ,__i2cxadmy_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CON,              0x40120000,__READ_WRITE ,__i2cxcon_bits);
__IO_REG32_BIT(I2C1ADDR0,            0x40120004,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C1DAT,              0x40120008,__READ_WRITE ,__i2cxdat_bits);
__IO_REG32_BIT(I2C1STATUS,           0x4012000C,__READ       ,__i2cxstatus_bits);
__IO_REG32_BIT(I2C1CLK,              0x40120010,__READ_WRITE ,__i2cxclk_bits);
__IO_REG32_BIT(I2C1TOC,              0x40120014,__READ_WRITE ,__i2cxtoc_bits);
__IO_REG32_BIT(I2C1ADDR1,            0x40120018,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C1ADDR2,            0x4012001C,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C1ADDR3,            0x40120020,__READ_WRITE ,__i2cxaddry_bits);
__IO_REG32_BIT(I2C1ADM0,             0x40120024,__READ_WRITE ,__i2cxadmy_bits);
__IO_REG32_BIT(I2C1ADM1,             0x40120028,__READ_WRITE ,__i2cxadmy_bits);
__IO_REG32_BIT(I2C1ADM2,             0x4012002C,__READ_WRITE ,__i2cxadmy_bits);
__IO_REG32_BIT(I2C1ADM3,             0x40120030,__READ_WRITE ,__i2cxadmy_bits);

/***************************************************************************
 **
 ** PWM03
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_PPR03,             0x40040000,__READ_WRITE ,__pwm_ppr03_bits);
__IO_REG32_BIT(PWM_CSR03,             0x40040004,__READ_WRITE ,__pwm_csr03_bits);
__IO_REG32_BIT(PWM_PCR03,             0x40040008,__READ_WRITE ,__pwm_pcr03_bits);
__IO_REG32_BIT(PWM_CNR0,              0x4004000C,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR0,              0x40040010,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR0,              0x40040014,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR1,              0x40040018,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR1,              0x4004001C,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR1,              0x40040020,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR2,              0x40040024,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR2,              0x40040028,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR2,              0x4004002C,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR3,              0x40040030,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR3,              0x40040034,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR3,              0x40040038,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_PIER03,            0x40040040,__READ_WRITE ,__pwm_pier03_bits);
__IO_REG32_BIT(PWM_PIIR03,            0x40040044,__READ_WRITE ,__pwm_piir03_bits);
__IO_REG32_BIT(PWM_CCR0,              0x40040050,__READ_WRITE ,__pwm_ccr0_bits);
__IO_REG32_BIT(PWM_CCR2,              0x40040054,__READ_WRITE ,__pwm_ccr2_bits);
__IO_REG32_BIT(PWM_CRLR0,             0x40040058,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR0,             0x4004005C,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CRLR1,             0x40040060,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR1,             0x40040064,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CRLR2,             0x40040068,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR2,             0x4004006C,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CRLR3,             0x40040070,__READ_WRITE ,__pwm_crlr_bits);
__IO_REG32_BIT(PWM_CFLR3,             0x40040074,__READ_WRITE ,__pwm_cflr_bits);
__IO_REG32_BIT(PWM_CAPENR03,          0x40040078,__READ_WRITE ,__pwm_capenr03_bits);
__IO_REG32_BIT(PWM_POE03,             0x4004007C,__READ_WRITE ,__pwm_poe03_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_INIR,              0x40008000,__READ_WRITE ,__rtc_inir_bits);
__IO_REG32_BIT(RTC_AER,               0x40008004,__READ_WRITE ,__rtc_aer_bits);
__IO_REG32_BIT(RTC_FCR,               0x40008008,__READ_WRITE ,__rtc_fcr_bits);
__IO_REG32_BIT(RTC_TLR,               0x4000800C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CLR,               0x40008010,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_TSSR,              0x40008014,__READ_WRITE ,__rtc_tssr_bits);
__IO_REG32_BIT(RTC_DWR,               0x40008018,__READ_WRITE ,__rtc_dwr_bits);
__IO_REG32_BIT(RTC_TAR,               0x4000801C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CAR,               0x40008020,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_LIR,               0x40008024,__READ       ,__rtc_lir_bits);
__IO_REG32_BIT(RTC_RIER,              0x40008028,__READ_WRITE ,__rtc_rier_bits);
__IO_REG32_BIT(RTC_RIIR,              0x4000802C,__READ_WRITE ,__rtc_riir_bits);
__IO_REG32_BIT(RTC_TTR,               0x40008030,__READ_WRITE ,__rtc_ttr_bits);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CNTRL,            0x40030000,__READ_WRITE ,__spi_cntrl_bits);
__IO_REG32_BIT(SPI0_DIVIDER,          0x40030004,__READ_WRITE ,__spi_divider_bits);
__IO_REG32_BIT(SPI0_SSR,              0x40030008,__READ_WRITE ,__spi_ssr_bits);
__IO_REG32(    SPI0_Rx0,              0x40030010,__READ       );
__IO_REG32(    SPI0_Rx1,              0x40030014,__READ       );
__IO_REG32(    SPI0_Tx0,              0x40030020,__WRITE      );
__IO_REG32(    SPI0_Tx1,              0x40030024,__WRITE      );
__IO_REG32_BIT(SPI0_DMA,              0x40030038,__READ_WRITE ,__spi_dma_bits);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CNTRL,            0x40034000,__READ_WRITE ,__spi_cntrl_bits);
__IO_REG32_BIT(SPI1_DIVIDER,          0x40034004,__READ_WRITE ,__spi_divider_bits);
__IO_REG32_BIT(SPI1_SSR,              0x40034008,__READ_WRITE ,__spi_ssr_bits);
__IO_REG32(    SPI1_Rx0,              0x40034010,__READ       );
__IO_REG32(    SPI1_Rx1,              0x40034014,__READ       );
__IO_REG32(    SPI1_Tx0,              0x40034020,__WRITE      );
__IO_REG32(    SPI1_Tx1,              0x40034024,__WRITE      );
__IO_REG32_BIT(SPI1_DMA,              0x40034038,__READ_WRITE ,__spi_dma_bits);

/***************************************************************************
 **
 ** TIMER 0
 **
 ***************************************************************************/
__IO_REG32_BIT(TCSR0,                 0x40010000,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCMPR0,                0x40010004,__READ_WRITE ,__tcmpr_bits);
__IO_REG32_BIT(TISR0,                 0x40010008,__READ_WRITE ,__tisr_bits);
__IO_REG32_BIT(TDR0,                  0x4001000C,__READ       ,__tdr_bits);

/***************************************************************************
 **
 ** TIMER 1
 **
 ***************************************************************************/
__IO_REG32_BIT(TCSR1,                 0x40010020,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCMPR1,                0x40010024,__READ_WRITE ,__tcmpr_bits);
__IO_REG32_BIT(TISR1,                 0x40010028,__READ_WRITE ,__tisr_bits);
__IO_REG32_BIT(TDR1,                  0x4001002C,__READ       ,__tdr_bits);

/***************************************************************************
 **
 ** TIMER 2
 **
 ***************************************************************************/
__IO_REG32_BIT(TCSR2,                 0x40110000,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCMPR2,                0x40110004,__READ_WRITE ,__tcmpr_bits);
__IO_REG32_BIT(TISR2,                 0x40110008,__READ_WRITE ,__tisr_bits);
__IO_REG32_BIT(TDR2,                  0x4011000C,__READ       ,__tdr_bits);

/***************************************************************************
 **
 ** TIMER 3
 **
 ***************************************************************************/
__IO_REG32_BIT(TCSR3,                 0x40110020,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCMPR3,                0x40110024,__READ_WRITE ,__tcmpr_bits);
__IO_REG32_BIT(TISR3,                 0x40110028,__READ_WRITE ,__tisr_bits);
__IO_REG32_BIT(TDR3,                  0x4011002C,__READ       ,__tdr_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WTCR,                  0x40004000,__READ_WRITE ,__wtcr_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG32_BIT(UA0_RBR,               0x40050000,__READ_WRITE ,__uart_rbr_bits);
#define UA0_THR     UA0_RBR
#define UA0_THR_bit UA0_RBR_bit
__IO_REG32_BIT(UA0_IER,               0x40050004,__READ_WRITE ,__uart_ier_bits);
__IO_REG32_BIT(UA0_FCR,               0x40050008,__READ_WRITE ,__uart_fcr_bits);
__IO_REG32_BIT(UA0_LCR,               0x4005000C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UA0_MCR,               0x40050010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UA0_MSR,               0x40050014,__READ_WRITE ,__uart_msr_bits);
__IO_REG32_BIT(UA0_FSR,               0x40050018,__READ_WRITE ,__uart_fsr_bits);
__IO_REG32_BIT(UA0_ISR,               0x4005001C,__READ_WRITE ,__uart_isr_bits);
__IO_REG32_BIT(UA0_TOR,               0x40050020,__READ_WRITE ,__uart_tor_bits);
__IO_REG32_BIT(UA0_BAUD,              0x40050024,__READ_WRITE ,__uart_baud_bits);
__IO_REG32_BIT(UA0_IRCR,              0x40050028,__READ_WRITE ,__uart_ircr_bits);
__IO_REG32_BIT(UA0_LIN_BCNT,          0x4005002C,__READ_WRITE ,__uart_lin_bcnt_bits);
__IO_REG32_BIT(UA0_FUN_SEL,           0x40050030,__READ_WRITE ,__uart_fun_sel_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(UA1_RBR,               0x40150000,__READ_WRITE ,__uart_rbr_bits);
#define UA1_THR     UA1_RBR
#define UA1_THR_bit UA1_RBR_bit
__IO_REG32_BIT(UA1_IER,               0x40150004,__READ_WRITE ,__uart_ier_bits);
__IO_REG32_BIT(UA1_FCR,               0x40150008,__READ_WRITE ,__uart_fcr_bits);
__IO_REG32_BIT(UA1_LCR,               0x4015000C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UA1_MCR,               0x40150010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UA1_MSR,               0x40150014,__READ_WRITE ,__uart_msr_bits);
__IO_REG32_BIT(UA1_FSR,               0x40150018,__READ_WRITE ,__uart_fsr_bits);
__IO_REG32_BIT(UA1_ISR,               0x4015001C,__READ_WRITE ,__uart_isr_bits);
__IO_REG32_BIT(UA1_TOR,               0x40150020,__READ_WRITE ,__uart_tor_bits);
__IO_REG32_BIT(UA1_BAUD,              0x40150024,__READ_WRITE ,__uart_baud_bits);
__IO_REG32_BIT(UA1_IRCR,              0x40150028,__READ_WRITE ,__uart_ircr_bits);
__IO_REG32_BIT(UA1_LIN_BCNT,          0x4015002C,__READ_WRITE ,__uart_lin_bcnt_bits);
__IO_REG32_BIT(UA1_FUN_SEL,           0x40150030,__READ_WRITE ,__uart_fun_sel_bits);

/***************************************************************************
 **
 ** PS2
 **
 ***************************************************************************/
__IO_REG32_BIT(PS2CON,                0x40100000,__READ_WRITE ,__ps2con_bits);
__IO_REG32(    PS2TXDATA0,            0x40100004,__READ_WRITE);
__IO_REG32(    PS2TXDATA1,            0x40100008,__READ_WRITE);
__IO_REG32(    PS2TXDATA2,            0x4010000C,__READ_WRITE);
__IO_REG32(    PS2TXDATA3,            0x40100010,__READ_WRITE);
__IO_REG32_BIT(PS2RXDATA,             0x40100014,__READ       ,__ps2rxdata_bits);
__IO_REG32_BIT(PS2STATUS,             0x40100018,__READ_WRITE ,__ps2status_bits);
__IO_REG32_BIT(PS2INTID,              0x4010001C,__READ_WRITE ,__ps2intid_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_CON,               0x401A0000,__READ_WRITE ,__i2s_con_bits);
__IO_REG32_BIT(I2S_CLKDIV,            0x401A0004,__READ_WRITE ,__i2s_clkdiv_bits);
__IO_REG32_BIT(I2S_IE,                0x401A0008,__READ_WRITE ,__i2s_ie_bits);
__IO_REG32_BIT(I2S_STATUS,            0x401A000C,__READ_WRITE ,__i2s_status_bits);
__IO_REG32(    I2S_TXFIFO,            0x401A0010,__READ_WRITE );
__IO_REG32(    I2S_RXFIFO,            0x401A0014,__READ_WRITE );

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADDR0,                 0x400E0000,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR1,                 0x400E0004,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR2,                 0x400E0008,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR3,                 0x400E000C,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR4,                 0x400E0010,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR5,                 0x400E0014,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR6,                 0x400E0018,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR7,                 0x400E001C,__READ       ,__addr_bits);
__IO_REG32_BIT(ADCR,                  0x400E0020,__READ_WRITE ,__adcr_bits);
__IO_REG32_BIT(ADCHER,                0x400E0024,__READ_WRITE ,__adcher_bits);
__IO_REG32_BIT(ADCMPR0,               0x400E0028,__READ_WRITE ,__adcmprx_bits);
__IO_REG32_BIT(ADCMPR1,               0x400E002C,__READ_WRITE ,__adcmprx_bits);
__IO_REG32_BIT(ADSR,                  0x400E0030,__READ_WRITE ,__adsr_bits);
__IO_REG32_BIT(ADCALR,                0x400E0034,__READ_WRITE ,__adcalr_bits);
__IO_REG32_BIT(ADPDMA,                0x400E0040,__READ       ,__adpdma_bits);

/***************************************************************************
 **
 ** Analog Comparator
 **
 ***************************************************************************/
__IO_REG32_BIT(CMP0CR,                0x400D0000,__READ_WRITE ,__cmp0cr_bits);
__IO_REG32_BIT(CMP1CR,                0x400D0004,__READ_WRITE ,__cmp1cr_bits);
__IO_REG32_BIT(CMPSR,                 0x400D0008,__READ_WRITE ,__cmpsr_bits);

/***************************************************************************
 **
 ** PDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(PDMA_CSR0,           0x50008000,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR0,           0x50008004,__READ_WRITE );
__IO_REG32(    PDMA_DAR0,           0x50008008,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR0,           0x5000800C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT0,         0x50008010,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR0,          0x50008014,__READ       );
__IO_REG32(    PDMA_CDAR0,          0x50008018,__READ       );
__IO_REG32_BIT(PDMA_CBCR0,          0x5000801C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER0,           0x50008020,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR0,           0x50008024,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c0,       0x50008080,__READ       );
__IO_REG32_BIT(PDMA_CSR1,           0x50008100,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR1,           0x50008104,__READ_WRITE );
__IO_REG32(    PDMA_DAR1,           0x50008108,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR1,           0x5000810C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT1,         0x50008110,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR1,          0x50008114,__READ       );
__IO_REG32(    PDMA_CDAR1,          0x50008118,__READ       );
__IO_REG32_BIT(PDMA_CBCR1,          0x5000811C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER1,           0x50008120,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR1,           0x50008124,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c1,       0x50008180,__READ       );
__IO_REG32_BIT(PDMA_CSR2,           0x50008200,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR2,           0x50008204,__READ_WRITE );
__IO_REG32(    PDMA_DAR2,           0x50008208,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR2,           0x5000820C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT2,         0x50008210,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR2,          0x50008214,__READ       );
__IO_REG32(    PDMA_CDAR2,          0x50008218,__READ       );
__IO_REG32_BIT(PDMA_CBCR2,          0x5000821C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER2,           0x50008220,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR2,           0x50008224,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c2,       0x50008280,__READ       );
__IO_REG32_BIT(PDMA_CSR3,           0x50008300,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR3,           0x50008304,__READ_WRITE );
__IO_REG32(    PDMA_DAR3,           0x50008308,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR3,           0x5000830C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT3,         0x50008310,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR3,          0x50008314,__READ       );
__IO_REG32(    PDMA_CDAR3,          0x50008318,__READ       );
__IO_REG32_BIT(PDMA_CBCR3,          0x5000831C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER3,           0x50008320,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR3,           0x50008324,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c3,       0x50008380,__READ       );
__IO_REG32_BIT(PDMA_CSR4,           0x50008400,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR4,           0x50008404,__READ_WRITE );
__IO_REG32(    PDMA_DAR4,           0x50008408,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR4,           0x5000840C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT4,         0x50008410,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR4,          0x50008414,__READ       );
__IO_REG32(    PDMA_CDAR4,          0x50008418,__READ       );
__IO_REG32_BIT(PDMA_CBCR4,          0x5000841C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER4,           0x50008420,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR4,           0x50008424,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c4,       0x50008480,__READ       );
__IO_REG32_BIT(PDMA_CSR5,           0x50008500,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR5,           0x50008504,__READ_WRITE );
__IO_REG32(    PDMA_DAR5,           0x50008508,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR5,           0x5000850C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT5,         0x50008510,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR5,          0x50008514,__READ       );
__IO_REG32(    PDMA_CDAR5,          0x50008518,__READ       );
__IO_REG32_BIT(PDMA_CBCR5,          0x5000851C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER5,           0x50008520,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR5,           0x50008524,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c5,       0x50008580,__READ       );
__IO_REG32_BIT(PDMA_CSR6,           0x50008600,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR6,           0x50008604,__READ_WRITE );
__IO_REG32(    PDMA_DAR6,           0x50008608,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR6,           0x5000860C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT6,         0x50008610,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR6,          0x50008614,__READ       );
__IO_REG32(    PDMA_CDAR6,          0x50008618,__READ       );
__IO_REG32_BIT(PDMA_CBCR6,          0x5000861C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER6,           0x50008620,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR6,           0x50008624,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c6,       0x50008680,__READ       );
__IO_REG32_BIT(PDMA_CSR7,           0x50008700,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR7,           0x50008704,__READ_WRITE );
__IO_REG32(    PDMA_DAR7,           0x50008708,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR7,           0x5000870C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT7,         0x50008710,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR7,          0x50008714,__READ       );
__IO_REG32(    PDMA_CDAR7,          0x50008718,__READ       );
__IO_REG32_BIT(PDMA_CBCR7,          0x5000871C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER7,           0x50008720,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR7,           0x50008724,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c7,       0x50008780,__READ       );
__IO_REG32_BIT(PDMA_CSR8,           0x50008800,__READ_WRITE ,__pdma_csrx_bits);
__IO_REG32(    PDMA_SAR8,           0x50008804,__READ_WRITE );
__IO_REG32(    PDMA_DAR8,           0x50008808,__READ_WRITE );
__IO_REG32_BIT(PDMA_BCR8,           0x5000880C,__READ_WRITE ,__pdma_bcrx_bits);
__IO_REG32_BIT(PDMA_POINT8,         0x50008810,__READ       ,__pdma_pointx_bits);
__IO_REG32(    PDMA_CSAR8,          0x50008814,__READ       );
__IO_REG32(    PDMA_CDAR8,          0x50008818,__READ       );
__IO_REG32_BIT(PDMA_CBCR8,          0x5000881C,__READ       ,__pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA_IER8,           0x50008820,__READ_WRITE ,__pdma_ierx_bits);
__IO_REG32_BIT(PDMA_ISR8,           0x50008824,__READ_WRITE ,__pdma_isrx_bits);
__IO_REG32(    PDMA_SBUF0_c8,       0x50008880,__READ       );
__IO_REG32_BIT(PDMA_GCRCSR,         0x50008F00,__READ_WRITE ,__pdma_gcrcsr_bits);
__IO_REG32_BIT(PDMA_PDSSR0,         0x50008F04,__READ_WRITE ,__pdma_pdssr0_bits);
__IO_REG32_BIT(PDMA_PDSSR1,         0x50008F08,__READ_WRITE ,__pdma_pdssr1_bits);
__IO_REG32_BIT(PDMA_GCRISR,         0x50008F0C,__READ_WRITE ,__pdma_gcrisr_bits);
__IO_REG32_BIT(PDMA_PDSSR2,         0x50008F10,__READ_WRITE ,__pdma_pdssr2_bits);

/***************************************************************************
 **
 ** FMC
 **
 ***************************************************************************/
__IO_REG32_BIT(ISPCON,              0x5000C000,__READ_WRITE ,__ispcon_bits);
__IO_REG32(    ISPADR,              0x5000C004,__READ_WRITE );
__IO_REG32(    ISPDAT,              0x5000C008,__READ_WRITE );
__IO_REG32_BIT(ISPCMD,              0x5000C00C,__READ_WRITE ,__ispcmd_bits);
__IO_REG32_BIT(ISPTRG,              0x5000C010,__READ_WRITE ,__isptrg_bits);
__IO_REG32(    DFBADR,              0x5000C014,__READ       );
__IO_REG32_BIT(FATCON,              0x5000C018,__READ_WRITE ,__fatcon_bits);

/***************************************************************************
 **
 ** EBI
 **
 ***************************************************************************/
__IO_REG32_BIT(EBICON,              0x50010000,__READ_WRITE ,__ebicon_bits);
__IO_REG32_BIT(EXTIME,              0x50010004,__READ_WRITE ,__extime_bits);

/***************************************************************************
 **  Assembler specific declarations
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
#define SVCI                  11  /* SVCall                                                 */
#define PSI                   14  /* PendSV                                                 */
#define STI                   15  /* SysTick                                                */
#define NVIC_BOD_OUT          16  /* Brownout low voltage detected interrupt                */
#define NVIC_WDT_INT          17  /* Watch Dog Timer interrupt                              */
#define NVIC_EINT0            18  /* External signal interrupt from PB.14 pin               */
#define NVIC_EINT1            19  /* External signal interrupt from PB.15 pin               */
#define NVIC_GPAB_INT         20  /* External signal interrupt from PA[15:0] / PB[13:0]     */                                    */
#define NVIC_GPCDE_INT        21  /* External interrupt from PC[15:0]/PD[15:0]/PE[15:0]     */
#define NVIC_PWMA_INT         22  /* PWM0, PWM1, PWM2 and PWM3 interrupt                    */
#define NVIC_TMR0_INT         24  /* Timer 0 interrupt                                      */
#define NVIC_TMR1_INT         25  /* Timer 1 interrupt                                      */
#define NVIC_TMR2_INT         26  /* Timer 2 interrupt                                      */
#define NVIC_TMR3_INT         27  /* Timer 3 interrupt                                      */
#define NVIC_UART02_INT       28  /* UART0 and UART2 interrupt                              */
#define NVIC_UART1_INT        29  /* UART1 interrupt                                        */
#define NVIC_SPI0_INT         30  /* SPI0 interrupt                                         */
#define NVIC_SPI1_INT         31  /* SPI1 interrupt                                         */
#define NVIC_I2C0_INT         34  /* I2C0 interrupt                                         */
#define NVIC_I2C2_INT         35  /* I2C1 interrupt                                         */
#define NVIC_PS2_INT          40  /* PS2 interrupt                                          */
#define NVIC_ACMP_INT         41  /* Analog Comparator-0 or Comaprator-1 interrupt          */
#define NVIC_PDMA_INT         42  /* PDMA interrupt                                         */
#define NVIC_I2S_INT          43  /* I2S interrupt                                         */
#define NVIC_PWRWU_INT        44  /* Clock controller interrupt for chip wake up from power-down state*/
#define NVIC_ADC_INT          45  /* ADC interrupt                                          */
#define NVIC_RTC_INT          47  /* Real time clock interrupt                              */

#endif    /* __NUC100B_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = SVC            0x2C
Interrupt3   = PendSV         0x38
Interrupt4   = SysTick        0x3C
Interrupt5   = BOD_OUT        0x40
Interrupt6   = WDT_INT        0x44
Interrupt7   = EINT0          0x48
Interrupt8   = EINT1          0x4C
Interrupt9   = GPAB_INT       0x50
Interrupt10  = GPCDE_INT      0x54
Interrupt11  = PWMA_INT       0x58
Interrupt12  = TMR0_INT       0x60
Interrupt13  = TMR1_INT       0x64
Interrupt14  = TMR2_INT       0x68
Interrupt15  = TMR3_INT       0x6C
Interrupt16  = UART02_INT     0x70
Interrupt17  = UART1_INT      0x74
Interrupt18  = SPI0_INT       0x78
Interrupt19  = SPI1_INT       0x7C
Interrupt20  = I2C0_INT       0x88
Interrupt21  = I2C2_INT       0x8C
Interrupt22  = PS2_INT        0xA0
Interrupt23  = ACMP_INT       0xA4
Interrupt24  = PDMA_INT       0xA8
Interrupt25  = I2S_INT        0xAC
Interrupt26  = PWRWU_INT      0xB0
Interrupt27  = ADC_INT        0xB4
Interrupt28  = RTC_INT        0xBC

###DDF-INTERRUPT-END###*/

