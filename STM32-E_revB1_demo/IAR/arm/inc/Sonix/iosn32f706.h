/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Sonix SN32F706
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2012
 **
 **    $Revision: 55537 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __SN32F706_H
#define __SN32F706_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   Sonix SN32F706 SPECIAL FUNCTION REGISTERS
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


/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
/* IRQ0~31 Interrupt Set-Enable Register (NVIC_ISER) */
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


/* IRQ0~31 Interrupt Clear-Enable Register (NVIC_ICER) */
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

/* IRQ0~31 Interrupt Set-Pending Register (NVIC_ISPR) */
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

/* 2.3.2.4 IRQ0~31 Interrupt Clear-Pending Register (NVIC_ICPR) */
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
} __IPR0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32  PRI_4          : 8;
  __REG32  PRI_5          : 8;
  __REG32  PRI_6          : 8;
  __REG32  PRI_7          : 8;
} __IPR1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32  PRI_8          : 8;
  __REG32  PRI_9          : 8;
  __REG32  PRI_10         : 8;
  __REG32  PRI_11         : 8;
} __IPR2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32  PRI_12         : 8;
  __REG32  PRI_13         : 8;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __IPR3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32  PRI_16         : 8;
  __REG32  PRI_17         : 8;
  __REG32  PRI_18         : 8;
  __REG32  PRI_19         : 8;
} __IPR4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32  PRI_20         : 8;
  __REG32  PRI_21         : 8;
  __REG32  PRI_22         : 8;
  __REG32  PRI_23         : 8;
} __IPR5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32  PRI_24         : 8;
  __REG32  PRI_25         : 8;
  __REG32  PRI_26         : 8;
  __REG32  PRI_27         : 8;
} __IPR6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32  PRI_28         : 8;
  __REG32  PRI_29         : 8;
  __REG32  PRI_30         : 8;
  __REG32  PRI_31         : 8;
} __IPR7_bits;

/***************************************************************************
 **
 ** Application interrupt and reset control(AIRC)
 **
 ***************************************************************************/

/* Application Interrupt and Reset Control */
typedef struct {
  __REG32             : 1;
  __REG32  VECTCLRACTIVE  : 1;
  __REG32  SYSRESETREQ    : 1;
  __REG32                 :12;
  __REG32  ENDIANESS      : 1;
  __REG32  VECTKEY        :16;
} __AIRC_bits;


/***************************************************************************
 **
 ** CODE OPTION TABLE
 **
 ***************************************************************************/
//typedef struct {
//  __REG32  BLEN         : 1;
//  __REG32               :15;
//  __REG32  CodeSecurity :16;
//} __CODE_OPTION_bits;


/***************************************************************************
 **
 ** SysTick register
 **
 ***************************************************************************/

/* SysTick Control and Status Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  TICKINT        : 1;
  __REG32  CLKSOURCE      : 1;
  __REG32                 :13;
  __REG32  COUNTFLAG      : 1;
  __REG32                 :15;
} __SYST_CSR_bits;

/* SysTick Reload Value Register */
typedef struct {
  __REG32  RELOAD         :24;
  __REG32                 : 8;
} __SYST_RVR_bits;

/* SysTick Current Value Register */
typedef struct {
  __REG32  CURRENT        :24;
  __REG32                 : 8;
} __SYST_CVR_bits;


/* System timer claibration value Register */
typedef struct {
  __REG32  TENMS         :24;
  __REG32                : 6;
  __REG32  SKEW          : 1;
  __REG32  NOREF         : 1;
} __SYST_CALIB_bits;



/***************************************************************************
 **
 ** PMU REGISTERS
 **
 ***************************************************************************/

/* Backup Register 0*/
typedef struct {
  __REG32  BACKUPDATA0    : 8;
  __REG32                 :24;
} __PMU_BKP0_bits;

/* Backup Register 1*/
typedef struct {
  __REG32  BACKUPDATA1    : 8;
  __REG32                 :24;
} __PMU_BKP1_bits;

/* Backup Register 2*/
typedef struct {
  __REG32  BACKUPDATA2    : 8;
  __REG32                 :24;
} __PMU_BKP2_bits;

/* Backup Register 3*/
typedef struct {
  __REG32  BACKUPDATA3    : 8;
  __REG32                 :24;
} __PMU_BKP3_bits;

/* Backup Register 4*/
typedef struct {
  __REG32  BACKUPDATA4    : 8;
  __REG32                 :24;
} __PMU_BKP4_bits;

/* Backup Register 5*/
typedef struct {
  __REG32  BACKUPDATA5    : 8;
  __REG32                 :24;
} __PMU_BKP5_bits;

/* Backup Register 6*/
typedef struct {
  __REG32  BACKUPDATA6    : 8;
  __REG32                 :24;
} __PMU_BKP6_bits;

/* Backup Register 7*/
typedef struct {
  __REG32  BACKUPDATA7    : 8;
  __REG32                 :24;
} __PMU_BKP7_bits;

/* Backup Register 8*/
typedef struct {
  __REG32  BACKUPDATA8    : 8;
  __REG32                 :24;
} __PMU_BKP8_bits;

/* Backup Register 9*/
typedef struct {
  __REG32  BACKUPDATA9    : 8;
  __REG32                 :24;
} __PMU_BKP9_bits;

/* Backup Register 10*/
typedef struct {
  __REG32  BACKUPDATA10   : 8;
  __REG32                 :24;
} __PMU_BKP10_bits;

/* Backup Register 11*/
typedef struct {
  __REG32  BACKUPDATA11   : 8;
  __REG32                 :24;
} __PMU_BKP11_bits;

/* Backup Register 12*/
typedef struct {
  __REG32  BACKUPDATA12   : 8;
  __REG32                 :24;
} __PMU_BKP12_bits;

/* Backup Register 13*/
typedef struct {
  __REG32  BACKUPDATA13   : 8;
  __REG32                 :24;
} __PMU_BKP13_bits;

/* Backup Register 14*/
typedef struct {
  __REG32  BACKUPDATA14   : 8;
  __REG32                 :24;
} __PMU_BKP14_bits;

/* Backup Register 15*/
typedef struct {
  __REG32  BACKUPDATA15   : 8;
  __REG32                 :24;
} __PMU_BKP15_bits;


/* Power control Register 15*/
typedef struct {
  __REG32  DPDEN      : 1;
  __REG32  DSLLEPEN   : 1;
  __REG32  SLEEPEN    : 1;
  __REG32             :29;
} __PMU_CTRL_bits;


/***************************************************************************
 **
 ** SYSTEM CONTROL REGISTERS 0
 **
 ***************************************************************************/

/* Analog Block Control register*/
typedef struct {
  __REG32  IHRCEN   : 1;
  __REG32           : 1;
  __REG32  ELSEN    : 1;
  __REG32           : 1;
  __REG32  EHSEN    : 1;
  __REG32  EHSFREQ  : 1;
  __REG32           :26;
} __SYS0_ANBCTRL_bits;


/* PLL control register*/
typedef struct
{
  __REG32 MSEL      :5;
  __REG32 PSEL      :3;
  __REG32 FSEL      :1;
  __REG32           :3;
  __REG32 PLLCLKSEL :2;
  __REG32           :1;
  __REG32 PLLEN     :1;
  __REG32           :16;
} __SYS0_PLLCTRL_bits;

/* Clock Source Status register*/
typedef struct
{
  __REG32 IHRCRDY   : 1;
  __REG32           : 1;
  __REG32 ELSRDY    : 1;
  __REG32           : 1;
  __REG32 EHSRDY    : 1;
  __REG32           : 1;
  __REG32 PLLDRY    : 1;
  __REG32           :25;
} __SYS0_CSST_bits;

/* System Clock Configuration register*/
typedef struct
{
  __REG32 SYSCLKSEL : 3;
  __REG32           : 1;
  __REG32 SYSCLKST  : 3;
  __REG32           :25;
} __SYS0_CLKCFG_bits;

/* AHB Clock Prescale register*/
typedef struct
{
  __REG32 AHBPRE    : 4;
  __REG32           :28;
} __SYS0_AHBCP_bits;


/* System Reset Status register*/
typedef struct
{
  __REG32 SWRSTF    : 1;
  __REG32 WDTRSTF   : 1;
  __REG32 LVDRSTF   : 1;
  __REG32 EXTRSTF   : 1;
  __REG32 PORRSTF   : 1;
  __REG32           :27;
} __SYS0_RSTST_bits;


/* LVD Control register*/
typedef struct
{
  __REG32 LVDRSTLVL : 2;
  __REG32           : 2;
  __REG32 LVDINTLVL : 2;
  __REG32           : 8;
  __REG32 LVDRSTEN  : 1;
  __REG32 LVDEN     : 1;
  __REG32           :16;
} __SYS0_LVDCTRL_bits;


/* External RESET Pin Control register */
typedef struct
{
  __REG32 RESETDIS  : 1;    // 0
  __REG32           :31;    // 1
} __SYS0_EXRSTCTRL_bits;

/* SWD Pin Control register */
typedef struct
{
  __REG32 SWDDIS    : 1;    // 0
  __REG32           :31;    // 1
} __SYS0_SWDCTRL_bits;


/* Interrupt Vector Table Mapping register */
typedef struct
{
  __REG32 IVTM    : 2;    // 0
  __REG32         :30;    // 1
} __SYS0_IVTM_bits;

#if 0
/* IHold Clock Control register */
typedef struct
{
  __REG32 Time    : 3;    // 0
  __REG32         :29;    // 1
} __SYS0_HOLDCKCTRL_bits;

#endif


/***************************************************************************
 **
 ** SYSTEM CONTROL REGISTERS 1
 **
 ***************************************************************************/

/* AHB Clock Enable register */
typedef struct
{
  __REG32             : 3;
  __REG32 GPIOCLKEN   : 1;
  __REG32             : 2;
  __REG32 CT16B0CLKEN : 1;
  __REG32 CT16B1CLKEN : 1;
  __REG32 CT32B0CLKEN : 1;
  __REG32 CT32B1CLKEN : 1;
  __REG32             : 1;
  __REG32 ADCCLKEN    : 1;
  __REG32 SSP0CLKEN   : 1;
  __REG32 SSP1CLKEN   : 1;
  __REG32             : 2;
  __REG32 USART0CLKEN : 1;
  __REG32 USART1CLKEN : 1;
  __REG32             : 2;
  __REG32 I2C1CLKEN   : 1;
  __REG32 I2C0CLKEN   : 1;
  __REG32 I2SCLKEN    : 1;
  __REG32 RTCCLKEN    : 1;
  __REG32 WDTCLKEN    : 1;
  __REG32             : 3;
  __REG32 CLKOUTSEL   : 3;
  __REG32             : 1;
} __SYS1_AHBCLKEN_bits;

/* APB Clock Prescale register 0 */
typedef struct
{
  __REG32 CT16B0PRE   : 3;
  __REG32             : 1;
  __REG32 CT16B1PRE   : 3;
  __REG32             : 1;
  __REG32 CT32B0PRE   : 3;
  __REG32             : 1;
  __REG32 CT32B1PRE   : 3;
  __REG32             : 1;
  __REG32 ADCPRE      : 3;
  __REG32             : 1;
  __REG32 SSP0PRE     : 3;
  __REG32             : 1;
  __REG32 SSP1PRE     : 3;
  __REG32             : 5;
} __SYS1_APBCP0_bits;

/* APB Clock Prescale register 1 */
typedef struct
{
  __REG32 USART0PRE   : 3;
  __REG32             : 1;
  __REG32 USART1PRE   : 3;
  __REG32             : 1;
  __REG32 I2C0PRE     : 3;
  __REG32             : 1;
  __REG32 I2SPRE      : 3;
  __REG32             : 1;
  __REG32 SYSTICKPRE  : 2;
  __REG32             : 2;
  __REG32 WDTPRE      : 3;
  __REG32             : 1;
  __REG32 I2C1PRE     : 3;
  __REG32             : 1;
  __REG32 CLKOUTPRE   : 4;
} __SYS1_APBCP1_bits;

/* Peripheral Reset register */
typedef struct
{
  __REG32 GPIOP0RST   : 1;
  __REG32 GPIOP1RST   : 1;
  __REG32 GPIOP2RST   : 1;
  __REG32 GPIOP3RST   : 1;
  __REG32             : 2;
  __REG32 CT16B0RST   : 1;
  __REG32 CT16B1RST   : 1;
  __REG32 CT32B0RST   : 1;
  __REG32 CT32B1RST   : 1;
  __REG32             : 1;
  __REG32 ADCRST      : 1;
  __REG32 SSP0RST     : 1;
  __REG32 SSP1RST     : 1;
  __REG32             : 2;
  __REG32 USART0RST   : 1;
  __REG32 USART1RST   : 1;
  __REG32             : 2;
  __REG32 I2C1RST     : 1;
  __REG32 I2C0RST     : 1;
  __REG32 I2SRST      : 1;
  __REG32 RTCRST      : 1;
  __REG32 WDTRST      : 1;
  __REG32             : 7;
} __SYS1_PRST_bits;


//#if 0
/***************************************************************************
 **
 ** GENERAL PURPOSE I/O PORT (GPIO)
 **
 ***************************************************************************/
/* GPIO Port n Data register */
typedef struct
{
  __REG32 DATA0     : 1;
  __REG32 DATA1     : 1;
  __REG32 DATA2     : 1;
  __REG32 DATA3     : 1;
  __REG32 DATA4     : 1;
  __REG32 DATA5     : 1;
  __REG32 DATA6     : 1;
  __REG32 DATA7     : 1;
  __REG32 DATA8     : 1;
  __REG32 DATA9     : 1;
  __REG32 DATA10    : 1;
  __REG32 DATA11    : 1;
  __REG32           :20;
} __GPIOx_DATA_bits;

/* GPIO Port n Mode register */
typedef struct
{
  __REG32 MODE0     : 1;
  __REG32 MODE1     : 1;
  __REG32 MODE2     : 1;
  __REG32 MODE3     : 1;
  __REG32 MODE4     : 1;
  __REG32 MODE5     : 1;
  __REG32 MODE6     : 1;
  __REG32 MODE7     : 1;
  __REG32 MODE8     : 1;
  __REG32 MODE9     : 1;
  __REG32 MODE10    : 1;
  __REG32 MODE11    : 1;
  __REG32           :20;
} __GPIOx_MODE_bits;

/* GPIO Port n Configuration register */
typedef struct
{
  __REG32 CFG0      : 2;
  __REG32 CFG1      : 2;
  __REG32 CFG2      : 2;
  __REG32 CFG3      : 2;
  __REG32 CFG4      : 2;
  __REG32 CFG5      : 2;
  __REG32 CFG6      : 2;
  __REG32 CFG7      : 2;
  __REG32 CFG8      : 2;
  __REG32 CFG9      : 2;
  __REG32 CFG10     : 2;
  __REG32 CFG11     : 2;
  __REG32           : 8;
} __GPIOx_CFG_bits;

/* GPIO Port n Interrupt Sense register */
typedef struct
{
  __REG32 IS0      : 1;
  __REG32 IS1      : 1;
  __REG32 IS2      : 1;
  __REG32 IS3      : 1;
  __REG32 IS4      : 1;
  __REG32 IS5      : 1;
  __REG32 IS6      : 1;
  __REG32 IS7      : 1;
  __REG32 IS8      : 1;
  __REG32 IS9      : 1;
  __REG32 IS10     : 1;
  __REG32 IS11     : 1;
  __REG32          :20;
} __GPIOx_ISENSE_bits;

/* GPIO Port n Interrupt Both-edge Sense register */
typedef struct
{
  __REG32 IBS0     : 1;
  __REG32 IBS1     : 1;
  __REG32 IBS2     : 1;
  __REG32 IBS3     : 1;
  __REG32 IBS4     : 1;
  __REG32 IBS5     : 1;
  __REG32 IBS6     : 1;
  __REG32 IBS7     : 1;
  __REG32 IBS8     : 1;
  __REG32 IBS9     : 1;
  __REG32 IBS10    : 1;
  __REG32 IBS11    : 1;
  __REG32          :20;
} __GPIOx_IBS_bits;

/* GPIO Port n Interrupt Event register */
typedef struct
{
  __REG32 IEV0     : 1;
  __REG32 IEV1     : 1;
  __REG32 IEV2     : 1;
  __REG32 IEV3     : 1;
  __REG32 IEV4     : 1;
  __REG32 IEV5     : 1;
  __REG32 IEV6     : 1;
  __REG32 IEV7     : 1;
  __REG32 IEV8     : 1;
  __REG32 IEV9     : 1;
  __REG32 IEV10    : 1;
  __REG32 IEV11    : 1;
  __REG32          :20;
} __GPIOx_IEV_bits;

/* GPIO Port n Interrupt Enable register */
typedef struct
{
  __REG32 IE0      : 1;
  __REG32 IE1      : 1;
  __REG32 IE2      : 1;
  __REG32 IE3      : 1;
  __REG32 IE4      : 1;
  __REG32 IE5      : 1;
  __REG32 IE6      : 1;
  __REG32 IE7      : 1;
  __REG32 IE8      : 1;
  __REG32 IE9      : 1;
  __REG32 IE10     : 1;
  __REG32 IE11     : 1;
  __REG32          :20;
} __GPIOx_IE_bits;

/* GPIO Port n Raw Interrupt Status register */
typedef struct
{
  __REG32 IF0      : 1;
  __REG32 IF1      : 1;
  __REG32 IF2      : 1;
  __REG32 IF3      : 1;
  __REG32 IF4      : 1;
  __REG32 IF5      : 1;
  __REG32 IF6      : 1;
  __REG32 IF7      : 1;
  __REG32 IF8      : 1;
  __REG32 IF9      : 1;
  __REG32 IF10     : 1;
  __REG32 IF11     : 1;
  __REG32          :20;
} __GPIOx_RIS_bits;


/* GPIO Port n Interrupt Clear register */
typedef struct
{
  __REG32 IC0      : 1;
  __REG32 IC1      : 1;
  __REG32 IC2      : 1;
  __REG32 IC3      : 1;
  __REG32 IC4      : 1;
  __REG32 IC5      : 1;
  __REG32 IC6      : 1;
  __REG32 IC7      : 1;
  __REG32 IC8      : 1;
  __REG32 IC9      : 1;
  __REG32 IC10     : 1;
  __REG32 IC11     : 1;
  __REG32          :20;
} __GPIOx_IC_bits;

/* GPIO Port n Bits Set Operation register */
typedef struct
{
  __REG32 SET_B0   : 1;
  __REG32 SET_B1   : 1;
  __REG32 SET_B2   : 1;
  __REG32 SET_B3   : 1;
  __REG32 SET_B4   : 1;
  __REG32 SET_B5   : 1;
  __REG32 SET_B6   : 1;
  __REG32 SET_B7   : 1;
  __REG32 SET_B8   : 1;
  __REG32 SET_B9   : 1;
  __REG32 SET_B10  : 1;
  __REG32 SET_B11  : 1;
  __REG32          :20;
} __GPIOx_BSET_bits;

/* GPIO Port n Bits Clear Operation register */
typedef struct
{
  __REG32 CLR_B0   : 1;
  __REG32 CLR_B1   : 1;
  __REG32 CLR_B2   : 1;
  __REG32 CLR_B3   : 1;
  __REG32 CLR_B4   : 1;
  __REG32 CLR_B5   : 1;
  __REG32 CLR_B6   : 1;
  __REG32 CLR_B7   : 1;
  __REG32 CLR_B8   : 1;
  __REG32 CLR_B9   : 1;
  __REG32 CLR_B10  : 1;
  __REG32 CLR_B11  : 1;
  __REG32          :20;
} __GPIOx_BCLR_bits;

/* GPIO Port n Open-Drain Control register */
typedef struct
{
  __REG32 Pn0OC     : 1;
  __REG32 Pn1OC     : 1;
  __REG32 Pn2OC     : 1;
  __REG32 Pn3OC     : 1;
  __REG32 Pn4OC     : 1;
  __REG32 Pn5OC     : 1;
  __REG32 Pn6OC     : 1;
  __REG32 Pn7OC     : 1;
  __REG32           :24;
} __GPIOx_ODCTRL_bits;


/***************************************************************************
 **
 ** 10-CHANNEL 12-BIT SAR ADC
 **
 ***************************************************************************/
/* ADC Management register */
typedef struct
{
  __REG32 CHS       : 4;
  __REG32 GCHS      : 1;
  __REG32 EOC       : 1;
  __REG32 ADS       : 1;
  __REG32 ADLEN     : 1;
  __REG32 ADCKS     : 3;
  __REG32 ADENB     : 1;
  __REG32 AVREFHSEL : 1;
  __REG32           :19;
} __ADC_ADM_bits;

/* ADC Data register */
typedef struct
{
  __REG32 ADB       :12;
  __REG32           :20;
} __ADC_ADB_bits;


/* Port 2 Control register */
typedef struct
{
  __REG32 P2CON     :10;
  __REG32           :22;
} __ADC_P2CON_bits;

/* ADC Interrupt Enable register */
typedef struct
{
  __REG32 ADC_IE     :10;
  __REG32            :22;
} __ADC_IE_bits;


/*ADC Raw Interrupt Status register */
typedef struct
{
  __REG32 ADC_IF    :10;
  __REG32           :22;
} __ADC_RIS_bits;


#if 0
/*ADC Calibration register */
typedef struct
{
  __REG32 ADT       : 4;
  __REG32 OCDC      : 1;
  __REG32           : 1;
  __REG32 OCCC      : 2;
  __REG32           :23;
} __ADC_CALI_bits;
#endif
// #endif
 /***************************************************************************
 **
 ** 16-BIT TIMER
 **
 ***************************************************************************/
/*CT16Bn Timer Control register */
typedef struct
{
  __REG32 CEN_16      : 1;
  __REG32 CRST_16     : 1;
  __REG32             :30;
} __CT16Bn_TMRCTRL_bits;

 /* CT16Bn Timer Counter register */
typedef struct
{
  __REG32 TC_16       :16;
  __REG32             :16;
} __CT16Bn_TC_bits;

  /* CT16Bn Prescale register */
typedef struct
{
  __REG32 PR_16       :16;
  __REG32             :16;
} __CT16Bn_PRE_bits;

  /* CT16Bn Prescale Counter register */
typedef struct
{
  __REG32 PC_16       :16;
  __REG32             :16;
} __CT16Bn_PC_bits;

  /* CT16Bn Count Control register */
typedef struct
{
  __REG32 CTM_16      : 2;
  __REG32 CIS_16      : 2;
  __REG32             :28;
} __CT16Bn_CNTCTRL_bits;

/* CT16Bn Match Control register */
typedef struct
{
  __REG32 MR0IE_16    : 1;
  __REG32 MR0RST_16   : 1;
  __REG32 MR0STOP_16  : 1;
  __REG32 MR1IE_16    : 1;
  __REG32 MR1RST_16   : 1;
  __REG32 MR1STOP_16  : 1;
  __REG32 MR2IE_16    : 1;
  __REG32 MR2RST_16   : 1;
  __REG32 MR2STOP_16  : 1;
  __REG32 MR3IE_16    : 1;
  __REG32 MR3RST_16   : 1;
  __REG32 MR3STOP_16  : 1;
  __REG32             :20;
} __CT16Bn_MCTRL_bits;

 /* CT16Bn Match register 0 */
typedef struct
{
  __REG32 MR0_16      :16;
  __REG32             :16;
} __CT16Bn_MR0_bits;

/* CT16Bn Match register 1 */
typedef struct
{
  __REG32 MR1_16      :16;
  __REG32             :16;
} __CT16Bn_MR1_bits;

/* CT16Bn Match register 2 */
typedef struct
{
  __REG32 MR2_16      :16;
  __REG32             :16;
} __CT16Bn_MR2_bits;

/* CT16Bn Match register 3 */
typedef struct
{
  __REG32 MR3_16      :16;
  __REG32             :16;
} __CT16Bn_MR3_bits;

/* CT16Bn Capture Control register */
typedef struct
{
  __REG32 CAP0RE_16   : 1;
  __REG32 CAP0FE_16   : 1;
  __REG32 CAP0IE_16   : 1;
  __REG32 CAP0EN_16   : 1;
  __REG32             :28;
} __CT16Bn_CAPCTRL_bits;

/* CT16Bn Capture 0 register */
typedef struct
{
  __REG32 CAP0_16     :16;
  __REG32             :16;
} __CT16Bn_CAP0_bits;

/* CT16Bn External Match register */
typedef struct
{
  __REG32 EM0_16      : 1;
  __REG32 EM1_16      : 1;
  __REG32 EM2_16      : 1;
  __REG32             : 1;
  __REG32 EMC0_16     : 2;
  __REG32 EMC1_16     : 2;
  __REG32 EMC2_16     : 2;
  __REG32             :22;
} __CT16Bn_EM_bits;

/* CT16Bn PWM Control register */
typedef struct
{
  __REG32 PWM0EN_16   : 1;
  __REG32 PWM1EN_16   : 1;
  __REG32 PWM2EN_16   : 1;
  __REG32             :17;
  __REG32 PWM0IOEN_16 : 1;
  __REG32 PWM1IOEN_16 : 1;
  __REG32 PWM2IOEN_16 : 1;
  __REG32             : 9;
} __CT16Bn_PWMCTRL_bits;


 /* CT16Bn Timer Raw Interrupt Status register */
typedef struct
{
  __REG32 MR0IF_16    : 1;
  __REG32 MR1IF_16    : 1;
  __REG32 MR2IF_16    : 1;
  __REG32 MR3IF_16    : 1;
  __REG32 CAP0IF_16   : 1;
  __REG32             :27;
} __CT16Bn_RIS_bits;

 /* CT16Bn Timer Interrupt Clear register */
typedef struct
{
  __REG32 MR0IC_16    : 1;
  __REG32 MR1IC_16    : 1;
  __REG32 MR2IC_16    : 1;
  __REG32 MR3IC_16    : 1;
  __REG32 CAP0IC_16   : 1;
  __REG32             :27;
} __CT16Bn_IC_bits;


 /***************************************************************************
 **
 ** 32-BIT TIMER
 **
 ***************************************************************************/
/*CT32Bn Timer Control register */
typedef struct
{
  __REG32 CEN_32      : 1;
  __REG32 CRST_32     : 1;
  __REG32             :30;
} __CT32Bn_TMRCTRL_bits;

 /* CT32Bn Timer Counter register */
typedef struct
{
  __REG32 TC_32       :32;
} __CT32Bn_TC_bits;

  /* CT32Bn Prescale register */
typedef struct
{
  __REG32 PRE_32      :32;
} __CT32Bn_PRE_bits;

  /* CT32Bn Prescale Counter register */
typedef struct
{
  __REG32 PC_32       :32;
} __CT32Bn_PC_bits;

  /* CT32Bn Count Control register */
typedef struct
{
  __REG32 CTM_32      : 2;
  __REG32 CIS_32      : 2;
  __REG32             :28;
} __CT32Bn_CNTCTRL_bits;

/* CT32Bn Match Control register */
typedef struct
{
  __REG32 MR0IE_32    : 1;
  __REG32 MR0RST_32   : 1;
  __REG32 MR0STOP_32  : 1;
  __REG32 MR1IE_32    : 1;
  __REG32 MR1RST_32   : 1;
  __REG32 MR1STOP_32  : 1;
  __REG32 MR2IE_32    : 1;
  __REG32 MR2RST_32   : 1;
  __REG32 MR2STOP_32  : 1;
  __REG32 MR3IE_32    : 1;
  __REG32 MR3RST_32   : 1;
  __REG32 MR3STOP_32  : 1;
  __REG32             :20;
} __CT32Bn_MCTRL_bits;

 /* CT32Bn Match register 0 */
typedef struct
{
  __REG32 MR0_32      :32;
} __CT32Bn_MR0_bits;

/* CT32Bn Match register 1 */
typedef struct
{
  __REG32 MR1_32      :32;
} __CT32Bn_MR1_bits;

/* CT32Bn Match register 2 */
typedef struct
{
  __REG32 MR2_32      :32;
} __CT32Bn_MR2_bits;

/* CT32Bn Match register 3 */
typedef struct
{
  __REG32 MR3_32      :32;
} __CT32Bn_MR3_bits;

/* CT32Bn Capture Control register */
typedef struct
{
  __REG32 CAP0RE_32   : 1;
  __REG32 CAP0FE_32   : 1;
  __REG32 CAP0IE_32   : 1;
  __REG32 CAP0EN_32   : 1;
  __REG32             :28;
} __CT32Bn_CAPCTRL_bits;

/* CT32Bn Capture 0 register */
typedef struct
{
  __REG32 CAP0_32     :32;
} __CT32Bn_CAP0_bits;

/* CT32Bn External Match register */
typedef struct
{
  __REG32 EM0_32      : 1;
  __REG32 EM1_32      : 1;
  __REG32 EM2_32      : 1;
  __REG32 EM3_32      : 1;
  __REG32 EMC0_32     : 2;
  __REG32 EMC1_32     : 2;
  __REG32 EMC2_32     : 2;
  __REG32 EMC3_32     : 2;
  __REG32             :20;
} __CT32Bn_EM_bits;

/* CT32Bn PWM Control register */
typedef struct
{
  __REG32 PWM0EN_32   : 1;
  __REG32 PWM1EN_32   : 1;
  __REG32 PWM2EN_32   : 1;
  __REG32 PWM3EN_32   : 1;
  __REG32             :16;
  __REG32 PWM0IOEN_32 : 1;
  __REG32 PWM1IOEN_32 : 1;
  __REG32 PWM2IOEN_32 : 1;
  __REG32 PWM3IOEN_32 : 1;
  __REG32             : 8;
} __CT32Bn_PWMCTRL_bits;

 /* CT32Bn Timer Raw Interrupt Status register */
typedef struct
{
  __REG32 MR0IF_32    : 1;
  __REG32 MR1IF_32    : 1;
  __REG32 MR2IF_32    : 1;
  __REG32 MR3IF_32    : 1;
  __REG32 CAP0IF_32   : 1;
  __REG32             :27;
} __CT32Bn_RIS_bits;

 /* CT32Bn Timer Interrupt Clear register */
typedef struct
{
  __REG32 MR0IC_32    : 1;
  __REG32 MR1IC_32    : 1;
  __REG32 MR2IC_32    : 1;
  __REG32 MR3IC_32    : 1;
  __REG32 CAP0IC_32   : 1;
  __REG32             :27;
} __CT32Bn_IC_bits;


 /***************************************************************************
 **
 ** WATCHDOG TIMER
 **
 ***************************************************************************/
/* Watchdog Configuration register */
typedef struct
{
  __REG32 WDTEN       : 1;
  __REG32 WDTIE       : 1;
  __REG32 WDINT       : 1;
  __REG32             :29;
} __WDT_CFG_bits;

/* Watchdog Clock Source register */
typedef struct
{
  __REG32 CLKSEL      : 2;
  __REG32             :30;
} __WDT_CLKSOURCE_bits;

/* Watchdog Timer Constant register */
typedef struct
{
  __REG32 WDTTC       : 8;
  __REG32             :24;
} __WDT_TC_bits;

/* Watchdog Feed register */
typedef struct
{
  __REG32 FV          :16;
  __REG32             :16;
} __WDT_FEED_bits;


 /***************************************************************************
 **
 ** REAL-TIME CLOCK (RTC)
 **
 ***************************************************************************/
/* RTC Control register */
typedef struct
{
  __REG32 RTCEN       : 1;
  __REG32             :31;
} __RTC_CTRL_bits;

/* RTC Clock Source Select register */
typedef struct
{
  __REG32 CLKSEL      : 2;
  __REG32             :30;
} __RTC_CLKS_bits;

/* RTC Interrupt Enable register */
typedef struct
{
  __REG32 SECIE       : 1;
  __REG32 ALMIE       : 1;
  __REG32 OVFIE       : 1;
  __REG32             :29;
} __RTC_IE_bits;

/* RTC Raw Interrupt Status register */
typedef struct
{
  __REG32 SECIF       : 1;
  __REG32 ALMIF       : 1;
  __REG32 OVFIF       : 1;
  __REG32             :29;
} __RTC_RIS_bits;

/* RTC Interrupt Clear register*/
typedef struct
{
  __REG32 SECIC       : 1;
  __REG32 ALMIC       : 1;
  __REG32 OVFIC       : 1;
  __REG32             :29;
} __RTC_IC_bits;

/* RTC Second Counter Reload Value register */
typedef struct
{
  __REG32 SECCNTV     :20;
  __REG32             :12;
} __RTC_SECCNTV_bits;

/* RTC Second Count register */
typedef struct
{
  __REG32 RTC_SECCNT  :32;
} __RTC_SECCNT_bits;


/* RTC Alarm Counter Reload Value register */
typedef struct
{
  __REG32 ALMCNTV     :32;
} __RTC_ALMCNTV_bits;

/* RTC Alarm Count register */
typedef struct
{
  __REG32 ALMCNT      :32;
} __RTC_ALMCNT_bits;



/***************************************************************************
 **
 ** RSPI/SSP
 **
 ***************************************************************************/
/* SSP n Control register 0 */
typedef struct
{
  __REG32 SSPEN       : 1;
  __REG32 LOOPBACK    : 1;
  __REG32 SDODIS      : 1;
  __REG32 MS          : 1;
  __REG32 FORMAT      : 1;
  __REG32             : 1;
  __REG32 FRESET      : 2;
  __REG32 DL          : 4;
  __REG32             :20;
} __SSPn_CTRL0_bits;

/* SSP n Control register 1 */
typedef struct {
  __REG32  MLSB       : 1;
  __REG32  CPOL       : 1;
  __REG32  CPHA       : 1;
  __REG32             :29;
} __SSPn_CTRL1_bits;

/* SSP n Clock Divider register */
typedef struct {
  __REG32  DIV        : 8;
  __REG32             :24;
} __SSPn_CLKDIV_bits;

/* SSP n Status register */
typedef struct {
  __REG32  TX_EMPTY    : 1;
  __REG32  TX_FULL     : 1;
  __REG32  RX_EMPTY    : 1;
  __REG32  RX_FULL     : 1;
  __REG32  BUSY        : 1;
  __REG32  TX_HF_EMPTY : 1;
  __REG32  RX_HF_FULL  : 1;
  __REG32              :25;
} __SSPn_STAT_bits;

/* SSP n Interrupt Enable register */
typedef struct {
  __REG32  RXOVFIE    : 1;
  __REG32  RXTOIE     : 1;
  __REG32  RXHFIE     : 1;
  __REG32  TXHEIE     : 1;
  __REG32             :28;
} __SSPn_IE_bits;

/* SSP n Raw Interrupt Status register */
typedef struct {
  __REG32  RXOVFIF    : 1;
  __REG32  RXTOIF     : 1;
  __REG32  RXHFIF     : 1;
  __REG32  TXHEIF     : 1;
  __REG32             :28;
} __SSPn_RIS_bits;

/* SSP n Interrupt Clear register */
typedef struct {
  __REG32  RXOVFIC    : 1;
  __REG32  RXTOIC     : 1;
  __REG32  RXHFIC     : 1;
  __REG32  TXHEIC     : 1;
  __REG32             :28;
} __SSPn_IC_bits;

/* SSP n Data register */
typedef struct {
  __REG32  SSP_DATA   :16;
  __REG32             :16;
} __SSPn_DATA_bits;


/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
/* I2C Control register */
typedef struct
{
  __REG32             : 1;
  __REG32 NACK        : 1;
  __REG32 ACK         : 1;
  __REG32             : 1;
  __REG32 STO         : 1;
  __REG32 STA         : 1;
  __REG32             : 1;
  __REG32 I2CMODE     : 1;
  __REG32 I2CEN       : 1;
  __REG32             :23;
} __I2C_CTRL_bits;

/* I2C Status register */
typedef struct
{
  __REG32 RX_DN       : 1;
  __REG32 ACK_STAT    : 1;
  __REG32 NACK_STAT   : 1;
  __REG32 STOP_DN     : 1;
  __REG32 START_DN    : 1;
  __REG32 I2C_MST     : 1;
  __REG32 SLV_RX_HIT  : 1;
  __REG32 SLV_TX_HIT  : 1;
  __REG32 LOST_ARB    : 1;
  __REG32 TIMEOUT     : 1;
  __REG32             : 5;
  __REG32 I2CIF       : 1;
  __REG32             :16;
} __I2C_STAT_bits;

/* I2C TX Data register */
typedef struct {
  __REG32  I2C_TXDATA : 8;
  __REG32             :24;
} __I2C_TXDATA_bits;

/* I2C RX Data register */
typedef struct {
  __REG32  I2C_RXDATA : 8;
  __REG32             :24;
} __I2C_RXDATA_bits;

/* I2C Slave Address 0 register */
typedef struct
{
  __REG32 SLVADDR0    :10;
  __REG32             :20;
  __REG32 GCEN        : 1;
  __REG32 ADD_MODE    : 1;
} __I2C_SLVADDR0_bits;

/* I2C Slave Address 1 register */
typedef struct
{
  __REG32 SLVADDR1    :10;
  __REG32             :22;
} __I2C_SLVADDR1_bits;

/* I2C Slave Address 2 register */
typedef struct
{
  __REG32 SLVADDR2    :10;
  __REG32             :22;
} __I2C_SLVADDR2_bits;

/* I2C Slave Address 3 register */
typedef struct
{
  __REG32 SLVADDR3    :10;
  __REG32             :22;
} __I2C_SLVADDR3_bits;

/* I2C SCL High Time register */
typedef struct
{
  __REG32 SCLH        : 8;
  __REG32             :24;
} __I2C_SCLHT_bits;

/* I2C SCL Low Time register */
typedef struct
{
  __REG32 SCLL        : 8;
  __REG32             :24;
} __I2C_SCLLT_bits;

/* I2C Timeout Control register */
typedef struct
{
  __REG32 TO          :16;
  __REG32             :16;
} __I2C_TOCTRL_bits;

/* I2C Monitor Mode Control register  */
typedef struct
{
  __REG32 MMEN        : 1;
  __REG32 SCLOEN      : 1;
  __REG32 MATCH_ALL   : 1;
  __REG32             :29;
} __I2C_MMCTRL_bits;

/* I2C Engineer Mode register  */
typedef struct
{
  __REG32 I2COD        : 1;
  __REG32 SLV_START_EN : 1;
  __REG32              :30;
} __I2C_Engineer_bits;


/***************************************************************************
 **
 ** UNIVERSAL SYNCHRONOUS AND ASYNCHRONOUS SERIAL RECEIVER AND TRANSMITTER (USART)
 **
 ***************************************************************************/
/* USART n Receiver Buffer register  */
/* USART n Transmitter Holding register */
/* USART n Divisor Latch LSB registers */
typedef union
{
  /*USART0_RB*/
  /*USART1_RB*/
  struct
  {
    __REG32 RB          : 8;
    __REG32             :24;
  };

  /*USART0_TH*/
  /*USART1_TH*/
  struct
  {
    __REG32 TH          : 8;
    __REG32             :24;
  };

  /*USART0_DLL*/
  /*USART1_DLL*/
  struct
  {
    __REG32 DLL         : 8;
    __REG32             :24;
  };
} __USARTn_RB_bits;

/* USART n Divisor Latch MSB register */
/* USART n Interrupt Enable register */
typedef union
{
  /*USART0_DLM*/
  /*USART1_DLM*/
  struct
  {
    __REG32 DLM         : 8;
    __REG32             :24;
  };

  /*USART0_IE*/
  /*USART1_IE*/
  struct
  {
    __REG32 RDAIE       : 1;
    __REG32 THREIE      : 1;
    __REG32 RLSIE       : 1;
    __REG32 MSIE        : 1;
    __REG32 TEMTIE      : 1;
    __REG32             : 3;
    __REG32 ABEOIE      : 1;
    __REG32 ABTOIE      : 1;
    __REG32 TXERRIE     : 1;
    __REG32             :21;
  };
} __USARTn_DLM_bits;

/* USART n Interrupt Identification register */
/* USART n FIFO Control register */
typedef union
{
  /*USART0_II*/
  /*USART1_II*/
  struct
  {
    __REG32 INTSTATUS   : 1;
    __REG32 INTID       : 3;
    __REG32             : 2;
    __REG32 FIFOEN      : 2;
    __REG32 ABEOIF      : 1;
    __REG32 ABTOIF      : 1;
    __REG32 TXERRIF     : 1;
    __REG32             :21;
  };

  /*USART0_FIFOCTRL*/
  /*USART1_FIFOCTRL*/
  struct
  {
    __REG32 FIFOEN      : 1;
    __REG32 RXFIFORST   : 1;
    __REG32 TXFIFORST   : 1;
    __REG32             : 3;
    __REG32 RXTL        : 2;
    __REG32             :24;
  } __USARTn_FIFOCTRL_bits;
} __USARTn_II_bits;

/* USART n Line Control register */
typedef struct
{
  __REG32 WLS         : 2;
  __REG32 SBS         : 1;
  __REG32 PE          : 1;
  __REG32 PS          : 2;
  __REG32 BC          : 1;
  __REG32 DLAB        : 1;
  __REG32             :24;
} __USARTn_LC_bits;

/* USART n Modem Control register */
typedef struct
{
  __REG32 DTRCTRL     : 1;
  __REG32 RTSCTRL     : 1;
  __REG32 OUT1        : 1;
  __REG32 OUT2        : 1;
  __REG32 LMS         : 1;
  __REG32             : 1;
  __REG32 RTSEN       : 1;
  __REG32 CTSEN       : 1;
  __REG32             :24;
} __USARTn_MC_bits;

/* USART n Line Status register */
typedef struct
{
  __REG32 RDR         : 1;
  __REG32 OE          : 1;
  __REG32 PE          : 1;
  __REG32 FE          : 1;
  __REG32 BI          : 1;
  __REG32 THRE        : 1;
  __REG32 TEMT        : 1;
  __REG32 RXFE        : 1;
  __REG32 TXERR       : 1;
  __REG32             :23;
} __USARTn_LS_bits;

/* USART n Modem Status register */
typedef struct
{
  __REG32 DCTS        : 1;
  __REG32 DDSR        : 1;
  __REG32 TERI        : 1;
  __REG32 DDCD        : 1;
  __REG32 CTS         : 1;
  __REG32 DSR         : 1;
  __REG32 RI          : 1;
  __REG32 DCD         : 1;
  __REG32             :24;
} __USARTn_MS_bits;

/* USART n Scratch Pad register */
typedef struct
{
  __REG32 PAD         : 8;
  __REG32             :24;
} __USARTn_SP_bits;

/* USART n Auto-baud Control register */
typedef struct
{
  __REG32 USART_START : 1;
  __REG32 USART_MODE  : 1;
  __REG32 AUTORESTART : 1;
  __REG32             : 5;
  __REG32 ABEOIFC     : 1;
  __REG32 ABTOIFC     : 1;
  __REG32             :22;
} __USARTn_ABCTRL_bits;

/* USART n IrDA Control register */
typedef struct
{
  __REG32             : 1;
  __REG32 IRDAINV     : 1;
  __REG32 FIXPULSEEN  : 1;
  __REG32 PULSEDIV    : 3;
  __REG32             :26;
} __USARTn_IRDACTRL_bits;

/* USART n Fractional Divider register */
typedef struct
{
  __REG32 DIVADDVAL   : 4;
  __REG32 MULVAL      : 4;
  __REG32 OVER8       : 1;
  __REG32             :23;
} __USARTn_FD_bits;

/* USART n Control register */
typedef struct
{
  __REG32 USARTEN     : 1;
  __REG32 UCMODE      : 3;
  __REG32             : 2;
  __REG32 RXEN        : 1;
  __REG32 TXEN        : 1;
  __REG32             :24;
} __USARTn_CTRL_bits;

/* USART n Half-duplex Enable register */
typedef struct
{
  __REG32 HDEN        : 1;
  __REG32             :31;
} __USARTn_HDEN_bits;

/* USART n Smardcard Interface Control register */
typedef struct
{
  __REG32             : 1;
  __REG32 NACKDIS     : 1;
  __REG32 PROTSEL     : 1;
  __REG32 SCLKEN      : 1;
  __REG32             : 1;
  __REG32 TXRETRY     : 3;
  __REG32 XTRAGUARD   : 8;
  __REG32 TC          : 8;
  __REG32             : 8;
} __USARTn_SCICTRL_bits;

/* USART n RS485 Control register */
typedef struct
{
  __REG32 NMMEN       : 1;
  __REG32 RXEN        : 1;
  __REG32 AADEN       : 1;
  __REG32             : 1;
  __REG32 ADCEN       : 1;
  __REG32 OINV        : 1;
  __REG32             :26;
} __USARTn_RS485CTRL_bits;

/* USART n RS485 Address Match register */
typedef struct
{
  __REG32 MATCH       : 8;
  __REG32             :24;
} __USARTn_RS485ADRMATCH_bits;

/* USART n RS485 Delay Value register */
typedef struct
{
  __REG32 DLY         : 8;
  __REG32             :24;
} __USARTn_RS485DLYV_bits;


/* USART n Synchronous Mode Control Register */
typedef struct
{
  __REG32             : 1;
  __REG32 CPOL        : 1;
  __REG32 CPHA        : 1;
  __REG32             :29;
} __USARTn_SYNCCTRL_bits;


/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
/* I2S Control register */
typedef struct
{
  __REG32 I2S_START   : 1;
  __REG32 MUTE        : 1;
  __REG32 MONO        : 1;
  __REG32 TRS         : 1;
  __REG32 I2S_MS      : 1;
  __REG32 FORMAT      : 2;
  __REG32 CLRFIFO     : 1;
  __REG32             : 2;
  __REG32 DL          : 2;
  __REG32 FIFOTH      : 3;
  __REG32 I2SEN       : 1;
  __REG32 CHLENGTH    : 5;
  __REG32             :11;
} __I2S_CTRL_bits;

/* I2S Clock register */
typedef struct
{
  __REG32 MCLKDIV     : 3;
  __REG32 MCLKOEN     : 1;
  __REG32 MCLKSEL     : 1;
  __REG32             : 3;
  __REG32 BCLKDIV     : 8;
  __REG32             :16;
} __I2S_CLK_bits;

/* I2S Status register */
typedef struct
{
  __REG32 I2SINT      : 1;
  __REG32 RIGHTCH     : 1;
  __REG32             : 4;
  __REG32 FIFOTHF     : 1;
  __REG32             : 3;
  __REG32 FIFOFULL    : 1;
  __REG32 FIFOEMPTY   : 1;
  __REG32 FIFOLV      : 4;
  __REG32             :16;
} __I2S_STATUS_bits;


/* I2S Interrupt Enable register */
typedef struct
{
  __REG32             : 4;
  __REG32 FIFOUDFIEN  : 1;
  __REG32 FIFOOVFIEN  : 1;
  __REG32 FIFOTHIEN   : 1;
  __REG32             :25;
} __I2S_IE_bits;


/* I2S Raw Interrupt Status register */
typedef struct
{
  __REG32             : 4;
  __REG32 FIFOUDIF    : 1;
  __REG32 FIFOOVIF    : 1;
  __REG32 FIFOTHIF    : 1;
  __REG32             :25;
} __I2S_RIS_bits;

/* I2S Interrupt Clear register */
typedef struct
{
  __REG32             : 4;
  __REG32 FIFOUDIC    : 1;
  __REG32 FIFOOVIC    : 1;
  __REG32 FIFOTHIC    : 1;
  __REG32             :25;
} __I2S_IC_bits;

/* I2S FIFO register */
typedef struct
{
  __REG32 I2S_FIFO    :32;
} __I2S_FIFO_bits;



 /***************************************************************************
 **
 ** FLASH
 **
 ***************************************************************************/
/* Flash Status register */
typedef struct
{
  __REG32 BUSY        : 1;
  __REG32             : 1;
  __REG32 PGERR       : 1;
  __REG32             : 2;
  __REG32 EOP         : 1;
  __REG32             :26;
} __FLASH_STATUS_bits;


/* Flash Control register */
typedef struct
{
  __REG32 PG          : 1;
  __REG32 PER         : 1;
  __REG32             : 4;
  __REG32 STARTE      : 1;
  __REG32 CHK         : 1;
  __REG32             :24;
} __FLASH_CTRL_bits;

/* Flash Data register */
typedef struct
{
  __REG32 DATA        :32;
} __FLASH_DATA_bits;


/* Flash Address register */
typedef struct
{
  __REG32 FAR         :32;
} __FLASH_ADDR_bits;

/* Flash Checksum register */
typedef struct
{
  __REG32 CHKSUM      :16;
  __REG32             :16;
} __FLASH_CHKSUM_bits;



#endif    /* __IAR_SYSTEMS_ICC__ */


/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SETENA0,         0xE000E100,__READ_WRITE ,__setena0_bits);
#define ISER        SETENA0
#define ISER_bit    SETENA0_bit
__IO_REG32_BIT(CLRENA0,         0xE000E180,__READ_WRITE ,__clrena0_bits);
#define ICER        CLRENA0
#define ICER_bit    CLRENA0_bit
__IO_REG32_BIT(SETPEND0,        0xE000E200,__READ_WRITE ,__setpend0_bits);
#define ISPR        SETPEND0
#define ISPR_bit    SETPEND0_bit
__IO_REG32_BIT(CLRPEND0,        0xE000E280,__READ_WRITE ,__clrpend0_bits);
#define ICPR        CLRPEND0
#define ICPR_bit    CLRPEND0_bit
__IO_REG32_BIT(IPR0,            0xE000E400,__READ_WRITE ,__IPR0_bits);
__IO_REG32_BIT(IPR1,            0xE000E404,__READ_WRITE ,__IPR1_bits);
__IO_REG32_BIT(IPR2,            0xE000E408,__READ_WRITE ,__IPR2_bits);
__IO_REG32_BIT(IPR3,            0xE000E40C,__READ_WRITE ,__IPR3_bits);
__IO_REG32_BIT(IPR4,            0xE000E410,__READ_WRITE ,__IPR4_bits);
__IO_REG32_BIT(IPR5,            0xE000E414,__READ_WRITE ,__IPR5_bits);
__IO_REG32_BIT(IPR6,            0xE000E418,__READ_WRITE ,__IPR6_bits);
__IO_REG32_BIT(IPR7,            0xE000E41C,__READ_WRITE ,__IPR7_bits);


/***************************************************************************
 **
 ** SysTick
 **
 ***************************************************************************/
__IO_REG32_BIT(SYST_CSR,        0xE000E010,__READ_WRITE ,__SYST_CSR_bits);
__IO_REG32_BIT(SYST_RVR,        0xE000E014,__READ_WRITE ,__SYST_RVR_bits);
__IO_REG32_BIT(SYST_CVR,        0xE000E018,__READ_WRITE ,__SYST_CVR_bits);
__IO_REG32_BIT(SYST_CALIB,      0xE000E01C,__READ_WRITE ,__SYST_CALIB_bits);


/***************************************************************************
 **
 ** AIRC
 **
 ***************************************************************************/
__IO_REG32_BIT(AIRC,           0xE000ED0C,__READ_WRITE ,__AIRC_bits);


/***************************************************************************
 **
 ** CODE OPTION
 **
 ***************************************************************************/
//__IO_REG32_BIT(CODE_OPTION,    0x1FFF2000,__READ_WRITE ,__CODE_OPTION_bits);


/***************************************************************************
 **
 ** PMU
 **
 ***************************************************************************/
__IO_REG32_BIT(PMU_BKP0,        0x40032000,__READ_WRITE ,__PMU_BKP0_bits);
__IO_REG32_BIT(PMU_BKP1,        0x40032004,__READ_WRITE ,__PMU_BKP1_bits);
__IO_REG32_BIT(PMU_BKP2,        0x40032008,__READ_WRITE ,__PMU_BKP2_bits);
__IO_REG32_BIT(PMU_BKP3,        0x4003200C,__READ_WRITE ,__PMU_BKP3_bits);
__IO_REG32_BIT(PMU_BKP4,        0x40032010,__READ_WRITE ,__PMU_BKP4_bits);
__IO_REG32_BIT(PMU_BKP5,        0x40032014,__READ_WRITE ,__PMU_BKP5_bits);
__IO_REG32_BIT(PMU_BKP6,        0x40032018,__READ_WRITE ,__PMU_BKP6_bits);
__IO_REG32_BIT(PMU_BKP7,        0x4003201C,__READ_WRITE ,__PMU_BKP7_bits);
__IO_REG32_BIT(PMU_BKP8,        0x40032020,__READ_WRITE ,__PMU_BKP8_bits);
__IO_REG32_BIT(PMU_BKP9,        0x40032024,__READ_WRITE ,__PMU_BKP9_bits);
__IO_REG32_BIT(PMU_BKP10,       0x40032028,__READ_WRITE ,__PMU_BKP10_bits);
__IO_REG32_BIT(PMU_BKP11,       0x4003202C,__READ_WRITE ,__PMU_BKP11_bits);
__IO_REG32_BIT(PMU_BKP12,       0x40032030,__READ_WRITE ,__PMU_BKP12_bits);
__IO_REG32_BIT(PMU_BKP13,       0x40032034,__READ_WRITE ,__PMU_BKP13_bits);
__IO_REG32_BIT(PMU_BKP14,       0x40032038,__READ_WRITE ,__PMU_BKP14_bits);
__IO_REG32_BIT(PMU_BKP15,       0x4003203C,__READ_WRITE ,__PMU_BKP15_bits);
__IO_REG32_BIT(PMU_CTRL,        0x40032040,__READ_WRITE ,__PMU_CTRL_bits);


/***************************************************************************
 **
 ** SCR
 **
 ***************************************************************************/
__IO_REG32_BIT(SYS0_ANBCTRL,    0x40060000,__READ_WRITE ,__SYS0_ANBCTRL_bits);
__IO_REG32_BIT(SYS0_PLLCTRL,    0x40060004,__READ_WRITE ,__SYS0_PLLCTRL_bits);
__IO_REG32_BIT(SYS0_CSST,       0x40060008,__READ_WRITE ,__SYS0_CSST_bits);
__IO_REG32_BIT(SYS0_CLKCFG,     0x4006000C,__READ_WRITE ,__SYS0_CLKCFG_bits);
__IO_REG32_BIT(SYS0_AHBCP,      0x40060010,__READ_WRITE ,__SYS0_AHBCP_bits);
__IO_REG32_BIT(SYS0_RSTST,      0x40060014,__READ_WRITE ,__SYS0_RSTST_bits);
__IO_REG32_BIT(SYS0_LVDCTRL,    0x40060018,__READ_WRITE ,__SYS0_LVDCTRL_bits);
__IO_REG32_BIT(SYS0_EXRSTCTRL,  0x4006001C,__READ_WRITE ,__SYS0_EXRSTCTRL_bits);
__IO_REG32_BIT(SYS0_SWDCTRL,    0x40060020,__READ_WRITE ,__SYS0_SWDCTRL_bits);
__IO_REG32_BIT(SYS0_IVTM,       0x40060024,__READ_WRITE ,__SYS0_IVTM_bits);

__IO_REG32_BIT(SYS1_AHBCLKEN,   0x4005E000,__READ_WRITE ,__SYS1_AHBCLKEN_bits);
__IO_REG32_BIT(SYS1_APBCP0,     0x4005E004,__READ_WRITE ,__SYS1_APBCP0_bits);
__IO_REG32_BIT(SYS1_APBCP1,     0x4005E008,__READ_WRITE ,__SYS1_APBCP1_bits);
__IO_REG32_BIT(SYS1_PRST,       0x4005E00C,__READ_WRITE ,__SYS1_PRST_bits);


/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO0_DATA,       0x40044000,__READ_WRITE ,__GPIOx_DATA_bits);
__IO_REG32_BIT(GPIO0_MODE,       0x40044004,__READ_WRITE ,__GPIOx_MODE_bits);
__IO_REG32_BIT(GPIO0_CFG,        0x40044008,__READ_WRITE ,__GPIOx_CFG_bits);
__IO_REG32_BIT(GPIO0_ISENSE,     0x4004400C,__READ_WRITE ,__GPIOx_ISENSE_bits);
__IO_REG32_BIT(GPIO0_IBS,        0x40044010,__READ_WRITE ,__GPIOx_IBS_bits);
__IO_REG32_BIT(GPIO0_IEV,        0x40044014,__READ_WRITE ,__GPIOx_IEV_bits);
__IO_REG32_BIT(GPIO0_IE,         0x40044018,__READ_WRITE ,__GPIOx_IE_bits);
__IO_REG32_BIT(GPIO0_RIS,        0x4004401C,__READ       ,__GPIOx_RIS_bits);
__IO_REG32_BIT(GPIO0_IC,         0x40044020,__READ_WRITE ,__GPIOx_IC_bits);
__IO_REG32_BIT(GPIO0_BSET,       0x40044024,__READ_WRITE ,__GPIOx_BSET_bits);
__IO_REG32_BIT(GPIO0_BCLR,       0x40044028,__READ_WRITE ,__GPIOx_BCLR_bits);
__IO_REG32_BIT(GPIO0_ODCTRL,     0x4004402C,__READ_WRITE ,__GPIOx_ODCTRL_bits);

__IO_REG32_BIT(GPIO1_DATA,       0x40046000,__READ_WRITE ,__GPIOx_DATA_bits);
__IO_REG32_BIT(GPIO1_MODE,       0x40046004,__READ_WRITE ,__GPIOx_MODE_bits);
__IO_REG32_BIT(GPIO1_CFG,        0x40046008,__READ_WRITE ,__GPIOx_CFG_bits);
__IO_REG32_BIT(GPIO1_ISENSE,     0x4004600C,__READ_WRITE ,__GPIOx_ISENSE_bits);
__IO_REG32_BIT(GPIO1_IBS,        0x40046010,__READ_WRITE ,__GPIOx_IBS_bits);
__IO_REG32_BIT(GPIO1_IEV,        0x40046014,__READ_WRITE ,__GPIOx_IEV_bits);
__IO_REG32_BIT(GPIO1_IE,         0x40046018,__READ_WRITE ,__GPIOx_IE_bits);
__IO_REG32_BIT(GPIO1_RIS,        0x4004601C,__READ       ,__GPIOx_RIS_bits);
__IO_REG32_BIT(GPIO1_IC,         0x40046020,__READ_WRITE ,__GPIOx_IC_bits);
__IO_REG32_BIT(GPIO1_BSET,       0x40046024,__READ_WRITE ,__GPIOx_BSET_bits);
__IO_REG32_BIT(GPIO1_BCLR,       0x40046028,__READ_WRITE ,__GPIOx_BCLR_bits);
__IO_REG32_BIT(GPIO1_ODCTRL,     0x4004602C,__READ_WRITE ,__GPIOx_ODCTRL_bits);

__IO_REG32_BIT(GPIO2_DATA,       0x40048000,__READ_WRITE ,__GPIOx_DATA_bits);
__IO_REG32_BIT(GPIO2_MODE,       0x40048004,__READ_WRITE ,__GPIOx_MODE_bits);
__IO_REG32_BIT(GPIO2_CFG,        0x40048008,__READ_WRITE ,__GPIOx_CFG_bits);
__IO_REG32_BIT(GPIO2_ISENSE,     0x4004800C,__READ_WRITE ,__GPIOx_ISENSE_bits);
__IO_REG32_BIT(GPIO2_IBS,        0x40048010,__READ_WRITE ,__GPIOx_IBS_bits);
__IO_REG32_BIT(GPIO2_IEV,        0x40048014,__READ_WRITE ,__GPIOx_IEV_bits);
__IO_REG32_BIT(GPIO2_IE,         0x40048018,__READ_WRITE ,__GPIOx_IE_bits);
__IO_REG32_BIT(GPIO2_RIS,        0x4004801C,__READ       ,__GPIOx_RIS_bits);
__IO_REG32_BIT(GPIO2_IC,         0x40048020,__READ_WRITE ,__GPIOx_IC_bits);
__IO_REG32_BIT(GPIO2_BSET,       0x40048024,__READ_WRITE ,__GPIOx_BSET_bits);
__IO_REG32_BIT(GPIO2_BCLR,       0x40048028,__READ_WRITE ,__GPIOx_BCLR_bits);
__IO_REG32_BIT(GPIO2_ODCTRL,     0x4004802C,__READ_WRITE ,__GPIOx_ODCTRL_bits);

__IO_REG32_BIT(GPIO3_DATA,       0x4004A000,__READ_WRITE ,__GPIOx_DATA_bits);
__IO_REG32_BIT(GPIO3_MODE,       0x4004A004,__READ_WRITE ,__GPIOx_MODE_bits);
__IO_REG32_BIT(GPIO3_CFG,        0x4004A008,__READ_WRITE ,__GPIOx_CFG_bits);
__IO_REG32_BIT(GPIO3_ISENSE,     0x4004A00C,__READ_WRITE ,__GPIOx_ISENSE_bits);
__IO_REG32_BIT(GPIO3_IBS,        0x4004A010,__READ_WRITE ,__GPIOx_IBS_bits);
__IO_REG32_BIT(GPIO3_IEV,        0x4004A014,__READ_WRITE ,__GPIOx_IEV_bits);
__IO_REG32_BIT(GPIO3_IE,         0x4004A018,__READ_WRITE ,__GPIOx_IE_bits);
__IO_REG32_BIT(GPIO3_RIS,        0x4004A01C,__READ       ,__GPIOx_RIS_bits);
__IO_REG32_BIT(GPIO3_IC,         0x4004A020,__READ_WRITE ,__GPIOx_IC_bits);
__IO_REG32_BIT(GPIO3_BSET,       0x4004A024,__READ_WRITE ,__GPIOx_BSET_bits);
__IO_REG32_BIT(GPIO3_BCLR,       0x4004A028,__READ_WRITE ,__GPIOx_BCLR_bits);
__IO_REG32_BIT(GPIO3_ODCTRL,     0x4004A02C,__READ_WRITE ,__GPIOx_ODCTRL_bits);


/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_ADM,          0x40026000,__READ_WRITE ,__ADC_ADM_bits);
__IO_REG32_BIT(ADC_ADB,          0x40026004,__READ       ,__ADC_ADB_bits);
__IO_REG32_BIT(ADC_P2CON,        0x40026008,__READ_WRITE ,__ADC_P2CON_bits);
__IO_REG32_BIT(ADC_IE,           0x4002600C,__READ_WRITE ,__ADC_IE_bits);
__IO_REG32_BIT(ADC_RIS,          0x40026010,__READ_WRITE ,__ADC_RIS_bits);
//__IO_REG32_BIT(ADC_CALI,         0x40026014,__READ_WRITE ,__ADC_CALI_bits);


/***************************************************************************
 **
 ** CT16B0
 **
 ***************************************************************************/
__IO_REG32_BIT(CT16B0_TMRCTRL,  0x40000000,__READ_WRITE ,__CT16Bn_TMRCTRL_bits);
__IO_REG32_BIT(CT16B0_TC,       0x40000004,__READ_WRITE ,__CT16Bn_TC_bits);
__IO_REG32_BIT(CT16B0_PRE,      0x40000008,__READ_WRITE ,__CT16Bn_PRE_bits);
__IO_REG32_BIT(CT16B0_PC,       0x4000000C,__READ_WRITE ,__CT16Bn_PC_bits);
__IO_REG32_BIT(CT16B0_CNTCTRL,  0x40000010,__READ_WRITE ,__CT16Bn_CNTCTRL_bits);
__IO_REG32_BIT(CT16B0_MCTRL,    0x40000014,__READ_WRITE ,__CT16Bn_MCTRL_bits);
__IO_REG32_BIT(CT16B0_MR0,      0x40000018,__READ_WRITE ,__CT16Bn_MR0_bits);
__IO_REG32_BIT(CT16B0_MR1,      0x4000001C,__READ_WRITE ,__CT16Bn_MR1_bits);
__IO_REG32_BIT(CT16B0_MR2,      0x40000020,__READ_WRITE ,__CT16Bn_MR2_bits);
__IO_REG32_BIT(CT16B0_MR3,      0x40000024,__READ_WRITE ,__CT16Bn_MR3_bits);
__IO_REG32_BIT(CT16B0_CAPCTRL,  0x40000028,__READ_WRITE ,__CT16Bn_CAPCTRL_bits);
__IO_REG32_BIT(CT16B0_CAP0,     0x4000002C,__READ_WRITE ,__CT16Bn_CAP0_bits);
__IO_REG32_BIT(CT16B0_EM,       0x40000030,__READ_WRITE ,__CT16Bn_EM_bits);
__IO_REG32_BIT(CT16B0_PWMCTRL,  0x40000034,__READ_WRITE ,__CT16Bn_PWMCTRL_bits);
__IO_REG32_BIT(CT16B0_RIS,      0x40000038,__READ_WRITE ,__CT16Bn_RIS_bits);
__IO_REG32_BIT(CT16B0_IC,       0x4000003C,__READ_WRITE ,__CT16Bn_IC_bits);


/***************************************************************************
 **
 ** CT16B1
 **
 ***************************************************************************/
__IO_REG32_BIT(CT16B1_TMRCTRL,  0x40002000,__READ_WRITE ,__CT16Bn_TMRCTRL_bits);
__IO_REG32_BIT(CT16B1_TC,       0x40002004,__READ_WRITE ,__CT16Bn_TC_bits);
__IO_REG32_BIT(CT16B1_PRE,      0x40002008,__READ_WRITE ,__CT16Bn_PRE_bits);
__IO_REG32_BIT(CT16B1_PC,       0x4000200C,__READ_WRITE ,__CT16Bn_PC_bits);
__IO_REG32_BIT(CT16B1_CNTCTRL,  0x40002010,__READ_WRITE ,__CT16Bn_CNTCTRL_bits);
__IO_REG32_BIT(CT16B1_MCTRL,    0x40002014,__READ_WRITE ,__CT16Bn_MCTRL_bits);
__IO_REG32_BIT(CT16B1_MR0,      0x40002018,__READ_WRITE ,__CT16Bn_MR0_bits);
__IO_REG32_BIT(CT16B1_MR1,      0x4000201C,__READ_WRITE ,__CT16Bn_MR1_bits);
__IO_REG32_BIT(CT16B1_MR2,      0x40002020,__READ_WRITE ,__CT16Bn_MR2_bits);
__IO_REG32_BIT(CT16B1_MR3,      0x40002024,__READ_WRITE ,__CT16Bn_MR3_bits);
__IO_REG32_BIT(CT16B1_CAPCTRL,  0x40002028,__READ_WRITE ,__CT16Bn_CAPCTRL_bits);
__IO_REG32_BIT(CT16B1_CAP0,     0x4000202C,__READ_WRITE ,__CT16Bn_CAP0_bits);
__IO_REG32_BIT(CT16B1_EM,       0x40002030,__READ_WRITE ,__CT16Bn_EM_bits);
__IO_REG32_BIT(CT16B1_PWMCTRL,  0x40002034,__READ_WRITE ,__CT16Bn_PWMCTRL_bits);
__IO_REG32_BIT(CT16B1_RIS,      0x40002038,__READ_WRITE ,__CT16Bn_RIS_bits);
__IO_REG32_BIT(CT16B1_IC,       0x4000203C,__READ_WRITE ,__CT16Bn_IC_bits);


/***************************************************************************
 **
 ** CT32B0
 **
 ***************************************************************************/
__IO_REG32_BIT(CT32B0_TMRCTRL,  0x40004000,__READ_WRITE ,__CT32Bn_TMRCTRL_bits);
__IO_REG32_BIT(CT32B0_TC,       0x40004004,__READ_WRITE ,__CT32Bn_TC_bits);
__IO_REG32_BIT(CT32B0_PRE,      0x40004008,__READ_WRITE ,__CT32Bn_PRE_bits);
__IO_REG32_BIT(CT32B0_PC,       0x4000400C,__READ_WRITE ,__CT32Bn_PC_bits);
__IO_REG32_BIT(CT32B0_CNTCTRL,  0x40004010,__READ_WRITE ,__CT32Bn_CNTCTRL_bits);
__IO_REG32_BIT(CT32B0_MCTRL,    0x40004014,__READ_WRITE ,__CT32Bn_MCTRL_bits);
__IO_REG32_BIT(CT32B0_MR0,      0x40004018,__READ_WRITE ,__CT32Bn_MR0_bits);
__IO_REG32_BIT(CT32B0_MR1,      0x4000401C,__READ_WRITE ,__CT32Bn_MR1_bits);
__IO_REG32_BIT(CT32B0_MR2,      0x40004020,__READ_WRITE ,__CT32Bn_MR2_bits);
__IO_REG32_BIT(CT32B0_MR3,      0x40004024,__READ_WRITE ,__CT32Bn_MR3_bits);
__IO_REG32_BIT(CT32B0_CAPCTRL,  0x40004028,__READ_WRITE ,__CT32Bn_CAPCTRL_bits);
__IO_REG32_BIT(CT32B0_CAP0,     0x4000402C,__READ_WRITE ,__CT32Bn_CAP0_bits);
__IO_REG32_BIT(CT32B0_EM,       0x40004030,__READ_WRITE ,__CT32Bn_EM_bits);
__IO_REG32_BIT(CT32B0_PWMCTRL,  0x40004034,__READ_WRITE ,__CT32Bn_PWMCTRL_bits);
__IO_REG32_BIT(CT32B0_RIS,      0x40004038,__READ_WRITE ,__CT32Bn_RIS_bits);
__IO_REG32_BIT(CT32B0_IC,       0x4000403C,__READ_WRITE ,__CT32Bn_IC_bits);


/***************************************************************************
 **
 ** CT32B1
 **
 ***************************************************************************/
__IO_REG32_BIT(CT32B1_TMRCTRL,  0x40006000,__READ_WRITE ,__CT32Bn_TMRCTRL_bits);
__IO_REG32_BIT(CT32B1_TC,       0x40006004,__READ_WRITE ,__CT32Bn_TC_bits);
__IO_REG32_BIT(CT32B1_PRE,      0x40006008,__READ_WRITE ,__CT32Bn_PRE_bits);
__IO_REG32_BIT(CT32B1_PC,       0x4000600C,__READ_WRITE ,__CT32Bn_PC_bits);
__IO_REG32_BIT(CT32B1_CNTCTRL,  0x40006010,__READ_WRITE ,__CT32Bn_CNTCTRL_bits);
__IO_REG32_BIT(CT32B1_MCTRL,    0x40006014,__READ_WRITE ,__CT32Bn_MCTRL_bits);
__IO_REG32_BIT(CT32B1_MR0,      0x40006018,__READ_WRITE ,__CT32Bn_MR0_bits);
__IO_REG32_BIT(CT32B1_MR1,      0x4000601C,__READ_WRITE ,__CT32Bn_MR1_bits);
__IO_REG32_BIT(CT32B1_MR2,      0x40006020,__READ_WRITE ,__CT32Bn_MR2_bits);
__IO_REG32_BIT(CT32B1_MR3,      0x40006024,__READ_WRITE ,__CT32Bn_MR3_bits);
__IO_REG32_BIT(CT32B1_CAPCTRL,  0x40006028,__READ_WRITE ,__CT32Bn_CAPCTRL_bits);
__IO_REG32_BIT(CT32B1_CAP0,     0x4000602C,__READ_WRITE ,__CT32Bn_CAP0_bits);
__IO_REG32_BIT(CT32B1_EM,       0x40006030,__READ_WRITE ,__CT32Bn_EM_bits);
__IO_REG32_BIT(CT32B1_PWMCTRL,  0x40006034,__READ_WRITE ,__CT32Bn_PWMCTRL_bits);
__IO_REG32_BIT(CT32B1_RIS,      0x40006038,__READ_WRITE ,__CT32Bn_RIS_bits);
__IO_REG32_BIT(CT32B1_IC,       0x4000603C,__READ_WRITE ,__CT32Bn_IC_bits);


/***************************************************************************
 **
 ** WATCHDOG
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_CFG,         0x40010000,__READ_WRITE ,__WDT_CFG_bits);
__IO_REG32_BIT(WDT_CLKSOURCE,   0x40010004,__READ_WRITE ,__WDT_CLKSOURCE_bits);
__IO_REG32_BIT(WDT_TC,          0x40010008,__READ_WRITE ,__WDT_TC_bits);
__IO_REG32_BIT(WDT_FEED,        0x4001000C,__READ_WRITE ,__WDT_FEED_bits);


/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_CTRL,        0x40012000,__READ_WRITE ,__RTC_CTRL_bits);
__IO_REG32_BIT(RTC_CLKS,        0x40012004,__READ_WRITE ,__RTC_CLKS_bits);
__IO_REG32_BIT(RTC_IE,          0x40012008,__READ_WRITE ,__RTC_IE_bits);
__IO_REG32_BIT(RTC_RIS,         0x4001200C,__READ_WRITE ,__RTC_RIS_bits);
__IO_REG32_BIT(RTC_IC,          0x40012010,__READ_WRITE ,__RTC_IC_bits);
__IO_REG32_BIT(RTC_SECCNTV,     0x40012014,__READ_WRITE ,__RTC_SECCNTV_bits);
__IO_REG32_BIT(RTC_SECCNT,      0x40012018,__READ_WRITE ,__RTC_SECCNT_bits);
__IO_REG32_BIT(RTC_ALMCNTV,     0x4001201C,__READ_WRITE ,__RTC_ALMCNTV_bits);
__IO_REG32_BIT(RTC_ALMCNT,      0x40012020,__READ_WRITE ,__RTC_ALMCNT_bits);


/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0_CTRL0,      0x4001C000,__READ_WRITE ,__SSPn_CTRL0_bits);
__IO_REG32_BIT(SSP0_CTRL1,      0x4001C004,__READ_WRITE ,__SSPn_CTRL1_bits);
__IO_REG32_BIT(SSP0_CLKDIV,     0x4001C008,__READ_WRITE ,__SSPn_CLKDIV_bits);
__IO_REG32_BIT(SSP0_STAT,       0x4001C00C,__READ_WRITE ,__SSPn_STAT_bits);
__IO_REG32_BIT(SSP0_IE,         0x4001C010,__READ_WRITE ,__SSPn_IE_bits);
__IO_REG32_BIT(SSP0_RIS,        0x4001C014,__READ_WRITE ,__SSPn_RIS_bits);
__IO_REG32_BIT(SSP0_IC,         0x4001C018,__READ_WRITE ,__SSPn_IC_bits);
__IO_REG32_BIT(SSP0_DATA,       0x4001C01C,__READ_WRITE ,__SSPn_DATA_bits);


/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1_CTRL0,      0x40058000,__READ_WRITE ,__SSPn_CTRL0_bits);
__IO_REG32_BIT(SSP1_CTRL1,      0x40058004,__READ_WRITE ,__SSPn_CTRL1_bits);
__IO_REG32_BIT(SSP1_CLKDIV,     0x40058008,__READ_WRITE ,__SSPn_CLKDIV_bits);
__IO_REG32_BIT(SSP1_STAT,       0x4005800C,__READ_WRITE ,__SSPn_STAT_bits);
__IO_REG32_BIT(SSP1_IE,         0x40058010,__READ_WRITE ,__SSPn_IE_bits);
__IO_REG32_BIT(SSP1_RIS,        0x40058014,__READ_WRITE ,__SSPn_RIS_bits);
__IO_REG32_BIT(SSP1_IC,         0x40058018,__READ_WRITE ,__SSPn_IC_bits);
__IO_REG32_BIT(SSP1_DATA,       0x4005801C,__READ_WRITE ,__SSPn_DATA_bits);


/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_CTRL,       0x40018000,__READ_WRITE ,__I2C_CTRL_bits);
__IO_REG32_BIT(I2C0_STAT,       0x40018004,__READ_WRITE ,__I2C_STAT_bits);
__IO_REG32_BIT(I2C0_TXDATA,     0x40018008,__READ_WRITE ,__I2C_TXDATA_bits);
__IO_REG32_BIT(I2C0_RXDATA,     0x4001800C,__READ_WRITE ,__I2C_RXDATA_bits);
__IO_REG32_BIT(I2C0_SLVADDR0,   0x40018010,__READ_WRITE ,__I2C_SLVADDR0_bits);
__IO_REG32_BIT(I2C0_SLVADDR1,   0x40018014,__READ_WRITE ,__I2C_SLVADDR1_bits);
__IO_REG32_BIT(I2C0_SLVADDR2,   0x40018018,__READ_WRITE ,__I2C_SLVADDR2_bits);
__IO_REG32_BIT(I2C0_SLVADDR3,   0x4001801C,__READ_WRITE ,__I2C_SLVADDR3_bits);
__IO_REG32_BIT(I2C0_SCLHT,      0x40018020,__READ_WRITE ,__I2C_SCLHT_bits);
__IO_REG32_BIT(I2C0_SCLLT,      0x40018024,__READ_WRITE ,__I2C_SCLLT_bits);
__IO_REG32_BIT(I2C0_TOCTRL,     0x4001802C,__READ_WRITE ,__I2C_TOCTRL_bits);
__IO_REG32_BIT(I2C0_MMCTRL,     0x40018030,__READ_WRITE ,__I2C_MMCTRL_bits);
__IO_REG32_BIT(I2C0_Engineer,   0x40018034,__READ_WRITE ,__I2C_Engineer_bits);


/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_CTRL,       0x4005A000,__READ_WRITE ,__I2C_CTRL_bits);
__IO_REG32_BIT(I2C1_STAT,       0x4005A004,__READ_WRITE ,__I2C_STAT_bits);
__IO_REG32_BIT(I2C1_TXDATA,     0x4005A008,__READ_WRITE ,__I2C_TXDATA_bits);
__IO_REG32_BIT(I2C1_RXDATA,     0x4005A00C,__READ_WRITE ,__I2C_RXDATA_bits);
__IO_REG32_BIT(I2C1_SLVADDR0,   0x4005A010,__READ_WRITE ,__I2C_SLVADDR0_bits);
__IO_REG32_BIT(I2C1_SLVADDR1,   0x4005A014,__READ_WRITE ,__I2C_SLVADDR1_bits);
__IO_REG32_BIT(I2C1_SLVADDR2,   0x4005A018,__READ_WRITE ,__I2C_SLVADDR2_bits);
__IO_REG32_BIT(I2C1_SLVADDR3,   0x4005A01C,__READ_WRITE ,__I2C_SLVADDR3_bits);
__IO_REG32_BIT(I2C1_SCLHT,      0x4005A020,__READ_WRITE ,__I2C_SCLHT_bits);
__IO_REG32_BIT(I2C1_SCLLT,      0x4005A024,__READ_WRITE ,__I2C_SCLLT_bits);
__IO_REG32_BIT(I2C1_TOCTRL,     0x4005A02C,__READ_WRITE ,__I2C_TOCTRL_bits);
__IO_REG32_BIT(I2C1_MMCTRL,     0x4005A030,__READ_WRITE ,__I2C_MMCTRL_bits);
__IO_REG32_BIT(I2C1_Engineer,   0x4005A034,__READ_WRITE ,__I2C_Engineer_bits);


/***************************************************************************
 **
 ** USART0
 **
 ***************************************************************************/
__IO_REG32_BIT(USART0_RB,            0x40016000,__READ_WRITE ,__USARTn_RB_bits);
#define USART0_TH           USART0_RB
#define USART0_TH_bit       USART0_RB_bit
#define USART0_DLL          USART0_RB
#define USART0_DLL_bit      USART0_RB_bit
__IO_REG32_BIT(USART0_DLM,           0x40016004,__READ_WRITE ,__USARTn_DLM_bits);
#define USART0_IE           USART0_DLM
#define USART0_IE_bit       USART0_DLM_bit
__IO_REG32_BIT(USART0_II,            0x40016008,__READ_WRITE ,__USARTn_II_bits);
#define USART0_FIFOCTRL     USART0_II
#define USART0_FIFOCTRL_bit USART0_II_bit.__USARTn_FIFOCTRL_bits
__IO_REG32_BIT(USART0_LC,            0x4001600C,__READ_WRITE ,__USARTn_LC_bits);
__IO_REG32_BIT(USART0_MC,            0x40016010,__READ_WRITE ,__USARTn_MC_bits);
__IO_REG32_BIT(USART0_LS,            0x40016014,__READ_WRITE ,__USARTn_LS_bits);
__IO_REG32_BIT(USART0_MS,            0x40016018,__READ_WRITE ,__USARTn_MS_bits);
__IO_REG32_BIT(USART0_SP,            0x4001601C,__READ_WRITE ,__USARTn_SP_bits);
__IO_REG32_BIT(USART0_ABCTRL,        0x40016020,__READ_WRITE ,__USARTn_ABCTRL_bits);
__IO_REG32_BIT(USART0_IRDACTRL,      0x40016024,__READ_WRITE ,__USARTn_IRDACTRL_bits);
__IO_REG32_BIT(USART0_FD,            0x40016028,__READ_WRITE ,__USARTn_FD_bits);
__IO_REG32_BIT(USART0_CTRL,          0x40016030,__READ_WRITE ,__USARTn_CTRL_bits);
__IO_REG32_BIT(USART0_HDEN,          0x40016034,__READ_WRITE ,__USARTn_HDEN_bits);
__IO_REG32_BIT(USART0_SCICTRL,       0x40016038,__READ_WRITE ,__USARTn_SCICTRL_bits);
__IO_REG32_BIT(USART0_RS485CTRL,     0x4001603C,__READ_WRITE ,__USARTn_RS485CTRL_bits);
__IO_REG32_BIT(USART0_RS485ADRMATCH, 0x40016040,__READ_WRITE ,__USARTn_RS485ADRMATCH_bits);
__IO_REG32_BIT(USART0_RS485DLYV,     0x40016044,__READ_WRITE ,__USARTn_RS485DLYV_bits);
__IO_REG32_BIT(USART0_SYNCCTRL,      0x40016048,__READ_WRITE ,__USARTn_SYNCCTRL_bits);


/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_CTRL,             0x4001A000,__READ_WRITE ,__I2S_CTRL_bits);
__IO_REG32_BIT(I2S_CLK,              0x4001A004,__READ_WRITE ,__I2S_CLK_bits);
__IO_REG32_BIT(I2S_STATUS,           0x4001A008,__READ_WRITE ,__I2S_STATUS_bits);
__IO_REG32_BIT(I2S_IE,               0x4001A00C,__READ_WRITE ,__I2S_IE_bits);
__IO_REG32_BIT(I2S_RIS,              0x4001A010,__READ_WRITE ,__I2S_RIS_bits);
__IO_REG32_BIT(I2S_IC,               0x4001A014,__READ_WRITE ,__I2S_IC_bits);
__IO_REG32_BIT(I2S_FIFO,             0x4001A018,__READ_WRITE ,__I2S_FIFO_bits);


/***************************************************************************
 **
 ** FMC
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASH_STATUS,         0x40062004,__READ_WRITE ,__FLASH_STATUS_bits);
__IO_REG32_BIT(FLASH_CTRL,           0x40062008,__READ_WRITE ,__FLASH_CTRL_bits);
__IO_REG32_BIT(FLASH_DATA,           0x4006200C,__READ_WRITE ,__FLASH_DATA_bits);
__IO_REG32_BIT(FLASH_ADDR,           0x40062010,__READ_WRITE ,__FLASH_ADDR_bits);
__IO_REG32_BIT(FLASH_CHKSUM,         0x40062014,__READ_WRITE ,__FLASH_CHKSUM_bits);


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

/******  Cortex-M0 Processor Exceptions Number  *****************************************/
/*
  NonMaskableInt_IRQn     = -14,    // 2 Non Maskable Interrupt             //
  HardFault_IRQn          = -13,    // 3 Cortex-M0 Hard Fault Interrupt     //
  SVCall_IRQn             = -5,     // 11 Cortex-M0 SV Call Interrupt       //
  PendSV_IRQn             = -2,     // 14 Cortex-M0 Pend SV Interrupt       //
  SysTick_IRQn            = -1,     // 15 Cortex-M0 System Tick Interrupt   //
*/
/******  SN32F706 Specific Interrupt Numbers  *****************************************/
#define MAIN_STACK           0
#define Reset                1
#define NMI_Handler          2
#define HardFault_Handler    3
#define SVCCall             11
#define PendSV              14
#define SysTick             15
#define NVIC_WAKE_INT       16
#define NVIC_SSP0_INT       29
#define NVIC_SSP1_INT       30
#define NVIC_I2C0_INT       31
#define NVIC_CT16B0_INT     32
#define NVIC_CT16B1_INT     33
#define NVIC_CT32B0_INT     34
#define NVIC_CT32B1_INT     35
#define NVIC_I2S_INT        36
#define NVIC_USART0_INT     37
#define NVIC_USART1_INT     38
#define NVIC_I2C1_INT       39
#define NVIC_ADC_INT        40
#define NVIC_WDT_INT        41
#define NVIC_LVD_INT        42
#define NVIC_RTC_INT        43
#define NVIC_P3_INT         44
#define NVIC_P2_INT         45
#define NVIC_P1_INT         46
#define NVIC_P0_INT         47

#endif  /* __SN32F706_H */
