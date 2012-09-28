/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC1777
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 50425 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC1777_H
#define __IOLPC1777_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC1777 SPECIAL FUNCTION REGISTERS
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

/* Reset Source Identification Register */
typedef struct{
__REG32 POR         : 1;
__REG32 EXTR        : 1;
__REG32 WDTR        : 1;
__REG32 BODR        : 1;
__REG32 SYSRESET    : 1;
__REG32 LOCKUP      : 1;
__REG32             :26;
} __rsid_bits;

/* External interrupt register */
typedef struct{
__REG32 EINT0       : 1;
__REG32 EINT1       : 1;
__REG32 EINT2       : 1;
__REG32 EINT3       : 1;
__REG32             :28;
} __extint_bits;

/* External Interrupt Mode Register */
typedef struct{
__REG32 EXTMODE0    : 1;
__REG32 EXTMODE1    : 1;
__REG32 EXTMODE2    : 1;
__REG32 EXTMODE3    : 1;
__REG32             :28;
} __extmode_bits;

/* External Interrupt Polarity Register */
typedef struct{
__REG32 EXTPOLAR0   : 1;
__REG32 EXTPOLAR1   : 1;
__REG32 EXTPOLAR2   : 1;
__REG32 EXTPOLAR3   : 1;
__REG32             :28;
} __extpolar_bits;

/* System Controls and Status register */
typedef struct{
__REG32 EMCSC       : 1;
__REG32 EMCRD       : 1;
__REG32 EMCBC       : 1;
__REG32 MCIPWR      : 1;
__REG32 OSCRANGE    : 1;
__REG32 OSCEN       : 1;
__REG32 OSCSTAT     : 1;
__REG32             :25;
} __scs_bits;

/* Clock Soucre Select register */
typedef struct{
__REG32 CLKSRC      : 1;
__REG32             :31;
} __clksrcsel_bits;

/* PLL control register */
typedef struct{
__REG32 PLLE        : 1;
__REG32             :31;
} __pllcon_bits;

/* PLL config register */
typedef struct{
__REG32 MSEL        : 5;
__REG32 PSEL        : 2;
__REG32             :25;
} __pllcfg_bits;

/* PLL status register */
typedef struct{
__REG32 MSEL        : 5;
__REG32 PSEL        : 2;
__REG32             : 1;
__REG32 PLLE_STAT   : 1;
__REG32             : 1;
__REG32 PLOCK       : 1;
__REG32             :21;
} __pllstat_bits;

/* PLL feed register */
typedef struct{
__REG32 FEED        : 8;
__REG32             :24;
} __pllfeed_bits;

/* CPU Clock Selection register */
typedef struct{
__REG32 CCLKDIV     : 5;
__REG32             : 3;
__REG32 CCLKSEL     : 1;
__REG32             :23;
} __cclksel_bits;

/* USB Clock Selection register */
typedef struct{
__REG32 USBDIV      : 5;
__REG32             : 3;
__REG32 USBSEL      : 2;
__REG32             :22;
} __usbclksel_bits;

/* EMC Clock Selection register */
typedef struct{
__REG32 EMCDIV      : 1;
__REG32             :31;
} __emcclksel_bits;

/* Peripheral Clock Selection register */
typedef struct{
__REG32 PCLKDIV     : 5;
__REG32             :27;
} __pclksel_bits;

/* Reset control register 0 */
typedef struct{
__REG32             : 1;
__REG32 RSTTIM0     : 1;
__REG32 RSTTIM1     : 1;
__REG32 RSTUART0    : 1;
__REG32 RSTUART1    : 1;
__REG32 RSTPWM0     : 1;
__REG32 RSTPWM1     : 1;
__REG32 RSTI2C0     : 1;
__REG32 RSTUART4    : 1;
__REG32 RSTRTC      : 1;
__REG32 RSTSSP1     : 1;
__REG32 RSTEMC      : 1;
__REG32 RSTADC      : 1;
__REG32 RSTCAN1     : 1;
__REG32 RSTCAN2     : 1;
__REG32 RSTGPIO     : 1;
__REG32             : 1;
__REG32 RSTMCPWM    : 1;
__REG32 RSTQEI      : 1;
__REG32 RSTI2C1     : 1;
__REG32 RSTSSP2     : 1;
__REG32 RSTSSP0     : 1;
__REG32 RSTTIM2     : 1;
__REG32 RSTTIM3     : 1;
__REG32 RSTUART2    : 1;
__REG32 RSTUART3    : 1;
__REG32 RSTI2C2     : 1;
__REG32 RSTI2S      : 1;
__REG32 RSTSDC      : 1;
__REG32 RSTGPDMA    : 1;
__REG32             : 1;
__REG32 RSTUSB      : 1;
} __rstcon0_bits;

/* Reset control register 1 */
typedef struct{
__REG32 RSTIOCON    : 1;
__REG32 RSTDAC      : 1;
__REG32 RSTCANACC   : 1;
__REG32             :29;
} __rstcon1_bits;

/* Clock Output Configuration register */
typedef struct{
__REG32 Boost       : 2;
__REG32             :30;
} __pboost_bits;

/* Power control register */
typedef struct{
__REG32 PM0         : 1;
__REG32 PM1         : 1;
__REG32 BODPDM      : 1;
__REG32 BOGD        : 1;
__REG32 BORD        : 1;
__REG32             : 3;
__REG32 SMFLAG      : 1;
__REG32 DSFLAG      : 1;
__REG32 PDFLAG      : 1;
__REG32 DPDFLAG     : 1;
__REG32             :20;
}__pcon_bits;

/* Power control for peripherals register */
typedef struct{
__REG32             : 1;
__REG32 PCTIM0      : 1;
__REG32 PCTIM1      : 1;
__REG32 PCUART0     : 1;
__REG32 PCUART1     : 1;
__REG32 PCPWM0      : 1;
__REG32 PCPWM1      : 1;
__REG32 PCI2C0      : 1;
__REG32 PCUART4     : 1;
__REG32 PCRTC       : 1;
__REG32 PCSSP1      : 1;
__REG32 PCEMC       : 1;
__REG32 PCAD        : 1;
__REG32 PCAN1       : 1;
__REG32 PCAN2       : 1;
__REG32 PCGPIO      : 1;
__REG32             : 1;
__REG32 PCMCPWM     : 1;
__REG32 PCQEI       : 1;
__REG32 PCI2C1      : 1;
__REG32 PCSSP2      : 1;
__REG32 PCSSP0      : 1;
__REG32 PCTIM2      : 1;
__REG32 PCTIM3      : 1;
__REG32 PCUART2     : 1;
__REG32 PCUART3     : 1;
__REG32 PCI2C2      : 1;
__REG32 PCI2S       : 1;
__REG32 PCSDC       : 1;
__REG32 PCGPDMA     : 1;
__REG32             : 1;
__REG32 PCUSB       : 1;
} __pconp_bits;

/* Clock Output Configuration register */
typedef struct{
__REG32 CLKOUTSEL   : 4;
__REG32 CLKOUTDIV   : 4;
__REG32 CLKOUT_EN   : 1;
__REG32 CLKOUT_ACT  : 1;
__REG32             :22;
} __clkoutcfg_bits;

/* Flash Accelerator Configuration register */
typedef struct{
__REG32             :12;
__REG32 FLASHTIM    : 4;
__REG32             :16;
} __flashcfg_bits;

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

/* Interrupt Set-Enable Registers 32-63 */
typedef struct {
  __REG32  SETENA32       : 1;
  __REG32  SETENA33       : 1;
  __REG32  SETENA34       : 1;
  __REG32  SETENA35       : 1;
  __REG32  SETENA36       : 1;
  __REG32  SETENA37       : 1;
  __REG32  SETENA38       : 1;
  __REG32  SETENA39       : 1;
  __REG32  SETENA40       : 1;
  __REG32  SETENA41       : 1;
  __REG32  SETENA42       : 1;
  __REG32  SETENA43       : 1;
  __REG32  SETENA44       : 1;
  __REG32  SETENA45       : 1;
  __REG32  SETENA46       : 1;
  __REG32  SETENA47       : 1;
  __REG32  SETENA48       : 1;
  __REG32  SETENA49       : 1;
  __REG32  SETENA50       : 1;
  __REG32  SETENA51       : 1;
  __REG32  SETENA52       : 1;
  __REG32  SETENA53       : 1;
  __REG32  SETENA54       : 1;
  __REG32  SETENA55       : 1;
  __REG32  SETENA56       : 1;
  __REG32  SETENA57       : 1;
  __REG32  SETENA58       : 1;
  __REG32  SETENA59       : 1;
  __REG32  SETENA60       : 1;
  __REG32  SETENA61       : 1;
  __REG32  SETENA62       : 1;
  __REG32  SETENA63       : 1;
} __setena1_bits;

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

/* Interrupt Clear-Enable Registers 32-63 */
typedef struct {
  __REG32  CLRENA32       : 1;
  __REG32  CLRENA33       : 1;
  __REG32  CLRENA34       : 1;
  __REG32  CLRENA35       : 1;
  __REG32  CLRENA36       : 1;
  __REG32  CLRENA37       : 1;
  __REG32  CLRENA38       : 1;
  __REG32  CLRENA39       : 1;
  __REG32  CLRENA40       : 1;
  __REG32  CLRENA41       : 1;
  __REG32  CLRENA42       : 1;
  __REG32  CLRENA43       : 1;
  __REG32  CLRENA44       : 1;
  __REG32  CLRENA45       : 1;
  __REG32  CLRENA46       : 1;
  __REG32  CLRENA47       : 1;
  __REG32  CLRENA48       : 1;
  __REG32  CLRENA49       : 1;
  __REG32  CLRENA50       : 1;
  __REG32  CLRENA51       : 1;
  __REG32  CLRENA52       : 1;
  __REG32  CLRENA53       : 1;
  __REG32  CLRENA54       : 1;
  __REG32  CLRENA55       : 1;
  __REG32  CLRENA56       : 1;
  __REG32  CLRENA57       : 1;
  __REG32  CLRENA58       : 1;
  __REG32  CLRENA59       : 1;
  __REG32  CLRENA60       : 1;
  __REG32  CLRENA61       : 1;
  __REG32  CLRENA62       : 1;
  __REG32  CLRENA63       : 1;
} __clrena1_bits;

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

/* Interrupt Set-Pending Register 32-63 */
typedef struct {
  __REG32  SETPEND32      : 1;
  __REG32  SETPEND33      : 1;
  __REG32  SETPEND34      : 1;
  __REG32  SETPEND35      : 1;
  __REG32  SETPEND36      : 1;
  __REG32  SETPEND37      : 1;
  __REG32  SETPEND38      : 1;
  __REG32  SETPEND39      : 1;
  __REG32  SETPEND40      : 1;
  __REG32  SETPEND41      : 1;
  __REG32  SETPEND42      : 1;
  __REG32  SETPEND43      : 1;
  __REG32  SETPEND44      : 1;
  __REG32  SETPEND45      : 1;
  __REG32  SETPEND46      : 1;
  __REG32  SETPEND47      : 1;
  __REG32  SETPEND48      : 1;
  __REG32  SETPEND49      : 1;
  __REG32  SETPEND50      : 1;
  __REG32  SETPEND51      : 1;
  __REG32  SETPEND52      : 1;
  __REG32  SETPEND53      : 1;
  __REG32  SETPEND54      : 1;
  __REG32  SETPEND55      : 1;
  __REG32  SETPEND56      : 1;
  __REG32  SETPEND57      : 1;
  __REG32  SETPEND58      : 1;
  __REG32  SETPEND59      : 1;
  __REG32  SETPEND60      : 1;
  __REG32  SETPEND61      : 1;
  __REG32  SETPEND62      : 1;
  __REG32  SETPEND63      : 1;
} __setpend1_bits;

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

/* Interrupt Clear-Pending Register 32-63 */
typedef struct {
  __REG32  CLRPEND32      : 1;
  __REG32  CLRPEND33      : 1;
  __REG32  CLRPEND34      : 1;
  __REG32  CLRPEND35      : 1;
  __REG32  CLRPEND36      : 1;
  __REG32  CLRPEND37      : 1;
  __REG32  CLRPEND38      : 1;
  __REG32  CLRPEND39      : 1;
  __REG32  CLRPEND40      : 1;
  __REG32  CLRPEND41      : 1;
  __REG32  CLRPEND42      : 1;
  __REG32  CLRPEND43      : 1;
  __REG32  CLRPEND44      : 1;
  __REG32  CLRPEND45      : 1;
  __REG32  CLRPEND46      : 1;
  __REG32  CLRPEND47      : 1;
  __REG32  CLRPEND48      : 1;
  __REG32  CLRPEND49      : 1;
  __REG32  CLRPEND50      : 1;
  __REG32  CLRPEND51      : 1;
  __REG32  CLRPEND52      : 1;
  __REG32  CLRPEND53      : 1;
  __REG32  CLRPEND54      : 1;
  __REG32  CLRPEND55      : 1;
  __REG32  CLRPEND56      : 1;
  __REG32  CLRPEND57      : 1;
  __REG32  CLRPEND58      : 1;
  __REG32  CLRPEND59      : 1;
  __REG32  CLRPEND60      : 1;
  __REG32  CLRPEND61      : 1;
  __REG32  CLRPEND62      : 1;
  __REG32  CLRPEND63      : 1;
} __clrpend1_bits;

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

/* Active Bit Register 32-63 */
typedef struct {
  __REG32  ACTIVE32       : 1;
  __REG32  ACTIVE33       : 1;
  __REG32  ACTIVE34       : 1;
  __REG32  ACTIVE35       : 1;
  __REG32  ACTIVE36       : 1;
  __REG32  ACTIVE37       : 1;
  __REG32  ACTIVE38       : 1;
  __REG32  ACTIVE39       : 1;
  __REG32  ACTIVE40       : 1;
  __REG32  ACTIVE41       : 1;
  __REG32  ACTIVE42       : 1;
  __REG32  ACTIVE43       : 1;
  __REG32  ACTIVE44       : 1;
  __REG32  ACTIVE45       : 1;
  __REG32  ACTIVE46       : 1;
  __REG32  ACTIVE47       : 1;
  __REG32  ACTIVE48       : 1;
  __REG32  ACTIVE49       : 1;
  __REG32  ACTIVE50       : 1;
  __REG32  ACTIVE51       : 1;
  __REG32  ACTIVE52       : 1;
  __REG32  ACTIVE53       : 1;
  __REG32  ACTIVE54       : 1;
  __REG32  ACTIVE55       : 1;
  __REG32  ACTIVE56       : 1;
  __REG32  ACTIVE57       : 1;
  __REG32  ACTIVE58       : 1;
  __REG32  ACTIVE59       : 1;
  __REG32  ACTIVE60       : 1;
  __REG32  ACTIVE61       : 1;
  __REG32  ACTIVE62       : 1;
  __REG32  ACTIVE63       : 1;
} __active1_bits;

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

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32  PRI_32         : 8;
  __REG32  PRI_33         : 8;
  __REG32  PRI_34         : 8;
  __REG32  PRI_35         : 8;
} __pri8_bits;

/* Interrupt Priority Registers 36-39 */
typedef struct {
  __REG32  PRI_36         : 8;
  __REG32  PRI_37         : 8;
  __REG32  PRI_38         : 8;
  __REG32  PRI_39         : 8;
} __pri9_bits;

/* Interrupt Priority Registers 40-43 */
typedef struct {
  __REG32  PRI_40         : 8;
  __REG32  PRI_41         : 8;
  __REG32  PRI_42         : 8;
  __REG32  PRI_43         : 8;
} __pri10_bits;

/* Interrupt Priority Registers 44-47 */
typedef struct {
  __REG32  PRI_44         : 8;
  __REG32  PRI_45         : 8;
  __REG32  PRI_46         : 8;
  __REG32  PRI_47         : 8;
} __pri11_bits;

/* Interrupt Priority Registers 48-51 */
typedef struct {
  __REG32  PRI_48         : 8;
  __REG32  PRI_49         : 8;
  __REG32  PRI_50         : 8;
  __REG32  PRI_51         : 8;
} __pri12_bits;

/* Interrupt Priority Registers 52-55 */
typedef struct {
  __REG32  PRI_52         : 8;
  __REG32  PRI_53         : 8;
  __REG32  PRI_54         : 8;
  __REG32  PRI_55         : 8;
} __pri13_bits;

/* Interrupt Priority Registers 56-59 */
typedef struct {
  __REG32  PRI_56         : 8;
  __REG32  PRI_57         : 8;
  __REG32  PRI_58         : 8;
  __REG32  PRI_59         : 8;
} __pri14_bits;

/* Interrupt Priority Registers 60-63 */
typedef struct {
  __REG32  PRI_60         : 8;
  __REG32  PRI_61         : 8;
  __REG32  PRI_62         : 8;
  __REG32  PRI_63         : 8;
} __pri15_bits;

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
  __REG32                 : 1;
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

/* Type D IOCON registers (applies to most GPIO port pins) */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32  INV            : 1;
  __REG32                 : 2;
  __REG32  SLEW           : 1;
  __REG32  OD             : 1;
  __REG32                 :21;
} __iocon_d_bits;

/* Type A IOCON registers (applies to pins that include an analog function) */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32  INV            : 1;
  __REG32  ADMODE         : 1;
  __REG32  FILTER         : 1;
  __REG32                 : 1;
  __REG32  OD             : 1;
  __REG32                 :21;
} __iocon_a_bits;

/* Type A IOCON P0[26] register*/
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32  INV            : 1;
  __REG32  ADMODE         : 1;
  __REG32  FILTER         : 1;
  __REG32                 : 1;
  __REG32  OD             : 1;
  __REG32                 : 5;
  __REG32  DACEN          : 1;
  __REG32                 :15;
} __iocon_p0_26_a_bits;

/* Type U IOCON registers (applies to pins that include a USB D+ or Dfunction) */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32                 :29;
} __iocon_u_bits;

/* Type I IOCON registers (applies to pins that include a specialized I2C function) */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32                 : 3;
  __REG32  INV            : 1;
  __REG32                 : 1;
  __REG32  HS             : 1;
  __REG32  HIDRIVE        : 1;
  __REG32                 :22;
} __iocon_i_bits;

/* Type W IOCON registers (these pins are otherwise the same as Type D, but
include a selectable input glitch filter, and default to pull-down/pull-up
disabled) */
typedef struct {
  __REG32  FUNC           : 3;
  __REG32  MODE           : 2;
  __REG32  HYS            : 1;
  __REG32  INV            : 1;
  __REG32                 : 1;
  __REG32  FILTER         : 1;
  __REG32  SLEW           : 1;
  __REG32  OD             : 1;
  __REG32                 :21;
} __iocon_w_bits;

/* GPIO 0 Registers */
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
__REG32 P0_24  : 1;
__REG32 P0_25  : 1;
__REG32 P0_26  : 1;
__REG32 P0_27  : 1;
__REG32 P0_28  : 1;
__REG32 P0_29  : 1;
__REG32 P0_30  : 1;
__REG32 P0_31  : 1;
} __gpio0_bits;

/* FGPIO 0 Registers*/
typedef union{
  /*FIO0DIR*/
  /*FIO0MASK*/
  /*FIO0PIN*/
  /*FIO0SET*/
  /*FIO0CLR*/
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
    __REG32 P0_24  : 1;
    __REG32 P0_25  : 1;
    __REG32 P0_26  : 1;
    __REG32 P0_27  : 1;
    __REG32 P0_28  : 1;
    __REG32 P0_29  : 1;
    __REG32 P0_30  : 1;
    __REG32 P0_31  : 1;
  };

  struct
  {
    union
    {
    /*FIO0DIR0*/
    /*FIO0MASK0*/
    /*FIO0PIN0*/
    /*FIO0SET0*/
    /*FIO0CLR0*/
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
    /*FIO0DIR1*/
    /*FIO0MASK1*/
    /*FIO0PIN1*/
    /*FIO0SET1*/
    /*FIO0CLR1*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8  P0_4   : 1;
        __REG8  P0_5   : 1;
        __REG8  P0_6   : 1;
        __REG8  P0_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    union
    {
     /*FIO0DIR2*/
     /*FIO0MASK2*/
     /*FIO0PIN2*/
     /*FIO0SET2*/
     /*FIO0CLR2*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8  P0_4   : 1;
        __REG8  P0_5   : 1;
        __REG8  P0_6   : 1;
        __REG8  P0_7   : 1;
      } __byte2_bit;
      __REG8 __byte2;
    };
    union
    {
      /*FIO0DIR3*/
      /*FIO0MASK3*/
      /*FIO0PIN3*/
      /*FIO0SET3*/
      /*FIO0CLR3*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8  P0_4   : 1;
        __REG8  P0_5   : 1;
        __REG8  P0_6   : 1;
        __REG8  P0_7   : 1;
      } __byte3_bit;
      __REG8 __byte3;
    };
  };

  struct
  {
    union
    {
      /*FIO0DIRL*/
      /*FIO0MASKL*/
      /*FIO0PINL*/
      /*FIO0SETL*/
      /*FIO0CLRL*/
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
        __REG16 P0_12  : 1;
        __REG16 P0_13  : 1;
        __REG16 P0_14  : 1;
        __REG16 P0_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    union
    {
      /*FIO0DIRU*/
      /*FIO0MASKU*/
      /*FIO0PINU*/
      /*FIO0SETU*/
      /*FIO0CLRU*/
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
        __REG16 P0_12  : 1;
        __REG16 P0_13  : 1;
        __REG16 P0_14  : 1;
        __REG16 P0_15  : 1;
      } __shortu_bit;
      __REG16 __shortu;
    };
  };
} __fgpio0_bits;

/* GPIO 1 Registers */
typedef struct {
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
__REG32 P1_12  : 1;
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
__REG32 P1_30  : 1;
__REG32 P1_31  : 1;
} __gpio1_bits;

/* FGPIO 1 Registers*/
typedef union{
  /*FIO1DIR*/
  /*FIO1MASK*/
  /*FIO1PIN*/
  /*FIO1SET*/
  /*FIO1CLR*/
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
    __REG32 P1_12  : 1;
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
    __REG32 P1_30  : 1;
    __REG32 P1_31  : 1;
  };

  struct
  {
    union
    {
      /*FIO1DIR0*/
      /*FIO1MASK0*/
      /*FIO1PIN0*/
      /*FIO1SET0*/
      /*FIO1CLR0*/
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
      /*FIO1DIR1*/
      /*FIO1MASK1*/
      /*FIO1PIN1*/
      /*FIO1SET1*/
      /*FIO1CLR1*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8  P1_4   : 1;
        __REG8  P1_5   : 1;
        __REG8  P1_6   : 1;
        __REG8  P1_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    union
    {
      /*FIO1DIR2*/
      /*FIO1MASK2*/
      /*FIO1PIN2*/
      /*FIO1SET2*/
      /*FIO1CLR2*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8  P1_4   : 1;
        __REG8  P1_5   : 1;
        __REG8  P1_6   : 1;
        __REG8  P1_7   : 1;
      } __byte2_bit;
      __REG8 __byte2;
    };
    union
    {
      /*FIO1DIR3*/
      /*FIO1MASK3*/
      /*FIO1PIN3*/
      /*FIO1SET3*/
      /*FIO1CLR3*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8  P1_4   : 1;
        __REG8  P1_5   : 1;
        __REG8  P1_6   : 1;
        __REG8  P1_7   : 1;
      } __byte3_bit;
      __REG8 __byte3;
    };
  };

  struct
  {
    union
    {
      /*FIO1DIRL*/
      /*FIO1MASKL*/
      /*FIO1PINL*/
      /*FIO1SETL*/
      /*FIO1CLRL*/
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
        __REG16 P1_12  : 1;
        __REG16 P1_13  : 1;
        __REG16 P1_14  : 1;
        __REG16 P1_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    union
    {
      /*FIO1DIRU*/
      /*FIO1MASKU*/
      /*FIO1PINU*/
      /*FIO1SETU*/
      /*FIO1CLRU*/
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
        __REG16 P1_12  : 1;
        __REG16 P1_13  : 1;
        __REG16 P1_14  : 1;
        __REG16 P1_15  : 1;
      } __shortu_bit;
      __REG16 __shortu;
    };
  };
} __fgpio1_bits;

/* GPIO 2 Registers */
typedef struct {
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
__REG32 P2_12  : 1;
__REG32 P2_13  : 1;
__REG32 P2_14  : 1;
__REG32 P2_15  : 1;
__REG32 P2_16  : 1;
__REG32 P2_17  : 1;
__REG32 P2_18  : 1;
__REG32 P2_19  : 1;
__REG32 P2_20  : 1;
__REG32 P2_21  : 1;
__REG32 P2_22  : 1;
__REG32 P2_23  : 1;
__REG32 P2_24  : 1;
__REG32 P2_25  : 1;
__REG32 P2_26  : 1;
__REG32 P2_27  : 1;
__REG32 P2_28  : 1;
__REG32 P2_29  : 1;
__REG32 P2_30  : 1;
__REG32 P2_31  : 1;
} __gpio2_bits;

/* FGPIO 2 Registers*/
typedef union{
  /*FIO2DIR*/
  /*FIO2MASK*/
  /*FIO2PIN*/
  /*FIO2SET*/
  /*FIO2CLR*/
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
    __REG32 P2_12  : 1;
    __REG32 P2_13  : 1;
    __REG32 P2_14  : 1;
    __REG32 P2_15  : 1;
    __REG32 P2_16  : 1;
    __REG32 P2_17  : 1;
    __REG32 P2_18  : 1;
    __REG32 P2_19  : 1;
    __REG32 P2_20  : 1;
    __REG32 P2_21  : 1;
    __REG32 P2_22  : 1;
    __REG32 P2_23  : 1;
    __REG32 P2_24  : 1;
    __REG32 P2_25  : 1;
    __REG32 P2_26  : 1;
    __REG32 P2_27  : 1;
    __REG32 P2_28  : 1;
    __REG32 P2_29  : 1;
    __REG32 P2_30  : 1;
    __REG32 P2_31  : 1;
  };

  struct
  {
    union
    {
      /*FIO2DIR0*/
      /*FIO2MASK0*/
      /*FIO2PIN0*/
      /*FIO2SET0*/
      /*FIO2CLR0*/
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
      /*FIO2DIR1*/
      /*FIO2MASK1*/
      /*FIO2PIN1*/
      /*FIO2SET1*/
      /*FIO2CLR1*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8  P2_4   : 1;
        __REG8  P2_5   : 1;
        __REG8  P2_6   : 1;
        __REG8  P2_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    union
    {
      /*FIO2DIR2*/
      /*FIO2MASK2*/
      /*FIO2PIN2*/
      /*FIO2SET2*/
      /*FIO2CLR2*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8  P2_4   : 1;
        __REG8  P2_5   : 1;
        __REG8  P2_6   : 1;
        __REG8  P2_7   : 1;
      } __byte2_bit;
      __REG8 __byte2;
    };
    union
    {
      /*FIO2DIR3*/
      /*FIO2MASK3*/
      /*FIO2PIN3*/
      /*FIO2SET3*/
      /*FIO2CLR3*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8  P2_4   : 1;
        __REG8  P2_5   : 1;
        __REG8  P2_6   : 1;
        __REG8  P2_7   : 1;
      } __byte3_bit;
      __REG8 __byte3;
    };
  };

  struct
  {
    union
    {
      /*FIO2DIRL*/
      /*FIO2MASKL*/
      /*FIO2PINL*/
      /*FIO2SETL*/
      /*FIO2CLRL*/
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
        __REG16 P2_12  : 1;
        __REG16 P2_13  : 1;
        __REG16 P2_14  : 1;
        __REG16 P2_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    union
    {
      /*FIO2DIRU*/
      /*FIO2MASKU*/
      /*FIO2PINU*/
      /*FIO2SETU*/
      /*FIO2CLRU*/
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
        __REG16 P2_12  : 1;
        __REG16 P2_13  : 1;
        __REG16 P2_14  : 1;
        __REG16 P2_15  : 1;
      } __shortu_bit;
      __REG16 __shortu;
    };
  };
} __fgpio2_bits;

/* GPIO 3 Registers */
typedef struct {
__REG32 P3_0   : 1;
__REG32 P3_1   : 1;
__REG32 P3_2   : 1;
__REG32 P3_3   : 1;
__REG32 P3_4   : 1;
__REG32 P3_5   : 1;
__REG32 P3_6   : 1;
__REG32 P3_7   : 1;
__REG32 P3_8   : 1;
__REG32 P3_9   : 1;
__REG32 P3_10  : 1;
__REG32 P3_11  : 1;
__REG32 P3_12  : 1;
__REG32 P3_13  : 1;
__REG32 P3_14  : 1;
__REG32 P3_15  : 1;
__REG32 P3_16  : 1;
__REG32 P3_17  : 1;
__REG32 P3_18  : 1;
__REG32 P3_19  : 1;
__REG32 P3_20  : 1;
__REG32 P3_21  : 1;
__REG32 P3_22  : 1;
__REG32 P3_23  : 1;
__REG32 P3_24  : 1;
__REG32 P3_25  : 1;
__REG32 P3_26  : 1;
__REG32 P3_27  : 1;
__REG32 P3_28  : 1;
__REG32 P3_29  : 1;
__REG32 P3_30  : 1;
__REG32 P3_31  : 1;
} __gpio3_bits;

/* FGPIO 3 Registers*/
typedef union{
  /*FIO3DIR*/
  /*FIO3MASK*/
  /*FIO3PIN*/
  /*FIO3SET*/
  /*FIO3CLR*/
  struct {
    __REG32 P3_0   : 1;
    __REG32 P3_1   : 1;
    __REG32 P3_2   : 1;
    __REG32 P3_3   : 1;
    __REG32 P3_4   : 1;
    __REG32 P3_5   : 1;
    __REG32 P3_6   : 1;
    __REG32 P3_7   : 1;
    __REG32 P3_8   : 1;
    __REG32 P3_9   : 1;
    __REG32 P3_10  : 1;
    __REG32 P3_11  : 1;
    __REG32 P3_12  : 1;
    __REG32 P3_13  : 1;
    __REG32 P3_14  : 1;
    __REG32 P3_15  : 1;
    __REG32 P3_16  : 1;
    __REG32 P3_17  : 1;
    __REG32 P3_18  : 1;
    __REG32 P3_19  : 1;
    __REG32 P3_20  : 1;
    __REG32 P3_21  : 1;
    __REG32 P3_22  : 1;
    __REG32 P3_23  : 1;
    __REG32 P3_24  : 1;
    __REG32 P3_25  : 1;
    __REG32 P3_26  : 1;
    __REG32 P3_27  : 1;
    __REG32 P3_28  : 1;
    __REG32 P3_29  : 1;
    __REG32 P3_30  : 1;
    __REG32 P3_31  : 1;
  };

  struct
  {
    union
    {
      /*FIO3DIR0*/
      /*FIO3MASK0*/
      /*FIO3PIN0*/
      /*FIO3SET0*/
      /*FIO3CLR0*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8  P3_6   : 1;
        __REG8  P3_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*FIO3DIR1*/
      /*FIO3MASK1*/
      /*FIO3PIN1*/
      /*FIO3SET1*/
      /*FIO3CLR1*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8  P3_6   : 1;
        __REG8  P3_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    union
    {
      /*FIO3DIR2*/
      /*FIO3MASK2*/
      /*FIO3PIN2*/
      /*FIO3SET2*/
      /*FIO3CLR2*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8  P3_6   : 1;
        __REG8  P3_7   : 1;
      } __byte2_bit;
      __REG8 __byte2;
    };
    union
    {
      /*FIO3DIR3*/
      /*FIO3MASK3*/
      /*FIO3PIN3*/
      /*FIO3SET3*/
      /*FIO3CLR3*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8  P3_6   : 1;
        __REG8  P3_7   : 1;
      } __byte3_bit;
      __REG8 __byte3;
    };
  };

  struct
  {
    union
    {
      /*FIO3DIRL*/
      /*FIO3MASKL*/
      /*FIO3PINL*/
      /*FIO3SETL*/
      /*FIO3CLRL*/
      struct{
        __REG16 P3_0   : 1;
        __REG16 P3_1   : 1;
        __REG16 P3_2   : 1;
        __REG16 P3_3   : 1;
        __REG16 P3_4   : 1;
        __REG16 P3_5   : 1;
        __REG16 P3_6   : 1;
        __REG16 P3_7   : 1;
        __REG16 P3_8   : 1;
        __REG16 P3_9   : 1;
        __REG16 P3_10  : 1;
        __REG16 P3_11  : 1;
        __REG16 P3_12  : 1;
        __REG16 P3_13  : 1;
        __REG16 P3_14  : 1;
        __REG16 P3_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    union
    {
      /*FIO3DIRU*/
      /*FIO3MASKU*/
      /*FIO3PINU*/
      /*FIO3SETU*/
      /*FIO3CLRU*/
      struct{
        __REG16 P3_0   : 1;
        __REG16 P3_1   : 1;
        __REG16 P3_2   : 1;
        __REG16 P3_3   : 1;
        __REG16 P3_4   : 1;
        __REG16 P3_5   : 1;
        __REG16 P3_6   : 1;
        __REG16 P3_7   : 1;
        __REG16 P3_8   : 1;
        __REG16 P3_9   : 1;
        __REG16 P3_10  : 1;
        __REG16 P3_11  : 1;
        __REG16 P3_12  : 1;
        __REG16 P3_13  : 1;
        __REG16 P3_14  : 1;
        __REG16 P3_15  : 1;
      } __shortu_bit;
      __REG16 __shortu;
    };
  };
} __fgpio3_bits;

/* GPIO 4 Registers */
typedef struct {
__REG32 P4_0   : 1;
__REG32 P4_1   : 1;
__REG32 P4_2   : 1;
__REG32 P4_3   : 1;
__REG32 P4_4   : 1;
__REG32 P4_5   : 1;
__REG32 P4_6   : 1;
__REG32 P4_7   : 1;
__REG32 P4_8   : 1;
__REG32 P4_9   : 1;
__REG32 P4_10  : 1;
__REG32 P4_11  : 1;
__REG32 P4_12  : 1;
__REG32 P4_13  : 1;
__REG32 P4_14  : 1;
__REG32 P4_15  : 1;
__REG32 P4_16  : 1;
__REG32 P4_17  : 1;
__REG32 P4_18  : 1;
__REG32 P4_19  : 1;
__REG32 P4_20  : 1;
__REG32 P4_21  : 1;
__REG32 P4_22  : 1;
__REG32 P4_23  : 1;
__REG32 P4_24  : 1;
__REG32 P4_25  : 1;
__REG32 P4_26  : 1;
__REG32 P4_27  : 1;
__REG32 P4_28  : 1;
__REG32 P4_29  : 1;
__REG32 P4_30  : 1;
__REG32 P4_31  : 1;
} __gpio4_bits;

/* FGPIO 4 Registers*/
typedef union{
  /*FIO4DIR*/
  /*FIO4MASK*/
  /*FIO4PIN*/
  /*FIO4SET*/
  /*FIO4CLR*/
  struct {
    __REG32 P4_0   : 1;
    __REG32 P4_1   : 1;
    __REG32 P4_2   : 1;
    __REG32 P4_3   : 1;
    __REG32 P4_4   : 1;
    __REG32 P4_5   : 1;
    __REG32 P4_6   : 1;
    __REG32 P4_7   : 1;
    __REG32 P4_8   : 1;
    __REG32 P4_9   : 1;
    __REG32 P4_10  : 1;
    __REG32 P4_11  : 1;
    __REG32 P4_12  : 1;
    __REG32 P4_13  : 1;
    __REG32 P4_14  : 1;
    __REG32 P4_15  : 1;
    __REG32 P4_16  : 1;
    __REG32 P4_17  : 1;
    __REG32 P4_18  : 1;
    __REG32 P4_19  : 1;
    __REG32 P4_20  : 1;
    __REG32 P4_21  : 1;
    __REG32 P4_22  : 1;
    __REG32 P4_23  : 1;
    __REG32 P4_24  : 1;
    __REG32 P4_25  : 1;
    __REG32 P4_26  : 1;
    __REG32 P4_27  : 1;
    __REG32 P4_28  : 1;
    __REG32 P4_29  : 1;
    __REG32 P4_30  : 1;
    __REG32 P4_31  : 1;
  };

  struct
  {
    union
    {
      /*FIO4DIR0*/
      /*FIO4MASK0*/
      /*FIO4PIN0*/
      /*FIO4SET0*/
      /*FIO4CLR0*/
      struct{
        __REG8  P4_0   : 1;
        __REG8  P4_1   : 1;
        __REG8  P4_2   : 1;
        __REG8  P4_3   : 1;
        __REG8  P4_4   : 1;
        __REG8  P4_5   : 1;
        __REG8  P4_6   : 1;
        __REG8  P4_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*FIO4DIR1*/
      /*FIO4MASK1*/
      /*FIO4PIN1*/
      /*FIO4SET1*/
      /*FIO4CLR1*/
      struct{
        __REG8  P4_0   : 1;
        __REG8  P4_1   : 1;
        __REG8  P4_2   : 1;
        __REG8  P4_3   : 1;
        __REG8  P4_4   : 1;
        __REG8  P4_5   : 1;
        __REG8  P4_6   : 1;
        __REG8  P4_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    union
    {
      /*FIO4DIR2*/
      /*FIO4MASK2*/
      /*FIO4PIN2*/
      /*FIO4SET2*/
      /*FIO4CLR2*/
      struct{
        __REG8  P4_0   : 1;
        __REG8  P4_1   : 1;
        __REG8  P4_2   : 1;
        __REG8  P4_3   : 1;
        __REG8  P4_4   : 1;
        __REG8  P4_5   : 1;
        __REG8  P4_6   : 1;
        __REG8  P4_7   : 1;
      } __byte2_bit;
      __REG8 __byte2;
    };
    union
    {
      /*FIO4DIR3*/
      /*FIO4MASK3*/
      /*FIO4PIN3*/
      /*FIO4SET3*/
      /*FIO4CLR3*/
      struct{
        __REG8  P4_0   : 1;
        __REG8  P4_1   : 1;
        __REG8  P4_2   : 1;
        __REG8  P4_3   : 1;
        __REG8  P4_4   : 1;
        __REG8  P4_5   : 1;
        __REG8  P4_6   : 1;
        __REG8  P4_7   : 1;
      } __byte3_bit;
      __REG8 __byte3;
    };
  };

  struct
  {
    union
    {
      /*FIO4DIRL*/
      /*FIO4MASKL*/
      /*FIO4PINL*/
      /*FIO4SETL*/
      /*FIO4CLRL*/
      struct{
        __REG16 P4_0   : 1;
        __REG16 P4_1   : 1;
        __REG16 P4_2   : 1;
        __REG16 P4_3   : 1;
        __REG16 P4_4   : 1;
        __REG16 P4_5   : 1;
        __REG16 P4_6   : 1;
        __REG16 P4_7   : 1;
        __REG16 P4_8   : 1;
        __REG16 P4_9   : 1;
        __REG16 P4_10  : 1;
        __REG16 P4_11  : 1;
        __REG16 P4_12  : 1;
        __REG16 P4_13  : 1;
        __REG16 P4_14  : 1;
        __REG16 P4_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    union
    {
      /*FIO4DIRU*/
      /*FIO4MASKU*/
      /*FIO4PINU*/
      /*FIO4SETU*/
      /*FIO4CLRU*/
      struct{
        __REG16 P4_0   : 1;
        __REG16 P4_1   : 1;
        __REG16 P4_2   : 1;
        __REG16 P4_3   : 1;
        __REG16 P4_4   : 1;
        __REG16 P4_5   : 1;
        __REG16 P4_6   : 1;
        __REG16 P4_7   : 1;
        __REG16 P4_8   : 1;
        __REG16 P4_9   : 1;
        __REG16 P4_10  : 1;
        __REG16 P4_11  : 1;
        __REG16 P4_12  : 1;
        __REG16 P4_13  : 1;
        __REG16 P4_14  : 1;
        __REG16 P4_15  : 1;
      } __shortu_bit;
      __REG16 __shortu;
    };
  };
} __fgpio4_bits;

/* FGPIO 5 Registers*/
typedef union{
  /*FIO5DIR*/
  /*FIO5MASK*/
  /*FIO5PIN*/
  /*FIO5SET*/
  /*FIO5CLR*/
  struct {
    __REG32 P5_0   : 1;
    __REG32 P5_1   : 1;
    __REG32 P5_2   : 1;
    __REG32 P5_3   : 1;
    __REG32 P5_4   : 1;
    __REG32        :27;
  };

  struct
  {
    union
    {
      /*FIO5DIR0*/
      /*FIO5MASK0*/
      /*FIO5PIN0*/
      /*FIO5SET0*/
      /*FIO5CLR0*/
      struct{
        __REG8  P5_0   : 1;
        __REG8  P5_1   : 1;
        __REG8  P5_2   : 1;
        __REG8  P5_3   : 1;
        __REG8  P5_4   : 1;
        __REG8         : 3;
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
      /*FIO5DIRL*/
      /*FIO5MASKL*/
      /*FIO5PINL*/
      /*FIO5SETL*/
      /*FIO5CLRL*/
      struct{
        __REG16 P5_0   : 1;
        __REG16 P5_1   : 1;
        __REG16 P5_2   : 1;
        __REG16 P5_3   : 1;
        __REG16 P5_4   : 1;
        __REG16        :11;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16 __shortu;
  };
} __fgpio5_bits;

/* GPIO overall Interrupt Status register */
typedef struct{
__REG32 P0INT  : 1;
__REG32        : 1;
__REG32 P2INT  : 1;
__REG32        :29;
}__iointst_bits;

/* EMC Control Register */
typedef struct {
  __REG32 E         : 1;
  __REG32 M         : 1;
  __REG32 L         : 1;
  __REG32           :29;
} __emc_ctrl_bits;

/* EMC Status Register */
typedef struct {
  __REG32 B         : 1;
  __REG32 S         : 1;
  __REG32 SA        : 1;
  __REG32           :29;
} __emc_st_bits;

/* EMC Configuration Register */
typedef struct {
  __REG32 ENDIAN    : 1;
  __REG32           :31;
} __emc_cfg_bits;

/* Dynamic Memory Control Register */
typedef struct {
  __REG32 CE        : 1;
  __REG32 CS        : 1;
  __REG32 SR        : 1;
  __REG32           : 2;
  __REG32 MMC       : 1;
  __REG32           : 1;
  __REG32 I         : 2;
  __REG32           : 4;
  __REG32 DP        : 1;
  __REG32           :18;
} __emc_dctrl_bits;

/* Dynamic Memory Refresh Timer Register */
typedef struct {
  __REG32 REFRESH   :11;
  __REG32           :21;
} __emc_drfr_bits;

/* Dynamic Memory Read Configuration Register */
typedef struct {
  __REG32 RD        : 2;
  __REG32           :30;
} __emc_drdcfg_bits;

/* Dynamic Memory Percentage Command Period Register */
typedef struct {
  __REG32 tRP       : 4;
  __REG32           :28;
} __emc_drp_bits;

/* Dynamic Memory Active to Precharge Command Period Register */
typedef struct {
  __REG32 tRAS      : 4;
  __REG32           :28;
} __emc_dras_bits;

/* Dynamic Memory Self-refresh Exit Time Register */
typedef struct {
  __REG32 tSREX     : 4;
  __REG32           :28;
} __emc_dsrex_bits;

/* Dynamic Memory Last Data Out to Active Time Register */
typedef struct {
  __REG32 tAPR      : 4;
  __REG32           :28;
} __emc_dapr_bits;

/* Dynamic Memory Data-in to Active Command Time Register */
typedef struct {
  __REG32 tDAL      : 4;
  __REG32           :28;
} __emc_ddal_bits;

/* Dynamic Memory Write Recovery Time Register */
typedef struct {
  __REG32 tWR       : 4;
  __REG32           :28;
} __emc_dwr_bits;

/* Dynamic Memory Active to Active Command Period Register */
typedef struct {
  __REG32 tRC       : 5;
  __REG32           :27;
} __emc_drc_bits;

/* Dynamic Memory Auto-refresh Period Register */
typedef struct {
  __REG32 tRFC      : 5;
  __REG32           :27;
} __emc_drfc_bits;

/* Dynamic Memory Exit Self-refresh Register */
typedef struct {
  __REG32 tXSR      : 5;
  __REG32           :27;
} __emc_dxsr_bits;

/* Dynamic Memory Active Bank A to Active Bank B Time Register */
typedef struct {
  __REG32 tRRD      : 4;
  __REG32           :28;
} __emc_drrd_bits;

/* Dynamic Memory Load Mode Register to Active Command Time */
typedef struct {
  __REG32 tMRD      : 4;
  __REG32           :28;
} __emc_dmrd_bits;

/* Static Memory Extended Wait Register */
typedef struct {
  __REG32 EXTENDEDWAIT  :10;
  __REG32               :22;
} __emc_s_ext_wait_bits;

/* Dynamic Memory Configuration Registers */
typedef struct {
  __REG32           : 3;
  __REG32 MD        : 2;
  __REG32           : 2;
  __REG32 AML       : 6;
  __REG32           : 1;
  __REG32 AMH       : 1;
  __REG32           : 4;
  __REG32 B         : 1;
  __REG32 P         : 1;
  __REG32           :11;
} __emc_d_config_bits;

/* Dynamic Memory RAS & CAS Delay Registers */
typedef struct {
  __REG32 RAS       : 2;
  __REG32           : 6;
  __REG32 CAS       : 2;
  __REG32           :22;
} __emc_d_ras_cas_bits;

/* Static Memory Configuration Registers */
typedef struct {
  __REG32 MW        : 2;
  __REG32           : 1;
  __REG32 PM        : 1;
  __REG32           : 2;
  __REG32 PC        : 1;
  __REG32 PB        : 1;
  __REG32 EW        : 1;
  __REG32           :10;
  __REG32 B         : 1;
  __REG32 P         : 1;
  __REG32           :11;
} __emc_s_config_bits;

/* Static Memory Write Enable Delay Registers */
typedef struct {
  __REG32 WAITWEN   : 4;
  __REG32           :28;
} __emc_s_wait_wen_bits;

/* Static Memory Output Enable Delay Registers */
typedef struct {
  __REG32 WAITOEN   : 4;
  __REG32           :28;
} __emc_s_wait_oen_bits;

/* Static Memory Read Delay Registers */
typedef struct {
  __REG32 WAITRD    : 5;
  __REG32           :27;
} __emc_s_wait_rd_bits;

/* Static Memory Page Mode Read Delay Registers */
typedef struct {
  __REG32 WAITPAGE  : 5;
  __REG32           :27;
} __emc_s_wait_pg_bits;

/* Static Memory Write Delay Registers */
typedef struct {
  __REG32 WAITWR    : 5;
  __REG32           :27;
} __emc_s_wait_wr_bits;

/* Static Memory Turn Round Delay Registers */
typedef struct {
  __REG32 WAITTURN  : 4;
  __REG32           :28;
} __emc_s_wait_turn_bits;

/* EMC Delay Control register bit description (EMCDLYCTL - 0x400F C1DC) */
typedef struct {
  __REG32 CMDDLY      : 5;
  __REG32             : 3;
  __REG32 FBCLKDLY    : 5;
  __REG32             : 3;
  __REG32 CLKOUT0DLY  : 5;
  __REG32             : 3;
  __REG32 CLKOUT1DLY  : 5;
  __REG32             : 3;
} __emcdlyctl_bits;

/* EMC Calibration register (EMCCAL - 0x400F C1E0) */
typedef struct {
  __REG32 CALVALUE    : 8;
  __REG32             : 6;
  __REG32 START       : 1;
  __REG32 DONE        : 1;
  __REG32             :16;
} __emccal_bits;

/* USB - Device Interrupt Status Register */
/* OTG_status and control Register */
typedef union {
/* USBPORTSEL*/
  struct {
  __REG32 PORTSEL           : 2;
  __REG32                   :30;
  };
/* OTGSTCTRL*/
  struct {
__REG32 PORT_FUNC           : 2;
__REG32 TMR_SCALE           : 2;
__REG32 TMR_MODE            : 1;
__REG32 TMR_EN              : 1;
__REG32 TMR_RST             : 1;
__REG32                     : 1;
__REG32 B_HNP_TRACK         : 1;
__REG32 A_HNP_TRACK         : 1;
__REG32 PU_REMOVED          : 1;
__REG32                     : 5;
__REG32 TMR_CNT             :16;
  };
} __usbportsel_bits;

/* USB Clock Control register (USBClkCtrl - 0xFFE0 CFF4) */
/* OTG_clock Registers */
typedef union {
  /* USBCLKCTRL*/
  struct{
__REG32                 : 1;
__REG32 DEV_CLK_EN      : 1;
__REG32                 : 1;
__REG32 PORTSEL_CLK_EN  : 1;
__REG32 AHB_CLK_EN      : 1;
__REG32                 :27;
  };
  /* OTGCLKCTRL*/
  struct{
__REG32 _HOST_CLK_EN  : 1;
__REG32 _DEV_CLK_EN   : 1;
__REG32 _I2C_CLK_EN   : 1;
__REG32 _OTG_CLK_EN   : 1;
__REG32 _AHB_CLK_EN   : 1;
__REG32               :27;
  };
} __usbclkctrl_bits;

/* USB Clock Status register (USBClkSt - 0xFFE0 CFF8) */
/* OTG_status Registers */
typedef union {
  /* USBCLKST*/
  struct{
__REG32                 : 1;
__REG32 DEV_CLK_ON      : 1;
__REG32                 : 1;
__REG32 PORTSEL_CLK_ON  : 1;
__REG32 AHB_CLK_ON      : 1;
__REG32                 :27;
  };
  /* OTGCLKST*/
  struct{
__REG32 _HOST_CLK_ON : 1;
__REG32 _DEV_CLK_ON  : 1;
__REG32 _I2C_CLK_ON  : 1;
__REG32 _OTG_CLK_ON  : 1;
__REG32 _AHB_CLK_ON  : 1;
__REG32              :27;
  };
} __usbclkst_bits;

/* USB - Device Interrupt Status Register */
typedef struct {
  __REG32 USB_INT_REQ_LP    : 1;
  __REG32 USB_INT_REQ_HP    : 1;
  __REG32 USB_INT_REQ_DMA   : 1;
  __REG32 USB_HOST_INT      : 1;
  __REG32 USB_ATX_INT       : 1;
  __REG32 USB_OTG_INT       : 1;
  __REG32 USB_I2C_INT       : 1;
  __REG32                   : 1;
  __REG32 USB_NEED_CLOCK    : 1;
  __REG32                   :22;
  __REG32 EN_USB_INTS       : 1;
} __usbints_bits;

/* USB - Device Interrupt Status Register */
/* USB - Device Interrupt Enable Register */
/* USB - Device Interrupt Clear Register */
/* USB - Device Interrupt Set Register */
typedef struct {
  __REG32 FRAME             : 1;
  __REG32 EP_FAST           : 1;
  __REG32 EP_SLOW           : 1;
  __REG32 DEV_STAT          : 1;
  __REG32 CCEMTY            : 1;
  __REG32 CDFULL            : 1;
  __REG32 RXENDPKT          : 1;
  __REG32 TXENDPKT          : 1;
  __REG32 EP_RLZED          : 1;
  __REG32 ERR_INT           : 1;
  __REG32                   :22;
} __usbdevintst_bits;

/* USB - Device Interrupt Priority Register */
typedef struct {
  __REG8  FRAME             : 1;
  __REG8  EP_FAST           : 1;
  __REG8                    : 6;
} __usbdevintpri_bits;

/* USB - Endpoint Interrupt Status Register */
/* USB - Endpoint Interrupt Enable Register */
/* USB - Endpoint Interrupt Clear Register */
/* USB - Endpoint Interrupt Set Register */
/* USB - Endpoint Interrupt Priority Register */
typedef struct {
  __REG32 EP_0RX            : 1;
  __REG32 EP_0TX            : 1;
  __REG32 EP_1RX            : 1;
  __REG32 EP_1TX            : 1;
  __REG32 EP_2RX            : 1;
  __REG32 EP_2TX            : 1;
  __REG32 EP_3RX            : 1;
  __REG32 EP_3TX            : 1;
  __REG32 EP_4RX            : 1;
  __REG32 EP_4TX            : 1;
  __REG32 EP_5RX            : 1;
  __REG32 EP_5TX            : 1;
  __REG32 EP_6RX            : 1;
  __REG32 EP_6TX            : 1;
  __REG32 EP_7RX            : 1;
  __REG32 EP_7TX            : 1;
  __REG32 EP_8RX            : 1;
  __REG32 EP_8TX            : 1;
  __REG32 EP_9RX            : 1;
  __REG32 EP_9TX            : 1;
  __REG32 EP_10RX           : 1;
  __REG32 EP_10TX           : 1;
  __REG32 EP_11RX           : 1;
  __REG32 EP_11TX           : 1;
  __REG32 EP_12RX           : 1;
  __REG32 EP_12TX           : 1;
  __REG32 EP_13RX           : 1;
  __REG32 EP_13TX           : 1;
  __REG32 EP_14RX           : 1;
  __REG32 EP_14TX           : 1;
  __REG32 EP_15RX           : 1;
  __REG32 EP_15TX           : 1;
} __usbepintst_bits;

/* USB - Realize Enpoint Register */
/* USB - DMA Request Status Register */
/* USB - DMA Request Clear Register */
/* USB - DMA Request Set Regiser */
/* USB - EP DMA Status Register */
/* USB - EP DMA Enable Register */
/* USB - EP DMA Disable Register */
/* USB - New DD Request Interrupt Status Register */
/* USB - New DD Request Interrupt Clear Register */
/* USB - New DD Request Interrupt Set Register */
/* USB - End Of Transfer Interrupt Status Register */
/* USB - End Of Transfer Interrupt Clear Register */
/* USB - End Of Transfer Interrupt Set Register */
/* USB - System Error Interrupt Status Register */
/* USB - System Error Interrupt Clear Register */
/* USB - System Error Interrupt Set Register */
typedef struct {
  __REG32 EP0               : 1;
  __REG32 EP1               : 1;
  __REG32 EP2               : 1;
  __REG32 EP3               : 1;
  __REG32 EP4               : 1;
  __REG32 EP5               : 1;
  __REG32 EP6               : 1;
  __REG32 EP7               : 1;
  __REG32 EP8               : 1;
  __REG32 EP9               : 1;
  __REG32 EP10              : 1;
  __REG32 EP11              : 1;
  __REG32 EP12              : 1;
  __REG32 EP13              : 1;
  __REG32 EP14              : 1;
  __REG32 EP15              : 1;
  __REG32 EP16              : 1;
  __REG32 EP17              : 1;
  __REG32 EP18              : 1;
  __REG32 EP19              : 1;
  __REG32 EP20              : 1;
  __REG32 EP21              : 1;
  __REG32 EP22              : 1;
  __REG32 EP23              : 1;
  __REG32 EP24              : 1;
  __REG32 EP25              : 1;
  __REG32 EP26              : 1;
  __REG32 EP27              : 1;
  __REG32 EP28              : 1;
  __REG32 EP29              : 1;
  __REG32 EP30              : 1;
  __REG32 EP31              : 1;
} __usbreep_bits;

/* USB - Endpoint Index Register */
typedef struct {
  __REG32 PHY_ENDP          : 5;
  __REG32                   :27;
} __usbepin_bits;

/* USB - MaxPacketSize Register */
typedef struct {
  __REG32 MPS               :10;
  __REG32                   :22;
} __usbmaxpsize_bits;

/* USB - Receive Packet Length Register */
typedef struct {
  __REG32 PKT_LNGTH         :10;
  __REG32 DV                : 1;
  __REG32 PKT_RDY           : 1;
  __REG32                   :20;
} __usbrxplen_bits;

/* USB - Transmit Packet Length Register */
typedef struct {
  __REG32 PKT_LNGHT         :10;
  __REG32                   :22;
} __usbtxplen_bits;

/* USB - Control Register */
typedef struct {
  __REG32 RD_EN             : 1;
  __REG32 WR_EN             : 1;
  __REG32 LOG_ENDPOINT      : 4;
  __REG32                   :26;
} __usbctrl_bits;

/* USB - Command Code Register */
typedef struct {
  __REG32                   : 8;
  __REG32 CMD_PHASE         : 8;
  __REG32 CMD_CODE          : 8;
  __REG32                   : 8;
} __usbcmdcode_bits;

/* USB - Command Data Register */
typedef struct {
  __REG32 CMD_DATA          : 8;
  __REG32                   :24;
} __usbcmddata_bits;

/* USB - DMA Interrupt Status Register */
/* USB - DMA Interrupt Enable Register */
typedef struct {
  __REG32 EOT       : 1;
  __REG32 NDDR      : 1;
  __REG32 ERR       : 1;
  __REG32           :29;
} __usbdmaintst_bits;

/* HcRevision Register */
typedef struct {
  __REG32 REV               : 8;
  __REG32                   :24;
} __hcrevision_bits;

/* HcControl Register */
typedef struct {
  __REG32 CBSR              : 2;
  __REG32 PLE               : 1;
  __REG32 IE                : 1;
  __REG32 CLE               : 1;
  __REG32 BLE               : 1;
  __REG32 HCFS              : 2;
  __REG32 IR                : 1;
  __REG32 RWC               : 1;
  __REG32 RWE               : 1;
  __REG32                   :21;
} __hccontrol_bits;

/* HcCommandStatus Register */
typedef struct {
  __REG32 HCR               : 1;
  __REG32 CLF               : 1;
  __REG32 BLF               : 1;
  __REG32 OCR               : 1;
  __REG32                   :12;
  __REG32 SOC               : 2;
  __REG32                   :14;
} __hccommandstatus_bits;

/* HcInterruptStatus Register */
typedef struct {
  __REG32 SO                : 1;
  __REG32 WDH               : 1;
  __REG32 SF                : 1;
  __REG32 RD                : 1;
  __REG32 UE                : 1;
  __REG32 FNO               : 1;
  __REG32 RHSC              : 1;
  __REG32                   :23;
  __REG32 OC                : 1;
  __REG32                   : 1;
} __hcinterruptstatus_bits;

/* HcInterruptEnable Register
   HcInterruptDisable Register */
typedef struct {
  __REG32 SO                : 1;
  __REG32 WDH               : 1;
  __REG32 SF                : 1;
  __REG32 RD                : 1;
  __REG32 UE                : 1;
  __REG32 FNO               : 1;
  __REG32 RHSC              : 1;
  __REG32                   :23;
  __REG32 OC                : 1;
  __REG32 MIE               : 1;
} __hcinterruptenable_bits;

/* HcHCCA Register */
typedef struct {
  __REG32                   : 8;
  __REG32 HCCA              :24;
} __hchcca_bits;

/* HcPeriodCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 PCED              :28;
} __hcperiodcurrented_bits;

/* HcControlHeadED Registerr */
typedef struct {
  __REG32                   : 4;
  __REG32 CHED              :28;
} __hccontrolheaded_bits;

/* HcControlCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 CCED              :28;
} __hccontrolcurrented_bits;

/* HcBulkHeadED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 BHED              :28;
} __hcbulkheaded_bits;

/* HcBulkCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 BCED              :28;
} __hcbulkcurrented_bits;

/* HcDoneHead Register */
typedef struct {
  __REG32                   : 4;
  __REG32 DH                :28;
} __hcdonehead_bits;

/* HcFmInterval Register */
typedef struct {
  __REG32 FI                :14;
  __REG32                   : 2;
  __REG32 FSMPS             :15;
  __REG32 FIT               : 1;
} __hcfminterval_bits;

/* HcFmRemaining Register */
typedef struct {
  __REG32 FR                :14;
  __REG32                   :17;
  __REG32 FRT               : 1;
} __hcfmremaining_bits;

/* HcFmNumber Register */
typedef struct {
  __REG32 FN                :16;
  __REG32                   :16;
} __hcfmnumber_bits;

/* HcPeriodicStart Register */
typedef struct {
  __REG32 PS                :14;
  __REG32                   :18;
} __hcperiodicstart_bits;

/* HcLSThreshold Register */
typedef struct {
  __REG32 LST               :12;
  __REG32                   :20;
} __hclsthreshold_bits;

/* HcRhDescriptorA Register */
typedef struct {
  __REG32 NDP               : 8;
  __REG32 PSM               : 1;  /* ??*/
  __REG32 NPS               : 1;  /* ??*/
  __REG32 DT                : 1;
  __REG32 OCPM              : 1;
  __REG32 NOCP              : 1;
  __REG32                   :11;
  __REG32 POTPGT            : 8;
} __hcrhdescriptora_bits;

/* HcRhDescriptorB Register */
typedef struct {
  __REG32 DR                :16;
  __REG32 PPCM              :16;
} __hcrhdescriptorb_bits;

/* HcRhStatus Register */
typedef struct {
  __REG32 LPS               : 1;
  __REG32 OCI               : 1;
  __REG32                   :13;
  __REG32 DRWE              : 1;
  __REG32 LPSC              : 1;
  __REG32 CCIC              : 1;
  __REG32                   :13;
  __REG32 CRWE              : 1;
} __hcrhstatus_bits;

/* HcRhPortStatus[1:2] Register */
typedef struct {
  __REG32 CCS               : 1;
  __REG32 PES               : 1;
  __REG32 PSS               : 1;
  __REG32 POCI              : 1;
  __REG32 PRS               : 1;
  __REG32                   : 3;
  __REG32 PPS               : 1;
  __REG32 LSDA              : 1;
  __REG32                   : 6;
  __REG32 CSC               : 1;
  __REG32 PESC              : 1;
  __REG32 PSSC              : 1;
  __REG32 OCIC              : 1;
  __REG32 PRSC              : 1;
  __REG32                   :11;
} __hcrhportstatus_bits;

/* OTG_int_status Register
   OTG_int_enable Register
   OTG_int_set Register
   OTG_int_clr Register */
typedef struct{
__REG32 TMR                 : 1;
__REG32 REMOVE_PU           : 1;
__REG32 HNP_FAILURE         : 1;
__REG32 HNP_SUCCESS         : 1;
__REG32                     :28;
} __otgintst_bits;

/* OTG Timer Register */
typedef struct{
__REG32 TIMEOUT_CNT         :16;
__REG32                     :16;
} __otgtmr_bits;

/* OTG I2C Clock High Register */
typedef struct{
__REG32 CDHI                : 8;
__REG32                     :24;
} __i2c_clkhi_bits;

/* OTG I2C Clock Low Register */
typedef struct{
__REG32 CDLO                : 8;
__REG32                     :24;
} __i2c_clklo_bits;

/* OTG I2C_TX/I2C_RX Register */
typedef union{
  /*I2C_RX*/
  struct {
__REG32 RX_DATA  : 8;
__REG32          :24;
  };
  /*I2C_TX*/
  struct {
__REG32 TX_DATA  : 8;
__REG32 START    : 1;
__REG32 STOP     : 1;
__REG32          :22;
  };
} __otg_i2c_rx_tx_bits;

/* OTG I2C_STS Register */
typedef struct{
__REG32 TDI     : 1;
__REG32 AFI     : 1;
__REG32 NAI     : 1;
__REG32 DRMI    : 1;
__REG32 DRSI    : 1;
__REG32 ACTIVE  : 1;
__REG32 SCL     : 1;
__REG32 SDA     : 1;
__REG32 RFF     : 1;
__REG32 RFE     : 1;
__REG32 TFF     : 1;
__REG32 TFE     : 1;
__REG32         :20;
} __otg_i2c_sts_bits;

/* OTG I2C_CTL Register */
typedef struct{
__REG32 TDIE    : 1;
__REG32 AFIE    : 1;
__REG32 NAIE    : 1;
__REG32 DRMIE   : 1;
__REG32 DRSIE   : 1;
__REG32 RFFIE   : 1;
__REG32 RFDAIE  : 1;
__REG32 TFFIE   : 1;
__REG32 SRST    : 1;
__REG32         :23;
} __otg_i2c_ctl_bits;

/* SD/MMC Power control register */
typedef struct{
__REG32 CTRL       : 2;
__REG32            : 4;
__REG32 OPENDRAIN  : 1;
__REG32 ROD        : 1;
__REG32            :24;
} __mcipower_bits;

/* SD/MMC Clock control register */
typedef struct{
__REG32 CLKDIV   : 8;
__REG32 ENABLE   : 1;
__REG32 PWRSAVE  : 1;
__REG32 BYPASS   : 1;
__REG32 WIDEBUS  : 1;
__REG32          :20;
} __mciclock_bits;

/* SD/MMC Command register */
typedef struct{
__REG32 CMDINDEX   : 6;
__REG32 RESPONSE   : 1;
__REG32 LONGRSP    : 1;
__REG32 INTERRUPT  : 1;
__REG32 PENDING    : 1;
__REG32 ENABLE     : 1;
__REG32            :21;
} __mcicommand_bits;

/* SD/MMC Command response register */
typedef struct{
__REG32 RESPCMD  : 6;
__REG32          :26;
} __mcirespcmd_bits;

/* SD/MMC Data control register */
typedef struct{
__REG32 ENABLE     : 1;
__REG32 DIRECTION  : 1;
__REG32 MODE       : 1;
__REG32 DMAENABLE  : 1;
__REG32 BLOCKSIZE  : 4;
__REG32            :24;
} __mcidatactrl_bits;

/* SD/MMC Status register */
typedef struct{
__REG32 CMDCRCFAIL       : 1;
__REG32 DATACRCFAIL      : 1;
__REG32 CMDTIMEOUT       : 1;
__REG32 DATATIMEOUT      : 1;
__REG32 TXUNDERRUN       : 1;
__REG32 RXOVERRUN        : 1;
__REG32 CMDRESPEND       : 1;
__REG32 CMDSENT          : 1;
__REG32 DATAEND          : 1;
__REG32 STARTBITERR      : 1;
__REG32 DATABLOCKEND     : 1;
__REG32 CMDACTIVE        : 1;
__REG32 TXACTIVE         : 1;
__REG32 RXACTIVE         : 1;
__REG32 TXFIFOHALFEMPTY  : 1;
__REG32 RXFIFOHALFFULL   : 1;
__REG32 TXFIFOFULL       : 1;
__REG32 RXFIFOFULL       : 1;
__REG32 TXFIFOEMPTY      : 1;
__REG32 RXFIFOEMPTY      : 1;
__REG32 TXDATAAVLBL      : 1;
__REG32 RXDATAAVLBL      : 1;
__REG32                  :10;
} __mcistatus_bits;

/* SD/MMC Clear register */
typedef struct{
__REG32 CMDCRCFAILCLR    : 1;
__REG32 DATACRCFAILCLR   : 1;
__REG32 CMDTIMEOUTCLR    : 1;
__REG32 DATATIMEOUTCLR   : 1;
__REG32 TXUNDERRUNCLR    : 1;
__REG32 RXOVERRUNCLR     : 1;
__REG32 CMDRESPENDCLR    : 1;
__REG32 CMDSENTCLR       : 1;
__REG32 DATAENDCLR       : 1;
__REG32 STARTBITERRCLR   : 1;
__REG32 DATABLOCKENDCLR  : 1;
__REG32                  :21;
} __mciclear_bits;

/* SD/MMC FIFO counter register */
typedef struct{
__REG32 DATACOUNT  :15;
__REG32            :17;
} __mcififocnt_bits;

/* UART interrupt enable register */
typedef struct{
__REG32 RDAIE     : 1;
__REG32 THREIE    : 1;
__REG32 RXLSIE    : 1;
__REG32           : 5;
__REG32 ABEOINTEN : 1;
__REG32 ABTOINTEN : 1;
__REG32           :22;
} __uartier0_bits;

/* UART1 interrupt enable register */
typedef struct{
__REG32 RDAIE     : 1;
__REG32 THREIE    : 1;
__REG32 RXLSIE    : 1;
__REG32 RXMSIE    : 1;
__REG32           : 3;
__REG32 CTSIE     : 1;
__REG32 ABEOINTEN : 1;
__REG32 ABTOINTEN : 1;
__REG32           :22;
} __uartier1_bits;

/* UART Transmit Enable Register */
typedef struct{
__REG8        : 7;
__REG8  TXEN  : 1;
} __uartter_bits;

/* UART line status register */
typedef struct{
__REG8  DR    : 1;
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
__REG32 DMA    : 1;
__REG32        : 2;
__REG32 RTLS   : 2;
__REG32        :24;
  };
} __uartfcriir_bits;

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

/* UART RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32            : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __uars485ctrl_bits;

/* UART1 RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32 SEL        : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __uars1485ctrl_bits;

/* UART Transmit Enable Register */
typedef struct{
  __REG32 TXEN            : 1;
  __REG32                 :31;
} __uart4ter_bits;

/* UART4 Oversampling Register (U4OSR ) */
typedef struct{
  __REG32                 : 1;
  __REG32 OSFRAC          : 3;
  __REG32 OSINT           : 4;
  __REG32 FDINT           : 7;
  __REG32                 :17;
} __uartosr_bits;

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

/* CAN acceptance filter mode register */
typedef struct {
  __REG32 ACCOFF          :1;
  __REG32 ACCBP           :1;
  __REG32 EFCAN           :1;
  __REG32                 :29;
} __afmr_bits;

/* CAN LUT Error Register */
typedef struct {
  __REG32 LUTERR          :1;
  __REG32                 :31;
} __luterr_bits;

/* Global FullCANInterrupt Enable register */
typedef struct {
  __REG32 FCANIE          :1;
  __REG32                 :31;
} __fcanie_bits;

/* FullCAN Interrupt and Capture registers 0 */
typedef struct {
  __REG32 INTPND0         :1;
  __REG32 INTPND1         :1;
  __REG32 INTPND2         :1;
  __REG32 INTPND3         :1;
  __REG32 INTPND4         :1;
  __REG32 INTPND5         :1;
  __REG32 INTPND6         :1;
  __REG32 INTPND7         :1;
  __REG32 INTPND8         :1;
  __REG32 INTPND9         :1;
  __REG32 INTPND10        :1;
  __REG32 INTPND11        :1;
  __REG32 INTPND12        :1;
  __REG32 INTPND13        :1;
  __REG32 INTPND14        :1;
  __REG32 INTPND15        :1;
  __REG32 INTPND16        :1;
  __REG32 INTPND17        :1;
  __REG32 INTPND18        :1;
  __REG32 INTPND19        :1;
  __REG32 INTPND20        :1;
  __REG32 INTPND21        :1;
  __REG32 INTPND22        :1;
  __REG32 INTPND23        :1;
  __REG32 INTPND24        :1;
  __REG32 INTPND25        :1;
  __REG32 INTPND26        :1;
  __REG32 INTPND27        :1;
  __REG32 INTPND28        :1;
  __REG32 INTPND29        :1;
  __REG32 INTPND30        :1;
  __REG32 INTPND31        :1;
} __fcanic0_bits;

/* FullCAN Interrupt and Capture registers 1 */
typedef struct {
  __REG32 INTPND32        :1;
  __REG32 INTPND33        :1;
  __REG32 INTPND34        :1;
  __REG32 INTPND35        :1;
  __REG32 INTPND36        :1;
  __REG32 INTPND37        :1;
  __REG32 INTPND38        :1;
  __REG32 INTPND39        :1;
  __REG32 INTPND40        :1;
  __REG32 INTPND41        :1;
  __REG32 INTPND42        :1;
  __REG32 INTPND43        :1;
  __REG32 INTPND44        :1;
  __REG32 INTPND45        :1;
  __REG32 INTPND46        :1;
  __REG32 INTPND47        :1;
  __REG32 INTPND48        :1;
  __REG32 INTPND49        :1;
  __REG32 INTPND50        :1;
  __REG32 INTPND51        :1;
  __REG32 INTPND52        :1;
  __REG32 INTPND53        :1;
  __REG32 INTPND54        :1;
  __REG32 INTPND55        :1;
  __REG32 INTPND56        :1;
  __REG32 INTPND57        :1;
  __REG32 INTPND58        :1;
  __REG32 INTPND59        :1;
  __REG32 INTPND60        :1;
  __REG32 INTPND61        :1;
  __REG32 INTPND62        :1;
  __REG32 INTPND63        :1;
} __fcanic1_bits;

/* CAN central transmit status register */
typedef struct {
  __REG32 TS1             : 1;
  __REG32 TS2             : 1;
  __REG32                 : 6;
  __REG32 TBS1            : 1;
  __REG32 TBS2            : 1;
  __REG32                 : 6;
  __REG32 TCS1            : 1;
  __REG32 TCS2            : 1;
  __REG32                 :14;
} __cantxsr_bits;

/* CAN central receive status register */
typedef struct {
  __REG32 RS1             : 1;
  __REG32 RS2             : 1;
  __REG32                 : 6;
  __REG32 RBS1            : 1;
  __REG32 RBS2            : 1;
  __REG32                 : 6;
  __REG32 DOS1            : 1;
  __REG32 DOS2            : 1;
  __REG32                 :14;
} __canrxsr_bits;

/* CAN miscellaneous status register */
typedef struct {
  __REG32 E1              : 1;
  __REG32 E2              : 1;
  __REG32                 : 6;
  __REG32 BS1             : 1;
  __REG32 BS2             : 1;
  __REG32                 :22;
} __canmsr_bits;

/* CAN mode register */
typedef struct {
  __REG32 RM              :1;
  __REG32 LOM             :1;
  __REG32 STM             :1;
  __REG32 TPM             :1;
  __REG32 SM              :1;
  __REG32 RPM             :1;
  __REG32                 :1;
  __REG32 TM              :1;
  __REG32                 :24;
} __canmod_bits;

/* CAN command register */
typedef struct {
  __REG32 TR              :1;
  __REG32 AT              :1;
  __REG32 RRB             :1;
  __REG32 CDO             :1;
  __REG32 SRR             :1;
  __REG32 STB1            :1;
  __REG32 STB2            :1;
  __REG32 STB3            :1;
  __REG32                 :24;
} __cancmr_bits;

/* CAN global status register */
typedef struct {
  __REG32 RBS              :1;
  __REG32 DOS              :1;
  __REG32 TBS              :1;
  __REG32 TCS              :1;
  __REG32 RS               :1;
  __REG32 TS               :1;
  __REG32 ES               :1;
  __REG32 BS               :1;
  __REG32                  :8;
  __REG32 RXERR            :8;
  __REG32 TXERR            :8;
} __cangsr_bits;

/* CAN interrupt capture register */
typedef struct {
  __REG32 RI               :1;
  __REG32 TI1              :1;
  __REG32 EI               :1;
  __REG32 DOI              :1;
  __REG32 WUI              :1;
  __REG32 EPI              :1;
  __REG32 ALI              :1;
  __REG32 BEI              :1;
  __REG32 IDI              :1;
  __REG32 TI2              :1;
  __REG32 TI3              :1;
  __REG32                  :5;
  __REG32 ERRBIT           :5;
  __REG32 ERRDIR           :1;
  __REG32 ERRC             :2;
  __REG32 ALCBIT           :8;
} __canicr_bits;

/* CAN interrupt enable register */
typedef struct {
  __REG32 RIE               :1;
  __REG32 TIE1              :1;
  __REG32 EIE               :1;
  __REG32 DOIE              :1;
  __REG32 WUIE              :1;
  __REG32 EPIE              :1;
  __REG32 ALIE              :1;
  __REG32 BEIE              :1;
  __REG32 IDIE              :1;
  __REG32 TIE2              :1;
  __REG32 TIE3              :1;
  __REG32                   :21;
} __canier_bits;

/* CAN bus timing register */
typedef struct {
  __REG32 BRP                :10;
  __REG32                    :4;
  __REG32 SJW                :2;
  __REG32 TSEG1              :4;
  __REG32 TSEG2              :3;
  __REG32 SAM                :1;
  __REG32                    :8;
} __canbtr_bits;

/* CAN error warning limit register */
typedef struct {
  __REG32 EWL                :8;
  __REG32                    :24;
} __canewl_bits;

/* CAN status register */
typedef struct {
  __REG32 RBS                :1;
  __REG32 DOS                :1;
  __REG32 TBS1               :1;
  __REG32 TCS1               :1;
  __REG32 RS                 :1;
  __REG32 TS1                :1;
  __REG32 ES                 :1;
  __REG32 BS                 :1;
  __REG32 /*RBS*/            :1;
  __REG32 /*DOS*/            :1;
  __REG32 TBS2               :1;
  __REG32 TCS2               :1;
  __REG32 /*RS*/             :1;
  __REG32 TS2                :1;
  __REG32 /*ES*/             :1;
  __REG32 /*BS*/             :1;
  __REG32 /*RBS*/            :1;
  __REG32 /*DOS*/            :1;
  __REG32 TBS3               :1;
  __REG32 TCS3               :1;
  __REG32 /*RS*/             :1;
  __REG32 TS3                :1;
  __REG32 /*ES*/             :1;
  __REG32 /*BS*/             :1;
  __REG32                    :8;
} __cansr_bits;

/* CAN rx frame status register */
typedef struct {
  __REG32 IDINDEX            :10;
  __REG32 BP                 :1;
  __REG32                    :5;
  __REG32 DLC                :4;
  __REG32                    :10;
  __REG32 RTR                :1;
  __REG32 FF                 :1;
} __canrfs_bits;

/* CAN rx identifier register */
typedef union {
  /*CANxRID*/
  struct {
   __REG32 ID10_0             :11;
   __REG32                    :21;
  };
  /*CANxRID*/
  struct {
   __REG32 ID29_18            :11;
   __REG32                    :21;
  };
  /*CANxRID*/
  struct {
   __REG32 ID29_0             :29;
   __REG32                    :3;
  };
} __canrid_bits;

/* CAN rx data register A */
typedef struct {
  __REG32 DATA1               :8;
  __REG32 DATA2               :8;
  __REG32 DATA3               :8;
  __REG32 DATA4               :8;
} __canrda_bits;

/* CAN rx data register B */
typedef struct {
  __REG32 DATA5               :8;
  __REG32 DATA6               :8;
  __REG32 DATA7               :8;
  __REG32 DATA8               :8;
} __canrdb_bits;

/* CAN tx frame information register */
typedef struct {
  __REG32 PRIO              :8;
  __REG32                   :8;
  __REG32 DLC               :4;
  __REG32                   :10;
  __REG32 RTR               :1;
  __REG32 FF                :1;
} __cantfi_bits;

/* CAN tx identifier register */
typedef union {
  /*CANxTIDy*/
  struct {
   __REG32 ID10_0             :11;
   __REG32                    :21;
  };
  /*CANxTIDy*/
  struct {
   __REG32 ID29_18            :11;
   __REG32                    :21;
  };
  /*CANxTIDy*/
  struct {
   __REG32 ID29_0             :29;
   __REG32                    :3;
  };
} __cantid_bits;

/* CAN tx data register A */
typedef struct {
  __REG32 DATA1               :8;
  __REG32 DATA2               :8;
  __REG32 DATA3               :8;
  __REG32 DATA4               :8;
} __cantda_bits;

/* CAN tx data register B */
typedef struct {
  __REG32 DATA5               :8;
  __REG32 DATA6               :8;
  __REG32 DATA7               :8;
  __REG32 DATA8               :8;
} __cantdb_bits;

/* CAN Sleep Clear register */
typedef struct {
  __REG32                     :1;
  __REG32 CAN1SLEEP           :1;
  __REG32 CAN2SLEEP           :1;
  __REG32                     :29;
} __cansleepclr_bits;

/* CAN Wakeup Flags register */
typedef struct {
  __REG32                     :1;
  __REG32 CAN1WAKE            :1;
  __REG32 CAN2WAKE            :1;
  __REG32                     :29;
} __canwakeflags_bits;

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

/* SSP DMA Control Register */
typedef struct{
__REG32 RXDMAE : 1;
__REG32 TXDMAE : 1;
__REG32        :30;
} __sspdmacr_bits;

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

/* I2S Digital Audio Output Registes */
typedef struct{
__REG32 WORS_WIDTH    : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32 MUTE          : 1;
__REG32               :16;
} __i2sdao_bits;

/* I2S Digital Audio Input Register */
typedef struct{
__REG32 WORS_WIDTH    : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32               :17;
} __i2sdai_bits;

/* I2S Status Feedback Register */
typedef struct{
__REG32 IRQ           : 1;
__REG32 DMAREQ1       : 1;
__REG32 DMAREQ2       : 1;
__REG32               : 5;
__REG32 RX_LEVEL      : 4;
__REG32               : 4;
__REG32 TX_LEVEL      : 4;
__REG32               :12;
} __i2sstate_bits;

/* I2S DMA Configuration Register */
typedef struct{
__REG32 RX_DMA_EN     : 1;
__REG32 TX_DMA_EN     : 1;
__REG32               : 6;
__REG32 RX_DEPTH_DMA  : 4;
__REG32               : 4;
__REG32 TX_DEPTH_DMA  : 4;
__REG32               :12;
} __i2sdma_bits;

/* I2S Interrupt Request Control register */
typedef struct{
__REG32 RX_IRQ_EN     : 1;
__REG32 TX_IRQ_EN     : 1;
__REG32               : 6;
__REG32 RX_DEPTH_IRQ  : 4;
__REG32               : 4;
__REG32 TX_DEPTH_IRQ  : 4;
__REG32               :12;
} __i2sirq_bits;

/* I2S Transmit Clock Rate Register */
typedef struct{
__REG32 Y_DIVIDER     : 8;
__REG32 X_DIVIDER     : 8;
__REG32               :16;
} __i2stxrate_bits;

/* Transmit Clock Rate register */
typedef struct{
__REG32 TX_BITRATE    : 6;
__REG32               :26;
} __i2stxbitrate_bits;

/* Receive Clock Rate register */
typedef struct{
__REG32 RX_BITRATE    : 6;
__REG32               :26;
} __i2srxbitrate_bits;

/* Transmit Mode Control register */
typedef struct{
__REG32 TXCLKSEL      : 2;
__REG32 TX4PIN        : 1;
__REG32 TXMCENA       : 1;
__REG32               :28;
} __i2stxmode_bits;

/* Receive Mode Control register */
typedef struct{
__REG32 RXCLKSEL      : 2;
__REG32 RX4PIN        : 1;
__REG32 RXMCENA       : 1;
__REG32               :28;
} __i2srxmode_bits;

/* TIMER interrupt register */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32 CR1INT  : 1;
__REG32         :26;
} __ir_bits;

/* TIMER control register */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tcr_bits;

/* TIMER count control register */
typedef struct{
__REG32 CTM : 2;     /*Counter/Timer Mode*/
__REG32 CIS : 2;     /*Count Input Select*/
__REG32     :28;
} __ctcr_bits;

/* TIMER match control register */
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

/* TIMER capture control register */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32 CAP1RE   : 1;
__REG32 CAP1FE   : 1;
__REG32 CAP1I    : 1;
__REG32          :26;
} __tccr_bits;

/* TIMER external match register */
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

/* PWM0 interrupt register */
typedef struct{
__REG32 PWMMR0I : 1;
__REG32 PWMMR1I : 1;
__REG32 PWMMR2I : 1;
__REG32 PWMMR3I : 1;
__REG32 PWMCAP0 : 1;
__REG32         : 3;
__REG32 PWMMR4I : 1;
__REG32 PWMMR5I : 1;
__REG32 PWMMR6I : 1;
__REG32         :21;
} __pwmir0_bits;

/* PWM interrupt register */
typedef struct{
__REG32 PWMMR0I : 1;
__REG32 PWMMR1I : 1;
__REG32 PWMMR2I : 1;
__REG32 PWMMR3I : 1;
__REG32 PWMCAP0 : 1;
__REG32 PWMCAP1 : 1;
__REG32         : 2;
__REG32 PWMMR4I : 1;
__REG32 PWMMR5I : 1;
__REG32 PWMMR6I : 1;
__REG32         :21;
} __pwmir1_bits;

/* PWM0 timer control register */
typedef struct{
__REG32 CE     : 1;
__REG32 CR     : 1;
__REG32        : 1;
__REG32 PWMEN  : 1;
__REG32 MD     : 1;
__REG32        :27;
} __pwmtcr0_bits;

/* PWM1 timer control register */
typedef struct{
__REG32 CE     : 1;
__REG32 CR     : 1;
__REG32        : 1;
__REG32 PWMEN  : 1;
__REG32        :28;
} __pwmtcr1_bits;

/* PWM Count Control Register */
typedef struct{
__REG32 CM     : 2;
__REG32 CIS    : 2;
__REG32        :28;
} __pwmctcr_bits;

/* PWM match control register */
typedef struct{
__REG32 PWMMR0I  : 1;
__REG32 PWMMR0R  : 1;
__REG32 PWMMR0S  : 1;
__REG32 PWMMR1I  : 1;
__REG32 PWMMR1R  : 1;
__REG32 PWMMR1S  : 1;
__REG32 PWMMR2I  : 1;
__REG32 PWMMR2R  : 1;
__REG32 PWMMR2S  : 1;
__REG32 PWMMR3I  : 1;
__REG32 PWMMR3R  : 1;
__REG32 PWMMR3S  : 1;
__REG32 PWMMR4I  : 1;
__REG32 PWMMR4R  : 1;
__REG32 PWMMR4S  : 1;
__REG32 PWMMR5I  : 1;
__REG32 PWMMR5R  : 1;
__REG32 PWMMR5S  : 1;
__REG32 PWMMR6I  : 1;
__REG32 PWMMR6R  : 1;
__REG32 PWMMR6S  : 1;
__REG32          :11;
} __pwmmcr_bits;

/* PWM0 Capture Control Register */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0INT  : 1;
__REG32          :29;
} __pwmccr0_bits;

/* PWM1 Capture Control Register */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0INT  : 1;
__REG32 CAP1RE   : 1;
__REG32 CAP1FE   : 1;
__REG32 CAP1INT  : 1;
__REG32          :26;
} __pwmccr1_bits;

/* PWM  control register */
typedef struct{
__REG32         : 2;
__REG32 PWMSEL2 : 1;
__REG32 PWMSEL3 : 1;
__REG32 PWMSEL4 : 1;
__REG32 PWMSEL5 : 1;
__REG32 PWMSEL6 : 1;
__REG32         : 2;
__REG32 PWMENA1 : 1;
__REG32 PWMENA2 : 1;
__REG32 PWMENA3 : 1;
__REG32 PWMENA4 : 1;
__REG32 PWMENA5 : 1;
__REG32 PWMENA6 : 1;
__REG32         :17;
} __pwmpcr_bits;

/* PWM latch enable register */
typedef struct{
__REG32 EM0L  : 1;
__REG32 EM1L  : 1;
__REG32 EM2L  : 1;
__REG32 EM3L  : 1;
__REG32 EM4L  : 1;
__REG32 EM5L  : 1;
__REG32 EM6L  : 1;
__REG32       :25;
} __pwmler_bits;

/* MCPWM Control read address */
typedef struct{
__REG32 RUN0    : 1;
__REG32 CENTER0 : 1;
__REG32 POLA0   : 1;
__REG32 DTE0    : 1;
__REG32 DISUP0  : 1;
__REG32         : 3;
__REG32 RUN1    : 1;
__REG32 CENTER1 : 1;
__REG32 POLA1   : 1;
__REG32 DTE1    : 1;
__REG32 DISUP1  : 1;
__REG32         : 3;
__REG32 RUN2    : 1;
__REG32 CENTER2 : 1;
__REG32 POLA2   : 1;
__REG32 DTE2    : 1;
__REG32 DISUP2  : 1;
__REG32         : 8;
__REG32 INVBDC  : 1;
__REG32 ACMODE  : 1;
__REG32 DCMODE  : 1;
} __mccon_bits;

/* Capture control register */
typedef struct{
__REG32 CAP0MCI0_RE : 1;
__REG32 CAP0MCI0_FE : 1;
__REG32 CAP0MCI1_RE : 1;
__REG32 CAP0MCI1_FE : 1;
__REG32 CAP0MCI2_RE : 1;
__REG32 CAP0MCI2_FE : 1;
__REG32 CAP1MCI0_RE : 1;
__REG32 CAP1MCI0_FE : 1;
__REG32 CAP1MCI1_RE : 1;
__REG32 CAP1MCI1_FE : 1;
__REG32 CAP1MCI2_RE : 1;
__REG32 CAP1MCI2_FE : 1;
__REG32 CAP2MCI0_RE : 1;
__REG32 CAP2MCI0_FE : 1;
__REG32 CAP2MCI1_RE : 1;
__REG32 CAP2MCI1_FE : 1;
__REG32 CAP2MCI2_RE : 1;
__REG32 CAP2MCI2_FE : 1;
__REG32 RT0         : 1;
__REG32 RT1         : 1;
__REG32 RT2         : 1;
__REG32 HNFCAP0     : 1;
__REG32 HNFCAP1     : 1;
__REG32 HNFCAP2     : 1;
__REG32             : 8;
} __mccapcon_bits;

/* PWM interrupt enable register */
typedef struct{
__REG32 ILIM0       : 1;
__REG32 IMAT0       : 1;
__REG32 ICAP0       : 1;
__REG32             : 1;
__REG32 ILIM1       : 1;
__REG32 IMAT1       : 1;
__REG32 ICAP1       : 1;
__REG32             : 1;
__REG32 ILIM2       : 1;
__REG32 IMAT2       : 1;
__REG32 ICAP2       : 1;
__REG32             : 4;
__REG32 ABORT       : 1;
__REG32             :16;
} __mcinten_bits;

/* Count control register */
typedef struct{
__REG32 TC0MCFB0_RE  : 1;
__REG32 TC0MCFB0_FE  : 1;
__REG32 TC0MCFB1_RE  : 1;
__REG32 TC0MCFB1_FE  : 1;
__REG32 TC0MCFB2_RE  : 1;
__REG32 TC0MCFB2_FE  : 1;
__REG32 TC1MCFB0_RE  : 1;
__REG32 TC1MCFB0_FE  : 1;
__REG32 TC1MCFB1_RE  : 1;
__REG32 TC1MCFB1_FE  : 1;
__REG32 TC1MCFB2_RE  : 1;
__REG32 TC1MCFB2_FE  : 1;
__REG32 TC2MCFB0_RE  : 1;
__REG32 TC2MCFB0_FE  : 1;
__REG32 TC2MCFB1_RE  : 1;
__REG32 TC2MCFB1_FE  : 1;
__REG32 TC2MCFB2_RE  : 1;
__REG32 TC2MCFB2_FE  : 1;
__REG32              :11;
__REG32 CNTR0        : 1;
__REG32 CNTR1        : 1;
__REG32 CNTR2        : 1;
} __mccntcon_bits;

/* Dead-time register */
typedef struct{
__REG32 DT0         :10;
__REG32 DT1         :10;
__REG32 DT2         :10;
__REG32             : 2;
} __mcdt_bits;

/* Current communication pattern register */
typedef struct{
__REG32 CCPA0       : 1;
__REG32 CCPB0       : 1;
__REG32 CCPA1       : 1;
__REG32 CCPB1       : 1;
__REG32 CCPA2       : 1;
__REG32 CCPB2       : 1;
__REG32             :26;
} __mcccp_bits;

/* Capture clear register */
typedef struct{
__REG32 CAP_CLR0    : 1;
__REG32 CAP_CLR1    : 1;
__REG32 CAP_CLR2    : 1;
__REG32             :29;
} __mccap_clr_bits;

/* QEI Control register */
typedef struct{
__REG32 RESP        : 1;
__REG32 RESPI       : 1;
__REG32 RESV        : 1;
__REG32 RESI        : 1;
__REG32             :28;
} __qeicon_bits;

/* QEI Configuration register */
typedef struct{
__REG32 DIRINV      : 1;
__REG32 SIGMODE     : 1;
__REG32 CAPMODE     : 1;
__REG32 INVINX      : 1;
__REG32 CRESPI      : 1;
__REG32             :11;
__REG32 INXGATE     : 4;
__REG32             :12;
} __qeiconf_bits;

/* QEI Status register */
typedef struct{
__REG32 DIR         : 1;
__REG32             :31;
} __qeistat_bits;

/* QEI Interrupt Status register */
/* QEI Interrupt Set register */
/* QEI Interrupt Clear register */
typedef struct{
__REG32 INX_INT     : 1;
__REG32 TIM_INT     : 1;
__REG32 VELC_INT    : 1;
__REG32 DIR_INT     : 1;
__REG32 ERR_INT     : 1;
__REG32 ENCLK_INT   : 1;
__REG32 POS0_INT    : 1;
__REG32 POS1_INT    : 1;
__REG32 POS2_INT    : 1;
__REG32 REV0_INT    : 1;
__REG32 POS0REV_INT : 1;
__REG32 POS1REV_INT : 1;
__REG32 POS2REV_INT : 1;
__REG32 REV1_INT    : 1;
__REG32 REV2_INT    : 1;
__REG32 MAXPOS_INT  : 1;
__REG32             :16;
} __qeiintstat_bits;

/* RTC interrupt location register */
typedef struct{
__REG32 RTCCIF  : 1;
__REG32 RTCALF  : 1;
__REG32         :30;
} __ilr_bits;

/* RTC clock control register */
typedef struct{
__REG32 CLKEN   : 1;
__REG32 CTCRST  : 1;
__REG32         : 2;
__REG32 CCALEN  : 1;
__REG32         :27;
} __rtcccr_bits;

/* RTC counter increment interrupt register */
typedef struct{
__REG32 IMSEC   : 1;
__REG32 IMMIN   : 1;
__REG32 IMHOUR  : 1;
__REG32 IMDOM   : 1;
__REG32 IMDOW   : 1;
__REG32 IMDOY   : 1;
__REG32 IMMON   : 1;
__REG32 IMYEAR  : 1;
__REG32         :24;
} __ciir_bits;

/* RTC Counter Increment Select Mask Register */
typedef struct{
__REG32 CALVAL    :17;
__REG32 CALDIR    : 1;
__REG32           :14;
} __calibration_bits;

/* RTC alarm mask register */
typedef struct{
__REG32 AMRSEC   : 1;
__REG32 AMRMIN   : 1;
__REG32 AMRHOUR  : 1;
__REG32 AMRDOM   : 1;
__REG32 AMRDOW   : 1;
__REG32 AMRDOY   : 1;
__REG32 AMRMON   : 1;
__REG32 AMRYEAR  : 1;
__REG32          :24;
} __amr_bits;

/* RTC Auxiliary control register */
typedef struct{
__REG32           : 4;
__REG32 RTC_OSCF  : 1;
__REG32           : 1;
__REG32 RTC_PDOUT : 1;
__REG32           :25;
} __rtcaux_bits;

/* RTC Auxiliary Enable register */
typedef struct{
__REG32           : 4;
__REG32 RTC_OSCFEN: 1;
__REG32           :27;
} __rtcauxen_bits;

/* RTC consolidated time register 0 */
typedef struct{
__REG32 SEC   : 6;
__REG32       : 2;
__REG32 MIN   : 6;
__REG32       : 2;
__REG32 HOUR  : 5;
__REG32       : 3;
__REG32 DOW   : 3;
__REG32       : 5;
} __ctime0_bits;

/* RTC consolidated time register 1 */
typedef struct{
__REG32 DOM   : 5;
__REG32       : 3;
__REG32 MON   : 4;
__REG32       : 4;
__REG32 YEAR  :12;
__REG32       : 4;
} __ctime1_bits;

/* RTC consolidated time register 2 */
typedef struct{
__REG32 DOY  :12;
__REG32      :20;
} __ctime2_bits;

/* RTC second register */
typedef struct{
__REG32 SEC  : 6;
__REG32      :26;
} __sec_bits;

/* RTC minute register */
typedef struct{
__REG32 MIN  : 6;
__REG32      :26;
} __min_bits;

/* RTC hour register */
typedef struct{
__REG32 HOUR  : 5;
__REG32       :27;
} __hour_bits;

/* RTC day of month register */
typedef struct{
__REG32 DOM  : 5;
__REG32      :27;
} __dom_bits;

/* RTC day of week register */
typedef struct{
__REG32 DOW  : 3;
__REG32      :29;
} __dow_bits;

/* RTC day of year register */
typedef struct{
__REG32 DOY  : 9;
__REG32      :23;
} __doy_bits;

/* RTC month register */
typedef struct{
__REG32 MON  : 4;
__REG32      :28;
} __month_bits;

/* RTC year register */
typedef struct{
__REG32 YEAR :12;
__REG32      :20;
} __year_bits;

/* Watchdog mode register */
typedef struct{
__REG32 WDEN     : 1;
__REG32 WDRESET  : 1;
__REG32 WDTOF    : 1;
__REG32 WDINT    : 1;
__REG32 WDPROTECT: 1;
__REG32          :27;
} __wdmod_bits;

/* Watchdog Timer Constant register */
typedef struct{
__REG32 Count    :24;
__REG32          : 8;
} __wdtc_bits;

/* Watchdog feed register */
typedef struct{
__REG32 FEED  : 8;
__REG32       :24;
} __wdfeed_bits;

/* Watchdog Timer Warning Interrupt register */
typedef struct{
__REG32 WARNINT :10;
__REG32         :22;
} __wdwarnint_bits;

/* Watchdog Timer Window register */
typedef struct{
__REG32 WINDOW  :24;
__REG32         : 8;
} __wdwindow_bits;

/* A/D Control Register */
typedef struct{
__REG32 SEL     : 8;
__REG32 CLKDIV  : 8;
__REG32 BURST   : 1;
__REG32         : 4;
__REG32 PDN     : 1;
__REG32         : 2;
__REG32 START   : 3;
__REG32 EDGE    : 1;
__REG32         : 4;
} __adcr_bits;

/* A/D Global Data Register */
typedef struct{
__REG32         : 4;
__REG32 RESULT  :12;
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
__REG32 RESULT  :12;
__REG32         :14;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __addr_bits;

/* A/D Trim register */
typedef struct{
__REG32         : 4;
__REG32 ADCOFFS : 4;
__REG32 TRIM    : 4;
__REG32         :20;
} __adtrm_bits;

/* D/A Converter Register */
typedef struct{
__REG32        : 6;
__REG32 VALUE  :10;
__REG32 BIAS   : 1;
__REG32        :15;
} __dacr_bits;

/* D/A Converter Control register */
typedef struct{
__REG32 INT_DMA_REQ : 1;
__REG32 DBLBUF_ENA  : 1;
__REG32 CNT_ENA     : 1;
__REG32 DMA_ENA     : 1;
__REG32             :28;
} __dacctrl_bits;

/* D/A Converter Counter Value register */
typedef struct{
__REG32 VALUE       :16;
__REG32             :16;
} __daccntval_bits;

/* DMA Interrupt Status Register */
typedef struct{
__REG32 INTSTATUS0  : 1;
__REG32 INTSTATUS1  : 1;
__REG32 INTSTATUS2  : 1;
__REG32 INTSTATUS3  : 1;
__REG32 INTSTATUS4  : 1;
__REG32 INTSTATUS5  : 1;
__REG32 INTSTATUS6  : 1;
__REG32 INTSTATUS7  : 1;
__REG32             :24;
} __dmacintstatus_bits;

/* DMA Interrupt Terminal Count Request Status Register */
typedef struct{
__REG32 INTTCSTATUS0  : 1;
__REG32 INTTCSTATUS1  : 1;
__REG32 INTTCSTATUS2  : 1;
__REG32 INTTCSTATUS3  : 1;
__REG32 INTTCSTATUS4  : 1;
__REG32 INTTCSTATUS5  : 1;
__REG32 INTTCSTATUS6  : 1;
__REG32 INTTCSTATUS7  : 1;
__REG32               :24;
} __dmacinttcstatus_bits;

/* DMA Interrupt Terminal Count Request Clear Register */
typedef struct{
__REG32 INTTCCLEAR0   : 1;
__REG32 INTTCCLEAR1   : 1;
__REG32 INTTCCLEAR2   : 1;
__REG32 INTTCCLEAR3   : 1;
__REG32 INTTCCLEAR4   : 1;
__REG32 INTTCCLEAR5   : 1;
__REG32 INTTCCLEAR6   : 1;
__REG32 INTTCCLEAR7   : 1;
__REG32               :24;
} __dmacinttcclear_bits;

/* DMA Interrupt Error Status Register */
typedef struct{
__REG32 INTERRORSTATUS0 : 1;
__REG32 INTERRORSTATUS1 : 1;
__REG32 INTERRORSTATUS2 : 1;
__REG32 INTERRORSTATUS3 : 1;
__REG32 INTERRORSTATUS4 : 1;
__REG32 INTERRORSTATUS5 : 1;
__REG32 INTERRORSTATUS6 : 1;
__REG32 INTERRORSTATUS7 : 1;
__REG32                 :24;
} __dmacinterrstat_bits;

/* DMA Interrupt Error Clear Register */
typedef struct{
__REG32 INTERRCLR0      : 1;
__REG32 INTERRCLR1      : 1;
__REG32 INTERRCLR2      : 1;
__REG32 INTERRCLR3      : 1;
__REG32 INTERRCLR4      : 1;
__REG32 INTERRCLR5      : 1;
__REG32 INTERRCLR6      : 1;
__REG32 INTERRCLR7      : 1;
__REG32                 :24;
} __dmacinterrclr_bits;

/* DMA Raw Interrupt Terminal Count Status Register */
typedef struct{
__REG32 RAWINTTCSTATUS0 : 1;
__REG32 RAWINTTCSTATUS1 : 1;
__REG32 RAWINTTCSTATUS2 : 1;
__REG32 RAWINTTCSTATUS3 : 1;
__REG32 RAWINTTCSTATUS4 : 1;
__REG32 RAWINTTCSTATUS5 : 1;
__REG32 RAWINTTCSTATUS6 : 1;
__REG32 RAWINTTCSTATUS7 : 1;
__REG32                 :24;
} __dmacrawinttcstatus_bits;

/* DMA Raw Error Interrupt Status Register */
typedef struct{
__REG32 RAWINTERRORSTATUS0  : 1;
__REG32 RAWINTERRORSTATUS1  : 1;
__REG32 RAWINTERRORSTATUS2  : 1;
__REG32 RAWINTERRORSTATUS3  : 1;
__REG32 RAWINTERRORSTATUS4  : 1;
__REG32 RAWINTERRORSTATUS5  : 1;
__REG32 RAWINTERRORSTATUS6  : 1;
__REG32 RAWINTERRORSTATUS7  : 1;
__REG32                     :24;
} __dmacrawinterrorstatus_bits;

/* DMA Enabled Channel Register */
typedef struct{
__REG32 ENABLEDCHANNELS0  : 1;
__REG32 ENABLEDCHANNELS1  : 1;
__REG32 ENABLEDCHANNELS2  : 1;
__REG32 ENABLEDCHANNELS3  : 1;
__REG32 ENABLEDCHANNELS4  : 1;
__REG32 ENABLEDCHANNELS5  : 1;
__REG32 ENABLEDCHANNELS6  : 1;
__REG32 ENABLEDCHANNELS7  : 1;
__REG32                   :24;
} __dmacenbldchns_bits;

/* DMA Software Burst Request Register */
typedef struct{
__REG32 SoftBReq0         : 1;
__REG32 SoftBReq1         : 1;
__REG32 SoftBReq2         : 1;
__REG32 SoftBReq3         : 1;
__REG32 SoftBReq4         : 1;
__REG32 SoftBReq5         : 1;
__REG32 SoftBReq6         : 1;
__REG32 SoftBReq7         : 1;
__REG32 SoftBReq8         : 1;
__REG32 SoftBReq9         : 1;
__REG32 SoftBReq10        : 1;
__REG32 SoftBReq11        : 1;
__REG32 SoftBReq12        : 1;
__REG32 SoftBReq13        : 1;
__REG32 SoftBReq14        : 1;
__REG32 SoftBReq15        : 1;
__REG32                   :16;
} __dmacsoftbreq_bits;

/* DMA Software Single Request Register */
typedef struct{
__REG32 SoftSReq0         : 1;
__REG32 SoftSReq1         : 1;
__REG32 SoftSReq2         : 1;
__REG32 SoftSReq3         : 1;
__REG32 SoftSReq4         : 1;
__REG32 SoftSReq5         : 1;
__REG32 SoftSReq6         : 1;
__REG32 SoftSReq7         : 1;
__REG32 SoftSReq8         : 1;
__REG32 SoftSReq9         : 1;
__REG32 SoftSReq10        : 1;
__REG32 SoftSReq11        : 1;
__REG32 SoftSReq12        : 1;
__REG32 SoftSReq13        : 1;
__REG32 SoftSReq14        : 1;
__REG32 SoftSReq15        : 1;
__REG32                   :16;
} __dmacsoftsreq_bits;

/* DMA Software Last Burst Request Register */
typedef struct{
__REG32 SoftLBReq0         : 1;
__REG32 SoftLBReq1         : 1;
__REG32 SoftLBReq2         : 1;
__REG32 SoftLBReq3         : 1;
__REG32 SoftLBReq4         : 1;
__REG32 SoftLBReq5         : 1;
__REG32 SoftLBReq6         : 1;
__REG32 SoftLBReq7         : 1;
__REG32 SoftLBReq8         : 1;
__REG32 SoftLBReq9         : 1;
__REG32 SoftLBReq10        : 1;
__REG32 SoftLBReq11        : 1;
__REG32 SoftLBReq12        : 1;
__REG32 SoftLBReq13        : 1;
__REG32 SoftLBReq14        : 1;
__REG32 SoftLBReq15        : 1;
__REG32                    :16;
} __dmacsoftlbreq_bits;

/* DMA Software Last Single Request Register */
typedef struct{
__REG32 SoftLSReq0          : 1;
__REG32 SoftLSReq1          : 1;
__REG32 SoftLSReq2          : 1;
__REG32 SoftLSReq3          : 1;
__REG32 SoftLSReq4          : 1;
__REG32 SoftLSReq5          : 1;
__REG32 SoftLSReq6          : 1;
__REG32 SoftLSReq7          : 1;
__REG32 SoftLSReq8          : 1;
__REG32 SoftLSReq9          : 1;
__REG32 SoftLSReq10         : 1;
__REG32 SoftLSReq11         : 1;
__REG32 SoftLSReq12         : 1;
__REG32 SoftLSReq13         : 1;
__REG32 SoftLSReq14         : 1;
__REG32 SoftLSReq15         : 1;
__REG32                     :16;
} __dmacsoftlsreq_bits;

/* DMA Synchronization Register */
typedef struct{
__REG32 DMACSYNC0   : 1;
__REG32 DMACSYNC1   : 1;
__REG32 DMACSYNC2   : 1;
__REG32 DMACSYNC3   : 1;
__REG32 DMACSYNC4   : 1;
__REG32 DMACSYNC5   : 1;
__REG32 DMACSYNC6   : 1;
__REG32 DMACSYNC7   : 1;
__REG32 DMACSYNC8   : 1;
__REG32 DMACSYNC9   : 1;
__REG32 DMACSYNC10  : 1;
__REG32 DMACSYNC11  : 1;
__REG32 DMACSYNC12  : 1;
__REG32 DMACSYNC13  : 1;
__REG32 DMACSYNC14  : 1;
__REG32 DMACSYNC15  : 1;
__REG32             :16;
} __dmacsync_bits;

/* DMA Configuration Register */
typedef struct{
__REG32 E   : 1;
__REG32 M   : 1;
__REG32     :30;
} __dmacconfig_bits;

/* DMA Request Select register */
typedef struct{
__REG32 DMASEL00  : 1;
__REG32 DMASEL01  : 1;
__REG32 DMASEL02  : 1;
__REG32 DMASEL03  : 1;
__REG32 DMASEL04  : 1;
__REG32 DMASEL05  : 1;
__REG32 DMASEL06  : 1;
__REG32 DMASEL07  : 1;
__REG32           : 2;
__REG32 DMASEL10  : 1;
__REG32 DMASEL11  : 1;
__REG32 DMASEL12  : 1;
__REG32 DMASEL13  : 1;
__REG32 DMASEL14  : 1;
__REG32 DMASEL15  : 1;
__REG32           :16;
} __dmareqsel_bits;

/* DMA Channel Control Registers */
typedef struct{
__REG32 TRANSFERSIZE  :12;
__REG32 SBSIZE        : 3;
__REG32 DBSIZE        : 3;
__REG32 SWIDTH        : 3;
__REG32 DWIDTH        : 3;
__REG32               : 2;
__REG32 SI            : 1;
__REG32 DI            : 1;
__REG32 PROT1         : 1;
__REG32 PROT2         : 1;
__REG32 PROT3         : 1;
__REG32 I             : 1;
} __dma_ctrl_bits;

/* DMA Channel Configuration Registers */
typedef struct{
__REG32 E               : 1;
__REG32 SRCPERIPHERAL   : 5;
__REG32 DESTPERIPHERAL  : 5;
__REG32 FLOWCNTRL       : 3;
__REG32 IE              : 1;
__REG32 ITC             : 1;
__REG32 L               : 1;
__REG32 A               : 1;
__REG32 H               : 1;
__REG32                 :13;
} __dma_cfg_bits;

/* CRC mode register */
typedef struct{
__REG32 CRC_POLY    : 2;
__REG32 BIT_RVS_WR  : 1;
__REG32 CMPL_WR     : 1;
__REG32 BIT_RVS_SUM : 1;
__REG32 CMPL_SUM    : 1;
__REG32             :26;
} __crc_mode_bits;

/* EEPROM command register */
typedef struct{
__REG32 CMD         : 3;
__REG32 RDPREFETCH  : 1;
__REG32             :28;
} __eecmd_bits;

/* EEPROM address register */
typedef struct{
__REG32 ADDR        :12;
__REG32             :20;
} __eeaddr_bits;

/* EEPROM wait state register */
typedef struct{
__REG32 PHASE3      : 8;
__REG32 PHASE2      : 8;
__REG32 PHASE1      : 8;
__REG32             : 8;
} __eewstate_bits;

/* EEPROM clock divider register */
typedef struct{
__REG32 CLKDIV      :16;
__REG32             :16;
} __eeclkdiv_bits;

/* EEPROM power down register */
typedef struct{
__REG32 PWRDWN      : 1;
__REG32             :31;
} __eepwrdwn_bits;

/* Interrupt enable register */
typedef struct{
__REG32             :26;
__REG32 EE_RW_DONE  : 1;
__REG32             : 1;
__REG32 EE_PROG_DONE: 1;
__REG32             : 3;
} __eeinten_bits;

/* Interrupt enable register */
typedef struct{
__REG32             :26;
__REG32 RDWR_CLR_EN : 1;
__REG32             : 1;
__REG32 PROG1_CLR_EN: 1;
__REG32             : 3;
} __eeintenclr_bits;

/* Interrupt enable set register */
typedef struct{
__REG32             :26;
__REG32 RDWR_SET_EN : 1;
__REG32             : 1;
__REG32 PROG1_SET_EN: 1;
__REG32             : 3;
} __eeintenset_bits;

/* Interrupt enable set register */
/* Flash module Status register */
typedef union {
/*EEINTSTAT*/
struct{
__REG32             :26;
__REG32 END_OF_RDWR : 1;
__REG32             : 1;
__REG32 END_OF_PROG1: 1;
__REG32             : 3;
};
/*FMSTAT*/
struct{
__REG32             : 2;
__REG32 SIG_DONE    : 1;
__REG32             :29;
};
} __eeintstat_bits;

/* Interrupt status clear register */
/* Flash Module Status Clear register */
typedef union {
/*EEINTSTATCLR*/
struct{
__REG32             :26;
__REG32 RDWR_CLR_ST : 1;
__REG32             : 1;
__REG32 PROG1_CLR_ST: 1;
__REG32             : 3;
};
/*FMSTATCLR*/
struct{
__REG32               : 2;
__REG32 SIG_DONE_CLR  : 1;
__REG32               :29;
};
} __eeintstatclr_bits;

/* Interrupt status set */
typedef struct{
__REG32             :26;
__REG32 RDWR_SET_ST : 1;
__REG32             : 1;
__REG32 PROG1_SET_ST: 1;
__REG32             : 3;
} __eeintstatset_bits;

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

/* Memory mapping control register */
typedef struct{
__REG32 MAP         : 1;
__REG32             :31;
} __memmap_bits;

/* Event Monitor/Recorder Control Register */
typedef struct{
__REG32 INTWAKE_EN0   : 1;
__REG32 GPCLEAR_EN0   : 1;
__REG32 POL0          : 1;
__REG32 EV0_INPUT_EN  : 1;
__REG32               : 6;
__REG32 INTWAKE_EN1   : 1;
__REG32 GPCLEAR_EN1   : 1;
__REG32 POL1          : 1;
__REG32 EV1_INPUT_EN  : 1;
__REG32               : 6;
__REG32 INTWAKE_EN2   : 1;
__REG32 GPCLEAR_EN2   : 1;
__REG32 POL2          : 1;
__REG32 EV2_INPUT_EN  : 1;
__REG32               : 6;
__REG32 ERMODE        : 2;
} __ercontrol_bits;

/* Event Monitor/Recorder Status Register */
typedef struct{
__REG32 EV0           : 1;
__REG32 EV1           : 1;
__REG32 EV2           : 1;
__REG32 GP_CLEARED    : 1;
__REG32               :27;
__REG32 WAKEUP        : 1;
} __erstatus_bits;

/* Event Monitor/Recorder Counters Register */
typedef struct{
__REG32 COUNTER0      : 3;
__REG32               : 5;
__REG32 COUNTER1      : 3;
__REG32               : 5;
__REG32 COUNTER2      : 3;
__REG32               :13;
} __ercounters_bits;

/* Event Monitor/Recorder First Stamp Register */
/* Event Monitor/Recorder Last Stamp Register  */
typedef struct{
__REG32 SEC           : 6;
__REG32 MIN           : 6;
__REG32 HOUR          : 5;
__REG32 DOY           : 9;
__REG32               : 6;
} __erfirststamp_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                  0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
#define STCTRL      SYSTICKCSR
#define STCTRL_bit  SYSTICKCSR_bit
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
#define STRELOAD      SYSTICKRVR
#define STRELOAD_bit  SYSTICKRVR_bit
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
#define STCURR      SYSTICKCVR
#define STCURR_bit  SYSTICKCVR_bit
__IO_REG32_BIT(SYSTICKCALVR,          0xE000E01C,__READ       ,__systickcalvr_bits);
#define STCALIB      SYSTICKCALVR
#define STCALIB_bit  SYSTICKCALVR_bit
__IO_REG32_BIT(SETENA0,               0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(SETENA1,               0xE000E104,__READ_WRITE ,__setena1_bits);
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,               0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,              0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,              0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(ACTIVE0,               0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,               0xE000E304,__READ       ,__active1_bits);
__IO_REG32_BIT(IP0,                   0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                   0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                   0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                   0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                   0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                   0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                   0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                   0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(IP8,                   0xE000E420,__READ_WRITE ,__pri8_bits);
__IO_REG32_BIT(IP9,                   0xE000E424,__READ_WRITE ,__pri9_bits);
__IO_REG32_BIT(IP10,                  0xE000E428,__READ_WRITE ,__pri10_bits);
__IO_REG32_BIT(IP11,                  0xE000E42C,__READ_WRITE ,__pri11_bits);
__IO_REG32_BIT(IP12,                  0xE000E430,__READ_WRITE ,__pri12_bits);
__IO_REG32_BIT(IP13,                  0xE000E434,__READ_WRITE ,__pri13_bits);
__IO_REG32_BIT(IP14,                  0xE000E438,__READ_WRITE ,__pri14_bits);
__IO_REG32_BIT(IP15,                  0xE000E43C,__READ_WRITE ,__pri15_bits);
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
__IO_REG32_BIT(MEMMAP,                0x400FC040,__READ_WRITE ,__memmap_bits);
__IO_REG32_BIT(EXTINT,                0x400FC140,__READ_WRITE ,__extint_bits);
__IO_REG32_BIT(EXTMODE,               0x400FC148,__READ_WRITE ,__extmode_bits);
__IO_REG32_BIT(EXTPOLAR,              0x400FC14C,__READ_WRITE ,__extpolar_bits);
__IO_REG32_BIT(RSID,                  0x400FC180,__READ_WRITE ,__rsid_bits);
__IO_REG32_BIT(SCS,                   0x400FC1A0,__READ_WRITE ,__scs_bits);
__IO_REG32_BIT(CLKSRCSEL,             0x400FC10C,__READ_WRITE ,__clksrcsel_bits);
__IO_REG32_BIT(PLL0CON,               0x400FC080,__READ_WRITE ,__pllcon_bits);
__IO_REG32_BIT(PLL0CFG,               0x400FC084,__READ_WRITE ,__pllcfg_bits);
__IO_REG32_BIT(PLL0STAT,              0x400FC088,__READ       ,__pllstat_bits);
__IO_REG32_BIT(PLL0FEED,              0x400FC08C,__WRITE      ,__pllfeed_bits);
__IO_REG32_BIT(PLL1CON,               0x400FC0A0,__READ_WRITE ,__pllcon_bits);
__IO_REG32_BIT(PLL1CFG,               0x400FC0A4,__READ_WRITE ,__pllcfg_bits);
__IO_REG32_BIT(PLL1STAT,              0x400FC0A8,__READ       ,__pllstat_bits);
__IO_REG32_BIT(PLL1FEED,              0x400FC0AC,__WRITE      ,__pllfeed_bits);
__IO_REG32_BIT(EMCCLKSEL,             0x400FC100,__READ_WRITE ,__emcclksel_bits);
__IO_REG32_BIT(CCLKSEL,               0x400FC104,__READ_WRITE ,__cclksel_bits);
__IO_REG32_BIT(USBCLKSEL,             0x400FC108,__READ_WRITE ,__usbclksel_bits);
__IO_REG32_BIT(PCLKSEL,               0x400FC1A8,__READ_WRITE ,__pclksel_bits);
__IO_REG32_BIT(PCON,                  0x400FC0C0,__READ_WRITE ,__pcon_bits);
__IO_REG32_BIT(PCONP,                 0x400FC0C4,__READ_WRITE ,__pconp_bits);
__IO_REG32_BIT(PBOOST,                0x400FC1B0,__READ_WRITE ,__pboost_bits);
__IO_REG32_BIT(CLKOUTCFG,             0x400FC1C8,__READ_WRITE ,__clkoutcfg_bits);
__IO_REG32_BIT(RSTCON0,               0x400FC1CC,__READ_WRITE ,__rstcon0_bits);
__IO_REG32_BIT(RSTCON1,               0x400FC1D0,__READ_WRITE ,__rstcon1_bits);

/***************************************************************************
 **
 ** Flash Accelerator
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASHCFG,              0x400FC000,__READ_WRITE ,__flashcfg_bits);

/***************************************************************************
 **
 ** Pin connect block
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCON_P0_00,           0x4002C000,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_01,           0x4002C004,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_02,           0x4002C008,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_03,           0x4002C00C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_04,           0x4002C010,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_05,           0x4002C014,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_06,           0x4002C018,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_07,           0x4002C01C,__READ_WRITE ,__iocon_w_bits);
__IO_REG32_BIT(IOCON_P0_08,           0x4002C020,__READ_WRITE ,__iocon_w_bits);
__IO_REG32_BIT(IOCON_P0_09,           0x4002C024,__READ_WRITE ,__iocon_w_bits);
__IO_REG32_BIT(IOCON_P0_10,           0x4002C028,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_11,           0x4002C02C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_12,           0x4002C030,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P0_13,           0x4002C034,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P0_14,           0x4002C038,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_15,           0x4002C03C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_16,           0x4002C040,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_17,           0x4002C044,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_18,           0x4002C048,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_19,           0x4002C04C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_20,           0x4002C050,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_21,           0x4002C054,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_22,           0x4002C058,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P0_23,           0x4002C05C,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P0_24,           0x4002C060,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P0_25,           0x4002C064,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P0_26,           0x4002C068,__READ_WRITE ,__iocon_p0_26_a_bits);
__IO_REG32_BIT(IOCON_P0_27,           0x4002C06C,__READ_WRITE ,__iocon_i_bits);
__IO_REG32_BIT(IOCON_P0_28,           0x4002C070,__READ_WRITE ,__iocon_i_bits);
__IO_REG32_BIT(IOCON_P0_29,           0x4002C074,__READ_WRITE ,__iocon_u_bits);
__IO_REG32_BIT(IOCON_P0_30,           0x4002C078,__READ_WRITE ,__iocon_u_bits);
__IO_REG32_BIT(IOCON_P0_31,           0x4002C07C,__READ_WRITE ,__iocon_u_bits);
__IO_REG32_BIT(IOCON_P1_00,           0x4002C080,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_01,           0x4002C084,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_02,           0x4002C088,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_03,           0x4002C08C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_04,           0x4002C090,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_05,           0x4002C094,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_06,           0x4002C098,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_07,           0x4002C09C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_08,           0x4002C0A0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_09,           0x4002C0A4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_10,           0x4002C0A8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_11,           0x4002C0AC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_12,           0x4002C0B0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_13,           0x4002C0B4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_14,           0x4002C0B8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_15,           0x4002C0BC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_16,           0x4002C0C0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_17,           0x4002C0C4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_18,           0x4002C0C8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_19,           0x4002C0CC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_20,           0x4002C0D0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_21,           0x4002C0D4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_22,           0x4002C0D8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_23,           0x4002C0DC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_24,           0x4002C0E0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_25,           0x4002C0E4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_26,           0x4002C0E8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_27,           0x4002C0EC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_28,           0x4002C0F0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_29,           0x4002C0F4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P1_30,           0x4002C0F8,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P1_31,           0x4002C0FC,__READ_WRITE ,__iocon_a_bits);
__IO_REG32_BIT(IOCON_P2_00,           0x4002C100,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_01,           0x4002C104,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_02,           0x4002C108,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_03,           0x4002C10C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_04,           0x4002C110,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_05,           0x4002C114,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_06,           0x4002C118,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_07,           0x4002C11C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_08,           0x4002C120,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_09,           0x4002C124,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_10,           0x4002C128,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_11,           0x4002C12C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_12,           0x4002C130,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_13,           0x4002C134,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_14,           0x4002C138,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_15,           0x4002C13C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_16,           0x4002C140,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_17,           0x4002C144,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_18,           0x4002C148,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_19,           0x4002C14C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_20,           0x4002C150,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_21,           0x4002C154,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_22,           0x4002C158,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_23,           0x4002C15C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_24,           0x4002C160,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_25,           0x4002C164,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_26,           0x4002C168,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_27,           0x4002C16C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_28,           0x4002C170,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_29,           0x4002C174,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_30,           0x4002C178,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P2_31,           0x4002C17C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_00,           0x4002C180,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_01,           0x4002C184,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_02,           0x4002C188,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_03,           0x4002C18C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_04,           0x4002C190,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_05,           0x4002C194,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_06,           0x4002C198,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_07,           0x4002C19C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_08,           0x4002C1A0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_09,           0x4002C1A4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_10,           0x4002C1A8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_11,           0x4002C1AC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_12,           0x4002C1B0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_13,           0x4002C1B4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_14,           0x4002C1B8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_15,           0x4002C1BC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_16,           0x4002C1C0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_17,           0x4002C1C4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_18,           0x4002C1C8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_19,           0x4002C1CC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_20,           0x4002C1D0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_21,           0x4002C1D4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_22,           0x4002C1D8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_23,           0x4002C1DC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_24,           0x4002C1E0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_25,           0x4002C1E4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_26,           0x4002C1E8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_27,           0x4002C1EC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_28,           0x4002C1F0,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_29,           0x4002C1F4,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_30,           0x4002C1F8,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P3_31,           0x4002C1FC,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_00,           0x4002C200,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_01,           0x4002C204,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_02,           0x4002C208,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_03,           0x4002C20C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_04,           0x4002C210,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_05,           0x4002C214,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_06,           0x4002C218,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_07,           0x4002C21C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_08,           0x4002C220,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_09,           0x4002C224,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_10,           0x4002C228,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_11,           0x4002C22C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_12,           0x4002C230,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_13,           0x4002C234,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_14,           0x4002C238,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_15,           0x4002C23C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_16,           0x4002C240,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_17,           0x4002C244,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_18,           0x4002C248,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_19,           0x4002C24C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_20,           0x4002C250,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_21,           0x4002C254,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_22,           0x4002C258,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_23,           0x4002C25C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_24,           0x4002C260,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_25,           0x4002C264,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_26,           0x4002C268,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_27,           0x4002C26C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_28,           0x4002C270,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_29,           0x4002C274,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_30,           0x4002C278,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P4_31,           0x4002C27C,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P5_00,           0x4002C280,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P5_01,           0x4002C284,__READ_WRITE ,__iocon_d_bits);
__IO_REG32_BIT(IOCON_P5_02,           0x4002C288,__READ_WRITE ,__iocon_i_bits);
__IO_REG32_BIT(IOCON_P5_03,           0x4002C28C,__READ_WRITE ,__iocon_i_bits);
__IO_REG32_BIT(IOCON_P5_04,           0x4002C290,__READ_WRITE ,__iocon_d_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(FIO0DIR,         0x20098000,__READ_WRITE,__fgpio0_bits);
#define FIO0DIR0          FIO0DIR_bit.__byte0
#define FIO0DIR0_bit      FIO0DIR_bit.__byte0_bit
#define FIO0DIR1          FIO0DIR_bit.__byte1
#define FIO0DIR1_bit      FIO0DIR_bit.__byte1_bit
#define FIO0DIR2          FIO0DIR_bit.__byte2
#define FIO0DIR2_bit      FIO0DIR_bit.__byte2_bit
#define FIO0DIR3          FIO0DIR_bit.__byte3
#define FIO0DIR3_bit      FIO0DIR_bit.__byte3_bit
#define FIO0DIRL          FIO0DIR_bit.__shortl
#define FIO0DIRL_bit      FIO0DIR_bit.__shortl_bit
#define FIO0DIRU          FIO0DIR_bit.__shortu
#define FIO0DIRU_bit      FIO0DIR_bit.__shortu_bit
__IO_REG32_BIT(FIO0MASK,        0x20098010,__READ_WRITE,__fgpio0_bits);
#define FIO0MASK0         FIO0MASK_bit.__byte0
#define FIO0MASK0_bit     FIO0MASK_bit.__byte0_bit
#define FIO0MASK1         FIO0MASK_bit.__byte1
#define FIO0MASK1_bit     FIO0MASK_bit.__byte1_bit
#define FIO0MASK2         FIO0MASK_bit.__byte2
#define FIO0MASK2_bit     FIO0MASK_bit.__byte2_bit
#define FIO0MASK3         FIO0MASK_bit.__byte3
#define FIO0MASK3_bit     FIO0MASK_bit.__byte3_bit
#define FIO0MASKL         FIO0MASK_bit.__shortl
#define FIO0MASKL_bit     FIO0MASK_bit.__shortl_bit
#define FIO0MASKU         FIO0MASK_bit.__shortu
#define FIO0MASKU_bit     FIO0MASK_bit.__shortu_bit
__IO_REG32_BIT(FIO0PIN,         0x20098014,__READ_WRITE,__fgpio0_bits);
#define FIO0PIN0          FIO0PIN_bit.__byte0
#define FIO0PIN0_bit      FIO0PIN_bit.__byte0_bit
#define FIO0PIN1          FIO0PIN_bit.__byte1
#define FIO0PIN1_bit      FIO0PIN_bit.__byte1_bit
#define FIO0PIN2          FIO0PIN_bit.__byte2
#define FIO0PIN2_bit      FIO0PIN_bit.__byte2_bit
#define FIO0PIN3          FIO0PIN_bit.__byte3
#define FIO0PIN3_bit      FIO0PIN_bit.__byte3_bit
#define FIO0PINL          FIO0PIN_bit.__shortl
#define FIO0PINL_bit      FIO0PIN_bit.__shortl_bit
#define FIO0PINU          FIO0PIN_bit.__shortu
#define FIO0PINU_bit      FIO0PIN_bit.__shortu_bit
__IO_REG32_BIT(FIO0SET,         0x20098018,__READ_WRITE,__fgpio0_bits);
#define FIO0SET0          FIO0SET_bit.__byte0
#define FIO0SET0_bit      FIO0SET_bit.__byte0_bit
#define FIO0SET1          FIO0SET_bit.__byte1
#define FIO0SET1_bit      FIO0SET_bit.__byte1_bit
#define FIO0SET2          FIO0SET_bit.__byte2
#define FIO0SET2_bit      FIO0SET_bit.__byte2_bit
#define FIO0SET3          FIO0SET_bit.__byte3
#define FIO0SET3_bit      FIO0SET_bit.__byte3_bit
#define FIO0SETL          FIO0SET_bit.__shortl
#define FIO0SETL_bit      FIO0SET_bit.__shortl_bit
#define FIO0SETU          FIO0SET_bit.__shortu
#define FIO0SETU_bit      FIO0SET_bit.__shortu_bit
__IO_REG32_BIT(FIO0CLR,         0x2009801C,__WRITE     ,__fgpio0_bits);
#define FIO0CLR0          FIO0CLR_bit.__byte0
#define FIO0CLR0_bit      FIO0CLR_bit.__byte0_bit
#define FIO0CLR1          FIO0CLR_bit.__byte1
#define FIO0CLR1_bit      FIO0CLR_bit.__byte1_bit
#define FIO0CLR2          FIO0CLR_bit.__byte2
#define FIO0CLR2_bit      FIO0CLR_bit.__byte2_bit
#define FIO0CLR3          FIO0CLR_bit.__byte3
#define FIO0CLR3_bit      FIO0CLR_bit.__byte3_bit
#define FIO0CLRL          FIO0CLR_bit.__shortl
#define FIO0CLRL_bit      FIO0CLR_bit.__shortl_bit
#define FIO0CLRU          FIO0CLR_bit.__shortu
#define FIO0CLRU_bit      FIO0CLR_bit.__shortu_bit
__IO_REG32_BIT(FIO1DIR,         0x20098020,__READ_WRITE,__fgpio1_bits);
#define FIO1DIR0          FIO1DIR_bit.__byte0
#define FIO1DIR0_bit      FIO1DIR_bit.__byte0_bit
#define FIO1DIR1          FIO1DIR_bit.__byte1
#define FIO1DIR1_bit      FIO1DIR_bit.__byte1_bit
#define FIO1DIR2          FIO1DIR_bit.__byte2
#define FIO1DIR2_bit      FIO1DIR_bit.__byte2_bit
#define FIO1DIR3          FIO1DIR_bit.__byte3
#define FIO1DIR3_bit      FIO1DIR_bit.__byte3_bit
#define FIO1DIRL          FIO1DIR_bit.__shortl
#define FIO1DIRL_bit      FIO1DIR_bit.__shortl_bit
#define FIO1DIRU          FIO1DIR_bit.__shortu
#define FIO1DIRU_bit      FIO1DIR_bit.__shortu_bit
__IO_REG32_BIT(FIO1MASK,        0x20098030,__READ_WRITE,__fgpio1_bits);
#define FIO1MASK0         FIO1MASK_bit.__byte0
#define FIO1MASK0_bit     FIO1MASK_bit.__byte0_bit
#define FIO1MASK1         FIO1MASK_bit.__byte1
#define FIO1MASK1_bit     FIO1MASK_bit.__byte1_bit
#define FIO1MASK2         FIO1MASK_bit.__byte2
#define FIO1MASK2_bit     FIO1MASK_bit.__byte2_bit
#define FIO1MASK3         FIO1MASK_bit.__byte3
#define FIO1MASK3_bit     FIO1MASK_bit.__byte3_bit
#define FIO1MASKL         FIO1MASK_bit.__shortl
#define FIO1MASKL_bit     FIO1MASK_bit.__shortl_bit
#define FIO1MASKU         FIO1MASK_bit.__shortu
#define FIO1MASKU_bit     FIO1MASK_bit.__shortu_bit
__IO_REG32_BIT(FIO1PIN,         0x20098034,__READ_WRITE,__fgpio1_bits);
#define FIO1PIN0          FIO1PIN_bit.__byte0
#define FIO1PIN0_bit      FIO1PIN_bit.__byte0_bit
#define FIO1PIN1          FIO1PIN_bit.__byte1
#define FIO1PIN1_bit      FIO1PIN_bit.__byte1_bit
#define FIO1PIN2          FIO1PIN_bit.__byte2
#define FIO1PIN2_bit      FIO1PIN_bit.__byte2_bit
#define FIO1PIN3          FIO1PIN_bit.__byte3
#define FIO1PIN3_bit      FIO1PIN_bit.__byte3_bit
#define FIO1PINL          FIO1PIN_bit.__shortl
#define FIO1PINL_bit      FIO1PIN_bit.__shortl_bit
#define FIO1PINU          FIO1PIN_bit.__shortu
#define FIO1PINU_bit      FIO1PIN_bit.__shortu_bit
__IO_REG32_BIT(FIO1SET,         0x20098038,__READ_WRITE,__fgpio1_bits);
#define FIO1SET0          FIO1SET_bit.__byte0
#define FIO1SET0_bit      FIO1SET_bit.__byte0_bit
#define FIO1SET1          FIO1SET_bit.__byte1
#define FIO1SET1_bit      FIO1SET_bit.__byte1_bit
#define FIO1SET2          FIO1SET_bit.__byte2
#define FIO1SET2_bit      FIO1SET_bit.__byte2_bit
#define FIO1SET3          FIO1SET_bit.__byte3
#define FIO1SET3_bit      FIO1SET_bit.__byte3_bit
#define FIO1SETL          FIO1SET_bit.__shortl
#define FIO1SETL_bit      FIO1SET_bit.__shortl_bit
#define FIO1SETU          FIO1SET_bit.__shortu
#define FIO1SETU_bit      FIO1SET_bit.__shortu_bit
__IO_REG32_BIT(FIO1CLR,         0x2009803C,__WRITE     ,__fgpio1_bits);
#define FIO1CLR0          FIO1CLR_bit.__byte0
#define FIO1CLR0_bit      FIO1CLR_bit.__byte0_bit
#define FIO1CLR1          FIO1CLR_bit.__byte1
#define FIO1CLR1_bit      FIO1CLR_bit.__byte1_bit
#define FIO1CLR2          FIO1CLR_bit.__byte2
#define FIO1CLR2_bit      FIO1CLR_bit.__byte2_bit
#define FIO1CLR3          FIO1CLR_bit.__byte3
#define FIO1CLR3_bit      FIO1CLR_bit.__byte3_bit
#define FIO1CLRL          FIO1CLR_bit.__shortl
#define FIO1CLRL_bit      FIO1CLR_bit.__shortl_bit
#define FIO1CLRU          FIO1CLR_bit.__shortu
#define FIO1CLRU_bit      FIO1CLR_bit.__shortu_bit
__IO_REG32_BIT(FIO2DIR,         0x20098040,__READ_WRITE,__fgpio2_bits);
#define FIO2DIR0          FIO2DIR_bit.__byte0
#define FIO2DIR0_bit      FIO2DIR_bit.__byte0_bit
#define FIO2DIR1          FIO2DIR_bit.__byte1
#define FIO2DIR1_bit      FIO2DIR_bit.__byte1_bit
#define FIO2DIR2          FIO2DIR_bit.__byte2
#define FIO2DIR2_bit      FIO2DIR_bit.__byte2_bit
#define FIO2DIR3          FIO2DIR_bit.__byte3
#define FIO2DIR3_bit      FIO2DIR_bit.__byte3_bit
#define FIO2DIRL          FIO2DIR_bit.__shortl
#define FIO2DIRL_bit      FIO2DIR_bit.__shortl_bit
#define FIO2DIRU          FIO2DIR_bit.__shortu
#define FIO2DIRU_bit      FIO2DIR_bit.__shortu_bit
__IO_REG32_BIT(FIO2MASK,        0x20098050,__READ_WRITE,__fgpio2_bits);
#define FIO2MASK0         FIO2MASK_bit.__byte0
#define FIO2MASK0_bit     FIO2MASK_bit.__byte0_bit
#define FIO2MASK1         FIO2MASK_bit.__byte1
#define FIO2MASK1_bit     FIO2MASK_bit.__byte1_bit
#define FIO2MASK2         FIO2MASK_bit.__byte2
#define FIO2MASK2_bit     FIO2MASK_bit.__byte2_bit
#define FIO2MASK3         FIO2MASK_bit.__byte3
#define FIO2MASK3_bit     FIO2MASK_bit.__byte3_bit
#define FIO2MASKL         FIO2MASK_bit.__shortl
#define FIO2MASKL_bit     FIO2MASK_bit.__shortl_bit
#define FIO2MASKU         FIO2MASK_bit.__shortu
#define FIO2MASKU_bit     FIO2MASK_bit.__shortu_bit
__IO_REG32_BIT(FIO2PIN,         0x20098054,__READ_WRITE,__fgpio2_bits);
#define FIO2PIN0          FIO2PIN_bit.__byte0
#define FIO2PIN0_bit      FIO2PIN_bit.__byte0_bit
#define FIO2PIN1          FIO2PIN_bit.__byte1
#define FIO2PIN1_bit      FIO2PIN_bit.__byte1_bit
#define FIO2PIN2          FIO2PIN_bit.__byte2
#define FIO2PIN2_bit      FIO2PIN_bit.__byte2_bit
#define FIO2PIN3          FIO2PIN_bit.__byte3
#define FIO2PIN3_bit      FIO2PIN_bit.__byte3_bit
#define FIO2PINL          FIO2PIN_bit.__shortl
#define FIO2PINL_bit      FIO2PIN_bit.__shortl_bit
#define FIO2PINU          FIO2PIN_bit.__shortu
#define FIO2PINU_bit      FIO2PIN_bit.__shortu_bit
__IO_REG32_BIT(FIO2SET,         0x20098058,__READ_WRITE,__fgpio2_bits);
#define FIO2SET0          FIO2SET_bit.__byte0
#define FIO2SET0_bit      FIO2SET_bit.__byte0_bit
#define FIO2SET1          FIO2SET_bit.__byte1
#define FIO2SET1_bit      FIO2SET_bit.__byte1_bit
#define FIO2SET2          FIO2SET_bit.__byte2
#define FIO2SET2_bit      FIO2SET_bit.__byte2_bit
#define FIO2SET3          FIO2SET_bit.__byte3
#define FIO2SET3_bit      FIO2SET_bit.__byte3_bit
#define FIO2SETL          FIO2SET_bit.__shortl
#define FIO2SETL_bit      FIO2SET_bit.__shortl_bit
#define FIO2SETU          FIO2SET_bit.__shortu
#define FIO2SETU_bit      FIO2SET_bit.__shortu_bit
__IO_REG32_BIT(FIO2CLR,         0x2009805C,__WRITE     ,__fgpio2_bits);
#define FIO2CLR0          FIO2CLR_bit.__byte0
#define FIO2CLR0_bit      FIO2CLR_bit.__byte0_bit
#define FIO2CLR1          FIO2CLR_bit.__byte1
#define FIO2CLR1_bit      FIO2CLR_bit.__byte1_bit
#define FIO2CLR2          FIO2CLR_bit.__byte2
#define FIO2CLR2_bit      FIO2CLR_bit.__byte2_bit
#define FIO2CLR3          FIO2CLR_bit.__byte3
#define FIO2CLR3_bit      FIO2CLR_bit.__byte3_bit
#define FIO2CLRL          FIO2CLR_bit.__shortl
#define FIO2CLRL_bit      FIO2CLR_bit.__shortl_bit
#define FIO2CLRU          FIO2CLR_bit.__shortu
#define FIO2CLRU_bit      FIO2CLR_bit.__shortu_bit
__IO_REG32_BIT(FIO3DIR,         0x20098060,__READ_WRITE,__fgpio3_bits);
#define FIO3DIR0          FIO3DIR_bit.__byte0
#define FIO3DIR0_bit      FIO3DIR_bit.__byte0_bit
#define FIO3DIR1          FIO3DIR_bit.__byte1
#define FIO3DIR1_bit      FIO3DIR_bit.__byte1_bit
#define FIO3DIR2          FIO3DIR_bit.__byte2
#define FIO3DIR2_bit      FIO3DIR_bit.__byte2_bit
#define FIO3DIR3          FIO3DIR_bit.__byte3
#define FIO3DIR3_bit      FIO3DIR_bit.__byte3_bit
#define FIO3DIRL          FIO3DIR_bit.__shortl
#define FIO3DIRL_bit      FIO3DIR_bit.__shortl_bit
#define FIO3DIRU          FIO3DIR_bit.__shortu
#define FIO3DIRU_bit      FIO3DIR_bit.__shortu_bit
__IO_REG32_BIT(FIO3MASK,        0x20098070,__READ_WRITE,__fgpio3_bits);
#define FIO3MASK0         FIO3MASK_bit.__byte0
#define FIO3MASK0_bit     FIO3MASK_bit.__byte0_bit
#define FIO3MASK1         FIO3MASK_bit.__byte1
#define FIO3MASK1_bit     FIO3MASK_bit.__byte1_bit
#define FIO3MASK2         FIO3MASK_bit.__byte2
#define FIO3MASK2_bit     FIO3MASK_bit.__byte2_bit
#define FIO3MASK3         FIO3MASK_bit.__byte3
#define FIO3MASK3_bit     FIO3MASK_bit.__byte3_bit
#define FIO3MASKL         FIO3MASK_bit.__shortl
#define FIO3MASKL_bit     FIO3MASK_bit.__shortl_bit
#define FIO3MASKU         FIO3MASK_bit.__shortu
#define FIO3MASKU_bit     FIO3MASK_bit.__shortu_bit
__IO_REG32_BIT(FIO3PIN,         0x20098074,__READ_WRITE,__fgpio3_bits);
#define FIO3PIN0          FIO3PIN_bit.__byte0
#define FIO3PIN0_bit      FIO3PIN_bit.__byte0_bit
#define FIO3PIN1          FIO3PIN_bit.__byte1
#define FIO3PIN1_bit      FIO3PIN_bit.__byte1_bit
#define FIO3PIN2          FIO3PIN_bit.__byte2
#define FIO3PIN2_bit      FIO3PIN_bit.__byte2_bit
#define FIO3PIN3          FIO3PIN_bit.__byte3
#define FIO3PIN3_bit      FIO3PIN_bit.__byte3_bit
#define FIO3PINL          FIO3PIN_bit.__shortl
#define FIO3PINL_bit      FIO3PIN_bit.__shortl_bit
#define FIO3PINU          FIO3PIN_bit.__shortu
#define FIO3PINU_bit      FIO3PIN_bit.__shortu_bit
__IO_REG32_BIT(FIO3SET,         0x20098078,__READ_WRITE,__fgpio3_bits);
#define FIO3SET0          FIO3SET_bit.__byte0
#define FIO3SET0_bit      FIO3SET_bit.__byte0_bit
#define FIO3SET1          FIO3SET_bit.__byte1
#define FIO3SET1_bit      FIO3SET_bit.__byte1_bit
#define FIO3SET2          FIO3SET_bit.__byte2
#define FIO3SET2_bit      FIO3SET_bit.__byte2_bit
#define FIO3SET3          FIO3SET_bit.__byte3
#define FIO3SET3_bit      FIO3SET_bit.__byte3_bit
#define FIO3SETL          FIO3SET_bit.__shortl
#define FIO3SETL_bit      FIO3SET_bit.__shortl_bit
#define FIO3SETU          FIO3SET_bit.__shortu
#define FIO3SETU_bit      FIO3SET_bit.__shortu_bit
__IO_REG32_BIT(FIO3CLR,         0x2009807C,__WRITE     ,__fgpio3_bits);
#define FIO3CLR0          FIO3CLR_bit.__byte0
#define FIO3CLR0_bit      FIO3CLR_bit.__byte0_bit
#define FIO3CLR1          FIO3CLR_bit.__byte1
#define FIO3CLR1_bit      FIO3CLR_bit.__byte1_bit
#define FIO3CLR2          FIO3CLR_bit.__byte2
#define FIO3CLR2_bit      FIO3CLR_bit.__byte2_bit
#define FIO3CLR3          FIO3CLR_bit.__byte3
#define FIO3CLR3_bit      FIO3CLR_bit.__byte3_bit
#define FIO3CLRL          FIO3CLR_bit.__shortl
#define FIO3CLRL_bit      FIO3CLR_bit.__shortl_bit
#define FIO3CLRU          FIO3CLR_bit.__shortu
#define FIO3CLRU_bit      FIO3CLR_bit.__shortu_bit
__IO_REG32_BIT(FIO4DIR,         0x20098080,__READ_WRITE,__fgpio4_bits);
#define FIO4DIR0          FIO4DIR_bit.__byte0
#define FIO4DIR0_bit      FIO4DIR_bit.__byte0_bit
#define FIO4DIR1          FIO4DIR_bit.__byte1
#define FIO4DIR1_bit      FIO4DIR_bit.__byte1_bit
#define FIO4DIR2          FIO4DIR_bit.__byte2
#define FIO4DIR2_bit      FIO4DIR_bit.__byte2_bit
#define FIO4DIR3          FIO4DIR_bit.__byte3
#define FIO4DIR3_bit      FIO4DIR_bit.__byte3_bit
#define FIO4DIRL          FIO4DIR_bit.__shortl
#define FIO4DIRL_bit      FIO4DIR_bit.__shortl_bit
#define FIO4DIRU          FIO4DIR_bit.__shortu
#define FIO4DIRU_bit      FIO4DIR_bit.__shortu_bit
__IO_REG32_BIT(FIO4MASK,        0x20098090,__READ_WRITE,__fgpio4_bits);
#define FIO4MASK0         FIO4MASK_bit.__byte0
#define FIO4MASK0_bit     FIO4MASK_bit.__byte0_bit
#define FIO4MASK1         FIO4MASK_bit.__byte1
#define FIO4MASK1_bit     FIO4MASK_bit.__byte1_bit
#define FIO4MASK2         FIO4MASK_bit.__byte2
#define FIO4MASK2_bit     FIO4MASK_bit.__byte2_bit
#define FIO4MASK3         FIO4MASK_bit.__byte3
#define FIO4MASK3_bit     FIO4MASK_bit.__byte3_bit
#define FIO4MASKL         FIO4MASK_bit.__shortl
#define FIO4MASKL_bit     FIO4MASK_bit.__shortl_bit
#define FIO4MASKU         FIO4MASK_bit.__shortu
#define FIO4MASKU_bit     FIO4MASK_bit.__shortu_bit
__IO_REG32_BIT(FIO4PIN,         0x20098094,__READ_WRITE,__fgpio4_bits);
#define FIO4PIN0          FIO4PIN_bit.__byte0
#define FIO4PIN0_bit      FIO4PIN_bit.__byte0_bit
#define FIO4PIN1          FIO4PIN_bit.__byte1
#define FIO4PIN1_bit      FIO4PIN_bit.__byte1_bit
#define FIO4PIN2          FIO4PIN_bit.__byte2
#define FIO4PIN2_bit      FIO4PIN_bit.__byte2_bit
#define FIO4PIN3          FIO4PIN_bit.__byte3
#define FIO4PIN3_bit      FIO4PIN_bit.__byte3_bit
#define FIO4PINL          FIO4PIN_bit.__shortl
#define FIO4PINL_bit      FIO4PIN_bit.__shortl_bit
#define FIO4PINU          FIO4PIN_bit.__shortu
#define FIO4PINU_bit      FIO4PIN_bit.__shortu_bit
__IO_REG32_BIT(FIO4SET,         0x20098098,__READ_WRITE,__fgpio4_bits);
#define FIO4SET0          FIO4SET_bit.__byte0
#define FIO4SET0_bit      FIO4SET_bit.__byte0_bit
#define FIO4SET1          FIO4SET_bit.__byte1
#define FIO4SET1_bit      FIO4SET_bit.__byte1_bit
#define FIO4SET2          FIO4SET_bit.__byte2
#define FIO4SET2_bit      FIO4SET_bit.__byte2_bit
#define FIO4SET3          FIO4SET_bit.__byte3
#define FIO4SET3_bit      FIO4SET_bit.__byte3_bit
#define FIO4SETL          FIO4SET_bit.__shortl
#define FIO4SETL_bit      FIO4SET_bit.__shortl_bit
#define FIO4SETU          FIO4SET_bit.__shortu
#define FIO4SETU_bit      FIO4SET_bit.__shortu_bit
__IO_REG32_BIT(FIO4CLR,         0x2009809C,__WRITE     ,__fgpio4_bits);
#define FIO4CLR0          FIO4CLR_bit.__byte0
#define FIO4CLR0_bit      FIO4CLR_bit.__byte0_bit
#define FIO4CLR1          FIO4CLR_bit.__byte1
#define FIO4CLR1_bit      FIO4CLR_bit.__byte1_bit
#define FIO4CLR2          FIO4CLR_bit.__byte2
#define FIO4CLR2_bit      FIO4CLR_bit.__byte2_bit
#define FIO4CLR3          FIO4CLR_bit.__byte3
#define FIO4CLR3_bit      FIO4CLR_bit.__byte3_bit
#define FIO4CLRL          FIO4CLR_bit.__shortl
#define FIO4CLRL_bit      FIO4CLR_bit.__shortl_bit
#define FIO4CLRU          FIO4CLR_bit.__shortu
#define FIO4CLRU_bit      FIO4CLR_bit.__shortu_bit
__IO_REG32_BIT(FIO5DIR,         0x200980A0,__READ_WRITE,__fgpio5_bits);
#define FIO5DIR0          FIO5DIR_bit.__byte0
#define FIO5DIR0_bit      FIO5DIR_bit.__byte0_bit
#define FIO5DIRL          FIO5DIR_bit.__shortl
#define FIO5DIRL_bit      FIO5DIR_bit.__shortl_bit
__IO_REG32_BIT(FIO5MASK,        0x200980B0,__READ_WRITE,__fgpio5_bits);
#define FIO5MASK0         FIO5MASK_bit.__byte0
#define FIO5MASK0_bit     FIO5MASK_bit.__byte0_bit
#define FIO5MASKL         FIO5MASK_bit.__shortl
#define FIO5MASKL_bit     FIO5MASK_bit.__shortl_bit
__IO_REG32_BIT(FIO5PIN,         0x200980B4,__READ_WRITE,__fgpio5_bits);
#define FIO5PIN0          FIO5PIN_bit.__byte0
#define FIO5PIN0_bit      FIO5PIN_bit.__byte0_bit
#define FIO5PINL          FIO5PIN_bit.__shortl
#define FIO5PINL_bit      FIO5PIN_bit.__shortl_bit
__IO_REG32_BIT(FIO5SET,         0x200980B8,__READ_WRITE,__fgpio5_bits);
#define FIO5SET0          FIO5SET_bit.__byte0
#define FIO5SET0_bit      FIO5SET_bit.__byte0_bit
#define FIO5SETL          FIO5SET_bit.__shortl
#define FIO5SETL_bit      FIO5SET_bit.__shortl_bit
__IO_REG32_BIT(FIO5CLR,         0x200980BC,__WRITE     ,__fgpio5_bits);
#define FIO5CLR0          FIO5CLR_bit.__byte0
#define FIO5CLR0_bit      FIO5CLR_bit.__byte0_bit
#define FIO5CLRL          FIO5CLR_bit.__shortl
#define FIO5CLRL_bit      FIO5CLR_bit.__shortl_bit
__IO_REG32_BIT(IO0INTENR,             0x40028090,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(IO0INTENF,             0x40028094,__READ_WRITE ,__gpio0_bits);
__IO_REG32_BIT(IO0INTSTATR,           0x40028084,__READ       ,__gpio0_bits);
__IO_REG32_BIT(IO0INTSTATF,           0x40028088,__READ       ,__gpio0_bits);
__IO_REG32_BIT(IO0INTCLR,             0x4002808C,__WRITE      ,__gpio0_bits);
__IO_REG32_BIT(IO2INTENR,             0x400280B0,__READ_WRITE ,__gpio2_bits);
__IO_REG32_BIT(IO2INTENF,             0x400280B4,__READ_WRITE ,__gpio2_bits);
__IO_REG32_BIT(IO2INTSTATR,           0x400280A4,__READ       ,__gpio2_bits);
__IO_REG32_BIT(IO2INTSTATF,           0x400280A8,__READ       ,__gpio2_bits);
__IO_REG32_BIT(IO2INTCLR,             0x400280AC,__WRITE      ,__gpio2_bits);
__IO_REG32_BIT(IOINTSTATUS,           0x40028080,__READ       ,__iointst_bits);

/***************************************************************************
 **
 ** EMC
 **
 ***************************************************************************/
__IO_REG32_BIT(EMCControl,            0x2009C000,__READ_WRITE ,__emc_ctrl_bits);
__IO_REG32_BIT(EMCStatus,             0x2009C004,__READ       ,__emc_st_bits);
__IO_REG32_BIT(EMCConfig,             0x2009C008,__READ_WRITE ,__emc_cfg_bits);
__IO_REG32_BIT(EMCDynamicControl,     0x2009C020,__READ_WRITE ,__emc_dctrl_bits);
__IO_REG32_BIT(EMCDynamicRefresh,     0x2009C024,__READ_WRITE ,__emc_drfr_bits);
__IO_REG32_BIT(EMCDynamicReadConfig,  0x2009C028,__READ_WRITE ,__emc_drdcfg_bits);
__IO_REG32_BIT(EMCDynamictRP,         0x2009C030,__READ_WRITE ,__emc_drp_bits);
__IO_REG32_BIT(EMCDynamictRAS,        0x2009C034,__READ_WRITE ,__emc_dras_bits);
__IO_REG32_BIT(EMCDynamictSREX,       0x2009C038,__READ_WRITE ,__emc_dsrex_bits);
__IO_REG32_BIT(EMCDynamictAPR,        0x2009C03C,__READ_WRITE ,__emc_dapr_bits);
__IO_REG32_BIT(EMCDynamictDAL,        0x2009C040,__READ_WRITE ,__emc_ddal_bits);
__IO_REG32_BIT(EMCDynamictWR,         0x2009C044,__READ_WRITE ,__emc_dwr_bits);
__IO_REG32_BIT(EMCDynamictRC,         0x2009C048,__READ_WRITE ,__emc_drc_bits);
__IO_REG32_BIT(EMCDynamictRFC,        0x2009C04C,__READ_WRITE ,__emc_drfc_bits);
__IO_REG32_BIT(EMCDynamictXSR,        0x2009C050,__READ_WRITE ,__emc_dxsr_bits);
__IO_REG32_BIT(EMCDynamictRRD,        0x2009C054,__READ_WRITE ,__emc_drrd_bits);
__IO_REG32_BIT(EMCDynamictMRD,        0x2009C058,__READ_WRITE ,__emc_dmrd_bits);
__IO_REG32_BIT(EMCStaticExtendedWait, 0x2009C080,__READ_WRITE ,__emc_s_ext_wait_bits);
__IO_REG32_BIT(EMCDynamicConfig0,     0x2009C100,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas0,     0x2009C104,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig1,     0x2009C120,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas1,     0x2009C124,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig2,     0x2009C140,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas2,     0x2009C144,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig3,     0x2009C160,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas3,     0x2009C164,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCStaticConfig0,      0x2009C200,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen0,     0x2009C204,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen0,     0x2009C208,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd0,      0x2009C20C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage0,    0x2009C210,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr0,      0x2009C214,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn0,    0x2009C218,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCStaticConfig1,      0x2009C220,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen1,     0x2009C224,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen1,     0x2009C228,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd1,      0x2009C22C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage1,    0x2009C230,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr1,      0x2009C234,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn1,    0x2009C238,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCStaticConfig2,      0x2009C240,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen2,     0x2009C244,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen2,     0x2009C248,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd2,      0x2009C24C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage2,    0x2009C250,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr2,      0x2009C254,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn2,    0x2009C258,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCStaticConfig3,      0x2009C260,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen3,     0x2009C264,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen3,     0x2009C268,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd3,      0x2009C26C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage3,    0x2009C270,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr3,      0x2009C274,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn3,    0x2009C278,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCDLYCTL,             0x400FC1DC,__READ_WRITE ,__emcdlyctl_bits);
__IO_REG32_BIT(EMCCAL,                0x400FC1E0,__READ_WRITE ,__emccal_bits);

/***************************************************************************
 **
 ** USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USBPORTSEL,            0x2008C110,__READ_WRITE ,__usbportsel_bits);
__IO_REG32_BIT(USBCLKCTRL,            0x2008CFF4,__READ_WRITE ,__usbclkctrl_bits);
__IO_REG32_BIT(USBCLKST,              0x2008CFF8,__READ       ,__usbclkst_bits);
__IO_REG32_BIT(USBINTS,               0x400FC1C0,__READ_WRITE ,__usbints_bits);
__IO_REG32_BIT(USBDEVINTST,           0x2008C200,__READ       ,__usbdevintst_bits);
__IO_REG32_BIT(USBDEVINTEN,           0x2008C204,__READ_WRITE ,__usbdevintst_bits);
__IO_REG32_BIT(USBDEVINTCLR,          0x2008C208,__WRITE      ,__usbdevintst_bits);
__IO_REG32_BIT(USBDEVINTSET,          0x2008C20C,__WRITE      ,__usbdevintst_bits);
__IO_REG8_BIT( USBDEVINTPRI,          0x2008C22C,__WRITE      ,__usbdevintpri_bits);
__IO_REG32_BIT(USBEPINTST,            0x2008C230,__READ       ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTEN,            0x2008C234,__READ_WRITE ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTCLR,           0x2008C238,__WRITE      ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTSET,           0x2008C23C,__WRITE      ,__usbepintst_bits);
__IO_REG32_BIT(USBEPINTPRI,           0x2008C240,__WRITE      ,__usbepintst_bits);
__IO_REG32_BIT(USBREEP,               0x2008C244,__READ_WRITE ,__usbreep_bits);
__IO_REG32_BIT(USBEPIN,               0x2008C248,__WRITE      ,__usbepin_bits);
__IO_REG32_BIT(USBMAXPSIZE,           0x2008C24C,__READ_WRITE ,__usbmaxpsize_bits);
__IO_REG32(    USBRXDATA,             0x2008C218,__READ);
__IO_REG32_BIT(USBRXPLEN,             0x2008C220,__READ       ,__usbrxplen_bits);
__IO_REG32(    TDATA,                 0x2008C21C,__WRITE);
__IO_REG32_BIT(USBTXPLEN,             0x2008C224,__WRITE      ,__usbtxplen_bits);
__IO_REG32_BIT(USBCTRL,               0x2008C228,__READ_WRITE ,__usbctrl_bits);
__IO_REG32_BIT(USBCMDCODE,            0x2008C210,__WRITE      ,__usbcmdcode_bits);
__IO_REG32_BIT(USBCMDDATA,            0x2008C214,__READ       ,__usbcmddata_bits);
__IO_REG32_BIT(USBDMARST,             0x2008C250,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBDMARCLR,            0x2008C254,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBDMARSET,            0x2008C258,__WRITE      ,__usbreep_bits);
__IO_REG32(    USBUDCAH,              0x2008C280,__READ_WRITE );
__IO_REG32_BIT(USBEPDMAST,            0x2008C284,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBEPDMAEN,            0x2008C288,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBEPDMADIS,           0x2008C28C,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBDMAINTST,           0x2008C290,__READ       ,__usbdmaintst_bits);
__IO_REG32_BIT(USBDMAINTEN,           0x2008C294,__READ_WRITE ,__usbdmaintst_bits);
__IO_REG32_BIT(USBEOTINTST,           0x2008C2A0,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBEOTINTCLR,          0x2008C2A4,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBEOTINTSET,          0x2008C2A8,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBNDDRINTST,          0x2008C2AC,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBNDDRINTCLR,         0x2008C2B0,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBNDDRINTSET,         0x2008C2B4,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBSYSERRINTST,        0x2008C2B8,__READ       ,__usbreep_bits);
__IO_REG32_BIT(USBSYSERRINTCLR,       0x2008C2BC,__WRITE      ,__usbreep_bits);
__IO_REG32_BIT(USBSYSERRINTSET,       0x2008C2C0,__WRITE      ,__usbreep_bits);

/***************************************************************************
 **
 ** USB HOST (OHCI) CONTROLLER
 **
 ***************************************************************************/
__IO_REG32_BIT(HCREVISION,            0x2008C000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(HCCONTROL,             0x2008C004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(HCCOMMANDSTATUS,       0x2008C008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(HCINTERRUPTSTATUS,     0x2008C00C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(HCINTERRUPTENABLE,     0x2008C010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCINTERRUPTDISABLE,    0x2008C014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCHCCA,                0x2008C018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(HCPERIODCURRENTED,     0x2008C01C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(HCCONTROLHEADED,       0x2008C020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(HCCONTROLCURRENTED,    0x2008C024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(HCBULKHEADED,          0x2008C028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(HCBULKCURRENTED,       0x2008C02C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(HCDONEHEAD,            0x2008C030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(HCFMINTERVAL,          0x2008C034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(HCFMREMAINING,         0x2008C038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(HCFMNUMBER,            0x2008C03C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(HCPERIODICSTART,       0x2008C040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(HCLSTHRESHOLD,         0x2008C044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(HCRHDESCRIPTORA,       0x2008C048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(HCRHDESCRIPTORB,       0x2008C04C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(HCRHSTATUS,            0x2008C050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS1,       0x2008C054,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS2,       0x2008C058,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32(    HCRMID,                0x2008C0FC,__READ);

/***************************************************************************
 **
 ** USB OTG Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(OTGINTST,              0x2008C100,__READ       ,__otgintst_bits);
__IO_REG32_BIT(OTGINTEN,              0x2008C104,__READ_WRITE ,__otgintst_bits);
__IO_REG32_BIT(OTGINTSET,             0x2008C108,__WRITE      ,__otgintst_bits);
__IO_REG32_BIT(OTGINTCLR,             0x2008C10C,__WRITE      ,__otgintst_bits);
#define OTGSTCTRL      USBPORTSEL
#define OTGSTCTRL_bit  USBPORTSEL_bit
__IO_REG32_BIT(OTGTMR,                0x2008C114,__READ_WRITE ,__otgtmr_bits);
__IO_REG32_BIT(I2C_RX,                0x2008C300,__READ_WRITE ,__otg_i2c_rx_tx_bits);
#define I2C_TX      I2C_RX
#define I2C_TX_bit  I2C_RX_bit
__IO_REG32_BIT(I2C_STS,               0x2008C304,__READ_WRITE ,__otg_i2c_sts_bits);
__IO_REG32_BIT(I2C_CTL,               0x2008C308,__READ_WRITE ,__otg_i2c_ctl_bits);
__IO_REG32_BIT(I2C_CLKHI,             0x2008C30C,__READ_WRITE ,__i2c_clkhi_bits);
__IO_REG32_BIT(I2C_CLKLO,             0x2008C310,__READ_WRITE ,__i2c_clklo_bits);
#define OTGCLKCTRL        USBCLKCTRL
#define OTGCLKCTRL_bit    USBCLKCTRL_bit
#define OTGCLKST          USBCLKST
#define OTGCLKST_bit      USBCLKST_bit

/***************************************************************************
 **
 ** SD/MMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MCIPOWER,              0x400C0000,__READ_WRITE ,__mcipower_bits);
__IO_REG32_BIT(MCICLOCK,              0x400C0004,__READ_WRITE ,__mciclock_bits);
__IO_REG32(    MCIARGUMENT,           0x400C0008,__READ_WRITE);
__IO_REG32_BIT(MCICOMMAND,            0x400C000C,__READ_WRITE ,__mcicommand_bits);
__IO_REG32_BIT(MCIRESPCMD,            0x400C0010,__READ       ,__mcirespcmd_bits);
__IO_REG32(    MCIRESPONSE0,          0x400C0014,__READ);
__IO_REG32(    MCIRESPONSE1,          0x400C0018,__READ);
__IO_REG32(    MCIRESPONSE2,          0x400C001C,__READ);
__IO_REG32(    MCIRESPONSE3,          0x400C0020,__READ);
__IO_REG32(    MCIDATATIMER,          0x400C0024,__READ_WRITE);
__IO_REG16(    MCIDATALENGTH,         0x400C0028,__READ_WRITE);
__IO_REG32_BIT(MCIDATACTRL,           0x400C002C,__READ_WRITE ,__mcidatactrl_bits);
__IO_REG16(    MCIDATACNT,            0x400C0030,__READ);
__IO_REG32_BIT(MCISTATUS,             0x400C0034,__READ       ,__mcistatus_bits);
__IO_REG32_BIT(MCICLEAR,              0x400C0038,__WRITE      ,__mciclear_bits);
__IO_REG32_BIT(MCIMASK0,              0x400C003C,__READ_WRITE ,__mcistatus_bits);
__IO_REG32_BIT(MCIFIFOCNT,            0x400C0048,__READ       ,__mcififocnt_bits);
__IO_REG32(    MCIFIFO0,              0x400C0080,__READ_WRITE);
__IO_REG32(    MCIFIFO1,              0x400C0084,__READ_WRITE);
__IO_REG32(    MCIFIFO2,              0x400C0088,__READ_WRITE);
__IO_REG32(    MCIFIFO3,              0x400C008C,__READ_WRITE);
__IO_REG32(    MCIFIFO4,              0x400C0090,__READ_WRITE);
__IO_REG32(    MCIFIFO5,              0x400C0094,__READ_WRITE);
__IO_REG32(    MCIFIFO6,              0x400C0098,__READ_WRITE);
__IO_REG32(    MCIFIFO7,              0x400C009C,__READ_WRITE);
__IO_REG32(    MCIFIFO8,              0x400C00A0,__READ_WRITE);
__IO_REG32(    MCIFIFO9,              0x400C00A4,__READ_WRITE);
__IO_REG32(    MCIFIFO10,             0x400C00A8,__READ_WRITE);
__IO_REG32(    MCIFIFO11,             0x400C00AC,__READ_WRITE);
__IO_REG32(    MCIFIFO12,             0x400C00B0,__READ_WRITE);
__IO_REG32(    MCIFIFO13,             0x400C00B4,__READ_WRITE);
__IO_REG32(    MCIFIFO14,             0x400C00B8,__READ_WRITE);
__IO_REG32(    MCIFIFO15,             0x400C00BC,__READ_WRITE);

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
/* U0DLL, U0RBR and U0THR share the same address */
__IO_REG8(     U0RBRTHR,              0x4000C000,__READ_WRITE);
#define U0DLL U0RBRTHR
#define U0RBR U0RBRTHR
#define U0THR U0RBRTHR

/* U0DLM and U0IER share the same address */
__IO_REG32_BIT(U0IER,                 0x4000C004,__READ_WRITE ,__uartier0_bits);
#define U0DLM      U0IER

/* U0FCR and U0IIR share the same address */
__IO_REG32_BIT(U0FCR,                 0x4000C008,__READ_WRITE ,__uartfcriir_bits);
#define U0IIR      U0FCR
#define U0IIR_bit  U0FCR_bit

__IO_REG8_BIT( U0LCR,                 0x4000C00C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U0LSR,                 0x4000C014,__READ       ,__uartlsr_bits);
__IO_REG8(     U0SCR,                 0x4000C01C,__READ_WRITE);
__IO_REG32_BIT(U0ACR,                 0x4000C020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U0ICR,                 0x4000C024,__READ_WRITE ,__uarticr_bits);
__IO_REG32_BIT(U0FDR,                 0x4000C028,__READ_WRITE ,__uartfdr_bits);
__IO_REG8_BIT( U0TER,                 0x4000C030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U0RS485CTRL,           0x4000C04C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U0ADRMATCH,            0x4000C050,__READ_WRITE );
__IO_REG8(     U0RS485DLY,            0x4000C054,__READ_WRITE );

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
/* U1DLL, U1RBR and U1THR share the same address */
__IO_REG8(     U1RBRTHR,              0x40010000,__READ_WRITE);
#define U1DLL U1RBRTHR
#define U1RBR U1RBRTHR
#define U1THR U1RBRTHR

/* U1DLM and U1IER share the same address */
__IO_REG32_BIT(U1IER,                 0x40010004,__READ_WRITE ,__uartier1_bits);
#define U1DLM      U1IER

/* U1FCR and U1IIR share the same address */
__IO_REG32_BIT(U1FCR,                 0x40010008,__READ_WRITE ,__uartfcriir_bits);
#define U1IIR      U1FCR
#define U1IIR_bit  U1FCR_bit

__IO_REG8_BIT( U1LCR,                 0x4001000C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U1MCR,                 0x40010010,__READ_WRITE ,__uartmcr_bits);
__IO_REG8_BIT( U1LSR,                 0x40010014,__READ       ,__uartlsr_bits);
__IO_REG8_BIT( U1MSR,                 0x40010018,__READ       ,__uartmsr_bits);
__IO_REG8(     U1SCR,                 0x4001001C,__READ_WRITE);
__IO_REG32_BIT(U1ACR,                 0x40010020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U1FDR,                 0x40010028,__READ_WRITE ,__uartfdr_bits);
__IO_REG8_BIT( U1TER,                 0x40010030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U1RS485CTRL,           0x4001004C,__READ_WRITE ,__uars1485ctrl_bits);
__IO_REG8(     U1ADRMATCH,            0x40010050,__READ_WRITE );
__IO_REG8(     U1RS485DLY,            0x40010054,__READ_WRITE );

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
/* U2DLL, U2RBR and U2THR share the same address */
__IO_REG8(     U2RBRTHR,              0x40098000,__READ_WRITE);
#define U2DLL U2RBRTHR
#define U2RBR U2RBRTHR
#define U2THR U2RBRTHR

/* U2DLM and U2IER share the same address */
__IO_REG32_BIT(U2IER,                 0x40098004,__READ_WRITE ,__uartier0_bits);
#define U2DLM      U2IER

/* U2FCR and U2IIR share the same address */
__IO_REG32_BIT(U2FCR,                 0x40098008,__READ_WRITE ,__uartfcriir_bits);
#define U2IIR      U2FCR
#define U2IIR_bit  U2FCR_bit

__IO_REG8_BIT( U2LCR,                 0x4009800C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U2LSR,                 0x40098014,__READ       ,__uartlsr_bits);
__IO_REG8(     U2SCR,                 0x4009801C,__READ_WRITE);
__IO_REG32_BIT(U2ACR,                 0x40098020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U2ICR,                 0x40098024,__READ_WRITE ,__uarticr_bits);
__IO_REG32_BIT(U2FDR,                 0x40098028,__READ_WRITE ,__uartfdr_bits);
__IO_REG8_BIT( U2TER,                 0x40098030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U2RS485CTRL,           0x4009804C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U2ADRMATCH,            0x40098050,__READ_WRITE );
__IO_REG8(     U2RS485DLY,            0x40098054,__READ_WRITE );

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
/* U3DLL, U3RBR and U3THR share the same address */
__IO_REG8(     U3RBRTHR,              0x4009C000,__READ_WRITE);
#define U3DLL U3RBRTHR
#define U3RBR U3RBRTHR
#define U3THR U3RBRTHR

/* U3DLM and U3IER share the same address */
__IO_REG32_BIT(U3IER,                 0x4009C004,__READ_WRITE ,__uartier0_bits);
#define U3DLM      U3IER

/* U3FCR and U3IIR share the same address */
__IO_REG32_BIT(U3FCR,                 0x4009C008,__READ_WRITE ,__uartfcriir_bits);
#define U3IIR      U3FCR
#define U3IIR_bit  U3FCR_bit

__IO_REG8_BIT( U3LCR,                 0x4009C00C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U3LSR,                 0x4009C014,__READ       ,__uartlsr_bits);
__IO_REG8(     U3SCR,                 0x4009C01C,__READ_WRITE);
__IO_REG32_BIT(U3ACR,                 0x4009C020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U3ICR,                 0x4009C024,__READ_WRITE ,__uarticr_bits);
__IO_REG32_BIT(U3FDR,                 0x4009C028,__READ_WRITE ,__uartfdr_bits);
__IO_REG8_BIT( U3TER,                 0x4009C030,__READ_WRITE ,__uartter_bits);
__IO_REG32_BIT(U3RS485CTRL,           0x4009C04C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U3ADRMATCH,            0x4009C050,__READ_WRITE );
__IO_REG8(     U3RS485DLY,            0x4009C054,__READ_WRITE );

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
/* U4DLL, U4RBR and U4THR share the same address */
__IO_REG8(     U4RBRTHR,              0x400A4000,__READ_WRITE);
#define U4DLL U4RBRTHR
#define U4RBR U4RBRTHR
#define U4THR U4RBRTHR

/* U4DLM and U4IER share the same address */
__IO_REG32_BIT(U4IER,                 0x400A4004,__READ_WRITE ,__uartier0_bits);
#define U4DLM      U4IER

/* U4FCR and U4IIR share the same address */
__IO_REG32_BIT(U4FCR,                 0x400A4008,__READ_WRITE ,__uartfcriir_bits);
#define U4IIR      U4FCR
#define U4IIR_bit  U4FCR_bit

__IO_REG8_BIT( U4LCR,                 0x400A400C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U4LSR,                 0x400A4014,__READ       ,__uartlsr_bits);
__IO_REG8(     U4SCR,                 0x400A401C,__READ_WRITE);
__IO_REG32_BIT(U4ACR,                 0x400A4020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U4ICR,                 0x400A4024,__READ_WRITE ,__uarticr_bits);
__IO_REG32_BIT(U4FDR,                 0x400A4028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U4OSR,                 0x400A4040,__READ_WRITE ,__uartosr_bits);
__IO_REG32_BIT(U4SCICTRL,             0x400A4048,__READ_WRITE ,__uartscictrl_bits);
__IO_REG32_BIT(U4RS485CTRL,           0x400A404C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U4ADRMATCH,            0x400A4050,__READ_WRITE );
__IO_REG8(     U4RS485DLY,            0x400A4054,__READ_WRITE );
__IO_REG32_BIT(U4SYNCCTRL,            0x400A4058,__READ_WRITE ,__uartsyncctrl_bits);
__IO_REG32_BIT(U4TER,                 0x400A405C,__READ_WRITE ,__uart4ter_bits);

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(AFMR,                  0x4003C000,__READ_WRITE ,__afmr_bits);
__IO_REG32(    SFF_SA,                0x4003C004,__READ_WRITE);
__IO_REG32(    SFF_GRP_SA,            0x4003C008,__READ_WRITE);
__IO_REG32(    EFF_SA,                0x4003C00C,__READ_WRITE);
__IO_REG32(    EFF_GRP_SA,            0x4003C010,__READ_WRITE);
__IO_REG32(    ENDOFTABLE,            0x4003C014,__READ_WRITE);
__IO_REG32(    LUTERRAD,              0x4003C018,__READ);
__IO_REG32_BIT(LUTERR,                0x4003C01C,__READ       ,__luterr_bits);
__IO_REG32_BIT(FCANIE,                0x4003C020,__READ_WRITE ,__fcanie_bits);
__IO_REG32_BIT(FCANIC0,               0x4003C024,__READ_WRITE ,__fcanic0_bits);
__IO_REG32_BIT(FCANIC1,               0x4003C028,__READ_WRITE ,__fcanic1_bits);
__IO_REG32_BIT(CANTXSR,               0x40040000,__READ       ,__cantxsr_bits);
__IO_REG32_BIT(CANRXSR,               0x40040004,__READ       ,__canrxsr_bits);
__IO_REG32_BIT(CANMSR,                0x40040008,__READ       ,__canmsr_bits);
__IO_REG32_BIT(CAN1MOD,               0x40044000,__READ_WRITE ,__canmod_bits);
__IO_REG32_BIT(CAN1CMR,               0x40044004,__WRITE      ,__cancmr_bits);
__IO_REG32_BIT(CAN1GSR,               0x40044008,__READ_WRITE ,__cangsr_bits);
__IO_REG32_BIT(CAN1ICR,               0x4004400C,__READ       ,__canicr_bits);
__IO_REG32_BIT(CAN1IER,               0x40044010,__READ_WRITE ,__canier_bits);
__IO_REG32_BIT(CAN1BTR,               0x40044014,__READ_WRITE ,__canbtr_bits);
__IO_REG32_BIT(CAN1EWL,               0x40044018,__READ_WRITE ,__canewl_bits);
__IO_REG32_BIT(CAN1SR,                0x4004401C,__READ       ,__cansr_bits);
__IO_REG32_BIT(CAN1RFS,               0x40044020,__READ_WRITE ,__canrfs_bits);
__IO_REG32_BIT(CAN1RID,               0x40044024,__READ_WRITE ,__canrid_bits);
__IO_REG32_BIT(CAN1RDA,               0x40044028,__READ_WRITE ,__canrda_bits);
__IO_REG32_BIT(CAN1RDB,               0x4004402C,__READ_WRITE ,__canrdb_bits);
__IO_REG32_BIT(CAN1TFI1,              0x40044030,__READ_WRITE ,__cantfi_bits);
__IO_REG32_BIT(CAN1TID1,              0x40044034,__READ_WRITE ,__cantid_bits);
__IO_REG32_BIT(CAN1TDA1,              0x40044038,__READ_WRITE ,__cantda_bits);
__IO_REG32_BIT(CAN1TDB1,              0x4004403C,__READ_WRITE ,__cantdb_bits);
__IO_REG32_BIT(CAN1TFI2,              0x40044040,__READ_WRITE ,__cantfi_bits);
__IO_REG32_BIT(CAN1TID2,              0x40044044,__READ_WRITE ,__cantid_bits);
__IO_REG32_BIT(CAN1TDA2,              0x40044048,__READ_WRITE ,__cantda_bits);
__IO_REG32_BIT(CAN1TDB2,              0x4004404C,__READ_WRITE ,__cantdb_bits);
__IO_REG32_BIT(CAN1TFI3,              0x40044050,__READ_WRITE ,__cantfi_bits);
__IO_REG32_BIT(CAN1TID3,              0x40044054,__READ_WRITE ,__cantid_bits);
__IO_REG32_BIT(CAN1TDA3,              0x40044058,__READ_WRITE ,__cantda_bits);
__IO_REG32_BIT(CAN1TDB3,              0x4004405C,__READ_WRITE ,__cantdb_bits);
__IO_REG32_BIT(CAN2MOD,               0x40048000,__READ_WRITE ,__canmod_bits);
__IO_REG32_BIT(CAN2CMR,               0x40048004,__WRITE      ,__cancmr_bits);
__IO_REG32_BIT(CAN2GSR,               0x40048008,__READ_WRITE ,__cangsr_bits);
__IO_REG32_BIT(CAN2ICR,               0x4004800C,__READ       ,__canicr_bits);
__IO_REG32_BIT(CAN2IER,               0x40048010,__READ_WRITE ,__canier_bits);
__IO_REG32_BIT(CAN2BTR,               0x40048014,__READ_WRITE ,__canbtr_bits);
__IO_REG32_BIT(CAN2EWL,               0x40048018,__READ_WRITE ,__canewl_bits);
__IO_REG32_BIT(CAN2SR,                0x4004801C,__READ       ,__cansr_bits);
__IO_REG32_BIT(CAN2RFS,               0x40048020,__READ_WRITE ,__canrfs_bits);
__IO_REG32_BIT(CAN2RID,               0x40048024,__READ_WRITE ,__canrid_bits);
__IO_REG32_BIT(CAN2RDA,               0x40048028,__READ_WRITE ,__canrda_bits);
__IO_REG32_BIT(CAN2RDB,               0x4004802C,__READ_WRITE ,__canrdb_bits);
__IO_REG32_BIT(CAN2TFI1,              0x40048030,__READ_WRITE ,__cantfi_bits);
__IO_REG32_BIT(CAN2TID1,              0x40048034,__READ_WRITE ,__cantid_bits);
__IO_REG32_BIT(CAN2TDA1,              0x40048038,__READ_WRITE ,__cantda_bits);
__IO_REG32_BIT(CAN2TDB1,              0x4004803C,__READ_WRITE ,__cantdb_bits);
__IO_REG32_BIT(CAN2TFI2,              0x40048040,__READ_WRITE ,__cantfi_bits);
__IO_REG32_BIT(CAN2TID2,              0x40048044,__READ_WRITE ,__cantid_bits);
__IO_REG32_BIT(CAN2TDA2,              0x40048048,__READ_WRITE ,__cantda_bits);
__IO_REG32_BIT(CAN2TDB2,              0x4004804C,__READ_WRITE ,__cantdb_bits);
__IO_REG32_BIT(CAN2TFI3,              0x40048050,__READ_WRITE ,__cantfi_bits);
__IO_REG32_BIT(CAN2TID3,              0x40048054,__READ_WRITE ,__cantid_bits);
__IO_REG32_BIT(CAN2TDA3,              0x40048058,__READ_WRITE ,__cantda_bits);
__IO_REG32_BIT(CAN2TDB3,              0x4004805C,__READ_WRITE ,__cantdb_bits);
__IO_REG32_BIT(CANSLEEPCLR,           0x400FC110,__READ_WRITE ,__cansleepclr_bits);
__IO_REG32_BIT(CANWAKEFLAGS,          0x400FC114,__READ_WRITE ,__canwakeflags_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,               0x40088000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,               0x40088004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,                0x40088008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,                0x4008800C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,              0x40088010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,              0x40088014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,               0x40088018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,               0x4008801C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,               0x40088020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,             0x40088024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,               0x40030000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,               0x40030004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,                0x40030008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,                0x4003000C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,              0x40030010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,              0x40030014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,               0x40030018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,               0x4003001C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,               0x40030020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,             0x40030024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** SSP2
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP2CR0,               0x400AC000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP2CR1,               0x400AC004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP2DR,                0x400AC008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP2SR,                0x400AC00C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP2CPSR,              0x400AC010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP2IMSC,              0x400AC014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP2RIS,               0x400AC018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP2MIS,               0x400AC01C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP2ICR,               0x400AC020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP2DMACR,             0x400AC024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CONSET,            0x4001C000,__READ_WRITE ,__i2conset_bits);
__IO_REG32_BIT(I2C0STAT,              0x4001C004,__READ       ,__i2stat_bits);
__IO_REG32_BIT(I2C0DAT,               0x4001C008,__READ_WRITE ,__i2dat_bits);
__IO_REG32_BIT(I2C0ADR0,              0x4001C00C,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0SCLH,              0x4001C010,__READ_WRITE ,__i2sch_bits);
__IO_REG32_BIT(I2C0SCLL,              0x4001C014,__READ_WRITE ,__i2scl_bits);
__IO_REG32_BIT(I2C0CONCLR,            0x4001C018,__WRITE      ,__i2conclr_bits);
__IO_REG32_BIT(I2C0MMCTRL,            0x4001C01C,__READ_WRITE ,__i2cmmctrl_bits);
__IO_REG32_BIT(I2C0ADR1,              0x4001C020,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR2,              0x4001C024,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR3,              0x4001C028,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0DATABUFFER,        0x4001C02C,__READ       ,__i2dat_bits);
__IO_REG32_BIT(I2C0MASK0,             0x4001C030,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK1,             0x4001C034,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK2,             0x4001C038,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK3,             0x4001C03C,__READ_WRITE ,__i2cmask_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CONSET,            0x4005C000,__READ_WRITE ,__i2conset_bits);
__IO_REG32_BIT(I2C1STAT,              0x4005C004,__READ       ,__i2stat_bits);
__IO_REG32_BIT(I2C1DAT,               0x4005C008,__READ_WRITE ,__i2dat_bits);
__IO_REG32_BIT(I2C1ADR0,              0x4005C00C,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1SCLH,              0x4005C010,__READ_WRITE ,__i2sch_bits);
__IO_REG32_BIT(I2C1SCLL,              0x4005C014,__READ_WRITE ,__i2scl_bits);
__IO_REG32_BIT(I2C1CONCLR,            0x4005C018,__WRITE      ,__i2conclr_bits);
__IO_REG32_BIT(I2C1MMCTRL,            0x4005C01C,__READ_WRITE ,__i2cmmctrl_bits);
__IO_REG32_BIT(I2C1ADR1,              0x4005C020,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1ADR2,              0x4005C024,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1ADR3,              0x4005C028,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1DATABUFFER,        0x4005C02C,__READ       ,__i2dat_bits);
__IO_REG32_BIT(I2C1MASK0,             0x4005C030,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C1MASK1,             0x4005C034,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C1MASK2,             0x4005C038,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C1MASK3,             0x4005C03C,__READ_WRITE ,__i2cmask_bits);

/***************************************************************************
 **
 ** I2C2
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C2CONSET,            0x400A0000,__READ_WRITE ,__i2conset_bits);
__IO_REG32_BIT(I2C2STAT,              0x400A0004,__READ       ,__i2stat_bits);
__IO_REG32_BIT(I2C2DAT,               0x400A0008,__READ_WRITE ,__i2dat_bits);
__IO_REG32_BIT(I2C2ADR0,              0x400A000C,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C2SCLH,              0x400A0010,__READ_WRITE ,__i2sch_bits);
__IO_REG32_BIT(I2C2SCLL,              0x400A0014,__READ_WRITE ,__i2scl_bits);
__IO_REG32_BIT(I2C2CONCLR,            0x400A0018,__WRITE      ,__i2conclr_bits);
__IO_REG32_BIT(I2C2MMCTRL,            0x400A001C,__READ_WRITE ,__i2cmmctrl_bits);
__IO_REG32_BIT(I2C2ADR1,              0x400A0020,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C2ADR2,              0x400A0024,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C2ADR3,              0x400A0028,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C2DATABUFFER,        0x400A002C,__READ       ,__i2dat_bits);
__IO_REG32_BIT(I2C2MASK0,             0x400A0030,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C2MASK1,             0x400A0034,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C2MASK2,             0x400A0038,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C2MASK3,             0x400A003C,__READ_WRITE ,__i2cmask_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2SDAO,                0x400A8000,__READ_WRITE ,__i2sdao_bits);
__IO_REG32_BIT(I2SDAI,                0x400A8004,__READ_WRITE ,__i2sdai_bits);
__IO_REG32(    I2STXFIFO,             0x400A8008,__WRITE);
__IO_REG32(    I2SRXFIFO,             0x400A800C,__READ);
__IO_REG32_BIT(I2SSTATE,              0x400A8010,__READ       ,__i2sstate_bits);
__IO_REG32_BIT(I2SDMA1,               0x400A8014,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2SDMA2,               0x400A8018,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2SIRQ,                0x400A801C,__READ_WRITE ,__i2sirq_bits);
__IO_REG32_BIT(I2STXRATE,             0x400A8020,__READ_WRITE ,__i2stxrate_bits);
__IO_REG32_BIT(I2SRXRATE,             0x400A8024,__READ_WRITE ,__i2stxrate_bits);
__IO_REG32_BIT(I2STXBITRATE,          0x400A8028,__READ_WRITE ,__i2stxbitrate_bits);
__IO_REG32_BIT(I2SRXBITRATE,          0x400A802C,__READ_WRITE ,__i2srxbitrate_bits);
__IO_REG32_BIT(I2STXMODE,             0x400A8030,__READ_WRITE ,__i2stxmode_bits);
__IO_REG32_BIT(I2SRXMODE,             0x400A8034,__READ_WRITE ,__i2srxmode_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(T0IR,                  0x40004000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T0TCR,                 0x40004004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T0TC,                  0x40004008,__READ_WRITE);
__IO_REG32(    T0PR,                  0x4000400C,__READ_WRITE);
__IO_REG32(    T0PC,                  0x40004010,__READ_WRITE);
__IO_REG32_BIT(T0MCR,                 0x40004014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T0MR0,                 0x40004018,__READ_WRITE);
__IO_REG32(    T0MR1,                 0x4000401C,__READ_WRITE);
__IO_REG32(    T0MR2,                 0x40004020,__READ_WRITE);
__IO_REG32(    T0MR3,                 0x40004024,__READ_WRITE);
__IO_REG32_BIT(T0CCR,                 0x40004028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T0CR0,                 0x4000402C,__READ);
__IO_REG32(    T0CR1,                 0x40004030,__READ);
__IO_REG32_BIT(T0EMR,                 0x4000403C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T0CTCR,                0x40004070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(T1IR,                  0x40008000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T1TCR,                 0x40008004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T1TC,                  0x40008008,__READ_WRITE);
__IO_REG32(    T1PR,                  0x4000800C,__READ_WRITE);
__IO_REG32(    T1PC,                  0x40008010,__READ_WRITE);
__IO_REG32_BIT(T1MCR,                 0x40008014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T1MR0,                 0x40008018,__READ_WRITE);
__IO_REG32(    T1MR1,                 0x4000801C,__READ_WRITE);
__IO_REG32(    T1MR2,                 0x40008020,__READ_WRITE);
__IO_REG32(    T1MR3,                 0x40008024,__READ_WRITE);
__IO_REG32_BIT(T1CCR,                 0x40008028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T1CR0,                 0x4000802C,__READ);
__IO_REG32(    T1CR1,                 0x40008030,__READ);
__IO_REG32_BIT(T1EMR,                 0x4000803C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T1CTCR,                0x40008070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(T2IR,                  0x40090000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T2TCR,                 0x40090004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T2TC,                  0x40090008,__READ_WRITE);
__IO_REG32(    T2PR,                  0x4009000C,__READ_WRITE);
__IO_REG32(    T2PC,                  0x40090010,__READ_WRITE);
__IO_REG32_BIT(T2MCR,                 0x40090014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T2MR0,                 0x40090018,__READ_WRITE);
__IO_REG32(    T2MR1,                 0x4009001C,__READ_WRITE);
__IO_REG32(    T2MR2,                 0x40090020,__READ_WRITE);
__IO_REG32(    T2MR3,                 0x40090024,__READ_WRITE);
__IO_REG32_BIT(T2CCR,                 0x40090028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T2CR0,                 0x4009002C,__READ);
__IO_REG32(    T2CR1,                 0x40090030,__READ);
__IO_REG32_BIT(T2EMR,                 0x4009003C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T2CTCR,                0x40090070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(T3IR,                  0x40094000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T3TCR,                 0x40094004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T3TC,                  0x40094008,__READ_WRITE);
__IO_REG32(    T3PR,                  0x4009400C,__READ_WRITE);
__IO_REG32(    T3PC,                  0x40094010,__READ_WRITE);
__IO_REG32_BIT(T3MCR,                 0x40094014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T3MR0,                 0x40094018,__READ_WRITE);
__IO_REG32(    T3MR1,                 0x4009401C,__READ_WRITE);
__IO_REG32(    T3MR2,                 0x40094020,__READ_WRITE);
__IO_REG32(    T3MR3,                 0x40094024,__READ_WRITE);
__IO_REG32_BIT(T3CCR,                 0x40094028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T3CR0,                 0x4009402C,__READ);
__IO_REG32(    T3CR1,                 0x40094030,__READ);
__IO_REG32_BIT(T3EMR,                 0x4009403C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T3CTCR,                0x40094070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0IR,                0x40014000,__READ_WRITE ,__pwmir0_bits);
__IO_REG32_BIT(PWM0TCR,               0x40014004,__READ_WRITE ,__pwmtcr0_bits);
__IO_REG32(    PWM0TC,                0x40014008,__READ_WRITE);
__IO_REG32(    PWM0PR,                0x4001400C,__READ_WRITE);
__IO_REG32(    PWM0PC,                0x40014010,__READ_WRITE);
__IO_REG32_BIT(PWM0MCR,               0x40014014,__READ_WRITE ,__pwmmcr_bits);
__IO_REG32(    PWM0MR0,               0x40014018,__READ_WRITE);
__IO_REG32(    PWM0MR1,               0x4001401C,__READ_WRITE);
__IO_REG32(    PWM0MR2,               0x40014020,__READ_WRITE);
__IO_REG32(    PWM0MR3,               0x40014024,__READ_WRITE);
__IO_REG32_BIT(PWM0CCR,               0x40014028,__READ_WRITE ,__pwmccr0_bits);
__IO_REG32(    PWM0CR0,               0x4001402C,__READ);
__IO_REG32(    PWM0MR4,               0x40014040,__READ_WRITE);
__IO_REG32(    PWM0MR5,               0x40014044,__READ_WRITE);
__IO_REG32(    PWM0MR6,               0x40014048,__READ_WRITE);
__IO_REG32_BIT(PWM0PCR,               0x4001404C,__READ_WRITE ,__pwmpcr_bits);
__IO_REG32_BIT(PWM0LER,               0x40014050,__READ_WRITE ,__pwmler_bits);
__IO_REG32_BIT(PWM0CTCR,              0x40014070,__READ_WRITE ,__pwmctcr_bits);

/***************************************************************************
 **
 ** PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1IR,                0x40018000,__READ_WRITE ,__pwmir1_bits);
__IO_REG32_BIT(PWM1TCR,               0x40018004,__READ_WRITE ,__pwmtcr1_bits);
__IO_REG32(    PWM1TC,                0x40018008,__READ_WRITE);
__IO_REG32(    PWM1PR,                0x4001800C,__READ_WRITE);
__IO_REG32(    PWM1PC,                0x40018010,__READ_WRITE);
__IO_REG32_BIT(PWM1MCR,               0x40018014,__READ_WRITE ,__pwmmcr_bits);
__IO_REG32(    PWM1MR0,               0x40018018,__READ_WRITE);
__IO_REG32(    PWM1MR1,               0x4001801C,__READ_WRITE);
__IO_REG32(    PWM1MR2,               0x40018020,__READ_WRITE);
__IO_REG32(    PWM1MR3,               0x40018024,__READ_WRITE);
__IO_REG32_BIT(PWM1CCR,               0x40018028,__READ_WRITE ,__pwmccr1_bits);
__IO_REG32(    PWM1CR0,               0x4001802C,__READ);
__IO_REG32(    PWM1CR1,               0x40018030,__READ);
__IO_REG32(    PWM1MR4,               0x40018040,__READ_WRITE);
__IO_REG32(    PWM1MR5,               0x40018044,__READ_WRITE);
__IO_REG32(    PWM1MR6,               0x40018048,__READ_WRITE);
__IO_REG32_BIT(PWM1PCR,               0x4001804C,__READ_WRITE ,__pwmpcr_bits);
__IO_REG32_BIT(PWM1LER,               0x40018050,__READ_WRITE ,__pwmler_bits);
__IO_REG32_BIT(PWM1CTCR,              0x40018070,__READ_WRITE ,__pwmctcr_bits);

/***************************************************************************
 **
 ** Motor control PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(MCCON,                 0x400B8000,__READ       ,__mccon_bits);
__IO_REG32_BIT(MCCON_SET,             0x400B8004,__WRITE      ,__mccon_bits);
__IO_REG32_BIT(MCCON_CLR,             0x400B8008,__WRITE      ,__mccon_bits);
__IO_REG32_BIT(MCCAPCON,              0x400B800C,__READ       ,__mccapcon_bits);
__IO_REG32_BIT(MCCAPCON_SET,          0x400B8010,__WRITE      ,__mccapcon_bits);
__IO_REG32_BIT(MCCAPCON_CLR,          0x400B8014,__WRITE      ,__mccapcon_bits);
__IO_REG32(    MCTC0,                 0x400B8018,__READ_WRITE );
__IO_REG32(    MCTC1,                 0x400B801C,__READ_WRITE );
__IO_REG32(    MCTC2,                 0x400B8020,__READ_WRITE );
__IO_REG32(    MCLIM0,                0x400B8024,__READ_WRITE );
__IO_REG32(    MCLIM1,                0x400B8028,__READ_WRITE );
__IO_REG32(    MCLIM2,                0x400B802C,__READ_WRITE );
__IO_REG32(    MCMAT0,                0x400B8030,__READ_WRITE );
__IO_REG32(    MCMAT1,                0x400B8034,__READ_WRITE );
__IO_REG32(    MCMAT2,                0x400B8038,__READ_WRITE );
__IO_REG32_BIT(MCDT,                  0x400B803C,__READ_WRITE ,__mcdt_bits);
__IO_REG32_BIT(MCCP,                  0x400B8040,__READ_WRITE ,__mcccp_bits);
__IO_REG32(    MCCAP0,                0x400B8044,__READ       );
__IO_REG32(    MCCAP1,                0x400B8048,__READ       );
__IO_REG32(    MCCAP2,                0x400B804C,__READ       );
__IO_REG32_BIT(MCINTEN,               0x400B8050,__READ       ,__mcinten_bits);
__IO_REG32_BIT(MCINTEN_SET,           0x400B8054,__WRITE      ,__mcinten_bits);
__IO_REG32_BIT(MCINTEN_CLR,           0x400B8058,__WRITE      ,__mcinten_bits);
__IO_REG32_BIT(MCCNTCON,              0x400B805C,__READ       ,__mccntcon_bits);
__IO_REG32_BIT(MCCNTCON_SET,          0x400B8060,__WRITE      ,__mccntcon_bits);
__IO_REG32_BIT(MCCNTCON_CLR,          0x400B8064,__WRITE      ,__mccntcon_bits);
__IO_REG32_BIT(MCINTF,                0x400B8068,__READ       ,__mcinten_bits);
__IO_REG32_BIT(MCINTF_SET,            0x400B806C,__WRITE      ,__mcinten_bits);
__IO_REG32_BIT(MCINTF_CLR,            0x400B8070,__WRITE      ,__mcinten_bits);
__IO_REG32_BIT(MCCAP_CLR,             0x400B8074,__WRITE      ,__mccap_clr_bits);

/***************************************************************************
 **
 ** Quadrature Encoder Interface
 **
 ***************************************************************************/
__IO_REG32_BIT(QEICON,                0x400BC000,__WRITE      ,__qeicon_bits);
__IO_REG32_BIT(QEISTAT,               0x400BC004,__READ       ,__qeistat_bits);
__IO_REG32_BIT(QEICONF,               0x400BC008,__READ_WRITE ,__qeiconf_bits);
__IO_REG32(    QEIPOS,                0x400BC00C,__READ       );
__IO_REG32(    QEIMAXPSOS,            0x400BC010,__READ_WRITE );
__IO_REG32(    CMPOS0,                0x400BC014,__READ_WRITE );
__IO_REG32(    CMPOS1,                0x400BC018,__READ_WRITE );
__IO_REG32(    CMPOS2,                0x400BC01C,__READ_WRITE );
__IO_REG32(    INXCNT,                0x400BC020,__READ       );
__IO_REG32(    INXCMP0,               0x400BC024,__READ_WRITE );
__IO_REG32(    QEILOAD,               0x400BC028,__READ_WRITE );
__IO_REG32(    QEITIME,               0x400BC02C,__READ       );
__IO_REG32(    QEIVEL,                0x400BC030,__READ       );
__IO_REG32(    QEICAP,                0x400BC034,__READ       );
__IO_REG32(    VELCOMP,               0x400BC038,__READ_WRITE );
__IO_REG32(    FILTERPHA,             0x400BC03C,__READ_WRITE );
__IO_REG32(    FILTERPHB,             0x400BC040,__READ_WRITE );
__IO_REG32(    FILTERINX,             0x400BC044,__READ_WRITE );
__IO_REG32(    WINDOW,                0x400BC048,__READ_WRITE );
__IO_REG32(    INXCMP1,               0x400BC04C,__READ_WRITE );
__IO_REG32(    INXCMP2,               0x400BC050,__READ_WRITE );
__IO_REG32_BIT(QEIIEC,                0x400BCFD8,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEIIES,                0x400BCFDC,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEIINTSTAT,            0x400BCFE0,__READ       ,__qeiintstat_bits);
__IO_REG32_BIT(QEIIE,                 0x400BCFE4,__READ       ,__qeiintstat_bits);
__IO_REG32_BIT(QEICLR,                0x400BCFE8,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEISET,                0x400BCFEC,__WRITE      ,__qeiintstat_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTCILR,                0x40024000,__READ_WRITE ,__ilr_bits);
__IO_REG32_BIT(RTCCCR,                0x40024008,__READ_WRITE ,__rtcccr_bits);
__IO_REG32_BIT(RTCCIIR,               0x4002400C,__READ_WRITE ,__ciir_bits);
__IO_REG32_BIT(RTCAMR,                0x40024010,__READ_WRITE ,__amr_bits);
__IO_REG32_BIT(RTCCTIME0,             0x40024014,__READ       ,__ctime0_bits);
__IO_REG32_BIT(RTCCTIME1,             0x40024018,__READ       ,__ctime1_bits);
__IO_REG32_BIT(RTCCTIME2,             0x4002401C,__READ       ,__ctime2_bits);
__IO_REG32_BIT(RTCSEC,                0x40024020,__READ_WRITE ,__sec_bits);
__IO_REG32_BIT(RTCMIN,                0x40024024,__READ_WRITE ,__min_bits);
__IO_REG32_BIT(RTCHOUR,               0x40024028,__READ_WRITE ,__hour_bits);
__IO_REG32_BIT(RTCDOM,                0x4002402C,__READ_WRITE ,__dom_bits);
__IO_REG32_BIT(RTCDOW,                0x40024030,__READ_WRITE ,__dow_bits);
__IO_REG32_BIT(RTCDOY,                0x40024034,__READ_WRITE ,__doy_bits);
__IO_REG32_BIT(RTCMONTH,              0x40024038,__READ_WRITE ,__month_bits);
__IO_REG32_BIT(RTCYEAR,               0x4002403C,__READ_WRITE ,__year_bits);
__IO_REG32_BIT(RTCCALIBRATION,        0x40024040,__READ_WRITE ,__calibration_bits);
__IO_REG32(    RTCGPREG0,             0x40024044,__READ_WRITE );
__IO_REG32(    RTCGPREG1,             0x40024048,__READ_WRITE );
__IO_REG32(    RTCGPREG2,             0x4002404C,__READ_WRITE );
__IO_REG32(    RTCGPREG3,             0x40024050,__READ_WRITE );
__IO_REG32(    RTCGPREG4,             0x40024054,__READ_WRITE );
__IO_REG32_BIT(RTCAUXEN,              0x40024058,__READ_WRITE ,__rtcauxen_bits);
__IO_REG32_BIT(RTCAUX,                0x4002405C,__READ_WRITE ,__rtcaux_bits);
__IO_REG32_BIT(RTCALSEC,              0x40024060,__READ_WRITE ,__sec_bits);
__IO_REG32_BIT(RTCALMIN,              0x40024064,__READ_WRITE ,__min_bits);
__IO_REG32_BIT(RTCALHOUR,             0x40024068,__READ_WRITE ,__hour_bits);
__IO_REG32_BIT(RTCALDOM,              0x4002406C,__READ_WRITE ,__dom_bits);
__IO_REG32_BIT(RTCALDOW,              0x40024070,__READ_WRITE ,__dow_bits);
__IO_REG32_BIT(RTCALDOY,              0x40024074,__READ_WRITE ,__doy_bits);
__IO_REG32_BIT(RTCALMON,              0x40024078,__READ_WRITE ,__month_bits);
__IO_REG32_BIT(RTCALYEAR,             0x4002407C,__READ_WRITE ,__year_bits);

/***************************************************************************
 **
 ** Watchdog
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,                 0x40000000,__READ_WRITE ,__wdmod_bits);
__IO_REG32_BIT(WDTC,                  0x40000004,__READ_WRITE ,__wdtc_bits);
__IO_REG32_BIT(WDFEED,                0x40000008,__WRITE      ,__wdfeed_bits);
__IO_REG32_BIT(WDTV,                  0x4000000C,__READ       ,__wdtc_bits);
__IO_REG32_BIT(WDWARNINT,             0x40000014,__READ_WRITE ,__wdwarnint_bits);
__IO_REG32_BIT(WDWINDOW,              0x40000018,__READ_WRITE ,__wdwindow_bits);

/***************************************************************************
 **
 ** A/D Converters
 **
 ***************************************************************************/
__IO_REG32_BIT(AD0CR,                 0x40034000,__READ_WRITE ,__adcr_bits);
__IO_REG32_BIT(AD0GDR,                0x40034004,__READ_WRITE ,__adgdr_bits);
__IO_REG32_BIT(ADINTEN,               0x4003400C,__READ_WRITE ,__adinten_bits);
__IO_REG32_BIT(ADDR0,                 0x40034010,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR1,                 0x40034014,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR2,                 0x40034018,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR3,                 0x4003401C,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR4,                 0x40034020,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR5,                 0x40034024,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR6,                 0x40034028,__READ       ,__addr_bits);
__IO_REG32_BIT(ADDR7,                 0x4003402C,__READ       ,__addr_bits);
__IO_REG32_BIT(ADSTAT,                0x40034030,__READ       ,__adstat_bits);
__IO_REG32_BIT(ADTRM,                 0x40034034,__READ_WRITE ,__adtrm_bits);

/***************************************************************************
 **
 ** D/A Converter
 **
 ***************************************************************************/
__IO_REG32_BIT(DACR,                  0x4008C000,__READ_WRITE ,__dacr_bits);
__IO_REG32_BIT(DACCTRL,               0x4008C004,__READ_WRITE ,__dacctrl_bits);
__IO_REG32_BIT(DACCNTVAL,             0x4008C008,__READ_WRITE ,__daccntval_bits);

/***************************************************************************
 **
 ** GPDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACINTSTATUS,         0x20080000,__READ      ,__dmacintstatus_bits);
__IO_REG32_BIT(DMACINTTCSTATUS,       0x20080004,__READ      ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACINTTCCLEAR,        0x20080008,__WRITE     ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACINTERRSTAT,        0x2008000C,__READ      ,__dmacinterrstat_bits);
__IO_REG32_BIT(DMACINTERRCLR,         0x20080010,__WRITE     ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRAWINTTCSTATUS,    0x20080014,__READ      ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRAWINTERRORSTATUS, 0x20080018,__READ      ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACENBLDCHNS,         0x2008001C,__READ      ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSOFTBREQ,          0x20080020,__READ_WRITE,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSOFTSREQ,          0x20080024,__READ_WRITE,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACSOFTLBREQ,         0x20080028,__READ_WRITE,__dmacsoftlbreq_bits);
__IO_REG32_BIT(DMACSOFTLSREQ,         0x2008002C,__READ_WRITE,__dmacsoftlsreq_bits);
__IO_REG32_BIT(DMACCONFIGURATION,     0x20080030,__READ_WRITE,__dmacconfig_bits);
__IO_REG32_BIT(DMACSYNC,              0x20080034,__READ_WRITE,__dmacsync_bits);
__IO_REG32_BIT(DMAREQSEL,             0x400FC1C4,__READ_WRITE,__dmareqsel_bits);
__IO_REG32(    DMACC0SRCADDR,         0x20080100,__READ_WRITE);
__IO_REG32(    DMACC0DESTADDR,        0x20080104,__READ_WRITE);
__IO_REG32(    DMACC0LLI,             0x20080108,__READ_WRITE);
__IO_REG32_BIT(DMACC0CONTROL,         0x2008010C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC0CONFIGURATION,   0x20080110,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC1SRCADDR,         0x20080120,__READ_WRITE);
__IO_REG32(    DMACC1DESTADDR,        0x20080124,__READ_WRITE);
__IO_REG32(    DMACC1LLI,             0x20080128,__READ_WRITE);
__IO_REG32_BIT(DMACC1CONTROL,         0x2008012C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC1CONFIGURATION,   0x20080130,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC2SRCADDR,         0x20080140,__READ_WRITE);
__IO_REG32(    DMACC2DESTADDR,        0x20080144,__READ_WRITE);
__IO_REG32(    DMACC2LLI,             0x20080148,__READ_WRITE);
__IO_REG32_BIT(DMACC2CONTROL,         0x2008014C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC2CONFIGURATION,   0x20080150,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC3SRCADDR,         0x20080160,__READ_WRITE);
__IO_REG32(    DMACC3DESTADDR,        0x20080164,__READ_WRITE);
__IO_REG32(    DMACC3LLI,             0x20080168,__READ_WRITE);
__IO_REG32_BIT(DMACC3CONTROL,         0x2008016C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC3CONFIGURATION,   0x20080170,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC4SRCADDR,         0x20080180,__READ_WRITE);
__IO_REG32(    DMACC4DESTADDR,        0x20080184,__READ_WRITE);
__IO_REG32(    DMACC4LLI,             0x20080188,__READ_WRITE);
__IO_REG32_BIT(DMACC4CONTROL,         0x2008018C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC4CONFIGURATION,   0x20080190,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC5SRCADDR,         0x200801A0,__READ_WRITE);
__IO_REG32(    DMACC5DESTADDR,        0x200801A4,__READ_WRITE);
__IO_REG32(    DMACC5LLI,             0x200801A8,__READ_WRITE);
__IO_REG32_BIT(DMACC5CONTROL,         0x200801AC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC5CONFIGURATION,   0x200801B0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC6SRCADDR,         0x200801C0,__READ_WRITE);
__IO_REG32(    DMACC6DESTADDR,        0x200801C4,__READ_WRITE);
__IO_REG32(    DMACC6LLI,             0x200801C8,__READ_WRITE);
__IO_REG32_BIT(DMACC6CONTROL,         0x200801CC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC6CONFIGURATION,   0x200801D0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC7SRCADDR,         0x200801E0,__READ_WRITE);
__IO_REG32(    DMACC7DESTADDR,        0x200801E4,__READ_WRITE);
__IO_REG32(    DMACC7LLI,             0x200801E8,__READ_WRITE);
__IO_REG32_BIT(DMACC7CONTROL,         0x200801EC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC7CONFIGURATION,   0x200801F0,__READ_WRITE,__dma_cfg_bits);

/***************************************************************************
 **
 ** CRC engine
 **
 ***************************************************************************/
__IO_REG32_BIT(CRC_MODE,              0x20090000,__READ_WRITE ,__crc_mode_bits);
__IO_REG32(    CRC_SEED,              0x20090004,__READ_WRITE );
__IO_REG32(    CRC_SUM,               0x20090008,__READ_WRITE );
#define CRC_WR_DATA CRC_SUM

/***************************************************************************
 **
 ** EEPROM
 **
 ***************************************************************************/
__IO_REG32_BIT(EECMD,                 0x00200080,__READ_WRITE ,__eecmd_bits);
__IO_REG32_BIT(EEADDR,                0x00200084,__READ_WRITE ,__eeaddr_bits);
__IO_REG32(    EEWDATA,               0x00200088,__WRITE      );
__IO_REG32(    EERDATA,               0x0020008C,__READ       );
__IO_REG32_BIT(EEWSTATE,              0x00200090,__READ_WRITE ,__eewstate_bits);
__IO_REG32_BIT(EECLKDIV,              0x00200094,__READ_WRITE ,__eeclkdiv_bits);
__IO_REG32_BIT(EEPWRDWN,              0x00200098,__READ_WRITE ,__eepwrdwn_bits);
__IO_REG32_BIT(EEINTENCLR,            0x00200FD8,__WRITE      ,__eeintenclr_bits);
__IO_REG32_BIT(EEINTENSET,            0x00200FDC,__WRITE      ,__eeintenset_bits);
__IO_REG32_BIT(EEINTSTAT,             0x00200FE0,__READ       ,__eeintstat_bits);
__IO_REG32_BIT(EEINTEN,               0x00200FE4,__READ       ,__eeinten_bits);
__IO_REG32_BIT(EEINTSTATCLR,          0x00200FE8,__WRITE      ,__eeintstatclr_bits);
__IO_REG32_BIT(EEINTSTATSET,          0x00200FEC,__WRITE      ,__eeintstatset_bits);

/***************************************************************************
 **
 ** Flash signature generation
 **
 ***************************************************************************/
__IO_REG32_BIT(FMSSTART,              0x00200020,__READ_WRITE ,__fmsstart_bits);
__IO_REG32_BIT(FMSSTOP,               0x00200024,__READ_WRITE ,__fmsstop_bits);
__IO_REG32(    FMSW0,                 0x0020002C,__READ       );
__IO_REG32(    FMSW1,                 0x00200030,__READ       );
__IO_REG32(    FMSW2,                 0x00200034,__READ       );
__IO_REG32(    FMSW3,                 0x00200038,__READ       );
#define FMSTAT          EEINTSTAT
#define FMSTAT_bit      EEINTSTAT_bit
#define FMSTATCLR       EEINTSTATCLR
#define FMSTATCLR_bit   EEINTSTATCLR_bit

/***************************************************************************
 **
 ** Event Monitor/Recorder
 **
 ***************************************************************************/
__IO_REG32_BIT(ERSTATUS,              0x40024080,__READ_WRITE ,__erstatus_bits);
__IO_REG32_BIT(ERCONTROL,             0x40024084,__READ_WRITE ,__ercontrol_bits);
__IO_REG32_BIT(ERCOUNTERS,            0x40024088,__READ       ,__ercounters_bits);
__IO_REG32_BIT(ERFIRSTSTAMP0,         0x40024090,__READ       ,__erfirststamp_bits);
__IO_REG32_BIT(ERFIRSTSTAMP1,         0x40024094,__READ       ,__erfirststamp_bits);
__IO_REG32_BIT(ERFIRSTSTAMP2,         0x40024098,__READ       ,__erfirststamp_bits);
__IO_REG32_BIT(ERLASTSTAMP0,          0x400240A0,__READ       ,__erfirststamp_bits);
__IO_REG32_BIT(ERLASTSTAMP1,          0x400240A4,__READ       ,__erfirststamp_bits);
__IO_REG32_BIT(ERLASTSTAMP2,          0x400240A8,__READ       ,__erfirststamp_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 **  GPDMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define GPDMA_T0_MAT0           0   /* T0_MAT[0]                               */
#define GPDMA_SD_TO_MAT1        1   /* SD card/T0_MAT[1]                       */
#define GPDMA_SSP0TX_T1_MAT0    2   /* SPI0 Tx/T1_MAT[0]                       */
#define GPDMA_SSP0RX_T1_MAT1    3   /* SPI0 Rx/T1_MAT[1]                       */
#define GPDMA_SSP1TX_T2_MAT0    4   /* SPI1 Tx/T2_MAT[0]                       */
#define GPDMA_SSP1RX_T2_MAT1    5   /* SPI1 Rx/T2_MAT[1]                       */
#define GPDMA_SSP2TX_I2S_CH0    6   /* SPI2 Tx/I2S channel 0                   */
#define GPDMA_SSP2RX_I2S_CH1    7   /* SPI2 Rx/I2S channel 1                   */
#define GPDMA_ADC               8   /* ADC                                     */
#define GPDMA_DAC               9   /* DAC                                     */
#define GPDMA_U0TX_U3TX        10   /* UART0 Tx/UART3 Tx                       */
#define GPDMA_U0RX_U3RX        11   /* UART0 Rx/UART3 Tx                       */
#define GPDMA_U1TX_U4TX        12   /* UART1 Tx/UART4 Tx                       */
#define GPDMA_U1RX_U3RX        13   /* UART1 Rx/UART4 Rx                       */
#define GPDMA_U2TX_MAT3_0      14   /* UART2 Tx/MAT3.0                         */
#define GPDMA_U2RX_MAT3_1      15   /* UART2 Rx/MAT3.1                         */

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
#define NVIC_WDT              16  /* Watchdog Interrupt (WDINT)                             */
#define NVIC_TIMER0           17  /* Match 0 - 1 (MR0, MR1), Capture 0 - 1 (CR0, CR1)       */
#define NVIC_TIMER1           18  /* Match 0 - 2 (MR0, MR1, MR2), Capture 0 - 1 (CR0, CR1)  */
#define NVIC_TIMER2           19  /* Match 0-3, Capture 0-1                                 */
#define NVIC_TIMER3           20  /* Match 0-3, Capture 0-1                                 */
#define NVIC_UART0            21  /* UART0                                                  */
#define NVIC_UART1            22  /* UART1                                                  */
#define NVIC_UART2            23  /* UART2                                                  */
#define NVIC_UART3            24  /* UART3                                                  */
#define NVIC_PWM1             25  /* Match 0 - 6 of PWM1, Capture 0-1 of PWM1               */
#define NVIC_I2C0             26  /* SI (state change)                                      */
#define NVIC_I2C1             27  /* SI (state change)                                      */
#define NVIC_I2C2             28  /* SI (state change)                                      */
#define NVIC_SSP0             30  /* SSP0                                                   */
#define NVIC_SSP1             31  /* SSP1                                                   */
#define NVIC_PLL0             32  /* PLL0 Lock                                              */
#define NVIC_RTC              33  /* Counter Increment (RTCCIF), Alarm (RTCALF)             */
#define NVIC_EINT0            34  /* External Interrupt 0 (EINT0)                           */
#define NVIC_EINT1            35  /* External Interrupt 1 (EINT1)                           */
#define NVIC_EINT2            36  /* External Interrupt 2 (EINT2)                           */
#define NVIC_EINT3            37  /* External Interrupt 3 (EINT3)                           */
#define NVIC_ADC              38  /* A/D Converter end of conversion                        */
#define NVIC_BOD              39  /* Brown Out detect                                       */
#define NVIC_USB              40  /* USB                                                    */
#define NVIC_CAN              41  /* CAN Common, CAN 0 Tx, CAN 0 Rx, CAN 1 Tx, CAN 1 Rx     */
#define NVIC_GP_DMA           42  /* IntStatus of DMA channel 0, IntStatus of DMA channel 1 */
#define NVIC_I2S              43  /* irq, dmareq1, dmareq2                                  */
#define NVIC_SD               45  /* SD Card Interface                                      */
#define NVIC_MC               46  /* Motor Control PWM                                      */
#define NVIC_QE               47  /* Quadrature Encoder                                     */
#define NVIC_PLL1             48  /* PLL1 Lock                                              */
#define NVIC_USB_ACT          49  /* USB Activity Interrupt                                 */
#define NVIC_CAN_ACT          50  /* CAN Activity Interrupt                                 */
#define NVIC_UART4            51  /* UART1                                                  */
#define NVIC_SSP2             52  /* SSP2                                                   */
#define NVIC_GPIO             54  /* GPIO interrupts                                        */
#define NVIC_PWM0             55  /* PWM0                                                   */
#define NVIC_EEPROM           56  /* EEPROM                                                 */

#endif    /* __IOLPC1777_H */

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
Interrupt9   = WDT            0x40
Interrupt10  = TIMER0         0x44
Interrupt11  = TIMER1         0x48
Interrupt12  = TIMER2         0x4C
Interrupt13  = TIMER3         0x50
Interrupt14  = UART0          0x54
Interrupt15  = UART1          0x58
Interrupt16  = UART2          0x5C
Interrupt17  = UART3          0x60
Interrupt18  = PWM1           0x64
Interrupt19  = I2C0           0x68
Interrupt20  = I2C1           0x6C
Interrupt21  = I2C2           0x70
Interrupt22  = SSP0           0x78
Interrupt23  = SSP1           0x7C
Interrupt24  = PLL0           0x80
Interrupt25  = RTC            0x84
Interrupt26  = EINT0          0x88
Interrupt27  = EINT1          0x8C
Interrupt28  = EINT2          0x90
Interrupt29  = EINT3          0x94
Interrupt30  = ADC            0x98
Interrupt31  = BOR            0x9C
Interrupt32  = USB            0xA0
Interrupt33  = CAN            0xA4
Interrupt34  = GPDMA          0xA8
Interrupt35  = I2S            0xAC
Interrupt36  = SD             0xB4
Interrupt37  = MC             0xB8
Interrupt38  = QE             0xBC
Interrupt39  = PLL1           0xC0
Interrupt40  = USB_ACT        0xC4
Interrupt41  = CAN_ACT        0xC8
Interrupt42  = UART4          0xCC
Interrupt43  = SSP2           0xD0
Interrupt44  = GPIO           0xD8
Interrupt45  = PWM0           0xDC
Interrupt46  = EEPROM         0xE0

###DDF-INTERRUPT-END###*/
