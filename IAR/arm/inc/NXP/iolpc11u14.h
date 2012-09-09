/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC11U14
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 54707 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __LPC11U14_H
#define __LPC11U14_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC11U14 SPECIAL FUNCTION REGISTERS
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

/*
 Used for registers:
 TMR16B1CCR
 TMR16B0CCR
*/
typedef struct {
  __REG32 CAP0RE               : 1;
  __REG32 CAP0FE               : 1;
  __REG32 CAP0I                : 1;
  __REG32                      :29;
} __sfrdef0_bits;

/*
 Used for registers:
 IST
*/
typedef struct {
  __REG32 PSTAT0               : 1;
  __REG32 PSTAT1               : 1;
  __REG32 PSTAT2               : 1;
  __REG32 PSTAT3               : 1;
  __REG32 PSTAT4               : 1;
  __REG32 PSTAT5               : 1;
  __REG32 PSTAT6               : 1;
  __REG32 PSTAT7               : 1;
  __REG32                      :24;
} __sfrdef1_bits;

/*
 Used for registers:
 CTRL0
 CTRL1
*/
typedef struct {
  __REG32 INT                  : 1;
  __REG32 COMB                 : 1;
  __REG32 TRIG                 : 1;
  __REG32                      :29;
} __sfrdef2_bits;

/*
 Used for registers:
 W23
 W56
 W24
 W57
 W25
 W58
 W26
 W59
 W27
 W60
 W28
 W61
 W29
 W62
 W30
 W63
 W31
 W32
 W0
 W33
 W1
 W34
 W2
 W35
 W3
 W36
 W4
 W37
 W5
 W38
 W6
 W39
 W7
 W40
 W8
 W41
 W9
 W42
 W10
 W43
 W11
 W44
 W12
 W45
 W13
 W46
 W14
 W47
 W15
 W48
 W16
 W49
 W17
 W50
 W18
 W51
 W19
 W52
 W20
 W53
 W21
 W54
 W22
 W55
*/
typedef struct {
  __REG32 PWORD                :32;
} __sfrdef3_bits;

/*
 Used for registers:
 TMR32B0MR2
 TMR32B1MR0
 TMR32B0MR3
 TMR32B1MR1
 TMR32B1MR2
 TMR32B1MR3
 TMR32B0MR0
 TMR32B0MR1
*/
typedef struct {
  __REG32 MATCH                :32;
} __sfrdef4_bits;

/*
 Used for registers:
 EPTOGGLE
*/
typedef struct {
  __REG32 TOGGLE0              : 1;
  __REG32 TOGGLE1              : 1;
  __REG32 TOGGLE2              : 1;
  __REG32 TOGGLE3              : 1;
  __REG32 TOGGLE4              : 1;
  __REG32 TOGGLE5              : 1;
  __REG32 TOGGLE6              : 1;
  __REG32 TOGGLE7              : 1;
  __REG32 TOGGLE8              : 1;
  __REG32 TOGGLE9              : 1;
  __REG32                      :22;
} __sfrdef5_bits;

/*
 Used for registers:
 TRST_PIO0_14
 SWDIO_PIO0_15
 PIO0_16
 PIO0_22
 PIO0_23
 PIO1_5
 TDI_PIO0_11
 TMS_PIO0_12
 TDO_PIO0_13
*/
typedef struct {
  __REG32 FUNC                 : 3;
  __REG32 MODE                 : 2;
  __REG32 HYS                  : 1;
  __REG32 INV                  : 1;
  __REG32 ADMODE               : 1;
  __REG32 FILTR                : 1;
  __REG32                      : 1;
  __REG32 OD                   : 1;
  __REG32                      :21;
} __sfrdef6_bits;

/*
 Used for registers:
 PIO1_23
 PIO1_24
 PIO1_25
 PIO0_17
 PIO1_26
 PIO0_18
 PIO1_27
 PIO0_19
 PIO1_28
 PIO0_20
 PIO1_29
 PIO0_21
 RESET_PIO0_0
 PIO0_1
 PIO0_2
 PIO0_3
 PIO1_13
 PIO1_14
 PIO0_6
 PIO1_15
 PIO0_7
 PIO1_16
 PIO0_8
 PIO0_9
 SWCLK_PIO0_10
 PIO1_19
 PIO1_20
 PIO1_21
 PIO1_22
*/
typedef struct {
  __REG32 FUNC                 : 3;
  __REG32 MODE                 : 2;
  __REG32 HYS                  : 1;
  __REG32 INV                  : 1;
  __REG32                      : 3;
  __REG32 OD                   : 1;
  __REG32                      :21;
} __sfrdef7_bits;

/*
 Used for registers:
 SYSRSTSTAT
*/
typedef struct {
  __REG32 POR                  : 1;
  __REG32 EXTRST               : 1;
  __REG32 WDT                  : 1;
  __REG32 BOD                  : 1;
  __REG32 SYSRST               : 1;
  __REG32                      :27;
} __sfrdef8_bits;

/*
 Used for registers:
 DR0
 DR1
 DR2
 DR3
 DR4
 DR5
 DR6
 DR7
*/
typedef struct {
  __REG32                      : 6;
  __REG32 V_VREF               :10;
  __REG32                      :14;
  __REG32 OVERRUN              : 1;
  __REG32 DONE                 : 1;
} __sfrdef9_bits;

/*
 Used for registers:
 SSP1SR
 SSP0SR
*/
typedef struct {
  __REG32 TFE                  : 1;
  __REG32 TNF                  : 1;
  __REG32 RNE                  : 1;
  __REG32 RFF                  : 1;
  __REG32 BSY                  : 1;
  __REG32                      :27;
} __sfrdef10_bits;

/*
 Used for registers:
 TMR16B1CR0
 TMR16B0CR0
*/
typedef struct {
  __REG32 CAP                  :16;
  __REG32                      :16;
} __sfrdef11_bits;

/*
 Used for registers:
 B0
 B1
 B2
 B3
 B4
 B5
 B6
 B7
 B8
 B9
 B10
 B11
 B12
 B13
 B14
 B15
 B16
 B17
 B18
 B19
 B20
 B21
 B22
 B23
 B24
 B25
 B26
 B27
 B28
 B29
 B30
 B31
 B32
 B33
 B34
 B35
 B36
 B37
 B38
 B39
 B40
 B41
 B42
 B43
 B44
 B45
 B46
 B47
 B48
 B49
 B50
 B51
 B52
 B53
 B54
 B55
 B56
 B57
 B58
 B59
 B60
 B61
 B62
 B63
*/
typedef struct {
  __REG8 PBYTE                 : 1;
  __REG8                       : 7;
} __sfrdef12_bits;

/*
 Used for registers:
 SSP1CPSR
 SSP0CPSR
*/
typedef struct {
  __REG32 CPSDVSR              : 8;
  __REG32                      :24;
} __sfrdef13_bits;

/*
 Used for registers:
 TMR32B0CCR
 TMR32B1CCR
*/
typedef struct {
  __REG32 CAP0RE               : 1;
  __REG32 CAP0FE               : 1;
  __REG32 CAP0I                : 1;
  __REG32 CAP1RE               : 1;
  __REG32 CAP1FE               : 1;
  __REG32 CAP1I                : 1;
  __REG32                      :26;
} __sfrdef14_bits;

/*
 Used for registers:
 TMR16B0EMR
 TMR16B1EMR
 TMR32B0EMR
 TMR32B1EMR
*/
typedef struct {
  __REG32 EM0                  : 1;
  __REG32 EM1                  : 1;
  __REG32 EM2                  : 1;
  __REG32 EM3                  : 1;
  __REG32 EMC0                 : 2;
  __REG32 EMC1                 : 2;
  __REG32 EMC2                 : 2;
  __REG32 EMC3                 : 2;
  __REG32                      :20;
} __sfrdef15_bits;

/*
 Used for registers:
 SSP1IMSC
 SSP0IMSC
*/
typedef struct {
  __REG32 RORIM                : 1;
  __REG32 RTIM                 : 1;
  __REG32 RXIM                 : 1;
  __REG32 TXIM                 : 1;
  __REG32                      :28;
} __sfrdef16_bits;

/*
 Used for registers:
 TMR32B0CR0
 TMR32B1CR0
 TMR32B1CR1
*/
typedef struct {
  __REG32 CAP                  :32;
} __sfrdef17_bits;

/*
 Used for registers:
 SCICTRL
*/
typedef struct {
  __REG32 SCIEN                : 1;
  __REG32 NACKDIS              : 1;
  __REG32 PROTSEL              : 1;
  __REG32                      : 2;
  __REG32 TXRETRY              : 3;
  __REG32 XTRAGUARD            : 8;
  __REG32                      :16;
} __sfrdef18_bits;

/*
 Used for registers:
 SSP1RIS
 SSP0RIS
*/
typedef struct {
  __REG32 RORRIS               : 1;
  __REG32 RTRIS                : 1;
  __REG32 RXRIS                : 1;
  __REG32 TXRIS                : 1;
  __REG32                      :28;
} __sfrdef19_bits;

/*
 Used for registers:
 USBCLKSEL
*/
typedef struct {
  __REG32 SEL                  : 2;
  __REG32                      :30;
} __sfrdef20_bits;

/*
 Used for registers:
 RS485CTRL
*/
typedef struct {
  __REG32 NMMEN                : 1;
  __REG32 RXDIS                : 1;
  __REG32 AADEN                : 1;
  __REG32 SEL                  : 1;
  __REG32 DCTRL                : 1;
  __REG32 OINV                 : 1;
  __REG32                      :26;
} __sfrdef21_bits;

/*
 Used for registers:
 SYSPLLCLKSEL
 USBPLLCLKSEL
 CLKOUTSEL
 MAINCLKSEL
*/
typedef struct {
  __REG32 SEL                  : 2;
  __REG32                      :30;
} __sfrdef22_bits;

/*
 Used for registers:
 SSP1MIS
 SSP0MIS
*/
typedef struct {
  __REG32 RORMIS               : 1;
  __REG32 RTMIS                : 1;
  __REG32 RXMIS                : 1;
  __REG32 TXMIS                : 1;
  __REG32                      :28;
} __sfrdef23_bits;

/*
 Used for registers:
 USBCLKUEN
 SYSPLLCLKUEN
 USBPLLCLKUEN
 CLKOUTUEN
 MAINCLKUEN
*/
typedef struct {
  __REG32 ENA                  : 1;
  __REG32                      :31;
} __sfrdef24_bits;

/*
 Used for registers:
 RS485ADRMATCH
*/
typedef struct {
  __REG32 ADRMATCH             : 8;
  __REG32                      :24;
} __sfrdef25_bits;

/*
 Used for registers:
 SSP1ICR
 SSP0ICR
*/
#define SSP0ICR_RORIC   0x00000001
#define SSP0ICR_RTIC    0x00000002

#define SSP1ICR_RORIC   SSP0ICR_RORIC
#define SSP1ICR_RTIC    SSP0ICR_RTIC

/*
 Used for registers:
 USBCLKDIV
 CLKOUTDIV
 SYSAHBCLKDIV
 SSP0CLKDIV
 UARTCLKDIV
 SSP1CLKDIV
*/
typedef struct {
  __REG32 DIV                  : 8;
  __REG32                      :24;
} __sfrdef27_bits;

/*
 Used for registers:
 RS485DLY
*/
typedef struct {
  __REG32 DLY                  : 8;
  __REG32                      :24;
} __sfrdef28_bits;

/*
 Used for registers:
 IPR0
*/
typedef struct {
  __REG32 PRI_0                : 8;
  __REG32 PRI_1                : 8;
  __REG32 PRI_2                : 8;
  __REG32 PRI_3                : 8;
} __sfrdef29_bits;

/*
 Used for registers:
 BODCTRL
*/
typedef struct {
  __REG32 BODRSTLEV            : 2;
  __REG32 BODINTVAL            : 2;
  __REG32 BODRSTENA            : 1;
  __REG32                      :27;
} __sfrdef30_bits;

/*
 Used for registers:
 SYNCCTRL
*/
typedef struct {
  __REG32 SYNC                 : 1;
  __REG32 CSRC                 : 1;
  __REG32 FES                  : 1;
  __REG32 TSBYPASS             : 1;
  __REG32 CSCEN                : 1;
  __REG32 SSDIS                : 1;
  __REG32 CCCLR                : 1;
  __REG32                      :25;
} __sfrdef31_bits;

/*
 Used for registers:
 PORT0_POL0
 PORT0_POL1
 PORT1_POL0
 PORT1_POL1
*/
typedef struct {
  __REG32 POL0                 : 1;
  __REG32 POL1                 : 1;
  __REG32 POL2                 : 1;
  __REG32 POL3                 : 1;
  __REG32 POL4                 : 1;
  __REG32 POL5                 : 1;
  __REG32 POL6                 : 1;
  __REG32 POL7                 : 1;
  __REG32 POL8                 : 1;
  __REG32 POL9                 : 1;
  __REG32 POL10                : 1;
  __REG32 POL11                : 1;
  __REG32 POL12                : 1;
  __REG32 POL13                : 1;
  __REG32 POL14                : 1;
  __REG32 POL15                : 1;
  __REG32 POL16                : 1;
  __REG32 POL17                : 1;
  __REG32 POL18                : 1;
  __REG32 POL19                : 1;
  __REG32 POL20                : 1;
  __REG32 POL21                : 1;
  __REG32 POL22                : 1;
  __REG32 POL23                : 1;
  __REG32 POL24                : 1;
  __REG32 POL25                : 1;
  __REG32 POL26                : 1;
  __REG32 POL27                : 1;
  __REG32 POL28                : 1;
  __REG32 POL29                : 1;
  __REG32 POL30                : 1;
  __REG32 POL31                : 1;
} __sfrdef32_bits;

/*
 Used for registers:
 IPR1
*/
typedef struct {
  __REG32 PRI_4                : 8;
  __REG32 PRI_5                : 8;
  __REG32 PRI_6                : 8;
  __REG32 PRI_7                : 8;
} __sfrdef33_bits;

/*
 Used for registers:
 SYSTCKCAL
*/
typedef struct {
  __REG32 CAL                  :26;
  __REG32                      : 6;
} __sfrdef34_bits;

/*
 Used for registers:
 ADC_STAT
*/
typedef struct {
  __REG32 DONE0                : 1;
  __REG32 DONE1                : 1;
  __REG32 DONE2                : 1;
  __REG32 DONE3                : 1;
  __REG32 DONE4                : 1;
  __REG32 DONE5                : 1;
  __REG32 DONE6                : 1;
  __REG32 DONE7                : 1;
  __REG32 OVERRUN8             : 1;
  __REG32 OVERRUN9             : 1;
  __REG32 OVERRUN10            : 1;
  __REG32 OVERRUN11            : 1;
  __REG32 OVERRUN12            : 1;
  __REG32 OVERRUN13            : 1;
  __REG32 OVERRUN14            : 1;
  __REG32 OVERRUN15            : 1;
  __REG32 ADINT                : 1;
  __REG32                      :15;
} __sfrdef35_bits;

/*
 Used for registers:
 IPR2
*/
typedef struct {
  __REG32 PRI_8                : 8;
  __REG32 PRI_9                : 8;
  __REG32 PRI_10               : 8;
  __REG32 PRI_11               : 8;
} __sfrdef36_bits;

/*
 Used for registers:
 ICPR
*/
typedef struct {
  __REG32 CLRPEND0             : 1;
  __REG32 CLRPEND1             : 1;
  __REG32 CLRPEND2             : 1;
  __REG32 CLRPEND3             : 1;
  __REG32 CLRPEND4             : 1;
  __REG32 CLRPEND5             : 1;
  __REG32 CLRPEND6             : 1;
  __REG32 CLRPEND7             : 1;
  __REG32 CLRPEND8             : 1;
  __REG32 CLRPEND9             : 1;
  __REG32 CLRPEND10            : 1;
  __REG32 CLRPEND11            : 1;
  __REG32 CLRPEND12            : 1;
  __REG32 CLRPEND13            : 1;
  __REG32 CLRPEND14            : 1;
  __REG32 CLRPEND15            : 1;
  __REG32 CLRPEND16            : 1;
  __REG32 CLRPEND17            : 1;
  __REG32 CLRPEND18            : 1;
  __REG32 CLRPEND19            : 1;
  __REG32 CLRPEND20            : 1;
  __REG32 CLRPEND21            : 1;
  __REG32 CLRPEND22            : 1;
  __REG32 CLRPEND23            : 1;
  __REG32 CLRPEND24            : 1;
  __REG32 CLRPEND25            : 1;
  __REG32 CLRPEND26            : 1;
  __REG32 CLRPEND27            : 1;
  __REG32 CLRPEND28            : 1;
  __REG32 CLRPEND29            : 1;
  __REG32 CLRPEND30            : 1;
  __REG32 CLRPEND31            : 1;
} __sfrdef37_bits;

/*
 Used for registers:
 ISPR
*/
typedef struct {
  __REG32 SETPEND0             : 1;
  __REG32 SETPEND1             : 1;
  __REG32 SETPEND2             : 1;
  __REG32 SETPEND3             : 1;
  __REG32 SETPEND4             : 1;
  __REG32 SETPEND5             : 1;
  __REG32 SETPEND6             : 1;
  __REG32 SETPEND7             : 1;
  __REG32 SETPEND8             : 1;
  __REG32 SETPEND9             : 1;
  __REG32 SETPEND10            : 1;
  __REG32 SETPEND11            : 1;
  __REG32 SETPEND12            : 1;
  __REG32 SETPEND13            : 1;
  __REG32 SETPEND14            : 1;
  __REG32 SETPEND15            : 1;
  __REG32 SETPEND16            : 1;
  __REG32 SETPEND17            : 1;
  __REG32 SETPEND18            : 1;
  __REG32 SETPEND19            : 1;
  __REG32 SETPEND20            : 1;
  __REG32 SETPEND21            : 1;
  __REG32 SETPEND22            : 1;
  __REG32 SETPEND23            : 1;
  __REG32 SETPEND24            : 1;
  __REG32 SETPEND25            : 1;
  __REG32 SETPEND26            : 1;
  __REG32 SETPEND27            : 1;
  __REG32 SETPEND28            : 1;
  __REG32 SETPEND29            : 1;
  __REG32 SETPEND30            : 1;
  __REG32 SETPEND31            : 1;
} __sfrdef38_bits;

/*
 Used for registers:
 IPR3
*/
typedef struct {
  __REG32 PRI_12               : 8;
  __REG32 PRI_13               : 8;
  __REG32 PRI_14               : 8;
  __REG32 PRI_15               : 8;
} __sfrdef39_bits;

/*
 Used for registers:
 ICER
*/
typedef struct {
  __REG32 CLRENA0              : 1;
  __REG32 CLRENA1              : 1;
  __REG32 CLRENA2              : 1;
  __REG32 CLRENA3              : 1;
  __REG32 CLRENA4              : 1;
  __REG32 CLRENA5              : 1;
  __REG32 CLRENA6              : 1;
  __REG32 CLRENA7              : 1;
  __REG32 CLRENA8              : 1;
  __REG32 CLRENA9              : 1;
  __REG32 CLRENA10             : 1;
  __REG32 CLRENA11             : 1;
  __REG32 CLRENA12             : 1;
  __REG32 CLRENA13             : 1;
  __REG32 CLRENA14             : 1;
  __REG32 CLRENA15             : 1;
  __REG32 CLRENA16             : 1;
  __REG32 CLRENA17             : 1;
  __REG32 CLRENA18             : 1;
  __REG32 CLRENA19             : 1;
  __REG32 CLRENA20             : 1;
  __REG32 CLRENA21             : 1;
  __REG32 CLRENA22             : 1;
  __REG32 CLRENA23             : 1;
  __REG32 CLRENA24             : 1;
  __REG32 CLRENA25             : 1;
  __REG32 CLRENA26             : 1;
  __REG32 CLRENA27             : 1;
  __REG32 CLRENA28             : 1;
  __REG32 CLRENA29             : 1;
  __REG32 CLRENA30             : 1;
  __REG32 CLRENA31             : 1;
} __sfrdef40_bits;

/*
 Used for registers:
 IPR4
*/
typedef struct {
  __REG32 PRI_16               : 8;
  __REG32 PRI_17               : 8;
  __REG32 PRI_18               : 8;
  __REG32 PRI_19               : 8;
} __sfrdef41_bits;

/*
 Used for registers:
 ISER
*/
typedef struct {
  __REG32 SETENA0              : 1;
  __REG32 SETENA1              : 1;
  __REG32 SETENA2              : 1;
  __REG32 SETENA3              : 1;
  __REG32 SETENA4              : 1;
  __REG32 SETENA5              : 1;
  __REG32 SETENA6              : 1;
  __REG32 SETENA7              : 1;
  __REG32 SETENA8              : 1;
  __REG32 SETENA9              : 1;
  __REG32 SETENA10             : 1;
  __REG32 SETENA11             : 1;
  __REG32 SETENA12             : 1;
  __REG32 SETENA13             : 1;
  __REG32 SETENA14             : 1;
  __REG32 SETENA15             : 1;
  __REG32 SETENA16             : 1;
  __REG32 SETENA17             : 1;
  __REG32 SETENA18             : 1;
  __REG32 SETENA19             : 1;
  __REG32 SETENA20             : 1;
  __REG32 SETENA21             : 1;
  __REG32 SETENA22             : 1;
  __REG32 SETENA23             : 1;
  __REG32 SETENA24             : 1;
  __REG32 SETENA25             : 1;
  __REG32 SETENA26             : 1;
  __REG32 SETENA27             : 1;
  __REG32 SETENA28             : 1;
  __REG32 SETENA29             : 1;
  __REG32 SETENA30             : 1;
  __REG32 SETENA31             : 1;
} __sfrdef42_bits;

/*
 Used for registers:
 IPR5
*/
typedef struct {
  __REG32 PRI_20               : 8;
  __REG32 PRI_21               : 8;
  __REG32 PRI_22               : 8;
  __REG32 PRI_23               : 8;
} __sfrdef43_bits;

/*
 Used for registers:
 DEVICE_ID
*/
typedef struct {
  __REG32 DEVICEID             :32;
} __sfrdef44_bits;

/*
 Used for registers:
 PCON
*/
typedef struct {
  __REG32 PM                   : 3;
  __REG32 NODPD                : 1;
  __REG32                      : 4;
  __REG32 SLEEPFLAG            : 1;
  __REG32                      : 2;
  __REG32 DPDFLAG              : 1;
  __REG32                      :20;
} __sfrdef45_bits;

/*
 Used for registers:
 IPR6
*/
typedef struct {
  __REG32 PRI_24               : 8;
  __REG32 PRI_25               : 8;
  __REG32 PRI_26               : 8;
  __REG32 PRI_27               : 8;
} __sfrdef46_bits;

/*
 Used for registers:
 GPREG0
 GPREG1
 GPREG2
 GPREG3
*/
typedef struct {
  __REG32 GPDATA               :32;
} __sfrdef47_bits;

/*
 Used for registers:
 CONSET
*/
typedef struct {
  __REG32                      : 2;
  __REG32 AA                   : 1;
  __REG32 SI                   : 1;
  __REG32 STO                  : 1;
  __REG32 STA                  : 1;
  __REG32 I2EN                 : 1;
  __REG32                      :25;
} __sfrdef48_bits;

/*
 Used for registers:
 IPR7
*/
typedef struct {
  __REG32 PRI_20               : 8;
  __REG32 PRI_29               : 8;
  __REG32 PRI_30               : 8;
  __REG32 PRI_31               : 8;
} __sfrdef49_bits;

/*
 Used for registers:
 NOT0
*/
#define NOT0_NOTP00      0x00000001
#define NOT0_NOTP01      0x00000002
#define NOT0_NOTP02      0x00000004
#define NOT0_NOTP03      0x00000008
#define NOT0_NOTP04      0x00000010
#define NOT0_NOTP05      0x00000020
#define NOT0_NOTP06      0x00000040
#define NOT0_NOTP07      0x00000080
#define NOT0_NOTP08      0x00000100
#define NOT0_NOTP09      0x00000200
#define NOT0_NOTP010     0x00000400
#define NOT0_NOTP011     0x00000800
#define NOT0_NOTP012     0x00001000
#define NOT0_NOTP013     0x00002000
#define NOT0_NOTP014     0x00004000
#define NOT0_NOTP015     0x00008000
#define NOT0_NOTP016     0x00010000
#define NOT0_NOTP017     0x00020000
#define NOT0_NOTP018     0x00040000
#define NOT0_NOTP019     0x00080000
#define NOT0_NOTP020     0x00100000
#define NOT0_NOTP021     0x00200000
#define NOT0_NOTP022     0x00400000
#define NOT0_NOTP023     0x00800000
#define NOT0_NOTP024     0x01000000
#define NOT0_NOTP025     0x02000000
#define NOT0_NOTP026     0x04000000
#define NOT0_NOTP027     0x08000000
#define NOT0_NOTP028     0x10000000
#define NOT0_NOTP029     0x20000000
#define NOT0_NOTP030     0x40000000
#define NOT0_NOTP031     0x80000000

/*
 Used for registers:
 I2C_STAT
*/
typedef struct {
  __REG32                      : 3;
  __REG32 Status               : 5;
  __REG32                      :24;
} __sfrdef51_bits;

/*
 Used for registers:
 CLR0
*/
#define CLR0_CLRP00      0x00000001
#define CLR0_CLRP01      0x00000002
#define CLR0_CLRP02      0x00000004
#define CLR0_CLRP03      0x00000008
#define CLR0_CLRP04      0x00000010
#define CLR0_CLRP05      0x00000020
#define CLR0_CLRP06      0x00000040
#define CLR0_CLRP07      0x00000080
#define CLR0_CLRP08      0x00000100
#define CLR0_CLRP09      0x00000200
#define CLR0_CLRP010     0x00000400
#define CLR0_CLRP011     0x00000800
#define CLR0_CLRP012     0x00001000
#define CLR0_CLRP013     0x00002000
#define CLR0_CLRP014     0x00004000
#define CLR0_CLRP015     0x00008000
#define CLR0_CLRP016     0x00010000
#define CLR0_CLRP017     0x00020000
#define CLR0_CLRP018     0x00040000
#define CLR0_CLRP019     0x00080000
#define CLR0_CLRP020     0x00100000
#define CLR0_CLRP021     0x00200000
#define CLR0_CLRP022     0x00400000
#define CLR0_CLRP023     0x00800000
#define CLR0_CLRP024     0x01000000
#define CLR0_CLRP025     0x02000000
#define CLR0_CLRP026     0x04000000
#define CLR0_CLRP027     0x08000000
#define CLR0_CLRP028     0x10000000
#define CLR0_CLRP029     0x20000000
#define CLR0_CLRP030     0x40000000
#define CLR0_CLRP031     0x80000000

/*
 Used for registers:
 IRQLATENCY
*/
typedef struct {
  __REG32 LATENCY              : 8;
  __REG32                      :24;
} __sfrdef53_bits;

/*
 Used for registers:
 NOT1
*/
#define NOT1_NOTP10      0x00000001
#define NOT1_NOTP11      0x00000002
#define NOT1_NOTP12      0x00000004
#define NOT1_NOTP13      0x00000008
#define NOT1_NOTP14      0x00000010
#define NOT1_NOTP15      0x00000020
#define NOT1_NOTP16      0x00000040
#define NOT1_NOTP17      0x00000080
#define NOT1_NOTP18      0x00000100
#define NOT1_NOTP19      0x00000200
#define NOT1_NOTP110     0x00000400
#define NOT1_NOTP111     0x00000800
#define NOT1_NOTP112     0x00001000
#define NOT1_NOTP113     0x00002000
#define NOT1_NOTP114     0x00004000
#define NOT1_NOTP115     0x00008000
#define NOT1_NOTP116     0x00010000
#define NOT1_NOTP117     0x00020000
#define NOT1_NOTP118     0x00040000
#define NOT1_NOTP119     0x00080000
#define NOT1_NOTP120     0x00100000
#define NOT1_NOTP121     0x00200000
#define NOT1_NOTP122     0x00400000
#define NOT1_NOTP123     0x00800000
#define NOT1_NOTP124     0x01000000
#define NOT1_NOTP125     0x02000000
#define NOT1_NOTP126     0x04000000
#define NOT1_NOTP127     0x08000000
#define NOT1_NOTP128     0x10000000
#define NOT1_NOTP129     0x20000000
#define NOT1_NOTP130     0x40000000
#define NOT1_NOTP131     0x80000000

/*
 Used for registers:
 TMR16B0CTCR
 TMR16B1CTCR
*/
typedef struct {
  __REG32 CTM                  : 2;
  __REG32 CIS                  : 2;
  __REG32 ENCC                 : 1;
  __REG32 SELCC                : 3;
  __REG32                      :24;
} __sfrdef55_bits;

/*
 Used for registers:
 SET0
*/
typedef struct {
  __REG32 SETP00               : 1;
  __REG32 SETP01               : 1;
  __REG32 SETP02               : 1;
  __REG32 SETP03               : 1;
  __REG32 SETP04               : 1;
  __REG32 SETP05               : 1;
  __REG32 SETP06               : 1;
  __REG32 SETP07               : 1;
  __REG32 SETP08               : 1;
  __REG32 SETP09               : 1;
  __REG32 SETP010              : 1;
  __REG32 SETP011              : 1;
  __REG32 SETP012              : 1;
  __REG32 SETP013              : 1;
  __REG32 SETP014              : 1;
  __REG32 SETP015              : 1;
  __REG32 SETP016              : 1;
  __REG32 SETP017              : 1;
  __REG32 SETP018              : 1;
  __REG32 SETP019              : 1;
  __REG32 SETP020              : 1;
  __REG32 SETP021              : 1;
  __REG32 SETP022              : 1;
  __REG32 SETP023              : 1;
  __REG32 SETP024              : 1;
  __REG32 SETP025              : 1;
  __REG32 SETP026              : 1;
  __REG32 SETP027              : 1;
  __REG32 SETP028              : 1;
  __REG32 SETP029              : 1;
  __REG32 SETP030              : 1;
  __REG32 SETP031              : 1;
} __sfrdef56_bits;

/*
 Used for registers:
 DAT
 DATA_BUFFER
*/
typedef struct {
  __REG32 Data                 : 8;
  __REG32                      :24;
} __sfrdef57_bits;

/*
 Used for registers:
 MOD
*/
typedef struct {
  __REG32 WDEN                 : 1;
  __REG32 WDRESET              : 1;
  __REG32 WDTOF                : 1;
  __REG32 WDINT                : 1;
  __REG32 WDPROTECT            : 1;
  __REG32 LOCK                 : 1;
  __REG32                      :26;
} __sfrdef58_bits;

/*
 Used for registers:
 PORT0_ENA0
 PORT1_ENA0
*/
typedef struct {
  __REG32 ENA00                : 1;
  __REG32 ENA01                : 1;
  __REG32 ENA02                : 1;
  __REG32 ENA03                : 1;
  __REG32 ENA04                : 1;
  __REG32 ENA05                : 1;
  __REG32 ENA06                : 1;
  __REG32 ENA07                : 1;
  __REG32 ENA08                : 1;
  __REG32 ENA09                : 1;
  __REG32 ENA010               : 1;
  __REG32 ENA011               : 1;
  __REG32 ENA012               : 1;
  __REG32 ENA013               : 1;
  __REG32 ENA014               : 1;
  __REG32 ENA015               : 1;
  __REG32 ENA016               : 1;
  __REG32 ENA017               : 1;
  __REG32 ENA018               : 1;
  __REG32 ENA019               : 1;
  __REG32 ENA020               : 1;
  __REG32 ENA021               : 1;
  __REG32 ENA022               : 1;
  __REG32 ENA023               : 1;
  __REG32 ENA024               : 1;
  __REG32 ENA025               : 1;
  __REG32 ENA026               : 1;
  __REG32 ENA027               : 1;
  __REG32 ENA028               : 1;
  __REG32 ENA029               : 1;
  __REG32 ENA030               : 1;
  __REG32 ENA031               : 1;
} __sfrdef59_bits;

/*
 Used for registers:
 CLR1
*/
#define CLR1_CLRP10      0x00000001
#define CLR1_CLRP11      0x00000002
#define CLR1_CLRP12      0x00000004
#define CLR1_CLRP13      0x00000008
#define CLR1_CLRP14      0x00000010
#define CLR1_CLRP15      0x00000020
#define CLR1_CLRP16      0x00000040
#define CLR1_CLRP17      0x00000080
#define CLR1_CLRP18      0x00000100
#define CLR1_CLRP19      0x00000200
#define CLR1_CLRP110     0x00000400
#define CLR1_CLRP111     0x00000800
#define CLR1_CLRP112     0x00001000
#define CLR1_CLRP113     0x00002000
#define CLR1_CLRP114     0x00004000
#define CLR1_CLRP115     0x00008000
#define CLR1_CLRP116     0x00010000
#define CLR1_CLRP117     0x00020000
#define CLR1_CLRP118     0x00040000
#define CLR1_CLRP119     0x00080000
#define CLR1_CLRP120     0x00100000
#define CLR1_CLRP121     0x00200000
#define CLR1_CLRP122     0x00400000
#define CLR1_CLRP123     0x00800000
#define CLR1_CLRP124     0x01000000
#define CLR1_CLRP125     0x02000000
#define CLR1_CLRP126     0x04000000
#define CLR1_CLRP127     0x08000000
#define CLR1_CLRP128     0x10000000
#define CLR1_CLRP129     0x20000000
#define CLR1_CLRP130     0x40000000
#define CLR1_CLRP131     0x80000000

/*
 Used for registers:
 NMISRC
*/
typedef struct {
  __REG32 IRQNO                : 5;
  __REG32                      :26;
  __REG32 NMIEN                : 1;
} __sfrdef61_bits;

/*
 Used for registers:
 MPIN0
*/
typedef struct {
  __REG32 MPORTP0              :32;
} __sfrdef62_bits;

/*
 Used for registers:
 TMR16B0PWMC
 TMR16B1PWMC
 TMR32B0PWMC
 TMR32B1PWMC
*/
typedef struct {
  __REG32 PWMEN0               : 1;
  __REG32 PWMEN1               : 1;
  __REG32 PWMEN2               : 1;
  __REG32 PWMEN3               : 1;
  __REG32                      :28;
} __sfrdef63_bits;

/*
 Used for registers:
 SET1
*/
typedef struct {
  __REG32 SETP10               : 1;
  __REG32 SETP11               : 1;
  __REG32 SETP12               : 1;
  __REG32 SETP13               : 1;
  __REG32 SETP14               : 1;
  __REG32 SETP15               : 1;
  __REG32 SETP16               : 1;
  __REG32 SETP17               : 1;
  __REG32 SETP18               : 1;
  __REG32 SETP19               : 1;
  __REG32 SETP110              : 1;
  __REG32 SETP111              : 1;
  __REG32 SETP112              : 1;
  __REG32 SETP113              : 1;
  __REG32 SETP114              : 1;
  __REG32 SETP115              : 1;
  __REG32 SETP116              : 1;
  __REG32 SETP117              : 1;
  __REG32 SETP118              : 1;
  __REG32 SETP119              : 1;
  __REG32 SETP120              : 1;
  __REG32 SETP121              : 1;
  __REG32 SETP122              : 1;
  __REG32 SETP123              : 1;
  __REG32 SETP124              : 1;
  __REG32 SETP125              : 1;
  __REG32 SETP126              : 1;
  __REG32 SETP127              : 1;
  __REG32 SETP128              : 1;
  __REG32 SETP129              : 1;
  __REG32 SETP130              : 1;
  __REG32 SETP131              : 1;
} __sfrdef64_bits;

/*
 Used for registers:
 ADR0
*/
typedef struct {
  __REG32 GC                   : 1;
  __REG32 Adress               : 7;
  __REG32                      :24;
} __sfrdef65_bits;

/*
 Used for registers:
 SSP0CR0
 SSP1CR0
*/
typedef struct {
  __REG32 DSS                  : 4;
  __REG32 FRF                  : 2;
  __REG32 CPOL                 : 1;
  __REG32 CPHA                 : 1;
  __REG32 SCR                  : 8;
  __REG32                      :16;
} __sfrdef66_bits;

/*
 Used for registers:
 WDT_TC
 TV
*/
typedef struct {
  __REG32 COUNT                :24;
  __REG32                      : 8;
} __sfrdef67_bits;

/*
 Used for registers:
 PORT0_ENA1
 PORT1_ENA1
*/
typedef struct {
  __REG32 ENA10                : 1;
  __REG32 ENA11                : 1;
  __REG32 ENA12                : 1;
  __REG32 ENA13                : 1;
  __REG32 ENA14                : 1;
  __REG32 ENA15                : 1;
  __REG32 ENA16                : 1;
  __REG32 ENA17                : 1;
  __REG32 ENA18                : 1;
  __REG32 ENA19                : 1;
  __REG32 ENA110               : 1;
  __REG32 ENA111               : 1;
  __REG32 ENA112               : 1;
  __REG32 ENA113               : 1;
  __REG32 ENA114               : 1;
  __REG32 ENA115               : 1;
  __REG32 ENA116               : 1;
  __REG32 ENA117               : 1;
  __REG32 ENA118               : 1;
  __REG32 ENA119               : 1;
  __REG32 ENA120               : 1;
  __REG32 ENA121               : 1;
  __REG32 ENA122               : 1;
  __REG32 ENA123               : 1;
  __REG32 ENA124               : 1;
  __REG32 ENA125               : 1;
  __REG32 ENA126               : 1;
  __REG32 ENA127               : 1;
  __REG32 ENA128               : 1;
  __REG32 ENA129               : 1;
  __REG32 ENA130               : 1;
  __REG32 ENA131               : 1;
} __sfrdef68_bits;

/*
 Used for registers:
 PIN0
*/
typedef struct {
  __REG32 PORT0                :32;
} __sfrdef69_bits;

/*
 Used for registers:
 PINTSEL0
 PINTSEL1
 PINTSEL2
 PINTSEL3
 PINTSEL4
 PINTSEL5
 PINTSEL6
 PINTSEL7
*/
typedef struct {
  __REG32 INTPIN               : 6;
  __REG32                      :26;
} __sfrdef70_bits;

/*
 Used for registers:
 GPREG4
*/
typedef struct {
  __REG32                      :10;
  __REG32 WAKEUPHYS            : 1;
  __REG32 GPDATA               :21;
} __sfrdef71_bits;

/*
 Used for registers:
 MPIN1
*/
typedef struct {
  __REG32 MPORTP1              :32;
} __sfrdef72_bits;

/*
 Used for registers:
 SCLH
*/
typedef struct {
  __REG32 SCLH                 :16;
  __REG32                      :16;
} __sfrdef73_bits;

/*
 Used for registers:
 SSP0CR1
 SSP1CR1
*/
typedef struct {
  __REG32 LBM                  : 1;
  __REG32 SSE                  : 1;
  __REG32 MS                   : 1;
  __REG32 SOD                  : 1;
  __REG32                      :28;
} __sfrdef74_bits;

/*
 Used for registers:
 FEED
*/
#define FEED_FEED   0x000000FF

/*
 Used for registers:
 GPIO_P_MASK0
*/
typedef struct {
  __REG32 MASKP0               :32;
} __sfrdef76_bits;

/*
 Used for registers:
 PIN1
*/
typedef struct {
  __REG32 PORT1                :32;
} __sfrdef77_bits;

/*
 Used for registers:
 RBR
 THR
 DLL
*/
typedef union {
  /* RBR */
  struct {
    __REG32 RBR                : 8;
    __REG32                    :24;
  };

  /* THR */
  struct {
    __REG32 THR                : 8;
    __REG32                    :24;
  };

  /* DLL */
  struct {
    __REG32 DLLSB              : 8;
    __REG32                    :24;
  };

} __sfrdef78_bits;

/*
 Used for registers:
 DIR0
*/
typedef struct {
  __REG32 DIRP0                :32;
} __sfrdef79_bits;

/*
 Used for registers:
 SCLL
*/
typedef struct {
  __REG32 SCLL                 :16;
  __REG32                      :16;
} __sfrdef80_bits;

/*
 Used for registers:
 SSP0DR
 SSP1DR
*/
typedef struct {
  __REG32 DATA                 :16;
  __REG32                      :16;
} __sfrdef81_bits;

/*
 Used for registers:
 GPIO_P_MASK1
*/
typedef struct {
  __REG32 MASKP1               :32;
} __sfrdef82_bits;

/*
 Used for registers:
 DLM
 IER
*/
typedef union {
  /* DLM */
  struct {
    __REG32 DLMSB              : 8;
    __REG32                    :24;
  };

  /* IER */
  struct {
    __REG32 RBRINTEN           : 1;
    __REG32 THREINTEN          : 1;
    __REG32 RLSINTEN           : 1;
    __REG32 MSINTEN            : 1;
    __REG32                    : 4;
    __REG32 ABEOINTEN          : 1;
    __REG32 ABTOINTEN          : 1;
    __REG32                    :22;
  };

} __sfrdef83_bits;

/*
 Used for registers:
 STARTERP0
*/
typedef struct {
  __REG32 PINT0                : 1;
  __REG32 PINT1                : 1;
  __REG32 PINT2                : 1;
  __REG32 PINT3                : 1;
  __REG32 PINT4                : 1;
  __REG32 PINT5                : 1;
  __REG32 PINT6                : 1;
  __REG32 PINT7                : 1;
  __REG32                      :24;
} __sfrdef84_bits;

/*
 Used for registers:
 DIR1
*/
typedef struct {
  __REG32 DIRP1                :32;
} __sfrdef85_bits;

/*
 Used for registers:
 CONCLR
*/
#define CONCLR_AAC      0x00000004
#define CONCLR_SIC      0x00000008
#define CONCLR_STAC     0x00000020
#define CONCLR_I2ENC    0x00000040

/*
 Used for registers:
 CLKSEL
*/
typedef struct {
  __REG32 CLKSEL             : 1;
  __REG32                    :30;
  __REG32 LOCK               : 1;
} __sfrdef87_bits;

/*
 Used for registers:
 SYST_CSR
*/
typedef struct {
  __REG32 ENABLE             : 1;
  __REG32 TICKINT            : 1;
  __REG32 CLKSOURCE          : 1;
  __REG32                    :13;
  __REG32 COUNTFLAG          : 1;
  __REG32                    :15;
} __sfrdef_systcsr_bits;

/*
 Used for registers:
 PIOPORCAP0
*/
typedef struct {
  __REG32 PIOSTAT0             : 1;
  __REG32 PIOSTAT1             : 1;
  __REG32 PIOSTAT2             : 1;
  __REG32 PIOSTAT3             : 1;
  __REG32 PIOSTAT4             : 1;
  __REG32 PIOSTAT5             : 1;
  __REG32 PIOSTAT6             : 1;
  __REG32 PIOSTAT7             : 1;
  __REG32 PIOSTAT8             : 1;
  __REG32 PIOSTAT9             : 1;
  __REG32 PIOSTAT10            : 1;
  __REG32 PIOSTAT11            : 1;
  __REG32 PIOSTAT12            : 1;
  __REG32 PIOSTAT13            : 1;
  __REG32 PIOSTAT14            : 1;
  __REG32 PIOSTAT15            : 1;
  __REG32 PIOSTAT16            : 1;
  __REG32 PIOSTAT17            : 1;
  __REG32 PIOSTAT18            : 1;
  __REG32 PIOSTAT19            : 1;
  __REG32 PIOSTAT20            : 1;
  __REG32 PIOSTAT21            : 1;
  __REG32 PIOSTAT22            : 1;
  __REG32 PIOSTAT23            : 1;
  __REG32 PIOSTAT24            : 1;
  __REG32 PIOSTAT25            : 1;
  __REG32 PIOSTAT26            : 1;
  __REG32 PIOSTAT27            : 1;
  __REG32 PIOSTAT28            : 1;
  __REG32 PIOSTAT29            : 1;
  __REG32 PIOSTAT30            : 1;
  __REG32 PIOSTAT31            : 1;
} __sfrdef88_bits;

/*
 Used for registers:
 TMR32B0CTCR
 TMR32B1CTCR
*/
typedef struct {
  __REG32 CTM                  : 2;
  __REG32 CIS                  : 2;
  __REG32 ENCC                 : 1;
  __REG32 SEICC                : 3;
  __REG32                      :24;
} __sfrdef89_bits;

/*
 Used for registers:
 DEVCMDSTAT
*/
typedef struct {
  __REG32 DEV_ADDR             : 7;
  __REG32 DEV_EN               : 1;
  __REG32 SETUP                : 1;
  __REG32 PLL_ON               : 1;
  __REG32                      : 1;
  __REG32 LPM_SUP              : 1;
  __REG32 INTONNAK_AO          : 1;
  __REG32 INTONNAK_AI          : 1;
  __REG32 INTONNAK_CO          : 1;
  __REG32 INTONNAK_CI          : 1;
  __REG32 DCON                 : 1;
  __REG32 DSUS                 : 1;
  __REG32                      : 1;
  __REG32 LPM_SUS              : 1;
  __REG32 LPM_REWP             : 1;
  __REG32                      : 3;
  __REG32 DCON_C               : 1;
  __REG32 DSUS_C               : 1;
  __REG32 DRES_C               : 1;
  __REG32                      : 1;
  __REG32 VBUSDEBOUNCED        : 1;
  __REG32                      : 3;
} __sfrdef90_bits;

/*
 Used for registers:
 IIR
 FCR
*/
typedef union {
  /* IIR */
  struct {
    __REG32 INTSTATUS          : 1;
    __REG32 INTID              : 3;
    __REG32                    : 2;
    __REG32 IIR_FIFOEN         : 2;
    __REG32 ABEOINT            : 1;
    __REG32 ABTOINT            : 1;
    __REG32                    :22;
  };

  /* FCR */
  struct {
    __REG32 FCR_FIFOEN         : 1;
    __REG32 RXFIFORES          : 1;
    __REG32 TXFIFORES          : 1;
    __REG32                    : 1;
    __REG32                    : 2;
    __REG32 RXTL               : 2;
    __REG32                    :24;
  };

} __sfrdef91_bits;

/*
 Used for registers:
 TMR16B0IR
 TMR16B1IR
*/
typedef struct {
  __REG32 MR0INT               : 1;
  __REG32 MR1INT               : 1;
  __REG32 MR2INT               : 1;
  __REG32 MR3INT               : 1;
  __REG32 CR0INT               : 1;
  __REG32                      :27;
} __sfrdef92_bits;

/*
 Used for registers:
 SYSAHBCLKCTRL
*/
typedef struct {
  __REG32 SYS                  : 1;
  __REG32 ROM                  : 1;
  __REG32 RAM                  : 1;
  __REG32 FLASHREG             : 1;
  __REG32 FLASHARRAY           : 1;
  __REG32 I2C                  : 1;
  __REG32 GPIO                 : 1;
  __REG32 CT16B0               : 1;
  __REG32 CT16B1               : 1;
  __REG32 CT32B0               : 1;
  __REG32 CT32B1               : 1;
  __REG32 SSP0                 : 1;
  __REG32 USART                : 1;
  __REG32 ADC                  : 1;
  __REG32 USB                  : 1;
  __REG32 WWDT                 : 1;
  __REG32 IOCON                : 1;
  __REG32                      : 1;
  __REG32 SSP1                 : 1;
  __REG32 PINT                 : 1;
  __REG32                      : 3;
  __REG32 GROUP0INT            : 1;
  __REG32 GROUP1INT            : 1;
  __REG32                      : 2;
  __REG32 USBRAM               : 1;
  __REG32                      : 4;
} __sfrdef93_bits;

/*
 Used for registers:
 MMCTRL
*/
typedef struct {
  __REG32 MM_ENA               : 1;
  __REG32 ENA_SCL              : 1;
  __REG32 MATCH_ALL            : 1;
  __REG32                      :29;
} __sfrdef94_bits;

/*
 Used for registers:
 WARNINT
*/
typedef struct {
  __REG32 WARNINT            :10;
  __REG32                    :22;
} __sfrdef95_bits;

/*
 Used for registers:
 SYST_RVR
*/
typedef struct {
  __REG32 RELOAD             :24;
  __REG32                    : 8;
} __sfrdef_systrvr_bits;

/*
 Used for registers:
 PIOPORCAP1
*/
typedef struct {
  __REG32 PIOSTAT0             : 1;
  __REG32 PIOSTAT1             : 1;
  __REG32 PIOSTAT2             : 1;
  __REG32 PIOSTAT3             : 1;
  __REG32 PIOSTAT4             : 1;
  __REG32 PIOSTAT5             : 1;
  __REG32 PIOSTAT6             : 1;
  __REG32 PIOSTAT7             : 1;
  __REG32 PIOSTAT8             : 1;
  __REG32 PIOSTAT9             : 1;
  __REG32                      :22;
} __sfrdef96_bits;

/*
 Used for registers:
 INFO
*/
typedef struct {
  __REG32 FRAME_NR             :11;
  __REG32 ERR_CODE             : 4;
  __REG32                      : 1;
  __REG32 CHIP_ID              :16;
} __sfrdef97_bits;

/*
 Used for registers:
 LCR
*/
typedef struct {
  __REG32 WLS                  : 2;
  __REG32 SBS                  : 1;
  __REG32 PE                   : 1;
  __REG32 PS                   : 2;
  __REG32 BC                   : 1;
  __REG32 DLAB                 : 1;
  __REG32                      :24;
} __sfrdef98_bits;

/*
 Used for registers:
 TMR16B0TCR
 TMR16B1TCR
 TMR32B0TCR
 TMR32B1TCR
*/
typedef struct {
  __REG32 CEN                  : 1;
  __REG32 CRST                 : 1;
  __REG32                      :30;
} __sfrdef99_bits;

/*
 Used for registers:
 SYSMEMREMAP
*/
typedef struct {
  __REG32 MAP                  : 2;
  __REG32                      :30;
} __sfrdef100_bits;

/*
 Used for registers:
 ADR1
 ADR2
 ADR3
*/
typedef struct {
  __REG32 GC                   : 1;
  __REG32 Address              : 7;
  __REG32                      :24;
} __sfrdef101_bits;

/*
 Used for registers:
 WINDOW
*/
typedef struct {
  __REG32 WINDOW             :24;
  __REG32                    : 8;
} __sfrdef102_bits;

/*
 Used for registers:
 SYST_CVR
*/
typedef struct {
  __REG32 CURRENT            :24;
  __REG32                    : 8;
} __sfrdef_systcvr_bits;

/*
 Used for registers:
 EPLISTSTART
*/
typedef struct {
  __REG32                      : 8;
  __REG32 EP_LIST              :24;
} __sfrdef103_bits;

/*
 Used for registers:
 USART_MCR
*/
typedef struct {
  __REG32 DTRCTRL              : 1;
  __REG32 RTSCTRL              : 1;
  __REG32                      : 2;
  __REG32 LMS                  : 1;
  __REG32                      : 1;
  __REG32 RTSEN                : 1;
  __REG32 CTSEN                : 1;
  __REG32                      :24;
} __sfrdef104_bits;

/*
 Used for registers:
 TMR16B0TC
 TMR16B1TC
*/
typedef struct {
  __REG32 TC                   :16;
  __REG32                      :16;
} __sfrdef105_bits;

/*
 Used for registers:
 PRESETCTRL
*/
typedef struct {
  __REG32 SSP0_RST_N           : 1;
  __REG32 I2C_RST_N            : 1;
  __REG32 SSP1_RST_N           : 1;
  __REG32                      : 1;
  __REG32                      :28;
} __sfrdef106_bits;

/*
 Used for registers:
 SYST_CALIB
*/
typedef struct {
  __REG32 TENMS                :24;
  __REG32                      : 6;
  __REG32 SKEW                 : 1;
  __REG32 NOREF                : 1;
} __sfrdef_systcalib_bits;

/*
 Used for registers:
 DATABUFSTART
*/
typedef struct {
  __REG32                      :22;
  __REG32 DA_BUF               :10;
} __sfrdef108_bits;

/*
 Used for registers:
 LSR
*/
typedef struct {
  __REG32 RDR                  : 1;
  __REG32 OE                   : 1;
  __REG32 PE                   : 1;
  __REG32 FE                   : 1;
  __REG32 BI                   : 1;
  __REG32 THRE                 : 1;
  __REG32 TEMT                 : 1;
  __REG32 RXFE                 : 1;
  __REG32 TXERR                : 1;
  __REG32                      :23;
} __sfrdef109_bits;

/*
 Used for registers:
 PIO0_4
 PIO0_5
*/
typedef struct {
  __REG32 FUNC                 : 3;
  __REG32                      : 5;
  __REG32 I2CMODE              : 2;
  __REG32 TOD                  : 1;
  __REG32                      :21;
} __sfrdef110_bits;

/*
 Used for registers:
 TMR16B0PR
 TMR16B1PR
*/
typedef struct {
  __REG32 PCVAL                :16;
  __REG32                      :16;
} __sfrdef111_bits;

/*
 Used for registers:
 SYSPLLCTRL
 USBPLLCTRL
*/
typedef struct {
  __REG32 MSEL                 : 5;
  __REG32 PSEL                 : 2;
  __REG32                      :25;
} __sfrdef112_bits;

/*
 Used for registers:
 STARTERP1
*/
typedef struct {
  __REG32                      :12;
  __REG32 WWDTINT              : 1;
  __REG32 BODINT               : 1;
  __REG32                      : 5;
  __REG32 USB_WAKEUP           : 1;
  __REG32 GPIOINT0             : 1;
  __REG32 GPIOINT1             : 1;
  __REG32                      :10;
} __sfrdef113_bits;

/*
 Used for registers:
 ISEL
*/
typedef struct {
  __REG32 PMODE0               : 1;
  __REG32 PMODE1               : 1;
  __REG32 PMODE2               : 1;
  __REG32 PMODE3               : 1;
  __REG32 PMODE4               : 1;
  __REG32 PMODE5               : 1;
  __REG32 PMODE6               : 1;
  __REG32 PMODE7               : 1;
  __REG32                      :24;
} __sfrdef114_bits;

/*
 Used for registers:
 LPM
*/
typedef struct {
  __REG32 HIRD_HW              : 4;
  __REG32 HIRD_SW              : 4;
  __REG32 DATA_PENDING         : 1;
  __REG32                      :23;
} __sfrdef115_bits;

/*
 Used for registers:
 MSR
*/
typedef struct {
  __REG32 DCTS                 : 1;
  __REG32 DDSR                 : 1;
  __REG32 TERI                 : 1;
  __REG32 DDCD                 : 1;
  __REG32 CTS                  : 1;
  __REG32 DSR                  : 1;
  __REG32 RI                   : 1;
  __REG32 DCD                  : 1;
  __REG32                      :24;
} __sfrdef116_bits;

/*
 Used for registers:
 TMR16B0PC
 TMR16B1PC
*/
typedef struct {
  __REG32 PC                   :16;
  __REG32                      :16;
} __sfrdef117_bits;

/*
 Used for registers:
 SYSPLLSTAT
 USBPLLSTAT
*/
typedef struct {
  __REG32 LOCK                 : 1;
  __REG32                      :31;
} __sfrdef118_bits;

/*
 Used for registers:
 IENR
*/
typedef struct {
  __REG32 ENRL0                : 1;
  __REG32 ENRL1                : 1;
  __REG32 ENRL2                : 1;
  __REG32 ENRL3                : 1;
  __REG32 ENRL4                : 1;
  __REG32 ENRL5                : 1;
  __REG32 ENRL6                : 1;
  __REG32 ENRL7                : 1;
  __REG32                      :24;
} __sfrdef119_bits;

/*
 Used for registers:
 TMR32B0IR
 TMR32B1IR
*/
typedef struct {
  __REG32 MR0INT               : 1;
  __REG32 MR1INT               : 1;
  __REG32 MR2INT               : 1;
  __REG32 MR3INT               : 1;
  __REG32 CR0INT               : 1;
  __REG32 CR1INT               : 1;
  __REG32                      :26;
} __sfrdef120_bits;

/*
 Used for registers:
 USBCLKCTRL
*/
typedef struct {
  __REG32 AP_CLK               : 1;
  __REG32 POL_CLK              : 1;
  __REG32                      :30;
} __sfrdef121_bits;

/*
 Used for registers:
 EPSKIP
*/
typedef struct {
  __REG32 SKIP0                : 1;
  __REG32 SKIP1                : 1;
  __REG32 SKIP2                : 1;
  __REG32 SKIP3                : 1;
  __REG32 SKIP4                : 1;
  __REG32 SKIP5                : 1;
  __REG32 SKIP6                : 1;
  __REG32 SKIP7                : 1;
  __REG32 SKIP8                : 1;
  __REG32 SKIP9                : 1;
  __REG32 SKIP10               : 1;
  __REG32 SKIP11               : 1;
  __REG32 SKIP12               : 1;
  __REG32 SKIP13               : 1;
  __REG32 SKIP14               : 1;
  __REG32 SKIP15               : 1;
  __REG32 SKIP16               : 1;
  __REG32 SKIP17               : 1;
  __REG32 SKIP18               : 1;
  __REG32 SKIP19               : 1;
  __REG32 SKIP20               : 1;
  __REG32 SKIP21               : 1;
  __REG32 SKIP22               : 1;
  __REG32 SKIP23               : 1;
  __REG32 SKIP24               : 1;
  __REG32 SKIP25               : 1;
  __REG32 SKIP26               : 1;
  __REG32 SKIP27               : 1;
  __REG32 SKIP28               : 1;
  __REG32 SKIP29               : 1;
  __REG32                      : 2;
} __sfrdef122_bits;

/*
 Used for registers:
 SCR
*/
typedef struct {
  __REG8 PAD                   : 8;
} __sfrdef123_bits;

/*
 Used for registers:
 TMR16B0MCR
 TMR16B1MCR
*/
typedef struct {
  __REG32 MR0I                 : 1;
  __REG32 MR0R                 : 1;
  __REG32 MR0S                 : 1;
  __REG32 MR1I                 : 1;
  __REG32 MR1R                 : 1;
  __REG32 MR1S                 : 1;
  __REG32 MR2I                 : 1;
  __REG32 MR2R                 : 1;
  __REG32 MR2S                 : 1;
  __REG32 MR3I                 : 1;
  __REG32 MR3R                 : 1;
  __REG32 MR3S                 : 1;
  __REG32                      :20;
} __sfrdef124_bits;

/*
 Used for registers:
 I2C_MASK0
 I2C_MASK1
 MASK2
 MASK3
*/
typedef struct {
  __REG32                      : 1;
  __REG32 MASK                 : 7;
  __REG32                      :24;
} __sfrdef125_bits;

/*
 Used for registers:
 SIENR
*/
#define SIENR_SETENRL0   0x00000001
#define SIENR_SETENRL1   0x00000002
#define SIENR_SETENRL2   0x00000004
#define SIENR_SETENRL3   0x00000008
#define SIENR_SETENRL4   0x00000010
#define SIENR_SETENRL5   0x00000020
#define SIENR_SETENRL6   0x00000040
#define SIENR_SETENRL7   0x00000080

/*
 Used for registers:
 USBCLKST
*/
typedef struct {
  __REG32 NEED_CLKST           : 1;
  __REG32                      :31;
} __sfrdef127_bits;

/*
 Used for registers:
 EPINUSE
*/
typedef struct {
  __REG32                      : 2;
  __REG32 BUF2                 : 1;
  __REG32 BUF3                 : 1;
  __REG32 BUF4                 : 1;
  __REG32 BUF5                 : 1;
  __REG32 BUF6                 : 1;
  __REG32 BUF7                 : 1;
  __REG32 BUF8                 : 1;
  __REG32 BUF9                 : 1;
  __REG32                      :22;
} __sfrdef128_bits;

/*
 Used for registers:
 ACR
*/
typedef struct {
  __REG32 START                : 1;
  __REG32 MODE                 : 1;
  __REG32 AUTORESTART          : 1;
  __REG32                      : 5;
  __REG32 ABEOINTCLR           : 1;
  __REG32 ABTOINTCLR           : 1;
  __REG32                      :22;
} __sfrdef129_bits;

/*
 Used for registers:
 TMR16B0MR0
 TMR16B0MR1
 TMR16B0MR2
 TMR16B1MR0
 TMR16B0MR3
 TMR16B1MR1
 TMR16B1MR2
 TMR16B1MR3
*/
typedef struct {
  __REG32 MATCH                :16;
  __REG32                      :16;
} __sfrdef130_bits;

/*
 Used for registers:
 CIENR
*/
#define CIENR_CENRL0    0x00000001
#define CIENR_CENRL1    0x00000002
#define CIENR_CENRL2    0x00000004
#define CIENR_CENRL3    0x00000008
#define CIENR_CENRL4    0x00000010
#define CIENR_CENRL5    0x00000020
#define CIENR_CENRL6    0x00000040
#define CIENR_CENRL7    0x00000080

/*
 Used for registers:
 EPBUFCFG
*/
typedef struct {
  __REG32                      : 2;
  __REG32 BUF_SB2              : 1;
  __REG32 BUF_SB3              : 1;
  __REG32 BUF_SB4              : 1;
  __REG32 BUF_SB5              : 1;
  __REG32 BUF_SB6              : 1;
  __REG32 BUF_SB7              : 1;
  __REG32 BUF_SB8              : 1;
  __REG32 BUF_SB9              : 1;
  __REG32                      :22;
} __sfrdef133_bits;

/*
 Used for registers:
 USART_ICR
*/
typedef struct {
  __REG32 IRDAEN               : 1;
  __REG32 IRDAINV              : 1;
  __REG32 FIXPULSEEN           : 1;
  __REG32 PULSEDIV             : 3;
  __REG32                      :26;
} __sfrdef134_bits;

/*
 Used for registers:
 IENF
*/
typedef struct {
  __REG32 ENAF0                : 1;
  __REG32 ENAF1                : 1;
  __REG32 ENAF2                : 1;
  __REG32 ENAF3                : 1;
  __REG32 ENAF4                : 1;
  __REG32 ENAF5                : 1;
  __REG32 ENAF6                : 1;
  __REG32 ENAF7                : 1;
  __REG32                      :24;
} __sfrdef135_bits;

/*
 Used for registers:
 TMR32B0PR
 TMR32B1PR
*/
typedef struct {
  __REG32 PCVAL                :32;
} __sfrdef136_bits;

/*
 Used for registers:
 INTSTAT
*/
typedef struct {
  __REG32 EP0OUT               : 1;
  __REG32 EP0IN                : 1;
  __REG32 EP1OUT               : 1;
  __REG32 EP1IN                : 1;
  __REG32 EP2OUT               : 1;
  __REG32 EP2IN                : 1;
  __REG32 EP3OUT               : 1;
  __REG32 EP3IN                : 1;
  __REG32 EP4OUT               : 1;
  __REG32 EP4IN                : 1;
  __REG32                      :20;
  __REG32 FRAME_INT            : 1;
  __REG32 DEV_INT              : 1;
} __sfrdef137_bits;

/*
 Used for registers:
 FDR
*/
typedef struct {
  __REG32 DIVADDVAL            : 4;
  __REG32 MULVAL               : 4;
  __REG32                      :24;
} __sfrdef138_bits;

/*
 Used for registers:
 SIENF
*/
#define SIENF_SETENAF0    0x00000001
#define SIENF_SETENAF1    0x00000002
#define SIENF_SETENAF2    0x00000004
#define SIENF_SETENAF3    0x00000008
#define SIENF_SETENAF4    0x00000010
#define SIENF_SETENAF5    0x00000020
#define SIENF_SETENAF6    0x00000040
#define SIENF_SETENAF7    0x00000080

/*
 Used for registers:
 USB_INTEN
*/
typedef struct {
  __REG32 EP_INT_EN            :10;
  __REG32                      :20;
  __REG32 FRAME_INT_EN         : 1;
  __REG32 DEV_INT_EN           : 1;
} __sfrdef140_bits;

/*
 Used for registers:
 OSR
*/
typedef struct {
  __REG32                      : 1;
  __REG32 OSFRAC               : 3;
  __REG32 OSINT                : 4;
  __REG32 FDINT                : 7;
  __REG32                      :17;
} __sfrdef141_bits;

/*
 Used for registers:
 CR
*/
typedef struct {
  __REG32 SEL                  : 8;
  __REG32 CLKDIV               : 8;
  __REG32 BURST                : 1;
  __REG32 CLKS                 : 3;
  __REG32                      : 4;
  __REG32 START                : 3;
  __REG32 EDGE                 : 1;
  __REG32                      : 4;
} __sfrdef142_bits;

/*
 Used for registers:
 SYSOSCCTRL
*/
typedef struct {
  __REG32 BYPASS               : 1;
  __REG32 FREQRANGE            : 1;
  __REG32                      :30;
} __sfrdef143_bits;

/*
 Used for registers:
 CIENF
*/
#define CIENF_CENAF0    0x00000001
#define CIENF_CENAF1    0x00000002
#define CIENF_CENAF2    0x00000004
#define CIENF_CENAF3    0x00000008
#define CIENF_CENAF4    0x00000010
#define CIENF_CENAF5    0x00000020
#define CIENF_CENAF6    0x00000040
#define CIENF_CENAF7    0x00000080

/*
 Used for registers:
 TMR32B0MCR
 TMR32B1MCR
*/
typedef struct {
  __REG32 MR0I                 : 1;
  __REG32 MR0R                 : 1;
  __REG32 MR0S                 : 1;
  __REG32 MR1I                 : 1;
  __REG32 MR1R                 : 1;
  __REG32 MR1S                 : 1;
  __REG32 MR2I                 : 1;
  __REG32 MR2R                 : 1;
  __REG32 MR2S                 : 1;
  __REG32 MR3I                 : 1;
  __REG32 MR3R                 : 1;
  __REG32 MR3S                 : 1;
  __REG32                      :20;
} __sfrdef145_bits;

/*
 Used for registers:
 INTSETSTAT
*/
typedef struct {
  __REG32 EP_SET_INT           :10;
  __REG32                      :20;
  __REG32 FRAME_SET_INT        : 1;
  __REG32 DEV_SET_INT          : 1;
} __sfrdef146_bits;

/*
 Used for registers:
 TER
*/
typedef struct {
  __REG32                      : 7;
  __REG32 TXEN                 : 1;
  __REG32                      :24;
} __sfrdef147_bits;

/*
 Used for registers:
 GDR
*/
typedef struct {
  __REG32                      : 6;
  __REG32 V_VREF               :10;
  __REG32                      : 8;
  __REG32 CHN                  : 3;
  __REG32                      : 3;
  __REG32 OVERRUN              : 1;
  __REG32 DONE                 : 1;
} __sfrdef148_bits;

/*
 Used for registers:
 HDEN
*/
typedef struct {
  __REG32 HDEN                 : 1;
  __REG32                      :31;
} __sfrdef_hden_bits;

/*
 Used for registers:
 WDTOSCCTRL
*/
typedef struct {
  __REG32 DIVSEL               : 5;
  __REG32 FREQSEL              : 4;
  __REG32                      :23;
} __sfrdef149_bits;

/*
 Used for registers:
 PDSLEEPCFG
*/
typedef struct {
  __REG32                      : 3;
  __REG32 BOD_PD               : 1;
  __REG32                      : 2;
  __REG32 WDTOSC_PD            : 1;
  __REG32                      :25;
} __sfrdef150_bits;

/*
 Used for registers:
 RISE
*/
typedef struct {
  __REG32 RDET0                : 1;
  __REG32 RDET1                : 1;
  __REG32 RDET2                : 1;
  __REG32 RDET3                : 1;
  __REG32 RDET4                : 1;
  __REG32 RDET5                : 1;
  __REG32 RDET6                : 1;
  __REG32 RDET7                : 1;
  __REG32                      :24;
} __sfrdef151_bits;

/*
 Used for registers:
 FALL
*/
typedef struct {
  __REG32 FDET0                : 1;
  __REG32 FDET1                : 1;
  __REG32 FDET2                : 1;
  __REG32 FDET3                : 1;
  __REG32 FDET4                : 1;
  __REG32 FDET5                : 1;
  __REG32 FDET6                : 1;
  __REG32 FDET7                : 1;
  __REG32                      :24;
} __sfrdef_fall_bits;

/*
 Used for registers:
 INTROUTING
*/
typedef struct {
  __REG32 ROUTE_INT0           : 1;
  __REG32 ROUTE_INT1           : 1;
  __REG32 ROUTE_INT2           : 1;
  __REG32 ROUTE_INT3           : 1;
  __REG32 ROUTE_INT4           : 1;
  __REG32 ROUTE_INT5           : 1;
  __REG32 ROUTE_INT6           : 1;
  __REG32 ROUTE_INT7           : 1;
  __REG32 ROUTE_INT8           : 1;
  __REG32 ROUTE_INT9           : 1;
  __REG32                      :20;
  __REG32 ROUTE_INT30          : 1;
  __REG32 ROUTE_INT31          : 1;
} __sfrdef152_bits;

/*
 Used for registers:
 PDAWAKECFG
 PDRUNCFG
*/
typedef struct {
  __REG32 IRCOUT_PD            : 1;
  __REG32 IRC_PD               : 1;
  __REG32 FLASH_PD             : 1;
  __REG32 BOD_PD               : 1;
  __REG32 ADC_PD               : 1;
  __REG32 SYSOSC_PD            : 1;
  __REG32 WDTOSC_PD            : 1;
  __REG32 SYSPLL_PD            : 1;
  __REG32 USBPLL_PD            : 1;
  __REG32                      : 1;
  __REG32 USBPAD_PD            : 1;
  __REG32                      : 1;
  __REG32                      : 1;
  __REG32                      :19;
} __sfrdef154_bits;

/*
 Used for registers:
 ADC_INTEN
*/
typedef struct {
  __REG32 ADINTEN0             : 1;
  __REG32 ADINTEN1             : 1;
  __REG32 ADINTEN2             : 1;
  __REG32 ADINTEN3             : 1;
  __REG32 ADINTEN4             : 1;
  __REG32 ADINTEN5             : 1;
  __REG32 ADINTEN6             : 1;
  __REG32 ADINTEN7             : 1;
  __REG32 ADGINTEN             : 1;
  __REG32                      :23;
} __sfrdef155_bits;

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

#define FMSTATCLR_SIG_DONE_CLR 0x00000004

#endif    /* __IAR_SYSTEMS_ICC__ */

/***************************************************************************
**
** ADC
**
***************************************************************************/
__IO_REG32_BIT(AD_CR,              0x4001C000,__READ_WRITE,__sfrdef142_bits);
__IO_REG32_BIT(AD_GDR,             0x4001C004,__READ_WRITE,__sfrdef148_bits);
__IO_REG32_BIT(AD_INTEN,           0x4001C00C,__READ_WRITE,__sfrdef155_bits);
__IO_REG32_BIT(AD_DR0,             0x4001C010,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR1,             0x4001C014,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR2,             0x4001C018,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR3,             0x4001C01C,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR4,             0x4001C020,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR5,             0x4001C024,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR6,             0x4001C028,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_DR7,             0x4001C02C,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(AD_STAT,            0x4001C030,__READ,__sfrdef35_bits);

/***************************************************************************
**
** SSP_SPI0
**
***************************************************************************/
__IO_REG32_BIT(SSP0CR0,            0x40040000,__READ_WRITE,__sfrdef66_bits);
__IO_REG32_BIT(SSP0CR1,            0x40040004,__READ_WRITE,__sfrdef74_bits);
__IO_REG32_BIT(SSP0DR,             0x40040008,__READ_WRITE,__sfrdef81_bits);
__IO_REG32_BIT(SSP0SR,             0x4004000C,__READ,__sfrdef10_bits);
__IO_REG32_BIT(SSP0CPSR,           0x40040010,__READ_WRITE,__sfrdef13_bits);
__IO_REG32_BIT(SSP0IMSC,           0x40040014,__READ_WRITE,__sfrdef16_bits);
__IO_REG32_BIT(SSP0RIS,            0x40040018,__READ,__sfrdef19_bits);
__IO_REG32_BIT(SSP0MIS,            0x4004001C,__READ,__sfrdef23_bits);
__IO_REG32(    SSP0ICR,            0x40040020,__WRITE);

/***************************************************************************
**
** NVIC
**
***************************************************************************/
__IO_REG32_BIT(ISER,                0xE000E100,__READ_WRITE,__sfrdef42_bits);
__IO_REG32_BIT(ICER,                0xE000E180,__READ_WRITE,__sfrdef40_bits);
__IO_REG32_BIT(ISPR,                0xE000E200,__READ_WRITE,__sfrdef38_bits);
__IO_REG32_BIT(ICPR,                0xE000E280,__READ_WRITE,__sfrdef37_bits);
__IO_REG32_BIT(IPR0,                0xE000E400,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(IPR1,                0xE000E404,__READ_WRITE,__sfrdef33_bits);
__IO_REG32_BIT(IPR2,                0xE000E408,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(IPR3,                0xE000E40C,__READ_WRITE,__sfrdef39_bits);
__IO_REG32_BIT(IPR4,                0xE000E410,__READ_WRITE,__sfrdef41_bits);
__IO_REG32_BIT(IPR5,                0xE000E414,__READ_WRITE,__sfrdef43_bits);
__IO_REG32_BIT(IPR6,                0xE000E418,__READ_WRITE,__sfrdef46_bits);
__IO_REG32_BIT(IPR7,                0xE000E41C,__READ_WRITE,__sfrdef49_bits);
__IO_REG32_BIT(CPUIDBR,             0xE000ED00,__READ       ,__cpuidbr_bits);
#define CPUID         CPUIDBR
#define CPUID_bit     CPUIDBR_bit
__IO_REG32_BIT(ICSR,                0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(AIRCR,               0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                 0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                 0xE000ED14,__READ       ,__ccr_bits);
__IO_REG32_BIT(SHPR2,               0xE000ED1C,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHPR3,               0xE000ED20,__READ_WRITE ,__shpr3_bits);

/***************************************************************************
**
** System Tick
**
***************************************************************************/
__IO_REG32_BIT(SYST_CSR,            0xE000E010,__READ_WRITE,__sfrdef_systcsr_bits);
__IO_REG32_BIT(SYST_RVR,            0xE000E014,__READ_WRITE,__sfrdef_systrvr_bits);
__IO_REG32_BIT(SYST_CVR,            0xE000E018,__READ_WRITE,__sfrdef_systcvr_bits);
__IO_REG32_BIT(SYST_CALIB,          0xE000E01C,__READ_WRITE,__sfrdef_systcalib_bits);

/***************************************************************************
**
** SSP_SPI1
**
***************************************************************************/
__IO_REG32_BIT(SSP1CR0,            0x40058000,__READ_WRITE,__sfrdef66_bits);
__IO_REG32_BIT(SSP1CR1,            0x40058004,__READ_WRITE,__sfrdef74_bits);
__IO_REG32_BIT(SSP1DR,             0x40058008,__READ_WRITE,__sfrdef81_bits);
__IO_REG32_BIT(SSP1SR,             0x4005800C,__READ,__sfrdef10_bits);
__IO_REG32_BIT(SSP1CPSR,           0x40058010,__READ_WRITE,__sfrdef13_bits);
__IO_REG32_BIT(SSP1IMSC,           0x40058014,__READ_WRITE,__sfrdef16_bits);
__IO_REG32_BIT(SSP1RIS,            0x40058018,__READ,__sfrdef19_bits);
__IO_REG32_BIT(SSP1MIS,            0x4005801C,__READ,__sfrdef23_bits);
__IO_REG32(    SSP1ICR,            0x40058020,__WRITE);

/***************************************************************************
**
** PMU
**
***************************************************************************/
__IO_REG32_BIT(PCON,                0x40038000,__READ_WRITE,__sfrdef45_bits);
__IO_REG32_BIT(GPREG0,              0x40038004,__READ_WRITE,__sfrdef47_bits);
__IO_REG32_BIT(GPREG1,              0x40038008,__READ_WRITE,__sfrdef47_bits);
__IO_REG32_BIT(GPREG2,              0x4003800C,__READ_WRITE,__sfrdef47_bits);
__IO_REG32_BIT(GPREG3,              0x40038010,__READ_WRITE,__sfrdef47_bits);
__IO_REG32_BIT(GPREG4,              0x40038014,__READ_WRITE,__sfrdef71_bits);

/***************************************************************************
**
** CT16B0
**
***************************************************************************/
__IO_REG32_BIT(TMR16B0IR,           0x4000C000,__READ_WRITE,__sfrdef92_bits);
__IO_REG32_BIT(TMR16B0TCR,          0x4000C004,__READ_WRITE,__sfrdef99_bits);
__IO_REG32_BIT(TMR16B0TC,           0x4000C008,__READ_WRITE,__sfrdef105_bits);
__IO_REG32_BIT(TMR16B0PR,           0x4000C00C,__READ_WRITE,__sfrdef111_bits);
__IO_REG32_BIT(TMR16B0PC,           0x4000C010,__READ_WRITE,__sfrdef117_bits);
__IO_REG32_BIT(TMR16B0MCR,          0x4000C014,__READ_WRITE,__sfrdef124_bits);
__IO_REG32_BIT(TMR16B0MR0,          0x4000C018,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B0MR1,          0x4000C01C,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B0MR2,          0x4000C020,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B0MR3,          0x4000C024,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B0CCR,          0x4000C028,__READ_WRITE,__sfrdef0_bits);
__IO_REG32_BIT(TMR16B0CR0,          0x4000C02C,__READ,__sfrdef11_bits);
__IO_REG32_BIT(TMR16B0EMR,          0x4000C03C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(TMR16B0CTCR,         0x4000C070,__READ_WRITE,__sfrdef55_bits);
__IO_REG32_BIT(TMR16B0PWMC,         0x4000C074,__READ_WRITE,__sfrdef63_bits);

/***************************************************************************
**
** I2C
**
***************************************************************************/
__IO_REG32_BIT(I2C_CONSET,          0x40000000,__READ_WRITE,__sfrdef48_bits);
__IO_REG32_BIT(I2C_STAT,            0x40000004,__READ,__sfrdef51_bits);
__IO_REG32_BIT(I2C_DAT,             0x40000008,__READ_WRITE,__sfrdef57_bits);
__IO_REG32_BIT(I2C_ADR0,            0x4000000C,__READ_WRITE,__sfrdef65_bits);
__IO_REG32_BIT(I2C_SCLH,            0x40000010,__READ_WRITE,__sfrdef73_bits);
__IO_REG32_BIT(I2C_SCLL,            0x40000014,__READ_WRITE,__sfrdef80_bits);
__IO_REG32(    I2C_CONCLR,          0x40000018,__WRITE);
__IO_REG32_BIT(I2C_MMCTRL,          0x4000001C,__READ_WRITE,__sfrdef94_bits);
__IO_REG32_BIT(I2C_ADR1,            0x40000020,__READ_WRITE,__sfrdef101_bits);
__IO_REG32_BIT(I2C_ADR2,            0x40000024,__READ_WRITE,__sfrdef101_bits);
__IO_REG32_BIT(I2C_ADR3,            0x40000028,__READ_WRITE,__sfrdef101_bits);
__IO_REG32_BIT(I2C_DATA_BUFFER,     0x4000002C,__READ,__sfrdef57_bits);
__IO_REG32_BIT(I2C_MASK0,           0x40000030,__READ_WRITE,__sfrdef125_bits);
__IO_REG32_BIT(I2C_MASK1,           0x40000034,__READ_WRITE,__sfrdef125_bits);
__IO_REG32_BIT(I2C_MASK2,           0x40000038,__READ_WRITE,__sfrdef125_bits);
__IO_REG32_BIT(I2C_MASK3,           0x4000003C,__READ_WRITE,__sfrdef125_bits);

/***************************************************************************
**
** USB
**
***************************************************************************/
__IO_REG32_BIT(DEVCMDSTAT,          0x40080000,__READ_WRITE,__sfrdef90_bits);
__IO_REG32_BIT(INFO,                0x40080004,__READ_WRITE,__sfrdef97_bits);
__IO_REG32_BIT(EPLISTSTART,         0x40080008,__READ_WRITE,__sfrdef103_bits);
__IO_REG32_BIT(DATABUFSTART,        0x4008000C,__READ_WRITE,__sfrdef108_bits);
__IO_REG32_BIT(LPM,                 0x40080010,__READ_WRITE,__sfrdef115_bits);
__IO_REG32_BIT(EPSKIP,              0x40080014,__READ_WRITE,__sfrdef122_bits);
__IO_REG32_BIT(EPINUSE,             0x40080018,__READ_WRITE,__sfrdef128_bits);
__IO_REG32_BIT(EPBUFCFG,            0x4008001C,__READ_WRITE,__sfrdef133_bits);
__IO_REG32_BIT(INTSTAT,             0x40080020,__READ_WRITE,__sfrdef137_bits);
__IO_REG32_BIT(USB_INTEN,           0x40080024,__READ_WRITE,__sfrdef140_bits);
__IO_REG32_BIT(INTSETSTAT,          0x40080028,__READ_WRITE,__sfrdef146_bits);
__IO_REG32_BIT(INTROUTING,          0x4008002C,__READ_WRITE,__sfrdef152_bits);
__IO_REG32_BIT(EPTOGGLE,            0x40080034,__READ,__sfrdef5_bits);

/***************************************************************************
**
** GPIO_P
**
***************************************************************************/
__IO_REG8_BIT(B0,                  0x50000000,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B1,                  0x50000001,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B2,                  0x50000002,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B3,                  0x50000003,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B4,                  0x50000004,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B5,                  0x50000005,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B6,                  0x50000006,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B7,                  0x50000007,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B8,                  0x50000008,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B9,                  0x50000009,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B10,                 0x5000000A,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B11,                 0x5000000B,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B12,                 0x5000000C,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B13,                 0x5000000D,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B14,                 0x5000000E,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B15,                 0x5000000F,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B16,                 0x50000010,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B17,                 0x50000011,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B18,                 0x50000012,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B19,                 0x50000013,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B20,                 0x50000014,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B21,                 0x50000015,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B22,                 0x50000016,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B23,                 0x50000017,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B24,                 0x50000018,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B25,                 0x50000019,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B26,                 0x5000001A,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B27,                 0x5000001B,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B28,                 0x5000001C,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B29,                 0x5000001D,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B30,                 0x5000001E,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B31,                 0x5000001F,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B32,                 0x50000020,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B33,                 0x50000021,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B34,                 0x50000022,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B35,                 0x50000023,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B36,                 0x50000024,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B37,                 0x50000025,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B38,                 0x50000026,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B39,                 0x50000027,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B40,                 0x50000028,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B41,                 0x50000029,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B42,                 0x5000002A,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B43,                 0x5000002B,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B44,                 0x5000002C,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B45,                 0x5000002D,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B46,                 0x5000002E,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B47,                 0x5000002F,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B48,                 0x50000030,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B49,                 0x50000031,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B50,                 0x50000032,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B51,                 0x50000033,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B52,                 0x50000034,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B53,                 0x50000035,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B54,                 0x50000036,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B55,                 0x50000037,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B56,                 0x50000038,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B57,                 0x50000039,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B58,                 0x5000003A,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B59,                 0x5000003B,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B60,                 0x5000003C,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B61,                 0x5000003D,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B62,                 0x5000003E,__READ_WRITE,__sfrdef12_bits);
__IO_REG8_BIT(B63,                 0x5000003F,__READ_WRITE,__sfrdef12_bits);
__IO_REG32_BIT(W0,                 0x50001000,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W1,                 0x50001004,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W2,                 0x50001008,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W3,                 0x5000100C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W4,                 0x50001010,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W5,                 0x50001014,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W6,                 0x50001018,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W7,                 0x5000101C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W8,                 0x50001020,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W9,                 0x50001024,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W10,                0x50001028,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W11,                0x5000102C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W12,                0x50001030,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W13,                0x50001034,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W14,                0x50001038,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W15,                0x5000103C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W16,                0x50001040,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W17,                0x50001044,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W18,                0x50001048,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W19,                0x5000104C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W20,                0x50001050,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W21,                0x50001054,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W22,                0x50001058,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W23,                0x5000105C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W24,                0x50001060,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W25,                0x50001064,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W26,                0x50001068,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W27,                0x5000106C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W28,                0x50001070,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W29,                0x50001074,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W30,                0x50001078,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W31,                0x5000107C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W32,                0x50001080,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W33,                0x50001084,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W34,                0x50001088,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W35,                0x5000108C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W36,                0x50001090,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W37,                0x50001094,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W38,                0x50001098,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W39,                0x5000109C,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W40,                0x500010A0,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W41,                0x500010A4,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W42,                0x500010A8,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W43,                0x500010AC,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W44,                0x500010B0,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W45,                0x500010B4,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W46,                0x500010B8,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W47,                0x500010BC,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W48,                0x500010C0,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W49,                0x500010C4,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W50,                0x500010C8,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W51,                0x500010CC,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W52,                0x500010D0,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W53,                0x500010D4,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W54,                0x500010D8,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W55,                0x500010DC,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W56,                0x500010E0,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W57,                0x500010E4,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W58,                0x500010E8,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W59,                0x500010EC,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W60,                0x500010F0,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W61,                0x500010F4,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W62,                0x500010F8,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(W63,                0x500010FC,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(DIR0,               0x50002000,__READ_WRITE,__sfrdef79_bits);
__IO_REG32_BIT(DIR1,               0x50002004,__READ_WRITE,__sfrdef85_bits);
__IO_REG32_BIT(GPIO_P_MASK0,       0x50002080,__READ_WRITE,__sfrdef76_bits);
__IO_REG32_BIT(GPIO_P_MASK1,       0x50002084,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(PIN0,               0x50002100,__READ_WRITE,__sfrdef69_bits);
__IO_REG32_BIT(PIN1,               0x50002104,__READ_WRITE,__sfrdef77_bits);
__IO_REG32_BIT(MPIN0,              0x50002180,__READ_WRITE,__sfrdef62_bits);
__IO_REG32_BIT(MPIN1,              0x50002184,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(SET0,               0x50002200,__READ_WRITE,__sfrdef56_bits);
__IO_REG32_BIT(SET1,               0x50002204,__READ_WRITE,__sfrdef64_bits);
__IO_REG32(    CLR0,               0x50002280,__WRITE);
__IO_REG32(    CLR1,               0x50002284,__WRITE);
__IO_REG32(    NOT0,               0x50002300,__WRITE);
__IO_REG32(    NOT1,               0x50002304,__WRITE);

/***************************************************************************
**
** CT16B1
**
***************************************************************************/
__IO_REG32_BIT(TMR16B1IR,           0x40010000,__READ_WRITE,__sfrdef92_bits);
__IO_REG32_BIT(TMR16B1TCR,          0x40010004,__READ_WRITE,__sfrdef99_bits);
__IO_REG32_BIT(TMR16B1TC,           0x40010008,__READ_WRITE,__sfrdef105_bits);
__IO_REG32_BIT(TMR16B1PR,           0x4001000C,__READ_WRITE,__sfrdef111_bits);
__IO_REG32_BIT(TMR16B1PC,           0x40010010,__READ_WRITE,__sfrdef117_bits);
__IO_REG32_BIT(TMR16B1MCR,          0x40010014,__READ_WRITE,__sfrdef124_bits);
__IO_REG32_BIT(TMR16B1MR0,          0x40010018,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B1MR1,          0x4001001C,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B1MR2,          0x40010020,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B1MR3,          0x40010024,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(TMR16B1CCR,          0x40010028,__READ_WRITE,__sfrdef0_bits);
__IO_REG32_BIT(TMR16B1CR0,          0x4001002C,__READ,__sfrdef11_bits);
__IO_REG32_BIT(TMR16B1EMR,          0x4001003C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(TMR16B1CTCR,         0x40010070,__READ_WRITE,__sfrdef55_bits);
__IO_REG32_BIT(TMR16B1PWMC,         0x40010074,__READ_WRITE,__sfrdef63_bits);

/***************************************************************************
**
** WWDT
**
***************************************************************************/
__IO_REG32_BIT(MOD,                 0x40004000,__READ_WRITE,__sfrdef58_bits);
__IO_REG32_BIT(WWDT_TC,             0x40004004,__READ_WRITE,__sfrdef67_bits);
__IO_REG32(    FEED,                0x40004008,__WRITE);
__IO_REG32_BIT(TV,                  0x4000400C,__READ,__sfrdef67_bits);
__IO_REG32_BIT(CLKSEL,              0x40004010,__READ_WRITE,__sfrdef87_bits);
__IO_REG32_BIT(WARNINT,             0x40004014,__READ_WRITE,__sfrdef95_bits);
__IO_REG32_BIT(WINDOW,              0x40004018,__READ_WRITE,__sfrdef102_bits);

/***************************************************************************
**
** SCB
**
***************************************************************************/
__IO_REG32_BIT(SYSMEMREMAP,         0x40048000,__READ_WRITE,__sfrdef100_bits);
__IO_REG32_BIT(PRESETCTRL,          0x40048004,__READ_WRITE,__sfrdef106_bits);
__IO_REG32_BIT(SYSPLLCTRL,          0x40048008,__READ_WRITE,__sfrdef112_bits);
__IO_REG32_BIT(SYSPLLSTAT,          0x4004800C,__READ,__sfrdef118_bits);
__IO_REG32_BIT(USBPLLCTRL,          0x40048010,__READ_WRITE,__sfrdef112_bits);
__IO_REG32_BIT(USBPLLSTAT,          0x40048014,__READ,__sfrdef118_bits);
__IO_REG32_BIT(SYSOSCCTRL,          0x40048020,__READ_WRITE,__sfrdef143_bits);
__IO_REG32_BIT(WDTOSCCTRL,          0x40048024,__READ_WRITE,__sfrdef149_bits);
__IO_REG32_BIT(SYSRSTSTAT,          0x40048030,__READ_WRITE,__sfrdef8_bits);
__IO_REG32_BIT(SYSPLLCLKSEL,        0x40048040,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(SYSPLLCLKUEN,        0x40048044,__READ_WRITE,__sfrdef24_bits);
__IO_REG32_BIT(USBPLLCLKSEL,        0x40048048,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(USBPLLCLKUEN,        0x4004804C,__READ_WRITE,__sfrdef24_bits);
__IO_REG32_BIT(MAINCLKSEL,          0x40048070,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(MAINCLKUEN,          0x40048074,__READ_WRITE,__sfrdef24_bits);
__IO_REG32_BIT(SYSAHBCLKDIV,        0x40048078,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(SYSAHBCLKCTRL,       0x40048080,__READ_WRITE,__sfrdef93_bits);
__IO_REG32_BIT(SSP0CLKDIV,          0x40048094,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(UARTCLKDIV,          0x40048098,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(SSP1CLKDIV,          0x4004809C,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(USBCLKSEL,           0x400480C0,__READ_WRITE,__sfrdef20_bits);
__IO_REG32_BIT(USBCLKUEN,           0x400480C4,__READ_WRITE,__sfrdef24_bits);
__IO_REG32_BIT(USBCLKDIV,           0x400480C8,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(CLKOUTSEL,           0x400480E0,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(CLKOUTUEN,           0x400480E4,__READ_WRITE,__sfrdef24_bits);
__IO_REG32_BIT(CLKOUTDIV,           0x400480E8,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(PIOPORCAP0,          0x40048100,__READ,__sfrdef88_bits);
__IO_REG32_BIT(PIOPORCAP1,          0x40048104,__READ,__sfrdef96_bits);
__IO_REG32_BIT(BODCTRL,             0x40048150,__READ_WRITE,__sfrdef30_bits);
__IO_REG32_BIT(SYSTCKCAL,           0x40048154,__READ_WRITE,__sfrdef34_bits);
__IO_REG32_BIT(IRQLATENCY,          0x40048170,__READ_WRITE,__sfrdef53_bits);
__IO_REG32_BIT(NMISRC,              0x40048174,__READ_WRITE,__sfrdef61_bits);
__IO_REG32_BIT(PINTSEL0,            0x40048178,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL1,            0x4004817C,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL2,            0x40048180,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL3,            0x40048184,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL4,            0x40048188,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL5,            0x4004818C,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL6,            0x40048190,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(PINTSEL7,            0x40048194,__READ_WRITE,__sfrdef70_bits);
__IO_REG32_BIT(USBCLKCTRL,          0x40048198,__READ_WRITE,__sfrdef121_bits);
__IO_REG32_BIT(USBCLKST,            0x4004819C,__READ,__sfrdef127_bits);
__IO_REG32_BIT(STARTERP0,           0x40048204,__READ_WRITE,__sfrdef84_bits);
__IO_REG32_BIT(STARTERP1,           0x40048214,__READ_WRITE,__sfrdef113_bits);
__IO_REG32_BIT(PDSLEEPCFG,          0x40048230,__READ_WRITE,__sfrdef150_bits);
__IO_REG32_BIT(PDAWAKECFG,          0x40048234,__READ_WRITE,__sfrdef154_bits);
__IO_REG32_BIT(PDRUNCFG,            0x40048238,__READ_WRITE,__sfrdef154_bits);
__IO_REG32_BIT(DEVICE_ID,           0x400483F4,__READ,__sfrdef44_bits);

/***************************************************************************
**
** CT32B0
**
***************************************************************************/
__IO_REG32_BIT(TMR32B0IR,           0x40014000,__READ_WRITE,__sfrdef120_bits);
__IO_REG32_BIT(TMR32B0TCR,          0x40014004,__READ_WRITE,__sfrdef99_bits);
__IO_REG32(    TMR32B0TC,           0x40014008,__READ_WRITE);
__IO_REG32_BIT(TMR32B0PR,           0x4001400C,__READ_WRITE,__sfrdef136_bits);
__IO_REG32(    TMR32B0PC,           0x40014010,__READ_WRITE);
__IO_REG32_BIT(TMR32B0MCR,          0x40014014,__READ_WRITE,__sfrdef145_bits);
__IO_REG32_BIT(TMR32B0MR0,          0x40014018,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B0MR1,          0x4001401C,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B0MR2,          0x40014020,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B0MR3,          0x40014024,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B0CCR,          0x40014028,__READ_WRITE,__sfrdef14_bits);
__IO_REG32_BIT(TMR32B0CR0,          0x4001402C,__READ,__sfrdef17_bits);
__IO_REG32_BIT(TMR32B0EMR,          0x4001403C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(TMR32B0CTCR,         0x40014070,__READ_WRITE,__sfrdef89_bits);
__IO_REG32_BIT(TMR32B0PWMC,         0x40014074,__READ_WRITE,__sfrdef63_bits);

/***************************************************************************
**
** CT32B1
**
***************************************************************************/
__IO_REG32_BIT(TMR32B1IR,           0x40018000,__READ_WRITE,__sfrdef120_bits);
__IO_REG32_BIT(TMR32B1TCR,          0x40018004,__READ_WRITE,__sfrdef99_bits);
__IO_REG32(    TMR32B1TC,           0x40018008,__READ_WRITE);
__IO_REG32_BIT(TMR32B1PR,           0x4001800C,__READ_WRITE,__sfrdef136_bits);
__IO_REG32(    TMR32B1PC,           0x40018010,__READ_WRITE);
__IO_REG32_BIT(TMR32B1MCR,          0x40018014,__READ_WRITE,__sfrdef145_bits);
__IO_REG32_BIT(TMR32B1MR0,          0x40018018,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B1MR1,          0x4001801C,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B1MR2,          0x40018020,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B1MR3,          0x40018024,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(TMR32B1CCR,          0x40018028,__READ_WRITE,__sfrdef14_bits);
__IO_REG32_BIT(TMR32B1CR0,          0x4001802C,__READ,__sfrdef17_bits);
__IO_REG32_BIT(TMR32B1CR1,          0x40018030,__READ,__sfrdef17_bits);
__IO_REG32_BIT(TMR32B1EMR,          0x4001803C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(TMR32B1CTCR,         0x40018070,__READ_WRITE,__sfrdef89_bits);
__IO_REG32_BIT(TMR32B1PWMC,         0x40018074,__READ_WRITE,__sfrdef63_bits);

/***************************************************************************
**
** USART
**
***************************************************************************/
__IO_REG32_BIT(URBR,                 0x40008000,__READ_WRITE,__sfrdef78_bits);
#define UTHR      URBR
#define UTHR_bit  URBR_bit
#define UDLL      URBR
#define UDLL_bit  URBR_bit
__IO_REG32_BIT(UDLM,                 0x40008004,__READ_WRITE,__sfrdef83_bits);
#define UIER      UDLM
#define UIER_bit  UDLM_bit
__IO_REG32_BIT(UIIR,                 0x40008008,__READ_WRITE,__sfrdef91_bits);
#define UFCR      UIIR
#define UFCR_bit  UIIR_bit
__IO_REG32_BIT(ULCR,                 0x4000800C,__READ_WRITE,__sfrdef98_bits);
__IO_REG32_BIT(UMCR,                 0x40008010,__READ_WRITE,__sfrdef104_bits);
__IO_REG32_BIT(ULSR,                 0x40008014,__READ,__sfrdef109_bits);
__IO_REG32_BIT(UMSR,                 0x40008018,__READ,__sfrdef116_bits);
__IO_REG8_BIT( USCR,                 0x4000801C,__READ_WRITE,__sfrdef123_bits);
__IO_REG32_BIT(UACR,                 0x40008020,__READ_WRITE,__sfrdef129_bits);
__IO_REG32_BIT(UICR,                 0x40008024,__READ_WRITE,__sfrdef134_bits);
__IO_REG32_BIT(UFDR,                 0x40008028,__READ_WRITE,__sfrdef138_bits);
__IO_REG32_BIT(UOSR,                 0x4000802C,__READ_WRITE,__sfrdef141_bits);
__IO_REG32_BIT(UTER,                 0x40008030,__READ_WRITE,__sfrdef147_bits);
__IO_REG32_BIT(UHDEN,                0x40008040,__READ_WRITE,__sfrdef_hden_bits);
__IO_REG32_BIT(USCICTRL,             0x40008048,__READ_WRITE,__sfrdef18_bits);
__IO_REG32_BIT(URS485CTRL,           0x4000804C,__READ_WRITE,__sfrdef21_bits);
__IO_REG32_BIT(URS485ADRMATCH,       0x40008050,__READ_WRITE,__sfrdef25_bits);
__IO_REG32_BIT(URS485DLY,            0x40008054,__READ_WRITE,__sfrdef28_bits);
__IO_REG32_BIT(USYNCCTRL,            0x40008058,__READ_WRITE,__sfrdef31_bits);

/***************************************************************************
**
** GPIO_GRP0
**
***************************************************************************/
__IO_REG32_BIT(CTRL0,               0x4005C000,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(PORT0_POL0,          0x4005C020,__READ_WRITE,__sfrdef32_bits);
__IO_REG32_BIT(PORT0_POL1,          0x4005C024,__READ_WRITE,__sfrdef32_bits);
__IO_REG32_BIT(PORT0_ENA0,          0x4005C040,__READ_WRITE,__sfrdef59_bits);
__IO_REG32_BIT(PORT0_ENA1,          0x4005C044,__READ_WRITE,__sfrdef68_bits);

/***************************************************************************
**
** GPIO_PIN
**
***************************************************************************/
__IO_REG32_BIT(ISEL,                0x4004C000,__READ_WRITE,__sfrdef114_bits);
__IO_REG32_BIT(IENR,                0x4004C004,__READ_WRITE,__sfrdef119_bits);
__IO_REG32(    SIENR,               0x4004C008,__WRITE);
__IO_REG32(    CIENR,               0x4004C00C,__WRITE);
__IO_REG32_BIT(IENF,                0x4004C010,__READ_WRITE,__sfrdef135_bits);
__IO_REG32(    SIENF,               0x4004C014,__WRITE);
__IO_REG32(    CIENF,               0x4004C018,__WRITE);
__IO_REG32_BIT(RISE,                0x4004C01C,__READ_WRITE,__sfrdef151_bits);
__IO_REG32_BIT(FALL,                0x4004C020,__READ_WRITE,__sfrdef_fall_bits);
__IO_REG32_BIT(IST,                 0x4004C024,__READ_WRITE,__sfrdef1_bits);

/***************************************************************************
**
** GPIO_GRP1
**
***************************************************************************/
__IO_REG32_BIT(CTRL1,               0x40060000,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(PORT1_POL0,          0x40060020,__READ_WRITE,__sfrdef32_bits);
__IO_REG32_BIT(PORT1_POL1,          0x40060024,__READ_WRITE,__sfrdef32_bits);
__IO_REG32_BIT(PORT1_ENA0,          0x40060040,__READ_WRITE,__sfrdef59_bits);
__IO_REG32_BIT(PORT1_ENA1,          0x40060044,__READ_WRITE,__sfrdef68_bits);

/***************************************************************************
**
** IOCON
**
***************************************************************************/
__IO_REG32_BIT(RESET_PIO0_0,        0x40044000,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_1,              0x40044004,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_2,              0x40044008,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_3,              0x4004400C,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_4,              0x40044010,__READ_WRITE,__sfrdef110_bits);
__IO_REG32_BIT(PIO0_5,              0x40044014,__READ_WRITE,__sfrdef110_bits);
__IO_REG32_BIT(PIO0_6,              0x40044018,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_7,              0x4004401C,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_8,              0x40044020,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_9,              0x40044024,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(SWCLK_PIO0_10,       0x40044028,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(TDI_PIO0_11,         0x4004402C,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(TMS_PIO0_12,         0x40044030,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(TDO_PIO0_13,         0x40044034,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(TRST_PIO0_14,        0x40044038,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(SWDIO_PIO0_15,       0x4004403C,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(PIO0_16,             0x40044040,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(PIO0_17,             0x40044044,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_18,             0x40044048,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_19,             0x4004404C,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_20,             0x40044050,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_21,             0x40044054,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO0_22,             0x40044058,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(PIO0_23,             0x4004405C,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(PIO1_5,              0x40044074,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(PIO1_13,             0x40044094,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_14,             0x40044098,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_15,             0x4004409C,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_16,             0x400440A0,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_19,             0x400440AC,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_20,             0x400440B0,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_21,             0x400440B4,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_22,             0x400440B8,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_23,             0x400440BC,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_24,             0x400440C0,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_25,             0x400440C4,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_26,             0x400440C8,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_27,             0x400440CC,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_28,             0x400440D0,__READ_WRITE,__sfrdef7_bits);
__IO_REG32_BIT(PIO1_29,             0x400440D4,__READ_WRITE,__sfrdef7_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASHCFG,            0x4003C010,__READ_WRITE ,__flashcfg_bits);

/***************************************************************************
 **
 ** Flash signature generation
 **
 ***************************************************************************/
__IO_REG32_BIT(FMSSTART,            0x4003C020,__READ_WRITE ,__fmsstart_bits);
__IO_REG32_BIT(FMSSTOP,             0x4003C024,__READ_WRITE ,__fmsstop_bits);
__IO_REG32(    FMSW0,               0x4003C02C,__READ       );
__IO_REG32(    FMSW1,               0x4003C030,__READ       );
__IO_REG32(    FMSW2,               0x4003C034,__READ       );
__IO_REG32(    FMSW3,               0x4003C038,__READ       );
__IO_REG32_BIT(FMSTAT,              0x4003CFE0,__READ       ,__fmstat_bits);
__IO_REG32(    FMSTATCLR,           0x4003CFE8,__WRITE      );

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
#define NVIC_SSP1             30  /* SSP1                                                   */
#define NVIC_I2C              31  /* I2C                                                    */
#define NVIC_CT16B0           32  /* Counter Timer 0 16 bit                                 */
#define NVIC_CT16B1           33  /* Counter Timer 1 16 bit                                 */
#define NVIC_CT32B0           34  /* Counter Timer 0 32 bit                                 */
#define NVIC_CT32B1           35  /* Counter Timer 1 32 bit                                 */
#define NVIC_SSP0             36  /* SSP0                                                   */
#define NVIC_USART            37  /* USART                                                  */
#define NVIC_USB_IRQ          38  /* USB IRQ interrupt                                      */
#define NVIC_USB_FIQ          39  /* USB FIQ interrupt                                      */
#define NVIC_ADC              40  /* A/D Converter end of conversion                        */
#define NVIC_WWDT             41  /* Windowed Watchdog interrupt (WDINT)                    */
#define NVIC_BOD              42  /* Brown-out detect                                       */
#define NVIC_FLASH            43  /* Flash interrupt                                 */
#define NVIC_USB_WAKEUP       46  /* USB wake-up interrupt                                  */

#endif    /* __LPC11U14_H */

