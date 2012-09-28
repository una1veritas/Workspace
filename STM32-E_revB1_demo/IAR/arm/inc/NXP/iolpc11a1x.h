/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC11A1X
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 53276 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __LPC11A1X_H
#define __LPC11A1X_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC11A1X SPECIAL FUNCTION REGISTERS
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
 TMR32B1CR0
 TMR32B1CR1
 TMR32B1CR2
 TMR32B1CR3
 TMR32B0CR0
 TMR32B0CR1
 TMR32B0CR2
 TMR32B0CR3
*/
typedef struct {
  __REG32 CAP                  :32;
} __sfrdef0_bits;

/*
 Used for registers:
 SSP0CLKDIV
 UARTCLKDIV
 SSP1CLKDIV
 CLKOUTDIV
 SYSTICKCLKDIV
 SYSAHBCLKDIV
*/
typedef struct {
  __REG32 DIV                  : 8;
  __REG32                      :24;
} __sfrdef2_bits;

/*
 Used for registers:
 SSP1CR1
 SSP0CR1
*/
typedef struct {
  __REG32 LBM                  : 1;
  __REG32 SSE                  : 1;
  __REG32 MS                   : 1;
  __REG32 SOD                  : 1;
  __REG32                      :28;
} __sfrdef3_bits;

/*
 Used for registers:
 AD0SEL
*/
typedef struct {
  __REG32                      :10;
  __REG32 AD5SEL               : 2;
  __REG32 AD6SEL               : 2;
  __REG32 AD7SEL               : 2;
  __REG32                      :16;
} __sfrdef4_bits;

/*
 Used for registers:
 TMR16B1PWMC
 TMR32B1PWMC
 TMR16B0PWMC
 TMR32B0PWMC
*/
typedef struct {
  __REG32 PWMEN0               : 1;
  __REG32 PWMEN1               : 1;
  __REG32 PWMEN2               : 1;
  __REG32 PWMEN3               : 1;
  __REG32                      :28;
} __sfrdef5_bits;

/*
 Used for registers:
 TMR32B1CTCR
 TMR32B0CTCR
*/
typedef struct {
  __REG32 CTM                  : 2;
  __REG32 CIS                  : 2;
  __REG32 EnCC                 : 1;
  __REG32 SelCC                : 3;
  __REG32                      :24;
} __sfrdef6_bits;

/*
 Used for registers:
 FMSW3
*/
typedef struct {
  __REG32 FMSW30               : 1;
  __REG32 FMSW31               : 1;
  __REG32 FMSW32               : 1;
  __REG32 FMSW33               : 1;
  __REG32 FMSW34               : 1;
  __REG32 FMSW35               : 1;
  __REG32 FMSW36               : 1;
  __REG32 FMSW37               : 1;
  __REG32 FMSW38               : 1;
  __REG32 FMSW39               : 1;
  __REG32 FMSW310              : 1;
  __REG32 FMSW311              : 1;
  __REG32 FMSW312              : 1;
  __REG32 FMSW313              : 1;
  __REG32 FMSW314              : 1;
  __REG32 FMSW315              : 1;
  __REG32 FMSW316              : 1;
  __REG32 FMSW317              : 1;
  __REG32 FMSW318              : 1;
  __REG32 FMSW319              : 1;
  __REG32 FMSW320              : 1;
  __REG32 FMSW321              : 1;
  __REG32 FMSW322              : 1;
  __REG32 FMSW323              : 1;
  __REG32 FMSW324              : 1;
  __REG32 FMSW325              : 1;
  __REG32 FMSW326              : 1;
  __REG32 FMSW327              : 1;
  __REG32 FMSW328              : 1;
  __REG32 FMSW329              : 1;
  __REG32 FMSW330              : 1;
  __REG32 FMSW331              : 1;
} __sfrdef8_bits;

/*
 Used for registers:
 I2MASK3
 I2MASK0
 I2MASK1
 I2MASK2
*/
typedef struct {
  __REG32                      : 1;
  __REG32 MASK                 : 7;
  __REG32                      :24;
} __sfrdef9_bits;

/*
 Used for registers:
 SSP0IMSC
 SSP1IMSC
*/
typedef struct {
  __REG32 RORIM                : 1;
  __REG32 RTIM                 : 1;
  __REG32 RXIM                 : 1;
  __REG32 TXIM                 : 1;
  __REG32                      :28;
} __sfrdef10_bits;

/*
 Used for registers:
 WDWINDOW
*/
typedef struct {
  __REG32 WINDOW               :24;
  __REG32                      : 8;
} __sfrdef11_bits;

/*
 Used for type I IOCON registers:
 IOCON_P0_2
 IOCON_P0_3
 
 
*/
typedef struct {
  __REG32 FUNC                 : 3;
  __REG32                      : 5;
  __REG32 HS                   : 1;
  __REG32 HIDRIVE              : 1;
  __REG32                      :22;
} __sfrdef12_bits;

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
} __sfrdef13_bits;

/*
 Used for registers:
 IPR1
*/
typedef struct {
  __REG32 PRI_4                : 8;
  __REG32 PRI_5                : 8;
  __REG32 PRI_6                : 8;
  __REG32 PRI_7                : 8;
} __sfrdef14_bits;

/*
 Used for type D IOCON registers:
 IOCON_RESET_P0_0
 IOCON_P0_1
 IOCON_P0_12
 IOCON_P0_18
 IOCON_P0_19
 IOCON_P0_20
 IOCON_P0_21
 IOCON_P0_23
 IOCON_P0_24
 IOCON_P0_25
 IOCON_P0_26
 IOCON_P0_28
 IOCON_P0_29
 IOCON_P0_30
 IOCON_P0_31
 IOCON_P1_0
 IOCON_P1_1 
 IOCON_P1_2 
 IOCON_P1_3 
 IOCON_P1_4 
 IOCON_P1_5 
 IOCON_P1_6
 IOCON_P1_7 
 IOCON_P1_8 
 IOCON_P1_9
 
*/
typedef struct {
  __REG32 FUNC                 : 3;
  __REG32 MODE                 : 2;
  __REG32 HYS                  : 1;
  __REG32 INV                  : 1;
  __REG32                      : 2;
  __REG32 SLEW                 : 1;
  __REG32 OD                   : 1;
  __REG32                      :21;
} __sfrdef15_bits;

/*
 Used for registers:
 TMR16B0CR0
 TMR16B0CR1
 TMR16B0CR2
 TMR16B0CR3
 TMR16B1CR0
 TMR16B1CR1
 TMR16B1CR2
 TMR16B1CR3 
*/
typedef struct {
  __REG32 CAP                  :16;
  __REG32                      :16;
} __sfrdef16_bits;

/*
 Used for registers:
 SYST_CVR
*/
typedef struct {
  __REG32 CURRENT              :24;
  __REG32                      : 8;
} __sfrdef17_bits;

/*
 Used for registers:
 SSP1DR
 SSP0DR
*/
typedef struct {
  __REG32 DATA                 :16;
  __REG32                      :16;
} __sfrdef18_bits;

/*
 Used for registers:
 AD0INTEN
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
} __sfrdef19_bits;

/*
 Used for registers:
 PINTMODE
*/
typedef struct {
  __REG8 MODE0                 : 1;
  __REG8 MODE1                 : 1;
  __REG8 MODE2                 : 1;
  __REG8 MODE3                 : 1;
  __REG8 MODE4                 : 1;
  __REG8 MODE5                 : 1;
  __REG8 MODE6                 : 1;
  __REG8 MODE7                 : 1;
} __sfrdef20_bits;

/*
 Used for registers:
 SSP0RIS
 SSP1RIS
*/
typedef struct {
  __REG32 RORRIS               : 1;
  __REG32 RTRIS                : 1;
  __REG32 RXRIS                : 1;
  __REG32 TXRIS                : 1;
  __REG32                      :28;
} __sfrdef21_bits;

/*
 Used for registers:
 TMR16B0MR0
 TMR16B0MR1
 TMR16B0MR2
 TMR16B0MR3
 TMR16B1MR0
 TMR16B1MR1
 TMR16B1MR2
 TMR16B1MR3
*/
typedef struct {
  __REG32 MATCH                :16;
  __REG32                      :16;
} __sfrdef22_bits;

/*
 Used for registers:
 EECMD
*/
typedef struct {
  __REG32 CMD                  : 3;
  __REG32 RDPREFETCH           : 1;
  __REG32 PAR_ACCESS           : 1;
  __REG32                      :27;
} __sfrdef23_bits;

/*
 Used for registers:
 IPR2
*/
typedef struct {
  __REG32 PRI_8                : 8;
  __REG32 PRI_9                : 8;
  __REG32 PRI_10               : 8;
  __REG32 PRI_11               : 8;
} __sfrdef24_bits;

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
} __sfrdef25_bits;

/*
 Used for registers:
 SYST_CALIB
*/
typedef struct {
  __REG32 TENMS                :24;
  __REG32                      : 6;
  __REG32 SKEW                 : 1;
  __REG32 NOREF                : 1;
} __sfrdef26_bits;

/*
 Used for registers:
 I2CONSET
*/
typedef struct {
  __REG32                      : 2;
  __REG32 AA                   : 1;
  __REG32 SI                   : 1;
  __REG32 STO                  : 1;
  __REG32 STA                  : 1;
  __REG32 I2EN                 : 1;
  __REG32                      :25;
} __sfrdef27_bits;

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
} __sfrdef28_bits;

/*
 Used for registers:
 AD0DR0
 AD0DR1
 AD0DR2
 AD0DR3
 AD0DR4
 AD0DR5
 AD0DR6
 AD0DR7
*/
typedef struct {
  __REG32                      : 6;
  __REG32 V_VREF               :10;
  __REG32                      :14;
  __REG32 OVERRUN              : 1;
  __REG32 DONE                 : 1;
} __sfrdef29_bits;

/*
 Used for registers:
 CLKOUTSEL
 MAINCLKSEL
 SYSPLLCLKSEL
*/
typedef struct {
  __REG32 SEL                  : 2;
  __REG32                      :30;
} __sfrdef30_bits;

/*
 Used for registers:
 PINTEN
 PINTENR
*/
typedef struct {
    __REG8 INTEN0              : 1;
    __REG8 INTEN1              : 1;
    __REG8 INTEN2              : 1;
    __REG8 INTEN3              : 1;
    __REG8 INTEN4              : 1;
    __REG8 INTEN5              : 1;
    __REG8 INTEN6              : 1;
    __REG8 INTEN7              : 1;
} __sfrdef31_bits;

/*
 Used for registers:
 SSP0MIS
 SSP1MIS
*/
typedef struct {
  __REG32 RORMIS               : 1;
  __REG32 RTMIS                : 1;
  __REG32 RXMIS                : 1;
  __REG32 TXMIS                : 1;
  __REG32                      :28;
} __sfrdef32_bits;

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
} __sfrdef33_bits;

/*
 Used for registers:
 EEADDR
*/
typedef struct {
  __REG32 ADDR                 :14;
  __REG32                      :18;
} __sfrdef34_bits;

/*
 Used for registers:
 IPR3
*/    
typedef struct {
  __REG32 PRI_12               : 8;
  __REG32 PRI_13               : 8;
  __REG32 PRI_14               : 8;
  __REG32 PRI_15               : 8;
} __sfrdef35_bits;

/*
 Used for registers:
 TMR32B0MR0
 TMR32B0MR1
 TMR32B0MR2
 TMR32B0MR3
 TMR32B1MR0
 TMR32B1MR1
 TMR32B1MR2
 TMR32B1MR3
*/
typedef struct {
  __REG32 MATCH                :32;
} __sfrdef36_bits;

/*
 Used for registers:
 TMR16B1EMR
 TMR32B1EMR
 TMR16B0EMR
 TMR32B0EMR
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
} __sfrdef37_bits;

/*
 Used for registers:
 SSP1CPSR
 SSP0CPSR
*/
typedef struct {
  __REG32 CPSDVSR              : 8;
  __REG32                      :24;
} __sfrdef38_bits;

/*
 Used for registers:
 I2STAT
*/
typedef struct {
  __REG32                      : 3;
  __REG32 Status               : 5;
  __REG32                      :24;
} __sfrdef39_bits;

/*
 Used for registers:
 CLKOUTUEN
 MAINCLKUEN
 SYSPLLCLKUEN
*/
typedef struct {
  __REG32 ENA                  : 1;
  __REG32                      :31;
} __sfrdef40_bits;

/*
 Used for registers:
 PINTSEN
 PINTSENR
*/
typedef struct {
    __REG8 INTEN0              : 1;
    __REG8 INTEN1              : 1;
    __REG8 INTEN2              : 1;
    __REG8 INTEN3              : 1;
    __REG8 INTEN4              : 1;
    __REG8 INTEN5              : 1;
    __REG8 INTEN6              : 1;
    __REG8 INTEN7              : 1;
} __sfrdef41_bits;

/*
 Used for registers:
 SSP0ICR
 SSP1ICR
*/
typedef struct {
  __REG32 RORIC                : 1;
  __REG32 RTIC                 : 1;
  __REG32                      :30;
} __sfrdef42_bits;

/*
 Used for registers:
 UART_RBR
 UART_THR
 UART_DLL
*/
typedef union {
  /* UART_RBR */
  struct {
    __REG32 RBR                : 8;
    __REG32                    :24;
  };

  /* UART_THR */
  struct {
    __REG32 THR                : 8;
    __REG32                    :24;
  };

  /* UART_DLL */
  struct {
    __REG32 DLLSB              : 8;
    __REG32                    :24;
  };
} __sfrdef43_bits;

/*
 Used for registers:
 P0NOT
*/
typedef struct {
  __REG32 NOT0                 : 1;
  __REG32 NOT1                 : 1;
  __REG32 NOT2                 : 1;
  __REG32 NOT3                 : 1;
  __REG32 NOT4                 : 1;
  __REG32 NOT5                 : 1;
  __REG32 NOT6                 : 1;
  __REG32 NOT7                 : 1;
  __REG32 NOT8                 : 1;
  __REG32 NOT9                 : 1;
  __REG32 NOT10                : 1;
  __REG32 NOT11                : 1;
  __REG32 NOT12                : 1;
  __REG32 NOT13                : 1;
  __REG32 NOT14                : 1;
  __REG32 NOT15                : 1;
  __REG32 NOT16                : 1;
  __REG32 NOT17                : 1;
  __REG32 NOT18                : 1;
  __REG32 NOT19                : 1;
  __REG32 NOT20                : 1;
  __REG32 NOT21                : 1;
  __REG32 NOT22                : 1;
  __REG32 NOT23                : 1;
  __REG32 NOT24                : 1;
  __REG32 NOT25                : 1;
  __REG32 NOT26                : 1;
  __REG32 NOT27                : 1;
  __REG32 NOT28                : 1;
  __REG32 NOT29                : 1;
  __REG32 NOT30                : 1;
  __REG32 NOT31                : 1;
} __sfrdef44_bits;

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
  __REG32 XTAL_PD              : 1;
  __REG32 WDTOSC_PD            : 1;
  __REG32 SYSPLL_PD            : 1;
  __REG32                      : 5;
  __REG32 LFOSC_PD             : 1;
  __REG32 DAC_PD               : 1;
  __REG32 TS_PD                : 1;
  __REG32 ACOMP_PD             : 1;  
  __REG32                      :15;
} __sfrdef45_bits;

/*
 Used for registers:
 EEWDATA
*/
typedef struct {
  __REG32 WDATA                :32;
} __sfrdef46_bits;

/*
 Used for registers:
 IPR4
*/
typedef struct {
  __REG32 PRI_16               : 8;
  __REG32 PRI_17               : 8;
  __REG32 PRI_18               : 8;
  __REG32 PRI_19               : 8;
} __sfrdef47_bits;

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
} __sfrdef48_bits;

/*
 Used for registers:
 I2DAT
 I2DATA_BUFFER
*/
typedef struct {
  __REG32 Data                 : 8;
  __REG32                      :24;
} __sfrdef49_bits;

/*
 Used for registers:
 PINTCEN
 PINTCENR
*/
typedef struct {
    __REG8 INTDIS0             : 1;
    __REG8 INTDIS1             : 1;
    __REG8 INTDIS2             : 1;
    __REG8 INTDIS3             : 1;
    __REG8 INTDIS4             : 1;
    __REG8 INTDIS5             : 1;
    __REG8 INTDIS6             : 1;
    __REG8 INTDIS7             : 1;
} __sfrdef50_bits;

/*
 Used for registers:
 UART_DLM
 UART_IER
*/
typedef union {
  /* UART_DLM */
  struct {
    __REG32 DLMSB              : 8;
    __REG32                    :24;
  };

  /* UART_IER */
  struct {
    __REG32 RBRIntEn           : 1;
    __REG32 THREIntEn          : 1;
    __REG32 RLSIntEN           : 1;
    __REG32 MSIntEn            : 1;
    __REG32                    : 4;
    __REG32 ABEOIntEn          : 1;
    __REG32 ABTOIntEn          : 1;
    __REG32                    :22;
  };
  
} __sfrdef51_bits;

/*
 Used for registers:
 P1NOT
*/
typedef struct {
  __REG32 NOT0                 : 1;
  __REG32 NOT1                 : 1;
  __REG32 NOT2                 : 1;
  __REG32 NOT3                 : 1;
  __REG32 NOT4                 : 1;
  __REG32 NOT5                 : 1;
  __REG32 NOT6                 : 1;
  __REG32 NOT7                 : 1;
  __REG32 NOT8                 : 1;
  __REG32 NOT9                 : 1;
  __REG32                      :22;
} __sfrdef52_bits;

/*
 Used for registers:
 EERDATA
*/
typedef struct {
  __REG32 RDATA                :32;
} __sfrdef53_bits;

/*
 Used for registers:
 IPR5
*/
typedef struct {
  __REG32 PRI_20               : 8;
  __REG32 PRI_21               : 8;
  __REG32 PRI_22               : 8;
  __REG32 PRI_23               : 8;
} __sfrdef54_bits;

/*
 Used for registers:
 UART_SCICTRL
*/
typedef struct {
    __REG32 SCIEN              : 1;
    __REG32 NACKDIS            : 1;
    __REG32 PROTSEL            : 1;
    __REG32                    : 2;
    __REG32 TXRETRY            : 3;
    __REG32 XTRAGUARD          : 8;
    __REG32                    :16;
} __sfrdef55_bits;

/*
 Used for registers:
 TMR16B1IR
 TMR16B0IR
*/
typedef struct {
  __REG32 MR0INT               : 1;
  __REG32 MR1INT               : 1;
  __REG32 MR2INT               : 1;
  __REG32 MR3INT               : 1;
  __REG32 CR0INT               : 1;
  __REG32 CR1INT               : 1;
  __REG32 CR2INT               : 1;
  __REG32 CR3INT               : 1;
  __REG32                      :24;
} __sfrdef56_bits;

/*
 Used for registers:
 P0CLR
*/
typedef struct {
  __REG32 CLR0                 : 1;
  __REG32 CLR1                 : 1;
  __REG32 CLR2                 : 1;
  __REG32 CLR3                 : 1;
  __REG32 CLR4                 : 1;
  __REG32 CLR5                 : 1;
  __REG32 CLR6                 : 1;
  __REG32 CLR7                 : 1;
  __REG32 CLR8                 : 1;
  __REG32 CLR9                 : 1;
  __REG32 CLR10                : 1;
  __REG32 CLR11                : 1;
  __REG32 CLR12                : 1;
  __REG32 CLR13                : 1;
  __REG32 CLR14                : 1;
  __REG32 CLR15                : 1;
  __REG32 CLR16                : 1;
  __REG32 CLR17                : 1;
  __REG32 CLR18                : 1;
  __REG32 CLR19                : 1;
  __REG32 CLR20                : 1;
  __REG32 CLR21                : 1;
  __REG32 CLR22                : 1;
  __REG32 CLR23                : 1;
  __REG32 CLR24                : 1;
  __REG32 CLR25                : 1;
  __REG32 CLR26                : 1;
  __REG32 CLR27                : 1;
  __REG32 CLR28                : 1;
  __REG32 CLR29                : 1;
  __REG32 CLR30                : 1;
  __REG32 CLR31                : 1;
} __sfrdef57_bits;

/*
 Used for registers:
 I2ADR0
*/
typedef struct {
  __REG32 GC                   : 1;
  __REG32 Adress               : 7;
  __REG32                      :24;
} __sfrdef58_bits;

/*
 Used for registers:
 UART_IIR
 UART_FCR
*/
typedef union {
  /* UART_IIR */
  struct {
    __REG32 IntStatus          : 1;
    __REG32 IntId              : 3;
    __REG32                    : 2;
    __REG32 IIR_FIFOEn         : 2;
    __REG32 ABEOInt            : 1;
    __REG32 ABTOInt            : 1;
    __REG32                    :22;
  };

  /* UART_FCR */
  struct {
    __REG32 FCR_FIFOEn         : 1;
    __REG32 RXFIFO             : 1;
    __REG32 TXFIFO             : 1;
    __REG32                    : 1;
    __REG32                    : 2;
    __REG32 RXTL               : 2;
    __REG32                    :24;
  }; 

} __sfrdef59_bits;

/*
 Used for registers:
 PINTACT
 PINTENF
*/
typedef union {
  /* PINTACT */
  struct {
    __REG8                     : 8;
  };

  /* PINTENF */
  struct {
    __REG8                     : 8;
  };

} __sfrdef60_bits;

/*
 Used for registers:
 WDTOSCCTRL
*/
typedef struct {
  __REG32 DIVSEL               : 5;
  __REG32 FREQSEL              : 4;
  __REG32                      :23;
} __sfrdef61_bits;

/*
 Used for registers:
 EEWSTATE
*/
typedef struct {
  __REG32 PHASE3               : 8;
  __REG32 PHASE2               : 8;
  __REG32 PHASE1               : 8;
  __REG32                      : 8;
} __sfrdef62_bits;

/*
 Used for registers:
 IPR6
*/
typedef struct {
  __REG32 PRI_24               : 8;
  __REG32 PRI_25               : 8;
  __REG32 PRI_26               : 8;
  __REG32 PRI_27               : 8;
} __sfrdef63_bits;

/*
 Used for registers:
 UART_RS485CTRL
*/
typedef union {
  /* UART_RS485CTRL */
  struct {
    __REG32 NMMEN              : 1;
    __REG32 RXDIS              : 1;
    __REG32 AADEN              : 1;
    __REG32 SEL                : 1;
    __REG32 DCTRL              : 1;
    __REG32 OINV               : 1; 
    __REG32                    :26;
  };
  
} __sfrdef64_bits;

/*
 Used for registers:
 TMR16B0CCR
 TMR16B1CCR
*/
typedef struct {
  __REG32 CAP0RE               : 1;
  __REG32 CAP0FE               : 1;
  __REG32 CAP0I                : 1;
  __REG32                      :29;
} __sfrdef65_bits;

/*
 Used for registers:
 TMR16B1TCR
 TMR32B1TCR
 TMR16B0TCR
 TMR32B0TCR
*/
typedef struct {
  __REG32 CEN                  : 1;
  __REG32 CRST                 : 1;
  __REG32                      :30;
} __sfrdef66_bits;

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
} __sfrdef67_bits;

/*
 Used for registers:
 TMR32B1IR
 TMR32B0IR
*/
typedef struct {
  __REG32 MR0INT               : 1;
  __REG32 MR1INT               : 1;
  __REG32 MR2INT               : 1;
  __REG32 MR3INT               : 1;
  __REG32 CR0INT               : 1;
  __REG32 CR1INT               : 1;
  __REG32                      :26;
} __sfrdef68_bits;

/*
 Used for registers:
 NMISRC
*/
typedef struct {
  __REG32 IRQNO                : 5;
  __REG32                      :26;
  __REG32 NMIEN                : 1;
} __sfrdef69_bits;

/*
 Used for registers:
 P1CLR
*/
typedef struct {
  __REG32 CLR0                 : 1;
  __REG32 CLR1                 : 1;
  __REG32 CLR2                 : 1;
  __REG32 CLR3                 : 1;
  __REG32 CLR4                 : 1;
  __REG32 CLR5                 : 1;
  __REG32 CLR6                 : 1;
  __REG32 CLR7                 : 1;
  __REG32 CLR8                 : 1;
  __REG32 CLR9                 : 1;
  __REG32                      :22;
} __sfrdef70_bits;

/*
 Used for registers:
 I2SCLH
*/
typedef struct {
  __REG32 SCLH                 :16;
  __REG32                      :16;
} __sfrdef71_bits;

/*
 Used for type A IOCON registers:
 IOCON_P0_4
 IOCON_TCK_P0_5
 IOCON_TDI_P0_6
 IOCON_TMS_P0_7
 IOCON_TDO_P0_8
 IOCON_TRST_P0_9
 IOCON_SWDIO_P0_10
 IOCON_P0_11
 IOCON_P0_13
 IOCON_P0_14
 IOCON_P0_15
 IOCON_P0_16
 IOCON_P0_17 
 IOCON_P0_22 
 IOCON_P0_27
*/
typedef struct {
  __REG32 FUNC                 : 3;
  __REG32 MODE                 : 2;
  __REG32 HYS                  : 1;
  __REG32 INV                  : 1;
  __REG32 ADMODE               : 1;
  __REG32 FILTR                : 1;
  __REG32 SLEW                 : 1;
  __REG32 OD                   : 1;
  __REG32                      :21;
} __sfrdef72_bits;

/*
 Used for registers:
 UART_LCR
*/
typedef struct {
    __REG32 WLS                : 2;
    __REG32 SBS                : 1;
    __REG32 PE                 : 1;
    __REG32 PS                 : 2;
    __REG32 BC                 : 1;
    __REG32 DLAB               : 1;
    __REG32                    :24;
} __sfrdef73_bits;

/*
 Used for registers:
 P0SET
*/
typedef struct {
  __REG32 SET0                 : 1;
  __REG32 SET1                 : 1;
  __REG32 SET2                 : 1;
  __REG32 SET3                 : 1;
  __REG32 SET4                 : 1;
  __REG32 SET5                 : 1;
  __REG32 SET6                 : 1;
  __REG32 SET7                 : 1;
  __REG32 SET8                 : 1;
  __REG32 SET9                 : 1;
  __REG32 SET10                : 1;
  __REG32 SET11                : 1;
  __REG32 SET12                : 1;
  __REG32 SET13                : 1;
  __REG32 SET14                : 1;
  __REG32 SET15                : 1;
  __REG32 SET16                : 1;
  __REG32 SET17                : 1;
  __REG32 SET18                : 1;
  __REG32 SET19                : 1;
  __REG32 SET20                : 1;
  __REG32 SET21                : 1;
  __REG32 SET22                : 1;
  __REG32 SET23                : 1;
  __REG32 SET24                : 1;
  __REG32 SET25                : 1;
  __REG32 SET26                : 1;
  __REG32 SET27                : 1;
  __REG32 SET28                : 1;
  __REG32 SET29                : 1;
  __REG32 SET30                : 1;
  __REG32 SET31                : 1;
} __sfrdef74_bits;

/*
 Used for registers:
 PINTSACT
 PINTSENF
*/
typedef union {
  /* PINTSACT */
  struct {
    __REG8                     : 8;
  };

  /* PINTSENF */
  struct {
    __REG8                     : 8;
  };

} __sfrdef75_bits;

/*
 Used for registers:
 IRCCTRL
*/
typedef struct {
  __REG32 TRIM                 : 8;
  __REG32                      :24;
} __sfrdef76_bits;

/*
 Used for registers:
 IPR7
*/
typedef struct {
  __REG32 PRI_20               : 8;
  __REG32 PRI_29               : 8;
  __REG32 PRI_30               : 8;
  __REG32 PRI_31               : 8;
} __sfrdef77_bits;

/*
 Used for registers:
 UART_RS485ADRMATCH
*/
typedef union {
  /* UART_RS485ADRMATCH */
  struct {
    __REG32 ADRMATCH           : 8;
    __REG32                    :24;
  }; 

} __sfrdef78_bits;

/*
 Used for registers:
 TMR16B1TC
 TMR16B0TC
*/
typedef struct {
  __REG32 TC                   :16;
  __REG32                      :16;
} __sfrdef79_bits;

/*
 Used for registers:
 EECLKDIV
*/
typedef struct {
  __REG32 CLKDIV               :16;
  __REG32                      :16;
} __sfrdef80_bits;

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
} __sfrdef81_bits;

/*
 Used for registers:
 SCB_PINTSEL0
 SCB_PINTSEL1
 SCB_PINTSEL2
 SCB_PINTSEL3
 SCB_PINTSEL4
 SCB_PINTSEL5
 SCB_PINTSEL6
 SCB_PINTSEL7
*/
typedef struct {
  __REG32 INTPIN               : 5;
  __REG32 PORTSEL              : 1;
  __REG32                      :26;
} __sfrdef82_bits;

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
} __sfrdef83_bits;

/*
 Used for registers:
 FLASHCFG
*/
typedef struct {
  __REG32 FLASHTIM0            : 1;
  __REG32 FLASHTIM1            : 1;
  __REG32                      :30;
} __sfrdef84_bits;

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
} __sfrdef85_bits;

/*
 Used for registers:
 I2SCLL
*/
typedef struct {
  __REG32 SCLL                 :16;
  __REG32                      :16;
} __sfrdef86_bits;

/*
 Used for registers:
 UART_MCR
*/
typedef struct {
  __REG32 DTRCTRL            : 1;
  __REG32 RTSCTRL            : 1;
  __REG32                    : 2;
  __REG32 LMS                : 1;
  __REG32                    : 1;
  __REG32 RTSen              : 1;
  __REG32 CTSen              : 1;
  __REG32                    :24;
} __sfrdef87_bits;

/*
 Used for registers:
 P1SET
*/
typedef struct {
  __REG32 SET0                 : 1;
  __REG32 SET1                 : 1;
  __REG32 SET2                 : 1;
  __REG32 SET3                 : 1;
  __REG32 SET4                 : 1;
  __REG32 SET5                 : 1;
  __REG32 SET6                 : 1;
  __REG32 SET7                 : 1;
  __REG32 SET8                 : 1;
  __REG32 SET9                 : 1;
  __REG32                      :22;
} __sfrdef88_bits;

/*
 Used for registers:
 PINTCACT
 PINTCENF
*/
typedef union {
  /* PINTCACT */
  struct {
    __REG8                     : 8;
  };

  /* PINTCENF */
  struct {
    __REG8                     : 8;
  };

} __sfrdef89_bits;

/*
 Used for registers:
 UART_RS485DLY
*/
typedef union {
  /* UART_RS485DLY */
  struct {
    __REG32 DLY                : 8;
    __REG32                    :24;
  };

} __sfrdef90_bits;

/*
 Used for registers:
 TMR16B1PR
 TMR16B0PR
*/
typedef struct {
  __REG32 PCVAL                :16;
  __REG32                      :16;
} __sfrdef91_bits;

/*
 Used for registers:
 EEPWRDWN
*/
typedef struct {
  __REG32 PWRDWIN              : 1;
  __REG32                      :31;
} __sfrdef92_bits;

/*
 Used for registers:
 I2CONCLR
*/
typedef struct {
  __REG32                      : 2;
  __REG32 AAC                  : 1;
  __REG32 SIC                  : 1;
  __REG32                      : 1;
  __REG32 STAC                 : 1;
  __REG32 I2ENC                : 1;
  __REG32                      : 1;
  __REG32                      :24;
} __sfrdef93_bits;

/*
 Used for registers:
 ACOMP_CTL
*/
typedef struct {
  __REG32                      : 3;
  __REG32 EDGESEL              : 2;
  __REG32                      : 1;
  __REG32 COMPSA               : 1;
  __REG32                      : 1;
  __REG32 COMP_VP_SEL          : 3;  
  __REG32 COMP_VM_SEL          : 3;
  __REG32                      : 6;
  __REG32 EDGECLR              : 1;
  __REG32 COMPSTAT             : 1;
  __REG32                      : 1;
  __REG32 COMPEDGE             : 1;
  __REG32                      : 1;
  __REG32 HYS                  : 2;
  __REG32                      : 5;
} __sfrdef94_bits;

/*
 Used for registers:
 DSWER0
*/
typedef struct {
  __REG32 PINDSWKP0            : 1;
  __REG32 PINDSWKP1            : 1;
  __REG32 PINDSWKP2            : 1;
  __REG32 PINDSWKP3            : 1;
  __REG32 PINDSWKP4            : 1;
  __REG32 PINDSWKP5            : 1;
  __REG32 PINDSWKP6            : 1;
  __REG32 PINDSWKP7            : 1;
  __REG32                      :24;
} __sfrdef95_bits;

/*
 Used for registers:
 UART_LSR
*/
typedef struct {
    __REG32 RDR       : 1;
    __REG32 OE        : 1;
    __REG32 PE        : 1;
    __REG32 FE        : 1;
    __REG32 BI        : 1;
    __REG32 THRE      : 1;
    __REG32 TEMT      : 1;
    __REG32 RXFE      : 1;
    __REG32 TXERR     : 1;
    __REG32           :23;
} __sfrdef96_bits;

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
} __sfrdef97_bits;

/*
 Used for registers:
 UART_SYNCCTRL
*/

typedef struct {
    __REG32 SYNC     : 1;
    __REG32 CSRC     : 1;
    __REG32 FES      : 1;
    __REG32 TSBYPASS : 1;
    __REG32 CSCEN    : 1;
    __REG32 SSSDIS   : 1;
    __REG32 CCCLR    : 1;
    __REG32          :25;
} __sfrdef98_bits;

/*
 Used for registers:
 TMR16B1PC
 TMR16B0PC
*/
typedef struct {
  __REG32 PC                   :16;
  __REG32                      :16;
} __sfrdef99_bits;

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
} __sfrdef100_bits;

/*
 Used for registers:
 EEMSSTART
*/
typedef struct {
  __REG32 STARTA               :14;
  __REG32                      :18;
} __sfrdef101_bits;

/*
 Used for registers:
 TMR32B1PR
 TMR32B0PR
*/
typedef struct {
  __REG32 PCVAL                :32;
} __sfrdef102_bits;

/*
 Used for registers:
 I2MMCTRL
*/
typedef struct {
  __REG32 MM_ENA               : 1;
  __REG32 ENA_SCL              : 1;
  __REG32 MATCH_ALL            : 1;
  __REG32                      :29;
} __sfrdef103_bits;

/*
 Used for registers:
 ACOMP_LAD
*/
typedef struct {
  __REG32 LADEN                : 1;
  __REG32 LADSEL               : 5;
  __REG32 LADREF               : 1;
  __REG32                      :25;
} __sfrdef104_bits;

/*
 Used for registers:
 P0PORT
*/
typedef struct {
  __REG32 PORT0                : 1;
  __REG32 PORT1                : 1;
  __REG32 PORT2                : 1;
  __REG32 PORT3                : 1;
  __REG32 PORT4                : 1;
  __REG32 PORT5                : 1;
  __REG32 PORT6                : 1;
  __REG32 PORT7                : 1;
  __REG32 PORT8                : 1;
  __REG32 PORT9                : 1;
  __REG32 PORT10               : 1;
  __REG32 PORT11               : 1;
  __REG32 PORT12               : 1;
  __REG32 PORT13               : 1;
  __REG32 PORT14               : 1;
  __REG32 PORT15               : 1;
  __REG32 PORT16               : 1;
  __REG32 PORT17               : 1;
  __REG32 PORT18               : 1;
  __REG32 PORT19               : 1;
  __REG32 PORT20               : 1;
  __REG32 PORT21               : 1;
  __REG32 PORT22               : 1;
  __REG32 PORT23               : 1;
  __REG32 PORT24               : 1;
  __REG32 PORT25               : 1;
  __REG32 PORT26               : 1;
  __REG32 PORT27               : 1;
  __REG32 PORT28               : 1;
  __REG32 PORT29               : 1;
  __REG32 PORT30               : 1;
  __REG32 PORT31               : 1;
} __sfrdef105_bits;

/*
 Used for registers:
 UART_MSR
*/
typedef struct {
    __REG32 DCTS               : 1;
    __REG32 DDSR               : 1;
    __REG32 TERI               : 1;
    __REG32 DDCD               : 1;
    __REG32 CTS                : 1;
    __REG32 DSR                : 1;
    __REG32 RI                 : 1;
    __REG32 DCD                : 1;
    __REG32                    :24;
} __sfrdef106_bits;

/*
 Used for registers:
 UART_TER
*/
typedef struct {
  __REG32 TXEN                 : 1;
  __REG32                      :31;
} __sfrdef107_bits;

/*
 Used for registers:
 EEMSSTOP
*/
typedef struct {
  __REG32 STOPA                :14;
  __REG32                      :16;
  __REG32 DEVSEL               : 1;
  __REG32 STRTBIST             : 1;
} __sfrdef108_bits;

/*
 Used for registers:
 AD0STAT
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
} __sfrdef109_bits;

/*
 Used for registers:
 FCRA
*/
typedef struct {
  __REG32 CLKDIVSET            :12;
  __REG32                      :20;
} __sfrdef110_bits;

/*
 Used for registers:
 I2ADR1
 I2ADR2
 I2ADR3
*/
typedef struct {
  __REG32 GC                   : 1;
  __REG32 Address              : 7;
  __REG32                      :24;
} __sfrdef111_bits;

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
} __sfrdef112_bits;

/*
 Used for registers:
 P1PORT
*/
typedef struct {
  __REG32 PORT0                : 1;
  __REG32 PORT1                : 1;
  __REG32 PORT2                : 1;
  __REG32 PORT3                : 1;
  __REG32 PORT4                : 1;
  __REG32 PORT5                : 1;
  __REG32 PORT6                : 1;
  __REG32 PORT7                : 1;
  __REG32 PORT8                : 1;
  __REG32 PORT9                : 1;
  __REG32                      :22;
} __sfrdef113_bits;

/*
 Used for registers:
 UART_SCR
*/
typedef union {
  /* UART_SCR */
  struct {
    __REG32 Pad                : 8;
    __REG32                    :24;
  };

} __sfrdef114_bits;

/*
 Used for registers:
 EEMSSIG
*/
typedef struct {
  __REG32 DATA_SIG             :16;
  __REG32 PARITY_SIG           :16;
} __sfrdef115_bits;

/*
 Used for registers:
 P0MASK
*/
typedef struct {
  __REG32 MASK0                : 1;
  __REG32 MASK1                : 1;
  __REG32 MASK2                : 1;
  __REG32 MASK3                : 1;
  __REG32 MASK4                : 1;
  __REG32 MASK5                : 1;
  __REG32 MASK6                : 1;
  __REG32 MASK7                : 1;
  __REG32 MASK8                : 1;
  __REG32 MASK9                : 1;
  __REG32 MASK10               : 1;
  __REG32 MASK11               : 1;
  __REG32 MASK12               : 1;
  __REG32 MASK13               : 1;
  __REG32 MASK14               : 1;
  __REG32 MASK15               : 1;
  __REG32 MASK16               : 1;
  __REG32 MASK17               : 1;
  __REG32 MASK18               : 1;
  __REG32 MASK19               : 1;
  __REG32 MASK20               : 1;
  __REG32 MASK21               : 1;
  __REG32 MASK22               : 1;
  __REG32 MASK23               : 1;
  __REG32 MASK24               : 1;
  __REG32 MASK25               : 1;
  __REG32 MASK26               : 1;
  __REG32 MASK27               : 1;
  __REG32 MASK28               : 1;
  __REG32 MASK29               : 1;
  __REG32 MASK30               : 1;
  __REG32 MASK31               : 1;
} __sfrdef117_bits;

/*
 Used for registers:
 FMSSTART
*/
typedef struct {
  __REG32 START                :17;
  __REG32                      :15;
} __sfrdef118_bits;

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
} __sfrdef119_bits;

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
} __sfrdef120_bits;

/*
 Used for registers:
 UART_ACR
*/
typedef union {
  /* UART_ACR */
  struct {
    __REG32 Start              : 1;
    __REG32 Mode               : 1;
    __REG32 AutoRestart        : 1;
    __REG32                    : 5;
    __REG32 ABEOIntClr         : 1;
    __REG32 ABTOIntClr         : 1;
    __REG32                    :22;
  };

} __sfrdef121_bits;

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
  __REG32 UART                 : 1;
  __REG32 ADC                  : 1;
  __REG32 USB                  : 1;
  __REG32 WDT                  : 1;
  __REG32 IOCON                : 1;
  __REG32                      : 1;
  __REG32 SSP1                 : 1;
  __REG32 PINT                 : 1;
  __REG32 ACOMP                : 1;
  __REG32 DAC                  : 1;
  __REG32                      : 1;
  __REG32 P0INT                : 1;
  __REG32 P1INT                : 1;
  __REG32                      : 7;  
} __sfrdef122_bits;

/*
 Used for registers:
 P1MASK
*/
typedef struct {
  __REG32 MASK0                : 1;
  __REG32 MASK1                : 1;
  __REG32 MASK2                : 1;
  __REG32 MASK3                : 1;
  __REG32 MASK4                : 1;
  __REG32 MASK5                : 1;
  __REG32 MASK6                : 1;
  __REG32 MASK7                : 1;
  __REG32 MASK8                : 1;
  __REG32 MASK9                : 1;
  __REG32                      :22;
} __sfrdef124_bits;

/*
 Used for registers:
 SSP0CR0
 SSP1CR0
*/
typedef struct {
  __REG32 DDS                  : 4;
  __REG32 FRF                  : 2;
  __REG32 CPOL                 : 1;
  __REG32 CPHA                 : 1;
  __REG32 SCR                  : 8;
  __REG32                      :16;
} __sfrdef125_bits;

/*
 Used for registers:
 WDT_TC
 TV
*/
typedef struct {
  __REG32 COUNT                :24;
  __REG32                      : 8;
} __sfrdef126_bits;

/*
 Used for registers:
 FMSSTOP
*/
typedef struct {
  __REG32 STOP                 :17;
  __REG32 SIG_START            : 1;
  __REG32                      :14;
} __sfrdef127_bits;

/*
 Used for registers:
 DSWER1
*/
typedef struct {
  __REG32                      :12;
  __REG32 WDDSWKP              : 1;
  __REG32                      : 3;
  __REG32 BODDSWKP             : 1;
  __REG32                      : 5;  
  __REG32 PORTDSWKP0           : 1;
  __REG32 PORTDSWKP1           : 1;
  __REG32                      : 8;
} __sfrdef128_bits;

/*
 Used for registers:
 P0DIR
*/
typedef struct {
  __REG32 DIR0                 : 1;
  __REG32 DIR1                 : 1;
  __REG32 DIR2                 : 1;
  __REG32 DIR3                 : 1;
  __REG32 DIR4                 : 1;
  __REG32 DIR5                 : 1;
  __REG32 DIR6                 : 1;
  __REG32 DIR7                 : 1;
  __REG32 DIR8                 : 1;
  __REG32 DIR9                 : 1;
  __REG32 DIR10                : 1;
  __REG32 DIR11                : 1;
  __REG32 DIR12                : 1;
  __REG32 DIR13                : 1;
  __REG32 DIR14                : 1;
  __REG32 DIR15                : 1;
  __REG32 DIR16                : 1;
  __REG32 DIR17                : 1;
  __REG32 DIR18                : 1;
  __REG32 DIR19                : 1;
  __REG32 DIR20                : 1;
  __REG32 DIR21                : 1;
  __REG32 DIR22                : 1;
  __REG32 DIR23                : 1;
  __REG32 DIR24                : 1;
  __REG32 DIR25                : 1;
  __REG32 DIR26                : 1;
  __REG32 DIR27                : 1;
  __REG32 DIR28                : 1;
  __REG32 DIR29                : 1;
  __REG32 DIR30                : 1;
  __REG32 DIR31                : 1;
} __sfrdef129_bits;

/*
 Used for registers:
 UART_ICR
*/
typedef union {
  /* UART_ICR */
  struct {
    __REG32 IrDAEn             : 1;
    __REG32 IrDAInv            : 1;
    __REG32 FixPulseEn         : 1;
    __REG32 PulseDiv           : 3;
    __REG32                    :26;
  };

} __sfrdef130_bits;

/*
 Used for registers:
 P0MPORT
*/
typedef struct {
  __REG32 MPORT0               : 1;
  __REG32 MPORT1               : 1;
  __REG32 MPORT2               : 1;
  __REG32 MPORT3               : 1;
  __REG32 MPORT4               : 1;
  __REG32 MPORT5               : 1;
  __REG32 MPORT6               : 1;
  __REG32 MPORT7               : 1;
  __REG32 MPORT8               : 1;
  __REG32 MPORT9               : 1;
  __REG32 MPORT10              : 1;
  __REG32 MPORT11              : 1;
  __REG32 MPORT12              : 1;
  __REG32 MPORT13              : 1;
  __REG32 MPORT14              : 1;
  __REG32 MPORT15              : 1;
  __REG32 MPORT16              : 1;
  __REG32 MPORT17              : 1;
  __REG32 MPORT18              : 1;
  __REG32 MPORT19              : 1;
  __REG32 MPORT20              : 1;
  __REG32 MPORT21              : 1;
  __REG32 MPORT22              : 1;
  __REG32 MPORT23              : 1;
  __REG32 MPORT24              : 1;
  __REG32 MPORT25              : 1;
  __REG32 MPORT26              : 1;
  __REG32 MPORT27              : 1;
  __REG32 MPORT28              : 1;
  __REG32 MPORT29              : 1;
  __REG32 MPORT30              : 1;
  __REG32 MPORT31              : 1;
} __sfrdef131_bits;

/*
 Used for registers:
 P0INTEN
 P1INTEN
 P0INTPOL
 P1INTPOL
*/
typedef union {
  /* P0INTEN */
  struct {
    __REG32 P0INTCONT0         : 1;
    __REG32 P0INTCONT1         : 1;
    __REG32 P0INTCONT2         : 1;
    __REG32 P0INTCONT3         : 1;
    __REG32 P0INTCONT4         : 1;
    __REG32 P0INTCONT5         : 1;
    __REG32 P0INTCONT6         : 1;
    __REG32 P0INTCONT7         : 1;
    __REG32 P0INTCONT8         : 1;
    __REG32 P0INTCONT9         : 1;
    __REG32 P0INTCONT10        : 1;
    __REG32 P0INTCONT11        : 1;
    __REG32 P0INTCONT12        : 1;
    __REG32 P0INTCONT13        : 1;
    __REG32 P0INTCONT14        : 1;
    __REG32 P0INTCONT15        : 1;
    __REG32 P0INTCONT16        : 1;
    __REG32 P0INTCONT17        : 1;
    __REG32 P0INTCONT18        : 1;
    __REG32 P0INTCONT19        : 1;
    __REG32 P0INTCONT20        : 1;
    __REG32 P0INTCONT21        : 1;
    __REG32 P0INTCONT22        : 1;
    __REG32 P0INTCONT23        : 1;
    __REG32 P0INTCONT24        : 1;
    __REG32 P0INTCONT25        : 1;
    __REG32 P0INTCONT26        : 1;
    __REG32 P0INTCONT27        : 1;
    __REG32 P0INTCONT28        : 1;
    __REG32 P0INTCONT29        : 1;
    __REG32 P0INTCONT30        : 1;
    __REG32 P0INTCONT31        : 1;
  };

  /* P1INTEN */
  struct {
    __REG32 P1INTCONT0         : 1;
    __REG32 P1INTCONT1         : 1;
    __REG32 P1INTCONT2         : 1;
    __REG32 P1INTCONT3         : 1;
    __REG32 P1INTCONT4         : 1;
    __REG32 P1INTCONT5         : 1;
    __REG32 P1INTCONT6         : 1;
    __REG32 P1INTCONT7         : 1;
    __REG32 P1INTCONT8         : 1;
    __REG32 P1INTCONT9         : 1;
    __REG32                    :22;
  };

  /* P0INTPOL */
  struct {
    __REG32 P0INTPOL_P0CONTINV :32;
  };

  /* P1INTPOL */
  struct {
    __REG32 P1INTPOL_P0CONTINV :10;
    __REG32                    :22;
  };

} __sfrdef132_bits;

/*
 Used for registers:
 FEED
*/
typedef struct {
  __REG32 FEED                 : 8;
  __REG32                      :24;
} __sfrdef133_bits;

/*
 Used for registers:
 FMS16
*/
typedef struct {
  __REG32 FMS16                :16;
  __REG32                      :16;
} __sfrdef134_bits;

/*
 Used for registers:
 SYSMEMREMAP
*/
typedef struct {
  __REG32 MAP                  : 2;
  __REG32                      :30;
} __sfrdef135_bits;

/*
 Used for registers:
 P1DIR
*/
typedef struct {
  __REG32 DIR0                 : 1;
  __REG32 DIR1                 : 1;
  __REG32 DIR2                 : 1;
  __REG32 DIR3                 : 1;
  __REG32 DIR4                 : 1;
  __REG32 DIR5                 : 1;
  __REG32 DIR6                 : 1;
  __REG32 DIR7                 : 1;
  __REG32 DIR8                 : 1;
  __REG32 DIR9                 : 1;
  __REG32                      :22;
} __sfrdef136_bits;

/*
 Used for registers:
 UART_FDR
*/
typedef union {
  /* UART_FDR */
  struct {
    __REG32 DIVADDVAL          : 4;
    __REG32 MULVAL             : 4;
    __REG32                    :24;
  };

} __sfrdef137_bits;

/*
 Used for registers:
 P1MPORT
*/
typedef struct {
  __REG32 MPORT0               : 1;
  __REG32 MPORT1               : 1;
  __REG32 MPORT2               : 1;
  __REG32 MPORT3               : 1;
  __REG32 MPORT4               : 1;
  __REG32 MPORT5               : 1;
  __REG32 MPORT6               : 1;
  __REG32 MPORT7               : 1;
  __REG32 MPORT8               : 1;
  __REG32 MPORT9               : 1;
  __REG32                      :22;
} __sfrdef138_bits;

/*
 Used for registers:
 BODR
*/
typedef struct {
  __REG32 BODRSTLEV            : 2;
  __REG32 BODINTVAL            : 2;
  __REG32 BODRSTENA            : 1;
  __REG32 BODRES               : 1;
  __REG32 BODINT               : 1;
  __REG32                      :25;
} __sfrdef139_bits;

/*
 Used for registers:
 FMSW0
*/
typedef struct {
  __REG32 FMSW00               : 1;
  __REG32 FMSW01               : 1;
  __REG32 FMSW02               : 1;
  __REG32 FMSW03               : 1;
  __REG32 FMSW04               : 1;
  __REG32 FMSW05               : 1;
  __REG32 FMSW06               : 1;
  __REG32 FMSW07               : 1;
  __REG32 FMSW08               : 1;
  __REG32 FMSW09               : 1;
  __REG32 FMSW010              : 1;
  __REG32 FMSW011              : 1;
  __REG32 FMSW012              : 1;
  __REG32 FMSW013              : 1;
  __REG32 FMSW014              : 1;
  __REG32 FMSW015              : 1;
  __REG32 FMSW016              : 1;
  __REG32 FMSW017              : 1;
  __REG32 FMSW018              : 1;
  __REG32 FMSW019              : 1;
  __REG32 FMSW020              : 1;
  __REG32 FMSW021              : 1;
  __REG32 FMSW022              : 1;
  __REG32 FMSW023              : 1;
  __REG32 FMSW024              : 1;
  __REG32 FMSW025              : 1;
  __REG32 FMSW026              : 1;
  __REG32 FMSW027              : 1;
  __REG32 FMSW028              : 1;
  __REG32 FMSW029              : 1;
  __REG32 FMSW030              : 1;
  __REG32 FMSW031              : 1;
} __sfrdef140_bits;

/*
 Used for registers:
 PRESETCTRL
*/
typedef struct {
  __REG32 SSP0_RST_N           : 1;
  __REG32 I2C_RST_N            : 1;
  __REG32 SSP1_RST_N           : 1;
  __REG32                      : 1;
  __REG32 UART_RST_N           : 1;
  __REG32 CT16B0_RST_N         : 1;
  __REG32 CT16B1_RST_N         : 1;
  __REG32 CT32B0_RST_N         : 1;
  __REG32 CT32B1_RST_N         : 1;
  __REG32 ACOMP_RST_N          : 1;
  __REG32 ADC_RST_N            : 1;
  __REG32 DAC_RST_N            : 1;
  __REG32                      :20;
} __sfrdef141_bits;

/*
 Used for registers:
 UART_OSR
*/
typedef union {
  /* UART_OSR */
  struct {
    __REG32                    : 1;
    __REG32 OSFrac             : 3;
    __REG32 OSInt              : 4;
    __REG32 FDInt              : 7;
    __REG32                    :17;
  };

} __sfrdef142_bits;

/*
 Used for registers:
 AD0CR
*/

typedef union {
   /* AD0CR */
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
    __REG32 CLKS                 : 3;
    __REG32                      : 3;
    __REG32 START                : 4;
    __REG32 EDGE                 : 1;
    __REG32                      : 4;
  };
  
  /* AD0CR */
  struct {
    __REG32 SEL                  :  8;
    __REG32                      : 24;
  };
   
} __sfrdef143_bits;



/*
 Used for registers:
 DEVICE_ID
*/
typedef struct {
  __REG32 DEVICEID             :32;
} __sfrdef144_bits;

/*
 Used for registers:
 FMSW1
*/
typedef struct {
  __REG32 FMSW10               : 1;
  __REG32 FMSW11               : 1;
  __REG32 FMSW12               : 1;
  __REG32 FMSW13               : 1;
  __REG32 FMSW14               : 1;
  __REG32 FMSW15               : 1;
  __REG32 FMSW16               : 1;
  __REG32 FMSW17               : 1;
  __REG32 FMSW18               : 1;
  __REG32 FMSW19               : 1;
  __REG32 FMSW110              : 1;
  __REG32 FMSW111              : 1;
  __REG32 FMSW112              : 1;
  __REG32 FMSW113              : 1;
  __REG32 FMSW114              : 1;
  __REG32 FMSW115              : 1;
  __REG32 FMSW116              : 1;
  __REG32 FMSW117              : 1;
  __REG32 FMSW118              : 1;
  __REG32 FMSW119              : 1;
  __REG32 FMSW120              : 1;
  __REG32 FMSW121              : 1;
  __REG32 FMSW122              : 1;
  __REG32 FMSW123              : 1;
  __REG32 FMSW124              : 1;
  __REG32 FMSW125              : 1;
  __REG32 FMSW126              : 1;
  __REG32 FMSW127              : 1;
  __REG32 FMSW128              : 1;
  __REG32 FMSW129              : 1;
  __REG32 FMSW130              : 1;
  __REG32 FMSW131              : 1;
} __sfrdef145_bits;

/*
 Used for registers:
 CLKSEL
*/
typedef struct {
  __REG32 CLKSEL               : 1;
  __REG32                      :30;
  __REG32 LOCK                 : 1;
} __sfrdef146_bits;

/*
 Used for registers:
 UART_TER
*/
typedef struct {
  __REG32                      : 7;
  __REG32 TXEN                 : 1;
  __REG32                      :24;
} __sfrdef147_bits;

/*
 Used for registers:
 SYSPLLCTRL
*/
typedef struct {
  __REG32 MSEL                 : 5;
  __REG32 PSEL                 : 2;
  __REG32                      :25;
} __sfrdef148_bits;

/*
 Used for registers:
 SYSTCKCAL
*/
typedef struct {
  __REG32 CAL                  :26;
  __REG32                      : 6;
} __sfrdef149_bits;

/*
 Used for registers:
 SYST_CSR
*/
typedef struct {
  __REG32 ENABLE               : 1;
  __REG32 TICKINT              : 1;
  __REG32 CLKSOURCE            : 1;
  __REG32                      :13;
  __REG32 COUNTFLAG            : 1;
  __REG32                      :15;
} __sfrdef150_bits;

/*
 Used for registers:
 AD0GDR
*/
typedef struct {
  __REG32                      : 6;
  __REG32 V_VREF               :10;
  __REG32                      : 8;
  __REG32 CHN                  : 3;
  __REG32                      : 3;
  __REG32 OVERRUN              : 1;
  __REG32 DONE                 : 1;
} __sfrdef152_bits;

/*
 Used for registers:
 FMSW2
*/
typedef struct {
  __REG32 FMSW20               : 1;
  __REG32 FMSW21               : 1;
  __REG32 FMSW22               : 1;
  __REG32 FMSW23               : 1;
  __REG32 FMSW24               : 1;
  __REG32 FMSW25               : 1;
  __REG32 FMSW26               : 1;
  __REG32 FMSW27               : 1;
  __REG32 FMSW28               : 1;
  __REG32 FMSW29               : 1;
  __REG32 FMSW210              : 1;
  __REG32 FMSW211              : 1;
  __REG32 FMSW212              : 1;
  __REG32 FMSW213              : 1;
  __REG32 FMSW214              : 1;
  __REG32 FMSW215              : 1;
  __REG32 FMSW216              : 1;
  __REG32 FMSW217              : 1;
  __REG32 FMSW218              : 1;
  __REG32 FMSW219              : 1;
  __REG32 FMSW220              : 1;
  __REG32 FMSW221              : 1;
  __REG32 FMSW222              : 1;
  __REG32 FMSW223              : 1;
  __REG32 FMSW224              : 1;
  __REG32 FMSW225              : 1;
  __REG32 FMSW226              : 1;
  __REG32 FMSW227              : 1;
  __REG32 FMSW228              : 1;
  __REG32 FMSW229              : 1;
  __REG32 FMSW230              : 1;
  __REG32 FMSW231              : 1;
} __sfrdef153_bits;

/*
 Used for registers:
 WARNINT
*/
typedef struct {
  __REG32 WARNINT              :10;
  __REG32                      :22;
} __sfrdef154_bits;

/*
 Used for registers:
 DACR
*/
typedef struct {
  __REG32                      : 6;
  __REG32 VALUE                :10;
  __REG32 BIAS                 : 1;
  __REG32 TRIG                 : 3;
  __REG32                      : 1;
  __REG32 EDGESEL              : 2;
  __REG32 TRIGERD              : 1;
  __REG32                      : 8;
} __sfrdef155_bits;

/*
 Used for registers:
 SYSPLLSTAT
*/
typedef struct {
  __REG32 LOCK                 : 1;
  __REG32                      :31;
} __sfrdef156_bits;

/*
 Used for registers:
 IPR0
*/
typedef struct {
  __REG32 PRI_0                : 8;
  __REG32 PRI_1                : 8;
  __REG32 PRI_2                : 8;
  __REG32 PRI_3                : 8;
} __sfrdef157_bits;

/*
 Used for registers:
 SYST_RVR
*/
typedef struct {
  __REG32 RELOAD               :24;
  __REG32                      : 8;
} __sfrdef158_bits;


#endif    /* __IAR_SYSTEMS_ICC__ */

/***************************************************************************
**
** WPIN
**
***************************************************************************/
__IO_REG32(P0W0,                0x50001000,__READ_WRITE);
__IO_REG32(P0W1,                0x50001004,__READ_WRITE);
__IO_REG32(P0W2,                0x50001008,__READ_WRITE);
__IO_REG32(P0W3,                0x5000100C,__READ_WRITE);
__IO_REG32(P0W4,                0x50001010,__READ_WRITE);
__IO_REG32(P0W5,                0x50001014,__READ_WRITE);
__IO_REG32(P0W6,                0x50001018,__READ_WRITE);
__IO_REG32(P0W7,                0x5000101C,__READ_WRITE);
__IO_REG32(P0W8,                0x50001020,__READ_WRITE);
__IO_REG32(P0W9,                0x50001024,__READ_WRITE);
__IO_REG32(P0W10,               0x50001028,__READ_WRITE);
__IO_REG32(P0W11,               0x5000102C,__READ_WRITE);
__IO_REG32(P0W12,               0x50001030,__READ_WRITE);
__IO_REG32(P0W13,               0x50001034,__READ_WRITE);
__IO_REG32(P0W14,               0x50001038,__READ_WRITE);
__IO_REG32(P0W15,               0x5000103C,__READ_WRITE);
__IO_REG32(P0W16,               0x50001040,__READ_WRITE);
__IO_REG32(P0W17,               0x50001044,__READ_WRITE);
__IO_REG32(P0W18,               0x50001048,__READ_WRITE);
__IO_REG32(P0W19,               0x5000104C,__READ_WRITE);
__IO_REG32(P0W20,               0x50001050,__READ_WRITE);
__IO_REG32(P0W21,               0x50001054,__READ_WRITE);
__IO_REG32(P0W22,               0x50001058,__READ_WRITE);
__IO_REG32(P0W23,               0x5000105C,__READ_WRITE);
__IO_REG32(P0W24,               0x50001060,__READ_WRITE);
__IO_REG32(P0W25,               0x50001064,__READ_WRITE);
__IO_REG32(P0W26,               0x50001068,__READ_WRITE);
__IO_REG32(P0W27,               0x5000106C,__READ_WRITE);
__IO_REG32(P0W28,               0x50001070,__READ_WRITE);
__IO_REG32(P0W29,               0x50001074,__READ_WRITE);
__IO_REG32(P0W30,               0x50001078,__READ_WRITE);
__IO_REG32(P0W31,               0x5000107C,__READ_WRITE);
__IO_REG32(P1W0,                0x50001080,__READ_WRITE);
__IO_REG32(P1W1,                0x50001084,__READ_WRITE);
__IO_REG32(P1W2,                0x50001088,__READ_WRITE);
__IO_REG32(P1W3,                0x5000108C,__READ_WRITE);
__IO_REG32(P1W4,                0x50001090,__READ_WRITE);
__IO_REG32(P1W5,                0x50001094,__READ_WRITE);
__IO_REG32(P1W6,                0x50001098,__READ_WRITE);
__IO_REG32(P1W7,                0x5000109C,__READ_WRITE);
__IO_REG32(P1W8,                0x500010A0,__READ_WRITE);
__IO_REG32(P1W9,                0x500010A4,__READ_WRITE);

/***************************************************************************
**
** SSP_SPI0
**
***************************************************************************/
__IO_REG32_BIT(SSP0CR0,        0x40040000,__READ_WRITE,__sfrdef125_bits);
__IO_REG32_BIT(SSP0CR1,        0x40040004,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(SSP0DR,         0x40040008,__READ_WRITE,__sfrdef18_bits);
__IO_REG32_BIT(SSP0SR,         0x4004000C,__READ,__sfrdef28_bits);
__IO_REG32_BIT(SSP0CPSR,       0x40040010,__READ_WRITE,__sfrdef38_bits);
__IO_REG32_BIT(SSP0IMSC,       0x40040014,__READ_WRITE,__sfrdef10_bits);
__IO_REG32_BIT(SSP0RIS,        0x40040018,__READ,__sfrdef21_bits);
__IO_REG32_BIT(SSP0MIS,        0x4004001C,__READ,__sfrdef32_bits);
__IO_REG32_BIT(SSP0ICR,        0x40040020,__WRITE,__sfrdef42_bits);

/***************************************************************************
**
** NVIC
**
***************************************************************************/
__IO_REG32_BIT(ISER,                0xE000E100,__READ_WRITE,__sfrdef100_bits);
__IO_REG32_BIT(ICER,                0xE000E180,__READ_WRITE,__sfrdef85_bits);
__IO_REG32_BIT(ISPR,                0xE000E200,__READ_WRITE,__sfrdef67_bits);
__IO_REG32_BIT(ICPR,                0xE000E280,__READ_WRITE,__sfrdef48_bits);
__IO_REG32_BIT(IPR0,                0xE000E400,__READ_WRITE,__sfrdef157_bits);
__IO_REG32_BIT(IPR1,                0xE000E404,__READ_WRITE,__sfrdef14_bits);
__IO_REG32_BIT(IPR2,                0xE000E408,__READ_WRITE,__sfrdef24_bits);
__IO_REG32_BIT(IPR3,                0xE000E40C,__READ_WRITE,__sfrdef35_bits);
__IO_REG32_BIT(IPR4,                0xE000E410,__READ_WRITE,__sfrdef47_bits);
__IO_REG32_BIT(IPR5,                0xE000E414,__READ_WRITE,__sfrdef54_bits);
__IO_REG32_BIT(IPR6,                0xE000E418,__READ_WRITE,__sfrdef63_bits);
__IO_REG32_BIT(IPR7,                0xE000E41C,__READ_WRITE,__sfrdef77_bits);
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
** AD0
**
***************************************************************************/
__IO_REG32_BIT(AD0CR,                  0x4001C000,__READ_WRITE,__sfrdef143_bits);
__IO_REG32_BIT(AD0SEL,                 0x4001C008,__READ_WRITE,__sfrdef4_bits);
__IO_REG32_BIT(AD0GDR,                 0x4001C004,__READ_WRITE,__sfrdef152_bits);
__IO_REG32_BIT(AD0INTEN,               0x4001C00C,__READ_WRITE,__sfrdef19_bits);
__IO_REG32_BIT(AD0DR0,                 0x4001C010,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR1,                 0x4001C014,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR2,                 0x4001C018,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR3,                 0x4001C01C,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR4,                 0x4001C020,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR5,                 0x4001C024,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR6,                 0x4001C028,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0DR7,                 0x4001C02C,__READ_WRITE,__sfrdef29_bits);
__IO_REG32_BIT(AD0STAT,                0x4001C030,__READ,__sfrdef109_bits);

/***************************************************************************
**
** SSP_SPI1
**
***************************************************************************/
__IO_REG32_BIT(SSP1CR0,        0x40058000,__READ_WRITE,__sfrdef125_bits);
__IO_REG32_BIT(SSP1CR1,        0x40058004,__READ_WRITE,__sfrdef3_bits);
__IO_REG32_BIT(SSP1DR,         0x40058008,__READ_WRITE,__sfrdef18_bits);
__IO_REG32_BIT(SSP1SR,         0x4005800C,__READ,__sfrdef28_bits);
__IO_REG32_BIT(SSP1CPSR,       0x40058010,__READ_WRITE,__sfrdef38_bits);
__IO_REG32_BIT(SSP1IMSC,       0x40058014,__READ_WRITE,__sfrdef10_bits);
__IO_REG32_BIT(SSP1RIS,        0x40058018,__READ,__sfrdef21_bits);
__IO_REG32_BIT(SSP1MIS,        0x4005801C,__READ,__sfrdef32_bits);
__IO_REG32_BIT(SSP1ICR,        0x40058020,__WRITE,__sfrdef42_bits);

/***************************************************************************
**
** PINTCON
**
***************************************************************************/
__IO_REG8_BIT(PINTMODE,            0x5000C000,__READ_WRITE,__sfrdef20_bits);
__IO_REG8_BIT(PINTEN,              0x5000C004,__READ_WRITE,__sfrdef31_bits);
#define PINTENR              PINTEN
#define PINTENR_bit          PINTEN_bit
__IO_REG8_BIT(PINTSEN,             0x5000C008,__WRITE,__sfrdef41_bits);
#define PINTSENR             PINTSEN
#define PINTSENR_bit         PINTSEN_bit
__IO_REG8_BIT(PINTCEN,             0x5000C00C,__WRITE,__sfrdef50_bits);
#define PINTCENR             PINTCEN
#define PINTCENR_bit         PINTCEN_bit
__IO_REG8(PINTACT,             0x5000C010,__READ_WRITE);
#define PINTENF              PINTACT
__IO_REG8(PINTSACT,            0x5000C014,__WRITE);
#define PINTSENF             PINTSACT
__IO_REG8(PINTCACT,            0x5000C018,__WRITE);
#define PINTCENF             PINTCACT
__IO_REG8(PINTRISE,            0x5000C01C,__READ_WRITE);
__IO_REG8(PINTFALL,            0x5000C020,__READ_WRITE);
__IO_REG8(PINTST,              0x5000C024,__READ_WRITE);
__IO_REG32_BIT(P0INTEN,             0x50011000,__READ_WRITE,__sfrdef132_bits);
#define P1INTEN              P0INTEN
#define P1INTEN_bit          P0INTEN_bit
#define P0INTPOL             P0INTEN
#define P0INTPOL_bit         P0INTEN_bit
#define P1INTPOL             P0INTEN
#define P1INTPOL_bit         P0INTEN_bit

/***************************************************************************
**
** EEPROM
**
***************************************************************************/
__IO_REG32_BIT(FCRA,                0x4003C01C,__READ_WRITE,__sfrdef110_bits);
__IO_REG32_BIT(FMSSTART,            0x4003C020,__READ_WRITE,__sfrdef118_bits);
__IO_REG32_BIT(FMSSTOP,             0x4003C024,__READ_WRITE,__sfrdef127_bits);
__IO_REG32_BIT(FMS16,               0x4003C028,__READ,__sfrdef134_bits);
__IO_REG32_BIT(FMSW0,               0x4003C02C,__READ,__sfrdef140_bits);
__IO_REG32_BIT(FMSW1,               0x4003C030,__READ,__sfrdef145_bits);
__IO_REG32_BIT(FMSW2,               0x4003C034,__READ,__sfrdef153_bits);
__IO_REG32_BIT(FMSW3,               0x4003C038,__READ,__sfrdef8_bits);
__IO_REG32_BIT(EECMD,               0x4003C080,__READ_WRITE,__sfrdef23_bits);
__IO_REG32_BIT(EEADDR,              0x4003C084,__READ_WRITE,__sfrdef34_bits);
__IO_REG32_BIT(EEWDATA,             0x4003C088,__WRITE,__sfrdef46_bits);
__IO_REG32_BIT(EERDATA,             0x4003C08C,__READ,__sfrdef53_bits);
__IO_REG32_BIT(EEWSTATE,            0x4003C090,__READ_WRITE,__sfrdef62_bits);
__IO_REG32_BIT(EECLKDIV,            0x4003C094,__READ_WRITE,__sfrdef80_bits);
__IO_REG32_BIT(EEPWRDWN,            0x4003C098,__READ_WRITE,__sfrdef92_bits);
__IO_REG32_BIT(EEMSSTART,           0x4003C09C,__READ_WRITE,__sfrdef101_bits);
__IO_REG32_BIT(EEMSSTOP,            0x4003C0A0,__READ_WRITE,__sfrdef108_bits);
__IO_REG32_BIT(EEMSSIG,             0x4003C0A4,__READ,__sfrdef115_bits);

/***************************************************************************
**
** CT16B0
**
***************************************************************************/
__IO_REG32_BIT(TMR16B0IR,           0x4000C000,__READ_WRITE,__sfrdef56_bits);
__IO_REG32_BIT(TMR16B0TCR,          0x4000C004,__READ_WRITE,__sfrdef66_bits);
__IO_REG32_BIT(TMR16B0TC,           0x4000C008,__READ_WRITE,__sfrdef79_bits);
__IO_REG32_BIT(TMR16B0PR,           0x4000C00C,__READ_WRITE,__sfrdef91_bits);
__IO_REG32_BIT(TMR16B0PC,           0x4000C010,__READ_WRITE,__sfrdef99_bits);
__IO_REG32_BIT(TMR16B0MCR,          0x4000C014,__READ_WRITE,__sfrdef13_bits);
__IO_REG32_BIT(TMR16B0MR0,          0x4000C018,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B0MR1,          0x4000C01C,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B0MR2,          0x4000C020,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B0MR3,          0x4000C024,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B0CCR,          0x4000C028,__READ_WRITE,__sfrdef65_bits);
__IO_REG32_BIT(TMR16B0CR0,          0x4000C02C,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B0CR1,          0x4000C030,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B0CR2,          0x4000C034,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B0CR3,          0x4000C038,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B0EMR,          0x4000C03C,__READ_WRITE,__sfrdef37_bits);
__IO_REG32_BIT(TMR16B0CTCR,         0x4000C070,__READ_WRITE,__sfrdef83_bits);
__IO_REG32_BIT(TMR16B0PWMC,         0x4000C074,__READ_WRITE,__sfrdef5_bits);

/***************************************************************************
**
** I2C
**
***************************************************************************/
__IO_REG32_BIT(I2CONSET,            0x40000000,__READ_WRITE,__sfrdef27_bits);
__IO_REG32_BIT(I2STAT,              0x40000004,__READ,__sfrdef39_bits);
__IO_REG32_BIT(I2DAT,               0x40000008,__READ_WRITE,__sfrdef49_bits);
__IO_REG32_BIT(I2ADR0,              0x4000000C,__READ_WRITE,__sfrdef58_bits);
__IO_REG32_BIT(I2SCLH,              0x40000010,__READ_WRITE,__sfrdef71_bits);
__IO_REG32_BIT(I2SCLL,              0x40000014,__READ_WRITE,__sfrdef86_bits);
__IO_REG32_BIT(I2CONCLR,            0x40000018,__WRITE,__sfrdef93_bits);
__IO_REG32_BIT(I2MMCTRL,            0x4000001C,__READ_WRITE,__sfrdef103_bits);
__IO_REG32_BIT(I2ADR1,              0x40000020,__READ_WRITE,__sfrdef111_bits);
__IO_REG32_BIT(I2ADR2,              0x40000024,__READ_WRITE,__sfrdef111_bits);
__IO_REG32_BIT(I2ADR3,              0x40000028,__READ_WRITE,__sfrdef111_bits);
__IO_REG32_BIT(I2DATA_BUFFER,       0x4000002C,__READ,__sfrdef49_bits);
__IO_REG32_BIT(I2MASK0,             0x40000030,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(I2MASK1,             0x40000034,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(I2MASK2,             0x40000038,__READ_WRITE,__sfrdef9_bits);
__IO_REG32_BIT(I2MASK3,             0x4000003C,__READ_WRITE,__sfrdef9_bits);

/***************************************************************************
**
** PINTSEL
**
***************************************************************************/
__IO_REG8(PINTSEL0,                 0x50008178,__READ_WRITE);
__IO_REG8(PINTSEL1,                 0x5000817C,__READ_WRITE);
__IO_REG8(PINTSEL2,                 0x50008180,__READ_WRITE);
__IO_REG8(PINTSEL3,                 0x50008184,__READ_WRITE);
__IO_REG8(PINTSEL4,                 0x50008188,__READ_WRITE);
__IO_REG8(PINTSEL5,                 0x5000818C,__READ_WRITE);
__IO_REG8(PINTSEL6,                 0x50008190,__READ_WRITE);
__IO_REG8(PINTSEL7,                 0x50008194,__READ_WRITE);

/***************************************************************************
**
** CT16B1
**
***************************************************************************/
__IO_REG32_BIT(TMR16B1IR,           0x40010000,__READ_WRITE,__sfrdef56_bits);
__IO_REG32_BIT(TMR16B1TCR,          0x40010004,__READ_WRITE,__sfrdef66_bits);
__IO_REG32_BIT(TMR16B1TC,           0x40010008,__READ_WRITE,__sfrdef79_bits);
__IO_REG32_BIT(TMR16B1PR,           0x4001000C,__READ_WRITE,__sfrdef91_bits);
__IO_REG32_BIT(TMR16B1PC,           0x40010010,__READ_WRITE,__sfrdef99_bits);
__IO_REG32_BIT(TMR16B1MCR,          0x40010014,__READ_WRITE,__sfrdef13_bits);
__IO_REG32_BIT(TMR16B1MR0,          0x40010018,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B1MR1,          0x4001001C,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B1MR2,          0x40010020,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B1MR3,          0x40010024,__READ_WRITE,__sfrdef22_bits);
__IO_REG32_BIT(TMR16B1CCR,          0x40010028,__READ_WRITE,__sfrdef65_bits);
__IO_REG32_BIT(TMR16B1CR0,          0x4001002C,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B1CR1,          0x40010030,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B1CR2,          0x40010034,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B1CR3,          0x40010038,__READ,__sfrdef16_bits);
__IO_REG32_BIT(TMR16B1EMR,          0x4001003C,__READ_WRITE,__sfrdef37_bits);
__IO_REG32_BIT(TMR16B1CTCR,         0x40010070,__READ_WRITE,__sfrdef83_bits);
__IO_REG32_BIT(TMR16B1PWMC,         0x40010074,__READ_WRITE,__sfrdef5_bits);

/***************************************************************************
**
** PINCON
**
***************************************************************************/
__IO_REG32_BIT(P0DIR,               0x50002000,__READ_WRITE,__sfrdef129_bits);
__IO_REG32_BIT(P1DIR,               0x50002004,__READ_WRITE,__sfrdef136_bits);
__IO_REG32_BIT(P0MASK,              0x50002080,__READ_WRITE,__sfrdef117_bits);
__IO_REG32_BIT(P1MASK,              0x50002084,__READ_WRITE,__sfrdef124_bits);
__IO_REG32_BIT(P0PORT,              0x50002100,__READ_WRITE,__sfrdef105_bits);
__IO_REG32_BIT(P1PORT,              0x50002104,__READ_WRITE,__sfrdef113_bits);
__IO_REG32_BIT(P0MPORT,             0x50004180,__READ_WRITE,__sfrdef131_bits);
__IO_REG32_BIT(P1MPORT,             0x50004184,__READ_WRITE,__sfrdef138_bits);
__IO_REG32_BIT(P0SET,               0x50002200,__READ_WRITE,__sfrdef74_bits);
__IO_REG32_BIT(P1SET,               0x50002204,__READ_WRITE,__sfrdef88_bits);
__IO_REG32_BIT(P0CLR,               0x50002280,__WRITE,__sfrdef57_bits);
__IO_REG32_BIT(P1CLR,               0x50002284,__WRITE,__sfrdef70_bits);
__IO_REG32_BIT(P0NOT,               0x50002300,__WRITE,__sfrdef44_bits);
__IO_REG32_BIT(P1NOT,               0x50002304,__WRITE,__sfrdef52_bits);

/***************************************************************************
**
** DAC
**
***************************************************************************/
__IO_REG32_BIT(DACR,                0x40024000,__READ_WRITE,__sfrdef155_bits);

/***************************************************************************
**
** WDT
**
***************************************************************************/
__IO_REG32_BIT(MOD,                 0x40004000,__READ_WRITE,__sfrdef119_bits);
__IO_REG32_BIT(WDT_TC,              0x40004004,__READ_WRITE,__sfrdef126_bits);
__IO_REG32_BIT(FEED,                0x40004008,__WRITE,__sfrdef133_bits);
__IO_REG32_BIT(TV,                  0x4000400C,__READ,__sfrdef126_bits);
__IO_REG32_BIT(CLKSEL,              0x40004010,__READ_WRITE,__sfrdef146_bits);
__IO_REG32_BIT(WARNINT,             0x40004014,__READ_WRITE,__sfrdef154_bits);
__IO_REG32_BIT(WDWINDOW,              0x40004018,__READ_WRITE,__sfrdef11_bits);

/***************************************************************************
**
** SYST
**
***************************************************************************/
__IO_REG32_BIT(SYST_CSR,            0xE000E010,__READ_WRITE,__sfrdef150_bits);
__IO_REG32_BIT(SYST_RVR,            0xE000E014,__READ_WRITE,__sfrdef158_bits);
__IO_REG32_BIT(SYST_CVR,            0xE000E018,__READ_WRITE,__sfrdef17_bits);
__IO_REG32_BIT(SYST_CALIB,          0xE000E01C,__READ_WRITE,__sfrdef26_bits);

/***************************************************************************
**
** UART
**
***************************************************************************/
__IO_REG32_BIT(UART_RBR,            0x40008000,__READ_WRITE,__sfrdef43_bits);
#define UART_THR             UART_RBR
#define UART_THR_bit         UART_RBR_bit
#define UART_DLL             UART_RBR
#define UART_DLL_bit         UART_RBR_bit
__IO_REG32_BIT(UART_DLM,            0x40008004,__READ_WRITE,__sfrdef51_bits);
#define UART_IER             UART_DLM
#define UART_IER_bit         UART_DLM_bit
__IO_REG32_BIT(UART_IIR,            0x40008008,__READ_WRITE,__sfrdef59_bits);
#define UART_FCR             UART_IIR
#define UART_FCR_bit         UART_IIR_bit
__IO_REG32_BIT(UART_LCR,            0x4000800C,__READ_WRITE,__sfrdef73_bits);
__IO_REG32_BIT(UART_MCR,            0x40008010,__READ_WRITE,__sfrdef87_bits);
__IO_REG32_BIT(UART_LSR,            0x40008014,__READ,__sfrdef96_bits);
__IO_REG32_BIT(UART_MSR,            0x40008018,__READ,__sfrdef106_bits);
__IO_REG32_BIT(UART_SCR,            0x4000801C,__READ_WRITE,__sfrdef114_bits);
__IO_REG32_BIT(UART_ACR,            0x40008020,__READ_WRITE,__sfrdef121_bits);
__IO_REG32_BIT(UART_ICR,            0x40008024,__READ_WRITE,__sfrdef130_bits);
__IO_REG32_BIT(UART_FDR,            0x40008028,__READ_WRITE,__sfrdef137_bits);
__IO_REG32_BIT(UART_OSR,            0x4000802C,__READ_WRITE,__sfrdef142_bits);
__IO_REG32_BIT(UART_SCICTRL,        0x40008048,__READ_WRITE,__sfrdef55_bits);
__IO_REG32_BIT(UART_RS485CTRL,      0x4000804C,__READ_WRITE,__sfrdef64_bits);
__IO_REG32_BIT(UART_RS485ADRMATCH,  0x40008050,__READ_WRITE,__sfrdef78_bits);
__IO_REG32_BIT(UART_RS485DLY,       0x40008054,__READ_WRITE,__sfrdef90_bits);
__IO_REG32_BIT(UART_SYNCCTRL,       0x40008058,__READ_WRITE,__sfrdef98_bits);
__IO_REG32_BIT(UART_TER,            0x4000805C,__READ_WRITE,__sfrdef107_bits);

/***************************************************************************
**
** SCB
**
***************************************************************************/
__IO_REG32_BIT(SYSMEMREMAP,         0x40048000,__READ_WRITE,__sfrdef135_bits);
__IO_REG32_BIT(PRESETCTRL,          0x40048004,__READ_WRITE,__sfrdef141_bits);
__IO_REG32_BIT(SYSPLLCTRL,          0x40048008,__READ_WRITE,__sfrdef148_bits);
__IO_REG32_BIT(SYSPLLSTAT,          0x4004800C,__READ,__sfrdef156_bits);
__IO_REG32_BIT(WDTOSCCTRL,          0x40048024,__READ_WRITE,__sfrdef61_bits);
__IO_REG32_BIT(IRCCTRL,             0x40048028,__READ_WRITE,__sfrdef76_bits);
__IO_REG32_BIT(SYSRSTSTAT,          0x40048030,__READ_WRITE,__sfrdef97_bits);
__IO_REG32_BIT(SYSPLLCLKSEL,        0x40048040,__READ_WRITE,__sfrdef30_bits);
__IO_REG32_BIT(SYSPLLCLKUEN,        0x40048044,__READ_WRITE,__sfrdef40_bits);
__IO_REG32_BIT(MAINCLKSEL,          0x40048070,__READ_WRITE,__sfrdef30_bits);
__IO_REG32_BIT(MAINCLKUEN,          0x40048074,__READ_WRITE,__sfrdef40_bits);
__IO_REG32_BIT(SYSAHBCLKDIV,        0x40048078,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(SYSAHBCLKCTRL,       0x40048080,__READ_WRITE,__sfrdef122_bits);
__IO_REG32_BIT(SSP0CLKDIV,          0x40048094,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(UARTCLKDIV,          0x40048098,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(SSP1CLKDIV,          0x4004809C,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(SYSTICKCLKDIV,       0x400480B0,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(CLKOUTSEL,           0x400480E0,__READ_WRITE,__sfrdef30_bits);
__IO_REG32_BIT(CLKOUTUEN,           0x400480E4,__READ_WRITE,__sfrdef40_bits);
__IO_REG32_BIT(CLKOUTDIV,           0x400480E8,__READ_WRITE,__sfrdef2_bits);
__IO_REG32_BIT(PIOPORCAP0,          0x40048100,__READ,__sfrdef112_bits);
__IO_REG32_BIT(PIOPORCAP1,          0x40048104,__READ,__sfrdef120_bits);
__IO_REG32_BIT(BODR,                0x40048150,__READ_WRITE,__sfrdef139_bits);
__IO_REG32_BIT(SYSTCKCAL,           0x40048158,__READ_WRITE,__sfrdef149_bits);
__IO_REG32_BIT(NMISRC,              0x40048174,__READ_WRITE,__sfrdef69_bits);
__IO_REG32_BIT(SCB_PINTSEL0,        0x40048178,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL1,        0x4004817C,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL2,        0x40048180,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL3,        0x40048184,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL4,        0x40048188,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL5,        0x4004818C,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL6,        0x40048190,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(SCB_PINTSEL7,        0x40048194,__READ_WRITE,__sfrdef82_bits);
__IO_REG32_BIT(DSWER0,              0x40048204,__READ_WRITE,__sfrdef95_bits);
__IO_REG32_BIT(DSWER1,              0x40048214,__READ_WRITE,__sfrdef128_bits);
__IO_REG32_BIT(PDSLEEPCFG,          0x40048230,__READ_WRITE,__sfrdef33_bits);
__IO_REG32_BIT(PDAWAKECFG,          0x40048234,__READ_WRITE,__sfrdef45_bits);
__IO_REG32_BIT(PDRUNCFG,            0x40048238,__READ_WRITE,__sfrdef45_bits);
__IO_REG32_BIT(DEVICE_ID,           0x400483F4,__READ,__sfrdef144_bits);

/***************************************************************************
**
** BPIN
**
***************************************************************************/
__IO_REG8(P0B0,                0x50000000,__READ_WRITE);
__IO_REG8(P0B1,                0x50000001,__READ_WRITE);
__IO_REG8(P0B2,                0x50000002,__READ_WRITE);
__IO_REG8(P0B3,                0x50000003,__READ_WRITE);
__IO_REG8(P0B4,                0x50000004,__READ_WRITE);
__IO_REG8(P0B5,                0x50000005,__READ_WRITE);
__IO_REG8(P0B6,                0x50000006,__READ_WRITE);
__IO_REG8(P0B7,                0x50000007,__READ_WRITE);
__IO_REG8(P0B8,                0x50000008,__READ_WRITE);
__IO_REG8(P0B9,                0x50000009,__READ_WRITE);
__IO_REG8(P0B10,               0x5000000A,__READ_WRITE);
__IO_REG8(P0B11,               0x5000000B,__READ_WRITE);
__IO_REG8(P0B12,               0x5000000C,__READ_WRITE);
__IO_REG8(P0B13,               0x5000000D,__READ_WRITE);
__IO_REG8(P0B14,               0x5000000E,__READ_WRITE);
__IO_REG8(P0B15,               0x5000000F,__READ_WRITE);
__IO_REG8(P0B16,               0x50000010,__READ_WRITE);
__IO_REG8(P0B17,               0x50000011,__READ_WRITE);
__IO_REG8(P0B18,               0x50000012,__READ_WRITE);
__IO_REG8(P0B19,               0x50000013,__READ_WRITE);
__IO_REG8(P0B20,               0x50000014,__READ_WRITE);
__IO_REG8(P0B21,               0x50000015,__READ_WRITE);
__IO_REG8(P0B22,               0x50000016,__READ_WRITE);
__IO_REG8(P0B23,               0x50000017,__READ_WRITE);
__IO_REG8(P0B24,               0x50000018,__READ_WRITE);
__IO_REG8(P0B25,               0x50000019,__READ_WRITE);
__IO_REG8(P0B26,               0x5000001A,__READ_WRITE);
__IO_REG8(P0B27,               0x5000001B,__READ_WRITE);
__IO_REG8(P0B28,               0x5000001C,__READ_WRITE);
__IO_REG8(P0B29,               0x5000001D,__READ_WRITE);
__IO_REG8(P0B30,               0x5000001E,__READ_WRITE);
__IO_REG8(P0B31,               0x5000001F,__READ_WRITE);
__IO_REG8(P1B0,                0x50000020,__READ_WRITE);
__IO_REG8(P1B1,                0x50000021,__READ_WRITE);
__IO_REG8(P1B2,                0x50000022,__READ_WRITE);
__IO_REG8(P1B3,                0x50000023,__READ_WRITE);
__IO_REG8(P1B4,                0x50000024,__READ_WRITE);
__IO_REG8(P1B5,                0x50000025,__READ_WRITE);
__IO_REG8(P1B6,                0x50000026,__READ_WRITE);
__IO_REG8(P1B7,                0x50000027,__READ_WRITE);
__IO_REG8(P1B8,                0x50000028,__READ_WRITE);
__IO_REG8(P1B9,                0x50000029,__READ_WRITE);

/***************************************************************************
**
** CT32B0
**
***************************************************************************/
__IO_REG32_BIT(TMR32B0IR,           0x40014000,__READ_WRITE,__sfrdef68_bits);
__IO_REG32_BIT(TMR32B0TCR,          0x40014004,__READ_WRITE,__sfrdef66_bits);
__IO_REG32(TMR32B0TC,           0x40014008,__READ_WRITE);
__IO_REG32_BIT(TMR32B0PR,           0x4001400C,__READ_WRITE,__sfrdef102_bits);
__IO_REG32(TMR32B0PC,           0x40014010,__READ_WRITE);
__IO_REG32_BIT(TMR32B0MCR,          0x40014014,__READ_WRITE,__sfrdef25_bits);
__IO_REG32_BIT(TMR32B0MR0,          0x40014018,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B0MR1,          0x4001401C,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B0MR2,          0x40014020,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B0MR3,          0x40014024,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B0CCR,          0x40014028,__READ_WRITE,__sfrdef81_bits);
__IO_REG32_BIT(TMR32B0CR0,          0x4001402C,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B0CR1,          0x40014030,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B0CR2,          0x40014034,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B0CR3,          0x40014038,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B0EMR,          0x4001403C,__READ_WRITE,__sfrdef37_bits);
__IO_REG32_BIT(TMR32B0CTCR,         0x40014070,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(TMR32B0PWMC,         0x40014074,__READ_WRITE,__sfrdef5_bits);

/***************************************************************************
**
** FLASHCFG
**
***************************************************************************/
__IO_REG32_BIT(FLASHCFG,            0x4003C010,__READ_WRITE,__sfrdef84_bits);

/***************************************************************************
**
** ACOMP
**
***************************************************************************/
__IO_REG32_BIT(ACOMP_CTL,           0x40028000,__READ_WRITE,__sfrdef94_bits);
__IO_REG32_BIT(ACOMP_LAD,           0x40028004,__READ_WRITE,__sfrdef104_bits);

/***************************************************************************
**
** CT32B1
**
***************************************************************************/
__IO_REG32_BIT(TMR32B1IR,           0x40018000,__READ_WRITE,__sfrdef68_bits);
__IO_REG32_BIT(TMR32B1TCR,          0x40018004,__READ_WRITE,__sfrdef66_bits);
__IO_REG32(TMR32B1TC,           0x40018008,__READ_WRITE);
__IO_REG32_BIT(TMR32B1PR,           0x4001800C,__READ_WRITE,__sfrdef102_bits);
__IO_REG32(TMR32B1PC,           0x40018010,__READ_WRITE);
__IO_REG32_BIT(TMR32B1MCR,          0x40018014,__READ_WRITE,__sfrdef25_bits);
__IO_REG32_BIT(TMR32B1MR0,          0x40018018,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B1MR1,          0x4001801C,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B1MR2,          0x40018020,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B1MR3,          0x40018024,__READ_WRITE,__sfrdef36_bits);
__IO_REG32_BIT(TMR32B1CCR,          0x40018028,__READ_WRITE,__sfrdef81_bits);
__IO_REG32_BIT(TMR32B1CR0,          0x4001802C,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B1CR1,          0x40018030,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B1CR2,          0x40018034,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B1CR3,          0x40018038,__READ,__sfrdef0_bits);
__IO_REG32_BIT(TMR32B1EMR,          0x4001803C,__READ_WRITE,__sfrdef37_bits);
__IO_REG32_BIT(TMR32B1CTCR,         0x40018070,__READ_WRITE,__sfrdef6_bits);
__IO_REG32_BIT(TMR32B1PWMC,         0x40018074,__READ_WRITE,__sfrdef5_bits);

/***************************************************************************
**
** IOCON
**
***************************************************************************/
__IO_REG32_BIT(IOCON_RESET_P0_0,    0x40044000,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_1,          0x40044004,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_2,          0x40044008,__READ_WRITE,__sfrdef12_bits);
__IO_REG32_BIT(IOCON_P0_3,          0x4004400C,__READ_WRITE,__sfrdef12_bits);
__IO_REG32_BIT(IOCON_P0_4,          0x40044010,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_TCK_P0_5,      0x40044014,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_TDI_P0_6,      0x40044018,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_TMS_P0_7,      0x4004401C,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_TDO_P0_8,      0x40044020,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_TRST_P0_9,     0x40044024,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_SWDIO_P0_10,   0x40044028,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_11,         0x4004402C,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_12,         0x40044030,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_13,         0x40044034,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_14,         0x40044038,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_15,         0x4004403C,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_16,         0x40044040,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_17,         0x40044044,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_18,         0x40044048,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_19,         0x4004404C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_20,         0x40044050,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_21,         0x40044054,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_22,         0x40044058,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_23,         0x4004405C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_24,         0x40044060,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_25,         0x40044064,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_26,         0x40044068,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_27,         0x4004406C,__READ_WRITE,__sfrdef72_bits);
__IO_REG32_BIT(IOCON_P0_28,         0x40044070,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_29,         0x40044074,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_30,         0x40044078,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P0_31,         0x4004407C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_0,          0x40044080,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_1,          0x40044084,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_2,          0x40044088,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_3,          0x4004408C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_4,          0x40044090,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_5,          0x40044094,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_6,          0x40044098,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_7,          0x4004409C,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_8,          0x400440A0,__READ_WRITE,__sfrdef15_bits);
__IO_REG32_BIT(IOCON_P1_9,          0x400440A4,__READ_WRITE,__sfrdef15_bits);
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
#define SVCI                  11  /* SVCall                                                 */
#define PSI                   14  /* PendSV                                                 */
#define STI                   15  /* SysTick                                                */
#define NVIC_GPIO0            16  /* GPIO pin interrupt 0                                   */
#define NVIC_GPIO1            17  /* GPIO pin interrupt 1                                   */
#define NVIC_GPIO2            18  /* GPIO pin interrupt 2                                   */
#define NVIC_GPIO3            19  /* GPIO pin interrupt 3                                   */
#define NVIC_GPIO4            20  /* GPIO pin interrupt 4                                   */                                    */
#define NVIC_GPIO5            21  /* GPIO pin interrupt 5                                   */
#define NVIC_GPIO6            22  /* GPIO pin interrupt 6                                   */
#define NVIC_GPIO7            23  /* GPIO pin interrupt 7                                   */
#define NVIC_GPIO_GROUP0      24  /* GPIO GROUP0 interrupt                                  */
#define NVIC_GPIO_GROUP1      25  /* GPIO GROUP1 interrupt                                  */
#define NVIC_ACOMP            26  /* Analog Comparator interrupt                            */
#define NVIC_DAC              27  /* Digital to Analog Converter interrupt                  */
#define NVIC_SSP1             30  /* SSP1                                                   */
#define NVIC_I2C              31  /* I2C                                                    */
#define NVIC_CT16B0           32  /* Counter Timer 0 16 bit                                 */
#define NVIC_CT16B1           33  /* Counter Timer 1 16 bit                                 */
#define NVIC_CT32B0           34  /* Counter Timer 0 32 bit                                 */
#define NVIC_CT32B1           35  /* Counter Timer 1 32 bit                                 */
#define NVIC_SSP0             36  /* SSP0                                                   */
#define NVIC_UART             37  /* UART                                                   */
#define NVIC_ADC              40  /* A/D Converter end of conversion                        */
#define NVIC_WDT              41  /* Windowed Watchdog interrupt (WDINT)                    */
#define NVIC_BOD              42  /* Brown-out detect                                       */
#define NVIC_FLASH            43  /* Flash Interface Interrupt                              */

#endif    /* __LPC11A1X_H */

