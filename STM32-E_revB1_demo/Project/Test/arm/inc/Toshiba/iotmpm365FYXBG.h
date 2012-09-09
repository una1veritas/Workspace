/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM365FYXBG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 52505 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM365FYXBG_H
#define __IOTMPM365FYXBG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM365FYXBG SPECIAL FUNCTION REGISTERS
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

/* System Control Register */
typedef struct {
  __REG32 GEAR    : 3;
  __REG32         : 5;
  __REG32 PRCK    : 3;
  __REG32         : 1;
  __REG32 FPSEL   : 1;
  __REG32         : 3;
  __REG32 SCOSEL  : 2;
  __REG32         : 2;
  __REG32 FCSTOP  : 1;
  __REG32         :11;
} __cgsyscr_bits;

/* Oscillation Control Register */
typedef struct {
  __REG32 WUEON     : 1;
  __REG32 WUEF      : 1;
  __REG32 PLLON     : 1;
  __REG32           : 5;
  __REG32 XEN1      : 1;
  __REG32           : 7;
  __REG32 XEN2      : 1;
  __REG32 OSCSEL    : 1;
  __REG32 EHOSCSEL  : 1;
  __REG32 HWUPSEL   : 1;
  __REG32 WUODR     :12;
} __cgosccr_bits;

/* Standby Control Register */
typedef struct {
  __REG32 STBY    : 3;
  __REG32         :13;
  __REG32 DRVE    : 1;
  __REG32         :15;
} __cgstbycr_bits;

/* PLL Selection Register */
typedef struct {
  __REG32 PLLSEL  : 1;
  __REG32 PLLSET  :15;
  __REG32         :16;
} __cgpllsel_bits;

/* CGUSBCTL (USB clock control register) */
typedef struct {
  __REG32           : 8;
  __REG32 USBCLKEN  : 1;
  __REG32 USBCLKSEL : 1;
  __REG32           :22;
} __cgusbctl_bits;

/* CGPROTECT (Protect register) */
typedef struct {
  __REG32 CGPROTECT : 8;
  __REG32           :24;
} __cgprotect_bits;

/* NMI Flag Register */
typedef struct {
  __REG32 NMIFLG0   : 1;
  __REG32 NMIFLG1   : 1;
  __REG32           :30;
} __cgnmiflg_bits;

/* Reset Flag Register */
typedef struct {
  __REG32 PINRSTF   : 1;
  __REG32           : 1;
  __REG32 WDTRSTF   : 1;
  __REG32 STOP2RSTF : 1;
  __REG32 DBGRSTF   : 1;
  __REG32           :27;
} __cgrstflg_bits;

/* CG interrupt Mode Control Register A */
typedef struct {
  __REG32 INT0EN    : 1;
  __REG32           : 1;
  __REG32 EMST0     : 2;
  __REG32 EMCG0     : 3;
  __REG32           : 1;
  __REG32 INT1EN    : 1;
  __REG32           : 1;
  __REG32 EMST1     : 2;
  __REG32 EMCG1     : 3;
  __REG32           : 1;
  __REG32 INT2EN    : 1;
  __REG32           : 1;
  __REG32 EMST2     : 2;
  __REG32 EMCG2     : 3;
  __REG32           : 1;
  __REG32 INT3EN    : 1;
  __REG32           : 1;
  __REG32 EMST3     : 2;
  __REG32 EMCG3     : 3;
  __REG32           : 1;
} __cgimcga_bits;

/* CG Interrupt Mode Control Register B */
typedef struct {
  __REG32 INT4EN    : 1;
  __REG32           : 1;
  __REG32 EMST4     : 2;
  __REG32 EMCG4     : 3;
  __REG32           : 1;
  __REG32 INT5EN    : 1;
  __REG32           : 1;
  __REG32 EMST5     : 2;
  __REG32 EMCG5     : 3;
  __REG32           : 1;
  __REG32 INT6EN    : 1;
  __REG32           : 1;
  __REG32 EMST6     : 2;
  __REG32 EMCG6     : 3;
  __REG32           : 1;
  __REG32 INT7EN    : 1;
  __REG32           : 1;
  __REG32 EMST7     : 2;
  __REG32 EMCG7     : 3;
  __REG32           : 1;
} __cgimcgb_bits;

/* CG Interrupt Mode Control Register C */
typedef struct {
  __REG32 INT8EN    : 1;
  __REG32           : 1;
  __REG32 EMST8     : 2;
  __REG32 EMCG8     : 3;
  __REG32           : 1;
  __REG32 INT9EN    : 1;
  __REG32           : 1;
  __REG32 EMST9     : 2;
  __REG32 EMCG9     : 3;
  __REG32           : 1;
  __REG32 INTAEN    : 1;
  __REG32           : 1;
  __REG32 EMSTA     : 2;
  __REG32 EMCGA     : 3;
  __REG32           : 1;
  __REG32 INTBEN    : 1;
  __REG32           : 1;
  __REG32 EMSTB     : 2;
  __REG32 EMCGB     : 3;
  __REG32           : 1;
} __cgimcgc_bits;

/* CG Interrupt Request Clear Register */
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           :27;
} __cgicrcg_bits;

/* PORT A Register */
typedef struct {
  __REG8  PA0  : 1;
  __REG8  PA1  : 1;
  __REG8  PA2  : 1;
  __REG8  PA3  : 1;
  __REG8  PA4  : 1;
  __REG8  PA5  : 1;
  __REG8  PA6  : 1;
  __REG8  PA7  : 1;
} __pa_bits;

/* PORT A Control Register */
typedef struct {
  __REG8  PA0C  : 1;
  __REG8  PA1C  : 1;
  __REG8  PA2C  : 1;
  __REG8  PA3C  : 1;
  __REG8  PA4C  : 1;
  __REG8  PA5C  : 1;
  __REG8  PA6C  : 1;
  __REG8  PA7C  : 1;
} __pacr_bits;

/* PortA open drain control register */
typedef struct {
  __REG8  PA0OD  : 1;
  __REG8  PA1OD  : 1;
  __REG8  PA2OD  : 1;
  __REG8  PA3OD  : 1;
  __REG8  PA4OD  : 1;
  __REG8  PA5OD  : 1;
  __REG8  PA6OD  : 1;
  __REG8  PA7OD  : 1;
} __paod_bits;

/* PORT A Pull-Up Control Register */
typedef struct {
  __REG8  PA0UP  : 1;
  __REG8  PA1UP  : 1;
  __REG8  PA2UP  : 1;
  __REG8  PA3UP  : 1;
  __REG8  PA4UP  : 1;
  __REG8  PA5UP  : 1;
  __REG8  PA6UP  : 1;
  __REG8  PA7UP  : 1;
} __papup_bits;

/* PORT A Input Enable Control Register */
typedef struct {
  __REG8  PA0IE  : 1;
  __REG8  PA1IE  : 1;
  __REG8  PA2IE  : 1;
  __REG8  PA3IE  : 1;
  __REG8  PA4IE  : 1;
  __REG8  PA5IE  : 1;
  __REG8  PA6IE  : 1;
  __REG8  PA7IE  : 1;
} __paie_bits;

/*PORT B Register*/
typedef struct {
  __REG8  PB0  : 1;
  __REG8  PB1  : 1;
  __REG8  PB2  : 1;
  __REG8  PB3  : 1;
  __REG8  PB4  : 1;
  __REG8  PB5  : 1;
  __REG8  PB6  : 1;
  __REG8  PB7  : 1;
} __pb_bits;

/* PORT B Control Register */
typedef struct {
  __REG8  PB0C  : 1;
  __REG8  PB1C  : 1;
  __REG8  PB2C  : 1;
  __REG8  PB3C  : 1;
  __REG8  PB4C  : 1;
  __REG8  PB5C  : 1;
  __REG8  PB6C  : 1;
  __REG8  PB7C  : 1;
} __pbcr_bits;

/* PORT B open drain control register */
typedef struct {
  __REG8  PB0OD  : 1;
  __REG8  PB1OD  : 1;
  __REG8  PB2OD  : 1;
  __REG8  PB3OD  : 1;
  __REG8  PB4OD  : 1;
  __REG8  PB5OD  : 1;
  __REG8  PB6OD  : 1;
  __REG8  PB7OD  : 1;
} __pbod_bits;

/* PORT B Pull-Up Control Register */
typedef struct {
  __REG8  PB0UP  : 1;
  __REG8  PB1UP  : 1;
  __REG8  PB2UP  : 1;
  __REG8  PB3UP  : 1;
  __REG8  PB4UP  : 1;
  __REG8  PB5UP  : 1;
  __REG8  PB6UP  : 1;
  __REG8  PB7UP  : 1;
} __pbpup_bits;

/* PORT B Input Enable Control Register */
typedef struct {
  __REG8  PB0IE  : 1;
  __REG8  PB1IE  : 1;
  __REG8  PB2IE  : 1;
  __REG8  PB3IE  : 1;
  __REG8  PB4IE  : 1;
  __REG8  PB5IE  : 1;
  __REG8  PB6IE  : 1;
  __REG8  PB7IE  : 1;
} __pbie_bits;

/* PORT C Register*/
typedef struct {
  __REG8  PC0  : 1;
  __REG8  PC1  : 1;
  __REG8  PC2  : 1;
  __REG8       : 5;
} __pc_bits;

/* PORT C Control Register */
typedef struct {
  __REG8  PC0C  : 1;
  __REG8  PC1C  : 1;
  __REG8  PC2C  : 1;
  __REG8        : 5;
} __pccr_bits;

/* PORT C Function Register 1 */
typedef struct {
  __REG8  PC0F1  : 1;
  __REG8  PC1F1  : 1;
  __REG8  PC2F1  : 1;
  __REG8         : 5;
} __pcfr1_bits;

/* PORT C Function Register 3 */
typedef struct {
  __REG8  PC0F3  : 1;
  __REG8  PC1F3  : 1;
  __REG8  PC2F3  : 1;
  __REG8         : 5;
} __pcfr3_bits;

/* PORT C Function Register 4 */
typedef struct {
  __REG8         : 2;
  __REG8  PC2F4  : 1;
  __REG8         : 5;
} __pcfr4_bits;

/* PortC open drain control register */
typedef struct {
  __REG8  PC0OD  : 1;
  __REG8  PC1OD  : 1;
  __REG8  PC2OD  : 1;
  __REG8         : 5;
} __pcod_bits;

/* PORT C Pull-Up Control Register */
typedef struct {
  __REG8  PC0UP  : 1;
  __REG8  PC1UP  : 1;
  __REG8  PC2UP  : 1;
  __REG8         : 5;
} __pcpup_bits;

/*PORT C Input Enable Control Register */
typedef struct {
  __REG8  PC0IE  : 1;
  __REG8  PC1IE  : 1;
  __REG8  PC2IE  : 1;
  __REG8         : 5;
} __pcie_bits;

/* PORT D Register */
typedef struct {
  __REG8  PD0  : 1;
  __REG8  PD1  : 1;
  __REG8  PD2  : 1;
  __REG8  PD3  : 1;
  __REG8  PD4  : 1;
  __REG8  PD5  : 1;
  __REG8  PD6  : 1;
  __REG8  PD7  : 1;
} __pd_bits;

/* PORT D Control Register */
typedef struct {
  __REG8  PD0C  : 1;
  __REG8  PD1C  : 1;
  __REG8  PD2C  : 1;
  __REG8  PD3C  : 1;
  __REG8  PD4C  : 1;
  __REG8  PD5C  : 1;
  __REG8  PD6C  : 1;
  __REG8  PD7C  : 1;
} __pdcr_bits;

/* PORT D Function Register 3 */
typedef struct {
  __REG8  PD0F3  : 1;
  __REG8  PD1F3  : 1;
  __REG8  PD2F3  : 1;
  __REG8  PD3F3  : 1;
  __REG8         : 3;
  __REG8  PD7F3  : 1;
} __pdfr3_bits;

/* PortD open drain control register */
typedef struct {
  __REG8  PD0OD  : 1;
  __REG8  PD1OD  : 1;
  __REG8  PD2OD  : 1;
  __REG8  PD3OD  : 1;
  __REG8  PD4OD  : 1;
  __REG8  PD5OD  : 1;
  __REG8  PD6OD  : 1;
  __REG8  PD7OD  : 1;
} __pdod_bits;

/*PORT D Pull-Up Control Register */
typedef struct {
  __REG8  PD0UP  : 1;
  __REG8  PD1UP  : 1;
  __REG8  PD2UP  : 1;
  __REG8  PD3UP  : 1;
  __REG8  PD4UP  : 1;
  __REG8  PD5UP  : 1;
  __REG8  PD6UP  : 1;
  __REG8  PD7UP  : 1;
} __pdpup_bits;

/*PORT D Input Enable Control Register */
typedef struct {
  __REG8  PD0IE  : 1;
  __REG8  PD1IE  : 1;
  __REG8  PD2IE  : 1;
  __REG8  PD3IE  : 1;
  __REG8  PD4IE  : 1;
  __REG8  PD5IE  : 1;
  __REG8  PD6IE  : 1;
  __REG8  PD7IE  : 1;
} __pdie_bits;

/* PORT E Register */
typedef struct {
  __REG8  PE0  : 1;
  __REG8  PE1  : 1;
  __REG8  PE2  : 1;
  __REG8  PE3  : 1;
  __REG8  PE4  : 1;
  __REG8  PE5  : 1;
  __REG8  PE6  : 1;
  __REG8  PE7  : 1;
} __pe_bits;

/* PORT E Control Register */
typedef struct {
  __REG8  PE0C  : 1;
  __REG8  PE1C  : 1;
  __REG8  PE2C  : 1;
  __REG8  PE3C  : 1;
  __REG8  PE4C  : 1;
  __REG8  PE5C  : 1;
  __REG8  PE6C  : 1;
  __REG8  PE7C  : 1;
} __pecr_bits;

/* PORT E Function Register 1 */
typedef struct {
  __REG8  PE0F1  : 1;
  __REG8  PE1F1  : 1;
  __REG8  PE2F1  : 1;
  __REG8  PE3F1  : 1;
  __REG8  PE4F1  : 1;
  __REG8  PE5F1  : 1;
  __REG8  PE6F1  : 1;
  __REG8  PE7F1  : 1;
} __pefr1_bits;

/* PORT E Function Register 3 */
typedef struct {
  __REG8         : 2;
  __REG8  PE2F3  : 1;
  __REG8  PE3F3  : 1;
  __REG8         : 4;
} __pefr3_bits;

/* PORT E Function Register 4 */
typedef struct {
  __REG8         : 2;
  __REG8  PE2F4  : 1;
  __REG8         : 5;
} __pefr4_bits;

/* PORT E Open Drain Control Register */
typedef struct {
  __REG8  PE0OD  : 1;
  __REG8  PE1OD  : 1;
  __REG8  PE2OD  : 1;
  __REG8  PE3OD  : 1;
  __REG8  PE4OD  : 1;
  __REG8  PE5OD  : 1;
  __REG8  PE6OD  : 1;
  __REG8  PE7OD  : 1;
} __peod_bits;

/* PORT E Pull-Up Control Register */
typedef struct {
  __REG8  PE0UP  : 1;
  __REG8  PE1UP  : 1;
  __REG8  PE2UP  : 1;
  __REG8  PE3UP  : 1;
  __REG8  PE4UP  : 1;
  __REG8  PE5UP  : 1;
  __REG8  PE6UP  : 1;
  __REG8  PE7UP  : 1;
} __pepup_bits;

/* PORT E Input Enable Control Register */
typedef struct {
  __REG8  PE0IE  : 1;
  __REG8  PE1IE  : 1;
  __REG8  PE2IE  : 1;
  __REG8  PE3IE  : 1;
  __REG8  PE4IE  : 1;
  __REG8  PE5IE  : 1;
  __REG8  PE6IE  : 1;
  __REG8  PE7IE  : 1;
} __peie_bits;

/* PORT F Register */
typedef struct {
  __REG8  PF0  : 1;
  __REG8  PF1  : 1;
  __REG8  PF2  : 1;
  __REG8  PF3  : 1;
  __REG8  PF4  : 1;
  __REG8  PF5  : 1;
  __REG8  PF6  : 1;
  __REG8  PF7  : 1;
} __pf_bits;

/* PORT F Control Register */
typedef struct {
  __REG8  PF0C  : 1;
  __REG8  PF1C  : 1;
  __REG8  PF2C  : 1;
  __REG8  PF3C  : 1;
  __REG8  PF4C  : 1;
  __REG8  PF5C  : 1;
  __REG8  PF6C  : 1;
  __REG8  PF7C  : 1;
} __pfcr_bits;

/* PORT F Function Register 2 */
typedef struct {
  __REG8         : 4;
  __REG8  PF4F2  : 1;
  __REG8  PF5F2  : 1;
  __REG8         : 2;
} __pffr2_bits;

/* PORT F Function Register 3 */
typedef struct {
  __REG8  PF0F3  : 1;
  __REG8         : 3;
  __REG8  PF4F3  : 1;
  __REG8  PF5F3  : 1;
  __REG8         : 2;
} __pffr3_bits;

/* PORT F Open Drain Control Register */
typedef struct {
  __REG8  PF0OD  : 1;
  __REG8  PF1OD  : 1;
  __REG8  PF2OD  : 1;
  __REG8  PF3OD  : 1;
  __REG8  PF4OD  : 1;
  __REG8  PF5OD  : 1;
  __REG8  PF6OD  : 1;
  __REG8  PF7OD  : 1;
} __pfod_bits;

/* PORT F Pull-Up Control Register */
typedef struct {
  __REG8  PF0UP  : 1;
  __REG8  PF1UP  : 1;
  __REG8  PF2UP  : 1;
  __REG8  PF3UP  : 1;
  __REG8  PF4UP  : 1;
  __REG8  PF5UP  : 1;
  __REG8  PF6UP  : 1;
  __REG8  PF7UP  : 1;
} __pfpup_bits;

/* PORT F Input Enable Control Register */
typedef struct {
  __REG8         : 1;
  __REG8  PF1IE  : 1;
  __REG8  PF2IE  : 1;
  __REG8  PF3IE  : 1;
  __REG8  PF4IE  : 1;
  __REG8  PF5IE  : 1;
  __REG8  PF6IE  : 1;
  __REG8  PF7IE  : 1;
} __pfie_bits;

/* PORT G Register */
typedef struct {
  __REG8  PG0  : 1;
  __REG8  PG1  : 1;
  __REG8  PG2  : 1;
  __REG8  PG3  : 1;
  __REG8  PG4  : 1;
  __REG8  PG5  : 1;
  __REG8       : 2;
} __pg_bits;

/* PortG control register */
typedef struct {
  __REG8  PG0C  : 1;
  __REG8  PG1C  : 1;
  __REG8  PG2C  : 1;
  __REG8  PG3C  : 1;
  __REG8  PG4C  : 1;
  __REG8  PG5C  : 1;
  __REG8        : 2;
} __pgcr_bits;

/* PORT G Function Register 1 */
typedef struct {
  __REG8  PG0F1  : 1;
  __REG8  PG1F1  : 1;
  __REG8  PG2F1  : 1;
  __REG8  PG3F1  : 1;
  __REG8         : 1;
  __REG8  PG5F1  : 1;
  __REG8         : 2;
} __pgfr1_bits;

/* PORT G Function Register 3 */
typedef struct {
  __REG8         : 1;
  __REG8  PG1F3  : 1;
  __REG8  PG2F3  : 1;
  __REG8  PG3F3  : 1;
  __REG8  PG4F3  : 1;
  __REG8         : 3;
} __pgfr3_bits;

/* PORT G Function Register 4 */
typedef struct {
  __REG8         : 5;
  __REG8  PG5F4  : 1;
  __REG8         : 2;
} __pgfr4_bits;

/* PORT G Open Drain Control Register */
typedef struct {
  __REG8  PG0OD  : 1;
  __REG8  PG1OD  : 1;
  __REG8  PG2OD  : 1;
  __REG8  PG3OD  : 1;
  __REG8  PG4OD  : 1;
  __REG8  PG5OD  : 1;
  __REG8         : 2;
} __pgod_bits;

/* PORT G Pull-Up Control Register */
typedef struct {
  __REG8  PG0UP  : 1;
  __REG8  PG1UP  : 1;
  __REG8  PG2UP  : 1;
  __REG8  PG3UP  : 1;
  __REG8  PG4UP  : 1;
  __REG8  PG5UP  : 1;
  __REG8         : 2;
} __pgpup_bits;

/* PORT G Input Enable Control Register */
typedef struct {
  __REG8  PG0IE  : 1;
  __REG8  PG1IE  : 1;
  __REG8  PG2IE  : 1;
  __REG8  PG3IE  : 1;
  __REG8  PG4IE  : 1;
  __REG8  PG5IE  : 1;
  __REG8         : 2;
} __pgie_bits;

/* PORT H Register */
typedef struct {
  __REG8  PH0  : 1;
  __REG8  PH1  : 1;
  __REG8  PH2  : 1;
  __REG8  PH3  : 1;
  __REG8  PH4  : 1;
  __REG8       : 3;
} __ph_bits;

/* PortH control register */
typedef struct {
  __REG8  PH0C  : 1;
  __REG8  PH1C  : 1;
  __REG8  PH2C  : 1;
  __REG8  PH3C  : 1;
  __REG8  PH4C  : 1;
  __REG8        : 3;
} __phcr_bits;

/* PORT H Function Register 1 */
typedef struct {
  __REG8  PH0F1  : 1;
  __REG8  PH1F1  : 1;
  __REG8         : 6;
} __phfr1_bits;

/* PORT H Function Register 3 */
typedef struct {
  __REG8         : 2;
  __REG8  PH2F3  : 1;
  __REG8  PH3F3  : 1;
  __REG8  PH4F3  : 1;
  __REG8         : 3;
} __phfr3_bits;

/* PortH open drain control register */
typedef struct {
  __REG8  PH0OD  : 1;
  __REG8  PH1OD  : 1;
  __REG8  PH2OD  : 1;
  __REG8  PH3OD  : 1;
  __REG8  PH4OD  : 1;
  __REG8         : 3;
} __phod_bits;

/* PORT H Pull-Up Control Register */
typedef struct {
  __REG8  PH0UP  : 1;
  __REG8  PH1UP  : 1;
  __REG8  PH2UP  : 1;
  __REG8  PH3UP  : 1;
  __REG8  PH4UP  : 1;
  __REG8         : 3;
} __phpup_bits;

/* PORT H Input Enable Control Register */
typedef struct {
  __REG8  PH0IE  : 1;
  __REG8  PH1IE  : 1;
  __REG8  PH2IE  : 1;
  __REG8  PH3IE  : 1;
  __REG8  PH4IE  : 1;
  __REG8         : 3;
} __phie_bits;

/* PORT I Register */
typedef struct {
  __REG8  PI0  : 1;
  __REG8  PI1  : 1;
  __REG8  PI2  : 1;
  __REG8  PI3  : 1;
  __REG8  PI4  : 1;
  __REG8  PI5  : 1;
  __REG8  PI6  : 1;
  __REG8  PI7  : 1;
} __pi_bits;

/* PORT I Control Register */
typedef struct {
  __REG8  PI0C  : 1;
  __REG8  PI1C  : 1;
  __REG8  PI2C  : 1;
  __REG8  PI3C  : 1;
  __REG8  PI4C  : 1;
  __REG8  PI5C  : 1;
  __REG8  PI6C  : 1;
  __REG8  PI7C  : 1;
} __picr_bits;

/* PORT I Function Register 1 */
typedef struct {
  __REG8  PI0F1  : 1;
  __REG8  PI1F1  : 1;
  __REG8  PI2F1  : 1;
  __REG8  PI3F1  : 1;
  __REG8  PI4F1  : 1;
  __REG8  PI5F1  : 1;
  __REG8  PI6F1  : 1;
  __REG8  PI7F1  : 1;
} __pifr1_bits;

/* PortI open drain control register */
typedef struct {
  __REG8  PI0OD  : 1;
  __REG8  PI1OD  : 1;
  __REG8  PI2OD  : 1;
  __REG8         : 5;
} __piod_bits;

/*PORT I Pull-Up Control Register */
typedef struct {
  __REG8  PI0UP  : 1;
  __REG8  PI1UP  : 1;
  __REG8  PI2UP  : 1;
  __REG8         : 1;
  __REG8  PI4UP  : 1;
  __REG8  PI5UP  : 1;
  __REG8  PI6UP  : 1;
  __REG8  PI7UP  : 1;
} __pipup_bits;

/*PORT I Pull-Down Control Register */
typedef struct {
  __REG8         : 3;
  __REG8  PI3DN  : 1;
  __REG8         : 4;
} __pipdn_bits;

/*PORT I Input Enable Control Register */
typedef struct {
  __REG8  PI0IE  : 1;
  __REG8  PI1IE  : 1;
  __REG8  PI2IE  : 1;
  __REG8  PI3IE  : 1;
  __REG8  PI4IE  : 1;
  __REG8  PI5IE  : 1;
  __REG8  PI6IE  : 1;
  __REG8  PI7IE  : 1;
} __piie_bits;

/* PORT J Register */
typedef struct {
  __REG8  PJ0  : 1;
  __REG8  PJ1  : 1;
  __REG8  PJ2  : 1;
  __REG8  PJ3  : 1;
  __REG8  PJ4  : 1;
  __REG8  PJ5  : 1;
  __REG8  PJ6  : 1;
  __REG8  PJ7  : 1;
} __pj_bits;

/* Port J output control register */
typedef struct {
  __REG8  PJ0C  : 1;
  __REG8  PJ1C  : 1;
  __REG8  PJ2C  : 1;
  __REG8  PJ3C  : 1;
  __REG8  PJ4C  : 1;
  __REG8  PJ5C  : 1;
  __REG8  PJ6C  : 1;
  __REG8  PJ7C  : 1;
} __pjcr_bits;

/* PORT J Function Register 2 */
typedef struct {
  __REG8         : 7;
  __REG8  PJ7F2  : 1;
} __pjfr2_bits;

/* PORT J Function Register 3 */
typedef struct {
  __REG8         : 6;
  __REG8  PJ6F3  : 1;
  __REG8  PJ7F3  : 1;
} __pjfr3_bits;

/* PORT J Pull-Up Control Register */
typedef struct {
  __REG8  PJ0UP  : 1;
  __REG8  PJ1UP  : 1;
  __REG8  PJ2UP  : 1;
  __REG8  PJ3UP  : 1;
  __REG8  PJ4UP  : 1;
  __REG8  PJ5UP  : 1;
  __REG8  PJ6UP  : 1;
  __REG8  PJ7UP  : 1;
} __pjpup_bits;

/* PORT J Input Enable Control Register */
typedef struct {
  __REG8  PJ0IE  : 1;
  __REG8  PJ1IE  : 1;
  __REG8  PJ2IE  : 1;
  __REG8  PJ3IE  : 1;
  __REG8  PJ4IE  : 1;
  __REG8  PJ5IE  : 1;
  __REG8  PJ6IE  : 1;
  __REG8  PJ7IE  : 1;
} __pjie_bits;

/* PORT K Register */
typedef struct {
  __REG8  PK0  : 1;
  __REG8  PK1  : 1;
  __REG8  PK2  : 1;
  __REG8  PK3  : 1;
  __REG8       : 4;
} __pk_bits;

/* Port K output control register */
typedef struct {
  __REG8  PK0C  : 1;
  __REG8  PK1C  : 1;
  __REG8  PK2C  : 1;
  __REG8  PK3C  : 1;
  __REG8        : 4;
} __pkcr_bits;

/* PORT K Function Register 2 */
typedef struct {
  __REG8  PK0F2  : 1;
  __REG8  PK1F2  : 1;
  __REG8         : 6;
} __pkfr2_bits;

/* PORT K Function Register 3 */
typedef struct {
  __REG8  PK0F3  : 1;
  __REG8  PK1F3  : 1;
  __REG8  PK2F3  : 1;
  __REG8  PK3F3  : 1;
  __REG8         : 4;
} __pkfr3_bits;

/* PORT K Pull-Up Control Register */
typedef struct {
  __REG8  PK0UP  : 1;
  __REG8  PK1UP  : 1;
  __REG8  PK2UP  : 1;
  __REG8  PK3UP  : 1;
  __REG8         : 4;
} __pkpup_bits;

/* PORT K Input Enable Control Register */
typedef struct {
  __REG8  PK0IE  : 1;
  __REG8  PK1IE  : 1;
  __REG8  PK2IE  : 1;
  __REG8  PK3IE  : 1;
  __REG8         : 4;
} __pkie_bits;

/* DMACIntStatus (DMAC Interrupt Status Register) */
typedef struct{
__REG32 IntStatus0            : 1;
__REG32 IntStatus1            : 1;
__REG32                       :30;
} __dmacintstaus_bits;

/* DMACIntTCStatus (DMAC Interrupt Terminal Count Status Register) */
typedef struct{
__REG32 IntTCStatus0          : 1;
__REG32 IntTCStatus1          : 1;
__REG32                       :30;
} __dmacinttcstatus_bits;

/* DMACIntTCClear (DMAC Interrupt Terminal Count Clear Register) */
typedef struct{
__REG32 IntTCClear0           : 1;
__REG32 IntTCClear1           : 1;
__REG32                       :30;
} __dmacinttcclear_bits;

/* DMACIntErrorStatus (DMAC Interrupt Error Status Register) */
typedef struct{
__REG32 IntErrStatus0         : 1;
__REG32 IntErrStatus1         : 1;
__REG32                       :30;
} __dmacinterrorstatus_bits;

/* DMACIntErrClr (DMAC Interrupt Error Clear Register) */
typedef struct{
__REG32 IntErrClr0            : 1;
__REG32 IntErrClr1            : 1;
__REG32                       :30;
} __dmacinterrclr_bits;

/* DMACRawIntTCStatus (DMAC Raw Interrupt Terminal Count Status Register) */
typedef struct{
__REG32 RawIntTCS0            : 1;
__REG32 RawIntTCS1            : 1;
__REG32                       :30;
} __dmacrawinttcstatus_bits;

/* DMACRawIntErrorStatus (DMAC Raw Error Interrupt Status Register) */
typedef struct{
__REG32 RawIntErrS0           : 1;
__REG32 RawIntErrS1           : 1;
__REG32                       :30;
} __dmacrawinterrorstatus_bits;

/* DMACEnbldChns (DMAC Enabled Channel Register) */
typedef struct{
__REG32 EnabledCH0            : 1;
__REG32 EnabledCH1            : 1;
__REG32                       :30;
} __dmacenbldchns_bits;

/* DMACSoftBReq (DMAC Software Burst Request Register) */
typedef struct{
__REG32 SoftBReq0             : 1;
__REG32 SoftBReq1             : 1;
__REG32 SoftBReq2             : 1;
__REG32 SoftBReq3             : 1;
__REG32 SoftBReq4             : 1;
__REG32 SoftBReq5             : 1;
__REG32 SoftBReq6             : 1;
__REG32 SoftBReq7             : 1;
__REG32 SoftBReq8             : 1;
__REG32 SoftBReq9             : 1;
__REG32 SoftBReq10            : 1;
__REG32 SoftBReq11            : 1;
__REG32 SoftBReq12            : 1;
__REG32 SoftBReq13            : 1;
__REG32 SoftBReq14            : 1;
__REG32 SoftBReq15            : 1;
__REG32                       :16;
} __dmacsoftbreq_bits;

/* DMACConfiguration (DMAC Configuration Register) */
typedef struct{
__REG32 E                     : 1;
__REG32 M                     : 1;
__REG32                       :30;
} __dmacconfiguration_bits;

/* DMACCxCTL (DMAC Channel x Control Register) */
typedef struct{
__REG32 TransferSize          :12;
__REG32 SBSize                : 3;
__REG32 DBSize                : 3;
__REG32 Swidth                : 3;
__REG32 Dwidth                : 3;
__REG32                       : 2;
__REG32 SI                    : 1;
__REG32 DI                    : 1;
__REG32                       : 3;
__REG32 I                     : 1;
} __dmacccontrol_bits;

/* DMACCxCFG (DMAC Channel x Configuration Register) */
typedef struct{
__REG32 E                     : 1;
__REG32 SrcPeripheral         : 4;
__REG32                       : 1;
__REG32 DestPeripheral        : 4;
__REG32                       : 1;
__REG32 FlowCntrl             : 3;
__REG32 IE                    : 1;
__REG32 ITC                   : 1;
__REG32 Lock                  : 1;
__REG32 Active                : 1;
__REG32 Halt                  : 1;
__REG32                       :13;
} __dmaccconfiguration_bits;

/*TMRBn enable register (channels 0 through 9)*/
typedef struct {
  __REG32           : 6;
  __REG32  TBHALT   : 1;
  __REG32  TBEN     : 1;
  __REG32           :24;
} __tbxen_bits;

/*TMRB RUN register (channels 0 through 9)*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32           : 1;
  __REG32  TBPRUN   : 1;
  __REG32           :29;
} __tbxrun_bits;

/*TMRB control register (channels 0 through 9)*/
typedef struct {
  __REG32  CSSEL    : 1;
  __REG32  TRGSEL   : 1;
  __REG32  TBINSEL  : 1;
  __REG32  I2TB     : 1;
  __REG32  FT0SEL   : 1;
  __REG32  TBSYNC   : 1;
  __REG32           : 1;
  __REG32  TBWBF    : 1;
  __REG32           :24;
} __tbxcr_bits;

/*TMRB mode register (channels 0 thorough 9)*/
typedef struct {
  __REG32  TBCLK    : 3;
  __REG32  TBCLE    : 1;
  __REG32  TBCPM    : 2;
  __REG32  TBCP     : 1;
  __REG32           :25;
} __tbxmod_bits;

/*TMRB flip-flop control register (channels 0 through 9)*/
typedef struct {
  __REG32  TBFF0C   : 2;
  __REG32  TBE0T1   : 1;
  __REG32  TBE1T1   : 1;
  __REG32  TBC0T1   : 1;
  __REG32  TBC1T1   : 1;
  __REG32           :26;
} __tbxffcr_bits;

/*TMRB status register (channels 0 through 9)*/
typedef struct {
  __REG32  INTTB0   : 1;
  __REG32  INTTB1   : 1;
  __REG32  INTTBOF  : 1;
  __REG32           :29;
} __tbxst_bits;

/*TMRB interrupt mask register (channels 0 through 9)*/
typedef struct {
  __REG32  TBIM0    : 1;
  __REG32  TBIM1    : 1;
  __REG32  TBIMOF   : 1;
  __REG32           :29;
} __tbxim_bits;

/*TMRB read capture register (channels 0 through 9)*/
typedef struct {
  __REG32  TBUC     :16;
  __REG32           :16;
} __tbxuc_bits;

/*TMRB timer register 0 (channels 0 through 9)*/
typedef struct {
  __REG32  TBRG0    :16;
  __REG32           :16;
} __tbxrg0_bits;

/*TMRB timer register 1 (channels 0 through 9)*/
typedef struct {
  __REG32  TBRG1    :16;
  __REG32           :16;
} __tbxrg1_bits;

/*TMRB capture register 0 (channels 0 through 9)*/
typedef struct {
  __REG32  TBCP0    :16;
  __REG32           :16;
} __tbxcp0_bits;

/*TMRB capture register 1 (channels 0 through 9)*/
typedef struct {
  __REG32  TBCP1    :16;
  __REG32           :16;
} __tbxcp1_bits;

/*TMRB DMA enable register (channels 0 through 9)*/
typedef struct {
  __REG32  TBDMAEN0 : 1;
  __REG32  TBDMAEN1 : 1;
  __REG32  TBDMAEN2 : 1;
  __REG32           :29;
} __tbxdma_bits;

/* UDINTSTS (Interrupt Status register) */
typedef struct
{
  __REG32 int_setup             : 1;
  __REG32 int_status_nak        : 1;
  __REG32 int_status            : 1;
  __REG32 int_rx_zero           : 1;
  __REG32 int_sof               : 1;
  __REG32 int_ep0               : 1;
  __REG32 int_ep                : 1;
  __REG32 int_nak               : 1;
  __REG32 int_suspend_resume    : 1;
  __REG32 int_usb_reset         : 1;
  __REG32 int_usb_reset_end     : 1;
  __REG32                       : 6;
  __REG32 int_mw_set_add        : 1;
  __REG32 int_mw_end_add        : 1;
  __REG32 int_mw_timeout        : 1;
  __REG32 int_mw_ahberr         : 1;
  __REG32 int_mr_end_add        : 1;
  __REG32 int_mr_ep_dset        : 1;
  __REG32 int_mr_ahberr         : 1;
  __REG32 int_udc2_reg_rd       : 1;
  __REG32 int_dmac_reg_rd       : 1;
  __REG32                       : 2;
  __REG32 int_mw_powerdetect    : 1;
  __REG32 int_mw_rerror         : 1;
  __REG32                       : 2;
} __udintsts_bits;

/* UDINTENB (Interrupt Enable register) */
typedef struct
{
  __REG32                       : 8;
  __REG32 suspend_resume_en     : 1;
  __REG32 usb_reset_en          : 1;
  __REG32 usb_reset_end_en      : 1;
  __REG32                       : 6;
  __REG32 mw_set_add_en         : 1;
  __REG32 mw_end_add_en         : 1;
  __REG32 mw_timeout_en         : 1;
  __REG32 mw_ahberr_en          : 1;
  __REG32 mr_end_add_en         : 1;
  __REG32 mr_ep_dset_en         : 1;
  __REG32 mr_ahberr_en          : 1;
  __REG32 udc2_reg_rd_en        : 1;
  __REG32 dmac_reg_rd_en        : 1;
  __REG32                       : 3;
  __REG32 mw_rerror_en          : 1;
  __REG32                       : 2;
} __udintenb_bits;

/* UDMWTOUT (Master Write Timeout register) */
typedef struct
{
  __REG32 timeout_en            : 1;
  __REG32 timeoutset            :31;
} __udmwtout_bits;

/* UDC2STSET (UDC2 Setting register) */
typedef struct
{
  __REG32 tx0                   : 1;
  __REG32                       : 3;
  __REG32 eopb_enable           : 1;
  __REG32                       :27;
} __udc2stset_bits;

/* UDMSTSET (DMAC Setting register) */
typedef struct
{
  __REG32 mw_enable             : 1;
  __REG32 mw_abort              : 1;
  __REG32 mw_reset              : 1;
  __REG32                       : 1;
  __REG32 mr_enable             : 1;
  __REG32 mr_abort              : 1;
  __REG32 mr_reset              : 1;
  __REG32                       : 1;
  __REG32 m_burst_type          : 1;
  __REG32                       :23;
} __udmstset_bits;

/* UDDMACRDREQ (DMAC Read Requset register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 dmardadr              : 6;
  __REG32                       :22;
  __REG32 dmardclr              : 1;
  __REG32 dmardreq              : 1;
} __uddmacrdreq_bits;

/* UDC2RDREQ (UDC2 Read Request register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 udc2rdadr             : 8;
  __REG32                       :20;
  __REG32 udc2rdclr             : 1;
  __REG32 udc2rdreq             : 1;
} __udc2rdreq_bits;

/* UDC2RDVL (UDC2 Read Value register) */
typedef struct
{
  __REG32 udc2rdata             :16;
  __REG32                       :16;
} __udc2rdvl_bits;

/* ARBTSET (Arbiter Setting register) */
typedef struct
{
  __REG32 abtpri_r0             : 2;
  __REG32                       : 2;
  __REG32 abtpri_r1             : 2;
  __REG32                       : 2;
  __REG32 abtpri_w0             : 2;
  __REG32                       : 2;
  __REG32 abtpri_w1             : 2;
  __REG32                       :14;
  __REG32 abtmod                : 1;
  __REG32                       : 2;
  __REG32 abt_en                : 1;
} __udarbtset_bits;

/* UDPWCTL (Power Detect Control register) */
typedef struct
{
  __REG32 usb_reset             : 1;
  __REG32 pw_resetb             : 1;
  __REG32 pw_detect             : 1;
  __REG32 phy_suspend           : 1;
  __REG32 suspend_x             : 1;
  __REG32 phy_resetb            : 1;
  __REG32 phy_remote_wkup       : 1;
  __REG32 wakeup_en             : 1;
  __REG32                       :24;
} __udpwctl_bits;

/* UDMSTSTS (Master Status register) */
typedef struct
{
  __REG32 mwepdset              : 1;
  __REG32 mrepdset              : 1;
  __REG32 mwbfemp               : 1;
  __REG32 mrbfemp               : 1;
  __REG32 mrepempty             : 1;
  __REG32                       :27;
} __udmststs_bits;

/* UD2ADR (Address-State register) */
typedef struct
{
  __REG32 dev_adr               : 7;
  __REG32                       : 1;
  __REG32 Default               : 1;
  __REG32 Addressed             : 1;
  __REG32 Configured            : 1;
  __REG32 Suspend               : 1;
  __REG32 cur_speed             : 2;
  __REG32 ep_bi_mode            : 1;
  __REG32 stage_err             : 1;
  __REG32                       :16;
} __ud2adr_bits;

/* UD2FRM (Frame register) */
typedef struct
{
  __REG32 frame                 :11;
  __REG32                       : 1;
  __REG32 f_status              : 2;
  __REG32                       : 1;
  __REG32 create_sof            : 1;
  __REG32                       :16;
} __ud2frm_bits;

/* UD2CMD (Command register) */
typedef struct
{
  __REG32 com                   : 4;
  __REG32 ep                    : 4;
  __REG32 rx_nullpkt_ep         : 4;
  __REG32                       : 3;
  __REG32 int_toggle            : 1;
  __REG32                       :16;
} __ud2cmd_bits;

/* UD2BRQ (bRequest-bmRequestType register) */
typedef struct
{
  __REG32 recipient             : 5;
  __REG32 req_type              : 2;
  __REG32 dir                   : 1;
  __REG32 request               : 8;
  __REG32                       :16;
} __ud2brq_bits;

/* UD2WVL (wValue register) */
typedef struct
{
  __REG32 value_l               : 8;
  __REG32 value_h               : 8;
  __REG32                       :16;
} __ud2wvl_bits;

/* UD2WIDX (wIndex register) */
typedef struct
{
  __REG32 index_l               : 8;
  __REG32 index_h               : 8;
  __REG32                       :16;
} __ud2widx_bits;

/* UD2WLGTH (wLength register) */
typedef struct
{
  __REG32 length_l              : 8;
  __REG32 length_h              : 8;
  __REG32                       :16;
} __ud2wlgth_bits;

/* UD2INT (INT register)*/
typedef struct
{
  __REG32 i_setup               : 1;
  __REG32 i_status_nak          : 1;
  __REG32 i_status              : 1;
  __REG32 i_rx_data0            : 1;
  __REG32 i_sof                 : 1;
  __REG32 i_ep0                 : 1;
  __REG32 i_ep                  : 1;
  __REG32 i_nak                 : 1;
  __REG32 m_setup               : 1;
  __REG32 m_status_nak          : 1;
  __REG32 m_status              : 1;
  __REG32 m_rx_data0            : 1;
  __REG32 m_sof                 : 1;
  __REG32 m_ep0                 : 1;
  __REG32 m_ep                  : 1;
  __REG32 m_nak                 : 1;
  __REG32                       :16;
} __ud2int_bits;

/* UD2INT (INT register)
   UD2INTNAK (INT_NAK register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 i_ep1                 : 1;
  __REG32 i_ep2                 : 1;
  __REG32 i_ep3                 : 1;
  __REG32 i_ep4                 : 1;
  __REG32 i_ep5                 : 1;
  __REG32 i_ep6                 : 1;
  __REG32 i_ep7                 : 1;
  __REG32                       :24;
} __ud2intep_bits;

/* UD2INTEPMSK (INT_EP_MASK register)
   UD2INTNAKMSK (INT_NAK_MASK register)*/
typedef struct
{
  __REG32 m_ep0                 : 1;
  __REG32 m_ep1                 : 1;
  __REG32 m_ep2                 : 1;
  __REG32 m_ep3                 : 1;
  __REG32 m_ep4                 : 1;
  __REG32 m_ep5                 : 1;
  __REG32 m_ep6                 : 1;
  __REG32 m_ep7                 : 1;
  __REG32                       :24;
} __ud2intepmsk_bits;

/* UD2INTRX0 (INT_RX_DATA0 register) */
typedef struct
{
  __REG32 rx_d0_ep0             : 1;
  __REG32 rx_d0_ep1             : 1;
  __REG32 rx_d0_ep2             : 1;
  __REG32 rx_d0_ep3             : 1;
  __REG32 rx_d0_ep4             : 1;
  __REG32 rx_d0_ep5             : 1;
  __REG32 rx_d0_ep6             : 1;
  __REG32 rx_d0_ep7             : 1;
  __REG32                       :24;
} __ud2intrx0_bits;

/* UD2EP0MSZ (EPn_Max0PacketSize register) */
typedef struct
{
  __REG32 max_pkt               : 7;
  __REG32                       : 5;
  __REG32 dset                  : 1;
  __REG32                       : 2;
  __REG32 tx_0data              : 1;
  __REG32                       :16;
} __ud2ep0msz_bits;

/* UD2EPnMSZ (EPn_MaxPacketSize register) */
typedef struct
{
  __REG32 max_pkt               :11;
  __REG32                       : 1;
  __REG32 dset                  : 1;
  __REG32                       : 2;
  __REG32 tx_0data              : 1;
  __REG32                       :16;
} __ud2epmsz_bits;

/* UD2EP0STS (EP0_ Status register) */
typedef struct
{
  __REG32                       : 9;
  __REG32 status                : 3;
  __REG32 toggle                : 2;
  __REG32                       : 1;
  __REG32 ep0_mask              : 1;
  __REG32                       :16;
} __ud2ep0sts_bits;

/* UD2EP0DSZ (EP0_Datasize register) */
typedef struct
{
  __REG32 size                  : 7;
  __REG32                       :25;
} __ud2ep0dsz_bits;

/* UD2EPnFIFO (EPn_FIFO register) */
typedef struct
{
  __REG32 data                  :16;
  __REG32                       :16;
} __ud2epfifo_bits;

/* UD2EP1STS (EP1_Status register) */
typedef struct
{
  __REG32 num_mf                : 2;
  __REG32 t_type                : 2;
  __REG32                       : 3;
  __REG32 dir                   : 1;
  __REG32 disable               : 1;
  __REG32 status                : 3;
  __REG32 toggle                : 2;
  __REG32 bus_sel               : 1;
  __REG32 pkt_mode              : 1;
  __REG32                       :16;
} __ud2ep1sts_bits;

/* UD2EP1DSZ (EP1_Datasize register) */
typedef struct
{
  __REG32 size                  :11;
  __REG32                       :21;
} __ud2ep1dsz_bits;

/*SIOx Enable register*/
typedef struct {
  __REG32  SIOE     : 1;
  __REG32           :31;
} __scxen_bits;

/*SIOx Buffer register*/
typedef struct {
  __REG32  RB_TB    : 8;
  __REG32           :24;
} __scxbuf_bits;

/*SIOx Control register*/
typedef struct {
  __REG32  IOC      : 1;
  __REG32  SCLKS    : 1;
  __REG32  FERR     : 1;
  __REG32  PERR     : 1;
  __REG32  OERR     : 1;
  __REG32  PE       : 1;
  __REG32  EVEN     : 1;
  __REG32  RB8      : 1;
  __REG32           :24;
} __scxcr_bits;

/*SIOx Mode control register 0*/
typedef struct {
  __REG32  SC       : 2;
  __REG32  SM       : 2;
  __REG32  WU       : 1;
  __REG32  RXE      : 1;
  __REG32  CTSE     : 1;
  __REG32  TB8      : 1;
  __REG32           :24;
} __scxmod0_bits;

/*SIOx Baud rate generator control register*/
typedef struct {
  __REG32  BRS      : 4;
  __REG32  BRCK     : 2;
  __REG32  BRADDE   : 1;
  __REG32           :25;
} __scxbrcr_bits;

/*SIOx Baud rate generator control register 2*/
typedef struct {
  __REG32  BRK      : 4;
  __REG32           :28;
} __scxbradd_bits;

/*SIOx Mode control register 1*/
typedef struct {
  __REG32           : 1;
  __REG32  SINT     : 3;
  __REG32  TXE      : 1;
  __REG32  FDPX     : 2;
  __REG32  I2SC     : 1;
  __REG32           :24;
} __scxmod1_bits;

/*SIOx Mode control register 2*/
typedef struct {
  __REG32  SWRST    : 2;
  __REG32  WBUF     : 1;
  __REG32  DRCHG    : 1;
  __REG32  SBLEN    : 1;
  __REG32  TXRUN    : 1;
  __REG32  RBFLL    : 1;
  __REG32  TBEMP    : 1;
  __REG32           :24;
} __scxmod2_bits;

/*SIOx RX FIFO configuration register*/
typedef struct {
  __REG32  RIL      : 2;
  __REG32           : 4;
  __REG32  RFIS     : 1;
  __REG32  RFCS     : 1;
  __REG32           :24;
} __scxrfc_bits;

/*SIOx TX FIFO configuration register*/
typedef struct {
  __REG32  TIL      : 2;
  __REG32           : 4;
  __REG32  TFIS     : 1;
  __REG32  TFCS     : 1;
  __REG32           :24;
} __scxtfc_bits;

/*SIOx RX FIFO status register*/
typedef struct {
  __REG32  RLVL     : 3;
  __REG32           : 4;
  __REG32  ROR      : 1;
  __REG32           :24;
} __scxrst_bits;

/*SIOx TX FIFO status register*/
typedef struct {
  __REG32  TLVL     : 3;
  __REG32           : 4;
  __REG32  TUR      : 1;
  __REG32           :24;
} __scxtst_bits;

/*SIOx FIFO configuration register*/
typedef struct {
  __REG32  CNFG     : 1;
  __REG32  RXTXCNT  : 1;
  __REG32  RFIE     : 1;
  __REG32  TFIE     : 1;
  __REG32  RFST     : 1;
  __REG32           :27;
} __scxfcnf_bits;

/*SIOx DMA enable register*/
typedef struct {
  __REG32  DMAEN0   : 1;
  __REG32  DMAEN1   : 1;
  __REG32           :30;
} __scxdma_bits;

/*Serial bus control register 0*/
typedef struct {
  __REG32           : 7;
  __REG32  SBIEN    : 1;
  __REG32           :24;
} __sbixcr0_bits;

/*Serial bus control register 1*/
typedef union {
  union{
    /*I2CxCR1*/
    struct{
      __REG32  SCK      : 3;
      __REG32           : 1;
      __REG32  ACK      : 1;
      __REG32  BC       : 3;
      __REG32           :24;
    };
    struct {
      __REG32  SWRMON   : 1;
      __REG32           :31;
    };
  };
  /*SIOxCR1*/
  struct {
      __REG32  SCK      : 3;
      __REG32           : 1;
      __REG32  SIOM     : 2;
      __REG32  SIOINH   : 1;
      __REG32  SIOS     : 1;
      __REG32           :24;
  } __sio;
} __sbixcr1_bits;

/*Serial bus control register 2*/
/*Serial bus status register*/
typedef union {
   union {
    /*I2CxCR2*/
    struct {
    __REG32 SWRST   : 2;
    __REG32 SBIM    : 2;
    __REG32 PIN     : 1;
    __REG32 BB      : 1;
    __REG32 TRX     : 1;
    __REG32 MST     : 1;
    __REG32         :24;
    };
    /*I2CxSR*/
    struct {
    __REG32 LRB     : 1;
    __REG32 ADO     : 1;
    __REG32 AAS     : 1;
    __REG32 AL      : 1;
    __REG32 PIN     : 1;
    __REG32 BB      : 1;
    __REG32 TRX     : 1;
    __REG32 MST     : 1;
    __REG32         :24;
    } __sr;
  };

  union {
    /*SIOxCR2*/
    struct {
    __REG32         : 2;
    __REG32 SBIM    : 2;
    __REG32         :28;
    };
    /*SIOxSR*/
    struct {
    __REG32         : 2;
    __REG32 SEF     : 1;
    __REG32 SIOF    : 1;
    __REG32         :28;
    } __sr;
  } __sio;
} __sbixcr2_sr_bits;

/*Serial bus interface data buffer register*/
typedef struct {
  __REG32  DB       : 8;
  __REG32           :24;
} __sbixdbr_bits;

/*I2C bus address register*/
typedef struct {
  __REG32 ALS     : 1;
  __REG32 SA      : 7;
  __REG32         :24;
} __sbixi2car_bits;

/*Serial bus interface baud rate register 0*/
typedef struct {
  __REG32         : 6;
  __REG32 I2SBI   : 1;
  __REG32         :25;
} __sbixbr0_bits;

/*A/D Conversion Clock Setting Register*/
typedef struct {
  __REG8  ADCLK   : 3;
  __REG8          : 1;
  __REG8  ADSH    : 4;
} __adclk_bits;

/*A/D Mode Control Register 0*/
typedef struct {
  __REG8  ADS     : 1;
  __REG8  HPADS   : 1;
  __REG8          : 6;
} __admod0_bits;

/*A/D Mode Control Register 1*/
typedef struct {
  __REG8  ADHWE   : 1;
  __REG8  ADHWS   : 1;
  __REG8  HPADHWE : 1;
  __REG8  HPADHWS : 1;
  __REG8          : 1;
  __REG8  RCUT    : 1;
  __REG8  I2AD    : 1;
  __REG8  VREFON  : 1;
} __admod1_bits;

/*A/D Mode Control Register 2*/
typedef struct {
  __REG8  ADCH    : 4;
  __REG8  HPADCH  : 4;
} __admod2_bits;

/*A/D Mode Control Register 3*/
typedef struct {
  __REG8  SCAN    : 1;
  __REG8  REPEAT  : 1;
  __REG8          : 2;
  __REG8  ITM     : 3;
  __REG8          : 1;
} __admod3_bits;

/*A/D Mode Control Register 4*/
typedef struct {
  __REG8  SCANSTA   : 4;
  __REG8  SCANAREA  : 4;
} __admod4_bits;

/*A/D Mode Control Register 5*/
typedef struct {
  __REG8  ADBF    : 1;
  __REG8  EOCF    : 1;
  __REG8  HPADBF  : 1;
  __REG8  HPEOCF  : 1;
  __REG8          : 4;
} __admod5_bits;

/*A/D Mode Control Register 6*/
typedef struct {
  __REG8  ADRST   : 2;
  __REG8          : 6;
} __admod6_bits;

/*A/D Mode Control Register 7*/
typedef struct {
  __REG8  INTADDMA   : 1;
  __REG8  INTADHPDMA : 1;
  __REG8             : 6;
} __admod7_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  ADR     :12;
  __REG32  ADRF    : 1;
  __REG32  ADOVRF  : 1;
  __REG32  ADPOSWF : 1;
  __REG32          :17;
} __adregx_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  ADR      :12;
  __REG32  SPADRARF : 1;
  __REG32  SPOVRA   : 1;
  __REG32           :18;
} __adregsp_bits;

/*A/D Conversion Comparison Control Register 0*/
typedef struct {
  __REG32  REGS0     : 4;
  __REG32  ADBIG0    : 1;
  __REG32            : 2;
  __REG32  CMP0EN    : 1;
  __REG32  CMPCNT0   : 4;
  __REG32            : 20;
} __adcmpcr0_bits;

/*A/D Conversion Comparison Control Register 1*/
typedef struct {
  __REG32  REGS1     : 4;
  __REG32  ADBIG1    : 1;
  __REG32            : 2;
  __REG32  CMP1EN    : 1;
  __REG32  CMPCNT1   : 4;
  __REG32            : 20;
} __adcmpcr1_bits;

/*A/D Conversion Result Comparison Register 0*/
typedef struct {
  __REG32  AD0CMP   :12;
  __REG32           :20;
} __adcmp0_bits;

/*A/D Conversion Result Comparison Register 1*/
typedef struct {
  __REG32  AD1CMP   :12;
  __REG32           :20;
} __adcmp1_bits;

/*Watchdog Timer Mode Register*/
typedef struct {
  __REG8          : 1;
  __REG8  RESCR   : 1;
  __REG8  I2WDT   : 1;
  __REG8          : 1;
  __REG8  WDTP    : 3;
  __REG8  WDTE    : 1;
} __wdmod_bits;

/*Security bit register*/
typedef struct {
  __REG32 SECBIT  : 1;
  __REG32         :31;
} __fcsecbit_bits;

/*Flash Control Register*/
typedef struct {
  __REG32 RDY_BSY : 1;
  __REG32         :15;
  __REG32 BLPRO0  : 1;
  __REG32 BLPRO1  : 1;
  __REG32 BLPRO2  : 1;
  __REG32 BLPRO3  : 1;
  __REG32 BLPRO4  : 1;
  __REG32 BLPRO5  : 1;
  __REG32         :10;
} __fcflcs_bits;

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
  __REG32                 : 1;
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
  __REG32                 : 1;
} __setena0_bits;

/* Interrupt Set-Enable Registers 32-63 */
typedef struct {
  __REG32                 : 1;
  __REG32                 : 1;
  __REG32  SETENA34       : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
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
  __REG32                 : 1;
} __clrena0_bits;

/* Interrupt Clear-Enable Registers 32-63 */
typedef struct {
  __REG32                 : 1;
  __REG32                 : 1;
  __REG32  CLRENA34       : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
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
  __REG32                 : 1;
} __setpend0_bits;

/* Interrupt Set-Pending Register 32-63 */
typedef struct {
  __REG32                 : 1;
  __REG32                 : 1;
  __REG32  SETPEND34      : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
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
  __REG32                 : 1;
} __clrpend0_bits;

/* Interrupt Clear-Pending Register 32-63 */
typedef struct {
  __REG32                 : 1;
  __REG32                 : 1;
  __REG32  CLRPEND34      : 1;
  __REG32                 : 1;
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
  __REG32                 : 1;
  __REG32                 : 1;
  __REG32  CLRPEND60      : 1;
  __REG32  CLRPEND61      : 1;
  __REG32  CLRPEND62      : 1;
  __REG32  CLRPEND63      : 1;
} __clrpend1_bits;

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
  __REG32                 : 8;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __pri3_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32  PRI_12         : 8;
  __REG32  PRI_13         : 8;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __spri3_bits;

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
  __REG32                 : 8;
} __pri7_bits;

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32                 : 8;
  __REG32                 : 8;
  __REG32  PRI_34         : 8;
  __REG32                 : 8;
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
  __REG32                 : 8;
  __REG32                 : 8;
} __pri14_bits;

/* Interrupt Priority Registers 60-63 */
typedef struct {
  __REG32  PRI_60         : 8;
  __REG32  PRI_61         : 8;
  __REG32  PRI_62         : 8;
  __REG32  PRI_63         : 8;
} __pri15_bits;

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
  __REG32  USGFAULTPENDED : 1;
  __REG32  MEMFAULTPENDED : 1;
  __REG32  BUSFAULTPENDED : 1;
  __REG32  SVCALLPENDED   : 1;
  __REG32  MEMFAULTENA    : 1;
  __REG32  BUSFAULTENA    : 1;
  __REG32  USGFAULTENA    : 1;
  __REG32                 :13;
} __shcsr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSTICKCSR,          0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,          0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,          0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,        0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,             0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(SETENA1,             0xE000E104,__READ_WRITE ,__setena1_bits);
__IO_REG32_BIT(CLRENA0,             0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,             0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(SETPEND0,            0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,            0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(CLRPEND0,            0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,            0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(IP0,                 0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                 0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                 0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                 0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                 0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                 0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                 0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                 0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(IP8,                 0xE000E420,__READ_WRITE ,__pri8_bits);
__IO_REG32_BIT(IP9,                 0xE000E424,__READ_WRITE ,__pri9_bits);
__IO_REG32_BIT(IP10,                0xE000E428,__READ_WRITE ,__pri10_bits);
__IO_REG32_BIT(IP11,                0xE000E42C,__READ_WRITE ,__pri11_bits);
__IO_REG32_BIT(IP12,                0xE000E430,__READ_WRITE ,__pri12_bits);
__IO_REG32_BIT(IP13,                0xE000E434,__READ_WRITE ,__pri13_bits);
__IO_REG32_BIT(IP14,                0xE000E438,__READ_WRITE ,__pri14_bits);
__IO_REG32_BIT(IP15,                0xE000E43C,__READ_WRITE ,__pri15_bits);
__IO_REG32_BIT(VTOR,                0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,               0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SHPR0,               0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,               0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,               0xE000ED20,__READ_WRITE ,__spri3_bits);
__IO_REG32_BIT(SHCSR,               0xE000ED24,__READ_WRITE ,__shcsr_bits);

/***************************************************************************
 **
 ** CG (Clcok generator)
 **
 ***************************************************************************/
__IO_REG32_BIT(CGSYSCR,             0x400F3000,__READ_WRITE ,__cgsyscr_bits);
__IO_REG32_BIT(CGOSCCR,             0x400F3004,__READ_WRITE ,__cgosccr_bits);
__IO_REG32_BIT(CGSTBYCR,            0x400F3008,__READ_WRITE ,__cgstbycr_bits);
__IO_REG32_BIT(CGPLLSEL,            0x400F300C,__READ_WRITE ,__cgpllsel_bits);
__IO_REG32_BIT(CGUSBCTL,            0x400F3038,__READ_WRITE ,__cgusbctl_bits);
__IO_REG32_BIT(CGPROTECT,           0x400F303C,__READ_WRITE ,__cgprotect_bits);
__IO_REG32_BIT(CGIMCGA,             0x400F3040,__READ_WRITE ,__cgimcga_bits);
__IO_REG32_BIT(CGIMCGB,             0x400F3044,__READ_WRITE ,__cgimcgb_bits);
__IO_REG32_BIT(CGIMCGC,             0x400F3048,__READ_WRITE ,__cgimcgc_bits);
__IO_REG32_BIT(CGICRCG,             0x400F3060,__WRITE      ,__cgicrcg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x400F3064,__READ_WRITE ,__cgrstflg_bits);
__IO_REG32_BIT(CGNMIFLG,            0x400F3068,__READ       ,__cgnmiflg_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x400C0000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x400C0004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAOD,                 0x400C0028,__READ_WRITE ,__paod_bits);
__IO_REG8_BIT(PAPUP,                0x400C002C,__READ_WRITE ,__papup_bits);
__IO_REG8_BIT(PAIE,                 0x400C0038,__READ_WRITE ,__paie_bits);

/***************************************************************************
 **
 ** PORTB
 **
 ***************************************************************************/
__IO_REG8_BIT(PBDATA,               0x400C0100,__READ_WRITE ,__pb_bits);
__IO_REG8_BIT(PBCR,                 0x400C0104,__READ_WRITE ,__pbcr_bits);
__IO_REG8_BIT(PBOD,                 0x400C0128,__READ_WRITE ,__pbod_bits);
__IO_REG8_BIT(PBPUP,                0x400C012C,__READ_WRITE ,__pbpup_bits);
__IO_REG8_BIT(PBIE,                 0x400C0138,__READ_WRITE ,__pbie_bits);

/***************************************************************************
 **
 ** PORTC
 **
 ***************************************************************************/
__IO_REG8_BIT(PCDATA,               0x400C0200,__READ_WRITE ,__pc_bits);
__IO_REG8_BIT(PCCR,                 0x400C0204,__READ_WRITE ,__pccr_bits);
__IO_REG8_BIT(PCFR1,                0x400C0208,__READ_WRITE ,__pcfr1_bits);
__IO_REG8_BIT(PCFR3,                0x400C0210,__READ_WRITE ,__pcfr3_bits);
__IO_REG8_BIT(PCFR4,                0x400C0214,__READ_WRITE ,__pcfr4_bits);
__IO_REG8_BIT(PCOD,                 0x400C0228,__READ_WRITE ,__pcod_bits);
__IO_REG8_BIT(PCPUP,                0x400C022C,__READ_WRITE ,__pcpup_bits);
__IO_REG8_BIT(PCIE,                 0x400C0238,__READ_WRITE ,__pcie_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG8_BIT(PDDATA,               0x400C0300,__READ_WRITE ,__pd_bits);
__IO_REG8_BIT(PDCR  ,               0x400C0304,__READ_WRITE ,__pdcr_bits);
__IO_REG8_BIT(PDFR3,                0x400C0310,__READ_WRITE ,__pdfr3_bits);
__IO_REG8_BIT(PDOD,                 0x400C0328,__READ_WRITE ,__pdod_bits);
__IO_REG8_BIT(PDPUP,                0x400C032C,__READ_WRITE ,__pdpup_bits);
__IO_REG8_BIT(PDIE,                 0x400C0338,__READ_WRITE ,__pdie_bits);

/***************************************************************************
 **
 ** PORTE
 **
 ***************************************************************************/
__IO_REG8_BIT(PEDATA,               0x400C0400,__READ_WRITE ,__pe_bits);
__IO_REG8_BIT(PECR,                 0x400C0404,__READ_WRITE ,__pecr_bits);
__IO_REG8_BIT(PEFR1,                0x400C0408,__READ_WRITE ,__pefr1_bits);
__IO_REG8_BIT(PEFR3,                0x400C0410,__READ_WRITE ,__pefr3_bits);
__IO_REG8_BIT(PEFR4,                0x400C0414,__READ_WRITE ,__pefr4_bits);
__IO_REG8_BIT(PEOD,                 0x400C0428,__READ_WRITE ,__peod_bits);
__IO_REG8_BIT(PEPUP,                0x400C042C,__READ_WRITE ,__pepup_bits);
__IO_REG8_BIT(PEIE,                 0x400C0438,__READ_WRITE ,__peie_bits);

/***************************************************************************
 **
 ** PORTF
 **
 ***************************************************************************/
__IO_REG8_BIT(PFDATA,               0x400C0500,__READ_WRITE ,__pf_bits);
__IO_REG8_BIT(PFCR,                 0x400C0504,__READ_WRITE ,__pfcr_bits);
__IO_REG8_BIT(PFFR2,                0x400C050C,__READ_WRITE ,__pffr2_bits);
__IO_REG8_BIT(PFFR3,                0x400C0510,__READ_WRITE ,__pffr3_bits);
__IO_REG8_BIT(PFOD,                 0x400C0528,__READ_WRITE ,__pfod_bits);
__IO_REG8_BIT(PFPUP,                0x400C052C,__READ_WRITE ,__pfpup_bits);
__IO_REG8_BIT(PFIE,                 0x400C0538,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG8_BIT(PGDATA,               0x400C0600,__READ_WRITE ,__pg_bits);
__IO_REG8_BIT(PGCR,                 0x400C0604,__READ_WRITE ,__pgcr_bits);
__IO_REG8_BIT(PGFR1,                0x400C0608,__READ_WRITE ,__pgfr1_bits);
__IO_REG8_BIT(PGFR3,                0x400C0610,__READ_WRITE ,__pgfr3_bits);
__IO_REG8_BIT(PGFR4,                0x400C0614,__READ_WRITE ,__pgfr4_bits);
__IO_REG8_BIT(PGOD,                 0x400C0628,__READ_WRITE ,__pgod_bits);
__IO_REG8_BIT(PGPUP,                0x400C062C,__READ_WRITE ,__pgpup_bits);
__IO_REG8_BIT(PGIE,                 0x400C0638,__READ_WRITE ,__pgie_bits);

/***************************************************************************
 **
 ** PORTH
 **
 ***************************************************************************/
__IO_REG8_BIT(PHDATA,               0x400C0700,__READ_WRITE ,__ph_bits);
__IO_REG8_BIT(PHCR,                 0x400C0704,__READ_WRITE ,__phcr_bits);
__IO_REG8_BIT(PHFR1,                0x400C0708,__READ_WRITE ,__phfr1_bits);
__IO_REG8_BIT(PHFR3,                0x400C0710,__READ_WRITE ,__phfr3_bits);
__IO_REG8_BIT(PHOD,                 0x400C0728,__READ_WRITE ,__phod_bits);
__IO_REG8_BIT(PHPUP,                0x400C072C,__READ_WRITE ,__phpup_bits);
__IO_REG8_BIT(PHIE,                 0x400C0738,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTI
 **
 ***************************************************************************/
__IO_REG8_BIT(PIDATA,               0x400C0800,__READ_WRITE ,__pi_bits);
__IO_REG8_BIT(PICR,                 0x400C0804,__READ_WRITE ,__picr_bits);
__IO_REG8_BIT(PIFR1,                0x400C0808,__READ_WRITE ,__pifr1_bits);
__IO_REG8_BIT(PIOD,                 0x400C0828,__READ_WRITE ,__piod_bits);
__IO_REG8_BIT(PIPUP,                0x400C082C,__READ_WRITE ,__pipup_bits);
__IO_REG8_BIT(PIPDN,                0x400C0830,__READ_WRITE ,__pipdn_bits);
__IO_REG8_BIT(PIIE,                 0x400C0838,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x400C0900,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJCR,                 0x400C0904,__READ_WRITE ,__pjcr_bits);
__IO_REG8_BIT(PJFR2,                0x400C090C,__READ_WRITE ,__pjfr2_bits);
__IO_REG8_BIT(PJFR3,                0x400C0910,__READ_WRITE ,__pjfr3_bits);
__IO_REG8_BIT(PJPUP,                0x400C092C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJIE,                 0x400C0938,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG8_BIT(PKDATA,               0x400C0A00,__READ_WRITE ,__pk_bits);
__IO_REG8_BIT(PKCR,                 0x400C0A04,__READ_WRITE ,__pkcr_bits);
__IO_REG8_BIT(PKFR2,                0x400C0A0C,__READ_WRITE ,__pkfr2_bits);
__IO_REG8_BIT(PKFR3,                0x400C0A10,__READ_WRITE ,__pkfr3_bits);
__IO_REG8_BIT(PKPUP,                0x400C0A2C,__READ_WRITE ,__pkpup_bits);
__IO_REG8_BIT(PKIE,                 0x400C0A38,__READ_WRITE ,__pkie_bits);

/***************************************************************************
 **
 ** DMACA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACAIntStaus,          0x40000000,__READ       ,__dmacintstaus_bits);
__IO_REG32_BIT(DMACAIntTCStatus,       0x40000004,__READ       ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACAIntTCClear,        0x40000008,__WRITE      ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACAIntErrorStatus,    0x4000000C,__READ       ,__dmacinterrorstatus_bits);
__IO_REG32_BIT(DMACAIntErrClr,         0x40000010,__WRITE      ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACARawIntTCStatus,    0x40000014,__READ       ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACARawIntErrorStatus, 0x40000018,__READ       ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACAEnbldChns,         0x4000001C,__READ       ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACASoftBReq,          0x40000020,__READ_WRITE ,__dmacsoftbreq_bits);
__IO_REG32(    DMACASoftSReq,          0x40000024,__READ_WRITE );
__IO_REG32_BIT(DMACAConfiguration,     0x40000030,__READ_WRITE ,__dmacconfiguration_bits);
__IO_REG32(    DMACAC0SrcAddr,         0x40000100,__READ_WRITE );
__IO_REG32(    DMACAC0DestAddr,        0x40000104,__READ_WRITE );
__IO_REG32(    DMACAC0LLI,             0x40000108,__READ_WRITE );
__IO_REG32_BIT(DMACAC0Control,         0x4000010C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACAC0Configuration,   0x40000110,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACAC1SrcAddr,         0x40000120,__READ_WRITE );
__IO_REG32(    DMACAC1DestAddr,        0x40000124,__READ_WRITE );
__IO_REG32(    DMACAC1LLI,             0x40000128,__READ_WRITE );
__IO_REG32_BIT(DMACAC1Control,         0x4000012C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACAC1Configuration,   0x40000130,__READ_WRITE ,__dmaccconfiguration_bits);

/***************************************************************************
 **
 ** TMRB0
 **
 ***************************************************************************/
__IO_REG32_BIT(TB0EN,               0x400C4000, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB0RUN,              0x400C4004, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB0CR,               0x400C4008, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB0MOD,              0x400C400C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB0FFCR,             0x400C4010, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB0ST,               0x400C4014, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB0IM,               0x400C4018, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB0UC,               0x400C401C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB0RG0,              0x400C4020, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB0RG1,              0x400C4024, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB0CP0,              0x400C4028, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB0CP1,              0x400C402C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB0DMA,              0x400C4030, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB1
 **
 ***************************************************************************/
__IO_REG32_BIT(TB1EN,               0x400C4100, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB1RUN,              0x400C4104, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB1CR,               0x400C4108, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB1MOD,              0x400C410C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB1FFCR,             0x400C4110, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB1ST,               0x400C4114, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB1IM,               0x400C4118, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB1UC,               0x400C411C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB1RG0,              0x400C4120, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB1RG1,              0x400C4124, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB1CP0,              0x400C4128, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB1CP1,              0x400C412C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB1DMA,              0x400C4130, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB2
 **
 ***************************************************************************/
__IO_REG32_BIT(TB2EN,               0x400C4200, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB2RUN,              0x400C4204, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB2CR,               0x400C4208, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB2MOD,              0x400C420C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB2FFCR,             0x400C4210, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB2ST,               0x400C4214, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB2IM,               0x400C4218, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB2UC,               0x400C421C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB2RG0,              0x400C4220, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB2RG1,              0x400C4224, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB2CP0,              0x400C4228, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB2CP1,              0x400C422C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB2DMA,              0x400C4230, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB3
 **
 ***************************************************************************/
__IO_REG32_BIT(TB3EN,               0x400C4300, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB3RUN,              0x400C4304, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB3CR,               0x400C4308, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB3MOD,              0x400C430C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB3FFCR,             0x400C4310, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB3ST,               0x400C4314, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB3IM,               0x400C4318, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB3UC,               0x400C431C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB3RG0,              0x400C4320, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB3RG1,              0x400C4324, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB3CP0,              0x400C4328, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB3CP1,              0x400C432C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB3DMA,              0x400C4330, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB4
 **
 ***************************************************************************/
__IO_REG32_BIT(TB4EN,               0x400C4400, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB4RUN,              0x400C4404, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB4CR,               0x400C4408, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB4MOD,              0x400C440C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB4FFCR,             0x400C4410, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB4ST,               0x400C4414, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB4IM,               0x400C4418, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB4UC,               0x400C441C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB4RG0,              0x400C4420, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB4RG1,              0x400C4424, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB4CP0,              0x400C4428, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB4CP1,              0x400C442C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB4DMA,              0x400C4430, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB5
 **
 ***************************************************************************/
__IO_REG32_BIT(TB5EN,               0x400C4500, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB5RUN,              0x400C4504, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB5CR,               0x400C4508, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB5MOD,              0x400C450C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB5FFCR,             0x400C4510, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB5ST,               0x400C4514, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB5IM,               0x400C4518, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB5UC,               0x400C451C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB5RG0,              0x400C4520, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB5RG1,              0x400C4524, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB5CP0,              0x400C4528, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB5CP1,              0x400C452C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB5DMA,              0x400C4530, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB6
 **
 ***************************************************************************/
__IO_REG32_BIT(TB6EN,               0x400C4600, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB6RUN,              0x400C4604, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB6CR,               0x400C4608, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB6MOD,              0x400C460C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB6FFCR,             0x400C4610, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB6ST,               0x400C4614, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB6IM,               0x400C4618, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB6UC,               0x400C461C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB6RG0,              0x400C4620, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB6RG1,              0x400C4624, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB6CP0,              0x400C4628, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB6CP1,              0x400C462C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB6DMA,              0x400C4630, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB7
 **
 ***************************************************************************/
__IO_REG32_BIT(TB7EN,               0x400C4700, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB7RUN,              0x400C4704, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB7CR,               0x400C4708, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB7MOD,              0x400C470C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB7FFCR,             0x400C4710, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB7ST,               0x400C4714, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB7IM,               0x400C4718, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB7UC,               0x400C471C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB7RG0,              0x400C4720, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB7RG1,              0x400C4724, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB7CP0,              0x400C4728, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB7CP1,              0x400C472C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB7DMA,              0x400C4730, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB8
 **
 ***************************************************************************/
__IO_REG32_BIT(TB8EN,               0x400C4800, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB8RUN,              0x400C4804, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB8CR,               0x400C4808, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB8MOD,              0x400C480C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB8FFCR,             0x400C4810, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB8ST,               0x400C4814, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB8IM,               0x400C4818, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB8UC,               0x400C481C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB8RG0,              0x400C4820, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB8RG1,              0x400C4824, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB8CP0,              0x400C4828, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB8CP1,              0x400C482C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB8DMA,              0x400C4830, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** TMRB9
 **
 ***************************************************************************/
__IO_REG32_BIT(TB9EN,               0x400C4900, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB9RUN,              0x400C4904, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB9CR,               0x400C4908, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB9MOD,              0x400C490C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB9FFCR,             0x400C4910, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB9ST,               0x400C4914, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB9IM,               0x400C4918, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB9UC,               0x400C491C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB9RG0,              0x400C4920, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB9RG1,              0x400C4924, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB9CP0,              0x400C4928, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB9CP1,              0x400C492C, __READ       , __tbxcp1_bits);
__IO_REG32_BIT(TB9DMA,              0x400C4930, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** UDC2AB Bridge
 **
 ***************************************************************************/
__IO_REG32_BIT(UDFSINTSTS,            0x40008000,__READ_WRITE ,__udintsts_bits);
__IO_REG32_BIT(UDFSINTENB,            0x40008004,__READ_WRITE ,__udintenb_bits);
__IO_REG32_BIT(UDFSMWTOUT,            0x40008008,__READ_WRITE ,__udmwtout_bits);
__IO_REG32_BIT(UDFSC2STSET,           0x4000800C,__READ_WRITE ,__udc2stset_bits);
__IO_REG32_BIT(UDFSMSTSET,            0x40008010,__READ_WRITE ,__udmstset_bits);
__IO_REG32_BIT(UDFSDMACRDREQ,         0x40008014,__READ_WRITE ,__uddmacrdreq_bits);
__IO_REG32(    UDFSDMACRDVL,          0x40008018,__READ       );
__IO_REG32_BIT(UDFSUDC2RDREQ,         0x4000801C,__READ_WRITE ,__udc2rdreq_bits);
__IO_REG32_BIT(UDFSUDC2RDVL,          0x40008020,__READ       ,__udc2rdvl_bits);
__IO_REG32_BIT(UDFSARBTSET,           0x4000803C,__READ_WRITE ,__udarbtset_bits);
__IO_REG32(    UDFSMWSADR,            0x40008040,__READ_WRITE );
__IO_REG32(    UDFSMWEADR,            0x40008044,__READ_WRITE );
__IO_REG32(    UDFSMWCADR,            0x40008048,__READ       );
__IO_REG32(    UDFSMWAHBADR,          0x4000804C,__READ       );
__IO_REG32(    UDFSMRSADR,            0x40008050,__READ_WRITE );
__IO_REG32(    UDFSMREADR,            0x40008054,__READ_WRITE );
__IO_REG32(    UDFSMRCADR,            0x40008058,__READ       );
__IO_REG32(    UDFSMRAHBADR,          0x4000805C,__READ       );
__IO_REG32_BIT(UDFSPWCTL,             0x40008080,__READ_WRITE ,__udpwctl_bits);
__IO_REG32_BIT(UDFSMSTSTS,            0x40008084,__READ       ,__udmststs_bits);
__IO_REG32(    UDFSTOUTCNT,           0x40008088,__READ       );

/***************************************************************************
 **
 ** UDC2
 **
 ***************************************************************************/
__IO_REG32_BIT(UDFS2ADR,              0x40008200,__READ_WRITE ,__ud2adr_bits);
__IO_REG32_BIT(UDFS2FRM,              0x40008204,__READ_WRITE ,__ud2frm_bits);
__IO_REG32_BIT(UDFS2CMD,              0x4000820C,__READ_WRITE ,__ud2cmd_bits);
__IO_REG32_BIT(UDFS2BRQ,              0x40008210,__READ       ,__ud2brq_bits);
__IO_REG32_BIT(UDFS2WVL,              0x40008214,__READ       ,__ud2wvl_bits);
__IO_REG32_BIT(UDFS2WIDX,             0x40008218,__READ       ,__ud2widx_bits);
__IO_REG32_BIT(UDFS2WLGTH,            0x4000821C,__READ       ,__ud2wlgth_bits);
__IO_REG32_BIT(UDFS2INT,              0x40008220,__READ_WRITE ,__ud2int_bits);
__IO_REG32_BIT(UDFS2INTEP,            0x40008224,__READ_WRITE ,__ud2intep_bits);
__IO_REG32_BIT(UDFS2INTEPMSK,         0x40008228,__READ_WRITE ,__ud2intepmsk_bits);
__IO_REG32_BIT(UDFS2INTRX0,           0x4000822C,__READ_WRITE ,__ud2intrx0_bits);
__IO_REG32_BIT(UDFS2EP0MSZ,           0x40008230,__READ_WRITE ,__ud2ep0msz_bits);
__IO_REG32_BIT(UDFS2EP0STS,           0x40008234,__READ       ,__ud2ep0sts_bits);
__IO_REG32_BIT(UDFS2EP0DSZ,           0x40008238,__READ       ,__ud2ep0dsz_bits);
__IO_REG32_BIT(UDFS2EP0FIFO,          0x4000823C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP1MSZ,           0x40008240,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP1STS,           0x40008244,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP1DSZ,           0x40008248,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP1FIFO,          0x4000824C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP2MSZ,           0x40008250,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP2STS,           0x40008254,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP2DSZ,           0x40008258,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP2FIFO,          0x4000825C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP3MSZ,           0x40008260,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP3STS,           0x40008264,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP3DSZ,           0x40008268,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP3FIFO,          0x4000826C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP4MSZ,           0x40008270,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP4STS,           0x40008274,__READ       ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP4DSZ,           0x40008278,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP4FIFO,          0x4000827C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP5MSZ,           0x40008280,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP5STS,           0x40008284,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP5DSZ,           0x40008288,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP5FIFO,          0x4000828C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP6MSZ,           0x40008290,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP6STS,           0x40008294,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP6DSZ,           0x40008298,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP6FIFO,          0x4000829C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP7MSZ,           0x400082A0,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP7STS,           0x400082A4,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP7DSZ,           0x400082A8,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP7FIFO,          0x400082AC,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2INTNAK,           0x40008330,__READ_WRITE ,__ud2intep_bits);
__IO_REG32_BIT(UDFS2INTNAKMSK,        0x40008334,__READ_WRITE ,__ud2intepmsk_bits);

/***************************************************************************
 **
 ** SIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(SC0EN,               0x400E1000, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC0BUF,              0x400E1004, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC0CR,               0x400E1008, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC0MOD0,             0x400E100C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC0BRCR,             0x400E1010, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC0BRADD,            0x400E1014, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC0MOD1,             0x400E1018, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC0MOD2,             0x400E101C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC0RFC,              0x400E1020, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC0TFC,              0x400E1024, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC0RST,              0x400E1028, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC0TST,              0x400E102C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC0FCNF,             0x400E1030, __READ_WRITE , __scxfcnf_bits);
__IO_REG32_BIT(SC0DMA,              0x400E1034, __READ_WRITE , __scxdma_bits);

/***************************************************************************
 **
 ** SIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(SC1EN,               0x400E1100, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC1BUF,              0x400E1104, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC1CR,               0x400E1108, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC1MOD0,             0x400E110C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC1BRCR,             0x400E1110, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC1BRADD,            0x400E1114, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC1MOD1,             0x400E1118, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC1MOD2,             0x400E111C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC1RFC,              0x400E1120, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC1TFC,              0x400E1124, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC1RST,              0x400E1128, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC1TST,              0x400E112C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC1FCNF,             0x400E1130, __READ_WRITE , __scxfcnf_bits);
__IO_REG32_BIT(SC1DMA,              0x400E1134, __READ_WRITE , __scxdma_bits);

/***************************************************************************
 **
 ** SBI0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CR0,             0x400E0000, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C0CR1,             0x400E0004, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C0DBR,             0x400E0008, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C0AR,              0x400E000C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C0CR2,             0x400E0010, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C0SR      I2C0CR2
#define I2C0SR_bit  I2C0CR2_bit.__sr
__IO_REG32_BIT(I2C0BR0,             0x400E0014, __READ_WRITE , __sbixbr0_bits);

#define SIO0CR0     I2C0CR0
#define SIO0CR0_bit I2C0CR0_bit
#define SIO0CR1     I2C0CR1
#define SIO0CR1_bit I2C0CR1_bit.__sio
#define SIO0DBR     I2C0DBR
#define SIO0DBR_bit I2C0DBR_bit
#define SIO0CR2     I2C0CR2
#define SIO0CR2_bit I2C0CR2_bit.__sio
#define SIO0SR      I2C0CR2
#define SIO0SR_bit  I2C0CR2_bit.__sio.__sr
#define SIO0BR0     I2C0BR0
#define SIO0BR0_bit I2C0BR0_bit

/***************************************************************************
 **
 ** SBI1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CR0,             0x400E0100, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C1CR1,             0x400E0104, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C1DBR,             0x400E0108, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C1AR,              0x400E010C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C1CR2,             0x400E0110, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C1SR      I2C1CR2
#define I2C1SR_bit  I2C1CR2_bit.__sr
__IO_REG32_BIT(I2C1BR0,             0x400E0114, __READ_WRITE , __sbixbr0_bits);

#define SIO1CR0     I2C1CR0
#define SIO1CR0_bit I2C1CR0_bit
#define SIO1CR1     I2C1CR1
#define SIO1CR1_bit I2C1CR1_bit.__sio
#define SIO1DBR     I2C1DBR
#define SIO1DBR_bit I2C1DBR_bit
#define SIO1CR2     I2C1CR2
#define SIO1CR2_bit I2C1CR2_bit.__sio
#define SIO1SR      I2C1CR2
#define SIO1SR_bit  I2C1CR2_bit.__sio.__sr
#define SIO1BR0     I2C1BR0
#define SIO1BR0_bit I2C1BR0_bit

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG8_BIT(ADCLK,                 0x40050000,__READ_WRITE ,__adclk_bits);
__IO_REG8_BIT(ADMOD0,                0x40050004,__READ_WRITE ,__admod0_bits);
__IO_REG8_BIT(ADMOD1,                0x40050008,__READ_WRITE ,__admod1_bits);
__IO_REG8_BIT(ADMOD2,                0x4005000C,__READ_WRITE ,__admod2_bits);
__IO_REG8_BIT(ADMOD3,                0x40050010,__READ_WRITE ,__admod3_bits);
__IO_REG8_BIT(ADMOD4,                0x40050014,__READ_WRITE ,__admod4_bits);
__IO_REG8_BIT(ADMOD5,                0x40050018,__READ_WRITE ,__admod5_bits);
__IO_REG8_BIT(ADMOD6,                0x4005001C,__READ_WRITE ,__admod6_bits);
__IO_REG8_BIT(ADMOD7,                0x40050020,__READ_WRITE ,__admod7_bits);
__IO_REG32_BIT(ADCMPCR0,             0x40050024,__READ_WRITE ,__adcmpcr0_bits);
__IO_REG32_BIT(ADCMPCR1,             0x40050028,__READ_WRITE ,__adcmpcr1_bits);
__IO_REG32_BIT(ADCMP0,               0x4005002C,__READ_WRITE ,__adcmp0_bits);
__IO_REG32_BIT(ADCMP1,               0x40050030,__READ_WRITE ,__adcmp1_bits);
__IO_REG32_BIT(ADREG00,              0x40050034,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG01,              0x40050038,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG02,              0x4005003C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG03,              0x40050040,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG04,              0x40050044,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG05,              0x40050048,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG06,              0x4005004C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG07,              0x40050050,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG08,              0x40050054,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG09,              0x40050058,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG10,              0x4005005C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG11,              0x40050060,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREGSP,              0x40050074,__READ       ,__adregsp_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG8_BIT(WDMOD,                0x400F2000,__READ_WRITE ,__wdmod_bits);
__IO_REG8(    WDCR,                 0x400F2004,__WRITE);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FCSECBIT,            0x41FFF010, __READ_WRITE , __fcsecbit_bits);
__IO_REG32_BIT(FCFLCS,              0x41FFF020, __READ       , __fcflcs_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  TMPM365FYXBG DMACA Request Lines
 **
 ***************************************************************************/
#define DMACA_SIO0_RX          0          /* SIO0/UART0 Reception          */
#define DMACA_SIO0_TX          1          /* SIO0/UART0 Transmission       */
#define DMACA_SIO1_RX          2          /* SIO1/UART1 Reception          */
#define DMACA_SIO1_TX          3          /* SIO1/UART1 Transmission       */
#define DMACA_TMRB8_COM        4          /* TMRB8 compare match           */
#define DMACA_TMRB9_COM        5          /* TMRB9 compare match           */
#define DMACA_TMRB0_CAP0       6          /* TMRB0 input capture 0         */
#define DMACA_TMRB4_CAP0       7          /* TMRB4 input capture 0         */
#define DMACA_TMRB4_CAP1       8          /* TMRB4 input capture 1         */
#define DMACA_TMRB5_CAP0       9          /* TMRB5 input capture 0         */
#define DMACA_TMRB5_CAP1      10          /* TMRB5 input capture 1         */
#define DMACA_ADC_TOP         11          /* Top priority A/D Conversion   */
#define DMACA_I2C0_RX         12          /* I2C1 / SIO0 Reception         */
#define DMACA_I2C0_TX         13          /* I2C1 / SIO0 Transmission      */
#define DMACA_I2C1_RX         14          /* I2C1 / SIO1 Reception         */
#define DMACA_I2C1_TX         15          /* I2C1 / SIO1 Transmission      */


/***************************************************************************
 **
 **  TMPM365FYXBG Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK             0          /* Main Stack                    */
#define RESETI                 1          /* Reset                         */
#define NMII                   2          /* Non-maskable Interrupt        */
#define HFI                    3          /* Hard Fault                    */
#define MMI                    4          /* Memory Management             */
#define BFI                    5          /* Bus Fault                     */
#define UFI                    6          /* Usage Fault                   */
#define SVCI                  11          /* SVCall                        */
#define DMI                   12          /* Debug Monitor                 */
#define PSI                   14          /* PendSV                        */
#define STI                   15          /* SysTick                       */
#define EII                   16          /* External Interrupt            */
#define INT_0                ( 0 + EII)   /* External Interrupt 0         */
#define INT_1                ( 1 + EII)   /* External Interrupt 1         */
#define INT_2                ( 2 + EII)   /* External Interrupt 2         */
#define INT_3                ( 3 + EII)   /* External Interrupt 3         */
#define INT_4                ( 4 + EII)   /* External Interrupt 4         */
#define INT_5                ( 5 + EII)   /* External Interrupt 5         */
#define INT_6                ( 6 + EII)   /* External Interrupt 6         */
#define INT_7                ( 7 + EII)   /* External Interrupt 7         */
#define INT_RX0              ( 8 + EII)   /* Serial reception (channel.0) */
#define INT_TX0              ( 9 + EII)   /* Serial transmit (channel.0)  */
#define INT_RX1              (10 + EII)   /* Serial reception (channel.1) */
#define INT_TX1              (11 + EII)   /* Serial transmit (channel.1)  */
#define INT_USBWKUP          (12 + EII)   /* USB Wake-up interrupt        */
#define INT_SBI0             (14 + EII)   /* Serial bus interface 0       */
#define INT_SBI1             (15 + EII)   /* Serial bus interface 1       */
#define INT_ADHP             (16 + EII)   /* Highest priority AD conversion complete interrupt*/
#define INT_AD               (17 + EII)   /* AD conversion complete interrupt */
#define INT_ADM0             (18 + EII)   /* AD conversion monitoring function interrupt 0*/
#define INT_ADM1             (19 + EII)   /* AD conversion monitoring function interrupt 1*/
#define INT_TB0              (20 + EII)   /* 16bit TMRB0 match detection  */
#define INT_TB1              (21 + EII)   /* 16bit TMRB1 match detection  */
#define INT_TB2              (22 + EII)   /* 16bit TMRB2 match detection  */
#define INT_TB3              (23 + EII)   /* 16bit TMRB3 match detection  */
#define INT_TB4              (24 + EII)   /* 16bit TMRB4 match detection  */
#define INT_TB5              (25 + EII)   /* 16bit TMRB5 match detection  */
#define INT_TB6              (26 + EII)   /* 16bit TMRB6 match detection  */
#define INT_TB7              (27 + EII)   /* 16bit TMRB7 match detection  */
#define INT_TB8              (28 + EII)   /* 16bit TMRB8 match detection  */
#define INT_TB9              (29 + EII)   /* 16bit TMRB9 match detection  */
#define INT_USB              (30 + EII)   /* USB interrupt */
#define INT_USBPON           (34 + EII)   /* USB Power On connection detection interrupt */
#define INT_CAP00            (36 + EII)   /* 16bit TMRB input capture 00  */
#define INT_CAP01            (37 + EII)   /* 16bit TMRB input capture 01  */
#define INT_CAP10            (38 + EII)   /* 16bit TMRB input capture 10  */
#define INT_CAP11            (39 + EII)   /* 16bit TMRB input capture 11  */
#define INT_CAP20            (40 + EII)   /* 16bit TMRB input capture 20  */
#define INT_CAP21            (41 + EII)   /* 16bit TMRB input capture 21  */
#define INT_CAP30            (42 + EII)   /* 16bit TMRB input capture 30  */
#define INT_CAP31            (43 + EII)   /* 16bit TMRB input capture 31  */
#define INT_CAP40            (44 + EII)   /* 16bit TMRB input capture 40  */
#define INT_CAP41            (45 + EII)   /* 16bit TMRB input capture 41  */
#define INT_CAP50            (46 + EII)   /* 16bit TMRB input capture 50  */
#define INT_CAP51            (47 + EII)   /* 16bit TMRB input capture 51  */
#define INT_CAP60            (48 + EII)   /* 16bit TMRB input capture 60  */
#define INT_CAP61            (49 + EII)   /* 16bit TMRB input capture 61  */
#define INT_CAP70            (50 + EII)   /* 16bit TMRB input capture 70  */
#define INT_CAP71            (51 + EII)   /* 16bit TMRB input capture 71  */
#define INT_CAP80            (52 + EII)   /* 16bit TMRB input capture 80  */
#define INT_CAP81            (53 + EII)   /* 16bit TMRB input capture 81  */
#define INT_CAP90            (54 + EII)   /* 16bit TMRB input capture 90  */
#define INT_CAP91            (55 + EII)   /* 16bit TMRB input capture 91  */
#define INT_8                (56 + EII)   /* External Interrupt 8         */
#define INT_9                (57 + EII)   /* External Interrupt 9         */
#define INT_DMAC0TC          (60 + EII)   /* DMAC0 transfer complete Interrupt */
#define INT_ABTLOSS0         (61 + EII)   /* I2C Arbitration loss Interrupt (channel 0) */
#define INT_DMAC0ERR         (62 + EII)   /* DMAC0 transfer error Interrupt */
#define INT_ABTLOSS1         (63 + EII)   /* I2C Arbitration loss Interrupt (channel 1) */

#endif    /* __IOTMPM365FYXBG_H */

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
Interrupt9   = INT0           0x40
Interrupt10  = INT1           0x44
Interrupt11  = INT2           0x48
Interrupt12  = INT3           0x4C
Interrupt13  = INT4           0x50
Interrupt14  = INT5           0x54
Interrupt15  = INT6           0x58
Interrupt16  = INT7           0x5C
Interrupt17  = INTRX0         0x60
Interrupt18  = INTTX0         0x64
Interrupt19  = INTRX1         0x68
Interrupt20  = INTTX1         0x6C
Interrupt21  = INTUSBWKUP     0x70
Interrupt22  = INTSBI0        0x78
Interrupt23  = INTSBI1        0x7C
Interrupt24  = INTADHP        0x80
Interrupt25  = INTAD          0x84
Interrupt26  = INTADM0        0x88
Interrupt27  = INTADM1        0x8C
Interrupt28  = INTTB0         0x90
Interrupt29  = INTTB1         0x94
Interrupt30  = INTTB2         0x98
Interrupt31  = INTTB3         0x9C
Interrupt32  = INTTB4         0xA0
Interrupt33  = INTTB5         0xA4
Interrupt34  = INTTB6         0xA8
Interrupt35  = INTTB7         0xAC
Interrupt36  = INTTB8         0xB0
Interrupt37  = INTTB9         0xB4
Interrupt38  = INTUSB         0xB8
Interrupt40  = INTUSBPON      0xC8
Interrupt42  = INTCAP00       0xD0
Interrupt43  = INTCAP01       0xD4
Interrupt44  = INTCAP10       0xD8
Interrupt45  = INTCAP11       0xDC
Interrupt46  = INTCAP20       0xE0
Interrupt47  = INTCAP21       0xE4
Interrupt48  = INTCAP30       0xE8
Interrupt49  = INTCAP31       0xEC
Interrupt50  = INTCAP40       0xF0
Interrupt51  = INTCAP41       0xF4
Interrupt52  = INTCAP50       0xF8
Interrupt53  = INTCAP51       0xFC
Interrupt54  = INTCAP60       0x100
Interrupt55  = INTCAP61       0x104
Interrupt56  = INTCAP70       0x108
Interrupt57  = INTCAP71       0x10C
Interrupt58  = INTCAP80       0x110
Interrupt59  = INTCAP81       0x114
Interrupt60  = INTCAP90       0x118
Interrupt61  = INTCAP91       0x11C
Interrupt62  = INT8           0x120
Interrupt63  = INT9           0x124
Interrupt66  = INTDMAC0TC     0x130
Interrupt67  = INTABTLOSS0    0x134
Interrupt68  = INTDMAC0ERR    0x138
Interrupt69  = INTABTLOSS1    0x13C

###DDF-INTERRUPT-END###*/
