/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM362F10FG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 41558 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM362F10FG_H
#define __IOTMPM362F10FG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM362F10FG SPECIAL FUNCTION REGISTERS
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

/* RAMWAIT Register */
typedef struct {
  __REG32  RAM1WAIT : 1;
  __REG32           :31;
} __rcwait_bits;

/* System Control Register */
typedef struct {
  __REG32 GEAR    : 3;
  __REG32         : 5;
  __REG32 PRCK    : 3;
  __REG32         : 1;
  __REG32 FPSEL   : 2;
  __REG32         : 2;
  __REG32 SCOSEL  : 2;
  __REG32         : 2;
  __REG32 FCSTOP  : 1;
  __REG32         :11;
} __cgsyscr_bits;

/* Oscillation Control Register */
typedef struct {
  __REG32 WUEON   : 1;
  __REG32 WUEF    : 1;
  __REG32 PLLON   : 1;
  __REG32 WUPSEL  : 1;
  __REG32         : 4;
  __REG32 XEN     : 1;
  __REG32 XTEN    : 1;
  __REG32         : 4;
  __REG32 WUPTL   : 2;
  __REG32         : 4;
  __REG32 WUPT    :12;
} __cgosccr_bits;

/* Standby Control Register */
typedef struct {
  __REG32 STBY    : 3;
  __REG32         : 5;
  __REG32 RXEN    : 1;
  __REG32 RXTEN   : 1;
  __REG32         : 6;
  __REG32 DRVE    : 1;
  __REG32 PTKEEP  : 1;
  __REG32         :14;
} __cgstbycr_bits;

/* PLL Selection Register */
typedef struct {
  __REG32 PLLSEL  : 1;
  __REG32         : 2;
  __REG32 ND      : 5;
  __REG32 C2S     : 1;
  __REG32 IS      : 2;
  __REG32         : 1;
  __REG32 RS      : 4;
  __REG32         :16;
} __cgpllsel_bits;

/* System Clock Selection Register */
typedef struct {
  __REG32 SYSCKFLG  : 1;
  __REG32 SYSCK     : 1;
  __REG32           :30;
} __cgcksel_bits;

/* INTCG Clear Register */
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           :27;
} __cgicrcg_bits;

/* NMI Flag Register */
typedef struct {
  __REG32 NMIFLG0   : 1;
  __REG32 NMIFLG1   : 1;
  __REG32           :30;
} __cgnmiflg_bits;

/* Reset Flag Register */
typedef struct {
  __REG32 PONRSTF   : 1;
  __REG32 PINRSTF   : 1;
  __REG32 WDTRSTF   : 1;
  __REG32 BUPRSTF   : 1;
  __REG32 SYSRSTF   : 1;
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

/* CG Interrupt Mode Control Register D */
typedef struct {
  __REG32 INTCEN    : 1;
  __REG32           : 1;
  __REG32 EMSTC     : 2;
  __REG32 EMCGC     : 3;
  __REG32           : 1;
  __REG32 INTDEN    : 1;
  __REG32           : 1;
  __REG32 EMSTD     : 2;
  __REG32 EMCGD     : 3;
  __REG32           : 1;
  __REG32 INTEEN    : 1;
  __REG32           : 1;
  __REG32 EMSTE     : 2;
  __REG32 EMCGE     : 3;
  __REG32           : 1;
  __REG32 INTFEN    : 1;
  __REG32           : 1;
  __REG32 EMSTF     : 2;
  __REG32 EMCGF     : 3;
  __REG32           : 1;
} __cgimcgd_bits;

/* CG Interrupt Mode Control Register E */
typedef struct {
  __REG32 INTGEN    : 1;
  __REG32           : 1;
  __REG32 EMSTG     : 2;
  __REG32 EMCGG     : 3;
  __REG32           : 1;
  __REG32 INTHEN    : 1;
  __REG32           : 1;
  __REG32 EMSTH     : 2;
  __REG32 EMCGH     : 3;
  __REG32           : 1;
  __REG32 INTIEN    : 1;
  __REG32           : 1;
  __REG32 EMSTI     : 2;
  __REG32 EMCGI     : 3;
  __REG32           : 1;
  __REG32 INTJEN    : 1;
  __REG32           : 1;
  __REG32 EMSTJ     : 2;
  __REG32 EMCGJ     : 3;
  __REG32           : 1;
} __cgimcge_bits;

/* CG Interrupt Mode Control Register F */
typedef struct {
  __REG32 INTKEN    : 1;
  __REG32           : 1;
  __REG32 EMSTK     : 2;
  __REG32 EMCGK     : 3;
  __REG32           : 1;
  __REG32 INTLEN    : 1;
  __REG32           : 1;
  __REG32 EMSTL     : 2;
  __REG32 EMCGL     : 3;
  __REG32           :17;
} __cgimcgf_bits;

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

/* PORT A Function Register 1 */
typedef struct {
  __REG8  PA0F1  : 1;
  __REG8  PA1F1  : 1;
  __REG8  PA2F1  : 1;
  __REG8  PA3F1  : 1;
  __REG8  PA4F1  : 1;
  __REG8  PA5F1  : 1;
  __REG8  PA6F1  : 1;
  __REG8         : 1;
} __pafr1_bits;

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

/* PORT B Function Register 1 */
typedef struct {
  __REG8  PB0F1  : 1;
  __REG8  PB1F1  : 1;
  __REG8  PB2F1  : 1;
  __REG8  PB3F1  : 1;
  __REG8  PB4F1  : 1;
  __REG8  PB5F1  : 1;
  __REG8  PB6F1  : 1;
  __REG8  PB7F1  : 1;
} __pbfr1_bits;

/* PortB open drain control register */
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
  __REG8  PC3  : 1;
  __REG8  PC4  : 1;
  __REG8  PC5  : 1;
  __REG8  PC6  : 1;
  __REG8  PC7  : 1;
} __pc_bits;

/* PORT C Control Register */
typedef struct {
  __REG8  PC0C  : 1;
  __REG8  PC1C  : 1;
  __REG8  PC2C  : 1;
  __REG8  PC3C  : 1;
  __REG8  PC4C  : 1;
  __REG8  PC5C  : 1;
  __REG8  PC6C  : 1;
  __REG8  PC7C  : 1;
} __pccr_bits;

/* PORT C Function Register 1 */
typedef struct {
  __REG8  PC0F1  : 1;
  __REG8  PC1F1  : 1;
  __REG8  PC2F1  : 1;
  __REG8  PC3F1  : 1;
  __REG8  PC4F1  : 1;
  __REG8  PC5F1  : 1;
  __REG8  PC6F1  : 1;
  __REG8  PC7F1  : 1;
} __pcfr1_bits;

/* PORT C Function Register 2 */
typedef struct {
  __REG8  PC0F2  : 1;
  __REG8  PC1F2  : 1;
  __REG8  PC2F2  : 1;
  __REG8         : 1;
  __REG8  PC4F2  : 1;
  __REG8  PC5F2  : 1;
  __REG8  PC6F2  : 1;
  __REG8         : 1;
} __pcfr2_bits;

/* PORT C Function Register 3 */
typedef struct {
  __REG8         : 2;
  __REG8  PC2F3  : 1;
  __REG8         : 3;
  __REG8  PC6F3  : 1;
  __REG8         : 1;
} __pcfr3_bits;

/* PortC open drain control register */
typedef struct {
  __REG8  PC0OD  : 1;
  __REG8  PC1OD  : 1;
  __REG8  PC2OD  : 1;
  __REG8  PC3OD  : 1;
  __REG8  PC4OD  : 1;
  __REG8  PC5OD  : 1;
  __REG8  PC6OD  : 1;
  __REG8  PC7OD  : 1;
} __pcod_bits;

/* PORT C Pull-Up Control Register */
typedef struct {
  __REG8  PC0UP  : 1;
  __REG8  PC1UP  : 1;
  __REG8  PC2UP  : 1;
  __REG8  PC3UP  : 1;
  __REG8  PC4UP  : 1;
  __REG8  PC5UP  : 1;
  __REG8  PC6UP  : 1;
  __REG8  PC7UP  : 1;
} __pcpup_bits;

/*PORT C Input Enable Control Register */
typedef struct {
  __REG8  PC0IE  : 1;
  __REG8  PC1IE  : 1;
  __REG8  PC2IE  : 1;
  __REG8  PC3IE  : 1;
  __REG8  PC4IE  : 1;
  __REG8  PC5IE  : 1;
  __REG8  PC6IE  : 1;
  __REG8  PC7IE  : 1;
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

/* PORT D Function Register 1 */
typedef struct {
  __REG8  PD0F1  : 1;
  __REG8  PD1F1  : 1;
  __REG8  PD2F1  : 1;
  __REG8  PD3F1  : 1;
  __REG8  PD4F1  : 1;
  __REG8  PD5F1  : 1;
  __REG8  PD6F1  : 1;
  __REG8  PD7F1  : 1;
} __pdfr1_bits;

/* PORT D Function Register 2 */
typedef struct {
  __REG8  PD0F2  : 1;
  __REG8  PD1F2  : 1;
  __REG8  PD2F2  : 1;
  __REG8         : 1;
  __REG8  PD4F2  : 1;
  __REG8  PD5F2  : 1;
  __REG8  PD6F2  : 1;
  __REG8  PD7F2  : 1;
} __pdfr2_bits;

/* PORT D Function Register 3 */
typedef struct {
  __REG8         : 2;
  __REG8  PD2F3  : 1;
  __REG8         : 3;
  __REG8  PD6F3  : 1;
  __REG8         : 1;
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
  __REG8         : 1;
} __pefr1_bits;

/* PORT E Function Register 2 */
typedef struct {
  __REG8  PE0F2  : 1;
  __REG8  PE1F2  : 1;
  __REG8  PE2F2  : 1;
  __REG8  PE3F2  : 1;
  __REG8  PE4F2  : 1;
  __REG8  PE5F2  : 1;
  __REG8         : 2;
} __pefr2_bits;

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
  __REG8       : 3;
} __pf_bits;

/* PORT F Control Register */
typedef struct {
  __REG8  PF0C  : 1;
  __REG8  PF1C  : 1;
  __REG8  PF2C  : 1;
  __REG8  PF3C  : 1;
  __REG8  PF4C  : 1;
  __REG8        : 3;
} __pfcr_bits;

/* PORT F Function Register 1 */
typedef struct {
  __REG8  PF0F1  : 1;
  __REG8  PF1F1  : 1;
  __REG8  PF2F1  : 1;
  __REG8  PF3F1  : 1;
  __REG8  PF4F1  : 1;
  __REG8         : 3;
} __pffr1_bits;

/* PORT F Open Drain Control Register */
typedef struct {
  __REG8  PF0OD  : 1;
  __REG8  PF1OD  : 1;
  __REG8  PF2OD  : 1;
  __REG8  PF3OD  : 1;
  __REG8  PF4OD  : 1;
  __REG8         : 3;
} __pfod_bits;

/* PORT F Pull-Up Control Register */
typedef struct {
  __REG8  PF0UP  : 1;
  __REG8  PF1UP  : 1;
  __REG8  PF2UP  : 1;
  __REG8  PF3UP  : 1;
  __REG8  PF4UP  : 1;
  __REG8         : 3;
} __pfpup_bits;

/* PORT F Input Enable Control Register */
typedef struct {
  __REG8  PF0IE  : 1;
  __REG8  PF1IE  : 1;
  __REG8  PF2IE  : 1;
  __REG8  PF3IE  : 1;
  __REG8  PF4IE  : 1;
  __REG8         : 3;
} __pfie_bits;

/* PORT G Register */
typedef struct {
  __REG8  PG0  : 1;
  __REG8  PG1  : 1;
  __REG8  PG2  : 1;
  __REG8  PG3  : 1;
  __REG8  PG4  : 1;
  __REG8  PG5  : 1;
  __REG8  PG6  : 1;
  __REG8  PG7  : 1;
} __pg_bits;

/* PortG control register */
typedef struct {
  __REG8  PG0C  : 1;
  __REG8  PG1C  : 1;
  __REG8  PG2C  : 1;
  __REG8  PG3C  : 1;
  __REG8  PG4C  : 1;
  __REG8  PG5C  : 1;
  __REG8  PG6C  : 1;
  __REG8  PG7C  : 1;
} __pgcr_bits;

/* PORT G Function Register 1 */
typedef struct {
  __REG8  PG0F1  : 1;
  __REG8  PG1F1  : 1;
  __REG8  PG2F1  : 1;
  __REG8  PG3F1  : 1;
  __REG8  PG4F1  : 1;
  __REG8  PG5F1  : 1;
  __REG8  PG6F1  : 1;
  __REG8  PG7F1  : 1;
} __pgfr1_bits;

/* PORT G Function Register 2 */
typedef struct {
  __REG8  PG0F2  : 1;
  __REG8  PG1F2  : 1;
  __REG8         : 2;
  __REG8  PG4F2  : 1;
  __REG8  PG5F2  : 1;
  __REG8         : 2;
} __pgfr2_bits;

/* PORT G Function Register 3 */
typedef struct {
  __REG8         : 2;
  __REG8  PG2F3  : 1;
  __REG8  PG3F3  : 1;
  __REG8         : 2;
  __REG8  PG6F3  : 1;
  __REG8  PG7F3  : 1;
} __pgfr3_bits;

/* PORT G Open Drain Control Register */
typedef struct {
  __REG8  PG0OD  : 1;
  __REG8  PG1OD  : 1;
  __REG8  PG2OD  : 1;
  __REG8  PG3OD  : 1;
  __REG8  PG4OD  : 1;
  __REG8  PG5OD  : 1;
  __REG8  PG6OD  : 1;
  __REG8  PG7OD  : 1;
} __pgod_bits;

/* PORT G Pull-Up Control Register */
typedef struct {
  __REG8  PG0UP  : 1;
  __REG8  PG1UP  : 1;
  __REG8  PG2UP  : 1;
  __REG8  PG3UP  : 1;
  __REG8  PG4UP  : 1;
  __REG8  PG5UP  : 1;
  __REG8  PG6UP  : 1;
  __REG8  PG7UP  : 1;
} __pgpup_bits;

/* PORT G Input Enable Control Register */
typedef struct {
  __REG8  PG0IE  : 1;
  __REG8  PG1IE  : 1;
  __REG8  PG2IE  : 1;
  __REG8  PG3IE  : 1;
  __REG8  PG4IE  : 1;
  __REG8  PG5IE  : 1;
  __REG8  PG6IE  : 1;
  __REG8  PG7IE  : 1;
} __pgie_bits;

/* PORT H Register */
typedef struct {
  __REG8  PH0  : 1;
  __REG8  PH1  : 1;
  __REG8  PH2  : 1;
  __REG8  PH3  : 1;
  __REG8  PH4  : 1;
  __REG8  PH5  : 1;
  __REG8  PH6  : 1;
  __REG8  PH7  : 1;
} __ph_bits;

/* PortH control register */
typedef struct {
  __REG8  PH0C  : 1;
  __REG8  PH1C  : 1;
  __REG8  PH2C  : 1;
  __REG8  PH3C  : 1;
  __REG8  PH4C  : 1;
  __REG8  PH5C  : 1;
  __REG8  PH6C  : 1;
  __REG8  PH7C  : 1;
} __phcr_bits;

/* PORT H Function Register 1 */
typedef struct {
  __REG8  PH0F1  : 1;
  __REG8  PH1F1  : 1;
  __REG8  PH2F1  : 1;
  __REG8  PH3F1  : 1;
  __REG8  PH4F1  : 1;
  __REG8  PH5F1  : 1;
  __REG8  PH6F1  : 1;
  __REG8  PH7F1  : 1;
} __phfr1_bits;

/* PORT H Function Register 2 */
typedef struct {
  __REG8  PH0F2  : 1;
  __REG8  PH1F2  : 1;
  __REG8  PH2F2  : 1;
  __REG8  PH3F2  : 1;
  __REG8  PH4F2  : 1;
  __REG8  PH5F2  : 1;
  __REG8  PH6F2  : 1;
  __REG8  PH7F2  : 1;
} __phfr2_bits;

/* PortH open drain control register */
typedef struct {
  __REG8  PH0OD  : 1;
  __REG8  PH1OD  : 1;
  __REG8  PH2OD  : 1;
  __REG8  PH3OD  : 1;
  __REG8  PH4OD  : 1;
  __REG8  PH5OD  : 1;
  __REG8  PH6OD  : 1;
  __REG8  PH7OD  : 1;
} __phod_bits;

/* PORT H Pull-Up Control Register */
typedef struct {
  __REG8  PH0UP  : 1;
  __REG8  PH1UP  : 1;
  __REG8  PH2UP  : 1;
  __REG8  PH3UP  : 1;
  __REG8  PH4UP  : 1;
  __REG8  PH5UP  : 1;
  __REG8  PH6UP  : 1;
  __REG8  PH7UP  : 1;
} __phpup_bits;

/* PORT H Input Enable Control Register */
typedef struct {
  __REG8  PH0IE  : 1;
  __REG8  PH1IE  : 1;
  __REG8  PH2IE  : 1;
  __REG8  PH3IE  : 1;
  __REG8  PH4IE  : 1;
  __REG8  PH5IE  : 1;
  __REG8  PH6IE  : 1;
  __REG8  PH7IE  : 1;
} __phie_bits;

/* PORT I Register */
typedef struct {
  __REG8  PI0  : 1;
  __REG8  PI1  : 1;
  __REG8  PI2  : 1;
  __REG8  PI3  : 1;
  __REG8       : 4;
} __pi_bits;

/* PORT I Control Register */
typedef struct {
  __REG8  PI0C  : 1;
  __REG8  PI1C  : 1;
  __REG8  PI2C  : 1;
  __REG8  PI3C  : 1;
  __REG8        : 4;
} __picr_bits;

/* PORT I Function Register 1 */
typedef struct {
  __REG8  PI0F1  : 1;
  __REG8  PI1F1  : 1;
  __REG8  PI2F1  : 1;
  __REG8  PI3F1  : 1;
  __REG8         : 4;
} __pifr1_bits;

/* PortI open drain control register */
typedef struct {
  __REG8  PI0OD  : 1;
  __REG8         : 1;
  __REG8  PI2OD  : 1;
  __REG8  PI3OD  : 1;
  __REG8         : 4;
} __piod_bits;

/*PORT I Pull-Up Control Register */
typedef struct {
  __REG8  PI0UP  : 1;
  __REG8         : 7;
} __pipup_bits;

/*PORT I Input Enable Control Register */
typedef struct {
  __REG8  PI0IE  : 1;
  __REG8  PI1IE  : 1;
  __REG8  PI2IE  : 1;
  __REG8  PI3IE  : 1;
  __REG8         : 4;
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

/* PORT J Function Register 2 */
typedef struct {
  __REG8         : 3;
  __REG8  PJ3F2  : 1;
  __REG8  PJ4F2  : 1;
  __REG8  PJ5F2  : 1;
  __REG8  PJ6F2  : 1;
  __REG8  PJ7F2  : 1;
} __pjfr2_bits;

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
  __REG8  PK4  : 1;
  __REG8  PK5  : 1;
  __REG8  PK6  : 1;
  __REG8  PK7  : 1;
} __pk_bits;

/* PORT K Pull-Up Control Register */
typedef struct {
  __REG8  PK0UP  : 1;
  __REG8  PK1UP  : 1;
  __REG8  PK2UP  : 1;
  __REG8  PK3UP  : 1;
  __REG8  PK4UP  : 1;
  __REG8  PK5UP  : 1;
  __REG8  PK6UP  : 1;
  __REG8  PK7UP  : 1;
} __pkpup_bits;

/* PORT K Input Enable Control Register */
typedef struct {
  __REG8  PK0IE  : 1;
  __REG8  PK1IE  : 1;
  __REG8  PK2IE  : 1;
  __REG8  PK3IE  : 1;
  __REG8  PK4IE  : 1;
  __REG8  PK5IE  : 1;
  __REG8  PK6IE  : 1;
  __REG8  PK7IE  : 1;
} __pkie_bits;

/* PortL register */
typedef struct {
  __REG8  PL0    : 1;
  __REG8  PL1    : 1;
  __REG8  PL2    : 1;
  __REG8  PL3    : 1;
  __REG8  PL4    : 1;
  __REG8  PL5    : 1;
  __REG8  PL6    : 1;
  __REG8  PL7    : 1;
} __pl_bits;

/* PortL control register */
typedef struct {
  __REG8  PL0C   : 1;
  __REG8  PL1C   : 1;
  __REG8  PL2C   : 1;
  __REG8  PL3C   : 1;
  __REG8  PL4C   : 1;
  __REG8  PL5C   : 1;
  __REG8  PL6C   : 1;
  __REG8  PL7C   : 1;
} __plcr_bits;

/* PortL function register1 */
typedef struct {
  __REG8  PL0F1  : 1;
  __REG8  PL1F1  : 1;
  __REG8  PL2F1  : 1;
  __REG8  PL3F1  : 1;
  __REG8  PL4F1  : 1;
  __REG8  PL5F1  : 1;
  __REG8  PL6F1  : 1;
  __REG8  PL7F1  : 1;
} __plfr1_bits;

/* PortL function register2 */
typedef struct {
  __REG8  PL0F2  : 1;
  __REG8  PL1F2  : 1;
  __REG8  PL2F2  : 1;
  __REG8  PL3F2  : 1;
  __REG8  PL4F2  : 1;
  __REG8  PL5F2  : 1;
  __REG8  PL6F2  : 1;
  __REG8  PL7F2  : 1;
} __plfr2_bits;

/* PortL function register3 */
typedef struct {
  __REG8         : 6;
  __REG8  PL6F3  : 1;
  __REG8         : 1;
} __plfr3_bits;

/* PortL open drain control register */
typedef struct {
  __REG8  PL0OD  : 1;
  __REG8  PL1OD  : 1;
  __REG8  PL2OD  : 1;
  __REG8  PL3OD  : 1;
  __REG8  PL4OD  : 1;
  __REG8  PL5OD  : 1;
  __REG8  PL6OD  : 1;
  __REG8  PL7OD  : 1;
} __plod_bits;

/* PortL pull-up control register */
typedef struct {
  __REG8  PL0UP  : 1;
  __REG8  PL1UP  : 1;
  __REG8  PL2UP  : 1;
  __REG8  PL3UP  : 1;
  __REG8  PL4UP  : 1;
  __REG8  PL5UP  : 1;
  __REG8  PL6UP  : 1;
  __REG8  PL7UP  : 1;
} __plpup_bits;

/* PortL input enable control register */
typedef struct {
  __REG8  PL0IE  : 1;
  __REG8  PL1IE  : 1;
  __REG8  PL2IE  : 1;
  __REG8  PL3IE  : 1;
  __REG8  PL4IE  : 1;
  __REG8  PL5IE  : 1;
  __REG8  PL6IE  : 1;
  __REG8  PL7IE  : 1;
} __plie_bits;

/* PortM register */
typedef struct {
  __REG8  PM0    : 1;
  __REG8  PM1    : 1;
  __REG8  PM2    : 1;
  __REG8  PM3    : 1;
  __REG8  PM4    : 1;
  __REG8  PM5    : 1;
  __REG8  PM6    : 1;
  __REG8  PM7    : 1;
} __pm_bits;

/* PortM control register */
typedef struct {
  __REG8  PM0C   : 1;
  __REG8  PM1C   : 1;
  __REG8  PM2C   : 1;
  __REG8  PM3C   : 1;
  __REG8  PM4C   : 1;
  __REG8  PM5C   : 1;
  __REG8  PM6C   : 1;
  __REG8  PM7C   : 1;
} __pmcr_bits;

/* PortM function register1 */
typedef struct {
  __REG8  PM0F1  : 1;
  __REG8  PM1F1  : 1;
  __REG8  PM2F1  : 1;
  __REG8  PM3F1  : 1;
  __REG8  PM4F1  : 1;
  __REG8  PM5F1  : 1;
  __REG8  PM6F1  : 1;
  __REG8  PM7F1  : 1;
} __pmfr1_bits;

/* PortM function register2 */
typedef struct {
  __REG8  PM0F2  : 1;
  __REG8  PM1F2  : 1;
  __REG8  PM2F2  : 1;
  __REG8  PM3F2  : 1;
  __REG8         : 4;
} __pmfr2_bits;

/* PortM function register3 */
typedef struct {
  __REG8  PM0F3  : 1;
  __REG8         : 3;
  __REG8  PM4F3  : 1;
  __REG8         : 3;
} __pmfr3_bits;

/* PortM open drain control register */
typedef struct {
  __REG8  PM0OD  : 1;
  __REG8  PM1OD  : 1;
  __REG8  PM2OD  : 1;
  __REG8  PM3OD  : 1;
  __REG8  PM4OD  : 1;
  __REG8  PM5OD  : 1;
  __REG8  PM6OD  : 1;
  __REG8  PM7OD  : 1;
} __pmod_bits;

/* PortM pull-up control register */
typedef struct {
  __REG8  PM0UP  : 1;
  __REG8  PM1UP  : 1;
  __REG8  PM2UP  : 1;
  __REG8  PM3UP  : 1;
  __REG8  PM4UP  : 1;
  __REG8  PM5UP  : 1;
  __REG8  PM6UP  : 1;
  __REG8  PM7UP  : 1;
} __pmpup_bits;

/* PortM input enable control register */
typedef struct {
  __REG8  PM0IE  : 1;
  __REG8  PM1IE  : 1;
  __REG8  PM2IE  : 1;
  __REG8  PM3IE  : 1;
  __REG8  PM4IE  : 1;
  __REG8  PM5IE  : 1;
  __REG8  PM6IE  : 1;
  __REG8  PM7IE  : 1;
} __pmie_bits;

/* PortN register */
typedef struct {
  __REG8  PN0    : 1;
  __REG8  PN1    : 1;
  __REG8  PN2    : 1;
  __REG8  PN3    : 1;
  __REG8  PN4    : 1;
  __REG8  PN5    : 1;
  __REG8  PN6    : 1;
  __REG8  PN7    : 1;
} __pn_bits;

/* PortN control register */
typedef struct {
  __REG8  PN0C   : 1;
  __REG8  PN1C   : 1;
  __REG8  PN2C   : 1;
  __REG8  PN3C   : 1;
  __REG8  PN4C   : 1;
  __REG8  PN5C   : 1;
  __REG8  PN6C   : 1;
  __REG8  PN7C   : 1;
} __pncr_bits;

/* PortN function register1 */
typedef struct {
  __REG8  PN0F1  : 1;
  __REG8  PN1F1  : 1;
  __REG8  PN2F1  : 1;
  __REG8  PN3F1  : 1;
  __REG8  PN4F1  : 1;
  __REG8  PN5F1  : 1;
  __REG8  PN6F1  : 1;
  __REG8  PN7F1  : 1;
} __pnfr1_bits;

/* PortN function register2 */
typedef struct {
  __REG8         : 2;
  __REG8  PN2F2  : 1;
  __REG8  PN3F2  : 1;
  __REG8         : 2;
  __REG8  PN6F2  : 1;
  __REG8  PN7F2  : 1;
} __pnfr2_bits;

/* PortN function register3 */
typedef struct {
  __REG8         : 2;
  __REG8  PN2F3  : 1;
  __REG8  PN3F3  : 1;
  __REG8         : 2;
  __REG8  PN6F3  : 1;
  __REG8  PN7F3  : 1;
} __pnfr3_bits;

/* PortN open drain control register */
typedef struct {
  __REG8  PN0OD  : 1;
  __REG8  PN1OD  : 1;
  __REG8  PN2OD  : 1;
  __REG8  PN3OD  : 1;
  __REG8  PN4OD  : 1;
  __REG8  PN5OD  : 1;
  __REG8  PN6OD  : 1;
  __REG8  PN7OD  : 1;
} __pnod_bits;

/* PortN pull-up control register */
typedef struct {
  __REG8  PN0UP  : 1;
  __REG8  PN1UP  : 1;
  __REG8  PN2UP  : 1;
  __REG8  PN3UP  : 1;
  __REG8  PN4UP  : 1;
  __REG8  PN5UP  : 1;
  __REG8  PN6UP  : 1;
  __REG8  PN7UP  : 1;
} __pnpup_bits;

/* PortM input enable control register */
typedef struct {
  __REG8  PN0IE  : 1;
  __REG8  PN1IE  : 1;
  __REG8  PN2IE  : 1;
  __REG8  PN3IE  : 1;
  __REG8  PN4IE  : 1;
  __REG8  PN5IE  : 1;
  __REG8  PN6IE  : 1;
  __REG8  PN7IE  : 1;
} __pnie_bits;

/* PortO register */
typedef struct {
  __REG8  PO0    : 1;
  __REG8  PO1    : 1;
  __REG8  PO2    : 1;
  __REG8  PO3    : 1;
  __REG8  PO4    : 1;
  __REG8  PO5    : 1;
  __REG8  PO6    : 1;
  __REG8  PO7    : 1;
} __po_bits;

/* PortO control register */
typedef struct {
  __REG8  PO0C   : 1;
  __REG8  PO1C   : 1;
  __REG8  PO2C   : 1;
  __REG8  PO3C   : 1;
  __REG8  PO4C   : 1;
  __REG8  PO5C   : 1;
  __REG8  PO6C   : 1;
  __REG8  PO7C   : 1;
} __pocr_bits;

/* PortO function register1 */
typedef struct {
  __REG8  PO0F1  : 1;
  __REG8  PO1F1  : 1;
  __REG8  PO2F1  : 1;
  __REG8  PO3F1  : 1;
  __REG8  PO4F1  : 1;
  __REG8  PO5F1  : 1;
  __REG8  PO6F1  : 1;
  __REG8  PO7F1  : 1;
} __pofr1_bits;

/* PortO function register2 */
typedef struct {
  __REG8  PO0F2  : 1;
  __REG8  PO1F2  : 1;
  __REG8  PO2F2  : 1;
  __REG8  PO3F2  : 1;
  __REG8  PO4F2  : 1;
  __REG8  PO5F2  : 1;
  __REG8  PO6F2  : 1;
  __REG8         : 1;
} __pofr2_bits;

/* PortO function register3 */
typedef struct {
  __REG8         : 2;
  __REG8  PO2F3  : 1;
  __REG8         : 3;
  __REG8  PO6F3  : 1;
  __REG8         : 1;
} __pofr3_bits;

/* PortO open drain control register */
typedef struct {
  __REG8  PO0OD  : 1;
  __REG8  PO1OD  : 1;
  __REG8  PO2OD  : 1;
  __REG8  PO3OD  : 1;
  __REG8  PO4OD  : 1;
  __REG8  PO5OD  : 1;
  __REG8  PO6OD  : 1;
  __REG8  PO7OD  : 1;
} __pood_bits;

/* PortO pull-up control register */
typedef struct {
  __REG8  PO0UP  : 1;
  __REG8  PO1UP  : 1;
  __REG8  PO2UP  : 1;
  __REG8  PO3UP  : 1;
  __REG8  PO4UP  : 1;
  __REG8  PO5UP  : 1;
  __REG8  PO6UP  : 1;
  __REG8  PO7UP  : 1;
} __popup_bits;

/* PortO input enable control register */
typedef struct {
  __REG8  PO0IE  : 1;
  __REG8  PO1IE  : 1;
  __REG8  PO2IE  : 1;
  __REG8  PO3IE  : 1;
  __REG8  PO4IE  : 1;
  __REG8  PO5IE  : 1;
  __REG8  PO6IE  : 1;
  __REG8  PO7IE  : 1;
} __poie_bits;

/* PortP register */
typedef struct {
  __REG8  PP0    : 1;
  __REG8  PP1    : 1;
  __REG8  PP2    : 1;
  __REG8  PP3    : 1;
  __REG8  PP4    : 1;
  __REG8  PP5    : 1;
  __REG8  PP6    : 1;
  __REG8         : 1;
} __pp_bits;

/* Portp control register */
typedef struct {
  __REG8  PP0C   : 1;
  __REG8  PP1C   : 1;
  __REG8  PP2C   : 1;
  __REG8  PP3C   : 1;
  __REG8  PP4C   : 1;
  __REG8  PP5C   : 1;
  __REG8  PP6C   : 1;
  __REG8         : 1;
} __ppcr_bits;

/* PortP function register1 */
typedef struct {
  __REG8  PP0F1  : 1;
  __REG8  PP1F1  : 1;
  __REG8  PP2F1  : 1;
  __REG8  PP3F1  : 1;
  __REG8  PP4F1  : 1;
  __REG8  PP5F1  : 1;
  __REG8  PP6F1  : 1;
  __REG8         : 1;
} __ppfr1_bits;

/* PortP function register2 */
typedef struct {
  __REG8         : 2;
  __REG8  PP2F2  : 1;
  __REG8  PP3F2  : 1;
  __REG8  PP4F2  : 1;
  __REG8  PP5F2  : 1;
  __REG8         : 2;
} __ppfr2_bits;

/* PortP open drain control register */
typedef struct {
  __REG8  PP0OD  : 1;
  __REG8  PP1OD  : 1;
  __REG8  PP2OD  : 1;
  __REG8  PP3OD  : 1;
  __REG8  PP4OD  : 1;
  __REG8  PP5OD  : 1;
  __REG8  PP6OD  : 1;
  __REG8         : 1;
} __ppod_bits;

/* PortP pull-up control register */
typedef struct {
  __REG8  PP0UP  : 1;
  __REG8  PP1UP  : 1;
  __REG8  PP2UP  : 1;
  __REG8  PP3UP  : 1;
  __REG8  PP4UP  : 1;
  __REG8  PP5UP  : 1;
  __REG8  PP6UP  : 1;
  __REG8         : 1;
} __pppup_bits;

/* PortP input enable control register */
typedef struct {
  __REG8  PP0IE  : 1;
  __REG8  PP1IE  : 1;
  __REG8  PP2IE  : 1;
  __REG8  PP3IE  : 1;
  __REG8  PP4IE  : 1;
  __REG8  PP5IE  : 1;
  __REG8  PP6IE  : 1;
  __REG8         : 1;
} __ppie_bits;

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

/* DMACSoftSReq (DMAC Software Burst Request Register) */
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
__REG32                       : 1;
__REG32 SoftBReq15            : 1;
__REG32                       :16;
} __dmacsoftbreq_bits;

/* DMACSoftSReq (DMAC Software Single Request Register ) */
typedef struct{
__REG32                       :13;
__REG32 SoftSReq13            : 1;
__REG32                       :18;
} __dmacsoftsreq_bits;

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

/* SMCMDMODE (SMC MODE Register) */
typedef struct {
  __REG32  IFSMCMUXMD         : 1;
  __REG32                     :31;
} __smcmdmode_bits;

/* smc_memif_cfg (SMC Memory Interface Configuration Register) */
typedef struct{
__REG32 memory_type           : 2;
__REG32 memory_chips          : 2;
__REG32 memory_width          : 2;
__REG32                       :26;
} __smc_memif_cfg_bits;

/* smc_direct_cmd (SMC Direct Command Register) */
typedef struct{
__REG32                       :21;
__REG32 cmd_type              : 2;
__REG32 chip_select           : 3;
__REG32                       : 6;
} __smc_direct_cmd_bits;

/* smc_set_cycles (SMC Set Cycles Register) */
typedef struct{
__REG32 Set_t0                : 4;
__REG32 Set_t1                : 4;
__REG32 Set_t2                : 3;
__REG32 Set_t3                : 3;
__REG32 Set_t4                : 3;
__REG32 Set_t5                : 3;
__REG32                       :12;
} __smc_set_cycles_bits;

/* smc_set_opmode (SMC Set Opmode Register) */
typedef struct{
__REG32 set_mw                : 2;
__REG32                       : 1;
__REG32 set_rd_bl             : 3;
__REG32                       : 5;
__REG32 set_adv               : 1;
__REG32                       :20;
} __smc_set_opmode_bits;

/* smc_sram_cycles0_n (SMC SRAM Cycles Registers 0 <0..3>) */
typedef struct{
__REG32 t_rc                  : 4;
__REG32 t_wc                  : 4;
__REG32 t_ceoe                : 3;
__REG32 t_wp                  : 3;
__REG32 t_pc                  : 3;
__REG32 t_tr                  : 3;
__REG32                       :12;
} __smc_sram_cycles0_bits;

/* smc_opmode0_n (SMC Opmode Registers 0<0..3>) */
typedef struct{
__REG32 mw                    : 2;
__REG32                       : 1;
__REG32 rd_bl                 : 3;
__REG32                       : 5;
__REG32 adv                   : 1;
__REG32                       :12;
__REG32 address_match         : 8;
} __smc_opmode0_bits;

/*TMRBn enable register (channels 0 through 9)*/
typedef struct {
  __REG32           : 7;
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
  __REG32           : 3;
  __REG32  I2TB     : 1;
  __REG32           : 1;
  __REG32  TBSYNC   : 1;
  __REG32           : 1;
  __REG32  TBWBF    : 1;
  __REG32           :24;
} __tbxcr_bits;

/*TMRB mode register (channels 0 thorough 9)*/
typedef struct {
  __REG32  TBCLK    : 2;
  __REG32  TBCLE    : 1;
  __REG32  TBCPM    : 2;
  __REG32  TBCP     : 1;
  __REG32           :26;
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

/*SSPxCR0 (SSP Control register 0)*/
typedef struct {
  __REG32 DSS     : 4;
  __REG32 FRF     : 2;
  __REG32 SPO     : 1;
  __REG32 SPH     : 1;
  __REG32 SCR     : 8;
  __REG32         :16;
} __sspcr0_bits;

/*SSPxCR1 (SSP Control register 1)*/
typedef struct {
  __REG32 LBM     : 1;
  __REG32 SSE     : 1;
  __REG32 MS      : 1;
  __REG32 SOD     : 1;
  __REG32         :28;
} __sspcr1_bits;

/*SSPxDR (SSP Data register)*/
typedef struct {
  __REG32 DATA    :16;
  __REG32         :16;
} __sspdr_bits;

/*SSPxSR (SSP Status register)*/
typedef struct {
  __REG32 TFE     : 1;
  __REG32 TNF     : 1;
  __REG32 RNE     : 1;
  __REG32 RFF     : 1;
  __REG32 BSY     : 1;
  __REG32         :27;
} __sspsr_bits;

/*SSPxCPSR (SSP Clock prescale register)*/
typedef struct {
  __REG32 CPSDVSR : 8;
  __REG32         :24;
} __sspcpsr_bits;

/*SSPxIMSC (SSP Interrupt mask set and clear register)*/
typedef struct {
  __REG32 RORIM   : 1;
  __REG32 RTIM    : 1;
  __REG32 RXIM    : 1;
  __REG32 TXIM    : 1;
  __REG32         :28;
} __sspimsc_bits;

/*SSPxRIS (SSP Raw interrupt status register)*/
typedef struct {
  __REG32 RORRIS  : 1;
  __REG32 RTRIS   : 1;
  __REG32 RXRIS   : 1;
  __REG32 TXRIS   : 1;
  __REG32         :28;
} __sspris_bits;

/*SSPxMIS (SSP Masked interrupt status register)*/
typedef struct {
  __REG32 RORMIS  : 1;
  __REG32 RTMIS   : 1;
  __REG32 RXMIS   : 1;
  __REG32 TXMIS   : 1;
  __REG32         :28;
} __sspmis_bits;

/*SSPxICR (SSP Interrupt clear register)*/
typedef struct {
  __REG32 RORIC   : 1;
  __REG32 RTIC    : 1;
  __REG32         :30;
} __sspicr_bits;

/*SSPxDMACR (SSP DMA control register*/
typedef struct {
  __REG32 RXDMAE  : 1;
  __REG32 TXDMAE  : 1;
  __REG32         :30;
} __sspdmacr_bits;

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

/*CEC Enable Register*/
typedef struct {
  __REG32 CECEN     : 1;
  __REG32 I2CEC     : 1;
  __REG32           :30;
} __cecen_bits;

/*Logical Address Register*/
typedef struct {
  __REG32 CECADD    :16;
  __REG32           :16;
} __cecadd_bits;

/*Software Reset Register*/
typedef struct {
  __REG32 CECRESET  : 1;
  __REG32           :31;
} __cecreset_bits;

/*Receive Enable Register*/
typedef struct {
  __REG32 CECREN    : 1;
  __REG32           :31;
} __cecren_bits;

/*Receive Buffer Register*/
typedef struct {
  __REG32 CECRBUF   : 8;
  __REG32 CECEOM    : 1;
  __REG32 CECACK    : 1;
  __REG32           :22;
} __cecrbuf_bits;

/*Receive Control Register 1*/
typedef struct {
  __REG32 CECOTH    : 1;
  __REG32 CECRIHLD  : 1;
  __REG32 CECTOUT   : 2;
  __REG32 CECDAT    : 3;
  __REG32           : 1;
  __REG32 CECMAX    : 3;
  __REG32           : 1;
  __REG32 CECMIN    : 3;
  __REG32           : 1;
  __REG32 CECLNC    : 3;
  __REG32           : 1;
  __REG32 CECHNC    : 2;
  __REG32           : 2;
  __REG32 CECACKDIS : 1;
  __REG32           : 7;
} __cecrcr1_bits;

/*Receive Control Register 2*/
typedef struct {
  __REG32 CECSWAV0  : 3;
  __REG32           : 1;
  __REG32 CECSWAV1  : 3;
  __REG32           : 1;
  __REG32 CECSWAV2  : 3;
  __REG32           : 1;
  __REG32 CECSWAV3  : 3;
  __REG32           : 1;
  __REG32           :16;
} __cecrcr2_bits;

/*Receive Control Register 3*/
typedef struct {
  __REG32 CECWAVEN  : 1;
  __REG32 CECRSTAEN : 1;
  __REG32           : 6;
  __REG32 CECWAV0   : 3;
  __REG32           : 1;
  __REG32 CECWAV1   : 3;
  __REG32           : 1;
  __REG32 CECWAV2   : 3;
  __REG32           : 1;
  __REG32 CECWAV3   : 3;
  __REG32           : 1;
  __REG32           : 8;
} __cecrcr3_bits;

/*Transmit Enable Register*/
typedef struct {
  __REG32 CECTEN    : 1;
  __REG32 CECTRANS  : 1;
  __REG32           :30;
} __cecten_bits;

/*Transmit Buffer Register*/
typedef struct {
  __REG32 CECTBUF   : 8;
  __REG32 CECTEOM   : 1;
  __REG32           :23;
} __cectbuf_bits;

/*Transmit Control Register*/
typedef struct {
  __REG32 CECFREE   : 4;
  __REG32 CECBRD    : 1;
  __REG32           : 3;
  __REG32 CECDPRD   : 4;
  __REG32 CECDTRS   : 3;
  __REG32           : 1;
  __REG32 CECSPRD   : 3;
  __REG32           : 1;
  __REG32 CECSTRS   : 3;
  __REG32           : 9;
} __cectcr_bits;

/*Receive Interrupt Status Register*/
typedef struct {
  __REG32 CECRIEND  : 1;
  __REG32 CECRISTA  : 1;
  __REG32 CECRIMAX  : 1;
  __REG32 CECRIMIN  : 1;
  __REG32 CECRIACK  : 1;
  __REG32 CECRIOR   : 1;
  __REG32 CECRIWAV  : 1;
  __REG32           :25;
} __cecrstat_bits;

/*Transmit Interrupt Status Register*/
typedef struct {
  __REG32 CECTISTA  : 1;
  __REG32 CECTIEND  : 1;
  __REG32 CECTIAL   : 1;
  __REG32 CECTIACK  : 1;
  __REG32 CECTIUR   : 1;
  __REG32           :27;
} __cectstat_bits;

/*CEC Source Clock Select Register*/
typedef struct {
  __REG32 CECCLK    : 1;
  __REG32           :31;
} __cecfssel_bits;

/*Remote Control Enable Register*/
typedef struct {
  __REG32 RMCEN     : 1;
  __REG32 I2RMC     : 1;
  __REG32           :30;
} __rmcen_bits;

/*Remote Control Receive Enable Register*/
typedef struct {
  __REG32 RMCREN    : 1;
  __REG32           :31;
} __rmcren_bits;

/*Remote Control Receive Control Register 1*/
typedef struct {
  __REG32 RMCLLMIN  : 8;
  __REG32 RMCLLMAX  : 8;
  __REG32 RMCLCMIN  : 8;
  __REG32 RMCLCMAX  : 8;
} __rmcrcr1_bits;

/*Remote Control Receive Control Register 2*/
typedef struct {
  __REG32 RMCDMAX   : 8;
  __REG32 RMCLL     : 8;
  __REG32           : 8;
  __REG32 RMCPHM    : 1;
  __REG32 RMCLD     : 1;
  __REG32           : 4;
  __REG32 RMCEDIEN  : 1;
  __REG32 RMCLIEN   : 1;
} __rmcrcr2_bits;

/*Remote Control Receive Control Register 3*/
typedef struct {
  __REG32 RMCDATL   : 7;
  __REG32           : 1;
  __REG32 RMCDATH   : 7;
  __REG32           :17;
} __rmcrcr3_bits;

/*Remote Control Receive Control Register 4*/
typedef struct {
  __REG32 RMCNC     : 4;
  __REG32           : 3;
  __REG32 RMCPO     : 1;
  __REG32           :24;
} __rmcrcr4_bits;

/*Remote Control Receive Status Register*/
typedef struct {
  __REG32 RMCRNUM   : 7;
  __REG32 RMCRLDR   : 1;
  __REG32           : 4;
  __REG32 RMCEDIF   : 1;
  __REG32 RMCDMAXIF : 1;
  __REG32 RMCLOIF   : 1;
  __REG32 RMCRLIF   : 1;
  __REG32           :16;
} __rmcrstat_bits;

/*Remote Control Receive End Bit Number Register 1-3*/
typedef struct {
  __REG32 RMCEND    : 7;
  __REG32           :25;
} __rmcend_bits;

/*Remote Control Source Clock selection Register*/
typedef struct {
  __REG32 RMCCLK    : 1;
  __REG32           :31;
} __rmcfssel_bits;

/*A/D Conversion Clock Setting Register*/
typedef struct {
  __REG8  ADCLK   : 3;
  __REG8          : 1;
  __REG8  TSH     : 4;
} __adclk_bits;

/*A/D Mode Control Register 0*/
typedef struct {
  __REG8  ADS     : 1;
  __REG8  SCAN    : 1;
  __REG8  REPEAT  : 1;
  __REG8  ITM     : 2;
  __REG8          : 1;
  __REG8  ADBFN   : 1;
  __REG8  EOCFN   : 1;
} __admod0_bits;

/*A/D Mode Control Register 1*/
typedef struct {
  __REG8  ADCH    : 4;
  __REG8          : 1;
  __REG8  ADSCN   : 1;
  __REG8  I2AD    : 1;
  __REG8  VREFON  : 1;
} __admod1_bits;

/*A/D Mode Control Register 2*/
typedef struct {
  __REG8  HPADCH  : 4;
  __REG8          : 1;
  __REG8  HPADCE  : 1;
  __REG8  ADBFHP  : 1;
  __REG8  EOCFHP  : 1;
} __admod2_bits;

/*A/D Mode Control Register 3*/
typedef struct {
  __REG8  ADOBSV  : 1;
  __REG8  REGS    : 4;
  __REG8  ADOBIC  : 1;
  __REG8          : 2;
} __admod3_bits;

/*A/D Mode Control Register 4*/
typedef struct {
  __REG8  ADRST   : 2;
  __REG8          : 2;
  __REG8  ADHTG   : 1;
  __REG8  ADHS    : 1;
  __REG8  HADHTG  : 1;
  __REG8  HADHS   : 1;
} __admod4_bits;

/*A/D Mode Control Register 5*/
typedef struct {
  __REG8  ADOBSV  : 1;
  __REG8  REGS    : 4;
  __REG8  ADOBIC  : 1;
  __REG8          : 2;
} __admod5_bits;

/*A/D Conversion Result Registers 08 - SP*/
typedef struct {
  __REG16  ADRRF   : 1;
  __REG16  OVR     : 1;
  __REG16          : 4;
  __REG16  ADR     :10;
} __adregx_bits;

/*A/D Conversion Result Comparison Registers*/
typedef struct {
  __REG16          : 6;
  __REG16  ADR     :10;
} __adcmpx_bits;

/*Second column register*/
typedef struct {
  __REG8  SE      : 7;
  __REG8          : 1;
} __secr_bits;

/*Minute column register*/
typedef struct {
  __REG8  MI      : 7;
  __REG8          : 1;
} __minr_bits;

/*Hour column register*/
typedef struct {
  __REG8  HO      : 6;
  __REG8          : 2;
} __hourr_bits;

/*Hour column register*/
typedef struct {
  __REG8  WE      : 3;
  __REG8          : 5;
} __dayr_bits;

/*Day column register*/
typedef struct {
  __REG8  DA      : 6;
  __REG8          : 2;
} __dater_bits;

/*Month column register*/
typedef struct {
  __REG8  MO      : 5;
  __REG8          : 3;
} __monthr_bits;

/*Year column register*/
typedef union {
  __REG8  YE      : 8;
  /*YEARR*/
  struct {
  __REG8  LEAP    : 2;
  __REG8          : 6;
  };
} __yearr_bits;

/*PAGE register */
typedef struct {
  __REG8  PAGE    : 1;
  __REG8          : 1;
  __REG8  ENAALM  : 1;
  __REG8  ENATMR  : 1;
  __REG8  ADJUST  : 1;
  __REG8          : 2;
  __REG8  INTENA  : 1;
} __pager_bits;

/*Reset register*/
typedef struct {
  __REG8          : 4;
  __REG8  RSTALM  : 1;
  __REG8  RSTTMR  : 1;
  __REG8  DIS16HZ : 1;
  __REG8  DIS1HZ  : 1;
} __restr_bits;

/*KWUP Control register n (n=0~3)*/
typedef struct {
  __REG32 KEYnEN  : 1;
  __REG32         : 3;
  __REG32 KEYn    : 3;
  __REG32 DPEn    : 1;
  __REG32         :24;
} __kwupcr_bits;

/*KWUP Port Monitor Register*/
typedef struct {
  __REG32 PKEY0   : 1;
  __REG32 PKEY1   : 1;
  __REG32 PKEY2   : 1;
  __REG32 PKEY3   : 1;
  __REG32         :28;
} __kwupkey_bits;

/*KWUP pull-up cycle register*/
typedef struct {
  __REG32         : 2;
  __REG32 T1S     : 2;
  __REG32 T2S     : 2;
  __REG32         :26;
} __kwupcnt_bits;

/*KWUP All Interrupt request Clear Register*/
typedef struct {
  __REG32 KEYCLR0 : 1;
  __REG32 KEYCLR1 : 1;
  __REG32 KEYCLR2 : 1;
  __REG32 KEYCLR3 : 1;
  __REG32         :28;
} __kwupclr_bits;

/*KWUP Interrupt Monitor Register*/
typedef struct {
  __REG32 KEYINT0 : 1;
  __REG32 KEYINT1 : 1;
  __REG32 KEYINT2 : 1;
  __REG32 KEYINT3 : 1;
  __REG32         :28;
} __kwupint_bits;

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
} __secbit_bits;

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
  __REG32 BLPRO6  : 1;
  __REG32 BLPRO7  : 1;
  __REG32 BLPRO8  : 1;
  __REG32 BLPRO9  : 1;
  __REG32         : 6;
} __flcs_bits;

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

/* Interrupt Set-Enable Registers 64-95 */
typedef struct {
  __REG32  SETENA64       : 1;
  __REG32  SETENA65       : 1;
  __REG32  SETENA66       : 1;
  __REG32  SETENA67       : 1;
  __REG32  SETENA68       : 1;
  __REG32  SETENA69       : 1;
  __REG32  SETENA70       : 1;
  __REG32  SETENA71       : 1;
  __REG32  SETENA72       : 1;
  __REG32  SETENA73       : 1;
  __REG32  SETENA74       : 1;
  __REG32  SETENA75       : 1;
  __REG32  SETENA76       : 1;
  __REG32  SETENA77       : 1;
  __REG32  SETENA78       : 1;
  __REG32  SETENA79       : 1;
  __REG32  SETENA80       : 1;
  __REG32  SETENA81       : 1;
  __REG32  SETENA82       : 1;
  __REG32  SETENA83       : 1;
  __REG32  SETENA84       : 1;
  __REG32  SETENA85       : 1;
  __REG32  SETENA86       : 1;
  __REG32  SETENA87       : 1;
  __REG32  SETENA88       : 1;
  __REG32  SETENA89       : 1;
  __REG32  SETENA90       : 1;
  __REG32  SETENA91       : 1;
  __REG32  SETENA92       : 1;
  __REG32  SETENA93       : 1;
  __REG32  SETENA94       : 1;
  __REG32  SETENA95       : 1;
} __setena2_bits;

/* Interrupt Set-Enable Registers 96-99 */
typedef struct {
  __REG32  SETENA96       : 1;
  __REG32  SETENA97       : 1;
  __REG32  SETENA98       : 1;
  __REG32  SETENA99       : 1;
  __REG32                 :28;
} __setena3_bits;

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

/* Interrupt Clear-Enable Registers 64-95 */
typedef struct {
  __REG32  CLRENA64       : 1;
  __REG32  CLRENA65       : 1;
  __REG32  CLRENA66       : 1;
  __REG32  CLRENA67       : 1;
  __REG32  CLRENA68       : 1;
  __REG32  CLRENA69       : 1;
  __REG32  CLRENA70       : 1;
  __REG32  CLRENA71       : 1;
  __REG32  CLRENA72       : 1;
  __REG32  CLRENA73       : 1;
  __REG32  CLRENA74       : 1;
  __REG32  CLRENA75       : 1;
  __REG32  CLRENA76       : 1;
  __REG32  CLRENA77       : 1;
  __REG32  CLRENA78       : 1;
  __REG32  CLRENA79       : 1;
  __REG32  CLRENA80       : 1;
  __REG32  CLRENA81       : 1;
  __REG32  CLRENA82       : 1;
  __REG32  CLRENA83       : 1;
  __REG32  CLRENA84       : 1;
  __REG32  CLRENA85       : 1;
  __REG32  CLRENA86       : 1;
  __REG32  CLRENA87       : 1;
  __REG32  CLRENA88       : 1;
  __REG32  CLRENA89       : 1;
  __REG32  CLRENA90       : 1;
  __REG32  CLRENA91       : 1;
  __REG32  CLRENA92       : 1;
  __REG32  CLRENA93       : 1;
  __REG32  CLRENA94       : 1;
  __REG32  CLRENA95       : 1;
} __clrena2_bits;

/* Interrupt Clear-Enable Registers 96-99 */
typedef struct {
  __REG32  CLRENA96       : 1;
  __REG32  CLRENA97       : 1;
  __REG32  CLRENA98       : 1;
  __REG32  CLRENA99       : 1;
  __REG32                 :28;
} __clrena3_bits;

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

/* Interrupt Set-Pending Register 64-95 */
typedef struct {
  __REG32  SETPEND64      : 1;
  __REG32  SETPEND65      : 1;
  __REG32  SETPEND66      : 1;
  __REG32  SETPEND67      : 1;
  __REG32  SETPEND68      : 1;
  __REG32  SETPEND69      : 1;
  __REG32  SETPEND70      : 1;
  __REG32  SETPEND71      : 1;
  __REG32  SETPEND72      : 1;
  __REG32  SETPEND73      : 1;
  __REG32  SETPEND74      : 1;
  __REG32  SETPEND75      : 1;
  __REG32  SETPEND76      : 1;
  __REG32  SETPEND77      : 1;
  __REG32  SETPEND78      : 1;
  __REG32  SETPEND79      : 1;
  __REG32  SETPEND80      : 1;
  __REG32  SETPEND81      : 1;
  __REG32  SETPEND82      : 1;
  __REG32  SETPEND83      : 1;
  __REG32  SETPEND84      : 1;
  __REG32  SETPEND85      : 1;
  __REG32  SETPEND86      : 1;
  __REG32  SETPEND87      : 1;
  __REG32  SETPEND88      : 1;
  __REG32  SETPEND89      : 1;
  __REG32  SETPEND90      : 1;
  __REG32  SETPEND91      : 1;
  __REG32  SETPEND92      : 1;
  __REG32  SETPEND93      : 1;
  __REG32  SETPEND94      : 1;
  __REG32  SETPEND95      : 1;
} __setpend2_bits;

/* Interrupt Set-Pending Register 96-99 */
typedef struct {
  __REG32  SETPEND96      : 1;
  __REG32  SETPEND97      : 1;
  __REG32  SETPEND98      : 1;
  __REG32  SETPEND99      : 1;
  __REG32                 :28;
} __setpend3_bits;

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

/* Interrupt Clear-Pending Register 64-95 */
typedef struct {
  __REG32  CLRPEND64      : 1;
  __REG32  CLRPEND65      : 1;
  __REG32  CLRPEND66      : 1;
  __REG32  CLRPEND67      : 1;
  __REG32  CLRPEND68      : 1;
  __REG32  CLRPEND69      : 1;
  __REG32  CLRPEND70      : 1;
  __REG32  CLRPEND71      : 1;
  __REG32  CLRPEND72      : 1;
  __REG32  CLRPEND73      : 1;
  __REG32  CLRPEND74      : 1;
  __REG32  CLRPEND75      : 1;
  __REG32  CLRPEND76      : 1;
  __REG32  CLRPEND77      : 1;
  __REG32  CLRPEND78      : 1;
  __REG32  CLRPEND79      : 1;
  __REG32  CLRPEND80      : 1;
  __REG32  CLRPEN881      : 1;
  __REG32  CLRPEND82      : 1;
  __REG32  CLRPEND83      : 1;
  __REG32  CLRPEND84      : 1;
  __REG32  CLRPEND85      : 1;
  __REG32  CLRPEND86      : 1;
  __REG32  CLRPEND87      : 1;
  __REG32  CLRPEND88      : 1;
  __REG32  CLRPEND89      : 1;
  __REG32  CLRPEND90      : 1;
  __REG32  CLRPEND91      : 1;
  __REG32  CLRPEND92      : 1;
  __REG32  CLRPEND93      : 1;
  __REG32  CLRPEND94      : 1;
  __REG32  CLRPEND95      : 1;
} __clrpend2_bits;

/* Interrupt Clear-Pending Register 96-99 */
typedef struct {
  __REG32  CLRPEND96      : 1;
  __REG32  CLRPEND97      : 1;
  __REG32  CLRPEND98      : 1;
  __REG32  CLRPEND99      : 1;
  __REG32                 :28;
} __clrpend3_bits;

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

/* Interrupt Priority Registers 64-67 */
typedef struct {
  __REG32  PRI_64         : 8;
  __REG32  PRI_65         : 8;
  __REG32  PRI_66         : 8;
  __REG32  PRI_67         : 8;
} __pri16_bits;

/* Interrupt Priority Registers 68-71 */
typedef struct {
  __REG32  PRI_68         : 8;
  __REG32  PRI_69         : 8;
  __REG32  PRI_70         : 8;
  __REG32  PRI_71         : 8;
} __pri17_bits;

/* Interrupt Priority Registers 72-75 */
typedef struct {
  __REG32  PRI_72         : 8;
  __REG32  PRI_73         : 8;
  __REG32  PRI_74         : 8;
  __REG32  PRI_75         : 8;
} __pri18_bits;

/* Interrupt Priority Registers 76-79 */
typedef struct {
  __REG32  PRI_76         : 8;
  __REG32  PRI_77         : 8;
  __REG32  PRI_78         : 8;
  __REG32  PRI_79         : 8;
} __pri19_bits;

/* Interrupt Priority Registers 80-83 */
typedef struct {
  __REG32  PRI_80         : 8;
  __REG32  PRI_81         : 8;
  __REG32  PRI_82         : 8;
  __REG32  PRI_83         : 8;
} __pri20_bits;

/* Interrupt Priority Registers 84-87 */
typedef struct {
  __REG32  PRI_84         : 8;
  __REG32  PRI_85         : 8;
  __REG32  PRI_86         : 8;
  __REG32  PRI_87         : 8;
} __pri21_bits;

/* Interrupt Priority Registers 88-91 */
typedef struct {
  __REG32  PRI_88         : 8;
  __REG32  PRI_89         : 8;
  __REG32  PRI_70         : 8;
  __REG32  PRI_71         : 8;
} __pri22_bits;

/* Interrupt Priority Registers 92-95 */
typedef struct {
  __REG32  PRI_92         : 8;
  __REG32  PRI_93         : 8;
  __REG32  PRI_94         : 8;
  __REG32  PRI_95         : 8;
} __pri23_bits;

/* Interrupt Priority Registers 96-99 */
typedef struct {
  __REG32  PRI_96         : 8;
  __REG32  PRI_97         : 8;
  __REG32  PRI_98         : 8;
  __REG32  PRI_99         : 8;
} __pri24_bits;

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
__IO_REG32_BIT(SETENA2,             0xE000E108,__READ_WRITE ,__setena2_bits);
__IO_REG32_BIT(SETENA3,             0xE000E10C,__READ_WRITE ,__setena3_bits);
__IO_REG32_BIT(CLRENA0,             0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,             0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(CLRENA2,             0xE000E188,__READ_WRITE ,__clrena2_bits);
__IO_REG32_BIT(CLRENA3,             0xE000E18C,__READ_WRITE ,__clrena3_bits);
__IO_REG32_BIT(SETPEND0,            0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,            0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(SETPEND2,            0xE000E208,__READ_WRITE ,__setpend2_bits);
__IO_REG32_BIT(SETPEND3,            0xE000E20C,__READ_WRITE ,__setpend3_bits);
__IO_REG32_BIT(CLRPEND0,            0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,            0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(CLRPEND2,            0xE000E288,__READ_WRITE ,__clrpend2_bits);
__IO_REG32_BIT(CLRPEND3,            0xE000E28C,__READ_WRITE ,__clrpend3_bits);
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
__IO_REG32_BIT(IP16,                0xE000E440,__READ_WRITE ,__pri16_bits);
__IO_REG32_BIT(IP17,                0xE000E444,__READ_WRITE ,__pri17_bits);
__IO_REG32_BIT(IP18,                0xE000E448,__READ_WRITE ,__pri18_bits);
__IO_REG32_BIT(IP19,                0xE000E44C,__READ_WRITE ,__pri19_bits);
__IO_REG32_BIT(IP20,                0xE000E450,__READ_WRITE ,__pri20_bits);
__IO_REG32_BIT(IP21,                0xE000E454,__READ_WRITE ,__pri21_bits);
__IO_REG32_BIT(IP22,                0xE000E458,__READ_WRITE ,__pri22_bits);
__IO_REG32_BIT(IP23,                0xE000E45C,__READ_WRITE ,__pri23_bits);
__IO_REG32_BIT(IP24,                0xE000E460,__READ_WRITE ,__pri24_bits);
__IO_REG32_BIT(VTOR,                0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,               0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SHPR0,               0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,               0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,               0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,               0xE000ED24,__READ_WRITE ,__shcsr_bits);

/***************************************************************************
 **
 ** CG (Clcok generator)
 **
 ***************************************************************************/
__IO_REG32_BIT(CGSYSCR,             0x400F4000,__READ_WRITE ,__cgsyscr_bits);
__IO_REG32_BIT(CGOSCCR,             0x400F4004,__READ_WRITE ,__cgosccr_bits);
__IO_REG32_BIT(CGSTBYCR,            0x400F4008,__READ_WRITE ,__cgstbycr_bits);
__IO_REG32_BIT(CGPLLSEL,            0x400F400C,__READ_WRITE ,__cgpllsel_bits);
__IO_REG32_BIT(CGCKSEL,             0x400F4010,__READ_WRITE ,__cgcksel_bits);
__IO_REG32_BIT(CGICRCG,             0x400F4014,__WRITE      ,__cgicrcg_bits);
__IO_REG32_BIT(CGNMIFLG,            0x400F4018,__READ       ,__cgnmiflg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x400F401C,__READ_WRITE ,__cgrstflg_bits);
__IO_REG32_BIT(CGIMCGA,             0x400F4020,__READ_WRITE ,__cgimcga_bits);
__IO_REG32_BIT(CGIMCGB,             0x400F4024,__READ_WRITE ,__cgimcgb_bits);
__IO_REG32_BIT(CGIMCGC,             0x400F4028,__READ_WRITE ,__cgimcgc_bits);
__IO_REG32_BIT(CGIMCGD,             0x400F402C,__READ_WRITE ,__cgimcgd_bits);
__IO_REG32_BIT(CGIMCGE,             0x400F4030,__READ_WRITE ,__cgimcge_bits);
__IO_REG32_BIT(CGIMCGF,             0x400F4034,__READ_WRITE ,__cgimcgf_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x400C0000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x400C0004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAFR1,                0x400C0008,__READ_WRITE ,__pafr1_bits);
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
__IO_REG8_BIT(PBFR1,                0x400C0108,__READ_WRITE ,__pbfr1_bits);
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
__IO_REG8_BIT(PCFR2,                0x400C020C,__READ_WRITE ,__pcfr2_bits);
__IO_REG8_BIT(PCFR3,                0x400C0210,__READ_WRITE ,__pcfr3_bits);
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
__IO_REG8_BIT(PDFR1,                0x400C0308,__READ_WRITE ,__pdfr1_bits);
__IO_REG8_BIT(PDFR2,                0x400C030C,__READ_WRITE ,__pdfr2_bits);
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
__IO_REG8_BIT(PEFR2,                0x400C040C,__READ_WRITE ,__pefr2_bits);
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
__IO_REG8_BIT(PFFR1,                0x400C0508,__READ_WRITE ,__pffr1_bits);
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
__IO_REG8_BIT(PGFR2,                0x400C060C,__READ_WRITE ,__pgfr2_bits);
__IO_REG8_BIT(PGFR3,                0x400C0610,__READ_WRITE ,__pgfr3_bits);
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
__IO_REG8_BIT(PHFR2,                0x400C070C,__READ_WRITE ,__phfr2_bits);
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
__IO_REG8_BIT(PIIE,                 0x400C0838,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x400C0900,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJFR2,                0x400C090C,__READ_WRITE ,__pjfr2_bits);
__IO_REG8_BIT(PJPUP,                0x400C092C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJIE,                 0x400C0938,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG8_BIT(PKDATA,               0x400C0A00,__READ_WRITE ,__pk_bits);
__IO_REG8_BIT(PKPUP,                0x400C0A2C,__READ_WRITE ,__pkpup_bits);
__IO_REG8_BIT(PKIE,                 0x400C0A38,__READ_WRITE ,__pkie_bits);

/***************************************************************************
 **
 ** PORTL
 **
 ***************************************************************************/
__IO_REG8_BIT(PLDATA,               0x400C0B00,__READ_WRITE ,__pl_bits);
__IO_REG8_BIT(PLCR,                 0x400C0B04,__READ_WRITE ,__plcr_bits);
__IO_REG8_BIT(PLFR1,                0x400C0B08,__READ_WRITE ,__plfr1_bits);
__IO_REG8_BIT(PLFR2,                0x400C0B0C,__READ_WRITE ,__plfr2_bits);
__IO_REG8_BIT(PLFR3,                0x400C0B10,__READ_WRITE ,__plfr3_bits);
__IO_REG8_BIT(PLOD,                 0x400C0B28,__READ_WRITE ,__plod_bits);
__IO_REG8_BIT(PLPUP,                0x400C0B2C,__READ_WRITE ,__plpup_bits);
__IO_REG8_BIT(PLIE,                 0x400C0B38,__READ_WRITE ,__plie_bits);

/***************************************************************************
 **
 ** PORTM
 **
 ***************************************************************************/
__IO_REG8_BIT(PMDATA,               0x400C0C00,__READ_WRITE ,__pm_bits);
__IO_REG8_BIT(PMCR,                 0x400C0C04,__READ_WRITE ,__pmcr_bits);
__IO_REG8_BIT(PMFR1,                0x400C0C08,__READ_WRITE ,__pmfr1_bits);
__IO_REG8_BIT(PMFR2,                0x400C0C0C,__READ_WRITE ,__pmfr2_bits);
__IO_REG8_BIT(PMFR3,                0x400C0C10,__READ_WRITE ,__pmfr3_bits);
__IO_REG8_BIT(PMOD,                 0x400C0C28,__READ_WRITE ,__pmod_bits);
__IO_REG8_BIT(PMPUP,                0x400C0C2C,__READ_WRITE ,__pmpup_bits);
__IO_REG8_BIT(PMIE,                 0x400C0C38,__READ_WRITE ,__pmie_bits);

/***************************************************************************
 **
 ** PORTN
 **
 ***************************************************************************/
__IO_REG8_BIT(PNDATA,               0x400C0D00,__READ_WRITE ,__pn_bits);
__IO_REG8_BIT(PNCR,                 0x400C0D04,__READ_WRITE ,__pncr_bits);
__IO_REG8_BIT(PNFR1,                0x400C0D08,__READ_WRITE ,__pnfr1_bits);
__IO_REG8_BIT(PNFR2,                0x400C0D0C,__READ_WRITE ,__pnfr2_bits);
__IO_REG8_BIT(PNFR3,                0x400C0D10,__READ_WRITE ,__pnfr3_bits);
__IO_REG8_BIT(PNOD,                 0x400C0D28,__READ_WRITE ,__pnod_bits);
__IO_REG8_BIT(PNPUP,                0x400C0D2C,__READ_WRITE ,__pnpup_bits);
__IO_REG8_BIT(PNIE,                 0x400C0D38,__READ_WRITE ,__pnie_bits);

/***************************************************************************
 **
 ** PORTO
 **
 ***************************************************************************/
__IO_REG8_BIT(PODATA,               0x400C0E00,__READ_WRITE ,__po_bits);
__IO_REG8_BIT(POCR,                 0x400C0E04,__READ_WRITE ,__pocr_bits);
__IO_REG8_BIT(POFR1,                0x400C0E08,__READ_WRITE ,__pofr1_bits);
__IO_REG8_BIT(POFR2,                0x400C0E0C,__READ_WRITE ,__pofr2_bits);
__IO_REG8_BIT(POFR3,                0x400C0E10,__READ_WRITE ,__pofr3_bits);
__IO_REG8_BIT(POOD,                 0x400C0E28,__READ_WRITE ,__pood_bits);
__IO_REG8_BIT(POPUP,                0x400C0E2C,__READ_WRITE ,__popup_bits);
__IO_REG8_BIT(POIE,                 0x400C0E38,__READ_WRITE ,__poie_bits);

/***************************************************************************
 **
 ** PORTP
 **
 ***************************************************************************/
__IO_REG8_BIT(PPDATA,               0x400C0F00,__READ_WRITE ,__pp_bits);
__IO_REG8_BIT(PPCR,                 0x400C0F04,__READ_WRITE ,__ppcr_bits);
__IO_REG8_BIT(PPFR1,                0x400C0F08,__READ_WRITE ,__ppfr1_bits);
__IO_REG8_BIT(PPFR2,                0x400C0F0C,__READ_WRITE ,__ppfr2_bits);
__IO_REG8_BIT(PPOD,                 0x400C0F28,__READ_WRITE ,__ppod_bits);
__IO_REG8_BIT(PPPUP,                0x400C0F2C,__READ_WRITE ,__pppup_bits);
__IO_REG8_BIT(PPIE,                 0x400C0F38,__READ_WRITE ,__ppie_bits);

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACIntStaus,          0x40000000,__READ       ,__dmacintstaus_bits);
__IO_REG32_BIT(DMACIntTCStatus,       0x40000004,__READ       ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACIntTCClear,        0x40000008,__WRITE      ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACIntErrorStatus,    0x4000000C,__READ       ,__dmacinterrorstatus_bits);
__IO_REG32_BIT(DMACIntErrClr,         0x40000010,__WRITE      ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRawIntTCStatus,    0x40000014,__READ       ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRawIntErrorStatus, 0x40000018,__READ       ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACEnbldChns,         0x4000001C,__READ       ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSoftBReq,          0x40000020,__READ_WRITE ,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSoftSReq,          0x40000024,__READ_WRITE ,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACConfiguration,     0x40000030,__READ_WRITE ,__dmacconfiguration_bits);
__IO_REG32(    DMACC0SrcAddr,         0x40000100,__READ_WRITE );
__IO_REG32(    DMACC0DestAddr,        0x40000104,__READ_WRITE );
__IO_REG32(    DMACC0LLI,             0x40000108,__READ_WRITE );
__IO_REG32_BIT(DMACC0Control,         0x4000010C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC0Configuration,   0x40000110,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC1SrcAddr,         0x40000120,__READ_WRITE );
__IO_REG32(    DMACC1DestAddr,        0x40000124,__READ_WRITE );
__IO_REG32(    DMACC1LLI,             0x40000128,__READ_WRITE );
__IO_REG32_BIT(DMACC1Control,         0x4000012C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC1Configuration,   0x40000130,__READ_WRITE ,__dmaccconfiguration_bits);

/***************************************************************************
 **
 ** SMC (Static Memory Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(smc_memif_cfg,         0x40001004,__READ       ,__smc_memif_cfg_bits);
__IO_REG32_BIT(smc_direct_cmd,        0x40001010,__WRITE      ,__smc_direct_cmd_bits);
__IO_REG32_BIT(smc_set_cycles,        0x40001014,__WRITE      ,__smc_set_cycles_bits);
__IO_REG32_BIT(smc_set_opmode,        0x40001018,__WRITE      ,__smc_set_opmode_bits);
__IO_REG32_BIT(smc_sram_cycles0_0,    0x40001100,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_opmode0_0,         0x40001104,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_sram_cycles0_1,    0x40001120,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_opmode0_1,         0x40001124,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_sram_cycles0_2,    0x40001140,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_opmode0_2,         0x40001144,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_sram_cycles0_3,    0x40001160,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_opmode0_3,         0x40001164,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(SMCMDMODE,             0x41FFF100,__READ_WRITE ,__smcmdmode_bits);

/***************************************************************************
 **
 ** TMRB0
 **
 ***************************************************************************/
__IO_REG32_BIT(TB0EN,               0x400D0000, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB0RUN,              0x400D0004, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB0CR,               0x400D0008, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB0MOD,              0x400D000C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB0FFCR,             0x400D0010, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB0ST,               0x400D0014, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB0IM,               0x400D0018, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB0UC,               0x400D001C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB0RG0,              0x400D0020, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB0RG1,              0x400D0024, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB0CP0,              0x400D0028, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB0CP1,              0x400D002C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB1
 **
 ***************************************************************************/
__IO_REG32_BIT(TB1EN,               0x400D0100, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB1RUN,              0x400D0104, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB1CR,               0x400D0108, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB1MOD,              0x400D010C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB1FFCR,             0x400D0110, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB1ST,               0x400D0114, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB1IM,               0x400D0118, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB1UC,               0x400D011C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB1RG0,              0x400D0120, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB1RG1,              0x400D0124, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB1CP0,              0x400D0128, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB1CP1,              0x400D012C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB2
 **
 ***************************************************************************/
__IO_REG32_BIT(TB2EN,               0x400D0200, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB2RUN,              0x400D0204, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB2CR,               0x400D0208, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB2MOD,              0x400D020C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB2FFCR,             0x400D0210, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB2ST,               0x400D0214, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB2IM,               0x400D0218, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB2UC,               0x400D021C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB2RG0,              0x400D0220, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB2RG1,              0x400D0224, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB2CP0,              0x400D0228, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB2CP1,              0x400D022C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB3
 **
 ***************************************************************************/
__IO_REG32_BIT(TB3EN,               0x400D0300, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB3RUN,              0x400D0304, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB3CR,               0x400D0308, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB3MOD,              0x400D030C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB3FFCR,             0x400D0310, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB3ST,               0x400D0314, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB3IM,               0x400D0318, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB3UC,               0x400D031C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB3RG0,              0x400D0320, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB3RG1,              0x400D0324, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB3CP0,              0x400D0328, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB3CP1,              0x400D032C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB4
 **
 ***************************************************************************/
__IO_REG32_BIT(TB4EN,               0x400D0400, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB4RUN,              0x400D0404, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB4CR,               0x400D0408, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB4MOD,              0x400D040C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB4FFCR,             0x400D0410, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB4ST,               0x400D0414, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB4IM,               0x400D0418, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB4UC,               0x400D041C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB4RG0,              0x400D0420, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB4RG1,              0x400D0424, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB4CP0,              0x400D0428, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB4CP1,              0x400D042C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB5
 **
 ***************************************************************************/
__IO_REG32_BIT(TB5EN,               0x400D0500, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB5RUN,              0x400D0504, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB5CR,               0x400D0508, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB5MOD,              0x400D050C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB5FFCR,             0x400D0510, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB5ST,               0x400D0514, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB5IM,               0x400D0518, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB5UC,               0x400D051C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB5RG0,              0x400D0520, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB5RG1,              0x400D0524, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB5CP0,              0x400D0528, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB5CP1,              0x400D052C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB6
 **
 ***************************************************************************/
__IO_REG32_BIT(TB6EN,               0x400D0600, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB6RUN,              0x400D0604, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB6CR,               0x400D0608, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB6MOD,              0x400D060C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB6FFCR,             0x400D0610, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB6ST,               0x400D0614, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB6IM,               0x400D0618, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB6UC,               0x400D061C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB6RG0,              0x400D0620, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB6RG1,              0x400D0624, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB6CP0,              0x400D0628, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB6CP1,              0x400D062C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB7
 **
 ***************************************************************************/
__IO_REG32_BIT(TB7EN,               0x400D0700, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB7RUN,              0x400D0704, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB7CR,               0x400D0708, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB7MOD,              0x400D070C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB7FFCR,             0x400D0710, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB7ST,               0x400D0714, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB7IM,               0x400D0718, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB7UC,               0x400D071C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB7RG0,              0x400D0720, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB7RG1,              0x400D0724, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB7CP0,              0x400D0728, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB7CP1,              0x400D072C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB8
 **
 ***************************************************************************/
__IO_REG32_BIT(TB8EN,               0x400D0800, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB8RUN,              0x400D0804, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB8CR,               0x400D0808, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB8MOD,              0x400D080C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB8FFCR,             0x400D0810, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB8ST,               0x400D0814, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB8IM,               0x400D0818, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB8UC,               0x400D081C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB8RG0,              0x400D0820, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB8RG1,              0x400D0824, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB8CP0,              0x400D0828, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB8CP1,              0x400D082C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB9
 **
 ***************************************************************************/
__IO_REG32_BIT(TB9EN,               0x400D0900, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB9RUN,              0x400D0904, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB9CR,               0x400D0908, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB9MOD,              0x400D090C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB9FFCR,             0x400D0910, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB9ST,               0x400D0914, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB9IM,               0x400D0918, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB9UC,               0x400D091C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB9RG0,              0x400D0920, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB9RG1,              0x400D0924, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB9CP0,              0x400D0928, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB9CP1,              0x400D092C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRBA
 **
 ***************************************************************************/
__IO_REG32_BIT(TBAEN,               0x400D0A00, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TBARUN,              0x400D0A04, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TBACR,               0x400D0A08, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TBAMOD,              0x400D0A0C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TBAFFCR,             0x400D0A10, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TBAST,               0x400D0A14, __READ       , __tbxst_bits);
__IO_REG32_BIT(TBAIM,               0x400D0A18, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TBAUC,               0x400D0A1C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TBARG0,              0x400D0A20, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TBARG1,              0x400D0A24, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TBACP0,              0x400D0A28, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TBACP1,              0x400D0A2C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRBB
 **
 ***************************************************************************/
__IO_REG32_BIT(TBBEN,               0x400D0B00, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TBBRUN,              0x400D0B04, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TBBCR,               0x400D0B08, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TBBMOD,              0x400D0B0C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TBBFFCR,             0x400D0B10, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TBBST,               0x400D0B14, __READ       , __tbxst_bits);
__IO_REG32_BIT(TBBIM,               0x400D0B18, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TBBUC,               0x400D0B1C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TBBRG0,              0x400D0B20, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TBBRG1,              0x400D0B24, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TBBCP0,              0x400D0B28, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TBBCP1,              0x400D0B2C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRBC
 **
 ***************************************************************************/
__IO_REG32_BIT(TBCEN,               0x400D0C00, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TBCRUN,              0x400D0C04, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TBCCR,               0x400D0C08, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TBCMOD,              0x400D0C0C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TBCFFCR,             0x400D0C10, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TBCST,               0x400D0C14, __READ       , __tbxst_bits);
__IO_REG32_BIT(TBCIM,               0x400D0C18, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TBCUC,               0x400D0C1C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TBCRG0,              0x400D0C20, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TBCRG1,              0x400D0C24, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TBCCP0,              0x400D0C28, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TBCCP1,              0x400D0C2C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRBD
 **
 ***************************************************************************/
__IO_REG32_BIT(TBDEN,               0x400D0D00, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TBDRUN,              0x400D0D04, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TBDCR,               0x400D0D08, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TBDMOD,              0x400D0D0C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TBDFFCR,             0x400D0D10, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TBDST,               0x400D0D14, __READ       , __tbxst_bits);
__IO_REG32_BIT(TBDIM,               0x400D0D18, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TBDUC,               0x400D0D1C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TBDRG0,              0x400D0D20, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TBDRG1,              0x400D0D24, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TBDCP0,              0x400D0D28, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TBDCP1,              0x400D0D2C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRBE
 **
 ***************************************************************************/
__IO_REG32_BIT(TBEEN,               0x400D0E00, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TBERUN,              0x400D0E04, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TBECR,               0x400D0E08, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TBEMOD,              0x400D0E0C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TBEFFCR,             0x400D0E10, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TBEST,               0x400D0E14, __READ       , __tbxst_bits);
__IO_REG32_BIT(TBEIM,               0x400D0E18, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TBEUC,               0x400D0E1C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TBERG0,              0x400D0E20, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TBERG1,              0x400D0E24, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TBECP0,              0x400D0E28, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TBECP1,              0x400D0E2C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRBF
 **
 ***************************************************************************/
__IO_REG32_BIT(TBFEN,               0x400D0F00, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TBFRUN,              0x400D0F04, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TBFCR,               0x400D0F08, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TBFMOD,              0x400D0F0C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TBFFFCR,             0x400D0F10, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TBFST,               0x400D0F14, __READ       , __tbxst_bits);
__IO_REG32_BIT(TBFIM,               0x400D0F18, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TBFUC,               0x400D0F1C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TBFRG0,              0x400D0F20, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TBFRG1,              0x400D0F24, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TBFCP0,              0x400D0F28, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TBFCP1,              0x400D0F2C, __READ_WRITE , __tbxcp1_bits);

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

/***************************************************************************
 **
 ** SIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(SC2EN,               0x400E1200, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC2BUF,              0x400E1204, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC2CR,               0x400E1208, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC2MOD0,             0x400E120C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC2BRCR,             0x400E1210, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC2BRADD,            0x400E1214, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC2MOD1,             0x400E1218, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC2MOD2,             0x400E121C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC2RFC,              0x400E1220, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC2TFC,              0x400E1224, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC2RST,              0x400E1228, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC2TST,              0x400E122C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC2FCNF,             0x400E1230, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(SC3EN,               0x400E1300, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC3BUF,              0x400E1304, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC3CR,               0x400E1308, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC3MOD0,             0x400E130C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC3BRCR,             0x400E1310, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC3BRADD,            0x400E1314, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC3MOD1,             0x400E1318, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC3MOD2,             0x400E131C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC3RFC,              0x400E1320, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC3TFC,              0x400E1324, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC3RST,              0x400E1328, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC3TST,              0x400E132C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC3FCNF,             0x400E1330, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO4
 **
 ***************************************************************************/
__IO_REG32_BIT(SC4EN,               0x400E1400, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC4BUF,              0x400E1404, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC4CR,               0x400E1408, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC4MOD0,             0x400E140C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC4BRCR,             0x400E1410, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC4BRADD,            0x400E1414, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC4MOD1,             0x400E1418, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC4MOD2,             0x400E141C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC4RFC,              0x400E1420, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC4TFC,              0x400E1424, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC4RST,              0x400E1428, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC4TST,              0x400E142C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC4FCNF,             0x400E1430, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO5
 **
 ***************************************************************************/
__IO_REG32_BIT(SC5EN,               0x400E1500, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC5BUF,              0x400E1504, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC5CR,               0x400E1508, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC5MOD0,             0x400E150C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC5BRCR,             0x400E1510, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC5BRADD,            0x400E1514, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC5MOD1,             0x400E1518, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC5MOD2,             0x400E151C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC5RFC,              0x400E1520, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC5TFC,              0x400E1524, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC5RST,              0x400E1528, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC5TST,              0x400E152C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC5FCNF,             0x400E1530, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO6
 **
 ***************************************************************************/
__IO_REG32_BIT(SC6EN,               0x400E1600, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC6BUF,              0x400E1604, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC6CR,               0x400E1608, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC6MOD0,             0x400E160C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC6BRCR,             0x400E1610, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC6BRADD,            0x400E1614, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC6MOD1,             0x400E1618, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC6MOD2,             0x400E161C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC6RFC,              0x400E1620, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC6TFC,              0x400E1624, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC6RST,              0x400E1628, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC6TST,              0x400E162C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC6FCNF,             0x400E1630, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO7
 **
 ***************************************************************************/
__IO_REG32_BIT(SC7EN,               0x400E1700, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC7BUF,              0x400E1704, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC7CR,               0x400E1708, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC7MOD0,             0x400E170C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC7BRCR,             0x400E1710, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC7BRADD,            0x400E1714, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC7MOD1,             0x400E1718, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC7MOD2,             0x400E171C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC7RFC,              0x400E1720, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC7TFC,              0x400E1724, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC7RST,              0x400E1728, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC7TST,              0x400E172C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC7FCNF,             0x400E1730, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO8
 **
 ***************************************************************************/
__IO_REG32_BIT(SC8EN,               0x400E1800, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC8BUF,              0x400E1804, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC8CR,               0x400E1808, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC8MOD0,             0x400E180C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC8BRCR,             0x400E1810, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC8BRADD,            0x400E1814, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC8MOD1,             0x400E1818, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC8MOD2,             0x400E181C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC8RFC,              0x400E1820, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC8TFC,              0x400E1824, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC8RST,              0x400E1828, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC8TST,              0x400E182C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC8FCNF,             0x400E1830, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO9
 **
 ***************************************************************************/
__IO_REG32_BIT(SC9EN,               0x400E1900, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC9BUF,              0x400E1904, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC9CR,               0x400E1908, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC9MOD0,             0x400E190C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC9BRCR,             0x400E1910, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC9BRADD,            0x400E1914, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC9MOD1,             0x400E1918, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC9MOD2,             0x400E191C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC9RFC,              0x400E1920, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC9TFC,              0x400E1924, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC9RST,              0x400E1928, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC9TST,              0x400E192C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC9FCNF,             0x400E1930, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO10
 **
 ***************************************************************************/
__IO_REG32_BIT(SC10EN,              0x400E1A00, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC10BUF,             0x400E1A04, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC10CR,              0x400E1A08, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC10MOD0,            0x400E1A0C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC10BRCR,            0x400E1A10, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC10BRADD,           0x400E1A14, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC10MOD1,            0x400E1A18, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC10MOD2,            0x400E1A1C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC10RFC,             0x400E1A20, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC10TFC,             0x400E1A24, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC10RST,             0x400E1A28, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC10TST,             0x400E1A2C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC10FCNF,            0x400E1A30, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO11
 **
 ***************************************************************************/
__IO_REG32_BIT(SC11EN,              0x400E1B00, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC11BUF,             0x400E1B04, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC11CR,              0x400E1B08, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC11MOD0,            0x400E1B0C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC11BRCR,            0x400E1B10, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC11BRADD,           0x400E1B14, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC11MOD1,            0x400E1B18, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC11MOD2,            0x400E1B1C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC11RFC,             0x400E1B20, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC11TFC,             0x400E1B24, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC11RST,             0x400E1B28, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC11TST,             0x400E1B2C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC11FCNF,            0x400E1B30, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SSP
 **
 ***************************************************************************/
__IO_REG32_BIT(SSPCR0,              0x40040000, __READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSPCR1,              0x40040004, __READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSPDR,               0x40040008, __READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSPSR,               0x4004000C, __READ       , __sspsr_bits);
__IO_REG32_BIT(SSPCPSR,             0x40040010, __READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSPIMSC,             0x40040014, __READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSPRIS,              0x40040018, __READ       , __sspris_bits);
__IO_REG32_BIT(SSPMIS,              0x4004001C, __READ       , __sspmis_bits);
__IO_REG32_BIT(SSPICR,              0x40040020, __WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSPDMACR,            0x40040024, __READ_WRITE , __sspdmacr_bits);

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
 ** SBI2
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C2CR0,             0x400E0200, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C2CR1,             0x400E0204, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C2DBR,             0x400E0208, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C2AR,              0x400E020C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C2CR2,             0x400E0210, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C2SR      I2C2CR2         
#define I2C2SR_bit  I2C2CR2_bit.__sr
__IO_REG32_BIT(I2C2BR0,             0x400E0214, __READ_WRITE , __sbixbr0_bits);

#define SIO2CR0     I2C2CR0
#define SIO2CR0_bit I2C2CR0_bit
#define SIO2CR1     I2C2CR1
#define SIO2CR1_bit I2C2CR1_bit.__sio
#define SIO2DBR     I2C2DBR
#define SIO2DBR_bit I2C2DBR_bit
#define SIO2CR2     I2C2CR2
#define SIO2CR2_bit I2C2CR2_bit.__sio
#define SIO2SR      I2C2CR2
#define SIO2SR_bit  I2C2CR2_bit.__sio.__sr
#define SIO2BR0     I2C2BR0
#define SIO2BR0_bit I2C2BR0_bit

/***************************************************************************
 **
 ** SBI3
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C3CR0,             0x400E0300, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C3CR1,             0x400E0304, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C3DBR,             0x400E0308, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C3AR,              0x400E030C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C3CR2,             0x400E0310, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C3SR      I2C3CR2         
#define I2C3SR_bit  I2C3CR2_bit.__sr
__IO_REG32_BIT(I2C3BR0,             0x400E0314, __READ_WRITE , __sbixbr0_bits);

#define SIO3CR0     I2C3CR0
#define SIO3CR0_bit I2C3CR0_bit
#define SIO3CR1     I2C3CR1
#define SIO3CR1_bit I2C3CR1_bit.__sio
#define SIO3DBR     I2C3DBR
#define SIO3DBR_bit I2C3DBR_bit
#define SIO3CR2     I2C3CR2
#define SIO3CR2_bit I2C3CR2_bit.__sio
#define SIO3SR      I2C3CR2
#define SIO3SR_bit  I2C3CR2_bit.__sio.__sr
#define SIO3BR0     I2C3BR0
#define SIO3BR0_bit I2C3BR0_bit

/***************************************************************************
 **
 ** SBI4
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C4CR0,             0x400E0400, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C4CR1,             0x400E0404, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C4DBR,             0x400E0408, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C4AR,              0x400E040C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C4CR2,             0x400E0410, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C4SR      I2C4CR2         
#define I2C4SR_bit  I2C4CR2_bit.__sr
__IO_REG32_BIT(I2C4BR0,             0x400E0414, __READ_WRITE , __sbixbr0_bits);

#define SIO4CR0     I2C4CR0
#define SIO4CR0_bit I2C4CR0_bit
#define SIO4CR1     I2C4CR1
#define SIO4CR1_bit I2C4CR1_bit.__sio
#define SIO4DBR     I2C4DBR
#define SIO4DBR_bit I2C4DBR_bit
#define SIO4CR2     I2C4CR2
#define SIO4CR2_bit I2C4CR2_bit.__sio
#define SIO4SR      I2C4CR2
#define SIO4SR_bit  I2C4CR2_bit.__sio.__sr
#define SIO4BR0     I2C4BR0
#define SIO4BR0_bit I2C4BR0_bit

/***************************************************************************
 **
 ** CEC
 **
 ***************************************************************************/
 __IO_REG32_BIT(CECEN,              0x400E2000, __READ_WRITE , __cecen_bits );
 __IO_REG32_BIT(CECADD,             0x400E2004, __READ_WRITE , __cecadd_bits);
 __IO_REG32_BIT(CECRESET,           0x400E2008, __WRITE      , __cecreset_bits);
 __IO_REG32_BIT(CECREN,             0x400E200C, __READ_WRITE , __cecren_bits);
 __IO_REG32_BIT(CECRBUF,            0x400E2010, __READ       , __cecrbuf_bits);
 __IO_REG32_BIT(CECRCR1,            0x400E2014, __READ_WRITE , __cecrcr1_bits);
 __IO_REG32_BIT(CECRCR2,            0x400E2018, __READ_WRITE , __cecrcr2_bits);
 __IO_REG32_BIT(CECRCR3,            0x400E201C, __READ_WRITE , __cecrcr3_bits);
 __IO_REG32_BIT(CECTEN,             0x400E2020, __READ_WRITE , __cecten_bits);
 __IO_REG32_BIT(CECTBUF,            0x400E2024, __READ_WRITE , __cectbuf_bits);
 __IO_REG32_BIT(CECTCR,             0x400E2028, __READ_WRITE , __cectcr_bits);
 __IO_REG32_BIT(CECRSTAT,           0x400E202C, __READ       , __cecrstat_bits);
 __IO_REG32_BIT(CECTSTAT,           0x400E2030, __READ       , __cectstat_bits);
 __IO_REG32_BIT(CECFSSEL,           0x400E2034, __READ_WRITE , __cecfssel_bits);

/***************************************************************************
 **
 ** RMC0
 **
 ***************************************************************************/
 __IO_REG32_BIT(RMC0EN,             0x400E3000, __READ_WRITE , __rmcen_bits   );
 __IO_REG32_BIT(RMC0REN,            0x400E3004, __READ_WRITE , __rmcren_bits  );
 __IO_REG32(    RMC0RBUF1,          0x400E3008, __READ);
 __IO_REG32(    RMC0RBUF2,          0x400E300C, __READ);
 __IO_REG32(    RMC0RBUF3,          0x400E3010, __READ);
 __IO_REG32_BIT(RMC0RCR1,           0x400E3014, __READ_WRITE , __rmcrcr1_bits );
 __IO_REG32_BIT(RMC0RCR2,           0x400E3018, __READ_WRITE , __rmcrcr2_bits );
 __IO_REG32_BIT(RMC0RCR3,           0x400E301C, __READ_WRITE , __rmcrcr3_bits );
 __IO_REG32_BIT(RMC0RCR4,           0x400E3020, __READ_WRITE , __rmcrcr4_bits );
 __IO_REG32_BIT(RMC0RSTAT,          0x400E3024, __READ       , __rmcrstat_bits);
 __IO_REG32_BIT(RMC0END1,           0x400E3028, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMC0END2,           0x400E302C, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMC0END3,           0x400E3030, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMC0FSSEL,          0x400E3034, __READ_WRITE , __rmcfssel_bits);

/***************************************************************************
 **
 ** RMC1
 **
 ***************************************************************************/
 __IO_REG32_BIT(RMC1EN,             0x400E3100, __READ_WRITE , __rmcen_bits   );
 __IO_REG32_BIT(RMC1REN,            0x400E3104, __READ_WRITE , __rmcren_bits  );
 __IO_REG32(    RMC1RBUF1,          0x400E3108, __READ);
 __IO_REG32(    RMC1RBUF2,          0x400E310C, __READ);
 __IO_REG32(    RMC1RBUF3,          0x400E3110, __READ);
 __IO_REG32_BIT(RMC1RCR1,           0x400E3114, __READ_WRITE , __rmcrcr1_bits );
 __IO_REG32_BIT(RMC1RCR2,           0x400E3118, __READ_WRITE , __rmcrcr2_bits );
 __IO_REG32_BIT(RMC1RCR3,           0x400E311C, __READ_WRITE , __rmcrcr3_bits );
 __IO_REG32_BIT(RMC1RCR4,           0x400E3120, __READ_WRITE , __rmcrcr4_bits );
 __IO_REG32_BIT(RMC1RSTAT,          0x400E3124, __READ       , __rmcrstat_bits);
 __IO_REG32_BIT(RMC1END1,           0x400E3128, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMC1END2,           0x400E312C, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMC1END3,           0x400E3130, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMC1FSSEL,          0x400E3134, __READ_WRITE , __rmcfssel_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG8_BIT(ADCLK,                 0x400F0000,__READ_WRITE ,__adclk_bits);
__IO_REG8_BIT(ADMOD0,                0x400F0004,__READ_WRITE ,__admod0_bits);
__IO_REG8_BIT(ADMOD1,                0x400F0008,__READ_WRITE ,__admod1_bits);
__IO_REG8_BIT(ADMOD2,                0x400F000C,__READ_WRITE ,__admod2_bits);
__IO_REG8_BIT(ADMOD3,                0x400F0010,__READ_WRITE ,__admod3_bits);
__IO_REG8_BIT(ADMOD4,                0x400F0014,__READ_WRITE ,__admod4_bits);
__IO_REG8_BIT(ADMOD5,                0x400F0018,__READ_WRITE ,__admod5_bits);
__IO_REG8(    ADCBAS,                0x400F0020,__READ_WRITE );
__IO_REG16_BIT(ADREG08,              0x400F0030,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG19,              0x400F0034,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG2A,              0x400F0038,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG3B,              0x400F003C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG4C,              0x400F0040,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG5D,              0x400F0044,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG6E,              0x400F0048,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG7F,              0x400F004C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREGSP,              0x400F0050,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADCMP0,               0x400F0054,__READ_WRITE ,__adcmpx_bits);
__IO_REG16_BIT(ADCMP1,               0x400F0058,__READ_WRITE ,__adcmpx_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG8_BIT(RTCSECR,              0x400F3000,__READ_WRITE ,__secr_bits);
__IO_REG8_BIT(RTCMINR,              0x400F3001,__READ_WRITE ,__minr_bits);
__IO_REG8_BIT(RTCHOURR,             0x400F3002,__READ_WRITE ,__hourr_bits);
__IO_REG8_BIT(RTCDAYR,              0x400F3004,__READ_WRITE ,__dayr_bits);
__IO_REG8_BIT(RTCDATER,             0x400F3005,__READ_WRITE ,__dater_bits);
__IO_REG8_BIT(RTCMONTHR,            0x400F3006,__READ_WRITE ,__monthr_bits);
__IO_REG8_BIT(RTCYEARR,             0x400F3007,__READ_WRITE ,__yearr_bits);
__IO_REG8_BIT(RTCPAGER,             0x400F3008,__READ_WRITE ,__pager_bits);
__IO_REG8_BIT(RTCRESTR,             0x400F300C,__WRITE      ,__restr_bits);

/***************************************************************************
 **
 ** KWUP (KEY-on Wakeup Circuit)
 **
 ***************************************************************************/
__IO_REG32_BIT(KWUPCR0,             0x400F1000,__READ_WRITE ,__kwupcr_bits);
__IO_REG32_BIT(KWUPCR1,             0x400F1004,__READ_WRITE ,__kwupcr_bits);
__IO_REG32_BIT(KWUPCR2,             0x400F1008,__READ_WRITE ,__kwupcr_bits);
__IO_REG32_BIT(KWUPCR3,             0x400F100C,__READ_WRITE ,__kwupcr_bits);
__IO_REG32_BIT(KWUPKEY,             0x400F1080,__READ       ,__kwupkey_bits);
__IO_REG32_BIT(KWUPCNT,             0x400F1084,__READ_WRITE ,__kwupcnt_bits);
__IO_REG32_BIT(KWUPCLR,             0x400F1088,__WRITE      ,__kwupclr_bits);
__IO_REG32_BIT(KWUPINT,             0x400F108C,__READ       ,__kwupint_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG8_BIT(WDMOD,                0x400F2000,__READ_WRITE ,__wdmod_bits);
__IO_REG8(    WDCR,                 0x400F2004,__WRITE);

/***************************************************************************
 **
 ** RAM
 **
 ***************************************************************************/
__IO_REG32_BIT(RAMWAIT,              0x41FFF058,__READ_WRITE ,__rcwait_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FCSECBIT,            0x41FFF010, __READ_WRITE , __secbit_bits);
__IO_REG32_BIT(FCFLCS,              0x41FFF020, __READ       , __flcs_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  TMPM362F10FG DMA Lines
 **
 ***************************************************************************/
#define DMA_SIO0               0          /* SIO0  Reception / Transmission*/
#define DMA_SIO1               1          /* SIO1  Reception / Transmission*/
#define DMA_SIO2               2          /* SIO2  Reception / Transmission*/
#define DMA_SIO3               3          /* SIO3  Reception / Transmission*/
#define DMA_SIO4               4          /* SIO4  Reception / Transmission*/
#define DMA_SIO5               5          /* SIO5  Reception / Transmission*/
#define DMA_SIO6               6          /* SIO6  Reception / Transmission*/
#define DMA_SIO7               7          /* SIO7  Reception / Transmission*/
#define DMA_SIO8               8          /* SIO8  Reception / Transmission*/
#define DMA_SIO9               9          /* SIO9  Reception / Transmission*/
#define DMA_SIO10             10          /* SIO10 Reception / Transmission*/
#define DMA_SIO11             11          /* SIO11 Reception / Transmission*/
#define DMA_SSPTX             12          /* SSP  Transmission             */
#define DMA_SSPRX             13          /* SSP  Reception                */
#define DMA_SSPRX_S           14          /* SSP  Reception single         */
#define DMA_ADC               15          /* A/D Conversion End            */

/***************************************************************************
 **
 **  TMPM362F10FG Interrupt Lines
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
#define INT_0                ( 0 + EII)   /* Interrupt pin 0               */
#define INT_1                ( 1 + EII)   /* Interrupt pin 1               */
#define INT_2                ( 2 + EII)   /* Interrupt pin 2               */
#define INT_3                ( 3 + EII)   /* Interrupt pin 3               */
#define INT_4                ( 4 + EII)   /* Interrupt pin 4               */
#define INT_5                ( 5 + EII)   /* Interrupt pin 5               */
#define INT_6                ( 6 + EII)   /* Interrupt pin 6               */
#define INT_7                ( 7 + EII)   /* Interrupt pin 7               */
#define INT_8                ( 8 + EII)   /* Interrupt pin 8               */
#define INT_9                ( 9 + EII)   /* Interrupt pin 9               */
#define INT_A                (10 + EII)   /* Interrupt pin 10              */
#define INT_B                (11 + EII)   /* Interrupt pin 11              */
#define INT_C                (12 + EII)   /* Interrupt pin 12              */
#define INT_D                (13 + EII)   /* Interrupt pin 13              */
#define INT_E                (14 + EII)   /* Interrupt pin 14              */
#define INT_F                (15 + EII)   /* Interrupt pin 15              */
#define INT_RX0              (16 + EII)   /* Serial reception (channel.0)  */
#define INT_TX0              (17 + EII)   /* Serial transmit (channel.0)   */
#define INT_RX1              (18 + EII)   /* Serial reception (channel.1)  */
#define INT_TX1              (19 + EII)   /* Serial transmit (channel.1)   */
#define INT_RX2              (20 + EII)   /* Serial reception (channel.2)  */
#define INT_TX2              (21 + EII)   /* Serial transmit (channel.2)   */
#define INT_RX3              (22 + EII)   /* Serial reception (channel.3)  */
#define INT_TX3              (23 + EII)   /* Serial transmit (channel.3)   */
#define INT_RX4              (24 + EII)   /* Serial reception (channel.4)  */
#define INT_TX4              (25 + EII)   /* Serial transmit (channel.4)   */
#define INT_SBI0             (26 + EII)   /* Serial bus interface 0        */
#define INT_SBI1             (27 + EII)   /* Serial bus interface 1        */
#define INT_CECRX            (28 + EII)   /* CEC reception                 */
#define INT_CECTX            (29 + EII)   /* CEC transmission              */
#define INT_AINTRMCRX0       (30 + EII)   /* Remote control signal reception (channel.0)*/
#define INT_AINTRMCRX1       (31 + EII)   /* Remote control signal reception (channel.1)*/
#define INT_RTC              (32 + EII)   /* Real time clock timer         */
#define INT_KWUP             (33 + EII)   /* Key On wakeup                 */
#define INT_SBI2             (34 + EII)   /* Serial bus interface 2        */
#define INT_SBI3             (35 + EII)   /* Serial bus interface 3        */
#define INT_SBI4             (36 + EII)   /* Serial bus interface 4        */
#define INT_ADHP             (37 + EII)   /* Highest priority AD conversion complete interrupt*/
#define INT_ADM0             (38 + EII)   /* AD conversion monitoring function interrupt 0 */
#define INT_ADM1             (39 + EII)   /* AD conversion monitoring function interrupt 1 */
#define INT_TB0              (40 + EII)   /* 16bit TMRB match detection 0  */
#define INT_TB1              (41 + EII)   /* 16bit TMRB match detection 1  */
#define INT_TB2              (42 + EII)   /* 16bit TMRB match detection 2  */
#define INT_TB3              (43 + EII)   /* 16bit TMRB match detection 3  */
#define INT_TB4              (44 + EII)   /* 16bit TMRB match detection 4  */
#define INT_TB5              (45 + EII)   /* 16bit TMRB match detection 5  */
#define INT_TB6              (46 + EII)   /* 16bit TMRB match detection 6  */
#define INT_TB7              (47 + EII)   /* 16bit TMRB match detection 7  */
#define INT_TB8              (48 + EII)   /* 16bit TMRB match detection 8  */
#define INT_TB9              (49 + EII)   /* 16bit TMRB match detection 9  */
#define INT_TBA              (50 + EII)   /* 16bit TMRB match detection A  */
#define INT_TBB              (51 + EII)   /* 16bit TMRB match detection B  */
#define INT_TBC              (52 + EII)   /* 16bit TMRB match detection C  */
#define INT_TBD              (53 + EII)   /* 16bit TMRB match detection D  */
#define INT_TBE              (54 + EII)   /* 16bit TMRB match detection E  */
#define INT_TBF              (55 + EII)   /* 16bit TMRB match detection F  */
#define INT_AD               (58 + EII)   /* A/D conversion completion     */
#define INT_SSP0             (59 + EII)   /* Syncronus serial port         */
#define INT_RX5              (60 + EII)   /* Serial reception (channel.5)  */
#define INT_TX5              (61 + EII)   /* Serial transmission (channel.5)*/
#define INT_RX6              (62 + EII)   /* Serial reception (channel.6)  */
#define INT_TX6              (63 + EII)   /* Serial transmission (channel.6)*/
#define INT_RX7              (64 + EII)   /* Serial reception (channel.7)  */
#define INT_TX7              (65 + EII)   /* Serial transmission (channel.7)*/
#define INT_RX8              (66 + EII)   /* Serial reception (channel.8)  */
#define INT_TX8              (67 + EII)   /* Serial transmission (channel.8)*/
#define INT_RX9              (68 + EII)   /* Serial reception (channel.9)  */
#define INT_TX9              (69 + EII)   /* Serial transmission (channel.9)*/
#define INT_RX10             (70 + EII)   /* Serial reception (channel.A)  */
#define INT_TX10             (71 + EII)   /* Serial transmission (channel.A)*/
#define INT_RX11             (72 + EII)   /* Serial reception (channel.B)  */
#define INT_TX11             (73 + EII)   /* Serial transmission (channel.B)*/
#define INT_CAP10            (74 + EII)   /* 16bit TMRB input capture 00   */
#define INT_CAP11            (75 + EII)   /* 16bit TMRB input capture 01   */
#define INT_CAP20            (76 + EII)   /* 16bit TMRB input capture 10   */
#define INT_CAP21            (77 + EII)   /* 16bit TMRB input capture 11   */
#define INT_CAP50            (80 + EII)   /* 16bit TMRB input capture 50   */
#define INT_CAP51            (81 + EII)   /* 16bit TMRB input capture 51   */
#define INT_CAP60            (82 + EII)   /* 16bit TMRB input capture 60   */
#define INT_CAP61            (83 + EII)   /* 16bit TMRB input capture 61   */
#define INT_CAP70            (84 + EII)   /* 16bit TMRB input capture 70   */
#define INT_CAP71            (85 + EII)   /* 16bit TMRB input capture 71   */
#define INT_CAP90            (86 + EII)   /* 16bit TMRB input capture 90   */
#define INT_CAP91            (87 + EII)   /* 16bit TMRB input capture 91   */
#define INT_CAPA0            (88 + EII)   /* 16bit TMRB input capture A0   */
#define INT_CAPA1            (89 + EII)   /* 16bit TMRB input capture A1   */
#define INT_CAPB0            (90 + EII)   /* 16bit TMRB input capture B0   */
#define INT_CAPB1            (91 + EII)   /* 16bit TMRB input capture B1   */
#define INT_CAPD0            (92 + EII)   /* 16bit TMRB input capture D0   */
#define INT_CAPD1            (93 + EII)   /* 16bit TMRB input capture D1   */
#define INT_CAPE0            (94 + EII)   /* 16bit TMRB input capture E0   */
#define INT_CAPE1            (95 + EII)   /* 16bit TMRB input capture E1   */
#define INT_CAPF0            (96 + EII)   /* 16bit TMRB input capture F0   */
#define INT_CAPF1            (97 + EII)   /* 16bit TMRB input capture F1   */
#define INT_DMACERR          (98 + EII)   /* DMA transmission error        */
#define INT_DMACTC0          (99 + EII)   /* DMA transmission completion   */

#endif    /* __IOTMPM362F10FG_H */

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
Interrupt17  = INT8           0x60
Interrupt18  = INT9           0x64
Interrupt19  = INTA           0x68
Interrupt20  = INTB           0x6C
Interrupt21  = INTC           0x70
Interrupt22  = INTD           0x74
Interrupt23  = INTE           0x78
Interrupt24  = INTF           0x7C
Interrupt25  = INTRX0         0x80
Interrupt26  = INTTX0         0x84
Interrupt27  = INTRX1         0x88
Interrupt28  = INTTX1         0x8C
Interrupt29  = INTRX2         0x90
Interrupt30  = INTTX2         0x94
Interrupt31  = INTRX3         0x98
Interrupt32  = INTTX3         0x9C
Interrupt33  = INTRX4         0xA0
Interrupt34  = INTTX4         0xA4
Interrupt35  = INTSBI0        0xA8
Interrupt36  = INTSBI1        0xAC
Interrupt37  = INTCECRX       0xB0
Interrupt38  = INTCECTX       0xB4
Interrupt39  = INTAINTRMCRX0  0xB8
Interrupt40  = INTAINTRMCRX1  0xBC
Interrupt41  = INTRTC         0xC0
Interrupt42  = INTKWUP        0xC4
Interrupt43  = INTSBI2        0xC8
Interrupt44  = INTSBI3        0xCC
Interrupt45  = INTSBI4        0xD0
Interrupt46  = INTADHP        0xD4
Interrupt47  = INTADM0        0xD8
Interrupt48  = INTADM1        0xDC
Interrupt49  = INTTB0         0xE0
Interrupt50  = INTTB1         0xE4
Interrupt51  = INTTB2         0xE8
Interrupt52  = INTTB3         0xEC
Interrupt53  = INTTB4         0xF0
Interrupt54  = INTTB5         0xF4
Interrupt55  = INTTB6         0xF8
Interrupt56  = INTTB7         0xFC
Interrupt57  = INTTB8         0x100
Interrupt58  = INTTB9         0x104
Interrupt59  = INTTBA         0x108
Interrupt60  = INTTBB         0x10C
Interrupt61  = INTTBC         0x110
Interrupt62  = INTTBD         0x114
Interrupt63  = INTTBE         0x118
Interrupt64  = INTTBF         0x11C
Interrupt65  = INTAD          0x128
Interrupt66  = INTSSP0        0x12C
Interrupt67  = INTRX5         0x130
Interrupt68  = INTTX5         0x134
Interrupt69  = INTRX6         0x138
Interrupt70  = INTTX6         0x13C
Interrupt71  = INTRX7         0x140
Interrupt72  = INTTX7         0x144
Interrupt73  = INTRX8         0x148
Interrupt74  = INTTX8         0x14C
Interrupt75  = INTRX9         0x150
Interrupt76  = INTTX9         0x154
Interrupt77  = INTRX10        0x158
Interrupt78  = INTTX10        0x15C
Interrupt79  = INTRX11        0x160
Interrupt80  = INTTX11        0x164
Interrupt81  = INTCAP10       0x168
Interrupt82  = INTCAP11       0x16C
Interrupt83  = INTCAP20       0x170
Interrupt84  = INTCAP21       0x174
Interrupt85  = INTCAP50       0x180
Interrupt86  = INTCAP51       0x184
Interrupt87  = INTCAP60       0x188
Interrupt88  = INTCAP61       0x18C
Interrupt89  = INTCAP70       0x190
Interrupt90  = INTCAP71       0x194
Interrupt91  = INTCAP90       0x198
Interrupt92  = INTCAP91       0x19C
Interrupt93  = INTCAPA0       0x1A0
Interrupt94  = INTCAPA1       0x1A4
Interrupt95  = INTCAPB0       0x1A8
Interrupt96  = INTCAPB1       0x1AC
Interrupt97  = INTCAPD0       0x1B0
Interrupt98  = INTCAPD1       0x1B4
Interrupt99  = INTCAPE0       0x1B8
Interrupt100 = INTCAPE1       0x1BC
Interrupt101 = INTCAPF0       0x1C0
Interrupt102 = INTCAPF1       0x1C4
Interrupt103 = INTDMACERR     0x1C8
Interrupt104 = INTDMACTC0     0x1CC

###DDF-INTERRUPT-END###*/
