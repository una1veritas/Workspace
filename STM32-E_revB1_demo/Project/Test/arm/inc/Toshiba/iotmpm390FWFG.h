/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM390FWFG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 54237 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM390FWFG_H
#define __IOTMPM390FWFG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM390FWFG SPECIAL FUNCTION REGISTERS
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

/*Voltage detection control register*/
typedef struct {
  __REG32 VD1LVL       : 2;
  __REG32              : 2;
  __REG32 VD2LVL       : 2;
  __REG32              : 2;
  __REG32 VD1EN        : 1;
  __REG32 VD2EN        : 1;
  __REG32 VD1MOD       : 1;
  __REG32 VD2MOD       : 1;
  __REG32              :20;
} __vdcr_bits;

/*Voltage detection status register*/
typedef struct {
  __REG32 LVD1ST       : 1;
  __REG32 LVD2ST       : 1;
  __REG32              :30;
} __vdsr_bits;

/* System Control Register */
typedef struct {
  __REG32 GEAR    : 3;
  __REG32         : 5;
  __REG32 PRCK    : 3;
  __REG32         : 1;
  __REG32 FPSEL   : 1;
  __REG32         : 3;
  __REG32 SCOSEL  : 2;
  __REG32         :14;
} __cgsyscr_bits;

/* Oscillation Control Register */
typedef struct {
  __REG32 WUEON     : 1;
  __REG32 WUEF      : 1;
  __REG32           : 1;
  __REG32 WUPSEL    : 1;
  __REG32           : 4;
  __REG32 XEN       : 1;
  __REG32           : 5;
  __REG32 WUDOR_L   : 2;
  __REG32 XEN2      : 1;
  __REG32 OSCSEL    : 1;
  __REG32           : 2;
  __REG32 WUDOR_H   :12;
} __cgosccr_bits;

/* Standby Control Register */
typedef struct {
  __REG32 STBY      : 3;
  __REG32           : 5;
  __REG32 RXEN      : 1;
  __REG32 RXTEN     : 1;
  __REG32           : 6;
  __REG32 DRVE      : 1;
  __REG32           : 1;
  __REG32 SDFLASH   : 1;
  __REG32 ISOFLASH  : 1;
  __REG32           :12;
} __cgstbycr_bits;

/* CGCKSEL Register */
typedef struct {
  __REG32  SYSCKFL   : 1;
  __REG32  SYSCK     : 1;
  __REG32            :30;
} __cgcksel_bits;

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
  __REG32 INT10EN   : 1;
  __REG32           : 1;
  __REG32 EMST10    : 2;
  __REG32 EMCG10    : 3;
  __REG32           :25;
} __cgimcge_bits;

/* CG Interrupt Request Clear Register */
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
  __REG32           : 1;
  __REG32 SYSRSTF   : 1;
  __REG32 OFDRSTF   : 1;
  __REG32           :26;
} __cgrstflg_bits;

/*PORT A Register*/
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

/*PORT A Control Register */
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

/*PORT A Function Register 1*/
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

/*PORT A Pull-Up Control Register */
typedef struct {
  __REG8  PA0UP  : 1;
  __REG8         : 1;
  __REG8  PA2UP  : 1;
  __REG8  PA3UP  : 1;
  __REG8  PA4UP  : 1;
  __REG8  PA5UP  : 1;
  __REG8  PA6UP  : 1;
  __REG8  PA7UP  : 1;
} __papup_bits;

/*PORT A Pull-Down Control Register */
typedef struct {
  __REG8         : 1;
  __REG8  PA1DN  : 1;
  __REG8         : 6;
} __papdn_bits;

/*PORT A Input Enable Control Register */
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
  __REG8       : 4;
} __pb_bits;

/*PORT B Control Register */
typedef struct {
  __REG8  PB0C  : 1;
  __REG8  PB1C  : 1;
  __REG8  PB2C  : 1;
  __REG8  PB3C  : 1;
  __REG8        : 4;
} __pbcr_bits;

/*PORT B Function Register 1*/
typedef struct {
  __REG8  PB0F1  : 1;
  __REG8  PB1F1  : 1;
  __REG8  PB2F1  : 1;
  __REG8         : 5;
} __pbfr1_bits;

/*PORT B Pull-Up Control Register */
typedef struct {
  __REG8  PB0UP  : 1;
  __REG8  PB1UP  : 1;
  __REG8  PB2UP  : 1;
  __REG8         : 5;
} __pbpup_bits;

/*PORT B Input Enable Control Register */
typedef struct {
  __REG8  PB0IE  : 1;
  __REG8  PB1IE  : 1;
  __REG8  PB2IE  : 1;
  __REG8  PB3IE  : 1;
  __REG8         : 4;
} __pbie_bits;

/*PORT C Register*/
typedef struct {
  __REG8  PC0  : 1;
  __REG8  PC1  : 1;
  __REG8  PC2  : 1;
  __REG8  PC3  : 1;
  __REG8       : 4;
} __pc_bits;

/*PORT C Pull-Up Control Register */
typedef struct {
  __REG8  PC0UP  : 1;
  __REG8  PC1UP  : 1;
  __REG8  PC2UP  : 1;
  __REG8  PC3UP  : 1;
  __REG8         : 4;
} __pcpup_bits;

/*PORT C Input Enable Control Register */
typedef struct {
  __REG8  PC0IE  : 1;
  __REG8  PC1IE  : 1;
  __REG8  PC2IE  : 1;
  __REG8  PC3IE  : 1;
  __REG8         : 4;
} __pcie_bits;

/*PORT D Register*/
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

/*PORT D Function Register 1*/
typedef struct {
  __REG8  PD0F1  : 1;
  __REG8  PD1F1  : 1;
  __REG8  PD2F1  : 1;
  __REG8  PD3F1  : 1;
  __REG8         : 4;
} __pdfr1_bits;

/*Port D pull-up control register*/
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

/*PORT E Register*/
typedef struct {
  __REG8  PE0  : 1;
  __REG8  PE1  : 1;
  __REG8  PE2  : 1;
  __REG8  PE3  : 1;
  __REG8  PE4  : 1;
  __REG8  PE5  : 1;
  __REG8  PE6  : 1;
  __REG8       : 1;
} __pe_bits;

/*PORT E Control Register */
typedef struct {
  __REG8  PE0C  : 1;
  __REG8  PE1C  : 1;
  __REG8  PE2C  : 1;
  __REG8  PE3C  : 1;
  __REG8  PE4C  : 1;
  __REG8  PE5C  : 1;
  __REG8  PE6C  : 1;
  __REG8        : 1;
} __pecr_bits;

/*PORT E Function Register 1*/
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

/*PORT E Function Register 2*/
typedef struct {
  __REG8         : 2;
  __REG8  PE2F2  : 1;
  __REG8         : 3;
  __REG8  PE6F2  : 1;
  __REG8         : 1;
} __pefr2_bits;

/*PORT E Open Drain Control Register */
typedef struct {
  __REG8  PE0OD  : 1;
  __REG8  PE1OD  : 1;
  __REG8  PE2OD  : 1;
  __REG8  PE3OD  : 1;
  __REG8         : 4;
} __peod_bits;

/*PORT E Pull-Up Control Register */
typedef struct {
  __REG8  PE0UP  : 1;
  __REG8  PE1UP  : 1;
  __REG8  PE2UP  : 1;
  __REG8  PE3UP  : 1;
  __REG8         : 4;
} __pepup_bits;

/*PORT E Input Enable Control Register */
typedef struct {
  __REG8  PE0IE  : 1;
  __REG8  PE1IE  : 1;
  __REG8  PE2IE  : 1;
  __REG8  PE3IE  : 1;
  __REG8  PE4IE  : 1;
  __REG8  PE5IE  : 1;
  __REG8  PE6IE  : 1;
  __REG8         : 1;
} __peie_bits;

/*PORT F Register*/
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

/*PORT F Control Register */
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

/*PORT F Function Register 1*/
typedef struct {
  __REG8  PF0F1  : 1;
  __REG8  PF1F1  : 1;
  __REG8  PF2F1  : 1;
  __REG8  PF3F1  : 1;
  __REG8  PF4F1  : 1;
  __REG8  PF5F1  : 1;
  __REG8  PF6F1  : 1;
  __REG8  PF7F1  : 1;
} __pffr1_bits;

/*PORT F Function Register 2*/
typedef struct {
  __REG8         : 2;
  __REG8  PF2F2  : 1;
  __REG8         : 5;
} __pffr2_bits;

/*Port F open drain control register */
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

/*PORT F Pull-Up Control Register */
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

/*PORT F Input Enable Control Register */
typedef struct {
  __REG8  PF0IE  : 1;
  __REG8  PF1IE  : 1;
  __REG8  PF2IE  : 1;
  __REG8  PF3IE  : 1;
  __REG8  PF4IE  : 1;
  __REG8  PF5IE  : 1;
  __REG8  PF6IE  : 1;
  __REG8  PF7IE  : 1;
} __pfie_bits;

/*PORT G Register*/
typedef struct {
  __REG8  PG0  : 1;
  __REG8  PG1  : 1;
  __REG8  PG2  : 1;
  __REG8  PG3  : 1;
  __REG8       : 4;
} __pg_bits;

/*PORT G Register*/
typedef struct {
  __REG8  PG0C  : 1;
  __REG8  PG1C  : 1;
  __REG8  PG2C  : 1;
  __REG8  PG3C  : 1;
  __REG8        : 4;
} __pgcr_bits;

/*PORT G Function Register 1*/
typedef struct {
  __REG8  PG0F1  : 1;
  __REG8  PG1F1  : 1;
  __REG8         : 1;
  __REG8  PG3F1  : 1;
  __REG8         : 4;
} __pgfr1_bits;

/*PORT G Open Drain Control Register */
typedef struct {
  __REG8  PG0OD  : 1;
  __REG8  PG1OD  : 1;
  __REG8         : 6;
} __pgod_bits;

/*PORT G Pull-Up Control Register */
typedef struct {
  __REG8  PG0UP  : 1;
  __REG8  PG1UP  : 1;
  __REG8  PG2UP  : 1;
  __REG8  PG3UP  : 1;
  __REG8         : 4;
} __pgpup_bits;

/*PORT G Input Enable Control Register */
typedef struct {
  __REG8  PG0IE  : 1;
  __REG8  PG1IE  : 1;
  __REG8  PG2IE  : 1;
  __REG8  PG3IE  : 1;
  __REG8         : 4;
} __pgie_bits;

/*PORT H Register*/
typedef struct {
  __REG8  PH0  : 1;
  __REG8  PH1  : 1;
  __REG8  PH2  : 1;
  __REG8  PH3  : 1;
  __REG8  PH4  : 1;
  __REG8  PH5  : 1;
  __REG8  PH6  : 1;
  __REG8       : 1;
} __ph_bits;

/*PORT H Control Register 1*/
typedef struct {
  __REG8  PH0C  : 1;
  __REG8  PH1C  : 1;
  __REG8  PH2C  : 1;
  __REG8  PH3C  : 1;
  __REG8  PH4C  : 1;
  __REG8  PH5C  : 1;
  __REG8  PH6C  : 1;
  __REG8        : 1;
} __phcr_bits;

/*PORT H Function Register 1*/
typedef struct {
  __REG8  PH0F1  : 1;
  __REG8  PH1F1  : 1;
  __REG8  PH2F1  : 1;
  __REG8  PH3F1  : 1;
  __REG8  PH4F1  : 1;
  __REG8  PH5F1  : 1;
  __REG8  PH6F1  : 1;
  __REG8         : 1;
} __phfr1_bits;

/*Port H open drain control register*/
typedef struct {
  __REG8         : 2;
  __REG8  PH2OD  : 1;
  __REG8         : 2;
  __REG8  PH5OD  : 1;
  __REG8  PH6OD  : 1;
  __REG8         : 1;
} __phod_bits;

/*PORT H Pull-Up Control Register */
typedef struct {
  __REG8  PH0UP  : 1;
  __REG8  PH1UP  : 1;
  __REG8  PH2UP  : 1;
  __REG8  PH3UP  : 1;
  __REG8  PH4UP  : 1;
  __REG8  PH5UP  : 1;
  __REG8  PH6UP  : 1;
  __REG8         : 1;
} __phpup_bits;

/*PORT H Input Enable Control Register */
typedef struct {
  __REG8  PH0IE  : 1;
  __REG8  PH1IE  : 1;
  __REG8  PH2IE  : 1;
  __REG8  PH3IE  : 1;
  __REG8  PH4IE  : 1;
  __REG8  PH5IE  : 1;
  __REG8  PH6IE  : 1;
  __REG8         : 1;
} __phie_bits;

/*PORT I Register*/
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

/*PORT I Control Register 1*/
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

/*Port I function register 1*/
typedef struct {
  __REG8  PI0F1 : 1;
  __REG8  PI1F1 : 1;
  __REG8  PI2F1 : 1;
  __REG8  PI3F1 : 1;
  __REG8  PI4F1 : 1;
  __REG8  PI5F1 : 1;
  __REG8  PI6F1 : 1;
  __REG8  PI7F1 : 1;
} __pifr1_bits;

/*Port I open drain control register*/
typedef struct {
  __REG8         : 2;
  __REG8  PI2OD  : 1;
  __REG8         : 2;
  __REG8  PI5OD  : 1;
  __REG8         : 2;
} __piod_bits;

/*PORT I Pull-Up Control Register */
typedef struct {
  __REG8  PI0UP  : 1;
  __REG8  PI1UP  : 1;
  __REG8  PI2UP  : 1;
  __REG8  PI3UP  : 1;
  __REG8  PI4UP  : 1;
  __REG8  PI5UP  : 1;
  __REG8  PI6UP  : 1;
  __REG8  PI7UP  : 1;
} __pipup_bits;

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

/*PORT J Register*/
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

/*PORT J Control Register 1*/
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

/*PORT J Function Register 1*/
typedef struct {
  __REG8  PJ0F1  : 1;
  __REG8  PJ1F1  : 1;
  __REG8  PJ2F1  : 1;
  __REG8  PJ3F1  : 1;
  __REG8  PJ4F1  : 1;
  __REG8  PJ5F1  : 1;
  __REG8  PJ6F1  : 1;
  __REG8  PJ7F1  : 1;
} __pjfr1_bits;

/*PORT J Pull-Up Control Register */
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

/*PORT J Input Enable Control Register */
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

/*PORT K Register*/
typedef struct {
  __REG8  PK0  : 1;
  __REG8  PK1  : 1;
  __REG8       : 6;
} __pk_bits;

/*PORT K Control Register*/
typedef struct {
  __REG8  PK0C  : 1;
  __REG8  PK1C  : 1;
  __REG8        : 6;
} __pkcr_bits;

/*PORT K Function Register 1*/
typedef struct {
  __REG8  PK0F1  : 1;
  __REG8  PK1F1  : 1;
  __REG8         : 6;
} __pkfr1_bits;

/*PORT K Function Register 2*/
typedef struct {
  __REG8         : 1;
  __REG8  PK1F2  : 1;
  __REG8         : 6;
} __pkfr2_bits;

/*PORT K Function Register 3*/
typedef struct {
  __REG8         : 1;
  __REG8  PK1F3  : 1;
  __REG8         : 6;
} __pkfr3_bits;

/*PORT K Pull-Up Control Register*/
typedef struct {
  __REG8         : 1;
  __REG8  PK1UP  : 1;
  __REG8         : 6;
} __pkpup_bits;

/*PORT K Input Enable Control Register*/
typedef struct {
  __REG8  PK0IE  : 1;
  __REG8  PK1IE  : 1;
  __REG8         : 6;
} __pkie_bits;

/*PORT L Register*/
typedef struct {
  __REG8  PL0  : 1;
  __REG8  PL1  : 1;
  __REG8  PL2  : 1;
  __REG8  PL3  : 1;
  __REG8  PL4  : 1;
  __REG8  PL5  : 1;
  __REG8       : 2;
} __pl_bits;

/*PORT L Register*/
typedef struct {
  __REG8  PL0C : 1;
  __REG8  PL1C : 1;
  __REG8  PL2C : 1;
  __REG8  PL3C : 1;
  __REG8  PL4C : 1;
  __REG8  PL5C : 1;
  __REG8       : 2;
} __plcr_bits;

/*PORT L Function Register 1*/
typedef struct {
  __REG8  PL0F1  : 1;
  __REG8  PL1F1  : 1;
  __REG8  PL2F1  : 1;
  __REG8  PL3F1  : 1;
  __REG8  PL4F1  : 1;
  __REG8  PL5F1  : 1;
  __REG8         : 2;
} __plfr1_bits;

/*Port L open drain control register*/
typedef struct {
  __REG8  PL0OD  : 1;
  __REG8  PL1OD  : 1;
  __REG8  PL2OD  : 1;
  __REG8  PL3OD  : 1;
  __REG8  PL4OD  : 1;
  __REG8  PL5OD  : 1;
  __REG8         : 2;
} __plod_bits;

/*Port L pull-up control register*/
typedef struct {
  __REG8  PL0UP  : 1;
  __REG8  PL1UP  : 1;
  __REG8  PL2UP  : 1;
  __REG8  PL3UP  : 1;
  __REG8  PL4UP  : 1;
  __REG8  PL5UP  : 1;
  __REG8         : 2;
} __plpup_bits;

/*PORT L Input Enable Control Register*/
typedef struct {
  __REG8  PL0IE  : 1;
  __REG8  PL1IE  : 1;
  __REG8  PL2IE  : 1;
  __REG8  PL3IE  : 1;
  __REG8  PL4IE  : 1;
  __REG8  PL5IE  : 1;
  __REG8         : 2;
} __plie_bits;

/*TMRBn enable register (channels 0 through 8)*/
typedef struct {
  __REG32           : 7;
  __REG32  TBEN     : 1;
  __REG32           :24;
} __tbxen_bits;

/*TMRB RUN register (channels 0 through 8)*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32           : 1;
  __REG32  TBPRUN   : 1;
  __REG32           :29;
} __tbxrun_bits;

/*TMRB control register (channels 0 through 8)*/
typedef struct {
  __REG32  CSSEL    : 1;
  __REG32  TRGSEL   : 1;
  __REG32           : 1;
  __REG32  I2TB     : 1;
  __REG32           : 1;
  __REG32  TBSYNC   : 1;
  __REG32           : 1;
  __REG32  TBWBF    : 1;
  __REG32           :24;
} __tbxcr_bits;

/*TMRB mode register (channels 0 thorough 8)*/
typedef struct {
  __REG32  TBCLK    : 2;
  __REG32  TBCLE    : 1;
  __REG32  TBCPM    : 2;
  __REG32  TBCP0    : 1;
  __REG32           :26;
} __tbxmod_bits;

/*TMRB flip-flop control register (channels 0 through 8)*/
typedef struct {
  __REG32  TBFF0C   : 2;
  __REG32  TBE0T1   : 1;
  __REG32  TBE1T1   : 1;
  __REG32  TBC0T1   : 1;
  __REG32  TBC1T1   : 1;
  __REG32           :26;
} __tbxffcr_bits;

/*TMRB status register (channels 0 through 8)*/
typedef struct {
  __REG32  INTTB0   : 1;
  __REG32  INTTB1   : 1;
  __REG32  INTTBOF  : 1;
  __REG32           :29;
} __tbxst_bits;

/*TMRB interrupt mask register (channels 0 through 8)*/
typedef struct {
  __REG32  TBIM0    : 1;
  __REG32  TBIM1    : 1;
  __REG32  TBIMOF   : 1;
  __REG32           :29;
} __tbxim_bits;

/*TMRB read capture register (channels 0 through 8)*/
typedef struct {
  __REG32  TBUC     :16;
  __REG32           :16;
} __tbxuc_bits;

/*TMRB timer register 0 (channels 0 through 8)*/
typedef struct {
  __REG32  TBRG0    :16;
  __REG32           :16;
} __tbxrg0_bits;

/*TMRB timer register 1 (channels 0 through 8)*/
typedef struct {
  __REG32  TBRG1    :16;
  __REG32           :16;
} __tbxrg1_bits;

/*TMRB capture register 0 (channels 0 through 8)*/
typedef struct {
  __REG32  TBCP0    :16;
  __REG32           :16;
} __tbxcp0_bits;

/*TMRB capture register 1 (channels 0 through 8)*/
typedef struct {
  __REG32  TBCP1    :16;
  __REG32           :16;
} __tbxcp1_bits;

/*PHCNT RUN register*/
typedef struct {
  __REG32  PHCRUN   : 1;
  __REG32           :31;
} __phcxrun_bits;

/*PHCNT control register*/
typedef struct {
  __REG32  PHCMD    : 1;
  __REG32  NFOFF    : 1;
  __REG32  CMP0EN   : 1;
  __REG32  CMP1EN   : 1;
  __REG32  EVRYINT  : 1;
  __REG32           :27;
} __phcxcr_bits;

/*PHCNT Timer Enable Register*/
typedef struct {
  __REG32  PHCEN    : 1;
  __REG32           :31;
} __phcxen_bits;

/*PHCNT Status register*/
typedef struct {
  __REG32  CMP0     : 1;
  __REG32  CMP1     : 1;
  __REG32  OVF      : 1;
  __REG32  UDF      : 1;
  __REG32           :28;
} __phcxflg_bits;

/*PHCNT Compare Register 0 */
typedef struct {
  __REG32  PHCCMP0  :16;
  __REG32           :16;
} __phcxcmp0_bits;

/*PHCNT Compare Register 1 */
typedef struct {
  __REG32  PHCCMP1  :16;
  __REG32           :16;
} __phcxcmp1_bits;

/*PHCNT Count Register */
typedef struct {
  __REG32  PHCCNT   :16;
  __REG32           :16;
} __phcxcnt_bits;

/*SIOx Enable register*/
typedef struct {
  __REG32  SIOE     : 1;
  __REG32  INTSEL   : 1;
  __REG32  SCLKR    : 1;
  __REG32  TXR      : 1;
  __REG32  BRCKSEL  : 1;
  __REG32           :27;
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

/*SIO1 RX FIFO configuration register*/
typedef struct {
  __REG32  RIL      : 6;
  __REG32  RFIS     : 1;
  __REG32  RFCS     : 1;
  __REG32           :24;
} __sc0rfc_bits;

/*SIOx RX FIFO configuration register*/
typedef struct {
  __REG32  RIL      : 2;
  __REG32           : 4;
  __REG32  RFIS     : 1;
  __REG32  RFCS     : 1;
  __REG32           :24;
} __scxrfc_bits;

/*SIO0 TX FIFO configuration register*/
typedef struct {
  __REG32  TIL      : 6;
  __REG32  TFIS     : 1;
  __REG32  TFCS     : 1;
  __REG32           :24;
} __sc0tfc_bits;

/*SIOx TX FIFO configuration register*/
typedef struct {
  __REG32  TIL      : 2;
  __REG32           : 4;
  __REG32  TFIS     : 1;
  __REG32  TFCS     : 1;
  __REG32           :24;
} __scxtfc_bits;

/*SIO0 RX FIFO status register*/
typedef struct {
  __REG32  RLVL     : 7;
  __REG32  ROR      : 1;
  __REG32           :24;
} __sc0rst_bits;

/*SIOx RX FIFO status register*/
typedef struct {
  __REG32  RLVL     : 3;
  __REG32           : 4;
  __REG32  ROR      : 1;
  __REG32           :24;
} __scxrst_bits;

/*SIO0 TX FIFO status register*/
typedef struct {
  __REG32  TLVL     : 7;
  __REG32  TUR      : 1;
  __REG32           :24;
} __sc0tst_bits;

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

/*Serial bus interrupt select register*/
typedef struct {
  __REG32 INTSEL  : 1;
  __REG32         :31;
} __sbixintsel_bits;

/*I2C0CR1 (I2C0 Control Register 1)*/
typedef struct {
  __REG32 SCK     : 3;
  __REG32 NOACK   : 1;
  __REG32 ACK     : 1;
  __REG32 BC      : 3;
  __REG32         :24;
} __i2ccr1_bits;

/*I2C0DBR (I2C0 Data Buffer Register)*/
typedef struct {
  __REG32 DB      : 8;
  __REG32         :24;
} __i2cdbr_bits;

/*I2C0AR (I2C0 (Slave) Address Register)*/
typedef struct {
  __REG32 ALS     : 1;
  __REG32 SA      : 7;
  __REG32         :24;
} __i2car_bits;

/*I2C0CR2 (I2C0 Control Register 2)*/
typedef union {
  /*I2CxCR2*/
  struct {
  __REG32 SWRES   : 2;
  __REG32         : 1;
  __REG32 I2CM    : 1;
  __REG32 PIN     : 1;
  __REG32 BB      : 1;
  __REG32 TRX     : 1;
  __REG32 MST     : 1;
  __REG32         :24;
  };
  /*I2CxSR*/
  struct {
  __REG32 LRB     : 1;
  __REG32 AD0     : 1;
  __REG32 AAS     : 1;
  __REG32 AL      : 1;
  __REG32 PIN     : 1;
  __REG32 BB      : 1;
  __REG32 TRX     : 1;
  __REG32 MST     : 1;
  __REG32         :24;
  } __sr;
} __i2ccr2_bits;

/*I2C0PRS (I2C0 Prescaler Clock Set Register)*/
typedef struct {
  __REG32 PRSCK   : 5;
  __REG32         :27;
} __i2cprs_bits;

/*I2C0IE (I2C0 Interrupt Enable Register)*/
typedef struct {
  __REG32 IE      : 1;
  __REG32         :31;
} __i2cie_bits;

/*I2C0IR (I2C0 Interrupt Register)*/
typedef struct {
  __REG32 IS      : 1;
  __REG32         :31;
} __i2cir_bits;

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
  __REG32         : 1;
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

/*SSP1INTSEL (SSP Interrupt clear register)*/
typedef struct {
  __REG32 INTSEL  : 1;
  __REG32 FSSSEL  : 1;
  __REG32         :30;
} __sspintsel_bits;

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

/*Remote Control Receive End Bit Number Register 1*/
typedef struct {
  __REG32 RMCEND    : 7;
  __REG32           :25;
} __rmcend_bits;

/*A/D Conversion Clock Setting Register*/
typedef struct {
  __REG8  ADCLK   : 3;
  __REG8          : 1;
  __REG8  TSH0    : 1;
  __REG8          : 1;
  __REG8  ENDAF   : 1;
  __REG8  ADFS    : 1;
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
  __REG8  ADSCN   : 2;
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

/*A/D Conversion Result Registers */
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

/*Watchdog Timer Mode Register*/
typedef struct {
  __REG8          : 1;
  __REG8  RESCR   : 1;
  __REG8  I2WDT   : 1;
  __REG8          : 1;
  __REG8  WDTP    : 3;
  __REG8  WDTE    : 1;
} __wdmod_bits;

/*Second column register*/
typedef struct {
  __REG8  SE      : 7;
  __REG8          : 1;
} __rtcsecr_bits;

/*Minute column register*/
typedef struct {
  __REG8  MI      : 7;
  __REG8          : 1;
} __rtcminr_bits;

/*Hour column register*/
typedef struct {
  __REG8  HO      : 6;
  __REG8          : 2;
} __rtchourr_bits;

/*Day of the week column register*/
typedef struct {
  __REG8  WE      : 3;
  __REG8          : 5;
} __rtcdayr_bits;

/*Day column register*/
typedef struct {
  __REG8  DA      : 6;
  __REG8          : 2;
} __rtcdater_bits;

/*Month column register*/
typedef struct {
  __REG8  MO      : 5;
  __REG8          : 3;
} __rtcmonthr_bits;

/*Year column register*/
typedef union {
  __REG8  YE      : 8;
  /*RTCYEARR*/
  struct {
  __REG8  LEAP    : 2;
  __REG8          : 6;
  };
} __rtcyearr_bits;

/*PAGE register */
typedef struct {
  __REG8  PAGE    : 1;
  __REG8          : 1;
  __REG8  ENAALM  : 1;
  __REG8  ENATMR  : 1;
  __REG8  ADJUST  : 1;
  __REG8          : 2;
  __REG8  INTENA  : 1;
} __rtcpager_bits;

/*Reset register*/
typedef struct {
  __REG8          : 4;
  __REG8  RSTALM  : 1;
  __REG8  RSTTMR  : 1;
  __REG8  DIS16HZ : 1;
  __REG8  DIS1HZ  : 1;
} __rtcrestr_bits;

/*RTC Status Monitor*/
typedef struct {
  __REG8  RTCSET  : 1;
  __REG8  RTCINI  : 1;
  __REG8          : 6;
} __rtcsta_bits;

/*RTC RTCADJCTL*/
typedef struct {
  __REG8  AJEN    : 1;
  __REG8  AJSEL   : 1;
  __REG8          : 6;
} __rtcadjctl_bits;

/*Oscillation frequency detection control register 1*/
typedef struct {
  __REG32 OFDWEN    : 8;
  __REG32           :24;
} __ofdcr1_bits;

/*Oscillation frequency detection control register 2*/
typedef struct {
  __REG32 OFDEN     : 8;
  __REG32           :24;
} __ofdcr2_bits;

/*Lower detection frequency setting register (OFDMN)*/
typedef struct {
  __REG32 OFDMN        : 9;
  __REG32              :23;
} __ofdmn_bits;

/*Higher detection frequency setting register (OFDMX)*/
typedef struct {
  __REG32 OFDMX        : 9;
  __REG32              :23;
} __ofdmx_bits;

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
  __REG32         :12;
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

/* Interrupt Set-Enable Registers 32-56 */
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
  __REG32                 : 7;
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

/* Interrupt Clear-Enable Registers 32-56 */
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
  __REG32                 : 7;
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

/* Interrupt Set-Pending Register 32-56 */
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
  __REG32                 : 7;
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

/* Interrupt Clear-Pending Register 32-56 */
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
  __REG32                 : 7;
} __clrpend1_bits;

/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_0          : 3;
  __REG32                 : 5;
  __REG32  PRI_1          : 3;
  __REG32                 : 5;
  __REG32  PRI_2          : 3;
  __REG32                 : 5;
  __REG32  PRI_3          : 3;
} __pri0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_4          : 3;
  __REG32                 : 5;
  __REG32  PRI_5          : 3;
  __REG32                 : 5;
  __REG32  PRI_6          : 3;
  __REG32                 : 5;
  __REG32  PRI_7          : 3;
} __pri1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_8          : 3;
  __REG32                 : 5;
  __REG32  PRI_9          : 3;
  __REG32                 : 5;
  __REG32  PRI_10         : 3;
  __REG32                 : 5;
  __REG32  PRI_11         : 3;
} __pri2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_12         : 3;
  __REG32                 : 5;
  __REG32  PRI_13         : 3;
  __REG32                 : 5;
  __REG32  PRI_14         : 3;
  __REG32                 : 5;
  __REG32  PRI_15         : 3;
} __pri3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_16         : 3;
  __REG32                 : 5;
  __REG32  PRI_17         : 3;
  __REG32                 : 5;
  __REG32  PRI_18         : 3;
  __REG32                 : 5;
  __REG32  PRI_19         : 3;
} __pri4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_20         : 3;
  __REG32                 : 5;
  __REG32  PRI_21         : 3;
  __REG32                 : 5;
  __REG32  PRI_22         : 3;
  __REG32                 : 5;
  __REG32  PRI_23         : 3;
} __pri5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_24         : 3;
  __REG32                 : 5;
  __REG32  PRI_25         : 3;
  __REG32                 : 5;
  __REG32  PRI_26         : 3;
  __REG32                 : 5;
  __REG32  PRI_27         : 3;
} __pri6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_28         : 3;
  __REG32                 : 5;
  __REG32  PRI_29         : 3;
  __REG32                 : 5;
  __REG32  PRI_30         : 3;
  __REG32                 : 5;
  __REG32  PRI_31         : 3;
} __pri7_bits;

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_32         : 3;
  __REG32                 : 5;
  __REG32  PRI_33         : 3;
  __REG32                 : 5;
  __REG32  PRI_34         : 3;
  __REG32                 : 5;
  __REG32  PRI_35         : 3;
} __pri8_bits;

/* Interrupt Priority Registers 36-39 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_36         : 3;
  __REG32                 : 5;
  __REG32  PRI_37         : 3;
  __REG32                 : 5;
  __REG32  PRI_38         : 3;
  __REG32                 : 5;
  __REG32  PRI_39         : 3;
} __pri9_bits;

/* Interrupt Priority Registers 40-43 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_40         : 3;
  __REG32                 : 5;
  __REG32  PRI_41         : 3;
  __REG32                 : 5;
  __REG32  PRI_42         : 3;
  __REG32                 : 5;
  __REG32  PRI_43         : 3;
} __pri10_bits;

/* Interrupt Priority Registers 44-47 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_44         : 3;
  __REG32                 : 5;
  __REG32  PRI_45         : 3;
  __REG32                 : 5;
  __REG32  PRI_46         : 3;
  __REG32                 : 5;
  __REG32  PRI_47         : 3;
} __pri11_bits;

/* Interrupt Priority Registers 48-51 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_48         : 3;
  __REG32                 : 5;
  __REG32  PRI_49         : 3;
  __REG32                 : 5;
  __REG32  PRI_50         : 3;
  __REG32                 : 5;
  __REG32  PRI_51         : 3;
} __pri12_bits;

/* Interrupt Priority Registers 52-55 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_52         : 3;
  __REG32                 : 5;
  __REG32  PRI_53         : 3;
  __REG32                 : 5;
  __REG32  PRI_54         : 3;
  __REG32                 : 5;
  __REG32  PRI_55         : 3;
} __pri13_bits;

/* Interrupt Priority Registers 56 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_56         : 3;
  __REG32                 :24;
} __pri14_bits;

/* Vector Table Offset Register */
typedef struct {
  __REG32                 : 7;
  __REG32  TBLOFF         :22;
  __REG32  TBLBASE        : 1;
  __REG32                 : 2;
} __vtor_bits;

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
__IO_REG32_BIT(SYSTICKCSR,        0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,        0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,        0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,      0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,           0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(SETENA1,           0xE000E104,__READ_WRITE ,__setena1_bits);
__IO_REG32_BIT(CLRENA0,           0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,           0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(SETPEND0,          0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,          0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(CLRPEND0,          0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,          0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(IP0,               0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,               0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,               0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,               0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,               0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,               0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,               0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,               0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(IP8,               0xE000E420,__READ_WRITE ,__pri8_bits);
__IO_REG32_BIT(IP9,               0xE000E424,__READ_WRITE ,__pri9_bits);
__IO_REG32_BIT(IP10,              0xE000E428,__READ_WRITE ,__pri10_bits);
__IO_REG32_BIT(IP11,              0xE000E42C,__READ_WRITE ,__pri11_bits);
__IO_REG32_BIT(IP12,              0xE000E430,__READ_WRITE ,__pri12_bits);
__IO_REG32_BIT(IP13,              0xE000E434,__READ_WRITE ,__pri13_bits);
__IO_REG32_BIT(IP14,              0xE000E438,__READ_WRITE ,__pri14_bits);
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(SHPR0,             0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,             0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,             0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);

/***************************************************************************
 **
 ** LVD
 **
 ***************************************************************************/
__IO_REG32_BIT(LVDCR1,              0x400F0500, __READ_WRITE , __vdcr_bits);
__IO_REG32_BIT(LVDST1,              0x400F0504, __READ       , __vdsr_bits);

/***************************************************************************
 **
 ** CG
 **
 ***************************************************************************/
__IO_REG32_BIT(CGSYSCR,             0x400F0200,__READ_WRITE ,__cgsyscr_bits);
__IO_REG32_BIT(CGOSCCR,             0x400F0204,__READ_WRITE ,__cgosccr_bits);
__IO_REG32_BIT(CGSTBYCR,            0x400F0208,__READ_WRITE ,__cgstbycr_bits);
__IO_REG32_BIT(CGCKSEL,             0x400F0210,__READ_WRITE ,__cgcksel_bits);
__IO_REG32_BIT(CGICRCG,             0x400F0214,__WRITE      ,__cgicrcg_bits);
__IO_REG32_BIT(CGNMIFLG,            0x400F0218,__READ       ,__cgnmiflg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x400F021C,__READ_WRITE ,__cgrstflg_bits);
__IO_REG32_BIT(CGIMCGA,             0x400F0220,__READ_WRITE ,__cgimcga_bits);
__IO_REG32_BIT(CGIMCGB,             0x400F0224,__READ_WRITE ,__cgimcgb_bits);
__IO_REG32_BIT(CGIMCGC,             0x400F0228,__READ_WRITE ,__cgimcgc_bits);
__IO_REG32_BIT(CGIMCGD,             0x400F022C,__READ_WRITE ,__cgimcgd_bits);
__IO_REG32_BIT(CGIMCGE,             0x400F0230,__READ_WRITE ,__cgimcge_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x400C0000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x400C0004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAFR1,                0x400C0008,__READ_WRITE ,__pafr1_bits);
__IO_REG8_BIT(PAPUP,                0x400C002C,__READ_WRITE ,__papup_bits);
__IO_REG8_BIT(PAPDN,                0x400C0030,__READ_WRITE ,__papdn_bits);
__IO_REG8_BIT(PAIE,                 0x400C0038,__READ_WRITE ,__paie_bits);

/***************************************************************************
 **
 ** PORTB
 **
 ***************************************************************************/
__IO_REG8_BIT(PBDATA,               0x400C0040,__READ_WRITE ,__pb_bits);
__IO_REG8_BIT(PBCR,                 0x400C0044,__READ_WRITE ,__pbcr_bits);
__IO_REG8_BIT(PBFR1,                0x400C0048,__READ_WRITE ,__pbfr1_bits);
__IO_REG8_BIT(PBPUP,                0x400C006C,__READ_WRITE ,__pbpup_bits);
__IO_REG8_BIT(PBIE,                 0x400C0078,__READ_WRITE ,__pbie_bits);

/***************************************************************************
 **
 ** PORTC
 **
 ***************************************************************************/
__IO_REG8_BIT(PCDATA,               0x400C0080,__READ       ,__pc_bits);
__IO_REG8_BIT(PCPUP,                0x400C00AC,__READ_WRITE ,__pcpup_bits);
__IO_REG8_BIT(PCIE,                 0x400C00B8,__READ_WRITE ,__pcie_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG8_BIT(PDDATA,               0x400C00C0,__READ       ,__pd_bits);
__IO_REG8_BIT(PDFR1,                0x400C00C8,__READ_WRITE ,__pdfr1_bits);
__IO_REG8_BIT(PDPUP,                0x400C00EC,__READ_WRITE ,__pdpup_bits);
__IO_REG8_BIT(PDIE,                 0x400C00F8,__READ_WRITE ,__pdie_bits);

/***************************************************************************
 **
 ** PORTE
 **
 ***************************************************************************/
__IO_REG8_BIT(PEDATA,               0x400C0100,__READ_WRITE ,__pe_bits);
__IO_REG8_BIT(PECR,                 0x400C0104,__READ_WRITE ,__pecr_bits);
__IO_REG8_BIT(PEFR1,                0x400C0108,__READ_WRITE ,__pefr1_bits);
__IO_REG8_BIT(PEFR2,                0x400C010C,__READ_WRITE ,__pefr2_bits);
__IO_REG8_BIT(PEOD,                 0x400C0128,__READ_WRITE ,__peod_bits);
__IO_REG8_BIT(PEPUP,                0x400C012C,__READ_WRITE ,__pepup_bits);
__IO_REG8_BIT(PEIE,                 0x400C0138,__READ_WRITE ,__peie_bits);

/***************************************************************************
 **
 ** PORTF
 **
 ***************************************************************************/
__IO_REG8_BIT(PFDATA,               0x400C0140,__READ_WRITE ,__pf_bits);
__IO_REG8_BIT(PFCR,                 0x400C0144,__READ_WRITE ,__pfcr_bits);
__IO_REG8_BIT(PFFR1,                0x400C0148,__READ_WRITE ,__pffr1_bits);
__IO_REG8_BIT(PFFR2,                0x400C014C,__READ_WRITE ,__pffr2_bits);
__IO_REG8_BIT(PFOD,                 0x400C0168,__READ_WRITE ,__pfod_bits);
__IO_REG8_BIT(PFPUP,                0x400C016C,__READ_WRITE ,__pfpup_bits);
__IO_REG8_BIT(PFIE,                 0x400C0178,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG8_BIT(PGDATA,               0x400C0180,__READ_WRITE ,__pg_bits);
__IO_REG8_BIT(PGCR,                 0x400C0184,__READ_WRITE ,__pgcr_bits);
__IO_REG8_BIT(PGFR1,                0x400C0188,__READ_WRITE ,__pgfr1_bits);
__IO_REG8_BIT(PGOD,                 0x400C01A8,__READ_WRITE ,__pgod_bits);
__IO_REG8_BIT(PGPUP,                0x400C01AC,__READ_WRITE ,__pgpup_bits);
__IO_REG8_BIT(PGIE,                 0x400C01B8,__READ_WRITE ,__pgie_bits);

/***************************************************************************
 **
 ** PORTH
 **
 ***************************************************************************/
__IO_REG8_BIT(PHDATA,               0x400C01C0,__READ_WRITE ,__ph_bits);
__IO_REG8_BIT(PHCR,                 0x400C01C4,__READ_WRITE ,__phcr_bits);
__IO_REG8_BIT(PHFR1,                0x400C01C8,__READ_WRITE ,__phfr1_bits);
__IO_REG8_BIT(PHOD,                 0x400C01E8,__READ_WRITE ,__phod_bits);
__IO_REG8_BIT(PHPUP,                0x400C01EC,__READ_WRITE ,__phpup_bits);
__IO_REG8_BIT(PHIE,                 0x400C01F8,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTI
 **
 ***************************************************************************/
__IO_REG8_BIT(PIDATA,               0x400C0200,__READ_WRITE ,__pi_bits);
__IO_REG8_BIT(PICR,                 0x400C0204,__READ_WRITE ,__picr_bits);
__IO_REG8_BIT(PIFR1,                0x400C0208,__READ_WRITE ,__pifr1_bits);
__IO_REG8_BIT(PIOD,                 0x400C0228,__READ_WRITE ,__piod_bits);
__IO_REG8_BIT(PIPUP,                0x400C022C,__READ_WRITE ,__pipup_bits);
__IO_REG8_BIT(PIIE,                 0x400C0238,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x400C0240,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJCR,                 0x400C0244,__READ_WRITE ,__pjcr_bits);
__IO_REG8_BIT(PJFR1,                0x400C0248,__READ_WRITE ,__pjfr1_bits);
__IO_REG8_BIT(PJPUP,                0x400C026C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJIE,                 0x400C0278,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG8_BIT(PKDATA,               0x400C0280,__READ_WRITE ,__pk_bits);
__IO_REG8_BIT(PKCR,                 0x400C0284,__READ_WRITE ,__pkcr_bits);
__IO_REG8_BIT(PKFR1,                0x400C0288,__READ_WRITE ,__pkfr1_bits);
__IO_REG8_BIT(PKFR2,                0x400C028C,__READ_WRITE ,__pkfr2_bits);
__IO_REG8_BIT(PKFR3,                0x400C0290,__READ_WRITE ,__pkfr3_bits);
__IO_REG8_BIT(PKPUP,                0x400C02AC,__READ_WRITE ,__pkpup_bits);
__IO_REG8_BIT(PKIE,                 0x400C02B8,__READ_WRITE ,__pkie_bits);

/***************************************************************************
 **
 ** PORTL
 **
 ***************************************************************************/
__IO_REG8_BIT(PLDATA,               0x400C02C0,__READ_WRITE ,__pl_bits);
__IO_REG8_BIT(PLCR,                 0x400C02C4,__READ_WRITE ,__plcr_bits);
__IO_REG8_BIT(PLFR1,                0x400C02C8,__READ_WRITE ,__plfr1_bits);
__IO_REG8_BIT(PLOD,                 0x400C02E8,__READ_WRITE ,__plod_bits);
__IO_REG8_BIT(PLPUP,                0x400C02EC,__READ_WRITE ,__plpup_bits);
__IO_REG8_BIT(PLIE,                 0x400C02F8,__READ_WRITE ,__plie_bits);

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
__IO_REG32_BIT(TB0IM,               0x400D0018, __READ       , __tbxim_bits);
__IO_REG32_BIT(TB0UC,               0x400D001C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB0RG0,              0x400D0020, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB0RG1,              0x400D0024, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB0CP0,              0x400D0028, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB0CP1,              0x400D002C, __READ       , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB1
 **
 ***************************************************************************/
__IO_REG32_BIT(TB1EN,               0x400D0040, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB1RUN,              0x400D0044, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB1CR,               0x400D0048, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB1MOD,              0x400D004C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB1FFCR,             0x400D0050, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB1ST,               0x400D0054, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB1IM,               0x400D0058, __READ       , __tbxim_bits);
__IO_REG32_BIT(TB1UC,               0x400D005C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB1RG0,              0x400D0060, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB1RG1,              0x400D0064, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB1CP0,              0x400D0068, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB1CP1,              0x400D006C, __READ       , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB2
 **
 ***************************************************************************/
__IO_REG32_BIT(TB2EN,               0x400D0080, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB2RUN,              0x400D0084, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB2CR,               0x400D0088, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB2MOD,              0x400D008C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB2FFCR,             0x400D0090, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB2ST,               0x400D0094, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB2IM,               0x400D0098, __READ       , __tbxim_bits);
__IO_REG32_BIT(TB2UC,               0x400D009C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB2RG0,              0x400D00A0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB2RG1,              0x400D00A4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB2CP0,              0x400D00A8, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB2CP1,              0x400D00AC, __READ       , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB3
 **
 ***************************************************************************/
__IO_REG32_BIT(TB3EN,               0x400D00C0, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB3RUN,              0x400D00C4, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB3CR,               0x400D00C8, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB3MOD,              0x400D00CC, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB3FFCR,             0x400D00D0, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB3ST,               0x400D00D4, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB3IM,               0x400D00D8, __READ       , __tbxim_bits);
__IO_REG32_BIT(TB3UC,               0x400D00DC, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB3RG0,              0x400D00E0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB3RG1,              0x400D00E4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB3CP0,              0x400D00E8, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB3CP1,              0x400D00EC, __READ       , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB4
 **
 ***************************************************************************/
__IO_REG32_BIT(TB4EN,               0x400D0100, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB4RUN,              0x400D0104, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB4CR,               0x400D0108, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB4MOD,              0x400D010C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB4FFCR,             0x400D0110, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB4ST,               0x400D0114, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB4IM,               0x400D0118, __READ       , __tbxim_bits);
__IO_REG32_BIT(TB4UC,               0x400D011C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB4RG0,              0x400D0120, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB4RG1,              0x400D0124, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB4CP0,              0x400D0128, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB4CP1,              0x400D012C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB5
 **
 ***************************************************************************/
__IO_REG32_BIT(TB5EN,               0x400D0140, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB5RUN,              0x400D0144, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB5CR,               0x400D0148, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB5MOD,              0x400D014C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB5FFCR,             0x400D0150, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB5ST,               0x400D0154, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB5IM,               0x400D0158, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB5UC,               0x400D015C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB5RG0,              0x400D0160, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB5RG1,              0x400D0164, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB5CP0,              0x400D0168, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB5CP1,              0x400D016C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB6
 **
 ***************************************************************************/
__IO_REG32_BIT(TB6EN,               0x400D0180, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB6RUN,              0x400D0184, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB6CR,               0x400D0188, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB6MOD,              0x400D018C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB6FFCR,             0x400D0190, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB6ST,               0x400D0194, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB6IM,               0x400D0198, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB6UC,               0x400D019C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB6RG0,              0x400D01A0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB6RG1,              0x400D01A4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB6CP0,              0x400D01A8, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB6CP1,              0x400D01AC, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB7
 **
 ***************************************************************************/
__IO_REG32_BIT(TB7EN,               0x400D01C0, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB7RUN,              0x400D01C4, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB7CR,               0x400D01C8, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB7MOD,              0x400D01CC, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB7FFCR,             0x400D01D0, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB7ST,               0x400D01D4, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB7IM,               0x400D01D8, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB7UC,               0x400D01DC, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB7RG0,              0x400D01E0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB7RG1,              0x400D01E4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB7CP0,              0x400D01E8, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB7CP1,              0x400D01EC, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB8
 **
 ***************************************************************************/
__IO_REG32_BIT(TB8EN,               0x400D0200, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB8RUN,              0x400D0204, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB8CR,               0x400D0208, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB8MOD,              0x400D020C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB8FFCR,             0x400D0210, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB8ST,               0x400D0214, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB8IM,               0x400D0218, __READ       , __tbxim_bits);
__IO_REG32_BIT(TB8UC,               0x400D021C, __READ       , __tbxuc_bits);
__IO_REG32_BIT(TB8RG0,              0x400D0220, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB8RG1,              0x400D0224, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB8CP0,              0x400D0228, __READ       , __tbxcp0_bits);
__IO_REG32_BIT(TB8CP1,              0x400D022C, __READ       , __tbxcp1_bits);

/***************************************************************************
 **
 ** PHCNT 0
 **
 ***************************************************************************/
__IO_REG32_BIT(PHC0RUN,             0x400D0240,  __READ_WRITE , __phcxrun_bits);
__IO_REG32_BIT(PHC0CR,              0x400D0244,  __READ_WRITE , __phcxcr_bits);
__IO_REG32_BIT(PHC0EN,              0x400D0248,  __READ_WRITE , __phcxen_bits);
__IO_REG32_BIT(PHC0FLG,             0x400D024C,  __READ_WRITE , __phcxflg_bits);
__IO_REG32_BIT(PHC0CMP0,            0x400D0250,  __READ_WRITE , __phcxcmp0_bits);
__IO_REG32_BIT(PHC0CMP1,            0x400D0254,  __READ_WRITE , __phcxcmp1_bits);
__IO_REG32_BIT(PHC0CNT,             0x400D0258,  __READ       , __phcxcnt_bits);        

/***************************************************************************
 **
 ** SIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(SC0EN,               0x400E0080, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC0BUF,              0x400E0084, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC0CR,               0x400E0088, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC0MOD0,             0x400E008C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC0BRCR,             0x400E0090, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC0BRADD,            0x400E0094, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC0MOD1,             0x400E0098, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC0MOD2,             0x400E009C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC0RFC,              0x400E00A0, __READ_WRITE , __sc0rfc_bits);
__IO_REG32_BIT(SC0TFC,              0x400E00A4, __READ_WRITE , __sc0tfc_bits);
__IO_REG32_BIT(SC0RST,              0x400E00A8, __READ       , __sc0rst_bits);
__IO_REG32_BIT(SC0TST,              0x400E00AC, __READ       , __sc0tst_bits);
__IO_REG32_BIT(SC0FCNF,             0x400E00B0, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(SC1EN,               0x400E00C0, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC1BUF,              0x400E00C4, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC1CR,               0x400E00C8, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC1MOD0,             0x400E00CC, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC1BRCR,             0x400E00D0, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC1BRADD,            0x400E00D4, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC1MOD1,             0x400E00D8, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC1MOD2,             0x400E00DC, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC1RFC,              0x400E00E0, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC1TFC,              0x400E00E4, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC1RST,              0x400E00E8, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC1TST,              0x400E00EC, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC1FCNF,             0x400E00F0, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(SC2EN,               0x400E0100, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC2BUF,              0x400E0104, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC2CR,               0x400E0108, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC2MOD0,             0x400E010C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC2BRCR,             0x400E0110, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC2BRADD,            0x400E0114, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC2MOD1,             0x400E0118, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC2MOD2,             0x400E011C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC2RFC,              0x400E0120, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC2TFC,              0x400E0124, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC2RST,              0x400E0128, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC2TST,              0x400E012C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC2FCNF,             0x400E0130, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SBI1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CR0,             0x400E0000, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C1CR1,             0x400E0004, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C1DBR,             0x400E0008, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C1AR,              0x400E000C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C1CR2,             0x400E0010, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C1SR      I2C1CR2
#define I2C1SR_bit  I2C1CR2_bit.__sr
__IO_REG32_BIT(I2C1BR0,             0x400E0014, __READ_WRITE , __sbixbr0_bits);
__IO_REG32_BIT(I2C1INTSEL,          0x400E1400, __READ_WRITE , __sbixintsel_bits);

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
#define SIO1INTSEL  I2C1INTSEL
#define SIO1INTSEL_bit I2C1INTSEL_bit

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CR1,             0x40070000, __READ_WRITE , __i2ccr1_bits);
__IO_REG32_BIT(I2C0DBR,             0x40070004, __READ_WRITE , __i2cdbr_bits);
__IO_REG32_BIT(I2C0AR,              0x40070008, __READ_WRITE , __i2car_bits);
__IO_REG32_BIT(I2C0CR2,             0x4007000C, __READ_WRITE , __i2ccr2_bits);
#define I2C0SR      I2C0CR2
#define I2C0SR_bit  I2C0CR2_bit.__sr
__IO_REG32_BIT(I2C0PRS,             0x40070010, __READ_WRITE , __i2cprs_bits);
__IO_REG32_BIT(I2CIE,               0x40070014, __READ_WRITE , __i2cie_bits);
__IO_REG32_BIT(I2CIR,               0x40070018, __READ_WRITE , __i2cir_bits);

/***************************************************************************
 **
 ** SSP 0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,             0x40060000, __READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,             0x40060004, __READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSP0DR,              0x40060008, __READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSP0SR,              0x4006000C, __READ       , __sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,            0x40060010, __READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,            0x40060014, __READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,             0x40060018, __READ       , __sspris_bits);
__IO_REG32_BIT(SSP0MIS,             0x4006001C, __READ       , __sspmis_bits);
__IO_REG32_BIT(SSP0ICR,             0x40060020, __WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSP0INTSEL,          0x400E1000, __READ_WRITE , __sspintsel_bits);

/***************************************************************************
 **
 ** CEC
 **
 ***************************************************************************/
__IO_REG32_BIT(CECEN,              0x400F0300, __READ_WRITE , __cecen_bits );
__IO_REG32_BIT(CECADD,             0x400F0304, __READ_WRITE , __cecadd_bits);
__IO_REG32_BIT(CECRESET,           0x400F0308, __WRITE      , __cecreset_bits);
__IO_REG32_BIT(CECREN,             0x400F030C, __READ_WRITE , __cecren_bits);
__IO_REG32_BIT(CECRBUF,            0x400F0310, __READ       , __cecrbuf_bits);
__IO_REG32_BIT(CECRCR1,            0x400F0314, __READ_WRITE , __cecrcr1_bits);
__IO_REG32_BIT(CECRCR2,            0x400F0318, __READ_WRITE , __cecrcr2_bits);
__IO_REG32_BIT(CECRCR3,            0x400F031C, __READ_WRITE , __cecrcr3_bits);
__IO_REG32_BIT(CECTEN,             0x400F0320, __READ_WRITE , __cecten_bits);
__IO_REG32_BIT(CECTBUF,            0x400F0324, __READ_WRITE , __cectbuf_bits);
__IO_REG32_BIT(CECTCR,             0x400F0328, __READ_WRITE , __cectcr_bits);
__IO_REG32_BIT(CECRSTAT,           0x400F032C, __READ       , __cecrstat_bits);
__IO_REG32_BIT(CECTSTAT,           0x400F0330, __READ       , __cectstat_bits);

/***************************************************************************
 **
 ** RMC0
 **
 ***************************************************************************/
__IO_REG32_BIT(RMC0EN,             0x400F0400, __READ_WRITE , __rmcen_bits);
__IO_REG32_BIT(RMC0REN,            0x400F0404, __READ_WRITE , __rmcren_bits);
__IO_REG32(    RMC0RBUF1,          0x400F0408, __READ);
__IO_REG32(    RMC0RBUF2,          0x400F040C, __READ);
__IO_REG32(    RMC0RBUF3,          0x400F0410, __READ);
__IO_REG32_BIT(RMC0RCR1,           0x400F0414, __READ_WRITE , __rmcrcr1_bits );
__IO_REG32_BIT(RMC0RCR2,           0x400F0418, __READ_WRITE , __rmcrcr2_bits );
__IO_REG32_BIT(RMC0RCR3,           0x400F041C, __READ_WRITE , __rmcrcr3_bits );
__IO_REG32_BIT(RMC0RCR4,           0x400F0420, __READ_WRITE , __rmcrcr4_bits );
__IO_REG32_BIT(RMC0RSTAT,          0x400F0424, __READ       , __rmcrstat_bits);
__IO_REG32_BIT(RMC0END1,           0x400F0428, __READ_WRITE , __rmcend_bits);
__IO_REG32_BIT(RMC0END2,           0x400F042C, __READ_WRITE , __rmcend_bits );
__IO_REG32_BIT(RMC0END3,           0x400F0430, __READ_WRITE , __rmcend_bits );

/***************************************************************************
 **
 ** RMC1
 **
 ***************************************************************************/
__IO_REG32_BIT(RMC1EN,             0x400F0440, __READ_WRITE , __rmcen_bits );
__IO_REG32_BIT(RMC1REN,            0x400F0444, __READ_WRITE , __rmcren_bits );
__IO_REG32(    RMC1RBUF1,          0x400F0448, __READ);
__IO_REG32(    RMC1RBUF2,          0x400F044C, __READ);
__IO_REG32(    RMC1RBUF3,          0x400F0450, __READ);
__IO_REG32_BIT(RMC1RCR1,           0x400F0454, __READ_WRITE , __rmcrcr1_bits );
__IO_REG32_BIT(RMC1RCR2,           0x400F0458, __READ_WRITE , __rmcrcr2_bits );
__IO_REG32_BIT(RMC1RCR3,           0x400F045C, __READ_WRITE , __rmcrcr3_bits );
__IO_REG32_BIT(RMC1RCR4,           0x400F0460, __READ_WRITE , __rmcrcr4_bits );
__IO_REG32_BIT(RMC1RSTAT,          0x400F0464, __READ       , __rmcrstat_bits );
__IO_REG32_BIT(RMC1END1,           0x400F0468, __READ_WRITE , __rmcend_bits );
__IO_REG32_BIT(RMC1END2,           0x400F046C, __READ_WRITE , __rmcend_bits );
__IO_REG32_BIT(RMC1END3,           0x400F0470, __READ_WRITE , __rmcend_bits );

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
/* __IO_REG16(   ADCBAS,                0x400F0020,__READ_WRITE ); */
__IO_REG16_BIT(ADREG08,              0x400F0030,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG19,              0x400F0034,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG2A,              0x400F0038,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG3B,              0x400F003C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG4C,              0x400F0040,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG5D,              0x400F0044,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG6E,              0x400F0048,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG7F,              0x400F004C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG88,              0x400F0050,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG99,              0x400F0054,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREGAA,              0x400F0058,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREGBB,              0x400F005C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREGSP,              0x400F0060,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADCMP0,               0x400F0064,__READ_WRITE ,__adcmpx_bits);
__IO_REG16_BIT(ADCMP1,               0x400F0068,__READ_WRITE ,__adcmpx_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG8_BIT(WDMOD,                0x400F0080,__READ_WRITE ,__wdmod_bits);
__IO_REG8(    WDCR,                 0x400F0084,__WRITE);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG8_BIT(RTCSECR,              0x400F0100,__READ_WRITE ,__rtcsecr_bits);
__IO_REG8_BIT(RTCMINR,              0x400F0101,__READ_WRITE ,__rtcminr_bits);
__IO_REG8_BIT(RTCHOURR,             0x400F0102,__READ_WRITE ,__rtchourr_bits);
__IO_REG8_BIT(RTCDAYR,              0x400F0104,__READ_WRITE ,__rtcdayr_bits);
__IO_REG8_BIT(RTCDATER,             0x400F0105,__READ_WRITE ,__rtcdater_bits);
__IO_REG8_BIT(RTCMONTHR,            0x400F0106,__READ_WRITE ,__rtcmonthr_bits);
__IO_REG8_BIT(RTCYEARR,             0x400F0107,__READ_WRITE ,__rtcyearr_bits);
__IO_REG8_BIT(RTCPAGER,             0x400F0108,__READ_WRITE ,__rtcpager_bits);
__IO_REG8_BIT(RTCSTA,               0x400F0109,__READ_WRITE ,__rtcsta_bits);
__IO_REG8_BIT(RTCRESTR,             0x400F010C,__WRITE      ,__rtcrestr_bits);
__IO_REG8_BIT(RTCADJCTL,            0x400F010E,__READ_WRITE ,__rtcadjctl_bits);
__IO_REG8(    RTCADJDAT,            0x400F010F,__READ_WRITE );

/***************************************************************************
 **
 ** OFD
 **
 ***************************************************************************/
__IO_REG32_BIT(OFDCR1,              0x400F0600, __READ_WRITE ,__ofdcr1_bits);
__IO_REG32_BIT(OFDCR2,              0x400F0604, __READ_WRITE ,__ofdcr2_bits);
__IO_REG32_BIT(OFDMN,               0x400F0608, __READ_WRITE ,__ofdmn_bits);
__IO_REG32_BIT(OFDMX,               0x400F0610, __READ_WRITE ,__ofdmx_bits);

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
 **  TMPM390FWFG Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK             0          /* Main Stack                   */
#define RESETI                 1          /* Reset                        */
#define NMII                   2          /* Non-maskable Interrupt       */
#define HFI                    3          /* Hard Fault                   */
#define MMI                    4          /* Memory Management            */
#define BFI                    5          /* Bus Fault                    */
#define UFI                    6          /* Usage Fault                  */
#define SVCI                  11          /* SVCall                       */
#define DMI                   12          /* Debug Monitor                */
#define PSI                   14          /* PendSV                       */
#define STI                   15          /* SysTick                      */
#define EII                   16          /* External Interrupt           */
#define INT_0                ( 0 + EII)   /* Interrupt Pin Interrupt pin (PJ0)*/
#define INT_1                ( 1 + EII)   /* Interrupt Pin Interrupt pin (PJ1)*/
#define INT_2                ( 2 + EII)   /* Interrupt Pin Interrupt pin (PJ2)*/
#define INT_3                ( 3 + EII)   /* Interrupt Pin Interrupt pin (PJ3)*/
#define INT_4                ( 4 + EII)   /* Interrupt Interrupt pin (PJ4)*/
#define INT_5                ( 5 + EII)   /* Interrupt Interrupt pin (PJ5)*/
#define INT_RX0              ( 6 + EII)   /* Serial reception (channel.0) */
#define INT_TX0              ( 7 + EII)   /* Serial transmit (channel.0)  */
#define INT_RX1              ( 8 + EII)   /* Serial reception (channel.1) */
#define INT_TX1              ( 9 + EII)   /* Serial transmit (channel.1)  */
#define INT_I2C0             (10 + EII)   /* Serial bus interface 0       */
#define INT_SBI1             (11 + EII)   /* Serial bus interface 1       */
#define INT_CECRX            (12 + EII)   /* CEC reception                */
#define INT_CECTX            (13 + EII)   /* CEC transmission             */
#define INT_RMCRX0           (14 + EII)   /* Remote control signal reception (channel.0)*/
#define INT_ADHP             (15 + EII)   /* Highest priority AD conversion complete interrupt*/
#define INT_ADM0             (16 + EII)   /* AD conversion monitoring function interrupt 0*/
#define INT_ADM1             (17 + EII)   /* AD conversion monitoring function interrupt 1*/
#define INT_TB0              (18 + EII)   /* 16bit TMRB match detection 0 */
#define INT_TB1              (19 + EII)   /* 16bit TMRB match detection 1 */
#define INT_TB2              (20 + EII)   /* 16bit TMRB match detection 2 */
#define INT_TB3              (21 + EII)   /* 16bit TMRB match detection 3 */
#define INT_TB4              (22 + EII)   /* 16bit TMRB match detection 4 */
#define INT_TB5              (23 + EII)   /* 16bit TMRB match detection 5 */
#define INT_TB6              (24 + EII)   /* 16bit TMRB match detection 6 */
#define INT_RTC              (25 + EII)   /* Real time clock timer        */
#define INT_CAP00            (26 + EII)   /* 16bit TMRB input capture 00  */
#define INT_CAP01            (27 + EII)   /* 16bit TMRB input capture 01  */
#define INT_CAP10            (28 + EII)   /* 16bit TMRB input capture 10  */
#define INT_CAP11            (29 + EII)   /* 16bit TMRB input capture 11  */
#define INT_CAP50            (30 + EII)   /* 16bit TMRB input capture 50  */
#define INT_CAP51            (31 + EII)   /* 16bit TMRB input capture 51  */
#define INT_CAP60            (32 + EII)   /* 16bit TMRB input capture 60  */
#define INT_CAP61            (33 + EII)   /* 16bit TMRB input capture 61  */
#define INT_6                (34 + EII)   /* Interrupt pin (PJ6/39pin)    */
#define INT_7                (35 + EII)   /* Interrupt pin (PJ7/58pin)    */
#define INT_RX2              (36 + EII)   /* Serial reception (channel.2) */
#define INT_TX2              (37 + EII)   /* Serial transmission (channel.2)*/
#define INT_LVD              (38 + EII)   /* Low voltage detection*/
#define INT_RMCRX1           (39 + EII)   /* Remote control signal reception (channel.1)*/
#define INT_TB7              (40 + EII)   /* 16bit TMRB match detection 7 */
#define INT_TB8              (41 + EII)   /* 16bit TMRB match detection 8 */
#define INT_PHT              (42 + EII)   /* 16bit TMRB (two phase pulse input counter)*/
#define INT_CAP20            (43 + EII)   /* 16bit TMRB input capture 20  */
#define INT_CAP21            (44 + EII)   /* 16bit TMRB input capture 21  */
#define INT_CAP30            (45 + EII)   /* 16bit TMRB input capture 30  */
#define INT_CAP31            (46 + EII)   /* 16bit TMRB input capture 31  */
#define INT_AD               (49 + EII)   /* A/D conversion completion    */
#define INT_SPI0             (53 + EII)   /* SPI serial interface (channel.0)*/

#endif    /* __IOTMPM390FWFG_H */

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
Interrupt15  = INTRX0         0x58
Interrupt16  = INTTX0         0x5C
Interrupt17  = INTRX1         0x60
Interrupt18  = INTTX1         0x64
Interrupt19  = I2CINT0        0x68
Interrupt20  = I2CINT0        0x6C
Interrupt21  = INTCECRX       0x70
Interrupt22  = INTCECTX       0x74
Interrupt23  = INTRMCRX0      0x78
Interrupt24  = INTADHP        0x7C
Interrupt25  = INTADM0        0x80
Interrupt26  = INTADM1        0x84
Interrupt27  = INTTB0         0x88
Interrupt28  = INTTB1         0x8C
Interrupt29  = INTTB2         0x90
Interrupt30  = INTTB3         0x94
Interrupt31  = INTTB4         0x98
Interrupt32  = INTTB5         0x9C
Interrupt33  = INTTB6         0xA0
Interrupt34  = INTRTC         0xA4
Interrupt35  = INTCAP00       0xA8
Interrupt36  = INTCAP01       0xAC
Interrupt37  = INTCAP10       0xB0
Interrupt38  = INTCAP11       0xB4
Interrupt39  = INTCAP50       0xB8
Interrupt40  = INTCAP51       0xBC
Interrupt41  = INTCAP60       0xC0
Interrupt42  = INTCAP61       0xC4
Interrupt43  = INT6           0xC8
Interrupt44  = INT7           0xCC
Interrupt45  = INTRX2         0xD0
Interrupt46  = INTTX2         0xD4
Interrupt47  = INTLVD         0xD8
Interrupt48  = INTRMCRX1      0xDC
Interrupt49  = INTTB7         0xE0
Interrupt50  = INTTB8         0xE4
Interrupt51  = INTPHT         0xE8
Interrupt52  = INTCAP20       0xEC
Interrupt53  = INTCAP21       0xF0
Interrupt54  = INTCAP30       0xF4
Interrupt55  = INTCAP31       0xF8
Interrupt56  = INTAD          0x104
Interrupt57  = INTSPI0        0x114

###DDF-INTERRUPT-END###*/
