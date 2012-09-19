/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM333FxFG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 41221 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM333FxFG_H
#define __IOTMPM333FxFG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM333FxFG SPECIAL FUNCTION REGISTERS
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
  __REG8  PB4  : 1;
  __REG8  PB5  : 1;
  __REG8  PB6  : 1;
  __REG8  PB7  : 1;
} __pb_bits;

/*PORT B Control Register */
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
  __REG8  PB3UP  : 1;
  __REG8  PB4UP  : 1;
  __REG8  PB5UP  : 1;
  __REG8  PB6UP  : 1;
  __REG8  PB7UP  : 1;
} __pbpup_bits;

/*PORT B Input Enable Control Register */
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
  __REG8  PE4OD  : 1;
  __REG8  PE5OD  : 1;
  __REG8  PE6OD  : 1;
  __REG8         : 1;
} __peod_bits;

/*PORT E Pull-Up Control Register */
typedef struct {
  __REG8  PE0UP  : 1;
  __REG8  PE1UP  : 1;
  __REG8  PE2UP  : 1;
  __REG8  PE3UP  : 1;
  __REG8  PE4UP  : 1;
  __REG8  PE5UP  : 1;
  __REG8  PE6UP  : 1;
  __REG8         : 1;
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

/*PORT F Open Drain Control Register */
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
  __REG8  PG4  : 1;
  __REG8  PG5  : 1;
  __REG8  PG6  : 1;
  __REG8  PG7  : 1;
} __pg_bits;

/*PORT G Register*/
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

/*PORT G Function Register 1*/
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

/*PORT G Open Drain Control Register */
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

/*PORT G Pull-Up Control Register */
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

/*PORT G Input Enable Control Register */
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

/*PORT H Register*/
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

/*PORT H Control Register 1*/
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

/*PORT H Function Register 1*/
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

/*PORT H Pull-Up Control Register */
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

/*PORT H Input Enable Control Register */
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

/*PORT I Function Register 1*/
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
  __REG8  PK2  : 1;
  __REG8       : 5;
} __pk_bits;

/*PORT K Control Register*/
typedef struct {
  __REG8  PK0C  : 1;
  __REG8  PK1C  : 1;
  __REG8  PK2C  : 1;
  __REG8        : 5;
} __pkcr_bits;

/*PORT K Function Register 1*/
typedef struct {
  __REG8  PK0F1  : 1;
  __REG8  PK1F1  : 1;
  __REG8  PK2F1  : 1;
  __REG8         : 5;
} __pkfr1_bits;

/*PORT K Function Register 2*/
typedef struct {
  __REG8         : 1;
  __REG8  PK1F2  : 1;
  __REG8         : 6;
} __pkfr2_bits;

/*PORT K Pull-Up Control Register*/
typedef struct {
  __REG8         : 1;
  __REG8  PK1UP  : 1;
  __REG8  PK2UP  : 1;
  __REG8         : 5;
} __pkpup_bits;

/*PORT K Input Enable Control Register*/
typedef struct {
  __REG8  PK0IE  : 1;
  __REG8  PK1IE  : 1;
  __REG8  PK2IE  : 1;
  __REG8         : 5;
} __pkie_bits;

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

/*System Control Register*/
typedef struct {
  __REG32 GEAR    : 3;
  __REG32         : 5;
  __REG32 PRCK    : 3;
  __REG32         : 1;
  __REG32 FPSEL   : 1;
  __REG32         : 3;
  __REG32 SCOSEL  : 2;
  __REG32         :14;
} __syscr_bits;

/*Oscillation Control Register*/
typedef struct {
  __REG32 WUEON   : 1;
  __REG32 WUEF    : 1;
  __REG32 PLLON   : 1;
  __REG32 WUPSEL  : 1;
  __REG32 WUPT    : 3;
  __REG32         : 1;
  __REG32 XEN     : 1;
  __REG32 XTEN    : 1;
  __REG32         :22;
} __osccr_bits;

/*Standby Control Register*/
typedef struct {
  __REG32 STBY    : 3;
  __REG32         : 5;
  __REG32 RXEN    : 1;
  __REG32 RXTEN   : 1;
  __REG32         : 6;
  __REG32 DRVE    : 1;
  __REG32         :15;
} __stbycr_bits;

/*PLL Selection Register*/
typedef struct {
  __REG32 PLLSEL  : 1;
  __REG32         :31;
} __pllsel_bits;

/*System Clock Selection Register*/
typedef struct {
  __REG32 SYSCKFLG  : 1;
  __REG32 SYSCK     : 1;
  __REG32           :30;
} __cksel_bits;

/*INTCG Clear Register*/
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           :27;
} __icrcg_bits;

/*NMI Flag Register*/
typedef struct {
  __REG32 NMIFLG0   : 1;
  __REG32 NMIFLG1   : 1;
  __REG32           :30;
} __nmiflg_bits;

/*Reset Flag Register*/
typedef struct {
  __REG32 PONRSTF   : 1;
  __REG32 PINRSTF   : 1;
  __REG32 WDTRSTF   : 1;
  __REG32           : 1;
  __REG32 SYSRSTF   : 1;
  __REG32           :27;
} __rstflg_bits;

/*CG Interrupt Mode Control Register A*/
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
} __imcga_bits;

/*CG Interrupt Mode Control Register B*/
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
} __imcgb_bits;

/*CG Interrupt Mode Control Register C*/
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
} __imcgc_bits;

/*CG Interrupt Mode Control Register D*/
typedef struct {
  __REG32 INTCEN    : 1;
  __REG32           : 1;
  __REG32 EMSTC     : 2;
  __REG32 EMCGC     : 3;
  __REG32           :25;
} __imcgd_bits;

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
  __REG32         :10;
} __flcs_bits;

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
  __REG32  NONEBASETHRDENA: 1;
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

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,              0xE000E004,__READ       ,__nvic_bits);
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
__IO_REG32_BIT(ACTIVE0,           0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,           0xE000E304,__READ       ,__active1_bits);
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
__IO_REG32_BIT(IP15,              0xE000E43C,__READ_WRITE ,__pri15_bits);
__IO_REG32_BIT(CPUIDBR,           0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,              0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,             0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,               0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,               0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,             0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,             0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,             0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,              0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,              0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,              0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,             0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,              0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,              0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x40000000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x40000004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAFR1,                0x40000008,__READ_WRITE ,__pafr1_bits);
__IO_REG8_BIT(PAPUP,                0x4000002C,__READ_WRITE ,__papup_bits);
__IO_REG8_BIT(PAPDN,                0x40000030,__READ_WRITE ,__papdn_bits);
__IO_REG8_BIT(PAIE,                 0x40000038,__READ_WRITE ,__paie_bits);

/***************************************************************************
 **
 ** PORTB
 **
 ***************************************************************************/
__IO_REG8_BIT(PBDATA,               0x40000040,__READ_WRITE ,__pb_bits);
__IO_REG8_BIT(PBCR,                 0x40000044,__READ_WRITE ,__pbcr_bits);
__IO_REG8_BIT(PBFR1,                0x40000048,__READ_WRITE ,__pbfr1_bits);
__IO_REG8_BIT(PBPUP,                0x4000006C,__READ_WRITE ,__pbpup_bits);
__IO_REG8_BIT(PBIE,                 0x40000078,__READ_WRITE ,__pbie_bits);

/***************************************************************************
 **
 ** PORTC
 **
 ***************************************************************************/
__IO_REG8_BIT(PCDATA,               0x40000080,__READ_WRITE ,__pc_bits);
__IO_REG8_BIT(PCPUP,                0x400000AC,__READ_WRITE ,__pcpup_bits);
__IO_REG8_BIT(PCIE,                 0x400000B8,__READ_WRITE ,__pcie_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG8_BIT(PDDATA,               0x400000C0,__READ_WRITE ,__pd_bits);
__IO_REG8_BIT(PDFR1,                0x400000C8,__READ_WRITE ,__pdfr1_bits);
__IO_REG8_BIT(PDPUP,                0x400000EC,__READ_WRITE ,__pdpup_bits);
__IO_REG8_BIT(PDIE,                 0x400000F8,__READ_WRITE ,__pdie_bits);

/***************************************************************************
 **
 ** PORTE
 **
 ***************************************************************************/
__IO_REG8_BIT(PEDATA,               0x40000100,__READ_WRITE ,__pe_bits);
__IO_REG8_BIT(PECR,                 0x40000104,__READ_WRITE ,__pecr_bits);
__IO_REG8_BIT(PEFR1,                0x40000108,__READ_WRITE ,__pefr1_bits);
__IO_REG8_BIT(PEFR2,                0x4000010C,__READ_WRITE ,__pefr2_bits);
__IO_REG8_BIT(PEOD,                 0x40000128,__READ_WRITE ,__peod_bits);
__IO_REG8_BIT(PEPUP,                0x4000012C,__READ_WRITE ,__pepup_bits);
__IO_REG8_BIT(PEIE,                 0x40000138,__READ_WRITE ,__peie_bits);

/***************************************************************************
 **
 ** PORTF
 **
 ***************************************************************************/
__IO_REG8_BIT(PFDATA,               0x40000140,__READ_WRITE ,__pf_bits);
__IO_REG8_BIT(PFCR,                 0x40000144,__READ_WRITE ,__pfcr_bits);
__IO_REG8_BIT(PFFR1,                0x40000148,__READ_WRITE ,__pffr1_bits);
__IO_REG8_BIT(PFFR2,                0x4000014C,__READ_WRITE ,__pffr2_bits);
__IO_REG8_BIT(PFOD,                 0x40000168,__READ_WRITE ,__pfod_bits);
__IO_REG8_BIT(PFPUP,                0x4000016C,__READ_WRITE ,__pfpup_bits);
__IO_REG8_BIT(PFIE,                 0x40000178,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG8_BIT(PGDATA,               0x40000180,__READ_WRITE ,__pg_bits);
__IO_REG8_BIT(PGCR,                 0x40000184,__READ_WRITE ,__pgcr_bits);
__IO_REG8_BIT(PGFR1,                0x40000188,__READ_WRITE ,__pgfr1_bits);
__IO_REG8_BIT(PGOD,                 0x400001A8,__READ_WRITE ,__pgod_bits);
__IO_REG8_BIT(PGPUP,                0x400001AC,__READ_WRITE ,__pgpup_bits);
__IO_REG8_BIT(PGIE,                 0x400001B8,__READ_WRITE ,__pgie_bits);

/***************************************************************************
 **
 ** PORTH
 **
 ***************************************************************************/
__IO_REG8_BIT(PHDATA,               0x400001C0,__READ_WRITE ,__ph_bits);
__IO_REG8_BIT(PHCR,                 0x400001C4,__READ_WRITE ,__phcr_bits);
__IO_REG8_BIT(PHFR1,                0x400001C8,__READ_WRITE ,__phfr1_bits);
__IO_REG8_BIT(PHPUP,                0x400001EC,__READ_WRITE ,__phpup_bits);
__IO_REG8_BIT(PHIE,                 0x400001F8,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTI
 **
 ***************************************************************************/
__IO_REG8_BIT(PIDATA,               0x40000200,__READ_WRITE ,__pi_bits);
__IO_REG8_BIT(PICR,                 0x40000204,__READ_WRITE ,__picr_bits);
__IO_REG8_BIT(PIFR1,                0x40000208,__READ_WRITE ,__pifr1_bits);
__IO_REG8_BIT(PIPUP,                0x4000022C,__READ_WRITE ,__pipup_bits);
__IO_REG8_BIT(PIIE,                 0x40000238,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x40000240,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJCR,                 0x40000244,__READ_WRITE ,__pjcr_bits);
__IO_REG8_BIT(PJFR1,                0x40000248,__READ_WRITE ,__pjfr1_bits);
__IO_REG8_BIT(PJPUP,                0x4000026C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJIE,                 0x40000278,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG8_BIT(PKDATA,               0x40000280,__READ_WRITE ,__pk_bits);
__IO_REG8_BIT(PKCR,                 0x40000284,__READ_WRITE ,__pkcr_bits);
__IO_REG8_BIT(PKFR1,                0x40000288,__READ_WRITE ,__pkfr1_bits);
__IO_REG8_BIT(PKFR2,                0x4000028C,__READ_WRITE ,__pkfr2_bits);
__IO_REG8_BIT(PKPUP,                0x400002AC,__READ_WRITE ,__pkpup_bits);
__IO_REG8_BIT(PKIE,                 0x400002B8,__READ_WRITE ,__pkie_bits);

/***************************************************************************
 **
 ** TMRB0
 **
 ***************************************************************************/
__IO_REG32_BIT(TB0EN,               0x40010000, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB0RUN,              0x40010004, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB0CR,               0x40010008, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB0MOD,              0x4001000C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB0FFCR,             0x40010010, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB0ST,               0x40010014, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB0IM,               0x40010018, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB0UC,               0x4001001C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB0RG0,              0x40010020, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB0RG1,              0x40010024, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB0CP0,              0x40010028, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB0CP1,              0x4001002C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB1
 **
 ***************************************************************************/
__IO_REG32_BIT(TB1EN,               0x40010040, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB1RUN,              0x40010044, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB1CR,               0x40010048, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB1MOD,              0x4001004C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB1FFCR,             0x40010050, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB1ST,               0x40010054, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB1IM,               0x40010058, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB1UC,               0x4001005C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB1RG0,              0x40010060, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB1RG1,              0x40010064, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB1CP0,              0x40010068, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB1CP1,              0x4001006C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB2
 **
 ***************************************************************************/
__IO_REG32_BIT(TB2EN,               0x40010080, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB2RUN,              0x40010084, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB2CR,               0x40010088, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB2MOD,              0x4001008C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB2FFCR,             0x40010090, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB2ST,               0x40010094, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB2IM,               0x40010098, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB2UC,               0x4001009C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB2RG0,              0x400100A0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB2RG1,              0x400100A4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB2CP0,              0x400100A8, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB2CP1,              0x400100AC, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB3
 **
 ***************************************************************************/
__IO_REG32_BIT(TB3EN,               0x400100C0, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB3RUN,              0x400100C4, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB3CR,               0x400100C8, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB3MOD,              0x400100CC, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB3FFCR,             0x400100D0, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB3ST,               0x400100D4, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB3IM,               0x400100D8, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB3UC,               0x400100DC, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB3RG0,              0x400100E0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB3RG1,              0x400100E4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB3CP0,              0x400100E8, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB3CP1,              0x400100EC, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB4
 **
 ***************************************************************************/
__IO_REG32_BIT(TB4EN,               0x40010100, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB4RUN,              0x40010104, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB4CR,               0x40010108, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB4MOD,              0x4001010C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB4FFCR,             0x40010110, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB4ST,               0x40010114, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB4IM,               0x40010118, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB4UC,               0x4001011C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB4RG0,              0x40010120, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB4RG1,              0x40010124, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB4CP0,              0x40010128, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB4CP1,              0x4001012C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB5
 **
 ***************************************************************************/
__IO_REG32_BIT(TB5EN,               0x40010140, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB5RUN,              0x40010144, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB5CR,               0x40010148, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB5MOD,              0x4001014C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB5FFCR,             0x40010150, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB5ST,               0x40010154, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB5IM,               0x40010158, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB5UC,               0x4001015C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB5RG0,              0x40010160, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB5RG1,              0x40010164, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB5CP0,              0x40010168, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB5CP1,              0x4001016C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB6
 **
 ***************************************************************************/
__IO_REG32_BIT(TB6EN,               0x40010180, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB6RUN,              0x40010184, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB6CR,               0x40010188, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB6MOD,              0x4001018C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB6FFCR,             0x40010190, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB6ST,               0x40010194, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB6IM,               0x40010198, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB6UC,               0x4001019C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB6RG0,              0x400101A0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB6RG1,              0x400101A4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB6CP0,              0x400101A8, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB6CP1,              0x400101AC, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB7
 **
 ***************************************************************************/
__IO_REG32_BIT(TB7EN,               0x400101C0, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB7RUN,              0x400101C4, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB7CR,               0x400101C8, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB7MOD,              0x400101CC, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB7FFCR,             0x400101D0, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB7ST,               0x400101D4, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB7IM,               0x400101D8, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB7UC,               0x400101DC, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB7RG0,              0x400101E0, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB7RG1,              0x400101E4, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB7CP0,              0x400101E8, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB7CP1,              0x400101EC, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB8
 **
 ***************************************************************************/
__IO_REG32_BIT(TB8EN,               0x40010200, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB8RUN,              0x40010204, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB8CR,               0x40010208, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB8MOD,              0x4001020C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB8FFCR,             0x40010210, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB8ST,               0x40010214, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB8IM,               0x40010218, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB8UC,               0x4001021C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB8RG0,              0x40010220, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB8RG1,              0x40010224, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB8CP0,              0x40010228, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB8CP1,              0x4001022C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** TMRB9
 **
 ***************************************************************************/
__IO_REG32_BIT(TB9EN,               0x40010240, __READ_WRITE , __tbxen_bits);
__IO_REG32_BIT(TB9RUN,              0x40010244, __READ_WRITE , __tbxrun_bits);
__IO_REG32_BIT(TB9CR,               0x40010248, __READ_WRITE , __tbxcr_bits);
__IO_REG32_BIT(TB9MOD,              0x4001024C, __READ_WRITE , __tbxmod_bits);
__IO_REG32_BIT(TB9FFCR,             0x40010250, __READ_WRITE , __tbxffcr_bits);
__IO_REG32_BIT(TB9ST,               0x40010254, __READ       , __tbxst_bits);
__IO_REG32_BIT(TB9IM,               0x40010258, __READ_WRITE , __tbxim_bits);
__IO_REG32_BIT(TB9UC,               0x4001025C, __READ_WRITE , __tbxuc_bits);
__IO_REG32_BIT(TB9RG0,              0x40010260, __READ_WRITE , __tbxrg0_bits);
__IO_REG32_BIT(TB9RG1,              0x40010264, __READ_WRITE , __tbxrg1_bits);
__IO_REG32_BIT(TB9CP0,              0x40010268, __READ_WRITE , __tbxcp0_bits);
__IO_REG32_BIT(TB9CP1,              0x4001026C, __READ_WRITE , __tbxcp1_bits);

/***************************************************************************
 **
 ** SIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(SC0EN,               0x40020080, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC0BUF,              0x40020084, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC0CR,               0x40020088, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC0MOD0,             0x4002008C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC0BRCR,             0x40020090, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC0BRADD,            0x40020094, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC0MOD1,             0x40020098, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC0MOD2,             0x4002009C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC0RFC,              0x400200A0, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC0TFC,              0x400200A4, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC0RST,              0x400200A8, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC0TST,              0x400200AC, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC0FCNF,             0x400200B0, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(SC1EN,               0x400200C0, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC1BUF,              0x400200C4, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC1CR,               0x400200C8, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC1MOD0,             0x400200CC, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC1BRCR,             0x400200D0, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC1BRADD,            0x400200D4, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC1MOD1,             0x400200D8, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC1MOD2,             0x400200DC, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC1RFC,              0x400200E0, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC1TFC,              0x400200E4, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC1RST,              0x400200E8, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC1TST,              0x400200EC, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC1FCNF,             0x400200F0, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(SC2EN,               0x40020100, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC2BUF,              0x40020104, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC2CR,               0x40020108, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC2MOD0,             0x4002010C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC2BRCR,             0x40020110, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC2BRADD,            0x40020114, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC2MOD1,             0x40020118, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC2MOD2,             0x4002011C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC2RFC,              0x40020120, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC2TFC,              0x40020124, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC2RST,              0x40020128, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC2TST,              0x4002012C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC2FCNF,             0x40020130, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SBI0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CR0,             0x40020000, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C0CR1,             0x40020004, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C0DBR,             0x40020008, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C0AR,              0x4002000C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C0CR2,             0x40020010, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C0SR      I2C0CR2
#define I2C0SR_bit  I2C0CR2_bit.__sr
__IO_REG32_BIT(I2C0BR0,             0x40020014, __READ_WRITE , __sbixbr0_bits);

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
__IO_REG32_BIT(I2C1CR0,             0x40020020, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C1CR1,             0x40020024, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C1DBR,             0x40020028, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C1AR,              0x4002002C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C1CR2,             0x40020030, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C1SR      I2C1CR2
#define I2C1SR_bit  I2C1CR2_bit.__sr
__IO_REG32_BIT(I2C1BR0,             0x40020034, __READ_WRITE , __sbixbr0_bits);

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
__IO_REG32_BIT(I2C2CR0,             0x40020040, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C2CR1,             0x40020044, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C2DBR,             0x40020048, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C2AR,              0x4002004C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C2CR2,             0x40020050, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C2SR      I2C2CR2
#define I2C2SR_bit  I2C2CR2_bit.__sr
__IO_REG32_BIT(I2C2BR0,             0x40020054, __READ_WRITE , __sbixbr0_bits);

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
 ** ADC
 **
 ***************************************************************************/
__IO_REG8_BIT(ADCLK,                 0x40030000,__READ_WRITE ,__adclk_bits);
__IO_REG8_BIT(ADMOD0,                0x40030004,__READ_WRITE ,__admod0_bits);
__IO_REG8_BIT(ADMOD1,                0x40030008,__READ_WRITE ,__admod1_bits);
__IO_REG8_BIT(ADMOD2,                0x4003000C,__READ_WRITE ,__admod2_bits);
__IO_REG8_BIT(ADMOD3,                0x40030010,__READ_WRITE ,__admod3_bits);
__IO_REG8_BIT(ADMOD4,                0x40030014,__READ_WRITE ,__admod4_bits);
__IO_REG8_BIT(ADMOD5,                0x40030018,__READ_WRITE ,__admod5_bits);
__IO_REG8(    ADCBAS,                0x40030020,__READ_WRITE );
__IO_REG16_BIT(ADREG08,              0x40030030,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG19,              0x40030034,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG2A,              0x40030038,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG3B,              0x4003003C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG4C,              0x40030040,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG5D,              0x40030044,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG6E,              0x40030048,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREG7F,              0x4003004C,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADREGSP,              0x40030050,__READ       ,__adregx_bits);
__IO_REG16_BIT(ADCMP0,               0x40030054,__READ_WRITE ,__adcmpx_bits);
__IO_REG16_BIT(ADCMP1,               0x40030058,__READ_WRITE ,__adcmpx_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG8_BIT(WDMOD,                0x40040000,__READ_WRITE ,__wdmod_bits);
__IO_REG8(    WDCR,                 0x40040004,__WRITE);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG8_BIT(RTCSECR,              0x40040100,__READ_WRITE ,__secr_bits);
__IO_REG8_BIT(RTCMINR,              0x40040101,__READ_WRITE ,__minr_bits);
__IO_REG8_BIT(RTCHOURR,             0x40040102,__READ_WRITE ,__hourr_bits);
__IO_REG8_BIT(RTCDAYR,              0x40040104,__READ_WRITE ,__dayr_bits);
__IO_REG8_BIT(RTCDATER,             0x40040105,__READ_WRITE ,__dater_bits);
__IO_REG8_BIT(RTCMONTHR,            0x40040106,__READ_WRITE ,__monthr_bits);
__IO_REG8_BIT(RTCYEARR,             0x40040107,__READ_WRITE ,__yearr_bits);
__IO_REG8_BIT(RTCPAGER,             0x40040108,__READ_WRITE ,__pager_bits);
__IO_REG8_BIT(RTCRESTR,             0x4004010C,__WRITE      ,__restr_bits);

/***************************************************************************
 **
 ** CG
 **
 ***************************************************************************/
__IO_REG32_BIT(CGSYSCR,             0x40040200, __READ_WRITE ,__syscr_bits);
__IO_REG32_BIT(CGOSCCR,             0x40040204, __READ_WRITE ,__osccr_bits);
__IO_REG32_BIT(CGSTBYCR,            0x40040208, __READ_WRITE ,__stbycr_bits);
__IO_REG32_BIT(CGPLLSEL,            0x4004020C, __READ_WRITE ,__pllsel_bits);
__IO_REG32_BIT(CGCKSEL,             0x40040210, __READ_WRITE ,__cksel_bits);
__IO_REG32_BIT(CGICRCG,             0x40040214, __WRITE      ,__icrcg_bits);
__IO_REG32_BIT(CGNMIFLG,            0x40040218, __READ       ,__nmiflg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x4004021C, __READ_WRITE ,__rstflg_bits);
__IO_REG32_BIT(CGIMCGA,             0x40040220, __READ_WRITE ,__imcga_bits);
__IO_REG32_BIT(CGIMCGB,             0x40040224, __READ_WRITE ,__imcgb_bits);
__IO_REG32_BIT(CGIMCGC,             0x40040228, __READ_WRITE ,__imcgc_bits);
__IO_REG32_BIT(CGIMCGD,             0x4004022C, __READ_WRITE ,__imcgd_bits);
 
/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FCSECBIT,            0x40040500, __READ_WRITE , __secbit_bits);
__IO_REG32_BIT(FCFLCS,              0x40040520, __READ       , __flcs_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  TMPM333FxFG Interrupt Lines
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
#define INT_0                ( 0 + EII)   /* Interrupt pin (PJ0/70pin)    */
#define INT_1                ( 1 + EII)   /* Interrupt pin (PJ1/49pin)    */
#define INT_2                ( 2 + EII)   /* Interrupt pin (PJ2/86pin)    */
#define INT_3                ( 3 + EII)   /* Interrupt pin (PJ3/87pin)    */
#define INT_4                ( 4 + EII)   /* Interrupt pin (PG3/6pin)     */
#define INT_5                ( 5 + EII)   /* Interrupt pin (PF7/19pin)    */
#define INT_RX0              ( 6 + EII)   /* Serial reception (channel.0) */
#define INT_TX0              ( 7 + EII)   /* Serial transmit (channel.0)  */
#define INT_RX1              ( 8 + EII)   /* Serial reception (channel.1) */
#define INT_TX1              ( 9 + EII)   /* Serial transmit (channel.1)  */
#define INT_SBI0             (10 + EII)   /* Serial bus interface 0       */
#define INT_SBI1             (11 + EII)   /* Serial bus interface 1       */
/*#define INT_CECRX            (12 + EII)*/   /* CEC reception                */
/*#define INT_CECTX            (13 + EII)*/   /* CEC transmission             */
/*#define INT_AINTRMCRX0       (14 + EII)*/   /* Remote control signal reception (channel.0)*/
#define INT_ADHP             (15 + EII)   /* Highest priority AD conversion complete interrupt*/
#define INT_ADM0             (16 + EII)   /* AD conversion monitoring function interrupt 0 */
#define INT_ADM1             (17 + EII)   /* AD conversion monitoring function interrupt 1 */
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
#define INT_CAP61            (33 + EII)   /* Input capture 61             */
#define INT_6                (34 + EII)   /* Interrupt pin (PJ6/39pin)    */
#define INT_7                (35 + EII)   /* Interrupt pin (PJ7/58pin)    */
#define INT_RX2              (36 + EII)   /* Serial reception (channel.2) */
#define INT_TX2              (37 + EII)   /* Serial transmission (channel.2)*/
#define INT_SBI2             (38 + EII)   /* Serial bus interface 2       */
/*#define INT_AINTRMCRX1       (39 + EII)*/   /* Remote control signal reception (channel.1) */
#define INT_TB7              (40 + EII)   /* 16bit TMRB match detection 7 */
#define INT_TB8              (41 + EII)   /* 16bit TMRB match detection 8 */
#define INT_TB9              (42 + EII)   /* 16bit TMRB match detection 9 */
#define INT_CAP20            (43 + EII)   /* 16bit TMRB input capture 20  */
#define INT_CAP21            (44 + EII)   /* 16bit TMRB input capture 21  */
#define INT_CAP30            (45 + EII)   /* 16bit TMRB input capture 30  */
#define INT_CAP31            (46 + EII)   /* 16bit TMRB input capture 31  */
#define INT_CAP40            (47 + EII)   /* 16bit TMRB input capture 40  */
#define INT_CAP41            (48 + EII)   /* 16bit TMRB input capture 41  */
#define INT_AD               (49 + EII)   /* A/D conversion completion    */

#endif    /* __IOTMPM333FxFG_H */

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
Interrupt19  = INTSBI0        0x68
Interrupt20  = INTSBI1        0x6C
Interrupt21  = INTADHP        0x7C
Interrupt22  = INTADM0        0x80
Interrupt23  = INTADM1        0x84
Interrupt24  = INTTB0         0x88
Interrupt25  = INTTB1         0x8C
Interrupt26  = INTTB2         0x90
Interrupt27  = INTTB3         0x94
Interrupt28  = INTTB4         0x98
Interrupt29  = INTTB5         0x9C
Interrupt30  = INTTB6         0xA0
Interrupt31  = INTRTC         0xA4
Interrupt32  = INTCAP00       0xA8
Interrupt33  = INTCAP01       0xAC
Interrupt34  = INTCAP10       0xB0
Interrupt35  = INTCAP11       0xB4
Interrupt36  = INTCAP50       0xB8
Interrupt37  = INTCAP51       0xBC
Interrupt38  = INTCAP60       0xC0
Interrupt39  = INTCAP61       0xC4
Interrupt40  = INT6           0xC8
Interrupt41  = INT7           0xCC
Interrupt42  = INTRX2         0xD0
Interrupt43  = INTTX2         0xD4
Interrupt44  = INTSBI2        0xD8
Interrupt45  = INTTB7         0xE0
Interrupt46  = INTTB8         0xE4
Interrupt47  = INTTB9         0xE8
Interrupt48  = INTCAP20       0xEC
Interrupt49  = INTCAP21       0xF0
Interrupt50  = INTCAP30       0xF4
Interrupt51  = INTCAP31       0xF8
Interrupt52  = INTCAP40       0xFC
Interrupt53  = INTCAP41       0x100
Interrupt54  = INTAD          0x104
 
###DDF-INTERRUPT-END###*/
