/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM380FxFG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 37328 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM380FxFG_H
#define __IOTMPM380FxFG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM380FxFG SPECIAL FUNCTION REGISTERS
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
  __REG8  PA7F1  : 1;
} __pafr1_bits;

/*PORT A Function Register 2*/
typedef struct {
  __REG8  PA0F2  : 1;
  __REG8  PA1F2  : 1;
  __REG8  PA2F2  : 1;
  __REG8  PA3F2  : 1;
  __REG8  PA4F2  : 1;
  __REG8  PA5F2  : 1;
  __REG8  PA6F2  : 1;
  __REG8  PA7F2  : 1;
} __pafr2_bits;

/*Port A open drain Control register */
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

/*PORT A Pull-Up Control Register */
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

/*PORT A Pull-Down Control Register */
typedef struct {
  __REG8  PA0DN  : 1;
  __REG8  PA1DN  : 1;
  __REG8  PA2DN  : 1;
  __REG8  PA3DN  : 1;
  __REG8  PA4DN  : 1;
  __REG8  PA5DN  : 1;
  __REG8  PA6DN  : 1;
  __REG8  PA7DN  : 1;
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
  __REG8  PB3F1  : 1;
  __REG8  PB4F1  : 1;
  __REG8  PB5F1  : 1;
  __REG8  PB6F1  : 1;
  __REG8  PB7F1  : 1;
} __pbfr1_bits;

/*Port B open drain Control register */
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

/*PORT B Pull-Up Control Register */
typedef struct {
  __REG8  PB0DN  : 1;
  __REG8  PB1DN  : 1;
  __REG8  PB2DN  : 1;
  __REG8  PB3DN  : 1;
  __REG8  PB4DN  : 1;
  __REG8  PB5DN  : 1;
  __REG8  PB6DN  : 1;
  __REG8  PB7DN  : 1;
  } __pbpdn_bits;

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
  __REG8  PC4  : 1;
  __REG8  PC5  : 1;
  __REG8  PC6  : 1;
  __REG8  PC7  : 1;
} __pc_bits;

/*Port C control register*/
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

/*Port C function register 1*/
typedef struct {
  __REG8  PC0F1  : 1;
  __REG8  PC1F1  : 1;
  __REG8  PC2F1  : 1;
  __REG8  PC3F1  : 1;
  __REG8  PC4F1  : 1;
  __REG8  PC5F1  : 1;
  __REG8  PC6F1  : 1;
  __REG8         : 1;
} __pcfr1_bits;

/*Port C function register 2*/
typedef struct {
  __REG8  PC0F2  : 1;
  __REG8  PC1F2  : 1;
  __REG8  PC2F2  : 1;
  __REG8  PC3F2  : 1;
  __REG8  PC4F2  : 1;
  __REG8  PC5F2  : 1;
  __REG8  PC6F2  : 1;
  __REG8  PC7F2  : 1;
} __pcfr2_bits;

/*Port C function register 3*/
typedef struct {
  __REG8  PC0F3  : 1;
  __REG8  PC1F3  : 1;
  __REG8  PC2F3  : 1;
  __REG8         : 1;
  __REG8  PC4F3  : 1;
  __REG8  PC5F3  : 1;
  __REG8         : 2;
} __pcfr3_bits;

/*Port C function register 4*/
typedef struct {
  __REG8         : 5;
  __REG8  PC5F4  : 1;
  __REG8  PC6F4  : 1;
  __REG8  PC7F4  : 1;
} __pcfr4_bits;

/*Port C function register 5*/
typedef struct {
  __REG8         : 5;
  __REG8  PC5F5  : 1;
  __REG8         : 2;
} __pcfr5_bits;

/*Port C open drain Control register*/
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

/*PORT C Pull-Up Control Register */
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

/*Port C pull-down control register*/
typedef struct {
  __REG8  PC0DN  : 1;
  __REG8  PC1DN  : 1;
  __REG8  PC2DN  : 1;
  __REG8  PC3DN  : 1;
  __REG8  PC4DN  : 1;
  __REG8  PC5DN  : 1;
  __REG8  PC6DN  : 1;
  __REG8  PC7DN  : 1;
} __pcpdn_bits;

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

/*PORT D Register*/
typedef struct {
  __REG8  PD0  : 1;
  __REG8  PD1  : 1;
  __REG8  PD2  : 1;
  __REG8  PD3  : 1;
  __REG8  PD4  : 1;
  __REG8  PD5  : 1;
  __REG8  PD6  : 1;
  __REG8       : 1;
} __pd_bits;

/*Port D control register*/
typedef struct {
  __REG8  PD0C  : 1;
  __REG8  PD1C  : 1;
  __REG8  PD2C  : 1;
  __REG8  PD3C  : 1;
  __REG8  PD4C  : 1;
  __REG8  PD5C  : 1;
  __REG8  PD6C  : 1;
  __REG8        : 1;
} __pdcr_bits;

/*PORT D Function Register 1*/
typedef struct {
  __REG8  PD0F1  : 1;
  __REG8  PD1F1  : 1;
  __REG8  PD2F1  : 1;
  __REG8  PD3F1  : 1;
  __REG8  PD4F1  : 1;
  __REG8  PD5F1  : 1;
  __REG8  PD6F1  : 1;
  __REG8         : 1;
} __pdfr1_bits;

/*PORT D Function Register 2*/
typedef struct {
  __REG8  PD0F2  : 1;
  __REG8  PD1F2  : 1;
  __REG8         : 2;
  __REG8  PD4F2  : 1;
  __REG8         : 3;
} __pdfr2_bits;

/*PORT D Function Register 3*/
typedef struct {
  __REG8  PD0F3  : 1;
  __REG8         : 2;
  __REG8  PD2F3  : 1;
  __REG8         : 4;
} __pdfr3_bits;

/*Port D open drain control register */
typedef struct {
  __REG8  PD0OD  : 1;
  __REG8  PD1OD  : 1;
  __REG8  PD2OD  : 1;
  __REG8  PD3OD  : 1;
  __REG8  PD4OD  : 1;
  __REG8  PD5OD  : 1;
  __REG8  PD6OD  : 1;
  __REG8         : 1;
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
  __REG8         : 1;
} __pdpup_bits;

/*Port D pull-down control register*/
typedef struct {
  __REG8  PD0DN  : 1;
  __REG8  PD1DN  : 1;
  __REG8  PD2DN  : 1;
  __REG8  PD3DN  : 1;
  __REG8  PD4DN  : 1;
  __REG8  PD5DN  : 1;
  __REG8  PD6DN  : 1;
  __REG8         : 1;
} __pdpdn_bits;

/*PORT D Input Enable Control Register */
typedef struct {
  __REG8  PD0IE  : 1;
  __REG8  PD1IE  : 1;
  __REG8  PD2IE  : 1;
  __REG8  PD3IE  : 1;
  __REG8  PD4IE  : 1;
  __REG8  PD5IE  : 1;
  __REG8  PD6IE  : 1;
  __REG8         : 1;
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
  __REG8  PE7  : 1;
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
  __REG8  PE7C  : 1;
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
  __REG8  PE7F1  : 1;
} __pefr1_bits;

/*PORT E Function Register 2*/
typedef struct {
  __REG8         : 2;
  __REG8  PE2F2  : 1;
  __REG8         : 1;
  __REG8  PE4F2  : 1;
  __REG8         : 1;
  __REG8  PE6F2  : 1;
  __REG8  PE7F2  : 1;
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
  __REG8  PE7OD  : 1;
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
  __REG8  PE7UP  : 1;
} __pepup_bits;

/*Port E pull-down control register */
typedef struct {
  __REG8  PE0DN  : 1;
  __REG8  PE1DN  : 1;
  __REG8  PE2DN  : 1;
  __REG8  PE3DN  : 1;
  __REG8  PE4DN  : 1;
  __REG8  PE5DN  : 1;
  __REG8  PE6DN  : 1;
  __REG8  PE7DN  : 1;
} __pepdn_bits;

/*PORT E Input Enable Control Register */
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

/*PORT F Register*/
typedef struct {
  __REG8  PF0  : 1;
  __REG8  PF1  : 1;
  __REG8  PF2  : 1;
  __REG8  PF3  : 1;
  __REG8  PF4  : 1;
  __REG8       : 3;
} __pf_bits;

/*PORT F Control Register */
typedef struct {
  __REG8  PF0C  : 1;
  __REG8  PF1C  : 1;
  __REG8  PF2C  : 1;
  __REG8  PF3C  : 1;
  __REG8  PF4C  : 1;
  __REG8        : 3;
} __pfcr_bits;

/*PORT F Function Register 1*/
typedef struct {
  __REG8  PF0F1  : 1;
  __REG8  PF1F1  : 1;
  __REG8  PF2F1  : 1;
  __REG8  PF3F1  : 1;
  __REG8  PF4F1  : 1;
  __REG8         : 3;
} __pffr1_bits;

/*PORT F Function Register 2*/
typedef struct {
  __REG8         : 1;
  __REG8  PF1F2  : 1;
  __REG8  PF2F2  : 1;
  __REG8  PF3F2  : 1;
  __REG8  PF4F2  : 1;
  __REG8         : 3;
} __pffr2_bits;

/*PORT F Function Register 3*/
typedef struct {
  __REG8         : 2;
  __REG8  PF2F3  : 1;
  __REG8         : 5;
} __pffr3_bits;

/*PORT F Open Drain Control Register */
typedef struct {
  __REG8  PF0OD  : 1;
  __REG8  PF1OD  : 1;
  __REG8  PF2OD  : 1;
  __REG8  PF3OD  : 1;
  __REG8  PF4OD  : 1;
  __REG8         : 3;
} __pfod_bits;

/*PORT F Pull-Up Control Register */
typedef struct {
  __REG8  PF0UP  : 1;
  __REG8  PF1UP  : 1;
  __REG8  PF2UP  : 1;
  __REG8  PF3UP  : 1;
  __REG8  PF4UP  : 1;
  __REG8         : 3;
} __pfpup_bits;

/*Port F pull-down control register */
typedef struct {
  __REG8  PF0DN  : 1;
  __REG8  PF1DN  : 1;
  __REG8  PF2DN  : 1;
  __REG8  PF3DN  : 1;
  __REG8  PF4DN  : 1;
  __REG8         : 3;
} __pfpdn_bits;

/*PORT F Input Enable Control Register */
typedef struct {
  __REG8  PF0IE  : 1;
  __REG8  PF1IE  : 1;
  __REG8  PF2IE  : 1;
  __REG8  PF3IE  : 1;
  __REG8  PF4IE  : 1;
  __REG8         : 3;
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
  __REG8         : 1;
} __pgfr1_bits;

/*PORT G Function Register 2*/
typedef struct {
  __REG8         : 4;
  __REG8  PG4F2  : 1;
  __REG8  PG5F2  : 1;
  __REG8  PG6F2  : 1;
  __REG8  PG7F2  : 1;
} __pgfr2_bits;

/*PORT G Function Register 3*/
typedef struct {
  __REG8  PG0F3  : 1;
  __REG8  PG1F3  : 1;
  __REG8  PG2F3  : 1;
  __REG8         : 1;
  __REG8  PG4F3  : 1;
  __REG8  PG5F3  : 1;
  __REG8         : 2;
} __pgfr3_bits;

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

/*Port G pull-down control register */
typedef struct {
  __REG8  PG0DN  : 1;
  __REG8  PG1DN  : 1;
  __REG8  PG2DN  : 1;
  __REG8  PG3DN  : 1;
  __REG8  PG4DN  : 1;
  __REG8  PG5DN  : 1;
  __REG8  PG6DN  : 1;
  __REG8  PG7DN  : 1;
} __pgpdn_bits;

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
  __REG8         : 5;
} __phfr1_bits;

/*Port H open drain control register*/
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

/*Port H pull-down control register*/
typedef struct {
  __REG8  PH0DN  : 1;
  __REG8  PH1DN  : 1;
  __REG8  PH2DN  : 1;
  __REG8  PH3DN  : 1;
  __REG8  PH4DN  : 1;
  __REG8  PH5DN  : 1;
  __REG8  PH6DN  : 1;
  __REG8  PH7DN  : 1;
} __phpdn_bits;

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
  __REG8       : 6;
} __pi_bits;

/*PORT I Control Register 1*/
typedef struct {
  __REG8  PI0C  : 1;
  __REG8  PI1C  : 1;
  __REG8        : 6;
} __picr_bits;

/*Port I open drain control register*/
typedef struct {
  __REG8  PI0OD  : 1;
  __REG8  PI1OD  : 1;
  __REG8         : 6;
} __piod_bits;

/*PORT I Pull-Up Control Register */
typedef struct {
  __REG8  PI0UP  : 1;
  __REG8  PI1UP  : 1;
  __REG8         : 6;
} __pipup_bits;

/*Port I pull-down control register */
typedef struct {
  __REG8  PI0DN  : 1;
  __REG8  PI1DN  : 1;
  __REG8         : 6;
} __pipdn_bits;

/*PORT I Input Enable Control Register */
typedef struct {
  __REG8  PI0IE  : 1;
  __REG8  PI1IE  : 1;
  __REG8         : 6;
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
  __REG8         : 6;
  __REG8  PJ6F1  : 1;
  __REG8  PJ7F1  : 1;
} __pjfr1_bits;

/*Port J open drain control register*/
typedef struct {
  __REG8  PJ0OD  : 1;
  __REG8  PJ1OD  : 1;
  __REG8  PJ2OD  : 1;
  __REG8  PJ3OD  : 1;
  __REG8  PJ4OD  : 1;
  __REG8  PJ5OD  : 1;
  __REG8  PJ6OD  : 1;
  __REG8  PJ7OD  : 1;
} __pjod_bits;

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

/*Port J pull-down control register*/
typedef struct {
  __REG8  PJ0DN  : 1;
  __REG8  PJ1DN  : 1;
  __REG8  PJ2DN  : 1;
  __REG8  PJ3DN  : 1;
  __REG8  PJ4DN  : 1;
  __REG8  PJ5DN  : 1;
  __REG8  PJ6DN  : 1;
  __REG8  PJ7DN  : 1;
} __pjpdn_bits;

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

/*PORT L Register*/
typedef struct {
  __REG8  PL0  : 1;
  __REG8       : 1;
  __REG8  PL2  : 1;
  __REG8       : 5;
} __pl_bits;

/*Port L control register*/
typedef struct {
  __REG8  PL0C : 1;
  __REG8       : 1;
  __REG8  PL2C : 1;
  __REG8       : 5;
} __plcr_bits;

/*PORT L Function Register 1*/
typedef struct {
  __REG8         : 2;
  __REG8  PL2F1  : 1;
  __REG8         : 5;
} __plfr1_bits;

/*Port L open drain control register*/
typedef struct {
  __REG8  PL0OD : 1;
  __REG8        : 1;
  __REG8  PL2OD : 1;
  __REG8        : 5;
} __plod_bits;

/*Port L pull-up control register*/
typedef struct {
  __REG8  PL0UP : 1;
  __REG8        : 1;
  __REG8  PL2UP : 1;
  __REG8        : 5;
} __plpup_bits;

/*Port L pull-down control registerr*/
typedef struct {
  __REG8  PL0DN : 1;
  __REG8        : 1;
  __REG8  PL2DN : 1;
  __REG8        : 5;
} __plpdn_bits;

/*PORT L Input Enable Control Register*/
typedef struct {
  __REG8         : 2;
  __REG8  PL2IE  : 1;
  __REG8         : 5;
} __plie_bits;

/*PORT M Register*/
typedef struct {
  __REG8  PM0  : 1;
  __REG8  PM1  : 1;
  __REG8       : 6;
} __pm_bits;

/*PORT M Control Register */
typedef struct {
  __REG8  PM0C  : 1;
  __REG8  PM1C  : 1;
  __REG8        : 6;
} __pmcr_bits;

/*Port M open drain Control register */
typedef struct {
  __REG8  PM0OD  : 1;
  __REG8  PM1OD  : 1;
  __REG8         : 6;
} __pmod_bits;

/*PORT M Pull-Up Control Register */
typedef struct {
  __REG8  PM0UP  : 1;
  __REG8  PM1UP  : 1;
  __REG8         : 6;
} __pmpup_bits;

/*PORT M Pull-Down Control Register */
typedef struct {
  __REG8  PM0DN  : 1;
  __REG8  PM1DN  : 1;
  __REG8         : 6;
} __pmpdn_bits;

/*PORT M Input Enable Control Register */
typedef struct {
  __REG8  PM0IE  : 1;
  __REG8  PM1IE  : 1;
  __REG8         : 6;
} __pmie_bits;

/*PORT N Register*/
typedef struct {
  __REG8  PN0  : 1;
  __REG8  PN1  : 1;
  __REG8  PN2  : 1;
  __REG8  PN3  : 1;
  __REG8  PN4  : 1;
  __REG8  PN5  : 1;
  __REG8  PN6  : 1;
  __REG8  PN7  : 1;
} __pn_bits;

/*PORT N Control Register */
typedef struct {
  __REG8  PN0C  : 1;
  __REG8  PN1C  : 1;
  __REG8  PN2C  : 1;
  __REG8  PN3C  : 1;
  __REG8  PN4C  : 1;
  __REG8  PN5C  : 1;
  __REG8  PN6C  : 1;
  __REG8  PN7C  : 1;
} __pncr_bits;

/*PORT N Function Register 1*/
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

/*PORT N Function Register 2*/
typedef struct {
  __REG8         : 4;
  __REG8  PN4F2  : 1;
  __REG8  PN5F2  : 1;
  __REG8         : 1;
  __REG8  PN7F2  : 1;
} __pnfr2_bits;

/*Port N open drain Control register */
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

/*PORT N Pull-Up Control Register */
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

/*PORT N Pull-Down Control Register */
typedef struct {
  __REG8  PN0DN  : 1;
  __REG8  PN1DN  : 1;
  __REG8  PN2DN  : 1;
  __REG8  PN3DN  : 1;
  __REG8  PN4DN  : 1;
  __REG8  PN5DN  : 1;
  __REG8  PN6DN  : 1;
  __REG8  PN7DN  : 1;
} __pnpdn_bits;

/*PORT N Input Enable Control Register */
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

/*PORT P Register*/
typedef struct {
  __REG8  PP0  : 1;
  __REG8  PP1  : 1;
  __REG8       : 6;
} __pp_bits;

/*PORT P Control Register */
typedef struct {
  __REG8  PP0C  : 1;
  __REG8  PP1C  : 1;
  __REG8        : 6;
} __ppcr_bits;

/*Port P open drain Control register */
typedef struct {
  __REG8  PP0OD  : 1;
  __REG8  PP1OD  : 1;
  __REG8         : 6;
} __ppod_bits;

/*PORT P Pull-Up Control Register */
typedef struct {
  __REG8  PP0UP  : 1;
  __REG8  PP1UP  : 1;
  __REG8         : 6;
} __pppup_bits;

/*PORT P Pull-Down Control Register */
typedef struct {
  __REG8  PP0DN  : 1;
  __REG8  PP1DN  : 1;
  __REG8         : 6;
} __pppdn_bits;

/*PORT P Input Enable Control Register */
typedef struct {
  __REG8  PP0IE  : 1;
  __REG8  PP1IE  : 1;
  __REG8         : 6;
} __ppie_bits;

/*TMRBn enable register (channels 0 through 7)*/
typedef struct {
  __REG32           : 7;
  __REG32  TBEN     : 1;
  __REG32           :24;
} __tbxen_bits;

/*TMRB RUN register (channels 0 through 7)*/
typedef struct {
  __REG32  TBRUN    : 1;
  __REG32           : 1;
  __REG32  TBPRUN   : 1;
  __REG32           :29;
} __tbxrun_bits;

/*TMRB control register (channels 0 through 7)*/
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

/*TMRB mode register (channels 0 thorough 7)*/
typedef struct {
  __REG32  TBCLK    : 2;
  __REG32  TBCLE    : 1;
  __REG32  TBCPM    : 2;
  __REG32  TBCP     : 1;
  __REG32  TBRSWR   : 1;
  __REG32           :25;
} __tbxmod_bits;

/*TMRB flip-flop control register (channels 0 through 7)*/
typedef struct {
  __REG32  TBFF0C   : 2;
  __REG32  TBE0T1   : 1;
  __REG32  TBE1T1   : 1;
  __REG32  TBC0T1   : 1;
  __REG32  TBC1T1   : 1;
  __REG32           :26;
} __tbxffcr_bits;

/*TMRB status register (channels 0 through 7)*/
typedef struct {
  __REG32  INTTB0   : 1;
  __REG32  INTTB1   : 1;
  __REG32  INTTBOF  : 1;
  __REG32           :29;
} __tbxst_bits;

/*TMRB interrupt mask register (channels 0 through 7)*/
typedef struct {
  __REG32  TBIM0    : 1;
  __REG32  TBIM1    : 1;
  __REG32  TBIMOF   : 1;
  __REG32           :29;
} __tbxim_bits;

/*TMRB read capture register (channels 0 through 7)*/
typedef struct {
  __REG32  TBUC     :16;
  __REG32           :16;
} __tbxuc_bits;

/*TMRB timer register 0 (channels 0 through 7)*/
typedef struct {
  __REG32  TBRG0    :16;
  __REG32           :16;
} __tbxrg0_bits;

/*TMRB timer register 1 (channels 0 through 7)*/
typedef struct {
  __REG32  TBRG1    :16;
  __REG32           :16;
} __tbxrg1_bits;

/*TMRB capture register 0 (channels 0 through 7)*/
typedef struct {
  __REG32  TBCP0    :16;
  __REG32           :16;
} __tbxcp0_bits;

/*TMRB capture register 1 (channels 0 through 7)*/
typedef struct {
  __REG32  TBCP1    :16;
  __REG32           :16;
} __tbxcp1_bits;

/*MPTn enable register*/
typedef struct {
  __REG32  MTMODE     : 1;
  __REG32             : 5;
  __REG32  MTHALT     : 1;
  __REG32  MTEN       : 1;
  __REG32             :24;
} __mtxen_bits;

/*MPT RUN register */
typedef struct {
  __REG32  MTRUN      : 1;
  __REG32             : 1;
  __REG32  MTPRUN     : 1;
  __REG32             :29;
} __mtxrun_bits;

/*MPT control register*/
typedef struct {
  __REG32  MTTBCSSEL  : 1;
  __REG32  MTTBTRGSEL : 1;
  __REG32             : 1;
  __REG32  MTI2TB     : 1;
  __REG32             : 3;
  __REG32  MTTBWBF    : 1;
  __REG32             :24;
} __mtxcr_bits;

/*MPT mode register*/
typedef struct {
  __REG32  MTTBCLK    : 2;
  __REG32  MTTBCLE    : 1;
  __REG32  MTTBCPM    : 2;
  __REG32  MTTBCP     : 1;
  __REG32  MTTBRSWR   : 1;
  __REG32             :25;
} __mtxmod_bits;

/*MPT flip-flop control register*/
typedef struct {
  __REG32  MTTBFF0C   : 2;
  __REG32  MTTBE0T1   : 1;
  __REG32  MTTBE1T1   : 1;
  __REG32  MTTBC0T1   : 1;
  __REG32  MTTBC1T1   : 1;
  __REG32             :26;
} __mtxffcr_bits;

/*MPT status register*/
typedef struct {
  __REG32  MTTBINTOF  : 1;
  __REG32  MTTBINT1   : 1;
  __REG32  MTTBINT0   : 1;
  __REG32             :29;
} __mtxst_bits;

/*MPT interrupt mask register*/
typedef struct {
  __REG32  MTTBIMOF   : 1;
  __REG32  MTTBIM1    : 1;
  __REG32  MTTBIM0    : 1;
  __REG32             :29;
} __mtxim_bits;

/*MPT read capture register*/
typedef struct {
  __REG32  UC         :16;
  __REG32             :16;
} __mtxuc_bits;

/*MPT timer register 0*/
typedef struct {
  __REG32  MTRG0      :16;
  __REG32             :16;
} __mtxrg0_bits;

/*MPT timer register 1*/
typedef struct {
  __REG32  MTRG1      :16;
  __REG32             :16;
} __mtxrg1_bits;

/*MPT capture register 0*/
typedef struct {
  __REG32  MTCP0      :16;
  __REG32             :16;
} __mtxcp0_bits;

/*MPT capture register 1*/
typedef struct {
  __REG32  MTCP1      :16;
  __REG32             :16;
} __mtxcp1_bits;

/*IGBT control register*/
typedef struct {
  __REG32  IGCLK      : 2;
  __REG32  IGSTA      : 2;
  __REG32  IGSTP      : 2;
  __REG32  IGSNGL     : 1;
  __REG32             : 1;
  __REG32  IGPRD      : 2;
  __REG32  IGIDIS     : 1;
  __REG32             :21;
} __mtigxcr_bits;

/*IGBT timer restart register */
typedef struct {
  __REG32  IGRESTA    : 1;
  __REG32             :31;
} __mtigxresta_bits;

/*IGBT timer status register */
typedef struct {
  __REG32  IGST       : 1;
  __REG32             :31;
} __mtigxst_bits;

/*IGBT input control register*/
typedef struct {
  __REG32  IGNCSEL    : 4;
  __REG32             : 2;
  __REG32  IGTRGSEL   : 1;
  __REG32  IGTRGM     : 1;
  __REG32             :24;
} __mtigxicr_bits;

/*IGBT output control register*/
typedef struct {
  __REG32  IGOEN0     : 1;
  __REG32  IGOEN1     : 1;
  __REG32             : 2;
  __REG32  IGPOL0     : 1;
  __REG32  IGPOL1     : 1;
  __REG32             :26;
} __mtigxocr_bits;

/*IGBT timer register 2*/
typedef struct {
  __REG32  IGRG2      :16;
  __REG32             :16;
} __mtigxrg2_bits;

/*IGBT timer register 3*/
typedef struct {
  __REG32  IGRG3      :16;
  __REG32             :16;
} __mtigxrg3_bits;

/*IGBT timer register 4*/
typedef struct {
  __REG32  IGRG4      :16;
  __REG32             :16;
} __mtigxrg4_bits;

/*IGBT EMG control register*/
typedef struct {
  __REG32  IGEMGEN    : 1;
  __REG32  IGEMGOC    : 1;
  __REG32  IGEMGRS    : 1;
  __REG32             : 1;
  __REG32  IGEMGCNT   : 4;
  __REG32             :24;
} __mtigxemgcr_bits;

/*IGBT EMG status register*/
typedef struct {
  __REG32  IGEMGST    : 1;
  __REG32  IGEMGIN    : 1;
  __REG32             :30;
} __mtigxemgst_bits;

/*Encoder  Input Control Register*/
typedef struct {
  __REG32  ENDEV    : 3;
  __REG32  INTEN    : 1;
  __REG32  NR       : 2;
  __REG32  ENRUN    : 1;
  __REG32  ZEN      : 1;
  __REG32  CMPEN    : 1;
  __REG32  ZESEL    : 1;
  __REG32  ENCLR    : 1;
  __REG32  SFTCAP   : 1;
  __REG32  ZDET     : 1;
  __REG32  U_D      : 1;
  __REG32  REVERR   : 1;
  __REG32  CMP      : 1;
  __REG32  P3EN     : 1;
  __REG32  MODE     : 2;
  __REG32           :13;
} __enxtncr_bits;

/*Encoder Counter Reload Register*/
typedef struct {
  __REG32  RELOAD   :16;
  __REG32           :16;
} __enxreload_bits;

/*Encoder Counter Compare Register*/
typedef struct {
  __REG32  INT      :24;
  __REG32           : 8;
} __enxint_bits;

/*Encoder Counter Register*/
typedef struct {
  __REG32  CNT      :24;
  __REG32           : 8;
} __enxcnt_bits;

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

/*SSPIMSC (SSP Interrupt mask set and clear register)*/
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
  /*SBIxCR1*/
  struct {
    __REG32  SCK      : 3;
    __REG32           : 1;
    __REG32  ACK      : 1;
    __REG32  BC       : 3;
    __REG32           :24;
  };
  /*Software reset status*/
  struct {
    __REG32  SWRMON   : 1;
    __REG32           :31;
  };
  /*_SBIxCR1*/
  struct {
    __REG32 _SCK      : 3;
    __REG32           : 1;
    __REG32 SIOM      : 2;
    __REG32 SIOINH    : 1;
    __REG32 SIOS      : 1;
    __REG32           :24;
  };
} __sbixcr1_bits;

/*Serial bus interface data buffer register*/
typedef struct {
  __REG32  RX_TX    : 8;
  __REG32           :24;
} __sbixdbr_bits;

/*I2C bus address register*/
typedef struct {
  __REG32 ALS     : 1;
  __REG32 SA      : 7;
  __REG32         :24;
} __sbixi2car_bits;

/*Serial bus control register 2*/
typedef union {
  /*SBIxCR2*/
  struct {
    __REG32 SWRST   : 2;
    __REG32 SBIM    : 2;
    __REG32 PIN     : 1;
    __REG32 BB      : 1;
    __REG32 TRX     : 1;
    __REG32 MST     : 1;
    __REG32         :24;
  };
  /*SBIxSR*/
  struct {
    __REG32 LRB     : 1;
    __REG32 ADO     : 1;
    __REG32 AAS     : 1;
    __REG32 AL      : 1;
    __REG32 _PIN    : 1;
    __REG32 _BB     : 1;
    __REG32 _TRX    : 1;
    __REG32 _MST    : 1;
    __REG32         :24;
  };
  /*_SBIxSR*/
  struct {
    __REG32         : 2;
    __REG32 SEF     : 1;
    __REG32 SIOF    : 1;
    __REG32         :28;
  };
} __sbixcr2_sr_bits;

/*Serial bus interface baud rate register*/
typedef struct {
  __REG32         : 6;
  __REG32 I2SBI   : 1;
  __REG32         :25;
} __sbixbr0_bits;

/*ADC Clock Setting Register (ADCLK)*/
typedef struct {
  __REG8  ADCLK     : 3;
  __REG8  TSH       : 4;
  __REG8            : 1;
} __adxclk_bits;

/*ADMOD0*/
typedef struct {
  __REG8  ADSS      : 1;
  __REG8  DACON     : 1;
  __REG8            : 6;
} __adxmod0_bits;

/*ADMOD1*/
typedef struct {
  __REG8  ADAS      : 1;
  __REG8            : 6;
  __REG8  ADEN      : 1;
} __adxmod1_bits;

/*ADMOD2*/
typedef struct {
  __REG8  ADBFN     : 1;
  __REG8  ADSFN     : 1;
  __REG8            : 6;
} __adxmod2_bits;

/*ADCMPCR0,ADCMPCR1*/
typedef struct {
  __REG16 REGS      : 4;
  __REG16 ADBIG     : 1;
  __REG16           : 2;
  __REG16 CMPEN     : 1;
  __REG16 CMPCNT    : 4;
  __REG16           : 4;
} __adxcmpcrx_bits;

/*ADCMP0 and ADCMP1*/
typedef struct {
  __REG16           : 4;
  __REG16 ADCMP     :12;
} __adxcmpx_bits;

/*ADREGx*/
typedef struct {
  __REG16 ADRRF     : 1;
  __REG16 OVR       : 1;
  __REG16           : 2;
  __REG16 ADR       :12;
} __adxregx_bits;

/*ADPSELx*/
typedef struct {
  __REG8  PMDS      : 3;
  __REG8            : 4;
  __REG8  PENS      : 1;
} __adxpselx_bits;

/*ADPINTSx*/
typedef struct {
  __REG8  INTSEL    : 2;
  __REG8            : 6;
} __adxpintsx_bits;

/*ADPSETx*/
typedef struct {
  __REG32  AINSP0   : 5;
  __REG32           : 2;
  __REG32  ENSP0    : 1;
  __REG32  AINSP1   : 5;
  __REG32           : 2;
  __REG32  ENSP1    : 1;
  __REG32  AINSP2   : 5;
  __REG32           : 2;
  __REG32  ENSP2    : 1;
  __REG32  AINSP3   : 5;
  __REG32           : 2;
  __REG32  ENSP3    : 1;
} __adxpsetx_bits;

/*ADTSETx*/
typedef struct {
  __REG8  AINST     : 5;
  __REG8            : 2;
  __REG8  ENST      : 1;
} __adxtsetx_bits;

/*ADSSETx*/
typedef struct {
  __REG8  AINSS     : 5;
  __REG8            : 2;
  __REG8  ENSS      : 1;
} __adxssetx_bits;

/*ADASETx*/
typedef struct {
  __REG8  AINSA     : 5;
  __REG8            : 2;
  __REG8  ENSA      : 1;
} __adxasetx_bits;

/*ADC Mode Setting Register 3*/
typedef struct {
  __REG16         : 3;
  __REG16 PMODE   : 3;
  __REG16         : 2;
  __REG16 RCUT    : 1;
  __REG16 PDCL    : 1;
  __REG16 BITS    : 2;
  __REG16         : 4; 
} __adxmod3_bits;

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
  __REG32 RMCEND1   : 7;
  __REG32           :25;
} __rmcend1_bits;

/*Remote Control Receive End Bit Number Register 2*/
typedef struct {
  __REG32 RMCEND2   : 7;
  __REG32           :25;
} __rmcend2_bits;

/*Remote Control Receive End Bit Number Register 3*/
typedef struct {
  __REG32 RMCEND3   : 7;
  __REG32           :25;
} __rmcend3_bits;

/*Remote Control Source Clock selection Register*/
typedef struct {
  __REG32 RMCCLK    : 1;
  __REG32           :31;
} __rmcfssel_bits;

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
  __REG32 FPSEL   : 2;
  __REG32         : 2;
  __REG32 SCOSEL  : 2;
  __REG32         : 2;
  __REG32 FCSTOP  : 1;
  __REG32         :11; 
} __syscr_bits;

/*Oscillation Control Register*/
typedef struct {
  __REG32 WUEON     : 1;
  __REG32 WUEF      : 1;
  __REG32 PLLON     : 1;
  __REG32 WUPSEL1   : 1;
  __REG32           : 4;
  __REG32 XEN1      : 1;
  __REG32 XTEN      : 1;
  __REG32           : 4;
  __REG32 WUODR0_1  : 2;
  __REG32 XEN2      : 1;
  __REG32 OSCSEL    : 1;
  __REG32 HOSCON    : 1;
  __REG32 WUPSEL2   : 1;
  __REG32 WUODR2_13 :12;
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

/*ICRCG Register*/
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           :27; 
} __icrcg_bits;

/*NMI Flag Register*/
typedef struct {
  __REG32 NMIFLG0   : 1;
  __REG32           : 1;
  __REG32 NMIFLG2   : 1;
  __REG32           :29; 
} __nmiflg_bits;

/*Reset Flag Register*/
typedef struct {
  __REG32 PONRSTF   : 1;
  __REG32 PINRSTF   : 1;
  __REG32 WDTRSTF   : 1;
  __REG32           : 1;
  __REG32 DBGRSTF   : 1;
  __REG32 OFDRSTF   : 1;
  __REG32           :26; 
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
} __imcgd_bits;

/*CG Interrupt Mode Control Register E*/
typedef struct {
  __REG32 INTRTCEN    : 1;
  __REG32             : 1;
  __REG32 EMSTRTC     : 2;
  __REG32 EMCGRTC     : 3;
  __REG32             : 1;
  __REG32 INTRMCRXEN  : 1;
  __REG32             : 1;
  __REG32 EMSTRMCRX   : 2;
  __REG32 EMCGRMCRX   : 3;
  __REG32             :17;
} __imcge_bits;

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
  __REG32 OFDMN        : 8;
  __REG32              :24;
} __ofdmn_bits;

/*Higher detection frequency setting register (OFDMX)*/
typedef struct {
  __REG32 OFDMX        : 8;
  __REG32              :24;
} __ofdmx_bits;

/*Oscillation frequency detector reset enable control register (OFDRST)*/
typedef struct {
  __REG32 OFDRSTEN     : 1;
  __REG32              :31;
} __ofdrst_bits;

/*Oscillation frequency detector Status register (OFDSTAT)*/
typedef struct {
  __REG32 FRQERR       : 1;
  __REG32 OFDBUSY      : 1;
  __REG32              :30;
} __ofdstat_bits;

/*Voltage detection control register*/
typedef struct {
  __REG32 VDEN         : 1;
  __REG32 VDLVL        : 2;
  __REG32              :29;
} __vdcr_bits;
 
/*Voltage detection status register*/
typedef struct {
  __REG32 VDSR         : 1;
  __REG32              :31;
} __vdsr_bits;

/* DMACIntStatus (DMAC Interrupt Status Register) */
typedef struct{
__REG32 IntStatus0            : 1;
__REG32 IntStatus1            : 1;
__REG32                       :30;
} __dmacintstaus_bits;

/* DMACIntTCStatus (DMAC Interrupt Terminal Count Status Register) */
typedef struct{
__REG32 IntStatusTC0          : 1;
__REG32 IntStatusTC1          : 1;
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
__REG32                       :23;
} __dmacsoftbreq_bits;

/* DMACSoftSReq (DMAC Software Single Request Register ) */
typedef struct{
__REG32                       : 6;
__REG32 SoftSReq6             : 1;
__REG32                       : 1;
__REG32 SoftSReq8             : 1;
__REG32                       :23;
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
__REG32 Prot1                 : 1;
__REG32 Prot2                 : 1;
__REG32 Prot3                 : 1;
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

/*PMD Enable Register (MDEN)*/
typedef struct {
  __REG32 PWMEN        : 1;
  __REG32              :31;
} __mden_bits;

/*Port Output Mode Register (PORTMD)*/
typedef struct {
  __REG32 PORTMD       : 2;
  __REG32              :30;
} __portmd_bits;

/*PMD Control Register (MDCR)*/
typedef struct {
  __REG32 PWMMD        : 1;
  __REG32 INTPRD       : 2;
  __REG32 PINT         : 1;
  __REG32 DTYMD        : 1;
  __REG32 SYNTMD       : 1;
  __REG32 PWMCK        : 1;
  __REG32              :25;
} __mdcr_bits;

/*PWM Counter Status Register (CNTSTA)*/
typedef struct {
  __REG32 UPDWN        : 1;
  __REG32              :31;
} __cntsta_bits;

/*PWM Counter Register (MDCNT)*/
typedef struct {
  __REG32 MDCNT        :16;
  __REG32              :16;
} __mdcnt_bits;

/*PWM Period Register (MDPRD)*/
typedef struct {
  __REG32 MDPRD        :16;
  __REG32              :16;
} __mdprd_bits;

/*PWM Compare Register (CMPU)*/
typedef struct {
  __REG32 CMPU         :16;
  __REG32              :16;
} __cmpu_bits;

/*PWM Compare Register (CMPV)*/
typedef struct {
  __REG32 CMPV         :16;
  __REG32              :16;
} __cmpv_bits;

/*PWM Compare Register (CMPW)*/
typedef struct {
  __REG32 CMPW         :16;
  __REG32              :16;
} __cmpw_bits;

/*Mode Select Register (MODESEL)*/
typedef struct {
  __REG32 MDSEL        : 1;
  __REG32              :31;
} __modesel_bits;

/*PMD Output Control Register (MDOUT)*/
typedef struct {
  __REG32 UOC          : 2;
  __REG32 VOC          : 2;
  __REG32 WOC          : 2;
  __REG32              : 2;
  __REG32 UPWM         : 1;
  __REG32 VPWM         : 1;
  __REG32 WPWM         : 1;
  __REG32              :21;
} __mdout_bits;

/*PMD Output Setting Register (MDPOT)*/
typedef struct {
  __REG32 PSYNCS       : 2;
  __REG32 POLL         : 1;
  __REG32 POLH         : 1;
  __REG32              :28;
} __mdpot_bits;

/*EMG Release Register (EMGREL)*/
typedef struct {
  __REG32 EMGREL       : 8;
  __REG32              :24;
} __emgrel_bits;

/*EMG Control Register (EMGCR)*/
typedef struct {
  __REG32 EMGEN        : 1;
  __REG32 EMGRS        : 1;
  __REG32              : 1;
  __REG32 EMGMD        : 2;
  __REG32 INHEN        : 1;
  __REG32              : 2;
  __REG32 EMGCNT       : 4;
  __REG32              :20;
} __emgcr_bits;

/*EMG Status Register (EMGSTA)*/
typedef struct {
  __REG32 EMGST        : 1;
  __REG32 EMGI         : 1;
  __REG32              :30;
} __emgsta_bits;

/*Dead Time Register (DTR)*/
typedef struct {
  __REG32 DTR          : 8;
  __REG32              :24;
} __dtr_bits;

/*Trigger Compare Register (TRGCMP0)*/
typedef struct {
  __REG32 TRGCMP0      :16;
  __REG32              :16;
} __trgcmp0_bits;

/*Trigger Compare Register (TRGCMP1)*/
typedef struct {
  __REG32 TRGCMP1      :16;
  __REG32              :16;
} __trgcmp1_bits;

/*Trigger Compare Register (TRGCMP2)*/
typedef struct {
  __REG32 TRGCMP2      :16;
  __REG32              :16;
} __trgcmp2_bits;

/*Trigger Compare Register (TRGCMP3)*/
typedef struct {
  __REG32 TRGCMP3      :16;
  __REG32              :16;
} __trgcmp3_bits;

/*Trigger Control Register (TRGCR)*/
typedef struct {
  __REG32 TRG0MD       : 3;
  __REG32 TRG0BE       : 1;
  __REG32 TRG1MD       : 3;
  __REG32 TRG1BE       : 1;
  __REG32              :24;
} __trgcr_bits;

/*Trigger Output Mode Setting Register (TRGMD)*/
typedef struct {
  __REG32 EMGTGE       : 1;
  __REG32              :31;
} __trgmd_bits;

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
  __REG32  CLRPEND81      : 1;
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

/* Interrupt Priority Registers 56-59 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_56         : 3;
  __REG32                 : 5;
  __REG32  PRI_57         : 3;
  __REG32                 : 5;
  __REG32  PRI_58         : 3;
  __REG32                 : 5;
  __REG32  PRI_59         : 3;
} __pri14_bits;

/* Interrupt Priority Registers 60-63 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_60         : 3;
  __REG32                 : 5;
  __REG32  PRI_61         : 3;
  __REG32                 : 5;
  __REG32  PRI_62         : 3;
  __REG32                 : 5;
  __REG32  PRI_63         : 3;
} __pri15_bits;

/* Interrupt Priority Registers 64-67 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_64         : 3;
  __REG32                 : 5;
  __REG32  PRI_65         : 3;
  __REG32                 : 5;
  __REG32  PRI_66         : 3;
  __REG32                 : 5;
  __REG32  PRI_67         : 3;
} __pri16_bits;

/* Interrupt Priority Registers 68-71 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_68         : 3;
  __REG32                 : 5;
  __REG32  PRI_69         : 3;
  __REG32                 : 5;
  __REG32  PRI_70         : 3;
  __REG32                 : 5;
  __REG32  PRI_71         : 3;
} __pri17_bits;

/* Interrupt Priority Registers 72-75 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_72         : 3;
  __REG32                 : 5;
  __REG32  PRI_73         : 3;
  __REG32                 : 5;
  __REG32  PRI_74         : 3;
  __REG32                 : 5;
  __REG32  PRI_75         : 3;
} __pri18_bits;

/* Interrupt Priority Registers 76-79 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_76         : 3;
  __REG32                 : 5;
  __REG32  PRI_77         : 3;
  __REG32                 : 5;
  __REG32  PRI_78         : 3;
  __REG32                 : 5;
  __REG32  PRI_79         : 3;
} __pri19_bits;

/* Interrupt Priority Registers 80-83 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_80         : 3;
  __REG32                 : 5;
  __REG32  PRI_81         : 3;
  __REG32                 : 5;
  __REG32  PRI_82         : 3;
  __REG32                 : 5;
  __REG32  PRI_83         : 3;
} __pri20_bits;

/* Interrupt Priority Registers 84-87 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_84         : 3;
  __REG32                 : 5;
  __REG32  PRI_85         : 3;
  __REG32                 : 5;
  __REG32  PRI_86         : 3;
  __REG32                 : 5;
  __REG32  PRI_87         : 3;
} __pri21_bits;

/* Interrupt Priority Registers 88-91 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_88         : 3;
  __REG32                 : 5;
  __REG32  PRI_89         : 3;
  __REG32                 : 5;
  __REG32  PRI_90         : 3;
  __REG32                 : 5;
  __REG32  PRI_91         : 3;
} __pri22_bits;

/* Interrupt Priority Registers 92-95 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_92         : 3;
  __REG32                 : 5;
  __REG32  PRI_93         : 3;
  __REG32                 : 5;
  __REG32  PRI_94         : 3;
  __REG32                 : 5;
  __REG32  PRI_95         : 3;
} __pri23_bits;

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
__IO_REG32_BIT(SETENA2,           0xE000E108,__READ_WRITE ,__setena2_bits);
__IO_REG32_BIT(CLRENA0,           0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,           0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(CLRENA2,           0xE000E188,__READ_WRITE ,__clrena2_bits);
__IO_REG32_BIT(SETPEND0,          0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,          0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(SETPEND2,          0xE000E208,__READ_WRITE ,__setpend2_bits);
__IO_REG32_BIT(CLRPEND0,          0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,          0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(CLRPEND2,          0xE000E288,__READ_WRITE ,__clrpend2_bits);
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
__IO_REG32_BIT(IP16,              0xE000E440,__READ_WRITE ,__pri16_bits);
__IO_REG32_BIT(IP17,              0xE000E444,__READ_WRITE ,__pri17_bits);
__IO_REG32_BIT(IP18,              0xE000E448,__READ_WRITE ,__pri18_bits);
__IO_REG32_BIT(IP19,              0xE000E44C,__READ_WRITE ,__pri19_bits);
__IO_REG32_BIT(IP20,              0xE000E450,__READ_WRITE ,__pri20_bits);
__IO_REG32_BIT(IP21,              0xE000E454,__READ_WRITE ,__pri21_bits);
__IO_REG32_BIT(IP22,              0xE000E458,__READ_WRITE ,__pri22_bits);
__IO_REG32_BIT(IP23,              0xE000E45C,__READ_WRITE ,__pri23_bits);
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(SHPR0,             0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,             0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,             0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x40000000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x40000004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAFR1,                0x40000008,__READ_WRITE ,__pafr1_bits);
__IO_REG8_BIT(PAFR2,                0x4000000C,__READ_WRITE ,__pafr2_bits);
__IO_REG8_BIT(PAOD,                 0x40000028,__READ_WRITE ,__paod_bits);
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
__IO_REG8_BIT(PBOD,                 0x40000068,__READ_WRITE ,__pbod_bits);
__IO_REG8_BIT(PBPUP,                0x4000006C,__READ_WRITE ,__pbpup_bits);
__IO_REG8_BIT(PBPDN,                0x40000070,__READ_WRITE ,__pbpdn_bits);
__IO_REG8_BIT(PBIE,                 0x40000078,__READ_WRITE ,__pbie_bits);

/***************************************************************************
 **
 ** PORTC
 **
 ***************************************************************************/
__IO_REG8_BIT(PCDATA,               0x40000080,__READ_WRITE ,__pc_bits);
__IO_REG8_BIT(PCCR,                 0x40000084,__READ_WRITE ,__pccr_bits);
__IO_REG8_BIT(PCFR1,                0x40000088,__READ_WRITE ,__pcfr1_bits);
__IO_REG8_BIT(PCFR2,                0x4000008C,__READ_WRITE ,__pcfr2_bits);
__IO_REG8_BIT(PCFR3,                0x40000090,__READ_WRITE ,__pcfr3_bits);
__IO_REG8_BIT(PCFR4,                0x40000094,__READ_WRITE ,__pcfr4_bits);
__IO_REG8_BIT(PCFR5,                0x40000098,__READ_WRITE ,__pcfr5_bits);
__IO_REG8_BIT(PCOD,                 0x400000A8,__READ_WRITE ,__pcod_bits);
__IO_REG8_BIT(PCPUP,                0x400000AC,__READ_WRITE ,__pcpup_bits);
__IO_REG8_BIT(PCPDN,                0x400000B0,__READ_WRITE ,__pcpdn_bits);
__IO_REG8_BIT(PCIE,                 0x400000B8,__READ_WRITE ,__pcie_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG8_BIT(PDDATA,               0x400000C0,__READ_WRITE ,__pd_bits);
__IO_REG8_BIT(PDCR,                 0x400000C4,__READ_WRITE ,__pdcr_bits);
__IO_REG8_BIT(PDFR1,                0x400000C8,__READ_WRITE ,__pdfr1_bits);
__IO_REG8_BIT(PDFR2,                0x400000CC,__READ_WRITE ,__pdfr2_bits);
__IO_REG8_BIT(PDFR3,                0x400000D0,__READ_WRITE ,__pdfr3_bits);
__IO_REG8_BIT(PDOD,                 0x400000E8,__READ_WRITE ,__pdod_bits);
__IO_REG8_BIT(PDPUP,                0x400000EC,__READ_WRITE ,__pdpup_bits);
__IO_REG8_BIT(PDPDN,                0x400000F0,__READ_WRITE ,__pdpdn_bits);
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
__IO_REG8_BIT(PEPDN,                0x40000130,__READ_WRITE ,__pepdn_bits);
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
__IO_REG8_BIT(PFFR3,                0x40000150,__READ_WRITE ,__pffr3_bits);
__IO_REG8_BIT(PFOD,                 0x40000168,__READ_WRITE ,__pfod_bits);
__IO_REG8_BIT(PFPUP,                0x4000016C,__READ_WRITE ,__pfpup_bits);
__IO_REG8_BIT(PFPDN,                0x40000170,__READ_WRITE ,__pfpdn_bits);
__IO_REG8_BIT(PFIE,                 0x40000178,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG8_BIT(PGDATA,               0x40000180,__READ_WRITE ,__pg_bits);
__IO_REG8_BIT(PGCR,                 0x40000184,__READ_WRITE ,__pgcr_bits);
__IO_REG8_BIT(PGFR1,                0x40000188,__READ_WRITE ,__pgfr1_bits);
__IO_REG8_BIT(PGFR2,                0x4000018C,__READ_WRITE ,__pgfr2_bits);
__IO_REG8_BIT(PGFR3,                0x40000190,__READ_WRITE ,__pgfr3_bits);
__IO_REG8_BIT(PGOD,                 0x400001A8,__READ_WRITE ,__pgod_bits);
__IO_REG8_BIT(PGPUP,                0x400001AC,__READ_WRITE ,__pgpup_bits);
__IO_REG8_BIT(PGPDN,                0x400001B0,__READ_WRITE ,__pgpdn_bits);
__IO_REG8_BIT(PGIE,                 0x400001B8,__READ_WRITE ,__pgie_bits);

/***************************************************************************
 **
 ** PORTH
 **
 ***************************************************************************/
__IO_REG8_BIT(PHDATA,               0x400001C0,__READ_WRITE ,__ph_bits);
__IO_REG8_BIT(PHCR,                 0x400001C4,__READ_WRITE ,__phcr_bits);
__IO_REG8_BIT(PHFR1,                0x400001C8,__READ_WRITE ,__phfr1_bits);
__IO_REG8_BIT(PHOD,                 0x400001E8,__READ_WRITE ,__phod_bits);
__IO_REG8_BIT(PHPUP,                0x400001EC,__READ_WRITE ,__phpup_bits);
__IO_REG8_BIT(PHPDN,                0x400001F0,__READ_WRITE ,__phpdn_bits);
__IO_REG8_BIT(PHIE,                 0x400001F8,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTI
 **
 ***************************************************************************/
__IO_REG8_BIT(PIDATA,               0x40000200,__READ_WRITE ,__pi_bits);
__IO_REG8_BIT(PICR,                 0x40000204,__READ_WRITE ,__picr_bits);
__IO_REG8_BIT(PIOD,                 0x40000228,__READ_WRITE ,__piod_bits);
__IO_REG8_BIT(PIPUP,                0x4000022C,__READ_WRITE ,__pipup_bits);
__IO_REG8_BIT(PIPDN,                0x40000230,__READ_WRITE ,__pipdn_bits);
__IO_REG8_BIT(PIIE,                 0x40000238,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x40000240,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJCR,                 0x40000244,__READ_WRITE ,__pjcr_bits);
__IO_REG8_BIT(PJFR1,                0x40000248,__READ_WRITE ,__pjfr1_bits);
__IO_REG8_BIT(PJOD,                 0x40000268,__READ_WRITE ,__pjod_bits);
__IO_REG8_BIT(PJPUP,                0x4000026C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJPDN,                0x40000270,__READ_WRITE ,__pjpdn_bits);
__IO_REG8_BIT(PJIE,                 0x40000278,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTL
 **
 ***************************************************************************/
__IO_REG8_BIT(PLDATA,               0x400002C0,__READ_WRITE ,__pl_bits);
__IO_REG8_BIT(PLCR,                 0x400002C4,__READ_WRITE ,__plcr_bits);
__IO_REG8_BIT(PLFR1,                0x400002C8,__READ_WRITE ,__plfr1_bits);
__IO_REG8_BIT(PLOD,                 0x400002E8,__READ_WRITE ,__plod_bits);
__IO_REG8_BIT(PLPUP,                0x400002EC,__READ_WRITE ,__plpup_bits);
__IO_REG8_BIT(PLPDN,                0x400002F0,__READ_WRITE ,__plpdn_bits);
__IO_REG8_BIT(PLIE,                 0x400002F8,__READ_WRITE ,__plie_bits);

/***************************************************************************
 **
 ** PORTM
 **
 ***************************************************************************/
__IO_REG8_BIT(PMDATA,               0x40000300,__READ_WRITE ,__pm_bits);
__IO_REG8_BIT(PMCR,                 0x40000304,__READ_WRITE ,__pmcr_bits);
__IO_REG8_BIT(PMOD,                 0x40000328,__READ_WRITE ,__pmod_bits);
__IO_REG8_BIT(PMPUP,                0x4000032C,__READ_WRITE ,__pmpup_bits);
__IO_REG8_BIT(PMPDN,                0x40000330,__READ_WRITE ,__pmpdn_bits);
__IO_REG8_BIT(PMIE,                 0x40000338,__READ_WRITE ,__pmie_bits);

/***************************************************************************
 **
 ** PORTN
 **
 ***************************************************************************/
__IO_REG8_BIT(PNDATA,               0x40000340,__READ_WRITE ,__pn_bits);
__IO_REG8_BIT(PNCR,                 0x40000344,__READ_WRITE ,__pncr_bits);
__IO_REG8_BIT(PNFR1,                0x40000348,__READ_WRITE ,__pnfr1_bits);
__IO_REG8_BIT(PNFR2,                0x4000034C,__READ_WRITE ,__pnfr2_bits);
__IO_REG8_BIT(PNOD,                 0x40000368,__READ_WRITE ,__pnod_bits);
__IO_REG8_BIT(PNPUP,                0x4000036C,__READ_WRITE ,__pnpup_bits);
__IO_REG8_BIT(PNPDN,                0x40000370,__READ_WRITE ,__pnpdn_bits);
__IO_REG8_BIT(PNIE,                 0x40000378,__READ_WRITE ,__pnie_bits);

/***************************************************************************
 **
 ** PORTP
 **
 ***************************************************************************/
__IO_REG8_BIT(PPDATA,               0x40000380,__READ_WRITE ,__pp_bits);
__IO_REG8_BIT(PPCR,                 0x40000384,__READ_WRITE ,__ppcr_bits);
__IO_REG8_BIT(PPOD,                 0x400003A8,__READ_WRITE ,__ppod_bits);
__IO_REG8_BIT(PPPUP,                0x400003AC,__READ_WRITE ,__pppup_bits);
__IO_REG8_BIT(PPPDN,                0x400003B0,__READ_WRITE ,__pppdn_bits);
__IO_REG8_BIT(PPIE,                 0x400003B8,__READ_WRITE ,__ppie_bits);

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
 ** MTP0
 **
 ***************************************************************************/
__IO_REG32_BIT(MT0EN,               0x40050800, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT0RUN,              0x40050804, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT0CR,               0x40050808, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT0MOD,              0x4005080C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT0FFCR,             0x40050810, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT0ST,               0x40050814, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT0IM,               0x40050818, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT0UC,               0x4005081C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT0RG0,              0x40050820, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT0RG1,              0x40050824, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT0CP0,              0x40050828, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT0CP1,              0x4005082C, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MTIG0CR,             0x40050830, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MTIG0RESTA,          0x40050834, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MTIG0ST,             0x40050838, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MTIG0ICR,            0x4005083C, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MTIG0OCR,            0x40050840, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MTIG0RG2,            0x40050844, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MTIG0RG3,            0x40050848, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MTIG0RG4,            0x4005084C, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MTIG0EMGCR,          0x40050850, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MTIG0EMGST,          0x40050854, __READ       , __mtigxemgst_bits);

/***************************************************************************
 **
 ** MTP0 PMD
 **
 ***************************************************************************/
__IO_REG32_BIT(MTPD0MDEN,            0x40050400, __READ_WRITE , __mden_bits);
__IO_REG32_BIT(MTPD0PORTMD,          0x40050404, __READ_WRITE , __portmd_bits);
__IO_REG32_BIT(MTPD0MDCR,            0x40050408, __READ_WRITE , __mdcr_bits);
__IO_REG32_BIT(MTPD0CNTSTA,          0x4005040C, __READ       , __cntsta_bits);
__IO_REG32_BIT(MTPD0MDCNT,           0x40050410, __READ       , __mdcnt_bits);
__IO_REG32_BIT(MTPD0MDPRD,           0x40050414, __READ_WRITE , __mdprd_bits);
__IO_REG32_BIT(MTPD0CMPU,            0x40050418, __READ_WRITE , __cmpu_bits);
__IO_REG32_BIT(MTPD0CMPV,            0x4005041C, __READ_WRITE , __cmpv_bits);
__IO_REG32_BIT(MTPD0CMPW,            0x40050420, __READ_WRITE , __cmpw_bits);
__IO_REG32_BIT(MTPD0MODESEL,         0x40050424, __READ_WRITE , __modesel_bits);
__IO_REG32_BIT(MTPD0MDOUT,           0x40050428, __READ_WRITE , __mdout_bits);
__IO_REG32_BIT(MTPD0MDPOT,           0x4005042C, __READ_WRITE , __mdpot_bits);
__IO_REG32_BIT(MTPD0REL,             0x40050430, __WRITE      , __emgrel_bits);
__IO_REG32_BIT(MTPD0EMGCR,           0x40050434, __READ_WRITE , __emgcr_bits);
__IO_REG32_BIT(MTPD0EMGST,           0x40050438, __READ       , __emgsta_bits);
__IO_REG32_BIT(MTPD0DTR,             0x40050444, __READ_WRITE , __dtr_bits);
__IO_REG32_BIT(MTPD0TRGCMP0,         0x40050448, __READ_WRITE , __trgcmp0_bits);
__IO_REG32_BIT(MTPD0TRGCMP1,         0x4005044C, __READ_WRITE , __trgcmp1_bits);
__IO_REG32_BIT(MTPD0TRGCMP2,         0x40050450, __READ_WRITE , __trgcmp2_bits);
__IO_REG32_BIT(MTPD0TRGCMP3,         0x40050454, __READ_WRITE , __trgcmp3_bits);
__IO_REG32_BIT(MTPD0TRGCR,           0x40050458, __READ_WRITE , __trgcr_bits);
__IO_REG32_BIT(MTPD0TRGMD,           0x4005045C, __READ_WRITE , __trgmd_bits);

/***************************************************************************
 **
 ** MTP1
 **
 ***************************************************************************/
__IO_REG32_BIT(MT1EN,               0x40050880, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT1RUN,              0x40050884, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT1CR,               0x40050888, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT1MOD,              0x4005088C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT1FFCR,             0x40050890, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT1ST,               0x40050894, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT1IM,               0x40050898, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT1UC,               0x4005089C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT1RG0,              0x400508A0, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT1RG1,              0x400508A4, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT1CP0,              0x400508A8, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT1CP1,              0x400508AC, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MTIG1CR,             0x400508B0, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MTIG1RESTA,          0x400508B4, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MTIG1ST,             0x400508B8, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MTIG1ICR,            0x400508BC, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MTIG1OCR,            0x400508C0, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MTIG1RG2,            0x400508C4, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MTIG1RG3,            0x400508C8, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MTIG1RG4,            0x400508CC, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MTIG1EMGCR,          0x400508D0, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MTIG1EMGST,          0x400508D4, __READ       , __mtigxemgst_bits);

/***************************************************************************
 **
 ** MTP1 PMD
 **
 ***************************************************************************/
__IO_REG32_BIT(MTPD1MDEN,            0x40050480, __READ_WRITE , __mden_bits);
__IO_REG32_BIT(MTPD1PORTMD,          0x40050484, __READ_WRITE , __portmd_bits);
__IO_REG32_BIT(MTPD1MDCR,            0x40050488, __READ_WRITE , __mdcr_bits);
__IO_REG32_BIT(MTPD1CNTSTA,          0x4005048C, __READ       , __cntsta_bits);
__IO_REG32_BIT(MTPD1MDCNT,           0x40050490, __READ       , __mdcnt_bits);
__IO_REG32_BIT(MTPD1MDPRD,           0x40050494, __READ_WRITE , __mdprd_bits);
__IO_REG32_BIT(MTPD1CMPU,            0x40050498, __READ_WRITE , __cmpu_bits);
__IO_REG32_BIT(MTPD1CMPV,            0x4005049C, __READ_WRITE , __cmpv_bits);
__IO_REG32_BIT(MTPD1CMPW,            0x400504A0, __READ_WRITE , __cmpw_bits);
__IO_REG32_BIT(MTPD1MODESEL,         0x400504A4, __READ_WRITE , __modesel_bits);
__IO_REG32_BIT(MTPD1MDOUT,           0x400504A8, __READ_WRITE , __mdout_bits);
__IO_REG32_BIT(MTPD1MDPOT,           0x400504AC, __READ_WRITE , __mdpot_bits);
__IO_REG32_BIT(MTPD1REL,             0x400504B0, __WRITE      , __emgrel_bits);
__IO_REG32_BIT(MTPD1EMGCR,           0x400504B4, __READ_WRITE , __emgcr_bits);
__IO_REG32_BIT(MTPD1EMGST,           0x400504B8, __READ       , __emgsta_bits);
__IO_REG32_BIT(MTPD1DTR,             0x400504C4, __READ_WRITE , __dtr_bits);
__IO_REG32_BIT(MTPD1TRGCMP0,         0x400504C8, __READ_WRITE , __trgcmp0_bits);
__IO_REG32_BIT(MTPD1TRGCMP1,         0x400504CC, __READ_WRITE , __trgcmp1_bits);
__IO_REG32_BIT(MTPD1TRGCMP2,         0x400504D0, __READ_WRITE , __trgcmp2_bits);
__IO_REG32_BIT(MTPD1TRGCMP3,         0x400504D4, __READ_WRITE , __trgcmp3_bits);
__IO_REG32_BIT(MTPD1TRGCR,           0x400504D8, __READ_WRITE , __trgcr_bits);
__IO_REG32_BIT(MTPD1TRGMD,           0x400504DC, __READ_WRITE , __trgmd_bits);

/***************************************************************************
 **
 ** MTP2
 **
 ***************************************************************************/
__IO_REG32_BIT(MT2EN,               0x40050900, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT2RUN,              0x40050904, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT2CR,               0x40050908, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT2MOD,              0x4005090C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT2FFCR,             0x40050910, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT2ST,               0x40050914, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT2IM,               0x40050918, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT2UC,               0x4005091C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT2RG0,              0x40050920, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT2RG1,              0x40050924, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT2CP0,              0x40050928, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT2CP1,              0x4005092C, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MTIG2CR,             0x40050930, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MTIG2RESTA,          0x40050934, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MTIG2ST,             0x40050938, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MTIG2ICR,            0x4005093C, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MTIG2OCR,            0x40050940, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MTIG2RG2,            0x40050944, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MTIG2RG3,            0x40050948, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MTIG2RG4,            0x4005094C, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MTIG2EMGCR,          0x40050950, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MTIG2EMGST,          0x40050954, __READ       , __mtigxemgst_bits);


/***************************************************************************
 **
 ** ENC 0
 **
 ***************************************************************************/
__IO_REG32_BIT(EN0TNCR,             0x40010400, __READ_WRITE , __enxtncr_bits);
__IO_REG32_BIT(EN0RELOAD,           0x40010404, __READ_WRITE , __enxreload_bits);
__IO_REG32_BIT(EN0INT,              0x40010408, __READ_WRITE , __enxint_bits);
__IO_REG32_BIT(EN0CNT,              0x4001040C, __READ_WRITE , __enxcnt_bits);

/***************************************************************************
 **
 ** ENC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(EN1TNCR,             0x40010500, __READ_WRITE , __enxtncr_bits);
__IO_REG32_BIT(EN1RELOAD,           0x40010504, __READ_WRITE , __enxreload_bits);
__IO_REG32_BIT(EN1INT,              0x40010508, __READ_WRITE , __enxint_bits);
__IO_REG32_BIT(EN1CNT,              0x4001050C, __READ_WRITE , __enxcnt_bits);

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
 ** SIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(SC3EN,               0x40020140, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC3BUF,              0x40020144, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC3CR,               0x40020148, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC3MOD0,             0x4002014C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC3BRCR,             0x40020150, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC3BRADD,            0x40020154, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC3MOD1,             0x40020158, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC3MOD2,             0x4002015C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC3RFC,              0x40020160, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC3TFC,              0x40020164, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC3RST,              0x40020168, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC3TST,              0x4002016C, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC3FCNF,             0x40020170, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO4
 **
 ***************************************************************************/
__IO_REG32_BIT(SC4EN,               0x40020180, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC4BUF,              0x40020184, __READ_WRITE , __scxbuf_bits);
__IO_REG32_BIT(SC4CR,               0x40020188, __READ_WRITE , __scxcr_bits);
__IO_REG32_BIT(SC4MOD0,             0x4002018C, __READ_WRITE , __scxmod0_bits);
__IO_REG32_BIT(SC4BRCR,             0x40020190, __READ_WRITE , __scxbrcr_bits);
__IO_REG32_BIT(SC4BRADD,            0x40020194, __READ_WRITE , __scxbradd_bits);
__IO_REG32_BIT(SC4MOD1,             0x40020198, __READ_WRITE , __scxmod1_bits);
__IO_REG32_BIT(SC4MOD2,             0x4002019C, __READ_WRITE , __scxmod2_bits);
__IO_REG32_BIT(SC4RFC,              0x400201A0, __READ_WRITE , __scxrfc_bits);
__IO_REG32_BIT(SC4TFC,              0x400201A4, __READ_WRITE , __scxtfc_bits);
__IO_REG32_BIT(SC4RST,              0x400201A8, __READ       , __scxrst_bits);
__IO_REG32_BIT(SC4TST,              0x400201AC, __READ       , __scxtst_bits);
__IO_REG32_BIT(SC4FCNF,             0x400201B0, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SSP 0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,             0x400C0000, __READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,             0x400C0004, __READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSP0DR,              0x400C0008, __READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSP0SR,              0x400C000C, __READ       , __sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,            0x400C0010, __READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,            0x400C0014, __READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,             0x400C0018, __READ       , __sspris_bits);
__IO_REG32_BIT(SSP0MIS,             0x400C001C, __READ       , __sspmis_bits);
__IO_REG32_BIT(SSP0ICR,             0x400C0020, __WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,           0x400C0024, __READ_WRITE , __sspdmacr_bits);

/***************************************************************************
 **
 ** SSP 1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,             0x400C1000, __READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,             0x400C1004, __READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSP1DR,              0x400C1008, __READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSP1SR,              0x400C100C, __READ       , __sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,            0x400C1010, __READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,            0x400C1014, __READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,             0x400C1018, __READ       , __sspris_bits);
__IO_REG32_BIT(SSP1MIS,             0x400C101C, __READ       , __sspmis_bits);
__IO_REG32_BIT(SSP1ICR,             0x400C1020, __WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,           0x400C1024, __READ_WRITE , __sspdmacr_bits);

/***************************************************************************
 **
 ** SBI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SBI0CR0,             0x40020000, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(SBI0CR1,             0x40020004, __READ_WRITE , __sbixcr1_bits);
#define _SBI0CR1 SBI0CR1
#define _SBI0CR1_bit SBI0CR1_bit
__IO_REG32_BIT(SBI0DBR,             0x40020008, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(SBI0I2CAR,           0x4002000C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(SBI0CR2,             0x40020010, __READ_WRITE , __sbixcr2_sr_bits);
#define SBI0SR SBI0CR2
#define SBI0SR_bit SBI0CR2_bit
#define _SBI0SR SBI0CR2
#define _SBI0SR_bit SBI0CR2_bit
__IO_REG32_BIT(SBI0BR0,             0x40020014, __READ_WRITE , __sbixbr0_bits);

/***************************************************************************
 **
 ** SBI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SBI1CR0,             0x40020020, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(SBI1CR1,             0x40020024, __READ_WRITE , __sbixcr1_bits);
#define _SBI1CR1 SBI1CR1
#define _SBI1CR1_bit SBI1CR1_bit
__IO_REG32_BIT(SBI1DBR,             0x40020028, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(SBI1I2CAR,           0x4002002C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(SBI1CR2,             0x40020030, __READ_WRITE , __sbixcr2_sr_bits);
#define SBI1SR SBI1CR2
#define SBI1SR_bit SBI1CR2_bit
#define _SBI1SR SBI1CR2
#define _SBI1SR_bit SBI1CR2_bit
__IO_REG32_BIT(SBI1BR0,             0x40020034, __READ_WRITE , __sbixbr0_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG8_BIT(ADCLK,                0x40030000, __READ_WRITE , __adxclk_bits);
__IO_REG8_BIT(ADMOD0,               0x40030004, __READ_WRITE , __adxmod0_bits);
__IO_REG8_BIT(ADMOD1,               0x40030008, __READ_WRITE , __adxmod1_bits);
__IO_REG8_BIT(ADMOD2,               0x4003000C, __READ       , __adxmod2_bits);
__IO_REG16_BIT(ADCMPCR0,            0x40030010, __READ_WRITE , __adxcmpcrx_bits);
__IO_REG16_BIT(ADCMPCR1,            0x40030014, __READ_WRITE , __adxcmpcrx_bits);
__IO_REG16_BIT(ADCMP0,              0x40030018, __READ_WRITE , __adxcmpx_bits);
__IO_REG16_BIT(ADCMP1,              0x4003001C, __READ_WRITE , __adxcmpx_bits);
__IO_REG16_BIT(ADREG0,              0x40030020, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG1,              0x40030024, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG2,              0x40030028, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG3,              0x4003002C, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG4,              0x40030030, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG5,              0x40030034, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG6,              0x40030038, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG7,              0x4003003C, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG8,              0x40030040, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG9,              0x40030044, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG10,             0x40030048, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADREG11,             0x4003004C, __READ       , __adxregx_bits);
__IO_REG8_BIT(ADPSEL0,              0x40030050, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADPSEL1,              0x40030054, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADPSEL2,              0x40030058, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADPSEL3,              0x4003005C, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADPINTS0,             0x40030080, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADPINTS1,             0x40030084, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADPINTS2,             0x40030088, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADPINTS3,             0x4003008C, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADPINTS4,             0x40030090, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADPINTS5,             0x40030094, __READ_WRITE , __adxpintsx_bits);
__IO_REG32_BIT(ADPSET0,             0x40030098, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADPSET1,             0x4003009C, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADPSET2,             0x400300A0, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADPSET3,             0x400300A4, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADPSET4,             0x400300A8, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADPSET5,             0x400300AC, __READ_WRITE , __adxpsetx_bits);
__IO_REG8_BIT(ADTSET0,              0x400300B0, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET1,              0x400300B1, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET2,              0x400300B2, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET3,              0x400300B3, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET4,              0x400300B4, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET5,              0x400300B5, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET6,              0x400300B6, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET7,              0x400300B7, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET8,              0x400300B8, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET9,              0x400300B9, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET10,             0x400300BA, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADTSET11,             0x400300BB, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADSSET0,              0x400300BC, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET1,              0x400300BD, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET2,              0x400300BE, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET3,              0x400300BF, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET4,              0x400300C0, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET5,              0x400300C1, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET6,              0x400300C2, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET7,              0x400300C3, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET8,              0x400300C4, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET9,              0x400300C5, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET10,             0x400300C6, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADSSET11,             0x400300C7, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADASET0,              0x400300C8, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET1,              0x400300C9, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET2,              0x400300CA, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET3,              0x400300CB, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET4,              0x400300CC, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET5,              0x400300CD, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET6,              0x400300CE, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET7,              0x400300CF, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET8,              0x400300D0, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET9,              0x400300D1, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET10,             0x400300D2, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADASET11,             0x400300D3, __READ_WRITE , __adxasetx_bits);
__IO_REG16_BIT(ADMOD3,              0x400300D4, __READ_WRITE , __adxmod3_bits);

/***************************************************************************
 **
 ** RMC
 **
 ***************************************************************************/
__IO_REG32_BIT(RMCEN,              0x40040400, __READ_WRITE , __rmcen_bits );
__IO_REG32_BIT(RMCREN,             0x40040404, __READ_WRITE , __rmcren_bits );
__IO_REG32(    RMCRBUF1,           0x40040408, __READ);
__IO_REG32(    RMCRBUF2,           0x4004040C, __READ);
__IO_REG32(    RMCRBUF3,           0x40040410, __READ);
__IO_REG32_BIT(RMCRCR1,            0x40040414, __READ_WRITE , __rmcrcr1_bits );
__IO_REG32_BIT(RMCRCR2,            0x40040418, __READ_WRITE , __rmcrcr2_bits );
__IO_REG32_BIT(RMCRCR3,            0x4004041C, __READ_WRITE , __rmcrcr3_bits );
__IO_REG32_BIT(RMCRCR4,            0x40040420, __READ_WRITE , __rmcrcr4_bits );
__IO_REG32_BIT(RMCRSTAT,           0x40040424, __READ       , __rmcrstat_bits );
__IO_REG32_BIT(RMCEND1,            0x40040428, __READ_WRITE , __rmcend1_bits );
__IO_REG32_BIT(RMCEND2,            0x4004042C, __READ_WRITE , __rmcend2_bits );
__IO_REG32_BIT(RMCEND3,            0x40040430, __READ_WRITE , __rmcend3_bits );
__IO_REG32_BIT(RMCFSSEL,           0x40040434, __READ_WRITE , __rmcfssel_bits );

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
__IO_REG32_BIT(CGIMCGE,             0x40040230, __READ_WRITE ,__imcge_bits);
 
/***************************************************************************
 **
 ** OFD
 **
 ***************************************************************************/
__IO_REG32_BIT(OFDCR1,              0x40040800, __READ_WRITE ,__ofdcr1_bits);
__IO_REG32_BIT(OFDCR2,              0x40040804, __READ_WRITE ,__ofdcr2_bits);
__IO_REG32_BIT(OFDMN,               0x40040808, __READ_WRITE ,__ofdmn_bits);
__IO_REG32_BIT(OFDMX,               0x40040810, __READ_WRITE ,__ofdmx_bits);
__IO_REG32_BIT(OFDRST,              0x40040818, __READ_WRITE ,__ofdrst_bits);
__IO_REG32_BIT(OFDSTAT,             0x4004081C, __READ       ,__ofdstat_bits);

/***************************************************************************
 **
 ** VLTD
 **
 ***************************************************************************/
__IO_REG32_BIT(VDCR,                0x40040900, __READ_WRITE , __vdcr_bits);
__IO_REG32_BIT(VDSR,                0x40040904, __READ       , __vdsr_bits);

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACIntStaus,          0x40080000,__READ       ,__dmacintstaus_bits);
__IO_REG32_BIT(DMACIntTCStatus,       0x40080004,__READ       ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACIntTCClear,        0x40080008,__WRITE      ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACIntErrorStatus,    0x4008000C,__READ       ,__dmacinterrorstatus_bits);
__IO_REG32_BIT(DMACIntErrClr,         0x40080010,__WRITE      ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRawIntTCStatus,    0x40080014,__READ       ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRawIntErrorStatus, 0x40080018,__READ       ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACEnbldChns,         0x4008001C,__READ       ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSoftBReq,          0x40080020,__READ_WRITE ,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSoftSReq,          0x40080024,__READ_WRITE ,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACConfiguration,     0x40080030,__READ_WRITE ,__dmacconfiguration_bits);
__IO_REG32(    DMACC0SrcAddr,         0x40080100,__READ_WRITE );
__IO_REG32(    DMACC0DestAddr,        0x40080104,__READ_WRITE );
__IO_REG32(    DMACC0LLI,             0x40080108,__READ_WRITE );
__IO_REG32_BIT(DMACC0Control,         0x4008010C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC0Configuration,   0x40080110,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC1SrcAddr,         0x40080120,__READ_WRITE );
__IO_REG32(    DMACC1DestAddr,        0x40080124,__READ_WRITE );
__IO_REG32(    DMACC1LLI,             0x40080128,__READ_WRITE );
__IO_REG32_BIT(DMACC1Control,         0x4008012C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC1Configuration,   0x40080130,__READ_WRITE ,__dmaccconfiguration_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FCSECBIT,              0x41FFF010, __READ_WRITE , __fcsecbit_bits);
__IO_REG32_BIT(FCFLCS,                0x41FFF020, __READ       , __fcflcs_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  DMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define _DMA_SIO0RXTX         0
#define _DMA_SIO1RXTX         1
#define _DMA_SIO2RXTX         2
#define _DMA_SIO3RXTX         3
#define _DMA_SIO4RXTX         4
#define _DMA_SSP0TX           5
#define _DMA_SSP0RX           6
#define _DMA_SSP1TX           7
#define _DMA_SSP1RX           8

/***************************************************************************
 **
 **  TMPM380FxFG Interrupt Lines
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
#define INT0                 ( 0 + EII)   /* Interrupt Pin (PH0/AIN0/INT0*/
#define INT1                 ( 1 + EII)   /* Interrupt Pin (PH1/AIN1/INT1*/
#define INT2                 ( 2 + EII)   /* Interrupt Pin (PH2/AIN2/INT2*/
#define INT3                 ( 3 + EII)   /* Interrupt Pin (PA0/TB0IN/INT3*/
#define INT4                 ( 4 + EII)   /* Interrupt Pin (PA2/TB1IN/INT4*/
#define INT5                 ( 5 + EII)   /* Interrupt Pin (PE4/TB2IN/INT5*/
#define INTRX0               ( 6 + EII)   /* Serial reception (channel.0) */
#define INTTX0               ( 7 + EII)   /* Serial transmit (channel.0)  */
#define INTRX1               ( 8 + EII)   /* Serial reception (channel.1) */
#define INTTX1               ( 9 + EII)   /* Serial transmit (channel.1)  */
#define INTSSP0              (10 + EII)   /* Syncronous Serial Port 0     */
#define INTSSP1              (11 + EII)   /* Syncronous Serial Port 1     */
#define INTEMG0              (12 + EII)   /* PMD0 EMG interrupt           */
#define INTEMG1              (13 + EII)   /* PMD1 EMG interrupt           */
#define INTSBI0              (14 + EII)   /* Serial Bus Interface 0 interrupt             */
#define INTSBI1              (15 + EII)   /* Serial Bus Interface 1 interrupt             */
#define INTADPD0             (16 + EII)   /* ADC conversion triggered by PMD0 is finished */
#define INTRTC               (17 + EII)   /* Realtime clock interrupt     */
#define INTADPD1             (18 + EII)   /* ADC conversion triggered by PMD1 is finished */
#define INTRMCRX             (19 + EII)   /* Remote Controller reception interrupt        */
#define INTTB00              (20 + EII)   /* 16bit TMRB0 compare match detection 0/ Over flow*/
#define INTTB01              (21 + EII)   /* 16bit TMRB0 compare match detection 1           */
#define INTTB10              (22 + EII)   /* 16bit TMRB1 compare match detection 0/ Over flow*/
#define INTTB11              (23 + EII)   /* 16bit TMRB1 compare match detection 1           */
#define INTTB40              (24 + EII)   /* 16bit TMRB4 compare match detection 0/ Over flow*/
#define INTTB41              (25 + EII)   /* 16bit TMRB4 compare match detection 1           */
#define INTTB50              (26 + EII)   /* 16bit TMRB5 compare match detection 0/ Over flow*/
#define INTTB51              (27 + EII)   /* 16bit TMRB5 compare match detection 1           */
#define INTPMD0              (28 + EII)   /* PMD0 PWM interrupt (MPT0)    */
#define INTPMD1              (29 + EII)   /* PMD1 PWM interrupt (MPT1)    */
#define INTCAP00             (30 + EII)   /* 16bit TMRB0 input capture 0  */
#define INTCAP01             (31 + EII)   /* 16bit TMRB0 input capture 1  */
#define INTCAP10             (32 + EII)   /* 16bit TMRB1 input capture 0  */
#define INTCAP11             (33 + EII)   /* 16bit TMRB1 input capture 1  */
#define INTCAP40             (34 + EII)   /* 16bit TMRB4 input capture 0  */
#define INTCAP41             (35 + EII)   /* 16bit TMRB4 input capture 1  */
#define INTCAP50             (36 + EII)   /* 16bit TMRB5 input capture 0  */
#define INTCAP51             (37 + EII)   /* 16bit TMRB5 input capture 1  */
#define INT6                 (38 + EII)   /* Interrupt Pin (PE6/TB3IN/INT6)*/
#define INT7                 (39 + EII)   /* Interrupt Pin (PE7/TB3OUT/INT7)*/
#define INTRX2               (40 + EII)   /* Serial reception (channel.2) */
#define INTTX2               (41 + EII)   /* Serial transmit (channel.2)  */
#define INTADCP0             (42 + EII)   /* ADC conversion monitoring function interrupt 0 */
#define INTADCP1             (43 + EII)   /* ADC conversion monitoring function interrupt 1 */
#define INTRX4               (44 + EII)   /* Serial reception (channel.4) */
#define INTTX4               (45 + EII)   /* Serial transmit (channel.4)  */
#define INTTB20              (46 + EII)   /* 16bit TMRB2 compare match detection 0/ Over flow*/
#define INTTB21              (47 + EII)   /* 16bit TMRB2 compare match detection 1           */
#define INTTB30              (48 + EII)   /* 16bit TMRB3 compare match detection 0/ Over flow*/
#define INTTB31              (49 + EII)   /* 16bit TMRB3 compare match detection 1           */
#define INTCAP20             (50 + EII)   /* 16bit TMRB2 input capture 0  */
#define INTCAP21             (51 + EII)   /* 16bit TMRB2 input capture 1  */
#define INTCAP30             (52 + EII)   /* 16bit TMRB3 input capture 0  */
#define INTCAP31             (53 + EII)   /* 16bit TMRB3 input capture 1  */
#define INTADSFT             (54 + EII)   /* ADC conversion started by software is finished    */
#define INTADTMR             (56 + EII)   /* ADC conversion triggered by timer is finished     */
#define INT8                 (58 + EII)   /* Interrupt Pin (PA7/TB4IN/INT8)                    */
#define INT9                 (59 + EII)   /* Interrupt Pin (PD3/INT9)                          */
#define INTA                 (60 + EII)   /* Interrupt Pin (PJ6/AIN6/INTA)                     */
#define INTB                 (61 + EII)   /* Interrupt Pin (PJ7/AIN7/INTB)                     */
#define INTENC0              (62 + EII)   /* Ender input0 interrupt                            */
#define INTENC1              (63 + EII)   /* Ender input1 interrupt                            */
#define INTRX3               (64 + EII)   /* Serial reception (channel.3)                      */
#define INTTX3               (65 + EII)   /* Serial transmit (channel.3)                       */
#define INTTB60              (66 + EII)   /* 16bit TMRB6 compare match detection 0 / Over flow */
#define INTTB61              (67 + EII)   /* 16bit TMRB6 compare match detection 1             */
#define INTTB70              (68 + EII)   /* 16bit TMRB7 compare match detection 0 / Over flow */
#define INTTB71              (69 + EII)   /* 16bit TMRB7 compare match detection 1     */
#define INTCAP60             (70 + EII)   /* 16bit TMRB6 input capture 0               */
#define INTCAP61             (71 + EII)   /* 16bit TMRB6 input capture 1               */
#define INTCAP70             (72 + EII)   /* 16bit TMRB7 input capture 0               */
#define INTCAP71             (73 + EII)   /* 16bit TMRB7 input capture 1               */
#define INTC                 (74 + EII)   /* Interrupt Pin (PD0/ENCA0/TB5IN/INTC)      */
#define INTD                 (75 + EII)   /* Interrupt Pin (PD2/ENCZ0/INTD)            */
#define INTE                 (76 + EII)   /* Interrupt Pin (PN7/MT2IN/INTE)            */
#define INTF                 (77 + EII)   /* Interrupt Pin (PL2/INTF)                  */
#define INTDMACERR           (78 + EII)   /* DMA transfer error                                            */
#define INTDMACTC            (79 + EII)   /* DMA end of transfer                                           */
#define INTMTPTB00           (80 + EII)   /* 16-bit MPT0 IGBT period/ compare match detection 0/ Over flow */
#define INTMTTTB01           (81 + EII)   /* 16-bit MPT0 IGBT trigger/ compare match detection 1           */
#define INTMTPTB10           (82 + EII)   /* 16-bit MPT1 IGBT period/ compare match detection 0/ Over flow */
#define INTMTTTB11           (83 + EII)   /* 16-bit MPT1 IGBT trigger/ compare match detection 1           */
#define INTMTPTB20           (84 + EII)   /* 16-bit MPT2 IGBT period/ compare match detection 0/ Over flow */
#define INTMTTTB21           (85 + EII)   /* 16-bit MPT2 IGBT trigger/ compare match detection 1           */
#define INTMTCAP00           (86 + EII)   /* 16-bit MPT0 input capture 0    */
#define INTMTCAP01           (87 + EII)   /* 16-bit MPT0 input capture 1    */
#define INTMTCAP10           (88 + EII)   /* 16-bit MPT1 input capture 0    */
#define INTMTCAP11           (89 + EII)   /* 16-bit MPT1 input capture 1    */
#define INTMTCAP20           (90 + EII)   /* 16-bit MPT2 input capture 0    */
#define INTMTCAP21           (91 + EII)   /* 16-bit MPT2 input capture 1    */
#define INTMTEMG0            (92 + EII)   /* 16-bit MPT0 IGBT EMG interrupt */
#define INTMTEMG1            (93 + EII)   /* 16-bit MPT1 IGBT EMG interrupt */
#define INTMTEMG2            (94 + EII)   /* 16-bit MPT2 IGBT EMG interrupt */

#endif    /* __IOTMPM380FxFG_H */

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
Interrupt19  = INTSSP0        0x68
Interrupt20  = INTSSP1        0x6C
Interrupt21  = INTEMG0        0x70
Interrupt22  = INTEMG1        0x74
Interrupt23  = INTSBI0        0x78
Interrupt24  = INTSBI1        0x7C
Interrupt25  = INTADPD0       0x80
Interrupt26  = INTRTC         0x84
Interrupt27  = INTADPD1       0x88
Interrupt28  = INTRMCRX       0x8C
Interrupt29  = INTTB00        0x90
Interrupt30  = INTTB01        0x94
Interrupt31  = INTTB10        0x98
Interrupt32  = INTTB11        0x9C
Interrupt33  = INTTB40        0xA0
Interrupt34  = INTTB41        0xA4
Interrupt35  = INTTB50        0xA8
Interrupt36  = INTTB51        0xAC
Interrupt37  = INTPMD0        0xB0
Interrupt38  = INTPMD1        0xB4
Interrupt39  = INTCAP00       0xB8
Interrupt40  = INTCAP01       0xBC
Interrupt41  = INTCAP10       0xC0
Interrupt42  = INTCAP11       0xC4
Interrupt43  = INTCAP40       0xC8
Interrupt44  = INTCAP41       0xCC
Interrupt45  = INTCAP50       0xD0
Interrupt46  = INTCAP51       0xD4
Interrupt47  = INT6           0xD8
Interrupt48  = INT7           0xDC
Interrupt49  = INTRX2         0xE0
Interrupt50  = INTTX2         0xE4
Interrupt51  = INTADCP0       0xE8
Interrupt52  = INTADCP1       0xEC
Interrupt53  = INTRX4         0xF0
Interrupt54  = INTTX4         0xF4
Interrupt55  = INTTB20        0xF8
Interrupt56  = INTTB21        0xFC
Interrupt57  = INTTB30        0x100
Interrupt58  = INTTB31        0x104
Interrupt59  = INTCAP20       0x108 
Interrupt60  = INTCAP21       0x10C 
Interrupt61  = INTCAP30       0x110 
Interrupt62  = INTCAP31       0x114 
Interrupt63  = INTADSFT       0x118 
Interrupt64  = INTADTMR       0x120 
Interrupt65  = INT8           0x128 
Interrupt66  = INT9           0x12C 
Interrupt67  = INTA           0x130 
Interrupt68  = INTB           0x134 
Interrupt69  = INTENC0        0x138 
Interrupt70  = INTENC1        0x13C 
Interrupt71  = INTRX3         0x140 
Interrupt72  = INTTX3         0x144 
Interrupt73  = INTTB60        0x148 
Interrupt74  = INTTB61        0x14C 
Interrupt75  = INTTB70        0x150 
Interrupt76  = INTTB71        0x154 
Interrupt77  = INTCAP60       0x158
Interrupt78  = INTCAP61       0x15C
Interrupt79  = INTCAP70       0x160
Interrupt80  = INTCAP71       0x164
Interrupt81  = INTC           0x168
Interrupt82  = INTD           0x16C
Interrupt83  = INTE           0x170
Interrupt84  = INTF           0x174
Interrupt85  = INTDMACERR     0x178
Interrupt86  = INTDMACTC      0x17C
Interrupt87  = INTMTPTB00     0x180
Interrupt88  = INTMTTTB01     0x184
Interrupt89  = INTMTPTB10     0x188
Interrupt90  = INTMTTTB11     0x18C
Interrupt91  = INTMTPTB20     0x190
Interrupt92  = INTMTTTB21     0x194
Interrupt93  = INTMTCAP00     0x198
Interrupt94  = INTMTCAP01     0x19C
Interrupt95  = INTMTCAP10     0x1A0
Interrupt96  = INTMTCAP11     0x1A4
Interrupt97  = INTMTCAP20     0x1A8
Interrupt98  = INTMTCAP21     0x1AC
Interrupt99  = INTMTEMG0      0x1B0
Interrupt100 = INTMTEMG1      0x1B4
Interrupt101 = INTMTEMG2      0x1B8

###DDF-INTERRUPT-END###*/
