/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM372FWUG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: #2 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM372FWUG_H
#define __IOTMPM372FWUG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM372FWUG SPECIAL FUNCTION REGISTERS
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
  __REG8         : 1;
  __REG8  PA2F2  : 1;
  __REG8         : 1;
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

/*PORT B Pull-Down Control Register */
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

/*PORT D Register*/
typedef struct {
  __REG8       : 4;
  __REG8  PD4  : 1;
  __REG8  PD5  : 1;
  __REG8  PD6  : 1;
  __REG8       : 1;
} __pd_bits;

/*Port D control register*/
typedef struct {
  __REG8        : 4;
  __REG8  PD4C  : 1;
  __REG8  PD5C  : 1;
  __REG8  PD6C  : 1;
  __REG8        : 1;
} __pdcr_bits;

/*PORT D Function Register 1*/
typedef struct {
  __REG8         : 4;
  __REG8  PD4F1  : 1;
  __REG8  PD5F1  : 1;
  __REG8  PD6F1  : 1;
  __REG8         : 1;
} __pdfr1_bits;

/*PORT D Function Register 2*/
typedef struct {
  __REG8         : 4;
  __REG8  PD4F2  : 1;
  __REG8         : 3;
} __pdfr2_bits;

/*Port D open drain control register */
typedef struct {
  __REG8         : 4;
  __REG8  PD4OD  : 1;
  __REG8  PD5OD  : 1;
  __REG8  PD6OD  : 1;
  __REG8         : 1;
} __pdod_bits;

/*PORT D Pull-Up Control Register */
typedef struct {
  __REG8         : 4;
  __REG8  PD4UP  : 1;
  __REG8  PD5UP  : 1;
  __REG8  PD6UP  : 1;
  __REG8         : 1;
} __pdpup_bits;

/*Port D pull-down control register*/
typedef struct {
  __REG8         : 4;
  __REG8  PD4DN  : 1;
  __REG8  PD5DN  : 1;
  __REG8  PD6DN  : 1;
  __REG8         : 1;
} __pdpdn_bits;

/*PORT D Input Enable Control Register */
typedef struct {
  __REG8         : 4;
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
  __REG8         : 2;
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

/*PORT I Register*/
typedef struct {
  __REG8       : 3;
  __REG8  PI3  : 1;
  __REG8       : 4;
} __pi_bits;

/*PORT I Control Register 1*/
typedef struct {
  __REG8        : 3;
  __REG8  PI3C  : 1;
  __REG8        : 4;
} __picr_bits;

/*Port I open drain control register*/
typedef struct {
  __REG8         : 3;
  __REG8  PI3OD  : 1;
  __REG8         : 4;
} __piod_bits;

/*PORT I Pull-Up Control Register */
typedef struct {
  __REG8         : 3;
  __REG8  PI3UP  : 1;
  __REG8         : 4;
} __pipup_bits;

/*Port I pull-down control register */
typedef struct {
  __REG8         : 3;
  __REG8  PI3DN  : 1;
  __REG8         : 4;
} __pipdn_bits;

/*PORT I Input Enable Control Register */
typedef struct {
  __REG8         : 3;
  __REG8  PI3IE  : 1;
  __REG8         : 4;
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

/*Port K open drain control register*/
typedef struct {
  __REG8  PK0OD  : 1;
  __REG8  PK1OD  : 1;
  __REG8         : 6;
} __pkod_bits;

/*PORT K Pull-Up Control Register*/
typedef struct {
  __REG8  PK0UP  : 1;
  __REG8  PK1UP  : 1;
  __REG8         : 6;
} __pkpup_bits;

/*Port K pull-down control register*/
typedef struct {
  __REG8  PK0DN  : 1;
  __REG8  PK1DN  : 1;
  __REG8         : 6;
} __pkpdn_bits;

/*PORT K Input Enable Control Register*/
typedef struct {
  __REG8  PK0IE  : 1;
  __REG8  PK1IE  : 1;
  __REG8         : 6;
} __pkie_bits;

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
  __REG32           : 1;
  __REG32  I2TB     : 1;
  __REG32           : 3;
  __REG32  TBWBF    : 1;
  __REG32           :24;
} __tbxcr_bits;

/*TMRB mode register (channels 0 thorough 9)*/
typedef struct {
  __REG32  TBCLK    : 2;
  __REG32  TBCLE    : 1;
  __REG32  TBCPM    : 2;
  __REG32  TBCP     : 1;
  __REG32  TBRSWR   : 1;
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
  __REG32  UD       : 1;
  __REG32  REVERR   : 1;
  __REG32  CMP      : 1;
  __REG32  P3EN     : 1;
  __REG32  MODE     : 2;
  __REG32           :13;
} __enxtncr_bits;

/*Encoder Counter Compare Register*/
typedef struct {
  __REG32  RELOAD   :16;
  __REG32           :16;
} __enxreload_bits;

/*Encoder Compare Register*/
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
  __REG32  UVWIS0   : 2;
  __REG32  ENSP0    : 1;
  __REG32  AINSP1   : 5;
  __REG32  UVWIS1   : 2;
  __REG32  ENSP1    : 1;
  __REG32  AINSP2   : 5;
  __REG32  UVWIS2   : 2;
  __REG32  ENSP2    : 1;
  __REG32  AINSP3   : 5;
  __REG32  UVWIS3   : 2;
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

/*Watchdog Timer Mode Register*/
typedef struct {
  __REG8          : 1;
  __REG8  RESCR   : 1;
  __REG8  I2WDT   : 1;
  __REG8          : 1;
  __REG8  WDTP    : 3;
  __REG8  WDTE    : 1;
} __wdmod_bits;

/*System Control Register*/
typedef struct {
  __REG32 GEAR    : 3;
  __REG32         : 5;
  __REG32 PRCK    : 3;
  __REG32         : 1;
  __REG32 FPSEL   : 1;
  __REG32         :19; 
} __syscr_bits;

/*Oscillation Control Register*/
typedef struct {
  __REG32 WUEON     : 1;
  __REG32 WUEF      : 1;
  __REG32 PLLON     : 1;
  __REG32 WUPSEL1   : 1;
  __REG32           : 4;
  __REG32 XEN1      : 1;
  __REG32           : 7;
  __REG32 XEN2      : 1;
  __REG32 OSCSEL    : 1;
  __REG32 HOSCON    : 1;
  __REG32 WUPSEL2   : 1;
  __REG32 WUODR     :12; 
} __osccr_bits;

/*Standby Control Register*/
typedef struct {
  __REG32 STBY    : 3;
  __REG32         : 5;
  __REG32 RXEN    : 1;
  __REG32         : 7;
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
  __REG32           : 1;
  __REG32 NMIFLG2   : 1;
  __REG32           :29;
} __nmiflg_bits;

/*Reset Flag Register*/
typedef struct {
  __REG32 PONRSTF   : 1;
  __REG32 PINRSTF   : 1;
  __REG32 WDTRSTF   : 1;
  __REG32 VLTDRSTF  : 1;
  __REG32 SYSRSTF   : 1;
  __REG32 OFDRSTF   : 1;
  __REG32           :26;
} __rstflg_bits;

/*CG Interrupt Mode Control Register A*/
typedef struct {
  __REG32           :24;
  __REG32 INT3EN    : 1;
  __REG32 INT3F     : 1;
  __REG32 EMST3     : 2;
  __REG32 EMCG3     : 3;
  __REG32           : 1;
} __imcga_bits;

/*CG Interrupt Mode Control Register B*/
typedef struct {
  __REG32 INT4EN    : 1;
  __REG32 INT4F     : 1;
  __REG32 EMST4     : 2;
  __REG32 EMCG4     : 3;
  __REG32           : 1;
  __REG32 INT5EN    : 1;
  __REG32 INT5F     : 1;
  __REG32 EMST5     : 2;
  __REG32 EMCG5     : 3;
  __REG32           : 1;
  __REG32 INT6EN    : 1;
  __REG32 INT6F     : 1;
  __REG32 EMST6     : 2;
  __REG32 EMCG6     : 3;
  __REG32           : 1;
  __REG32 INT7EN    : 1;
  __REG32 INT7F     : 1;
  __REG32 EMST7     : 2;
  __REG32 EMCG7     : 3;
  __REG32           : 1;
} __imcgb_bits;

/*CG Interrupt Mode Control Register C*/
typedef struct {
  __REG32 INT8EN    : 1;
  __REG32 INT8F     : 1;
  __REG32 EMST8     : 2;
  __REG32 EMCG8     : 3;
  __REG32           :25;
} __imcgc_bits;

/*CG Interrupt Mode Control Register D*/
typedef struct {
  __REG32 INTCEN    : 1;
  __REG32 INTCF     : 1;
  __REG32 EMSTC     : 2;
  __REG32 EMCGC     : 3;
  __REG32           : 1;
  __REG32 INTDEN    : 1;
  __REG32 INTDF     : 1;
  __REG32 EMSTD     : 2;
  __REG32 EMCGD     : 3;
  __REG32           : 1;
  __REG32 INTEEN    : 1;
  __REG32 INTEF     : 1;
  __REG32 EMSTE     : 2;
  __REG32 EMCGE     : 3;
  __REG32           : 1;
  __REG32 INTFEN    : 1;
  __REG32 INTFF     : 1;
  __REG32 EMSTF     : 2;
  __REG32 EMCGF     : 3;
  __REG32           : 1;
} __imcgd_bits;

/*Oscillation frequency detection control register 1*/
typedef struct {
  __REG32 CLKWEN    : 8;
  __REG32           :24;
} __ofdcr1_bits;

/*Oscillation frequency detection control register 2*/
typedef struct {
  __REG32 CLKSEN    : 8;
  __REG32           :24;
} __ofdcr2_bits;

/*Lower detection frequency setting register (In case of PLL OFF)*/
typedef struct {
  __REG32 OFDMNPLLOFF  : 9;
  __REG32              :23;
} __ofdmnplloff_bits;

/*Lower detection frequency setting register (In case of PLL ON)*/
typedef struct {
  __REG32 OFDMNPLLON   : 9;
  __REG32              :23;
} __ofdmnpllon_bits;

/*Higher detection frequency setting register (In case of PLL OFF)*/
typedef struct {
  __REG32 OFDMXPLLOFF  : 9;
  __REG32              :23;
} __ofdmxplloff_bits;

/*Higher detection frequency setting register (In case of PLL ON)*/
typedef struct {
  __REG32 OFDMXPLLON   : 9;
  __REG32              :23;
} __ofdmxpllon_bits;

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

/*VE Control Registers*/
typedef struct {
  __REG32 VEEN         : 1;
  __REG32 VEIDLEN      : 1;
  __REG32              :30;
} __veen_bits;

/*VE Control Registers*/
typedef struct {
  __REG32              : 1;
  __REG32 VCPURTB      : 1;
  __REG32              :30;
} __cpuruntrg_bits;

/*TASKAPP Register*/
typedef struct {
  __REG32              : 4;
  __REG32 VTASKB       : 4;
  __REG32              :24;
} __taskapp_bits;

/*ACTSCH Register*/
typedef struct {
  __REG32              : 4;
  __REG32 VACTB        : 4;
  __REG32              :24;
} __actsch_bits;

/*REPTIME Register*/
typedef struct {
  __REG32              : 4;
  __REG32 VREPB        : 4;
  __REG32              :24;
} __reptime_bits;

/*TRGMODE Register*/
typedef struct {
  __REG32              : 2;
  __REG32 VTRGB        : 2;
  __REG32              :28;
} __trgmode_bits;

/*ERRINTEN Register*/
typedef struct {
  __REG32              : 1;
  __REG32 VERRENB      : 1;
  __REG32              :30;
} __errinten_bits;

/*COMPEND Register*/
typedef struct {
  __REG32              : 1;
  __REG32 VCENDB       : 1;
  __REG32              :30;
} __compend_bits;

/*ERRDET Register*/
typedef struct {
  __REG32              : 1;
  __REG32 VERRDB       : 1;
  __REG32              :30;
} __errdet_bits;

/*SCHTASKRUN Register*/
typedef struct {
  __REG32              : 5;
  __REG32 VRSCHB       : 1;
  __REG32 VRTASKB      : 4;
  __REG32              :22;
} __schtaskrun_bits;

/*MCTLF Register*/
typedef struct {
  __REG32 LAVF         : 1;
  __REG32 LAVFM        : 1;
  __REG32 LVTF         : 1;
  __REG32              : 1;
  __REG32 PLSLF        : 1;
  __REG32 PLSLFM       : 1;
  __REG32              :26;
} __mctlfx_bits;

/*MODE Register*/
typedef struct {
  __REG32 PVIEN        : 1;
  __REG32 ZIEN         : 1;
  __REG32 OCRMD        : 2;
  __REG32              :28;
} __modex_bits;

/*FMODE Register*/
typedef struct {
  __REG32 C2PEN        : 1;
  __REG32 SPWMEN       : 1;
  __REG32 IDMODE       : 2;
  __REG32              : 5;
  __REG32 MREGDIS      : 1;
  __REG32              :22;
} __fmodex_bits;

/*TPWM Register*/
typedef struct {
  __REG32 TPWM         :16;
  __REG32              :16;
} __tpwmx_bits;

/*OMEGA Register*/
typedef struct {
  __REG32 OMEGA        :16;
  __REG32              :16;
} __omegax_bits;

/*THETA Register*/
typedef struct {
  __REG32 THETA        :16;
  __REG32              :16;
} __thetax_bits;

/*IDREF Register*/
typedef struct {
  __REG32 ID_REF       :16;
  __REG32              :16;
} __idrefx_bits;

/*IQREF Register*/
typedef struct {
  __REG32 IQ_REF       :16;
  __REG32              :16;
} __iqrefx_bits;

/*VD Register*/
typedef struct {
  __REG32 VD           :32;
} __vdx_bits;

/*VQ Register*/
typedef struct {
  __REG32 VQ           :32;
} __vqx_bits;

/*CIDKI Register*/
typedef struct {
  __REG32 CIDKI        :16;
  __REG32              :16;
} __cidkix_bits;

/*CIDKP Register*/
typedef struct {
  __REG32 CIDKP        :16;
  __REG32              :16;
} __cidkpx_bits;

/*CIQKI Register*/
typedef struct {
  __REG32 CIQKI        :16;
  __REG32              :16;
} __ciqkix_bits;

/*CIQKP Register*/
typedef struct {
  __REG32 CIQKP        :16;
  __REG32              :16;
} __ciqkpx_bits;

/*VDIH Register*/
typedef struct {
  __REG32 VDIH         :32;
} __vdihx_bits;

/*VDILH Register*/
typedef struct {
  __REG32              :16;
  __REG32 VDILH        :16;
} __vdilhx_bits;

/*VQIH Register*/
typedef struct {
  __REG32 VQIH         :32;
} __vqihx_bits;

/*VQILH Register*/
typedef struct {
  __REG32              :16;
  __REG32 VQILH        :16;
} __vqilhx_bits;

/*FPWMCHG Register*/
typedef struct {
  __REG32 FPWMCHG      :16;
  __REG32              :16;
} __fpwmchgx_bits;

/*PWM Period Register*/
typedef struct {
  __REG32 VMDPRD       :16;
  __REG32              :16;
} __vmdprdx_bits;

/*MINPLS Register*/
typedef struct {
  __REG32 MINPLS       :16;
  __REG32              :16;
} __minplsx_bits;

/*TRGCRC Register*/
typedef struct {
  __REG32 TRGCRC       :16;
  __REG32              :16;
} __trgcrcx_bits;

/*COS Registers*/
typedef struct {
  __REG32 COS          :16;
  __REG32              :16;
} __cosx_bits;

/*SIN Registers*/
typedef struct {
  __REG32 SIN          :16;
  __REG32              :16;
} __sinx_bits;

/*COSM Registers*/
typedef struct {
  __REG32 COSM         :16;
  __REG32              :16;
} __cosmx_bits;

/*SINM Registers*/
typedef struct {
  __REG32 SINM         :16;
  __REG32              :16;
} __sinmx_bits;

/*Sector Register*/
typedef struct {
  __REG32 SECTOR       : 4;
  __REG32              :28;
} __sectorx_bits;

/*Sector Register*/
typedef struct {
  __REG32 SECTORM      : 4;
  __REG32              :28;
} __sectormx_bits;

/*Zero-Current Registers*/
typedef struct {
  __REG32 IA0          :16;
  __REG32              :16;
} __iaox_bits;

/*Zero-Current Registers*/
typedef struct {
  __REG32 IB0          :16;
  __REG32              :16;
} __ibox_bits;

/*Zero-Current Registers*/
typedef struct {
  __REG32 IC0          :16;
  __REG32              :16;
} __icox_bits;

/*Current ADC Result Registers*/
typedef struct {
  __REG32 IAADC        :16;
  __REG32              :16;
} __iaadcx_bits;

/*Current ADC Result Registers*/
typedef struct {
  __REG32 IBADC        :16;
  __REG32              :16;
} __ibadcx_bits;

/*Current ADC Result Registers*/
typedef struct {
  __REG32 ICADC        :16;
  __REG32              :16;
} __icadcx_bits;

/*Supply Voltage Register*/
typedef struct {
  __REG32 VDC          :16;
  __REG32              :16;
} __vdcx_bits;

/*dq Current Registers*/
typedef struct {
  __REG32 ID           :32;
} __idx_bits;

/*dq Current Registers*/
typedef struct {
  __REG32 IQ           :32;
} __iqx_bits;     

/*TADC Register*/
typedef struct {
  __REG32 TADC         :16;
  __REG32              :16;
} __tadc_bits;

/*PWM Duty Register U Phase*/
typedef struct {
  __REG32 VCMPU        :16;
  __REG32              :16;
} __vcmpux_bits;

/*PWM Duty Register V Phase*/
typedef struct {
  __REG32 VCMPV        :16;
  __REG32              :16;
} __vcmpvx_bits;

/*PWM Duty Register W Phase*/
typedef struct {
  __REG32 VCMPW        :16;
  __REG32              :16;
} __vcmpwx_bits;

/*6-Phase Output Control Register*/
typedef struct {
  __REG32 UOC          : 2;
  __REG32 VOC          : 2;
  __REG32 WOC          : 2;
  __REG32 UPWM         : 1;
  __REG32 VPWM         : 1;
  __REG32 WPWM         : 1;
  __REG32              :23;
} __outcrx_bits;

/*VTRGCMP Register 0*/
typedef struct {
  __REG32 VTRGCMP0     :16;
  __REG32              :16;
} __vtrgcmp0x_bits;

/*VTRGCMP Register 1*/
typedef struct {
  __REG32 VTRGCMP1     :16;
  __REG32              :16;
} __vtrgcmp1x_bits;

/*VTRGSEL Register*/
typedef struct {
  __REG32 VTRGSEL      : 3;
  __REG32              :29;
} __vtrgselx_bits;

/*EMGRS Register*/
typedef struct {
  __REG32 EMGRS        : 1;
  __REG32              :31;
} __emgrsx_bits;

/*PMD Enable Register (MDEN)*/
typedef struct {
  __REG32 PWMEN        : 1;
  __REG32              :31;
} __mdenx_bits;

/*Port Output Mode Register (PORTMD)*/
typedef struct {
  __REG32 PORTMD       : 2;
  __REG32              :30;
} __portmdx_bits;

/*PMD Control Register (MDCR)*/
typedef struct {
  __REG32 PWMMD        : 1;
  __REG32 INTPRD       : 2;
  __REG32 PINT         : 1;
  __REG32 DTYMD        : 1;
  __REG32 SYNTMD       : 1;
  __REG32 PWMCK        : 1;
  __REG32              :25;
} __mdcrx_bits;

/*PWM Counter Status Register (CNTSTA)*/
typedef struct {
  __REG32 UPDWN        : 1;
  __REG32              :31;
} __cntstax_bits;

/*PWM Counter Register (MDCNT)*/
typedef struct {
  __REG32 MDCNT        :16;
  __REG32              :16;
} __mdcntx_bits;

/*PWM Period Register (MDPRD)*/
typedef struct {
  __REG32 MDPRD        :16;
  __REG32              :16;
} __mdprdx_bits;

/*PWM Compare Register (CMPU)*/
typedef struct {
  __REG32 CMPU         :16;
  __REG32              :16;
} __cmpux_bits;

/*PWM Compare Register (CMPV)*/
typedef struct {
  __REG32 CMPV         :16;
  __REG32              :16;
} __cmpvx_bits;

/*PWM Compare Register (CMPW)*/
typedef struct {
  __REG32 CMPW         :16;
  __REG32              :16;
} __cmpwx_bits;

/*Mode Select Register (MODESEL)*/
typedef struct {
  __REG32 MDSEL        : 1;
  __REG32              :31;
} __modeselx_bits;

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
} __mdoutx_bits;

/*PMD Output Setting Register (MDPOT)*/
typedef struct {
  __REG32 PSYNCS       : 2;
  __REG32 POLL         : 1;
  __REG32 POLH         : 1;
  __REG32              :28;
} __mdpotx_bits;

/*EMG Release Register (EMGREL)*/
typedef struct {
  __REG32 EMGREL       : 8;
  __REG32              :24;
} __emgrelx_bits;

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
} __emgcrx_bits;

/*EMG Status Register (EMGSTA)*/
typedef struct {
  __REG32 EMGST        : 1;
  __REG32 EMGI         : 1;
  __REG32              :30;
} __emgstax_bits;

/*OVV Control Register (OVVCR)*/
typedef struct {
  __REG32 OVVEN        : 1;
  __REG32 OVVRS        : 1;
  __REG32 OVVISEL      : 1;
  __REG32 OVVMD        : 2;
  __REG32 ADIN0EN      : 1;
  __REG32 ADIN1EN      : 1;
  __REG32              : 1;
  __REG32 OVVCNT       : 4;
  __REG32              :20;
} __ovvcrx_bits;

/*OVV Status Register (OVVSTA)*/
typedef struct {
  __REG32 OVVST        : 1;
  __REG32 OVVI         : 1;
  __REG32              :30;
} __ovvstax_bits;

/*Dead Time Register (DTR)*/
typedef struct {
  __REG32 DTR          : 8;
  __REG32              :24;
} __dtrx_bits;

/*Trigger Compare Register (TRGCMP0)*/
typedef struct {
  __REG32 TRGCMP0      :16;
  __REG32              :16;
} __trgcmp0x_bits;

/*Trigger Compare Register (TRGCMP1)*/
typedef struct {
  __REG32 TRGCMP1      :16;
  __REG32              :16;
} __trgcmp1x_bits;

/*Trigger Compare Register (TRGCMP2)*/
typedef struct {
  __REG32 TRGCMP2      :16;
  __REG32              :16;
} __trgcmp2x_bits;

/*Trigger Compare Register (TRGCMP3)*/
typedef struct {
  __REG32 TRGCMP3      :16;
  __REG32              :16;
} __trgcmp3x_bits;

/*Trigger Control Register (TRGCR)*/
typedef struct {
  __REG32 TRG0MD       : 3;
  __REG32 TRG0BE       : 1;
  __REG32 TRG1MD       : 3;
  __REG32 TRG1BE       : 1;
  __REG32 TRG2MD       : 3;
  __REG32 TRG2BE       : 1;
  __REG32 TRG3MD       : 3;
  __REG32 TRG3BE       : 1;
  __REG32              :16;
} __trgcrx_bits;

/*Trigger Output Mode Setting Register (TRGMD)*/
typedef struct {
  __REG32 EMGTGE       : 1;
  __REG32 TRGOUT       : 1;
  __REG32              :30;
} __trgmdx_bits;

/*Trigger Output Select Register (TRGSEL)*/
typedef struct {
  __REG32 TRGSEL       : 3;
  __REG32              :29;
} __trgselx_bits;

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

/* Active Bit Register 64-95 */
typedef struct {
  __REG32  ACTIVE64       : 1;
  __REG32  ACTIVE65       : 1;
  __REG32  ACTIVE66       : 1;
  __REG32  ACTIVE67       : 1;
  __REG32  ACTIVE68       : 1;
  __REG32  ACTIVE69       : 1;
  __REG32  ACTIVE70       : 1;
  __REG32  ACTIVE71       : 1;
  __REG32  ACTIVE72       : 1;
  __REG32  ACTIVE73       : 1;
  __REG32  ACTIVE74       : 1;
  __REG32  ACTIVE75       : 1;
  __REG32  ACTIVE76       : 1;
  __REG32  ACTIVE77       : 1;
  __REG32  ACTIVE78       : 1;
  __REG32  ACTIVE79       : 1;
  __REG32  ACTIVE80       : 1;
  __REG32  ACTIVE81       : 1;
  __REG32  ACTIVE82       : 1;
  __REG32  ACTIVE83       : 1;
  __REG32  ACTIVE84       : 1;
  __REG32  ACTIVE85       : 1;
  __REG32  ACTIVE86       : 1;
  __REG32  ACTIVE87       : 1;
  __REG32  ACTIVE88       : 1;
  __REG32  ACTIVE89       : 1;
  __REG32  ACTIVE90       : 1;
  __REG32  ACTIVE91       : 1;
  __REG32  ACTIVE92       : 1;
  __REG32  ACTIVE93       : 1;
  __REG32  ACTIVE94       : 1;
  __REG32  ACTIVE95       : 1;
} __active2_bits;

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
__IO_REG32_BIT(NVIC,              0xE000E004,__READ       ,__nvic_bits);
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
__IO_REG32_BIT(ACTIVE0,           0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,           0xE000E304,__READ       ,__active1_bits);
__IO_REG32_BIT(ACTIVE2,           0xE000E308,__READ       ,__active2_bits);
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
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,             0xE000ED0C,__READ_WRITE ,__aircr_bits);
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
 ** PORTD
 **
 ***************************************************************************/
__IO_REG8_BIT(PDDATA,               0x400000C0,__READ_WRITE ,__pd_bits);
__IO_REG8_BIT(PDCR,                 0x400000C4,__READ_WRITE ,__pdcr_bits);
__IO_REG8_BIT(PDFR1,                0x400000C8,__READ_WRITE ,__pdfr1_bits);
__IO_REG8_BIT(PDFR2,                0x400000CC,__READ_WRITE ,__pdfr2_bits);
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
__IO_REG8_BIT(PGOD,                 0x400001A8,__READ_WRITE ,__pgod_bits);
__IO_REG8_BIT(PGPUP,                0x400001AC,__READ_WRITE ,__pgpup_bits);
__IO_REG8_BIT(PGPDN,                0x400001B0,__READ_WRITE ,__pgpdn_bits);
__IO_REG8_BIT(PGIE,                 0x400001B8,__READ_WRITE ,__pgie_bits);

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
 ** PORTK
 **
 ***************************************************************************/
__IO_REG8_BIT(PKDATA,               0x40000280,__READ_WRITE ,__pk_bits);
__IO_REG8_BIT(PKCR,                 0x40000284,__READ_WRITE ,__pkcr_bits);
__IO_REG8_BIT(PKFR1,                0x40000288,__READ_WRITE ,__pkfr1_bits);
__IO_REG8_BIT(PKOD,                 0x400002A8,__READ_WRITE ,__pkod_bits);
__IO_REG8_BIT(PKPUP,                0x400002AC,__READ_WRITE ,__pkpup_bits);
__IO_REG8_BIT(PKPDN,                0x400002B0,__READ_WRITE ,__pkpdn_bits);
__IO_REG8_BIT(PKIE,                 0x400002B8,__READ_WRITE ,__pkie_bits);

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
 ** ADC B
 **
 ***************************************************************************/
__IO_REG8_BIT(ADBCLK,               0x40030200, __READ_WRITE , __adxclk_bits);
__IO_REG8_BIT(ADBMOD0,              0x40030204, __READ_WRITE , __adxmod0_bits);
__IO_REG8_BIT(ADBMOD1,              0x40030208, __READ_WRITE , __adxmod1_bits);
__IO_REG8_BIT(ADBMOD2,              0x4003020C, __READ       , __adxmod2_bits);
__IO_REG16_BIT(ADBCMPCR0,           0x40030210, __READ_WRITE , __adxcmpcrx_bits);
__IO_REG16_BIT(ADBCMPCR1,           0x40030214, __READ_WRITE , __adxcmpcrx_bits);
__IO_REG16_BIT(ADBCMP0,             0x40030218, __READ_WRITE , __adxcmpx_bits);
__IO_REG16_BIT(ADBCMP1,             0x4003021C, __READ_WRITE , __adxcmpx_bits);
__IO_REG16_BIT(ADBREG0,             0x40030220, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG1,             0x40030224, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG2,             0x40030228, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG3,             0x4003022C, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG4,             0x40030230, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG5,             0x40030234, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG6,             0x40030238, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG7,             0x4003023C, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG8,             0x40030240, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG9,             0x40030244, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG10,            0x40030248, __READ       , __adxregx_bits);
__IO_REG16_BIT(ADBREG11,            0x4003024C, __READ       , __adxregx_bits);
/* __IO_REG8_BIT(ADBPSEL0,             0x40030250, __READ_WRITE , __adxpselx_bits); */
/* __IO_REG8_BIT(ADBPSEL1,             0x40030254, __READ_WRITE , __adxpselx_bits); */
/* __IO_REG8_BIT(ADBPSEL2,             0x40030258, __READ_WRITE , __adxpselx_bits); */
/* __IO_REG8_BIT(ADBPSEL3,             0x4003025C, __READ_WRITE , __adxpselx_bits); */
/* __IO_REG8_BIT(ADBPSEL4,             0x40030260, __READ_WRITE , __adxpselx_bits); */
/* __IO_REG8_BIT(ADBPSEL5,             0x40030264, __READ_WRITE , __adxpselx_bits); */
__IO_REG8_BIT(ADBPSEL6,             0x40030268, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADBPSEL7,             0x4003026C, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADBPSEL8,             0x40030270, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADBPSEL9,             0x40030274, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADBPSEL10,            0x40030278, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADBPSEL11,            0x4003027C, __READ_WRITE , __adxpselx_bits);
__IO_REG8_BIT(ADBPINTS0,            0x40030280, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADBPINTS1,            0x40030284, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADBPINTS2,            0x40030288, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADBPINTS3,            0x4003028C, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADBPINTS4,            0x40030290, __READ_WRITE , __adxpintsx_bits);
__IO_REG8_BIT(ADBPINTS5,            0x40030294, __READ_WRITE , __adxpintsx_bits);
__IO_REG32_BIT(ADBPSET0,            0x40030298, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADBPSET1,            0x4003029C, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADBPSET2,            0x400302A0, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADBPSET3,            0x400302A4, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADBPSET4,            0x400302A8, __READ_WRITE , __adxpsetx_bits);
__IO_REG32_BIT(ADBPSET5,            0x400302AC, __READ_WRITE , __adxpsetx_bits);
__IO_REG8_BIT(ADBTSET0,             0x400302B0, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET1,             0x400302B1, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET2,             0x400302B2, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET3,             0x400302B3, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET4,             0x400302B4, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET5,             0x400302B5, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET6,             0x400302B6, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET7,             0x400302B7, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET8,             0x400302B8, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET9,             0x400302B9, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET10,            0x400302BA, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBTSET11,            0x400302BB, __READ_WRITE , __adxtsetx_bits);
__IO_REG8_BIT(ADBSSET0,             0x400302BC, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET1,             0x400302BD, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET2,             0x400302BE, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET3,             0x400302BF, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET4,             0x400302C0, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET5,             0x400302C1, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET6,             0x400302C2, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET7,             0x400302C3, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET8,             0x400302C4, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET9,             0x400302C5, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET10,            0x400302C6, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBSSET11,            0x400302C7, __READ_WRITE , __adxssetx_bits);
__IO_REG8_BIT(ADBASET0,             0x400302C8, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET1,             0x400302C9, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET2,             0x400302CA, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET3,             0x400302CB, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET4,             0x400302CC, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET5,             0x400302CD, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET6,             0x400302CE, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET7,             0x400302CF, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET8,             0x400302D0, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET9,             0x400302D1, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET10,            0x400302D2, __READ_WRITE , __adxasetx_bits);
__IO_REG8_BIT(ADBASET11,            0x400302D3, __READ_WRITE , __adxasetx_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG8_BIT(WDMOD,                0x40040000,__READ_WRITE ,__wdmod_bits);
__IO_REG8(    WDCR,                 0x40040004,__WRITE);

/***************************************************************************
 **
 ** CG
 **
 ***************************************************************************/
 __IO_REG32_BIT(CGSYSCR,            0x40040200, __READ_WRITE ,__syscr_bits);
 __IO_REG32_BIT(CGOSCCR,            0x40040204, __READ_WRITE ,__osccr_bits);
 __IO_REG32_BIT(CGSTBYCR,           0x40040208, __READ_WRITE ,__stbycr_bits);
 __IO_REG32_BIT(CGPLLSEL,           0x4004020C, __READ_WRITE ,__pllsel_bits);
 __IO_REG32_BIT(CGCKSEL,            0x40040210, __READ_WRITE ,__cksel_bits);
 __IO_REG32_BIT(CGICRCG,            0x40040214, __WRITE      ,__icrcg_bits);
 __IO_REG32_BIT(CGNMIFLG,           0x40040218, __READ       ,__nmiflg_bits);
 __IO_REG32_BIT(CGRSTFLG,           0x4004021C, __READ_WRITE ,__rstflg_bits);
 __IO_REG32_BIT(CGIMCGA,            0x40040220, __READ_WRITE ,__imcga_bits);
 __IO_REG32_BIT(CGIMCGB,            0x40040224, __READ_WRITE ,__imcgb_bits);
 __IO_REG32_BIT(CGIMCGC,            0x40040228, __READ_WRITE ,__imcgc_bits);
 __IO_REG32_BIT(CGIMCGD,            0x4004022C, __READ_WRITE ,__imcgd_bits);
 
/***************************************************************************
 **
 ** OFD
 **
 ***************************************************************************/
__IO_REG32_BIT(OFDCR1,              0x40040800, __READ_WRITE ,__ofdcr1_bits);
__IO_REG32_BIT(OFDCR2,              0x40040804, __READ_WRITE ,__ofdcr2_bits);
__IO_REG32_BIT(OFDMNPLLOFF,         0x40040808, __READ_WRITE ,__ofdmnplloff_bits);
__IO_REG32_BIT(OFDMNPLLON,          0x4004080C, __READ_WRITE ,__ofdmnpllon_bits);
__IO_REG32_BIT(OFDMXPLLOFF,         0x40040810, __READ_WRITE ,__ofdmxplloff_bits);
__IO_REG32_BIT(OFDMXPLLON,          0x40040814, __READ_WRITE ,__ofdmxpllon_bits);

/***************************************************************************
 **
 ** VLTD
 **
 ***************************************************************************/
__IO_REG32_BIT(VDCR,                0x40040900, __READ_WRITE , __vdcr_bits);
/* __IO_REG32_BIT(VDSR,                0x40040904, __READ       , __vdsr_bits); */

/***************************************************************************
 **
 ** VE
 **
 ***************************************************************************/
__IO_REG32_BIT(VEEN,                0x40050000, __READ_WRITE , __veen_bits);
__IO_REG32_BIT(VECPURUNTRG,         0x40050004, __WRITE      , __cpuruntrg_bits);
__IO_REG32_BIT(VETASKAPP,           0x40050008, __READ_WRITE , __taskapp_bits);
__IO_REG32_BIT(VEACTSCH,            0x4005000C, __READ_WRITE , __actsch_bits);
__IO_REG32_BIT(VEREPTIME,           0x40050010, __READ_WRITE , __reptime_bits);
__IO_REG32_BIT(VETRGMODE,           0x40050014, __READ_WRITE , __trgmode_bits);
__IO_REG32_BIT(VEERRINTEN,          0x40050018, __READ_WRITE , __errinten_bits);
__IO_REG32_BIT(VECOMPEND,           0x4005001C, __WRITE      , __compend_bits);
__IO_REG32_BIT(VEERRDET,            0x40050020, __READ       , __errdet_bits);
__IO_REG32_BIT(VESCHTASKRUN,        0x40050024, __READ       , __schtaskrun_bits);
__IO_REG32(    VETMPREG0,           0x4005002C, __READ_WRITE );
__IO_REG32(    VETMPREG1,           0x40050030, __READ_WRITE );
__IO_REG32(    VETMPREG2,           0x40050034, __READ_WRITE );
__IO_REG32(    VETMPREG3,           0x40050038, __READ_WRITE );
__IO_REG32(    VETMPREG4,           0x4005003C, __READ_WRITE );
__IO_REG32(    VETMPREG5,           0x40050040, __READ_WRITE );
__IO_REG32_BIT(VEMCTLF1,            0x400500DC, __READ_WRITE , __mctlfx_bits);
__IO_REG32_BIT(VEMODE1,             0x400500E0, __READ_WRITE , __modex_bits);
__IO_REG32_BIT(VEFMODE1,            0x400500E4, __READ_WRITE , __fmodex_bits);
__IO_REG32_BIT(VETPWM1,             0x400500E8, __READ_WRITE , __tpwmx_bits);
__IO_REG32_BIT(VEOMEGA1,            0x400500EC, __READ_WRITE , __omegax_bits);
__IO_REG32_BIT(VETHETA1,            0x400500F0, __READ_WRITE , __thetax_bits);
__IO_REG32_BIT(VEIDREF1,            0x400500F4, __READ_WRITE , __idrefx_bits);
__IO_REG32_BIT(VEIQREF1,            0x400500F8, __READ_WRITE , __iqrefx_bits);
__IO_REG32_BIT(VEVD1,               0x400500FC, __READ_WRITE , __vdx_bits);
__IO_REG32_BIT(VEVQ1,               0x40050100, __READ_WRITE , __vqx_bits);
__IO_REG32_BIT(VECIDKI1,            0x40050104, __READ_WRITE , __cidkix_bits);
__IO_REG32_BIT(VECIDKP1,            0x40050108, __READ_WRITE , __cidkpx_bits);
__IO_REG32_BIT(VECIQKI1,            0x4005010C, __READ_WRITE , __ciqkix_bits);
__IO_REG32_BIT(VECIQKP1,            0x40050110, __READ_WRITE , __ciqkpx_bits);
__IO_REG32_BIT(VEVDIH1,             0x40050114, __READ_WRITE , __vdihx_bits);
__IO_REG32_BIT(VEVDILH1,            0x40050118, __READ_WRITE , __vdilhx_bits);
__IO_REG32_BIT(VEVQIH1,             0x4005011C, __READ_WRITE , __vqihx_bits);
__IO_REG32_BIT(VEVQILH1,            0x40050120, __READ_WRITE , __vqilhx_bits);
__IO_REG32_BIT(VEFPWMCHG1,          0x40050124, __READ_WRITE , __fpwmchgx_bits);
__IO_REG32_BIT(VEMDPRD1,            0x40050128, __READ_WRITE , __vmdprdx_bits);
__IO_REG32_BIT(VEMINPLS1,           0x4005012C, __READ_WRITE , __minplsx_bits);
__IO_REG32_BIT(VETRGCRC1,           0x40050130, __READ_WRITE , __trgcrcx_bits);
__IO_REG32_BIT(VECOS1,              0x40050138, __READ_WRITE , __cosx_bits);
__IO_REG32_BIT(VESIN1,              0x4005013C, __READ_WRITE , __sinx_bits);
__IO_REG32_BIT(VECOSM1,             0x40050140, __READ_WRITE , __cosmx_bits);
__IO_REG32_BIT(VESINM1,             0x40050144, __READ_WRITE , __sinmx_bits);
__IO_REG32_BIT(VESECTOR1,           0x40050148, __READ_WRITE , __sectorx_bits);
__IO_REG32_BIT(VESECTORM1,          0x4005014C, __READ_WRITE , __sectormx_bits);
__IO_REG32_BIT(VEIAO1,              0x40050150, __READ_WRITE , __iaox_bits);
__IO_REG32_BIT(VEIBO1,              0x40050154, __READ_WRITE , __ibox_bits);
__IO_REG32_BIT(VEICO1,              0x40050158, __READ_WRITE , __icox_bits);
__IO_REG32_BIT(VEIAADC1,            0x4005015C, __READ_WRITE , __iaadcx_bits);
__IO_REG32_BIT(VEIBADC1,            0x40050160, __READ_WRITE , __ibadcx_bits);
__IO_REG32_BIT(VEICADC1,            0x40050164, __READ_WRITE , __icadcx_bits);
__IO_REG32_BIT(VEVDC1,              0x40050168, __READ_WRITE , __vdcx_bits);
__IO_REG32_BIT(VEID1,               0x4005016C, __READ_WRITE , __idx_bits);
__IO_REG32_BIT(VEIQ1,               0x40050170, __READ_WRITE , __iqx_bits);     
__IO_REG32_BIT(VETADC,              0x40050178, __READ_WRITE , __tadc_bits);
__IO_REG32_BIT(VECMPU1,             0x4005019C, __READ_WRITE , __vcmpux_bits);
__IO_REG32_BIT(VECMPV1,             0x400501A0, __READ_WRITE , __vcmpvx_bits);
__IO_REG32_BIT(VECMPW1,             0x400501A4, __READ_WRITE , __vcmpwx_bits);
__IO_REG32_BIT(VEOUTCR1,            0x400501A8, __READ_WRITE , __outcrx_bits);
__IO_REG32_BIT(VETRGCMP01,          0x400501AC, __READ_WRITE , __vtrgcmp0x_bits);
__IO_REG32_BIT(VETRGCMP11,          0x400501B0, __READ_WRITE , __vtrgcmp1x_bits);
__IO_REG32_BIT(VETRGSEL1,           0x400501B4, __READ_WRITE , __vtrgselx_bits);
__IO_REG32_BIT(VEEMGRS1,            0x400501B8, __WRITE      , __emgrsx_bits);   

/***************************************************************************
 **
 ** PMD 1
 **
 ***************************************************************************/
__IO_REG32_BIT(PMD1MDEN,            0x40050480, __READ_WRITE , __mdenx_bits);
__IO_REG32_BIT(PMD1PORTMD,          0x40050484, __READ_WRITE , __portmdx_bits);
__IO_REG32_BIT(PMD1MDCR,            0x40050488, __READ_WRITE , __mdcrx_bits);
__IO_REG32_BIT(PMD1CNTSTA,          0x4005048C, __READ       , __cntstax_bits);
__IO_REG32_BIT(PMD1MDCNT,           0x40050490, __READ       , __mdcntx_bits);
__IO_REG32_BIT(PMD1MDPRD,           0x40050494, __READ_WRITE , __mdprdx_bits);
__IO_REG32_BIT(PMD1CMPU,            0x40050498, __READ_WRITE , __cmpux_bits);
__IO_REG32_BIT(PMD1CMPV,            0x4005049C, __READ_WRITE , __cmpvx_bits);
__IO_REG32_BIT(PMD1CMPW,            0x400504A0, __READ_WRITE , __cmpwx_bits);
__IO_REG32_BIT(PMD1MODESEL,         0x400504A4, __READ_WRITE , __modeselx_bits);
__IO_REG32_BIT(PMD1MDOUT,           0x400504A8, __READ_WRITE , __mdoutx_bits);
__IO_REG32_BIT(PMD1MDPOT,           0x400504AC, __READ_WRITE , __mdpotx_bits);
__IO_REG32_BIT(PMD1EMGREL,          0x400504B0, __WRITE      , __emgrelx_bits);
__IO_REG32_BIT(PMD1EMGCR,           0x400504B4, __READ_WRITE , __emgcrx_bits);
__IO_REG32_BIT(PMD1EMGSTA,          0x400504B8, __READ       , __emgstax_bits);
__IO_REG32_BIT(PMD1OVVCR,           0x400504BC, __READ_WRITE , __ovvcrx_bits);
__IO_REG32_BIT(PMD1OVVSTA,          0x400504C0, __READ       , __ovvstax_bits);
__IO_REG32_BIT(PMD1DTR,             0x400504C4, __READ_WRITE , __dtrx_bits);
__IO_REG32_BIT(PMD1TRGCMP0,         0x400504C8, __READ_WRITE , __trgcmp0x_bits);
__IO_REG32_BIT(PMD1TRGCMP1,         0x400504CC, __READ_WRITE , __trgcmp1x_bits);
__IO_REG32_BIT(PMD1TRGCMP2,         0x400504D0, __READ_WRITE , __trgcmp2x_bits);
__IO_REG32_BIT(PMD1TRGCMP3,         0x400504D4, __READ_WRITE , __trgcmp3x_bits);
__IO_REG32_BIT(PMD1TRGCR,           0x400504D8, __READ_WRITE , __trgcrx_bits);
__IO_REG32_BIT(PMD1TRGMD,           0x400504DC, __READ_WRITE , __trgmdx_bits);
__IO_REG32_BIT(PMD1TRGSEL,          0x400504E0, __READ_WRITE , __trgselx_bits);

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
 **  TMPM372FWUG Interrupt Lines
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
#define INT_3                ( 3 + EII)   /* Interrupt Pin (PA0/TB0IN/2pin or 4pin)*/
#define INT_4                ( 4 + EII)   /* Interrupt Pin (PA2/TB1IN/4pin or 6pin)*/
#define INT_5                ( 5 + EII)   /* Interrupt Pin (PE4/TB2IN/15pin or 17pin)*/
#define INT_RX0              ( 6 + EII)   /* Serial reception (channel.0) */
#define INT_TX0              ( 7 + EII)   /* Serial transmit (channel.0)  */
#define INT_RX1              ( 8 + EII)   /* Serial reception (channel.1) */
#define INT_TX1              ( 9 + EII)   /* Serial transmit (channel.1)  */
#define INT_VCNB             (11 + EII)   /* Vector Engine interrupt B    */
#define INT_EMG1             (13 + EII)   /* PMD1 EMG interrupt           */
#define INT_OVV1             (15 + EII)   /* PMD1 OVV interrupt*/
#define INT_ADBPDB           (19 + EII)   /* ADCB conversion triggered by PMD1 is finished*/
#define INT_TB00             (20 + EII)   /* 16bit TMRB0 compare match detection 0/ Over flow*/
#define INT_TB01             (21 + EII)   /* 16bit TMRB0 compare match detection 1           */
#define INT_TB10             (22 + EII)   /* 16bit TMRB1 compare match detection 0/ Over flow*/
#define INT_TB11             (23 + EII)   /* 16bit TMRB1 compare match detection 1           */
#define INT_TB40             (24 + EII)   /* 16bit TMRB4 compare match detection 0/ Over flow*/
#define INT_TB41             (25 + EII)   /* 16bit TMRB4 compare match detection 1           */
#define INT_TB50             (26 + EII)   /* 16bit TMRB5 compare match detection 0/ Over flow*/
#define INT_TB51             (27 + EII)   /* 16bit TMRB5 compare match detection 1           */
#define INT_PMD1             (29 + EII)   /* PMD1 PWM interrupt           */
#define INT_CAP00            (30 + EII)   /* 16bit TMRB0 input capture 0  */
#define INT_CAP01            (31 + EII)   /* 16bit TMRB0 input capture 1  */
#define INT_CAP10            (32 + EII)   /* 16bit TMRB1 input capture 0  */
#define INT_CAP11            (33 + EII)   /* 16bit TMRB1 input capture 1  */
#define INT_CAP40            (34 + EII)   /* 16bit TMRB4 input capture 0  */
#define INT_CAP41            (35 + EII)   /* 16bit TMRB4 input capture 1  */
#define INT_CAP50            (36 + EII)   /* 16bit TMRB5 input capture 0  */
#define INT_6                (38 + EII)   /* Interrupt Pin (PE6/TB3IN/17pin or 19pin)*/
#define INT_7                (39 + EII)   /* Interrupt Pin (PE7/TB3OUT/18pin or 20pin)*/
#define INT_RX2              (40 + EII)   /* Serial reception (channel.2) */
#define INT_TX2              (41 + EII)   /* Serial transmit (channel.2)  */
#define INT_ADBCPA           (43 + EII)   /* ADB conversion monitoring function interrupt A */
#define INT_ADBCPB           (45 + EII)   /* ADB conversion monitoring function interrupt B */
#define INT_TB20             (46 + EII)   /* 16bit TMRB2 compare match detection 0/ Over flow*/
#define INT_TB21             (47 + EII)   /* 16bit TMRB2 compare match detection 1           */
#define INT_TB30             (48 + EII)   /* 16bit TMRB3 compare match detection 0/ Over flow*/
#define INT_TB31             (49 + EII)   /* 16bit TMRB3 compare match detection 1           */
#define INT_CAP20            (50 + EII)   /* 16bit TMRB2 input capture 0  */
#define INT_CAP21            (51 + EII)   /* 16bit TMRB2 input capture 1  */
#define INT_CAP30            (52 + EII)   /* 16bit TMRB3 input capture 0  */
#define INT_CAP31            (53 + EII)   /* 16bit TMRB3 input capture 1  */
#define INT_ADBSFT           (55 + EII)   /* ADCB conversion started by software is finished   */
#define INT_ADBTMR           (57 + EII)   /* ADCB conversion triggered by timer is finished    */
#define INT_8                (58 + EII)   /* Interrupt Pin (PA7/TB4IN/9pin or 11pin)           */
#define INT_ENC1             (63 + EII)   /* Ender input1 interrupt                            */
#define INT_RX3              (64 + EII)   /* Serial reception (channel.3)                      */
#define INT_TX3              (65 + EII)   /* Serial transmit (channel.3)                       */
#define INT_TB60             (66 + EII)   /* 16bit TMRB6 compare match detection 0 / Over flow */
#define INT_TB61             (67 + EII)   /* 16bit TMRB6 compare match detection 1             */
#define INT_TB70             (68 + EII)   /* 16bit TMRB7 compare match detection 0 / Over flow */
#define INT_TB71             (69 + EII)   /* 16bit TMRB7 compare match detection 1     */
#define INT_CAP60            (70 + EII)   /* 16bit TMRB6 input capture 0               */
#define INT_CAP61            (71 + EII)   /* 16bit TMRB6 input capture 1               */
#define INT_CAP70            (72 + EII)   /* 16bit TMRB7 input capture 0               */
#define INT_CAP71            (73 + EII)   /* 16bit TMRB7 input capture 1               */
#define INT_C                (74 + EII)   /* Interrupt Pin (PJ6/AINB9/74pin or 76 pin) */
#define INT_D                (75 + EII)   /* Interrupt Pin (PJ7/AINB10/73pin or 75pin) */
#define INT_E                (76 + EII)   /* Interrupt Pin (PK0/AINB11/72pin or 74pin) */
#define INT_F                (77 + EII)   /* Interrupt Pin (PK1/AINB12/71pin or 73pin) */

#endif    /* __IOTMPM372FWUG_H */

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
Interrupt9   = INT3           0x4C
Interrupt10  = INT4           0x50
Interrupt11  = INT5           0x54
Interrupt12  = INTRX0         0x58
Interrupt13  = INTTX0         0x5C
Interrupt14  = INTRX1         0x60
Interrupt15  = INTTX1         0x64
Interrupt16  = INTVCNB        0x6C
Interrupt17  = INTEMG1        0x74
Interrupt18  = INTOVV1        0x7C
Interrupt19  = INTADBPDB      0x8C
Interrupt20  = INTTB00        0x90
Interrupt21  = INTTB01        0x94
Interrupt22  = INTTB10        0x98
Interrupt23  = INTTB11        0x9C
Interrupt24  = INTTB40        0xA0
Interrupt25  = INTTB41        0xA4
Interrupt26  = INTTB50        0xA8
Interrupt27  = INTTB51        0xAC
Interrupt28  = INTPMD1        0xB4
Interrupt29  = INTCAP00       0xB8
Interrupt30  = INTCAP01       0xBC
Interrupt31  = INTCAP10       0xC0
Interrupt32  = INTCAP11       0xC4
Interrupt33  = INTCAP40       0xC8
Interrupt34  = INTCAP41       0xCC
Interrupt35  = INTCAP50       0xD0
Interrupt36  = INT6           0xD8
Interrupt37  = INT7           0xDC
Interrupt38  = INTRX2         0xE0
Interrupt39  = INTTX2         0xE4
Interrupt40  = INTADBCPA      0xEC
Interrupt41  = INTADBCPB      0xF4
Interrupt42  = INTTB20        0xF8
Interrupt43  = INTTB21        0xFC
Interrupt44  = INTTB30        0x100
Interrupt45  = INTTB31        0x104
Interrupt46  = INTCAP20       0x108 
Interrupt47  = INTCAP21       0x10C 
Interrupt48  = INTCAP30       0x110 
Interrupt49  = INTCAP31       0x114 
Interrupt50  = INTADBSFT      0x11C 
Interrupt51  = INTADBTMR      0x124 
Interrupt52  = INT8           0x128 
Interrupt53  = INTENC1        0x13C 
Interrupt54  = INTRX3         0x140 
Interrupt55  = INTTX3         0x144 
Interrupt56  = INTTB60        0x148 
Interrupt57  = INTTB61        0x14C 
Interrupt58  = INTTB70        0x150 
Interrupt59  = INTTB71        0x154 
Interrupt60  = INTCAP60       0x158
Interrupt61  = INTCAP61       0x15C
Interrupt62  = INTCAP70       0x160
Interrupt63  = INTCAP71       0x164
Interrupt64  = INTC           0x168
Interrupt65  = INTD           0x16C
Interrupt66  = INTE           0x170
Interrupt67  = INTF           0x174

###DDF-INTERRUPT-END###*/
