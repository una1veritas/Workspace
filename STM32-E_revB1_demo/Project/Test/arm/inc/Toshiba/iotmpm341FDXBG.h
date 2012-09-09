/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM341FDXBG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 50669 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM341FDXBG_H
#define __IOTMPM341FDXBG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM341FDXBG SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
#pragma system_include
#endif

#if 1 != __LITTLE_ENDIAN__
#pragma bitfields=disjoint_types
#endif

/* DMAC Interrupt Status Register */
typedef struct {
  __REG32 INTSTATUS0    : 1;
  __REG32 INTSTATUS1    : 1;
  __REG32               :30;
} __dmacxintstatus_bits;

/* DMAC Interrupt Terminal Count Status Register */
typedef struct {
  __REG32 INTTCSTATUS0  : 1;
  __REG32 INTTCSTATUS1  : 1;
  __REG32               :30;
} __dmacxinttcstatus_bits;

/* DMAC Interrupt Terminal Count Clear Register */
typedef struct {
  __REG32 INTTCCLEAR0   : 1;
  __REG32 INTTCCLEAR1   : 1;
  __REG32               :30;
} __dmacxinttcclear_bits;

/* DMAC Interrupt Error Status Register */
typedef struct {
  __REG32 INTERRSTATUS0 : 1;
  __REG32 INTERRSTATUS1 : 1;
  __REG32               :30;
} __dmacxinterrorstatus_bits;

/* DMAC Interrupt Error Clear Register */
typedef struct {
  __REG32 INTERRCLR0    : 1;
  __REG32 INTERRCLR1    : 1;
  __REG32               :30;
} __dmacxinterrclr_bits;

/* DMAC Raw Interrupt Terminal Count Status Register */
typedef struct {
  __REG32 RAWINTTCS0    : 1;
  __REG32 RAWINTTCS1    : 1;
  __REG32               :30;
} __dmacxrawinttcstatus_bits;

/* DMAC Raw Error Interrupt Status Register */
typedef struct {
  __REG32 RAWINTERRS0   : 1;
  __REG32 RAWINTERRS1   : 1;
  __REG32               :30;
} __dmacxrawinterrorstatus_bits;

/* DMAC Enabled Channel Register */
typedef struct {
  __REG32 ENABLEDCH0    : 1;
  __REG32 ENABLEDCH1    : 1;
  __REG32               :30;
} __dmacxenbldchns_bits;

/* DMAC Software Burst Request Register */
typedef struct {
  __REG32 SOFTBREQ0     : 1;
  __REG32 SOFTBREQ1     : 1;
  __REG32 SOFTBREQ2     : 1;
  __REG32 SOFTBREQ3     : 1;
  __REG32 SOFTBREQ4     : 1;
  __REG32 SOFTBREQ5     : 1;
  __REG32 SOFTBREQ6     : 1;
  __REG32 SOFTBREQ7     : 1;
  __REG32 SOFTBREQ8     : 1;
  __REG32 SOFTBREQ9     : 1;
  __REG32 SOFTBREQ10    : 1;
  __REG32 SOFTBREQ11    : 1;
  __REG32 SOFTBREQ12    : 1;
  __REG32 SOFTBREQ13    : 1;
  __REG32 SOFTBREQ14    : 1;
  __REG32 SOFTBREQ15    : 1;
  __REG32               :16;
} __dmacxsoftbreq_bits;

/* DMAC-B Software Single Request Register */
typedef struct {
  __REG32               :14;
  __REG32 SOFTSREQ14    : 1;
  __REG32 SOFTSREQ15    : 1;
  __REG32               :16;
} __dmacbsoftsreq_bits;

/* DMAC Configuration Register */
typedef struct {
  __REG32 E     : 1;
  __REG32 M     : 1;
  __REG32       :30;
} __dmacxconfiguration_bits;

/* DMAC Channel0 Linked List Item Register */
typedef struct {
  __REG32       : 2;
  __REG32 LLI   :30;
} __dmacxc0lli_bits;

/* DMAC Channel0 Control Register */
typedef struct {
  __REG32 TRANSFERSIZE  : 12;
  __REG32 SBSIZE        : 3;
  __REG32 DBSIZE        : 3;
  __REG32 SWIDTH        : 3;
  __REG32 DWIDTH        : 3;
  __REG32               : 2;
  __REG32 SI            : 1;
  __REG32 DI            : 1;
  __REG32               : 3;
  __REG32 I             : 1;
} __dmacxc0control_bits;

/* DMAC Channel0 Configuration Register */
typedef struct {
  __REG32 E                 : 1;
  __REG32 SRCPERIPHERAL     : 4;
  __REG32                   : 1;
  __REG32 DESTPERIPHERAL    : 4;
  __REG32                   : 1;
  __REG32 FLOWCNTRL         : 3;
  __REG32 IE                : 1;
  __REG32 ITC               : 1;
  __REG32 LOCK              : 1;
  __REG32 ACTIVE            : 1;
  __REG32 HALT              : 1;
  __REG32                   :13;
} __dmacxc0configuration_bits;

/*PORT A Register*/
typedef struct {
  __REG32 PA0  	: 1;
  __REG32 PA1  	: 1;
  __REG32 PA2  	: 1;
  __REG32 PA3  	: 1;
  __REG32 PA4  	: 1;
  __REG32 PA5  	: 1;
  __REG32 PA6  	: 1;
  __REG32 PA7  	: 1;
  __REG32 		 	:24;
} __pa_bits;

/*PORT A Control Register */
typedef struct {
  __REG32 PA0C  : 1;
  __REG32 PA1C  : 1;
  __REG32 PA2C  : 1;
  __REG32 PA3C  : 1;
  __REG32 PA4C  : 1;
  __REG32 PA5C  : 1;
  __REG32 PA6C  : 1;
  __REG32 PA7C  : 1;
  __REG32 		 	:24;
} __pacr_bits;

/*PORT A Function Register 1*/
typedef struct {
  __REG32 PA0F1 : 1;
  __REG32 PA1F1 : 1;
  __REG32 PA2F1 : 1;
  __REG32 PA3F1 : 1;
  __REG32 PA4F1 : 1;
  __REG32 PA5F1 : 1;
  __REG32 PA6F1 : 1;
  __REG32 PA7F1 : 1;
  __REG32 		 	:24;
} __pafr1_bits;

/*PORT A Open-Drain Control Register */
typedef struct {
  __REG32 PA0OD	: 1;
  __REG32 PA1OD	: 1;
  __REG32 PA2OD	: 1;
  __REG32 PA3OD	: 1;
  __REG32 PA4OD	: 1;
  __REG32 PA5OD	: 1;
  __REG32 PA6OD	: 1;
  __REG32 PA7OD	: 1;
  __REG32 		 	:24;
} __paod_bits;

/*PORT A Pull-Up Control Register */
typedef struct {
  __REG32 PA0UP	: 1;
  __REG32 PA1UP	: 1;
  __REG32 PA2UP	: 1;
  __REG32 PA3UP	: 1;
  __REG32 PA4UP	: 1;
  __REG32 PA5UP	: 1;
  __REG32 PA6UP	: 1;
  __REG32 PA7UP	: 1;
  __REG32 		 	:24;
} __papup_bits;

/*PORT A Input Enable Control Register */
typedef struct {
  __REG32 PA0IE : 1;
  __REG32 PA1IE : 1;
  __REG32 PA2IE : 1;
  __REG32 PA3IE : 1;
  __REG32 PA4IE : 1;
  __REG32 PA5IE : 1;
  __REG32 PA6IE : 1;
  __REG32 PA7IE : 1;
  __REG32 		 	:24;
} __paie_bits;

/*PORT B Register*/
typedef struct {
  __REG32 PB0  	: 1;
  __REG32 PB1  	: 1;
  __REG32 PB2  	: 1;
  __REG32 PB3  	: 1;
  __REG32 PB4  	: 1;
  __REG32 PB5  	: 1;
  __REG32 PB6  	: 1;
  __REG32 PB7  	: 1;
  __REG32 		 	:24;
} __pb_bits;

/*PORT B Control Register */
typedef struct {
  __REG32 PB0C  : 1;
  __REG32 PB1C  : 1;
  __REG32 PB2C  : 1;
  __REG32 PB3C  : 1;
  __REG32 PB4C  : 1;
  __REG32 PB5C  : 1;
  __REG32 PB6C  : 1;
  __REG32 PB7C  : 1;
  __REG32 		 	:24;
} __pbcr_bits;

/*PORT B Function Register 1*/
typedef struct {
  __REG32 PB0F1 : 1;
  __REG32 PB1F1 : 1;
  __REG32 PB2F1 : 1;
  __REG32 PB3F1 : 1;
  __REG32 PB4F1 : 1;
  __REG32 PB5F1 : 1;
  __REG32 PB6F1 : 1;
  __REG32 PB7F1 : 1;
  __REG32 		 	:24;
} __pbfr1_bits;

/*PORT B Function Register 2*/
typedef struct {
  __REG32 PB0F2 : 1;
  __REG32 PB1F2 : 1;
  __REG32 PB2F2 : 1;
  __REG32 PB3F2 : 1;
  __REG32 PB4F2 : 1;
  __REG32 PB5F2 : 1;
  __REG32 PB6F2 : 1;
  __REG32 PB7F2 : 1;
  __REG32 		 	:24;
} __pbfr2_bits;

/*Port B open drain control register*/
typedef struct {
  __REG32 PB0OD : 1;
  __REG32 PB1OD : 1;
  __REG32 PB2OD : 1;
  __REG32 PB3OD : 1;
  __REG32 PB4OD : 1;
  __REG32 PB5OD : 1;
  __REG32 PB6OD : 1;
  __REG32 PB7OD : 1;
  __REG32 		 	:24;
} __pbod_bits;

/*PORT B Pull-Up Control Register */
typedef struct {
  __REG32 PB0UP : 1;
  __REG32 PB1UP : 1;
  __REG32 PB2UP : 1;
  __REG32 PB3UP : 1;
  __REG32 PB4UP : 1;
  __REG32 PB5UP : 1;
  __REG32 PB6UP : 1;
  __REG32 PB7UP : 1;
  __REG32 		 	:24;
} __pbpup_bits;

/*PORT B Input Enable Control Register */
typedef struct {
  __REG32 PB0IE : 1;
  __REG32 PB1IE : 1;
  __REG32 PB2IE : 1;
  __REG32 PB3IE : 1;
  __REG32 PB4IE : 1;
  __REG32 PB5IE : 1;
  __REG32 PB6IE : 1;
  __REG32 PB7IE : 1;
  __REG32 		 	:24;
} __pbie_bits;

/*PORT C Register*/
typedef struct {
  __REG32 PC0  	: 1;
  __REG32 PC1  	: 1;
  __REG32 PC2  	: 1;
  __REG32 PC3  	: 1;
  __REG32 PC4  	: 1;
  __REG32 PC5  	: 1;
  __REG32 PC6  	: 1;
  __REG32 PC7  	: 1;
  __REG32 		 	:24;
} __pc_bits;

/*PORT C Control Register */
typedef struct {
  __REG32 PC0C  : 1;
  __REG32 PC1C  : 1;
  __REG32 PC2C  : 1;
  __REG32 PC3C  : 1;
  __REG32 PC4C  : 1;
  __REG32 PC5C  : 1;
  __REG32 PC6C  : 1;
  __REG32 PC7C  : 1;
  __REG32 		 	:24;
} __pccr_bits;

/*PORT C Function Register 1*/
typedef struct {
  __REG32 PC0F1 : 1;
  __REG32 PC1F1 : 1;
  __REG32 PC2F1 : 1;
  __REG32 PC3F1 : 1;
  __REG32 PC4F1 : 1;
  __REG32 PC5F1 : 1;
  __REG32 PC6F1 : 1;
  __REG32 PC7F1 : 1;
  __REG32 		 	:24;
} __pcfr1_bits;

/*PORT C Function Register 2*/
typedef struct {
  __REG32 PC0F2 : 1;
  __REG32 PC1F2 : 1;
  __REG32 PC2F2 : 1;
  __REG32 PC3F2 : 1;
  __REG32 PC4F2 : 1;
  __REG32 PC5F2 : 1;
  __REG32 PC6F2 : 1;
  __REG32 PC7F2 : 1;
  __REG32 		 	:24;
} __pcfr2_bits;

/*PORT C Function Register 3*/
typedef struct {
  __REG32 PC0F3 : 1;
  __REG32 PC1F3 : 1;
  __REG32 PC2F3 : 1;
  __REG32 PC3F3 : 1;
  __REG32 PC4F3 : 1;
  __REG32 PC5F3 : 1;
  __REG32 PC6F3 : 1;
  __REG32 PC7F3 : 1;
  __REG32 		 	:24;
} __pcfr3_bits;

/*PORT C Function Register 4*/
typedef struct {
  __REG32       : 2;
  __REG32 PC2F4 : 1;
  __REG32       : 3;
  __REG32 PC6F4 : 1;
  __REG32 		 	:25;
} __pcfr4_bits;

/*Port C open drain control register*/
typedef struct {
  __REG32 PC0OD : 1;
  __REG32 PC1OD : 1;
  __REG32 PC2OD : 1;
  __REG32 PC3OD : 1;
  __REG32 PC4OD : 1;
  __REG32 PC5OD : 1;
  __REG32 PC6OD : 1;
  __REG32 PC7OD : 1;
  __REG32 		 	:24;
} __pcod_bits;

/*PORT C Pull-Up Control Register */
typedef struct {
  __REG32 PC0UP : 1;
  __REG32 PC1UP : 1;
  __REG32 PC2UP : 1;
  __REG32 PC3UP : 1;
  __REG32 PC4UP : 1;
  __REG32 PC5UP : 1;
  __REG32 PC6UP : 1;
  __REG32 PC7UP : 1;
  __REG32 		 	:24;
} __pcpup_bits;

/*PORT C Input Enable Control Register */
typedef struct {
  __REG32 PC0IE : 1;
  __REG32 PC1IE : 1;
  __REG32 PC2IE : 1;
  __REG32 PC3IE : 1;
  __REG32 PC4IE : 1;
  __REG32 PC5IE : 1;
  __REG32 PC6IE : 1;
  __REG32 PC7IE : 1;
  __REG32 		 	:24;
} __pcie_bits;

/*PORT D Register*/
typedef struct {
  __REG32 PD0  	: 1;
  __REG32 PD1  	: 1;
  __REG32 PD2  	: 1;
  __REG32 PD3  	: 1;
  __REG32 PD4  	: 1;
  __REG32 PD5  	: 1;
  __REG32 PD6  	: 1;
  __REG32 PD7  	: 1;
  __REG32 		 	:24;
} __pd_bits;

/*PORT D Control Register */
typedef struct {
  __REG32 PD0C  : 1;
  __REG32 PD1C  : 1;
  __REG32 PD2C  : 1;
  __REG32 PD3C  : 1;
  __REG32 PD4C  : 1;
  __REG32 PD5C  : 1;
  __REG32 PD6C  : 1;
  __REG32 PD7C  : 1;
  __REG32 		 	:24;
} __pdcr_bits;

/*PORT D Function Register 1*/
typedef struct {
  __REG32 PD0F1 : 1;
  __REG32 PD1F1 : 1;
  __REG32 PD2F1 : 1;
  __REG32 PD3F1 : 1;
  __REG32 PD4F1 : 1;
  __REG32 PD5F1 : 1;
  __REG32 PD6F1 : 1;
  __REG32 PD7F1 : 1;
  __REG32 		 	:24;
} __pdfr1_bits;

/*PORT D Function Register 2*/
typedef struct {
  __REG32 PD0F2 : 1;
  __REG32 PD1F2 : 1;
  __REG32 PD2F2 : 1;
  __REG32 PD3F2 : 1;
  __REG32 PD4F2 : 1;
  __REG32 PD5F2 : 1;
  __REG32 PD6F2 : 1;
  __REG32 PD7F2 : 1;
  __REG32 		 	:24;
} __pdfr2_bits;

/*PORT D Function Register 3*/
typedef struct {
  __REG32 PD0F3 : 1;
  __REG32 PD1F3 : 1;
  __REG32 PD2F3 : 1;
  __REG32 PD3F3 : 1;
  __REG32       : 3;
  __REG32 PD7F3 : 1;
  __REG32 		 	:24;
} __pdfr3_bits;

/*Port D open drain control register*/
typedef struct {
  __REG32 PD0OD : 1;
  __REG32 PD1OD : 1;
  __REG32 PD2OD : 1;
  __REG32 PD3OD : 1;
  __REG32 PD4OD : 1;
  __REG32 PD5OD : 1;
  __REG32 PD6OD : 1;
  __REG32 PD7OD : 1;
  __REG32 		 	:24;
} __pdod_bits;

/*Port D pull-up control register*/
typedef struct {
  __REG32 PD0UP : 1;
  __REG32 PD1UP : 1;
  __REG32 PD2UP : 1;
  __REG32 PD3UP : 1;
  __REG32 PD4UP : 1;
  __REG32 PD5UP : 1;
  __REG32 PD6UP : 1;
  __REG32 PD7UP : 1;
  __REG32 		 	:24;
} __pdpup_bits;

/*PORT D Input Enable Control Register */
typedef struct {
  __REG32 PD0IE : 1;
  __REG32 PD1IE : 1;
  __REG32 PD2IE : 1;
  __REG32 PD3IE : 1;
  __REG32 PD4IE : 1;
  __REG32 PD5IE : 1;
  __REG32 PD6IE : 1;
  __REG32 PD7IE : 1;
  __REG32 		 	:24;
} __pdie_bits;

/*PORT E Register*/
typedef struct {
  __REG32 PE0  	: 1;
  __REG32 PE1  	: 1;
  __REG32 PE2  	: 1;
  __REG32 PE3  	: 1;
  __REG32 PE4  	: 1;
  __REG32 PE5  	: 1;
  __REG32 PE6  	: 1;
  __REG32 PE7  	: 1;
  __REG32 		 	:24;
} __pe_bits;

/*PORT E Control Register */
typedef struct {
  __REG32 PE0C  : 1;
  __REG32 PE1C  : 1;
  __REG32 PE2C  : 1;
  __REG32 PE3C  : 1;
  __REG32 PE4C  : 1;
  __REG32 PE5C  : 1;
  __REG32 PE6C  : 1;
  __REG32 PE7C  : 1;
  __REG32 		 	:24;
} __pecr_bits;

/*PORT E Function Register 1*/
typedef struct {
  __REG32 PE0F1 : 1;
  __REG32 PE1F1 : 1;
  __REG32 PE2F1 : 1;
  __REG32 PE3F1 : 1;
  __REG32 		 	:28;
} __pefr1_bits;

/*PORT E Function Register 2*/
typedef struct {
  __REG32 PE0F2 : 1;
  __REG32 PE1F2 : 1;
  __REG32 PE2F2 : 1;
  __REG32 PE3F2 : 1;
  __REG32 PE4F2 : 1;
  __REG32 PE5F2 : 1;
  __REG32 PE6F2 : 1;
  __REG32 PE7F2 : 1;
  __REG32 		 	:24;
} __pefr2_bits;

/*PORT E Function Register 3*/
typedef struct {
  __REG32       : 2;
  __REG32 PE2F3 : 1;
  __REG32 PE3F3 : 1;
  __REG32 PE4F3 : 1;
  __REG32 PE5F3 : 1;
  __REG32 PE6F3 : 1;
  __REG32 PE7F3 : 1;
  __REG32 		 	:24;
} __pefr3_bits;

/*PORT E Function Register 1*/
typedef struct {
  __REG32       : 2;
  __REG32 PE2F4 : 1;
  __REG32 		 	:29;
} __pefr4_bits;

/*PORT E Open Drain Control Register */
typedef struct {
  __REG32 PE0OD : 1;
  __REG32 PE1OD : 1;
  __REG32 PE2OD : 1;
  __REG32 PE3OD : 1;
  __REG32 PE4OD : 1;
  __REG32 PE5OD : 1;
  __REG32 PE6OD : 1;
  __REG32 PE7OD : 1;
  __REG32 		 	:24;
} __peod_bits;

/*PORT E Pull-Up Control Register */
typedef struct {
  __REG32 PE0UP : 1;
  __REG32 PE1UP : 1;
  __REG32 PE2UP : 1;
  __REG32 PE3UP : 1;
  __REG32 PE4UP : 1;
  __REG32 PE5UP : 1;
  __REG32 PE6UP : 1;
  __REG32 PE7UP : 1;
  __REG32 		 	:24;
} __pepup_bits;

/*PORT E Input Enable Control Register */
typedef struct {
  __REG32 PE0IE : 1;
  __REG32 PE1IE : 1;
  __REG32 PE2IE : 1;
  __REG32 PE3IE : 1;
  __REG32 PE4IE : 1;
  __REG32 PE5IE : 1;
  __REG32 PE6IE : 1;
  __REG32 PE7IE : 1;
  __REG32 		 	:24;
} __peie_bits;

/*PORT F Register*/
typedef struct {
  __REG32 PF0  	: 1;
  __REG32 PF1  	: 1;
  __REG32 PF2  	: 1;
  __REG32 PF3  	: 1;
  __REG32 PF4  	: 1;
  __REG32 PF5  	: 1;
  __REG32 PF6  	: 1;
  __REG32 PF7  	: 1;
  __REG32 		 	:24;
} __pf_bits;

/*PORT F Control Register */
typedef struct {
  __REG32 PF0C  : 1;
  __REG32 PF1C  : 1;
  __REG32 PF2C  : 1;
  __REG32 PF3C  : 1;
  __REG32 PF4C  : 1;
  __REG32 PF5C  : 1;
  __REG32 PF6C  : 1;
  __REG32 PF7C  : 1;
  __REG32 		 	:24;
} __pfcr_bits;

/*PORT F Function Register 1*/
typedef struct {
  __REG32 PF0F1 : 1;
  __REG32 PF1F1 : 1;
  __REG32 PF2F1 : 1;
  __REG32 PF3F1 : 1;
  __REG32 PF4F1 : 1;
  __REG32 PF5F1 : 1;
  __REG32 PF6F1 : 1;
  __REG32 PF7F1 : 1;
  __REG32 		 	:24;
} __pffr1_bits;

/*PORT F Function Register 2*/
typedef struct {
  __REG32       : 4;
  __REG32 PF4F2 : 1;
  __REG32 PF5F2 : 1;
  __REG32 		 	:26;
} __pffr2_bits;

/*PORT F Function Register 3*/
typedef struct {
  __REG32 PF0F3 : 1;
  __REG32       : 3;
  __REG32 PF4F3 : 1;
  __REG32 PF5F3 : 1;
  __REG32 		 	:26;
} __pffr3_bits;

/*Port F open drain control register */
typedef struct {
  __REG32 PF0OD : 1;
  __REG32 PF1OD : 1;
  __REG32 PF2OD : 1;
  __REG32 PF3OD : 1;
  __REG32 PF4OD : 1;
  __REG32 PF5OD : 1;
  __REG32 PF6OD : 1;
  __REG32 PF7OD : 1;
  __REG32 		 	:24;
} __pfod_bits;

/*PORT F Pull-Up Control Register */
typedef struct {
  __REG32 PF0UP : 1;
  __REG32 PF1UP : 1;
  __REG32 PF2UP : 1;
  __REG32 PF3UP : 1;
  __REG32 PF4UP : 1;
  __REG32 PF5UP : 1;
  __REG32 PF6UP : 1;
  __REG32 PF7UP : 1;
  __REG32 		 	:24;
} __pfpup_bits;

/*PORT F Input Enable Control Register */
typedef struct {
  __REG32       : 1;
  __REG32 PF1IE : 1;
  __REG32 PF2IE : 1;
  __REG32 PF3IE : 1;
  __REG32 PF4IE : 1;
  __REG32 PF5IE : 1;
  __REG32 PF6IE : 1;
  __REG32 PF7IE : 1;
  __REG32 		 	:24;
} __pfie_bits;

/*PORT G Register*/
typedef struct {
  __REG32 PG0  	: 1;
  __REG32 PG1  	: 1;
  __REG32 PG2  	: 1;
  __REG32 PG3  	: 1;
  __REG32 PG4  	: 1;
  __REG32 PG5  	: 1;
  __REG32 PG6  	: 1;
  __REG32 PG7  	: 1;
  __REG32 		 	:24;
} __pg_bits;

/*PORT G Control Register*/
typedef struct {
  __REG32 PG0C  : 1;
  __REG32 PG1C  : 1;
  __REG32 PG2C  : 1;
  __REG32 PG3C  : 1;
  __REG32 PG4C  : 1;
  __REG32 PG5C  : 1;
  __REG32 PG6C  : 1;
  __REG32 PG7C  : 1;
  __REG32 		 	:24;
} __pgcr_bits;

/*PORT G Function Register 2*/
typedef struct {
  __REG32 PG0F2 : 1;
  __REG32 PG1F2 : 1;
  __REG32 PG2F2 : 1;
  __REG32 PG3F2 : 1;
  __REG32 PG4F2 : 1;
  __REG32 PG5F2 : 1;
  __REG32 PG6F2 : 1;
  __REG32 PG7F2 : 1;
  __REG32 		 	:24;
} __pgfr2_bits;

/*PORT G Function Register 3*/
typedef struct {
  __REG32 PG0F3 : 1;
  __REG32 PG1F3 : 1;
  __REG32 PG2F3 : 1;
  __REG32       : 1;
  __REG32 PG4F3 : 1;
  __REG32 PG5F3 : 1;
  __REG32 PG6F3 : 1;
  __REG32 PG7F3 : 1;
  __REG32 		 	:24;
} __pgfr3_bits;

/*PORT G Function Register 4*/
typedef struct {
  __REG32       : 6;
  __REG32 PG6F4 : 1;
  __REG32 		 	:25;
} __pgfr4_bits;

/*PORT G Open Drain Control Register */
typedef struct {
  __REG32 PG0OD : 1;
  __REG32 PG1OD : 1;
  __REG32 PG2OD : 1;
  __REG32 PG3OD : 1;
  __REG32 PG4OD : 1;
  __REG32 PG5OD : 1;
  __REG32 PG6OD : 1;
  __REG32 PG7OD : 1;
  __REG32 		 	:24;
} __pgod_bits;

/*PORT G Pull-Up Control Register */
typedef struct {
  __REG32 PG0UP : 1;
  __REG32 PG1UP : 1;
  __REG32 PG2UP : 1;
  __REG32 PG3UP : 1;
  __REG32 PG4UP : 1;
  __REG32 PG5UP : 1;
  __REG32 PG6UP : 1;
  __REG32 PG7UP : 1;
  __REG32 		 	:24;
} __pgpup_bits;

/*PORT G Input Enable Control Register */
typedef struct {
  __REG32 PG0IE : 1;
  __REG32 PG1IE : 1;
  __REG32 PG2IE : 1;
  __REG32 PG3IE : 1;
  __REG32 PG4IE : 1;
  __REG32 PG5IE : 1;
  __REG32 PG6IE : 1;
  __REG32 PG7IE : 1;
  __REG32 		 	:24;
} __pgie_bits;

/*PORT H Register*/
typedef struct {
  __REG32 PH0  	: 1;
  __REG32 PH1  	: 1;
  __REG32 PH2  	: 1;
  __REG32 PH3  	: 1;
  __REG32 PH4  	: 1;
  __REG32 PH5  	: 1;
  __REG32 PH6  	: 1;
  __REG32 		 	:25;
} __ph_bits;

/*PORT H Control Register 1*/
typedef struct {
  __REG32 PH0C  : 1;
  __REG32 PH1C  : 1;
  __REG32 PH2C  : 1;
  __REG32 PH3C  : 1;
  __REG32 PH4C  : 1;
  __REG32 PH5C  : 1;
  __REG32 PH6C  : 1;
  __REG32 		 	:25;
} __phcr_bits;

/*PORT H Function Register 1*/
typedef struct {
  __REG32       : 5;
  __REG32 PH5F1 : 1;
  __REG32 PH6F1 : 1;
  __REG32 		 	:25;
} __phfr1_bits;

/*PORT H Function Register 2*/
typedef struct {
  __REG32 PH0F2 : 1;
  __REG32 PH1F2 : 1;
  __REG32 PH2F2 : 1;
  __REG32 PH3F2 : 1;
  __REG32 PH4F2 : 1;
  __REG32 		 	:27;
} __phfr2_bits;

/*PORT H Function Register 3*/
typedef struct {
  __REG32       : 3;
  __REG32 PH3F3 : 1;
  __REG32 PH4F3 : 1;
  __REG32 		 	:27;
} __phfr3_bits;

/*PORT H Function Register 4*/
typedef struct {
  __REG32       : 2;
  __REG32 PH2F4 : 1;
  __REG32 		 	:29;
} __phfr4_bits;

/*Port H open drain control register*/
typedef struct {
  __REG32 PH0OD : 1;
  __REG32 PH1OD : 1;
  __REG32 PH2OD : 1;
  __REG32 PH3OD : 1;
  __REG32 PH4OD : 1;
  __REG32 PH5OD : 1;
  __REG32 PH6OD : 1;
  __REG32 		 	:25;
} __phod_bits;

/*PORT H Pull-Up Control Register */
typedef struct {
  __REG32 PH0UP : 1;
  __REG32 PH1UP : 1;
  __REG32 PH2UP : 1;
  __REG32 PH3UP : 1;
  __REG32 PH4UP : 1;
  __REG32 PH5UP : 1;
  __REG32 PH6UP : 1;
  __REG32 		 	:25;
} __phpup_bits;

/*PORT H Input Enable Control Register */
typedef struct {
  __REG32 PH0IE : 1;
  __REG32 PH1IE : 1;
  __REG32 PH2IE : 1;
  __REG32 PH3IE : 1;
  __REG32 PH4IE : 1;
  __REG32 PH5IE : 1;
  __REG32 PH6IE : 1;
  __REG32 		 	:25;
} __phie_bits;

/*PORT I Register*/
typedef struct {
  __REG32 PI0  	: 1;
  __REG32 PI1  	: 1;
  __REG32 PI2  	: 1;
  __REG32 PI3  	: 1;
  __REG32 PI4  	: 1;
  __REG32 PI5  	: 1;
  __REG32 PI6  	: 1;
  __REG32 PI7  	: 1;
  __REG32 		 	:24;
} __pi_bits;

/*PORT I Control Register 1*/
typedef struct {
  __REG32 PI0C  : 1;
  __REG32 PI1C  : 1;
  __REG32 PI2C  : 1;
  __REG32 PI3C  : 1;
  __REG32 PI4C  : 1;
  __REG32 PI5C  : 1;
  __REG32 PI6C  : 1;
  __REG32 PI7C  : 1;
  __REG32 		 	:24;
} __picr_bits;

/*Port I function register 1*/
typedef struct {
  __REG32 PI0F1 : 1;
  __REG32 PI1F1 : 1;
  __REG32 PI2F1 : 1;
  __REG32 PI3F1 : 1;
  __REG32 PI4F1 : 1;
  __REG32 PI5F1 : 1;
  __REG32 PI6F1 : 1;
  __REG32 PI7F1 : 1;
  __REG32 		 	:24;
} __pifr1_bits;

/*Port I open drain control register*/
typedef struct {
  __REG32 PI0OD : 1;
  __REG32 PI1OD : 1;
  __REG32 PI2OD : 1;
  __REG32 		 	:29;
} __piod_bits;

/*PORT I Pull-Up Control Register */
typedef struct {
  __REG32 PI0UP : 1;
  __REG32 PI1UP : 1;
  __REG32 PI2UP : 1;
  __REG32 PI3UP : 1;
  __REG32 PI4UP : 1;
  __REG32       : 1;
  __REG32 PI6UP : 1;
  __REG32 PI7UP : 1;
  __REG32 		 	:24;
} __pipup_bits;

/*Port I Pull-Down control register*/
typedef struct {
  __REG32       : 5;
  __REG32 PI5DN : 1;
  __REG32 		 	:26;
} __pipdn_bits;

/*PORT I Input Enable Control Register */
typedef struct {
  __REG32 PI0IE : 1;
  __REG32 PI1IE : 1;
  __REG32 PI2IE : 1;
  __REG32 PI3IE : 1;
  __REG32 PI4IE : 1;
  __REG32 PI5IE : 1;
  __REG32 PI6IE : 1;
  __REG32 PI7IE : 1;
  __REG32 		 	:24;
} __piie_bits;

/*PORT J Register*/
typedef struct {
  __REG32 PJ0  	: 1;
  __REG32 PJ1  	: 1;
  __REG32 PJ2  	: 1;
  __REG32 PJ3  	: 1;
  __REG32 PJ4  	: 1;
  __REG32 PJ5  	: 1;
  __REG32 PJ6  	: 1;
  __REG32 PJ7  	: 1;
  __REG32 		 	:24;
} __pj_bits;

/*PORT J Control Register 1*/
typedef struct {
  __REG32 PJ0C  : 1;
  __REG32 PJ1C  : 1;
  __REG32 PJ2C  : 1;
  __REG32 PJ3C  : 1;
  __REG32 PJ4C  : 1;
  __REG32 PJ5C  : 1;
  __REG32 PJ6C  : 1;
  __REG32 PJ7C  : 1;
  __REG32 		 	:24;
} __pjcr_bits;

/*PORT J Function Register 2*/
typedef struct {
  __REG32       : 7;
  __REG32 PJ7F2 : 1;
  __REG32 		 	:24;
} __pjfr2_bits;

/*PORT J Function Register 3*/
typedef struct {
  __REG32 PJ0F3 : 1;
  __REG32 PJ1F3 : 1;
  __REG32 PJ2F3 : 1;
  __REG32 PJ3F3 : 1;
  __REG32 PJ4F3 : 1;
  __REG32 PJ5F3 : 1;
  __REG32 PJ6F3 : 1;
  __REG32 PJ7F3 : 1;
  __REG32 		 	:24;
} __pjfr3_bits;

/*PORT J Pull-Up Control Register */
typedef struct {
  __REG32 PJ0UP : 1;
  __REG32 PJ1UP : 1;
  __REG32 PJ2UP : 1;
  __REG32 PJ3UP : 1;
  __REG32 PJ4UP : 1;
  __REG32 PJ5UP : 1;
  __REG32 PJ6UP : 1;
  __REG32 PJ7UP : 1;
  __REG32 		 	:24;
} __pjpup_bits;

/*PORT J Input Enable Control Register */
typedef struct {
  __REG32 PJ0IE : 1;
  __REG32 PJ1IE : 1;
  __REG32 PJ2IE : 1;
  __REG32 PJ3IE : 1;
  __REG32 PJ4IE : 1;
  __REG32 PJ5IE : 1;
  __REG32 PJ6IE : 1;
  __REG32 PJ7IE : 1;
  __REG32 		 	:24;
} __pjie_bits;

/*PORT K Register*/
typedef struct {
  __REG32 PK0  	: 1;
  __REG32 PK1  	: 1;
  __REG32 PK2  	: 1;
  __REG32 PK3  	: 1;
  __REG32 PK4  	: 1;
  __REG32 PK5  	: 1;
  __REG32 PK6  	: 1;
  __REG32 		 	:25;
} __pk_bits;

/*PORT K Control Register*/
typedef struct {
  __REG32 PK0C  : 1;
  __REG32 PK1C  : 1;
  __REG32 PK2C  : 1;
  __REG32 PK3C  : 1;
  __REG32 PK4C  : 1;
  __REG32 PK5C  : 1;
  __REG32 PK6C  : 1;
  __REG32 		 	:25;
} __pkcr_bits;

/*PORT K Function Register 2*/
typedef struct {
  __REG32       : 1;
  __REG32 PK1F2 : 1;
  __REG32       : 1;
  __REG32 PK3F2 : 1;
  __REG32 		 	:28;
} __pkfr2_bits;

/*PORT K Function Register 3*/
typedef struct {
  __REG32 PK0F3 : 1;
  __REG32 PK1F3 : 1;
  __REG32 PK2F3 : 1;
  __REG32 PK3F3 : 1;
  __REG32 		 	:28;
} __pkfr3_bits;

/*PORT K Pull-Up Control Register*/
typedef struct {
  __REG32 PK0UP : 1;
  __REG32 PK1UP : 1;
  __REG32 PK2UP : 1;
  __REG32 PK3UP : 1;
  __REG32 PK4UP : 1;
  __REG32 PK5UP : 1;
  __REG32 PK6UP : 1;
  __REG32 		 	:25;
} __pkpup_bits;

/*PORT K Input Enable Control Register*/
typedef struct {
  __REG32 PK0IE : 1;
  __REG32 PK1IE : 1;
  __REG32 PK2IE : 1;
  __REG32 PK3IE : 1;
  __REG32 PK4IE : 1;
  __REG32 PK5IE : 1;
  __REG32 PK6IE : 1;
  __REG32 		 	:25;
} __pkie_bits;

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
  __REG32           : 1;
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
  __REG32  UC       :16;
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

/*PHCNT RUN register (channels 0 through 3)*/
typedef struct {
  __REG32  PHCRUN   : 1;
  __REG32  PHCCLR   : 1;
  __REG32           :30;
} __phcxrun_bits;

/*PHCNT control register (channels 0 through 3)*/
typedef struct {
  __REG32  PHCMD    : 2;
  __REG32  NFOFF    : 1;
  __REG32  CMP0EN   : 1;
  __REG32  CMP1EN   : 1;
  __REG32  EVRYINT  : 1;
  __REG32           :26;
} __phcxcr_bits;

/*PHCNT Timer Enable Register (channels 0 through 3)*/
typedef struct {
  __REG32  PHCEN    : 1;
  __REG32           :31;
} __phcxen_bits;

/*PHCNT Status register (channels 0 through 3)*/
typedef struct {
  __REG32  CMP0     : 1;
  __REG32  CMP1     : 1;
  __REG32  OVF      : 1;
  __REG32  UDF      : 1;
  __REG32           :28;
} __phcxflg_bits;

/*PHCNT Compare Register 0 (channels 0 through 3)*/
typedef struct {
  __REG32  PHCCMP0  :16;
  __REG32           :16;
} __phcxcmp0_bits;

/*PHCNT Compare Register 1 (channels 0 through 3)*/
typedef struct {
  __REG32  PHCCMP1  :16;
  __REG32           :16;
} __phcxcmp1_bits;

/*PHCNT Count Register (channels 0 through 3)*/
typedef struct {
  __REG32  PHCCNT   :16;
  __REG32           :16;
} __phcxcnt_bits;

/*PHCNT DMA enable Register (channels 0 through 3)*/
typedef struct {
  __REG32           : 2;
  __REG32  PHCDMA2  : 1;
  __REG32           :29;
} __phcxdma_bits;

/*SIOx Enable register*/
typedef struct {
  __REG32 SIOE     : 1;
  __REG32          :31;
} __scxen_bits;

/*SIOx Control register*/
typedef struct {
  __REG32 IOC      : 1;
  __REG32 SCLKS    : 1;
  __REG32 FERR     : 1;
  __REG32 PERR     : 1;
  __REG32 OERR     : 1;
  __REG32 PE       : 1;
  __REG32 EVEN     : 1;
  __REG32 RB8      : 1;
  __REG32          :24;
} __scxcr_bits;

/*SIOx Mode control register 0*/
typedef struct {
  __REG32 SC       : 2;
  __REG32 SM       : 2;
  __REG32 WU       : 1;
  __REG32 RXE      : 1;
  __REG32 CTSE     : 1;
  __REG32 TB8      : 1;
  __REG32          :24;
} __scxmod0_bits;

/*SIOx Baud rate generator control register*/
typedef struct {
  __REG32 BRS      : 4;
  __REG32 BRCK     : 2;
  __REG32 BRADDE   : 1;
  __REG32          :25;
} __scxbrcr_bits;

/*SIOx Baud rate generator control register 2*/
typedef struct {
  __REG32 BRK      : 4;
  __REG32          :28;
} __scxbradd_bits;

/*SIOx Mode control register 1*/
typedef struct {
  __REG32          : 1;
  __REG32 SINT     : 3;
  __REG32 TXE      : 1;
  __REG32 FDPX     : 2;
  __REG32 I2SC     : 1;
  __REG32          :24;
} __scxmod1_bits;

/*SIOx Mode control register 2*/
typedef struct {
  __REG32 SWRST    : 2;
  __REG32 WBUF     : 1;
  __REG32 DRCHG    : 1;
  __REG32 SBLEN    : 1;
  __REG32 TXRUN    : 1;
  __REG32 RBFLL    : 1;
  __REG32 TBEMP    : 1;
  __REG32          :24;
} __scxmod2_bits;

/*SIOx RX FIFO configuration register*/
typedef struct {
  __REG32 RIL      : 2;
  __REG32          : 4;
  __REG32 RFIS     : 1;
  __REG32 RFCS     : 1;
  __REG32          :24;
} __scxrfc_bits;

/*SIOx TX FIFO configuration register*/
typedef struct {
  __REG32 TIL      : 2;
  __REG32          : 4;
  __REG32 TFIS     : 1;
  __REG32 TFCS     : 1;
  __REG32          :24;
} __scxtfc_bits;

/*SIOx RX FIFO status register*/
typedef struct {
  __REG32 RLVL     : 3;
  __REG32          : 4;
  __REG32 ROR      : 1;
  __REG32          :24;
} __scxrst_bits;

/*SIOx TX FIFO status register*/
typedef struct {
  __REG32 TLVL     : 3;
  __REG32          : 4;
  __REG32 TUR      : 1;
  __REG32          :24;
} __scxtst_bits;

/*SIOx FIFO configuration register*/
typedef struct {
  __REG32 CNFG     : 1;
  __REG32 RXTXCNT  : 1;
  __REG32 RFIE     : 1;
  __REG32 TFIE     : 1;
  __REG32 RFST     : 1;
  __REG32          :27;
} __scxfcnf_bits;

/*SIOx DMA enable register*/
typedef struct {
  __REG32 DMAEN0   : 1;
  __REG32 DMAEN1   : 1;
  __REG32          :30;
} __scxdma_bits;

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
  struct {
    __REG32  SWRMON   : 1;
    __REG32           :31;
  };
} __sbixcr1_bits;

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

/*Serial bus control register 2*/
/*Serial bus status register*/
typedef union {
  /*SBI0CR2*/
  /*SBI1CR2*/
  struct {
  __REG32 SWRST   : 2;
  __REG32 SBIM    : 2;
  __REG32 PIN     : 1;
  __REG32 BB      : 1;
  __REG32 TRX     : 1;
  __REG32 MST     : 1;
  __REG32         :24;
  };
  /*SBI0SR*/
  /*SBI1SR*/
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
  } __sbixsr;
} __sbixcr2_bits;

/*Serial bus interface baud rate register 0*/
typedef struct {
  __REG32         : 6;
  __REG32 I2SBI   : 1;
  __REG32         :25;
} __sbixbr0_bits;

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

/*SSPDMACR (SSP DMA control register)*/
typedef struct {
  __REG32 RXDMAE  : 1;
  __REG32 TXDMAE  : 1;
  __REG32         :30;
} __sspdmacr_bits;

/*A/D Conversion Clock Setting Register*/
typedef struct {
  __REG32 ADCLK   : 3;
  __REG32         : 1;
  __REG32 ADSH    : 4;
  __REG32         :24;
} __adclk_bits;

/*A/D Mode Control Register 0*/
typedef struct {
  __REG32 ADS     : 1;
  __REG32 HPADS   : 1;
  __REG32         :30;
} __admod0_bits;

/*A/D Mode Control Register 1*/
typedef struct {
  __REG32 ADHWE   : 1;
  __REG32 ADHWS   : 1;
  __REG32 HPADHWE : 1;
  __REG32 HPADHWS : 1;
  __REG32         : 1;
  __REG32 RCUT    : 1;
  __REG32 I2AD    : 1;
  __REG32 VREFON  : 1;
  __REG32         :24;
} __admod1_bits;

/*A/D Mode Control Register 2*/
typedef struct {
  __REG32 ADCH    : 4;
  __REG32 HPADCH  : 4;
  __REG32         :24;
} __admod2_bits;

/*A/D Mode Control Register 3*/
typedef struct {
  __REG32 SCAN    : 1;
  __REG32 REPEAT  : 1;
  __REG32         : 2;
  __REG32 ITM     : 3;
  __REG32         :25;
} __admod3_bits;

/*A/D Mode Control Register 4*/
typedef struct {
  __REG32 SCANSTA   : 4;
  __REG32 SCANAREA  : 4;
  __REG32         	:24;
} __admod4_bits;

/*A/D Mode Control Register 5*/
typedef struct {
  __REG32 ADBF    : 1;
  __REG32 EOCF    : 1;
  __REG32 HPADBF  : 1;
  __REG32 HPEOCF  : 1;
  __REG32         :28;
} __admod5_bits;

/*A/D Mode Control Register 6*/
typedef struct {
  __REG32 ADRST   : 2;
  __REG32         :30;
} __admod6_bits;

/*A/D Mode Control Register 7*/
typedef struct {
  __REG32 INTADDMA   	: 1;
  __REG32 INTADHPDMA 	: 1;
  __REG32         		:30;
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
  __REG32  ADRSP    :12;
  __REG32  ADRFSP   : 1;
  __REG32  ADOVRFSP : 1;
  __REG32  ADPOSWFSP: 1;
  __REG32           :17;
} __adregsp_bits;

/*A/D Conversion Comparison Control Register 0*/
typedef struct {
  __REG32  AINS0    : 4;
  __REG32  ADBIG0   : 1;
  __REG32  CMPCOND0 : 1;
  __REG32           : 1;
  __REG32  CMP0EN   : 1;
  __REG32  CMPCNT0  : 4;
  __REG32           : 20;
} __adcmpcr0_bits;

/*A/D Conversion Comparison Control Register 1*/
typedef struct {
  __REG32  AINS1    : 4;
  __REG32  ADBIG1   : 1;
  __REG32  CMPCOND1 : 1;
  __REG32           : 1;
  __REG32  CMP1EN   : 1;
  __REG32  CMPCNT1  : 4;
  __REG32           : 20;
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

/*D/A Conversion Control register */
typedef struct {
  __REG32  OP       : 1;
  __REG32  VREFON   : 1;
  __REG32           :30;
} __daxctl_bits;

/*D/A Conversion data register 1*/
typedef struct {
  __REG32  DAC      :10;
  __REG32           :22;
} __daxreg_bits;

/*TD Enable Register*/
typedef struct {
  __REG32           : 5;
  __REG32  TDHALT   : 1;
  __REG32  TDEN0    : 1;
  __REG32  TDEN1    : 1;
  __REG32           :24;
} __tdxen_bits;

/*TD Configuration Register*/
typedef struct {
  __REG32  TMRDMOD  : 3;
  __REG32           : 3;
  __REG32  TDI2TD0  : 1;
  __REG32  TDI2TD1  : 1;
  __REG32           :24;
} __tdxconf_bits;

/*TD Mode Register*/
typedef struct {
  __REG32  TDCLK    : 4;
  __REG32  TDCLE    : 1;
  __REG32           : 1;
  __REG32  TDIV0    : 1;
  __REG32  TDIV1    : 1;
  __REG32           :24;
} __tdxmod_bits;

/*TD Control Register*/
typedef struct {
  __REG32  TDISO0   : 1;
  __REG32  TDISO1   : 1;
  __REG32  TDRDE    : 1;
  __REG32           :29;
} __tdxcr_bits;

/*TD RUN Register*/
typedef struct {
  __REG32  TDRUN    : 1;
  __REG32           :31;
} __tdxrun_bits;

/*TD0 BCR Register*/
typedef struct {
  __REG32  TDSFT    : 1;
  __REG32  PHSCHG   : 1;
  __REG32           :30;
} __tdxbcr0_bits;

/*TD1 BCR Register*/
typedef struct {
  __REG32  TDSFT    : 1;
  __REG32           :31;
} __tdxbcr1_bits;

/*TD DMA enable Register*/
typedef struct {
  __REG32  DMAEN    : 1;
  __REG32           :31;
} __tdxdma_bits;

/*TD Timer Register 0*/
typedef struct {
  __REG32  TDRG0    :16;
  __REG32           :16;
} __tdxrg0_bits;

/*TD Compare Register 0*/
typedef struct {
  __REG32  CPRG0    :16;
  __REG32           :16;
} __tdxcp0_bits;

/*TD Timer Register 1*/
typedef struct {
  __REG32  TDRG1    :16;
  __REG32           :16;
} __tdxrg1_bits;

/*TD Compare Register 1*/
typedef struct {
  __REG32  CPRG1    :16;
  __REG32           :16;
} __tdxcp1_bits;

/*TD Timer Register 2*/
typedef struct {
  __REG32  TDRG2    :16;
  __REG32           :16;
} __tdxrg2_bits;

/*TD Compare Register 2*/
typedef struct {
  __REG32  CPRG2    :16;
  __REG32           :16;
} __tdxcp2_bits;

/*TD Timer Register 3*/
typedef struct {
  __REG32  TDRG3    :16;
  __REG32           :16;
} __tdxrg3_bits;

/*TD Compare Register 3*/
typedef struct {
  __REG32  CPRG3    :16;
  __REG32           :16;
} __tdxcp3_bits;

/*TD Timer Register 4*/
typedef struct {
  __REG32  TDRG4    :16;
  __REG32           :16;
} __tdxrg4_bits;

/*TD Compare Register 4*/
typedef struct {
  __REG32  CPRG4    :16;
  __REG32           :16;
} __tdxcp4_bits;

/*TD Timer Register 5*/
typedef struct {
  __REG32  TDRG5    :16;
  __REG32           :16;
} __tdxrg5_bits;

/*TD Compare Register 5*/
typedef struct {
  __REG32  CPRG5    :16;
  __REG32           :16;
} __tdxcp5_bits;

/*External Bus Mode Controller Register*/
typedef struct {
  __REG32  EXBSEL   : 1;
  __REG32  EXBWAIT  : 2;
  __REG32           :29;
} __exbmod_bits;

/*External Bus Start Address Register*/
typedef struct {
  __REG32  EXAR     : 8;
  __REG32           : 8;
  __REG32  SA       : 8;
  __REG32           : 8;
} __exbas_bits;

/*External Bus Chip Control Register*/
typedef struct {
  __REG32  CSW0     : 1;
  __REG32  CSW      : 2;
  __REG32           : 4;
  __REG32  ENDTYPE  : 1;
  __REG32  CSIW     : 5;
  __REG32           : 3;
  __REG32  RDS      : 2;
  __REG32  WRS      : 2;
  __REG32  ALEW     : 2;
  __REG32           : 2;
  __REG32  RDR      : 3;
  __REG32  WRR      : 3;
  __REG32  CSR      : 2;
} __exbcs_bits;

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

/*Lower detection frequency setting register*/
typedef struct {
  __REG32 OFDMN     : 8;
  __REG32           :24;
} __ofdmn_bits;

/*Higher detection frequency setting register*/
typedef struct {
  __REG32 OFDMX     : 8;
  __REG32           :24;
} __ofdmx_bits;

/*Reset Enable Control Register*/
typedef struct {
  __REG32 OFDRSTEN  : 1;
  __REG32           :31;
} __ofdrst_bits;

/*Status Register*/
typedef struct {
  __REG32 FRQERR    : 1;
  __REG32 OFDBUSY   : 1;
  __REG32           :30;
} __ofdstat_bits;

/*Watchdog Timer Mode Register*/
typedef struct {
  __REG32         : 1;
  __REG32  RESCR  : 1;
  __REG32  I2WDT  : 1;
  __REG32         : 1;
  __REG32  WDTP   : 3;
  __REG32  WDTE   : 1;
  __REG32         :24;
} __wdmod_bits;


/* CGSYSCR Register */
typedef struct {
  __REG32  GEAR     : 3;
  __REG32           : 5;
  __REG32  PRCK     : 3;
  __REG32           : 1;
  __REG32  FPSEL    : 1;
  __REG32           : 3;
  __REG32  SCOSEL   : 2;
  __REG32           : 2;
  __REG32  FCSTOP   : 1;
  __REG32           : 11;
} __cgsyscr_bits;

/* CGOSCCR Register */
typedef struct {
  __REG32  WUEON    : 1;
  __REG32  WUEF     : 1;
  __REG32  PLLON    : 1;
  __REG32           : 5;
  __REG32  XEN1     : 1;
  __REG32           : 7;
  __REG32  XEN2     : 1;
  __REG32  OSCSEL   : 1;
  __REG32  EHOSCSEL : 1;
  __REG32  HWUPSEL  : 1;
  __REG32  WUODR    : 12;
} __cgosccr_bits;

/* CGSTBYCR Register */
typedef struct {
  __REG32  STBY     : 3;
  __REG32           : 5;
  __REG32           : 8;
  __REG32  DRVE     : 1;
  __REG32  PTKEEP   : 1;
  __REG32           : 14;
} __cgstbycr_bits;

/* CGPLLSEL Register */
typedef struct {
  __REG32  PLLSEL    : 1;
  __REG32  PLLSET    : 15;
  __REG32            : 16;
} __cgpllsel_bits;

/* CGPWMGEAR Register */
typedef struct {
  __REG32  TMRDCLKEN : 1;
  __REG32            : 3;
  __REG32  PWMGEAR   : 2;
  __REG32            : 26;
} __cgpwngear_bits;

/* CGPROTECT Register */
typedef struct {
  __REG32  CGPROTECT : 8;
  __REG32            : 24;
} __cgprotect_bits;



/* CG Interrupt Mode Control Register A */
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
  __REG32 INTC10N   : 1;
  __REG32           : 1;
  __REG32 EMSTC10   : 2;
  __REG32 EMCGC10   : 3;
  __REG32           : 1;
  __REG32 INTC11N   : 1;
  __REG32           : 1;
  __REG32 EMSTC11   : 2;
  __REG32 EMCGC11   : 3;
  __REG32           : 1;
  __REG32 INTC12N   : 1;
  __REG32           : 1;
  __REG32 EMSTC12   : 2;
  __REG32 EMCGC12   : 3;
  __REG32           : 1;
  __REG32 INTC13N   : 1;
  __REG32           : 1;
  __REG32 EMSTC13   : 2;
  __REG32 EMCGC13   : 3;
  __REG32           : 1;
} __cgimcge_bits;

/* CG Interrupt Mode Control Register F */
typedef struct {
  __REG32 INTC14N   : 1;
  __REG32           : 1;
  __REG32 EMSTC14   : 2;
  __REG32 EMCGC14   : 3;
  __REG32           : 1;
  __REG32 INTC15N   : 1;
  __REG32           : 1;
  __REG32 EMSTC15   : 2;
  __REG32 EMCGC15   : 3;
  __REG32           : 1;
  __REG32 INTC16N   : 1;
  __REG32           : 1;
  __REG32 EMSTC16   : 2;
  __REG32 EMCGC16   : 3;
  __REG32           : 1;
  __REG32 INTC17N   : 1;
  __REG32           : 1;
  __REG32 EMSTC17   : 2;
  __REG32 EMCGC17   : 3;
  __REG32           : 1;
} __cgimcgf_bits;

/* CGICRCG Register */
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           : 27;
} __cgicrcg_bits;

/* CGRSTFLG Register */
typedef struct {
  __REG32 PINRSTF   : 1;
  __REG32           : 1;
  __REG32 WDTRSTF   : 1;
  __REG32 STOP2RSTF : 1;
  __REG32 DBGRSTF   : 1;
  __REG32 OFDRSTF   : 1;
  __REG32           : 26;
} __cgrstflg_bits;

/* CGNMIFLG Register */
typedef struct {
  __REG32 NMIFLG0   : 1;
  __REG32 NMIFLG1   : 1;
  __REG32           :30;
} __cgnmiflg_bits;


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

/* Interrupt Set-Enable Registers 64-84 */
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
  __REG32                 : 11;
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

/* Interrupt Clear-Enable Registers 64-84 */
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
  __REG32                 : 11;
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

/* Interrupt Set-Pending Registers 64-84 */
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
  __REG32                 : 11;
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

/* Interrupt Clear-Pending Registers 64-84 */
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
  __REG32                 : 11;
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

/* Interrupt Priority Registers 84 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_84         : 3;
  __REG32                 :24;
} __pri21_bits;

/* Vector Table Offset Register */
typedef struct {
  __REG32                 : 7;
  __REG32  TBLOFF         :22;
  __REG32  TBLBASE        : 1;
  __REG32                 : 2;
} __vtor_bits;

/* System Handler Priority Registers 4-7 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_4          : 3;
  __REG32                 : 5;
  __REG32  PRI_5          : 3;
  __REG32                 : 5;
  __REG32  PRI_6          : 3;
  __REG32                 : 5;
  __REG32  PRI_7          : 3;
} __ship0_bits;

/* System Handler Priority Registers 8-11 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_8          : 3;
  __REG32                 : 5;
  __REG32  PRI_9          : 3;
  __REG32                 : 5;
  __REG32  PRI_10         : 3;
  __REG32                 : 5;
  __REG32  PRI_11         : 3;
} __ship1_bits;

/* System Handler Priority Registers 12-15 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_12         : 3;
  __REG32                 : 5;
  __REG32  PRI_13         : 3;
  __REG32                 : 5;
  __REG32  PRI_14         : 3;
  __REG32                 : 5;
  __REG32  PRI_15         : 3;
} __ship2_bits;

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
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(SHIP0,             0xE000ED18,__READ_WRITE ,__ship0_bits);
__IO_REG32_BIT(SHIP1,             0xE000ED1C,__READ_WRITE ,__ship1_bits);
__IO_REG32_BIT(SHIP2,             0xE000ED20,__READ_WRITE ,__ship2_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);

/***************************************************************************
 **
 ** DMAC 0
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACAINTSTATUS,        0x40000000,__READ       ,__dmacxintstatus_bits);
__IO_REG32_BIT(DMACAINTTCSTATUS,      0x40000004,__READ       ,__dmacxinttcstatus_bits);
__IO_REG32_BIT(DMACAINTTCCLEAR,       0x40000008,__WRITE      ,__dmacxinttcclear_bits);
__IO_REG32_BIT(DMACAINTERRORSTATUS,   0x4000000C,__READ       ,__dmacxinterrorstatus_bits);
__IO_REG32_BIT(DMACAINTERRCLR,        0x40000010,__WRITE      ,__dmacxinterrclr_bits);
__IO_REG32_BIT(DMACARAWINTTCSTATUS,   0x40000014,__READ       ,__dmacxrawinttcstatus_bits);
__IO_REG32_BIT(DMACARAWINTERRORSTATUS, 0x40000018,__READ       ,__dmacxrawinterrorstatus_bits);
__IO_REG32_BIT(DMACAENBLDCHNS,        0x4000001C,__READ       ,__dmacxenbldchns_bits);
__IO_REG32_BIT(DMACASOFTBREQ,         0x40000020,__READ_WRITE ,__dmacxsoftbreq_bits);
__IO_REG32(    DMACASOFTSREQ,         0x40000024,__READ_WRITE);
__IO_REG32_BIT(DMACACONFIGURATION,    0x40000030,__READ_WRITE ,__dmacxconfiguration_bits);
__IO_REG32(    DMACAC0SRCADDR,        0x40000100,__READ_WRITE);
__IO_REG32(    DMACAC0DESTADDR,       0x40000104,__READ_WRITE);
__IO_REG32_BIT(DMACAC0LLI,            0x40000108,__READ_WRITE ,__dmacxc0lli_bits);
__IO_REG32_BIT(DMACAC0CONTROL,        0x4000010C,__READ_WRITE ,__dmacxc0control_bits);
__IO_REG32_BIT(DMACAC0CONFIGURATION,  0x40000110,__READ_WRITE ,__dmacxc0configuration_bits);
__IO_REG32(    DMACAC1SRCADDR,        0x40000120,__READ_WRITE);
__IO_REG32(    DMACAC1DESTADDR,       0x40000124,__READ_WRITE);
__IO_REG32_BIT(DMACAC1LLI,            0x40000128,__READ_WRITE ,__dmacxc0lli_bits);
__IO_REG32_BIT(DMACAC1CONTROL,        0x4000012C,__READ_WRITE ,__dmacxc0control_bits);
__IO_REG32_BIT(DMACAC1CONFIGURATION,  0x40000130,__READ_WRITE ,__dmacxc0configuration_bits);

/***************************************************************************
 **
 ** DMAC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACBINTSTATUS,        0x40001000,__READ       ,__dmacxintstatus_bits);
__IO_REG32_BIT(DMACBINTTCSTATUS,      0x40001004,__READ       ,__dmacxinttcstatus_bits);
__IO_REG32_BIT(DMACBINTTCCLEAR,       0x40001008,__WRITE      ,__dmacxinttcclear_bits);
__IO_REG32_BIT(DMACBINTERRORSTATUS,   0x4000100C,__READ       ,__dmacxinterrorstatus_bits);
__IO_REG32_BIT(DMACBINTERRCLR,        0x40001010,__WRITE      ,__dmacxinterrclr_bits);
__IO_REG32_BIT(DMACBRAWINTTCSTATUS,   0x40001014,__READ       ,__dmacxrawinttcstatus_bits);
__IO_REG32_BIT(DMACBRAWINTERRORSTATUS, 0x40001018,__READ       ,__dmacxrawinterrorstatus_bits);
__IO_REG32_BIT(DMACBENBLDCHNS,        0x4000101C,__READ       ,__dmacxenbldchns_bits);
__IO_REG32_BIT(DMACBSOFTBREQ,         0x40001020,__READ_WRITE ,__dmacxsoftbreq_bits);
__IO_REG32_BIT(DMACBSOFTSREQ,         0x40001024,__READ_WRITE ,__dmacbsoftsreq_bits);
__IO_REG32_BIT(DMACBCONFIGURATION,    0x40001030,__READ_WRITE ,__dmacxconfiguration_bits);
__IO_REG32(    DMACBC0SRCADDR,        0x40001100,__READ_WRITE);
__IO_REG32(    DMACBC0DESTADDR,       0x40001104,__READ_WRITE);
__IO_REG32_BIT(DMACBC0LLI,            0x40001108,__READ_WRITE ,__dmacxc0lli_bits);
__IO_REG32_BIT(DMACBC0CONTROL,        0x4000110C,__READ_WRITE ,__dmacxc0control_bits);
__IO_REG32_BIT(DMACBC0CONFIGURATION,  0x40001110,__READ_WRITE ,__dmacxc0configuration_bits);
__IO_REG32(    DMACBC1SRCADDR,        0x40001120,__READ_WRITE);
__IO_REG32(    DMACBC1DESTADDR,       0x40001124,__READ_WRITE);
__IO_REG32_BIT(DMACBC1LLI,            0x40001128,__READ_WRITE ,__dmacxc0lli_bits);
__IO_REG32_BIT(DMACBC1CONTROL,        0x4000112C,__READ_WRITE ,__dmacxc0control_bits);
__IO_REG32_BIT(DMACBC1CONFIGURATION,  0x40001130,__READ_WRITE ,__dmacxc0configuration_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG32_BIT(PADATA,               0x400C0000,__READ_WRITE ,__pa_bits);
__IO_REG32_BIT(PACR,                 0x400C0004,__READ_WRITE ,__pacr_bits);
__IO_REG32_BIT(PAFR1,                0x400C0008,__READ_WRITE ,__pafr1_bits);
__IO_REG32_BIT(PAOD,                 0x400C0028,__READ_WRITE ,__paod_bits);
__IO_REG32_BIT(PAPUP,                0x400C002C,__READ_WRITE ,__papup_bits);
__IO_REG32_BIT(PAIE,                 0x400C0038,__READ_WRITE ,__paie_bits);

/***************************************************************************
 **
 ** PORTB
 **
 ***************************************************************************/
__IO_REG32_BIT(PBDATA,               0x400C0100,__READ_WRITE ,__pb_bits);
__IO_REG32_BIT(PBCR,                 0x400C0104,__READ_WRITE ,__pbcr_bits);
__IO_REG32_BIT(PBFR1,                0x400C0108,__READ_WRITE ,__pbfr1_bits);
__IO_REG32_BIT(PBFR2,                0x400C010C,__READ_WRITE ,__pbfr2_bits);
__IO_REG32_BIT(PBOD,                 0x400C0128,__READ_WRITE ,__pbod_bits);
__IO_REG32_BIT(PBPUP,                0x400C012C,__READ_WRITE ,__pbpup_bits);
__IO_REG32_BIT(PBIE,                 0x400C0138,__READ_WRITE ,__pbie_bits);

/***************************************************************************
 **
 ** PORTC
 **
 ***************************************************************************/
__IO_REG32_BIT(PCDATA,               0x400C0200,__READ_WRITE ,__pc_bits);
__IO_REG32_BIT(PCCR,                 0x400C0204,__READ_WRITE ,__pccr_bits);
__IO_REG32_BIT(PCFR1,                0x400C0208,__READ_WRITE ,__pcfr1_bits);
__IO_REG32_BIT(PCFR2,                0x400C020C,__READ_WRITE ,__pcfr2_bits);
__IO_REG32_BIT(PCFR3,                0x400C0210,__READ_WRITE ,__pcfr3_bits);
__IO_REG32_BIT(PCFR4,                0x400C0214,__READ_WRITE ,__pcfr4_bits);
__IO_REG32_BIT(PCOD,                 0x400C0228,__READ_WRITE ,__pcod_bits);
__IO_REG32_BIT(PCPUP,                0x400C022C,__READ_WRITE ,__pcpup_bits);
__IO_REG32_BIT(PCIE,                 0x400C0238,__READ_WRITE ,__pcie_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG32_BIT(PDDATA,               0x400C0300,__READ_WRITE ,__pd_bits);
__IO_REG32_BIT(PDCR,                 0x400C0304,__READ_WRITE ,__pdcr_bits);
__IO_REG32_BIT(PDFR1,                0x400C0308,__READ_WRITE ,__pdfr1_bits);
__IO_REG32_BIT(PDFR2,                0x400C030C,__READ_WRITE ,__pdfr2_bits);
__IO_REG32_BIT(PDFR3,                0x400C0310,__READ_WRITE ,__pdfr3_bits);
__IO_REG32_BIT(PDOD,                 0x400C0328,__READ_WRITE ,__pdod_bits);
__IO_REG32_BIT(PDPUP,                0x400C032C,__READ_WRITE ,__pdpup_bits);
__IO_REG32_BIT(PDIE,                 0x400C0338,__READ_WRITE ,__pdie_bits);

/***************************************************************************
 **
 ** PORTE
 **
 ***************************************************************************/
__IO_REG32_BIT(PEDATA,               0x400C0400,__READ_WRITE ,__pe_bits);
__IO_REG32_BIT(PECR,                 0x400C0404,__READ_WRITE ,__pecr_bits);
__IO_REG32_BIT(PEFR1,                0x400C0408,__READ_WRITE ,__pefr1_bits);
__IO_REG32_BIT(PEFR2,                0x400C040C,__READ_WRITE ,__pefr2_bits);
__IO_REG32_BIT(PEFR3,                0x400C0410,__READ_WRITE ,__pefr3_bits);
__IO_REG32_BIT(PEFR4,                0x400C0414,__READ_WRITE ,__pefr4_bits);
__IO_REG32_BIT(PEOD,                 0x400C0428,__READ_WRITE ,__peod_bits);
__IO_REG32_BIT(PEPUP,                0x400C042C,__READ_WRITE ,__pepup_bits);
__IO_REG32_BIT(PEIE,                 0x400C0438,__READ_WRITE ,__peie_bits);

/***************************************************************************
 **
 ** PORTF
 **
 ***************************************************************************/
__IO_REG32_BIT(PFDATA,               0x400C0500,__READ_WRITE ,__pf_bits);
__IO_REG32_BIT(PFCR,                 0x400C0504,__READ_WRITE ,__pfcr_bits);
__IO_REG32_BIT(PFFR1,                0x400C0508,__READ_WRITE ,__pffr1_bits);
__IO_REG32_BIT(PFFR2,                0x400C050C,__READ_WRITE ,__pffr2_bits);
__IO_REG32_BIT(PFFR3,                0x400C0510,__READ_WRITE ,__pffr3_bits);
__IO_REG32_BIT(PFOD,                 0x400C0528,__READ_WRITE ,__pfod_bits);
__IO_REG32_BIT(PFPUP,                0x400C052C,__READ_WRITE ,__pfpup_bits);
__IO_REG32_BIT(PFIE,                 0x400C0538,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG32_BIT(PGDATA,               0x400C0600,__READ_WRITE ,__pg_bits);
__IO_REG32_BIT(PGCR,                 0x400C0604,__READ_WRITE ,__pgcr_bits);
__IO_REG32_BIT(PGFR2,                0x400C060C,__READ_WRITE ,__pgfr2_bits);
__IO_REG32_BIT(PGFR3,                0x400C0610,__READ_WRITE ,__pgfr3_bits);
__IO_REG32_BIT(PGFR4,                0x400C0614,__READ_WRITE ,__pgfr4_bits);
__IO_REG32_BIT(PGOD,                 0x400C0628,__READ_WRITE ,__pgod_bits);
__IO_REG32_BIT(PGPUP,                0x400C062C,__READ_WRITE ,__pgpup_bits);
__IO_REG32_BIT(PGIE,                 0x400C0638,__READ_WRITE ,__pgie_bits);

/***************************************************************************
 **
 ** PORTH
 **
 ***************************************************************************/
__IO_REG32_BIT(PHDATA,               0x400C0700,__READ_WRITE ,__ph_bits);
__IO_REG32_BIT(PHCR,                 0x400C0704,__READ_WRITE ,__phcr_bits);
__IO_REG32_BIT(PHFR1,                0x400C0708,__READ_WRITE ,__phfr1_bits);
__IO_REG32_BIT(PHFR2,                0x400C070C,__READ_WRITE ,__phfr2_bits);
__IO_REG32_BIT(PHFR3,                0x400C0710,__READ_WRITE ,__phfr3_bits);
__IO_REG32_BIT(PHFR4,                0x400C0714,__READ_WRITE ,__phfr4_bits);
__IO_REG32_BIT(PHOD,                 0x400C0728,__READ_WRITE ,__phod_bits);
__IO_REG32_BIT(PHPUP,                0x400C072C,__READ_WRITE ,__phpup_bits);
__IO_REG32_BIT(PHIE,                 0x400C0738,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTI
 **
 ***************************************************************************/
__IO_REG32_BIT(PIDATA,               0x400C0800,__READ_WRITE ,__pi_bits);
__IO_REG32_BIT(PICR,                 0x400C0804,__READ_WRITE ,__picr_bits);
__IO_REG32_BIT(PIFR1,                0x400C0808,__READ_WRITE ,__pifr1_bits);
__IO_REG32_BIT(PIOD,                 0x400C0828,__READ_WRITE ,__piod_bits);
__IO_REG32_BIT(PIPUP,                0x400C082C,__READ_WRITE ,__pipup_bits);
__IO_REG32_BIT(PIPDN,                0x400C0830,__READ_WRITE ,__pipdn_bits);
__IO_REG32_BIT(PIIE,                 0x400C0838,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG32_BIT(PJDATA,               0x400C0900,__READ_WRITE ,__pj_bits);
__IO_REG32_BIT(PJCR,                 0x400C0904,__READ_WRITE ,__pjcr_bits);
__IO_REG32_BIT(PJFR2,                0x400C090C,__READ_WRITE ,__pjfr2_bits);
__IO_REG32_BIT(PJFR3,                0x400C0910,__READ_WRITE ,__pjfr3_bits);
__IO_REG32_BIT(PJPUP,                0x400C092C,__READ_WRITE ,__pjpup_bits);
__IO_REG32_BIT(PJIE,                 0x400C0938,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG32_BIT(PKDATA,               0x400C0A00,__READ_WRITE ,__pk_bits);
__IO_REG32_BIT(PKCR,                 0x400C0A04,__READ_WRITE ,__pkcr_bits);
__IO_REG32_BIT(PKFR2,                0x400C0A0C,__READ_WRITE ,__pkfr2_bits);
__IO_REG32_BIT(PKFR3,                0x400C0A10,__READ_WRITE ,__pkfr3_bits);
__IO_REG32_BIT(PKPUP,                0x400C0A2C,__READ_WRITE ,__pkpup_bits);
__IO_REG32_BIT(PKIE,                 0x400C0A38,__READ_WRITE ,__pkie_bits);

/***************************************************************************
 **
 ** TMRB 0
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
 ** TMRB 1
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
 ** TMRB 2
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
 ** TMRB 3
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
 ** TMRB 4
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
 ** TMRB 5
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
 ** TMRB 6
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
 ** TMRB 7
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
 ** TMRB 8
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
 ** TMRB 9
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
 ** PHCNT 0
 **
 ***************************************************************************/
__IO_REG32_BIT(PHC0RUN,             0x400CA000, __READ_WRITE , __phcxrun_bits);
__IO_REG32_BIT(PHC0CR,              0x400CA004, __READ_WRITE , __phcxcr_bits);
__IO_REG32_BIT(PHC0EN,              0x400CA008, __READ_WRITE , __phcxen_bits);
__IO_REG32_BIT(PHC0FLG,             0x400CA00C, __READ_WRITE , __phcxflg_bits);
__IO_REG32_BIT(PHC0CMP0,            0x400CA010, __READ_WRITE , __phcxcmp0_bits);
__IO_REG32_BIT(PHC0CMP1,            0x400CA014, __READ_WRITE , __phcxcmp1_bits);
__IO_REG32_BIT(PHC0CNT,             0x400CA018, __READ       , __phcxcnt_bits);
__IO_REG32_BIT(PHC0DMA,             0x400CA01C, __READ_WRITE , __phcxdma_bits);

/***************************************************************************
 **
 ** PHCNT 1
 **
 ***************************************************************************/
__IO_REG32_BIT(PHC1RUN,             0x400CA100, __READ_WRITE , __phcxrun_bits);
__IO_REG32_BIT(PHC1CR,              0x400CA104, __READ_WRITE , __phcxcr_bits);
__IO_REG32_BIT(PHC1EN,              0x400CA108, __READ_WRITE , __phcxen_bits);
__IO_REG32_BIT(PHC1FLG,             0x400CA10C, __READ_WRITE , __phcxflg_bits);
__IO_REG32_BIT(PHC1CMP0,            0x400CA110, __READ_WRITE , __phcxcmp0_bits);
__IO_REG32_BIT(PHC1CMP1,            0x400CA114, __READ_WRITE , __phcxcmp1_bits);
__IO_REG32_BIT(PHC1CNT,             0x400CA118, __READ       , __phcxcnt_bits);
__IO_REG32_BIT(PHC1DMA,             0x400CA11C, __READ_WRITE , __phcxdma_bits);

/***************************************************************************
 **
 ** PHCNT 2
 **
 ***************************************************************************/
__IO_REG32_BIT(PHC2RUN,             0x400CA200, __READ_WRITE , __phcxrun_bits);
__IO_REG32_BIT(PHC2CR,              0x400CA204, __READ_WRITE , __phcxcr_bits);
__IO_REG32_BIT(PHC2EN,              0x400CA208, __READ_WRITE , __phcxen_bits);
__IO_REG32_BIT(PHC2FLG,             0x400CA20C, __READ_WRITE , __phcxflg_bits);
__IO_REG32_BIT(PHC2CMP0,            0x400CA210, __READ_WRITE , __phcxcmp0_bits);
__IO_REG32_BIT(PHC2CMP1,            0x400CA214, __READ_WRITE , __phcxcmp1_bits);
__IO_REG32_BIT(PHC2CNT,             0x400CA218, __READ       , __phcxcnt_bits);
__IO_REG32_BIT(PHC2DMA,             0x400CA21C, __READ_WRITE , __phcxdma_bits);

/***************************************************************************
 **
 ** PHCNT 3
 **
 ***************************************************************************/
__IO_REG32_BIT(PHC3RUN,             0x400CA300, __READ_WRITE , __phcxrun_bits);
__IO_REG32_BIT(PHC3CR,              0x400CA304, __READ_WRITE , __phcxcr_bits);
__IO_REG32_BIT(PHC3EN,              0x400CA308, __READ_WRITE , __phcxen_bits);
__IO_REG32_BIT(PHC3FLG,             0x400CA30C, __READ_WRITE , __phcxflg_bits);
__IO_REG32_BIT(PHC3CMP0,            0x400CA310, __READ_WRITE , __phcxcmp0_bits);
__IO_REG32_BIT(PHC3CMP1,            0x400CA314, __READ_WRITE , __phcxcmp1_bits);
__IO_REG32_BIT(PHC3CNT,             0x400CA318, __READ       , __phcxcnt_bits);
__IO_REG32_BIT(PHC3DMA,             0x400CA31C, __READ_WRITE , __phcxdma_bits);

/***************************************************************************
 **
 ** SBI 0
 **
 ***************************************************************************/
__IO_REG32_BIT(SBI0CR0,             0x400E0000, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(SBI0CR1,             0x400E0004, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(SBI0DBR,             0x400E0008, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(SBI0I2CAR,           0x400E000C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(SBI0CR2,             0x400E0010, __READ_WRITE , __sbixcr2_bits);
#define SBI0SR      SBI0CR2
#define SBI0SR_bit  SBI0CR2_bit.__sbixsr
__IO_REG32_BIT(SBI0BR0,             0x400E0014, __READ_WRITE , __sbixbr0_bits);

/***************************************************************************
 **
 ** SBI 1
 **
 ***************************************************************************/
__IO_REG32_BIT(SBI1CR0,             0x400E0100, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(SBI1CR1,             0x400E0104, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(SBI1DBR,             0x400E0108, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(SBI1I2CAR,           0x400E010C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(SBI1CR2,             0x400E0110, __READ_WRITE , __sbixcr2_bits);
#define SBI1SR      SBI1CR2
#define SBI1SR_bit  SBI1CR2_bit.__sbixsr
__IO_REG32_BIT(SBI1BR0,             0x400E0114, __READ_WRITE , __sbixbr0_bits);

/***************************************************************************
 **
 ** SIO 0
 **
 ***************************************************************************/
__IO_REG32_BIT(SC0EN,               0x400E1000, __READ_WRITE , __scxen_bits);
__IO_REG32(    SC0BUF,              0x400E1004, __READ_WRITE );
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
 ** SIO 1
 **
 ***************************************************************************/
__IO_REG32_BIT(SC1EN,               0x400E1100, __READ_WRITE , __scxen_bits);
__IO_REG32(    SC1BUF,              0x400E1104, __READ_WRITE );
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
 ** SIO 2
 **
 ***************************************************************************/
__IO_REG32_BIT(SC2EN,               0x400E1200, __READ_WRITE , __scxen_bits);
__IO_REG32(    SC2BUF,              0x400E1204, __READ_WRITE );
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
__IO_REG32_BIT(SC2DMA,              0x400E1234, __READ_WRITE , __scxdma_bits);

/***************************************************************************
 **
 ** SIO 3
 **
 ***************************************************************************/
__IO_REG32_BIT(SC3EN,               0x400E1300, __READ_WRITE , __scxen_bits);
__IO_REG32(    SC3BUF,              0x400E1304, __READ_WRITE );
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
 ** SIO 4
 **
 ***************************************************************************/
__IO_REG32_BIT(SC4EN,               0x400E1400, __READ_WRITE , __scxen_bits);
__IO_REG32(    SC4BUF,              0x400E1404, __READ_WRITE );
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
__IO_REG32_BIT(SC4DMA,              0x400E1434, __READ_WRITE , __scxdma_bits);

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
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADCLK,                0x40050000,__READ_WRITE ,__adclk_bits);
__IO_REG32_BIT(ADMOD0,               0x40050004,__READ_WRITE ,__admod0_bits);
__IO_REG32_BIT(ADMOD1,               0x40050008,__READ_WRITE ,__admod1_bits);
__IO_REG32_BIT(ADMOD2,               0x4005000C,__READ_WRITE ,__admod2_bits);
__IO_REG32_BIT(ADMOD3,               0x40050010,__READ_WRITE ,__admod3_bits);
__IO_REG32_BIT(ADMOD4,               0x40050014,__READ_WRITE ,__admod4_bits);
__IO_REG32_BIT(ADMOD5,               0x40050018,__READ_WRITE ,__admod5_bits);
__IO_REG32_BIT(ADMOD6,               0x4005001C,__READ_WRITE ,__admod6_bits);
__IO_REG32_BIT(ADMOD7,               0x40050020,__READ_WRITE ,__admod7_bits);
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
__IO_REG32_BIT(ADREG12,              0x40050064,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG13,              0x40050068,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREG14,              0x4005006C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADREGSP,              0x40050074,__READ       ,__adregsp_bits);

/***************************************************************************
 **
 ** DAC 0
 **
 ***************************************************************************/
__IO_REG32_BIT(DA0CTL,              0x40054000, __READ_WRITE ,__daxctl_bits);
__IO_REG32_BIT(DA0REG,              0x40054004, __READ_WRITE ,__daxreg_bits);

/***************************************************************************
 **
 ** DAC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(DA1CTL,              0x40055000, __READ_WRITE ,__daxctl_bits);
__IO_REG32_BIT(DA1REG,              0x40055004, __READ_WRITE ,__daxreg_bits);

/***************************************************************************
 **
 ** TMRD 0
 **
 ***************************************************************************/
__IO_REG32_BIT(TD0RUN,              0x40058000, __WRITE      ,__tdxrun_bits);
__IO_REG32_BIT(TD0CR,               0x40058004, __READ_WRITE ,__tdxcr_bits);
__IO_REG32_BIT(TD0MOD,              0x40058008, __READ_WRITE ,__tdxmod_bits);
__IO_REG32_BIT(TD0BCR,              0x4005800C, __READ_WRITE ,__tdxbcr0_bits);
__IO_REG32_BIT(TD0DMA,              0x40058010, __READ_WRITE ,__tdxdma_bits);
__IO_REG32_BIT(TD0RG0,              0x40058014, __READ_WRITE ,__tdxrg0_bits);
__IO_REG32_BIT(TD0RG1,              0x40058018, __READ_WRITE ,__tdxrg1_bits);
__IO_REG32_BIT(TD0RG2,              0x4005801C, __READ_WRITE ,__tdxrg2_bits);
__IO_REG32_BIT(TD0RG3,              0x40058020, __READ_WRITE ,__tdxrg3_bits);
__IO_REG32_BIT(TD0RG4,              0x40058024, __READ_WRITE ,__tdxrg4_bits);
__IO_REG32_BIT(TD0RG5,              0x40058028, __READ_WRITE ,__tdxrg5_bits);
__IO_REG32_BIT(TD0CP0,              0x4005802C, __READ       ,__tdxcp0_bits);
__IO_REG32_BIT(TD0CP1,              0x40058030, __READ       ,__tdxcp1_bits);
__IO_REG32_BIT(TD0CP2,              0x40058034, __READ       ,__tdxcp2_bits);
__IO_REG32_BIT(TD0CP3,              0x40058038, __READ       ,__tdxcp3_bits);
__IO_REG32_BIT(TD0CP4,              0x4005803C, __READ       ,__tdxcp4_bits);
__IO_REG32_BIT(TD0CP5,              0x40058040, __READ       ,__tdxcp5_bits);
__IO_REG32_BIT(TD0EN,               0x40058050, __READ_WRITE ,__tdxen_bits);
__IO_REG32_BIT(TD0CONF,             0x40058054, __READ_WRITE ,__tdxconf_bits);

/***************************************************************************
 **
 ** TMRD 1
 **
 ***************************************************************************/
__IO_REG32_BIT(TD1RUN,              0x40058100, __WRITE      ,__tdxrun_bits);
__IO_REG32_BIT(TD1CR,               0x40058104, __READ_WRITE ,__tdxcr_bits);
__IO_REG32_BIT(TD1MOD,              0x40058108, __READ_WRITE ,__tdxmod_bits);
__IO_REG32_BIT(TD1BCR,              0x4005810C, __WRITE      ,__tdxbcr1_bits);
__IO_REG32_BIT(TD1DMA,              0x40058110, __READ_WRITE ,__tdxdma_bits);
__IO_REG32_BIT(TD1RG0,              0x40058114, __READ_WRITE ,__tdxrg0_bits);
__IO_REG32_BIT(TD1RG1,              0x40058118, __READ_WRITE ,__tdxrg1_bits);
__IO_REG32_BIT(TD1RG2,              0x4005811C, __READ_WRITE ,__tdxrg2_bits);
__IO_REG32_BIT(TD1RG3,              0x40058120, __READ_WRITE ,__tdxrg3_bits);
__IO_REG32_BIT(TD1RG4,              0x40058124, __READ_WRITE ,__tdxrg4_bits);
__IO_REG32_BIT(TD1CP0,              0x4005812C, __READ       ,__tdxcp0_bits);
__IO_REG32_BIT(TD1CP1,              0x40058130, __READ       ,__tdxcp1_bits);
__IO_REG32_BIT(TD1CP2,              0x40058134, __READ       ,__tdxcp2_bits);
__IO_REG32_BIT(TD1CP3,              0x40058138, __READ       ,__tdxcp3_bits);
__IO_REG32_BIT(TD1CP4,              0x4005813C, __READ       ,__tdxcp4_bits);

/***************************************************************************
 **
 ** EBIF
 **
 ***************************************************************************/
__IO_REG32_BIT(EXBMOD,              0x4005C000, __READ_WRITE ,__exbmod_bits);
__IO_REG32_BIT(EXBAS0,              0x4005C010, __READ_WRITE ,__exbas_bits);
__IO_REG32_BIT(EXBAS1,              0x4005C014, __READ_WRITE ,__exbas_bits);
__IO_REG32_BIT(EXBCS0,              0x4005C040, __READ_WRITE ,__exbcs_bits);
__IO_REG32_BIT(EXBCS1,              0x4005C044, __READ_WRITE ,__exbcs_bits);

/***************************************************************************
 **
 ** OFD
 **
 ***************************************************************************/
__IO_REG32_BIT(OFDCR1,              0x400F1000, __READ_WRITE ,__ofdcr1_bits);
__IO_REG32_BIT(OFDCR2,              0x400F1004, __READ_WRITE ,__ofdcr2_bits);
__IO_REG32_BIT(OFDMN,               0x400F1008, __READ_WRITE ,__ofdmn_bits);
__IO_REG32_BIT(OFDMX,               0x400F1010, __READ_WRITE ,__ofdmx_bits);
__IO_REG32_BIT(OFDRST,              0x400F1018, __READ_WRITE ,__ofdrst_bits);
__IO_REG32_BIT(OFDSTAT,             0x400F101C, __READ       ,__ofdstat_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,               0x400F2000,__READ_WRITE ,__wdmod_bits);
__IO_REG32(    WDCR,                0x400F2004,__WRITE);

/***************************************************************************
 **
 ** CG
 **
 ***************************************************************************/
__IO_REG32_BIT(CGSYSCR,             0x400F3000,__READ_WRITE ,__cgsyscr_bits);
__IO_REG32_BIT(CGOSCCR,             0x400F3004,__READ_WRITE ,__cgosccr_bits);
__IO_REG32_BIT(CGSTBYCR,            0x400F3008,__READ_WRITE ,__cgstbycr_bits);
__IO_REG32_BIT(CGPLLSEL,            0x400F300C,__READ_WRITE ,__cgpllsel_bits);
__IO_REG32(    CGCKSEL,             0x400F3010,__READ_WRITE);
__IO_REG32_BIT(CGPWMGEAR,           0x400F3014,__READ_WRITE ,__cgpwngear_bits);
__IO_REG32_BIT(CGPROTECT,           0x400F303C,__READ_WRITE ,__cgprotect_bits);
__IO_REG32_BIT(CGIMCGA,             0x400F3040,__READ_WRITE ,__cgimcga_bits);
__IO_REG32_BIT(CGIMCGB,             0x400F3044,__READ_WRITE ,__cgimcgb_bits);
__IO_REG32_BIT(CGIMCGC,             0x400F3048,__READ_WRITE ,__cgimcgc_bits);
__IO_REG32_BIT(CGIMCGD,             0x400F304C,__READ_WRITE ,__cgimcgd_bits);
__IO_REG32_BIT(CGIMCGE,             0x400F3050,__READ_WRITE ,__cgimcge_bits);
__IO_REG32_BIT(CGIMCGF,             0x400F3054,__READ_WRITE ,__cgimcgf_bits);
__IO_REG32_BIT(CGICRCG,             0x400F3060,__READ_WRITE ,__cgicrcg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x400F3064,__READ_WRITE ,__cgrstflg_bits);
__IO_REG32_BIT(CGNMIFLG,            0x400F3068,__READ_WRITE ,__cgnmiflg_bits);

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
 **  TMPM341FDXBG Interrupt Lines
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
#define INT_RX2              (12 + EII)   /* Serial reception (channel.2) */
#define INT_TX2              (13 + EII)   /* Serial transmit (channel.2)  */
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
#define INT_TD0CMP0          (30 + EII)   /* 16bit TMRD0 match detection 0 */
#define INT_TD0CMP1          (31 + EII)   /* 16bit TMRD0 match detection 1 */
#define INT_TD0CMP2          (32 + EII)   /* 16bit TMRD0 match detection 2 */
#define INT_TD0CMP3          (33 + EII)   /* 16bit TMRD0 match detection 3 */
#define INT_TD0CMP4          (34 + EII)   /* 16bit TMRD0 match detection 4 */
#define INT_TD1CMP0          (35 + EII)   /* 16bit TMRD1 match detection 0 */
#define INT_TD1CMP1          (36 + EII)   /* 16bit TMRD1 match detection 1 */
#define INT_TD1CMP2          (37 + EII)   /* 16bit TMRD1 match detection 2 */
#define INT_TD1CMP3          (38 + EII)   /* 16bit TMRD1 match detection 3 */
#define INT_TD1CMP4          (39 + EII)   /* 16bit TMRD1 match detection 4 */
#define INT_PHT00            (40 + EII)   /* Pulse input counter0 interrupt 0 */
#define INT_PHT01            (41 + EII)   /* Pulse input counter0 interrupt 1 */
#define INT_PHT10            (42 + EII)   /* Pulse input counter1 interrupt 0 */
#define INT_PHT11            (43 + EII)   /* Pulse input counter1 interrupt 1 */
#define INT_PHT20            (44 + EII)   /* Pulse input counter2 interrupt 0 */
#define INT_PHT21            (45 + EII)   /* Pulse input counter2 interrupt 1 */
#define INT_PHT30            (46 + EII)   /* Pulse input counter3 interrupt 0 */
#define INT_PHT31            (47 + EII)   /* Pulse input counter3 interrupt 1 */
#define INT_PHEVRY0          (48 + EII)   /* Pulse input counter interrupt 0 */
#define INT_PHEVRY1          (49 + EII)   /* Pulse input counter interrupt 1 */
#define INT_PHEVRY2          (50 + EII)   /* Pulse input counter interrupt 2 */
#define INT_PHEVRY3          (51 + EII)   /* Pulse input counter interrupt 3 */
#define INT_RX3              (52 + EII)   /* Serial reception (channel.3) */
#define INT_TX3              (53 + EII)   /* Serial transmission (channel.3)*/
#define INT_RX4              (54 + EII)   /* Serial reception (channel.4) */
#define INT_TX4              (55 + EII)   /* Serial transmission (channel.4)*/
#define INT_CAP00            (56 + EII)   /* 16bit TMRB input capture 00  */
#define INT_CAP01            (57 + EII)   /* 16bit TMRB input capture 01  */
#define INT_CAP10            (58 + EII)   /* 16bit TMRB input capture 10  */
#define INT_CAP11            (59 + EII)   /* 16bit TMRB input capture 11  */
#define INT_CAP20            (60 + EII)   /* 16bit TMRB input capture 20  */
#define INT_CAP21            (61 + EII)   /* 16bit TMRB input capture 21  */
#define INT_CAP30            (62 + EII)   /* 16bit TMRB input capture 30  */
#define INT_CAP31            (63 + EII)   /* 16bit TMRB input capture 31  */
#define INT_CAP40            (64 + EII)   /* 16bit TMRB input capture 40  */
#define INT_CAP41            (65 + EII)   /* 16bit TMRB input capture 41  */
#define INT_CAP50            (66 + EII)   /* 16bit TMRB input capture 50  */
#define INT_CAP51            (67 + EII)   /* 16bit TMRB input capture 51  */
#define INT_CAP60            (68 + EII)   /* 16bit TMRB input capture 60  */
#define INT_CAP61            (69 + EII)   /* 16bit TMRB input capture 61  */
#define INT_CAP70            (70 + EII)   /* 16bit TMRB input capture 70  */
#define INT_CAP71            (71 + EII)   /* 16bit TMRB input capture 71  */
#define INT_CAP80            (72 + EII)   /* 16bit TMRB input capture 80  */
#define INT_CAP81            (73 + EII)   /* 16bit TMRB input capture 81  */
#define INT_CAP90            (74 + EII)   /* 16bit TMRB input capture 90  */
#define INT_CAP91            (75 + EII)   /* 16bit TMRB input capture 91  */
#define INT_8                (76 + EII)   /* External Interrupt 8         */
#define INT_9                (77 + EII)   /* External Interrupt 9         */
#define INT_A                (78 + EII)   /* External Interrupt A         */
#define INT_B                (79 + EII)   /* External Interrupt B         */
#define INT_DMAC0TC          (80 + EII)   /* DMAC0 transfer complete Interrupt */
#define INT_DMAC1TC          (81 + EII)   /* DMAC1 transfer complete Interrupt */
#define INT_DMAC0ERR         (82 + EII)   /* DMAC0 transfer error Interrupt */
#define INT_DMAC1ERR         (83 + EII)   /* DMAC0 transfer error Interrupt */
#define INT_SSP              (84 + EII)   /* Synchronous serial port interrupt */

#endif    /* __IOTMPM341FDXBG_H */

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
Interrupt21  = INTRX2         0x70
Interrupt22  = INTTX2         0x74
Interrupt23  = INTSBI0        0x78
Interrupt24  = INTSBI1        0x7C
Interrupt25  = INTADHP        0x80
Interrupt26  = INTAD          0x84
Interrupt27  = INTADM0        0x88
Interrupt28  = INTADM1        0x8C
Interrupt29  = INTTB0         0x90
Interrupt30  = INTTB1         0x94
Interrupt31  = INTTB2         0x98
Interrupt32  = INTTB3         0x9C
Interrupt33  = INTTB4         0xA0
Interrupt34  = INTTB5         0xA4
Interrupt35  = INTTB6         0xA8
Interrupt36  = INTTB7         0xAC
Interrupt37  = INTTB8         0xB0
Interrupt38  = INTTB9         0xB4
Interrupt39  = INTTD0CMP0     0xB8
Interrupt40  = INTTD0CMP1     0xBC
Interrupt41  = INTTD0CMP2     0xC0
Interrupt42  = INTTD0CMP3     0xC4
Interrupt43  = INTTD0CMP4     0xC8
Interrupt44  = INTTD1CMP0     0xCC
Interrupt45  = INTTD1CMP1     0xD0
Interrupt46  = INTTD1CMP2     0xD4
Interrupt47  = INTTD1CMP3     0xD8
Interrupt48  = INTTD1CMP4     0xDC
Interrupt49  = INTPHT00       0xE0
Interrupt50  = INTPHT01       0xE4
Interrupt51  = INTPHT10       0xE8
Interrupt52  = INTPHT11       0xEC
Interrupt53  = INTPHT20       0xF0
Interrupt54  = INTPHT21       0xF4
Interrupt55  = INTPHT30       0xF8
Interrupt56  = INTPHT31       0xFC
Interrupt57  = INTPHEVRY0     0x100
Interrupt58  = INTPHEVRY1     0x104
Interrupt59  = INTPHEVRY2     0x108
Interrupt60  = INTPHEVRY3     0x10C
Interrupt61  = INTRX3         0x110
Interrupt62  = INTTX3         0x114
Interrupt63  = INTRX4         0x118
Interrupt64  = INTTX4         0x11C
Interrupt65  = INTCAP00       0x120
Interrupt66  = INTCAP01       0x124
Interrupt67  = INTCAP10       0x128
Interrupt68  = INTCAP11       0x12C
Interrupt69  = INTCAP20       0x130
Interrupt70  = INTCAP21       0x134
Interrupt71  = INTCAP30       0x138
Interrupt72  = INTCAP31       0x13C
Interrupt73  = INTCAP40       0x140
Interrupt74  = INTCAP41       0x144
Interrupt75  = INTCAP50       0x148
Interrupt76  = INTCAP51       0x14C
Interrupt77  = INTCAP60       0x150
Interrupt78  = INTCAP61       0x154
Interrupt79  = INTCAP70       0x158
Interrupt80  = INTCAP71       0x15C
Interrupt81  = INTCAP80       0x160
Interrupt82  = INTCAP81       0x164
Interrupt83  = INTCAP90       0x168
Interrupt84  = INTCAP91       0x16C
Interrupt85  = INT8           0x170
Interrupt86  = INT9           0x174
Interrupt87  = INTA           0x178
Interrupt88  = INTB           0x17C
Interrupt89  = INTDMAC0TC     0x180
Interrupt90  = INTDMAC1TC     0x184
Interrupt91  = INTDMAC0ERR    0x188
Interrupt92  = INTDMAC1ERR    0x18C
Interrupt93  = INTSSP         0x190
###DDF-INTERRUPT-END###*/
