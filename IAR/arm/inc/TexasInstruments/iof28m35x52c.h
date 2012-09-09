/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments F28M35x52C
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **    (c) Copyright IAR Systems and Texas Instruments 2012
 **
 **    $Revision: 51422 $
 **
***************************************************************************/

#ifndef __IOF28M35x52C_H
#define __IOF28M35x52C_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    F28M35x52C SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
***************************************************************************/

/* C-compiler specific declarations **********************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#if __LITTLE_ENDIAN__ != 1
#error This file should only be compiled in little endian mode
#endif

/* Reverse the bitfield order in the structs */
#pragma bitfields=disjoint_types

/* Interrupt Controller Type Register */
typedef struct {
  __REG32  INTLINESNUM    : 5;
  __REG32                 :27;
} __nvic_bits;

/* Auxiliary Control (ACTLR) Register */
typedef struct {
  __REG32  DISMCYC        : 1;
  __REG32  DISWBUF        : 1;
  __REG32  DISFOLD        : 1;
  __REG32                 :29;
} __actlr_bits;

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
  __REG32  PRI_90         : 8;
  __REG32  PRI_91         : 8;
} __pri22_bits;

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
  __REG32                 : 9;
  __REG32  TBLOFF         :20;
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

/* System Handler Priority 0 */
typedef struct {
  __REG32  PRI_MMI        : 8;
  __REG32  PRI_BFI        : 8;
  __REG32  PRI_UFI        : 8;
  __REG32                 : 8;
} __shpr0_bits;

/* System Handler Priority 1 */
typedef struct {
  __REG32                 :24;
  __REG32  PRI_SVCI       : 8;
} __shpr1_bits;

/* System Handler Priority 2 */
typedef struct {
  __REG32  PRI_DMI        : 8;
  __REG32                 : 8;
  __REG32  PRI_PSI        : 8;
  __REG32  PRI_STI        : 8;
} __shpr2_bits;

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

/*Software Trigger Interrupt Register INTID MASK*/
#define    STIR_INTID_MASK  (0x1FFUL)

/* MPU Type register */
typedef struct {
  __REG32  SEPARATE       : 1;
  __REG32                 : 7;
  __REG32  DREGION        : 8;
  __REG32  IREGION        : 8;
  __REG32                 : 8;
} __mpu_type_bits;

/* MPU Control register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  HFNMIENA       : 1;
  __REG32  PRIVDEFENA     : 1;
  __REG32                 :29;
} __mpu_ctrl_bits;

/* MPU Region Number register */
typedef struct {
  __REG32  REGION         : 8;
  __REG32                 :24;
} __mpu_rnr_bits;

/* MPU Region Base Address register */
typedef struct {
  __REG32  REGION         : 4;
  __REG32  VALID          : 1;
  __REG32  ADDR           :27;
} __mpu_rbar_bits;

/* MPU Region Attrbute and Size register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  SIZE           : 5;
  __REG32                 : 2;
  __REG32  SRD            : 8;
  __REG32  ATTR_B         : 1;
  __REG32  ATTR_C         : 1;
  __REG32  ATTR_S         : 1;
  __REG32  ATTR_TEX       : 3;
  __REG32                 : 2;
  __REG32  ATTR_AP        : 3;
  __REG32                 : 1;
  __REG32  ATTR_XN        : 1;
  __REG32                 : 3;
} __mpu_rasr_bits;

/* System PLL Configuration (SYSPLLCTL) Register */
typedef struct {
  __REG32  SPLLEN         : 1;
  __REG32  SPLLCLKEN      : 1;
  __REG32                 :30;
} __syspllctl_bits;

/* System PLL Multiplier (SYSPLLMULT) Register */
typedef struct {
  __REG32  SPLLIMULT      : 7;
  __REG32                 : 1;
  __REG32  SPLLFMULT      : 2;
  __REG32                 :22;
} __syspllmult_bits;

/* System PLL Lock Status (SYSPLLSTS) Register */
typedef struct {
  __REG32  SYSPLLLOCKS    : 1;
  __REG32  SPLLSLIPS      : 1;
  __REG32                 :30;
} __syspllsts_bits;

/* System Clock Divider (SYSDIVSEL) Register */
typedef struct {
  __REG32  SYSDIVSEL      : 2;
  __REG32                 :30;
} __sysdivsel_bits;

/* Master Subsystem Clock Divider (M3SSDIVSEL) Register */
typedef struct {
  __REG32  M3SSDIVSEL     : 2;
  __REG32                 :30;
} __m3ssdivsel_bits;

/* USB PLL Configuration (UPLLCTL) Register */
typedef struct {
  __REG32  UPLLCLKSRCSEL  : 1;
  __REG32  UPLLEN         : 1;
  __REG32  UPLLCLKEN      : 1;
  __REG32                 :29;
} __upllctl_bits;

/* USB PLL Multiplier (UPLLMULT) Register */
typedef struct {
  __REG32  UPLLIMULT      : 6;
  __REG32                 : 2;
  __REG32  UPLLFMULT      : 2;
  __REG32                 :22;
} __upllmult_bits;

/*USB PLL Lock Status (UPLLSTS) Register */
typedef struct {
  __REG32  UPLLLOCKS      : 1;
  __REG32  UPLLSLIPS      : 1;
  __REG32                 :30;
} __upllsts_bits;

/* Missing Clock Status (MCLKSTS) Register */
typedef struct {
  __REG32  REFCLKCNT      : 8;
  __REG32                 : 8;
  __REG32  MCLKFLG        : 1;
  __REG32                 :15;
} __mclksts_bits;

/* Missing Clock Force (MCLKFRCCLR) Register */
typedef struct {
  __REG32  REFCLKOFF      : 1;
  __REG32                 :15;
  __REG32  MCLKCLR        : 1;
  __REG32                 :15;
} __mclkfrcclr_bits;

/* Missing Clock Enable (MCLKEN) Register */
typedef struct {
  __REG32                 : 8;
  __REG32  MCLKNMIEN      : 1;
  __REG32                 :23;
} __mclken_bits;

/* Missing Clock Reference Limit (MCLKLIMIT) Register */
typedef struct {
  __REG32  REFCLKLOLIMIT  : 8;
  __REG32  REFCLKHILIMIT  : 8;
  __REG32                 :16;
} __mclklimit_bits;

/* XPLL CLKOUT Control (XPLLCLKCFG) Register */
typedef struct {
  __REG32  XPLLCLKOUTDIV  : 2;
  __REG32                 :30;
} __xpllclkcfg_bits;

/* Control subsystem Clock Disable (CCLKOFF) Register */
typedef struct {
  __REG32  C28CLKINDIS    : 1;
  __REG32                 :31;
} __cclkoff_bits;

/* Bit Clock Source Selection for CAN0 (CAN0BCLKSEL) Register */
/* Bit Clock Source Selection for CAN1 (CAN1BCLKSEL) Register */
typedef struct {
  __REG32  BCLKSEL        : 2;
  __REG32                 :30;
} __canbclksel_bits;

/* Subsystem Reset Configuration/Control (CRESCNF) Register */
typedef struct {
  __REG32                 :16;
  __REG32  M3RSnIN        : 1;
  __REG32  ACIBRESETn     : 1;
  __REG32                 :14;
} __crescnf_bits;

/* Control Subsystem Reset Status (CRESSTS) Register */
typedef struct {
  __REG32  CRES           : 1;
  __REG32                 :31;
} __cressts_bits;

/* Master Susystem Wait-In-Reset (MWIR) Register */
typedef struct {
  __REG32  EMU0           : 1;
  __REG32  EMU1           : 1;
  __REG32  SAMPLE         : 1;
  __REG32                 :29;
} __mwir_bits;

/* Master Subsystem Configuration (MCNF) Register */
typedef struct {
  __REG32  FLASH          : 3;
  __REG32                 : 5;
  __REG32  uCRC           : 1;
  __REG32                 :23;
} __mcnf_bits;

/* Serial Port Loop Back Control (SERPLOOP) Register */
typedef struct {
  __REG32  SSI3TOSPIA     : 2;
  __REG32                 : 6;
  __REG32  UART4TOSCIA    : 1;
  __REG32                 :23;
} __serploop_bits;

/* Master Subsystem: ACIB Status (MCIBSTATUS) Register */
typedef struct {
  __REG32  APGOODSTS      : 1;
  __REG32  READY          : 1;
  __REG32  INTS           : 1;
  __REG32                 : 5;
  __REG32  CIBBUSCLKCNT   : 8;
  __REG32                 :16;
} __mcibstatus_bits;

/* Control Subsystem Peripheral Configuration 0 (CCNF0) Register */
typedef struct {
  __REG32  HRPWM          : 1;
  __REG32                 : 3;
  __REG32  I2C            : 1;
  __REG32                 : 3;
  __REG32  SPI            : 1;
  __REG32                 : 1;
  __REG32  SCI            : 1;
  __REG32                 : 1;
  __REG32  MCBSP          : 1;
  __REG32                 :19;
} __ccnf0_bits;

/* Control Subsystem Peripheral Configuration 1 (CCNF1) Register */
typedef struct {
  __REG32  EPWM1          : 1;
  __REG32  EPWM2          : 1;
  __REG32  EPWM3          : 1;
  __REG32  EPWM4          : 1;
  __REG32  EPWM5          : 1;
  __REG32  EPWM6          : 1;
  __REG32  EPWM7          : 1;
  __REG32  EPWM8          : 1;
  __REG32  ECAP1          : 1;
  __REG32  ECAP2          : 1;
  __REG32  ECAP3          : 1;
  __REG32  ECAP4          : 1;
  __REG32  ECAP5          : 1;
  __REG32  ECAP6          : 1;
  __REG32  EQEP1          : 1;
  __REG32  EQEP2          : 1;
  __REG32                 :16;
} __ccnf1_bits;

/* Control Subsystem Peripheral Configuration 2 (CCNF2) Register */
typedef struct {
  __REG32  EPWM9          : 1;
  __REG32                 : 7;
  __REG32  EQEP3          : 1;
  __REG32                 :23;
} __ccnf2_bits;

/* Control Subsystem Peripheral Configuration 3 (CCNF3) Register */
typedef struct {
  __REG32                 :11;
  __REG32  C28DMA         : 1;
  __REG32                 :20;
} __ccnf3_bits;

/* Control Subsystem Peripheral Configuration 4 (CCNF4) Register */
typedef struct {
  __REG32  FLASH          : 3;
  __REG32                 :29;
} __ccnf4_bits;

/* Master Subsystem Memory Configuration (MEMCNF) Register */
typedef struct {
  __REG32  S0             : 1;
  __REG32  S1             : 1;
  __REG32  S2             : 1;
  __REG32  S3             : 1;
  __REG32  S4             : 1;
  __REG32  S5             : 1;
  __REG32  S6             : 1;
  __REG32  S7             : 1;
  __REG32                 :24;
} __memcnf_bits;

/* M3 Configuration Lock (MLOCK) Register */
typedef struct {
  __REG32  MSxMSELLOCK    : 1;
  __REG32                 :31;
} __mlock_bits;

/* M3NMI Configuration (MNMICFG) Register */
typedef struct {
  __REG32  NMIE           : 1;
  __REG32                 : 8;
  __REG32  ACIBERRE       : 1;
  __REG32  VREGWARNE      : 1;
  __REG32                 :21;
} __mnmicfg_bits;

/* M3NMI Flag (MNMIFLG) Register */
typedef struct {
  __REG32  NMIINT         : 1;
  __REG32  CLOCKFAIL      : 1;
  __REG32                 : 4;
  __REG32  EXTGPIO        : 1;
  __REG32  C28PIENMIERR   : 1;
  __REG32  C28NMIWDRST    : 1;
  __REG32  ACIBERR        : 1;
  __REG32  VREGWARN       : 1;
  __REG32                 :21;
} __mnmiflg_bits;

/* M3NMI Flag Clear (MNMIFLGCLR) Register */
typedef struct {
  __REG32  NMIINT         : 1;
  __REG32  CLOCKFAIL      : 1;
  __REG32                 : 4;
  __REG32  EXTGPIO        : 1;
  __REG32  C28PIENMIERR   : 1;
  __REG32  C28NMIWDRST    : 1;
  __REG32  ACIBERR        : 1;
  __REG32  VREGWARN       : 1;
  __REG32                 :21;
} __mnmiflgclr_bits;

/* M3NMI Flag Force (MNMIFLGFRC) Register */
typedef struct {
  __REG32                 : 1;
  __REG32  CLOCKFAIL      : 1;
  __REG32                 : 4;
  __REG32  EXTGPIO        : 1;
  __REG32  C28PIENMIERR   : 1;
  __REG32  C28NMIWDRST    : 1;
  __REG32  ACIBERR        : 1;
  __REG32  VREGWARN       : 1;
  __REG32                 :21;
} __mnmiflgfrc_bits;

/* M3NMI Watchdog Counter (MNMIWDCNT) Register */
typedef struct {
  __REG32  NMIWDCNT       :16;
  __REG32                 :16;
} __mnmiwdcnt_bits;

/* M3NMI Watchdog Period (MNMIWDPRD) Register */
typedef struct {
  __REG32  NMIWDPRD       :16;
  __REG32                 :16;
} __mnmiwdprd_bits;

/* Device Identification 0 (DID0) Register */
typedef struct {
  __REG32  REVID          :16;
  __REG32  CLASS          : 8;
  __REG32                 : 4;
  __REG32  VER            : 3;
  __REG32                 : 1;
} __did0_bits;

/* Device Identification 1 (DID1) Register */
typedef struct {
  __REG32  QUAL           : 2;
  __REG32  ROHS           : 1;
  __REG32  PACKAGE        : 2;
  __REG32  TEMP           : 3;
  __REG32                 : 5;
  __REG32  PINCOUNT       : 3;
  __REG32  PARTNO         : 8;
  __REG32  FAM            : 4;
  __REG32  VER            : 4;
} __did1_bits;

/* Device Configuration 1 (DC1) Register */
typedef struct {
  __REG32  JTAG           : 1;
  __REG32                 : 2;
  __REG32  WDT0           : 1;
  __REG32  PLL            : 1;
  __REG32                 :23;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __dc1_bits;

/* Device Configuration 2 (DC2) Register */
typedef struct {
  __REG32  UART0          : 1;
  __REG32  UART1          : 1;
  __REG32  UART2          : 1;
  __REG32  UART3          : 1;
  __REG32  SSI0           : 1;
  __REG32  SSI1           : 1;
  __REG32  SSI2           : 1;
  __REG32  SSI3           : 1;
  __REG32                 : 4;
  __REG32  I2C0           : 1;
  __REG32                 : 1;
  __REG32  I2C1           : 1;
  __REG32                 : 1;
  __REG32  TIMER0         : 1;
  __REG32  TIMER1         : 1;
  __REG32  TIMER2         : 1;
  __REG32  TIMER3         : 1;
  __REG32                 :10;
  __REG32  EPI            : 1;
  __REG32                 : 1;
} __dc2_bits;

/* Device Configuration 4 (DC4) Register */
typedef struct {
  __REG32  GPIOA          : 1;
  __REG32  GPIOB          : 1;
  __REG32  GPIOC          : 1;
  __REG32  GPIOD          : 1;
  __REG32  GPIOE          : 1;
  __REG32  GPIOF          : 1;
  __REG32  GPIOG          : 1;
  __REG32  GPIOH          : 1;
  __REG32  GPIOJ          : 1;
  __REG32                 : 3;
  __REG32  ROM            : 1;
  __REG32  uDMA           : 1;
  __REG32                 :10;
  __REG32  E1588          : 1;
  __REG32                 : 3;
  __REG32  EMAC0          : 1;
  __REG32                 : 3;
} __dc4_bits;

/* Device Configuration 6 (DC6) Register */
typedef struct {
  __REG32  USB0           : 2;
  __REG32                 : 2;
  __REG32  USBPHY0        : 1;
  __REG32                 :27;
} __dc6_bits;

/* Device Configuration 7 (DC7) Register */
typedef struct {
  __REG32  DMACH0         : 1;
  __REG32  DMACH1         : 1;
  __REG32  DMACH2         : 1;
  __REG32  DMACH3         : 1;
  __REG32  DMACH4         : 1;
  __REG32  DMACH5         : 1;
  __REG32  DMACH6         : 1;
  __REG32  DMACH7         : 1;
  __REG32  DMACH8         : 1;
  __REG32  DMACH9         : 1;
  __REG32  DMACH10        : 1;
  __REG32  DMACH11        : 1;
  __REG32  DMACH12        : 1;
  __REG32  DMACH13        : 1;
  __REG32  DMACH14        : 1;
  __REG32  DMACH15        : 1;
  __REG32  DMACH16        : 1;
  __REG32  DMACH17        : 1;
  __REG32  DMACH18        : 1;
  __REG32  DMACH19        : 1;
  __REG32  DMACH20        : 1;
  __REG32  DMACH21        : 1;
  __REG32  DMACH22        : 1;
  __REG32  DMACH23        : 1;
  __REG32  DMACH24        : 1;
  __REG32  DMACH25        : 1;
  __REG32  DMACH26        : 1;
  __REG32  DMACH27        : 1;
  __REG32  DMACH28        : 1;
  __REG32  DMACH29        : 1;
  __REG32  DMACH30        : 1;
  __REG32  DMACH31        : 1;
} __dc7_bits;

/* Software Reset Control 0 (SRCR0) Register */
typedef struct {
  __REG32                 : 3;
  __REG32  WDT0           : 1;
  __REG32                 :24;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __srcr0_bits;

/* Software Reset Control 1 (SRCR1) Register */
typedef struct {
  __REG32  UART0          : 1;
  __REG32  UART1          : 1;
  __REG32  UART2          : 1;
  __REG32  UART3          : 1;
  __REG32  SSI0           : 1;
  __REG32  SSI1           : 1;
  __REG32  SSI2           : 1;
  __REG32  SSI3           : 1;
  __REG32                 : 4;
  __REG32  I2C0           : 1;
  __REG32                 : 1;
  __REG32  I2C1           : 1;
  __REG32                 : 1;
  __REG32  TIMER0         : 1;
  __REG32  TIMER1         : 1;
  __REG32  TIMER2         : 1;
  __REG32  TIMER3         : 1;
  __REG32                 :10;
  __REG32  EPI            : 1;
  __REG32                 : 1;
} __srcr1_bits;

/* Software Reset Control 2 (SRCR2) Register */
typedef struct {
  __REG32  GPIOA          : 1;
  __REG32  GPIOB          : 1;
  __REG32  GPIOC          : 1;
  __REG32  GPIOD          : 1;
  __REG32  GPIOE          : 1;
  __REG32  GPIOF          : 1;
  __REG32  GPIOG          : 1;
  __REG32  GPIOH          : 1;
  __REG32  GPIOJ          : 1;
  __REG32                 : 4;
  __REG32  uDMA           : 1;
  __REG32                 : 2;
  __REG32  USB            : 1;
  __REG32                 :11;
  __REG32  EMAC0          : 1;
  __REG32                 : 3;
} __srcr2_bits;

/* Software Reset Control 3 (SRCR3) Register */
typedef struct {
  __REG32  UART4          : 1;
  __REG32                 :23;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32                 : 6;
} __srcr3_bits;

/* Master Reset Cause (MRESC) Register */
typedef struct {
  __REG32  XRSn           : 1;
  __REG32  VMON           : 1;
  __REG32                 : 1;
  __REG32  WDT0           : 1;
  __REG32  SW             : 1;
  __REG32  WDT1           : 1;
  __REG32                 :10;
  __REG32  MCLKNMI        : 1;
  __REG32                 : 7;
  __REG32  EXTGPIO        : 1;
  __REG32                 : 2;
  __REG32  C28PIENMI      : 1;
  __REG32                 : 2;
  __REG32  C28NMIWDRST    : 1;
  __REG32  ACIBERRNMI     : 1;
} __mresc_bits;

/* Run Mode Clock Configuration (RCC) Register */
typedef struct {
  __REG32                 :27;
  __REG32  ACG            : 1;
  __REG32                 : 4;
} __rcc_bits;

/* Master GPIO High Performance Bus Control (GPIOHBCTL) Register */
typedef struct {
  __REG32  PORTA          : 1;
  __REG32  PORTB          : 1;
  __REG32  PORTC          : 1;
  __REG32  PORTD          : 1;
  __REG32  PORTE          : 1;
  __REG32  PORTF          : 1;
  __REG32  PORTG          : 1;
  __REG32  PORTH          : 1;
  __REG32  PORTJ          : 1;
  __REG32                 :23;
} __gpiohbctl_bits;

/* Run Mode Clock Gating Control Register 0 (RCGC0) */
typedef struct {
  __REG32                 : 3;
  __REG32  WDT0           : 1;
  __REG32                 :24;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __rcgc0_bits;

/* Run Mode Clock Gating Control Register 1 (RCGC1) */
typedef struct {
  __REG32  UART0          : 1;
  __REG32  UART1          : 1;
  __REG32  UART2          : 1;
  __REG32  UART3          : 1;
  __REG32  SSI0           : 1;
  __REG32  SSI1           : 1;
  __REG32  SSI2           : 1;
  __REG32  SSI3           : 1;
  __REG32                 : 4;
  __REG32  I2C0           : 1;
  __REG32                 : 1;
  __REG32  I2C1           : 1;
  __REG32                 : 1;
  __REG32  TIMER0         : 1;
  __REG32  TIMER1         : 1;
  __REG32  TIMER2         : 1;
  __REG32  TIMER3         : 1;
  __REG32                 :10;
  __REG32  EPI            : 1;
  __REG32                 : 1;
} __rcgc1_bits;

/* Run Mode Clock Gating Control Register 2 (RCGC2) */
typedef struct {
  __REG32  GPIOA          : 1;
  __REG32  GPIOB          : 1;
  __REG32  GPIOC          : 1;
  __REG32  GPIOD          : 1;
  __REG32  GPIOE          : 1;
  __REG32  GPIOF          : 1;
  __REG32  GPIOG          : 1;
  __REG32  GPIOH          : 1;
  __REG32  GPIOJ          : 1;
  __REG32                 : 4;
  __REG32  uDMA           : 1;
  __REG32                 : 2;
  __REG32  USB            : 1;
  __REG32                 :11;
  __REG32  EMAC0          : 1;
  __REG32                 : 3;
} __rcgc2_bits;

/* Run Mode Clock Gating Control Register 3 (RCGC3) */
typedef struct {
  __REG32  UART4          : 1;
  __REG32                 :23;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32                 : 6;
} __rcgc3_bits;

/* Deep Sleep Clock Configuration (DSLPCLKCFG) Register */
typedef struct {
  __REG32                 : 4;
  __REG32  DSOSCSRC       : 3;
  __REG32                 :16;
  __REG32  DSDIVOVRIDE    : 4;
  __REG32                 : 5;
} __dslpclkcfg_bits;

/* Device Configuration 10 (DC10) Register */
typedef struct {
  __REG32  UART4          : 1;
  __REG32                 :23;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32                 : 6;
} __dc10_bits;

/* Z1_CSMCR Register */
/* Z2_CSMCR Register */
typedef struct {
  __REG32                 : 5;
  __REG32  CSM_ALLZERO    : 1;
  __REG32  ECSL_ALLZERO   : 1;
  __REG32  CSM_ALLONE     : 1;
  __REG32  ECSL_ALLONE    : 1;
  __REG32  CSM_MATCH      : 1;
  __REG32  ECSL_MATCH     : 1;
  __REG32  CSM_ARMED      : 1;
  __REG32  ECSL_ARMED     : 1;
  __REG32                 : 2;
  __REG32  FORCESEC       : 1;
  __REG32                 :16;
} __z_csmcr_bits;

/* Z1_GRABSECTR Register */
/* Z2_GRABSECTR Register */
typedef struct {
  __REG32  GRABSECTM      : 2;
  __REG32  GRABSECTL      : 2;
  __REG32  GRABSECTK      : 2;
  __REG32  GRABSECTJ      : 2;
  __REG32  GRABSECTI      : 2;
  __REG32  GRABSECTH      : 2;
  __REG32  GRABSECTG      : 2;
  __REG32  GRABSECTF      : 2;
  __REG32  GRABSECTE      : 2;
  __REG32  GRABSECTD      : 2;
  __REG32  GRABSECTC      : 2;
  __REG32  GRABSECTB      : 2;
  __REG32                 : 8;
} __z_grabsectr_bits;

/* Z1_GRABRAMR Register */
/* Z2_GRABRAMR Register */
typedef struct {
  __REG32  GRABRAM_C0     : 2;
  __REG32  GRABRAM_C1     : 2;
  __REG32                 :28;
} __z_grabramr_bits;

/* Z1_EXEONLYR Register */
/* Z2_EXEONLYR Register */
typedef struct {
  __REG32  EXEONLY_SECTN  : 1;
  __REG32  EXEONLY_SECTM  : 1;
  __REG32  EXEONLY_SECTL  : 1;
  __REG32  EXEONLY_SECTK  : 1;
  __REG32  EXEONLY_SECTJ  : 1;
  __REG32  EXEONLY_SECTI  : 1;
  __REG32  EXEONLY_SECTH  : 1;
  __REG32  EXEONLY_SECTG  : 1;
  __REG32  EXEONLY_SECTF  : 1;
  __REG32  EXEONLY_SECTE  : 1;
  __REG32  EXEONLY_SECTD  : 1;
  __REG32  EXEONLY_SECTC  : 1;
  __REG32  EXEONLY_SECTB  : 1;
  __REG32                 :19;
} __z_exeonlyr_bits;

/* OTPSECLOCK Register */
typedef struct {
  __REG32  JTAGLOCK       : 4;
  __REG32  C28xPSWDLOCK   : 4;
  __REG32  M3Z1PSWDLOCK   : 4;
  __REG32  M3Z2PSWDLOCK   : 4;
  __REG32  VCUCLOCK       : 4;
  __REG32  uCRCCRCLOCK    : 4;
  __REG32                 : 8;
} __otpseclock_bits;

/* uCRCCONFIG Register */
typedef struct {
  __REG32  CRCTYPE        : 2;
  __REG32                 :30;
} __ucrcconfig_bits;

/* uCRCCONTROL Register */
typedef struct {
  __REG32  CLEAR          : 1;
  __REG32  SOFTRESET      : 1;
  __REG32                 :30;
} __ucrccontrol_bits;

/* C28 to M3 Core IPC Acknowledge (CTOMIPCACK) Register bit definitions */
#define CTOMIPCACK_IPC1            (0x1UL<<0)
#define CTOMIPCACK_IPC2            (0x1UL<<1)
#define CTOMIPCACK_IPC3            (0x1UL<<2)
#define CTOMIPCACK_IPC4            (0x1UL<<3)
#define CTOMIPCACK_IPC5            (0x1UL<<4)
#define CTOMIPCACK_IPC6            (0x1UL<<5)
#define CTOMIPCACK_IPC7            (0x1UL<<6)
#define CTOMIPCACK_IPC8            (0x1UL<<7)
#define CTOMIPCACK_IPC9            (0x1UL<<8)
#define CTOMIPCACK_IPC10           (0x1UL<<9)
#define CTOMIPCACK_IPC11           (0x1UL<<10)
#define CTOMIPCACK_IPC12           (0x1UL<<11)
#define CTOMIPCACK_IPC13           (0x1UL<<12)
#define CTOMIPCACK_IPC14           (0x1UL<<13)
#define CTOMIPCACK_IPC15           (0x1UL<<14)
#define CTOMIPCACK_IPC16           (0x1UL<<15)
#define CTOMIPCACK_IPC17           (0x1UL<<16)
#define CTOMIPCACK_IPC18           (0x1UL<<17)
#define CTOMIPCACK_IPC19           (0x1UL<<18)
#define CTOMIPCACK_IPC20           (0x1UL<<19)
#define CTOMIPCACK_IPC21           (0x1UL<<20)
#define CTOMIPCACK_IPC22           (0x1UL<<21)
#define CTOMIPCACK_IPC23           (0x1UL<<22)
#define CTOMIPCACK_IPC24           (0x1UL<<23)
#define CTOMIPCACK_IPC25           (0x1UL<<24)
#define CTOMIPCACK_IPC26           (0x1UL<<25)
#define CTOMIPCACK_IPC27           (0x1UL<<26)
#define CTOMIPCACK_IPC28           (0x1UL<<27)
#define CTOMIPCACK_IPC29           (0x1UL<<28)
#define CTOMIPCACK_IPC30           (0x1UL<<29)
#define CTOMIPCACK_IPC31           (0x1UL<<30)
#define CTOMIPCACK_IPC32           (0x1UL<<31)

/* C28 to M3 Core IPC Status (CTOMIPCSTS) Register */
typedef struct {
  __REG32  IPC1           : 1;
  __REG32  IPC2           : 1;
  __REG32  IPC3           : 1;
  __REG32  IPC4           : 1;
  __REG32  IPC5           : 1;
  __REG32  IPC6           : 1;
  __REG32  IPC7           : 1;
  __REG32  IPC8           : 1;
  __REG32  IPC9           : 1;
  __REG32  IPC10          : 1;
  __REG32  IPC11          : 1;
  __REG32  IPC12          : 1;
  __REG32  IPC13          : 1;
  __REG32  IPC14          : 1;
  __REG32  IPC15          : 1;
  __REG32  IPC16          : 1;
  __REG32  IPC17          : 1;
  __REG32  IPC18          : 1;
  __REG32  IPC19          : 1;
  __REG32  IPC20          : 1;
  __REG32  IPC21          : 1;
  __REG32  IPC22          : 1;
  __REG32  IPC23          : 1;
  __REG32  IPC24          : 1;
  __REG32  IPC25          : 1;
  __REG32  IPC26          : 1;
  __REG32  IPC27          : 1;
  __REG32  IPC28          : 1;
  __REG32  IPC29          : 1;
  __REG32  IPC30          : 1;
  __REG32  IPC31          : 1;
  __REG32  IPC32          : 1;
} __ctomipcsts_bits;

/* M3 to C28 IPC Set (MTOCIPCSET) Register bit definitions */
#define MTOCIPCSET_IPC1            (0x1UL<<0)
#define MTOCIPCSET_IPC2            (0x1UL<<1)
#define MTOCIPCSET_IPC3            (0x1UL<<2)
#define MTOCIPCSET_IPC4            (0x1UL<<3)
#define MTOCIPCSET_IPC5            (0x1UL<<4)
#define MTOCIPCSET_IPC6            (0x1UL<<5)
#define MTOCIPCSET_IPC7            (0x1UL<<6)
#define MTOCIPCSET_IPC8            (0x1UL<<7)
#define MTOCIPCSET_IPC9            (0x1UL<<8)
#define MTOCIPCSET_IPC10           (0x1UL<<9)
#define MTOCIPCSET_IPC11           (0x1UL<<10)
#define MTOCIPCSET_IPC12           (0x1UL<<11)
#define MTOCIPCSET_IPC13           (0x1UL<<12)
#define MTOCIPCSET_IPC14           (0x1UL<<13)
#define MTOCIPCSET_IPC15           (0x1UL<<14)
#define MTOCIPCSET_IPC16           (0x1UL<<15)
#define MTOCIPCSET_IPC17           (0x1UL<<16)
#define MTOCIPCSET_IPC18           (0x1UL<<17)
#define MTOCIPCSET_IPC19           (0x1UL<<18)
#define MTOCIPCSET_IPC20           (0x1UL<<19)
#define MTOCIPCSET_IPC21           (0x1UL<<20)
#define MTOCIPCSET_IPC22           (0x1UL<<21)
#define MTOCIPCSET_IPC23           (0x1UL<<22)
#define MTOCIPCSET_IPC24           (0x1UL<<23)
#define MTOCIPCSET_IPC25           (0x1UL<<24)
#define MTOCIPCSET_IPC26           (0x1UL<<25)
#define MTOCIPCSET_IPC27           (0x1UL<<26)
#define MTOCIPCSET_IPC28           (0x1UL<<27)
#define MTOCIPCSET_IPC29           (0x1UL<<28)
#define MTOCIPCSET_IPC30           (0x1UL<<29)
#define MTOCIPCSET_IPC31           (0x1UL<<30)
#define MTOCIPCSET_IPC32           (0x1UL<<31)

/* M3 to C28 IPC Clear (MTOCIPCCLR) Register bit definitions */
#define MTOCIPCCLR_IPC1            (0x1UL<<0)
#define MTOCIPCCLR_IPC2            (0x1UL<<1)
#define MTOCIPCCLR_IPC3            (0x1UL<<2)
#define MTOCIPCCLR_IPC4            (0x1UL<<3)
#define MTOCIPCCLR_IPC5            (0x1UL<<4)
#define MTOCIPCCLR_IPC6            (0x1UL<<5)
#define MTOCIPCCLR_IPC7            (0x1UL<<6)
#define MTOCIPCCLR_IPC8            (0x1UL<<7)
#define MTOCIPCCLR_IPC9            (0x1UL<<8)
#define MTOCIPCCLR_IPC10           (0x1UL<<9)
#define MTOCIPCCLR_IPC11           (0x1UL<<10)
#define MTOCIPCCLR_IPC12           (0x1UL<<11)
#define MTOCIPCCLR_IPC13           (0x1UL<<12)
#define MTOCIPCCLR_IPC14           (0x1UL<<13)
#define MTOCIPCCLR_IPC15           (0x1UL<<14)
#define MTOCIPCCLR_IPC16           (0x1UL<<15)
#define MTOCIPCCLR_IPC17           (0x1UL<<16)
#define MTOCIPCCLR_IPC18           (0x1UL<<17)
#define MTOCIPCCLR_IPC19           (0x1UL<<18)
#define MTOCIPCCLR_IPC20           (0x1UL<<19)
#define MTOCIPCCLR_IPC21           (0x1UL<<20)
#define MTOCIPCCLR_IPC22           (0x1UL<<21)
#define MTOCIPCCLR_IPC23           (0x1UL<<22)
#define MTOCIPCCLR_IPC24           (0x1UL<<23)
#define MTOCIPCCLR_IPC25           (0x1UL<<24)
#define MTOCIPCCLR_IPC26           (0x1UL<<25)
#define MTOCIPCCLR_IPC27           (0x1UL<<26)
#define MTOCIPCCLR_IPC28           (0x1UL<<27)
#define MTOCIPCCLR_IPC29           (0x1UL<<28)
#define MTOCIPCCLR_IPC30           (0x1UL<<29)
#define MTOCIPCCLR_IPC31           (0x1UL<<30)
#define MTOCIPCCLR_IPC32           (0x1UL<<31)

/* M3 to C28 Core Flag (MTOCIPCFLG) Register */
typedef struct {
  __REG32  IPC1           : 1;
  __REG32  IPC2           : 1;
  __REG32  IPC3           : 1;
  __REG32  IPC4           : 1;
  __REG32  IPC5           : 1;
  __REG32  IPC6           : 1;
  __REG32  IPC7           : 1;
  __REG32  IPC8           : 1;
  __REG32  IPC9           : 1;
  __REG32  IPC10          : 1;
  __REG32  IPC11          : 1;
  __REG32  IPC12          : 1;
  __REG32  IPC13          : 1;
  __REG32  IPC14          : 1;
  __REG32  IPC15          : 1;
  __REG32  IPC16          : 1;
  __REG32  IPC17          : 1;
  __REG32  IPC18          : 1;
  __REG32  IPC19          : 1;
  __REG32  IPC20          : 1;
  __REG32  IPC21          : 1;
  __REG32  IPC22          : 1;
  __REG32  IPC23          : 1;
  __REG32  IPC24          : 1;
  __REG32  IPC25          : 1;
  __REG32  IPC26          : 1;
  __REG32  IPC27          : 1;
  __REG32  IPC28          : 1;
  __REG32  IPC29          : 1;
  __REG32  IPC30          : 1;
  __REG32  IPC31          : 1;
  __REG32  IPC32          : 1;
} __mtocipcflg_bits;

/* M3 Flash Semaphore (MPUMPREQUEST) Register */
/* M3 Clock Semaphore (MCLKREQUEST) Register */
typedef struct {
  __REG32  SEM            : 2;
  __REG32                 : 2;
  __REG32  KEY            :28;
} __mpumprequest_bits;

/* GPTM Configuration (GPTMCFG) */
typedef struct {
  __REG32  GPTMCFG        : 3;
  __REG32                 :29;
} __gptmcfg_bits;

/* GPTM TimerA Mode (GPTMTAMR) */
typedef struct {
  __REG32  TAMR           : 2;
  __REG32  TACMR          : 1;
  __REG32  TAAMS          : 1;
  __REG32  TACDIR         : 1;
  __REG32  TAMIE          : 1;
  __REG32  TAWOT          : 1;
  __REG32  TASNAPS        : 1;
  __REG32                 :24;
} __gptmtamr_bits;

/* GPTM TimerB Mode (GPTMTBMR) */
typedef struct {
  __REG32  TBMR           : 2;
  __REG32  TBCMR          : 1;
  __REG32  TBAMS          : 1;
  __REG32  TBCDIR         : 1;
  __REG32  TBMIE          : 1;
  __REG32  TBWOT          : 1;
  __REG32  TBSNAPS        : 1;
  __REG32                 :24;
} __gptmtbmr_bits;

/* GPTM Control (GPTMCTL) */
typedef struct {
  __REG32  TAEN           : 1;
  __REG32  TASTALL        : 1;
  __REG32  TAEVENT        : 2;
  __REG32  RTCEN          : 1;
  __REG32                 : 1;
  __REG32  TAPWML         : 1;
  __REG32                 : 1;
  __REG32  TBEN           : 1;
  __REG32  TBSTALL        : 1;
  __REG32  TBEVENT        : 2;
  __REG32                 : 2;
  __REG32  TBPWML         : 1;
  __REG32                 :17;
} __gptmctl_bits;

/* GPTM Interrupt Mask (GPTMIMR) */
typedef struct {
  __REG32  TATOIM         : 1;
  __REG32  CAMIM          : 1;
  __REG32  CAEIM          : 1;
  __REG32  RTCIM          : 1;
  __REG32  TAMIM          : 1;
  __REG32                 : 3;
  __REG32  TBTOIM         : 1;
  __REG32  CBMIM          : 1;
  __REG32  CBEIM          : 1;
  __REG32  TBMIM          : 1;
  __REG32                 :20;
} __gptmimr_bits;

/* GPTM Raw Interrupt Status (GPTMRIS) */
typedef struct {
  __REG32  TATORIS        : 1;
  __REG32  CAMRIS         : 1;
  __REG32  CAERIS         : 1;
  __REG32  RTCRIS         : 1;
  __REG32  TAMRIS         : 1;
  __REG32                 : 3;
  __REG32  TBTORIS        : 1;
  __REG32  CBMRIS         : 1;
  __REG32  CBERIS         : 1;
  __REG32  TBMRIS         : 1;
  __REG32                 :20;
} __gptmris_bits;

/* GPTM Masked Interrupt Status (GPTMMIS) */
typedef struct {
  __REG32  TATOMIS        : 1;
  __REG32  CAMMIS         : 1;
  __REG32  CAEMIS         : 1;
  __REG32  RTCMIS         : 1;
  __REG32  TAMMIS         : 1;
  __REG32                 : 3;
  __REG32  TBTOMIS        : 1;
  __REG32  CBMMIS         : 1;
  __REG32  CBEMIS         : 1;
  __REG32  TBMMIS         : 1;
  __REG32                 :20;
} __gptmmis_bits;

/* GPTM Interrupt Clear (GPTMICR) */
typedef struct {
  __REG32  TATOCINT       : 1;
  __REG32  CAMCINT        : 1;
  __REG32  CAECINT        : 1;
  __REG32  RTCCINT        : 1;
  __REG32  TAMCINT        : 1;
  __REG32                 : 3;
  __REG32  TBTOCINT       : 1;
  __REG32  CBMCINT        : 1;
  __REG32  CBECINT        : 1;
  __REG32  TBMCINT        : 1;
  __REG32                 :20;
} __gptmicr_bits;

#define GPTMICR_TATOCINT    (0x1UL<<0)
#define GPTMICR_CAMCINT     (0x1UL<<1)
#define GPTMICR_CAECINT     (0x1UL<<2)
#define GPTMICR_RTCCINT     (0x1UL<<3)
#define GPTMICR_TAMCINT     (0x1UL<<4)
#define GPTMICR_TBTOCINT    (0x1UL<<8)
#define GPTMICR_CBMCINT     (0x1UL<<9)
#define GPTMICR_CBECINT     (0x1UL<<10)
#define GPTMICR_TBMCINT     (0x1UL<<11)

/* GPTM TimerA Interval Load (GPTMTAILR) */
typedef struct {
  __REG32  TAILRL         :16;
  __REG32  TAILRH         :16;
} __gptmtailr_bits;

/* GPTM TimerB Interval Load (GPTMTBILR) */
typedef struct {
  __REG32  TBILRL         :16;
  __REG32                 :16;
} __gptmtbilr_bits;

/* GPTM TimerA Match (GPTMTAMATCHR) */
typedef struct {
  __REG32  TAMRL          :16;
  __REG32  TAMRH          :16;
} __gptmtamatchr_bits;

/* GPTM TimerB Match (GPTMTBMATCHR) */
typedef struct {
  __REG32  TBMRL          :16;
  __REG32                 :16;
} __gptmtbmatchr_bits;

/* GPTM Timer A Prescale (GPTMTAPR) */
typedef struct {
  __REG32  TAPSR          : 8;
  __REG32                 :24;
} __gptmtapr_bits;

/* GPTM Timer B Prescale (GPTMTBPR) */
typedef struct {
  __REG32  TBPSR          : 8;
  __REG32                 :24;
} __gptmtbpr_bits;

/* GPTM TimerA Prescale Match (GPTMTAPMR) */
typedef struct {
  __REG32  TAPSMR         : 8;
  __REG32                 :24;
} __gptmtapmr_bits;

/* GPTM TimerB Prescale Match (GPTMTBPMR) */
typedef struct {
  __REG32  TBPSMR         : 8;
  __REG32                 :24;
} __gptmtbpmr_bits;

/* GPTM TimerA (GPTMTAR) */
typedef struct {
  __REG32  TARL           :16;
  __REG32  TARH           :16;
} __gptmtar_bits;

/* GPTM TimerB (GPTMTBR) */
typedef struct {
  __REG32  TBRL           :16;
  __REG32  TBRH           : 8;
  __REG32                 : 8;
} __gptmtbr_bits;

/* GPTM Timer A Value (GPTMTAV) */
typedef struct {
  __REG32  TAVL           :16;
  __REG32  TAVH           :16;
} __gptmtav_bits;

/* GPTM Timer B Value (GPTMTBV) */
typedef struct {
  __REG32  TBVL           :16;
  __REG32  TBVH           : 8;
  __REG32                 : 8;
} __gptmtbv_bits;

/* Watchdog 0 Control (WDTCTL) */
typedef struct {
  __REG32  INTEN          : 1;
  __REG32  RESEN          : 1;
  __REG32                 :30;
} __wdt0ctl_bits;

/* Watchdog 1 Control (WDTCTL) */
typedef struct {
  __REG32  INTEN          : 1;
  __REG32  RESEN          : 1;
  __REG32                 :29;
  __REG32  WRC            : 1;
} __wdt1ctl_bits;

/* Watchdog Raw Interrupt Status (WDTRIS) */
typedef struct {
  __REG32  WDTRIS         : 1;
  __REG32                 :31;
} __wdtris_bits;

/* Watchdog Masked Interrupt Status (WDTMIS) */
typedef struct {
  __REG32  WDTMIS         : 1;
  __REG32                 :31;
} __wdtmis_bits;

/* Watchdog Test (WDTTEST) */
typedef struct {
  __REG32                 : 8;
  __REG32  STALL          : 1;
  __REG32                 :23;
} __wdttest_bits;

/* GPIOA registers */
typedef struct {
  __REG32  PA0            : 1;
  __REG32  PA1            : 1;
  __REG32  PA2            : 1;
  __REG32  PA3            : 1;
  __REG32  PA4            : 1;
  __REG32  PA5            : 1;
  __REG32  PA6            : 1;
  __REG32  PA7            : 1;
  __REG32                 :24;
} __gpioa_bits;

#define GPIOA_PA0_MASK  (0x1UL<<0)
#define GPIOA_PA1_MASK  (0x1UL<<1)
#define GPIOA_PA2_MASK  (0x1UL<<2)
#define GPIOA_PA3_MASK  (0x1UL<<3)
#define GPIOA_PA4_MASK  (0x1UL<<4)
#define GPIOA_PA5_MASK  (0x1UL<<5)
#define GPIOA_PA6_MASK  (0x1UL<<6)
#define GPIOA_PA7_MASK  (0x1UL<<7)

/* GPIOB registers */
typedef struct {
  __REG32  PB0            : 1;
  __REG32  PB1            : 1;
  __REG32  PB2            : 1;
  __REG32  PB3            : 1;
  __REG32  PB4            : 1;
  __REG32  PB5            : 1;
  __REG32  PB6            : 1;
  __REG32  PB7            : 1;
  __REG32                 :24;
} __gpiob_bits;

#define GPIOB_PB0_MASK  (0x1UL<<0)
#define GPIOB_PB1_MASK  (0x1UL<<1)
#define GPIOB_PB2_MASK  (0x1UL<<2)
#define GPIOB_PB3_MASK  (0x1UL<<3)
#define GPIOB_PB4_MASK  (0x1UL<<4)
#define GPIOB_PB5_MASK  (0x1UL<<5)
#define GPIOB_PB6_MASK  (0x1UL<<6)
#define GPIOB_PB7_MASK  (0x1UL<<7)

/* GPIOC registers */
typedef struct {
  __REG32  PC0            : 1;
  __REG32  PC1            : 1;
  __REG32  PC2            : 1;
  __REG32  PC3            : 1;
  __REG32  PC4            : 1;
  __REG32  PC5            : 1;
  __REG32  PC6            : 1;
  __REG32  PC7            : 1;
  __REG32                 :24;
} __gpioc_bits;

#define GPIOC_PC0_MASK  (0x1UL<<0)
#define GPIOC_PC1_MASK  (0x1UL<<1)
#define GPIOC_PC2_MASK  (0x1UL<<2)
#define GPIOC_PC3_MASK  (0x1UL<<3)
#define GPIOC_PC4_MASK  (0x1UL<<4)
#define GPIOC_PC5_MASK  (0x1UL<<5)
#define GPIOC_PC6_MASK  (0x1UL<<6)
#define GPIOC_PC7_MASK  (0x1UL<<7)

/* GPIOD registers */
typedef struct {
  __REG32  PD0            : 1;
  __REG32  PD1            : 1;
  __REG32  PD2            : 1;
  __REG32  PD3            : 1;
  __REG32  PD4            : 1;
  __REG32  PD5            : 1;
  __REG32  PD6            : 1;
  __REG32  PD7            : 1;
  __REG32                 :24;
} __gpiod_bits;

#define GPIOD_PD0_MASK  (0x1UL<<0)
#define GPIOD_PD1_MASK  (0x1UL<<1)
#define GPIOD_PD2_MASK  (0x1UL<<2)
#define GPIOD_PD3_MASK  (0x1UL<<3)
#define GPIOD_PD4_MASK  (0x1UL<<4)
#define GPIOD_PD5_MASK  (0x1UL<<5)
#define GPIOD_PD6_MASK  (0x1UL<<6)
#define GPIOD_PD7_MASK  (0x1UL<<7)

/* GPIOE registers */
typedef struct {
  __REG32  PE0            : 1;
  __REG32  PE1            : 1;
  __REG32  PE2            : 1;
  __REG32  PE3            : 1;
  __REG32  PE4            : 1;
  __REG32  PE5            : 1;
  __REG32  PE6            : 1;
  __REG32  PE7            : 1;
  __REG32                 :24;
} __gpioe_bits;

#define GPIOE_PE0_MASK  (0x1UL<<0)
#define GPIOE_PE1_MASK  (0x1UL<<1)
#define GPIOE_PE2_MASK  (0x1UL<<2)
#define GPIOE_PE3_MASK  (0x1UL<<3)
#define GPIOE_PE4_MASK  (0x1UL<<4)
#define GPIOE_PE5_MASK  (0x1UL<<5)
#define GPIOE_PE6_MASK  (0x1UL<<6)
#define GPIOE_PE7_MASK  (0x1UL<<7)

/* GPIOF registers */
typedef struct {
  __REG32  PF0            : 1;
  __REG32  PF1            : 1;
  __REG32  PF2            : 1;
  __REG32  PF3            : 1;
  __REG32  PF4            : 1;
  __REG32  PF5            : 1;
  __REG32  PF6            : 1;
  __REG32  PF7            : 1;
  __REG32                 :24;
} __gpiof_bits;

#define GPIOF_PF0_MASK  (0x1UL<<0)
#define GPIOF_PF1_MASK  (0x1UL<<1)
#define GPIOF_PF2_MASK  (0x1UL<<2)
#define GPIOF_PF3_MASK  (0x1UL<<3)
#define GPIOF_PF4_MASK  (0x1UL<<4)
#define GPIOF_PF5_MASK  (0x1UL<<5)
#define GPIOF_PF6_MASK  (0x1UL<<6)
#define GPIOF_PF7_MASK  (0x1UL<<7)

/* GPIOG registers */
typedef struct {
  __REG32  PG0            : 1;
  __REG32  PG1            : 1;
  __REG32  PG2            : 1;
  __REG32  PG3            : 1;
  __REG32  PG4            : 1;
  __REG32  PG5            : 1;
  __REG32  PG6            : 1;
  __REG32  PG7            : 1;
  __REG32                 :24;
} __gpiog_bits;

#define GPIOG_PG0_MASK  (0x1UL<<0)
#define GPIOG_PG1_MASK  (0x1UL<<1)
#define GPIOG_PG2_MASK  (0x1UL<<2)
#define GPIOG_PG3_MASK  (0x1UL<<3)
#define GPIOG_PG4_MASK  (0x1UL<<4)
#define GPIOG_PG5_MASK  (0x1UL<<5)
#define GPIOG_PG6_MASK  (0x1UL<<6)
#define GPIOG_PG7_MASK  (0x1UL<<7)

/* GPIOH registers */
typedef struct {
  __REG32  PH0            : 1;
  __REG32  PH1            : 1;
  __REG32  PH2            : 1;
  __REG32  PH3            : 1;
  __REG32  PH4            : 1;
  __REG32  PH5            : 1;
  __REG32  PH6            : 1;
  __REG32  PH7            : 1;
  __REG32                 :24;
} __gpioh_bits;

#define GPIOH_PH0_MASK  (0x1UL<<0)
#define GPIOH_PH1_MASK  (0x1UL<<1)
#define GPIOH_PH2_MASK  (0x1UL<<2)
#define GPIOH_PH3_MASK  (0x1UL<<3)
#define GPIOH_PH4_MASK  (0x1UL<<4)
#define GPIOH_PH5_MASK  (0x1UL<<5)
#define GPIOH_PH6_MASK  (0x1UL<<6)
#define GPIOH_PH7_MASK  (0x1UL<<7)

/* GPIOJ registers */
typedef struct {
  __REG32  PJ0            : 1;
  __REG32  PJ1            : 1;
  __REG32  PJ2            : 1;
  __REG32  PJ3            : 1;
  __REG32  PJ4            : 1;
  __REG32  PJ5            : 1;
  __REG32  PJ6            : 1;
  __REG32  PJ7            : 1;
  __REG32                 :24;
} __gpioj_bits;

#define GPIOJ_PJ0_MASK  (0x1UL<<0)
#define GPIOJ_PJ1_MASK  (0x1UL<<1)
#define GPIOJ_PJ2_MASK  (0x1UL<<2)
#define GPIOJ_PJ3_MASK  (0x1UL<<3)
#define GPIOJ_PJ4_MASK  (0x1UL<<4)
#define GPIOJ_PJ5_MASK  (0x1UL<<5)
#define GPIOJ_PJ6_MASK  (0x1UL<<6)
#define GPIOJ_PJ7_MASK  (0x1UL<<7)

/* GPIO Port Control (GPIOPCTL) */
typedef struct {
  __REG32  PMC0           : 4;
  __REG32  PMC1           : 4;
  __REG32  PMC2           : 4;
  __REG32  PMC3           : 4;
  __REG32  PMC4           : 4;
  __REG32  PMC5           : 4;
  __REG32  PMC6           : 4;
  __REG32  PMC7           : 4;
} __gpiopctl_bits;

/* Cx DEDRAM Configuration Register 1 (CxDRCR1) */
typedef struct {
  __REG32  FETCHPROTC0    : 1;
  __REG32                 : 1;
  __REG32  CPUWRPROTC0    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTC1    : 1;
  __REG32                 : 1;
  __REG32  CPUWRPROTC1    : 1;
  __REG32                 :21;
} __cxdrcr1_bits;

/* Cx SHRAM Configuration Register 1 (CxSRCR1) */
typedef struct {
  __REG32  FETCHPROTC2    : 1;
  __REG32  DMAWRPROTC2    : 1;
  __REG32  CPUWRPROTC2    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTC3    : 1;
  __REG32  DMAWRPROTC3    : 1;
  __REG32  CPUWRPROTC3    : 1;
  __REG32                 :21;
} __cxsrcr1_bits;

/* Sx SHRAM Master Select Register (MSxMSEL) */
typedef struct {
  __REG32  S0MSEL         : 1;
  __REG32  S1MSEL         : 1;
  __REG32  S2MSEL         : 1;
  __REG32  S3MSEL         : 1;
  __REG32  S4MSEL         : 1;
  __REG32  S5MSEL         : 1;
  __REG32  S6MSEL         : 1;
  __REG32  S7MSEL         : 1;
  __REG32                 :24;
} __msxmsel_bits;

/* M3 Sx SHRAM Configuration Register 1 (MSxSRCR1) */
typedef struct {
  __REG32  FETCHPROTS0    : 1;
  __REG32  DMAWRPROTS0    : 1;
  __REG32  CPUWRPROTS0    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTS1    : 1;
  __REG32  DMAWRPROTS1    : 1;
  __REG32  CPUWRPROTS1    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTS2    : 1;
  __REG32  DMAWRPROTS2    : 1;
  __REG32  CPUWRPROTS2    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTS3    : 1;
  __REG32  DMAWRPROTS3    : 1;
  __REG32  CPUWRPROTS3    : 1;
  __REG32                 : 5;
} __msxsrcr1_bits;

/* M3 Sx SHRAM Configuration Register 2 (MSxSRCR2) */
typedef struct {
  __REG32  FETCHPROTS4    : 1;
  __REG32  DMAWRPROTS4    : 1;
  __REG32  CPUWRPROTS4    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTS5    : 1;
  __REG32  DMAWRPROTS5    : 1;
  __REG32  CPUWRPROTS5    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTS6    : 1;
  __REG32  DMAWRPROTS6    : 1;
  __REG32  CPUWRPROTS6    : 1;
  __REG32                 : 5;
  __REG32  FETCHPROTS7    : 1;
  __REG32  DMAWRPROTS7    : 1;
  __REG32  CPUWRPROTS7    : 1;
  __REG32                 : 5;
} __msxsrcr2_bits;

/* M3TOC28_MSG_RAM Configuration Register (MTOCMSGRCR) */
typedef struct {
  __REG32                 : 1;
  __REG32  DMAWRPROT      : 1;
  __REG32                 :30;
} __mtocmsgrcr_bits;

/* Cx RAM Test and Initialization Register 1 (CxRTESTINIT1) */
typedef struct {
  __REG32  RAMINITC0      : 1;
  __REG32  ECCPARTESTC0   : 1;
  __REG32  RAMINITC1      : 1;
  __REG32  ECCPARTESTC1   : 1;
  __REG32  RAMINITC2      : 1;
  __REG32  ECCPARTESTC2   : 1;
  __REG32  RAMINITC3      : 1;
  __REG32  ECCPARTESTC3   : 1;
  __REG32                 :24;
} __cxrtestinit1_bits;

/* M3 Sx RAM Test and Initialization Register 1 (MSxRTESTINIT1) */
typedef struct {
  __REG32  RAMINITS0      : 1;
  __REG32  ECCPARTESTS0   : 1;
  __REG32  RAMINITS1      : 1;
  __REG32  ECCPARTESTS1   : 1;
  __REG32  RAMINITS2      : 1;
  __REG32  ECCPARTESTS2   : 1;
  __REG32  RAMINITS3      : 1;
  __REG32  ECCPARTESTS3   : 1;
  __REG32  RAMINITS4      : 1;
  __REG32  ECCPARTESTS4   : 1;
  __REG32  RAMINITS5      : 1;
  __REG32  ECCPARTESTS5   : 1;
  __REG32  RAMINITS6      : 1;
  __REG32  ECCPARTESTS6   : 1;
  __REG32  RAMINITS7      : 1;
  __REG32  ECCPARTESTS7   : 1;
  __REG32                 :16;
} __msxrtestinit1_bits;

/* MTOC_MSG_RAM Test and Initialization Register (MTOCRTESTINIT) */
typedef struct {
  __REG32  RAMINITMTOCMSGRAM    : 1;
  __REG32  ECCPARTESTMTOCMSGRAM : 1;
  __REG32                       :30;
} __mtocrtestinit_bits;

/* Cx RAM INITDONE Register 1 (CxRINITDONE1) */
typedef struct {
  __REG32  RAMINITDONEC0  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONEC1  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONEC2  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONEC3  : 1;
  __REG32                 :25;
} __cxrinitdone1_bits;

/* M3 Sx RAM INITDONE Register 1 (MSxRINITDONE1) */
typedef struct {
  __REG32  RAMINITDONES0  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES1  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES2  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES3  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES4  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES5  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES6  : 1;
  __REG32                 : 1;
  __REG32  RAMINITDONES7  : 1;
  __REG32                 :17;
} __msxrinitdone1_bits;

/* MTOC_MSG_RAM INITDONE Register (MTOCRINITDONE) */
typedef struct {
  __REG32  RAMINITDONEMTOCMSGRAM  : 1;
  __REG32                         :31;
} __mtocrinitdone_bits;

/* M3 Uncorrectable Error Flag Register (MUEFLG) */
/* M3 Uncorrectable Error Force Register (MUEFRC) */
/* M3 Uncorrectable Error Flag Clear Register (MUECLR) */
typedef struct {
  __REG32  M3CPUWE        : 1;
  __REG32  UDMAWE         : 1;
  __REG32  M3CPURE        : 1;
  __REG32  UDMARE         : 1;
  __REG32                 :28;
} __mueflg_bits;

/* M3 Corrected Error Counter Register (MCECNTR) */
typedef struct {
  __REG32  MCECNTR        :16;
  __REG32                 :16;
} __mcecntr_bits;

/* M3 Corrected Error Threshold Register (MCETRES) */
typedef struct {
  __REG32  MCETRES        :16;
  __REG32                 :16;
} __mcetres_bits;

/* M3 Corrected Error Threshold Exceeded Flag Register (MCEFLG) */
typedef struct {
  __REG32  MCEFLG         : 1;
  __REG32                 :31;
} __mceflg_bits;

/* M3 Corrected Error Threshold Exceeded Force Register (MCEFRC) */
typedef struct {
  __REG32  MCEFRC         : 1;
  __REG32                 :31;
} __mcefrc_bits;

/* M3 Corrected Error Threshold Exceeded Flag Clear Register (MCECLR) */
typedef struct {
  __REG32  MCECLR         : 1;
  __REG32                 :31;
} __mceclr_bits;

/* M3 Single Error Interrupt Enable Register (MCEIE) */
typedef struct {
  __REG32  MCEIE          : 1;
  __REG32                 :31;
} __mceie_bits;

/* Non-Master Access Violation Flag Register (MNMAVFLG) */
/* Non-Master Access Violation Flag Clear Register (MNMAVCLR) */
typedef struct {
  __REG32  CPUFETCH       : 1;
  __REG32  DMAWRITE       : 1;
  __REG32  CPUWRITE       : 1;
  __REG32                 :29;
} __mnmavflg_bits;

/* Master Access Violation Flag Register (MMAVFLG) */
/* Master Access Violation Flag Clear Register (MMAVCLR) */
typedef struct {
  __REG32  CPUFETCH       : 1;
  __REG32  DMAWRITE       : 1;
  __REG32  CPUWRITE       : 1;
  __REG32                 :29;
} __mmavflg_bits;

/* Flash Read Control Register (FRDCNTL) */
typedef struct {
  __REG32                 : 8;
  __REG32  RWAIT          : 4;
  __REG32                 :20;
} __frdcntl_bits;

/* Flash Read Margin Control Register (FSPRD) */
typedef struct {
  __REG32  RM0            : 1;
  __REG32  RM1            : 1;
  __REG32                 :30;
} __fsprd_bits;

/* Flash Bank Access Control Register (FBAC) */
typedef struct {
  __REG32  VREADST        : 8;
  __REG32  BAGP           : 8;
  __REG32                 :16;
} __fbac_bits;

/* Flash Bank Fallback Power Register (FBFALLBACK) */
typedef struct {
  __REG32  BNKPWR         : 2;
  __REG32                 :30;
} __fbfallback_bits;

/* Flash Bank Pump Control Register 1 (FBPRDY) */
typedef struct {
  __REG32  BANKRDY        : 1;
  __REG32                 :14;
  __REG32  PUMPRDY        : 1;
  __REG32                 :16;
} __fbprdy_bits;

/* Flash Bank Pump Control Register 1 (FPAC1) */
typedef struct {
  __REG32  PMPPWR         : 1;
  __REG32                 :15;
  __REG32  PSLEEP         :11;
  __REG32                 : 5;
} __fpac1_bits;

/* Flash Bank Pump Control Register 2 (FPAC2) */
typedef struct {
  __REG32  PAGP           :16;
  __REG32                 :16;
} __fpac2_bits;

/* Flash Module Access Control Register (FMAC) */
typedef struct {
  __REG32  BANK           : 3;
  __REG32                 :29;
} __fmac_bits;

/* FMSTAT Register */
typedef struct {
  __REG32                 : 1;
  __REG32  PSUSP          : 1;
  __REG32  ESUSP          : 1;
  __REG32  VOLTSTAT       : 1;
  __REG32  CSTAT          : 1;
  __REG32  INVDAT         : 1;
  __REG32  PGM            : 1;
  __REG32  ERS            : 1;
  __REG32  Busy           : 1;
  __REG32  CV             : 1;
  __REG32  EV             : 1;
  __REG32  PCV            : 1;
  __REG32  PGV            : 1;
  __REG32  DBF            : 1;
  __REG32  ILA            : 1;
  __REG32  RVF            : 1;
  __REG32  RDVER          : 1;
  __REG32  RVSUSP         : 1;
  __REG32                 :14;
} __fmstat_bits;

/* SECZONEREQUEST Register (SEM) Register */
typedef struct {
  __REG32  SEM            : 2;
  __REG32                 : 2;
  __REG32  KEY            :28;
} __seczonerequest_bits;

/* Flash Read Interface Control Register (FRD_INTF_CTRL) */
typedef struct {
  __REG32  PROG_CACHE_EN  : 1;
  __REG32  DATA_CACHE_EN  : 1;
  __REG32                 :30;
} __frd_intf_ctrl_bits;

/* ECC Enable Register (ECC_Enable) */
typedef struct {
  __REG32  ENABLE         : 4;
  __REG32                 :28;
} __ecc_enable_bits;

/* Error Status Register (ERR_STATUS) */
typedef struct {
  __REG32  FAIL_0         : 1;
  __REG32  FAIL_1         : 1;
  __REG32  UNC_ERR        : 1;
  __REG32                 :29;
} __err_status_bits;

/* Error Position Register (ERR_POS) */
typedef struct {
  __REG32  ERR_POS        : 6;
  __REG32  ECC_L_OR_H     : 1;
  __REG32                 : 1;
  __REG32  ERR_TYPE       : 1;
  __REG32                 :23;
} __err_pos_bits;

/* Error Status Clear Register (ERR_STATUS_CLR) */
typedef struct {
  __REG32  FAIL_0_CLR     : 1;
  __REG32  FAIL_1_CLR     : 1;
  __REG32  UNC_ERR_CLR    : 1;
  __REG32                 :29;
} __err_status_clr_bits;

/* Error Counter Register (ERR_CNT) */
typedef struct {
  __REG32  ERR_CNT        :16;
  __REG32                 :16;
} __err_cnt_bits;

/* Error Threshold Register (ERR_THRESHOLD) */
typedef struct {
  __REG32  THRESHOLD      :16;
  __REG32                 :16;
} __err_threshold_bits;

/* Error Interrupt Flag Register (ERR_INTFLG) */
typedef struct {
  __REG32  SINGLE_ERR_INT_FLG : 1;
  __REG32  UNC_ERR_INT_FLG    : 1;
  __REG32                     :30;
} __err_intflg_bits;

/* Error Interrupt Flag Register (ERR_INTFLG) */
typedef struct {
  __REG32  SINGLE_ERR_INT_CLR : 1;
  __REG32  UNC_ERR_INT_CLR    : 1;
  __REG32                     :30;
} __err_intclr_bits;

/* ECC Test Address Register (FADDR_TEST) */
typedef struct {
  __REG32  ADDR           :19;
  __REG32                 :13;
} __faddr_test_bits;

/* ECC Test Register (FECC_TEST) */
typedef struct {
  __REG32  ECC            : 8;
  __REG32                 :24;
} __fecc_test_bits;

/* ECC Control Register (FECC_CTRL) */
typedef struct {
  __REG32  ECC_TEST_EN    : 1;
  __REG32  ECC_SELECT     : 1;
  __REG32                 :30;
} __fecc_ctrl_bits;

/* ECC Status Register (FECC_STATUS) */
typedef struct {
  __REG32  SINGLE_ERR     : 1;
  __REG32  UNC_ERR        : 1;
  __REG32  DATA_ERR_POS   : 6;
  __REG32  CHK_ERR_POS    : 3;
  __REG32                 :21;
} __fecc_status_bits;

/* ADC Result Registers (ADCxRESULTy) */
typedef struct {
  __REG16  RESULT         :12;
  __REG16                 : 4;
} __adcresult_bits;

/* DMA Status (DMASTAT) */
typedef struct {
  __REG32  MASTEN         : 1;
  __REG32                 : 3;
  __REG32  STATE          : 4;
  __REG32                 : 8;
  __REG32  DMACHANS       : 5;
  __REG32                 :11;
} __dmastat_bits;

/* DMA Configuration (DMACFG) */
#define DMACFG_MASTEN     (0x1UL<<0)

/* DMA Channel Wait-on-Request Status (DMAWAITSTAT) */
typedef struct {
  __REG32  WAITREQ0       : 1;
  __REG32  WAITREQ1       : 1;
  __REG32  WAITREQ2       : 1;
  __REG32  WAITREQ3       : 1;
  __REG32  WAITREQ4       : 1;
  __REG32  WAITREQ5       : 1;
  __REG32  WAITREQ6       : 1;
  __REG32  WAITREQ7       : 1;
  __REG32  WAITREQ8       : 1;
  __REG32  WAITREQ9       : 1;
  __REG32  WAITREQ10      : 1;
  __REG32  WAITREQ11      : 1;
  __REG32  WAITREQ12      : 1;
  __REG32  WAITREQ13      : 1;
  __REG32  WAITREQ14      : 1;
  __REG32  WAITREQ15      : 1;
  __REG32  WAITREQ16      : 1;
  __REG32  WAITREQ17      : 1;
  __REG32  WAITREQ18      : 1;
  __REG32  WAITREQ19      : 1;
  __REG32  WAITREQ20      : 1;
  __REG32  WAITREQ21      : 1;
  __REG32  WAITREQ22      : 1;
  __REG32  WAITREQ23      : 1;
  __REG32  WAITREQ24      : 1;
  __REG32  WAITREQ25      : 1;
  __REG32  WAITREQ26      : 1;
  __REG32  WAITREQ27      : 1;
  __REG32  WAITREQ28      : 1;
  __REG32  WAITREQ29      : 1;
  __REG32  WAITREQ30      : 1;
  __REG32  WAITREQ31      : 1;
} __dmawaitstat_bits;

/* DMA Channel Useburst Set (DMAUSEBURSTSET) 
   DMA Channel Request Mask Set (DMAREQMASKSET)
   DMA Channel Enable Set (DMAENASET)
   DMA Channel Primary Alternate Set (DMAALTSET)
   DMA Channel Priority Set (DMAPRIOSET) */
typedef struct {
  __REG32  SET0         : 1;
  __REG32  SET1         : 1;
  __REG32  SET2         : 1;
  __REG32  SET3         : 1;
  __REG32  SET4         : 1;
  __REG32  SET5         : 1;
  __REG32  SET6         : 1;
  __REG32  SET7         : 1;
  __REG32  SET8         : 1;
  __REG32  SET9         : 1;
  __REG32  SET10        : 1;
  __REG32  SET11        : 1;
  __REG32  SET12        : 1;
  __REG32  SET13        : 1;
  __REG32  SET14        : 1;
  __REG32  SET15        : 1;
  __REG32  SET16        : 1;
  __REG32  SET17        : 1;
  __REG32  SET18        : 1;
  __REG32  SET19        : 1;
  __REG32  SET20        : 1;
  __REG32  SET21        : 1;
  __REG32  SET22        : 1;
  __REG32  SET23        : 1;
  __REG32  SET24        : 1;
  __REG32  SET25        : 1;
  __REG32  SET26        : 1;
  __REG32  SET27        : 1;
  __REG32  SET28        : 1;
  __REG32  SET29        : 1;
  __REG32  SET30        : 1;
  __REG32  SET31        : 1;
} __dmauseburstset_bits;

/* DMA Bus Error Clear (DMAERRCLR) */
typedef struct {
  __REG32  ERRCLR       : 1;
  __REG32               :31;
} __dmaerrclr_bits;

/* DMA Channel Alternate Select (DMACHALT) */
typedef struct {
  __REG32  CHASGN0        : 1;
  __REG32  CHASGN1        : 1;
  __REG32  CHASGN2        : 1;
  __REG32  CHASGN3        : 1;
  __REG32  CHASGN4        : 1;
  __REG32  CHASGN5        : 1;
  __REG32  CHASGN6        : 1;
  __REG32  CHASGN7        : 1;
  __REG32  CHASGN8        : 1;
  __REG32  CHASGN9        : 1;
  __REG32  CHASGN10       : 1;
  __REG32  CHASGN11       : 1;
  __REG32  CHASGN12       : 1;
  __REG32  CHASGN13       : 1;
  __REG32  CHASGN14       : 1;
  __REG32  CHASGN15       : 1;
  __REG32  CHASGN16       : 1;
  __REG32  CHASGN17       : 1;
  __REG32  CHASGN18       : 1;
  __REG32  CHASGN19       : 1;
  __REG32  CHASGN20       : 1;
  __REG32  CHASGN21       : 1;
  __REG32  CHASGN22       : 1;
  __REG32  CHASGN23       : 1;
  __REG32  CHASGN24       : 1;
  __REG32  CHASGN25       : 1;
  __REG32  CHASGN26       : 1;
  __REG32  CHASGN27       : 1;
  __REG32  CHASGN28       : 1;
  __REG32  CHASGN29       : 1;
  __REG32  CHASGN30       : 1;
  __REG32  CHASGN31       : 1;
} __dmachalt_bits;

/* DMA Channel Map Assignment (DMACHMAP0) Register */
typedef struct {
  __REG32  CH0MAP         : 4;
  __REG32  CH1MAP         : 4;
  __REG32  CH2MAP         : 4;
  __REG32  CH3MAP         : 4;
  __REG32  CH4MAP         : 4;
  __REG32  CH5MAP         : 4;
  __REG32  CH6MAP         : 4;
  __REG32  CH7MAP         : 4;
} __dmachmap0_bits;

/* DMA Channel Map Assignment (DMACHMAP1) Register */
typedef struct {
  __REG32  CH8MAP         : 4;
  __REG32  CH9MAP         : 4;
  __REG32  CH10MAP        : 4;
  __REG32  CH11MAP        : 4;
  __REG32  CH12MAP        : 4;
  __REG32  CH13MAP        : 4;
  __REG32  CH14MAP        : 4;
  __REG32  CH15MAP        : 4;
} __dmachmap1_bits;

/* DMA Channel Map Assignment (DMACHMAP2) Register */
typedef struct {
  __REG32  CH16MAP         : 4;
  __REG32  CH17MAP         : 4;
  __REG32  CH18MAP        : 4;
  __REG32  CH19MAP        : 4;
  __REG32  CH20MAP        : 4;
  __REG32  CH21MAP        : 4;
  __REG32  CH22MAP        : 4;
  __REG32  CH23MAP        : 4;
} __dmachmap2_bits;

/* DMA Channel Map Assignment (DMACHMAP3) Register */
typedef struct {
  __REG32  CH24MAP         : 4;
  __REG32  CH25MAP         : 4;
  __REG32  CH26MAP        : 4;
  __REG32  CH27MAP        : 4;
  __REG32  CH28MAP        : 4;
  __REG32  CH29MAP        : 4;
  __REG32  CH30MAP        : 4;
  __REG32  CH31MAP        : 4;
} __dmachmap3_bits;

#define DMA_CH0_MASK      (0x1UL<<0)
#define DMA_CH1_MASK      (0x1UL<<1)
#define DMA_CH2_MASK      (0x1UL<<2)
#define DMA_CH3_MASK      (0x1UL<<3)
#define DMA_CH4_MASK      (0x1UL<<4)
#define DMA_CH5_MASK      (0x1UL<<5)
#define DMA_CH6_MASK      (0x1UL<<6)
#define DMA_CH7_MASK      (0x1UL<<7)
#define DMA_CH8_MASK      (0x1UL<<8)
#define DMA_CH9_MASK      (0x1UL<<9)
#define DMA_CH10_MASK     (0x1UL<<10)
#define DMA_CH11_MASK     (0x1UL<<11)
#define DMA_CH12_MASK     (0x1UL<<12)
#define DMA_CH13_MASK     (0x1UL<<13)
#define DMA_CH14_MASK     (0x1UL<<14)
#define DMA_CH15_MASK     (0x1UL<<15)
#define DMA_CH16_MASK     (0x1UL<<16)
#define DMA_CH17_MASK     (0x1UL<<17)
#define DMA_CH18_MASK     (0x1UL<<18)
#define DMA_CH19_MASK     (0x1UL<<19)
#define DMA_CH20_MASK     (0x1UL<<20)
#define DMA_CH21_MASK     (0x1UL<<21)
#define DMA_CH22_MASK     (0x1UL<<22)
#define DMA_CH23_MASK     (0x1UL<<23)
#define DMA_CH24_MASK     (0x1UL<<24)
#define DMA_CH25_MASK     (0x1UL<<25)
#define DMA_CH26_MASK     (0x1UL<<26)
#define DMA_CH27_MASK     (0x1UL<<27)
#define DMA_CH28_MASK     (0x1UL<<28)
#define DMA_CH29_MASK     (0x1UL<<29)
#define DMA_CH30_MASK     (0x1UL<<30)
#define DMA_CH31_MASK     (0x1UL<<31)

/* EPI Configuration (EPICFG) */
typedef struct {
  __REG32  MODE           : 4;
  __REG32  BLKEN          : 1;
  __REG32                 :27;
} __epicfg_bits;

/* EPI Main Baud Rate (EPIBAUD) */
typedef struct {
  __REG32  COUNT0         :16;
  __REG32  COUNT1         :16;
} __epibaud_bits;

/* EPI SDRAM Configuration (EPISDRAMCFG) */
typedef union{
  /* EPISDRAMCFG */
  struct {
  __REG32  SIZE           : 2;
  __REG32                 : 7;
  __REG32  SLEEP          : 1;
  __REG32                 : 6;
  __REG32  RFSH           :11;
  __REG32                 : 3;
  __REG32  FREQ           : 2;
  };
  /* EPIHB8CFG */
  struct {
  __REG32  MODE           : 2;
  __REG32                 : 2;
  __REG32  RDWS           : 2;
  __REG32  WRWS           : 2;
  __REG32  MAXWAIT        : 8;
  __REG32                 : 4;
  __REG32  RDHIGH         : 1;
  __REG32  WRHIGH         : 1;
  __REG32  XFEEN          : 1;
  __REG32  XFFEN          : 1;
  __REG32                 : 8;
  } __epihb8cfg;
    
  /* EPIHB16CFG */
  struct {
  __REG32  MODE           : 2;
  __REG32  BSEL           : 1;
  __REG32                 : 1;
  __REG32  RDWS           : 2;
  __REG32  WRWS           : 2;
  __REG32  MAXWAIT        : 8;
  __REG32                 : 4;
  __REG32  RDHIGH         : 1;
  __REG32  WRHIGH         : 1;
  __REG32  XFEEN          : 1;
  __REG32  XFFEN          : 1;
  __REG32                 : 8;
  } __epihb16cfg;
  
  /* EPIGPCFG */
  struct {
  __REG32  DSIZE          : 2;
  __REG32                 : 2;
  __REG32  ASIZE          : 2;
  __REG32                 : 2;
  __REG32  MAXWAIT        : 8;
  __REG32                 : 2;
  __REG32  RD2CYC         : 1;
  __REG32  WR2CYC         : 1;
  __REG32                 : 1;
  __REG32  RW             : 1;
  __REG32  FRMCNT         : 4;
  __REG32  FRM50          : 1;
  __REG32  FRMPIN         : 1;
  __REG32  RDYEN          : 1;
  __REG32                 : 1;
  __REG32  CLKGATE        : 1;
  __REG32  CLKPIN         : 1;
  } __epigpcfg;
} __episdramcfg_bits;

/* EPI Host-Bus 8 Configuration 2 (EPIHB8CFG2) */
typedef union {
/* EPIHB8CFG2 */
/* EPIHB16CFG2 */
  struct {
  __REG32                 :24;
  __REG32  CSCFG          : 2;
  __REG32  CSBAUD         : 1;
  __REG32                 : 4;
  __REG32  WORD           : 1;
  };
  /* EPIGPCFG2 */
  struct {
  __REG32                 :31;
  __REG32   WORD          : 1;
  } __epigpcfg2;
} __epihb8cfg2_bits;

/* EPI Address Map (EPIADDRMAP) */
typedef struct {
  __REG32  ERADR          : 2;
  __REG32  ERSZ           : 2;
  __REG32  EPADR          : 2;
  __REG32  EPSZ           : 2;
  __REG32                 :24;
} __epiaddrmap_bits;

/* EPI Read Address 0 (EPIRADDR0) 
   EPI Read Address 1 (EPIRADDR1) */
typedef struct {
  __REG32  ADDR           :29;
  __REG32                 : 3;
} __epiraddr_bits;

/* EPI Read Size 0 (EPIRSIZE0) 
   EPI Read Size 1 (EPIRSIZE1) */
typedef struct {
  __REG32  SIZE           : 2;
  __REG32                 :30;
} __epirsize_bits;

/* EPI Non-Blocking Read Data 0 (EPIRPSTD0)
   EPI Non-Blocking Read Data 1 (EPIRPSTD1) */
typedef struct {
  __REG32  POSTCNT        :13;
  __REG32                 :19;
} __epirpstd_bits;

/* EPI Status (EPISTAT) */
typedef struct {
  __REG32  ACTIVE         : 1;
  __REG32                 : 3;
  __REG32  NBRBUSY        : 1;
  __REG32  WBUSY          : 1;
  __REG32  INITSEQ        : 1;
  __REG32  XFEMPTY        : 1;
  __REG32  XFFULL         : 1;
  __REG32  CELOW          : 1;
  __REG32                 :22;
} __epistat_bits;

/* EPI Read FIFO Count (EPIRFIFOCNT) */
typedef struct {
  __REG32  COUNT          : 3;
  __REG32                 :29;
} __epirfifocnt_bits;

/* EPI FIFO Level Selects (EPIFIFOLVL) */
typedef struct {
  __REG32  RDFIFO         : 3;
  __REG32                 : 1;
  __REG32  WRFIFO         : 3;
  __REG32                 : 9;
  __REG32  RSERR          : 1;
  __REG32  WFERR          : 1;
  __REG32                 :14;
} __epififolvl_bits;

/* EPI Write FIFO Count (EPIWFIFOCNT) */
typedef struct {
  __REG32  WTAV           : 3;
  __REG32                 :29;
} __epiwfifocnt_bits;

/* EPI Interrupt Mask (EPIIM) */
typedef struct {
  __REG32  ERRIM          : 1;
  __REG32  RDIM           : 1;
  __REG32  WRIM           : 1;
  __REG32                 :29;
} __epiim_bits;

/* EPI Raw Interrupt Status (EPIRIS) */
typedef struct {
  __REG32  ERRRIS         : 1;
  __REG32  RDRIS          : 1;
  __REG32  WRRIS          : 1;
  __REG32                 :29;
} __epiris_bits;

/* EPI Masked Interrupt Status (EPIMIS) */
typedef struct {
  __REG32  ERRMIS         : 1;
  __REG32  RDMIS          : 1;
  __REG32  WRMIS          : 1;
  __REG32                 :29;
} __epimis_bits;

/* EPI Error Interrupt Status and Clear (EPIEISC) */
typedef struct {
  __REG32  TOUT           : 1;
  __REG32  RSTALL         : 1;
  __REG32  WTFULL         : 1;
  __REG32                 :29;
} __epieisc_bits;

#define EPIEISC_TOUT      (0x1UL<<0)
#define EPIEISC_RSTALL    (0x1UL<<1)
#define EPIEISC_WTFULL    (0x1UL<<2)

/* USB Device Functional Address (USBFADDR) */
typedef struct {
  __REG8   FUNCADDR       : 7;
  __REG8                  : 1;
} __usbfaddr_bits;

/* USB Power (USBPOWER) */
typedef struct {
  __REG8   PWRDNPHY       : 1;
  __REG8   SUSPEND        : 1;
  __REG8   RESUME         : 1;
  __REG8   RESET          : 1;
  __REG8                  : 2;
  __REG8   SOFTCONN       : 1;
  __REG8   ISOUPDATE      : 1;
} __usbpower_bits;

/* USB Transmit Interrupt Status (USBTXIS) */
typedef struct {
  __REG16  EP0            : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbtxis_bits;

/* USB Receive Interrupt Status (USBRXIS) */
typedef struct {
  __REG16                 : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbrxis_bits;

/* USB Transmit Interrupt Enable (USBTXIE) */
typedef struct {
  __REG16  EP0            : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbtxie_bits;

/* USB Receive Interrupt Enable (USBRXIE) */
typedef struct {
  __REG16                 : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbrxie_bits;

/* USB General Interrupt Status (USBIS) */
typedef struct {
  __REG8   SUSPEND        : 1;
  __REG8   RESUME         : 1;
  __REG8   BABBLE_RESET   : 1;
  __REG8   SOF            : 1;
  __REG8   CONN           : 1;
  __REG8   DISCON         : 1;
  __REG8   SESREQ         : 1;
  __REG8   VBUSERR        : 1;
} __usbis_bits;

/* USB Interrupt Enable (USBIE) */
typedef struct {
  __REG8   SUSPEND        : 1;
  __REG8   RESUME         : 1;
  __REG8   BABBLE_RESET   : 1;
  __REG8   SOF            : 1;
  __REG8   CONN           : 1;
  __REG8   DISCON         : 1;
  __REG8   SESREQ         : 1;
  __REG8   VBUSERR        : 1;
} __usbie_bits;

/* USB Frame Value (USBFRAME) */
typedef struct {
  __REG16  FRAME          :11;
  __REG16                 : 5;
} __usbframe_bits;

/* USB Endpoint Index (USBEPIDX) */
typedef struct {
  __REG8   EPIDX          : 4;
  __REG8                  : 4;
} __usbepidx_bits;

/* USB Test Mode (USBTEST) */
typedef struct {
  __REG8                  : 5;
  __REG8   FORCEFS        : 1;
  __REG8   FIFOACC        : 1;
  __REG8   FORCEH         : 1;
} __usbtest_bits;

/* USB Device Control (USBDEVCTL) */
typedef struct {
  __REG8   SESSION        : 1;
  __REG8   HOSTREQ        : 1;
  __REG8   HOSTMODE       : 1;
  __REG8   VBUS           : 2;
  __REG8   LSDEV          : 1;
  __REG8   FSDEV          : 1;
  __REG8   DEV            : 1;
} __usbdevctl_bits;

/* USB Transmit Dynamic FIFO Sizing (USBTXFIFOSZ)
   USB Receive Dynamic FIFO Sizing (USBRXFIFOSZ) */
typedef struct {
  __REG8   SIZE           : 4;
  __REG8   DPB            : 1;
  __REG8                  : 3;
} __usbtxfifosz_bits;

/* USB Transmit FIFO Start Address (USBTXFIFOADD)
   USB Receive FIFO Start Address (USBRXFIFOADD) */
typedef struct {
  __REG16  ADDR           : 9;
  __REG16                 : 7;
} __usbtxfifoadd_bits;

/* USB Connect Timing (USBCONTIM) */
typedef struct {
  __REG8   WTID           : 4;
  __REG8   WTCON          : 4;
} __usbcontim_bits;

/* USB Transmit Functional Address Endpoint x (USBTXFUNCADDRx) */
/* USB Receive Functional Address Endpoint x (USBRXFUNCADDRx) */
typedef struct {
  __REG8   ADDR           : 7;
  __REG8                  : 1;
} __usbtxfuncaddr_bits;

/* USB Transmit Hub Address Endpoint x (USBTXHUBADDRx) */
typedef struct {
  __REG8   ADDR           : 7;
  __REG8                  : 1;
} __usbtxhubaddr_bits;

/* USB Receive Hub Address Endpoint x (USBRXHUBADDRx) */
typedef struct {
  __REG8   ADDR           : 7;
  __REG8   MULTTRAN       : 1;
} __usbrxhubaddr_bits;

/* USB Transmit Hub Port Endpoint x (USBTXHUBPORTx) */
/* USB Receive Hub Port Endpoint x (USBRXHUBPORTx) */
typedef struct {
  __REG8   PORT           : 7;
  __REG8                  : 1;
} __usbtxhubport_bits;

/* USB Maximum Transmit Data Endpoint x (USBTXMAXPx) */
typedef struct {
  __REG16  MAXLOAD        :11;
  __REG16                 : 5;
} __usbtxmaxp_bits;

/* USB Control and Status Endpoint 0 Low (USBCSRL0) */
typedef struct {
  __REG8   RXRDY          : 1;
  __REG8   TXRDY          : 1;
  __REG8   STALLED        : 1;
  __REG8   SETUP_DATAEND  : 1;
  __REG8   ERROR_SETEND   : 1;
  __REG8   REQPKT_STALL   : 1;
  __REG8   STATUS_RXRDYC  : 1;
  __REG8   NAKTO_SETENDC  : 1;
} __usbcsrl0_bits;

/* USB Control and Status Endpoint 0 High (USBCSRH0) */
typedef struct {
  __REG8   FLUSH          : 1;
  __REG8   DT             : 1;
  __REG8   DTWE           : 1;
  __REG8                  : 5;
} __usbcsrh0_bits;

/* USB Receive Byte Count Endpoint 0 (USBCOUNT0) */
typedef struct {
  __REG8   COUNT          : 7;
  __REG8                  : 1;
} __usbcount0_bits;

/* USB Type Endpoint 0 (USBTYPE0) */
typedef struct {
  __REG8                  : 6;
  __REG8   SPEED          : 2;
} __usbtype0_bits;

/* USB NAK Limit (USBNAKLMT) */
typedef struct {
  __REG8   NAKLMT         : 5;
  __REG8                  : 3;
} __usbnaklmt_bits;

/* USB Transmit Control and Status Endpoint x Low (USBTXCSRLx) */
typedef struct {
  __REG8   TXRDY          : 1;
  __REG8   FIFONE         : 1;
  __REG8   ERROR_UNDRN    : 1;
  __REG8   FLUSH          : 1;
  __REG8   SETUP_STALL    : 1;
  __REG8   STALLED        : 1;
  __REG8   CLRDT          : 1;
  __REG8   NAKTO          : 1;
} __usbtxcsrl_bits;

/* USB Transmit Control and Status Endpoint x High (USBTXCSRHx) */
typedef struct {
  __REG8   DT             : 1;
  __REG8   DTWE           : 1;
  __REG8   DMAMOD         : 1;
  __REG8   FDT            : 1;
  __REG8   DMAEN          : 1;
  __REG8   MODE           : 1;
  __REG8   ISO            : 1;
  __REG8   AUTOSET        : 1;
} __usbtxcsrh_bits;

/* USB Maximum Receive Data Endpoint x (USBRXMAXPx) */
typedef struct {
  __REG16  MAXLOAD        :11;
  __REG16                 : 5;
} __usbrxmaxp_bits;

/* USB Receive Control and Status Endpoint x Low (USBRXCSRLx) */
typedef struct
{
  __REG8   RXRDY          : 1;
  __REG8   FULL           : 1;
  __REG8   ERROR_OVER     : 1;
  __REG8   DATAERR_NAKTO  : 1;
  __REG8   FLUSH          : 1;
  __REG8   REQPKT_STALL   : 1;
  __REG8   STALLED        : 1;
  __REG8   CLRDT          : 1;
} __usbrxcsrl_bits;

/* USB Receive Control and Status Endpoint x High (USBRXCSRHx) */
typedef struct {
  __REG8                  : 1;
  __REG8   DT             : 1;
  __REG8   DTWE           : 1;
  __REG8   DMAMOD         : 1;
  __REG8   DISNYET_PIDERR : 1;
  __REG8   DMAEN          : 1;
  __REG8   AUTORQ_ISO     : 1;
  __REG8   AUTOCL         : 1;
} __usbrxcsrh_bits;

/* USB Receive Byte Count Endpoint x (USBRXCOUNTx) */
typedef struct {
  __REG16  COUNT          :13;
  __REG16                 : 3;
} __usbrxcount_bits;

/* USB Host Transmit Configure Type Endpoint x (USBTXTYPEx) */
/* USB Host Configure Receive Type Endpoint x (USBRXTYPEx) */
typedef struct {
  __REG8   TEP            : 4;
  __REG8   PROTO          : 2;
  __REG8   SPEED          : 2;
} __usbtxtype_bits;

/* USB Request Packet Count in Block Transfer Endpoint n Registers (USBRQPKTCOUNTx) */
typedef struct {
  __REG16  COUNT          :13;
  __REG16                 : 3;
} __usbrqpktcount_bits;

/* USB Receive Double Packet Buffer Disable (USBRXDPKTBUFDIS) 
   USB Transmit Double Packet Buffer Disable (USBTXDPKTBUFDIS) */
typedef struct {
  __REG16                 : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbrxdpktbufdis_bits;

/* USB External Power Control (USBEPC) */
typedef struct {
  __REG32  EPEN           : 2;  
  __REG32  EPENDE         : 1;  
  __REG32                 : 1;  
  __REG32  PFLTEN         : 1;  
  __REG32  PFLTSEN        : 1;  
  __REG32  PFLTAEN        : 1;  
  __REG32                 : 1;  
  __REG32  PFLTACT        : 2;  
  __REG32                 :22;  
} __usbepc_bits;

/* USB External Power Control Raw Interrupt Status (USBEPCRIS) 
   USB External Power Control Interrupt Mask (USBEPCIM) 
   USB External Power Control Interrupt Status and Clear (USBEPCISC) */
typedef struct {
  __REG32  PF             : 1;  
  __REG32                 :31;  
} __usbepcris_bits;

/* USB Device RESUME Raw Interrupt Status (USBDRRIS)
   USB Device RESUME Interrupt Mask (USBDRIM)*/
typedef struct {
  __REG32  RESUME         : 1;  
  __REG32                 :31;  
} __usbdrris_bits;

/* USB General-Purpose Control and Status (USBGPCS) */
typedef struct {
  __REG32  DEVMOD         : 1;  
  __REG32  DEVMODOTG      : 1;  
  __REG32                 :30;  
} __usbgpcs_bits;

/* USB VBUS Droop Control (USBVDC) */
typedef struct {
  __REG32  VBDEN          : 1;  
  __REG32                 :31;  
} __usbvdc_bits;

/* USB VBUS Droop Control Raw Interrupt Status (USBVDCRIS)
   USB VBUS Droop Control Interrupt Mask (USBVDCIM)
   USB VBUS Droop Control Interrupt Status and Clear (USBVDCISC) */
typedef struct {
  __REG32  VD             : 1;  
  __REG32                 :31;  
} __usbvdcris_bits;

/* USB ID Valid Detect Raw Interrupt Status (USBIDVRIS)
   USB ID Valid Detect Interrupt Mask (USBIDVIM)
   USB ID Valid Detect Interrupt Status and Clear (USBIDVISC) */
typedef struct {
  __REG32  ID             : 1;  
  __REG32                 :31;  
} __usbidvris_bits;

/* USB DMA Select (USBDMASEL) */
typedef struct {
  __REG32  DMAARX         : 4;  
  __REG32  DMAATX         : 4;  
  __REG32  DMABRX         : 4;  
  __REG32  DMABTX         : 4;  
  __REG32  DMACRX         : 4;  
  __REG32  DMACTX         : 4;  
  __REG32                 : 8;  
} __usbdmasel_bits;

/* Ethernet MAC Raw Interrupt Status (MACRIS) */
/* Ethernet MAC Interrupt Acknowledge (MACIACK) */
typedef struct {
  __REG32  RXINT          : 1;
  __REG32  TXER           : 1;     
  __REG32  TXEMP          : 1; 
  __REG32  FOV            : 1;   
  __REG32  RXER           : 1; 
  __REG32  MDINT          : 1;     
  __REG32  PHYINT         : 1;     
  __REG32                 :25;    
} __macris_bits;

#define   MACIACK_RXINT   (0x1UL<<0)
#define   MACIACK_TXER    (0x1UL<<1)
#define   MACIACK_TXEMP   (0x1UL<<2)
#define   MACIACK_FOV     (0x1UL<<3)
#define   MACIACK_RXER    (0x1UL<<4)
#define   MACIACK_MDINT   (0x1UL<<5)
#define   MACIACK_PHYINT  (0x1UL<<6)

/* Ethernet MAC Interrupt Mask (MACIM) */
typedef struct {
  __REG32  RXINTM         : 1;
  __REG32  TXERM          : 1;     
  __REG32  TXEMPM         : 1; 
  __REG32  FOVM           : 1;   
  __REG32  RXERM          : 1; 
  __REG32  MDINTM         : 1;     
  __REG32  PHYINTM        : 1;     
  __REG32                 :25;   
} __macim_bits;

/* Ethernet MAC Receive Control (MACRCTL) */
typedef struct {
  __REG32  RXEN           : 1; 
  __REG32  AMUL           : 1; 
  __REG32  PRMS           : 1;   
  __REG32  BADCRC         : 1;     
  __REG32  RSTFIFO        : 1;     
  __REG32                 :27;    
} __macrctl_bits;

/* Ethernet MAC Transmit Control (MACTCTL) */
typedef struct {
  __REG32  TXEN           : 1;   
  __REG32  PADEN          : 1; 
  __REG32  CRC            : 1;   
  __REG32                 : 1; 
  __REG32  DUPLEX         : 1;   
  __REG32                 : 27;   
} __mactctl_bits;

/* Ethernet MAC Data (MACDATA) */
typedef struct {
  __REG32  DATA           :32;   
} __macdata_bits;

/* Ethernet MAC Individual Address 0 (MACIA0) */
typedef struct {
  __REG32  MACOCT1        : 8;
  __REG32  MACOCT2        : 8;
  __REG32  MACOCT3        : 8;
  __REG32  MACOCT4        : 8;   
} __macia0_bits;

/* Ethernet MAC Individual Address 1 (MACIA1) */
typedef struct {
  __REG32  MACOCT5        : 8;
  __REG32  MACOCT6        : 8;    
  __REG32                 :16;   
} __macia1_bits;

/* Ethernet MAC Threshold (MACTHR) */
typedef struct {
  __REG32  THRESH         : 6;  
  __REG32                 :26;   
} __macthr_bits;

/* Ethernet MAC Management Control (MACMCTL) */
typedef struct {
  __REG32  START          : 1; 
  __REG32  WRITE          : 1;   
  __REG32                 : 1; 
  __REG32  REGADR         : 5;   
  __REG32                 :24;   
} __macmctl_bits;

/* Ethernet MAC Management Divider (MACMDV) */
typedef struct {
  __REG32  DIV            : 8;
  __REG32                 :24;      
} __macmdv_bits;

/* Ethernet MAC Management Transmit Data (MACMTXD) */
typedef struct {
  __REG32  MDTX           :16;
  __REG32                 :16;      
} __macmtxd_bits;

/* Ethernet MAC Management Receive Data (MACMRXD) */
typedef struct {
  __REG32  MDRX           :16;   
  __REG32                 :16;   
} __macmrxd_bits;

/* Ethernet MAC Number of Packets (MACNP) */
typedef struct {
  __REG32  NPR            : 6;   
  __REG32                 :26;   
} __macnp_bits;

/* Ethernet MAC Transmission Request (MACTR) */
typedef struct {
  __REG32  NEWTX          : 1;   
  __REG32                 :31;   
} __mactr_bits;

/* Ethernet MAC Timer Support (MACTS) */
typedef struct {
  __REG32  TSEN           : 1;
  __REG32                 :31;      
} __macts_bits;

/* SSI Control 0 (SSICR0) */
typedef struct {
  __REG32  DSS            : 4;
  __REG32  FRF            : 2;
  __REG32  SPO            : 1;
  __REG32  SPH            : 1;
  __REG32  SCR            : 8;
  __REG32                 :16;
} __ssicr0_bits;

/* SSI Control 1 (SSICR1) */
typedef struct {
  __REG32  LBM            : 1;
  __REG32  SSE            : 1;
  __REG32  MS             : 1;
  __REG32  SOD            : 1;
  __REG32  EOT            : 1;
  __REG32                 :27;
} __ssicr1_bits;

/* SSI Data (SSIDR) */
typedef struct {
  __REG32  DATA           :16;
  __REG32                 :16;
} __ssidr_bits;

/* SSI Status (SSISR) */
typedef struct {
  __REG32  TFE            : 1;
  __REG32  TNF            : 1;
  __REG32  RNE            : 1;
  __REG32  RFF            : 1;
  __REG32  BSY            : 1;
  __REG32                 :27;
} __ssisr_bits;

/* SSI Clock Prescale (SSICPSR) */
typedef struct {
  __REG32  CPSDVSR        :16;
  __REG32                 :16;
} __ssicpsr_bits;

/* SSI Interrupt Mask (SSIIM) */
typedef struct {
  __REG32  RORIM          : 1;
  __REG32  RTIM           : 1;
  __REG32  RXIM           : 1;
  __REG32  TXIM           : 1;
  __REG32                 :28;
} __ssiim_bits;

/* SSI Raw Interrupt Status (SSIRIS) */
typedef struct {
  __REG32  RORRIS         : 1;
  __REG32  RTRIS          : 1;
  __REG32  RXRIS          : 1;
  __REG32  TXRIS          : 1;
  __REG32                 :28;
} __ssiris_bits;

/* SSI Masked Interrupt Status (SSIMIS) */
typedef struct {
  __REG32  RORMIS         : 1;
  __REG32  RTMIS          : 1;
  __REG32  RXMIS          : 1;
  __REG32  TXMIS          : 1;
  __REG32                 :28;
} __ssimis_bits;

/* SSI Interrup Clear (SSIICR) */
#define    SSIICR_RORIC   (0x1UL<<0)
#define    SSIICR_RTIC    (0x1UL<<01)

/* SSI DMA Control (SSIDMACTL) */
typedef struct {
  __REG32  RXDMAE         : 1;
  __REG32  TXDMAE         : 1;
  __REG32                 :30;
} __ssidmactl_bits;

/* UART Data (UARTDR) */
typedef struct {
  __REG32  DATA           : 8;
  __REG32  FE             : 1;
  __REG32  PE             : 1;
  __REG32  BE             : 1;
  __REG32  OE             : 1;
  __REG32                 :20;
} __uartdr_bits;

/* UART Receive Status/Error Clear (UARTRSR/UARTECR) */
typedef union {
  /* UARTxRSR */
  struct {
    __REG32  FE           : 1;
    __REG32  PE           : 1;
    __REG32  BE           : 1;
    __REG32  OE           : 1;
    __REG32               :28;
  };
  /* UARTxECR */
  struct {
    __REG32  DATA         : 8;
    __REG32               :24;
  };
} __uartrsr_bits;

/* UART Flag (UARTFR) */
typedef struct {
  __REG32  CTS            : 1;
  __REG32  DSR            : 1;
  __REG32  DCD            : 1;
  __REG32  BUSY           : 1;
  __REG32  RXFE           : 1;
  __REG32  TXFF           : 1;
  __REG32  RXFF           : 1;
  __REG32  TXFE           : 1;
  __REG32  RI             : 1;
  __REG32                 :23;
} __uartfr_bits;

/* UART IrDA Low-Power Register (UARTILPR) */
typedef struct {
  __REG32  ILPDVSR        : 8;
  __REG32                 :24;
} __uartilpr_bits;

/* UART Integer Baud-Rate Divisor Register (UARTIBRD) */
typedef struct {
  __REG32  DIVINT         :16;
  __REG32                 :16;
} __uartibrd_bits;

/* UART Fractional Baud-Rate Divisor (UARTFBRD) */
typedef struct {
  __REG32  DIVFRAC        : 6;
  __REG32                 :26;
} __uartfbrd_bits;

/* UART Line Control (UARTLCRH) */
typedef struct {
  __REG32  BRK            : 1;
  __REG32  PEN            : 1;
  __REG32  EPS            : 1;
  __REG32  STP2           : 1;
  __REG32  FEN            : 1;
  __REG32  WLEN           : 2;
  __REG32  SPS            : 1;
  __REG32                 :24;
} __uartlcrh_bits;

/* UART Control (UARTCTL) */
typedef struct {
  __REG32  UARTEN         : 1;
  __REG32  SIREN          : 1;
  __REG32  SIRLP          : 1;
  __REG32  SMART          : 1;
  __REG32  EOT            : 1;
  __REG32  HSE            : 1;
  __REG32  LIN            : 1;   
  __REG32  LBE            : 1;
  __REG32  TXE            : 1;
  __REG32  RXE            : 1;
  __REG32  DTR            : 1;
  __REG32  RTS            : 1;
  __REG32                 : 2;
  __REG32  RTSEN          : 1;
  __REG32  CTSEN          : 1;
  __REG32                 :16;
} __uartctl_bits;

/* UART Interrupt FIFO Level Select (UARTIFLS) */
typedef struct {
  __REG32  TXIFLSEL       : 3;
  __REG32  RXIFLSEL       : 3;
  __REG32                 :26;
} __uartifls_bits;

/* UART Interrupt Mask (UARTIM) */
typedef struct {
  __REG32  RIIM           : 1;
  __REG32  CTSIM          : 1;
  __REG32  DCDIM          : 1;
  __REG32  DSRIM          : 1;
  __REG32  RXIM           : 1;
  __REG32  TXIM           : 1;
  __REG32  RTIM           : 1;
  __REG32  FEIM           : 1;
  __REG32  PEIM           : 1;
  __REG32  BEIM           : 1;
  __REG32  OEIM           : 1;
  __REG32                 : 2;
  __REG32  LMSBIM         : 1;
  __REG32  LME1IM         : 1;
  __REG32  LME5IM         : 1;
  __REG32                 :16;
} __uartim_bits;

/* UART Raw Interrupt Status (UARTRIS) */
typedef struct {
  __REG32  RIRIS          : 1;
  __REG32  CTSRIS         : 1;
  __REG32  DCDRIS         : 1;
  __REG32  DSRRIS         : 1;
  __REG32  RXRIS          : 1;
  __REG32  TXRIS          : 1;
  __REG32  RTRIS          : 1;
  __REG32  FERIS          : 1;
  __REG32  PERIS          : 1;
  __REG32  BERIS          : 1;
  __REG32  OERIS          : 1;
  __REG32                 : 2;
  __REG32  LMSBRIS        : 1;
  __REG32  LME1RIS        : 1;
  __REG32  LME5RIS        : 1;
  __REG32                 :16;
} __uartris_bits;

/* UART Masked Interrupt Status (UARTMIS) */
typedef struct {
  __REG32  RIMIS          : 1;
  __REG32  CTSMIS         : 1;
  __REG32  DCDMIS         : 1;
  __REG32  DSRMIS         : 1;
  __REG32  RXMIS          : 1;
  __REG32  TXMIS          : 1;
  __REG32  RTMIS          : 1;
  __REG32  FEMIS          : 1;
  __REG32  PEMIS          : 1;
  __REG32  BEMIS          : 1;
  __REG32  OEMIS          : 1;
  __REG32                 : 2;
  __REG32  LMSBMIS        : 1;
  __REG32  LME1MIS        : 1;
  __REG32  LME5MIS        : 1;
  __REG32                 :16;
} __uartmis_bits;

/* UART Interrupt Clear (UARTICR) */
#define   UARTICR_RIMIC   (0x1UL<<0)
#define   UARTICR_CTSMIC  (0x1UL<<1)
#define   UARTICR_DCDMIC  (0x1UL<<2)
#define   UARTICR_DSRMIC  (0x1UL<<3)
#define   UARTICR_RXIC    (0x1UL<<4)
#define   UARTICR_TXIC    (0x1UL<<5)
#define   UARTICR_RTIC    (0x1UL<<6)
#define   UARTICR_FEIC    (0x1UL<<7)
#define   UARTICR_PEIC    (0x1UL<<8)
#define   UARTICR_BEIC    (0x1UL<<9)
#define   UARTICR_OEIC    (0x1UL<<10)
#define   UARTICR_LMSBMIC (0x1UL<<13)
#define   UARTICR_LME1MIC (0x1UL<<14)
#define   UARTICR_LME5MIC (0x1UL<<15)

/* UART DMA Control (UARTDMACTL) */
typedef struct {
  __REG32  RXDMAE         : 1;
  __REG32  TXDMAE         : 1;
  __REG32  DMAERR         : 1;
  __REG32                 :29;
} __uartdmactl_bits;

/* UART LIN Control (UARTLCTL) */
typedef struct {
  __REG32  MASTER         : 1;
  __REG32                 : 3;
  __REG32  BLEN           : 2;
  __REG32                 :26;
} __uartlctl_bits;

/* UART LIN Snap Shot (UARTLSS) */
typedef struct {
  __REG32  TSS            :16;
  __REG32                 :16;
} __uartlss_bits;

/* UART LIN Timer (UARTLTIM */
typedef struct {
  __REG32  TIMER          :16;
  __REG32                 :16;
} __uartltim_bits;

/* I2C Master Slave Address (I2CMSA) */
typedef struct {
  __REG32  R_S            : 1;
  __REG32  SA             : 7;
  __REG32                 :24;
} __i2cmsa_bits;

/* I2C Master Control/Status (I2CMCS) */
typedef struct {
  __REG32  BUSY         : 1;
  __REG32  ERROR        : 1;
  __REG32  ADRACK       : 1;
  __REG32  DATACK       : 1;
  __REG32  ARBLST       : 1;
  __REG32  IDLE         : 1;
  __REG32  BUSBSY       : 1;
  __REG32               :25;
} __i2cmcs_bits;

#define    I2CMCS_RUN   (0x1UL<<0)
#define    I2CMCS_START (0x1UL<<1)
#define    I2CMCS_STOP  (0x1UL<<2)
#define    I2CMCS_ACK   (0x1UL<<3)

/* I2C Master Data (I2CMDR) */
typedef struct {
  __REG32  DATA           : 8;
  __REG32                 :24;
} __i2cmdr_bits;

/* I2C Master Timer Period (I2CMTPR) */
typedef struct {
  __REG32  TPR            : 7;
  __REG32                 :25;
} __i2cmtpr_bits;

/* I2C Master Interrupt Mask (I2CMIMR) */
typedef struct {
  __REG32  IM             : 1;
  __REG32                 :31;
} __i2cmimr_bits;

/* I2C Master Raw Interrupt Status (I2CMRIS) */
typedef struct {
  __REG32  RIS            : 1;
  __REG32                 :31;
} __i2cmris_bits;

/* I2C Master Masked Interrupt Status (I2CMMIS) */
typedef struct {
  __REG32  MIS            : 1;
  __REG32                 :31;
} __i2cmmis_bits;

/* I2C Master Interrupt Clear (I2CMICR) */
#define   I2CMICR_IC      (0x1UL<<0)

/* I2C Master Configuration (I2CMCR) */
typedef struct {
  __REG32  LPBK           : 1;
  __REG32                 : 3;
  __REG32  MFE            : 1;
  __REG32  SFE            : 1;
  __REG32                 :26;
} __i2cmcr_bits;

/* I2C Slave Own Address (I2CSOAR) */
typedef struct {
  __REG32  OAR            : 7;
  __REG32                 :25;
} __i2csoar_bits;

/* I2C Slave Control/Status (I2CSCSR) */
typedef struct {
    __REG32  RREQ         : 1;
    __REG32  TREQ         : 1;
    __REG32  FBR          : 1;
    __REG32               :29;
} __i2cscsr_bits;

#define      I2CSCSR_DA   (0x1UL<<0)

/* I2C Slave Data (I2CSDR) Register */
typedef struct {
    __REG32  DATA         : 8;
    __REG32               :24;
} __i2csdr_bits;

/* I2C Slave Interrupt Mask (I2CSIMR) */
typedef struct {
  __REG32  DATAIM         : 1;
  __REG32  STARTIM        : 1;
  __REG32  STOPIM         : 1;
  __REG32                 :29;
} __i2csimr_bits;

/* I2C Slave Raw Interrupt Status (I2CSRIS) */
typedef struct {
  __REG32  DATARIS        : 1;
  __REG32  STARTRIS       : 1;
  __REG32  STOPRIS        : 1;
  __REG32                 :29;
} __i2csris_bits;

/* I2C Slave Masked Interrupt Status (I2CSMIS) */
typedef struct {
  __REG32  DATAMIS        : 1;
  __REG32  STARTMIS       : 1;
  __REG32  STOPMIS        : 1;
  __REG32                 :29;
} __i2csmis_bits;

/* I2C Slave Interrupt Clear (I2CSICR) */
#define I2CSICR_DATAIC         (0x1UL<<0)
#define I2CSICR_STARTIC        (0x1UL<<1)
#define I2CSICR_STOPIC         (0x1UL<<2)

/* CAN Control Register (CANxCTL) */
typedef struct {
  __REG32 Init                : 1;
  __REG32 IE0                 : 1;
  __REG32 SIE                 : 1;
  __REG32 EIE                 : 1;
  __REG32                     : 1;
  __REG32 DAR                 : 1;
  __REG32 CCE                 : 1;
  __REG32 Test                : 1;
  __REG32 IDS                 : 1;
  __REG32 ABO                 : 1;
  __REG32 PMD                 : 4;
  __REG32                     : 1;
  __REG32 SWR                 : 1;
  __REG32 InitDbg             : 1;
  __REG32 IE1                 : 1;
  __REG32 DE1                 : 1;
  __REG32 DE2                 : 1;
  __REG32 DE3                 : 1;
  __REG32                     :11;
} __canctl_bits;

/* CAN Error and Status Register (CANxES) */
typedef struct {
  __REG32 LEC                 : 3;
  __REG32 TxOK                : 1;
  __REG32 RxOK                : 1;
  __REG32 EPass               : 1;
  __REG32 EWarn               : 1;
  __REG32 BOff                : 1;
  __REG32 PER                 : 1;
  __REG32 WakeUpPnd           : 1;
  __REG32                     :22;
} __canes_bits;

/* CAN Error Counter Register (CANxERRC) */
typedef struct {
  __REG32 TEC                 : 8;
  __REG32 REC                 : 7;
  __REG32 RP                  : 1;
  __REG32                     :16;
} __canerrc_bits;

/* CAN Bit Timing Register (CANxBTR) */
typedef struct {
  __REG32 BRP                 : 6;
  __REG32 SJW                 : 2;
  __REG32 TSeg1               : 4;
  __REG32 TSeg2               : 3;
  __REG32                     : 1;
  __REG32 BRPE                : 4;
  __REG32                     :12;
} __canbtr_bits;

/* CAN Interrupt Register (CANxINT) */
typedef struct {
  __REG32 Int0ID              :16;
  __REG32 Int1ID              : 8;
  __REG32                     : 8;
} __canint_bits;

/* CAN Test Register (CANxTEST) */
typedef struct {
  __REG32                     : 3;
  __REG32 Silent              : 1;
  __REG32 LBack               : 1;
  __REG32 Tx                  : 2;
  __REG32 Rx                  : 1;
  __REG32 EXL                 : 1;
  __REG32 RDA                 : 1;
  __REG32                     :22;
} __cantest_bits;

/* CAN Parity Error Code Register (CANxPERR) */
typedef struct {
  __REG32 MessageNumber       : 8;
  __REG32 WordNumber          : 3;
  __REG32                     :21;
} __canperr_bits;

/* CAN Transmission Request Registers (CANxTXRQ) */
typedef struct {
  __REG32 TxRqst1           : 1;
  __REG32 TxRqst2           : 1;
  __REG32 TxRqst3           : 1;
  __REG32 TxRqst4           : 1;
  __REG32 TxRqst5           : 1;
  __REG32 TxRqst6           : 1;
  __REG32 TxRqst7           : 1;
  __REG32 TxRqst8           : 1;
  __REG32 TxRqst9           : 1;
  __REG32 TxRqst10          : 1;
  __REG32 TxRqst11          : 1;
  __REG32 TxRqst12          : 1;
  __REG32 TxRqst13          : 1;
  __REG32 TxRqst14          : 1;
  __REG32 TxRqst15          : 1;
  __REG32 TxRqst16          : 1;
  __REG32 TxRqst17          : 1;
  __REG32 TxRqst18          : 1;
  __REG32 TxRqst19          : 1;
  __REG32 TxRqst20          : 1;
  __REG32 TxRqst21          : 1;
  __REG32 TxRqst22          : 1;
  __REG32 TxRqst23          : 1;
  __REG32 TxRqst24          : 1;
  __REG32 TxRqst25          : 1;
  __REG32 TxRqst26          : 1;
  __REG32 TxRqst27          : 1;
  __REG32 TxRqst28          : 1;
  __REG32 TxRqst29          : 1;
  __REG32 TxRqst30          : 1;
  __REG32 TxRqst31          : 1;
  __REG32 TxRqst32          : 1;
} __cantxrq_bits;

/* CAN New Data Registers (CANxNWDAT) */
typedef struct {
  __REG32 NewDat1           : 1;
  __REG32 NewDat2           : 1;
  __REG32 NewDat3           : 1;
  __REG32 NewDat4           : 1;
  __REG32 NewDat5           : 1;
  __REG32 NewDat6           : 1;
  __REG32 NewDat7           : 1;
  __REG32 NewDat8           : 1;
  __REG32 NewDat9           : 1;
  __REG32 NewDat10          : 1;
  __REG32 NewDat11          : 1;
  __REG32 NewDat12          : 1;
  __REG32 NewDat13          : 1;
  __REG32 NewDat14          : 1;
  __REG32 NewDat15          : 1;
  __REG32 NewDat16          : 1;
  __REG32 NewDat17          : 1;
  __REG32 NewDat18          : 1;
  __REG32 NewDat19          : 1;
  __REG32 NewDat20          : 1;
  __REG32 NewDat21          : 1;
  __REG32 NewDat22          : 1;
  __REG32 NewDat23          : 1;
  __REG32 NewDat24          : 1;
  __REG32 NewDat25          : 1;
  __REG32 NewDat26          : 1;
  __REG32 NewDat27          : 1;
  __REG32 NewDat28          : 1;
  __REG32 NewDat29          : 1;
  __REG32 NewDat30          : 1;
  __REG32 NewDat31          : 1;
  __REG32 NewDat32          : 1;
} __cannwdat_bits;

/* CAN Interrupt Pending Registers (CANxINTPND) */
typedef struct {
  __REG32 IntPnd1           : 1;
  __REG32 IntPnd2           : 1;
  __REG32 IntPnd3           : 1;
  __REG32 IntPnd4           : 1;
  __REG32 IntPnd5           : 1;
  __REG32 IntPnd6           : 1;
  __REG32 IntPnd7           : 1;
  __REG32 IntPnd8           : 1;
  __REG32 IntPnd9           : 1;
  __REG32 IntPnd10          : 1;
  __REG32 IntPnd11          : 1;
  __REG32 IntPnd12          : 1;
  __REG32 IntPnd13          : 1;
  __REG32 IntPnd14          : 1;
  __REG32 IntPnd15          : 1;
  __REG32 IntPnd16          : 1;
  __REG32 IntPnd17          : 1;
  __REG32 IntPnd18          : 1;
  __REG32 IntPnd19          : 1;
  __REG32 IntPnd20          : 1;
  __REG32 IntPnd21          : 1;
  __REG32 IntPnd22          : 1;
  __REG32 IntPnd23          : 1;
  __REG32 IntPnd24          : 1;
  __REG32 IntPnd25          : 1;
  __REG32 IntPnd26          : 1;
  __REG32 IntPnd27          : 1;
  __REG32 IntPnd28          : 1;
  __REG32 IntPnd29          : 1;
  __REG32 IntPnd30          : 1;
  __REG32 IntPnd31          : 1;
  __REG32 IntPnd32          : 1;
} __canintpnd_bits;

/* CAN Message Valid Registers (CANxMSGVAL) */
typedef struct {
  __REG32 MsgVal1           : 1;
  __REG32 MsgVal2           : 1;
  __REG32 MsgVal3           : 1;
  __REG32 MsgVal4           : 1;
  __REG32 MsgVal5           : 1;
  __REG32 MsgVal6           : 1;
  __REG32 MsgVal7           : 1;
  __REG32 MsgVal8           : 1;
  __REG32 MsgVal9           : 1;
  __REG32 MsgVal10          : 1;
  __REG32 MsgVal11          : 1;
  __REG32 MsgVal12          : 1;
  __REG32 MsgVal13          : 1;
  __REG32 MsgVal14          : 1;
  __REG32 MsgVal15          : 1;
  __REG32 MsgVal16          : 1;
  __REG32 MsgVal17          : 1;
  __REG32 MsgVal18          : 1;
  __REG32 MsgVal19          : 1;
  __REG32 MsgVal20          : 1;
  __REG32 MsgVal21          : 1;
  __REG32 MsgVal22          : 1;
  __REG32 MsgVal23          : 1;
  __REG32 MsgVal24          : 1;
  __REG32 MsgVal25          : 1;
  __REG32 MsgVal26          : 1;
  __REG32 MsgVal27          : 1;
  __REG32 MsgVal28          : 1;
  __REG32 MsgVal29          : 1;
  __REG32 MsgVal30          : 1;
  __REG32 MsgVal31          : 1;
  __REG32 MsgVal32          : 1;
} __canmsgval_bits;

/* CAN Interrupt Multiplexer Registers (CANxINTMUX) */
typedef struct {
  __REG32 IntMux1           : 1;
  __REG32 IntMux2           : 1;
  __REG32 IntMux3           : 1;
  __REG32 IntMux4           : 1;
  __REG32 IntMux5           : 1;
  __REG32 IntMux6           : 1;
  __REG32 IntMux7           : 1;
  __REG32 IntMux8           : 1;
  __REG32 IntMux9           : 1;
  __REG32 IntMux10          : 1;
  __REG32 IntMux11          : 1;
  __REG32 IntMux12          : 1;
  __REG32 IntMux13          : 1;
  __REG32 IntMux14          : 1;
  __REG32 IntMux15          : 1;
  __REG32 IntMux16          : 1;
  __REG32 IntMux17          : 1;
  __REG32 IntMux18          : 1;
  __REG32 IntMux19          : 1;
  __REG32 IntMux20          : 1;
  __REG32 IntMux21          : 1;
  __REG32 IntMux22          : 1;
  __REG32 IntMux23          : 1;
  __REG32 IntMux24          : 1;
  __REG32 IntMux25          : 1;
  __REG32 IntMux26          : 1;
  __REG32 IntMux27          : 1;
  __REG32 IntMux28          : 1;
  __REG32 IntMux29          : 1;
  __REG32 IntMux30          : 1;
  __REG32 IntMux31          : 1;
  __REG32 IntMux32          : 1;
} __canintmux_bits;

/* CAN IF1/2 Command Registers (CANxIFyCMD) */
typedef struct {
  __REG32 MessageNumber       : 8;
  __REG32                     : 6;
  __REG32 DMAactive           : 1;
  __REG32 Busy                : 1;
  __REG32 DataB               : 1;
  __REG32 DataA               : 1;
  __REG32 TxRqst_NewDat       : 1;
  __REG32 ClrIntPnd           : 1;
  __REG32 Control             : 1;
  __REG32 Arb                 : 1;
  __REG32 Mask                : 1;
  __REG32 WR_RD               : 1;
  __REG32                     : 8;
} __canifcmd_bits;

/* CAN IF1/IF2/IF2 Mask Registers (CANxIFyMSK) */
typedef struct {
  __REG32 Msk                 :29;
  __REG32                     : 1;
  __REG32 MDir                : 1;
  __REG32 MXtd                : 1;
} __canifmsk_bits;

/* CAN IF1/IF2/IF3 Arbitration Registers (CANxIFyARB) */
typedef struct {
  __REG32 ID                  :29;
  __REG32 Dir                 : 1;
  __REG32 Xtd                 : 1;
  __REG32 MsgVal              : 1;
} __canifarb_bits;

/* CAN IF1/IF2/IF3 Message Control Registers (CANxIFyMCTL) */
typedef struct {
  __REG32 DLC                 : 4;
  __REG32                     : 3;
  __REG32 EoB                 : 1;
  __REG32 TxRqst              : 1;
  __REG32 RmtEn               : 1;
  __REG32 RxIE                : 1;
  __REG32 TxIE                : 1;
  __REG32 UMask               : 1;
  __REG32 IntPnd              : 1;
  __REG32 MsgLst              : 1;
  __REG32 NewDat              : 1;
  __REG32                     :16;
} __canifmctl_bits;

/* CAN IF1/2/3 Data A Register (CANxIFyDATA) */
typedef struct {
  __REG32 Data0               : 8;
  __REG32 Data1               : 8;
  __REG32 Data2               : 8;
  __REG32 Data3               : 8;
} __canifdata_bits;

/* CAN IF1/2/3 Data B Register (CANxIFyDATB) */
typedef struct {
  __REG32 Data4               : 8;
  __REG32 Data5               : 8;
  __REG32 Data6               : 8;
  __REG32 Data7               : 8;
} __canifdatb_bits;

/* CAN IF3 Observation Register (CANxIF3OBS) */
typedef struct {
  __REG32 Mask                : 1;
  __REG32 Arb                 : 1;
  __REG32 Ctrl                : 1;
  __REG32 DataA               : 1;
  __REG32 DataB               : 1;
  __REG32                     :27;
} __canif3obs_bits;

/* CAN IF3 Update Enable Registers (CANxIF3UPD) */
typedef struct {
  __REG32 IF3UpdEn1           : 1;
  __REG32 IF3UpdEn2           : 1;
  __REG32 IF3UpdEn3           : 1;
  __REG32 IF3UpdEn4           : 1;
  __REG32 IF3UpdEn5           : 1;
  __REG32 IF3UpdEn6           : 1;
  __REG32 IF3UpdEn7           : 1;
  __REG32 IF3UpdEn8           : 1;
  __REG32 IF3UpdEn9           : 1;
  __REG32 IF3UpdEn10          : 1;
  __REG32 IF3UpdEn11          : 1;
  __REG32 IF3UpdEn12          : 1;
  __REG32 IF3UpdEn13          : 1;
  __REG32 IF3UpdEn14          : 1;
  __REG32 IF3UpdEn15          : 1;
  __REG32 IF3UpdEn16          : 1;
  __REG32 IF3UpdEn17          : 1;
  __REG32 IF3UpdEn18          : 1;
  __REG32 IF3UpdEn19          : 1;
  __REG32 IF3UpdEn20          : 1;
  __REG32 IF3UpdEn21          : 1;
  __REG32 IF3UpdEn22          : 1;
  __REG32 IF3UpdEn23          : 1;
  __REG32 IF3UpdEn24          : 1;
  __REG32 IF3UpdEn25          : 1;
  __REG32 IF3UpdEn26          : 1;
  __REG32 IF3UpdEn27          : 1;
  __REG32 IF3UpdEn28          : 1;
  __REG32 IF3UpdEn29          : 1;
  __REG32 IF3UpdEn30          : 1;
  __REG32 IF3UpdEn31          : 1;
  __REG32 IF3UpdEn32          : 1;
} __canif3upd_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler *********************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                  0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(ACTLR,                 0xE000E008,__READ       ,__actlr_bits);
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
__IO_REG32_BIT(SETENA2,               0xE000E108,__READ_WRITE ,__setena2_bits);
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,               0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(CLRENA2,               0xE000E188,__READ_WRITE ,__clrena2_bits);
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,              0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(SETPEND2,              0xE000E208,__READ_WRITE ,__setpend2_bits);
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,              0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(CLRPEND2,              0xE000E288,__READ_WRITE ,__clrpend2_bits);
__IO_REG32_BIT(ACTIVE0,               0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,               0xE000E304,__READ       ,__active1_bits);
__IO_REG32_BIT(ACTIVE2,               0xE000E308,__READ       ,__active2_bits);
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
__IO_REG32_BIT(IP16,                  0xE000E440,__READ_WRITE ,__pri16_bits);
__IO_REG32_BIT(IP17,                  0xE000E444,__READ_WRITE ,__pri17_bits);
__IO_REG32_BIT(IP18,                  0xE000E448,__READ_WRITE ,__pri18_bits);
__IO_REG32_BIT(IP19,                  0xE000E44C,__READ_WRITE ,__pri19_bits);
__IO_REG32_BIT(IP20,                  0xE000E450,__READ_WRITE ,__pri20_bits);
__IO_REG32_BIT(IP21,                  0xE000E454,__READ_WRITE ,__pri21_bits);
__IO_REG32_BIT(IP22,                  0xE000E458,__READ_WRITE ,__pri22_bits);
__IO_REG32_BIT(CPUIDBR,               0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,                  0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,                 0xE000ED18,__READ_WRITE ,__shpr0_bits);
__IO_REG32_BIT(SHPR1,                 0xE000ED1C,__READ_WRITE ,__shpr1_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED20,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHCSR,                 0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                  0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                  0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                  0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,                 0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                  0xE000ED38,__READ_WRITE);
__IO_REG32(    STIR,                  0xE000EF00,__WRITE      );

/***************************************************************************
 **
 ** MPU
 **
 ***************************************************************************/
__IO_REG32_BIT(MPUTYPE,               0xE000ED90,__READ       ,__mpu_type_bits);
__IO_REG32_BIT(MPUCTRL,               0xE000ED94,__READ_WRITE ,__mpu_ctrl_bits);
__IO_REG32_BIT(MPUNUMBER,             0xE000ED98,__READ_WRITE ,__mpu_rnr_bits);
__IO_REG32_BIT(MPUBASE,               0xE000ED9C,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPUATTR,               0xE000EDA0,__READ_WRITE ,__mpu_rasr_bits);
__IO_REG32_BIT(MPUBASE1,              0xE000EDA4,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPUATTR1,              0xE000EDA8,__READ_WRITE ,__mpu_rasr_bits);
__IO_REG32_BIT(MPUBASE2,              0xE000EDAC,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPUATTR2,              0xE000EDB0,__READ_WRITE ,__mpu_rasr_bits);
__IO_REG32_BIT(MPUBASE3,              0xE000EDB4,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPUATTR3,              0xE000EDB8,__READ_WRITE ,__mpu_rasr_bits);

/***************************************************************************
 **
 ** System Control and Configuration
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSPLLCTL,             0x400FB800,__READ_WRITE ,__syspllctl_bits);
__IO_REG32_BIT(SYSPLLMULT,            0x400FB804,__READ_WRITE ,__syspllmult_bits);
__IO_REG32_BIT(SYSPLLSTS,             0x400FB808,__READ       ,__syspllsts_bits);
__IO_REG32_BIT(SYSDIVSEL,             0x400FB80C,__READ_WRITE ,__sysdivsel_bits);
__IO_REG32_BIT(M3SSDIVSEL,            0x400FB810,__READ_WRITE ,__m3ssdivsel_bits);
__IO_REG32_BIT(UPLLCTL,               0x400FB820,__READ_WRITE ,__upllctl_bits);
__IO_REG32_BIT(UPLLMULT,              0x400FB824,__READ_WRITE ,__upllmult_bits);
__IO_REG32_BIT(UPLLSTS,               0x400FB828,__READ       ,__upllsts_bits);
__IO_REG32_BIT(MCLKSTS,               0x400FB830,__READ       ,__mclksts_bits);
__IO_REG32_BIT(MCLKFRCCLR,            0x400FB838,__READ_WRITE ,__mclkfrcclr_bits);
__IO_REG32_BIT(MCLKEN,                0x400FB83C,__READ_WRITE ,__mclken_bits);
__IO_REG32_BIT(MCLKLIMIT,             0x400FB840,__READ_WRITE ,__mclklimit_bits);
__IO_REG32_BIT(XPLLCLKCFG,            0x400FB850,__READ_WRITE ,__xpllclkcfg_bits);
__IO_REG32_BIT(CCLKOFF,               0x400FB860,__READ_WRITE ,__cclkoff_bits);
__IO_REG32_BIT(CAN0BCLKSEL,           0x400FB870,__READ_WRITE ,__canbclksel_bits);
__IO_REG32_BIT(CAN1BCLKSEL,           0x400FB874,__READ_WRITE ,__canbclksel_bits);
__IO_REG32(    CLPMSTAT,              0x400FB880,__READ_WRITE );
__IO_REG32_BIT(CRESCNF,               0x400FB8C0,__READ_WRITE ,__crescnf_bits);
__IO_REG32_BIT(CRESSTS,               0x400FB8C4,__READ_WRITE ,__cressts_bits);
__IO_REG32_BIT(MWIR,                  0x400FB8CC,__READ_WRITE ,__mwir_bits);
__IO_REG32_BIT(MCNF,                  0x400FB900,__READ       ,__mcnf_bits);
__IO_REG32_BIT(SERPLOOP,              0x400FB908,__READ_WRITE ,__serploop_bits);
__IO_REG32_BIT(MCIBSTATUS,            0x400FB90C,__READ       ,__mcibstatus_bits);
__IO_REG32_BIT(CCNF0,                 0x400FB910,__READ       ,__ccnf0_bits);
__IO_REG32_BIT(CCNF1,                 0x400FB914,__READ       ,__ccnf1_bits);
__IO_REG32_BIT(CCNF2,                 0x400FB918,__READ       ,__ccnf2_bits);
__IO_REG32_BIT(CCNF3,                 0x400FB91C,__READ       ,__ccnf3_bits);
__IO_REG32_BIT(CCNF4,                 0x400FB920,__READ       ,__ccnf4_bits);
__IO_REG32_BIT(MEMCNF,                0x400FB930,__READ       ,__memcnf_bits);
__IO_REG32(    MWRALLOW,              0x400FB980,__READ_WRITE );
__IO_REG32_BIT(MLOCK,                 0x400FB984,__READ_WRITE ,__mlock_bits);
__IO_REG32_BIT(MNMICFG,               0x400FBA00,__READ_WRITE ,__mnmicfg_bits);
__IO_REG32_BIT(MNMIFLG,               0x400FBA04,__READ       ,__mnmiflg_bits);
__IO_REG32_BIT(MNMIFLGCLR,            0x400FBA08,__READ_WRITE ,__mnmiflgclr_bits);
__IO_REG32_BIT(MNMIFLGFRC,            0x400FBA0C,__READ_WRITE ,__mnmiflgfrc_bits);
__IO_REG32_BIT(MNMIWDCNT,             0x400FBA10,__READ       ,__mnmiwdcnt_bits);
__IO_REG32_BIT(MNMIWDPRD,             0x400FBA14,__READ_WRITE ,__mnmiwdprd_bits);
__IO_REG32_BIT(DID0,                  0x400FE000,__READ       ,__did0_bits);
__IO_REG32_BIT(DID1,                  0x400FE004,__READ       ,__did1_bits);
__IO_REG32_BIT(DC1,                   0x400FE010,__READ       ,__dc1_bits);
__IO_REG32_BIT(DC2,                   0x400FE014,__READ       ,__dc2_bits);
__IO_REG32_BIT(DC4,                   0x400FE01C,__READ       ,__dc4_bits);
__IO_REG32_BIT(DC6,                   0x400FE024,__READ       ,__dc6_bits);
__IO_REG32_BIT(DC7,                   0x400FE028,__READ       ,__dc7_bits);
__IO_REG32_BIT(SRCR0,                 0x400FE040,__READ_WRITE ,__srcr0_bits);
__IO_REG32_BIT(SRCR1,                 0x400FE044,__READ_WRITE ,__srcr1_bits);
__IO_REG32_BIT(SRCR2,                 0x400FE048,__READ_WRITE ,__srcr2_bits);
__IO_REG32_BIT(SRCR3,                 0x400FE04C,__READ_WRITE ,__srcr3_bits);
__IO_REG32_BIT(MRESC,                 0x400FE05C,__READ_WRITE ,__mresc_bits);
__IO_REG32_BIT(RCC,                   0x400FE060,__READ_WRITE ,__rcc_bits);
__IO_REG32_BIT(GPIOHBCTL,             0x400FE06C,__READ_WRITE ,__gpiohbctl_bits);
__IO_REG32_BIT(RCGC0,                 0x400FE100,__READ_WRITE ,__rcgc0_bits);
__IO_REG32_BIT(RCGC1,                 0x400FE104,__READ_WRITE ,__rcgc1_bits);
__IO_REG32_BIT(RCGC2,                 0x400FE108,__READ_WRITE ,__rcgc2_bits);
__IO_REG32_BIT(RCGC3,                 0x400FE10C,__READ_WRITE ,__rcgc3_bits);
__IO_REG32_BIT(SCGC0,                 0x400FE110,__READ_WRITE ,__rcgc0_bits);
__IO_REG32_BIT(SCGC1,                 0x400FE114,__READ_WRITE ,__rcgc1_bits);
__IO_REG32_BIT(SCGC2,                 0x400FE118,__READ_WRITE ,__rcgc2_bits);
__IO_REG32_BIT(SCGC3,                 0x400FE11C,__READ_WRITE ,__rcgc3_bits);
__IO_REG32_BIT(DCGC0,                 0x400FE120,__READ_WRITE ,__rcgc0_bits);
__IO_REG32_BIT(DCGC1,                 0x400FE124,__READ_WRITE ,__rcgc1_bits);
__IO_REG32_BIT(DCGC2,                 0x400FE128,__READ_WRITE ,__rcgc2_bits);
__IO_REG32_BIT(DCGC3,                 0x400FE12C,__READ_WRITE ,__rcgc3_bits);
__IO_REG32_BIT(DSLPCLKCFG,            0x400FE144,__READ_WRITE ,__dslpclkcfg_bits);
__IO_REG32_BIT(DC10,                  0x400FE194,__READ_WRITE ,__dc10_bits);

/***************************************************************************
 **
 ** Code Security Module (CSM)
 **
 ***************************************************************************/
__IO_REG32(    Z1_CSMKEY0,            0x400FB400,__READ_WRITE );
__IO_REG32(    Z1_CSMKEY1,            0x400FB404,__READ_WRITE );
__IO_REG32(    Z1_CSMKEY2,            0x400FB408,__READ_WRITE );
__IO_REG32(    Z1_CSMKEY3,            0x400FB40C,__READ_WRITE );
__IO_REG32(    Z1_ECSLKEY0,           0x400FB410,__READ_WRITE );
__IO_REG32(    Z1_ECSLKEY1,           0x400FB414,__READ_WRITE );
__IO_REG32(    Z2_CSMKEY0,            0x400FB418,__READ_WRITE );
__IO_REG32(    Z2_CSMKEY1,            0x400FB41C,__READ_WRITE );
__IO_REG32(    Z2_CSMKEY2,            0x400FB420,__READ_WRITE );
__IO_REG32(    Z2_CSMKEY3,            0x400FB424,__READ_WRITE );
__IO_REG32(    Z2_ECSLKEY0,           0x400FB428,__READ_WRITE );
__IO_REG32(    Z2_ECSLKEY1,           0x400FB42C,__READ_WRITE );
__IO_REG32_BIT(Z1_CSMCR,              0x400FB480,__READ_WRITE ,__z_csmcr_bits);
__IO_REG32_BIT(Z2_CSMCR,              0x400FB484,__READ_WRITE ,__z_csmcr_bits);
__IO_REG32_BIT(Z1_GRABSECTR,          0x400FB490,__READ       ,__z_grabsectr_bits);
__IO_REG32_BIT(Z1_GRABRAMR,           0x400FB494,__READ       ,__z_grabramr_bits);
__IO_REG32_BIT(Z2_GRABSECTR,          0x400FB498,__READ       ,__z_grabsectr_bits);
__IO_REG32_BIT(Z2_GRABRAMR,           0x400FB49C,__READ       ,__z_grabramr_bits);
__IO_REG32_BIT(Z1_EXEONLYR,           0x400FB4B0,__READ       ,__z_exeonlyr_bits);
__IO_REG32_BIT(Z2_EXEONLYR,           0x400FB4B4,__READ       ,__z_exeonlyr_bits);
__IO_REG32_BIT(OTPSECLOCK,            0x400FB520,__READ       ,__otpseclock_bits);

/***************************************************************************
 **
 ** uCRC
 **
 ***************************************************************************/
__IO_REG32_BIT(uCRCCONFIG,            0x400FB600,__READ_WRITE ,__ucrcconfig_bits);
__IO_REG32_BIT(uCRCCONTROL,           0x400FB604,__READ_WRITE ,__ucrccontrol_bits);
__IO_REG32(    uCRCRES,               0x400FB608,__READ_WRITE );

/***************************************************************************
 **
 ** IPC
 **
 ***************************************************************************/
__IO_REG32(    CTOMIPCACK,            0x400FB700,__WRITE      );
__IO_REG32_BIT(CTOMIPCSTS,            0x400FB704,__READ       ,__ctomipcsts_bits);
__IO_REG32(    MTOCIPCSET,            0x400FB708,__WRITE      );
__IO_REG32(    MTOCIPCCLR,            0x400FB70C,__WRITE      );
__IO_REG32_BIT(MTOCIPCFLG,            0x400FB710,__READ       ,__mtocipcflg_bits);
__IO_REG32(    MIPCCOUNTERL,          0x400FB718,__READ       );
__IO_REG32(    MIPCCOUNTERH,          0x400FB71C,__READ       );
__IO_REG32(    CTOMIPCCOM,            0x400FB720,__READ       );
__IO_REG32(    CTOMIPCADDR,           0x400FB724,__READ       );
__IO_REG32(    CTOMIPCDATAW,          0x400FB728,__READ       );
__IO_REG32(    CTOMIPCDATAR,          0x400FB72C,__READ_WRITE);
__IO_REG32(    MTOCIPCCOM,            0x400FB730,__READ_WRITE );
__IO_REG32(    MTOCIPCADDR,           0x400FB734,__READ_WRITE );
__IO_REG32(    MTOCIPCDATAW,          0x400FB738,__READ_WRITE );
__IO_REG32(    MTOCIPCDATAR,          0x400FB73C,__READ       );
__IO_REG32(    CTOMIPCBOOTSTS,        0x400FB740,__READ       );
__IO_REG32(    MTOCIPCBOOTMODE,       0x400FB744,__READ_WRITE );
__IO_REG32_BIT(MPUMPREQUEST,          0x400FB748,__READ_WRITE ,__mpumprequest_bits);
__IO_REG32_BIT(MCLKREQUEST,           0x400FB74C,__READ_WRITE ,__mpumprequest_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM0CFG,              0x40030000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM0TAMR,             0x40030004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM0TBMR,             0x40030008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM0CTL,              0x4003000C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM0IMR,              0x40030018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM0RIS,              0x4003001C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM0MIS,              0x40030020,__READ       ,__gptmmis_bits);
__IO_REG32(    GPTM0ICR,              0x40030024,__WRITE      );
__IO_REG32_BIT(GPTM0TAILR,            0x40030028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM0TBILR,            0x4003002C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM0TAMATCHR,         0x40030030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM0TBMATCHR,         0x40030034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM0TAPR,             0x40030038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM0TBPR,             0x4003003C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM0TAPMR,            0x40030040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM0TBPMR,            0x40030044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM0TAR,              0x40030048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM0TBR,              0x4003004C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM0TAV,              0x40030050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM0TBV,              0x40030054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM1CFG,              0x40031000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM1TAMR,             0x40031004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM1TBMR,             0x40031008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM1CTL,              0x4003100C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM1IMR,              0x40031018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM1RIS,              0x4003101C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM1MIS,              0x40031020,__READ       ,__gptmmis_bits);
__IO_REG32(    GPTM1ICR,              0x40031024,__WRITE      );
__IO_REG32_BIT(GPTM1TAILR,            0x40031028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM1TBILR,            0x4003102C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM1TAMATCHR,         0x40031030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM1TBMATCHR,         0x40031034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM1TAPR,             0x40031038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM1TBPR,             0x4003103C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM1TAPMR,            0x40031040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM1TBPMR,            0x40031044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM1TAR,              0x40031048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM1TBR,              0x4003104C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM1TAV,              0x40031050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM1TBV,              0x40031054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM2CFG,              0x40032000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM2TAMR,             0x40032004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM2TBMR,             0x40032008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM2CTL,              0x4003200C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM2IMR,              0x40032018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM2RIS,              0x4003201C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM2MIS,              0x40032020,__READ       ,__gptmmis_bits);
__IO_REG32(    GPTM2ICR,              0x40032024,__WRITE      );
__IO_REG32_BIT(GPTM2TAILR,            0x40032028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM2TBILR,            0x4003202C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM2TAMATCHR,         0x40032030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM2TBMATCHR,         0x40032034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM2TAPR,             0x40032038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM2TBPR,             0x4003203C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM2TAPMR,            0x40032040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM2TBPMR,            0x40032044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM2TAR,              0x40032048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM2TBR,              0x4003204C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM2TAV,              0x40032050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM2TBV,              0x40032054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM3CFG,              0x40033000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM3TAMR,             0x40033004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM3TBMR,             0x40033008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM3CTL,              0x4003300C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM3IMR,              0x40033018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM3RIS,              0x4003301C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM3MIS,              0x40033020,__READ       ,__gptmmis_bits);
__IO_REG32(    GPTM3ICR,              0x40033024,__WRITE      );
__IO_REG32_BIT(GPTM3TAILR,            0x40033028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM3TBILR,            0x4003302C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM3TAMATCHR,         0x40033030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM3TBMATCHR,         0x40033034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM3TAPR,             0x40033038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM3TBPR,             0x4003303C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM3TAPMR,            0x40033040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM3TBPMR,            0x40033044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM3TAR,              0x40033048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM3TBR,              0x4003304C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM3TAV,              0x40033050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM3TBV,              0x40033054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** WDT0
 **
 ***************************************************************************/
__IO_REG32(    WDT0LOAD,              0x40000000,__READ_WRITE);
__IO_REG32(    WDT0VALUE,             0x40000004,__READ);
__IO_REG32_BIT(WDT0CTL,               0x40000008,__READ_WRITE ,__wdt0ctl_bits);
__IO_REG32(    WDT0ICR,               0x4000000C,__WRITE);
__IO_REG32_BIT(WDT0RIS,               0x40000010,__READ       ,__wdtris_bits);
__IO_REG32_BIT(WDT0MIS,               0x40000014,__READ       ,__wdtmis_bits);
__IO_REG32_BIT(WDT0TEST,              0x40000018,__READ_WRITE ,__wdttest_bits);
__IO_REG32(    WDT0LOCK,              0x40000C00,__READ_WRITE);
__IO_REG8(     WDT0PERIPHID4,         0x40000FD0,__READ);
__IO_REG8(     WDT0PERIPHID5,         0x40000FD4,__READ);
__IO_REG8(     WDT0PERIPHID6,         0x40000FD8,__READ);
__IO_REG8(     WDT0PERIPHID7,         0x40000FDC,__READ);
__IO_REG8(     WDT0PERIPHID0,         0x40000FE0,__READ);
__IO_REG8(     WDT0PERIPHID1,         0x40000FE4,__READ);
__IO_REG8(     WDT0PERIPHID2,         0x40000FE8,__READ);
__IO_REG8(     WDT0PERIPHID3,         0x40000FEC,__READ);
__IO_REG8(     WDT0PCELLID0,          0x40000FF0,__READ);
__IO_REG8(     WDT0PCELLID1,          0x40000FF4,__READ);
__IO_REG8(     WDT0PCELLID2,          0x40000FF8,__READ);
__IO_REG8(     WDT0PCELLID3,          0x40000FFC,__READ);

/***************************************************************************
 **
 ** WDT1
 **
 ***************************************************************************/
__IO_REG32(    WDT1LOAD,              0x40001000,__READ_WRITE);
__IO_REG32(    WDT1VALUE,             0x40001004,__READ);
__IO_REG32_BIT(WDT1CTL,               0x40001008,__READ_WRITE ,__wdt1ctl_bits);
__IO_REG32(    WDT1ICR,               0x4000100C,__WRITE);
__IO_REG32_BIT(WDT1RIS,               0x40001010,__READ       ,__wdtris_bits);
__IO_REG32_BIT(WDT1MIS,               0x40001014,__READ       ,__wdtmis_bits);
__IO_REG32_BIT(WDT1TEST,              0x40001018,__READ_WRITE ,__wdttest_bits);
__IO_REG32(    WDT1LOCK,              0x40001C00,__READ_WRITE);
__IO_REG8(     WDT1PERIPHID4,         0x40001FD0,__READ);
__IO_REG8(     WDT1PERIPHID5,         0x40001FD4,__READ);
__IO_REG8(     WDT1PERIPHID6,         0x40001FD8,__READ);
__IO_REG8(     WDT1PERIPHID7,         0x40001FDC,__READ);
__IO_REG8(     WDT1PERIPHID0,         0x40001FE0,__READ);
__IO_REG8(     WDT1PERIPHID1,         0x40001FE4,__READ);
__IO_REG8(     WDT1PERIPHID2,         0x40001FE8,__READ);
__IO_REG8(     WDT1PERIPHID3,         0x40001FEC,__READ);
__IO_REG8(     WDT1PCELLID0,          0x40001FF0,__READ);
__IO_REG8(     WDT1PCELLID1,          0x40001FF4,__READ);
__IO_REG8(     WDT1PCELLID2,          0x40001FF8,__READ);
__IO_REG8(     WDT1PCELLID3,          0x40001FFC,__READ);

/***************************************************************************
 **
 ** GPIOA APB
 **
 ***************************************************************************/
#define GPIOADATA_BASE,               0x40004000
__IO_REG32_BIT(GPIOADATA,             0x400043FC,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOADIR,              0x40004400,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAIS,               0x40004404,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAIBE,              0x40004408,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAIEV,              0x4000440C,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAIM,               0x40004410,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOARIS,              0x40004414,__READ       ,__gpioa_bits);
__IO_REG32_BIT(GPIOAMIS,              0x40004418,__READ       ,__gpioa_bits);
__IO_REG32(    GPIOAICR,              0x4000441C,__WRITE      );
__IO_REG32_BIT(GPIOAAFSEL,            0x40004420,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAODR,              0x4000450C,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAPUR,              0x40004510,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOADEN,              0x4000451C,__READ_WRITE ,__gpioa_bits);
__IO_REG32(    GPIOALOCK,             0x40004520,__READ_WRITE );
__IO_REG32_BIT(GPIOACR,               0x40004524,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAAMSEL,            0x40004528,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOAPCTL,             0x4000452C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOAAPSEL,            0x40004530,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOACSEL,             0x40004534,__READ_WRITE ,__gpioa_bits);
__IO_REG8(     GPIOAPERIPHID4,        0x40004FD0,__READ);
__IO_REG8(     GPIOAPERIPHID5,        0x40004FD4,__READ);
__IO_REG8(     GPIOAPERIPHID6,        0x40004FD8,__READ);
__IO_REG8(     GPIOAPERIPHID7,        0x40004FDC,__READ);
__IO_REG8(     GPIOAPERIPHID0,        0x40004FE0,__READ);
__IO_REG8(     GPIOAPERIPHID1,        0x40004FE4,__READ);
__IO_REG8(     GPIOAPERIPHID2,        0x40004FE8,__READ);
__IO_REG8(     GPIOAPERIPHID3,        0x40004FEC,__READ);
__IO_REG8(     GPIOAPCELLID0,         0x40004FF0,__READ);
__IO_REG8(     GPIOAPCELLID1,         0x40004FF4,__READ);
__IO_REG8(     GPIOAPCELLID2,         0x40004FF8,__READ);
__IO_REG8(     GPIOAPCELLID3,         0x40004FFC,__READ);

/***************************************************************************
 **
 ** GPIOA AHB
 **
 ***************************************************************************/
#define GPIOA_AHB_DATA_BASE,          0x40058000
__IO_REG32_BIT(GPIOA_AHB_DATA,        0x400583FC,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_DIR,         0x40058400,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_IS,          0x40058404,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_IBE,         0x40058408,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_IEV,         0x4005840C,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_IM,          0x40058410,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_RIS,         0x40058414,__READ       ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_MIS,         0x40058418,__READ       ,__gpioa_bits);
__IO_REG32(    GPIOA_AHB_ICR,         0x4005841C,__WRITE      );
__IO_REG32_BIT(GPIOA_AHB_AFSEL,       0x40058420,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_ODR,         0x4005850C,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_PUR,         0x40058510,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_DEN,         0x4005851C,__READ_WRITE ,__gpioa_bits);
__IO_REG32(    GPIOA_AHB_LOCK,        0x40058520,__READ_WRITE );
__IO_REG32_BIT(GPIOA_AHB_CR,          0x40058524,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_AMSEL,       0x40058528,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_PCTL,        0x4005852C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOA_AHB_APSEL,       0x40058530,__READ_WRITE ,__gpioa_bits);
__IO_REG32_BIT(GPIOA_AHB_CSEL,        0x40058534,__READ_WRITE ,__gpioa_bits);
__IO_REG8(     GPIOA_AHB_PERIPHID4,   0x40058FD0,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID5,   0x40058FD4,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID6,   0x40058FD8,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID7,   0x40058FDC,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID0,   0x40058FE0,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID1,   0x40058FE4,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID2,   0x40058FE8,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID3,   0x40058FEC,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID0,    0x40058FF0,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID1,    0x40058FF4,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID2,    0x40058FF8,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID3,    0x40058FFC,__READ);

/***************************************************************************
 **
 ** GPIOB APB
 **
 ***************************************************************************/
#define GPIOBDATA_BASE,               0x40005000
__IO_REG32_BIT(GPIOBDATA,             0x400053FC,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBDIR,              0x40005400,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBIS,               0x40005404,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBIBE,              0x40005408,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBIEV,              0x4000540C,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBIM,               0x40005410,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBRIS,              0x40005414,__READ       ,__gpiob_bits);
__IO_REG32_BIT(GPIOBMIS,              0x40005418,__READ       ,__gpiob_bits);
__IO_REG32(    GPIOBICR,              0x4000541C,__WRITE      );
__IO_REG32_BIT(GPIOBAFSEL,            0x40005420,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBODR,              0x4000550C,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBPUR,              0x40005510,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBDEN,              0x4000551C,__READ_WRITE ,__gpiob_bits);
__IO_REG32(    GPIOBLOCK,             0x40005520,__READ_WRITE );
__IO_REG32_BIT(GPIOBCR,               0x40005524,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBAMSEL,            0x40005528,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBPCTL,             0x4000552C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOBAPSEL,            0x40005530,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOBCSEL,             0x40005534,__READ_WRITE ,__gpiob_bits);
__IO_REG8(     GPIOBPERIPHID4,        0x40005FD0,__READ);
__IO_REG8(     GPIOBPERIPHID5,        0x40005FD4,__READ);
__IO_REG8(     GPIOBPERIPHID6,        0x40005FD8,__READ);
__IO_REG8(     GPIOBPERIPHID7,        0x40005FDC,__READ);
__IO_REG8(     GPIOBPERIPHID0,        0x40005FE0,__READ);
__IO_REG8(     GPIOBPERIPHID1,        0x40005FE4,__READ);
__IO_REG8(     GPIOBPERIPHID2,        0x40005FE8,__READ);
__IO_REG8(     GPIOBPERIPHID3,        0x40005FEC,__READ);
__IO_REG8(     GPIOBPCELLID0,         0x40005FF0,__READ);
__IO_REG8(     GPIOBPCELLID1,         0x40005FF4,__READ);
__IO_REG8(     GPIOBPCELLID2,         0x40005FF8,__READ);
__IO_REG8(     GPIOBPCELLID3,         0x40005FFC,__READ);

/***************************************************************************
 **
 ** GPIOB AHB
 **
 ***************************************************************************/
#define GPIOB_AHB_DATA_BASE,          0x40059000
__IO_REG32_BIT(GPIOB_AHB_DATA,        0x400593FC,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_DIR,         0x40059400,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_IS,          0x40059404,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_IBE,         0x40059408,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_IEV,         0x4005940C,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_IM,          0x40059410,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_RIS,         0x40059414,__READ       ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_MIS,         0x40059418,__READ       ,__gpiob_bits);
__IO_REG32(    GPIOB_AHB_ICR,         0x4005941C,__WRITE      );
__IO_REG32_BIT(GPIOB_AHB_AFSEL,       0x40059420,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_ODR,         0x4005950C,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_PUR,         0x40059510,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_DEN,         0x4005951C,__READ_WRITE ,__gpiob_bits);
__IO_REG32(    GPIOB_AHB_LOCK,        0x40059520,__READ_WRITE );
__IO_REG32_BIT(GPIOB_AHB_CR,          0x40059524,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_AMSEL,       0x40059528,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_PCTL,        0x4005952C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOB_AHB_APSEL,       0x40059530,__READ_WRITE ,__gpiob_bits);
__IO_REG32_BIT(GPIOB_AHB_CSEL,        0x40059534,__READ_WRITE ,__gpiob_bits);
__IO_REG8(     GPIOB_AHB_PERIPHID4,   0x40059FD0,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID5,   0x40059FD4,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID6,   0x40059FD8,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID7,   0x40059FDC,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID0,   0x40059FE0,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID1,   0x40059FE4,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID2,   0x40059FE8,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID3,   0x40059FEC,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID0,    0x40059FF0,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID1,    0x40059FF4,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID2,    0x40059FF8,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID3,    0x40059FFC,__READ);

/***************************************************************************
 **
 ** GPIOC APB
 **
 ***************************************************************************/
#define GPIOCDATA_BASE,               0x40006000
__IO_REG32_BIT(GPIOCDATA,             0x400063FC,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCDIR,              0x40006400,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCIS,               0x40006404,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCIBE,              0x40006408,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCIEV,              0x4000640C,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCIM,               0x40006410,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCRIS,              0x40006414,__READ       ,__gpioc_bits);
__IO_REG32_BIT(GPIOCMIS,              0x40006418,__READ       ,__gpioc_bits);
__IO_REG32(    GPIOCICR,              0x4000641C,__WRITE      );
__IO_REG32_BIT(GPIOCAFSEL,            0x40006420,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCODR,              0x4000650C,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCPUR,              0x40006510,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCDEN,              0x4000651C,__READ_WRITE ,__gpioc_bits);
__IO_REG32(    GPIOCLOCK,             0x40006520,__READ_WRITE );
__IO_REG32_BIT(GPIOCCR,               0x40006524,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCAMSEL,            0x40006528,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCPCTL,             0x4000652C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOCAPSEL,            0x40006530,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOCCSEL,             0x40006534,__READ_WRITE ,__gpioc_bits);
__IO_REG8(     GPIOCPERIPHID4,        0x40006FD0,__READ);
__IO_REG8(     GPIOCPERIPHID5,        0x40006FD4,__READ);
__IO_REG8(     GPIOCPERIPHID6,        0x40006FD8,__READ);
__IO_REG8(     GPIOCPERIPHID7,        0x40006FDC,__READ);
__IO_REG8(     GPIOCPERIPHID0,        0x40006FE0,__READ);
__IO_REG8(     GPIOCPERIPHID1,        0x40006FE4,__READ);
__IO_REG8(     GPIOCPERIPHID2,        0x40006FE8,__READ);
__IO_REG8(     GPIOCPERIPHID3,        0x40006FEC,__READ);
__IO_REG8(     GPIOCPCELLID0,         0x40006FF0,__READ);
__IO_REG8(     GPIOCPCELLID1,         0x40006FF4,__READ);
__IO_REG8(     GPIOCPCELLID2,         0x40006FF8,__READ);
__IO_REG8(     GPIOCPCELLID3,         0x40006FFC,__READ);

/***************************************************************************
 **
 ** GPIOC AHB
 **
 ***************************************************************************/
#define GPIOC_AHB_DATA_BASE,          0x4005A000
__IO_REG32_BIT(GPIOC_AHB_DATA,        0x4005A3FC,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_DIR,         0x4005A400,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_IS,          0x4005A404,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_IBE,         0x4005A408,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_IEV,         0x4005A40C,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_IM,          0x4005A410,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_RIS,         0x4005A414,__READ       ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_MIS,         0x4005A418,__READ       ,__gpioc_bits);
__IO_REG32(    GPIOC_AHB_ICR,         0x4005A41C,__WRITE      );
__IO_REG32_BIT(GPIOC_AHB_AFSEL,       0x4005A420,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_ODR,         0x4005A50C,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_PUR,         0x4005A510,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_DEN,         0x4005A51C,__READ_WRITE ,__gpioc_bits);
__IO_REG32(    GPIOC_AHB_LOCK,        0x4005A520,__READ_WRITE );
__IO_REG32_BIT(GPIOC_AHB_CR,          0x4005A524,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_AMSEL,       0x4005A528,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_PCTL,        0x4005A52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOC_AHB_APSEL,       0x4005A530,__READ_WRITE ,__gpioc_bits);
__IO_REG32_BIT(GPIOC_AHB_CSEL,        0x4005A534,__READ_WRITE ,__gpioc_bits);
__IO_REG8(     GPIOC_AHB_PERIPHID4,   0x4005AFD0,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID5,   0x4005AFD4,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID6,   0x4005AFD8,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID7,   0x4005AFDC,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID0,   0x4005AFE0,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID1,   0x4005AFE4,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID2,   0x4005AFE8,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID3,   0x4005AFEC,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID0,    0x4005AFF0,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID1,    0x4005AFF4,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID2,    0x4005AFF8,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID3,    0x4005AFFC,__READ);

/***************************************************************************
 **
 ** GPIOD APB
 **
 ***************************************************************************/
#define GPIODDATA_BASE,               0x40007000
__IO_REG32_BIT(GPIODDATA,             0x400073FC,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODDIR,              0x40007400,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODIS,               0x40007404,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODIBE,              0x40007408,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODIEV,              0x4000740C,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODIM,               0x40007410,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODRIS,              0x40007414,__READ       ,__gpiod_bits);
__IO_REG32_BIT(GPIODMIS,              0x40007418,__READ       ,__gpiod_bits);
__IO_REG32(    GPIODICR,              0x4000741C,__WRITE      );
__IO_REG32_BIT(GPIODAFSEL,            0x40007420,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODODR,              0x4000750C,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODPUR,              0x40007510,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODDEN,              0x4000751C,__READ_WRITE ,__gpiod_bits);
__IO_REG32(    GPIODLOCK,             0x40007520,__READ_WRITE );
__IO_REG32_BIT(GPIODCR,               0x40007524,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODAMSEL,            0x40007528,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODPCTL,             0x4000752C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIODAPSEL,            0x40007530,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIODCSEL,             0x40007534,__READ_WRITE ,__gpiod_bits);
__IO_REG8(     GPIODPERIPHID4,        0x40007FD0,__READ);
__IO_REG8(     GPIODPERIPHID5,        0x40007FD4,__READ);
__IO_REG8(     GPIODPERIPHID6,        0x40007FD8,__READ);
__IO_REG8(     GPIODPERIPHID7,        0x40007FDC,__READ);
__IO_REG8(     GPIODPERIPHID0,        0x40007FE0,__READ);
__IO_REG8(     GPIODPERIPHID1,        0x40007FE4,__READ);
__IO_REG8(     GPIODPERIPHID2,        0x40007FE8,__READ);
__IO_REG8(     GPIODPERIPHID3,        0x40007FEC,__READ);
__IO_REG8(     GPIODPCELLID0,         0x40007FF0,__READ);
__IO_REG8(     GPIODPCELLID1,         0x40007FF4,__READ);
__IO_REG8(     GPIODPCELLID2,         0x40007FF8,__READ);
__IO_REG8(     GPIODPCELLID3,         0x40007FFC,__READ);

/***************************************************************************
 **
 ** GPIOD AHB
 **
 ***************************************************************************/
#define GPIOD_AHB_DATA_BASE,          0x4005B000
__IO_REG32_BIT(GPIOD_AHB_DATA,        0x4005B3FC,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_DIR,         0x4005B400,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_IS,          0x4005B404,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_IBE,         0x4005B408,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_IEV,         0x4005B40C,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_IM,          0x4005B410,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_RIS,         0x4005B414,__READ       ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_MIS,         0x4005B418,__READ       ,__gpiod_bits);
__IO_REG32(    GPIOD_AHB_ICR,         0x4005B41C,__WRITE      );
__IO_REG32_BIT(GPIOD_AHB_AFSEL,       0x4005B420,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_ODR,         0x4005B50C,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_PUR,         0x4005B510,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_DEN,         0x4005B51C,__READ_WRITE ,__gpiod_bits);
__IO_REG32(    GPIOD_AHB_LOCK,        0x4005B520,__READ_WRITE );
__IO_REG32_BIT(GPIOD_AHB_CR,          0x4005B524,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_AMSEL,       0x4005B528,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_PCTL,        0x4005B52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOD_AHB_APSEL,       0x4005B530,__READ_WRITE ,__gpiod_bits);
__IO_REG32_BIT(GPIOD_AHB_CSEL,        0x4005B534,__READ_WRITE ,__gpiod_bits);
__IO_REG8(     GPIOD_AHB_PERIPHID4,   0x4005BFD0,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID5,   0x4005BFD4,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID6,   0x4005BFD8,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID7,   0x4005BFDC,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID0,   0x4005BFE0,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID1,   0x4005BFE4,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID2,   0x4005BFE8,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID3,   0x4005BFEC,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID0,    0x4005BFF0,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID1,    0x4005BFF4,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID2,    0x4005BFF8,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID3,    0x4005BFFC,__READ);

/***************************************************************************
 **
 ** GPIOE APB
 **
 ***************************************************************************/
#define GPIOEDATA_BASE,               0x40024000
__IO_REG32_BIT(GPIOEDATA,             0x400243FC,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEDIR,              0x40024400,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEIS,               0x40024404,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEIBE,              0x40024408,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEIEV,              0x4002440C,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEIM,               0x40024410,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOERIS,              0x40024414,__READ       ,__gpioe_bits);
__IO_REG32_BIT(GPIOEMIS,              0x40024418,__READ       ,__gpioe_bits);
__IO_REG32(    GPIOEICR,              0x4002441C,__WRITE      );
__IO_REG32_BIT(GPIOEAFSEL,            0x40024420,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEODR,              0x4002450C,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEPUR,              0x40024510,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEDEN,              0x4002451C,__READ_WRITE ,__gpioe_bits);
__IO_REG32(    GPIOELOCK,             0x40024520,__READ_WRITE );
__IO_REG32_BIT(GPIOECR,               0x40024524,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEAMSEL,            0x40024528,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOEPCTL,             0x4002452C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOEAPSEL,            0x40024530,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOECSEL,             0x40024534,__READ_WRITE ,__gpioe_bits);
__IO_REG8(     GPIOEPERIPHID4,        0x40024FD0,__READ);
__IO_REG8(     GPIOEPERIPHID5,        0x40024FD4,__READ);
__IO_REG8(     GPIOEPERIPHID6,        0x40024FD8,__READ);
__IO_REG8(     GPIOEPERIPHID7,        0x40024FDC,__READ);
__IO_REG8(     GPIOEPERIPHID0,        0x40024FE0,__READ);
__IO_REG8(     GPIOEPERIPHID1,        0x40024FE4,__READ);
__IO_REG8(     GPIOEPERIPHID2,        0x40024FE8,__READ);
__IO_REG8(     GPIOEPERIPHID3,        0x40024FEC,__READ);
__IO_REG8(     GPIOEPCELLID0,         0x40024FF0,__READ);
__IO_REG8(     GPIOEPCELLID1,         0x40024FF4,__READ);
__IO_REG8(     GPIOEPCELLID2,         0x40024FF8,__READ);
__IO_REG8(     GPIOEPCELLID3,         0x40024FFC,__READ);

/***************************************************************************
 **
 ** GPIOE AHB
 **
 ***************************************************************************/
#define GPIOE_AHB_DATA_BASE,          0x4005C000
__IO_REG32_BIT(GPIOE_AHB_DATA,        0x4005C3FC,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_DIR,         0x4005C400,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_IS,          0x4005C404,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_IBE,         0x4005C408,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_IEV,         0x4005C40C,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_IM,          0x4005C410,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_RIS,         0x4005C414,__READ       ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_MIS,         0x4005C418,__READ       ,__gpioe_bits);
__IO_REG32(    GPIOE_AHB_ICR,         0x4005C41C,__WRITE      );
__IO_REG32_BIT(GPIOE_AHB_AFSEL,       0x4005C420,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_ODR,         0x4005C50C,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_PUR,         0x4005C510,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_DEN,         0x4005C51C,__READ_WRITE ,__gpioe_bits);
__IO_REG32(    GPIOE_AHB_LOCK,        0x4005C520,__READ_WRITE );
__IO_REG32_BIT(GPIOE_AHB_CR,          0x4005C524,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_AMSEL,       0x4005C528,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_PCTL,        0x4005C52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOE_AHB_APSEL,       0x4005C530,__READ_WRITE ,__gpioe_bits);
__IO_REG32_BIT(GPIOE_AHB_CSEL,        0x4005C534,__READ_WRITE ,__gpioe_bits);
__IO_REG8(     GPIOE_AHB_PERIPHID4,   0x4005CFD0,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID5,   0x4005CFD4,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID6,   0x4005CFD8,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID7,   0x4005CFDC,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID0,   0x4005CFE0,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID1,   0x4005CFE4,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID2,   0x4005CFE8,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID3,   0x4005CFEC,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID0,    0x4005CFF0,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID1,    0x4005CFF4,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID2,    0x4005CFF8,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID3,    0x4005CFFC,__READ);

/***************************************************************************
 **
 ** GPIOF APB
 **
 ***************************************************************************/
#define GPIOFDATA_BASE,               0x40025000
__IO_REG32_BIT(GPIOFDATA,             0x400253FC,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFDIR,              0x40025400,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFIS,               0x40025404,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFIBE,              0x40025408,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFIEV,              0x4002540C,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFIM,               0x40025410,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFRIS,              0x40025414,__READ       ,__gpiof_bits);
__IO_REG32_BIT(GPIOFMIS,              0x40025418,__READ       ,__gpiof_bits);
__IO_REG32(    GPIOFICR,              0x4002541C,__WRITE      );
__IO_REG32_BIT(GPIOFAFSEL,            0x40025420,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFODR,              0x4002550C,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFPUR,              0x40025510,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFDEN,              0x4002551C,__READ_WRITE ,__gpiof_bits);
__IO_REG32(    GPIOFLOCK,             0x40025520,__READ_WRITE );
__IO_REG32_BIT(GPIOFCR,               0x40025524,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFAMSEL,            0x40025528,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFPCTL,             0x4002552C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOFAPSEL,            0x40025530,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOFCSEL,             0x40025534,__READ_WRITE ,__gpiof_bits);
__IO_REG8(     GPIOFPERIPHID4,        0x40025FD0,__READ);
__IO_REG8(     GPIOFPERIPHID5,        0x40025FD4,__READ);
__IO_REG8(     GPIOFPERIPHID6,        0x40025FD8,__READ);
__IO_REG8(     GPIOFPERIPHID7,        0x40025FDC,__READ);
__IO_REG8(     GPIOFPERIPHID0,        0x40025FE0,__READ);
__IO_REG8(     GPIOFPERIPHID1,        0x40025FE4,__READ);
__IO_REG8(     GPIOFPERIPHID2,        0x40025FE8,__READ);
__IO_REG8(     GPIOFPERIPHID3,        0x40025FEC,__READ);
__IO_REG8(     GPIOFPCELLID0,         0x40025FF0,__READ);
__IO_REG8(     GPIOFPCELLID1,         0x40025FF4,__READ);
__IO_REG8(     GPIOFPCELLID2,         0x40025FF8,__READ);
__IO_REG8(     GPIOFPCELLID3,         0x40025FFC,__READ);

/***************************************************************************
 **
 ** GPIOF AHB
 **
 ***************************************************************************/
#define GPIOF_AHB_DATA_BASE,          0x4005D000
__IO_REG32_BIT(GPIOF_AHB_DATA,        0x4005D3FC,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_DIR,         0x4005D400,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_IS,          0x4005D404,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_IBE,         0x4005D408,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_IEV,         0x4005D40C,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_IM,          0x4005D410,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_RIS,         0x4005D414,__READ       ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_MIS,         0x4005D418,__READ       ,__gpiof_bits);
__IO_REG32(    GPIOF_AHB_ICR,         0x4005D41C,__WRITE      );
__IO_REG32_BIT(GPIOF_AHB_AFSEL,       0x4005D420,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_ODR,         0x4005D50C,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_PUR,         0x4005D510,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_DEN,         0x4005D51C,__READ_WRITE ,__gpiof_bits);
__IO_REG32(    GPIOF_AHB_LOCK,        0x4005D520,__READ_WRITE );
__IO_REG32_BIT(GPIOF_AHB_CR,          0x4005D524,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_AMSEL,       0x4005D528,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_PCTL,        0x4005D52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOF_AHB_APSEL,       0x4005D530,__READ_WRITE ,__gpiof_bits);
__IO_REG32_BIT(GPIOF_AHB_CSEL,        0x4005D534,__READ_WRITE ,__gpiof_bits);
__IO_REG8(     GPIOF_AHB_PERIPHID4,   0x4005DFD0,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID5,   0x4005DFD4,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID6,   0x4005DFD8,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID7,   0x4005DFDC,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID0,   0x4005DFE0,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID1,   0x4005DFE4,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID2,   0x4005DFE8,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID3,   0x4005DFEC,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID0,    0x4005DFF0,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID1,    0x4005DFF4,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID2,    0x4005DFF8,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID3,    0x4005DFFC,__READ);

/***************************************************************************
 **
 ** GPIOG APB
 **
 ***************************************************************************/
#define GPIOGDATA_BASE,               0x40026000
__IO_REG32_BIT(GPIOGDATA,             0x400263FC,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGDIR,              0x40026400,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGIS,               0x40026404,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGIBE,              0x40026408,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGIEV,              0x4002640C,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGIM,               0x40026410,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGRIS,              0x40026414,__READ       ,__gpiog_bits);
__IO_REG32_BIT(GPIOGMIS,              0x40026418,__READ       ,__gpiog_bits);
__IO_REG32(    GPIOGICR,              0x4002641C,__WRITE      );
__IO_REG32_BIT(GPIOGAFSEL,            0x40026420,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGODR,              0x4002650C,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGPUR,              0x40026510,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGDEN,              0x4002651C,__READ_WRITE ,__gpiog_bits);
__IO_REG32(    GPIOGLOCK,             0x40026520,__READ_WRITE );
__IO_REG32_BIT(GPIOGCR,               0x40026524,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGAMSEL,            0x40026528,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGPCTL,             0x4002652C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOGAPSEL,            0x40026530,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOGCSEL,             0x40026534,__READ_WRITE ,__gpiog_bits);
__IO_REG8(     GPIOGPERIPHID4,        0x40026FD0,__READ);
__IO_REG8(     GPIOGPERIPHID5,        0x40026FD4,__READ);
__IO_REG8(     GPIOGPERIPHID6,        0x40026FD8,__READ);
__IO_REG8(     GPIOGPERIPHID7,        0x40026FDC,__READ);
__IO_REG8(     GPIOGPERIPHID0,        0x40026FE0,__READ);
__IO_REG8(     GPIOGPERIPHID1,        0x40026FE4,__READ);
__IO_REG8(     GPIOGPERIPHID2,        0x40026FE8,__READ);
__IO_REG8(     GPIOGPERIPHID3,        0x40026FEC,__READ);
__IO_REG8(     GPIOGPCELLID0,         0x40026FF0,__READ);
__IO_REG8(     GPIOGPCELLID1,         0x40026FF4,__READ);
__IO_REG8(     GPIOGPCELLID2,         0x40026FF8,__READ);
__IO_REG8(     GPIOGPCELLID3,         0x40026FFC,__READ);

/***************************************************************************
 **
 ** GPIOG AHB
 **
 ***************************************************************************/
#define GPIOG_AHB_DATA_BASE,          0x4005E000
__IO_REG32_BIT(GPIOG_AHB_DATA,        0x4005E3FC,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_DIR,         0x4005E400,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_IS,          0x4005E404,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_IBE,         0x4005E408,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_IEV,         0x4005E40C,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_IM,          0x4005E410,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_RIS,         0x4005E414,__READ       ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_MIS,         0x4005E418,__READ       ,__gpiog_bits);
__IO_REG32(    GPIOG_AHB_ICR,         0x4005E41C,__WRITE      );
__IO_REG32_BIT(GPIOG_AHB_AFSEL,       0x4005E420,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_ODR,         0x4005E50C,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_PUR,         0x4005E510,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_DEN,         0x4005E51C,__READ_WRITE ,__gpiog_bits);
__IO_REG32(    GPIOG_AHB_LOCK,        0x4005E520,__READ_WRITE );
__IO_REG32_BIT(GPIOG_AHB_CR,          0x4005E524,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_AMSEL,       0x4005E528,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_PCTL,        0x4005E52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOG_AHB_APSEL,       0x4005E530,__READ_WRITE ,__gpiog_bits);
__IO_REG32_BIT(GPIOG_AHB_CSEL,        0x4005E534,__READ_WRITE ,__gpiog_bits);
__IO_REG8(     GPIOG_AHB_PERIPHID4,   0x4005EFD0,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID5,   0x4005EFD4,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID6,   0x4005EFD8,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID7,   0x4005EFDC,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID0,   0x4005EFE0,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID1,   0x4005EFE4,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID2,   0x4005EFE8,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID3,   0x4005EFEC,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID0,    0x4005EFF0,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID1,    0x4005EFF4,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID2,    0x4005EFF8,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID3,    0x4005EFFC,__READ);

/***************************************************************************
 **
 ** GPIOH APB
 **
 ***************************************************************************/
#define GPIOHDATA_BASE,               0x40027000
__IO_REG32_BIT(GPIOHDATA,             0x400273FC,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHDIR,              0x40027400,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHIS,               0x40027404,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHIBE,              0x40027408,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHIEV,              0x4002740C,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHIM,               0x40027410,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHRIS,              0x40027414,__READ       ,__gpioh_bits);
__IO_REG32_BIT(GPIOHMIS,              0x40027418,__READ       ,__gpioh_bits);
__IO_REG32(    GPIOHICR,              0x4002741C,__WRITE      );
__IO_REG32_BIT(GPIOHAFSEL,            0x40027420,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHODR,              0x4002750C,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHPUR,              0x40027510,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHDEN,              0x4002751C,__READ_WRITE ,__gpioh_bits);
__IO_REG32(    GPIOHLOCK,             0x40027520,__READ_WRITE );
__IO_REG32_BIT(GPIOHCR,               0x40027524,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHAMSEL,            0x40027528,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHPCTL,             0x4002752C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOHAPSEL,            0x40027530,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOHCSEL,             0x40027534,__READ_WRITE ,__gpioh_bits);
__IO_REG8(     GPIOHPERIPHID4,        0x40027FD0,__READ);
__IO_REG8(     GPIOHPERIPHID5,        0x40027FD4,__READ);
__IO_REG8(     GPIOHPERIPHID6,        0x40027FD8,__READ);
__IO_REG8(     GPIOHPERIPHID7,        0x40027FDC,__READ);
__IO_REG8(     GPIOHPERIPHID0,        0x40027FE0,__READ);
__IO_REG8(     GPIOHPERIPHID1,        0x40027FE4,__READ);
__IO_REG8(     GPIOHPERIPHID2,        0x40027FE8,__READ);
__IO_REG8(     GPIOHPERIPHID3,        0x40027FEC,__READ);
__IO_REG8(     GPIOHPCELLID0,         0x40027FF0,__READ);
__IO_REG8(     GPIOHPCELLID1,         0x40027FF4,__READ);
__IO_REG8(     GPIOHPCELLID2,         0x40027FF8,__READ);
__IO_REG8(     GPIOHPCELLID3,         0x40027FFC,__READ);

/***************************************************************************
 **
 ** GPIOH AHB
 **
 ***************************************************************************/
#define GPIOH_AHB_DATA_BASE,          0x4005F000
__IO_REG32_BIT(GPIOH_AHB_DATA,        0x4005F3FC,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_DIR,         0x4005F400,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_IS,          0x4005F404,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_IBE,         0x4005F408,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_IEV,         0x4005F40C,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_IM,          0x4005F410,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_RIS,         0x4005F414,__READ       ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_MIS,         0x4005F418,__READ       ,__gpioh_bits);
__IO_REG32(    GPIOH_AHB_ICR,         0x4005F41C,__WRITE      );
__IO_REG32_BIT(GPIOH_AHB_AFSEL,       0x4005F420,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_ODR,         0x4005F50C,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_PUR,         0x4005F510,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_DEN,         0x4005F51C,__READ_WRITE ,__gpioh_bits);
__IO_REG32(    GPIOH_AHB_LOCK,        0x4005F520,__READ_WRITE );
__IO_REG32_BIT(GPIOH_AHB_CR,          0x4005F524,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_AMSEL,       0x4005F528,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_PCTL,        0x4005F52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOH_AHB_APSEL,       0x4005F530,__READ_WRITE ,__gpioh_bits);
__IO_REG32_BIT(GPIOH_AHB_CSEL,        0x4005F534,__READ_WRITE ,__gpioh_bits);
__IO_REG8(     GPIOH_AHB_PERIPHID4,   0x4005FFD0,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID5,   0x4005FFD4,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID6,   0x4005FFD8,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID7,   0x4005FFDC,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID0,   0x4005FFE0,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID1,   0x4005FFE4,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID2,   0x4005FFE8,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID3,   0x4005FFEC,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID0,    0x4005FFF0,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID1,    0x4005FFF4,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID2,    0x4005FFF8,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID3,    0x4005FFFC,__READ);

/***************************************************************************
 **
 ** GPIOJ APB
 **
 ***************************************************************************/
#define GPIOJDATA_BASE,               0x4003D000
__IO_REG32_BIT(GPIOJDATA,             0x4003D3FC,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJDIR,              0x4003D400,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJIS,               0x4003D404,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJIBE,              0x4003D408,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJIEV,              0x4003D40C,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJIM,               0x4003D410,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJRIS,              0x4003D414,__READ       ,__gpioj_bits);
__IO_REG32_BIT(GPIOJMIS,              0x4003D418,__READ       ,__gpioj_bits);
__IO_REG32(    GPIOJICR,              0x4003D41C,__WRITE      );
__IO_REG32_BIT(GPIOJAFSEL,            0x4003D420,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJODR,              0x4003D50C,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJPUR,              0x4003D510,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJDEN,              0x4003D51C,__READ_WRITE ,__gpioj_bits);
__IO_REG32(    GPIOJLOCK,             0x4003D520,__READ_WRITE );
__IO_REG32_BIT(GPIOJCR,               0x4003D524,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJAMSEL,            0x4003D528,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJPCTL,             0x4003D52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOJAPSEL,            0x4003D530,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJCSEL,             0x4003D534,__READ_WRITE ,__gpioj_bits);
__IO_REG8(     GPIOJPERIPHID4,        0x4003DFD0,__READ);
__IO_REG8(     GPIOJPERIPHID5,        0x4003DFD4,__READ);
__IO_REG8(     GPIOJPERIPHID6,        0x4003DFD8,__READ);
__IO_REG8(     GPIOJPERIPHID7,        0x4003DFDC,__READ);
__IO_REG8(     GPIOJPERIPHID0,        0x4003DFE0,__READ);
__IO_REG8(     GPIOJPERIPHID1,        0x4003DFE4,__READ);
__IO_REG8(     GPIOJPERIPHID2,        0x4003DFE8,__READ);
__IO_REG8(     GPIOJPERIPHID3,        0x4003DFEC,__READ);
__IO_REG8(     GPIOJPCELLID0,         0x4003DFF0,__READ);
__IO_REG8(     GPIOJPCELLID1,         0x4003DFF4,__READ);
__IO_REG8(     GPIOJPCELLID2,         0x4003DFF8,__READ);
__IO_REG8(     GPIOJPCELLID3,         0x4003DFFC,__READ);

/***************************************************************************
 **
 ** GPIOJ AHB
 **
 ***************************************************************************/
#define GPIOJ_AHB_DATA_BASE,          0x40060000
__IO_REG32_BIT(GPIOJ_AHB_DATA,        0x400603FC,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_DIR,         0x40060400,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_IS,          0x40060404,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_IBE,         0x40060408,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_IEV,         0x4006040C,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_IM,          0x40060410,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_RIS,         0x40060414,__READ       ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_MIS,         0x40060418,__READ       ,__gpioj_bits);
__IO_REG32(    GPIOJ_AHB_ICR,         0x4006041C,__WRITE      );
__IO_REG32_BIT(GPIOJ_AHB_AFSEL,       0x40060420,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_ODR,         0x4006050C,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_PUR,         0x40060510,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_DEN,         0x4006051C,__READ_WRITE ,__gpioj_bits);
__IO_REG32(    GPIOJ_AHB_LOCK,        0x40060520,__READ_WRITE );
__IO_REG32_BIT(GPIOJ_AHB_CR,          0x40060524,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_AMSEL,       0x40060528,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_PCTL,        0x4006052C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG32_BIT(GPIOJ_AHB_APSEL,       0x40060530,__READ_WRITE ,__gpioj_bits);
__IO_REG32_BIT(GPIOJ_AHB_CSEL,        0x40060534,__READ_WRITE ,__gpioj_bits);
__IO_REG8(     GPIOJ_AHB_PERIPHID4,   0x40060FD0,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID5,   0x40060FD4,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID6,   0x40060FD8,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID7,   0x40060FDC,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID0,   0x40060FE0,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID1,   0x40060FE4,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID2,   0x40060FE8,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID3,   0x40060FEC,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID0,    0x40060FF0,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID1,    0x40060FF4,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID2,    0x40060FF8,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID3,    0x40060FFC,__READ);

/***************************************************************************
 **
 ** RAM Control
 **
 ***************************************************************************/
__IO_REG32_BIT(CxDRCR1,               0x400FB200,__READ_WRITE ,__cxdrcr1_bits);
__IO_REG32_BIT(CxSRCR1,               0x400FB208,__READ_WRITE ,__cxsrcr1_bits);
__IO_REG32_BIT(MSxMSEL,               0x400FB210,__READ_WRITE ,__msxmsel_bits);
__IO_REG32_BIT(MSxSRCR1,              0x400FB220,__READ_WRITE ,__msxsrcr1_bits);
__IO_REG32_BIT(MSxSRCR2,              0x400FB224,__READ_WRITE ,__msxsrcr2_bits);
__IO_REG32_BIT(MTOCMSGRCR,            0x400FB230,__READ_WRITE ,__mtocmsgrcr_bits);
__IO_REG32_BIT(CxRTESTINIT1,          0x400FB240,__READ_WRITE ,__cxrtestinit1_bits);
__IO_REG32_BIT(MSxRTESTINIT1,         0x400FB250,__READ_WRITE ,__msxrtestinit1_bits);
__IO_REG32_BIT(MTOCRTESTINIT,         0x400FB260,__READ_WRITE ,__mtocrtestinit_bits);
__IO_REG32_BIT(CxRINITDONE1,          0x400FB270,__READ       ,__cxrinitdone1_bits);
__IO_REG32_BIT(MSxRINITDONE1,         0x400FB278,__READ_WRITE ,__msxrinitdone1_bits);
__IO_REG32_BIT(MTOCRINITDONE,         0x400FB288,__READ_WRITE ,__mtocrinitdone_bits);

/***************************************************************************
 **
 ** RAM Error
 **
 ***************************************************************************/
__IO_REG32(    MCUNCWEADDR,           0x400FB300,__READ       );
__IO_REG32(    MDUNCWEADDR,           0x400FB304,__READ       );
__IO_REG32(    MCUNCREADDR,           0x400FB308,__READ       );
__IO_REG32(    MDUNCREADDR,           0x400FB30C,__READ       );
__IO_REG32(    MCPUCREADDR,           0x400FB310,__READ       );
__IO_REG32(    MDMACREADDR,           0x400FB314,__READ       );
__IO_REG32_BIT(MUEFLG,                0x400FB320,__READ       ,__mueflg_bits);
__IO_REG32_BIT(MUEFRC,                0x400FB324,__READ_WRITE ,__mueflg_bits);
__IO_REG32_BIT(MUECLR,                0x400FB328,__READ_WRITE ,__mueflg_bits);
__IO_REG32_BIT(MCECNTR,               0x400FB32C,__READ_WRITE ,__mcecntr_bits);
__IO_REG32_BIT(MCETRES,               0x400FB330,__READ_WRITE ,__mcetres_bits);
__IO_REG32_BIT(MCEFLG,                0x400FB338,__READ       ,__mceflg_bits);
__IO_REG32_BIT(MCEFRC,                0x400FB33C,__READ_WRITE ,__mcefrc_bits);
__IO_REG32_BIT(MCECLR,                0x400FB340,__READ_WRITE ,__mceclr_bits);
__IO_REG32_BIT(MCEIE,                 0x400FB344,__READ_WRITE ,__mceie_bits);
__IO_REG32_BIT(MNMAVFLG,              0x400FB350,__READ       ,__mnmavflg_bits);
__IO_REG32_BIT(MNMAVCLR,              0x400FB358,__READ_WRITE ,__mnmavflg_bits);
__IO_REG32_BIT(MMAVFLG,               0x400FB360,__READ       ,__mmavflg_bits);
__IO_REG32_BIT(MMAVCLR,               0x400FB368,__READ_WRITE ,__mmavflg_bits);
__IO_REG32(    MNMWRAVADDR,           0x400FB370,__READ       );
__IO_REG32(    MNMDMAWRAVADDR,        0x400FB374,__READ       );
__IO_REG32(    MNMFAVADDR,            0x400FB378,__READ       );
__IO_REG32(    MMWRAVADDR,            0x400FB380,__READ       );
__IO_REG32(    MMDMAWRAVADDR,         0x400FB384,__READ       );
__IO_REG32(    MMFAVADDR,             0x400FB388,__READ       );

/***************************************************************************
 **
 ** Flash Control
 **
 ***************************************************************************/
__IO_REG32_BIT(FRDCNTL,               0x400FA000,__READ_WRITE ,__frdcntl_bits);
__IO_REG32_BIT(FSPRD,                 0x400FA004,__READ_WRITE ,__fsprd_bits);
__IO_REG32_BIT(FBAC,                  0x400FA03C,__READ_WRITE ,__fbac_bits);
__IO_REG32_BIT(FBFALLBACK,            0x400FA040,__READ_WRITE ,__fbfallback_bits);
__IO_REG32_BIT(FBPRDY,                0x400FA044,__READ       ,__fbprdy_bits);
__IO_REG32_BIT(FPAC1,                 0x400FA048,__READ_WRITE ,__fpac1_bits);
__IO_REG32_BIT(FPAC2,                 0x400FA04C,__READ_WRITE ,__fpac2_bits);
__IO_REG32_BIT(FMAC,                  0x400FA050,__READ       ,__fmac_bits);
__IO_REG32_BIT(FMSTAT,                0x400FA054,__READ       ,__fmstat_bits);
__IO_REG32_BIT(SECZONEREQUEST,        0x400FA160,__READ_WRITE ,__seczonerequest_bits);
__IO_REG32_BIT(FRD_INTF_CTRL,         0x400FA300,__READ_WRITE ,__frd_intf_ctrl_bits);

/***************************************************************************
 **
 ** Flash Error
 **
 ***************************************************************************/
__IO_REG32_BIT(ECC_ENABLE,            0x400FA600,__READ_WRITE ,__ecc_enable_bits);
__IO_REG32(    SINGLE_ERR_ADDR,       0x400FA604,__READ       );
__IO_REG32(    UNC_ERR_ADDR,          0x400FA608,__READ       );
__IO_REG32_BIT(ERR_STATUS,            0x400FA60C,__READ       ,__err_status_bits);
__IO_REG32_BIT(ERR_POS,               0x400FA610,__READ       ,__err_pos_bits);
__IO_REG32_BIT(ERR_STATUS_CLR,        0x400FA614,__READ_WRITE ,__err_status_clr_bits);
__IO_REG32_BIT(ERR_CNT,               0x400FA618,__READ       ,__err_cnt_bits);
__IO_REG32_BIT(ERR_THRESHOLD,         0x400FA61C,__READ_WRITE ,__err_threshold_bits);
__IO_REG32_BIT(ERR_INTFLG,            0x400FA620,__READ       ,__err_intflg_bits);
__IO_REG32_BIT(ERR_INTCLR,            0x400FA624,__READ_WRITE ,__err_intclr_bits);
__IO_REG32(    FDATAH_TEST,           0x400FA628,__READ_WRITE );
__IO_REG32(    FDATAL_TEST,           0x400FA62C,__READ_WRITE );
__IO_REG32_BIT(FADDR_TEST,            0x400FA630,__READ_WRITE ,__faddr_test_bits);
__IO_REG32_BIT(FECC_TEST,             0x400FA634,__READ_WRITE ,__fecc_test_bits);
__IO_REG32_BIT(FECC_CTRL,             0x400FA638,__READ_WRITE ,__fecc_ctrl_bits);
__IO_REG32(    FECC_FOUTH_TEST,       0x400FA63C,__READ       );
__IO_REG32(    FECC_FOUTL_TEST,       0x400FA640,__READ       );
__IO_REG32_BIT(FECC_STATUS,           0x400FA644,__READ       ,__fecc_status_bits);

/***************************************************************************
 **
 ** ADC1 Result
 **
 ***************************************************************************/
__IO_REG16_BIT(ADC1RESULT0,           0x50001600,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT1,           0x50001602,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT2,           0x50001604,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT3,           0x50001606,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT4,           0x50001608,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT5,           0x5000160A,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT6,           0x5000160C,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT7,           0x5000160E,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT8,           0x50001610,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT9,           0x50001612,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT10,          0x50001614,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT11,          0x50001616,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT12,          0x50001618,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT13,          0x5000161A,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT14,          0x5000161C,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC1RESULT15,          0x5000161E,__READ       ,__adcresult_bits);

/***************************************************************************
 **
 ** ADC2 Result
 **
 ***************************************************************************/
__IO_REG16_BIT(ADC2RESULT0,           0x50001680,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT1,           0x50001682,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT2,           0x50001684,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT3,           0x50001686,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT4,           0x50001688,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT5,           0x5000168A,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT6,           0x5000168C,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT7,           0x5000168E,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT8,           0x50001690,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT9,           0x50001692,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT10,          0x50001694,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT11,          0x50001696,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT12,          0x50001698,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT13,          0x5000169A,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT14,          0x5000169C,__READ       ,__adcresult_bits);
__IO_REG16_BIT(ADC2RESULT15,          0x5000169E,__READ       ,__adcresult_bits);

/***************************************************************************
 **
 ** uDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMASTAT,               0x400FF000,__READ       ,__dmastat_bits);
__IO_REG32(    DMACFG,                0x400FF004,__WRITE      );
__IO_REG32(    DMACTLBASE,            0x400FF008,__READ_WRITE );
__IO_REG32(    DMAALTBASE,            0x400FF00C,__READ       );
__IO_REG32_BIT(DMAWAITSTAT,           0x400FF010,__READ       ,__dmawaitstat_bits);
__IO_REG32(    DMASWREQ,              0x400FF014,__WRITE      );
__IO_REG32_BIT(DMAUSEBURSTSET,        0x400FF018,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32(    DMAUSEBURSTCLR,        0x400FF01C,__WRITE      );
__IO_REG32_BIT(DMAREQMASKSET,         0x400FF020,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32(    DMAREQMASKCLR,         0x400FF024,__WRITE      );
__IO_REG32_BIT(DMAENASET,             0x400FF028,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32(    DMAENACLR,             0x400FF02C,__WRITE      );
__IO_REG32_BIT(DMAALTSET,             0x400FF030,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32(    DMAALTCLR,             0x400FF034,__WRITE      );
__IO_REG32_BIT(DMAPRIOSET,            0x400FF038,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32(    DMAPRIOCLR,            0x400FF03C,__WRITE      );
__IO_REG32_BIT(DMAERRCLR,             0x400FF04C,__READ_WRITE ,__dmaerrclr_bits);
__IO_REG32_BIT(DMACHALT,              0x400FF500,__READ_WRITE ,__dmachalt_bits);
__IO_REG32_BIT(DMACHMAP0,             0x400FF510,__READ_WRITE ,__dmachmap0_bits);
__IO_REG32_BIT(DMACHMAP1,             0x400FF514,__READ_WRITE ,__dmachmap1_bits);
__IO_REG32_BIT(DMACHMAP2,             0x400FF518,__READ_WRITE ,__dmachmap2_bits);
__IO_REG32_BIT(DMACHMAP3,             0x400FF51C,__READ_WRITE ,__dmachmap3_bits);
__IO_REG8(     DMAPeriphID4,          0x400FFFD0,__READ       );
__IO_REG8(     DMAPeriphID0,          0x400FFFE0,__READ       );
__IO_REG8(     DMAPeriphID1,          0x400FFFE4,__READ       );
__IO_REG8(     DMAPeriphID2,          0x400FFFE8,__READ       );
__IO_REG8(     DMAPeriphID3,          0x400FFFEC,__READ       );
__IO_REG8(     DMAPCellID0,           0x400FFFF0,__READ       );
__IO_REG8(     DMAPCellID1,           0x400FFFF4,__READ       );
__IO_REG8(     DMAPCellID2,           0x400FFFF8,__READ       );
__IO_REG8(     DMAPCellID3,           0x400FFFFC,__READ       );

/***************************************************************************
 **
 ** EPI
 **
 ***************************************************************************/
__IO_REG32_BIT(EPICFG,                0x400D0000,__READ_WRITE ,__epicfg_bits);
__IO_REG32_BIT(EPIBAUD,               0x400D0004,__READ_WRITE ,__epibaud_bits);
__IO_REG32_BIT(EPISDRAMCFG,           0x400D0010,__READ_WRITE ,__episdramcfg_bits);
#define EPIHB8CFG       EPISDRAMCFG
#define EPIHB8CFG_bit   EPISDRAMCFG_bit.__epihb8cfg
#define EPIHB16CFG      EPISDRAMCFG
#define EPIHB16CFG_bit  EPISDRAMCFG_bit.__epihb16cfg
#define EPIGPCFG        EPISDRAMCFG
#define EPIGPCFG_bit    EPISDRAMCFG_bit.__epigpcfg
__IO_REG32_BIT(EPIHB8CFG2,            0x400D0014,__READ_WRITE ,__epihb8cfg2_bits);
#define EPIHB16CFG2     EPIHB8CFG2
#define EPIHB16CFG2_bit EPIHB8CFG2_bit
#define EPIGPCFG2       EPIHB8CFG2
#define EPIGPCFG2_bit   EPIHB8CFG2_bit.__epigpcfg2
__IO_REG32_BIT(EPIADDRMAP,            0x400D001C,__READ_WRITE ,__epiaddrmap_bits);
__IO_REG32_BIT(EPIRSIZE0,             0x400D0020,__READ_WRITE ,__epirsize_bits);
__IO_REG32_BIT(EPIRADDR0,             0x400D0024,__READ_WRITE ,__epiraddr_bits);
__IO_REG32_BIT(EPIRPSTD0,             0x400D0028,__READ_WRITE ,__epirpstd_bits);
__IO_REG32_BIT(EPIRSIZE1,             0x400D0030,__READ_WRITE ,__epirsize_bits);
__IO_REG32_BIT(EPIRADDR1,             0x400D0034,__READ_WRITE ,__epiraddr_bits);
__IO_REG32_BIT(EPIRPSTD1,             0x400D0038,__READ_WRITE ,__epirpstd_bits);
__IO_REG32_BIT(EPISTAT,               0x400D0060,__READ       ,__epistat_bits);
__IO_REG32_BIT(EPIRFIFOCNT,           0x400D006C,__READ       ,__epirfifocnt_bits);
__IO_REG32(    EPIREADFIFO,           0x400D0070,__READ       );
__IO_REG32(    EPIREADFIFO1,          0x400D0074,__READ       );
__IO_REG32(    EPIREADFIFO2,          0x400D0078,__READ       );
__IO_REG32(    EPIREADFIFO3,          0x400D007C,__READ       );
__IO_REG32(    EPIREADFIFO4,          0x400D0080,__READ       );
__IO_REG32(    EPIREADFIFO5,          0x400D0084,__READ       );
__IO_REG32(    EPIREADFIFO6,          0x400D0088,__READ       );
__IO_REG32(    EPIREADFIFO7,          0x400D008C,__READ       );
__IO_REG32_BIT(EPIFIFOLVL,            0x400D0200,__READ_WRITE ,__epififolvl_bits);
__IO_REG32_BIT(EPIWFIFOCNT,           0x400D0204,__READ       ,__epiwfifocnt_bits);
__IO_REG32_BIT(EPIIM,                 0x400D0210,__READ_WRITE ,__epiim_bits);
__IO_REG32_BIT(EPIRIS,                0x400D0214,__READ       ,__epiris_bits);
__IO_REG32_BIT(EPIMIS,                0x400D0218,__READ       ,__epimis_bits);
__IO_REG32_BIT(EPIEISC,               0x400D021C,__READ_WRITE ,__epieisc_bits);

/***************************************************************************
 **
 ** USB
 **
 ***************************************************************************/
__IO_REG8_BIT( USBFADDR,              0x40050000,__READ_WRITE ,__usbfaddr_bits);
__IO_REG8_BIT( USBPOWER,              0x40050001,__READ_WRITE ,__usbpower_bits);
__IO_REG16_BIT(USBTXIS,               0x40050002,__READ       ,__usbtxis_bits);
__IO_REG16_BIT(USBRXIS,               0x40050004,__READ       ,__usbrxis_bits);
__IO_REG16_BIT(USBTXIE,               0x40050006,__READ_WRITE ,__usbtxie_bits);
__IO_REG16_BIT(USBRXIE,               0x40050008,__READ_WRITE ,__usbrxie_bits);
__IO_REG8_BIT( USBIS,                 0x4005000A,__READ       ,__usbis_bits);
__IO_REG8_BIT( USBIE,                 0x4005000B,__READ_WRITE ,__usbie_bits);
__IO_REG16_BIT(USBFRAME,              0x4005000C,__READ       ,__usbframe_bits);
__IO_REG8_BIT( USBEPIDX,              0x4005000E,__READ_WRITE ,__usbepidx_bits);
__IO_REG8_BIT( USBTEST,               0x4005000F,__READ_WRITE ,__usbtest_bits);
__IO_REG32(    USBFIFO0,              0x40050020,__READ_WRITE );
__IO_REG32(    USBFIFO1,              0x40050024,__READ_WRITE );
__IO_REG32(    USBFIFO2,              0x40050028,__READ_WRITE );
__IO_REG32(    USBFIFO3,              0x4005002C,__READ_WRITE );
__IO_REG32(    USBFIFO4,              0x40050030,__READ_WRITE );
__IO_REG32(    USBFIFO5,              0x40050034,__READ_WRITE );
__IO_REG32(    USBFIFO6,              0x40050038,__READ_WRITE );
__IO_REG32(    USBFIFO7,              0x4005003C,__READ_WRITE );
__IO_REG32(    USBFIFO8,              0x40050040,__READ_WRITE );
__IO_REG32(    USBFIFO9,              0x40050044,__READ_WRITE );
__IO_REG32(    USBFIFO10,             0x40050048,__READ_WRITE );
__IO_REG32(    USBFIFO11,             0x4005004C,__READ_WRITE );
__IO_REG32(    USBFIFO12,             0x40050050,__READ_WRITE );
__IO_REG32(    USBFIFO13,             0x40050054,__READ_WRITE );
__IO_REG32(    USBFIFO14,             0x40050058,__READ_WRITE );
__IO_REG32(    USBFIFO15,             0x4005005C,__READ_WRITE );
__IO_REG8_BIT( USBDEVCTL,             0x40050060,__READ       ,__usbdevctl_bits);
__IO_REG8_BIT( USBTXFIFOSZ,           0x40050062,__READ_WRITE ,__usbtxfifosz_bits);
__IO_REG8_BIT( USBRXFIFOSZ,           0x40050063,__READ_WRITE ,__usbtxfifosz_bits);
__IO_REG16_BIT(USBTXFIFOADD,          0x40050064,__READ_WRITE ,__usbtxfifoadd_bits);
__IO_REG16_BIT(USBRXFIFOADD,          0x40050066,__READ_WRITE ,__usbtxfifoadd_bits);
__IO_REG8_BIT( USBCONTIM,             0x4005007A,__READ_WRITE ,__usbcontim_bits);
__IO_REG8(     USBVPLEN,              0x4005007B,__READ_WRITE );
__IO_REG8(     USBFSEOF,              0x4005007D,__READ_WRITE );
__IO_REG8(     USBLSEOF,              0x4005007E,__READ_WRITE );
__IO_REG8_BIT( USBTXFUNCADDR0,        0x40050080,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR0,         0x40050082,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT0,         0x40050083,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR1,        0x40050088,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR1,         0x4005008A,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT1,         0x4005008B,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR1,        0x4005008C,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR1,         0x4005008E,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT1,         0x4005008F,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR2,        0x40050090,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR2,         0x40050092,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT2,         0x40050093,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR2,        0x40050094,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR2,         0x40050096,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT2,         0x40050097,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR3,        0x40050098,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR3,         0x4005009A,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT3,         0x4005009B,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR3,        0x4005009C,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR3,         0x4005009E,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT3,         0x4005009F,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR4,        0x400500A0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR4,         0x400500A2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT4,         0x400500A3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR4,        0x400500A4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR4,         0x400500A6,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT4,         0x400500A7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR5,        0x400500A8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR5,         0x400500AA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT5,         0x400500AB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR5,        0x400500AC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR5,         0x400500AE,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT5,         0x400500AF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR6,        0x400500B0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR6,         0x400500B2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT6,         0x400500B3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR6,        0x400500B4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR6,         0x400500B6,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT6,         0x400500B7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR7,        0x400500B8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR7,         0x400500BA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT7,         0x400500BB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR7,        0x400500BC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR7,         0x400500BE,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT7,         0x400500BF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR8,        0x400500C0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR8,         0x400500C2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT8,         0x400500C3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR8,        0x400500C4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR8,         0x400500C6,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT8,         0x400500C7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR9,        0x400500C8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR9,         0x400500CA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT9,         0x400500CB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR9,        0x400500CC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR9,         0x400500CE,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT9,         0x400500CF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR10,       0x400500D0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR10,        0x400500D2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT10,        0x400500D3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR10,       0x400500D4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR10,        0x400500D6,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT10,        0x400500D7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR11,       0x400500D8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR11,        0x400500DA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT11,        0x400500DB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR11,       0x400500DC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR11,        0x400500DE,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT11,        0x400500DF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR12,       0x400500E0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR12,        0x400500E2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT12,        0x400500E3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR12,       0x400500E4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR12,        0x400500E6,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT12,        0x400500E7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR13,       0x400500E8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR13,        0x400500EA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT13,        0x400500EB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR13,       0x400500EC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR13,        0x400500EE,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT13,        0x400500EF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR14,       0x400500F0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR14,        0x400500F2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT14,        0x400500F3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR14,       0x400500F4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR14,        0x400500F6,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT14,        0x400500F7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR15,       0x400500F8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR15,        0x400500FA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT15,        0x400500FB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR15,       0x400500FC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR15,        0x400500FE,__READ_WRITE ,__usbrxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT15,        0x400500FF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBCSRL0,              0x40050102,__READ_WRITE ,__usbcsrl0_bits);
__IO_REG8_BIT( USBCSRH0,              0x40050103,__READ_WRITE ,__usbcsrh0_bits);
__IO_REG8_BIT( USBCOUNT0,             0x40050108,__READ_WRITE ,__usbcount0_bits);
__IO_REG8_BIT( USBTYPE0,              0x4005010A,__READ_WRITE ,__usbtype0_bits);
__IO_REG8_BIT( USBNAKLMT,             0x4005010B,__READ_WRITE ,__usbnaklmt_bits);
__IO_REG16_BIT(USBTXMAXP1,            0x40050110,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL1,            0x40050112,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH1,            0x40050113,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP1,            0x40050114,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL1,            0x40050116,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH1,            0x40050117,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT1,           0x40050118,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE1,            0x4005011A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL1,        0x4005011B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE1,            0x4005011C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL1,        0x4005011D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP2,            0x40050120,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL2,            0x40050122,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH2,            0x40050123,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP2,            0x40050124,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL2,            0x40050126,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH2,            0x40050127,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT2,           0x40050128,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE2,            0x4005012A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL2,        0x4005012B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE2,            0x4005012C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL2,        0x4005012D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP3,            0x40050130,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL3,            0x40050132,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH3,            0x40050133,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP3,            0x40050134,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL3,            0x40050136,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH3,            0x40050137,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT3,           0x40050138,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE3,            0x4005013A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL3,        0x4005013B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE3,            0x4005013C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL3,        0x4005013D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP4,            0x40050140,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL4,            0x40050142,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH4,            0x40050143,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP4,            0x40050144,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL4,            0x40050146,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH4,            0x40050147,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT4,           0x40050148,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE4,            0x4005014A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL4,        0x4005014B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE4,            0x4005014C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL4,        0x4005014D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP5,            0x40050150,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL5,            0x40050152,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH5,            0x40050153,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP5,            0x40050154,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL5,            0x40050156,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH5,            0x40050157,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT5,           0x40050158,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE5,            0x4005015A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL5,        0x4005015B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE5,            0x4005015C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL5,        0x4005015D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP6,            0x40050160,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL6,            0x40050162,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH6,            0x40050163,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP6,            0x40050164,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL6,            0x40050166,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH6,            0x40050167,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT6,           0x40050168,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE6,            0x4005016A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL6,        0x4005016B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE6,            0x4005016C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL6,        0x4005016D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP7,            0x40050170,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL7,            0x40050172,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH7,            0x40050173,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP7,            0x40050174,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL7,            0x40050176,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH7,            0x40050177,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT7,           0x40050178,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE7,            0x4005017A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL7,        0x4005017B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE7,            0x4005017C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL7,        0x4005017D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP8,            0x40050180,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL8,            0x40050182,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH8,            0x40050183,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP8,            0x40050184,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL8,            0x40050186,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH8,            0x40050187,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT8,           0x40050188,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE8,            0x4005018A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL8,        0x4005018B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE8,            0x4005018C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL8,        0x4005018D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP9,            0x40050190,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL9,            0x40050192,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH9,            0x40050193,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP9,            0x40050194,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL9,            0x40050196,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH9,            0x40050197,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT9,           0x40050198,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE9,            0x4005019A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL9,        0x4005019B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE9,            0x4005019C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL9,        0x4005019D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP10,           0x400501A0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL10,           0x400501A2,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH10,           0x400501A3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP10,           0x400501A4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL10,           0x400501A6,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH10,           0x400501A7,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT10,          0x400501A8,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE10,           0x400501AA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL10,       0x400501AB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE10,           0x400501AC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL10,       0x400501AD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP11,           0x400501B0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL11,           0x400501B2,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH11,           0x400501B3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP11,           0x400501B4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL11,           0x400501B6,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH11,           0x400501B7,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT11,          0x400501B8,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE11,           0x400501BA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL11,       0x400501BB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE11,           0x400501BC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL11,       0x400501BD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP12,           0x400501C0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL12,           0x400501C2,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH12,           0x400501C3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP12,           0x400501C4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL12,           0x400501C6,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH12,           0x400501C7,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT12,          0x400501C8,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE12,           0x400501CA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL12,       0x400501CB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE12,           0x400501CC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL12,       0x400501CD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP13,           0x400501D0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL13,           0x400501D2,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH13,           0x400501D3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP13,           0x400501D4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL13,           0x400501D6,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH13,           0x400501D7,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT13,          0x400501D8,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE13,           0x400501DA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL13,       0x400501DB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE13,           0x400501DC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL13,       0x400501DD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP14,           0x400501E0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL14,           0x400501E2,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH14,           0x400501E3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP14,           0x400501E4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL14,           0x400501E6,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH14,           0x400501E7,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT14,          0x400501E8,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE14,           0x400501EA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL14,       0x400501EB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE14,           0x400501EC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL14,       0x400501ED,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP15,           0x400501F0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL15,           0x400501F2,__READ_WRITE ,__usbtxcsrl_bits);
__IO_REG8_BIT( USBTXCSRH15,           0x400501F3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP15,           0x400501F4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL15,           0x400501F6,__READ_WRITE ,__usbrxcsrl_bits);
__IO_REG8_BIT( USBRXCSRH15,           0x400501F7,__READ_WRITE ,__usbrxcsrh_bits);
__IO_REG16_BIT(USBRXCOUNT15,          0x400501F8,__READ       ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE15,           0x400501FA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL15,       0x400501FB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE15,           0x400501FC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL15,       0x400501FD,__READ_WRITE );
__IO_REG16_BIT(USBRQPKTCOUNT1,        0x40050304,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT2,        0x40050308,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT3,        0x4005030C,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT4,        0x40050310,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT5,        0x40050314,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT6,        0x40050318,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT7,        0x4005031C,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT8,        0x40050320,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT9,        0x40050324,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT10,       0x40050328,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT11,       0x4005032C,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT12,       0x40050330,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT13,       0x40050334,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT14,       0x40050338,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRQPKTCOUNT15,       0x4005033C,__READ_WRITE ,__usbrqpktcount_bits);
__IO_REG16_BIT(USBRXDPKTBUFDIS,       0x40050340,__READ_WRITE ,__usbrxdpktbufdis_bits);
__IO_REG16_BIT(USBTXDPKTBUFDIS,       0x40050342,__READ_WRITE ,__usbrxdpktbufdis_bits);
__IO_REG32_BIT(USBEPC,                0x40050400,__READ_WRITE ,__usbepc_bits);
__IO_REG32_BIT(USBEPCRIS,             0x40050404,__READ       ,__usbepcris_bits);
__IO_REG32_BIT(USBEPCIM,              0x40050408,__READ_WRITE ,__usbepcris_bits);
__IO_REG32_BIT(USBEPCISC,             0x4005040C,__READ_WRITE ,__usbepcris_bits);
__IO_REG32_BIT(USBDRRIS,              0x40050410,__READ       ,__usbdrris_bits);
__IO_REG32_BIT(USBDRIM,               0x40050414,__READ_WRITE ,__usbdrris_bits);
__IO_REG32_BIT(USBDRISC,              0x40050418,__READ_WRITE ,__usbdrris_bits);
__IO_REG32_BIT(USBGPCS,               0x4005041C,__READ_WRITE ,__usbgpcs_bits);
__IO_REG32_BIT(USBVDC,                0x40050430,__READ_WRITE ,__usbvdc_bits);
__IO_REG32_BIT(USBVDCRIS,             0x40050434,__READ       ,__usbvdcris_bits);
__IO_REG32_BIT(USBVDCIM,              0x40050438,__READ_WRITE ,__usbvdcris_bits);
__IO_REG32_BIT(USBVDCISC,             0x4005043C,__READ_WRITE ,__usbvdcris_bits);
__IO_REG32_BIT(USBIDVRIS,             0x40050444,__READ       ,__usbidvris_bits);
__IO_REG32_BIT(USBIDVIM,              0x40050448,__READ_WRITE ,__usbidvris_bits);
__IO_REG32_BIT(USBIDVISC,             0x4005044C,__READ_WRITE ,__usbidvris_bits);
__IO_REG32_BIT(USBDMASEL,             0x40050450,__READ_WRITE ,__usbdmasel_bits);

/***************************************************************************
 **
 ** Ethernet MAC
 **
 ***************************************************************************/
__IO_REG32_BIT(MACRIS,                0x40048000,__READ_WRITE  ,__macris_bits);
#define MACIACK MACRIS    
#define MACIACK_bit MACRIS_bit    
__IO_REG32_BIT(MACIM,                 0x40048004,__READ_WRITE  ,__macim_bits);
__IO_REG32_BIT(MACRCTL,               0x40048008,__READ_WRITE  ,__macrctl_bits);
__IO_REG32_BIT(MACTCTL,               0x4004800C,__READ_WRITE  ,__mactctl_bits);
__IO_REG32_BIT(MACDATA,               0x40048010,__READ_WRITE  ,__macdata_bits);
__IO_REG32_BIT(MACIA0,                0x40048014,__READ_WRITE  ,__macia0_bits);
__IO_REG32_BIT(MACIA1,                0x40048018,__READ_WRITE  ,__macia1_bits);
__IO_REG32_BIT(MACTHR,                0x4004801C,__READ_WRITE  ,__macthr_bits);
__IO_REG32_BIT(MACMCTL,               0x40048020,__READ_WRITE  ,__macmctl_bits);
__IO_REG32_BIT(MACMDV,                0x40048024,__READ_WRITE  ,__macmdv_bits);
__IO_REG32_BIT(MACMTXD,               0x4004802C,__READ_WRITE  ,__macmtxd_bits);
__IO_REG32_BIT(MACMRXD,               0x40048030,__READ_WRITE  ,__macmrxd_bits);
__IO_REG32_BIT(MACNP,                 0x40048034,__READ        ,__macnp_bits);
__IO_REG32_BIT(MACTR,                 0x40048038,__READ_WRITE  ,__mactr_bits);
__IO_REG32_BIT(MACTS,                 0x4004803C,__READ_WRITE  ,__macts_bits);

/***************************************************************************
 **
 ** SSI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSI0CR0,               0x40008000,__READ_WRITE ,__ssicr0_bits);
__IO_REG32_BIT(SSI0CR1,               0x40008004,__READ_WRITE ,__ssicr1_bits);
__IO_REG32_BIT(SSI0DR,                0x40008008,__READ_WRITE ,__ssidr_bits);
__IO_REG32_BIT(SSI0SR,                0x4000800C,__READ       ,__ssisr_bits);
__IO_REG32_BIT(SSI0CPSR,              0x40008010,__READ_WRITE ,__ssicpsr_bits);
__IO_REG32_BIT(SSI0IM,                0x40008014,__READ_WRITE ,__ssiim_bits);
__IO_REG32_BIT(SSI0RIS,               0x40008018,__READ       ,__ssiris_bits);
__IO_REG32_BIT(SSI0MIS,               0x4000801C,__READ       ,__ssimis_bits);
__IO_REG32(    SSI0ICR,               0x40008020,__WRITE      );
__IO_REG32_BIT(SSI0DMACTL,            0x40008024,__READ_WRITE ,__ssidmactl_bits);
__IO_REG8(     SSI0PERIPHID4,         0x40008FD0,__READ);
__IO_REG8(     SSI0PERIPHID5,         0x40008FD4,__READ);
__IO_REG8(     SSI0PERIPHID6,         0x40008FD8,__READ);
__IO_REG8(     SSI0PERIPHID7,         0x40008FDC,__READ);
__IO_REG8(     SSI0PERIPHID0,         0x40008FE0,__READ);
__IO_REG8(     SSI0PERIPHID1,         0x40008FE4,__READ);
__IO_REG8(     SSI0PERIPHID2,         0x40008FE8,__READ);
__IO_REG8(     SSI0PERIPHID3,         0x40008FEC,__READ);
__IO_REG8(     SSI0PCELLID0,          0x40008FF0,__READ);
__IO_REG8(     SSI0PCELLID1,          0x40008FF4,__READ);
__IO_REG8(     SSI0PCELLID2,          0x40008FF8,__READ);
__IO_REG8(     SSI0PCELLID3,          0x40008FFC,__READ);

/***************************************************************************
 **
 ** SSI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSI1CR0,               0x40009000,__READ_WRITE ,__ssicr0_bits);
__IO_REG32_BIT(SSI1CR1,               0x40009004,__READ_WRITE ,__ssicr1_bits);
__IO_REG32_BIT(SSI1DR,                0x40009008,__READ_WRITE ,__ssidr_bits);
__IO_REG32_BIT(SSI1SR,                0x4000900C,__READ       ,__ssisr_bits);
__IO_REG32_BIT(SSI1CPSR,              0x40009010,__READ_WRITE ,__ssicpsr_bits);
__IO_REG32_BIT(SSI1IM,                0x40009014,__READ_WRITE ,__ssiim_bits);
__IO_REG32_BIT(SSI1RIS,               0x40009018,__READ       ,__ssiris_bits);
__IO_REG32_BIT(SSI1MIS,               0x4000901C,__READ       ,__ssimis_bits);
__IO_REG32(    SSI1ICR,               0x40009020,__WRITE      );
__IO_REG32_BIT(SSI1DMACTL,            0x40009024,__READ_WRITE ,__ssidmactl_bits);
__IO_REG8(     SSI1PERIPHID4,         0x40009FD0,__READ);
__IO_REG8(     SSI1PERIPHID5,         0x40009FD4,__READ);
__IO_REG8(     SSI1PERIPHID6,         0x40009FD8,__READ);
__IO_REG8(     SSI1PERIPHID7,         0x40009FDC,__READ);
__IO_REG8(     SSI1PERIPHID0,         0x40009FE0,__READ);
__IO_REG8(     SSI1PERIPHID1,         0x40009FE4,__READ);
__IO_REG8(     SSI1PERIPHID2,         0x40009FE8,__READ);
__IO_REG8(     SSI1PERIPHID3,         0x40009FEC,__READ);
__IO_REG8(     SSI1PCELLID0,          0x40009FF0,__READ);
__IO_REG8(     SSI1PCELLID1,          0x40009FF4,__READ);
__IO_REG8(     SSI1PCELLID2,          0x40009FF8,__READ);
__IO_REG8(     SSI1PCELLID3,          0x40009FFC,__READ);

/***************************************************************************
 **
 ** SSI2
 **
 ***************************************************************************/
__IO_REG32_BIT(SSI2CR0,               0x4000A000,__READ_WRITE ,__ssicr0_bits);
__IO_REG32_BIT(SSI2CR1,               0x4000A004,__READ_WRITE ,__ssicr1_bits);
__IO_REG32_BIT(SSI2DR,                0x4000A008,__READ_WRITE ,__ssidr_bits);
__IO_REG32_BIT(SSI2SR,                0x4000A00C,__READ       ,__ssisr_bits);
__IO_REG32_BIT(SSI2CPSR,              0x4000A010,__READ_WRITE ,__ssicpsr_bits);
__IO_REG32_BIT(SSI2IM,                0x4000A014,__READ_WRITE ,__ssiim_bits);
__IO_REG32_BIT(SSI2RIS,               0x4000A018,__READ       ,__ssiris_bits);
__IO_REG32_BIT(SSI2MIS,               0x4000A01C,__READ       ,__ssimis_bits);
__IO_REG32(    SSI2ICR,               0x4000A020,__WRITE      );
__IO_REG32_BIT(SSI2DMACTL,            0x4000A024,__READ_WRITE ,__ssidmactl_bits);
__IO_REG8(     SSI2PERIPHID4,         0x4000AFD0,__READ);
__IO_REG8(     SSI2PERIPHID5,         0x4000AFD4,__READ);
__IO_REG8(     SSI2PERIPHID6,         0x4000AFD8,__READ);
__IO_REG8(     SSI2PERIPHID7,         0x4000AFDC,__READ);
__IO_REG8(     SSI2PERIPHID0,         0x4000AFE0,__READ);
__IO_REG8(     SSI2PERIPHID1,         0x4000AFE4,__READ);
__IO_REG8(     SSI2PERIPHID2,         0x4000AFE8,__READ);
__IO_REG8(     SSI2PERIPHID3,         0x4000AFEC,__READ);
__IO_REG8(     SSI2PCELLID0,          0x4000AFF0,__READ);
__IO_REG8(     SSI2PCELLID1,          0x4000AFF4,__READ);
__IO_REG8(     SSI2PCELLID2,          0x4000AFF8,__READ);
__IO_REG8(     SSI2PCELLID3,          0x4000AFFC,__READ);

/***************************************************************************
 **
 ** SSI3
 **
 ***************************************************************************/
__IO_REG32_BIT(SSI3CR0,               0x4000B000,__READ_WRITE ,__ssicr0_bits);
__IO_REG32_BIT(SSI3CR1,               0x4000B004,__READ_WRITE ,__ssicr1_bits);
__IO_REG32_BIT(SSI3DR,                0x4000B008,__READ_WRITE ,__ssidr_bits);
__IO_REG32_BIT(SSI3SR,                0x4000B00C,__READ       ,__ssisr_bits);
__IO_REG32_BIT(SSI3CPSR,              0x4000B010,__READ_WRITE ,__ssicpsr_bits);
__IO_REG32_BIT(SSI3IM,                0x4000B014,__READ_WRITE ,__ssiim_bits);
__IO_REG32_BIT(SSI3RIS,               0x4000B018,__READ       ,__ssiris_bits);
__IO_REG32_BIT(SSI3MIS,               0x4000B01C,__READ       ,__ssimis_bits);
__IO_REG32(    SSI3ICR,               0x4000B020,__WRITE      );
__IO_REG32_BIT(SSI3DMACTL,            0x4000B024,__READ_WRITE ,__ssidmactl_bits);
__IO_REG8(     SSI3PERIPHID4,         0x4000BFD0,__READ);
__IO_REG8(     SSI3PERIPHID5,         0x4000BFD4,__READ);
__IO_REG8(     SSI3PERIPHID6,         0x4000BFD8,__READ);
__IO_REG8(     SSI3PERIPHID7,         0x4000BFDC,__READ);
__IO_REG8(     SSI3PERIPHID0,         0x4000BFE0,__READ);
__IO_REG8(     SSI3PERIPHID1,         0x4000BFE4,__READ);
__IO_REG8(     SSI3PERIPHID2,         0x4000BFE8,__READ);
__IO_REG8(     SSI3PERIPHID3,         0x4000BFEC,__READ);
__IO_REG8(     SSI3PCELLID0,          0x4000BFF0,__READ);
__IO_REG8(     SSI3PCELLID1,          0x4000BFF4,__READ);
__IO_REG8(     SSI3PCELLID2,          0x4000BFF8,__READ);
__IO_REG8(     SSI3PCELLID3,          0x4000BFFC,__READ);

/***************************************************************************
 **
 ** UART 0
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0DR,               0x4000C000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART0RSR,              0x4000C004,__READ_WRITE ,__uartrsr_bits);
#define UART0ECR         UART0RSR
#define UART0ECR_bit     UART0RSR_bit
__IO_REG32_BIT(UART0FR,               0x4000C018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART0ILPR,             0x4000C020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART0IBRD,             0x4000C024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART0FBRD,             0x4000C028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART0LCRH,             0x4000C02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART0CTL,              0x4000C030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART0IFLS,             0x4000C034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART0IM,               0x4000C038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART0RIS,              0x4000C03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART0MIS,              0x4000C040,__READ       ,__uartmis_bits);
__IO_REG32(    UART0ICR,              0x4000C044,__WRITE      );
__IO_REG32_BIT(UART0DMACTL,           0x4000C048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART0LCTL,             0x4000C090,__READ_WRITE ,__uartlctl_bits);
__IO_REG32_BIT(UART0LSS,              0x4000C094,__READ       ,__uartlss_bits);
__IO_REG32_BIT(UART0LTIM,             0x4000C098,__READ       ,__uartltim_bits);
__IO_REG8(     UART0PERIPHID4,        0x4000CFD0,__READ);
__IO_REG8(     UART0PERIPHID5,        0x4000CFD4,__READ);
__IO_REG8(     UART0PERIPHID6,        0x4000CFD8,__READ);
__IO_REG8(     UART0PERIPHID7,        0x4000CFDC,__READ);
__IO_REG8(     UART0PERIPHID0,        0x4000CFE0,__READ);
__IO_REG8(     UART0PERIPHID1,        0x4000CFE4,__READ);
__IO_REG8(     UART0PERIPHID2,        0x4000CFE8,__READ);
__IO_REG8(     UART0PERIPHID3,        0x4000CFEC,__READ);
__IO_REG8(     UART0PCELLID0,         0x4000CFF0,__READ);
__IO_REG8(     UART0PCELLID1,         0x4000CFF4,__READ);
__IO_REG8(     UART0PCELLID2,         0x4000CFF8,__READ);
__IO_REG8(     UART0PCELLID3,         0x4000CFFC,__READ);

/***************************************************************************
 **
 ** UART 1
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1DR,               0x4000D000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART1RSR,              0x4000D004,__READ_WRITE ,__uartrsr_bits);
#define UART1ECR         UART1RSR
#define UART1ECR_bit     UART1RSR_bit
__IO_REG32_BIT(UART1FR,               0x4000D018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART1ILPR,             0x4000D020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART1IBRD,             0x4000D024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART1FBRD,             0x4000D028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART1LCRH,             0x4000D02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART1CTL,              0x4000D030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART1IFLS,             0x4000D034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART1IM,               0x4000D038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART1RIS,              0x4000D03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART1MIS,              0x4000D040,__READ       ,__uartmis_bits);
__IO_REG32(    UART1ICR,              0x4000D044,__WRITE      );
__IO_REG32_BIT(UART1DMACTL,           0x4000D048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART1LCTL,             0x4000D090,__READ_WRITE ,__uartlctl_bits);
__IO_REG32_BIT(UART1LSS,              0x4000D094,__READ       ,__uartlss_bits);
__IO_REG32_BIT(UART1LTIM,             0x4000D098,__READ       ,__uartltim_bits);
__IO_REG8(     UART1PERIPHID4,        0x4000DFD0,__READ);
__IO_REG8(     UART1PERIPHID5,        0x4000DFD4,__READ);
__IO_REG8(     UART1PERIPHID6,        0x4000DFD8,__READ);
__IO_REG8(     UART1PERIPHID7,        0x4000DFDC,__READ);
__IO_REG8(     UART1PERIPHID0,        0x4000DFE0,__READ);
__IO_REG8(     UART1PERIPHID1,        0x4000DFE4,__READ);
__IO_REG8(     UART1PERIPHID2,        0x4000DFE8,__READ);
__IO_REG8(     UART1PERIPHID3,        0x4000DFEC,__READ);
__IO_REG8(     UART1PCELLID0,         0x4000DFF0,__READ);
__IO_REG8(     UART1PCELLID1,         0x4000DFF4,__READ);
__IO_REG8(     UART1PCELLID2,         0x4000DFF8,__READ);
__IO_REG8(     UART1PCELLID3,         0x4000DFFC,__READ);

/***************************************************************************
 **
 ** UART 2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2DR,               0x4000E000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART2RSR,              0x4000E004,__READ_WRITE ,__uartrsr_bits);
#define UART2ECR         UART2RSR
#define UART2ECR_bit     UART2RSR_bit
__IO_REG32_BIT(UART2FR,               0x4000E018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART2ILPR,             0x4000E020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART2IBRD,             0x4000E024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART2FBRD,             0x4000E028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART2LCRH,             0x4000E02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART2CTL,              0x4000E030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART2IFLS,             0x4000E034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART2IM,               0x4000E038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART2RIS,              0x4000E03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART2MIS,              0x4000E040,__READ       ,__uartmis_bits);
__IO_REG32(    UART2ICR,              0x4000E044,__WRITE      );
__IO_REG32_BIT(UART2DMACTL,           0x4000E048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART2LCTL,             0x4000E090,__READ_WRITE ,__uartlctl_bits);
__IO_REG32_BIT(UART2LSS,              0x4000E094,__READ       ,__uartlss_bits);
__IO_REG32_BIT(UART2LTIM,             0x4000E098,__READ       ,__uartltim_bits);
__IO_REG8(     UART2PERIPHID4,        0x4000EFD0,__READ);
__IO_REG8(     UART2PERIPHID5,        0x4000EFD4,__READ);
__IO_REG8(     UART2PERIPHID6,        0x4000EFD8,__READ);
__IO_REG8(     UART2PERIPHID7,        0x4000EFDC,__READ);
__IO_REG8(     UART2PERIPHID0,        0x4000EFE0,__READ);
__IO_REG8(     UART2PERIPHID1,        0x4000EFE4,__READ);
__IO_REG8(     UART2PERIPHID2,        0x4000EFE8,__READ);
__IO_REG8(     UART2PERIPHID3,        0x4000EFEC,__READ);
__IO_REG8(     UART2PCELLID0,         0x4000EFF0,__READ);
__IO_REG8(     UART2PCELLID1,         0x4000EFF4,__READ);
__IO_REG8(     UART2PCELLID2,         0x4000EFF8,__READ);
__IO_REG8(     UART2PCELLID3,         0x4000EFFC,__READ);

/***************************************************************************
 **
 ** UART 3
 **
 ***************************************************************************/
__IO_REG32_BIT(UART3DR,               0x4000F000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART3RSR,              0x4000F004,__READ_WRITE ,__uartrsr_bits);
#define UART3ECR         UART3RSR
#define UART3ECR_bit     UART3RSR_bit
__IO_REG32_BIT(UART3FR,               0x4000F018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART3ILPR,             0x4000F020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART3IBRD,             0x4000F024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART3FBRD,             0x4000F028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART3LCRH,             0x4000F02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART3CTL,              0x4000F030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART3IFLS,             0x4000F034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART3IM,               0x4000F038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART3RIS,              0x4000F03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART3MIS,              0x4000F040,__READ       ,__uartmis_bits);
__IO_REG32(    UART3ICR,              0x4000F044,__WRITE      );
__IO_REG32_BIT(UART3DMACTL,           0x4000F048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART3LCTL,             0x4000F090,__READ_WRITE ,__uartlctl_bits);
__IO_REG32_BIT(UART3LSS,              0x4000F094,__READ       ,__uartlss_bits);
__IO_REG32_BIT(UART3LTIM,             0x4000F098,__READ       ,__uartltim_bits);
__IO_REG8(     UART3PERIPHID4,        0x4000FFD0,__READ);
__IO_REG8(     UART3PERIPHID5,        0x4000FFD4,__READ);
__IO_REG8(     UART3PERIPHID6,        0x4000FFD8,__READ);
__IO_REG8(     UART3PERIPHID7,        0x4000FFDC,__READ);
__IO_REG8(     UART3PERIPHID0,        0x4000FFE0,__READ);
__IO_REG8(     UART3PERIPHID1,        0x4000FFE4,__READ);
__IO_REG8(     UART3PERIPHID2,        0x4000FFE8,__READ);
__IO_REG8(     UART3PERIPHID3,        0x4000FFEC,__READ);
__IO_REG8(     UART3PCELLID0,         0x4000FFF0,__READ);
__IO_REG8(     UART3PCELLID1,         0x4000FFF4,__READ);
__IO_REG8(     UART3PCELLID2,         0x4000FFF8,__READ);
__IO_REG8(     UART3PCELLID3,         0x4000FFFC,__READ);

/***************************************************************************
 **
 ** UART 4
 **
 ***************************************************************************/
__IO_REG32_BIT(UART4DR,               0x40010000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART4RSR,              0x40010004,__READ_WRITE ,__uartrsr_bits);
#define UART4ECR         UART4RSR
#define UART4ECR_bit     UART4RSR_bit
__IO_REG32_BIT(UART4FR,               0x40010018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART4ILPR,             0x40010020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART4IBRD,             0x40010024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART4FBRD,             0x40010028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART4LCRH,             0x4001002C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART4CTL,              0x40010030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART4IFLS,             0x40010034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART4IM,               0x40010038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART4RIS,              0x4001003C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART4MIS,              0x40010040,__READ       ,__uartmis_bits);
__IO_REG32(    UART4ICR,              0x40010044,__WRITE      );
__IO_REG32_BIT(UART4DMACTL,           0x40010048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART4LCTL,             0x40010090,__READ_WRITE ,__uartlctl_bits);
__IO_REG32_BIT(UART4LSS,              0x40010094,__READ       ,__uartlss_bits);
__IO_REG32_BIT(UART4LTIM,             0x40010098,__READ       ,__uartltim_bits);
__IO_REG8(     UART4PERIPHID4,        0x40010FD0,__READ);
__IO_REG8(     UART4PERIPHID5,        0x40010FD4,__READ);
__IO_REG8(     UART4PERIPHID6,        0x40010FD8,__READ);
__IO_REG8(     UART4PERIPHID7,        0x40010FDC,__READ);
__IO_REG8(     UART4PERIPHID0,        0x40010FE0,__READ);
__IO_REG8(     UART4PERIPHID1,        0x40010FE4,__READ);
__IO_REG8(     UART4PERIPHID2,        0x40010FE8,__READ);
__IO_REG8(     UART4PERIPHID3,        0x40010FEC,__READ);
__IO_REG8(     UART4PCELLID0,         0x40010FF0,__READ);
__IO_REG8(     UART4PCELLID1,         0x40010FF4,__READ);
__IO_REG8(     UART4PCELLID2,         0x40010FF8,__READ);
__IO_REG8(     UART4PCELLID3,         0x40010FFC,__READ);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0MSA,               0x40020000,__READ_WRITE ,__i2cmsa_bits);
__IO_REG32_BIT(I2C0MCS,               0x40020004,__READ_WRITE ,__i2cmcs_bits);
__IO_REG32_BIT(I2C0MDR,               0x40020008,__READ_WRITE ,__i2cmdr_bits);
__IO_REG32_BIT(I2C0MTPR,              0x4002000C,__READ_WRITE ,__i2cmtpr_bits);
__IO_REG32_BIT(I2C0MIMR,              0x40020010,__READ_WRITE ,__i2cmimr_bits);
__IO_REG32_BIT(I2C0MRIS,              0x40020014,__READ       ,__i2cmris_bits);
__IO_REG32_BIT(I2C0MMIS,              0x40020018,__READ       ,__i2cmmis_bits);
__IO_REG32(    I2C0MICR,              0x4002001C,__WRITE      );
__IO_REG32_BIT(I2C0MCR,               0x40020020,__READ_WRITE ,__i2cmcr_bits);
__IO_REG32_BIT(I2C0SOAR,              0x40020800,__READ_WRITE ,__i2csoar_bits);
__IO_REG32_BIT(I2C0SCSR,              0x40020804,__READ_WRITE ,__i2cscsr_bits);
__IO_REG32_BIT(I2C0SDR,               0x40020808,__READ_WRITE ,__i2csdr_bits);
__IO_REG32_BIT(I2C0SIMR,              0x4002080C,__READ_WRITE ,__i2csimr_bits);
__IO_REG32_BIT(I2C0SRIS,              0x40020810,__READ       ,__i2csris_bits);
__IO_REG32_BIT(I2C0SMIS,              0x40020814,__READ       ,__i2csmis_bits);
__IO_REG32(    I2C0SICR,              0x40020818,__WRITE      );

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1MSA,               0x40021000,__READ_WRITE ,__i2cmsa_bits);
__IO_REG32_BIT(I2C1MCS,               0x40021004,__READ_WRITE ,__i2cmcs_bits);
__IO_REG32_BIT(I2C1MDR,               0x40021008,__READ_WRITE ,__i2cmdr_bits);
__IO_REG32_BIT(I2C1MTPR,              0x4002100C,__READ_WRITE ,__i2cmtpr_bits);
__IO_REG32_BIT(I2C1MIMR,              0x40021010,__READ_WRITE ,__i2cmimr_bits);
__IO_REG32_BIT(I2C1MRIS,              0x40021014,__READ       ,__i2cmris_bits);
__IO_REG32_BIT(I2C1MMIS,              0x40021018,__READ       ,__i2cmmis_bits);
__IO_REG32(    I2C1MICR,              0x4002101C,__WRITE      );
__IO_REG32_BIT(I2C1MCR,               0x40021020,__READ_WRITE ,__i2cmcr_bits);
__IO_REG32_BIT(I2C1SOAR,              0x40021800,__READ_WRITE ,__i2csoar_bits);
__IO_REG32_BIT(I2C1SCSR,              0x40021804,__READ_WRITE ,__i2cscsr_bits);
__IO_REG32_BIT(I2C1SDR,               0x40021808,__READ_WRITE ,__i2csdr_bits);
__IO_REG32_BIT(I2C1SIMR,              0x4002180C,__READ_WRITE ,__i2csimr_bits);
__IO_REG32_BIT(I2C1SRIS,              0x40021810,__READ       ,__i2csris_bits);
__IO_REG32_BIT(I2C1SMIS,              0x40021814,__READ       ,__i2csmis_bits);
__IO_REG32(    I2C1SICR,              0x40021818,__WRITE      );

/***************************************************************************
 **
 ** CAN0 (Controller Area Network)
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0CTL,               0x40070000,__READ_WRITE ,__canctl_bits);
__IO_REG32_BIT(CAN0ES,                0x40070004,__READ       ,__canes_bits);
__IO_REG32_BIT(CAN0ERRC,              0x40070008,__READ       ,__canerrc_bits);
__IO_REG32_BIT(CAN0BTR,               0x4007000C,__READ_WRITE ,__canbtr_bits);
__IO_REG32_BIT(CAN0INT,               0x40070010,__READ       ,__canint_bits);
__IO_REG32_BIT(CAN0TEST,              0x40070014,__READ_WRITE ,__cantest_bits);
__IO_REG32_BIT(CAN0PERR,              0x4007001C,__READ       ,__canperr_bits);
__IO_REG32(    CAN0ABOTR,             0x40070080,__READ_WRITE );
__IO_REG32_BIT(CAN0TXRQ,              0x40070088,__READ       ,__cantxrq_bits);
__IO_REG32_BIT(CAN0NWDAT,             0x4007009C,__READ       ,__cannwdat_bits);
__IO_REG32_BIT(CAN0INTPND,            0x400700B0,__READ       ,__canintpnd_bits);
__IO_REG32_BIT(CAN0MSGVAL,            0x400700C4,__READ       ,__canmsgval_bits);
__IO_REG32_BIT(CAN0INTMUX,            0x400700D8,__READ_WRITE ,__canintmux_bits);
__IO_REG32_BIT(CAN0IF1CMD,            0x40070100,__READ_WRITE ,__canifcmd_bits);
__IO_REG32_BIT(CAN0IF1MSK,            0x40070104,__READ_WRITE ,__canifmsk_bits);
__IO_REG32_BIT(CAN0IF1ARB,            0x40070108,__READ_WRITE ,__canifarb_bits);
__IO_REG32_BIT(CAN0IF1MCTL,           0x4007010C,__READ_WRITE ,__canifmctl_bits);
__IO_REG32_BIT(CAN0IF1DATA,           0x40070110,__READ_WRITE ,__canifdata_bits);
__IO_REG32_BIT(CAN0IF1DATB,           0x40070114,__READ_WRITE ,__canifdatb_bits);
__IO_REG32_BIT(CAN0IF2CMD,            0x40070120,__READ_WRITE ,__canifcmd_bits);
__IO_REG32_BIT(CAN0IF2MSK,            0x40070124,__READ_WRITE ,__canifmsk_bits);
__IO_REG32_BIT(CAN0IF2ARB,            0x40070128,__READ_WRITE ,__canifarb_bits);
__IO_REG32_BIT(CAN0IF2MCTL,           0x4007012C,__READ_WRITE ,__canifmctl_bits);
__IO_REG32_BIT(CAN0IF2DATA,           0x40070130,__READ_WRITE ,__canifdata_bits);
__IO_REG32_BIT(CAN0IF2DATB,           0x40070134,__READ_WRITE ,__canifdatb_bits);
__IO_REG32_BIT(CAN0IF3OBS,            0x40070140,__READ_WRITE ,__canif3obs_bits);
__IO_REG32_BIT(CAN0IF3MSK,            0x40070144,__READ       ,__canifmsk_bits);
__IO_REG32_BIT(CAN0IF3ARB,            0x40070148,__READ       ,__canifarb_bits);
__IO_REG32_BIT(CAN0IF3MCTL,           0x4007014C,__READ       ,__canifmctl_bits);
__IO_REG32_BIT(CAN0IF3DATA,           0x40070150,__READ       ,__canifdata_bits);
__IO_REG32_BIT(CAN0IF3DATB,           0x40070154,__READ       ,__canifdatb_bits);
__IO_REG32_BIT(CAN0IF3UPD,            0x40070160,__READ_WRITE ,__canif3upd_bits);

/***************************************************************************
 **
 ** CAN1 (Controller Area Network)
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1CTL,               0x40074000,__READ_WRITE ,__canctl_bits);
__IO_REG32_BIT(CAN1ES,                0x40074004,__READ       ,__canes_bits);
__IO_REG32_BIT(CAN1ERRC,              0x40074008,__READ       ,__canerrc_bits);
__IO_REG32_BIT(CAN1BTR,               0x4007400C,__READ_WRITE ,__canbtr_bits);
__IO_REG32_BIT(CAN1INT,               0x40074010,__READ       ,__canint_bits);
__IO_REG32_BIT(CAN1TEST,              0x40074014,__READ_WRITE ,__cantest_bits);
__IO_REG32_BIT(CAN1PERR,              0x4007401C,__READ       ,__canperr_bits);
__IO_REG32(    CAN1ABOTR,             0x40074080,__READ_WRITE );
__IO_REG32_BIT(CAN1TXRQ,              0x40074088,__READ       ,__cantxrq_bits);
__IO_REG32_BIT(CAN1NWDAT,             0x4007409C,__READ       ,__cannwdat_bits);
__IO_REG32_BIT(CAN1INTPND,            0x400740B0,__READ       ,__canintpnd_bits);
__IO_REG32_BIT(CAN1MSGVAL,            0x400740C4,__READ       ,__canmsgval_bits);
__IO_REG32_BIT(CAN1INTMUX,            0x400740D8,__READ_WRITE ,__canintmux_bits);
__IO_REG32_BIT(CAN1IF1CMD,            0x40074100,__READ_WRITE ,__canifcmd_bits);
__IO_REG32_BIT(CAN1IF1MSK,            0x40074104,__READ_WRITE ,__canifmsk_bits);
__IO_REG32_BIT(CAN1IF1ARB,            0x40074108,__READ_WRITE ,__canifarb_bits);
__IO_REG32_BIT(CAN1IF1MCTL,           0x4007410C,__READ_WRITE ,__canifmctl_bits);
__IO_REG32_BIT(CAN1IF1DATA,           0x40074110,__READ_WRITE ,__canifdata_bits);
__IO_REG32_BIT(CAN1IF1DATB,           0x40074114,__READ_WRITE ,__canifdatb_bits);
__IO_REG32_BIT(CAN1IF2CMD,            0x40074120,__READ_WRITE ,__canifcmd_bits);
__IO_REG32_BIT(CAN1IF2MSK,            0x40074124,__READ_WRITE ,__canifmsk_bits);
__IO_REG32_BIT(CAN1IF2ARB,            0x40074128,__READ_WRITE ,__canifarb_bits);
__IO_REG32_BIT(CAN1IF2MCTL,           0x4007412C,__READ_WRITE ,__canifmctl_bits);
__IO_REG32_BIT(CAN1IF2DATA,           0x40074130,__READ_WRITE ,__canifdata_bits);
__IO_REG32_BIT(CAN1IF2DATB,           0x40074134,__READ_WRITE ,__canifdatb_bits);
__IO_REG32_BIT(CAN1IF3OBS,            0x40074140,__READ_WRITE ,__canif3obs_bits);
__IO_REG32_BIT(CAN1IF3MSK,            0x40074144,__READ       ,__canifmsk_bits);
__IO_REG32_BIT(CAN1IF3ARB,            0x40074148,__READ       ,__canifarb_bits);
__IO_REG32_BIT(CAN1IF3MCTL,           0x4007414C,__READ       ,__canifmctl_bits);
__IO_REG32_BIT(CAN1IF3DATA,           0x40074150,__READ       ,__canifdata_bits);
__IO_REG32_BIT(CAN1IF3DATB,           0x40074154,__READ       ,__canifdatb_bits);
__IO_REG32_BIT(CAN1IF3UPD,            0x40074160,__READ_WRITE ,__canif3upd_bits);

/* Assembler-specific declarations **********************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **    F28M35x52C INTERRUPT VALUES
 **
***************************************************************************/
/***************************************************************************
 **
 **  NVIC M3VIM Interrupt channels
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
#define NVIC_GPIOA           ( 0 + EII)   /* PORTA                        */
#define NVIC_GPIOB           ( 1 + EII)   /* PORTB                        */
#define NVIC_GPIOC           ( 2 + EII)   /* PORTC                        */
#define NVIC_GPIOD           ( 3 + EII)   /* PORTD                        */
#define NVIC_GPIOE           ( 4 + EII)   /* PORTE                        */
#define NVIC_UART0           ( 5 + EII)   /* UART 0                       */
#define NVIC_UART1           ( 6 + EII)   /* UART 1                       */
#define NVIC_SSI0            ( 7 + EII)   /* SSI0                         */
#define NVIC_I2C0            ( 8 + EII)   /* I2C0                         */
#define NVIC_WDT             (18 + EII)   /* WDT0 and WDT1                */
#define NVIC_TIMER0A         (19 + EII)   /* Timer 0 Channel A            */
#define NVIC_TIMER0B         (20 + EII)   /* Timer 0 Channel B            */
#define NVIC_TIMER1A         (21 + EII)   /* Timer 1 Channel A            */
#define NVIC_TIMER1B         (22 + EII)   /* Timer 1 Channel B            */
#define NVIC_TIMER2A         (23 + EII)   /* Timer 2 Channel A            */
#define NVIC_TIMER2B         (24 + EII)   /* Timer 2 Channel B            */
#define NVIC_SYS_CTRL        (28 + EII)   /* System control               */
#define NVIC_FLASH_CTRL      (29 + EII)   /* Flash controller             */
#define NVIC_GPIOF           (30 + EII)   /* PORTF                        */
#define NVIC_GPIOG           (31 + EII)   /* PORTG                        */
#define NVIC_GPIOH           (32 + EII)   /* PORTH                        */
#define NVIC_UART2           (33 + EII)   /* UART2                        */
#define NVIC_SSI1            (34 + EII)   /* SSI1                         */
#define NVIC_TIMER3A         (35 + EII)   /* Timer 3 Channel A            */
#define NVIC_TIMER3B         (36 + EII)   /* Timer 3 Channel B            */
#define NVIC_I2C1            (37 + EII)   /* I2C1                         */
#define NVIC_ENET            (42 + EII)   /* Ethernet MAC                 */
#define NVIC_USB             (44 + EII)   /* USB Module                   */
#define NVIC_UDMA_SOFT       (46 + EII)   /* uDMA Software                */
#define NVIC_UDMA_ERR        (47 + EII)   /* uDMA Error                   */
#define NVIC_EPI             (53 + EII)   /* EPI                          */
#define NVIC_GPIOJ           (54 + EII)   /* PORTJ                        */
#define NVIC_SSI2            (57 + EII)   /* SSI2                         */
#define NVIC_SSI3            (58 + EII)   /* SSI3                         */
#define NVIC_UART3           (59 + EII)   /* UART3                        */
#define NVIC_UART4           (60 + EII)   /* UART4                        */
#define NVIC_CAN0INT0        (64 + EII)   /* CAN0 INT0                    */
#define NVIC_CAN0INT1        (65 + EII)   /* CAN0 INT1                    */
#define NVIC_CAN1INT0        (66 + EII)   /* CAN1 INT0                    */
#define NVIC_CAN1INT1        (67 + EII)   /* CAN1 INT1                    */
#define NVIC_ADCINT1         (72 + EII)   /* ADCINT1                      */
#define NVIC_ADCINT2         (73 + EII)   /* ADCINT2                      */
#define NVIC_ADCINT3         (74 + EII)   /* ADCINT3                      */
#define NVIC_ADCINT4         (75 + EII)   /* ADCINT4                      */
#define NVIC_ADCINT5         (76 + EII)   /* ADCINT5                      */
#define NVIC_ADCINT6         (77 + EII)   /* ADCINT6                      */
#define NVIC_ADCINT7         (78 + EII)   /* ADCINT7                      */
#define NVIC_ADCINT8         (79 + EII)   /* ADCINT8                      */
#define NVIC_CTOMIPC1        (80 + EII)   /* CTOMIPC1                     */
#define NVIC_CTOMIPC2        (81 + EII)   /* CTOMIPC2                     */
#define NVIC_CTOMIPC3        (82 + EII)   /* CTOMIPC3                     */
#define NVIC_CTOMIPC4        (83 + EII)   /* CTOMIPC4                     */
#define NVIC_RAM             (88 + EII)   /* RAM Single Error             */
#define NVIC_USBPLL          (89 + EII)   /* System / USB PLL Out of Lock */
#define NVIC_Flash           (90 + EII)   /* M3 Flash Single Error        */

#endif    /* __IOF28M35x52C_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI                    0x08
Interrupt1   = HardFault              0x0C
Interrupt2   = MemManage              0x10
Interrupt3   = BusFault               0x14
Interrupt4   = UsageFault             0x18
Interrupt5   = SVC                    0x2C
Interrupt6   = DebugMon               0x30
Interrupt7   = PendSV                 0x38
Interrupt8   = SysTick                0x3C
Interrupt9   = GPIOA                  0x40
Interrupt10  = GPIOB                  0x44
Interrupt11  = GPIOC                  0x48
Interrupt12  = GPIOD                  0x4C
Interrupt13  = GPIOE                  0x50
Interrupt14  = UART0                  0x54
Interrupt15  = UART1                  0x58
Interrupt16  = SSI0                   0x5C
Interrupt17  = I2C0                   0x60
Interrupt18  = WDT                    0x88
Interrupt19  = TIMER0A                0x8C
Interrupt20  = TIMER0B                0x90
Interrupt21  = TIMER1A                0x94
Interrupt22  = TIMER1B                0x98
Interrupt23  = TIMER2A                0x9C
Interrupt24  = TIMER2B                0xA0
Interrupt25  = SYS_CTRL               0xB0
Interrupt26  = FLASH_CTRL             0xB4
Interrupt27  = GPIOF                  0xB8
Interrupt28  = GPIOG                  0xBC
Interrupt29  = GPIOH                  0xC0
Interrupt30  = UART2                  0xC4
Interrupt31  = SSI1                   0xC8
Interrupt32  = TIMER3A                0xCC
Interrupt33  = TIMER3B                0xD0
Interrupt34  = I2C1                   0xD4
Interrupt35  = ENET                   0xE8
Interrupt36  = USB                    0xF0
Interrupt37  = UDMA_SOFT              0xF8
Interrupt38  = UDMA_ERR               0xFC
Interrupt39  = EPI                    0x114
Interrupt40  = GPIOJ                  0x118
Interrupt41  = SSI2                   0x124
Interrupt42  = SSI3                   0x128
Interrupt43  = UART3                  0x12C
Interrupt44  = UART4                  0x130
Interrupt45  = CAN0INT0               0x140
Interrupt46  = CAN0INT1               0x144
Interrupt47  = CAN1INT0               0x148
Interrupt48  = CAN1INT1               0x14C
Interrupt49  = ADCINT1                0x160
Interrupt50  = ADCINT2                0x164
Interrupt51  = ADCINT3                0x168
Interrupt52  = ADCINT4                0x16C
Interrupt53  = ADCINT5                0x170
Interrupt54  = ADCINT6                0x174
Interrupt55  = ADCINT7                0x178
Interrupt56  = ADCINT8                0x17C
Interrupt57  = CTOMIPC1               0x180
Interrupt58  = CTOMIPC2               0x184
Interrupt59  = CTOMIPC3               0x188
Interrupt60  = CTOMIPC4               0x18C
Interrupt61  = RAM                    0x1A0
Interrupt62  = USBPLL                 0x1A4
Interrupt63  = Flash                  0x1A8

###DDF-INTERRUPT-END###*/
