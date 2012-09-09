/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments TMS470MF06607
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 ** **    (c) Copyright IAR Systems and Texas Instruments 2011
 **
 **    $Revision: 46445 $
 **
***************************************************************************/

#ifndef __IOTMS470MF06607_H
#define __IOTMS470MF06607_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMS470MF06607 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
***************************************************************************/

/* C-compiler specific declarations **********************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 1
#error This file should only be compiled in big endian mode
#endif

/* Reverse the bitfield order in the structs */
#pragma bitfields=disjoint_types

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

/* INTISR Index Offset Vector Register (INTISRIVEC) */
typedef struct {
  __REG32  INTISRIVEC	    : 8;
  __REG32                 :24;
} __intisrivec_bits;

/* INTISR Index Offset Vector Register (INTISRIVEC) */
typedef struct {
  __REG32  INTNMIIVEC	    : 8;
  __REG32                 :24;
} __intnmiivec_bits;

/* NESTCTRL Register */
typedef struct {
  __REG32  NEST_ENABLE    : 4;
  __REG32                 :28;
} __nestctrl_bits;

/* NESTSTAT Register */
typedef struct {
  __REG32  NEST_OVRN	    : 1;
  __REG32  NEST_STAT	    : 1;
  __REG32                 : 6;
  __REG32  NEST_LEVEL	    : 8;
  __REG32                 :16;
} __neststat_bits;

/* NMI0 Program Control Register (NMIPR0) */
typedef struct {
  __REG32  NMIPR0	    	: 1;
  __REG32  NMIPR1	    	: 1;
  __REG32  NMIPR2	    	: 1;
  __REG32  NMIPR3	    	: 1;
  __REG32  NMIPR4	    	: 1;
  __REG32  NMIPR5	    	: 1;
  __REG32  NMIPR6	    	: 1;
  __REG32  NMIPR7	    	: 1;
  __REG32  NMIPR8	    	: 1;
  __REG32  NMIPR9	    	: 1;
  __REG32  NMIPR10    	: 1;
  __REG32  NMIPR11    	: 1;
  __REG32  NMIPR12    	: 1;
  __REG32  NMIPR13    	: 1;
  __REG32  NMIPR14    	: 1;
  __REG32  NMIPR15    	: 1;
  __REG32  NMIPR16    	: 1;
  __REG32  NMIPR17    	: 1;
  __REG32  NMIPR18    	: 1;
  __REG32  NMIPR19    	: 1;
  __REG32  NMIPR20    	: 1;
  __REG32  NMIPR21    	: 1;
  __REG32  NMIPR22    	: 1;
  __REG32  NMIPR23    	: 1;
  __REG32  NMIPR24    	: 1;
  __REG32  NMIPR25    	: 1;
  __REG32  NMIPR26    	: 1;
  __REG32  NMIPR27    	: 1;
  __REG32  NMIPR28    	: 1;
  __REG32  NMIPR29    	: 1;
  __REG32  NMIPR30    	: 1;
  __REG32  NMIPR31    	: 1;
} __nmipr0_bits;

/* NMI1 Program Control Register (NMIPR1) */
typedef struct {
  __REG32  NMIPR32    	: 1;
  __REG32  NMIPR33    	: 1;
  __REG32  NMIPR34    	: 1;
  __REG32  NMIPR35    	: 1;
  __REG32  NMIPR36    	: 1;
  __REG32  NMIPR37    	: 1;
  __REG32  NMIPR38    	: 1;
  __REG32  NMIPR39    	: 1;
  __REG32  NMIPR40    	: 1;
  __REG32  NMIPR41    	: 1;
  __REG32  NMIPR42    	: 1;
  __REG32  NMIPR43    	: 1;
  __REG32  NMIPR44    	: 1;
  __REG32  NMIPR45    	: 1;
  __REG32  NMIPR46    	: 1;
  __REG32  NMIPR47    	: 1;
  __REG32  				    	:16;
} __nmipr1_bits;

/* Pending Interrupt Read Location Register 0 (INTREQ0) */
typedef struct {
  __REG32  INTREQ0	    : 1;
  __REG32  INTREQ1	    : 1;
  __REG32  INTREQ2	    : 1;
  __REG32  INTREQ3	    : 1;
  __REG32  INTREQ4	    : 1;
  __REG32  INTREQ5	    : 1;
  __REG32  INTREQ6	    : 1;
  __REG32  INTREQ7	    : 1;
  __REG32  INTREQ8	    : 1;
  __REG32  INTREQ9	    : 1;
  __REG32  INTREQ10    	: 1;
  __REG32  INTREQ11    	: 1;
  __REG32  INTREQ12    	: 1;
  __REG32  INTREQ13    	: 1;
  __REG32  INTREQ14    	: 1;
  __REG32  INTREQ15    	: 1;
  __REG32  INTREQ16    	: 1;
  __REG32  INTREQ17    	: 1;
  __REG32  INTREQ18    	: 1;
  __REG32  INTREQ19    	: 1;
  __REG32  INTREQ20    	: 1;
  __REG32  INTREQ21    	: 1;
  __REG32  INTREQ22    	: 1;
  __REG32  INTREQ23    	: 1;
  __REG32  INTREQ24    	: 1;
  __REG32  INTREQ25    	: 1;
  __REG32  INTREQ26    	: 1;
  __REG32  INTREQ27    	: 1;
  __REG32  INTREQ28    	: 1;
  __REG32  INTREQ29    	: 1;
  __REG32  INTREQ30    	: 1;
  __REG32  INTREQ31    	: 1;
} __intreq0_bits;

/* Pending Interrupt Read Location Register 1 (INTREQ1) */
typedef struct {
  __REG32  INTREQ32    	: 1;
  __REG32  INTREQ33    	: 1;
  __REG32  INTREQ34    	: 1;
  __REG32  INTREQ35    	: 1;
  __REG32  INTREQ36    	: 1;
  __REG32  INTREQ37    	: 1;
  __REG32  INTREQ38    	: 1;
  __REG32  INTREQ39    	: 1;
  __REG32  INTREQ40    	: 1;
  __REG32  INTREQ41    	: 1;
  __REG32  INTREQ42    	: 1;
  __REG32  INTREQ43    	: 1;
  __REG32  INTREQ44    	: 1;
  __REG32  INTREQ45    	: 1;
  __REG32  INTREQ46    	: 1;
  __REG32  INTREQ47    	: 1;
  __REG32  				    	:16;
} __intreq1_bits;

/* Interrupt Mask Set Register 0 (REQMASKSET0) */
typedef struct {
  __REG32  REQMASKSET0	    : 1;
  __REG32  REQMASKSET1	    : 1;
  __REG32  REQMASKSET2	    : 1;
  __REG32  REQMASKSET3	    : 1;
  __REG32  REQMASKSET4	    : 1;
  __REG32  REQMASKSET5	    : 1;
  __REG32  REQMASKSET6	    : 1;
  __REG32  REQMASKSET7	    : 1;
  __REG32  REQMASKSET8	    : 1;
  __REG32  REQMASKSET9	    : 1;
  __REG32  REQMASKSET10    	: 1;
  __REG32  REQMASKSET11    	: 1;
  __REG32  REQMASKSET12    	: 1;
  __REG32  REQMASKSET13    	: 1;
  __REG32  REQMASKSET14    	: 1;
  __REG32  REQMASKSET15    	: 1;
  __REG32  REQMASKSET16    	: 1;
  __REG32  REQMASKSET17    	: 1;
  __REG32  REQMASKSET18    	: 1;
  __REG32  REQMASKSET19    	: 1;
  __REG32  REQMASKSET20    	: 1;
  __REG32  REQMASKSET21    	: 1;
  __REG32  REQMASKSET22    	: 1;
  __REG32  REQMASKSET23    	: 1;
  __REG32  REQMASKSET24    	: 1;
  __REG32  REQMASKSET25    	: 1;
  __REG32  REQMASKSET26    	: 1;
  __REG32  REQMASKSET27    	: 1;
  __REG32  REQMASKSET28    	: 1;
  __REG32  REQMASKSET29    	: 1;
  __REG32  REQMASKSET30    	: 1;
  __REG32  REQMASKSET31    	: 1;
} __reqmaskset0_bits;

/* Interrupt Mask Set Register 1 (REQMASKSET1) */
typedef struct {
  __REG32  REQMASKSET32    	: 1;
  __REG32  REQMASKSET33    	: 1;
  __REG32  REQMASKSET34    	: 1;
  __REG32  REQMASKSET35    	: 1;
  __REG32  REQMASKSET36    	: 1;
  __REG32  REQMASKSET37    	: 1;
  __REG32  REQMASKSET38    	: 1;
  __REG32  REQMASKSET39    	: 1;
  __REG32  REQMASKSET40    	: 1;
  __REG32  REQMASKSET41    	: 1;
  __REG32  REQMASKSET42    	: 1;
  __REG32  REQMASKSET43    	: 1;
  __REG32  REQMASKSET44    	: 1;
  __REG32  REQMASKSET45    	: 1;
  __REG32  REQMASKSET46    	: 1;
  __REG32  REQMASKSET47    	: 1;
  __REG32  				    			:16;
} __reqmaskset1_bits;

/* Interrupt Mask Clear Register 0 (REQMASKCLR0) */
typedef struct {
  __REG32  REQMASKCLR0	    : 1;
  __REG32  REQMASKCLR1	    : 1;
  __REG32  REQMASKCLR2	    : 1;
  __REG32  REQMASKCLR3	    : 1;
  __REG32  REQMASKCLR4	    : 1;
  __REG32  REQMASKCLR5	    : 1;
  __REG32  REQMASKCLR6	    : 1;
  __REG32  REQMASKCLR7	    : 1;
  __REG32  REQMASKCLR8	    : 1;
  __REG32  REQMASKCLR9	    : 1;
  __REG32  REQMASKCLR10    	: 1;
  __REG32  REQMASKCLR11    	: 1;
  __REG32  REQMASKCLR12    	: 1;
  __REG32  REQMASKCLR13    	: 1;
  __REG32  REQMASKCLR14    	: 1;
  __REG32  REQMASKCLR15    	: 1;
  __REG32  REQMASKCLR16    	: 1;
  __REG32  REQMASKCLR17    	: 1;
  __REG32  REQMASKCLR18    	: 1;
  __REG32  REQMASKCLR19    	: 1;
  __REG32  REQMASKCLR20    	: 1;
  __REG32  REQMASKCLR21    	: 1;
  __REG32  REQMASKCLR22    	: 1;
  __REG32  REQMASKCLR23    	: 1;
  __REG32  REQMASKCLR24    	: 1;
  __REG32  REQMASKCLR25    	: 1;
  __REG32  REQMASKCLR26    	: 1;
  __REG32  REQMASKCLR27    	: 1;
  __REG32  REQMASKCLR28    	: 1;
  __REG32  REQMASKCLR29    	: 1;
  __REG32  REQMASKCLR30    	: 1;
  __REG32  REQMASKCLR31    	: 1;
} __reqmaskclr0_bits;

/* Interrupt Mask Clear Register 1 (REQMASKCLR1) */
typedef struct {
  __REG32  REQMASKSET32    	: 1;
  __REG32  REQMASKSET33    	: 1;
  __REG32  REQMASKSET34    	: 1;
  __REG32  REQMASKSET35    	: 1;
  __REG32  REQMASKSET36    	: 1;
  __REG32  REQMASKSET37    	: 1;
  __REG32  REQMASKSET38    	: 1;
  __REG32  REQMASKSET39    	: 1;
  __REG32  REQMASKSET40    	: 1;
  __REG32  REQMASKSET41    	: 1;
  __REG32  REQMASKSET42    	: 1;
  __REG32  REQMASKSET43    	: 1;
  __REG32  REQMASKSET44    	: 1;
  __REG32  REQMASKSET45    	: 1;
  __REG32  REQMASKSET46    	: 1;
  __REG32  REQMASKSET47    	: 1;
  __REG32  				    			:16;
} __reqmaskclr1_bits;

/* Wake-up Mask Set Register 0 (WAKEMASKSET0) */
typedef struct {
  __REG32  WAKEMASKSET0	    : 1;
  __REG32  WAKEMASKSET1	    : 1;
  __REG32  WAKEMASKSET2	    : 1;
  __REG32  WAKEMASKSET3	    : 1;
  __REG32  WAKEMASKSET4	    : 1;
  __REG32  WAKEMASKSET5	    : 1;
  __REG32  WAKEMASKSET6	    : 1;
  __REG32  WAKEMASKSET7	    : 1;
  __REG32  WAKEMASKSET8	    : 1;
  __REG32  WAKEMASKSET9	    : 1;
  __REG32  WAKEMASKSET10    : 1;
  __REG32  WAKEMASKSET11    : 1;
  __REG32  WAKEMASKSET12    : 1;
  __REG32  WAKEMASKSET13    : 1;
  __REG32  WAKEMASKSET14    : 1;
  __REG32  WAKEMASKSET15    : 1;
  __REG32  WAKEMASKSET16    : 1;
  __REG32  WAKEMASKSET17    : 1;
  __REG32  WAKEMASKSET18    : 1;
  __REG32  WAKEMASKSET19    : 1;
  __REG32  WAKEMASKSET20    : 1;
  __REG32  WAKEMASKSET21    : 1;
  __REG32  WAKEMASKSET22    : 1;
  __REG32  WAKEMASKSET23    : 1;
  __REG32  WAKEMASKSET24    : 1;
  __REG32  WAKEMASKSET25    : 1;
  __REG32  WAKEMASKSET26    : 1;
  __REG32  WAKEMASKSET27    : 1;
  __REG32  WAKEMASKSET28    : 1;
  __REG32  WAKEMASKSET29    : 1;
  __REG32  WAKEMASKSET30    : 1;
  __REG32  WAKEMASKSET31    : 1;
} __wakemaskset0_bits;

/* Wake-up Mask Set Register 1 (WAKEMASKSET1) */
typedef struct {
  __REG32  WAKEMASKSET32    : 1;
  __REG32  WAKEMASKSET33    : 1;
  __REG32  WAKEMASKSET34    : 1;
  __REG32  WAKEMASKSET35    : 1;
  __REG32  WAKEMASKSET36    : 1;
  __REG32  WAKEMASKSET37    : 1;
  __REG32  WAKEMASKSET38    : 1;
  __REG32  WAKEMASKSET39    : 1;
  __REG32  WAKEMASKSET40    : 1;
  __REG32  WAKEMASKSET41    : 1;
  __REG32  WAKEMASKSET42    : 1;
  __REG32  WAKEMASKSET43    : 1;
  __REG32  WAKEMASKSET44    : 1;
  __REG32  WAKEMASKSET45    : 1;
  __REG32  WAKEMASKSET46    : 1;
  __REG32  WAKEMASKSET47    : 1;
  __REG32  				    			:16;
} __wakemaskset1_bits;

/* Wake-up Mask Clear Register 0 (WAKEMASKCLR0) */
typedef struct {
  __REG32  WAKEMASKCLR0	    : 1;
  __REG32  WAKEMASKCLR1	    : 1;
  __REG32  WAKEMASKCLR2	    : 1;
  __REG32  WAKEMASKCLR3	    : 1;
  __REG32  WAKEMASKCLR4	    : 1;
  __REG32  WAKEMASKCLR5	    : 1;
  __REG32  WAKEMASKCLR6	    : 1;
  __REG32  WAKEMASKCLR7	    : 1;
  __REG32  WAKEMASKCLR8	    : 1;
  __REG32  WAKEMASKCLR9	    : 1;
  __REG32  WAKEMASKCLR10    : 1;
  __REG32  WAKEMASKCLR11    : 1;
  __REG32  WAKEMASKCLR12    : 1;
  __REG32  WAKEMASKCLR13    : 1;
  __REG32  WAKEMASKCLR14    : 1;
  __REG32  WAKEMASKCLR15    : 1;
  __REG32  WAKEMASKCLR16    : 1;
  __REG32  WAKEMASKCLR17    : 1;
  __REG32  WAKEMASKCLR18    : 1;
  __REG32  WAKEMASKCLR19    : 1;
  __REG32  WAKEMASKCLR20    : 1;
  __REG32  WAKEMASKCLR21    : 1;
  __REG32  WAKEMASKCLR22    : 1;
  __REG32  WAKEMASKCLR23    : 1;
  __REG32  WAKEMASKCLR24    : 1;
  __REG32  WAKEMASKCLR25    : 1;
  __REG32  WAKEMASKCLR26    : 1;
  __REG32  WAKEMASKCLR27    : 1;
  __REG32  WAKEMASKCLR28    : 1;
  __REG32  WAKEMASKCLR29    : 1;
  __REG32  WAKEMASKCLR30    : 1;
  __REG32  WAKEMASKCLR31    : 1;
} __wakemaskclr0_bits;

/* Wake-up Mask Clear Register 1 (WAKEMASKCLR1) */
typedef struct {
  __REG32  WAKEMASKCLR32    : 1;
  __REG32  WAKEMASKCLR33    : 1;
  __REG32  WAKEMASKCLR34    : 1;
  __REG32  WAKEMASKCLR35    : 1;
  __REG32  WAKEMASKCLR36    : 1;
  __REG32  WAKEMASKCLR37    : 1;
  __REG32  WAKEMASKCLR38    : 1;
  __REG32  WAKEMASKCLR39    : 1;
  __REG32  WAKEMASKCLR40    : 1;
  __REG32  WAKEMASKCLR41    : 1;
  __REG32  WAKEMASKCLR42    : 1;
  __REG32  WAKEMASKCLR43    : 1;
  __REG32  WAKEMASKCLR44    : 1;
  __REG32  WAKEMASKCLR45    : 1;
  __REG32  WAKEMASKCLR46    : 1;
  __REG32  WAKEMASKCLR47    : 1;
  __REG32  				    			:16;
} __wakemaskclr1_bits;

/* Capture Event Register (CAPEVT) */
typedef struct {
  __REG32  CAPEVTSRC0		    : 7;
  __REG32  							    : 9;
  __REG32  CAPEVTSRC1		    : 7;
  __REG32  							    : 9;
} __capevt_bits;

/* Interrupt Control Register 0 */
typedef struct {
  __REG32  CHANMAP3			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP2			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP1			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP0			    : 7;
  __REG32  							    : 1;
} __chanctrl0_bits;

/* Interrupt Control Register 1 */
typedef struct {
  __REG32  CHANMAP7			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP6			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP5			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP4			    : 7;
  __REG32  							    : 1;
} __chanctrl1_bits;

/* Interrupt Control Register 2 */
typedef struct {
  __REG32  CHANMAP11		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP10		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP9			    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP8			    : 7;
  __REG32  							    : 1;
} __chanctrl2_bits;

/* Interrupt Control Register 3 */
typedef struct {
  __REG32  CHANMAP15		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP14		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP13		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP12		    : 7;
  __REG32  							    : 1;
} __chanctrl3_bits;

/* Interrupt Control Register 4 */
typedef struct {
  __REG32  CHANMAP19		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP18		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP17		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP16		    : 7;
  __REG32  							    : 1;
} __chanctrl4_bits;

/* Interrupt Control Register 5 */
typedef struct {
  __REG32  CHANMAP23		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP22		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP21		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP20		    : 7;
  __REG32  							    : 1;
} __chanctrl5_bits;

/* Interrupt Control Register 6 */
typedef struct {
  __REG32  CHANMAP27		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP26		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP25		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP24		    : 7;
  __REG32  							    : 1;
} __chanctrl6_bits;

/* Interrupt Control Register 7 */
typedef struct {
  __REG32  CHANMAP31		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP30		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP29		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP28		    : 7;
  __REG32  							    : 1;
} __chanctrl7_bits;

/* Interrupt Control Register 8 */
typedef struct {
  __REG32  CHANMAP35		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP34		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP33		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP32		    : 7;
  __REG32  							    : 1;
} __chanctrl8_bits;

/* Interrupt Control Register 9 */
typedef struct {
  __REG32  CHANMAP39		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP38		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP37		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP36		    : 7;
  __REG32  							    : 1;
} __chanctrl9_bits;

/* Interrupt Control Register 10 */
typedef struct {
  __REG32  CHANMAP43		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP42		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP41		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP40		    : 7;
  __REG32  							    : 1;
} __chanctrl10_bits;

/* Interrupt Control Register 11 */
typedef struct {
  __REG32  CHANMAP47		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP46		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP45		    : 7;
  __REG32  							    : 1;
  __REG32  CHANMAP44		    : 7;
  __REG32  							    : 1;
} __chanctrl11_bits;

/* SYS Pin Control Register 1 (SYSPC1) */
typedef struct {
  __REG32 ECPCLKFUN       : 1;
  __REG32                 :31;
} __syspc1_bits;

/* SYS Pin Control Register 2 (SYSPC2) */
typedef struct {
  __REG32 ECPCLK_DIR      : 1;
  __REG32                 :31;
} __syspc2_bits;

/* SYS Pin Control Register 3 (SYSPC3) */
typedef struct {
  __REG32 ECPCLK_DIN      : 1;
  __REG32                 :31;
} __syspc3_bits;

/* SYS Pin Control Register 4 (SYSPC4) */
typedef struct {
  __REG32 ECPCLK_DOUT     : 1;
  __REG32                 :31;
} __syspc4_bits;

/* SYS Pin Control Register 5 (SYSPC5) */
typedef struct {
  __REG32 ECPCLK_SET      : 1;
  __REG32                 :31;
} __syspc5_bits;

/* SYS Pin Control Register 6 (SYSPC6) */
typedef struct {
  __REG32 ECPCLK_CLR      : 1;
  __REG32                 :31;
} __syspc6_bits;

/* SYS Pin Control Register 7 (SYSPC7) */
typedef struct {
  __REG32 ECPCLK_ODE      : 1;
  __REG32                 :31;
} __syspc7_bits;

/* SYS Pin Control Register 8 (SYSPC8) */
typedef struct {
  __REG32 ECPCLK_PUE	    : 1;
  __REG32                 :31;
} __syspc8_bits;

/* SYS Pin Control Register 9 (SYSPC9) */
typedef struct {
  __REG32 ECPCLK_PS	      : 1;
  __REG32                 :31;
} __syspc9_bits;

/* SSW PLL BIST Control Register 1 (SSWPLL1) */
typedef struct {
  __REG32 EXT_COUNTER_EN  		: 1;
  __REG32 TAP_COUNTER_DIS 		: 3;
  __REG32 COUNTER_EN		  		: 1;
  __REG32 COUNTER_RESET   		: 1;
  __REG32 COUNTER_READ_READY	: 1;
  __REG32                 		: 1;
  __REG32 MOD_PH_CAP_INDEX		: 8;
  __REG32                 		:16;
} __sswpll1_bits;

/* Clock Source Disable Register (CSDIS) */
typedef struct {
  __REG32 CLKSR0OFF       : 1;
  __REG32 CLKSR1OFF       : 1;
  __REG32 CLKSR2OFF       : 1;
  __REG32 CLKSR3OFF       : 1;
  __REG32 CLKSR4OFF       : 1;
  __REG32 CLKSR5OFF       : 1;
  __REG32 CLKSR6OFF       : 1;
  __REG32 CLKSR7OFF       : 1;
  __REG32                 :24;
} __csdis_bits;

/* Clock Source Disable Set Register (CSDISSET) */
typedef struct {
  __REG32 SETCLKSR0OFF    : 1;
  __REG32 SETCLKSR1OFF    : 1;
  __REG32 SETCLKSR2OFF    : 1;
  __REG32 SETCLKSR3OFF    : 1;
  __REG32 SETCLKSR4OFF    : 1;
  __REG32 SETCLKSR5OFF    : 1;
  __REG32 SETCLKSR6OFF    : 1;
  __REG32 SETCLKSR7OFF    : 1;
  __REG32                 :24;
} __csdisset_bits;

/* Clock Source Disable Clear Register (CSDISCLR) */
typedef struct {
  __REG32 CLRCLKSR0OFF    : 1;
  __REG32 CLRCLKSR1OFF    : 1;
  __REG32 CLRCLKSR2OFF    : 1;
  __REG32 CLRCLKSR3OFF    : 1;
  __REG32 CLRCLKSR4OFF    : 1;
  __REG32 CLRCLKSR5OFF    : 1;
  __REG32 CLRCLKSR6OFF    : 1;
  __REG32 CLRCLKSR7OFF    : 1;
  __REG32                 :24;
} __csdisclr_bits;

/* Clock Domain Disable Register (CDDIS) */
typedef struct {
  __REG32 GCLKOFF         : 1;
  __REG32 HCLKOFF         : 1;
  __REG32 VCLKPOFF        : 1;
  __REG32 VCLK2OFF        : 1;
  __REG32 VCLKA1OFF       : 1;
  __REG32 VCLKA2OFF       : 1;
  __REG32 RTICLK1OFF      : 1;
  __REG32 RTICLK2OFF      : 1;
  __REG32                 :24;
} __cddis_bits;

/* Clock Domain Disable Set Register (CDDISSET) */
typedef struct {
  __REG32 SETGCLKOFF      : 1;
  __REG32 SETHCLKOFF      : 1;
  __REG32 SETVCLKPOFF     : 1;
  __REG32 SETVCLK2OFF     : 1;
  __REG32 SETVCLKA1OFF    : 1;
  __REG32 SETVCLKA2OFF    : 1;
  __REG32 SETRTICLK1OFF   : 1;
  __REG32 SETRTICLK2OFF   : 1;
  __REG32                 :24;
} __cddisset_bits;

/* Clock Domain Disable Clear Register (CDDISCLR) */
typedef struct {
  __REG32 CLRGCLKOFF      : 1;
  __REG32 CLRHCLKOFF      : 1;
  __REG32 CLRVCLKPOFF     : 1;
  __REG32 CLRVCLK2OFF     : 1;
  __REG32 CLRVCLKA1OFF    : 1;
  __REG32 CLRVCLKA2OFF    : 1;
  __REG32 CLRRTICLK1OFF   : 1;
  __REG32 CLRRTICLK2OFF   : 1;
  __REG32                 :24;
} __cddisclr_bits;

/* GCLK, HCLK, VCLK, and VCLK2 Source Register (GHVSRC) */
typedef struct {
  __REG32 GHVSRC          : 4;
  __REG32                 :12;
  __REG32 HVLPM           : 4;
  __REG32                 : 4;
  __REG32 GHVWAKE         : 4;
  __REG32                 : 4;
} __ghvsrc_bits;

/* Peripheral Asynchronous Clock Source Register (VCLKASRC) */
typedef struct {
  __REG32 VCLKA1S         : 4;
  __REG32                 : 4;
  __REG32 VCLKA2S         : 4;
  __REG32                 :20;
} __vclkasrc_bits;

/* RTI Clock Source Register (RCLKSRC) */
typedef struct {
  __REG32 RTI1SRC         : 4;
  __REG32                 : 4;
  __REG32 RTI1DIV         : 2;
  __REG32                 : 6;
  __REG32 RTI2SRC         : 4;
  __REG32                 : 4;
  __REG32 RTI2DIV         : 2;
  __REG32                 : 6;
} __rclksrc_bits;

/* Clock Source Valid Status Register (CSVSTAT) */
typedef struct {
  __REG32 CLKSR0V         : 1;
  __REG32 CLKSR1V         : 1;
  __REG32 CLKSR2V         : 1;
  __REG32 CLKSR3V         : 1;
  __REG32 CLKSR4V         : 1;
  __REG32 CLKSR5V         : 1;
  __REG32 CLKSR6V         : 1;
  __REG32 CLKSR7V         : 1;
  __REG32                 :24;
} __csvstat_bits;

/* Memory Self-Test Global Control Register (MSTGCR) */
typedef struct {
  __REG32 MSTGENA         : 4;
  __REG32                 : 4;
  __REG32 ROM_DIV         : 2;
  __REG32                 : 6;
  __REG32 MBIST_ALGSEL    : 8;
  __REG32                 : 8;
} __mstgcr_bits;

/* Memory Hardware Initialization Global Control Register (MINITGCR) */
typedef struct {
  __REG32 MINITGENA       : 4;
  __REG32                 :28;
} __minitgcr_bits;

/* MBIST Controller/Memory Initialization Enable Register (MSINENA) */
typedef struct {
  __REG32 MSIENA0         : 1;
  __REG32 MSIENA1         : 1;
  __REG32 MSIENA2         : 1;
  __REG32 MSIENA3         : 1;
  __REG32 MSIENA4         : 1;
  __REG32 MSIENA5         : 1;
  __REG32 MSIENA6         : 1;
  __REG32 MSIENA7         : 1;
  __REG32 MSIENA8         : 1;
  __REG32 MSIENA9         : 1;
  __REG32 MSIENA10        : 1;
  __REG32 MSIENA11        : 1;
  __REG32 MSIENA12        : 1;
  __REG32 MSIENA13        : 1;
  __REG32 MSIENA14        : 1;
  __REG32 MSIENA15        : 1;
  __REG32 MSIENA16        : 1;
  __REG32 MSIENA17        : 1;
  __REG32 MSIENA18        : 1;
  __REG32 MSIENA19        : 1;
  __REG32 MSIENA20        : 1;
  __REG32 MSIENA21        : 1;
  __REG32 MSIENA22        : 1;
  __REG32 MSIENA23        : 1;
  __REG32 MSIENA24        : 1;
  __REG32 MSIENA25        : 1;
  __REG32 MSIENA26        : 1;
  __REG32 MSIENA27        : 1;
  __REG32 MSIENA28        : 1;
  __REG32 MSIENA29        : 1;
  __REG32 MSIENA30        : 1;
  __REG32 MSIENA31        : 1;
} __msiena_bits;

/* Memory Self-Test Fail Status Register (MSTFAIL) */
typedef struct {
  __REG32 MSTF0         : 1;
  __REG32 MSTF1         : 1;
  __REG32 MSTF2         : 1;
  __REG32 MSTF3         : 1;
  __REG32 MSTF4         : 1;
  __REG32 MSTF5         : 1;
  __REG32 MSTF6         : 1;
  __REG32 MSTF7         : 1;
  __REG32 MSTF8         : 1;
  __REG32 MSTF9         : 1;
  __REG32 MSTF10        : 1;
  __REG32 MSTF11        : 1;
  __REG32 MSTF12        : 1;
  __REG32 MSTF13        : 1;
  __REG32 MSTF14        : 1;
  __REG32 MSTF15        : 1;
  __REG32 MSTF16        : 1;
  __REG32 MSTF17        : 1;
  __REG32 MSTF18        : 1;
  __REG32 MSTF19        : 1;
  __REG32 MSTF20        : 1;
  __REG32 MSTF21        : 1;
  __REG32 MSTF22        : 1;
  __REG32 MSTF23        : 1;
  __REG32 MSTF24        : 1;
  __REG32 MSTF25        : 1;
  __REG32 MSTF26        : 1;
  __REG32 MSTF27        : 1;
  __REG32 MSTF28        : 1;
  __REG32 MSTF29        : 1;
  __REG32 MSTF30        : 1;
  __REG32 MSTF31        : 1;
} __mstfail_bits;

/* MSTC Global Status Register (MSTCGSTAT) */
typedef struct {
  __REG32 MSTDONE         : 1;
  __REG32                 : 7;
  __REG32 MINIDONE        : 1;
  __REG32                 :23;
} __mstcgstat_bits;

/* Memory Hardware Initialization Status Register (MINISTAT) */
typedef struct {
  __REG32 MIDONE0         : 1;
  __REG32 MIDONE1         : 1;
  __REG32 MIDONE2         : 1;
  __REG32 MIDONE3         : 1;
  __REG32 MIDONE4         : 1;
  __REG32 MIDONE5         : 1;
  __REG32 MIDONE6         : 1;
  __REG32 MIDONE7         : 1;
  __REG32 MIDONE8         : 1;
  __REG32 MIDONE9         : 1;
  __REG32 MIDONE10        : 1;
  __REG32 MIDONE11        : 1;
  __REG32 MIDONE12        : 1;
  __REG32 MIDONE13        : 1;
  __REG32 MIDONE14        : 1;
  __REG32 MIDONE15        : 1;
  __REG32 MIDONE16        : 1;
  __REG32 MIDONE17        : 1;
  __REG32 MIDONE18        : 1;
  __REG32 MIDONE19        : 1;
  __REG32 MIDONE20        : 1;
  __REG32 MIDONE21        : 1;
  __REG32 MIDONE22        : 1;
  __REG32 MIDONE23        : 1;
  __REG32 MIDONE24        : 1;
  __REG32 MIDONE25        : 1;
  __REG32 MIDONE26        : 1;
  __REG32 MIDONE27        : 1;
  __REG32 MIDONE28        : 1;
  __REG32 MIDONE29        : 1;
  __REG32 MIDONE30        : 1;
  __REG32 MIDONE31        : 1;
} __ministat_bits;

/* PLL Control Register 1 (PLLCTL1) */
typedef struct {
  __REG32 PLLMUL          :16;
  __REG32 REFCLKDIV       : 6;
  __REG32                 : 1;
  __REG32 ROF             : 1;
  __REG32 PLLDIV          : 5;
  __REG32 BPOS            : 2;
  __REG32 ROS             : 1;
} __pllctl1_bits;

/* PLL Control Register 2 (PLLCTL2) */
typedef struct {
  __REG32 SPR_AMOUNT      : 9;
  __REG32 ODPLL           : 3;
  __REG32 MULMOD          : 9;
  __REG32                 : 1;
  __REG32 SPREADINGRATE   : 9;
  __REG32 FMENA           : 1;
} __pllctl2_bits;

/* Voltage Regulator Control Register (VRCTL) */
typedef struct {
  __REG32 VSLEEPENA           : 4;
  __REG32 VLPMENA		          : 4;
  __REG32                     :24;
} __vrctl_bits;

/* LPO/Clock Monitor Control Register (LPOMONCTL) */
typedef struct {
  __REG32 LFTRIM              : 4;
  __REG32                     : 4;
  __REG32 HFTRIM              : 4;
  __REG32                     :12;
  __REG32 BIAS_ENABLE         : 1;
  __REG32                     : 7;
} __lpomonctl_bits;

/* Clock Test Register (CLKTEST) */
typedef struct {
  __REG32 SEL_ECP_PIN         : 4;
  __REG32                     : 4;
  __REG32 SEL_GIO_PIN         : 4;
  __REG32                     : 4;
  __REG32 CLK_TEST_EN         : 4;
  __REG32                     : 4;
  __REG32 RANGEDETENSSEL      : 1;
  __REG32 RANGEDETCTRL        : 1;
  __REG32                     : 6;
} __clktest_bits;

/* General Purpose Register (GPREG1) */
typedef struct {
  __REG32 ECLK_SBEN										: 1;
  __REG32 RST_SBEN										: 1;
  __REG32 HET_SBEN										: 1;
  __REG32 LIN1SCI1_SBEN								: 1;
  __REG32 LIN2SCI2_SBEN								: 1;
  __REG32 MIBSPI1											: 1;
  __REG32 MIBSPIP2_SBEN								: 1;
  __REG32 														: 1;
  __REG32 ADC_ADEVT_SBEN							: 1;
  __REG32 DCAN1_SBEN									: 1;
  __REG32 DCAN2_SBEN									: 1;
  __REG32 GIOA_SBEN										: 1;
  __REG32                     				:20;
} __gpreg1_bits;

/* Imprecise Fault Status Register (IMPFASTS) */
typedef struct {
  __REG32 ATYPE               : 1;
  __REG32                     : 7;
  __REG32 VBUSA               : 1;
  __REG32 NCBA                : 1;
  __REG32 EMIFA               : 1;
  __REG32                     : 5;
  __REG32 MASTERID            : 8;
  __REG32                     : 8;
} __impfasts_bits;

/* System Software Interrupt Request 1 Register (SSIR1) */
typedef struct {
  __REG32 SSDATA1             : 8;
  __REG32 SSKEY1              : 8;
  __REG32                     :16;
} __ssir1_bits;

/* System Software Interrupt Request 2 Register (SSIR2) */
typedef struct {
  __REG32 SSDATA2             : 8;
  __REG32 SSKEY2              : 8;
  __REG32                     :16;
} __ssir2_bits;

/* System Software Interrupt Request 3 Register (SSIR3) */
typedef struct {
  __REG32 SSDATA3             : 8;
  __REG32 SSKEY3              : 8;
  __REG32                     :16;
} __ssir3_bits;

/* System Software Interrupt Request 4 Register (SSIR4) */
typedef struct {
  __REG32 SSDATA4             : 8;
  __REG32 SSKEY4              : 8;
  __REG32                     :16;
} __ssir4_bits;

/* RAM Control Register (RAMGCR) */
typedef struct {
  __REG32 WST_DENA0           : 1;
  __REG32                     : 1;
  __REG32 WST_AENA0           : 1;
  __REG32                     : 1;
  __REG32 WST_DENA1           : 1;
  __REG32                     : 1;
  __REG32 WST_AENA1           : 1;
  __REG32                     : 1;
  __REG32 WST_DENA2           : 1;
  __REG32                     : 1;
  __REG32 WST_AENA2           : 1;
  __REG32                     : 1;
  __REG32 WST_DENA3           : 1;
  __REG32                     : 1;
  __REG32 WST_AENA3           : 1;
  __REG32                     : 1;
  __REG32 RAM_DFT_EN          : 4;
  __REG32                     :12;
} __ramgcr_bits;

/* Bus Matrix Module Control Register 1 (BMMCR1) */
typedef struct {
  __REG32 MEMSW               : 4;
  __REG32                     :28;
} __bmmcr1_bits;

/* Bus Matrix Module Control Register 2 (BMMCR2) */
typedef struct {
  __REG32 PRTY_RAM0           : 1;
  __REG32 PRTY_FLASH          : 1;
  __REG32 PRTY_PRG            : 1;
  __REG32 PRTY_CRC            : 1;
  __REG32 PRTY_RAM2           : 1;
  __REG32 PRTY_RAM3           : 1;
  __REG32 PRTY_HPI	          : 1;
  __REG32                     :25;
} __bmmcr2_bits;

/* MMU Global Control Register (MMUGCR) */
typedef struct {
  __REG32 MPMENA              : 1;
  __REG32                     :31;
} __mmugcr_bits;

/* Clock Control Register (CLKCNTL) */
typedef struct {
  __REG32                     : 8;
  __REG32 PENA		            : 1;
  __REG32                     : 7;
  __REG32 VCLKR		            : 4;
  __REG32                     : 4;
  __REG32 VCLK2R	            : 4;
  __REG32                     : 4;
} __clkcntl_bits;

/* ECP Control Register (ECPCNTL) */
typedef struct {
  __REG32 ECPDIV              :16;
  __REG32                     : 7;
  __REG32 ECPCOS	            : 1;
  __REG32 ECPSSEL	            : 1;
  __REG32                     : 7;
} __ecpcntl_bits;

/* DEV Parity Control Register1 (DEVCR1) */
typedef struct {
  __REG32 DEVPARSEL           : 4;
  __REG32                     :28;
} __devcr1_bits;

/* System Exception Control Register (SYSECR) */
typedef struct {
  __REG32                     :14;
  __REG32 RESET0	            : 2;
  __REG32                     :16;
} __sysecr_bits;

/* System Exception Status Register (SYSESR) */
typedef struct {
  __REG32                     : 2;
  __REG32 VSWRST	            : 1;
  __REG32 EXTRST	            : 1;
  __REG32 SWRST 	            : 1;
  __REG32 CPURST	            : 1;
  __REG32                     : 7;
  __REG32 WDRST 	            : 1;
  __REG32 OSCRST 	            : 1;
  __REG32 PORST 	            : 1;
  __REG32                     :16;
} __sysesr_bits;

/* System Test Abort Status Register (SYSTASR) */
typedef struct {
  __REG32 OSCFAIL		          : 1;
  __REG32                     : 7;
  __REG32 RFSLIP		          : 1;
  __REG32 FBSLIP		          : 1;
  __REG32                     :22;
} __glbstat_bits;

/* Device Identification Register (DEVID) */
typedef struct {
  __REG32 Platform_ID         : 3;
  __REG32 Version             : 5;
  __REG32 RAM_ECC             : 1;
  __REG32 FLASH_ECC      			: 2;
  __REG32 PPAR                : 1;
  __REG32 IOV                 : 1;
  __REG32 TECH                : 4;
  __REG32 ID                  :14;
  __REG32 CP15                : 1;
} __devid_bits;

/* Software Interrupt Vector Register (SSIVEC) */
typedef struct {
  __REG32 SSIVECT             : 8;
  __REG32 SSIDATA             : 8;
  __REG32                     :16;
} __ssivec_bits;

/* System Software Interrupt Flag Register (SSIF) */
typedef struct {
  __REG32 SSI_FLAG1           : 1;
  __REG32 SSI_FLAG2           : 1;
  __REG32 SSI_FLAG3           : 1;
  __REG32 SSI_FLAG4           : 1;
  __REG32                     :28;
} __ssif_bits;

/* Peripheral Memory Protection Set Register 0 (PMPROTSET0) */
typedef struct {
  __REG32 PCS0PROTSET         : 1;
  __REG32 PCS1PROTSET         : 1;
  __REG32 PCS2PROTSET         : 1;
  __REG32 PCS3PROTSET         : 1;
  __REG32 PCS4PROTSET         : 1;
  __REG32 PCS5PROTSET         : 1;
  __REG32 PCS6PROTSET         : 1;
  __REG32 PCS7PROTSET         : 1;
  __REG32 PCS8PROTSET         : 1;
  __REG32 PCS9PROTSET         : 1;
  __REG32 PCS10PROTSET        : 1;
  __REG32 PCS11PROTSET        : 1;
  __REG32 PCS12PROTSET        : 1;
  __REG32 PCS13PROTSET        : 1;
  __REG32 PCS14PROTSET        : 1;
  __REG32 PCS15PROTSET        : 1;
  __REG32 PCS16PROTSET        : 1;
  __REG32 PCS17PROTSET        : 1;
  __REG32 PCS18PROTSET        : 1;
  __REG32 PCS19PROTSET        : 1;
  __REG32 PCS20PROTSET        : 1;
  __REG32 PCS21PROTSET        : 1;
  __REG32 PCS22PROTSET        : 1;
  __REG32 PCS23PROTSET        : 1;
  __REG32 PCS24PROTSET        : 1;
  __REG32 PCS25PROTSET        : 1;
  __REG32 PCS26PROTSET        : 1;
  __REG32 PCS27PROTSET        : 1;
  __REG32 PCS28PROTSET        : 1;
  __REG32 PCS29PROTSET        : 1;
  __REG32 PCS30PROTSET        : 1;
  __REG32 PCS31PROTSET        : 1;
} __pmprotset0_bits;

/* Peripheral Memory Protection Set Register 1 (PMPROTSET1) */
typedef struct {
  __REG32 PCS32PROTSET        : 1;
  __REG32 PCS33PROTSET        : 1;
  __REG32 PCS34PROTSET        : 1;
  __REG32 PCS35PROTSET        : 1;
  __REG32 PCS36PROTSET        : 1;
  __REG32 PCS37PROTSET        : 1;
  __REG32 PCS38PROTSET        : 1;
  __REG32 PCS39PROTSET        : 1;
  __REG32 PCS40PROTSET        : 1;
  __REG32 PCS41PROTSET        : 1;
  __REG32 PCS42PROTSET        : 1;
  __REG32 PCS43PROTSET        : 1;
  __REG32 PCS44PROTSET        : 1;
  __REG32 PCS45PROTSET        : 1;
  __REG32 PCS46PROTSET        : 1;
  __REG32 PCS47PROTSET        : 1;
  __REG32 PCS48PROTSET        : 1;
  __REG32 PCS49PROTSET        : 1;
  __REG32 PCS50PROTSET        : 1;
  __REG32 PCS51PROTSET        : 1;
  __REG32 PCS52PROTSET        : 1;
  __REG32 PCS53PROTSET        : 1;
  __REG32 PCS54PROTSET        : 1;
  __REG32 PCS55PROTSET        : 1;
  __REG32 PCS56PROTSET        : 1;
  __REG32 PCS57PROTSET        : 1;
  __REG32 PCS58PROTSET        : 1;
  __REG32 PCS59PROTSET        : 1;
  __REG32 PCS60PROTSET        : 1;
  __REG32 PCS61PROTSET        : 1;
  __REG32 PCS62PROTSET        : 1;
  __REG32 PCS63PROTSET        : 1;
} __pmprotset1_bits;

/* Peripheral Memory Protection Clear Register 0 (PMPROTCLR0) */
typedef struct {
  __REG32 PCS0PROTCLR         : 1;
  __REG32 PCS1PROTCLR         : 1;
  __REG32 PCS2PROTCLR         : 1;
  __REG32 PCS3PROTCLR         : 1;
  __REG32 PCS4PROTCLR         : 1;
  __REG32 PCS5PROTCLR         : 1;
  __REG32 PCS6PROTCLR         : 1;
  __REG32 PCS7PROTCLR         : 1;
  __REG32 PCS8PROTCLR         : 1;
  __REG32 PCS9PROTCLR         : 1;
  __REG32 PCS10PROTCLR        : 1;
  __REG32 PCS11PROTCLR        : 1;
  __REG32 PCS12PROTCLR        : 1;
  __REG32 PCS13PROTCLR        : 1;
  __REG32 PCS14PROTCLR        : 1;
  __REG32 PCS15PROTCLR        : 1;
  __REG32 PCS16PROTCLR        : 1;
  __REG32 PCS17PROTCLR        : 1;
  __REG32 PCS18PROTCLR        : 1;
  __REG32 PCS19PROTCLR        : 1;
  __REG32 PCS20PROTCLR        : 1;
  __REG32 PCS21PROTCLR        : 1;
  __REG32 PCS22PROTCLR        : 1;
  __REG32 PCS23PROTCLR        : 1;
  __REG32 PCS24PROTCLR        : 1;
  __REG32 PCS25PROTCLR        : 1;
  __REG32 PCS26PROTCLR        : 1;
  __REG32 PCS27PROTCLR        : 1;
  __REG32 PCS28PROTCLR        : 1;
  __REG32 PCS29PROTCLR        : 1;
  __REG32 PCS30PROTCLR        : 1;
  __REG32 PCS31PROTCLR        : 1;
} __pmprotclr0_bits;

/* Peripheral Memory Protection Clear Register 1 (PMPROTCLR1) */
typedef struct {
  __REG32 PCS32PROTCLR        : 1;
  __REG32 PCS33PROTCLR        : 1;
  __REG32 PCS34PROTCLR        : 1;
  __REG32 PCS35PROTCLR        : 1;
  __REG32 PCS36PROTCLR        : 1;
  __REG32 PCS37PROTCLR        : 1;
  __REG32 PCS38PROTCLR        : 1;
  __REG32 PCS39PROTCLR        : 1;
  __REG32 PCS40PROTCLR        : 1;
  __REG32 PCS41PROTCLR        : 1;
  __REG32 PCS42PROTCLR        : 1;
  __REG32 PCS43PROTCLR        : 1;
  __REG32 PCS44PROTCLR        : 1;
  __REG32 PCS45PROTCLR        : 1;
  __REG32 PCS46PROTCLR        : 1;
  __REG32 PCS47PROTCLR        : 1;
  __REG32 PCS48PROTCLR        : 1;
  __REG32 PCS49PROTCLR        : 1;
  __REG32 PCS50PROTCLR        : 1;
  __REG32 PCS51PROTCLR        : 1;
  __REG32 PCS52PROTCLR        : 1;
  __REG32 PCS53PROTCLR        : 1;
  __REG32 PCS54PROTCLR        : 1;
  __REG32 PCS55PROTCLR        : 1;
  __REG32 PCS56PROTCLR        : 1;
  __REG32 PCS57PROTCLR        : 1;
  __REG32 PCS58PROTCLR        : 1;
  __REG32 PCS59PROTCLR        : 1;
  __REG32 PCS60PROTCLR        : 1;
  __REG32 PCS61PROTCLR        : 1;
  __REG32 PCS62PROTCLR        : 1;
  __REG32 PCS63PROTCLR        : 1;
} __pmprotclr1_bits;

/* Peripheral Protection Set Register 0 (PPROTSET0) */
typedef struct {
  __REG32 PS0QUAD0PROTSET     : 1;
  __REG32 PS0QUAD1PROTSET     : 1;
  __REG32 PS0QUAD2PROTSET     : 1;
  __REG32 PS0QUAD3PROTSET     : 1;
  __REG32 PS1QUAD0PROTSET     : 1;
  __REG32 PS1QUAD1PROTSET     : 1;
  __REG32 PS1QUAD2PROTSET     : 1;
  __REG32 PS1QUAD3PROTSET     : 1;
  __REG32 PS2QUAD0PROTSET     : 1;
  __REG32 PS2QUAD1PROTSET     : 1;
  __REG32 PS2QUAD2PROTSET     : 1;
  __REG32 PS2QUAD3PROTSET     : 1;
  __REG32 PS3QUAD0PROTSET     : 1;
  __REG32 PS3QUAD1PROTSET     : 1;
  __REG32 PS3QUAD2PROTSET     : 1;
  __REG32 PS3QUAD3PROTSET     : 1;
  __REG32 PS4QUAD0PROTSET     : 1;
  __REG32 PS4QUAD1PROTSET     : 1;
  __REG32 PS4QUAD2PROTSET     : 1;
  __REG32 PS4QUAD3PROTSET     : 1;
  __REG32 PS5QUAD0PROTSET     : 1;
  __REG32 PS5QUAD1PROTSET     : 1;
  __REG32 PS5QUAD2PROTSET     : 1;
  __REG32 PS5QUAD3PROTSET     : 1;
  __REG32 PS6QUAD0PROTSET     : 1;
  __REG32 PS6QUAD1PROTSET     : 1;
  __REG32 PS6QUAD2PROTSET     : 1;
  __REG32 PS6QUAD3PROTSET     : 1;
  __REG32 PS7QUAD0PROTSET     : 1;
  __REG32 PS7QUAD1PROTSET     : 1;
  __REG32 PS7QUAD2PROTSET     : 1;
  __REG32 PS7QUAD3PROTSET     : 1;
} __pprotset0_bits;

/* Peripheral Protection Set Register 1 (PPROTSET1) */
typedef struct {
  __REG32 PS8QUAD0PROTSET     : 1;
  __REG32 PS8QUAD1PROTSET     : 1;
  __REG32 PS8QUAD2PROTSET     : 1;
  __REG32 PS8QUAD3PROTSET     : 1;
  __REG32 PS9QUAD0PROTSET     : 1;
  __REG32 PS9QUAD1PROTSET     : 1;
  __REG32 PS9QUAD2PROTSET     : 1;
  __REG32 PS9QUAD3PROTSET     : 1;
  __REG32 PS10QUAD0PROTSET    : 1;
  __REG32 PS10QUAD1PROTSET    : 1;
  __REG32 PS10QUAD2PROTSET    : 1;
  __REG32 PS10QUAD3PROTSET    : 1;
  __REG32 PS11QUAD0PROTSET    : 1;
  __REG32 PS11QUAD1PROTSET    : 1;
  __REG32 PS11QUAD2PROTSET    : 1;
  __REG32 PS11QUAD3PROTSET    : 1;
  __REG32 PS12QUAD0PROTSET    : 1;
  __REG32 PS12QUAD1PROTSET    : 1;
  __REG32 PS12QUAD2PROTSET    : 1;
  __REG32 PS12QUAD3PROTSET    : 1;
  __REG32 PS13QUAD0PROTSET    : 1;
  __REG32 PS13QUAD1PROTSET    : 1;
  __REG32 PS13QUAD2PROTSET    : 1;
  __REG32 PS13QUAD3PROTSET    : 1;
  __REG32 PS14QUAD0PROTSET    : 1;
  __REG32 PS14QUAD1PROTSET    : 1;
  __REG32 PS14QUAD2PROTSET    : 1;
  __REG32 PS14QUAD3PROTSET    : 1;
  __REG32 PS15QUAD0PROTSET    : 1;
  __REG32 PS15QUAD1PROTSET    : 1;
  __REG32 PS15QUAD2PROTSET    : 1;
  __REG32 PS15QUAD3PROTSET    : 1;
} __pprotset1_bits;

/* Peripheral Protection Set Register 2 (PPROTSET2) */
typedef struct {
  __REG32 PS16QUAD0PROTSET    : 1;
  __REG32 PS16QUAD1PROTSET    : 1;
  __REG32 PS16QUAD2PROTSET    : 1;
  __REG32 PS16QUAD3PROTSET    : 1;
  __REG32 PS17QUAD0PROTSET    : 1;
  __REG32 PS17QUAD1PROTSET    : 1;
  __REG32 PS17QUAD2PROTSET    : 1;
  __REG32 PS17QUAD3PROTSET    : 1;
  __REG32 PS18QUAD0PROTSET    : 1;
  __REG32 PS18QUAD1PROTSET    : 1;
  __REG32 PS18QUAD2PROTSET    : 1;
  __REG32 PS18QUAD3PROTSET    : 1;
  __REG32 PS19QUAD0PROTSET    : 1;
  __REG32 PS19QUAD1PROTSET    : 1;
  __REG32 PS19QUAD2PROTSET    : 1;
  __REG32 PS19QUAD3PROTSET    : 1;
  __REG32 PS20QUAD0PROTSET    : 1;
  __REG32 PS20QUAD1PROTSET    : 1;
  __REG32 PS20QUAD2PROTSET    : 1;
  __REG32 PS20QUAD3PROTSET    : 1;
  __REG32 PS21QUAD0PROTSET    : 1;
  __REG32 PS21QUAD1PROTSET    : 1;
  __REG32 PS21QUAD2PROTSET    : 1;
  __REG32 PS21QUAD3PROTSET    : 1;
  __REG32 PS22QUAD0PROTSET    : 1;
  __REG32 PS22QUAD1PROTSET    : 1;
  __REG32 PS22QUAD2PROTSET    : 1;
  __REG32 PS22QUAD3PROTSET    : 1;
  __REG32 PS23QUAD0PROTSET    : 1;
  __REG32 PS23QUAD1PROTSET    : 1;
  __REG32 PS23QUAD2PROTSET    : 1;
  __REG32 PS23QUAD3PROTSET    : 1;
} __pprotset2_bits;

/* Peripheral Protection Set Register 3 (PPROTSET3) */
typedef struct {
  __REG32 PS24QUAD0PROTSET    : 1;
  __REG32 PS24QUAD1PROTSET    : 1;
  __REG32 PS24QUAD2PROTSET    : 1;
  __REG32 PS24QUAD3PROTSET    : 1;
  __REG32 PS25QUAD0PROTSET    : 1;
  __REG32 PS25QUAD1PROTSET    : 1;
  __REG32 PS25QUAD2PROTSET    : 1;
  __REG32 PS25QUAD3PROTSET    : 1;
  __REG32 PS26QUAD0PROTSET    : 1;
  __REG32 PS26QUAD1PROTSET    : 1;
  __REG32 PS26QUAD2PROTSET    : 1;
  __REG32 PS26QUAD3PROTSET    : 1;
  __REG32 PS27QUAD0PROTSET    : 1;
  __REG32 PS27QUAD1PROTSET    : 1;
  __REG32 PS27QUAD2PROTSET    : 1;
  __REG32 PS27QUAD3PROTSET    : 1;
  __REG32 PS28QUAD0PROTSET    : 1;
  __REG32 PS28QUAD1PROTSET    : 1;
  __REG32 PS28QUAD2PROTSET    : 1;
  __REG32 PS28QUAD3PROTSET    : 1;
  __REG32 PS29QUAD0PROTSET    : 1;
  __REG32 PS29QUAD1PROTSET    : 1;
  __REG32 PS29QUAD2PROTSET    : 1;
  __REG32 PS29QUAD3PROTSET    : 1;
  __REG32 PS30QUAD0PROTSET    : 1;
  __REG32 PS30QUAD1PROTSET    : 1;
  __REG32 PS30QUAD2PROTSET    : 1;
  __REG32 PS30QUAD3PROTSET    : 1;
  __REG32 PS31QUAD0PROTSET    : 1;
  __REG32 PS31QUAD1PROTSET    : 1;
  __REG32 PS31QUAD2PROTSET    : 1;
  __REG32 PS31QUAD3PROTSET    : 1;
} __pprotset3_bits;

/* Peripheral Protection Clear Register 0 (PPROTCLR0) */
typedef struct {
  __REG32 PS0QUAD0PROTCLR     : 1;
  __REG32 PS0QUAD1PROTCLR     : 1;
  __REG32 PS0QUAD2PROTCLR     : 1;
  __REG32 PS0QUAD3PROTCLR     : 1;
  __REG32 PS1QUAD0PROTCLR     : 1;
  __REG32 PS1QUAD1PROTCLR     : 1;
  __REG32 PS1QUAD2PROTCLR     : 1;
  __REG32 PS1QUAD3PROTCLR     : 1;
  __REG32 PS2QUAD0PROTCLR     : 1;
  __REG32 PS2QUAD1PROTCLR     : 1;
  __REG32 PS2QUAD2PROTCLR     : 1;
  __REG32 PS2QUAD3PROTCLR     : 1;
  __REG32 PS3QUAD0PROTCLR     : 1;
  __REG32 PS3QUAD1PROTCLR     : 1;
  __REG32 PS3QUAD2PROTCLR     : 1;
  __REG32 PS3QUAD3PROTCLR     : 1;
  __REG32 PS4QUAD0PROTCLR     : 1;
  __REG32 PS4QUAD1PROTCLR     : 1;
  __REG32 PS4QUAD2PROTCLR     : 1;
  __REG32 PS4QUAD3PROTCLR     : 1;
  __REG32 PS5QUAD0PROTCLR     : 1;
  __REG32 PS5QUAD1PROTCLR     : 1;
  __REG32 PS5QUAD2PROTCLR     : 1;
  __REG32 PS5QUAD3PROTCLR     : 1;
  __REG32 PS6QUAD0PROTCLR     : 1;
  __REG32 PS6QUAD1PROTCLR     : 1;
  __REG32 PS6QUAD2PROTCLR     : 1;
  __REG32 PS6QUAD3PROTCLR     : 1;
  __REG32 PS7QUAD0PROTCLR     : 1;
  __REG32 PS7QUAD1PROTCLR     : 1;
  __REG32 PS7QUAD2PROTCLR     : 1;
  __REG32 PS7QUAD3PROTCLR     : 1;
} __pprotclr0_bits;

/* Peripheral Protection Clear Register 1 (PPROTCLR1) */
typedef struct {
  __REG32 PS8QUAD0PROTCLR     : 1;
  __REG32 PS8QUAD1PROTCLR     : 1;
  __REG32 PS8QUAD2PROTCLR     : 1;
  __REG32 PS8QUAD3PROTCLR     : 1;
  __REG32 PS9QUAD0PROTCLR     : 1;
  __REG32 PS9QUAD1PROTCLR     : 1;
  __REG32 PS9QUAD2PROTCLR     : 1;
  __REG32 PS9QUAD3PROTCLR     : 1;
  __REG32 PS10QUAD0PROTCLR    : 1;
  __REG32 PS10QUAD1PROTCLR    : 1;
  __REG32 PS10QUAD2PROTCLR    : 1;
  __REG32 PS10QUAD3PROTCLR    : 1;
  __REG32 PS11QUAD0PROTCLR    : 1;
  __REG32 PS11QUAD1PROTCLR    : 1;
  __REG32 PS11QUAD2PROTCLR    : 1;
  __REG32 PS11QUAD3PROTCLR    : 1;
  __REG32 PS12QUAD0PROTCLR    : 1;
  __REG32 PS12QUAD1PROTCLR    : 1;
  __REG32 PS12QUAD2PROTCLR    : 1;
  __REG32 PS12QUAD3PROTCLR    : 1;
  __REG32 PS13QUAD0PROTCLR    : 1;
  __REG32 PS13QUAD1PROTCLR    : 1;
  __REG32 PS13QUAD2PROTCLR    : 1;
  __REG32 PS13QUAD3PROTCLR    : 1;
  __REG32 PS14QUAD0PROTCLR    : 1;
  __REG32 PS14QUAD1PROTCLR    : 1;
  __REG32 PS14QUAD2PROTCLR    : 1;
  __REG32 PS14QUAD3PROTCLR    : 1;
  __REG32 PS15QUAD0PROTCLR    : 1;
  __REG32 PS15QUAD1PROTCLR    : 1;
  __REG32 PS15QUAD2PROTCLR    : 1;
  __REG32 PS15QUAD3PROTCLR    : 1;
} __pprotclr1_bits;

/* Peripheral Protection Clear Register 2 (PPROTCLR2) */
typedef struct {
  __REG32 PS16QUAD0PROTCLR    : 1;
  __REG32 PS16QUAD1PROTCLR    : 1;
  __REG32 PS16QUAD2PROTCLR    : 1;
  __REG32 PS16QUAD3PROTCLR    : 1;
  __REG32 PS17QUAD0PROTCLR    : 1;
  __REG32 PS17QUAD1PROTCLR    : 1;
  __REG32 PS17QUAD2PROTCLR    : 1;
  __REG32 PS17QUAD3PROTCLR    : 1;
  __REG32 PS18QUAD0PROTCLR    : 1;
  __REG32 PS18QUAD1PROTCLR    : 1;
  __REG32 PS18QUAD2PROTCLR    : 1;
  __REG32 PS18QUAD3PROTCLR    : 1;
  __REG32 PS19QUAD0PROTCLR    : 1;
  __REG32 PS19QUAD1PROTCLR    : 1;
  __REG32 PS19QUAD2PROTCLR    : 1;
  __REG32 PS19QUAD3PROTCLR    : 1;
  __REG32 PS20QUAD0PROTCLR    : 1;
  __REG32 PS20QUAD1PROTCLR    : 1;
  __REG32 PS20QUAD2PROTCLR    : 1;
  __REG32 PS20QUAD3PROTCLR    : 1;
  __REG32 PS21QUAD0PROTCLR    : 1;
  __REG32 PS21QUAD1PROTCLR    : 1;
  __REG32 PS21QUAD2PROTCLR    : 1;
  __REG32 PS21QUAD3PROTCLR    : 1;
  __REG32 PS22QUAD0PROTCLR    : 1;
  __REG32 PS22QUAD1PROTCLR    : 1;
  __REG32 PS22QUAD2PROTCLR    : 1;
  __REG32 PS22QUAD3PROTCLR    : 1;
  __REG32 PS23QUAD0PROTCLR    : 1;
  __REG32 PS23QUAD1PROTCLR    : 1;
  __REG32 PS23QUAD2PROTCLR    : 1;
  __REG32 PS23QUAD3PROTCLR    : 1;
} __pprotclr2_bits;

/* Peripheral Protection Clear Register 3 (PPROTCLR3) */
typedef struct {
  __REG32 PS24QUAD0PROTCLR    : 1;
  __REG32 PS24QUAD1PROTCLR    : 1;
  __REG32 PS24QUAD2PROTCLR    : 1;
  __REG32 PS24QUAD3PROTCLR    : 1;
  __REG32 PS25QUAD0PROTCLR    : 1;
  __REG32 PS25QUAD1PROTCLR    : 1;
  __REG32 PS25QUAD2PROTCLR    : 1;
  __REG32 PS25QUAD3PROTCLR    : 1;
  __REG32 PS26QUAD0PROTCLR    : 1;
  __REG32 PS26QUAD1PROTCLR    : 1;
  __REG32 PS26QUAD2PROTCLR    : 1;
  __REG32 PS26QUAD3PROTCLR    : 1;
  __REG32 PS27QUAD0PROTCLR    : 1;
  __REG32 PS27QUAD1PROTCLR    : 1;
  __REG32 PS27QUAD2PROTCLR    : 1;
  __REG32 PS27QUAD3PROTCLR    : 1;
  __REG32 PS28QUAD0PROTCLR    : 1;
  __REG32 PS28QUAD1PROTCLR    : 1;
  __REG32 PS28QUAD2PROTCLR    : 1;
  __REG32 PS28QUAD3PROTCLR    : 1;
  __REG32 PS29QUAD0PROTCLR    : 1;
  __REG32 PS29QUAD1PROTCLR    : 1;
  __REG32 PS29QUAD2PROTCLR    : 1;
  __REG32 PS29QUAD3PROTCLR    : 1;
  __REG32 PS30QUAD0PROTCLR    : 1;
  __REG32 PS30QUAD1PROTCLR    : 1;
  __REG32 PS30QUAD2PROTCLR    : 1;
  __REG32 PS30QUAD3PROTCLR    : 1;
  __REG32 PS31QUAD0PROTCLR    : 1;
  __REG32 PS31QUAD1PROTCLR    : 1;
  __REG32 PS31QUAD2PROTCLR    : 1;
  __REG32 PS31QUAD3PROTCLR    : 1;
} __pprotclr3_bits;

/* Peripheral Memory Power-Down Set Register 0 (PCSPWRDWNSET0) */
typedef struct {
  __REG32 PCS0PWRDWNSET       : 1;
  __REG32 PCS1PWRDWNSET       : 1;
  __REG32 PCS2PWRDWNSET       : 1;
  __REG32 PCS3PWRDWNSET       : 1;
  __REG32 PCS4PWRDWNSET       : 1;
  __REG32 PCS5PWRDWNSET       : 1;
  __REG32 PCS6PWRDWNSET       : 1;
  __REG32 PCS7PWRDWNSET       : 1;
  __REG32 PCS8PWRDWNSET       : 1;
  __REG32 PCS9PWRDWNSET       : 1;
  __REG32 PCS10PWRDWNSET      : 1;
  __REG32 PCS11PWRDWNSET      : 1;
  __REG32 PCS12PWRDWNSET      : 1;
  __REG32 PCS13PWRDWNSET      : 1;
  __REG32 PCS14PWRDWNSET      : 1;
  __REG32 PCS15PWRDWNSET      : 1;
  __REG32 PCS16PWRDWNSET      : 1;
  __REG32 PCS17PWRDWNSET      : 1;
  __REG32 PCS18PWRDWNSET      : 1;
  __REG32 PCS19PWRDWNSET      : 1;
  __REG32 PCS20PWRDWNSET      : 1;
  __REG32 PCS21PWRDWNSET      : 1;
  __REG32 PCS22PWRDWNSET      : 1;
  __REG32 PCS23PWRDWNSET      : 1;
  __REG32 PCS24PWRDWNSET      : 1;
  __REG32 PCS25PWRDWNSET      : 1;
  __REG32 PCS26PWRDWNSET      : 1;
  __REG32 PCS27PWRDWNSET      : 1;
  __REG32 PCS28PWRDWNSET      : 1;
  __REG32 PCS29PWRDWNSET      : 1;
  __REG32 PCS30PWRDWNSET      : 1;
  __REG32 PCS31PWRDWNSET      : 1;
} __pcspwrdwnset0_bits;

/* Peripheral Memory Power-Down Set Register 1 (PCSPWRDWNSET1) */
typedef struct {
  __REG32 PCS32PWRDWNSET      : 1;
  __REG32 PCS33PWRDWNSET      : 1;
  __REG32 PCS34PWRDWNSET      : 1;
  __REG32 PCS35PWRDWNSET      : 1;
  __REG32 PCS36PWRDWNSET      : 1;
  __REG32 PCS37PWRDWNSET      : 1;
  __REG32 PCS38PWRDWNSET      : 1;
  __REG32 PCS39PWRDWNSET      : 1;
  __REG32 PCS40PWRDWNSET      : 1;
  __REG32 PCS41PWRDWNSET      : 1;
  __REG32 PCS42PWRDWNSET      : 1;
  __REG32 PCS43PWRDWNSET      : 1;
  __REG32 PCS44PWRDWNSET      : 1;
  __REG32 PCS45PWRDWNSET      : 1;
  __REG32 PCS46PWRDWNSET      : 1;
  __REG32 PCS47PWRDWNSET      : 1;
  __REG32 PCS48PWRDWNSET      : 1;
  __REG32 PCS49PWRDWNSET      : 1;
  __REG32 PCS50PWRDWNSET      : 1;
  __REG32 PCS51PWRDWNSET      : 1;
  __REG32 PCS52PWRDWNSET      : 1;
  __REG32 PCS53PWRDWNSET      : 1;
  __REG32 PCS54PWRDWNSET      : 1;
  __REG32 PCS55PWRDWNSET      : 1;
  __REG32 PCS56PWRDWNSET      : 1;
  __REG32 PCS57PWRDWNSET      : 1;
  __REG32 PCS58PWRDWNSET      : 1;
  __REG32 PCS59PWRDWNSET      : 1;
  __REG32 PCS60PWRDWNSET      : 1;
  __REG32 PCS61PWRDWNSET      : 1;
  __REG32 PCS62PWRDWNSET      : 1;
  __REG32 PCS63PWRDWNSET      : 1;
} __pcspwrdwnset1_bits;

/* Peripheral Memory Power-Down Clear Register 0 (PCSPWRDWNCLR0) */
typedef struct {
  __REG32 PCS0PWRDWNCLR       : 1;
  __REG32 PCS1PWRDWNCLR       : 1;
  __REG32 PCS2PWRDWNCLR       : 1;
  __REG32 PCS3PWRDWNCLR       : 1;
  __REG32 PCS4PWRDWNCLR       : 1;
  __REG32 PCS5PWRDWNCLR       : 1;
  __REG32 PCS6PWRDWNCLR       : 1;
  __REG32 PCS7PWRDWNCLR       : 1;
  __REG32 PCS8PWRDWNCLR       : 1;
  __REG32 PCS9PWRDWNCLR       : 1;
  __REG32 PCS10PWRDWNCLR      : 1;
  __REG32 PCS11PWRDWNCLR      : 1;
  __REG32 PCS12PWRDWNCLR      : 1;
  __REG32 PCS13PWRDWNCLR      : 1;
  __REG32 PCS14PWRDWNCLR      : 1;
  __REG32 PCS15PWRDWNCLR      : 1;
  __REG32 PCS16PWRDWNCLR      : 1;
  __REG32 PCS17PWRDWNCLR      : 1;
  __REG32 PCS18PWRDWNCLR      : 1;
  __REG32 PCS19PWRDWNCLR      : 1;
  __REG32 PCS20PWRDWNCLR      : 1;
  __REG32 PCS21PWRDWNCLR      : 1;
  __REG32 PCS22PWRDWNCLR      : 1;
  __REG32 PCS23PWRDWNCLR      : 1;
  __REG32 PCS24PWRDWNCLR      : 1;
  __REG32 PCS25PWRDWNCLR      : 1;
  __REG32 PCS26PWRDWNCLR      : 1;
  __REG32 PCS27PWRDWNCLR      : 1;
  __REG32 PCS28PWRDWNCLR      : 1;
  __REG32 PCS29PWRDWNCLR      : 1;
  __REG32 PCS30PWRDWNCLR      : 1;
  __REG32 PCS31PWRDWNCLR      : 1;
} __pcspwrdwnclr0_bits;

/* Peripheral Memory Power-Down Clear Register 1 (PCSPWRDWNCLR1) */
typedef struct {
  __REG32 PCS32PWRDWNCLR      : 1;
  __REG32 PCS33PWRDWNCLR      : 1;
  __REG32 PCS34PWRDWNCLR      : 1;
  __REG32 PCS35PWRDWNCLR      : 1;
  __REG32 PCS36PWRDWNCLR      : 1;
  __REG32 PCS37PWRDWNCLR      : 1;
  __REG32 PCS38PWRDWNCLR      : 1;
  __REG32 PCS39PWRDWNCLR      : 1;
  __REG32 PCS40PWRDWNCLR      : 1;
  __REG32 PCS41PWRDWNCLR      : 1;
  __REG32 PCS42PWRDWNCLR      : 1;
  __REG32 PCS43PWRDWNCLR      : 1;
  __REG32 PCS44PWRDWNCLR      : 1;
  __REG32 PCS45PWRDWNCLR      : 1;
  __REG32 PCS46PWRDWNCLR      : 1;
  __REG32 PCS47PWRDWNCLR      : 1;
  __REG32 PCS48PWRDWNCLR      : 1;
  __REG32 PCS49PWRDWNCLR      : 1;
  __REG32 PCS50PWRDWNCLR      : 1;
  __REG32 PCS51PWRDWNCLR      : 1;
  __REG32 PCS52PWRDWNCLR      : 1;
  __REG32 PCS53PWRDWNCLR      : 1;
  __REG32 PCS54PWRDWNCLR      : 1;
  __REG32 PCS55PWRDWNCLR      : 1;
  __REG32 PCS56PWRDWNCLR      : 1;
  __REG32 PCS57PWRDWNCLR      : 1;
  __REG32 PCS58PWRDWNCLR      : 1;
  __REG32 PCS59PWRDWNCLR      : 1;
  __REG32 PCS60PWRDWNCLR      : 1;
  __REG32 PCS61PWRDWNCLR      : 1;
  __REG32 PCS62PWRDWNCLR      : 1;
  __REG32 PCS63PWRDWNCLR      : 1;
} __pcspwrdwnclr1_bits;

/* Peripheral Power-Down Set Register 0 (PSPWRDWNSET0) */
typedef struct {
  __REG32 PS0QUAD0PWRDWNSET   : 1;
  __REG32 PS0QUAD1PWRDWNSET   : 1;
  __REG32 PS0QUAD2PWRDWNSET   : 1;
  __REG32 PS0QUAD3PWRDWNSET   : 1;
  __REG32 PS1QUAD0PWRDWNSET   : 1;
  __REG32 PS1QUAD1PWRDWNSET   : 1;
  __REG32 PS1QUAD2PWRDWNSET   : 1;
  __REG32 PS1QUAD3PWRDWNSET   : 1;
  __REG32 PS2QUAD0PWRDWNSET   : 1;
  __REG32 PS2QUAD1PWRDWNSET   : 1;
  __REG32 PS2QUAD2PWRDWNSET   : 1;
  __REG32 PS2QUAD3PWRDWNSET   : 1;
  __REG32 PS3QUAD0PWRDWNSET   : 1;
  __REG32 PS3QUAD1PWRDWNSET   : 1;
  __REG32 PS3QUAD2PWRDWNSET   : 1;
  __REG32 PS3QUAD3PWRDWNSET   : 1;
  __REG32 PS4QUAD0PWRDWNSET   : 1;
  __REG32 PS4QUAD1PWRDWNSET   : 1;
  __REG32 PS4QUAD2PWRDWNSET   : 1;
  __REG32 PS4QUAD3PWRDWNSET   : 1;
  __REG32 PS5QUAD0PWRDWNSET   : 1;
  __REG32 PS5QUAD1PWRDWNSET   : 1;
  __REG32 PS5QUAD2PWRDWNSET   : 1;
  __REG32 PS5QUAD3PWRDWNSET   : 1;
  __REG32 PS6QUAD0PWRDWNSET   : 1;
  __REG32 PS6QUAD1PWRDWNSET   : 1;
  __REG32 PS6QUAD2PWRDWNSET   : 1;
  __REG32 PS6QUAD3PWRDWNSET   : 1;
  __REG32 PS7QUAD0PWRDWNSET   : 1;
  __REG32 PS7QUAD1PWRDWNSET   : 1;
  __REG32 PS7QUAD2PWRDWNSET   : 1;
  __REG32 PS7QUAD3PWRDWNSET   : 1;
} __pspwrdwnset0_bits;

/* Peripheral Power-Down Set Register 1 (PSPWRDWNSET1) */
typedef struct {
  __REG32 PS8QUAD0PWRDWNSET   : 1;
  __REG32 PS8QUAD1PWRDWNSET   : 1;
  __REG32 PS8QUAD2PWRDWNSET   : 1;
  __REG32 PS8QUAD3PWRDWNSET   : 1;
  __REG32 PS9QUAD0PWRDWNSET   : 1;
  __REG32 PS9QUAD1PWRDWNSET   : 1;
  __REG32 PS9QUAD2PWRDWNSET   : 1;
  __REG32 PS9QUAD3PWRDWNSET   : 1;
  __REG32 PS10QUAD0PWRDWNSET  : 1;
  __REG32 PS10QUAD1PWRDWNSET  : 1;
  __REG32 PS10QUAD2PWRDWNSET  : 1;
  __REG32 PS10QUAD3PWRDWNSET  : 1;
  __REG32 PS11QUAD0PWRDWNSET  : 1;
  __REG32 PS11QUAD1PWRDWNSET  : 1;
  __REG32 PS11QUAD2PWRDWNSET  : 1;
  __REG32 PS11QUAD3PWRDWNSET  : 1;
  __REG32 PS12QUAD0PWRDWNSET  : 1;
  __REG32 PS12QUAD1PWRDWNSET  : 1;
  __REG32 PS12QUAD2PWRDWNSET  : 1;
  __REG32 PS12QUAD3PWRDWNSET  : 1;
  __REG32 PS13QUAD0PWRDWNSET  : 1;
  __REG32 PS13QUAD1PWRDWNSET  : 1;
  __REG32 PS13QUAD2PWRDWNSET  : 1;
  __REG32 PS13QUAD3PWRDWNSET  : 1;
  __REG32 PS14QUAD0PWRDWNSET  : 1;
  __REG32 PS14QUAD1PWRDWNSET  : 1;
  __REG32 PS14QUAD2PWRDWNSET  : 1;
  __REG32 PS14QUAD3PWRDWNSET  : 1;
  __REG32 PS15QUAD0PWRDWNSET  : 1;
  __REG32 PS15QUAD1PWRDWNSET  : 1;
  __REG32 PS15QUAD2PWRDWNSET  : 1;
  __REG32 PS15QUAD3PWRDWNSET  : 1;
} __pspwrdwnset1_bits;

/* Peripheral Power-Down Set Register 2 (PSPWRDWNSET2) */
typedef struct {
  __REG32 PS16QUAD0PWRDWNSET  : 1;
  __REG32 PS16QUAD1PWRDWNSET  : 1;
  __REG32 PS16QUAD2PWRDWNSET  : 1;
  __REG32 PS16QUAD3PWRDWNSET  : 1;
  __REG32 PS17QUAD0PWRDWNSET  : 1;
  __REG32 PS17QUAD1PWRDWNSET  : 1;
  __REG32 PS17QUAD2PWRDWNSET  : 1;
  __REG32 PS17QUAD3PWRDWNSET  : 1;
  __REG32 PS18QUAD0PWRDWNSET  : 1;
  __REG32 PS18QUAD1PWRDWNSET  : 1;
  __REG32 PS18QUAD2PWRDWNSET  : 1;
  __REG32 PS18QUAD3PWRDWNSET  : 1;
  __REG32 PS19QUAD0PWRDWNSET  : 1;
  __REG32 PS19QUAD1PWRDWNSET  : 1;
  __REG32 PS19QUAD2PWRDWNSET  : 1;
  __REG32 PS19QUAD3PWRDWNSET  : 1;
  __REG32 PS20QUAD0PWRDWNSET  : 1;
  __REG32 PS20QUAD1PWRDWNSET  : 1;
  __REG32 PS20QUAD2PWRDWNSET  : 1;
  __REG32 PS20QUAD3PWRDWNSET  : 1;
  __REG32 PS21QUAD0PWRDWNSET  : 1;
  __REG32 PS21QUAD1PWRDWNSET  : 1;
  __REG32 PS21QUAD2PWRDWNSET  : 1;
  __REG32 PS21QUAD3PWRDWNSET  : 1;
  __REG32 PS22QUAD0PWRDWNSET  : 1;
  __REG32 PS22QUAD1PWRDWNSET  : 1;
  __REG32 PS22QUAD2PWRDWNSET  : 1;
  __REG32 PS22QUAD3PWRDWNSET  : 1;
  __REG32 PS23QUAD0PWRDWNSET  : 1;
  __REG32 PS23QUAD1PWRDWNSET  : 1;
  __REG32 PS23QUAD2PWRDWNSET  : 1;
  __REG32 PS23QUAD3PWRDWNSET  : 1;
} __pspwrdwnset2_bits;

/* Peripheral Power-Down Set Register 3 (PSPWRDWNSET3) */
typedef struct {
  __REG32 PS24QUAD0PWRDWNSET  : 1;
  __REG32 PS24QUAD1PWRDWNSET  : 1;
  __REG32 PS24QUAD2PWRDWNSET  : 1;
  __REG32 PS24QUAD3PWRDWNSET  : 1;
  __REG32 PS25QUAD0PWRDWNSET  : 1;
  __REG32 PS25QUAD1PWRDWNSET  : 1;
  __REG32 PS25QUAD2PWRDWNSET  : 1;
  __REG32 PS25QUAD3PWRDWNSET  : 1;
  __REG32 PS26QUAD0PWRDWNSET  : 1;
  __REG32 PS26QUAD1PWRDWNSET  : 1;
  __REG32 PS26QUAD2PWRDWNSET  : 1;
  __REG32 PS26QUAD3PWRDWNSET  : 1;
  __REG32 PS27QUAD0PWRDWNSET  : 1;
  __REG32 PS27QUAD1PWRDWNSET  : 1;
  __REG32 PS27QUAD2PWRDWNSET  : 1;
  __REG32 PS27QUAD3PWRDWNSET  : 1;
  __REG32 PS28QUAD0PWRDWNSET  : 1;
  __REG32 PS28QUAD1PWRDWNSET  : 1;
  __REG32 PS28QUAD2PWRDWNSET  : 1;
  __REG32 PS28QUAD3PWRDWNSET  : 1;
  __REG32 PS29QUAD0PWRDWNSET  : 1;
  __REG32 PS29QUAD1PWRDWNSET  : 1;
  __REG32 PS29QUAD2PWRDWNSET  : 1;
  __REG32 PS29QUAD3PWRDWNSET  : 1;
  __REG32 PS30QUAD0PWRDWNSET  : 1;
  __REG32 PS30QUAD1PWRDWNSET  : 1;
  __REG32 PS30QUAD2PWRDWNSET  : 1;
  __REG32 PS30QUAD3PWRDWNSET  : 1;
  __REG32 PS31QUAD0PWRDWNSET  : 1;
  __REG32 PS31QUAD1PWRDWNSET  : 1;
  __REG32 PS31QUAD2PWRDWNSET  : 1;
  __REG32 PS31QUAD3PWRDWNSET  : 1;
} __pspwrdwnset3_bits;

/* Peripheral Power-Down Clear Register 0 (PSPWRDWNCLR0) */
typedef struct {
  __REG32 PS0QUAD0PWRDWNCLR   : 1;
  __REG32 PS0QUAD1PWRDWNCLR   : 1;
  __REG32 PS0QUAD2PWRDWNCLR   : 1;
  __REG32 PS0QUAD3PWRDWNCLR   : 1;
  __REG32 PS1QUAD0PWRDWNCLR   : 1;
  __REG32 PS1QUAD1PWRDWNCLR   : 1;
  __REG32 PS1QUAD2PWRDWNCLR   : 1;
  __REG32 PS1QUAD3PWRDWNCLR   : 1;
  __REG32 PS2QUAD0PWRDWNCLR   : 1;
  __REG32 PS2QUAD1PWRDWNCLR   : 1;
  __REG32 PS2QUAD2PWRDWNCLR   : 1;
  __REG32 PS2QUAD3PWRDWNCLR   : 1;
  __REG32 PS3QUAD0PWRDWNCLR   : 1;
  __REG32 PS3QUAD1PWRDWNCLR   : 1;
  __REG32 PS3QUAD2PWRDWNCLR   : 1;
  __REG32 PS3QUAD3PWRDWNCLR   : 1;
  __REG32 PS4QUAD0PWRDWNCLR   : 1;
  __REG32 PS4QUAD1PWRDWNCLR   : 1;
  __REG32 PS4QUAD2PWRDWNCLR   : 1;
  __REG32 PS4QUAD3PWRDWNCLR   : 1;
  __REG32 PS5QUAD0PWRDWNCLR   : 1;
  __REG32 PS5QUAD1PWRDWNCLR   : 1;
  __REG32 PS5QUAD2PWRDWNCLR   : 1;
  __REG32 PS5QUAD3PWRDWNCLR   : 1;
  __REG32 PS6QUAD0PWRDWNCLR   : 1;
  __REG32 PS6QUAD1PWRDWNCLR   : 1;
  __REG32 PS6QUAD2PWRDWNCLR   : 1;
  __REG32 PS6QUAD3PWRDWNCLR   : 1;
  __REG32 PS7QUAD0PWRDWNCLR   : 1;
  __REG32 PS7QUAD1PWRDWNCLR   : 1;
  __REG32 PS7QUAD2PWRDWNCLR   : 1;
  __REG32 PS7QUAD3PWRDWNCLR   : 1;
} __pspwrdwnclr0_bits;

/* Peripheral Power-Down Clear Register 1 (PSPWRDWNCLR1) */
typedef struct {
  __REG32 PS8QUAD0PWRDWNCLR   : 1;
  __REG32 PS8QUAD1PWRDWNCLR   : 1;
  __REG32 PS8QUAD2PWRDWNCLR   : 1;
  __REG32 PS8QUAD3PWRDWNCLR   : 1;
  __REG32 PS9QUAD0PWRDWNCLR   : 1;
  __REG32 PS9QUAD1PWRDWNCLR   : 1;
  __REG32 PS9QUAD2PWRDWNCLR   : 1;
  __REG32 PS9QUAD3PWRDWNCLR   : 1;
  __REG32 PS10QUAD0PWRDWNCLR  : 1;
  __REG32 PS10QUAD1PWRDWNCLR  : 1;
  __REG32 PS10QUAD2PWRDWNCLR  : 1;
  __REG32 PS10QUAD3PWRDWNCLR  : 1;
  __REG32 PS11QUAD0PWRDWNCLR  : 1;
  __REG32 PS11QUAD1PWRDWNCLR  : 1;
  __REG32 PS11QUAD2PWRDWNCLR  : 1;
  __REG32 PS11QUAD3PWRDWNCLR  : 1;
  __REG32 PS12QUAD0PWRDWNCLR  : 1;
  __REG32 PS12QUAD1PWRDWNCLR  : 1;
  __REG32 PS12QUAD2PWRDWNCLR  : 1;
  __REG32 PS12QUAD3PWRDWNCLR  : 1;
  __REG32 PS13QUAD0PWRDWNCLR  : 1;
  __REG32 PS13QUAD1PWRDWNCLR  : 1;
  __REG32 PS13QUAD2PWRDWNCLR  : 1;
  __REG32 PS13QUAD3PWRDWNCLR  : 1;
  __REG32 PS14QUAD0PWRDWNCLR  : 1;
  __REG32 PS14QUAD1PWRDWNCLR  : 1;
  __REG32 PS14QUAD2PWRDWNCLR  : 1;
  __REG32 PS14QUAD3PWRDWNCLR  : 1;
  __REG32 PS15QUAD0PWRDWNCLR  : 1;
  __REG32 PS15QUAD1PWRDWNCLR  : 1;
  __REG32 PS15QUAD2PWRDWNCLR  : 1;
  __REG32 PS15QUAD3PWRDWNCLR  : 1;
} __pspwrdwnclr1_bits;

/* Peripheral Power-Down Clear Register 2 (PSPWRDWNCLR2) */
typedef struct {
  __REG32 PS16QUAD0PWRDWNCLR  : 1;
  __REG32 PS16QUAD1PWRDWNCLR  : 1;
  __REG32 PS16QUAD2PWRDWNCLR  : 1;
  __REG32 PS16QUAD3PWRDWNCLR  : 1;
  __REG32 PS17QUAD0PWRDWNCLR  : 1;
  __REG32 PS17QUAD1PWRDWNCLR  : 1;
  __REG32 PS17QUAD2PWRDWNCLR  : 1;
  __REG32 PS17QUAD3PWRDWNCLR  : 1;
  __REG32 PS18QUAD0PWRDWNCLR  : 1;
  __REG32 PS18QUAD1PWRDWNCLR  : 1;
  __REG32 PS18QUAD2PWRDWNCLR  : 1;
  __REG32 PS18QUAD3PWRDWNCLR  : 1;
  __REG32 PS19QUAD0PWRDWNCLR  : 1;
  __REG32 PS19QUAD1PWRDWNCLR  : 1;
  __REG32 PS19QUAD2PWRDWNCLR  : 1;
  __REG32 PS19QUAD3PWRDWNCLR  : 1;
  __REG32 PS20QUAD0PWRDWNCLR  : 1;
  __REG32 PS20QUAD1PWRDWNCLR  : 1;
  __REG32 PS20QUAD2PWRDWNCLR  : 1;
  __REG32 PS20QUAD3PWRDWNCLR  : 1;
  __REG32 PS21QUAD0PWRDWNCLR  : 1;
  __REG32 PS21QUAD1PWRDWNCLR  : 1;
  __REG32 PS21QUAD2PWRDWNCLR  : 1;
  __REG32 PS21QUAD3PWRDWNCLR  : 1;
  __REG32 PS22QUAD0PWRDWNCLR  : 1;
  __REG32 PS22QUAD1PWRDWNCLR  : 1;
  __REG32 PS22QUAD2PWRDWNCLR  : 1;
  __REG32 PS22QUAD3PWRDWNCLR  : 1;
  __REG32 PS23QUAD0PWRDWNCLR  : 1;
  __REG32 PS23QUAD1PWRDWNCLR  : 1;
  __REG32 PS23QUAD2PWRDWNCLR  : 1;
  __REG32 PS23QUAD3PWRDWNCLR  : 1;
} __pspwrdwnclr2_bits;

/* Peripheral Power-Down Clear Register 3 (PSPWRDWNCLR) */
typedef struct {
  __REG32 PS24QUAD0PWRDWNCLR  : 1;
  __REG32 PS24QUAD1PWRDWNCLR  : 1;
  __REG32 PS24QUAD2PWRDWNCLR  : 1;
  __REG32 PS24QUAD3PWRDWNCLR  : 1;
  __REG32 PS25QUAD0PWRDWNCLR  : 1;
  __REG32 PS25QUAD1PWRDWNCLR  : 1;
  __REG32 PS25QUAD2PWRDWNCLR  : 1;
  __REG32 PS25QUAD3PWRDWNCLR  : 1;
  __REG32 PS26QUAD0PWRDWNCLR  : 1;
  __REG32 PS26QUAD1PWRDWNCLR  : 1;
  __REG32 PS26QUAD2PWRDWNCLR  : 1;
  __REG32 PS26QUAD3PWRDWNCLR  : 1;
  __REG32 PS27QUAD0PWRDWNCLR  : 1;
  __REG32 PS27QUAD1PWRDWNCLR  : 1;
  __REG32 PS27QUAD2PWRDWNCLR  : 1;
  __REG32 PS27QUAD3PWRDWNCLR  : 1;
  __REG32 PS28QUAD0PWRDWNCLR  : 1;
  __REG32 PS28QUAD1PWRDWNCLR  : 1;
  __REG32 PS28QUAD2PWRDWNCLR  : 1;
  __REG32 PS28QUAD3PWRDWNCLR  : 1;
  __REG32 PS29QUAD0PWRDWNCLR  : 1;
  __REG32 PS29QUAD1PWRDWNCLR  : 1;
  __REG32 PS29QUAD2PWRDWNCLR  : 1;
  __REG32 PS29QUAD3PWRDWNCLR  : 1;
  __REG32 PS30QUAD0PWRDWNCLR  : 1;
  __REG32 PS30QUAD1PWRDWNCLR  : 1;
  __REG32 PS30QUAD2PWRDWNCLR  : 1;
  __REG32 PS30QUAD3PWRDWNCLR  : 1;
  __REG32 PS31QUAD0PWRDWNCLR  : 1;
  __REG32 PS31QUAD1PWRDWNCLR  : 1;
  __REG32 PS31QUAD2PWRDWNCLR  : 1;
  __REG32 PS31QUAD3PWRDWNCLR  : 1;
} __pspwrdwnclr3_bits;

/* Wakeup Reset Control Register (RSTCR) */
typedef struct {
  __REG32 nRST	                : 1;
  __REG32                       :31;
} __rstcr_bits;

/* External Wakeup Enable Register (EXTWAKENR) */
typedef struct {
  __REG32 WAKE_ENA0             : 1;
  __REG32 WAKE_ENA1             : 1;
  __REG32 WAKE_ENA2             : 1;
  __REG32 WAKE_ENA3             : 1;
  __REG32 WAKE_ENA4             : 1;
  __REG32 WAKE_ENA5             : 1;
  __REG32 WAKE_ENA6             : 1;
  __REG32 WAKE_ENA7             : 1;
  __REG32 WAKE_ENA8             : 1;
  __REG32 WAKE_ENA9             : 1;
  __REG32 WAKE_ENA10            : 1;
  __REG32 WAKE_ENA11            : 1;
  __REG32 WAKE_ENA12            : 1;
  __REG32 WAKE_ENA13            : 1;
  __REG32 WAKE_ENA14            : 1;
  __REG32 WAKE_ENA15            : 1;
  __REG32                       :16;
} __extwakenr_bits;

/* External Wakeup Level Register (EXTWAKLVR) */
typedef struct {
  __REG32 WAKE_LVL0             : 1;
  __REG32 WAKE_LVL1             : 1;
  __REG32 WAKE_LVL2             : 1;
  __REG32 WAKE_LVL3             : 1;
  __REG32 WAKE_LVL4             : 1;
  __REG32 WAKE_LVL5             : 1;
  __REG32 WAKE_LVL6             : 1;
  __REG32 WAKE_LVL7             : 1;
  __REG32 WAKE_LVL8             : 1;
  __REG32 WAKE_LVL9             : 1;
  __REG32 WAKE_LVL10            : 1;
  __REG32 WAKE_LVL11            : 1;
  __REG32 WAKE_LVL12            : 1;
  __REG32 WAKE_LVL13            : 1;
  __REG32 WAKE_LVL14            : 1;
  __REG32 WAKE_LVL15            : 1;
  __REG32                       :16;
} __extwaklvr_bits;

/* External Wakeup Status Register (EXTWAKESR) */
typedef struct {
  __REG32 WAKE_FLAG0            : 1;
  __REG32 WAKE_FLAG1            : 1;
  __REG32 WAKE_FLAG2            : 1;
  __REG32 WAKE_FLAG3            : 1;
  __REG32 WAKE_FLAG4            : 1;
  __REG32 WAKE_FLAG5            : 1;
  __REG32 WAKE_FLAG6            : 1;
  __REG32 WAKE_FLAG7            : 1;
  __REG32 WAKE_FLAG8            : 1;
  __REG32 WAKE_FLAG9            : 1;
  __REG32 WAKE_FLAG10           : 1;
  __REG32 WAKE_FLAG11           : 1;
  __REG32 WAKE_FLAG12           : 1;
  __REG32 WAKE_FLAG13           : 1;
  __REG32 WAKE_FLAG14           : 1;
  __REG32 WAKE_FLAG15           : 1;
  __REG32                       :16;
} __extwakesr_bits;

/* Hibernate Exit and Status Register (HIBXSTATR) */
typedef struct {
  __REG32 D0                    : 1;
  __REG32 	                    :31;
} __hibxstatr_bits;

/* CPU Logic BIST Clock Divider (STCLKDIV) */
typedef struct {
  __REG32 	                    :24;
  __REG32 CLKDIV                : 3;
  __REG32 	                    : 5;
} __stclkdiv_bits;

/* RAM Control Register (RAMCTRL) */
typedef struct {
  __REG16 ECC_ENABLE            : 4;
  __REG16 	                    : 4;
  __REG16 ECC_WRT_ENA           : 1;
  __REG16 RMWCBYP			          : 4;
  __REG16 	                    : 3;
} __ramctrl_bits;

/* Interrupt Control Register (RAMINTCTRL) */
typedef struct {
  __REG16 SECINTEN	            : 1;
  __REG16 	                    :15;
} __ramintctrl_bits;

/* Memory Fault Detect Status Register (RAMERRSTATUS) */
typedef struct {
  __REG16 SECINTFLAG            : 1;
  __REG16 	                    :15;
} __ramerrstatus_bits;

/* Single Error Address Register (RAMSERRADD) */
typedef struct {
  __REG16 SERRADDR	            :15;
  __REG16 	                    : 1;
} __rammserraddr_bits;

/* RAM Error Position Register (RAMERRPOSITION) */
typedef struct {
  __REG16 SERRPOSITION          : 6;
  __REG16 						          : 2;
  __REG16 ERRTYPE			          : 1;
  __REG16 	                    : 7;
} __ramserrposition_bits;

/* Double Error Address Register (RAMDERRADD) */
typedef struct {
  __REG16 DERRADDR		          :15;
  __REG16 						          : 1;
} __ramderraddr_bits;

/* RAM Control Register (RAMCTRL2) */
typedef struct {
  __REG16 EDACCMODE		          : 4;
  __REG16 						          : 4;
  __REG16 EMULATION_TRACE_DIS		: 1;
  __REG16 						          : 7;
} __ramctrl2_bits;

/* Option Control Register (FRDCNTL) */
typedef struct {
  __REG32 ENPIPE              : 1;
  __REG32                     : 3;
  __REG32 ASWSTEN             : 1;
  __REG32                     : 3;
  __REG32 RWAIT               : 4;
  __REG32                     :20;
} __ftudcntl_bits;

/* Special Read Control Register (FSPRD) */
typedef struct {
  __REG32 RM0                 : 1;
  __REG32 RM1                 : 1;
  __REG32                     :30;
} __fsprd_bits;

/* Error Detection Control Register 1 (FEDACCTRL1) */
typedef struct {
  __REG32 EDACEN              : 4;
  __REG32 EZCV                : 1;
  __REG32 EOCV                : 1;
  __REG32                     : 2;
  __REG32 EPEN                : 1;
  __REG32 EZFEN               : 1;
  __REG32 EOFEN               : 1;
  __REG32                     : 5;
  __REG32 EDACMODE            : 4;
  __REG32                     : 4;
  __REG32 SUSP_IGNR           : 1;
  __REG32                     : 7;
} __fedacctrl1_bits;

/* Error Correction Control Register 2 (FEDACCTRL2) */
typedef struct {
  __REG32 SEC_THRESHOLD       :16;
  __REG32                     :16;
} __fedacctrl2_bits;

/* Error Correction Counter Register (FCOR_ERR_CNT) */
typedef struct {
  __REG32 COR_ERR_CNT         :16;
  __REG32                     :16;
} __fcor_err_cnt_bits;

/* Correctable Error Position Register (FCOR_ERR_POS) */
typedef struct {
  __REG32 SERR_POS		        : 8;
  __REG32 ECC_ERR			        : 1;
  __REG32                     :23;
} __fcor_err_pos_bits;

/* Error Status Register (FEDACSTATUS) */
typedef struct {
  __REG32 ERR_PRF_FLG         : 1;
  __REG32 ERR_ZERO_FLG        : 1;
  __REG32 ERR_ONE_FLG         : 1;
  __REG32                     : 5;
  __REG32 ECC_MUL_ERR         : 1;
  __REG32                     :23;
} __fedacstatus_bits;

/* Error Detection Sector Disable (FEDACSDIS) */
typedef struct {
  __REG32 SectorID0           : 4;
  __REG32                     : 1;
  __REG32 BankID0             : 3;
  __REG32 SectorID0_inverse   : 4;
  __REG32                     : 1;
  __REG32 BankID0_inverse     : 3;
  __REG32 SectorID1           : 4;
  __REG32                     : 1;
  __REG32 BankID1             : 3;
  __REG32 SectorID1_inverse   : 4;
  __REG32                     : 1;
  __REG32 BankID1_inverse     : 3;
} __fedacsdis_bits;

/* Bank Protection Register (FBPROT) */
typedef struct {
  __REG32 PROTL1DIS           : 1;
  __REG32                     :31;
} __fbprot_bits;

/* Bank Sector Enable Register (FBSE) */
typedef struct {
  __REG32 BSE0                : 1;
  __REG32 BSE1                : 1;
  __REG32 BSE2                : 1;
  __REG32 BSE3                : 1;
  __REG32 BSE4                : 1;
  __REG32 BSE5                : 1;
  __REG32 BSE6                : 1;
  __REG32 BSE7                : 1;
  __REG32 BSE8                : 1;
  __REG32 BSE9                : 1;
  __REG32 BSE10               : 1;
  __REG32 BSE11               : 1;
  __REG32 BSE12               : 1;
  __REG32 BSE13               : 1;
  __REG32 BSE14               : 1;
  __REG32 BSE15               : 1;
  __REG32                     :16;
} __fbse_bits;

/* Bank Access Control Register (FBAC) */
typedef struct {
  __REG32 VREADST             : 8;
  __REG32 BAGP                : 8;
  __REG32 OTPPROTDIS0         : 1;
  __REG32 OTPPROTDIS1         : 1;
  __REG32 OTPPROTDIS2         : 1;
  __REG32 OTPPROTDIS3         : 1;
  __REG32 OTPPROTDIS4         : 1;
  __REG32 OTPPROTDIS5         : 1;
  __REG32 OTPPROTDIS6         : 1;
  __REG32 OTPPROTDIS7         : 1;
  __REG32                     : 8;
} __fbac_bits;

/* Bank Fallback Power Register (FBFALLBACK) */
typedef struct {
  __REG32 BANKPWR0            : 2;
  __REG32 BANKPWR1            : 2;
  __REG32                     :28;
} __fbfallback_bits;

/* Bank/Pump Ready Register (FBPRDY) */
typedef struct {
  __REG32 BANKRDY0            : 1;
  __REG32 BANKRDY1            : 1;
  __REG32                     :13;
  __REG32 PUMPRDY             : 1;
  __REG32                     :16;
} __fbprdy_bits;

/* Pump Access Control Register 1 (FPAC1) */
typedef struct {
  __REG32 PUMPPWR             : 1;
  __REG32                     :15;
  __REG32 PSLEEP              :11;
  __REG32                     : 5;
} __fpac1_bits;

/* Pump Access Control Register 2 (FPAC2) */
typedef struct {
  __REG32 PAGP                :16;
  __REG32                     :16;
} __fpac2_bits;

/* Module Access Control Register (FMAC) */
typedef struct {
  __REG32 BANK                : 3;
  __REG32                     :29;
} __fmac_bits;

/* Flash Emulation ECC Register (FEMU_ECC) */
typedef struct {
  __REG32 EMU_ECC             : 8;
  __REG32                     :24;
} __femu_ecc_bits;

/* Flash Error Detection Sector Disable (FEDACSDIS2) */
typedef struct {
  __REG32 SectorID2           : 4;
  __REG32                     : 1;
  __REG32 BankID2             : 3;
  __REG32 SectorID2_inverse   : 4;
  __REG32                     : 1;
  __REG32 BankID2_inverse     : 3;
  __REG32 SectorID3           : 4;
  __REG32                     : 1;
  __REG32 BankID3             : 3;
  __REG32 SectorID3_inverse   : 4;
  __REG32                     : 1;
  __REG32 BankID3_inverse     : 3;
} __fedacsdis2_bits;

/* CRC_CTRL0: CRC Global Control Register 0 */
typedef struct {
  __REG32 CH1_PSA_SWREST      : 1;
  __REG32                     :31;
} __crc_ctrl0_bits;

/* CRC_CTRL1: CRC Global Control Register 1 */
typedef struct {
  __REG32 PWDN                : 1;
  __REG32                     :31;
} __crc_ctrl1_bits;

/* CRC_CTRL2: CRC Global Control Register 2 */
typedef struct {
  __REG32 CH1_MODE            : 2;
  __REG32                     : 2;
  __REG32 CH1_TRACEEN         : 1;
  __REG32                     :27;
} __crc_ctrl2_bits;

/* CAN Control Register (DCAN CTL) */
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
  __REG32                     : 3;
  __REG32 PDR                 : 1;
  __REG32 WUBA                : 1;
  __REG32                     : 6;
} __dcanctl_bits;

/* CAN Error and Status Register (DCAN ES) */
typedef struct {
  __REG32 LEC                 : 3;
  __REG32 TxOK                : 1;
  __REG32 RxOK                : 1;
  __REG32 EPass               : 1;
  __REG32 EWarn               : 1;
  __REG32 BOff                : 1;
  __REG32 PER                 : 1;
  __REG32 WakeUpPnd           : 1;
  __REG32 PDA                 : 1;
  __REG32                     :21;
} __dcanes_bits;

/* CAN Error Counter Register (DCAN ERRC) */
typedef struct {
  __REG32 TEC                 : 8;
  __REG32 REC                 : 7;
  __REG32 RP                  : 1;
  __REG32                     :16;
} __dcanerrc_bits;

/* CAN Bit Timing Register (DCAN BTR) */
typedef struct {
  __REG32 BRP                 : 6;
  __REG32 SJW                 : 2;
  __REG32 TSeg1               : 4;
  __REG32 TSeg2               : 3;
  __REG32                     : 1;
  __REG32 BRPE                : 4;
  __REG32                     :12;
} __dcanbtr_bits;

/* CAN Interrupt Register (DCANINT) */
typedef struct {
  __REG32 Int0ID              :16;
  __REG32 Int1ID              : 8;
  __REG32                     : 8;
} __dcanint_bits;

/* CAN Test Register (DCAN TEST) */
typedef struct {
  __REG32                     : 3;
  __REG32 Silent              : 1;
  __REG32 LBack               : 1;
  __REG32 Tx                  : 2;
  __REG32 Rx                  : 1;
  __REG32 EXL                 : 1;
  __REG32 RDA                 : 1;
  __REG32                     :22;
} __dcantest_bits;

/* CAN Parity Error Code Register (DCAN PERR) */
typedef struct {
  __REG32 MessageNumber       : 8;
  __REG32 WordNumber          : 3;
  __REG32                     :21;
} __dcanperr_bits;

/* DCAN Core Release Register (DCAN REL) */
typedef struct {
  __REG32 DAY					        : 8;
  __REG32 MON					        : 8;
  __REG32 YEAR			          : 4;
  __REG32 SUBSTEP		          : 4;
  __REG32 STEP			          : 4;
  __REG32 REL				          : 4;
} __dcanrel_bits;

/* CAN Transmission Request X Register (DCAN TXRQ X) */
typedef struct {
  __REG32 TxRqstReg1          : 2;
  __REG32 TxRqstReg2          : 2;
  __REG32 TxRqstReg3          : 2;
  __REG32 TxRqstReg4          : 2;
  __REG32 TxRqstReg5          : 2;
  __REG32 TxRqstReg6          : 2;
  __REG32 TxRqstReg7          : 2;
  __REG32 TxRqstReg8          : 2;
  __REG32                     :16;
} __dcantxrqx_bits;

/* CAN Transmission Request Registers (DCAN TXRQ12) */
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
} __dcantxrq12_bits;

/* CAN Transmission Request Registers (DCAN TXRQ34) */
typedef struct {
  __REG32 TxRqst33          : 1;
  __REG32 TxRqst34          : 1;
  __REG32 TxRqst35          : 1;
  __REG32 TxRqst36          : 1;
  __REG32 TxRqst37          : 1;
  __REG32 TxRqst38          : 1;
  __REG32 TxRqst39          : 1;
  __REG32 TxRqst40          : 1;
  __REG32 TxRqst41          : 1;
  __REG32 TxRqst42          : 1;
  __REG32 TxRqst43          : 1;
  __REG32 TxRqst44          : 1;
  __REG32 TxRqst45          : 1;
  __REG32 TxRqst46          : 1;
  __REG32 TxRqst47          : 1;
  __REG32 TxRqst48          : 1;
  __REG32 TxRqst49          : 1;
  __REG32 TxRqst50          : 1;
  __REG32 TxRqst51          : 1;
  __REG32 TxRqst52          : 1;
  __REG32 TxRqst53          : 1;
  __REG32 TxRqst54          : 1;
  __REG32 TxRqst55          : 1;
  __REG32 TxRqst56          : 1;
  __REG32 TxRqst57          : 1;
  __REG32 TxRqst58          : 1;
  __REG32 TxRqst59          : 1;
  __REG32 TxRqst60          : 1;
  __REG32 TxRqst61          : 1;
  __REG32 TxRqst62          : 1;
  __REG32 TxRqst63          : 1;
  __REG32 TxRqst64          : 1;
} __dcantxrq34_bits;

/* CAN Transmission Request Registers (DCAN TXRQ56) */
typedef struct {
  __REG32 TxRqst65          : 1;
  __REG32 TxRqst66          : 1;
  __REG32 TxRqst67          : 1;
  __REG32 TxRqst68          : 1;
  __REG32 TxRqst69          : 1;
  __REG32 TxRqst70          : 1;
  __REG32 TxRqst71          : 1;
  __REG32 TxRqst72          : 1;
  __REG32 TxRqst73          : 1;
  __REG32 TxRqst74          : 1;
  __REG32 TxRqst75          : 1;
  __REG32 TxRqst76          : 1;
  __REG32 TxRqst77          : 1;
  __REG32 TxRqst78          : 1;
  __REG32 TxRqst79          : 1;
  __REG32 TxRqst80          : 1;
  __REG32 TxRqst81          : 1;
  __REG32 TxRqst82          : 1;
  __REG32 TxRqst83          : 1;
  __REG32 TxRqst84          : 1;
  __REG32 TxRqst85          : 1;
  __REG32 TxRqst86          : 1;
  __REG32 TxRqst87          : 1;
  __REG32 TxRqst88          : 1;
  __REG32 TxRqst89          : 1;
  __REG32 TxRqst90          : 1;
  __REG32 TxRqst91          : 1;
  __REG32 TxRqst92          : 1;
  __REG32 TxRqst93          : 1;
  __REG32 TxRqst94          : 1;
  __REG32 TxRqst95          : 1;
  __REG32 TxRqst96          : 1;
} __dcantxrq56_bits;

/* CAN Transmission Request Registers (DCAN TXRQ78) */
typedef struct {
  __REG32 TxRqst97          : 1;
  __REG32 TxRqst98          : 1;
  __REG32 TxRqst99          : 1;
  __REG32 TxRqst100         : 1;
  __REG32 TxRqst101         : 1;
  __REG32 TxRqst102         : 1;
  __REG32 TxRqst103         : 1;
  __REG32 TxRqst104         : 1;
  __REG32 TxRqst105         : 1;
  __REG32 TxRqst106         : 1;
  __REG32 TxRqst107         : 1;
  __REG32 TxRqst108         : 1;
  __REG32 TxRqst109         : 1;
  __REG32 TxRqst110         : 1;
  __REG32 TxRqst111         : 1;
  __REG32 TxRqst112         : 1;
  __REG32 TxRqst113         : 1;
  __REG32 TxRqst114         : 1;
  __REG32 TxRqst115         : 1;
  __REG32 TxRqst116         : 1;
  __REG32 TxRqst117         : 1;
  __REG32 TxRqst118         : 1;
  __REG32 TxRqst119         : 1;
  __REG32 TxRqst120         : 1;
  __REG32 TxRqst121         : 1;
  __REG32 TxRqst122         : 1;
  __REG32 TxRqst123         : 1;
  __REG32 TxRqst124         : 1;
  __REG32 TxRqst125         : 1;
  __REG32 TxRqst126         : 1;
  __REG32 TxRqst127         : 1;
  __REG32 TxRqst128         : 1;
} __dcantxrq78_bits;

/* CAN New Data X Register (DCAN NWDAT X) */
typedef struct {
  __REG32 NewDatReg1          : 2;
  __REG32 NewDatReg2          : 2;
  __REG32 NewDatReg3          : 2;
  __REG32 NewDatReg4          : 2;
  __REG32 NewDatReg5          : 2;
  __REG32 NewDatReg6          : 2;
  __REG32 NewDatReg7          : 2;
  __REG32 NewDatReg8          : 2;
  __REG32                     :16;
} __dcannwdatx_bits;

/* CAN New Data Registers (DCAN NWDAT12) */
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
} __dcannwdat12_bits;

/* CAN New Data Registers (DCAN NWDAT34) */
typedef struct {
  __REG32 NewDat33          : 1;
  __REG32 NewDat34          : 1;
  __REG32 NewDat35          : 1;
  __REG32 NewDat36          : 1;
  __REG32 NewDat37          : 1;
  __REG32 NewDat38          : 1;
  __REG32 NewDat39          : 1;
  __REG32 NewDat40          : 1;
  __REG32 NewDat41          : 1;
  __REG32 NewDat42          : 1;
  __REG32 NewDat43          : 1;
  __REG32 NewDat44          : 1;
  __REG32 NewDat45          : 1;
  __REG32 NewDat46          : 1;
  __REG32 NewDat47          : 1;
  __REG32 NewDat48          : 1;
  __REG32 NewDat49          : 1;
  __REG32 NewDat50          : 1;
  __REG32 NewDat51          : 1;
  __REG32 NewDat52          : 1;
  __REG32 NewDat53          : 1;
  __REG32 NewDat54          : 1;
  __REG32 NewDat55          : 1;
  __REG32 NewDat56          : 1;
  __REG32 NewDat57          : 1;
  __REG32 NewDat58          : 1;
  __REG32 NewDat59          : 1;
  __REG32 NewDat60          : 1;
  __REG32 NewDat61          : 1;
  __REG32 NewDat62          : 1;
  __REG32 NewDat63          : 1;
  __REG32 NewDat64          : 1;
} __dcannwdat34_bits;

/* CAN New Data Registers (DCAN NWDAT56) */
typedef struct {
  __REG32 NewDat65          : 1;
  __REG32 NewDat66          : 1;
  __REG32 NewDat67          : 1;
  __REG32 NewDat68          : 1;
  __REG32 NewDat69          : 1;
  __REG32 NewDat70          : 1;
  __REG32 NewDat71          : 1;
  __REG32 NewDat72          : 1;
  __REG32 NewDat73          : 1;
  __REG32 NewDat74          : 1;
  __REG32 NewDat75          : 1;
  __REG32 NewDat76          : 1;
  __REG32 NewDat77          : 1;
  __REG32 NewDat78          : 1;
  __REG32 NewDat79          : 1;
  __REG32 NewDat80          : 1;
  __REG32 NewDat81          : 1;
  __REG32 NewDat82          : 1;
  __REG32 NewDat83          : 1;
  __REG32 NewDat84          : 1;
  __REG32 NewDat85          : 1;
  __REG32 NewDat86          : 1;
  __REG32 NewDat87          : 1;
  __REG32 NewDat88          : 1;
  __REG32 NewDat89          : 1;
  __REG32 NewDat90          : 1;
  __REG32 NewDat91          : 1;
  __REG32 NewDat92          : 1;
  __REG32 NewDat93          : 1;
  __REG32 NewDat94          : 1;
  __REG32 NewDat95          : 1;
  __REG32 NewDat96          : 1;
} __dcannwdat56_bits;

/* CAN New Data Registers (DCAN NWDAT78) */
typedef struct {
  __REG32 NewDat97          : 1;
  __REG32 NewDat98          : 1;
  __REG32 NewDat99          : 1;
  __REG32 NewDat100         : 1;
  __REG32 NewDat101         : 1;
  __REG32 NewDat102         : 1;
  __REG32 NewDat103         : 1;
  __REG32 NewDat104         : 1;
  __REG32 NewDat105         : 1;
  __REG32 NewDat106         : 1;
  __REG32 NewDat107         : 1;
  __REG32 NewDat108         : 1;
  __REG32 NewDat109         : 1;
  __REG32 NewDat110         : 1;
  __REG32 NewDat111         : 1;
  __REG32 NewDat112         : 1;
  __REG32 NewDat113         : 1;
  __REG32 NewDat114         : 1;
  __REG32 NewDat115         : 1;
  __REG32 NewDat116         : 1;
  __REG32 NewDat117         : 1;
  __REG32 NewDat118         : 1;
  __REG32 NewDat119         : 1;
  __REG32 NewDat120         : 1;
  __REG32 NewDat121         : 1;
  __REG32 NewDat122         : 1;
  __REG32 NewDat123         : 1;
  __REG32 NewDat124         : 1;
  __REG32 NewDat125         : 1;
  __REG32 NewDat126         : 1;
  __REG32 NewDat127         : 1;
  __REG32 NewDat128         : 1;
} __dcannwdat78_bits;

/* CAN Interrupt Pending X Register (DCAN INTPND X) */
typedef struct {
  __REG32 IntPndReg1          : 2;
  __REG32 IntPndReg2          : 2;
  __REG32 IntPndReg3          : 2;
  __REG32 IntPndReg4          : 2;
  __REG32 IntPndReg5          : 2;
  __REG32 IntPndReg6          : 2;
  __REG32 IntPndReg7          : 2;
  __REG32 IntPndReg8          : 2;
  __REG32                     :16;
} __dcanintpnd_bits;

/* CAN Interrupt Pending Registers (DCAN INTPND12) */
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
} __dcanintpnd12_bits;

/* CAN Interrupt Pending Registers (DCAN INTPND34) */
typedef struct {
  __REG32 IntPnd33          : 1;
  __REG32 IntPnd34          : 1;
  __REG32 IntPnd35          : 1;
  __REG32 IntPnd36          : 1;
  __REG32 IntPnd37          : 1;
  __REG32 IntPnd38          : 1;
  __REG32 IntPnd39          : 1;
  __REG32 IntPnd40          : 1;
  __REG32 IntPnd41          : 1;
  __REG32 IntPnd42          : 1;
  __REG32 IntPnd43          : 1;
  __REG32 IntPnd44          : 1;
  __REG32 IntPnd45          : 1;
  __REG32 IntPnd46          : 1;
  __REG32 IntPnd47          : 1;
  __REG32 IntPnd48          : 1;
  __REG32 IntPnd49          : 1;
  __REG32 IntPnd50          : 1;
  __REG32 IntPnd51          : 1;
  __REG32 IntPnd52          : 1;
  __REG32 IntPnd53          : 1;
  __REG32 IntPnd54          : 1;
  __REG32 IntPnd55          : 1;
  __REG32 IntPnd56          : 1;
  __REG32 IntPnd57          : 1;
  __REG32 IntPnd58          : 1;
  __REG32 IntPnd59          : 1;
  __REG32 IntPnd60          : 1;
  __REG32 IntPnd61          : 1;
  __REG32 IntPnd62          : 1;
  __REG32 IntPnd63          : 1;
  __REG32 IntPnd64          : 1;
} __dcanintpnd34_bits;

/* CAN Interrupt Pending Registers (DCAN INTPND56) */
typedef struct {
  __REG32 IntPnd65          : 1;
  __REG32 IntPnd66          : 1;
  __REG32 IntPnd67          : 1;
  __REG32 IntPnd68          : 1;
  __REG32 IntPnd69          : 1;
  __REG32 IntPnd70          : 1;
  __REG32 IntPnd71          : 1;
  __REG32 IntPnd72          : 1;
  __REG32 IntPnd73          : 1;
  __REG32 IntPnd74          : 1;
  __REG32 IntPnd75          : 1;
  __REG32 IntPnd76          : 1;
  __REG32 IntPnd77          : 1;
  __REG32 IntPnd78          : 1;
  __REG32 IntPnd79          : 1;
  __REG32 IntPnd80          : 1;
  __REG32 IntPnd81          : 1;
  __REG32 IntPnd82          : 1;
  __REG32 IntPnd83          : 1;
  __REG32 IntPnd84          : 1;
  __REG32 IntPnd85          : 1;
  __REG32 IntPnd86          : 1;
  __REG32 IntPnd87          : 1;
  __REG32 IntPnd88          : 1;
  __REG32 IntPnd89          : 1;
  __REG32 IntPnd90          : 1;
  __REG32 IntPnd91          : 1;
  __REG32 IntPnd92          : 1;
  __REG32 IntPnd93          : 1;
  __REG32 IntPnd94          : 1;
  __REG32 IntPnd95          : 1;
  __REG32 IntPnd96          : 1;
} __dcanintpnd56_bits;

/* CAN Interrupt Pending Registers (DCAN INTPND78) */
typedef struct {
  __REG32 IntPnd97          : 1;
  __REG32 IntPnd98          : 1;
  __REG32 IntPnd99          : 1;
  __REG32 IntPnd100         : 1;
  __REG32 IntPnd101         : 1;
  __REG32 IntPnd102         : 1;
  __REG32 IntPnd103         : 1;
  __REG32 IntPnd104         : 1;
  __REG32 IntPnd105         : 1;
  __REG32 IntPnd106         : 1;
  __REG32 IntPnd107         : 1;
  __REG32 IntPnd108         : 1;
  __REG32 IntPnd109         : 1;
  __REG32 IntPnd110         : 1;
  __REG32 IntPnd111         : 1;
  __REG32 IntPnd112         : 1;
  __REG32 IntPnd113         : 1;
  __REG32 IntPnd114         : 1;
  __REG32 IntPnd115         : 1;
  __REG32 IntPnd116         : 1;
  __REG32 IntPnd117         : 1;
  __REG32 IntPnd118         : 1;
  __REG32 IntPnd119         : 1;
  __REG32 IntPnd120         : 1;
  __REG32 IntPnd121         : 1;
  __REG32 IntPnd122         : 1;
  __REG32 IntPnd123         : 1;
  __REG32 IntPnd124         : 1;
  __REG32 IntPnd125         : 1;
  __REG32 IntPnd126         : 1;
  __REG32 IntPnd127         : 1;
  __REG32 IntPnd128         : 1;
} __dcanintpnd78_bits;

/* CAN Message Valid X Register (DCAN MSGVAL X) */
typedef struct {
  __REG32 MsgValReg1          : 2;
  __REG32 MsgValReg2          : 2;
  __REG32 MsgValReg3          : 2;
  __REG32 MsgValReg4          : 2;
  __REG32 MsgValReg5          : 2;
  __REG32 MsgValReg6          : 2;
  __REG32 MsgValReg7          : 2;
  __REG32 MsgValReg8          : 2;
  __REG32                     :16;
} __dcanmsgval_bits;

/* CAN Message Valid Registers (DCAN MSGVAL12) */
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
} __dcanmsgval12_bits;

/* CAN Message Valid Registers (DCAN MSGVAL34) */
typedef struct {
  __REG32 MsgVal33          : 1;
  __REG32 MsgVal34          : 1;
  __REG32 MsgVal35          : 1;
  __REG32 MsgVal36          : 1;
  __REG32 MsgVal37          : 1;
  __REG32 MsgVal38          : 1;
  __REG32 MsgVal39          : 1;
  __REG32 MsgVal40          : 1;
  __REG32 MsgVal41          : 1;
  __REG32 MsgVal42          : 1;
  __REG32 MsgVal43          : 1;
  __REG32 MsgVal44          : 1;
  __REG32 MsgVal45          : 1;
  __REG32 MsgVal46          : 1;
  __REG32 MsgVal47          : 1;
  __REG32 MsgVal48          : 1;
  __REG32 MsgVal49          : 1;
  __REG32 MsgVal50          : 1;
  __REG32 MsgVal51          : 1;
  __REG32 MsgVal52          : 1;
  __REG32 MsgVal53          : 1;
  __REG32 MsgVal54          : 1;
  __REG32 MsgVal55          : 1;
  __REG32 MsgVal56          : 1;
  __REG32 MsgVal57          : 1;
  __REG32 MsgVal58          : 1;
  __REG32 MsgVal59          : 1;
  __REG32 MsgVal60          : 1;
  __REG32 MsgVal61          : 1;
  __REG32 MsgVal62          : 1;
  __REG32 MsgVal63          : 1;
  __REG32 MsgVal64          : 1;
} __dcanmsgval34_bits;

/* CAN Message Valid Registers (DCAN MSGVAL56) */
typedef struct {
  __REG32 MsgVal65          : 1;
  __REG32 MsgVal66          : 1;
  __REG32 MsgVal67          : 1;
  __REG32 MsgVal68          : 1;
  __REG32 MsgVal69          : 1;
  __REG32 MsgVal70          : 1;
  __REG32 MsgVal71          : 1;
  __REG32 MsgVal72          : 1;
  __REG32 MsgVal73          : 1;
  __REG32 MsgVal74          : 1;
  __REG32 MsgVal75          : 1;
  __REG32 MsgVal76          : 1;
  __REG32 MsgVal77          : 1;
  __REG32 MsgVal78          : 1;
  __REG32 MsgVal79          : 1;
  __REG32 MsgVal80          : 1;
  __REG32 MsgVal81          : 1;
  __REG32 MsgVal82          : 1;
  __REG32 MsgVal83          : 1;
  __REG32 MsgVal84          : 1;
  __REG32 MsgVal85          : 1;
  __REG32 MsgVal86          : 1;
  __REG32 MsgVal87          : 1;
  __REG32 MsgVal88          : 1;
  __REG32 MsgVal89          : 1;
  __REG32 MsgVal90          : 1;
  __REG32 MsgVal91          : 1;
  __REG32 MsgVal92          : 1;
  __REG32 MsgVal93          : 1;
  __REG32 MsgVal94          : 1;
  __REG32 MsgVal95          : 1;
  __REG32 MsgVal96          : 1;
} __dcanmsgval56_bits;

/* CAN Message Valid Registers (DCAN MSGVAL78) */
typedef struct {
  __REG32 MsgVal97          : 1;
  __REG32 MsgVal98          : 1;
  __REG32 MsgVal99          : 1;
  __REG32 MsgVal100         : 1;
  __REG32 MsgVal101         : 1;
  __REG32 MsgVal102         : 1;
  __REG32 MsgVal103         : 1;
  __REG32 MsgVal104         : 1;
  __REG32 MsgVal105         : 1;
  __REG32 MsgVal106         : 1;
  __REG32 MsgVal107         : 1;
  __REG32 MsgVal108         : 1;
  __REG32 MsgVal109         : 1;
  __REG32 MsgVal110         : 1;
  __REG32 MsgVal111         : 1;
  __REG32 MsgVal112         : 1;
  __REG32 MsgVal113         : 1;
  __REG32 MsgVal114         : 1;
  __REG32 MsgVal115         : 1;
  __REG32 MsgVal116         : 1;
  __REG32 MsgVal117         : 1;
  __REG32 MsgVal118         : 1;
  __REG32 MsgVal119         : 1;
  __REG32 MsgVal120         : 1;
  __REG32 MsgVal121         : 1;
  __REG32 MsgVal122         : 1;
  __REG32 MsgVal123         : 1;
  __REG32 MsgVal124         : 1;
  __REG32 MsgVal125         : 1;
  __REG32 MsgVal126         : 1;
  __REG32 MsgVal127         : 1;
  __REG32 MsgVal128         : 1;
} __dcanmsgval78_bits;

/* CAN Interrupt Multiplexer Registers (DCAN INTMUX12) */
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
} __dcanintmux12_bits;

/* CAN Interrupt Multiplexer Registers (DCAN INTMUX34) */
typedef struct {
  __REG32 IntMux33          : 1;
  __REG32 IntMux34          : 1;
  __REG32 IntMux35          : 1;
  __REG32 IntMux36          : 1;
  __REG32 IntMux37          : 1;
  __REG32 IntMux38          : 1;
  __REG32 IntMux39          : 1;
  __REG32 IntMux40          : 1;
  __REG32 IntMux41          : 1;
  __REG32 IntMux42          : 1;
  __REG32 IntMux43          : 1;
  __REG32 IntMux44          : 1;
  __REG32 IntMux45          : 1;
  __REG32 IntMux46          : 1;
  __REG32 IntMux47          : 1;
  __REG32 IntMux48          : 1;
  __REG32 IntMux49          : 1;
  __REG32 IntMux50          : 1;
  __REG32 IntMux51          : 1;
  __REG32 IntMux52          : 1;
  __REG32 IntMux53          : 1;
  __REG32 IntMux54          : 1;
  __REG32 IntMux55          : 1;
  __REG32 IntMux56          : 1;
  __REG32 IntMux57          : 1;
  __REG32 IntMux58          : 1;
  __REG32 IntMux59          : 1;
  __REG32 IntMux60          : 1;
  __REG32 IntMux61          : 1;
  __REG32 IntMux62          : 1;
  __REG32 IntMux63          : 1;
  __REG32 IntMux64          : 1;
} __dcanintmux34_bits;

/* CAN Interrupt Multiplexer Registers (DCAN INTMUX56) */
typedef struct {
  __REG32 IntMux65          : 1;
  __REG32 IntMux66          : 1;
  __REG32 IntMux67          : 1;
  __REG32 IntMux68          : 1;
  __REG32 IntMux69          : 1;
  __REG32 IntMux70          : 1;
  __REG32 IntMux71          : 1;
  __REG32 IntMux72          : 1;
  __REG32 IntMux73          : 1;
  __REG32 IntMux74          : 1;
  __REG32 IntMux75          : 1;
  __REG32 IntMux76          : 1;
  __REG32 IntMux77          : 1;
  __REG32 IntMux78          : 1;
  __REG32 IntMux79          : 1;
  __REG32 IntMux80          : 1;
  __REG32 IntMux81          : 1;
  __REG32 IntMux82          : 1;
  __REG32 IntMux83          : 1;
  __REG32 IntMux84          : 1;
  __REG32 IntMux85          : 1;
  __REG32 IntMux86          : 1;
  __REG32 IntMux87          : 1;
  __REG32 IntMux88          : 1;
  __REG32 IntMux89          : 1;
  __REG32 IntMux90          : 1;
  __REG32 IntMux91          : 1;
  __REG32 IntMux92          : 1;
  __REG32 IntMux93          : 1;
  __REG32 IntMux94          : 1;
  __REG32 IntMux95          : 1;
  __REG32 IntMux96          : 1;
} __dcanintmux56_bits;

/* CAN Interrupt Multiplexer Registers (DCAN INTMUX78) */
typedef struct {
  __REG32 IntMux97          : 1;
  __REG32 IntMux98          : 1;
  __REG32 IntMux99          : 1;
  __REG32 IntMux100         : 1;
  __REG32 IntMux101         : 1;
  __REG32 IntMux102         : 1;
  __REG32 IntMux103         : 1;
  __REG32 IntMux104         : 1;
  __REG32 IntMux105         : 1;
  __REG32 IntMux106         : 1;
  __REG32 IntMux107         : 1;
  __REG32 IntMux108         : 1;
  __REG32 IntMux109         : 1;
  __REG32 IntMux110         : 1;
  __REG32 IntMux111         : 1;
  __REG32 IntMux112         : 1;
  __REG32 IntMux113         : 1;
  __REG32 IntMux114         : 1;
  __REG32 IntMux115         : 1;
  __REG32 IntMux116         : 1;
  __REG32 IntMux117         : 1;
  __REG32 IntMux118         : 1;
  __REG32 IntMux119         : 1;
  __REG32 IntMux120         : 1;
  __REG32 IntMux121         : 1;
  __REG32 IntMux122         : 1;
  __REG32 IntMux123         : 1;
  __REG32 IntMux124         : 1;
  __REG32 IntMux125         : 1;
  __REG32 IntMux126         : 1;
  __REG32 IntMux127         : 1;
  __REG32 IntMux128         : 1;
} __dcanintmux78_bits;

/* CAN IF1/2 Command Registers (DCAN IF1/2CMD) */
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
} __dcanifcmd_bits;

/* CAN IF1/IF2/IF2 Mask Registers (DCAN IF1MSK, DCAN IF2MSK, DCAN IF3MSK) */
typedef struct {
  __REG32 Msk0                : 1;
  __REG32 Msk1                : 1;
  __REG32 Msk2                : 1;
  __REG32 Msk3                : 1;
  __REG32 Msk4                : 1;
  __REG32 Msk5                : 1;
  __REG32 Msk6                : 1;
  __REG32 Msk7                : 1;
  __REG32 Msk8                : 1;
  __REG32 Msk9                : 1;
  __REG32 Msk10               : 1;
  __REG32 Msk11               : 1;
  __REG32 Msk12               : 1;
  __REG32 Msk13               : 1;
  __REG32 Msk14               : 1;
  __REG32 Msk15               : 1;
  __REG32 Msk16               : 1;
  __REG32 Msk17               : 1;
  __REG32 Msk18               : 1;
  __REG32 Msk19               : 1;
  __REG32 Msk20               : 1;
  __REG32 Msk21               : 1;
  __REG32 Msk22               : 1;
  __REG32 Msk23               : 1;
  __REG32 Msk24               : 1;
  __REG32 Msk25               : 1;
  __REG32 Msk26               : 1;
  __REG32 Msk27               : 1;
  __REG32 Msk28               : 1;
  __REG32                     : 1;
  __REG32 MDir                : 1;
  __REG32 MXtd                : 1;
} __dcanifmsk_bits;

/* CAN IF1/IF2/IF3 Arbitration Registers (DCAN IF1ARB, DCAN IF2ARB, DCAN IF3ARB) */
typedef struct {
  __REG32 ID0                 : 1;
  __REG32 ID1                 : 1;
  __REG32 ID2                 : 1;
  __REG32 ID3                 : 1;
  __REG32 ID4                 : 1;
  __REG32 ID5                 : 1;
  __REG32 ID6                 : 1;
  __REG32 ID7                 : 1;
  __REG32 ID8                 : 1;
  __REG32 ID9                 : 1;
  __REG32 ID10                : 1;
  __REG32 ID11                : 1;
  __REG32 ID12                : 1;
  __REG32 ID13                : 1;
  __REG32 ID14                : 1;
  __REG32 ID15                : 1;
  __REG32 ID16                : 1;
  __REG32 ID17                : 1;
  __REG32 ID18                : 1;
  __REG32 ID19                : 1;
  __REG32 ID20                : 1;
  __REG32 ID21                : 1;
  __REG32 ID22                : 1;
  __REG32 ID23                : 1;
  __REG32 ID24                : 1;
  __REG32 ID25                : 1;
  __REG32 ID26                : 1;
  __REG32 ID27                : 1;
  __REG32 ID28                : 1;
  __REG32 Dir                 : 1;
  __REG32 Xtd                 : 1;
  __REG32 MsgVal              : 1;
} __dcanifarb_bits;

/* CAN IF1/IF2/IF3 Message Control Registers (DCAN IF1MCTL, DCAN IF2MCTL, DCAN IF3MCTL) */
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
} __dcanifmctl_bits;

/* CAN IF1/2/3 Data A Register (DCAN IF1/2/3DATA) */
typedef struct {
  __REG32 Data0               : 8;
  __REG32 Data1               : 8;
  __REG32 Data2               : 8;
  __REG32 Data3               : 8;
} __dcanifdata_bits;

/* CAN IF1/2/3 Data B Register (DCAN IF1/2/3DATB) */
typedef struct {
  __REG32 Data4               : 8;
  __REG32 Data5               : 8;
  __REG32 Data6               : 8;
  __REG32 Data7               : 8;
} __dcanifdatb_bits;

/* CAN IF3 Observation Register (DCAN IF3OBS) */
typedef struct {
  __REG32 Mask                : 1;
  __REG32 Arb                 : 1;
  __REG32 Ctrl                : 1;
  __REG32 DataA               : 1;
  __REG32 DataB               : 1;
  __REG32                     : 3;
  __REG32 IF3_SM              : 1;
  __REG32 IF3_SA              : 1;
  __REG32 IF3_SC              : 1;
  __REG32 IF3_SDA             : 1;
  __REG32 IF3_SDB             : 1;
  __REG32                     : 2;
  __REG32 IF3_Upd             : 1;
  __REG32                     :16;
} __dcanif3obs_bits;

/* CAN IF3 Update Enable Registers (DCAN IF3UPD12) */
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
} __dcanif3upd12_bits;

/* CAN IF3 Update Enable Registers (DCAN IF3UPD34) */
typedef struct {
  __REG32 IF3UpdEn33          : 1;
  __REG32 IF3UpdEn34          : 1;
  __REG32 IF3UpdEn35          : 1;
  __REG32 IF3UpdEn36          : 1;
  __REG32 IF3UpdEn37          : 1;
  __REG32 IF3UpdEn38          : 1;
  __REG32 IF3UpdEn39          : 1;
  __REG32 IF3UpdEn40          : 1;
  __REG32 IF3UpdEn41          : 1;
  __REG32 IF3UpdEn42          : 1;
  __REG32 IF3UpdEn43          : 1;
  __REG32 IF3UpdEn44          : 1;
  __REG32 IF3UpdEn45          : 1;
  __REG32 IF3UpdEn46          : 1;
  __REG32 IF3UpdEn47          : 1;
  __REG32 IF3UpdEn48          : 1;
  __REG32 IF3UpdEn49          : 1;
  __REG32 IF3UpdEn50          : 1;
  __REG32 IF3UpdEn51          : 1;
  __REG32 IF3UpdEn52          : 1;
  __REG32 IF3UpdEn53          : 1;
  __REG32 IF3UpdEn54          : 1;
  __REG32 IF3UpdEn55          : 1;
  __REG32 IF3UpdEn56          : 1;
  __REG32 IF3UpdEn57          : 1;
  __REG32 IF3UpdEn58          : 1;
  __REG32 IF3UpdEn59          : 1;
  __REG32 IF3UpdEn60          : 1;
  __REG32 IF3UpdEn61          : 1;
  __REG32 IF3UpdEn62          : 1;
  __REG32 IF3UpdEn63          : 1;
  __REG32 IF3UpdEn64          : 1;
} __dcanif3upd34_bits;

/* CAN IF3 Update Enable Registers (DCAN IF3UPD56) */
typedef struct {
  __REG32 IF3UpdEn65          : 1;
  __REG32 IF3UpdEn66          : 1;
  __REG32 IF3UpdEn67          : 1;
  __REG32 IF3UpdEn68          : 1;
  __REG32 IF3UpdEn69          : 1;
  __REG32 IF3UpdEn70          : 1;
  __REG32 IF3UpdEn71          : 1;
  __REG32 IF3UpdEn72          : 1;
  __REG32 IF3UpdEn73          : 1;
  __REG32 IF3UpdEn74          : 1;
  __REG32 IF3UpdEn75          : 1;
  __REG32 IF3UpdEn76          : 1;
  __REG32 IF3UpdEn77          : 1;
  __REG32 IF3UpdEn78          : 1;
  __REG32 IF3UpdEn79          : 1;
  __REG32 IF3UpdEn80          : 1;
  __REG32 IF3UpdEn81          : 1;
  __REG32 IF3UpdEn82          : 1;
  __REG32 IF3UpdEn83          : 1;
  __REG32 IF3UpdEn84          : 1;
  __REG32 IF3UpdEn85          : 1;
  __REG32 IF3UpdEn86          : 1;
  __REG32 IF3UpdEn87          : 1;
  __REG32 IF3UpdEn88          : 1;
  __REG32 IF3UpdEn89          : 1;
  __REG32 IF3UpdEn90          : 1;
  __REG32 IF3UpdEn91          : 1;
  __REG32 IF3UpdEn92          : 1;
  __REG32 IF3UpdEn93          : 1;
  __REG32 IF3UpdEn94          : 1;
  __REG32 IF3UpdEn95          : 1;
  __REG32 IF3UpdEn96          : 1;
} __dcanif3upd56_bits;

/* CAN IF3 Update Enable Registers (DCAN IF3UPD78) */
typedef struct {
  __REG32 IF3UpdEn97          : 1;
  __REG32 IF3UpdEn98          : 1;
  __REG32 IF3UpdEn99          : 1;
  __REG32 IF3UpdEn100         : 1;
  __REG32 IF3UpdEn101         : 1;
  __REG32 IF3UpdEn102         : 1;
  __REG32 IF3UpdEn103         : 1;
  __REG32 IF3UpdEn104         : 1;
  __REG32 IF3UpdEn105         : 1;
  __REG32 IF3UpdEn106         : 1;
  __REG32 IF3UpdEn107         : 1;
  __REG32 IF3UpdEn108         : 1;
  __REG32 IF3UpdEn109         : 1;
  __REG32 IF3UpdEn110         : 1;
  __REG32 IF3UpdEn111         : 1;
  __REG32 IF3UpdEn112         : 1;
  __REG32 IF3UpdEn113         : 1;
  __REG32 IF3UpdEn114         : 1;
  __REG32 IF3UpdEn115         : 1;
  __REG32 IF3UpdEn116         : 1;
  __REG32 IF3UpdEn117         : 1;
  __REG32 IF3UpdEn118         : 1;
  __REG32 IF3UpdEn119         : 1;
  __REG32 IF3UpdEn120         : 1;
  __REG32 IF3UpdEn121         : 1;
  __REG32 IF3UpdEn122         : 1;
  __REG32 IF3UpdEn123         : 1;
  __REG32 IF3UpdEn124         : 1;
  __REG32 IF3UpdEn125         : 1;
  __REG32 IF3UpdEn126         : 1;
  __REG32 IF3UpdEn127         : 1;
  __REG32 IF3UpdEn128         : 1;
} __dcanif3upd78_bits;

/* CAN TX IO Control Register (DCAN TIOC) */
typedef struct {
  __REG32 In                  : 1;
  __REG32 Out                 : 1;
  __REG32 Dir                 : 1;
  __REG32 Func                : 1;
  __REG32                     :12;
  __REG32 OD                  : 1;
  __REG32 PD                  : 1;
  __REG32 PU                  : 1;
  __REG32                     :13;
} __dcantioc_bits;

/* SPI Global Control Register 0 (SPIGCR0) */
typedef struct {
  __REG32 nRESET              : 1;
  __REG32                     :31;
} __spigcr0_bits;

/* SPI Global Control Register 1 (SPIGCR1) */
typedef struct {
  __REG32 MASTER              : 1;
  __REG32 CLKMOD              : 1;
  __REG32                     : 6;
  __REG32 POWERDOWN           : 1;
  __REG32                     : 7;
  __REG32 LOOPBACK            : 1;
  __REG32                     : 7;
  __REG32 SPIEN               : 1;
  __REG32                     : 7;
} __spigcr1_bits;

/* SPI Interrupt Register (SPIINT0) */
typedef struct {
  __REG32 DLENERRENA          : 1;
  __REG32 TIMEOUTENA          : 1;
  __REG32 PARERRENA           : 1;
  __REG32 DESYNCENA           : 1;
  __REG32 BITERRENA           : 1;
  __REG32                     : 1;
  __REG32 RXOVRNINTENA        : 1;
  __REG32                     : 1;
  __REG32 RXINTENA            : 1;
  __REG32 TXINTENA            : 1;
  __REG32                     : 6;
  __REG32 DMAREQEN            : 1;
  __REG32                     : 7;
  __REG32 ENABLEHIGHZ         : 1;
  __REG32                     : 7;
} __spiint0_bits;

/* SPI Interrupt Level Register (SPILVL) */
typedef struct {
  __REG32 DLEN_ERR_LVL        : 1;
  __REG32 TIMEOUTLVL          : 1;
  __REG32 PARERRLVL           : 1;
  __REG32 DESYNCLVL           : 1;
  __REG32 BITERRLVL           : 1;
  __REG32                     : 1;
  __REG32 RXOVRNINTLVL        : 1;
  __REG32                     : 1;
  __REG32 RXINTLVL            : 1;
  __REG32 TXINTLVL            : 1;
  __REG32                     :22;
} __spilvl_bits;

/* SPI Flag Register (SPIFLG) */
typedef struct {
  __REG32 DLEN_ERR_FLG        : 1;
  __REG32 TIMEOUTFLG          : 1;
  __REG32 PARERRFLG           : 1;
  __REG32 DESYNCFLG           : 1;
  __REG32 BITERRFLG           : 1;
  __REG32                     : 1;
  __REG32 RXOVRNINTFLG        : 1;
  __REG32                     : 1;
  __REG32 RXINTFLG            : 1;
  __REG32 TXINTFLG            : 1;
  __REG32                     :14;
  __REG32 BUFINITACTIVE       : 1;
  __REG32                     : 7;
} __spiflg_bits;

/* SPIP Pin Control Register 0 (SPIPPC0) */
typedef struct {
  __REG32 SCSFUN0             : 1;
  __REG32 SCSFUN1             : 1;
  __REG32 SCSFUN2             : 1;
  __REG32 SCSFUN3             : 1;
  __REG32 SCSFUN4             : 1;
  __REG32 SCSFUN5             : 1;
  __REG32 SCSFUN6             : 1;
  __REG32 SCSFUN7             : 1;
  __REG32 ENAFUN              : 1;
  __REG32 CLKFUN              : 1;
  __REG32 SIMOFUN0            : 1;
  __REG32 SOMIFUN0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOFUN0           : 1;
  __REG32 SIMOFUN1            : 1;
  __REG32 SIMOFUN2            : 1;
  __REG32 SIMOFUN3            : 1;
  __REG32 SIMOFUN4            : 1;
  __REG32 SIMOFUN5            : 1;
  __REG32 SIMOFUN6            : 1;
  __REG32 SIMOFUN7            : 1;
  __REG32 _SOMIFUN0           : 1;
  __REG32 SOMIFUN1            : 1;
  __REG32 SOMIFUN2            : 1;
  __REG32 SOMIFUN3            : 1;
  __REG32 SOMIFUN4            : 1;
  __REG32 SOMIFUN5            : 1;
  __REG32 SOMIFUN6            : 1;
  __REG32 SOMIFUN7            : 1;
} __spippc0_bits;

/* SPI Pin Control Register 0 (SPIPC0) */
typedef struct {
  __REG32 SCSFUN0             : 1;
  __REG32                     : 7;
  __REG32 ENAFUN              : 1;
  __REG32 CLKFUN              : 1;
  __REG32 SIMOFUN0            : 1;
  __REG32 SOMIFUN0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOFUN0           : 1;
  __REG32                     : 7;
  __REG32 _SOMIFUN0           : 1;
  __REG32                     : 7;
} __spipc0_bits;

/* SPIP Pin Control Register 1 (SPIPPC1) */
typedef struct {
  __REG32 SCSDIR0             : 1;
  __REG32 SCSDIR1             : 1;
  __REG32 SCSDIR2             : 1;
  __REG32 SCSDIR3             : 1;
  __REG32 SCSDIR4             : 1;
  __REG32 SCSDIR5             : 1;
  __REG32 SCSDIR6             : 1;
  __REG32 SCSDIR7             : 1;
  __REG32 ENADIR              : 1;
  __REG32 CLKDIR              : 1;
  __REG32 SIMODIR0            : 1;
  __REG32 SOMIDIR0            : 1;
  __REG32                     : 4;
  __REG32 _SIMODIR0           : 1;
  __REG32 SIMODIR1            : 1;
  __REG32 SIMODIR2            : 1;
  __REG32 SIMODIR3            : 1;
  __REG32 SIMODIR4            : 1;
  __REG32 SIMODIR5            : 1;
  __REG32 SIMODIR6            : 1;
  __REG32 SIMODIR7            : 1;
  __REG32 _SOMIDIR0           : 1;
  __REG32 SOMIDIR1            : 1;
  __REG32 SOMIDIR2            : 1;
  __REG32 SOMIDIR3            : 1;
  __REG32 SOMIDIR4            : 1;
  __REG32 SOMIDIR5            : 1;
  __REG32 SOMIDIR6            : 1;
  __REG32 SOMIDIR7            : 1;
} __spippc1_bits;

/* SPI Pin Control Register 1 (SPIPC1) */
typedef struct {
  __REG32 SCSDIR0             : 1;
  __REG32                     : 7;
  __REG32 ENADIR              : 1;
  __REG32 CLKDIR              : 1;
  __REG32 SIMODIR0            : 1;
  __REG32 SOMIDIR0            : 1;
  __REG32                     : 4;
  __REG32 _SIMODIR0           : 1;
  __REG32                     : 7;
  __REG32 _SOMIDIR0           : 1;
  __REG32                     : 7;
} __spipc1_bits;

/* SPIP Pin Control Register 2 (SPIPPC2) */
typedef struct {
  __REG32 SCSIN0              : 1;
  __REG32 SCSIN1              : 1;
  __REG32 SCSIN2              : 1;
  __REG32 SCSIN3              : 1;
  __REG32 SCSIN4              : 1;
  __REG32 SCSIN5              : 1;
  __REG32 SCSIN6              : 1;
  __REG32 SCSIN7              : 1;
  __REG32 ENAIN               : 1;
  __REG32 CLKIN               : 1;
  __REG32 SIMOIN0             : 1;
  __REG32 SOMIIN0             : 1;
  __REG32                     : 4;
  __REG32 _SIMOIN0            : 1;
  __REG32 SIMOIN1             : 1;
  __REG32 SIMOIN2             : 1;
  __REG32 SIMOIN3             : 1;
  __REG32 SIMOIN4             : 1;
  __REG32 SIMOIN5             : 1;
  __REG32 SIMOIN6             : 1;
  __REG32 SIMOIN7             : 1;
  __REG32 _SOMIIN0            : 1;
  __REG32 SOMIIN1             : 1;
  __REG32 SOMIIN2             : 1;
  __REG32 SOMIIN3             : 1;
  __REG32 SOMIIN4             : 1;
  __REG32 SOMIIN5             : 1;
  __REG32 SOMIIN6             : 1;
  __REG32 SOMIIN7             : 1;
} __spippc2_bits;

/* SPI Pin Control Register 2 (SPIPC2) */
typedef struct {
  __REG32 SCSIN0              : 1;
  __REG32                     : 7;
  __REG32 ENAIN               : 1;
  __REG32 CLKIN               : 1;
  __REG32 SIMOIN0             : 1;
  __REG32 SOMIIN0             : 1;
  __REG32                     : 4;
  __REG32 _SIMOIN0            : 1;
  __REG32                     : 7;
  __REG32 _SOMIIN0            : 1;
  __REG32                     : 7;
} __spipc2_bits;

/* SPIP Pin Control Register 3 (SPIPPC3) */
typedef struct {
  __REG32 SCSOUT0             : 1;
  __REG32 SCSOUT1             : 1;
  __REG32 SCSOUT2             : 1;
  __REG32 SCSOUT3             : 1;
  __REG32 SCSOUT4             : 1;
  __REG32 SCSOUT5             : 1;
  __REG32 SCSOUT6             : 1;
  __REG32 SCSOUT7             : 1;
  __REG32 ENAOUT              : 1;
  __REG32 CLKOUT              : 1;
  __REG32 SIMOOUT0            : 1;
  __REG32 SOMIOUT0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOOUT0           : 1;
  __REG32 SIMOOUT1            : 1;
  __REG32 SIMOOUT2            : 1;
  __REG32 SIMOOUT3            : 1;
  __REG32 SIMOOUT4            : 1;
  __REG32 SIMOOUT5            : 1;
  __REG32 SIMOOUT6            : 1;
  __REG32 SIMOOUT7            : 1;
  __REG32 _SOMIOUT0           : 1;
  __REG32 SOMIOUT1            : 1;
  __REG32 SOMIOUT2            : 1;
  __REG32 SOMIOUT3            : 1;
  __REG32 SOMIOUT4            : 1;
  __REG32 SOMIOUT5            : 1;
  __REG32 SOMIOUT6            : 1;
  __REG32 SOMIOUT7            : 1;
} __spippc3_bits;

/* SPI Pin Control Register 3 (SPIPC3) */
typedef struct {
  __REG32 SCSOUT0             : 1;
  __REG32                     : 7;
  __REG32 ENAOUT              : 1;
  __REG32 CLKOUT              : 1;
  __REG32 SIMOOUT0            : 1;
  __REG32 SOMIOUT0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOOUT0           : 1;
  __REG32                     : 7;
  __REG32 _SOMIOUT0           : 1;
  __REG32                     : 7;
} __spipc3_bits;

/* SPIP Pin Control Register 4 (SPIPPC4) */
typedef struct {
  __REG32 SCSSET0             : 1;
  __REG32 SCSSET1             : 1;
  __REG32 SCSSET2             : 1;
  __REG32 SCSSET3             : 1;
  __REG32 SCSSET4             : 1;
  __REG32 SCSSET5             : 1;
  __REG32 SCSSET6             : 1;
  __REG32 SCSSET7             : 1;
  __REG32 ENASET              : 1;
  __REG32 CLKSET              : 1;
  __REG32 SIMOSET0            : 1;
  __REG32 SOMISET0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOSET0           : 1;
  __REG32 SIMOSET1            : 1;
  __REG32 SIMOSET2            : 1;
  __REG32 SIMOSET3            : 1;
  __REG32 SIMOSET4            : 1;
  __REG32 SIMOSET5            : 1;
  __REG32 SIMOSET6            : 1;
  __REG32 SIMOSET7            : 1;
  __REG32 _SOMISET0           : 1;
  __REG32 SOMISET1            : 1;
  __REG32 SOMISET2            : 1;
  __REG32 SOMISET3            : 1;
  __REG32 SOMISET4            : 1;
  __REG32 SOMISET5            : 1;
  __REG32 SOMISET6            : 1;
  __REG32 SOMISET7            : 1;
} __spippc4_bits;

/* SPI Pin Control Register 4 (SPIPC4) */
typedef struct {
  __REG32 SCSSET0             : 1;
  __REG32                     : 7;
  __REG32 ENASET              : 1;
  __REG32 CLKSET              : 1;
  __REG32 SIMOSET0            : 1;
  __REG32 SOMISET0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOSET0           : 1;
  __REG32                     : 7;
  __REG32 _SOMISET0           : 1;
  __REG32                     : 7;
} __spipc4_bits;

/* SPIP Pin Control Register 5 (SPIPC5) */
typedef struct {
  __REG32 SCSCLR0             : 1;
  __REG32 SCSCLR1             : 1;
  __REG32 SCSCLR2             : 1;
  __REG32 SCSCLR3             : 1;
  __REG32 SCSCLR4             : 1;
  __REG32 SCSCLR5             : 1;
  __REG32 SCSCLR6             : 1;
  __REG32 SCSCLR7             : 1;
  __REG32 ENACLR              : 1;
  __REG32 CLKCLR              : 1;
  __REG32 SIMOCLR0            : 1;
  __REG32 SOMICLR0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOCLR0           : 1;
  __REG32 SIMOCLR1            : 1;
  __REG32 SIMOCLR2            : 1;
  __REG32 SIMOCLR3            : 1;
  __REG32 SIMOCLR4            : 1;
  __REG32 SIMOCLR5            : 1;
  __REG32 SIMOCLR6            : 1;
  __REG32 SIMOCLR7            : 1;
  __REG32 _SOMICLR0           : 1;
  __REG32 SOMICLR1            : 1;
  __REG32 SOMICLR2            : 1;
  __REG32 SOMICLR3            : 1;
  __REG32 SOMICLR4            : 1;
  __REG32 SOMICLR5            : 1;
  __REG32 SOMICLR6            : 1;
  __REG32 SOMICLR7            : 1;
} __spippc5_bits;

/* SPI Pin Control Register 5 (SPIPC5) */
typedef struct {
  __REG32 SCSCLR0             : 1;
  __REG32                     : 7;
  __REG32 ENACLR              : 1;
  __REG32 CLKCLR              : 1;
  __REG32 SIMOCLR0            : 1;
  __REG32 SOMICLR0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOCLR0           : 1;
  __REG32                     : 7;
  __REG32 _SOMICLR0           : 1;
  __REG32                     : 7;
} __spipc5_bits;

/* SPIP Pin Control Register 6 (SPIPPC6) */
typedef struct {
  __REG32 SCSPDR0             : 1;
  __REG32 SCSPDR1             : 1;
  __REG32 SCSPDR2             : 1;
  __REG32 SCSPDR3             : 1;
  __REG32 SCSPDR4             : 1;
  __REG32 SCSPDR5             : 1;
  __REG32 SCSPDR6             : 1;
  __REG32 SCSPDR7             : 1;
  __REG32 ENAPDR              : 1;
  __REG32 CLKPDR              : 1;
  __REG32 SIMOPDR0            : 1;
  __REG32 SOMIPDR0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOPDR0           : 1;
  __REG32 SIMOPDR1            : 1;
  __REG32 SIMOPDR2            : 1;
  __REG32 SIMOPDR3            : 1;
  __REG32 SIMOPDR4            : 1;
  __REG32 SIMOPDR5            : 1;
  __REG32 SIMOPDR6            : 1;
  __REG32 SIMOPDR7            : 1;
  __REG32 _SOMIPDR0           : 1;
  __REG32 SOMIPDR1            : 1;
  __REG32 SOMIPDR2            : 1;
  __REG32 SOMIPDR3            : 1;
  __REG32 SOMIPDR4            : 1;
  __REG32 SOMIPDR5            : 1;
  __REG32 SOMIPDR6            : 1;
  __REG32 SOMIPDR7            : 1;
} __spippc6_bits;

/* SPI Pin Control Register 6 (SPIPC6) */
typedef struct {
  __REG32 SCSPDR0             : 1;
  __REG32                     : 7;
  __REG32 ENAPDR              : 1;
  __REG32 CLKPDR              : 1;
  __REG32 SIMOPDR0            : 1;
  __REG32 SOMIPDR0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOPDR0           : 1;
  __REG32                     : 7;
  __REG32 _SOMIPDR0           : 1;
  __REG32                     : 7;
} __spipc6_bits;

/* SPIP Pin Control Register 7 (SPIPPC7) */
typedef struct {
  __REG32 SCSPDIS0            : 1;
  __REG32 SCSPDIS1            : 1;
  __REG32 SCSPDIS2            : 1;
  __REG32 SCSPDIS3            : 1;
  __REG32 SCSPDIS4            : 1;
  __REG32 SCSPDIS5            : 1;
  __REG32 SCSPDIS6            : 1;
  __REG32 SCSPDIS7            : 1;
  __REG32 ENAPDIS             : 1;
  __REG32 CLKPDIS             : 1;
  __REG32 SIMOPDIS0           : 1;
  __REG32 SOMIPDIS0           : 1;
  __REG32                     : 4;
  __REG32 _SIMOPDIS0          : 1;
  __REG32 SIMOPDIS1           : 1;
  __REG32 SIMOPDIS2           : 1;
  __REG32 SIMOPDIS3           : 1;
  __REG32 SIMOPDIS4           : 1;
  __REG32 SIMOPDIS5           : 1;
  __REG32 SIMOPDIS6           : 1;
  __REG32 SIMOPDIS7           : 1;
  __REG32 _SOMIPDIS0          : 1;
  __REG32 SOMIPDIS1           : 1;
  __REG32 SOMIPDIS2           : 1;
  __REG32 SOMIPDIS3           : 1;
  __REG32 SOMIPDIS4           : 1;
  __REG32 SOMIPDIS5           : 1;
  __REG32 SOMIPDIS6           : 1;
  __REG32 SOMIPDIS7           : 1;
} __spippc7_bits;

/* SPI Pin Control Register 7 (SPIPC7) */
typedef struct {
  __REG32 SCSPDIS0            : 1;
  __REG32                     : 7;
  __REG32 ENAPDIS             : 1;
  __REG32 CLKPDIS             : 1;
  __REG32 SIMOPDIS0           : 1;
  __REG32 SOMIPDIS0           : 1;
  __REG32                     : 4;
  __REG32 _SIMOPDIS0          : 1;
  __REG32                     : 7;
  __REG32 _SOMIPDIS0          : 1;
  __REG32                     : 7;
} __spipc7_bits;

/* SPI Pin Control Register 8 (SPIPPC8) */
typedef struct {
  __REG32 SCSPSEL0            : 1;
  __REG32 SCSPSEL1            : 1;
  __REG32 SCSPSEL2            : 1;
  __REG32 SCSPSEL3            : 1;
  __REG32 SCSPSEL4            : 1;
  __REG32 SCSPSEL5            : 1;
  __REG32 SCSPSEL6            : 1;
  __REG32 SCSPSEL7            : 1;
  __REG32 ENAPSEL             : 1;
  __REG32 CLKPSEL             : 1;
  __REG32 SIMOPSEL0           : 1;
  __REG32 SOMIPSEL0           : 1;
  __REG32                     : 4;
  __REG32 _SIMOPSEL0          : 1;
  __REG32 SIMOPSEL1           : 1;
  __REG32 SIMOPSEL2           : 1;
  __REG32 SIMOPSEL3           : 1;
  __REG32 SIMOPSEL4           : 1;
  __REG32 SIMOPSEL5           : 1;
  __REG32 SIMOPSEL6           : 1;
  __REG32 SIMOPSEL7           : 1;
  __REG32 _SOMIPSEL0          : 1;
  __REG32 SOMIPSEL1           : 1;
  __REG32 SOMIPSEL2           : 1;
  __REG32 SOMIPSEL3           : 1;
  __REG32 SOMIPSEL4           : 1;
  __REG32 SOMIPSEL5           : 1;
  __REG32 SOMIPSEL6           : 1;
  __REG32 SOMIPSEL7           : 1;
} __spippc8_bits;

/* SPI Pin Control Register 8 (SPIPC8) */
typedef struct {
  __REG32 SCSPSEL0            : 1;
  __REG32                     : 7;
  __REG32 ENAPSEL             : 1;
  __REG32 CLKPSEL             : 1;
  __REG32 SIMOPSEL0           : 1;
  __REG32 SOMIPSEL0           : 1;
  __REG32                     : 4;
  __REG32 _SIMOPSEL0          : 1;
  __REG32                     : 7;
  __REG32 _SOMIPSEL0          : 1;
  __REG32                     : 7;
} __spipc8_bits;

/* SPI Transmit Data Register 0 (SPIDAT0) */
typedef struct {
  __REG32 TXDATA              :16;
  __REG32                     :16;
} __spidat0_bits;

/* SPI Transmit Data Register 1 (SPIDAT1) */
typedef struct {
  __REG32 TXDATA              :16;
  __REG32 CSNR                : 8;
  __REG32 DFSEL               : 2;
  __REG32 WDEL                : 1;
  __REG32                     : 1;
  __REG32 CSHOLD              : 1;
  __REG32                     : 3;
} __spidat1_bits;

/* SPI Receive Buffer Register (SPIBUF) */
typedef struct {
  __REG32 RXDATA              :16;
  __REG32 LCSNR               : 8;
  __REG32 DLENERR             : 1;
  __REG32 TIMEOUT             : 1;
  __REG32 PARITYERR           : 1;
  __REG32 DESYNC              : 1;
  __REG32 BITERR              : 1;
  __REG32 TXFULL              : 1;
  __REG32 RXOVR               : 1;
  __REG32 RXEMPTY             : 1;
} __spibuf_bits;

/* SPI Emulation Register (SPIEMU) */
typedef struct {
  __REG32 EMU_RXDATA          :16;
  __REG32                     :16;
} __spiemu_bits;

/* SPI Delay Register (SPIDELAY) */
typedef struct {
  __REG32 C2EDELAY            : 8;
  __REG32 T2EDELAY            : 8;
  __REG32 T2CDELAY            : 8;
  __REG32 C2TDELAY            : 8;
} __spidelay_bits;

/* SPI Default Chip Select Register (SPIDEF) */
typedef struct {
  __REG32 CSDEF               : 8;
  __REG32                     :24;
} __spidef_bits;

/* SPI Data Format Registers (SPIFMT[3:0]) */
typedef struct {
  __REG32 CHARLEN             : 5;
  __REG32                     : 3;
  __REG32 PRESCALE            : 8;
  __REG32 PHASE               : 1;
  __REG32 POLARITY            : 1;
  __REG32 DIS_CS_TIMERS       : 1;
  __REG32                     : 1;
  __REG32 SHIFTDIR            : 1;
  __REG32 WAITENA             : 1;
  __REG32 PARITYENA           : 1;
  __REG32 PARPOL              : 1;
  __REG32 WDELAY              : 6;
  __REG32                     : 2;
} __spifmt_bits;

/* Interrupt Vector 0 (INTVECT0) */
typedef struct {
  __REG32 SUSPEND0            : 1;
  __REG32 INTVECT0            : 5;
  __REG32                     :26;
} __tgintvect0_bits;

/* Interrupt Vector 1 (INTVECT1) */
typedef struct {
  __REG32 SUSPEND1            : 1;
  __REG32 INTVECT1            : 5;
  __REG32                     :26;
} __tgintvect1_bits;

/* SPI Parallel/Modulo Mode Control Register (SPIPMCTRL)  */
typedef struct {
  __REG32 PMODE0              : 2;
  __REG32 MMODE0              : 3;
  __REG32 MOD_CLK_POL0        : 1;
  __REG32                     : 2;
  __REG32 PMODE1              : 2;
  __REG32 MMODE1              : 3;
  __REG32 MOD_CLK_POL1        : 1;
  __REG32                     : 2;
  __REG32 PMODE2              : 2;
  __REG32 MMODE2              : 3;
  __REG32 MOD_CLK_POL2        : 1;
  __REG32                     : 2;
  __REG32 PMODE3              : 2;
  __REG32 MMODE3              : 3;
  __REG32 MOD_CLK_POL3        : 1;
  __REG32                     : 2;
} __spipmctrl_bits;

/* SPI Multi-buffer Mode Enable Register (MIBSPIE)  */
typedef struct {
  __REG32 MSPIENA             : 1;
  __REG32                     :15;
  __REG32 RXRAMACCESS         : 1;
  __REG32                     :15;
} __spimibspie_bits;

/* TG Interrupt Enable Set Register (TGITENST) */
typedef struct {
  __REG32 SETINTENSUS0        : 1;
  __REG32 SETINTENSUS1        : 1;
  __REG32 SETINTENSUS2        : 1;
  __REG32 SETINTENSUS3        : 1;
  __REG32 SETINTENSUS4        : 1;
  __REG32 SETINTENSUS5        : 1;
  __REG32 SETINTENSUS6        : 1;
  __REG32 SETINTENSUS7        : 1;
  __REG32 SETINTENSUS8        : 1;
  __REG32 SETINTENSUS9        : 1;
  __REG32 SETINTENSUS10       : 1;
  __REG32 SETINTENSUS11       : 1;
  __REG32 SETINTENSUS12       : 1;
  __REG32 SETINTENSUS13       : 1;
  __REG32 SETINTENSUS14       : 1;
  __REG32 SETINTENSUS15       : 1;
  __REG32 SETINTENRDY0        : 1;
  __REG32 SETINTENRDY1        : 1;
  __REG32 SETINTENRDY2        : 1;
  __REG32 SETINTENRDY3        : 1;
  __REG32 SETINTENRDY4        : 1;
  __REG32 SETINTENRDY5        : 1;
  __REG32 SETINTENRDY6        : 1;
  __REG32 SETINTENRDY7        : 1;
  __REG32 SETINTENRDY8        : 1;
  __REG32 SETINTENRDY9        : 1;
  __REG32 SETINTENRDY10       : 1;
  __REG32 SETINTENRDY11       : 1;
  __REG32 SETINTENRDY12       : 1;
  __REG32 SETINTENRDY13       : 1;
  __REG32 SETINTENRDY14       : 1;
  __REG32 SETINTENRDY15       : 1;
} __tgitenst_bits;

/* TG MibSPI TG Interrupt Enable Clear Register (TGITENCR) */
typedef struct {
  __REG32 CLRINTENSUS0        : 1;
  __REG32 CLRINTENSUS1        : 1;
  __REG32 CLRINTENSUS2        : 1;
  __REG32 CLRINTENSUS3        : 1;
  __REG32 CLRINTENSUS4        : 1;
  __REG32 CLRINTENSUS5        : 1;
  __REG32 CLRINTENSUS6        : 1;
  __REG32 CLRINTENSUS7        : 1;
  __REG32 CLRINTENSUS8        : 1;
  __REG32 CLRINTENSUS9        : 1;
  __REG32 CLRINTENSUS10       : 1;
  __REG32 CLRINTENSUS11       : 1;
  __REG32 CLRINTENSUS12       : 1;
  __REG32 CLRINTENSUS13       : 1;
  __REG32 CLRINTENSUS14       : 1;
  __REG32 CLRINTENSUS15       : 1;
  __REG32 CLRINTENRDY0        : 1;
  __REG32 CLRINTENRDY1        : 1;
  __REG32 CLRINTENRDY2        : 1;
  __REG32 CLRINTENRDY3        : 1;
  __REG32 CLRINTENRDY4        : 1;
  __REG32 CLRINTENRDY5        : 1;
  __REG32 CLRINTENRDY6        : 1;
  __REG32 CLRINTENRDY7        : 1;
  __REG32 CLRINTENRDY8        : 1;
  __REG32 CLRINTENRDY9        : 1;
  __REG32 CLRINTENRDY10       : 1;
  __REG32 CLRINTENRDY11       : 1;
  __REG32 CLRINTENRDY12       : 1;
  __REG32 CLRINTENRDY13       : 1;
  __REG32 CLRINTENRDY14       : 1;
  __REG32 CLRINTENRDY15       : 1;
} __tgitencr_bits;

/* TG Transfer Group Interrupt Level Set Register (TGITLVST) */
typedef struct {
  __REG32 SETINTLVLSUS0       : 1;
  __REG32 SETINTLVLSUS1       : 1;
  __REG32 SETINTLVLSUS2       : 1;
  __REG32 SETINTLVLSUS3       : 1;
  __REG32 SETINTLVLSUS4       : 1;
  __REG32 SETINTLVLSUS5       : 1;
  __REG32 SETINTLVLSUS6       : 1;
  __REG32 SETINTLVLSUS7       : 1;
  __REG32 SETINTLVLSUS8       : 1;
  __REG32 SETINTLVLSUS9       : 1;
  __REG32 SETINTLVLSUS10      : 1;
  __REG32 SETINTLVLSUS11      : 1;
  __REG32 SETINTLVLSUS12      : 1;
  __REG32 SETINTLVLSUS13      : 1;
  __REG32 SETINTLVLSUS14      : 1;
  __REG32 SETINTLVLSUS15      : 1;
  __REG32 SETINTLVLRDY0       : 1;
  __REG32 SETINTLVLRDY1       : 1;
  __REG32 SETINTLVLRDY2       : 1;
  __REG32 SETINTLVLRDY3       : 1;
  __REG32 SETINTLVLRDY4       : 1;
  __REG32 SETINTLVLRDY5       : 1;
  __REG32 SETINTLVLRDY6       : 1;
  __REG32 SETINTLVLRDY7       : 1;
  __REG32 SETINTLVLRDY8       : 1;
  __REG32 SETINTLVLRDY9       : 1;
  __REG32 SETINTLVLRDY10      : 1;
  __REG32 SETINTLVLRDY11      : 1;
  __REG32 SETINTLVLRDY12      : 1;
  __REG32 SETINTLVLRDY13      : 1;
  __REG32 SETINTLVLRDY14      : 1;
  __REG32 SETINTLVLRDY15      : 1;
} __tgitlvst_bits;

/* TG Transfer Group Interrupt Level Clear Register (TGITLVCR) */
typedef struct {
  __REG32 CLRINTLVLSUS0       : 1;
  __REG32 CLRINTLVLSUS1       : 1;
  __REG32 CLRINTLVLSUS2       : 1;
  __REG32 CLRINTLVLSUS3       : 1;
  __REG32 CLRINTLVLSUS4       : 1;
  __REG32 CLRINTLVLSUS5       : 1;
  __REG32 CLRINTLVLSUS6       : 1;
  __REG32 CLRINTLVLSUS7       : 1;
  __REG32 CLRINTLVLSUS8       : 1;
  __REG32 CLRINTLVLSUS9       : 1;
  __REG32 CLRINTLVLSUS10      : 1;
  __REG32 CLRINTLVLSUS11      : 1;
  __REG32 CLRINTLVLSUS12      : 1;
  __REG32 CLRINTLVLSUS13      : 1;
  __REG32 CLRINTLVLSUS14      : 1;
  __REG32 CLRINTLVLSUS15      : 1;
  __REG32 CLRINTLVLRDY0       : 1;
  __REG32 CLRINTLVLRDY1       : 1;
  __REG32 CLRINTLVLRDY2       : 1;
  __REG32 CLRINTLVLRDY3       : 1;
  __REG32 CLRINTLVLRDY4       : 1;
  __REG32 CLRINTLVLRDY5       : 1;
  __REG32 CLRINTLVLRDY6       : 1;
  __REG32 CLRINTLVLRDY7       : 1;
  __REG32 CLRINTLVLRDY8       : 1;
  __REG32 CLRINTLVLRDY9       : 1;
  __REG32 CLRINTLVLRDY10      : 1;
  __REG32 CLRINTLVLRDY11      : 1;
  __REG32 CLRINTLVLRDY12      : 1;
  __REG32 CLRINTLVLRDY13      : 1;
  __REG32 CLRINTLVLRDY14      : 1;
  __REG32 CLRINTLVLRDY15      : 1;
} __tgitlvcr_bits;

/* TG Transfer Group Interrupt Flag Register (TGINTFLAG) */
typedef struct {
  __REG32 INTFLGSUS0          : 1;
  __REG32 INTFLGSUS1          : 1;
  __REG32 INTFLGSUS2          : 1;
  __REG32 INTFLGSUS3          : 1;
  __REG32 INTFLGSUS4          : 1;
  __REG32 INTFLGSUS5          : 1;
  __REG32 INTFLGSUS6          : 1;
  __REG32 INTFLGSUS7          : 1;
  __REG32 INTFLGSUS8          : 1;
  __REG32 INTFLGSUS9          : 1;
  __REG32 INTFLGSUS10         : 1;
  __REG32 INTFLGSUS11         : 1;
  __REG32 INTFLGSUS12         : 1;
  __REG32 INTFLGSUS13         : 1;
  __REG32 INTFLGSUS14         : 1;
  __REG32 INTFLGSUS15         : 1;
  __REG32 INTFLGRDY0          : 1;
  __REG32 INTFLGRDY1          : 1;
  __REG32 INTFLGRDY2          : 1;
  __REG32 INTFLGRDY3          : 1;
  __REG32 INTFLGRDY4          : 1;
  __REG32 INTFLGRDY5          : 1;
  __REG32 INTFLGRDY6          : 1;
  __REG32 INTFLGRDY7          : 1;
  __REG32 INTFLGRDY8          : 1;
  __REG32 INTFLGRDY9          : 1;
  __REG32 INTFLGRDY10         : 1;
  __REG32 INTFLGRDY11         : 1;
  __REG32 INTFLGRDY12         : 1;
  __REG32 INTFLGRDY13         : 1;
  __REG32 INTFLGRDY14         : 1;
  __REG32 INTFLGRDY15         : 1;
} __tgitflg_bits;

/* Tick Count Register (TICKCNT) */
typedef struct {
  __REG32 TICKVALUE           :16;
  __REG32                     :12;
  __REG32 CLKCTRL             : 2;
  __REG32 RELOAD              : 1;
  __REG32 TICKENA             : 1;
} __tgtickcnt_bits;

/* Last TG End Pointer (LTGPEND) */
typedef struct {
  __REG32                     : 8;
  __REG32 LPEND               : 7;
  __REG32                     : 9;
  __REG32 TG_IN_SERVICE       : 5;
  __REG32                     : 3;
} __tgltgpend_bits;

/* TGx Control Registers (TGCTRL) */
typedef struct {
  __REG32 PCURRENT            : 7;
  __REG32                     : 1;
  __REG32 PSTART              : 7;
  __REG32                     : 1;
  __REG32 TRIGSRC             : 4;
  __REG32 TRIGEVT             : 4;
  __REG32                     : 4;
  __REG32 TGTD                : 1;
  __REG32 PRST                : 1;
  __REG32 ONESHOT             : 1;
  __REG32 TGENA               : 1;
} __tgctrl_bits;

/* SPI DMA Channel Control Register (DMAxCTRL) */
typedef struct {
  __REG32 COUNT               : 6;
  __REG32 COUNT_BIT17         : 1;
  __REG32                     : 1;
  __REG32 ICOUNT              : 5;
  __REG32 NOBRK               : 1;
  __REG32 TXDMAENA            : 1;
  __REG32 RXDMAENA            : 1;
  __REG32 TXDMA_MAP           : 4;
  __REG32 RXDMA_MAP           : 4;
  __REG32 BUFID               : 7;
  __REG32 ONESHOT             : 1;
} __spidmactrl_bits;

/* SPI DMAxCOUNT Register (ICOUNT) */
typedef struct {
  __REG32 COUNT               :16;
  __REG32 ICOUNT              :16;
} __spidmacount_bits;

/* SPI DMA Large Count Register (DMACNTLEN) */
typedef struct {
  __REG32 LARGE_COUNT         : 1;
  __REG32                     :31;
} __spidmacntlen_bits;

/* SPI DMA Large Count Register (DMACNTLEN) */
typedef struct {
  __REG32 EDEN                : 4;
  __REG32                     : 4;
  __REG32 PTESTEN             : 1;
  __REG32                     :23;
} __spiuerrctrl_bits;

/* SPI Multi-buffer RAM Uncorrectable Parity Error Status Register (UERRSTAT) */
typedef struct {
  __REG32 EDFLG0              : 1;
  __REG32 EDFLG1              : 1;
  __REG32                     :30;
} __spiuerrstat_bits;

/* SPI RXRAM Uncorrectable Parity Error Address Register (UERRADDR1) */
typedef struct {
  __REG32 UERRADDR1           :10;
  __REG32                     :22;
} __spiuerraddr1_bits;

/* SPI RXRAM Uncorrectable Parity Error Address Register (UERRADDR0) */
typedef struct {
  __REG32 UERRADDR0           : 9;
  __REG32                     :23;
} __spiuerraddr0_bits;

/* SPI RXRAM Overrun Buffer Address Register (RXOVRN_BUF_ADDR) */
typedef struct {
  __REG32 RXOVRN_BUF_ADDR     :10;
  __REG32                     :22;
} __spirxovrn_buf_addr_bits;

/* SPI I/O-Loopback Test Control Register (IOLPBKTSTCR) */
typedef struct {
  __REG32 RXP_ENA             : 1;
  __REG32 LPBK_TYPE           : 1;
  __REG32 CTRL_SCS_PIN_ERR    : 1;
  __REG32 ERR_SCS_PIN         : 3;
  __REG32                     : 2;
  __REG32 IOLPBKTSTENA        : 4;
  __REG32                     : 4;
  __REG32 CTRL_DLEN_ERR       : 1;
  __REG32 CTRL_TIME_OUT       : 1;
  __REG32 CTRL_PAR_ERR        : 1;
  __REG32 CTRLD_ES_YNC        : 1;
  __REG32 CTRL_BIT_ERR        : 1;
  __REG32                     : 3;
  __REG32 SCS_FAIL_FLG        : 1;
  __REG32                     : 7;
} __spiiolpbktstcr_bits;

/* Global Configuration Register (HETGCR) */
typedef struct {
  __REG32 TO                  : 1;
  __REG32 IS                  : 1;
  __REG32 DSF                 : 1;
  __REG32                     : 5;
  __REG32 _64A                : 1;
  __REG32                     : 7;
  __REG32 CMS                 : 1;
  __REG32                     : 7;
  __REG32 PD                  : 1;
  __REG32                     : 7;
} __hetgcr_bits;

/* Prescale Factor Register (HETPFR) */
typedef struct {
  __REG32 HRPFC               : 6;
  __REG32                     : 2;
  __REG32 LRPFC               : 3;
  __REG32                     :21;
} __hetpfr_bits;

/* NHET Current Address Register (HETADDR) */
typedef struct {
  __REG32 HETADDR             : 8;
  __REG32                     :24;
} __hetaddr_bits;

/* NHET Offset Index Priority Level 1 Register (HETOFF1) */
typedef struct {
  __REG32 OFFSET1             : 8;
  __REG32                     :24;
} __hetoff1_bits;

/* NHET Offset Index Priority Level 2 Register (HETOFF2) */
typedef struct {
  __REG32 OFFSET2             : 8;
  __REG32                     :24;
} __hetoff2_bits;

/* Exception Control Register 1 (HETEXC1) */
typedef struct {
  __REG32 PrgmOvrflPry        : 1;
  __REG32 APCNTUndrflPry      : 1;
  __REG32 APCNTOvrflPry       : 1;
  __REG32                     : 5;
  __REG32 PrgmOvrflEna        : 1;
  __REG32                     : 7;
  __REG32 APCNTUndrflEna      : 1;
  __REG32                     : 7;
  __REG32 APCNTOvrflEna       : 1;
  __REG32                     : 7;
} __hetexc1_bits;

/* Exception Control Register 2 (HETEXC2) */
typedef struct {
  __REG32 PrgmOvrflflg        : 1;
  __REG32 APCNTUndflflg       : 1;
  __REG32 APCNTOvrflflg       : 1;
  __REG32                     :29;
} __hetexc2_bits;

/* Interrupt Priority Register (HETPRY) */
typedef struct {
  __REG32 HETPRY0             : 1;
  __REG32 HETPRY1             : 1;
  __REG32 HETPRY2             : 1;
  __REG32 HETPRY3             : 1;
  __REG32 HETPRY4             : 1;
  __REG32 HETPRY5             : 1;
  __REG32 HETPRY6             : 1;
  __REG32 HETPRY7             : 1;
  __REG32 HETPRY8             : 1;
  __REG32 HETPRY9             : 1;
  __REG32 HETPRY10            : 1;
  __REG32 HETPRY11            : 1;
  __REG32 HETPRY12            : 1;
  __REG32 HETPRY13            : 1;
  __REG32 HETPRY14            : 1;
  __REG32 HETPRY15            : 1;
  __REG32 HETPRY16            : 1;
  __REG32 HETPRY17            : 1;
  __REG32 HETPRY18            : 1;
  __REG32 HETPRY19            : 1;
  __REG32 HETPRY20            : 1;
  __REG32 HETPRY21            : 1;
  __REG32 HETPRY22            : 1;
  __REG32 HETPRY23            : 1;
  __REG32 HETPRY24            : 1;
  __REG32 HETPRY25            : 1;
  __REG32 HETPRY26            : 1;
  __REG32 HETPRY27            : 1;
  __REG32 HETPRY28            : 1;
  __REG32 HETPRY29            : 1;
  __REG32 HETPRY30            : 1;
  __REG32 HETPRY31            : 1;
} __hetpry_bits;

/* Interrupt Flag Register (HETFLG) */
typedef struct {
  __REG32 HETFLAG0             : 1;
  __REG32 HETFLAG1             : 1;
  __REG32 HETFLAG2             : 1;
  __REG32 HETFLAG3             : 1;
  __REG32 HETFLAG4             : 1;
  __REG32 HETFLAG5             : 1;
  __REG32 HETFLAG6             : 1;
  __REG32 HETFLAG7             : 1;
  __REG32 HETFLAG8             : 1;
  __REG32 HETFLAG9             : 1;
  __REG32 HETFLAG10            : 1;
  __REG32 HETFLAG11            : 1;
  __REG32 HETFLAG12            : 1;
  __REG32 HETFLAG13            : 1;
  __REG32 HETFLAG14            : 1;
  __REG32 HETFLAG15            : 1;
  __REG32 HETFLAG16            : 1;
  __REG32 HETFLAG17            : 1;
  __REG32 HETFLAG18            : 1;
  __REG32 HETFLAG19            : 1;
  __REG32 HETFLAG20            : 1;
  __REG32 HETFLAG21            : 1;
  __REG32 HETFLAG22            : 1;
  __REG32 HETFLAG23            : 1;
  __REG32 HETFLAG24            : 1;
  __REG32 HETFLAG25            : 1;
  __REG32 HETFLAG26            : 1;
  __REG32 HETFLAG27            : 1;
  __REG32 HETFLAG28            : 1;
  __REG32 HETFLAG29            : 1;
  __REG32 HETFLAG30            : 1;
  __REG32 HETFLAG31            : 1;
} __hetflg_bits;

/* HR Share Control Register (HETHRSH) */
typedef struct {
  __REG32 HRShare_1_0          : 1;
  __REG32 HRShare_3_2          : 1;
  __REG32 HRShare_5_4          : 1;
  __REG32 HRShare_7_6          : 1;
  __REG32 HRShare_9_8          : 1;
  __REG32 HRShare_11_10        : 1;
  __REG32 HRShare_13_12        : 1;
  __REG32 HRShare_15_14        : 1;
  __REG32 HRShare_17_16        : 1;
  __REG32 HRShare_19_18        : 1;
  __REG32 HRShare_21_20        : 1;
  __REG32 HRShare_23_22        : 1;
  __REG32 						         :20;
} __hethrsh_bits;

/* HR XOR-share Control Register (HETXOR) */
typedef struct {
  __REG32 HR_XORShare_1_0      : 1;
  __REG32 HR_XORShare_3_2      : 1;
  __REG32 HR_XORShare_5_4      : 1;
  __REG32 HR_XORShare_7_6      : 1;
  __REG32 HR_XORShare_9_8      : 1;
  __REG32 HR_XORShare_11_10    : 1;
  __REG32 HR_XORShare_13_12    : 1;
  __REG32 HR_XORShare_15_14    : 1;
  __REG32 HR_XORShare_17_16    : 1;
  __REG32 HR_XORShare_19_18    : 1;
  __REG32 HR_XORShare_21_20    : 1;
  __REG32 HR_XORShare_23_22    : 1;
  __REG32 									   :20;
} __hetxor_bits;

/* NHET Direction Register (HETDIR) */
typedef struct {
  __REG32 HETDIR0              : 1;
  __REG32 HETDIR1              : 1;
  __REG32 HETDIR2              : 1;
  __REG32 HETDIR3              : 1;
  __REG32 HETDIR4              : 1;
  __REG32 HETDIR5              : 1;
  __REG32 HETDIR6              : 1;
  __REG32 HETDIR7              : 1;
  __REG32 HETDIR8              : 1;
  __REG32 HETDIR9              : 1;
  __REG32 HETDIR10             : 1;
  __REG32 HETDIR11             : 1;
  __REG32 HETDIR12             : 1;
  __REG32 HETDIR13             : 1;
  __REG32 HETDIR14             : 1;
  __REG32 HETDIR15             : 1;
  __REG32 HETDIR16             : 1;
  __REG32 HETDIR17             : 1;
  __REG32 HETDIR18             : 1;
  __REG32 HETDIR19             : 1;
  __REG32 HETDIR20             : 1;
  __REG32 HETDIR21             : 1;
  __REG32 HETDIR22             : 1;
  __REG32 HETDIR23             : 1;
  __REG32 HETDIR24             : 1;
  __REG32 HETDIR25             : 1;
  __REG32 HETDIR26             : 1;
  __REG32 HETDIR27             : 1;
  __REG32 HETDIR28             : 1;
  __REG32 HETDIR29             : 1;
  __REG32 HETDIR30             : 1;
  __REG32 HETDIR31             : 1;
} __hetdir_bits;

/* NHET Data Input Register (HETDIN) */
typedef struct {
  __REG32 HETDIN0              : 1;
  __REG32 HETDIN1              : 1;
  __REG32 HETDIN2              : 1;
  __REG32 HETDIN3              : 1;
  __REG32 HETDIN4              : 1;
  __REG32 HETDIN5              : 1;
  __REG32 HETDIN6              : 1;
  __REG32 HETDIN7              : 1;
  __REG32 HETDIN8              : 1;
  __REG32 HETDIN9              : 1;
  __REG32 HETDIN10             : 1;
  __REG32 HETDIN11             : 1;
  __REG32 HETDIN12             : 1;
  __REG32 HETDIN13             : 1;
  __REG32 HETDIN14             : 1;
  __REG32 HETDIN15             : 1;
  __REG32 HETDIN16             : 1;
  __REG32 HETDIN17             : 1;
  __REG32 HETDIN18             : 1;
  __REG32 HETDIN19             : 1;
  __REG32 HETDIN20             : 1;
  __REG32 HETDIN21             : 1;
  __REG32 HETDIN22             : 1;
  __REG32 HETDIN23             : 1;
  __REG32 HETDIN24             : 1;
  __REG32 HETDIN25             : 1;
  __REG32 HETDIN26             : 1;
  __REG32 HETDIN27             : 1;
  __REG32 HETDIN28             : 1;
  __REG32 HETDIN29             : 1;
  __REG32 HETDIN30             : 1;
  __REG32 HETDIN31             : 1;
} __hetdin_bits;

/* NHET Data Output Register (HETDOUT) */
typedef struct {
  __REG32 HETDOUT0              : 1;
  __REG32 HETDOUT1              : 1;
  __REG32 HETDOUT2              : 1;
  __REG32 HETDOUT3              : 1;
  __REG32 HETDOUT4              : 1;
  __REG32 HETDOUT5              : 1;
  __REG32 HETDOUT6              : 1;
  __REG32 HETDOUT7              : 1;
  __REG32 HETDOUT8              : 1;
  __REG32 HETDOUT9              : 1;
  __REG32 HETDOUT10             : 1;
  __REG32 HETDOUT11             : 1;
  __REG32 HETDOUT12             : 1;
  __REG32 HETDOUT13             : 1;
  __REG32 HETDOUT14             : 1;
  __REG32 HETDOUT15             : 1;
  __REG32 HETDOUT16             : 1;
  __REG32 HETDOUT17             : 1;
  __REG32 HETDOUT18             : 1;
  __REG32 HETDOUT19             : 1;
  __REG32 HETDOUT20             : 1;
  __REG32 HETDOUT21             : 1;
  __REG32 HETDOUT22             : 1;
  __REG32 HETDOUT23             : 1;
  __REG32 HETDOUT24             : 1;
  __REG32 HETDOUT25             : 1;
  __REG32 HETDOUT26             : 1;
  __REG32 HETDOUT27             : 1;
  __REG32 HETDOUT28             : 1;
  __REG32 HETDOUT29             : 1;
  __REG32 HETDOUT30             : 1;
  __REG32 HETDOUT31             : 1;
} __hetdout_bits;

/* NHET Data Set Register (HETDSET) */
typedef struct {
  __REG32 HETDSET0              : 1;
  __REG32 HETDSET1              : 1;
  __REG32 HETDSET2              : 1;
  __REG32 HETDSET3              : 1;
  __REG32 HETDSET4              : 1;
  __REG32 HETDSET5              : 1;
  __REG32 HETDSET6              : 1;
  __REG32 HETDSET7              : 1;
  __REG32 HETDSET8              : 1;
  __REG32 HETDSET9              : 1;
  __REG32 HETDSET10             : 1;
  __REG32 HETDSET11             : 1;
  __REG32 HETDSET12             : 1;
  __REG32 HETDSET13             : 1;
  __REG32 HETDSET14             : 1;
  __REG32 HETDSET15             : 1;
  __REG32 HETDSET16             : 1;
  __REG32 HETDSET17             : 1;
  __REG32 HETDSET18             : 1;
  __REG32 HETDSET19             : 1;
  __REG32 HETDSET20             : 1;
  __REG32 HETDSET21             : 1;
  __REG32 HETDSET22             : 1;
  __REG32 HETDSET23             : 1;
  __REG32 HETDSET24             : 1;
  __REG32 HETDSET25             : 1;
  __REG32 HETDSET26             : 1;
  __REG32 HETDSET27             : 1;
  __REG32 HETDSET28             : 1;
  __REG32 HETDSET29             : 1;
  __REG32 HETDSET30             : 1;
  __REG32 HETDSET31             : 1;
} __hetdset_bits;

/* NHET Data Clear Register (HETDCLR) */
typedef struct {
  __REG32 HETDCLR0              : 1;
  __REG32 HETDCLR1              : 1;
  __REG32 HETDCLR2              : 1;
  __REG32 HETDCLR3              : 1;
  __REG32 HETDCLR4              : 1;
  __REG32 HETDCLR5              : 1;
  __REG32 HETDCLR6              : 1;
  __REG32 HETDCLR7              : 1;
  __REG32 HETDCLR8              : 1;
  __REG32 HETDCLR9              : 1;
  __REG32 HETDCLR10             : 1;
  __REG32 HETDCLR11             : 1;
  __REG32 HETDCLR12             : 1;
  __REG32 HETDCLR13             : 1;
  __REG32 HETDCLR14             : 1;
  __REG32 HETDCLR15             : 1;
  __REG32 HETDCLR16             : 1;
  __REG32 HETDCLR17             : 1;
  __REG32 HETDCLR18             : 1;
  __REG32 HETDCLR19             : 1;
  __REG32 HETDCLR20             : 1;
  __REG32 HETDCLR21             : 1;
  __REG32 HETDCLR22             : 1;
  __REG32 HETDCLR23             : 1;
  __REG32 HETDCLR24             : 1;
  __REG32 HETDCLR25             : 1;
  __REG32 HETDCLR26             : 1;
  __REG32 HETDCLR27             : 1;
  __REG32 HETDCLR28             : 1;
  __REG32 HETDCLR29             : 1;
  __REG32 HETDCLR30             : 1;
  __REG32 HETDCLR31             : 1;
} __hetdclr_bits;

/* NHET Open Drain Register (HETPDR) */
typedef struct {
  __REG32 HETPDR0              : 1;
  __REG32 HETPDR1              : 1;
  __REG32 HETPDR2              : 1;
  __REG32 HETPDR3              : 1;
  __REG32 HETPDR4              : 1;
  __REG32 HETPDR5              : 1;
  __REG32 HETPDR6              : 1;
  __REG32 HETPDR7              : 1;
  __REG32 HETPDR8              : 1;
  __REG32 HETPDR9              : 1;
  __REG32 HETPDR10             : 1;
  __REG32 HETPDR11             : 1;
  __REG32 HETPDR12             : 1;
  __REG32 HETPDR13             : 1;
  __REG32 HETPDR14             : 1;
  __REG32 HETPDR15             : 1;
  __REG32 HETPDR16             : 1;
  __REG32 HETPDR17             : 1;
  __REG32 HETPDR18             : 1;
  __REG32 HETPDR19             : 1;
  __REG32 HETPDR20             : 1;
  __REG32 HETPDR21             : 1;
  __REG32 HETPDR22             : 1;
  __REG32 HETPDR23             : 1;
  __REG32 HETPDR24             : 1;
  __REG32 HETPDR25             : 1;
  __REG32 HETPDR26             : 1;
  __REG32 HETPDR27             : 1;
  __REG32 HETPDR28             : 1;
  __REG32 HETPDR29             : 1;
  __REG32 HETPDR30             : 1;
  __REG32 HETPDR31             : 1;
} __hetpdr_bits;

/* NHET Pull Disable Register (HETPULDIS) */
typedef struct {
  __REG32 HETPULDIS0              : 1;
  __REG32 HETPULDIS1              : 1;
  __REG32 HETPULDIS2              : 1;
  __REG32 HETPULDIS3              : 1;
  __REG32 HETPULDIS4              : 1;
  __REG32 HETPULDIS5              : 1;
  __REG32 HETPULDIS6              : 1;
  __REG32 HETPULDIS7              : 1;
  __REG32 HETPULDIS8              : 1;
  __REG32 HETPULDIS9              : 1;
  __REG32 HETPULDIS10             : 1;
  __REG32 HETPULDIS11             : 1;
  __REG32 HETPULDIS12             : 1;
  __REG32 HETPULDIS13             : 1;
  __REG32 HETPULDIS14             : 1;
  __REG32 HETPULDIS15             : 1;
  __REG32 HETPULDIS16             : 1;
  __REG32 HETPULDIS17             : 1;
  __REG32 HETPULDIS18             : 1;
  __REG32 HETPULDIS19             : 1;
  __REG32 HETPULDIS20             : 1;
  __REG32 HETPULDIS21             : 1;
  __REG32 HETPULDIS22             : 1;
  __REG32 HETPULDIS23             : 1;
  __REG32 HETPULDIS24             : 1;
  __REG32 HETPULDIS25             : 1;
  __REG32 HETPULDIS26             : 1;
  __REG32 HETPULDIS27             : 1;
  __REG32 HETPULDIS28             : 1;
  __REG32 HETPULDIS29             : 1;
  __REG32 HETPULDIS30             : 1;
  __REG32 HETPULDIS31             : 1;
} __hetpuldis_bits;

/* NHET Pull Select Register (HETPSL) */
typedef struct {
  __REG32 HETPSL0              : 1;
  __REG32 HETPSL1              : 1;
  __REG32 HETPSL2              : 1;
  __REG32 HETPSL3              : 1;
  __REG32 HETPSL4              : 1;
  __REG32 HETPSL5              : 1;
  __REG32 HETPSL6              : 1;
  __REG32 HETPSL7              : 1;
  __REG32 HETPSL8              : 1;
  __REG32 HETPSL9              : 1;
  __REG32 HETPSL10             : 1;
  __REG32 HETPSL11             : 1;
  __REG32 HETPSL12             : 1;
  __REG32 HETPSL13             : 1;
  __REG32 HETPSL14             : 1;
  __REG32 HETPSL15             : 1;
  __REG32 HETPSL16             : 1;
  __REG32 HETPSL17             : 1;
  __REG32 HETPSL18             : 1;
  __REG32 HETPSL19             : 1;
  __REG32 HETPSL20             : 1;
  __REG32 HETPSL21             : 1;
  __REG32 HETPSL22             : 1;
  __REG32 HETPSL23             : 1;
  __REG32 HETPSL24             : 1;
  __REG32 HETPSL25             : 1;
  __REG32 HETPSL26             : 1;
  __REG32 HETPSL27             : 1;
  __REG32 HETPSL28             : 1;
  __REG32 HETPSL29             : 1;
  __REG32 HETPSL30             : 1;
  __REG32 HETPSL31             : 1;
} __hetpsl_bits;

/* HET Loopback Pair Select Register (HETLPBSEL) */
typedef struct {
  __REG32 HETLPBSEL_1_0        : 1;
  __REG32 HETLPBSEL_3_2        : 1;
  __REG32 HETLPBSEL_5_4        : 1;
  __REG32 HETLPBSEL_7_6        : 1;
  __REG32 HETLPBSEL_9_8        : 1;
  __REG32 HETLPBSEL_11_10      : 1;
  __REG32 HETLPBSEL_13_12      : 1;
  __REG32 HETLPBSEL_15_14      : 1;
  __REG32 HETLPBSEL_17_16      : 1;
  __REG32 HETLPBSEL_19_18      : 1;
  __REG32 HETLPBSEL_21_20      : 1;
  __REG32 HETLPBSEL_23_22      : 1;
  __REG32 HETLPBSEL_25_24      : 1;
  __REG32 HETLPBSEL_27_26      : 1;
  __REG32 HETLPBSEL_29_28      : 1;
  __REG32 HETLPBSEL_31_30      : 1;
  __REG32 HETLPBTYPE_1_0       : 1;
  __REG32 HETLPBTYPE_3_2       : 1;
  __REG32 HETLPBTYPE_5_4       : 1;
  __REG32 HETLPBTYPE_7_6       : 1;
  __REG32 HETLPBTYPE_9_8       : 1;
  __REG32 HETLPBTYPE_11_10     : 1;
  __REG32 HETLPBTYPE_13_12     : 1;
  __REG32 HETLPBTYPE_15_14     : 1;
  __REG32 HETLPBTYPE_17_16     : 1;
  __REG32 HETLPBTYPE_19_18     : 1;
  __REG32 HETLPBTYPE_21_20     : 1;
  __REG32 HETLPBTYPE_23_22     : 1;
  __REG32 HETLPBTYPE_25_24     : 1;
  __REG32 HETLPBTYPE_27_26     : 1;
  __REG32 HETLPBTYPE_29_28     : 1;
  __REG32 HETLPBTYPE_31_30     : 1;
} __hetlpbsel_bits;

/* HET Loopback Pair Direction Register (HETLPBDIR) */
typedef struct {
  __REG32 HETLPBDIR_1_0        : 1;
  __REG32 HETLPBDIR_3_2        : 1;
  __REG32 HETLPBDIR_5_4        : 1;
  __REG32 HETLPBDIR_7_6        : 1;
  __REG32 HETLPBDIR_9_8        : 1;
  __REG32 HETLPBDIR_11_10      : 1;
  __REG32 HETLPBDIR_13_12      : 1;
  __REG32 HETLPBDIR_15_14      : 1;
  __REG32 HETLPBDIR_17_16      : 1;
  __REG32 HETLPBDIR_19_18      : 1;
  __REG32 HETLPBDIR_21_20      : 1;
  __REG32 HETLPBDIR_23_22      : 1;
  __REG32 HETLPBDIR_25_24      : 1;
  __REG32 HETLPBDIR_27_26      : 1;
  __REG32 HETLPBDIR_29_28      : 1;
  __REG32 HETLPBDIR_31_30      : 1;
  __REG32 							       :16;
} __hetlpbdir_bits;

/* Parity Control Register (HETPCR) */
typedef struct {
  __REG32 PARITY_ENA		       : 4;
  __REG32 						         : 4;
  __REG32 TEST				         : 1;
  __REG32 						         : 7;
  __REG32 HETSTOP			         : 1;
  __REG32 						         :15;
} __hetpcr_bits;

/* HET Parity Interrupt Enable Register (HETPIEN) */
typedef struct {
  __REG32 INTEN		       			 : 1;
  __REG32 						         :31;
} __hetpien_bits;

/* HET Parity Interrupt Flag Register (HETPIFLG) */
typedef struct {
  __REG32 INTFLG		       		 : 1;
  __REG32 						         :31;
} __hetpiflg_bits;

/* HET Parity Address Register (HETPAR) */
typedef struct {
  __REG32 ERROR_ADDRESS    		 :11;
  __REG32 						         :21;
} __hetpar_bits;

/* STC global control register0 (STCGCR0) */
typedef struct {
  __REG32 RS_CNT			    		 : 1;
  __REG32 						         :15;
  __REG32 INTCOUNT		    		 :16;
} __stcgcr0_bits;

/* STC Global Control Register1 (STCGCR1) */
typedef struct {
  __REG32 STC_ENA			    		 : 4;
  __REG32 						         :28;
} __stcgcr1_bits;

/* STC Current Interval Count Register (STCCICR) */
typedef struct {
  __REG32 N						    		 :16;
  __REG32 						         :16;
} __stccicr_bits;

/* SelfTest Global Status Register (STCGSTAT) */
typedef struct {
  __REG32 TEST_DONE						 : 1;
  __REG32 						         :31;
} __stcgstat_bits;

/* SelfTest Fail Status Register (STCFSTAT) */
typedef struct {
  __REG32 CPU1_FAIL						 : 1;
  __REG32 						         : 1;
  __REG32 TO_ERR							 : 1;
  __REG32 						         :29;
} __stcfstat_bits;

/* ADC Reset Control Register (ADRSTCR) */
typedef struct {
  __REG32 RESET               : 1;
  __REG32                     :31;
} __adrstcr_bits;

/* ADC Operating Mode Control Register (ADOPMODECR) */
typedef struct {
  __REG32 ADC_EN              : 1;
  __REG32                     :15;
  __REG32 RAM_TEST_EN         : 1;
  __REG32                     : 7;
  __REG32 COS                 : 1;
  __REG32                     : 7;
} __adopmodecr_bits;

/* ADC Clock Control Register (ADCLOCKCR) */
typedef struct {
  __REG32 PS                  : 5;
  __REG32                     :27;
} __adclockcr_bits;

/* ADC Calibration Mode Control Register (ADCALCR) */
typedef struct {
  __REG32 CAL_EN              : 1;
  __REG32                     : 7;
  __REG32 HILO                : 1;
  __REG32 BRIDGE_EN           : 1;
  __REG32                     : 6;
  __REG32 CAL_ST              : 1;
  __REG32                     : 7;
  __REG32 SELF_TEST           : 1;
  __REG32                     : 7;
} __adcalcr_bits;

/* ADC Event Group Operating Mode Control Register (ADEVMODECR) */
typedef struct {
  __REG32 FRZ_EV              : 1;
  __REG32 EV_MODE             : 1;
  __REG32 EV_8BIT             : 1;
  __REG32                     : 1;
  __REG32 OVR_EV_RAM_IGN      : 1;
  __REG32 EV_CHID             : 1;
  __REG32                     :26;
} __adevmodecr_bits;

/* ADC Group1 Operating Mode Control Register (ADG1MODECR) */
typedef struct {
  __REG32 FRZ_G1              : 1;
  __REG32 G1_MODE             : 1;
  __REG32 G1_8Bit             : 1;
  __REG32 G1_HW_TRIG          : 1;
  __REG32 OVR_G1_RAM_IGN      : 1;
  __REG32 G1_CHID             : 1;
  __REG32                     :26;
} __adg1modecr_bits;

/* ADC Group2 Operating Mode Control Register (ADG2MODECR) */
typedef struct {
  __REG32 FRZ_G2              : 1;
  __REG32 G2_MODE             : 1;
  __REG32 G2_8Bit             : 1;
  __REG32 G2_GW_TRIG          : 1;
  __REG32 OVR_G2_RAM_IGN      : 1;
  __REG32 G2_CHID             : 1;
  __REG32                     :26;
} __adg2modecr_bits;

/* ADC Event Group Trigger Source Select Register (ADEVSRC) */
typedef struct {
  __REG32 EV_SRC              : 3;
  __REG32 EV_EDG_SEL          : 1;
  __REG32                     :28;
} __adevsrc_bits;

/* ADC Group1 Trigger Source Select Register (ADG1SRC) */
typedef struct {
  __REG32 G1_SRC              : 3;
  __REG32 G1_EDG_SEL          : 1;
  __REG32                     :28;
} __adg1src_bits;

/* ADC Group2 Trigger Source Select Register (ADG2SRC) */
typedef struct {
  __REG32 G2_SRC              : 3;
  __REG32 G2_EDG_SEL          : 1;
  __REG32                     :28;
} __adg2src_bits;

/* ADC Event Group Interrupt Enable Control Register (ADEVINTENA) */
typedef struct {
  __REG32 EV_THR_INT_EN       : 1;
  __REG32 EV_OVR_INT_EN       : 1;
  __REG32                     : 1;
  __REG32 EV_END_INT_EN       : 1;
  __REG32                     :28;
} __adevintena_bits;

/* ADC Group1 Interrupt Enable Control Register (ADG1INTENA) */
typedef struct {
  __REG32 G1_THR_INT_EN       : 1;
  __REG32 G1_OVR_INT_EN       : 1;
  __REG32                     : 1;
  __REG32 G1_END_INT_EN       : 1;
  __REG32                     :28;
} __adg1intena_bits;

/* ADC Group2 Interrupt Enable Control Register (ADG2INTENA) */
typedef struct {
  __REG32 G2_THR_INT_EN       : 1;
  __REG32 G2_OVR_INT_EN       : 1;
  __REG32                     : 1;
  __REG32 G2_END_INT_EN       : 1;
  __REG32                     :28;
} __adg2intena_bits;

/* ADC Event Group Interrupt Flag Register (ADEVINTFLG) */
typedef struct {
  __REG32 EV_THR_INT_FLG      : 1;
  __REG32 EV_MEM_OVERRUN      : 1;
  __REG32 EV_MEM_EMPTY        : 1;
  __REG32 EV_END              : 1;
  __REG32                     :28;
} __adevintflg_bits;

/* ADC Group1 Interrupt Flag Register (ADG1INTFLG) */
typedef struct {
  __REG32 G1_THR_INT_FLG      : 1;
  __REG32 G1_MEM_OVERRUN      : 1;
  __REG32 G1_MEM_EMPTY        : 1;
  __REG32 G1_END              : 1;
  __REG32                     :28;
} __adg1intflg_bits;

/* ADC Group2 Interrupt Flag Register (ADG2INTFLG) */
typedef struct {
  __REG32 G2_THR_INT_FLG      : 1;
  __REG32 G2_MEM_OVERRUN      : 1;
  __REG32 G2_MEM_EMPTY        : 1;
  __REG32 G2_END              : 1;
  __REG32                     :28;
} __adg2intflg_bits;

/* ADC Event Group Threshold Interrupt Control Register (ADEVTHRINTCR) */
typedef struct {
  __REG32 EVTHR               : 9;
  __REG32 SING                : 7;
  __REG32                     :16;
} __adevintcr_bits;

/* ADC Group1 Threshold Interrupt Control Register (ADG1THRINTCR) */
typedef struct {
  __REG32 G1THR               : 9;
  __REG32 SING                : 7;
  __REG32                     :16;
} __adg1intcr_bits;

/* ADC Group2 Threshold Interrupt Control Register (ADG2THRINTCR) */
typedef struct {
  __REG32 G2THR               : 9;
  __REG32 SING                : 7;
  __REG32                     :16;
} __adg2intcr_bits;

/* ADC Results Memory Configuration Register (ADBNDCR) */
typedef struct {
  __REG32 BNDB                : 9;
  __REG32                     : 7;
  __REG32 BNDA                : 9;
  __REG32                     : 7;
} __adbndcr_bits;

/* ADC Results Memory Size Configuration Register (ADBNDEND) */
typedef struct {
  __REG32 BNDEND              : 3;
  __REG32                     :13;
  __REG32 BUF_INIT_ACTIVE     : 1;
  __REG32                     :15;
} __adbndend_bits;

/* ADC Event Group Sampling Time Configuration Register (ADEVSAMP) */
typedef struct {
  __REG32 EV_ACQ              :12;
  __REG32                     :20;
} __adevsamp_bits;

/* ADC Group1 Sampling Time Configuration Register (ADG1SAMP) */
typedef struct {
  __REG32 G1_ACQ              :12;
  __REG32                     :20;
} __adg1samp_bits;

/* ADC Group2 Sampling Time Configuration Register (ADG2SAMP) */
typedef struct {
  __REG32 G2_ACQ              :12;
  __REG32                     :20;
} __adg2samp_bits;

/* ADC Event Group Status Register (ADEVSR) */
typedef struct {
  __REG32 EV_END              : 1;
  __REG32 EV_STOP             : 1;
  __REG32 EVBUSY              : 1;
  __REG32 EV_MEM_EMPTY        : 1;
  __REG32                     :28;
} __adevsr_bits;

/* ADC Group1 Status Register (ADG1SR) */
typedef struct {
  __REG32 G1_END              : 1;
  __REG32 G1_STOP             : 1;
  __REG32 G1BUSY              : 1;
  __REG32 G1_MEM_EMPTY        : 1;
  __REG32                     :28;
} __adg1sr_bits;

/* ADC Group2 Status Register (ADG2SR) */
typedef struct {
  __REG32 G2_END              : 1;
  __REG32 G2_STOP             : 1;
  __REG32 G2BUSY              : 1;
  __REG32 G2_MEM_EMPTY        : 1;
  __REG32                     :28;
} __adg2sr_bits;

/* ADC Event Group Channel Select Register (ADEVSEL) */
typedef struct {
  __REG32 EV_SEL0             : 1;
  __REG32 EV_SEL1             : 1;
  __REG32 EV_SEL2             : 1;
  __REG32 EV_SEL3             : 1;
  __REG32 EV_SEL4             : 1;
  __REG32 EV_SEL5             : 1;
  __REG32 EV_SEL6             : 1;
  __REG32 EV_SEL7             : 1;
  __REG32 EV_SEL8             : 1;
  __REG32 EV_SEL9             : 1;
  __REG32 EV_SEL10            : 1;
  __REG32 EV_SEL11            : 1;
  __REG32 EV_SEL12            : 1;
  __REG32 EV_SEL13            : 1;
  __REG32 EV_SEL14            : 1;
  __REG32 EV_SEL15            : 1;
  __REG32 				            :16;
} __adevsel_bits;

/* ADC Group1 Channel Select Register (ADG1SEL) */
typedef struct {
  __REG32 G1_SEL0             : 1;
  __REG32 G1_SEL1             : 1;
  __REG32 G1_SEL2             : 1;
  __REG32 G1_SEL3             : 1;
  __REG32 G1_SEL4             : 1;
  __REG32 G1_SEL5             : 1;
  __REG32 G1_SEL6             : 1;
  __REG32 G1_SEL7             : 1;
  __REG32 G1_SEL8             : 1;
  __REG32 G1_SEL9             : 1;
  __REG32 G1_SEL10            : 1;
  __REG32 G1_SEL11            : 1;
  __REG32 G1_SEL12            : 1;
  __REG32 G1_SEL13            : 1;
  __REG32 G1_SEL14            : 1;
  __REG32 G1_SEL15            : 1;
  __REG32 				            :16;
} __adg1sel_bits;

/* ADC Group2 Channel Select Register (ADG2SEL) */
typedef struct {
  __REG32 G2_SEL0             : 1;
  __REG32 G2_SEL1             : 1;
  __REG32 G2_SEL2             : 1;
  __REG32 G2_SEL3             : 1;
  __REG32 G2_SEL4             : 1;
  __REG32 G2_SEL5             : 1;
  __REG32 G2_SEL6             : 1;
  __REG32 G2_SEL7             : 1;
  __REG32 G2_SEL8             : 1;
  __REG32 G2_SEL9             : 1;
  __REG32 G2_SEL10            : 1;
  __REG32 G2_SEL11            : 1;
  __REG32 G2_SEL12            : 1;
  __REG32 G2_SEL13            : 1;
  __REG32 G2_SEL14            : 1;
  __REG32 G2_SEL15            : 1;
  __REG32 				            :16;
} __adg2sel_bits;

/* ADC Calibration and Error Offset Correction Register (ADCALR) */
typedef struct {
  __REG32 ADCALR              :10;
  __REG32                     :22;
} __adcalr_bits;

/* ADC Channel Last Conversion Value Register (ADLASTCONV) */
typedef struct {
  __REG32 LAST_CONV0          : 1;
  __REG32 LAST_CONV1          : 1;
  __REG32 LAST_CONV2          : 1;
  __REG32 LAST_CONV3          : 1;
  __REG32 LAST_CONV4          : 1;
  __REG32 LAST_CONV5          : 1;
  __REG32 LAST_CONV6          : 1;
  __REG32 LAST_CONV7          : 1;
  __REG32 LAST_CONV8          : 1;
  __REG32 LAST_CONV9          : 1;
  __REG32 LAST_CONV10         : 1;
  __REG32 LAST_CONV11         : 1;
  __REG32 LAST_CONV12         : 1;
  __REG32 LAST_CONV13         : 1;
  __REG32 LAST_CONV14         : 1;
  __REG32 LAST_CONV15         : 1;
  __REG32 					          :16;
} __adlastconv_bits;

/* ADC Event Group Results FIFO (ADEVBUFFER) */
typedef struct {
  __REG32 EV_DR		            :10;
  __REG32 EV_CHID             : 5;
  __REG32 EV_EMPTY		        : 1;
  __REG32                     :16;
} __adevbuffer_bits;

/* ADC Group1 Results FIFO (ADG1BUFFER) */
typedef struct {
  __REG32 G1_DR               :10;
  __REG32 G1_CHID             : 5;
  __REG32 G1_EMPTY            : 1;
  __REG32                     :16;
} __adg1buffer_bits;

/* ADC Group2 Results FIFO (ADG2BUFFER) */
typedef struct {
  __REG32 G2_DR               :10;
  __REG32 G2_CHID             : 5;
  __REG32 G2_EMPTY            : 1;
  __REG32                     :16;
} __adg2buffer_bits;

/* ADC ADEVT Pin Direction Control Register (ADEVTDIR) */
typedef struct {
  __REG32 ADEVT_DIR           : 1;
  __REG32                     :31;
} __adevtdir_bits;

/* ADC ADEVT Pin Output Value Control Register (ADEVTOUT) */
typedef struct {
  __REG32 ADEVT_OUT           : 1;
  __REG32                     :31;
} __adevtout_bits;

/* ADC ADEVT Pin Input Value Register (ADEVTIN) */
typedef struct {
  __REG32 ADEVT_IN            : 1;
  __REG32                     :31;
} __adevtin_bits;

/* ADC ADEVT Pin Set Register (ADEVTSET) */
typedef struct {
  __REG32 ADEVT_SET           : 1;
  __REG32                     :31;
} __adevtset_bits;

/* ADC ADEVT Pin Clear Register (ADEVTCLR) */
typedef struct {
  __REG32 ADEVT_CLR           : 1;
  __REG32                     :31;
} __adevtclr_bits;

/* ADC ADEVT Pin Open Drain Enable Register (ADEVTPDR) */
typedef struct {
  __REG32 ADEVT_PDR           : 1;
  __REG32                     :31;
} __adevtpdr_bits;

/* ADC ADEVT Pin Pull Control Disable Register (ADEVTPDIS) */
typedef struct {
  __REG32 ADEVT_PDIS          : 1;
  __REG32                     :31;
} __adevtpdis_bits;

/* ADC ADEVT Pin Pull Control Select Register (ADEVTPSEL) */
typedef struct {
  __REG32 ADEVT_PSEL          : 1;
  __REG32                     :31;
} __adevtpsel_bits;

/* ADC Event Group Sample Cap Discharge Control Register (ADEVSAMPDISEN) */
typedef struct {
  __REG32 EV_SAMP_DIS_EN      : 1;
  __REG32                     : 7;
  __REG32 EV_SAMP_DIS_CYC     : 8;
  __REG32                     :16;
} __adevsampdisen_bits;

/* ADC Group1 Sample Cap Discharge Control Register (ADG1SAMPDISEN) */
typedef struct {
  __REG32 G1_SAMP_DIS_EN      : 1;
  __REG32                     : 7;
  __REG32 G1_SAMP_DIS_CYC     : 8;
  __REG32                     :16;
} __adg1sampdisen_bits;

/* ADC Group2 Sample Cap Discharge Control Register (ADG2SAMPDISEN) */
typedef struct {
  __REG32 G2_SAMP_DIS_EN      : 1;
  __REG32                     : 7;
  __REG32 G2_SAMP_DIS_CYC     : 8;
  __REG32                     :16;
} __adg2sampdisen_bits;

/* ADC Magnitude Compare Interruptx Control Registers (ADMAGINTxCR) */
typedef struct {
  __REG32 CMP_GE_LT           : 1;
  __REG32 CHN_THR_COMP        : 1;
  __REG32                     : 6;
  __REG32 COMP_CHID           : 5;
  __REG32                     : 3;
  __REG32 MAG_THR             :10;
  __REG32 MAG_CHID		        : 5;
  __REG32                     : 1;
} __admagintcr_bits;

/* ADC Magnitude Compare Mask (ADMAGxMASK) */
typedef struct {
  __REG32 MAG_INT_MASK0       : 1;
  __REG32 MAG_INT_MASK1       : 1;
  __REG32 MAG_INT_MASK2       : 1;
  __REG32 MAG_INT_MASK3       : 1;
  __REG32 MAG_INT_MASK4       : 1;
  __REG32 MAG_INT_MASK5       : 1;
  __REG32 MAG_INT_MASK6       : 1;
  __REG32 MAG_INT_MASK7       : 1;
  __REG32 MAG_INT_MASK8       : 1;
  __REG32 MAG_INT_MASK9       : 1;
  __REG32                     :22;
} __admagmask_bits;

/* ADC Magnitude Compare Interrupt Enable Set (ADMAGINTENASET) */
typedef struct {
  __REG32 MAG_INT_ENA_SET0    : 1;
  __REG32 MAG_INT_ENA_SET1    : 1;
  __REG32 MAG_INT_ENA_SET2    : 1;
  __REG32 MAG_INT_ENA_SET3    : 1;
  __REG32 MAG_INT_ENA_SET4    : 1;
  __REG32 MAG_INT_ENA_SET5    : 1;
  __REG32                     :26;
} __admagintenaset_bits;

/* ADC Magnitude Compare Interrupt Enable Clear (ADMAGINTENACLR) */
typedef struct {
  __REG32 MAG_INT_ENA_CLR0    : 1;
  __REG32 MAG_INT_ENA_CLR1    : 1;
  __REG32 MAG_INT_ENA_CLR2    : 1;
  __REG32 MAG_INT_ENA_CLR3    : 1;
  __REG32 MAG_INT_ENA_CLR4    : 1;
  __REG32 MAG_INT_ENA_CLR5    : 1;
  __REG32                     :26;
} __admagintenaclr_bits;

/* ADC Magnitude Compare Interrupt Flag (ADMAGINTFLG) */
typedef struct {
  __REG32 MAG_INT_FLG0        : 1;
  __REG32 MAG_INT_FLG1        : 1;
  __REG32 MAG_INT_FLG2        : 1;
  __REG32 MAG_INT_FLG3        : 1;
  __REG32 MAG_INT_FLG4        : 1;
  __REG32 MAG_INT_FLG5        : 1;
  __REG32                     :26;
} __admagthrintflg_bits;

/* ADC Magnitude Compare Interrupt Offset (ADMAGINTOFF) */
typedef struct {
  __REG32 MAG_INT_OFF         : 4;
  __REG32                     :28;
} __admagthrintoffset_bits;

/* ADC Event Group FIFO Reset Control Register (ADEVFIFORESETCR) */
typedef struct {
  __REG32 EV_FIFO_RESET       : 1;
  __REG32                     :31;
} __adevfiforesetcr_bits;

/* ADC Group1 FIFO Reset Control Register (ADG1FIFORESETCR) */
typedef struct {
  __REG32 G1_FIFO_RESET       : 1;
  __REG32                     :31;
} __adg1fiforesetcr_bits;

/* ADC Group2 FIFO Reset Control Register (ADG2FIFORESETCR) */
typedef struct {
  __REG32 G2_FIFO_RESET       : 1;
  __REG32                     :31;
} __adg2fiforesetcr_bits;

/* ADC Event Group RAM Write Address (ADEVRAMWRADDR) */
typedef struct {
  __REG32 EV_RAM_ADDR         : 9;
  __REG32                     :23;
} __adevramaddr_bits;

/* ADC Group1 RAM Write Address (ADG1RAMWRADDR) */
typedef struct {
  __REG32 G1_RAM_ADDR         : 9;
  __REG32                     :23;
} __adg1ramaddr_bits;

/* ADC Group2 RAM Write Address (ADG2RAMWRADDR) */
typedef struct {
  __REG32 G2_RAM_ADDR         : 9;
  __REG32                     :23;
} __adg2ramaddr_bits;

/* ADC Parity Control Register (ADPARCR) */
typedef struct {
  __REG32 PARITY_ENA          : 4;
  __REG32                     : 4;
  __REG32 TEST                : 1;
  __REG32                     :23;
} __adparcr_bits;

/* ADC Parity Error Address (ADPARADDR) */
typedef struct {
  __REG32 ADDR                :12;
  __REG32                     :20;
} __adparaddr_bits;

/* RTI Global Control Register (RTIGCTRL) */
typedef struct {
  __REG32 CNT0EN             : 1;
  __REG32 CNT1EN             : 1;
  __REG32                    :13;
  __REG32 COS                : 1;
  __REG32                    :16;
} __rtigctrl_bits;

/* RTI Capture Control Register (RTICAPCTRL) */
typedef struct {
  __REG32 CAPCNTR0            : 1;
  __REG32 CAPCNTR1            : 1;
  __REG32                     :30;
} __rticapctrl_bits;

/* RTI Compare Control Register (RTICOMPCTRL) */
typedef struct {
  __REG32 COMPSEL0            : 1;
  __REG32                     : 3;
  __REG32 COMPSEL1            : 1;
  __REG32                     : 3;
  __REG32 COMPSEL2            : 1;
  __REG32                     : 3;
  __REG32 COMPSEL3            : 1;
  __REG32                     :19;
} __rticompctrl_bits;

/* RTI Set/Status Interrupt Register (RTISETINT) */
typedef struct {
  __REG32 SETINT0             : 1;
  __REG32 SETINT1             : 1;
  __REG32 SETINT2             : 1;
  __REG32 SETINT3             : 1;
  __REG32                     :13;
  __REG32 SETOVL0INT          : 1;
  __REG32 SETOVL1INT          : 1;
  __REG32                     :13;
} __rtisetint_bits;

/* RTI Clear/Status Interrupt Register (RTICLEARINT) */
typedef struct {
  __REG32 CLEARINT0           : 1;
  __REG32 CLEARINT1           : 1;
  __REG32 CLEARINT2           : 1;
  __REG32 CLEARINT3           : 1;
  __REG32                     :13;
  __REG32 CLEAROVL0INT        : 1;
  __REG32 CLEAROVL1INT        : 1;
  __REG32                     :13;
} __rticlearint_bits;

/* RTI Interrupt Flag Register (RTIINTFLAG) */
typedef struct {
  __REG32 INT0                : 1;
  __REG32 INT1                : 1;
  __REG32 INT2                : 1;
  __REG32 INT3                : 1;
  __REG32                     :13;
  __REG32 OVL0INT             : 1;
  __REG32 OVL1INT             : 1;
  __REG32                     :13;
} __rtiintflag_bits;

/* Digital Watchdog Preload Register (RTIDWDPRLD) */
typedef struct {
  __REG32 DWDPRLD             :12;
  __REG32                     :20;
} __rtidwdprld_bits;

/* Watchdog Status Register (RTIWDSTATUS) */
typedef struct {
  __REG32                     : 1;
  __REG32 DWD_ST             	: 1;
  __REG32 KEY_ST             	: 1;
  __REG32                     :29;
} __rtiwdstatus_bits;

/* RTI Watchdog Key Register (RTIWDKEY) */
typedef struct {
  __REG32 WDKEY             	:16;
  __REG32                     :16;
} __rtiwdkey_bits;

/* RTI Digital Watchdog Down Counter (RTIDWDCNTR) */
typedef struct {
  __REG32 DWDCNTR            	:25;
  __REG32                     : 7;
} __rtidwdcntr_bits;

/* ESM Influence Error Pin Set/Status Register 1 (ESMIEPSR1) */
typedef struct {
  __REG32 IEPSET0            	: 1;
  __REG32 IEPSET1            	: 1;
  __REG32 IEPSET2            	: 1;
  __REG32 IEPSET3            	: 1;
  __REG32 IEPSET4            	: 1;
  __REG32 IEPSET5            	: 1;
  __REG32 IEPSET6            	: 1;
  __REG32 IEPSET7            	: 1;
  __REG32 IEPSET8            	: 1;
  __REG32 IEPSET9            	: 1;
  __REG32 IEPSET10           	: 1;
  __REG32 IEPSET11           	: 1;
  __REG32 IEPSET12           	: 1;
  __REG32 IEPSET13           	: 1;
  __REG32 IEPSET14           	: 1;
  __REG32 IEPSET15           	: 1;
  __REG32 IEPSET16           	: 1;
  __REG32 IEPSET17           	: 1;
  __REG32 IEPSET18           	: 1;
  __REG32 IEPSET19           	: 1;
  __REG32 IEPSET20           	: 1;
  __REG32 IEPSET21           	: 1;
  __REG32 IEPSET22           	: 1;
  __REG32 IEPSET23           	: 1;
  __REG32 IEPSET24           	: 1;
  __REG32 IEPSET25           	: 1;
  __REG32 IEPSET26           	: 1;
  __REG32 IEPSET27           	: 1;
  __REG32 IEPSET28           	: 1;
  __REG32 IEPSET29           	: 1;
  __REG32 IEPSET30           	: 1;
  __REG32 IEPSET31           	: 1;
} __esmiepsr1_bits;

/* ESM Influence Error Pin Clear/Status Register 1 (ESMIEPCR1) */
typedef struct {
  __REG32 IEPCLR0            	: 1;
  __REG32 IEPCLR1            	: 1;
  __REG32 IEPCLR2            	: 1;
  __REG32 IEPCLR3            	: 1;
  __REG32 IEPCLR4            	: 1;
  __REG32 IEPCLR5            	: 1;
  __REG32 IEPCLR6            	: 1;
  __REG32 IEPCLR7            	: 1;
  __REG32 IEPCLR8            	: 1;
  __REG32 IEPCLR9            	: 1;
  __REG32 IEPCLR10           	: 1;
  __REG32 IEPCLR11           	: 1;
  __REG32 IEPCLR12           	: 1;
  __REG32 IEPCLR13           	: 1;
  __REG32 IEPCLR14           	: 1;
  __REG32 IEPCLR15           	: 1;
  __REG32 IEPCLR16           	: 1;
  __REG32 IEPCLR17           	: 1;
  __REG32 IEPCLR18           	: 1;
  __REG32 IEPCLR19           	: 1;
  __REG32 IEPCLR20           	: 1;
  __REG32 IEPCLR21           	: 1;
  __REG32 IEPCLR22           	: 1;
  __REG32 IEPCLR23           	: 1;
  __REG32 IEPCLR24           	: 1;
  __REG32 IEPCLR25           	: 1;
  __REG32 IEPCLR26           	: 1;
  __REG32 IEPCLR27           	: 1;
  __REG32 IEPCLR28           	: 1;
  __REG32 IEPCLR29           	: 1;
  __REG32 IEPCLR30           	: 1;
  __REG32 IEPCLR31           	: 1;
} __esmiepcr1_bits;

/* ESM Interrupt Enable Set/Status Register 1 (ESMIESR1) */
typedef struct {
  __REG32 INTENSET0            	: 1;
  __REG32 INTENSET1            	: 1;
  __REG32 INTENSET2            	: 1;
  __REG32 INTENSET3            	: 1;
  __REG32 INTENSET4            	: 1;
  __REG32 INTENSET5            	: 1;
  __REG32 INTENSET6            	: 1;
  __REG32 INTENSET7            	: 1;
  __REG32 INTENSET8            	: 1;
  __REG32 INTENSET9            	: 1;
  __REG32 INTENSET10           	: 1;
  __REG32 INTENSET11           	: 1;
  __REG32 INTENSET12           	: 1;
  __REG32 INTENSET13           	: 1;
  __REG32 INTENSET14           	: 1;
  __REG32 INTENSET15           	: 1;
  __REG32 INTENSET16           	: 1;
  __REG32 INTENSET17           	: 1;
  __REG32 INTENSET18           	: 1;
  __REG32 INTENSET19           	: 1;
  __REG32 INTENSET20           	: 1;
  __REG32 INTENSET21           	: 1;
  __REG32 INTENSET22           	: 1;
  __REG32 INTENSET23           	: 1;
  __REG32 INTENSET24           	: 1;
  __REG32 INTENSET25           	: 1;
  __REG32 INTENSET26           	: 1;
  __REG32 INTENSET27           	: 1;
  __REG32 INTENSET28           	: 1;
  __REG32 INTENSET29           	: 1;
  __REG32 INTENSET30           	: 1;
  __REG32 INTENSET31           	: 1;
} __esmiesr1_bits;

/* ESM Interrupt Enable Set/Status Register 1 (ESMIESR1) */
typedef struct {
  __REG32 INTENCLR0            	: 1;
  __REG32 INTENCLR1            	: 1;
  __REG32 INTENCLR2            	: 1;
  __REG32 INTENCLR3            	: 1;
  __REG32 INTENCLR4            	: 1;
  __REG32 INTENCLR5            	: 1;
  __REG32 INTENCLR6            	: 1;
  __REG32 INTENCLR7            	: 1;
  __REG32 INTENCLR8            	: 1;
  __REG32 INTENCLR9            	: 1;
  __REG32 INTENCLR10           	: 1;
  __REG32 INTENCLR11           	: 1;
  __REG32 INTENCLR12           	: 1;
  __REG32 INTENCLR13           	: 1;
  __REG32 INTENCLR14           	: 1;
  __REG32 INTENCLR15           	: 1;
  __REG32 INTENCLR16           	: 1;
  __REG32 INTENCLR17           	: 1;
  __REG32 INTENCLR18           	: 1;
  __REG32 INTENCLR19           	: 1;
  __REG32 INTENCLR20           	: 1;
  __REG32 INTENCLR21           	: 1;
  __REG32 INTENCLR22           	: 1;
  __REG32 INTENCLR23           	: 1;
  __REG32 INTENCLR24           	: 1;
  __REG32 INTENCLR25           	: 1;
  __REG32 INTENCLR26           	: 1;
  __REG32 INTENCLR27           	: 1;
  __REG32 INTENCLR28           	: 1;
  __REG32 INTENCLR29           	: 1;
  __REG32 INTENCLR30           	: 1;
  __REG32 INTENCLR31           	: 1;
} __esmiecr1_bits;

/* ESM Interrupt Level Set/Status Register 1 (ESMILSR1) */
typedef struct {
  __REG32 INTLVLSET0            	: 1;
  __REG32 INTLVLSET1            	: 1;
  __REG32 INTLVLSET2            	: 1;
  __REG32 INTLVLSET3            	: 1;
  __REG32 INTLVLSET4            	: 1;
  __REG32 INTLVLSET5            	: 1;
  __REG32 INTLVLSET6            	: 1;
  __REG32 INTLVLSET7            	: 1;
  __REG32 INTLVLSET8            	: 1;
  __REG32 INTLVLSET9            	: 1;
  __REG32 INTLVLSET10           	: 1;
  __REG32 INTLVLSET11           	: 1;
  __REG32 INTLVLSET12           	: 1;
  __REG32 INTLVLSET13           	: 1;
  __REG32 INTLVLSET14           	: 1;
  __REG32 INTLVLSET15           	: 1;
  __REG32 INTLVLSET16           	: 1;
  __REG32 INTLVLSET17           	: 1;
  __REG32 INTLVLSET18           	: 1;
  __REG32 INTLVLSET19           	: 1;
  __REG32 INTLVLSET20           	: 1;
  __REG32 INTLVLSET21           	: 1;
  __REG32 INTLVLSET22           	: 1;
  __REG32 INTLVLSET23           	: 1;
  __REG32 INTLVLSET24           	: 1;
  __REG32 INTLVLSET25           	: 1;
  __REG32 INTLVLSET26           	: 1;
  __REG32 INTLVLSET27           	: 1;
  __REG32 INTLVLSET28           	: 1;
  __REG32 INTLVLSET29           	: 1;
  __REG32 INTLVLSET30           	: 1;
  __REG32 INTLVLSET31           	: 1;
} __esmilsr1_bits;

/* ESM Interrupt Level Clear/Status Register 1 (ESMILCR1) */
typedef struct {
  __REG32 INTLVLCLR0            	: 1;
  __REG32 INTLVLCLR1            	: 1;
  __REG32 INTLVLCLR2            	: 1;
  __REG32 INTLVLCLR3            	: 1;
  __REG32 INTLVLCLR4            	: 1;
  __REG32 INTLVLCLR5            	: 1;
  __REG32 INTLVLCLR6            	: 1;
  __REG32 INTLVLCLR7            	: 1;
  __REG32 INTLVLCLR8            	: 1;
  __REG32 INTLVLCLR9            	: 1;
  __REG32 INTLVLCLR10           	: 1;
  __REG32 INTLVLCLR11           	: 1;
  __REG32 INTLVLCLR12           	: 1;
  __REG32 INTLVLCLR13           	: 1;
  __REG32 INTLVLCLR14           	: 1;
  __REG32 INTLVLCLR15           	: 1;
  __REG32 INTLVLCLR16           	: 1;
  __REG32 INTLVLCLR17           	: 1;
  __REG32 INTLVLCLR18           	: 1;
  __REG32 INTLVLCLR19           	: 1;
  __REG32 INTLVLCLR20           	: 1;
  __REG32 INTLVLCLR21           	: 1;
  __REG32 INTLVLCLR22           	: 1;
  __REG32 INTLVLCLR23           	: 1;
  __REG32 INTLVLCLR24           	: 1;
  __REG32 INTLVLCLR25           	: 1;
  __REG32 INTLVLCLR26           	: 1;
  __REG32 INTLVLCLR27           	: 1;
  __REG32 INTLVLCLR28           	: 1;
  __REG32 INTLVLCLR29           	: 1;
  __REG32 INTLVLCLR30           	: 1;
  __REG32 INTLVLCLR31           	: 1;
} __esmilcr1_bits;

/* ESM Interrupt Level Clear/Status Register 1-3 (ESMILCR1-3) */
/* ESM Status Shadow Register 2 (ESMSSR2) */
typedef struct {
  __REG32 ESF0            	: 1;
  __REG32 ESF1            	: 1;
  __REG32 ESF2            	: 1;
  __REG32 ESF3            	: 1;
  __REG32 ESF4            	: 1;
  __REG32 ESF5            	: 1;
  __REG32 ESF6            	: 1;
  __REG32 ESF7            	: 1;
  __REG32 ESF8            	: 1;
  __REG32 ESF9            	: 1;
  __REG32 ESF10           	: 1;
  __REG32 ESF11           	: 1;
  __REG32 ESF12           	: 1;
  __REG32 ESF13           	: 1;
  __REG32 ESF14           	: 1;
  __REG32 ESF15           	: 1;
  __REG32 ESF16           	: 1;
  __REG32 ESF17           	: 1;
  __REG32 ESF18           	: 1;
  __REG32 ESF19           	: 1;
  __REG32 ESF20           	: 1;
  __REG32 ESF21           	: 1;
  __REG32 ESF22           	: 1;
  __REG32 ESF23           	: 1;
  __REG32 ESF24           	: 1;
  __REG32 ESF25           	: 1;
  __REG32 ESF26           	: 1;
  __REG32 ESF27           	: 1;
  __REG32 ESF28           	: 1;
  __REG32 ESF29           	: 1;
  __REG32 ESF30           	: 1;
  __REG32 ESF31           	: 1;
} __esmsr_bits;

/* ESM Error Pin Status Register (ESMEPSR) */
typedef struct {
  __REG32 EPSF            	: 1;
  __REG32 		            	:31;
} __esmepsr_bits;

/* ESM Interrupt Offset High Register (ESMIOFFHR) */
typedef struct {
  __REG32 INTOFFH          	: 7;
  __REG32 		            	:25;
} __esmioffhr_bits;

/* ESM Interrupt Offset Low Register (ESMIOFFLR) */
typedef struct {
  __REG32 INTOFFL          	: 7;
  __REG32 		            	:25;
} __esmiofflr_bits;

/* ESM Low-Time Counter Register (ESMLTCR) */
typedef struct {
  __REG32 LTC		          	:16;
  __REG32 		            	:16;
} __esmltcr_bits;

/* ESM Low-Time Counter Preload Register (ESMLTCPR) */
typedef struct {
  __REG32 LTC		          	:16;
  __REG32 		            	:16;
} __esmltcpr_bits;

/* ESM Error Key Register (ESMEKR) */
typedef struct {
  __REG32 EKEY	          	: 4;
  __REG32 		            	:28;
} __esmekr_bits;

/* SCI Global Control Register 0 (SCIGCR0) */
typedef struct {
  __REG32 RESET               : 1;
  __REG32                     :31;
} __scigcr0_bits;

/* SCI Global Control Register 1 (SCIGCR1) */
typedef struct {
  __REG32 COMM_MODE           : 1;
  __REG32 TIMING_MODE         : 1;
  __REG32 PARITY_ENA          : 1;
  __REG32 PARITY              : 1;
  __REG32 STOP                : 1;
  __REG32 CLOCK               : 1;
  __REG32 LIN_MODE            : 1;
  __REG32 SW_nRST             : 1;
  __REG32 SLEEP               : 1;
  __REG32 ADAPT               : 1;
  __REG32 MBUF_MODE           : 1;
  __REG32 CTYPE               : 1;
  __REG32 HGEN_CTRL           : 1;
  __REG32 STOP_EXT_FRAME      : 1;
  __REG32                     : 2;
  __REG32 LOOP_BACK           : 1;
  __REG32 CONT                : 1;
  __REG32                     : 6;
  __REG32 RXENA               : 1;
  __REG32 TXENA               : 1;
  __REG32                     : 6;
} __scigcr1_bits;

/* SCI Global Control Register 2 (SCIGCR2) */
typedef struct {
  __REG32 POWER_DOWN          : 1;
  __REG32                     : 7;
  __REG32 GEN_WU              : 1;
  __REG32                     : 7;
  __REG32 SC                  : 1;
  __REG32 CC                  : 1;
  __REG32                     :14;
} __scigcr2_bits;

/* SCI Set Interrupt Register (SCISETINT) */
typedef struct {
  __REG32 SET_BRKDT_INT       : 1;
  __REG32 SET_WAKEUP_INT      : 1;
  __REG32                     : 2;
  __REG32 SET_TIMEOUT_INT     : 1;
  __REG32                     : 1;
  __REG32 SET_TOAWUS_INT      : 1;
  __REG32 SET_TOA3WUS_INT     : 1;
  __REG32 SET_TX_INT          : 1;
  __REG32 SET_RX_INT          : 1;
  __REG32                     : 3;
  __REG32 SET_ID_INT          : 1;
  __REG32                     : 2;
  __REG32 SET_TX_DMA          : 1;
  __REG32 SET_RX_DMA          : 1;
  __REG32 SET_RX_DMA_ALL      : 1;
  __REG32                     : 5;
  __REG32 SET_PE_INT          : 1;
  __REG32 SET_OE_INT          : 1;
  __REG32 SET_FE_INT          : 1;
  __REG32 SET_NRE_INT         : 1;
  __REG32 SET_ISFE_INT        : 1;
  __REG32 SET_CE_INT          : 1;
  __REG32 SET_PBE_INT         : 1;
  __REG32 SET_BE_INT          : 1;
} __scisetint_bits;

/* SCI Clear Interrupt Register (SCICLEARINT) */
typedef struct {
  __REG32 CLR_BRKDT_INT       : 1;
  __REG32 CLR_WAKEUP_INT      : 1;
  __REG32                     : 2;
  __REG32 CLR_TIMEOUT_INT     : 1;
  __REG32                     : 1;
  __REG32 CLR_TOAWUS_INT      : 1;
  __REG32 CLR_TOA3WUS_INT     : 1;
  __REG32 CLR_TX_INT          : 1;
  __REG32 CLR_RX_INT          : 1;
  __REG32                     : 3;
  __REG32 CLR_ID_INT          : 1;
  __REG32                     : 2;
  __REG32 CLR_TX_DMA          : 1;
  __REG32 CLR_RX_DMA          : 1;
  __REG32 CLR_RX_DMA_ALL      : 1;
  __REG32                     : 5;
  __REG32 CLR_PE_INT          : 1;
  __REG32 CLR_OE_INT          : 1;
  __REG32 CLR_FE_INT          : 1;
  __REG32 CLR_NRE_INT         : 1;
  __REG32 CLR_ISFE_INT        : 1;
  __REG32 CLR_CE_INT          : 1;
  __REG32 CLR_PBE_INT         : 1;
  __REG32 CLR_BE_INT          : 1;
} __sciclearint_bits;

/* SCI Set Interrupt Level Register (SCISETINTLVL) */
typedef struct {
  __REG32 SET_BRKDT_INT_LVL   : 1;
  __REG32 SET_WAKEUP_INT_LVL  : 1;
  __REG32                     : 2;
  __REG32 SET_TIMEOUT_INT_LVL : 1;
  __REG32                     : 1;
  __REG32 SET_TOAWUS_INT_LVL  : 1;
  __REG32 SET_TOA3WUS_INT_LVL : 1;
  __REG32 SET_TX_INT_LVL      : 1;
  __REG32 SET_RX_INT_LVL      : 1;
  __REG32                     : 3;
  __REG32 SET_ID_INT_LVL      : 1;
  __REG32                     : 4;
  __REG32 SET_RX_DMA_ALL_LVL  : 1;
  __REG32                     : 5;
  __REG32 SET_PE_INT_LVL      : 1;
  __REG32 SET_OE_INT_LVL      : 1;
  __REG32 SET_FE_INT_LVL      : 1;
  __REG32 SET_NRE_INT_LVL     : 1;
  __REG32 SET_ISFE_INT_LVL    : 1;
  __REG32 SET_CE_INT_LVL      : 1;
  __REG32 SET_PBE_INT_LVL     : 1;
  __REG32 SET_BE_INT_LVL      : 1;
} __scisetintlvl_bits;

/* SCI Clear Interrupt Level Register (SCICLEARINTLVL) */
typedef struct {
  __REG32 CLR_BRKDT_INT_LVL   : 1;
  __REG32 CLR_WAKEUP_INT_LVL  : 1;
  __REG32                     : 2;
  __REG32 CLR_TIMEOUT_INT_LVL : 1;
  __REG32                     : 1;
  __REG32 CLR_TOAWUS_INT_LVL  : 1;
  __REG32 CLR_TOA3WUS_INT_LVL : 1;
  __REG32 CLR_TX_INT_LVL      : 1;
  __REG32 CLR_RX_INT_LVL      : 1;
  __REG32                     : 3;
  __REG32 CLR_ID_INT_LVL      : 1;
  __REG32                     : 4;
  __REG32 CLR_RX_DMA_ALL_LVL  : 1;
  __REG32                     : 5;
  __REG32 CLR_PE_INT_LVL      : 1;
  __REG32 CLR_OE_INT_LVL      : 1;
  __REG32 CLR_FE_INT_LVL      : 1;
  __REG32 CLR_NRE_INT_LVL     : 1;
  __REG32 CLR_ISFE_INT_LVL    : 1;
  __REG32 CLR_CE_INT_LVL      : 1;
  __REG32 CLR_PBE_INT_LVL     : 1;
  __REG32 CLR_BE_INT_LVL      : 1;
} __sciclearintlvl_bits;

/* SCI Flags Register (SCIFLR) */
typedef struct {
  __REG32 BRKDT               : 1;
  __REG32 WAKEUP              : 1;
  __REG32 IDLE                : 1;
  __REG32 BUSY                : 1;
  __REG32 TIMEOUT             : 1;
  __REG32                     : 1;
  __REG32 TOAWUS              : 1;
  __REG32 TOA3WUS             : 1;
  __REG32 TXRDY               : 1;
  __REG32 RXRDY               : 1;
  __REG32 TXWAKE              : 1;
  __REG32 TX_EMPTY            : 1;
  __REG32 RXWAKE              : 1;
  __REG32 ID_TX_Flag          : 1;
  __REG32 ID_RX_Flag          : 1;
  __REG32                     : 9;
  __REG32 PE                  : 1;
  __REG32 OE                  : 1;
  __REG32 FE                  : 1;
  __REG32 NRE                 : 1;
  __REG32 ISFE                : 1;
  __REG32 CE                  : 1;
  __REG32 PBE                 : 1;
  __REG32 BE                  : 1;
} __sciflr_bits;

/* SCI Interrupt Vector Offset 0 (SCIINTVECT0) */
typedef struct {
  __REG32 INTVECT0            : 5;
  __REG32                     :27;
} __sciintvect0_bits;

/* SCI Interrupt Vector Offset 1 (SCIINTVECT1) */
typedef struct {
  __REG32 INTVECT1            : 5;
  __REG32                     :27;
} __sciintvect1_bits;

/* SCI Format Control Register (SCIFORMAT) */
typedef struct {
  __REG32 CHAR                : 3;
  __REG32                     :13;
  __REG32 LENGTH              : 3;
  __REG32                     :13;
} __sciformat_bits;

/* SCI Baud Rate Selection Register (SCIBRS) */
typedef struct {
  __REG32 PRESCALER_P         :24;
  __REG32 M                   : 4;
  __REG32 U                   : 3;
  __REG32                     : 1;
} __scibrs_bits;

/* Receiver Emulation Data Buffer (SCIED) */
typedef struct {
  __REG32 ED                  : 8;
  __REG32                     :24;
} __scied_bits;

/* Receiver Data Buffer (SCIRD) */
typedef struct {
  __REG32 RD                  : 8;
  __REG32                     :24;
} __scird_bits;

/* Transmit Data Buffer Register (SCITD) */
typedef struct {
  __REG32 TD                  : 8;
  __REG32                     :24;
} __scitd_bits;

/* SCI Pin I/O Control Register 0 (SCIPIO0) */
typedef struct {
  __REG32 CLK_FUNC            : 1;
  __REG32 RX_FUNC             : 1;
  __REG32 TX_FUNC             : 1;
  __REG32                     :29;
} __scipio0_bits;

/* SCI Pin I/O Control Register 1 (SCIPIO1) */
typedef struct {
  __REG32 CLK_DIR             : 1;
  __REG32 RX_DIR              : 1;
  __REG32 TX_DIR              : 1;
  __REG32                     :29;
} __scipio1_bits;

/* SCI Pin I/O Control Register 2 (SCIPIO2) */
typedef struct {
  __REG32 CLK_IN              : 1;
  __REG32 RX_IN               : 1;
  __REG32 TX_IN               : 1;
  __REG32                     :29;
} __scipio2_bits;

/* SCI Pin I/O Control Register 3 (SCIPIO3) */
typedef struct {
  __REG32 CLK_OUT             : 1;
  __REG32 RX_OUT              : 1;
  __REG32 TX_OUT              : 1;
  __REG32                     :29;
} __scipio3_bits;

/* SCI Pin I/O Control Register 4 (SCIPIO4) */
typedef struct {
  __REG32 CLK_SET             : 1;
  __REG32 RX_SET              : 1;
  __REG32 TX_SET              : 1;
  __REG32                     :29;
} __scipio4_bits;

/* SCI Pin I/O Control Register 5 (SCIPIO5) */
typedef struct {
  __REG32 CLK_CLR             : 1;
  __REG32 RX_CLR              : 1;
  __REG32 TX_CLR              : 1;
  __REG32                     :29;
} __scipio5_bits;

/* SCI Pin I/O Control Register 6 (SCIPIO6) */
typedef struct {
  __REG32 CLK_PDR             : 1;
  __REG32 RX_PDR              : 1;
  __REG32 TX_PDR              : 1;
  __REG32                     :29;
} __scipio6_bits;

/* SCI Pin I/O Control Register 7 (SCIPIO7) */
typedef struct {
  __REG32 CLK_PD              : 1;
  __REG32 RX_PD               : 1;
  __REG32 TX_PD               : 1;
  __REG32                     :29;
} __scipio7_bits;

/* SCI Pin I/O Control Register 8 (SCIPIO8) */
typedef struct {
  __REG32 CLK_PSL             : 1;
  __REG32 RX_PSL              : 1;
  __REG32 TX_PSL              : 1;
  __REG32                     :29;
} __scipio8_bits;

/* LIN Compare Register (LINCOMPARE) */
typedef struct {
  __REG32 SBREAK              : 3;
  __REG32                     : 5;
  __REG32 SDEL                : 2;
  __REG32                     :22;
} __lincompare_bits;

/* LIN Receive Buffer 0 Register (LINRD0) */
typedef struct {
  __REG32 RD3                 : 8;
  __REG32 RD2                 : 8;
  __REG32 RD1                 : 8;
  __REG32 RD0                 : 8;
} __linrd0_bits;

/* LIN Receive Buffer 1 Register (LINRD1) */
typedef struct {
  __REG32 RD7                 : 8;
  __REG32 RD6                 : 8;
  __REG32 RD5                 : 8;
  __REG32 RD4                 : 8;
} __linrd1_bits;

/* LIN Mask Register (LINMASK) */
typedef struct {
  __REG32 TX_ID_MASK          : 8;
  __REG32                     : 8;
  __REG32 RX_ID_MASK          : 8;
  __REG32                     : 8;
} __linmask_bits;

/* LIN Identification Register (LINID) */
typedef struct {
  __REG32 ID_BYTE             : 8;
  __REG32 ID_SlaveTask_BYTE   : 8;
  __REG32 RECEIVED_ID         : 8;
  __REG32                     : 8;
} __linid_bits;

/* LIN Transmit Buffer 0 Register (LINTD0) */
typedef struct {
  __REG32 TD3                 : 8;
  __REG32 TD2                 : 8;
  __REG32 TD1                 : 8;
  __REG32 TD0                 : 8;
} __lintd0_bits;

/* LIN Transmit Buffer 1 Register (LINTD1) */
typedef struct {
  __REG32 TD7                 : 8;
  __REG32 TD6                 : 8;
  __REG32 TD5                 : 8;
  __REG32 TD4                 : 8;
} __lintd1_bits;

/* Maximum Baud Rate Selection Register (MBRS) */
typedef struct {
  __REG32 MBR                 :13;
  __REG32                     :19;
} __linmbrs_bits;

/* Input/Output Error Enable Register (IODFTCTRL) */
typedef struct {
  __REG32 RXPENA              : 1;
  __REG32 LPBENA              : 1;
  __REG32                     : 6;
  __REG32 IODFTENA            : 4;
  __REG32                     : 4;
  __REG32 TX_SHIFT            : 3;
  __REG32 PIN_SAMPLE_MASK     : 2;
  __REG32                     : 3;
  __REG32 BRKD_TENA           : 1;
  __REG32 PEN                 : 1;
  __REG32 FEN                 : 1;
  __REG32                     : 1;
  __REG32 ISFE                : 1;
  __REG32 CEN                 : 1;
  __REG32 PBEN                : 1;
  __REG32 BEN                 : 1;
} __iodftctrl_bits;

/* GIO Global Control Register (GIOGCR0) */
typedef struct {
  __REG32 RESET               : 1;
  __REG32                     :31;
} __giogcr0_bits;

/* GIO Interrupt Detect Register (GIOINTDET) */
typedef struct {
  __REG32 GIOINTDET00         : 1;
  __REG32 GIOINTDET01         : 1;
  __REG32 GIOINTDET02         : 1;
  __REG32 GIOINTDET03         : 1;
  __REG32 GIOINTDET04         : 1;
  __REG32 GIOINTDET05         : 1;
  __REG32 GIOINTDET06         : 1;
  __REG32 GIOINTDET07         : 1;
  __REG32 GIOINTDET10         : 1;
  __REG32 GIOINTDET11         : 1;
  __REG32 GIOINTDET12         : 1;
  __REG32 GIOINTDET13         : 1;
  __REG32 GIOINTDET14         : 1;
  __REG32 GIOINTDET15         : 1;
  __REG32 GIOINTDET16         : 1;
  __REG32 GIOINTDET17         : 1;
  __REG32 GIOINTDET20         : 1;
  __REG32 GIOINTDET21         : 1;
  __REG32 GIOINTDET22         : 1;
  __REG32 GIOINTDET23         : 1;
  __REG32 GIOINTDET24         : 1;
  __REG32 GIOINTDET25         : 1;
  __REG32 GIOINTDET26         : 1;
  __REG32 GIOINTDET27         : 1;
  __REG32 GIOINTDET30         : 1;
  __REG32 GIOINTDET31         : 1;
  __REG32 GIOINTDET32         : 1;
  __REG32 GIOINTDET33         : 1;
  __REG32 GIOINTDET34         : 1;
  __REG32 GIOINTDET35         : 1;
  __REG32 GIOINTDET36         : 1;
  __REG32 GIOINTDET37         : 1;
} __giointdet_bits;

/* GIO Interrupt Polarity Register (GIOPOL) */
typedef struct {
  __REG32 GIOPOL00            : 1;
  __REG32 GIOPOL01            : 1;
  __REG32 GIOPOL02            : 1;
  __REG32 GIOPOL03            : 1;
  __REG32 GIOPOL04            : 1;
  __REG32 GIOPOL05            : 1;
  __REG32 GIOPOL06            : 1;
  __REG32 GIOPOL07            : 1;
  __REG32 GIOPOL10            : 1;
  __REG32 GIOPOL11            : 1;
  __REG32 GIOPOL12            : 1;
  __REG32 GIOPOL13            : 1;
  __REG32 GIOPOL14            : 1;
  __REG32 GIOPOL15            : 1;
  __REG32 GIOPOL16            : 1;
  __REG32 GIOPOL17            : 1;
  __REG32 GIOPOL20            : 1;
  __REG32 GIOPOL21            : 1;
  __REG32 GIOPOL22            : 1;
  __REG32 GIOPOL23            : 1;
  __REG32 GIOPOL24            : 1;
  __REG32 GIOPOL25            : 1;
  __REG32 GIOPOL26            : 1;
  __REG32 GIOPOL27            : 1;
  __REG32 GIOPOL30            : 1;
  __REG32 GIOPOL31            : 1;
  __REG32 GIOPOL32            : 1;
  __REG32 GIOPOL33            : 1;
  __REG32 GIOPOL34            : 1;
  __REG32 GIOPOL35            : 1;
  __REG32 GIOPOL36            : 1;
  __REG32 GIOPOL37            : 1;
} __giopol_bits;

/* GIO Interrupt Enable Register (GIOENASET) */
typedef struct {
  __REG32 GIOENASET00         : 1;
  __REG32 GIOENASET01         : 1;
  __REG32 GIOENASET02         : 1;
  __REG32 GIOENASET03         : 1;
  __REG32 GIOENASET04         : 1;
  __REG32 GIOENASET05         : 1;
  __REG32 GIOENASET06         : 1;
  __REG32 GIOENASET07         : 1;
  __REG32 GIOENASET10         : 1;
  __REG32 GIOENASET11         : 1;
  __REG32 GIOENASET12         : 1;
  __REG32 GIOENASET13         : 1;
  __REG32 GIOENASET14         : 1;
  __REG32 GIOENASET15         : 1;
  __REG32 GIOENASET16         : 1;
  __REG32 GIOENASET17         : 1;
  __REG32 GIOENASET20         : 1;
  __REG32 GIOENASET21         : 1;
  __REG32 GIOENASET22         : 1;
  __REG32 GIOENASET23         : 1;
  __REG32 GIOENASET24         : 1;
  __REG32 GIOENASET25         : 1;
  __REG32 GIOENASET26         : 1;
  __REG32 GIOENASET27         : 1;
  __REG32 GIOENASET30         : 1;
  __REG32 GIOENASET31         : 1;
  __REG32 GIOENASET32         : 1;
  __REG32 GIOENASET33         : 1;
  __REG32 GIOENASET34         : 1;
  __REG32 GIOENASET35         : 1;
  __REG32 GIOENASET36         : 1;
  __REG32 GIOENASET37         : 1;
} __gioenaset_bits;

/* GIO Interrupt Enable Register (GIOENACLR) */
typedef struct {
  __REG32 GIOENACLR00         : 1;
  __REG32 GIOENACLR01         : 1;
  __REG32 GIOENACLR02         : 1;
  __REG32 GIOENACLR03         : 1;
  __REG32 GIOENACLR04         : 1;
  __REG32 GIOENACLR05         : 1;
  __REG32 GIOENACLR06         : 1;
  __REG32 GIOENACLR07         : 1;
  __REG32 GIOENACLR10         : 1;
  __REG32 GIOENACLR11         : 1;
  __REG32 GIOENACLR12         : 1;
  __REG32 GIOENACLR13         : 1;
  __REG32 GIOENACLR14         : 1;
  __REG32 GIOENACLR15         : 1;
  __REG32 GIOENACLR16         : 1;
  __REG32 GIOENACLR17         : 1;
  __REG32 GIOENACLR20         : 1;
  __REG32 GIOENACLR21         : 1;
  __REG32 GIOENACLR22         : 1;
  __REG32 GIOENACLR23         : 1;
  __REG32 GIOENACLR24         : 1;
  __REG32 GIOENACLR25         : 1;
  __REG32 GIOENACLR26         : 1;
  __REG32 GIOENACLR27         : 1;
  __REG32 GIOENACLR30         : 1;
  __REG32 GIOENACLR31         : 1;
  __REG32 GIOENACLR32         : 1;
  __REG32 GIOENACLR33         : 1;
  __REG32 GIOENACLR34         : 1;
  __REG32 GIOENACLR35         : 1;
  __REG32 GIOENACLR36         : 1;
  __REG32 GIOENACLR37         : 1;
} __gioenaclr_bits;

/* GIO Interrupt Priority Register (GIOLVSLSET) */
typedef struct {
  __REG32 GIOLVLSET00         : 1;
  __REG32 GIOLVLSET01         : 1;
  __REG32 GIOLVLSET02         : 1;
  __REG32 GIOLVLSET03         : 1;
  __REG32 GIOLVLSET04         : 1;
  __REG32 GIOLVLSET05         : 1;
  __REG32 GIOLVLSET06         : 1;
  __REG32 GIOLVLSET07         : 1;
  __REG32 GIOLVLSET10         : 1;
  __REG32 GIOLVLSET11         : 1;
  __REG32 GIOLVLSET12         : 1;
  __REG32 GIOLVLSET13         : 1;
  __REG32 GIOLVLSET14         : 1;
  __REG32 GIOLVLSET15         : 1;
  __REG32 GIOLVLSET16         : 1;
  __REG32 GIOLVLSET17         : 1;
  __REG32 GIOLVLSET20         : 1;
  __REG32 GIOLVLSET21         : 1;
  __REG32 GIOLVLSET22         : 1;
  __REG32 GIOLVLSET23         : 1;
  __REG32 GIOLVLSET24         : 1;
  __REG32 GIOLVLSET25         : 1;
  __REG32 GIOLVLSET26         : 1;
  __REG32 GIOLVLSET27         : 1;
  __REG32 GIOLVLSET30         : 1;
  __REG32 GIOLVLSET31         : 1;
  __REG32 GIOLVLSET32         : 1;
  __REG32 GIOLVLSET33         : 1;
  __REG32 GIOLVLSET34         : 1;
  __REG32 GIOLVLSET35         : 1;
  __REG32 GIOLVLSET36         : 1;
  __REG32 GIOLVLSET37         : 1;
} __giolvlset_bits;

/* GIO Interrupt Priority Register (GIOLVLCLR) */
typedef struct {
  __REG32 GIOLVLCLR0          : 1;
  __REG32 GIOLVLCLR1          : 1;
  __REG32 GIOLVLCLR2          : 1;
  __REG32 GIOLVLCLR3          : 1;
  __REG32 GIOLVLCLR4          : 1;
  __REG32 GIOLVLCLR5          : 1;
  __REG32 GIOLVLCLR6          : 1;
  __REG32 GIOLVLCLR7          : 1;
  __REG32                     :24;
} __giolvlclr_bits;

/* GIO Interrupt Flag Register (GIOFLG) */
typedef struct {
  __REG32 GIOFLG00             : 1;
  __REG32 GIOFLG01             : 1;
  __REG32 GIOFLG02             : 1;
  __REG32 GIOFLG03             : 1;
  __REG32 GIOFLG04             : 1;
  __REG32 GIOFLG05             : 1;
  __REG32 GIOFLG06             : 1;
  __REG32 GIOFLG07             : 1;
  __REG32 GIOFLG10             : 1;
  __REG32 GIOFLG11             : 1;
  __REG32 GIOFLG12             : 1;
  __REG32 GIOFLG13             : 1;
  __REG32 GIOFLG14             : 1;
  __REG32 GIOFLG15             : 1;
  __REG32 GIOFLG16             : 1;
  __REG32 GIOFLG17             : 1;
  __REG32 GIOFLG20             : 1;
  __REG32 GIOFLG21             : 1;
  __REG32 GIOFLG22             : 1;
  __REG32 GIOFLG23             : 1;
  __REG32 GIOFLG24             : 1;
  __REG32 GIOFLG25             : 1;
  __REG32 GIOFLG26             : 1;
  __REG32 GIOFLG27             : 1;
  __REG32 GIOFLG30             : 1;
  __REG32 GIOFLG31             : 1;
  __REG32 GIOFLG32             : 1;
  __REG32 GIOFLG33             : 1;
  __REG32 GIOFLG34             : 1;
  __REG32 GIOFLG35             : 1;
  __REG32 GIOFLG36             : 1;
  __REG32 GIOFLG37             : 1;
} __gioflg_bits;

/* GIO Offset A Register (GIOOFFA) */
typedef struct {
  __REG32 GIOOFFA0            : 1;
  __REG32 GIOOFFA1            : 1;
  __REG32 GIOOFFA2            : 1;
  __REG32 GIOOFFA3            : 1;
  __REG32 GIOOFFA4            : 1;
  __REG32 GIOOFFA5            : 1;
  __REG32                     :26;
} __giooffa_bits;

/* GIO Offset B Register (GIOOFFB) */
typedef struct {
  __REG32 GIOOFFB0            : 1;
  __REG32 GIOOFFB1            : 1;
  __REG32 GIOOFFB2            : 1;
  __REG32 GIOOFFB3            : 1;
  __REG32 GIOOFFB4            : 1;
  __REG32 GIOOFFB5            : 1;
  __REG32                     :26;
} __giooffb_bits;

/* GIO Emulation A Register (GIOEMUA) */
typedef struct {
  __REG32 GIOEMUA0            : 1;
  __REG32 GIOEMUA1            : 1;
  __REG32 GIOEMUA2            : 1;
  __REG32 GIOEMUA3            : 1;
  __REG32 GIOEMUA4            : 1;
  __REG32 GIOEMUA5            : 1;
  __REG32                     :26;
} __gioemua_bits;

/* GIO Emulation B Register (GIOEMUB) */
typedef struct {
  __REG32 GIOEMUB0            : 1;
  __REG32 GIOEMUB1            : 1;
  __REG32 GIOEMUB2            : 1;
  __REG32 GIOEMUB3            : 1;
  __REG32 GIOEMUB4            : 1;
  __REG32 GIOEMUB5            : 1;
  __REG32                     :26;
} __gioemub_bits;

/* GIO Data Direction Registers [A-H][7:0] (GIODIR[A-H][7:0]) */
typedef struct {
  __REG32 GIODIR0             : 1;
  __REG32 GIODIR1             : 1;
  __REG32 GIODIR2             : 1;
  __REG32 GIODIR3             : 1;
  __REG32 GIODIR4             : 1;
  __REG32 GIODIR5             : 1;
  __REG32 GIODIR6             : 1;
  __REG32 GIODIR7             : 1;
  __REG32                     :24;
} __giodir_bits;

/* GIO Data Input Registers [A-B][7:0] (GIODIN[A-B][7:0]) */
typedef struct {
  __REG32 GIODIN0             : 1;
  __REG32 GIODIN1             : 1;
  __REG32 GIODIN2             : 1;
  __REG32 GIODIN3             : 1;
  __REG32 GIODIN4             : 1;
  __REG32 GIODIN5             : 1;
  __REG32 GIODIN6             : 1;
  __REG32 GIODIN7             : 1;
  __REG32                     :24;
} __giodin_bits;

/* GIO Data Output Registers [A-B][7:0] (GIODOUT[A-B][7:0]) */
typedef struct {
  __REG32 GIODOUT0            : 1;
  __REG32 GIODOUT1            : 1;
  __REG32 GIODOUT2            : 1;
  __REG32 GIODOUT3            : 1;
  __REG32 GIODOUT4            : 1;
  __REG32 GIODOUT5            : 1;
  __REG32 GIODOUT6            : 1;
  __REG32 GIODOUT7            : 1;
  __REG32                     :24;
} __giodout_bits;

/* GIO Data Set Registers [A-B][7:0] (GIODSET[A-B][7:0]) */
typedef struct {
  __REG32 GIODSET0            : 1;
  __REG32 GIODSET1            : 1;
  __REG32 GIODSET2            : 1;
  __REG32 GIODSET3            : 1;
  __REG32 GIODSET4            : 1;
  __REG32 GIODSET5            : 1;
  __REG32 GIODSET6            : 1;
  __REG32 GIODSET7            : 1;
  __REG32                     :24;
} __gioset_bits;

/* GIO Data Clear Registers [A-B][7:0] (GIODCLR[A-B][7:0]) */
typedef struct {
  __REG32 GIODCLR0            : 1;
  __REG32 GIODCLR1            : 1;
  __REG32 GIODCLR2            : 1;
  __REG32 GIODCLR3            : 1;
  __REG32 GIODCLR4            : 1;
  __REG32 GIODCLR5            : 1;
  __REG32 GIODCLR6            : 1;
  __REG32 GIODCLR7            : 1;
  __REG32                     :24;
} __gioclr_bits;

/* GIO Open Drain Register [A-B][7:0] (GIOPDR[A-B][7:0]) */
typedef struct {
  __REG32 GIOPDR0             : 1;
  __REG32 GIOPDR1             : 1;
  __REG32 GIOPDR2             : 1;
  __REG32 GIOPDR3             : 1;
  __REG32 GIOPDR4             : 1;
  __REG32 GIOPDR5             : 1;
  __REG32 GIOPDR6             : 1;
  __REG32 GIOPDR7             : 1;
  __REG32                     :24;
} __giopdr_bits;

/* GIO Pull Disable Registers [A-B][7:0] (GIOPULDIS[A-B][7:0]) */
typedef struct {
  __REG32 GIOPULDIS0          : 1;
  __REG32 GIOPULDIS1          : 1;
  __REG32 GIOPULDIS2          : 1;
  __REG32 GIOPULDIS3          : 1;
  __REG32 GIOPULDIS4          : 1;
  __REG32 GIOPULDIS5          : 1;
  __REG32 GIOPULDIS6          : 1;
  __REG32 GIOPULDIS7          : 1;
  __REG32                     :24;
} __giopuldis_bits;

/* GIO Pull Select Register [A-B][7:0] (GIOPSL[A-B][7:0]) */
typedef struct {
  __REG32 GIOPSL0             : 1;
  __REG32 GIOPSL1             : 1;
  __REG32 GIOPSL2             : 1;
  __REG32 GIOPSL3             : 1;
  __REG32 GIOPSL4             : 1;
  __REG32 GIOPSL5             : 1;
  __REG32 GIOPSL6             : 1;
  __REG32 GIOPSL7             : 1;
  __REG32                     :24;
} __giopsl_bits;


#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler *********************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                  0xE000E004,__READ       ,__nvic_bits);
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
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,               0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,              0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,              0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(ACTIVE0,               0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,               0xE000E304,__READ       ,__active1_bits);
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
__IO_REG32_BIT(CPUIDBR,               0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,                  0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,                 0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,                 0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,                 0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                  0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                  0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                  0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,                 0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                  0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,                  0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** M3VIM
 **
 ***************************************************************************/
__IO_REG32_BIT(INTISRIVEC,            0xFFFFFE00,__READ_WRITE	,__intisrivec_bits);
__IO_REG32_BIT(INTNMIIVEC,            0xFFFFFE04,__READ_WRITE	,__intnmiivec_bits);
__IO_REG32_BIT(NESTCTRL,            	0xFFFFFE08,__READ_WRITE	,__nestctrl_bits);
__IO_REG32_BIT(NESTSTAT,            	0xFFFFFE0C,__READ_WRITE	,__neststat_bits);
__IO_REG32_BIT(NMIPR0,            		0xFFFFFE10,__READ_WRITE	,__nmipr0_bits);
__IO_REG32_BIT(NMIPR1,            		0xFFFFFE14,__READ_WRITE	,__nmipr1_bits);
__IO_REG32_BIT(INTREQ0,            		0xFFFFFE20,__READ_WRITE	,__intreq0_bits);
__IO_REG32_BIT(INTREQ1,            		0xFFFFFE24,__READ_WRITE	,__intreq1_bits);
__IO_REG32_BIT(REQMASKSET0,           0xFFFFFE30,__READ_WRITE	,__reqmaskset0_bits);
__IO_REG32_BIT(REQMASKSET1,           0xFFFFFE34,__READ_WRITE	,__reqmaskset1_bits);
__IO_REG32_BIT(REQMASKCLR0,           0xFFFFFE40,__READ_WRITE	,__reqmaskclr0_bits);
__IO_REG32_BIT(REQMASKCLR1,           0xFFFFFE44,__READ_WRITE	,__reqmaskclr1_bits);
__IO_REG32_BIT(WAKEMASKSET0,          0xFFFFFE50,__READ_WRITE	,__wakemaskset0_bits);
__IO_REG32_BIT(WAKEMASKSET1,          0xFFFFFE54,__READ_WRITE	,__wakemaskset1_bits);
__IO_REG32_BIT(WAKEMASKCLR0,          0xFFFFFE60,__READ_WRITE	,__wakemaskclr0_bits);
__IO_REG32_BIT(WAKEMASKCLR1,          0xFFFFFE64,__READ_WRITE	,__wakemaskclr1_bits);
__IO_REG32_BIT(CAPEVT,            		0xFFFFFE78,__READ_WRITE	,__capevt_bits);
__IO_REG32_BIT(CHANCTRL0,            	0xFFFFFE80,__READ_WRITE	,__chanctrl0_bits);
__IO_REG32_BIT(CHANCTRL1,            	0xFFFFFE84,__READ_WRITE	,__chanctrl1_bits);
__IO_REG32_BIT(CHANCTRL2,            	0xFFFFFE88,__READ_WRITE	,__chanctrl2_bits);
__IO_REG32_BIT(CHANCTRL3,            	0xFFFFFE8C,__READ_WRITE	,__chanctrl3_bits);
__IO_REG32_BIT(CHANCTRL4,            	0xFFFFFE90,__READ_WRITE	,__chanctrl4_bits);
__IO_REG32_BIT(CHANCTRL5,            	0xFFFFFE94,__READ_WRITE	,__chanctrl5_bits);
__IO_REG32_BIT(CHANCTRL6,            	0xFFFFFE98,__READ_WRITE	,__chanctrl6_bits);
__IO_REG32_BIT(CHANCTRL7,            	0xFFFFFE9C,__READ_WRITE	,__chanctrl7_bits);
__IO_REG32_BIT(CHANCTRL8,            	0xFFFFFEA0,__READ_WRITE	,__chanctrl8_bits);
__IO_REG32_BIT(CHANCTRL9,            	0xFFFFFEA4,__READ_WRITE	,__chanctrl9_bits);
__IO_REG32_BIT(CHANCTRL10,           	0xFFFFFEA8,__READ_WRITE	,__chanctrl10_bits);
__IO_REG32_BIT(CHANCTRL11,           	0xFFFFFEAC,__READ_WRITE	,__chanctrl11_bits);

/***************************************************************************
 **
 ** SYS1 (Primary System Control)
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSPC1,            0xFFFFFF00,__READ_WRITE ,__syspc1_bits);
__IO_REG32_BIT(SYSPC2,            0xFFFFFF04,__READ_WRITE ,__syspc2_bits);
__IO_REG32_BIT(SYSPC3,            0xFFFFFF08,__READ       ,__syspc3_bits);
__IO_REG32_BIT(SYSPC4,            0xFFFFFF0C,__READ_WRITE ,__syspc4_bits);
__IO_REG32_BIT(SYSPC5,            0xFFFFFF10,__READ_WRITE ,__syspc5_bits);
__IO_REG32_BIT(SYSPC6,            0xFFFFFF14,__READ_WRITE ,__syspc6_bits);
__IO_REG32_BIT(SYSPC7,            0xFFFFFF18,__READ_WRITE ,__syspc7_bits);
__IO_REG32_BIT(SYSPC8,            0xFFFFFF1C,__READ_WRITE ,__syspc8_bits);
__IO_REG32_BIT(SYSPC9,            0xFFFFFF20,__READ_WRITE ,__syspc9_bits);
__IO_REG32_BIT(SSWPLL1,           0xFFFFFF24,__READ_WRITE ,__sswpll1_bits);
__IO_REG32(		 SSWPLL2,           0xFFFFFF28,__READ				);
__IO_REG32(		 SSWPLL3,           0xFFFFFF2C,__READ				);
__IO_REG32_BIT(CSDIS,          		0xFFFFFF30,__READ_WRITE ,__csdis_bits);
__IO_REG32_BIT(CSDISSET,          0xFFFFFF34,__READ_WRITE ,__csdisset_bits);
__IO_REG32_BIT(CSDISCLR,          0xFFFFFF38,__READ_WRITE ,__csdisclr_bits);
__IO_REG32_BIT(CDDIS,             0xFFFFFF3C,__READ_WRITE ,__cddis_bits);
__IO_REG32_BIT(CDDISSET,          0xFFFFFF40,__READ_WRITE ,__cddisset_bits);
__IO_REG32_BIT(CDDISCLR,          0xFFFFFF44,__READ_WRITE ,__cddisclr_bits);
__IO_REG32_BIT(GHVSRC,            0xFFFFFF48,__READ_WRITE ,__ghvsrc_bits);
__IO_REG32_BIT(VCLKASRC,          0xFFFFFF4C,__READ_WRITE ,__vclkasrc_bits);
__IO_REG32_BIT(RCLKSRC,           0xFFFFFF50,__READ_WRITE ,__rclksrc_bits);
__IO_REG32_BIT(CSVSTAT,           0xFFFFFF54,__READ       ,__csvstat_bits);
__IO_REG32_BIT(MSTGCR,            0xFFFFFF58,__READ_WRITE ,__mstgcr_bits);
__IO_REG32_BIT(MINITGCR,          0xFFFFFF5C,__READ_WRITE ,__minitgcr_bits);
__IO_REG32_BIT(MSIENA,            0xFFFFFF60,__READ_WRITE ,__msiena_bits);
__IO_REG32_BIT(MSTFAIL,           0xFFFFFF64,__READ_WRITE ,__mstfail_bits);
__IO_REG32_BIT(MSTCGSTAT,         0xFFFFFF68,__READ_WRITE ,__mstcgstat_bits);
__IO_REG32_BIT(MINISTAT,          0xFFFFFF6C,__READ_WRITE ,__ministat_bits);
__IO_REG32_BIT(PLLCTL1,           0xFFFFFF70,__READ_WRITE ,__pllctl1_bits);
__IO_REG32_BIT(PLLCTL2,           0xFFFFFF74,__READ_WRITE ,__pllctl2_bits);
__IO_REG32(		 DIEIDL,            0xFFFFFF7C,__READ       );
__IO_REG32(		 DIEIDH,            0xFFFFFF80,__READ       );
__IO_REG32_BIT(VRCTL,         		0xFFFFFF84,__READ_WRITE ,__vrctl_bits);
__IO_REG32_BIT(LPOMONCTL,         0xFFFFFF88,__READ_WRITE ,__lpomonctl_bits);
__IO_REG32_BIT(CLKTEST,           0xFFFFFF8C,__READ_WRITE ,__clktest_bits);
__IO_REG32_BIT(GPREG1,          	0xFFFFFFA0,__READ_WRITE	,__gpreg1_bits);
__IO_REG32_BIT(IMPFASTS,          0xFFFFFFA8,__READ       ,__impfasts_bits);
__IO_REG32(    IMPFTADD,          0xFFFFFFAC,__READ       );
__IO_REG32_BIT(SSIR1,             0xFFFFFFB0,__READ_WRITE ,__ssir1_bits);
__IO_REG32_BIT(SSIR2,             0xFFFFFFB4,__READ_WRITE ,__ssir2_bits);
__IO_REG32_BIT(SSIR3,             0xFFFFFFB8,__READ_WRITE ,__ssir3_bits);
__IO_REG32_BIT(SSIR4,             0xFFFFFFBC,__READ_WRITE ,__ssir4_bits);
__IO_REG32_BIT(RAMGCR,            0xFFFFFFC0,__READ_WRITE ,__ramgcr_bits);
__IO_REG32_BIT(BMMCR1,            0xFFFFFFC4,__READ_WRITE ,__bmmcr1_bits);
__IO_REG32_BIT(BMMCR2,            0xFFFFFFC8,__READ_WRITE ,__bmmcr2_bits);
__IO_REG32_BIT(MMUGCR,            0xFFFFFFCC,__READ_WRITE ,__mmugcr_bits);
__IO_REG32_BIT(CLKCNTL,          	0xFFFFFFD0,__READ_WRITE ,__clkcntl_bits);
__IO_REG32_BIT(ECPCNTRL,          0xFFFFFFD4,__READ_WRITE ,__ecpcntl_bits);
__IO_REG32_BIT(DEVCR1,          	0xFFFFFFDC,__READ_WRITE ,__devcr1_bits);
__IO_REG32_BIT(SYSECR,            0xFFFFFFE0,__READ_WRITE ,__sysecr_bits);
__IO_REG32_BIT(SYSESR,            0xFFFFFFE4,__READ_WRITE ,__sysesr_bits);
__IO_REG32_BIT(GLBSTAT,           0xFFFFFFEC,__READ_WRITE ,__glbstat_bits);
__IO_REG32_BIT(DEVID,             0xFFFFFFF0,__READ       ,__devid_bits);
__IO_REG32_BIT(SSIVEC,            0xFFFFFFF4,__READ       ,__ssivec_bits);
__IO_REG32_BIT(SSIF,              0xFFFFFFF8,__READ_WRITE ,__ssif_bits);
__IO_REG32_BIT(SSIR1_MIRROR,      0xFFFFFFFC,__READ_WRITE ,__ssir1_bits);

/***************************************************************************
 **
 ** PCR (Peripheral Central Resource)
 **
 ***************************************************************************/
__IO_REG32_BIT(PMPROTSET0,        0xFFFFE000,__READ_WRITE ,__pmprotset0_bits);
__IO_REG32_BIT(PMPROTSET1,        0xFFFFE004,__READ_WRITE ,__pmprotset1_bits);
__IO_REG32_BIT(PMPROTCLR0,        0xFFFFE010,__READ_WRITE ,__pmprotclr0_bits);
__IO_REG32_BIT(PMPROTCLR1,        0xFFFFE014,__READ_WRITE ,__pmprotclr1_bits);
__IO_REG32_BIT(PPROTSET0,         0xFFFFE020,__READ_WRITE ,__pprotset0_bits);
__IO_REG32_BIT(PPROTSET1,         0xFFFFE024,__READ_WRITE ,__pprotset1_bits);
__IO_REG32_BIT(PPROTSET2,         0xFFFFE028,__READ_WRITE ,__pprotset2_bits);
__IO_REG32_BIT(PPROTSET3,         0xFFFFE02C,__READ_WRITE ,__pprotset3_bits);
__IO_REG32_BIT(PPROTCLR0,         0xFFFFE040,__READ_WRITE ,__pprotclr0_bits);
__IO_REG32_BIT(PPROTCLR1,         0xFFFFE044,__READ_WRITE ,__pprotclr1_bits);
__IO_REG32_BIT(PPROTCLR2,         0xFFFFE048,__READ_WRITE ,__pprotclr2_bits);
__IO_REG32_BIT(PPROTCLR3,         0xFFFFE04C,__READ_WRITE ,__pprotclr3_bits);
__IO_REG32_BIT(PCSPWRDWNSET0,     0xFFFFE060,__READ_WRITE ,__pcspwrdwnset0_bits);
__IO_REG32_BIT(PCSPWRDWNSET1,     0xFFFFE064,__READ_WRITE ,__pcspwrdwnset1_bits);
__IO_REG32_BIT(PCSPWRDWNCLR0,     0xFFFFE070,__READ_WRITE ,__pcspwrdwnclr0_bits);
__IO_REG32_BIT(PCSPWRDWNCLR1,     0xFFFFE074,__READ_WRITE ,__pcspwrdwnclr1_bits);
__IO_REG32_BIT(PSPWRDWNSET0,      0xFFFFE080,__READ_WRITE ,__pspwrdwnset0_bits);
__IO_REG32_BIT(PSPWRDWNSET1,      0xFFFFE084,__READ_WRITE ,__pspwrdwnset1_bits);
__IO_REG32_BIT(PSPWRDWNSET2,      0xFFFFE088,__READ_WRITE ,__pspwrdwnset2_bits);
__IO_REG32_BIT(PSPWRDWNSET3,      0xFFFFE08C,__READ_WRITE ,__pspwrdwnset3_bits);
__IO_REG32_BIT(PSPWRDWNCLR0,      0xFFFFE0A0,__READ_WRITE ,__pspwrdwnclr0_bits);
__IO_REG32_BIT(PSPWRDWNCLR1,      0xFFFFE0A4,__READ_WRITE ,__pspwrdwnclr1_bits);
__IO_REG32_BIT(PSPWRDWNCLR2,      0xFFFFE0A8,__READ_WRITE ,__pspwrdwnclr2_bits);
__IO_REG32_BIT(PSPWRDWNCLR3,      0xFFFFE0AC,__READ_WRITE ,__pspwrdwnclr3_bits);

/***************************************************************************
 **
 ** WAKEUPC (Wakeup Control)
 **
 ***************************************************************************/
__IO_REG32_BIT(RSTCR,      				0xFFFFED00,__READ_WRITE ,__rstcr_bits);
__IO_REG32_BIT(EXTWAKENR,      		0xFFFFED04,__READ_WRITE ,__extwakenr_bits);
__IO_REG32_BIT(EXTWAKLVR,      		0xFFFFED08,__READ_WRITE ,__extwaklvr_bits);
__IO_REG32_BIT(EXTWAKESR,      		0xFFFFED0C,__READ_WRITE ,__extwakesr_bits);
__IO_REG32_BIT(HIBXSTATR,      		0xFFFFED10,__READ_WRITE ,__hibxstatr_bits);

/***************************************************************************
 **
 ** SYS2 (Secondary System Control)
 **
 ***************************************************************************/
__IO_REG32_BIT(STCLKDIV,      		0xFFFFE108,__READ_WRITE ,__stclkdiv_bits);

/***************************************************************************
 **
 ** eSRAM
 **
 ***************************************************************************/
__IO_REG16_BIT(RAMCTRL,      			0xFFFFF900,__READ_WRITE ,__ramctrl_bits);
__IO_REG16(		 RAMTHRESHOLD,      0xFFFFF904,__READ_WRITE );
__IO_REG16(		 RAMOCCUR,      		0xFFFFF908,__READ_WRITE );
__IO_REG16_BIT(RAMINTCTRL,      	0xFFFFF90C,__READ_WRITE ,__ramintctrl_bits);
__IO_REG16_BIT(RAMERRSTATUS,      0xFFFFF910,__READ_WRITE ,__ramerrstatus_bits);
__IO_REG16_BIT(RAMMSERRADDR,      0xFFFFF914,__READ				,__rammserraddr_bits);
__IO_REG16_BIT(RAMSERRPOSITION,   0xFFFFF918,__READ				,__ramserrposition_bits);
__IO_REG16_BIT(RAMDERRADDR,      	0xFFFFF91C,__READ_WRITE ,__ramderraddr_bits);
__IO_REG16_BIT(RAMCTRL2,      		0xFFFFF920,__READ_WRITE ,__ramctrl2_bits);

/***************************************************************************
 **
 ** Flash memory
 **
 ***************************************************************************/
__IO_REG32_BIT(FRDCNTL,           0xFFF87000,__READ_WRITE ,__ftudcntl_bits);
__IO_REG32_BIT(FSPRD,             0xFFF87004,__READ_WRITE ,__fsprd_bits);
__IO_REG32_BIT(FEDACCTRL1,        0xFFF87008,__READ_WRITE ,__fedacctrl1_bits);
__IO_REG32_BIT(FEDACCTRL2,        0xFFF8700C,__READ_WRITE ,__fedacctrl2_bits);
__IO_REG32_BIT(FCOR_ERR_CNT,      0xFFF87010,__READ_WRITE ,__fcor_err_cnt_bits);
__IO_REG32(    FCOR_ERR_ADD,      0xFFF87014,__READ       );
__IO_REG32_BIT(FCOR_ERR_POS,      0xFFF87018,__READ_WRITE ,__fcor_err_pos_bits);
__IO_REG32_BIT(FEDACSTATUS,       0xFFF8701C,__READ_WRITE ,__fedacstatus_bits);
__IO_REG32(    FUNC_ERR_ADD,      0xFFF87020,__READ       );
__IO_REG32_BIT(FEDACSDIS,         0xFFF87024,__READ_WRITE ,__fedacsdis_bits);
__IO_REG32_BIT(FBPROT,            0xFFF87030,__READ_WRITE ,__fbprot_bits);
__IO_REG32_BIT(FBSE,              0xFFF87034,__READ_WRITE ,__fbse_bits);
__IO_REG32_BIT(FBAC,              0xFFF8703C,__READ_WRITE ,__fbac_bits);
__IO_REG32_BIT(FBFALLBACK,        0xFFF87040,__READ_WRITE ,__fbfallback_bits);
__IO_REG32_BIT(FBPRDY,            0xFFF87044,__READ       ,__fbprdy_bits);
__IO_REG32_BIT(FPAC1,             0xFFF87048,__READ_WRITE ,__fpac1_bits);
__IO_REG32_BIT(FPAC2,             0xFFF8704C,__READ_WRITE ,__fpac2_bits);
__IO_REG32_BIT(FMAC,              0xFFF87050,__READ_WRITE ,__fmac_bits);
__IO_REG32(		 FEMU_DMSW,         0xFFF87058,__READ_WRITE );
__IO_REG32(		 FEMU_DLSW,         0xFFF8705C,__READ_WRITE );
__IO_REG32_BIT(FEMU_ECC,         	0xFFF87060,__READ_WRITE ,__femu_ecc_bits);
__IO_REG32_BIT(FEDACSDIS2,        0xFFF870C0,__READ_WRITE ,__fedacsdis2_bits);

/***************************************************************************
 **
 ** CRC (Cyclic Redundancy Check Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(CRC_CTRL0,         0xFE000000,__READ_WRITE ,__crc_ctrl0_bits);
__IO_REG32_BIT(CRC_CTRL1,         0xFE000008,__READ_WRITE ,__crc_ctrl1_bits);
__IO_REG32_BIT(CRC_CTRL2,         0xFE000010,__READ_WRITE ,__crc_ctrl2_bits);
__IO_REG32(    PSA_SIGREGL1,      0xFE000060,__READ_WRITE );
__IO_REG32(    PSA_SIGREGH1,      0xFE000064,__READ_WRITE );
__IO_REG32(    PSA_SECSIGREGL1,   0xFE000070,__READ       );
__IO_REG32(    PSA_SECSIGREGH1,   0xFE000074,__READ       );
__IO_REG32(    RAW_DATAREGL1,     0xFE000078,__READ       );
__IO_REG32(    RAW_DATAREGH1,     0xFE00007C,__READ       );

/***************************************************************************
 **
 ** DCAN1 (Controller Area Network)
 **
 ***************************************************************************/
__IO_REG32_BIT(DCAN1CTL,          0xFFF7DC00,__READ_WRITE ,__dcanctl_bits);
__IO_REG32_BIT(DCAN1ES,           0xFFF7DC04,__READ       ,__dcanes_bits);
__IO_REG32_BIT(DCAN1ERRC,         0xFFF7DC08,__READ       ,__dcanerrc_bits);
__IO_REG32_BIT(DCAN1BTR,          0xFFF7DC0C,__READ_WRITE ,__dcanbtr_bits);
__IO_REG32_BIT(DCAN1INT,          0xFFF7DC10,__READ       ,__dcanint_bits);
__IO_REG32_BIT(DCAN1TEST,         0xFFF7DC14,__READ_WRITE ,__dcantest_bits);
__IO_REG32_BIT(DCAN1PERR,         0xFFF7DC1C,__READ       ,__dcanperr_bits);
__IO_REG32_BIT(DCAN1REL,         	0xFFF7DC20,__READ_WRITE ,__dcanrel_bits);
__IO_REG32(    DCAN1ABOTR,        0xFFF7DC80,__READ_WRITE );
__IO_REG32_BIT(DCAN1TXRQX,        0xFFF7DC84,__READ       ,__dcantxrqx_bits);
__IO_REG32_BIT(DCAN1TXRQ12,       0xFFF7DC88,__READ       ,__dcantxrq12_bits);
__IO_REG32_BIT(DCAN1TXRQ34,       0xFFF7DC8C,__READ       ,__dcantxrq34_bits);
__IO_REG32_BIT(DCAN1TXRQ56,       0xFFF7DC90,__READ       ,__dcantxrq56_bits);
__IO_REG32_BIT(DCAN1TXRQ78,       0xFFF7DC94,__READ       ,__dcantxrq78_bits);
__IO_REG32_BIT(DCAN1NWDATX,       0xFFF7DC98,__READ       ,__dcannwdatx_bits);
__IO_REG32_BIT(DCAN1NWDAT12,      0xFFF7DC9C,__READ       ,__dcannwdat12_bits);
__IO_REG32_BIT(DCAN1NWDAT34,      0xFFF7DCA0,__READ       ,__dcannwdat34_bits);
__IO_REG32_BIT(DCAN1NWDAT56,      0xFFF7DCA4,__READ       ,__dcannwdat56_bits);
__IO_REG32_BIT(DCAN1NWDAT78,      0xFFF7DCA8,__READ       ,__dcannwdat78_bits);
__IO_REG32_BIT(DCAN1INTPND,       0xFFF7DCAC,__READ       ,__dcanintpnd_bits);
__IO_REG32_BIT(DCAN1INTPND12,     0xFFF7DCB0,__READ       ,__dcanintpnd12_bits);
__IO_REG32_BIT(DCAN1INTPND34,     0xFFF7DCB4,__READ       ,__dcanintpnd34_bits);
__IO_REG32_BIT(DCAN1INTPND56,     0xFFF7DCB8,__READ       ,__dcanintpnd56_bits);
__IO_REG32_BIT(DCAN1INTPND78,     0xFFF7DCBC,__READ       ,__dcanintpnd78_bits);
__IO_REG32_BIT(DCAN1MSGVAL,       0xFFF7DCC0,__READ       ,__dcanmsgval_bits);
__IO_REG32_BIT(DCAN1MSGVAL12,     0xFFF7DCC4,__READ       ,__dcanmsgval12_bits);
__IO_REG32_BIT(DCAN1MSGVAL34,     0xFFF7DCC8,__READ       ,__dcanmsgval34_bits);
__IO_REG32_BIT(DCAN1MSGVAL56,     0xFFF7DCCC,__READ       ,__dcanmsgval56_bits);
__IO_REG32_BIT(DCAN1MSGVAL78,     0xFFF7DCD0,__READ       ,__dcanmsgval78_bits);
__IO_REG32_BIT(DCAN1INTMUX12,     0xFFF7DCD8,__READ_WRITE ,__dcanintmux12_bits);
__IO_REG32_BIT(DCAN1INTMUX34,     0xFFF7DCDC,__READ_WRITE ,__dcanintmux34_bits);
__IO_REG32_BIT(DCAN1INTMUX56,     0xFFF7DCE0,__READ_WRITE ,__dcanintmux56_bits);
__IO_REG32_BIT(DCAN1INTMUX78,     0xFFF7DCE4,__READ_WRITE ,__dcanintmux78_bits);
__IO_REG32_BIT(DCAN1IF1CMD,       0xFFF7DD00,__READ_WRITE ,__dcanifcmd_bits);
__IO_REG32_BIT(DCAN1IF1MSK,       0xFFF7DD04,__READ_WRITE ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN1IF1ARB,       0xFFF7DD08,__READ_WRITE ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN1IF1MCTL,      0xFFF7DD0C,__READ_WRITE ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN1IF1DATA,      0xFFF7DD10,__READ_WRITE ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN1IF1DATB,      0xFFF7DD14,__READ_WRITE ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN1IF2CMD,       0xFFF7DD20,__READ_WRITE ,__dcanifcmd_bits);
__IO_REG32_BIT(DCAN1IF2MSK,       0xFFF7DD24,__READ_WRITE ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN1IF2ARB,       0xFFF7DD28,__READ_WRITE ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN1IF2MCTL,      0xFFF7DD2C,__READ_WRITE ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN1IF2DATA,      0xFFF7DD30,__READ_WRITE ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN1IF2DATB,      0xFFF7DD34,__READ_WRITE ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN1IF3OBS,       0xFFF7DD40,__READ_WRITE ,__dcanif3obs_bits);
__IO_REG32_BIT(DCAN1IF3MSK,       0xFFF7DD44,__READ       ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN1IF3ARB,       0xFFF7DD48,__READ       ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN1IF3MCTL,      0xFFF7DD4C,__READ       ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN1IF3DATA,      0xFFF7DD50,__READ       ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN1IF3DATB,      0xFFF7DD54,__READ       ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN1IF3UPD12,     0xFFF7DD60,__READ_WRITE ,__dcanif3upd12_bits);
__IO_REG32_BIT(DCAN1IF3UPD34,     0xFFF7DD64,__READ_WRITE ,__dcanif3upd34_bits);
__IO_REG32_BIT(DCAN1IF3UPD56,     0xFFF7DD68,__READ_WRITE ,__dcanif3upd56_bits);
__IO_REG32_BIT(DCAN1IF3UPD78,     0xFFF7DD6C,__READ_WRITE ,__dcanif3upd78_bits);
__IO_REG32_BIT(DCAN1TIOC,         0xFFF7DDE0,__READ_WRITE ,__dcantioc_bits);
__IO_REG32_BIT(DCAN1RIOC,         0xFFF7DDE4,__READ_WRITE ,__dcantioc_bits);
__IO_REG32(    DCAN1RAM_BASE,     0xFF1E0000,__READ_WRITE );

/***************************************************************************
 **
 ** DCAN2 (Controller Area Network)
 **
 ***************************************************************************/
__IO_REG32_BIT(DCAN2CTL,          0xFFF7DE00,__READ_WRITE ,__dcanctl_bits);
__IO_REG32_BIT(DCAN2ES,           0xFFF7DE04,__READ       ,__dcanes_bits);
__IO_REG32_BIT(DCAN2ERRC,         0xFFF7DE08,__READ       ,__dcanerrc_bits);
__IO_REG32_BIT(DCAN2BTR,          0xFFF7DE0C,__READ_WRITE ,__dcanbtr_bits);
__IO_REG32_BIT(DCAN2INT,          0xFFF7DE10,__READ       ,__dcanint_bits);
__IO_REG32_BIT(DCAN2TEST,         0xFFF7DE14,__READ_WRITE ,__dcantest_bits);
__IO_REG32_BIT(DCAN2PERR,         0xFFF7DE1C,__READ       ,__dcanperr_bits);
__IO_REG32_BIT(DCAN2REL,         	0xFFF7DE20,__READ				,__dcanrel_bits);
__IO_REG32(    DCAN2ABOTR,        0xFFF7DE80,__READ_WRITE );
__IO_REG32_BIT(DCAN2TXRQX,        0xFFF7DE84,__READ       ,__dcantxrqx_bits);
__IO_REG32_BIT(DCAN2TXRQ12,       0xFFF7DE88,__READ       ,__dcantxrq12_bits);
__IO_REG32_BIT(DCAN2TXRQ34,       0xFFF7DE8C,__READ       ,__dcantxrq34_bits);
__IO_REG32_BIT(DCAN2TXRQ56,       0xFFF7DE90,__READ       ,__dcantxrq56_bits);
__IO_REG32_BIT(DCAN2TXRQ78,       0xFFF7DE94,__READ       ,__dcantxrq78_bits);
__IO_REG32_BIT(DCAN2NWDATX,       0xFFF7DE98,__READ       ,__dcannwdatx_bits);
__IO_REG32_BIT(DCAN2NWDAT12,      0xFFF7DE9C,__READ       ,__dcannwdat12_bits);
__IO_REG32_BIT(DCAN2NWDAT34,      0xFFF7DEA0,__READ       ,__dcannwdat34_bits);
__IO_REG32_BIT(DCAN2NWDAT56,      0xFFF7DEA4,__READ       ,__dcannwdat56_bits);
__IO_REG32_BIT(DCAN2NWDAT78,      0xFFF7DEA8,__READ       ,__dcannwdat78_bits);
__IO_REG32_BIT(DCAN2INTPND,       0xFFF7DEAC,__READ       ,__dcanintpnd_bits);
__IO_REG32_BIT(DCAN2INTPND12,     0xFFF7DEB0,__READ       ,__dcanintpnd12_bits);
__IO_REG32_BIT(DCAN2INTPND34,     0xFFF7DEB4,__READ       ,__dcanintpnd34_bits);
__IO_REG32_BIT(DCAN2INTPND56,     0xFFF7DEB8,__READ       ,__dcanintpnd56_bits);
__IO_REG32_BIT(DCAN2INTPND78,     0xFFF7DEBC,__READ       ,__dcanintpnd78_bits);
__IO_REG32_BIT(DCAN2MSGVAL,       0xFFF7DEC0,__READ       ,__dcanmsgval_bits);
__IO_REG32_BIT(DCAN2MSGVAL12,     0xFFF7DEC4,__READ       ,__dcanmsgval12_bits);
__IO_REG32_BIT(DCAN2MSGVAL34,     0xFFF7DEC8,__READ       ,__dcanmsgval34_bits);
__IO_REG32_BIT(DCAN2MSGVAL56,     0xFFF7DECC,__READ       ,__dcanmsgval56_bits);
__IO_REG32_BIT(DCAN2MSGVAL78,     0xFFF7DED0,__READ       ,__dcanmsgval78_bits);
__IO_REG32_BIT(DCAN2INTMUX12,     0xFFF7DED8,__READ_WRITE ,__dcanintmux12_bits);
__IO_REG32_BIT(DCAN2INTMUX34,     0xFFF7DEDC,__READ_WRITE ,__dcanintmux34_bits);
__IO_REG32_BIT(DCAN2INTMUX56,     0xFFF7DEE0,__READ_WRITE ,__dcanintmux56_bits);
__IO_REG32_BIT(DCAN2INTMUX78,     0xFFF7DEE4,__READ_WRITE ,__dcanintmux78_bits);
__IO_REG32_BIT(DCAN2IF1CMD,       0xFFF7DF00,__READ_WRITE ,__dcanifcmd_bits);
__IO_REG32_BIT(DCAN2IF1MSK,       0xFFF7DF04,__READ_WRITE ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN2IF1ARB,       0xFFF7DF08,__READ_WRITE ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN2IF1MCTL,      0xFFF7DF0C,__READ_WRITE ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN2IF1DATA,      0xFFF7DF10,__READ_WRITE ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN2IF1DATB,      0xFFF7DF14,__READ_WRITE ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN2IF2CMD,       0xFFF7DF20,__READ_WRITE ,__dcanifcmd_bits);
__IO_REG32_BIT(DCAN2IF2MSK,       0xFFF7DF24,__READ_WRITE ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN2IF2ARB,       0xFFF7DF28,__READ_WRITE ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN2IF2MCTL,      0xFFF7DF2C,__READ_WRITE ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN2IF2DATA,      0xFFF7DF30,__READ_WRITE ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN2IF2DATB,      0xFFF7DF34,__READ_WRITE ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN2IF3OBS,       0xFFF7DF40,__READ_WRITE ,__dcanif3obs_bits);
__IO_REG32_BIT(DCAN2IF3MSK,       0xFFF7DF44,__READ       ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN2IF3ARB,       0xFFF7DF48,__READ       ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN2IF3MCTL,      0xFFF7DF4C,__READ       ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN2IF3DATA,      0xFFF7DF50,__READ       ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN2IF3DATB,      0xFFF7DF54,__READ       ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN2IF3UPD12,     0xFFF7DF60,__READ_WRITE ,__dcanif3upd12_bits);
__IO_REG32_BIT(DCAN2IF3UPD34,     0xFFF7DF64,__READ_WRITE ,__dcanif3upd34_bits);
__IO_REG32_BIT(DCAN2IF3UPD56,     0xFFF7DF68,__READ_WRITE ,__dcanif3upd56_bits);
__IO_REG32_BIT(DCAN2IF3UPD78,     0xFFF7DF6C,__READ_WRITE ,__dcanif3upd78_bits);
__IO_REG32_BIT(DCAN2TIOC,         0xFFF7DFE0,__READ_WRITE ,__dcantioc_bits);
__IO_REG32_BIT(DCAN2RIOC,         0xFFF7DFE4,__READ_WRITE ,__dcantioc_bits);
__IO_REG32(    DCAN2RAM_BASE,     0xFF1C0000,__READ_WRITE );

/***************************************************************************
 **
 ** MibSPIP2 (Multi-Buffered Serial Peripheral Interface with Parallel Pin)
 **
 ***************************************************************************/
__IO_REG32_BIT(MibSPIP2GCR0,      0xFFF7F600,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(MibSPIP2GCR1,      0xFFF7F604,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(MibSPIP2INT0,      0xFFF7F608,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(MibSPIP2LVL,       0xFFF7F60C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(MibSPIP2FLG,       0xFFF7F610,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(MibSPIP2PC0,       0xFFF7F614,__READ_WRITE ,__spippc0_bits);
__IO_REG32_BIT(MibSPIP2PC1,       0xFFF7F618,__READ_WRITE ,__spippc1_bits);
__IO_REG32_BIT(MibSPIP2PC2,       0xFFF7F61C,__READ       ,__spippc2_bits);
__IO_REG32_BIT(MibSPIP2PC3,       0xFFF7F620,__READ_WRITE ,__spippc3_bits);
__IO_REG32_BIT(MibSPIP2PC4,       0xFFF7F624,__READ_WRITE ,__spippc4_bits);
__IO_REG32_BIT(MibSPIP2PC5,       0xFFF7F628,__READ_WRITE ,__spippc5_bits);
__IO_REG32_BIT(MibSPIP2PC6,       0xFFF7F62C,__READ_WRITE ,__spippc6_bits);
__IO_REG32_BIT(MibSPIP2PC7,       0xFFF7F630,__READ_WRITE ,__spippc7_bits);
__IO_REG32_BIT(MibSPIP2PC8,       0xFFF7F634,__READ_WRITE ,__spippc8_bits);
__IO_REG32_BIT(MibSPIP2DAT0,      0xFFF7F638,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(MibSPIP2DAT1,      0xFFF7F63C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(MibSPIP2BUF,       0xFFF7F640,__READ       ,__spibuf_bits);
__IO_REG32_BIT(MibSPIP2EMU,       0xFFF7F644,__READ       ,__spiemu_bits);
__IO_REG32_BIT(MibSPIP2DELAY,     0xFFF7F648,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(MibSPIP2DEF,       0xFFF7F64C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(MibSPIP2FMT0,      0xFFF7F650,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPIP2FMT1,      0xFFF7F654,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPIP2FMT2,      0xFFF7F658,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPIP2FMT3,      0xFFF7F65C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(TG2INTVECT0,       0xFFF7F660,__READ       ,__tgintvect0_bits);
__IO_REG32_BIT(TG2INTVECT1,       0xFFF7F664,__READ       ,__tgintvect1_bits);
__IO_REG32_BIT(MibSPIP2PMCTRL,    0xFFF7F66C,__READ_WRITE ,__spipmctrl_bits);
__IO_REG32_BIT(MibSPIP2MIBSPIE,   0xFFF7F670,__READ_WRITE ,__spimibspie_bits);
__IO_REG32_BIT(TG2ITENST,         0xFFF7F674,__READ_WRITE ,__tgitenst_bits);
__IO_REG32_BIT(TG2ITENCR,         0xFFF7F678,__READ_WRITE ,__tgitencr_bits);
__IO_REG32_BIT(TG2ITLVST,         0xFFF7F67C,__READ_WRITE ,__tgitlvst_bits);
__IO_REG32_BIT(TG2ITLVCR,         0xFFF7F680,__READ_WRITE ,__tgitlvcr_bits);
__IO_REG32_BIT(TG2ITFLG,          0xFFF7F684,__READ_WRITE ,__tgitflg_bits);
__IO_REG32_BIT(TG2TICKCNT,        0xFFF7F690,__READ_WRITE ,__tgtickcnt_bits);
__IO_REG32_BIT(TG2LTGPEND,        0xFFF7F694,__READ_WRITE ,__tgltgpend_bits);
__IO_REG32_BIT(TG2CTRL0,          0xFFF7F698,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL1,          0xFFF7F69C,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL2,          0xFFF7F6A0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL3,          0xFFF7F6A4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL4,          0xFFF7F6A8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL5,          0xFFF7F6AC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL6,          0xFFF7F6B0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL7,          0xFFF7F6B4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL8,          0xFFF7F6B8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL9,          0xFFF7F6BC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL10,         0xFFF7F6C0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL11,         0xFFF7F6C4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL12,         0xFFF7F6C8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL13,         0xFFF7F6CC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL14,         0xFFF7F6D0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG2CTRL15,         0xFFF7F6D4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(MibSPIP2DMA0CTRL,  0xFFF7F6D8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA1CTRL,  0xFFF7F6DC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA2CTRL,  0xFFF7F6E0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA3CTRL,  0xFFF7F6E4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA4CTRL,  0xFFF7F6E8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA5CTRL,  0xFFF7F6EC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA6CTRL,  0xFFF7F6F0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA7CTRL,  0xFFF7F6F4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP2DMA0COUNT, 0xFFF7F6F8,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA1COUNT, 0xFFF7F6FC,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA2COUNT, 0xFFF7F700,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA3COUNT, 0xFFF7F704,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA4COUNT, 0xFFF7F708,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA5COUNT, 0xFFF7F70C,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA6COUNT, 0xFFF7F710,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMA7COUNT, 0xFFF7F714,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP2DMACNTLEN, 0xFFF7F718,__READ_WRITE ,__spidmacntlen_bits);
__IO_REG32_BIT(MibSPIP2UERRCTRL,  0xFFF7F720,__READ_WRITE ,__spiuerrctrl_bits);
__IO_REG32_BIT(MibSPIP2UERRSTAT,  0xFFF7F724,__READ_WRITE ,__spiuerrstat_bits);
__IO_REG32_BIT(MibSPIP2UERRADDR1, 0xFFF7F728,__READ_WRITE ,__spiuerraddr1_bits);
__IO_REG32_BIT(MibSPIP2UERRADDR0, 0xFFF7F72C,__READ_WRITE ,__spiuerraddr0_bits);
__IO_REG32_BIT(MibSPIP2RXOVRN_BUF_ADDR,0xFFF7F730,__READ  ,__spirxovrn_buf_addr_bits);
__IO_REG32_BIT(MibSPIP2IOLPBKTSTCR,0xFFF7F734,__READ_WRITE,__spiiolpbktstcr_bits);
__IO_REG32(MibSPIP2_BUFER_TX_BASE,0xFF0C0000,__READ_WRITE );
__IO_REG32(MibSPIP2_BUFER_RX_BASE,0xFF0C0200,__READ_WRITE );

/***************************************************************************
 **
 ** MibSPI1 (Multi-Buffered Serial Peripheral Interface)
 **
 ***************************************************************************/
__IO_REG32_BIT(MibSPI1GCR0,       0xFFF7F400,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(MibSPI1GCR1,       0xFFF7F404,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(MibSPI1INT0,       0xFFF7F408,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(MibSPI1LVL,        0xFFF7F40C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(MibSPI1FLG,        0xFFF7F410,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(MibSPI1PC0,        0xFFF7F414,__READ_WRITE ,__spipc0_bits);
__IO_REG32_BIT(MibSPI1PC1,        0xFFF7F418,__READ_WRITE ,__spipc1_bits);
__IO_REG32_BIT(MibSPI1PC2,        0xFFF7F41C,__READ       ,__spipc2_bits);
__IO_REG32_BIT(MibSPI1PC3,        0xFFF7F420,__READ_WRITE ,__spipc3_bits);
__IO_REG32_BIT(MibSPI1PC4,        0xFFF7F424,__READ_WRITE ,__spipc4_bits);
__IO_REG32_BIT(MibSPI1PC5,        0xFFF7F428,__READ_WRITE ,__spipc5_bits);
__IO_REG32_BIT(MibSPI1PC6,        0xFFF7F42C,__READ_WRITE ,__spipc6_bits);
__IO_REG32_BIT(MibSPI1PC7,        0xFFF7F430,__READ_WRITE ,__spipc7_bits);
__IO_REG32_BIT(MibSPI1PC8,        0xFFF7F434,__READ_WRITE ,__spipc8_bits);
__IO_REG32_BIT(MibSPI1DAT0,       0xFFF7F438,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(MibSPI1DAT1,       0xFFF7F43C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(MibSPI1BUF,        0xFFF7F440,__READ       ,__spibuf_bits);
__IO_REG32_BIT(MibSPI1EMU,        0xFFF7F444,__READ       ,__spiemu_bits);
__IO_REG32_BIT(MibSPI1DELAY,      0xFFF7F448,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(MibSPI1DEF,        0xFFF7F44C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(MibSPI1FMT0,       0xFFF7F450,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPI1FMT1,       0xFFF7F454,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPI1FMT2,       0xFFF7F458,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPI1FMT3,       0xFFF7F45C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(TG1INTVECT0,       0xFFF7F460,__READ       ,__tgintvect0_bits);
__IO_REG32_BIT(TG1INTVECT1,       0xFFF7F464,__READ       ,__tgintvect1_bits);
__IO_REG32_BIT(MibSPI1PMCTRL,     0xFFF7F46C,__READ_WRITE ,__spipmctrl_bits);
__IO_REG32_BIT(MibSPI1MIBSPIE,    0xFFF7F470,__READ_WRITE ,__spimibspie_bits);
__IO_REG32_BIT(TG1ITENST,         0xFFF7F474,__READ_WRITE ,__tgitenst_bits);
__IO_REG32_BIT(TG1ITENCR,         0xFFF7F478,__READ_WRITE ,__tgitencr_bits);
__IO_REG32_BIT(TG1ITLVST,         0xFFF7F47C,__READ_WRITE ,__tgitlvst_bits);
__IO_REG32_BIT(TG1ITLVCR,         0xFFF7F480,__READ_WRITE ,__tgitlvcr_bits);
__IO_REG32_BIT(TG1ITFLG,          0xFFF7F484,__READ_WRITE ,__tgitflg_bits);
__IO_REG32_BIT(TG1TICKCNT,        0xFFF7F490,__READ_WRITE ,__tgtickcnt_bits);
__IO_REG32_BIT(TG1LTGPEND,        0xFFF7F494,__READ_WRITE ,__tgltgpend_bits);
__IO_REG32_BIT(TG1CTRL0,          0xFFF7F498,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL1,          0xFFF7F49C,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL2,          0xFFF7F4A0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL3,          0xFFF7F4A4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL4,          0xFFF7F4A8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL5,          0xFFF7F4AC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL6,          0xFFF7F4B0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL7,          0xFFF7F4B4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL8,          0xFFF7F4B8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL9,          0xFFF7F4BC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL10,         0xFFF7F4C0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL11,         0xFFF7F4C4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL12,         0xFFF7F4C8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL13,         0xFFF7F4CC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL14,         0xFFF7F4D0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG1CTRL15,         0xFFF7F4D4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(MibSPI1DMA0CTRL,   0xFFF7F4D8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA1CTRL,   0xFFF7F4DC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA2CTRL,   0xFFF7F4E0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA3CTRL,   0xFFF7F4E4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA4CTRL,   0xFFF7F4E8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA5CTRL,   0xFFF7F4EC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA6CTRL,   0xFFF7F4F0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA7CTRL,   0xFFF7F4F4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI1DMA0COUNT,  0xFFF7F4F8,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA1COUNT,  0xFFF7F4FC,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA2COUNT,  0xFFF7F500,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA3COUNT,  0xFFF7F504,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA4COUNT,  0xFFF7F508,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA5COUNT,  0xFFF7F50C,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA6COUNT,  0xFFF7F510,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMA7COUNT,  0xFFF7F514,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI1DMACNTLEN,  0xFFF7F518,__READ_WRITE ,__spidmacntlen_bits);
__IO_REG32_BIT(MibSPI1UERRCTRL,   0xFFF7F520,__READ_WRITE ,__spiuerrctrl_bits);
__IO_REG32_BIT(MibSPI1UERRSTAT,   0xFFF7F524,__READ_WRITE ,__spiuerrstat_bits);
__IO_REG32_BIT(MibSPI1UERRADDR1,  0xFFF7F528,__READ_WRITE ,__spiuerraddr1_bits);
__IO_REG32_BIT(MibSPI1UERRADDR0,  0xFFF7F52C,__READ_WRITE ,__spiuerraddr0_bits);
__IO_REG32_BIT(MibSPI1RXOVRN_BUF_ADDR,0xFFF7F530,__READ   ,__spirxovrn_buf_addr_bits);
__IO_REG32_BIT(MibSPI1IOLPBKTSTCR,0xFFF7F534,__READ_WRITE ,__spiiolpbktstcr_bits);
__IO_REG32(MibSPI1_BUFER_TX_BASE, 0xFF0E0000,__READ_WRITE );
__IO_REG32(MibSPI1_BUFER_RX_BASE, 0xFF0E0200,__READ_WRITE );

/***************************************************************************
 **
 ** NHET (High-End Timer)
 **
 ***************************************************************************/
__IO_REG32_BIT(HETGCR,            0xFFF7B800,__READ_WRITE ,__hetgcr_bits);
__IO_REG32_BIT(HETPFR,            0xFFF7B804,__READ_WRITE ,__hetpfr_bits);
__IO_REG32_BIT(HETADDR,           0xFFF7B808,__READ       ,__hetaddr_bits);
__IO_REG32_BIT(HETOFF1,           0xFFF7B80C,__READ       ,__hetoff1_bits);
__IO_REG32_BIT(HETOFF2,           0xFFF7B810,__READ       ,__hetoff2_bits);
__IO_REG32_BIT(HETEXC1,           0xFFF7B814,__READ_WRITE ,__hetexc1_bits);
__IO_REG32_BIT(HETEXC2,           0xFFF7B818,__READ_WRITE ,__hetexc2_bits);
__IO_REG32_BIT(HETPRY,            0xFFF7B81C,__READ_WRITE ,__hetpry_bits);
__IO_REG32_BIT(HETFLG,            0xFFF7B820,__READ_WRITE ,__hetflg_bits);
__IO_REG32_BIT(HETHRSH,           0xFFF7B82C,__READ_WRITE ,__hethrsh_bits);
__IO_REG32_BIT(HETXOR,            0xFFF7B830,__READ_WRITE ,__hetxor_bits);
__IO_REG32_BIT(HETDIR,            0xFFF7B834,__READ_WRITE ,__hetdir_bits);
__IO_REG32_BIT(HETDIN,            0xFFF7B838,__READ       ,__hetdin_bits);
__IO_REG32_BIT(HETDOUT,           0xFFF7B83C,__READ_WRITE ,__hetdout_bits);
__IO_REG32_BIT(HETDSET,           0xFFF7B840,__READ_WRITE ,__hetdset_bits);
__IO_REG32_BIT(HETDCLR,           0xFFF7B844,__READ_WRITE ,__hetdclr_bits);
__IO_REG32_BIT(HETPDR,            0xFFF7B848,__READ_WRITE ,__hetpdr_bits);
__IO_REG32_BIT(HETPULDIS,         0xFFF7B84C,__READ_WRITE ,__hetpuldis_bits);
__IO_REG32_BIT(HETPSL,            0xFFF7B850,__READ_WRITE ,__hetpsl_bits);
__IO_REG32_BIT(HETLPBSEL,         0xFFF7B860,__READ_WRITE ,__hetlpbsel_bits);
__IO_REG32_BIT(HETLPBDIR,         0xFFF7B864,__READ_WRITE ,__hetlpbdir_bits);
__IO_REG32_BIT(HETPCR,            0xFFF7B868,__READ_WRITE ,__hetpcr_bits);
__IO_REG32_BIT(HETPIEN,           0xFFF7B86C,__READ_WRITE ,__hetpien_bits);
__IO_REG32_BIT(HETPIFLG,         	0xFFF7B870,__READ_WRITE ,__hetpiflg_bits);
__IO_REG32_BIT(HETPAR,            0xFFF7B874,__READ       ,__hetpar_bits);
__IO_REG32(    HETP_RAM_BASE,     0xFF460000,__READ_WRITE );

/***************************************************************************
 **
 ** STC (CPU Self Test Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(STCGCR0,         	0xFFFFF800,__READ_WRITE ,__stcgcr0_bits);
__IO_REG32_BIT(STCGCR1,	         	0xFFFFF804,__READ_WRITE ,__stcgcr1_bits);
__IO_REG32(		 STCTPR,         		0xFFFFF808,__READ_WRITE );
__IO_REG32(		 STC_CADDR,         0xFFFFF80C,__READ				);
__IO_REG32_BIT(STCCICR,         	0xFFFFF810,__READ				,__stccicr_bits);
__IO_REG32_BIT(STCGSTAT,         	0xFFFFF814,__READ_WRITE ,__stcgstat_bits);
__IO_REG32_BIT(STCFSTAT,         	0xFFFFF818,__READ_WRITE ,__stcfstat_bits);
__IO_REG32(		 CPU1_CURMISR3,     0xFFFFF82C,__READ				);
__IO_REG32(		 CPU1_CURMISR2,     0xFFFFF830,__READ				);
__IO_REG32(		 CPU1_CURMISR1,     0xFFFFF834,__READ				);
__IO_REG32(		 CPU1_CURMISR0,     0xFFFFF838,__READ				);

/***************************************************************************
 **
 ** ADC (Analog To Digital Converter)
 **
 ***************************************************************************/
__IO_REG32_BIT(ADRSTCR,           0xFFF7C000,__READ_WRITE ,__adrstcr_bits);
__IO_REG32_BIT(ADOPMODECR,        0xFFF7C004,__READ_WRITE ,__adopmodecr_bits);
__IO_REG32_BIT(ADCLOCKCR,         0xFFF7C008,__READ_WRITE ,__adclockcr_bits);
__IO_REG32_BIT(ADCALCR,           0xFFF7C00C,__READ_WRITE ,__adcalcr_bits);
__IO_REG32_BIT(ADEVMODECR,        0xFFF7C010,__READ_WRITE ,__adevmodecr_bits);
__IO_REG32_BIT(ADG1MODECR,        0xFFF7C014,__READ_WRITE ,__adg1modecr_bits);
__IO_REG32_BIT(ADG2MODECR,        0xFFF7C018,__READ_WRITE ,__adg2modecr_bits);
__IO_REG32_BIT(ADEVSRC,           0xFFF7C01C,__READ_WRITE ,__adevsrc_bits);
__IO_REG32_BIT(ADG1SRC,           0xFFF7C020,__READ_WRITE ,__adg1src_bits);
__IO_REG32_BIT(ADG2SRC,           0xFFF7C024,__READ_WRITE ,__adg2src_bits);
__IO_REG32_BIT(ADEVINTENA,        0xFFF7C028,__READ_WRITE ,__adevintena_bits);
__IO_REG32_BIT(ADG1INTENA,        0xFFF7C02C,__READ_WRITE ,__adg1intena_bits);
__IO_REG32_BIT(ADG2INTENA,        0xFFF7C030,__READ_WRITE ,__adg2intena_bits);
__IO_REG32_BIT(ADEVINTFLG,        0xFFF7C034,__READ       ,__adevintflg_bits);
__IO_REG32_BIT(ADG1INTFLG,        0xFFF7C038,__READ       ,__adg1intflg_bits);
__IO_REG32_BIT(ADG2INTFLG,        0xFFF7C03C,__READ       ,__adg2intflg_bits);
__IO_REG32_BIT(ADEVINTCR,         0xFFF7C040,__READ_WRITE ,__adevintcr_bits);
__IO_REG32_BIT(ADG1INTCR,         0xFFF7C044,__READ_WRITE ,__adg1intcr_bits);
__IO_REG32_BIT(ADG2INTCR,         0xFFF7C048,__READ_WRITE ,__adg2intcr_bits);
__IO_REG32_BIT(ADBNDCR,           0xFFF7C058,__READ_WRITE ,__adbndcr_bits);
__IO_REG32_BIT(ADBNDEND,          0xFFF7C05C,__READ_WRITE ,__adbndend_bits);
__IO_REG32_BIT(ADEVSAMP,          0xFFF7C060,__READ_WRITE ,__adevsamp_bits);
__IO_REG32_BIT(ADG1SAMP,          0xFFF7C064,__READ_WRITE ,__adg1samp_bits);
__IO_REG32_BIT(ADG2SAMP,          0xFFF7C068,__READ_WRITE ,__adg2samp_bits);
__IO_REG32_BIT(ADEVSR,            0xFFF7C06C,__READ       ,__adevsr_bits);
__IO_REG32_BIT(ADG1SR,            0xFFF7C070,__READ       ,__adg1sr_bits);
__IO_REG32_BIT(ADG2SR,            0xFFF7C074,__READ       ,__adg2sr_bits);
__IO_REG32_BIT(ADEVSEL,           0xFFF7C078,__READ_WRITE ,__adevsel_bits);
__IO_REG32_BIT(ADG1SEL,           0xFFF7C07C,__READ_WRITE ,__adg1sel_bits);
__IO_REG32_BIT(ADG2SEL,           0xFFF7C080,__READ_WRITE ,__adg2sel_bits);
__IO_REG32_BIT(ADCALR,            0xFFF7C084,__READ_WRITE ,__adcalr_bits);
__IO_REG32_BIT(ADLASTCONV,        0xFFF7C08C,__READ       ,__adlastconv_bits);
__IO_REG32_BIT(ADEVBUFFER0,       0xFFF7C090,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER1,       0xFFF7C094,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER2,       0xFFF7C098,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER3,       0xFFF7C09C,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER4,       0xFFF7C0A0,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER5,       0xFFF7C0A4,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER6,       0xFFF7C0A8,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADEVBUFFER7,       0xFFF7C0AC,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADG1BUFFER0,       0xFFF7C0B0,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER1,       0xFFF7C0B4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER2,       0xFFF7C0B8,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER3,       0xFFF7C0BC,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER4,       0xFFF7C0C0,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER5,       0xFFF7C0C4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER6,       0xFFF7C0C8,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG1BUFFER7,       0xFFF7C0CC,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG2BUFFER0,       0xFFF7C0D0,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER1,       0xFFF7C0D4,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER2,       0xFFF7C0D8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER3,       0xFFF7C0DC,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER4,       0xFFF7C0E0,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER5,       0xFFF7C0E4,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER6,       0xFFF7C0E8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADG2BUFFER7,       0xFFF7C0EC,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADEVEMUBUFFER,     0xFFF7C0F0,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(ADG1EMUBUFFER,     0xFFF7C0F4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(ADG2EMUBUFFER,     0xFFF7C0F8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(ADEVTDIR,          0xFFF7C0FC,__READ_WRITE ,__adevtdir_bits);
__IO_REG32_BIT(ADEVTOUT,          0xFFF7C100,__READ_WRITE ,__adevtout_bits);
__IO_REG32_BIT(ADEVTIN,           0xFFF7C104,__READ       ,__adevtin_bits);
__IO_REG32_BIT(ADEVTSET,          0xFFF7C108,__READ_WRITE ,__adevtset_bits);
__IO_REG32_BIT(ADEVTCLR,          0xFFF7C10C,__READ_WRITE ,__adevtclr_bits);
__IO_REG32_BIT(ADEVTPDR,          0xFFF7C110,__READ_WRITE ,__adevtpdr_bits);
__IO_REG32_BIT(ADEVTPDIS,         0xFFF7C114,__READ_WRITE ,__adevtpdis_bits);
__IO_REG32_BIT(ADEVTPSEL,         0xFFF7C118,__READ_WRITE ,__adevtpsel_bits);
__IO_REG32_BIT(ADEVSAMPDISEN,     0xFFF7C11C,__READ_WRITE ,__adevsampdisen_bits);
__IO_REG32_BIT(ADG1SAMPDISEN,     0xFFF7C120,__READ_WRITE ,__adg1sampdisen_bits);
__IO_REG32_BIT(ADG2SAMPDISEN,     0xFFF7C124,__READ_WRITE ,__adg2sampdisen_bits);
__IO_REG32_BIT(ADMAGINTCR1,       0xFFF7C128,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(ADMAG1MASK,     		0xFFF7C12C,__READ_WRITE ,__admagmask_bits);
__IO_REG32_BIT(ADMAGINTCR2,       0xFFF7C130,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(ADMAG2MASK,     		0xFFF7C134,__READ_WRITE ,__admagmask_bits);
__IO_REG32_BIT(ADMAGINTCR3,       0xFFF7C138,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(ADMAG3MASK,     		0xFFF7C13C,__READ_WRITE ,__admagmask_bits);
__IO_REG32_BIT(ADMAGINTCR4,     	0xFFF7C140,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(ADMAG4MASK,     		0xFFF7C144,__READ_WRITE ,__admagmask_bits);
__IO_REG32_BIT(ADMAGINTCR5,     	0xFFF7C148,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(ADMAG5MASK,     		0xFFF7C14C,__READ_WRITE ,__admagmask_bits);
__IO_REG32_BIT(ADMAGINTCR6,     	0xFFF7C150,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(ADMAG6MASK,     		0xFFF7C154,__READ_WRITE ,__admagmask_bits);
__IO_REG32_BIT(ADMAGINTENASET, 		0xFFF7C158,__READ_WRITE ,__admagintenaset_bits);
__IO_REG32_BIT(ADMAGINTENACLR, 		0xFFF7C15C,__READ_WRITE ,__admagintenaclr_bits);
__IO_REG32_BIT(ADMAGTHRINTFLG,    0xFFF7C160,__READ_WRITE ,__admagthrintflg_bits);
__IO_REG32_BIT(ADMAGTHRINTOFFSET, 0xFFF7C164,__READ       ,__admagthrintoffset_bits);
__IO_REG32_BIT(ADEVFIFORESETCR,   0xFFF7C168,__READ_WRITE ,__adevfiforesetcr_bits);
__IO_REG32_BIT(ADG1FIFORESETCR,   0xFFF7C16C,__READ_WRITE ,__adg1fiforesetcr_bits);
__IO_REG32_BIT(ADG2FIFORESETCR,   0xFFF7C170,__READ_WRITE ,__adg2fiforesetcr_bits);
__IO_REG32_BIT(ADEVRAMADDR,       0xFFF7C174,__READ       ,__adevramaddr_bits);
__IO_REG32_BIT(ADG1RAMADDR,       0xFFF7C178,__READ       ,__adg1ramaddr_bits);
__IO_REG32_BIT(ADG2RAMADDR,       0xFFF7C17C,__READ       ,__adg2ramaddr_bits);
__IO_REG32_BIT(ADPARCR,           0xFFF7C180,__READ_WRITE ,__adparcr_bits);
__IO_REG32_BIT(ADPARADDR,         0xFFF7C184,__READ       ,__adparaddr_bits);
__IO_REG32(    ADBUFER_BASE,      0xFF3E0000,__READ_WRITE );

/***************************************************************************
 **
 ** RTI/DWWD
 **
 ***************************************************************************/
__IO_REG32_BIT(RTIGCTRL,          0xFFFFFC00,__READ_WRITE ,__rtigctrl_bits);
__IO_REG32_BIT(RTICAPCTRL,        0xFFFFFC08,__READ_WRITE ,__rticapctrl_bits);
__IO_REG32_BIT(RTICOMPCTRL,       0xFFFFFC0C,__READ_WRITE ,__rticompctrl_bits);
__IO_REG32(    RTIFRC0,           0xFFFFFC10,__READ_WRITE );
__IO_REG32(    RTIUC0,            0xFFFFFC14,__READ_WRITE );
__IO_REG32(    RTICPUC0,          0xFFFFFC18,__READ_WRITE );
__IO_REG32(    RTICAFRC0,         0xFFFFFC20,__READ       );
__IO_REG32(    RTICAUC0,          0xFFFFFC24,__READ       );
__IO_REG32(    RTIFRC1,           0xFFFFFC30,__READ_WRITE );
__IO_REG32(    RTIUC1,            0xFFFFFC34,__READ_WRITE );
__IO_REG32(    RTICPUC1,          0xFFFFFC38,__READ_WRITE );
__IO_REG32(    RTICAFRC1,         0xFFFFFC40,__READ       );
__IO_REG32(    RTICAUC1,          0xFFFFFC44,__READ       );
__IO_REG32(    RTICOMP0,          0xFFFFFC50,__READ_WRITE );
__IO_REG32(    RTIUDCP0,          0xFFFFFC54,__READ_WRITE );
__IO_REG32(    RTICOMP1,          0xFFFFFC58,__READ_WRITE );
__IO_REG32(    RTIUDCP1,          0xFFFFFC5C,__READ_WRITE );
__IO_REG32(    RTICOMP2,          0xFFFFFC60,__READ_WRITE );
__IO_REG32(    RTIUDCP2,          0xFFFFFC64,__READ_WRITE );
__IO_REG32(    RTICOMP3,          0xFFFFFC68,__READ_WRITE );
__IO_REG32(    RTIUDCP3,          0xFFFFFC6C,__READ_WRITE );
__IO_REG32_BIT(RTISETINT,      		0xFFFFFC80,__READ_WRITE ,__rtisetint_bits);
__IO_REG32_BIT(RTICLEARINT,    		0xFFFFFC84,__READ_WRITE ,__rticlearint_bits);
__IO_REG32_BIT(RTIINTFLAG,        0xFFFFFC88,__READ_WRITE ,__rtiintflag_bits);
__IO_REG32(		 RTIDWDCTRL,        0xFFFFFC90,__READ_WRITE );
__IO_REG32_BIT(RTIDWDPRLD,        0xFFFFFC94,__READ_WRITE ,__rtidwdprld_bits);
__IO_REG32_BIT(RTIWDSTATUS,       0xFFFFFC98,__READ_WRITE ,__rtiwdstatus_bits);
__IO_REG32_BIT(RTIWDKEY,       		0xFFFFFC9C,__READ_WRITE ,__rtiwdkey_bits);
__IO_REG32_BIT(RTIDWDCNTR,       	0xFFFFFCA0,__READ				,__rtidwdcntr_bits);

/***************************************************************************
 **
 ** ESM (Error Signaling Module)
 **
 ***************************************************************************/
__IO_REG32_BIT(ESMIEPSR1,	        0xFFFFF500,__READ_WRITE ,__esmiepsr1_bits);
__IO_REG32_BIT(ESMIEPCR1,         0xFFFFF504,__READ_WRITE	,__esmiepcr1_bits);
__IO_REG32_BIT(ESMIESR1,      		0xFFFFF508,__READ_WRITE ,__esmiesr1_bits);
__IO_REG32_BIT(ESMIECR1,      		0xFFFFF50C,__READ_WRITE ,__esmiecr1_bits);
__IO_REG32_BIT(ESMILSR1,      		0xFFFFF510,__READ_WRITE ,__esmilsr1_bits);
__IO_REG32_BIT(ESMILCR1,      		0xFFFFF514,__READ_WRITE ,__esmilcr1_bits);
__IO_REG32_BIT(ESMSR1,      			0xFFFFF518,__READ_WRITE ,__esmsr_bits);
__IO_REG32_BIT(ESMSR2,      			0xFFFFF51C,__READ_WRITE ,__esmsr_bits);
__IO_REG32_BIT(ESMSR3,      			0xFFFFF520,__READ_WRITE ,__esmsr_bits);
__IO_REG32_BIT(ESMEPSR,      			0xFFFFF524,__READ				,__esmepsr_bits);
__IO_REG32_BIT(ESMIOFFHR,      		0xFFFFF528,__READ				,__esmioffhr_bits);
__IO_REG32_BIT(ESMIOFFLR,      		0xFFFFF52C,__READ				,__esmiofflr_bits);
__IO_REG32_BIT(ESMLTCR,      			0xFFFFF530,__READ				,__esmltcr_bits);
__IO_REG32_BIT(ESMLTCPR,      		0xFFFFF534,__READ				,__esmltcpr_bits);
__IO_REG32_BIT(ESMEKR,      			0xFFFFF538,__READ				,__esmekr_bits);
__IO_REG32_BIT(ESMSSR2,      			0xFFFFF53C,__READ_WRITE ,__esmsr_bits);

/***************************************************************************
 **
 ** SCI1/LIN1 (Serial Communication Interface/Local InterconnectNetwork)
 **
 ***************************************************************************/
__IO_REG32_BIT(SCI1GCR0,          0xFFF7E500,__READ_WRITE ,__scigcr0_bits);
__IO_REG32_BIT(SCI1GCR1,          0xFFF7E504,__READ_WRITE ,__scigcr1_bits);
__IO_REG32_BIT(SCI1GCR2,          0xFFF7E508,__READ_WRITE ,__scigcr2_bits);
__IO_REG32_BIT(SCI1SETINT,        0xFFF7E50C,__READ_WRITE ,__scisetint_bits);
__IO_REG32_BIT(SCI1CLEARINT,      0xFFF7E510,__READ_WRITE ,__sciclearint_bits);
__IO_REG32_BIT(SCI1SETINTLVL,     0xFFF7E514,__READ_WRITE ,__scisetintlvl_bits);
__IO_REG32_BIT(SCI1CLEARINTLVL,   0xFFF7E518,__READ_WRITE ,__sciclearintlvl_bits);
__IO_REG32_BIT(SCI1FLR,           0xFFF7E51C,__READ_WRITE ,__sciflr_bits);
__IO_REG32_BIT(SCI1INTVECT0,      0xFFF7E520,__READ       ,__sciintvect0_bits);
__IO_REG32_BIT(SCI1INTVECT1,      0xFFF7E524,__READ       ,__sciintvect1_bits);
__IO_REG32_BIT(SCI1FORMAT,        0xFFF7E528,__READ_WRITE ,__sciformat_bits);
__IO_REG32_BIT(SCI1BRS,           0xFFF7E52C,__READ_WRITE ,__scibrs_bits);
__IO_REG32_BIT(SCI1ED,            0xFFF7E530,__READ       ,__scied_bits);
__IO_REG32_BIT(SCI1RD,            0xFFF7E534,__READ       ,__scird_bits);
__IO_REG32_BIT(SCI1TD,            0xFFF7E538,__READ_WRITE ,__scitd_bits);
__IO_REG32_BIT(SCI1PIO0,          0xFFF7E53C,__READ_WRITE ,__scipio0_bits);
__IO_REG32_BIT(SCI1PIO1,          0xFFF7E540,__READ_WRITE ,__scipio1_bits);
__IO_REG32_BIT(SCI1PIO2,          0xFFF7E544,__READ       ,__scipio2_bits);
__IO_REG32_BIT(SCI1PIO3,          0xFFF7E548,__READ_WRITE ,__scipio3_bits);
__IO_REG32_BIT(SCI1PIO4,          0xFFF7E54C,__READ_WRITE ,__scipio4_bits);
__IO_REG32_BIT(SCI1PIO5,          0xFFF7E550,__READ_WRITE ,__scipio5_bits);
__IO_REG32_BIT(SCI1PIO6,          0xFFF7E554,__READ_WRITE ,__scipio6_bits);
__IO_REG32_BIT(SCI1PIO7,          0xFFF7E558,__READ_WRITE ,__scipio7_bits);
__IO_REG32_BIT(SCI1PIO8,          0xFFF7E55C,__READ_WRITE ,__scipio8_bits);
__IO_REG32_BIT(LIN1COMPARE,       0xFFF7E560,__READ_WRITE ,__lincompare_bits);
__IO_REG32_BIT(LIN1RD0,           0xFFF7E564,__READ       ,__linrd0_bits);
__IO_REG32_BIT(LIN1RD1,           0xFFF7E568,__READ       ,__linrd1_bits);
__IO_REG32_BIT(LIN1MASK,          0xFFF7E56C,__READ_WRITE ,__linmask_bits);
__IO_REG32_BIT(LIN1ID,            0xFFF7E570,__READ_WRITE ,__linid_bits);
__IO_REG32_BIT(LIN1TD0,           0xFFF7E574,__READ_WRITE ,__lintd0_bits);
__IO_REG32_BIT(LIN1TD1,           0xFFF7E578,__READ_WRITE ,__lintd1_bits);
__IO_REG32_BIT(LIN1MBRS,          0xFFF7E57C,__READ_WRITE ,__linmbrs_bits);
__IO_REG32_BIT(IO1DFTCTRL,        0xFFF7E590,__READ_WRITE ,__iodftctrl_bits);

/***************************************************************************
 **
 ** SCI2/LIN2 (Serial Communication Interface/Local InterconnectNetwork)
 **
 ***************************************************************************/
__IO_REG32_BIT(SCI2GCR0,          0xFFF7E400,__READ_WRITE ,__scigcr0_bits);
__IO_REG32_BIT(SCI2GCR1,          0xFFF7E404,__READ_WRITE ,__scigcr1_bits);
__IO_REG32_BIT(SCI2GCR2,          0xFFF7E408,__READ_WRITE ,__scigcr2_bits);
__IO_REG32_BIT(SCI2SETINT,        0xFFF7E40C,__READ_WRITE ,__scisetint_bits);
__IO_REG32_BIT(SCI2CLEARINT,      0xFFF7E410,__READ_WRITE ,__sciclearint_bits);
__IO_REG32_BIT(SCI2SETINTLVL,     0xFFF7E414,__READ_WRITE ,__scisetintlvl_bits);
__IO_REG32_BIT(SCI2CLEARINTLVL,   0xFFF7E418,__READ_WRITE ,__sciclearintlvl_bits);
__IO_REG32_BIT(SCI2FLR,           0xFFF7E41C,__READ_WRITE ,__sciflr_bits);
__IO_REG32_BIT(SCI2INTVECT0,      0xFFF7E420,__READ       ,__sciintvect0_bits);
__IO_REG32_BIT(SCI2INTVECT1,      0xFFF7E424,__READ       ,__sciintvect1_bits);
__IO_REG32_BIT(SCI2FORMAT,        0xFFF7E428,__READ_WRITE ,__sciformat_bits);
__IO_REG32_BIT(SCI2BRS,           0xFFF7E42C,__READ_WRITE ,__scibrs_bits);
__IO_REG32_BIT(SCI2ED,            0xFFF7E430,__READ       ,__scied_bits);
__IO_REG32_BIT(SCI2RD,            0xFFF7E434,__READ       ,__scird_bits);
__IO_REG32_BIT(SCI2TD,            0xFFF7E438,__READ_WRITE ,__scitd_bits);
__IO_REG32_BIT(SCI2PIO0,          0xFFF7E43C,__READ_WRITE ,__scipio0_bits);
__IO_REG32_BIT(SCI2PIO1,          0xFFF7E440,__READ_WRITE ,__scipio1_bits);
__IO_REG32_BIT(SCI2PIO2,          0xFFF7E444,__READ       ,__scipio2_bits);
__IO_REG32_BIT(SCI2PIO3,          0xFFF7E448,__READ_WRITE ,__scipio3_bits);
__IO_REG32_BIT(SCI2PIO4,          0xFFF7E44C,__READ_WRITE ,__scipio4_bits);
__IO_REG32_BIT(SCI2PIO5,          0xFFF7E450,__READ_WRITE ,__scipio5_bits);
__IO_REG32_BIT(SCI2PIO6,          0xFFF7E454,__READ_WRITE ,__scipio6_bits);
__IO_REG32_BIT(SCI2PIO7,          0xFFF7E458,__READ_WRITE ,__scipio7_bits);
__IO_REG32_BIT(SCI2PIO8,          0xFFF7E45C,__READ_WRITE ,__scipio8_bits);
__IO_REG32_BIT(LIN2COMPARE,       0xFFF7E460,__READ_WRITE ,__lincompare_bits);
__IO_REG32_BIT(LIN2RD0,           0xFFF7E464,__READ       ,__linrd0_bits);
__IO_REG32_BIT(LIN2RD1,           0xFFF7E468,__READ       ,__linrd1_bits);
__IO_REG32_BIT(LIN2MASK,          0xFFF7E46C,__READ_WRITE ,__linmask_bits);
__IO_REG32_BIT(LIN2ID,            0xFFF7E470,__READ_WRITE ,__linid_bits);
__IO_REG32_BIT(LIN2TD0,           0xFFF7E474,__READ_WRITE ,__lintd0_bits);
__IO_REG32_BIT(LIN2TD1,           0xFFF7E478,__READ_WRITE ,__lintd1_bits);
__IO_REG32_BIT(LIN2MBRS,          0xFFF7E47C,__READ_WRITE ,__linmbrs_bits);
__IO_REG32_BIT(IO2DFTCTRL,        0xFFF7E490,__READ_WRITE ,__iodftctrl_bits);

/***************************************************************************
 **
 ** GIO (General-Purpose Input/Output)
 **
 ***************************************************************************/
__IO_REG32_BIT(GIOGCR0,           0xFFF7BC00,__READ_WRITE ,__giogcr0_bits);
__IO_REG32_BIT(GIOINTDET,         0xFFF7BC08,__READ_WRITE ,__giointdet_bits);
__IO_REG32_BIT(GIOPOL,            0xFFF7BC0C,__READ_WRITE ,__giopol_bits);
__IO_REG32_BIT(GIOENASET,         0xFFF7BC10,__READ_WRITE ,__gioenaset_bits);
__IO_REG32_BIT(GIOENACLR,         0xFFF7BC14,__READ_WRITE ,__gioenaclr_bits);
__IO_REG32_BIT(GIOLVLSET,         0xFFF7BC18,__READ_WRITE ,__giolvlset_bits);
__IO_REG32_BIT(GIOLVLCLR,         0xFFF7BC1C,__READ_WRITE ,__giolvlclr_bits);
__IO_REG32_BIT(GIOFLG,            0xFFF7BC20,__READ_WRITE ,__gioflg_bits);
__IO_REG32_BIT(GIOOFFA,           0xFFF7BC24,__READ       ,__giooffa_bits);
__IO_REG32_BIT(GIOOFFB,           0xFFF7BC28,__READ       ,__giooffb_bits);
__IO_REG32_BIT(GIOEMUA,           0xFFF7BC2C,__READ       ,__gioemua_bits);
__IO_REG32_BIT(GIOEMUB,           0xFFF7BC30,__READ       ,__gioemub_bits);
__IO_REG32_BIT(GIODIRA,           0xFFF7BC34,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODINA,           0xFFF7BC38,__READ       ,__giodin_bits);
__IO_REG32_BIT(GIODOUTA,          0xFFF7BC3C,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETA,           0xFFF7BC40,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRA,           0xFFF7BC44,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRA,           0xFFF7BC48,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISA,        0xFFF7BC4C,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLA,           0xFFF7BC50,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRB,           0xFFF7BC54,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODINB,           0xFFF7BC58,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTB,          0xFFF7BC5C,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETB,           0xFFF7BC60,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRB,           0xFFF7BC64,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRB,           0xFFF7BC68,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISB,        0xFFF7BC6C,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLB,           0xFFF7BC70,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRC,           0xFFF7BC74,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODINC,           0xFFF7BC78,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTC,          0xFFF7BC7C,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETC,           0xFFF7BC80,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRC,           0xFFF7BC84,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRC,           0xFFF7BC88,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISC,        0xFFF7BC8C,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLC,           0xFFF7BC90,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRD,           0xFFF7BC94,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODIND,           0xFFF7BC98,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTD,          0xFFF7BC9C,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETD,           0xFFF7BCA0,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRD,           0xFFF7BCA4,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRD,           0xFFF7BCA8,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISD,        0xFFF7BCAC,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLD,           0xFFF7BCB0,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRE,           0xFFF7BCB4,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODINE,           0xFFF7BCB8,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTE,          0xFFF7BCBC,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETE,           0xFFF7BCC0,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRE,           0xFFF7BCC4,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRE,           0xFFF7BCC8,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISE,        0xFFF7BCCC,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLE,           0xFFF7BCD0,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRF,           0xFFF7BCD4,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODINF,           0xFFF7BCD8,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTF,          0xFFF7BCDC,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETF,           0xFFF7BCE0,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRF,           0xFFF7BCE4,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRF,           0xFFF7BCE8,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISF,        0xFFF7BCEC,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLF,           0xFFF7BCF0,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRG,           0xFFF7BCF4,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODING,           0xFFF7BCF8,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTG,          0xFFF7BCFC,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETG,           0xFFF7BD00,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRG,           0xFFF7BD04,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRG,           0xFFF7BD08,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISG,        0xFFF7BD0C,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLG,           0xFFF7BD10,__READ_WRITE ,__giopsl_bits);
__IO_REG32_BIT(GIODIRH,           0xFFF7BD14,__READ_WRITE ,__giodir_bits);
__IO_REG32_BIT(GIODINH,           0xFFF7BD18,__READ_WRITE ,__giodin_bits);
__IO_REG32_BIT(GIODOUTH,          0xFFF7BD1C,__READ_WRITE ,__giodout_bits);
__IO_REG32_BIT(GIOSETH,           0xFFF7BD20,__READ_WRITE ,__gioset_bits);
__IO_REG32_BIT(GIOCLRH,           0xFFF7BD24,__READ_WRITE ,__gioclr_bits);
__IO_REG32_BIT(GIOPDRH,           0xFFF7BD28,__READ_WRITE ,__giopdr_bits);
__IO_REG32_BIT(GIOPULDISH,        0xFFF7BD2C,__READ_WRITE ,__giopuldis_bits);
__IO_REG32_BIT(GIOPSLH,           0xFFF7BD30,__READ_WRITE ,__giopsl_bits);

/* Assembler-specific declarations **********************************************/

#ifdef __IAR_SYSTEMS_ASM__


#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **    TMS470MF06607 INTERRUPT VALUES
 **
***************************************************************************/
/***************************************************************************
 **
 **  NVIC M3VIM Interrupt channels
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
#define M3VIM_ESMH         		16
#define M3VIM_NMI         		17
#define M3VIM_ESML         		18
#define M3VIM_SSI         		19
#define M3VIM_RTIC_0       		20
#define M3VIM_RTIC_1       		21
#define M3VIM_RTIC_2       		22
#define M3VIM_RTIC_3       		23
#define M3VIM_RTIO_0         	24
#define M3VIM_RTIO_1         	25
#define M3VIM_GPIOA         	27
#define M3VIM_GPIOB         	28
#define M3VIM_HET_0         	29
#define M3VIM_HET_1	         	30
#define M3VIM_MibSPI1_0      	31
#define M3VIM_MibSPI1_1      	32
#define M3VIM_LIN2_SCI2_0    	34
#define M3VIM_LIN2_SCI2_1    	35
#define M3VIM_LIN1_SCI1_0    	36
#define M3VIM_LIN1_SCI1_1    	37
#define M3VIM_DCAN1_0	      	38
#define M3VIM_DCAN1_1	      	39
#define M3VIM_ADCEG		      	40
#define M3VIM_ADCSG1	      	41
#define M3VIM_ADCSG2	      	42
#define M3VIM_MibSPI2_0      	43
#define M3VIM_MibSPI2_1      	44
#define M3VIM_DCAN2_0	      	45
#define M3VIM_DCAN2_1	      	46
#define M3VIM_ADCMT 	      	47
#define M3VIM_DCAN1IF3      	50
#define M3VIM_DCAN2IF3      	51

#endif    /* __IOTMS470MF06607_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            				0x08
Interrupt1   = HardFault      				0x0C
Interrupt2   = MemManage      				0x10
Interrupt3   = BusFault       				0x14
Interrupt4   = UsageFault     				0x18
Interrupt5   = SVC            				0x2C
Interrupt6   = DebugMon       				0x30
Interrupt7   = PendSV         				0x38
Interrupt8   = SysTick        				0x3C
Interrupt9   = M3VIM_ESMH         		0x40
Interrupt10  = M3VIM_NMI         		  0x44
Interrupt11  = M3VIM_ESML         		0x48
Interrupt12  = M3VIM_SSI         		  0x4C
Interrupt13  = M3VIM_RTIC_0       		0x50
Interrupt14  = M3VIM_RTIC_1       		0x54
Interrupt15  = M3VIM_RTIC_2       		0x58
Interrupt16  = M3VIM_RTIC_3       		0x5C
Interrupt17  = M3VIM_RTIO_0         	0x60
Interrupt18  = M3VIM_RTIO_1         	0x64
Interrupt19  = M3VIM_GPIOA         	  0x6C
Interrupt20  = M3VIM_GPIOB         	  0x70
Interrupt21  = M3VIM_HET_0         	  0x74
Interrupt22  = M3VIM_HET_1	         	0x78
Interrupt23  = M3VIM_MibSPI1_0      	0x7C
Interrupt24  = M3VIM_MibSPI1_1      	0x80
Interrupt25  = M3VIM_LIN2_SCI2_0    	0x88
Interrupt26  = M3VIM_LIN2_SCI2_1    	0x8C
Interrupt27  = M3VIM_LIN1_SCI1_0    	0x90
Interrupt28  = M3VIM_LIN1_SCI1_1    	0x94
Interrupt29  = M3VIM_DCAN1_0	      	0x98
Interrupt30  = M3VIM_DCAN1_1	      	0x9C
Interrupt31  = M3VIM_ADCEG		      	0xA0
Interrupt32  = M3VIM_ADCSG1	      	  0xA4
Interrupt33  = M3VIM_ADCSG2	      	  0xA8
Interrupt34  = M3VIM_MibSPI2_0      	0xAC
Interrupt35  = M3VIM_MibSPI2_1      	0xB0
Interrupt36  = M3VIM_DCAN2_0	      	0xB4
Interrupt37  = M3VIM_DCAN2_1	      	0xB8
Interrupt38  = M3VIM_ADCMT 	      	  0xBC
Interrupt39  = M3VIM_DCAN1IF3      	  0xC8
Interrupt40  = M3VIM_DCAN2IF3      	  0xCC
###DDF-INTERRUPT-END###*/
