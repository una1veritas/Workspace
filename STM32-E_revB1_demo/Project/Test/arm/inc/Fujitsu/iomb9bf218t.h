/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Fujitsu MB9BF218T
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 54417 $
 **
 ***************************************************************************/

#ifndef __IOMB9BF218T_H
#define __IOMB9BF218T_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   MB9BF218T SPECIAL FUNCTION REGISTERS
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

/* FASZR (Flash Access Size Register) */
typedef struct {
  __REG32  ASZ            : 2;
  __REG32                 :30;
} __faszr_bits;

/* FRWTR (Flash Read Wait Register) */
typedef struct {
  __REG32  RWT            : 2;
  __REG32                 :30;
} __frwtr_bits;

/* FSTR (Flash Status Register) */
typedef struct {
  __REG32  RDY            : 1;
  __REG32  HNG            : 1;
  __REG32  EER            : 1;
  __REG32                 :29;
} __fstr_bits;

/* FSYNDN (Flash Sync Down Register) */
typedef struct {
  __REG32  SD             : 3;
  __REG32                 :29;
} __fsyndn_bits;

/* FBFCR (Flash Buffer Control Register) */
typedef struct {
  __REG32  BE             : 1;
  __REG32  BS             : 1;
  __REG32                 :30;
} __fbfcr_bits;

/* CRTRMM (CR Trimming Data Mirror Register) */
typedef struct {
  __REG32  TRMM           :10;
  __REG32                 :22;
} __crtrmm_bits;

/* System Clock Mode Control Register (SCM_CTL) */
typedef struct {
  __REG32                 : 1;
  __REG32  MOSCE          : 1;
  __REG32                 : 1;
  __REG32  SOSCE          : 1;
  __REG32  PLLE           : 1;
  __REG32  RCS            : 3;
  __REG32                 :24;
} __scm_ctl_bits;

/* System Clock Mode Status Register (SCM_STR) */
typedef struct {
  __REG32                 : 1;
  __REG32  MORDY          : 1;
  __REG32                 : 1;
  __REG32  SORDY          : 1;
  __REG32  PLRDY          : 1;
  __REG32  RCM            : 3;
  __REG32                 :24;
} __scm_str_bits;

/* Standby Mode Control Register (STB_CTL) */
typedef struct {
  __REG32  STM            : 2;
  __REG32                 : 2;
  __REG32  SPL            : 1;
  __REG32                 :11;
  __REG32  KEY            :16;
} __stb_ctl_bits;

/* Reset Cause Register (RST_STR) */
typedef struct {
  __REG32  PONR           : 1;
  __REG32  INITX          : 1;
  __REG32                 : 2;
  __REG32  SWDT           : 1;
  __REG32  HWDT           : 1;
  __REG32  CSVR           : 1;
  __REG32  FCSR           : 1;
  __REG32  SRST           : 1;
  __REG32                 :23;
} __rst_str_bits;

/* Base Clock Prescaler Register (BSC_PSR) */
typedef struct {
  __REG32  BSR            : 3;
  __REG32                 :29;
} __bsc_psr_bits;

/* APB0 Prescaler Register (APBC0_PSR) */
typedef struct {
  __REG32  APBC0          : 2;
  __REG32                 :30;
} __apbc0_psr_bits;

/* APB1 Prescaler Register (APBC1_PSR) */
typedef struct {
  __REG32  APBC1          : 2;
  __REG32                 : 2;
  __REG32  APBC1RST       : 1;
  __REG32                 : 2;
  __REG32  APBC1EN        : 1;
  __REG32                 :24;
} __apbc1_psr_bits;

/* APB2 Prescaler Register (APBC2_PSR) */
typedef struct {
  __REG32  APBC2          : 2;
  __REG32                 : 2;
  __REG32  APBC2RST       : 1;
  __REG32                 : 2;
  __REG32  APBC2EN        : 1;
  __REG32                 :24;
} __apbc2_psr_bits;

/* Software Watchdog Clock Prescaler Register (SWC_PSR) */
typedef struct {
  __REG32  SWDS           : 2;
  __REG32                 : 5;
  __REG32  TESTB          : 1;
  __REG32                 :24;
} __swc_psr_bits;

/* Trace Clock Prescaler Register (TTC_PSR) */
typedef struct {
  __REG32  TTC            : 2;
  __REG32                 :30;
} __ttc_psr_bits;

/* Clock Stabilization Wait Time Register (CSW_TMR) */
typedef struct {
  __REG32  MOWT           : 4;
  __REG32  SOWT           : 3;
  __REG32                 :25;
} __csw_tmr_bits;

/* PLL Clock Stabilization Wait Time Setup Register (PSW_TMR) */
typedef struct {
  __REG32  POWT           : 3;
  __REG32                 : 1;
  __REG32  PINC           : 1;
  __REG32                 :27;
} __psw_tmr_bits;

/* PLL Control Register 1 (PLL_CTL1) */
typedef struct {
  __REG32  PLLM           : 4;
  __REG32  PLLK           : 4;
  __REG32                 :24;
} __pll_ctl1_bits;

/* PLL Control Register 2 (PLL_CTL2) */
typedef struct {
  __REG32  PLLN           : 6;
  __REG32                 :26;
} __pll_ctl2_bits;

/* CSV control register (CSV_CTL) */
typedef struct {
  __REG32  MCSVE          : 1;
  __REG32  SCSVE          : 1;
  __REG32                 : 6;
  __REG32  FCSDE          : 1;
  __REG32  FCSRE          : 1;
  __REG32                 : 2;
  __REG32  FCD            : 3;
  __REG32                 :17;
} __csv_ctl_bits;

/* CSV status register (CSV_STR) */
typedef struct {
  __REG32  MCMF           : 1;
  __REG32  SCMF           : 1;
  __REG32                 :30;
} __csv_str_bits;

/* Frequency detection window setting register (Upper)(FCSWH_CTL) */
typedef struct {
  __REG32  FWH            :16;
  __REG32                 :16;
} __fcswh_ctl_bits;

/* Frequency detection window setting register (Lower)(FCSWL_CTL) */
typedef struct {
  __REG32  FWL            :16;
  __REG32                 :16;
} __fcswl_ctl_bits;

/* Frequency detection counter register (FCSWD_CTL) */
typedef struct {
  __REG32  FWD            :16;
  __REG32                 :16;
} __fcswd_ctl_bits;

/* Debug Break Watchdog Timer Control Register (DBWDT_CTL) */
typedef struct {
  __REG32                 : 5;
  __REG32  DPSWBE         : 1;
  __REG32                 : 1;
  __REG32  DPHWBE         : 1;
  __REG32                 :24;
} __dbwdt_ctl_bits;

/* Interrupt Enable Register (INT_ENR) */
typedef struct {
  __REG32  MCSE           : 1;
  __REG32  SCSE           : 1;
  __REG32  PCSE           : 1;
  __REG32                 : 2;
  __REG32  FCSE           : 1;
  __REG32                 :26;
} __int_enr_bits;

/* Interrupt Status Register (INT_STR) */
typedef struct {
  __REG32  MCSI           : 1;
  __REG32  SCSI           : 1;
  __REG32  PCSI           : 1;
  __REG32                 : 2;
  __REG32  FCSI           : 1;
  __REG32                 :26;
} __int_str_bits;

/* Interrupt Clear Register (INT_CLR) */
typedef struct {
  __REG32  MCSC           : 1;
  __REG32  SCSC           : 1;
  __REG32  PCSC           : 1;
  __REG32                 : 2;
  __REG32  FCSC           : 1;
  __REG32                 :26;
} __int_clr_bits;

/* Software Watchdog Timer Control Register (WdogControl) */
/* Hardware Watchdog Timer Control Register (WDG_CTL) */
typedef struct {
  __REG32  INTEN          : 1;
  __REG32  RESEN          : 1;
  __REG32                 :30;
} __wdg_ctl_bits;

/* Software Watchdog Timer Interrupt Status Register (WdogRIS) */
/* Hardware Watchdog Timer Interrupt Status Register (WDG_RIS) */
typedef struct {
  __REG32  RIS            : 1;
  __REG32                 :31;
} __wdg_ris_bits;

/* Control Register (TimerXControl) X=1 or 2 */
typedef struct {
  __REG32  OneShot        : 1;
  __REG32  TimerSize      : 1;
  __REG32  TimerPre       : 2;
  __REG32                 : 1;
  __REG32  IntEnable      : 1;
  __REG32  TimerMode      : 1;
  __REG32  TimerEn        : 1;
  __REG32                 :24;
} __timercontrol_bits;

/* Interrupt Status Register (TimerXRIS) X=1 or 2 */
typedef struct {
  __REG32  TimerXRIS      : 1;
  __REG32                 :31;
} __timerris_bits;

/* Masked Interrupt Status Register (TimerXMIS) X=1 or 2 */
typedef struct {
  __REG32  TimerXMIS      : 1;
  __REG32                 :31;
} __timermis_bits;

/* OCSA10 (OCU Control Register A OCU ch1 and OCU ch0) */
typedef struct {
  __REG8   CST0           : 1;
  __REG8   CST1           : 1;
  __REG8   BDIS0          : 1;
  __REG8   BDIS1          : 1;
  __REG8   IOE0           : 1;
  __REG8   IOE1           : 1;
  __REG8   IOP0           : 1;
  __REG8   IOP1           : 1;
} __mft_ocsa10_bits;

/* OCSB10 (OCU Control Register B OCU ch1 and OCU ch0) */
typedef struct {
  __REG8   OTD0           : 1;
  __REG8   OTD1           : 1;
  __REG8                  : 2;
  __REG8   CMOD           : 1;
  __REG8   BTS0           : 1;
  __REG8   BTS1           : 1;
  __REG8                  : 1;
} __mft_ocsb10_bits;

/* OCSA32 (OCU Control Register A OCU ch3 and OCU ch2) */
typedef struct {
  __REG8   CST2           : 1;
  __REG8   CST3           : 1;
  __REG8   BDIS2          : 1;
  __REG8   BDIS3          : 1;
  __REG8   IOE2           : 1;
  __REG8   IOE3           : 1;
  __REG8   IOP2           : 1;
  __REG8   IOP3           : 1;
} __mft_ocsa32_bits;

/* OCSB32 (OCU Control Register B OCU ch3 and OCU ch2) */
typedef struct {
  __REG8   OTD2           : 1;
  __REG8   OTD3           : 1;
  __REG8                  : 2;
  __REG8   CMOD           : 1;
  __REG8   BTS2           : 1;
  __REG8   BTS3           : 1;
  __REG8                  : 1;
} __mft_ocsb32_bits;

/* OCSA54 (OCU Control Register A OCU ch5 and OCU ch4) */
typedef struct {
  __REG8   CST4           : 1;
  __REG8   CST5           : 1;
  __REG8   BDIS4          : 1;
  __REG8   BDIS5          : 1;
  __REG8   IOE4           : 1;
  __REG8   IOE5           : 1;
  __REG8   IOP4           : 1;
  __REG8   IOP5           : 1;
} __mft_ocsa54_bits;

/* OCSB54 (OCU Control Register B OCU ch5 and OCU ch4) */
typedef struct {
  __REG8   OTD4           : 1;
  __REG8   OTD5           : 1;
  __REG8                  : 2;
  __REG8   CMOD           : 1;
  __REG8   BTS4           : 1;
  __REG8   BTS5           : 1;
  __REG8                  : 1;
} __mft_ocsb54_bits;

/* OCSC (OCU Control Register C) */
typedef struct {
  __REG8   MOD0           : 1;
  __REG8   MOD1           : 1;
  __REG8   MOD2           : 1;
  __REG8   MOD3           : 1;
  __REG8   MOD4           : 1;
  __REG8   MOD5           : 1;
  __REG8                  : 2;
} __mft_ocsc_bits;

/* TCSA0, TCSA1, TCSA2 (FRT Control Register A) */
typedef struct {
  __REG16  CLK            : 4;
  __REG16  SCLR           : 1;
  __REG16  MODE           : 1;
  __REG16  STOP           : 1;
  __REG16  BFE            : 1;
  __REG16  ICRE           : 1;
  __REG16  ICLR           : 1;
  __REG16                 : 3;
  __REG16  IRQZE          : 1;
  __REG16  IRQZF          : 1;
  __REG16  ECKE           : 1;
} __mft_tcsax_bits;

/* TCSB0, TCSB1, TCSB2 (FRT Control Register B) */
typedef struct {
  __REG16  AD0E           : 1;
  __REG16  AD1E           : 1;
  __REG16  AD2E           : 1;
  __REG16                 :13;
} __mft_tcsbx_bits;

/* OCFS10 (OCU Connecting FRT Select Register ch1 and OCU ch0) */
typedef struct {
  __REG8   FSO0           : 4;
  __REG8   FSO1           : 4;
} __mft_ocfs10_bits;

/* OCFS32 (OCU Connecting FRT Select Register ch3 and OCU ch2) */
typedef struct {
  __REG8   FSO2           : 4;
  __REG8   FSO3           : 4;
} __mft_ocfs32_bits;

/* OCFS54 (OCU Connecting FRT Select Register ch5 and OCU ch4) */
typedef struct {
  __REG8   FSO4           : 4;
  __REG8   FSO5           : 4;
} __mft_ocfs54_bits;

/* ICFS10 (ICU Connecting FRT Select Register ch1 and ICU ch0) */
typedef struct {
  __REG8   FSI0           : 4;
  __REG8   FSI1           : 4;
} __mft_icfs10_bits;

/* ICFS32 (ICU Connecting FRT Select Register ch3 and ICU ch2) */
typedef struct {
  __REG8   FSI2           : 4;
  __REG8   FSI3           : 4;
} __mft_icfs32_bits;

/* ICSA10 (ICU Control Register A ch1 and ICU ch0) */
typedef struct {
  __REG8   EG0            : 2;
  __REG8   EG1            : 2;
  __REG8   ICE0           : 1;
  __REG8   ICE1           : 1;
  __REG8   ICP0           : 1;
  __REG8   ICP1           : 1;
} __mft_icsa10_bits;

/* ICSB10 (ICU Control Register B ICU ch1 and ICU ch0) */
typedef struct {
  __REG8   IEI0           : 1;
  __REG8   IEI1           : 1;
  __REG8                  : 6;
} __mft_icsb10_bits;

/* ICSA32 (ICU Control Register A ch3 and ICU ch2) */
typedef struct {
  __REG8   EG2            : 2;
  __REG8   EG3            : 2;
  __REG8   ICE2           : 1;
  __REG8   ICE3           : 1;
  __REG8   ICP2           : 1;
  __REG8   ICP3           : 1;
} __mft_icsa32_bits;

/* ICSB32 (ICU Control Register B ICU ch3 and ICU ch2) */
typedef struct {
  __REG8   IEI2           : 1;
  __REG8   IEI3           : 1;
  __REG8                  : 6;
} __mft_icsb32_bits;

/* WFSA10, WFSA32, WFSA54 (WFG Control Register A) */
typedef struct {
  __REG16  DCK            : 3;
  __REG16  TMD            : 3;
  __REG16  GTEN           : 2;
  __REG16  PSEL           : 2;
  __REG16  PGEN           : 2;
  __REG16  DMOD           : 1;
  __REG16                 : 3;
} __mft_wfsa_bits;

/* WFIR (WFG Interrupt Control Register) */
typedef struct {
  __REG16  DTIF           : 1;
  __REG16  DTIC           : 1;
  __REG16                 : 2;
  __REG16  TMIF10         : 1;
  __REG16  TMIC10         : 1;
  __REG16  TMIE10         : 1;
  __REG16  TMIS10         : 1;
  __REG16  TMIF32         : 1;
  __REG16  TMIC32         : 1;
  __REG16  TMIE32         : 1;
  __REG16  TMIS32         : 1;
  __REG16  TMIF54         : 1;
  __REG16  TMIC54         : 1;
  __REG16  TMIE54         : 1;
  __REG16  TMIS54         : 1;
} __mft_wfir_bits;

/* NZCL (NZCL Control Register) */
typedef struct {
  __REG16  DTIE           : 1;
  __REG16  NWS            : 3;
  __REG16  SDTI           : 1;
  __REG16                 :11;
} __mft_nzcl_bits;

/* ACSB (ADCMP Control Register B) */
typedef struct {
  __REG16  BDIS0          : 1;
  __REG16  BDIS1          : 1;
  __REG16  BDIS2          : 1;
  __REG16                 : 1;
  __REG16  BTS0           : 1;
  __REG16  BTS1           : 1;
  __REG16  BTS2           : 1;
  __REG16                 : 9;
} __mft_acsb_bits;

/* ACSB (ADCMP Control Register B) */
typedef struct {
  __REG16  CE0            : 2;
  __REG16  CE1            : 2;
  __REG16  CE2            : 2;
  __REG16                 : 2;
  __REG16  SEL0           : 2;
  __REG16  SEL1           : 2;
  __REG16  SEL2           : 2;
  __REG16                 : 2;
} __mft_acsa_bits;

/* ATSA (ADC Start Trigger Select Register) */
typedef struct {
  __REG16  AD0S           : 2;
  __REG16  AD1S           : 2;
  __REG16  AD2S           : 2;
  __REG16                 : 2;
  __REG16  AD0P           : 2;
  __REG16  AD1P           : 2;
  __REG16  AD2P           : 2;
  __REG16                 : 2;
} __mft_atsa_bits;

/* PPG Start Trigger Control Register 0 (TTCR0) */
typedef struct {
  __REG8   STR0           : 1;
  __REG8   MONI0          : 1;
  __REG8   CS0            : 2;
  __REG8   TRG0O          : 1;
  __REG8   TRG2O          : 1;
  __REG8   TRG4O          : 1;
  __REG8   TRG6O          : 1;
} __ppg_ttcr0_bits;

/* PPG Start Trigger Control Register 1 (TTCR1) */
typedef struct {
  __REG8   STR1           : 1;
  __REG8   MONI1          : 1;
  __REG8   CS1            : 2;
  __REG8   TRG1O          : 1;
  __REG8   TRG3O          : 1;
  __REG8   TRG5O          : 1;
  __REG8   TRG7O          : 1;
} __ppg_ttcr1_bits;

/* PPG Start Trigger Control Register 2 (TTCR2) */
typedef struct {
  __REG8   STR2           : 1;
  __REG8   MONI2          : 1;
  __REG8   CS2            : 2;
  __REG8   TRG16O         : 1;
  __REG8   TRG18O         : 1;
  __REG8   TRG20O         : 1;
  __REG8   TRG22O         : 1;
} __ppg_ttcr2_bits;

/* PPG Start Register (TRG) */
typedef struct {
  __REG16  PEN00          : 1;
  __REG16  PEN01          : 1;
  __REG16  PEN02          : 1;
  __REG16  PEN03          : 1;
  __REG16  PEN04          : 1;
  __REG16  PEN05          : 1;
  __REG16  PEN06          : 1;
  __REG16  PEN07          : 1;
  __REG16  PEN08          : 1;
  __REG16  PEN09          : 1;
  __REG16  PEN10          : 1;
  __REG16  PEN11          : 1;
  __REG16  PEN12          : 1;
  __REG16  PEN13          : 1;
  __REG16  PEN14          : 1;
  __REG16  PEN15          : 1;
} __ppg_trg_bits;

/* PPG Start Register 1 (TRG1) */
typedef struct {
  __REG16  PEN16          : 1;
  __REG16  PEN17          : 1;
  __REG16  PEN18          : 1;
  __REG16  PEN19          : 1;
  __REG16  PEN20          : 1;
  __REG16  PEN21          : 1;
  __REG16  PEN22          : 1;
  __REG16  PEN23          : 1;
  __REG16                 : 8;
} __ppg_trg1_bits;

/* Output Reverse Register (REVC) */
typedef struct {
  __REG16  REV00          : 1;
  __REG16  REV01          : 1;
  __REG16  REV02          : 1;
  __REG16  REV03          : 1;
  __REG16  REV04          : 1;
  __REG16  REV05          : 1;
  __REG16  REV06          : 1;
  __REG16  REV07          : 1;
  __REG16  REV08          : 1;
  __REG16  REV09          : 1;
  __REG16  REV10          : 1;
  __REG16  REV11          : 1;
  __REG16  REV12          : 1;
  __REG16  REV13          : 1;
  __REG16  REV14          : 1;
  __REG16  REV15          : 1;
} __ppg_revc_bits;

/* Output Reverse Register 1 (REVC1) */
typedef struct {
  __REG16  REV16          : 1;
  __REG16  REV17          : 1;
  __REG16  REV18          : 1;
  __REG16  REV19          : 1;
  __REG16  REV20          : 1;
  __REG16  REV21          : 1;
  __REG16  REV22          : 1;
  __REG16  REV23          : 1;
  __REG16                 : 8;
} __ppg_revc1_bits;

/* PPG Operation Mode Control Register (PPGC) */
typedef struct {
  __REG8   TTRG           : 1;
  __REG8   MD             : 2;
  __REG8   PCS            : 2;
  __REG8   INTM           : 1;
  __REG8   PUF            : 1;
  __REG8   PIE            : 1;
} __ppg_ppgcx_bits;

/* PPG Reload Registers (PRLH, PRLL) */
typedef union {
  /*PPG_PRLx*/
  struct {
    __REG16   PRLL        : 8;
    __REG16   PRLH        : 8;
  };
  struct {
    __REG8    __byte0 ;
    __REG8    __byte1 ;
  };
}__ppg_prlx_bits;

/* PPG Gate Function Control Registers (GATEC0) */
typedef struct {
  __REG8   EDGE0          : 1;
  __REG8   STRG0          : 1;
  __REG8                  : 2;
  __REG8   EDGE2          : 1;
  __REG8   STRG2          : 1;
  __REG8                  : 2;
} __ppg_gatec0_bits;

/* PPG Gate Function Control Registers (GATEC4) */
typedef struct {
  __REG8   EDGE4          : 1;
  __REG8   STRG4          : 1;
  __REG8                  : 2;
  __REG8   EDGE6          : 1;
  __REG8   STRG6          : 1;
  __REG8                  : 2;
} __ppg_gatec4_bits;

/* PPG Gate Function Control Registers (GATEC8) */
typedef struct {
  __REG8   EDGE8          : 1;
  __REG8   STRG8          : 1;
  __REG8                  : 2;
  __REG8   EDGE10         : 1;
  __REG8   STRG10          : 1;
  __REG8                  : 2;
} __ppg_gatec8_bits;

/* PPG Gate Function Control Registers (GATEC12) */
typedef struct {
  __REG8   EDGE12         : 1;
  __REG8   STRG12         : 1;
  __REG8                  : 2;
  __REG8   EDGE14         : 1;
  __REG8   STRG14         : 1;
  __REG8                  : 2;
} __ppg_gatec12_bits;

/* PPG Gate Function Control Registers (GATEC16) */
typedef struct {
  __REG8   EDGE16         : 1;
  __REG8   STRG16         : 1;
  __REG8                  : 2;
  __REG8   EDGE18         : 1;
  __REG8   STRG18         : 1;
  __REG8                  : 2;
} __ppg_gatec16_bits;

/* PPG Gate Function Control Registers (GATEC20) */
typedef struct {
  __REG8   EDGE20         : 1;
  __REG8   STRG20         : 1;
  __REG8                  : 2;
  __REG8   EDGE22         : 1;
  __REG8   STRG22         : 1;
  __REG8                  : 2;
} __ppg_gatec20_bits;

/* Timer Control Registers (BTxTMCR) */
typedef union {
    /* BTyx_PPG_TMCR */
    /* BTyx_PWM_TMCR */
    struct {
      __REG16  STRG           : 1;
      __REG16  CTEN           : 1;
      __REG16  MDSE           : 1;
      __REG16  OSEL           : 1;
      __REG16  FMD            : 3;
      __REG16                 : 1;
      __REG16  EGS            : 2;
      __REG16  PMSK           : 1;
      __REG16  RTGEN          : 1;
      __REG16  CKS0           : 1;
      __REG16  CKS1           : 1;
      __REG16  CKS2           : 1;
      __REG16                 : 1;
    };
    /* BTyx_RT_TMCR */
    struct {
      __REG16  STRG           : 1;
      __REG16  CTEN           : 1;
      __REG16  MDSE           : 1;
      __REG16  OSEL           : 1;
      __REG16  FMD            : 3;
      __REG16  T32            : 1;
      __REG16  EGS            : 2;
      __REG16                 : 2;
      __REG16  CKS0           : 1;
      __REG16  CKS1           : 1;
      __REG16  CKS2           : 1;
      __REG16                 : 1;
    } RT;
    /* BTyx_PWC_TMCR */
    struct {
      __REG16                 : 1;
      __REG16  CTEN           : 1;
      __REG16  MDSE           : 1;
      __REG16                 : 1;
      __REG16  FMD            : 3;
      __REG16  T32            : 1;
      __REG16  EGS            : 3;
      __REG16                 : 1;
      __REG16  CKS0           : 1;
      __REG16  CKS1           : 1;
      __REG16  CKS2           : 1;
      __REG16                 : 1;
    } PWC;
} __btxtmcr_bits;

/* Status Control Register (STC) */
typedef union {
    /* BTyx_PPG_STC */
    /* BTyx_RT_STC */
    struct {
      __REG8   UDIR           : 1;
      __REG8                  : 1;
      __REG8   TGIR           : 1;
      __REG8                  : 1;
      __REG8   UDIE           : 1;
      __REG8                  : 1;
      __REG8   TGIE           : 1;
      __REG8                  : 1;
    };
    /* BTyx_PWM_STC */
    struct {
      __REG8   UDIR           : 1;
      __REG8   DTIR           : 1;
      __REG8   TGIR           : 1;
      __REG8                  : 1;
      __REG8   UDIE           : 1;
      __REG8   DTIE           : 1;
      __REG8   TGIE           : 1;
      __REG8                  : 1;
    } PWM;
    /* BTyx_PWC_STC */
    struct {
      __REG8   OVIR           : 1;
      __REG8                  : 1;
      __REG8   EDIR           : 1;
      __REG8                  : 1;
      __REG8   OVIE           : 1;
      __REG8                  : 1;
      __REG8   EDIE           : 1;
      __REG8   ERR            : 1;
    } PWC;
} __btxstc_bits;

/* Timer Control Register 2 (TMCR2) */
typedef struct {
  __REG8   CKS3           : 1;
  __REG8                  : 7;
} __btxtmcr2_bits;

/* I/O Select Register (BTSEL0123) */
typedef struct {
  __REG8   SEL01          : 4;
  __REG8   SEL23          : 4;
} __btsel0123_bits;

/* I/O Select Register (BTSEL4567) */
typedef struct {
  __REG8   SEL45          : 4;
  __REG8   SEL67          : 4;
} __btsel4567_bits;

/* I/O Select Register (BTSEL89AB) */
typedef struct {
  __REG8   SEL89          : 4;
  __REG8   SELAB          : 4;
} __btsel89ab_bits;

/* I/O Select Register (BTSELCDEF) */
typedef struct {
  __REG8   SELCD          : 4;
  __REG8   SELEF          : 4;
} __btselcdef_bits;

/* Software-based Simultaneous Startup Register (BTSSSR) */
typedef struct {
  __REG16  SSSR0          : 1;
  __REG16  SSSR1          : 1;
  __REG16  SSSR2          : 1;
  __REG16  SSSR3          : 1;
  __REG16  SSSR4          : 1;
  __REG16  SSSR5          : 1;
  __REG16  SSSR6          : 1;
  __REG16  SSSR7          : 1;
  __REG16  SSSR8          : 1;
  __REG16  SSSR9          : 1;
  __REG16  SSSR10         : 1;
  __REG16  SSSR11         : 1;
  __REG16  SSSR12         : 1;
  __REG16  SSSR13         : 1;
  __REG16  SSSR14         : 1;
  __REG16  SSSR15         : 1;
} __btsssr_bits;

/* Interrupt Control Register (QICR) */
typedef struct {
  __REG16  QPCMIE         : 1;
  __REG16  QPCMF          : 1;
  __REG16  QPRCMIE        : 1;
  __REG16  QPRCMF         : 1;
  __REG16  OUZIE          : 1;
  __REG16  UFDF           : 1;
  __REG16  OFDF           : 1;
  __REG16  ZIIF           : 1;
  __REG16  CDCIE          : 1;
  __REG16  CDCF           : 1;
  __REG16  DIRPC          : 1;
  __REG16  DIROU          : 1;
  __REG16  QPCNRCMIE      : 1;
  __REG16  QPCNRCMF       : 1;
  __REG16                 : 2;
} __qdu_qicr_bits;

/* QPRC Control Register (QCR) */
typedef struct {
  __REG16  PCM            : 2;
  __REG16  RCM            : 2;
  __REG16  PSTP           : 1;
  __REG16  CGSC           : 1;
  __REG16  RSEL           : 1;
  __REG16  SWAP           : 1;
  __REG16  PCRM           : 2;
  __REG16  AES            : 2;
  __REG16  BES            : 2;
  __REG16  CGE            : 2;
} __qdu_qcr_bits;

/* QPRC Extension Control Register (QECR) */
typedef struct {
  __REG16  ORNGMD         : 1;
  __REG16  ORNGF          : 1;
  __REG16  ORNGIE         : 1;
  __REG16                 :13;
} __qdu_qecr_bits;

/* Quad Counter Position Rotation Count Register(QPRCRR) */
typedef struct {
  __REG32  QRCRR          :16;
  __REG32  QPCRR          :16;
} __qdu_qprcrr_bits;

/* A/D Status Register (ADSR) */
typedef struct {
  __REG8   SCS            : 1;
  __REG8   PCS            : 1;
  __REG8   PCNS           : 1;
  __REG8                  : 3;
  __REG8   FDAS           : 1;
  __REG8   ADSTP          : 1;
} __adc_adsr_bits;

/* A/D Control Register (ADCR) */
typedef struct {
  __REG8   OVRIE          : 1;
  __REG8   CMPIE          : 1;
  __REG8   PCIE           : 1;
  __REG8   SCIE           : 1;
  __REG8                  : 1;
  __REG8   CMPIF          : 1;
  __REG8   PCIF           : 1;
  __REG8   SCIF           : 1;
} __adc_adcr_bits;

/* Scan Conversion FIFO Stage Count Setup Register (SFNS) */
typedef struct {
  __REG8   SFS            : 4;
  __REG8                  : 4;
} __adc_sfns_bits;

/* Scan Conversion Control Register (SCCR) */
typedef struct {
  __REG8   SSTR           : 1;
  __REG8   SHEN           : 1;
  __REG8   RPT            : 1;
  __REG8                  : 1;
  __REG8   SFCLR          : 1;
  __REG8   SOVR           : 1;
  __REG8   SFUL           : 1;
  __REG8   SEMP           : 1;
} __adc_sccr_bits;

/* Scan Conversion FIFO Data Register (SCFD) */
typedef struct {
  __REG32  SC             : 5;
  __REG32                 : 3;
  __REG32  RS             : 2;
  __REG32                 : 2;
  __REG32  INVL           : 1;
  __REG32                 : 3;
  __REG32  SD             :16;
} __adc_scfd_bits;

/* Scan Conversion Input Selection Register 0 (SCIS0) */
typedef struct {
  __REG8   AN0            : 1;
  __REG8   AN1            : 1;
  __REG8   AN2            : 1;
  __REG8   AN3            : 1;
  __REG8   AN4            : 1;
  __REG8   AN5            : 1;
  __REG8   AN6            : 1;
  __REG8   AN7            : 1;
} __adc_scis0_bits;

/* Scan Conversion Input Selection Register 1 (SCIS1) */
typedef struct {
  __REG8   AN8            : 1;
  __REG8   AN9            : 1;
  __REG8   AN10           : 1;
  __REG8   AN11           : 1;
  __REG8   AN12           : 1;
  __REG8   AN13           : 1;
  __REG8   AN14           : 1;
  __REG8   AN15           : 1;
} __adc_scis1_bits;

/* Scan Conversion Input Selection Register 2 (SCIS2) */
typedef struct {
  __REG8   AN16           : 1;
  __REG8   AN17           : 1;
  __REG8   AN18           : 1;
  __REG8   AN19           : 1;
  __REG8   AN20           : 1;
  __REG8   AN21           : 1;
  __REG8   AN22           : 1;
  __REG8   AN23           : 1;
} __adc_scis2_bits;

/* Scan Conversion Input Selection Register 3 (SCIS3) */
typedef struct {
  __REG8   AN24           : 1;
  __REG8   AN25           : 1;
  __REG8   AN26           : 1;
  __REG8   AN27           : 1;
  __REG8   AN28           : 1;
  __REG8   AN29           : 1;
  __REG8   AN30           : 1;
  __REG8   AN31           : 1;
} __adc_scis3_bits;

/* Priority Conversion FIFO Stage Count Setup Register (PFNS) */
typedef struct {
  __REG8   PFS            : 2;
  __REG8                  : 2;
  __REG8   TEST           : 2;
  __REG8                  : 2;
} __adc_pfns_bits;

/* Priority Conversion Control Register (PCCR) */
typedef struct {
  __REG8   PSTR           : 1;
  __REG8   PHEN           : 1;
  __REG8   PEEN           : 1;
  __REG8   ESCE           : 1;
  __REG8   PFCLR          : 1;
  __REG8   POVR           : 1;
  __REG8   PFUL           : 1;
  __REG8   PEMP           : 1;
} __adc_pccr_bits;

/* Priority Conversion FIFO Data Register (PCFD) */
typedef struct {
  __REG32  PC             : 5;
  __REG32                 : 3;
  __REG32  RS             : 3;
  __REG32                 : 1;
  __REG32  INVL           : 1;
  __REG32                 : 3;
  __REG32  PD             :16;
} __adc_pcfd_bits;

/* Priority Conversion Input Selection Register (PCIS) */
typedef struct {
  __REG8   P1A            : 3;
  __REG8   P2A            : 5;
} __adc_pcis_bits;

/* A/D Comparison Control Register (CMPCR) */
typedef struct {
  __REG8   CCH            : 5;
  __REG8   CMD0           : 1;
  __REG8   CMD1           : 1;
  __REG8   CMPEN          : 1;
} __adc_cmpcr_bits;

/* A/D Comparison Control Register (CMPD) */
typedef struct {
  __REG16                 : 4;
  __REG16  CMAD           :12;
} __adc_cmpd_bits;

/* Sampling Time Selection Register 0 (ADSS0) */
typedef struct {
  __REG8   TS0            : 1;
  __REG8   TS1            : 1;
  __REG8   TS2            : 1;
  __REG8   TS3            : 1;
  __REG8   TS4            : 1;
  __REG8   TS5            : 1;
  __REG8   TS6            : 1;
  __REG8   TS7            : 1;
} __adc_adss0_bits;

/* Sampling Time Selection Register 1 (ADSS1) */
typedef struct {
  __REG8   TS8            : 1;
  __REG8   TS9            : 1;
  __REG8   TS10           : 1;
  __REG8   TS11           : 1;
  __REG8   TS12           : 1;
  __REG8   TS13           : 1;
  __REG8   TS14           : 1;
  __REG8   TS15           : 1;
} __adc_adss1_bits;

/* Sampling Time Selection Register 2 (ADSS2) */
typedef struct {
  __REG8   TS16           : 1;
  __REG8   TS17           : 1;
  __REG8   TS18           : 1;
  __REG8   TS19           : 1;
  __REG8   TS20           : 1;
  __REG8   TS21           : 1;
  __REG8   TS22           : 1;
  __REG8   TS23           : 1;
} __adc_adss2_bits;

/* Sampling Time Selection Register 3 (ADSS3) */
typedef struct {
  __REG8   TS24           : 1;
  __REG8   TS25           : 1;
  __REG8   TS26           : 1;
  __REG8   TS27           : 1;
  __REG8   TS28           : 1;
  __REG8   TS29           : 1;
  __REG8   TS30           : 1;
  __REG8   TS31           : 1;
} __adc_adss3_bits;

/* Sampling Time Setup Register 0 (ADST0) */
typedef struct {
  __REG8   ST0            : 5;
  __REG8   STX0           : 3;
} __adc_adst0_bits;

/* Sampling Time Setup Register 0 (ADST1) */
typedef struct {
  __REG8   ST1            : 5;
  __REG8   STX1           : 3;
} __adc_adst1_bits;

/* Comparison Time Setup Register (ADCT) */
typedef struct {
  __REG8   CT   : 8;
} __adc_adct_bits;

/* Priority Conversion Timer Trigger Selection Register (PRTSL) */
typedef struct {
  __REG8   PRTSL          : 4;
  __REG8                  : 4;
} __adc_prtsl_bits;

/* Scan Conversion Timer Trigger Selection Register (SCTSL) */
typedef struct {
  __REG8   SCTSL          : 4;
  __REG8                  : 4;
} __adc_sctsl_bits;

/* A/D Operation Enable Setup Register (ADCEN) */
typedef struct {
  __REG8   ENBL           : 1;
  __REG8   READY          : 1;
  __REG8                  : 2;
  __REG8   CYCLSL         : 2;
  __REG8                  : 2;
} __adc_adcen_bits;

/* High-speed CR oscillation Frequency Division Setup Register (MCR_PSR) */
typedef struct {
  __REG8   CSR            : 2;
  __REG8                  : 6;
} __mcr_psr_bits;

/* High-speed CR oscillation Frequency Trimming Register (MCR_FTRM) */
typedef struct {
  __REG16  TRD            : 8;
  __REG16                 : 8;
} __mcr_ftrm_bits;

/* Enable Interrupt Request Register [ENIR] */
typedef struct {
  __REG32  EN0            : 1;
  __REG32  EN1            : 1;
  __REG32  EN2            : 1;
  __REG32  EN3            : 1;
  __REG32  EN4            : 1;
  __REG32  EN5            : 1;
  __REG32  EN6            : 1;
  __REG32  EN7            : 1;
  __REG32  EN8            : 1;
  __REG32  EN9            : 1;
  __REG32  EN10           : 1;
  __REG32  EN11           : 1;
  __REG32  EN12           : 1;
  __REG32  EN13           : 1;
  __REG32  EN14           : 1;
  __REG32  EN15           : 1;
  __REG32  EN16           : 1;
  __REG32  EN17           : 1;
  __REG32  EN18           : 1;
  __REG32  EN19           : 1;
  __REG32  EN20           : 1;
  __REG32  EN21           : 1;
  __REG32  EN22           : 1;
  __REG32  EN23           : 1;
  __REG32  EN24           : 1;
  __REG32  EN25           : 1;
  __REG32  EN26           : 1;
  __REG32  EN27           : 1;
  __REG32  EN28           : 1;
  __REG32  EN29           : 1;
  __REG32  EN30           : 1;
  __REG32  EN31           : 1;
} __enir_bits;

/* External Interrupt Request Register [EIRR] */
typedef struct {
  __REG32  ER0            : 1;
  __REG32  ER1            : 1;
  __REG32  ER2            : 1;
  __REG32  ER3            : 1;
  __REG32  ER4            : 1;
  __REG32  ER5            : 1;
  __REG32  ER6            : 1;
  __REG32  ER7            : 1;
  __REG32  ER8            : 1;
  __REG32  ER9            : 1;
  __REG32  ER10           : 1;
  __REG32  ER11           : 1;
  __REG32  ER12           : 1;
  __REG32  ER13           : 1;
  __REG32  ER14           : 1;
  __REG32  ER15           : 1;
  __REG32  ER16           : 1;
  __REG32  ER17           : 1;
  __REG32  ER18           : 1;
  __REG32  ER19           : 1;
  __REG32  ER20           : 1;
  __REG32  ER21           : 1;
  __REG32  ER22           : 1;
  __REG32  ER23           : 1;
  __REG32  ER24           : 1;
  __REG32  ER25           : 1;
  __REG32  ER26           : 1;
  __REG32  ER27           : 1;
  __REG32  ER28           : 1;
  __REG32  ER29           : 1;
  __REG32  ER30           : 1;
  __REG32  ER31           : 1;
} __eirr_bits;

/* External Interrupt Clear Register [EICL] */
typedef struct {
  __REG32  ECL0           : 1;
  __REG32  ECL1           : 1;
  __REG32  ECL2           : 1;
  __REG32  ECL3           : 1;
  __REG32  ECL4           : 1;
  __REG32  ECL5           : 1;
  __REG32  ECL6           : 1;
  __REG32  ECL7           : 1;
  __REG32  ECL8           : 1;
  __REG32  ECL9           : 1;
  __REG32  ECL10          : 1;
  __REG32  ECL11          : 1;
  __REG32  ECL12          : 1;
  __REG32  ECL13          : 1;
  __REG32  ECL14          : 1;
  __REG32  ECL15          : 1;
  __REG32  ECL16          : 1;
  __REG32  ECL17          : 1;
  __REG32  ECL18          : 1;
  __REG32  ECL19          : 1;
  __REG32  ECL20          : 1;
  __REG32  ECL21          : 1;
  __REG32  ECL22          : 1;
  __REG32  ECL23          : 1;
  __REG32  ECL24          : 1;
  __REG32  ECL25          : 1;
  __REG32  ECL26          : 1;
  __REG32  ECL27          : 1;
  __REG32  ECL28          : 1;
  __REG32  ECL29          : 1;
  __REG32  ECL30          : 1;
  __REG32  ECL31          : 1;
} __eicl_bits;

/* External Interrupt Level Register [ELVR] */
typedef struct {
  __REG32  LA0            : 1;
  __REG32  LB0            : 1;
  __REG32  LA1            : 1;
  __REG32  LB1            : 1;
  __REG32  LA2            : 1;
  __REG32  LB2            : 1;
  __REG32  LA3            : 1;
  __REG32  LB3            : 1;
  __REG32  LA4            : 1;
  __REG32  LB4            : 1;
  __REG32  LA5            : 1;
  __REG32  LB5            : 1;
  __REG32  LA6            : 1;
  __REG32  LB6            : 1;
  __REG32  LA7            : 1;
  __REG32  LB7            : 1;
  __REG32  LA8            : 1;
  __REG32  LB8            : 1;
  __REG32  LA9            : 1;
  __REG32  LB9            : 1;
  __REG32  LA10           : 1;
  __REG32  LB10           : 1;
  __REG32  LA11           : 1;
  __REG32  LB11           : 1;
  __REG32  LA12           : 1;
  __REG32  LB12           : 1;
  __REG32  LA13           : 1;
  __REG32  LB13           : 1;
  __REG32  LA14           : 1;
  __REG32  LB14           : 1;
  __REG32  LA15           : 1;
  __REG32  LB15           : 1;
} __elvr_bits;

/* External Interrupt Level Register [ELVR] */
typedef struct {
  __REG32  LA16           : 1;
  __REG32  LB16           : 1;
  __REG32  LA17           : 1;
  __REG32  LB17           : 1;
  __REG32  LA18           : 1;
  __REG32  LB18           : 1;
  __REG32  LA19           : 1;
  __REG32  LB19           : 1;
  __REG32  LA20           : 1;
  __REG32  LB20           : 1;
  __REG32  LA21           : 1;
  __REG32  LB21           : 1;
  __REG32  LA22           : 1;
  __REG32  LB22           : 1;
  __REG32  LA23           : 1;
  __REG32  LB23           : 1;
  __REG32  LA24           : 1;
  __REG32  LB24           : 1;
  __REG32  LA25           : 1;
  __REG32  LB25           : 1;
  __REG32  LA26           : 1;
  __REG32  LB26           : 1;
  __REG32  LA27           : 1;
  __REG32  LB27           : 1;
  __REG32  LA28           : 1;
  __REG32  LB28           : 1;
  __REG32  LA29           : 1;
  __REG32  LB29           : 1;
  __REG32  LA30           : 1;
  __REG32  LB30           : 1;
  __REG32  LA31           : 1;
  __REG32  LB31           : 1;
} __elvr1_bits;

/* Non Maskable Interrupt Request Register [NMIRR] */
typedef struct {
  __REG16  NR0            : 1;
  __REG16                 :15;
} __nmirr_bits;

/* Non Maskable Interrupt Clear Register [NMICL] */
typedef struct {
  __REG16  NCL0           : 1;
  __REG16                 :15;
} __nmicl_bits;

/* DMA Request Selection Register (DRQSEL) */
typedef struct {
  __REG32  DRQSEL0        : 1;
  __REG32  DRQSEL1        : 1;
  __REG32  DRQSEL2        : 1;
  __REG32  DRQSEL3        : 1;
  __REG32  DRQSEL4        : 1;
  __REG32  DRQSEL5        : 1;
  __REG32  DRQSEL6        : 1;
  __REG32  DRQSEL7        : 1;
  __REG32  DRQSEL8        : 1;
  __REG32  DRQSEL9        : 1;
  __REG32  DRQSEL10       : 1;
  __REG32  DRQSEL11       : 1;
  __REG32  DRQSEL12       : 1;
  __REG32  DRQSEL13       : 1;
  __REG32  DRQSEL14       : 1;
  __REG32  DRQSEL15       : 1;
  __REG32  DRQSEL16       : 1;
  __REG32  DRQSEL17       : 1;
  __REG32  DRQSEL18       : 1;
  __REG32  DRQSEL19       : 1;
  __REG32  DRQSEL20       : 1;
  __REG32  DRQSEL21       : 1;
  __REG32  DRQSEL22       : 1;
  __REG32  DRQSEL23       : 1;
  __REG32  DRQSEL24       : 1;
  __REG32  DRQSEL25       : 1;
  __REG32  DRQSEL26       : 1;
  __REG32  DRQSEL27       : 1;
  __REG32  DRQSEL28       : 1;
  __REG32  DRQSEL29       : 1;
  __REG32  DRQSEL30       : 1;
  __REG32  DRQSEL31       : 1;
} __drqsel_bits;

/* USB Odd Packet Size DMA Enable Register (ODDPKS)) */
typedef struct {
  __REG32                 :24;
  __REG32  ODDPKS0        : 1;
  __REG32  ODDPKS1        : 1;
  __REG32  ODDPKS2        : 1;
  __REG32  ODDPKS3        : 1;
  __REG32  ODDPKS4        : 1;
  __REG32                 : 3;
} __oddpks_bits;

/* EXC02 Batch Read Register (EXC02MON) EXC02MON indicates */
typedef struct {
  __REG32  NMI            : 1;
  __REG32  HWINT          : 1;
  __REG32                 :30;
} __exc02mon_bits;

/* IRQ00 Batch Read Register (IRQ00MON) */
typedef struct {
  __REG32  FCSINT         : 1;
  __REG32                 :31;
} __irqmon0_bits;

/* IRQ01 Batch Read Register (IRQ01MON) */
typedef struct {
  __REG32  SWWDTINT       : 1;
  __REG32                 :31;
} __irqmon1_bits;

/* IRQ02 Batch Read Register (IRQ02MON) */
typedef struct {
  __REG32  LVDINT         : 1;
  __REG32                 :31;
} __irqmon2_bits;

/* IRQ03 Batch Read Register (IRQ03MON) */
typedef struct {
  __REG32  WAVE0INT       : 4;
  __REG32  WAVE1INT       : 4;
  __REG32  WAVE2INT       : 4;
  __REG32                 :20;
} __irqmon3_bits;

/* IRQ04 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  EXTINT00       : 1;
  __REG32  EXTINT01       : 1;
  __REG32  EXTINT02       : 1;
  __REG32  EXTINT03       : 1;
  __REG32  EXTINT04       : 1;
  __REG32  EXTINT05       : 1;
  __REG32  EXTINT06       : 1;
  __REG32  EXTINT07       : 1;
  __REG32                 :24;
} __irqmon4_bits;

/* IRQ05 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  EXTINT08       : 1;
  __REG32  EXTINT09       : 1;
  __REG32  EXTINT10       : 1;
  __REG32  EXTINT11       : 1;
  __REG32  EXTINT12       : 1;
  __REG32  EXTINT13       : 1;
  __REG32  EXTINT14       : 1;
  __REG32  EXTINT15       : 1;
  __REG32  EXTINT16       : 1;
  __REG32  EXTINT17       : 1;
  __REG32  EXTINT18       : 1;
  __REG32  EXTINT19       : 1;
  __REG32  EXTINT20       : 1;
  __REG32  EXTINT21       : 1;
  __REG32  EXTINT22       : 1;
  __REG32  EXTINT23       : 1;
  __REG32  EXTINT24       : 1;
  __REG32  EXTINT25       : 1;
  __REG32  EXTINT26       : 1;
  __REG32  EXTINT27       : 1;
  __REG32  EXTINT28       : 1;
  __REG32  EXTINT29       : 1;
  __REG32  EXTINT30       : 1;
  __REG32  EXTINT31       : 1;
  __REG32                 : 8;
} __irqmon5_bits;

/* IRQ06 Batch Read Register (IRQ06MON) */
typedef struct {
  __REG32  TIMINT         : 2;
  __REG32  QUD0INT        : 6;
  __REG32  QUD1INT        : 6;
  __REG32  QUD2INT        : 6;
  __REG32                 :12;
} __irqmon6_bits;

/* IRQ07/09/11/13/15/17/19/21 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  MFSINT         : 1;
  __REG32                 :31;
} __irqmon7_bits;

/* IRQ08/10/12/14/16/18/20/22 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  MFSINT         : 2;
  __REG32                 :30;
} __irqmon8_bits;

/* IRQ23 Batch Read Register (IRQ23MON) */
typedef struct {
  __REG32  PPG0INT        : 1;
  __REG32  PPG2INT        : 1;
  __REG32  PPG4INT        : 1;
  __REG32  PPG8INT        : 1;
  __REG32  PPG10INT       : 1;
  __REG32  PPG12INT       : 1;
  __REG32  PPG16INT       : 1;
  __REG32  PPG18INT       : 1;
  __REG32  PPG20INT       : 1;
  __REG32                 :23;
} __irqmon23_bits;

/* IRQ24 Batch Read Register (IRQ24MON) */
typedef struct {
  __REG32  MOSCINT        : 1;
  __REG32  SOSCINT        : 1;
  __REG32  MPLLINT        : 1;
  __REG32  UPLLINT        : 1;
  __REG32  WCINT          : 1;
  __REG32  RTC            : 1;
  __REG32                 :26;
} __irqmon24_bits;

/* IRQ25/26/27 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  ADCINT         : 4;
  __REG32                 :28;
} __irqmon25_bits;

/* IRQ28 Batch Read Register (IRQ28MON) */
typedef struct {
  __REG32  FRT0INT        : 6;
  __REG32  FRT1INT        : 6;
  __REG32  FRT2INT        : 6;
  __REG32                 :14;
} __irqmon28_bits;

/* IRQ29 Batch Read Register (IRQ29MON) */
typedef struct {
  __REG32  ICU0INT        : 4;
  __REG32  ICU1INT        : 4;
  __REG32  ICU2INT        : 4;
  __REG32                 :20;
} __irqmon29_bits;

/* IRQ30 Batch Read Register (IRQ30MON) */
typedef struct {
  __REG32  OCU0INT        : 6;
  __REG32  OCU1INT        : 6;
  __REG32  OCU2INT        : 6;
  __REG32                 :14;
} __irqmon30_bits;

/* IRQ31 Batch Read Register (IRQ31MON) */
typedef struct {
  __REG32  BT0INT0        : 1;
  __REG32  BT0INT1        : 1;
  __REG32  BT1INT0        : 1;
  __REG32  BT1INT1        : 1;
  __REG32  BT2INT0        : 1;
  __REG32  BT2INT1        : 1;
  __REG32  BT3INT0        : 1;
  __REG32  BT3INT1        : 1;
  __REG32  BT4INT0        : 1;
  __REG32  BT4INT1        : 1;
  __REG32  BT5INT0        : 1;
  __REG32  BT5INT1        : 1;
  __REG32  BT6INT0        : 1;
  __REG32  BT6INT1        : 1;
  __REG32  BT7INT0        : 1;
  __REG32  BT7INT1        : 1;
  __REG32                 :16;
} __irqmon31_bits;

/* IRQ32 Batch Read Register (IRQ32MON) */
typedef struct {
  __REG32                 : 1;
  __REG32  MAC0SBD        : 1;
  __REG32  MAC0PMI        : 1;
  __REG32  MAC0LPI        : 1;
  __REG32                 :28;
} __irqmon32_bits;

/* IRQ34 Batch Read Register (IRQ34MON) */
typedef struct {
  __REG32  USB0INT        : 5;
  __REG32                 :27;
} __irqmon34_bits;

/* IRQ35 Batch Read Register (IRQ35MON) */
typedef struct {
  __REG32  USB0INT        : 6;
  __REG32                 :26;
} __irqmon35_bits;

/* IRQ36 Batch Read Register (IRQ36MON) */
typedef struct {
  __REG32  USB1INT        : 5;
  __REG32                 :27;
} __irqmon36_bits;

/* IRQ37 Batch Read Register (IRQ37MON) */
typedef struct {
  __REG32  USB1INT        : 6;
  __REG32                 :26;
} __irqmon37_bits;

/* IRQ38/39/40/41/42/43/44/45 Batch Read Register (IRQxxMON) */
typedef struct {
  __REG32  DMAINT         : 1;
  __REG32                 :31;
} __irqmon38_bits;

/* IRQ46 Batch Read Register (IRQ46MON) */
typedef struct {
  __REG32  BT8INT0        : 1;
  __REG32  BT8INT1        : 1;
  __REG32  BT9INT0        : 1;
  __REG32  BT9INT1        : 1;
  __REG32  BT10INT0       : 1;
  __REG32  BT10INT1       : 1;
  __REG32  BT11INT0       : 1;
  __REG32  BT11INT1       : 1;
  __REG32  BT12INT0       : 1;
  __REG32  BT12INT1       : 1;
  __REG32  BT13INT0       : 1;
  __REG32  BT13INT1       : 1;
  __REG32  BT14INT0       : 1;
  __REG32  BT14INT1       : 1;
  __REG32  BT15INT0       : 1;
  __REG32  BT15INT1       : 1;
  __REG32                 :16;
} __irqmon46_bits;

/* DMA Request Select Register 1 (DRQSEL1) */
typedef struct {
  __REG32  USB1EP1DRQ     : 1;
  __REG32  USB1EP2DRQ     : 1;
  __REG32  USB1EP3DRQ     : 1;
  __REG32  USB1EP4DRQ     : 1;
  __REG32  USB1EP5DRQ     : 1;
  __REG32                 :27;
} __drqsel1_bits;

/* IRQ46 Batch Read Register (IRQ46MON) */
typedef struct {
  __REG32  ESEL10         : 4;
  __REG32  ESEL11         : 4;
  __REG32  ESEL24         : 4;
  __REG32  ESEL25         : 4;
  __REG32  ESEL26         : 4;
  __REG32  ESEL27         : 4;
  __REG32  ESEL30         : 4;
  __REG32  ESEL31         : 4;
} __dqesel_bits;

/* Port0 Function Setting Register (PFR0) */
/* Port0 Pull-up Setting Register (PCR0) */
/* Port0 input/output Direction Setting Register (DDR0) */
/* Port0 Input Data Register (PDIR0) */
/* Port0 Pseudo Open Drain Setting Register (PZR0) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32  P6            : 1;
  __REG32  P7            : 1;
  __REG32  P8            : 1;
  __REG32  P9            : 1;
  __REG32                :22;
} __port0_bits;

/* Port1 Function Setting Register (PFR1) */
/* Port1 Pull-up Setting Register (PCR1) */
/* Port1 input/output Direction Setting Register (DDR1) */
/* Port1 Input Data Register (PDIR1) */
/* Port1 Output Data Register (PDOR1) */
/* Port1 Pseudo Open Drain Setting Register (PZR1) */
typedef struct {
  __REG32  P0           : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32  P4           : 1;
  __REG32  P5           : 1;
  __REG32  P6           : 1;
  __REG32  P7           : 1;
  __REG32  P8           : 1;
  __REG32  P9           : 1;
  __REG32  PA           : 1;
  __REG32  PB           : 1;
  __REG32  PC           : 1;
  __REG32  PD           : 1;
  __REG32  PE           : 1;
  __REG32  PF           : 1;
  __REG32               :16;
} __port1_bits;

/* Port2 Function Setting Register (PFR2) */
/* Port2 Pull-up Setting Register (PCR2) */
/* Port2 input/output Direction Setting Register (DDR2) */
/* Port2 Input Data Register (PDIR2) */
/* Port2 Output Data Register (PDOR2) */
/* Port2 Pseudo Open Drain Setting Register (PZR2) */
typedef struct {
  __REG32  P0           : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32  P4           : 1;
  __REG32  P5           : 1;
  __REG32  P6           : 1;
  __REG32  P7           : 1;
  __REG32  P8           : 1;
  __REG32  P9           : 1;
  __REG32               :22;
} __port2_bits;

/* Port3 Function Setting Register (PFR3) */
/* Port3 Pull-up Setting Register (PCR3) */
/* Port3 input/output Direction Setting Register (DDR3) */
/* Port3 Input Data Register (PDIR3) */
/* Port3 Output Data Register (PDOR3) */
/* Port3 Pseudo Open Drain Setting Register (PZR3) */
typedef struct {
  __REG32  P0           : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32  P4           : 1;
  __REG32  P5           : 1;
  __REG32  P6           : 1;
  __REG32  P7           : 1;
  __REG32  P8           : 1;
  __REG32  P9           : 1;
  __REG32  PA           : 1;
  __REG32  PB           : 1;
  __REG32  PC           : 1;
  __REG32  PD           : 1;
  __REG32  PE           : 1;
  __REG32  PF           : 1;
  __REG32               :16;
} __port3_bits;

/* Port4 Function Setting Register (PFR4) */
/* Port4 Pull-up Setting Register (PCR4) */
/* Port4 input/output Direction Setting Register (DDR4) */
/* Port4 Input Data Register (PDIR4) */
/* Port4 Output Data Register (PDOR4) */
/* Port4 Pseudo Open Drain Setting Register (PZR4) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32  P6            : 1;
  __REG32  P7            : 1;
  __REG32  P8            : 1;
  __REG32  P9            : 1;
  __REG32  PA            : 1;
  __REG32  PB            : 1;
  __REG32  PC            : 1;
  __REG32  PD            : 1;
  __REG32  PE            : 1;
  __REG32                :17;
} __port4_bits;

/* Port5 Function Setting Register (PFR5) */
/* Port5 Pull-up Setting Register (PCR5) */
/* Port5 input/output Direction Setting Register (DDR5) */
/* Port5 Input Data Register (PDIR5) */
/* Port5 Output Data Register (PDOR5) */
/* Port5 Pseudo Open Drain Setting Register (PZR5) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32  P6            : 1;
  __REG32  P7            : 1;
  __REG32  P8            : 1;
  __REG32  P9            : 1;
  __REG32  PA            : 1;
  __REG32  PB            : 1;
  __REG32  PC            : 1;
  __REG32  PD            : 1;
  __REG32                :18;
} __port5_bits;

/* Port6 Function Setting Register (PFR6) */
/* Port6 Pull-up Setting Register (PCR6) */
/* Port6 input/output Direction Setting Register (DDR6) */
/* Port6 Input Data Register (PDIR6) */
/* Port6 Output Data Register (PDOR6) */
/* Port6 Pseudo Open Drain Setting Register (PZR6) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32                :29;
} __port6_bits;

/* Port7 Function Setting Register (PFR7) */
/* Port7 Pull-up Setting Register (PCR7) */
/* Port7 input/output Direction Setting Register (DDR7) */
/* Port7 Input Data Register (PDIR7) */
/* Port7 Output Data Register (PDOR7) */
/* Port7 Pseudo Open Drain Setting Register (PZR7) */
typedef struct {
  __REG32  P0           : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32  P4           : 1;
  __REG32  P5           : 1;
  __REG32  P6           : 1;
  __REG32  P7           : 1;
  __REG32  P8           : 1;
  __REG32  P9           : 1;
  __REG32  PA           : 1;
  __REG32  PB           : 1;
  __REG32  PC           : 1;
  __REG32  PD           : 1;
  __REG32  PE           : 1;
  __REG32  PF           : 1;
  __REG32               :16;
} __port7_bits;

/* Port8 Function Setting Register (PFR8) */
/* Port8 input/output Direction Setting Register (DDR8) */
/* Port8 Input Data Register (PDIR8) */
/* Port8 Output Data Register (PDOR8) */
/* Port8 Pseudo Open Drain Setting Register (PZR8) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32                :28;
} __port8_bits;

/* Port9 Function Setting Register (PFR9) */
/* Port9 input/output Direction Setting Register (DDR9) */
/* Port9 Input Data Register (PDIR9) */
/* Port9 Output Data Register (PDOR9) */
/* Port9 Pseudo Open Drain Setting Register (PZR9) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32                :26;
} __port9_bits;

/* PortA Function Setting Register (PFRA) */
/* PortA input/output Direction Setting Register (DDRA) */
/* PortA Input Data Register (PDIRA) */
/* PortA Output Data Register (PDORA) */
/* PortA Pseudo Open Drain Setting Register (PZRA) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32                :26;
} __porta_bits;

/* PortB Function Setting Register (PFRB) */
/* PortB input/output Direction Setting Register (DDRB) */
/* PortB Input Data Register (PDIRB) */
/* PortB Output Data Register (PDORB) */
/* PortB Pseudo Open Drain Setting Register (PZRB) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32  P6            : 1;
  __REG32  P7            : 1;
  __REG32                :24;
} __portb_bits;

/* PortC Function Setting Register (PFRC) */
/* PortC input/output Direction Setting Register (DDRC) */
/* PortC Input Data Register (PDIRC) */
/* PortC Output Data Register (PDORC) */
/* PortC Pseudo Open Drain Setting Register (PZRC) */
typedef struct {
  __REG32  P0           : 1;
  __REG32  P1           : 1;
  __REG32  P2           : 1;
  __REG32  P3           : 1;
  __REG32  P4           : 1;
  __REG32  P5           : 1;
  __REG32  P6           : 1;
  __REG32  P7           : 1;
  __REG32  P8           : 1;
  __REG32  P9           : 1;
  __REG32  PA           : 1;
  __REG32  PB           : 1;
  __REG32  PC           : 1;
  __REG32  PD           : 1;
  __REG32  PE           : 1;
  __REG32  PF           : 1;
  __REG32               :16;
} __portc_bits;

/* PortD Function Setting Register (PFRD) */
/* PortD input/output Direction Setting Register (DDRD) */
/* PortD Input Data Register (PDIRD) */
/* PortD Output Data Register (PDORD) */
/* PortD Pseudo Open Drain Setting Register (PZRD) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32                :28;
} __portd_bits;

/* PortE Function Setting Register (PFRE) */
/* PortE input/output Direction Setting Register (DDRE) */
/* PortE Input Data Register (PDIRE) */
/* PortE Output Data Register (PDORE) */
/* PortB Pseudo Open Drain Setting Register (PZRE) */
typedef struct {
  __REG32  P0            : 1;
  __REG32                : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32                :28;
} __porte_bits;

/* PortF Function Setting Register (PFRF) */
/* PortF input/output Direction Setting Register (DDRF) */
/* PortF Input Data Register (PDIRF) */
/* PortF Output Data Register (PDORF) */
/* PortF Pseudo Open Drain Setting Register (PZRF) */
typedef struct {
  __REG32  P0            : 1;
  __REG32  P1            : 1;
  __REG32  P2            : 1;
  __REG32  P3            : 1;
  __REG32  P4            : 1;
  __REG32  P5            : 1;
  __REG32  P6            : 1;
  __REG32                :25;
} __portf_bits;

/* Analog Input Setting Register (ADE) */
typedef struct {
  __REG32  AN00           : 1;
  __REG32  AN01           : 1;
  __REG32  AN02           : 1;
  __REG32  AN03           : 1;
  __REG32  AN04           : 1;
  __REG32  AN05           : 1;
  __REG32  AN06           : 1;
  __REG32  AN07           : 1;
  __REG32  AN08           : 1;
  __REG32  AN09           : 1;
  __REG32  AN10           : 1;
  __REG32  AN11           : 1;
  __REG32  AN12           : 1;
  __REG32  AN13           : 1;
  __REG32  AN14           : 1;
  __REG32  AN15           : 1;
  __REG32  AN16           : 1;
  __REG32  AN17           : 1;
  __REG32  AN18           : 1;
  __REG32  AN19           : 1;
  __REG32  AN20           : 1;
  __REG32  AN21           : 1;
  __REG32  AN22           : 1;
  __REG32  AN23           : 1;
  __REG32  AN24           : 1;
  __REG32  AN25           : 1;
  __REG32  AN26           : 1;
  __REG32  AN27           : 1;
  __REG32  AN28           : 1;
  __REG32  AN29           : 1;
  __REG32  AN30           : 1;
  __REG32  AN31           : 1;
} __ade_bits;

/* Special Port Setting Register (SPSR) */
typedef struct {
  __REG32  SUBXC          : 1;
  __REG32                 : 1;
  __REG32  MAINXC         : 1;
  __REG32                 : 1;
  __REG32  USB0C          : 1;
  __REG32  USB1C          : 1;
  __REG32                 :26;
} __spsr_bits;

/* Extended Pin Function Setting Register 00 (EPFR00) */
typedef struct {
  __REG32  NMIS           : 1;
  __REG32  CROUTE         : 2;
  __REG32                 : 1;
  __REG32  RTCCOE         : 2;
  __REG32  SUBOUTE        : 2;
  __REG32                 : 1;
  __REG32  USBP0E         : 1;
  __REG32                 : 3;
  __REG32  USBP1E         : 1;
  __REG32                 : 2;
  __REG32  JTAGEN0B       : 1;
  __REG32  JTAGEN1S       : 1;
  __REG32                 : 6;
  __REG32  TRC0E          : 1;
  __REG32  TRC1E          : 1;
  __REG32                 : 6;
} __epfr00_bits;

/* Extended Pin Function Setting Register 01 (EPFR01) */
typedef struct {
  __REG32  RTO00E         : 2;
  __REG32  RTO01E         : 2;
  __REG32  RTO02E         : 2;
  __REG32  RTO03E         : 2;
  __REG32  RTO04E         : 2;
  __REG32  RTO05E         : 2;
  __REG32  DTTI0C         : 1;
  __REG32                 : 3;
  __REG32  DTTI0S         : 2;
  __REG32  FRCK0S         : 2;
  __REG32  IC00S          : 3;
  __REG32  IC01S          : 3;
  __REG32  IC02S          : 3;
  __REG32  IC03S          : 3;
} __epfr01_bits;

/* Extended Pin Function Setting Register 02 (EPFR02) */
typedef struct {
  __REG32  RTO10E         : 2;
  __REG32  RTO11E         : 2;
  __REG32  RTO12E         : 2;
  __REG32  RTO13E         : 2;
  __REG32  RTO14E         : 2;
  __REG32  RTO15E         : 2;
  __REG32  DTTI1C         : 1;
  __REG32                 : 3;
  __REG32  DTTI1S         : 2;
  __REG32  FRCK1S         : 2;
  __REG32  IC10S          : 3;
  __REG32  IC11S          : 3;
  __REG32  IC12S          : 3;
  __REG32  IC13S          : 3;
} __epfr02_bits;

/* Extended Pin Function Setting Register 03 (EPFR03) */
typedef struct {
  __REG32  RTO20E         : 2;
  __REG32  RTO21E         : 2;
  __REG32  RTO22E         : 2;
  __REG32  RTO23E         : 2;
  __REG32  RTO24E         : 2;
  __REG32  RTO25E         : 2;
  __REG32  DTTI2C         : 1;
  __REG32                 : 3;
  __REG32  DTTI2S         : 2;
  __REG32  FRCK2S         : 2;
  __REG32  IC20S          : 3;
  __REG32  IC21S          : 3;
  __REG32  IC22S          : 3;
  __REG32  IC23S          : 3;
} __epfr03_bits;

/* Extended Pin Function Setting Register 04 (EPFR04) */
typedef struct {
  __REG32                 : 2;
  __REG32  TIOA0E         : 2;
  __REG32  TIOB0S         : 2;
  __REG32                 : 2;
  __REG32  TIOA1S         : 2;
  __REG32  TIOA1E         : 2;
  __REG32  TIOB1S         : 2;
  __REG32                 : 4;
  __REG32  TIOA2E         : 2;
  __REG32  TIOB2S         : 2;
  __REG32                 : 2;
  __REG32  TIOA3S         : 2;
  __REG32  TIOA3E         : 2;
  __REG32  TIOB3S         : 2;
  __REG32                 : 2;
} __epfr04_bits;

/* Extended Pin Function Setting Register 05 (EPFR05) */
typedef struct {
  __REG32                 : 2;
  __REG32  TIOA4E         : 2;
  __REG32  TIOB4S         : 2;
  __REG32                 : 2;
  __REG32  TIOA5S         : 2;
  __REG32  TIOA5E         : 2;
  __REG32  TIOB5S         : 2;
  __REG32                 : 4;
  __REG32  TIOA6E         : 2;
  __REG32  TIOB6S         : 2;
  __REG32                 : 2;
  __REG32  TIOA7S         : 2;
  __REG32  TIOA7E         : 2;
  __REG32  TIOB7S         : 2;
  __REG32                 : 2;
} __epfr05_bits;

/* Extended Pin Function Setting Register 06 (EPFR06) */
typedef struct {
  __REG32  EINT00S        : 2;
  __REG32  EINT01S        : 2;
  __REG32  EINT02S        : 2;
  __REG32  EINT03S        : 2;
  __REG32  EINT04S        : 2;
  __REG32  EINT05S        : 2;
  __REG32  EINT06S        : 2;
  __REG32  EINT07S        : 2;
  __REG32  EINT08S        : 2;
  __REG32  EINT09S        : 2;
  __REG32  EINT10S        : 2;
  __REG32  EINT11S        : 2;
  __REG32  EINT12S        : 2;
  __REG32  EINT13S        : 2;
  __REG32  EINT14S        : 2;
  __REG32  EINT15S        : 2;
} __epfr06_bits;

/* Extended Pin Function Setting Register 07 (EPFR07) */
typedef struct {
  __REG32                 : 4;
  __REG32  SIN0S          : 2;
  __REG32  SOT0B          : 2;
  __REG32  SCK0B          : 2;
  __REG32  SIN1S          : 2;
  __REG32  SOT1B          : 2;
  __REG32  SCK1B          : 2;
  __REG32  SIN2S          : 2;
  __REG32  SOT2B          : 2;
  __REG32  SCK2B          : 2;
  __REG32  SIN3S          : 2;
  __REG32  SOT3B          : 2;
  __REG32  SCK3B          : 2;
  __REG32                 : 4;
} __epfr07_bits;

/* Extended Pin Function Setting Register 08 (EPFR08) */
typedef struct {
  __REG32  RTS4E          : 2;
  __REG32  CTS4S          : 2;
  __REG32  SIN4S          : 2;
  __REG32  SOT4B          : 2;
  __REG32  SCK4B          : 2;
  __REG32  SIN5S          : 2;
  __REG32  SOT5B          : 2;
  __REG32  SCK5B          : 2;
  __REG32  SIN6S          : 2;
  __REG32  SOT6B          : 2;
  __REG32  SCK6B          : 2;
  __REG32  SIN7S          : 2;
  __REG32  SOT7B          : 2;
  __REG32  SCK7B          : 2;
  __REG32                 : 4;
} __epfr08_bits;

/* Extended Pin Function Setting Register 09 (EPFR09) */
typedef struct {
  __REG32  QAIN0S         : 2;
  __REG32  QBIN0S         : 2;
  __REG32  QZIN0S         : 2;
  __REG32  QAIN1S         : 2;
  __REG32  QBIN1S         : 2;
  __REG32  QZIN1S         : 2;
  __REG32  ADTRG0S        : 4;
  __REG32  ADTRG1S        : 4;
  __REG32  ADTRG2S        : 4;
  __REG32                 : 8;
} __epfr09_bits;

/* Extended Pin Function Setting Register 10 (EPFR10) */
typedef struct {
  __REG32  UEDEFB         : 1;
  __REG32  UEDTHB         : 1;
  __REG32  UECLKE         : 1;
  __REG32  UEWEXE         : 1;
  __REG32  UEDQME         : 1;
  __REG32  UEOEXE         : 1;
  __REG32  UEFLSE         : 1;
  __REG32  UECS1E         : 1;
  __REG32  UECS2E         : 1;
  __REG32  UECS3E         : 1;
  __REG32  UECS4E         : 1;
  __REG32  UECS5E         : 1;
  __REG32  UECS6E         : 1;
  __REG32  UECS7E         : 1;
  __REG32  UEAOOE         : 1;
  __REG32  UEA08E         : 1;
  __REG32  UEA09E         : 1;
  __REG32  UEA10E         : 1;
  __REG32  UEA11E         : 1;
  __REG32  UEA12E         : 1;
  __REG32  UEA13E         : 1;
  __REG32  UEA14E         : 1;
  __REG32  UEA15E         : 1;
  __REG32  UEA16E         : 1;
  __REG32  UEA17E         : 1;
  __REG32  UEA18E         : 1;
  __REG32  UEA19E         : 1;
  __REG32  UEA20E         : 1;
  __REG32  UEA21E         : 1;
  __REG32  UEA22E         : 1;
  __REG32  UEA23E         : 1;
  __REG32  UEA24E         : 1;
} __epfr10_bits;

/* Extended Pin Function Setting Register 11 (EPFR11) */
typedef struct {
  __REG32  UEALEE         : 1;
  __REG32  UECS0E         : 1;
  __REG32  UEA01E         : 1;
  __REG32  UEA02E         : 1;
  __REG32  UEA03E         : 1;
  __REG32  UEA04E         : 1;
  __REG32  UEA05E         : 1;
  __REG32  UEA06E         : 1;
  __REG32  UEA07E         : 1;
  __REG32  UED00B         : 1;
  __REG32  UED01B         : 1;
  __REG32  UED02B         : 1;
  __REG32  UED03B         : 1;
  __REG32  UED04B         : 1;
  __REG32  UED05B         : 1;
  __REG32  UED06B         : 1;
  __REG32  UED07B         : 1;
  __REG32  UED08B         : 1;
  __REG32  UED09B         : 1;
  __REG32  UED10B         : 1;
  __REG32  UED11B         : 1;
  __REG32  UED12B         : 1;
  __REG32  UED13B         : 1;
  __REG32  UED14B         : 1;
  __REG32  UED15B         : 1;
  __REG32  UERLC          : 1;
  __REG32  			          : 6;
} __epfr11_bits;

/* Extended Pin Function Setting Register 12 (EPFR12) */
typedef struct {
  __REG32                 : 2;
  __REG32  TIOA8E         : 2;
  __REG32  TIOB8S         : 2;
  __REG32                 : 2;
  __REG32  TIOA9S         : 2;
  __REG32  TIOA9E         : 2;
  __REG32  TIOB9S         : 2;
  __REG32                 : 4;
  __REG32  TIOA10E        : 2;
  __REG32  TIOB10S        : 2;
  __REG32                 : 2;
  __REG32  TIOA11S        : 2;
  __REG32  TIOA11E        : 2;
  __REG32  TIOB11S        : 2;
  __REG32                 : 2;
} __epfr12_bits;

/* Extended Pin Function Setting Register 13 (EPFR13) */
typedef struct {
  __REG32                 : 2;
  __REG32  TIOA12E        : 2;
  __REG32  TIOB12S        : 2;
  __REG32                 : 2;
  __REG32  TIOA13S        : 2;
  __REG32  TIOA13E        : 2;
  __REG32  TIOB13S        : 2;
  __REG32                 : 4;
  __REG32  TIOA14E        : 2;
  __REG32  TIOB14S        : 2;
  __REG32                 : 2;
  __REG32  TIOA15S        : 2;
  __REG32  TIOA15E        : 2;
  __REG32  TIOB15S        : 2;
  __REG32                 : 2;
} __epfr13_bits;

/* Extended Pin Function Setting Register 14 (EPFR14) */
typedef struct {
  __REG32  QAIN2S         : 2;
  __REG32  QBIN2S         : 2;
  __REG32  QZIN2S         : 2;
  __REG32                 :12;
  __REG32  E_TD0E         : 1;
  __REG32  E_TD1E         : 1;
  __REG32  E_TE0E         : 1;
  __REG32  E_TE1E         : 1;
  __REG32  E_MC0E         : 1;
  __REG32  E_MC1B         : 1;
  __REG32  E_MD0B         : 1;
  __REG32  E_MD1B         : 1;
  __REG32  E_CKE          : 1;
  __REG32  E_PSE          : 1;
  __REG32  E_SPLC         : 2;
  __REG32                 : 2;
} __epfr14_bits;

/* Extended Pin Function Setting Register 15 (EPFR15) */
typedef struct {
  __REG32  EINT16S        : 2;
  __REG32  EINT17S        : 2;
  __REG32  EINT18S        : 2;
  __REG32  EINT19S        : 2;
  __REG32  EINT20S        : 2;
  __REG32  EINT21S        : 2;
  __REG32  EINT22S        : 2;
  __REG32  EINT23S        : 2;
  __REG32  EINT24S        : 2;
  __REG32  EINT25S        : 2;
  __REG32  EINT26S        : 2;
  __REG32  EINT27S        : 2;
  __REG32  EINT28S        : 2;
  __REG32  EINT29S        : 2;
  __REG32  EINT30S        : 2;
  __REG32  EINT31S        : 2;
} __epfr15_bits;

/* Low-voltage Detection Voltage Control Register (LVD_CTL) */
typedef struct {
  __REG8                  : 2;
  __REG8   SVHI           : 4;
  __REG8                  : 1;
  __REG8   LVDIE          : 1;
} __lvd_ctl_bits;

/* Low-voltage Detection Interrupt Register (LVD_STR) */
typedef struct {
  __REG8                  : 7;
  __REG8   LVDIR          : 1;
} __lvd_str_bits;

/* Low-voltage Detection Interrupt Clear Register (LVD_CLR) */
typedef struct {
  __REG8                  : 7;
  __REG8   LVDCL          : 1;
} __lvd_clr_bits;

/* Low-voltage Detection Circuit Status Register (LVD_STR2) */
typedef struct {
  __REG8                  : 7;
  __REG8   LVDIRDY        : 1;
} __lvd_str2_bits;

/* USB Clock Setup Register (UCCR) */
typedef struct {
  __REG8   UCEN0          : 1;
  __REG8   UCSEL          : 2;
  __REG8   UCEN1          : 1;
  __REG8   ECEN           : 1;
  __REG8   ECSEL          : 2;
  __REG8                  : 1;
} __uccr_bits;

/* USB/Ethernet-PLL Setting Register1(UPCR1) */
typedef struct {
  __REG8   UPLLEN         : 1;
  __REG8   UPINC          : 1;
  __REG8                  : 6;
} __upcr1_bits;

/* USB/Ethernet-PLL Setting Register2 (UPCR2) */
typedef struct {
  __REG8   UPOWT          : 3;
  __REG8                  : 5;
} __upcr2_bits;

/* USB/Ethernet-PLL Setting Register3 (UPCR3) */
typedef struct {
  __REG8   UPLLK          : 5;
  __REG8                  : 3;
} __upcr3_bits;

/* USB/Ethernet-PLL Setting Register4(UPCR4) */
typedef struct {
  __REG8   UPLLN          : 7;
  __REG8                  : 1;
} __upcr4_bits;

/* USB/Ethernet-PLL Setting Register5 (UPCR5) */
typedef struct {
  __REG8   UPLLM          : 4;
  __REG8                  : 4;
} __upcr5_bits;

/* USB/Ethernet-PLL Setting Register6 (UPCR6) */
typedef struct {
  __REG8   UBSR           : 4;
  __REG8                  : 4;
} __upcr6_bits;

/* USB/Ethernet-PLL Setting Register7 (UPCR7) */
typedef struct {
  __REG8   EPLLEN         : 1;
  __REG8                  : 7;
} __upcr7_bits;

/* USB/Ethernet-PLL State Register (UP_STR) */
typedef struct {
  __REG8   UPRDY          : 1;
  __REG8                  : 7;
} __up_str_bits;

/* USB/Ethernet-PLL Interrupt Factor Enable Register (UPINT_ENR) */
typedef struct {
  __REG8   UPCSE          : 1;
  __REG8                  : 7;
} __upint_enr_bits;

/* USB/Ethernet-PLL Interrupt Factor State Register(UPINT_STR) */
typedef struct {
  __REG8   UPCSI          : 1;
  __REG8                  : 7;
} __upin_str_bits;

/* USB Enable Request Register (USBEN0) */
typedef struct {
  __REG8   USBEN0         : 1;
  __REG8                  : 7;
} __usben0_bits;

/* USB Enable Request Register (USBEN1) */
typedef struct {
  __REG8   USBEN1         : 1;
  __REG8                  : 7;
} __usben1_bits;

/* Serial Mode Register (SMR) */
typedef union {
  /*UARTx_SMR*/
  struct {
    __REG8   SOE            : 1;
    __REG8                  : 1;
    __REG8   BDS            : 1;
    __REG8   SBL            : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  };
  /*CSIOx_SMR*/
  struct {
    __REG8   SOE            : 1;
    __REG8   SCKE           : 1;
    __REG8   BDS            : 1;
    __REG8   SCINV          : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  } CSIO;
  /*LINx_SMR*/
  struct {
    __REG8   SOE            : 1;
    __REG8                  : 2;
    __REG8   SBL            : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  } LIN;
  /*I2Cx_SMR*/
  struct {
    __REG8                  : 2;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8   WUCR           : 1;
    __REG8   MD             : 3;
  } I2C;
} __mfsx_smr_bits;

/* Serial Control Register (SCR)
  I2C Bus Control Register (IBCR) */
typedef union {
  /*UARTx_SCR*/
  struct {
    __REG8   TXE            : 1;
    __REG8   RXE            : 1;
    __REG8   TBIE           : 1;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8                  : 2;
    __REG8   UPCL           : 1;
  };
  /*CSIOx_SCR*/
  struct {
    __REG8   TXE            : 1;
    __REG8   RXE            : 1;
    __REG8   TBIE           : 1;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8   SPI            : 1;
    __REG8   MS             : 1;
    __REG8   UPCL           : 1;
  } CSIO;
  /*LINx_SCR*/
  struct {
    __REG8   TXE            : 1;
    __REG8   RXE            : 1;
    __REG8   TBIE           : 1;
    __REG8   TIE            : 1;
    __REG8   RIE            : 1;
    __REG8   LBR            : 1;
    __REG8   MS             : 1;
    __REG8   UPCL           : 1;
  } LIN;
  /*I2Cx_IBCR*/
  struct {
    __REG8   INT            : 1;
    __REG8   BER            : 1;
    __REG8   INTE           : 1;
    __REG8   CNDE           : 1;
    __REG8   WSEL           : 1;
    __REG8   ACKE           : 1;
    __REG8   ACT_SCC        : 1;
    __REG8   MSS            : 1;
  } I2C;
} __mfsx_scr_bits;

/* Extended Communication Control Register (ESCR)
   I2C Bus Status Register (IBSR) */
typedef union {
  /*UARTx_ESCR*/
  struct {
    __REG8   L              : 3;
    __REG8   P              : 1;
    __REG8   PEN            : 1;
    __REG8   INV            : 1;
    __REG8   ESBL           : 1;
    __REG8   FLWEN          : 1;
  };
  /*CSIOx_ESCR*/
  struct {
    __REG8   L              : 3;
    __REG8   WT             : 2;
    __REG8                  : 2;
    __REG8   SOP            : 1;
  } CSIO;
  /*LINx_ESCR*/
  struct {
    __REG8   DEL            : 2;
    __REG8   LBL            : 2;
    __REG8   LBIE           : 1;
    __REG8                  : 1;
    __REG8   ESBL           : 1;
    __REG8                  : 1;
  } LIN;
  /*I2Cx_IBSR*/
  struct {
    __REG8   BB             : 1;
    __REG8   SPC            : 1;
    __REG8   RSC            : 1;
    __REG8   AL             : 1;
    __REG8   TRX            : 1;
    __REG8   RSA            : 1;
    __REG8   RACK           : 1;
    __REG8   FBT            : 1;
  } I2C;
} __mfsx_escr_bits;

/* Serial Status Register (SSR) */
typedef union {
  /*UARTx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8   FRE            : 1;
    __REG8   PE             : 1;
    __REG8                  : 1;
    __REG8   REC            : 1;
  };
  /*CSIOx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8                  : 3;
    __REG8   REC            : 1;
  } CSIO;
  /*LINx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8   FRE            : 1;
    __REG8   LBD            : 1;
    __REG8                  : 1;
    __REG8   REC            : 1;
  } LIN;
  /*I2Cx_SSR*/
  struct {
    __REG8   TBI            : 1;
    __REG8   TDRE           : 1;
    __REG8   RDRF           : 1;
    __REG8   ORE            : 1;
    __REG8   TBIE           : 1;
    __REG8   DMA            : 1;
    __REG8   TSET           : 1;
    __REG8   REC            : 1;
  } I2C;
} __mfsx_ssr_bits;

/* Serial Status Register (SSR) */
typedef union {
  /*UARTx_RDR*/
  /*UARTx_TDR*/
  struct {
    __REG16  D              : 9;
    __REG16                 : 7;
  };
  /*CSIOx_RDR*/
  /*CSIOx_TDR*/
  struct {
    __REG16  D              : 9;
    __REG16                 : 7;
  } CSIO;
  /*LINx_RDR*/
  /*LINx_TDR*/
  struct {
    __REG16  D              : 8;
    __REG16                 : 8;
  } LIN;
  /*I2Cx_RDR*/
  /*I2Cx_TDR*/
  struct {
    __REG16  D              : 8;
    __REG16                 : 8;
  } I2C;
} __mfsx_rdr_tdr_bits;

/* Baud Rate Generator Registers BGR */
typedef union {
  /*UARTx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16  EXT            : 1;
  };
  /*CSIOx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16                 : 1;
  } CSIO;
  /*LINx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16  EXT            : 1;
  } LIN;
  /*I2Cx_BGR*/
  struct {
    __REG16  BGR            :15;
    __REG16                 : 1;
  } I2C;
} __mfsx_bgr_bits;

/* 7-bit Slave Address Register (ISBA) */
typedef  struct {
  __REG8   SA             : 7;
  __REG8   SAEN           : 1;
} __mfsx_isba_bits;

/* 7-bit Slave  Address Mask Register (ISMK) */
typedef struct {
  __REG8   SM             : 7;
  __REG8   EN             : 1;
} __mfsx_ismk_bits;

/* FIFO Control Register (FCR) */
typedef union {
  /*UARTx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } ;
  /*CSIOx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } CSIO;
  /*LINx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } LIN;
  /*I2Cx_FCR*/
  struct {
    __REG16  FE1            : 1;
    __REG16  FE2            : 1;
    __REG16  FCL1           : 1;
    __REG16  FCL2           : 1;
    __REG16  FSET           : 1;
    __REG16  FLD            : 1;
    __REG16  FLST           : 1;
    __REG16                 : 1;
    __REG16  FSEL           : 1;
    __REG16  FTIE           : 1;
    __REG16  FDRQ           : 1;
    __REG16  FRIIE          : 1;
    __REG16  FLSTE          : 1;
    __REG16                 : 3;
  } I2C;
} __mfsx_fcr_bits;

/* I2C Auxiliary Noise Filter Setting Register (I2CDNF) */
typedef struct {
  __REG16 I2CDNF0        : 2;
  __REG16 I2CDNF1        : 2;
  __REG16 I2CDNF2        : 2;
  __REG16 I2CDNF3        : 2;
  __REG16 I2CDNF4        : 2;
  __REG16 I2CDNF5        : 2;
  __REG16 I2CDNF6        : 2;
  __REG16 I2CDNF7        : 2;
} __i2cdnf_bits;

/* CRC Control Register (CRCCR) */
typedef struct {
    __REG8   INIT           : 1;
    __REG8   CRC32          : 1;
    __REG8   LTLEND         : 1;
    __REG8   LSBFST         : 1;
    __REG8   CRCLTE         : 1;
    __REG8   CRCLSF         : 1;
    __REG8   FXOR           : 1;
    __REG8                  : 1;
} __crccr_bits;

/* CRC Input Data Register (CRCIN) */
typedef union {
  /*CRCIN*/
  struct {
    __REG32  LL  : 8;
    __REG32  LH  : 8;
    __REG32  HL  : 8;
    __REG32  HH  : 8;
  };
  struct
  {
    union
    {
      /*CRCINL*/
      struct{
        __REG16 L  : 8;
        __REG16 H  : 8;
      } __shortl_bit;
      __REG16 __shortl;
    };
    union
    {
     /*CRCINH*/
      struct{
        __REG16 L  : 8;
        __REG16 H  : 8;
      } __shorth_bit;
      __REG16 __shorth;
    };
  };
  struct
  {
      __REG8  __byte0;
      __REG8  __byte1;
      __REG8  __byte2;
      __REG8  __byte3;
  };
} __crcin_bits;

/* Watch Counter Read Register (WCRD) */
typedef struct {
    __REG8   CTR            : 6;
    __REG8                  : 2;
} __wcrd_bits;

/* Watch Counter Reload Register (WCRL) */
typedef struct {
    __REG8   RLC            : 6;
    __REG8                  : 2;
} __wcrl_bits;

/* Watch Counter Control Register (WCCR) */
typedef struct {
    __REG8   WCIF           : 1;
    __REG8   WCIE           : 1;
    __REG8   CS             : 2;
    __REG8                  : 2;
    __REG8   WCOP           : 1;
    __REG8   WCEN           : 1;
} __wccr_bits;

/* Clock Selection Register (CLK_SEL) */
typedef struct {
    __REG16  SEL_IN         : 1;
    __REG16                 : 7;
    __REG16  SEL_OUT        : 1;
    __REG16                 : 7;
} __clk_sel_bits;

/* Division Clock Enable Register (CLK_EN) */
typedef struct {
    __REG8   CLK_EN         : 1;
    __REG8   CLK_EN_R       : 1;
    __REG8                  : 6;
} __clk_en_bits;

/* Mode Register 0 to Mode Register 7 */
typedef struct {
    __REG32  WDTH           : 2;
    __REG32  RBMON          : 1;
    __REG32  WEOFF          : 1;
    __REG32  NAND           : 1;
    __REG32  PAGE           : 1;
    __REG32  RDY            : 1;
    __REG32  SHRTDOUT       : 1;
    __REG32  MPXMODE        : 1;
    __REG32  ALEINV         : 1;
    __REG32                 : 1;
    __REG32  MPXDOFF        : 1;
    __REG32  MPXCSOF        : 1;
    __REG32  MOEXEUP        : 1;
    __REG32                 :18;
} __modex_bits;

/* Timing Register 0 to Timing Register 7 */
typedef struct {
    __REG32  RACC           : 4;
    __REG32  RADC           : 4;
    __REG32  FRADC          : 4;
    __REG32  RIDLC          : 4;
    __REG32  WACC           : 4;
    __REG32  WADC           : 4;
    __REG32  WWEC           : 4;
    __REG32  WIDLC          : 4;
} __timx_bits;

/* Area Register 0 to Area Register 7 */
typedef struct {
    __REG32  ADDR           : 8;
    __REG32                 : 8;
    __REG32  MASK           : 7;
    __REG32                 : 9;
} __areax_bits;

/* ALE Timing Register 0 to 7 (ATIM0 to 7)  */
typedef struct {
    __REG32  ALC            : 4;
    __REG32  ALES           : 4;
    __REG32  ALEW           : 4;
    __REG32                 :20;
} __atimx_bits;

/* Division Clock Register (DCLKR) */
typedef struct {
    __REG32  MDIV           : 4;
    __REG32  MCLKON         : 1;
    __REG32                 :27;
} __dclkr_bits;

/* Host Control Registers  (HCNT) */
typedef struct {
    __REG16  HOST           : 1;
    __REG16  URST           : 1;
    __REG16  SOFIRE         : 1;
    __REG16  DIRE           : 1;
    __REG16  CNNIRE         : 1;
    __REG16  CMPIRE         : 1;
    __REG16  URIRE          : 1;
    __REG16  RWKIRE         : 1;
    __REG16  RETRY          : 1;
    __REG16  CANCEL         : 1;
    __REG16  SOFSTEP        : 1;
    __REG16                 : 5;
} __usb_hcnt_bits;

/* Host Interrupt Register (HIRQ) */
typedef struct {
    __REG8   SOFIRQ         : 1;
    __REG8   DIRQ           : 1;
    __REG8   CNNIRQ         : 1;
    __REG8   CMPIRQ         : 1;
    __REG8   URIRQ          : 1;
    __REG8   RWKIRQ         : 1;
    __REG8                  : 1;
    __REG8   TCAN           : 1;
} __usb_hirq_bits;

/* Host Error Status Register (HERR) */
typedef struct {
    __REG8   HS             : 2;
    __REG8   STUFF          : 1;
    __REG8   TGERR          : 1;
    __REG8   CRC            : 1;
    __REG8   TOUT           : 1;
    __REG8   RERR           : 1;
    __REG8   LSTSOF         : 1;
} __usb_herr_bits;

/* Host Status Register (HSTATE) */
typedef struct {
    __REG8   CSTAT          : 1;
    __REG8   TMODE          : 1;
    __REG8   SUSP           : 1;
    __REG8   SOFBUSY        : 1;
    __REG8   CLKSEL         : 1;
    __REG8   ALIVE          : 1;
    __REG8                  : 2;
} __usb_hstate_bits;

/* Retry Timer Setup Register 2(HRTIMER)2*/
typedef struct {
    __REG8   RTIMER2        : 2;
    __REG8                  : 6;
} __usb_hrtimer2_bits;

/* Host Address Register (HADR)*/
typedef struct {
    __REG8   Address        : 7;
    __REG8                  : 1;
} __usb_hadr_bits;

/* EOF Setup Register (HEOF) */
typedef struct {
    __REG16  HEOF           :14;
    __REG16                 : 2;
} __usb_heof_bits;

/* Frame Setup Register (HFRAME) */
typedef struct {
    __REG16  FRAME          :11;
    __REG16                 : 5;
} __usb_hframe_bits;

/* Host Token Endpoint Register (HTOKEN) */
typedef struct {
    __REG8   ENDPT          : 4;
    __REG8   TKNEN          : 3;
    __REG8   TGGL           : 1;
} __usb_htoken_bits;

/* UDC Control Register (UDCC) */
typedef struct {
    __REG16  PWC            : 1;
    __REG16  RFBK           : 1;
    __REG16                 : 1;
    __REG16  STALCLREN      : 1;
    __REG16  USTP           : 1;
    __REG16  HCONX          : 1;
    __REG16  RESUM          : 1;
    __REG16  RST            : 1;
    __REG16                 : 8;
} __usb_udcc_bits;

/* EP0 Control Register (EP0C) */
typedef struct {
    __REG16  PKS            : 7;
    __REG16                 : 2;
    __REG16  STAL           : 1;
    __REG16                 : 6;
} __usb_ep0c_bits;

/* EP1 Control Register (EP1C) */
typedef struct {
    __REG16  PKS            : 9;
    __REG16  STAL           : 1;
    __REG16  NULE           : 1;
    __REG16  DMAE           : 1;
    __REG16  DIR            : 1;
    __REG16  TYPE           : 2;
    __REG16  EPEN           : 1;
} __usb_ep1c_bits;

/*EP2 to 5 Control Registers (EP2C to EP5C) */
typedef struct {
    __REG16  PKS            : 7;
    __REG16                 : 2;
    __REG16  STAL           : 1;
    __REG16  NULE           : 1;
    __REG16  DMAE           : 1;
    __REG16  DIR            : 1;
    __REG16  TYPE           : 2;
    __REG16  EPEN           : 1;
} __usb_epxc_bits;

/*EP2 to 5 Control Registers (EP2C to EP5C) */
typedef struct {
    __REG16  TMSP           :11;
    __REG16                 : 5;
} __usb_tmsp_bits;

/* UDC Status Register (UDCS) */
typedef struct {
    __REG8   CONF           : 1;
    __REG8   SETP           : 1;
    __REG8   WKUP           : 1;
    __REG8   BRST           : 1;
    __REG8   SOF            : 1;
    __REG8   SUSP           : 1;
    __REG8                  : 2;
} __usb_udcs_bits;

/* UDC Interrupt Enable Register (UDCIE) */
typedef struct {
    __REG8   CONFIE         : 1;
    __REG8   CONFN          : 1;
    __REG8   WKUPIE         : 1;
    __REG8   BRSTIE         : 1;
    __REG8   SOFIE          : 1;
    __REG8   SUSPIE         : 1;
    __REG8                  : 2;
} __usb_udcie_bits;

/* EP0I Status Register (EP0IS) */
typedef struct {
    __REG16                 :10;
    __REG16  DRQI           : 1;
    __REG16                 : 3;
    __REG16  DRQIIE         : 1;
    __REG16  BFINI          : 1;
} __usb_ep0is_bits;

/* EP0O Status Register (EP0OS) */
typedef struct {
    __REG16  SIZE           : 7;
    __REG16                 : 2;
    __REG16  SPK            : 1;
    __REG16  DRQO           : 1;
    __REG16                 : 2;
    __REG16  SPKIE          : 1;
    __REG16  DRQOIE         : 1;
    __REG16  BFINI          : 1;
} __usb_ep0os_bits;

/* EP1 Status Register (EP1S) */
typedef struct {
    __REG16  SIZE           : 9;
    __REG16  SPK            : 1;
    __REG16  DRQ            : 1;
    __REG16  BUSY           : 1;
    __REG16                 : 1;
    __REG16  SPKIE          : 1;
    __REG16  DRQIE          : 1;
    __REG16  BFINI          : 1;
} __usb_ep1s_bits;

/* EP2 to 5 Status Registers (EP2S to EP5S) */
typedef struct {
    __REG16  SIZE           : 7;
    __REG16                 : 2;
    __REG16  SPK            : 1;
    __REG16  DRQ            : 1;
    __REG16  BUSY           : 1;
    __REG16                 : 1;
    __REG16  SPKIE          : 1;
    __REG16  DRQIE          : 1;
    __REG16  BFINI          : 1;
} __usb_epxs_bits;

/* EP0 to 5 Data Registers (EP0DTH to EP5DTH/EP0DTL to EP5DTL) */
typedef union {
    /*USB0_EPxDT*/
    /*USB1_EPxDT*/
    struct {
      __REG16 DTL             : 8;
      __REG16 DTH             : 8;
    };
    struct {
      __REG8  __byte0 ;
      __REG8  __byte1 ;
    };
} __usb_epxdt_bits;

/* DMACR (Entire DMAC Configuration Register) */
typedef struct {
    __REG32                 :24;
    __REG32  DH             : 4;
    __REG32  PR             : 1;
    __REG32                 : 1;
    __REG32  DS             : 1;
    __REG32  DE             : 1;
} __dmacr_bits;

/* DMACA (Configuration A Register) */
typedef struct {
    __REG32  TC             :16;
    __REG32  BC             : 4;
    __REG32                 : 3;
    __REG32  IS             : 6;
    __REG32  ST             : 1;
    __REG32  PB             : 1;
    __REG32  EB             : 1;
} __dmacax_bits;

/* DMACB (Configuration B Register) */
typedef struct {
    __REG32  EM             : 1;
    __REG32                 :15;
    __REG32  SS             : 3;
    __REG32  CI             : 1;
    __REG32  EI             : 1;
    __REG32  RD             : 1;
    __REG32  RS             : 1;
    __REG32  RC             : 1;
    __REG32  FD             : 1;
    __REG32  FS             : 1;
    __REG32  TW             : 2;
    __REG32  MS             : 2;
    __REG32                 : 2;
} __dmacbx_bits;

/* Ethernet Mode Select Register (ETH_MODE) */
typedef struct {
  __REG32  IFMODE         : 1;
  __REG32                 : 7;
  __REG32  RST0           : 1;
  __REG32                 :23;
} __eth_mode_bits;

/* Ethernet Clock Gating Register (ETH_CLKG) */
typedef struct {
  __REG32  MACEN          : 2;
  __REG32                 :30;
} __eth_clkg_bits;

/* Ethernet MCR (GMAC Register 0) */
typedef struct {
  __REG32                 : 2;
  __REG32  RE             : 1;
  __REG32  TE             : 1;
  __REG32  DC             : 1;
  __REG32  BL             : 2;
  __REG32  ACS            : 1;
  __REG32  LUD            : 1;
  __REG32  DR             : 1;
  __REG32  IPC            : 1;
  __REG32  DM             : 1;
  __REG32  LM             : 1;
  __REG32  DO             : 1;
  __REG32  FES            : 1;
  __REG32  PS             : 1;
  __REG32  DCRS           : 1;
  __REG32  IFG            : 3;
  __REG32  JE             : 1;
  __REG32  BE             : 1;
  __REG32  JD             : 1;
  __REG32  WD             : 1;
  __REG32  TC             : 1;
  __REG32  CST            : 1;
  __REG32                 : 6;
} __emac_mcr_bits;

/* Ethernet MFFR(GMAC Register 1) */
typedef struct {
  __REG32  PR             : 1;
  __REG32  HUC            : 1;
  __REG32  HMC            : 1;
  __REG32  DAIF           : 1;
  __REG32  PM             : 1;
  __REG32  DB             : 1;
  __REG32  PCF            : 2;
  __REG32  SAIF           : 1;
  __REG32  SAF            : 1;
  __REG32  HPF            : 1;
  __REG32                 :20;
  __REG32  RA             : 1;
} __emac_mffr_bits;

/* Ethernet GAR (GMAC Register 4) */
typedef struct {
  __REG32  GB             : 1;
  __REG32  GW             : 1;
  __REG32  CR             : 4;
  __REG32  GR             : 5;
  __REG32  PA             : 5;
  __REG32                 :16;
} __emac_gar_bits;

/* Ethernet GDR (GMAC Register 5) */
typedef struct {
  __REG32  GD             :16;
  __REG32                 :16;
} __emac_gdr_bits;

/* Ethernet FCR (GMAC Register 6) */
typedef struct {
  __REG32  FCB_BPA        : 1;
  __REG32  TFE            : 1;
  __REG32  RFE            : 1;
  __REG32  UP             : 1;
  __REG32  PLT            : 2;
  __REG32                 : 1;
  __REG32  DZPQ           : 1;
  __REG32                 : 8;
  __REG32  PT             :16;
} __emac_fcr_bits;

/* Ethernet VTR (GMAC Register 7) */
typedef struct {
  __REG32  VL             :16;
  __REG32  ETV            : 1;
  __REG32                 :15;
} __emac_vtr_bits;

/* Ethernet PMTR (GMAC Register 11) */
typedef struct {
  __REG32  PD             : 1;
  __REG32  MPE            : 1;
  __REG32  WFE            : 1;
  __REG32                 : 2;
  __REG32  MPR            : 1;
  __REG32  WPR            : 1;
  __REG32                 : 2;
  __REG32  GU             : 1;
  __REG32                 :21;
  __REG32  RWFFRPR        : 1;
} __emac_pmtr_bits;

/* Ethernet LPICSR (GMAC Register 12) */
typedef struct {
  __REG32  TLPIEN         : 1;
  __REG32  TLPIEX         : 1;
  __REG32  RLPIEN         : 1;
  __REG32  RLPIEX         : 1;
  __REG32                 : 4;
  __REG32  TLPIST         : 1;
  __REG32  RLPIST         : 1;
  __REG32                 : 6;
  __REG32  LPIEN          : 1;
  __REG32  PLS            : 1;
  __REG32  PLSEN          : 1;
  __REG32  LPITXA         : 1;
  __REG32                 :12;
} __emac_lpicsr_bits;

/* Ethernet LPITCR (GMAC Register 13) */
typedef struct {
  __REG32  TWT            :16;
  __REG32  LIT            :10;
  __REG32                 : 6;
} __emac_lpitcr_bits;

/* Ethernet ISR (GMAC Register 14) */
typedef struct {
  __REG32  RGIS           : 1;
  __REG32                 : 2;
  __REG32  PIS            : 1;
  __REG32  MIS            : 1;
  __REG32  RIS            : 1;
  __REG32  TIS            : 1;
  __REG32  COIS           : 1;
  __REG32                 : 1;
  __REG32  TSIS           : 1;
  __REG32  LPIIS          : 1;
  __REG32                 :21;
} __emac_isr_bits;

/* Ethernet IMR (GMAC Register 15) */
typedef struct {
  __REG32  RGIM           : 1;
  __REG32                 : 2;
  __REG32  PIM            : 1;
  __REG32                 : 5;
  __REG32  TSIM           : 1;
  __REG32  LPIIM          : 1;
  __REG32                 :21;
} __emac_imr_bits;

/* Ethernet MAR0H (GMAC Register 16) */
typedef struct {
  __REG32  A0             :16;
  __REG32                 :15;
  __REG32  MO             : 1;
} __emac_mar0h_bits;

/* Ethernet MAR1H to MAR31H */
typedef struct {
  __REG32  A              :16;
  __REG32                 : 8;
  __REG32  MBC            : 6;
  __REG32  SA             : 1;
  __REG32  AE             : 1;
} __emac_marxh_bits;

/* Ethernet RGSR (GMAC Register 54) */
typedef struct {
  __REG32  A              :16;
  __REG32                 : 8;
  __REG32  MBC            : 6;
  __REG32  SA             : 1;
  __REG32  AE             : 1;
} __emac_rgsr_bits;

/* Ethernet TSCR (GMAC Register 448) */
typedef struct {
  __REG32  TSE            : 1;
  __REG32  TFCU           : 1;
  __REG32  TSI            : 1;
  __REG32  TSU            : 1;
  __REG32  TITE           : 1;
  __REG32  TARU           : 1;
  __REG32                 : 2;
  __REG32  TSEA           : 1;
  __REG32  TSDB           : 1;
  __REG32  TSV2E          : 1;
  __REG32  TETSP          : 1;
  __REG32  TSIP6E         : 1;
  __REG32  TSIP4E         : 1;
  __REG32  TETSEM         : 1;
  __REG32  TSMRM          : 1;
  __REG32  TSPS           : 2;
  __REG32  TSENMF         : 1;
  __REG32                 : 5;
  __REG32  ATSFC          : 1;
  __REG32                 : 7;
} __emac_tscr_bits;

/* Ethernet SSIR (GMAC Register 449) */
typedef struct {
  __REG32  SSINC          : 8;
  __REG32                 :24;
} __emac_ssir_bits;

/* Ethernet STNR (GMAC Register 451) */
typedef struct {
  __REG32  TSSS           :31;
  __REG32                 : 1;
} __emac_stnr_bits;

/* Ethernet STNUR (GMAC Register 453) */
typedef struct {
  __REG32  TSSS           :31;
  __REG32  ADDSUB         : 1;
} __emac_stsnur_bits;

/* Ethernet TTNR (GMAC Register 456) */
typedef struct {
  __REG32  TSTR           :31;
  __REG32                 : 1;
} __emac_ttnr_bits;

/* Ethernet STHWSR (GMAC Register 457) */
typedef struct {
  __REG32  TSHWR          :16;
  __REG32                 :16;
} __emac_sthwsr_bits;

/* Ethernet TSR (GMAC Register 458) */
typedef struct {
  __REG32  TSSOVF         : 1;
  __REG32  TSTART         : 1;
  __REG32  ATSTS          : 1;
  __REG32  TRGTER         : 1;
  __REG32                 :20;
  __REG32  ATSSTM         : 1;
  __REG32  ATSNS          : 3;
  __REG32                 : 4;
} __emac_tsr_bits;

/* Ethernet PPSCR (GMAC Register 459) */
typedef struct {
  __REG32  PPSCTRL        : 4;
  __REG32                 :28;
} __emac_ppscr_bits;

/* Ethernet ATNR (GMAC Register 460) */
typedef struct {
  __REG32  ATN            :31;
  __REG32                 : 1;
} __emac_atnr_bits;

/* Ethernet BMR (DMA Register 0) */
typedef struct {
  __REG32  SWR            : 1;
  __REG32  DA             : 2;
  __REG32  DSL            : 4;
  __REG32  ATDS           : 1;
  __REG32  PBL            : 6;
  __REG32  PR             : 2;
  __REG32  FB             : 1;
  __REG32  RPBL           : 6;
  __REG32  USP            : 1;
  __REG32  _8xPBL         : 1;
  __REG32  AAL            : 1;
  __REG32  MB             : 1;
  __REG32  TXPR           : 1;
  __REG32                 : 4;
} __emac_bmr_bits;

/* Ethernet SR (DMA Register 5) */
typedef struct {
  __REG32  TI             : 1;
  __REG32  TPS            : 1;
  __REG32  TU             : 1;
  __REG32  TJT            : 1;
  __REG32  OVF            : 1;
  __REG32  UNF            : 1;
  __REG32  RI             : 1;
  __REG32  RU             : 1;
  __REG32  RPS            : 1;
  __REG32  RWT            : 1;
  __REG32  ETI            : 1;
  __REG32                 : 2;
  __REG32  FBI            : 1;
  __REG32  ERI            : 1;
  __REG32  AIS            : 1;
  __REG32  NIS            : 1;
  __REG32  RS             : 3;
  __REG32  TS             : 3;
  __REG32  EB             : 3;
  __REG32  GLI            : 1;
  __REG32  GMI            : 1;
  __REG32  GPI            : 1;
  __REG32  TTI            : 1;
  __REG32  GLPII          : 1;
  __REG32                 : 1;
} __emac_sr_bits;

/* Ethernet OMR (DMA Register 6) */
typedef struct {
  __REG32                 : 1;
  __REG32  SR             : 1;
  __REG32  OSF            : 1;
  __REG32  RTC            : 2;
  __REG32                 : 1;
  __REG32  FUF            : 1;
  __REG32  FEF            : 1;
  __REG32                 : 5;
  __REG32  ST             : 1;
  __REG32  TTC            : 3;
  __REG32                 : 3;
  __REG32  FTF            : 1;
  __REG32  TSF            : 1;
  __REG32                 : 2;
  __REG32  DFF            : 1;
  __REG32  RSF            : 1;
  __REG32  DT             : 2;
  __REG32                 : 4;
} __emac_omr_bits;

/* Ethernet IER (DMA Register 7) */
typedef struct {
  __REG32  TIE            : 1;
  __REG32  TSE            : 1;
  __REG32  TUE            : 1;
  __REG32  TJE            : 1;
  __REG32  OVE            : 1;
  __REG32  UNE            : 1;
  __REG32  RIE            : 1;
  __REG32  RUE            : 1;
  __REG32  RSE            : 1;
  __REG32  RWE            : 1;
  __REG32  ETE            : 1;
  __REG32                 : 2;
  __REG32  FBE            : 1;
  __REG32  ERE            : 1;
  __REG32  AIE            : 1;
  __REG32  NIE            : 1;
  __REG32                 :15;
} __emac_ier_bits;

/* Ethernet MFBOCR (DMA Register 8) */
typedef struct {
  __REG32  NMFH           :16;
  __REG32  ONMFH          : 1;
  __REG32  NMFF           :11;
  __REG32  ONMFF          : 1;
  __REG32                 : 3;
} __emac_mfbocr_bits;

/* Ethernet RIWTR (DMA Register 9) */
typedef struct {
  __REG32  RIWT           : 8;
  __REG32                 :24;
} __emac_riwtr_bits;

/* AHBSR (DMA Register 11) */
typedef struct {
  __REG32  AHBS           : 1;
  __REG32                 :31;
} __emac_ahbsr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/
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
 ** FLASH IF
 **
 ***************************************************************************/
__IO_REG32_BIT(FASZR,             0x40000000,__READ_WRITE ,__faszr_bits);
__IO_REG32_BIT(FRWTR,             0x40000004,__READ_WRITE ,__frwtr_bits);
__IO_REG32_BIT(FSTR,              0x40000008,__READ_WRITE ,__fstr_bits);
__IO_REG32_BIT(FSYNDN,            0x40000010,__READ_WRITE ,__fsyndn_bits);
__IO_REG32_BIT(FBFCR,             0x40000014,__READ_WRITE ,__fbfcr_bits);
__IO_REG32_BIT(CRTRMM,            0x40000100,__READ       ,__crtrmm_bits);

/***************************************************************************
 **
 ** Clock/Reset
 **
 ***************************************************************************/
__IO_REG32_BIT(SCM_CTL,           0x40010000,__READ_WRITE ,__scm_ctl_bits);
__IO_REG32_BIT(SCM_STR,           0x40010004,__READ       ,__scm_str_bits);
__IO_REG32_BIT(STB_CTL,           0x40010008,__READ_WRITE ,__stb_ctl_bits);
__IO_REG32_BIT(RST_STR,           0x4001000C,__READ       ,__rst_str_bits);
__IO_REG32_BIT(BSC_PSR,           0x40010010,__READ_WRITE ,__bsc_psr_bits);
__IO_REG32_BIT(APBC0_PSR,         0x40010014,__READ_WRITE ,__apbc0_psr_bits);
__IO_REG32_BIT(APBC1_PSR,         0x40010018,__READ_WRITE ,__apbc1_psr_bits);
__IO_REG32_BIT(APBC2_PSR,         0x4001001C,__READ_WRITE ,__apbc2_psr_bits);
__IO_REG32_BIT(SWC_PSR,           0x40010020,__READ_WRITE ,__swc_psr_bits);
__IO_REG32_BIT(TTC_PSR,           0x40010028,__READ_WRITE ,__ttc_psr_bits);
__IO_REG32_BIT(CSW_TMR,           0x40010030,__READ_WRITE ,__csw_tmr_bits);
__IO_REG32_BIT(PSW_TMR,           0x40010034,__READ_WRITE ,__psw_tmr_bits);
__IO_REG32_BIT(PLL_CTL1,          0x40010038,__READ_WRITE ,__pll_ctl1_bits);
__IO_REG32_BIT(PLL_CTL2,          0x4001003C,__READ_WRITE ,__pll_ctl2_bits);
__IO_REG32_BIT(CSV_CTL,           0x40010040,__READ_WRITE ,__csv_ctl_bits);
__IO_REG32_BIT(CSV_STR,           0x40010044,__READ       ,__csv_str_bits);
__IO_REG32_BIT(FCSWH_CTL,         0x40010048,__READ_WRITE ,__fcswh_ctl_bits);
__IO_REG32_BIT(FCSWL_CTL,         0x4001004C,__READ_WRITE ,__fcswl_ctl_bits);
__IO_REG32_BIT(FCSWD_CTL,         0x40010050,__READ       ,__fcswd_ctl_bits);
__IO_REG32_BIT(DBWDT_CTL,         0x40010054,__READ_WRITE ,__dbwdt_ctl_bits);
__IO_REG32(    DSPS_CTL,          0x40010058,__READ_WRITE );
__IO_REG32_BIT(INT_ENR,           0x40010060,__READ_WRITE ,__int_enr_bits);
__IO_REG32_BIT(INT_STR,           0x40010064,__READ       ,__int_str_bits);
__IO_REG32_BIT(INT_CLR,           0x40010068,__WRITE      ,__int_clr_bits);

/***************************************************************************
 **
 ** HW WDT
 **
 ***************************************************************************/
__IO_REG32(    WDG_LDR,           0x40011000,__READ_WRITE );
__IO_REG32(    WDG_VLR,           0x40011004,__READ       );
__IO_REG32_BIT(WDG_CTL,           0x40011008,__READ_WRITE ,__wdg_ctl_bits);
__IO_REG8(     WDG_ICL,           0x4001100C,__READ_WRITE );
__IO_REG32_BIT(WDG_RIS,           0x40011010,__READ       ,__wdg_ris_bits);
__IO_REG32(    WDG_LCK,           0x40011C00,__READ_WRITE );

/***************************************************************************
 **
 ** SW WDT
 **
 ***************************************************************************/
__IO_REG32(    WdogLoad,          0x40012000,__READ_WRITE );
__IO_REG32(    WdogValue,         0x40012004,__READ       );
__IO_REG32_BIT(WdogControl,       0x40012008,__READ_WRITE ,__wdg_ctl_bits);
__IO_REG32(    WdogIntClr,        0x4001200C,__READ_WRITE );
__IO_REG32_BIT(WdogRIS,           0x40012010,__READ_WRITE ,__wdg_ris_bits);
__IO_REG32(    WdogLock,          0x40012C00,__READ_WRITE );

/***************************************************************************
 **
 ** Dual Timer
 **
 ***************************************************************************/
__IO_REG32(    Timer1Load,        0x40015000,__READ_WRITE );
__IO_REG32(    Timer1Value,       0x40015004,__READ       );
__IO_REG32_BIT(Timer1Control,     0x40015008,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer1IntClr,      0x4001500C,__WRITE      );
__IO_REG32_BIT(Timer1RIS,         0x40015010,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer1MIS,         0x40015014,__READ       ,__timermis_bits);
__IO_REG32(    Timer1BGLoad,      0x40015018,__READ_WRITE );
__IO_REG32(    Timer2Load,        0x40015020,__READ_WRITE );
__IO_REG32(    Timer2Value,       0x40015024,__READ       );
__IO_REG32_BIT(Timer2Control,     0x40015028,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer2IntClr,      0x4001502C,__WRITE      );
__IO_REG32_BIT(Timer2RIS,         0x40015030,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer2MIS,         0x40015034,__READ       ,__timermis_bits);
__IO_REG32(    Timer2BGLoad,      0x40015038,__READ_WRITE );

/***************************************************************************
 **
 ** MFT0
 **
 ***************************************************************************/
__IO_REG16(    MFT0_OCCP0,        0x40020000,__READ_WRITE );
__IO_REG16(    MFT0_OCCP1,        0x40020004,__READ_WRITE );
__IO_REG16(    MFT0_OCCP2,        0x40020008,__READ_WRITE );
__IO_REG16(    MFT0_OCCP3,        0x4002000C,__READ_WRITE );
__IO_REG16(    MFT0_OCCP4,        0x40020010,__READ_WRITE );
__IO_REG16(    MFT0_OCCP5,        0x40020014,__READ_WRITE );
__IO_REG8_BIT( MFT0_OCSA10,       0x40020018,__READ_WRITE ,__mft_ocsa10_bits);
__IO_REG8_BIT( MFT0_OCSB10,       0x40020019,__READ_WRITE ,__mft_ocsb10_bits);
__IO_REG8_BIT( MFT0_OCSA32,       0x4002001C,__READ_WRITE ,__mft_ocsa32_bits);
__IO_REG8_BIT( MFT0_OCSB32,       0x4002001D,__READ_WRITE ,__mft_ocsb32_bits);
__IO_REG8_BIT( MFT0_OCSA54,       0x40020020,__READ_WRITE ,__mft_ocsa54_bits);
__IO_REG8_BIT( MFT0_OCSB54,       0x40020021,__READ_WRITE ,__mft_ocsb54_bits);
__IO_REG8_BIT( MFT0_OCSC,         0x40020025,__READ_WRITE ,__mft_ocsc_bits);
__IO_REG16(    MFT0_TCCP0,        0x40020028,__READ_WRITE );
__IO_REG16(    MFT0_TCDT0,        0x4002002C,__READ_WRITE );
__IO_REG16_BIT(MFT0_TCSA0,        0x40020030,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT0_TCSB0,        0x40020034,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT0_TCCP1,        0x40020038,__READ_WRITE );
__IO_REG16(    MFT0_TCDT1,        0x4002003C,__READ_WRITE );
__IO_REG16_BIT(MFT0_TCSA1,        0x40020040,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT0_TCSB1,        0x40020044,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT0_TCCP2,        0x40020048,__READ_WRITE );
__IO_REG16(    MFT0_TCDT2,        0x4002004C,__READ_WRITE );
__IO_REG16_BIT(MFT0_TCSA2,        0x40020050,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT0_TCSB2,        0x40020054,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG8_BIT( MFT0_OCFS10,       0x40020058,__READ_WRITE ,__mft_ocfs10_bits);
__IO_REG8_BIT( MFT0_OCFS32,       0x40020059,__READ_WRITE ,__mft_ocfs32_bits);
__IO_REG8_BIT( MFT0_OCFS54,       0x4002005C,__READ_WRITE ,__mft_ocfs54_bits);
__IO_REG8_BIT( MFT0_ICFS10,       0x40020060,__READ_WRITE ,__mft_icfs10_bits);
__IO_REG8_BIT( MFT0_ICFS32,       0x40020061,__READ_WRITE ,__mft_icfs32_bits);
__IO_REG16(    MFT0_ICCP0,        0x40020068,__READ       );
__IO_REG16(    MFT0_ICCP1,        0x4002006C,__READ       );
__IO_REG16(    MFT0_ICCP2,        0x40020070,__READ       );
__IO_REG16(    MFT0_ICCP3,        0x40020074,__READ       );
__IO_REG8_BIT( MFT0_ICSA10,       0x40020078,__READ_WRITE ,__mft_icsa10_bits);
__IO_REG8_BIT( MFT0_ICSB10,       0x40020079,__READ       ,__mft_icsb10_bits);
__IO_REG8_BIT( MFT0_ICSA32,       0x4002007C,__READ_WRITE ,__mft_icsa32_bits);
__IO_REG8_BIT( MFT0_ICSB32,       0x4002007D,__READ       ,__mft_icsb32_bits);
__IO_REG16(    MFT0_WFTM10,       0x40020080,__READ_WRITE );
__IO_REG16(    MFT0_WFTM32,       0x40020084,__READ_WRITE );
__IO_REG16(    MFT0_WFTM54,       0x40020088,__READ_WRITE );
__IO_REG16_BIT(MFT0_WFSA10,       0x4002008C,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT0_WFSA32,       0x40020090,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT0_WFSA54,       0x40020094,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT0_WFIR,         0x40020098,__READ_WRITE ,__mft_wfir_bits);
__IO_REG16_BIT(MFT0_NZCL,         0x4002009C,__READ_WRITE ,__mft_nzcl_bits);
__IO_REG16(    MFT0_ACCP0,        0x400200A0,__READ_WRITE );
__IO_REG16(    MFT0_ACCPDN0,      0x400200A4,__READ_WRITE );
__IO_REG16(    MFT0_ACCP1,        0x400200A8,__READ_WRITE );
__IO_REG16(    MFT0_ACCPDN1,      0x400200AC,__READ_WRITE );
__IO_REG16(    MFT0_ACCP2,        0x400200B0,__READ_WRITE );
__IO_REG16(    MFT0_ACCPDN2,      0x400200B4,__READ_WRITE );
__IO_REG16_BIT(MFT0_ACSB,         0x400200B8,__READ_WRITE ,__mft_acsb_bits);
__IO_REG16_BIT(MFT0_ACSA,         0x400200BC,__READ_WRITE ,__mft_acsa_bits);
__IO_REG16_BIT(MFT0_ATSA,         0x400200C0,__READ_WRITE ,__mft_atsa_bits);

/***************************************************************************
 **
 ** MFT1
 **
 ***************************************************************************/
__IO_REG16(    MFT1_OCCP0,        0x40021000,__READ_WRITE );
__IO_REG16(    MFT1_OCCP1,        0x40021004,__READ_WRITE );
__IO_REG16(    MFT1_OCCP2,        0x40021008,__READ_WRITE );
__IO_REG16(    MFT1_OCCP3,        0x4002100C,__READ_WRITE );
__IO_REG16(    MFT1_OCCP4,        0x40021010,__READ_WRITE );
__IO_REG16(    MFT1_OCCP5,        0x40021014,__READ_WRITE );
__IO_REG8_BIT( MFT1_OCSA10,       0x40021018,__READ_WRITE ,__mft_ocsa10_bits);
__IO_REG8_BIT( MFT1_OCSB10,       0x40021019,__READ_WRITE ,__mft_ocsb10_bits);
__IO_REG8_BIT( MFT1_OCSA32,       0x4002101C,__READ_WRITE ,__mft_ocsa32_bits);
__IO_REG8_BIT( MFT1_OCSB32,       0x4002101D,__READ_WRITE ,__mft_ocsb32_bits);
__IO_REG8_BIT( MFT1_OCSA54,       0x40021020,__READ_WRITE ,__mft_ocsa54_bits);
__IO_REG8_BIT( MFT1_OCSB54,       0x40021021,__READ_WRITE ,__mft_ocsb54_bits);
__IO_REG8_BIT( MFT1_OCSC,         0x40021025,__READ_WRITE ,__mft_ocsc_bits);
__IO_REG16(    MFT1_TCCP0,        0x40021028,__READ_WRITE );
__IO_REG16(    MFT1_TCDT0,        0x4002102C,__READ_WRITE );
__IO_REG16_BIT(MFT1_TCSA0,        0x40021030,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT1_TCSB0,        0x40021034,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT1_TCCP1,        0x40021038,__READ_WRITE );
__IO_REG16(    MFT1_TCDT1,        0x4002103C,__READ_WRITE );
__IO_REG16_BIT(MFT1_TCSA1,        0x40021040,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT1_TCSB1,        0x40021044,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT1_TCCP2,        0x40021048,__READ_WRITE );
__IO_REG16(    MFT1_TCDT2,        0x4002104C,__READ_WRITE );
__IO_REG16_BIT(MFT1_TCSA2,        0x40021050,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT1_TCSB2,        0x40021054,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG8_BIT( MFT1_OCFS10,       0x40021058,__READ_WRITE ,__mft_ocfs10_bits);
__IO_REG8_BIT( MFT1_OCFS32,       0x40021059,__READ_WRITE ,__mft_ocfs32_bits);
__IO_REG8_BIT( MFT1_OCFS54,       0x4002105C,__READ_WRITE ,__mft_ocfs54_bits);
__IO_REG8_BIT( MFT1_ICFS10,       0x40021060,__READ_WRITE ,__mft_icfs10_bits);
__IO_REG8_BIT( MFT1_ICFS32,       0x40021061,__READ_WRITE ,__mft_icfs32_bits);
__IO_REG16(    MFT1_ICCP0,        0x40021068,__READ       );
__IO_REG16(    MFT1_ICCP1,        0x4002106C,__READ       );
__IO_REG16(    MFT1_ICCP2,        0x40021070,__READ       );
__IO_REG16(    MFT1_ICCP3,        0x40021074,__READ       );
__IO_REG8_BIT( MFT1_ICSA10,       0x40021078,__READ_WRITE ,__mft_icsa10_bits);
__IO_REG8_BIT( MFT1_ICSB10,       0x40021079,__READ       ,__mft_icsb10_bits);
__IO_REG8_BIT( MFT1_ICSA32,       0x4002107C,__READ_WRITE ,__mft_icsa32_bits);
__IO_REG8_BIT( MFT1_ICSB32,       0x4002107D,__READ       ,__mft_icsb32_bits);
__IO_REG16(    MFT1_WFTM10,       0x40021080,__READ_WRITE );
__IO_REG16(    MFT1_WFTM32,       0x40021084,__READ_WRITE );
__IO_REG16(    MFT1_WFTM54,       0x40021088,__READ_WRITE );
__IO_REG16_BIT(MFT1_WFSA10,       0x4002108C,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT1_WFSA32,       0x40021090,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT1_WFSA54,       0x40021094,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT1_WFIR,         0x40021098,__READ_WRITE ,__mft_wfir_bits);
__IO_REG16_BIT(MFT1_NZCL,         0x4002109C,__READ_WRITE ,__mft_nzcl_bits);
__IO_REG16(    MFT1_ACCP0,        0x400210A0,__READ_WRITE );
__IO_REG16(    MFT1_ACCPDN0,      0x400210A4,__READ_WRITE );
__IO_REG16(    MFT1_ACCP1,        0x400210A8,__READ_WRITE );
__IO_REG16(    MFT1_ACCPDN1,      0x400210AC,__READ_WRITE );
__IO_REG16(    MFT1_ACCP2,        0x400210B0,__READ_WRITE );
__IO_REG16(    MFT1_ACCPDN2,      0x400210B4,__READ_WRITE );
__IO_REG16_BIT(MFT1_ACSB,         0x400210B8,__READ_WRITE ,__mft_acsb_bits);
__IO_REG16_BIT(MFT1_ACSA,         0x400210BC,__READ_WRITE ,__mft_acsa_bits);
__IO_REG16_BIT(MFT1_ATSA,         0x400210C0,__READ_WRITE ,__mft_atsa_bits);

/***************************************************************************
 **
 ** MFT2
 **
 ***************************************************************************/
__IO_REG16(    MFT2_OCCP0,        0x40022000,__READ_WRITE );
__IO_REG16(    MFT2_OCCP1,        0x40022004,__READ_WRITE );
__IO_REG16(    MFT2_OCCP2,        0x40022008,__READ_WRITE );
__IO_REG16(    MFT2_OCCP3,        0x4002200C,__READ_WRITE );
__IO_REG16(    MFT2_OCCP4,        0x40022010,__READ_WRITE );
__IO_REG16(    MFT2_OCCP5,        0x40022014,__READ_WRITE );
__IO_REG8_BIT( MFT2_OCSA10,       0x40022018,__READ_WRITE ,__mft_ocsa10_bits);
__IO_REG8_BIT( MFT2_OCSB10,       0x40022019,__READ_WRITE ,__mft_ocsb10_bits);
__IO_REG8_BIT( MFT2_OCSA32,       0x4002201C,__READ_WRITE ,__mft_ocsa32_bits);
__IO_REG8_BIT( MFT2_OCSB32,       0x4002201D,__READ_WRITE ,__mft_ocsb32_bits);
__IO_REG8_BIT( MFT2_OCSA54,       0x40022020,__READ_WRITE ,__mft_ocsa54_bits);
__IO_REG8_BIT( MFT2_OCSB54,       0x40022021,__READ_WRITE ,__mft_ocsb54_bits);
__IO_REG8_BIT( MFT2_OCSC,         0x40022025,__READ_WRITE ,__mft_ocsc_bits);
__IO_REG16(    MFT2_TCCP0,        0x40022028,__READ_WRITE );
__IO_REG16(    MFT2_TCDT0,        0x4002202C,__READ_WRITE );
__IO_REG16_BIT(MFT2_TCSA0,        0x40022030,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT2_TCSB0,        0x40022034,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT2_TCCP1,        0x40022038,__READ_WRITE );
__IO_REG16(    MFT2_TCDT1,        0x4002203C,__READ_WRITE );
__IO_REG16_BIT(MFT2_TCSA1,        0x40022040,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT2_TCSB1,        0x40022044,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG16(    MFT2_TCCP2,        0x40022048,__READ_WRITE );
__IO_REG16(    MFT2_TCDT2,        0x4002204C,__READ_WRITE );
__IO_REG16_BIT(MFT2_TCSA2,        0x40022050,__READ_WRITE ,__mft_tcsax_bits);
__IO_REG16_BIT(MFT2_TCSB2,        0x40022054,__READ_WRITE ,__mft_tcsbx_bits);
__IO_REG8_BIT( MFT2_OCFS10,       0x40022058,__READ_WRITE ,__mft_ocfs10_bits);
__IO_REG8_BIT( MFT2_OCFS32,       0x40022059,__READ_WRITE ,__mft_ocfs32_bits);
__IO_REG8_BIT( MFT2_OCFS54,       0x4002205C,__READ_WRITE ,__mft_ocfs54_bits);
__IO_REG8_BIT( MFT2_ICFS10,       0x40022060,__READ_WRITE ,__mft_icfs10_bits);
__IO_REG8_BIT( MFT2_ICFS32,       0x40022061,__READ_WRITE ,__mft_icfs32_bits);
__IO_REG16(    MFT2_ICCP0,        0x40022068,__READ       );
__IO_REG16(    MFT2_ICCP1,        0x4002206C,__READ       );
__IO_REG16(    MFT2_ICCP2,        0x40022070,__READ       );
__IO_REG16(    MFT2_ICCP3,        0x40022074,__READ       );
__IO_REG8_BIT( MFT2_ICSA10,       0x40022078,__READ_WRITE ,__mft_icsa10_bits);
__IO_REG8_BIT( MFT2_ICSB10,       0x40022079,__READ       ,__mft_icsb10_bits);
__IO_REG8_BIT( MFT2_ICSA32,       0x4002207C,__READ_WRITE ,__mft_icsa32_bits);
__IO_REG8_BIT( MFT2_ICSB32,       0x4002207D,__READ       ,__mft_icsb32_bits);
__IO_REG16(    MFT2_WFTM10,       0x40022080,__READ_WRITE );
__IO_REG16(    MFT2_WFTM32,       0x40022084,__READ_WRITE );
__IO_REG16(    MFT2_WFTM54,       0x40022088,__READ_WRITE );
__IO_REG16_BIT(MFT2_WFSA10,       0x4002208C,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT2_WFSA32,       0x40022090,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT2_WFSA54,       0x40022094,__READ_WRITE ,__mft_wfsa_bits);
__IO_REG16_BIT(MFT2_WFIR,         0x40022098,__READ_WRITE ,__mft_wfir_bits);
__IO_REG16_BIT(MFT2_NZCL,         0x4002209C,__READ_WRITE ,__mft_nzcl_bits);
__IO_REG16(    MFT2_ACCP0,        0x400220A0,__READ_WRITE );
__IO_REG16(    MFT2_ACCPDN0,      0x400220A4,__READ_WRITE );
__IO_REG16(    MFT2_ACCP1,        0x400220A8,__READ_WRITE );
__IO_REG16(    MFT2_ACCPDN1,      0x400220AC,__READ_WRITE );
__IO_REG16(    MFT2_ACCP2,        0x400220B0,__READ_WRITE );
__IO_REG16(    MFT2_ACCPDN2,      0x400220B4,__READ_WRITE );
__IO_REG16_BIT(MFT2_ACSB,         0x400220B8,__READ_WRITE ,__mft_acsb_bits);
__IO_REG16_BIT(MFT2_ACSA,         0x400220BC,__READ_WRITE ,__mft_acsa_bits);
__IO_REG16_BIT(MFT2_ATSA,         0x400220C0,__READ_WRITE ,__mft_atsa_bits);

/***************************************************************************
 **
 ** PPG
 **
 ***************************************************************************/
__IO_REG8_BIT( PPG_TTCR0,         0x40024001,__READ_WRITE ,__ppg_ttcr0_bits);
__IO_REG8(     PPG_COMP0,         0x40024009,__READ_WRITE );
__IO_REG8(     PPG_COMP2,         0x4002400C,__READ_WRITE );
__IO_REG8(     PPG_COMP4,         0x40024011,__READ_WRITE );
__IO_REG8(     PPG_COMP6,         0x40024014,__READ_WRITE );
__IO_REG8_BIT( PPG_TTCR1,         0x40024021,__READ_WRITE ,__ppg_ttcr1_bits);
__IO_REG8(     PPG_COMP1,         0x40024029,__READ_WRITE );
__IO_REG8(     PPG_COMP3,         0x4002402C,__READ_WRITE );
__IO_REG8(     PPG_COMP5,         0x40024031,__READ_WRITE );
__IO_REG8(     PPG_COMP7,         0x40024034,__READ_WRITE );
__IO_REG8_BIT( PPG_TTCR2,         0x40024041,__READ_WRITE ,__ppg_ttcr2_bits);
__IO_REG8(     PPG_COMP8,         0x40024049,__READ_WRITE );
__IO_REG8(     PPG_COMP10,        0x4002404C,__READ_WRITE );
__IO_REG8(     PPG_COMP12,        0x40024051,__READ_WRITE );
__IO_REG8(     PPG_COMP14,        0x40024054,__READ_WRITE );
__IO_REG16_BIT(PPG_TRG,           0x40024100,__READ_WRITE ,__ppg_trg_bits);
__IO_REG16_BIT(PPG_REVC,          0x40024104,__READ_WRITE ,__ppg_revc_bits);
__IO_REG16_BIT(PPG_TRG1,          0x40024140,__READ_WRITE ,__ppg_trg1_bits);
__IO_REG16_BIT(PPG_REVC1,         0x40024144,__READ_WRITE ,__ppg_revc1_bits);
__IO_REG8_BIT( PPG_PPGC1,         0x40024200,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC0,         0x40024201,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC3,         0x40024204,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC2,         0x40024205,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL0,          0x40024208,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL0 PPG_PRL0_bit.__byte0
#define PPG_PRLH0 PPG_PRL0_bit.__byte1
__IO_REG16_BIT(PPG_PRL1,          0x4002420C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL1 PPG_PRL1_bit.__byte0
#define PPG_PRLH1 PPG_PRL1_bit.__byte1
__IO_REG16_BIT(PPG_PRL2,          0x40024210,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL2 PPG_PRL2_bit.__byte0
#define PPG_PRLH2 PPG_PRL2_bit.__byte1
__IO_REG16_BIT(PPG_PRL3,          0x40024214,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL3 PPG_PRL3_bit.__byte0
#define PPG_PRLH3 PPG_PRL3_bit.__byte1
__IO_REG8_BIT( PPG_GATEC0,        0x40024218,__READ_WRITE ,__ppg_gatec0_bits);
__IO_REG8_BIT( PPG_PPGC5,         0x40024240,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC4,         0x40024241,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC7,         0x40024244,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC6,         0x40024245,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL4,          0x40024248,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL4 PPG_PRL4_bit.__byte0
#define PPG_PRLH4 PPG_PRL4_bit.__byte1
__IO_REG16_BIT(PPG_PRL5,          0x4002424C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL5 PPG_PRL5_bit.__byte0
#define PPG_PRLH5 PPG_PRL5_bit.__byte1
__IO_REG16_BIT(PPG_PRL6,          0x40024250,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL6 PPG_PRL6_bit.__byte0
#define PPG_PRLH6 PPG_PRL6_bit.__byte1
__IO_REG16_BIT(PPG_PRL7,          0x40024254,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL7 PPG_PRL7_bit.__byte0
#define PPG_PRLH7 PPG_PRL7_bit.__byte1
__IO_REG8_BIT( PPG_GATEC4,        0x40024258,__READ_WRITE ,__ppg_gatec4_bits);
__IO_REG8_BIT( PPG_PPGC9,         0x40024280,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC8,         0x40024281,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC11,        0x40024284,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC10,        0x40024285,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL8,          0x40024288,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL8 PPG_PRL8_bit.__byte0
#define PPG_PRLH8 PPG_PRL8_bit.__byte1
__IO_REG16_BIT(PPG_PRL9,          0x4002428C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL9 PPG_PRL9_bit.__byte0
#define PPG_PRLH9 PPG_PRL9_bit.__byte1
__IO_REG16_BIT(PPG_PRL10,         0x40024290,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL10 PPG_PRL10_bit.__byte0
#define PPG_PRLH10 PPG_PRL10_bit.__byte1
__IO_REG16_BIT(PPG_PRL11,         0x40024294,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL11 PPG_PRL11_bit.__byte0
#define PPG_PRLH11 PPG_PRL11_bit.__byte1
__IO_REG8_BIT( PPG_GATEC8,        0x40024298,__READ_WRITE ,__ppg_gatec8_bits);
__IO_REG8_BIT( PPG_PPGC13,        0x400242C0,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC12,        0x400242C1,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC15,        0x400242C4,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC14,        0x400242C5,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL12,         0x400242C8,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL12 PPG_PRL12_bit.__byte0
#define PPG_PRLH12 PPG_PRL12_bit.__byte1
__IO_REG16_BIT(PPG_PRL13,         0x400242CC,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL13 PPG_PRL13_bit.__byte0
#define PPG_PRLH13 PPG_PRL13_bit.__byte1
__IO_REG16_BIT(PPG_PRL14,         0x400242D0,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL14 PPG_PRL14_bit.__byte0
#define PPG_PRLH14 PPG_PRL14_bit.__byte1
__IO_REG16_BIT(PPG_PRL15,         0x400242D4,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL15 PPG_PRL15_bit.__byte0
#define PPG_PRLH15 PPG_PRL15_bit.__byte1
__IO_REG8_BIT( PPG_GATEC12,       0x400242D8,__READ_WRITE ,__ppg_gatec12_bits);
__IO_REG8_BIT( PPG_PPGC17,        0x40024300,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC16,        0x40024301,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC19,        0x40024304,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC18,        0x40024305,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL16,         0x40024308,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL16 PPG_PRL16_bit.__byte0
#define PPG_PRLH16 PPG_PRL16_bit.__byte1
__IO_REG16_BIT(PPG_PRL17,         0x4002430C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL17 PPG_PRL17_bit.__byte0
#define PPG_PRLH17 PPG_PRL17_bit.__byte1
__IO_REG16_BIT(PPG_PRL18,         0x40024310,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL18 PPG_PRL18_bit.__byte0
#define PPG_PRLH18 PPG_PRL18_bit.__byte1
__IO_REG16_BIT(PPG_PRL19,         0x40024314,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL19 PPG_PRL19_bit.__byte0
#define PPG_PRLH19 PPG_PRL19_bit.__byte1
__IO_REG8_BIT( PPG_GATEC16,       0x40024318,__READ_WRITE ,__ppg_gatec16_bits);
__IO_REG8_BIT( PPG_PPGC21,        0x40024340,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC20,        0x40024341,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC23,        0x40024344,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG8_BIT( PPG_PPGC22,        0x40024345,__READ_WRITE ,__ppg_ppgcx_bits);
__IO_REG16_BIT(PPG_PRL20,         0x40024348,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL20 PPG_PRL20_bit.__byte0
#define PPG_PRLH20 PPG_PRL20_bit.__byte1
__IO_REG16_BIT(PPG_PRL21,         0x4002434C,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL21 PPG_PRL21_bit.__byte0
#define PPG_PRLH21 PPG_PRL21_bit.__byte1
__IO_REG16_BIT(PPG_PRL22,         0x40024350,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL22 PPG_PRL22_bit.__byte0
#define PPG_PRLH22 PPG_PRL22_bit.__byte1
__IO_REG16_BIT(PPG_PRL23,         0x40024354,__READ_WRITE ,__ppg_prlx_bits);
#define PPG_PRLL23 PPG_PRL23_bit.__byte0
#define PPG_PRLH23 PPG_PRL23_bit.__byte1
__IO_REG8_BIT( PPG_GATEC20,        0x40024358,__READ_WRITE ,__ppg_gatec20_bits);

/***************************************************************************
 **
 ** BT0_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT0_PPG_PRLL,      0x40025000,__READ_WRITE );
__IO_REG16(    BT0_PPG_PRLH,      0x40025004,__READ_WRITE );
__IO_REG16(    BT0_PPG_TMR,       0x40025008,__READ       );
__IO_REG16_BIT(BT0_PPG_TMCR,      0x4002500C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT0_PPG_STC,       0x40025010,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT0_PPG_TMCR2,     0x40025011,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT1_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT1_PPG_PRLL,      0x40025040,__READ_WRITE );
__IO_REG16(    BT1_PPG_PRLH,      0x40025044,__READ_WRITE );
__IO_REG16(    BT1_PPG_TMR,       0x40025048,__READ       );
__IO_REG16_BIT(BT1_PPG_TMCR,      0x4002504C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT1_PPG_STC,       0x40025050,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT1_PPG_TMCR2,     0x40025051,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT2_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT2_PPG_PRLL,      0x40025080,__READ_WRITE );
__IO_REG16(    BT2_PPG_PRLH,      0x40025084,__READ_WRITE );
__IO_REG16(    BT2_PPG_TMR,       0x40025088,__READ       );
__IO_REG16_BIT(BT2_PPG_TMCR,      0x4002508C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT2_PPG_STC,       0x40025090,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT2_PPG_TMCR2,     0x40025091,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT3_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT3_PPG_PRLL,      0x400250C0,__READ_WRITE );
__IO_REG16(    BT3_PPG_PRLH,      0x400250C4,__READ_WRITE );
__IO_REG16(    BT3_PPG_TMR,       0x400250C8,__READ       );
__IO_REG16_BIT(BT3_PPG_TMCR,      0x400250CC,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT3_PPG_STC,       0x400250D0,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT3_PPG_TMCR2,     0x400250D1,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT4_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT4_PPG_PRLL,      0x40025200,__READ_WRITE );
__IO_REG16(    BT4_PPG_PRLH,      0x40025204,__READ_WRITE );
__IO_REG16(    BT4_PPG_TMR,       0x40025208,__READ       );
__IO_REG16_BIT(BT4_PPG_TMCR,      0x4002520C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT4_PPG_STC,       0x40025210,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT4_PPG_TMCR2,     0x40025211,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT5_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT5_PPG_PRLL,      0x40025240,__READ_WRITE );
__IO_REG16(    BT5_PPG_PRLH,      0x40025244,__READ_WRITE );
__IO_REG16(    BT5_PPG_TMR,       0x40025248,__READ       );
__IO_REG16_BIT(BT5_PPG_TMCR,      0x4002524C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT5_PPG_STC,       0x40025250,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT5_PPG_TMCR2,     0x40025251,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT6_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT6_PPG_PRLL,      0x40025280,__READ_WRITE );
__IO_REG16(    BT6_PPG_PRLH,      0x40025284,__READ_WRITE );
__IO_REG16(    BT6_PPG_TMR,       0x40025288,__READ       );
__IO_REG16_BIT(BT6_PPG_TMCR,      0x4002528C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT6_PPG_STC,       0x40025290,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT6_PPG_TMCR2,     0x40025291,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT7_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT7_PPG_PRLL,      0x400252C0,__READ_WRITE );
__IO_REG16(    BT7_PPG_PRLH,      0x400252C4,__READ_WRITE );
__IO_REG16(    BT7_PPG_TMR,       0x400252C8,__READ       );
__IO_REG16_BIT(BT7_PPG_TMCR,      0x400252CC,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT7_PPG_STC,       0x400252D0,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT7_PPG_TMCR2,     0x400252D1,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT8_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT8_PPG_PRLL,      0x40025400,__READ_WRITE );
__IO_REG16(    BT8_PPG_PRLH,      0x40025404,__READ_WRITE );
__IO_REG16(    BT8_PPG_TMR,       0x40025408,__READ       );
__IO_REG16_BIT(BT8_PPG_TMCR,      0x4002540C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT8_PPG_STC,       0x40025410,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT8_PPG_TMCR2,     0x40025411,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT9_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT9_PPG_PRLL,      0x40025440,__READ_WRITE );
__IO_REG16(    BT9_PPG_PRLH,      0x40025444,__READ_WRITE );
__IO_REG16(    BT9_PPG_TMR,       0x40025448,__READ       );
__IO_REG16_BIT(BT9_PPG_TMCR,      0x4002544C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT9_PPG_STC,       0x40025450,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT9_PPG_TMCR2,     0x40025451,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT10_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT10_PPG_PRLL,     0x40025480,__READ_WRITE );
__IO_REG16(    BT10_PPG_PRLH,     0x40025484,__READ_WRITE );
__IO_REG16(    BT10_PPG_TMR,      0x40025488,__READ       );
__IO_REG16_BIT(BT10_PPG_TMCR,     0x4002548C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT10_PPG_STC,      0x40025490,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT10_PPG_TMCR2,    0x40025491,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT11_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT11_PPG_PRLL,     0x400254C0,__READ_WRITE );
__IO_REG16(    BT11_PPG_PRLH,     0x400254C4,__READ_WRITE );
__IO_REG16(    BT11_PPG_TMR,      0x400254C8,__READ       );
__IO_REG16_BIT(BT11_PPG_TMCR,     0x400254CC,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT11_PPG_STC,      0x400254D0,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT11_PPG_TMCR2,    0x400254D1,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT12_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT12_PPG_PRLL,     0x40025600,__READ_WRITE );
__IO_REG16(    BT12_PPG_PRLH,     0x40025604,__READ_WRITE );
__IO_REG16(    BT12_PPG_TMR,      0x40025608,__READ       );
__IO_REG16_BIT(BT12_PPG_TMCR,     0x4002560C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT12_PPG_STC,      0x40025610,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT12_PPG_TMCR2,    0x40025611,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT13_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT13_PPG_PRLL,     0x40025640,__READ_WRITE );
__IO_REG16(    BT13_PPG_PRLH,     0x40025644,__READ_WRITE );
__IO_REG16(    BT13_PPG_TMR,      0x40025648,__READ       );
__IO_REG16_BIT(BT13_PPG_TMCR,     0x4002564C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT13_PPG_STC,      0x40025650,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT13_PPG_TMCR2,    0x40025651,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT14_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT14_PPG_PRLL,     0x40025680,__READ_WRITE );
__IO_REG16(    BT14_PPG_PRLH,     0x40025684,__READ_WRITE );
__IO_REG16(    BT14_PPG_TMR,      0x40025688,__READ       );
__IO_REG16_BIT(BT14_PPG_TMCR,     0x4002568C,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT14_PPG_STC,      0x40025690,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT14_PPG_TMCR2,    0x40025691,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT15_PPG
 **
 ***************************************************************************/
__IO_REG16(    BT15_PPG_PRLL,     0x400256C0,__READ_WRITE );
__IO_REG16(    BT15_PPG_PRLH,     0x400256C4,__READ_WRITE );
__IO_REG16(    BT15_PPG_TMR,      0x400256C8,__READ       );
__IO_REG16_BIT(BT15_PPG_TMCR,     0x400256CC,__READ_WRITE ,__btxtmcr_bits);
__IO_REG8_BIT( BT15_PPG_STC,      0x400256D0,__READ_WRITE ,__btxstc_bits);
__IO_REG8_BIT( BT15_PPG_TMCR2,    0x400256D1,__READ_WRITE ,__btxtmcr2_bits);

/***************************************************************************
 **
 ** BT0_PWM
 **
 ***************************************************************************/
#define BT0_PWM_PCSR        BT0_PPG_PRLL
#define BT0_PWM_PDUT        BT0_PPG_PRLH
#define BT0_PWM_TMR         BT0_PPG_TMR
#define BT0_PWM_TMCR        BT0_PPG_TMCR
#define BT0_PWM_TMCR_bit    BT0_PPG_TMCR_bit
#define BT0_PWM_STC         BT0_PPG_STC
#define BT0_PWM_STC_bit     BT0_PPG_STC_bit.PWM
#define BT0_PWM_TMCR2       BT0_PPG_TMCR2
#define BT0_PWM_TMCR2_bit   BT0_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT1_PWM
 **
 ***************************************************************************/
#define BT1_PWM_PCSR        BT1_PPG_PRLL
#define BT1_PWM_PDUT        BT1_PPG_PRLH
#define BT1_PWM_TMR         BT1_PPG_TMR
#define BT1_PWM_TMCR        BT1_PPG_TMCR
#define BT1_PWM_TMCR_bit    BT1_PPG_TMCR_bit
#define BT1_PWM_STC         BT1_PPG_STC
#define BT1_PWM_STC_bit     BT1_PPG_STC_bit.PWM
#define BT1_PWM_TMCR2       BT1_PPG_TMCR2
#define BT1_PWM_TMCR2_bit   BT1_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT2_PWM
 **
 ***************************************************************************/
#define BT2_PWM_PCSR        BT2_PPG_PRLL
#define BT2_PWM_PDUT        BT2_PPG_PRLH
#define BT2_PWM_TMR         BT2_PPG_TMR
#define BT2_PWM_TMCR        BT2_PPG_TMCR
#define BT2_PWM_TMCR_bit    BT2_PPG_TMCR_bit
#define BT2_PWM_STC         BT2_PPG_STC
#define BT2_PWM_STC_bit     BT2_PPG_STC_bit.PWM
#define BT2_PWM_TMCR2       BT2_PPG_TMCR2
#define BT2_PWM_TMCR2_bit   BT2_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT3_PWM
 **
 ***************************************************************************/
#define BT3_PWM_PCSR        BT3_PPG_PRLL
#define BT3_PWM_PDUT        BT3_PPG_PRLH
#define BT3_PWM_TMR         BT3_PPG_TMR
#define BT3_PWM_TMCR        BT3_PPG_TMCR
#define BT3_PWM_TMCR_bit    BT3_PPG_TMCR_bit
#define BT3_PWM_STC         BT3_PPG_STC
#define BT3_PWM_STC_bit     BT3_PPG_STC_bit.PWM
#define BT3_PWM_TMCR2       BT3_PPG_TMCR2
#define BT3_PWM_TMCR2_bit   BT3_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT4_PWM
 **
 ***************************************************************************/
#define BT4_PWM_PCSR        BT4_PPG_PRLL
#define BT4_PWM_PDUT        BT4_PPG_PRLH
#define BT4_PWM_TMR         BT4_PPG_TMR
#define BT4_PWM_TMCR        BT4_PPG_TMCR
#define BT4_PWM_TMCR_bit    BT4_PPG_TMCR_bit
#define BT4_PWM_STC         BT4_PPG_STC
#define BT4_PWM_STC_bit     BT4_PPG_STC_bit.PWM
#define BT4_PWM_TMCR2       BT4_PPG_TMCR2
#define BT4_PWM_TMCR2_bit   BT4_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT5_PWM
 **
 ***************************************************************************/
#define BT5_PWM_PCSR        BT5_PPG_PRLL
#define BT5_PWM_PDUT        BT5_PPG_PRLH
#define BT5_PWM_TMR         BT5_PPG_TMR
#define BT5_PWM_TMCR        BT5_PPG_TMCR
#define BT5_PWM_TMCR_bit    BT5_PPG_TMCR_bit
#define BT5_PWM_STC         BT5_PPG_STC
#define BT5_PWM_STC_bit     BT5_PPG_STC_bit.PWM
#define BT5_PWM_TMCR2       BT5_PPG_TMCR2
#define BT5_PWM_TMCR2_bit   BT5_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT6_PWM
 **
 ***************************************************************************/
#define BT6_PWM_PCSR        BT6_PPG_PRLL
#define BT6_PWM_PDUT        BT6_PPG_PRLH
#define BT6_PWM_TMR         BT6_PPG_TMR
#define BT6_PWM_TMCR        BT6_PPG_TMCR
#define BT6_PWM_TMCR_bit    BT6_PPG_TMCR_bit
#define BT6_PWM_STC         BT6_PPG_STC
#define BT6_PWM_STC_bit     BT6_PPG_STC_bit.PWM
#define BT6_PWM_TMCR2       BT6_PPG_TMCR2
#define BT6_PWM_TMCR2_bit   BT6_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT7_PWM
 **
 ***************************************************************************/
#define BT7_PWM_PCSR        BT7_PPG_PRLL
#define BT7_PWM_PDUT        BT7_PPG_PRLH
#define BT7_PWM_TMR         BT7_PPG_TMR
#define BT7_PWM_TMCR        BT7_PPG_TMCR
#define BT7_PWM_TMCR_bit    BT7_PPG_TMCR_bit
#define BT7_PWM_STC         BT7_PPG_STC
#define BT7_PWM_STC_bit     BT7_PPG_STC_bit.PWM
#define BT7_PWM_TMCR2       BT7_PPG_TMCR2
#define BT7_PWM_TMCR2_bit   BT7_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT8_PWM
 **
 ***************************************************************************/
#define BT8_PWM_PCSR        BT8_PPG_PRLL
#define BT8_PWM_PDUT        BT8_PPG_PRLH
#define BT8_PWM_TMR         BT8_PPG_TMR
#define BT8_PWM_TMCR        BT8_PPG_TMCR
#define BT8_PWM_TMCR_bit    BT8_PPG_TMCR_bit
#define BT8_PWM_STC         BT8_PPG_STC
#define BT8_PWM_STC_bit     BT8_PPG_STC_bit.PWM
#define BT8_PWM_TMCR2       BT8_PPG_TMCR2
#define BT8_PWM_TMCR2_bit   BT8_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT9_PWM
 **
 ***************************************************************************/
#define BT9_PWM_PCSR        BT9_PPG_PRLL
#define BT9_PWM_PDUT        BT9_PPG_PRLH
#define BT9_PWM_TMR         BT9_PPG_TMR
#define BT9_PWM_TMCR        BT9_PPG_TMCR
#define BT9_PWM_TMCR_bit    BT9_PPG_TMCR_bit
#define BT9_PWM_STC         BT9_PPG_STC
#define BT9_PWM_STC_bit     BT9_PPG_STC_bit.PWM
#define BT9_PWM_TMCR2       BT9_PPG_TMCR2
#define BT9_PWM_TMCR2_bit   BT9_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT10_PWM
 **
 ***************************************************************************/
#define BT10_PWM_PCSR       BT10_PPG_PRLL
#define BT10_PWM_PDUT       BT10_PPG_PRLH
#define BT10_PWM_TMR        BT10_PPG_TMR
#define BT10_PWM_TMCR       BT10_PPG_TMCR
#define BT10_PWM_TMCR_bit   BT10_PPG_TMCR_bit
#define BT10_PWM_STC        BT10_PPG_STC
#define BT10_PWM_STC_bit    BT10_PPG_STC_bit.PWM
#define BT10_PWM_TMCR2      BT10_PPG_TMCR2
#define BT10_PWM_TMCR2_bit  BT10_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT11_PWM
 **
 ***************************************************************************/
#define BT11_PWM_PCSR       BT11_PPG_PRLL
#define BT11_PWM_PDUT       BT11_PPG_PRLH
#define BT11_PWM_TMR        BT11_PPG_TMR
#define BT11_PWM_TMCR       BT11_PPG_TMCR
#define BT11_PWM_TMCR_bit   BT11_PPG_TMCR_bit
#define BT11_PWM_STC        BT11_PPG_STC
#define BT11_PWM_STC_bit    BT11_PPG_STC_bit.PWM
#define BT11_PWM_TMCR2      BT11_PPG_TMCR2
#define BT11_PWM_TMCR2_bit  BT11_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT12_PWM
 **
 ***************************************************************************/
#define BT12_PWM_PCSR       BT12_PPG_PRLL
#define BT12_PWM_PDUT       BT12_PPG_PRLH
#define BT12_PWM_TMR        BT12_PPG_TMR
#define BT12_PWM_TMCR       BT12_PPG_TMCR
#define BT12_PWM_TMCR_bit   BT12_PPG_TMCR_bit
#define BT12_PWM_STC        BT12_PPG_STC
#define BT12_PWM_STC_bit    BT12_PPG_STC_bit.PWM
#define BT12_PWM_TMCR2      BT12_PPG_TMCR2
#define BT12_PWM_TMCR2_bit  BT12_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT13_PWM
 **
 ***************************************************************************/
#define BT13_PWM_PCSR       BT13_PPG_PRLL
#define BT13_PWM_PDUT       BT13_PPG_PRLH
#define BT13_PWM_TMR        BT13_PPG_TMR
#define BT13_PWM_TMCR       BT13_PPG_TMCR
#define BT13_PWM_TMCR_bit   BT13_PPG_TMCR_bit
#define BT13_PWM_STC        BT13_PPG_STC
#define BT13_PWM_STC_bit    BT13_PPG_STC_bit.PWM
#define BT13_PWM_TMCR2      BT13_PPG_TMCR2
#define BT13_PWM_TMCR2_bit  BT13_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT14_PWM
 **
 ***************************************************************************/
#define BT14_PWM_PCSR       BT14_PPG_PRLL
#define BT14_PWM_PDUT       BT14_PPG_PRLH
#define BT14_PWM_TMR        BT14_PPG_TMR
#define BT14_PWM_TMCR       BT14_PPG_TMCR
#define BT14_PWM_TMCR_bit   BT14_PPG_TMCR_bit
#define BT14_PWM_STC        BT14_PPG_STC
#define BT14_PWM_STC_bit    BT14_PPG_STC_bit.PWM
#define BT14_PWM_TMCR2      BT14_PPG_TMCR2
#define BT14_PWM_TMCR2_bit  BT14_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT15_PWM
 **
 ***************************************************************************/
#define BT15_PWM_PCSR       BT15_PPG_PRLL
#define BT15_PWM_PDUT       BT15_PPG_PRLH
#define BT15_PWM_TMR        BT15_PPG_TMR
#define BT15_PWM_TMCR       BT15_PPG_TMCR
#define BT15_PWM_TMCR_bit   BT15_PPG_TMCR_bit
#define BT15_PWM_STC        BT15_PPG_STC
#define BT15_PWM_STC_bit    BT15_PPG_STC_bit.PWM
#define BT15_PWM_TMCR2      BT15_PPG_TMCR2
#define BT15_PWM_TMCR2_bit  BT15_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT0_RT
 **
 ***************************************************************************/
#define BT0_RT_PCSR         BT0_PPG_PRLL
#define BT0_RT_TMR          BT0_PPG_TMR
#define BT0_RT_TMCR         BT0_PPG_TMCR
#define BT0_RT_TMCR_bit     BT0_PPG_TMCR_bit.RT
#define BT0_RT_STC          BT0_PPG_STC
#define BT0_RT_STC_bit      BT0_PPG_STC_bit
#define BT0_RT_TMCR2        BT0_PPG_TMCR2
#define BT0_RT_TMCR2_bit    BT0_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT1_RT
 **
 ***************************************************************************/
#define BT1_RT_PCSR         BT1_PPG_PRLL
#define BT1_RT_TMR          BT1_PPG_TMR
#define BT1_RT_TMCR         BT1_PPG_TMCR
#define BT1_RT_TMCR_bit     BT1_PPG_TMCR_bit.RT
#define BT1_RT_STC          BT1_PPG_STC
#define BT1_RT_STC_bit      BT1_PPG_STC_bit
#define BT1_RT_TMCR2        BT1_PPG_TMCR2
#define BT1_RT_TMCR2_bit    BT1_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT2_RT
 **
 ***************************************************************************/
#define BT2_RT_PCSR         BT2_PPG_PRLL
#define BT2_RT_TMR          BT2_PPG_TMR
#define BT2_RT_TMCR         BT2_PPG_TMCR
#define BT2_RT_TMCR_bit     BT2_PPG_TMCR_bit.RT
#define BT2_RT_STC          BT2_PPG_STC
#define BT2_RT_STC_bit      BT2_PPG_STC_bit
#define BT2_RT_TMCR2        BT2_PPG_TMCR2
#define BT2_RT_TMCR2_bit    BT2_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT3_RT
 **
 ***************************************************************************/
#define BT3_RT_PCSR         BT3_PPG_PRLL
#define BT3_RT_TMR          BT3_PPG_TMR
#define BT3_RT_TMCR         BT3_PPG_TMCR
#define BT3_RT_TMCR_bit     BT3_PPG_TMCR_bit.RT
#define BT3_RT_STC          BT3_PPG_STC
#define BT3_RT_STC_bit      BT3_PPG_STC_bit
#define BT3_RT_TMCR2        BT3_PPG_TMCR2
#define BT3_RT_TMCR2_bit    BT3_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT4_RT
 **
 ***************************************************************************/
#define BT4_RT_PCSR         BT4_PPG_PRLL
#define BT4_RT_TMR          BT4_PPG_TMR
#define BT4_RT_TMCR         BT4_PPG_TMCR
#define BT4_RT_TMCR_bit     BT4_PPG_TMCR_bit.RT
#define BT4_RT_STC          BT4_PPG_STC
#define BT4_RT_STC_bit      BT4_PPG_STC_bit
#define BT4_RT_TMCR2        BT4_PPG_TMCR2
#define BT4_RT_TMCR2_bit    BT4_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT5_RT
 **
 ***************************************************************************/
#define BT5_RT_PCSR         BT5_PPG_PRLL
#define BT5_RT_TMR          BT5_PPG_TMR
#define BT5_RT_TMCR         BT5_PPG_TMCR
#define BT5_RT_TMCR_bit     BT5_PPG_TMCR_bit.RT
#define BT5_RT_STC          BT5_PPG_STC
#define BT5_RT_STC_bit      BT5_PPG_STC_bit
#define BT5_RT_TMCR2        BT5_PPG_TMCR2
#define BT5_RT_TMCR2_bit    BT5_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT6_RT
 **
 ***************************************************************************/
#define BT6_RT_PCSR         BT6_PPG_PRLL
#define BT6_RT_TMR          BT6_PPG_TMR
#define BT6_RT_TMCR         BT6_PPG_TMCR
#define BT6_RT_TMCR_bit     BT6_PPG_TMCR_bit.RT
#define BT6_RT_STC          BT6_PPG_STC
#define BT6_RT_STC_bit      BT6_PPG_STC_bit
#define BT6_RT_TMCR2        BT6_PPG_TMCR2
#define BT6_RT_TMCR2_bit    BT6_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT7_RT
 **
 ***************************************************************************/
#define BT7_RT_PCSR         BT7_PPG_PRLL
#define BT7_RT_TMR          BT7_PPG_TMR
#define BT7_RT_TMCR         BT7_PPG_TMCR
#define BT7_RT_TMCR_bit     BT7_PPG_TMCR_bit.RT
#define BT7_RT_STC          BT7_PPG_STC
#define BT7_RT_STC_bit      BT7_PPG_STC_bit
#define BT7_RT_TMCR2        BT7_PPG_TMCR2
#define BT7_RT_TMCR2_bit    BT7_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT8_RT
 **
 ***************************************************************************/
#define BT8_RT_PCSR         BT8_PPG_PRLL
#define BT8_RT_TMR          BT8_PPG_TMR
#define BT8_RT_TMCR         BT8_PPG_TMCR
#define BT8_RT_TMCR_bit     BT8_PPG_TMCR_bit.RT
#define BT8_RT_STC          BT8_PPG_STC
#define BT8_RT_STC_bit      BT8_PPG_STC_bit
#define BT8_RT_TMCR2        BT8_PPG_TMCR2
#define BT8_RT_TMCR2_bit    BT8_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT9_RT
 **
 ***************************************************************************/
#define BT9_RT_PCSR         BT9_PPG_PRLL
#define BT9_RT_TMR          BT9_PPG_TMR
#define BT9_RT_TMCR         BT9_PPG_TMCR
#define BT9_RT_TMCR_bit     BT9_PPG_TMCR_bit.RT
#define BT9_RT_STC          BT9_PPG_STC
#define BT9_RT_STC_bit      BT9_PPG_STC_bit
#define BT9_RT_TMCR2        BT9_PPG_TMCR2
#define BT9_RT_TMCR2_bit    BT9_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT10_RT
 **
 ***************************************************************************/
#define BT10_RT_PCSR        BT10_PPG_PRLL
#define BT10_RT_TMR         BT10_PPG_TMR
#define BT10_RT_TMCR        BT10_PPG_TMCR
#define BT10_RT_TMCR_bit    BT10_PPG_TMCR_bit.RT
#define BT10_RT_STC         BT10_PPG_STC
#define BT10_RT_STC_bit     BT10_PPG_STC_bit
#define BT10_RT_TMCR2       BT10_PPG_TMCR2
#define BT10_RT_TMCR2_bit   BT10_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT11_RT
 **
 ***************************************************************************/
#define BT11_RT_PCSR        BT11_PPG_PRLL
#define BT11_RT_TMR         BT11_PPG_TMR
#define BT11_RT_TMCR        BT11_PPG_TMCR
#define BT11_RT_TMCR_bit    BT11_PPG_TMCR_bit.RT
#define BT11_RT_STC         BT11_PPG_STC
#define BT11_RT_STC_bit     BT11_PPG_STC_bit
#define BT11_RT_TMCR2       BT11_PPG_TMCR2
#define BT11_RT_TMCR2_bit   BT11_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT12_RT
 **
 ***************************************************************************/
#define BT12_RT_PCSR        BT12_PPG_PRLL
#define BT12_RT_TMR         BT12_PPG_TMR
#define BT12_RT_TMCR        BT12_PPG_TMCR
#define BT12_RT_TMCR_bit    BT12_PPG_TMCR_bit.RT
#define BT12_RT_STC         BT12_PPG_STC
#define BT12_RT_STC_bit     BT12_PPG_STC_bit
#define BT12_RT_TMCR2       BT12_PPG_TMCR2
#define BT12_RT_TMCR2_bit   BT12_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT13_RT
 **
 ***************************************************************************/
#define BT13_RT_PCSR        BT13_PPG_PRLL
#define BT13_RT_TMR         BT13_PPG_TMR
#define BT13_RT_TMCR        BT13_PPG_TMCR
#define BT13_RT_TMCR_bit    BT13_PPG_TMCR_bit.RT
#define BT13_RT_STC         BT13_PPG_STC
#define BT13_RT_STC_bit     BT13_PPG_STC_bit
#define BT13_RT_TMCR2       BT13_PPG_TMCR2
#define BT13_RT_TMCR2_bit   BT13_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT14_RT
 **
 ***************************************************************************/
#define BT14_RT_PCSR        BT14_PPG_PRLL
#define BT14_RT_TMR         BT14_PPG_TMR
#define BT14_RT_TMCR        BT14_PPG_TMCR
#define BT14_RT_TMCR_bit    BT14_PPG_TMCR_bit.RT
#define BT14_RT_STC         BT14_PPG_STC
#define BT14_RT_STC_bit     BT14_PPG_STC_bit
#define BT14_RT_TMCR2       BT14_PPG_TMCR2
#define BT14_RT_TMCR2_bit   BT14_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT15_RT
 **
 ***************************************************************************/
#define BT15_RT_PCSR        BT15_PPG_PRLL
#define BT15_RT_TMR         BT15_PPG_TMR
#define BT15_RT_TMCR        BT15_PPG_TMCR
#define BT15_RT_TMCR_bit    BT15_PPG_TMCR_bit.RT
#define BT15_RT_STC         BT15_PPG_STC
#define BT15_RT_STC_bit     BT15_PPG_STC_bit
#define BT15_RT_TMCR2       BT15_PPG_TMCR2
#define BT15_RT_TMCR2_bit   BT15_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT0_PWC
 **
 ***************************************************************************/
#define BT0_PWC_DTBF        BT0_PPG_PRLH
#define BT0_PWC_TMCR        BT0_PPG_TMCR
#define BT0_PWC_TMCR_bit    BT0_PPG_TMCR_bit.PWC
#define BT0_PWC_STC         BT0_PPG_STC
#define BT0_PWC_STC_bit     BT0_PPG_STC_bit.PWC
#define BT0_PWC_TMCR2       BT0_PPG_TMCR2
#define BT0_PWC_TMCR2_bit   BT0_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT1_PWC
 **
 ***************************************************************************/
#define BT1_PWC_DTBF        BT1_PPG_PRLH
#define BT1_PWC_TMCR        BT1_PPG_TMCR
#define BT1_PWC_TMCR_bit    BT1_PPG_TMCR_bit.PWC
#define BT1_PWC_STC         BT1_PPG_STC
#define BT1_PWC_STC_bit     BT1_PPG_STC_bit.PWC
#define BT1_PWC_TMCR2       BT1_PPG_TMCR2
#define BT1_PWC_TMCR2_bit   BT1_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT2_PWC
 **
 ***************************************************************************/
#define BT2_PWC_DTBF        BT2_PPG_PRLH
#define BT2_PWC_TMCR        BT2_PPG_TMCR
#define BT2_PWC_TMCR_bit    BT2_PPG_TMCR_bit.PWC
#define BT2_PWC_STC         BT2_PPG_STC
#define BT2_PWC_STC_bit     BT2_PPG_STC_bit.PWC
#define BT2_PWC_TMCR2       BT2_PPG_TMCR2
#define BT2_PWC_TMCR2_bit   BT2_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT3_PWC
 **
 ***************************************************************************/
#define BT3_PWC_DTBF        BT3_PPG_PRLH
#define BT3_PWC_TMCR        BT3_PPG_TMCR
#define BT3_PWC_TMCR_bit    BT3_PPG_TMCR_bit.PWC
#define BT3_PWC_STC         BT3_PPG_STC
#define BT3_PWC_STC_bit     BT3_PPG_STC_bit.PWC
#define BT3_PWC_TMCR2       BT3_PPG_TMCR2
#define BT3_PWC_TMCR2_bit   BT3_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT4_PWC
 **
 ***************************************************************************/
#define BT4_PWC_DTBF        BT4_PPG_PRLH
#define BT4_PWC_TMCR        BT4_PPG_TMCR
#define BT4_PWC_TMCR_bit    BT4_PPG_TMCR_bit.PWC
#define BT4_PWC_STC         BT4_PPG_STC
#define BT4_PWC_STC_bit     BT4_PPG_STC_bit.PWC
#define BT4_PWC_TMCR2       BT4_PPG_TMCR2
#define BT4_PWC_TMCR2_bit   BT4_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT5_PWC
 **
 ***************************************************************************/
#define BT5_PWC_DTBF        BT5_PPG_PRLH
#define BT5_PWC_TMCR        BT5_PPG_TMCR
#define BT5_PWC_TMCR_bit    BT5_PPG_TMCR_bit.PWC
#define BT5_PWC_STC         BT5_PPG_STC
#define BT5_PWC_STC_bit     BT5_PPG_STC_bit.PWC
#define BT5_PWC_TMCR2       BT5_PPG_TMCR2
#define BT5_PWC_TMCR2_bit   BT5_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT6_PWC
 **
 ***************************************************************************/
#define BT6_PWC_DTBF        BT6_PPG_PRLH
#define BT6_PWC_TMCR        BT6_PPG_TMCR
#define BT6_PWC_TMCR_bit    BT6_PPG_TMCR_bit.PWC
#define BT6_PWC_STC         BT6_PPG_STC
#define BT6_PWC_STC_bit     BT6_PPG_STC_bit.PWC
#define BT6_PWC_TMCR2       BT6_PPG_TMCR2
#define BT6_PWC_TMCR2_bit   BT6_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT7_PWC
 **
 ***************************************************************************/
#define BT7_PWC_DTBF        BT7_PPG_PRLH
#define BT7_PWC_TMCR        BT7_PPG_TMCR
#define BT7_PWC_TMCR_bit    BT7_PPG_TMCR_bit.PWC
#define BT7_PWC_STC         BT7_PPG_STC
#define BT7_PWC_STC_bit     BT7_PPG_STC_bit.PWC
#define BT7_PWC_TMCR2       BT7_PPG_TMCR2
#define BT7_PWC_TMCR2_bit   BT7_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT8_PWC
 **
 ***************************************************************************/
#define BT8_PWC_DTBF        BT8_PPG_PRLH
#define BT8_PWC_TMCR        BT8_PPG_TMCR
#define BT8_PWC_TMCR_bit    BT8_PPG_TMCR_bit.PWC
#define BT8_PWC_STC         BT8_PPG_STC
#define BT8_PWC_STC_bit     BT8_PPG_STC_bit.PWC
#define BT8_PWC_TMCR2       BT8_PPG_TMCR2
#define BT8_PWC_TMCR2_bit   BT8_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT9_PWC
 **
 ***************************************************************************/
#define BT9_PWC_DTBF        BT9_PPG_PRLH
#define BT9_PWC_TMCR        BT9_PPG_TMCR
#define BT9_PWC_TMCR_bit    BT9_PPG_TMCR_bit.PWC
#define BT9_PWC_STC         BT9_PPG_STC
#define BT9_PWC_STC_bit     BT9_PPG_STC_bit.PWC
#define BT9_PWC_TMCR2       BT9_PPG_TMCR2
#define BT9_PWC_TMCR2_bit   BT9_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT10_PWC
 **
 ***************************************************************************/
#define BT10_PWC_DTBF       BT10_PPG_PRLH
#define BT10_PWC_TMCR       BT10_PPG_TMCR
#define BT10_PWC_TMCR_bit   BT10_PPG_TMCR_bit.PWC
#define BT10_PWC_STC        BT10_PPG_STC
#define BT10_PWC_STC_bit    BT10_PPG_STC_bit.PWC
#define BT10_PWC_TMCR2      BT10_PPG_TMCR2
#define BT10_PWC_TMCR2_bit  BT10_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT11_PWC
 **
 ***************************************************************************/
#define BT11_PWC_DTBF       BT11_PPG_PRLH
#define BT11_PWC_TMCR       BT11_PPG_TMCR
#define BT11_PWC_TMCR_bit   BT11_PPG_TMCR_bit.PWC
#define BT11_PWC_STC        BT11_PPG_STC
#define BT11_PWC_STC_bit    BT11_PPG_STC_bit.PWC
#define BT11_PWC_TMCR2      BT11_PPG_TMCR2
#define BT11_PWC_TMCR2_bit  BT11_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT12_PWC
 **
 ***************************************************************************/
#define BT12_PWC_DTBF       BT12_PPG_PRLH
#define BT12_PWC_TMCR       BT12_PPG_TMCR
#define BT12_PWC_TMCR_bit   BT12_PPG_TMCR_bit.PWC
#define BT12_PWC_STC        BT12_PPG_STC
#define BT12_PWC_STC_bit    BT12_PPG_STC_bit.PWC
#define BT12_PWC_TMCR2      BT12_PPG_TMCR2
#define BT12_PWC_TMCR2_bit  BT12_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT13_PWC
 **
 ***************************************************************************/
#define BT13_PWC_DTBF       BT13_PPG_PRLH
#define BT13_PWC_TMCR       BT13_PPG_TMCR
#define BT13_PWC_TMCR_bit   BT13_PPG_TMCR_bit.PWC
#define BT13_PWC_STC        BT13_PPG_STC
#define BT13_PWC_STC_bit    BT13_PPG_STC_bit.PWC
#define BT13_PWC_TMCR2      BT13_PPG_TMCR2
#define BT13_PWC_TMCR2_bit  BT13_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT14_PWC
 **
 ***************************************************************************/
#define BT14_PWC_DTBF       BT14_PPG_PRLH
#define BT14_PWC_TMCR       BT14_PPG_TMCR
#define BT14_PWC_TMCR_bit   BT14_PPG_TMCR_bit.PWC
#define BT14_PWC_STC        BT14_PPG_STC
#define BT14_PWC_STC_bit    BT14_PPG_STC_bit.PWC
#define BT14_PWC_TMCR2      BT14_PPG_TMCR2
#define BT14_PWC_TMCR2_bit  BT14_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT15_PWC
 **
 ***************************************************************************/
#define BT15_PWC_DTBF       BT15_PPG_PRLH
#define BT15_PWC_TMCR       BT15_PPG_TMCR
#define BT15_PWC_TMCR_bit   BT15_PPG_TMCR_bit.PWC
#define BT15_PWC_STC        BT15_PPG_STC
#define BT15_PWC_STC_bit    BT15_PPG_STC_bit.PWC
#define BT15_PWC_TMCR2      BT15_PPG_TMCR2
#define BT15_PWC_TMCR2_bit  BT15_PPG_TMCR2_bit

/***************************************************************************
 **
 ** BT I/O Select
 **
 ***************************************************************************/
__IO_REG8_BIT( BTSEL0123,         0x40025101,__READ_WRITE ,__btsel0123_bits);
__IO_REG8_BIT( BTSEL4567,         0x40025301,__READ_WRITE ,__btsel4567_bits);
__IO_REG8_BIT( BTSEL86AB,         0x40025501,__READ_WRITE ,__btsel89ab_bits);
__IO_REG8_BIT( BTSELCDEF,         0x40025701,__READ_WRITE ,__btselcdef_bits);
__IO_REG16_BIT(BTSSSR,            0x40025FFC,__WRITE      ,__btsssr_bits);

/***************************************************************************
 **
 ** QPRC0
 **
 ***************************************************************************/
__IO_REG16(    QPRC0_QPCR,        0x40026000,__READ_WRITE );
__IO_REG16(    QPRC0_QRCR,        0x40026004,__READ_WRITE );
__IO_REG16(    QPRC0_QPCCR,       0x40026008,__READ_WRITE );
__IO_REG16(    QPRC0_QPRCR,       0x4002600C,__READ_WRITE );
__IO_REG16(    QPRC0_QMPR,        0x40026010,__READ_WRITE );
__IO_REG16_BIT(QPRC0_QICR,        0x40026014,__READ_WRITE ,__qdu_qicr_bits);
__IO_REG16_BIT(QPRC0_QCR,         0x40026018,__READ_WRITE ,__qdu_qcr_bits);
__IO_REG16_BIT(QPRC0_QECR,        0x4002601C,__READ_WRITE ,__qdu_qecr_bits);
__IO_REG32_BIT(QPRC0_QPRCRR,      0x4002603C,__READ       ,__qdu_qprcrr_bits);

/***************************************************************************
 **
 ** QPRC1
 **
 ***************************************************************************/
__IO_REG16(    QPRC1_QPCR,        0x40026040,__READ_WRITE );
__IO_REG16(    QPRC1_QRCR,        0x40026044,__READ_WRITE );
__IO_REG16(    QPRC1_QPCCR,       0x40026048,__READ_WRITE );
__IO_REG16(    QPRC1_QPRCR,       0x4002604C,__READ_WRITE );
__IO_REG16(    QPRC1_QMPR,        0x40026050,__READ_WRITE );
__IO_REG16_BIT(QPRC1_QICR,        0x40026054,__READ_WRITE ,__qdu_qicr_bits);
__IO_REG16_BIT(QPRC1_QCR,         0x40026058,__READ_WRITE ,__qdu_qcr_bits);
__IO_REG16_BIT(QPRC1_QECR,        0x4002605C,__READ_WRITE ,__qdu_qecr_bits);
__IO_REG32_BIT(QPRC1_QPRCRR,      0x4002607C,__READ       ,__qdu_qprcrr_bits);

/***************************************************************************
 **
 ** QPRC2
 **
 ***************************************************************************/
__IO_REG16(    QPRC2_QPCR,        0x40026080,__READ_WRITE );
__IO_REG16(    QPRC2_QRCR,        0x40026084,__READ_WRITE );
__IO_REG16(    QPRC2_QPCCR,       0x40026088,__READ_WRITE );
__IO_REG16(    QPRC2_QPRCR,       0x4002608C,__READ_WRITE );
__IO_REG16(    QPRC2_QMPR,        0x40026090,__READ_WRITE );
__IO_REG16_BIT(QPRC2_QICR,        0x40026094,__READ_WRITE ,__qdu_qicr_bits);
__IO_REG16_BIT(QPRC2_QCR,         0x40026098,__READ_WRITE ,__qdu_qcr_bits);
__IO_REG16_BIT(QPRC2_QECR,        0x4002609C,__READ_WRITE ,__qdu_qecr_bits);
__IO_REG32_BIT(QPRC2_QPRCRR,      0x400260BC,__READ       ,__qdu_qprcrr_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG8_BIT( ADC0_ADSR,         0x40027000,__READ_WRITE ,__adc_adsr_bits);
__IO_REG8_BIT( ADC0_ADCR,         0x40027001,__READ_WRITE ,__adc_adcr_bits);
__IO_REG8_BIT( ADC0_SFNS,         0x40027008,__READ_WRITE ,__adc_sfns_bits);
__IO_REG8_BIT( ADC0_SCCR,         0x40027009,__READ_WRITE ,__adc_sccr_bits);
__IO_REG32_BIT(ADC0_SCFD,         0x4002700C,__READ       ,__adc_scfd_bits);
__IO_REG8_BIT( ADC0_SCIS2,        0x40027010,__READ_WRITE ,__adc_scis2_bits);
__IO_REG8_BIT( ADC0_SCIS3,        0x40027011,__READ_WRITE ,__adc_scis3_bits);
__IO_REG8_BIT( ADC0_SCIS0,        0x40027014,__READ_WRITE ,__adc_scis0_bits);
__IO_REG8_BIT( ADC0_SCIS1,        0x40027015,__READ_WRITE ,__adc_scis1_bits);
__IO_REG8_BIT( ADC0_PFNS,         0x40027018,__READ_WRITE ,__adc_pfns_bits);
__IO_REG8_BIT( ADC0_PCCR,         0x40027019,__READ_WRITE ,__adc_pccr_bits);
__IO_REG32_BIT(ADC0_PCFD,         0x4002701C,__READ       ,__adc_pcfd_bits);
__IO_REG8_BIT( ADC0_PCIS,         0x40027020,__READ_WRITE ,__adc_pcis_bits);
__IO_REG8_BIT( ADC0_CMPCR,        0x40027024,__READ_WRITE ,__adc_cmpcr_bits);
__IO_REG16_BIT(ADC0_CMPD,         0x40027026,__READ_WRITE ,__adc_cmpd_bits);
__IO_REG8_BIT( ADC0_ADSS2,        0x40027028,__READ_WRITE ,__adc_adss2_bits);
__IO_REG8_BIT( ADC0_ADSS3,        0x40027029,__READ_WRITE ,__adc_adss3_bits);
__IO_REG8_BIT( ADC0_ADSS0,        0x4002702C,__READ_WRITE ,__adc_adss0_bits);
__IO_REG8_BIT( ADC0_ADSS1,        0x4002702D,__READ_WRITE ,__adc_adss1_bits);
__IO_REG8_BIT( ADC0_ADST1,        0x40027030,__READ_WRITE ,__adc_adst1_bits);
__IO_REG8_BIT( ADC0_ADST0,        0x40027031,__READ_WRITE ,__adc_adst0_bits);
__IO_REG8_BIT( ADC0_ADCT,         0x40027034,__READ_WRITE ,__adc_adct_bits);
__IO_REG8_BIT( ADC0_PRTSL,        0x40027038,__READ_WRITE ,__adc_prtsl_bits);
__IO_REG8_BIT( ADC0_SCTSL,        0x40027039,__READ_WRITE ,__adc_sctsl_bits);
__IO_REG8_BIT( ADC0_ADCEN,        0x4002703C,__READ_WRITE ,__adc_adcen_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG8_BIT( ADC1_ADSR,         0x40027100,__READ_WRITE ,__adc_adsr_bits);
__IO_REG8_BIT( ADC1_ADCR,         0x40027101,__READ_WRITE ,__adc_adcr_bits);
__IO_REG8_BIT( ADC1_SFNS,         0x40027108,__READ_WRITE ,__adc_sfns_bits);
__IO_REG8_BIT( ADC1_SCCR,         0x40027109,__READ_WRITE ,__adc_sccr_bits);
__IO_REG32_BIT(ADC1_SCFD,         0x4002710C,__READ       ,__adc_scfd_bits);
__IO_REG8_BIT( ADC1_SCIS2,        0x40027110,__READ_WRITE ,__adc_scis2_bits);
__IO_REG8_BIT( ADC1_SCIS3,        0x40027111,__READ_WRITE ,__adc_scis3_bits);
__IO_REG8_BIT( ADC1_SCIS0,        0x40027114,__READ_WRITE ,__adc_scis0_bits);
__IO_REG8_BIT( ADC1_SCIS1,        0x40027115,__READ_WRITE ,__adc_scis1_bits);
__IO_REG8_BIT( ADC1_PFNS,         0x40027118,__READ_WRITE ,__adc_pfns_bits);
__IO_REG8_BIT( ADC1_PCCR,         0x40027119,__READ_WRITE ,__adc_pccr_bits);
__IO_REG32_BIT(ADC1_PCFD,         0x4002711C,__READ       ,__adc_pcfd_bits);
__IO_REG8_BIT( ADC1_PCIS,         0x40027120,__READ_WRITE ,__adc_pcis_bits);
__IO_REG8_BIT( ADC1_CMPCR,        0x40027124,__READ_WRITE ,__adc_cmpcr_bits);
__IO_REG16_BIT(ADC1_CMPD,         0x40027126,__READ_WRITE ,__adc_cmpd_bits);
__IO_REG8_BIT( ADC1_ADSS2,        0x40027128,__READ_WRITE ,__adc_adss2_bits);
__IO_REG8_BIT( ADC1_ADSS3,        0x40027129,__READ_WRITE ,__adc_adss3_bits);
__IO_REG8_BIT( ADC1_ADSS0,        0x4002712C,__READ_WRITE ,__adc_adss0_bits);
__IO_REG8_BIT( ADC1_ADSS1,        0x4002712D,__READ_WRITE ,__adc_adss1_bits);
__IO_REG8_BIT( ADC1_ADST1,        0x40027130,__READ_WRITE ,__adc_adst1_bits);
__IO_REG8_BIT( ADC1_ADST0,        0x40027131,__READ_WRITE ,__adc_adst0_bits);
__IO_REG8_BIT( ADC1_ADCT,         0x40027134,__READ_WRITE ,__adc_adct_bits);
__IO_REG8_BIT( ADC1_PRTSL,        0x40027138,__READ_WRITE ,__adc_prtsl_bits);
__IO_REG8_BIT( ADC1_SCTSL,        0x40027139,__READ_WRITE ,__adc_sctsl_bits);
__IO_REG8_BIT( ADC1_ADCEN,        0x4002713C,__READ_WRITE ,__adc_adcen_bits);

/***************************************************************************
 **
 ** ADC2
 **
 ***************************************************************************/
__IO_REG8_BIT( ADC2_ADSR,         0x40027200,__READ_WRITE ,__adc_adsr_bits);
__IO_REG8_BIT( ADC2_ADCR,         0x40027201,__READ_WRITE ,__adc_adcr_bits);
__IO_REG8_BIT( ADC2_SFNS,         0x40027208,__READ_WRITE ,__adc_sfns_bits);
__IO_REG8_BIT( ADC2_SCCR,         0x40027209,__READ_WRITE ,__adc_sccr_bits);
__IO_REG32_BIT(ADC2_SCFD,         0x4002720C,__READ       ,__adc_scfd_bits);
__IO_REG8_BIT( ADC2_SCIS2,        0x40027210,__READ_WRITE ,__adc_scis2_bits);
__IO_REG8_BIT( ADC2_SCIS3,        0x40027211,__READ_WRITE ,__adc_scis3_bits);
__IO_REG8_BIT( ADC2_SCIS0,        0x40027214,__READ_WRITE ,__adc_scis0_bits);
__IO_REG8_BIT( ADC2_SCIS1,        0x40027215,__READ_WRITE ,__adc_scis1_bits);
__IO_REG8_BIT( ADC2_PFNS,         0x40027218,__READ_WRITE ,__adc_pfns_bits);
__IO_REG8_BIT( ADC2_PCCR,         0x40027219,__READ_WRITE ,__adc_pccr_bits);
__IO_REG32_BIT(ADC2_PCFD,         0x4002721C,__READ       ,__adc_pcfd_bits);
__IO_REG8_BIT( ADC2_PCIS,         0x40027220,__READ_WRITE ,__adc_pcis_bits);
__IO_REG8_BIT( ADC2_CMPCR,        0x40027224,__READ_WRITE ,__adc_cmpcr_bits);
__IO_REG16_BIT(ADC2_CMPD,         0x40027226,__READ_WRITE ,__adc_cmpd_bits);
__IO_REG8_BIT( ADC2_ADSS2,        0x40027228,__READ_WRITE ,__adc_adss2_bits);
__IO_REG8_BIT( ADC2_ADSS3,        0x40027229,__READ_WRITE ,__adc_adss3_bits);
__IO_REG8_BIT( ADC2_ADSS0,        0x4002722C,__READ_WRITE ,__adc_adss0_bits);
__IO_REG8_BIT( ADC2_ADSS1,        0x4002722D,__READ_WRITE ,__adc_adss1_bits);
__IO_REG8_BIT( ADC2_ADST1,        0x40027230,__READ_WRITE ,__adc_adst1_bits);
__IO_REG8_BIT( ADC2_ADST0,        0x40027231,__READ_WRITE ,__adc_adst0_bits);
__IO_REG8_BIT( ADC2_ADCT,         0x40027234,__READ_WRITE ,__adc_adct_bits);
__IO_REG8_BIT( ADC2_PRTSL,        0x40027238,__READ_WRITE ,__adc_prtsl_bits);
__IO_REG8_BIT( ADC2_SCTSL,        0x40027239,__READ_WRITE ,__adc_sctsl_bits);
__IO_REG8_BIT( ADC2_ADCEN,        0x4002723C,__READ_WRITE ,__adc_adcen_bits);

/***************************************************************************
 **
 ** CR Trim
 **
 ***************************************************************************/
__IO_REG8_BIT( MCR_PSR,           0x4002E000,__READ_WRITE ,__mcr_psr_bits);
__IO_REG16_BIT(MCR_FTRM,          0x4002E004,__READ_WRITE ,__mcr_ftrm_bits);
__IO_REG32(    MCR_RLR,           0x4002E00C,__READ_WRITE );

/***************************************************************************
 **
 ** EXTI
 **
 ***************************************************************************/
__IO_REG32_BIT(ENIR,              0x40030000,__READ_WRITE ,__enir_bits);
__IO_REG32_BIT(EIRR,              0x40030004,__READ       ,__eirr_bits);
__IO_REG32_BIT(EICL,              0x40030008,__READ_WRITE ,__eicl_bits);
__IO_REG32_BIT(ELVR,              0x4003000C,__READ_WRITE ,__elvr_bits);
__IO_REG32_BIT(ELVR1,             0x40030010,__READ_WRITE ,__elvr1_bits);
__IO_REG16_BIT(NMIRR,             0x40030014,__READ       ,__nmirr_bits);
__IO_REG16_BIT(NMICL,             0x40030018,__READ_WRITE ,__nmicl_bits);

/***************************************************************************
 **
 ** INT Req Read
 **
 ***************************************************************************/
__IO_REG32_BIT(DRQSEL,            0x40031000,__READ_WRITE ,__drqsel_bits);
__IO_REG32_BIT(ODDPKS,            0x40031008,__READ_WRITE ,__oddpks_bits);
__IO_REG32_BIT(EXC02MON,          0x40031010,__READ       ,__exc02mon_bits);
__IO_REG32_BIT(IRQ00MON,          0x40031014,__READ       ,__irqmon0_bits);
__IO_REG32_BIT(IRQ01MON,          0x40031018,__READ       ,__irqmon1_bits);
__IO_REG32_BIT(IRQ02MON,          0x4003101C,__READ       ,__irqmon2_bits);
__IO_REG32_BIT(IRQ03MON,          0x40031020,__READ       ,__irqmon3_bits);
__IO_REG32_BIT(IRQ04MON,          0x40031024,__READ       ,__irqmon4_bits);
__IO_REG32_BIT(IRQ05MON,          0x40031028,__READ       ,__irqmon5_bits);
__IO_REG32_BIT(IRQ06MON,          0x4003102C,__READ       ,__irqmon6_bits);
__IO_REG32_BIT(IRQ07MON,          0x40031030,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ08MON,          0x40031034,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ09MON,          0x40031038,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ10MON,          0x4003103C,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ11MON,          0x40031040,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ12MON,          0x40031044,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ13MON,          0x40031048,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ14MON,          0x4003104C,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ15MON,          0x40031050,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ16MON,          0x40031054,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ17MON,          0x40031058,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ18MON,          0x4003105C,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ19MON,          0x40031060,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ20MON,          0x40031064,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ21MON,          0x40031068,__READ       ,__irqmon7_bits);
__IO_REG32_BIT(IRQ22MON,          0x4003106C,__READ       ,__irqmon8_bits);
__IO_REG32_BIT(IRQ23MON,          0x40031070,__READ       ,__irqmon23_bits);
__IO_REG32_BIT(IRQ24MON,          0x40031074,__READ       ,__irqmon24_bits);
__IO_REG32_BIT(IRQ25MON,          0x40031078,__READ       ,__irqmon25_bits);
__IO_REG32_BIT(IRQ26MON,          0x4003107C,__READ       ,__irqmon25_bits);
__IO_REG32_BIT(IRQ27MON,          0x40031080,__READ       ,__irqmon25_bits);
__IO_REG32_BIT(IRQ28MON,          0x40031084,__READ       ,__irqmon28_bits);
__IO_REG32_BIT(IRQ29MON,          0x40031088,__READ       ,__irqmon29_bits);
__IO_REG32_BIT(IRQ30MON,          0x4003108C,__READ       ,__irqmon30_bits);
__IO_REG32_BIT(IRQ31MON,          0x40031090,__READ       ,__irqmon31_bits);
__IO_REG32_BIT(IRQ32MON,          0x40031094,__READ       ,__irqmon32_bits);
__IO_REG32_BIT(IRQ34MON,          0x4003109C,__READ       ,__irqmon34_bits);
__IO_REG32_BIT(IRQ35MON,          0x400310A0,__READ       ,__irqmon35_bits);
__IO_REG32_BIT(IRQ36MON,          0x400310A4,__READ       ,__irqmon36_bits);
__IO_REG32_BIT(IRQ37MON,          0x400310A8,__READ       ,__irqmon37_bits);
__IO_REG32_BIT(IRQ38MON,          0x400310AC,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ39MON,          0x400310B0,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ40MON,          0x400310B4,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ41MON,          0x400310B8,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ42MON,          0x400310BC,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ43MON,          0x400310C0,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ44MON,          0x400310C4,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ45MON,          0x400310C8,__READ       ,__irqmon38_bits);
__IO_REG32_BIT(IRQ46MON,          0x400310CC,__READ       ,__irqmon46_bits);
__IO_REG32_BIT(DRQSEL1,           0x40031200,__READ_WRITE ,__drqsel1_bits);
__IO_REG32_BIT(DQESEL,            0x40031204,__READ_WRITE ,__dqesel_bits);
__IO_REG32_BIT(ODDPKS1,           0x4003120C,__READ_WRITE ,__oddpks_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(PFR0,              0x40033000,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PFR1,              0x40033004,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PFR2,              0x40033008,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PFR3,              0x4003300C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PFR4,              0x40033010,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PFR5,              0x40033014,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PFR6,              0x40033018,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PFR7,              0x4003301C,__READ_WRITE ,__port7_bits);
__IO_REG32_BIT(PFR8,              0x40033020,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PFR9,              0x40033024,__READ_WRITE ,__port9_bits);
__IO_REG32_BIT(PFRA,              0x40033028,__READ_WRITE ,__porta_bits);
__IO_REG32_BIT(PFRB,              0x4003302C,__READ_WRITE ,__portb_bits);
__IO_REG32_BIT(PFRC,              0x40033030,__READ_WRITE ,__portc_bits);
__IO_REG32_BIT(PFRD,              0x40033034,__READ_WRITE ,__portd_bits);
__IO_REG32_BIT(PFRE,              0x40033038,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PFRF,              0x4003303C,__READ_WRITE ,__portf_bits);
__IO_REG32_BIT(PCR0,              0x40033100,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PCR1,              0x40033104,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PCR2,              0x40033108,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PCR3,              0x4003310C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PCR4,              0x40033110,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PCR5,              0x40033114,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PCR6,              0x40033118,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PCR7,              0x4003311C,__READ_WRITE ,__port7_bits);
__IO_REG32_BIT(PCR8,              0x40033120,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PCR9,              0x40033124,__READ_WRITE ,__port9_bits);
__IO_REG32_BIT(PCRA,              0x40033128,__READ_WRITE ,__porta_bits);
__IO_REG32_BIT(PCRB,              0x4003312C,__READ_WRITE ,__portb_bits);
__IO_REG32_BIT(PCRC,              0x40033130,__READ_WRITE ,__portc_bits);
__IO_REG32_BIT(PCRD,              0x40033134,__READ_WRITE ,__portd_bits);
__IO_REG32_BIT(PCRE,              0x40033138,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PCRF,              0x4003313C,__READ_WRITE ,__portf_bits);
__IO_REG32_BIT(DDR0,              0x40033200,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(DDR1,              0x40033204,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(DDR2,              0x40033208,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(DDR3,              0x4003320C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(DDR4,              0x40033210,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(DDR5,              0x40033214,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(DDR6,              0x40033218,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(DDR7,              0x4003321C,__READ_WRITE ,__port7_bits);
__IO_REG32_BIT(DDR8,              0x40033220,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(DDR9,              0x40033224,__READ_WRITE ,__port9_bits);
__IO_REG32_BIT(DDRA,              0x40033228,__READ_WRITE ,__porta_bits);
__IO_REG32_BIT(DDRB,              0x4003322C,__READ_WRITE ,__portb_bits);
__IO_REG32_BIT(DDRC,              0x40033230,__READ_WRITE ,__portc_bits);
__IO_REG32_BIT(DDRD,              0x40033234,__READ_WRITE ,__portd_bits);
__IO_REG32_BIT(DDRE,              0x40033238,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(DDRF,              0x4003323C,__READ_WRITE ,__portf_bits);
__IO_REG32_BIT(PDIR0,             0x40033300,__READ       ,__port0_bits);
__IO_REG32_BIT(PDIR1,             0x40033304,__READ       ,__port1_bits);
__IO_REG32_BIT(PDIR2,             0x40033308,__READ       ,__port2_bits);
__IO_REG32_BIT(PDIR3,             0x4003330C,__READ       ,__port3_bits);
__IO_REG32_BIT(PDIR4,             0x40033310,__READ       ,__port4_bits);
__IO_REG32_BIT(PDIR5,             0x40033314,__READ       ,__port5_bits);
__IO_REG32_BIT(PDIR6,             0x40033318,__READ       ,__port6_bits);
__IO_REG32_BIT(PDIR7,             0x4003331C,__READ       ,__port7_bits);
__IO_REG32_BIT(PDIR8,             0x40033320,__READ       ,__port8_bits);
__IO_REG32_BIT(PDIR9,             0x40033324,__READ       ,__port9_bits);
__IO_REG32_BIT(PDIRA,             0x40033328,__READ       ,__porta_bits);
__IO_REG32_BIT(PDIRB,             0x4003332C,__READ       ,__portb_bits);
__IO_REG32_BIT(PDIRC,             0x40033330,__READ       ,__portc_bits);
__IO_REG32_BIT(PDIRD,             0x40033334,__READ       ,__portd_bits);
__IO_REG32_BIT(PDIRE,             0x40033338,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PDIRF,             0x4003333C,__READ_WRITE ,__portf_bits);
__IO_REG32_BIT(PDOR0,             0x40033400,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PDOR1,             0x40033404,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PDOR2,             0x40033408,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PDOR3,             0x4003340C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PDOR4,             0x40033410,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PDOR5,             0x40033414,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PDOR6,             0x40033418,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PDOR7,             0x4003341C,__READ_WRITE ,__port7_bits);
__IO_REG32_BIT(PDOR8,             0x40033420,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PDOR9,             0x40033424,__READ_WRITE ,__port9_bits);
__IO_REG32_BIT(PDORA,             0x40033428,__READ_WRITE ,__porta_bits);
__IO_REG32_BIT(PDORB,             0x4003342C,__READ_WRITE ,__portb_bits);
__IO_REG32_BIT(PDORC,             0x40033430,__READ_WRITE ,__portc_bits);
__IO_REG32_BIT(PDORD,             0x40033434,__READ_WRITE ,__portd_bits);
__IO_REG32_BIT(PDORE,             0x40033438,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PDORF,             0x4003343C,__READ_WRITE ,__portf_bits);
__IO_REG32_BIT(ADE,               0x40033500,__READ_WRITE ,__ade_bits);
__IO_REG32_BIT(SPSR,              0x40033580,__READ_WRITE ,__spsr_bits);
__IO_REG32_BIT(EPFR00,            0x40033600,__READ_WRITE ,__epfr00_bits);
__IO_REG32_BIT(EPFR01,            0x40033604,__READ_WRITE ,__epfr01_bits);
__IO_REG32_BIT(EPFR02,            0x40033608,__READ_WRITE ,__epfr02_bits);
__IO_REG32_BIT(EPFR03,            0x4003360C,__READ_WRITE ,__epfr03_bits);
__IO_REG32_BIT(EPFR04,            0x40033610,__READ_WRITE ,__epfr04_bits);
__IO_REG32_BIT(EPFR05,            0x40033614,__READ_WRITE ,__epfr05_bits);
__IO_REG32_BIT(EPFR06,            0x40033618,__READ_WRITE ,__epfr06_bits);
__IO_REG32_BIT(EPFR07,            0x4003361C,__READ_WRITE ,__epfr07_bits);
__IO_REG32_BIT(EPFR08,            0x40033620,__READ_WRITE ,__epfr08_bits);
__IO_REG32_BIT(EPFR09,            0x40033624,__READ_WRITE ,__epfr09_bits);
__IO_REG32_BIT(EPFR10,            0x40033628,__READ_WRITE ,__epfr10_bits);
__IO_REG32_BIT(EPFR11,            0x4003362C,__READ_WRITE ,__epfr11_bits);
__IO_REG32_BIT(EPFR12,            0x40033630,__READ_WRITE ,__epfr12_bits);
__IO_REG32_BIT(EPFR13,            0x40033634,__READ_WRITE ,__epfr13_bits);
__IO_REG32_BIT(EPFR14,            0x40033638,__READ_WRITE ,__epfr14_bits);
__IO_REG32_BIT(EPFR15,            0x4003363C,__READ_WRITE ,__epfr15_bits);
__IO_REG32_BIT(PZR0,              0x40033700,__READ_WRITE ,__port0_bits);
__IO_REG32_BIT(PZR1,              0x40033704,__READ_WRITE ,__port1_bits);
__IO_REG32_BIT(PZR2,              0x40033708,__READ_WRITE ,__port2_bits);
__IO_REG32_BIT(PZR3,              0x4003370C,__READ_WRITE ,__port3_bits);
__IO_REG32_BIT(PZR4,              0x40033710,__READ_WRITE ,__port4_bits);
__IO_REG32_BIT(PZR5,              0x40033714,__READ_WRITE ,__port5_bits);
__IO_REG32_BIT(PZR6,              0x40033718,__READ_WRITE ,__port6_bits);
__IO_REG32_BIT(PZR7,              0x4003371C,__READ_WRITE ,__port7_bits);
__IO_REG32_BIT(PZR8,              0x40033720,__READ_WRITE ,__port8_bits);
__IO_REG32_BIT(PZR9,              0x40033724,__READ_WRITE ,__port9_bits);
__IO_REG32_BIT(PZRA,              0x40033728,__READ_WRITE ,__porta_bits);
__IO_REG32_BIT(PZRB,              0x4003372C,__READ_WRITE ,__portb_bits);
__IO_REG32_BIT(PZRC,              0x40033730,__READ_WRITE ,__portc_bits);
__IO_REG32_BIT(PZRD,              0x40033734,__READ_WRITE ,__portd_bits);
__IO_REG32_BIT(PZRE,              0x40033738,__READ_WRITE ,__porte_bits);
__IO_REG32_BIT(PZRF,              0x4003373C,__READ_WRITE ,__portf_bits);

/***************************************************************************
 **
 ** LVD
 **
 ***************************************************************************/
__IO_REG8_BIT( LVD_CTL,           0x40035000,__READ_WRITE ,__lvd_ctl_bits);
__IO_REG8_BIT( LVD_STR,           0x40035004,__READ       ,__lvd_str_bits);
__IO_REG8_BIT( LVD_CLR,           0x40035008,__READ_WRITE ,__lvd_clr_bits);
__IO_REG32(    LVD_RLR,           0x4003500C,__READ_WRITE );
__IO_REG8_BIT( LVD_STR2,          0x40035010,__READ       ,__lvd_str2_bits);

/***************************************************************************
 **
 ** USB Clock
 **
 ***************************************************************************/
__IO_REG8_BIT( UCCR,              0x40036000,__READ_WRITE ,__uccr_bits);
__IO_REG8_BIT( UPCR1,             0x40036004,__READ_WRITE ,__upcr1_bits);
__IO_REG8_BIT( UPCR2,             0x40036008,__READ_WRITE ,__upcr2_bits);
__IO_REG8_BIT( UPCR3,             0x4003600C,__READ_WRITE ,__upcr3_bits);
__IO_REG8_BIT( UPCR4,             0x40036010,__READ_WRITE ,__upcr4_bits);
__IO_REG8_BIT( UP_STR,            0x40036014,__READ       ,__up_str_bits);
__IO_REG8_BIT( UPINT_ENR,         0x40036018,__READ_WRITE ,__upint_enr_bits);
__IO_REG8(     UPINT_CLR,         0x4003601C,__WRITE      );
__IO_REG8_BIT( UPINT_STR,         0x40036020,__READ       ,__upin_str_bits);
__IO_REG8_BIT( UPCR5,             0x40036024,__READ_WRITE ,__upcr5_bits);
__IO_REG8_BIT( UPCR6,             0x40036028,__READ_WRITE ,__upcr6_bits);
__IO_REG8_BIT( UPCR7,             0x4003602C,__READ_WRITE ,__upcr7_bits);
__IO_REG8_BIT( USBEN0,            0x40036030,__READ_WRITE ,__usben0_bits);
__IO_REG8_BIT( USBEN1,            0x40036034,__READ_WRITE ,__usben1_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG8_BIT( UART0_SMR,         0x40038000,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART0_SCR,         0x40038001,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART0_ESCR,        0x40038004,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART0_SSR,         0x40038005,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART0_RDR,         0x40038008,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART0_TDR     UART0_RDR
#define UART0_TDR_bit UART0_RDR_bit
__IO_REG16_BIT(UART0_BGR,         0x4003800C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG8_BIT( UART1_SMR,         0x40038100,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART1_SCR,         0x40038101,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART1_ESCR,        0x40038104,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART1_SSR,         0x40038105,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART1_RDR,         0x40038108,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART1_TDR     UART1_RDR
#define UART1_TDR_bit UART1_RDR_bit
__IO_REG16_BIT(UART1_BGR,         0x4003810C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG8_BIT( UART2_SMR,         0x40038200,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART2_SCR,         0x40038201,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART2_ESCR,        0x40038204,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART2_SSR,         0x40038205,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART2_RDR,         0x40038208,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART2_TDR     UART2_RDR
#define UART2_TDR_bit UART2_RDR_bit
__IO_REG16_BIT(UART2_BGR,         0x4003820C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
__IO_REG8_BIT( UART3_SMR,         0x40038300,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART3_SCR,         0x40038301,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART3_ESCR,        0x40038304,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART3_SSR,         0x40038305,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART3_RDR,         0x40038308,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART3_TDR     UART3_RDR
#define UART3_TDR_bit UART3_RDR_bit
__IO_REG16_BIT(UART3_BGR,         0x4003830C,__READ_WRITE ,__mfsx_bgr_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
__IO_REG8_BIT( UART4_SMR,         0x40038400,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART4_SCR,         0x40038401,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART4_ESCR,        0x40038404,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART4_SSR,         0x40038405,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART4_RDR,         0x40038408,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART4_TDR     UART4_RDR
#define UART4_TDR_bit UART4_RDR_bit
__IO_REG16_BIT(UART4_BGR,         0x4003840C,__READ_WRITE ,__mfsx_bgr_bits);
__IO_REG16_BIT(UART4_FCR,         0x40038414,__READ_WRITE ,__mfsx_fcr_bits);
__IO_REG8(     UART4_FBYTE1,      0x40038418,__READ_WRITE );
__IO_REG8(     UART4_FBYTE2,      0x40038419,__READ_WRITE );

/***************************************************************************
 **
 ** UART5
 **
 ***************************************************************************/
__IO_REG8_BIT( UART5_SMR,         0x40038500,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART5_SCR,         0x40038501,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART5_ESCR,        0x40038504,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART5_SSR,         0x40038505,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART5_RDR,         0x40038508,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART5_TDR     UART5_RDR
#define UART5_TDR_bit UART5_RDR_bit
__IO_REG16_BIT(UART5_BGR,         0x4003850C,__READ_WRITE ,__mfsx_bgr_bits);
__IO_REG16_BIT(UART5_FCR,         0x40038514,__READ_WRITE ,__mfsx_fcr_bits);
__IO_REG8(     UART5_FBYTE1,      0x40038518,__READ_WRITE );
__IO_REG8(     UART5_FBYTE2,      0x40038519,__READ_WRITE );

/***************************************************************************
 **
 ** UART6
 **
 ***************************************************************************/
__IO_REG8_BIT( UART6_SMR,         0x40038600,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART6_SCR,         0x40038601,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART6_ESCR,        0x40038604,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART6_SSR,         0x40038605,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART6_RDR,         0x40038608,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART6_TDR     UART6_RDR
#define UART6_TDR_bit UART6_RDR_bit
__IO_REG16_BIT(UART6_BGR,         0x4003860C,__READ_WRITE ,__mfsx_bgr_bits);
__IO_REG16_BIT(UART6_FCR,         0x40038614,__READ_WRITE ,__mfsx_fcr_bits);
__IO_REG8(     UART6_FBYTE1,      0x40038618,__READ_WRITE );
__IO_REG8(     UART6_FBYTE2,      0x40038619,__READ_WRITE );

/***************************************************************************
 **
 ** UART7
 **
 ***************************************************************************/
__IO_REG8_BIT( UART7_SMR,         0x40038700,__READ_WRITE ,__mfsx_smr_bits);
__IO_REG8_BIT( UART7_SCR,         0x40038701,__READ_WRITE ,__mfsx_scr_bits);
__IO_REG8_BIT( UART7_ESCR,        0x40038704,__READ_WRITE ,__mfsx_escr_bits);
__IO_REG8_BIT( UART7_SSR,         0x40038705,__READ_WRITE ,__mfsx_ssr_bits);
__IO_REG16_BIT(UART7_RDR,         0x40038708,__READ_WRITE ,__mfsx_rdr_tdr_bits);
#define UART7_TDR     UART7_RDR
#define UART7_TDR_bit UART7_RDR_bit
__IO_REG16_BIT(UART7_BGR,         0x4003870C,__READ_WRITE ,__mfsx_bgr_bits);
__IO_REG16_BIT(UART7_FCR,         0x40038714,__READ_WRITE ,__mfsx_fcr_bits);
__IO_REG8(     UART7_FBYTE1,      0x40038718,__READ_WRITE );
__IO_REG8(     UART7_FBYTE2,      0x40038719,__READ_WRITE );

/***************************************************************************
 **
 ** CSIO0
 **
 ***************************************************************************/
#define CSIO0_SMR       UART0_SMR
#define CSIO0_SMR_bit   UART0_SMR_bit.CSIO
#define CSIO0_SCR       UART0_SCR
#define CSIO0_SCR_bit   UART0_SCR_bit.CSIO
#define CSIO0_ESCR      UART0_ESCR
#define CSIO0_ESCR_bit  UART0_ESCR_bit.CSIO
#define CSIO0_SSR       UART0_SSR
#define CSIO0_SSR_bit   UART0_SSR_bit.CSIO
#define CSIO0_RDR       UART0_RDR
#define CSIO0_RDR_bit   UART0_RDR_bit.CSIO
#define CSIO0_TDR       UART0_RDR
#define CSIO0_TDR_bit   UART0_RDR_bit.CSIO
#define CSIO0_BGR       UART0_BGR
#define CSIO0_BGR_bit   UART0_BGR_bit.CSIO

/***************************************************************************
 **
 ** CSIO1
 **
 ***************************************************************************/
#define CSIO1_SMR       UART1_SMR
#define CSIO1_SMR_bit   UART1_SMR_bit.CSIO
#define CSIO1_SCR       UART1_SCR
#define CSIO1_SCR_bit   UART1_SCR_bit.CSIO
#define CSIO1_ESCR      UART1_ESCR
#define CSIO1_ESCR_bit  UART1_ESCR_bit.CSIO
#define CSIO1_SSR       UART1_SSR
#define CSIO1_SSR_bit   UART1_SSR_bit.CSIO
#define CSIO1_RDR       UART1_RDR
#define CSIO1_RDR_bit   UART1_RDR_bit.CSIO
#define CSIO1_TDR       UART1_RDR
#define CSIO1_TDR_bit   UART1_RDR_bit.CSIO
#define CSIO1_BGR       UART1_BGR
#define CSIO1_BGR_bit   UART1_BGR_bit.CSIO

/***************************************************************************
 **
 ** CSIO2
 **
 ***************************************************************************/
#define CSIO2_SMR       UART2_SMR
#define CSIO2_SMR_bit   UART2_SMR_bit.CSIO
#define CSIO2_SCR       UART2_SCR
#define CSIO2_SCR_bit   UART2_SCR_bit.CSIO
#define CSIO2_ESCR      UART2_ESCR
#define CSIO2_ESCR_bit  UART2_ESCR_bit.CSIO
#define CSIO2_SSR       UART2_SSR
#define CSIO2_SSR_bit   UART2_SSR_bit.CSIO
#define CSIO2_RDR       UART2_RDR
#define CSIO2_RDR_bit   UART2_RDR_bit.CSIO
#define CSIO2_TDR       UART2_RDR
#define CSIO2_TDR_bit   UART2_RDR_bit.CSIO
#define CSIO2_BGR       UART2_BGR
#define CSIO2_BGR_bit   UART2_BGR_bit.CSIO

/***************************************************************************
 **
 ** CSIO3
 **
 ***************************************************************************/
#define CSIO3_SMR       UART3_SMR
#define CSIO3_SMR_bit   UART3_SMR_bit.CSIO
#define CSIO3_SCR       UART3_SCR
#define CSIO3_SCR_bit   UART3_SCR_bit.CSIO
#define CSIO3_ESCR      UART3_ESCR
#define CSIO3_ESCR_bit  UART3_ESCR_bit.CSIO
#define CSIO3_SSR       UART3_SSR
#define CSIO3_SSR_bit   UART3_SSR_bit.CSIO
#define CSIO3_RDR       UART3_RDR
#define CSIO3_RDR_bit   UART3_RDR_bit.CSIO
#define CSIO3_TDR       UART3_RDR
#define CSIO3_TDR_bit   UART3_RDR_bit.CSIO
#define CSIO3_BGR       UART3_BGR
#define CSIO3_BGR_bit   UART3_BGR_bit.CSIO

/***************************************************************************
 **
 ** CSIO4
 **
 ***************************************************************************/
#define CSIO4_SMR       UART4_SMR
#define CSIO4_SMR_bit   UART4_SMR_bit.CSIO
#define CSIO4_SCR       UART4_SCR
#define CSIO4_SCR_bit   UART4_SCR_bit.CSIO
#define CSIO4_ESCR      UART4_ESCR
#define CSIO4_ESCR_bit  UART4_ESCR_bit.CSIO
#define CSIO4_SSR       UART4_SSR
#define CSIO4_SSR_bit   UART4_SSR_bit.CSIO
#define CSIO4_RDR       UART4_RDR
#define CSIO4_RDR_bit   UART4_RDR_bit.CSIO
#define CSIO4_TDR       UART4_RDR
#define CSIO4_TDR_bit   UART4_RDR_bit.CSIO
#define CSIO4_BGR       UART4_BGR
#define CSIO4_BGR_bit   UART4_BGR_bit.CSIO
#define CSIO4_FCR       UART4_FCR
#define CSIO4_FCR_bit   UART4_FCR_bit.CSIO
#define CSIO4_FBYTE1    UART4_FBYTE1
#define CSIO4_FBYTE2    UART4_FBYTE2

/***************************************************************************
 **
 ** CSIO5
 **
 ***************************************************************************/
#define CSIO5_SMR       UART5_SMR
#define CSIO5_SMR_bit   UART5_SMR_bit.CSIO
#define CSIO5_SCR       UART5_SCR
#define CSIO5_SCR_bit   UART5_SCR_bit.CSIO
#define CSIO5_ESCR      UART5_ESCR
#define CSIO5_ESCR_bit  UART5_ESCR_bit.CSIO
#define CSIO5_SSR       UART5_SSR
#define CSIO5_SSR_bit   UART5_SSR_bit.CSIO
#define CSIO5_RDR       UART5_RDR
#define CSIO5_RDR_bit   UART5_RDR_bit.CSIO
#define CSIO5_TDR       UART5_RDR
#define CSIO5_TDR_bit   UART5_RDR_bit.CSIO
#define CSIO5_BGR       UART5_BGR
#define CSIO5_BGR_bit   UART5_BGR_bit.CSIO
#define CSIO5_FCR       UART5_FCR
#define CSIO5_FCR_bit   UART5_FCR_bit.CSIO
#define CSIO5_FBYTE1    UART5_FBYTE1
#define CSIO5_FBYTE2    UART5_FBYTE2

/***************************************************************************
 **
 ** CSIO6
 **
 ***************************************************************************/
#define CSIO6_SMR       UART6_SMR
#define CSIO6_SMR_bit   UART6_SMR_bit.CSIO
#define CSIO6_SCR       UART6_SCR
#define CSIO6_SCR_bit   UART6_SCR_bit.CSIO
#define CSIO6_ESCR      UART6_ESCR
#define CSIO6_ESCR_bit  UART6_ESCR_bit.CSIO
#define CSIO6_SSR       UART6_SSR
#define CSIO6_SSR_bit   UART6_SSR_bit.CSIO
#define CSIO6_RDR       UART6_RDR
#define CSIO6_RDR_bit   UART6_RDR_bit.CSIO
#define CSIO6_TDR       UART6_RDR
#define CSIO6_TDR_bit   UART6_RDR_bit.CSIO
#define CSIO6_BGR       UART6_BGR
#define CSIO6_BGR_bit   UART6_BGR_bit.CSIO
#define CSIO6_FCR       UART6_FCR
#define CSIO6_FCR_bit   UART6_FCR_bit.CSIO
#define CSIO6_FBYTE1    UART6_FBYTE1
#define CSIO6_FBYTE2    UART6_FBYTE2

/***************************************************************************
 **
 ** CSIO7
 **
 ***************************************************************************/
#define CSIO7_SMR       UART7_SMR
#define CSIO7_SMR_bit   UART7_SMR_bit.CSIO
#define CSIO7_SCR       UART7_SCR
#define CSIO7_SCR_bit   UART7_SCR_bit.CSIO
#define CSIO7_ESCR      UART7_ESCR
#define CSIO7_ESCR_bit  UART7_ESCR_bit.CSIO
#define CSIO7_SSR       UART7_SSR
#define CSIO7_SSR_bit   UART7_SSR_bit.CSIO
#define CSIO7_RDR       UART7_RDR
#define CSIO7_RDR_bit   UART7_RDR_bit.CSIO
#define CSIO7_TDR       UART7_RDR
#define CSIO7_TDR_bit   UART7_RDR_bit.CSIO
#define CSIO7_BGR       UART7_BGR
#define CSIO7_BGR_bit   UART7_BGR_bit.CSIO
#define CSIO7_FCR       UART7_FCR
#define CSIO7_FCR_bit   UART7_FCR_bit.CSIO
#define CSIO7_FBYTE1    UART7_FBYTE1
#define CSIO7_FBYTE2    UART7_FBYTE2

/***************************************************************************
 **
 ** LIN0
 **
 ***************************************************************************/
#define LIN0_SMR        UART0_SMR
#define LIN0_SMR_bit    UART0_SMR_bit.LIN
#define LIN0_SCR        UART0_SCR
#define LIN0_SCR_bit    UART0_SCR_bit.LIN
#define LIN0_ESCR       UART0_ESCR
#define LIN0_ESCR_bit   UART0_ESCR_bit.LIN
#define LIN0_SSR        UART0_SSR
#define LIN0_SSR_bit    UART0_SSR_bit.LIN
#define LIN0_RDR        UART0_RDR
#define LIN0_RDR_bit    UART0_RDR_bit.LIN
#define LIN0_TDR        UART0_RDR
#define LIN0_TDR_bit    UART0_RDR_bit.LIN
#define LIN0_BGR        UART0_BGR
#define LIN0_BGR_bit    UART0_BGR_bit.LIN

/***************************************************************************
 **
 ** LIN1
 **
 ***************************************************************************/
#define LIN1_SMR        UART1_SMR
#define LIN1_SMR_bit    UART1_SMR_bit.LIN
#define LIN1_SCR        UART1_SCR
#define LIN1_SCR_bit    UART1_SCR_bit.LIN
#define LIN1_ESCR       UART1_ESCR
#define LIN1_ESCR_bit   UART1_ESCR_bit.LIN
#define LIN1_SSR        UART1_SSR
#define LIN1_SSR_bit    UART1_SSR_bit.LIN
#define LIN1_RDR        UART1_RDR
#define LIN1_RDR_bit    UART1_RDR_bit.LIN
#define LIN1_TDR        UART1_RDR
#define LIN1_TDR_bit    UART1_RDR_bit.LIN
#define LIN1_BGR        UART1_BGR
#define LIN1_BGR_bit    UART1_BGR_bit.LIN

/***************************************************************************
 **
 ** LIN2
 **
 ***************************************************************************/
#define LIN2_SMR        UART2_SMR
#define LIN2_SMR_bit    UART2_SMR_bit.LIN
#define LIN2_SCR        UART2_SCR
#define LIN2_SCR_bit    UART2_SCR_bit.LIN
#define LIN2_ESCR       UART2_ESCR
#define LIN2_ESCR_bit   UART2_ESCR_bit.LIN
#define LIN2_SSR        UART2_SSR
#define LIN2_SSR_bit    UART2_SSR_bit.LIN
#define LIN2_RDR        UART2_RDR
#define LIN2_RDR_bit    UART2_RDR_bit.LIN
#define LIN2_TDR        UART2_RDR
#define LIN2_TDR_bit    UART2_RDR_bit.LIN
#define LIN2_BGR        UART2_BGR
#define LIN2_BGR_bit    UART2_BGR_bit.LIN

/***************************************************************************
 **
 ** LIN3
 **
 ***************************************************************************/
#define LIN3_SMR        UART3_SMR
#define LIN3_SMR_bit    UART3_SMR_bit.LIN
#define LIN3_SCR        UART3_SCR
#define LIN3_SCR_bit    UART3_SCR_bit.LIN
#define LIN3_ESCR       UART3_ESCR
#define LIN3_ESCR_bit   UART3_ESCR_bit.LIN
#define LIN3_SSR        UART3_SSR
#define LIN3_SSR_bit    UART3_SSR_bit.LIN
#define LIN3_RDR        UART3_RDR
#define LIN3_RDR_bit    UART3_RDR_bit.LIN
#define LIN3_TDR        UART3_RDR
#define LIN3_TDR_bit    UART3_RDR_bit.LIN
#define LIN3_BGR        UART3_BGR
#define LIN3_BGR_bit    UART3_BGR_bit.LIN

/***************************************************************************
 **
 ** LIN4
 **
 ***************************************************************************/
#define LIN4_SMR        UART4_SMR
#define LIN4_SMR_bit    UART4_SMR_bit.LIN
#define LIN4_SCR        UART4_SCR
#define LIN4_SCR_bit    UART4_SCR_bit.LIN
#define LIN4_ESCR       UART4_ESCR
#define LIN4_ESCR_bit   UART4_ESCR_bit.LIN
#define LIN4_SSR        UART4_SSR
#define LIN4_SSR_bit    UART4_SSR_bit.LIN
#define LIN4_RDR        UART4_RDR
#define LIN4_RDR_bit    UART4_RDR_bit.LIN
#define LIN4_TDR        UART4_RDR
#define LIN4_TDR_bit    UART4_RDR_bit.LIN
#define LIN4_BGR        UART4_BGR
#define LIN4_BGR_bit    UART4_BGR_bit.LIN
#define LIN4_FCR        UART4_FCR
#define LIN4_FCR_bit    UART4_FCR_bit.LIN
#define LIN4_FBYTE1     UART4_FBYTE1
#define LIN4_FBYTE2     UART4_FBYTE2

/***************************************************************************
 **
 ** LIN5
 **
 ***************************************************************************/
#define LIN5_SMR        UART5_SMR
#define LIN5_SMR_bit    UART5_SMR_bit.LIN
#define LIN5_SCR        UART5_SCR
#define LIN5_SCR_bit    UART5_SCR_bit.LIN
#define LIN5_ESCR       UART5_ESCR
#define LIN5_ESCR_bit   UART5_ESCR_bit.LIN
#define LIN5_SSR        UART5_SSR
#define LIN5_SSR_bit    UART5_SSR_bit.LIN
#define LIN5_RDR        UART5_RDR
#define LIN5_RDR_bit    UART5_RDR_bit.LIN
#define LIN5_TDR        UART5_RDR
#define LIN5_TDR_bit    UART5_RDR_bit.LIN
#define LIN5_BGR        UART5_BGR
#define LIN5_BGR_bit    UART5_BGR_bit.LIN
#define LIN5_FCR        UART5_FCR
#define LIN5_FCR_bit    UART5_FCR_bit.LIN
#define LIN5_FBYTE1     UART5_FBYTE1
#define LIN5_FBYTE2     UART5_FBYTE2

/***************************************************************************
 **
 ** LIN6
 **
 ***************************************************************************/
#define LIN6_SMR        UART6_SMR
#define LIN6_SMR_bit    UART6_SMR_bit.LIN
#define LIN6_SCR        UART6_SCR
#define LIN6_SCR_bit    UART6_SCR_bit.LIN
#define LIN6_ESCR       UART6_ESCR
#define LIN6_ESCR_bit   UART6_ESCR_bit.LIN
#define LIN6_SSR        UART6_SSR
#define LIN6_SSR_bit    UART6_SSR_bit.LIN
#define LIN6_RDR        UART6_RDR
#define LIN6_RDR_bit    UART6_RDR_bit.LIN
#define LIN6_TDR        UART6_RDR
#define LIN6_TDR_bit    UART6_RDR_bit.LIN
#define LIN6_BGR        UART6_BGR
#define LIN6_BGR_bit    UART6_BGR_bit.LIN
#define LIN6_FCR        UART6_FCR
#define LIN6_FCR_bit    UART6_FCR_bit.LIN
#define LIN6_FBYTE1     UART6_FBYTE1
#define LIN6_FBYTE2     UART6_FBYTE2

/***************************************************************************
 **
 ** LIN7
 **
 ***************************************************************************/
#define LIN7_SMR        UART7_SMR
#define LIN7_SMR_bit    UART7_SMR_bit.LIN
#define LIN7_SCR        UART7_SCR
#define LIN7_SCR_bit    UART7_SCR_bit.LIN
#define LIN7_ESCR       UART7_ESCR
#define LIN7_ESCR_bit   UART7_ESCR_bit.LIN
#define LIN7_SSR        UART7_SSR
#define LIN7_SSR_bit    UART7_SSR_bit.LIN
#define LIN7_RDR        UART7_RDR
#define LIN7_RDR_bit    UART7_RDR_bit.LIN
#define LIN7_TDR        UART7_RDR
#define LIN7_TDR_bit    UART7_RDR_bit.LIN
#define LIN7_BGR        UART7_BGR
#define LIN7_BGR_bit    UART7_BGR_bit.LIN
#define LIN7_FCR        UART7_FCR
#define LIN7_FCR_bit    UART7_FCR_bit.LIN
#define LIN7_FBYTE1     UART7_FBYTE1
#define LIN7_FBYTE2     UART7_FBYTE2

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
#define I2C0_SMR        UART0_SMR
#define I2C0_SMR_bit    UART0_SMR_bit.I2C
#define I2C0_IBCR       UART0_SCR
#define I2C0_IBCR_bit   UART0_SCR_bit.I2C
#define I2C0_IBSR       UART0_ESCR
#define I2C0_IBSR_bit   UART0_ESCR_bit.I2C
#define I2C0_SSR        UART0_SSR
#define I2C0_SSR_bit    UART0_SSR_bit.I2C
#define I2C0_RDR        UART0_RDR
#define I2C0_RDR_bit    UART0_RDR_bit.I2C
#define I2C0_TDR        UART0_RDR
#define I2C0_TDR_bit    UART0_RDR_bit.I2C
#define I2C0_BGR        UART0_BGR
#define I2C0_BGR_bit    UART0_BGR_bit.I2C
__IO_REG8_BIT( I2C0_ISBA,         0x40038010,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C0_ISMK,         0x40038011,__READ_WRITE ,__mfsx_ismk_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
#define I2C1_SMR        UART1_SMR
#define I2C1_SMR_bit    UART1_SMR_bit.I2C
#define I2C1_IBCR       UART1_SCR
#define I2C1_IBCR_bit   UART1_SCR_bit.I2C
#define I2C1_IBSR       UART1_ESCR
#define I2C1_IBSR_bit   UART1_ESCR_bit.I2C
#define I2C1_SSR        UART1_SSR
#define I2C1_SSR_bit    UART1_SSR_bit.I2C
#define I2C1_RDR        UART1_RDR
#define I2C1_RDR_bit    UART1_RDR_bit.I2C
#define I2C1_TDR        UART1_RDR
#define I2C1_TDR_bit    UART1_RDR_bit.I2C
#define I2C1_BGR        UART1_BGR
#define I2C1_BGR_bit    UART1_BGR_bit.I2C
__IO_REG8_BIT( I2C1_ISBA,        0x40038110,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C1_ISMK,        0x40038111,__READ_WRITE ,__mfsx_ismk_bits);

/***************************************************************************
 **
 ** I2C2
 **
 ***************************************************************************/
#define I2C2_SMR        UART2_SMR
#define I2C2_SMR_bit    UART2_SMR_bit.I2C
#define I2C2_IBCR       UART2_SCR
#define I2C2_IBCR_bit   UART2_SCR_bit.I2C
#define I2C2_IBSR       UART2_ESCR
#define I2C2_IBSR_bit   UART2_ESCR_bit.I2C
#define I2C2_SSR        UART2_SSR
#define I2C2_SSR_bit    UART2_SSR_bit.I2C
#define I2C2_RDR        UART2_RDR
#define I2C2_RDR_bit    UART2_RDR_bit.I2C
#define I2C2_TDR        UART2_RDR
#define I2C2_TDR_bit    UART2_RDR_bit.I2C
#define I2C2_BGR        UART2_BGR
#define I2C2_BGR_bit    UART2_BGR_bit.I2C
__IO_REG8_BIT( I2C2_ISBA,        0x40038210,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C2_ISMK,        0x40038211,__READ_WRITE ,__mfsx_ismk_bits);

/***************************************************************************
 **
 ** I2C3
 **
 ***************************************************************************/
#define I2C3_SMR        UART3_SMR
#define I2C3_SMR_bit    UART3_SMR_bit.I2C
#define I2C3_IBCR       UART3_SCR
#define I2C3_IBCR_bit   UART3_SCR_bit.I2C
#define I2C3_IBSR       UART3_ESCR
#define I2C3_IBSR_bit   UART3_ESCR_bit.I2C
#define I2C3_SSR        UART3_SSR
#define I2C3_SSR_bit    UART3_SSR_bit.I2C
#define I2C3_RDR        UART3_RDR
#define I2C3_RDR_bit    UART3_RDR_bit.I2C
#define I2C3_TDR        UART3_RDR
#define I2C3_TDR_bit    UART3_RDR_bit.I2C
#define I2C3_BGR        UART3_BGR
#define I2C3_BGR_bit    UART3_BGR_bit.I2C
__IO_REG8_BIT( I2C3_ISBA,        0x40038310,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C3_ISMK,        0x40038311,__READ_WRITE ,__mfsx_ismk_bits);

/***************************************************************************
 **
 ** I2C4
 **
 ***************************************************************************/
#define I2C4_SMR        UART4_SMR
#define I2C4_SMR_bit    UART4_SMR_bit.I2C
#define I2C4_IBCR       UART4_SCR
#define I2C4_IBCR_bit   UART4_SCR_bit.I2C
#define I2C4_IBSR       UART4_ESCR
#define I2C4_IBSR_bit   UART4_ESCR_bit.I2C
#define I2C4_SSR        UART4_SSR
#define I2C4_SSR_bit    UART4_SSR_bit.I2C
#define I2C4_RDR        UART4_RDR
#define I2C4_RDR_bit    UART4_RDR_bit.I2C
#define I2C4_TDR        UART4_RDR
#define I2C4_TDR_bit    UART4_RDR_bit.I2C
#define I2C4_BGR        UART4_BGR
#define I2C4_BGR_bit    UART4_BGR_bit.I2C
__IO_REG8_BIT( I2C4_ISBA,        0x40038410,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C4_ISMK,        0x40038411,__READ_WRITE ,__mfsx_ismk_bits);
#define I2C4_FCR        UART4_FCR
#define I2C4_FCR_bit    UART4_FCR_bit.I2C
#define I2C4_FBYTE1     UART4_FBYTE1
#define I2C4_FBYTE2     UART4_FBYTE2

/***************************************************************************
 **
 ** I2C5
 **
 ***************************************************************************/
#define I2C5_SMR        UART5_SMR
#define I2C5_SMR_bit    UART5_SMR_bit.I2C
#define I2C5_IBCR       UART5_SCR
#define I2C5_IBCR_bit   UART5_SCR_bit.I2C
#define I2C5_IBSR       UART5_ESCR
#define I2C5_IBSR_bit   UART5_ESCR_bit.I2C
#define I2C5_SSR        UART5_SSR
#define I2C5_SSR_bit    UART5_SSR_bit.I2C
#define I2C5_RDR        UART5_RDR
#define I2C5_RDR_bit    UART5_RDR_bit.I2C
#define I2C5_TDR        UART5_RDR
#define I2C5_TDR_bit    UART5_RDR_bit.I2C
#define I2C5_BGR        UART5_BGR
#define I2C5_BGR_bit    UART5_BGR_bit.I2C
__IO_REG8_BIT( I2C5_ISBA,         0x40038510,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C5_ISMK,         0x40038511,__READ_WRITE ,__mfsx_ismk_bits);
#define I2C5_FCR        UART5_FCR
#define I2C5_FCR_bit    UART5_FCR_bit.I2C
#define I2C5_FBYTE1     UART5_FBYTE1
#define I2C5_FBYTE2     UART5_FBYTE2

/***************************************************************************
 **
 ** I2C6
 **
 ***************************************************************************/
#define I2C6_SMR        UART6_SMR
#define I2C6_SMR_bit    UART6_SMR_bit.I2C
#define I2C6_IBCR       UART6_SCR
#define I2C6_IBCR_bit   UART6_SCR_bit.I2C
#define I2C6_IBSR       UART6_ESCR
#define I2C6_IBSR_bit   UART6_ESCR_bit.I2C
#define I2C6_SSR        UART6_SSR
#define I2C6_SSR_bit    UART6_SSR_bit.I2C
#define I2C6_RDR        UART6_RDR
#define I2C6_RDR_bit    UART6_RDR_bit.I2C
#define I2C6_TDR        UART6_RDR
#define I2C6_TDR_bit    UART6_RDR_bit.I2C
#define I2C6_BGR        UART6_BGR
#define I2C6_BGR_bit    UART6_BGR_bit.I2C
__IO_REG8_BIT( I2C6_ISBA,         0x40038610,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C6_ISMK,         0x40038611,__READ_WRITE ,__mfsx_ismk_bits);
#define I2C6_FCR        UART6_FCR
#define I2C6_FCR_bit    UART6_FCR_bit.I2C
#define I2C6_FBYTE1     UART6_FBYTE1
#define I2C6_FBYTE2     UART6_FBYTE2

/***************************************************************************
 **
 ** I2C7
 **
 ***************************************************************************/
#define I2C7_SMR        UART7_SMR
#define I2C7_SMR_bit    UART7_SMR_bit.I2C
#define I2C7_IBCR       UART7_SCR
#define I2C7_IBCR_bit   UART7_SCR_bit.I2C
#define I2C7_IBSR       UART7_ESCR
#define I2C7_IBSR_bit   UART7_ESCR_bit.I2C
#define I2C7_SSR        UART7_SSR
#define I2C7_SSR_bit    UART7_SSR_bit.I2C
#define I2C7_RDR        UART7_RDR
#define I2C7_RDR_bit    UART7_RDR_bit.I2C
#define I2C7_TDR        UART7_RDR
#define I2C7_TDR_bit    UART7_RDR_bit.I2C
#define I2C7_BGR        UART7_BGR
#define I2C7_BGR_bit    UART7_BGR_bit.I2C
__IO_REG8_BIT( I2C7_ISBA,        0x40038710,__READ_WRITE ,__mfsx_isba_bits);
__IO_REG8_BIT( I2C7_ISMK,        0x40038711,__READ_WRITE ,__mfsx_ismk_bits);
#define I2C7_FCR        UART7_FCR
#define I2C7_FCR_bit    UART7_FCR_bit.I2C
#define I2C7_FBYTE1     UART7_FBYTE1
#define I2C7_FBYTE2     UART7_FBYTE2

/***************************************************************************
 **
 ** MFS Noise Filter Cntrol
 **
 ***************************************************************************/
__IO_REG16_BIT(I2CDNF,           0x40038800,__READ_WRITE ,__i2cdnf_bits);

/***************************************************************************
 **
 ** CRC
 **
 ***************************************************************************/
__IO_REG8_BIT( CRCCR,             0x40039000,__READ_WRITE ,__crccr_bits);
__IO_REG32(    CRCINIT,           0x40039004,__READ_WRITE );
__IO_REG32_BIT(CRCIN,             0x40039008,__READ_WRITE ,__crcin_bits);
#define CRCINL      CRCIN_bit.__shortl
#define CRCINL_bit  CRCIN_bit.__shortl_bit
#define CRCINH      CRCIN_bit.__shorth
#define CRCINH_bit  CRCIN_bit.__shorth_bit
#define CRCINLL     CRCIN_bit.__byte0
#define CRCINLH     CRCIN_bit.__byte1
#define CRCINHL     CRCIN_bit.__byte2
#define CRCINHH     CRCIN_bit.__byte3
__IO_REG32(    CRCR,              0x4003900C,__READ       );

/***************************************************************************
 **
 ** Watch Counter
 **
 ***************************************************************************/
__IO_REG8_BIT( WCRD,              0x4003A000,__READ       ,__wcrd_bits);
__IO_REG8_BIT( WCRL,              0x4003A001,__READ_WRITE ,__wcrl_bits);
__IO_REG8_BIT( WCCR,              0x4003A002,__READ_WRITE ,__wccr_bits);
__IO_REG16_BIT(CLK_SEL,           0x4003A010,__READ_WRITE ,__clk_sel_bits);
__IO_REG8_BIT( CLK_EN,            0x4003A014,__READ_WRITE ,__clk_en_bits);

/***************************************************************************
 **
 ** EXT I/F
 **
 ***************************************************************************/
__IO_REG32_BIT(MODE0,             0x4003F000,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE1,             0x4003F004,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE2,             0x4003F008,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE3,             0x4003F00C,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE4,             0x4003F010,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE5,             0x4003F014,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE6,             0x4003F018,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(MODE7,             0x4003F01C,__READ_WRITE ,__modex_bits);
__IO_REG32_BIT(TIM0,              0x4003F020,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM1,              0x4003F024,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM2,              0x4003F028,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM3,              0x4003F02C,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM4,              0x4003F030,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM5,              0x4003F034,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM6,              0x4003F038,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(TIM7,              0x4003F03C,__READ_WRITE ,__timx_bits);
__IO_REG32_BIT(AREA0,             0x4003F040,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA1,             0x4003F044,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA2,             0x4003F048,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA3,             0x4003F04C,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA4,             0x4003F050,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA5,             0x4003F054,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA6,             0x4003F058,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(AREA7,             0x4003F05C,__READ_WRITE ,__areax_bits);
__IO_REG32_BIT(ATIM0,             0x4003F060,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM1,             0x4003F064,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM2,             0x4003F068,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM3,             0x4003F06C,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM4,             0x4003F070,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM5,             0x4003F074,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM6,             0x4003F078,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(ATIM7,             0x4003F07C,__READ_WRITE ,__atimx_bits);
__IO_REG32_BIT(DCLKR,             0x4003F300,__READ_WRITE ,__dclkr_bits);

/***************************************************************************
 **
 ** USB0
 **
 ***************************************************************************/
__IO_REG16_BIT(USB0_HCNT,         0x40042100,__READ_WRITE ,__usb_hcnt_bits);
__IO_REG8_BIT( USB0_HIRQ,         0x40042104,__READ_WRITE ,__usb_hirq_bits);
__IO_REG8_BIT( USB0_HERR,         0x40042105,__READ_WRITE ,__usb_herr_bits);
__IO_REG8_BIT( USB0_HSTATE,       0x40042108,__READ_WRITE ,__usb_hstate_bits);
__IO_REG8(     USB0_HFCOMP,       0x40042109,__READ_WRITE );
__IO_REG8(     USB0_HRTIMER0,     0x4004210C,__READ_WRITE );
__IO_REG8(     USB0_HRTIMER1,     0x4004210D,__READ_WRITE );
__IO_REG8_BIT( USB0_HRTIMER2,     0x40042110,__READ_WRITE ,__usb_hrtimer2_bits);
__IO_REG8_BIT( USB0_HADR,         0x40042111,__READ_WRITE ,__usb_hadr_bits);
__IO_REG16_BIT(USB0_HEOF,         0x40042114,__READ_WRITE ,__usb_heof_bits);
__IO_REG16_BIT(USB0_HFRAME,       0x40042118,__READ_WRITE ,__usb_hframe_bits);
__IO_REG8_BIT( USB0_HTOKEN,       0x4004211C,__READ_WRITE ,__usb_htoken_bits);
__IO_REG16_BIT(USB0_UDCC,         0x40042120,__READ_WRITE ,__usb_udcc_bits);
__IO_REG16_BIT(USB0_EP0C,         0x40042124,__READ_WRITE ,__usb_ep0c_bits);
__IO_REG16_BIT(USB0_EP1C,         0x40042128,__READ_WRITE ,__usb_ep1c_bits);
__IO_REG16_BIT(USB0_EP2C,         0x4004212C,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB0_EP3C,         0x40042130,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB0_EP4C,         0x40042134,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB0_EP5C,         0x40042138,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB0_TMSP,         0x4004213C,__READ       ,__usb_tmsp_bits);
__IO_REG8_BIT( USB0_UDCS,         0x40042140,__READ_WRITE ,__usb_udcs_bits);
__IO_REG8_BIT( USB0_UDCIE,        0x40042141,__READ_WRITE ,__usb_udcie_bits);
__IO_REG16_BIT(USB0_EP0IS,        0x40042144,__READ_WRITE ,__usb_ep0is_bits);
__IO_REG16_BIT(USB0_EP0OS,        0x40042148,__READ_WRITE ,__usb_ep0os_bits);
__IO_REG16_BIT(USB0_EP1S,         0x4004214C,__READ_WRITE ,__usb_ep1s_bits);
__IO_REG16_BIT(USB0_EP2S,         0x40042150,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB0_EP3S,         0x40042154,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB0_EP4S,         0x40042158,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB0_EP5S,         0x4004215C,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB0_EP0DT,        0x40042160,__READ_WRITE ,__usb_epxdt_bits);
#define USB0_EP0DTL USB0_EP0DT_bit.__byte0
#define USB0_EP0DTH USB0_EP0DT_bit.__byte1
__IO_REG16_BIT(USB0_EP1DT,        0x40042164,__READ_WRITE ,__usb_epxdt_bits);
#define USB0_EP1DTL USB0_EP1DT_bit.__byte0
#define USB0_EP1DTH USB0_EP1DT_bit.__byte1
__IO_REG16_BIT(USB0_EP2DT,        0x40042168,__READ_WRITE ,__usb_epxdt_bits);
#define USB0_EP2DTL USB0_EP2DT_bit.__byte0
#define USB0_EP2DTH USB0_EP2DT_bit.__byte1
__IO_REG16_BIT(USB0_EP3DT,        0x4004216C,__READ_WRITE ,__usb_epxdt_bits);
#define USB0_EP3DTL USB0_EP3DT_bit.__byte0
#define USB0_EP3DTH USB0_EP3DT_bit.__byte1
__IO_REG16_BIT(USB0_EP4DT,        0x40042170,__READ_WRITE ,__usb_epxdt_bits);
#define USB0_EP4DTL USB0_EP4DT_bit.__byte0
#define USB0_EP4DTH USB0_EP4DT_bit.__byte1
__IO_REG16_BIT(USB0_EP5DT,        0x40042174,__READ_WRITE ,__usb_epxdt_bits);
#define USB0_EP5DTL USB0_EP5DT_bit.__byte0
#define USB0_EP5DTH USB0_EP5DT_bit.__byte1

/***************************************************************************
 **
 ** USB1
 **
 ***************************************************************************/
__IO_REG16_BIT(USB1_HCNT,         0x40052100,__READ_WRITE ,__usb_hcnt_bits);
__IO_REG8_BIT( USB1_HIRQ,         0x40052104,__READ_WRITE ,__usb_hirq_bits);
__IO_REG8_BIT( USB1_HERR,         0x40052105,__READ_WRITE ,__usb_herr_bits);
__IO_REG8_BIT( USB1_HSTATE,       0x40052108,__READ_WRITE ,__usb_hstate_bits);
__IO_REG8(     USB1_HFCOMP,       0x40052109,__READ_WRITE );
__IO_REG8(     USB1_HRTIMER0,     0x4005210C,__READ_WRITE );
__IO_REG8(     USB1_HRTIMER1,     0x4005210D,__READ_WRITE );
__IO_REG8_BIT( USB1_HRTIMER2,     0x40052110,__READ_WRITE ,__usb_hrtimer2_bits);
__IO_REG8_BIT( USB1_HADR,         0x40052111,__READ_WRITE ,__usb_hadr_bits);
__IO_REG16_BIT(USB1_HEOF,         0x40052114,__READ_WRITE ,__usb_heof_bits);
__IO_REG16_BIT(USB1_HFRAME,       0x40052118,__READ_WRITE ,__usb_hframe_bits);
__IO_REG8_BIT( USB1_HTOKEN,       0x4005211C,__READ_WRITE ,__usb_htoken_bits);
__IO_REG16_BIT(USB1_UDCC,         0x40052120,__READ_WRITE ,__usb_udcc_bits);
__IO_REG16_BIT(USB1_EP0C,         0x40052124,__READ_WRITE ,__usb_ep0c_bits);
__IO_REG16_BIT(USB1_EP1C,         0x40052128,__READ_WRITE ,__usb_ep1c_bits);
__IO_REG16_BIT(USB1_EP2C,         0x4005212C,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB1_EP3C,         0x40052130,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB1_EP4C,         0x40052134,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB1_EP5C,         0x40052138,__READ_WRITE ,__usb_epxc_bits);
__IO_REG16_BIT(USB1_TMSP,         0x4005213C,__READ       ,__usb_tmsp_bits);
__IO_REG8_BIT( USB1_UDCS,         0x40052140,__READ_WRITE ,__usb_udcs_bits);
__IO_REG8_BIT( USB1_UDCIE,        0x40052141,__READ_WRITE ,__usb_udcie_bits);
__IO_REG16_BIT(USB1_EP0IS,        0x40052144,__READ_WRITE ,__usb_ep0is_bits);
__IO_REG16_BIT(USB1_EP0OS,        0x40052148,__READ_WRITE ,__usb_ep0os_bits);
__IO_REG16_BIT(USB1_EP1S,         0x4005214C,__READ_WRITE ,__usb_ep1s_bits);
__IO_REG16_BIT(USB1_EP2S,         0x40052150,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB1_EP3S,         0x40052154,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB1_EP4S,         0x40052158,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB1_EP5S,         0x4005215C,__READ_WRITE ,__usb_epxs_bits);
__IO_REG16_BIT(USB1_EP0DT,        0x40052160,__READ_WRITE ,__usb_epxdt_bits);
#define USB1_EP0DTL USB1_EP0DT_bit.__byte0
#define USB1_EP0DTH USB1_EP0DT_bit.__byte1
__IO_REG16_BIT(USB1_EP1DT,        0x40052164,__READ_WRITE ,__usb_epxdt_bits);
#define USB1_EP1DTL USB1_EP1DT_bit.__byte0
#define USB1_EP1DTH USB1_EP1DT_bit.__byte1
__IO_REG16_BIT(USB1_EP2DT,        0x40052168,__READ_WRITE ,__usb_epxdt_bits);
#define USB1_EP2DTL USB1_EP2DT_bit.__byte0
#define USB1_EP2DTH USB1_EP2DT_bit.__byte1
__IO_REG16_BIT(USB1_EP3DT,        0x4005216C,__READ_WRITE ,__usb_epxdt_bits);
#define USB1_EP3DTL USB1_EP3DT_bit.__byte0
#define USB1_EP3DTH USB1_EP3DT_bit.__byte1
__IO_REG16_BIT(USB1_EP4DT,        0x40052170,__READ_WRITE ,__usb_epxdt_bits);
#define USB1_EP4DTL USB1_EP4DT_bit.__byte0
#define USB1_EP4DTH USB1_EP4DT_bit.__byte1
__IO_REG16_BIT(USB1_EP5DT,        0x40052174,__READ_WRITE ,__usb_epxdt_bits);
#define USB1_EP5DTL USB1_EP5DT_bit.__byte0
#define USB1_EP5DTH USB1_EP5DT_bit.__byte1

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACR,             0x40060000,__READ_WRITE ,__dmacr_bits);
__IO_REG32_BIT(DMACA0,            0x40060010,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB0,            0x40060014,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA0,           0x40060018,__READ_WRITE );
__IO_REG32(    DMACDA0,           0x4006001C,__READ_WRITE );
__IO_REG32_BIT(DMACA1,            0x40060020,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB1,            0x40060024,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA1,           0x40060028,__READ_WRITE );
__IO_REG32(    DMACDA1,           0x4006002C,__READ_WRITE );
__IO_REG32_BIT(DMACA2,            0x40060030,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB2,            0x40060034,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA2,           0x40060038,__READ_WRITE );
__IO_REG32(    DMACDA2,           0x4006003C,__READ_WRITE );
__IO_REG32_BIT(DMACA3,            0x40060040,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB3,            0x40060044,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA3,           0x40060048,__READ_WRITE );
__IO_REG32(    DMACDA3,           0x4006004C,__READ_WRITE );
__IO_REG32_BIT(DMACA4,            0x40060050,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB4,            0x40060054,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA4,           0x40060058,__READ_WRITE );
__IO_REG32(    DMACDA4,           0x4006005C,__READ_WRITE );
__IO_REG32_BIT(DMACA5,            0x40060060,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB5,            0x40060064,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA5,           0x40060068,__READ_WRITE );
__IO_REG32(    DMACDA5,           0x4006006C,__READ_WRITE );
__IO_REG32_BIT(DMACA6,            0x40060070,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB6,            0x40060074,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA6,           0x40060078,__READ_WRITE );
__IO_REG32(    DMACDA6,           0x4006007C,__READ_WRITE );
__IO_REG32_BIT(DMACA7,            0x40060080,__READ_WRITE ,__dmacax_bits);
__IO_REG32_BIT(DMACB7,            0x40060084,__READ_WRITE ,__dmacbx_bits);
__IO_REG32(    DMACSA7,           0x40060088,__READ_WRITE );
__IO_REG32(    DMACDA7,           0x4006008C,__READ_WRITE );

/***************************************************************************
 **
 ** Ethernet System Control
 **
 ***************************************************************************/
__IO_REG32_BIT(ETH_MODE,          0x40066000,__READ_WRITE ,__eth_mode_bits);
__IO_REG32_BIT(ETH_CLKG,          0x40066008,__READ_WRITE ,__eth_clkg_bits);

/***************************************************************************
 **
 ** Ethernet MAC0
 **
 ***************************************************************************/
__IO_REG32_BIT(EMAC0_MCR,                   0x40064000,__READ_WRITE ,__emac_mcr_bits);
__IO_REG32_BIT(EMAC0_MFFR,                  0x40064004,__READ_WRITE ,__emac_mffr_bits);
__IO_REG32(    EMAC0_MHTRH,                 0x40064008,__READ_WRITE );
__IO_REG32(    EMAC0_MHTRL,                 0x4006400C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_GAR,                   0x40064010,__READ_WRITE ,__emac_gar_bits);
__IO_REG32_BIT(EMAC0_GDR,                   0x40064014,__READ_WRITE ,__emac_gdr_bits);
__IO_REG32_BIT(EMAC0_FCR,                   0x40064018,__READ_WRITE ,__emac_fcr_bits);
__IO_REG32_BIT(EMAC0_VTR,                   0x4006401C,__READ_WRITE ,__emac_vtr_bits);
__IO_REG32(    EMAC0_RWFFR,                 0x40064028,__READ_WRITE );
__IO_REG32_BIT(EMAC0_PMTR,                  0x4006402C,__READ_WRITE ,__emac_pmtr_bits);
__IO_REG32_BIT(EMAC0_LPICSR,                0x40064030,__READ_WRITE ,__emac_lpicsr_bits);
__IO_REG32_BIT(EMAC0_LPITCR,                0x40064034,__READ_WRITE ,__emac_lpitcr_bits);
__IO_REG32_BIT(EMAC0_ISR,                   0x40064038,__READ       ,__emac_isr_bits);
__IO_REG32_BIT(EMAC0_IMR,                   0x4006403C,__READ_WRITE ,__emac_imr_bits);
__IO_REG32_BIT(EMAC0_MAR0H,                 0x40064040,__READ_WRITE ,__emac_mar0h_bits);
__IO_REG32(    EMAC0_MAR0L,                 0x40064044,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR1H,                 0x40064048,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR1L,                 0x4006404C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR2H,                 0x40064050,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR2L,                 0x40064054,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR3H,                 0x40064058,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR3L,                 0x4006405C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR4H,                 0x40064060,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR4L,                 0x40064064,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR5H,                 0x40064068,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR5L,                 0x4006406C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR6H,                 0x40064070,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR6L,                 0x40064074,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR7H,                 0x40064078,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR7L,                 0x4006407C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR8H,                 0x40064080,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR8L,                 0x40064084,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR9H,                 0x40064088,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR9L,                 0x4006408C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR10H,                0x40064090,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR10L,                0x40064094,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR11H,                0x40064098,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR11L,                0x4006409C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR12H,                0x400640A0,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR12L,                0x400640A4,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR13H,                0x400640A8,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR13L,                0x400640AC,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR14H,                0x400640B0,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR14L,                0x400640B4,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR15H,                0x400640B8,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR15L,                0x400640BC,__READ_WRITE );
__IO_REG32_BIT(EMAC0_RGSR,                  0x400640D8,__READ       ,__emac_rgsr_bits);
__IO_REG32(    EMAC0_mmc_cntl,              0x40064100,__READ_WRITE );
__IO_REG32(    EMAC0_mmc_intr_rx,           0x40064104,__READ_WRITE );
__IO_REG32(    EMAC0_mmc_intr_tx,           0x40064108,__READ_WRITE );
__IO_REG32(    EMAC0_mmc_intr_mask_rx,      0x4006410C,__READ_WRITE );
__IO_REG32(    EMAC0_mmc_intr_mask_tx,      0x40064110,__READ_WRITE );
__IO_REG32(    EMAC0_txoctetcount_gb,       0x40064114,__READ       );
__IO_REG32(    EMAC0_txframecount_gb,       0x40064118,__READ       );
__IO_REG32(    EMAC0_txbroadcastframes_g,   0x4006411C,__READ       );
__IO_REG32(    EMAC0_txmulticastframes_g,   0x40064120,__READ       );
__IO_REG32(    EMAC0_tx64octets_gb,         0x40064124,__READ       );
__IO_REG32(    EMAC0_tx65to127octets_gb,    0x40064128,__READ       );
__IO_REG32(    EMAC0_tx128to255octets_gb,   0x4006412C,__READ       );
__IO_REG32(    EMAC0_tx256to511octets_gb,   0x40064130,__READ       );
__IO_REG32(    EMAC0_tx512to1023octets_gb,  0x40064134,__READ       );
__IO_REG32(    EMAC0_tx1024tomaxoctets_gb,  0x40064138,__READ       );
__IO_REG32(    EMAC0_txunicastframes_gb,    0x4006413C,__READ       );
__IO_REG32(    EMAC0_txmulticastframes_gb,  0x40064140,__READ       );
__IO_REG32(    EMAC0_txbroadcastframes_gb,  0x40064144,__READ       );
__IO_REG32(    EMAC0_txunderflowerror,      0x40064148,__READ       );
__IO_REG32(    EMAC0_txsinglecol_g,         0x4006414C,__READ       );
__IO_REG32(    EMAC0_txmulticol_g,          0x40064150,__READ       );
__IO_REG32(    EMAC0_txdeferred,            0x40064154,__READ       );
__IO_REG32(    EMAC0_txlatecol,             0x40064158,__READ       );
__IO_REG32(    EMAC0_txexesscol,            0x4006415C,__READ       );
__IO_REG32(    EMAC0_txcarriererrror,       0x40064160,__READ       );
__IO_REG32(    EMAC0_txoctetcount_g,        0x40064164,__READ       );
__IO_REG32(    EMAC0_txframecount_g,        0x40064168,__READ       );
__IO_REG32(    EMAC0_txexecessdef,          0x4006416C,__READ       );
__IO_REG32(    EMAC0_txpauseframes,         0x40064170,__READ       );
__IO_REG32(    EMAC0_txvlanframes_g,        0x40064174,__READ       );
__IO_REG32(    EMAC0_rxframecount_gb,       0x40064180,__READ       );
__IO_REG32(    EMAC0_rxoctetcount_gb,       0x40064184,__READ       );
__IO_REG32(    EMAC0_rxoctetcount_g,        0x40064188,__READ       );
__IO_REG32(    EMAC0_rxbroadcastframes_g,   0x4006418C,__READ       );
__IO_REG32(    EMAC0_rxmulticastframes_g,   0x40064190,__READ       );
__IO_REG32(    EMAC0_rxcrcerror,            0x40064194,__READ       );
__IO_REG32(    EMAC0_rxallignmenterror,     0x40064198,__READ       );
__IO_REG32(    EMAC0_rxrunterror,           0x4006419C,__READ       );
__IO_REG32(    EMAC0_rxjabbererror,         0x400641A0,__READ       );
__IO_REG32(    EMAC0_rxundersize_g,         0x400641A4,__READ       );
__IO_REG32(    EMAC0_rxoversize_g,          0x400641A8,__READ       );
__IO_REG32(    EMAC0_rx64octets_gb,         0x400641AC,__READ       );
__IO_REG32(    EMAC0_rx65to127octets_gb,    0x400641B0,__READ       );
__IO_REG32(    EMAC0_rx128to255octets_gb,   0x400641B4,__READ       );
__IO_REG32(    EMAC0_rx256to511octets_gb,   0x400641B8,__READ       );
__IO_REG32(    EMAC0_rx512to1023octets_gb,  0x400641BC,__READ       );
__IO_REG32(    EMAC0_rx1024tomaxoctets_gb,  0x400641C0,__READ       );
__IO_REG32(    EMAC0_rxunicastframes_g,     0x400641C4,__READ       );
__IO_REG32(    EMAC0_rxlengtherror,         0x400641C8,__READ       );
__IO_REG32(    EMAC0_rxoutofrangetype,      0x400641CC,__READ       );
__IO_REG32(    EMAC0_rxpauseframes,         0x400641D0,__READ       );
__IO_REG32(    EMAC0_rxfifooverflow,        0x400641D4,__READ       );
__IO_REG32(    EMAC0_rxvlanframes_gb,       0x400641D8,__READ       );
__IO_REG32(    EMAC0_rxwatchdogerror,       0x400641DC,__READ       );
__IO_REG32(    EMAC0_mmc_ipc_intr_mask_rx,  0x40064200,__READ_WRITE );
__IO_REG32(    EMAC0_mmc_ipc_intr_rx,       0x40064208,__READ_WRITE );
__IO_REG32(    EMAC0_rxipv4_gd_frms,        0x40064210,__READ       );
__IO_REG32(    EMAC0_rxipv4_hdrerr_frms,    0x40064214,__READ       );
__IO_REG32(    EMAC0_rxipv4_nopay_frms,     0x40064218,__READ       );
__IO_REG32(    EMAC0_rxipv4_frag_frms,      0x4006421C,__READ       );
__IO_REG32(    EMAC0_rxipv4_udsbl_frms,     0x40064220,__READ       );
__IO_REG32(    EMAC0_rxipv6_gd_frms,        0x40064224,__READ       );
__IO_REG32(    EMAC0_rxipv6_hdrerr_frms,    0x40064228,__READ       );
__IO_REG32(    EMAC0_rxipv6_nopay_frms,     0x4006422C,__READ       );
__IO_REG32(    EMAC0_rxudp_gd_frms,         0x40064230,__READ       );
__IO_REG32(    EMAC0_rxudp_err_frms,        0x40064234,__READ       );
__IO_REG32(    EMAC0_rxtcp_gd_frms,         0x40064238,__READ       );
__IO_REG32(    EMAC0_rxtcp_err_frms,        0x4006423C,__READ       );
__IO_REG32(    EMAC0_rxicmp_gd_frms,        0x40064240,__READ       );
__IO_REG32(    EMAC0_rxicmp_err_frms,       0x40064244,__READ       );
__IO_REG32(    EMAC0_rxipv4_gd_octets,      0x40064250,__READ       );
__IO_REG32(    EMAC0_rxipv4_hdrerr_octets,  0x40064254,__READ       );
__IO_REG32(    EMAC0_rxipv4_nopay_octets,   0x40064258,__READ       );
__IO_REG32(    EMAC0_rxipv4_frag_octets,    0x4006425C,__READ       );
__IO_REG32(    EMAC0_rxipv4_udsbl_octets,   0x40064260,__READ       );
__IO_REG32(    EMAC0_rxipv6_gd_octets,      0x40064264,__READ       );
__IO_REG32(    EMAC0_rxipv6_hdrerr_octets,  0x40064268,__READ       );
__IO_REG32(    EMAC0_rxipv6_nopay_octets,   0x4006426C,__READ       );
__IO_REG32(    EMAC0_rxudp_gd_octets,       0x40064270,__READ       );
__IO_REG32(    EMAC0_rxudp_err_octets,      0x40064274,__READ       );
__IO_REG32(    EMAC0_rxtcp_gd_octets,       0x40064278,__READ       );
__IO_REG32(    EMAC0_rxtcp_err_octets,      0x4006427C,__READ       );
__IO_REG32(    EMAC0_rxicmp_gd_octets,      0x40064280,__READ       );
__IO_REG32(    EMAC0_rxicmp_err_octets,     0x40064284,__READ       );
__IO_REG32_BIT(EMAC0_TSCR,                  0x40064700,__READ_WRITE ,__emac_tscr_bits);
__IO_REG32_BIT(EMAC0_SSIR,                  0x40064704,__READ_WRITE ,__emac_ssir_bits);
__IO_REG32(    EMAC0_STSR,                  0x40064708,__READ       );
__IO_REG32_BIT(EMAC0_STNR,                  0x4006470C,__READ       ,__emac_stnr_bits);
__IO_REG32(    EMAC0_STSUR,                 0x40064710,__READ_WRITE );
__IO_REG32_BIT(EMAC0_STSNUR,                0x40064714,__READ_WRITE ,__emac_stsnur_bits);
__IO_REG32(    EMAC0_TSAR,                  0x40064718,__READ_WRITE );
__IO_REG32(    EMAC0_TTSR,                  0x4006471C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_TTNR,                  0x40064720,__READ_WRITE ,__emac_ttnr_bits);
__IO_REG32_BIT(EMAC0_STHWSR,                0x40064724,__READ_WRITE ,__emac_sthwsr_bits);
__IO_REG32_BIT(EMAC0_TSR,                   0x40064728,__READ_WRITE ,__emac_tsr_bits);
__IO_REG32_BIT(EMAC0_PPSCR,                 0x4006472C,__READ_WRITE ,__emac_ppscr_bits);
__IO_REG32_BIT(EMAC0_ATNR,                  0x40064730,__READ       ,__emac_atnr_bits);
__IO_REG32(    EMAC0_ATSR,                  0x40064734,__READ       );
__IO_REG32_BIT(EMAC0_MAR16H,                0x40064800,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR16L,                0x40064804,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR17H,                0x40064808,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR17L,                0x4006480C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR18H,                0x40064810,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR18L,                0x40064814,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR19H,                0x40064818,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR19L,                0x4006481C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR20H,                0x40064820,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR20L,                0x40064824,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR21H,                0x40064828,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR21L,                0x4006482C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR22H,                0x40064830,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR22L,                0x40064834,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR23H,                0x40064838,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR23L,                0x4006483C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR24H,                0x40064840,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR24L,                0x40064844,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR25H,                0x40064848,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR25L,                0x4006484C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR26H,                0x40064850,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR26L,                0x40064854,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR27H,                0x40064858,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR27L,                0x4006485C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR28H,                0x40064860,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR28L,                0x40064864,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR29H,                0x40064868,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR29L,                0x4006486C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR30H,                0x40064870,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR30L,                0x40064874,__READ_WRITE );
__IO_REG32_BIT(EMAC0_MAR31H,                0x40064878,__READ_WRITE ,__emac_marxh_bits);
__IO_REG32(    EMAC0_MAR31L,                0x4006487C,__READ_WRITE );
__IO_REG32_BIT(EMAC0_BMR,                   0x40065000,__READ_WRITE ,__emac_bmr_bits);
__IO_REG32(    EMAC0_TPDR,                  0x40065004,__READ_WRITE );
__IO_REG32(    EMAC0_RPDR,                  0x40065008,__READ_WRITE );
__IO_REG32(    EMAC0_RDLAR,                 0x4006500C,__READ_WRITE );
__IO_REG32(    EMAC0_TDLAR,                 0x40065010,__READ_WRITE );
__IO_REG32_BIT(EMAC0_SR,                    0x40065014,__READ_WRITE ,__emac_sr_bits);
__IO_REG32_BIT(EMAC0_OMR,                   0x40065018,__READ_WRITE ,__emac_omr_bits);
__IO_REG32_BIT(EMAC0_IER,                   0x4006501C,__READ_WRITE ,__emac_ier_bits);
__IO_REG32_BIT(EMAC0_MFBOCR,                0x40065020,__READ_WRITE ,__emac_mfbocr_bits);
__IO_REG32_BIT(EMAC0_RIWTR,                 0x40065024,__READ_WRITE ,__emac_riwtr_bits);
__IO_REG32_BIT(EMAC0_AHBSR,                 0x4006502C,__READ       ,__emac_ahbsr_bits);
__IO_REG32(    EMAC0_CHTDR,                 0x40065048,__READ       );
__IO_REG32(    EMAC0_CHRDR,                 0x4006504C,__READ       );
__IO_REG32(    EMAC0_CHTBAR,                0x40065050,__READ       );
__IO_REG32(    EMAC0_CHRBAR,                0x40065054,__READ       );

/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  MB9BF218T Interrupt Lines
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
#define NVIC_CSV              16          /* Anomalous Frequency Detection by Clock Supervisor (FCS)                                             */
#define NVIC_SWDT             17          /* Software Watchdog Timer                                                                             */
#define NVIC_LVD              18          /* Low Voltage Detector (LVD)                                                                          */
#define NVIC_WFG              19          /* MFT unit0, unit1, unit2 Wave Form Generator / DTIF(Motor Emergency Stop)                            */
#define NVIC_EXTI0_7          20          /* External Interrupt Request ch.0 to ch.7                                                             */
#define NVIC_EXTI8_31         21          /* External Interrupt Request ch.8 to ch.31                                                            */
#define NVIC_DTIM_QDU         22          /* Dual Timer / Quad Counter (QPRC) ch.0, ch.1, ch.2                                                   */
#define NVIC_MFSI0RX          23          /* Reception Interrupt Request of Multi-Function Serial Interface ch.0                                 */
#define NVIC_MFSI0TX          24          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.0 */
#define NVIC_MFSI1RX          25          /* Reception Interrupt Request of Multi-Function Serial Interface ch.1                                 */
#define NVIC_MFSI1TX          26          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.1 */
#define NVIC_MFSI2RX          27          /* Reception Interrupt Request of Multi-Function Serial Interface ch.2                                 */
#define NVIC_MFSI2TX          28          /* Transmission Interrupt Request and Status Interrupt Request of Multi-FunctionSerial Interface ch.2  */
#define NVIC_MFSI3RX          29          /* Reception Interrupt Request of Multi-Function Serial Interface ch.3                                 */
#define NVIC_MFSI3TX          30          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.3 */
#define NVIC_MFSI4RX          31          /* Reception Interrupt Request of Multi-Function Serial Interface ch.4                                 */
#define NVIC_MFSI4TX          32          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.4 */
#define NVIC_MFSI5RX          33          /* Reception Interrupt Request of Multi-Function Serial Interface ch.5                                 */
#define NVIC_MFSI5TX          34          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.5 */
#define NVIC_MFSI6RX          35          /* Reception Interrupt Request of Multi-Function Serial Interface ch.6                                 */
#define NVIC_MFSI6TX          36          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.6 */
#define NVIC_MFSI7RX          37          /* Reception Interrupt Request of Multi-Function Serial Interface ch.7                                 */
#define NVIC_MFSI7TX          38          /* Transmission Interrupt Request and Status Interrupt Request of Multi-Function Serial Interface ch.7 */
#define NVIC_PPG              39          /* PPG ch.0/2/4/8/10/12/16/18/20                                                                       */
#define NVIC_OSC_PLL_WC       40          /* External Main OSC / External Sub OSC / Main PLL / PLL for USB / Watch Counter 0xA0                  */
#define NVIC_ADC0             41          /* A/D Converter unit0                                                                                 */
#define NVIC_ADC1             42          /* A/D Converter unit1                                                                                 */
#define NVIC_ADC2             43          /* A/D Converter unit2                                                               */
#define NVIC_FRTIM            44          /* MFT unit0, unit1, unit2 Free-run Timer                                            */
#define NVIC_INCAP            45          /* MFT unit0, unit1, unit2 Input Capture                                             */
#define NVIC_OUTCOMP          46          /* MFT unit0, unit1, unit2 Output Compare                                            */
#define NVIC_BTIM0_7          47          /* Base Timer ch.0 to ch.7                                                           */
#define NVIC_ETH0             48          /* Ethernet ch.0                                                                     */
#define NVIC_USB0F            50          /* USB ch.0 Function (DRQ of End Point 1 to 5)                                       */
#define NVIC_USB0F_USB0H      51          /* USB ch.0 Function (DRQI of End Point 0, DRQO and each status) / USB HOST (each status) */
#define NVIC_USB1F            52          /* USB ch.1 Function (DRQ of End Point 1 to 5)                                       */
#define NVIC_USB1F_USB1H      53          /* USB ch.1 Function (DRQI of End Point 0, DRQO and each status) / USB HOST (each status) */
#define NVIC_DMAC0            54          /* DMA Controller (DMAC) ch.0                                                        */
#define NVIC_DMAC1            55          /* DMA Controller (DMAC) ch.1                                                        */
#define NVIC_DMAC2            56          /* DMA Controller (DMAC) ch.2                                                        */
#define NVIC_DMAC3            57          /* DMA Controller (DMAC) ch.3                                                        */
#define NVIC_DMAC4            58          /* DMA Controller (DMAC) ch.4                                                        */
#define NVIC_DMAC5            59          /* DMA Controller (DMAC) ch.5                                                        */
#define NVIC_DMAC6            60          /* DMA Controller (DMAC) ch.6                                                        */
#define NVIC_DMAC7            61          /* DMA Controller (DMAC) ch.7                                                        */
#define NVIC_BTIM8_15         62          /* Base Timer ch.8 to ch.15                                                          */

#endif    /* __IOMB9BF218T_H */

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
Interrupt9   = CSV            0x40
Interrupt10  = SWDT           0x44
Interrupt11  = LVD            0x48
Interrupt12  = WFG            0x4C
Interrupt13  = EXTI0_7        0x50
Interrupt14  = EXTI8_31       0x54
Interrupt15  = DTIM_QDU       0x58
Interrupt16  = MFSI0RX        0x5C
Interrupt17  = MFSI0TX        0x60
Interrupt18  = MFSI1RX        0x64
Interrupt19  = MFSI1TX        0x68
Interrupt20  = MFSI2RX        0x6C
Interrupt21  = MFSI2TX        0x70
Interrupt22  = MFSI3RX        0x74
Interrupt23  = MFSI3TX        0x78
Interrupt24  = MFSI4RX        0x7C
Interrupt25  = MFSI4TX        0x80
Interrupt26  = MFSI5RX        0x84
Interrupt27  = MFSI5TX        0x88
Interrupt28  = MFSI6RX        0x8C
Interrupt29  = MFSI6TX        0x90
Interrupt30  = MFSI7RX        0x94
Interrupt31  = MFSI7TX        0x98
Interrupt32  = PPG            0x9C
Interrupt33  = OSC_PLL_WC     0xA0
Interrupt34  = ADC0           0xA4
Interrupt35  = ADC1           0xA8
Interrupt36  = ADC2           0xAC
Interrupt37  = FRTIM          0xB0
Interrupt38  = INCAP          0xB4
Interrupt39  = OUTCOMP        0xB8
Interrupt40  = BTIM0_7        0xBC
Interrupt41  = ETH0           0xC0
Interrupt42  = USB0F          0xC8
Interrupt43  = USB0F_USB0H    0xCC
Interrupt44  = USB1F          0xD0
Interrupt45  = USB1F_USB1H    0xD4
Interrupt46  = DMAC0          0xD8
Interrupt47  = DMAC1          0xDC
Interrupt48  = DMAC2          0xE0
Interrupt49  = DMAC3          0xE4
Interrupt50  = DMAC4          0xE8
Interrupt51  = DMAC5          0xEC
Interrupt52  = DMAC6          0xF0
Interrupt53  = DMAC7          0xF4
Interrupt54  = BTIM8_15       0xF8

###DDF-INTERRUPT-END###*/
