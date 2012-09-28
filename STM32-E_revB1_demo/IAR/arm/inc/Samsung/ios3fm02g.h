/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Samsung S3FM02G
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 40214 $
 **
 ***************************************************************************/

#ifndef __IOS3FM02G_H
#define __IOS3FM02G_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    S3FM02G SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C specific declarations  ************************************************/

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

/* ADC ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __adc_idr_bits;

/* ADC Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :30;
  __REG32  DBGEN          : 1;
} __adc_cedr_bits;

/* ADC Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __adc_srr_bits;

/* ADC Control Set Register */
typedef struct {
  __REG32  ADCEN0         : 1;
  __REG32  START0         : 1;
  __REG32  STOP0          : 1;
  __REG32                 : 5;
  __REG32  ADCEN1         : 1;
  __REG32  START1         : 1;
  __REG32  STOP1          : 1;
  __REG32                 :21;
} __adc_csr_bits;

/* ADC Control Clear Register */
typedef struct {
  __REG32  ADCEN0         : 1;
  __REG32                 : 7;
  __REG32  ADCEN1         : 1;
  __REG32                 :23;
} __adc_ccr_bits;

/* ADC Control Divider Register */
typedef struct {
  __REG32  CDIV           : 5;
  __REG32                 :27;
} __adc_cdr_bits;

/* ADC Mode Register */
typedef struct {
  __REG32                 : 5;
  __REG32  TRIG0          : 3;
  __REG32                 : 5;
  __REG32  TRIG1          : 3;
  __REG32  CALEN0         : 1;
  __REG32  ICRV0          : 1;
  __REG32  EICR0          : 1;
  __REG32                 : 1;
  __REG32  CMODE0         : 1;
  __REG32  CCNT0          : 3;
  __REG32  CALEN1         : 1;
  __REG32  ICRV1          : 1;
  __REG32  EICR1          : 1;
  __REG32                 : 1;
  __REG32  CMODE1         : 1;
  __REG32  CCNT1          : 3;
} __adc_mr_bits;

/* ADC Conversion Channel Sequence Register */
typedef struct {
  __REG32  ICNUM0         : 4;
  __REG32  ICNUM1         : 4;
  __REG32  ICNUM2         : 4;
  __REG32  ICNUM3         : 4;
  __REG32  ICNUM4         : 4;
  __REG32  ICNUM5         : 4;
  __REG32  ICNUM6         : 4;
  __REG32  ICNUM7         : 4;
} __adc_ccsr_bits;

/* ADC Status Register */
typedef struct {
  __REG32  ADCSTABLE0     : 1;
  __REG32  BUSY0          : 1;
  __REG32  ADCEN0         : 1;
  __REG32  CTCVS0         : 1;
  __REG32                 : 4;
  __REG32  ADCSTABLE1     : 1;
  __REG32  BUSY1          : 1;
  __REG32  ADCEN1         : 1;
  __REG32  CTCVS1         : 1;
  __REG32                 :20;
} __adc_sr_bits;

/* ADC Interrupt Mask Set/Clear Register */
/* ADC Raw Interrupt Status Register */
/* ADC Masked Interrupt Status Register */
/* ADC Interrupt Clear Register */
typedef struct {
  __REG32  EOC0           : 1;
  __REG32  OVR0           : 1;
  __REG32                 : 6;
  __REG32  EOC1           : 1;
  __REG32  OVR1           : 1;
  __REG32                 :22;
} __adc_imscr_bits;

/* ADC Conversion Result Register */
typedef struct {
  __REG32  DATA           :12;
  __REG32                 :20;
} __adc_crr_bits;

/* ADC Gain Calibration Register */
typedef struct {
  __REG32  GCC_FRAC       :14;
  __REG32  GCC_INT        : 1;
  __REG32                 :17;
} __adc_gcr_bits;

/* ADC Gain Calibration Register */
typedef struct {
  __REG32  ADCOCC         :14;
  __REG32                 :18;
} __adc_ocr_bits;

/* ADC Conversion Result Register */
typedef struct {
  __REG32  DMAE0          : 1;
  __REG32  DMAE1          : 1;
  __REG32                 :30;
} __adc_dmacr_bits;

/* ADC1 Control Set Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32  START          : 1;
  __REG32  STOP           : 1;
  __REG32                 :29;
} __adc1_csr_bits;

/* ADC1 Control Clear Register */
typedef struct {
  __REG32  ADCEN          : 1;
  __REG32                 :31;
} __adc1_ccr_bits;

/* ADC1 Mode Register */
typedef struct {
  __REG32                 : 5;
  __REG32  TRIG           : 3;
  __REG32                 :16;
  __REG32  CALEN          : 1;
  __REG32                 : 1;
  __REG32  EICR           : 1;
  __REG32                 : 1;
  __REG32  CMODE          : 1;
  __REG32  CCNT           : 3;
} __adc1_mr_bits;

/* ADC1 Conversion Channel Sequence Register */
typedef struct {
  __REG32  ICNUM0         : 3;
  __REG32                 : 1;
  __REG32  ICNUM1         : 3;
  __REG32                 : 1;
  __REG32  ICNUM2         : 3;
  __REG32                 : 1;
  __REG32  ICNUM3         : 3;
  __REG32                 : 1;
  __REG32  ICNUM4         : 3;
  __REG32                 : 1;
  __REG32  ICNUM5         : 3;
  __REG32                 : 1;
  __REG32  ICNUM6         : 3;
  __REG32                 : 1;
  __REG32  ICNUM7         : 3;
  __REG32                 : 1;
} __adc1_ccsr_bits;

/* ADC1 Status Register */
typedef struct {
  __REG32  ADCSTABLE      : 1;
  __REG32  BUSY           : 1;
  __REG32  ADCEN          : 1;
  __REG32  CTCVS          : 1;
  __REG32                 :28;
} __adc1_sr_bits;

/* ADC1 Interrupt Mask Set/Clear Register */
/* ADC1 Raw Interrupt Status Register */
/* ADC1 Masked Interrupt Status Register */
/* ADC1 Interrupt Clear Register */
typedef struct {
  __REG32  EOC            : 1;
  __REG32  OVR            : 1;
  __REG32                 :30;
} __adc1_imscr_bits;

/* ADC1 Conversion Result Register */
typedef struct {
  __REG32  DATA           :10;
  __REG32                 :22;
} __adc1_crr_bits;

/* ADC1 Gain Calibration Register */
typedef struct {
  __REG32  ADCOCC         :12;
  __REG32                 :20;
} __adc1_ocr_bits;

/* ADC1 Conversion Result Register */
typedef struct {
  __REG32  DMAE           : 1;
  __REG32                 :31;
} __adc1_dmacr_bits;

/* CAN Enable Clock Register
   CAN Disable Clock Register
   CAN Power Management Status Register */
typedef struct {
  __REG32           : 1;
  __REG32 CAN       : 1;
  __REG32           :29;
  __REG32 DBGEN     : 1;
} __can_ecr_bits;

/* CAN Control Register */
typedef struct {
  __REG32 SWRST     : 1;
  __REG32 CANEN     : 1;
  __REG32 CANDIS    : 1;
  __REG32 CCEN      : 1;
  __REG32 CCDIS     : 1;
  __REG32           : 3;
  __REG32 RQBTX     : 1;
  __REG32 ABBTX     : 1;
  __REG32 STSR      : 1;
  __REG32           :21;
} __can_cr_bits;

/* CAN Mode Register */
typedef struct {
  __REG32 BD        :10;
  __REG32 CSSEL     : 1;
  __REG32           : 1;
  __REG32 SJW       : 2;
  __REG32 AR        : 1;
  __REG32           : 1;
  __REG32 PHSEG1    : 4;
  __REG32 PHSEG2    : 3;
  __REG32           : 9;
} __can_mr_bits;

/* CAN Clear Status Register */
typedef struct {
  __REG32           : 1;
  __REG32 ERWARNTR  : 1;
  __REG32 ERPASSTR  : 1;
  __REG32 BUSOFFTR  : 1;
  __REG32 ACTVT     : 1;
  __REG32           : 3;
  __REG32 RXOK      : 1;
  __REG32 TXOK      : 1;
  __REG32 STUFF     : 1;
  __REG32 FORM      : 1;
  __REG32 ACK       : 1;
  __REG32 BIT1      : 1;
  __REG32 BIT0      : 1;
  __REG32 CRC       : 1;
  __REG32           :16;
} __can_csr_bits;

/* CAN Status Register */
typedef struct {
  __REG32 ISS       : 1;
  __REG32 ERWARNTR  : 1;
  __REG32 ERPASSTR  : 1;
  __REG32 BUSOFFTR  : 1;
  __REG32 ACTVT     : 1;
  __REG32           : 3;
  __REG32 RXOK      : 1;
  __REG32 TXOK      : 1;
  __REG32 STUFF     : 1;
  __REG32 FORM      : 1;
  __REG32 ACK       : 1;
  __REG32 BIT1      : 1;
  __REG32 BIT0      : 1;
  __REG32 CRC       : 1;
  __REG32 CANENS    : 1;
  __REG32 ERWARN    : 1;
  __REG32 ERPASS    : 1;
  __REG32 BUSOFF    : 1;
  __REG32 BUSY0     : 1;
  __REG32 BUSY1     : 1;
  __REG32 RS        : 1;
  __REG32 TS        : 1;
  __REG32 CCENS     : 1;
  __REG32 BTXPD     : 1;
  __REG32           : 6;
} __can_sr_bits;

/* CAN Interrupt Enable Register
   CAN Interrupt Disable Register
   CAN Interrupt Mask Register */
typedef struct {
  __REG32           : 1;
  __REG32 ERWARNTR  : 1;
  __REG32 ERPASSTR  : 1;
  __REG32 BUSOFFTR  : 1;
  __REG32 ACTVT     : 1;
  __REG32           : 3;
  __REG32 RXOK      : 1;
  __REG32 TXOK      : 1;
  __REG32 STUFF     : 1;
  __REG32 FORM      : 1;
  __REG32 ACK       : 1;
  __REG32 BIT1      : 1;
  __REG32 BIT0      : 1;
  __REG32 CRC       : 1;
  __REG32           :16;
} __can_ier_bits;

/* CAN Interrupt Source Status Register
   CAN Source Interrupt Enable Register
   CAN Source Interrupt Disable Register
   CAN Source Interrupt Mask Register
   CAN Transmission Request Register
   CAN New Data Register
   CAN Message Valid Register */
typedef struct {
  __REG32 CH1       : 1;
  __REG32 CH2       : 1;
  __REG32 CH3       : 1;
  __REG32 CH4       : 1;
  __REG32 CH5       : 1;
  __REG32 CH6       : 1;
  __REG32 CH7       : 1;
  __REG32 CH8       : 1;
  __REG32 CH9       : 1;
  __REG32 CH10      : 1;
  __REG32 CH11      : 1;
  __REG32 CH12      : 1;
  __REG32 CH13      : 1;
  __REG32 CH14      : 1;
  __REG32 CH15      : 1;
  __REG32 CH16      : 1;
  __REG32 CH17      : 1;
  __REG32 CH18      : 1;
  __REG32 CH19      : 1;
  __REG32 CH20      : 1;
  __REG32 CH21      : 1;
  __REG32 CH22      : 1;
  __REG32 CH23      : 1;
  __REG32 CH24      : 1;
  __REG32 CH25      : 1;
  __REG32 CH26      : 1;
  __REG32 CH27      : 1;
  __REG32 CH28      : 1;
  __REG32 CH29      : 1;
  __REG32 CH30      : 1;
  __REG32 CH31      : 1;
  __REG32 CH32      : 1;
} __can_issr_bits;

/* CAN Highest Priority Interrupt Register */
typedef struct {
  __REG32 INTID     :16;
  __REG32           :16;
} __can_hpir_bits;

/* CAN Error Counter Register */
typedef struct {
  __REG32 REC       : 7;
  __REG32 REP       : 1;
  __REG32 TEC       : 8;
  __REG32           :16;
} __can_ercr_bits;

/* CAN Interface X Transfer Management Register */
typedef struct {
  __REG32 NUMBER    : 6;
  __REG32           : 1;
  __REG32 WR        : 1;
  __REG32 ADAR      : 1;
  __REG32 ADBR      : 1;
  __REG32 AMSKR     : 1;
  __REG32 AIR       : 1;
  __REG32 AMCR      : 1;
  __REG32           : 1;
  __REG32 TRND      : 1;
  __REG32 CLRIT     : 1;
  __REG32           :16;
} __can_tmr_bits;

/* CAN Interface X Data A/B Register */
typedef struct {
  __REG32 DATA0     : 8;
  __REG32 DATA1     : 8;
  __REG32 DATA2     : 8;
  __REG32 DATA3     : 8;
} __can_dar_bits;

/* CAN Interface X Mask Register */
typedef struct {
  __REG32 EXTMASK   :18;
  __REG32 BASEMASK  :11;
  __REG32           : 1;
  __REG32 MMDIR     : 1;
  __REG32 MXTD      : 1;
} __can_mskr_bits;

/* CAN Interface X Identifier Register */
typedef struct {
  __REG32 EXTID     :18;
  __REG32 BASEID    :11;
  __REG32 MDIR      : 1;
  __REG32 XTD       : 1;
  __REG32 MSGVAL    : 1;
} __can_ir_bits;

/* CAN Interface X Message Control Register */
typedef struct {
  __REG32 DLC       : 4;
  __REG32           : 3;
  __REG32 OVERWRITE : 1;
  __REG32 TXRQST    : 1;
  __REG32 RMTEN     : 1;
  __REG32 RXIE      : 1;
  __REG32 TXIE      : 1;
  __REG32 UMASK     : 1;
  __REG32 ITPND     : 1;
  __REG32 MSGLST    : 1;
  __REG32 NEWDAT    : 1;
  __REG32           :16;
} __can_mcr_bits;

/* CAN Test Register */
typedef struct {
  __REG32 BASIC     : 1;
  __REG32 SILENT    : 1;
  __REG32 LBACK     : 1;
  __REG32 TX        : 2;
  __REG32 TXOPD     : 1;
  __REG32 RX        : 1;
  __REG32           : 9;
  __REG32 TSTKEY    :16;
} __can_tstr_bits;

/* CM ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __cm_idr_bits;

/* CM Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __cm_srr_bits;

/* CM Control Set Register */
/* CM Control Clear Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32                 : 1;
  __REG32  FWAKE          : 1;
  __REG32                 : 1;
  __REG32  PLL            : 1;
  __REG32  STCLK          : 1;
  __REG32  PCLK           : 1;
  __REG32  ISCLKS         : 1;
  __REG32  IDLEW          : 1;
  __REG32                 : 7;
  __REG32  IDLESP         : 1;
  __REG32  ESCMRST        : 1;
  __REG32  ESCM           : 1;
  __REG32  EMCMRST        : 1;
  __REG32  EMCM           : 1;
  __REG32                 : 8;
} __cm_csr_bits;

/* CM Peripheral Clock Set Register 0 */
/* CM Peripheral Clock Clear Register 0 */
/* CM Peripheral Clock Status Register 0 */
typedef struct {
  __REG32  SFMCLK         : 1;
  __REG32  OPACLK         : 1;
  __REG32  WDTCLK         : 1;
  __REG32  FRTCLK         : 1;
  __REG32  PWM0CLK        : 1;
  __REG32  PWM1CLK        : 1;
  __REG32  ENCCLK         : 1;
  __REG32  IMCCLK         : 1;
  __REG32  TC0CLK         : 1;
  __REG32  TC1CLK         : 1;
  __REG32  TC2CLK         : 1;
  __REG32  TC3CLK         : 1;
  __REG32  TC4CLK         : 1;
  __REG32  TC5CLK         : 1;
  __REG32  TC6CLK         : 1;
  __REG32  TC7CLK         : 1;
  __REG32  USART0CLK      : 1;
  __REG32  USART1CLK      : 1;
  __REG32  USART2CLK      : 1;
  __REG32  USART3CLK      : 1;
  __REG32  CAN0CLK        : 1;
  __REG32  CAN1CLK        : 1;
  __REG32  ADC0CLK        : 1;
  __REG32  LCDCLK         : 1;
  __REG32  SPI0CLK        : 1;
  __REG32  SPI1CLK        : 1;
  __REG32  I2C0CLK        : 1;
  __REG32  I2C1CLK        : 1;
  __REG32                 : 1;
  __REG32  PFCCLK         : 1;
  __REG32  IOCLK          : 1;
  __REG32  STTCLK         : 1;
} __cm_pcsr0_bits;

/* CM Peripheral Clock Set Register 1 */
/* CM Peripheral Clock Clear Register 1 */
/* CM Peripheral Clock Status Register 1 */
typedef struct {
  __REG32  PWM2CLK        : 1;
  __REG32  PWM3CLK        : 1;
  __REG32  PWM4CLK        : 1;
  __REG32  PWM5CLK        : 1;
  __REG32  PWM6CLK        : 1;
  __REG32  PWM7CLK        : 1;
  __REG32  ENC1CLK        : 1;
  __REG32  IMC1CLK        : 1;
  __REG32  ADC1CLK        : 1;
  __REG32  DFCCLK         : 1;
  __REG32                 :22;
} __cm_pcsr1_bits;

/* CM Mode Register 0 */
typedef struct {
  __REG32  LVDRL          : 3;
  __REG32  LVDRSTEN       : 1;
  __REG32  LVDIL          : 3;
  __REG32  LVDINTEN       : 1;
  __REG32                 : 1;
  __REG32  RXEV           : 1;
  __REG32  STCLKEN        : 1;
  __REG32  LVDPD          : 1;
  __REG32  CLKOUT         : 3;
  __REG32                 :17;
} __cm_mr0_bits;

/* CM Mode Register 1 */
typedef struct {
  __REG32  SYSCLK         : 3;
  __REG32                 : 1;
  __REG32  WDTCLK         : 3;
  __REG32                 : 1;
  __REG32  FRTCLK         : 3;
  __REG32                 : 1;
  __REG32  STTCLK         : 3;
  __REG32                 : 1;
  __REG32  LCDCLK         : 3;
  __REG32                 :13;
} __cm_mr1_bits;

/* CM Interrupt Mask Set/Clear Register */
/* CM Masked Interrupt Status Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32  STABLE         : 1;
  __REG32                 : 2;
  __REG32  PLL            : 1;
  __REG32                 : 4;
  __REG32  ESCKFAIL_END   : 1;
  __REG32  ESCKFAIL       : 1;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32                 : 1;
  __REG32  CMDERR         : 1;
  __REG32                 :13;
} __cm_imscr_bits;

/* CM RAW Interrupt Status Register */
/* CM Interrupt Clear Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32  STABLE         : 1;
  __REG32                 : 2;
  __REG32  PLL            : 1;
  __REG32                 : 4;
  __REG32  ESCKFAIL_END   : 1;
  __REG32  ESCKFAIL       : 1;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32  LVDRS          : 1;
  __REG32  CMDERR         : 1;
  __REG32                 :13;
} __cm_risr_bits;

/* CM Status Register */
typedef struct {
  __REG32  EMCLK          : 1;
  __REG32  IMCLK          : 1;
  __REG32  ESCLK          : 1;
  __REG32  ISCLK          : 1;
  __REG32  STABLE         : 1;
  __REG32  FWAKE          : 1;
  __REG32                 : 1;
  __REG32  PLL            : 1;
  __REG32  STCLK          : 1;
  __REG32  PCLK           : 1;
  __REG32  ISCLKS         : 1;
  __REG32  IDLEW          : 1;
  __REG32  ESCKFAIL_END   : 1;
  __REG32  ESCKFAIL       : 1;
  __REG32  EMCKFAIL_END   : 1;
  __REG32  EMCKFAIL       : 1;
  __REG32  LVDINT         : 1;
  __REG32  LVDRS          : 1;
  __REG32  CMDERR         : 1;
  __REG32  IDLESP         : 1;
  __REG32  ESCMRST        : 1;
  __REG32  ESCM           : 1;
  __REG32  EMCMRST        : 1;
  __REG32  EMCM           : 1;
  __REG32  SWRSTS         : 1;
  __REG32  NRSTS          : 1;
  __REG32  LVDRSTS        : 1;
  __REG32  WDTRSTS        : 1;
  __REG32  PORRSTS        : 1;
  __REG32  ESCMRSTS       : 1;
  __REG32  EMCMRSTS       : 1;
  __REG32  SYSRSTS        : 1;
} __cm_sr_bits;

/* CM System Clock Divider Register */
typedef struct {
  __REG32  SDIV           : 3;
  __REG32                 :13;
  __REG32  SDIVKEY        :16;
} __cm_scdr_bits;

/* CM Peripheral Clock Divider Register */
typedef struct {
  __REG32  PDIV           : 4;
  __REG32                 :12;
  __REG32  PDIVKEY        :16;
} __cm_pcdr_bits;

/* CM FRT Clock Divider Register */
typedef struct {
  __REG32  NDIV           : 4;
  __REG32  MDIV           : 3;
  __REG32                 : 9;
  __REG32  FDIVKEY        :16;
} __cm_fcdr_bits;

/* CM STT Clock Divider Register */
typedef struct {
  __REG32  DDIV           : 4;
  __REG32  CDIV           : 3;
  __REG32                 : 9;
  __REG32  STDIVKEY       :16;
} __cm_stcdr_bits;

/* CM LCD Clock Divider Register */
typedef struct {
  __REG32  KDIV           : 4;
  __REG32  JDIV           : 3;
  __REG32                 : 9;
  __REG32  LDIVKEY        :16;
} __cm_lcdr_bits;

/* CM PLL Stabilization Time Register */
typedef struct {
  __REG32  PST            :11;
  __REG32                 : 5;
  __REG32  PLLSKEY        :16;
} __cm_pstr_bits;

/* CM PLL Divider Parameters Register */
typedef struct {
  __REG32  PLLMUL         : 8;
  __REG32  PLLPRE         : 6;
  __REG32                 : 2;
  __REG32  PLLPOST        : 2;
  __REG32                 : 1;
  __REG32  PLLCUS         : 3;
  __REG32                 : 2;
  __REG32  PLLKEY         : 8;
} __cm_pdpr_bits;

/* CM External Main Clock Stabilization Time Register */
typedef struct {
  __REG32  EMST           :16;
  __REG32  EMSKEY         :16;
} __cm_emstr_bits;

/* CM External Sub Clock Stabilization Time Register */
typedef struct {
  __REG32  ESST           :16;
  __REG32  ESSKEY         :16;
} __cm_esstr_bits;

/* CM Basic Timer Clock Divider Register */
typedef struct {
  __REG32  BTCDIV         : 4;
  __REG32                 :12;
  __REG32  BTCDKEY        :16;
} __cm_btcdr_bits;

/* CM Basic Timer Register */
typedef struct {
  __REG32  BTCV           :16;
  __REG32                 :16;
} __cm_btr_bits;

/* CM Wakeup Control Register 0 */
typedef struct {
  __REG32  WSRC0          : 5;
  __REG32                 : 1;
  __REG32  EDGE0          : 1;
  __REG32  WEN0           : 1;
  __REG32  WSRC1          : 5;
  __REG32                 : 1;
  __REG32  EDGE1          : 1;
  __REG32  WEN1           : 1;
  __REG32  WSRC2          : 5;
  __REG32                 : 1;
  __REG32  EDGE2          : 1;
  __REG32  WEN2           : 1;
  __REG32  WSRC3          : 5;
  __REG32                 : 1;
  __REG32  EDGE3          : 1;
  __REG32  WEN3           : 1;
} __cm_wcr0_bits;

/* CM Wakeup Control Register 1 */
typedef struct {
  __REG32  WSRC4          : 5;
  __REG32                 : 1;
  __REG32  EDGE4          : 1;
  __REG32  WEN4           : 1;
  __REG32  WSRC5          : 5;
  __REG32                 : 1;
  __REG32  EDGE5          : 1;
  __REG32  WEN5           : 1;
  __REG32  WSRC6          : 5;
  __REG32                 : 1;
  __REG32  EDGE6          : 1;
  __REG32  WEN6           : 1;
  __REG32  WSRC7          : 5;
  __REG32                 : 1;
  __REG32  EDGE7          : 1;
  __REG32  WEN7           : 1;
} __cm_wcr1_bits;

/* CM Wakeup Control Register 2 */
typedef struct {
  __REG32  WSRC8          : 5;
  __REG32                 : 1;
  __REG32  EDGE8          : 1;
  __REG32  WEN8           : 1;
  __REG32  WSRC9          : 5;
  __REG32                 : 1;
  __REG32  EDGE9          : 1;
  __REG32  WEN9           : 1;
  __REG32  WSRC10         : 5;
  __REG32                 : 1;
  __REG32  EDGE10         : 1;
  __REG32  WEN10          : 1;
  __REG32  WSRC11         : 5;
  __REG32                 : 1;
  __REG32  EDGE11         : 1;
  __REG32  WEN11          : 1;
} __cm_wcr2_bits;

/* CM Wakeup Control Register 3 */
typedef struct {
  __REG32  WSRC12         : 5;
  __REG32                 : 1;
  __REG32  EDGE12         : 1;
  __REG32  WEN12          : 1;
  __REG32  WSRC13         : 5;
  __REG32                 : 1;
  __REG32  EDGE13         : 1;
  __REG32  WEN13          : 1;
  __REG32  WSRC14         : 5;
  __REG32                 : 1;
  __REG32  EDGE14         : 1;
  __REG32  WEN14          : 1;
  __REG32  WSRC15         : 5;
  __REG32                 : 1;
  __REG32  EDGE15         : 1;
  __REG32  WEN15          : 1;
} __cm_wcr3_bits;

/* CM Wakeup Interrupt Mask Set/Clear Register */
/* CM Wakeup Raw Interrupt Status Register */
/* CM Wakeup Masked Interrupt Status Register */
/* CM Wakeup Interrupt Clear Register */
typedef struct {
  __REG32  WI0            : 1;
  __REG32  WI1            : 1;
  __REG32  WI2            : 1;
  __REG32  WI3            : 1;
  __REG32  WI4            : 1;
  __REG32  WI5            : 1;
  __REG32  WI6            : 1;
  __REG32  WI7            : 1;
  __REG32  WI8            : 1;
  __REG32  WI9            : 1;
  __REG32  WI10           : 1;
  __REG32  WI11           : 1;
  __REG32  WI12           : 1;
  __REG32  WI13           : 1;
  __REG32  WI14           : 1;
  __REG32  WI15           : 1;
  __REG32                 :16;
} __cm_wimscr_bits;

/* CM Wakeup Interrupt Clear Register 0 */
typedef struct {
  __REG32  NVIC0          : 1;
  __REG32  NVIC1          : 1;
  __REG32  NVIC2          : 1;
  __REG32  NVIC3          : 1;
  __REG32  NVIC4          : 1;
  __REG32  NVIC5          : 1;
  __REG32  NVIC6          : 1;
  __REG32  NVIC7          : 1;
  __REG32  NVIC8          : 1;
  __REG32  NVIC9          : 1;
  __REG32  NVIC10         : 1;
  __REG32  NVIC11         : 1;
  __REG32  NVIC12         : 1;
  __REG32  NVIC13         : 1;
  __REG32  NVIC14         : 1;
  __REG32  NVIC15         : 1;
  __REG32  NVIC16         : 1;
  __REG32  NVIC17         : 1;
  __REG32  NVIC18         : 1;
  __REG32  NVIC19         : 1;
  __REG32  NVIC20         : 1;
  __REG32  NVIC21         : 1;
  __REG32  NVIC22         : 1;
  __REG32  NVIC23         : 1;
  __REG32  NVIC24         : 1;
  __REG32  NVIC25         : 1;
  __REG32  NVIC26         : 1;
  __REG32  NVIC27         : 1;
  __REG32  NVIC28         : 1;
  __REG32  NVIC29         : 1;
  __REG32  NVIC30         : 1;
  __REG32  NVIC31         : 1;
} __cm_nisr0_bits;

/* CM Wakeup Interrupt Clear Register 1 */
typedef struct {
  __REG32  NVIC32         : 1;
  __REG32  NVIC33         : 1;
  __REG32  NVIC34         : 1;
  __REG32  NVIC35         : 1;
  __REG32  NVIC36         : 1;
  __REG32  NVIC37         : 1;
  __REG32  NVIC38         : 1;
  __REG32  NVIC39         : 1;
  __REG32  NVIC40         : 1;
  __REG32  NVIC41         : 1;
  __REG32  NVIC42         : 1;
  __REG32  NVIC43         : 1;
  __REG32  NVIC44         : 1;
  __REG32  NVIC45         : 1;
  __REG32  NVIC46         : 1;
  __REG32  NVIC47         : 1;
  __REG32  NVIC48         : 1;
  __REG32  NVIC49         : 1;
  __REG32  NVIC50         : 1;
  __REG32  NVIC51         : 1;
  __REG32  NVIC52         : 1;
  __REG32  NVIC53         : 1;
  __REG32  NVIC54         : 1;
  __REG32  NVIC55         : 1;
  __REG32  NVIC56         : 1;
  __REG32  NVIC57         : 1;
  __REG32  NVIC58         : 1;
  __REG32  NVIC59         : 1;
  __REG32  NVIC60         : 1;
  __REG32  NVIC61         : 1;
  __REG32  NVIC62         : 1;
  __REG32  NVIC63         : 1;
} __cm_nisr1_bits;

/* Data Flash ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __df_idr_bits;

/* Data Flash Clock Enable/Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :31;
} __df_cedr_bits;

/* Data Flash Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __df_srr_bits;

/* Data Flash Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  DFEN                  : 1;
  __REG32  DFSTABLE              : 1;
  __REG32                        : 1;
  __REG32  CMD                   : 3;
  __REG32                        :25;
} __df_cr_bits;

/* Data Flash Control Register */
typedef struct {
  __REG32  WAIT                  : 4;
  __REG32                        :28;
} __df_mr_bits;

/* Data Flash Interrupt Mask Set/Clear Register */
/* Data Flash Raw Interrupt Status Register */
/* Data Flash Masked Interrupt Status Register */
/* Data Flash Interrupt Clear Register */
typedef struct {
  __REG32  END                   : 1;
  __REG32                        : 7;
  __REG32  ERR0                  : 1;
  __REG32  ERR1                  : 1;
  __REG32  ERR2                  : 1;
  __REG32                        :21;
} __df_imscr_bits;

/* Data Flash Protection Control Register */
typedef struct {
  __REG32  WEPR0                 : 1;
  __REG32  WEPR1                 : 1;
  __REG32  WEPR2                 : 1;
  __REG32  WEPR3                 : 1;
  __REG32  WEPR4                 : 1;
  __REG32  WEPR5                 : 1;
  __REG32  WEPR6                 : 1;
  __REG32  WEPR7                 : 1;
  __REG32  WEPR8                 : 1;
  __REG32  WEPR9                 : 1;
  __REG32  WEPR10                : 1;
  __REG32  WEPR11                : 1;
  __REG32  WEPR12                : 1;
  __REG32  WEPR13                : 1;
  __REG32  WEPR15                : 1;
  __REG32  WEPR16                : 1;
  __REG32  PRKEY                 :16;
} __df_pcr_bits;

/* DMA Channel x Initial Source Control Register */
/* DMA Channel x Initial Destination Control Register */
typedef struct {
  __REG32  LINC                  : 1;
  __REG32  HINC                  : 1;
  __REG32                        :30;
} __dma_iscr_bits;

/* DMA Channel x Control Register */
typedef struct {
  __REG32  LTC                   :12;
  __REG32  HTC                   :12;
  __REG32  DSIZE                 : 2;
  __REG32  RELOAD                : 1;
  __REG32  SMODE                 : 1;
  __REG32  TSIZE                 : 1;
  __REG32  LTCINT                : 1;
  __REG32  TCINT                 : 1;
  __REG32                        : 1;
} __dma_cr_bits;

/* DMA Channel x Status Register */
typedef struct {
  __REG32  CURR_LTC              :12;
  __REG32  CURR_HTC              :12;
  __REG32                        : 7;
  __REG32  LTCST                 : 1;
} __dma_sr_bits;

/* DMA Channel x Mask Trigger Register */
typedef struct {
  __REG32  SWTRIG                : 1;
  __REG32  CHEN                  : 1;
  __REG32  STOP                  : 1;
  __REG32                        :29;
} __dma_mtr_bits;

/* DMA Channel x Request Selection Register */
typedef struct {
  __REG32  REQ                   : 1;
  __REG32  HWSRC                 : 5;
  __REG32                        :26;
} __dma_rsr_bits;

/* DMA ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __dma_idr_bits;

/* DMA Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __dma_srr_bits;

/* DMA Channel Enable Status Register */
typedef struct {
  __REG32  CH0EN                 : 1;
  __REG32  CH1EN                 : 1;
  __REG32  CH2EN                 : 1;
  __REG32  CH3EN                 : 1;
  __REG32  CH4EN                 : 1;
  __REG32  CH5EN                 : 1;
  __REG32                        :26;
} __dma_cesr_bits;

/* DMA Interrupt Status Register */
typedef struct {
  __REG32  CH0_LTCIT             : 1;
  __REG32  CH1_LTCIT             : 1;
  __REG32  CH2_LTCIT             : 1;
  __REG32  CH3_LTCIT             : 1;
  __REG32  CH4_LTCIT             : 1;
  __REG32  CH5_LTCIT             : 1;
  __REG32                        :10;
  __REG32  CH0_TCIT              : 1;
  __REG32  CH1_TCIT              : 1;
  __REG32  CH2_TCIT              : 1;
  __REG32  CH3_TCIT              : 1;
  __REG32  CH4_TCIT              : 1;
  __REG32  CH5_TCIT              : 1;
  __REG32                        :10;
} __dma_isr_bits;

/* DMA Interrupt Status Register */
typedef struct {
  __REG32  CH0_IT                : 1;
  __REG32  CH1_IT                : 1;
  __REG32  CH2_IT                : 1;
  __REG32  CH3_IT                : 1;
  __REG32  CH4_IT                : 1;
  __REG32  CH5_IT                : 1;
  __REG32                        :26;
} __dma_icr_bits;

/* ENC ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __enc_idr_bits;

/* ENC Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __enc_cedr_bits;

/* ENC Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __enc_srr_bits;

/* ENC Control Register 0 */
typedef struct {
  __REG32  PCRCL                 : 1;
  __REG32  SPCRCL                : 1;
  __REG32  ENCEN                 : 1;
  __REG32  ESELZ                 : 1;
  __REG32  ENCFILTER             : 3;
  __REG32  PZCLEN                : 1;
  __REG32  ENCCLKSEL             : 3;
  __REG32                        :21;
} __enc_cr0_bits;

/* ENC Control Register 1 */
typedef struct {
  __REG32  PBCCRCL               : 1;
  __REG32  PBEN                  : 1;
  __REG32  ESELB                 : 2;
  __REG32  PRESCALEB             : 4;
  __REG32  PACCRCL               : 1;
  __REG32  PAEN                  : 1;
  __REG32  ESELA                 : 2;
  __REG32  PRESCALEA             : 4;
  __REG32                        :16;
} __enc_cr1_bits;

/* ENC Status Register */
typedef struct {
  __REG32  DIRECTION             : 1;
  __REG32  GLITCH                : 1;
  __REG32  PBSTAT                : 1;
  __REG32  PASTAT                : 1;
  __REG32  OFPCNT                : 1;
  __REG32  UFPCNT                : 1;
  __REG32  OFSCNT                : 1;
  __REG32  UFSCNT                : 1;
  __REG32                        :24;
} __enc_sr_bits;

/* ENC Interrupt Mask Set and Clear Register */
/* ENC Raw Interrupt Status Register */
/* ENC Masked Interrupt Status Register */
/* ENC Interrupt Clear Register */
typedef struct {
  __REG32  PAOVF                 : 1;
  __REG32  PACAP                 : 1;
  __REG32  PBOVF                 : 1;
  __REG32  PBCAP                 : 1;
  __REG32  PCMAT                 : 1;
  __REG32  SCMAT                 : 1;
  __REG32                        : 1;
  __REG32  PHASEZ                : 1;
  __REG32                        :24;
} __enc_imscr_bits;

/* FRT ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __frt_idr_bits;

/* Clock Enable/Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __frt_cedr_bits;

/* Software Reset Register */
/* Status Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __frt_srr_bits;

/* Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 7;
  __REG32  FRTSIZE               : 5;
  __REG32                        : 3;
  __REG32  CKDIV                 :16;
} __frt_cr_bits;

/* Interrupt Enable Disable Register */
/* Raw Interrupt Status Register */
/* Masked Interrupt Status Register */
/* Interrupt Clear Register */
typedef struct {
  __REG32  OVF                   : 1;
  __REG32                        : 1;
  __REG32  MATCH                 : 1;
  __REG32                        :29;
} __frt_imscr_bits;

/* GPIO ID-Code Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __gpio_idr_bits;

/* GPIO Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :31;
} __gpio_cedr_bits;

/* GPIO Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __gpio_srr_bits;

/* GPIO Interrupt Mask Set/Clear Register */
/* GPIO Raw Interrupt Status Register */
/* GPIO Masked Interrupt Status Register */
/* GPIO Interrupt Clear Register */
/* GPIO Output Enable Register */
/* GPIO Output Status Register */
/* GPIO Write Output Data Register */
/* GPIO Set Output Data Register */
/* GPIO Clear Output Data Register */
/* GPIO Output Data Status Register */
/* GPIO Pin Data Status Register */
typedef struct {
  __REG32  P0             : 1;
  __REG32  P1             : 1;
  __REG32  P2             : 1;
  __REG32  P3             : 1;
  __REG32  P4             : 1;
  __REG32  P5             : 1;
  __REG32  P6             : 1;
  __REG32  P7             : 1;
  __REG32  P8             : 1;
  __REG32  P9             : 1;
  __REG32  P10            : 1;
  __REG32  P11            : 1;
  __REG32  P12            : 1;
  __REG32  P13            : 1;
  __REG32  P14            : 1;
  __REG32  P15            : 1;
  __REG32  P16            : 1;
  __REG32  P17            : 1;
  __REG32  P18            : 1;
  __REG32  P19            : 1;
  __REG32  P20            : 1;
  __REG32  P21            : 1;
  __REG32  P22            : 1;
  __REG32  P23            : 1;
  __REG32  P24            : 1;
  __REG32  P25            : 1;
  __REG32  P26            : 1;
  __REG32  P27            : 1;
  __REG32  P28            : 1;
  __REG32  P29            : 1;
  __REG32  P30            : 1;
  __REG32  P31            : 1;
} __gpio_imscr_bits;

/* GPIO2 Interrupt Mask Set/Clear Register */
/* GPIO2 Raw Interrupt Status Register */
/* GPIO2 Masked Interrupt Status Register */
/* GPIO2 Interrupt Clear Register */
/* GPIO2 Output Enable Register */
/* GPIO2 Output Status Register */
/* GPIO2 Write Output Data Register */
/* GPIO2 Set Output Data Register */
/* GPIO2 Clear Output Data Register */
/* GPIO2 Output Data Status Register */
/* GPIO2 Pin Data Status Register */
typedef struct {
  __REG32  P0             : 1;
  __REG32  P1             : 1;
  __REG32  P2             : 1;
  __REG32  P3             : 1;
  __REG32  P4             : 1;
  __REG32  P5             : 1;
  __REG32  P6             : 1;
  __REG32  P7             : 1;
  __REG32  P8             : 1;
  __REG32  P9             : 1;
  __REG32  P10            : 1;
  __REG32  P11            : 1;
  __REG32  P12            : 1;
  __REG32  P13            : 1;
  __REG32  P14            : 1;
  __REG32  P15            : 1;
  __REG32  P16            : 1;
  __REG32  P17            : 1;
  __REG32  P18            : 1;
  __REG32  P19            : 1;
  __REG32  P20            : 1;
  __REG32  P21            : 1;
  __REG32  P22            : 1;
  __REG32  P23            : 1;
  __REG32  P24            : 1;
  __REG32  P25            : 1;
  __REG32  P26            : 1;
  __REG32  P27            : 1;
  __REG32                 : 4;
} __gpio2_imscr_bits;

/* GPIO3 Interrupt Mask Set/Clear Register */
/* GPIO3 Raw Interrupt Status Register */
/* GPIO3 Masked Interrupt Status Register */
/* GPIO3 Interrupt Clear Register */
/* GPIO3 Output Enable Register */
/* GPIO3 Output Status Register */
/* GPIO3 Write Output Data Register */
/* GPIO3 Set Output Data Register */
/* GPIO3 Clear Output Data Register */
/* GPIO3 Output Data Status Register */
/* GPIO3 Pin Data Status Register */
typedef struct {
  __REG32  P0             : 1;
  __REG32  P1             : 1;
  __REG32  P2             : 1;
  __REG32  P3             : 1;
  __REG32  P4             : 1;
  __REG32  P5             : 1;
  __REG32  P6             : 1;
  __REG32  P7             : 1;
  __REG32  P8             : 1;
  __REG32  P9             : 1;
  __REG32                 :22;
} __gpio3_imscr_bits;

/* I2C ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __i2c_idr_bits;

/* I2C Clock Enable Disable Register */
typedef struct {
  __REG32  CKEN                  : 1;
  __REG32                        :31;
} __i2c_cedr_bits;

/* I2C Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __i2c_srr_bits;

/* I2C Control Register */
typedef struct {
  __REG32                        : 1;
  __REG32  AA                    : 1;
  __REG32  STO                   : 1;
  __REG32  STA                   : 1;
  __REG32                        : 4;
  __REG32  ENA                   : 1;
  __REG32                        :23;
} __i2c_cr_bits;

/* I2C Mode Register */
typedef struct {
  __REG32  PRV                   :12;
  __REG32  FAST                  : 1;
  __REG32                        :19;
} __i2c_mr_bits;

/* I2C Status Register */
typedef struct {
  __REG32                        : 3;
  __REG32  SR                    : 5;
  __REG32                        :24;
} __i2c_sr_bits;

/* I2C Interrupt Mask Set and Clear Register */
/* I2C Raw Interrupt Status Register */
/* I2C Masked Interrupt Status Register */
/* I2C Clear Interrupt Status Register */
typedef struct {
  __REG32                        : 4;
  __REG32  SI                    : 1;
  __REG32                        :27;
} __i2c_imscr_bits;

/* I2C Serial Data Register */
typedef struct {
  __REG32  DAT                   : 8;
  __REG32                        :24;
} __i2c_sdr_bits;

/* I2C Serial Slave Address Register */
typedef struct {
  __REG32  GC                    : 1;
  __REG32  ADR                   : 7;
  __REG32                        :24;
} __i2c_ssar_bits;

/* I2C Hold/Setup Delay Register */
typedef struct {
  __REG32  DL                    : 8;
  __REG32                        :24;
} __i2c_hsdr_bits;

/* I2C DMA Control register */
typedef struct {
  __REG32  RXDMAE                : 1;
  __REG32  TXDMAE                : 1;
  __REG32                        :30;
} __i2c_dmacr_bits;

/* IMC ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __imc_idr_bits;

/* IMC Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __imc_cedr_bits;

/* IMC Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __imc_srr_bits;

/* IMC Control Register 0 */
typedef struct {
  __REG32  IMEN                  : 1;
  __REG32  IMMODE                : 1;
  __REG32  WMODE                 : 1;
  __REG32  PWMSWAP               : 1;
  __REG32  PWMPOLU               : 1;
  __REG32  PWMPOLD               : 1;
  __REG32  ESELPWMOFF            : 2;
  __REG32  IMFILTER              : 3;
  __REG32                        : 1;
  __REG32  PWMOFFEN              : 1;
  __REG32  PWMOUTOFFEN           : 1;
  __REG32  PWMOUTEN              : 1;
  __REG32  PWMOUTOFFENBYOPAMP    : 1;
  __REG32  IMCLKSEL              : 3;
  __REG32                        : 1;
  __REG32  NUMSKIP               : 6;
  __REG32  SYNCSEL               : 2;
  __REG32                        : 4;
} __imc_cr0_bits;

/* IMC Control Register 1 */
typedef struct {
  __REG32  PWMxD2EN              : 1;
  __REG32  PWMxD1EN              : 1;
  __REG32  PWMxD0EN              : 1;
  __REG32  PWMxU2EN              : 1;
  __REG32  PWMxU1EN              : 1;
  __REG32  PWMxU0EN              : 1;
  __REG32                        : 2;
  __REG32  PWMxD2LEVEL           : 1;
  __REG32  PWMxD1LEVEL           : 1;
  __REG32  PWMxD0LEVEL           : 1;
  __REG32  PWMxU2LEVEL           : 1;
  __REG32  PWMxU1LEVEL           : 1;
  __REG32  PWMxU0LEVEL           : 1;
  __REG32                        : 2;
  __REG32  PWMxD2DT              : 1;
  __REG32  PWMxD1DT              : 1;
  __REG32  PWMxD0DT              : 1;
  __REG32  PWMxU2DT              : 1;
  __REG32  PWMxU1DT              : 1;
  __REG32  PWMxU0DT              : 1;
  __REG32                        :10;
} __imc_cr1_bits;

/* IMC Status Register */
typedef struct {
  __REG32  FAULTSTAT             : 1;
  __REG32  UPDOWN                : 1;
  __REG32                        :30;
} __imc_sr_bits;

/* IMC Interrupt Mask Set and Clear Register */
/* IMC Raw Interrupt Status Register */
/* IMC Masked Interrupt Status Register */
/* IMC Interrupt Clear Register */
typedef struct {
  __REG32  FAULT                 : 1;
  __REG32                        : 5;
  __REG32  ZERO                  : 1;
  __REG32  TOP                   : 1;
  __REG32  ADCRM0                : 1;
  __REG32  ADCFM0                : 1;
  __REG32  ADCRM1                : 1;
  __REG32  ADCFM1                : 1;
  __REG32  ADCRM2                : 1;
  __REG32  ADCFM2                : 1;
  __REG32                        :18;
} __imc_imscr_bits;

/* IMC ADC Start Signal Select Register */
typedef struct {
  __REG32  TOPCMPSEL             : 1;
  __REG32  _0SEL                 : 1;
  __REG32  ADCMPR0SEL            : 1;
  __REG32  ADCMPF0SEL            : 1;
  __REG32  ADCMPR1SEL            : 1;
  __REG32  ADCMPF1SEL            : 1;
  __REG32  ADCMPR2SEL            : 1;
  __REG32  ADCMPF2SEL            : 1;
  __REG32                        :24;
} __imc_astsr_bits;

/* IO P0 Mode Low Register*/
typedef struct {
  __REG32  IO0_0_FSEL            : 2;
  __REG32  IO0_1_FSEL            : 2;
  __REG32  IO0_2_FSEL            : 2;
  __REG32  IO0_3_FSEL            : 2;
  __REG32  IO0_4_FSEL            : 2;
  __REG32  IO0_5_FSEL            : 2;
  __REG32  IO0_6_FSEL            : 2;
  __REG32  IO0_7_FSEL            : 2;
  __REG32  IO0_8_FSEL            : 2;
  __REG32  IO0_9_FSEL            : 2;
  __REG32  IO0_10_FSEL           : 2;
  __REG32  IO0_11_FSEL           : 2;
  __REG32  IO0_12_FSEL           : 2;
  __REG32  IO0_13_FSEL           : 2;
  __REG32  IO0_14_FSEL           : 2;
  __REG32  IO0_15_FSEL           : 2;
} __ioconf_mlr0_bits;

/* IO P0 Mode High Register */
typedef struct {
  __REG32  IO0_16_FSEL           : 2;
  __REG32  IO0_17_FSEL           : 2;
  __REG32  IO0_18_FSEL           : 2;
  __REG32  IO0_19_FSEL           : 2;
  __REG32  IO0_20_FSEL           : 2;
  __REG32  IO0_21_FSEL           : 2;
  __REG32  IO0_22_FSEL           : 2;
  __REG32  IO0_23_FSEL           : 2;
  __REG32  IO0_24_FSEL           : 2;
  __REG32  IO0_25_FSEL           : 2;
  __REG32  IO0_26_FSEL           : 2;
  __REG32  IO0_27_FSEL           : 2;
  __REG32  IO0_28_FSEL           : 2;
  __REG32  IO0_29_FSEL           : 2;
  __REG32  IO0_30_FSEL           : 2;
  __REG32  IO0_31_FSEL           : 2;
} __ioconf_mhr0_bits;

/* IO P0 Pull-Up Configuration Register*/
typedef struct {
  __REG32  IO0_0_PUEN            : 1;
  __REG32  IO0_1_PUEN            : 1;
  __REG32  IO0_2_PUEN            : 1;
  __REG32  IO0_3_PUEN            : 1;
  __REG32  IO0_4_PUEN            : 1;
  __REG32  IO0_5_PUEN            : 1;
  __REG32  IO0_6_PUEN            : 1;
  __REG32  IO0_7_PUEN            : 1;
  __REG32  IO0_8_PUEN            : 1;
  __REG32  IO0_9_PUEN            : 1;
  __REG32  IO0_10_PUEN           : 1;
  __REG32  IO0_11_PUEN           : 1;
  __REG32  IO0_12_PUEN           : 1;
  __REG32  IO0_13_PUEN           : 1;
  __REG32  IO0_14_PUEN           : 1;
  __REG32  IO0_15_PUEN           : 1;
  __REG32  IO0_16_PUEN           : 1;
  __REG32  IO0_17_PUEN           : 1;
  __REG32  IO0_18_PUEN           : 1;
  __REG32  IO0_19_PUEN           : 1;
  __REG32  IO0_20_PUEN           : 1;
  __REG32  IO0_21_PUEN           : 1;
  __REG32  IO0_22_PUEN           : 1;
  __REG32  IO0_23_PUEN           : 1;
  __REG32  IO0_24_PUEN           : 1;
  __REG32  IO0_25_PUEN           : 1;
  __REG32  IO0_26_PUEN           : 1;
  __REG32  IO0_27_PUEN           : 1;
  __REG32  IO0_28_PUEN           : 1;
  __REG32  IO0_29_PUEN           : 1;
  __REG32  IO0_30_PUEN           : 1;
  __REG32  IO0_31_PUEN           : 1;
} __ioconf_pucr0_bits;

/* IO P0 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO0_0_ODEN            : 1;
  __REG32  IO0_1_ODEN            : 1;
  __REG32  IO0_2_ODEN            : 1;
  __REG32  IO0_3_ODEN            : 1;
  __REG32  IO0_4_ODEN            : 1;
  __REG32  IO0_5_ODEN            : 1;
  __REG32  IO0_6_ODEN            : 1;
  __REG32  IO0_7_ODEN            : 1;
  __REG32  IO0_8_ODEN            : 1;
  __REG32  IO0_9_ODEN            : 1;
  __REG32  IO0_10_ODEN           : 1;
  __REG32  IO0_11_ODEN           : 1;
  __REG32  IO0_12_ODEN           : 1;
  __REG32  IO0_13_ODEN           : 1;
  __REG32  IO0_14_ODEN           : 1;
  __REG32  IO0_15_ODEN           : 1;
  __REG32  IO0_16_ODEN           : 1;
  __REG32  IO0_17_ODEN           : 1;
  __REG32  IO0_18_ODEN           : 1;
  __REG32  IO0_19_ODEN           : 1;
  __REG32  IO0_20_ODEN           : 1;
  __REG32  IO0_21_ODEN           : 1;
  __REG32  IO0_22_ODEN           : 1;
  __REG32  IO0_23_ODEN           : 1;
  __REG32  IO0_24_ODEN           : 1;
  __REG32  IO0_25_ODEN           : 1;
  __REG32  IO0_26_ODEN           : 1;
  __REG32  IO0_27_ODEN           : 1;
  __REG32  IO0_28_ODEN           : 1;
  __REG32  IO0_29_ODEN           : 1;
  __REG32  IO0_30_ODEN           : 1;
  __REG32  IO0_31_ODEN           : 1;
} __ioconf_odcr0_bits;

/* IO P1 Mode Low Register */
typedef struct {
  __REG32  IO1_0_FSEL            : 2;
  __REG32  IO1_1_FSEL            : 2;
  __REG32  IO1_2_FSEL            : 2;
  __REG32  IO1_3_FSEL            : 2;
  __REG32  IO1_4_FSEL            : 2;
  __REG32  IO1_5_FSEL            : 2;
  __REG32  IO1_6_FSEL            : 2;
  __REG32  IO1_7_FSEL            : 2;
  __REG32  IO1_8_FSEL            : 2;
  __REG32  IO1_9_FSEL            : 2;
  __REG32  IO1_10_FSEL           : 2;
  __REG32  IO1_11_FSEL           : 2;
  __REG32  IO1_12_FSEL           : 2;
  __REG32  IO1_13_FSEL           : 2;
  __REG32  IO1_14_FSEL           : 2;
  __REG32  IO1_15_FSEL           : 2;
} __ioconf_mlr1_bits;

/* IO P1 Mode High Register */
typedef struct {
  __REG32  IO1_16_FSEL           : 2;
  __REG32  IO1_17_FSEL           : 2;
  __REG32  IO1_18_FSEL           : 2;
  __REG32  IO1_19_FSEL           : 2;
  __REG32  IO1_20_FSEL           : 2;
  __REG32  IO1_21_FSEL           : 2;
  __REG32  IO1_22_FSEL           : 2;
  __REG32  IO1_23_FSEL           : 2;
  __REG32  IO1_24_FSEL           : 2;
  __REG32  IO1_25_FSEL           : 2;
  __REG32  IO1_26_FSEL           : 2;
  __REG32  IO1_27_FSEL           : 2;
  __REG32  IO1_28_FSEL           : 2;
  __REG32  IO1_29_FSEL           : 2;
  __REG32  IO1_30_FSEL           : 2;
  __REG32  IO1_31_FSEL           : 2;
} __ioconf_mhr1_bits;

/* IO P1 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO1_0_PUEN            : 1;
  __REG32  IO1_1_PUEN            : 1;
  __REG32  IO1_2_PUEN            : 1;
  __REG32  IO1_3_PUEN            : 1;
  __REG32  IO1_4_PUEN            : 1;
  __REG32  IO1_5_PUEN            : 1;
  __REG32  IO1_6_PUEN            : 1;
  __REG32  IO1_7_PUEN            : 1;
  __REG32  IO1_8_PUEN            : 1;
  __REG32  IO1_9_PUEN            : 1;
  __REG32  IO1_10_PUEN           : 1;
  __REG32  IO1_11_PUEN           : 1;
  __REG32  IO1_12_PUEN           : 1;
  __REG32  IO1_13_PUEN           : 1;
  __REG32  IO1_14_PUEN           : 1;
  __REG32  IO1_15_PUEN           : 1;
  __REG32  IO1_16_PUEN           : 1;
  __REG32  IO1_17_PUEN           : 1;
  __REG32  IO1_18_PUEN           : 1;
  __REG32  IO1_19_PUEN           : 1;
  __REG32  IO1_20_PUEN           : 1;
  __REG32  IO1_21_PUEN           : 1;
  __REG32  IO1_22_PUEN           : 1;
  __REG32  IO1_23_PUEN           : 1;
  __REG32  IO1_24_PUEN           : 1;
  __REG32  IO1_25_PUEN           : 1;
  __REG32  IO1_26_PUEN           : 1;
  __REG32  IO1_27_PUEN           : 1;
  __REG32  IO1_28_PUEN           : 1;
  __REG32  IO1_29_PUEN           : 1;
  __REG32  IO1_30_PUEN           : 1;
  __REG32  IO1_31_PUEN           : 1;
} __ioconf_pucr1_bits;

/* IO P1 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO1_0_ODEN            : 1;
  __REG32  IO1_1_ODEN            : 1;
  __REG32  IO1_2_ODEN            : 1;
  __REG32  IO1_3_ODEN            : 1;
  __REG32  IO1_4_ODEN            : 1;
  __REG32  IO1_5_ODEN            : 1;
  __REG32  IO1_6_ODEN            : 1;
  __REG32  IO1_7_ODEN            : 1;
  __REG32  IO1_8_ODEN            : 1;
  __REG32  IO1_9_ODEN            : 1;
  __REG32  IO1_10_ODEN           : 1;
  __REG32  IO1_11_ODEN           : 1;
  __REG32  IO1_12_ODEN           : 1;
  __REG32  IO1_13_ODEN           : 1;
  __REG32  IO1_14_ODEN           : 1;
  __REG32  IO1_15_ODEN           : 1;
  __REG32  IO1_16_ODEN           : 1;
  __REG32  IO1_17_ODEN           : 1;
  __REG32  IO1_18_ODEN           : 1;
  __REG32  IO1_19_ODEN           : 1;
  __REG32  IO1_20_ODEN           : 1;
  __REG32  IO1_21_ODEN           : 1;
  __REG32  IO1_22_ODEN           : 1;
  __REG32  IO1_23_ODEN           : 1;
  __REG32  IO1_24_ODEN           : 1;
  __REG32  IO1_25_ODEN           : 1;
  __REG32  IO1_26_ODEN           : 1;
  __REG32  IO1_27_ODEN           : 1;
  __REG32  IO1_28_ODEN           : 1;
  __REG32  IO1_29_ODEN           : 1;
  __REG32  IO1_30_ODEN           : 1;
  __REG32  IO1_31_ODEN           : 1;
} __ioconf_odcr1_bits;

/* IO P2 Mode Low Register */
typedef struct {
  __REG32  IO2_0_FSEL            : 2;
  __REG32  IO2_1_FSEL            : 2;
  __REG32  IO2_2_FSEL            : 2;
  __REG32  IO2_3_FSEL            : 2;
  __REG32  IO2_4_FSEL            : 2;
  __REG32  IO2_5_FSEL            : 2;
  __REG32  IO2_6_FSEL            : 2;
  __REG32  IO2_7_FSEL            : 2;
  __REG32  IO2_8_FSEL            : 2;
  __REG32  IO2_9_FSEL            : 2;
  __REG32  IO2_10_FSEL           : 2;
  __REG32  IO2_11_FSEL           : 2;
  __REG32  IO2_12_FSEL           : 2;
  __REG32  IO2_13_FSEL           : 2;
  __REG32  IO2_14_FSEL           : 2;
  __REG32  IO2_15_FSEL           : 2;
} __ioconf_mlr2_bits;

/* IO P2 Mode High Register */
typedef struct {
  __REG32  IO2_16_FSEL           : 2;
  __REG32  IO2_17_FSEL           : 2;
  __REG32  IO2_18_FSEL           : 2;
  __REG32  IO2_19_FSEL           : 2;
  __REG32  IO2_20_FSEL           : 2;
  __REG32  IO2_21_FSEL           : 2;
  __REG32  IO2_22_FSEL           : 2;
  __REG32  IO2_23_FSEL           : 2;
  __REG32  IO2_24_FSEL           : 2;
  __REG32  IO2_25_FSEL           : 2;
  __REG32  IO2_26_FSEL           : 2;
  __REG32  IO2_27_FSEL           : 2;
  __REG32                        : 8;
} __ioconf_mhr2_bits;

/* IO P2 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO2_0_PUEN            : 1;
  __REG32  IO2_1_PUEN            : 1;
  __REG32  IO2_2_PUEN            : 1;
  __REG32  IO2_3_PUEN            : 1;
  __REG32  IO2_4_PUEN            : 1;
  __REG32  IO2_5_PUEN            : 1;
  __REG32  IO2_6_PUEN            : 1;
  __REG32  IO2_7_PUEN            : 1;
  __REG32  IO2_8_PUEN            : 1;
  __REG32  IO2_9_PUEN            : 1;
  __REG32  IO2_10_PUEN           : 1;
  __REG32  IO2_11_PUEN           : 1;
  __REG32  IO2_12_PUEN           : 1;
  __REG32  IO2_13_PUEN           : 1;
  __REG32  IO2_14_PUEN           : 1;
  __REG32  IO2_15_PUEN           : 1;
  __REG32  IO2_16_PUEN           : 1;
  __REG32  IO2_17_PUEN           : 1;
  __REG32  IO2_18_PUEN           : 1;
  __REG32  IO2_19_PUEN           : 1;
  __REG32  IO2_20_PUEN           : 1;
  __REG32  IO2_21_PUEN           : 1;
  __REG32  IO2_22_PUEN           : 1;
  __REG32  IO2_23_PUEN           : 1;
  __REG32  IO2_24_PUEN           : 1;
  __REG32  IO2_25_PUEN           : 1;
  __REG32  IO2_26_PUEN           : 1;
  __REG32  IO2_27_PUEN           : 1;
  __REG32                        : 4;
} __ioconf_pucr2_bits;

/* IO P2 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO2_0_ODEN            : 1;
  __REG32  IO2_1_ODEN            : 1;
  __REG32  IO2_2_ODEN            : 1;
  __REG32  IO2_3_ODEN            : 1;
  __REG32  IO2_4_ODEN            : 1;
  __REG32  IO2_5_ODEN            : 1;
  __REG32  IO2_6_ODEN            : 1;
  __REG32  IO2_7_ODEN            : 1;
  __REG32  IO2_8_ODEN            : 1;
  __REG32  IO2_9_ODEN            : 1;
  __REG32  IO2_10_ODEN           : 1;
  __REG32  IO2_11_ODEN           : 1;
  __REG32  IO2_12_ODEN           : 1;
  __REG32  IO2_13_ODEN           : 1;
  __REG32  IO2_14_ODEN           : 1;
  __REG32  IO2_15_ODEN           : 1;
  __REG32  IO2_16_ODEN           : 1;
  __REG32  IO2_17_ODEN           : 1;
  __REG32  IO2_18_ODEN           : 1;
  __REG32  IO2_19_ODEN           : 1;
  __REG32  IO2_20_ODEN           : 1;
  __REG32  IO2_21_ODEN           : 1;
  __REG32  IO2_22_ODEN           : 1;
  __REG32  IO2_23_ODEN           : 1;
  __REG32  IO2_24_ODEN           : 1;
  __REG32  IO2_25_ODEN           : 1;
  __REG32  IO2_26_ODEN           : 1;
  __REG32  IO2_27_ODEN           : 1;
  __REG32                        : 4;
} __ioconf_odcr2_bits;

/* IO P3 Mode Low Register */
typedef struct {
  __REG32  IO3_0_FSEL            : 2;
  __REG32  IO3_1_FSEL            : 2;
  __REG32  IO3_2_FSEL            : 2;
  __REG32  IO3_3_FSEL            : 2;
  __REG32  IO3_4_FSEL            : 2;
  __REG32  IO3_5_FSEL            : 2;
  __REG32  IO3_6_FSEL            : 2;
  __REG32  IO3_7_FSEL            : 2;
  __REG32  IO3_8_FSEL            : 2;
  __REG32  IO3_9_FSEL            : 2;
  __REG32                        :12;
} __ioconf_mlr3_bits;

/* IO P3 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO3_0_PUEN            : 1;
  __REG32  IO3_1_PUEN            : 1;
  __REG32  IO3_2_PUEN            : 1;
  __REG32  IO3_3_PUEN            : 1;
  __REG32  IO3_4_PUEN            : 1;
  __REG32  IO3_5_PUEN            : 1;
  __REG32  IO3_6_PUEN            : 1;
  __REG32  IO3_7_PUEN            : 1;
  __REG32  IO3_8_PUEN            : 1;
  __REG32  IO3_9_PUEN            : 1;
  __REG32                        :22;
} __ioconf_pucr3_bits;

/* IO P3 Pull-Up Configuration Register */
typedef struct {
  __REG32  IO3_0_ODEN            : 1;
  __REG32  IO3_1_ODEN            : 1;
  __REG32  IO3_2_ODEN            : 1;
  __REG32  IO3_3_ODEN            : 1;
  __REG32  IO3_4_ODEN            : 1;
  __REG32  IO3_5_ODEN            : 1;
  __REG32  IO3_6_ODEN            : 1;
  __REG32  IO3_7_ODEN            : 1;
  __REG32  IO3_8_ODEN            : 1;
  __REG32  IO3_9_ODEN            : 1;
  __REG32                        :22;
} __ioconf_odcr3_bits;

/* LCD ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __lcd_idr_bits;

/* LCD Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :31;
} __lcd_cedr_bits;

/* LCD Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __lcd_srr_bits;

/* LCD Control Register */
typedef struct {
  __REG32  LCDEN          : 1;
  __REG32  DISC           : 2;
  __REG32                 : 1;
  __REG32  BTSEL          : 1;
  __REG32                 : 3;
  __REG32  DBSEL          : 3;
  __REG32                 : 5;
  __REG32  CONTRASTEN     : 1;
  __REG32  DIMINISHEN     : 1;
  __REG32                 : 2;
  __REG32  CONTRASTLEVEL  : 4;
  __REG32                 : 8;
} __lcd_cr_bits;

/* LCD Clock Divide Register */
typedef struct {
  __REG32  CDIV           : 3;
  __REG32                 : 4;
  __REG32  CDC            : 1;
  __REG32  CPRE           :16;
  __REG32                 : 8;
} __lcd_cdr_bits;

/* OPAMP ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __opa_idr_bits;

/* OPAMP Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :31;
} __opa_cedr_bits;

/* OPAMP Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __opa_srr_bits;

/* OPAMP Control Register */
typedef struct {
  __REG32  OPA0           : 1;
  __REG32  OPA1           : 1;
  __REG32  OPA2           : 1;
  __REG32                 : 5;
  __REG32  OPAM0          : 1;
  __REG32  OPAM1          : 1;
  __REG32  OPAM2          : 1;
  __REG32                 :21;
} __opa_cr_bits;

/* OPAMP Gain Control Register */
typedef struct {
  __REG32  GV0            : 4;
  __REG32                 : 3;
  __REG32  GCT0           : 1;
  __REG32  GV1            : 4;
  __REG32                 : 3;
  __REG32  GCT1           : 1;
  __REG32  GV2            : 4;
  __REG32                 : 3;
  __REG32  GCT2           : 1;
  __REG32                 : 8;
} __opa_gcr_bits;

/* OPAMP Interrupt Mask Set/Clear Register */
/* OPAMP Raw Interrupt Status Register */
/* OPAMP Masked Interrupt Status Register */
/* OPAMP Interrupt Clear Register */
typedef struct {
  __REG32  OPA0           : 1;
  __REG32  OPA1           : 1;
  __REG32  OPA2           : 1;
  __REG32                 : 5;
  __REG32  OPAM0          : 1;
  __REG32  OPAM1          : 1;
  __REG32  OPAM2          : 1;
  __REG32                 :21;
} __opa_imscr_bits;

/* Flash ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __pf_idr_bits;

/* Flash Software Reset Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :31;
} __pf_cedr_bits;

/* Flash Clock Enable/Disable Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __pf_srr_bits;

/* Flash Control Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 3;
  __REG32  CMD                   : 3;
  __REG32                        :25;
} __pf_cr_bits;

/* Flash Mode Register */
typedef struct {
  __REG32  BACEN                 : 1;
  __REG32                        : 3;
  __REG32  WAIT                  : 2;
  __REG32                        : 2;
  __REG32  PBUFEN                : 1;
  __REG32                        :23;
} __pf_mr_bits;

/* Flash Interrupt Mask Set and Clear Register */
/* Flash Raw Interrupt Status Register */
/* Flash Masked Interrupt Status Register */
/* Flash Interrupt Clear Register */
typedef struct {
  __REG32  END                   : 1;
  __REG32                        : 7;
  __REG32  ERR0                  : 1;
  __REG32  ERR1                  : 1;
  __REG32  ERR2                  : 1;
  __REG32                        :21;
} __pf_imscr_bits;

/* Flash Status Register */
typedef struct {
  __REG32  BUSY                  : 1;
  __REG32                        :31;
} __pf_sr_bits;

/* Smart Option Protection Status Register */
typedef struct {
  __REG32                        : 4;
  __REG32  HWPA0                 : 1;
  __REG32  HWPA1                 : 1;
  __REG32  HWPA2                 : 1;
  __REG32  HWPA3                 : 1;
  __REG32  nJTAGP                : 1;
  __REG32                        : 3;
  __REG32  HWPA4                 : 1;
  __REG32  HWPA5                 : 1;
  __REG32  HWPA6                 : 1;
  __REG32  HWPA7                 : 1;
  __REG32                        : 1;
  __REG32  HWP                   : 1;
  __REG32                        : 2;
  __REG32  HWPA8                 : 1;
  __REG32  HWPA9                 : 1;
  __REG32  HWPA10                : 1;
  __REG32  HWPA11                : 1;
  __REG32                        : 3;
  __REG32  nSRP                  : 1;
  __REG32                        : 4;
} __so_psr_bits;

/* Smart Option Configuration Status Register */
typedef struct {
  __REG32  POCCS                 : 2;
  __REG32                        : 4;
  __REG32  IMSEL                 : 2;
  __REG32                        : 4;
  __REG32  BTDIV                 : 4;
  __REG32                        :16;
} __so_csr_bits;

/* Internal OSC Trimming Register */
typedef struct {
  __REG32  OSC0                  : 7;
  __REG32                        : 1;
  __REG32  OSC1                  : 7;
  __REG32                        : 1;
  __REG32  OSC2                  : 6;
  __REG32                        : 2;
  __REG32  IOTKEY                : 8;
} __pf_iotr_bits;

/* PWM ID Register */
typedef struct {
  __REG32  IDCODE         :26;
  __REG32                 : 6;
} __pwm_idr_bits;

/* PWM Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN          : 1;
  __REG32                 :30;
  __REG32  DBGEN          : 1;
} __pwm_cedr_bits;

/* PWM Software Reset Register */
typedef struct {
  __REG32  SWRST          : 1;
  __REG32                 :31;
} __pwm_srr_bits;

/* PWM Control Set Register */
typedef struct {
  __REG32  START          : 1;
  __REG32  UPDATE         : 1;
  __REG32                 : 6;
  __REG32  IDLESL         : 1;
  __REG32  OUTSL          : 1;
  __REG32  KEEP           : 1;
  __REG32  PWMIM          : 1;
  __REG32                 :12;
  __REG32  PWMEX0         : 1;
  __REG32  PWMEX1         : 1;
  __REG32  PWMEX2         : 1;
  __REG32  PWMEX3         : 1;
  __REG32  PWMEX4         : 1;
  __REG32  PWMEX5         : 1;
  __REG32                 : 2;
} __pwm_csr_bits;

/* PWM Control Clear Register */
/* PWM Status Register */
typedef struct {
  __REG32  START          : 1;
  __REG32                 : 7;
  __REG32  IDLESL         : 1;
  __REG32  OUTSL          : 1;
  __REG32  KEEP           : 1;
  __REG32  PWMIM          : 1;
  __REG32                 :12;
  __REG32  PWMEX0         : 1;
  __REG32  PWMEX1         : 1;
  __REG32  PWMEX2         : 1;
  __REG32  PWMEX3         : 1;
  __REG32  PWMEX4         : 1;
  __REG32  PWMEX5         : 1;
  __REG32                 : 2;
} __pwm_ccr_bits;

/* PWM Interrupt Mask Set/Clear Register */
/* PWM Raw Interrupt Status Register */
/* PWM Masked Interrupt Status Register */
/* PWM Interrupt Clear Register */
typedef struct {
  __REG32  PWMSTART       : 1;
  __REG32  PWMSTOP        : 1;
  __REG32  PSTART         : 1;
  __REG32  PEND           : 1;
  __REG32  PMATCH         : 1;
  __REG32                 :27;
} __pwm_imscr_bits;

/* PWM Clock Divider Register */
typedef struct {
  __REG32  DIVN           : 4;
  __REG32  DIVM           :11;
  __REG32                 :17;
} __pwm_cdr_bits;

/* PWM Period Register */
/* PWM Current Period Register */
typedef struct {
  __REG32  PERIOD         :16;
  __REG32                 :16;
} __pwm_prdr_bits;

/* PWM Pulse Register */
/* PWM Current Pulse Register */
typedef struct {
  __REG32  PULSE          :16;
  __REG32                 :16;
} __pwm_pulr_bits;

/* PWM Current Clock Divider Register */
/* PWM Current Clock Divider Register */
typedef struct {
  __REG32  DIVN           : 4;
  __REG32  DIVM           :11;
  __REG32                 :17;
} __pwm_ccdr_bits;

/* SPIx Control Register 0 */
typedef struct {
  __REG32  DSS                   : 4;
  __REG32  FRF                   : 2;
  __REG32  SPO                   : 1;
  __REG32  SPH                   : 1;
  __REG32  SCR                   : 8;
  __REG32                        :16;
} __spi_cr0_bits;

/* SPIx Control Register 1 */
typedef struct {
  __REG32  LBM                   : 1;
  __REG32  SSE                   : 1;
  __REG32  MS                    : 1;
  __REG32  SOD                   : 1;
  __REG32  RXIFLSEL              : 3;
  __REG32                        :25;
} __spi_cr1_bits;

/* SPIx status register */
typedef struct {
  __REG32  TFE                   : 1;
  __REG32  TNF                   : 1;
  __REG32  RNE                   : 1;
  __REG32  RFF                   : 1;
  __REG32  BSY                   : 1;
  __REG32                        :27;
} __spi_sr_bits;

/* SPIx Clock prescaler register */
typedef struct {
  __REG32  CPSDVSR               : 8;
  __REG32                        :24;
} __spi_cpsr_bits;

/* SPIx interrupt mask set or clear register */
typedef struct {
  __REG32  RORIM                 : 1;
  __REG32  RTIM                  : 1;
  __REG32  RXIM                  : 1;
  __REG32  TXIM                  : 1;
  __REG32                        :28;
} __spi_imsc_bits;

/* SPIx raw interrupt status register */
typedef struct {
  __REG32  RORRIS                : 1;
  __REG32  RTRIS                 : 1;
  __REG32  RXRIS                 : 1;
  __REG32  TXRIS                 : 1;
  __REG32                        :28;
} __spi_ris_bits;

/* SPIx interrupt priority register */
typedef struct {
  __REG32  RORMIS                : 1;
  __REG32  RTMIS                 : 1;
  __REG32  RXMIS                 : 1;
  __REG32  TXMIS                 : 1;
  __REG32                        :28;
} __spi_misr_bits;

/* SPIx interrupt clear register */
typedef struct {
  __REG32  RORIC                 : 1;
  __REG32  RTIC                  : 1;
  __REG32                        :30;
} __spi_icr_bits;

/* SPIx DMA control register */
typedef struct {
  __REG32  RXDMAE                : 1;
  __REG32  TXDMAE                : 1;
  __REG32                        :30;
} __spi_dmacr_bits;

/* STT ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __stt_idr_bits;

/* STT ID Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __stt_cedr_bits;

/* STT Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __stt_srr_bits;

/* STT Control Register */
typedef struct {
  __REG32                        : 1;
  __REG32  CNTEN                 : 1;
  __REG32  CNTDIS                : 1;
  __REG32  ALARMEN               : 1;
  __REG32  ALARMDIS              : 1;
  __REG32                        :27;
} __stt_cr_bits;

/* STT Mode Register */
typedef struct {
  __REG32  CNTRST                : 1;
  __REG32                        :31;
} __stt_mr_bits;

/* STT Status Register */
typedef struct {
  __REG32                        : 5;
  __REG32  WSEC                  : 1;
  __REG32                        : 2;
  __REG32  CNTENS                : 1;
  __REG32  ALARMENS              : 1;
  __REG32                        :22;
} __stt_sr_bits;

/* STT Interrupt Mask Set/Clear Register */
/* STT Raw Interrupt Status Register */
/* STT Masked Interrupt Status Register */
/* STT Interrupt Clear Register */
typedef struct {
  __REG32  ALARM                 : 1;
  __REG32  CNTEN                 : 1;
  __REG32  CNTDIS                : 1;
  __REG32  ALARMEN               : 1;
  __REG32  ALARMDIS              : 1;
  __REG32                        :27;
} __stt_imscr_bits;

/* Timer/Counter ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __tc_idr_bits;

/* Timer/Counter Clock Source Selection Register */
typedef struct {
  __REG32  CLKSRC                : 1;
  __REG32                        :31;
} __tc_cssr_bits;

/* Timer/Counter Clock Enable/Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __tc_cedr_bits;

/* Timer/Counter Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __tc_srr_bits;

/* Timer/Counter Control Set Register */
/* Timer/Counter Control Clear Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  UPDATE                : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32  ADCTRG                : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc_csr_bits;

/* Timer/Counter Status Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32  ADCTRG                : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc_sr_bits;

/* Timer/Counter Interrupt Mask Set/Clear Register */
/* Timer/Counter Raw Interrupt Status Register */
/* Timer/Counter Masked Interrupt Status Register */
/* Timer/Counter Interrupt Clear Register */
typedef struct {
  __REG32  STARTI                : 1;
  __REG32  STOPI                 : 1;
  __REG32  PSTARTI               : 1;
  __REG32  PENDI                 : 1;
  __REG32  MATI                  : 1;
  __REG32  OVFI                  : 1;
  __REG32  CAPTI                 : 1;
  __REG32                        :25;
} __tc_imscr_bits;

/* Timer/Counter Clock Divider Register */
/* Timer/Counter Current Clock Divider Register */
typedef struct {
  __REG32  DIVN                  : 4;
  __REG32  DIVM                  :11;
  __REG32                        :17;
} __tc_cdr_bits;

/* Timer/Counter Counter Size Mask Register */
/* Timer/Counter Current Counter Size Mask Register */
typedef struct {
  __REG32  SIZE                  : 4;
  __REG32                        :28;
} __tc_csmr_bits;

/* Timer/Counter Control Set Register */
/* Timer/Counter Control Clear Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32  UPDATE                : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32                        : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc32_csr_bits;

/* Timer/Counter Status Register */
typedef struct {
  __REG32  START                 : 1;
  __REG32                        : 1;
  __REG32  STOPHOLD              : 1;
  __REG32  STOPCLEAR             : 1;
  __REG32                        : 4;
  __REG32  IDLESL                : 1;
  __REG32  OUTSL                 : 1;
  __REG32  KEEP                  : 1;
  __REG32  PWMIM                 : 1;
  __REG32  PWMEN                 : 1;
  __REG32  REPEAT                : 1;
  __REG32  OVFM                  : 1;
  __REG32                        : 1;
  __REG32  CAPTEN                : 1;
  __REG32  CAPT_F                : 1;
  __REG32  CAPT_R                : 1;
  __REG32                        : 5;
  __REG32  PWMEX0                : 1;
  __REG32  PWMEX1                : 1;
  __REG32  PWMEX2                : 1;
  __REG32  PWMEX3                : 1;
  __REG32  PWMEX4                : 1;
  __REG32  PWMEX5                : 1;
  __REG32                        : 2;
} __tc32_sr_bits;

/* Timer/Counter Interrupt Mask Set/Clear Register */
/* Timer/Counter Raw Interrupt Status Register */
/* Timer/Counter Masked Interrupt Status Register */
/* Timer/Counter Interrupt Clear Register */
typedef struct {
  __REG32  STARTI                : 1;
  __REG32  STOPI                 : 1;
  __REG32  PSTARTI               : 1;
  __REG32  PENDI                 : 1;
  __REG32  MATCHI                : 1;
  __REG32  OVFI                  : 1;
  __REG32  CAPTI                 : 1;
  __REG32                        :25;
} __tc32_imscr_bits;

/* Timer/Counter Counter Size Mask Register */
typedef struct {
  __REG32  SIZE                  : 5;
  __REG32                        :27;
} __tc32_csmr_bits;

/* Timer/Counter Clock Divider Register */
typedef struct {
  __REG32  DIVN                  : 4;
  __REG32  DIVM                  :28;
} __tc32_cdr_bits;

/* USART ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __us_idr_bits;

/* USART Clock Enable Disable Register */
typedef struct {
  __REG32  CLKEN                 : 1;
  __REG32                        :30;
  __REG32  DBGEN                 : 1;
} __us_cedr_bits;

/* USART Software Reset Register */
typedef struct {
  __REG32  SWRST                 : 1;
  __REG32                        :31;
} __us_srr_bits;

/* USART Control Register */
typedef struct {
  __REG32                        : 2;
  __REG32  RSTRX                 : 1;
  __REG32  RSTTX                 : 1;
  __REG32  RXEN                  : 1;
  __REG32  RXDIS                 : 1;
  __REG32  TXEN                  : 1;
  __REG32  TXDIS                 : 1;
  __REG32                        : 1;
  __REG32  STTBRK                : 1;
  __REG32  STPBRK                : 1;
  __REG32  STTTO                 : 1;
  __REG32  SENDA                 : 1;
  __REG32                        : 3;
  __REG32  STHEADER              : 1;
  __REG32  STREPS                : 1;
  __REG32  STMESSAGE             : 1;
  __REG32  RSTLIN                : 1;
  __REG32                        :12;
} __us_cr_bits;

/* USART Mode Register */
typedef struct {
  __REG32  LIN                   : 1;
  __REG32  SENDTIME              : 3;
  __REG32  CLKS                  : 2;
  __REG32  CHRL                  : 2;
  __REG32  SYNC                  : 1;
  __REG32  PAR                   : 3;
  __REG32  NBSTOP                : 2;
  __REG32  CHMODE                : 2;
  __REG32  SMCARDPT              : 1;
  __REG32  MODE9                 : 1;
  __REG32  CLKO                  : 1;
  __REG32  LIN2_0                : 1;
  __REG32  DSB                   : 1;
  __REG32                        :11;
} __us_mr_bits;

/* USART Interrupt Mask Set and Clear Register */
/* USART Raw Interrupt Status Register */
/* USART Masked Interrupt Status Register */
typedef struct {
  __REG32  RXRDY                 : 1;
  __REG32  TXRDY                 : 1;
  __REG32  RXBRK                 : 1;
  __REG32                        : 2;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32                        :13;
  __REG32  ENDHEADER             : 1;
  __REG32  ENDMESS               : 1;
  __REG32  NOTRESP               : 1;
  __REG32  BITERROR              : 1;
  __REG32  IPERROR               : 1;
  __REG32  CHECKSUM              : 1;
  __REG32  WAKEUP                : 1;
  __REG32                        : 1;
} __us_imscr_bits;

/* USART Interrupt Clear Register */
typedef struct {
  __REG32                        : 2;
  __REG32  RXBRK                 : 1;
  __REG32                        : 2;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32                        :13;
  __REG32  ENDHEADER             : 1;
  __REG32  ENDMESS               : 1;
  __REG32  NOTRESP               : 1;
  __REG32  BITERROR              : 1;
  __REG32  IPERROR               : 1;
  __REG32  CHECKSUM              : 1;
  __REG32  WAKEUP                : 1;
  __REG32                        : 1;
} __us_icr_bits;

/* USART Status Register */
typedef struct {
  __REG32  RXRDY                 : 1;
  __REG32  TXRDY                 : 1;
  __REG32  RXBRK                 : 1;
  __REG32                        : 2;
  __REG32  OVRE                  : 1;
  __REG32  FRAME                 : 1;
  __REG32  PARE                  : 1;
  __REG32  TIMEOUT               : 1;
  __REG32  TXEMPTY               : 1;
  __REG32  IDLE                  : 1;
  __REG32  IDLEFLAG              : 1;
  __REG32                        :12;
  __REG32  ENDHEADER             : 1;
  __REG32  ENDMESS               : 1;
  __REG32  NOTRESP               : 1;
  __REG32  BITERROR              : 1;
  __REG32  IPERROR               : 1;
  __REG32  CHECKSUM              : 1;
  __REG32  WAKEUP                : 1;
  __REG32  LINBUSY               : 1;
} __us_sr_bits;

/* USART Receiver Holding Register */
typedef struct {
  __REG32  RXCHR                 : 9;
  __REG32                        :23;
} __us_rhr_bits;

/* USART Transmit Holding Register */
typedef struct {
  __REG32  TXCHR                 : 9;
  __REG32                        :23;
} __us_thr_bits;

/* USART Baud Rate Generator Register */
typedef struct {
  __REG32  CD                    :16;
  __REG32                        :16;
} __us_brgr_bits;

/* USART Receiver Time-Out Register */
typedef struct {
  __REG32  TO                    :16;
  __REG32                        :16;
} __us_rtor_bits;

/* USART Transmit Time-Guard Register */
typedef struct {
  __REG32  TG                    : 8;
  __REG32                        :24;
} __us_ttgr_bits;

/* USART LIN Identifier Register */
typedef struct {
  __REG32  IDENTIFIER            : 6;
  __REG32  NDATA                 : 3;
  __REG32  CHK_SEL               : 1;
  __REG32                        : 6;
  __REG32  WAKE_UP_TIME          :14;
  __REG32                        : 2;
} __us_lir_bits;

/* USART Data Field Write 0 Register */
/* USART Data Field Read 0 Register */
typedef struct {
  __REG32  DATA0                 : 8;
  __REG32  DATA1                 : 8;
  __REG32  DATA2                 : 8;
  __REG32  DATA3                 : 8;
} __us_dfwr0_bits;

/* USART Data Field Write 1 Register */
/* USART Data Field Read 1 Register */
typedef struct {
  __REG32  DATA4                 : 8;
  __REG32  DATA5                 : 8;
  __REG32  DATA6                 : 8;
  __REG32  DATA7                 : 8;
} __us_dfwr1_bits;

/* USART Synchronous Break Length Register */
typedef struct {
  __REG32  SYNC_BRK              : 5;
  __REG32                        :27;
} __us_sblr_bits;

/* USART Synchronous Break Length Register 1 */
typedef struct {
  __REG32  LCP0                  : 8;
  __REG32  LCP1                  : 8;
  __REG32  LCP2                  : 8;
  __REG32  LCP3                  : 8;
} __us_lcp1_bits;

/* USART Synchronous Break Length Register 2 */
typedef struct {
  __REG32  LCP4                  : 8;
  __REG32  LCP5                  : 8;
  __REG32  LCP6                  : 8;
  __REG32  LCP7                  : 8;
} __us_lcp2_bits;

/* USART DMA Control Register */
typedef struct {
  __REG32  RXDMAE                : 1;
  __REG32  TXDMAE                : 1;
  __REG32                        :30;
} __us_dmacr_bits;

/* WDT ID Register */
typedef struct {
  __REG32  IDCODE                :26;
  __REG32                        : 6;
} __wdt_idr_bits;

/* WDT Control Register */
typedef struct {
  __REG32  RSTKEY                :16;
  __REG32                        :15;
  __REG32  DBGEN                 : 1;
} __wdt_cr_bits;

/* WDT Mode Register */
typedef struct {
  __REG32  WDTPDIV               : 3;
  __REG32                        : 5;
  __REG32  PCV                   :16;
  __REG32  CKEY                  : 8;
} __wdt_mr_bits;

/* WDT Overflow Mode Register */
typedef struct {
  __REG32  WDTEN                 : 1;
  __REG32  RSTEN                 : 1;
  __REG32  LOCKRSTEN             : 1;
  __REG32                        : 1;
  __REG32  OKEY                  :12;
  __REG32                        :16;
} __wdt_omr_bits;

/* WDT Status Register */
typedef struct {
  __REG32                        : 8;
  __REG32  PENDING               : 1;
  __REG32  CLEAR_STATUS          : 1;
  __REG32                        :21;
  __REG32  DBGEN                 : 1;
} __wdt_sr_bits;

/* WDT Interrupt Mask Set and Clear Register */
/* WDT Interrupt Raw Interrupt Status Register */
/* WDT Interrupt Masked Interrupt Status Register */
/* WDT Interrupt Clear Register */
typedef struct {
  __REG32  WDTPEND               : 1;
  __REG32  WDTOVF                : 1;
  __REG32                        :30;
} __wdt_imscr_bits;

/* WDT Pending Windows Register */
typedef struct {
  __REG32  RSTALW                : 1;
  __REG32                        : 7;
  __REG32  PWL                   :16;
  __REG32  PWKEY                 : 8;
} __wdt_pwr_bits;

/* WDT Counter Test Register */
typedef struct {
  __REG32  COUNT                 :16;
  __REG32                        :16;
} __wdt_ctr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                  0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,          0xE000E01C,__READ       ,__systickcalvr_bits);
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
 **  ADC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC0_IDR,        0x40040000,__READ       ,__adc_idr_bits);
__IO_REG32_BIT(ADC0_CEDR,       0x40040004,__READ_WRITE ,__adc_cedr_bits);
__IO_REG32_BIT(ADC0_SRR,        0x40040008,__WRITE      ,__adc_srr_bits);
__IO_REG32_BIT(ADC0_CSR,        0x4004000C,__WRITE      ,__adc_csr_bits);
__IO_REG32_BIT(ADC0_CCR,        0x40040010,__WRITE      ,__adc_ccr_bits);
__IO_REG32_BIT(ADC0_CDR,        0x40040014,__READ_WRITE ,__adc_cdr_bits);
__IO_REG32_BIT(ADC0_MR,         0x40040018,__READ_WRITE ,__adc_mr_bits);
__IO_REG32_BIT(ADC0_CCSR0,      0x4004001C,__READ       ,__adc_ccsr_bits);
__IO_REG32_BIT(ADC0_CCSR1,      0x40040020,__READ       ,__adc_ccsr_bits);
__IO_REG32_BIT(ADC0_SR,         0x40040024,__READ       ,__adc_sr_bits);
__IO_REG32_BIT(ADC0_IMSCR,      0x40040028,__READ_WRITE ,__adc_imscr_bits);
__IO_REG32_BIT(ADC0_RISR,       0x4004002C,__READ       ,__adc_imscr_bits);
__IO_REG32_BIT(ADC0_MISR,       0x40040030,__READ       ,__adc_imscr_bits);
__IO_REG32_BIT(ADC0_ICR,        0x40040034,__WRITE      ,__adc_imscr_bits);
__IO_REG32_BIT(ADC0_CRR0,       0x40040038,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC0_CRR1,       0x4004003C,__READ       ,__adc_crr_bits);
__IO_REG32_BIT(ADC0_GCR0,       0x40040040,__READ_WRITE ,__adc_gcr_bits);
__IO_REG32_BIT(ADC0_OCR0,       0x40040044,__READ_WRITE ,__adc_ocr_bits);
__IO_REG32_BIT(ADC0_GCR1,       0x40040048,__READ_WRITE ,__adc_gcr_bits);
__IO_REG32_BIT(ADC0_OCR1,       0x4004004C,__READ_WRITE ,__adc_ocr_bits);
__IO_REG32_BIT(ADC0_DMACR,      0x40040050,__READ_WRITE ,__adc_dmacr_bits);

/***************************************************************************
 **
 **  ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1_IDR,        0x40041000,__READ       ,__adc_idr_bits);
__IO_REG32_BIT(ADC1_CEDR,       0x40041004,__READ_WRITE ,__adc_cedr_bits);
__IO_REG32_BIT(ADC1_SRR,        0x40041008,__WRITE      ,__adc_srr_bits);
__IO_REG32_BIT(ADC1_CSR,        0x4004100C,__WRITE      ,__adc1_csr_bits);
__IO_REG32_BIT(ADC1_CCR,        0x40041010,__WRITE      ,__adc1_ccr_bits);
__IO_REG32_BIT(ADC1_CDR,        0x40041014,__READ_WRITE ,__adc_cdr_bits);
__IO_REG32_BIT(ADC1_MR,         0x40041018,__READ_WRITE ,__adc1_mr_bits);
__IO_REG32_BIT(ADC1_CCSR,       0x4004101C,__READ       ,__adc1_ccsr_bits);
__IO_REG32_BIT(ADC1_SR,         0x40041020,__READ       ,__adc1_sr_bits);
__IO_REG32_BIT(ADC1_IMSCR,      0x40041024,__READ_WRITE ,__adc1_imscr_bits);
__IO_REG32_BIT(ADC1_RISR,       0x40041028,__READ       ,__adc1_imscr_bits);
__IO_REG32_BIT(ADC1_MISR,       0x4004102C,__READ       ,__adc1_imscr_bits);
__IO_REG32_BIT(ADC1_ICR,        0x40041030,__WRITE      ,__adc1_imscr_bits);
__IO_REG32_BIT(ADC1_CRR,        0x40041034,__READ       ,__adc1_crr_bits);
__IO_REG32_BIT(ADC1_GCR,        0x40041038,__READ_WRITE ,__adc_gcr_bits);
__IO_REG32_BIT(ADC1_OCR,        0x4004103C,__READ_WRITE ,__adc1_ocr_bits);
__IO_REG32_BIT(ADC1_DMACR,      0x40041040,__READ_WRITE ,__adc1_dmacr_bits);

/***************************************************************************
 **
 **  CAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0_ECR,        0x400E0050,__WRITE      ,__can_ecr_bits);
__IO_REG32_BIT(CAN0_DCR,        0x400E0054,__WRITE      ,__can_ecr_bits);
__IO_REG32_BIT(CAN0_PMSR,       0x400E0058,__READ       ,__can_ecr_bits);
__IO_REG32_BIT(CAN0_CR,         0x400E0060,__WRITE      ,__can_cr_bits);
__IO_REG32_BIT(CAN0_MR,         0x400E0064,__READ_WRITE ,__can_mr_bits);
__IO_REG32_BIT(CAN0_CSR,        0x400E006C,__WRITE      ,__can_csr_bits);
__IO_REG32_BIT(CAN0_SR,         0x400E0070,__READ       ,__can_sr_bits);
__IO_REG32_BIT(CAN0_IER,        0x400E0074,__WRITE      ,__can_ier_bits);
__IO_REG32_BIT(CAN0_IDR,        0x400E0078,__WRITE      ,__can_ier_bits);
__IO_REG32_BIT(CAN0_IMR,        0x400E007C,__READ       ,__can_ier_bits);
__IO_REG32_BIT(CAN0_ISSR,       0x400E0084,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_SIER,       0x400E0088,__WRITE      ,__can_issr_bits);
__IO_REG32_BIT(CAN0_SIDR,       0x400E008C,__WRITE      ,__can_issr_bits);
__IO_REG32_BIT(CAN0_SIMR,       0x400E0090,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_HPIR,       0x400E0094,__READ       ,__can_hpir_bits);
__IO_REG32_BIT(CAN0_ERCR,       0x400E0098,__READ       ,__can_ercr_bits);
__IO_REG32_BIT(CAN0_TMR0,       0x400E0100,__READ_WRITE ,__can_tmr_bits);
__IO_REG32_BIT(CAN0_DAR0,       0x400E0104,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_DBR0,       0x400E0108,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_MSKR0,      0x400E010C,__READ_WRITE ,__can_mskr_bits);
__IO_REG32_BIT(CAN0_IR0,        0x400E0110,__READ_WRITE ,__can_ir_bits);
__IO_REG32_BIT(CAN0_MCR0,       0x400E0114,__READ_WRITE ,__can_mcr_bits);
__IO_REG32(    CAN0_STPR0,      0x400E0118,__READ       );
__IO_REG32_BIT(CAN0_TMR1,       0x400E0120,__READ_WRITE ,__can_tmr_bits);
__IO_REG32_BIT(CAN0_DAR1,       0x400E0124,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_DBR1,       0x400E0128,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN0_MSKR1,      0x400E012C,__READ_WRITE ,__can_mskr_bits);
__IO_REG32_BIT(CAN0_IR1,        0x400E0130,__READ_WRITE ,__can_ir_bits);
__IO_REG32_BIT(CAN0_MCR1,       0x400E0134,__READ_WRITE ,__can_mcr_bits);
__IO_REG32(    CAN0_STPR1,      0x400E0138,__READ       );
__IO_REG32_BIT(CAN0_TRR,        0x400E0140,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_NDR,        0x400E0144,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_MVR,        0x400E0148,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN0_TSTR,       0x400E014C,__READ_WRITE ,__can_tstr_bits);

/***************************************************************************
 **
 **  CAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1_ECR,        0x400E1050,__WRITE      ,__can_ecr_bits);
__IO_REG32_BIT(CAN1_DCR,        0x400E1054,__WRITE      ,__can_ecr_bits);
__IO_REG32_BIT(CAN1_PMSR,       0x400E1058,__READ       ,__can_ecr_bits);
__IO_REG32_BIT(CAN1_CR,         0x400E1060,__WRITE      ,__can_cr_bits);
__IO_REG32_BIT(CAN1_MR,         0x400E1064,__READ_WRITE ,__can_mr_bits);
__IO_REG32_BIT(CAN1_CSR,        0x400E106C,__WRITE      ,__can_csr_bits);
__IO_REG32_BIT(CAN1_SR,         0x400E1070,__READ       ,__can_sr_bits);
__IO_REG32_BIT(CAN1_IER,        0x400E1074,__WRITE      ,__can_ier_bits);
__IO_REG32_BIT(CAN1_IDR,        0x400E1078,__WRITE      ,__can_ier_bits);
__IO_REG32_BIT(CAN1_IMR,        0x400E107C,__READ       ,__can_ier_bits);
__IO_REG32_BIT(CAN1_ISSR,       0x400E1084,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN1_SIER,       0x400E1088,__WRITE      ,__can_issr_bits);
__IO_REG32_BIT(CAN1_SIDR,       0x400E108C,__WRITE      ,__can_issr_bits);
__IO_REG32_BIT(CAN1_SIMR,       0x400E1090,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN1_HPIR,       0x400E1094,__READ       ,__can_hpir_bits);
__IO_REG32_BIT(CAN1_ERCR,       0x400E1098,__READ       ,__can_ercr_bits);
__IO_REG32_BIT(CAN1_TMR0,       0x400E1100,__READ_WRITE ,__can_tmr_bits);
__IO_REG32_BIT(CAN1_DAR0,       0x400E1104,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN1_DBR0,       0x400E1108,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN1_MSKR0,      0x400E110C,__READ_WRITE ,__can_mskr_bits);
__IO_REG32_BIT(CAN1_IR0,        0x400E1110,__READ_WRITE ,__can_ir_bits);
__IO_REG32_BIT(CAN1_MCR0,       0x400E1114,__READ_WRITE ,__can_mcr_bits);
__IO_REG32(    CAN1_STPR0,      0x400E1118,__READ       );
__IO_REG32_BIT(CAN1_TMR1,       0x400E1120,__READ_WRITE ,__can_tmr_bits);
__IO_REG32_BIT(CAN1_DAR1,       0x400E1124,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN1_DBR1,       0x400E1128,__READ_WRITE ,__can_dar_bits);
__IO_REG32_BIT(CAN1_MSKR1,      0x400E112C,__READ_WRITE ,__can_mskr_bits);
__IO_REG32_BIT(CAN1_IR1,        0x400E1130,__READ_WRITE ,__can_ir_bits);
__IO_REG32_BIT(CAN1_MCR1,       0x400E1134,__READ_WRITE ,__can_mcr_bits);
__IO_REG32(    CAN1_STPR1,      0x400E1138,__READ       );
__IO_REG32_BIT(CAN1_TRR,        0x400E1140,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN1_NDR,        0x400E1144,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN1_MVR,        0x400E1148,__READ       ,__can_issr_bits);
__IO_REG32_BIT(CAN1_TSTR,       0x400E114C,__READ_WRITE ,__can_tstr_bits);

/***************************************************************************
 **
 **  System
 **
 ***************************************************************************/
__IO_REG32_BIT(CM_IDR,          0x40020000,__READ       ,__cm_idr_bits);
__IO_REG32_BIT(CM_SRR,          0x40020004,__WRITE      ,__cm_srr_bits);
__IO_REG32_BIT(CM_CSR,          0x40020008,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_CCR,          0x4002000C,__WRITE      ,__cm_csr_bits);
__IO_REG32_BIT(CM_PCSR0,        0x40020010,__WRITE      ,__cm_pcsr0_bits);
__IO_REG32_BIT(CM_PCSR1,        0x40020014,__WRITE      ,__cm_pcsr1_bits);
__IO_REG32_BIT(CM_PCCR0,        0x40020018,__WRITE      ,__cm_pcsr0_bits);
__IO_REG32_BIT(CM_PCCR1,        0x4002001C,__WRITE      ,__cm_pcsr1_bits);
__IO_REG32_BIT(CM_PCKSR0,       0x40020020,__READ       ,__cm_pcsr0_bits);
__IO_REG32_BIT(CM_PCKSR1,       0x40020024,__READ       ,__cm_pcsr1_bits);
__IO_REG32_BIT(CM_MR0,          0x40020028,__READ_WRITE ,__cm_mr0_bits);
__IO_REG32_BIT(CM_MR1,          0x4002002C,__READ_WRITE ,__cm_mr1_bits);
__IO_REG32_BIT(CM_IMSCR,        0x40020030,__WRITE      ,__cm_imscr_bits);
__IO_REG32_BIT(CM_RISR,         0x40020034,__READ       ,__cm_risr_bits);
__IO_REG32_BIT(CM_MISR,         0x40020038,__READ       ,__cm_imscr_bits);
__IO_REG32_BIT(CM_ICR,          0x4002003C,__WRITE      ,__cm_risr_bits);
__IO_REG32_BIT(CM_SR,           0x40020040,__READ_WRITE ,__cm_sr_bits);
__IO_REG32_BIT(CM_SCDR,         0x40020044,__READ_WRITE ,__cm_scdr_bits);
__IO_REG32_BIT(CM_PCDR,         0x40020048,__READ_WRITE ,__cm_pcdr_bits);
__IO_REG32_BIT(CM_FCDR,         0x4002004C,__READ_WRITE ,__cm_fcdr_bits);
__IO_REG32_BIT(CM_STCDR,        0x40020050,__READ_WRITE ,__cm_stcdr_bits);
__IO_REG32_BIT(CM_LCDR,         0x40020054,__READ_WRITE ,__cm_lcdr_bits);
__IO_REG32_BIT(CM_PSTR,         0x40020058,__READ_WRITE ,__cm_pstr_bits);
__IO_REG32_BIT(CM_PDPR,         0x4002005C,__READ_WRITE ,__cm_pdpr_bits);
__IO_REG32_BIT(CM_EMSTR,        0x40020068,__READ_WRITE ,__cm_emstr_bits);
__IO_REG32_BIT(CM_ESSTR,        0x4002006C,__READ_WRITE ,__cm_esstr_bits);
__IO_REG32_BIT(CM_BTCDR,        0x40020070,__READ_WRITE ,__cm_btcdr_bits);
__IO_REG32_BIT(CM_BTR,          0x40020074,__READ_WRITE ,__cm_btr_bits);
__IO_REG32_BIT(CM_WCR0,         0x40020078,__READ_WRITE ,__cm_wcr0_bits);
__IO_REG32_BIT(CM_WCR1,         0x4002007C,__READ_WRITE ,__cm_wcr1_bits);
__IO_REG32_BIT(CM_WCR2,         0x40020080,__READ_WRITE ,__cm_wcr2_bits);
__IO_REG32_BIT(CM_WCR3,         0x40020084,__READ_WRITE ,__cm_wcr3_bits);
__IO_REG32_BIT(CM_WIMSCR,       0x40020088,__READ_WRITE ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_WRISR,        0x4002008C,__READ       ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_WMISR,        0x40020090,__READ       ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_WICR,         0x40020094,__WRITE      ,__cm_wimscr_bits);
__IO_REG32_BIT(CM_NISR0,        0x40020098,__READ_WRITE ,__cm_nisr0_bits);
__IO_REG32_BIT(CM_NISR1,        0x4002009C,__READ_WRITE ,__cm_nisr1_bits);

/***************************************************************************
 **
 **  DFC
 **
 ***************************************************************************/
__IO_REG32_BIT(DF_IDR,          0x40011000,__READ       ,__df_idr_bits);
__IO_REG32_BIT(DF_CEDR,         0x40011004,__READ_WRITE ,__df_cedr_bits);
__IO_REG32_BIT(DF_SRR,          0x40011008,__WRITE      ,__df_srr_bits);
__IO_REG32_BIT(DF_CR,           0x4001100C,__READ_WRITE ,__df_cr_bits);
__IO_REG32_BIT(DF_MR,           0x40011010,__READ_WRITE ,__df_mr_bits);
__IO_REG32_BIT(DF_IMSCR,        0x40011014,__READ_WRITE ,__df_imscr_bits);
__IO_REG32_BIT(DF_RISR,         0x40011018,__READ       ,__df_imscr_bits);
__IO_REG32_BIT(DF_MISR,         0x4001101C,__READ       ,__df_imscr_bits);
__IO_REG32_BIT(DF_ICR,          0x40011020,__WRITE      ,__df_imscr_bits);
__IO_REG32(    DF_AR,           0x40011024,__READ_WRITE );
__IO_REG32(    DF_DR,           0x40011028,__READ_WRITE );
__IO_REG32(    DF_KR,           0x4001102C,__WRITE      );
__IO_REG32_BIT(DF_PCR,          0x40011030,__READ_WRITE ,__df_pcr_bits);

/***************************************************************************
 **
 **  DMA
 **
 ***************************************************************************/
__IO_REG32(    DMA_ISR0,        0x400F0000,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR0,       0x400F0004,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR0,        0x400F0008,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR0,       0x400F000C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR0,         0x400F0010,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR0,         0x400F0014,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR0,        0x400F0018,__READ       );
__IO_REG32(    DMA_CDR0,        0x400F001C,__READ       );
__IO_REG32_BIT(DMA_MTR0,        0x400F0020,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR0,        0x400F0024,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR1,        0x400F0080,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR1,       0x400F0084,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR1,        0x400F0088,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR1,       0x400F008C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR1,         0x400F0090,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR1,         0x400F0094,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR1,        0x400F0098,__READ       );
__IO_REG32(    DMA_CDR1,        0x400F009C,__READ       );
__IO_REG32_BIT(DMA_MTR1,        0x400F00A0,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR1,        0x400F00A4,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR2,        0x400F0100,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR2,       0x400F0104,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR2,        0x400F0108,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR2,       0x400F010C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR2,         0x400F0110,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR2,         0x400F0114,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR2,        0x400F0118,__READ       );
__IO_REG32(    DMA_CDR2,        0x400F011C,__READ       );
__IO_REG32_BIT(DMA_MTR2,        0x400F0120,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR2,        0x400F0124,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR3,        0x400F0180,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR3,       0x400F0184,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR3,        0x400F0188,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR3,       0x400F018C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR3,         0x400F0190,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR3,         0x400F0194,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR3,        0x400F0198,__READ       );
__IO_REG32(    DMA_CDR3,        0x400F019C,__READ       );
__IO_REG32_BIT(DMA_MTR3,        0x400F01A0,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR3,        0x400F01A4,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR4,        0x400F0200,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR4,       0x400F0204,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR4,        0x400F0208,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR4,       0x400F020C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR4,         0x400F0210,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR4,         0x400F0214,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR4,        0x400F0218,__READ       );
__IO_REG32(    DMA_CDR4,        0x400F021C,__READ       );
__IO_REG32_BIT(DMA_MTR4,        0x400F0220,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR4,        0x400F0224,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32(    DMA_ISR5,        0x400F0280,__READ_WRITE );
__IO_REG32_BIT(DMA_ISCR5,       0x400F0284,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32(    DMA_IDR5,        0x400F0288,__READ_WRITE );
__IO_REG32_BIT(DMA_IDCR5,       0x400F028C,__READ_WRITE ,__dma_iscr_bits);
__IO_REG32_BIT(DMA_CR5,         0x400F0290,__READ_WRITE ,__dma_cr_bits);
__IO_REG32_BIT(DMA_SR5,         0x400F0294,__READ       ,__dma_sr_bits);
__IO_REG32(    DMA_CSR5,        0x400F0298,__READ       );
__IO_REG32(    DMA_CDR5,        0x400F029C,__READ       );
__IO_REG32_BIT(DMA_MTR5,        0x400F02A0,__READ_WRITE ,__dma_mtr_bits);
__IO_REG32_BIT(DMA_RSR5,        0x400F02A4,__READ_WRITE ,__dma_rsr_bits);
__IO_REG32_BIT(DMA_IDR,         0x400F0500,__READ       ,__dma_idr_bits);
__IO_REG32_BIT(DMA_SRR,         0x400F0504,__WRITE      ,__dma_srr_bits);
__IO_REG32_BIT(DMA_CESR,        0x400F0508,__READ       ,__dma_cesr_bits);
__IO_REG32_BIT(DMA_ISR,         0x400F050C,__READ       ,__dma_isr_bits);
__IO_REG32_BIT(DMA_ICR,         0x400F0510,__WRITE      ,__dma_icr_bits);

/***************************************************************************
 **
 **  ENC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ENC0_IDR,        0x400C0000,__READ       ,__enc_idr_bits);
__IO_REG32_BIT(ENC0_CEDR,       0x400C0004,__READ_WRITE ,__enc_cedr_bits);
__IO_REG32_BIT(ENC0_SRR,        0x400C0008,__WRITE      ,__enc_srr_bits);
__IO_REG32_BIT(ENC0_CR0,        0x400C000C,__READ_WRITE ,__enc_cr0_bits);
__IO_REG32_BIT(ENC0_CR1,        0x400C0010,__READ_WRITE ,__enc_cr1_bits);
__IO_REG32_BIT(ENC0_SR,         0x400C0014,__READ_WRITE ,__enc_sr_bits);
__IO_REG32_BIT(ENC0_IMSCR,      0x400C0018,__READ_WRITE ,__enc_imscr_bits);
__IO_REG32_BIT(ENC0_RISR,       0x400C001C,__READ       ,__enc_imscr_bits);
__IO_REG32_BIT(ENC0_MISR,       0x400C0020,__READ       ,__enc_imscr_bits);
__IO_REG32_BIT(ENC0_ICR,        0x400C0024,__WRITE      ,__enc_imscr_bits);
__IO_REG16(    ENC0_PCR,        0x400C0028,__READ_WRITE );
__IO_REG16(    ENC0_PRR,        0x400C002C,__READ_WRITE );
__IO_REG16(    ENC0_SPCR,       0x400C0030,__READ_WRITE );
__IO_REG16(    ENC0_SPRR,       0x400C0034,__READ_WRITE );
__IO_REG16(    ENC0_PACCR,      0x400C0038,__READ_WRITE );
__IO_REG16(    ENC0_PACDR,      0x400C003C,__READ_WRITE );
__IO_REG16(    ENC0_PBCCR,      0x400C0040,__READ_WRITE );
__IO_REG16(    ENC0_PBCDR,      0x400C0044,__READ_WRITE );

/***************************************************************************
 **
 **  ENC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ENC1_IDR,        0x400C1000,__READ       ,__enc_idr_bits);
__IO_REG32_BIT(ENC1_CEDR,       0x400C1004,__READ_WRITE ,__enc_cedr_bits);
__IO_REG32_BIT(ENC1_SRR,        0x400C1008,__WRITE      ,__enc_srr_bits);
__IO_REG32_BIT(ENC1_CR0,        0x400C100C,__READ_WRITE ,__enc_cr0_bits);
__IO_REG32_BIT(ENC1_CR1,        0x400C1010,__READ_WRITE ,__enc_cr1_bits);
__IO_REG32_BIT(ENC1_SR,         0x400C1014,__READ_WRITE ,__enc_sr_bits);
__IO_REG32_BIT(ENC1_IMSCR,      0x400C1018,__READ_WRITE ,__enc_imscr_bits);
__IO_REG32_BIT(ENC1_RISR,       0x400C101C,__READ       ,__enc_imscr_bits);
__IO_REG32_BIT(ENC1_MISR,       0x400C1020,__READ       ,__enc_imscr_bits);
__IO_REG32_BIT(ENC1_ICR,        0x400C1024,__WRITE      ,__enc_imscr_bits);
__IO_REG16(    ENC1_PCR,        0x400C1028,__READ_WRITE );
__IO_REG16(    ENC1_PRR,        0x400C102C,__READ_WRITE );
__IO_REG16(    ENC1_SPCR,       0x400C1030,__READ_WRITE );
__IO_REG16(    ENC1_SPRR,       0x400C1034,__READ_WRITE );
__IO_REG16(    ENC1_PACCR,      0x400C1038,__READ_WRITE );
__IO_REG16(    ENC1_PACDR,      0x400C103C,__READ_WRITE );
__IO_REG16(    ENC1_PBCCR,      0x400C1040,__READ_WRITE );
__IO_REG16(    ENC1_PBCDR,      0x400C1044,__READ_WRITE );

/***************************************************************************
 **
 **  Free Running Timer
 **
 ***************************************************************************/
__IO_REG32_BIT(FRT_IDR,         0x40031000,__READ       ,__frt_idr_bits);
__IO_REG32_BIT(FRT_CEDR,        0x40031004,__READ_WRITE ,__frt_cedr_bits);
__IO_REG32_BIT(FRT_SRR,         0x40031008,__WRITE      ,__frt_srr_bits);
__IO_REG32_BIT(FRT_CR,          0x4003100C,__READ_WRITE ,__frt_cr_bits);
__IO_REG32_BIT(FRT_SR,          0x40031010,__READ       ,__frt_srr_bits);
__IO_REG32_BIT(FRT_IMSCR,       0x40031014,__READ_WRITE ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_RISR,        0x40031018,__READ       ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_MISR,        0x4003101C,__READ       ,__frt_imscr_bits);
__IO_REG32_BIT(FRT_ICR,         0x40031020,__WRITE      ,__frt_imscr_bits);
__IO_REG32(    FRT_DR,          0x40031024,__READ_WRITE );
__IO_REG32(    FRT_DBR,         0x40031028,__READ       );
__IO_REG32(    FRT_CVR,         0x4003102C,__READ       );

/***************************************************************************
 **
 **  Port 0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO0_IDR,       0x40050000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO0_CEDR,      0x40050004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO0_SRR,       0x40050008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO0_IMSCR,     0x4005000C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_RISR,      0x40050010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_MISR,      0x40050014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_ICR,       0x40050018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_OER,       0x4005001C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_ODR,       0x40050020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_OSR,       0x40050024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_WODR,      0x40050028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_SODR,      0x4005002C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_CODR,      0x40050030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_ODSR,      0x40050034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO0_PDSR,      0x40050038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_IDR,       0x40051000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO1_CEDR,      0x40051004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO1_SRR,       0x40051008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO1_IMSCR,     0x4005100C,__READ_WRITE ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_RISR,      0x40051010,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_MISR,      0x40051014,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_ICR,       0x40051018,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_OER,       0x4005101C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_ODR,       0x40051020,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_OSR,       0x40051024,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_WODR,      0x40051028,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_SODR,      0x4005102C,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_CODR,      0x40051030,__WRITE      ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_ODSR,      0x40051034,__READ       ,__gpio_imscr_bits);
__IO_REG32_BIT(GPIO1_PDSR,      0x40051038,__READ       ,__gpio_imscr_bits);

/***************************************************************************
 **
 **  Port 2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_IDR,       0x40052000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO2_CEDR,      0x40052004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO2_SRR,       0x40052008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO2_IMSCR,     0x4005200C,__READ_WRITE ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_RISR,      0x40052010,__READ       ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_MISR,      0x40052014,__READ       ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_ICR,       0x40052018,__WRITE      ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_OER,       0x4005201C,__WRITE      ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_ODR,       0x40052020,__WRITE      ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_OSR,       0x40052024,__READ       ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_WODR,      0x40052028,__WRITE      ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_SODR,      0x4005202C,__WRITE      ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_CODR,      0x40052030,__WRITE      ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_ODSR,      0x40052034,__READ       ,__gpio2_imscr_bits);
__IO_REG32_BIT(GPIO2_PDSR,      0x40052038,__READ       ,__gpio2_imscr_bits);

/***************************************************************************
 **
 **  Port 3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_IDR,       0x40053000,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIO3_CEDR,      0x40053004,__READ_WRITE ,__gpio_cedr_bits);
__IO_REG32_BIT(GPIO3_SRR,       0x40053008,__WRITE      ,__gpio_srr_bits);
__IO_REG32_BIT(GPIO3_IMSCR,     0x4005300C,__READ_WRITE ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_RISR,      0x40053010,__READ       ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_MISR,      0x40053014,__READ       ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_ICR,       0x40053018,__WRITE      ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_OER,       0x4005301C,__WRITE      ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_ODR,       0x40053020,__WRITE      ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_OSR,       0x40053024,__READ       ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_WODR,      0x40053028,__WRITE      ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_SODR,      0x4005302C,__WRITE      ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_CODR,      0x40053030,__WRITE      ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_ODSR,      0x40053034,__READ       ,__gpio3_imscr_bits);
__IO_REG32_BIT(GPIO3_PDSR,      0x40053038,__READ       ,__gpio3_imscr_bits);

/***************************************************************************
 **
 **  I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_IDR,        0x400A0000,__READ       ,__i2c_idr_bits);
__IO_REG32_BIT(I2C0_CEDR,       0x400A0004,__READ_WRITE ,__i2c_cedr_bits);
__IO_REG32_BIT(I2C0_SRR,        0x400A0008,__WRITE      ,__i2c_srr_bits);
__IO_REG32_BIT(I2C0_CR,         0x400A000C,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C0_MR,         0x400A0010,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C0_SR,         0x400A0014,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C0_IMSCR,      0x400A0018,__READ_WRITE ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_RISR,       0x400A001C,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_MISR,       0x400A0020,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_ICR,        0x400A0024,__WRITE      ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C0_SDR,        0x400A0028,__READ_WRITE ,__i2c_sdr_bits);
__IO_REG32_BIT(I2C0_SSAR,       0x400A002C,__READ_WRITE ,__i2c_ssar_bits);
__IO_REG32_BIT(I2C0_HSDR,       0x400A0030,__READ_WRITE ,__i2c_hsdr_bits);
__IO_REG32_BIT(I2C0_DMACR,      0x400A0034,__READ_WRITE ,__i2c_dmacr_bits);

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_IDR,        0x400A1000,__READ       ,__i2c_idr_bits);
__IO_REG32_BIT(I2C1_CEDR,       0x400A1004,__READ_WRITE ,__i2c_cedr_bits);
__IO_REG32_BIT(I2C1_SRR,        0x400A1008,__WRITE      ,__i2c_srr_bits);
__IO_REG32_BIT(I2C1_CR,         0x400A100C,__READ_WRITE ,__i2c_cr_bits);
__IO_REG32_BIT(I2C1_MR,         0x400A1010,__READ_WRITE ,__i2c_mr_bits);
__IO_REG32_BIT(I2C1_SR,         0x400A1014,__READ       ,__i2c_sr_bits);
__IO_REG32_BIT(I2C1_IMSCR,      0x400A1018,__READ_WRITE ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_RISR,       0x400A101C,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_MISR,       0x400A1020,__READ       ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_ICR,        0x400A1024,__WRITE      ,__i2c_imscr_bits);
__IO_REG32_BIT(I2C1_SDR,        0x400A1028,__READ_WRITE ,__i2c_sdr_bits);
__IO_REG32_BIT(I2C1_SSAR,       0x400A102C,__READ_WRITE ,__i2c_ssar_bits);
__IO_REG32_BIT(I2C1_HSDR,       0x400A1030,__READ_WRITE ,__i2c_hsdr_bits);
__IO_REG32_BIT(I2C1_DMACR,      0x400A1034,__READ_WRITE ,__i2c_dmacr_bits);

/***************************************************************************
 **
 **  IMC0
 **
 ***************************************************************************/
__IO_REG32_BIT(IMC0_IDR,        0x400B0000,__READ       ,__imc_idr_bits);
__IO_REG32_BIT(IMC0_CEDR,       0x400B0004,__READ_WRITE ,__imc_cedr_bits);
__IO_REG32_BIT(IMC0_SRR,        0x400B0008,__WRITE      ,__imc_srr_bits);
__IO_REG32_BIT(IMC0_CR0,        0x400B000C,__READ_WRITE ,__imc_cr0_bits);
__IO_REG32_BIT(IMC0_CR1,        0x400B0010,__READ_WRITE ,__imc_cr1_bits);
__IO_REG16(    IMC0_CNTR,       0x400B0014,__READ       );
__IO_REG32_BIT(IMC0_SR,         0x400B0018,__READ_WRITE ,__imc_sr_bits);
__IO_REG32_BIT(IMC0_IMSCR,      0x400B001C,__READ_WRITE ,__imc_imscr_bits);
__IO_REG32_BIT(IMC0_RISR,       0x400B0020,__READ       ,__imc_imscr_bits);
__IO_REG32_BIT(IMC0_MISR,       0x400B0024,__READ       ,__imc_imscr_bits);
__IO_REG32_BIT(IMC0_ICR,        0x400B0028,__WRITE      ,__imc_imscr_bits);
__IO_REG16(    IMC0_TCR,        0x400B002C,__READ_WRITE );
__IO_REG16(    IMC0_DTCR,       0x400B0030,__READ_WRITE );
__IO_REG16(    IMC0_PACRR,      0x400B0034,__READ_WRITE );
__IO_REG16(    IMC0_PBCRR,      0x400B0038,__READ_WRITE );
__IO_REG16(    IMC0_PCCRR,      0x400B003C,__READ_WRITE );
__IO_REG16(    IMC0_PACFR,      0x400B0040,__READ_WRITE );
__IO_REG16(    IMC0_PBCFR,      0x400B0044,__READ_WRITE );
__IO_REG16(    IMC0_PCCFR,      0x400B0048,__READ_WRITE );
__IO_REG32_BIT(IMC0_ASTSR,      0x400B004C,__READ_WRITE ,__imc_astsr_bits);
__IO_REG16(    IMC0_ASCRR0,     0x400B0050,__READ_WRITE );
__IO_REG16(    IMC0_ASCRR1,     0x400B0054,__READ_WRITE );
__IO_REG16(    IMC0_ASCRR2,     0x400B0058,__READ_WRITE );
__IO_REG16(    IMC0_ASCFR0,     0x400B005C,__READ_WRITE );
__IO_REG16(    IMC0_ASCFR1,     0x400B0060,__READ_WRITE );
__IO_REG16(    IMC0_ASCFR2,     0x400B0064,__READ_WRITE );

/***************************************************************************
 **
 **  IMC1
 **
 ***************************************************************************/
__IO_REG32_BIT(IMC1_IDR,        0x400B1000,__READ       ,__imc_idr_bits);
__IO_REG32_BIT(IMC1_CEDR,       0x400B1004,__READ_WRITE ,__imc_cedr_bits);
__IO_REG32_BIT(IMC1_SRR,        0x400B1008,__WRITE      ,__imc_srr_bits);
__IO_REG32_BIT(IMC1_CR0,        0x400B100C,__READ_WRITE ,__imc_cr0_bits);
__IO_REG32_BIT(IMC1_CR1,        0x400B1010,__READ_WRITE ,__imc_cr1_bits);
__IO_REG16(    IMC1_CNTR,       0x400B1014,__READ       );
__IO_REG32_BIT(IMC1_SR,         0x400B1018,__READ_WRITE ,__imc_sr_bits);
__IO_REG32_BIT(IMC1_IMSCR,      0x400B101C,__READ_WRITE ,__imc_imscr_bits);
__IO_REG32_BIT(IMC1_RISR,       0x400B1020,__READ       ,__imc_imscr_bits);
__IO_REG32_BIT(IMC1_MISR,       0x400B1024,__READ       ,__imc_imscr_bits);
__IO_REG32_BIT(IMC1_ICR,        0x400B1028,__WRITE      ,__imc_imscr_bits);
__IO_REG16(    IMC1_TCR,        0x400B102C,__READ_WRITE );
__IO_REG16(    IMC1_DTCR,       0x400B1030,__READ_WRITE );
__IO_REG16(    IMC1_PACRR,      0x400B1034,__READ_WRITE );
__IO_REG16(    IMC1_PBCRR,      0x400B1038,__READ_WRITE );
__IO_REG16(    IMC1_PCCRR,      0x400B103C,__READ_WRITE );
__IO_REG16(    IMC1_PACFR,      0x400B1040,__READ_WRITE );
__IO_REG16(    IMC1_PBCFR,      0x400B1044,__READ_WRITE );
__IO_REG16(    IMC1_PCCFR,      0x400B1048,__READ_WRITE );
__IO_REG32_BIT(IMC1_ASTSR,      0x400B104C,__READ_WRITE ,__imc_astsr_bits);
__IO_REG16(    IMC1_ASCRR0,     0x400B1050,__READ_WRITE );
__IO_REG16(    IMC1_ASCRR1,     0x400B1054,__READ_WRITE );
__IO_REG16(    IMC1_ASCRR2,     0x400B1058,__READ_WRITE );
__IO_REG16(    IMC1_ASCFR0,     0x400B105C,__READ_WRITE );
__IO_REG16(    IMC1_ASCFR1,     0x400B1060,__READ_WRITE );
__IO_REG16(    IMC1_ASCFR2,     0x400B1064,__READ_WRITE );

/***************************************************************************
 **
 **  IOCONF
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCONF_P0MLR,    0x40058000,__READ_WRITE ,__ioconf_mlr0_bits);
__IO_REG32_BIT(IOCONF_P0MHR,    0x40058004,__READ_WRITE ,__ioconf_mhr0_bits);
__IO_REG32_BIT(IOCONF_P0PUR,    0x40058008,__READ_WRITE ,__ioconf_pucr0_bits);
__IO_REG32_BIT(IOCONF_P0ODCR,   0x4005800C,__READ_WRITE ,__ioconf_odcr0_bits);
__IO_REG32_BIT(IOCONF_P1MLR,    0x40058010,__READ_WRITE ,__ioconf_mlr1_bits);
__IO_REG32_BIT(IOCONF_P1MHR,    0x40058014,__READ_WRITE ,__ioconf_mhr1_bits);
__IO_REG32_BIT(IOCONF_P1PUR,    0x40058018,__READ_WRITE ,__ioconf_pucr1_bits);
__IO_REG32_BIT(IOCONF_P1ODCR,   0x4005801C,__READ_WRITE ,__ioconf_odcr1_bits);
__IO_REG32_BIT(IOCONF_P2MLR,    0x40058020,__READ_WRITE ,__ioconf_mlr2_bits);
__IO_REG32_BIT(IOCONF_P2MHR,    0x40058024,__READ_WRITE ,__ioconf_mhr2_bits);
__IO_REG32_BIT(IOCONF_P2PUR,    0x40058028,__READ_WRITE ,__ioconf_pucr2_bits);
__IO_REG32_BIT(IOCONF_P2ODCR,   0x4005802C,__READ_WRITE ,__ioconf_odcr2_bits);
__IO_REG32_BIT(IOCONF_P3MLR,    0x40058030,__READ_WRITE ,__ioconf_mlr3_bits);
__IO_REG32_BIT(IOCONF_P3PUR,    0x40058038,__READ_WRITE ,__ioconf_pucr3_bits);
__IO_REG32_BIT(IOCONF_P3ODCR,   0x4005803C,__READ_WRITE ,__ioconf_odcr3_bits);

/***************************************************************************
 **
 **  LCD
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_IDR,         0x400D0000,__READ       ,__lcd_idr_bits);
__IO_REG32_BIT(LCD_CEDR,        0x400D0004,__READ_WRITE ,__lcd_cedr_bits);
__IO_REG32_BIT(LCD_SRR,         0x400D0008,__WRITE      ,__lcd_srr_bits);
__IO_REG32_BIT(LCD_CR,          0x400D000C,__READ_WRITE ,__lcd_cr_bits);
__IO_REG32_BIT(LCD_CDR,         0x400D0010,__READ_WRITE ,__lcd_cdr_bits);
__IO_REG32(    LCD_DMR_BASE,    0x400D0400,__READ_WRITE );

/***************************************************************************
 **
 **  OPAMP
 **
 ***************************************************************************/
__IO_REG32_BIT(OPA_IDR,         0x40042000,__READ       ,__opa_idr_bits);
__IO_REG32_BIT(OPA_CEDR,        0x40042004,__READ_WRITE ,__opa_cedr_bits);
__IO_REG32_BIT(OPA_SRR,         0x40042008,__WRITE      ,__opa_srr_bits);
__IO_REG32_BIT(OPA_CR0,         0x4004200C,__READ_WRITE ,__opa_cr_bits);
__IO_REG32_BIT(OPA_CR1,         0x40042010,__READ_WRITE ,__opa_cr_bits);
__IO_REG32_BIT(OPA_GCR0,        0x40042014,__READ_WRITE ,__opa_gcr_bits);
__IO_REG32_BIT(OPA_GCR1,        0x40042018,__READ_WRITE ,__opa_gcr_bits);
__IO_REG32_BIT(OPA_IMSCR,       0x4004201C,__READ_WRITE ,__opa_imscr_bits);
__IO_REG32_BIT(OPA_RISR,        0x40042020,__READ_WRITE ,__opa_imscr_bits);
__IO_REG32_BIT(OPA_MISR,        0x40042024,__READ_WRITE ,__opa_imscr_bits);
__IO_REG32_BIT(OPA_ICR,         0x40042028,__READ_WRITE ,__opa_imscr_bits);

/***************************************************************************
 **
 **  PFC
 **
 ***************************************************************************/
__IO_REG32_BIT(PF_IDR,          0x40010000,__READ       ,__pf_idr_bits);
__IO_REG32_BIT(PF_CEDR,         0x40010004,__READ_WRITE ,__pf_cedr_bits);
__IO_REG32_BIT(PF_SRR,          0x40010008,__WRITE      ,__pf_srr_bits);
__IO_REG32_BIT(PF_CR,           0x4001000C,__READ_WRITE ,__pf_cr_bits);
__IO_REG32_BIT(PF_MR,           0x40010010,__READ_WRITE ,__pf_mr_bits);
__IO_REG32_BIT(PF_IMSCR,        0x40010014,__READ_WRITE ,__pf_imscr_bits);
__IO_REG32_BIT(PF_RISR,         0x40010018,__READ       ,__pf_imscr_bits);
__IO_REG32_BIT(PF_MISR,         0x4001001C,__READ       ,__pf_imscr_bits);
__IO_REG32_BIT(PF_ICR,          0x40010020,__WRITE      ,__pf_imscr_bits);
__IO_REG32_BIT(PF_SR,           0x40010024,__READ       ,__pf_sr_bits);
__IO_REG32(    PF_AR,           0x40010028,__READ_WRITE );
__IO_REG32(    PF_DR,           0x4001002C,__READ_WRITE );
__IO_REG32(    PF_KR,           0x40010030,__WRITE      );
__IO_REG32_BIT(SO_PSR,          0x40010034,__READ       ,__so_psr_bits);
__IO_REG32_BIT(SO_CSR,          0x40010038,__READ       ,__so_csr_bits);
__IO_REG32_BIT(PF_IOTR,         0x4001003C,__READ_WRITE ,__pf_iotr_bits);

/***************************************************************************
 **
 **  PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0_IDR,        0x40070000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM0_CEDR,       0x40070004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM0_SRR,        0x40070008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM0_CSR,        0x4007000C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM0_CCR,        0x40070010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM0_SR,         0x40070014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM0_IMSCR,      0x40070018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_RISR,       0x4007001C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_MISR,       0x40070020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_ICR,        0x40070024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM0_CDR,        0x40070028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM0_PRDR,       0x4007002C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM0_PULR,       0x40070030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM0_CCDR,       0x40070034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM0_CPRDR,      0x40070038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM0_CPULR,      0x4007003C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_IDR,        0x40071000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM1_CEDR,       0x40071004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM1_SRR,        0x40071008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM1_CSR,        0x4007100C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM1_CCR,        0x40071010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM1_SR,         0x40071014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM1_IMSCR,      0x40071018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_RISR,       0x4007101C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_MISR,       0x40071020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_ICR,        0x40071024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM1_CDR,        0x40071028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM1_PRDR,       0x4007102C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM1_PULR,       0x40071030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM1_CCDR,       0x40071034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM1_CPRDR,      0x40071038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM1_CPULR,      0x4007103C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2_IDR,        0x40072000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM2_CEDR,       0x40072004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM2_SRR,        0x40072008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM2_CSR,        0x4007200C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM2_CCR,        0x40072010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM2_SR,         0x40072014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM2_IMSCR,      0x40072018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_RISR,       0x4007201C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_MISR,       0x40072020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_ICR,        0x40072024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM2_CDR,        0x40072028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM2_PRDR,       0x4007202C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM2_PULR,       0x40072030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM2_CCDR,       0x40072034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM2_CPRDR,      0x40072038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM2_CPULR,      0x4007203C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM3
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM3_IDR,        0x40073000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM3_CEDR,       0x40073004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM3_SRR,        0x40073008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM3_CSR,        0x4007300C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM3_CCR,        0x40073010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM3_SR,         0x40073014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM3_IMSCR,      0x40073018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_RISR,       0x4007301C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_MISR,       0x40073020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_ICR,        0x40073024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM3_CDR,        0x40073028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM3_PRDR,       0x4007302C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM3_PULR,       0x40073030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM3_CCDR,       0x40073034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM3_CPRDR,      0x40073038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM3_CPULR,      0x4007303C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM4
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM4_IDR,        0x40074000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM4_CEDR,       0x40074004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM4_SRR,        0x40074008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM4_CSR,        0x4007400C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM4_CCR,        0x40074010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM4_SR,         0x40074014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM4_IMSCR,      0x40074018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM4_RISR,       0x4007401C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM4_MISR,       0x40074020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM4_ICR,        0x40074024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM4_CDR,        0x40074028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM4_PRDR,       0x4007402C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM4_PULR,       0x40074030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM4_CCDR,       0x40074034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM4_CPRDR,      0x40074038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM4_CPULR,      0x4007403C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM5
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM5_IDR,        0x40075000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM5_CEDR,       0x40075004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM5_SRR,        0x40075008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM5_CSR,        0x4007500C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM5_CCR,        0x40075010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM5_SR,         0x40075014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM5_IMSCR,      0x40075018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM5_RISR,       0x4007501C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM5_MISR,       0x40075020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM5_ICR,        0x40075024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM5_CDR,        0x40075028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM5_PRDR,       0x4007502C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM5_PULR,       0x40075030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM5_CCDR,       0x40075034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM5_CPRDR,      0x40075038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM5_CPULR,      0x4007503C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM6
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM6_IDR,        0x40076000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM6_CEDR,       0x40076004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM6_SRR,        0x40076008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM6_CSR,        0x4007600C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM6_CCR,        0x40076010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM6_SR,         0x40076014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM6_IMSCR,      0x40076018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM6_RISR,       0x4007601C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM6_MISR,       0x40076020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM6_ICR,        0x40076024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM6_CDR,        0x40076028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM6_PRDR,       0x4007602C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM6_PULR,       0x40076030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM6_CCDR,       0x40076034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM6_CPRDR,      0x40076038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM6_CPULR,      0x4007603C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  PWM7
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM7_IDR,        0x40077000,__READ       ,__pwm_idr_bits);
__IO_REG32_BIT(PWM7_CEDR,       0x40077004,__READ_WRITE ,__pwm_cedr_bits);
__IO_REG32_BIT(PWM7_SRR,        0x40077008,__WRITE      ,__pwm_srr_bits);
__IO_REG32_BIT(PWM7_CSR,        0x4007700C,__WRITE      ,__pwm_csr_bits);
__IO_REG32_BIT(PWM7_CCR,        0x40077010,__WRITE      ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM7_SR,         0x40077014,__READ       ,__pwm_ccr_bits);
__IO_REG32_BIT(PWM7_IMSCR,      0x40077018,__READ_WRITE ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM7_RISR,       0x4007701C,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM7_MISR,       0x40077020,__READ       ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM7_ICR,        0x40077024,__WRITE      ,__pwm_imscr_bits);
__IO_REG32_BIT(PWM7_CDR,        0x40077028,__READ_WRITE ,__pwm_cdr_bits);
__IO_REG32_BIT(PWM7_PRDR,       0x4007702C,__READ_WRITE ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM7_PULR,       0x40077030,__READ_WRITE ,__pwm_pulr_bits);
__IO_REG32_BIT(PWM7_CCDR,       0x40077034,__READ       ,__pwm_ccdr_bits);
__IO_REG32_BIT(PWM7_CPRDR,      0x40077038,__READ       ,__pwm_prdr_bits);
__IO_REG32_BIT(PWM7_CPULR,      0x4007703C,__READ       ,__pwm_pulr_bits);

/***************************************************************************
 **
 **  SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CR0,        0x40090000,__READ_WRITE ,__spi_cr0_bits);
__IO_REG32_BIT(SPI0_CR1,        0x40090004,__READ_WRITE ,__spi_cr1_bits);
__IO_REG16(    SPI0_DR,         0x40090008,__READ_WRITE );
__IO_REG32_BIT(SPI0_SR,         0x4009000C,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI0_CPSR,       0x40090010,__READ_WRITE ,__spi_cpsr_bits);
__IO_REG32_BIT(SPI0_IMSC,       0x40090014,__READ_WRITE ,__spi_imsc_bits);
__IO_REG32_BIT(SPI0_RISR,       0x40090018,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI0_MISR,       0x4009001C,__READ       ,__spi_misr_bits);
__IO_REG32_BIT(SPI0_ICR,        0x40090020,__WRITE      ,__spi_icr_bits);
__IO_REG32_BIT(SPI0_DMACR,      0x40090024,__READ_WRITE ,__spi_dmacr_bits);

/***************************************************************************
 **
 **  SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CR0,        0x40091000,__READ_WRITE ,__spi_cr0_bits);
__IO_REG32_BIT(SPI1_CR1,        0x40091004,__READ_WRITE ,__spi_cr1_bits);
__IO_REG16(    SPI1_DR,         0x40091008,__READ_WRITE );
__IO_REG32_BIT(SPI1_SR,         0x4009100C,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI1_CPSR,       0x40091010,__READ_WRITE ,__spi_cpsr_bits);
__IO_REG32_BIT(SPI1_IMSC,       0x40091014,__READ_WRITE ,__spi_imsc_bits);
__IO_REG32_BIT(SPI1_RISR,       0x40091018,__READ       ,__spi_ris_bits);
__IO_REG32_BIT(SPI1_MISR,       0x4009101C,__READ       ,__spi_misr_bits);
__IO_REG32_BIT(SPI1_ICR,        0x40091020,__WRITE      ,__spi_icr_bits);
__IO_REG32_BIT(SPI1_DMACR,      0x40091024,__READ_WRITE ,__spi_dmacr_bits);

/***************************************************************************
 **
 **  STT
 **
 ***************************************************************************/
__IO_REG32_BIT(STT_IDR,         0x40078000,__READ       ,__stt_idr_bits);
__IO_REG32_BIT(STT_CEDR,        0x40078004,__READ_WRITE ,__stt_cedr_bits);
__IO_REG32_BIT(STT_SRR,         0x40078008,__WRITE      ,__stt_srr_bits);
__IO_REG32_BIT(STT_CR,          0x4007800C,__WRITE      ,__stt_cr_bits);
__IO_REG32_BIT(STT_MR,          0x40078010,__READ_WRITE ,__stt_mr_bits);
__IO_REG32_BIT(STT_IMSCR,       0x40078014,__READ_WRITE ,__stt_imscr_bits);
__IO_REG32_BIT(STT_RISR,        0x40078018,__READ       ,__stt_imscr_bits);
__IO_REG32_BIT(STT_MISR,        0x4007801C,__READ       ,__stt_imscr_bits);
__IO_REG32_BIT(STT_ICR,         0x40078020,__WRITE      ,__stt_imscr_bits);
__IO_REG32_BIT(STT_SR,          0x40078024,__READ       ,__stt_sr_bits);
__IO_REG32(    STT_CNTR,        0x40078028,__READ_WRITE );
__IO_REG32(    STT_ALR,         0x4007802C,__READ_WRITE );

/***************************************************************************
 **
 **  TC0
 **
 ***************************************************************************/
__IO_REG32_BIT(TC0_IDR,         0x40060000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC0_CSSR,        0x40060004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC0_CEDR,        0x40060008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC0_SRR,         0x4006000C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC0_CSR,         0x40060010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC0_CCR,         0x40060014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC0_SR,          0x40060018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC0_IMSCR,       0x4006001C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_RISR,        0x40060020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_MISR,        0x40060024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_ICR,         0x40060028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC0_CDR,         0x4006002C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC0_CSMR,        0x40060030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC0_PRDR,        0x40060034,__READ_WRITE );
__IO_REG16(    TC0_PULR,        0x40060038,__READ_WRITE );
__IO_REG32_BIT(TC0_CCDR,        0x4006003C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC0_CCSMR,       0x40060040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC0_CPRDR,       0x40060044,__READ       );
__IO_REG16(    TC0_CPULR,       0x40060048,__READ       );
__IO_REG16(    TC0_CUCR,        0x4006004C,__READ       );
__IO_REG16(    TC0_CDCR,        0x40060050,__READ       );
__IO_REG16(    TC0_CVR,         0x40060054,__READ       );

/***************************************************************************
 **
 **  TC1
 **
 ***************************************************************************/
__IO_REG32_BIT(TC1_IDR,         0x40061000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC1_CSSR,        0x40061004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC1_CEDR,        0x40061008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC1_SRR,         0x4006100C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC1_CSR,         0x40061010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC1_CCR,         0x40061014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC1_SR,          0x40061018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC1_IMSCR,       0x4006101C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_RISR,        0x40061020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_MISR,        0x40061024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_ICR,         0x40061028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC1_CDR,         0x4006102C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC1_CSMR,        0x40061030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC1_PRDR,        0x40061034,__READ_WRITE );
__IO_REG16(    TC1_PULR,        0x40061038,__READ_WRITE );
__IO_REG32_BIT(TC1_CCDR,        0x4006103C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC1_CCSMR,       0x40061040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC1_CPRDR,       0x40061044,__READ       );
__IO_REG16(    TC1_CPULR,       0x40061048,__READ       );
__IO_REG16(    TC1_CUCR,        0x4006104C,__READ       );
__IO_REG16(    TC1_CDCR,        0x40061050,__READ       );
__IO_REG16(    TC1_CVR,         0x40061054,__READ       );

/***************************************************************************
 **
 **  TC2
 **
 ***************************************************************************/
__IO_REG32_BIT(TC2_IDR,         0x40062000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC2_CSSR,        0x40062004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC2_CEDR,        0x40062008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC2_SRR,         0x4006200C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC2_CSR,         0x40062010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC2_CCR,         0x40062014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC2_SR,          0x40062018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC2_IMSCR,       0x4006201C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_RISR,        0x40062020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_MISR,        0x40062024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_ICR,         0x40062028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC2_CDR,         0x4006202C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC2_CSMR,        0x40062030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC2_PRDR,        0x40062034,__READ_WRITE );
__IO_REG16(    TC2_PULR,        0x40062038,__READ_WRITE );
__IO_REG32_BIT(TC2_CCDR,        0x4006203C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC2_CCSMR,       0x40062040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC2_CPRDR,       0x40062044,__READ       );
__IO_REG16(    TC2_CPULR,       0x40062048,__READ       );
__IO_REG16(    TC2_CUCR,        0x4006204C,__READ       );
__IO_REG16(    TC2_CDCR,        0x40062050,__READ       );
__IO_REG16(    TC2_CVR,         0x40062054,__READ       );

/***************************************************************************
 **
 **  TC3
 **
 ***************************************************************************/
__IO_REG32_BIT(TC3_IDR,         0x40063000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC3_CSSR,        0x40063004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC3_CEDR,        0x40063008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC3_SRR,         0x4006300C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC3_CSR,         0x40063010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC3_CCR,         0x40063014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC3_SR,          0x40063018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC3_IMSCR,       0x4006301C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_RISR,        0x40063020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_MISR,        0x40063024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_ICR,         0x40063028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC3_CDR,         0x4006302C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC3_CSMR,        0x40063030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC3_PRDR,        0x40063034,__READ_WRITE );
__IO_REG16(    TC3_PULR,        0x40063038,__READ_WRITE );
__IO_REG32_BIT(TC3_CCDR,        0x4006303C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC3_CCSMR,       0x40063040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC3_CPRDR,       0x40063044,__READ       );
__IO_REG16(    TC3_CPULR,       0x40063048,__READ       );
__IO_REG16(    TC3_CUCR,        0x4006304C,__READ       );
__IO_REG16(    TC3_CDCR,        0x40063050,__READ       );
__IO_REG16(    TC3_CVR,         0x40063054,__READ       );

/***************************************************************************
 **
 **  TC4
 **
 ***************************************************************************/
__IO_REG32_BIT(TC4_IDR,         0x40064000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC4_CSSR,        0x40064004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC4_CEDR,        0x40064008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC4_SRR,         0x4006400C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC4_CSR,         0x40064010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC4_CCR,         0x40064014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC4_SR,          0x40064018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC4_IMSCR,       0x4006401C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_RISR,        0x40064020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_MISR,        0x40064024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_ICR,         0x40064028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC4_CDR,         0x4006402C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC4_CSMR,        0x40064030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC4_PRDR,        0x40064034,__READ_WRITE );
__IO_REG16(    TC4_PULR,        0x40064038,__READ_WRITE );
__IO_REG32_BIT(TC4_CCDR,        0x4006403C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC4_CCSMR,       0x40064040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC4_CPRDR,       0x40064044,__READ       );
__IO_REG16(    TC4_CPULR,       0x40064048,__READ       );
__IO_REG16(    TC4_CUCR,        0x4006404C,__READ       );
__IO_REG16(    TC4_CDCR,        0x40064050,__READ       );
__IO_REG16(    TC4_CVR,         0x40064054,__READ       );

/***************************************************************************
 **
 **  TC5
 **
 ***************************************************************************/
__IO_REG32_BIT(TC5_IDR,         0x40065000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC5_CSSR,        0x40065004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC5_CEDR,        0x40065008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC5_SRR,         0x4006500C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC5_CSR,         0x40065010,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC5_CCR,         0x40065014,__WRITE      ,__tc_csr_bits);
__IO_REG32_BIT(TC5_SR,          0x40065018,__READ       ,__tc_sr_bits);
__IO_REG32_BIT(TC5_IMSCR,       0x4006501C,__READ_WRITE ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_RISR,        0x40065020,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_MISR,        0x40065024,__READ       ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_ICR,         0x40065028,__WRITE      ,__tc_imscr_bits);
__IO_REG32_BIT(TC5_CDR,         0x4006502C,__READ_WRITE ,__tc_cdr_bits);
__IO_REG32_BIT(TC5_CSMR,        0x40065030,__READ_WRITE ,__tc_csmr_bits);
__IO_REG16(    TC5_PRDR,        0x40065034,__READ_WRITE );
__IO_REG16(    TC5_PULR,        0x40065038,__READ_WRITE );
__IO_REG32_BIT(TC5_CCDR,        0x4006503C,__READ       ,__tc_cdr_bits);
__IO_REG32_BIT(TC5_CCSMR,       0x40065040,__READ       ,__tc_csmr_bits);
__IO_REG16(    TC5_CPRDR,       0x40065044,__READ       );
__IO_REG16(    TC5_CPULR,       0x40065048,__READ       );
__IO_REG16(    TC5_CUCR,        0x4006504C,__READ       );
__IO_REG16(    TC5_CDCR,        0x40065050,__READ       );
__IO_REG16(    TC5_CVR,         0x40065054,__READ       );

/***************************************************************************
 **
 **  TC6
 **
 ***************************************************************************/
__IO_REG32_BIT(TC6_IDR,         0x40066000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC6_CSSR,        0x40066004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC6_CEDR,        0x40066008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC6_SRR,         0x4006600C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC6_CSR,         0x40066010,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC6_CCR,         0x40066014,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC6_SR,          0x40066018,__READ       ,__tc32_sr_bits);
__IO_REG32_BIT(TC6_IMSCR,       0x4006601C,__READ_WRITE ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_RISR,        0x40066020,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_MISR,        0x40066024,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_ICR,         0x40066028,__WRITE      ,__tc32_imscr_bits);
__IO_REG32_BIT(TC6_CDR,         0x4006602C,__READ_WRITE ,__tc32_cdr_bits);
__IO_REG32_BIT(TC6_CSMR,        0x40066030,__READ_WRITE ,__tc32_csmr_bits);
__IO_REG32(    TC6_PRDR,        0x40066034,__READ_WRITE );
__IO_REG32(    TC6_PULR,        0x40066038,__READ_WRITE );
__IO_REG32_BIT(TC6_CCDR,        0x4006603C,__READ       ,__tc32_cdr_bits);
__IO_REG32_BIT(TC6_CCSMR,       0x40066040,__READ       ,__tc32_csmr_bits);
__IO_REG32(    TC6_CPRDR,       0x40066044,__READ       );
__IO_REG32(    TC6_CPULR,       0x40066048,__READ       );
__IO_REG32(    TC6_CUCR,        0x4006604C,__READ       );
__IO_REG32(    TC6_CDCR,        0x40066050,__READ       );
__IO_REG32(    TC6_CVR,         0x40066054,__READ       );

/***************************************************************************
 **
 **  TC7
 **
 ***************************************************************************/
__IO_REG32_BIT(TC7_IDR,         0x40067000,__READ       ,__tc_idr_bits);
__IO_REG32_BIT(TC7_CSSR,        0x40067004,__READ_WRITE ,__tc_cssr_bits);
__IO_REG32_BIT(TC7_CEDR,        0x40067008,__READ_WRITE ,__tc_cedr_bits);
__IO_REG32_BIT(TC7_SRR,         0x4006700C,__WRITE      ,__tc_srr_bits);
__IO_REG32_BIT(TC7_CSR,         0x40067010,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC7_CCR,         0x40067014,__WRITE      ,__tc32_csr_bits);
__IO_REG32_BIT(TC7_SR,          0x40067018,__READ       ,__tc32_sr_bits);
__IO_REG32_BIT(TC7_IMSCR,       0x4006701C,__READ_WRITE ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_RISR,        0x40067020,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_MISR,        0x40067024,__READ       ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_ICR,         0x40067028,__WRITE      ,__tc32_imscr_bits);
__IO_REG32_BIT(TC7_CDR,         0x4006702C,__READ_WRITE ,__tc32_cdr_bits);
__IO_REG32_BIT(TC7_CSMR,        0x40067030,__READ_WRITE ,__tc32_csmr_bits);
__IO_REG32(    TC7_PRDR,        0x40067034,__READ_WRITE );
__IO_REG32(    TC7_PULR,        0x40067038,__READ_WRITE );
__IO_REG32_BIT(TC7_CCDR,        0x4006703C,__READ       ,__tc32_cdr_bits);
__IO_REG32_BIT(TC7_CCSMR,       0x40067040,__READ       ,__tc32_csmr_bits);
__IO_REG32(    TC7_CPRDR,       0x40067044,__READ       );
__IO_REG32(    TC7_CPULR,       0x40067048,__READ       );
__IO_REG32(    TC7_CUCR,        0x4006704C,__READ       );
__IO_REG32(    TC7_CDCR,        0x40067050,__READ       );
__IO_REG32(    TC7_CVR,         0x40067054,__READ       );

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
__IO_REG32_BIT(US0_IDR,         0x40080000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US0_CEDR,        0x40080004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US0_SRR,         0x40080008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US0_CR,          0x4008000C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US0_MR,          0x40080010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US0_IMSCR,       0x40080014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US0_RISR,        0x40080018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US0_MISR,        0x4008001C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US0_ICR,         0x40080020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US0_SR,          0x40080024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US0_RHR,         0x40080028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US0_THR,         0x4008002C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US0_BRGR,        0x40080030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US0_RTOR,        0x40080034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US0_TTGR,        0x40080038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US0_LIR,         0x4008003C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US0_DFWR0,       0x40080040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US0_DFWR1,       0x40080044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US0_DFRR0,       0x40080048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US0_DFRR1,       0x4008004C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US0_SBLR,        0x40080050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US0_LCP1,        0x40080054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US0_LCP2,        0x40080058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US0_DMACR,       0x4008005C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(US1_IDR,         0x40081000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US1_CEDR,        0x40081004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US1_SRR,         0x40081008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US1_CR,          0x4008100C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US1_MR,          0x40081010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US1_IMSCR,       0x40081014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US1_RISR,        0x40081018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US1_MISR,        0x4008101C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US1_ICR,         0x40081020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US1_SR,          0x40081024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US1_RHR,         0x40081028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US1_THR,         0x4008102C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US1_BRGR,        0x40081030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US1_RTOR,        0x40081034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US1_TTGR,        0x40081038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US1_LIR,         0x4008103C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US1_DFWR0,       0x40081040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US1_DFWR1,       0x40081044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US1_DFRR0,       0x40081048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US1_DFRR1,       0x4008104C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US1_SBLR,        0x40081050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US1_LCP1,        0x40081054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US1_LCP2,        0x40081058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US1_DMACR,       0x4008105C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(US2_IDR,         0x40082000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US2_CEDR,        0x40082004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US2_SRR,         0x40082008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US2_CR,          0x4008200C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US2_MR,          0x40082010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US2_IMSCR,       0x40082014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US2_RISR,        0x40082018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US2_MISR,        0x4008201C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US2_ICR,         0x40082020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US2_SR,          0x40082024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US2_RHR,         0x40082028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US2_THR,         0x4008202C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US2_BRGR,        0x40082030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US2_RTOR,        0x40082034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US2_TTGR,        0x40082038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US2_LIR,         0x4008203C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US2_DFWR0,       0x40082040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US2_DFWR1,       0x40082044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US2_DFRR0,       0x40082048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US2_DFRR1,       0x4008204C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US2_SBLR,        0x40082050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US2_LCP1,        0x40082054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US2_LCP2,        0x40082058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US2_DMACR,       0x4008205C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(US3_IDR,         0x40083000,__READ       ,__us_idr_bits);
__IO_REG32_BIT(US3_CEDR,        0x40083004,__READ_WRITE ,__us_cedr_bits);
__IO_REG32_BIT(US3_SRR,         0x40083008,__WRITE      ,__us_srr_bits);
__IO_REG32_BIT(US3_CR,          0x4008300C,__WRITE      ,__us_cr_bits);
__IO_REG32_BIT(US3_MR,          0x40083010,__READ_WRITE ,__us_mr_bits);
__IO_REG32_BIT(US3_IMSCR,       0x40083014,__READ_WRITE ,__us_imscr_bits);
__IO_REG32_BIT(US3_RISR,        0x40083018,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US3_MISR,        0x4008301C,__READ       ,__us_imscr_bits);
__IO_REG32_BIT(US3_ICR,         0x40083020,__WRITE      ,__us_icr_bits);
__IO_REG32_BIT(US3_SR,          0x40083024,__READ       ,__us_sr_bits);
__IO_REG32_BIT(US3_RHR,         0x40083028,__READ       ,__us_rhr_bits);
__IO_REG32_BIT(US3_THR,         0x4008302C,__WRITE      ,__us_thr_bits);
__IO_REG32_BIT(US3_BRGR,        0x40083030,__READ_WRITE ,__us_brgr_bits);
__IO_REG32_BIT(US3_RTOR,        0x40083034,__READ_WRITE ,__us_rtor_bits);
__IO_REG32_BIT(US3_TTGR,        0x40083038,__READ_WRITE ,__us_ttgr_bits);
__IO_REG32_BIT(US3_LIR,         0x4008303C,__READ_WRITE ,__us_lir_bits);
__IO_REG32_BIT(US3_DFWR0,       0x40083040,__READ_WRITE ,__us_dfwr0_bits);
__IO_REG32_BIT(US3_DFWR1,       0x40083044,__READ_WRITE ,__us_dfwr1_bits);
__IO_REG32_BIT(US3_DFRR0,       0x40083048,__READ       ,__us_dfwr0_bits);
__IO_REG32_BIT(US3_DFRR1,       0x4008304C,__READ       ,__us_dfwr1_bits);
__IO_REG32_BIT(US3_SBLR,        0x40083050,__READ_WRITE ,__us_sblr_bits);
__IO_REG32_BIT(US3_LCP1,        0x40083054,__READ_WRITE ,__us_lcp1_bits);
__IO_REG32_BIT(US3_LCP2,        0x40083058,__READ_WRITE ,__us_lcp2_bits);
__IO_REG32_BIT(US3_DMACR,       0x4008305C,__READ_WRITE ,__us_dmacr_bits);

/***************************************************************************
 **
 **  WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_IDR,         0x40030000,__READ       ,__wdt_idr_bits);
__IO_REG32_BIT(WDT_CR,          0x40030004,__WRITE      ,__wdt_cr_bits);
__IO_REG32_BIT(WDT_MR,          0x40030008,__READ_WRITE ,__wdt_mr_bits);
__IO_REG32_BIT(WDT_OMR,         0x4003000C,__READ_WRITE ,__wdt_omr_bits);
__IO_REG32_BIT(WDT_SR,          0x40030010,__READ       ,__wdt_sr_bits);
__IO_REG32_BIT(WDT_IMSCR,       0x40030014,__READ_WRITE ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_RISR,        0x40030018,__READ       ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_MISR,        0x4003001C,__READ       ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_ICR,         0x40030020,__WRITE      ,__wdt_imscr_bits);
__IO_REG32_BIT(WDT_PWR,         0x40030024,__READ_WRITE ,__wdt_pwr_bits);
__IO_REG32_BIT(WDT_CTR,         0x40030028,__READ       ,__wdt_ctr_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  S3FM02G DMA channels number
 **
 ***************************************************************************/
#define UART0_TX_DMA      0x00
#define UART0_RX_DMA      0x01
#define UART1_TX_DMA      0x02
#define UART1_RX_DMA      0x03
#define UART2_TX_DMA      0x04
#define UART2_RX_DMA      0x05
#define UART3_TX_DMA      0x06
#define UART3_RX_DMA      0x07
#define SPI0_TX_DMA       0x0A
#define SPI0_RX_DMA       0x0B
#define SPI1_TX_DMA       0x0C
#define SPI1_RX_DMA       0x0D
#define I2C0_TX_DMA       0x12
#define I2C0_RX_DMA       0x13
#define I2C1_TX_DMA       0x14
#define I2C1_RX_DMA       0x15
#define ADC_DMA           0x16

/***************************************************************************
 **
 **  S3FM02G interrupt source number
 **
 ***************************************************************************/
#define WDTINT            0x00    /* Watch-dog Timer Interrupt */
#define CMINT             0x01    /* Clock Manager Interrupt */
#define PFCINT            0x02    /* Program Flash Controller Interrupt */
#define DFCINT            0x03    /* Data Flash Controller Interrupt */
#define DMAINT            0x04    /* DMA Controller Interrupt */
#define FRTINT            0x05    /* Free-running Timer Interrupt */
#define WSI0INT           0x06    /* Wakeup source 0 */
#define WSI1INT           0x07    /* Wakeup source 1 */
#define IMC0INT           0x08    /* Inverter Motor Controller 0 Interrupt */
#define ENC0INT           0x09    /* Encoder Counter 0 Interrupt */
#define IMC1INT           0x0A    /* Inverter Motor Controller 1 Interrupt */
#define ENC1INT           0x0B    /* Encoder Counter 1 Interrupt */
#define CAN0INT           0x0C    /* CAN0 Interrupt */
#define USART0INT         0x0D    /* USART0 Interrupt */
#define ADC0INT           0x0E    /* ADC0 Interrupt */
#define ADC1INT           0x0F    /* ADC1 Interrupt */
#define SSP0INT           0x10    /* SSP0 Interrupt */
#define I2C0INT           0x11    /* I2C0 Interrupt */
#define TC0INT            0x12    /* Timer/Counter0 Interrupt */
#define PWM0INT           0x13    /* PWM0 Interrupt */
#define WSI2INT           0x14    /* Wakeup source 2 */
#define WSI3INT           0x15    /* Wakeup source 3 */
#define TC1INT            0x16    /* Timer/Counter1 Interrupt */
#define PWM1INT           0x17    /* PWM1 Interrupt */
#define USART1INT         0x18    /* USART1 Interrupt */
#define SSP1INT           0x19    /* SSP1 Interrupt */
#define I2C1INT           0x1A    /* I2C1 Interrupt */
#define CAN1INT           0x1B    /* CAN1 Interrupt */
#define STTINT            0x1C    /* STT Interrupt */
#define USART2INT         0x1D    /* USART2 Interrupt */
#define TC2INT            0x1E    /* Timer/Counter2 Interrupt */
#define TC3INT            0x1F    /* Timer/Counter3 Interrupt */
#define PWM2INT           0x20    /* PWM2 Interrupt */
#define WSI4INT           0x21    /* Wakeup source 4 */
#define WSI5INT           0x22    /* Wakeup source 5 */
#define PWM3INT           0x23    /* PWM3 Interrupt */
#define USART3INT         0x24    /* USART4 Interrupt */
#define GPIO0INT          0x25    /* GPIO 0 Interrupt */
#define GPIO1INT          0x26    /* GPIO 1 Interrupt */
#define TC4INT            0x27    /* Timer/Counter4 Interrupt */
#define WSI6INT           0x28    /* Wakeup source 6 */
#define WSI7INT           0x29    /* Wakeup source 7 */
#define PWM4INT           0x2A    /* PWM4 Interrupt */
#define TC5INT            0x2B    /* Timer/Counter5 Interrupt */
#define WSI8INT           0x2C    /* Wakeup source 8 */
#define WSI9INT           0x2D    /* Wakeup source 9 */
#define WSI10INT          0x2E    /* Wakeup source 10 */
#define WSI11INT          0x2F    /* Wakeup source 11 */
#define PWM5INT           0x30    /* PWM5 Interrupt */
#define TC6INT            0x31    /* Timer/Counter6 Interrupt */
#define WSI12INT          0x32    /* Wakeup source 12 */
#define WSI13INT          0x33    /* Wakeup source 13 */
#define WSI14INT          0x34    /* Wakeup source 14 */
#define WSI15INT          0x35    /* Wakeup source 15 */
#define PWM6INT           0x36    /* PWM6 Interrupt */
#define TC7INT            0x37    /* Timer/Counter7 Interrupt */
#define PWM7INT           0x38    /* PWM8 Interrupt */
#define GPIO2INT          0x39    /* GPIO 2 Interrupt */
#define GPIO3INT          0x3A    /* GPIO 3 Interrupt */
#define OPAMPINT          0x3B    /* OP-AMP interrupt */

#endif    /* __IOS3FM02G_H */

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
Interrupt9   = WDTINT         0x40
Interrupt10  = CMINT          0x44
Interrupt11  = PFCINT         0x48
Interrupt12  = DFCINT         0x4C
Interrupt13  = DMAINT         0x50
Interrupt14  = FRTINT         0x54
Interrupt25  = WSI0INT        0x58
Interrupt26  = WSI1INT        0x5C
Interrupt17  = IMC0INT        0x60
Interrupt18  = ENC0INT        0x64
Interrupt19  = IMC1INT        0x68
Interrupt20  = ENC1INT        0x6C
Interrupt21  = CAN0INT        0x70
Interrupt22  = USART0INT      0x74
Interrupt23  = ADC0INT        0x78
Interrupt24  = ADC1INT        0x7C
Interrupt25  = SSP0INT        0x80
Interrupt26  = I2C0INT        0x84
Interrupt27  = TC0INT         0x88
Interrupt28  = PWM0INT        0x8C
Interrupt29  = WSI2INT        0x90
Interrupt30  = WSI3INT        0x94
Interrupt31  = TC1INT         0x98
Interrupt32  = PWM1INT        0x9C
Interrupt33  = USART1INT      0xA0
Interrupt34  = SSP1INT        0xA4
Interrupt35  = I2C1INT        0xA8
Interrupt36  = CAN1INT        0xAC
Interrupt37  = STTINT         0xB0
Interrupt38  = USART2INT      0xB4
Interrupt39  = TC2INT         0xB8
Interrupt40  = TC3INT         0xBC
Interrupt41  = PWM2INT        0xC0
Interrupt42  = WSI4INT        0xC4
Interrupt43  = WSI5INT        0xC8
Interrupt44  = PWM3INT        0xCC
Interrupt45  = USART3INT      0xD0
Interrupt46  = GPIO0INT       0xD4
Interrupt47  = GPIO1INT       0xD8
Interrupt48  = TC4INT         0xDC
Interrupt49  = WSI6INT        0xE0
Interrupt50  = WSI7INT        0xE4
Interrupt51  = PWM4INT        0xE8
Interrupt52  = TC5INT         0xEC
Interrupt53  = WSI8INT        0xF0
Interrupt54  = WSI9INT        0xF4
Interrupt55  = WSI10INT       0xF8
Interrupt56  = WSI11INT       0xFC
Interrupt57  = PWM5INT        0x100
Interrupt58  = TC6INT         0x104
Interrupt59  = WSI12INT       0x108
Interrupt60  = WSI13INT       0x10C
Interrupt61  = WSI14INT       0x110
Interrupt62  = WSI15INT       0x114
Interrupt63  = PWM6INT        0x118
Interrupt64  = TC7INT         0x11C
Interrupt65  = PWM7INT        0x120
Interrupt66  = GPIO2INT       0x124
Interrupt67  = GPIO3INT       0x128
Interrupt68  = OPAMPINT       0x12C

###DDF-INTERRUPT-END###*/
