/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM342FYXBG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 47013 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM342FYXBG_H
#define __IOTMPM342FYXBG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM342FYXBG SPECIAL FUNCTION REGISTERS
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
  __REG32  PSCSTOP  : 1;
  __REG32           : 10;
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
  __REG32  TMRDACLKEN    : 1;
  __REG32  TMRDBCLKEN    : 1;
  __REG32                : 2;
  __REG32  TMRDAGEAR     : 2;
  __REG32  TMRDBGEAR     : 2;
  __REG32                : 24;
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
  __REG32 INT10EN   : 1;
  __REG32           : 1;
  __REG32 EMST10    : 2;
  __REG32 EMCG10    : 3;
  __REG32           : 1;
  __REG32 INT11EN   : 1;
  __REG32           : 1;
  __REG32 EMST11    : 2;
  __REG32 EMCG11    : 3;
  __REG32           : 1;
  __REG32 INT12EN   : 1;
  __REG32           : 1;
  __REG32 EMST12    : 2;
  __REG32 EMCG12    : 3;
  __REG32           : 1;
  __REG32 INT13EN   : 1;
  __REG32           : 1;
  __REG32 EMST13    : 2;
  __REG32 EMCG13    : 3;
  __REG32           : 1;
} __cgimcge_bits;

/* CG Interrupt Mode Control Register F */
typedef struct {
  __REG32 INT14EN   : 1;
  __REG32           : 1;
  __REG32 EMST14    : 2;
  __REG32 EMCG14    : 3;
  __REG32           : 1;
  __REG32 INT15EN   : 1;
  __REG32           : 1;
  __REG32 EMST15    : 2;
  __REG32 EMCG15    : 3;
  __REG32           : 1;
  __REG32 INT16EN   : 1;
  __REG32           : 1;
  __REG32 EMST16    : 2;
  __REG32 EMCG16    : 3;
  __REG32           : 1;
  __REG32 INT17EN   : 1;
  __REG32           : 1;
  __REG32 EMST17    : 2;
  __REG32 EMCG17    : 3;
  __REG32           : 1;
} __cgimcgf_bits;

/* CG Interrupt Mode Control Register G */
typedef struct {
  __REG32 INT18EN   : 1;
  __REG32           : 1;
  __REG32 EMST18    : 2;
  __REG32 EMCG18    : 3;
  __REG32           : 1;
  __REG32 INT19EN   : 1;
  __REG32           : 1;
  __REG32 EMST19    : 2;
  __REG32 EMCG19    : 3;
  __REG32           : 17;
} __cgimcgg_bits;

/* CGICRCG Register */
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           : 27;
} __cgicrcg_bits;

/* CGRSTFLG Register */
typedef struct {
  __REG32 PSTPINRSTF    : 1;
  __REG32               : 1;
  __REG32 WDTRSTF       : 1;
  __REG32 STOP2RSTF     : 1;
  __REG32 DBGRSTF       : 1;
  __REG32               : 27;
} __cgrstflg_bits;


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

/* Interrupt Set-Enable Registers 64-86 */
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
  __REG32                 : 1;
  __REG32  SETENA84       : 1;
  __REG32  SETENA85       : 1;
  __REG32  SETENA86       : 1;
  __REG32                 : 9;
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

/* Interrupt Clear-Enable Registers 64-86 */
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
  __REG32                 : 1;
  __REG32  CLRENA84       : 1;
  __REG32  CLRENA85       : 1;
  __REG32  CLRENA86       : 1;
  __REG32                 : 9;
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
  __REG32                 : 1;
  __REG32  SETPEND84      : 1;
  __REG32  SETPEND85      : 1;
  __REG32  SETPEND86      : 1;
  __REG32                 : 9;
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
  __REG32                 : 1;
  __REG32  CLRPEND84      : 1;
  __REG32  CLRPEND85      : 1;
  __REG32  CLRPEND86      : 1;
  __REG32                 : 9;
} __clrpend2_bits;

/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_INT0       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT1       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT2       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT3       : 3;
} __pri0_bits;


/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_INT4       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT5       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT6       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT7       : 3;
} __pri1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_PSCSTOP    : 3;
  __REG32                 : 5;
  __REG32  PRI_PSCBRK     : 3;
  __REG32                 : 5;
  __REG32  PRI_PSCSTEP    : 3;
  __REG32                 : 5;
  __REG32  PRI_PSCII      : 3;
} __pri2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_PSCIA      : 3;
  __REG32                 : 5;
  __REG32  PRI_TB0        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB1        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB2        : 3;
} __pri3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TX0        : 3;
  __REG32                 : 5;
  __REG32  PRI_RX0        : 3;
  __REG32                 : 5;
  __REG32  PRI_TX1        : 3;
  __REG32                 : 5;
  __REG32  PRI_RX1        : 3;
} __pri4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TX2        : 3;
  __REG32                 : 5;
  __REG32  PRI_RX2        : 3;
  __REG32                 : 5;
  __REG32  PRI_TX3        : 3;
  __REG32                 : 5;
  __REG32  PRI_S          : 3;
} __pri5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_AD0HP      : 3;
  __REG32                 : 5;
  __REG32  PRI_AD0        : 3;
  __REG32                 : 5;
  __REG32  PRI_AD1HP      : 3;
  __REG32                 : 5;
  __REG32  PRI_AD1        : 3;
} __pri6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_DSADHP     : 3;
  __REG32                 : 5;
  __REG32  PRI_DSAD       : 3;
  __REG32                 : 5;
  __REG32  PRI_I2C        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB3        : 3;
} __pri7_bits;

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TB4        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB5        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB6        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB7        : 3;
} __pri8_bits;

/* Interrupt Priority Registers 36-39 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TB8        : 3;
  __REG32                 : 5;
  __REG32  PRI_TB9        : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA0CMP0   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA0CMP1   : 3;
} __pri9_bits;

/* Interrupt Priority Registers 40-43 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TDA0CMP2   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA0CMP3   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA0CMP4   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA1CMP0   : 3;
} __pri10_bits;

/* Interrupt Priority Registers 44-47 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TDA1CMP1   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA1CMP2   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA1CMP3   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDA1CMP4   : 3;
} __pri11_bits;

/* Interrupt Priority Registers 48-51 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TDB0CMP0   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB0CMP1   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB0CMP2   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB0CMP3   : 3;
} __pri12_bits;

/* Interrupt Priority Registers 52-55 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TDB0CMP4   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB1CMP0   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB1CMP1   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB1CMP2   : 3;
} __pri13_bits;

/* Interrupt Priority Registers 56-59 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_TDB1CMP3   : 3;
  __REG32                 : 5;
  __REG32  PRI_TDB1CMP4   : 3;
  __REG32                 : 5;
  __REG32  PRI_EC0        : 3;
  __REG32                 : 5;
  __REG32  PRI_EC0OVF     : 3;
} __pri14_bits;

/* Interrupt Priority Registers 60-63 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_EC0DIR     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC0DT0     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC0DT1     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC0DT2     : 3;
} __pri15_bits;

/* Interrupt Priority Registers 64-67 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_EC0DT3     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC1        : 3;
  __REG32                 : 5;
  __REG32  PRI_EC1OVF     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC1DIR     : 3;
} __pri16_bits;

/* Interrupt Priority Registers 68-71 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_EC1DT0     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC1DT1     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC1DT2     : 3;
  __REG32                 : 5;
  __REG32  PRI_EC1DT3     : 3;
} __pri17_bits;

/* Interrupt Priority Registers 72-75 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_DMACATC    : 3;
  __REG32                 : 5;
  __REG32  PRI_DMACAERR   : 3;
  __REG32                 : 5;
  __REG32  PRI_DMACBTC    : 3;
  __REG32                 : 5;
  __REG32  PRI_DMACBERR   : 3;
} __pri18_bits;

/* Interrupt Priority Registers 76-79 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_SSP        : 3;
  __REG32                 : 5;
  __REG32  PRI_VTX        : 3;
  __REG32                 : 5;
  __REG32  PRI_VRX        : 3;
  __REG32                 : 8;
} __pri19_bits;

/* Interrupt Priority Registers 80-83 */
typedef struct {
  __REG32                 : 5;
  __REG32  PRI_INT8       : 3;
  __REG32                 : 5;
  __REG32  PRI_INT9       : 3;
  __REG32                 : 5;
  __REG32  PRI_INTA       : 3;
  __REG32                 : 8;
} __pri20_bits;

/* Vector Table Offset Register */
typedef struct {
  __REG32                 : 7;
  __REG32  TBLOFF         :24;
  __REG32  TBLBASE        : 1;
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

/*PORT A Open-Drain Control Register */
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

/*PORT B Function Register 2*/
typedef struct {
  __REG8         : 1;
  __REG8  PB1F2  : 1;
  __REG8  PB2F2  : 1;
  __REG8         : 5;
} __pbfr2_bits;

/*PORT B Function Register 3*/
typedef struct {
  __REG8         : 2;
  __REG8  PB2F3  : 1;
  __REG8         : 5;
} __pbfr3_bits;

/*Port B open drain control register*/
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

/*PORT C Control Register */
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

/*PORT C Function Register 1*/
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

/*PORT C Function Register 2*/
typedef struct {
  __REG8  PC0F2  : 1;
  __REG8  PC1F2  : 1;
  __REG8  PC2F2  : 1;
  __REG8  PC3F2  : 1;
  __REG8         : 4;
} __pcfr2_bits;

/*PORT C Function Register 3*/
typedef struct {
  __REG8         : 2;
  __REG8  PC2F3  : 1;
  __REG8         : 5;
} __pcfr3_bits;

/*Port C open drain control register*/
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
  __REG8  PD7  : 1;
} __pd_bits;

/*PORT D Control Register */
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

/*PORT D Function Register 1*/
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

/*PORT D Function Register 2*/
typedef struct {
  __REG8  PD0F2  : 1;
  __REG8  PD1F2  : 1;
  __REG8  PD2F2  : 1;
  __REG8  PD3F2  : 1;
  __REG8  PD4F2  : 1;
  __REG8  PD5F2  : 1;
  __REG8  PD6F2  : 1;
  __REG8  PD7F2  : 1;
} __pdfr2_bits;

/*PORT D Function Register 3*/
typedef struct {
  __REG8         : 2;
  __REG8  PD2F3  : 1;
  __REG8         : 5;
} __pdfr3_bits;

/*Port D open drain control register*/
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
  __REG8       : 3;
} __pe_bits;

/*PORT E Control Register */
typedef struct {
  __REG8  PE0C  : 1;
  __REG8  PE1C  : 1;
  __REG8  PE2C  : 1;
  __REG8  PE3C  : 1;
  __REG8  PE4C  : 1;
  __REG8        : 3;
} __pecr_bits;

/*PORT E Function Register 1*/
typedef struct {
  __REG8  PE0F1  : 1;
  __REG8  PE1F1  : 1;
  __REG8  PE2F1  : 1;
  __REG8         : 5;
} __pefr1_bits;

/*PORT E Function Register 2*/
typedef struct {
  __REG8         : 2;
  __REG8  PE2F2  : 1;
  __REG8  PE3F2  : 1;
  __REG8  PE4F2  : 1;
  __REG8         : 3;
} __pefr2_bits;

/*PORT E Open Drain Control Register */
typedef struct {
  __REG8  PE0OD  : 1;
  __REG8  PE1OD  : 1;
  __REG8  PE2OD  : 1;
  __REG8  PE3OD  : 1;
  __REG8  PE4OD  : 1;
  __REG8         : 3;
} __peod_bits;

/*PORT E Pull-Up Control Register */
typedef struct {
  __REG8  PE0UP  : 1;
  __REG8  PE1UP  : 1;
  __REG8  PE2UP  : 1;
  __REG8  PE3UP  : 1;
  __REG8  PE4UP  : 1;
  __REG8         : 3;
} __pepup_bits;

/*PORT E Input Enable Control Register */
typedef struct {
  __REG8  PE0IE  : 1;
  __REG8  PE1IE  : 1;
  __REG8  PE2IE  : 1;
  __REG8  PE3IE  : 1;
  __REG8  PE4IE  : 1;
  __REG8         : 3;
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
  __REG8       : 2;
} __ph_bits;

/*PORT H Control Register 1*/
typedef struct {
  __REG8  PH0C  : 1;
  __REG8  PH1C  : 1;
  __REG8  PH2C  : 1;
  __REG8  PH3C  : 1;
  __REG8  PH4C  : 1;
  __REG8  PH5C  : 1;
  __REG8        : 2;
} __phcr_bits;

/*PORT H Function Register 1*/
typedef struct {
  __REG8  PH0F1  : 1;
  __REG8  PH1F1  : 1;
  __REG8  PH2F1  : 1;
  __REG8  PH3F1  : 1;
  __REG8  PH4F1  : 1;
  __REG8  PH5F1  : 1;
  __REG8         : 2;
} __phfr1_bits;

/*PORT H Pull-Up Control Register */
typedef struct {
  __REG8  PH0UP  : 1;
  __REG8         : 1;
  __REG8  PH2UP  : 1;
  __REG8  PH3UP  : 1;
  __REG8  PH4UP  : 1;
  __REG8  PH5UP  : 1;
  __REG8         : 2;
} __phpup_bits;

/*PORT H Pull-Down Control Register */
typedef struct {
  __REG8         : 1;
  __REG8  PH1DN  : 1;
  __REG8         : 6;
} __phpdn_bits;

/*PORT H Input Enable Control Register */
typedef struct {
  __REG8  PH0IE  : 1;
  __REG8  PH1IE  : 1;
  __REG8  PH2IE  : 1;
  __REG8  PH3IE  : 1;
  __REG8  PH4IE  : 1;
  __REG8  PH5IE  : 1;
  __REG8         : 2;
} __phie_bits;

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
  __REG8  PK3  : 1;
  __REG8  PK4  : 1;
  __REG8  PK5  : 1;
  __REG8  PK6  : 1;
  __REG8  PK7  : 1;
} __pk_bits;

/*PORT K Control Register*/
typedef struct {
  __REG8  PK0C  : 1;
  __REG8  PK1C  : 1;
  __REG8  PK2C  : 1;
  __REG8  PK3C  : 1;
  __REG8  PK4C  : 1;
  __REG8  PK5C  : 1;
  __REG8  PK6C  : 1;
  __REG8  PK7C  : 1;
} __pkcr_bits;

/*PORT K Function Register 1*/
typedef struct {
  __REG8  PK0F1  : 1;
  __REG8  PK1F1  : 1;
  __REG8  PK2F1  : 1;
  __REG8  PK3F1  : 1;
  __REG8  PK4F1  : 1;
  __REG8  PK5F1  : 1;
  __REG8  PK6F1  : 1;
  __REG8  PK7F1  : 1;
} __pkfr1_bits;

/*PORT K Function Register 2*/
typedef struct {
  __REG8  PK0F2  : 1;
  __REG8  PK1F2  : 1;
  __REG8  PK2F2  : 1;
  __REG8  PK3F2  : 1;
  __REG8         : 3;
  __REG8  PK7F2  : 1;
} __pkfr2_bits;

/*PORT K Function Register 3*/
typedef struct {
  __REG8  PK0F3  : 1;
  __REG8  PK1F3  : 1;
  __REG8  PK2F3  : 1;
  __REG8  PK3F3  : 1;
  __REG8         : 4;
} __pkfr3_bits;


/*PORT L Register*/
typedef struct {
  __REG8  PL0  : 1;
  __REG8  PL1  : 1;
  __REG8  PL2  : 1;
  __REG8  PL3  : 1;
  __REG8  PL4  : 1;
  __REG8  PL5  : 1;
  __REG8  PL6  : 1;
  __REG8       : 1;
} __pl_bits;

/*PORT L Control Register*/
typedef struct {
  __REG8  PL0C  : 1;
  __REG8  PL1C  : 1;
  __REG8  PL2C  : 1;
  __REG8  PL3C  : 1;
  __REG8  PL4C  : 1;
  __REG8  PL5C  : 1;
  __REG8  PL6C  : 1;
  __REG8        : 1;
} __plcr_bits;

/*PORT L Function Register 1*/
typedef struct {
  __REG8  PL0F1  : 1;
  __REG8  PL1F1  : 1;
  __REG8  PL2F1  : 1;
  __REG8  PL3F1  : 1;
  __REG8  PL4F1  : 1;
  __REG8  PL5F1  : 1;
  __REG8  PL6F1  : 1;
  __REG8         : 1;
} __plfr1_bits;


/*PORT M Register*/
typedef struct {
  __REG8  PM0  : 1;
  __REG8  PM1  : 1;
  __REG8       : 6;
} __pm_bits;

/*PORT M Function Register 1*/
typedef struct {
  __REG8  PM0F1  : 1;
  __REG8  PM1F1  : 1;
  __REG8         : 6;
} __pmfr1_bits;

/*PORT M Function Register 2*/
typedef struct {
  __REG8  PM0F2  : 1;
  __REG8  PM1F2  : 1;
  __REG8         : 6;
} __pmfr2_bits;

/*PORT M Pull-Up Control Register */
typedef struct {
  __REG8  PM0UP  : 1;
  __REG8  PM1UP  : 1;
  __REG8         : 6;
} __pmpup_bits;

/*PORT M Input Enable Control Register */
typedef struct {
  __REG8  PM0IE  : 1;
  __REG8  PM1IE  : 1;
  __REG8         : 6;
} __pmie_bits;


/*PSC SGN Register*/
typedef struct {
  __REG32         : 16;
  __REG32  SA0    : 1;
  __REG32  SM0    : 1;
  __REG32  SM1    : 1;
  __REG32  SL0    : 1;
  __REG32  SL1    : 1;
  __REG32  SR0    : 1;
  __REG32  SR1    : 1;
  __REG32         : 9;
} __sgn_bits;

/*PSC CNT Register*/
typedef struct {
  __REG32         : 16;
  __REG32  START  : 1;
  __REG32  TENB   : 1;
  __REG32         : 6;
  __REG32  STEP   : 1;
  __REG32  BRK    : 1;
  __REG32         : 6;
} __psccnt_bits;

/*PSC FLG Register*/
typedef struct {
  __REG32         : 16;
  __REG32  OVER   : 1;
  __REG32  UNDER  : 1;
  __REG32         : 6;
  __REG32  ZERO   : 1;
  __REG32         : 7;
} __pscflg_bits;

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

/* DMAC-A Software Single Request Register */
typedef struct {
  __REG32 SOFTSREQ0     : 1;
  __REG32 SOFTSREQ1     : 1;
  __REG32               :30;
} __dmacasoftsreq_bits;

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
  __REG32  TBCLK    : 3;
  __REG32  TBCLE    : 1;
  __REG32           :28;
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
  __REG32  INTOVF   : 1;
  __REG32           :29;
} __tbxst_bits;

/*TMRB interrupt mask register (channels 0 through 9)*/
typedef struct {
  __REG32  TBIMCMP0 : 1;
  __REG32  TBIMCMP1 : 1;
  __REG32  TBIMOVF  : 1;
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

/*TMRB DMA enable register (channels 0 through 9)*/
typedef struct {
  __REG32           : 2;
  __REG32  TBDMAEN2 : 1;
  __REG32           :29;
} __tbxdma_bits;

/*ENC enable register*/
typedef struct {
  __REG32  ENCEN    : 1;
  __REG32           :31;
} __encxen_bits;

/*ENC control register*/
typedef struct {
  __REG32  MA1DN    : 3;
  __REG32           : 1;
  __REG32  MA1UP    : 3;
  __REG32           : 1;
  __REG32  BRCK     : 2;
  __REG32           : 1;
  __REG32  PBDIR    : 1;
  __REG32  MA12     : 1;
  __REG32  MA2DIR   : 1;
  __REG32  ENCNF    : 2;
  __REG32           :16;
} __encxcnt_bits;

/*ENC interrupt enable register*/
typedef struct {
  __REG32  INTECNCP0     : 1;
  __REG32  INTEC0CP1     : 1;
  __REG32  INTEC0OVF     : 1;
  __REG32  INTEC0UDF     : 1;
  __REG32  INTEC0DT0     : 1;
  __REG32  INTEC0DT1     : 1;
  __REG32  INTEC0DT2     : 1;
  __REG32  INTEC0DT3     : 1;
  __REG32  INTEC0DIR     : 1;
  __REG32  INTECUOVF     : 1;
  __REG32                :22;
} __encxie_bits;

/*ENC status register*/
typedef struct {
  __REG32  CMP0F     : 1;
  __REG32  CMP1F     : 1;
  __REG32  OVF       : 1;
  __REG32  UDF       : 1;
  __REG32  SB0F      : 1;
  __REG32  SB1F      : 1;
  __REG32  SB2F      : 1;
  __REG32  SB3F      : 1;
  __REG32  DIRF      : 1;
  __REG32            :23;
} __encxflg_bits;

/*ENC 16-bit counter run register*/
typedef struct {
  __REG32  ENCRUN    : 1;
  __REG32  ENCCLR    : 1;
  __REG32            :30;
} __encxarun_bits;

/*ENC pulse counter compare 0 register*/
typedef struct {
  __REG32  ENCCMP0   :16;
  __REG32            :16;
} __encxacp0_bits;

/*ENC pulse counter compare 1 register*/
typedef struct {
  __REG32  ENCCMP1   :16;
  __REG32            :16;
} __encxacp1_bits;

/*ENC 16-bit counter read register*/
typedef struct {
  __REG32  ENCDAT    :16;
  __REG32            :16;
} __encxadat_bits;

/*ENC 24-bit counter run register*/
typedef struct {
  __REG32  T24RUN    : 1;
  __REG32            :31;
} __encxbrun_bits;

/*ENC DMAE request enable register*/
typedef struct {
  __REG32  DMADT0    : 1;
  __REG32  DMADT1    : 1;
  __REG32  DMADT2    : 1;
  __REG32  DMADT3    : 1;
  __REG32            :28;
} __encxbdma_bits;

/*ENC 24-bit counter read register*/
typedef struct {
  __REG32  UC    :24;
  __REG32        : 8;
} __encxbuc_bits;

/*ENC capture 00 register*/
typedef struct {
  __REG32  CAP00     :24;
  __REG32  OVF00     : 1;
  __REG32            : 7;
} __encxbcap00_bits;

/*ENC capture 10 register*/
typedef struct {
  __REG32  CAP10     :24;
  __REG32  OVF10     : 1;
  __REG32            : 7;
} __encxbcap10_bits;

/*ENC capture 20 register*/
typedef struct {
  __REG32  CAP20     :24;
  __REG32  OVF20     : 1;
  __REG32            : 7;
} __encxbcap20_bits;

/*ENC capture 30 register*/
typedef struct {
  __REG32  CAP30     :24;
  __REG32  OVF30     : 1;
  __REG32            : 7;
} __encxbcap30_bits;

/*ENC cycle 0 register*/
typedef struct {
  __REG32  B0DAT     :24;
  __REG32            : 8;
} __encxbodat_bits;

/*ENC cycle 1 register*/
typedef struct {
  __REG32  B1DAT     :24;
  __REG32            : 8;
} __encxb1dat_bits;

/*ENC cycle 2 register*/
typedef struct {
  __REG32  B2DAT     :24;
  __REG32            : 8;
} __encxb2dat_bits;

/*ENC cycle 3 register*/
typedef struct {
  __REG32  B3DAT     :24;
  __REG32            : 8;
} __encxb3dat_bits;

/*ENC cycle common register*/
typedef struct {
  __REG32  BCDAT     :24;
  __REG32            : 8;
} __encxbcdat_bits;

/*ENC phase difference 0 register*/
typedef struct {
  __REG32  B0PDT     :24;
  __REG32            : 8;
} __encxb0pdt_bits;

/*ENC phase difference 1 register*/
typedef struct {
  __REG32  B1PDT     :24;
  __REG32            : 8;
} __encxb1pdt_bits;

/*ENC phase difference 2 register*/
typedef struct {
  __REG32  B2PDT     :24;
  __REG32            : 8;
} __encxb2pdt_bits;

/*ENC phase difference 3 register*/
typedef struct {
  __REG32  B3PDT     :24;
  __REG32            : 8;
} __encxb3pdt_bits;

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

/*TD0 Control Register*/
typedef struct {
  __REG32  TDISO0   : 1;
  __REG32  TDISO1   : 1;
  __REG32  TDRDE    : 1;
  __REG32           : 1;
  __REG32  TDMDPT0  : 1;
  __REG32  TDMDCY00 : 3;
  __REG32  TDMDPT01 : 1;
  __REG32  TDMDCY01 : 3;
  __REG32           :20;
} __tdxcr0_bits;

/*TD1 Control Register*/
typedef struct {
  __REG32  TDISO0   : 1;
  __REG32  TDISO1   : 1;
  __REG32  TDRDE    : 1;
  __REG32           : 1;
  __REG32  TDMDPT1  : 1;
  __REG32  TDMDCY10 : 3;
  __REG32  TDMDPT11 : 1;
  __REG32  TDMDCY11 : 3;
  __REG32           :20;
} __tdxcr1_bits;

/*TD RUN Register*/
typedef struct {
  __REG32  TDRUN    : 1;
  __REG32           :31;
} __tdxrun_bits;

/*TD0 BCR Register*/
typedef struct {
  __REG32  TDSFT00  : 1;
  __REG32  TDSFT01  : 1;
  __REG32  TDSFT10  : 1;
  __REG32  TDSFT11  : 1;
  __REG32  PHSCHG   : 1;
  __REG32           :27;
} __tdxbcr_bits;

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
  __REG32  TDMDRT   : 4;
  __REG32  TDRG2    :16;
  __REG32           :12;
} __tdxrg2_bits;

/*TD Compare Register 2*/
typedef struct {
  __REG32  CPMDRT   : 4;
  __REG32  CPRG2    :16;
  __REG32           :12;
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
  __REG32  TDMDRT   : 4;
  __REG32  TDRG4    :16;
  __REG32           :12;
} __tdxrg4_bits;

/*TD Compare Register 4*/
typedef struct {
  __REG32  CPMDRT   : 4;
  __REG32  CPRG4    :16;
  __REG32           :12;
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

/*Watchdog Timer Mode Register*/
typedef struct {
  __REG8          : 1;
  __REG8  RESCR   : 1;
  __REG8  I2WDT   : 1;
  __REG8          : 1;
  __REG8  WDTP    : 3;
  __REG8  WDTE    : 1;
} __wdmod_bits;


/*SIOx Enable register*/
typedef struct {
  __REG8  SIOE     : 1;
  __REG8           : 7;
} __scxen_bits;

/*SIOx Control register*/
typedef struct {
  __REG8  IOC      : 1;
  __REG8  SCLKS    : 1;
  __REG8  FERR     : 1;
  __REG8  PERR     : 1;
  __REG8  OERR     : 1;
  __REG8  PE       : 1;
  __REG8  EVEN     : 1;
  __REG8  RB8      : 1;
} __scxcr_bits;

/*SIOx Mode control register 0*/
typedef struct {
  __REG8  SC       : 2;
  __REG8  SM       : 2;
  __REG8  WU       : 1;
  __REG8  RXE      : 1;
  __REG8  CTSE     : 1;
  __REG8  TB8      : 1;
} __scxmod0_bits;

/*SIOx Baud rate generator control register*/
typedef struct {
  __REG8  BRS      : 4;
  __REG8  BRCK     : 2;
  __REG8  BRADDE   : 1;
  __REG8           : 1;
} __scxbrcr_bits;

/*SIOx Baud rate generator control register 2*/
typedef struct {
  __REG8  BRK      : 4;
  __REG8           : 4;
} __scxbradd_bits;

/*SIOx Mode control register 1*/
typedef struct {
  __REG8           : 1;
  __REG8  SINT     : 3;
  __REG8  TXE      : 1;
  __REG8  FDPX     : 2;
  __REG8  I2SC     : 1;
} __scxmod1_bits;

/*SIOx Mode control register 2*/
typedef struct {
  __REG8  SWRST    : 2;
  __REG8  WBUF     : 1;
  __REG8  DRCHG    : 1;
  __REG8  SBLEN    : 1;
  __REG8  TXRUN    : 1;
  __REG8  RBFLL    : 1;
  __REG8  TBEMP    : 1;
} __scxmod2_bits;

/*SIOx RX FIFO configuration register*/
typedef struct {
  __REG8  RIL      : 2;
  __REG8           : 4;
  __REG8  RFIS     : 1;
  __REG8  RFCS     : 1;
} __scxrfc_bits;

/*SIOx TX FIFO configuration register*/
typedef struct {
  __REG8  TIL      : 2;
  __REG8           : 4;
  __REG8  TFIS     : 1;
  __REG8  TFCS     : 1;
} __scxtfc_bits;

/*SIOx RX FIFO status register*/
typedef struct {
  __REG8  RLVL     : 3;
  __REG8           : 4;
  __REG8  ROR      : 1;
} __scxrst_bits;

/*SIOx TX FIFO status register*/
typedef struct {
  __REG8  TLVL     : 3;
  __REG8           : 4;
  __REG8  TUR      : 1;
} __scxtst_bits;

/*SIOx FIFO configuration register*/
typedef struct {
  __REG8  CNFG     : 1;
  __REG8  RXTXCNT  : 1;
  __REG8  RFIE     : 1;
  __REG8  TFIE     : 1;
  __REG8  RFST     : 1;
  __REG8           : 3;
} __scxfcnf_bits;

/*SIOx DMA enable register*/
typedef struct {
  __REG8  DMAEN0   : 1;
  __REG8  DMAEN1   : 1;
  __REG8           : 6;
} __scxdma_bits;

/* UARTDR (UART Data Register) */
typedef struct{
__REG32 DATA                  : 8;
__REG32 FE                    : 1;
__REG32 PE                    : 1;
__REG32 BE                    : 1;
__REG32 OE                    : 1;
__REG32                       :20;
} __uartdr_bits;

/* UARTSR (UART receive status register) */
typedef struct
{
  __REG32 FE                    : 1;
  __REG32 PE                    : 1;
  __REG32 BE                    : 1;
  __REG32 OE                    : 1;
  __REG32                       :28;
} __uartsr_bits;

/* UARTFR (UART flag register) */
typedef struct
{
  __REG32 CTS                   : 1;
  __REG32                       : 2;
  __REG32 BUSY                  : 1;
  __REG32 RXFE                  : 1;
  __REG32 TXFF                  : 1;
  __REG32 RXFF                  : 1;
  __REG32 TXFE                  : 1;
  __REG32                       :24;
} __uartfr_bits;

/* UARTIBRD (UART integer baud rate divisor register) */
typedef struct
{
  __REG32 BAUDDIVINT            :16;
  __REG32                       :16;
} __uartibrd_bits;

/* UARTFBRD (UART fractional baud rate divisor register) */
typedef struct
{
  __REG32 BAUDDIVFRAC           : 6;
  __REG32                       :26;
} __uartfbrd_bits;

/* UARTLCR_H (UART line control register) */
typedef struct
{
  __REG32 BRK                   : 1;
  __REG32 PEN                   : 1;
  __REG32 EPS                   : 1;
  __REG32 STP2                  : 1;
  __REG32 FEN                   : 1;
  __REG32 WLEN                  : 2;
  __REG32 SPS                   : 1;
  __REG32                       :24;
} __uartlcr_h_bits;

/* UARTCR (UART control register) */
typedef struct
{
  __REG32 UARTEN                : 1;
  __REG32                       : 7;
  __REG32 TXE                   : 1;
  __REG32 RXE                   : 1;
  __REG32                       : 1;
  __REG32 RTS                   : 1;
  __REG32                       : 2;
  __REG32 RTSEN                 : 1;
  __REG32 CTSEN                 : 1;
  __REG32                       :16;
} __uartcr_bits;

/* UARTIFLS (UART interrupt FIFO level select register) */
typedef struct
{
  __REG32 TXIFLSEL              : 3;
  __REG32 RXIFLSEL              : 3;
  __REG32                       :26;
} __uartifls_bits;

/* UARTIMSC (UART interrupt mask set/clear register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSMIM                : 1;
  __REG32                       : 2;
  __REG32 RXIM                  : 1;
  __REG32 TXIM                  : 1;
  __REG32 RTIM                  : 1;
  __REG32 FEIM                  : 1;
  __REG32 PEIM                  : 1;
  __REG32 BEIM                  : 1;
  __REG32 OEIM                  : 1;
  __REG32                       :21;
} __uartimsc_bits;

/* UARTRIS (UART raw interrupt status register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSRMIS               : 1;
  __REG32                       : 2;
  __REG32 RXRIS                 : 1;
  __REG32 TXRIS                 : 1;
  __REG32 RTRIS                 : 1;
  __REG32 FERIS                 : 1;
  __REG32 PERIS                 : 1;
  __REG32 BERIS                 : 1;
  __REG32 OERIS                 : 1;
  __REG32                       :21;
} __uartris_bits;

/* UARTMIS (UART masked interrupt status register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSMMIS               : 1;
  __REG32                       : 2;
  __REG32 RXMIS                 : 1;
  __REG32 TXMIS                 : 1;
  __REG32 RTMIS                 : 1;
  __REG32 FEMIS                 : 1;
  __REG32 PEMIS                 : 1;
  __REG32 BEMIS                 : 1;
  __REG32 OEMIS                 : 1;
  __REG32                       :21;
} __uartmis_bits;

/* UARTICR (UART interrupt clear register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSMIC                : 1;
  __REG32                       : 2;
  __REG32 RXIC                  : 1;
  __REG32 TXIC                  : 1;
  __REG32 RTIC                  : 1;
  __REG32 FEIC                  : 1;
  __REG32 PEIC                  : 1;
  __REG32 BEIC                  : 1;
  __REG32 OEIC                  : 1;
  __REG32                       :21;
} __uarticr_bits;

/* UARTDMACR (UART DMA control register) */
typedef struct
{
  __REG32 RXDMAE                : 1;
  __REG32 TXDMAE                : 1;
  __REG32 DMAONERR              : 1;
  __REG32                       :29;
} __uartdmacr_bits;


/*Serial bus control register 0*/
typedef struct {
  __REG32           : 7;
  __REG32  SBIEN    : 1;
  __REG32           :24;
} __sbicr0_bits;

/*Serial bus control register 1*/
typedef union {
  /*SBICR1*/
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
} __sbicr1_bits;

/*Serial bus interface data buffer register*/
typedef struct {
  __REG32  DB       : 8;
  __REG32           :24;
} __sbidbr_bits;

/*I2C bus address register*/
typedef struct {
  __REG32 ALS     : 1;
  __REG32 SA      : 7;
  __REG32         :24;
} __sbii2car_bits;

/*Serial bus control register 2*/
/*Serial bus status register*/
typedef union {
  /*SBICR2*/
  struct {
  __REG32 SWRST   : 2;
  __REG32 SBIM    : 2;
  __REG32 PIN     : 1;
  __REG32 BB      : 1;
  __REG32 TRX     : 1;
  __REG32 MST     : 1;
  __REG32         :24;
  };
  /*SBISR*/
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
  } __sbisr;
} __sbicr2_bits;

/*Serial bus interface baud rate register 0*/
typedef struct {
  __REG32         : 6;
  __REG32 I2SBI   : 1;
  __REG32         :25;
} __sbibr0_bits;


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

/*SSPDMACR (SSP DMA control register)*/
typedef struct {
  __REG32 RXDMAE  : 1;
  __REG32 TXDMAE  : 1;
  __REG32         :30;
} __sspdmacr_bits;

/*VSIO Enable register*/
typedef struct {
  __REG32 SIOE        : 1;
  __REG32             : 5;
  __REG32 SWRST0      : 1;
  __REG32 SWRST1      : 1;
  __REG32             :24;
} __vsioen_bits;

/*VSIO Control register 0*/
typedef struct {
  __REG32 DL          : 6;
  __REG32 DRCHG       : 1;
  __REG32 CKPH        : 1;
  __REG32 CS0PH       : 1;
  __REG32 CS1PH       : 1;
  __REG32             :22;
} __vsiocr0_bits;

/*VSIO Control register 1*/
typedef struct {
  __REG32 SC          : 2;
  __REG32 SM          : 2;
  __REG32 CSS         : 1;
  __REG32             : 2;
  __REG32 TRXE        : 1;
  __REG32             :24;
} __vsiocr1_bits;

/*VSIO Control register 2*/
typedef struct {
  __REG32 SINT        : 3;
  __REG32 CSHD        : 2;
  __REG32             : 1;
  __REG32 FDPX        : 2;
  __REG32             :24;
} __vsiocr2_bits;

/*VSIO Control register 3*/
typedef struct {
  __REG32 INTTXWE     : 1;
  __REG32 INTTXFE     : 1;
  __REG32 INTRXWE     : 1;
  __REG32 INTRXFE     : 1;
  __REG32 INTRXEE     : 1;
  __REG32             : 1;
  __REG32 DMATE       : 1;
  __REG32 DMARE       : 1;
  __REG32 TWEND       : 1;
  __REG32 RWEND       : 1;
  __REG32             : 1;
  __REG32 RXRUN       : 1;
  __REG32 TXRUN       : 1;
  __REG32 ORER        : 1;
  __REG32 RBFLL       : 1;
  __REG32 TBEMP       : 1;
  __REG32             :16;
} __vsiocr3_bits;

/*VSIO Baud rate generator control register*/
typedef struct {
  __REG32 BRS         : 4;
  __REG32 BRCK        : 4;
  __REG32             :24;
} __vsiobrcr_bits;

/*VSIO Receive FIFO buffer configuration register*/
typedef struct {
  __REG32 RIL         : 3;
  __REG32             : 4;
  __REG32 RFCS        : 1;
  __REG32             :24;
} __vsiorfc_bits;

/*VSIO Transmit FIFO buffer configuration register*/
typedef struct {
  __REG32 TIL         : 3;
  __REG32             : 4;
  __REG32 TFCS        : 1;
  __REG32             :24;
} __vsiotfc_bits;

/*VSIO Receive FIFO status register*/
typedef struct {
  __REG32 RLVL        : 3;
  __REG32             :29;
} __vsiorst_bits;

/*VSIO Transmit FIFO status register*/
typedef struct {
  __REG32 TLVL        : 3;
  __REG32             :29;
} __vsiotst_bits;


/*SIO3 Enable register*/
typedef struct {
  __REG32 SIOE     : 1;
  __REG32          :31;
} __sc3en_bits;

/*SIO3 Buffer register*/
typedef struct {
  __REG32 TB       : 8;
  __REG32          :24;
} __sc3buf_bits;

/*SIO3 Mode control register 0*/
typedef struct {
  __REG32 SC       : 2;
  __REG32 SM       : 2;
  __REG32          :28;
} __sc3mod0_bits;

/*SIO3 Baud rate generator control register*/
typedef struct {
  __REG32 BR3S     : 4;
  __REG32 BR3CK    : 2;
  __REG32          :26;
} __sc3brcr_bits;

/*SIO3 Mode control register 1*/
typedef struct {
  __REG32          : 1;
  __REG32 SINT     : 3;
  __REG32 TXE      : 1;
  __REG32 FDPX     : 2;
  __REG32          :25;
} __sc3mod1_bits;

/*SIO3 Mode control register 2*/
typedef struct {
  __REG32 SWRST    : 2;
  __REG32 WBUF     : 1;
  __REG32 DRCHG    : 1;
  __REG32          : 1;
  __REG32 TXRUN    : 1;
  __REG32          : 1;
  __REG32 TBEMP    : 1;
  __REG32          :24;
} __sc3mod2_bits;


/*A/D Conversion Clock Setting Register*/
typedef struct {
  __REG8  ADCLK   : 3;
  __REG8          : 1;
  __REG8  ADSH    : 4;
} __adclk_bits;

/*A/D Conversion Clock Setting Register*/
typedef struct {
  __REG8  ADCLK   : 3;
  __REG8          : 5;
} __dsclk_bits;

/*A/D Mode Control Register 0*/
typedef struct {
  __REG8  ADS     : 1;
  __REG8  HPADS   : 1;
  __REG8          : 6;
} __admod0_bits;

/*A/D Mode Control Register 1*/
typedef struct {
  __REG8  ADHWE   : 1;
  __REG8  ADHWS   : 1;
  __REG8  HPADHWE : 1;
  __REG8  HPADHWS : 1;
  __REG8          : 1;
  __REG8  RCUT    : 1;
  __REG8  I2AD    : 1;
  __REG8  DACON   : 1;
} __admod1_bits;

/*A/D Mode Control Register 1*/
typedef struct {
  __REG8  ADHWE   : 1;
  __REG8  ADHWS   : 1;
  __REG8  HPADHWE : 1;
  __REG8  HPADHWS : 1;
  __REG8          : 1;
  __REG8  MOD_EN  : 1;
  __REG8  I2AD    : 1;
  __REG8  BIAS_EN : 1;
} __dsmod1_bits;

/*A/D Mode Control Register 2*/
typedef struct {
  __REG8  ADCH    : 3;
  __REG8          : 1;
  __REG8  HPADCH  : 3;
  __REG8          : 1;
} __admod2_bits;

/*A/D Mode Control Register 2*/
typedef struct {
  __REG8  ADCH    : 4;
  __REG8  HPADCH  : 4;
} __dsmod2_bits;

/*A/D Mode Control Register 3*/
typedef struct {
  __REG8  SCAN    : 1;
  __REG8  REPEAT  : 1;
  __REG8          : 2;
  __REG8  ITM     : 3;
  __REG8          : 1;
} __admod3_bits;

/*A/D Mode Control Register 4*/
typedef struct {
  __REG8  SCANSTA   : 3;
  __REG8            : 1;
  __REG8  SCANAREA  : 3;
  __REG8            : 1;
} __admod4_bits;

/*A/D Mode Control Register 4*/
typedef struct {
  __REG8  SCANSTA   : 4;
  __REG8  SCANAREA  : 4;
} __dsmod4_bits;

/*A/D Mode Control Register 5*/
typedef struct {
  __REG8  ADBF    : 1;
  __REG8  EOCF    : 1;
  __REG8  HPADBF  : 1;
  __REG8  HPEOCF  : 1;
  __REG8          : 4;
} __admod5_bits;

/*A/D Mode Control Register 6*/
typedef struct {
  __REG8  ADRST   : 2;
  __REG8          : 6;
} __admod6_bits;

/*A/D Mode Control Register 7*/
typedef struct {
  __REG8  INTADDMA   : 1;
  __REG8  INTADHPDMA : 1;
  __REG8             : 6;
} __admod7_bits;

/*A/D Mode Control Register 8*/
typedef struct {
  __REG16  DSGAIN0   : 2;
  __REG16  DSGAIN1   : 2;
  __REG16  DSGAIN2   : 2;
  __REG16  DSGAIN3   : 2;
  __REG16  DSGAIN4   : 2;
  __REG16  DSGAIN5   : 2;
  __REG16            : 4;
} __dsmod8_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  ADR     :12;
  __REG32  ADRF    : 1;
  __REG32  ADOVRF  : 1;
  __REG32          :18;
} __adregx_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  ADR     :16;
  __REG32  ADRF    : 1;
  __REG32  ADOVRF  : 1;
  __REG32          :14;
} __dsregx_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  SPADR    :12;
  __REG32  ADRFSP   : 1;
  __REG32  ADOVRFSP : 1;
  __REG32           :18;
} __adregsp_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  SPADR    :16;
  __REG32  ADRFSP   : 1;
  __REG32  ADOVRFSP : 1;
  __REG32           :14;
} __dsregsp_bits;

/*A/D Conversion Comparison Control Register 0*/
typedef struct {
  __REG32  AINS0    : 3;
  __REG32           : 1;
  __REG32  ADBIG0   : 1;
  __REG32  CMPCOND0 : 1;
  __REG32           : 1;
  __REG32  CMP0EN   : 1;
  __REG32  CMPCNT0  : 4;
  __REG32           : 20;
} __adcmpcr0_bits;

/*A/D Conversion Comparison Control Register 1*/
typedef struct {
  __REG32  REGS1    : 3;
  __REG32           : 1;
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
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(SHIP0,             0xE000ED18,__READ_WRITE ,__ship0_bits);
__IO_REG32_BIT(SHIP1,             0xE000ED1C,__READ_WRITE ,__ship1_bits);
__IO_REG32_BIT(SHIP2,             0xE000ED20,__READ_WRITE ,__ship2_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);


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
__IO_REG32_BIT(CGIMCGG,             0x400F3058,__READ_WRITE ,__cgimcgg_bits);
__IO_REG32_BIT(CGICRCG,             0x400F3060,__READ_WRITE ,__cgicrcg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x400F3064,__READ_WRITE ,__cgrstflg_bits);


/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x400C0000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x400C0004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAFR1,                0x400C0008,__READ_WRITE ,__pafr1_bits);
__IO_REG8_BIT(PAFR2,                0x400C000C,__READ_WRITE ,__pafr2_bits);
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
__IO_REG8_BIT(PBFR2,                0x400C010C,__READ_WRITE ,__pbfr2_bits);
__IO_REG8_BIT(PBFR3,                0x400C0110,__READ_WRITE ,__pbfr3_bits);
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
__IO_REG8_BIT(PDCR,                 0x400C0304,__READ_WRITE ,__pdcr_bits);
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
__IO_REG8_BIT(PFPUP,                0x400C052C,__READ_WRITE ,__pfpup_bits);
__IO_REG8_BIT(PFIE,                 0x400C0538,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG8_BIT(PGDATA,               0x400C0600,__READ_WRITE ,__pg_bits);
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
__IO_REG8_BIT(PHPUP,                0x400C072C,__READ_WRITE ,__phpup_bits);
__IO_REG8_BIT(PHPDN,                0x400C0730,__READ_WRITE ,__phpdn_bits);
__IO_REG8_BIT(PHIE,                 0x400C0738,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x400C0900,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJPUP,                0x400C092C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJIE,                 0x400C0938,__READ_WRITE ,__pjie_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG8_BIT(PKDATA,               0x400C0A00,__READ_WRITE ,__pk_bits);
__IO_REG8_BIT(PKCR,                 0x400C0A04,__READ_WRITE ,__pkcr_bits);
__IO_REG8_BIT(PKFR1,                0x400C0A08,__READ_WRITE ,__pkfr1_bits);
__IO_REG8_BIT(PKFR2,                0x400C0A0C,__READ_WRITE ,__pkfr2_bits);
__IO_REG8_BIT(PKFR3,                0x400C0A10,__READ_WRITE ,__pkfr3_bits);

/***************************************************************************
 **
 ** PORTL
 **
 ***************************************************************************/
__IO_REG8_BIT(PLDATA,               0x400C0B00,__READ_WRITE ,__pl_bits);
__IO_REG8_BIT(PLCR,                 0x400C0B04,__READ_WRITE ,__plcr_bits);
__IO_REG8_BIT(PLFR1,                0x400C0B08,__READ_WRITE ,__plfr1_bits);

/***************************************************************************
 **
 ** PORTM
 **
 ***************************************************************************/
__IO_REG8_BIT(PMDATA,               0x400C0C00,__READ_WRITE ,__pm_bits);
__IO_REG8_BIT(PMFR1,                0x400C0C08,__READ_WRITE ,__pmfr1_bits);
__IO_REG8_BIT(PMFR2,                0x400C0C0C,__READ_WRITE ,__pmfr2_bits);
__IO_REG8_BIT(PMPUP,                0x400C0C2C,__READ_WRITE ,__pmpup_bits);
__IO_REG8_BIT(PMIE,                 0x400C0C38,__READ_WRITE ,__pmie_bits);

/***************************************************************************
 **
 ** PSC
 **
 ***************************************************************************/
__IO_REG32(    UA0,                 0x4000DC00,__READ_WRITE );
__IO_REG32(    UM0,                 0x4000DC04,__READ_WRITE );
__IO_REG32(    UM1,                 0x4000DC08,__READ_WRITE );
__IO_REG32(    UL0,                 0x4000DC0C,__READ_WRITE );
__IO_REG32(    UL1,                 0x4000DC10,__READ_WRITE );
__IO_REG32(    UR0,                 0x4000DC14,__READ_WRITE );
__IO_REG32(    UR1,                 0x4000DC18,__READ_WRITE );
__IO_REG32_BIT(SGN,                 0x4000DC1C,__READ_WRITE ,__sgn_bits);
__IO_REG32(    AP0,                 0x4000DC20,__READ_WRITE );
__IO_REG32(    AP1,                 0x4000DC24,__READ_WRITE );
__IO_REG32(    AP2,                 0x4000DC28,__READ_WRITE );
__IO_REG32(    BR0,                 0x4000DC2C,__READ_WRITE );
__IO_REG32(    PG0,                 0x4000DC30,__READ_WRITE );
__IO_REG32(    VG0,                 0x4000DC34,__READ_WRITE );
__IO_REG32_BIT(PSCCNT,              0x4000DD00,__READ_WRITE ,__psccnt_bits);
__IO_REG32_BIT(PSCFLG,              0x4000DD04,__READ_WRITE ,__pscflg_bits);

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
__IO_REG32_BIT(DMACASOFTSREQ,         0x40000024,__READ_WRITE ,__dmacasoftsreq_bits);
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
__IO_REG32_BIT(TB9DMA,              0x400C4930, __READ_WRITE , __tbxdma_bits);

/***************************************************************************
 **
 ** ENC 0
 **
 ***************************************************************************/
__IO_REG32_BIT(ENC0EN,              0x40060000, __READ_WRITE , __encxen_bits);
__IO_REG32_BIT(ENC0CNT,             0x40060004, __READ_WRITE , __encxcnt_bits);
__IO_REG32_BIT(ENC0IE,              0x40060008, __READ_WRITE , __encxie_bits);
__IO_REG32_BIT(ENC0FLG,             0x4006000C, __READ_WRITE , __encxflg_bits);
__IO_REG32_BIT(ENC0ARUN,            0x40060010, __READ_WRITE , __encxarun_bits);
__IO_REG32_BIT(ENC0ACP0,            0x40060014, __READ_WRITE , __encxacp0_bits);
__IO_REG32_BIT(ENC0ACP1,            0x40060018, __READ_WRITE , __encxacp1_bits);
__IO_REG32_BIT(ENC0ADAT,            0x4006001C, __READ_WRITE , __encxadat_bits);
__IO_REG32_BIT(ENC0BRUN,            0x40060020, __READ_WRITE , __encxbrun_bits);
__IO_REG32_BIT(ENC0BDMA,            0x40060024, __READ_WRITE , __encxbdma_bits);
__IO_REG32_BIT(ENC00BUC,            0x40060028, __READ       , __encxbuc_bits);
__IO_REG32_BIT(ENC0BCAP00,          0x40060030, __READ       , __encxbcap00_bits);
__IO_REG32_BIT(ENC0BCAP10,          0x40060034, __READ       , __encxbcap10_bits);
__IO_REG32_BIT(ENC0BCAP20,          0x40060038, __READ       , __encxbcap20_bits);
__IO_REG32_BIT(ENC0BCAP30,          0x4006003C, __READ       , __encxbcap30_bits);
__IO_REG32_BIT(ENC0B0DAT,           0x40060040, __READ       , __encxbodat_bits);
__IO_REG32_BIT(ENC0B1DAT,           0x40060044, __READ       , __encxb1dat_bits);
__IO_REG32_BIT(ENC0B2DAT,           0x40060048, __READ       , __encxb2dat_bits);
__IO_REG32_BIT(ENC0B3DAT,           0x4006004C, __READ       , __encxb3dat_bits);
__IO_REG32_BIT(ENC0BCDAT,           0x40060050, __READ       , __encxbcdat_bits);
__IO_REG32_BIT(ENC0B0PDT,           0x40060060, __READ       , __encxb0pdt_bits);
__IO_REG32_BIT(ENC0B1PDT,           0x40060064, __READ       , __encxb1pdt_bits);
__IO_REG32_BIT(ENC0B2PDT,           0x40060068, __READ       , __encxb2pdt_bits);
__IO_REG32_BIT(ENC0B3PDT,           0x4006006C, __READ       , __encxb3pdt_bits);

/***************************************************************************
 **
 ** ENC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(ENC1EN,              0x40061000, __READ_WRITE , __encxen_bits);
__IO_REG32_BIT(ENC1CNT,             0x40061004, __READ_WRITE , __encxcnt_bits);
__IO_REG32_BIT(ENC1IE,              0x40061008, __READ_WRITE , __encxie_bits);
__IO_REG32_BIT(ENC1FLG,             0x4006100C, __READ_WRITE , __encxflg_bits);
__IO_REG32_BIT(ENC1ARUN,            0x40061010, __READ_WRITE , __encxarun_bits);
__IO_REG32_BIT(ENC1ACP0,            0x40061014, __READ_WRITE , __encxacp0_bits);
__IO_REG32_BIT(ENC1ACP1,            0x40061018, __READ_WRITE , __encxacp1_bits);
__IO_REG32_BIT(ENC1ADAT,            0x4006101C, __READ_WRITE , __encxadat_bits);
__IO_REG32_BIT(ENC1BRUN,            0x40061020, __READ_WRITE , __encxbrun_bits);
__IO_REG32_BIT(ENC1BDMA,            0x40061024, __READ_WRITE , __encxbdma_bits);
__IO_REG32_BIT(ENC10BUC,            0x40061028, __READ       , __encxbuc_bits);
__IO_REG32_BIT(ENC1BCAP00,          0x40061030, __READ       , __encxbcap00_bits);
__IO_REG32_BIT(ENC1BCAP10,          0x40061034, __READ       , __encxbcap10_bits);
__IO_REG32_BIT(ENC1BCAP20,          0x40061038, __READ       , __encxbcap20_bits);
__IO_REG32_BIT(ENC1BCAP30,          0x4006103C, __READ       , __encxbcap30_bits);
__IO_REG32_BIT(ENC1B0DAT,           0x40061040, __READ       , __encxbodat_bits);
__IO_REG32_BIT(ENC1B1DAT,           0x40061044, __READ       , __encxb1dat_bits);
__IO_REG32_BIT(ENC1B2DAT,           0x40061048, __READ       , __encxb2dat_bits);
__IO_REG32_BIT(ENC1B3DAT,           0x4006104C, __READ       , __encxb3dat_bits);
__IO_REG32_BIT(ENC1BCDAT,           0x40061050, __READ       , __encxbcdat_bits);
__IO_REG32_BIT(ENC1B0PDT,           0x40061060, __READ       , __encxb0pdt_bits);
__IO_REG32_BIT(ENC1B1PDT,           0x40061064, __READ       , __encxb1pdt_bits);
__IO_REG32_BIT(ENC1B2PDT,           0x40061068, __READ       , __encxb2pdt_bits);
__IO_REG32_BIT(ENC1B3PDT,           0x4006106C, __READ       , __encxb3pdt_bits);

/***************************************************************************
 **
 ** TMRD_A 0
 **
 ***************************************************************************/
__IO_REG32_BIT(TD0ARUN,             0x40058000, __WRITE      ,__tdxrun_bits);
__IO_REG32_BIT(TD0ACR,              0x40058004, __READ_WRITE ,__tdxcr0_bits);
__IO_REG32_BIT(TD0AMOD,             0x40058008, __READ_WRITE ,__tdxmod_bits);
__IO_REG32_BIT(TD0ADMA,             0x4005800C, __READ_WRITE ,__tdxdma_bits);
__IO_REG32_BIT(TD0ARG0,             0x40058014, __READ_WRITE ,__tdxrg0_bits);
__IO_REG32_BIT(TD0ARG1,             0x40058018, __READ_WRITE ,__tdxrg1_bits);
__IO_REG32_BIT(TD0ARG2,             0x4005801C, __READ_WRITE ,__tdxrg2_bits);
__IO_REG32_BIT(TD0ARG3,             0x40058020, __READ_WRITE ,__tdxrg3_bits);
__IO_REG32_BIT(TD0ARG4,             0x40058024, __READ_WRITE ,__tdxrg4_bits);
__IO_REG32_BIT(TD0ARG5,             0x40058028, __READ_WRITE ,__tdxrg5_bits);
__IO_REG32_BIT(TDABCR,              0x40058040, __READ_WRITE ,__tdxbcr_bits);
__IO_REG32_BIT(TDAEN,               0x40058050, __READ_WRITE ,__tdxen_bits);
__IO_REG32_BIT(TDACONF,             0x40058054, __READ_WRITE ,__tdxconf_bits);
__IO_REG32_BIT(TD0ACP0,             0x40058114, __READ       ,__tdxcp0_bits);
__IO_REG32_BIT(TD0ACP1,             0x40058118, __READ       ,__tdxcp1_bits);
__IO_REG32_BIT(TD0ACP2,             0x4005811C, __READ       ,__tdxcp2_bits);
__IO_REG32_BIT(TD0ACP3,             0x40058120, __READ       ,__tdxcp3_bits);
__IO_REG32_BIT(TD0ACP4,             0x40058124, __READ       ,__tdxcp4_bits);
__IO_REG32_BIT(TD0ACP5,             0x40058128, __READ       ,__tdxcp5_bits);

/***************************************************************************
 **
 ** TMRD_A 1
 **
 ***************************************************************************/
__IO_REG32_BIT(TD1ARUN,             0x40058100, __WRITE      ,__tdxrun_bits);
__IO_REG32_BIT(TD1ACR,              0x40058104, __READ_WRITE ,__tdxcr1_bits);
__IO_REG32_BIT(TD1AMOD,             0x40058108, __READ_WRITE ,__tdxmod_bits);
__IO_REG32_BIT(TD1ADMA,             0x4005810C, __READ_WRITE ,__tdxdma_bits);
__IO_REG32_BIT(TD1ARG0,             0x4005802C, __READ_WRITE ,__tdxrg0_bits);
__IO_REG32_BIT(TD1ARG1,             0x40058030, __READ_WRITE ,__tdxrg1_bits);
__IO_REG32_BIT(TD1ARG2,             0x40058034, __READ_WRITE ,__tdxrg2_bits);
__IO_REG32_BIT(TD1ARG3,             0x40058038, __READ_WRITE ,__tdxrg3_bits);
__IO_REG32_BIT(TD1ARG4,             0x4005803C, __READ_WRITE ,__tdxrg4_bits);
__IO_REG32_BIT(TD1ACP0,             0x4005812C, __READ       ,__tdxcp0_bits);
__IO_REG32_BIT(TD1ACP1,             0x40058130, __READ       ,__tdxcp1_bits);
__IO_REG32_BIT(TD1ACP2,             0x40058134, __READ       ,__tdxcp2_bits);
__IO_REG32_BIT(TD1ACP3,             0x40058138, __READ       ,__tdxcp3_bits);
__IO_REG32_BIT(TD1ACP4,             0x4005813C, __READ       ,__tdxcp4_bits);

/***************************************************************************
 **
 ** TMRD_B 0
 **
 ***************************************************************************/
__IO_REG32_BIT(TD0BRUN,             0x40059000, __WRITE      ,__tdxrun_bits);
__IO_REG32_BIT(TD0BCR,              0x40059004, __READ_WRITE ,__tdxcr0_bits);
__IO_REG32_BIT(TD0BMOD,             0x40059008, __READ_WRITE ,__tdxmod_bits);
__IO_REG32_BIT(TD0BDMA,             0x4005900C, __READ_WRITE ,__tdxdma_bits);
__IO_REG32_BIT(TD0BRG0,             0x40059014, __READ_WRITE ,__tdxrg0_bits);
__IO_REG32_BIT(TD0BRG1,             0x40059018, __READ_WRITE ,__tdxrg1_bits);
__IO_REG32_BIT(TD0BRG2,             0x4005901C, __READ_WRITE ,__tdxrg2_bits);
__IO_REG32_BIT(TD0BRG3,             0x40059020, __READ_WRITE ,__tdxrg3_bits);
__IO_REG32_BIT(TD0BRG4,             0x40059024, __READ_WRITE ,__tdxrg4_bits);
__IO_REG32_BIT(TD0BRG5,             0x40059028, __READ_WRITE ,__tdxrg5_bits);
__IO_REG32_BIT(TDBBCR,              0x40059040, __READ_WRITE ,__tdxbcr_bits);
__IO_REG32_BIT(TDBEN,               0x40059050, __READ_WRITE ,__tdxen_bits);
__IO_REG32_BIT(TDBCONF,             0x40059054, __READ_WRITE ,__tdxconf_bits);
__IO_REG32_BIT(TD0BCP0,             0x40059114, __READ       ,__tdxcp0_bits);
__IO_REG32_BIT(TD0BCP1,             0x40059118, __READ       ,__tdxcp1_bits);
__IO_REG32_BIT(TD0BCP2,             0x4005911C, __READ       ,__tdxcp2_bits);
__IO_REG32_BIT(TD0BCP3,             0x40059120, __READ       ,__tdxcp3_bits);
__IO_REG32_BIT(TD0BCP4,             0x40059124, __READ       ,__tdxcp4_bits);
__IO_REG32_BIT(TD0BCP5,             0x40059128, __READ       ,__tdxcp5_bits);

/***************************************************************************
 **
 ** TMRD_B 1
 **
 ***************************************************************************/
__IO_REG32_BIT(TD1BRUN,             0x40059100, __WRITE      ,__tdxrun_bits);
__IO_REG32_BIT(TD1BCR,              0x40059104, __READ_WRITE ,__tdxcr1_bits);
__IO_REG32_BIT(TD1BMOD,             0x40059108, __READ_WRITE ,__tdxmod_bits);
__IO_REG32_BIT(TD1BDMA,             0x4005910C, __READ_WRITE ,__tdxdma_bits);
__IO_REG32_BIT(TD1BRG0,             0x4005902C, __READ_WRITE ,__tdxrg0_bits);
__IO_REG32_BIT(TD1BRG1,             0x40059030, __READ_WRITE ,__tdxrg1_bits);
__IO_REG32_BIT(TD1BRG2,             0x40059034, __READ_WRITE ,__tdxrg2_bits);
__IO_REG32_BIT(TD1BRG3,             0x40059038, __READ_WRITE ,__tdxrg3_bits);
__IO_REG32_BIT(TD1BRG4,             0x4005903C, __READ_WRITE ,__tdxrg4_bits);
__IO_REG32_BIT(TD1BCP0,             0x4005912C, __READ       ,__tdxcp0_bits);
__IO_REG32_BIT(TD1BCP1,             0x40059130, __READ       ,__tdxcp1_bits);
__IO_REG32_BIT(TD1BCP2,             0x40059134, __READ       ,__tdxcp2_bits);
__IO_REG32_BIT(TD1BCP3,             0x40059138, __READ       ,__tdxcp3_bits);
__IO_REG32_BIT(TD1BCP4,             0x4005913C, __READ       ,__tdxcp4_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG8_BIT(WDMOD,                0x400F2000,__READ_WRITE ,__wdmod_bits);
__IO_REG8(    WDCR,                 0x400F2004,__WRITE);

/***************************************************************************
 **
 ** SIO 0
 **
 ***************************************************************************/
__IO_REG8_BIT( SC0EN,               0x400E1000, __READ_WRITE , __scxen_bits);
__IO_REG8(     SC0BUF,              0x400E1004, __READ_WRITE );
__IO_REG8_BIT( SC0CR,               0x400E1008, __READ_WRITE , __scxcr_bits);
__IO_REG8_BIT( SC0MOD0,             0x400E100C, __READ_WRITE , __scxmod0_bits);
__IO_REG8_BIT( SC0BRCR,             0x400E1010, __READ_WRITE , __scxbrcr_bits);
__IO_REG8_BIT( SC0BRADD,            0x400E1014, __READ_WRITE , __scxbradd_bits);
__IO_REG8_BIT( SC0MOD1,             0x400E1018, __READ_WRITE , __scxmod1_bits);
__IO_REG8_BIT( SC0MOD2,             0x400E101C, __READ_WRITE , __scxmod2_bits);
__IO_REG8_BIT( SC0RFC,              0x400E1020, __READ_WRITE , __scxrfc_bits);
__IO_REG8_BIT( SC0TFC,              0x400E1024, __READ_WRITE , __scxtfc_bits);
__IO_REG8_BIT( SC0RST,              0x400E1028, __READ       , __scxrst_bits);
__IO_REG8_BIT( SC0TST,              0x400E102C, __READ       , __scxtst_bits);
__IO_REG8_BIT( SC0FCNF,             0x400E1030, __READ_WRITE , __scxfcnf_bits);
__IO_REG8_BIT( SC0DMA,              0x400E1034, __READ_WRITE , __scxdma_bits);

/***************************************************************************
 **
 ** SIO 1
 **
 ***************************************************************************/
__IO_REG8_BIT( SC1EN,               0x400E1100, __READ_WRITE , __scxen_bits);
__IO_REG8(     SC1BUF,              0x400E1104, __READ_WRITE );
__IO_REG8_BIT( SC1CR,               0x400E1108, __READ_WRITE , __scxcr_bits);
__IO_REG8_BIT( SC1MOD0,             0x400E110C, __READ_WRITE , __scxmod0_bits);
__IO_REG8_BIT( SC1BRCR,             0x400E1110, __READ_WRITE , __scxbrcr_bits);
__IO_REG8_BIT( SC1BRADD,            0x400E1114, __READ_WRITE , __scxbradd_bits);
__IO_REG8_BIT( SC1MOD1,             0x400E1118, __READ_WRITE , __scxmod1_bits);
__IO_REG8_BIT( SC1MOD2,             0x400E111C, __READ_WRITE , __scxmod2_bits);
__IO_REG8_BIT( SC1RFC,              0x400E1120, __READ_WRITE , __scxrfc_bits);
__IO_REG8_BIT( SC1TFC,              0x400E1124, __READ_WRITE , __scxtfc_bits);
__IO_REG8_BIT( SC1RST,              0x400E1128, __READ       , __scxrst_bits);
__IO_REG8_BIT( SC1TST,              0x400E112C, __READ       , __scxtst_bits);
__IO_REG8_BIT( SC1FCNF,             0x400E1130, __READ_WRITE , __scxfcnf_bits);

/***************************************************************************
 **
 ** SIO 2
 **
 ***************************************************************************/
__IO_REG8_BIT( SC2EN,               0x400E1200, __READ_WRITE , __scxen_bits);
__IO_REG8(     SC2BUF,              0x400E1204, __READ_WRITE );
__IO_REG8_BIT( SC2CR,               0x400E1208, __READ_WRITE , __scxcr_bits);
__IO_REG8_BIT( SC2MOD0,             0x400E120C, __READ_WRITE , __scxmod0_bits);
__IO_REG8_BIT( SC2BRCR,             0x400E1210, __READ_WRITE , __scxbrcr_bits);
__IO_REG8_BIT( SC2BRADD,            0x400E1214, __READ_WRITE , __scxbradd_bits);
__IO_REG8_BIT( SC2MOD1,             0x400E1218, __READ_WRITE , __scxmod1_bits);
__IO_REG8_BIT( SC2MOD2,             0x400E121C, __READ_WRITE , __scxmod2_bits);
__IO_REG8_BIT( SC2RFC,              0x400E1220, __READ_WRITE , __scxrfc_bits);
__IO_REG8_BIT( SC2TFC,              0x400E1224, __READ_WRITE , __scxtfc_bits);
__IO_REG8_BIT( SC2RST,              0x400E1228, __READ       , __scxrst_bits);
__IO_REG8_BIT( SC2TST,              0x400E122C, __READ       , __scxtst_bits);
__IO_REG8_BIT( SC2FCNF,             0x400E1230, __READ_WRITE , __scxfcnf_bits);
__IO_REG8_BIT( SC2DMA,              0x400E1234, __READ_WRITE , __scxdma_bits);

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG32_BIT(UARTDR,               0x40048000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UARTSR,               0x40048004,__READ_WRITE ,__uartsr_bits);
#define UARTECR     UARTSR
__IO_REG32_BIT(UARTFR,               0x40048018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UARTIBRD,             0x40048024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UARTFBRD,             0x40048028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UARTLCR_H,            0x4004802C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG32_BIT(UARTCR,               0x40048030,__READ_WRITE ,__uartcr_bits);
__IO_REG32_BIT(UARTIFLS,             0x40048034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UARTIMSC,             0x40048038,__READ_WRITE ,__uartimsc_bits);
__IO_REG32_BIT(UARTRIS,              0x4004803C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UARTMIS,              0x40048040,__READ       ,__uartmis_bits);
__IO_REG32_BIT(UARTICR,              0x40048044,__WRITE      ,__uarticr_bits);
__IO_REG32_BIT(UARTDMACR,            0x40048048,__READ_WRITE ,__uartdmacr_bits);

/***************************************************************************
 **
 ** SBI
 **
 ***************************************************************************/
__IO_REG32_BIT(SBICR0,              0x400E0000, __READ_WRITE , __sbicr0_bits);
__IO_REG32_BIT(SBICR1,              0x400E0004, __READ_WRITE , __sbicr1_bits);
__IO_REG32_BIT(SBIDBR,              0x400E0008, __READ_WRITE , __sbidbr_bits);
__IO_REG32_BIT(SBII2CAR,            0x400E000C, __READ_WRITE , __sbii2car_bits);
__IO_REG32_BIT(SBICR2,              0x400E0010, __READ_WRITE , __sbicr2_bits);
#define SBISR       SBICR2
#define SBISR_bit   SBICR2_bit.__sbisr
__IO_REG32_BIT(SBIBR0,              0x400E0014, __READ_WRITE , __sbibr0_bits);

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
 ** VSIO
 **
 ***************************************************************************/
__IO_REG32_BIT(VSIOEN,              0x40062000, __READ_WRITE , __vsioen_bits);
__IO_REG32(    VSIOBUF,             0x40062004, __READ_WRITE );
__IO_REG32_BIT(VSIOCR0,             0x40062008, __READ_WRITE , __vsiocr0_bits);
__IO_REG32_BIT(VSIOCR1,             0x4006200C, __READ_WRITE , __vsiocr1_bits);
__IO_REG32_BIT(VSIOCR2,             0x40062010, __READ_WRITE , __vsiocr2_bits);
__IO_REG32_BIT(VSIOCR3,             0x40062014, __READ_WRITE , __vsiocr3_bits);
__IO_REG32_BIT(VSIOBRCR,            0x40062018, __READ_WRITE , __vsiobrcr_bits);
__IO_REG32_BIT(VSIORFC,             0x4006201C, __READ_WRITE , __vsiorfc_bits);
__IO_REG32_BIT(VSIOTFC,             0x40062020, __READ_WRITE , __vsiotfc_bits);
__IO_REG32_BIT(VSIORST,             0x40062024, __READ       , __vsiorst_bits);
__IO_REG32_BIT(VSIOTST,             0x40062028, __READ       , __vsiotst_bits);

/***************************************************************************
 **
 ** MCDSIO
 **
 ***************************************************************************/
__IO_REG32_BIT( SC3EN,              0x400E1300, __READ_WRITE , __sc3en_bits);
__IO_REG32_BIT( SC3BUF,             0x400E1304, __WRITE      , __sc3buf_bits);
__IO_REG32_BIT( SC3MOD0,            0x400E130C, __READ_WRITE , __sc3mod0_bits);
__IO_REG32_BIT( SC3BRCR,            0x400E1310, __READ_WRITE , __sc3brcr_bits);
__IO_REG32_BIT( SC3MOD1,            0x400E1318, __READ_WRITE , __sc3mod1_bits);
__IO_REG32_BIT( SC3MOD2,            0x400E131C, __READ_WRITE , __sc3mod2_bits);

/***************************************************************************
 **
 ** ADC A
 **
 ***************************************************************************/
__IO_REG8_BIT( ADACLK,               0x40050000,__READ_WRITE ,__adclk_bits);
__IO_REG8_BIT( ADAMOD0,              0x40050004,__WRITE      ,__admod0_bits);
__IO_REG8_BIT( ADAMOD1,              0x40050008,__READ_WRITE ,__admod1_bits);
__IO_REG8_BIT( ADAMOD2,              0x4005000C,__READ_WRITE ,__admod2_bits);
__IO_REG8_BIT( ADAMOD3,              0x40050010,__READ_WRITE ,__admod3_bits);
__IO_REG8_BIT( ADAMOD4,              0x40050014,__READ_WRITE ,__admod4_bits);
__IO_REG8_BIT( ADAMOD5,              0x40050018,__READ       ,__admod5_bits);
__IO_REG8_BIT( ADAMOD6,              0x4005001C,__WRITE      ,__admod6_bits);
__IO_REG8_BIT( ADAMOD7,              0x40050020,__READ_WRITE ,__admod7_bits);
__IO_REG32_BIT(ADACMPCR0,            0x40050024,__READ_WRITE ,__adcmpcr0_bits);
__IO_REG32_BIT(ADACMPCR1,            0x40050028,__READ_WRITE ,__adcmpcr1_bits);
__IO_REG32_BIT(ADACMP0,              0x4005002C,__READ_WRITE ,__adcmp0_bits);
__IO_REG32_BIT(ADACMP1,              0x40050030,__READ_WRITE ,__adcmp1_bits);
__IO_REG32_BIT(ADAREG0,              0x40050034,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG1,              0x40050038,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG2,              0x4005003C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG3,              0x40050040,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG4,              0x40050044,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG5,              0x40050048,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG6,              0x4005004C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG7,              0x40050050,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREGSP,             0x40050074,__READ       ,__adregsp_bits);

/***************************************************************************
 **
 ** ADC B
 **
 ***************************************************************************/
__IO_REG8_BIT( ADBCLK,               0x40051000,__READ_WRITE ,__adclk_bits);
__IO_REG8_BIT( ADBMOD0,              0x40051004,__WRITE      ,__admod0_bits);
__IO_REG8_BIT( ADBMOD1,              0x40051008,__READ_WRITE ,__admod1_bits);
__IO_REG8_BIT( ADBMOD2,              0x4005100C,__READ_WRITE ,__admod2_bits);
__IO_REG8_BIT( ADBMOD3,              0x40051010,__READ_WRITE ,__admod3_bits);
__IO_REG8_BIT( ADBMOD4,              0x40051014,__READ_WRITE ,__admod4_bits);
__IO_REG8_BIT( ADBMOD5,              0x40051018,__READ       ,__admod5_bits);
__IO_REG8_BIT( ADBMOD6,              0x4005101C,__WRITE      ,__admod6_bits);
__IO_REG8_BIT( ADBMOD7,              0x40051020,__READ_WRITE ,__admod7_bits);
__IO_REG32_BIT(ADBCMPCR0,            0x40051024,__READ_WRITE ,__adcmpcr0_bits);
__IO_REG32_BIT(ADBCMPCR1,            0x40051028,__READ_WRITE ,__adcmpcr1_bits);
__IO_REG32_BIT(ADBCMP0,              0x4005102C,__READ_WRITE ,__adcmp0_bits);
__IO_REG32_BIT(ADBCMP1,              0x40051030,__READ_WRITE ,__adcmp1_bits);
__IO_REG32_BIT(ADBREG0,              0x40051034,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG1,              0x40051038,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG2,              0x4005103C,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG3,              0x40051040,__READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREGSP,             0x40051074,__READ       ,__adregsp_bits);

/***************************************************************************
 **
 ** DS ADC 16Bit
 **
 ***************************************************************************/
__IO_REG8_BIT( DSCLK,                0x40067000,__READ_WRITE ,__dsclk_bits);
__IO_REG8_BIT( DSMOD0,               0x40067004,__WRITE      ,__admod0_bits);
__IO_REG8_BIT( DSMOD1,               0x40067008,__READ_WRITE ,__dsmod1_bits);
__IO_REG8_BIT( DSMOD2,               0x4006700C,__READ_WRITE ,__dsmod2_bits);
__IO_REG8_BIT( DSMOD3,               0x40067010,__READ_WRITE ,__admod3_bits);
__IO_REG8_BIT( DSMOD4,               0x40067014,__READ_WRITE ,__dsmod4_bits);
__IO_REG8_BIT( DSMOD5,               0x40067018,__READ       ,__admod5_bits);
__IO_REG8_BIT( DSMOD6,               0x4006701C,__WRITE      ,__admod6_bits);
__IO_REG8_BIT( DSMOD7,               0x40067020,__READ_WRITE ,__admod7_bits);
__IO_REG16_BIT(DSMOD8,               0x40067024,__READ_WRITE ,__dsmod8_bits);
__IO_REG32_BIT(DSREG0,               0x40067100,__READ       ,__dsregx_bits);
__IO_REG32_BIT(DSREG1,               0x40067104,__READ       ,__dsregx_bits);
__IO_REG32_BIT(DSREG2,               0x40067108,__READ       ,__dsregx_bits);
__IO_REG32_BIT(DSREG3,               0x4006710C,__READ       ,__dsregx_bits);
__IO_REG32_BIT(DSREG4,               0x40067110,__READ       ,__dsregx_bits);
__IO_REG32_BIT(DSREG5,               0x40067114,__READ       ,__dsregx_bits);
__IO_REG32_BIT(DSREGSP,              0x40067200,__READ       ,__dsregsp_bits);

/***************************************************************************
 **
 ** DAC A
 **
 ***************************************************************************/
__IO_REG32_BIT(DAACTL,              0x40054000, __READ_WRITE ,__daxctl_bits);
__IO_REG32_BIT(DAAREG,              0x40054004, __READ_WRITE ,__daxreg_bits);

/***************************************************************************
 **
 ** DAC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(DABCTL,              0x40055000, __READ_WRITE ,__daxctl_bits);
__IO_REG32_BIT(DABREG,              0x40055004, __READ_WRITE ,__daxreg_bits);

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
 **  TMPM342FYXBG Interrupt Lines
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

#define INT_0                ( 0 + EII)   /* Interrupt pin 0              */
#define INT_1                ( 1 + EII)   /* Interrupt pin 1              */
#define INT_2                ( 2 + EII)   /* Interrupt pin 2              */
#define INT_3                ( 3 + EII)   /* Interrupt pin 3              */
#define INT_4                ( 4 + EII)   /* Interrupt pin 4              */
#define INT_5                ( 5 + EII)   /* Interrupt pin 5              */
#define INT_6                ( 6 + EII)   /* Interrupt pin 6              */
#define INT_7                ( 7 + EII)   /* Interrupt pin 7              */
#define INT_PSCSTOP          ( 8 + EII)   /* PSC completion interrupt     */
#define INT_PSCBRK           ( 9 + EII)   /* PSC break interrupt          */
#define INT_PSCSTEP          (10 + EII)   /* PSC step interrupt           */
#define INT_PSCII            (11 + EII)   /* PSC wrong instruction interrupt  */
#define INT_PSCIA            (12 + EII)   /* PSC wrong address interrupt      */
#define INT_TB0              (13 + EII)   /* TB0 compare match interrupt      */
#define INT_TB1              (14 + EII)   /* TB1 compare match interrupt      */
#define INT_TB2              (15 + EII)   /* TB2 compare match interrupt      */
#define INT_TX0              (16 + EII)   /* SIO/UART 0 transmit interrupt    */
#define INT_RX0              (17 + EII)   /* SIO/UART 0 receive interrupt     */
#define INT_TX1              (18 + EII)   /* SIO/UART 1 transmit interrupt    */
#define INT_RX1              (19 + EII)   /* SIO/UART 1 receive interrupt     */
#define INT_TX2              (20 + EII)   /* SIO/UART 2 transmit interrupt    */
#define INT_RX2              (21 + EII)   /* SIO/UART 2 receive interrupt     */
#define INT_TX3              (22 + EII)   /* SIO/UART 3 transmit interrupt    */
#define INT_S                (23 + EII)   /* UART transmit/receive interrupt  */
#define INT_ADAHP            (24 + EII)   /* Highest priority UnitA conversion complete interrupt   */
#define INT_ADA              (25 + EII)   /* Normal UnitA conversion completion interrupt           */
#define INT_ADBHP            (26 + EII)   /* Highest priority UnitB conversion completion interrupt */
#define INT_ADB              (27 + EII)   /* Normal UnitB conversion completion interrupt           */
#define INT_DSADHP           (28 + EII)   /* Highest priority ADC conversion completion interrupt   */
#define INT_DSAD             (29 + EII)   /* Normal ADC conversion completion interrupt             */
#define INT_I2C              (30 + EII)   /* I2C interrupt                    */
#define INT_TB3              (31 + EII)   /* TB3 compare match interrupt      */
#define INT_TB4              (32 + EII)   /* TB4 compare match interrupt      */
#define INT_TB5              (33 + EII)   /* TB5 compare match interrupt      */
#define INT_TB6              (34 + EII)   /* TB6 compare match interrupt      */
#define INT_TB7              (35 + EII)   /* TB7 compare match interrupt      */
#define INT_TB8              (36 + EII)   /* TB8 compare match interrupt      */
#define INT_TB9              (37 + EII)   /* TB9 compare match interrupt      */
#define INT_TDA0CMP0         (38 + EII)   /* TMRDA00 compare match interrupt  */
#define INT_TDA0CMP1         (39 + EII)   /* TMRDA01 compare match interrupt  */
#define INT_TDA0CMP2         (40 + EII)   /* TMRDA02 compare match interrupt  */
#define INT_TDA0CMP3         (41 + EII)   /* TMRDA03 compare match interrupt  */
#define INT_TDA0CMP4         (42 + EII)   /* TMRDA04 compare match interrupt  */
#define INT_TDA1CMP0         (43 + EII)   /* TMRDA10 compare match interrupt  */
#define INT_TDA1CMP1         (44 + EII)   /* TMRDA11 compare match interrupt  */
#define INT_TDA1CMP2         (45 + EII)   /* TMRDA12 compare match interrupt  */
#define INT_TDA1CMP3         (46 + EII)   /* TMRDA13 compare match interrupt  */
#define INT_TDA1CMP4         (47 + EII)   /* TMRDA14 compare match interrupt  */
#define INT_TDB0CMP0         (48 + EII)   /* TMRDB0 compare match interrupt   */
#define INT_TDB0CMP1         (49 + EII)   /* TMRDB0 compare match interrupt   */
#define INT_TDB0CMP2         (50 + EII)   /* TMRDB0 compare match interrupt   */
#define INT_TDB0CMP3         (51 + EII)   /* TMRDB0 compare match interrupt   */
#define INT_TDB0CMP4         (52 + EII)   /* TMRDB0 compare match interrupt   */
#define INT_TDB1CMP0         (53 + EII)   /* TMRDB1 compare match interrupt   */
#define INT_TDB1CMP1         (54 + EII)   /* TMRDB1 compare match interrupt   */
#define INT_TDB1CMP2         (55 + EII)   /* TMRDB1 compare match interrupt   */
#define INT_TDB1CMP3         (56 + EII)   /* TMRDB1 compare match interrupt   */
#define INT_TDB1CMP4         (57 + EII)   /* TMRDB1 compare match interrupt   */
#define INT_EC0              (58 + EII)   /* PHC0 compare interrupt           */
#define INT_EC0OVF           (59 + EII)   /* PHC0 overflow interrupt          */
#define INT_EC0DIR           (60 + EII)   /* PHC0 phase error interrupt       */
#define INT_EC0DT0           (61 + EII)   /* PHC0 cycle 0 interrupt           */
#define INT_EC0DT1           (62 + EII)   /* PHC0 cycle 1 interrupt           */
#define INT_EC0DT2           (63 + EII)   /* PHC0 cycle 2 interrupt           */
#define INT_EC0DT3           (64 + EII)   /* PHC0 cycle 3 interrupt           */
#define INT_EC1              (65 + EII)   /* PHC1 compare interrupt           */
#define INT_EC1OVF           (66 + EII)   /* PHC1 overflow interrupt          */
#define INT_EC1DIR           (67 + EII)   /* PHC1 phase error interrupt       */
#define INT_EC1DT0           (68 + EII)   /* PHC1 cycle 0 interrupt           */
#define INT_EC1DT1           (69 + EII)   /* PHC1 cycle 1 interrupt           */
#define INT_EC1DT2           (70 + EII)   /* PHC1 cycle 2 interrupt           */
#define INT_EC1DT3           (71 + EII)   /* PHC1 cycle 3 interrupt           */
#define INT_AD0M0            (72 + EII)   /* ADC0 monitoring interrupt 0      */
#define INT_AD0M1            (73 + EII)   /* ADC0 monitoring interrupt 1      */
#define INT_AD1M0            (74 + EII)   /* ADC1 monitoring interrupt 0      */
#define INT_AD1M1            (75 + EII)   /* ADC1 monitoring interrupt 1      */
#define INT_DMACATC          (76 + EII)   /* DMACA transfer end interrupt     */
#define INT_DMACAERR         (77 + EII)   /* DMACA error interrupt            */
#define INT_DMACBTC          (78 + EII)   /* DMACB transfer end interrupt     */
#define INT_DMACBERR         (79 + EII)   /* DMACB error interrupt            */
#define INT_SSP              (80 + EII)   /* SSP interrupt                    */
#define INT_VTX              (81 + EII)   /* VSIO transmit interrupt          */
#define INT_VRX              (82 + EII)   /* VSIO receive interrupt           */
#define INT_8                (84 + EII)   /* Interrupt pin 8(for MO1 function)    */
#define INT_9                (85 + EII)   /* Interrupt pin 9(for MO2 function)    */
#define INT_A                (86 + EII)   /* Interrupt pin A                      */


#endif    /* __IOTMPM342FYXBG_H */

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
Interrupt17  = INTPSCSTOP     0x60
Interrupt18  = INTPSCBRK      0x64
Interrupt19  = INTPSCSTEP     0x68
Interrupt20  = INTPSCII       0x6C
Interrupt21  = INTPSCIA       0x70
Interrupt22  = INTTB0         0x74
Interrupt23  = INTTB1         0x78
Interrupt24  = INTTB2         0x7C
Interrupt25  = INTTX0         0x80
Interrupt26  = INTRX0         0x84
Interrupt27  = INTTX1         0x88
Interrupt28  = INTRX1         0x8C
Interrupt29  = INTTX2         0x90
Interrupt30  = INTRX2         0x94
Interrupt31  = INTTX3         0x98
Interrupt32  = INTS           0x9C
Interrupt33  = INTADAHP       0xA0
Interrupt34  = INTADA         0xA4
Interrupt35  = INTADBHP       0xA8
Interrupt36  = INTADB         0xAC
Interrupt37  = INTDSADHP      0xB0
Interrupt38  = INTDSAD        0xB4
Interrupt39  = INTI2C         0xB8
Interrupt40  = INTTB3         0xBC
Interrupt41  = INTTB4         0xC0
Interrupt42  = INTTB5         0xC4
Interrupt43  = INTTB6         0xC8
Interrupt44  = INTTB7         0xCC
Interrupt45  = INTTB8         0xD0
Interrupt46  = INTTB9         0xD4
Interrupt47  = INTTDA0CMP0    0xD8
Interrupt48  = INTTDA0CMP1    0xDC
Interrupt49  = INTTDA0CMP2    0xE0
Interrupt50  = INTTDA0CMP3    0xE4
Interrupt51  = INTTDA0CMP4    0xE8
Interrupt52  = INTTDA1CMP0    0xEC
Interrupt53  = INTTDA1CMP1    0xF0
Interrupt54  = INTTDA1CMP2    0xF4
Interrupt55  = INTTDA1CMP3    0xF8
Interrupt56  = INTTDA1CMP4    0xFC
Interrupt57  = INTTDB0CMP0    0x100
Interrupt58  = INTTDB0CMP1    0x104
Interrupt59  = INTTDB0CMP2    0x108
Interrupt60  = INTTDB0CMP3    0x10C
Interrupt61  = INTTDB0CMP4    0x110
Interrupt62  = INTTDB1CMP0    0x114
Interrupt63  = INTTDB1CMP1    0x118
Interrupt64  = INTTDB1CMP2    0x11C
Interrupt65  = INTTDB1CMP3    0x120
Interrupt66  = INTTDB1CMP4    0x124
Interrupt67  = INTEC0         0x128
Interrupt68  = INTEC0OVF      0x12C
Interrupt69  = INTEC0DIR      0x130
Interrupt70  = INTEC0DT0      0x134
Interrupt71  = INTEC0DT1      0x138
Interrupt72  = INTEC0DT2      0x13C
Interrupt73  = INTEC0DT3      0x140
Interrupt74  = INTEC1         0x144
Interrupt75  = INTEC1OVF      0x148
Interrupt76  = INTEC1DIR      0x14C
Interrupt77  = INTEC1DT0      0x150
Interrupt78  = INTEC1DT1      0x154
Interrupt79  = INTEC1DT2      0x158
Interrupt80  = INTEC1DT3      0x15C
Interrupt81  = INTAD0M0       0x160
Interrupt82  = INTAD0M1       0x164
Interrupt83  = INTAD1M0       0x168
Interrupt84  = INTAD1M1       0x16C
Interrupt85  = INTDMACATC     0x170
Interrupt86  = INTDMACAERR    0x174
Interrupt87  = INTDMACBTC     0x178
Interrupt88  = INTDMACBERR    0x17C
Interrupt89  = INTSSP         0x180
Interrupt90  = INTVTX         0x184
Interrupt91  = INTVRX         0x188
Interrupt93  = INT8           0x190
Interrupt94  = INT9           0x194
Interrupt95  = INTA           0x198

###DDF-INTERRUPT-END###*/
