/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPM369FxFG
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50001 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPM369FxFG_H
#define __IOTMPM369FxFG_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPM369FxFG SPECIAL FUNCTION REGISTERS
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

/* System Control Register */
typedef struct {
  __REG32 GEAR    : 3;
  __REG32         : 5;
  __REG32 PRCK    : 3;
  __REG32         : 1;
  __REG32 FPSEL   : 1;
  __REG32         : 3;
  __REG32 SCOSEL  : 2;
  __REG32         : 2;
  __REG32 FCSTOP  : 1;
  __REG32         :11;
} __cgsyscr_bits;

/* Oscillation Control Register */
typedef struct {
  __REG32 WUEON     : 1;
  __REG32 WUEF      : 1;
  __REG32 PLLON     : 1;
  __REG32 WUPSEL1   : 1;
  __REG32           : 4;
  __REG32 XEN1      : 1;
  __REG32 XTEN      : 1;
  __REG32 XEN3      : 1;
  __REG32           : 3;
  __REG32 WUPTL     : 2;
  __REG32 XEN2      : 1;
  __REG32 OSCSEL    : 1;
  __REG32 EHOSCSEL  : 1;
  __REG32 WUPSEL2   : 1;
  __REG32 WUPT      :12; 
} __cgosccr_bits;

/* Standby Control Register */
typedef struct {
  __REG32 STBY    : 3;
  __REG32         :13;
  __REG32 DRVE    : 1;
  __REG32 PTKEEP  : 1;
  __REG32         :14;
} __cgstbycr_bits;

/* PLL Selection Register */
typedef struct {
  __REG32 PLLSEL  : 1;
  __REG32 PLLSET  :15;
  __REG32         :16;
} __cgpllsel_bits;

/* System Clock Selection Register */
typedef struct {
  __REG32 SYSCKFLG  : 1;
  __REG32 SYSCK     : 1;
  __REG32           :30;
} __cgcksel_bits;

/* CGCKSTP */
typedef struct {
  __REG32 ETH       : 1;
  __REG32 CAN       : 1;
  __REG32 USBH      : 1;
  __REG32 USBD      : 1;
  __REG32           :28;
} __cgckstp_bits;

/* CGPROTECT (Protect register) */
typedef struct {
  __REG32 CGPROTECT : 8;
  __REG32           :24;
} __cgprotect_bits;

/* NMI Flag Register */
typedef struct {
  __REG32 NMIFLG0   : 1;
  __REG32 NMIFLG1   : 1;
  __REG32 NMIFLG2   : 1;
  __REG32 NMIFLG3   : 1;
  __REG32           :28;
} __cgnmiflg_bits;

/* Reset Flag Register */
typedef struct {
  __REG32 PONRSTF   : 1;
  __REG32 PINRSTF   : 1;
  __REG32 WDTRSTF   : 1;
  __REG32 STOP2RSTF : 1;
  __REG32 DBGRSTF   : 1;
  __REG32 OFDRSTF   : 1;
  __REG32 LVDRSTF   : 1;
  __REG32           :25;
} __cgrstflg_bits;

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
  __REG32 INTUSBWKUPEN  : 1;
  __REG32               : 1;
  __REG32 EMSTUSBWKUP   : 2;
  __REG32 EMCGUSBWKUP   : 3;
  __REG32               : 1;
  __REG32 INTDEN        : 1;
  __REG32               : 1;
  __REG32 EMSTD         : 2;
  __REG32 EMCGD         : 3;
  __REG32               : 1;
  __REG32 INTRTCEN      : 1;
  __REG32               : 1;
  __REG32 EMSTRTC       : 2;
  __REG32 EMCGRTC       : 3;
  __REG32               : 1;
  __REG32 INTRMCRXEN    : 1;
  __REG32               : 1;
  __REG32 EMSTRMCRX     : 2;
  __REG32 EMCGRMCRX2    : 3;
  __REG32               : 1;
} __cgimcgd_bits;

/* CG Interrupt Request Clear Register */
typedef struct {
  __REG32 ICRCG     : 5;
  __REG32           :27;
} __cgicrcg_bits;

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

/* Interrupt Set-Enable Registers 96-127 */
typedef struct {
  __REG32  SETENA96       : 1;
  __REG32  SETENA97       : 1;
  __REG32  SETENA98       : 1;
  __REG32  SETENA99       : 1;
  __REG32  SETENA100      : 1;
  __REG32  SETENA101      : 1;
  __REG32  SETENA102      : 1;
  __REG32  SETENA103      : 1;
  __REG32  SETENA104      : 1;
  __REG32  SETENA105      : 1;
  __REG32  SETENA106      : 1;
  __REG32  SETENA107      : 1;
  __REG32  SETENA108      : 1;
  __REG32  SETENA109      : 1;
  __REG32  SETENA110      : 1;
  __REG32  SETENA111      : 1;
  __REG32  SETENA112      : 1;
  __REG32  SETENA113      : 1;
  __REG32  SETENA114      : 1;
  __REG32  SETENA115      : 1;
  __REG32  SETENA116      : 1;
  __REG32  SETENA117      : 1;
  __REG32  SETENA118      : 1;
  __REG32  SETENA119      : 1;
  __REG32  SETENA120      : 1;
  __REG32  SETENA121      : 1;
  __REG32  SETENA122      : 1;
  __REG32  SETENA123      : 1;
  __REG32  SETENA124      : 1;
  __REG32  SETENA125      : 1;
  __REG32  SETENA126      : 1;
  __REG32  SETENA127      : 1;
} __setena3_bits;

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

/* Interrupt Clear-Enable Registers 96-127 */
typedef struct {
  __REG32  CLRENA96       : 1;
  __REG32  CLRENA97       : 1;
  __REG32  CLRENA98       : 1;
  __REG32  CLRENA99       : 1;
  __REG32  CLRENA100      : 1;
  __REG32  CLRENA101      : 1;
  __REG32  CLRENA102      : 1;
  __REG32  CLRENA103      : 1;
  __REG32  CLRENA104      : 1;
  __REG32  CLRENA105      : 1;
  __REG32  CLRENA106      : 1;
  __REG32  CLRENA107      : 1;
  __REG32  CLRENA108      : 1;
  __REG32  CLRENA109      : 1;
  __REG32  CLRENA110      : 1;
  __REG32  CLRENA111      : 1;
  __REG32  CLRENA112      : 1;
  __REG32  CLRENA113      : 1;
  __REG32  CLRENA114      : 1;
  __REG32  CLRENA115      : 1;
  __REG32  CLRENA116      : 1;
  __REG32  CLRENA117      : 1;
  __REG32  CLRENA118      : 1;
  __REG32  CLRENA119      : 1;
  __REG32  CLRENA120      : 1;
  __REG32  CLRENA121      : 1;
  __REG32  CLRENA122      : 1;
  __REG32  CLRENA123      : 1;
  __REG32  CLRENA124      : 1;
  __REG32  CLRENA125      : 1;
  __REG32  CLRENA126      : 1;
  __REG32  CLRENA127      : 1;
} __clrena3_bits;

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

/* Interrupt Set-Pending Register 96-127 */
typedef struct {
  __REG32  SETPEND96      : 1;
  __REG32  SETPEND97      : 1;
  __REG32  SETPEND98      : 1;
  __REG32  SETPEND99      : 1;
  __REG32  SETPEND100     : 1;
  __REG32  SETPEND101     : 1;
  __REG32  SETPEND102     : 1;
  __REG32  SETPEND103     : 1;
  __REG32  SETPEND104     : 1;
  __REG32  SETPEND105     : 1;
  __REG32  SETPEND106     : 1;
  __REG32  SETPEND107     : 1;
  __REG32  SETPEND108     : 1;
  __REG32  SETPEND109     : 1;
  __REG32  SETPEND110     : 1;
  __REG32  SETPEND111     : 1;
  __REG32  SETPEND112     : 1;
  __REG32  SETPEND113     : 1;
  __REG32  SETPEND114     : 1;
  __REG32  SETPEND115     : 1;
  __REG32  SETPEND116     : 1;
  __REG32  SETPEND117     : 1;
  __REG32  SETPEND118     : 1;
  __REG32  SETPEND119     : 1;
  __REG32  SETPEND120     : 1;
  __REG32  SETPEND121     : 1;
  __REG32  SETPEND122     : 1;
  __REG32  SETPEND123     : 1;
  __REG32  SETPEND124     : 1;
  __REG32  SETPEND125     : 1;
  __REG32  SETPEND126     : 1;
  __REG32  SETPEND127     : 1;
} __setpend3_bits;

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
  __REG32  CLRPEN881      : 1;
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

/* Interrupt Clear-Pending Register 96-127 */
typedef struct {
  __REG32  CLRPEND96      : 1;
  __REG32  CLRPEND97      : 1;
  __REG32  CLRPEND98      : 1;
  __REG32  CLRPEND99      : 1;
  __REG32  CLRPEND100     : 1;
  __REG32  CLRPEND101     : 1;
  __REG32  CLRPEND102     : 1;
  __REG32  CLRPEND103     : 1;
  __REG32  CLRPEND104     : 1;
  __REG32  CLRPEND105     : 1;
  __REG32  CLRPEND106     : 1;
  __REG32  CLRPEND107     : 1;
  __REG32  CLRPEND108     : 1;
  __REG32  CLRPEND109     : 1;
  __REG32  CLRPEND110     : 1;
  __REG32  CLRPEND111     : 1;
  __REG32  CLRPEND112     : 1;
  __REG32  CLRPEND113     : 1;
  __REG32  CLRPEND114     : 1;
  __REG32  CLRPEND115     : 1;
  __REG32  CLRPEND116     : 1;
  __REG32  CLRPEND117     : 1;
  __REG32  CLRPEND118     : 1;
  __REG32  CLRPEND119     : 1;
  __REG32  CLRPEND120     : 1;
  __REG32  CLRPEND121     : 1;
  __REG32  CLRPEND122     : 1;
  __REG32  CLRPEND123     : 1;
  __REG32  CLRPEND124     : 1;
  __REG32  CLRPEND125     : 1;
  __REG32  CLRPEND126     : 1;
  __REG32  CLRPEND127     : 1;
} __clrpend3_bits;

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

/* Interrupt Priority Registers 92-95 */
typedef struct {
  __REG32  PRI_92         : 8;
  __REG32  PRI_93         : 8;
  __REG32  PRI_94         : 8;
  __REG32  PRI_95         : 8;
} __pri23_bits;

/* Interrupt Priority Registers 96-99 */
typedef struct {
  __REG32  PRI_96         : 8;
  __REG32  PRI_97         : 8;
  __REG32  PRI_98         : 8;
  __REG32  PRI_99         : 8;
} __pri24_bits;

/* Interrupt Priority Registers 100-103 */
typedef struct {
  __REG32  PRI_100        : 8;
  __REG32  PRI_101        : 8;
  __REG32  PRI_102        : 8;
  __REG32  PRI_103        : 8;
} __pri25_bits;

/* Interrupt Priority Registers 104-107 */
typedef struct {
  __REG32  PRI_104        : 8;
  __REG32  PRI_105        : 8;
  __REG32  PRI_106        : 8;
  __REG32  PRI_107        : 8;
} __pri26_bits;

/* Interrupt Priority Registers 108-111 */
typedef struct {
  __REG32  PRI_108        : 8;
  __REG32  PRI_109        : 8;
  __REG32  PRI_110        : 8;
  __REG32  PRI_111        : 8;
} __pri27_bits;

/* Interrupt Priority Registers 112-115 */
typedef struct {
  __REG32  PRI_112        : 8;
  __REG32  PRI_113        : 8;
  __REG32  PRI_114        : 8;
  __REG32  PRI_115        : 8;
} __pri28_bits;

/* Interrupt Priority Registers 116-119 */
typedef struct {
  __REG32  PRI_116        : 8;
  __REG32  PRI_117        : 8;
  __REG32  PRI_118        : 8;
  __REG32  PRI_119        : 8;
} __pri29_bits;

/* Interrupt Priority Registers 120-123 */
typedef struct {
  __REG32  PRI_120        : 8;
  __REG32  PRI_121        : 8;
  __REG32  PRI_122        : 8;
  __REG32  PRI_123        : 8;
} __pri30_bits;

/* Interrupt Priority Registers 124-127 */
typedef struct {
  __REG32  PRI_124        : 8;
  __REG32  PRI_125        : 8;
  __REG32  PRI_126        : 8;
  __REG32  PRI_127        : 8;
} __pri31_bits;

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


/* TRMOSCPRO Register */
typedef struct{
  __REG32 PROTECT             : 8;
  __REG32                     :24;
} __trmoscpro_bits;

/* TRMOSCEN Register */
typedef struct{
  __REG32 TRIMEN              : 1;
  __REG32                     :31;
} __trmoscen_bits;

/* TRMOSCINIT Register */
typedef struct{
  __REG32 TRIMINITF           : 4;
  __REG32                     : 4;
  __REG32 TRIMINITC           : 6;
  __REG32                     :18;
} __trmoscinit_bits;

/* TRMOSCSET Register */
typedef struct{
  __REG32 TRIMSETF            : 4;
  __REG32                     : 4;
  __REG32 TRIMSETC            : 6;
  __REG32                     :18;
} __trmoscset_bits;


/* DMACxStatus (DMAC Status Register) */
typedef struct{
  __REG32 MASTER_ENABLE       : 1;
  __REG32                     :31;
} __dmacstatus_bits;

/* DMACxCfg (DMAC Configuration Register) */
typedef struct{
  __REG32 MASTER_ENABLE       : 1;
  __REG32                     :31;
} __dmaccfg_bits;

/* DMACxCtrlBasePtr (DMAC Channel control data base pointer Register) */
typedef struct{
  __REG32                     :10;
  __REG32 CTRL_BASR_PTR       :22;
} __dmacctrlbaseptr_bits;

/* DMACxChnlSwRequest, DMACxChnlUseburstSet, DMACBChnlUseburstClr, DMACxChnlReqMaskSet */
/* DMACxChnlReqMaskClr, DMACxChnlEnableSet, DMACxChnlEnableClr, DMACxChnlPriAltSet  */
/* DMACxChnlPriAltClr, DMACxChnlPrioritySet, DMACxChnlPriorityClr */
typedef struct{
  __REG32  CH0    : 1;
  __REG32  CH1    : 1;
  __REG32  CH2    : 1;
  __REG32  CH3    : 1;
  __REG32  CH4    : 1;
  __REG32  CH5    : 1;
  __REG32  CH6    : 1;
  __REG32  CH7    : 1;
  __REG32  CH8    : 1;
  __REG32  CH9    : 1;
  __REG32  CH10   : 1;
  __REG32  CH11   : 1;
  __REG32  CH12   : 1;
  __REG32  CH13   : 1;
  __REG32  CH14   : 1;
  __REG32  CH15   : 1;
  __REG32  CH16   : 1;
  __REG32  CH17   : 1;
  __REG32  CH18   : 1;
  __REG32  CH19   : 1;
  __REG32  CH20   : 1;
  __REG32  CH21   : 1;
  __REG32  CH22   : 1;
  __REG32  CH23   : 1;
  __REG32  CH24   : 1;
  __REG32  CH25   : 1;
  __REG32  CH26   : 1;
  __REG32  CH27   : 1;
  __REG32  CH28   : 1;
  __REG32  CH29   : 1;
  __REG32  CH30   : 1;
  __REG32  CH31   : 1;
} __dmacchnlctl_bits;

/* DMACxErrClr (Bus error clear Register) */
typedef struct{
  __REG32  ERR_CLR    : 1;
  __REG32             :31;
} __dmacerrclr_bits;

/* PORT A Register */
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

/* PORT A Control Register */
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

/* PORT A Function Register 1 */
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

/* PORT A Function Register 2 */
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

/* PORT A Function Register 3 */
typedef struct {
  __REG8         : 3;
  __REG8  PA3F3  : 1;
  __REG8         : 1;
  __REG8  PA5F3  : 1;
  __REG8  PA6F3  : 1;
  __REG8  PA7F3  : 1;
} __pafr3_bits;

/* PORT A Function Register 4 */
typedef struct {
  __REG8         : 7;
  __REG8  PA7F4  : 1;
} __pafr4_bits;

/* PORT A Function Register 5 */
typedef struct {
  __REG8         : 7;
  __REG8  PA7F5  : 1;
} __pafr5_bits;

/* PortA open drain control register */
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

/* PORT A Pull-Up Control Register */
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

/* PORT A Pull-Down Control Register */
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

/* PORT A Input Enable Control Register */
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
  __REG8       : 1;
} __pb_bits;

/* PORT B Control Register */
typedef struct {
  __REG8  PB0C  : 1;
  __REG8  PB1C  : 1;
  __REG8  PB2C  : 1;
  __REG8  PB3C  : 1;
  __REG8  PB4C  : 1;
  __REG8  PB5C  : 1;
  __REG8  PB6C  : 1;
  __REG8       : 1;
} __pbcr_bits;

/* PORT B Function Register 1 */
typedef struct {
  __REG8  PB0F1  : 1;
  __REG8  PB1F1  : 1;
  __REG8  PB2F1  : 1;
  __REG8  PB3F1  : 1;
  __REG8  PB4F1  : 1;
  __REG8  PB5F1  : 1;
  __REG8  PB6F1  : 1;
  __REG8         : 1;
} __pbfr1_bits;

/* PORT B Function Register 2 */
typedef struct {
  __REG8         : 2;
  __REG8  PB2F2  : 1;
  __REG8  PB3F2  : 1;
  __REG8  PB4F2  : 1;
  __REG8  PB5F2  : 1;
  __REG8  PB6F2  : 1;
  __REG8         : 1;
} __pbfr2_bits;

/* PORT B Function Register 3 */
typedef struct {
  __REG8  PB0F3  : 1;
  __REG8  PB1F3  : 1;
  __REG8  PB2F3  : 1;
  __REG8  PB3F3  : 1;
  __REG8  PB4F3  : 1;
  __REG8  PB5F3  : 1;
  __REG8         : 2;
} __pbfr3_bits;

/* PORT B Function Register 4 */
typedef struct {
  __REG8         : 2;
  __REG8  PB2F4  : 1;
  __REG8  PB3F4  : 1;
  __REG8  PB4F4  : 1;
  __REG8  PB5F4  : 1;
  __REG8  PB6F4  : 1;
  __REG8         : 1;
} __pbfr4_bits;

/* PortB open drain control register */
typedef struct {
  __REG8  PB0OD  : 1;
  __REG8  PB1OD  : 1;
  __REG8  PB2OD  : 1;
  __REG8  PB3OD  : 1;
  __REG8  PB4OD  : 1;
  __REG8  PB5OD  : 1;
  __REG8  PB6OD  : 1;
  __REG8         : 1;
} __pbod_bits;

/* PORT B Pull-Up Control Register */
typedef struct {
  __REG8  PB0UP  : 1;
  __REG8  PB1UP  : 1;
  __REG8  PB2UP  : 1;
  __REG8  PB3UP  : 1;
  __REG8  PB4UP  : 1;
  __REG8  PB5UP  : 1;
  __REG8  PB6UP  : 1;
  __REG8         : 1;
} __pbpup_bits;

/* PORT B Pull-Down Control Register */
typedef struct {
  __REG8  PB0DN  : 1;
  __REG8  PB1DN  : 1;
  __REG8  PB2DN  : 1;
  __REG8  PB3DN  : 1;
  __REG8  PB4DN  : 1;
  __REG8  PB5DN  : 1;
  __REG8  PB6DN  : 1;
  __REG8         : 1;
} __pbpdn_bits;

/* PORT B Input Enable Control Register */
typedef struct {
  __REG8  PB0IE  : 1;
  __REG8  PB1IE  : 1;
  __REG8  PB2IE  : 1;
  __REG8  PB3IE  : 1;
  __REG8  PB4IE  : 1;
  __REG8  PB5IE  : 1;
  __REG8  PB6IE  : 1;
  __REG8         : 1;
} __pbie_bits;

/* PORT C Register*/
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

/* PORT C Control Register */
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

/* PORT C Function Register 1 */
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

/* PORT C Function Register 2 */
typedef struct {
  __REG8  PC0F2  : 1;
  __REG8  PC1F2  : 1;
  __REG8         : 6;
} __pcfr2_bits;

/* PortC open drain control register */
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

/* PORT C Pull-Up Control Register */
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

/* PORT C Pull-Down Control Register */
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

/* PORT D Register */
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

/* PORT D Control Register */
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

/* PORT D Function Register 1 */
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

/* PortD open drain control register */
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

/*PORT D Pull-Down Control Register */
typedef struct {
  __REG8  PD0DN  : 1;
  __REG8  PD1DN  : 1;
  __REG8  PD2DN  : 1;
  __REG8  PD3DN  : 1;
  __REG8  PD4DN  : 1;
  __REG8  PD5DN  : 1;
  __REG8  PD6DN  : 1;
  __REG8  PD7DN  : 1;
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
  __REG8  PD7IE  : 1;
} __pdie_bits;

/* PORT E Register */
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

/* PORT E Control Register */
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

/* PORT E Function Register 1 */
typedef struct {
  __REG8         : 1;
  __REG8  PE1F1  : 1;
  __REG8  PE2F1  : 1;
  __REG8  PE3F1  : 1;
  __REG8  PE4F1  : 1;
  __REG8  PE5F1  : 1;
  __REG8  PE6F1  : 1;
  __REG8         : 1;
} __pefr1_bits;

/* PORT E Function Register 2 */
typedef struct {
  __REG8  PE0F2  : 1;
  __REG8  PE1F2  : 1;
  __REG8  PE2F2  : 1;
  __REG8  PE3F2  : 1;
  __REG8  PE4F2  : 1;
  __REG8  PE5F2  : 1;
  __REG8  PE6F2  : 1;
  __REG8  PE7F2  : 1;
} __pefr2_bits;

/* PORT E Function Register 3 */
typedef struct {
  __REG8  PE0F3  : 1;
  __REG8  PE1F3  : 1;
  __REG8  PE2F3  : 1;
  __REG8  PE3F3  : 1;
  __REG8  PE4F3  : 1;
  __REG8  PE5F3  : 1;
  __REG8  PE6F3  : 1;
  __REG8  PE7F3  : 1;
} __pefr3_bits;

/* PORT E Function Register 4 */
typedef struct {
  __REG8  PE0F4  : 1;
  __REG8  PE1F4  : 1;
  __REG8         : 1;
  __REG8  PE3F4  : 1;
  __REG8  PE4F4  : 1;
  __REG8         : 2;
  __REG8  PE7F4  : 1;
} __pefr4_bits;

/* PORT E Function Register 5 */
typedef struct {
  __REG8  PE0F5  : 1;
  __REG8  PE1F5  : 1;
  __REG8  PE2F5  : 1;
  __REG8  PE3F5  : 1;
  __REG8  PE4F5  : 1;
  __REG8         : 2;
  __REG8  PE7F5  : 1;
} __pefr5_bits;

/* PORT E Open Drain Control Register */
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

/* PORT E Pull-Up Control Register */
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

/*PORT E Pull-Down Control Register */
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

/* PORT E Input Enable Control Register */
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

/* PORT F Register */
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

/* PORT F Control Register */
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

/* PORT F Function Register 1 */
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

/* PORT F Function Register 2 */
typedef struct {
  __REG8         : 4;
  __REG8  PF4F2  : 1;
  __REG8  PF5F2  : 1;
  __REG8  PF6F2  : 1;
  __REG8  PF7F2  : 1;
} __pffr2_bits;

/* PORT F Function Register 3 */
typedef struct {
  __REG8  PF0F3  : 1;
  __REG8  PF1F3  : 1;
  __REG8  PF2F3  : 1;
  __REG8  PF3F3  : 1;
  __REG8  PF4F3  : 1;
  __REG8  PF5F3  : 1;
  __REG8  PF6F3  : 1;
  __REG8  PF7F3  : 1;
} __pffr3_bits;

/* PORT F Function Register 4 */
typedef struct {
  __REG8         : 1;
  __REG8  PF1F4  : 1;
  __REG8  PF2F4  : 1;
  __REG8         : 2;
  __REG8  PF5F4  : 1;
  __REG8  PF6F4  : 1;
  __REG8  PF7F4  : 1;
} __pffr4_bits;

/* PORT F Open Drain Control Register */
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

/* PORT F Pull-Up Control Register */
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

/*PORT F Pull-Down Control Register */
typedef struct {
  __REG8  PF0DN  : 1;
  __REG8  PF1DN  : 1;
  __REG8  PF2DN  : 1;
  __REG8  PF3DN  : 1;
  __REG8  PF4DN  : 1;
  __REG8  PF5DN  : 1;
  __REG8  PF6DN  : 1;
  __REG8  PF7DN  : 1;
} __pfpdn_bits;

/* PORT F Input Enable Control Register */
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

/* PORT G Register */
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

/* PortG control register */
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

/* PORT G Function Register 1 */
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

/* PORT G Function Register 2 */
typedef struct {
  __REG8         : 1;
  __REG8  PG1F2  : 1;
  __REG8  PG2F2  : 1;
  __REG8  PG3F2  : 1;
  __REG8  PG4F2  : 1;
  __REG8  PG5F2  : 1;
  __REG8  PG6F2  : 1;
  __REG8  PG7F2  : 1;
} __pgfr2_bits;

/* PORT G Function Register 3 */
typedef struct {
  __REG8  PG0F3  : 1;
  __REG8  PG1F3  : 1;
  __REG8  PG2F3  : 1;
  __REG8  PG3F3  : 1;
  __REG8  PG4F3  : 1;
  __REG8  PG5F3  : 1;
  __REG8  PG6F3  : 1;
  __REG8  PG7F3  : 1;
} __pgfr3_bits;

/* PORT G Function Register 4 */
typedef struct {
  __REG8         : 2;
  __REG8  PG2F4  : 1;
  __REG8  PG3F4  : 1;
  __REG8         : 4;
} __pgfr4_bits;

/* PORT G Open Drain Control Register */
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

/* PORT G Pull-Up Control Register */
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

/*PORT G Pull-Down Control Register */
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

/* PORT G Input Enable Control Register */
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

/* PORT H Register */
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

/* PortH control register */
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

/* PORT H Function Register 1 */
typedef struct {
  __REG8  PH0F1  : 1;
  __REG8  PH1F1  : 1;
  __REG8  PH2F1  : 1;
  __REG8  PH3F1  : 1;
  __REG8         : 1;
  __REG8  PH5F1  : 1;
  __REG8  PH6F1  : 1;
  __REG8  PH7F1  : 1;
} __phfr1_bits;

/* PORT H Function Register 2 */
typedef struct {
  __REG8  PH0F2  : 1;
  __REG8  PH1F2  : 1;
  __REG8  PH2F2  : 1;
  __REG8  PH3F2  : 1;
  __REG8         : 1;
  __REG8  PH5F2  : 1;
  __REG8         : 1;
  __REG8  PH7F2  : 1;
} __phfr2_bits;

/* PORT H Function Register 3 */
typedef struct {
  __REG8  PH0F3  : 1;
  __REG8  PH1F3  : 1;
  __REG8  PH2F3  : 1;
  __REG8  PH3F3  : 1;
  __REG8         : 4;
} __phfr3_bits;

/* PORT H Function Register 4 */
typedef struct {
  __REG8         : 2;
  __REG8  PH2F4  : 1;
  __REG8  PH3F4  : 1;
  __REG8         : 4;
} __phfr4_bits;

/* PORT H Function Register 5 */
typedef struct {
  __REG8  PH0F5  : 1;
  __REG8  PH1F5  : 1;
  __REG8  PH2F5  : 1;
  __REG8  PH3F5  : 1;
  __REG8  PH4F5  : 1;
  __REG8         : 3;
} __phfr5_bits;

/* PortH open drain control register */
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

/* PORT H Pull-Up Control Register */
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

/*PORT H Pull-Down Control Register */
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

/* PORT H Input Enable Control Register */
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

/* PORT I Register */
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

/* PORT I Control Register */
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

/* PORT I Function Register 1 */
typedef struct {
  __REG8  PI0F1  : 1;
  __REG8  PI1F1  : 1;
  __REG8  PI2F1  : 1;
  __REG8  PI3F1  : 1;
  __REG8         : 4;
} __pifr1_bits;

/* PORT I Function Register 2 */
typedef struct {
  __REG8         : 3;
  __REG8  PI3F2  : 1;
  __REG8         : 4;
} __pifr2_bits;

/* Port I open drain control register */
typedef struct {
  __REG8  PI0OD  : 1;
  __REG8  PI1OD  : 1;
  __REG8  PI2OD  : 1;
  __REG8  PI3OD  : 1;
  __REG8  PI4OD  : 1;
  __REG8  PI5OD  : 1;
  __REG8  PI6OD  : 1;
  __REG8  PI7OD  : 1;
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

/*PORT I Pull-Down Control Register */
typedef struct {
  __REG8  PI0DN  : 1;
  __REG8  PI1DN  : 1;
  __REG8  PI2DN  : 1;
  __REG8  PI3DN  : 1;
  __REG8  PI4DN  : 1;
  __REG8  PI5DN  : 1;
  __REG8  PI6DN  : 1;
  __REG8  PI7DN  : 1;
} __pipdn_bits;

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

/* PORT J Register */
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

/* Port J output control register */
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

/* Port J open drain control register */
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

/* PORT J Pull-Up Control Register */
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

/* PORT J Pull-Down Control Register */
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

/* PORT J Input Enable Control Register */
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

/* PORT K Register */
typedef struct {
  __REG8  PK0  : 1;
  __REG8  PK1  : 1;
  __REG8  PK2  : 1;
  __REG8  PK3  : 1;
  __REG8  PK4  : 1;
  __REG8       : 3;
} __pk_bits;

/* Port K output control register */
typedef struct {
  __REG8  PK0C  : 1;
  __REG8  PK1C  : 1;
  __REG8  PK2C  : 1;
  __REG8  PK3C  : 1;
  __REG8  PK4C  : 1;
  __REG8        : 3;
} __pkcr_bits;

/* PORT K Function Register 1 */
typedef struct {
  __REG8  PK0F1  : 1;
  __REG8  PK1F1  : 1;
  __REG8  PK2F1  : 1;
  __REG8  PK3F1  : 1;
  __REG8  PK4F1  : 1;
  __REG8         : 3;
} __pkfr1_bits;

/* PORT K Function Register 2 */
typedef struct {
  __REG8         : 1;
  __REG8  PK1F2  : 1;
  __REG8  PK2F2  : 1;
  __REG8  PK3F2  : 1;
  __REG8  PK4F2  : 1;
  __REG8         : 3;
} __pkfr2_bits;

/* PORT K Function Register 3 */
typedef struct {
  __REG8         : 1;
  __REG8  PK1F3  : 1;
  __REG8  PK2F3  : 1;
  __REG8  PK3F3  : 1;
  __REG8  PK4F3  : 1;
  __REG8         : 3;
} __pkfr3_bits;

/* PORT K Function Register 4 */
typedef struct {
  __REG8         : 1;
  __REG8  PK1F4  : 1;
  __REG8         : 6;
} __pkfr4_bits;

/* Port K open drain control register */
typedef struct {
  __REG8  PK0OD  : 1;
  __REG8  PK1OD  : 1;
  __REG8  PK2OD  : 1;
  __REG8  PK3OD  : 1;
  __REG8  PK4OD  : 1;
  __REG8         : 3;
} __pkod_bits;

/* PORT K Pull-Up Control Register */
typedef struct {
  __REG8  PK0UP  : 1;
  __REG8  PK1UP  : 1;
  __REG8  PK2UP  : 1;
  __REG8  PK3UP  : 1;
  __REG8  PK4UP  : 1;
  __REG8         : 3;
} __pkpup_bits;

/* PORT K Pull-Down Control Register */
typedef struct {
  __REG8  PK0DN  : 1;
  __REG8  PK1DN  : 1;
  __REG8  PK2DN  : 1;
  __REG8  PK3DN  : 1;
  __REG8  PK4DN  : 1;
  __REG8         : 3;
} __pkpdn_bits;

/* PORT K Input Enable Control Register */
typedef struct {
  __REG8  PK0IE  : 1;
  __REG8  PK1IE  : 1;
  __REG8  PK2IE  : 1;
  __REG8  PK3IE  : 1;
  __REG8  PK4IE  : 1;
  __REG8         : 3;
} __pkie_bits;

/* PortL register */
typedef struct {
  __REG8  PL0    : 1;
  __REG8  PL1    : 1;
  __REG8  PL2    : 1;
  __REG8  PL3    : 1;
  __REG8  PL4    : 1;
  __REG8  PL5    : 1;
  __REG8  PL6    : 1;
  __REG8  PL7    : 1;
} __pl_bits;

/* PortL control register */
typedef struct {
  __REG8  PL0C   : 1;
  __REG8  PL1C   : 1;
  __REG8  PL2C   : 1;
  __REG8  PL3C   : 1;
  __REG8  PL4C   : 1;
  __REG8  PL5C   : 1;
  __REG8  PL6C   : 1;
  __REG8  PL7C   : 1;
} __plcr_bits;

/* PortL function register1 */
typedef struct {
  __REG8  PL0F1  : 1;
  __REG8  PL1F1  : 1;
  __REG8  PL2F1  : 1;
  __REG8  PL3F1  : 1;
  __REG8  PL4F1  : 1;
  __REG8  PL5F1  : 1;
  __REG8  PL6F1  : 1;
  __REG8  PL7F1  : 1;
} __plfr1_bits;

/* PortL function register2 */
typedef struct {
  __REG8  PL0F2  : 1;
  __REG8  PL1F2  : 1;
  __REG8  PL2F2  : 1;
  __REG8  PL3F2  : 1;
  __REG8  PL4F2  : 1;
  __REG8  PL5F2  : 1;
  __REG8  PL6F2  : 1;
  __REG8  PL7F2  : 1;
} __plfr2_bits;

/* PortL function register3 */
typedef struct {
  __REG8  PL0F3  : 1;
  __REG8  PL1F3  : 1;
  __REG8  PL2F3  : 1;
  __REG8  PL3F3  : 1;
  __REG8         : 4;
} __plfr3_bits;

/* PortL function register4 */
typedef struct {
  __REG8  PL0F4  : 1;
  __REG8  PL1F4  : 1;
  __REG8  PL2F4  : 1;
  __REG8  PL3F4  : 1;
  __REG8         : 4;
} __plfr4_bits;

/* PortL function register5 */
typedef struct {
  __REG8         : 1;
  __REG8  PL1F5  : 1;
  __REG8  PL2F5  : 1;
  __REG8  PL3F5  : 1;
  __REG8         : 4;
} __plfr5_bits;

/* PortL function register6 */
typedef struct {
  __REG8         : 3;
  __REG8  PL3F6  : 1;
  __REG8         : 4;
} __plfr6_bits;

/* PortL open drain control register */
typedef struct {
  __REG8  PL0OD  : 1;
  __REG8  PL1OD  : 1;
  __REG8  PL2OD  : 1;
  __REG8  PL3OD  : 1;
  __REG8  PL4OD  : 1;
  __REG8  PL5OD  : 1;
  __REG8  PL6OD  : 1;
  __REG8  PL7OD  : 1;
} __plod_bits;

/* PortL pull-up control register */
typedef struct {
  __REG8  PL0UP  : 1;
  __REG8  PL1UP  : 1;
  __REG8  PL2UP  : 1;
  __REG8  PL3UP  : 1;
  __REG8  PL4UP  : 1;
  __REG8  PL5UP  : 1;
  __REG8  PL6UP  : 1;
  __REG8  PL7UP  : 1;
} __plpup_bits;

/* PORT L Pull-Down Control Register */
typedef struct {
  __REG8  PL0DN  : 1;
  __REG8  PL1DN  : 1;
  __REG8  PL2DN  : 1;
  __REG8  PL3DN  : 1;
  __REG8  PL4DN  : 1;
  __REG8  PL5DN  : 1;
  __REG8  PL6DN  : 1;
  __REG8  PL7DN  : 1;
} __plpdn_bits;

/* PortL input enable control register */
typedef struct {
  __REG8  PL0IE  : 1;
  __REG8  PL1IE  : 1;
  __REG8  PL2IE  : 1;
  __REG8  PL3IE  : 1;
  __REG8  PL4IE  : 1;
  __REG8  PL5IE  : 1;
  __REG8  PL6IE  : 1;
  __REG8  PL7IE  : 1;
} __plie_bits;

/* PortM register */
typedef struct {
  __REG8  PM0    : 1;
  __REG8  PM1    : 1;
  __REG8  PM2    : 1;
  __REG8  PM3    : 1;
  __REG8         : 4;
} __pm_bits;

/* PortM control register */
typedef struct {
  __REG8  PM0C   : 1;
  __REG8  PM1C   : 1;
  __REG8  PM2C   : 1;
  __REG8  PM3C   : 1;
  __REG8         : 4;
} __pmcr_bits;

/* PortM function register1 */
typedef struct {
  __REG8  PM0F1  : 1;
  __REG8  PM1F1  : 1;
  __REG8  PM2F1  : 1;
  __REG8         : 5;
} __pmfr1_bits;

/* PortM function register2 */
typedef struct {
  __REG8  PM0F2  : 1;
  __REG8  PM1F2  : 1;
  __REG8  PM2F2  : 1;
  __REG8  PM3F2  : 1;
  __REG8         : 4;
} __pmfr2_bits;

/* PortM open drain control register */
typedef struct {
  __REG8  PM0OD  : 1;
  __REG8  PM1OD  : 1;
  __REG8  PM2OD  : 1;
  __REG8  PM3OD  : 1;
  __REG8         : 4;
} __pmod_bits;

/* PortM pull-up control register */
typedef struct {
  __REG8  PM0UP  : 1;
  __REG8  PM1UP  : 1;
  __REG8  PM2UP  : 1;
  __REG8  PM3UP  : 1;
  __REG8         : 4;
} __pmpup_bits;

/* PortM pull-down control register */
typedef struct {
  __REG8  PM0DN  : 1;
  __REG8  PM1DN  : 1;
  __REG8  PM2DN  : 1;
  __REG8  PM3DN  : 1;
  __REG8         : 4;
} __pmpdn_bits;

/* PortM input enable control register */
typedef struct {
  __REG8  PM0IE  : 1;
  __REG8  PM1IE  : 1;
  __REG8  PM2IE  : 1;
  __REG8  PM3IE  : 1;
  __REG8         : 4;
} __pmie_bits;

/* PortN register */
typedef struct {
  __REG8  PN0    : 1;
  __REG8  PN1    : 1;
  __REG8  PN2    : 1;
  __REG8  PN3    : 1;
  __REG8  PN4    : 1;
  __REG8  PN5    : 1;
  __REG8         : 2;
} __pn_bits;

/* PortN control register */
typedef struct {
  __REG8  PN0C   : 1;
  __REG8  PN1C   : 1;
  __REG8  PN2C   : 1;
  __REG8  PN3C   : 1;
  __REG8  PN4C   : 1;
  __REG8  PN5C   : 1;
  __REG8         : 2;
} __pncr_bits;

/* PortN function register1 */
typedef struct {
  __REG8  PN0F1  : 1;
  __REG8  PN1F1  : 1;
  __REG8  PN2F1  : 1;
  __REG8  PN3F1  : 1;
  __REG8  PN4F1  : 1;
  __REG8  PN5F1  : 1;
  __REG8         : 2;
} __pnfr1_bits;

/* PortN function register2 */
typedef struct {
  __REG8         : 4;
  __REG8  PN4F2  : 1;
  __REG8  PN5F2  : 1;
  __REG8         : 2;
} __pnfr2_bits;

/* PortN open drain control register */
typedef struct {
  __REG8  PN0OD  : 1;
  __REG8  PN1OD  : 1;
  __REG8  PN2OD  : 1;
  __REG8  PN3OD  : 1;
  __REG8  PN4OD  : 1;
  __REG8  PN5OD  : 1;
  __REG8         : 2;
} __pnod_bits;

/* PortN pull-up control register */
typedef struct {
  __REG8  PN0UP  : 1;
  __REG8  PN1UP  : 1;
  __REG8  PN2UP  : 1;
  __REG8  PN3UP  : 1;
  __REG8  PN4UP  : 1;
  __REG8  PN5UP  : 1;
  __REG8         : 2;
} __pnpup_bits;

/* PortN pull-down control register */
typedef struct {
  __REG8  PN0DN  : 1;
  __REG8  PN1DN  : 1;
  __REG8  PN2DN  : 1;
  __REG8  PN3DN  : 1;
  __REG8  PN4DN  : 1;
  __REG8  PN5DN  : 1;
  __REG8         : 2;
} __pnpdn_bits;

/* PortN input enable control register */
typedef struct {
  __REG8  PN0IE  : 1;
  __REG8  PN1IE  : 1;
  __REG8  PN2IE  : 1;
  __REG8  PN3IE  : 1;
  __REG8  PN4IE  : 1;
  __REG8  PN5IE  : 1;
  __REG8         : 2;
} __pnie_bits;

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
  __REG32  SA       :16;
} __exbas_bits;

/*External Bus Chip Control Register*/
typedef struct {
  __REG32  CSW0     : 1;
  __REG32  CSW      : 2;
  __REG32           : 5;
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

/*TMRBn enable register (channels 0 through 7)*/
typedef struct {
  __REG32           : 6;
  __REG32  TBHALT   : 1;
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
  __REG32  TBINSEL  : 1;
  __REG32  I2TB     : 1;
  __REG32           : 1;
  __REG32  TBSYNC   : 1;
  __REG32           : 1;
  __REG32  TBWBF    : 1;
  __REG32           :24;
} __tbxcr_bits;

/*TMRB mode register (channels 0 thorough 7)*/
typedef struct {
  __REG32  TBCLK    : 3;
  __REG32  TBCLE    : 1;
  __REG32  TBCPM    : 2;
  __REG32  TBCP     : 1;
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

/*TMRB DMA enable register (channels 0 through 7)*/
typedef struct {
  __REG32  TBDMAEN0 : 1;
  __REG32  TBDMAEN1 : 1;
  __REG32  TBDMAEN2 : 1;
  __REG32           :29;
} __tbxdma_bits;

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

/* UARTDR (UART Data Register) */
typedef struct{
  __REG32 DATA                    : 8;
  __REG32 FE                      : 1;
  __REG32 PE                      : 1;
  __REG32 BE                      : 1;
  __REG32 OE                      : 1;
  __REG32                         :20;
} __uartdr_bits;

/* UARTRSR (UART receive status register) */
typedef struct
{
  __REG32 FE                    : 1;
  __REG32 PE                    : 1;
  __REG32 BE                    : 1;
  __REG32 OE                    : 1;
  __REG32                       :28;
} __uartrsr_bits;

/* UARTFR (UART flag register) */
typedef struct
{
  __REG32 CTS                   : 1;
  __REG32 DSR                   : 1;
  __REG32 DCD                   : 1;
  __REG32 BUSY                  : 1;
  __REG32 RXFE                  : 1;
  __REG32 TXFF                  : 1;
  __REG32 RXFF                  : 1;
  __REG32 TXFE                  : 1;
  __REG32 RI                    : 1;
  __REG32                       :23;
} __uartfr_bits;

/* UARTILPR (UART IrDA low-power counter register) */
typedef struct
{
  __REG32 ILPDVSR               : 8;
  __REG32                       :24;
} __uartilpr_bits;

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
  __REG32 SIREN                 : 1;
  __REG32 SIRLP                 : 1;
  __REG32                       : 5;
  __REG32 TXE                   : 1;
  __REG32 RXE                   : 1;
  __REG32 DTR                   : 1;
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
  __REG32 RIMIM                 : 1;
  __REG32 CTSMIM                : 1;
  __REG32 DCDMIM                : 1;
  __REG32 DSRMIM                : 1;
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
  __REG32 RIRMIS                : 1;
  __REG32 CTSRMIS               : 1;
  __REG32 DCDRMIS               : 1;
  __REG32 DSRRMIS               : 1;
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
  __REG32 RIMMIS                : 1;
  __REG32 CTSMMIS               : 1;
  __REG32 DCDMMIS               : 1;
  __REG32 DSRMMIS               : 1;
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
  __REG32 RIMIC                 : 1;
  __REG32 CTSMIC                : 1;
  __REG32 DCDMIC                : 1;
  __REG32 DSRMIC                : 1;
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

/*SSPxDMACR (SSP DMA control register)*/
typedef struct {
  __REG32 RXDMAE  : 1;
  __REG32 TXDMAE  : 1;
  __REG32         :30;
} __sspdmacr_bits;

/*USBPLLCR (USB PLL control register)*/
typedef struct {
  __REG32 USBPLLON    : 1;
  __REG32             :31;
} __usbpllcr_bits;

/*USBPLLEN (USB PLL enable register)*/
typedef struct {
  __REG32 USBDEN      : 1;
  __REG32 USBHEN      : 1;
  __REG32             :30;
} __usbpllen_bits;

/*USBPLLSEL (USB PLL select register)*/
typedef struct {
  __REG32 USBPLLSEL   : 1;
  __REG32 USBPLLSET   :15;
  __REG32             :16;
} __usbpllsel_bits;

/* HcRevision Register */
typedef struct {
  __REG32  Rev            : 8;
  __REG32                 :24;
} __hcrevision_bits;

/* HcControl Register */
typedef struct {
  __REG32  CBSR           : 2;
  __REG32  PLE            : 1;
  __REG32  IE             : 1;
  __REG32  CLE            : 1;
  __REG32  BLE            : 1;
  __REG32  HCFS           : 2;
  __REG32  IR             : 1;
  __REG32  RWC            : 1;
  __REG32  RWE            : 1;
  __REG32                 :21;
} __hccontrol_bits;

/* HcCommandStatus Register */
typedef struct {
  __REG32  HCR            : 1;
  __REG32  CLF            : 1;
  __REG32  BLF            : 1;
  __REG32  OCR            : 1;
  __REG32                 :12;
  __REG32  SOC            : 2;
  __REG32                 :14;
} __hccommandstatus_bits;

/* HcInterruptStatus Register */
typedef struct {
  __REG32  SO             : 1;
  __REG32  WDH            : 1;
  __REG32  SF             : 1;
  __REG32  RD             : 1;
  __REG32  UE             : 1;
  __REG32  FNO            : 1;
  __REG32  RHSC           : 1;
  __REG32                 :23;
  __REG32  OC             : 1;
  __REG32                 : 1;
} __hcinterruptstatus_bits;

/* HcInterruptEnable Register
   HcInterruptDisable Register */
typedef struct {
  __REG32  SO             : 1;
  __REG32  WDH            : 1;
  __REG32  SF             : 1;
  __REG32  RD             : 1;
  __REG32  UE             : 1;
  __REG32  FNO            : 1;
  __REG32  RHSC           : 1;
  __REG32                 :23;
  __REG32  OC             : 1;
  __REG32  MIE            : 1;
} __hcinterruptenable_bits;

/* HcFmInterval Register */
typedef struct {
  __REG32  FI             :14;
  __REG32                 : 2;
  __REG32  FSMPS          :15;
  __REG32  FIT            : 1;
} __hcfminterval_bits;

/* HcFmRemaining Register */
typedef struct {
  __REG32  FR             :14;
  __REG32                 :17;
  __REG32  FRT            : 1;
} __hcfmremaining_bits;

/* HcFmNumber Register */
typedef struct {
  __REG32  FN             :16;
  __REG32                 :16;
} __hcfmnumber_bits;

/* HcPeriodicStart Register */
typedef struct {
  __REG32  PS             :14;
  __REG32                 :18;
} __hcperiodstart_bits;

/* HcLSThreshold Register */
typedef struct {
  __REG32  LST            :12;
  __REG32                 :20;
} __hclsthreshold_bits;

/* HcRhDescriptorA Register */
typedef struct {
  __REG32  NDP            : 8;
  __REG32  PSM            : 1;
  __REG32  NPS            : 1;
  __REG32  DT             : 1;
  __REG32  OCPM           : 1;
  __REG32  NOCP           : 1;
  __REG32                 :11;
  __REG32  POTPGT         : 8;
} __hcrhdescriptora_bits;

/* HcRhDescriptorb Register */
typedef struct {
  __REG32  DR             :16;
  __REG32  PPCM           :16;
} __hcrhdescriptorb_bits;

/* HcRhStatus Register */
typedef struct {
  __REG32  LPS            : 1;
  __REG32  OCI            : 1;
  __REG32                 :13;
  __REG32  DRWE           : 1;
  __REG32  LPSC           : 1;
  __REG32  OCIC           : 1;
  __REG32                 :13;
  __REG32  CRWE           : 1;
} __hcrhstatus_bits;

/* HcRhPortStatus Register */
typedef struct {
  __REG32  CCS            : 1;
  __REG32  PES            : 1;
  __REG32  PSS            : 1;
  __REG32  POCI           : 1;
  __REG32  PRS            : 1;
  __REG32                 : 3;
  __REG32  PPS            : 1;
  __REG32  LSDA           : 1;
  __REG32                 : 6;
  __REG32  CSC            : 1;
  __REG32  PESC           : 1;
  __REG32  PSSC           : 1;
  __REG32  OCIC           : 1;
  __REG32  PRSC           : 1;
  __REG32                 :11;
} __hcrhportstatus_bits;

/* HcBCR0 Register */
typedef struct {
  __REG32                 :29;
  __REG32  OVCE           : 1;
  __REG32  TRNS_SUSP      : 1;
  __REG32                 : 1;
} __hcbcr0_bits;

/* UDINTSTS (Interrupt Status register) */
typedef struct
{
  __REG32 int_setup             : 1;
  __REG32 int_status_nak        : 1;
  __REG32 int_status            : 1;
  __REG32 int_rx_zero           : 1;
  __REG32 int_sof               : 1;
  __REG32 int_ep0               : 1;
  __REG32 int_ep                : 1;
  __REG32 int_nak               : 1;
  __REG32 int_suspend_resume    : 1;
  __REG32 int_usb_reset         : 1;
  __REG32 int_usb_reset_end     : 1;
  __REG32                       : 6;
  __REG32 int_mw_set_add        : 1;
  __REG32 int_mw_end_add        : 1;
  __REG32 int_mw_timeout        : 1;
  __REG32 int_mw_ahberr         : 1;
  __REG32 int_mr_end_add        : 1;
  __REG32 int_mr_ep_dset        : 1;
  __REG32 int_mr_ahberr         : 1;
  __REG32 int_udc2_reg_rd       : 1;
  __REG32 int_dmac_reg_rd       : 1;
  __REG32                       : 2;
  __REG32 int_powerdetect       : 1;
  __REG32 int_mw_rerror         : 1;
  __REG32                       : 2;
} __udintsts_bits;

/* UDINTENB (Interrupt Enable register) */
typedef struct
{
  __REG32                       : 8;
  __REG32 suspend_resume_en     : 1;
  __REG32 usb_reset_en          : 1;
  __REG32 usb_reset_end_en      : 1;
  __REG32                       : 6;
  __REG32 mw_set_add_en         : 1;
  __REG32 mw_end_add_en         : 1;
  __REG32 mw_timeout_en         : 1;
  __REG32 mw_ahberr_en          : 1;
  __REG32 mr_end_add_en         : 1;
  __REG32 mr_ep_dset_en         : 1;
  __REG32 mr_ahberr_en          : 1;
  __REG32 udc2_reg_rd_en        : 1;
  __REG32 dmac_reg_rd_en        : 1;
  __REG32                       : 3;
  __REG32 mw_rerror_en          : 1;
  __REG32                       : 2;
} __udintenb_bits;

/* UDMWTOUT (Master Write Timeout register) */
typedef struct
{
  __REG32 timeout_en            : 1;
  __REG32 timeoutset            :31;
} __udmwtout_bits;

/* UDC2STSET (UDC2 Setting register) */
typedef struct
{
  __REG32 tx0                   : 1;
  __REG32                       : 3;
  __REG32 eopb_enable           : 1;
  __REG32                       :27;
} __udc2stset_bits;

/* UDMSTSET (DMAC Setting register) */
typedef struct
{
  __REG32 mw_enable             : 1;
  __REG32 mw_abort              : 1;
  __REG32 mw_reset              : 1;
  __REG32                       : 1;
  __REG32 mr_enable             : 1;
  __REG32 mr_abort              : 1;
  __REG32 mr_reset              : 1;
  __REG32                       : 1;
  __REG32 m_burst_type          : 1;
  __REG32                       :23;
} __udmstset_bits;

/* UDDMACRDREQ (DMAC Read Requset register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 dmardadr              : 6;
  __REG32                       :22;
  __REG32 dmardclr              : 1;
  __REG32 dmardreq              : 1;
} __uddmacrdreq_bits;

/* UDC2RDREQ (UDC2 Read Request register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 udc2rdadr             : 8;
  __REG32                       :20;
  __REG32 udc2rdclr             : 1;
  __REG32 udc2rdreq             : 1;
} __udc2rdreq_bits;

/* UDC2RDVL (UDC2 Read Value register) */
typedef struct
{
  __REG32 udc2rdata             :16;
  __REG32                       :16;
} __udc2rdvl_bits;

/* ARBTSET (Arbiter Setting register) */
typedef struct
{
  __REG32 abtpri_r0             : 2;
  __REG32                       : 2;
  __REG32 abtpri_r1             : 2;
  __REG32                       : 2;
  __REG32 abtpri_w0             : 2;
  __REG32                       : 2;
  __REG32 abtpri_w1             : 2;
  __REG32                       :14;
  __REG32 abtmod                : 1;
  __REG32                       : 2;
  __REG32 abt_en                : 1;
} __udarbtset_bits;

/* UDPWCTL (Power Detect Control register) */
typedef struct
{
  __REG32 usb_reset             : 1;
  __REG32 pw_resetb             : 1;
  __REG32 pw_detect             : 1;
  __REG32 phy_suspend           : 1;
  __REG32 suspend_x             : 1;
  __REG32 phy_resetb            : 1;
  __REG32 phy_remote_wkup       : 1;
  __REG32 wakeup_en             : 1;
  __REG32                       :24;
} __udpwctl_bits;

/* UDMSTSTS (Master Status register) */
typedef struct
{
  __REG32 mwepdset              : 1;
  __REG32 mrepdset              : 1;
  __REG32 mwbfemp               : 1;
  __REG32 mrbfemp               : 1;
  __REG32 mrepempty             : 1;
  __REG32                       :27;
} __udmststs_bits;

/* UD2ADR (Address-State register) */
typedef struct
{
  __REG32 dev_adr               : 7;
  __REG32                       : 1;
  __REG32 Default               : 1;
  __REG32 Addressed             : 1;
  __REG32 Configured            : 1;
  __REG32 Suspend               : 1;
  __REG32 cur_speed             : 2;
  __REG32 ep_bi_mode            : 1;
  __REG32 stage_err             : 1;
  __REG32                       :16;
} __ud2adr_bits;

/* UD2FRM (Frame register) */
typedef struct
{
  __REG32 frame                 :11;
  __REG32                       : 1;
  __REG32 f_status              : 2;
  __REG32                       : 1;
  __REG32 create_sof            : 1;
  __REG32                       :16;
} __ud2frm_bits;

/* UD2CMD (Command register) */
typedef struct
{
  __REG32 com                   : 4;
  __REG32 ep                    : 4;
  __REG32 rx_nullpkt_ep         : 4;
  __REG32                       : 3;
  __REG32 int_toggle            : 1;
  __REG32                       :16;
} __ud2cmd_bits;

/* UD2BRQ (bRequest-bmRequestType register) */
typedef struct
{
  __REG32 recipient             : 5;
  __REG32 req_type              : 2;
  __REG32 dir                   : 1;
  __REG32 request               : 8;
  __REG32                       :16;
} __ud2brq_bits;

/* UD2WVL (wValue register) */
typedef struct
{
  __REG32 value_l               : 8;
  __REG32 value_h               : 8;
  __REG32                       :16;
} __ud2wvl_bits;

/* UD2WIDX (wIndex register) */
typedef struct
{
  __REG32 index_l               : 8;
  __REG32 index_h               : 8;
  __REG32                       :16;
} __ud2widx_bits;

/* UD2WLGTH (wLength register) */
typedef struct
{
  __REG32 length_l              : 8;
  __REG32 length_h              : 8;
  __REG32                       :16;
} __ud2wlgth_bits;

/* UD2INT (INT register)*/
typedef struct
{
  __REG32 i_setup               : 1;
  __REG32 i_status_nak          : 1;
  __REG32 i_status              : 1;
  __REG32 i_rx_data0            : 1;
  __REG32 i_sof                 : 1;
  __REG32 i_ep0                 : 1;
  __REG32 i_ep                  : 1;
  __REG32 i_nak                 : 1;
  __REG32 m_setup               : 1;
  __REG32 m_status_nak          : 1;
  __REG32 m_status              : 1;
  __REG32 m_rx_data0            : 1;
  __REG32 m_sof                 : 1;
  __REG32 m_ep0                 : 1;
  __REG32 m_ep                  : 1;
  __REG32 m_nak                 : 1;
  __REG32                       :16;
} __ud2int_bits;

/* UD2INT (INT register) 
   UD2INTNAK (INT_NAK register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 i_ep1                 : 1;
  __REG32 i_ep2                 : 1;
  __REG32 i_ep3                 : 1;
  __REG32 i_ep4                 : 1;
  __REG32 i_ep5                 : 1;
  __REG32 i_ep6                 : 1;
  __REG32 i_ep7                 : 1;
  __REG32                       :24;
} __ud2intep_bits;

/* UD2INTEPMSK (INT_EP_MASK register)
   UD2INTNAKMSK (INT_NAK_MASK register)*/
typedef struct
{
  __REG32                       : 1;
  __REG32 m_ep1                 : 1;
  __REG32 m_ep2                 : 1;
  __REG32 m_ep3                 : 1;
  __REG32 m_ep4                 : 1;
  __REG32 m_ep5                 : 1;
  __REG32 m_ep6                 : 1;
  __REG32 m_ep7                 : 1;
  __REG32                       :24;
} __ud2intepmsk_bits;

/* UD2INTRX0 (INT_RX_DATA0 register) */
typedef struct
{
  __REG32 rx_d0_ep0             : 1;
  __REG32 rx_d0_ep1             : 1;
  __REG32 rx_d0_ep2             : 1;
  __REG32 rx_d0_ep3             : 1;
  __REG32 rx_d0_ep4             : 1;
  __REG32 rx_d0_ep5             : 1;
  __REG32 rx_d0_ep6             : 1;
  __REG32 rx_d0_ep7             : 1;
  __REG32                       :24;
} __ud2intrx0_bits;

/* UD2EP0MSZ (EPn_Max0PacketSize register) */
typedef struct
{
  __REG32 max_pkt               : 7;
  __REG32                       : 5;
  __REG32 dset                  : 1;
  __REG32                       : 2;
  __REG32 tx_0data              : 1;
  __REG32                       :16;
} __ud2ep0msz_bits;

/* UD2EPnMSZ (EPn_MaxPacketSize register) */
typedef struct
{
  __REG32 max_pkt               :11;
  __REG32                       : 1;
  __REG32 dset                  : 1;
  __REG32                       : 2;
  __REG32 tx_0data              : 1;
  __REG32                       :16;
} __ud2epmsz_bits;

/* UD2EP0STS (EP0_ Status register) */
typedef struct
{
  __REG32                       : 9;
  __REG32 status                : 3;
  __REG32 toggle                : 2;
  __REG32                       : 1;
  __REG32 ep0_mask              : 1;
  __REG32                       :16;
} __ud2ep0sts_bits;

/* UD2EP0DSZ (EP0_Datasize register) */
typedef struct
{
  __REG32 size                  : 7;
  __REG32                       :25;
} __ud2ep0dsz_bits;

/* UD2EPnFIFO (EPn_FIFO register) */
typedef struct
{
  __REG32 data                  :16;
  __REG32                       :16;
} __ud2epfifo_bits;

/* UD2EP1STS (EP1_Status register) */
typedef struct
{
  __REG32 num_mf                : 2;
  __REG32 t_type                : 2;
  __REG32                       : 3;
  __REG32 dir                   : 1;
  __REG32 disable               : 1;
  __REG32 status                : 3;
  __REG32 toggle                : 2;
  __REG32 bus_sel               : 1;
  __REG32 pkt_mode              : 1;
  __REG32                       :16;
} __ud2ep1sts_bits;

/* UD2EP1DSZ (EP1_Datasize register) */
typedef struct
{
  __REG32 size                  :11;
  __REG32                       :21;
} __ud2ep1dsz_bits;

/* CAN Message Identifier (ID0 .. ID3) */
typedef struct {
  __REG32  ID             :29;
  __REG32  RFH            : 1;
  __REG32  GAME           : 1;
  __REG32  IDE            : 1;
} __canmbid_bits;

/* CAN Message Identifier (ID0 .. ID3) */
typedef struct {
  __REG32  ID             :29;
  __REG32  RFH            : 1;
  __REG32  LAME           : 1;
  __REG32  IDE            : 1;
} __canmb31id_bits;

/* CAN Time Stamp Value (TSV)/Message Control Field (MCF) */
typedef struct {
  __REG32  DLC            : 4;
  __REG32  RTR            : 1;
  __REG32                 :11;
  __REG32  TSV            :16;
} __canmbtcr_bits;

/* CAN Mailbox Configuration Register (MC) */
typedef struct {
  __REG32  MC0            : 1;
  __REG32  MC1            : 1;
  __REG32  MC2            : 1;
  __REG32  MC3            : 1;
  __REG32  MC4            : 1;
  __REG32  MC5            : 1;
  __REG32  MC6            : 1;
  __REG32  MC7            : 1;
  __REG32  MC8            : 1;
  __REG32  MC9            : 1;
  __REG32  MC10           : 1;
  __REG32  MC11           : 1;
  __REG32  MC12           : 1;
  __REG32  MC13           : 1;
  __REG32  MC14           : 1;
  __REG32  MC15           : 1;
  __REG32  MC16           : 1;
  __REG32  MC17           : 1;
  __REG32  MC18           : 1;
  __REG32  MC19           : 1;
  __REG32  MC20           : 1;
  __REG32  MC21           : 1;
  __REG32  MC22           : 1;
  __REG32  MC23           : 1;
  __REG32  MC24           : 1;
  __REG32  MC25           : 1;
  __REG32  MC26           : 1;
  __REG32  MC27           : 1;
  __REG32  MC28           : 1;
  __REG32  MC29           : 1;
  __REG32  MC30           : 1;
  __REG32  MC31           : 1;
} __canmc_bits;

/* CAN Mailbox Direction Register (MD) */
typedef struct {
  __REG32  MD0            : 1;
  __REG32  MD1            : 1;
  __REG32  MD2            : 1;
  __REG32  MD3            : 1;
  __REG32  MD4            : 1;
  __REG32  MD5            : 1;
  __REG32  MD6            : 1;
  __REG32  MD7            : 1;
  __REG32  MD8            : 1;
  __REG32  MD9            : 1;
  __REG32  MD10           : 1;
  __REG32  MD11           : 1;
  __REG32  MD12           : 1;
  __REG32  MD13           : 1;
  __REG32  MD14           : 1;
  __REG32  MD15           : 1;
  __REG32  MD16           : 1;
  __REG32  MD17           : 1;
  __REG32  MD18           : 1;
  __REG32  MD19           : 1;
  __REG32  MD20           : 1;
  __REG32  MD21           : 1;
  __REG32  MD22           : 1;
  __REG32  MD23           : 1;
  __REG32  MD24           : 1;
  __REG32  MD25           : 1;
  __REG32  MD26           : 1;
  __REG32  MD27           : 1;
  __REG32  MD28           : 1;
  __REG32  MD29           : 1;
  __REG32  MD30           : 1;
  __REG32  MD31           : 1;
} __canmd_bits;

/* CAN Transmission Request Set Register (TRS) */
typedef struct {
  __REG32  TRS0            : 1;
  __REG32  TRS1            : 1;
  __REG32  TRS2            : 1;
  __REG32  TRS3            : 1;
  __REG32  TRS4            : 1;
  __REG32  TRS5            : 1;
  __REG32  TRS6            : 1;
  __REG32  TRS7            : 1;
  __REG32  TRS8            : 1;
  __REG32  TRS9            : 1;
  __REG32  TRS10           : 1;
  __REG32  TRS11           : 1;
  __REG32  TRS12           : 1;
  __REG32  TRS13           : 1;
  __REG32  TRS14           : 1;
  __REG32  TRS15           : 1;
  __REG32  TRS16           : 1;
  __REG32  TRS17           : 1;
  __REG32  TRS18           : 1;
  __REG32  TRS19           : 1;
  __REG32  TRS20           : 1;
  __REG32  TRS21           : 1;
  __REG32  TRS22           : 1;
  __REG32  TRS23           : 1;
  __REG32  TRS24           : 1;
  __REG32  TRS25           : 1;
  __REG32  TRS26           : 1;
  __REG32  TRS27           : 1;
  __REG32  TRS28           : 1;
  __REG32  TRS29           : 1;
  __REG32  TRS30           : 1;
  __REG32                  : 1;
} __cantrs_bits;

/* CAN Transmission Request Reset Register (TRR) */
typedef struct {
  __REG32  TRR0            : 1;
  __REG32  TRR1            : 1;
  __REG32  TRR2            : 1;
  __REG32  TRR3            : 1;
  __REG32  TRR4            : 1;
  __REG32  TRR5            : 1;
  __REG32  TRR6            : 1;
  __REG32  TRR7            : 1;
  __REG32  TRR8            : 1;
  __REG32  TRR9            : 1;
  __REG32  TRR10           : 1;
  __REG32  TRR11           : 1;
  __REG32  TRR12           : 1;
  __REG32  TRR13           : 1;
  __REG32  TRR14           : 1;
  __REG32  TRR15           : 1;
  __REG32  TRR16           : 1;
  __REG32  TRR17           : 1;
  __REG32  TRR18           : 1;
  __REG32  TRR19           : 1;
  __REG32  TRR20           : 1;
  __REG32  TRR21           : 1;
  __REG32  TRR22           : 1;
  __REG32  TRR23           : 1;
  __REG32  TRR24           : 1;
  __REG32  TRR25           : 1;
  __REG32  TRR26           : 1;
  __REG32  TRR27           : 1;
  __REG32  TRR28           : 1;
  __REG32  TRR29           : 1;
  __REG32  TRR30           : 1;
  __REG32                  : 1;
} __cantrr_bits;

/* CAN Transmission Acknowledge Register (TA) */
typedef struct {
  __REG32  TA0            : 1;
  __REG32  TA1            : 1;
  __REG32  TA2            : 1;
  __REG32  TA3            : 1;
  __REG32  TA4            : 1;
  __REG32  TA5            : 1;
  __REG32  TA6            : 1;
  __REG32  TA7            : 1;
  __REG32  TA8            : 1;
  __REG32  TA9            : 1;
  __REG32  TA10           : 1;
  __REG32  TA11           : 1;
  __REG32  TA12           : 1;
  __REG32  TA13           : 1;
  __REG32  TA14           : 1;
  __REG32  TA15           : 1;
  __REG32  TA16           : 1;
  __REG32  TA17           : 1;
  __REG32  TA18           : 1;
  __REG32  TA19           : 1;
  __REG32  TA20           : 1;
  __REG32  TA21           : 1;
  __REG32  TA22           : 1;
  __REG32  TA23           : 1;
  __REG32  TA24           : 1;
  __REG32  TA25           : 1;
  __REG32  TA26           : 1;
  __REG32  TA27           : 1;
  __REG32  TA28           : 1;
  __REG32  TA29           : 1;
  __REG32  TA30           : 1;
  __REG32                 : 1;
} __canta_bits;

/* CAN Abort Acknowledge Register (AA) */
typedef struct {
  __REG32  AA0            : 1;
  __REG32  AA1            : 1;
  __REG32  AA2            : 1;
  __REG32  AA3            : 1;
  __REG32  AA4            : 1;
  __REG32  AA5            : 1;
  __REG32  AA6            : 1;
  __REG32  AA7            : 1;
  __REG32  AA8            : 1;
  __REG32  AA9            : 1;
  __REG32  AA10           : 1;
  __REG32  AA11           : 1;
  __REG32  AA12           : 1;
  __REG32  AA13           : 1;
  __REG32  AA14           : 1;
  __REG32  AA15           : 1;
  __REG32  AA16           : 1;
  __REG32  AA17           : 1;
  __REG32  AA18           : 1;
  __REG32  AA19           : 1;
  __REG32  AA20           : 1;
  __REG32  AA21           : 1;
  __REG32  AA22           : 1;
  __REG32  AA23           : 1;
  __REG32  AA24           : 1;
  __REG32  AA25           : 1;
  __REG32  AA26           : 1;
  __REG32  AA27           : 1;
  __REG32  AA28           : 1;
  __REG32  AA29           : 1;
  __REG32  AA30           : 1;
  __REG32                 : 1;
} __canaa_bits;

/* CAN Change Data Request (CDR) */
typedef struct {
  __REG32  CDR0            : 1;
  __REG32  CDR1            : 1;
  __REG32  CDR2            : 1;
  __REG32  CDR3            : 1;
  __REG32  CDR4            : 1;
  __REG32  CDR5            : 1;
  __REG32  CDR6            : 1;
  __REG32  CDR7            : 1;
  __REG32  CDR8            : 1;
  __REG32  CDR9            : 1;
  __REG32  CDR10           : 1;
  __REG32  CDR11           : 1;
  __REG32  CDR12           : 1;
  __REG32  CDR13           : 1;
  __REG32  CDR14           : 1;
  __REG32  CDR15           : 1;
  __REG32  CDR16           : 1;
  __REG32  CDR17           : 1;
  __REG32  CDR18           : 1;
  __REG32  CDR19           : 1;
  __REG32  CDR20           : 1;
  __REG32  CDR21           : 1;
  __REG32  CDR22           : 1;
  __REG32  CDR23           : 1;
  __REG32  CDR24           : 1;
  __REG32  CDR25           : 1;
  __REG32  CDR26           : 1;
  __REG32  CDR27           : 1;
  __REG32  CDR28           : 1;
  __REG32  CDR29           : 1;
  __REG32  CDR30           : 1;
  __REG32                 : 1;
} __cancdr_bits;

/* CAN Receive Message Pending Register (RMP) */
typedef struct {
  __REG32  RMP0            : 1;
  __REG32  RMP1            : 1;
  __REG32  RMP2            : 1;
  __REG32  RMP3            : 1;
  __REG32  RMP4            : 1;
  __REG32  RMP5            : 1;
  __REG32  RMP6            : 1;
  __REG32  RMP7            : 1;
  __REG32  RMP8            : 1;
  __REG32  RMP9            : 1;
  __REG32  RMP10           : 1;
  __REG32  RMP11           : 1;
  __REG32  RMP12           : 1;
  __REG32  RMP13           : 1;
  __REG32  RMP14           : 1;
  __REG32  RMP15           : 1;
  __REG32  RMP16           : 1;
  __REG32  RMP17           : 1;
  __REG32  RMP18           : 1;
  __REG32  RMP19           : 1;
  __REG32  RMP20           : 1;
  __REG32  RMP21           : 1;
  __REG32  RMP22           : 1;
  __REG32  RMP23           : 1;
  __REG32  RMP24           : 1;
  __REG32  RMP25           : 1;
  __REG32  RMP26           : 1;
  __REG32  RMP27           : 1;
  __REG32  RMP28           : 1;
  __REG32  RMP29           : 1;
  __REG32  RMP30           : 1;
  __REG32  RMP31           : 1;
} __canrmp_bits;

/* CAN Receive Message Lost Register (RML) */
typedef struct {
  __REG32  RML0            : 1;
  __REG32  RML1            : 1;
  __REG32  RML2            : 1;
  __REG32  RML3            : 1;
  __REG32  RML4            : 1;
  __REG32  RML5            : 1;
  __REG32  RML6            : 1;
  __REG32  RML7            : 1;
  __REG32  RML8            : 1;
  __REG32  RML9            : 1;
  __REG32  RML10           : 1;
  __REG32  RML11           : 1;
  __REG32  RML12           : 1;
  __REG32  RML13           : 1;
  __REG32  RML14           : 1;
  __REG32  RML15           : 1;
  __REG32  RML16           : 1;
  __REG32  RML17           : 1;
  __REG32  RML18           : 1;
  __REG32  RML19           : 1;
  __REG32  RML20           : 1;
  __REG32  RML21           : 1;
  __REG32  RML22           : 1;
  __REG32  RML23           : 1;
  __REG32  RML24           : 1;
  __REG32  RML25           : 1;
  __REG32  RML26           : 1;
  __REG32  RML27           : 1;
  __REG32  RML28           : 1;
  __REG32  RML29           : 1;
  __REG32  RML30           : 1;
  __REG32  RML31           : 1;
} __canrml_bits;

/* CAN Remote Frame Pending Register (RFP) */
typedef struct {
  __REG32  RFP0            : 1;
  __REG32  RFP1            : 1;
  __REG32  RFP2            : 1;
  __REG32  RFP3            : 1;
  __REG32  RFP4            : 1;
  __REG32  RFP5            : 1;
  __REG32  RFP6            : 1;
  __REG32  RFP7            : 1;
  __REG32  RFP8            : 1;
  __REG32  RFP9            : 1;
  __REG32  RFP10           : 1;
  __REG32  RFP11           : 1;
  __REG32  RFP12           : 1;
  __REG32  RFP13           : 1;
  __REG32  RFP14           : 1;
  __REG32  RFP15           : 1;
  __REG32  RFP16           : 1;
  __REG32  RFP17           : 1;
  __REG32  RFP18           : 1;
  __REG32  RFP19           : 1;
  __REG32  RFP20           : 1;
  __REG32  RFP21           : 1;
  __REG32  RFP22           : 1;
  __REG32  RFP23           : 1;
  __REG32  RFP24           : 1;
  __REG32  RFP25           : 1;
  __REG32  RFP26           : 1;
  __REG32  RFP27           : 1;
  __REG32  RFP28           : 1;
  __REG32  RFP29           : 1;
  __REG32  RFP30           : 1;
  __REG32  RFP31           : 1;
} __canrfp_bits;

/* CAN Local Acceptance Mask (LAM) */
typedef struct {
  __REG32  LAM0            : 1;
  __REG32  LAM1            : 1;
  __REG32  LAM2            : 1;
  __REG32  LAM3            : 1;
  __REG32  LAM4            : 1;
  __REG32  LAM5            : 1;
  __REG32  LAM6            : 1;
  __REG32  LAM7            : 1;
  __REG32  LAM8            : 1;
  __REG32  LAM9            : 1;
  __REG32  LAM10           : 1;
  __REG32  LAM11           : 1;
  __REG32  LAM12           : 1;
  __REG32  LAM13           : 1;
  __REG32  LAM14           : 1;
  __REG32  LAM15           : 1;
  __REG32  LAM16           : 1;
  __REG32  LAM17           : 1;
  __REG32  LAM18           : 1;
  __REG32  LAM19           : 1;
  __REG32  LAM20           : 1;
  __REG32  LAM21           : 1;
  __REG32  LAM22           : 1;
  __REG32  LAM23           : 1;
  __REG32  LAM24           : 1;
  __REG32  LAM25           : 1;
  __REG32  LAM26           : 1;
  __REG32  LAM27           : 1;
  __REG32  LAM28           : 1;
  __REG32                  : 2;
  __REG32  LAMI            : 1;
} __canlam_bits;

/* CAN Global Acceptance Mask (GAM) */
typedef struct {
  __REG32  GAM0            : 1;
  __REG32  GAM1            : 1;
  __REG32  GAM2            : 1;
  __REG32  GAM3            : 1;
  __REG32  GAM4            : 1;
  __REG32  GAM5            : 1;
  __REG32  GAM6            : 1;
  __REG32  GAM7            : 1;
  __REG32  GAM8            : 1;
  __REG32  GAM9            : 1;
  __REG32  GAM10           : 1;
  __REG32  GAM11           : 1;
  __REG32  GAM12           : 1;
  __REG32  GAM13           : 1;
  __REG32  GAM14           : 1;
  __REG32  GAM15           : 1;
  __REG32  GAM16           : 1;
  __REG32  GAM17           : 1;
  __REG32  GAM18           : 1;
  __REG32  GAM19           : 1;
  __REG32  GAM20           : 1;
  __REG32  GAM21           : 1;
  __REG32  GAM22           : 1;
  __REG32  GAM23           : 1;
  __REG32  GAM24           : 1;
  __REG32  GAM25           : 1;
  __REG32  GAM26           : 1;
  __REG32  GAM27           : 1;
  __REG32  GAM28           : 1;
  __REG32                  : 2;
  __REG32  GAMI            : 1;
} __cangam_bits;

/* CAN Master Control Register (MCR) */
typedef struct {
  __REG32  SRES            : 1;
  __REG32  TSCC            : 1;
  __REG32                  : 1;
  __REG32  MTOS            : 1;
  __REG32  WUBA            : 1;
  __REG32                  : 1;
  __REG32  SMR             : 1;
  __REG32  CCR             : 1;
  __REG32  TSTERR          : 1;
  __REG32  TSTLB           : 1;
  __REG32                  : 1;
  __REG32  SUR             : 1;
  __REG32                  :20;
} __canmcr_bits;

/* CAN Bit Configuration Register 1 (BCR1) */
typedef struct {
  __REG32  BRP             :10;
  __REG32                  :22;
} __canbcr1_bits;

/* CAN Bit Configuration Register 2 (BCR2) */
typedef struct {
  __REG32  TSEG1           : 4;
  __REG32  TSEG2           : 3;
  __REG32  SAM             : 1;
  __REG32  SJW             : 2;
  __REG32                  :22;
} __canbcr2_bits;

/* CAN Time Stamp Counter Register */
typedef struct {
  __REG32  TSC             :16;
  __REG32                  :16;
} __cantsc_bits;

/* CAN Time Stamp Counter Prescaler Register */
typedef struct {
  __REG32  TSP             : 4;
  __REG32                  :28;
} __cantsp_bits;

/* CAN Global Status Register (GSR) */
typedef struct {
  __REG32  EW              : 1;
  __REG32  EP              : 1;
  __REG32  BO              : 1;
  __REG32  TSO             : 1;
  __REG32                  : 2;
  __REG32  SMA             : 1;
  __REG32  CCE             : 1;
  __REG32  SUA             : 1;
  __REG32                  : 1;
  __REG32  TM              : 1;
  __REG32  RM              : 1;
  __REG32  MIS             : 5;
  __REG32                  :15;
} __cangsr_bits;

/* CAN Error Counter Register (CEC) */
typedef struct {
  __REG32  REC             : 8;
  __REG32  TEC             : 8;
  __REG32                  :16;
} __cancec_bits;

/* CAN Global Interrupt Flag Register (GIF) */
typedef struct {
  __REG32  WLIF            : 1;
  __REG32  EPIF            : 1;
  __REG32  BOIF            : 1;
  __REG32  TSOIF           : 1;
  __REG32  TRMABF          : 1;
  __REG32  RMLIF           : 1;
  __REG32  WUIF            : 1;
  __REG32  RFPF            : 1;
  __REG32                  :24;
} __cangif_bits;

/* CAN Global Interrupt Mask Register (GIM) */
typedef struct {
  __REG32  WLIM            : 1;
  __REG32  EPIM            : 1;
  __REG32  BOIM            : 1;
  __REG32  TSOIM           : 1;
  __REG32  TRMABM          : 1;
  __REG32  RMLIM           : 1;
  __REG32  WUIM            : 1;
  __REG32  RFPM            : 1;
  __REG32                  :24;
} __cangim_bits;

/* CAN Mailbox Interrupt Mask Register (MBIM) */
typedef struct {
  __REG32  MBIM0            : 1;
  __REG32  MBIM1            : 1;
  __REG32  MBIM2            : 1;
  __REG32  MBIM3            : 1;
  __REG32  MBIM4            : 1;
  __REG32  MBIM5            : 1;
  __REG32  MBIM6            : 1;
  __REG32  MBIM7            : 1;
  __REG32  MBIM8            : 1;
  __REG32  MBIM9            : 1;
  __REG32  MBIM10           : 1;
  __REG32  MBIM11           : 1;
  __REG32  MBIM12           : 1;
  __REG32  MBIM13           : 1;
  __REG32  MBIM14           : 1;
  __REG32  MBIM15           : 1;
  __REG32  MBIM16           : 1;
  __REG32  MBIM17           : 1;
  __REG32  MBIM18           : 1;
  __REG32  MBIM19           : 1;
  __REG32  MBIM20           : 1;
  __REG32  MBIM21           : 1;
  __REG32  MBIM22           : 1;
  __REG32  MBIM23           : 1;
  __REG32  MBIM24           : 1;
  __REG32  MBIM25           : 1;
  __REG32  MBIM26           : 1;
  __REG32  MBIM27           : 1;
  __REG32  MBIM28           : 1;
  __REG32  MBIM29           : 1;
  __REG32  MBIM30           : 1;
  __REG32  MBIM31           : 1;
} __canmbim_bits;

/* CAN Mailbox Transmit Interrupt Flag Registers (MBTIF) */
typedef struct {
  __REG32  MBTIF0            : 1;
  __REG32  MBTIF1            : 1;
  __REG32  MBTIF2            : 1;
  __REG32  MBTIF3            : 1;
  __REG32  MBTIF4            : 1;
  __REG32  MBTIF5            : 1;
  __REG32  MBTIF6            : 1;
  __REG32  MBTIF7            : 1;
  __REG32  MBTIF8            : 1;
  __REG32  MBTIF9            : 1;
  __REG32  MBTIF10           : 1;
  __REG32  MBTIF11           : 1;
  __REG32  MBTIF12           : 1;
  __REG32  MBTIF13           : 1;
  __REG32  MBTIF14           : 1;
  __REG32  MBTIF15           : 1;
  __REG32  MBTIF16           : 1;
  __REG32  MBTIF17           : 1;
  __REG32  MBTIF18           : 1;
  __REG32  MBTIF19           : 1;
  __REG32  MBTIF20           : 1;
  __REG32  MBTIF21           : 1;
  __REG32  MBTIF22           : 1;
  __REG32  MBTIF23           : 1;
  __REG32  MBTIF24           : 1;
  __REG32  MBTIF25           : 1;
  __REG32  MBTIF26           : 1;
  __REG32  MBTIF27           : 1;
  __REG32  MBTIF28           : 1;
  __REG32  MBTIF29           : 1;
  __REG32  MBTIF30           : 1;
  __REG32                    : 1;
} __canmbtif_bits;

/* CAN Mailbox Receive Interrupt Flag Registers (MBRIF) */
typedef struct {
  __REG32  MBRIF0            : 1;
  __REG32  MBRIF1            : 1;
  __REG32  MBRIF2            : 1;
  __REG32  MBRIF3            : 1;
  __REG32  MBRIF4            : 1;
  __REG32  MBRIF5            : 1;
  __REG32  MBRIF6            : 1;
  __REG32  MBRIF7            : 1;
  __REG32  MBRIF8            : 1;
  __REG32  MBRIF9            : 1;
  __REG32  MBRIF10           : 1;
  __REG32  MBRIF11           : 1;
  __REG32  MBRIF12           : 1;
  __REG32  MBRIF13           : 1;
  __REG32  MBRIF14           : 1;
  __REG32  MBRIF15           : 1;
  __REG32  MBRIF16           : 1;
  __REG32  MBRIF17           : 1;
  __REG32  MBRIF18           : 1;
  __REG32  MBRIF19           : 1;
  __REG32  MBRIF20           : 1;
  __REG32  MBRIF21           : 1;
  __REG32  MBRIF22           : 1;
  __REG32  MBRIF23           : 1;
  __REG32  MBRIF24           : 1;
  __REG32  MBRIF25           : 1;
  __REG32  MBRIF26           : 1;
  __REG32  MBRIF27           : 1;
  __REG32  MBRIF28           : 1;
  __REG32  MBRIF29           : 1;
  __REG32  MBRIF30           : 1;
  __REG32  MBRIF31           : 1;
} __canmbrif_bits;

/*EMIFVRR Register*/
typedef struct {
  __REG32 REV            :16;
  __REG32 VER            :16;
} __emifvrr_bits;

/*EMIFCR Register*/
typedef struct {
  __REG32                : 8;
  __REG32 IOTOUT         : 1;
  __REG32 DMATOUT        : 1;
  __REG32 DMAERR         : 1;
  __REG32                :21;
} __emifcr_bits;

/*EMIFINTST Register*/
typedef struct {
  __REG32 EMDMACINT      : 1;
  __REG32                : 7;
  __REG32 SLAVE0         : 1;
  __REG32 SLAVE1         : 1;
  __REG32 SLAVE2         : 1;
  __REG32 SLAVE3         : 1;
  __REG32                : 3;
  __REG32 IOTOUT         : 1;
  __REG32 MASTER0        : 1;
  __REG32 MASTER1        : 1;
  __REG32 MASTER2        : 1;
  __REG32                : 4;
  __REG32 DMAOUT         : 1;
  __REG32                : 8;
} __emifintst_bits;

/*EMIFINTEN Register*/
typedef struct {
  __REG32 EMDMACINT      : 1;
  __REG32                : 7;
  __REG32 SLAVE0         : 1;
  __REG32 SLAVE1         : 1;
  __REG32 SLAVE2         : 1;
  __REG32 SLAVE3         : 1;
  __REG32                : 3;
  __REG32 IOTOUT         : 1;
  __REG32 MASTER0        : 1;
  __REG32 MASTER1        : 1;
  __REG32 MASTER2        : 1;
  __REG32                : 4;
  __REG32 DMATOUT        : 1;
  __REG32                : 8;
} __emifinten_bits;

/*EMDCR Register*/
typedef struct {
  __REG32 SWR           : 1;
  __REG32               : 3;
  __REG32 DL            : 2;
  __REG32 DE            : 1;
  __REG32               :25;
} __emdcr_bits;

/*EMDTEN Register*/
typedef struct {
  __REG32 TRNS          : 1;
  __REG32               :31;
} __emdten_bits;

/*EMDREN Register*/
typedef struct {
  __REG32 RCV           : 1;
  __REG32               :31;
} __emdren_bits;

/*EMDINTST,EMDINTEN Register*/
typedef struct {
  __REG32 RINT1         : 1;
  __REG32 RINT2         : 1;
  __REG32 RINT3         : 1;
  __REG32 RINT4         : 1;
  __REG32 RINT5         : 1;
  __REG32               : 2;
  __REG32 RINT8         : 1;
  __REG32 TINT1         : 1;
  __REG32 TINT2         : 1;
  __REG32 TINT3         : 1;
  __REG32 TINT4         : 1;
  __REG32               : 4;
  __REG32 RFE           : 1;
  __REG32 RDE           : 1;
  __REG32 FRC           : 1;
  __REG32 TFE           : 1;
  __REG32 TDE           : 1;
  __REG32 FTC           : 1;
  __REG32 MINT          : 1;
  __REG32 BER           : 1;
  __REG32 RFRMER        : 1;
  __REG32 RABT          : 1;
  __REG32 TABT          : 1;
  __REG32               : 3;
  __REG32 TWB           : 1;
  __REG32               : 1;
} __emdint_bits;

/*EMDERR Register*/
typedef struct {
  __REG32 RINT1         : 1;
  __REG32 RINT2         : 1;
  __REG32 RINT3         : 1;
  __REG32 RINT4         : 1;
  __REG32 RINT5         : 1;
  __REG32               : 2;
  __REG32 RINT8         : 1;
  __REG32 TINT1         : 1;
  __REG32 TINT2         : 1;
  __REG32 TINT3         : 1;
  __REG32 TINT4         : 1;
  __REG32               :20;
} __emderr_bits;

/*EMDMFCR Register*/
typedef struct {
  __REG32 MIS           :16;
  __REG32               :16;
} __emdmfcr_bits;

/*EMDTFTR Register*/
typedef struct {
  __REG32 TFTH          :11;
  __REG32               :21;
} __emdtftr_bits;

/*EMDFCDR Register*/
typedef struct {
  __REG32 RA            : 5;
  __REG32               : 3;
  __REG32 TA            : 5;
  __REG32               :19;
} __emdfcdr_bits;

/*EMDRRCR Register*/
typedef struct {
  __REG32 RR            : 1;
  __REG32 RNR           : 1;
  __REG32               :30;
} __emdrrcr_bits;

/*EMDTUFR Register*/
typedef struct {
  __REG32 UNDER         :16;
  __REG32               :16;
} __emdtufr_bits;

/*EMDROFR Register*/
typedef struct {
  __REG32 OVER          :16;
  __REG32               :16;
} __emdrofr_bits;

/*EMDRBTR Register*/
typedef struct {
  __REG32 RFDO          : 3;
  __REG32               :13;
  __REG32 RFFO          : 3;
  __REG32               :13;
} __emdrbtr_bits;

/*EMDRPDR Register*/
typedef struct {
  __REG32 PADR          : 6;
  __REG32               :10;
  __REG32 PADS          : 2;
  __REG32               :14;
} __emdrpdr_bits;

/*EMDTIMER Register*/
typedef struct {
  __REG32 TIS           : 1;
  __REG32               : 3;
  __REG32 TIM           : 1;
  __REG32               :27;
} __emdtimer_bits;

/*EMCCR Register*/
typedef struct {
  __REG32 RPM           : 1;
  __REG32 DPM           : 1;
  __REG32 OLB           : 1;
  __REG32 ILB           : 1;
  __REG32               : 1;
  __REG32 TPE           : 1;
  __REG32 RPE           : 1;
  __REG32               : 2;
  __REG32 MPM           : 1;
  __REG32               : 2;
  __REG32 CER           : 1;
  __REG32               : 3;
  __REG32 TXF           : 1;
  __REG32 RXF           : 1;
  __REG32 PFR           : 1;
  __REG32 ZPF           : 1;
  __REG32 TPC           : 1;
  __REG32               :11;
} __emccr_bits;

/*EMCRFLR Register*/
typedef struct {
  __REG32 FLEN_ULMT     :12;
  __REG32               :20;
} __emcrflr_bits;

/*EMCINTST,EMCINTEN Register*/
typedef struct {
  __REG32 FCIDTCT       : 1;
  __REG32 MAGICPKT      : 1;
  __REG32 LINKCHNG      : 1;
  __REG32               : 1;
  __REG32 PSRTROV       : 1;
  __REG32 BRDSRCV       : 1;
  __REG32               :26;
} __emcint_bits;

/*EMCMIICR Register*/
typedef struct {
  __REG32 MDC           : 1;
  __REG32 MMD           : 1;
  __REG32 MDO           : 1;
  __REG32 MDI           : 1;
  __REG32               :28;
} __emcmiicr_bits;

/*EMCPHYST Register*/
typedef struct {
  __REG32 LINK          : 1;
  __REG32               :31;
} __emcphyst_bits;

/*EMCICR Register*/
typedef struct {
  __REG32 IPG           : 5;
  __REG32               :27;
} __emcicr_bits;

/*EMCAPR Register*/
typedef struct {
  __REG32 APAUSE        :16;
  __REG32               :16;
} __emcapr_bits;

/*EMCMPR Register*/
typedef struct {
  __REG32 MPAUSE        :16;
  __REG32               :16;
} __emcmpr_bits;

/*EMCRPCR Register*/
typedef struct {
  __REG32 RPAUSE        : 8;
  __REG32               :24;
} __emcrpcr_bits;

/*EMCPULR Register*/
typedef struct {
  __REG32 TXPAUSE       :16;
  __REG32               :16;
} __emcpulr_bits;

/*EMCPCR Register*/
typedef struct {
  __REG32 TPAUSE        :16;
  __REG32               :16;
} __emcpcr_bits;

/*EMCBRR Register*/
typedef struct {
  __REG32 BCF           :16;
  __REG32               :16;
} __emcbrr_bits;

/*EMCMALR (MAC Low Address Register)*/
typedef struct {
  __REG32 MACADRL       :16;
  __REG32               :16;
} __emcmalr_bits;

/*Remote Control Enable Register*/
typedef struct {
  __REG32 RMCEN     : 1;
  __REG32           :31;
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

/*Remote Control Receive End Bit Number Register 1-3*/
typedef struct {
  __REG32 RMCEND    : 7;
  __REG32           :25;
} __rmcend_bits;

/*Remote Control Source Clock selection Register*/
typedef struct {
  __REG32 RMCCLK    : 1;
  __REG32           :31;
} __rmcfssel_bits;

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
  __REG32 DACON   : 1;
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

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  ADR          :12;
  __REG32  ADRF         : 1;
  __REG32  ADOVRF       : 1;
  __REG32               : 2;
  __REG32  _ADOVRF      : 1;
  __REG32  _ADRF        : 1;
  __REG32               : 2;
  __REG32  _ADR         :12;
} __adregx_bits;

/*A/D Conversion Result Registers */
typedef struct {
  __REG32  ADSPR        :12;
  __REG32  ADSPRF       : 1;
  __REG32  ADOVRSPF     : 1;
  __REG32               : 2;
  __REG32  _ADOVRSPF    : 1;
  __REG32  _ADSPRF      : 1;
  __REG32               : 2;
  __REG32  _ADSPR       :12;
} __adregsp_bits;

/*A/D Conversion Comparison Control Register 0*/
typedef struct {
  __REG32  AINS0        : 4;
  __REG32  ADBIG0       : 1;
  __REG32  CMPCOND0     : 1;
  __REG32               : 1;
  __REG32  CMP0EN       : 1;
  __REG32  CMPCNT0      : 4;
  __REG32               : 20;
} __adcmpcr0_bits;

/*A/D Conversion Comparison Control Register 1*/
typedef struct {
  __REG32  AINS1        : 4;
  __REG32  ADBIG1       : 1;
  __REG32  CMPCOND1     : 1;
  __REG32               : 1;
  __REG32  CMP1EN       : 1;
  __REG32  CMPCNT1      : 4;
  __REG32               : 20;
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

/*ADILVMO1 */
typedef struct {
  __REG32           : 7;
  __REG32  SWATRG   : 1;
  __REG32           :24;
} __adilvmo1_bits;

/*ADILVMO2 */
typedef struct {
  __REG32  TRGAEN   : 1;
  __REG32  TRGASEL  : 3;
  __REG32  TRGASTA  : 1;
  __REG32           : 2;
  __REG32  ADILV    : 1;
  __REG32           :24;
} __adilvmo2_bits;

/*ADILVMO3 */
typedef struct {
  __REG32  CORCNT   : 8;
  __REG32           :24;
} __adilvmo3_bits;

/*D/A Conversion Control register */
typedef struct {
  __REG32  OP       : 1;
  __REG32  VREFON   : 1;
  __REG32           :30;
} __daccntx_bits;

/*D/A Conversion data register 1*/
typedef struct {
  __REG32           : 6;
  __REG32  DAC      :10;
  __REG32           :16;
} __dacregx_bits;

/*D/A Conversion Output Control register */
typedef struct {
  __REG32  WAVE     : 2;
  __REG32           : 5;
  __REG32  DMAEN    : 1;
  __REG32  TRGEN    : 1;
  __REG32  TRGSEL   : 3;
  __REG32           : 4;
  __REG32  AMPSEL   : 2;
  __REG32  OFFSET   : 3;
  __REG32           :11;
} __dacdctlx_bits;

/*D/A waveform trigger control register */
typedef struct {
  __REG32  SWTRG    : 1;
  __REG32           :14;
  __REG32  DACCLR   : 1;
  __REG32           :16;
} __dactctlx_bits;

/*D/A VOUTHOLD adjustment register */
typedef struct {
  __REG32  VHOLDCTF     : 4;
  __REG32  VHOLDCTB     : 4;
  __REG32               :24;
} __dacvctlx_bits;

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
  __REG32  MTTBINTTB0   : 1;
  __REG32  MTTBINTTB1   : 1;
  __REG32  MTTBINTTBOF  : 1;
  __REG32               :29;
} __mtxst_bits;

/*MPT interrupt mask register*/
typedef struct {
  __REG32  MTTBIM0   : 1;
  __REG32  MTTBIM1   : 1;
  __REG32  MTTBIMOF  : 1;
  __REG32            :29;
} __mtxim_bits;

/*MPT read capture register*/
typedef struct {
  __REG32  MTUC       :16;
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

/*PMD Enable Register (MDEN)*/
typedef struct {
  __REG32 PWMEN        : 1;
  __REG32              :31;
} __mden_bits;

/*Port Output Mode Register (PORTMD)*/
typedef struct {
  __REG32 PORTMD       : 1;
  __REG32              :31;
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
  __REG32  UD      	: 1;
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
  __REG8  DIS8HZ  : 1;
  __REG8  DIS4HZ  : 1;
  __REG8  DIS2HZ  : 1;
  __REG8          : 2;
  __REG8  RSTTMR  : 1;
  __REG8  DIS16HZ : 1;
  __REG8  DIS1HZ  : 1;
} __restr_bits;

/*LVD-REST control Register*/
typedef struct {
  __REG32 LVDEN1       : 1;
  __REG32 LVDLVL1      : 3;
  __REG32              : 1;
  __REG32 LVDRSTEN     : 1;
  __REG32              :26;
} __lvdrcr_bits;

/*LVD-INT control Register*/
typedef struct {
  __REG32 LVDEN2       : 1;
  __REG32 LVDLVL2      : 3;
  __REG32 INTSEL       : 1;
  __REG32 LVDINTEN     : 1;
  __REG32              :26;
} __lvdicr_bits;

/*LVD status Register*/
typedef struct {
  __REG32 LVDST1       : 1;
  __REG32 LVDST2       : 1;
  __REG32              :30;
} __lvdsr_bits;

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
  __REG32 OFDMN     : 9;
  __REG32           :23;
} __ofdmn_bits;

/*Higher detection frequency setting register*/
typedef struct {
  __REG32 OFDMX     : 9;
  __REG32           :23;
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

/*monitor Register*/
typedef struct {
  __REG32 OFDMON    : 1;
  __REG32           :31;
} __ofdmon_bits;

/*Watchdog Timer Mode Register*/
typedef struct {
  __REG32         : 1;
  __REG32 RESCR   : 1;
  __REG32 I2WDT   : 1;
  __REG32         : 1;
  __REG32 WDTP    : 3;
  __REG32 WDTE    : 1;
  __REG32         :24;
} __wdmod_bits;

/*WDCR (Watchdog Timer Control Register)*/
typedef struct {
  __REG32 WDCR    : 8;
  __REG32         :24;
} __wdcr_bits;

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

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSTICKCSR,          0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,          0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,          0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,        0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,             0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(SETENA1,             0xE000E104,__READ_WRITE ,__setena1_bits);
__IO_REG32_BIT(SETENA2,             0xE000E108,__READ_WRITE ,__setena2_bits);
__IO_REG32_BIT(SETENA3,             0xE000E10C,__READ_WRITE ,__setena3_bits);
__IO_REG32_BIT(CLRENA0,             0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,             0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(CLRENA2,             0xE000E188,__READ_WRITE ,__clrena2_bits);
__IO_REG32_BIT(CLRENA3,             0xE000E18C,__READ_WRITE ,__clrena3_bits);
__IO_REG32_BIT(SETPEND0,            0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,            0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(SETPEND2,            0xE000E208,__READ_WRITE ,__setpend2_bits);
__IO_REG32_BIT(SETPEND3,            0xE000E20C,__READ_WRITE ,__setpend3_bits);
__IO_REG32_BIT(CLRPEND0,            0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,            0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(CLRPEND2,            0xE000E288,__READ_WRITE ,__clrpend2_bits);
__IO_REG32_BIT(CLRPEND3,            0xE000E28C,__READ_WRITE ,__clrpend3_bits);
__IO_REG32_BIT(IP0,                 0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                 0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                 0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                 0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                 0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                 0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                 0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                 0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(IP8,                 0xE000E420,__READ_WRITE ,__pri8_bits);
__IO_REG32_BIT(IP9,                 0xE000E424,__READ_WRITE ,__pri9_bits);
__IO_REG32_BIT(IP10,                0xE000E428,__READ_WRITE ,__pri10_bits);
__IO_REG32_BIT(IP11,                0xE000E42C,__READ_WRITE ,__pri11_bits);
__IO_REG32_BIT(IP12,                0xE000E430,__READ_WRITE ,__pri12_bits);
__IO_REG32_BIT(IP13,                0xE000E434,__READ_WRITE ,__pri13_bits);
__IO_REG32_BIT(IP14,                0xE000E438,__READ_WRITE ,__pri14_bits);
__IO_REG32_BIT(IP15,                0xE000E43C,__READ_WRITE ,__pri15_bits);
__IO_REG32_BIT(IP16,                0xE000E440,__READ_WRITE ,__pri16_bits);
__IO_REG32_BIT(IP17,                0xE000E444,__READ_WRITE ,__pri17_bits);
__IO_REG32_BIT(IP18,                0xE000E448,__READ_WRITE ,__pri18_bits);
__IO_REG32_BIT(IP19,                0xE000E44C,__READ_WRITE ,__pri19_bits);
__IO_REG32_BIT(IP20,                0xE000E450,__READ_WRITE ,__pri20_bits);
__IO_REG32_BIT(IP21,                0xE000E454,__READ_WRITE ,__pri21_bits);
__IO_REG32_BIT(IP22,                0xE000E458,__READ_WRITE ,__pri22_bits);
__IO_REG32_BIT(IP23,                0xE000E45C,__READ_WRITE ,__pri23_bits);
__IO_REG32_BIT(IP24,                0xE000E460,__READ_WRITE ,__pri24_bits);
__IO_REG32_BIT(IP25,                0xE000E464,__READ_WRITE ,__pri25_bits);
__IO_REG32_BIT(IP26,                0xE000E468,__READ_WRITE ,__pri26_bits);
__IO_REG32_BIT(IP27,                0xE000E46C,__READ_WRITE ,__pri27_bits);
__IO_REG32_BIT(IP28,                0xE000E470,__READ_WRITE ,__pri28_bits);
__IO_REG32_BIT(IP29,                0xE000E474,__READ_WRITE ,__pri29_bits);
__IO_REG32_BIT(IP30,                0xE000E478,__READ_WRITE ,__pri30_bits);
__IO_REG32_BIT(IP31,                0xE000E47C,__READ_WRITE ,__pri31_bits);
__IO_REG32_BIT(VTOR,                0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,               0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SHPR0,               0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,               0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,               0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,               0xE000ED24,__READ_WRITE ,__shcsr_bits);

/***************************************************************************
 **
 ** TRMOSC
 **
 ***************************************************************************/
__IO_REG32_BIT(TRMOSCPRO,           0x400F3200,__READ_WRITE ,__trmoscpro_bits);
__IO_REG32_BIT(TRMOSCEN,            0x400F3204,__READ_WRITE ,__trmoscen_bits);
__IO_REG32_BIT(TRMOSCINIT,          0x400F3208,__READ       ,__trmoscinit_bits);
__IO_REG32_BIT(TRMOSCSET,           0x400F320C,__READ_WRITE ,__trmoscset_bits);

/***************************************************************************
 **
 ** CG (Clock generator)
 **
 ***************************************************************************/
__IO_REG32_BIT(CGSYSCR,             0x400F3000,__READ_WRITE ,__cgsyscr_bits);
__IO_REG32_BIT(CGOSCCR,             0x400F3004,__READ_WRITE ,__cgosccr_bits);
__IO_REG32_BIT(CGSTBYCR,            0x400F3008,__READ_WRITE ,__cgstbycr_bits);
__IO_REG32_BIT(CGPLLSEL,            0x400F300C,__READ_WRITE ,__cgpllsel_bits);
__IO_REG32_BIT(CGCKSEL,             0x400F3010,__READ_WRITE ,__cgcksel_bits);
__IO_REG32_BIT(CGCKSTP,             0x400F3018,__READ_WRITE ,__cgckstp_bits);
__IO_REG32_BIT(CGPROTECT,           0x400F303C,__READ_WRITE ,__cgprotect_bits);
__IO_REG32_BIT(CGIMCGA,             0x400F3040,__READ_WRITE ,__cgimcga_bits);
__IO_REG32_BIT(CGIMCGB,             0x400F3044,__READ_WRITE ,__cgimcgb_bits);
__IO_REG32_BIT(CGIMCGC,             0x400F3048,__READ_WRITE ,__cgimcgc_bits);
__IO_REG32_BIT(CGIMCGD,             0x400F304C,__READ_WRITE ,__cgimcgd_bits);
__IO_REG32_BIT(CGICRCG,             0x400F3060,__WRITE      ,__cgicrcg_bits);
__IO_REG32_BIT(CGRSTFLG,            0x400F3064,__READ_WRITE ,__cgrstflg_bits);
__IO_REG32_BIT(CGNMIFLG,            0x400F3068,__READ       ,__cgnmiflg_bits);

/***************************************************************************
 **
 ** DMAC A
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACAStatus,            0x4004C000,__READ        ,__dmacstatus_bits);
__IO_REG32_BIT(DMACACfg,               0x4004C004,__WRITE       ,__dmaccfg_bits);
__IO_REG32_BIT(DMACACtrlBasePtr,       0x4004C008,__READ_WRITE  ,__dmacctrlbaseptr_bits);
__IO_REG32(    DMACAAltCtrlBasePtr,    0x4004C00C,__READ );
__IO_REG32_BIT(DMACAChnlSwRequest,     0x4004C014,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlUseburstSet,   0x4004C018,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlUseburstClr,   0x4004C01C,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlReqMaskSet,    0x4004C020,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlReqMaskClr,    0x4004C024,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlEnableSet,     0x4004C028,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlEnableClr,     0x4004C02C,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlPriAltSet,     0x4004C030,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlPriAltClr,     0x4004C034,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlPrioritySet,   0x4004C038,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAChnlPriorityClr,   0x4004C03C,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACAErrClr,            0x4004C04C,__READ_WRITE  ,__dmacerrclr_bits);

/***************************************************************************
 **
 ** DMAC B
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACBStatus,            0x4004D000,__READ        ,__dmacstatus_bits);
__IO_REG32_BIT(DMACBCfg,               0x4004D004,__WRITE       ,__dmaccfg_bits);
__IO_REG32_BIT(DMACBCtrlBasePtr,       0x4004D008,__READ_WRITE  ,__dmacctrlbaseptr_bits);
__IO_REG32(    DMACBAltCtrlBasePtr,    0x4004D00C,__READ );
__IO_REG32_BIT(DMACBChnlSwRequest,     0x4004D014,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlUseburstSet,   0x4004D018,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlUseburstClr,   0x4004D01C,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlReqMaskSet,    0x4004D020,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlReqMaskClr,    0x4004D024,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlEnableSet,     0x4004D028,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlEnableClr,     0x4004D02C,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlPriAltSet,     0x4004D030,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlPriAltClr,     0x4004D034,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlPrioritySet,   0x4004D038,__READ_WRITE  ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBChnlPriorityClr,   0x4004D03C,__WRITE       ,__dmacchnlctl_bits);
__IO_REG32_BIT(DMACBErrClr,            0x4004D04C,__READ_WRITE  ,__dmacerrclr_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG8_BIT(PADATA,               0x400C0000,__READ_WRITE ,__pa_bits);
__IO_REG8_BIT(PACR,                 0x400C0004,__READ_WRITE ,__pacr_bits);
__IO_REG8_BIT(PAFR1,                0x400C0008,__READ_WRITE ,__pafr1_bits);
__IO_REG8_BIT(PAFR2,                0x400C000C,__READ_WRITE ,__pafr2_bits);
__IO_REG8_BIT(PAFR3,                0x400C0010,__READ_WRITE ,__pafr3_bits);
__IO_REG8_BIT(PAFR4,                0x400C0014,__READ_WRITE ,__pafr4_bits);
__IO_REG8_BIT(PAFR5,                0x400C0018,__READ_WRITE ,__pafr5_bits);
__IO_REG8_BIT(PAOD,                 0x400C0028,__READ_WRITE ,__paod_bits);
__IO_REG8_BIT(PAPUP,                0x400C002C,__READ_WRITE ,__papup_bits);
__IO_REG8_BIT(PAPDN,                0x400C0030,__READ_WRITE ,__papdn_bits);
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
__IO_REG8_BIT(PBFR4,                0x400C0114,__READ_WRITE ,__pbfr4_bits);
__IO_REG8_BIT(PBOD,                 0x400C0128,__READ_WRITE ,__pbod_bits);
__IO_REG8_BIT(PBPUP,                0x400C012C,__READ_WRITE ,__pbpup_bits);
__IO_REG8_BIT(PBPDN,                0x400C0130,__READ_WRITE ,__pbpdn_bits);
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
__IO_REG8_BIT(PCOD,                 0x400C0228,__READ_WRITE ,__pcod_bits);
__IO_REG8_BIT(PCPUP,                0x400C022C,__READ_WRITE ,__pcpup_bits);
__IO_REG8_BIT(PCPDN,                0x400C0230,__READ_WRITE ,__pcpdn_bits);
__IO_REG8_BIT(PCIE,                 0x400C0238,__READ_WRITE ,__pcie_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG8_BIT(PDDATA,               0x400C0300,__READ_WRITE ,__pd_bits);
__IO_REG8_BIT(PDCR  ,               0x400C0304,__READ_WRITE ,__pdcr_bits);
__IO_REG8_BIT(PDFR1,                0x400C0308,__READ_WRITE ,__pdfr1_bits);
__IO_REG8_BIT(PDOD,                 0x400C0328,__READ_WRITE ,__pdod_bits);
__IO_REG8_BIT(PDPUP,                0x400C032C,__READ_WRITE ,__pdpup_bits);
__IO_REG8_BIT(PDPDN,                0x400C0330,__READ_WRITE ,__pdpdn_bits);
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
__IO_REG8_BIT(PEFR3,                0x400C0410,__READ_WRITE ,__pefr3_bits);
__IO_REG8_BIT(PEFR4,                0x400C0414,__READ_WRITE ,__pefr4_bits);
__IO_REG8_BIT(PEFR5,                0x400C0418,__READ_WRITE ,__pefr5_bits);
__IO_REG8_BIT(PEOD,                 0x400C0428,__READ_WRITE ,__peod_bits);
__IO_REG8_BIT(PEPUP,                0x400C042C,__READ_WRITE ,__pepup_bits);
__IO_REG8_BIT(PEPDN,                0x400C0430,__READ_WRITE ,__pepdn_bits);
__IO_REG8_BIT(PEIE,                 0x400C0438,__READ_WRITE ,__peie_bits);

/***************************************************************************
 **
 ** PORTF
 **
 ***************************************************************************/
__IO_REG8_BIT(PFDATA,               0x400C0500,__READ_WRITE ,__pf_bits);
__IO_REG8_BIT(PFCR,                 0x400C0504,__READ_WRITE ,__pfcr_bits);
__IO_REG8_BIT(PFFR1,                0x400C0508,__READ_WRITE ,__pffr1_bits);
__IO_REG8_BIT(PFFR2,                0x400C050C,__READ_WRITE ,__pffr2_bits);
__IO_REG8_BIT(PFFR3,                0x400C0510,__READ_WRITE ,__pffr3_bits);
__IO_REG8_BIT(PFFR4,                0x400C0514,__READ_WRITE ,__pffr4_bits);
__IO_REG8_BIT(PFOD,                 0x400C0528,__READ_WRITE ,__pfod_bits);
__IO_REG8_BIT(PFPUP,                0x400C052C,__READ_WRITE ,__pfpup_bits);
__IO_REG8_BIT(PFPDN,                0x400C0530,__READ_WRITE ,__pfpdn_bits);
__IO_REG8_BIT(PFIE,                 0x400C0538,__READ_WRITE ,__pfie_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG8_BIT(PGDATA,               0x400C0600,__READ_WRITE ,__pg_bits);
__IO_REG8_BIT(PGCR,                 0x400C0604,__READ_WRITE ,__pgcr_bits);
__IO_REG8_BIT(PGFR1,                0x400C0608,__READ_WRITE ,__pgfr1_bits);
__IO_REG8_BIT(PGFR2,                0x400C060C,__READ_WRITE ,__pgfr2_bits);
__IO_REG8_BIT(PGFR3,                0x400C0610,__READ_WRITE ,__pgfr3_bits);
__IO_REG8_BIT(PGFR4,                0x400C0614,__READ_WRITE ,__pgfr4_bits);
__IO_REG8_BIT(PGOD,                 0x400C0628,__READ_WRITE ,__pgod_bits);
__IO_REG8_BIT(PGPUP,                0x400C062C,__READ_WRITE ,__pgpup_bits);
__IO_REG8_BIT(PGPDN,                0x400C0630,__READ_WRITE ,__pgpdn_bits);
__IO_REG8_BIT(PGIE,                 0x400C0638,__READ_WRITE ,__pgie_bits);

/***************************************************************************
 **
 ** PORTH
 **
 ***************************************************************************/
__IO_REG8_BIT(PHDATA,               0x400C0700,__READ_WRITE ,__ph_bits);
__IO_REG8_BIT(PHCR,                 0x400C0704,__READ_WRITE ,__phcr_bits);
__IO_REG8_BIT(PHFR1,                0x400C0708,__READ_WRITE ,__phfr1_bits);
__IO_REG8_BIT(PHFR2,                0x400C070C,__READ_WRITE ,__phfr2_bits);
__IO_REG8_BIT(PHFR3,                0x400C0710,__READ_WRITE ,__phfr3_bits);
__IO_REG8_BIT(PHFR4,                0x400C0714,__READ_WRITE ,__phfr4_bits);
__IO_REG8_BIT(PHFR5,                0x400C0718,__READ_WRITE ,__phfr5_bits);
__IO_REG8_BIT(PHOD,                 0x400C0728,__READ_WRITE ,__phod_bits);
__IO_REG8_BIT(PHPUP,                0x400C072C,__READ_WRITE ,__phpup_bits);
__IO_REG8_BIT(PHPDN,                0x400C0730,__READ_WRITE ,__phpdn_bits);
__IO_REG8_BIT(PHIE,                 0x400C0738,__READ_WRITE ,__phie_bits);

/***************************************************************************
 **
 ** PORTI
 **
 ***************************************************************************/
__IO_REG8_BIT(PIDATA,               0x400C0800,__READ_WRITE ,__pi_bits);
__IO_REG8_BIT(PICR,                 0x400C0804,__READ_WRITE ,__picr_bits);
__IO_REG8_BIT(PIFR1,                0x400C0808,__READ_WRITE ,__pifr1_bits);
__IO_REG8_BIT(PIFR2,                0x400C080C,__READ_WRITE ,__pifr2_bits);
__IO_REG8_BIT(PIOD,                 0x400C0828,__READ_WRITE ,__piod_bits);
__IO_REG8_BIT(PIPUP,                0x400C082C,__READ_WRITE ,__pipup_bits);
__IO_REG8_BIT(PIPDN,                0x400C0830,__READ_WRITE ,__pipdn_bits);
__IO_REG8_BIT(PIIE,                 0x400C0838,__READ_WRITE ,__piie_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG8_BIT(PJDATA,               0x400C0900,__READ_WRITE ,__pj_bits);
__IO_REG8_BIT(PJCR,                 0x400C0904,__READ_WRITE ,__pjcr_bits);
__IO_REG8_BIT(PJOD,                 0x400C0928,__READ_WRITE ,__pjod_bits);
__IO_REG8_BIT(PJPUP,                0x400C092C,__READ_WRITE ,__pjpup_bits);
__IO_REG8_BIT(PJPDN,                0x400C0930,__READ_WRITE ,__pjpdn_bits);
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
__IO_REG8_BIT(PKFR4,                0x400C0A14,__READ_WRITE ,__pkfr4_bits);
__IO_REG8_BIT(PKOD,                 0x400C0A28,__READ_WRITE ,__pkod_bits);
__IO_REG8_BIT(PKPUP,                0x400C0A2C,__READ_WRITE ,__pkpup_bits);
__IO_REG8_BIT(PKPDN,                0x400C0A30,__READ_WRITE ,__pkpdn_bits);
__IO_REG8_BIT(PKIE,                 0x400C0A38,__READ_WRITE ,__pkie_bits);

/***************************************************************************
 **
 ** PORTL
 **
 ***************************************************************************/
__IO_REG8_BIT(PLDATA,               0x400C0B00,__READ_WRITE ,__pl_bits);
__IO_REG8_BIT(PLCR,                 0x400C0B04,__READ_WRITE ,__plcr_bits);
__IO_REG8_BIT(PLFR1,                0x400C0B08,__READ_WRITE ,__plfr1_bits);
__IO_REG8_BIT(PLFR2,                0x400C0B0C,__READ_WRITE ,__plfr2_bits);
__IO_REG8_BIT(PLFR3,                0x400C0B10,__READ_WRITE ,__plfr3_bits);
__IO_REG8_BIT(PLFR4,                0x400C0B14,__READ_WRITE ,__plfr4_bits);
__IO_REG8_BIT(PLFR5,                0x400C0B18,__READ_WRITE ,__plfr5_bits);
__IO_REG8_BIT(PLFR6,                0x400C0B1C,__READ_WRITE ,__plfr6_bits);
__IO_REG8_BIT(PLOD,                 0x400C0B28,__READ_WRITE ,__plod_bits);
__IO_REG8_BIT(PLPUP,                0x400C0B2C,__READ_WRITE ,__plpup_bits);
__IO_REG8_BIT(PLPDN,                0x400C0B30,__READ_WRITE ,__plpdn_bits);
__IO_REG8_BIT(PLIE,                 0x400C0B38,__READ_WRITE ,__plie_bits);

/***************************************************************************
 **
 ** PORTM
 **
 ***************************************************************************/
__IO_REG8_BIT(PMDATA,               0x400C0C00,__READ_WRITE ,__pm_bits);
__IO_REG8_BIT(PMCR,                 0x400C0C04,__READ_WRITE ,__pmcr_bits);
__IO_REG8_BIT(PMFR1,                0x400C0C08,__READ_WRITE ,__pmfr1_bits);
__IO_REG8_BIT(PMFR2,                0x400C0C0C,__READ_WRITE ,__pmfr2_bits);
__IO_REG8_BIT(PMOD,                 0x400C0C28,__READ_WRITE ,__pmod_bits);
__IO_REG8_BIT(PMPUP,                0x400C0C2C,__READ_WRITE ,__pmpup_bits);
__IO_REG8_BIT(PMPDN,                0x400C0C30,__READ_WRITE ,__pmpdn_bits);
__IO_REG8_BIT(PMIE,                 0x400C0C38,__READ_WRITE ,__pmie_bits);

/***************************************************************************
 **
 ** PORTN
 **
 ***************************************************************************/
__IO_REG8_BIT(PNDATA,               0x400C0D00,__READ_WRITE ,__pn_bits);
__IO_REG8_BIT(PNCR,                 0x400C0D04,__READ_WRITE ,__pncr_bits);
__IO_REG8_BIT(PNFR1,                0x400C0D08,__READ_WRITE ,__pnfr1_bits);
__IO_REG8_BIT(PNFR2,                0x400C0D0C,__READ_WRITE ,__pnfr2_bits);
__IO_REG8_BIT(PNOD,                 0x400C0D28,__READ_WRITE ,__pnod_bits);
__IO_REG8_BIT(PNPUP,                0x400C0D2C,__READ_WRITE ,__pnpup_bits);
__IO_REG8_BIT(PNPDN,                0x400C0D30,__READ_WRITE ,__pnpdn_bits);
__IO_REG8_BIT(PNIE,                 0x400C0D38,__READ_WRITE ,__pnie_bits);

/***************************************************************************
 **
 ** EBIF
 **
 ***************************************************************************/
__IO_REG32_BIT(EXBMOD,              0x4005C000, __READ_WRITE ,__exbmod_bits);
__IO_REG32_BIT(EXBAS0,              0x4005C010, __READ_WRITE ,__exbas_bits);
__IO_REG32_BIT(EXBAS1,              0x4005C014, __READ_WRITE ,__exbas_bits);
__IO_REG32_BIT(EXBAS2,              0x4005C018, __READ_WRITE ,__exbas_bits);
__IO_REG32_BIT(EXBAS3,              0x4005C01C, __READ_WRITE ,__exbas_bits);
__IO_REG32_BIT(EXBCS0,              0x4005C040, __READ_WRITE ,__exbcs_bits);
__IO_REG32_BIT(EXBCS1,              0x4005C044, __READ_WRITE ,__exbcs_bits);
__IO_REG32_BIT(EXBCS2,              0x4005C048, __READ_WRITE ,__exbcs_bits);
__IO_REG32_BIT(EXBCS3,              0x4005C04C, __READ_WRITE ,__exbcs_bits);

/***************************************************************************
 **
 ** TMRB0
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
 ** TMRB1
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
 ** TMRB2
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
 ** TMRB3
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
 ** TMRB4
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
 ** TMRB5
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
 ** TMRB6
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
 ** TMRB7
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
 ** SIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(SC0EN,               0x400E1000, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC0BUF,              0x400E1004, __READ_WRITE , __scxbuf_bits);
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

/***************************************************************************
 **
 ** SIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(SC1EN,               0x400E1100, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC1BUF,              0x400E1104, __READ_WRITE , __scxbuf_bits);
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
 ** SIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(SC2EN,               0x400E1200, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC2BUF,              0x400E1204, __READ_WRITE , __scxbuf_bits);
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

/***************************************************************************
 **
 ** SIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(SC3EN,               0x400E1300, __READ_WRITE , __scxen_bits);
__IO_REG32_BIT(SC3BUF,              0x400E1304, __READ_WRITE , __scxbuf_bits);
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
 ** UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(UART4DR,              0x40048000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART4RSR,             0x40048004,__READ_WRITE ,__uartrsr_bits);
#define UART4ECR    UART4RSR
__IO_REG32_BIT(UART4FR,              0x40048018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART4ILPR,            0x40048020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART4IBRD,            0x40048024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART4FBRD,            0x40048028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART4LCR_H,           0x4004802C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG32_BIT(UART4CR,              0x40048030,__READ_WRITE ,__uartcr_bits);
__IO_REG32_BIT(UART4IFLS,            0x40048034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART4IMSC,            0x40048038,__READ_WRITE ,__uartimsc_bits);
__IO_REG32_BIT(UART4RIS,             0x4004803C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART4MIS,             0x40048040,__READ       ,__uartmis_bits);
__IO_REG32_BIT(UART4ICR,             0x40048044,__WRITE      ,__uarticr_bits);
__IO_REG32_BIT(UART4DMACR,           0x40048048,__READ_WRITE ,__uartdmacr_bits);

/***************************************************************************
 **
 ** UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(UART5DR,              0x40049000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART5RSR,             0x40049004,__READ_WRITE ,__uartrsr_bits);
#define UART5ECR    UART5RSR
__IO_REG32_BIT(UART5FR,              0x40049018,__READ       ,__uartfr_bits);
__IO_REG32_BIT(UART5ILPR,            0x40049020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART5IBRD,            0x40049024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART5FBRD,            0x40049028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART5LCR_H,           0x4004902C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG32_BIT(UART5CR,              0x40049030,__READ_WRITE ,__uartcr_bits);
__IO_REG32_BIT(UART5IFLS,            0x40049034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART5IMSC,            0x40049038,__READ_WRITE ,__uartimsc_bits);
__IO_REG32_BIT(UART5RIS,             0x4004903C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART5MIS,             0x40049040,__READ       ,__uartmis_bits);
__IO_REG32_BIT(UART5ICR,             0x40049044,__WRITE      ,__uarticr_bits);
__IO_REG32_BIT(UART5DMACR,           0x40049048,__READ_WRITE ,__uartdmacr_bits);

/***************************************************************************
 **
 ** SBI0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CR0,             0x400E0000, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C0CR1,             0x400E0004, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C0DBR,             0x400E0008, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C0AR,              0x400E000C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C0CR2,             0x400E0010, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C0SR      I2C0CR2
#define I2C0SR_bit  I2C0CR2_bit.__sr
__IO_REG32_BIT(I2C0BR0,             0x400E0014, __READ_WRITE , __sbixbr0_bits);

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
__IO_REG32_BIT(I2C1CR0,             0x400E0100, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C1CR1,             0x400E0104, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C1DBR,             0x400E0108, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C1AR,              0x400E010C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C1CR2,             0x400E0110, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C1SR      I2C1CR2
#define I2C1SR_bit  I2C1CR2_bit.__sr
__IO_REG32_BIT(I2C1BR0,             0x400E0114, __READ_WRITE , __sbixbr0_bits);

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
__IO_REG32_BIT(I2C2CR0,             0x400E0200, __READ_WRITE , __sbixcr0_bits);
__IO_REG32_BIT(I2C2CR1,             0x400E0204, __READ_WRITE , __sbixcr1_bits);
__IO_REG32_BIT(I2C2DBR,             0x400E0208, __READ_WRITE , __sbixdbr_bits);
__IO_REG32_BIT(I2C2AR,              0x400E020C, __READ_WRITE , __sbixi2car_bits);
__IO_REG32_BIT(I2C2CR2,             0x400E0210, __READ_WRITE , __sbixcr2_sr_bits);
#define I2C2SR      I2C2CR2
#define I2C2SR_bit  I2C2CR2_bit.__sr
__IO_REG32_BIT(I2C2BR0,             0x400E0214, __READ_WRITE , __sbixbr0_bits);

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
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,             0x40040000,__READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,             0x40040004,__READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSP0DR,              0x40040008,__READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSP0SR,              0x4004000C,__READ       , __sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,            0x40040010,__READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,            0x40040014,__READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,             0x40040018,__READ       , __sspris_bits);
__IO_REG32_BIT(SSP0MIS,             0x4004001C,__READ       , __sspmis_bits);
__IO_REG32_BIT(SSP0ICR,             0x40040020,__WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,           0x40040024,__READ_WRITE , __sspdmacr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,             0x40041000,__READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,             0x40041004,__READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSP1DR,              0x40041008,__READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSP1SR,              0x4004100C,__READ       , __sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,            0x40041010,__READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,            0x40041014,__READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,             0x40041018,__READ       , __sspris_bits);
__IO_REG32_BIT(SSP1MIS,             0x4004101C,__READ       , __sspmis_bits);
__IO_REG32_BIT(SSP1ICR,             0x40041020,__WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,           0x40041024,__READ_WRITE , __sspdmacr_bits);

/***************************************************************************
 **
 ** SSP2
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP2CR0,             0x40042000,__READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSP2CR1,             0x40042004,__READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSP2DR,              0x40042008,__READ_WRITE , __sspdr_bits);
__IO_REG32_BIT(SSP2SR,              0x4004200C,__READ       , __sspsr_bits);
__IO_REG32_BIT(SSP2CPSR,            0x40042010,__READ_WRITE , __sspcpsr_bits);
__IO_REG32_BIT(SSP2IMSC,            0x40042014,__READ_WRITE , __sspimsc_bits);
__IO_REG32_BIT(SSP2RIS,             0x40042018,__READ       , __sspris_bits);
__IO_REG32_BIT(SSP2MIS,             0x4004201C,__READ       , __sspmis_bits);
__IO_REG32_BIT(SSP2ICR,             0x40042020,__WRITE      , __sspicr_bits);
__IO_REG32_BIT(SSP2DMACR,           0x40042024,__READ_WRITE , __sspdmacr_bits);

/***************************************************************************
 **
 ** USBPLLIF
 **
 ***************************************************************************/
__IO_REG32_BIT(USBPLLCR,             0x400F3100,__READ_WRITE ,__usbpllcr_bits);
__IO_REG32_BIT(USBPLLEN,             0x400F3104,__READ_WRITE ,__usbpllen_bits);
__IO_REG32_BIT(USBPLLSEL,            0x400F3108,__READ_WRITE ,__usbpllsel_bits);

/***************************************************************************
 **
 ** USB OHCI
 **
 ***************************************************************************/
__IO_REG32_BIT(HcRevision,          0x40006000, __READ       , __hcrevision_bits);
__IO_REG32_BIT(HcControl,           0x40006004, __READ_WRITE , __hccontrol_bits);
__IO_REG32_BIT(HcCommandStatus,     0x40006008, __READ_WRITE , __hccommandstatus_bits);
__IO_REG32_BIT(HcInterruptStatus,   0x4000600C, __READ_WRITE , __hcinterruptstatus_bits);
__IO_REG32_BIT(HcInterruptEnable,   0x40006010, __READ_WRITE , __hcinterruptenable_bits);
__IO_REG32_BIT(HcInterruptDisable,  0x40006014, __READ_WRITE , __hcinterruptenable_bits);
__IO_REG32(    HcHCCA,              0x40006018, __READ_WRITE );
__IO_REG32(    HcPeriodCurrentED,   0x4000601C, __READ_WRITE );
__IO_REG32(    HcControlHeadED,     0x40006020, __READ_WRITE );
__IO_REG32(    HcControlCurrentED,  0x40006024, __READ_WRITE );
__IO_REG32(    HcBulkHeadED,        0x40006028, __READ_WRITE );
__IO_REG32(    HcBulkCurrentED,     0x4000602C, __READ_WRITE );
__IO_REG32(    HcDoneHead,          0x40006030, __READ_WRITE );
__IO_REG32_BIT(HcFmInterval,        0x40006034, __READ_WRITE , __hcfminterval_bits);
__IO_REG32_BIT(HcFmRemaining,       0x40006038, __READ_WRITE , __hcfmremaining_bits);
__IO_REG32_BIT(HcFmNumber,          0x4000603C, __READ_WRITE , __hcfmnumber_bits);
__IO_REG32_BIT(HcPeriodStart,       0x40006040, __READ_WRITE , __hcperiodstart_bits);
__IO_REG32_BIT(HcLSThreshold,       0x40006044, __READ_WRITE , __hclsthreshold_bits);
__IO_REG32_BIT(HcRhDescriptorA,     0x40006048, __READ_WRITE , __hcrhdescriptora_bits);
__IO_REG32_BIT(HcRhDescriptorB,     0x4000604C, __READ_WRITE , __hcrhdescriptorb_bits);
__IO_REG32_BIT(HcRhStatus,          0x40006050, __READ_WRITE , __hcrhstatus_bits);
__IO_REG32_BIT(HcRhPortStatus1,     0x40006054, __READ_WRITE , __hcrhportstatus_bits);
__IO_REG32_BIT(HcBCR0,              0x40006080, __READ_WRITE , __hcbcr0_bits);

/***************************************************************************
 **
 ** UDC2AB Bridge
 **
 ***************************************************************************/
__IO_REG32_BIT(UDFSINTSTS,            0x40008000,__READ_WRITE ,__udintsts_bits);
__IO_REG32_BIT(UDFSINTENB,            0x40008004,__READ_WRITE ,__udintenb_bits);
__IO_REG32_BIT(UDFSMWTOUT,            0x40008008,__READ_WRITE ,__udmwtout_bits);
__IO_REG32_BIT(UDFSC2STSET,           0x4000800C,__READ_WRITE ,__udc2stset_bits);
__IO_REG32_BIT(UDFSMSTSET,            0x40008010,__READ_WRITE ,__udmstset_bits);
__IO_REG32_BIT(UDFSDMACRDREQ,         0x40008014,__READ_WRITE ,__uddmacrdreq_bits);
__IO_REG32(    UDFSDMACRDVL,          0x40008018,__READ       );
__IO_REG32_BIT(UDFSUDC2RDREQ,         0x4000801C,__READ_WRITE ,__udc2rdreq_bits);
__IO_REG32_BIT(UDFSUDC2RDVL,          0x40008020,__READ       ,__udc2rdvl_bits);
__IO_REG32_BIT(UDFSARBTSET,           0x4000803C,__READ_WRITE ,__udarbtset_bits);
__IO_REG32(    UDFSMWSADR,            0x40008040,__READ_WRITE );
__IO_REG32(    UDFSMWEADR,            0x40008044,__READ_WRITE );
__IO_REG32(    UDFSMWCADR,            0x40008048,__READ       );
__IO_REG32(    UDFSMWAHBADR,          0x4000804C,__READ       );
__IO_REG32(    UDFSMRSADR,            0x40008050,__READ_WRITE );
__IO_REG32(    UDFSMREADR,            0x40008054,__READ_WRITE );
__IO_REG32(    UDFSMRCADR,            0x40008058,__READ       );
__IO_REG32(    UDFSMRAHBADR,          0x4000805C,__READ       );
__IO_REG32_BIT(UDFSPWCTL,             0x40008080,__READ_WRITE ,__udpwctl_bits);
__IO_REG32_BIT(UDFSMSTSTS,            0x40008084,__READ       ,__udmststs_bits);
__IO_REG32(    UDFSTOUTCNT,           0x40008088,__READ       );

/***************************************************************************
 **
 ** UDC2
 **
 ***************************************************************************/
__IO_REG32_BIT(UDFS2ADR,              0x40008200,__READ_WRITE ,__ud2adr_bits);
__IO_REG32_BIT(UDFS2FRM,              0x40008204,__READ_WRITE ,__ud2frm_bits);
__IO_REG32_BIT(UDFS2CMD,              0x4000820C,__READ_WRITE ,__ud2cmd_bits);
__IO_REG32_BIT(UDFS2BRQ,              0x40008210,__READ       ,__ud2brq_bits);
__IO_REG32_BIT(UDFS2WVL,              0x40008214,__READ       ,__ud2wvl_bits);
__IO_REG32_BIT(UDFS2WIDX,             0x40008218,__READ       ,__ud2widx_bits);
__IO_REG32_BIT(UDFS2WLGTH,            0x4000821C,__READ       ,__ud2wlgth_bits);
__IO_REG32_BIT(UDFS2INT,              0x40008220,__READ_WRITE ,__ud2int_bits);
__IO_REG32_BIT(UDFS2INTEP,            0x40008224,__READ_WRITE ,__ud2intep_bits);
__IO_REG32_BIT(UDFS2INTEPMSK,         0x40008228,__READ_WRITE ,__ud2intepmsk_bits);
__IO_REG32_BIT(UDFS2INTRX0,           0x4000822C,__READ_WRITE ,__ud2intrx0_bits);
__IO_REG32_BIT(UDFS2EP0MSZ,           0x40008230,__READ_WRITE ,__ud2ep0msz_bits);
__IO_REG32_BIT(UDFS2EP0STS,           0x40008234,__READ       ,__ud2ep0sts_bits);
__IO_REG32_BIT(UDFS2EP0DSZ,           0x40008238,__READ       ,__ud2ep0dsz_bits);
__IO_REG32_BIT(UDFS2EP0FIFO,          0x4000823C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP1MSZ,           0x40008240,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP1STS,           0x40008244,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP1DSZ,           0x40008248,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP1FIFO,          0x4000824C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP2MSZ,           0x40008250,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP2STS,           0x40008254,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP2DSZ,           0x40008258,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP2FIFO,          0x4000825C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP3MSZ,           0x40008260,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP3STS,           0x40008264,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP3DSZ,           0x40008268,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP3FIFO,          0x4000826C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP4MSZ,           0x40008270,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP4STS,           0x40008274,__READ       ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP4DSZ,           0x40008278,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP4FIFO,          0x4000827C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP5MSZ,           0x40008280,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP5STS,           0x40008284,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP5DSZ,           0x40008288,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP5FIFO,          0x4000828C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP6MSZ,           0x40008290,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP6STS,           0x40008294,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP6DSZ,           0x40008298,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP6FIFO,          0x4000829C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2EP7MSZ,           0x400082A0,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UDFS2EP7STS,           0x400082A4,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UDFS2EP7DSZ,           0x400082A8,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UDFS2EP7FIFO,          0x400082AC,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UDFS2INTNAK,           0x40008330,__READ_WRITE ,__ud2intep_bits);
__IO_REG32_BIT(UDFS2INTNAKMSK,        0x40008334,__READ_WRITE ,__ud2intepmsk_bits);

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CANMB0ID,            0x40005000, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB0TSVMCF,        0x40005008, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB0DH,            0x40005010, __READ_WRITE );
__IO_REG32(    CANMB0DL,            0x40005018, __READ_WRITE );
__IO_REG32_BIT(CANMB1ID,            0x40005020, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB1TSVMCF,        0x40005028, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB1DH,            0x40005030, __READ_WRITE );
__IO_REG32(    CANMB1DL,            0x40005038, __READ_WRITE );
__IO_REG32_BIT(CANMB2ID,            0x40005040, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB2TSVMCF,        0x40005048, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB2DH,            0x40005050, __READ_WRITE );
__IO_REG32(    CANMB2DL,            0x40005058, __READ_WRITE );
__IO_REG32_BIT(CANMB3ID,            0x40005060, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB3TSVMCF,        0x40005068, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB3DH,            0x40005070, __READ_WRITE );
__IO_REG32(    CANMB3DL,            0x40005078, __READ_WRITE );
__IO_REG32_BIT(CANMB4ID,            0x40005080, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB4TSVMCF,        0x40005088, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB4DH,            0x40005090, __READ_WRITE );
__IO_REG32(    CANMB4DL,            0x40005098, __READ_WRITE );
__IO_REG32_BIT(CANMB5ID,            0x400050A0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB5TSVMCF,        0x400050A8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB5DH,            0x400050B0, __READ_WRITE );
__IO_REG32(    CANMB5DL,            0x400050B8, __READ_WRITE );
__IO_REG32_BIT(CANMB6ID,            0x400050C0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB6TSVMCF,        0x400050C8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB6DH,            0x400050D0, __READ_WRITE );
__IO_REG32(    CANMB6DL,            0x400050D8, __READ_WRITE );
__IO_REG32_BIT(CANMB7ID,            0x400050E0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB7TSVMCF,        0x400050E8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB7DH,            0x400050F0, __READ_WRITE );
__IO_REG32(    CANMB7DL,            0x400050F8, __READ_WRITE );
__IO_REG32_BIT(CANMB8ID,            0x40005100, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB8TSVMCF,        0x40005108, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB8DH,            0x40005110, __READ_WRITE );
__IO_REG32(    CANMB8DL,            0x40005118, __READ_WRITE );
__IO_REG32_BIT(CANMB9ID,            0x40005120, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB9TSVMCF,        0x40005128, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB9DH,            0x40005130, __READ_WRITE );
__IO_REG32(    CANMB9DL,            0x40005138, __READ_WRITE );
__IO_REG32_BIT(CANMB10ID,           0x40005140, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB10TSVMCF,       0x40005148, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB10DH,           0x40005150, __READ_WRITE );
__IO_REG32(    CANMB10DL,           0x40005158, __READ_WRITE );
__IO_REG32_BIT(CANMB11ID,           0x40005160, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB11TSVMCF,       0x40005168, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB11DH,           0x40005170, __READ_WRITE );
__IO_REG32(    CANMB11DL,           0x40005178, __READ_WRITE );
__IO_REG32_BIT(CANMB12ID,           0x40005180, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB12TSVMCF,       0x40005188, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB12DH,           0x40005190, __READ_WRITE );
__IO_REG32(    CANMB12DL,           0x40005198, __READ_WRITE );
__IO_REG32_BIT(CANMB13ID,           0x400051A0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB13TSVMCF,       0x400051A8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB13DH,           0x400051B0, __READ_WRITE );
__IO_REG32(    CANMB13DL,           0x400051B8, __READ_WRITE );
__IO_REG32_BIT(CANMB14ID,           0x400051C0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB14TSVMCF,       0x400051C8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB14DH,           0x400051D0, __READ_WRITE );
__IO_REG32(    CANMB14DL,           0x400051D8, __READ_WRITE );
__IO_REG32_BIT(CANMB15ID,           0x400051E0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB15TSVMCF,       0x400051E8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB15DH,           0x400051F0, __READ_WRITE );
__IO_REG32(    CANMB15DL,           0x400051F8, __READ_WRITE );
__IO_REG32_BIT(CANMB16ID,           0x40005200, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB16TSVMCF,       0x40005208, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB16DH,           0x40005210, __READ_WRITE );
__IO_REG32(    CANMB16DL,           0x40005218, __READ_WRITE );
__IO_REG32_BIT(CANMB17ID,           0x40005220, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB17TSVMCF,       0x40005228, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB17DH,           0x40005230, __READ_WRITE );
__IO_REG32(    CANMB17DL,           0x40005238, __READ_WRITE );
__IO_REG32_BIT(CANMB18ID,           0x40005240, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB18TSVMCF,       0x40005248, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB18DH,           0x40005250, __READ_WRITE );
__IO_REG32(    CANMB18DL,           0x40005258, __READ_WRITE );
__IO_REG32_BIT(CANMB19ID,           0x40005260, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB19TSVMCF,       0x40005268, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB19DH,           0x40005270, __READ_WRITE );
__IO_REG32(    CANMB19DL,           0x40005278, __READ_WRITE );
__IO_REG32_BIT(CANMB20ID,           0x40005280, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB20TSVMCF,       0x40005288, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB20DH,           0x40005290, __READ_WRITE );
__IO_REG32(    CANMB20DL,           0x40005298, __READ_WRITE );
__IO_REG32_BIT(CANMB21ID,           0x400052A0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB21TSVMCF,       0x400052A8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB21DH,           0x400052B0, __READ_WRITE );
__IO_REG32(    CANMB21DL,           0x400052B8, __READ_WRITE );
__IO_REG32_BIT(CANMB22ID,           0x400052C0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB22TSVMCF,       0x400052C8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB22DH,           0x400052D0, __READ_WRITE );
__IO_REG32(    CANMB22DL,           0x400052D8, __READ_WRITE );
__IO_REG32_BIT(CANMB23ID,           0x400052E0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB23TSVMCF,       0x400052E8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB23DH,           0x400052F0, __READ_WRITE );
__IO_REG32(    CANMB23DL,           0x400052F8, __READ_WRITE );
__IO_REG32_BIT(CANMB24ID,           0x40005300, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB24TSVMCF,       0x40005308, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB24DH,           0x40005310, __READ_WRITE );
__IO_REG32(    CANMB24DL,           0x40005318, __READ_WRITE );
__IO_REG32_BIT(CANMB25ID,           0x40005320, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB25TSVMCF,       0x40005328, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB25DH,           0x40005330, __READ_WRITE );
__IO_REG32(    CANMB25DL,           0x40005338, __READ_WRITE );
__IO_REG32_BIT(CANMB26ID,           0x40005340, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB26TSVMCF,       0x40005348, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB26DH,           0x40005350, __READ_WRITE );
__IO_REG32(    CANMB26DL,           0x40005358, __READ_WRITE );
__IO_REG32_BIT(CANMB27ID,           0x40005360, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB27TSVMCF,       0x40005368, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB27DH,           0x40005370, __READ_WRITE );
__IO_REG32(    CANMB27DL,           0x40005378, __READ_WRITE );
__IO_REG32_BIT(CANMB28ID,           0x40005380, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB28TSVMCF,       0x40005388, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB28DH,           0x40005390, __READ_WRITE );
__IO_REG32(    CANMB28DL,           0x40005398, __READ_WRITE );
__IO_REG32_BIT(CANMB29ID,           0x400053A0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB29TSVMCF,       0x400053A8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB29DH,           0x400053B0, __READ_WRITE );
__IO_REG32(    CANMB29DL,           0x400053B8, __READ_WRITE );
__IO_REG32_BIT(CANMB30ID,           0x400053C0, __READ_WRITE , __canmbid_bits);
__IO_REG32_BIT(CANMB30TSVMCF,       0x400053C8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB30DH,           0x400053D0, __READ_WRITE );
__IO_REG32(    CANMB30DL,           0x400053D8, __READ_WRITE );
__IO_REG32_BIT(CANMB31ID,           0x400053E0, __READ_WRITE , __canmb31id_bits);
__IO_REG32_BIT(CANMB31TSVMCF,       0x400053E8, __READ_WRITE , __canmbtcr_bits);
__IO_REG32(    CANMB31DH,           0x400053F0, __READ_WRITE );
__IO_REG32(    CANMB31DL,           0x400053F8, __READ_WRITE );
__IO_REG32_BIT(CANMC,               0x40005400, __READ_WRITE , __canmc_bits);
__IO_REG32_BIT(CANMD,               0x40005408, __READ_WRITE , __canmd_bits);
__IO_REG32_BIT(CANTRS,              0x40005410, __READ_WRITE , __cantrs_bits);
__IO_REG32_BIT(CANTRR,              0x40005418, __READ_WRITE , __cantrr_bits);
__IO_REG32_BIT(CANTA,               0x40005420, __READ_WRITE , __canta_bits);
__IO_REG32_BIT(CANAA,               0x40005428, __READ_WRITE , __canaa_bits);
__IO_REG32_BIT(CANRMP,              0x40005430, __READ_WRITE , __canrmp_bits);
__IO_REG32_BIT(CANRML,              0x40005438, __READ_WRITE , __canrml_bits);
__IO_REG32_BIT(CANLAM,              0x40005440, __READ_WRITE , __canlam_bits);
__IO_REG32_BIT(CANGAM,              0x40005448, __READ_WRITE , __cangam_bits);
__IO_REG32_BIT(CANMCR,              0x40005450, __READ_WRITE , __canmcr_bits);
__IO_REG32_BIT(CANGSR,              0x40005458, __READ       , __cangsr_bits);
__IO_REG32_BIT(CANBCR1,             0x40005460, __READ_WRITE , __canbcr1_bits);
__IO_REG32_BIT(CANBCR2,             0x40005468, __READ_WRITE , __canbcr2_bits);
__IO_REG32_BIT(CANGIF,              0x40005470, __READ_WRITE , __cangif_bits);
__IO_REG32_BIT(CANGIM,              0x40005478, __READ_WRITE , __cangim_bits);
__IO_REG32_BIT(CANMBTIF,            0x40005480, __READ_WRITE , __canmbtif_bits);
__IO_REG32_BIT(CANMBRIF,            0x40005488, __READ_WRITE , __canmbrif_bits);
__IO_REG32_BIT(CANMBIM,             0x40005490, __READ_WRITE , __canmbim_bits);
__IO_REG32_BIT(CANCDR,              0x40005498, __READ_WRITE , __cancdr_bits);
__IO_REG32_BIT(CANRFP,              0x400054A0, __READ_WRITE , __canrfp_bits);
__IO_REG32_BIT(CANCEC,              0x400054A8, __READ       , __cancec_bits);
__IO_REG32_BIT(CANTSP,              0x400054B0, __READ_WRITE , __cantsp_bits);
__IO_REG32_BIT(CANTSC,              0x400054B8, __READ_WRITE , __cantsc_bits);

/***************************************************************************
 **
 ** EtherMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(EMIFVRR,             0x4000A000, __READ_WRITE , __emifvrr_bits);
__IO_REG32_BIT(EMIFCR,              0x4000A004, __READ_WRITE , __emifcr_bits);
__IO_REG32_BIT(EMIFINTST,           0x4000A010, __READ_WRITE , __emifintst_bits);
__IO_REG32_BIT(EMIFINTEN,           0x4000A014, __READ_WRITE , __emifinten_bits);
__IO_REG32_BIT(EMDCR,               0x4000B000, __READ_WRITE , __emdcr_bits);
__IO_REG32_BIT(EMDTEN,              0x4000B008, __READ_WRITE , __emdten_bits);
__IO_REG32_BIT(EMDREN,              0x4000B010, __READ_WRITE , __emdren_bits);
__IO_REG32(    EMDTDPR,             0x4000B018, __READ_WRITE );
__IO_REG32(    EMDRDPR,             0x4000B020, __READ_WRITE );
__IO_REG32_BIT(EMDINTST,            0x4000B028, __READ_WRITE , __emdint_bits);
__IO_REG32_BIT(EMDINTEN,            0x4000B030, __READ_WRITE , __emdint_bits);
__IO_REG32_BIT(EMDERR,              0x4000B038, __READ_WRITE , __emderr_bits);
__IO_REG32_BIT(EMDMFCR,             0x4000B040, __READ_WRITE , __emdmfcr_bits);
__IO_REG32_BIT(EMDTFTR,             0x4000B048, __READ_WRITE , __emdtftr_bits);
__IO_REG32_BIT(EMDFCDR,             0x4000B050, __READ_WRITE , __emdfcdr_bits);
__IO_REG32_BIT(EMDRRCR,             0x4000B058, __READ_WRITE , __emdrrcr_bits);
__IO_REG32(    EMDEADR,             0x4000B060, __READ_WRITE );
__IO_REG32_BIT(EMDTUFR,             0x4000B064, __READ_WRITE , __emdtufr_bits);
__IO_REG32_BIT(EMDROFR,             0x4000B068, __READ_WRITE , __emdrofr_bits);
__IO_REG32_BIT(EMDRBTR,             0x4000B070, __READ_WRITE , __emdrbtr_bits);
__IO_REG32_BIT(EMDRPDR,             0x4000B078, __READ_WRITE , __emdrpdr_bits);
__IO_REG32_BIT(EMDTIMER,            0x4000B07C, __READ_WRITE , __emdtimer_bits);
__IO_REG32(    EMDRBAR,             0x4000B0C8, __READ );
__IO_REG32(    EMDRDAR,             0x4000B0CC, __READ );
__IO_REG32(    EMDTBAR,             0x4000B0D4, __READ );
__IO_REG32(    EMDTDAR,             0x4000B0D8, __READ );
__IO_REG32_BIT(EMCCR,               0x4000B100, __READ_WRITE , __emccr_bits);
__IO_REG32_BIT(EMCRFLR,             0x4000B108, __READ_WRITE , __emcrflr_bits);
__IO_REG32_BIT(EMCINTST,            0x4000B110, __READ_WRITE , __emcint_bits);
__IO_REG32_BIT(EMCINTEN,            0x4000B118, __READ_WRITE , __emcint_bits);
__IO_REG32_BIT(EMCMIICR,            0x4000B120, __READ_WRITE , __emcmiicr_bits);
__IO_REG32_BIT(EMCPHYST,            0x4000B128, __READ       , __emcphyst_bits);
__IO_REG32_BIT(EMCICR,              0x4000B150, __READ_WRITE , __emcicr_bits);
__IO_REG32_BIT(EMCAPR,              0x4000B154, __READ_WRITE , __emcapr_bits);
__IO_REG32_BIT(EMCMPR,              0x4000B158, __WRITE      , __emcmpr_bits);
__IO_REG32_BIT(EMCRPCR,             0x4000B160, __READ       , __emcrpcr_bits);
__IO_REG32_BIT(EMCPULR,             0x4000B164, __READ_WRITE , __emcpulr_bits);
__IO_REG32_BIT(EMCPCR,              0x4000B168, __READ       , __emcpcr_bits);
__IO_REG32_BIT(EMCBRR,              0x4000B16C, __READ_WRITE , __emcbrr_bits);
__IO_REG32(    EMCMAHR,             0x4000B1C0, __READ_WRITE );
__IO_REG32_BIT(EMCMALR,             0x4000B1C8, __READ_WRITE , __emcmalr_bits);
__IO_REG32(    EMCTRER,             0x4000B1D0, __READ_WRITE );
__IO_REG32(    EMCTCDR,             0x4000B1D4, __READ_WRITE );
__IO_REG32(    EMCTCLR,             0x4000B1D8, __READ_WRITE );
__IO_REG32(    EMCTNDR,             0x4000B1DC, __READ_WRITE );
__IO_REG32(    EMCRCER,             0x4000B1E4, __READ_WRITE );
__IO_REG32(    EMCRFER,             0x4000B1E8, __READ_WRITE );
__IO_REG32(    EMCRLSR,             0x4000B1EC, __READ_WRITE );
__IO_REG32(    EMCRLOR,             0x4000B1F0, __READ_WRITE );
__IO_REG32(    EMCRRFR,             0x4000B1F4, __READ_WRITE );
__IO_REG32(    EMCRMFR,             0x4000B1F8, __READ_WRITE );

/***************************************************************************
 **
 ** RMC
 **
 ***************************************************************************/
 __IO_REG32_BIT(RMCEN,              0x400E7000, __READ_WRITE , __rmcen_bits   );
 __IO_REG32_BIT(RMCREN,             0x400E7004, __READ_WRITE , __rmcren_bits  );
 __IO_REG32(    RMCRBUF1,           0x400E7008, __READ);
 __IO_REG32(    RMCRBUF2,           0x400E700C, __READ);
 __IO_REG32(    RMCRBUF3,           0x400E7010, __READ);
 __IO_REG32_BIT(RMCRCR1,            0x400E7014, __READ_WRITE , __rmcrcr1_bits );
 __IO_REG32_BIT(RMCRCR2,            0x400E7018, __READ_WRITE , __rmcrcr2_bits );
 __IO_REG32_BIT(RMCRCR3,            0x400E701C, __READ_WRITE , __rmcrcr3_bits );
 __IO_REG32_BIT(RMCRCR4,            0x400E7020, __READ_WRITE , __rmcrcr4_bits );
 __IO_REG32_BIT(RMCRSTAT,           0x400E7024, __READ       , __rmcrstat_bits);
 __IO_REG32_BIT(RMCEND1,            0x400E7028, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMCEND2,            0x400E702C, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMCEND3,            0x400E7030, __READ_WRITE , __rmcend_bits );
 __IO_REG32_BIT(RMCFSSEL,           0x400E7034, __READ_WRITE , __rmcfssel_bits);

/***************************************************************************
 **
 ** ADC-A
 **
 ***************************************************************************/
__IO_REG32_BIT(ADACLK,               0x40050000, __READ_WRITE ,__adclk_bits);
__IO_REG32_BIT(ADAMOD0,              0x40050004, __READ_WRITE ,__admod0_bits);
__IO_REG32_BIT(ADAMOD1,              0x40050008, __READ_WRITE ,__admod1_bits);
__IO_REG32_BIT(ADAMOD2,              0x4005000C, __READ_WRITE ,__admod2_bits);
__IO_REG32_BIT(ADAMOD3,              0x40050010, __READ_WRITE ,__admod3_bits);
__IO_REG32_BIT(ADAMOD4,              0x40050014, __READ_WRITE ,__admod4_bits);
__IO_REG32_BIT(ADAMOD5,              0x40050018, __READ_WRITE ,__admod5_bits);
__IO_REG32_BIT(ADAMOD6,              0x4005001C, __READ_WRITE ,__admod6_bits);
__IO_REG32_BIT(ADACMPCR0,            0x40050024, __READ_WRITE ,__adcmpcr0_bits);
__IO_REG32_BIT(ADACMPCR1,            0x40050028, __READ_WRITE ,__adcmpcr1_bits);
__IO_REG32_BIT(ADACMP0,              0x4005002C, __READ_WRITE ,__adcmp0_bits);
__IO_REG32_BIT(ADACMP1,              0x40050030, __READ_WRITE ,__adcmp1_bits);
__IO_REG32_BIT(ADAREG00,             0x40050034, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG01,             0x40050038, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG02,             0x4005003C, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG03,             0x40050040, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG04,             0x40050044, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG05,             0x40050048, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG06,             0x4005004C, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREG07,             0x40050050, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADAREGSP,             0x40050074, __READ       ,__adregsp_bits);

/***************************************************************************
 **
 ** ADC-B
 **
 ***************************************************************************/
__IO_REG32_BIT(ADBCLK,               0x40051000, __READ_WRITE ,__adclk_bits);
__IO_REG32_BIT(ADBMOD0,              0x40051004, __READ_WRITE ,__admod0_bits);
__IO_REG32_BIT(ADBMOD1,              0x40051008, __READ_WRITE ,__admod1_bits);
__IO_REG32_BIT(ADBMOD2,              0x4005100C, __READ_WRITE ,__admod2_bits);
__IO_REG32_BIT(ADBMOD3,              0x40051010, __READ_WRITE ,__admod3_bits);
__IO_REG32_BIT(ADBMOD4,              0x40051014, __READ_WRITE ,__admod4_bits);
__IO_REG32_BIT(ADBMOD5,              0x40051018, __READ_WRITE ,__admod5_bits);
__IO_REG32_BIT(ADBMOD6,              0x4005101C, __READ_WRITE ,__admod6_bits);
__IO_REG32_BIT(ADBCMPCR0,            0x40051024, __READ_WRITE ,__adcmpcr0_bits);
__IO_REG32_BIT(ADBCMPCR1,            0x40051028, __READ_WRITE ,__adcmpcr1_bits);
__IO_REG32_BIT(ADBCMP0,              0x4005102C, __READ_WRITE ,__adcmp0_bits);
__IO_REG32_BIT(ADBCMP1,              0x40051030, __READ_WRITE ,__adcmp1_bits);
__IO_REG32_BIT(ADBREG00,             0x40051034, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG01,             0x40051038, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG02,             0x4005103C, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG03,             0x40051040, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG04,             0x40051044, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG05,             0x40051048, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG06,             0x4005104C, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG07,             0x40051050, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG08,             0x40051054, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG09,             0x40051058, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG10,             0x4005105C, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREG11,             0x40051060, __READ       ,__adregx_bits);
__IO_REG32_BIT(ADBREGSP,             0x40051074, __READ       ,__adregsp_bits);

/***************************************************************************
 **
 ** ADILV
 **
 ***************************************************************************/
__IO_REG32_BIT(ADILVMO1,             0x40066000, __READ_WRITE ,__adilvmo1_bits);
__IO_REG32_BIT(ADILVMO2,             0x40066004, __READ_WRITE ,__adilvmo2_bits);
__IO_REG32_BIT(ADILVMO3,             0x40066008, __READ_WRITE ,__adilvmo3_bits);

/***************************************************************************
 **
 ** DAC 0
 **
 ***************************************************************************/
__IO_REG32_BIT(DACCNT0,              0x40054000, __READ_WRITE ,__daccntx_bits);
__IO_REG32_BIT(DACREG0,              0x40054004, __READ_WRITE ,__dacregx_bits);
__IO_REG32_BIT(DACDCTL0,             0x40054008, __READ_WRITE ,__dacdctlx_bits);
__IO_REG32_BIT(DACTCTL0,             0x4005400C, __READ_WRITE ,__dactctlx_bits);
__IO_REG32_BIT(DACVCTL0,             0x40054010, __READ_WRITE ,__dacvctlx_bits);

/***************************************************************************
 **
 ** DAC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(DACCNT1,              0x40055000, __READ_WRITE ,__daccntx_bits);
__IO_REG32_BIT(DACREG1,              0x40055004, __READ_WRITE ,__dacregx_bits);
__IO_REG32_BIT(DACDCTL1,             0x40055008, __READ_WRITE ,__dacdctlx_bits);
__IO_REG32_BIT(DACTCTL1,             0x4005500C, __READ_WRITE ,__dactctlx_bits);
__IO_REG32_BIT(DACVCTL1,             0x40055010, __READ_WRITE ,__dacvctlx_bits);

/***************************************************************************
 **
 ** MPT 0
 **
 ***************************************************************************/
__IO_REG32_BIT(MT0EN,               0x400C7000, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT0RUN,              0x400C7004, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT0TBCR,             0x400C7008, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT0TBMOD,            0x400C700C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT0TBFFCR,           0x400C7010, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT0TBST,             0x400C7014, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT0TBIM,             0x400C7018, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT0TBUC,             0x400C701C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT0RG0,              0x400C7020, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT0RG1,              0x400C7024, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT0CP0,              0x400C7028, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT0CP1,              0x400C702C, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MT0IGCR,             0x400C7030, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MT0IGRESTA,          0x400C7034, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MT0IGST,             0x400C7038, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MT0IGICR,            0x400C703C, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MT0IGOCR,            0x400C7040, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MT0IGRG2,            0x400C7044, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MT0IGRG3,            0x400C7048, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MT0IGRG4,            0x400C704C, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MT0IGEMGCR,          0x400C7050, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MT0IGEMGST,          0x400C7054, __READ       , __mtigxemgst_bits);

/***************************************************************************
 **
 ** MPT 1
 **
 ***************************************************************************/
__IO_REG32_BIT(MT1EN,               0x400C7100, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT1RUN,              0x400C7104, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT1TBCR,             0x400C7108, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT1TBMOD,            0x400C710C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT1TBFFCR,           0x400C7110, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT1TBST,             0x400C7114, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT1TBIM,             0x400C7118, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT1TBUC,             0x400C711C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT1RG0,              0x400C7120, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT1RG1,              0x400C7124, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT1CP0,              0x400C7128, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT1CP1,              0x400C712C, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MT1IGCR,             0x400C7130, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MT1IGRESTA,          0x400C7134, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MT1IGST,             0x400C7138, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MT1IGICR,            0x400C713C, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MT1IGOCR,            0x400C7140, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MT1IGRG2,            0x400C7144, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MT1IGRG3,            0x400C7148, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MT1IGRG4,            0x400C714C, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MT1IGEMGCR,          0x400C7150, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MT1IGEMGST,          0x400C7154, __READ       , __mtigxemgst_bits);

/***************************************************************************
 **
 ** MPT 2
 **
 ***************************************************************************/
__IO_REG32_BIT(MT2EN,               0x400C7200, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT2RUN,              0x400C7204, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT2TBCR,             0x400C7208, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT2TBMOD,            0x400C720C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT2TBFFCR,           0x400C7210, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT2TBST,             0x400C7214, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT2TBIM,             0x400C7218, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT2TBUC,             0x400C721C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT2RG0,              0x400C7220, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT2RG1,              0x400C7224, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT2CP0,              0x400C7228, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT2CP1,              0x400C722C, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MT2IGCR,             0x400C7230, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MT2IGRESTA,          0x400C7234, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MT2IGST,             0x400C7238, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MT2IGICR,            0x400C723C, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MT2IGOCR,            0x400C7240, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MT2IGRG2,            0x400C7244, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MT2IGRG3,            0x400C7248, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MT2IGRG4,            0x400C724C, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MT2IGEMGCR,          0x400C7250, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MT2IGEMGST,          0x400C7254, __READ       , __mtigxemgst_bits);

/***************************************************************************
 **
 ** MPT 3
 **
 ***************************************************************************/
__IO_REG32_BIT(MT3EN,               0x400C7300, __READ_WRITE , __mtxen_bits);
__IO_REG32_BIT(MT3RUN,              0x400C7304, __READ_WRITE , __mtxrun_bits);
__IO_REG32_BIT(MT3TBCR,             0x400C7308, __READ_WRITE , __mtxcr_bits);
__IO_REG32_BIT(MT3TBMOD,            0x400C730C, __READ_WRITE , __mtxmod_bits);
__IO_REG32_BIT(MT3TBFFCR,           0x400C7310, __READ_WRITE , __mtxffcr_bits);
__IO_REG32_BIT(MT3TBST,             0x400C7314, __READ       , __mtxst_bits);
__IO_REG32_BIT(MT3TBIM,             0x400C7318, __READ_WRITE , __mtxim_bits);
__IO_REG32_BIT(MT3TBUC,             0x400C731C, __READ       , __mtxuc_bits);
__IO_REG32_BIT(MT3RG0,              0x400C7320, __READ_WRITE , __mtxrg0_bits);
__IO_REG32_BIT(MT3RG1,              0x400C7324, __READ_WRITE , __mtxrg1_bits);
__IO_REG32_BIT(MT3CP0,              0x400C7328, __READ       , __mtxcp0_bits);
__IO_REG32_BIT(MT3CP1,              0x400C732C, __READ       , __mtxcp1_bits);
__IO_REG32_BIT(MT3IGCR,             0x400C7330, __READ_WRITE , __mtigxcr_bits);
__IO_REG32_BIT(MT3IGRESTA,          0x400C7334, __WRITE      , __mtigxresta_bits);
__IO_REG32_BIT(MT3IGST,             0x400C7338, __READ_WRITE , __mtigxst_bits);
__IO_REG32_BIT(MT3IGICR,            0x400C733C, __READ_WRITE , __mtigxicr_bits);
__IO_REG32_BIT(MT3IGOCR,            0x400C7340, __READ_WRITE , __mtigxocr_bits);
__IO_REG32_BIT(MT3IGRG2,            0x400C7344, __READ_WRITE , __mtigxrg2_bits);
__IO_REG32_BIT(MT3IGRG3,            0x400C7348, __READ_WRITE , __mtigxrg3_bits);
__IO_REG32_BIT(MT3IGRG4,            0x400C734C, __READ_WRITE , __mtigxrg4_bits);
__IO_REG32_BIT(MT3IGEMGCR,          0x400C7350, __READ_WRITE , __mtigxemgcr_bits);
__IO_REG32_BIT(MT3IGEMGST,          0x400C7354, __READ       , __mtigxemgst_bits);

/***************************************************************************
 **
 ** MPT PMD 0
 **
 ***************************************************************************/
__IO_REG32_BIT(MTPD0MDEN,            0x400F6000, __READ_WRITE , __mden_bits);
__IO_REG32_BIT(MTPD0PORTMD,          0x400F6004, __READ_WRITE , __portmd_bits);
__IO_REG32_BIT(MTPD0MDCR,            0x400F6008, __READ_WRITE , __mdcr_bits);
__IO_REG32_BIT(MTPD0CNTSTA,          0x400F600C, __READ       , __cntsta_bits);
__IO_REG32_BIT(MTPD0MDCNT,           0x400F6010, __READ       , __mdcnt_bits);
__IO_REG32_BIT(MTPD0MDPRD,           0x400F6014, __READ_WRITE , __mdprd_bits);
__IO_REG32_BIT(MTPD0CMPU,            0x400F6018, __READ_WRITE , __cmpu_bits);
__IO_REG32_BIT(MTPD0CMPV,            0x400F601C, __READ_WRITE , __cmpv_bits);
__IO_REG32_BIT(MTPD0CMPW,            0x400F6020, __READ_WRITE , __cmpw_bits);
__IO_REG32_BIT(MTPD0MDOUT,           0x400F6028, __READ_WRITE , __mdout_bits);
__IO_REG32_BIT(MTPD0MDPOT,           0x400F602C, __READ_WRITE , __mdpot_bits);
__IO_REG32_BIT(MTPD0EMGREL,          0x400F6030, __WRITE      , __emgrel_bits);
__IO_REG32_BIT(MTPD0EMGCR,           0x400F6034, __READ_WRITE , __emgcr_bits);
__IO_REG32_BIT(MTPD0EMGSTA,          0x400F6038, __READ       , __emgsta_bits);
__IO_REG32_BIT(MTPD0DTR,             0x400F6044, __READ_WRITE , __dtr_bits);

/***************************************************************************
 **
 ** MPT PMD 1
 **
 ***************************************************************************/
__IO_REG32_BIT(MTPD1MDEN,            0x400F6100, __READ_WRITE , __mden_bits);
__IO_REG32_BIT(MTPD1PORTMD,          0x400F6104, __READ_WRITE , __portmd_bits);
__IO_REG32_BIT(MTPD1MDCR,            0x400F6108, __READ_WRITE , __mdcr_bits);
__IO_REG32_BIT(MTPD1CNTSTA,          0x400F610C, __READ       , __cntsta_bits);
__IO_REG32_BIT(MTPD1MDCNT,           0x400F6110, __READ       , __mdcnt_bits);
__IO_REG32_BIT(MTPD1MDPRD,           0x400F6114, __READ_WRITE , __mdprd_bits);
__IO_REG32_BIT(MTPD1CMPU,            0x400F6118, __READ_WRITE , __cmpu_bits);
__IO_REG32_BIT(MTPD1CMPV,            0x400F611C, __READ_WRITE , __cmpv_bits);
__IO_REG32_BIT(MTPD1CMPW,            0x400F6120, __READ_WRITE , __cmpw_bits);
__IO_REG32_BIT(MTPD1MDOUT,           0x400F6128, __READ_WRITE , __mdout_bits);
__IO_REG32_BIT(MTPD1MDPOT,           0x400F612C, __READ_WRITE , __mdpot_bits);
__IO_REG32_BIT(MTPD1EMGREL,          0x400F6130, __WRITE      , __emgrel_bits);
__IO_REG32_BIT(MTPD1EMGCR,           0x400F6134, __READ_WRITE , __emgcr_bits);
__IO_REG32_BIT(MTPD1EMGSTA,          0x400F6138, __READ       , __emgsta_bits);
__IO_REG32_BIT(MTPD1DTR,             0x400F6144, __READ_WRITE , __dtr_bits);

/***************************************************************************
 **
 ** ENC 0
 **
 ***************************************************************************/
__IO_REG32_BIT(EN0TNCR,             0x400F7000, __READ_WRITE , __enxtncr_bits);
__IO_REG32_BIT(EN0RELOAD,           0x400F7004, __READ_WRITE , __enxreload_bits);
__IO_REG32_BIT(EN0INT,              0x400F7008, __READ_WRITE , __enxint_bits);
__IO_REG32_BIT(EN0CNT,              0x400F700C, __READ_WRITE , __enxcnt_bits);

/***************************************************************************
 **
 ** ENC 1
 **
 ***************************************************************************/
__IO_REG32_BIT(EN1TNCR,             0x400F7100, __READ_WRITE , __enxtncr_bits);
__IO_REG32_BIT(EN1RELOAD,           0x400F7104, __READ_WRITE , __enxreload_bits);
__IO_REG32_BIT(EN1INT,              0x400F7108, __READ_WRITE , __enxint_bits);
__IO_REG32_BIT(EN1CNT,              0x400F710C, __READ_WRITE , __enxcnt_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG8_BIT(RTCSECR,              0x400CC000, __READ_WRITE ,__secr_bits);
__IO_REG8_BIT(RTCMINR,              0x400CC001, __READ_WRITE ,__minr_bits);
__IO_REG8_BIT(RTCHOURR,             0x400CC002, __READ_WRITE ,__hourr_bits);
__IO_REG8_BIT(RTCDAYR,              0x400CC004, __READ_WRITE ,__dayr_bits);
__IO_REG8_BIT(RTCDATER,             0x400CC005, __READ_WRITE ,__dater_bits);
__IO_REG8_BIT(RTCMONTHR,            0x400CC006, __READ_WRITE ,__monthr_bits);
__IO_REG8_BIT(RTCYEARR,             0x400CC007, __READ_WRITE ,__yearr_bits);
__IO_REG8_BIT(RTCPAGER,             0x400CC008, __READ_WRITE ,__pager_bits);
__IO_REG8_BIT(RTCRESTR,             0x400CC00C, __WRITE      ,__restr_bits);

/***************************************************************************
 **
 ** LVD
 **
 ***************************************************************************/
__IO_REG32_BIT(LVDRCR,              0x400F4000, __READ_WRITE , __lvdrcr_bits);
__IO_REG32_BIT(LVDICR,              0x400F4004, __READ_WRITE , __lvdicr_bits);
__IO_REG32_BIT(LVDSR,               0x400F4008, __READ_WRITE , __lvdsr_bits);

/***************************************************************************
 **
 ** OFD
 **
 ***************************************************************************/
__IO_REG32_BIT(OFDCR1,              0x400F1000, __READ_WRITE ,__ofdcr1_bits);
__IO_REG32_BIT(OFDCR2,              0x400F1004, __READ_WRITE ,__ofdcr2_bits);
__IO_REG32_BIT(OFDMN0,              0x400F1008, __READ_WRITE ,__ofdmn_bits);
__IO_REG32_BIT(OFDMN1,              0x400F100C, __READ_WRITE ,__ofdmn_bits);
__IO_REG32_BIT(OFDMX0,              0x400F1010, __READ_WRITE ,__ofdmx_bits);
__IO_REG32_BIT(OFDMX1,              0x400F1014, __READ_WRITE ,__ofdmx_bits);
__IO_REG32_BIT(OFDRST,              0x400F1018, __READ_WRITE ,__ofdrst_bits);
__IO_REG32_BIT(OFDSTAT,             0x400F101C, __READ       ,__ofdstat_bits);
__IO_REG32_BIT(OFDMON,              0x400F1020, __READ_WRITE ,__ofdmon_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,               0x400F2000,__READ_WRITE ,__wdmod_bits);
__IO_REG32_BIT(WDCR,                0x400F2004,__WRITE			,__wdcr_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FCSECBIT,            0x41FFF010, __READ_WRITE , __fcsecbit_bits);
__IO_REG32_BIT(FCFLCS,              0x41FFF020, __READ       , __fcflcs_bits);


/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  TMPM369FxFG DMACA Request Lines
 **
 ***************************************************************************/
#define DMAC_ADC0           0
#define DMAC_ADC1           1
#define DMAC_DAC0           2
#define DMAC_DAC1           3
#define DMAC_SSP0_RX        4
#define DMAC_SSP0_TX        5
#define DMAC_SSP1_RX        6
#define DMAC_SSP1_TX        7
#define DMAC_SSP2_RX        8
#define DMAC_SSP2_TX        9
#define DMAC_UART0_RX       10
#define DMAC_UART0_TX       11
#define DMAC_UART1_RX       12
#define DMAC_UART1_TX       13
#define DMAC_SIO0_RX        14
#define DMAC_SIO0_TX        15
#define DMAC_SIO1_RX        16
#define DMAC_SIO1_TX        17
#define DMAC_SIO2_RX        18
#define DMAC_SIO2_TX        19
#define DMAC_SIO3_RX        20
#define DMAC_SIO3_TX        21
#define DMAC_I2C0           22
#define DMAC_I2C1           23
#define DMAC_I2C2           24
#define DMAC_TMRB0          25
#define DMAC_TMRB1          26
#define DMAC_TMRB2          27
#define DMAC_TMRB3          28
#define DMAC_TMRB4          29
#define DMAC_REQ_PIN        30
#define DMAC_SOFT_TRG       31


/***************************************************************************
 **
 **  TMPM369FxFG Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK             0          /* Main Stack                    */
#define RESETI                 1          /* Reset                         */
#define NMII                   2          /* Non-maskable Interrupt        */
#define HFI                    3          /* Hard Fault                    */
#define MMI                    4          /* Memory Management             */
#define BFI                    5          /* Bus Fault                     */
#define UFI                    6          /* Usage Fault                   */
#define SVCI                  11          /* SVCall                        */
#define DMI                   12          /* Debug Monitor                 */
#define PSI                   14          /* PendSV                        */
#define STI                   15          /* SysTick                       */
#define EII                   16          /* External Interrupt            */
#define INT_0                ( 0 + EII)   /* External Interrupt 0         */
#define INT_1                ( 1 + EII)   /* External Interrupt 1         */
#define INT_2                ( 2 + EII)   /* External Interrupt 2         */
#define INT_3                ( 3 + EII)   /* External Interrupt 3         */
#define INT_4                ( 4 + EII)   /* External Interrupt 4         */
#define INT_5                ( 5 + EII)   /* External Interrupt 5         */
#define INT_6                ( 6 + EII)   /* External Interrupt 6         */
#define INT_7                ( 7 + EII)   /* External Interrupt 7         */
#define INT_8                ( 8 + EII)   /* External Interrupt 8         */
#define INT_9                ( 9 + EII)   /* External Interrupt 9         */
#define INT_A                (10 + EII)   /* External Interrupt A         */
#define INT_B                (11 + EII)   /* External Interrupt B         */
#define INT_C                (12 + EII)   /* External Interrupt C         */
#define INT_USBPON           (13 + EII)   /* USB Power On connection detection interrupt */
#define INT_E                (14 + EII)   /* External Interrupt E         */
#define INT_F                (15 + EII)   /* External Interrupt F         */
#define INT_RX0              (16 + EII)   /* Serial reception (channel.0) */
#define INT_TX0              (17 + EII)   /* Serial transmit (channel.0)  */
#define INT_RX1              (18 + EII)   /* Serial reception (channel.1) */
#define INT_TX1              (19 + EII)   /* Serial transmit (channel.1)  */
#define INT_RX2              (20 + EII)   /* Serial reception (channel.2) */
#define INT_TX2              (21 + EII)   /* Serial transmit (channel.2)  */
#define INT_RX3              (22 + EII)   /* Serial reception (channel.3) */
#define INT_TX3              (23 + EII)   /* Serial transmit (channel.3)  */
#define INT_UART0            (24 + EII)   /* UART interrupt (UART channel.0) */
#define INT_UART1            (25 + EII)   /* UART interrupt (UART channel.1) */
#define INT_SBI0             (26 + EII)   /* Serial bus interface 0       */
#define INT_SBI1             (27 + EII)   /* Serial bus interface 1       */
#define INT_SBI2             (28 + EII)   /* Serial bus interface 2       */
#define INT_SSP0             (29 + EII)   /* SPI serial interface 0       */
#define INT_SSP1             (30 + EII)   /* SPI serial interface 1       */
#define INT_SSP2             (31 + EII)   /* SPI serial interface 2       */
#define INT_USBH             (32 + EII)   /* USB Host Interrupt           */
#define INT_USBD             (33 + EII)   /* USB Device Interrupt         */
#define INT_USBWKUP          (34 + EII)   /* USB device wakeup Interrupt  */
#define INT_CANRX            (35 + EII)   /* CAN reception                */
#define INT_CANTX            (36 + EII)   /* CAN transmission             */
#define INT_CANGB            (37 + EII)   /* CAN status                   */
#define INT_ETH              (38 + EII)   /* Ethernet interrupt           */
#define INT_ETHWK            (39 + EII)   /* Ethernet interrupt           */
#define INT_ADAHP            (40 + EII)   /* Highest priority AD conversion complete interrupt CH-A */
#define INT_ADAM0            (41 + EII)   /* AD conversion monitoring function interrupt 0 CH-A */
#define INT_ADAM1            (42 + EII)   /* AD conversion monitoring function interrupt 1 CH-A */
#define INT_ADA              (43 + EII)   /* AD conversion complete interrupt CH-A */
#define INT_ADBHP            (44 + EII)   /* Highest priority AD conversion complete interrupt CH-B*/
#define INT_ADBM0            (45 + EII)   /* AD conversion monitoring function interrupt 0 CH-B */
#define INT_ADBM1            (46 + EII)   /* AD conversion monitoring function interrupt 1 CH-B */
#define INT_ADB              (47 + EII)   /* AD conversion complete interrupt CH-B */
#define INT_EMG0             (48 + EII)   /* PMD EMG interrupt (ch 0)     */
#define INT_PMD0             (49 + EII)   /* PMD PWM interrupt (ch 0)     */
#define INT_ENC0             (50 + EII)   /* PMD encoder input interrupt (ch 0)   */
#define INT_EMG1             (51 + EII)   /* PMD EMG interrupt (ch 1)     */
#define INT_PMD1             (52 + EII)   /* PMD PWM interrupt (ch 1)     */
#define INT_ENC1             (53 + EII)   /* PMD encoder input interrupt (ch 1)   */
#define INT_MTEMG0           (54 + EII)   /* MPT EMG interrupt (ch 0)     */
#define INT_MTPTB00          (55 + EII)   /* MPT compare match0 interrupt (ch 0)      */
#define INT_MTPTB01          (56 + EII)   /* MPT compare match1 interrupt (ch 0)      */
#define INT_MTCAP00          (57 + EII)   /* MPT input capture0 interrupt (ch 0)      */
#define INT_MTCAP01          (58 + EII)   /* MPT input capture1 interrupt (ch 0)      */
#define INT_MTEMG1           (59 + EII)   /* MPT EMG interrupt (ch 1)     */
#define INT_MTPTB10          (60 + EII)   /* MPT compare match0 interrupt (ch 1)      */
#define INT_MTPTB11          (61 + EII)   /* MPT compare match1 interrupt (ch 1)      */
#define INT_MTCAP10          (62 + EII)   /* MPT input capture0 interrupt (ch 1)      */
#define INT_MTCAP11          (63 + EII)   /* MPT input capture1 interrupt (ch 1)      */
#define INT_MTEMG2           (64 + EII)   /* MPT EMG interrupt (ch 2)     */
#define INT_MTPTB20          (65 + EII)   /* MPT compare match0 interrupt (ch 2)      */
#define INT_MTPTB21          (66 + EII)   /* MPT compare match1 interrupt (ch 2)      */
#define INT_MTCAP20          (67 + EII)   /* MPT input capture0 interrupt (ch 2)      */
#define INT_MTCAP21          (68 + EII)   /* MPT input capture1 interrupt (ch 2)      */
#define INT_MTEMG3           (69 + EII)   /* MPT EMG interrupt (ch 3)     */
#define INT_MTPTB30          (70 + EII)   /* MPT compare match0 interrupt (ch 3)      */
#define INT_MTPTB31          (71 + EII)   /* MPT compare match1 interrupt (ch 3)      */
#define INT_MTCAP30          (72 + EII)   /* MPT input capture0 interrupt (ch 3)      */
#define INT_MTCAP31          (73 + EII)   /* MPT input capture1 interrupt (ch 3)      */
#define INT_RMCRX            (74 + EII)   /* Remocon reception    */
#define INT_TB0              (75 + EII)   /* TMRB compare match0 (ch-0)   */
#define INT_TCAP00           (76 + EII)   /* TMRB input capture0 (ch-0)   */
#define INT_TCAP01           (77 + EII)   /* TMRB input capture1 (ch-0)   */
#define INT_TB1              (78 + EII)   /* TMRB compare match0 (ch-1)   */
#define INT_TCAP10           (79 + EII)   /* TMRB input capture0 (ch-1)   */
#define INT_TCAP11           (80 + EII)   /* TMRB input capture1 (ch-1)   */
#define INT_TB2              (81 + EII)   /* TMRB compare match0 (ch-2)   */
#define INT_TCAP20           (82 + EII)   /* TMRB input capture0 (ch-2)   */
#define INT_TCAP21           (83 + EII)   /* TMRB input capture1 (ch-2)   */
#define INT_TB3              (84 + EII)   /* TMRB compare match0 (ch-3)   */
#define INT_TCAP30           (85 + EII)   /* TMRB input capture0 (ch-3)   */
#define INT_TCAP31           (86 + EII)   /* TMRB input capture1 (ch-3)   */
#define INT_TB4              (87 + EII)   /* TMRB compare match0 (ch-4)   */
#define INT_TCAP40           (88 + EII)   /* TMRB input capture0 (ch-4)   */
#define INT_TCAP41           (89 + EII)   /* TMRB input capture1 (ch-4)   */
#define INT_TB5              (90 + EII)   /* TMRB compare match0 (ch-5)   */
#define INT_TCAP50           (91 + EII)   /* TMRB input capture0 (ch-5)   */
#define INT_TCAP51           (92 + EII)   /* TMRB input capture1 (ch-5)   */
#define INT_TB6              (93 + EII)   /* TMRB compare match0 (ch-6)   */
#define INT_TCAP60           (94 + EII)   /* TMRB input capture0 (ch-6)   */
#define INT_TCAP61           (95 + EII)   /* TMRB input capture1 (ch-6)   */
#define INT_TB7              (96 + EII)   /* TMRB compare match0 (ch-7)   */
#define INT_TCAP70           (97 + EII)   /* TMRB input capture0 (ch-7)   */
#define INT_TCAP71           (98 + EII)   /* TMRB input capture1 (ch-7)   */
#define INT_RTC              (99 + EII)   /* RTC interrupt        */
#define INT_DMAADA           (100 + EII)  /* DMA ADC conversion complete (ch-A)  */
#define INT_DMAADB           (101 + EII)  /* DMA ADC conversion complete (ch-B)  */
#define INT_DMADAA           (102 + EII)  /* DMA ADC conversion trigger (ch-A)  */
#define INT_DMADAB           (103 + EII)  /* DMA ADC conversion trigger (ch-B)  */
#define INT_DMASPT0          (104 + EII)  /* DMA SSP transmission (ch-0)  */
#define INT_DMASPR0          (105 + EII)  /* DMA SSP reception (ch-0)  */
#define INT_DMASPT1          (106 + EII)  /* DMA SSP transmission (ch-1)  */
#define INT_DMASPR1          (107 + EII)  /* DMA SSP reception (ch-1)  */
#define INT_DMASPT2          (108 + EII)  /* DMA SSP transmission (ch-2)  */
#define INT_DMASPR2          (109 + EII)  /* DMA SSP reception (ch-2)  */
#define INT_DMAUTR0          (110 + EII)  /* DMA UART reception (ch-0)  */
#define INT_DMAUTT0          (111 + EII)  /* DMA UART transmission (ch-0)  */
#define INT_DMAUTR1          (112 + EII)  /* DMA UART reception (ch-1)  */
#define INT_DMAUTT1          (113 + EII)  /* DMA UART transmission (ch-1)  */
#define INT_DMARX0           (114 + EII)  /* DMA SIO/UART reception (ch-0)  */
#define INT_DMATX0           (115 + EII)  /* DMA SIO/UART transmission (ch-0)  */
#define INT_DMARX1           (116 + EII)  /* DMA SIO/UART reception (ch-1)  */
#define INT_DMATX1           (117 + EII)  /* DMA SIO/UART transmission (ch-1)  */
#define INT_DMARX2           (118 + EII)  /* DMA SIO/UART reception (ch-2)  */
#define INT_DMATX2           (119 + EII)  /* DMA SIO/UART transmission (ch-2)  */
#define INT_DMARX3           (120 + EII)  /* DMA SIO/UART reception (ch-3)  */
#define INT_DMATX3           (121 + EII)  /* DMA SIO/UART transmission (ch-3)  */
#define INT_I2CI1            (122 + EII)  /* DMA I2C/SIO (ch-1)   */
#define INT_I2CI2            (123 + EII)  /* DMA I2C/SIO (ch-2)   */
#define INT_DMATB            (124 + EII)  /* DMA TMRB compare match (ch 0-4)  */
#define INT_DMARQ            (125 + EII)  /* DMA request pin      */
#define INT_DMAAERR          (126 + EII)  /* DMA transmission error interrupt (ch-A)  */
#define INT_DMABERR          (127 + EII)  /* DMA transmission error interrupt (ch-B)  */

#endif    /* __IOTMPM369FxFG_H */

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
Interrupt17  = INT8           0x60
Interrupt18  = INT9           0x64
Interrupt19  = INTA           0x68
Interrupt20  = INTB           0x6C
Interrupt21  = INTC           0x70
Interrupt22  = INTUSBPON      0x70
Interrupt23  = INTE           0x78
Interrupt24  = INTF           0x7C
Interrupt25  = INTRX0         0x80
Interrupt26  = INTTX0         0x84
Interrupt27  = INTRX1         0x88
Interrupt28  = INTTX1         0x8C
Interrupt29  = INTRX2         0x90
Interrupt30  = INTTX2         0x94
Interrupt31  = INTRX3         0x98
Interrupt32  = INTTX3         0x9C
Interrupt33  = INTUART0       0xA0
Interrupt34  = INTUART1       0xA4
Interrupt35  = INTSBI0        0xA8
Interrupt36  = INTSBI1        0xAC
Interrupt37  = INTSBI2        0xB0
Interrupt38  = INTSSP0        0xB4
Interrupt39  = INTSSP1        0xB8
Interrupt40  = INTSSP2        0xBC
Interrupt41  = INTUSBH        0xC0
Interrupt42  = INTUSBD        0xC4
Interrupt43  = INTUSBWKUP     0xC8
Interrupt44  = INTCANRX       0xCC
Interrupt45  = INTCANTX       0xD0
Interrupt46  = INTCANGB       0xD4
Interrupt47  = INTETH         0xD8
Interrupt48  = INTETHWK       0xDC
Interrupt49  = INTADAHP       0xE0
Interrupt50  = INTADAM0       0xE4
Interrupt51  = INTADAM1       0xE8
Interrupt52  = INTADA         0xEC
Interrupt53  = INTADBHP       0xF0
Interrupt54  = INTADBM0       0xF4
Interrupt55  = INTADBM1       0xF8
Interrupt56  = INTADB         0xFC
Interrupt57  = INTEMG0        0x100
Interrupt58  = INTPMD0        0x104
Interrupt59  = INTENC0        0x108
Interrupt60  = INTEMG1        0x10C
Interrupt61  = INTPMD1        0x110
Interrupt62  = INTENC1        0x114
Interrupt63  = INTMTEMG0      0x118
Interrupt64  = INTMTPTB00     0x11C
Interrupt65  = INTMTPTB01     0x120
Interrupt66  = INTMTCAP00     0x124
Interrupt67  = INTMTCAP01     0x128
Interrupt68  = INTMTEMG1      0x12C
Interrupt69  = INTMTPTB10     0x130
Interrupt70  = INTMTPTB11     0x134
Interrupt71  = INTMTCAP10     0x138
Interrupt72  = INTMTCAP11     0x13C
Interrupt73  = INTMTEMG2      0x140
Interrupt74  = INTMTPTB20     0x144
Interrupt75  = INTMTPTB21     0x148
Interrupt76  = INTMTCAP20     0x14C
Interrupt77  = INTMTCAP21     0x150
Interrupt78  = INTMTEMG3      0x154
Interrupt79  = INTMTPTB30     0x158
Interrupt80  = INTMTPTB31     0x15C
Interrupt81  = INTMTCAP30     0x160
Interrupt82  = INTMTCAP31     0x164
Interrupt83  = INTRMCRX       0x168
Interrupt84  = INTTB0         0x16C
Interrupt85  = INTTCAP00      0x170
Interrupt86  = INTTCAP01      0x174
Interrupt87  = INTTB1         0x178
Interrupt88  = INTTCAP10      0x17C
Interrupt89  = INTTCAP11      0x180
Interrupt90  = INTTB2         0x184
Interrupt91  = INTTCAP20      0x188
Interrupt92  = INTTCAP21      0x18C
Interrupt93  = INTTB3         0x190
Interrupt94  = INTTCAP30      0x194
Interrupt95  = INTTCAP31      0x198
Interrupt96  = INTTB4         0x19C
Interrupt97  = INTTCAP40      0x1A0
Interrupt98  = INTTCAP41      0x1A4
Interrupt99  = INTTB5         0x1A8
Interrupt100  = INTTCAP50     0x1AC
Interrupt101  = INTTCAP51     0x1B0
Interrupt102  = INTTB6        0x1B4
Interrupt103  = INTTCAP60     0x1B8
Interrupt104  = INTTCAP61     0x1BC
Interrupt105  = INTTB7        0x1C0
Interrupt106  = INTTCAP70     0x1C4
Interrupt107  = INTTCAP71     0x1C8
Interrupt108  = INTRTC        0x1CC
Interrupt109  = INTDMAADA     0x1D0
Interrupt110  = INTDMAADB     0x1D4
Interrupt111  = INTDMADAA     0x1D8
Interrupt112  = INTDMADAB     0x1DC
Interrupt113  = INTDMASPT0    0x1E0
Interrupt114  = INTDMASPR0    0x1E4
Interrupt115  = INTDMASPT1    0x1E8
Interrupt116  = INTDMASPR1    0x1EC
Interrupt117  = INTDMASPT2    0x1F0
Interrupt118  = INTDMASPR2    0x1F4
Interrupt119  = INTDMAUTR0    0x1F8
Interrupt120  = INTDMAUTT0    0x1FC
Interrupt121  = INTDMAUTR1    0x200
Interrupt122  = INTDMAUTT1    0x204
Interrupt123  = INTDMARX0     0x208
Interrupt124  = INTDMATX0     0x20C
Interrupt125  = INTDMARX1     0x210
Interrupt126  = INTDMATX1     0x214
Interrupt127  = INTDMARX2     0x218
Interrupt128  = INTDMATX2     0x21C
Interrupt129  = INTDMARX3     0x220
Interrupt130  = INTDMATX3     0x224
Interrupt131  = INTSBI1       0x228
Interrupt132  = INTSBI2       0x22C
Interrupt133  = INTDMATB      0x230
Interrupt134  = INTDMARQ      0x234
Interrupt135  = INTDMAAERR    0x238
Interrupt136  = INTDMABERR    0x23C

###DDF-INTERRUPT-END###*/
