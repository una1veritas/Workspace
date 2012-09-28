/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Infineon xmc4500 Devices
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50748 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOXMC4500_H
#define __IOXMC4500_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    XMC4500 SPECIAL FUNCTION REGISTERS
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

/* Auxiliary Control Register */
typedef struct {
  __REG32  DISMCYCINT     : 1;
  __REG32  DISDEFWBUF     : 1;
  __REG32  DISFOLD        : 1;
  __REG32                 : 5;
  __REG32  DISFPCA        : 1;
  __REG32  DISOOFP        : 1;
  __REG32                 :22;
} __actlr_bits;

/* CPU ID Base Register */
typedef struct {
  __REG32  Revision       : 4;
  __REG32  PartNo         :12;
  __REG32                 : 4;
  __REG32  Variant        : 4;
  __REG32  Implementer    : 8;
} __cpuid_bits;

/* Interrupt Control and State Register */
typedef struct {
  __REG32  VECTACTIVE     : 9;
  __REG32                 : 2;
  __REG32  RETTOBASE      : 1;
  __REG32  VECTPENDING    : 6;
  __REG32                 : 4;
  __REG32  ISRPENDING     : 1;
  __REG32                 : 2;
  __REG32  PENDSTCLR      : 1;
  __REG32  PENDSTSET      : 1;
  __REG32  PENDSVCLR      : 1;
  __REG32  PENDSVSET      : 1;
  __REG32                 : 2;
  __REG32  NMIPENDSET     : 1;
} __icsr_bits;

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

/* System Handler Priority Register 1 */
typedef struct {
  __REG32  PRI_4          : 8;
  __REG32  PRI_5          : 8;
  __REG32  PRI_6          : 8;
  __REG32                 : 8;
} __shpr1_bits;

/* System Handler Priority Register 2 */
typedef struct {
  __REG32                 :24;
  __REG32  PRI_11         : 8;
} __shpr2_bits;

/* System Handler Priority Register 3 */
typedef struct {
  __REG32                 :16;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __shpr3_bits;

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
  __REG32  FAULTPENDED    : 1;
  __REG32  MEMFAULTPENDED : 1;
  __REG32  BUSFAULTPENDED : 1;
  __REG32  SVCALLPENDED   : 1;
  __REG32  MEMFAULTENA    : 1;
  __REG32  BUSFAULTENA    : 1;
  __REG32  USGFAULTENA    : 1;
  __REG32                 :13;
} __shcsr_bits;

/* HardFault Status Register */
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

/* SysTick Control and Status Register  */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  TICKINT        : 1;
  __REG32  CLKSOURCE      : 1;
  __REG32                 :13;
  __REG32  COUNTFLAG      : 1;
  __REG32                 :15;
} __syst_csr_bits;

/* SysTick Reload Value Register */
typedef struct {
  __REG32  RELOAD         :24;
  __REG32                 : 8;
} __syst_rvr_bits;

/* SysTick Current Value Register */
typedef struct {
  __REG32  CURRENT        :24;
  __REG32                 : 8;
} __syst_cvr_bits;

/* SysTick Calibration Value Registe */
typedef struct {
  __REG32  TENMS          :24;
  __REG32                 : 6;
  __REG32  SKEW           : 1;
  __REG32  NOREF          : 1;
} __syst_calib_bits;

/* Interrupt Set-enable Register 0-31 */
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
} __nvic_iser0_bits;

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
} __nvic_iser1_bits;

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
} __nvic_iser2_bits;

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
} __nvic_iser3_bits;

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
} __nvic_icer0_bits;

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
} __nvic_icer1_bits;

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
} __nvic_icer2_bits;

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
} __nvic_icer3_bits;

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
} __nvic_ispr0_bits;

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
} __nvic_ispr1_bits;

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
} __nvic_ispr2_bits;

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
} __nvic_ispr3_bits;

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
} __nvic_icpr0_bits;

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
} __nvic_icpr1_bits;

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
} __nvic_icpr2_bits;

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
} __nvic_icpr3_bits;

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
} __nvic_iabr0_bits;

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
} __nvic_iabr1_bits;

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
} __nvic_iabr2_bits;

/* Active Bit Register 96-127 */
typedef struct {
  __REG32  ACTIVE96       : 1;
  __REG32  ACTIVE97       : 1;
  __REG32  ACTIVE98       : 1;
  __REG32  ACTIVE99       : 1;
  __REG32  ACTIVE100      : 1;
  __REG32  ACTIVE101      : 1;
  __REG32  ACTIVE102      : 1;
  __REG32  ACTIVE103      : 1;
  __REG32  ACTIVE104      : 1;
  __REG32  ACTIVE105      : 1;
  __REG32  ACTIVE106      : 1;
  __REG32  ACTIVE107      : 1;
  __REG32  ACTIVE108      : 1;
  __REG32  ACTIVE109      : 1;
  __REG32  ACTIVE110      : 1;
  __REG32  ACTIVE111      : 1;
  __REG32  ACTIVE112      : 1;
  __REG32  ACTIVE113      : 1;
  __REG32  ACTIVE114      : 1;
  __REG32  ACTIVE115      : 1;
  __REG32  ACTIVE116      : 1;
  __REG32  ACTIVE117      : 1;
  __REG32  ACTIVE118      : 1;
  __REG32  ACTIVE119      : 1;
  __REG32  ACTIVE120      : 1;
  __REG32  ACTIVE121      : 1;
  __REG32  ACTIVE122      : 1;
  __REG32  ACTIVE123      : 1;
  __REG32  ACTIVE124      : 1;
  __REG32  ACTIVE125      : 1;
  __REG32  ACTIVE126      : 1;
  __REG32  ACTIVE127      : 1;
} __nvic_iabr3_bits;

/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32  PRI_0          : 8;
  __REG32  PRI_1          : 8;
  __REG32  PRI_2          : 8;
  __REG32  PRI_3          : 8;
} __nvic_ipr0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32  PRI_4          : 8;
  __REG32  PRI_5          : 8;
  __REG32  PRI_6          : 8;
  __REG32  PRI_7          : 8;
} __nvic_ipr1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32  PRI_8          : 8;
  __REG32  PRI_9          : 8;
  __REG32  PRI_10         : 8;
  __REG32  PRI_11         : 8;
} __nvic_ipr2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32  PRI_12         : 8;
  __REG32  PRI_13         : 8;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __nvic_ipr3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32  PRI_16         : 8;
  __REG32  PRI_17         : 8;
  __REG32  PRI_18         : 8;
  __REG32  PRI_19         : 8;
} __nvic_ipr4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32  PRI_20         : 8;
  __REG32  PRI_21         : 8;
  __REG32  PRI_22         : 8;
  __REG32  PRI_23         : 8;
} __nvic_ipr5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32  PRI_24         : 8;
  __REG32  PRI_25         : 8;
  __REG32  PRI_26         : 8;
  __REG32  PRI_27         : 8;
} __nvic_ipr6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32  PRI_28         : 8;
  __REG32  PRI_29         : 8;
  __REG32  PRI_30         : 8;
  __REG32  PRI_31         : 8;
} __nvic_ipr7_bits;

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32  PRI_32         : 8;
  __REG32  PRI_33         : 8;
  __REG32  PRI_34         : 8;
  __REG32  PRI_35         : 8;
} __nvic_ipr8_bits;

/* Interrupt Priority Registers 36-39 */
typedef struct {
  __REG32  PRI_36         : 8;
  __REG32  PRI_37         : 8;
  __REG32  PRI_38         : 8;
  __REG32  PRI_39         : 8;
} __nvic_ipr9_bits;

/* Interrupt Priority Registers 40-43 */
typedef struct {
  __REG32  PRI_40         : 8;
  __REG32  PRI_41         : 8;
  __REG32  PRI_42         : 8;
  __REG32  PRI_43         : 8;
} __nvic_ipr10_bits;

/* Interrupt Priority Registers 44-47 */
typedef struct {
  __REG32  PRI_44         : 8;
  __REG32  PRI_45         : 8;
  __REG32  PRI_46         : 8;
  __REG32  PRI_47         : 8;
} __nvic_ipr11_bits;

/* Interrupt Priority Registers 48-51 */
typedef struct {
  __REG32  PRI_48         : 8;
  __REG32  PRI_49         : 8;
  __REG32  PRI_50         : 8;
  __REG32  PRI_51         : 8;
} __nvic_ipr12_bits;

/* Interrupt Priority Registers 52-55 */
typedef struct {
  __REG32  PRI_52         : 8;
  __REG32  PRI_53         : 8;
  __REG32  PRI_54         : 8;
  __REG32  PRI_55         : 8;
} __nvic_ipr13_bits;

/* Interrupt Priority Registers 56-59 */
typedef struct {
  __REG32  PRI_56         : 8;
  __REG32  PRI_57         : 8;
  __REG32  PRI_58         : 8;
  __REG32  PRI_59         : 8;
} __nvic_ipr14_bits;

/* Interrupt Priority Registers 60-63 */
typedef struct {
  __REG32  PRI_60         : 8;
  __REG32  PRI_61         : 8;
  __REG32  PRI_62         : 8;
  __REG32  PRI_63         : 8;
} __nvic_ipr15_bits;

/* Interrupt Priority Registers 64-67 */
typedef struct {
  __REG32  PRI_64         : 8;
  __REG32  PRI_65         : 8;
  __REG32  PRI_66         : 8;
  __REG32  PRI_67         : 8;
} __nvic_ipr16_bits;

/* Interrupt Priority Registers 68-71 */
typedef struct {
  __REG32  PRI_68         : 8;
  __REG32  PRI_69         : 8;
  __REG32  PRI_70         : 8;
  __REG32  PRI_71         : 8;
} __nvic_ipr17_bits;

/* Interrupt Priority Registers 72-75 */
typedef struct {
  __REG32  PRI_72         : 8;
  __REG32  PRI_73         : 8;
  __REG32  PRI_74         : 8;
  __REG32  PRI_75         : 8;
} __nvic_ipr18_bits;

/* Interrupt Priority Registers 76-79 */
typedef struct {
  __REG32  PRI_76         : 8;
  __REG32  PRI_77         : 8;
  __REG32  PRI_78         : 8;
  __REG32  PRI_79         : 8;
} __nvic_ipr19_bits;

/* Interrupt Priority Registers 80-83 */
typedef struct {
  __REG32  PRI_80         : 8;
  __REG32  PRI_81         : 8;
  __REG32  PRI_82         : 8;
  __REG32  PRI_83         : 8;
} __nvic_ipr20_bits;

/* Interrupt Priority Registers 84-87 */
typedef struct {
  __REG32  PRI_84         : 8;
  __REG32  PRI_85         : 8;
  __REG32  PRI_86         : 8;
  __REG32  PRI_87         : 8;
} __nvic_ipr21_bits;

/* Interrupt Priority Registers 88-91 */
typedef struct {
  __REG32  PRI_88         : 8;
  __REG32  PRI_89         : 8;
  __REG32  PRI_90         : 8;
  __REG32  PRI_91         : 8;
} __nvic_ipr22_bits;

/* Interrupt Priority Registers 92-95 */
typedef struct {
  __REG32  PRI_92         : 8;
  __REG32  PRI_93         : 8;
  __REG32  PRI_94         : 8;
  __REG32  PRI_95         : 8;
} __nvic_ipr23_bits;

/* Interrupt Priority Registers 96-99 */
typedef struct {
  __REG32  PRI_96         : 8;
  __REG32  PRI_97         : 8;
  __REG32  PRI_98         : 8;
  __REG32  PRI_99         : 8;
} __nvic_ipr24_bits;

/* Interrupt Priority Registers 100-103 */
typedef struct {
  __REG32  PRI_100        : 8;
  __REG32  PRI_101        : 8;
  __REG32  PRI_102        : 8;
  __REG32  PRI_103        : 8;
} __nvic_ipr25_bits;

/* Interrupt Priority Registers 104-107 */
typedef struct {
  __REG32  PRI_104        : 8;
  __REG32  PRI_105        : 8;
  __REG32  PRI_106        : 8;
  __REG32  PRI_107        : 8;
} __nvic_ipr26_bits;

/* Interrupt Priority Registers 108-111 */
typedef struct {
  __REG32  PRI_108        : 8;
  __REG32  PRI_109        : 8;
  __REG32  PRI_110        : 8;
  __REG32  PRI_111        : 8;
} __nvic_ipr27_bits;

/* Software Trigger Interrupt Register */
typedef struct {
  __REG32  INTID          : 9;
  __REG32                 :23;
} __stir_bits;

/* MPU Type Register */
typedef struct {
  __REG32  SEPARATE       : 1;
  __REG32                 : 7;
  __REG32  DREGION        : 8;
  __REG32  IREGION        : 8;
  __REG32                 : 8;
} __mpu_type_bits;

/* MPU Controlr Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  HFNMIENA       : 1;
  __REG32  PRIVDEFENA     : 1;
  __REG32                 :29;
} __mpu_ctrl_bits;

/* MPU Region Number Register */
typedef struct {
  __REG32  REGION         : 8;
  __REG32                 :24;
} __mpu_rnr_bits;

/* MPU Region Base Address Register */
typedef struct {
  __REG32  REGION         : 4;
  __REG32  VALID          : 1;
  __REG32                 : 4;
  __REG32  ADDR           :23;
} __mpu_rbar_bits;

/* MPU Region Attribute and Size Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  SIZE           : 5;
  __REG32                 : 2;
  __REG32  SRD            : 8;
  __REG32  B              : 1;
  __REG32  C              : 1;
  __REG32  S              : 1;
  __REG32  TEX            : 3;
  __REG32                 : 2;
  __REG32  AP             : 3;
  __REG32                 : 1;
  __REG32  XN             : 1;
  __REG32                 : 3;
} __mpu_rasr_bits;

/* Coprocessor Access Control Register */
typedef struct {
  __REG32                 :20;
  __REG32  CP10           : 2;
  __REG32  CP11           : 2;
  __REG32                 : 8;
} __fpu_cpacr_bits;

/* Floating-point Context Control Register */
typedef struct {
  __REG32  LSPACT         : 1;
  __REG32  USER           : 1;
  __REG32                 : 1;
  __REG32  THREAD         : 1;
  __REG32  HFRDY          : 1;
  __REG32  MMRDY          : 1;
  __REG32  BFRDY          : 1;
  __REG32                 : 1;
  __REG32  MONRDY         : 1;
  __REG32                 :21;
  __REG32  LSPEN          : 1;
  __REG32  ASPEN          : 1;
} __fpu_fpccr_bits;

/* Floating-point Default Status Control Register */
typedef struct {
  __REG32                 :22;
  __REG32  RMode          : 2;
  __REG32  FZ             : 1;
  __REG32  DN             : 1;
  __REG32  AHP            : 1;
  __REG32                 : 5;
} __fpu_fpdscr_bits;

/* Peripheral Bridge Status Register */
typedef struct {
  __REG32  WERR           : 1;
  __REG32                 :31;
} __pba_sts_bits;

/* PMU0 Identification Register */
typedef struct {
  __REG32  MOD_REV        : 8;
  __REG32  MOD_TYPE       : 8;
  __REG32  MOD_NUMBER     :16;
} __pmu_id_bits;

/* Flash Module Identification Register */
typedef struct {
  __REG32  MOD_REV        : 8;
  __REG32  MOD_TYPE       : 8;
  __REG32  MOD_NUMBER     :16;
} __flash_id_bits;

/* Flash Status Register */
typedef struct {
  __REG32  PBUSY          : 1;
  __REG32  FABUSY         : 1;
  __REG32                 : 2;
  __REG32  PROG           : 1;
  __REG32  ERASE          : 1;
  __REG32  PFPAGE         : 1;
  __REG32                 : 1;
  __REG32  PFOPER         : 1;
  __REG32                 : 1;
  __REG32  SQER           : 1;
  __REG32  PROER          : 1;
  __REG32  PFSBER         : 1;
  __REG32                 : 1;
  __REG32  PFDBER         : 1;
  __REG32                 : 1;
  __REG32  PROIN          : 1;
  __REG32                 : 1;
  __REG32  RPROIN         : 1;
  __REG32  RPRODIS        : 1;
  __REG32                 : 1;
  __REG32  WPROIN0        : 1;
  __REG32  WPROIN1        : 1;
  __REG32  WPROIN2        : 1;
  __REG32                 : 1;
  __REG32  WPRODIS0       : 1;
  __REG32  WPRODIS1       : 1;
  __REG32                 : 1;
  __REG32  SLM            : 1;
  __REG32                 : 1;
  __REG32  X              : 1;
  __REG32  VER            : 1;
} __flash_fsr_bits;

/* Flash Configuration Register */
typedef struct {
  __REG32  WSPFLASH       : 4;
  __REG32  WSECPF         : 1;
  __REG32                 : 8;
  __REG32  IDLE           : 1;
  __REG32  ESLDIS         : 1;
  __REG32  SLEEP          : 1;
  __REG32  RPA            : 1;
  __REG32  DCF            : 1;
  __REG32  DDF            : 1;
  __REG32                 : 5;
  __REG32  VOPERM         : 1;
  __REG32  SQERM          : 1;
  __REG32  PROERM         : 1;
  __REG32  PFSBERM        : 1;
  __REG32                 : 1;
  __REG32  PFDBERM        : 1;
  __REG32                 : 1;
  __REG32  EOBM           : 1;
} __flash_fcon_bits;

/* Margin Control Register */
typedef struct {
  __REG32  MARGIN         : 4;
  __REG32                 :11;
  __REG32  TRAPDIS        : 1;
  __REG32                 :16;
} __flash_marp_bits;

/* Margin Control Register 0 */
typedef struct {
  __REG32  S0L            : 1;
  __REG32  S1L            : 1;
  __REG32  S2L            : 1;
  __REG32  S3L            : 1;
  __REG32  S4L            : 1;
  __REG32  S5L            : 1;
  __REG32  S6L            : 1;
  __REG32  S7L            : 1;
  __REG32  S8L            : 1;
  __REG32  S9L            : 1;
  __REG32  S10_S11L       : 1;
  __REG32                 : 4;
  __REG32  RPRO           : 1;
  __REG32                 :16;
} __flash_procon0_bits;

/* Margin Control Register 1 */
typedef struct {
  __REG32  S0L            : 1;
  __REG32  S1L            : 1;
  __REG32  S2L            : 1;
  __REG32  S3L            : 1;
  __REG32  S4L            : 1;
  __REG32  S5L            : 1;
  __REG32  S6L            : 1;
  __REG32  S7L            : 1;
  __REG32  S8L            : 1;
  __REG32  S9L            : 1;
  __REG32  S10_S11L       : 1;
  __REG32                 :21;
} __flash_procon1_bits;

/* Margin Control Register 2 */
typedef struct {
  __REG32  S0ROM          : 1;
  __REG32  S1ROM          : 1;
  __REG32  S2ROM          : 1;
  __REG32  S3ROM          : 1;
  __REG32  S4ROM          : 1;
  __REG32  S5ROM          : 1;
  __REG32  S6ROM          : 1;
  __REG32  S7ROM          : 1;
  __REG32  S8ROM          : 1;
  __REG32  S9ROM          : 1;
  __REG32  S10_S11ROM     : 1;
  __REG32                 :21;
} __flash_procon2_bits;

/* Prefetch Configuration Register */
typedef struct {
  __REG32  IBYP           : 1;
  __REG32  IINV           : 1;
  __REG32                 :14;
  __REG32  PBS            : 1;
  __REG32                 :15;
} __pcon_bits;

/* GPDMA Configuration Register */
typedef struct {
  __REG32  DMA_EN         : 1;
  __REG32                 :31;
} __gpdma_dmacfgreg_bits;

/* GPDMA0 Channel Enable Register */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32  CH4            : 1;
  __REG32  CH5            : 1;
  __REG32  CH6            : 1;
  __REG32  CH7            : 1;
  __REG32  WE_CH0         : 1;
  __REG32  WE_CH1         : 1;
  __REG32  WE_CH2         : 1;
  __REG32  WE_CH3         : 1;
  __REG32  WE_CH4         : 1;
  __REG32  WE_CH5         : 1;
  __REG32  WE_CH6         : 1;
  __REG32  WE_CH7         : 1;
  __REG32                 :16;
} __gpdma0_chenreg_bits;

/* GPDMA1 Channel Enable Register */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32                 : 4;
  __REG32  WE_CH0         : 1;
  __REG32  WE_CH1         : 1;
  __REG32  WE_CH2         : 1;
  __REG32  WE_CH3         : 1;
  __REG32                 :20;
} __gpdma1_chenreg_bits;

/* Control Register High for Channel */
typedef struct {
  __REG32  BLOCK_TS       :12;
  __REG32  DONE           : 1;
  __REG32                 :19;
} __gpdma_ctlh_bits;

/* Control Register Low for GPDMA0 Channel 0 - 1 */
typedef struct {
  __REG32  INT_EN         : 1;
  __REG32  DST_TR_WIDTH   : 3;
  __REG32  SRC_TR_WIDTH   : 3;
  __REG32  DINC           : 2;
  __REG32  SINC           : 2;
  __REG32  DEST_MSIZE     : 3;
  __REG32  SRC_MSIZE      : 3;
  __REG32  SRC_GATHER_EN  : 1;
  __REG32  DST_SCATTER_EN : 1;
  __REG32                 : 1;
  __REG32  TT_FC          : 3;
  __REG32                 : 4;
  __REG32  LLP_DST_EN     : 1;
  __REG32  LLP_SRC_EN     : 1;
  __REG32                 : 3;
} __gpdma_ctll0_bits;

/* Control Register Low for GPDMA0 Channel 2 - 7, GPDMA1 0 - 3 */
typedef struct {
  __REG32  INT_EN         : 1;
  __REG32  DST_TR_WIDTH   : 3;
  __REG32  SRC_TR_WIDTH   : 3;
  __REG32  DINC           : 2;
  __REG32  SINC           : 2;
  __REG32  DEST_MSIZE     : 3;
  __REG32  SRC_MSIZE      : 3;
  __REG32                 : 3;
  __REG32  TT_FC          : 3;
  __REG32                 : 9;
} __gpdma_ctll2_bits;

/* Configuration Register High for GPDMA0 Channel 0 - 1 */
typedef struct {
  __REG32  FCMODE         : 1;
  __REG32  FIFO_MODE      : 1;
  __REG32  PROTCTL        : 3;
  __REG32  DS_UPD_EN      : 1;
  __REG32  SS_UPD_EN      : 1;
  __REG32  SRC_PER        : 4;
  __REG32  DEST_PER       : 4;
  __REG32                 :17;
} __gpdma_cfgh0_bits;

/* Configuration Register High for GPDMA0 Channel 2 - 7, GPDMA1 0 - 3 */
typedef struct {
  __REG32  FCMODE         : 1;
  __REG32  FIFO_MODE      : 1;
  __REG32  PROTCTL        : 3;
  __REG32                 : 2;
  __REG32  SRC_PER        : 4;
  __REG32  DEST_PER       : 4;
  __REG32                 :17;
} __gpdma_cfgh2_bits;

/* Configuration Register Low for GPDMA0 Channel 0 - 1 */
typedef struct {
  __REG32                 : 5;
  __REG32  CH_PRIOR       : 3;
  __REG32  CH_SUSP        : 1;
  __REG32  FIFO_EMPTY     : 1;
  __REG32  HS_SEL_DST     : 1;
  __REG32  HS_SEL_SRC     : 1;
  __REG32  LOCK_CH_L      : 2;
  __REG32  LOCK_B_L       : 2;
  __REG32  LOCK_CH        : 1;
  __REG32  LOCK_B         : 1;
  __REG32  DST_HS_POL     : 1;
  __REG32  SRC_HS_POL     : 1;
  __REG32  MAX_ABRST      :10;
  __REG32  RELOAD_SRC     : 1;
  __REG32  RELOAD_DST     : 1;
} __gpdma_cfgl0_bits;

/* Configuration Register Low for GPDMA0 Channel 2 - 7, GPDMA1 0 - 3 */
typedef struct {
  __REG32                 : 5;
  __REG32  CH_PRIOR       : 3;
  __REG32  CH_SUSP        : 1;
  __REG32  FIFO_EMPTY     : 1;
  __REG32  HS_SEL_DST     : 1;
  __REG32  HS_SEL_SRC     : 1;
  __REG32  LOCK_CH_L      : 2;
  __REG32  LOCK_B_L       : 2;
  __REG32  LOCK_CH        : 1;
  __REG32  LOCK_B         : 1;
  __REG32  DST_HS_POL     : 1;
  __REG32  SRC_HS_POL     : 1;
  __REG32  MAX_ABRST      :10;
  __REG32                 : 2;
} __gpdma_cfgl2_bits;

/* Source Gather Register for Channel */
typedef struct {
  __REG32  SGI            :20;
  __REG32  SGC            :12;
} __gpdma_sgr_bits;

/* Destination Scatter Register for Channel */
typedef struct {
  __REG32  DSI            :20;
  __REG32  DSC            :12;
} __gpdma_dsr_bits;

/* Interrupt Raw Status Registers */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32  CH4            : 1;
  __REG32  CH5            : 1;
  __REG32  CH6            : 1;
  __REG32  CH7            : 1;
  __REG32                 :24;
} __gpdma0_rawtfr_bits;

/* Interrupt Raw Status Registers */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32                 :28;
} __gpdma1_rawtfr_bits;

/* Interrupt Mask Registers */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32  CH4            : 1;
  __REG32  CH5            : 1;
  __REG32  CH6            : 1;
  __REG32  CH7            : 1;
  __REG32  WE_CH0         : 1;
  __REG32  WE_CH1         : 1;
  __REG32  WE_CH2         : 1;
  __REG32  WE_CH3         : 1;
  __REG32  WE_CH4         : 1;
  __REG32  WE_CH5         : 1;
  __REG32  WE_CH6         : 1;
  __REG32  WE_CH7         : 1;
  __REG32                 :16;
} __gpdma0_masktfr_bits;

/* Interrupt Mask Registers */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32                 : 4;
  __REG32  WE_CH0         : 1;
  __REG32  WE_CH1         : 1;
  __REG32  WE_CH2         : 1;
  __REG32  WE_CH3         : 1;
  __REG32                 :20;
} __gpdma1_masktfr_bits;

/* Interrupt Mask Registers */
typedef struct {
  __REG32  TFR            : 1;
  __REG32  BLOCK          : 1;
  __REG32  SRCT           : 1;
  __REG32  DSTT           : 1;
  __REG32  ERR            : 1;
  __REG32                 :27;
} __gpdma_statusint_bits;

/* Clock Control Register */
typedef struct {
  __REG32  DISR           : 1;
  __REG32  DISS           : 1;
  __REG32                 :30;
} __fce_clc_bits;

/* Module Identification Register */
typedef struct {
  __REG32  MOD_REV        : 8;
  __REG32  MOD_TYPE       : 8;
  __REG32  MOD_NUMBER     :16;
} __fce_id_bits;

/* Service Request Control Register */
typedef struct {
  __REG32  SRPN           : 8;
  __REG32                 : 2;
  __REG32  TOS            : 1;
  __REG32                 : 1;
  __REG32  SRE            : 1;
  __REG32  SRR            : 1;
  __REG32  CLRR           : 1;
  __REG32  SETR           : 1;
  __REG32                 :16;
} __fce_src_bits;

/* CRC Engine Input Register 2 */
typedef struct {
  __REG32  IR             :16;
  __REG32                 :16;
} __fce_ir2_bits;

/* CRC Engine Input Register 3 */
typedef struct {
  __REG32  IR             : 8;
  __REG32                 :24;
} __fce_ir3_bits;

/* CRC Engine Result Register 2 */
typedef struct {
  __REG32  RES            :16;
  __REG32                 :16;
} __fce_res2_bits;

/* CRC Engine Result Register 3 */
typedef struct {
  __REG32  RES            : 8;
  __REG32                 :24;
} __fce_res3_bits;

/* CRC Engine Configuration Register */
typedef struct {
  __REG32  CMI            : 1;
  __REG32  CEI            : 1;
  __REG32  LEI            : 1;
  __REG32  BEI            : 1;
  __REG32  CCE            : 1;
  __REG32  ALR            : 1;
  __REG32                 : 2;
  __REG32  REFIN          : 1;
  __REG32  REFOUT         : 1;
  __REG32  XSEL           : 1;
  __REG32                 :21;
} __fce_cfg_bits;

/* CRC Engine Status Register */
typedef struct {
  __REG32  CMF            : 1;
  __REG32  CEF            : 1;
  __REG32  LEF            : 1;
  __REG32  BEF            : 1;
  __REG32                 :28;
} __fce_sts_bits;

/* CRC Engine Length Register */
typedef struct {
  __REG32  LENGTH         :16;
  __REG32                 :16;
} __fce_length_bits;

/* CRC Engine Check Register 2 */
typedef struct {
  __REG32  CHECK          :16;
  __REG32                 :16;
} __fce_check2_bits;

/* CRC Engine Check Register 3 */
typedef struct {
  __REG32  CHECK          : 8;
  __REG32                 :24;
} __fce_check3_bits;

/* CRC Engine Initialization Register 2 */
typedef struct {
  __REG32  CRC            :16;
  __REG32                 :16;
} __fce_crc2_bits;

/* CRC Engine Initialization Register 3 */
typedef struct {
  __REG32  CRC            : 8;
  __REG32                 :24;
} __fce_crc3_bits;

/* CRC Test Register */
typedef struct {
  __REG32  FCM            : 1;
  __REG32  FRM_CFG        : 1;
  __REG32  FRM_CHECK      : 1;
  __REG32                 :29;
} __fce_ctr_bits;

/* WDT Control Register */
typedef struct {
  __REG32  ENB            : 1;
  __REG32  PRE            : 1;
  __REG32                 : 2;
  __REG32  DSP            : 1;
  __REG32                 : 3;
  __REG32  SPW            : 8;
  __REG32                 :16;
} __wdt_ctr_bits;

/* WDT Status Register */
typedef struct {
  __REG32  ALMS           : 1;
  __REG32                 :31;
} __wdt_sts_bits;

/* WDT Clear Register */
typedef struct {
  __REG32  ALMC           : 1;
  __REG32                 :31;
} __wdt_clr_bits;

/* RTC Control Register */
typedef struct {
  __REG32  ENB            : 1;
  __REG32                 : 1;
  __REG32  TAE            : 1;
  __REG32                 : 5;
  __REG32  ESEC           : 1;
  __REG32  EMIC           : 1;
  __REG32  EHOC           : 1;
  __REG32  EDAC           : 1;
  __REG32  EDAWEC         : 1;
  __REG32  EMOC           : 1;
  __REG32  EYEC           : 1;
  __REG32                 : 1;
  __REG32  DIV            :16;
} __rtc_ctr_bits;

/* RTC Raw Service Request Register */
typedef struct {
  __REG32  RPSE           : 1;
  __REG32  RPMI           : 1;
  __REG32  RPHO           : 1;
  __REG32  RPDA           : 1;
  __REG32                 : 1;
  __REG32  RPMO           : 1;
  __REG32  RPYE           : 1;
  __REG32                 : 1;
  __REG32  RAI            : 1;
  __REG32                 :23;
} __rtc_rawstat_bits;

/* RTC Service Request Status Register */
typedef struct {
  __REG32  SPSE           : 1;
  __REG32  SPMI           : 1;
  __REG32  SPHO           : 1;
  __REG32  SPDA           : 1;
  __REG32                 : 1;
  __REG32  SPMO           : 1;
  __REG32  SPYE           : 1;
  __REG32                 : 1;
  __REG32  SAI            : 1;
  __REG32                 :23;
} __rtc_stssr_bits;

/* RTC Service Request Status Register */
typedef struct {
  __REG32  MPSE           : 1;
  __REG32  MPMI           : 1;
  __REG32  MPHO           : 1;
  __REG32  MPDA           : 1;
  __REG32                 : 1;
  __REG32  MPMO           : 1;
  __REG32  MPYE           : 1;
  __REG32                 : 1;
  __REG32  MAI            : 1;
  __REG32                 :23;
} __rtc_msksr_bits;

/* RTC Alarm Time Register 0 */
typedef struct {
  __REG32  ASE            : 6;
  __REG32                 : 2;
  __REG32  AMI            : 6;
  __REG32                 : 2;
  __REG32  AHO            : 5;
  __REG32                 : 3;
  __REG32  ADA            : 5;
  __REG32                 : 3;
} __rtc_atim0_bits;

/* RTC Alarm Time Register 1 */
typedef struct {
  __REG32  ADAWE          : 3;
  __REG32                 : 5;
  __REG32  AMO            : 4;
  __REG32                 : 4;
  __REG32  AYE            :16;
} __rtc_atim1_bits;

/* RTC Time Register 0 */
typedef struct {
  __REG32  SE             : 6;
  __REG32                 : 2;
  __REG32  MI             : 6;
  __REG32                 : 2;
  __REG32  HO             : 5;
  __REG32                 : 3;
  __REG32  DA             : 5;
  __REG32                 : 3;
} __rtc_tim0_bits;

/* RTC Time Register 1 */
typedef struct {
  __REG32  DAWE           : 3;
  __REG32                 : 5;
  __REG32  MO             : 4;
  __REG32                 : 4;
  __REG32  YE             :16;
} __rtc_tim1_bits;

/* Event Input Select Register */
typedef struct {
  __REG32  EXS0A          : 2;
  __REG32  EXS0B          : 2;
  __REG32  EXS1A          : 2;
  __REG32  EXS1B          : 2;
  __REG32  EXS2A          : 2;
  __REG32  EXS2B          : 2;
  __REG32  EXS3A          : 2;
  __REG32  EXS3B          : 2;
  __REG32                 :16;
} __eru_exisel_bits;

/* Event Input Control Register */
typedef struct {
  __REG32  PE             : 1;
  __REG32  LD             : 1;
  __REG32  RE             : 1;
  __REG32  FE             : 1;
  __REG32  OCS            : 3;
  __REG32  FL             : 1;
  __REG32  SS             : 2;
  __REG32  NA             : 1;
  __REG32  NB             : 1;
  __REG32                 :20;
} __eru_exicon_bits;

/* Event Output Trigger Control Register */
typedef struct {
  __REG32  ISS            : 2;
  __REG32  GEEN           : 1;
  __REG32  PDR            : 1;
  __REG32  GP             : 2;
  __REG32                 : 6;
  __REG32  IPEN0          : 1;
  __REG32  IPEN1          : 1;
  __REG32  IPEN2          : 1;
  __REG32  IPEN3          : 1;
  __REG32                 :16;
} __eru_exocon_bits;

/* GPDMA Overrun Status */
typedef struct {
  __REG32  LN0            : 1;
  __REG32  LN1            : 1;
  __REG32  LN2            : 1;
  __REG32  LN3            : 1;
  __REG32  LN4            : 1;
  __REG32  LN5            : 1;
  __REG32  LN6            : 1;
  __REG32  LN7            : 1;
  __REG32  LN8            : 1;
  __REG32  LN9            : 1;
  __REG32  LN10           : 1;
  __REG32  LN11           : 1;
  __REG32                 :20;
} __gpdma_ovrstat_bits;

/* GPDMA Service Request Selection 0 */
typedef struct {
  __REG32  RS0            : 4;
  __REG32  RS1            : 4;
  __REG32  RS2            : 4;
  __REG32  RS3            : 4;
  __REG32  RS4            : 4;
  __REG32  RS5            : 4;
  __REG32  RS6            : 4;
  __REG32  RS7            : 4;
} __gpdma_srsel0_bits;

/* GPDMA Service Request Selection 1 */
typedef struct {
  __REG32  RS8            : 4;
  __REG32  RS9            : 4;
  __REG32  RS10           : 4;
  __REG32  RS11           : 4;
  __REG32                 :16;
} __gpdma_srsel1_bits;

/* Startup Configuration Register */
typedef struct {
  __REG32  HWCON          : 2;
  __REG32                 : 6;
  __REG32  SWCON          : 4;
  __REG32                 :20;
} __scu_stcon_bits;

/* Ethernet 0 Port Control Register */
typedef struct {
  __REG32  RXD0           : 2;
  __REG32  RXD1           : 2;
  __REG32  RXD2           : 2;
  __REG32  RXD3           : 2;
  __REG32  CLK_RMII       : 2;
  __REG32  CRS_DV         : 2;
  __REG32  CRS            : 2;
  __REG32  RXER           : 2;
  __REG32  COL            : 2;
  __REG32  CLK_TX         : 2;
  __REG32                 : 2;
  __REG32  MDIO           : 2;
  __REG32                 : 2;
  __REG32  INFSEL         : 1;
  __REG32                 : 5;
} __scu_eth0_con_bits;

/* TCU Control Register */
typedef struct {
  __REG32  FTM            : 7;
  __REG32  TMD            : 1;
  __REG32                 :24;
} __scu_tcucon_bits;

/* CCU Control Register */
typedef struct {
  __REG32  GSC40          : 1;
  __REG32  GSC41          : 1;
  __REG32  GSC42          : 1;
  __REG32  GSC43          : 1;
  __REG32                 : 4;
  __REG32  GSC80          : 1;
  __REG32  GSC81          : 1;
  __REG32                 :22;
} __scu_ccucon_bits;

/* Debug Configuration Register 0 */
typedef struct {
  __REG32                 : 1;
  __REG32  DAPSA          : 1;
  __REG32                 :30;
} __scu_dbgcon0_bits;

/* Debug Configuration Register 1 */
typedef struct {
  __REG32  DPUR           : 1;
  __REG32                 :31;
} __scu_dbgcon1_bits;

/* SCU Service Request Status */
typedef struct {
  __REG32  PRWARN         : 1;
  __REG32  PI             : 1;
  __REG32  AI             : 1;
  __REG32  DLROVR         : 1;
  __REG32                 :12;
  __REG32  HDSTAT         : 1;
  __REG32  HDCLR          : 1;
  __REG32  HDSET          : 1;
  __REG32  HDCR           : 1;
  __REG32  OSCSITRIM      : 1;
  __REG32  OSCSICTRL      : 1;
  __REG32  OSCULSTAT      : 1;
  __REG32  OSCULCTRL      : 1;
  __REG32  RTC_CTR        : 1;
  __REG32  RTC_ATIM0      : 1;
  __REG32  RTC_ATIM1      : 1;
  __REG32  RTC_TIM0       : 1;
  __REG32  RTC_TIM1       : 1;
  __REG32  RMX            : 1;
  __REG32                 : 2;
} __scu_srstat_bits;

/* SCU Service Request Mask */
typedef struct {
  __REG32  PRWARN         : 1;
  __REG32  PI             : 1;
  __REG32  AI             : 1;
  __REG32                 :13;
  __REG32  ERU00          : 1;
  __REG32  ERU01          : 1;
  __REG32  ERU02          : 1;
  __REG32  ERU03          : 1;
  __REG32                 :12;
} __scu_nmireqen_bits;

/* Die Temperature Sensor Control Register */
typedef struct {
  __REG32  PWD            : 1;
  __REG32  START          : 1;
  __REG32                 : 2;
  __REG32  OFFSET         : 7;
  __REG32  GAIN           : 6;
  __REG32  REFTRIM        : 3;
  __REG32  BGTRIM         : 4;
  __REG32                 : 8;
} __scu_dtscon_bits;

/* Die Temperature Sensor Status Register */
typedef struct {
  __REG32  RESULT         :10;
  __REG32                 : 4;
  __REG32  RDY            : 1;
  __REG32  BUSY           : 1;
  __REG32                 :16;
} __scu_dtsstat_bits;

/* Start-up Protection Control */
typedef struct {
  __REG32  STP            : 1;
  __REG32  FCBAE          : 1;
  __REG32                 :30;
} __scu_stpcon_bits;

/* USB PHY Trim Register */
typedef struct {
  __REG32  OPTUNE         : 2;
  __REG32  ONTUNE         : 2;
  __REG32                 : 4;
  __REG32  VBUSTUNE       : 2;
  __REG32                 :22;
} __scu_usbtrim_bits;

/* SD-MMC Delay Control Register */
typedef struct {
  __REG32  TAPEN          : 1;
  __REG32                 : 1;
  __REG32  DLYCTRL        : 2;
  __REG32  TAPDEL         : 4;
  __REG32                 :24;
} __scu_sdmmcdel_bits;

/* Out of Range Comparator Enable Register 0/1 */
typedef struct {
  __REG32                 : 6;
  __REG32  ENORC6         : 1;
  __REG32  ENORC7         : 1;
  __REG32                 :24;
} __scu_gorcen_bits;

/* Mirror Update Status Register*/
typedef struct {
  __REG32  HDSTAT         : 1;
  __REG32  HDCLR          : 1;
  __REG32  HDSET          : 1;
  __REG32  HDCR           : 1;
  __REG32  OSCSITRIM      : 1;
  __REG32  OSCSICTRL      : 1;
  __REG32  OSCULSTAT      : 1;
  __REG32  OSCULCTRL      : 1;
  __REG32  RTC_CTR        : 1;
  __REG32  RTC_ATIM0      : 1;
  __REG32  RTC_ATIM1      : 1;
  __REG32  RTC_TIM0       : 1;
  __REG32  RTC_TIM1       : 1;
  __REG32  RMX            : 1;
  __REG32                 :18;
} __scu_mirrsts_bits;

/* Retention Memory Access Control Register */
typedef struct {
  __REG32  RDWR           : 1;
  __REG32                 :15;
  __REG32  ADDR           : 4;
  __REG32                 :12;
} __scu_rmacr_bits;

/* Flash Redundancy Status Register */
typedef struct {
  __REG32  BUSY           : 1;
  __REG32                 :31;
} __scu_hdfrstat_bits;

/* Flash Redundancy Shift Control Register */
typedef struct {
  __REG32  START          : 1;
  __REG32                 :31;
} __scu_hdfrcmd_bits;

/* Parity Error Trap Enable Register */
typedef struct {
  __REG32  PETEPS         : 1;
  __REG32  PETEDS1        : 1;
  __REG32  PETEDS2        : 1;
  __REG32                 : 5;
  __REG32  PETEU0         : 1;
  __REG32  PETEU1         : 1;
  __REG32  PETEU2         : 1;
  __REG32                 : 1;
  __REG32  PETEMC         : 1;
  __REG32  PETEPPRF       : 1;
  __REG32                 : 2;
  __REG32  PETEUSB        : 1;
  __REG32  PETEETH0TX     : 1;
  __REG32  PETEETH0RX     : 1;
  __REG32  PETESD0        : 1;
  __REG32  PETESD1        : 1;
  __REG32                 :11;
} __scu_pete_bits;

/* Memory Checking Control Register */
typedef struct {
  __REG32  SELPS          : 1;
  __REG32  SELDS1         : 1;
  __REG32  SELDS2         : 1;
  __REG32                 : 5;
  __REG32  USIC0DRA       : 1;
  __REG32  USIC1DRA       : 1;
  __REG32  USIC2DRA       : 1;
  __REG32                 : 1;
  __REG32  MCANDRA        : 1;
  __REG32  PPRFDRA        : 1;
  __REG32                 : 2;
  __REG32  SELUSB         : 1;
  __REG32  SELETH0TX      : 1;
  __REG32  SELETH0RX      : 1;
  __REG32  SELSD0         : 1;
  __REG32  SELSD1         : 1;
  __REG32                 :11;
} __scu_mchkcon_bits;

/* Parity Error Enable Register */
typedef struct {
  __REG32  PEENPS         : 1;
  __REG32  PEENDS1        : 1;
  __REG32  PEENDS2        : 1;
  __REG32                 : 5;
  __REG32  PEENU0         : 1;
  __REG32  PEENU1         : 1;
  __REG32  PEENU2         : 1;
  __REG32                 : 1;
  __REG32  PEENMC         : 1;
  __REG32  PEENPPRF       : 1;
  __REG32                 : 2;
  __REG32  PEENUSB        : 1;
  __REG32  PEENETH0TX     : 1;
  __REG32  PEENETH0RX     : 1;
  __REG32  PEENSD0        : 1;
  __REG32  PEENSD1        : 1;
  __REG32                 :11;
} __scu_peen_bits;

/* Parity Error Reset Enable Register */
typedef struct {
  __REG32  RSEN           : 1;
  __REG32                 :31;
} __scu_persten_bits;

/* Parity Error Flag Register */
typedef struct {
  __REG32  PEFPS          : 1;
  __REG32  PEFDS1         : 1;
  __REG32  PEFDS2         : 1;
  __REG32                 : 5;
  __REG32  PEFU0          : 1;
  __REG32  PEFU1          : 1;
  __REG32  PEFU2          : 1;
  __REG32                 : 1;
  __REG32  PEFMC          : 1;
  __REG32  PEFPPRF        : 1;
  __REG32                 : 2;
  __REG32  PEUSB          : 1;
  __REG32  PEETH0TX       : 1;
  __REG32  PEETH0RX       : 1;
  __REG32  PESD0          : 1;
  __REG32  PESD1          : 1;
  __REG32                 :11;
} __scu_peflag_bits;

/* Parity Memory Test Pattern Register */
typedef struct {
  __REG32  PWR            : 8;
  __REG32  PRD            : 8;
  __REG32                 :16;
} __scu_pmtpr_bits;

/* Parity Memory Test Select Register */
typedef struct {
  __REG32  MTENPS         : 1;
  __REG32  MTENDS1        : 1;
  __REG32  MTENDS2        : 1;
  __REG32                 : 5;
  __REG32  MTEU0          : 1;
  __REG32  MTEU1          : 1;
  __REG32  MTEU2          : 1;
  __REG32                 : 1;
  __REG32  MTEMC          : 1;
  __REG32  MTEPPRF        : 1;
  __REG32                 : 2;
  __REG32  MTUSB          : 1;
  __REG32  MTETH0TX       : 1;
  __REG32  MTETH0RX       : 1;
  __REG32  MTSD0          : 1;
  __REG32  MTSD1          : 1;
  __REG32                 :11;
} __scu_pmtsr_bits;

/* Trap Status Register */
typedef struct {
  __REG32  SOSCWDGT       : 1;
  __REG32                 : 1;
  __REG32  SVCOLCKT       : 1;
  __REG32  UVCOLCKT       : 1;
  __REG32  PET            : 1;
  __REG32  BRWNT          : 1;
  __REG32  ULPWDGT        : 1;
  __REG32  BWERR0T        : 1;
  __REG32  BWERR1T        : 1;
  __REG32                 :23;
} __scu_trapstat_bits;

/* PCU Status Register */
typedef struct {
  __REG32  HIBEN          : 1;
  __REG32  HPSS           : 1;
  __REG32                 :14;
  __REG32  USBPHYPDQ      : 1;
  __REG32  USBOTGEN       : 1;
  __REG32  USBPUWQ        : 1;
  __REG32                 :13;
} __pcu_pwrstat_bits;

/* PCU Set Control Register */
typedef struct {
  __REG32  HIB            : 1;
  __REG32                 :15;
  __REG32  USBPHYPDQ      : 1;
  __REG32  USBOTGEN       : 1;
  __REG32  USBPUWQ        : 1;
  __REG32                 :13;
} __pcu_pwrset_bits;

/* EVR Status Register */
typedef struct {
  __REG32                 : 1;
  __REG32  OV13           : 1;
  __REG32                 :30;
} __pcu_evrstat_bits;

/* EVR VADC Status Register */
typedef struct {
  __REG32  VADC13V        : 8;
  __REG32  VADC33V        : 8;
  __REG32                 :16;
} __pcu_evrvadcstat_bits;

/* EVR Trim Register */
typedef struct {
  __REG32  EVR13TRIM      : 8;
  __REG32                 : 4;
  __REG32  EVR13OFF       : 1;
  __REG32                 :19;
} __pcu_evrtrim_bits;

/* EVR Reset Control Register */
typedef struct {
  __REG32  RST13TRIM      : 8;
  __REG32  RST33TRIM      : 8;
  __REG32                 : 8;
  __REG32  RST13OFF       : 1;
  __REG32                 : 1;
  __REG32  RST33OFF       : 1;
  __REG32                 : 5;
} __pcu_evrrstcon_bits;

/* EVR Reset Control Register */
typedef struct {
  __REG32  EVR13KP        : 4;
  __REG32                 : 4;
  __REG32  EVR13KI        : 4;
  __REG32                 : 4;
  __REG32  EVR13KD        : 4;
  __REG32                 :12;
} __pcu_evr13con_bits;

/* EVR Oscillator Register */
typedef struct {
  __REG32  EVR20MTRIM     : 5;
  __REG32                 :27;
} __pcu_evrosc_bits;

/* Power Monitor Control */
typedef struct {
  __REG32  THRS           : 8;
  __REG32  INTV           : 8;
  __REG32  ENB            : 1;
  __REG32                 :15;
} __pcu_pwrmon_bits;

/* Hibernate Domain Control & Status Register */
typedef struct {
  __REG32  EPEV           : 1;
  __REG32  ENEV           : 1;
  __REG32  RTCEV          : 1;
  __REG32  ULPWDG         : 1;
  __REG32  HIBNOUT        : 1;
  __REG32                 :27;
} __hcu_hdstat_bits;

/* Hibernate Domain Configuration Register */
typedef struct {
  __REG32  WKPEP          : 1;
  __REG32  WKPEN          : 1;
  __REG32  RTCE           : 1;
  __REG32  ULPWDGEN       : 1;
  __REG32  HIB            : 1;
  __REG32                 : 1;
  __REG32  RCS            : 1;
  __REG32  STDBYSEL       : 1;
  __REG32  WKUPSEL        : 1;
  __REG32                 : 1;
  __REG32  GPI0SEL        : 1;
  __REG32                 : 1;
  __REG32  HIBIO0POL      : 1;
  __REG32  HIBIO1POL      : 1;
  __REG32                 : 2;
  __REG32  HIBIO0SEL      : 4;
  __REG32  HIBIO1SEL      : 4;
  __REG32                 : 8;
} __hcu_hdcr_bits;

/* OSC_SI Trim Register */
typedef struct {
  __REG32  OTRM           :10;
  __REG32                 :22;
} __hcu_oscsitrim_bits;

/* OSC_SI Control Register */
typedef struct {
  __REG32  PWD            : 1;
  __REG32                 :31;
} __hcu_oscsictrl_bits;

/* OSC_ULP Status Register */
typedef struct {
  __REG32  X1D            : 1;
  __REG32                 :31;
} __hcu_osculstat_bits;

/* OSC_ULP Control Register */
typedef struct {
  __REG32  X1DEN          : 1;
  __REG32                 : 3;
  __REG32  MODE           : 2;
  __REG32                 :26;
} __hcu_osculctrl_bits;

/* Analog Wakeup Request Control Register */
typedef struct {
  __REG32  CMPEN          : 3;
  __REG32                 : 1;
  __REG32  TRIGSEL        : 3;
  __REG32                 : 1;
  __REG32  VBATLO         : 1;
  __REG32  VBATHI         : 1;
  __REG32  AHIBIO0LO      : 1;
  __REG32  AHIBIO0HI      : 1;
  __REG32  AHIBIO1LO      : 1;
  __REG32  AHIBIO1HI      : 1;
  __REG32                 : 2;
  __REG32  INTERVCNT      :12;
  __REG32  SETTLECNT      : 4;
} __hcu_lpaccr_bits;

/* Analog Wakeup Threshold Register 0/1 */
typedef union {
  /* HCU_LPACTH0 */
  struct {
  __REG32  VBATLO         : 6;
  __REG32                 : 2;
  __REG32  VBATHI         : 6;
  __REG32                 :18;
  };
  /* HCU_LPACTH1 */
  struct {
  __REG32  AHIBIO0LO      : 6;
  __REG32                 : 2;
  __REG32  AHIBIO0HI      : 6;
  __REG32                 : 2;
  __REG32  AHIBIO1LO      : 6;
  __REG32                 : 2;
  __REG32  AHIBIO1HI      : 6;
  __REG32                 : 2;
  };
} __hcu_lpacth_bits;

/* RCU Reset Status */
typedef struct {
  __REG32  RSTSTAT        : 8;
  __REG32  HIBWK          : 1;
  __REG32  HIBRS          : 1;
  __REG32  LCKEN          : 1;
  __REG32                 :21;
} __rcu_rststat_bits;

/* RCU Reset Set Register */
typedef struct {
  __REG32                 : 8;
  __REG32  HIBWK          : 1;
  __REG32  HIBRS          : 1;
  __REG32  LCKEN          : 1;
  __REG32                 :21;
} __rcu_rstset_bits;

/* RCU Peripheral 0 Reset Status */
typedef struct {
  __REG32  VADCRS         : 1;
  __REG32  DSDRS          : 1;
  __REG32  CCU40RS        : 1;
  __REG32  CCU41RS        : 1;
  __REG32  CCU42RS        : 1;
  __REG32                 : 2;
  __REG32  CCU80RS        : 1;
  __REG32  CCU81RS        : 1;
  __REG32  POSIF0RS       : 1;
  __REG32  POSIF1RS       : 1;
  __REG32  USIC0RS        : 1;
  __REG32                 : 4;
  __REG32  ERU1RS         : 1;
  __REG32                 :15;
} __rcu_prstat_bits;

/* RCU Peripheral 1 Reset Status */
typedef struct {
  __REG32  CCU43RS        : 1;
  __REG32                 : 2;
  __REG32  LEDTSCU0RS     : 1;
  __REG32  MCAN0RS        : 1;
  __REG32  DACRS          : 1;
  __REG32  MMCIRS         : 1;
  __REG32  USIC1RS        : 1;
  __REG32  USIC2RS        : 1;
  __REG32  PPORTSRS       : 1;
  __REG32                 :22;
} __rcu_prstat1_bits;

/* RCU Peripheral 2 Reset Status */
typedef struct {
  __REG32                 : 1;
  __REG32  WDTRS          : 1;
  __REG32  ETH0RS         : 1;
  __REG32                 : 1;
  __REG32  DMA0RS         : 1;
  __REG32  DMA1RS         : 1;
  __REG32  FCERS          : 1;
  __REG32  USBRS          : 1;
  __REG32                 :24;
} __rcu_prstat2_bits;

/* RCU Peripheral 3 Reset Status */
typedef struct {
  __REG32                 : 2;
  __REG32  EBURS          : 1;
  __REG32                 :29;
} __rcu_prstat3_bits;

/* Clock Status Register */
typedef struct {
  __REG32  USBCST         : 1;
  __REG32  MMCCST         : 1;
  __REG32  ETH0CST        : 1;
  __REG32  EBUCST         : 1;
  __REG32  CCUCST         : 1;
  __REG32  WDTCST         : 1;
  __REG32                 :26;
} __ccu_clkstat_bits;

/* CLK Set Register */
typedef struct {
  __REG32  USBCEN         : 1;
  __REG32  MMCCEN         : 1;
  __REG32  ETH0CEN        : 1;
  __REG32  EBUCEN         : 1;
  __REG32  CCUCEN         : 1;
  __REG32  WDTCEN         : 1;
  __REG32                 :26;
} __ccu_clkset_bits;

/* CLK Clear Register */
typedef struct {
  __REG32  USBCDI         : 1;
  __REG32  MMCCDI         : 1;
  __REG32  ETH0CDI        : 1;
  __REG32  EBUCDI         : 1;
  __REG32  CCUCDI         : 1;
  __REG32  WDTCDI         : 1;
  __REG32                 :26;
} __ccu_clkclr_bits;

/* System Clock Control Register */
typedef struct {
  __REG32  SYSDIV         : 8;
  __REG32                 : 8;
  __REG32  SYSSEL         : 2;
  __REG32                 :14;
} __ccu_sysclkcr_bits;

/* CPU Clock Control Register */
typedef struct {
  __REG32  CPUDIV         : 1;
  __REG32                 :31;
} __ccu_cpuclkcr_bits;

/* Peripheral Bus Clock Control Register */
typedef struct {
  __REG32  PBDIV          : 1;
  __REG32                 :31;
} __ccu_pbclkcr_bits;

/* USB Clock Control Register */
typedef struct {
  __REG32  USBSEL         : 3;
  __REG32                 :13;
  __REG32  USBDIV         : 1;
  __REG32                 :15;
} __ccu_usbclkcr_bits;

/* EBU Clock Control Register */
typedef struct {
  __REG32  EBUDIV         : 6;
  __REG32                 :26;
} __ccu_ebuclkcr_bits;

/* CCU Clock Control Register */
typedef struct {
  __REG32  CCUDIV         : 6;
  __REG32                 :26;
} __ccu_ccuclkcr_bits;

/* WDT Clock Control Register */
typedef struct {
  __REG32  WDTDIV         : 8;
  __REG32                 : 8;
  __REG32  WDTSEL         : 2;
  __REG32                 :14;
} __ccu_wdtclkcr_bits;

/* External Clock Control */
typedef struct {
  __REG32  ECKSEL         : 2;
  __REG32                 :14;
  __REG32  ECKDIV         : 9;
  __REG32                 : 7;
} __ccu_extclkcr_bits;

/* Sleep Control Register */
typedef struct {
  __REG32  SYSSEL         : 2;
  __REG32                 :14;
  __REG32  USBCR          : 1;
  __REG32  MMCCR          : 1;
  __REG32  ETH0CR         : 1;
  __REG32  EBUCR          : 1;
  __REG32  CCUCR          : 1;
  __REG32  WDTCR          : 1;
  __REG32                 :10;
} __ccu_sleepcr_bits;

/* Deep Sleep Control Register */
typedef struct {
  __REG32  SYSSEL         : 2;
  __REG32                 : 9;
  __REG32  FPDN           : 1;
  __REG32  PLLPDN         : 1;
  __REG32  VCOPDN         : 1;
  __REG32                 : 2;
  __REG32  USBCR          : 1;
  __REG32  MMCCR          : 1;
  __REG32  ETH0CR         : 1;
  __REG32  EBUCR          : 1;
  __REG32  CCUCR          : 1;
  __REG32  WDTCR          : 1;
  __REG32                 :10;
} __ccu_dsleepcr_bits;

/* OSC_HP Status Register */
typedef struct {
  __REG32  X1D            : 1;
  __REG32                 :31;
} __ccu_oschpstat_bits;

/* OSC_HP Status Register */
typedef struct {
  __REG32  X1DEN          : 1;
  __REG32  SHBY           : 1;
  __REG32  GAINSEL        : 2;
  __REG32  MODE           : 2;
  __REG32                 :10;
  __REG32  OSCVAL         : 5;
  __REG32                 :11;
} __ccu_oschpctrl_bits;

/* OSC_FI Trim Register */
typedef struct {
  __REG32  OTRM           : 4;
  __REG32                 :28;
} __ccu_oscfitrim_bits;

/* PLL Status Register */
typedef struct {
  __REG32  VCOBYST        : 1;
  __REG32  PWDSTAT        : 1;
  __REG32  VCOLOCK        : 1;
  __REG32                 : 1;
  __REG32  K1RDY          : 1;
  __REG32  K2RDY          : 1;
  __REG32  BY             : 1;
  __REG32  PLLLV          : 1;
  __REG32  PLLHV          : 1;
  __REG32  PLLSP          : 1;
  __REG32                 :22;
} __ccu_pllstat_bits;

/* PLL Configuration 0 Register */
typedef struct {
  __REG32  VCOBYP         : 1;
  __REG32  VCOPWD         : 1;
  __REG32  VCOTR          : 1;
  __REG32                 : 1;
  __REG32  FINDIS         : 1;
  __REG32                 : 1;
  __REG32  OSCDISCDIS     : 1;
  __REG32                 : 9;
  __REG32  PLLPWD         : 1;
  __REG32  OSCRES         : 1;
  __REG32  RESLD          : 1;
  __REG32  AOTREN         : 1;
  __REG32  FOTR           : 1;
  __REG32                 :11;
} __ccu_pllcon0_bits;

/* PLL Configuration 1 Register */
typedef struct {
  __REG32  K1DIV          : 7;
  __REG32                 : 1;
  __REG32  NDIV           : 7;
  __REG32                 : 1;
  __REG32  K2DIV          : 7;
  __REG32                 : 1;
  __REG32  PDIV           : 4;
  __REG32                 : 4;
} __ccu_pllcon1_bits;

/* PLL Configuration 2 Register */
typedef struct {
  __REG32  PINSEL         : 1;
  __REG32                 : 7;
  __REG32  K1INSEL        : 1;
  __REG32                 :23;
} __ccu_pllcon2_bits;

/* USB PLL Status Register */
typedef struct {
  __REG32  VCOBYST        : 1;
  __REG32  PWDSTAT        : 1;
  __REG32  VCOLOCK        : 1;
  __REG32                 : 3;
  __REG32  BY             : 1;
  __REG32  VCOLOCKED      : 1;
  __REG32                 :24;
} __ccu_usbpllstat_bits;

/* USB PLL Configuration Register */
typedef struct {
  __REG32  VCOBYP         : 1;
  __REG32  VCOPWD         : 1;
  __REG32  VCOTR          : 1;
  __REG32                 : 1;
  __REG32  FINDIS         : 1;
  __REG32                 : 1;
  __REG32  OSCDISCDIS     : 1;
  __REG32                 : 1;
  __REG32  NDIV           : 7;
  __REG32                 : 1;
  __REG32  PLLPWD         : 1;
  __REG32                 : 1;
  __REG32  RESLD          : 1;
  __REG32                 : 5;
  __REG32  PDIV           : 4;
  __REG32                 : 4;
} __ccu_usbpllcon_bits;

/* Clock Multiplexing Status Register */
typedef struct {
  __REG32  SYSCLKMUX      : 3;
  __REG32                 :29;
} __ccu_clkmxstat_bits;

/* EBU Clock Control Register */
typedef struct {
  __REG32  DISR           : 1;
  __REG32  DISS           : 1;
  __REG32                 :14;
  __REG32  SYNC           : 1;
  __REG32  DIV2           : 1;
  __REG32  EBUDIV         : 2;
  __REG32  SYNCACK        : 1;
  __REG32  DIV2ACK        : 1;
  __REG32  EBUDIVACK      : 2;
  __REG32                 : 8;
} __ebu_clc_bits;

/* Configuration Register */
typedef struct {
  __REG32  STS            : 1;
  __REG32  LCKABRT        : 1;
  __REG32  SDTRI          : 1;
  __REG32                 : 1;
  __REG32  EXTLOCK        : 1;
  __REG32  ARBSYNC        : 1;
  __REG32  ARBMODE        : 2;
  __REG32  TIMEOUTC       : 8;
  __REG32  LOCKTIMEOUT    : 8;
  __REG32  GLOBALCS       : 4;
  __REG32  ACCSINH        : 1;
  __REG32  ACCSINHACK     : 1;
  __REG32                 : 1;
  __REG32  ALE            : 1;
} __ebu_modcon_bits;

/* Address Select Register */
typedef struct {
  __REG32  REGENAB        : 1;
  __REG32  ALTENAB        : 1;
  __REG32  WPROT          : 1;
  __REG32                 :29;
} __ebu_addrsel_bits;

/* EBU Bus Configuration Register */
typedef struct {
  __REG32  FETBLEN        : 3;
  __REG32  FBBMSEL        : 1;
  __REG32  BFSSS          : 1;
  __REG32  FDBKEN         : 1;
  __REG32  BFCMSEL        : 1;
  __REG32  NAA            : 1;
  __REG32                 : 8;
  __REG32  ECSE           : 1;
  __REG32  EBSE           : 1;
  __REG32  DBA            : 1;
  __REG32  WAITINV        : 1;
  __REG32  BCGEN          : 2;
  __REG32  PORTW          : 2;
  __REG32  WAIT           : 2;
  __REG32  AAP            : 1;
  __REG32                 : 1;
  __REG32  AGEN           : 4;
} __ebu_busrcon_bits;

/* EBU Bus Write Configuration Register */
typedef struct {
  __REG32  FETBLEN        : 3;
  __REG32  FBBMSEL        : 1;
  __REG32                 : 3;
  __REG32  NAA            : 1;
  __REG32                 : 8;
  __REG32  ECSE           : 1;
  __REG32  EBSE           : 1;
  __REG32                 : 1;
  __REG32  WAITINV        : 1;
  __REG32  BCGEN          : 2;
  __REG32  PORTW          : 2;
  __REG32  WAIT           : 2;
  __REG32  AAP            : 1;
  __REG32  LOCKCS         : 1;
  __REG32  AGEN           : 4;
} __ebu_buswcon_bits;

/* EBU Bus Read Access Parameter Register */
typedef struct {
  __REG32  RDDTACS        : 4;
  __REG32  RDRECOVC       : 3;
  __REG32  WAITRDC        : 5;
  __REG32  DATAC          : 4;
  __REG32  EXTCLOCK       : 2;
  __REG32  EXTDATA        : 2;
  __REG32  CMDDELAY       : 4;
  __REG32  AHOLDC         : 4;
  __REG32  ADDRC          : 4;
} __ebu_busrap_bits;

/* EBU Bus Write Access Parameter Register */
typedef struct {
  __REG32  WRDTACS        : 4;
  __REG32  WRRECOVC       : 3;
  __REG32  WAITWRC        : 5;
  __REG32  DATAC          : 4;
  __REG32  EXTCLOCK       : 2;
  __REG32  EXTDATA        : 2;
  __REG32  CMDDELAY       : 4;
  __REG32  AHOLDC         : 4;
  __REG32  ADDRC          : 4;
} __ebu_buswap_bits;

/* EBU SDRAM Control Register */
typedef struct {
  __REG32  CRAS           : 4;
  __REG32  CRFSH          : 4;
  __REG32  CRSC           : 2;
  __REG32  CRP            : 2;
  __REG32  AWIDTH         : 2;
  __REG32  CRCD           : 2;
  __REG32  CRC            : 3;
  __REG32  ROWM           : 3;
  __REG32  BANKM          : 3;
  __REG32  CRCE           : 3;
  __REG32  CLKDIS         : 1;
  __REG32  PWR_MODE       : 2;
  __REG32  SDCMSEL        : 1;
} __ebu_sdrmcon_bits;

/* EBU SDRAM Mode Register */
typedef struct {
  __REG32  BURSTL         : 3;
  __REG32  BTYP           : 1;
  __REG32  CASLAT         : 3;
  __REG32  OPMODE         : 7;
  __REG32                 : 1;
  __REG32  COLDSTART      : 1;
  __REG32  XOPM           :12;
  __REG32  XBA            : 4;
} __ebu_sdrmod_bits;

/* EBU SDRAM Refresh Control Register */
typedef struct {
  __REG32  REFRESHC       : 6;
  __REG32  REFRESHR       : 3;
  __REG32  SELFREXST      : 1;
  __REG32  SELFREX        : 1;
  __REG32  SELFRENST      : 1;
  __REG32  SELFREN        : 1;
  __REG32  AUTOSELFR      : 1;
  __REG32  ERFSHC         : 2;
  __REG32  SELFREX_DLY    : 8;
  __REG32  ARFSH          : 1;
  __REG32  RES_DLY        : 3;
  __REG32                 : 4;
} __ebu_sdrmref_bits;

/* EBU SDRAM Status Register */
typedef struct {
  __REG32  REFERR         : 1;
  __REG32  SDRMBUSY       : 1;
  __REG32  SDERR          : 1;
  __REG32                 :29;
} __ebu_sdrstat_bits;

/* EBU Test/Control Configuration Register */
typedef struct {
  __REG32  DIP            : 1;
  __REG32                 :15;
  __REG32  ADDIO          : 9;
  __REG32  ADVIO          : 1;
  __REG32                 : 6;
} __ebu_usercon_bits;

/* EBU Test/Control Configuration Register */
typedef struct {
  __REG32  MOD_REV        : 8;
  __REG32  MOD_TYPE       : 8;
  __REG32  MOD_NUMBER     :16;
} __ebu_id_bits;

/* Global Control Register */
typedef struct {
  __REG32  TS_EN          : 1;
  __REG32  LD_EN          : 1;
  __REG32  CMTR           : 1;
  __REG32  ENSYNC         : 1;
  __REG32                 : 4;
  __REG32  SUSCFG         : 1;
  __REG32  MASKVAL        : 3;
  __REG32  FENVAL         : 1;
  __REG32  ITS_EN         : 1;
  __REG32  ITF_EN         : 1;
  __REG32  ITP_EN         : 1;
  __REG32  CLK_PS         :16;
} __ledtscu_globctl_bits;

/* Function Control Register */
typedef struct {
  __REG32  PADT           : 3;
  __REG32  PADTSW         : 1;
  __REG32  EPULL          : 1;
  __REG32  FNCOL          : 3;
  __REG32                 : 8;
  __REG32  ACCCNT         : 4;
  __REG32  TSCCMP         : 1;
  __REG32  TSOEXT         : 2;
  __REG32  TSCTRR         : 1;
  __REG32  TSCTRSAT       : 1;
  __REG32  NR_TSIN        : 3;
  __REG32  COLLEV         : 1;
  __REG32  NR_LEDCOL      : 3;
} __ledtscu_fnctl_bits;

/* Function Control Register */
typedef struct {
  __REG32  TSF            : 1;
  __REG32  TFF            : 1;
  __REG32  TPF            : 1;
  __REG32  TSCTROVF       : 1;
  __REG32                 :12;
  __REG32  CTSF           : 1;
  __REG32  CTFF           : 1;
  __REG32  CTPF           : 1;
  __REG32                 :13;
} __ledtscu_evfr_bits;

/* Touch-sense TS-Counter Value */
typedef struct {
  __REG32  TSCTRVALR      :16;
  __REG32  TSCTRVAL       :16;
} __ledtscu_tsval_bits;

/* Line Pattern Register 0 */
typedef struct {
  __REG32  LINE_0         : 8;
  __REG32  LINE_1         : 8;
  __REG32  LINE_2         : 8;
  __REG32  LINE_3         : 8;
} __ledtscu_line0_bits;

/* Line Pattern Register 1 */
typedef struct {
  __REG32  LINE_4         : 8;
  __REG32  LINE_5         : 8;
  __REG32  LINE_6         : 8;
  __REG32  LINE_A         : 8;
} __ledtscu_line1_bits;

/* LED Compare Register 0 */
typedef struct {
  __REG32  CMP_LD0        : 8;
  __REG32  CMP_LD1        : 8;
  __REG32  CMP_LD2        : 8;
  __REG32  CMP_LD3        : 8;
} __ledtscu_ldcmp0_bits;

/* LED Compare Register 1 */
typedef struct {
  __REG32  CMP_LD4        : 8;
  __REG32  CMP_LD5        : 8;
  __REG32  CMP_LD6        : 8;
  __REG32  CMP_LDATSCOM   : 8;
} __ledtscu_ldcmp1_bits;

/* Touch-sense Compare Register 0 */
typedef struct {
  __REG32  CMP_TS0        : 8;
  __REG32  CMP_TS1        : 8;
  __REG32  CMP_TS2        : 8;
  __REG32  CMP_TS3        : 8;
} __ledtscu_tscmp0_bits;

/* Touch-sense Compare Register 1 */
typedef struct {
  __REG32  CMP_TS4        : 8;
  __REG32  CMP_TS5        : 8;
  __REG32  CMP_TS6        : 8;
  __REG32  CMP_TS7        : 8;
} __ledtscu_tscmp1_bits;

/* Input Selector Configuration */
typedef struct {
  __REG32  TSIN0S         : 4;
  __REG32  TSIN1S         : 4;
  __REG32  TSIN2S         : 4;
  __REG32  TSIN3S         : 4;
  __REG32  TSIN4S         : 4;
  __REG32  TSIN5S         : 4;
  __REG32  TSIN6S         : 4;
  __REG32  TSIN7S         : 4;
} __ledtscu_ins_bits;

/* Module Identification Register */
typedef struct {
  __REG32  MOD_REV        : 8;
  __REG32  MOD_TYPE       : 8;
  __REG32  MOD_NUMBER     :16;
} __ledtscu_id_bits;

/* Block Size Register */
typedef struct {
  __REG16  TX_BLOCK_SIZE    :12;
  __REG16                   : 3;
  __REG16  TX_BLOCK_SIZE_12 : 1;
} __sdmmc_block_size_bits;

/* Transfer Mode Register */
typedef struct {
  __REG16                     : 1;
  __REG16  BLOCK_COUNT_EN     : 1;
  __REG16  ACMD_EN            : 2;
  __REG16  TX_DIR_SELECT      : 1;
  __REG16  MULTI_BLOCK_SELECT : 1;
  __REG16  CMD_COMP_ATA       : 1;
  __REG16                     : 9;
} __sdmmc_transfer_mode_bits;

/* Transfer Mode Register */
typedef struct {
  __REG16  RESP_TYPE_SELECT   : 2;
  __REG16                     : 1;
  __REG16  CMD_CRC_CHECK_E    : 1;
  __REG16  CMD_IND_CHECK_EN   : 1;
  __REG16  DATA_PRESENT_SE    : 1;
  __REG16  CMD_TYPE           : 2;
  __REG16  CMD_IND            : 6;
  __REG16                     : 2;
} __sdmmc_command_bits;

/* Response 0 Register */
typedef struct {
  __REG32  RESPONSE0          :16;
  __REG32  RESPONSE1          :16;
} __sdmmc_response0_bits;

/* Response 2 Register */
typedef struct {
  __REG32  RESPONSE2          :16;
  __REG32  RESPONSE3          :16;
} __sdmmc_response2_bits;

/* Response 4 Register */
typedef struct {
  __REG32  RESPONSE4          :16;
  __REG32  RESPONSE5          :16;
} __sdmmc_response4_bits;

/* Response 6 Register */
typedef struct {
  __REG32  RESPONSE6          :16;
  __REG32  RESPONSE7          :16;
} __sdmmc_response6_bits;

/* Present State Register */
typedef struct {
  __REG32  COMMAND_INHIBIT_C        : 1;
  __REG32  COMMAND_INHIBIT_D        : 1;
  __REG32  DAT_LINE_ACTIVE          : 1;
  __REG32                           : 5;
  __REG32  WRITE_TRANSFER_A         : 1;
  __REG32  READ_TRANSFER_A          : 1;
  __REG32  BUFFER_WRITE_ENA         : 1;
  __REG32  BUFFER_READ_ENA          : 1;
  __REG32                           : 4;
  __REG32  CARD_INSERTED            : 1;
  __REG32  CARD_STATE_STABLE        : 1;
  __REG32  CARD_DETECT_PIN_LEVEL    : 1;
  __REG32  WRITE_PROTECT_PIN_LEVEL  : 1;
  __REG32  DAT_3_0_PIN_LEVEL        : 4;
  __REG32  CMD_LINE_LEVEL           : 1;
  __REG32  DAT_7_4_PIN_LEVEL        : 4;
  __REG32                           : 3;
} __sdmmc_present_state_bits;

/* Host Control Register */
typedef struct {
  __REG8  LED_CTRL              : 1;
  __REG8  DATA_TX_WIDTH         : 1;
  __REG8  HIGH_SPEED_EN         : 1;
  __REG8                        : 3;
  __REG8  CARD_DETECT_TEST_LEVEL: 1;
  __REG8  CARD_DET_SIGNAL_DETECT: 1;
} __sdmmc_host_ctrl_bits;

/* Power Control Register */
typedef struct {
  __REG8  SD_BUS_POWER          : 1;
  __REG8  SD_BUS_VOLTAGE_SEL    : 3;
  __REG8  HARDWARE_RESET        : 1;
  __REG8                        : 3;
} __sdmmc_power_ctrl_bits;

/* Block Gap Control Register */
typedef struct {
  __REG8  STOP_AT_BLOCK_GAP     : 1;
  __REG8  CONTINUE_REQ          : 1;
  __REG8  READ_WAIT_CTRL        : 1;
  __REG8  INT_AT_BLOCK_GAP      : 1;
  __REG8  SPI_MODE              : 1;
  __REG8                        : 3;
} __sdmmc_block_gap_ctrl_bits;

/* Wake-up Control Register */
typedef struct {
  __REG8  WAKEUP_EVENT_EN_INT   : 1;
  __REG8  WAKEUP_EVENT_EN_INS   : 1;
  __REG8  WAKEUP_EVENT_EN_REM   : 1;
  __REG8                        : 5;
} __sdmmc_wakeup_ctrl_bits;

/* Clock Control Register */
typedef struct {
  __REG16 INTERNAL_CLOCK_EN     : 1;
  __REG16 INTERNAL_CLOCK_STABLE : 1;
  __REG16 SDCLOCK_EN            : 1;
  __REG16                       : 5;
  __REG16 SDCLK_FREQ_SEL        : 8;
} __sdmmc_clock_ctrl_bits;

/* Wake-up Control Register */
typedef struct {
  __REG8  DAT_TIMEOUT_CNT_VAL   : 4;
  __REG8                        : 4;
} __sdmmc_timeout_ctrl_bits;

/* Wake-up Control Register */
typedef struct {
  __REG8  SW_RST_ALL            : 1;
  __REG8  SW_RST_CMD_LINE       : 1;
  __REG8  SW_RST_DAT_LINE       : 1;
  __REG8                        : 5;
} __sdmmc_sw_reset_bits;

/* Normal Interrupt Status Register */
typedef struct {
  __REG16 CMD_COMPLETE          : 1;
  __REG16 TX_COMPLETE           : 1;
  __REG16 BLOCK_GAP_EVENT       : 1;
  __REG16                       : 1;
  __REG16 BUFF_WRITE_READY      : 1;
  __REG16 BUFF_READ_READY       : 1;
  __REG16 CARD_INS              : 1;
  __REG16 CARD_REMOVAL          : 1;
  __REG16 CARD_INT              : 1;
  __REG16                       : 6;
  __REG16 ERR_INT               : 1;
} __sdmmc_int_status_norm_bits;

/* Error Interrupt Status Register */
typedef struct {
  __REG16 CMD_TIMEOUT_ERR       : 1;
  __REG16 CMD_CRC_ERR           : 1;
  __REG16 CMD_END_BIT_ERR       : 1;
  __REG16 CMD_IND_ERR           : 1;
  __REG16 DATA_TIMEOUT_ERR      : 1;
  __REG16 DATA_CRC_ERR          : 1;
  __REG16 DATA_END_BIT_ERR      : 1;
  __REG16 CURRENT_LIMIT_ERR     : 1;
  __REG16 ACMD_ERR              : 1;
  __REG16                       : 4;
  __REG16 CEATA_ERR             : 1;
  __REG16                       : 2;
} __sdmmc_int_status_err_bits;

/* Normal Interrupt Status Enable Register */
typedef struct {
  __REG16 CMD_COMPLETE_EN       : 1;
  __REG16 TX_COMPLETE_EN        : 1;
  __REG16 BLOCK_GAP_EVENT       : 1;
  __REG16                       : 1;
  __REG16 BUFF_WRITE_READY      : 1;
  __REG16 BUFF_READ_READY       : 1;
  __REG16 CARD_INS_EN           : 1;
  __REG16 CARD_REMOVAL_EN       : 1;
  __REG16 CARD_INT_EN           : 1;
  __REG16                       : 6;
  __REG16 FIXED_TO_0            : 1;
} __sdmmc_en_int_status_norm_bits;

/* Error Interrupt Status Enable Register */
typedef struct {
  __REG16 CMD_TIMEOUT_ERR_EN    : 1;
  __REG16 CMD_CRC_ERR_EN        : 1;
  __REG16 CMD_END_BIT_ERR_EN    : 1;
  __REG16 CMD_IND_ERR_EN        : 1;
  __REG16 DATA_TIMEOUT_ERR_EN   : 1;
  __REG16 DATA_CRC_ERR_EN       : 1;
  __REG16 DATA_END_BIT_ERR_EN   : 1;
  __REG16 CURRENT_LIMIT_ERR_EN  : 1;
  __REG16 ACMD_ERR_EN           : 1;
  __REG16                       : 3;
  __REG16 TARGET_RESP_ERR_EN    : 1;
  __REG16 CEATA_ERR_EN          : 1;
  __REG16 VSES1514_EN           : 2;
} __sdmmc_en_int_status_err_bits;

/* Normal Interrupt Signal Enable Register */
typedef struct {
  __REG16 CMD_COMPLETE_EN       : 1;
  __REG16 TX_COMPLETE_EN        : 1;
  __REG16 BLOCK_GAP_EVENT       : 1;
  __REG16                       : 1;
  __REG16 BUFF_WRITE_READY      : 1;
  __REG16 BUFF_READ_READY       : 1;
  __REG16 CARD_INS_EN           : 1;
  __REG16 CARD_REMOVAL_EN       : 1;
  __REG16 CARD_INT_EN           : 1;
  __REG16                       : 6;
  __REG16 FIXED_TO_0            : 1;
} __sdmmc_en_int_signal_norm_bits;

/* Error Interrupt Signal Enable Register */
typedef struct {
  __REG16 CMD_TIMEOUT_ERR_EN    : 1;
  __REG16 CMD_CRC_ERR_EN        : 1;
  __REG16 CMD_END_BIT_ERR_EN    : 1;
  __REG16 CMD_IND_ERR_EN        : 1;
  __REG16 DATA_TIMEOUT_ERR_EN   : 1;
  __REG16 DATA_CRC_ERR_EN       : 1;
  __REG16 DATA_END_BIT_ERR_EN   : 1;
  __REG16 CURRENT_LIMIT_ERR_EN  : 1;
  __REG16 ACMD_ERR_EN           : 1;
  __REG16                       : 3;
  __REG16 TARGET_RESP_ERR_EN    : 1;
  __REG16 CEATA_ERR_EN          : 1;
  __REG16                       : 2;
} __sdmmc_en_int_signal_err_bits;

/* Auto CMD Error Status Register */
typedef struct {
  __REG16 ACMD12_NOT_EXEC_ERR         : 1;
  __REG16 ACMD_TIMEOUT_ERR            : 1;
  __REG16 ACMD_CRC_ERR                : 1;
  __REG16 ACMD_END_BIT_ERR            : 1;
  __REG16 ACMD_IND_ERR                : 1;
  __REG16                             : 2;
  __REG16 CMD_NOT_ISSUED_BY_ACMD12_ERR: 1;
  __REG16                             : 8;
} __sdmmc_acmd_err_status_bits;

/* Force Event Register for Auto CMD Error Status */
typedef struct {
  __REG16 FE_ACMD_NOT_EXEC                : 1;
  __REG16 FE_ACMD_TIMEOUT_ERR             : 1;
  __REG16 FE_ACMD_CRC_ERR                 : 1;
  __REG16 FE_ACMD_END_BIT_ERR             : 1;
  __REG16 FE_ACMD_IND_ERR                 : 1;
  __REG16                                 : 2;
  __REG16 FE_CMD_NOT_ISSUED_BY_ACMD12_ERR : 1;
  __REG16                                 : 8;
} __sdmmc_force_event_acmd_err_status_bits;

/* Error Interrupt Signal Enable Register */
typedef struct {
  __REG16 FE_CMD_TIMEOUT_ERR      : 1;
  __REG16 FE_CMD_CRC_ERR_EN       : 1;
  __REG16 FE_CMD_END_BIT_ERR_EN   : 1;
  __REG16 FE_CMD_IND_ERR_EN       : 1;
  __REG16 FE_DATA_TIMEOUT_ERR_EN  : 1;
  __REG16 FE_DATA_CRC_ERR_EN      : 1;
  __REG16 FE_DATA_END_BIT_ERR_EN  : 1;
  __REG16 FE_CURRENT_LIMIT_ERR_EN : 1;
  __REG16 FE_ACMD_ERR_EN          : 1;
  __REG16                         : 3;
  __REG16 FE_TARGET_RESP_ERR_EN   : 1;
  __REG16 FE_CEATA_ERR_EN         : 1;
  __REG16                         : 2;
} __sdmmc_force_event_err_status_bits;

/* Debug Selection Register */
typedef struct {
  __REG32 DEBUG_SEL           : 1;
  __REG32                     :31;
} __sdmmc_debug_sel_bits;

/* SPI Interrupt Support Register */
typedef struct {
  __REG32 SPI_INT_SUPPORT     : 8;
  __REG32                     :24;
} __sdmmc_spi_bits;

/* Slot Interrupt Status Register */
typedef struct {
  __REG16 SLOT_INT_STATUS     : 8;
  __REG16                     : 8;
} __sdmmc_slot_int_status_bits;

/* ETH0_MAC_Configuration */
typedef struct {
  __REG32 PRELEN              : 2;
  __REG32 RE                  : 1;
  __REG32 TE                  : 1;
  __REG32 DC                  : 1;
  __REG32 BL                  : 2;
  __REG32 ACS                 : 1;
  __REG32 LUD                 : 1;
  __REG32 DR                  : 1;
  __REG32 IPC                 : 1;
  __REG32 DM                  : 1;
  __REG32 LM                  : 1;
  __REG32 DO                  : 1;
  __REG32 FES                 : 1;
  __REG32                     : 1;
  __REG32 DCRS                : 1;
  __REG32 IFG                 : 3;
  __REG32 JE                  : 1;
  __REG32 BE                  : 1;
  __REG32 JD                  : 1;
  __REG32 WD                  : 1;
  __REG32 TC                  : 1;
  __REG32 CST                 : 1;
  __REG32                     : 1;
  __REG32 _2KPE               : 1;
  __REG32 SARC                : 3;
  __REG32                     : 1;
} __eth_mac_configuration_bits;

/* ETH0_MAC_Frame_Filter */
typedef struct {
  __REG32 PR                  : 1;
  __REG32 HUC                 : 1;
  __REG32 HMC                 : 1;
  __REG32 DAIF                : 1;
  __REG32 PM                  : 1;
  __REG32 DBF                 : 1;
  __REG32 PCF                 : 2;
  __REG32 SAIF                : 1;
  __REG32 SAF                 : 1;
  __REG32 HPF                 : 1;
  __REG32                     : 5;
  __REG32 VTFE                : 1;
  __REG32                     : 3;
  __REG32 IPFE                : 1;
  __REG32 DNTU                : 1;
  __REG32                     : 9;
  __REG32 RA                  : 1;
} __eth_mac_frame_filter_bits;

/* ETH0_MAC_Frame_Filter */
typedef struct {
  __REG32 MB                  : 1;
  __REG32 MW                  : 1;
  __REG32 CR                  : 4;
  __REG32 MR                  : 5;
  __REG32 PA                  : 5;
  __REG32                     :16;
} __eth_mii_address_bits;

/* ETH0_MII_Data */
typedef struct {
  __REG32 MD                  :16;
  __REG32                     :16;
} __eth_mii_data_bits;

/* ETH0_Flow_Control */
typedef struct {
  __REG32 FCA_BPA             : 1;
  __REG32 TFE                 : 1;
  __REG32 RFE                 : 1;
  __REG32 UP                  : 1;
  __REG32 PLT                 : 2;
  __REG32                     : 1;
  __REG32 DZPQ                : 1;
  __REG32                     : 8;
  __REG32 PT                  :16;
} __eth_flow_control_bits;

/* ETH0_VLAN_Tag */
typedef struct {
  __REG32 VL                  :16;
  __REG32 ETV                 : 1;
  __REG32 VTIM                : 1;
  __REG32 ESVL                : 1;
  __REG32 VTHM                : 1;
  __REG32                     :12;
} __eth_vlan_tag_bits;

/* ETH0_Version */
typedef struct {
  __REG32 SNPSVER             : 8;
  __REG32 USERVER             : 8;
  __REG32                     :16;
} __eth_version_bits;

/* ETH0_Debug */
typedef struct {
  __REG32 RPESTS              : 1;
  __REG32 RFCFCSTS            : 2;
  __REG32                     : 1;
  __REG32 RWCSTS              : 1;
  __REG32 RRCSTS              : 2;
  __REG32                     : 1;
  __REG32 RXFSTS              : 2;
  __REG32                     : 6;
  __REG32 TPESTS              : 1;
  __REG32 TFCSTS              : 2;
  __REG32 TXPAUSED            : 1;
  __REG32 TRCSTS              : 2;
  __REG32 TWCSTS              : 1;
  __REG32                     : 1;
  __REG32 TXFSTS              : 1;
  __REG32 TXSTSFSTS           : 1;
  __REG32                     : 6;
} __eth_debug_bits;

/* ETH0_PMT_Control_Status */
typedef struct {
  __REG32 PWRDWN              : 1;
  __REG32 MGKPKTEN            : 1;
  __REG32 RWKPKTEN            : 1;
  __REG32                     : 2;
  __REG32 MGKPRCVD            : 1;
  __REG32 RWKPRCVD            : 1;
  __REG32                     : 2;
  __REG32 GLBLUCAST           : 1;
  __REG32                     :21;
  __REG32 RWKFILTRST          : 1;
} __eth_pmt_control_status_bits;

/* ETH0_Interrupt_Status */
typedef struct {
  __REG32 RMIIIS              : 1;
  __REG32                     : 2;
  __REG32 PMTIS               : 1;
  __REG32 MMCIS               : 1;
  __REG32 MMCRXIS5            : 1;
  __REG32 MMCTXIS             : 1;
  __REG32 MMCRXIPIS           : 1;
  __REG32                     : 1;
  __REG32 TSIS                : 1;
  __REG32 LPIIS               : 1;
  __REG32                     :21;
} __eth_interrupt_status_bits;

/* ETH0_MAC_Address0_High */
typedef struct {
  __REG32 ADDRHI              :16;
  __REG32                     :15;
  __REG32 AE                  : 1;
} __eth_mac_address0_high_bits;

/* ETH0_MAC_Address1_High */
typedef struct {
  __REG32 ADDRHI              :16;
  __REG32                     : 8;
  __REG32 MBC                 : 6;
  __REG32 SA                  : 1;
  __REG32 AE                  : 1;
} __eth_mac_address1_high_bits;

/* ETH0_MMC_Control */
typedef struct {
  __REG32 CNTRST              : 1;
  __REG32 CNTSTOPRO           : 1;
  __REG32 RSTONRD             : 1;
  __REG32 CNTFREEZ            : 1;
  __REG32 CNTPRST             : 1;
  __REG32 CNTPRSTLVL          : 1;
  __REG32                     : 2;
  __REG32 UCDBC               : 1;
  __REG32                     :23;
} __eth_mmc_control_bits;

/* ETH0_MMC_Receive_Interrupt */
typedef struct {
  __REG32 RXGBFRMIS           : 1;
  __REG32 RXGBOCTIS           : 1;
  __REG32 RXGOCTIS            : 1;
  __REG32 RXBCGFIS            : 1;
  __REG32 RXMCGFIS            : 1;
  __REG32 RXCRCERFIS          : 1;
  __REG32 RXALGNERFIS         : 1;
  __REG32 RXRUNTFIS           : 1;
  __REG32 RXJABERFIS          : 1;
  __REG32 RXUSIZEGFIS         : 1;
  __REG32 RXOSIZEGFIS         : 1;
  __REG32 RX64OCTGBFIS        : 1;
  __REG32 RX65T127OCTGBFIS    : 1;
  __REG32 RX128T255OCTGBFIS   : 1;
  __REG32 RX256T511OCTGBFIS   : 1;
  __REG32 RX512T1023OCTGBFIS  : 1;
  __REG32 RX1024TMAXOCTGBFIS  : 1;
  __REG32 RXUCGFIS            : 1;
  __REG32 RXLENERFIS          : 1;
  __REG32 RXORANGEFIS         : 1;
  __REG32 RXPAUSFIS           : 1;
  __REG32 RXFOVFIS            : 1;
  __REG32 RXVLANGBFIS         : 1;
  __REG32 RXWDOGFIS           : 1;
  __REG32 RXRCVERRFIS         : 1;
  __REG32 RXCTRLFIS           : 1;
  __REG32                     : 6;
} __eth_mmc_receive_interrupt_bits;

/* ETH0_MMC_Transmit_Interrupt */
typedef struct {
  __REG32 TXGBOCTIS           : 1;
  __REG32 TXGBFRMIS           : 1;
  __REG32 TXBCGFIS            : 1;
  __REG32 TXMCGFIS            : 1;
  __REG32 TX64OCTGBFIS        : 1;
  __REG32 TX65T127OCTGBFIS    : 1;
  __REG32 TX128T255OCTGBFIS   : 1;
  __REG32 TX256T511OCTGBFIS   : 1;
  __REG32 TX512T1023OCTGBFIS  : 1;
  __REG32 TX1024TMAXOCTGBFIS  : 1;
  __REG32 TXUCGBFIS           : 1;
  __REG32 TXMCGBFIS           : 1;
  __REG32 TXBCGBFIS           : 1;
  __REG32 TXUFLOWERFIS        : 1;
  __REG32 TXSCOLGFIS          : 1;
  __REG32 TXMCOLGFIS          : 1;
  __REG32 TXDEFFIS            : 1;
  __REG32 TXLATCOLFIS         : 1;
  __REG32 TXEXCOLFIS          : 1;
  __REG32 TXCARERFIS          : 1;
  __REG32 TXGOCTIS            : 1;
  __REG32 TXGFRMIS            : 1;
  __REG32 TXEXDEFFIS          : 1;
  __REG32 TXPAUSFIS           : 1;
  __REG32 TXVLANGFIS          : 1;
  __REG32 TXOSIZEGFIS         : 1;
  __REG32                     : 6;
} __eth_mmc_transmit_interrupt_bits;

/* ETH0_MMC_Receive_Interrupt_Mask */
typedef struct {
  __REG32 RXGBFRMIM           : 1;
  __REG32 RXGBOCTIM           : 1;
  __REG32 RXGOCTIM            : 1;
  __REG32 RXBCGFIM            : 1;
  __REG32 RXMCGFIM            : 1;
  __REG32 RXCRCERFIM          : 1;
  __REG32 RXALGNERFIM         : 1;
  __REG32 RXRUNTFIM           : 1;
  __REG32 RXJABERFIM          : 1;
  __REG32 RXUSIZEGFIM         : 1;
  __REG32 RXOSIZEGFIM         : 1;
  __REG32 RX64OCTGBFIM        : 1;
  __REG32 RX65T127OCTGBFIM    : 1;
  __REG32 RX128T255OCTGBFIM   : 1;
  __REG32 RX256T511OCTGBFIM   : 1;
  __REG32 RX512T1023OCTGBFIM  : 1;
  __REG32 RX1024TMAXOCTGBFIM  : 1;
  __REG32 RXUCGFIM            : 1;
  __REG32 RXLENERFIM          : 1;
  __REG32 RXORANGEFIM         : 1;
  __REG32 RXPAUSFIM           : 1;
  __REG32 RXFOVFIM            : 1;
  __REG32 RXVLANGBFIM         : 1;
  __REG32 RXWDOGFIM           : 1;
  __REG32 RXRCVERRFIM         : 1;
  __REG32 RXCTRLFIM           : 1;
  __REG32                     : 6;
} __eth_mmc_receive_interrupt_mask_bits;

/* ETH0_MMC_Transmit_Interrupt_Mask */
typedef struct {
  __REG32 TXGBOCTIM           : 1;
  __REG32 TXGBFRMIM           : 1;
  __REG32 TXBCGFIM            : 1;
  __REG32 TXMCGFIM            : 1;
  __REG32 TX64OCTGBFIM        : 1;
  __REG32 TX65T127OCTGBFIM    : 1;
  __REG32 TX128T255OCTGBFIM   : 1;
  __REG32 TX256T511OCTGBFIM   : 1;
  __REG32 TX512T1023OCTGBFIM  : 1;
  __REG32 TX1024TMAXOCTGBFIM  : 1;
  __REG32 TXUCGBFIM           : 1;
  __REG32 TXMCGBFIM           : 1;
  __REG32 TXBCGBFIM           : 1;
  __REG32 TXUFLOWERFIM        : 1;
  __REG32 TXSCOLGFIM          : 1;
  __REG32 TXMCOLGFIM          : 1;
  __REG32 TXDEFFIM            : 1;
  __REG32 TXLATCOLFIM         : 1;
  __REG32 TXEXCOLFIM          : 1;
  __REG32 TXCARERFIM          : 1;
  __REG32 TXGOCTIM            : 1;
  __REG32 TXGFRMIM            : 1;
  __REG32 TXEXDEFFIM          : 1;
  __REG32 TXPAUSFIM           : 1;
  __REG32 TXVLANGFIM          : 1;
  __REG32 TXOSIZEGFIM         : 1;
  __REG32                     : 6;
} __eth_mmc_transmit_interrupt_mask_bits;

/* ETH0_MMC_IPC_Receive_Interrupt_Mask */
typedef struct {
  __REG32 RXIPV4GFIM          : 1;
  __REG32 RXIPV4HERFIM        : 1;
  __REG32 RXIPV4NOPAYFIM      : 1;
  __REG32 RXIPV4FRAGFIM       : 1;
  __REG32 RXIPV4UDSBLFIM      : 1;
  __REG32 RXIPV6GFIM          : 1;
  __REG32 RXIPV6HERFIM        : 1;
  __REG32 RXIPV6NOPAYFIM      : 1;
  __REG32 RXUDPGFIM           : 1;
  __REG32 RXUDPERFIM          : 1;
  __REG32 RXTCPGFIM           : 1;
  __REG32 RXTCPERFIM          : 1;
  __REG32 RXICMPGFIM          : 1;
  __REG32 RXICMPERFIM         : 1;
  __REG32                     : 2;
  __REG32 RXIPV4GOIM          : 1;
  __REG32 RXIPV4HEROIM        : 1;
  __REG32 RXIPV4NOPAYOIM      : 1;
  __REG32 RXIPV4FRAGOIM       : 1;
  __REG32 RXIPV4UDSBLOIM      : 1;
  __REG32 RXIPV6GOIM          : 1;
  __REG32 RXIPV6HEROIM        : 1;
  __REG32 RXIPV6NOPAYOIM      : 1;
  __REG32 RXUDPGOIM           : 1;
  __REG32 RXUDPEROIM          : 1;
  __REG32 RXTCPGOIM           : 1;
  __REG32 RXTCPEROIM          : 1;
  __REG32 RXICMPGOIM          : 1;
  __REG32 RXICMPEROIM         : 1;
  __REG32                     : 2;
} __eth_mmc_ipc_receive_interrupt_mask_bits;

/* ETH0_MMC_IPC_Receive_Interrupt */
typedef struct {
  __REG32 RXIPV4GFIS          : 1;
  __REG32 RXIPV4HERFIS        : 1;
  __REG32 RXIPV4NOPAYFIS      : 1;
  __REG32 RXIPV4FRAGFIS       : 1;
  __REG32 RXIPV4UDSBLFIS      : 1;
  __REG32 RXIPV6GFIS          : 1;
  __REG32 RXIPV6HERFIS        : 1;
  __REG32 RXIPV6NOPAYFIS      : 1;
  __REG32 RXUDPGFIS           : 1;
  __REG32 RXUDPERFIS          : 1;
  __REG32 RXTCPGFIS           : 1;
  __REG32 RXTCPERFIS          : 1;
  __REG32 RXICMPGFIS          : 1;
  __REG32 RXICMPERFIS         : 1;
  __REG32                     : 2;
  __REG32 RXIPV4GOIS          : 1;
  __REG32 RXIPV4HEROIS        : 1;
  __REG32 RXIPV4NOPAYOIS      : 1;
  __REG32 RXIPV4FRAGOIS       : 1;
  __REG32 RXIPV4UDSBLOIS      : 1;
  __REG32 RXIPV6GOIS          : 1;
  __REG32 RXIPV6HEROIS        : 1;
  __REG32 RXIPV6NOPAYOIS      : 1;
  __REG32 RXUDPGOIS           : 1;
  __REG32 RXUDPEROIS          : 1;
  __REG32 RXTCPGOIS           : 1;
  __REG32 RXTCPEROIS          : 1;
  __REG32 RXICMPGOIS          : 1;
  __REG32 RXICMPEROIS         : 1;
  __REG32                     : 2;
} __eth_mmc_ipc_receive_interrupt_bits;

/* ETH0_Timestamp_Control */
typedef struct {
  __REG32 TSENA               : 1;
  __REG32 TSCFUPDT            : 1;
  __REG32 TSINIT              : 1;
  __REG32 TSUPDT              : 1;
  __REG32 TSTRIG              : 1;
  __REG32 TSADDREG            : 1;
  __REG32                     : 2;
  __REG32 TSENALL             : 1;
  __REG32 TSCTRLSSR           : 1;
  __REG32 TSVER2ENA           : 1;
  __REG32 TSIPENA             : 1;
  __REG32 TSIPV6ENA           : 1;
  __REG32 TSIPV4ENA           : 1;
  __REG32 TSEVNTENA           : 1;
  __REG32 TSMSTRENA           : 1;
  __REG32 SNAPTYPSEL          : 2;
  __REG32 TSENMACADDR         : 1;
  __REG32                     : 5;
  __REG32 ATSFC               : 1;
  __REG32 ATSEN0              : 1;
  __REG32 ATSEN1              : 1;
  __REG32 ATSEN2              : 1;
  __REG32 ATSEN3              : 1;
  __REG32                     : 3;
} __eth_timestamp_control_bits;

/* ETH0_Sub_Second_Increment */
typedef struct {
  __REG32 SSINC               : 8;
  __REG32                     :24;
} __eth_sub_second_increment_bits;

/* ETH0_System_Time_Nanoseconds */
typedef struct {
  __REG32 TSSS                :31;
  __REG32                     : 1;
} __eth_system_time_nanoseconds_bits;

/* ETH0_System_Time_Nanoseconds_Update */
typedef struct {
  __REG32 TSSS                :31;
  __REG32 ADDSUB              : 1;
} __eth_system_time_nanoseconds_update_bits;

/* ETH0_Target_Time_Nanoseconds */
typedef struct {
  __REG32 TTSLO               :31;
  __REG32 TRGTBUSY            : 1;
} __eth_target_time_nanoseconds_bits;

/* ETH0_System_Time_Higher_Word_Seconds */
typedef struct {
  __REG32 TSHWR               :16;
  __REG32                     :16;
} __eth_system_time_higher_word_seconds_bits;

/* ETH0_Timestamp_Status */
typedef struct {
  __REG32 TSSOVF              : 1;
  __REG32 TSTARGT             : 1;
  __REG32 AUXTSTRIG           : 1;
  __REG32 TSTRGTERR           : 1;
  __REG32 TSTARGT1            : 1;
  __REG32 TSTRGTERR1          : 1;
  __REG32 TSTARGT2            : 1;
  __REG32 TSTRGTERR2          : 1;
  __REG32 TSTARGT3            : 1;
  __REG32 TSTRGTERR3          : 1;
  __REG32                     : 6;
  __REG32 ATSSTN              : 4;
  __REG32                     : 4;
  __REG32 ATSSTM              : 1;
  __REG32 ATSNS               : 5;
  __REG32                     : 2;
} __eth_timestamp_status_bits;

/* ETH0_PPS_Control */
typedef struct {
  __REG32 PPSCTRL_PPSCMD      : 4;
  __REG32 PPSEN0              : 1;
  __REG32 TRGTMODSEL0         : 2;
  __REG32                     : 1;
  __REG32 PPSCMD1             : 3;
  __REG32                     : 2;
  __REG32 TRGTMODSEL1         : 2;
  __REG32                     : 1;
  __REG32 PPSCMD2             : 3;
  __REG32                     : 2;
  __REG32 TRGTMODSEL2         : 2;
  __REG32                     : 1;
  __REG32 PPSCMD3             : 3;
  __REG32                     : 2;
  __REG32 TRGTMODSEL3         : 2;
  __REG32                     : 1;
} __eth_pps_control_bits;

/* ETH0_Bus_Mode */
typedef struct {
  __REG32 SWR                 : 1;
  __REG32 DA                  : 1;
  __REG32 DSL                 : 5;
  __REG32 ATDS                : 1;
  __REG32 PBL                 : 6;
  __REG32 PR                  : 2;
  __REG32 FB                  : 1;
  __REG32 RPBL                : 6;
  __REG32 USP                 : 1;
  __REG32 _8xPBL              : 1;
  __REG32 AAL                 : 1;
  __REG32 MB                  : 1;
  __REG32 TXPR                : 1;
  __REG32 PRWG                : 2;
  __REG32                     : 2;
} __eth_bus_mode_bits;

/* ETH0_Status */
typedef struct {
  __REG32 TI                  : 1;
  __REG32 TPS                 : 1;
  __REG32 TU                  : 1;
  __REG32 TJT                 : 1;
  __REG32 OVF                 : 1;
  __REG32 UNF                 : 1;
  __REG32 RI                  : 1;
  __REG32 RU                  : 1;
  __REG32 RPS                 : 1;
  __REG32 RWT                 : 1;
  __REG32 ETI                 : 1;
  __REG32                     : 2;
  __REG32 FBI                 : 1;
  __REG32 ERI                 : 1;
  __REG32 AIS                 : 1;
  __REG32 NIS                 : 1;
  __REG32 RS                  : 3;
  __REG32 TS                  : 3;
  __REG32 EB                  : 3;
  __REG32 ELI                 : 1;
  __REG32 EMI                 : 1;
  __REG32 EPI                 : 1;
  __REG32 TTI                 : 1;
  __REG32 ELPII               : 1;
  __REG32                     : 1;
} __eth_status_bits;

/* ETH0_Operation_Mode */
typedef struct {
  __REG32                     : 1;
  __REG32 SR                  : 1;
  __REG32 OSF                 : 1;
  __REG32 RTC                 : 2;
  __REG32                     : 1;
  __REG32 FUF                 : 1;
  __REG32 FEF                 : 1;
  __REG32 EFC                 : 1;
  __REG32 RFA                 : 2;
  __REG32 RFD                 : 2;
  __REG32 ST                  : 1;
  __REG32 TTC                 : 3;
  __REG32                     : 3;
  __REG32 FTF                 : 1;
  __REG32 TSF                 : 1;
  __REG32 _RFD                : 1;
  __REG32 _RFA                : 1;
  __REG32 DFF                 : 1;
  __REG32 RSF                 : 1;
  __REG32 DT                  : 1;
  __REG32                     : 5;
} __eth_operation_mode_bits;

/* ETH0_Interrupt_Enable */
typedef struct {
  __REG32 TIE                 : 1;
  __REG32 TSE                 : 1;
  __REG32 TUE                 : 1;
  __REG32 TJE                 : 1;
  __REG32 OVE                 : 1;
  __REG32 UNE                 : 1;
  __REG32 RIE                 : 1;
  __REG32 RUE                 : 1;
  __REG32 RSE                 : 1;
  __REG32 RWE                 : 1;
  __REG32 ETE                 : 1;
  __REG32                     : 2;
  __REG32 FBE                 : 1;
  __REG32 ERE                 : 1;
  __REG32 AIE                 : 1;
  __REG32 NIE                 : 1;
  __REG32                     :15;
} __eth_interrupt_enable_bits;

/* ETH0_Missed_Frame_And_Buffer_Overflow_Counter */
typedef struct {
  __REG32 MISFRMCNT           :16;
  __REG32 MISCNTOVF           : 1;
  __REG32 OVFFRMCNT           :11;
  __REG32 OVFCNTOVF           : 1;
  __REG32                     : 3;
} __eth_missed_frame_and_buffer_overflow_counter_bits;

/* ETH0_Receive_Interrupt_Watchdog_Timer */
typedef struct {
  __REG32 RIWT                : 8;
  __REG32                     :24;
} __eth_receive_interrupt_watchdog_timer_bits;

/* ETH0_AHB_Status */
typedef struct {
  __REG32 AHBMSS              : 1;
  __REG32                     :31;
} __eth_ahb_status_bits;

/* ETH0_HW_Feature */
typedef struct {
  __REG32 MIISEL              : 1;
  __REG32 GMIISEL             : 1;
  __REG32 HDSEL               : 1;
  __REG32 EXTHASHEN           : 1;
  __REG32 HASHSEL             : 1;
  __REG32 ADDMACADRSEL        : 1;
  __REG32 PCSSEL              : 1;
  __REG32 L3L4FLTREN          : 1;
  __REG32 SMASEL              : 1;
  __REG32 RWKSEL              : 1;
  __REG32 MGKSEL              : 1;
  __REG32 MMCSEL              : 1;
  __REG32 TSVER1SEL           : 1;
  __REG32 TSVER2SEL           : 1;
  __REG32 EEESEL              : 1;
  __REG32 AVSEL               : 1;
  __REG32 TXCOESEL            : 1;
  __REG32 RXTYP1COE           : 1;
  __REG32 RXTYP2COE           : 1;
  __REG32 RXFIFOSIZE          : 1;
  __REG32 RXCHCNT             : 2;
  __REG32 TXCHCNT             : 2;
  __REG32 ENHDESSEL           : 1;
  __REG32 INTTSEN             : 1;
  __REG32 FLEXIPPSEN          : 1;
  __REG32 SAVLANINS           : 1;
  __REG32 ACTPHYIF            : 3;
  __REG32                     : 1;
} __eth_hw_feature_bits;

/* Control and Status Register (GOTGCTL) */
typedef struct {
  __REG32 SesReqScs           : 1;
  __REG32 SesReq              : 1;
  __REG32 VbvalidOvEn         : 1;
  __REG32 VbvalidOvVal        : 1;
  __REG32 AvalidOvEn          : 1;
  __REG32 AvalidOvVal         : 1;
  __REG32 BvalidOvEn          : 1;
  __REG32 BvalidOvVal         : 1;
  __REG32 HstNegScs           : 1;
  __REG32 HNPReq              : 1;
  __REG32 HstSetHNPEn         : 1;
  __REG32 DevHNPEn            : 1;
  __REG32                     : 4;
  __REG32 ConlDSts            : 1;
  __REG32 DbncTime            : 1;
  __REG32 ASesVId             : 1;
  __REG32 BSesVld             : 1;
  __REG32 OTGVer              : 1;
  __REG32                     :11;
} __usb_gotgctl_bits;

/* Interrupt Register (GOTGINT) */
typedef struct {
  __REG32                     : 2;
  __REG32 SesEndDet           : 1;
  __REG32                     : 5;
  __REG32 SesReqSucStsChng    : 1;
  __REG32 HstNegSucStsChng    : 1;
  __REG32                     : 7;
  __REG32 HstNegDet           : 1;
  __REG32 ADevTOUTChg         : 1;
  __REG32 DbnceDone           : 1;
  __REG32                     :12;
} __usb_gotgint_bits;

/* AHB Configuration Register (GAHBCFG) */
typedef struct {
  __REG32 GlblIntrMsk         : 1;
  __REG32 HBstLen             : 4;
  __REG32 DMAEn               : 1;
  __REG32                     : 1;
  __REG32 NPTxFEmpLvl         : 1;
  __REG32 PTxFEmpLvl          : 1;
  __REG32                     :12;
  __REG32 RemMemSupp          : 1;
  __REG32 NotiAllDmaWrit      : 1;
  __REG32                     : 9;
} __usb_gahbcfg_bits;

/* USB Configuration Register (GUSBCFG) */
typedef struct {
  __REG32 TOutCal             : 3;
  __REG32 PHYIf               : 1;
  __REG32                     : 1;
  __REG32 FSIntf              : 1;
  __REG32 PHYSel              : 1;
  __REG32                     : 1;
  __REG32 SRPCap              : 1;
  __REG32 HNPCap              : 1;
  __REG32 USBTrdTim           : 4;
  __REG32                     : 1;
  __REG32 PhyLPwrClkSel       : 1;
  __REG32 OtgI2CSel           : 1;
  __REG32                     :11;
  __REG32 TxEndDelay          : 1;
  __REG32 ForceHstMode        : 1;
  __REG32 ForceDevMode        : 1;
  __REG32 CTP                 : 1;
} __usb_gusbcfg_bits;

/* Reset Register (GRSTCTL) */
typedef struct {
  __REG32 CSftRst             : 1;
  __REG32                     : 1;
  __REG32 FrmCntrRst          : 1;
  __REG32 INTknQFlsh          : 1;
  __REG32 RxFFlsh             : 1;
  __REG32 TxFFlsh             : 1;
  __REG32 TxFNum              : 5;
  __REG32                     :19;
  __REG32 DMAReq              : 1;
  __REG32 AHBIdle             : 1;
} __usb_grstctl_bits;

/* Interrupt Register (GINTSTS) */
typedef union {
  /* USB_GINTSTS */
  /* USBH_GINTSTS */
  struct {
  __REG32 CurMod              : 1;
  __REG32 ModeMis             : 1;
  __REG32 OTGInt              : 1;
  __REG32 Sof                 : 1;
  __REG32 RxFLvl              : 1;
  __REG32                     :16;
  __REG32 incomplP            : 1;
  __REG32                     : 2;
  __REG32 PrtInt              : 1;
  __REG32 HChInt              : 1;
  __REG32 PTxFEmp             : 1;
  __REG32                     : 1;
  __REG32 ConIDStsChng        : 1;
  __REG32 DisconnInt          : 1;
  __REG32 SessReqInt          : 1;
  __REG32 WkUpInt             : 1;
  };
  /* USBD_GINTSTS */
  struct {
  __REG32 CurMod              : 1;
  __REG32 ModeMis             : 1;
  __REG32 OTGInt              : 1;
  __REG32 Sof                 : 1;
  __REG32 RxFLvl              : 1;
  __REG32                     : 1;
  __REG32 GINNakEff           : 1;
  __REG32 GOUTNakEff          : 1;
  __REG32                     : 2;
  __REG32 ErlySusp            : 1;
  __REG32 USBSusp             : 1;
  __REG32 USBRst              : 1;
  __REG32 EnumDone            : 1;
  __REG32 ISOOutDrop          : 1;
  __REG32 EOPF                : 1;
  __REG32                     : 2;
  __REG32 IEPInt              : 1;
  __REG32 OEPInt              : 1;
  __REG32 incompISOIN         : 1;
  __REG32 incomplSOOUT        : 1;
  __REG32 FetSusp             : 1;
  __REG32 ResetDet            : 1;
  __REG32                     : 4;
  __REG32 ConIDStsChng        : 1;
  __REG32 DisconnInt          : 1;
  __REG32 SessReqInt          : 1;
  __REG32 WkUpInt             : 1;
  } dev;
} __usb_gintsts_bits;

/* Interrupt Mask Register (GINTMSK) */
typedef union {
  /* USB_GINTMSK */
  /* USBH_GINTMSK */
  struct {
  __REG32                     : 1;
  __REG32 ModeMisMsk          : 1;
  __REG32 OTGIntMsk           : 1;
  __REG32 SofMsk              : 1;
  __REG32 RxFLvlMsk           : 1;
  __REG32                     :16;
  __REG32 incomplPMsk         : 1;
  __REG32                     : 2;
  __REG32 PrtIntMsk           : 1;
  __REG32 HChIntMsk           : 1;
  __REG32 PTxFEmpMsk          : 1;
  __REG32                     : 1;
  __REG32 ConIDStsChngMsk     : 1;
  __REG32 DisconnIntMsk       : 1;
  __REG32 SessReqIntMsk       : 1;
  __REG32 WkUpIntMsk          : 1;
  };
  /* USBD_GINTMSK */
  struct {
  __REG32                     : 1;
  __REG32 ModeMisMsk          : 1;
  __REG32 OTGIntMsk           : 1;
  __REG32 SofMsk              : 1;
  __REG32 RxFLvlMsk           : 1;
  __REG32                     : 1;
  __REG32 GINNakEffMsk        : 1;
  __REG32 GOUTNakEffMsk       : 1;
  __REG32                     : 2;
  __REG32 ErlySuspMsk         : 1;
  __REG32 USBSuspMsk          : 1;
  __REG32 USBRstMsk           : 1;
  __REG32 EnumDoneMsk         : 1;
  __REG32 ISOOutDropMsk       : 1;
  __REG32 EOPFMsk             : 1;
  __REG32                     : 2;
  __REG32 IEPIntMsk           : 1;
  __REG32 OEPIntMsk           : 1;
  __REG32 incompISOINMsk      : 1;
  __REG32 incomplSOOUTMsk     : 1;
  __REG32 FetSuspMsk          : 1;
  __REG32 ResetDetMsk         : 1;
  __REG32                     : 4;
  __REG32 ConIDStsChngMsk     : 1;
  __REG32 DisconnIntMsk       : 1;
  __REG32 SessReqIntMsk       : 1;
  __REG32 WkUpIntMsk          : 1;
  } dev;
} __usb_gintmsk_bits;

/* Receive Status Debug Read/Status Read and Pop Registers (GRXSTSR/GRXSTSP) */
typedef struct {
  __REG32 ChNum               : 4;
  __REG32 BCnt                :11;
  __REG32 DPID                : 2;
  __REG32 PktSts              : 4;
  __REG32 FN                  : 4;
  __REG32                     : 7;
} __usb_grxstsr_bits;

/* Receive FIFO Size Register (GRXFSIZ) */
typedef struct {
  __REG32 RxFDep              :16;
  __REG32                     :16;
} __usb_grxfsiz_bits;

/* Non-Periodic Transmit FIFO Size Register (GNPTXFSIZ) */
typedef union {
  /* USB_GNPTXFSIZ */
  /* USBH_GNPTXFSIZ */
  struct {
  __REG32 NPTxFStAddr         :16;
  __REG32 NPTxFDep            :16;
  };
  /* USBD_GNPTXFSIZ */
  struct {
  __REG32                     :16;
  __REG32 INEPTxF0Dep         :16;
  } dev;
} __usb_gnptxfsiz_bits;

/* Non-Periodic Transmit FIFO/Queue Status Register (GNPTXSTS) */
typedef struct {
  __REG32 NPTxFSpcAvail       :16;
  __REG32 NPTxQSpcAvail       : 8;
  __REG32 NPTxQTop            : 7;
  __REG32                     : 1;
} __usb_gnptxsts_bits;

/* General Purpose Input/Output Register (GGPIO) */
typedef struct {
  __REG32 GPI0                : 1;
  __REG32 GPI1                : 1;
  __REG32 GPI2                : 1;
  __REG32 GPI3                : 1;
  __REG32 GPI4                : 1;
  __REG32 GPI5                : 1;
  __REG32 GPI6                : 1;
  __REG32 GPI7                : 1;
  __REG32 GPI8                : 1;
  __REG32 GPI9                : 1;
  __REG32 GPI10               : 1;
  __REG32 GPI11               : 1;
  __REG32 GPI12               : 1;
  __REG32 GPI13               : 1;
  __REG32 GPI14               : 1;
  __REG32 GPI15               : 1;
  __REG32 GPO0                : 1;
  __REG32 GPO1                : 1;
  __REG32 GPO2                : 1;
  __REG32 GPO3                : 1;
  __REG32 GPO4                : 1;
  __REG32 GPO5                : 1;
  __REG32 GPO6                : 1;
  __REG32 GPO7                : 1;
  __REG32 GPO8                : 1;
  __REG32 GPO9                : 1;
  __REG32 GPO10               : 1;
  __REG32 GPO11               : 1;
  __REG32 GPO12               : 1;
  __REG32 GPO13               : 1;
  __REG32 GPO14               : 1;
  __REG32 GPO15               : 1;
} __usb_ggpio_bits;

/* Global DFIFO Software Config Register (GDFIFOCFG) */
typedef struct {
  __REG32 GDFIFOCfg           :16;
  __REG32 EPInfoBaseAddr      :16;
} __usb_gdfifocfg_bits;

/* Host Periodic Transmit FIFO Size Register (HPTXFSIZ) */
typedef struct {
  __REG32 PTxFStAddr          :16;
  __REG32 PTxFSize            :16;
} __usb_hptxfsiz_bits;

/* Device IN Endpoint Transmit FIFO Size Register (DIEPTXFn) */
typedef struct {
  __REG32 INEPnTxFStAddr      :16;
  __REG32 INEPnTxFDep         :16;
} __usb_dieptxf_bits;

/* Host Configuration Register (HCFG) */
typedef struct {
  __REG32 FSLSPclkSel         : 2;
  __REG32 FSLSSupp            : 1;
  __REG32                     : 4;
  __REG32 Ena32KHzS           : 1;
  __REG32                     :15;
  __REG32 DescDMA             : 1;
  __REG32 FrListEn            : 2;
  __REG32 PerSchedEna         : 1;
  __REG32                     : 5;
} __usb_hcfg_bits;

/* Host Frame Interval Register (HFIR) */
typedef struct {
  __REG32 FrInt               :16;
  __REG32                     :16;
} __usb_hfir_bits;

/* Host Frame Number/Frame Time Remaining Register (HFNUM) */
typedef struct {
  __REG32 FrNum               :16;
  __REG32 FrRem               :16;
} __usb_hfnum_bits;

/* Host Periodic Transmit FIFO/Queue Status Register (HPTXSTS) */
typedef struct {
  __REG32 PTxFSpcAvail        :16;
  __REG32 PTxQSpcAvail        : 8;
  __REG32 PTxQTop             : 8;
} __usb_hptxsts_bits;

/* Host All Channels Interrupt Register (HAINT) */
typedef struct {
  __REG32 HAINT0              : 1;
  __REG32 HAINT1              : 1;
  __REG32 HAINT2              : 1;
  __REG32 HAINT3              : 1;
  __REG32 HAINT4              : 1;
  __REG32 HAINT5              : 1;
  __REG32 HAINT6              : 1;
  __REG32 HAINT7              : 1;
  __REG32 HAINT8              : 1;
  __REG32 HAINT9              : 1;
  __REG32 HAINT10             : 1;
  __REG32 HAINT11             : 1;
  __REG32 HAINT12             : 1;
  __REG32 HAINT13             : 1;
  __REG32                     :18;
} __usb_haint_bits;

/* Host All Channels Interrupt Mask Register (HAINTMSK) */
typedef struct {
  __REG32 HAINTMsk0           : 1;
  __REG32 HAINTMsk1           : 1;
  __REG32 HAINTMsk2           : 1;
  __REG32 HAINTMsk3           : 1;
  __REG32 HAINTMsk4           : 1;
  __REG32 HAINTMsk5           : 1;
  __REG32 HAINTMsk6           : 1;
  __REG32 HAINTMsk7           : 1;
  __REG32 HAINTMsk8           : 1;
  __REG32 HAINTMsk9           : 1;
  __REG32 HAINTMsk10          : 1;
  __REG32 HAINTMsk11          : 1;
  __REG32 HAINTMsk12          : 1;
  __REG32 HAINTMsk13          : 1;
  __REG32                     :18;
} __usb_haintmsk_bits;

/* Host Port Control and Status Register (HPRT) */
typedef struct {
  __REG32 PrtConnSts          : 1;
  __REG32 PrtConnDet          : 1;
  __REG32 PrtEna              : 1;
  __REG32 PrtEnChng           : 1;
  __REG32 PrtOvrCurrAct       : 1;
  __REG32 PrtOvrCurrChng      : 1;
  __REG32 PrtRes              : 1;
  __REG32 PrtSusp             : 1;
  __REG32 PrtRst              : 1;
  __REG32                     : 1;
  __REG32 PrtLnSts            : 2;
  __REG32 PrtPwr              : 1;
  __REG32                     : 4;
  __REG32 PrtSpd              : 2;
  __REG32                     :13;
} __usb_hprt_bits;

/* Host Channel-n Characteristics Register (HCCHARx) */
typedef struct {
  __REG32 MPS                 :11;
  __REG32 EPNum               : 4;
  __REG32 EPDir               : 1;
  __REG32                     : 1;
  __REG32 LSpdDev             : 1;
  __REG32 EPType              : 2;
  __REG32 MC_EC               : 2;
  __REG32 DevAddr             : 7;
  __REG32 OddFrm              : 1;
  __REG32 ChDis               : 1;
  __REG32 ChEna               : 1;
} __usb_hcchar_bits;

/* Host Channel-n Interrupt Register (HCINTx) */
typedef struct {
  __REG32 XferCompl           : 1;
  __REG32 ChHltd              : 1;
  __REG32 AHBErr              : 1;
  __REG32 STALL               : 1;
  __REG32 NAK                 : 1;
  __REG32 ACK                 : 1;
  __REG32 NYET                : 1;
  __REG32 XactErr             : 1;
  __REG32 BblErr              : 1;
  __REG32 FrmOvrun            : 1;
  __REG32 DataTglErr          : 1;
  __REG32 BNAIntr             : 1;
  __REG32 XCS_XACT_ERR        : 1;
  __REG32 DESC_LST_ROLLIntr   : 1;
  __REG32                     :18;
} __usb_hcint_bits;

/* Host Channel-n Interrupt Mask Register (HCINTMSKx) */
typedef struct {
  __REG32 XferComplMsk        : 1;
  __REG32 ChHltdMsk           : 1;
  __REG32 AHBErrMsk           : 1;
  __REG32 StallMsk            : 1;
  __REG32 NakMsk              : 1;
  __REG32 AckMsk              : 1;
  __REG32 NyetMsk             : 1;
  __REG32 XactErrMsk          : 1;
  __REG32 BblErrMsk           : 1;
  __REG32 FrmOvrunMsk         : 1;
  __REG32 DataTglErrMsk       : 1;
  __REG32 BNAIntrMsk          : 1;
  __REG32                     : 1;
  __REG32 DESC_LST_ROLLIntrMsk: 1;
  __REG32                     :18;
} __usb_hcintmsk_bits;

/* Host Channel-n Interrupt Mask Register (HCINTMSKx) */
typedef union {
  /* USB_HCTSIZx */
  /* USB_HCTSIZ_SGx */
  struct {
  __REG32 SCHED_INFO          : 8;
  __REG32 NTD                 : 8;
  __REG32                     :13;
  __REG32 Pid                 : 2;
  __REG32 DoPng               : 1;
  };
  /* USB_HCTSIZ_Bx */
  struct {
  __REG32 XferSize            :19;
  __REG32 PktCnt              :10;
  __REG32 Pid                 : 2;
  __REG32 DoPng               : 1;
  } buffer;
} __usb_hctsiz_bits;

/* Host Channel-n DMA Address Register (HCDMAx) */
typedef struct {
  __REG32                     : 3;
  __REG32 CTD                 : 6;
  __REG32 DMAAddr             :23;
} __usb_hcdma_bits;

/* Device Configuration Register (DCFG) */
typedef struct {
  __REG32 DevSpd              : 2;
  __REG32 NZStsOUTHShk        : 1;
  __REG32 Ena32KHzS           : 1;
  __REG32 DevAddr             : 7;
  __REG32 PerFrInt            : 2;
  __REG32                     : 5;
  __REG32 EPMisCnt            : 5;
  __REG32 DescDMA             : 1;
  __REG32 PerSchIntvl         : 2;
  __REG32 ResValid            : 6;
} __usb_dcfg_bits;

/* Device Control Register (DCTL) */
typedef struct {
  __REG32 RmtWkUpSig          : 1;
  __REG32 SftDiscon           : 1;
  __REG32 GNPINNakSts         : 1;
  __REG32 GOUTNakSts          : 1;
  __REG32                     : 3;
  __REG32 SGNPInNak           : 1;
  __REG32 CGNPInNak           : 1;
  __REG32 SGOUTNak            : 1;
  __REG32 CGOUTNak            : 1;
  __REG32 PWROnPrgDone        : 1;
  __REG32                     : 1;
  __REG32 GMC                 : 2;
  __REG32 IgnrFrmNum          : 1;
  __REG32 NakOnBble           : 1;
  __REG32                     :15;
} __usb_dctl_bits;

/* Device Status Register (DSTS) */
typedef struct {
  __REG32 SuspSts             : 1;
  __REG32 EnumSpd             : 2;
  __REG32 ErrticErr           : 1;
  __REG32                     : 4;
  __REG32 SOFFN               :14;
  __REG32                     :10;
} __usb_dsts_bits;

/* Device IN Endpoint Common Interrupt Mask Register (DIEPMSK) */
typedef struct {
  __REG32 XferComplMsk        : 1;
  __REG32 EPDisbldMsk         : 1;
  __REG32 AHBErrMsk           : 1;
  __REG32 TimeOUTMsk          : 1;
  __REG32 INTknTXFEmpMsk      : 1;
  __REG32 INTknEPMisMsk       : 1;
  __REG32 INEPNakEffMsk       : 1;
  __REG32                     : 1;
  __REG32 TxfifoUndrnMsk      : 1;
  __REG32 BNAInIntrMsk        : 1;
  __REG32                     : 3;
  __REG32 NAKMsk              : 1;
  __REG32                     :18;
} __usb_diepmsk_bits;

/* Device OUT Endpoint Common Interrupt Mask Register (DOEPMSK) */
typedef struct {
  __REG32 XferComplMsk        : 1;
  __REG32 EPDisbldMsk         : 1;
  __REG32 AHBErrMsk           : 1;
  __REG32 SetUPMsk            : 1;
  __REG32 OUTTknEPdisMsk      : 1;
  __REG32                     : 1;
  __REG32 Back2BackSETup      : 1;
  __REG32                     : 1;
  __REG32 OutPktErrMsk        : 1;
  __REG32 BnaOutIntrMsk       : 1;
  __REG32                     : 2;
  __REG32 BbleErrMsk          : 1;
  __REG32 NAKMsk              : 1;
  __REG32 NYETMsk             : 1;
  __REG32                     :17;
} __usb_doepmsk_bits;

/* Device All Endpoints Interrupt Register (DAINT) */
typedef struct {
  __REG32 InEpInt0            : 1;
  __REG32 InEpInt1            : 1;
  __REG32 InEpInt2            : 1;
  __REG32 InEpInt3            : 1;
  __REG32 InEpInt4            : 1;
  __REG32 InEpInt5            : 1;
  __REG32 InEpInt6            : 1;
  __REG32                     : 9;
  __REG32 OutEPInt0           : 1;
  __REG32 OutEPInt1           : 1;
  __REG32 OutEPInt2           : 1;
  __REG32 OutEPInt3           : 1;
  __REG32 OutEPInt4           : 1;
  __REG32 OutEPInt5           : 1;
  __REG32 OutEPInt6           : 1;
  __REG32                     : 9;
} __usb_daint_bits;

/* Device All Endpoints Interrupt Mask Register (DAINTMSK) */
typedef struct {
  __REG32 InEpMsk0            : 1;
  __REG32 InEpMsk1            : 1;
  __REG32 InEpMsk2            : 1;
  __REG32 InEpMsk3            : 1;
  __REG32 InEpMsk4            : 1;
  __REG32 InEpMsk5            : 1;
  __REG32 InEpMsk6            : 1;
  __REG32                     : 9;
  __REG32 OutEpMsk0           : 1;
  __REG32 OutEpMsk1           : 1;
  __REG32 OutEpMsk2           : 1;
  __REG32 OutEpMsk3           : 1;
  __REG32 OutEpMsk4           : 1;
  __REG32 OutEpMsk5           : 1;
  __REG32 OutEpMsk6           : 1;
  __REG32                     : 9;
} __usb_daintmsk_bits;

/* Device VBUS Discharge Time Register (DVBUSDIS) */
typedef struct {
  __REG32 DVBUSDis            :16;
  __REG32                     :16;
} __usb_dvbusdis_bits;

/* Device VBUS Pulsing Time Register (DVBUSPULSE) */
typedef struct {
  __REG32 DVBUSPulse          :12;
  __REG32                     :20;
} __usb_dvbuspulse_bits;

/* Device IN Endpoint FIFO Empty Interrupt Mask Register (DIEPEMPMSK) */
typedef struct {
  __REG32 InEpTxfEmpMsk0      : 1;
  __REG32 InEpTxfEmpMsk1      : 1;
  __REG32 InEpTxfEmpMsk2      : 1;
  __REG32 InEpTxfEmpMsk3      : 1;
  __REG32 InEpTxfEmpMsk4      : 1;
  __REG32 InEpTxfEmpMsk5      : 1;
  __REG32 InEpTxfEmpMsk6      : 1;
  __REG32                     :25;
} __usb_diepempmsk_bits;

/* Device Control IN Endpoint 0 Control Register (DIEPCTL0) */
typedef struct {
  __REG32 MPS                 : 2;
  __REG32                     : 9;
  __REG32 NextEp              : 4;
  __REG32 USBActEP            : 1;
  __REG32                     : 1;
  __REG32 NAKSts              : 1;
  __REG32 EPType              : 2;
  __REG32                     : 1;
  __REG32 Stall               : 1;
  __REG32 TxFNum              : 4;
  __REG32 CNAK                : 1;
  __REG32 SNAK                : 1;
  __REG32                     : 2;
  __REG32 EPDis               : 1;
  __REG32 EPEna               : 1;
} __usb_diepctl0_bits;

/* Device Control OUT Endpoint 0 Control Register (DOEPCTL0) */
typedef struct {
  __REG32 MPS                 : 2;
  __REG32                     :13;
  __REG32 USBActEP            : 1;
  __REG32                     : 1;
  __REG32 NAKSts              : 1;
  __REG32 EPType              : 2;
  __REG32 Snp                 : 1;
  __REG32 Stall               : 1;
  __REG32                     : 4;
  __REG32 CNAK                : 1;
  __REG32 SNAK                : 1;
  __REG32                     : 2;
  __REG32 EPDis               : 1;
  __REG32 EPEna               : 1;
} __usb_doepctl0_bits;

/* Device Endpoint-n Control Register (DIEPCTLx/DOEPCTLx) */
typedef struct {
  __REG32 MPS                 :11;
  __REG32 NextEp              : 4;
  __REG32 USBActEP            : 1;
  __REG32 EO_FrNumDPID        : 1;
  __REG32 NAKSts              : 1;
  __REG32 EPType              : 2;
  __REG32 Snp                 : 1;
  __REG32 Stall               : 1;
  __REG32 TxFNum              : 4;
  __REG32 CNAK                : 1;
  __REG32 SNAK                : 1;
  __REG32 SetEvenFrSetD0PID   : 1;
  __REG32 SetOddFrSetD1PID    : 1;
  __REG32 EPDis               : 1;
  __REG32 EPEna               : 1;
} __usb_diepctl_bits;

/* Device Endpoint-n Interrupt Register (DIEPINTx) */
typedef struct {
  __REG32 XferCompl           : 1;
  __REG32 EPDisbld            : 1;
  __REG32 AHBErr              : 1;
  __REG32 SetUp               : 1;
  __REG32 OUTTknEPdis         : 1;
  __REG32 StsPhseRcvd         : 1;
  __REG32 Back2BackSETup      : 1;
  __REG32 TxFEmp              : 1;
  __REG32 OutPktErrTxfifoUndrn: 1;
  __REG32 BNAIntr             : 1;
  __REG32                     : 1;
  __REG32 PktDrpSts           : 1;
  __REG32 BbleErrIntrpt       : 1;
  __REG32 NAKIntrpt           : 1;
  __REG32 NYETIntrpt          : 1;
  __REG32                     :17;
} __usb_diepint_bits;

/* Device Endpoint-x Interrupt Register (DOEPINTx) */
typedef struct {
  __REG32 XferCompl           : 1;
  __REG32 EPDisbld            : 1;
  __REG32 AHBErr              : 1;
  __REG32 SetUp               : 1;
  __REG32 OUTTknEPdis         : 1;
  __REG32 StsPhseRcvd         : 1;
  __REG32 Back2BackSETup      : 1;
  __REG32                     : 1;
  __REG32 OutPktErrTxfifoUndrn: 1;
  __REG32 BNAIntr             : 1;
  __REG32                     : 1;
  __REG32 PktDrpSts           : 1;
  __REG32 BbleErrIntrpt       : 1;
  __REG32 NAKIntrpt           : 1;
  __REG32 NYETIntrpt          : 1;
  __REG32                     :17;
} __usb_doepint_bits;

/* Device IN Endpoint 0 Transfer Size Register (DIEPTSIZ0) */
typedef struct {
  __REG32 XferSize            : 7;
  __REG32                     :12;
  __REG32 PktCnt              : 2;
  __REG32                     :11;
} __usb_dieptsiz0_bits;

/* Device IN Endpoint 0 Transfer Size Register (DIEPTSIZ0) */
typedef struct {
  __REG32 XferSize            : 7;
  __REG32                     :12;
  __REG32 PktCnt              : 2;
  __REG32                     : 8;
  __REG32 SUPCnt              : 2;
  __REG32                     : 1;
} __usb_doeptsiz0_bits;

/*Device Endpoint-x Transfer Size Register (DIEPTSIZx) */
typedef struct {
  __REG32 XferSize            :19;
  __REG32 PktCnt              :10;
  __REG32                     : 3;
} __usb_dieptsiz_bits;

/* Device Endpoint-x Transfer Size Register (DIEPTSIZx) */
typedef struct {
  __REG32 XferSize            :19;
  __REG32 PktCnt              :10;
  __REG32 RxDPID              : 2;
  __REG32                     : 1;
} __usb_doeptsiz_bits;

/* Device IN Endpoint Transmit FIFO Status Register (DTXFSTSx) */
typedef struct {
  __REG32 INEPTxFSpcAvail     :16;
  __REG32                     :16;
} __usb_dtxfsts_bits;

/* Power and Clock Gating Control Register (PCGCCTL) */
typedef struct {
  __REG32 StopPclk            : 1;
  __REG32 GateHclk            : 1;
  __REG32 PwrClmp             : 1;
  __REG32 RstPdwnModule       : 1;
  __REG32                     : 1;
  __REG32 Enbl_L1Gating       : 1;
  __REG32 PhySleep            : 1;
  __REG32 L1_Suspended        : 1;
  __REG32 ResetAfterSusp      : 1;
  __REG32                     :23;
} __usb_pcgcr_bits;

/* Module Identification Registers */
typedef struct {
  __REG32 MOD_REV             : 8;
  __REG32 MOD_TYPE            : 8;
  __REG32 MOD_NUMBER          :16;
} __usic_id_bits;

/* Channel Control Register */
typedef struct {
  __REG32 MODE                : 4;
  __REG32                     : 2;
  __REG32 HPCEN               : 2;
  __REG32 PM                  : 2;
  __REG32 RSIEN               : 1;
  __REG32 DLIEN               : 1;
  __REG32 TSIEN               : 1;
  __REG32 TBIEN               : 1;
  __REG32 RIEN                : 1;
  __REG32 AIEN                : 1;
  __REG32 BRGIEN              : 1;
  __REG32                     :15;
} __usic_ccr_bits;

/* Channel Configuration Register */
typedef struct {
  __REG32 SSC                 : 1;
  __REG32 ASC                 : 1;
  __REG32 IIC                 : 1;
  __REG32 IIS                 : 1;
  __REG32                     : 2;
  __REG32 RB                  : 1;
  __REG32 TB                  : 1;
  __REG32                     :24;
} __usic_ccfg_bits;

/* Kernel State Configuration Register */
typedef struct {
  __REG32 MODEN               : 1;
  __REG32 BPMODEN             : 1;
  __REG32                     : 2;
  __REG32 NOMCFG              : 2;
  __REG32                     : 1;
  __REG32 BPNOM               : 1;
  __REG32 SUMCFG              : 2;
  __REG32                     : 1;
  __REG32 BPSUM               : 1;
  __REG32                     :20;
} __usic_kscfg_bits;

/* Interrupt Node Pointer Register */
typedef struct {
  __REG32 TSINP               : 3;
  __REG32                     : 1;
  __REG32 TBINP               : 3;
  __REG32                     : 1;
  __REG32 RINP                : 3;
  __REG32                     : 1;
  __REG32 AINP                : 3;
  __REG32                     : 1;
  __REG32 PINP                : 3;
  __REG32                     :13;
} __usic_inpr_bits;

/* Protocol Status Clear Register */
typedef struct {
  __REG32 CST0                : 1;
  __REG32 CST1                : 1;
  __REG32 CST2                : 1;
  __REG32 CST3                : 1;
  __REG32 CST4                : 1;
  __REG32 CST5                : 1;
  __REG32 CST6                : 1;
  __REG32 CST7                : 1;
  __REG32 CST8                : 1;
  __REG32 CST9                : 1;
  __REG32 CRSIF               : 1;
  __REG32 CDLIF               : 1;
  __REG32 CTSIF               : 1;
  __REG32 CTBIF               : 1;
  __REG32 CRIF                : 1;
  __REG32 CAIF                : 1;
  __REG32 CBRGIF              : 1;
  __REG32                     :15;
} __usic_pscr_bits;

/* Input Control Registers 0, 2-5*/
typedef struct {
  __REG32 DSEL                : 3;
  __REG32                     : 1;
  __REG32 INSW                : 1;
  __REG32 DFEN                : 1;
  __REG32 DSEN                : 1;
  __REG32                     : 1;
  __REG32 DPOL                : 1;
  __REG32 SFSEL               : 1;
  __REG32 CM                  : 2;
  __REG32                     : 3;
  __REG32 DXS                 : 1;
  __REG32                     :16;
} __usic_dxcr_bits;

/* Input Control Register 1 */
typedef struct {
  __REG32 DSEL                : 3;
  __REG32 DCEN                : 1;
  __REG32 INSW                : 1;
  __REG32 DFEN                : 1;
  __REG32 DSEN                : 1;
  __REG32                     : 1;
  __REG32 DPOL                : 1;
  __REG32 SFSEL               : 1;
  __REG32 CM                  : 2;
  __REG32                     : 3;
  __REG32 DXS                 : 1;
  __REG32                     :16;
} __usic_dx1cr_bits;

/* Fractional Divider Register */
typedef struct {
  __REG32 STEP                :10;
  __REG32                     : 4;
  __REG32 DM                  : 2;
  __REG32 RESULT              :10;
  __REG32                     : 6;
} __usic_fdr_bits;

/* Baud Rate Generator Register */
typedef struct {
  __REG32 CLKSEL              : 2;
  __REG32                     : 1;
  __REG32 TMEN                : 1;
  __REG32 PPPEN               : 1;
  __REG32                     : 1;
  __REG32 CTQSEL              : 2;
  __REG32 PCTQ                : 2;
  __REG32 DCTQ                : 5;
  __REG32                     : 1;
  __REG32 PDIV                :10;
  __REG32                     : 3;
  __REG32 MCLKCFG             : 1;
  __REG32 SCLKCFG             : 2;
} __usic_brg_bits;

/* Capture Mode Timer Register */
typedef struct {
  __REG32 CTV                 :10;
  __REG32                     :22;
} __usic_cmtr_bits;

/* Shift Control Register */
typedef struct {
  __REG32 SDIR                : 1;
  __REG32 PDL                 : 1;
  __REG32 DSM                 : 2;
  __REG32 HPCDIR              : 1;
  __REG32                     : 1;
  __REG32 DOCFG               : 2;
  __REG32 TRM                 : 2;
  __REG32                     : 6;
  __REG32 FLE                 : 6;
  __REG32                     : 2;
  __REG32 WLE                 : 4;
  __REG32                     : 4;
} __usic_sctr_bits;

/* Transmit Control/Status Register */
typedef struct {
  __REG32 WLEMD               : 1;
  __REG32 SELMD               : 1;
  __REG32 FLEMD               : 1;
  __REG32 WAMD                : 1;
  __REG32 HPCMD               : 1;
  __REG32 SOF                 : 1;
  __REG32 _EOF                : 1;
  __REG32 TDV                 : 1;
  __REG32 TDSSM               : 1;
  __REG32                     : 1;
  __REG32 TDEN                : 2;
  __REG32 TDVTR               : 1;
  __REG32 WA                  : 1;
  __REG32                     :10;
  __REG32 TSOF                : 1;
  __REG32                     : 1;
  __REG32 TV                  : 1;
  __REG32 TVC                 : 1;
  __REG32 TE                  : 1;
  __REG32                     : 3;
} __usic_tcsr_bits;

/* Flag Modification Register */
typedef struct {
  __REG32 MTDV                : 2;
  __REG32                     : 2;
  __REG32 ATVC                : 1;
  __REG32                     : 9;
  __REG32 CRDV0               : 1;
  __REG32 CRDV1               : 1;
  __REG32 SIO0                : 1;
  __REG32 SIO1                : 1;
  __REG32 SIO2                : 1;
  __REG32 SIO3                : 1;
  __REG32 SIO4                : 1;
  __REG32 SIO5                : 1;
  __REG32                     :10;
} __usic_fmr_bits;

/* Transmit Buffer Locations */
typedef struct {
  __REG32 TDATA               :16;
  __REG32                     :16;
} __usic_tbuf_bits;

/* Receiver Buffer Register 0/1 */
typedef struct {
  __REG32 DSR               :16;
  __REG32                   :16;
} __usic_rbuf_bits;

/* Receiver Buffer 01 Status Register */
typedef struct {
  __REG32 WLEN0             : 4;
  __REG32                   : 2;
  __REG32 SOF0              : 1;
  __REG32                   : 1;
  __REG32 PAR0              : 1;
  __REG32 PERR0             : 1;
  __REG32                   : 3;
  __REG32 RDV00             : 1;
  __REG32 RDV01             : 1;
  __REG32 DS0               : 1;
  __REG32 WLEN1             : 4;
  __REG32                   : 2;
  __REG32 SOF1              : 1;
  __REG32                   : 1;
  __REG32 PAR1              : 1;
  __REG32 PERR1             : 1;
  __REG32                   : 3;
  __REG32 RDV10             : 1;
  __REG32 RDV11             : 1;
  __REG32 DS1               : 1;
} __usic_rbuf01sr_bits;

/* Receiver Buffer Status Register */
typedef struct {
  __REG32 WLEN              : 4;
  __REG32                   : 2;
  __REG32 SOF               : 1;
  __REG32                   : 1;
  __REG32 PAR               : 1;
  __REG32 PERR              : 1;
  __REG32                   : 3;
  __REG32 RDV0              : 1;
  __REG32 RDV1              : 1;
  __REG32 DS                : 1;
  __REG32                   :16;
} __usic_rbufsr_bits;

/* Bypass Data Register */
typedef struct {
  __REG32 BDATA             :16;
  __REG32                   :16;
} __usic_byp_bits;

/* Bypass Control Register */
typedef struct {
  __REG32 BWLE              : 4;
  __REG32                   : 4;
  __REG32 BDSSM             : 1;
  __REG32                   : 1;
  __REG32 BDEN              : 2;
  __REG32 BDVTR             : 1;
  __REG32 BPRIO             : 1;
  __REG32                   : 1;
  __REG32 BDV               : 1;
  __REG32 BSELO             : 5;
  __REG32 BHPC              : 3;
  __REG32                   : 8;
} __usic_bypcr_bits;

/* Transmit/Receive Buffer Status Register */
typedef struct {
  __REG32 SRBI              : 1;
  __REG32 RBERI             : 1;
  __REG32 ARBI              : 1;
  __REG32 REMPTY            : 1;
  __REG32 RFULL             : 1;
  __REG32 RBUS              : 1;
  __REG32 SRBT              : 1;
  __REG32                   : 1;
  __REG32 STBI              : 1;
  __REG32 TBERI             : 1;
  __REG32                   : 1;
  __REG32 TEMPTY            : 1;
  __REG32 TFULL             : 1;
  __REG32 TBUS              : 1;
  __REG32 STBT              : 1;
  __REG32                   : 1;
  __REG32 RBFLVL            : 7;
  __REG32                   : 1;
  __REG32 TBFLVL            : 7;
  __REG32                   : 1;
} __usic_trbsr_bits;

/* Transmit/Receive Buffer Status Clear Register */
typedef struct {
  __REG32 CSRBI             : 1;
  __REG32 CRBERI            : 1;
  __REG32 CARBI             : 1;
  __REG32                   : 5;
  __REG32 CSTBI             : 1;
  __REG32 CTBERI            : 1;
  __REG32 CBDV              : 1;
  __REG32                   : 3;
  __REG32 FLUSHRB           : 1;
  __REG32 FLUSHTB           : 1;
  __REG32                   :16;
} __usic_trbscr_bits;

/* Transmitter Buffer Control Register */
typedef struct {
  __REG32 DPTR              : 6;
  __REG32                   : 2;
  __REG32 LIMIT             : 6;
  __REG32 STBTM             : 1;
  __REG32 STBTEN            : 1;
  __REG32 STBINP            : 3;
  __REG32 ATBINP            : 3;
  __REG32                   : 2;
  __REG32 SIZE              : 3;
  __REG32                   : 1;
  __REG32 LOF               : 1;
  __REG32                   : 1;
  __REG32 STBIEN            : 1;
  __REG32 TBERIEN           : 1;
} __usic_tbctr_bits;

/* Receiver Buffer Control Register */
typedef struct {
  __REG32 DPTR              : 6;
  __REG32                   : 2;
  __REG32 LIMIT             : 6;
  __REG32 STBTM             : 1;
  __REG32 STBTEN            : 1;
  __REG32 STBINP            : 3;
  __REG32 ATBINP            : 3;
  __REG32 RCIM              : 2;
  __REG32 SIZE              : 3;
  __REG32 RNM               : 1;
  __REG32 LOF               : 1;
  __REG32 ARBIEN            : 1;
  __REG32 STBIEN            : 1;
  __REG32 TBERIEN           : 1;
} __usic_rbctr_bits;

/* Transmit FIFO Buffer Input Location x */
typedef struct {
  __REG32 TDATA             :16;
  __REG32                   :16;
} __usic_in_bits;

/* Receiver Buffer Output Register */
typedef struct {
  __REG32 DSR               :16;
  __REG32 RCI               : 5;
  __REG32                   :11;
} __usic_outr_bits;

/* Transmit/Receive Buffer Pointer Register */
typedef struct {
  __REG32 TDIPTR            : 6;
  __REG32                   : 2;
  __REG32 TDOPTR            : 6;
  __REG32                   : 2;
  __REG32 RDIPTR            : 6;
  __REG32                   : 2;
  __REG32 RDOPTR            : 6;
  __REG32                   : 2;
} __usic_trbptr_bits;

/* Protocol Registers */
typedef union {
  /*USICx_Cy_PCR*/
  /*USICx_C0_PCR_ASC*/
  /*USICx_C1_PCR_ASC*/
  struct {
  __REG32 SMD               : 1;
  __REG32 STPB              : 1;
  __REG32 IDM               : 1;
  __REG32 SBIEN             : 1;
  __REG32 CDEN              : 1;
  __REG32 RNIEN             : 1;
  __REG32 FEIEN             : 1;
  __REG32 FFIEN             : 1;
  __REG32 SP                : 5;
  __REG32 PL                : 3;
  __REG32 RSTEN             : 1;
  __REG32 TSTEN             : 1;
  __REG32                   :13;
  __REG32 MCLK              : 1;
  };
  /*USICx_C0_PCR_SSC*/
  /*USICx_C1_PCR_SSC*/
  struct {
  __REG32 MSLSEN            : 1;
  __REG32 SELCTR            : 1;
  __REG32 SELINV            : 1;
  __REG32 FEM               : 1;
  __REG32 CTQSEL1           : 2;
  __REG32 PCTQ1             : 2;
  __REG32 DCTQ1             : 5;
  __REG32 PARIEN            : 1;
  __REG32 MSLSIEN           : 1;
  __REG32 DX2TIEN           : 1;
  __REG32 SELO              : 8;
  __REG32 TIWEN             : 1;
  __REG32                   : 6;
  __REG32 MCLK              : 1;
  } ssc;
  /*USICx_C0_PCR_IIC*/
  /*USICx_C1_PCR_IIC*/
  struct {
  __REG32 SLAD              :16;
  __REG32 ACK00             : 1;
  __REG32 STIM              : 1;
  __REG32 SCRIEN            : 1;
  __REG32 RSCRIEN           : 1;
  __REG32 PCRIEN            : 1;
  __REG32 NACKIEN           : 1;
  __REG32 ARLIEN            : 1;
  __REG32 SRRIEN            : 1;
  __REG32 ERRIEN            : 1;
  __REG32 SACKDIS           : 1;
  __REG32 HDEL              : 4;
  __REG32 ACKIEN            : 1;
  __REG32 MCLK              : 1;
  } iic;
  /*USICx_C0_PCR_IIS*/
  /*USICx_C1_PCR_IIS*/
  struct {
  __REG32 WAGEN             : 1;
  __REG32 DTEN              : 1;
  __REG32 SELINV            : 1;
  __REG32                   : 1;
  __REG32 WAFEIEN           : 1;
  __REG32 WAREIEN           : 1;
  __REG32 ENDIEN            : 1;
  __REG32                   : 8;
  __REG32 DX2TIEN           : 1;
  __REG32 TDEL              : 6;
  __REG32                   : 9;
  __REG32 MCLK              : 1;
  } iis;
} __usic_pcr_bits;

/* Protocol Status Registers */
typedef union {
  /*USICx_Cy_PSR*/
  /*USICx_C0_PSR_ASC*/
  /*USICx_C1_PSR_ASC*/
  struct {
  __REG32 TXIDLE            : 1;
  __REG32 RXIDLE            : 1;
  __REG32 SBD               : 1;
  __REG32 COL               : 1;
  __REG32 RNS               : 1;
  __REG32 FER0              : 1;
  __REG32 FER1              : 1;
  __REG32 RFF               : 1;
  __REG32 TFF               : 1;
  __REG32 BUSY              : 1;
  __REG32 RSIF              : 1;
  __REG32 DLIF              : 1;
  __REG32 TSIF              : 1;
  __REG32 TBIF              : 1;
  __REG32 RIF               : 1;
  __REG32 AIF               : 1;
  __REG32 BRGIF             : 1;
  __REG32                   :15;
  };
  /*USICx_C0_PSR_SSC*/
  /*USICx_C1_PSR_SSC*/
  struct {
  __REG32 MSLS              : 1;
  __REG32 DX2S              : 1;
  __REG32 MSLSEV            : 1;
  __REG32 DX2TEV            : 1;
  __REG32 PARERR            : 1;
  __REG32                   : 5;
  __REG32 RSIF              : 1;
  __REG32 DLIF              : 1;
  __REG32 TSIF              : 1;
  __REG32 TBIF              : 1;
  __REG32 RIF               : 1;
  __REG32 AIF               : 1;
  __REG32 BRGIF             : 1;
  __REG32                   :15;
  } ssc;
  /*USICx_C0_PSR_IIC*/
  /*USICx_C1_PSR_IIC*/
  struct {
  __REG32 SLSEL             : 1;
  __REG32 WTDF              : 1;
  __REG32 SCR               : 1;
  __REG32 RSCR              : 1;
  __REG32 PCR               : 1;
  __REG32 NACK              : 1;
  __REG32 ARL               : 1;
  __REG32 SRR               : 1;
  __REG32 ERR               : 1;
  __REG32 ACK               : 1;
  __REG32 RSIF              : 1;
  __REG32 DLIF              : 1;
  __REG32 TSIF              : 1;
  __REG32 TBIF              : 1;
  __REG32 RIF               : 1;
  __REG32 AIF               : 1;
  __REG32 BRGIF             : 1;
  __REG32                   :15;
  } iic;
  /*USICx_C0_PSR_IIS*/
  /*USICx_C1_PSR_IIS*/
  struct {
  __REG32 WA                : 1;
  __REG32 DX2S              : 1;
  __REG32                   : 1;
  __REG32 DX2TEV            : 1;
  __REG32 WAFE              : 1;
  __REG32 WARE              : 1;
  __REG32 END               : 1;
  __REG32                   : 3;
  __REG32 RSIF              : 1;
  __REG32 DLIF              : 1;
  __REG32 TSIF              : 1;
  __REG32 TBIF              : 1;
  __REG32 RIF               : 1;
  __REG32 AIF               : 1;
  __REG32 BRGIF             : 1;
  __REG32                   :15;
  } iis;
} __usic_psr_bits;

/* Module Identification Register */
typedef struct {
  __REG32 MODE_REV          : 8;
  __REG32 MOD_TYPE          : 8;
  __REG32 MOD_NUMBER        :16;
} __can_id_bits;

/* Panel Control Register */
typedef struct {
  __REG32 PANCMD            : 8;
  __REG32 BUSY              : 1;
  __REG32 RBUSY             : 1;
  __REG32                   : 6;
  __REG32 PANAR1            : 8;
  __REG32 PANAR2            : 8;
} __can_panctr_bits;

/* Module Control Register */
typedef struct {
  __REG32                   :12;
  __REG32 MPSEL             : 4;
  __REG32                   :16;
} __can_mcr_bits;

/* Module Control Register */
typedef struct {
  __REG32 IT                :16;
  __REG32                   :16;
} __can_mitr_bits;

/* List Register 0-7 */
typedef struct {
  __REG32 BEGIN             : 8;
  __REG32 END               : 8;
  __REG32 SIZE              : 8;
  __REG32 EMPTY             : 1;
  __REG32                   : 7;
} __can_list_bits;

/* Message Index Register 0-7 */
typedef struct {
  __REG32 INDEX             : 6;
  __REG32                   :26;
} __can_msid_bits;

/* Message Index Mask Register */
typedef struct {
  __REG32 IM0               : 1;
  __REG32 IM1               : 1;
  __REG32 IM2               : 1;
  __REG32 IM3               : 1;
  __REG32 IM4               : 1;
  __REG32 IM5               : 1;
  __REG32 IM6               : 1;
  __REG32 IM7               : 1;
  __REG32 IM8               : 1;
  __REG32 IM9               : 1;
  __REG32 IM10              : 1;
  __REG32 IM11              : 1;
  __REG32 IM12              : 1;
  __REG32 IM13              : 1;
  __REG32 IM14              : 1;
  __REG32 IM15              : 1;
  __REG32 IM16              : 1;
  __REG32 IM17              : 1;
  __REG32 IM18              : 1;
  __REG32 IM19              : 1;
  __REG32 IM20              : 1;
  __REG32 IM21              : 1;
  __REG32 IM22              : 1;
  __REG32 IM23              : 1;
  __REG32 IM24              : 1;
  __REG32 IM25              : 1;
  __REG32 IM26              : 1;
  __REG32 IM27              : 1;
  __REG32 IM28              : 1;
  __REG32 IM29              : 1;
  __REG32 IM30              : 1;
  __REG32 IM31              : 1;
} __can_msimask_bits;

/* Node x Control Register 0-2 */
typedef struct {
  __REG32 INIT              : 1;
  __REG32 TRIE              : 1;
  __REG32 LECIE             : 1;
  __REG32 ALIE              : 1;
  __REG32 CANDIS            : 1;
  __REG32                   : 1;
  __REG32 CCE               : 1;
  __REG32 CALM              : 1;
  __REG32 SUSEN             : 1;
  __REG32                   :23;
} __can_ncr_bits;

/* Node x Status Register 0-2 */
typedef struct {
  __REG32 LEC               : 3;
  __REG32 TXOK              : 1;
  __REG32 RXOK              : 1;
  __REG32 ALERT             : 1;
  __REG32 EWRN              : 1;
  __REG32 BOFF              : 1;
  __REG32 LLE               : 1;
  __REG32 LOE               : 1;
  __REG32 SUSACK            : 1;
  __REG32                   :21;
} __can_nsr_bits;

/* Node x Status Register 0-2 */
typedef struct {
  __REG32 ALINP             : 3;
  __REG32                   : 1;
  __REG32 LECINP            : 3;
  __REG32                   : 1;
  __REG32 TRINP             : 3;
  __REG32                   : 1;
  __REG32 CFCINP            : 3;
  __REG32                   :17;
} __can_nipr_bits;

/* Node x Port Control Register 0-2 */
typedef struct {
  __REG32 RXSEL             : 3;
  __REG32                   : 5;
  __REG32 LBM               : 1;
  __REG32                   :23;
} __can_npcr_bits;

/* Node x Bit Timing Register 0-2 */
typedef struct {
  __REG32 BRP               : 6;
  __REG32 SJW               : 2;
  __REG32 TSEG1             : 4;
  __REG32 TSEG2             : 3;
  __REG32 DIV8              : 1;
  __REG32                   :16;
} __can_nbtr_bits;

/* Node x Error Counter Register 0-2 */
typedef struct {
  __REG32 REC               : 8;
  __REG32 TEC               : 8;
  __REG32 EWRNLVL           : 8;
  __REG32 LETD              : 1;
  __REG32 LEINC             : 1;
  __REG32                   : 6;
} __can_necnt_bits;

/* Node x Error Counter Register 0-2 */
typedef struct {
  __REG32 CFC               :16;
  __REG32 CFSEL             : 3;
  __REG32 CFMOD             : 2;
  __REG32                   : 1;
  __REG32 CFCIE             : 1;
  __REG32 CFCOV             : 1;
  __REG32                   : 8;
} __can_nfcr_bits;

/* Message Object n Control Register */
/* Message Object n Status Register */
typedef union {
  /*CAN_MOCTRx*/
  struct {
  __REG32 RESRXPND          : 1;
  __REG32 RESTXPND          : 1;
  __REG32 RESRXUPD          : 1;
  __REG32 RESNEWDAT         : 1;
  __REG32 RESMSGLST         : 1;
  __REG32 RESMSGVAL         : 1;
  __REG32 RESRTSEL          : 1;
  __REG32 RESRXEN           : 1;
  __REG32 RESTXRQ           : 1;
  __REG32 RESTXEN0          : 1;
  __REG32 RESTXEN1          : 1;
  __REG32 RESDIR            : 1;
  __REG32                   : 4;
  __REG32 SETRXPND          : 1;
  __REG32 SETTXPND          : 1;
  __REG32 SETRXUPD          : 1;
  __REG32 SETNEWDAT         : 1;
  __REG32 SETMSGLST         : 1;
  __REG32 SETMSGVAL         : 1;
  __REG32 SETRTSEL          : 1;
  __REG32 SETRXEN           : 1;
  __REG32 SETTXRQ           : 1;
  __REG32 SETTXEN0          : 1;
  __REG32 SETTXEN1          : 1;
  __REG32 SETDIR            : 1;
  __REG32                   : 4;
  };
  /*CAN_MOSTATx*/
  struct {
  __REG32 RXPND             : 1;
  __REG32 TXPND             : 1;
  __REG32 RXUPD             : 1;
  __REG32 NEWDAT            : 1;
  __REG32 MSGLST            : 1;
  __REG32 MSGVAL            : 1;
  __REG32 RTSEL             : 1;
  __REG32 RXEN              : 1;
  __REG32 TXRQ              : 1;
  __REG32 TXEN0             : 1;
  __REG32 TXEN1             : 1;
  __REG32 DIR               : 1;
  __REG32 LIST              : 4;
  __REG32 PPREV             : 8;
  __REG32 PNEXT             : 8;
  };
} __can_moctr_bits;

/* Message Object n Interrupt Pointer Register */
typedef struct {
  __REG32 RXINP             : 3;
  __REG32                   : 1;
  __REG32 TXINP             : 3;
  __REG32                   : 1;
  __REG32 MPN               : 8;
  __REG32 CFCVAL            :16;
} __can_moipr_bits;

/* Message Object n Function Control Register */
typedef struct {
  __REG32 MMC               : 4;
  __REG32                   : 4;
  __REG32 GDFS              : 1;
  __REG32 IDC               : 1;
  __REG32 DLCC              : 1;
  __REG32 DATC              : 1;
  __REG32                   : 4;
  __REG32 RXIE              : 1;
  __REG32 TXIE              : 1;
  __REG32 OVIE              : 1;
  __REG32                   : 1;
  __REG32 FRREN             : 1;
  __REG32 RMM               : 1;
  __REG32 SDT               : 1;
  __REG32 STT               : 1;
  __REG32 DLC               : 4;
  __REG32                   : 4;
} __can_mofcr_bits;

/* Message Object n FIFO/Gateway Pointer Register */
typedef struct {
  __REG32 BOT               : 8;
  __REG32 TOP               : 8;
  __REG32 CUR               : 8;
  __REG32 SEL               : 8;
} __can_mofgpr_bits;

/* Message Object n Acceptance Mask Register */
typedef struct {
  __REG32 AM                :29;
  __REG32 MIDE              : 1;
  __REG32                   : 2;
} __can_moamr_bits;

/* Message Object n Arbitration Register */
typedef struct {
  __REG32 ID                :29;
  __REG32 IDE               : 1;
  __REG32 PRI               : 2;
} __can_moar_bits;

/* Message Object n Data Register Low */
typedef struct {
  __REG32 DB0               : 8;
  __REG32 DB1               : 8;
  __REG32 DB2               : 8;
  __REG32 DB3               : 8;
} __can_modatal_bits;

/* Message Object n Data Register High */
typedef struct {
  __REG32 DB4               : 8;
  __REG32 DB5               : 8;
  __REG32 DB6               : 8;
  __REG32 DB7               : 8;
} __can_modatah_bits;

/* Message Object n Data Register High */
typedef struct {
  __REG32 DISR              : 1;
  __REG32 DISS              : 1;
  __REG32                   : 1;
  __REG32 EDIS              : 1;
  __REG32 SBWE              : 1;
  __REG32                   :27;
} __can_clc_bits;

/* CAN Fractional Divider Register */
typedef struct {
  __REG32 STEP              :10;
  __REG32                   : 1;
  __REG32 SM                : 1;
  __REG32 SC                : 2;
  __REG32 DM                : 2;
  __REG32 RESULT            :10;
  __REG32                   : 2;
  __REG32 SUSACK            : 1;
  __REG32 SUSREQ            : 1;
  __REG32 ENHW              : 1;
  __REG32 DISCLK            : 1;
} __can_fdr_bits;

/* Module Identification Register */
typedef struct {
  __REG32 MOD_REV           : 8;
  __REG32 MOD_TYPE          : 8;
  __REG32 MOD_NUMBER        :16;
} __adc_id_bits;

/* Clock Control Register */
typedef struct {
  __REG32 DISR              : 1;
  __REG32 DISS              : 1;
  __REG32                   : 1;
  __REG32 EDIS              : 1;
  __REG32                   :28;
} __adc_clc_bits;

/* OCDS Control and Status Register */
typedef struct {
  __REG32 TGS               : 2;
  __REG32 TGB               : 1;
  __REG32 TG_P              : 1;
  __REG32                   :20;
  __REG32 SUS               : 4;
  __REG32 SUS_P             : 1;
  __REG32 SUSSTA            : 1;
  __REG32                   : 2;
} __adc_ocs_bits;

/* Global Configuration Register */
typedef struct {
  __REG32 DIVA              : 5;
  __REG32                   : 2;
  __REG32 DCMSB             : 1;
  __REG32 DIVD              : 2;
  __REG32                   : 5;
  __REG32 DIVWC             : 1;
  __REG32 DPCAL0            : 1;
  __REG32 DPCAL1            : 1;
  __REG32 DPCAL2            : 1;
  __REG32 DPCAL3            : 1;
  __REG32                   :11;
  __REG32 SUCAL             : 1;
} __adc_globcfg_bits;

/* Channel Assignment Register, Group x */
typedef struct {
  __REG32 ASSCH0            : 1;
  __REG32 ASSCH1            : 1;
  __REG32 ASSCH2            : 1;
  __REG32 ASSCH3            : 1;
  __REG32 ASSCH4            : 1;
  __REG32 ASSCH5            : 1;
  __REG32 ASSCH6            : 1;
  __REG32 ASSCH7            : 1;
  __REG32                   :24;
} __adc_gchass_bits;

/* Arbitration Configuration Register, Group x */
typedef struct {
  __REG32 ANONC             : 2;
  __REG32                   : 2;
  __REG32 ARBRND            : 2;
  __REG32                   : 1;
  __REG32 ARBM              : 1;
  __REG32                   : 8;
  __REG32 ANONS             : 2;
  __REG32                   :10;
  __REG32 CAL               : 1;
  __REG32                   : 1;
  __REG32 BUSY              : 1;
  __REG32 SAMPLE            : 1;
} __adc_garbcfg_bits;

/* Arbitration Priority Register, Group x */
typedef struct {
  __REG32 PRIO0             : 2;
  __REG32                   : 1;
  __REG32 CSM0              : 1;
  __REG32 PRIO1             : 2;
  __REG32                   : 1;
  __REG32 CSM1              : 1;
  __REG32 PRIO2             : 2;
  __REG32                   : 1;
  __REG32 CSM2              : 1;
  __REG32                   :12;
  __REG32 ASEN0             : 1;
  __REG32 ASEN1             : 1;
  __REG32 ASEN2             : 1;
  __REG32                   : 5;
} __adc_garbpr_bits;

/* Queue 0 Source Control Register, Group x */
typedef struct {
  __REG32 SRCRESREG         : 4;
  __REG32                   : 4;
  __REG32 XTSEL             : 4;
  __REG32 XTLVL             : 1;
  __REG32 XTMODE            : 2;
  __REG32 XTWC              : 1;
  __REG32 GTSEL             : 4;
  __REG32 GTLVL             : 1;
  __REG32                   : 2;
  __REG32 GTWC              : 1;
  __REG32                   : 4;
  __REG32 TMEN              : 1;
  __REG32                   : 2;
  __REG32 TMWC              : 1;
} __adc_gqctrl0_bits;

/* Queue 0 Mode Register, Group x */
typedef struct {
  __REG32 ENGT              : 2;
  __REG32 ENTR              : 1;
  __REG32                   : 5;
  __REG32 CLRV              : 1;
  __REG32 TREV              : 1;
  __REG32 FLUSH             : 1;
  __REG32 CEV               : 1;
  __REG32                   : 4;
  __REG32 RPTDIS            : 1;
  __REG32                   :15;
} __adc_gqmr0_bits;

/* Queue 0 Status Register, Group x */
typedef struct {
  __REG32 FILL              : 4;
  __REG32                   : 1;
  __REG32 EMPTY             : 1;
  __REG32                   : 1;
  __REG32 REQGT             : 1;
  __REG32 EV                : 1;
  __REG32                   :23;
} __adc_gqsr0_bits;

/* Queue 0 Input Register, Group x */
typedef struct {
  __REG32 REQCHNR           : 5;
  __REG32 RF                : 1;
  __REG32 ENSI              : 1;
  __REG32 EXTR              : 1;
  __REG32                   :24;
} __adc_gqinr0_bits;

/* Queue 0 Register 0, Group x */
/* Queue 0 Backup Register, Group x */
typedef struct {
  __REG32 REQCHNR           : 5;
  __REG32 RF                : 1;
  __REG32 ENSI              : 1;
  __REG32 EXTR              : 1;
  __REG32 V                 : 1;
  __REG32                   :23;
} __adc_gq0r0_bits;

/* Autoscan Source Control Register, Group x */
typedef struct {
  __REG32 SRCRESREG         : 4;
  __REG32                   : 4;
  __REG32 XTSEL             : 4;
  __REG32 XTLVL             : 1;
  __REG32 XTMODE            : 2;
  __REG32 XTWC              : 1;
  __REG32 GTSEL             : 4;
  __REG32 GTLVL             : 1;
  __REG32                   : 2;
  __REG32 GTWC              : 1;
  __REG32                   : 4;
  __REG32 TMEN              : 1;
  __REG32                   : 2;
  __REG32 TMWC              : 1;
} __adc_gasctrl_bits;

/* Autoscan Source Control Register, Group x */
typedef struct {
  __REG32 ENGT              : 2;
  __REG32 ENTR              : 1;
  __REG32 ENSI              : 1;
  __REG32 SCAN              : 1;
  __REG32 LDM               : 1;
  __REG32                   : 1;
  __REG32 REQGT             : 1;
  __REG32 CLRPND            : 1;
  __REG32 LDEV              : 1;
  __REG32                   : 6;
  __REG32 RPTDIS            : 1;
  __REG32                   :15;
} __adc_gasmr_bits;

/* Autoscan Source Channel Select Register, Group x */
typedef struct {
  __REG32 CHSEL0            : 1;
  __REG32 CHSEL1            : 1;
  __REG32 CHSEL2            : 1;
  __REG32 CHSEL3            : 1;
  __REG32 CHSEL4            : 1;
  __REG32 CHSEL5            : 1;
  __REG32 CHSEL6            : 1;
  __REG32 CHSEL7            : 1;
  __REG32 CHSEL8            : 1;
  __REG32 CHSEL9            : 1;
  __REG32 CHSEL10           : 1;
  __REG32 CHSEL11           : 1;
  __REG32 CHSEL12           : 1;
  __REG32 CHSEL13           : 1;
  __REG32 CHSEL14           : 1;
  __REG32 CHSEL15           : 1;
  __REG32 CHSEL16           : 1;
  __REG32 CHSEL17           : 1;
  __REG32 CHSEL18           : 1;
  __REG32 CHSEL19           : 1;
  __REG32 CHSEL20           : 1;
  __REG32 CHSEL21           : 1;
  __REG32 CHSEL22           : 1;
  __REG32 CHSEL23           : 1;
  __REG32 CHSEL24           : 1;
  __REG32 CHSEL25           : 1;
  __REG32 CHSEL26           : 1;
  __REG32 CHSEL27           : 1;
  __REG32 CHSEL28           : 1;
  __REG32 CHSEL29           : 1;
  __REG32 CHSEL30           : 1;
  __REG32 CHSEL31           : 1;
} __adc_gassel_bits;

/* Autoscan Source Pending Register, Group x */
typedef struct {
  __REG32 CHPND0            : 1;
  __REG32 CHPND1            : 1;
  __REG32 CHPND2            : 1;
  __REG32 CHPND3            : 1;
  __REG32 CHPND4            : 1;
  __REG32 CHPND5            : 1;
  __REG32 CHPND6            : 1;
  __REG32 CHPND7            : 1;
  __REG32 CHPND8            : 1;
  __REG32 CHPND9            : 1;
  __REG32 CHPND10           : 1;
  __REG32 CHPND11           : 1;
  __REG32 CHPND12           : 1;
  __REG32 CHPND13           : 1;
  __REG32 CHPND14           : 1;
  __REG32 CHPND15           : 1;
  __REG32 CHPND16           : 1;
  __REG32 CHPND17           : 1;
  __REG32 CHPND18           : 1;
  __REG32 CHPND19           : 1;
  __REG32 CHPND20           : 1;
  __REG32 CHPND21           : 1;
  __REG32 CHPND22           : 1;
  __REG32 CHPND23           : 1;
  __REG32 CHPND24           : 1;
  __REG32 CHPND25           : 1;
  __REG32 CHPND26           : 1;
  __REG32 CHPND27           : 1;
  __REG32 CHPND28           : 1;
  __REG32 CHPND29           : 1;
  __REG32 CHPND30           : 1;
  __REG32 CHPND31           : 1;
} __adc_gaspnd_bits;

/* Background Request Source Control Register */
typedef struct {
  __REG32 SRCRESREG         : 4;
  __REG32                   : 4;
  __REG32 XTSEL             : 4;
  __REG32 XTLVL             : 1;
  __REG32 XTMODE            : 2;
  __REG32 XTWC              : 1;
  __REG32 GTSEL             : 4;
  __REG32 GTLVL             : 1;
  __REG32                   : 2;
  __REG32 GTWC              : 1;
  __REG32                   : 8;
} __adc_brsctrl_bits;

/* Background Request Source Mode Register */
typedef struct {
  __REG32 ENGT              : 2;
  __REG32 ENTR              : 1;
  __REG32 ENSI              : 1;
  __REG32 SCAN              : 1;
  __REG32 LDM               : 1;
  __REG32                   : 1;
  __REG32 REQGT             : 1;
  __REG32 CLRPND            : 1;
  __REG32 LDEV              : 1;
  __REG32                   : 6;
  __REG32 RPTDIS            : 1;
  __REG32                   :15;
} __adc_brsmr_bits;

/* Background Request Source Channel Select Register, Group x */
typedef struct {
  __REG32 CHSELG0           : 1;
  __REG32 CHSELG1           : 1;
  __REG32 CHSELG2           : 1;
  __REG32 CHSELG3           : 1;
  __REG32 CHSELG4           : 1;
  __REG32 CHSELG5           : 1;
  __REG32 CHSELG6           : 1;
  __REG32 CHSELG7           : 1;
  __REG32 CHSELG8           : 1;
  __REG32 CHSELG9           : 1;
  __REG32 CHSELG10          : 1;
  __REG32 CHSELG11          : 1;
  __REG32 CHSELG12          : 1;
  __REG32 CHSELG13          : 1;
  __REG32 CHSELG14          : 1;
  __REG32 CHSELG15          : 1;
  __REG32 CHSELG16          : 1;
  __REG32 CHSELG17          : 1;
  __REG32 CHSELG18          : 1;
  __REG32 CHSELG19          : 1;
  __REG32 CHSELG20          : 1;
  __REG32 CHSELG21          : 1;
  __REG32 CHSELG22          : 1;
  __REG32 CHSELG23          : 1;
  __REG32 CHSELG24          : 1;
  __REG32 CHSELG25          : 1;
  __REG32 CHSELG26          : 1;
  __REG32 CHSELG27          : 1;
  __REG32 CHSELG28          : 1;
  __REG32 CHSELG29          : 1;
  __REG32 CHSELG30          : 1;
  __REG32 CHSELG31          : 1;
} __adc_brssel_bits;

/* Background Request Source Pending Register, Group x */
typedef struct {
  __REG32 CHPNDG0           : 1;
  __REG32 CHPNDG1           : 1;
  __REG32 CHPNDG2           : 1;
  __REG32 CHPNDG3           : 1;
  __REG32 CHPNDG4           : 1;
  __REG32 CHPNDG5           : 1;
  __REG32 CHPNDG6           : 1;
  __REG32 CHPNDG7           : 1;
  __REG32 CHPNDG8           : 1;
  __REG32 CHPNDG9           : 1;
  __REG32 CHPNDG10          : 1;
  __REG32 CHPNDG11          : 1;
  __REG32 CHPNDG12          : 1;
  __REG32 CHPNDG13          : 1;
  __REG32 CHPNDG14          : 1;
  __REG32 CHPNDG15          : 1;
  __REG32 CHPNDG16          : 1;
  __REG32 CHPNDG17          : 1;
  __REG32 CHPNDG18          : 1;
  __REG32 CHPNDG19          : 1;
  __REG32 CHPNDG20          : 1;
  __REG32 CHPNDG21          : 1;
  __REG32 CHPNDG22          : 1;
  __REG32 CHPNDG23          : 1;
  __REG32 CHPNDG24          : 1;
  __REG32 CHPNDG25          : 1;
  __REG32 CHPNDG26          : 1;
  __REG32 CHPNDG27          : 1;
  __REG32 CHPNDG28          : 1;
  __REG32 CHPNDG29          : 1;
  __REG32 CHPNDG30          : 1;
  __REG32 CHPNDG31          : 1;
} __adc_brspnd_bits;

/* Channel Control Registers */
typedef struct {
  __REG32 ICLSEL            : 2;
  __REG32                   : 2;
  __REG32 BNDSELL           : 2;
  __REG32 BNDSELU           : 2;
  __REG32 CHEVMODE          : 2;
  __REG32 SYNC              : 1;
  __REG32 REFSEL            : 1;
  __REG32 BNDSELX           : 4;
  __REG32 RESREG            : 4;
  __REG32 RESTBS            : 1;
  __REG32 RESPOS            : 1;
  __REG32                   : 6;
  __REG32 BWDCH             : 2;
  __REG32 BWDEN             : 1;
  __REG32                   : 1;
} __adc_gchctr_bits;

/* Channel Control Registers */
typedef struct {
  __REG32 STCS              : 5;
  __REG32                   : 3;
  __REG32 CMS               : 3;
  __REG32                   : 5;
  __REG32 STCE              : 5;
  __REG32                   : 3;
  __REG32 CME               : 3;
  __REG32                   : 5;
} __adc_globiclass_bits;

/* Result Control Registers */
typedef struct {
  __REG32                   :16;
  __REG32 DRCTR             : 4;
  __REG32 DMM               : 2;
  __REG32                   : 2;
  __REG32 WFR               : 1;
  __REG32 FEN               : 2;
  __REG32                   : 4;
  __REG32 SRGEN             : 1;
} __adc_grcr_bits;

/* Result Registers */
typedef struct {
  __REG32 RESULT            :16;
  __REG32 DRC               : 4;
  __REG32 CHNR              : 5;
  __REG32 EMUX              : 3;
  __REG32 CRS               : 2;
  __REG32 FCR               : 1;
  __REG32 VF                : 1;
} __adc_gres_bits;

/* Global Result Control Register */
typedef struct {
  __REG32                   :16;
  __REG32 DRCTR             : 4;
  __REG32                   : 4;
  __REG32 WFR               : 1;
  __REG32                   : 6;
  __REG32 SRGEN             : 1;
} __adc_globrcr_bits;

/* Valid Flag Register, Group x */
typedef struct {
  __REG32 VF0               : 1;
  __REG32 VF1               : 1;
  __REG32 VF2               : 1;
  __REG32 VF3               : 1;
  __REG32 VF4               : 1;
  __REG32 VF5               : 1;
  __REG32 VF6               : 1;
  __REG32 VF7               : 1;
  __REG32 VF8               : 1;
  __REG32 VF9               : 1;
  __REG32 VF10              : 1;
  __REG32 VF11              : 1;
  __REG32 VF12              : 1;
  __REG32 VF13              : 1;
  __REG32 VF14              : 1;
  __REG32 VF15              : 1;
  __REG32                   :16;
} __adc_gvfr_bits;

/* Alias Register, Group x */
typedef struct {
  __REG32 ALIAS0            : 5;
  __REG32                   : 3;
  __REG32 ALIAS1            : 5;
  __REG32                   :19;
} __adc_galias_bits;

/* Boundary Select Register, Group x */
typedef struct {
  __REG32 BOUNDARY0         :12;
  __REG32                   : 4;
  __REG32 BOUNDARY1         :12;
  __REG32                   : 4;
} __adc_gbound_bits;

/* Boundary Flag Register, Group x */
typedef struct {
  __REG32 BFL0              : 1;
  __REG32 BFL1              : 1;
  __REG32 BFL2              : 1;
  __REG32 BFL3              : 1;
  __REG32                   : 4;
  __REG32 BFA0              : 1;
  __REG32 BFA1              : 1;
  __REG32 BFA2              : 1;
  __REG32 BFA3              : 1;
  __REG32                   : 4;
  __REG32 BFI0              : 1;
  __REG32 BFI1              : 1;
  __REG32 BFI2              : 1;
  __REG32 BFI3              : 1;
  __REG32                   :12;
} __adc_gbfl_bits;

/* Boundary Flag Software Register, Group x */
typedef struct {
  __REG32 BFC0              : 1;
  __REG32 BFC1              : 1;
  __REG32 BFC2              : 1;
  __REG32 BFC3              : 1;
  __REG32                   :12;
  __REG32 BFS0              : 1;
  __REG32 BFS1              : 1;
  __REG32 BFS2              : 1;
  __REG32 BFS3              : 1;
  __REG32                   :12;
} __adc_gbfls_bits;

/* Boundary Flag Control Register, Group x */
typedef struct {
  __REG32 BFM0              : 4;
  __REG32 BFM1              : 4;
  __REG32 BFM2              : 4;
  __REG32 BFM3              : 4;
  __REG32                   :16;
} __adc_gbflc_bits;

/* Boundary Flag Control Register, Group x */
typedef struct {
  __REG32 STSEL             : 2;
  __REG32                   : 2;
  __REG32 EVALR1            : 1;
  __REG32 EVALR2            : 1;
  __REG32 EVALR3            : 1;
  __REG32                   :25;
} __adc_gsynctr_bits;

/* Global Test Functions Register */
typedef struct {
  __REG32 CDCH              : 4;
  __REG32 CDGR              : 4;
  __REG32 CDEN              : 1;
  __REG32 CDSEL             : 2;
  __REG32                   : 4;
  __REG32 CDWC              : 1;
  __REG32 PDD               : 1;
  __REG32 MDPD              : 1;
  __REG32 MDPU              : 1;
  __REG32                   : 4;
  __REG32 MDWC              : 1;
  __REG32 TRSW              : 1;
  __REG32 TREV              : 1;
  __REG32                   : 3;
  __REG32 SRGMODE           : 2;
  __REG32 SRGWC             : 1;
} __adc_globtf_bits;

/* Global Test Functions Register */
typedef struct {
  __REG32 EMUXSET           : 3;
  __REG32                   : 5;
  __REG32 EMUXACT           : 3;
  __REG32                   : 5;
  __REG32 EMUXCH            :10;
  __REG32 EMUXMODE          : 2;
  __REG32 EMXCOD            : 1;
  __REG32 EMXST             : 1;
  __REG32 EMXCSS            : 1;
  __REG32 EMXWC             : 1;
} __adc_gemuxctr_bits;

/* External Multiplexer Select Register */
typedef struct {
  __REG32 EMUXGRP0          : 4;
  __REG32 EMUXGRP1          : 4;
  __REG32                   :24;
} __adc_emuxsel_bits;

/* External Multiplexer Select Register */
typedef struct {
  __REG32 SEV0              : 1;
  __REG32 SEV1              : 1;
  __REG32                   :30;
} __adc_gseflag_bits;

/* External Multiplexer Select Register */
typedef struct {
  __REG32 CEV0              : 1;
  __REG32 CEV1              : 1;
  __REG32 CEV2              : 1;
  __REG32 CEV3              : 1;
  __REG32 CEV4              : 1;
  __REG32 CEV5              : 1;
  __REG32 CEV6              : 1;
  __REG32 CEV7              : 1;
  __REG32                   :24;
} __adc_gceflag_bits;

/* Result Event Flag Register, Group x */
typedef struct {
  __REG32 REV0              : 1;
  __REG32 REV1              : 1;
  __REG32 REV2              : 1;
  __REG32 REV3              : 1;
  __REG32 REV4              : 1;
  __REG32 REV5              : 1;
  __REG32 REV6              : 1;
  __REG32 REV7              : 1;
  __REG32 REV8              : 1;
  __REG32 REV9              : 1;
  __REG32 REV10             : 1;
  __REG32 REV11             : 1;
  __REG32 REV12             : 1;
  __REG32 REV13             : 1;
  __REG32 REV14             : 1;
  __REG32 REV15             : 1;
  __REG32                   :16;
} __adc_greflag_bits;

/* Global Event Flag Register */
typedef struct {
  __REG32 SEVGLB            : 1;
  __REG32                   : 7;
  __REG32 REVGLB            : 1;
  __REG32                   : 7;
  __REG32 SEVGLBCLR         : 1;
  __REG32                   : 7;
  __REG32 REVGLBCLR         : 1;
  __REG32                   : 7;
} __adc_globeflag_bits;

/* Global Event Flag Register */
typedef struct {
  __REG32 SEV0NP            : 4;
  __REG32 SEV1NP            : 4;
  __REG32                   :24;
} __adc_gsevnp_bits;

/* Channel Event Node Pointer Register 0, Group x */
typedef struct {
  __REG32 CEV0NP            : 4;
  __REG32 CEV1NP            : 4;
  __REG32 CEV2NP            : 4;
  __REG32 CEV3NP            : 4;
  __REG32 CEV4NP            : 4;
  __REG32 CEV5NP            : 4;
  __REG32 CEV6NP            : 4;
  __REG32 CEV7NP            : 4;
} __adc_gcevnp0_bits;

/* Channel Event Node Pointer Register 0, Group x */
typedef struct {
  __REG32 REV0NP            : 4;
  __REG32 REV1NP            : 4;
  __REG32 REV2NP            : 4;
  __REG32 REV3NP            : 4;
  __REG32 REV4NP            : 4;
  __REG32 REV5NP            : 4;
  __REG32 REV6NP            : 4;
  __REG32 REV7NP            : 4;
} __adc_grevnp0_bits;

/* Channel Event Node Pointer Register 1, Group x */
typedef struct {
  __REG32 REV8NP            : 4;
  __REG32 REV9NP            : 4;
  __REG32 REV10NP           : 4;
  __REG32 REV11NP           : 4;
  __REG32 REV12NP           : 4;
  __REG32 REV13NP           : 4;
  __REG32 REV14NP           : 4;
  __REG32 REV15NP           : 4;
} __adc_grevnp1_bits;

/* Global Event Node Pointer Register */
typedef struct {
  __REG32 SEV0NP            : 4;
  __REG32                   :12;
  __REG32 REV0NP            : 4;
  __REG32                   :12;
} __adc_globevnp_bits;

/* Global Event Node Pointer Register */
typedef struct {
  __REG32 AGSR0             : 1;
  __REG32 AGSR1             : 1;
  __REG32 AGSR2             : 1;
  __REG32 AGSR3             : 1;
  __REG32                   : 4;
  __REG32 ASSR0             : 1;
  __REG32 ASSR1             : 1;
  __REG32 ASSR2             : 1;
  __REG32 ASSR3             : 1;
  __REG32                   :20;
} __adc_gsract_bits;

/* Global Control Register */
typedef struct {
  __REG32 PRBC              : 3;
  __REG32                   : 1;
  __REG32 PCIS              : 2;
  __REG32                   : 2;
  __REG32 SUSCFG            : 2;
  __REG32 MSE0              : 1;
  __REG32 MSE1              : 1;
  __REG32 MSE2              : 1;
  __REG32 MSE3              : 1;
  __REG32 MSDE              : 2;
  __REG32                   :16;
} __capcom_gctrl_bits;

/* Global Status Register */
typedef struct {
  __REG32 S0I               : 1;
  __REG32 S1I               : 1;
  __REG32 S2I               : 1;
  __REG32 S3I               : 1;
  __REG32                   : 4;
  __REG32 PRB               : 1;
  __REG32                   :23;
} __capcom_gstat_bits;

/* Global Idle Set */
typedef struct {
  __REG32 SS0I              : 1;
  __REG32 SS1I              : 1;
  __REG32 SS2I              : 1;
  __REG32 SS3I              : 1;
  __REG32                   : 4;
  __REG32 CPRB              : 1;
  __REG32 PSIC              : 1;
  __REG32                   :22;
} __capcom_gidls_bits;

/* Global Idle Clear */
typedef struct {
  __REG32 CS0I              : 1;
  __REG32 CS1I              : 1;
  __REG32 CS2I              : 1;
  __REG32 CS3I              : 1;
  __REG32                   : 4;
  __REG32 SPRB              : 1;
  __REG32                   :23;
} __capcom_gidlc_bits;

/* Global Channel Set */
typedef struct {
  __REG32 S0SE              : 1;
  __REG32 S0DSE             : 1;
  __REG32 S0PSE             : 1;
  __REG32                   : 1;
  __REG32 S1SE              : 1;
  __REG32 S1DSE             : 1;
  __REG32 S1PSE             : 1;
  __REG32                   : 1;
  __REG32 S2SE              : 1;
  __REG32 S2DSE             : 1;
  __REG32 S2PSE             : 1;
  __REG32                   : 1;
  __REG32 S3SE              : 1;
  __REG32 S3DSE             : 1;
  __REG32 S3PSE             : 1;
  __REG32                   : 1;
  __REG32 S0STS             : 1;
  __REG32 S1STS             : 1;
  __REG32 S2STS             : 1;
  __REG32 S3STS             : 1;
  __REG32                   :12;
} __capcom_gcss_bits;

/* Global Channel clear */
typedef struct {
  __REG32 S0SC              : 1;
  __REG32 S0DSC             : 1;
  __REG32 S0PSC             : 1;
  __REG32                   : 1;
  __REG32 S1SC              : 1;
  __REG32 S1DSC             : 1;
  __REG32 S1PSC             : 1;
  __REG32                   : 1;
  __REG32 S2SC              : 1;
  __REG32 S2DSC             : 1;
  __REG32 S2PSC             : 1;
  __REG32                   : 1;
  __REG32 S3SC              : 1;
  __REG32 S3DSC             : 1;
  __REG32 S3PSC             : 1;
  __REG32                   : 1;
  __REG32 S0STC             : 1;
  __REG32 S1STC             : 1;
  __REG32 S2STC             : 1;
  __REG32 S3STC             : 1;
  __REG32                   :12;
} __capcom_gcsc_bits;

/* Global Channel Status */
typedef struct {
  __REG32 S0SS              : 1;
  __REG32 S0DSS             : 1;
  __REG32 S0PSS             : 1;
  __REG32                   : 1;
  __REG32 S1SS              : 1;
  __REG32 S1DSS             : 1;
  __REG32 S1PSS             : 1;
  __REG32                   : 1;
  __REG32 S2SS              : 1;
  __REG32 S2DSS             : 1;
  __REG32 S2PSS             : 1;
  __REG32                   : 1;
  __REG32 S3SS              : 1;
  __REG32 S3DSS             : 1;
  __REG32 S3PSS             : 1;
  __REG32                   : 1;
  __REG32 CC40ST            : 1;
  __REG32 CC41ST            : 1;
  __REG32 CC42ST            : 1;
  __REG32 CC43ST            : 1;
  __REG32                   :12;
} __capcom_gcst_bits;

/* Extended Capture Mode Read */
typedef struct {
  __REG32 CAPV              :16;
  __REG32 FPCV              : 4;
  __REG32 SPTR              : 2;
  __REG32 VPTR              : 2;
  __REG32 FFL               : 1;
  __REG32                   : 7;
} __capcom_ecrd_bits;

/* Module Identification */
typedef struct {
  __REG32 MODR              : 8;
  __REG32 MODT              : 8;
  __REG32 MODN              :16;
} __capcom_midr_bits;

/* Input Selector Configuration */
typedef struct {
  __REG32 EV0IS             : 4;
  __REG32 EV1IS             : 4;
  __REG32 EV2IS             : 4;
  __REG32                   : 4;
  __REG32 EV0EM             : 2;
  __REG32 EV1EM             : 2;
  __REG32 EV2EM             : 2;
  __REG32 EV0LM             : 1;
  __REG32 EV1LM             : 1;
  __REG32 EV2LM             : 1;
  __REG32 LPF0M             : 2;
  __REG32 LPF1M             : 2;
  __REG32 LPF2M             : 2;
  __REG32                   : 1;
} __ccins_bits;

/* Connection Matrix Control */
typedef struct {
  __REG32 STRTS             : 2;
  __REG32 ENDS              : 2;
  __REG32 CAP0S             : 2;
  __REG32 CAP1S             : 2;
  __REG32 GATES             : 2;
  __REG32 UDS               : 2;
  __REG32 LDS               : 2;
  __REG32 CNTS              : 2;
  __REG32 OFS               : 1;
  __REG32 TS                : 1;
  __REG32 MOS               : 2;
  __REG32 TCE               : 1;
  __REG32                   :11;
} __cccmc_bits;

/* Slice Timer Status */
typedef struct {
  __REG32 TRB               : 1;
  __REG32 CDIR              : 1;
  __REG32                   :30;
} __cctst_bits;

/* Slice Timer Run Set */
typedef struct {
  __REG32 TRBS              : 1;
  __REG32                   :31;
} __cctcset_bits;

/* Slice Timer Clear */
typedef struct {
  __REG32 TRBC              : 1;
  __REG32 TCC               : 1;
  __REG32 DITC              : 1;
  __REG32                   :29;
} __cctcclr_bits;

/* Slice Timer Control */
typedef struct {
  __REG32 TCM               : 1;
  __REG32 TSSM              : 1;
  __REG32 CLST              : 1;
  __REG32 CMOD              : 1;
  __REG32 ECM               : 1;
  __REG32 CAPC              : 2;
  __REG32                   : 1;
  __REG32 ENDM              : 2;
  __REG32 STRM              : 1;
  __REG32 SCE               : 1;
  __REG32 CCS               : 1;
  __REG32 DITHE             : 2;
  __REG32 DIM               : 1;
  __REG32 FPE               : 1;
  __REG32 TRAPE             : 1;
  __REG32                   : 3;
  __REG32 TRPSE             : 1;
  __REG32 TRPSW             : 1;
  __REG32 EMS               : 1;
  __REG32 EMT               : 1;
  __REG32 MCME              : 1;
  __REG32                   : 6;
} __cctc_bits;

/* Passive Level Config */
typedef struct {
  __REG32 PSL               : 1;
  __REG32                   :31;
} __ccpsl_bits;

/* Passive Level Config */
typedef struct {
  __REG32 DCV               : 4;
  __REG32                   : 4;
  __REG32 DCNT              : 4;
  __REG32                   :20;
} __ccdit_bits;

/* Dither Shadow Register */
typedef struct {
  __REG32 DCVS              : 4;
  __REG32                   :28;
} __ccdits_bits;

/* Prescaler Control */
typedef struct {
  __REG32 PSIV              : 4;
  __REG32                   :28;
} __ccpsc_bits;

/* Prescaler Control */
typedef struct {
  __REG32 PCMP              : 4;
  __REG32                   : 4;
  __REG32 PVAL              : 4;
  __REG32                   :20;
} __ccfpc_bits;

/* Floating Prescaler Shadow */
typedef struct {
  __REG32 PCMP              : 4;
  __REG32                   :28;
} __ccfpcs_bits;

/* Timer Period Value */
typedef struct {
  __REG32 PR                :16;
  __REG32                   :16;
} __ccpr_bits;

/* Timer Shadow Period Value */
typedef struct {
  __REG32 PRS               :16;
  __REG32                   :16;
} __ccprs_bits;

/* Timer Compare Value */
typedef struct {
  __REG32 CR                :16;
  __REG32                   :16;
} __cccr_bits;

/* Timer Shadow Compare Value */
typedef struct {
  __REG32 CRS               :16;
  __REG32                   :16;
} __cccrs_bits;

/* Timer Value */
typedef struct {
  __REG32 TVAL              :16;
  __REG32                   :16;
} __cctimer_bits;

/* Capture Register n */
typedef struct {
  __REG32 CAPTV             :16;
  __REG32 FPCV              : 4;
  __REG32 FFL               : 1;
  __REG32                   :11;
} __cccv_bits;

/* Interrupt Status */
typedef struct {
  __REG32 PMUS              : 1;
  __REG32 OMDS              : 1;
  __REG32 CMUS              : 1;
  __REG32 CMDS              : 1;
  __REG32                   : 4;
  __REG32 E0AS              : 1;
  __REG32 E1AS              : 1;
  __REG32 E2AS              : 1;
  __REG32 TRPF              : 1;
  __REG32                   :20;
} __ccints_bits;

/* Interrupt Enable Control */
typedef struct {
  __REG32 PME               : 1;
  __REG32 OME               : 1;
  __REG32 CMUE              : 1;
  __REG32 CMDE              : 1;
  __REG32                   : 4;
  __REG32 E0AE              : 1;
  __REG32 E1AE              : 1;
  __REG32 E2AE              : 1;
  __REG32                   :21;
} __ccinte_bits;

/* Interrupt Enable Control */
typedef struct {
  __REG32 POSR              : 2;
  __REG32 CMSR              : 2;
  __REG32                   : 4;
  __REG32 E0SR              : 2;
  __REG32 E1SR              : 2;
  __REG32 E2SR              : 2;
  __REG32                   :18;
} __ccsrs_bits;

/* Interrupt Status Set */
typedef struct {
  __REG32 SPM               : 1;
  __REG32 SOM               : 1;
  __REG32 SCMU              : 1;
  __REG32 SCMD              : 1;
  __REG32                   : 4;
  __REG32 SE0A              : 1;
  __REG32 SE1A              : 1;
  __REG32 SE2A              : 1;
  __REG32 STRPF             : 1;
  __REG32                   :20;
} __ccsws_bits;

/* Interrupt Status Clear */
typedef struct {
  __REG32 RPM               : 1;
  __REG32 ROM               : 1;
  __REG32 RCMU              : 1;
  __REG32 RCMD              : 1;
  __REG32                   : 4;
  __REG32 RE0A              : 1;
  __REG32 RE1A              : 1;
  __REG32 RE2A              : 1;
  __REG32 RTRPF             : 1;
  __REG32                   :20;
} __ccswr_bits;

/* Global Control Register */
typedef struct {
  __REG32 PRBC              : 3;
  __REG32                   : 1;
  __REG32 PCIS              : 2;
  __REG32                   : 2;
  __REG32 SUSCFG            : 2;
  __REG32 MSE0              : 1;
  __REG32 MSE1              : 1;
  __REG32 MSE2              : 1;
  __REG32 MSE3              : 1;
  __REG32 MSDE              : 2;
  __REG32                   :16;
} __capcom8_gctrl_bits;

/* Global Status Register */
typedef struct {
  __REG32 S0I               : 1;
  __REG32 S1I               : 1;
  __REG32 S2I               : 1;
  __REG32 S3I               : 1;
  __REG32                   : 4;
  __REG32 PRB               : 1;
  __REG32                   : 1;
  __REG32 PCRB              : 1;
  __REG32                   :21;
} __capcom8_gstat_bits;

/* Global Idle Set */
typedef struct {
  __REG32 SS0I              : 1;
  __REG32 SS1I              : 1;
  __REG32 SS2I              : 1;
  __REG32 SS3I              : 1;
  __REG32                   : 4;
  __REG32 CPRB              : 1;
  __REG32 PSIC              : 1;
  __REG32 CPCH              : 1;
  __REG32                   :21;
} __capcom8_gidls_bits;

/* Global Idle Clear */
typedef struct {
  __REG32 CS0I              : 1;
  __REG32 CS1I              : 1;
  __REG32 CS2I              : 1;
  __REG32 CS3I              : 1;
  __REG32                   : 4;
  __REG32 SPRB              : 1;
  __REG32                   : 1;
  __REG32 SPCH              : 1;
  __REG32                   :21;
} __capcom8_gidlc_bits;

/* Global Channel Set */
typedef struct {
  __REG32 S0SE              : 1;
  __REG32 S0DSE             : 1;
  __REG32 S0PSE             : 1;
  __REG32                   : 1;
  __REG32 S1SE              : 1;
  __REG32 S1DSE             : 1;
  __REG32 S1PSE             : 1;
  __REG32                   : 1;
  __REG32 S2SE              : 1;
  __REG32 S2DSE             : 1;
  __REG32 S2PSE             : 1;
  __REG32                   : 1;
  __REG32 S3SE              : 1;
  __REG32 S3DSE             : 1;
  __REG32 S3PSE             : 1;
  __REG32                   : 1;
  __REG32 S0STS             : 1;
  __REG32 S1STS             : 1;
  __REG32 S2STS             : 1;
  __REG32 S3STS             : 1;
  __REG32 S0ST2S            : 1;
  __REG32 S1ST2S            : 1;
  __REG32 S2ST2S            : 1;
  __REG32 S3ST2S            : 1;
  __REG32                   : 8;
} __capcom8_gcss_bits;

/* Global Channel Clear */
typedef struct {
  __REG32 S0SC              : 1;
  __REG32 S0DSC             : 1;
  __REG32 S0PSC             : 1;
  __REG32                   : 1;
  __REG32 S1SC              : 1;
  __REG32 S1DSC             : 1;
  __REG32 S1PSC             : 1;
  __REG32                   : 1;
  __REG32 S2SC              : 1;
  __REG32 S2DSC             : 1;
  __REG32 S2PSC             : 1;
  __REG32                   : 1;
  __REG32 S3SC              : 1;
  __REG32 S3DSC             : 1;
  __REG32 S3PSC             : 1;
  __REG32                   : 1;
  __REG32 S0STC             : 1;
  __REG32 S1STC             : 1;
  __REG32 S2STC             : 1;
  __REG32 S3STC             : 1;
  __REG32 S0ST2C            : 1;
  __REG32 S1ST2C            : 1;
  __REG32 S2ST2C            : 1;
  __REG32 S3ST2C            : 1;
  __REG32                   : 8;
} __capcom8_gcsc_bits;

/* Global Channel Status */
typedef struct {
  __REG32 S0SS              : 1;
  __REG32 S0DSS             : 1;
  __REG32 S0PSS             : 1;
  __REG32                   : 1;
  __REG32 S1SS              : 1;
  __REG32 S1DSS             : 1;
  __REG32 S1PSS             : 1;
  __REG32                   : 1;
  __REG32 S2SS              : 1;
  __REG32 S2DSS             : 1;
  __REG32 S2PSS             : 1;
  __REG32                   : 1;
  __REG32 S3SS              : 1;
  __REG32 S3DSS             : 1;
  __REG32 S3PSS             : 1;
  __REG32                   : 1;
  __REG32 CC80ST            : 1;
  __REG32 CC81ST            : 1;
  __REG32 CC82ST            : 1;
  __REG32 CC83ST            : 1;
  __REG32 CC80ST2           : 1;
  __REG32 CC81ST2           : 1;
  __REG32 CC82ST2           : 1;
  __REG32 CC83ST2           : 1;
  __REG32                   : 8;
} __capcom8_gcst_bits;

/* Parity Checker Configuration */
typedef struct {
  __REG32 PASE              : 1;
  __REG32 PACS              : 2;
  __REG32 PISEL             : 2;
  __REG32 PCDS              : 2;
  __REG32 PCTS              : 1;
  __REG32                   : 7;
  __REG32 PCST              : 1;
  __REG32 PCSEL0            : 4;
  __REG32 PCSEL1            : 4;
  __REG32 PCSEL2            : 4;
  __REG32 PCSEL3            : 4;
} __capcom8_gpchk_bits;

/* Extended Capture Mode Read */
typedef struct {
  __REG32 CAPV              :16;
  __REG32 FPCV              : 4;
  __REG32 SPTR              : 2;
  __REG32 VPTR              : 2;
  __REG32 FFL               : 1;
  __REG32                   : 7;
} __capcom8_ecrd_bits;

/* Module Identification */
typedef struct {
  __REG32 MODR              : 8;
  __REG32 MODT              : 8;
  __REG32 MODN              :16;
} __capcom8_midr_bits;

/* Input Selector Configuration */
typedef struct {
  __REG32 EV0IS             : 4;
  __REG32 EV1IS             : 4;
  __REG32 EV2IS             : 4;
  __REG32                   : 4;
  __REG32 EV0EM             : 2;
  __REG32 EV1EM             : 2;
  __REG32 EV2EM             : 2;
  __REG32 EV0LM             : 1;
  __REG32 EV1LM             : 1;
  __REG32 EV2LM             : 1;
  __REG32 LPF0M             : 2;
  __REG32 LPF1M             : 2;
  __REG32 LPF2M             : 2;
  __REG32                   : 1;
} __cc8ins_bits;

/* Connection Matrix Control */
typedef struct {
  __REG32 STRTS             : 2;
  __REG32 ENDS              : 2;
  __REG32 CAP0S             : 2;
  __REG32 CAP1S             : 2;
  __REG32 GATES             : 2;
  __REG32 UDS               : 2;
  __REG32 LDS               : 2;
  __REG32 CNTS              : 2;
  __REG32 OFS               : 1;
  __REG32 TS                : 1;
  __REG32 MOS               : 2;
  __REG32 TCE               : 1;
  __REG32                   :11;
} __cc8cmc_bits;

/* Slice Timer Status */
typedef struct {
  __REG32 TRB               : 1;
  __REG32 CDIR              : 1;
  __REG32                   : 1;
  __REG32 DTR1              : 1;
  __REG32 DTR2              : 1;
  __REG32                   :27;
} __cc8tcst_bits;

/* Slice Timer Run Set */
typedef struct {
  __REG32 TRBS              : 1;
  __REG32                   :31;
} __cc8tcset_bits;

/* Slice Timer Clear */
typedef struct {
  __REG32 TRBC              : 1;
  __REG32 TCC               : 1;
  __REG32 DITC              : 1;
  __REG32 DTC1C             : 1;
  __REG32 DTC2C             : 1;
  __REG32                   :27;
} __cc8tcclr_bits;

/* Slice Timer Control */
typedef struct {
  __REG32 TCM               : 1;
  __REG32 TSSM              : 1;
  __REG32 CLST              : 1;
  __REG32 CMOD              : 1;
  __REG32 ECM               : 1;
  __REG32 CAPC              : 2;
  __REG32 TLS               : 1;
  __REG32 ENDM              : 2;
  __REG32 STRM              : 1;
  __REG32 SCE               : 1;
  __REG32 CCS               : 1;
  __REG32 DITHE             : 2;
  __REG32 DIM               : 1;
  __REG32 FPE               : 1;
  __REG32 TRAPE0            : 1;
  __REG32 TRAPE1            : 1;
  __REG32 TRAPE2            : 1;
  __REG32 TRAPE3            : 1;
  __REG32 TRPSE             : 1;
  __REG32 TRPSW             : 1;
  __REG32 EMS               : 1;
  __REG32 EMT               : 1;
  __REG32 MCME1             : 1;
  __REG32 MCME2             : 1;
  __REG32 EME               : 2;
  __REG32 STOS              : 2;
  __REG32                   : 1;
} __cc8tc_bits;

/* Passive Level Config */
typedef struct {
  __REG32 PSL11             : 1;
  __REG32 PSL12             : 1;
  __REG32 PSL21             : 1;
  __REG32 PSL22             : 1;
  __REG32                   :28;
} __cc8psl_bits;

/* Passive Level Config */
typedef struct {
  __REG32 DCV               : 4;
  __REG32                   : 4;
  __REG32 DCNT              : 4;
  __REG32                   :20;
} __cc8dit_bits;

/* Dither Shadow Register */
typedef struct {
  __REG32 DCVS              : 4;
  __REG32                   :28;
} __cc8dits_bits;

/* Prescaler Control */
typedef struct {
  __REG32 PSIV              : 4;
  __REG32                   :28;
} __cc8psc_bits;

/* Prescaler Control */
typedef struct {
  __REG32 PCMP              : 4;
  __REG32                   : 4;
  __REG32 PVAL              : 4;
  __REG32                   :20;
} __cc8fpc_bits;

/* Floating Prescaler Shadow */
typedef struct {
  __REG32 PCMP              : 4;
  __REG32                   :28;
} __cc8fpcs_bits;

/* Timer Period Value */
typedef struct {
  __REG32 PR                :16;
  __REG32                   :16;
} __cc8pr_bits;

/* Timer Shadow Period Value */
typedef struct {
  __REG32 PRS               :16;
  __REG32                   :16;
} __cc8prs_bits;

/* Timer Compare Value */
typedef struct {
  __REG32 CR                :16;
  __REG32                   :16;
} __cc8cr_bits;

/* Timer Shadow Compare Value */
typedef struct {
  __REG32 CRS               :16;
  __REG32                   :16;
} __cc8crs_bits;

/* Channel Control */
typedef struct {
  __REG32 ASE               : 1;
  __REG32 OCS1              : 1;
  __REG32 OCS2              : 1;
  __REG32 OCS3              : 1;
  __REG32 OCS4              : 1;
  __REG32                   :27;
} __cc8chc_bits;

/* Dead Time Control */
typedef struct {
  __REG32 DTE1              : 1;
  __REG32 DTE2              : 1;
  __REG32 DCEN1             : 1;
  __REG32 DCEN2             : 1;
  __REG32 DCEN3             : 1;
  __REG32 DCEN4             : 1;
  __REG32 DTCC              : 2;
  __REG32                   :24;
} __cc8dtc_bits;

/* Channel 1 Dead Time Values */
typedef struct {
  __REG32 DTR             : 8;
  __REG32 DTF               : 8;
  __REG32                   :16;
} __cc8dcr_bits;

/* Timer Value */
typedef struct {
  __REG32 TVAL              :16;
  __REG32                   :16;
} __cc8timer_bits;

/* Capture Register n */
typedef struct {
  __REG32 CAPTV             :16;
  __REG32 FPCV              : 4;
  __REG32 FFL               : 1;
  __REG32                   :11;
} __cc8cv_bits;

/* Interrupt Status */
typedef struct {
  __REG32 PMUS              : 1;
  __REG32 OMDS              : 1;
  __REG32 CMU1S             : 1;
  __REG32 CMD1S             : 1;
  __REG32 CMU2S             : 1;
  __REG32 CMD2S             : 1;
  __REG32                   : 2;
  __REG32 E0AS              : 1;
  __REG32 E1AS              : 1;
  __REG32 E2AS              : 1;
  __REG32 TRPF              : 1;
  __REG32                   :20;
} __cc8ints_bits;

/* Interrupt Enable Control */
typedef struct {
  __REG32 PME               : 1;
  __REG32 OME               : 1;
  __REG32 CMU1E             : 1;
  __REG32 CMD1E             : 1;
  __REG32 CMU2E             : 1;
  __REG32 CMD2E             : 1;
  __REG32                   : 2;
  __REG32 E0AE              : 1;
  __REG32 E1AE              : 1;
  __REG32 E2AE              : 1;
  __REG32                   :21;
} __cc8inte_bits;

/* Interrupt Enable Control */
typedef struct {
  __REG32 POSR              : 2;
  __REG32 CM1SR             : 2;
  __REG32 CM2SR             : 2;
  __REG32                   : 2;
  __REG32 E0SR              : 2;
  __REG32 E1SR              : 2;
  __REG32 E2SR              : 2;
  __REG32                   :18;
} __cc8srs_bits;

/* Interrupt Status Set */
typedef struct {
  __REG32 SPM               : 1;
  __REG32 SOM               : 1;
  __REG32 SCM1U             : 1;
  __REG32 SCM1D             : 1;
  __REG32 SCM2U             : 1;
  __REG32 SCM2D             : 1;
  __REG32                   : 2;
  __REG32 SE0A              : 1;
  __REG32 SE1A              : 1;
  __REG32 SE2A              : 1;
  __REG32 STRPF             : 1;
  __REG32                   :20;
} __cc8sws_bits;

/* Interrupt Status Clear */
typedef struct {
  __REG32 RPM               : 1;
  __REG32 ROM               : 1;
  __REG32 RCM1U             : 1;
  __REG32 RCM1D             : 1;
  __REG32 RCM2U             : 1;
  __REG32 RCM2D             : 1;
  __REG32                   : 2;
  __REG32 RE0A              : 1;
  __REG32 RE1A              : 1;
  __REG32 RE2A              : 1;
  __REG32 RTRPF             : 1;
  __REG32                   :20;
} __cc8swr_bits;

/* POSIF configuration */
typedef struct {
  __REG32 FSEL              : 2;
  __REG32 QDCM              : 1;
  __REG32                   : 1;
  __REG32 HIDG              : 1;
  __REG32 MCUE              : 1;
  __REG32                   : 2;
  __REG32 INSEL0            : 2;
  __REG32 INSEL1            : 2;
  __REG32 INSEL2            : 2;
  __REG32                   : 2;
  __REG32 DSEL              : 1;
  __REG32 SPES              : 1;
  __REG32 MSETS             : 3;
  __REG32 MSES              : 1;
  __REG32 MSYNS             : 2;
  __REG32 EWIS              : 2;
  __REG32 EWIE              : 1;
  __REG32 EWIL              : 1;
  __REG32 LPC               : 3;
  __REG32                   : 1;
} __posif_pconf_bits;

/* POSIF Suspend Config */
typedef struct {
  __REG32 QSUS              : 2;
  __REG32 MSUS              : 2;
  __REG32                   :28;
} __posif_psus_bits;

/* POSIF Suspend Config */
typedef struct {
  __REG32 SRB               : 1;
  __REG32                   :31;
} __posif_pruns_bits;

/* POSIF Run Bit Clear */
typedef struct {
  __REG32 CRB               : 1;
  __REG32 CSM               : 1;
  __REG32                   :30;
} __posif_prunc_bits;

/* POSIF Run Bit Status */
typedef struct {
  __REG32 RB                : 1;
  __REG32                   :31;
} __posif_prun_bits;

/* POSIF Module Identification register */
typedef struct {
  __REG32 MODR              : 8;
  __REG32 MODT              : 8;
  __REG32 MODN              :16;
} __posif_midr_bits;

/* POSIF Hall Sensor Patterns */
typedef struct {
  __REG32 HCP               : 3;
  __REG32 HEP               : 3;
  __REG32                   :26;
} __posif_halp_bits;

/* POSIF Hall Sensor Shadow Patterns */
typedef struct {
  __REG32 HCPS              : 3;
  __REG32 HEPS              : 3;
  __REG32                   :26;
} __posif_halps_bits;

/* Multi Channel Pattern */
typedef struct {
  __REG32 MCMP              :16;
  __REG32                   :16;
} __posif_mcm_bits;

/* Multi Channel Shadow Pattern */
typedef struct {
  __REG32 MCMPS             :16;
  __REG32                   :16;
} __posif_mcsm_bits;

/* Multi Channel Pattern Control set */
typedef struct {
  __REG32 MNPS              : 1;
  __REG32 STHR              : 1;
  __REG32 STMR              : 1;
  __REG32                   :29;
} __posif_mcms_bits;

/* Multi Channel Pattern Control clear */
typedef struct {
  __REG32 MNPC              : 1;
  __REG32 MPC               : 1;
  __REG32                   :30;
} __posif_mcmc_bits;

/* Multi Channel Pattern Control flag */
typedef struct {
  __REG32 MSS               : 1;
  __REG32                   :31;
} __posif_mcmf_bits;

/* Quadrature Decoder Control */
typedef struct {
  __REG32 PALS              : 1;
  __REG32 PBLS              : 1;
  __REG32 PHS               : 1;
  __REG32                   : 1;
  __REG32 ICM               : 2;
  __REG32                   : 2;
  __REG32 DVAL              : 1;
  __REG32                   :23;
} __posif_qdc_bits;

/* Quadrature Decoder Control */
typedef struct {
  __REG32 CHES              : 1;
  __REG32 WHES              : 1;
  __REG32 HIES              : 1;
  __REG32                   : 1;
  __REG32 MSTS              : 1;
  __REG32                   : 3;
  __REG32 INDXS             : 1;
  __REG32 ERRS              : 1;
  __REG32 CNTS              : 1;
  __REG32 DIRS              : 1;
  __REG32 PCLKS             : 1;
  __REG32                   :19;
} __posif_pflg_bits;

/* POSIF Interrupt Enable */
typedef struct {
  __REG32 ECHE              : 1;
  __REG32 EWHE              : 1;
  __REG32 EHIE              : 1;
  __REG32                   : 1;
  __REG32 EMST              : 1;
  __REG32                   : 3;
  __REG32 EINDX             : 1;
  __REG32 EERR              : 1;
  __REG32 ECNT              : 1;
  __REG32 EDIR              : 1;
  __REG32 EPCLK             : 1;
  __REG32                   : 3;
  __REG32 CHESEL            : 1;
  __REG32 WHESEL            : 1;
  __REG32 HIESEL            : 1;
  __REG32                   : 1;
  __REG32 MSTSEL            : 1;
  __REG32                   : 3;
  __REG32 INDSEL            : 1;
  __REG32 ERRSEL            : 1;
  __REG32 CNTSEL            : 1;
  __REG32 DIRSEL            : 1;
  __REG32 PCLSEL            : 1;
  __REG32                   : 3;
} __posif_pflge_bits;

/* POSIF Interrupt Set */
typedef struct {
  __REG32 SCHE              : 1;
  __REG32 SWHE              : 1;
  __REG32 SHIE              : 1;
  __REG32                   : 1;
  __REG32 SMST              : 1;
  __REG32                   : 3;
  __REG32 SINDX             : 1;
  __REG32 SERR              : 1;
  __REG32 SCNT              : 1;
  __REG32 SDIR              : 1;
  __REG32 SPCLK             : 1;
  __REG32                   :19;
} __posif_spflg_bits;

/* POSIF Interrupt Clear */
typedef struct {
  __REG32 RCHE              : 1;
  __REG32 RWHE              : 1;
  __REG32 RHIE              : 1;
  __REG32                   : 1;
  __REG32 RMST              : 1;
  __REG32                   : 3;
  __REG32 RINDX             : 1;
  __REG32 RERR              : 1;
  __REG32 RCNT              : 1;
  __REG32 RDIR              : 1;
  __REG32 RPCLK             : 1;
  __REG32                   :19;
} __posif_rpflg_bits;

/* Port n Input/Output Control Register 0 */
typedef struct {
  __REG32                   : 3;
  __REG32 PC0               : 5;
  __REG32                   : 3;
  __REG32 PC1               : 5;
  __REG32                   : 3;
  __REG32 PC2               : 5;
  __REG32                   : 3;
  __REG32 PC3               : 5;
} __p_iocr0_bits;

/* Port n Input/Output Control Register 4 */
typedef struct {
  __REG32                   : 3;
  __REG32 PC4               : 5;
  __REG32                   : 3;
  __REG32 PC5               : 5;
  __REG32                   : 3;
  __REG32 PC6               : 5;
  __REG32                   : 3;
  __REG32 PC7               : 5;
} __p_iocr4_bits;

/* Port n Input/Output Control Register 8 */
typedef struct {
  __REG32                   : 3;
  __REG32 PC8               : 5;
  __REG32                   : 3;
  __REG32 PC9               : 5;
  __REG32                   : 3;
  __REG32 PC10              : 5;
  __REG32                   : 3;
  __REG32 PC11              : 5;
} __p_iocr8_bits;

/* Port n Input/Output Control Register 12 */
typedef struct {
  __REG32                   : 3;
  __REG32 PC12              : 5;
  __REG32                   : 3;
  __REG32 PC13              : 5;
  __REG32                   : 3;
  __REG32 PC14              : 5;
  __REG32                   : 3;
  __REG32 PC15              : 5;
} __p_iocr12_bits;

/* Port n Pad Driver Mode 0 Register */
typedef struct {
  __REG32 PD0               : 3;
  __REG32                   : 1;
  __REG32 PD1               : 3;
  __REG32                   : 1;
  __REG32 PD2               : 3;
  __REG32                   : 1;
  __REG32 PD3               : 3;
  __REG32                   : 1;
  __REG32 PD4               : 3;
  __REG32                   : 1;
  __REG32 PD5               : 3;
  __REG32                   : 1;
  __REG32 PD6               : 3;
  __REG32                   : 1;
  __REG32 PD7               : 3;
  __REG32                   : 1;
} __p_pdr0_bits;

/* Port n Pad Driver Mode 1 Register */
typedef struct {
  __REG32 PD8               : 3;
  __REG32                   : 1;
  __REG32 PD9               : 3;
  __REG32                   : 1;
  __REG32 PD10              : 3;
  __REG32                   : 1;
  __REG32 PD11              : 3;
  __REG32                   : 1;
  __REG32 PD12              : 3;
  __REG32                   : 1;
  __REG32 PD13              : 3;
  __REG32                   : 1;
  __REG32 PD14              : 3;
  __REG32                   : 1;
  __REG32 PD15              : 3;
  __REG32                   : 1;
} __p_pdr1_bits;

/* Port n Pin Function Decision Control Register */
typedef struct {
  __REG32 PDIS0             : 1;
  __REG32 PDIS1             : 1;
  __REG32 PDIS2             : 1;
  __REG32 PDIS3             : 1;
  __REG32 PDIS4             : 1;
  __REG32 PDIS5             : 1;
  __REG32 PDIS6             : 1;
  __REG32 PDIS7             : 1;
  __REG32 PDIS8             : 1;
  __REG32 PDIS9             : 1;
  __REG32 PDIS10            : 1;
  __REG32 PDIS11            : 1;
  __REG32 PDIS12            : 1;
  __REG32 PDIS13            : 1;
  __REG32 PDIS14            : 1;
  __REG32 PDIS15            : 1;
  __REG32                   :16;
} __p_pdisc_bits;

/* Port 14 Pin Function Decision Control Register */
typedef struct {
  __REG32 PDIS0             : 1;
  __REG32 PDIS1             : 1;
  __REG32 PDIS2             : 1;
  __REG32 PDIS3             : 1;
  __REG32 PDIS4             : 1;
  __REG32 PDIS5             : 1;
  __REG32 PDIS6             : 1;
  __REG32 PDIS7             : 1;
  __REG32 PDIS8             : 1;
  __REG32 PDIS9             : 1;
  __REG32                   : 2;
  __REG32 PDIS12            : 1;
  __REG32 PDIS13            : 1;
  __REG32 PDIS14            : 1;
  __REG32 PDIS15            : 1;
  __REG32                   :16;
} __p_pdisc14_bits;

/* Port 15 Pin Function Decision Control Register */
typedef struct {
  __REG32                   : 2;
  __REG32 PDIS2             : 1;
  __REG32 PDIS3             : 1;
  __REG32 PDIS4             : 1;
  __REG32 PDIS5             : 1;
  __REG32 PDIS6             : 1;
  __REG32 PDIS7             : 1;
  __REG32 PDIS8             : 1;
  __REG32 PDIS9             : 1;
  __REG32                   : 2;
  __REG32 PDIS12            : 1;
  __REG32 PDIS13            : 1;
  __REG32 PDIS14            : 1;
  __REG32 PDIS15            : 1;
  __REG32                   :16;
} __p_pdisc15_bits;

/* Port n Output Modification Register */
typedef struct {
  __REG32 PS0               : 1;
  __REG32 PS1               : 1;
  __REG32 PS2               : 1;
  __REG32 PS3               : 1;
  __REG32 PS4               : 1;
  __REG32 PS5               : 1;
  __REG32 PS6               : 1;
  __REG32 PS7               : 1;
  __REG32 PS8               : 1;
  __REG32 PS9               : 1;
  __REG32 PS10              : 1;
  __REG32 PS11              : 1;
  __REG32 PS12              : 1;
  __REG32 PS13              : 1;
  __REG32 PS14              : 1;
  __REG32 PS15              : 1;
  __REG32 PR0               : 1;
  __REG32 PR1               : 1;
  __REG32 PR2               : 1;
  __REG32 PR3               : 1;
  __REG32 PR4               : 1;
  __REG32 PR5               : 1;
  __REG32 PR6               : 1;
  __REG32 PR7               : 1;
  __REG32 PR8               : 1;
  __REG32 PR9               : 1;
  __REG32 PR10              : 1;
  __REG32 PR11              : 1;
  __REG32 PR12              : 1;
  __REG32 PR13              : 1;
  __REG32 PR14              : 1;
  __REG32 PR15              : 1;
} __p_omr_bits;

/* Port n Output Register */
typedef struct {
  __REG32 P0                : 1;
  __REG32 P1                : 1;
  __REG32 P2                : 1;
  __REG32 P3                : 1;
  __REG32 P4                : 1;
  __REG32 P5                : 1;
  __REG32 P6                : 1;
  __REG32 P7                : 1;
  __REG32 P8                : 1;
  __REG32 P9                : 1;
  __REG32 P10               : 1;
  __REG32 P11               : 1;
  __REG32 P12               : 1;
  __REG32 P13               : 1;
  __REG32 P14               : 1;
  __REG32 P15               : 1;
  __REG32                   :16;
} __p_out_bits;

/* Port n Pin Power Save Register */
typedef struct {
  __REG32 PPS0                : 1;
  __REG32 PPS1                : 1;
  __REG32 PPS2                : 1;
  __REG32 PPS3                : 1;
  __REG32 PPS4                : 1;
  __REG32 PPS5                : 1;
  __REG32 PPS6                : 1;
  __REG32 PPS7                : 1;
  __REG32 PPS8                : 1;
  __REG32 PPS9                : 1;
  __REG32 PPS10               : 1;
  __REG32 PPS11               : 1;
  __REG32 PPS12               : 1;
  __REG32 PPS13               : 1;
  __REG32 PPS14               : 1;
  __REG32 PPS15               : 1;
  __REG32                     :16;
} __p_pps_bits;

/* Port n Pin Hardware Select Register */
typedef struct {
  __REG32 HW0                 : 2;
  __REG32 HW1                 : 2;
  __REG32 HW2                 : 2;
  __REG32 HW3                 : 2;
  __REG32 HW4                 : 2;
  __REG32 HW5                 : 2;
  __REG32 HW6                 : 2;
  __REG32 HW7                 : 2;
  __REG32 HW8                 : 2;
  __REG32 HW9                 : 2;
  __REG32 HW10                : 2;
  __REG32 HW11                : 2;
  __REG32 HW12                : 2;
  __REG32 HW13                : 2;
  __REG32 HW14                : 2;
  __REG32 HW15                : 2;
} __p_hwsel_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(ACTLR,                 0xE000E008,__READ_WRITE ,__actlr_bits);
__IO_REG32_BIT(SYST_CSR,              0xE000E010,__READ_WRITE ,__syst_csr_bits);
__IO_REG32_BIT(SYST_RVR,              0xE000E014,__READ_WRITE ,__syst_rvr_bits);
__IO_REG32_BIT(SYST_CVR,              0xE000E018,__READ_WRITE ,__syst_cvr_bits);
__IO_REG32_BIT(SYST_CALIB,            0xE000E01C,__READ       ,__syst_calib_bits);
__IO_REG32_BIT(NVIC_ISER0,            0xE000E100,__READ_WRITE ,__nvic_iser0_bits);
__IO_REG32_BIT(NVIC_ISER1,            0xE000E104,__READ_WRITE ,__nvic_iser1_bits);
__IO_REG32_BIT(NVIC_ISER2,            0xE000E108,__READ_WRITE ,__nvic_iser2_bits);
__IO_REG32_BIT(NVIC_ISER3,            0xE000E10C,__READ_WRITE ,__nvic_iser3_bits);
__IO_REG32_BIT(NVIC_ICER0,            0xE000E180,__READ_WRITE ,__nvic_icer0_bits);
__IO_REG32_BIT(NVIC_ICER1,            0xE000E184,__READ_WRITE ,__nvic_icer1_bits);
__IO_REG32_BIT(NVIC_ICER2,            0xE000E188,__READ_WRITE ,__nvic_icer2_bits);
__IO_REG32_BIT(NVIC_ICER3,            0xE000E18C,__READ_WRITE ,__nvic_icer3_bits);
__IO_REG32_BIT(NVIC_ISPR0,            0xE000E200,__READ_WRITE ,__nvic_ispr0_bits);
__IO_REG32_BIT(NVIC_ISPR1,            0xE000E204,__READ_WRITE ,__nvic_ispr1_bits);
__IO_REG32_BIT(NVIC_ISPR2,            0xE000E208,__READ_WRITE ,__nvic_ispr2_bits);
__IO_REG32_BIT(NVIC_ISPR3,            0xE000E20C,__READ_WRITE ,__nvic_ispr3_bits);
__IO_REG32_BIT(NVIC_ICPR0,            0xE000E280,__READ_WRITE ,__nvic_icpr0_bits);
__IO_REG32_BIT(NVIC_ICPR1,            0xE000E284,__READ_WRITE ,__nvic_icpr1_bits);
__IO_REG32_BIT(NVIC_ICPR2,            0xE000E288,__READ_WRITE ,__nvic_icpr2_bits);
__IO_REG32_BIT(NVIC_ICPR3,            0xE000E28C,__READ_WRITE ,__nvic_icpr3_bits);
__IO_REG32_BIT(NVIC_IABR0,            0xE000E300,__READ_WRITE ,__nvic_iabr0_bits);
__IO_REG32_BIT(NVIC_IABR1,            0xE000E304,__READ_WRITE ,__nvic_iabr1_bits);
__IO_REG32_BIT(NVIC_IABR2,            0xE000E308,__READ_WRITE ,__nvic_iabr2_bits);
__IO_REG32_BIT(NVIC_IABR3,            0xE000E30C,__READ_WRITE ,__nvic_iabr3_bits);
__IO_REG32_BIT(NVIC_IPR0,             0xE000E400,__READ_WRITE ,__nvic_ipr0_bits);
__IO_REG32_BIT(NVIC_IPR1,             0xE000E404,__READ_WRITE ,__nvic_ipr1_bits);
__IO_REG32_BIT(NVIC_IPR2,             0xE000E408,__READ_WRITE ,__nvic_ipr2_bits);
__IO_REG32_BIT(NVIC_IPR3,             0xE000E40C,__READ_WRITE ,__nvic_ipr3_bits);
__IO_REG32_BIT(NVIC_IPR4,             0xE000E410,__READ_WRITE ,__nvic_ipr4_bits);
__IO_REG32_BIT(NVIC_IPR5,             0xE000E414,__READ_WRITE ,__nvic_ipr5_bits);
__IO_REG32_BIT(NVIC_IPR6,             0xE000E418,__READ_WRITE ,__nvic_ipr6_bits);
__IO_REG32_BIT(NVIC_IPR7,             0xE000E41C,__READ_WRITE ,__nvic_ipr7_bits);
__IO_REG32_BIT(NVIC_IPR8,             0xE000E420,__READ_WRITE ,__nvic_ipr8_bits);
__IO_REG32_BIT(NVIC_IPR9,             0xE000E424,__READ_WRITE ,__nvic_ipr9_bits);
__IO_REG32_BIT(NVIC_IPR10,            0xE000E428,__READ_WRITE ,__nvic_ipr10_bits);
__IO_REG32_BIT(NVIC_IPR11,            0xE000E42C,__READ_WRITE ,__nvic_ipr11_bits);
__IO_REG32_BIT(NVIC_IPR12,            0xE000E430,__READ_WRITE ,__nvic_ipr12_bits);
__IO_REG32_BIT(NVIC_IPR13,            0xE000E434,__READ_WRITE ,__nvic_ipr13_bits);
__IO_REG32_BIT(NVIC_IPR14,            0xE000E438,__READ_WRITE ,__nvic_ipr14_bits);
__IO_REG32_BIT(NVIC_IPR15,            0xE000E43C,__READ_WRITE ,__nvic_ipr15_bits);
__IO_REG32_BIT(NVIC_IPR16,            0xE000E440,__READ_WRITE ,__nvic_ipr16_bits);
__IO_REG32_BIT(NVIC_IPR17,            0xE000E444,__READ_WRITE ,__nvic_ipr17_bits);
__IO_REG32_BIT(NVIC_IPR18,            0xE000E448,__READ_WRITE ,__nvic_ipr18_bits);
__IO_REG32_BIT(NVIC_IPR19,            0xE000E44C,__READ_WRITE ,__nvic_ipr19_bits);
__IO_REG32_BIT(NVIC_IPR20,            0xE000E450,__READ_WRITE ,__nvic_ipr20_bits);
__IO_REG32_BIT(NVIC_IPR21,            0xE000E454,__READ_WRITE ,__nvic_ipr21_bits);
__IO_REG32_BIT(NVIC_IPR22,            0xE000E458,__READ_WRITE ,__nvic_ipr22_bits);
__IO_REG32_BIT(NVIC_IPR23,            0xE000E45C,__READ_WRITE ,__nvic_ipr23_bits);
__IO_REG32_BIT(NVIC_IPR24,            0xE000E460,__READ_WRITE ,__nvic_ipr24_bits);
__IO_REG32_BIT(NVIC_IPR25,            0xE000E464,__READ_WRITE ,__nvic_ipr25_bits);
__IO_REG32_BIT(NVIC_IPR26,            0xE000E468,__READ_WRITE ,__nvic_ipr26_bits);
__IO_REG32_BIT(NVIC_IPR27,            0xE000E46C,__READ_WRITE ,__nvic_ipr27_bits);
__IO_REG32_BIT(CPUID,                 0xE000ED00,__READ       ,__cpuid_bits);
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32(    VTOR,                  0xE000ED08,__READ_WRITE );
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR1,                 0xE000ED18,__READ_WRITE ,__shpr1_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED1C,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHPR3,                 0xE000ED20,__READ_WRITE ,__shpr3_bits);
__IO_REG32_BIT(SHCSR,                 0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                  0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                  0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                  0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,                 0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                  0xE000ED38,__READ_WRITE);
__IO_REG32(    AFSR,                  0xE000ED3C,__READ_WRITE );
__IO_REG32_BIT(STIR,                  0xE000EF00,__READ_WRITE ,__stir_bits);

/***************************************************************************
 **
 ** MPU
 **
 ***************************************************************************/
__IO_REG32_BIT(MPU_TYPE,              0xE000ED90,__READ_WRITE ,__mpu_type_bits);
__IO_REG32_BIT(MPU_CTRL,              0xE000ED94,__READ_WRITE ,__mpu_ctrl_bits);
__IO_REG32_BIT(MPU_RNR,               0xE000ED98,__READ_WRITE ,__mpu_rnr_bits);
__IO_REG32_BIT(MPU_RBAR,              0xE000ED9C,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPU_RASR,              0xE000EDA0,__READ_WRITE ,__mpu_rasr_bits);
__IO_REG32_BIT(MPU_RBAR_A1,           0xE000EDA4,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPU_RASR_A1,           0xE000EDA8,__READ_WRITE ,__mpu_rasr_bits);
__IO_REG32_BIT(MPU_RBAR_A2,           0xE000EDAC,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPU_RASR_A2,           0xE000EDB0,__READ_WRITE ,__mpu_rasr_bits);
__IO_REG32_BIT(MPU_RBAR_A3,           0xE000EDB4,__READ_WRITE ,__mpu_rbar_bits);
__IO_REG32_BIT(MPU_RASR_A3,           0xE000EDB8,__READ_WRITE ,__mpu_rasr_bits);

/***************************************************************************
 **
 ** FPU
 **
 ***************************************************************************/
__IO_REG32_BIT(FPU_CPACR,             0xE000ED88,__READ_WRITE ,__fpu_cpacr_bits);
__IO_REG32_BIT(FPU_FPCCR,             0xE000EF34,__READ_WRITE ,__fpu_fpccr_bits);
__IO_REG32(    FPU_FPCAR,             0xE000EF38,__READ_WRITE );
__IO_REG32_BIT(FPU_FPDSCR,            0xE000EF3C,__READ_WRITE ,__fpu_fpdscr_bits);

/***************************************************************************
 **
 ** PBA0
 **
 ***************************************************************************/
__IO_REG32_BIT(PBA0_STS,              0x40000000,__READ_WRITE ,__pba_sts_bits);
__IO_REG32(    PBA0_WADDR,            0x40000004,__READ       );

/***************************************************************************
 **
 ** PBA1
 **
 ***************************************************************************/
__IO_REG32_BIT(PBA1_STS,              0x48000000,__READ_WRITE ,__pba_sts_bits);
__IO_REG32(    PBA1_WADDR,            0x48000004,__READ       );

/***************************************************************************
 **
 ** PBA2
 **
 ***************************************************************************/
__IO_REG32_BIT(PBA2_STS,              0x50000000,__READ_WRITE ,__pba_sts_bits);
__IO_REG32(    PBA2_WADDR,            0x50000004,__READ       );


/***************************************************************************
 **
 ** PMU
 **
 ***************************************************************************/
__IO_REG32_BIT(PMU0_ID,               0x58000508,__READ       ,__pmu_id_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASH0_ID,             0x58002008,__READ       ,__flash_id_bits);
__IO_REG32_BIT(FLASH0_FSR,            0x58002010,__READ       ,__flash_fsr_bits);
__IO_REG32_BIT(FLASH0_FCON,           0x58002014,__READ_WRITE ,__flash_fcon_bits);
__IO_REG32_BIT(FLASH0_MARP,           0x58002018,__READ_WRITE ,__flash_marp_bits);
__IO_REG32_BIT(FLASH0_PROCON0,        0x58002020,__READ       ,__flash_procon0_bits);
__IO_REG32_BIT(FLASH0_PROCON1,        0x58002024,__READ       ,__flash_procon1_bits);
__IO_REG32_BIT(FLASH0_PROCON2,        0x58002028,__READ       ,__flash_procon2_bits);

/***************************************************************************
 **
 ** PREF
 **
 ***************************************************************************/
__IO_REG32_BIT(PCON,                  0x58004000,__READ_WRITE ,__pcon_bits);

/***************************************************************************
 **
 ** GPDMA (ICU)
 **
 ***************************************************************************/
__IO_REG32_BIT(GPDMA_OVRSTAT,         0x50004900,__READ       ,__gpdma_ovrstat_bits);
__IO_REG32_BIT(GPDMA_OVRCLR,          0x50004904,__WRITE      ,__gpdma_ovrstat_bits);
__IO_REG32_BIT(GPDMA_SRSEL0,          0x50004908,__READ_WRITE ,__gpdma_srsel0_bits);
__IO_REG32_BIT(GPDMA_SRSEL1,          0x5000490C,__READ_WRITE ,__gpdma_srsel1_bits);
__IO_REG32_BIT(GPDMA_LNEN,            0x50004910,__READ_WRITE ,__gpdma_ovrstat_bits);

/***************************************************************************
 **
 ** GPDMA0
 **
 ***************************************************************************/
__IO_REG32(    GPDMA0_SAR0,           0x50014000,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR0,           0x50014008,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP0,           0x50014010,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL0,          0x50014018,__READ_WRITE ,__gpdma_ctll0_bits);
__IO_REG32_BIT(GPDMA0_CTLH0,          0x5001401C,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT0,         0x50014020,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT0,         0x50014028,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR0,       0x50014030,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR0,       0x50014038,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL0,          0x50014040,__READ_WRITE ,__gpdma_cfgl0_bits);
__IO_REG32_BIT(GPDMA0_CFGH0,          0x50014044,__READ_WRITE ,__gpdma_cfgh0_bits);
__IO_REG32_BIT(GPDMA0_SGR0,           0x50014048,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR0,           0x50014050,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR1,           0x50014058,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR1,           0x50014060,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP1,           0x50014068,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL1,          0x50014070,__READ_WRITE ,__gpdma_ctll0_bits);
__IO_REG32_BIT(GPDMA0_CTLH1,          0x50014074,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT1,         0x50014078,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT1,         0x50014080,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR1,       0x50014088,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR1,       0x50014090,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL1,          0x50014098,__READ_WRITE ,__gpdma_cfgl0_bits);
__IO_REG32_BIT(GPDMA0_CFGH1,          0x5001409C,__READ_WRITE ,__gpdma_cfgh0_bits);
__IO_REG32_BIT(GPDMA0_SGR1,           0x500140A0,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR1,           0x500140A8,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR2,           0x500140B0,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR2,           0x500140B8,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP2,           0x500140C0,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL2,          0x500140C8,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA0_CTLH2,          0x500140CC,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT2,         0x500140D0,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT2,         0x500140D8,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR2,       0x500140E0,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR2,       0x500140E8,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL2,          0x500140F0,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA0_CFGH2,          0x500140F4,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA0_SGR2,           0x500140F8,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR2,           0x50014100,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR3,           0x50014108,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR3,           0x50014110,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP3,           0x50014118,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL3,          0x50014120,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA0_CTLH3,          0x50014124,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT3,         0x50014128,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT3,         0x50014130,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR3,       0x50014138,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR3,       0x50014140,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL3,          0x50014148,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA0_CFGH3,          0x5001414C,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA0_SGR3,           0x50014150,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR3,           0x50014158,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR4,           0x50014160,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR4,           0x50014168,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP4,           0x50014170,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL4,          0x50014178,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA0_CTLH4,          0x5001417C,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT4,         0x50014180,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT4,         0x50014188,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR4,       0x50014190,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR4,       0x50014198,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL4,          0x500141A0,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA0_CFGH4,          0x500141A4,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA0_SGR4,           0x500141A8,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR4,           0x500141B0,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR5,           0x500141B8,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR5,           0x500141C0,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP5,           0x500141C8,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL5,          0x500141D0,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA0_CTLH5,          0x500141D4,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT5,         0x500141D8,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT5,         0x500141E0,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR5,       0x500141E8,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR5,       0x500141F0,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL5,          0x500141F8,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA0_CFGH5,          0x500141FC,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA0_SGR5,           0x50014200,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR5,           0x50014208,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR6,           0x50014210,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR6,           0x50014218,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP6,           0x50014220,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL6,          0x50014228,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA0_CTLH6,          0x5001422C,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT6,         0x50014230,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT6,         0x50014238,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR6,       0x50014240,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR6,       0x50014248,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL6,          0x50014250,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA0_CFGH6,          0x50014254,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA0_SGR6,           0x50014258,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR6,           0x50014260,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA0_SAR7,           0x50014268,__READ_WRITE );
__IO_REG32(    GPDMA0_DAR7,           0x50014270,__READ_WRITE );
__IO_REG32(    GPDMA0_LLP7,           0x50014278,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CTLL7,          0x50014280,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA0_CTLH7,          0x50014284,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA0_SSTAT7,         0x50014288,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTAT7,         0x50014290,__READ_WRITE );
__IO_REG32(    GPDMA0_SSTATAR7,       0x50014298,__READ_WRITE );
__IO_REG32(    GPDMA0_DSTATAR7,       0x500142A0,__READ_WRITE );
__IO_REG32_BIT(GPDMA0_CFGL7,          0x500142A8,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA0_CFGH7,          0x500142AC,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA0_SGR7,           0x500142B0,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA0_DSR7,           0x500142B8,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32_BIT(GPDMA0_RAWTFR,         0x500142C0,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_RAWBLOCK,       0x500142C8,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_RAWSRCTRAN,     0x500142D0,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_RAWDSTTRAN,     0x500142D8,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_RAWERR,         0x500142E0,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_STATUSTFR,      0x500142E8,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_STATUSBLOCK,    0x500142F0,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_STATUSSRCTRAN,  0x500142F8,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_STATUSDSTTRAN,  0x50014300,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_STATUSERR,      0x50014308,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_MASKTFR,        0x50014310,__READ_WRITE ,__gpdma0_masktfr_bits);
__IO_REG32_BIT(GPDMA0_MASKBLOCK,      0x50014318,__READ_WRITE ,__gpdma0_masktfr_bits);
__IO_REG32_BIT(GPDMA0_MASKSRCTRAN,    0x50014320,__READ_WRITE ,__gpdma0_masktfr_bits);
__IO_REG32_BIT(GPDMA0_MASKDSTTRAN,    0x50014328,__READ_WRITE ,__gpdma0_masktfr_bits);
__IO_REG32_BIT(GPDMA0_MASKERR,        0x50014330,__READ_WRITE ,__gpdma0_masktfr_bits);
__IO_REG32_BIT(GPDMA0_CLEARTFR,       0x50014338,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_CLEARBLOCK,     0x50014340,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_CLEARSRCTRAN,   0x50014348,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_CLEARDSTTRAN,   0x50014350,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_CLEARERR,       0x50014358,__READ_WRITE ,__gpdma0_rawtfr_bits);
__IO_REG32_BIT(GPDMA0_STATUSINT,      0x50014360,__READ_WRITE ,__gpdma_statusint_bits);
__IO_REG32_BIT(GPDMA0_REQSRCREG,      0x50014368,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32_BIT(GPDMA0_REQDSTREG,      0x50014370,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32_BIT(GPDMA0_SGLREQSRCREG,   0x50014378,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32_BIT(GPDMA0_SGLREQDSTREG,   0x50014380,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32_BIT(GPDMA0_LSTSRCREG,      0x50014388,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32_BIT(GPDMA0_LSTDSTREG,      0x50014390,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32_BIT(GPDMA0_DMACFGREG,      0x50014398,__READ_WRITE ,__gpdma_dmacfgreg_bits);
__IO_REG32_BIT(GPDMA0_CHENREG,        0x500143A0,__READ_WRITE ,__gpdma0_chenreg_bits);
__IO_REG32(    GPDMA0_ID,             0x500143A8,__READ_WRITE );
__IO_REG32(    GPDMA0_TYPE,           0x500143F8,__READ_WRITE );
__IO_REG32(    GPDMA0_VERSION,        0x500143FC,__READ_WRITE );

/***************************************************************************
 **
 ** GPDMA1
 **
 ***************************************************************************/
__IO_REG32(    GPDMA1_SAR0,           0x50018000,__READ_WRITE );
__IO_REG32(    GPDMA1_DAR0,           0x50018008,__READ_WRITE );
__IO_REG32(    GPDMA1_LLP0,           0x50018010,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CTLL0,          0x50018018,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA1_CTLH0,          0x5001801C,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA1_SSTAT0,         0x50018020,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTAT0,         0x50018028,__READ_WRITE );
__IO_REG32(    GPDMA1_SSTATAR0,       0x50018030,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTATAR0,       0x50018038,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CFGL0,          0x50018040,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA1_CFGH0,          0x50018044,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA1_SGR0,           0x50018048,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA1_DSR0,           0x50018050,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA1_SAR1,           0x50018058,__READ_WRITE );
__IO_REG32(    GPDMA1_DAR1,           0x50018060,__READ_WRITE );
__IO_REG32(    GPDMA1_LLP1,           0x50018068,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CTLL1,          0x50018070,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA1_CTLH1,          0x50018074,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA1_SSTAT1,         0x50018078,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTAT1,         0x50018080,__READ_WRITE );
__IO_REG32(    GPDMA1_SSTATAR1,       0x50018088,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTATAR1,       0x50018090,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CFGL1,          0x50018098,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA1_CFGH1,          0x5001809C,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA1_SGR1,           0x500180A0,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA1_DSR1,           0x500180A8,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA1_SAR2,           0x500180B0,__READ_WRITE );
__IO_REG32(    GPDMA1_DAR2,           0x500180B8,__READ_WRITE );
__IO_REG32(    GPDMA1_LLP2,           0x500180C0,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CTLL2,          0x500180C8,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA1_CTLH2,          0x500180CC,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA1_SSTAT2,         0x500180D0,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTAT2,         0x500180D8,__READ_WRITE );
__IO_REG32(    GPDMA1_SSTATAR2,       0x500180E0,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTATAR2,       0x500180E8,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CFGL2,          0x500180F0,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA1_CFGH2,          0x500180F4,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA1_SGR2,           0x500180F8,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA1_DSR2,           0x50018100,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32(    GPDMA1_SAR3,           0x50018108,__READ_WRITE );
__IO_REG32(    GPDMA1_DAR3,           0x50018110,__READ_WRITE );
__IO_REG32(    GPDMA1_LLP3,           0x50018118,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CTLL3,          0x50018120,__READ_WRITE ,__gpdma_ctll2_bits);
__IO_REG32_BIT(GPDMA1_CTLH3,          0x50018124,__READ_WRITE ,__gpdma_ctlh_bits);
__IO_REG32(    GPDMA1_SSTAT3,         0x50018128,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTAT3,         0x50018130,__READ_WRITE );
__IO_REG32(    GPDMA1_SSTATAR3,       0x50018138,__READ_WRITE );
__IO_REG32(    GPDMA1_DSTATAR3,       0x50018140,__READ_WRITE );
__IO_REG32_BIT(GPDMA1_CFGL3,          0x50018148,__READ_WRITE ,__gpdma_cfgl2_bits);
__IO_REG32_BIT(GPDMA1_CFGH3,          0x5001814C,__READ_WRITE ,__gpdma_cfgh2_bits);
__IO_REG32_BIT(GPDMA1_SGR3,           0x50018150,__READ_WRITE ,__gpdma_sgr_bits);
__IO_REG32_BIT(GPDMA1_DSR3,           0x50018158,__READ_WRITE ,__gpdma_dsr_bits);
__IO_REG32_BIT(GPDMA1_RAWTFR,         0x500182C0,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_RAWBLOCK,       0x500182C8,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_RAWSRCTRAN,     0x500182D0,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_RAWDSTTRAN,     0x500182D8,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_RAWERR,         0x500182E0,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_STATUSTFR,      0x500182E8,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_STATUSBLOCK,    0x500182F0,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_STATUSSRCTRAN,  0x500182F8,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_STATUSDSTTRAN,  0x50018300,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_STATUSERR,      0x50018308,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_MASKTFR,        0x50018310,__READ_WRITE ,__gpdma1_masktfr_bits);
__IO_REG32_BIT(GPDMA1_MASKBLOCK,      0x50018318,__READ_WRITE ,__gpdma1_masktfr_bits);
__IO_REG32_BIT(GPDMA1_MASKSRCTRAN,    0x50018320,__READ_WRITE ,__gpdma1_masktfr_bits);
__IO_REG32_BIT(GPDMA1_MASKDSTTRAN,    0x50018328,__READ_WRITE ,__gpdma1_masktfr_bits);
__IO_REG32_BIT(GPDMA1_MASKERR,        0x50018330,__READ_WRITE ,__gpdma1_masktfr_bits);
__IO_REG32_BIT(GPDMA1_CLEARTFR,       0x50018338,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_CLEARBLOCK,     0x50018340,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_CLEARSRCTRAN,   0x50018348,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_CLEARDSTTRAN,   0x50018350,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_CLEARERR,       0x50018358,__READ_WRITE ,__gpdma1_rawtfr_bits);
__IO_REG32_BIT(GPDMA1_STATUSINT,      0x50018360,__READ_WRITE ,__gpdma_statusint_bits);
__IO_REG32_BIT(GPDMA1_REQSRCREG,      0x50018368,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32_BIT(GPDMA1_REQDSTREG,      0x50018370,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32_BIT(GPDMA1_SGLREQSRCREG,   0x50018378,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32_BIT(GPDMA1_SGLREQDSTREG,   0x50018380,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32_BIT(GPDMA1_LSTSRCREG,      0x50018388,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32_BIT(GPDMA1_LSTDSTREG,      0x50018390,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32_BIT(GPDMA1_DMACFGREG,      0x50018398,__READ_WRITE ,__gpdma_dmacfgreg_bits);
__IO_REG32_BIT(GPDMA1_CHENREG,        0x500183A0,__READ_WRITE ,__gpdma1_chenreg_bits);
__IO_REG32(    GPDMA1_ID,             0x500183A8,__READ_WRITE );
__IO_REG32(    GPDMA1_TYPE,           0x500183F8,__READ_WRITE );
__IO_REG32(    GPDMA1_VERSION,        0x500183FC,__READ_WRITE );

/***************************************************************************
 **
 ** FCE
 **
 ***************************************************************************/
__IO_REG32_BIT(FCE_CLC,               0x50020000,__READ_WRITE ,__fce_clc_bits);
__IO_REG32_BIT(FCE_ID,                0x50020008,__READ       ,__fce_id_bits);
__IO_REG32(    FCE_IR0,               0x50020020,__READ_WRITE );
__IO_REG32(    FCE_RES0,              0x50020024,__READ       );
__IO_REG32_BIT(FCE_CFG0,              0x50020028,__READ_WRITE ,__fce_cfg_bits);
__IO_REG32_BIT(FCE_STS0,              0x5002002C,__READ_WRITE ,__fce_sts_bits);
__IO_REG32_BIT(FCE_LENGTH0,           0x50020030,__READ_WRITE ,__fce_length_bits);
__IO_REG32(    FCE_CHECK0,            0x50020034,__READ_WRITE );
__IO_REG32(    FCE_CRC0,              0x50020038,__READ_WRITE );
__IO_REG32_BIT(FCE_CTR0,              0x5002003C,__READ_WRITE ,__fce_ctr_bits);
__IO_REG32(    FCE_IR1,               0x50020040,__READ_WRITE );
__IO_REG32(    FCE_RES1,              0x50020044,__READ       );
__IO_REG32_BIT(FCE_CFG1,              0x50020048,__READ_WRITE ,__fce_cfg_bits);
__IO_REG32_BIT(FCE_STS1,              0x5002004C,__READ_WRITE ,__fce_sts_bits);
__IO_REG32_BIT(FCE_LENGTH1,           0x50020050,__READ_WRITE ,__fce_length_bits);
__IO_REG32(    FCE_CHECK1,            0x50020054,__READ_WRITE );
__IO_REG32(    FCE_CRC1,              0x50020058,__READ_WRITE );
__IO_REG32_BIT(FCE_CTR1,              0x5002005C,__READ_WRITE ,__fce_ctr_bits);
__IO_REG32_BIT(FCE_IR2,               0x50020060,__READ_WRITE ,__fce_ir2_bits);
__IO_REG32_BIT(FCE_RES2,              0x50020064,__READ       ,__fce_res2_bits);
__IO_REG32_BIT(FCE_CFG2,              0x50020068,__READ_WRITE ,__fce_cfg_bits);
__IO_REG32_BIT(FCE_STS2,              0x5002006C,__READ_WRITE ,__fce_sts_bits);
__IO_REG32_BIT(FCE_LENGTH2,           0x50020070,__READ_WRITE ,__fce_length_bits);
__IO_REG32_BIT(FCE_CHECK2,            0x50020074,__READ_WRITE ,__fce_check2_bits);
__IO_REG32_BIT(FCE_CRC2,              0x50020078,__READ_WRITE ,__fce_crc2_bits);
__IO_REG32_BIT(FCE_CTR2,              0x5002007C,__READ_WRITE ,__fce_ctr_bits);
__IO_REG32_BIT(FCE_IR3,               0x50020080,__READ_WRITE ,__fce_ir3_bits);
__IO_REG32_BIT(FCE_RES3,              0x50020084,__READ       ,__fce_res3_bits);
__IO_REG32_BIT(FCE_CFG3,              0x50020088,__READ_WRITE ,__fce_cfg_bits);
__IO_REG32_BIT(FCE_STS3,              0x5002008C,__READ_WRITE ,__fce_sts_bits);
__IO_REG32_BIT(FCE_LENGTH3,           0x50020090,__READ_WRITE ,__fce_length_bits);
__IO_REG32_BIT(FCE_CHECK3,            0x50020094,__READ_WRITE ,__fce_check3_bits);
__IO_REG32_BIT(FCE_CRC3,              0x50020098,__READ_WRITE ,__fce_crc3_bits);
__IO_REG32_BIT(FCE_CTR3,              0x5002009C,__READ_WRITE ,__fce_ctr_bits);
__IO_REG32_BIT(FCE_SRC,               0x500200FC,__READ       ,__fce_src_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32(    WDT_ID,                0x50008000,__READ       );
__IO_REG32_BIT(WDT_CTR,               0x50008004,__READ_WRITE ,__wdt_ctr_bits);
__IO_REG32(    WDT_SRV,               0x50008008,__WRITE      );
__IO_REG32(    WDT_TIM,               0x5000800C,__READ       );
__IO_REG32(    WDT_WLB,               0x50008010,__READ_WRITE );
__IO_REG32(    WDT_WUB,               0x50008014,__READ_WRITE );
__IO_REG32_BIT(WDT_STS,               0x50008018,__READ       ,__wdt_sts_bits);
__IO_REG32_BIT(WDT_CLR,               0x5000801C,__WRITE      ,__wdt_clr_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RTC_ID,                0x50004A00,__READ       );
__IO_REG32_BIT(RTC_CTR,               0x50004A04,__READ_WRITE ,__rtc_ctr_bits);
__IO_REG32_BIT(RTC_RAWSTAT,           0x50004A08,__READ       ,__rtc_rawstat_bits);
__IO_REG32_BIT(RTC_STSSR,             0x50004A0C,__READ       ,__rtc_stssr_bits);
__IO_REG32_BIT(RTC_MSKSR,             0x50004A10,__READ_WRITE ,__rtc_msksr_bits);
__IO_REG32_BIT(RTC_CLRSR,             0x50004A14,__WRITE      ,__rtc_rawstat_bits);
__IO_REG32_BIT(RTC_ATIM0,             0x50004A18,__READ_WRITE ,__rtc_atim0_bits);
__IO_REG32_BIT(RTC_ATIM1,             0x50004A1C,__READ_WRITE ,__rtc_atim1_bits);
__IO_REG32_BIT(RTC_TIM0,              0x50004A20,__READ_WRITE ,__rtc_tim0_bits);
__IO_REG32_BIT(RTC_TIM1,              0x50004A24,__READ_WRITE ,__rtc_tim1_bits);

/***************************************************************************
 **
 ** ERU0 (ICU)
 **
 ***************************************************************************/
__IO_REG32_BIT(ERU0_EXISEL,           0x50004800,__READ_WRITE ,__eru_exisel_bits);
__IO_REG32_BIT(ERU0_EXICON0,          0x50004810,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU0_EXICON1,          0x50004814,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU0_EXICON2,          0x50004818,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU0_EXICON3,          0x5000481C,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU0_EXOCON0,          0x50004820,__READ_WRITE ,__eru_exocon_bits);
__IO_REG32_BIT(ERU0_EXOCON1,          0x50004824,__READ_WRITE ,__eru_exocon_bits);
__IO_REG32_BIT(ERU0_EXOCON2,          0x50004828,__READ_WRITE ,__eru_exocon_bits);
__IO_REG32_BIT(ERU0_EXOCON3,          0x5000482C,__READ_WRITE ,__eru_exocon_bits);

/***************************************************************************
 **
 ** ERU1 (ICU)
 **
 ***************************************************************************/
__IO_REG32_BIT(ERU1_EXISEL,           0x40044000,__READ_WRITE ,__eru_exisel_bits);
__IO_REG32_BIT(ERU1_EXICON0,          0x40044010,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU1_EXICON1,          0x40044014,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU1_EXICON2,          0x40044018,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU1_EXICON3,          0x4004401C,__READ_WRITE ,__eru_exicon_bits);
__IO_REG32_BIT(ERU1_EXOCON0,          0x40044020,__READ_WRITE ,__eru_exocon_bits);
__IO_REG32_BIT(ERU1_EXOCON1,          0x40044024,__READ_WRITE ,__eru_exocon_bits);
__IO_REG32_BIT(ERU1_EXOCON2,          0x40044028,__READ_WRITE ,__eru_exocon_bits);
__IO_REG32_BIT(ERU1_EXOCON3,          0x4004402C,__READ_WRITE ,__eru_exocon_bits);

/***************************************************************************
 **
 ** SCU
 **
 ***************************************************************************/
__IO_REG32(    SCU_ID,                0x50004000,__READ       );
__IO_REG32(    SCU_IDCHIP,            0x50004004,__READ_WRITE );
__IO_REG32(    SCU_IDMANUF,           0x50004008,__READ_WRITE );
__IO_REG32_BIT(SCU_STCON,             0x50004010,__READ_WRITE ,__scu_stcon_bits);
__IO_REG32(    SCU_SSW0,              0x50004014,__READ_WRITE );
__IO_REG32(    SCU_SSW1,              0x50004018,__READ_WRITE );
__IO_REG32(    SCU_SSW2,              0x5000401C,__READ_WRITE );
__IO_REG32(    SCU_SSW3,              0x50004020,__READ_WRITE );
__IO_REG32(    SCU_SSW4,              0x50004024,__READ_WRITE );
__IO_REG32(    SCU_SSW5,              0x50004028,__READ_WRITE );
__IO_REG32(    SCU_SSW6,              0x5000402C,__READ_WRITE );
__IO_REG32(    SCU_SSW7,              0x50004030,__READ_WRITE );
__IO_REG32_BIT(SCU_ETH0_CON,          0x50004040,__READ_WRITE ,__scu_eth0_con_bits);
__IO_REG32_BIT(SCU_TCUCON,            0x50004044,__READ       ,__scu_tcucon_bits);
__IO_REG32(    SCU_TSW0,              0x50004048,__READ_WRITE );
__IO_REG32_BIT(SCU_CCUCON,            0x5000404C,__READ_WRITE ,__scu_ccucon_bits);
__IO_REG32_BIT(SCU_DBGCON0,           0x50004050,__READ_WRITE ,__scu_dbgcon0_bits);
__IO_REG32_BIT(SCU_DBGCON1,           0x50004054,__READ       ,__scu_dbgcon1_bits);
__IO_REG32_BIT(SCU_SRSTAT,            0x50004074,__READ       ,__scu_srstat_bits);
__IO_REG32_BIT(SCU_RAWSR,             0x50004078,__READ       ,__scu_srstat_bits);
__IO_REG32_BIT(SCU_SRMSK,             0x5000407C,__READ_WRITE ,__scu_srstat_bits);
__IO_REG32_BIT(SCU_SRCLR,             0x50004080,__WRITE      ,__scu_srstat_bits);
__IO_REG32_BIT(SCU_SRSET,             0x50004084,__WRITE      ,__scu_srstat_bits);
__IO_REG32_BIT(SCU_NMIREQEN,          0x50004088,__READ_WRITE ,__scu_nmireqen_bits);
__IO_REG32_BIT(SCU_DTSCON,            0x5000408C,__READ_WRITE ,__scu_dtscon_bits);
__IO_REG32_BIT(SCU_DTSSTAT,           0x50004090,__READ       ,__scu_dtsstat_bits);
__IO_REG32_BIT(SCU_STPCON,            0x50004094,__READ_WRITE ,__scu_stpcon_bits);
__IO_REG32_BIT(SCU_USBTRIM,           0x50004098,__READ_WRITE ,__scu_usbtrim_bits);
__IO_REG32_BIT(SCU_SDMMCDEL,          0x5000409C,__READ_WRITE ,__scu_sdmmcdel_bits);
__IO_REG32_BIT(SCU_G0ORCEN,           0x500040A0,__READ_WRITE ,__scu_gorcen_bits);
__IO_REG32_BIT(SCU_G1ORCEN,           0x500040A4,__READ_WRITE ,__scu_gorcen_bits);
__IO_REG32_BIT(SCU_MIRRSTS,           0x500040C4,__READ       ,__scu_mirrsts_bits);
__IO_REG32_BIT(SCU_RMACR,             0x500040C8,__READ_WRITE ,__scu_rmacr_bits);
__IO_REG32(    SCU_RMADATA,           0x500040CC,__READ_WRITE );
__IO_REG32(    SCU_IDRT,              0x50004100,__READ_WRITE );
__IO_REG32(    SCU_HDFRDATA,          0x50004104,__READ_WRITE );
__IO_REG32_BIT(SCU_HDFRSTAT,          0x50004108,__READ       ,__scu_hdfrstat_bits);
__IO_REG32_BIT(SCU_HDFRCMD,           0x5000410C,__READ       ,__scu_hdfrcmd_bits);
__IO_REG32_BIT(SCU_PEEN,              0x5000413C,__READ_WRITE ,__scu_peen_bits);
__IO_REG32_BIT(SCU_MCHKCON,           0x50004140,__READ_WRITE ,__scu_mchkcon_bits);
__IO_REG32_BIT(SCU_PETE,              0x50004144,__READ_WRITE ,__scu_pete_bits);
__IO_REG32_BIT(SCU_PERSTEN,           0x50004148,__READ_WRITE ,__scu_persten_bits);
__IO_REG32_BIT(SCU_PEFLAG,            0x50004150,__READ_WRITE ,__scu_peflag_bits);
__IO_REG32_BIT(SCU_PMTPR,             0x50004154,__READ_WRITE ,__scu_pmtpr_bits);
__IO_REG32_BIT(SCU_PMTSR,             0x50004158,__READ_WRITE ,__scu_pmtsr_bits);
__IO_REG32_BIT(SCU_TRAPSTAT,          0x50004160,__READ       ,__scu_trapstat_bits);
__IO_REG32_BIT(SCU_TRAPRAW,           0x50004164,__READ       ,__scu_trapstat_bits);
__IO_REG32_BIT(SCU_TRAPDIS,           0x50004168,__READ_WRITE ,__scu_trapstat_bits);
__IO_REG32_BIT(SCU_PTRAPCLR,          0x5000416C,__WRITE      ,__scu_trapstat_bits);
__IO_REG32_BIT(SCU_TRAPSET,           0x50004170,__WRITE      ,__scu_trapstat_bits);

/***************************************************************************
 **
 ** PCU
 **
 ***************************************************************************/
__IO_REG32_BIT(PCU_PWRSTAT,           0x50004200,__READ       ,__pcu_pwrstat_bits);
__IO_REG32_BIT(PCU_PWRSET,            0x50004204,__WRITE      ,__pcu_pwrset_bits);
__IO_REG32_BIT(PCU_PWRCLR,            0x50004208,__WRITE      ,__pcu_pwrset_bits);
__IO_REG32_BIT(PCU_EVRSTAT,           0x50004210,__READ       ,__pcu_evrstat_bits);
__IO_REG32_BIT(PCU_EVRVADCSTAT,       0x50004214,__READ       ,__pcu_evrvadcstat_bits);
__IO_REG32_BIT(PCU_EVRTRIM,           0x50004218,__READ_WRITE ,__pcu_evrtrim_bits);
__IO_REG32_BIT(PCU_EVRRSTCON,         0x5000421C,__READ_WRITE ,__pcu_evrrstcon_bits);
__IO_REG32_BIT(PCU_EVR13CON,          0x50004220,__READ_WRITE ,__pcu_evr13con_bits);
__IO_REG32_BIT(PCU_EVROSC,            0x50004228,__READ_WRITE ,__pcu_evrosc_bits);
__IO_REG32_BIT(PCU_PWRMON,            0x5000422C,__READ_WRITE ,__pcu_pwrmon_bits);

/***************************************************************************
 **
 ** HCU
 **
 ***************************************************************************/
__IO_REG32_BIT(HCU_HDSTAT,            0x50004300,__READ       ,__hcu_hdstat_bits);
__IO_REG32_BIT(HCU_HDCLR,             0x50004304,__WRITE      ,__hcu_hdstat_bits);
__IO_REG32_BIT(HCU_HDSET,             0x50004308,__WRITE      ,__hcu_hdstat_bits);
__IO_REG32_BIT(HCU_HDCR,              0x5000430C,__READ_WRITE ,__hcu_hdcr_bits);
__IO_REG32_BIT(HCU_OSCSITRIM,         0x50004310,__READ_WRITE ,__hcu_oscsitrim_bits);
__IO_REG32_BIT(HCU_OSCSICTRL,         0x50004314,__READ_WRITE ,__hcu_oscsictrl_bits);
__IO_REG32_BIT(HCU_OSCULSTAT,         0x50004318,__READ       ,__hcu_osculstat_bits);
__IO_REG32_BIT(HCU_OSCULCTRL,         0x5000431C,__READ_WRITE ,__hcu_osculctrl_bits);
__IO_REG32_BIT(HCU_LPACCR,            0x50004320,__READ_WRITE ,__hcu_lpaccr_bits);
__IO_REG32_BIT(HCU_LPACTH0,           0x50004324,__READ_WRITE ,__hcu_lpacth_bits);
#define HCU_LPACTH1     HCU_LPACTH0
#define HCU_LPACTH1_bit HCU_LPACTH0_bit

/***************************************************************************
 **
 ** RCU
 **
 ***************************************************************************/
__IO_REG32_BIT(RCU_RSTSTAT,           0x50004400,__READ       ,__rcu_rststat_bits);
__IO_REG32_BIT(RCU_RSTSET,            0x50004404,__WRITE      ,__rcu_rstset_bits);
__IO_REG32_BIT(RCU_RSTCLR,            0x50004408,__WRITE      ,__rcu_rstset_bits);
__IO_REG32_BIT(RCU_PRSTAT0,           0x5000440C,__READ       ,__rcu_prstat_bits);
__IO_REG32_BIT(RCU_PRSET0,            0x50004410,__WRITE      ,__rcu_prstat_bits);
__IO_REG32_BIT(RCU_PRCLR0,            0x50004414,__WRITE      ,__rcu_prstat_bits);
__IO_REG32_BIT(RCU_PRSTAT1,           0x50004418,__READ       ,__rcu_prstat1_bits);
__IO_REG32_BIT(RCU_PRSET1,            0x5000441C,__WRITE      ,__rcu_prstat1_bits);
__IO_REG32_BIT(RCU_PRCLR1,            0x50004420,__WRITE      ,__rcu_prstat1_bits);
__IO_REG32_BIT(RCU_PRSTAT2,           0x50004424,__READ       ,__rcu_prstat2_bits);
__IO_REG32_BIT(RCU_PRSET2,            0x50004428,__WRITE      ,__rcu_prstat2_bits);
__IO_REG32_BIT(RCU_PRCLR2,            0x5000442C,__WRITE      ,__rcu_prstat2_bits);
__IO_REG32_BIT(RCU_PRSTAT3,           0x50004430,__READ       ,__rcu_prstat3_bits);
__IO_REG32_BIT(RCU_PRSET3,            0x50004434,__WRITE      ,__rcu_prstat3_bits);
__IO_REG32_BIT(RCU_PRCLR3,            0x50004438,__WRITE      ,__rcu_prstat3_bits);

/***************************************************************************
 **
 ** CCU
 **
 ***************************************************************************/
__IO_REG32_BIT(CCU_CLKSTAT,           0x50004600,__READ       ,__ccu_clkstat_bits);
__IO_REG32_BIT(CCU_CLKSET,            0x50004604,__WRITE      ,__ccu_clkset_bits);
__IO_REG32_BIT(CCU_CLKCLR,            0x50004608,__READ_WRITE ,__ccu_clkclr_bits);
__IO_REG32_BIT(CCU_SYSCLKCR,          0x5000460C,__READ_WRITE ,__ccu_sysclkcr_bits);
__IO_REG32_BIT(CCU_CPUCLKCR,          0x50004610,__READ_WRITE ,__ccu_cpuclkcr_bits);
__IO_REG32_BIT(CCU_PBCLKCR,           0x50004614,__READ_WRITE ,__ccu_pbclkcr_bits);
__IO_REG32_BIT(CCU_USBCLKCR,          0x50004618,__READ_WRITE ,__ccu_usbclkcr_bits);
__IO_REG32_BIT(CCU_EBUCLKCR,          0x5000461C,__READ_WRITE ,__ccu_ebuclkcr_bits);
__IO_REG32_BIT(CCU_CCUCLKCR,          0x50004620,__READ_WRITE ,__ccu_ccuclkcr_bits);
__IO_REG32_BIT(CCU_WDTCLKCR,          0x50004624,__READ_WRITE ,__ccu_wdtclkcr_bits);
__IO_REG32_BIT(CCU_EXTCLKCR,          0x50004628,__READ_WRITE ,__ccu_extclkcr_bits);
__IO_REG32_BIT(CCU_SLEEPCR,           0x50004630,__READ_WRITE ,__ccu_sleepcr_bits);
__IO_REG32_BIT(CCU_DSLEEPCR,          0x50004634,__READ_WRITE ,__ccu_dsleepcr_bits);
__IO_REG32_BIT(CCU_OSCHPSTAT,         0x50004700,__READ       ,__ccu_oschpstat_bits);
__IO_REG32_BIT(CCU_OSCHPCTRL,         0x50004704,__READ_WRITE ,__ccu_oschpctrl_bits);
__IO_REG32_BIT(CCU_OSCFITRIM,         0x5000470C,__READ_WRITE ,__ccu_oscfitrim_bits);
__IO_REG32_BIT(CCU_PLLSTAT,           0x50004710,__READ       ,__ccu_pllstat_bits);
__IO_REG32_BIT(CCU_PLLCON0,           0x50004714,__READ_WRITE ,__ccu_pllcon0_bits);
__IO_REG32_BIT(CCU_PLLCON1,           0x50004718,__READ_WRITE ,__ccu_pllcon1_bits);
__IO_REG32_BIT(CCU_PLLCON2,           0x5000471C,__READ_WRITE ,__ccu_pllcon2_bits);
__IO_REG32_BIT(CCU_USBPLLSTAT,        0x50004720,__READ       ,__ccu_usbpllstat_bits);
__IO_REG32_BIT(CCU_USBPLLCON,         0x50004724,__READ_WRITE ,__ccu_usbpllcon_bits);
__IO_REG32_BIT(CCU_CLKMXSTAT,         0x50004738,__READ       ,__ccu_clkmxstat_bits);

/***************************************************************************
 **
 ** EBU
 **
 ***************************************************************************/
__IO_REG32_BIT(EBU_CLC,               0x58008000,__READ_WRITE ,__ebu_clc_bits);
__IO_REG32_BIT(EBU_MODCON,            0x58008004,__READ_WRITE ,__ebu_modcon_bits);
__IO_REG32_BIT(EBU_ID,                0x58008008,__READ       ,__ebu_id_bits);
__IO_REG32_BIT(EBU_USERCON,           0x5800800C,__READ_WRITE ,__ebu_usercon_bits);
__IO_REG32_BIT(EBU_ADDRSEL0,          0x58008018,__READ_WRITE ,__ebu_addrsel_bits);
__IO_REG32_BIT(EBU_ADDRSEL1,          0x5800801C,__READ_WRITE ,__ebu_addrsel_bits);
__IO_REG32_BIT(EBU_ADDRSEL2,          0x58008020,__READ_WRITE ,__ebu_addrsel_bits);
__IO_REG32_BIT(EBU_ADDRSEL3,          0x58008024,__READ_WRITE ,__ebu_addrsel_bits);
__IO_REG32_BIT(EBU_BUSRCON0,          0x58008028,__READ_WRITE ,__ebu_busrcon_bits);
__IO_REG32_BIT(EBU_BUSRAP0,           0x5800802C,__READ_WRITE ,__ebu_busrap_bits);
__IO_REG32_BIT(EBU_BUSWCON0,          0x58008030,__READ_WRITE ,__ebu_buswcon_bits);
__IO_REG32_BIT(EBU_BUSWAP0,           0x58008034,__READ_WRITE ,__ebu_buswap_bits);
__IO_REG32_BIT(EBU_BUSRCON1,          0x58008038,__READ_WRITE ,__ebu_busrcon_bits);
__IO_REG32_BIT(EBU_BUSRAP1,           0x5800803C,__READ_WRITE ,__ebu_busrap_bits);
__IO_REG32_BIT(EBU_BUSWCON1,          0x58008040,__READ_WRITE ,__ebu_buswcon_bits);
__IO_REG32_BIT(EBU_BUSWAP1,           0x58008044,__READ_WRITE ,__ebu_buswap_bits);
__IO_REG32_BIT(EBU_BUSRCON2,          0x58008048,__READ_WRITE ,__ebu_busrcon_bits);
__IO_REG32_BIT(EBU_BUSRAP2,           0x5800804C,__READ_WRITE ,__ebu_busrap_bits);
__IO_REG32_BIT(EBU_BUSWCON2,          0x58008050,__READ_WRITE ,__ebu_buswcon_bits);
__IO_REG32_BIT(EBU_BUSWAP2,           0x58008054,__READ_WRITE ,__ebu_buswap_bits);
__IO_REG32_BIT(EBU_BUSRCON3,          0x58008058,__READ_WRITE ,__ebu_busrcon_bits);
__IO_REG32_BIT(EBU_BUSRAP3,           0x5800805C,__READ_WRITE ,__ebu_busrap_bits);
__IO_REG32_BIT(EBU_BUSWCON3,          0x58008060,__READ_WRITE ,__ebu_buswcon_bits);
__IO_REG32_BIT(EBU_BUSWAP3,           0x58008064,__READ_WRITE ,__ebu_buswap_bits);
__IO_REG32_BIT(EBU_SDRMCON,           0x58008068,__READ_WRITE ,__ebu_sdrmcon_bits);
__IO_REG32_BIT(EBU_SDRMOD,            0x5800806C,__READ_WRITE ,__ebu_sdrmod_bits);
__IO_REG32_BIT(EBU_SDRMREF,           0x58008070,__READ_WRITE ,__ebu_sdrmref_bits);
__IO_REG32_BIT(EBU_SDRSTAT,           0x58008074,__READ       ,__ebu_sdrstat_bits);

/***************************************************************************
 **
 ** LEDTSCU
 **
 ***************************************************************************/
__IO_REG32_BIT(LEDTSCU_ID,            0x48010000,__READ       ,__ledtscu_id_bits);
__IO_REG32_BIT(LEDTSCU_GLOBCTL,       0x48010004,__READ_WRITE ,__ledtscu_globctl_bits);
__IO_REG32_BIT(LEDTSCU_FNCTL,         0x48010008,__READ_WRITE ,__ledtscu_fnctl_bits);
__IO_REG32_BIT(LEDTSCU_EVFR,          0x4801000C,__READ_WRITE ,__ledtscu_evfr_bits);
__IO_REG32_BIT(LEDTSCU_TSVAL,         0x48010010,__READ_WRITE ,__ledtscu_tsval_bits);
__IO_REG32_BIT(LEDTSCU_LINE0,         0x48010014,__READ_WRITE ,__ledtscu_line0_bits);
__IO_REG32_BIT(LEDTSCU_LINE1,         0x48010018,__READ_WRITE ,__ledtscu_line1_bits);
__IO_REG32_BIT(LEDTSCU_LDCMP0,        0x4801001C,__READ_WRITE ,__ledtscu_ldcmp0_bits);
__IO_REG32_BIT(LEDTSCU_LDCMP1,        0x48010020,__READ_WRITE ,__ledtscu_ldcmp1_bits);
__IO_REG32_BIT(LEDTSCU_TSCMP0,        0x48010024,__READ_WRITE ,__ledtscu_tscmp0_bits);
__IO_REG32_BIT(LEDTSCU_TSCMP1,        0x48010028,__READ_WRITE ,__ledtscu_tscmp1_bits);
__IO_REG32_BIT(LEDTSCU_INS,           0x4801002C,__READ_WRITE ,__ledtscu_ins_bits);

/***************************************************************************
 **
 ** SDMMC
 **
 ***************************************************************************/
__IO_REG16_BIT(SDMMC_BLOCK_SIZE,        0x4801C004,__READ_WRITE ,__sdmmc_block_size_bits);
__IO_REG16(    SDMMC_BLOCK_COUNT,       0x4801C006,__READ_WRITE );
__IO_REG32(    SDMMC_ARGUMENT1,         0x4801C008,__READ_WRITE );
__IO_REG16_BIT(SDMMC_TRANSFER_MODE,     0x4801C00C,__READ_WRITE ,__sdmmc_transfer_mode_bits);
__IO_REG16_BIT(SDMMC_COMMAND,           0x4801C00E,__READ_WRITE ,__sdmmc_command_bits);
__IO_REG32_BIT(SDMMC_RESPONSE0,         0x4801C010,__READ       ,__sdmmc_response0_bits);
__IO_REG32_BIT(SDMMC_RESPONSE2,         0x4801C014,__READ       ,__sdmmc_response2_bits);
__IO_REG32_BIT(SDMMC_RESPONSE4,         0x4801C018,__READ       ,__sdmmc_response4_bits);
__IO_REG32_BIT(SDMMC_RESPONSE6,         0x4801C01C,__READ       ,__sdmmc_response6_bits);
__IO_REG32(    SDMMC_DATA_BUFFER,       0x4801C020,__READ_WRITE );
__IO_REG32_BIT(SDMMC_PRESENT_STATE,     0x4801C024,__READ       ,__sdmmc_present_state_bits);
__IO_REG8_BIT( SDMMC_HOST_CTRL,         0x4801C028,__READ_WRITE ,__sdmmc_host_ctrl_bits);
__IO_REG8_BIT( SDMMC_POWER_CTRL,        0x4801C029,__READ_WRITE ,__sdmmc_power_ctrl_bits);
__IO_REG8_BIT( SDMMC_BLOCK_GAP_CTRL,    0x4801C02A,__READ_WRITE ,__sdmmc_block_gap_ctrl_bits);
__IO_REG8_BIT( SDMMC_WAKEUP_CTRL,       0x4801C02B,__READ_WRITE ,__sdmmc_wakeup_ctrl_bits);
__IO_REG16_BIT(SDMMC_CLOCK_CTRL,        0x4801C02C,__READ_WRITE ,__sdmmc_clock_ctrl_bits);
__IO_REG8_BIT( SDMMC_TIMEOUT_CTRL,      0x4801C02E,__READ_WRITE ,__sdmmc_timeout_ctrl_bits);
__IO_REG8_BIT( SDMMC_SW_RESET,          0x4801C02F,__READ_WRITE ,__sdmmc_sw_reset_bits);
__IO_REG16_BIT(SDMMC_INT_STATUS_NORM,   0x4801C030,__READ_WRITE ,__sdmmc_int_status_norm_bits);
__IO_REG16_BIT(SDMMC_INT_STATUS_ERR,    0x4801C032,__READ_WRITE ,__sdmmc_int_status_err_bits);
__IO_REG16_BIT(SDMMC_EN_INT_STATUS_NORM,0x4801C034,__READ_WRITE ,__sdmmc_en_int_status_norm_bits);
__IO_REG16_BIT(SDMMC_EN_INT_STATUS_ERR, 0x4801C036,__READ_WRITE ,__sdmmc_en_int_status_err_bits);
__IO_REG16_BIT(SDMMC_EN_INT_SIGNAL_NORM,0x4801C038,__READ_WRITE ,__sdmmc_en_int_signal_norm_bits);
__IO_REG16_BIT(SDMMC_EN_INT_SIGNAL_ERR, 0x4801C03A,__READ_WRITE ,__sdmmc_en_int_signal_err_bits);
__IO_REG16_BIT(SDMMC_ACMD_ERR_STATUS,   0x4801C03C,__READ       ,__sdmmc_acmd_err_status_bits);
__IO_REG16_BIT(SDMMC_FORCE_EVENT_ACMD_ERR_STATUS, 0x4801C050,__WRITE      ,__sdmmc_force_event_acmd_err_status_bits);
__IO_REG16_BIT(SDMMC_FORCE_EVENT_ERR_STATUS,      0x4801C052,__WRITE      ,__sdmmc_force_event_err_status_bits);
__IO_REG32_BIT(SDMMC_DEBUG_SEL,         0x4801C074,__WRITE      ,__sdmmc_debug_sel_bits);
__IO_REG32_BIT(SDMMC_SPI,               0x4801C0F0,__READ_WRITE ,__sdmmc_spi_bits);
__IO_REG16_BIT(SDMMC_SLOT_INT_STATUS,   0x4801C0FC,__READ_WRITE ,__sdmmc_slot_int_status_bits);

/***************************************************************************
 **
 ** ETH0
 **
 ***************************************************************************/
__IO_REG32_BIT(ETH0_MAC_Configuration,            0x5000C000,__READ_WRITE ,__eth_mac_configuration_bits);
__IO_REG32_BIT(ETH0_MAC_Frame_Filter,             0x5000C004,__READ_WRITE ,__eth_mac_frame_filter_bits);
__IO_REG32(    ETH0_Hash_Table_High,              0x5000C008,__READ_WRITE );
__IO_REG32(    ETH0_Hash_Table_Low,               0x5000C00C,__READ_WRITE );
__IO_REG32_BIT(ETH0_MII_Address,                  0x5000C010,__READ_WRITE ,__eth_mii_address_bits);
__IO_REG32_BIT(ETH0_MII_Data,                     0x5000C014,__READ_WRITE ,__eth_mii_data_bits);
__IO_REG32_BIT(ETH0_Flow_Control,                 0x5000C018,__READ_WRITE ,__eth_flow_control_bits);
__IO_REG32_BIT(ETH0_VLAN_Tag,                     0x5000C01C,__READ_WRITE ,__eth_vlan_tag_bits);
__IO_REG32_BIT(ETH0_Version,                      0x5000C020,__READ       ,__eth_version_bits);
__IO_REG32_BIT(ETH0_Debug,                        0x5000C024,__READ       ,__eth_debug_bits);
__IO_REG32(    ETH0_Remote_Wake_Up_Frame_Filter,  0x5000C028,__READ_WRITE );
__IO_REG32_BIT(ETH0_PMT_Control_Status,           0x5000C02C,__READ_WRITE ,__eth_pmt_control_status_bits);
__IO_REG32_BIT(ETH0_Interrupt_Status,             0x5000C038,__READ       ,__eth_interrupt_status_bits);
__IO_REG32_BIT(ETH0_Interrupt_Mask,               0x5000C03C,__READ_WRITE ,__eth_interrupt_status_bits);
__IO_REG32_BIT(ETH0_MAC_Address0_High,            0x5000C040,__READ_WRITE ,__eth_mac_address0_high_bits);
__IO_REG32(    ETH0_MAC_Address0_Low,             0x5000C044,__READ_WRITE );
__IO_REG32_BIT(ETH0_MAC_Address1_High,            0x5000C048,__READ_WRITE ,__eth_mac_address1_high_bits);
__IO_REG32(    ETH0_MAC_Address1_Low,             0x5000C04C,__READ_WRITE );
__IO_REG32_BIT(ETH0_MAC_Address2_High,            0x5000C050,__READ_WRITE ,__eth_mac_address1_high_bits);
__IO_REG32(    ETH0_MAC_Address2_Low,             0x5000C054,__READ_WRITE );
__IO_REG32_BIT(ETH0_MAC_Address3_High,            0x5000C058,__READ_WRITE ,__eth_mac_address1_high_bits);
__IO_REG32(    ETH0_MAC_Address3_Low,             0x5000C05C,__READ_WRITE );
__IO_REG32_BIT(ETH0_MMC_Control,                  0x5000C100,__READ_WRITE ,__eth_mmc_control_bits);
__IO_REG32_BIT(ETH0_MMC_Receive_Interrupt,        0x5000C104,__READ       ,__eth_mmc_receive_interrupt_bits);
__IO_REG32_BIT(ETH0_MMC_Transmit_Interrupt,       0x5000C108,__READ       ,__eth_mmc_transmit_interrupt_bits);
__IO_REG32_BIT(ETH0_MMC_Receive_Interrupt_Mask,   0x5000C10C,__READ_WRITE ,__eth_mmc_receive_interrupt_mask_bits);
__IO_REG32_BIT(ETH0_MMC_Transmit_Interrupt_Mask,  0x5000C110,__READ_WRITE ,__eth_mmc_transmit_interrupt_mask_bits);
__IO_REG32(    ETH0_Tx_Octet_Count_Good_Bad,      0x5000C114,__READ       );
__IO_REG32(    ETH0_Tx_Frame_Count_Good_Bad,      0x5000C118,__READ       );
__IO_REG32(    ETH0_Tx_Broadcast_Frames_Good,     0x5000C11C,__READ       );
__IO_REG32(    ETH0_Tx_Multicast_Frames_Good,     0x5000C120,__READ       );
__IO_REG32(    ETH0_Tx_64Octets_Frames_Good_Bad,  0x5000C124,__READ       );
__IO_REG32(    ETH0_Tx_65To127Octets_Frames_Good_Bad,   0x5000C128,__READ       );
__IO_REG32(    ETH0_Tx_128To255Octets_Frames_Good_Bad,  0x5000C12C,__READ       );
__IO_REG32(    ETH0_Tx_256To511Octets_Frames_Good_Bad,  0x5000C130,__READ       );
__IO_REG32(    ETH0_Tx_512To1023Octets_Frames_Good_Bad, 0x5000C134,__READ       );
__IO_REG32(    ETH0_Tx_1024ToMaxOctets_Frames_Good_Bad, 0x5000C138,__READ       );
__IO_REG32(    ETH0_Tx_Unicast_Frames_Good_Bad,   0x5000C13C,__READ       );
__IO_REG32(    ETH0_Tx_Multicast_Frames_Good_Bad, 0x5000C140,__READ       );
__IO_REG32(    ETH0_Tx_Broadcast_Frames_Good_Bad, 0x5000C144,__READ       );
__IO_REG32(    ETH0_Tx_Underflow_Error_Frames,    0x5000C148,__READ       );
__IO_REG32(    ETH0_Tx_Single_Collision_Good_Frames,    0x5000C14C,__READ       );
__IO_REG32(    ETH0_Tx_Multiple_Collision_Good_Frames,  0x5000C150,__READ       );
__IO_REG32(    ETH0_Tx_Deferred_Frames,           0x5000C154,__READ       );
__IO_REG32(    ETH0_Tx_Late_Collision_Frames,     0x5000C158,__READ       );
__IO_REG32(    ETH0_Tx_Excessive_Collision_Frames,0x5000C15C,__READ       );
__IO_REG32(    ETH0_Tx_Carrier_Error_Frames,      0x5000C160,__READ       );
__IO_REG32(    ETH0_Tx_Octet_Count_Good,          0x5000C164,__READ       );
__IO_REG32(    ETH0_Tx_Frame_Count_Good,          0x5000C168,__READ       );
__IO_REG32(    ETH0_Tx_Excessive_Deferral_Error,  0x5000C16C,__READ       );
__IO_REG32(    ETH0_Tx_Pause_Frames,              0x5000C170,__READ       );
__IO_REG32(    ETH0_Tx_VLAN_Frames_Good,          0x5000C174,__READ       );
__IO_REG32(    ETH0_Tx_OSize_Frames_Good,         0x5000C178,__READ       );
__IO_REG32(    ETH0_Rx_Frames_Count_Good_Bad,     0x5000C180,__READ       );
__IO_REG32(    ETH0_Rx_Octet_Count_Good_Bad,      0x5000C184,__READ       );
__IO_REG32(    ETH0_Rx_Octet_Count_Good,          0x5000C188,__READ       );
__IO_REG32(    ETH0_Rx_Broadcast_Frames_Good,     0x5000C18C,__READ       );
__IO_REG32(    ETH0_Rx_Multicast_Frames_Good,     0x5000C190,__READ       );
__IO_REG32(    ETH0_Rx_CRC_Error_Frames,          0x5000C194,__READ       );
__IO_REG32(    ETH0_Rx_Alignment_Error_Frames,    0x5000C198,__READ       );
__IO_REG32(    ETH0_Rx_Runt_Error_Frames,         0x5000C19C,__READ       );
__IO_REG32(    ETH0_Rx_Jabber_Error_Frames,       0x5000C1A0,__READ       );
__IO_REG32(    ETH0_Rx_Undersize_Frames_Good,     0x5000C1A4,__READ       );
__IO_REG32(    ETH0_Rx_Oversize_Frames_Good,      0x5000C1A8,__READ       );
__IO_REG32(    ETH0_Rx_64Octets_Frames_Good_Bad,  0x5000C1AC,__READ       );
__IO_REG32(    ETH0_Rx_65To127Octets_Frames_Good_Bad,   0x5000C1B0,__READ       );
__IO_REG32(    ETH0_Rx_128To255Octets_Frames_Good_Bad,  0x5000C1B4,__READ       );
__IO_REG32(    ETH0_Rx_256To511Octets_Frames_Good_Bad,  0x5000C1B8,__READ       );
__IO_REG32(    ETH0_Rx_512To1023Octets_Frames_Good_Bad, 0x5000C1BC,__READ       );
__IO_REG32(    ETH0_Rx_1024ToMaxOctets_Frames_Good_Bad, 0x5000C1C0,__READ       );
__IO_REG32(    ETH0_Rx_Unicast_Frames_Good,       0x5000C1C4,__READ       );
__IO_REG32(    ETH0_Rx_Length_Error_Frames,       0x5000C1C8,__READ       );
__IO_REG32(    ETH0_Rx_Out_Of_Range_Type_Frames,  0x5000C1CC,__READ       );
__IO_REG32(    ETH0_Rx_Pause_Frames,              0x5000C1D0,__READ       );
__IO_REG32(    ETH0_Rx_FIFO_Overflow_Frames,      0x5000C1D4,__READ       );
__IO_REG32(    ETH0_Rx_VLAN_Frames_Good_Bad,      0x5000C1D8,__READ       );
__IO_REG32(    ETH0_Rx_Watchdog_Error_Frames,     0x5000C1DC,__READ       );
__IO_REG32(    ETH0_Rx_Receive_Error_Frames,      0x5000C1E0,__READ       );
__IO_REG32(    ETH0_Rx_Control_Frames_Good,       0x5000C1E4,__READ       );
__IO_REG32_BIT(ETH0_MMC_IPC_Receive_Interrupt_Mask,       0x5000C200,__READ_WRITE ,__eth_mmc_ipc_receive_interrupt_mask_bits);
__IO_REG32_BIT(ETH0_MMC_IPC_Receive_Interrupt,    0x5000C208,__READ       ,__eth_mmc_ipc_receive_interrupt_bits);
__IO_REG32(    ETH0_RxIPv4_Good_Frames,           0x5000C210,__READ       );
__IO_REG32(    ETH0_RxIPv4_Header_Error_Frames,   0x5000C214,__READ       );
__IO_REG32(    ETH0_RxIPv4_No_Payload_Frames,     0x5000C218,__READ       );
__IO_REG32(    ETH0_RxIPv4_Fragmented_Frames,     0x5000C21C,__READ       );
__IO_REG32(    ETH0_RxIPv4_UDP_Checksum_Disabled_Frames,  0x5000C220,__READ       );
__IO_REG32(    ETH0_RxIPv6_Good_Frames,           0x5000C224,__READ       );
__IO_REG32(    ETH0_RxIPv6_Header_Error_Frames,   0x5000C228,__READ       );
__IO_REG32(    ETH0_RxIPv6_No_Payload_Frames,     0x5000C22C,__READ       );
__IO_REG32(    ETH0_RxUDP_Good_Frames,            0x5000C230,__READ       );
__IO_REG32(    ETH0_RxUDP_Error_Frames,           0x5000C234,__READ       );
__IO_REG32(    ETH0_RxTCP_Good_Frames,            0x5000C238,__READ       );
__IO_REG32(    ETH0_RxTCP_Error_Frames,           0x5000C23C,__READ       );
__IO_REG32(    ETH0_RxICMP_Good_Frames,           0x5000C240,__READ       );
__IO_REG32(    ETH0_RxICMP_Error_Frames,          0x5000C244,__READ       );
__IO_REG32(    ETH0_RxIPv4_Good_Octets,           0x5000C250,__READ       );
__IO_REG32(    ETH0_RxIPv4_Header_Error_Octets,   0x5000C254,__READ       );
__IO_REG32(    ETH0_RxIPv4_No_Payload_Octets,     0x5000C258,__READ       );
__IO_REG32(    ETH0_RxIPv4_Fragmented_Octets,     0x5000C25C,__READ       );
__IO_REG32(    ETH0_RxIPv4_UDP_Checksum_Disable_Octets,   0x5000C260,__READ       );
__IO_REG32(    ETH0_RxIPv6_Good_Octets,           0x5000C264,__READ       );
__IO_REG32(    ETH0_RxIPv6_Header_Error_Octets,   0x5000C268,__READ       );
__IO_REG32(    ETH0_RxIPv6_No_Payload_Octets,     0x5000C26C,__READ       );
__IO_REG32(    ETH0_RxUDP_Good_Octets,            0x5000C270,__READ       );
__IO_REG32(    ETH0_RxUDP_Error_Octets,           0x5000C274,__READ       );
__IO_REG32(    ETH0_RxTCP_Good_Octets,            0x5000C278,__READ       );
__IO_REG32(    ETH0_RxTCP_Error_Octets,           0x5000C27C,__READ       );
__IO_REG32(    ETH0_RxICMP_Good_Octets,           0x5000C280,__READ       );
__IO_REG32(    ETH0_RxICMP_Error_Octets,          0x5000C284,__READ       );
__IO_REG32_BIT(ETH0_Timestamp_Control,            0x5000C700,__READ       ,__eth_timestamp_control_bits);
__IO_REG32_BIT(ETH0_Sub_Second_Increment,         0x5000C704,__READ       ,__eth_sub_second_increment_bits);
__IO_REG32(    ETH0_System_Time_Seconds,          0x5000C708,__READ       );
__IO_REG32_BIT(ETH0_System_Time_Nanoseconds,      0x5000C70C,__READ       ,__eth_system_time_nanoseconds_bits);
__IO_REG32(    ETH0_System_Time_Seconds_Update,   0x5000C710,__READ       );
__IO_REG32_BIT(ETH0_System_Time_Nanoseconds_Update,     0x5000C714,__READ_WRITE ,__eth_system_time_nanoseconds_update_bits);
__IO_REG32(    ETH0_Timestamp_Addend,             0x5000C718,__READ_WRITE );
__IO_REG32(    ETH0_Target_Time_Seconds,          0x5000C71C,__READ_WRITE );
__IO_REG32_BIT(ETH0_Target_Time_Nanoseconds,      0x5000C720,__READ_WRITE ,__eth_target_time_nanoseconds_bits);
__IO_REG32_BIT(ETH0_System_Time_Higher_Word_Seconds,    0x5000C724,__READ_WRITE ,__eth_system_time_higher_word_seconds_bits);
__IO_REG32_BIT(ETH0_Timestamp_Status,             0x5000C728,__READ       ,__eth_timestamp_status_bits);
__IO_REG32_BIT(ETH0_PPS_Control,                  0x5000C72C,__READ_WRITE ,__eth_pps_control_bits);
__IO_REG32_BIT(ETH0_Bus_Mode,                     0x5000E000,__READ_WRITE ,__eth_bus_mode_bits);
__IO_REG32(    ETH0_Transmit_Poll_Demand,         0x5000E004,__READ_WRITE );
__IO_REG32(    ETH0_Receive_Poll_Demand,          0x5000E008,__READ_WRITE );
__IO_REG32(    ETH0_Receive_Descriptor_List_Address,    0x5000E00C,__READ_WRITE );
__IO_REG32(    ETH0_Transmit_Descriptor_List_Address,   0x5000E010,__READ_WRITE );
__IO_REG32_BIT(ETH0_Status,                       0x5000E014,__READ_WRITE ,__eth_status_bits);
__IO_REG32_BIT(ETH0_Operation_Mode,               0x5000E018,__READ_WRITE ,__eth_operation_mode_bits);
__IO_REG32_BIT(ETH0_Interrupt_Enable,             0x5000E01C,__READ_WRITE ,__eth_interrupt_enable_bits);
__IO_REG32_BIT(ETH0_Missed_Frame_And_Buffer_Overflow_Counter, 0x5000E020,__READ       ,__eth_missed_frame_and_buffer_overflow_counter_bits);
__IO_REG32_BIT(ETH0_Receive_Interrupt_Watchdog_Timer,         0x5000E024,__READ_WRITE ,__eth_receive_interrupt_watchdog_timer_bits);
__IO_REG32_BIT(ETH0_AHB_Status,                   0x5000E02C,__READ       ,__eth_ahb_status_bits);
__IO_REG32(    ETH0_Current_Host_Transmit_Descriptor,         0x5000E048,__READ       );
__IO_REG32(    ETH0_Current_Host_Receive_Descriptor,          0x5000E04C,__READ       );
__IO_REG32(    ETH0_Current_Host_Transmit_Buffer_Address,     0x5000E050,__READ       );
__IO_REG32(    ETH0_Current_Host_Receive_Buffer_Address,      0x5000E054,__READ       );
__IO_REG32_BIT(ETH0_HW_Feature,                   0x5000E058,__READ_WRITE ,__eth_hw_feature_bits);

/***************************************************************************
 **
 ** USB Global
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_GOTGCTL,             0x50040000,__READ_WRITE ,__usb_gotgctl_bits);
__IO_REG32_BIT(USB_GOTGINT,             0x50040004,__READ_WRITE ,__usb_gotgint_bits);
__IO_REG32_BIT(USB_GAHBCFG,             0x50040008,__READ_WRITE ,__usb_gahbcfg_bits);
__IO_REG32_BIT(USB_GUSBCFG,             0x5004000C,__READ_WRITE ,__usb_gusbcfg_bits);
__IO_REG32_BIT(USB_GRSTCTL,             0x50040010,__READ_WRITE ,__usb_grstctl_bits);
__IO_REG32_BIT(USB_GINTSTS,             0x50040014,__READ_WRITE ,__usb_gintsts_bits);
#define USBH_GINTSTS        USB_GINTSTS
#define USBH_GINTSTS_bit    USB_GINTSTS_bit
#define USBD_GINTSTS        USB_GINTSTS
#define USBD_GINTSTS_bit    USB_GINTSTS_bit.dev
__IO_REG32_BIT(USB_GINTMSK,             0x50040018,__READ_WRITE ,__usb_gintmsk_bits);
#define USBH_GINTMSK        USB_GINTMSK
#define USBH_GINTMSK_bit    USB_GINTMSK_bit
#define USBD_GINTMSK        USB_GINTMSK
#define USBD_GINTMSK_bit    USB_GINTMSK_bit.dev
__IO_REG32_BIT(USB_GRXSTSR,             0x5004001C,__READ       ,__usb_grxstsr_bits);
__IO_REG32_BIT(USB_GRXSTSP,             0x50040020,__READ       ,__usb_grxstsr_bits);
__IO_REG32_BIT(USB_GRXFSIZ,             0x50040024,__READ_WRITE ,__usb_grxfsiz_bits);
__IO_REG32_BIT(USB_GNPTXFSIZ,           0x50040028,__READ_WRITE ,__usb_gnptxfsiz_bits);
#define USBH_GNPTXFSIZ      USB_GNPTXFSIZ
#define USBH_GNPTXFSIZ_bit  USB_GNPTXFSIZ_bit
#define USBD_GNPTXFSIZ      USB_GNPTXFSIZ
#define USBD_GNPTXFSIZ_bit  USB_GNPTXFSIZ_bit.dev
__IO_REG32_BIT(USB_GNPTXSTS,            0x5004002C,__READ       ,__usb_gnptxsts_bits);
__IO_REG32_BIT(USB_GGPIO,               0x50040038,__READ_WRITE ,__usb_ggpio_bits);
__IO_REG32(    USB_GUID,                0x5004003C,__READ_WRITE );
__IO_REG32_BIT(USB_GDFIFOCFG,           0x5004005C,__READ_WRITE ,__usb_gdfifocfg_bits);
__IO_REG32_BIT(USB_HPTXFSIZ,            0x50040100,__READ_WRITE ,__usb_hptxfsiz_bits);
__IO_REG32_BIT(USB_DIEPTXF1,            0x50040104,__READ_WRITE ,__usb_dieptxf_bits);
__IO_REG32_BIT(USB_DIEPTXF2,            0x50040108,__READ_WRITE ,__usb_dieptxf_bits);
__IO_REG32_BIT(USB_DIEPTXF3,            0x5004010C,__READ_WRITE ,__usb_dieptxf_bits);
__IO_REG32_BIT(USB_DIEPTXF4,            0x50040110,__READ_WRITE ,__usb_dieptxf_bits);
__IO_REG32_BIT(USB_DIEPTXF5,            0x50040114,__READ_WRITE ,__usb_dieptxf_bits);
__IO_REG32_BIT(USB_DIEPTXF6,            0x50040118,__READ_WRITE ,__usb_dieptxf_bits);

/***************************************************************************
 **
 ** USB Host
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_HCFG,                0x50040400,__READ_WRITE ,__usb_hcfg_bits);
__IO_REG32_BIT(USB_HFIR,                0x50040404,__READ_WRITE ,__usb_hfir_bits);
__IO_REG32_BIT(USB_HFNUM,               0x50040408,__READ_WRITE ,__usb_hfnum_bits);
__IO_REG32_BIT(USB_HPTXSTS,             0x50040410,__READ_WRITE ,__usb_hptxsts_bits);
__IO_REG32_BIT(USB_HAINT,               0x50040414,__READ       ,__usb_haint_bits);
__IO_REG32_BIT(USB_HAINTMSK,            0x50040418,__READ_WRITE ,__usb_haintmsk_bits);
__IO_REG32(    USB_HFLBADDR,            0x5004041C,__READ_WRITE );
__IO_REG32_BIT(USB_HPRT,                0x50040440,__READ_WRITE ,__usb_hprt_bits);
__IO_REG32_BIT(USB_HCCHAR0,             0x50040500,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT0,              0x50040508,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK0,           0x5004050C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ0,             0x50040510,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG0      USB_HCTSIZ0
#define USB_HCTSIZ_SG0_bit  USB_HCTSIZ0_bit
#define USB_HCTSIZ_B0       USB_HCTSIZ0
#define USB_HCTSIZ_B0_bit   USB_HCTSIZ0_bit.buffer
__IO_REG32_BIT(USB_HCDMA0,              0x50040514,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB0,             0x5004051C,__READ       );
__IO_REG32_BIT(USB_HCCHAR1,             0x50040520,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT1,              0x50040528,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK1,           0x5004052C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ1,             0x50040530,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG1      USB_HCTSIZ1
#define USB_HCTSIZ_SG1_bit  USB_HCTSIZ1_bit
#define USB_HCTSIZ_B1       USB_HCTSIZ1
#define USB_HCTSIZ_B1_bit   USB_HCTSIZ1_bit.buffer
__IO_REG32_BIT(USB_HCDMA1,              0x50040534,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB1,             0x5004053C,__READ       );
__IO_REG32_BIT(USB_HCCHAR2,             0x50040540,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT2,              0x50040548,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK2,           0x5004054C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ2,             0x50040550,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG2      USB_HCTSIZ2
#define USB_HCTSIZ_SG2_bit  USB_HCTSIZ2_bit
#define USB_HCTSIZ_B2       USB_HCTSIZ2
#define USB_HCTSIZ_B2_bit   USB_HCTSIZ2_bit.buffer
__IO_REG32_BIT(USB_HCDMA2,              0x50040554,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB2,             0x5004055C,__READ       );
__IO_REG32_BIT(USB_HCCHAR3,             0x50040560,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT3,              0x50040568,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK3,           0x5004056C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ3,             0x50040570,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG3      USB_HCTSIZ3
#define USB_HCTSIZ_SG3_bit  USB_HCTSIZ3_bit
#define USB_HCTSIZ_B3       USB_HCTSIZ3
#define USB_HCTSIZ_B3_bit   USB_HCTSIZ3_bit.buffer
__IO_REG32_BIT(USB_HCDMA3,              0x50040574,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB3,             0x5004057C,__READ       );
__IO_REG32_BIT(USB_HCCHAR4,             0x50040580,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT4,              0x50040588,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK4,           0x5004058C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ4,             0x50040590,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG4      USB_HCTSIZ4
#define USB_HCTSIZ_SG4_bit  USB_HCTSIZ4_bit
#define USB_HCTSIZ_B4       USB_HCTSIZ4
#define USB_HCTSIZ_B4_bit   USB_HCTSIZ4_bit.buffer
__IO_REG32_BIT(USB_HCDMA4,              0x50040594,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB4,             0x5004059C,__READ       );
__IO_REG32_BIT(USB_HCCHAR5,             0x500405A0,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT5,              0x500405A8,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK5,           0x500405AC,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ5,             0x500405B0,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG5      USB_HCTSIZ5
#define USB_HCTSIZ_SG5_bit  USB_HCTSIZ5_bit
#define USB_HCTSIZ_B5       USB_HCTSIZ5
#define USB_HCTSIZ_B5_bit   USB_HCTSIZ5_bit.buffer
__IO_REG32_BIT(USB_HCDMA5,              0x500405B4,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB5,             0x500405BC,__READ       );
__IO_REG32_BIT(USB_HCCHAR6,             0x500405C0,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT6,              0x500405C8,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK6,           0x500405CC,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ6,             0x500405D0,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG6      USB_HCTSIZ6
#define USB_HCTSIZ_SG6_bit  USB_HCTSIZ6_bit
#define USB_HCTSIZ_B6       USB_HCTSIZ6
#define USB_HCTSIZ_B6_bit   USB_HCTSIZ6_bit.buffer
__IO_REG32_BIT(USB_HCDMA6,              0x500405D4,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB6,             0x500405DC,__READ       );
__IO_REG32_BIT(USB_HCCHAR7,             0x500405E0,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT7,              0x500405E8,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK7,           0x500405EC,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ7,             0x500405F0,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG7      USB_HCTSIZ7
#define USB_HCTSIZ_SG7_bit  USB_HCTSIZ7_bit
#define USB_HCTSIZ_B7       USB_HCTSIZ7
#define USB_HCTSIZ_B7_bit   USB_HCTSIZ7_bit.buffer
__IO_REG32_BIT(USB_HCDMA7,              0x500405F4,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB7,             0x500405FC,__READ       );
__IO_REG32_BIT(USB_HCCHAR8,             0x50040600,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT8,              0x50040608,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK8,           0x5004060C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ8,             0x50040610,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG8      USB_HCTSIZ8
#define USB_HCTSIZ_SG8_bit  USB_HCTSIZ8_bit
#define USB_HCTSIZ_B8       USB_HCTSIZ8
#define USB_HCTSIZ_B8_bit   USB_HCTSIZ8_bit.buffer
__IO_REG32_BIT(USB_HCDMA8,              0x50040614,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB8,             0x5004061C,__READ       );
__IO_REG32_BIT(USB_HCCHAR9,             0x50040620,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT9,              0x50040628,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK9,           0x5004062C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ9,             0x50040630,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG9      USB_HCTSIZ9
#define USB_HCTSIZ_SG9_bit  USB_HCTSIZ9_bit
#define USB_HCTSIZ_B9       USB_HCTSIZ9
#define USB_HCTSIZ_B9_bit   USB_HCTSIZ9_bit.buffer
__IO_REG32_BIT(USB_HCDMA9,              0x50040634,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB9,             0x5004063C,__READ       );
__IO_REG32_BIT(USB_HCCHAR10,            0x50040640,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT10,             0x50040648,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK10,          0x5004064C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ10,            0x50040650,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG10     USB_HCTSIZ10
#define USB_HCTSIZ_SG10_bit USB_HCTSIZ10_bit
#define USB_HCTSIZ_B10      USB_HCTSIZ10
#define USB_HCTSIZ_B10_bit  USB_HCTSIZ10_bit.buffer
__IO_REG32_BIT(USB_HCDMA10,             0x50040654,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB10,            0x5004065C,__READ       );
__IO_REG32_BIT(USB_HCCHAR11,            0x50040660,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT11,             0x50040668,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK11,          0x5004066C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ11,            0x50040670,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG11     USB_HCTSIZ11
#define USB_HCTSIZ_SG11_bit USB_HCTSIZ11_bit
#define USB_HCTSIZ_B11      USB_HCTSIZ11
#define USB_HCTSIZ_B11_bit  USB_HCTSIZ11_bit.buffer
__IO_REG32_BIT(USB_HCDMA11,             0x50040674,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB11,            0x5004067C,__READ       );
__IO_REG32_BIT(USB_HCCHAR12,            0x50040680,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT12,             0x50040688,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK12,          0x5004068C,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ12,            0x50040690,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG12     USB_HCTSIZ12
#define USB_HCTSIZ_SG12_bit USB_HCTSIZ12_bit
#define USB_HCTSIZ_B12      USB_HCTSIZ12
#define USB_HCTSIZ_B12_bit  USB_HCTSIZ12_bit.buffer
__IO_REG32_BIT(USB_HCDMA12,             0x50040694,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB12,            0x5004069C,__READ       );
__IO_REG32_BIT(USB_HCCHAR13,            0x500406A0,__READ_WRITE ,__usb_hcchar_bits);
__IO_REG32_BIT(USB_HCINT13,             0x500406A8,__READ_WRITE ,__usb_hcint_bits);
__IO_REG32_BIT(USB_HCINTMSK13,          0x500406AC,__READ_WRITE ,__usb_hcintmsk_bits);
__IO_REG32_BIT(USB_HCTSIZ13,            0x500406B0,__READ_WRITE ,__usb_hctsiz_bits);
#define USB_HCTSIZ_SG13     USB_HCTSIZ13
#define USB_HCTSIZ_SG13_bit USB_HCTSIZ13_bit
#define USB_HCTSIZ_B13      USB_HCTSIZ13
#define USB_HCTSIZ_B13_bit  USB_HCTSIZ13_bit.buffer
__IO_REG32_BIT(USB_HCDMA13,             0x500406B4,__READ_WRITE ,__usb_hcdma_bits);
__IO_REG32(    USB_HCDMAB13,            0x500406BC,__READ       );

/***************************************************************************
 **
 ** USB Device
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_DCFG,                0x50040800,__READ_WRITE ,__usb_dcfg_bits);
__IO_REG32_BIT(USB_DCTL,                0x50040804,__READ_WRITE ,__usb_dctl_bits);
__IO_REG32_BIT(USB_DSTS,                0x50040808,__READ       ,__usb_dsts_bits);
__IO_REG32_BIT(USB_DIEPMSK,             0x50040810,__READ_WRITE ,__usb_diepmsk_bits);
__IO_REG32_BIT(USB_DOEPMSK,             0x50040814,__READ_WRITE ,__usb_doepmsk_bits);
__IO_REG32_BIT(USB_DAINT,               0x50040818,__READ       ,__usb_daint_bits);
__IO_REG32_BIT(USB_DAINTMSK,            0x5004081C,__READ_WRITE ,__usb_daintmsk_bits);
__IO_REG32_BIT(USB_DVBUSDIS,            0x50040828,__READ_WRITE ,__usb_dvbusdis_bits);
__IO_REG32_BIT(USB_DVBUSPULSE,          0x5004082C,__READ_WRITE ,__usb_dvbuspulse_bits);
__IO_REG32_BIT(USB_DIEPEMPMSK,          0x50040834,__READ_WRITE ,__usb_diepempmsk_bits);
__IO_REG32_BIT(USB_DIEPCTL0,            0x50040900,__READ_WRITE ,__usb_diepctl0_bits);
__IO_REG32_BIT(USB_DIEPINT0,            0x50040908,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ0,           0x50040910,__READ_WRITE ,__usb_dieptsiz0_bits);
__IO_REG32(    USB_DIEPDMA0,            0x50040914,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS0,            0x50040918,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB0,           0x5004091C,__READ       );
__IO_REG32_BIT(USB_DIEPCTL1,            0x50040920,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DIEPINT1,            0x50040928,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ1,           0x50040930,__READ_WRITE ,__usb_dieptsiz_bits);
__IO_REG32(    USB_DIEPDMA1,            0x50040934,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS1,            0x50040938,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB1,           0x5004093C,__READ       );
__IO_REG32_BIT(USB_DIEPCTL2,            0x50040940,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DIEPINT2,            0x50040948,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ2,           0x50040950,__READ_WRITE ,__usb_dieptsiz_bits);
__IO_REG32(    USB_DIEPDMA2,            0x50040954,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS2,            0x50040958,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB2,           0x5004095C,__READ       );
__IO_REG32_BIT(USB_DIEPCTL3,            0x50040960,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DIEPINT3,            0x50040968,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ3,           0x50040970,__READ_WRITE ,__usb_dieptsiz_bits);
__IO_REG32(    USB_DIEPDMA3,            0x50040974,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS3,            0x50040978,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB3,           0x5004097C,__READ       );
__IO_REG32_BIT(USB_DIEPCTL4,            0x50040980,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DIEPINT4,            0x50040988,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ4,           0x50040990,__READ_WRITE ,__usb_dieptsiz_bits);
__IO_REG32(    USB_DIEPDMA4,            0x50040994,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS4,            0x50040998,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB4,           0x5004099C,__READ       );
__IO_REG32_BIT(USB_DIEPCTL5,            0x500409A0,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DIEPINT5,            0x500409A8,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ5,           0x500409B0,__READ_WRITE ,__usb_dieptsiz_bits);
__IO_REG32(    USB_DIEPDMA5,            0x500409B4,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS5,            0x500409B8,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB5,           0x500409BC,__READ       );
__IO_REG32_BIT(USB_DIEPCTL6,            0x500409C0,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DIEPINT6,            0x500409C8,__READ_WRITE ,__usb_diepint_bits);
__IO_REG32_BIT(USB_DIEPTSIZ6,           0x500409D0,__READ_WRITE ,__usb_dieptsiz_bits);
__IO_REG32(    USB_DIEPDMA6,            0x500409D4,__READ_WRITE );
__IO_REG32_BIT(USB_DTXFSTS6,            0x500409D8,__READ_WRITE ,__usb_dtxfsts_bits);
__IO_REG32(    USB_DIEPDMAB6,           0x500409DC,__READ       );
__IO_REG32_BIT(USB_DOEPCTL0,            0x50040B00,__READ_WRITE ,__usb_doepctl0_bits);
__IO_REG32_BIT(USB_DOEPINT0,            0x50040B08,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ0,           0x50040B10,__READ_WRITE ,__usb_doeptsiz0_bits);
__IO_REG32(    USB_DOEPDMA0,            0x50040B14,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB0,           0x50040B1C,__READ       );
__IO_REG32_BIT(USB_DOEPCTL1,            0x50040B20,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DOEPINT1,            0x50040B28,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ1,           0x50040B30,__READ_WRITE ,__usb_doeptsiz_bits);
__IO_REG32(    USB_DOEPDMA1,            0x50040B34,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB1,           0x50040B3C,__READ       );
__IO_REG32_BIT(USB_DOEPCTL2,            0x50040B40,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DOEPINT2,            0x50040B48,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ2,           0x50040B50,__READ_WRITE ,__usb_doeptsiz_bits);
__IO_REG32(    USB_DOEPDMA2,            0x50040B54,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB2,           0x50040B5C,__READ       );
__IO_REG32_BIT(USB_DOEPCTL3,            0x50040B60,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DOEPINT3,            0x50040B68,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ3,           0x50040B70,__READ_WRITE ,__usb_doeptsiz_bits);
__IO_REG32(    USB_DOEPDMA3,            0x50040B74,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB3,           0x50040B7C,__READ       );
__IO_REG32_BIT(USB_DOEPCTL4,            0x50040B80,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DOEPINT4,            0x50040B88,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ4,           0x50040B90,__READ_WRITE ,__usb_doeptsiz_bits);
__IO_REG32(    USB_DOEPDMA4,            0x50040B94,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB4,           0x50040B9C,__READ       );
__IO_REG32_BIT(USB_DOEPCTL5,            0x50040BA0,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DOEPINT5,            0x50040BA8,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ5,           0x50040BB0,__READ_WRITE ,__usb_doeptsiz_bits);
__IO_REG32(    USB_DOEPDMA5,            0x50040BB4,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB5,           0x50040BBC,__READ       );
__IO_REG32_BIT(USB_DOEPCTL6,            0x50040BC0,__READ_WRITE ,__usb_diepctl_bits);
__IO_REG32_BIT(USB_DOEPINT6,            0x50040BC8,__READ_WRITE ,__usb_doepint_bits);
__IO_REG32_BIT(USB_DOEPTSIZ6,           0x50040BD0,__READ_WRITE ,__usb_doeptsiz_bits);
__IO_REG32(    USB_DOEPDMA6,            0x50040BD4,__READ_WRITE );
__IO_REG32(    USB_DOEPDMAB6,           0x50040BDC,__READ       );

/***************************************************************************
 **
 ** USB Power
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_PCGCR,               0x50040E00,__READ_WRITE ,__usb_pcgcr_bits);

/***************************************************************************
 **
 ** USIC0
 **
 ***************************************************************************/
__IO_REG32_BIT(USIC0_ID,                0x40030000,__READ_WRITE ,__usic_id_bits);
__IO_REG32_BIT(USIC0_C0_CCFG,           0x40030004,__READ       ,__usic_ccfg_bits);
__IO_REG32_BIT(USIC0_C0_KSCFG,          0x4003000C,__READ_WRITE ,__usic_kscfg_bits);
__IO_REG32_BIT(USIC0_C0_FDR,            0x40030010,__READ_WRITE ,__usic_fdr_bits);
__IO_REG32_BIT(USIC0_C0_BRG,            0x40030014,__READ_WRITE ,__usic_brg_bits);
__IO_REG32_BIT(USIC0_C0_INPR,           0x40030018,__READ_WRITE ,__usic_inpr_bits);
__IO_REG32_BIT(USIC0_C0_DX0CR,          0x4003001C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C0_DX1CR,          0x40030020,__READ_WRITE ,__usic_dx1cr_bits);
__IO_REG32_BIT(USIC0_C0_DX2CR,          0x40030024,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C0_DX3CR,          0x40030028,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C0_DX4CR,          0x4003002C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C0_DX5CR,          0x40030030,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C0_SCTR,           0x40030034,__READ_WRITE ,__usic_sctr_bits);
__IO_REG32_BIT(USIC0_C0_TCSR,           0x40030038,__READ_WRITE ,__usic_tcsr_bits);
__IO_REG32_BIT(USIC0_C0_PCR,            0x4003003C,__READ_WRITE ,__usic_pcr_bits);
#define        USIC0_C0_PCR_ASC       USIC0_C0_PCR
#define        USIC0_C0_PCR_ASC_bit   USIC0_C0_PCR_bit
#define        USIC0_C0_PCR_SSC       USIC0_C0_PCR
#define        USIC0_C0_PCR_SSC_bit   USIC0_C0_PCR_bit.ssc
#define        USIC0_C0_PCR_IIC       USIC0_C0_PCR
#define        USIC0_C0_PCR_IIC_bit   USIC0_C0_PCR_bit.iic
#define        USIC0_C0_PCR_IIS       USIC0_C0_PCR
#define        USIC0_C0_PCR_IIS_bit   USIC0_C0_PCR_bit.iis
__IO_REG32_BIT(USIC0_C0_CCR,            0x40030040,__READ_WRITE ,__usic_ccr_bits);
__IO_REG32_BIT(USIC0_C0_CMTR,           0x40030044,__READ       ,__usic_cmtr_bits);
__IO_REG32_BIT(USIC0_C0_PSR,            0x40030048,__READ_WRITE ,__usic_psr_bits);
#define        USIC0_C0_PSR_ASC       USIC0_C0_PSR
#define        USIC0_C0_PSR_ASC_bit   USIC0_C0_PSR_bit
#define        USIC0_C0_PSR_SSC       USIC0_C0_PSR
#define        USIC0_C0_PSR_SSC_bit   USIC0_C0_PSR_bit.ssc
#define        USIC0_C0_PSR_IIC       USIC0_C0_PSR
#define        USIC0_C0_PSR_IIC_bit   USIC0_C0_PSR_bit.iic
#define        USIC0_C0_PSR_IIS       USIC0_C0_PSR
#define        USIC0_C0_PSR_IIS_bit   USIC0_C0_PSR_bit.iis
__IO_REG32_BIT(USIC0_C0_PSCR,           0x4003004C,__READ_WRITE ,__usic_pscr_bits);
__IO_REG32_BIT(USIC0_C0_RBUFSR,         0x40030050,__READ       ,__usic_rbufsr_bits);
__IO_REG32_BIT(USIC0_C0_RBUF,           0x40030054,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C0_RBUFD,          0x40030058,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C0_RBUF0,          0x4003005C,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C0_RBUF1,          0x40030060,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C0_RBUF01SR,       0x40030064,__READ       ,__usic_rbuf01sr_bits);
__IO_REG32_BIT(USIC0_C0_FMR,            0x40030068,__WRITE      ,__usic_fmr_bits);
__IO_REG32_BIT(USIC0_C0_TBUF0,          0x40030080,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF1,          0x40030084,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF2,          0x40030088,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF3,          0x4003008C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF4,          0x40030090,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF5,          0x40030094,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF6,          0x40030098,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF7,          0x4003009C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF8,          0x400300A0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF9,          0x400300A4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF10,         0x400300A8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF11,         0x400300AC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF12,         0x400300B0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF13,         0x400300B4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF14,         0x400300B8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF15,         0x400300BC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF16,         0x400300C0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF17,         0x400300C4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF18,         0x400300C8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF19,         0x400300CC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF20,         0x400300D0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF21,         0x400300D4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF22,         0x400300D8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF23,         0x400300DC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF24,         0x400300E0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF25,         0x400300E4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF26,         0x400300E8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF27,         0x400300EC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF28,         0x400300F0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF29,         0x400300F4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF30,         0x400300F8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_TBUF31,         0x400300FC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C0_BYP,            0x40030100,__READ_WRITE ,__usic_byp_bits);
__IO_REG32_BIT(USIC0_C0_BYPCR,          0x40030104,__READ_WRITE ,__usic_bypcr_bits);
__IO_REG32_BIT(USIC0_C0_TBCTR,          0x40030108,__READ_WRITE ,__usic_tbctr_bits);
__IO_REG32_BIT(USIC0_C0_RBCTR,          0x4003010C,__READ_WRITE ,__usic_rbctr_bits);
__IO_REG32_BIT(USIC0_C0_TRBPTR,         0x40030110,__READ       ,__usic_trbptr_bits);
__IO_REG32_BIT(USIC0_C0_TRBSR,          0x40030114,__READ_WRITE ,__usic_trbsr_bits);
__IO_REG32_BIT(USIC0_C0_TRBSCR,         0x40030118,__READ_WRITE ,__usic_trbscr_bits);
__IO_REG32_BIT(USIC0_C0_OUTR,           0x4003011C,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC0_C0_OUTDR,          0x40030120,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC0_C0_IN0,            0x40030180,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN1,            0x40030184,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN2,            0x40030188,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN3,            0x4003018C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN4,            0x40030190,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN5,            0x40030194,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN6,            0x40030198,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN7,            0x4003019C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN8,            0x400301A0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN9,            0x400301A4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN10,           0x400301A8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN11,           0x400301AC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN12,           0x400301B0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN13,           0x400301B4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN14,           0x400301B8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN15,           0x400301BC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN16,           0x400301C0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN17,           0x400301C4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN18,           0x400301C8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN19,           0x400301CC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN20,           0x400301D0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN21,           0x400301D4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN22,           0x400301D8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN23,           0x400301DC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN24,           0x400301E0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN25,           0x400301E4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN26,           0x400301E8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN27,           0x400301EC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN28,           0x400301F0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN29,           0x400301F4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN30,           0x400301F8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C0_IN31,           0x400301FC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_CCFG,           0x40030204,__READ       ,__usic_ccfg_bits);
__IO_REG32_BIT(USIC0_C1_KSCFG,          0x4003020C,__READ_WRITE ,__usic_kscfg_bits);
__IO_REG32_BIT(USIC0_C1_FDR,            0x40030210,__READ_WRITE ,__usic_fdr_bits);
__IO_REG32_BIT(USIC0_C1_BRG,            0x40030214,__READ_WRITE ,__usic_brg_bits);
__IO_REG32_BIT(USIC0_C1_INPR,           0x40030218,__READ_WRITE ,__usic_inpr_bits);
__IO_REG32_BIT(USIC0_C1_DX0CR,          0x4003021C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C1_DX1CR,          0x40030220,__READ_WRITE ,__usic_dx1cr_bits);
__IO_REG32_BIT(USIC0_C1_DX2CR,          0x40030224,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C1_DX3CR,          0x40030228,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C1_DX4CR,          0x4003022C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C1_DX5CR,          0x40030230,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC0_C1_SCTR,           0x40030234,__READ_WRITE ,__usic_sctr_bits);
__IO_REG32_BIT(USIC0_C1_TCSR,           0x40030238,__READ_WRITE ,__usic_tcsr_bits);
__IO_REG32_BIT(USIC0_C1_PCR,            0x4003023C,__READ_WRITE ,__usic_pcr_bits);
#define        USIC0_C1_PCR_ASC       USIC0_C1_PCR
#define        USIC0_C1_PCR_ASC_bit   USIC0_C1_PCR_bit
#define        USIC0_C1_PCR_SSC       USIC0_C1_PCR
#define        USIC0_C1_PCR_SSC_bit   USIC0_C1_PCR_bit.ssc
#define        USIC0_C1_PCR_IIC       USIC0_C1_PCR
#define        USIC0_C1_PCR_IIC_bit   USIC0_C1_PCR_bit.iic
#define        USIC0_C1_PCR_IIS       USIC0_C1_PCR
#define        USIC0_C1_PCR_IIS_bit   USIC0_C1_PCR_bit.iis
__IO_REG32_BIT(USIC0_C1_CCR,            0x40030240,__READ_WRITE ,__usic_ccr_bits);
__IO_REG32_BIT(USIC0_C1_CMTR,           0x40030244,__READ       ,__usic_cmtr_bits);
__IO_REG32_BIT(USIC0_C1_PSR,            0x40030248,__READ_WRITE ,__usic_psr_bits);
#define        USIC0_C1_PSR_ASC       USIC0_C1_PSR
#define        USIC0_C1_PSR_ASC_bit   USIC0_C1_PSR_bit
#define        USIC0_C1_PSR_SSC       USIC0_C1_PSR
#define        USIC0_C1_PSR_SSC_bit   USIC0_C1_PSR_bit.ssc
#define        USIC0_C1_PSR_IIC       USIC0_C1_PSR
#define        USIC0_C1_PSR_IIC_bit   USIC0_C1_PSR_bit.iic
#define        USIC0_C1_PSR_IIS       USIC0_C1_PSR
#define        USIC0_C1_PSR_IIS_bit   USIC0_C1_PSR_bit.iis
__IO_REG32_BIT(USIC0_C1_PSCR,           0x4003024C,__READ_WRITE ,__usic_pscr_bits);
__IO_REG32_BIT(USIC0_C1_RBUFSR,         0x40030250,__READ       ,__usic_rbufsr_bits);
__IO_REG32_BIT(USIC0_C1_RBUF,           0x40030254,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C1_RBUFD,          0x40030258,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C1_RBUF0,          0x4003025C,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C1_RBUF1,          0x40030260,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC0_C1_RBUF01SR,       0x40030264,__READ       ,__usic_rbuf01sr_bits);
__IO_REG32_BIT(USIC0_C1_FMR,            0x40030268,__WRITE      ,__usic_fmr_bits);
__IO_REG32_BIT(USIC0_C1_TBUF0,          0x40030280,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF1,          0x40030284,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF2,          0x40030288,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF3,          0x4003028C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF4,          0x40030290,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF5,          0x40030294,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF6,          0x40030298,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF7,          0x4003029C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF8,          0x400302A0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF9,          0x400302A4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF10,         0x400302A8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF11,         0x400302AC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF12,         0x400302B0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF13,         0x400302B4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF14,         0x400302B8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF15,         0x400302BC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF16,         0x400302C0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF17,         0x400302C4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF18,         0x400302C8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF19,         0x400302CC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF20,         0x400302D0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF21,         0x400302D4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF22,         0x400302D8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF23,         0x400302DC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF24,         0x400302E0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF25,         0x400302E4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF26,         0x400302E8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF27,         0x400302EC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF28,         0x400302F0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF29,         0x400302F4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF30,         0x400302F8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_TBUF31,         0x400302FC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC0_C1_BYP,            0x40030300,__READ_WRITE ,__usic_byp_bits);
__IO_REG32_BIT(USIC0_C1_BYPCR,          0x40030304,__READ_WRITE ,__usic_bypcr_bits);
__IO_REG32_BIT(USIC0_C1_TBCTR,          0x40030308,__READ_WRITE ,__usic_tbctr_bits);
__IO_REG32_BIT(USIC0_C1_RBCTR,          0x4003030C,__READ_WRITE ,__usic_rbctr_bits);
__IO_REG32_BIT(USIC0_C1_TRBPTR,         0x40030310,__READ       ,__usic_trbptr_bits);
__IO_REG32_BIT(USIC0_C1_TRBSR,          0x40030314,__READ_WRITE ,__usic_trbsr_bits);
__IO_REG32_BIT(USIC0_C1_TRBSCR,         0x40030318,__READ_WRITE ,__usic_trbscr_bits);
__IO_REG32_BIT(USIC0_C1_OUTR,           0x4003031C,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC0_C1_OUTDR,          0x40030320,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC0_C1_IN0,            0x40030380,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN1,            0x40030384,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN2,            0x40030388,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN3,            0x4003038C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN4,            0x40030390,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN5,            0x40030394,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN6,            0x40030398,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN7,            0x4003039C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN8,            0x400303A0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN9,            0x400303A4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN10,           0x400303A8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN11,           0x400303AC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN12,           0x400303B0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN13,           0x400303B4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN14,           0x400303B8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN15,           0x400303BC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN16,           0x400303C0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN17,           0x400303C4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN18,           0x400303C8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN19,           0x400303CC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN20,           0x400303D0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN21,           0x400303D4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN22,           0x400303D8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN23,           0x400303DC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN24,           0x400303E0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN25,           0x400303E4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN26,           0x400303E8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN27,           0x400303EC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN28,           0x400303F0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN29,           0x400303F4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN30,           0x400303F8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC0_C1_IN31,           0x400303FC,__READ       ,__usic_in_bits);
__IO_REG32(    USIC0_RAM_BASE,          0x40030400,__READ_WRITE );

/***************************************************************************
 **
 ** USIC1
 **
 ***************************************************************************/
__IO_REG32_BIT(USIC1_ID,                0x48020000,__READ_WRITE ,__usic_id_bits);
__IO_REG32_BIT(USIC1_C0_CCFG,           0x48020004,__READ       ,__usic_ccfg_bits);
__IO_REG32_BIT(USIC1_C0_KSCFG,          0x4802000C,__READ_WRITE ,__usic_kscfg_bits);
__IO_REG32_BIT(USIC1_C0_FDR,            0x48020010,__READ_WRITE ,__usic_fdr_bits);
__IO_REG32_BIT(USIC1_C0_BRG,            0x48020014,__READ_WRITE ,__usic_brg_bits);
__IO_REG32_BIT(USIC1_C0_INPR,           0x48020018,__READ_WRITE ,__usic_inpr_bits);
__IO_REG32_BIT(USIC1_C0_DX0CR,          0x4802001C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C0_DX1CR,          0x48020020,__READ_WRITE ,__usic_dx1cr_bits);
__IO_REG32_BIT(USIC1_C0_DX2CR,          0x48020024,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C0_DX3CR,          0x48020028,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C0_DX4CR,          0x4802002C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C0_DX5CR,          0x48020030,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C0_SCTR,           0x48020034,__READ_WRITE ,__usic_sctr_bits);
__IO_REG32_BIT(USIC1_C0_TCSR,           0x48020038,__READ_WRITE ,__usic_tcsr_bits);
__IO_REG32_BIT(USIC1_C0_PCR,            0x4802003C,__READ_WRITE ,__usic_pcr_bits);
#define        USIC1_C0_PCR_ASC       USIC1_C0_PCR
#define        USIC1_C0_PCR_ASC_bit   USIC1_C0_PCR_bit
#define        USIC1_C0_PCR_SSC       USIC1_C0_PCR
#define        USIC1_C0_PCR_SSC_bit   USIC1_C0_PCR_bit.ssc
#define        USIC1_C0_PCR_IIC       USIC1_C0_PCR
#define        USIC1_C0_PCR_IIC_bit   USIC1_C0_PCR_bit.iic
#define        USIC1_C0_PCR_IIS       USIC1_C0_PCR
#define        USIC1_C0_PCR_IIS_bit   USIC1_C0_PCR_bit.iis
__IO_REG32_BIT(USIC1_C0_CCR,            0x48020040,__READ_WRITE ,__usic_ccr_bits);
__IO_REG32_BIT(USIC1_C0_CMTR,           0x48020044,__READ       ,__usic_cmtr_bits);
__IO_REG32_BIT(USIC1_C0_PSR,            0x48020048,__READ_WRITE ,__usic_psr_bits);
#define        USIC1_C0_PSR_ASC       USIC1_C0_PSR
#define        USIC1_C0_PSR_ASC_bit   USIC1_C0_PSR_bit
#define        USIC1_C0_PSR_SSC       USIC1_C0_PSR
#define        USIC1_C0_PSR_SSC_bit   USIC1_C0_PSR_bit.ssc
#define        USIC1_C0_PSR_IIC       USIC1_C0_PSR
#define        USIC1_C0_PSR_IIC_bit   USIC1_C0_PSR_bit.iic
#define        USIC1_C0_PSR_IIS       USIC1_C0_PSR
#define        USIC1_C0_PSR_IIS_bit   USIC1_C0_PSR_bit.iis
__IO_REG32_BIT(USIC1_C0_PSCR,           0x4802004C,__READ_WRITE ,__usic_pscr_bits);
__IO_REG32_BIT(USIC1_C0_RBUFSR,         0x48020050,__READ       ,__usic_rbufsr_bits);
__IO_REG32_BIT(USIC1_C0_RBUF,           0x48020054,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C0_RBUFD,          0x48020058,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C0_RBUF0,          0x4802005C,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C0_RBUF1,          0x48020060,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C0_RBUF01SR,       0x48020064,__READ       ,__usic_rbuf01sr_bits);
__IO_REG32_BIT(USIC1_C0_FMR,            0x48020068,__WRITE      ,__usic_fmr_bits);
__IO_REG32_BIT(USIC1_C0_TBUF0,          0x48020080,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF1,          0x48020084,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF2,          0x48020088,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF3,          0x4802008C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF4,          0x48020090,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF5,          0x48020094,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF6,          0x48020098,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF7,          0x4802009C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF8,          0x480200A0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF9,          0x480200A4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF10,         0x480200A8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF11,         0x480200AC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF12,         0x480200B0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF13,         0x480200B4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF14,         0x480200B8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF15,         0x480200BC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF16,         0x480200C0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF17,         0x480200C4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF18,         0x480200C8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF19,         0x480200CC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF20,         0x480200D0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF21,         0x480200D4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF22,         0x480200D8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF23,         0x480200DC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF24,         0x480200E0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF25,         0x480200E4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF26,         0x480200E8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF27,         0x480200EC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF28,         0x480200F0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF29,         0x480200F4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF30,         0x480200F8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_TBUF31,         0x480200FC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C0_BYP,            0x48020100,__READ_WRITE ,__usic_byp_bits);
__IO_REG32_BIT(USIC1_C0_BYPCR,          0x48020104,__READ_WRITE ,__usic_bypcr_bits);
__IO_REG32_BIT(USIC1_C0_TBCTR,          0x48020108,__READ_WRITE ,__usic_tbctr_bits);
__IO_REG32_BIT(USIC1_C0_RBCTR,          0x4802010C,__READ_WRITE ,__usic_rbctr_bits);
__IO_REG32_BIT(USIC1_C0_TRBPTR,         0x48020110,__READ       ,__usic_trbptr_bits);
__IO_REG32_BIT(USIC1_C0_TRBSR,          0x48020114,__READ_WRITE ,__usic_trbsr_bits);
__IO_REG32_BIT(USIC1_C0_TRBSCR,         0x48020118,__READ_WRITE ,__usic_trbscr_bits);
__IO_REG32_BIT(USIC1_C0_OUTR,           0x4802011C,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC1_C0_OUTDR,          0x48020120,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC1_C0_IN0,            0x48020180,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN1,            0x48020184,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN2,            0x48020188,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN3,            0x4802018C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN4,            0x48020190,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN5,            0x48020194,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN6,            0x48020198,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN7,            0x4802019C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN8,            0x480201A0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN9,            0x480201A4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN10,           0x480201A8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN11,           0x480201AC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN12,           0x480201B0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN13,           0x480201B4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN14,           0x480201B8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN15,           0x480201BC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN16,           0x480201C0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN17,           0x480201C4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN18,           0x480201C8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN19,           0x480201CC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN20,           0x480201D0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN21,           0x480201D4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN22,           0x480201D8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN23,           0x480201DC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN24,           0x480201E0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN25,           0x480201E4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN26,           0x480201E8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN27,           0x480201EC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN28,           0x480201F0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN29,           0x480201F4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN30,           0x480201F8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C0_IN31,           0x480201FC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_CCFG,           0x48020204,__READ       ,__usic_ccfg_bits);
__IO_REG32_BIT(USIC1_C1_KSCFG,          0x4802020C,__READ_WRITE ,__usic_kscfg_bits);
__IO_REG32_BIT(USIC1_C1_FDR,            0x48020210,__READ_WRITE ,__usic_fdr_bits);
__IO_REG32_BIT(USIC1_C1_BRG,            0x48020214,__READ_WRITE ,__usic_brg_bits);
__IO_REG32_BIT(USIC1_C1_INPR,           0x48020218,__READ_WRITE ,__usic_inpr_bits);
__IO_REG32_BIT(USIC1_C1_DX0CR,          0x4802021C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C1_DX1CR,          0x48020220,__READ_WRITE ,__usic_dx1cr_bits);
__IO_REG32_BIT(USIC1_C1_DX2CR,          0x48020224,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C1_DX3CR,          0x48020228,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C1_DX4CR,          0x4802022C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C1_DX5CR,          0x48020230,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC1_C1_SCTR,           0x48020234,__READ_WRITE ,__usic_sctr_bits);
__IO_REG32_BIT(USIC1_C1_TCSR,           0x48020238,__READ_WRITE ,__usic_tcsr_bits);
__IO_REG32_BIT(USIC1_C1_PCR,            0x4802023C,__READ_WRITE ,__usic_pcr_bits);
#define        USIC1_C1_PCR_ASC       USIC1_C1_PCR
#define        USIC1_C1_PCR_ASC_bit   USIC1_C1_PCR_bit
#define        USIC1_C1_PCR_SSC       USIC1_C1_PCR
#define        USIC1_C1_PCR_SSC_bit   USIC1_C1_PCR_bit.ssc
#define        USIC1_C1_PCR_IIC       USIC1_C1_PCR
#define        USIC1_C1_PCR_IIC_bit   USIC1_C1_PCR_bit.iic
#define        USIC1_C1_PCR_IIS       USIC1_C1_PCR
#define        USIC1_C1_PCR_IIS_bit   USIC1_C1_PCR_bit.iis
__IO_REG32_BIT(USIC1_C1_CCR,            0x48020240,__READ_WRITE ,__usic_ccr_bits);
__IO_REG32_BIT(USIC1_C1_CMTR,           0x48020244,__READ       ,__usic_cmtr_bits);
__IO_REG32_BIT(USIC1_C1_PSR,            0x48020248,__READ_WRITE ,__usic_psr_bits);
#define        USIC1_C1_PSR_ASC       USIC1_C1_PSR
#define        USIC1_C1_PSR_ASC_bit   USIC1_C1_PSR_bit
#define        USIC1_C1_PSR_SSC       USIC1_C1_PSR
#define        USIC1_C1_PSR_SSC_bit   USIC1_C1_PSR_bit.ssc
#define        USIC1_C1_PSR_IIC       USIC1_C1_PSR
#define        USIC1_C1_PSR_IIC_bit   USIC1_C1_PSR_bit.iic
#define        USIC1_C1_PSR_IIS       USIC1_C1_PSR
#define        USIC1_C1_PSR_IIS_bit   USIC1_C1_PSR_bit.iis
__IO_REG32_BIT(USIC1_C1_PSCR,           0x4802024C,__READ_WRITE ,__usic_pscr_bits);
__IO_REG32_BIT(USIC1_C1_RBUFSR,         0x48020250,__READ       ,__usic_rbufsr_bits);
__IO_REG32_BIT(USIC1_C1_RBUF,           0x48020254,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C1_RBUFD,          0x48020258,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C1_RBUF0,          0x4802025C,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C1_RBUF1,          0x48020260,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC1_C1_RBUF01SR,       0x48020264,__READ       ,__usic_rbuf01sr_bits);
__IO_REG32_BIT(USIC1_C1_FMR,            0x48020268,__WRITE      ,__usic_fmr_bits);
__IO_REG32_BIT(USIC1_C1_TBUF0,          0x48020280,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF1,          0x48020284,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF2,          0x48020288,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF3,          0x4802028C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF4,          0x48020290,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF5,          0x48020294,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF6,          0x48020298,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF7,          0x4802029C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF8,          0x480202A0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF9,          0x480202A4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF10,         0x480202A8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF11,         0x480202AC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF12,         0x480202B0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF13,         0x480202B4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF14,         0x480202B8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF15,         0x480202BC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF16,         0x480202C0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF17,         0x480202C4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF18,         0x480202C8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF19,         0x480202CC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF20,         0x480202D0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF21,         0x480202D4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF22,         0x480202D8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF23,         0x480202DC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF24,         0x480202E0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF25,         0x480202E4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF26,         0x480202E8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF27,         0x480202EC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF28,         0x480202F0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF29,         0x480202F4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF30,         0x480202F8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_TBUF31,         0x480202FC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC1_C1_BYP,            0x48020300,__READ_WRITE ,__usic_byp_bits);
__IO_REG32_BIT(USIC1_C1_BYPCR,          0x48020304,__READ_WRITE ,__usic_bypcr_bits);
__IO_REG32_BIT(USIC1_C1_TBCTR,          0x48020308,__READ_WRITE ,__usic_tbctr_bits);
__IO_REG32_BIT(USIC1_C1_RBCTR,          0x4802030C,__READ_WRITE ,__usic_rbctr_bits);
__IO_REG32_BIT(USIC1_C1_TRBPTR,         0x48020310,__READ       ,__usic_trbptr_bits);
__IO_REG32_BIT(USIC1_C1_TRBSR,          0x48020314,__READ_WRITE ,__usic_trbsr_bits);
__IO_REG32_BIT(USIC1_C1_TRBSCR,         0x48020318,__READ_WRITE ,__usic_trbscr_bits);
__IO_REG32_BIT(USIC1_C1_OUTR,           0x4802031C,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC1_C1_OUTDR,          0x48020320,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC1_C1_IN0,            0x48020380,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN1,            0x48020384,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN2,            0x48020388,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN3,            0x4802038C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN4,            0x48020390,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN5,            0x48020394,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN6,            0x48020398,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN7,            0x4802039C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN8,            0x480203A0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN9,            0x480203A4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN10,           0x480203A8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN11,           0x480203AC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN12,           0x480203B0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN13,           0x480203B4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN14,           0x480203B8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN15,           0x480203BC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN16,           0x480203C0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN17,           0x480203C4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN18,           0x480203C8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN19,           0x480203CC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN20,           0x480203D0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN21,           0x480203D4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN22,           0x480203D8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN23,           0x480203DC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN24,           0x480203E0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN25,           0x480203E4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN26,           0x480203E8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN27,           0x480203EC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN28,           0x480203F0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN29,           0x480203F4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN30,           0x480203F8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC1_C1_IN31,           0x480203FC,__READ       ,__usic_in_bits);
__IO_REG32(    USIC1_RAM_BASE,          0x48020400,__READ_WRITE );

/***************************************************************************
 **
 ** USIC2
 **
 ***************************************************************************/
__IO_REG32_BIT(USIC2_ID,                0x48024000,__READ_WRITE ,__usic_id_bits);
__IO_REG32_BIT(USIC2_C0_CCFG,           0x48024004,__READ       ,__usic_ccfg_bits);
__IO_REG32_BIT(USIC2_C0_KSCFG,          0x4802400C,__READ_WRITE ,__usic_kscfg_bits);
__IO_REG32_BIT(USIC2_C0_FDR,            0x48024010,__READ_WRITE ,__usic_fdr_bits);
__IO_REG32_BIT(USIC2_C0_BRG,            0x48024014,__READ_WRITE ,__usic_brg_bits);
__IO_REG32_BIT(USIC2_C0_INPR,           0x48024018,__READ_WRITE ,__usic_inpr_bits);
__IO_REG32_BIT(USIC2_C0_DX0CR,          0x4802401C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C0_DX1CR,          0x48024020,__READ_WRITE ,__usic_dx1cr_bits);
__IO_REG32_BIT(USIC2_C0_DX2CR,          0x48024024,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C0_DX3CR,          0x48024028,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C0_DX4CR,          0x4802402C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C0_DX5CR,          0x48024030,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C0_SCTR,           0x48024034,__READ_WRITE ,__usic_sctr_bits);
__IO_REG32_BIT(USIC2_C0_TCSR,           0x48024038,__READ_WRITE ,__usic_tcsr_bits);
__IO_REG32_BIT(USIC2_C0_PCR,            0x4802403C,__READ_WRITE ,__usic_pcr_bits);
#define        USIC2_C0_PCR_ASC       USIC4_C0_PCR
#define        USIC2_C0_PCR_ASC_bit   USIC4_C0_PCR_bit
#define        USIC2_C0_PCR_SSC       USIC4_C0_PCR
#define        USIC2_C0_PCR_SSC_bit   USIC4_C0_PCR_bit.ssc
#define        USIC2_C0_PCR_IIC       USIC4_C0_PCR
#define        USIC2_C0_PCR_IIC_bit   USIC4_C0_PCR_bit.iic
#define        USIC2_C0_PCR_IIS       USIC4_C0_PCR
#define        USIC2_C0_PCR_IIS_bit   USIC4_C0_PCR_bit.iis
__IO_REG32_BIT(USIC2_C0_CCR,            0x48024040,__READ_WRITE ,__usic_ccr_bits);
__IO_REG32_BIT(USIC2_C0_CMTR,           0x48024044,__READ       ,__usic_cmtr_bits);
__IO_REG32_BIT(USIC2_C0_PSR,            0x48024048,__READ_WRITE ,__usic_psr_bits);
#define        USIC2_C0_PSR_ASC       USIC2_C0_PSR
#define        USIC2_C0_PSR_ASC_bit   USIC2_C0_PSR_bit
#define        USIC2_C0_PSR_SSC       USIC2_C0_PSR
#define        USIC2_C0_PSR_SSC_bit   USIC2_C0_PSR_bit.ssc
#define        USIC2_C0_PSR_IIC       USIC2_C0_PSR
#define        USIC2_C0_PSR_IIC_bit   USIC2_C0_PSR_bit.iic
#define        USIC2_C0_PSR_IIS       USIC2_C0_PSR
#define        USIC2_C0_PSR_IIS_bit   USIC2_C0_PSR_bit.iis
__IO_REG32_BIT(USIC2_C0_PSCR,           0x4802404C,__READ_WRITE ,__usic_pscr_bits);
__IO_REG32_BIT(USIC2_C0_RBUFSR,         0x48024050,__READ       ,__usic_rbufsr_bits);
__IO_REG32_BIT(USIC2_C0_RBUF,           0x48024054,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C0_RBUFD,          0x48024058,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C0_RBUF0,          0x4802405C,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C0_RBUF1,          0x48024060,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C0_RBUF01SR,       0x48024064,__READ       ,__usic_rbuf01sr_bits);
__IO_REG32_BIT(USIC2_C0_FMR,            0x48024068,__WRITE      ,__usic_fmr_bits);
__IO_REG32_BIT(USIC2_C0_TBUF0,          0x48024080,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF1,          0x48024084,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF2,          0x48024088,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF3,          0x4802408C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF4,          0x48024090,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF5,          0x48024094,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF6,          0x48024098,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF7,          0x4802409C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF8,          0x480240A0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF9,          0x480240A4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF10,         0x480240A8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF11,         0x480240AC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF12,         0x480240B0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF13,         0x480240B4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF14,         0x480240B8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF15,         0x480240BC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF16,         0x480240C0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF17,         0x480240C4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF18,         0x480240C8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF19,         0x480240CC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF20,         0x480240D0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF21,         0x480240D4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF22,         0x480240D8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF23,         0x480240DC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF24,         0x480240E0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF25,         0x480240E4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF26,         0x480240E8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF27,         0x480240EC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF28,         0x480240F0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF29,         0x480240F4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF30,         0x480240F8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_TBUF31,         0x480240FC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C0_BYP,            0x48024100,__READ_WRITE ,__usic_byp_bits);
__IO_REG32_BIT(USIC2_C0_BYPCR,          0x48024104,__READ_WRITE ,__usic_bypcr_bits);
__IO_REG32_BIT(USIC2_C0_TBCTR,          0x48024108,__READ_WRITE ,__usic_tbctr_bits);
__IO_REG32_BIT(USIC2_C0_RBCTR,          0x4802410C,__READ_WRITE ,__usic_rbctr_bits);
__IO_REG32_BIT(USIC2_C0_TRBPTR,         0x48024110,__READ       ,__usic_trbptr_bits);
__IO_REG32_BIT(USIC2_C0_TRBSR,          0x48024114,__READ_WRITE ,__usic_trbsr_bits);
__IO_REG32_BIT(USIC2_C0_TRBSCR,         0x48024118,__READ_WRITE ,__usic_trbscr_bits);
__IO_REG32_BIT(USIC2_C0_OUTR,           0x4802411C,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC2_C0_OUTDR,          0x48024120,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC2_C0_IN0,            0x48024180,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN1,            0x48024184,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN2,            0x48024188,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN3,            0x4802418C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN4,            0x48024190,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN5,            0x48024194,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN6,            0x48024198,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN7,            0x4802419C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN8,            0x480241A0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN9,            0x480241A4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN10,           0x480241A8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN11,           0x480241AC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN12,           0x480241B0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN13,           0x480241B4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN14,           0x480241B8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN15,           0x480241BC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN16,           0x480241C0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN17,           0x480241C4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN18,           0x480241C8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN19,           0x480241CC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN20,           0x480241D0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN21,           0x480241D4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN22,           0x480241D8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN23,           0x480241DC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN24,           0x480241E0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN25,           0x480241E4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN26,           0x480241E8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN27,           0x480241EC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN28,           0x480241F0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN29,           0x480241F4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN30,           0x480241F8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C0_IN31,           0x480241FC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_CCFG,           0x48024204,__READ       ,__usic_ccfg_bits);
__IO_REG32_BIT(USIC2_C1_KSCFG,          0x4802420C,__READ_WRITE ,__usic_kscfg_bits);
__IO_REG32_BIT(USIC2_C1_FDR,            0x48024210,__READ_WRITE ,__usic_fdr_bits);
__IO_REG32_BIT(USIC2_C1_BRG,            0x48024214,__READ_WRITE ,__usic_brg_bits);
__IO_REG32_BIT(USIC2_C1_INPR,           0x48024218,__READ_WRITE ,__usic_inpr_bits);
__IO_REG32_BIT(USIC2_C1_DX0CR,          0x4802421C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C1_DX1CR,          0x48024220,__READ_WRITE ,__usic_dx1cr_bits);
__IO_REG32_BIT(USIC2_C1_DX2CR,          0x48024224,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C1_DX3CR,          0x48024228,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C1_DX4CR,          0x4802422C,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C1_DX5CR,          0x48024230,__READ_WRITE ,__usic_dxcr_bits);
__IO_REG32_BIT(USIC2_C1_SCTR,           0x48024234,__READ_WRITE ,__usic_sctr_bits);
__IO_REG32_BIT(USIC2_C1_TCSR,           0x48024238,__READ_WRITE ,__usic_tcsr_bits);
__IO_REG32_BIT(USIC2_C1_PCR,            0x4802423C,__READ_WRITE ,__usic_pcr_bits);
#define        USIC2_C1_PCR_ASC       USIC2_C1_PCR
#define        USIC2_C1_PCR_ASC_bit   USIC2_C1_PCR_bit
#define        USIC2_C1_PCR_SSC       USIC2_C1_PCR
#define        USIC2_C1_PCR_SSC_bit   USIC2_C1_PCR_bit.ssc
#define        USIC2_C1_PCR_IIC       USIC2_C1_PCR
#define        USIC2_C1_PCR_IIC_bit   USIC2_C1_PCR_bit.iic
#define        USIC2_C1_PCR_IIS       USIC2_C1_PCR
#define        USIC2_C1_PCR_IIS_bit   USIC2_C1_PCR_bit.iis
__IO_REG32_BIT(USIC2_C1_CCR,            0x48024240,__READ_WRITE ,__usic_ccr_bits);
__IO_REG32_BIT(USIC2_C1_CMTR,           0x48024244,__READ       ,__usic_cmtr_bits);
__IO_REG32_BIT(USIC2_C1_PSR,            0x48024248,__READ_WRITE ,__usic_psr_bits);
#define        USIC2_C1_PSR_ASC       USIC2_C1_PSR
#define        USIC2_C1_PSR_ASC_bit   USIC2_C1_PSR_bit
#define        USIC2_C1_PSR_SSC       USIC2_C1_PSR
#define        USIC2_C1_PSR_SSC_bit   USIC2_C1_PSR_bit.ssc
#define        USIC2_C1_PSR_IIC       USIC2_C1_PSR
#define        USIC2_C1_PSR_IIC_bit   USIC2_C1_PSR_bit.iic
#define        USIC2_C1_PSR_IIS       USIC2_C1_PSR
#define        USIC2_C1_PSR_IIS_bit   USIC2_C1_PSR_bit.iis
__IO_REG32_BIT(USIC2_C1_PSCR,           0x4802424C,__READ_WRITE ,__usic_pscr_bits);
__IO_REG32_BIT(USIC2_C1_RBUFSR,         0x48024250,__READ       ,__usic_rbufsr_bits);
__IO_REG32_BIT(USIC2_C1_RBUF,           0x48024254,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C1_RBUFD,          0x48024258,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C1_RBUF0,          0x4802425C,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C1_RBUF1,          0x48024260,__READ       ,__usic_rbuf_bits);
__IO_REG32_BIT(USIC2_C1_RBUF01SR,       0x48024264,__READ       ,__usic_rbuf01sr_bits);
__IO_REG32_BIT(USIC2_C1_FMR,            0x48024268,__WRITE      ,__usic_fmr_bits);
__IO_REG32_BIT(USIC2_C1_TBUF0,          0x48024280,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF1,          0x48024284,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF2,          0x48024288,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF3,          0x4802428C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF4,          0x48024290,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF5,          0x48024294,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF6,          0x48024298,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF7,          0x4802429C,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF8,          0x480242A0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF9,          0x480242A4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF10,         0x480242A8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF11,         0x480242AC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF12,         0x480242B0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF13,         0x480242B4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF14,         0x480242B8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF15,         0x480242BC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF16,         0x480242C0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF17,         0x480242C4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF18,         0x480242C8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF19,         0x480242CC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF20,         0x480242D0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF21,         0x480242D4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF22,         0x480242D8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF23,         0x480242DC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF24,         0x480242E0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF25,         0x480242E4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF26,         0x480242E8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF27,         0x480242EC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF28,         0x480242F0,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF29,         0x480242F4,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF30,         0x480242F8,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_TBUF31,         0x480242FC,__READ_WRITE ,__usic_tbuf_bits);
__IO_REG32_BIT(USIC2_C1_BYP,            0x48024300,__READ_WRITE ,__usic_byp_bits);
__IO_REG32_BIT(USIC2_C1_BYPCR,          0x48024304,__READ_WRITE ,__usic_bypcr_bits);
__IO_REG32_BIT(USIC2_C1_TBCTR,          0x48024308,__READ_WRITE ,__usic_tbctr_bits);
__IO_REG32_BIT(USIC2_C1_RBCTR,          0x4802430C,__READ_WRITE ,__usic_rbctr_bits);
__IO_REG32_BIT(USIC2_C1_TRBPTR,         0x48024310,__READ       ,__usic_trbptr_bits);
__IO_REG32_BIT(USIC2_C1_TRBSR,          0x48024314,__READ_WRITE ,__usic_trbsr_bits);
__IO_REG32_BIT(USIC2_C1_TRBSCR,         0x48024318,__READ_WRITE ,__usic_trbscr_bits);
__IO_REG32_BIT(USIC2_C1_OUTR,           0x4802431C,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC2_C1_OUTDR,          0x48024320,__READ       ,__usic_outr_bits);
__IO_REG32_BIT(USIC2_C1_IN0,            0x48024380,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN1,            0x48024384,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN2,            0x48024388,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN3,            0x4802438C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN4,            0x48024390,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN5,            0x48024394,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN6,            0x48024398,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN7,            0x4802439C,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN8,            0x480243A0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN9,            0x480243A4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN10,           0x480243A8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN11,           0x480243AC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN12,           0x480243B0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN13,           0x480243B4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN14,           0x480243B8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN15,           0x480243BC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN16,           0x480243C0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN17,           0x480243C4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN18,           0x480243C8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN19,           0x480243CC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN20,           0x480243D0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN21,           0x480243D4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN22,           0x480243D8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN23,           0x480243DC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN24,           0x480243E0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN25,           0x480243E4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN26,           0x480243E8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN27,           0x480243EC,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN28,           0x480243F0,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN29,           0x480243F4,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN30,           0x480243F8,__READ       ,__usic_in_bits);
__IO_REG32_BIT(USIC2_C1_IN31,           0x480243FC,__READ       ,__usic_in_bits);
__IO_REG32(    USIC2_RAM_BASE,          0x48024400,__READ_WRITE );

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN_CLC,                 0x48014000,__READ_WRITE ,__can_clc_bits);
__IO_REG32_BIT(CAN_ID,                  0x48014008,__READ_WRITE ,__can_id_bits);
__IO_REG32_BIT(CAN_FDR,                 0x4801400C,__READ_WRITE ,__can_fdr_bits);
__IO_REG32_BIT(CAN_LIST0,               0x48014100,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST1,               0x48014104,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST2,               0x48014108,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST3,               0x4801410C,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST4,               0x48014110,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST5,               0x48014114,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST6,               0x48014118,__READ_WRITE ,__can_list_bits);
__IO_REG32_BIT(CAN_LIST7,               0x4801411C,__READ_WRITE ,__can_list_bits);
__IO_REG32(    CAN_MSPND0,              0x48014140,__READ_WRITE );
__IO_REG32(    CAN_MSPND1,              0x48014144,__READ_WRITE );
__IO_REG32(    CAN_MSPND2,              0x48014148,__READ_WRITE );
__IO_REG32(    CAN_MSPND3,              0x4801414C,__READ_WRITE );
__IO_REG32(    CAN_MSPND4,              0x48014150,__READ_WRITE );
__IO_REG32(    CAN_MSPND5,              0x48014154,__READ_WRITE );
__IO_REG32(    CAN_MSPND6,              0x48014158,__READ_WRITE );
__IO_REG32(    CAN_MSPND7,              0x4801415C,__READ_WRITE );
__IO_REG32_BIT(CAN_MSID0,               0x48014180,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID1,               0x48014184,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID2,               0x48014188,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID3,               0x4801418C,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID4,               0x48014190,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID5,               0x48014194,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID6,               0x48014198,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSID7,               0x4801419C,__READ       ,__can_msid_bits);
__IO_REG32_BIT(CAN_MSIMASK,             0x480141C0,__READ_WRITE ,__can_msimask_bits);
__IO_REG32_BIT(CAN_PANCTR,              0x480141C4,__READ_WRITE ,__can_panctr_bits);
__IO_REG32_BIT(CAN_MCR,                 0x480141C8,__READ_WRITE ,__can_mcr_bits);
__IO_REG32_BIT(CAN_MITR,                0x480141CC,__WRITE      ,__can_mitr_bits);

/***************************************************************************
 **
 ** CAN Node
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN_NCR0,                0x48014200,__READ_WRITE ,__can_ncr_bits);
__IO_REG32_BIT(CAN_NSR0,                0x48014204,__READ_WRITE ,__can_nsr_bits);
__IO_REG32_BIT(CAN_NIPR0,               0x48014208,__READ_WRITE ,__can_nipr_bits);
__IO_REG32_BIT(CAN_NPCR0,               0x4801420C,__READ_WRITE ,__can_npcr_bits);
__IO_REG32_BIT(CAN_NBTR0,               0x48014210,__READ_WRITE ,__can_nbtr_bits);
__IO_REG32_BIT(CAN_NECNT0,              0x48014214,__READ_WRITE ,__can_necnt_bits);
__IO_REG32_BIT(CAN_NFCR0,               0x48014218,__READ_WRITE ,__can_nfcr_bits);
__IO_REG32_BIT(CAN_NCR1,                0x48014300,__READ_WRITE ,__can_ncr_bits);
__IO_REG32_BIT(CAN_NSR1,                0x48014304,__READ_WRITE ,__can_nsr_bits);
__IO_REG32_BIT(CAN_NIPR1,               0x48014308,__READ_WRITE ,__can_nipr_bits);
__IO_REG32_BIT(CAN_NPCR1,               0x4801430C,__READ_WRITE ,__can_npcr_bits);
__IO_REG32_BIT(CAN_NBTR1,               0x48014310,__READ_WRITE ,__can_nbtr_bits);
__IO_REG32_BIT(CAN_NECNT1,              0x48014314,__READ_WRITE ,__can_necnt_bits);
__IO_REG32_BIT(CAN_NFCR1,               0x48014318,__READ_WRITE ,__can_nfcr_bits);
__IO_REG32_BIT(CAN_NCR2,                0x48014400,__READ_WRITE ,__can_ncr_bits);
__IO_REG32_BIT(CAN_NSR2,                0x48014404,__READ_WRITE ,__can_nsr_bits);
__IO_REG32_BIT(CAN_NIPR2,               0x48014408,__READ_WRITE ,__can_nipr_bits);
__IO_REG32_BIT(CAN_NPCR2,               0x4801440C,__READ_WRITE ,__can_npcr_bits);
__IO_REG32_BIT(CAN_NBTR2,               0x48014410,__READ_WRITE ,__can_nbtr_bits);
__IO_REG32_BIT(CAN_NECNT2,              0x48014414,__READ_WRITE ,__can_necnt_bits);
__IO_REG32_BIT(CAN_NFCR2,               0x48014418,__READ_WRITE ,__can_nfcr_bits);

/***************************************************************************
 **
 ** CAN Mess
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN_MOFCR0,              0x48015000,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR0,             0x48015004,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR0,              0x48015008,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR0,              0x4801500C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL0,            0x48015010,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH0,            0x48015014,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR0,               0x48015018,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR0,              0x4801501C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT0        CAN_MOCTR0
#define        CAN_MOSTAT0_bit    CAN_MOCTR0_bit
__IO_REG32_BIT(CAN_MOFCR1,              0x48015020,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR1,             0x48015024,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR1,              0x48015028,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR1,              0x4801502C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL1,            0x48015030,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH1,            0x48015034,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR1,               0x48015038,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR1,              0x4801503C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT1        CAN_MOCTR1
#define        CAN_MOSTAT1_bit    CAN_MOCTR1_bit
__IO_REG32_BIT(CAN_MOFCR2,              0x48015040,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR2,             0x48015044,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR2,              0x48015048,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR2,              0x4801504C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL2,            0x48015050,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH2,            0x48015054,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR2,               0x48015058,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR2,              0x4801505C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT2        CAN_MOCTR2
#define        CAN_MOSTAT2_bit    CAN_MOCTR2_bit
__IO_REG32_BIT(CAN_MOFCR3,              0x48015060,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR3,             0x48015064,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR3,              0x48015068,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR3,              0x4801506C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL3,            0x48015070,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH3,            0x48015074,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR3,               0x48015078,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR3,              0x4801507C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT3        CAN_MOCTR3
#define        CAN_MOSTAT3_bit    CAN_MOCTR3_bit
__IO_REG32_BIT(CAN_MOFCR4,              0x48015080,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR4,             0x48015084,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR4,              0x48015088,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR4,              0x4801508C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL4,            0x48015090,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH4,            0x48015094,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR4,               0x48015098,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR4,              0x4801509C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT4        CAN_MOCTR4
#define        CAN_MOSTAT4_bit    CAN_MOCTR4_bit
__IO_REG32_BIT(CAN_MOFCR5,              0x480150A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR5,             0x480150A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR5,              0x480150A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR5,              0x480150AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL5,            0x480150B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH5,            0x480150B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR5,               0x480150B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR5,              0x480150BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT5        CAN_MOCTR5
#define        CAN_MOSTAT5_bit    CAN_MOCTR5_bit
__IO_REG32_BIT(CAN_MOFCR6,              0x480150C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR6,             0x480150C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR6,              0x480150C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR6,              0x480150CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL6,            0x480150D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH6,            0x480150D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR6,               0x480150D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR6,              0x480150DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT6        CAN_MOCTR6
#define        CAN_MOSTAT6_bit    CAN_MOCTR6_bit
__IO_REG32_BIT(CAN_MOFCR7,              0x480150E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR7,             0x480150E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR7,              0x480150E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR7,              0x480150EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL7,            0x480150F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH7,            0x480150F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR7,               0x480150F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR7,              0x480150FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT7        CAN_MOCTR7
#define        CAN_MOSTAT7_bit    CAN_MOCTR7_bit
__IO_REG32_BIT(CAN_MOFCR8,              0x48015100,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR8,             0x48015104,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR8,              0x48015108,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR8,              0x4801510C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL8,            0x48015110,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH8,            0x48015114,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR8,               0x48015118,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR8,              0x4801511C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT8        CAN_MOCTR8
#define        CAN_MOSTAT8_bit    CAN_MOCTR8_bit
__IO_REG32_BIT(CAN_MOFCR9,              0x48015120,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR9,             0x48015124,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR9,              0x48015128,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR9,              0x4801512C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL9,            0x48015130,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH9,            0x48015134,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR9,               0x48015138,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR9,              0x4801513C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT9        CAN_MOCTR9
#define        CAN_MOSTAT9_bit    CAN_MOCTR9_bit
__IO_REG32_BIT(CAN_MOFCR10,             0x48015140,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR10,            0x48015144,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR10,             0x48015148,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR10,             0x4801514C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL10,           0x48015150,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH10,           0x48015154,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR10,              0x48015158,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR10,             0x4801515C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT10       CAN_MOCTR10
#define        CAN_MOSTAT10_bit   CAN_MOCTR10_bit
__IO_REG32_BIT(CAN_MOFCR11,             0x48015160,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR11,            0x48015164,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR11,             0x48015168,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR11,             0x4801516C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL11,           0x48015170,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH11,           0x48015174,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR11,              0x48015178,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR11,             0x4801517C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT11       CAN_MOCTR11
#define        CAN_MOSTAT11_bit   CAN_MOCTR11_bit
__IO_REG32_BIT(CAN_MOFCR12,             0x48015180,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR12,            0x48015184,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR12,             0x48015188,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR12,             0x4801518C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL12,           0x48015190,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH12,           0x48015194,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR12,              0x48015198,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR12,             0x4801519C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT12       CAN_MOCTR12
#define        CAN_MOSTAT12_bit   CAN_MOCTR12_bit
__IO_REG32_BIT(CAN_MOFCR13,             0x480151A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR13,            0x480151A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR13,             0x480151A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR13,             0x480151AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL13,           0x480151B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH13,           0x480151B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR13,              0x480151B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR13,             0x480151BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT13       CAN_MOCTR13
#define        CAN_MOSTAT13_bit   CAN_MOCTR13_bit
__IO_REG32_BIT(CAN_MOFCR14,             0x480151C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR14,            0x480151C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR14,             0x480151C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR14,             0x480151CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL14,           0x480151D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH14,           0x480151D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR14,              0x480151D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR14,             0x480151DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT14       CAN_MOCTR14
#define        CAN_MOSTAT14_bit   CAN_MOCTR14_bit
__IO_REG32_BIT(CAN_MOFCR15,             0x480151E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR15,            0x480151E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR15,             0x480151E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR15,             0x480151EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL15,           0x480151F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH15,           0x480151F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR15,              0x480151F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR15,             0x480151FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT15       CAN_MOCTR15
#define        CAN_MOSTAT15_bit   CAN_MOCTR15_bit
__IO_REG32_BIT(CAN_MOFCR16,             0x48015200,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR16,            0x48015204,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR16,             0x48015208,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR16,             0x4801520C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL16,           0x48015210,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH16,           0x48015214,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR16,              0x48015218,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR16,             0x4801521C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT16       CAN_MOCTR16
#define        CAN_MOSTAT16_bit   CAN_MOCTR16_bit
__IO_REG32_BIT(CAN_MOFCR17,             0x48015220,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR17,            0x48015224,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR17,             0x48015228,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR17,             0x4801522C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL17,           0x48015230,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH17,           0x48015234,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR17,              0x48015238,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR17,             0x4801523C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT17       CAN_MOCTR17
#define        CAN_MOSTAT17_bit   CAN_MOCTR17_bit
__IO_REG32_BIT(CAN_MOFCR18,             0x48015240,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR18,            0x48015244,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR18,             0x48015248,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR18,             0x4801524C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL18,           0x48015250,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH18,           0x48015254,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR18,              0x48015258,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR18,             0x4801525C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT18       CAN_MOCTR18
#define        CAN_MOSTAT18_bit   CAN_MOCTR18_bit
__IO_REG32_BIT(CAN_MOFCR19,             0x48015260,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR19,            0x48015264,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR19,             0x48015268,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR19,             0x4801526C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL19,           0x48015270,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH19,           0x48015274,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR19,              0x48015278,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR19,             0x4801527C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT19       CAN_MOCTR19
#define        CAN_MOSTAT19_bit   CAN_MOCTR19_bit
__IO_REG32_BIT(CAN_MOFCR20,             0x48015280,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR20,            0x48015284,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR20,             0x48015288,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR20,             0x4801528C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL20,           0x48015290,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH20,           0x48015294,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR20,              0x48015298,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR20,             0x4801529C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT20       CAN_MOCTR20
#define        CAN_MOSTAT20_bit   CAN_MOCTR20_bit
__IO_REG32_BIT(CAN_MOFCR21,             0x480152A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR21,            0x480152A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR21,             0x480152A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR21,             0x480152AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL21,           0x480152B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH21,           0x480152B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR21,              0x480152B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR21,             0x480152BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT21       CAN_MOCTR21
#define        CAN_MOSTAT21_bit   CAN_MOCTR21_bit
__IO_REG32_BIT(CAN_MOFCR22,             0x480152C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR22,            0x480152C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR22,             0x480152C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR22,             0x480152CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL22,           0x480152D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH22,           0x480152D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR22,              0x480152D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR22,             0x480152DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT22       CAN_MOCTR22
#define        CAN_MOSTAT22_bit   CAN_MOCTR22_bit
__IO_REG32_BIT(CAN_MOFCR23,             0x480152E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR23,            0x480152E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR23,             0x480152E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR23,             0x480152EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL23,           0x480152F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH23,           0x480152F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR23,              0x480152F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR23,             0x480152FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT23       CAN_MOCTR23
#define        CAN_MOSTAT23_bit   CAN_MOCTR23_bit
__IO_REG32_BIT(CAN_MOFCR24,             0x48015300,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR24,            0x48015304,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR24,             0x48015308,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR24,             0x4801530C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL24,           0x48015310,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH24,           0x48015314,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR24,              0x48015318,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR24,             0x4801531C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT24       CAN_MOCTR24
#define        CAN_MOSTAT24_bit   CAN_MOCTR24_bit
__IO_REG32_BIT(CAN_MOFCR25,             0x48015320,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR25,            0x48015324,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR25,             0x48015328,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR25,             0x4801532C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL25,           0x48015330,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH25,           0x48015334,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR25,              0x48015338,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR25,             0x4801533C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT25       CAN_MOCTR25
#define        CAN_MOSTAT25_bit   CAN_MOCTR25_bit
__IO_REG32_BIT(CAN_MOFCR26,             0x48015340,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR26,            0x48015344,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR26,             0x48015348,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR26,             0x4801534C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL26,           0x48015350,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH26,           0x48015354,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR26,              0x48015358,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR26,             0x4801535C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT26       CAN_MOCTR26
#define        CAN_MOSTAT26_bit   CAN_MOCTR26_bit
__IO_REG32_BIT(CAN_MOFCR27,             0x48015360,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR27,            0x48015364,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR27,             0x48015368,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR27,             0x4801536C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL27,           0x48015370,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH27,           0x48015374,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR27,              0x48015378,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR27,             0x4801537C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT27       CAN_MOCTR27
#define        CAN_MOSTAT27_bit   CAN_MOCTR27_bit
__IO_REG32_BIT(CAN_MOFCR28,             0x48015380,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR28,            0x48015384,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR28,             0x48015388,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR28,             0x4801538C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL28,           0x48015390,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH28,           0x48015394,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR28,              0x48015398,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR28,             0x4801539C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT28       CAN_MOCTR28
#define        CAN_MOSTAT28_bit   CAN_MOCTR28_bit
__IO_REG32_BIT(CAN_MOFCR29,             0x480153A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR29,            0x480153A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR29,             0x480153A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR29,             0x480153AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL29,           0x480153B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH29,           0x480153B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR29,              0x480153B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR29,             0x480153BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT29       CAN_MOCTR29
#define        CAN_MOSTAT29_bit   CAN_MOCTR29_bit
__IO_REG32_BIT(CAN_MOFCR30,             0x480153C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR30,            0x480153C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR30,             0x480153C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR30,             0x480153CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL30,           0x480153D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH30,           0x480153D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR30,              0x480153D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR30,             0x480153DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT30       CAN_MOCTR30
#define        CAN_MOSTAT30_bit   CAN_MOCTR30_bit
__IO_REG32_BIT(CAN_MOFCR31,             0x480153E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR31,            0x480153E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR31,             0x480153E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR31,             0x480153EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL31,           0x480153F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH31,           0x480153F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR31,              0x480153F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR31,             0x480153FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT31       CAN_MOCTR31
#define        CAN_MOSTAT31_bit   CAN_MOCTR31_bit
__IO_REG32_BIT(CAN_MOFCR32,             0x48015400,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR32,            0x48015404,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR32,             0x48015408,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR32,             0x4801540C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL32,           0x48015410,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH32,           0x48015414,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR32,              0x48015418,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR32,             0x4801541C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT32       CAN_MOCTR32
#define        CAN_MOSTAT32_bit   CAN_MOCTR32_bit
__IO_REG32_BIT(CAN_MOFCR33,             0x48015420,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR33,            0x48015424,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR33,             0x48015428,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR33,             0x4801542C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL33,           0x48015430,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH33,           0x48015434,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR33,              0x48015438,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR33,             0x4801543C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT33       CAN_MOCTR33
#define        CAN_MOSTAT33_bit   CAN_MOCTR33_bit
__IO_REG32_BIT(CAN_MOFCR34,             0x48015440,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR34,            0x48015444,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR34,             0x48015448,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR34,             0x4801544C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL34,           0x48015450,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH34,           0x48015454,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR34,              0x48015458,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR34,             0x4801545C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT34       CAN_MOCTR34
#define        CAN_MOSTAT34_bit   CAN_MOCTR34_bit
__IO_REG32_BIT(CAN_MOFCR35,             0x48015460,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR35,            0x48015464,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR35,             0x48015468,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR35,             0x4801546C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL35,           0x48015470,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH35,           0x48015474,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR35,              0x48015478,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR35,             0x4801547C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT35       CAN_MOCTR35
#define        CAN_MOSTAT35_bit   CAN_MOCTR35_bit
__IO_REG32_BIT(CAN_MOFCR36,             0x48015480,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR36,            0x48015484,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR36,             0x48015488,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR36,             0x4801548C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL36,           0x48015490,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH36,           0x48015494,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR36,              0x48015498,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR36,             0x4801549C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT36       CAN_MOCTR36
#define        CAN_MOSTAT36_bit   CAN_MOCTR36_bit
__IO_REG32_BIT(CAN_MOFCR37,             0x480154A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR37,            0x480154A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR37,             0x480154A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR37,             0x480154AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL37,           0x480154B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH37,           0x480154B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR37,              0x480154B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR37,             0x480154BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT37       CAN_MOCTR37
#define        CAN_MOSTAT37_bit   CAN_MOCTR37_bit
__IO_REG32_BIT(CAN_MOFCR38,             0x480154C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR38,            0x480154C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR38,             0x480154C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR38,             0x480154CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL38,           0x480154D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH38,           0x480154D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR38,              0x480154D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR38,             0x480154DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT38       CAN_MOCTR38
#define        CAN_MOSTAT38_bit   CAN_MOCTR38_bit
__IO_REG32_BIT(CAN_MOFCR39,             0x480154E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR39,            0x480154E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR39,             0x480154E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR39,             0x480154EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL39,           0x480154F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH39,           0x480154F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR39,              0x480154F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR39,             0x480154FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT39       CAN_MOCTR39
#define        CAN_MOSTAT39_bit   CAN_MOCTR39_bit
__IO_REG32_BIT(CAN_MOFCR40,             0x48015500,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR40,            0x48015504,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR40,             0x48015508,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR40,             0x4801550C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL40,           0x48015510,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH40,           0x48015514,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR40,              0x48015518,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR40,             0x4801551C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT40       CAN_MOCTR40
#define        CAN_MOSTAT40_bit   CAN_MOCTR40_bit
__IO_REG32_BIT(CAN_MOFCR41,             0x48015520,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR41,            0x48015524,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR41,             0x48015528,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR41,             0x4801552C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL41,           0x48015530,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH41,           0x48015534,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR41,              0x48015538,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR41,             0x4801553C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT41       CAN_MOCTR41
#define        CAN_MOSTAT41_bit   CAN_MOCTR41_bit
__IO_REG32_BIT(CAN_MOFCR42,             0x48015540,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR42,            0x48015544,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR42,             0x48015548,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR42,             0x4801554C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL42,           0x48015550,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH42,           0x48015554,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR42,              0x48015558,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR42,             0x4801555C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT42       CAN_MOCTR42
#define        CAN_MOSTAT42_bit   CAN_MOCTR42_bit
__IO_REG32_BIT(CAN_MOFCR43,             0x48015560,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR43,            0x48015564,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR43,             0x48015568,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR43,             0x4801556C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL43,           0x48015570,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH43,           0x48015574,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR43,              0x48015578,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR43,             0x4801557C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT43       CAN_MOCTR43
#define        CAN_MOSTAT43_bit   CAN_MOCTR43_bit
__IO_REG32_BIT(CAN_MOFCR44,             0x48015580,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR44,            0x48015584,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR44,             0x48015588,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR44,             0x4801558C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL44,           0x48015590,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH44,           0x48015594,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR44,              0x48015598,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR44,             0x4801559C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT44       CAN_MOCTR44
#define        CAN_MOSTAT44_bit   CAN_MOCTR44_bit
__IO_REG32_BIT(CAN_MOFCR45,             0x480155A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR45,            0x480155A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR45,             0x480155A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR45,             0x480155AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL45,           0x480155B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH45,           0x480155B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR45,              0x480155B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR45,             0x480155BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT45       CAN_MOCTR45
#define        CAN_MOSTAT45_bit   CAN_MOCTR45_bit
__IO_REG32_BIT(CAN_MOFCR46,             0x480155C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR46,            0x480155C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR46,             0x480155C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR46,             0x480155CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL46,           0x480155D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH46,           0x480155D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR46,              0x480155D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR46,             0x480155DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT46       CAN_MOCTR46
#define        CAN_MOSTAT46_bit   CAN_MOCTR46_bit
__IO_REG32_BIT(CAN_MOFCR47,             0x480155E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR47,            0x480155E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR47,             0x480155E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR47,             0x480155EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL47,           0x480155F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH47,           0x480155F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR47,              0x480155F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR47,             0x480155FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT47       CAN_MOCTR47
#define        CAN_MOSTAT47_bit   CAN_MOCTR47_bit
__IO_REG32_BIT(CAN_MOFCR48,             0x48015600,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR48,            0x48015604,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR48,             0x48015608,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR48,             0x4801560C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL48,           0x48015610,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH48,           0x48015614,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR48,              0x48015618,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR48,             0x4801561C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT48       CAN_MOCTR48
#define        CAN_MOSTAT48_bit   CAN_MOCTR48_bit
__IO_REG32_BIT(CAN_MOFCR49,             0x48015620,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR49,            0x48015624,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR49,             0x48015628,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR49,             0x4801562C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL49,           0x48015630,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH49,           0x48015634,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR49,              0x48015638,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR49,             0x4801563C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT49       CAN_MOCTR49
#define        CAN_MOSTAT49_bit   CAN_MOCTR49_bit
__IO_REG32_BIT(CAN_MOFCR50,             0x48015640,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR50,            0x48015644,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR50,             0x48015648,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR50,             0x4801564C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL50,           0x48015650,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH50,           0x48015654,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR50,              0x48015658,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR50,             0x4801565C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT50       CAN_MOCTR50
#define        CAN_MOSTAT50_bit   CAN_MOCTR50_bit
__IO_REG32_BIT(CAN_MOFCR51,             0x48015660,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR51,            0x48015664,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR51,             0x48015668,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR51,             0x4801566C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL51,           0x48015670,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH51,           0x48015674,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR51,              0x48015678,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR51,             0x4801567C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT51       CAN_MOCTR51
#define        CAN_MOSTAT51_bit   CAN_MOCTR51_bit
__IO_REG32_BIT(CAN_MOFCR52,             0x48015680,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR52,            0x48015684,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR52,             0x48015688,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR52,             0x4801568C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL52,           0x48015690,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH52,           0x48015694,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR52,              0x48015698,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR52,             0x4801569C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT52       CAN_MOCTR52
#define        CAN_MOSTAT52_bit   CAN_MOCTR52_bit
__IO_REG32_BIT(CAN_MOFCR53,             0x480156A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR53,            0x480156A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR53,             0x480156A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR53,             0x480156AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL53,           0x480156B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH53,           0x480156B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR53,              0x480156B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR53,             0x480156BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT53       CAN_MOCTR53
#define        CAN_MOSTAT53_bit   CAN_MOCTR53_bit
__IO_REG32_BIT(CAN_MOFCR54,             0x480156C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR54,            0x480156C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR54,             0x480156C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR54,             0x480156CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL54,           0x480156D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH54,           0x480156D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR54,              0x480156D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR54,             0x480156DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT54       CAN_MOCTR54
#define        CAN_MOSTAT54_bit   CAN_MOCTR54_bit
__IO_REG32_BIT(CAN_MOFCR55,             0x480156E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR55,            0x480156E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR55,             0x480156E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR55,             0x480156EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL55,           0x480156F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH55,           0x480156F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR55,              0x480156F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR55,             0x480156FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT55       CAN_MOCTR55
#define        CAN_MOSTAT55_bit   CAN_MOCTR55_bit
__IO_REG32_BIT(CAN_MOFCR56,             0x48015700,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR56,            0x48015704,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR56,             0x48015708,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR56,             0x4801570C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL56,           0x48015710,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH56,           0x48015714,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR56,              0x48015718,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR56,             0x4801571C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT56       CAN_MOCTR56
#define        CAN_MOSTAT56_bit   CAN_MOCTR56_bit
__IO_REG32_BIT(CAN_MOFCR57,             0x48015720,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR57,            0x48015724,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR57,             0x48015728,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR57,             0x4801572C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL57,           0x48015730,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH57,           0x48015734,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR57,              0x48015738,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR57,             0x4801573C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT57       CAN_MOCTR57
#define        CAN_MOSTAT57_bit   CAN_MOCTR57_bit
__IO_REG32_BIT(CAN_MOFCR58,             0x48015740,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR58,            0x48015744,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR58,             0x48015748,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR58,             0x4801574C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL58,           0x48015750,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH58,           0x48015754,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR58,              0x48015758,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR58,             0x4801575C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT58       CAN_MOCTR58
#define        CAN_MOSTAT58_bit   CAN_MOCTR58_bit
__IO_REG32_BIT(CAN_MOFCR59,             0x48015760,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR59,            0x48015764,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR59,             0x48015768,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR59,             0x4801576C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL59,           0x48015770,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH59,           0x48015774,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR59,              0x48015778,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR59,             0x4801577C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT59       CAN_MOCTR59
#define        CAN_MOSTAT59_bit   CAN_MOCTR59_bit
__IO_REG32_BIT(CAN_MOFCR60,             0x48015780,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR60,            0x48015784,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR60,             0x48015788,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR60,             0x4801578C,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL60,           0x48015790,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH60,           0x48015794,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR60,              0x48015798,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR60,             0x4801579C,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT60       CAN_MOCTR60
#define        CAN_MOSTAT60_bit   CAN_MOCTR60_bit
__IO_REG32_BIT(CAN_MOFCR61,             0x480157A0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR61,            0x480157A4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR61,             0x480157A8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR61,             0x480157AC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL61,           0x480157B0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH61,           0x480157B4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR61,              0x480157B8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR61,             0x480157BC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT61       CAN_MOCTR61
#define        CAN_MOSTAT61_bit   CAN_MOCTR61_bit
__IO_REG32_BIT(CAN_MOFCR62,             0x480157C0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR62,            0x480157C4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR62,             0x480157C8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR62,             0x480157CC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL62,           0x480157D0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH62,           0x480157D4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR62,              0x480157D8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR62,             0x480157DC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT62       CAN_MOCTR62
#define        CAN_MOSTAT62_bit   CAN_MOCTR62_bit
__IO_REG32_BIT(CAN_MOFCR63,             0x480157E0,__READ_WRITE ,__can_mofcr_bits);
__IO_REG32_BIT(CAN_MOFGPR63,            0x480157E4,__READ_WRITE ,__can_mofgpr_bits);
__IO_REG32_BIT(CAN_MOIPR63,             0x480157E8,__READ_WRITE ,__can_moipr_bits);
__IO_REG32_BIT(CAN_MOAMR63,             0x480157EC,__READ_WRITE ,__can_moamr_bits);
__IO_REG32_BIT(CAN_MODATAL63,           0x480157F0,__READ_WRITE ,__can_modatal_bits);
__IO_REG32_BIT(CAN_MODATAH63,           0x480157F4,__READ_WRITE ,__can_modatah_bits);
__IO_REG32_BIT(CAN_MOAR63,              0x480157F8,__READ_WRITE ,__can_moar_bits);
__IO_REG32_BIT(CAN_MOCTR63,             0x480157FC,__READ_WRITE ,__can_moctr_bits);
#define        CAN_MOSTAT63       CAN_MOCTR63
#define        CAN_MOSTAT63_bit   CAN_MOCTR63_bit

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_CLC,                 0x40004000,__READ_WRITE ,__adc_clc_bits);
__IO_REG32_BIT(ADC_ID,                  0x40004008,__READ       ,__adc_id_bits);
__IO_REG32_BIT(ADC_OCS,                 0x40004028,__READ_WRITE ,__adc_ocs_bits);
__IO_REG32_BIT(ADC_GLOBCFG,             0x40004080,__READ_WRITE ,__adc_globcfg_bits);
__IO_REG32_BIT(ADC_GLOBEFLAG,           0x400040E0,__READ_WRITE ,__adc_globeflag_bits);
__IO_REG32_BIT(ADC_GLOBEVNP,            0x40004140,__READ_WRITE ,__adc_globevnp_bits);
__IO_REG32_BIT(ADC_GLOBTF,              0x40004160,__READ_WRITE ,__adc_globtf_bits);
__IO_REG32_BIT(ADC_BRSSEL0,             0x40004180,__READ_WRITE ,__adc_brssel_bits);
__IO_REG32_BIT(ADC_BRSSEL1,             0x40004184,__READ_WRITE ,__adc_brssel_bits);
__IO_REG32_BIT(ADC_BRSSEL2,             0x40004188,__READ_WRITE ,__adc_brssel_bits);
__IO_REG32_BIT(ADC_BRSSEL3,             0x4000418C,__READ_WRITE ,__adc_brssel_bits);
__IO_REG32_BIT(ADC_GLOBICLASS0,         0x400041A0,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_GLOBICLASS1,         0x400041A4,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_GLOBBOUND,           0x400041B8,__READ_WRITE ,__adc_gbound_bits);
__IO_REG32_BIT(ADC_BRSPND0,             0x400041C0,__READ_WRITE ,__adc_brspnd_bits);
__IO_REG32_BIT(ADC_BRSPND1,             0x400041C4,__READ_WRITE ,__adc_brspnd_bits);
__IO_REG32_BIT(ADC_BRSPND2,             0x400041C8,__READ_WRITE ,__adc_brspnd_bits);
__IO_REG32_BIT(ADC_BRSPND3,             0x400041CC,__READ_WRITE ,__adc_brspnd_bits);
__IO_REG32_BIT(ADC_BRSCTRL,             0x40004200,__READ_WRITE ,__adc_brsctrl_bits);
__IO_REG32_BIT(ADC_BRSMR,               0x40004204,__READ_WRITE ,__adc_brsmr_bits);
__IO_REG32_BIT(ADC_GLOBRCR,             0x40004280,__READ_WRITE ,__adc_globrcr_bits);
__IO_REG32_BIT(ADC_GLOBRES,             0x40004300,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_GLOBRESD,            0x40004380,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_EMUXSEL,             0x400043F0,__READ_WRITE ,__adc_emuxsel_bits);

/***************************************************************************
 **
 ** ADC G0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_G0ARBCFG,            0x40004480,__READ_WRITE ,__adc_garbcfg_bits);
__IO_REG32_BIT(ADC_G0ARBPR,             0x40004484,__READ_WRITE ,__adc_garbpr_bits);
__IO_REG32_BIT(ADC_G0CHASS,             0x40004488,__READ_WRITE ,__adc_gchass_bits);
__IO_REG32_BIT(ADC_G0ICLASS0,           0x400044A0,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G0ICLASS1,           0x400044A4,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G0ALIAS,             0x400044B0,__READ_WRITE ,__adc_galias_bits);
__IO_REG32_BIT(ADC_G0BOUND,             0x400044B8,__READ_WRITE ,__adc_gbound_bits);
__IO_REG32_BIT(ADC_G0BFL,               0x400044C8,__READ_WRITE ,__adc_gbfl_bits);
__IO_REG32_BIT(ADC_G0BFLS,              0x400044CC,__WRITE      ,__adc_gbfls_bits);
__IO_REG32_BIT(ADC_G0BFLC,              0x400044D0,__READ_WRITE ,__adc_gbflc_bits);
__IO_REG32_BIT(ADC_GOSYNCTR,            0x400044C0,__READ_WRITE ,__adc_gsynctr_bits);
__IO_REG32_BIT(ADC_G0QCTRL0,            0x40004500,__READ_WRITE ,__adc_gqctrl0_bits);
__IO_REG32_BIT(ADC_G0QMR0,              0x40004504,__READ_WRITE ,__adc_gqmr0_bits);
__IO_REG32_BIT(ADC_G0QSR0,              0x40004508,__READ       ,__adc_gqsr0_bits);
__IO_REG32_BIT(ADC_G0Q0R0,              0x4000450C,__READ       ,__adc_gq0r0_bits);
__IO_REG32_BIT(ADC_G0QINR0,             0x40004510,__WRITE      ,__adc_gqinr0_bits);
__IO_REG32_BIT(ADC_G0ASCTRL,            0x40004520,__READ_WRITE ,__adc_gasctrl_bits);
__IO_REG32_BIT(ADC_G0ASMR,              0x40004524,__READ_WRITE ,__adc_gasmr_bits);
__IO_REG32_BIT(ADC_G0ASSEL,             0x40004528,__READ_WRITE ,__adc_gassel_bits);
__IO_REG32_BIT(ADC_G0ASPND,             0x4000452C,__READ_WRITE ,__adc_gaspnd_bits);
__IO_REG32_BIT(ADC_G0CEFLAG,            0x40004580,__READ_WRITE ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G0REFLAG,            0x40004584,__READ_WRITE ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G0SEFLAG,            0x40004588,__READ_WRITE ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G0CEFCLR,            0x40004590,__WRITE      ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G0REFCLR,            0x40004594,__WRITE      ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G0SEFCLR,            0x40004598,__WRITE      ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G0CEVNP0,            0x400045A0,__READ_WRITE ,__adc_gcevnp0_bits);
__IO_REG32_BIT(ADC_G0REVNP0,            0x400045B0,__READ_WRITE ,__adc_grevnp0_bits);
__IO_REG32_BIT(ADC_G0REVNP1,            0x400045B4,__READ_WRITE ,__adc_grevnp1_bits);
__IO_REG32_BIT(ADC_G0SEVNP,             0x400045C0,__READ_WRITE ,__adc_gsevnp_bits);
__IO_REG32_BIT(ADC_G0SRACT,             0x400045C8,__WRITE      ,__adc_gsract_bits);
__IO_REG32_BIT(ADC_G0EMUXCTR,           0x400045F0,__READ_WRITE ,__adc_gemuxctr_bits);
__IO_REG32_BIT(ADC_G0VFR,               0x400045F8,__READ_WRITE ,__adc_gvfr_bits);
__IO_REG32_BIT(ADC_G0CHCTR0,            0x40004600,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR1,            0x40004604,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR2,            0x40004608,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR3,            0x4000460C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR4,            0x40004610,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR5,            0x40004614,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR6,            0x40004618,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0CHCTR7,            0x4000461C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G0RCR0,              0x40004680,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR1,              0x40004684,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR2,              0x40004688,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR3,              0x4000468C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR4,              0x40004690,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR5,              0x40004694,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR6,              0x40004698,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR7,              0x4000469C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR8,              0x400046A0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR9,              0x400046A4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR10,             0x400046A8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR11,             0x400046AC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR12,             0x400046B0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR13,             0x400046B4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR14,             0x400046B8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RCR15,             0x400046BC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G0RES0,              0x40004700,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES1,              0x40004704,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES2,              0x40004708,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES3,              0x4000470C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES4,              0x40004710,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES5,              0x40004714,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES6,              0x40004718,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES7,              0x4000471C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES8,              0x40004720,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES9,              0x40004724,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES10,             0x40004728,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES11,             0x4000472C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES12,             0x40004730,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES13,             0x40004734,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES14,             0x40004738,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RES15,             0x4000473C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD0,             0x40004780,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD1,             0x40004784,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD2,             0x40004788,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD3,             0x4000478C,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD4,             0x40004790,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD5,             0x40004794,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD6,             0x40004798,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD7,             0x4000479C,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD8,             0x400047A0,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD9,             0x400047A4,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD10,            0x400047A8,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD11,            0x400047AC,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD12,            0x400047B0,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD13,            0x400047B4,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD14,            0x400047B8,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G0RESD15,            0x400047BC,__READ       ,__adc_gres_bits);

/***************************************************************************
 **
 ** ADC G1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_G1ARBCFG,            0x40004880,__READ_WRITE ,__adc_garbcfg_bits);
__IO_REG32_BIT(ADC_G1ARBPR,             0x40004884,__READ_WRITE ,__adc_garbpr_bits);
__IO_REG32_BIT(ADC_G1CHASS,             0x40004888,__READ_WRITE ,__adc_gchass_bits);
__IO_REG32_BIT(ADC_G1ICLASS0,           0x400048A0,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G1ICLASS1,           0x400048A4,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G1ALIAS,             0x400048B0,__READ_WRITE ,__adc_galias_bits);
__IO_REG32_BIT(ADC_G1BOUND,             0x400048B8,__READ_WRITE ,__adc_gbound_bits);
__IO_REG32_BIT(ADC_G1BFL,               0x400048C8,__READ_WRITE ,__adc_gbfl_bits);
__IO_REG32_BIT(ADC_G1BFLS,              0x400048CC,__WRITE      ,__adc_gbfls_bits);
__IO_REG32_BIT(ADC_G1BFLC,              0x400048D0,__READ_WRITE ,__adc_gbflc_bits);
__IO_REG32_BIT(ADC_G1SYNCTR,            0x400048C0,__READ_WRITE ,__adc_gsynctr_bits);
__IO_REG32_BIT(ADC_G1QCTRL0,            0x40004900,__READ_WRITE ,__adc_gqctrl0_bits);
__IO_REG32_BIT(ADC_G1QMR0,              0x40004904,__READ_WRITE ,__adc_gqmr0_bits);
__IO_REG32_BIT(ADC_G1QSR0,              0x40004908,__READ       ,__adc_gqsr0_bits);
__IO_REG32_BIT(ADC_G1Q0R0,              0x4000490C,__READ       ,__adc_gq0r0_bits);
__IO_REG32_BIT(ADC_G1QINR0,             0x40004910,__WRITE      ,__adc_gqinr0_bits);
__IO_REG32_BIT(ADC_G1ASCTRL,            0x40004920,__READ_WRITE ,__adc_gasctrl_bits);
__IO_REG32_BIT(ADC_G1ASMR,              0x40004924,__READ_WRITE ,__adc_gasmr_bits);
__IO_REG32_BIT(ADC_G1ASSEL,             0x40004928,__READ_WRITE ,__adc_gassel_bits);
__IO_REG32_BIT(ADC_G1ASPND,             0x4000492C,__READ_WRITE ,__adc_gaspnd_bits);
__IO_REG32_BIT(ADC_G1CEFLAG,            0x40004980,__READ_WRITE ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G1REFLAG,            0x40004984,__READ_WRITE ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G1SEFLAG,            0x40004988,__READ_WRITE ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G1CEFCLR,            0x40004990,__WRITE      ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G1REFCLR,            0x40004994,__WRITE      ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G1SEFCLR,            0x40004998,__WRITE      ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G1CEVNP0,            0x400049A0,__READ_WRITE ,__adc_gcevnp0_bits);
__IO_REG32_BIT(ADC_G1REVNP0,            0x400049B0,__READ_WRITE ,__adc_grevnp0_bits);
__IO_REG32_BIT(ADC_G1REVNP1,            0x400049B4,__READ_WRITE ,__adc_grevnp1_bits);
__IO_REG32_BIT(ADC_G1SEVNP,             0x400049C0,__READ_WRITE ,__adc_gsevnp_bits);
__IO_REG32_BIT(ADC_G1SRACT,             0x400049C8,__WRITE      ,__adc_gsract_bits);
__IO_REG32_BIT(ADC_G1EMUXCTR,           0x400049F0,__READ_WRITE ,__adc_gemuxctr_bits);
__IO_REG32_BIT(ADC_G1VFR,               0x400049F8,__READ_WRITE ,__adc_gvfr_bits);
__IO_REG32_BIT(ADC_G1CHCTR0,            0x40004A00,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR1,            0x40004A04,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR2,            0x40004A08,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR3,            0x40004A0C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR4,            0x40004A10,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR5,            0x40004A14,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR6,            0x40004A18,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1CHCTR7,            0x40004A1C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G1RCR0,              0x40004A80,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR1,              0x40004A84,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR2,              0x40004A88,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR3,              0x40004A8C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR4,              0x40004A90,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR5,              0x40004A94,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR6,              0x40004A98,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR7,              0x40004A9C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR8,              0x40004AA0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR9,              0x40004AA4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR10,             0x40004AA8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR11,             0x40004AAC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR12,             0x40004AB0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR13,             0x40004AB4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR14,             0x40004AB8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RCR15,             0x40004ABC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G1RES0,              0x40004B00,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES1,              0x40004B04,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES2,              0x40004B08,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES3,              0x40004B0C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES4,              0x40004B10,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES5,              0x40004B14,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES6,              0x40004B18,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES7,              0x40004B1C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES8,              0x40004B20,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES9,              0x40004B24,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES10,             0x40004B28,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES11,             0x40004B2C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES12,             0x40004B30,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES13,             0x40004B34,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES14,             0x40004B38,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RES15,             0x40004B3C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD0,             0x40004B80,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD1,             0x40004B84,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD2,             0x40004B88,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD3,             0x40004B8C,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD4,             0x40004B90,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD5,             0x40004B94,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD6,             0x40004B98,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD7,             0x40004B9C,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD8,             0x40004BA0,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD9,             0x40004BA4,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD10,            0x40004BA8,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD11,            0x40004BAC,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD12,            0x40004BB0,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD13,            0x40004BB4,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD14,            0x40004BB8,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G1RESD15,            0x40004BBC,__READ       ,__adc_gres_bits);

/***************************************************************************
 **
 ** ADC G2
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_G2ARBCFG,            0x40004C80,__READ_WRITE ,__adc_garbcfg_bits);
__IO_REG32_BIT(ADC_G2ARBPR,             0x40004C84,__READ_WRITE ,__adc_garbpr_bits);
__IO_REG32_BIT(ADC_G2CHASS,             0x40004C88,__READ_WRITE ,__adc_gchass_bits);
__IO_REG32_BIT(ADC_G2ICLASS0,           0x40004CA0,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G2ICLASS1,           0x40004CA4,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G2ALIAS,             0x40004CB0,__READ_WRITE ,__adc_galias_bits);
__IO_REG32_BIT(ADC_G2BOUND,             0x40004CB8,__READ_WRITE ,__adc_gbound_bits);
__IO_REG32_BIT(ADC_G2BFL,               0x40004CC8,__READ_WRITE ,__adc_gbfl_bits);
__IO_REG32_BIT(ADC_G2BFLS,              0x40004CCC,__WRITE      ,__adc_gbfls_bits);
__IO_REG32_BIT(ADC_G2BFLC,              0x40004CD0,__READ_WRITE ,__adc_gbflc_bits);
__IO_REG32_BIT(ADC_G2SYNCTR,            0x40004CC0,__READ_WRITE ,__adc_gsynctr_bits);
__IO_REG32_BIT(ADC_G2QCTRL0,            0x40004D00,__READ_WRITE ,__adc_gqctrl0_bits);
__IO_REG32_BIT(ADC_G2QMR0,              0x40004D04,__READ_WRITE ,__adc_gqmr0_bits);
__IO_REG32_BIT(ADC_G2QSR0,              0x40004D08,__READ       ,__adc_gqsr0_bits);
__IO_REG32_BIT(ADC_G2Q0R0,              0x40004D0C,__READ       ,__adc_gq0r0_bits);
__IO_REG32_BIT(ADC_G2QINR0,             0x40004D10,__WRITE      ,__adc_gqinr0_bits);
__IO_REG32_BIT(ADC_G2ASCTRL,            0x40004D20,__READ_WRITE ,__adc_gasctrl_bits);
__IO_REG32_BIT(ADC_G2ASMR,              0x40004D24,__READ_WRITE ,__adc_gasmr_bits);
__IO_REG32_BIT(ADC_G2ASSEL,             0x40004D28,__READ_WRITE ,__adc_gassel_bits);
__IO_REG32_BIT(ADC_G2ASPND,             0x40004D2C,__READ_WRITE ,__adc_gaspnd_bits);
__IO_REG32_BIT(ADC_G2CEFLAG,            0x40004D80,__READ_WRITE ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G2REFLAG,            0x40004D84,__READ_WRITE ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G2SEFLAG,            0x40004D88,__READ_WRITE ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G2CEFCLR,            0x40004D90,__WRITE      ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G2REFCLR,            0x40004D94,__WRITE      ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G2SEFCLR,            0x40004D98,__WRITE      ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G2CEVNP0,            0x40004DA0,__READ_WRITE ,__adc_gcevnp0_bits);
__IO_REG32_BIT(ADC_G2REVNP0,            0x40004DB0,__READ_WRITE ,__adc_grevnp0_bits);
__IO_REG32_BIT(ADC_G2REVNP1,            0x40004DB4,__READ_WRITE ,__adc_grevnp1_bits);
__IO_REG32_BIT(ADC_G2SEVNP,             0x40004DC0,__READ_WRITE ,__adc_gsevnp_bits);
__IO_REG32_BIT(ADC_G2SRACT,             0x40004DC8,__WRITE      ,__adc_gsract_bits);
__IO_REG32_BIT(ADC_G2EMUXCTR,           0x40004DF0,__READ_WRITE ,__adc_gemuxctr_bits);
__IO_REG32_BIT(ADC_G2VFR,               0x40004DF8,__READ_WRITE ,__adc_gvfr_bits);
__IO_REG32_BIT(ADC_G2CHCTR0,            0x40004E00,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR1,            0x40004E04,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR2,            0x40004E08,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR3,            0x40004E0C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR4,            0x40004E10,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR5,            0x40004E14,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR6,            0x40004E18,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2CHCTR7,            0x40004E1C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G2RCR0,              0x40004E80,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR1,              0x40004E84,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR2,              0x40004E88,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR3,              0x40004E8C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR4,              0x40004E90,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR5,              0x40004E94,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR6,              0x40004E98,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR7,              0x40004E9C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR8,              0x40004EA0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR9,              0x40004EA4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR10,             0x40004EA8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR11,             0x40004EAC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR12,             0x40004EB0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR13,             0x40004EB4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR14,             0x40004EB8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RCR15,             0x40004EBC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G2RES0,              0x40004F00,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES1,              0x40004F04,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES2,              0x40004F08,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES3,              0x40004F0C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES4,              0x40004F10,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES5,              0x40004F14,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES6,              0x40004F18,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES7,              0x40004F1C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES8,              0x40004F20,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES9,              0x40004F24,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES10,             0x40004F28,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES11,             0x40004F2C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES12,             0x40004F30,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES13,             0x40004F34,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES14,             0x40004F38,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RES15,             0x40004F3C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD0,             0x40004F80,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD1,             0x40004F84,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD2,             0x40004F88,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD3,             0x40004F8C,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD4,             0x40004F90,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD5,             0x40004F94,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD6,             0x40004F98,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD7,             0x40004F9C,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD8,             0x40004FA0,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD9,             0x40004FA4,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD10,            0x40004FA8,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD11,            0x40004FAC,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD12,            0x40004FB0,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD13,            0x40004FB4,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD14,            0x40004FB8,__READ       ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G2RESD15,            0x40004FBC,__READ       ,__adc_gres_bits);

/***************************************************************************
 **
 ** ADC G3
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_G3ARBCFG,            0x40005080,__READ_WRITE ,__adc_garbcfg_bits);
__IO_REG32_BIT(ADC_G3ARBPR,             0x40005084,__READ_WRITE ,__adc_garbpr_bits);
__IO_REG32_BIT(ADC_G3CHASS,             0x40005088,__READ_WRITE ,__adc_gchass_bits);
__IO_REG32_BIT(ADC_G3ICLASS0,           0x400050A0,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G3ICLASS1,           0x400050A4,__READ_WRITE ,__adc_globiclass_bits);
__IO_REG32_BIT(ADC_G3ALIAS,             0x400050B0,__READ_WRITE ,__adc_galias_bits);
__IO_REG32_BIT(ADC_G3BOUND,             0x400050B8,__READ_WRITE ,__adc_gbound_bits);
__IO_REG32_BIT(ADC_G3BFL,               0x400050C8,__READ_WRITE ,__adc_gbfl_bits);
__IO_REG32_BIT(ADC_G3BFLS,              0x400050CC,__WRITE      ,__adc_gbfls_bits);
__IO_REG32_BIT(ADC_G3BFLC,              0x400050D0,__READ_WRITE ,__adc_gbflc_bits);
__IO_REG32_BIT(ADC_G3SYNCTR,            0x400050C0,__READ_WRITE ,__adc_gsynctr_bits);
__IO_REG32_BIT(ADC_G3QCTRL0,            0x40005100,__READ_WRITE ,__adc_gqctrl0_bits);
__IO_REG32_BIT(ADC_G3QMR0,              0x40005104,__READ_WRITE ,__adc_gqmr0_bits);
__IO_REG32_BIT(ADC_G3QSR0,              0x40005108,__READ       ,__adc_gqsr0_bits);
__IO_REG32_BIT(ADC_G3Q0R0,              0x4000510C,__READ       ,__adc_gq0r0_bits);
__IO_REG32_BIT(ADC_G3QINR0,             0x40005110,__WRITE      ,__adc_gqinr0_bits);
__IO_REG32_BIT(ADC_G3ASCTRL,            0x40005120,__READ_WRITE ,__adc_gasctrl_bits);
__IO_REG32_BIT(ADC_G3ASMR,              0x40005124,__READ_WRITE ,__adc_gasmr_bits);
__IO_REG32_BIT(ADC_G3ASSEL,             0x40005128,__READ_WRITE ,__adc_gassel_bits);
__IO_REG32_BIT(ADC_G3ASPND,             0x4000512C,__READ_WRITE ,__adc_gaspnd_bits);
__IO_REG32_BIT(ADC_G3CEFLAG,            0x40005180,__READ_WRITE ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G3REFLAG,            0x40005184,__READ_WRITE ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G3SEFLAG,            0x40005188,__READ_WRITE ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G3CEFCLR,            0x40005190,__WRITE      ,__adc_gceflag_bits);
__IO_REG32_BIT(ADC_G3REFCLR,            0x40005194,__WRITE      ,__adc_greflag_bits);
__IO_REG32_BIT(ADC_G3SEFCLR,            0x40005198,__WRITE      ,__adc_gseflag_bits);
__IO_REG32_BIT(ADC_G3CEVNP0,            0x400051A0,__READ_WRITE ,__adc_gcevnp0_bits);
__IO_REG32_BIT(ADC_G3REVNP0,            0x400051B0,__READ_WRITE ,__adc_grevnp0_bits);
__IO_REG32_BIT(ADC_G3REVNP1,            0x400051B4,__READ_WRITE ,__adc_grevnp1_bits);
__IO_REG32_BIT(ADC_G3SEVNP,             0x400051C0,__READ_WRITE ,__adc_gsevnp_bits);
__IO_REG32_BIT(ADC_G3SRACT,             0x400051C8,__WRITE      ,__adc_gsract_bits);
__IO_REG32_BIT(ADC_G3EMUXCTR,           0x400051F0,__READ_WRITE ,__adc_gemuxctr_bits);
__IO_REG32_BIT(ADC_G3VFR,               0x400051F8,__READ_WRITE ,__adc_gvfr_bits);
__IO_REG32_BIT(ADC_G3CHCTR0,            0x40005200,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR1,            0x40005204,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR2,            0x40005208,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR3,            0x4000520C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR4,            0x40005210,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR5,            0x40005214,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR6,            0x40005218,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3CHCTR7,            0x4000521C,__READ_WRITE ,__adc_gchctr_bits);
__IO_REG32_BIT(ADC_G3RCR0,              0x40005280,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR1,              0x40005284,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR2,              0x40005288,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR3,              0x4000528C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR4,              0x40005290,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR5,              0x40005294,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR6,              0x40005298,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR7,              0x4000529C,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR8,              0x400052A0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR9,              0x400052A4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR10,             0x400052A8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR11,             0x400052AC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR12,             0x400052B0,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR13,             0x400052B4,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR14,             0x400052B8,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RCR15,             0x400052BC,__READ_WRITE ,__adc_grcr_bits);
__IO_REG32_BIT(ADC_G3RES0,              0x40005300,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES1,              0x40005304,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES2,              0x40005308,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES3,              0x4000530C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES4,              0x40005310,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES5,              0x40005314,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES6,              0x40005318,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES7,              0x4000531C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES8,              0x40005320,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES9,              0x40005324,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES10,             0x40005328,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES11,             0x4000532C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES12,             0x40005330,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES13,             0x40005334,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES14,             0x40005338,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RES15,             0x4000533C,__READ_WRITE ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD0,             0x40005380,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD1,             0x40005384,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD2,             0x40005388,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD3,             0x4000538C,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD4,             0x40005390,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD5,             0x40005394,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD6,             0x40005398,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD7,             0x4000539C,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD8,             0x400053A0,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD9,             0x400053A4,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD10,            0x400053A8,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD11,            0x400053AC,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD12,            0x400053B0,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD13,            0x400053B4,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD14,            0x400053B8,__READ     ,__adc_gres_bits);
__IO_REG32_BIT(ADC_G3RESD15,            0x400053BC,__READ     ,__adc_gres_bits);

/***************************************************************************
 **
 ** CCU40
 **
 ***************************************************************************/
__IO_REG32_BIT(CAPCOM40_GCTRL,          0x4000C000,__READ_WRITE ,__capcom_gctrl_bits);
__IO_REG32_BIT(CAPCOM40_GSTAT,          0x4000C004,__READ       ,__capcom_gstat_bits);
__IO_REG32_BIT(CAPCOM40_GIDLS,          0x4000C008,__WRITE      ,__capcom_gidls_bits);
__IO_REG32_BIT(CAPCOM40_GIDLC,          0x4000C00C,__WRITE      ,__capcom_gidlc_bits);
__IO_REG32_BIT(CAPCOM40_GCSS,           0x4000C010,__WRITE      ,__capcom_gcss_bits);
__IO_REG32_BIT(CAPCOM40_GCSC,           0x4000C014,__WRITE      ,__capcom_gcsc_bits);
__IO_REG32_BIT(CAPCOM40_GCST,           0x4000C018,__READ       ,__capcom_gcst_bits);
__IO_REG32_BIT(CAPCOM40_ECRD,           0x4000C050,__READ       ,__capcom_ecrd_bits);
__IO_REG32_BIT(CAPCOM40_MIDR,           0x4000C080,__READ       ,__capcom_midr_bits);

/***************************************************************************
 **
 ** CC40_0
 **
 ***************************************************************************/
__IO_REG32_BIT(CC40_0_INS,              0x4000C100,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC40_0_CMC,              0x4000C104,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC40_0_TST,              0x4000C108,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC40_0_TCSET,            0x4000C10C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC40_0_TCCLR,            0x4000C110,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC40_0_TC,               0x4000C114,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC40_0_PSL,              0x4000C118,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC40_0_DIT,              0x4000C11C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC40_0_DITS,             0x4000C120,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC40_0_PSC,              0x4000C124,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC40_0_FPC,              0x4000C128,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC40_0_FPCS,             0x4000C12C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC40_0_PR,               0x4000C130,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC40_0_PRS,              0x4000C134,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC40_0_CR,               0x4000C138,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC40_0_CRS,              0x4000C13C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC40_0_TIMER,            0x4000C170,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC40_0_C0V,              0x4000C174,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_0_C1V,              0x4000C178,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_0_C2V,              0x4000C17C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_0_C3V,              0x4000C180,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_0_INTS,             0x4000C1A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC40_0_INTE,             0x4000C1A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC40_0_SRS,              0x4000C1A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC40_0_SWS,              0x4000C1AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC40_0_SWR,              0x4000C1B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC40_1
 **
 ***************************************************************************/
__IO_REG32_BIT(CC40_1_INS,              0x4000C200,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC40_1_CMC,              0x4000C204,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC40_1_TST,              0x4000C208,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC40_1_TCSET,            0x4000C20C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC40_1_TCCLR,            0x4000C210,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC40_1_TC,               0x4000C214,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC40_1_PSL,              0x4000C218,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC40_1_DIT,              0x4000C21C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC40_1_DITS,             0x4000C220,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC40_1_PSC,              0x4000C224,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC40_1_FPC,              0x4000C228,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC40_1_FPCS,             0x4000C22C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC40_1_PR,               0x4000C230,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC40_1_PRS,              0x4000C234,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC40_1_CR,               0x4000C238,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC40_1_CRS,              0x4000C23C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC40_1_TIMER,            0x4000C270,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC40_1_C0V,              0x4000C274,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_1_C1V,              0x4000C278,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_1_C2V,              0x4000C27C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_1_C3V,              0x4000C280,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_1_INTS,             0x4000C2A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC40_1_INTE,             0x4000C2A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC40_1_SRS,              0x4000C2A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC40_1_SWS,              0x4000C2AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC40_1_SWR,              0x4000C2B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC40_2
 **
 ***************************************************************************/
__IO_REG32_BIT(CC40_2_INS,              0x4000C300,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC40_2_CMC,              0x4000C304,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC40_2_TST,              0x4000C308,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC40_2_TCSET,            0x4000C30C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC40_2_TCCLR,            0x4000C310,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC40_2_TC,               0x4000C314,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC40_2_PSL,              0x4000C318,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC40_2_DIT,              0x4000C31C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC40_2_DITS,             0x4000C320,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC40_2_PSC,              0x4000C324,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC40_2_FPC,              0x4000C328,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC40_2_FPCS,             0x4000C32C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC40_2_PR,               0x4000C330,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC40_2_PRS,              0x4000C334,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC40_2_CR,               0x4000C338,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC40_2_CRS,              0x4000C33C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC40_2_TIMER,            0x4000C370,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC40_2_C0V,              0x4000C374,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_2_C1V,              0x4000C378,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_2_C2V,              0x4000C37C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_2_C3V,              0x4000C380,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_2_INTS,             0x4000C3A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC40_2_INTE,             0x4000C3A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC40_2_SRS,              0x4000C3A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC40_2_SWS,              0x4000C3AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC40_2_SWR,              0x4000C3B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC40_3
 **
 ***************************************************************************/
__IO_REG32_BIT(CC40_3_INS,              0x4000C400,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC40_3_CMC,              0x4000C404,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC40_3_TST,              0x4000C408,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC40_3_TCSET,            0x4000C40C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC40_3_TCCLR,            0x4000C410,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC40_3_TC,               0x4000C414,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC40_3_PSL,              0x4000C418,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC40_3_DIT,              0x4000C41C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC40_3_DITS,             0x4000C420,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC40_3_PSC,              0x4000C424,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC40_3_FPC,              0x4000C428,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC40_3_FPCS,             0x4000C42C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC40_3_PR,               0x4000C430,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC40_3_PRS,              0x4000C434,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC40_3_CR,               0x4000C438,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC40_3_CRS,              0x4000C43C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC40_3_TIMER,            0x4000C470,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC40_3_C0V,              0x4000C474,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_3_C1V,              0x4000C478,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_3_C2V,              0x4000C47C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_3_C3V,              0x4000C480,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC40_3_INTS,             0x4000C4A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC40_3_INTE,             0x4000C4A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC40_3_SRS,              0x4000C4A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC40_3_SWS,              0x4000C4AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC40_3_SWR,              0x4000C4B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CCU41
 **
 ***************************************************************************/
__IO_REG32_BIT(CAPCOM41_GCTRL,          0x40010000,__READ_WRITE ,__capcom_gctrl_bits);
__IO_REG32_BIT(CAPCOM41_GSTAT,          0x40010004,__READ       ,__capcom_gstat_bits);
__IO_REG32_BIT(CAPCOM41_GIDLS,          0x40010008,__WRITE      ,__capcom_gidls_bits);
__IO_REG32_BIT(CAPCOM41_GIDLC,          0x4001000C,__WRITE      ,__capcom_gidlc_bits);
__IO_REG32_BIT(CAPCOM41_GCSS,           0x40010010,__WRITE      ,__capcom_gcss_bits);
__IO_REG32_BIT(CAPCOM41_GCSC,           0x40010014,__WRITE      ,__capcom_gcsc_bits);
__IO_REG32_BIT(CAPCOM41_GCST,           0x40010018,__READ       ,__capcom_gcst_bits);
__IO_REG32_BIT(CAPCOM41_ECRD,           0x40010050,__READ       ,__capcom_ecrd_bits);
__IO_REG32_BIT(CAPCOM41_MIDR,           0x40010080,__READ       ,__capcom_midr_bits);

/***************************************************************************
 **
 ** CC41_0
 **
 ***************************************************************************/
__IO_REG32_BIT(CC41_0_INS,              0x40010100,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC41_0_CMC,              0x40010104,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC41_0_TST,              0x40010108,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC41_0_TCSET,            0x4001010C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC41_0_TCCLR,            0x40010110,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC41_0_TC,               0x40010114,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC41_0_PSL,              0x40010118,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC41_0_DIT,              0x4001011C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC41_0_DITS,             0x40010120,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC41_0_PSC,              0x40010124,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC41_0_FPC,              0x40010128,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC41_0_FPCS,             0x4001012C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC41_0_PR,               0x40010130,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC41_0_PRS,              0x40010134,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC41_0_CR,               0x40010138,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC41_0_CRS,              0x4001013C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC41_0_TIMER,            0x40010170,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC41_0_C0V,              0x40010174,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_0_C1V,              0x40010178,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_0_C2V,              0x4001017C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_0_C3V,              0x40010180,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_0_INTS,             0x400101A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC41_0_INTE,             0x400101A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC41_0_SRS,              0x400101A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC41_0_SWS,              0x400101AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC41_0_SWR,              0x400101B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC41_1
 **
 ***************************************************************************/
__IO_REG32_BIT(CC41_1_INS,              0x40010200,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC41_1_CMC,              0x40010204,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC41_1_TST,              0x40010208,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC41_1_TCSET,            0x4001020C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC41_1_TCCLR,            0x40010210,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC41_1_TC,               0x40010214,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC41_1_PSL,              0x40010218,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC41_1_DIT,              0x4001021C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC41_1_DITS,             0x40010220,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC41_1_PSC,              0x40010224,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC41_1_FPC,              0x40010228,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC41_1_FPCS,             0x4001022C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC41_1_PR,               0x40010230,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC41_1_PRS,              0x40010234,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC41_1_CR,               0x40010238,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC41_1_CRS,              0x4001023C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC41_1_TIMER,            0x40010270,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC41_1_C0V,              0x40010274,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_1_C1V,              0x40010278,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_1_C2V,              0x4001027C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_1_C3V,              0x40010280,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_1_INTS,             0x400102A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC41_1_INTE,             0x400102A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC41_1_SRS,              0x400102A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC41_1_SWS,              0x400102AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC41_1_SWR,              0x400102B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC41_2
 **
 ***************************************************************************/
__IO_REG32_BIT(CC41_2_INS,              0x40010300,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC41_2_CMC,              0x40010304,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC41_2_TST,              0x40010308,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC41_2_TCSET,            0x4001030C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC41_2_TCCLR,            0x40010310,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC41_2_TC,               0x40010314,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC41_2_PSL,              0x40010318,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC41_2_DIT,              0x4001031C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC41_2_DITS,             0x40010320,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC41_2_PSC,              0x40010324,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC41_2_FPC,              0x40010328,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC41_2_FPCS,             0x4001032C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC41_2_PR,               0x40010330,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC41_2_PRS,              0x40010334,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC41_2_CR,               0x40010338,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC41_2_CRS,              0x4001033C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC41_2_TIMER,            0x40010370,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC41_2_C0V,              0x40010374,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_2_C1V,              0x40010378,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_2_C2V,              0x4001037C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_2_C3V,              0x40010380,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_2_INTS,             0x400103A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC41_2_INTE,             0x400103A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC41_2_SRS,              0x400103A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC41_2_SWS,              0x400103AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC41_2_SWR,              0x400103B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC41_3
 **
 ***************************************************************************/
__IO_REG32_BIT(CC41_3_INS,              0x40010400,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC41_3_CMC,              0x40010404,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC41_3_TST,              0x40010408,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC41_3_TCSET,            0x4001040C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC41_3_TCCLR,            0x40010410,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC41_3_TC,               0x40010414,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC41_3_PSL,              0x40010418,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC41_3_DIT,              0x4001041C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC41_3_DITS,             0x40010420,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC41_3_PSC,              0x40010424,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC41_3_FPC,              0x40010428,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC41_3_FPCS,             0x4001042C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC41_3_PR,               0x40010430,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC41_3_PRS,              0x40010434,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC41_3_CR,               0x40010438,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC41_3_CRS,              0x4001043C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC41_3_TIMER,            0x40010470,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC41_3_C0V,              0x40010474,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_3_C1V,              0x40010478,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_3_C2V,              0x4001047C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_3_C3V,              0x40010480,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC41_3_INTS,             0x400104A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC41_3_INTE,             0x400104A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC41_3_SRS,              0x400104A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC41_3_SWS,              0x400104AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC41_3_SWR,              0x400104B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CCU42
 **
 ***************************************************************************/
__IO_REG32_BIT(CAPCOM42_GCTRL,          0x40014000,__READ_WRITE ,__capcom_gctrl_bits);
__IO_REG32_BIT(CAPCOM42_GSTAT,          0x40014004,__READ       ,__capcom_gstat_bits);
__IO_REG32_BIT(CAPCOM42_GIDLS,          0x40014008,__WRITE      ,__capcom_gidls_bits);
__IO_REG32_BIT(CAPCOM42_GIDLC,          0x4001400C,__WRITE      ,__capcom_gidlc_bits);
__IO_REG32_BIT(CAPCOM42_GCSS,           0x40014010,__WRITE      ,__capcom_gcss_bits);
__IO_REG32_BIT(CAPCOM42_GCSC,           0x40014014,__WRITE      ,__capcom_gcsc_bits);
__IO_REG32_BIT(CAPCOM42_GCST,           0x40014018,__READ       ,__capcom_gcst_bits);
__IO_REG32_BIT(CAPCOM42_ECRD,           0x40014050,__READ       ,__capcom_ecrd_bits);
__IO_REG32_BIT(CAPCOM42_MIDR,           0x40014080,__READ       ,__capcom_midr_bits);

/***************************************************************************
 **
 ** CC42_0
 **
 ***************************************************************************/
__IO_REG32_BIT(CC42_0_INS,              0x40014100,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC42_0_CMC,              0x40014104,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC42_0_TST,              0x40014108,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC42_0_TCSET,            0x4001410C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC42_0_TCCLR,            0x40014110,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC42_0_TC,               0x40014114,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC42_0_PSL,              0x40014118,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC42_0_DIT,              0x4001411C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC42_0_DITS,             0x40014120,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC42_0_PSC,              0x40014124,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC42_0_FPC,              0x40014128,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC42_0_FPCS,             0x4001412C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC42_0_PR,               0x40014130,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC42_0_PRS,              0x40014134,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC42_0_CR,               0x40014138,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC42_0_CRS,              0x4001413C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC42_0_TIMER,            0x40014170,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC42_0_C0V,              0x40014174,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_0_C1V,              0x40014178,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_0_C2V,              0x4001417C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_0_C3V,              0x40014180,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_0_INTS,             0x400141A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC42_0_INTE,             0x400141A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC42_0_SRS,              0x400141A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC42_0_SWS,              0x400141AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC42_0_SWR,              0x400141B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC42_1
 **
 ***************************************************************************/
__IO_REG32_BIT(CC42_1_INS,              0x40014200,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC42_1_CMC,              0x40014204,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC42_1_TST,              0x40014208,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC42_1_TCSET,            0x4001420C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC42_1_TCCLR,            0x40014210,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC42_1_TC,               0x40014214,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC42_1_PSL,              0x40014218,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC42_1_DIT,              0x4001421C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC42_1_DITS,             0x40014220,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC42_1_PSC,              0x40014224,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC42_1_FPC,              0x40014228,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC42_1_FPCS,             0x4001422C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC42_1_PR,               0x40014230,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC42_1_PRS,              0x40014234,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC42_1_CR,               0x40014238,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC42_1_CRS,              0x4001423C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC42_1_TIMER,            0x40014270,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC42_1_C0V,              0x40014274,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_1_C1V,              0x40014278,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_1_C2V,              0x4001427C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_1_C3V,              0x40014280,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_1_INTS,             0x400142A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC42_1_INTE,             0x400142A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC42_1_SRS,              0x400142A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC42_1_SWS,              0x400142AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC42_1_SWR,              0x400142B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC42_2
 **
 ***************************************************************************/
__IO_REG32_BIT(CC42_2_INS,              0x40014300,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC42_2_CMC,              0x40014304,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC42_2_TST,              0x40014308,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC42_2_TCSET,            0x4001430C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC42_2_TCCLR,            0x40014310,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC42_2_TC,               0x40014314,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC42_2_PSL,              0x40014318,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC42_2_DIT,              0x4001431C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC42_2_DITS,             0x40014320,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC42_2_PSC,              0x40014324,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC42_2_FPC,              0x40014328,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC42_2_FPCS,             0x4001432C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC42_2_PR,               0x40014330,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC42_2_PRS,              0x40014334,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC42_2_CR,               0x40014338,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC42_2_CRS,              0x4001433C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC42_2_TIMER,            0x40014370,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC42_2_C0V,              0x40014374,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_2_C1V,              0x40014378,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_2_C2V,              0x4001437C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_2_C3V,              0x40014380,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_2_INTS,             0x400143A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC42_2_INTE,             0x400143A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC42_2_SRS,              0x400143A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC42_2_SWS,              0x400143AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC42_2_SWR,              0x400143B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC42_3
 **
 ***************************************************************************/
__IO_REG32_BIT(CC42_3_INS,              0x40014400,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC42_3_CMC,              0x40014404,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC42_3_TST,              0x40014408,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC42_3_TCSET,            0x4001440C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC42_3_TCCLR,            0x40014410,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC42_3_TC,               0x40014414,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC42_3_PSL,              0x40014418,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC42_3_DIT,              0x4001441C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC42_3_DITS,             0x40014420,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC42_3_PSC,              0x40014424,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC42_3_FPC,              0x40014428,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC42_3_FPCS,             0x4001442C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC42_3_PR,               0x40014430,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC42_3_PRS,              0x40014434,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC42_3_CR,               0x40014438,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC42_3_CRS,              0x4001443C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC42_3_TIMER,            0x40014470,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC42_3_C0V,              0x40014474,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_3_C1V,              0x40014478,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_3_C2V,              0x4001447C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_3_C3V,              0x40014480,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC42_3_INTS,             0x400144A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC42_3_INTE,             0x400144A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC42_3_SRS,              0x400144A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC42_3_SWS,              0x400144AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC42_3_SWR,              0x400144B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CCU43
 **
 ***************************************************************************/
__IO_REG32_BIT(CAPCOM43_GCTRL,          0x48004000,__READ_WRITE ,__capcom_gctrl_bits);
__IO_REG32_BIT(CAPCOM43_GSTAT,          0x48004004,__READ       ,__capcom_gstat_bits);
__IO_REG32_BIT(CAPCOM43_GIDLS,          0x48004008,__WRITE      ,__capcom_gidls_bits);
__IO_REG32_BIT(CAPCOM43_GIDLC,          0x4800400C,__WRITE      ,__capcom_gidlc_bits);
__IO_REG32_BIT(CAPCOM43_GCSS,           0x48004010,__WRITE      ,__capcom_gcss_bits);
__IO_REG32_BIT(CAPCOM43_GCSC,           0x48004014,__WRITE      ,__capcom_gcsc_bits);
__IO_REG32_BIT(CAPCOM43_GCST,           0x48004018,__READ       ,__capcom_gcst_bits);
__IO_REG32_BIT(CAPCOM43_ECRD,           0x48004050,__READ       ,__capcom_ecrd_bits);
__IO_REG32_BIT(CAPCOM43_MIDR,           0x48004080,__READ       ,__capcom_midr_bits);

/***************************************************************************
 **
 ** CC43_0
 **
 ***************************************************************************/
__IO_REG32_BIT(CC43_0_INS,              0x48004100,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC43_0_CMC,              0x48004104,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC43_0_TST,              0x48004108,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC43_0_TCSET,            0x4800410C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC43_0_TCCLR,            0x48004110,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC43_0_TC,               0x48004114,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC43_0_PSL,              0x48004118,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC43_0_DIT,              0x4800411C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC43_0_DITS,             0x48004120,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC43_0_PSC,              0x48004124,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC43_0_FPC,              0x48004128,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC43_0_FPCS,             0x4800412C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC43_0_PR,               0x48004130,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC43_0_PRS,              0x48004134,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC43_0_CR,               0x48004138,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC43_0_CRS,              0x4800413C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC43_0_TIMER,            0x48004170,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC43_0_C0V,              0x48004174,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_0_C1V,              0x48004178,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_0_C2V,              0x4800417C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_0_C3V,              0x48004180,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_0_INTS,             0x480041A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC43_0_INTE,             0x480041A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC43_0_SRS,              0x480041A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC43_0_SWS,              0x480041AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC43_0_SWR,              0x480041B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC43_1
 **
 ***************************************************************************/
__IO_REG32_BIT(CC43_1_INS,              0x48004200,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC43_1_CMC,              0x48004204,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC43_1_TST,              0x48004208,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC43_1_TCSET,            0x4800420C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC43_1_TCCLR,            0x48004210,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC43_1_TC,               0x48004214,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC43_1_PSL,              0x48004218,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC43_1_DIT,              0x4800421C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC43_1_DITS,             0x48004220,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC43_1_PSC,              0x48004224,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC43_1_FPC,              0x48004228,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC43_1_FPCS,             0x4800422C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC43_1_PR,               0x48004230,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC43_1_PRS,              0x48004234,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC43_1_CR,               0x48004238,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC43_1_CRS,              0x4800423C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC43_1_TIMER,            0x48004270,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC43_1_C0V,              0x48004274,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_1_C1V,              0x48004278,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_1_C2V,              0x4800427C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_1_C3V,              0x48004280,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_1_INTS,             0x480042A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC43_1_INTE,             0x480042A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC43_1_SRS,              0x480042A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC43_1_SWS,              0x480042AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC43_1_SWR,              0x480042B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC43_2
 **
 ***************************************************************************/
__IO_REG32_BIT(CC43_2_INS,              0x48004300,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC43_2_CMC,              0x48004304,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC43_2_TST,              0x48004308,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC43_2_TCSET,            0x4800430C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC43_2_TCCLR,            0x48004310,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC43_2_TC,               0x48004314,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC43_2_PSL,              0x48004318,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC43_2_DIT,              0x4800431C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC43_2_DITS,             0x48004320,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC43_2_PSC,              0x48004324,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC43_2_FPC,              0x48004328,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC43_2_FPCS,             0x4800432C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC43_2_PR,               0x48004330,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC43_2_PRS,              0x48004334,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC43_2_CR,               0x48004338,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC43_2_CRS,              0x4800433C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC43_2_TIMER,            0x48004370,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC43_2_C0V,              0x48004374,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_2_C1V,              0x48004378,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_2_C2V,              0x4800437C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_2_C3V,              0x48004380,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_2_INTS,             0x480043A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC43_2_INTE,             0x480043A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC43_2_SRS,              0x480043A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC43_2_SWS,              0x480043AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC43_2_SWR,              0x480043B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CC43_3
 **
 ***************************************************************************/
__IO_REG32_BIT(CC43_3_INS,              0x48004400,__READ_WRITE ,__ccins_bits);
__IO_REG32_BIT(CC43_3_CMC,              0x48004404,__READ_WRITE ,__cccmc_bits);
__IO_REG32_BIT(CC43_3_TST,              0x48004408,__READ       ,__cctst_bits);
__IO_REG32_BIT(CC43_3_TCSET,            0x4800440C,__WRITE      ,__cctcset_bits);
__IO_REG32_BIT(CC43_3_TCCLR,            0x48004410,__READ_WRITE ,__cctcclr_bits);
__IO_REG32_BIT(CC43_3_TC,               0x48004414,__READ_WRITE ,__cctc_bits);
__IO_REG32_BIT(CC43_3_PSL,              0x48004418,__READ_WRITE ,__ccpsl_bits);
__IO_REG32_BIT(CC43_3_DIT,              0x4800441C,__READ       ,__ccdit_bits);
__IO_REG32_BIT(CC43_3_DITS,             0x48004420,__READ_WRITE ,__ccdits_bits);
__IO_REG32_BIT(CC43_3_PSC,              0x48004424,__READ_WRITE ,__ccpsc_bits);
__IO_REG32_BIT(CC43_3_FPC,              0x48004428,__READ_WRITE ,__ccfpc_bits);
__IO_REG32_BIT(CC43_3_FPCS,             0x4800442C,__READ_WRITE ,__ccfpcs_bits);
__IO_REG32_BIT(CC43_3_PR,               0x48004430,__READ_WRITE ,__ccpr_bits);
__IO_REG32_BIT(CC43_3_PRS,              0x48004434,__READ_WRITE ,__ccprs_bits);
__IO_REG32_BIT(CC43_3_CR,               0x48004438,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CC43_3_CRS,              0x4800443C,__READ_WRITE ,__cccrs_bits);
__IO_REG32_BIT(CC43_3_TIMER,            0x48004470,__READ_WRITE ,__cctimer_bits);
__IO_REG32_BIT(CC43_3_C0V,              0x48004474,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_3_C1V,              0x48004478,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_3_C2V,              0x4800447C,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_3_C3V,              0x48004480,__READ       ,__cccv_bits);
__IO_REG32_BIT(CC43_3_INTS,             0x480044A0,__READ_WRITE ,__ccints_bits);
__IO_REG32_BIT(CC43_3_INTE,             0x480044A4,__READ_WRITE ,__ccinte_bits);
__IO_REG32_BIT(CC43_3_SRS,              0x480044A8,__READ_WRITE ,__ccsrs_bits);
__IO_REG32_BIT(CC43_3_SWS,              0x480044AC,__WRITE      ,__ccsws_bits);
__IO_REG32_BIT(CC43_3_SWR,              0x480044B0,__WRITE      ,__ccswr_bits);

/***************************************************************************
 **
 ** CCU80
 **
 ***************************************************************************/
__IO_REG32_BIT(CAPCOM80_GCTRL,          0x40020000,__READ_WRITE ,__capcom8_gctrl_bits);
__IO_REG32_BIT(CAPCOM80_GSTAT,          0x40020004,__READ       ,__capcom8_gstat_bits);
__IO_REG32_BIT(CAPCOM80_GIDLS,          0x40020008,__WRITE      ,__capcom8_gidls_bits);
__IO_REG32_BIT(CAPCOM80_GIDLC,          0x4002000C,__WRITE      ,__capcom8_gidlc_bits);
__IO_REG32_BIT(CAPCOM80_GCSS,           0x40020010,__WRITE      ,__capcom8_gcss_bits);
__IO_REG32_BIT(CAPCOM80_GCSC,           0x40020014,__WRITE      ,__capcom8_gcsc_bits);
__IO_REG32_BIT(CAPCOM80_GCST,           0x40020018,__READ       ,__capcom8_gcst_bits);
__IO_REG32_BIT(CAPCOM80_GPCHK,          0x4002001C,__READ_WRITE ,__capcom8_gpchk_bits);
__IO_REG32_BIT(CAPCOM80_ECRD,           0x40020050,__READ       ,__capcom8_ecrd_bits);
__IO_REG32_BIT(CAPCOM80_MIDR,           0x40020080,__READ       ,__capcom8_midr_bits);

/***************************************************************************
 **
 ** CC80_0
 **
 ***************************************************************************/
__IO_REG32_BIT(CC80_0_INS,              0x40020100,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC80_0_CMC,              0x40020104,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC80_0_TCST,             0x40020108,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC80_0_TCSET,            0x4002010C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC80_0_TCCLR,            0x40020110,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC80_0_TC,               0x40020114,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC80_0_PSL,              0x40020118,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC80_0_DIT,              0x4002011C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC80_0_DITS,             0x40020120,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC80_0_PSC,              0x40020124,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC80_0_FPC,              0x40020128,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC80_0_FPCS,             0x4002012C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC80_0_PR,               0x40020130,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC80_0_PRS,              0x40020134,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC80_0_CR1,              0x40020138,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_0_CR1S,             0x4002013C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_0_CR2,              0x40020140,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_0_CR2S,             0x40020144,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_0_CHC,              0x40020148,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC80_0_DTC,              0x4002014C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC80_0_DC1R,             0x40020150,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_0_DC2R,             0x40020154,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_0_TIMER,            0x40020170,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC80_0_C0V,              0x40020174,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_0_C1V,              0x40020178,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_0_C2V,              0x4002017C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_0_C3V,              0x40020180,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_0_INTS,             0x400201A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC80_0_INTE,             0x400201A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC80_0_SRS,              0x400201A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC80_0_SWS,              0x400201AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC80_0_SWR,              0x400201B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CC80_1
 **
 ***************************************************************************/
__IO_REG32_BIT(CC80_1_INS,              0x40020200,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC80_1_CMC,              0x40020204,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC80_1_TCST,             0x40020208,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC80_1_TCSET,            0x4002020C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC80_1_TCCLR,            0x40020210,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC80_1_TC,               0x40020214,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC80_1_PSL,              0x40020218,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC80_1_DIT,              0x4002021C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC80_1_DITS,             0x40020220,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC80_1_PSC,              0x40020224,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC80_1_FPC,              0x40020228,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC80_1_FPCS,             0x4002022C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC80_1_PR,               0x40020230,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC80_1_PRS,              0x40020234,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC80_1_CR1,              0x40020238,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_1_CR1S,             0x4002023C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_1_CR2,              0x40020240,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_1_CR2S,             0x40020244,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_1_CHC,              0x40020248,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC80_1_DTC,              0x4002024C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC80_1_DC1R,             0x40020250,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_1_DC2R,             0x40020254,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_1_TIMER,            0x40020270,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC80_1_C0V,              0x40020274,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_1_C1V,              0x40020278,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_1_C2V,              0x4002027C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_1_C3V,              0x40020280,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_1_INTS,             0x400202A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC80_1_INTE,             0x400202A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC80_1_SRS,              0x400202A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC80_1_SWS,              0x400202AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC80_1_SWR,              0x400202B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CC80_2
 **
 ***************************************************************************/
__IO_REG32_BIT(CC80_2_INS,              0x40020300,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC80_2_CMC,              0x40020304,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC80_2_TCST,             0x40020308,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC80_2_TCSET,            0x4002030C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC80_2_TCCLR,            0x40020310,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC80_2_TC,               0x40020314,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC80_2_PSL,              0x40020318,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC80_2_DIT,              0x4002031C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC80_2_DITS,             0x40020320,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC80_2_PSC,              0x40020324,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC80_2_FPC,              0x40020328,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC80_2_FPCS,             0x4002032C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC80_2_PR,               0x40020330,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC80_2_PRS,              0x40020334,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC80_2_CR1,              0x40020338,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_2_CR1S,             0x4002033C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_2_CR2,              0x40020340,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_2_CR2S,             0x40020344,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_2_CHC,              0x40020348,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC80_2_DTC,              0x4002034C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC80_2_DC1R,             0x40020350,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_2_DC2R,             0x40020354,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_2_TIMER,            0x40020370,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC80_2_C0V,              0x40020374,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_2_C1V,              0x40020378,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_2_C2V,              0x4002037C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_2_C3V,              0x40020380,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_2_INTS,             0x400203A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC80_2_INTE,             0x400203A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC80_2_SRS,              0x400203A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC80_2_SWS,              0x400203AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC80_2_SWR,              0x400203B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CC80_3
 **
 ***************************************************************************/
__IO_REG32_BIT(CC80_3_INS,              0x40020400,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC80_3_CMC,              0x40020404,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC80_3_TCST,             0x40020408,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC80_3_TCSET,            0x4002040C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC80_3_TCCLR,            0x40020410,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC80_3_TC,               0x40020414,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC80_3_PSL,              0x40020418,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC80_3_DIT,              0x4002041C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC80_3_DITS,             0x40020420,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC80_3_PSC,              0x40020424,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC80_3_FPC,              0x40020428,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC80_3_FPCS,             0x4002042C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC80_3_PR,               0x40020430,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC80_3_PRS,              0x40020434,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC80_3_CR1,              0x40020438,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_3_CR1S,             0x4002043C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_3_CR2,              0x40020440,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC80_3_CR2S,             0x40020444,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC80_3_CHC,              0x40020448,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC80_3_DTC,              0x4002044C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC80_3_DC1R,             0x40020450,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_3_DC2R,             0x40020454,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC80_3_TIMER,            0x40020470,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC80_3_C0V,              0x40020474,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_3_C1V,              0x40020478,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_3_C2V,              0x4002047C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_3_C3V,              0x40020480,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC80_3_INTS,             0x400204A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC80_3_INTE,             0x400204A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC80_3_SRS,              0x400204A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC80_3_SWS,              0x400204AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC80_3_SWR,              0x400204B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CCU81
 **
 ***************************************************************************/
__IO_REG32_BIT(CAPCOM81_GCTRL,          0x40024000,__READ_WRITE ,__capcom8_gctrl_bits);
__IO_REG32_BIT(CAPCOM81_GSTAT,          0x40024004,__READ       ,__capcom8_gstat_bits);
__IO_REG32_BIT(CAPCOM81_GIDLS,          0x40024008,__WRITE      ,__capcom8_gidls_bits);
__IO_REG32_BIT(CAPCOM81_GIDLC,          0x4002400C,__WRITE      ,__capcom8_gidlc_bits);
__IO_REG32_BIT(CAPCOM81_GCSS,           0x40024010,__WRITE      ,__capcom8_gcss_bits);
__IO_REG32_BIT(CAPCOM81_GCSC,           0x40024014,__WRITE      ,__capcom8_gcsc_bits);
__IO_REG32_BIT(CAPCOM81_GCST,           0x40024018,__READ       ,__capcom8_gcst_bits);
__IO_REG32_BIT(CAPCOM81_GPCHK,          0x4002401C,__READ_WRITE ,__capcom8_gpchk_bits);
__IO_REG32_BIT(CAPCOM81_ECRD,           0x40024050,__READ       ,__capcom8_ecrd_bits);
__IO_REG32_BIT(CAPCOM81_MIDR,           0x40024080,__READ       ,__capcom8_midr_bits);

/***************************************************************************
 **
 ** CC81_0
 **
 ***************************************************************************/
__IO_REG32_BIT(CC81_0_INS,              0x40024100,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC81_0_CMC,              0x40024104,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC81_0_TCST,             0x40024108,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC81_0_TCSET,            0x4002410C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC81_0_TCCLR,            0x40024110,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC81_0_TC,               0x40024114,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC81_0_PSL,              0x40024118,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC81_0_DIT,              0x4002411C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC81_0_DITS,             0x40024120,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC81_0_PSC,              0x40024124,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC81_0_FPC,              0x40024128,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC81_0_FPCS,             0x4002412C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC81_0_PR,               0x40024130,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC81_0_PRS,              0x40024134,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC81_0_CR1,              0x40024138,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_0_CR1S,             0x4002413C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_0_CR2,              0x40024140,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_0_CR2S,             0x40024144,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_0_CHC,              0x40024148,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC81_0_DTC,              0x4002414C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC81_0_DC1R,             0x40024150,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_0_DC2R,             0x40024154,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_0_TIMER,            0x40024170,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC81_0_C0V,              0x40024174,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_0_C1V,              0x40024178,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_0_C2V,              0x4002417C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_0_C3V,              0x40024180,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_0_INTS,             0x400241A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC81_0_INTE,             0x400241A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC81_0_SRS,              0x400241A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC81_0_SWS,              0x400241AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC81_0_SWR,              0x400241B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CC81_1
 **
 ***************************************************************************/
__IO_REG32_BIT(CC81_1_INS,              0x40024200,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC81_1_CMC,              0x40024204,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC81_1_TCST,             0x40024208,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC81_1_TCSET,            0x4002420C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC81_1_TCCLR,            0x40024210,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC81_1_TC,               0x40024214,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC81_1_PSL,              0x40024218,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC81_1_DIT,              0x4002421C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC81_1_DITS,             0x40024220,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC81_1_PSC,              0x40024224,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC81_1_FPC,              0x40024228,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC81_1_FPCS,             0x4002422C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC81_1_PR,               0x40024230,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC81_1_PRS,              0x40024234,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC81_1_CR1,              0x40024238,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_1_CR1S,             0x4002423C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_1_CR2,              0x40024240,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_1_CR2S,             0x40024244,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_1_CHC,              0x40024248,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC81_1_DTC,              0x4002424C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC81_1_DC1R,             0x40024250,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_1_DC2R,             0x40024254,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_1_TIMER,            0x40024270,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC81_1_C0V,              0x40024274,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_1_C1V,              0x40024278,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_1_C2V,              0x4002427C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_1_C3V,              0x40024280,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_1_INTS,             0x400242A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC81_1_INTE,             0x400242A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC81_1_SRS,              0x400242A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC81_1_SWS,              0x400242AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC81_1_SWR,              0x400242B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CC81_2
 **
 ***************************************************************************/
__IO_REG32_BIT(CC81_2_INS,              0x40024300,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC81_2_CMC,              0x40024304,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC81_2_TCST,             0x40024308,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC81_2_TCSET,            0x4002430C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC81_2_TCCLR,            0x40024310,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC81_2_TC,               0x40024314,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC81_2_PSL,              0x40024318,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC81_2_DIT,              0x4002431C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC81_2_DITS,             0x40024320,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC81_2_PSC,              0x40024324,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC81_2_FPC,              0x40024328,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC81_2_FPCS,             0x4002432C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC81_2_PR,               0x40024330,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC81_2_PRS,              0x40024334,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC81_2_CR1,              0x40024338,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_2_CR1S,             0x4002433C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_2_CR2,              0x40024340,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_2_CR2S,             0x40024344,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_2_CHC,              0x40024348,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC81_2_DTC,              0x4002434C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC81_2_DC1R,             0x40024350,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_2_DC2R,             0x40024354,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_2_TIMER,            0x40024370,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC81_2_C0V,              0x40024374,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_2_C1V,              0x40024378,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_2_C2V,              0x4002437C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_2_C3V,              0x40024380,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_2_INTS,             0x400243A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC81_2_INTE,             0x400243A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC81_2_SRS,              0x400243A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC81_2_SWS,              0x400243AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC81_2_SWR,              0x400243B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** CC81_3
 **
 ***************************************************************************/
__IO_REG32_BIT(CC81_3_INS,              0x40024400,__READ_WRITE ,__cc8ins_bits);
__IO_REG32_BIT(CC81_3_CMC,              0x40024404,__READ_WRITE ,__cc8cmc_bits);
__IO_REG32_BIT(CC81_3_TCST,             0x40024408,__READ       ,__cc8tcst_bits);
__IO_REG32_BIT(CC81_3_TCSET,            0x4002440C,__WRITE      ,__cc8tcset_bits);
__IO_REG32_BIT(CC81_3_TCCLR,            0x40024410,__READ_WRITE ,__cc8tcclr_bits);
__IO_REG32_BIT(CC81_3_TC,               0x40024414,__READ_WRITE ,__cc8tc_bits);
__IO_REG32_BIT(CC81_3_PSL,              0x40024418,__READ_WRITE ,__cc8psl_bits);
__IO_REG32_BIT(CC81_3_DIT,              0x4002441C,__READ       ,__cc8dit_bits);
__IO_REG32_BIT(CC81_3_DITS,             0x40024420,__READ_WRITE ,__cc8dits_bits);
__IO_REG32_BIT(CC81_3_PSC,              0x40024424,__READ_WRITE ,__cc8psc_bits);
__IO_REG32_BIT(CC81_3_FPC,              0x40024428,__READ_WRITE ,__cc8fpc_bits);
__IO_REG32_BIT(CC81_3_FPCS,             0x4002442C,__READ_WRITE ,__cc8fpcs_bits);
__IO_REG32_BIT(CC81_3_PR,               0x40024430,__READ_WRITE ,__cc8pr_bits);
__IO_REG32_BIT(CC81_3_PRS,              0x40024434,__READ_WRITE ,__cc8prs_bits);
__IO_REG32_BIT(CC81_3_CR1,              0x40024438,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_3_CR1S,             0x4002443C,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_3_CR2,              0x40024440,__READ_WRITE ,__cc8cr_bits);
__IO_REG32_BIT(CC81_3_CR2S,             0x40024444,__READ_WRITE ,__cc8crs_bits);
__IO_REG32_BIT(CC81_3_CHC,              0x40024448,__READ_WRITE ,__cc8chc_bits);
__IO_REG32_BIT(CC81_3_DTC,              0x4002444C,__READ_WRITE ,__cc8dtc_bits);
__IO_REG32_BIT(CC81_3_DC1R,             0x40024450,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_3_DC2R,             0x40024454,__READ_WRITE ,__cc8dcr_bits);
__IO_REG32_BIT(CC81_3_TIMER,            0x40024470,__READ_WRITE ,__cc8timer_bits);
__IO_REG32_BIT(CC81_3_C0V,              0x40024474,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_3_C1V,              0x40024478,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_3_C2V,              0x4002447C,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_3_C3V,              0x40024480,__READ       ,__cc8cv_bits);
__IO_REG32_BIT(CC81_3_INTS,             0x400244A0,__READ_WRITE ,__cc8ints_bits);
__IO_REG32_BIT(CC81_3_INTE,             0x400244A4,__READ_WRITE ,__cc8inte_bits);
__IO_REG32_BIT(CC81_3_SRS,              0x400244A8,__READ_WRITE ,__cc8srs_bits);
__IO_REG32_BIT(CC81_3_SWS,              0x400244AC,__WRITE      ,__cc8sws_bits);
__IO_REG32_BIT(CC81_3_SWR,              0x400244B0,__WRITE      ,__cc8swr_bits);

/***************************************************************************
 **
 ** POSIF0
 **
 ***************************************************************************/
__IO_REG32_BIT(POSIF0_PCONF,            0x40028000,__READ_WRITE ,__posif_pconf_bits);
__IO_REG32_BIT(POSIF0_PSUS,             0x40028004,__READ_WRITE ,__posif_psus_bits);
__IO_REG32_BIT(POSIF0_PRUNS,            0x40028008,__READ_WRITE ,__posif_pruns_bits);
__IO_REG32_BIT(POSIF0_PRUNC,            0x4002800C,__WRITE      ,__posif_prunc_bits);
__IO_REG32_BIT(POSIF0_PRUN,             0x40028010,__READ       ,__posif_prun_bits);
__IO_REG32_BIT(POSIF0_MIDR,             0x40028020,__READ       ,__posif_midr_bits);
__IO_REG32_BIT(POSIF0_HALP,             0x40028030,__READ       ,__posif_halp_bits);
__IO_REG32_BIT(POSIF0_HALPS,            0x40028034,__READ_WRITE ,__posif_halps_bits);
__IO_REG32_BIT(POSIF0_MCM,              0x40028040,__READ       ,__posif_mcm_bits);
__IO_REG32_BIT(POSIF0_MCSM,             0x40028044,__READ_WRITE ,__posif_mcsm_bits);
__IO_REG32_BIT(POSIF0_MCMS,             0x40028048,__WRITE      ,__posif_mcms_bits);
__IO_REG32_BIT(POSIF0_MCMC,             0x4002804C,__WRITE      ,__posif_mcmc_bits);
__IO_REG32_BIT(POSIF0_MCMF,             0x40028050,__READ       ,__posif_mcmf_bits);
__IO_REG32_BIT(POSIF0_QDC,              0x40028060,__READ_WRITE ,__posif_qdc_bits);
__IO_REG32_BIT(POSIF0_PFLG,             0x40028070,__READ       ,__posif_pflg_bits);
__IO_REG32_BIT(POSIF0_PFLGE,            0x40028074,__READ_WRITE ,__posif_pflge_bits);
__IO_REG32_BIT(POSIF0_SPFLG,            0x40028078,__WRITE      ,__posif_spflg_bits);
__IO_REG32_BIT(POSIF0_RPFLG,            0x4002807C,__WRITE      ,__posif_rpflg_bits);

/***************************************************************************
 **
 ** POSIF1
 **
 ***************************************************************************/
__IO_REG32_BIT(POSIF1_PCONF,            0x4002C000,__READ_WRITE ,__posif_pconf_bits);
__IO_REG32_BIT(POSIF1_PSUS,             0x4002C004,__READ_WRITE ,__posif_psus_bits);
__IO_REG32_BIT(POSIF1_PRUNS,            0x4002C008,__READ_WRITE ,__posif_pruns_bits);
__IO_REG32_BIT(POSIF1_PRUNC,            0x4002C00C,__WRITE      ,__posif_prunc_bits);
__IO_REG32_BIT(POSIF1_PRUN,             0x4002C010,__READ       ,__posif_prun_bits);
__IO_REG32_BIT(POSIF1_MIDR,             0x4002C020,__READ       ,__posif_midr_bits);
__IO_REG32_BIT(POSIF1_HALP,             0x4002C030,__READ       ,__posif_halp_bits);
__IO_REG32_BIT(POSIF1_HALPS,            0x4002C034,__READ_WRITE ,__posif_halps_bits);
__IO_REG32_BIT(POSIF1_MCM,              0x4002C040,__READ       ,__posif_mcm_bits);
__IO_REG32_BIT(POSIF1_MCSM,             0x4002C044,__READ_WRITE ,__posif_mcsm_bits);
__IO_REG32_BIT(POSIF1_MCMS,             0x4002C048,__WRITE      ,__posif_mcms_bits);
__IO_REG32_BIT(POSIF1_MCMC,             0x4002C04C,__WRITE      ,__posif_mcmc_bits);
__IO_REG32_BIT(POSIF1_MCMF,             0x4002C050,__READ       ,__posif_mcmf_bits);
__IO_REG32_BIT(POSIF1_QDC,              0x4002C060,__READ_WRITE ,__posif_qdc_bits);
__IO_REG32_BIT(POSIF1_PFLG,             0x4002C070,__READ       ,__posif_pflg_bits);
__IO_REG32_BIT(POSIF1_PFLGE,            0x4002C074,__READ_WRITE ,__posif_pflge_bits);
__IO_REG32_BIT(POSIF1_SPFLG,            0x4002C078,__WRITE      ,__posif_spflg_bits);
__IO_REG32_BIT(POSIF1_RPFLG,            0x4002C07C,__WRITE      ,__posif_rpflg_bits);

/***************************************************************************
 **
 ** P0
 **
 ***************************************************************************/
__IO_REG32_BIT(P0_OUT,                  0x48028000,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P0_OMR,                  0x48028004,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P0_IOCR0,                0x48028010,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P0_IOCR4,                0x48028014,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P0_IOCR8,                0x48028018,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P0_IOCR12,               0x4802801C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P0_IN,                   0x48028024,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P0_PDR0,                 0x48028040,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P0_PDR1,                 0x48028044,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P0_PDISC,                0x48028060,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P0_PPS,                  0x48028070,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P0_HWSEL,                0x48028074,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P1
 **
 ***************************************************************************/
__IO_REG32_BIT(P1_OUT,                  0x48028100,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P1_OMR,                  0x48028104,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P1_IOCR0,                0x48028110,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P1_IOCR4,                0x48028114,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P1_IOCR8,                0x48028118,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P1_IOCR12,               0x4802811C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P1_IN,                   0x48028124,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P1_PDR0,                 0x48028140,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P1_PDR1,                 0x48028144,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P1_PDISC,                0x48028160,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P1_PPS,                  0x48028170,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P1_HWSEL,                0x48028174,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P2
 **
 ***************************************************************************/
__IO_REG32_BIT(P2_OUT,                  0x48028200,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P2_OMR,                  0x48028204,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P2_IOCR0,                0x48028210,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P2_IOCR4,                0x48028214,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P2_IOCR8,                0x48028218,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P2_IOCR12,               0x4802821C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P2_IN,                   0x48028224,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P2_PDR0,                 0x48028240,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P2_PDR1,                 0x48028244,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P2_PDISC,                0x48028260,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P2_PPS,                  0x48028270,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P2_HWSEL,                0x48028274,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P3
 **
 ***************************************************************************/
__IO_REG32_BIT(P3_OUT,                  0x48028300,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P3_OMR,                  0x48028304,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P3_IOCR0,                0x48028310,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P3_IOCR4,                0x48028314,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P3_IOCR8,                0x48028318,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P3_IOCR12,               0x4802831C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P3_IN,                   0x48028324,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P3_PDR0,                 0x48028340,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P3_PDR1,                 0x48028344,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P3_PDISC,                0x48028360,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P3_PPS,                  0x48028370,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P3_HWSEL,                0x48028374,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P4
 **
 ***************************************************************************/
__IO_REG32_BIT(P4_OUT,                  0x48028400,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P4_OMR,                  0x48028404,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P4_IOCR0,                0x48028410,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P4_IOCR4,                0x48028414,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P4_IOCR8,                0x48028418,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P4_IOCR12,               0x4802841C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P4_IN,                   0x48028424,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P4_PDR0,                 0x48028440,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P4_PDR1,                 0x48028444,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P4_PDISC,                0x48028460,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P4_PPS,                  0x48028470,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P4_HWSEL,                0x48028474,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P5
 **
 ***************************************************************************/
__IO_REG32_BIT(P5_OUT,                  0x48028500,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P5_OMR,                  0x48028504,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P5_IOCR0,                0x48028510,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P5_IOCR4,                0x48028514,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P5_IOCR8,                0x48028518,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P5_IOCR12,               0x4802851C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P5_IN,                   0x48028524,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P5_PDR0,                 0x48028540,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P5_PDR1,                 0x48028544,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P5_PDISC,                0x48028560,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P5_PPS,                  0x48028570,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P5_HWSEL,                0x48028574,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P6
 **
 ***************************************************************************/
__IO_REG32_BIT(P6_OUT,                  0x48028600,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P6_OMR,                  0x48028604,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P6_IOCR0,                0x48028610,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P6_IOCR4,                0x48028614,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P6_IOCR8,                0x48028618,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P6_IOCR12,               0x4802861C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P6_IN,                   0x48028624,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P6_PDR0,                 0x48028640,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P6_PDR1,                 0x48028644,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P6_PDISC,                0x48028660,__READ       ,__p_pdisc_bits);
__IO_REG32_BIT(P6_PPS,                  0x48028670,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P6_HWSEL,                0x48028674,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P14
 **
 ***************************************************************************/
__IO_REG32_BIT(P14_OUT,                 0x48028E00,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P14_OMR,                 0x48028E04,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P14_IOCR0,               0x48028E10,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P14_IOCR4,               0x48028E14,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P14_IOCR8,               0x48028E18,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P14_IOCR12,              0x48028E1C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P14_IN,                  0x48028E24,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P14_PDR0,                0x48028E40,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P14_PDR1,                0x48028E44,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P14_PDISC,               0x48028E60,__READ       ,__p_pdisc14_bits);
__IO_REG32_BIT(P14_PPS,                 0x48028E70,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P14_HWSEL,               0x48028E74,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **
 ** P15
 **
 ***************************************************************************/
__IO_REG32_BIT(P15_OUT,                 0x48028F00,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P15_OMR,                 0x48028F04,__READ_WRITE ,__p_omr_bits);
__IO_REG32_BIT(P15_IOCR0,               0x48028F10,__READ_WRITE ,__p_iocr0_bits);
__IO_REG32_BIT(P15_IOCR4,               0x48028F14,__READ_WRITE ,__p_iocr4_bits);
__IO_REG32_BIT(P15_IOCR8,               0x48028F18,__READ_WRITE ,__p_iocr8_bits);
__IO_REG32_BIT(P15_IOCR12,              0x48028F1C,__READ_WRITE ,__p_iocr12_bits);
__IO_REG32_BIT(P15_IN,                  0x48028F24,__READ_WRITE ,__p_out_bits);
__IO_REG32_BIT(P15_PDR0,                0x48028F40,__READ_WRITE ,__p_pdr0_bits);
__IO_REG32_BIT(P15_PDR1,                0x48028F44,__READ_WRITE ,__p_pdr1_bits);
__IO_REG32_BIT(P15_PDISC,               0x48028F60,__READ       ,__p_pdisc15_bits);
__IO_REG32_BIT(P15_PPS,                 0x48028F70,__READ_WRITE ,__p_pps_bits);
__IO_REG32_BIT(P15_HWSEL,               0x48028F74,__READ_WRITE ,__p_hwsel_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  eDMA channels
 **
 ***************************************************************************/
#define DMAMUX_SCU_SR1            0
#define DMAMUX_VADC_C0SR0         1
#define DMAMUX_VADC_G0SR3         2
#define DMAMUX_VADC_G2SR0         3
#define DMAMUX_VADC_G2SR3         4
#define DMAMUX_DSD_SR0            5
#define DMAMUX_CCU40_SR0          6
#define DMAMUX_CCU80_SR0          7
#define DMAMUX_MCAN0_SR0          9
#define DMAMUX_USIC0_SR0         10
#define DMAMUX_USIC1_SR0         11
#define DMAMUX_VADC_G3SR3        13
#define DMAMUX_CCU42_SR0         14
#define DMAMUX_SCU_SR4           16
#define DMAMUX_VADC_C0SR1        17
#define DMAMUX_VADC_G0SR2        18
#define DMAMUX_VADC_G1SR0        19
#define DMAMUX_VADC_G2SR2        20
#define DMAMUX_DAC_SR0           21
#define DMAMUX_CCU40_SR0         22
#define DMAMUX_CCU80_SR0         23
#define DMAMUX_MCAN0_SR0         25
#define DMAMUX_USIC0_SR0         26
#define DMAMUX_USIC1_SR0         27
#define DMAMUX_VADC_G3SR0        29
#define DMAMUX_CCU42_SR0         30
#define DMAMUX_SCU_SR2           32
#define DMAMUX_VADC_C0SR2        33
#define DMAMUX_VADC_C0SR3        34
#define DMAMUX_VADC_G1SR3        35
#define DMAMUX_VADC_G2SR1        36
#define DMAMUX_DSD_SR1           37
#define DMAMUX_DSD_SR3           38
#define DMAMUX_CCU40_SR1         39
#define DMAMUX_CCU80_SR1         40
#define DMAMUX_MCAN0_SR1         42
#define DMAMUX_USIC0_SR1         43
#define DMAMUX_USIC1_SR1         44
#define DMAMUX_VADC_G3SR2        45
#define DMAMUX_CCU42_SR1         46
#define DMAMUX_SCU_SR3           48
#define DMAMUX_VADC_C0SR2        49
#define DMAMUX_VADC_C0SR3        50
#define DMAMUX_VADC_G1SR1        51
#define DMAMUX_VADC_G1SR2        52
#define DMAMUX_DSD_SR2           53
#define DMAMUX_DAC_SR1           54
#define DMAMUX_CCU40_SR1         55
#define DMAMUX_CCU80_SR1         56
#define DMAMUX_MCAN0_SR1         58
#define DMAMUX_USIC0_SR1         59
#define DMAMUX_USIC1_SR1         60
#define DMAMUX_VADC_G3SR1        61
#define DMAMUX_CCU42_SR1         62
#define DMAMUX_SCU_SR3           64
#define DMAMUX_VADC_G0SR0        65
#define DMAMUX_VADC_G0SR1        66
#define DMAMUX_VADC_G2SR1        67
#define DMAMUX_VADC_G2SR2        68
#define DMAMUX_DSD_SR2           69
#define DMAMUX_DAC_SR1           70
#define DMAMUX_CCU41_SR0         71
#define DMAMUX_CCU81_SR0         72
#define DMAMUX_MCAN0_SR2         74
#define DMAMUX_USIC0_SR0         75
#define DMAMUX_USIC1_SR0         76
#define DMAMUX_VADC_G3SR1        77
#define DMAMUX_CCU43_SR0         78
#define DMAMUX_SCU_SR2           80
#define DMAMUX_VADC_G0SR0        81
#define DMAMUX_VADC_G0SR1        82
#define DMAMUX_VADC_G1SR2        83
#define DMAMUX_VADC_G2SR0        84
#define DMAMUX_DAC_SR0           85
#define DMAMUX_CCU41_SR0         86
#define DMAMUX_CCU81_SR0         87
#define DMAMUX_MCAN0_SR2         89
#define DMAMUX_USIC0_SR0         90
#define DMAMUX_USIC1_SR0         91
#define DMAMUX_VADC_G3SR2        93
#define DMAMUX_CCU43_SR0         94
#define DMAMUX_SCU_SR4           96
#define DMAMUX_VADC_C0SR1        97
#define DMAMUX_VADC_G0SR2        98
#define DMAMUX_VADC_G1SR1        99
#define DMAMUX_VADC_G2SR3       100
#define DMAMUX_DSD_SR1          101
#define DMAMUX_DSD_SR3          102
#define DMAMUX_CCU41_SR1        103
#define DMAMUX_CCU81_SR1        104
#define DMAMUX_MCAN0_SR3        106
#define DMAMUX_USIC0_SR1        107
#define DMAMUX_USIC1_SR1        108
#define DMAMUX_VADC_G3SR0       109
#define DMAMUX_CCU43_SR1        110
#define DMAMUX_SCU_SR1          112
#define DMAMUX_VADC_C0SR0       113
#define DMAMUX_VADC_G0SR3       114
#define DMAMUX_VADC_G1SR0       115
#define DMAMUX_VADC_G1SR3       116
#define DMAMUX_DSD_SR0          117
#define DMAMUX_CCU41_SR1        118
#define DMAMUX_CCU81_SR1        119
#define DMAMUX_MCAN0_SR3        121
#define DMAMUX_USIC0_SR1        122
#define DMAMUX_USIC1_SR1        123
#define DMAMUX_VADC_G3SR3       125
#define DMAMUX_CCU43_SR1        126
#define DMAMUX_SCU_SR1          128
#define DMAMUX_VADC_C0SR0       129
#define DMAMUX_VADC_G3SR0       130
#define DMAMUX_DSD_SR0          131
#define DMAMUX_DAC_SR0          132
#define DMAMUX_CCU42_SR0        133
#define DMAMUX_USIC2.SR0        134
#define DMAMUX_USIC2_SR2        135
#define DMAMUX_SCU_SR2          144
#define DMAMUX_SCU_SR2          144
#define DMAMUX_VADC_C0SR1       145
#define DMAMUX_VADC_G3SR1       146
#define DMAMUX_DSD_SR1          147
#define DMAMUX_DAC_SR1          148
#define DMAMUX_CCU42_SR1        149
#define DMAMUX_USIC2_SR1        150
#define DMAMUX_USIC2_SR3        151
#define DMAMUX_SCU_SR3          160
#define DMAMUX_VADC_C0SR2       161
#define DMAMUX_VADC_G3SR2       162
#define DMAMUX_DSD_SR2          163
#define DMAMUX_DAC_SR0          164
#define DMAMUX_CCU43_SR0        165
#define DMAMUX_USIC2_SR0        166
#define DMAMUX_USIC2_SR2        167
#define DMAMUX_SCU_SR4          176
#define DMAMUX_VADC_C0SR3       177
#define DMAMUX_VADC_G3SR3       178
#define DMAMUX_DSD_SR3          179
#define DMAMUX_DAC_SR1          180
#define DMAMUX_CCU43_SR1        181
#define DMAMUX_USIC2_SR1        182
#define DMAMUX_USIC2_SR3        183

/***************************************************************************
 **
 **  NVIC Interrupt channels
 **
 ***************************************************************************/
#define MAIN_STACK             0  /* Main Stack                                                   */
#define RESETI                 1  /* Reset                                                        */
#define NMII                   2  /* Non-maskable Interrupt                                       */
#define HFI                    3  /* Hard Fault                                                   */
#define MMI                    4  /* Memory Management                                            */
#define BFI                    5  /* Bus Fault                                                    */
#define UFI                    6  /* Usage Fault                                                  */
#define SVCI                  11  /* SVCall                                                       */
#define DMI                   12  /* Debug Monitor                                                */
#define PSI                   14  /* PendSV                                                       */
#define STI                   15  /* SysTick                                                      */
#define NVIC_SCU_0            16  /* System Control 0                                             */
#define NVIC_ERU0_0           17  /* System Control 1                                             */
#define NVIC_ERU0_1           18  /* System Control 2                                             */
#define NVIC_ERU0_2           19  /* System Control 3                                             */
#define NVIC_ERU0_3           20  /* System Control 4                                             */
#define NVIC_ERU1_0           21  /* System Control 5                                             */
#define NVIC_ERU1_1           22  /* System Control 6                                             */
#define NVIC_ERU1_2           23  /* System Control 7                                             */
#define NVIC_ERU1_3           24  /* System Control 8                                             */
#define NVIC_WDT_0            27  /* Watchdog Timer                                               */
#define NVIC_PMU0_0           28  /* Program Management Unit 0                                    */
#define NVIC_PMU0_1           29  /* Program Management Unit 1                                    */
#define NVIC_VADC0_C0_0       30  /* Analog to Digital Converter Common Block 0 0                 */
#define NVIC_VADC0_C0_1       31  /* Analog to Digital Converter Common Block 0 1                 */
#define NVIC_VADC0_C0_2       32  /* Analog to Digital Converter Common Block 0 2                 */
#define NVIC_VADC0_C0_3       33  /* Analog to Digital Converter Common Block 0 3                 */
#define NVIC_VADC0_G0_0       34  /* Analog to Digital Converter Group 0 0                        */
#define NVIC_VADC0_G0_1       35  /* Analog to Digital Converter Group 0 1                        */
#define NVIC_VADC0_G0_2       36  /* Analog to Digital Converter Group 0 2                        */
#define NVIC_VADC0_G0_3       37  /* Analog to Digital Converter Group 0 3                        */
#define NVIC_VADC0_G1_0       38  /* Analog to Digital Converter Group 1 0                        */
#define NVIC_VADC0_G1_1       39  /* Analog to Digital Converter Group 1 1                        */
#define NVIC_VADC0_G1_2       40  /* Analog to Digital Converter Group 1 2                        */
#define NVIC_VADC0_G1_3       41  /* Analog to Digital Converter Group 1 3                        */
#define NVIC_VADC0_G2_0       42  /* Analog to Digital Converter Group 2 0                        */
#define NVIC_VADC0_G2_1       43  /* Analog to Digital Converter Group 2 1                        */
#define NVIC_VADC0_G2_2       44  /* Analog to Digital Converter Group 2 2                        */
#define NVIC_VADC0_G2_3       45  /* Analog to Digital Converter Group 2 3                        */
#define NVIC_VADC0_G3_0       46  /* Analog to Digital Converter Group 3 0                        */
#define NVIC_VADC0_G3_1       47  /* Analog to Digital Converter Group 3 1                        */
#define NVIC_VADC0_G3_2       48  /* Analog to Digital Converter Group 3 2                        */
#define NVIC_VADC0_G3_3       49  /* Analog to Digital Converter Group 3 3                        */
#define NVIC_DSD0_0           50  /* Delta Sigma Demodulator 0                                    */
#define NVIC_DSD0_1           51  /* Delta Sigma Demodulator 1                                    */
#define NVIC_DSD0_2           52  /* Delta Sigma Demodulator 2                                    */
#define NVIC_DSD0_3           53  /* Delta Sigma Demodulator 3                                    */
#define NVIC_DSD0_4           54  /* Delta Sigma Demodulator 4                                    */
#define NVIC_DSD0_5           55  /* Delta Sigma Demodulator 5                                    */
#define NVIC_DSD0_6           56  /* Delta Sigma Demodulator 6                                    */
#define NVIC_DSD0_7           57  /* Delta Sigma Demodulator 7                                    */
#define NVIC_DAC0_0           58  /* Digital to Analog Converter 0                                */
#define NVIC_DAC0_1           59  /* Digital to Analog Converter 1                                */
#define NVIC_CCU40_0          60  /* Capture Compare Unit 4 (Module 0) 0                          */
#define NVIC_CCU40_1          61  /* Capture Compare Unit 4 (Module 0) 1                          */
#define NVIC_CCU40_2          62  /* Capture Compare Unit 4 (Module 0) 2                          */
#define NVIC_CCU40_3          63  /* Capture Compare Unit 4 (Module 0) 3                          */
#define NVIC_CCU41_0          64  /* Capture Compare Unit 4 (Module 1) 0                          */
#define NVIC_CCU41_1          65  /* Capture Compare Unit 4 (Module 1) 1                          */
#define NVIC_CCU41_2          66  /* Capture Compare Unit 4 (Module 1) 2                          */
#define NVIC_CCU41_3          67  /* Capture Compare Unit 4 (Module 1) 3                          */
#define NVIC_CCU42_0          68  /* Capture Compare Unit 4 (Module 2) 0                          */
#define NVIC_CCU42_1          69  /* Capture Compare Unit 4 (Module 2) 1                          */
#define NVIC_CCU42_2          70  /* Capture Compare Unit 4 (Module 2) 2                          */
#define NVIC_CCU42_3          71  /* Capture Compare Unit 4 (Module 2) 3                          */
#define NVIC_CCU43_0          72  /* Capture Compare Unit 4 (Module 3) 0                          */
#define NVIC_CCU43_1          73  /* Capture Compare Unit 4 (Module 3) 1                          */
#define NVIC_CCU43_2          74  /* Capture Compare Unit 4 (Module 3) 2                          */
#define NVIC_CCU43_3          75  /* Capture Compare Unit 4 (Module 3) 3                          */
#define NVIC_CCU80_0          76  /* Capture Compare Unit 8 (Module 0) 0                          */
#define NVIC_CCU80_1          77  /* Capture Compare Unit 8 (Module 0) 1                          */
#define NVIC_CCU80_2          78  /* Capture Compare Unit 8 (Module 0) 2                          */
#define NVIC_CCU80_3          79  /* Capture Compare Unit 8 (Module 0) 3                          */
#define NVIC_CCU81_0          80  /* Capture Compare Unit 8 (Module 1) 0                          */
#define NVIC_CCU81_1          81  /* Capture Compare Unit 8 (Module 1) 1                          */
#define NVIC_CCU81_2          82  /* Capture Compare Unit 8 (Module 1) 2                          */
#define NVIC_CCU81_3          83  /* Capture Compare Unit 8 (Module 1) 3                          */
#define NVIC_POSIF0_0         84  /* Position Interface (Module 0) 0                              */
#define NVIC_POSIF0_1         85  /* Position Interface (Module 0) 1                              */
#define NVIC_POSIF1_0         86  /* Position Interface (Module 1) 0                              */
#define NVIC_POSIF1_1         87  /* Position Interface (Module 1) 1                              */
#define NVIC_CAN0_0           92  /* MultiCAN 0                                                   */
#define NVIC_CAN0_1           93  /* MultiCAN 1                                                   */
#define NVIC_CAN0_2           94  /* MultiCAN 2                                                   */
#define NVIC_CAN0_3           95  /* MultiCAN 3                                                   */
#define NVIC_CAN0_4           96  /* MultiCAN 4                                                   */
#define NVIC_CAN0_5           97  /* MultiCAN 5                                                   */
#define NVIC_CAN0_6           98  /* MultiCAN 6                                                   */
#define NVIC_CAN0_7           99  /* MultiCAN 7                                                   */
#define NVIC_USIC0_0         100  /* Universal Serial Interface Channel (Module 0) 0              */
#define NVIC_USIC0_1         101  /* Universal Serial Interface Channel (Module 0) 1              */
#define NVIC_USIC0_2         102  /* Universal Serial Interface Channel (Module 0) 2              */
#define NVIC_USIC0_3         103  /* Universal Serial Interface Channel (Module 0) 3              */
#define NVIC_USIC0_4         104  /* Universal Serial Interface Channel (Module 0) 4              */
#define NVIC_USIC0_5         105  /* Universal Serial Interface Channel (Module 0) 5              */
#define NVIC_USIC1_0         106  /* Universal Serial Interface Channel (Module 1) 0              */
#define NVIC_USIC1_1         107  /* Universal Serial Interface Channel (Module 1) 1              */
#define NVIC_USIC1_2         108  /* Universal Serial Interface Channel (Module 1) 2              */
#define NVIC_USIC1_3         109  /* Universal Serial Interface Channel (Module 1) 3              */
#define NVIC_USIC1_4         110  /* Universal Serial Interface Channel (Module 1) 4              */
#define NVIC_USIC1_5         111  /* Universal Serial Interface Channel (Module 1) 5              */
#define NVIC_USIC2_0         112  /* Universal Serial Interface Channel (Module 2) 0              */
#define NVIC_USIC2_1         113  /* Universal Serial Interface Channel (Module 2) 1              */
#define NVIC_USIC2_2         114  /* Universal Serial Interface Channel (Module 2) 2              */
#define NVIC_USIC2_3         115  /* Universal Serial Interface Channel (Module 2) 3              */
#define NVIC_USIC2_4         116  /* Universal Serial Interface Channel (Module 2) 4              */
#define NVIC_USIC2_5         117  /* Universal Serial Interface Channel (Module 2) 5              */
#define NVIC_LEDTS0_0        118  /* LED and Touch Sense Control Unit (Module 0)                  */
#define NVIC_FCE0_0          120  /* Flexible CRC Engine                                          */
#define NVIC_GPDMA0_0        121  /* General Purpose DMA unit 0                                   */
#define NVIC_SDMMC0_0        122  /* Multi Media Card Interface                                   */
#define NVIC_USB0_0          123  /* Universal Serial Bus                                         */
#define NVIC_ETH0_0          124  /* Ethernet (Module 0)                                          */
#define NVIC_GPDMA1_0        126  /* General Purpose DMA unit 1                                   */

#endif    /* __IOXMC4500_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMII                  0x08
Interrupt1   = HFI                   0x0C
Interrupt2   = MMI                   0x10
Interrupt3   = BFI                   0x14
Interrupt4   = UFI                   0x18
Interrupt5   = SVCI                  0x2C
Interrupt6   = DMI                   0x30
Interrupt7   = PSI                   0x38
Interrupt8   = STI                   0x3C
Interrupt9   = NVIC_SCU_0            0x40
Interrupt10  = NVIC_ERU0_0           0x44
Interrupt11  = NVIC_ERU0_1           0x48
Interrupt12  = NVIC_ERU0_2           0x4C
Interrupt13  = NVIC_ERU0_3           0x50
Interrupt14  = NVIC_ERU1_0           0x54
Interrupt15  = NVIC_ERU1_1           0x58
Interrupt16  = NVIC_ERU1_2           0x5C
Interrupt17  = NVIC_ERU1_3           0x68
Interrupt18  = NVIC_WDT_0            0x6C
Interrupt19  = NVIC_PMU0_0           0x70
Interrupt20  = NVIC_PMU0_1           0x74
Interrupt21  = NVIC_VADC0_C0_0       0x78
Interrupt22  = NVIC_VADC0_C0_1       0x7C
Interrupt23  = NVIC_VADC0_C0_2       0x80
Interrupt24  = NVIC_VADC0_C0_3       0x84
Interrupt25  = NVIC_VADC0_G0_0       0x88
Interrupt26  = NVIC_VADC0_G0_1       0x8C
Interrupt27  = NVIC_VADC0_G0_2       0x90
Interrupt28  = NVIC_VADC0_G0_3       0x94
Interrupt29  = NVIC_VADC0_G1_0       0x98
Interrupt30  = NVIC_VADC0_G1_1       0x9C
Interrupt31  = NVIC_VADC0_G1_2       0xA0
Interrupt32  = NVIC_VADC0_G1_3       0xA4
Interrupt33  = NVIC_VADC0_G2_0       0xA8
Interrupt34  = NVIC_VADC0_G2_1       0xAC
Interrupt35  = NVIC_VADC0_G2_2       0xB0
Interrupt36  = NVIC_VADC0_G2_3       0xB4
Interrupt37  = NVIC_VADC0_G3_0       0xB8
Interrupt38  = NVIC_VADC0_G3_1       0xBC
Interrupt39  = NVIC_VADC0_G3_2       0xC0
Interrupt40  = NVIC_VADC0_G3_3       0xC4
Interrupt41  = NVIC_DSD0_0           0xC8
Interrupt42  = NVIC_DSD0_1           0xCC
Interrupt43  = NVIC_DSD0_2           0xD0
Interrupt44  = NVIC_DSD0_3           0xD4
Interrupt45  = NVIC_DSD0_4           0xD8
Interrupt46  = NVIC_DSD0_5           0xDC
Interrupt47  = NVIC_DSD0_6           0xE0
Interrupt48  = NVIC_DSD0_7           0xE4
Interrupt49  = NVIC_DAC0_0           0xE8
Interrupt50  = NVIC_DAC0_1           0xEC
Interrupt51  = NVIC_CCU40_0          0xF0
Interrupt52  = NVIC_CCU40_1          0xF4
Interrupt53  = NVIC_CCU40_2          0xF8
Interrupt54  = NVIC_CCU40_3          0xFC
Interrupt55  = NVIC_CCU41_0          0x100
Interrupt56  = NVIC_CCU41_1          0x104
Interrupt57  = NVIC_CCU41_2          0x108
Interrupt58  = NVIC_CCU41_3          0x10C
Interrupt59  = NVIC_CCU42_0          0x110
Interrupt60  = NVIC_CCU42_1          0x114
Interrupt61  = NVIC_CCU42_2          0x118
Interrupt62  = NVIC_CCU42_3          0x11C
Interrupt63  = NVIC_CCU43_0          0x120
Interrupt64  = NVIC_CCU43_1          0x124
Interrupt65  = NVIC_CCU43_2          0x128
Interrupt66  = NVIC_CCU43_3          0x12C
Interrupt67  = NVIC_CCU80_0          0x130
Interrupt68  = NVIC_CCU80_1          0x134
Interrupt69  = NVIC_CCU80_2          0x138
Interrupt70  = NVIC_CCU80_3          0x13C
Interrupt71  = NVIC_CCU81_0          0x140
Interrupt72  = NVIC_CCU81_1          0x144
Interrupt73  = NVIC_CCU81_2          0x148
Interrupt74  = NVIC_CCU81_3          0x14C
Interrupt75  = NVIC_POSIF0_0         0x150
Interrupt76  = NVIC_POSIF0_1         0x154
Interrupt77  = NVIC_POSIF1_0         0x158
Interrupt78  = NVIC_POSIF1_1         0x15C
Interrupt79  = NVIC_CAN0_0           0x170
Interrupt80  = NVIC_CAN0_1           0x174
Interrupt81  = NVIC_CAN0_2           0x178
Interrupt82  = NVIC_CAN0_3           0x17C
Interrupt83  = NVIC_CAN0_4           0x180
Interrupt84  = NVIC_CAN0_5           0x184
Interrupt85  = NVIC_CAN0_6           0x188
Interrupt86  = NVIC_CAN0_7           0x18C
Interrupt87  = NVIC_USIC0_0          0x190
Interrupt88  = NVIC_USIC0_1          0x194
Interrupt89  = NVIC_USIC0_2          0x198
Interrupt90  = NVIC_USIC0_3          0x19C
Interrupt91  = NVIC_USIC0_4          0x1A0
Interrupt92  = NVIC_USIC0_5          0x1A4
Interrupt93  = NVIC_USIC1_0          0x1A8
Interrupt94  = NVIC_USIC1_1          0x1AC
Interrupt95  = NVIC_USIC1_2          0x1B0
Interrupt96  = NVIC_USIC1_3          0x1B4
Interrupt97  = NVIC_USIC1_4          0x1B8
Interrupt98  = NVIC_USIC1_5          0x1BC
Interrupt99  = NVIC_USIC2_0          0x1C0
Interrupt100 = NVIC_USIC2_1          0x1C4
Interrupt101 = NVIC_USIC2_2          0x1C8
Interrupt102 = NVIC_USIC2_3          0x1CC
Interrupt103 = NVIC_USIC2_4          0x1D0
Interrupt104 = NVIC_USIC2_5          0x1D4
Interrupt105 = NVIC_LEDTS0_0         0x1D8
Interrupt106 = NVIC_FCE0_0           0x1E0
Interrupt107 = NVIC_GPDMA0_0         0x1E4
Interrupt108 = NVIC_SDMMC0_0         0x1E8
Interrupt109 = NVIC_USB0_0           0x1EC
Interrupt110 = NVIC_ETH0_0           0x1F0
Interrupt111 = NVIC_GPDMA1_0         0x1F8

###DDF-INTERRUPT-END###*/
