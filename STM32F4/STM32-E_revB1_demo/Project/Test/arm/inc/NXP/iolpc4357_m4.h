/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC4357 Cortex-M4 Core
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50467 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC4357_M4_H
#define __IOLPC4357_M4_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC4357_M4 SPECIAL FUNCTION REGISTERS
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

/* Interrupt Controller Type Register */
typedef struct {
  __REG32  INTLINESNUM    : 5;
  __REG32                 :27;
} __nvic_ictr_bits;


/* SysTick Control and Status Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  TICKINT        : 1;
  __REG32  CLKSOURCE      : 1;
  __REG32                 :13;
  __REG32  COUNTFLAG      : 1;
  __REG32                 :15;
} __systick_csr_bits;

/* SysTick Reload Value Register */
typedef struct {
  __REG32  RELOAD         :24;
  __REG32                 : 8;
} __systick_rvr_bits;

/* SysTick Current Value Register */
typedef struct {
  __REG32  CURRENT        :24;
  __REG32                 : 8;
} __systick_cvr_bits;

/* SysTick Calibration Value Register */
typedef struct {
  __REG32  TENMS          :24;
  __REG32                 : 6;
  __REG32  SKEW           : 1;
  __REG32  NOREF          : 1;
} __systick_calvr_bits;


/* Interrupt Set-Enable Registers 0-31 */
typedef struct {
  __REG32  ISE0        : 1;
  __REG32  ISE1        : 1;
  __REG32  ISE2        : 1;
  __REG32  ISE3        : 1;
  __REG32  ISE4        : 1;
  __REG32  ISE5        : 1;
  __REG32  ISE6        : 1;
  __REG32  ISE7        : 1;
  __REG32  ISE8        : 1;
  __REG32  ISE9        : 1;
  __REG32  ISE10       : 1;
  __REG32  ISE11       : 1;
  __REG32  ISE12       : 1;
  __REG32  ISE13       : 1;
  __REG32  ISE14       : 1;
  __REG32  ISE15       : 1;
  __REG32  ISE16       : 1;
  __REG32  ISE17       : 1;
  __REG32  ISE18       : 1;
  __REG32  ISE19       : 1;
  __REG32  ISE20       : 1;
  __REG32  ISE21       : 1;
  __REG32  ISE22       : 1;
  __REG32  ISE23       : 1;
  __REG32  ISE24       : 1;
  __REG32  ISE25       : 1;
  __REG32  ISE26       : 1;
  __REG32  ISE27       : 1;
  __REG32  ISE28       : 1;
  __REG32  ISE29       : 1;
  __REG32  ISE30       : 1;
  __REG32  ISE31       : 1;
} __nvic_iser0_bits;

/* Interrupt Set-Enable Registers 0-31 */
typedef struct {
  __REG32  ISE32        : 1;
  __REG32  ISE33        : 1;
  __REG32  ISE34        : 1;
  __REG32  ISE35        : 1;
  __REG32  ISE36        : 1;
  __REG32  ISE37        : 1;
  __REG32  ISE38        : 1;
  __REG32  ISE39        : 1;
  __REG32  ISE40        : 1;
  __REG32  ISE41        : 1;
  __REG32  ISE42        : 1;
  __REG32  ISE43        : 1;
  __REG32  ISE44        : 1;
  __REG32  ISE45        : 1;
  __REG32  ISE46        : 1;
  __REG32  ISE47        : 1;
  __REG32  ISE48        : 1;
  __REG32  ISE49        : 1;
  __REG32  ISE50        : 1;
  __REG32  ISE51        : 1;
  __REG32  ISE52        : 1;
  __REG32               :11;
} __nvic_iser1_bits;

/* Interrupt Clear-Enable Registers 0-31 */
typedef struct {
  __REG32  ICE0        : 1;
  __REG32  ICE1        : 1;
  __REG32  ICE2        : 1;
  __REG32  ICE3        : 1;
  __REG32  ICE4        : 1;
  __REG32  ICE5        : 1;
  __REG32  ICE6        : 1;
  __REG32  ICE7        : 1;
  __REG32  ICE8        : 1;
  __REG32  ICE9        : 1;
  __REG32  ICE10       : 1;
  __REG32  ICE11       : 1;
  __REG32  ICE12       : 1;
  __REG32  ICE13       : 1;
  __REG32  ICE14       : 1;
  __REG32  ICE15       : 1;
  __REG32  ICE16       : 1;
  __REG32  ICE17       : 1;
  __REG32  ICE18       : 1;
  __REG32  ICE19       : 1;
  __REG32  ICE20       : 1;
  __REG32  ICE21       : 1;
  __REG32  ICE22       : 1;
  __REG32  ICE23       : 1;
  __REG32  ICE24       : 1;
  __REG32  ICE25       : 1;
  __REG32  ICE26       : 1;
  __REG32  ICE27       : 1;
  __REG32  ICE28       : 1;
  __REG32  ICE29       : 1;
  __REG32  ICE30       : 1;
  __REG32  ICE31       : 1;
} __nvic_icer0_bits;

/* Interrupt Clear-Enable Registers 0-31 */
typedef struct {
  __REG32  ICE32        : 1;
  __REG32  ICE33        : 1;
  __REG32  ICE34        : 1;
  __REG32  ICE35        : 1;
  __REG32  ICE36        : 1;
  __REG32  ICE37        : 1;
  __REG32  ICE38        : 1;
  __REG32  ICE39        : 1;
  __REG32  ICE40        : 1;
  __REG32  ICE41        : 1;
  __REG32  ICE42        : 1;
  __REG32  ICE43        : 1;
  __REG32  ICE44        : 1;
  __REG32  ICE45        : 1;
  __REG32  ICE46        : 1;
  __REG32  ICE47        : 1;
  __REG32  ICE48        : 1;
  __REG32  ICE49        : 1;
  __REG32  ICE50        : 1;
  __REG32  ICE51        : 1;
  __REG32  ICE52        : 1;
  __REG32               :11;
} __nvic_icer1_bits;

/* Interrupt Set-Pending Register 0-31 */
typedef struct {
  __REG32  ISP0        : 1;
  __REG32  ISP1        : 1;
  __REG32  ISP2        : 1;
  __REG32  ISP3        : 1;
  __REG32  ISP4        : 1;
  __REG32  ISP5        : 1;
  __REG32  ISP6        : 1;
  __REG32  ISP7        : 1;
  __REG32  ISP8        : 1;
  __REG32  ISP9        : 1;
  __REG32  ISP10       : 1;
  __REG32  ISP11       : 1;
  __REG32  ISP12       : 1;
  __REG32  ISP13       : 1;
  __REG32  ISP14       : 1;
  __REG32  ISP15       : 1;
  __REG32  ISP16       : 1;
  __REG32  ISP17       : 1;
  __REG32  ISP18       : 1;
  __REG32  ISP19       : 1;
  __REG32  ISP20       : 1;
  __REG32  ISP21       : 1;
  __REG32  ISP22       : 1;
  __REG32  ISP23       : 1;
  __REG32  ISP24       : 1;
  __REG32  ISP25       : 1;
  __REG32  ISP26       : 1;
  __REG32  ISP27       : 1;
  __REG32  ISP28       : 1;
  __REG32  ISP29       : 1;
  __REG32  ISP30       : 1;
  __REG32  ISP31       : 1;
} __nvic_ispr0_bits;

/* Interrupt Clear-Enable Registers 0-31 */
typedef struct {
  __REG32  ISP32        : 1;
  __REG32  ISP33        : 1;
  __REG32  ISP34        : 1;
  __REG32  ISP35        : 1;
  __REG32  ISP36        : 1;
  __REG32  ISP37        : 1;
  __REG32  ISP38        : 1;
  __REG32  ISP39        : 1;
  __REG32  ISP40        : 1;
  __REG32  ISP41        : 1;
  __REG32  ISP42        : 1;
  __REG32  ISP43        : 1;
  __REG32  ISP44        : 1;
  __REG32  ISP45        : 1;
  __REG32  ISP46        : 1;
  __REG32  ISP47        : 1;
  __REG32  ISP48        : 1;
  __REG32  ISP49        : 1;
  __REG32  ISP50        : 1;
  __REG32  ISP51        : 1;
  __REG32  ISP52        : 1;
  __REG32               :11;
} __nvic_ispr1_bits;

/* Interrupt Clear-Pending Register 0-31 */
typedef struct {
  __REG32  ICP0        : 1;
  __REG32  ICP1        : 1;
  __REG32  ICP2        : 1;
  __REG32  ICP3        : 1;
  __REG32  ICP4        : 1;
  __REG32  ICP5        : 1;
  __REG32  ICP6        : 1;
  __REG32  ICP7        : 1;
  __REG32  ICP8        : 1;
  __REG32  ICP9        : 1;
  __REG32  ICP10       : 1;
  __REG32  ICP11       : 1;
  __REG32  ICP12       : 1;
  __REG32  ICP13       : 1;
  __REG32  ICP14       : 1;
  __REG32  ICP15       : 1;
  __REG32  ICP16       : 1;
  __REG32  ICP17       : 1;
  __REG32  ICP18       : 1;
  __REG32  ICP19       : 1;
  __REG32  ICP20       : 1;
  __REG32  ICP21       : 1;
  __REG32  ICP22       : 1;
  __REG32  ICP23       : 1;
  __REG32  ICP24       : 1;
  __REG32  ICP25       : 1;
  __REG32  ICP26       : 1;
  __REG32  ICP27       : 1;
  __REG32  ICP28       : 1;
  __REG32  ICP29       : 1;
  __REG32  ICP30       : 1;
  __REG32  ICP31       : 1;
} __nvic_icpr0_bits;

/* Interrupt Clear-Pending Register 0-31 */
typedef struct {
  __REG32  ICP32        : 1;
  __REG32  ICP33        : 1;
  __REG32  ICP34        : 1;
  __REG32  ICP35        : 1;
  __REG32  ICP36        : 1;
  __REG32  ICP37        : 1;
  __REG32  ICP38        : 1;
  __REG32  ICP39        : 1;
  __REG32  ICP40        : 1;
  __REG32  ICP41        : 1;
  __REG32  ICP42        : 1;
  __REG32  ICP43        : 1;
  __REG32  ICP44        : 1;
  __REG32  ICP45        : 1;
  __REG32  ICP46        : 1;
  __REG32  ICP47        : 1;
  __REG32  ICP48        : 1;
  __REG32  ICP49        : 1;
  __REG32  ICP50        : 1;
  __REG32  ICP51        : 1;
  __REG32  ICP52        : 1;
  __REG32               :11;
} __nvic_icpr1_bits;

/* Active Bit Register 0-31 */
typedef struct {
  __REG32  IABP0        : 1;
  __REG32  IABP1        : 1;
  __REG32  IABP2        : 1;
  __REG32  IABP3        : 1;
  __REG32  IABP4        : 1;
  __REG32  IABP5        : 1;
  __REG32  IABP6        : 1;
  __REG32  IABP7        : 1;
  __REG32  IABP8        : 1;
  __REG32  IABP9        : 1;
  __REG32  IABP10       : 1;
  __REG32  IABP11       : 1;
  __REG32  IABP12       : 1;
  __REG32  IABP13       : 1;
  __REG32  IABP14       : 1;
  __REG32  IABP15       : 1;
  __REG32  IABP16       : 1;
  __REG32  IABP17       : 1;
  __REG32  IABP18       : 1;
  __REG32  IABP19       : 1;
  __REG32  IABP20       : 1;
  __REG32  IABP21       : 1;
  __REG32  IABP22       : 1;
  __REG32  IABP23       : 1;
  __REG32  IABP24       : 1;
  __REG32  IABP25       : 1;
  __REG32  IABP26       : 1;
  __REG32  IABP27       : 1;
  __REG32  IABP28       : 1;
  __REG32  IABP29       : 1;
  __REG32  IABP30       : 1;
  __REG32  IABP31       : 1;
} __nvic_iabr0_bits;

/* Active Bit Register 0-31 */
typedef struct {
  __REG32  IABP32        : 1;
  __REG32  IABP33        : 1;
  __REG32  IABP34        : 1;
  __REG32  IABP35        : 1;
  __REG32  IABP36        : 1;
  __REG32  IABP37        : 1;
  __REG32  IABP38        : 1;
  __REG32  IABP39        : 1;
  __REG32  IABP40        : 1;
  __REG32  IABP41        : 1;
  __REG32  IABP42        : 1;
  __REG32  IABP43        : 1;
  __REG32  IABP44        : 1;
  __REG32  IABP45        : 1;
  __REG32  IABP46        : 1;
  __REG32  IABP47        : 1;
  __REG32  IABP48        : 1;
  __REG32  IABP49        : 1;
  __REG32  IABP50        : 1;
  __REG32  IABP51        : 1;
  __REG32  IABP52        : 1;
  __REG32                :11;
} __nvic_iabr1_bits;


/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32  IP0            : 8;
  __REG32  IP1            : 8;
  __REG32  IP2            : 8;
  __REG32  IP3            : 8;
} __nvic_ipr0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32  IP4            : 8;
  __REG32  IP5            : 8;
  __REG32  IP6            : 8;
  __REG32  IP7            : 8;
} __nvic_ipr1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32  IP8           : 8;
  __REG32  IP9           : 8;
  __REG32  IP10          : 8;
  __REG32  IP11          : 8;
} __nvic_ipr2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32  IP12          : 8;
  __REG32  IP13          : 8;
  __REG32  IP14          : 8;
  __REG32  IP15          : 8;
} __nvic_ipr3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32  IP16          : 8;
  __REG32  IP17          : 8;
  __REG32  IP18          : 8;
  __REG32  IP19          : 8;
} __nvic_ipr4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32  IP20          : 8;
  __REG32  IP21          : 8;
  __REG32  IP22          : 8;
  __REG32  IP23          : 8;
} __nvic_ipr5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32  IP24          : 8;
  __REG32  IP25          : 8;
  __REG32  IP26          : 8;
  __REG32  IP27          : 8;
} __nvic_ipr6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32  IP28          : 8;
  __REG32  IP29          : 8;
  __REG32  IP30          : 8;
  __REG32  IP31          : 8;
} __nvic_ipr7_bits;

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32  IP32          : 8;
  __REG32  IP33          : 8;
  __REG32  IP34          : 8;
  __REG32  IP35          : 8;
} __nvic_ipr8_bits;

/* Interrupt Priority Registers 36-39 */
typedef struct {
  __REG32  IP36          : 8;
  __REG32  IP37          : 8;
  __REG32  IP38          : 8;
  __REG32  IP39          : 8;
} __nvic_ipr9_bits;

/* Interrupt  Priority Registers 40-43 */
typedef struct {
  __REG32  IP40          : 8;
  __REG32  IP41          : 8;
  __REG32  IP42          : 8;
  __REG32  IP43          : 8;
} __nvic_ipr10_bits;

/* Interrupt Priority Registers 44-47 */
typedef struct {
  __REG32  IP44          : 8;
  __REG32  IP45          : 8;
  __REG32  IP46          : 8;
  __REG32  IP47          : 8;
} __nvic_ipr11_bits;

/* Interrupt Priority Registers 48-51 */
typedef struct {
  __REG32  IP48          : 8;
  __REG32  IP49          : 8;
  __REG32  IP50          : 8;
  __REG32  IP51          : 8;
} __nvic_ipr12_bits;

/* Interrupt Priority Register 52 */
typedef struct {
  __REG32  IP52          : 8;
  __REG32                :24;
} __nvic_ipr13_bits;


/* CPU ID Base Register */
typedef struct {
  __REG32  REVISION       : 4;
  __REG32  PARTNO         :12;
  __REG32  ARCHITECTURE   : 4;
  __REG32  VARIANT        : 4;
  __REG32  IMPLEMENTER    : 8;
} __cpuid_bits;

/* Interrupt Control State Register */
typedef struct {
  __REG32  VECTACTIVE     : 9;
  __REG32                 : 2;
  __REG32  RETTOBASE      : 1;
  __REG32  VECTPENDING    : 9;
  __REG32                 : 1;
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
  __REG32  TBLOFF         :25;
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
  __REG32  MLSPERR        : 1;
  __REG32                 : 1;
  __REG32  MMARVALID      : 1;
  __REG32  IBUSERR        : 1;
  __REG32  PRECISERR      : 1;
  __REG32  IMPRECISERR    : 1;
  __REG32  UNSTKERR       : 1;
  __REG32  STKERR         : 1;
  __REG32  LSPERR         : 1;
  __REG32                 : 1;
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

/* Debug Halting Control and Status Register */
typedef union {
  /* DHCSR */
  /* DHCSR_READ */
  struct {
  __REG32  C_DEBUGEN         : 1;
  __REG32  C_HALT            : 1;
  __REG32  C_STEP            : 1;
  __REG32  C_MASKINTS        : 1;
  __REG32                    : 1;
  __REG32  C_SNAPSTALL       : 1;
  __REG32                    :10;
  __REG32  S_REGRDY          : 1;
  __REG32  S_HALT            : 1;
  __REG32  S_SLEEP           : 1;
  __REG32  S_LOCKUP          : 1;
  __REG32                    : 4;
  __REG32  S_RETIRE_ST       : 1;
  __REG32  S_RESET_ST        : 1;
  __REG32                    : 6;
  };
  /* DHCSR_WRITE */
  struct {
  __REG32  C_DEBUGEN         : 1;
  __REG32  C_HALT            : 1;
  __REG32  C_STEP            : 1;
  __REG32  C_MASKINTS        : 1;
  __REG32                    : 1;
  __REG32  C_SNAPSTALL       : 1;
  __REG32                    :10;
  __REG32  DBGKEY            :16;
  } __dhcsr_write_bits;
} __dhcsr_bits;

/* Debug Core Register Selector Register */

#define DCRSR_REGSEL      ((uint32_t)0x0000007F)
#define DCRSR_REGWnR      ((uint32_t)0x00010000)

/* Debug Exception and Monitor Control Register */
typedef struct {
  __REG32  VC_CORERESET      : 1;
  __REG32                    : 3;
  __REG32  VC_MMERR          : 1;
  __REG32  VC_NOCPERR        : 1;
  __REG32  VC_CHKERR         : 1;
  __REG32  VC_STATERR        : 1;
  __REG32  VC_BUSERR         : 1;
  __REG32  VC_INTERR         : 1;
  __REG32  VC_HARDERR        : 1;
  __REG32                    : 5;
  __REG32  MON_EN            : 1;
  __REG32  MON_PEND          : 1;
  __REG32  MON_STEP          : 1;
  __REG32  MON_REQ           : 1;
  __REG32                    : 4;
  __REG32  TRCENA            : 1;
  __REG32                    : 7;
} __demcr_bits;

/* Software Trigger Interrupt Register */

#define STIR_INTID      ((uint32_t)0x000001FF)

/* Floating Point Context Control Register */
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
} __fpccr_bits;

/* Floating Point Context Adress Register */
typedef struct {
  __REG32                  : 3;
  __REG32  ADDRESS         :29;
} __fpcar_bits;

/* Floating Point Context Adress Register */
typedef struct {
  __REG32                  :22;
  __REG32  RMode           : 2;
  __REG32  FZ              : 1;
  __REG32  DN              : 1;
  __REG32  AHP             : 1;
  __REG32                  : 5;
} __fpdscr_bits;

/* Media and FP Feature Register 0 */
typedef struct {
  __REG32  A_SIMD_registers      : 4;
  __REG32  Single_precision      : 4;
  __REG32  Double_precision      : 4;
  __REG32  FP_Exception_trapping : 4;
  __REG32  Divide                : 4;
  __REG32  Square_root           : 4;
  __REG32  Short_vectors         : 4;
  __REG32  FP_rounding_modes     : 4;
} __mvfr0_bits;

/* Media and FP Feature Register 1 */
typedef struct {
  __REG32  FtZ_mode              : 4;
  __REG32  D_NaN_mode            : 4;
  __REG32                        :16;
  __REG32  FP_HPFP               : 4;
  __REG32  FP_fused_MAC          : 4;
} __mvfr1_bits;

/* Coprocessor access control register */
typedef struct {
  __REG32  CP0            : 2;
  __REG32  CP1            : 2;
  __REG32  CP2            : 2;
  __REG32  CP3            : 2;
  __REG32  CP4            : 2;
  __REG32  CP5            : 2;
  __REG32  CP6            : 2;
  __REG32  CP7            : 2;
  __REG32                 : 4;
  __REG32  CP10           : 2;
  __REG32  CP11           : 2;
  __REG32                 : 8;
} __cpacr_bits;

/* Level configuration register */
typedef struct {
  __REG32  WAKEUP0_L      : 1;
  __REG32  WAKEUP1_L      : 1;
  __REG32  WAKEUP2_L      : 1;
  __REG32  WAKEUP3_L      : 1;
  __REG32  ATIMER_L       : 1;
  __REG32  RTC_L          : 1;
  __REG32  BOD_L          : 1;
  __REG32  WWDT_L         : 1;
  __REG32  ETH_L          : 1;
  __REG32  USB0_L         : 1;
  __REG32  USB1_L         : 1;
  __REG32  SDMMC_L        : 1;
  __REG32  CAN_L          : 1;
  __REG32  TIM2_L         : 1;
  __REG32  TIM6_L         : 1;
  __REG32  QEI_L          : 1;
  __REG32  TIM14_L        : 1;
  __REG32                 : 2;
  __REG32  RESET_L        : 1;
  __REG32                 :12;
} __er_hilo_bits;

/* Edge configuration register */
typedef struct {
  __REG32  WAKEUP0_E      : 1;
  __REG32  WAKEUP1_E      : 1;
  __REG32  WAKEUP2_E      : 1;
  __REG32  WAKEUP3_E      : 1;
  __REG32  ATIMER_E       : 1;
  __REG32  RTC_E          : 1;
  __REG32  BOD_E          : 1;
  __REG32  WWDT_E         : 1;
  __REG32  ETH_E          : 1;
  __REG32  USB0_E         : 1;
  __REG32  USB1_E         : 1;
  __REG32  SDMMC_E        : 1;
  __REG32  CAN_E          : 1;
  __REG32  TIM2_E         : 1;
  __REG32  TIM6_E         : 1;
  __REG32  QEI_E          : 1;
  __REG32  TIM14_E        : 1;
  __REG32                 : 2;
  __REG32  RESET_E        : 1;
  __REG32                 :12;
} __er_edge_bits;

/* Interrupt clear enable register */

#define ER_CLR_EN_WAKEUP0_CLREN     (0x00000001UL)
#define ER_CLR_EN_WAKEUP1_CLREN     (0x00000002UL)
#define ER_CLR_EN_WAKEUP2_CLREN     (0x00000004UL)
#define ER_CLR_EN_WAKEUP3_CLREN     (0x00000008UL)
#define ER_CLR_EN_ATIMER_CLREN      (0x00000010UL)
#define ER_CLR_EN_RTC_CLREN         (0x00000020UL)
#define ER_CLR_EN_BOD_CLREN         (0x00000040UL)
#define ER_CLR_EN_WWDT_CLREN        (0x00000080UL)
#define ER_CLR_EN_ETH_CLREN         (0x00000100UL)
#define ER_CLR_EN_USB0_CLREN        (0x00000200UL)
#define ER_CLR_EN_USB1_CLREN        (0x00000400UL)
#define ER_CLR_EN_SDMMC_CLREN       (0x00000800UL)
#define ER_CLR_EN_CAN_CLREN         (0x00001000UL)
#define ER_CLR_EN_TIM2_CLREN        (0x00002000UL)
#define ER_CLR_EN_TIM6_CLREN        (0x00004000UL)
#define ER_CLR_EN_QEI_CLREN         (0x00008000UL)
#define ER_CLR_EN_TIM14_CLREN       (0x00010000UL)
#define ER_CLR_EN_RESET_CLREN       (0x00080000UL)

/* Interrupt set enable register */

#define ER_SET_EN_WAKEUP0_SETEN     (0x00000001UL)
#define ER_SET_EN_WAKEUP1_SETEN     (0x00000002UL)
#define ER_SET_EN_WAKEUP2_SETEN     (0x00000004UL)
#define ER_SET_EN_WAKEUP3_SETEN     (0x00000008UL)
#define ER_SET_EN_ATIMER_SETEN      (0x00000010UL)
#define ER_SET_EN_RTC_SETEN         (0x00000020UL)
#define ER_SET_EN_BOD_SETEN         (0x00000040UL)
#define ER_SET_EN_WWDT_SETEN        (0x00000080UL)
#define ER_SET_EN_ETH_SETEN         (0x00000100UL)
#define ER_SET_EN_USB0_SETEN        (0x00000200UL)
#define ER_SET_EN_USB1_SETEN        (0x00000400UL)
#define ER_SET_EN_SDMMC_SETEN       (0x00000800UL)
#define ER_SET_EN_CAN_SETEN         (0x00001000UL)
#define ER_SET_EN_TIM2_SETEN        (0x00002000UL)
#define ER_SET_EN_TIM6_SETEN        (0x00004000UL)
#define ER_SET_EN_QEI_SETEN         (0x00008000UL)
#define ER_SET_EN_TIM14_SETEN       (0x00010000UL)
#define ER_SET_EN_RESET_SETEN       (0x00080000UL)

/* Interrupt status register */
typedef struct {
  __REG32  WAKEUP0_ST     : 1;
  __REG32  WAKEUP1_ST     : 1;
  __REG32  WAKEUP2_ST     : 1;
  __REG32  WAKEUP3_ST     : 1;
  __REG32  ATIMER_ST      : 1;
  __REG32  RTC_ST         : 1;
  __REG32  BOD_ST         : 1;
  __REG32  WWDT_ST        : 1;
  __REG32  ETH_ST         : 1;
  __REG32  USB0_ST        : 1;
  __REG32  USB1_ST        : 1;
  __REG32  SDMMC_ST       : 1;
  __REG32  CAN_ST         : 1;
  __REG32  TIM2_ST        : 1;
  __REG32  TIM6_ST        : 1;
  __REG32  QEI_ST         : 1;
  __REG32  TIM14_ST       : 1;
  __REG32                 : 2;
  __REG32  RESET_ST       : 1;
  __REG32                 :12;
} __er_status_bits;

/* Event enable register */
typedef struct {
  __REG32  WAKEUP0_EN     : 1;
  __REG32  WAKEUP1_EN     : 1;
  __REG32  WAKEUP2_EN     : 1;
  __REG32  WAKEUP3_EN     : 1;
  __REG32  ATIMER_EN      : 1;
  __REG32  RTC_EN         : 1;
  __REG32  BOD_EN         : 1;
  __REG32  WWDT_EN        : 1;
  __REG32  ETH_EN         : 1;
  __REG32  USB0_EN        : 1;
  __REG32  USB1_EN        : 1;
  __REG32  SDMMC_EN       : 1;
  __REG32  CAN_EN         : 1;
  __REG32  TIM2_EN        : 1;
  __REG32  TIM6_EN        : 1;
  __REG32  QEI_EN         : 1;
  __REG32  TIM14_EN       : 1;
  __REG32                 : 2;
  __REG32  RESET_EN       : 1;
  __REG32                 :12;
} __er_enable_bits;

/* Interrupt clear status register */

#define ER_CLR_STAT_WAKEUP0_CLRST     (0x00000001UL)
#define ER_CLR_STAT_WAKEUP1_CLRST     (0x00000002UL)
#define ER_CLR_STAT_WAKEUP2_CLRST     (0x00000004UL)
#define ER_CLR_STAT_WAKEUP3_CLRST     (0x00000008UL)
#define ER_CLR_STAT_ATIMER_CLRST      (0x00000010UL)
#define ER_CLR_STAT_RTC_CLRST         (0x00000020UL)
#define ER_CLR_STAT_BOD_CLRST         (0x00000040UL)
#define ER_CLR_STAT_WWDT_CLRST        (0x00000080UL)
#define ER_CLR_STAT_ETH_CLRST         (0x00000100UL)
#define ER_CLR_STAT_USB0_CLRST        (0x00000200UL)
#define ER_CLR_STAT_USB1_CLRST        (0x00000400UL)
#define ER_CLR_STAT_SDMMC_CLRST       (0x00000800UL)
#define ER_CLR_STAT_CAN_CLRST         (0x00001000UL)
#define ER_CLR_STAT_TIM2_CLRST        (0x00002000UL)
#define ER_CLR_STAT_TIM6_CLRST        (0x00004000UL)
#define ER_CLR_STAT_QEI_CLRST         (0x00008000UL)
#define ER_CLR_STAT_TIM14_CLRST       (0x00010000UL)
#define ER_CLR_STAT_RESET_CLRST       (0x00080000UL)

/* Interrupt set status register */

#define ER_SET_STAT_WAKEUP0_SETST     (0x00000001UL)
#define ER_SET_STAT_WAKEUP1_SETST     (0x00000002UL)
#define ER_SET_STAT_WAKEUP2_SETST     (0x00000004UL)
#define ER_SET_STAT_WAKEUP3_SETST     (0x00000008UL)
#define ER_SET_STAT_ATIMER_SETST      (0x00000010UL)
#define ER_SET_STAT_RTC_SETST         (0x00000020UL)
#define ER_SET_STAT_BOD_SETST         (0x00000040UL)
#define ER_SET_STAT_WWDT_SETST        (0x00000080UL)
#define ER_SET_STAT_ETH_SETST         (0x00000100UL)
#define ER_SET_STAT_USB0_SETST        (0x00000200UL)
#define ER_SET_STAT_USB1_SETST        (0x00000400UL)
#define ER_SET_STAT_SDMMC_SETST       (0x00000800UL)
#define ER_SET_STAT_CAN_SETST         (0x00001000UL)
#define ER_SET_STAT_TIM2_SETST        (0x00002000UL)
#define ER_SET_STAT_TIM6_SETST        (0x00004000UL)
#define ER_SET_STAT_QEI_SETST         (0x00008000UL)
#define ER_SET_STAT_TIM14_SETST       (0x00010000UL)
#define ER_SET_STAT_RESET_SETST       (0x00080000UL)

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
  __REG32                 : 29;
} __mpu_ctrl_bits;

/* MPU Region Number register */
typedef struct {
  __REG32  REGION         : 8;
  __REG32                 : 24;
} __mpu_rnr_bits;

/* MPU Region Base Address register */
typedef struct {
  __REG32  REGION         : 4;
  __REG32  VALID          : 1;
  __REG32  ADDR           : 27;
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


/* CREG0 control register */
typedef struct {
  __REG32  EN1KHZ         : 1;
  __REG32  EN32KHZ        : 1;
  __REG32  RESET32KHZ     : 1;
  __REG32  PD32KHZ        : 1;
  __REG32                 : 1;
  __REG32  USB0PHY        : 1;
  __REG32  ALARMCTRL      : 2;
  __REG32  BODLVL1        : 2;
  __REG32  BODLVL2        : 2;
  __REG32                 : 2;
  __REG32  WAKEUP0CTRL    : 2;
  __REG32  WAKEUP1CTRL    : 2;
  __REG32                 :14;
} __creg_creg0_bits;

/* Power mode control register */
/*
typedef struct {
  __REG32  PWRCTRL        : 9;
  __REG32                 : 6;
  __REG32  DYNAMICPWRCTRL : 1;
  __REG32                 :16;
} __creg_pmucon_bits;
*/

/* ARM Cortex-M4 Memory mapping register */
typedef struct {
  __REG32                 :12;
  __REG32  M4MAP          :20;
} __creg_m4memmap_bits;

/* CREG5 control register */
typedef struct {
  __REG32                 : 6;
  __REG32  M4TAPSEL       : 1;
  __REG32                 :25;
} __creg_creg5_bits;

/* DMA muxing register */
typedef struct {
  __REG32  DMAMUXCH0      : 2;
  __REG32  DMAMUXCH1      : 2;
  __REG32  DMAMUXCH2      : 2;
  __REG32  DMAMUXCH3      : 2;
  __REG32  DMAMUXCH4      : 2;
  __REG32  DMAMUXCH5      : 2;
  __REG32  DMAMUXCH6      : 2;
  __REG32  DMAMUXCH7      : 2;
  __REG32  DMAMUXCH8      : 2;
  __REG32  DMAMUXCH9      : 2;
  __REG32  DMAMUXCH10     : 2;
  __REG32  DMAMUXCH11     : 2;
  __REG32  DMAMUXCH12     : 2;
  __REG32  DMAMUXCH13     : 2;
  __REG32  DMAMUXCH14     : 2;
  __REG32  DMAMUXCH15     : 2;
} __creg_dmamux_bits;

/* ETB SRAM configuration register */
typedef struct {
  __REG32  ETB            : 1;
  __REG32                 :31;
} __creg_etbcfg_bits;

/* CREG6 control register */
typedef struct {
  __REG32  ETHMODE             : 3;
  __REG32                      : 1;
  __REG32  CTOUTCTRL           : 1;
  __REG32                      : 7;
  __REG32  I2S0_TX_SCK_IN_SEL  : 1;
  __REG32  I2S0_RX_SCK_IN_SEL  : 1;
  __REG32  I2S1_TX_SCK_IN_SEL  : 1;
  __REG32  I2S1_RX_SCK_IN_SEL  : 1;
  __REG32  EMC_CLK_SEL         : 1;
  __REG32                      :15;
} __creg_creg6_bits;

/* Cortex-M4 TXEV event clear register */
typedef struct {
  __REG32  TXEVCLR        : 1;
  __REG32                 :31;
} __creg_m4txevent_bits;

/* Cortex-M0 TXEV event clear register */
typedef struct {
  __REG32  TXEVCLR        : 1;
  __REG32                 :31;
} __creg_m0txevent_bits;

/* ARM Cortex-M0 memory mapping register */
typedef struct {
  __REG32                 :12;
  __REG32  M0APPMAP       :20;
} __creg_m0appmemmap_bits;


/* Hardware sleep event enable register */
typedef struct {
  __REG32  ENA_EVENT0     : 1;
  __REG32                 :31;
} __pmc_pd0_sleep0_hw_ena_bits;

/* Sleep power mode register */
typedef struct {
  __REG32  PWR_STATE      :32;
} __pmc_pd0_sleep0_mode_bits;


/* FREQ_MON register */
typedef struct {
  __REG32  RCNT           : 9;
  __REG32  FCNT           :14;
  __REG32  MEAS           : 1;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_freq_mon_bits;

/* XTAL_OSC_CTRL register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  BYPASS         : 1;
  __REG32  HF             : 1;
  __REG32                 :29;
} __cgu_xtal_osc_ctrl_bits;

/* PLL0USB_STAT register */
typedef struct {
  __REG32  LOCK           : 1;
  __REG32  FR             : 1;
  __REG32                 :30;
} __cgu_pll0usb_stat_bits;

/* PLL0USB control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32  BYPASS         : 1;
  __REG32  DIRECTI        : 1;
  __REG32  DIRECTO        : 1;
  __REG32  CLKEN          : 1;
  __REG32                 : 1;
  __REG32  FRM            : 1;
  __REG32                 : 4;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_pll0usb_ctrl_bits;

/* PLL0USB_MDIV register */
typedef struct {
  __REG32  MDEC           :17;
  __REG32  SELP           : 5;
  __REG32  SELI           : 6;
  __REG32  SELR           : 4;
} __cgu_pll0usb_mdiv_bits;

/* PLL0USB NP-divider register */
typedef struct {
  __REG32  PDEC           : 7;
  __REG32                 : 5;
  __REG32  NDEC           :10;
  __REG32                 :10;
} __cgu_pll0usb_np_div_bits;

/* PLL0AUDIO_STAT register */
typedef struct {
  __REG32  LOCK           : 1;
  __REG32  FR             : 1;
  __REG32                 :30;
} __cgu_pll0audio_stat_bits;

/* PLL0AUDIO control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32  BYPASS         : 1;
  __REG32  DIRECTI        : 1;
  __REG32  DIRECTO        : 1;
  __REG32  CLKEN          : 1;
  __REG32                 : 1;
  __REG32  FRM            : 1;
  __REG32                 : 4;
  __REG32  AUTOBLOCK      : 1;
  __REG32  PLLFRAQ_REQ    : 1;
  __REG32  SEL_EXT        : 1;
  __REG32  MOD_PD         : 1;
  __REG32                 : 9;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_pll0audio_ctrl_bits;

/* PLL0AUDIO_MDIV register */
typedef struct {
  __REG32  MDEC           :17;
  __REG32                 :15;
} __cgu_pll0audio_mdiv_bits;

/* PLL0AUDIO NP-divider register */
typedef struct {
  __REG32  PDEC           : 7;
  __REG32                 : 5;
  __REG32  NDEC           :10;
  __REG32                 :10;
} __cgu_pll0audio_np_div_bits;

/* PLL0AUDIO fractional divider register */
typedef struct {
  __REG32  PLLFRACT_CTRL  :22;
  __REG32                 :10;
} __cgu_pll0audio_frac_bits;

/* PLL1_STAT register */
typedef struct {
  __REG32  LOCK           : 1;
  __REG32                 :31;
} __cgu_pll1_stat_bits;

/* PLL1_CTRL register */
typedef struct {
  __REG32  PD             : 1;
  __REG32  BYPASS         : 1;
  __REG32                 : 4;
  __REG32  FBSEL          : 1;
  __REG32  DIRECT         : 1;
  __REG32  PSEL           : 2;
  __REG32                 : 1;
  __REG32  AUTOBLOCK      : 1;
  __REG32  NSEL           : 2;
  __REG32                 : 2;
  __REG32  MSEL           : 8;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_pll1_ctrl_bits;

/* IDIVA control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 : 1;
  __REG32  IDIV           : 2;
  __REG32                 : 7;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_idiva_ctrl_bits;

/* IDIVB/C/D control registers */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 : 1;
  __REG32  IDIV           : 4;
  __REG32                 : 5;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_idivx_ctrl_bits;

/* IDIVE control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 : 1;
  __REG32  IDIV           : 8;
  __REG32                 : 1;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_idive_ctrl_bits;

/* BASE_SAFE_CLK control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 :10;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 5;
  __REG32                 : 3;
} __cgu_base_xxx_clk_bits;


/* CCU1/2 power mode register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 :31;
} __ccu_pm_bits;

/* CCU1 base clock status register */
typedef struct {
  __REG32  BASE_APB3_CLK_IND   : 1;
  __REG32  BASE_APB1_CLK_IND   : 1;
  __REG32  BASE_SPIFI_CLK_IND  : 1;
  __REG32  BASE_M4_CLK_IND     : 1;
  __REG32                      : 2;
  __REG32  BASE_PERIPH_CLK_IND : 1;
  __REG32  BASE_USB0_CLK_IND   : 1;
  __REG32  BASE_USB1_CLK_IND   : 1;
  __REG32  BASE_SPI_CLK_IND    : 1;
  __REG32                      :22;
} __ccu1_base_stat_bits;

/* CCU2 base clock status register */
typedef struct {
  __REG32                     : 1;
  __REG32  BASE_UART3_CLK     : 1;
  __REG32  BASE_UART2_CLK     : 1;
  __REG32  BASE_UART1_CLK     : 1;
  __REG32  BASE_UART0_CLK     : 1;
  __REG32  BASE_SSP1_CLK      : 1;
  __REG32  BASE_SSP0_CLK      : 1;
  __REG32                     :25;
} __ccu2_base_stat_bits;

/* CCU1/2 branch clock configuration register */
/* CCU1/2 branch clock status register */
typedef struct {
  __REG32  RUN                : 1;
  __REG32  AUTO               : 1;
  __REG32  WAKEUP             : 1;
  __REG32                     :29;
} __ccu_clk_cfg_bits;

/* CCU1 CLK_EMCDIV_CFG branch clock configuration register */
typedef struct {
  __REG32  RUN                : 1;
  __REG32  AUTO               : 1;
  __REG32  WAKEUP             : 1;
  __REG32                     : 2;
  __REG32  DIV                : 3;
  __REG32                     :24;
} __ccu1_clk_emcdiv_cfg_bits;


/* Reset control register 0 */

#define RGU_RESET_CTRL0_CORE_RST      (0x00000001UL)
#define RGU_RESET_CTRL0_PERIPH_RST    (0x00000002UL)
#define RGU_RESET_CTRL0_MASTER_RST    (0x00000004UL)
#define RGU_RESET_CTRL0_WWDT_RST      (0x00000010UL)
#define RGU_RESET_CTRL0_CREG_RST      (0x00000020UL)
#define RGU_RESET_CTRL0_BUS_RST       (0x00000100UL)
#define RGU_RESET_CTRL0_SCU_RST       (0x00000200UL)
#define RGU_RESET_CTRL0_PINMUX_RST    (0x00000400UL)
#define RGU_RESET_CTRL0_M4_RST        (0x00002000UL)
#define RGU_RESET_CTRL0_LCD_RST       (0x00010000UL)
#define RGU_RESET_CTRL0_USB0_RST      (0x00020000UL)
#define RGU_RESET_CTRL0_USB1_RST      (0x00040000UL)
#define RGU_RESET_CTRL0_DMA_RST       (0x00080000UL)
#define RGU_RESET_CTRL0_SDIO_RST      (0x00100000UL)
#define RGU_RESET_CTRL0_EMC_RST       (0x00200000UL)
#define RGU_RESET_CTRL0_ETHERNET_RST  (0x00400000UL)
#define RGU_RESET_CTRL0_GPIO_RST      (0x10000000UL)

/* Reset active status register 0 */
typedef struct {
  __REG32  CORE_RST           : 1;
  __REG32  PERIPH_RST         : 1;
  __REG32  MASTER_RST         : 1;
  __REG32                     : 1;
  __REG32  WWDT_RST           : 1;
  __REG32  CREG_RST           : 1;
  __REG32                     : 2;
  __REG32  BUS_RST            : 1;
  __REG32  SCU_RST            : 1;
  __REG32  PINMUX_RST         : 1;
  __REG32                     : 2;
  __REG32  M4_RST             : 1;
  __REG32                     : 2;
  __REG32  LCD_RST            : 1;
  __REG32  USB0_RST           : 1;
  __REG32  USB1_RST           : 1;
  __REG32  DMA_RST            : 1;
  __REG32  SDIO_RST           : 1;
  __REG32  EMC_RST            : 1;
  __REG32  ETHERNET_RST       : 1;
  __REG32                     : 5;
  __REG32  GPIO_RST           : 1;
  __REG32                     : 3;
} __rgu_reset_active_status0_bits;

/* Reset control register 1 */

#define RGU_RESET_CTRL1_TIMER0_RST      (0x00000001UL)
#define RGU_RESET_CTRL1_TIMER1_RST      (0x00000002UL)
#define RGU_RESET_CTRL1_TIMER2_RST      (0x00000004UL)
#define RGU_RESET_CTRL1_TIMER3_RST      (0x00000008UL)
#define RGU_RESET_CTRL1_RITIMER_RST     (0x00000010UL)
#define RGU_RESET_CTRL1_SCT_RST         (0x00000020UL)
#define RGU_RESET_CTRL1_MOTOCONPWM_RST  (0x00000040UL)
#define RGU_RESET_CTRL1_QEI_RST         (0x00000080UL)
#define RGU_RESET_CTRL1_ADC0_RST        (0x00000100UL)
#define RGU_RESET_CTRL1_ADC1_RST        (0x00000200UL)
#define RGU_RESET_CTRL1_DAC_RST         (0x00000400UL)
#define RGU_RESET_CTRL1_UART0_RST       (0x00001000UL)
#define RGU_RESET_CTRL1_UART1_RST       (0x00002000UL)
#define RGU_RESET_CTRL1_UART2_RST       (0x00004000UL)
#define RGU_RESET_CTRL1_UART3_RST       (0x00008000UL)
#define RGU_RESET_CTRL1_I2C0_RST        (0x00010000UL)
#define RGU_RESET_CTRL1_I2C1_RST        (0x00020000UL)
#define RGU_RESET_CTRL1_SSP0_RST        (0x00040000UL)
#define RGU_RESET_CTRL1_SSP1_RST        (0x00080000UL)
#define RGU_RESET_CTRL1_I2S_RST         (0x00100000UL)
#define RGU_RESET_CTRL1_SPIFI_RST       (0x00200000UL)
#define RGU_RESET_CTRL1_CAN1_RST        (0x00400000UL)
#define RGU_RESET_CTRL1_CAN0_RST        (0x00800000UL)
#define RGU_RESET_CTRL1_M0APP_RST       (0x01000000UL)
#define RGU_RESET_CTRL1_SGPIO_RST       (0x02000000UL)
#define RGU_RESET_CTRL1_SPI_RST         (0x04000000UL)

/* Reset active status register 1 */
typedef struct {
  __REG32  TIMER0_RST         : 1;
  __REG32  TIMER1_RST         : 1;
  __REG32  TIMER2_RST         : 1;
  __REG32  TIMER3_RST         : 1;
  __REG32  RITIMER_RST        : 1;
  __REG32  SCT_RST            : 1;
  __REG32  MOTOCONPWM_RST     : 1;
  __REG32  QEI_RST            : 1;
  __REG32  ADC0_RST           : 1;
  __REG32  ADC1_RST           : 1;
  __REG32  DAC_RST            : 1;
  __REG32                     : 1;
  __REG32  UART0_RST          : 1;
  __REG32  UART1_RST          : 1;
  __REG32  UART2_RST          : 1;
  __REG32  UART3_RST          : 1;
  __REG32  I2C0_RST           : 1;
  __REG32  I2C1_RST           : 1;
  __REG32  SSP0_RST           : 1;
  __REG32  SSP1_RST           : 1;
  __REG32  I2S_RST            : 1;
  __REG32  SPIFI_RST          : 1;
  __REG32  CAN1_RST           : 1;
  __REG32  CAN0_RST           : 1;
  __REG32  M0APP_RST          : 1;
  __REG32  SGPIO_RST          : 1;
  __REG32  SPI_RST            : 1;
  __REG32                     : 5;
} __rgu_reset_active_status1_bits;

/* Reset status register 0 */
typedef struct {
  __REG32  CORE_RST           : 2;
  __REG32  PERIPH_RST         : 2;
  __REG32  MASTER_RST         : 2;
  __REG32                     : 2;
  __REG32  WWDT_RST           : 2;
  __REG32  CREG_RST           : 2;
  __REG32                     : 4;
  __REG32  BUS_RST            : 2;
  __REG32  SCU_RST            : 2;
  __REG32                     : 6;
  __REG32  M4_RST             : 2;
  __REG32                     : 4;
} __rgu_reset_status0_bits;

/* Reset status register 1 */
typedef struct {
  __REG32  LCD_RST            : 2;
  __REG32  USB0_RST           : 2;
  __REG32  USB1_RST           : 2;
  __REG32  DMA_RST            : 2;
  __REG32  SDIO_RST           : 2;
  __REG32  EMC_RST            : 2;
  __REG32  ETHERNET_RST       : 2;
  __REG32                     :10;
  __REG32  GPIO_RST           : 2;
  __REG32                     : 6;
} __rgu_reset_status1_bits;

/* Reset status register 2 */
typedef struct {
  __REG32  TIMER0_RST         : 2;
  __REG32  TIMER1_RST         : 2;
  __REG32  TIMER2_RST         : 2;
  __REG32  TIMER3_RST         : 2;
  __REG32  RITIMER_RST        : 2;
  __REG32  SCT_RST            : 2;
  __REG32  MOTOCONPWM_RST     : 2;
  __REG32  QEI_RST            : 2;
  __REG32  ADC0_RST           : 2;
  __REG32  ADC1_RST           : 2;
  __REG32  DAC_RST            : 2;
  __REG32                     : 2;
  __REG32  UART0_RST          : 2;
  __REG32  UART1_RST          : 2;
  __REG32  UART2_RST          : 2;
  __REG32  UART3_RST          : 2;
} __rgu_reset_status2_bits;

/* Reset status register 3 */
typedef struct {
  __REG32  I2C0_RST           : 2;
  __REG32  I2C1_RST           : 2;
  __REG32  SSP0_RST           : 2;
  __REG32  SSP1_RST           : 2;
  __REG32  I2S_RST            : 2;
  __REG32  SPIFI_RST          : 2;
  __REG32  CAN1_RST           : 2;
  __REG32  CAN0_RST           : 2;
  __REG32  M0APP_RST          : 2;
  __REG32  SGPIO_RST          : 2;
  __REG32  SPI_RST            : 2;
  __REG32                     :10;
} __rgu_reset_status3_bits;

/* Reset external status register 0 */
typedef struct {
  __REG32  EXT_RESET          : 1;
  __REG32                     : 3;
  __REG32  BOD_RESET          : 1;
  __REG32  WWDT_RESET         : 1;
  __REG32                     :26;
} __rgu_reset_ext_stat0_bits;

/* Reset external status register 1 */
typedef struct {
  __REG32                     : 1;
  __REG32  CORE_RESET         : 1;
  __REG32                     :30;
} __rgu_reset_ext_stat1_bits;

/* Reset external status register 2 */
typedef struct {
  __REG32                     : 2;
  __REG32  PERIPHERAL_RESET   : 1;
  __REG32                     :29;
} __rgu_reset_ext_stat2_bits;

/* Reset external status register 4 */
typedef struct {
  __REG32                     : 1;
  __REG32  CORE_RESET         : 1;
  __REG32                     :30;
} __rgu_reset_ext_stat4_bits;

/* Reset external status register 5 */
typedef struct {
  __REG32                     : 1;
  __REG32  CORE_RESET         : 1;
  __REG32                     :30;
} __rgu_reset_ext_stat5_bits;

/* Reset external status registers for PERIPHERAL_RESET */
typedef struct {
  __REG32                     : 2;
  __REG32  PERIPHERAL_RESET   : 1;
  __REG32                     :29;
} __rgu_peripheral_reset_bits;

/* Reset external status registers for MASTER_RESET */
typedef struct {
  __REG32                     : 3;
  __REG32  MASTER_RESET       : 1;
  __REG32                     :28;
} __rgu_master_reset_bits;


/* Pin configuration registers for normal drive pins - pins P0_0 to PF_11 and CLK0 to CLK3 */
/* Pin configuration registers for high speed pins - pins P0_0 to PF_11 and CLK0 to CLK3 */
typedef struct {
  __REG32  MODE               : 3;
  __REG32  EPD                : 1;
  __REG32  EPUN               : 1;
  __REG32  EHS                : 1;
  __REG32  EZI                : 1;
  __REG32  ZIF                : 1;
  __REG32                     :24;
} __scu_sfspx_normdrv_hispd_bits;

/* Pin configuration registers for high drive pins - pins P0_0 to PF_11 and CLK0 to CLK3 */
typedef struct {
  __REG32  MODE               : 3;
  __REG32  EPD                : 1;
  __REG32  EPUN               : 1;
  __REG32                     : 1;
  __REG32  EZI                : 1;
  __REG32  ZIF                : 1;
  __REG32  EHD                : 2;
  __REG32                     :22;
} __scu_sfspx_hidrv_bits;

/* Pin configuration register for USB pin DP1/DM1 */
typedef struct {
  __REG32  USB_AIM            : 1;
  __REG32  USB_ESEA           : 1;
  __REG32  USB_EPD            : 1;
  __REG32                     : 1;
  __REG32  USB_EPWR           : 1;
  __REG32  USB_VBUS           : 1;
  __REG32                     :26;
} __scu_sfsusb_bits;

/* Pin configuration register for open-drain I2C-bus pins */
typedef struct {
  __REG32  SCL_EFP            : 1;
  __REG32                     : 1;
  __REG32  SCL_EHD            : 1;
  __REG32  SCL_EZI            : 1;
  __REG32                     : 3;
  __REG32  SCL_ZIF            : 1;
  __REG32  SDA_EFP            : 1;
  __REG32                     : 1;
  __REG32  SDA_EHD            : 1;
  __REG32  SDA_EZI            : 1;
  __REG32                     : 3;
  __REG32  SDA_ZIF            : 1;
  __REG32                     :16;
} __scu_sfsi2c0_bits;

/* ADC0 function select register */
typedef struct {
  __REG32  ADC0_0             : 1;
  __REG32  ADC0_1             : 1;
  __REG32  ADC0_2             : 1;
  __REG32  ADC0_3             : 1;
  __REG32  ADC0_4             : 1;
  __REG32  ADC0_5             : 1;
  __REG32  ADC0_6             : 1;
  __REG32                     :25;
} __scu_enaio0_bits;

/* ADC1 function select register */
typedef struct {
  __REG32  ADC1_0             : 1;
  __REG32  ADC1_1             : 1;
  __REG32  ADC1_2             : 1;
  __REG32  ADC1_3             : 1;
  __REG32  ADC1_4             : 1;
  __REG32  ADC1_5             : 1;
  __REG32  ADC1_6             : 1;
  __REG32  ADC1_7             : 1;
  __REG32                     :24;
} __scu_enaio1_bits;

/* Analog function select register */
typedef struct {
  __REG32  DAC                : 1;
  __REG32                     : 3;
  __REG32  BG                 : 1;
  __REG32                     :27;
} __scu_enaio2_bits;

/* EMC clock delay register */
typedef struct {
  __REG32  CLK_DELAY          :16;
  __REG32                     :16;
} __scu_emcdelayclk_bits;

/* Pin interrupt select register 0 */
typedef struct {
  __REG32  INTPIN0            : 5;
  __REG32  PORTSEL0           : 3;
  __REG32  INTPIN1            : 5;
  __REG32  PORTSEL1           : 3;
  __REG32  INTPIN2            : 5;
  __REG32  PORTSEL2           : 3;
  __REG32  INTPIN3            : 5;
  __REG32  PORTSEL3           : 3;
} __scu_pintsel0_bits;

/* Pin interrupt select register 1 */
typedef struct {
  __REG32  INTPIN4            : 5;
  __REG32  PORTSEL4           : 3;
  __REG32  INTPIN5            : 5;
  __REG32  PORTSEL5           : 3;
  __REG32  INTPIN6            : 5;
  __REG32  PORTSEL6           : 3;
  __REG32  INTPIN7            : 5;
  __REG32  PORTSEL7           : 3;
} __scu_pintsel1_bits;


/* Input multiplexer registers */
typedef struct {
  __REG32  INV                : 1;
  __REG32  EDGE               : 1;
  __REG32  SYNCH              : 1;
  __REG32  PULSE              : 1;
  __REG32  SELECT             : 4;
  __REG32                     :24;
} __gima_in_bits;


/* Pin interrupt mode register */
typedef struct {
  __REG32  PMODE              : 8;
  __REG32                     :24;
} __gpio_isel_bits;

/* Pin interrupt level enable register */
typedef struct {
  __REG32  ENRL               : 8;
  __REG32                     :24;
} __gpio_ienr_bits;

/* Pin interrupt level set register */

#define GPIO_SIENR_SETENRL         (0x000000FFUL)

/* Pin interrupt level clear register */

#define GPIO_CIENR_CENRL           (0x000000FFUL)

/* Pin interrupt active level register */
typedef struct {
  __REG32  ENAF               : 8;
  __REG32                     :24;
} __gpio_ienf_bits;

/* Pin interrupt active level set register */

#define GPIO_SIENF_SETENAF         (0x000000FFUL)

/* Pin interrupt active level clear register */

#define GPIO_CIENF_CENAF           (0x000000FFUL)

/* Pin interrupt rising edge register */
typedef struct {
  __REG32  RDET               : 8;
  __REG32                     :24;
} __gpio_rise_bits;

/* Pin interrupt falling edge register */
typedef struct {
  __REG32  FDET               : 8;
  __REG32                     :24;
} __gpio_fall_bits;

/* Pin interrupt status register */
typedef struct {
  __REG32  PSTAT              : 8;
  __REG32                     :24;
} __gpio_ist_bits;

/* Grouped interrupt control register */
typedef struct {
  __REG32  INT                : 1;
  __REG32  COMB               : 1;
  __REG32  TRIG               : 1;
  __REG32                     :29;
} __gpiogx_ctrl_bits;


/* Pin multiplexer configuration registers */
typedef struct {
  __REG32  P_OUT_CFG          : 4;
  __REG32  P_OE_CFG           : 3;
  __REG32                     :25;
} __sgpio_out_mux_cfgx_bits;

/* SGPIO multiplexer configuration registers */
typedef struct {
  __REG32  EXT_CLK_ENABLE        : 1;
  __REG32  CLK_SOURCE_PIN_MODE   : 2;
  __REG32  CLK_SOURCE_SLICE_MODE : 2;
  __REG32  QUALIFIER_MODE        : 2;
  __REG32  QUALIFIER_PIN_MODE    : 2;
  __REG32  QUALIFIER_SLICE_MODE  : 2;
  __REG32  CONCAT_ENABLE         : 1;
  __REG32  CONCAT_ORDER          : 2;
  __REG32                        :18;
} __sgpio_mux_cfgx_bits;

/* Slice multiplexer configuration registers */
typedef struct {
  __REG32  MATCH_MODE            : 1;
  __REG32  CLK_CAPTURE_MODE      : 1;
  __REG32  CLKGEN_MODE           : 1;
  __REG32  INV_OUT_CLK           : 1;
  __REG32  DATA_CAPTURE_MODE     : 2;
  __REG32  PARALLEL_MODE         : 2;
  __REG32  INV_QUALIFIER         : 1;
  __REG32                        :23;
} __sgpio_slice_mux_cfgx_bits;

/* Reload registers */
typedef struct {
  __REG32  PRESET                :12;
  __REG32                        :20;
} __sgpio_presetx_bits;

/* Down counter registers */
typedef struct {
  __REG32  COUNT                 :12;
  __REG32                        :20;
} __sgpio_countx_bits;

/* Position registers */
typedef struct {
  __REG32  POS                   : 8;
  __REG32  POS_RESET             : 8;
  __REG32                        :16;
} __sgpio_posx_bits;

/* GPIO input status register */
typedef struct {
  __REG32  GPIO_IN0              : 1;
  __REG32  GPIO_IN1              : 1;
  __REG32  GPIO_IN2              : 1;
  __REG32  GPIO_IN3              : 1;
  __REG32  GPIO_IN4              : 1;
  __REG32  GPIO_IN5              : 1;
  __REG32  GPIO_IN6              : 1;
  __REG32  GPIO_IN7              : 1;
  __REG32  GPIO_IN8              : 1;
  __REG32  GPIO_IN9              : 1;
  __REG32  GPIO_IN10             : 1;
  __REG32  GPIO_IN11             : 1;
  __REG32  GPIO_IN12             : 1;
  __REG32  GPIO_IN13             : 1;
  __REG32  GPIO_IN14             : 1;
  __REG32  GPIO_IN15             : 1;
  __REG32                        :16;
} __sgpio_gpio_inreg_bits;

/* GPIO output control register */
typedef struct {
  __REG32  GPIO_OUT0             : 1;
  __REG32  GPIO_OUT1             : 1;
  __REG32  GPIO_OUT2             : 1;
  __REG32  GPIO_OUT3             : 1;
  __REG32  GPIO_OUT4             : 1;
  __REG32  GPIO_OUT5             : 1;
  __REG32  GPIO_OUT6             : 1;
  __REG32  GPIO_OUT7             : 1;
  __REG32  GPIO_OUT8             : 1;
  __REG32  GPIO_OUT9             : 1;
  __REG32  GPIO_OUT10            : 1;
  __REG32  GPIO_OUT11            : 1;
  __REG32  GPIO_OUT12            : 1;
  __REG32  GPIO_OUT13            : 1;
  __REG32  GPIO_OUT14            : 1;
  __REG32  GPIO_OUT15            : 1;
  __REG32                        :16;
} __sgpio_gpio_outreg_bits;

/* GPIO output enable register */
typedef struct {
  __REG32  GPIO_OE0             : 1;
  __REG32  GPIO_OE1             : 1;
  __REG32  GPIO_OE2             : 1;
  __REG32  GPIO_OE3             : 1;
  __REG32  GPIO_OE4             : 1;
  __REG32  GPIO_OE5             : 1;
  __REG32  GPIO_OE6             : 1;
  __REG32  GPIO_OE7             : 1;
  __REG32  GPIO_OE8             : 1;
  __REG32  GPIO_OE9             : 1;
  __REG32  GPIO_OE10            : 1;
  __REG32  GPIO_OE11            : 1;
  __REG32  GPIO_OE12            : 1;
  __REG32  GPIO_OE13            : 1;
  __REG32  GPIO_OE14            : 1;
  __REG32  GPIO_OE15            : 1;
  __REG32                        :16;
} __sgpio_gpio_oenreg_bits;

/* Slice count enable register */
typedef struct {
  __REG32  CTRL_ENABLED0         : 1;
  __REG32  CTRL_ENABLED1         : 1;
  __REG32  CTRL_ENABLED2         : 1;
  __REG32  CTRL_ENABLED3         : 1;
  __REG32  CTRL_ENABLED4         : 1;
  __REG32  CTRL_ENABLED5         : 1;
  __REG32  CTRL_ENABLED6         : 1;
  __REG32  CTRL_ENABLED7         : 1;
  __REG32  CTRL_ENABLED8         : 1;
  __REG32  CTRL_ENABLED9         : 1;
  __REG32  CTRL_ENABLED10        : 1;
  __REG32  CTRL_ENABLED11        : 1;
  __REG32  CTRL_ENABLED12        : 1;
  __REG32  CTRL_ENABLED13        : 1;
  __REG32  CTRL_ENABLED14        : 1;
  __REG32  CTRL_ENABLED15        : 1;
  __REG32                        :16;
} __sgpio_ctrl_enabled_bits;

/* Slice count disable register */
typedef struct {
  __REG32  CTRL_DISABLED0        : 1;
  __REG32  CTRL_DISABLED1        : 1;
  __REG32  CTRL_DISABLED2        : 1;
  __REG32  CTRL_DISABLED3        : 1;
  __REG32  CTRL_DISABLED4        : 1;
  __REG32  CTRL_DISABLED5        : 1;
  __REG32  CTRL_DISABLED6        : 1;
  __REG32  CTRL_DISABLED7        : 1;
  __REG32  CTRL_DISABLED8        : 1;
  __REG32  CTRL_DISABLED9        : 1;
  __REG32  CTRL_DISABLED10       : 1;
  __REG32  CTRL_DISABLED11       : 1;
  __REG32  CTRL_DISABLED12       : 1;
  __REG32  CTRL_DISABLED13       : 1;
  __REG32  CTRL_DISABLED14       : 1;
  __REG32  CTRL_DISABLED15       : 1;
  __REG32                        :16;
} __sgpio_ctrl_disabled_bits;

/* Shift clock interrupt clear mask register */

#define SGPIO_CLR_EN_0_CLR_SCI    (0x0000FFFFUL)
#define SGPIO_CLR_EN_0_CLR_SCI0   (0x00000001UL)
#define SGPIO_CLR_EN_0_CLR_SCI1   (0x00000002UL)
#define SGPIO_CLR_EN_0_CLR_SCI2   (0x00000004UL)
#define SGPIO_CLR_EN_0_CLR_SCI3   (0x00000008UL)
#define SGPIO_CLR_EN_0_CLR_SCI4   (0x00000010UL)
#define SGPIO_CLR_EN_0_CLR_SCI5   (0x00000020UL)
#define SGPIO_CLR_EN_0_CLR_SCI6   (0x00000040UL)
#define SGPIO_CLR_EN_0_CLR_SCI7   (0x00000080UL)
#define SGPIO_CLR_EN_0_CLR_SCI8   (0x00000100UL)
#define SGPIO_CLR_EN_0_CLR_SCI9   (0x00000200UL)
#define SGPIO_CLR_EN_0_CLR_SCI10  (0x00000400UL)
#define SGPIO_CLR_EN_0_CLR_SCI11  (0x00000800UL)
#define SGPIO_CLR_EN_0_CLR_SCI12  (0x00001000UL)
#define SGPIO_CLR_EN_0_CLR_SCI13  (0x00002000UL)
#define SGPIO_CLR_EN_0_CLR_SCI14  (0x00004000UL)
#define SGPIO_CLR_EN_0_CLR_SCI15  (0x00008000UL)

/* Shift clock interrupt set mask register */
#define SGPIO_SET_EN_0_SET_SCI    (0x0000FFFFUL)
#define SGPIO_SET_EN_0_SET_SCI0   (0x00000001UL)
#define SGPIO_SET_EN_0_SET_SCI1   (0x00000002UL)
#define SGPIO_SET_EN_0_SET_SCI2   (0x00000004UL)
#define SGPIO_SET_EN_0_SET_SCI3   (0x00000008UL)
#define SGPIO_SET_EN_0_SET_SCI4   (0x00000010UL)
#define SGPIO_SET_EN_0_SET_SCI5   (0x00000020UL)
#define SGPIO_SET_EN_0_SET_SCI6   (0x00000040UL)
#define SGPIO_SET_EN_0_SET_SCI7   (0x00000080UL)
#define SGPIO_SET_EN_0_SET_SCI8   (0x00000100UL)
#define SGPIO_SET_EN_0_SET_SCI9   (0x00000200UL)
#define SGPIO_SET_EN_0_SET_SCI10  (0x00000400UL)
#define SGPIO_SET_EN_0_SET_SCI11  (0x00000800UL)
#define SGPIO_SET_EN_0_SET_SCI12  (0x00001000UL)
#define SGPIO_SET_EN_0_SET_SCI13  (0x00002000UL)
#define SGPIO_SET_EN_0_SET_SCI14  (0x00004000UL)
#define SGPIO_SET_EN_0_SET_SCI15  (0x00008000UL)

/* Shift clock interrupt enable register */
typedef struct {
  __REG32  ENABLE_SCI0           : 1;
  __REG32  ENABLE_SCI1           : 1;
  __REG32  ENABLE_SCI2           : 1;
  __REG32  ENABLE_SCI3           : 1;
  __REG32  ENABLE_SCI4           : 1;
  __REG32  ENABLE_SCI5           : 1;
  __REG32  ENABLE_SCI6           : 1;
  __REG32  ENABLE_SCI7           : 1;
  __REG32  ENABLE_SCI8           : 1;
  __REG32  ENABLE_SCI9           : 1;
  __REG32  ENABLE_SCI10          : 1;
  __REG32  ENABLE_SCI11          : 1;
  __REG32  ENABLE_SCI12          : 1;
  __REG32  ENABLE_SCI13          : 1;
  __REG32  ENABLE_SCI14          : 1;
  __REG32  ENABLE_SCI15          : 1;
  __REG32                        :16;
} __sgpio_enable_0_bits;

/* Shift clock interrupt status register */
typedef struct {
  __REG32  STATUS_SCI0           : 1;
  __REG32  STATUS_SCI1           : 1;
  __REG32  STATUS_SCI2           : 1;
  __REG32  STATUS_SCI3           : 1;
  __REG32  STATUS_SCI4           : 1;
  __REG32  STATUS_SCI5           : 1;
  __REG32  STATUS_SCI6           : 1;
  __REG32  STATUS_SCI7           : 1;
  __REG32  STATUS_SCI8           : 1;
  __REG32  STATUS_SCI9           : 1;
  __REG32  STATUS_SCI10          : 1;
  __REG32  STATUS_SCI11          : 1;
  __REG32  STATUS_SCI12          : 1;
  __REG32  STATUS_SCI13          : 1;
  __REG32  STATUS_SCI14          : 1;
  __REG32  STATUS_SCI15          : 1;
  __REG32                        :16;
} __sgpio_status_0_bits;

/* Shift clock interrupt clear status register */

#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI    (0x0000FFFFUL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI0   (0x00000001UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI1   (0x00000002UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI2   (0x00000004UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI3   (0x00000008UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI4   (0x00000010UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI5   (0x00000020UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI6   (0x00000040UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI7   (0x00000080UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI8   (0x00000100UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI9   (0x00000200UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI10  (0x00000400UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI11  (0x00000800UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI12  (0x00001000UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI13  (0x00002000UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI14  (0x00004000UL)
#define SGPIO_CTR_STATUS_0_CTR_STATUS_SCI15  (0x00008000UL)

/* Shift clock interrupt set status register */

#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI    (0x0000FFFFUL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI0   (0x00000001UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI1   (0x00000002UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI2   (0x00000004UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI3   (0x00000008UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI4   (0x00000010UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI5   (0x00000020UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI6   (0x00000040UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI7   (0x00000080UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI8   (0x00000100UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI9   (0x00000200UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI10  (0x00000400UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI11  (0x00000800UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI12  (0x00001000UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI13  (0x00002000UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI14  (0x00004000UL)
#define SGPIO_SET_STATUS_0_CTR_STATUS_SCI15  (0x00008000UL)

/* Capture clock interrupt clear mask register */

#define SGPIO_CLR_EN_1_CLR_CCI    (0x0000FFFFUL)
#define SGPIO_CLR_EN_1_CLR_CCI0   (0x00000001UL)
#define SGPIO_CLR_EN_1_CLR_CCI1   (0x00000002UL)
#define SGPIO_CLR_EN_1_CLR_CCI2   (0x00000004UL)
#define SGPIO_CLR_EN_1_CLR_CCI3   (0x00000008UL)
#define SGPIO_CLR_EN_1_CLR_CCI4   (0x00000010UL)
#define SGPIO_CLR_EN_1_CLR_CCI5   (0x00000020UL)
#define SGPIO_CLR_EN_1_CLR_CCI6   (0x00000040UL)
#define SGPIO_CLR_EN_1_CLR_CCI7   (0x00000080UL)
#define SGPIO_CLR_EN_1_CLR_CCI8   (0x00000100UL)
#define SGPIO_CLR_EN_1_CLR_CCI9   (0x00000200UL)
#define SGPIO_CLR_EN_1_CLR_CCI10  (0x00000400UL)
#define SGPIO_CLR_EN_1_CLR_CCI11  (0x00000800UL)
#define SGPIO_CLR_EN_1_CLR_CCI12  (0x00001000UL)
#define SGPIO_CLR_EN_1_CLR_CCI13  (0x00002000UL)
#define SGPIO_CLR_EN_1_CLR_CCI14  (0x00004000UL)
#define SGPIO_CLR_EN_1_CLR_CCI15  (0x00008000UL)

/* Capture clock interrupt set mask register */

#define SGPIO_SET_EN_1_SET_CCI    (0x0000FFFFUL)
#define SGPIO_SET_EN_1_SET_CCI0   (0x00000001UL)
#define SGPIO_SET_EN_1_SET_CCI1   (0x00000002UL)
#define SGPIO_SET_EN_1_SET_CCI2   (0x00000004UL)
#define SGPIO_SET_EN_1_SET_CCI3   (0x00000008UL)
#define SGPIO_SET_EN_1_SET_CCI4   (0x00000010UL)
#define SGPIO_SET_EN_1_SET_CCI5   (0x00000020UL)
#define SGPIO_SET_EN_1_SET_CCI6   (0x00000040UL)
#define SGPIO_SET_EN_1_SET_CCI7   (0x00000080UL)
#define SGPIO_SET_EN_1_SET_CCI8   (0x00000100UL)
#define SGPIO_SET_EN_1_SET_CCI9   (0x00000200UL)
#define SGPIO_SET_EN_1_SET_CCI10  (0x00000400UL)
#define SGPIO_SET_EN_1_SET_CCI11  (0x00000800UL)
#define SGPIO_SET_EN_1_SET_CCI12  (0x00001000UL)
#define SGPIO_SET_EN_1_SET_CCI13  (0x00002000UL)
#define SGPIO_SET_EN_1_SET_CCI14  (0x00004000UL)
#define SGPIO_SET_EN_1_SET_CCI15  (0x00008000UL)

/* Capture clock interrupt enable register */
typedef struct {
  __REG32  ENABLE_CCI0           : 1;
  __REG32  ENABLE_CCI1           : 1;
  __REG32  ENABLE_CCI2           : 1;
  __REG32  ENABLE_CCI3           : 1;
  __REG32  ENABLE_CCI4           : 1;
  __REG32  ENABLE_CCI5           : 1;
  __REG32  ENABLE_CCI6           : 1;
  __REG32  ENABLE_CCI7           : 1;
  __REG32  ENABLE_CCI8           : 1;
  __REG32  ENABLE_CCI9           : 1;
  __REG32  ENABLE_CCI10          : 1;
  __REG32  ENABLE_CCI11          : 1;
  __REG32  ENABLE_CCI12          : 1;
  __REG32  ENABLE_CCI13          : 1;
  __REG32  ENABLE_CCI14          : 1;
  __REG32  ENABLE_CCI15          : 1;
  __REG32                        :16;
} __sgpio_enable_1_bits;

/* Capture clock interrupt status register */
typedef struct {
  __REG32  STATUS_CCI0           : 1;
  __REG32  STATUS_CCI1           : 1;
  __REG32  STATUS_CCI2           : 1;
  __REG32  STATUS_CCI3           : 1;
  __REG32  STATUS_CCI4           : 1;
  __REG32  STATUS_CCI5           : 1;
  __REG32  STATUS_CCI6           : 1;
  __REG32  STATUS_CCI7           : 1;
  __REG32  STATUS_CCI8           : 1;
  __REG32  STATUS_CCI9           : 1;
  __REG32  STATUS_CCI10          : 1;
  __REG32  STATUS_CCI11          : 1;
  __REG32  STATUS_CCI12          : 1;
  __REG32  STATUS_CCI13          : 1;
  __REG32  STATUS_CCI14          : 1;
  __REG32  STATUS_CCI15          : 1;
  __REG32                        :16;
} __sgpio_status_1_bits;

/* Capture clock interrupt clear status register */

#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI    (0x0000FFFFUL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI0   (0x00000001UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI1   (0x00000002UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI2   (0x00000004UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI3   (0x00000008UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI4   (0x00000010UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI5   (0x00000020UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI6   (0x00000040UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI7   (0x00000080UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI8   (0x00000100UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI9   (0x00000200UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI10  (0x00000400UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI11  (0x00000800UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI12  (0x00001000UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI13  (0x00002000UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI14  (0x00004000UL)
#define SGPIO_CTR_STATUS_1_CTR_STATUS_CCI15  (0x00008000UL)

/* Capture clock interrupt set status register */

#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI    (0x0000FFFFUL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI0   (0x00000001UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI1   (0x00000002UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI2   (0x00000004UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI3   (0x00000008UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI4   (0x00000010UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI5   (0x00000020UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI6   (0x00000040UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI7   (0x00000080UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI8   (0x00000100UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI9   (0x00000200UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI10  (0x00000400UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI11  (0x00000800UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI12  (0x00001000UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI13  (0x00002000UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI14  (0x00004000UL)
#define SGPIO_SET_STATUS_1_CTR_STATUS_CCI15  (0x00008000UL)

/* Pattern match interrupt clear mask register */

#define SGPIO_CLR_EN_2_CLR_EN2_PMI    (0x0000FFFFUL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI0   (0x00000001UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI1   (0x00000002UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI2   (0x00000004UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI3   (0x00000008UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI4   (0x00000010UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI5   (0x00000020UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI6   (0x00000040UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI7   (0x00000080UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI8   (0x00000100UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI9   (0x00000200UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI10  (0x00000400UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI11  (0x00000800UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI12  (0x00001000UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI13  (0x00002000UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI14  (0x00004000UL)
#define SGPIO_CLR_EN_2_CLR_EN2_PMI15  (0x00008000UL)

/* Pattern match interrupt set mask register */

#define SGPIO_SET_EN_2_SET_EN_PMI    (0x0000FFFFUL)
#define SGPIO_SET_EN_2_SET_EN_PMI0   (0x00000001UL)
#define SGPIO_SET_EN_2_SET_EN_PMI1   (0x00000002UL)
#define SGPIO_SET_EN_2_SET_EN_PMI2   (0x00000004UL)
#define SGPIO_SET_EN_2_SET_EN_PMI3   (0x00000008UL)
#define SGPIO_SET_EN_2_SET_EN_PMI4   (0x00000010UL)
#define SGPIO_SET_EN_2_SET_EN_PMI5   (0x00000020UL)
#define SGPIO_SET_EN_2_SET_EN_PMI6   (0x00000040UL)
#define SGPIO_SET_EN_2_SET_EN_PMI7   (0x00000080UL)
#define SGPIO_SET_EN_2_SET_EN_PMI8   (0x00000100UL)
#define SGPIO_SET_EN_2_SET_EN_PMI9   (0x00000200UL)
#define SGPIO_SET_EN_2_SET_EN_PMI10  (0x00000400UL)
#define SGPIO_SET_EN_2_SET_EN_PMI11  (0x00000800UL)
#define SGPIO_SET_EN_2_SET_EN_PMI12  (0x00001000UL)
#define SGPIO_SET_EN_2_SET_EN_PMI13  (0x00002000UL)
#define SGPIO_SET_EN_2_SET_EN_PMI14  (0x00004000UL)
#define SGPIO_SET_EN_2_SET_EN_PMI15  (0x00008000UL)

/* Pattern match interrupt enable register */
typedef struct {
  __REG32  ENABLE_PMI0          : 1;
  __REG32  ENABLE_PMI1          : 1;
  __REG32  ENABLE_PMI2          : 1;
  __REG32  ENABLE_PMI3          : 1;
  __REG32  ENABLE_PMI4          : 1;
  __REG32  ENABLE_PMI5          : 1;
  __REG32  ENABLE_PMI6          : 1;
  __REG32  ENABLE_PMI7          : 1;
  __REG32  ENABLE_PMI8          : 1;
  __REG32  ENABLE_PMI9          : 1;
  __REG32  ENABLE_PMI10         : 1;
  __REG32  ENABLE_PMI11         : 1;
  __REG32  ENABLE_PMI12         : 1;
  __REG32  ENABLE_PMI13         : 1;
  __REG32  ENABLE_PMI14         : 1;
  __REG32  ENABLE_PMI15         : 1;
  __REG32                       :16;
} __sgpio_enable_2_bits;

/* Pattern match interrupt status register */
typedef struct {
  __REG32  STATUS_PMI0          : 1;
  __REG32  STATUS_PMI1          : 1;
  __REG32  STATUS_PMI2          : 1;
  __REG32  STATUS_PMI3          : 1;
  __REG32  STATUS_PMI4          : 1;
  __REG32  STATUS_PMI5          : 1;
  __REG32  STATUS_PMI6          : 1;
  __REG32  STATUS_PMI7          : 1;
  __REG32  STATUS_PMI8          : 1;
  __REG32  STATUS_PMI9          : 1;
  __REG32  STATUS_PMI10         : 1;
  __REG32  STATUS_PMI11         : 1;
  __REG32  STATUS_PMI12         : 1;
  __REG32  STATUS_PMI13         : 1;
  __REG32  STATUS_PMI14         : 1;
  __REG32  STATUS_PMI15         : 1;
  __REG32                       :16;
} __sgpio_status_2_bits;

/* Pattern match interrupt clear status register */

#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI    (0x0000FFFFUL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI0   (0x00000001UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI1   (0x00000002UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI2   (0x00000004UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI3   (0x00000008UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI4   (0x00000010UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI5   (0x00000020UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI6   (0x00000040UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI7   (0x00000080UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI8   (0x00000100UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI9   (0x00000200UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI10  (0x00000400UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI11  (0x00000800UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI12  (0x00001000UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI13  (0x00002000UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI14  (0x00004000UL)
#define SGPIO_CTR_STATUS_2_CTR_STATUS_PMI15  (0x00008000UL)

/* Pattern match interrupt set status register */

#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI    (0x0000FFFFUL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI0   (0x00000001UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI1   (0x00000002UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI2   (0x00000004UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI3   (0x00000008UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI4   (0x00000010UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI5   (0x00000020UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI6   (0x00000040UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI7   (0x00000080UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI8   (0x00000100UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI9   (0x00000200UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI10  (0x00000400UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI11  (0x00000800UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI12  (0x00001000UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI13  (0x00002000UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI14  (0x00004000UL)
#define SGPIO_SET_STATUS_2_CTR_STATUS_PMI15  (0x00008000UL)

/* Input interrupt clear mask register */

#define SGPIO_CLR_EN_3_CLR_EN_INPI    (0x0000FFFFUL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI0   (0x00000001UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI1   (0x00000002UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI2   (0x00000004UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI3   (0x00000008UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI4   (0x00000010UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI5   (0x00000020UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI6   (0x00000040UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI7   (0x00000080UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI8   (0x00000100UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI9   (0x00000200UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI10  (0x00000400UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI11  (0x00000800UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI12  (0x00001000UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI13  (0x00002000UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI14  (0x00004000UL)
#define SGPIO_CLR_EN_3_CLR_EN_INPI15  (0x00008000UL)

/* Input interrupt set mask register */

#define SGPIO_SET_EN_3_SET_EN_INPI    (0x0000FFFFUL)
#define SGPIO_SET_EN_3_SET_EN_INPI0   (0x00000001UL)
#define SGPIO_SET_EN_3_SET_EN_INPI1   (0x00000002UL)
#define SGPIO_SET_EN_3_SET_EN_INPI2   (0x00000004UL)
#define SGPIO_SET_EN_3_SET_EN_INPI3   (0x00000008UL)
#define SGPIO_SET_EN_3_SET_EN_INPI4   (0x00000010UL)
#define SGPIO_SET_EN_3_SET_EN_INPI5   (0x00000020UL)
#define SGPIO_SET_EN_3_SET_EN_INPI6   (0x00000040UL)
#define SGPIO_SET_EN_3_SET_EN_INPI7   (0x00000080UL)
#define SGPIO_SET_EN_3_SET_EN_INPI8   (0x00000100UL)
#define SGPIO_SET_EN_3_SET_EN_INPI9   (0x00000200UL)
#define SGPIO_SET_EN_3_SET_EN_INPI10  (0x00000400UL)
#define SGPIO_SET_EN_3_SET_EN_INPI11  (0x00000800UL)
#define SGPIO_SET_EN_3_SET_EN_INPI12  (0x00001000UL)
#define SGPIO_SET_EN_3_SET_EN_INPI13  (0x00002000UL)
#define SGPIO_SET_EN_3_SET_EN_INPI14  (0x00004000UL)
#define SGPIO_SET_EN_3_SET_EN_INPI15  (0x00008000UL)

/* Input interrupt enable register */
typedef struct {
  __REG32  ENABLE3_INPI0          : 1;
  __REG32  ENABLE3_INPI1          : 1;
  __REG32  ENABLE3_INPI2          : 1;
  __REG32  ENABLE3_INPI3          : 1;
  __REG32  ENABLE3_INPI4          : 1;
  __REG32  ENABLE3_INPI5          : 1;
  __REG32  ENABLE3_INPI6          : 1;
  __REG32  ENABLE3_INPI7          : 1;
  __REG32  ENABLE3_INPI8          : 1;
  __REG32  ENABLE3_INPI9          : 1;
  __REG32  ENABLE3_INPI10         : 1;
  __REG32  ENABLE3_INPI11         : 1;
  __REG32  ENABLE3_INPI12         : 1;
  __REG32  ENABLE3_INPI13         : 1;
  __REG32  ENABLE3_INPI14         : 1;
  __REG32  ENABLE3_INPI15         : 1;
  __REG32                         :16;
} __sgpio_enable_3_bits;

/* Input interrupt status register */
typedef struct {
  __REG32  STATUS_INPI0          : 1;
  __REG32  STATUS_INPI1          : 1;
  __REG32  STATUS_INPI2          : 1;
  __REG32  STATUS_INPI3          : 1;
  __REG32  STATUS_INPI4          : 1;
  __REG32  STATUS_INPI5          : 1;
  __REG32  STATUS_INPI6          : 1;
  __REG32  STATUS_INPI7          : 1;
  __REG32  STATUS_INPI8          : 1;
  __REG32  STATUS_INPI9          : 1;
  __REG32  STATUS_INPI10         : 1;
  __REG32  STATUS_INPI11         : 1;
  __REG32  STATUS_INPI12         : 1;
  __REG32  STATUS_INPI13         : 1;
  __REG32  STATUS_INPI14         : 1;
  __REG32  STATUS_INPI15         : 1;
  __REG32                        :16;
} __sgpio_status_3_bits;

/* Input interrupt clear status register */

#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI    (0x0000FFFFUL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI0   (0x00000001UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI1   (0x00000002UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI2   (0x00000004UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI3   (0x00000008UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI4   (0x00000010UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI5   (0x00000020UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI6   (0x00000040UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI7   (0x00000080UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI8   (0x00000100UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI9   (0x00000200UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI10  (0x00000400UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI11  (0x00000800UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI12  (0x00001000UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI13  (0x00002000UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI14  (0x00004000UL)
#define SGPIO_CTR_STATUS_3_CTR_STATUS_INPI15  (0x00008000UL)

/* Input interrupt set status register */

#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI    (0x0000FFFFUL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI0   (0x00000001UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI1   (0x00000002UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI2   (0x00000004UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI3   (0x00000008UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI4   (0x00000010UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI5   (0x00000020UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI6   (0x00000040UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI7   (0x00000080UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI8   (0x00000100UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI9   (0x00000200UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI10  (0x00000400UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI11  (0x00000800UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI12  (0x00001000UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI13  (0x00002000UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI14  (0x00004000UL)
#define SGPIO_SET_STATUS_3_CTR_STATUS_INPI15  (0x00008000UL)

/* DMA Interrupt Status Register */
typedef struct{
__REG32 INTSTAT0  : 1;
__REG32 INTSTAT1  : 1;
__REG32 INTSTAT2  : 1;
__REG32 INTSTAT3  : 1;
__REG32 INTSTAT4  : 1;
__REG32 INTSTAT5  : 1;
__REG32 INTSTAT6  : 1;
__REG32 INTSTAT7  : 1;
__REG32             :24;
} __dmacintstatus_bits;

/* DMA Interrupt Terminal Count Request Status Register */
typedef struct{
__REG32 INTTCSTAT0  : 1;
__REG32 INTTCSTAT1  : 1;
__REG32 INTTCSTAT2  : 1;
__REG32 INTTCSTAT3  : 1;
__REG32 INTTCSTAT4  : 1;
__REG32 INTTCSTAT5  : 1;
__REG32 INTTCSTAT6  : 1;
__REG32 INTTCSTAT7  : 1;
__REG32               :24;
} __dmacinttcstatus_bits;

/* DMA Interrupt Terminal Count Request Clear Register */

#define GPDMA_INTTCCLEAR_INTTCCLEAR       (0x000000FFUL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR0      (0x00000001UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR1      (0x00000002UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR2      (0x00000004UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR3      (0x00000008UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR4      (0x00000010UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR5      (0x00000020UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR6      (0x00000040UL)
#define GPDMA_INTTCCLEAR_INTTCCLEAR7      (0x00000080UL)

/* DMA Interrupt Error Status Register */
typedef struct{
__REG32 INTERRSTAT0 : 1;
__REG32 INTERRSTAT1 : 1;
__REG32 INTERRSTAT2 : 1;
__REG32 INTERRSTAT3 : 1;
__REG32 INTERRSTAT4 : 1;
__REG32 INTERRSTAT5 : 1;
__REG32 INTERRSTAT6 : 1;
__REG32 INTERRSTAT7 : 1;
__REG32             :24;
} __dmacinterrstat_bits;

/* DMA Interrupt Error Clear Register */

#define GPDMA_INTERRCLR_INTERRCLR       (0x000000FFUL)
#define GPDMA_INTERRCLR_INTERRCLR0      (0x00000001UL)
#define GPDMA_INTERRCLR_INTERRCLR1      (0x00000002UL)
#define GPDMA_INTERRCLR_INTERRCLR2      (0x00000004UL)
#define GPDMA_INTERRCLR_INTERRCLR3      (0x00000008UL)
#define GPDMA_INTERRCLR_INTERRCLR4      (0x00000010UL)
#define GPDMA_INTERRCLR_INTERRCLR5      (0x00000020UL)
#define GPDMA_INTERRCLR_INTERRCLR6      (0x00000040UL)
#define GPDMA_INTERRCLR_INTERRCLR7      (0x00000080UL)

/* DMA Raw Interrupt Terminal Count Status Register */
typedef struct{
__REG32 RAWINTTCSTAT0 : 1;
__REG32 RAWINTTCSTAT1 : 1;
__REG32 RAWINTTCSTAT2 : 1;
__REG32 RAWINTTCSTAT3 : 1;
__REG32 RAWINTTCSTAT4 : 1;
__REG32 RAWINTTCSTAT5 : 1;
__REG32 RAWINTTCSTAT6 : 1;
__REG32 RAWINTTCSTAT7 : 1;
__REG32                 :24;
} __dmacrawinttcstatus_bits;

/* DMA Raw Error Interrupt Status Register */
typedef struct{
__REG32 RAWINTERRSTAT0  : 1;
__REG32 RAWINTERRSTAT1  : 1;
__REG32 RAWINTERRSTAT2  : 1;
__REG32 RAWINTERRSTAT3  : 1;
__REG32 RAWINTERRSTAT4  : 1;
__REG32 RAWINTERRSTAT5  : 1;
__REG32 RAWINTERRSTAT6  : 1;
__REG32 RAWINTERRSTAT7  : 1;
__REG32                 :24;
} __dmacrawinterrorstatus_bits;

/* DMA Enabled Channel Register */
typedef struct{
__REG32 ENABLEDCHANNELS0  : 1;
__REG32 ENABLEDCHANNELS1  : 1;
__REG32 ENABLEDCHANNELS2  : 1;
__REG32 ENABLEDCHANNELS3  : 1;
__REG32 ENABLEDCHANNELS4  : 1;
__REG32 ENABLEDCHANNELS5  : 1;
__REG32 ENABLEDCHANNELS6  : 1;
__REG32 ENABLEDCHANNELS7  : 1;
__REG32                   :24;
} __dmacenbldchns_bits;

/* DMA Software Burst Request Register */
typedef struct{
__REG32 SOFTBREQ0         : 1;
__REG32 SOFTBREQ1         : 1;
__REG32 SOFTBREQ2         : 1;
__REG32 SOFTBREQ3         : 1;
__REG32 SOFTBREQ4         : 1;
__REG32 SOFTBREQ5         : 1;
__REG32 SOFTBREQ6         : 1;
__REG32 SOFTBREQ7         : 1;
__REG32 SOFTBREQ8         : 1;
__REG32 SOFTBREQ9         : 1;
__REG32 SOFTBREQ10        : 1;
__REG32 SOFTBREQ11        : 1;
__REG32 SOFTBREQ12        : 1;
__REG32 SOFTBREQ13        : 1;
__REG32 SOFTBREQ14        : 1;
__REG32 SOFTBREQ15        : 1;
__REG32                   :16;
} __dmacsoftbreq_bits;

/* DMA Software Single Request Register */
typedef struct{
__REG32 SOFTSREQ0         : 1;
__REG32 SOFTSREQ1         : 1;
__REG32 SOFTSREQ2         : 1;
__REG32 SOFTSREQ3         : 1;
__REG32 SOFTSREQ4         : 1;
__REG32 SOFTSREQ5         : 1;
__REG32 SOFTSREQ6         : 1;
__REG32 SOFTSREQ7         : 1;
__REG32 SOFTSREQ8         : 1;
__REG32 SOFTSREQ9         : 1;
__REG32 SOFTSREQ10        : 1;
__REG32 SOFTSREQ11        : 1;
__REG32 SOFTSREQ12        : 1;
__REG32 SOFTSREQ13        : 1;
__REG32 SOFTSREQ14        : 1;
__REG32 SOFTSREQ15        : 1;
__REG32                   :16;
} __dmacsoftsreq_bits;

/* DMA Software Last Burst Request Register */
typedef struct{
__REG32 SOFTLBREQ0         : 1;
__REG32 SOFTLBREQ1         : 1;
__REG32 SOFTLBREQ2         : 1;
__REG32 SOFTLBREQ3         : 1;
__REG32 SOFTLBREQ4         : 1;
__REG32 SOFTLBREQ5         : 1;
__REG32 SOFTLBREQ6         : 1;
__REG32 SOFTLBREQ7         : 1;
__REG32 SOFTLBREQ8         : 1;
__REG32 SOFTLBREQ9         : 1;
__REG32 SOFTLBREQ10        : 1;
__REG32 SOFTLBREQ11        : 1;
__REG32 SOFTLBREQ12        : 1;
__REG32 SOFTLBREQ13        : 1;
__REG32 SOFTLBREQ14        : 1;
__REG32 SOFTLBREQ15        : 1;
__REG32                    :16;
} __dmacsoftlbreq_bits;

/* DMA Software Last Single Request Register */
typedef struct{
__REG32 SOFTLSREQ0          : 1;
__REG32 SOFTLSREQ1          : 1;
__REG32 SOFTLSREQ2          : 1;
__REG32 SOFTLSREQ3          : 1;
__REG32 SOFTLSREQ4          : 1;
__REG32 SOFTLSREQ5          : 1;
__REG32 SOFTLSREQ6          : 1;
__REG32 SOFTLSREQ7          : 1;
__REG32 SOFTLSREQ8          : 1;
__REG32 SOFTLSREQ9          : 1;
__REG32 SOFTLSREQ10         : 1;
__REG32 SOFTLSREQ11         : 1;
__REG32 SOFTLSREQ12         : 1;
__REG32 SOFTLSREQ13         : 1;
__REG32 SOFTLSREQ14         : 1;
__REG32 SOFTLSREQ15         : 1;
__REG32                     :16;
} __dmacsoftlsreq_bits;

/* DMA Synchronization Register */
typedef struct{
__REG32 DMACSYNC0   : 1;
__REG32 DMACSYNC1   : 1;
__REG32 DMACSYNC2   : 1;
__REG32 DMACSYNC3   : 1;
__REG32 DMACSYNC4   : 1;
__REG32 DMACSYNC5   : 1;
__REG32 DMACSYNC6   : 1;
__REG32 DMACSYNC7   : 1;
__REG32 DMACSYNC8   : 1;
__REG32 DMACSYNC9   : 1;
__REG32 DMACSYNC10  : 1;
__REG32 DMACSYNC11  : 1;
__REG32 DMACSYNC12  : 1;
__REG32 DMACSYNC13  : 1;
__REG32 DMACSYNC14  : 1;
__REG32 DMACSYNC15  : 1;
__REG32             :16;
} __dmacsync_bits;

/* DMA Configuration Register */
typedef struct{
__REG32 E   : 1;
__REG32 M0  : 1;
__REG32 M1  : 1;
__REG32     :29;
} __dmacconfig_bits;

/* DMA Channel Linked List Item registers */
typedef struct{
__REG32 LM  : 1;
__REG32     : 1;
__REG32 LLI :30;
} __dma_lli_bits;

/* DMA Channel Control Registers */
typedef struct{
__REG32 TRANSFERSIZE  :12;
__REG32 SBSIZE        : 3;
__REG32 DBSIZE        : 3;
__REG32 SWIDTH        : 3;
__REG32 DWIDTH        : 3;
__REG32 S             : 1;
__REG32 D             : 1;
__REG32 SI            : 1;
__REG32 DI            : 1;
__REG32 PROT1         : 1;
__REG32 PROT2         : 1;
__REG32 PROT3         : 1;
__REG32 I             : 1;
} __dma_ctrl_bits;

/* DMA Channel Configuration Registers */
typedef struct{
__REG32 E               : 1;
__REG32 SRCPERIPHERAL   : 5;
__REG32 DESTPERIPHERAL  : 5;
__REG32 FLOWCNTRL       : 3;
__REG32 IE              : 1;
__REG32 ITC             : 1;
__REG32 L               : 1;
__REG32 A               : 1;
__REG32 H               : 1;
__REG32                 :13;
} __dma_cfg_bits;


/* SDIO Control register */
typedef struct{
__REG32 CONTROLLER_RESET              : 1;
__REG32 FIFO_RESET                    : 1;
__REG32 DMA_RESET                     : 1;
__REG32                               : 1;
__REG32 INT_ENABLE                    : 1;
__REG32 DMA_ENABLE                    : 1;
__REG32 READ_WAIT                     : 1;
__REG32 SEND_IRQ_RESPONSE             : 1;
__REG32 ABORT_READ_DATA               : 1;
__REG32 SEND_CCSD                     : 1;
__REG32 SEND_AUTO_STOP_CCSD           : 1;
__REG32 CEATA_DEVICE_INTERRUPT_STATUS : 1;
__REG32                               : 4;
__REG32 CARD_VOLTAGE_A                : 4;
__REG32 CARD_VOLTAGE_B                : 4;
__REG32 ENABLE_OD_PULLUP              : 1;
__REG32 USE_INTERNAL_DMAC             : 1;
__REG32                               : 6;
} __sdio_ctrl_bits;

/* SDIO Power Enable register */
typedef struct{
__REG32 POWER_ENABLE    : 1;
__REG32                 :31;
} __sdio_pwren_bits;

/* SDIO Clock Divider register */
typedef struct{
__REG32 CLK_DIVIDER0    : 8;
__REG32 CLK_DIVIDER1    : 8;
__REG32 CLK_DIVIDER2    : 8;
__REG32 CLK_DIVIDER3    : 8;
} __sdio_clkdiv_bits;

/* SDIO Clock Source register */
typedef struct{
__REG32 CLK_SOURCE      : 2;
__REG32                 :30;
} __sdio_clksrc_bits;

/* SDIO Clock Enable register */
typedef struct{
__REG32 CCLK_ENABLE     : 1;
__REG32                 :15;
__REG32 CCLK_LOW_POWER  : 1;
__REG32                 :15;
} __sdio_clkena_bits;

/* SDIO Time-out register */
typedef struct{
__REG32 RESPONSE_TIMEOUT : 8;
__REG32 DATA_TIMEOUT     :24;
} __sdio_tmout_bits;

/* SDIO Card Type register */
typedef struct{
__REG32 CARD_WIDTH0        : 1;
__REG32                    :15;
__REG32 CARD_WIDTH1        : 1;
__REG32                    :15;
} __sdio_ctype_bits;

/* SDIO Block Size register */
typedef struct{
__REG32 BLOCK_SIZE      :16;
__REG32                 :16;
} __sdio_blksiz_bits;

/* SDIO Interrupt Mask register */
typedef struct{
__REG32 CDET              : 1;
__REG32 RE                : 1;
__REG32 CDONE             : 1;
__REG32 DTO               : 1;
__REG32 TXDR              : 1;
__REG32 RXDR              : 1;
__REG32 RCRC              : 1;
__REG32 DCRC              : 1;
__REG32 RTO               : 1;
__REG32 DRTO              : 1;
__REG32 HTO               : 1;
__REG32 FRUN              : 1;
__REG32 HLE               : 1;
__REG32 SBE               : 1;
__REG32 ACD               : 1;
__REG32 EBE               : 1;
__REG32 SDIO_INT_MASK     : 1;
__REG32                   :15;
} __sdio_intmask_bits;

/* SDIO Command register */
typedef struct{
__REG32 CMD_INDEX                  : 6;
__REG32 RESPONSE_EXPECT            : 1;
__REG32 RESPONSE_LENGTH            : 1;
__REG32 CHECK_RESPONSE_CRC         : 1;
__REG32 DATA_EXPECTED              : 1;
__REG32 READ_WRITE                 : 1;
__REG32 TRANSFER_MODE              : 1;
__REG32 SEND_AUTO_STOP             : 1;
__REG32 WAIT_PRVDATA_COMPLETE      : 1;
__REG32 STOP_ABORT_CMD             : 1;
__REG32 SEND_INITIALIZATION        : 1;
__REG32                            : 5;
__REG32 UPDATE_CLOC_REGISTERS_ONLY : 1;
__REG32 READ_CEATA_DEVICE          : 1;
__REG32 CCS_EXPECTED               : 1;
__REG32 ENABLE_BOOT                : 1;
__REG32 EXPECT_BOOT_ACK            : 1;
__REG32 DISABLE_BOOT               : 1;
__REG32 BOOT_MODE                  : 1;
__REG32 VOLT_SWITCH                : 1;
__REG32                            : 2;
__REG32 START_CMD                  : 1;
} __sdio_cmd_bits;

/* SDIO Masked Interrupt Status register */
typedef struct{
__REG32 CDET              : 1;
__REG32 RE                : 1;
__REG32 CDONE             : 1;
__REG32 DTO               : 1;
__REG32 TXDR              : 1;
__REG32 RXDR              : 1;
__REG32 RCRC              : 1;
__REG32 DCRC              : 1;
__REG32 RTO               : 1;
__REG32 DRTO              : 1;
__REG32 HTO               : 1;
__REG32 FRUN              : 1;
__REG32 HLE               : 1;
__REG32 SBE               : 1;
__REG32 ACD               : 1;
__REG32 EBE               : 1;
__REG32 SDIO_INTERRUPT    : 1;
__REG32                   :15;
} __sdio_mintsts_bits;

/* SDIO Raw Interrupt Status register */
typedef struct{
__REG32 CDET              : 1;
__REG32 RE                : 1;
__REG32 CDONE             : 1;
__REG32 DTO               : 1;
__REG32 TXDR              : 1;
__REG32 RXDR              : 1;
__REG32 RCRC              : 1;
__REG32 DCRC              : 1;
__REG32 RTO_BAR           : 1;
__REG32 DRTO_BDS          : 1;
__REG32 HTO               : 1;
__REG32 FRUN              : 1;
__REG32 HLE               : 1;
__REG32 SBE               : 1;
__REG32 ACD               : 1;
__REG32 EBE               : 1;
__REG32 SDIO_INTERRUPT    : 1;
__REG32                   :15;
} __sdio_rintsts_bits;

/* SDIO Status register */
typedef struct{
__REG32 FIFO_RX_WATERMARK   : 1;
__REG32 FIFO_TX_WATERMARK   : 1;
__REG32 FIFO_EMPTY          : 1;
__REG32 FIFO_FULL           : 1;
__REG32 CMDFSMSTATES        : 4;
__REG32 DATA_3_STATUS       : 1;
__REG32 DATA_BUSY           : 1;
__REG32 DATA_STATE_MC_BUSY  : 1;
__REG32 RESPONSE_INDEX      : 6;
__REG32 FIFO_COUNT          :13;
__REG32 DMA_ACK             : 1;
__REG32 DMA_REQ             : 1;
} __sdio_status_bits;

/* SDIO FIFO threshold Watermark register */
typedef struct{
__REG32 TX_WMARK                        :12;
__REG32                                 : 4;
__REG32 RX_WMARK                        :12;
__REG32 DW_DMA_MUTIPLE_TRANSACTION_SIZE : 3;
__REG32                                 : 1;
} __sdio_fifoth_bits;

/* SDIO Card Detect register */
typedef struct{
__REG32 CARD_DETECT         : 1;
__REG32                     :31;
} __sdio_cdetect_bits;

/* SDIO Write Protect register */
typedef struct{
__REG32 WRITE_PROTECT       : 1;
__REG32                     :31;
} __sdio_wrtprt_bits;

/* SDIO General Purpose Input/Ouput register */
/*
typedef struct{
__REG32 GPI                 : 8;
__REG32 GPO                 :16;
__REG32                     : 8;
} __sdio_gpio_bits;
*/

/* SDIO Debounce Count register */
typedef struct{
__REG32 DEBOUNCE_COUNT      :24;
__REG32                     : 8;
} __sdio_debnce_bits;

/* SDIO UHS-1 register */
typedef struct{
__REG32 VOLT_REG            : 1;
__REG32                     :15;
__REG32 DDR_REG             : 1;
__REG32                     :15;
} __sdio_uhs_reg_bits;

/* SDIO Hardware Reset register */
typedef struct{
__REG32 CARD_RESET          : 1;
__REG32                     :31;
} __sdio_rst_n_bits;

/* SDIO Bus Mode register */
typedef struct{
__REG32 SWR                 : 1;
__REG32 FB                  : 1;
__REG32 DSL                 : 5;
__REG32 DE                  : 1;
__REG32 PBL                 : 3;
__REG32                     :21;
} __sdio_bmod_bits;

/* SDIO Internal DMAC Status register */
typedef struct{
__REG32 TI                  : 1;
__REG32 RI                  : 1;
__REG32 FBE                 : 1;
__REG32                     : 1;
__REG32 DU                  : 1;
__REG32 CES                 : 1;
__REG32                     : 2;
__REG32 NIS                 : 1;
__REG32 AIS                 : 1;
__REG32 EB                  : 3;
__REG32 FSM                 : 4;
__REG32                     :15;
} __sdio_idsts_bits;

/* SDIO Internal DMAC Interrupt Enable register */
typedef struct{
__REG32 TI                  : 1;
__REG32 RI                  : 1;
__REG32 FBE                 : 1;
__REG32                     : 1;
__REG32 DU                  : 1;
__REG32 CES                 : 1;
__REG32                     : 2;
__REG32 NIS                 : 1;
__REG32 AIS                 : 1;
__REG32                     :22;
} __sdio_idinten_bits;


/* EMC Control Register */
typedef struct {
  __REG32 E         : 1;
  __REG32 M         : 1;
  __REG32 L         : 1;
  __REG32           :29;
} __emc_control_bits;

/* EMC Status Register */
typedef struct {
  __REG32 B         : 1;
  __REG32 S         : 1;
  __REG32 SA        : 1;
  __REG32           :29;
} __emc_status_bits;

/* EMC Configuration Register */
typedef struct {
  __REG32 EM        : 1;
  __REG32           : 7;
  __REG32 CR        : 1;
  __REG32           :23;
} __emc_config_bits;

/* Dynamic Memory Control Register */
typedef struct {
  __REG32 CE        : 1;
  __REG32 CS        : 1;
  __REG32 SR        : 1;
  __REG32           : 2;
  __REG32 MMC       : 1;
  __REG32           : 1;
  __REG32 I         : 2;
  __REG32           : 4;
  __REG32 DP        : 1;
  __REG32           :18;
} __emc_dctrl_bits;

/* Dynamic Memory Refresh Timer Register */
typedef struct {
  __REG32 REFRESH   :11;
  __REG32           :21;
} __emc_drfr_bits;

/* Dynamic Memory Read Configuration Register */
typedef struct {
  __REG32 RD        : 2;
  __REG32           :30;
} __emc_drdcfg_bits;

/* Dynamic Memory Percentage Command Period Register */
typedef struct {
  __REG32 tRP       : 4;
  __REG32           :28;
} __emc_drp_bits;

/* Dynamic Memory Active to Precharge Command Period Register */
typedef struct {
  __REG32 tRAS      : 4;
  __REG32           :28;
} __emc_dras_bits;

/* Dynamic Memory Self-refresh Exit Time Register */
typedef struct {
  __REG32 tSREX     : 4;
  __REG32           :28;
} __emc_dsrex_bits;

/* Dynamic Memory Last Data Out to Active Time Register */
typedef struct {
  __REG32 tAPR      : 4;
  __REG32           :28;
} __emc_dapr_bits;

/* Dynamic Memory Data-in to Active Command Time Register */
typedef struct {
  __REG32 tDAL      : 4;
  __REG32           :28;
} __emc_ddal_bits;

/* Dynamic Memory Write Recovery Time Register */
typedef struct {
  __REG32 tWR       : 4;
  __REG32           :28;
} __emc_dwr_bits;

/* Dynamic Memory Active to Active Command Period Register */
typedef struct {
  __REG32 tRC       : 5;
  __REG32           :27;
} __emc_drc_bits;

/* Dynamic Memory Auto-refresh Period Register */
typedef struct {
  __REG32 tRFC      : 5;
  __REG32           :27;
} __emc_drfc_bits;

/* Dynamic Memory Exit Self-refresh Register */
typedef struct {
  __REG32 tXSR      : 5;
  __REG32           :27;
} __emc_dxsr_bits;

/* Dynamic Memory Active Bank A to Active Bank B Time Register */
typedef struct {
  __REG32 tRRD      : 4;
  __REG32           :28;
} __emc_drrd_bits;

/* Dynamic Memory Load Mode Register to Active Command Time */
typedef struct {
  __REG32 tMRD      : 4;
  __REG32           :28;
} __emc_dmrd_bits;

/* Static Memory Extended Wait Register */
typedef struct {
  __REG32 EXTENDEDWAIT  :10;
  __REG32               :22;
} __emc_s_ext_wait_bits;

/* Dynamic Memory Configuration Registers */
typedef struct {
  __REG32           : 3;
  __REG32 MD        : 2;
  __REG32           : 2;
  __REG32 AM0       : 6;
  __REG32           : 1;
  __REG32 AM1       : 1;
  __REG32           : 4;
  __REG32 B         : 1;
  __REG32 P         : 1;
  __REG32           :11;
} __emc_d_config_bits;

/* Dynamic Memory RAS & CAS Delay Registers */
typedef struct {
  __REG32 RAS       : 2;
  __REG32           : 6;
  __REG32 CAS       : 2;
  __REG32           :22;
} __emc_d_ras_cas_bits;

/* Static Memory Configuration Registers */
typedef struct {
  __REG32 MW        : 2;
  __REG32           : 1;
  __REG32 PM        : 1;
  __REG32           : 2;
  __REG32 PC        : 1;
  __REG32 PB        : 1;
  __REG32 EW        : 1;
  __REG32           :10;
  __REG32 B         : 1;
  __REG32 P         : 1;
  __REG32           :11;
} __emc_s_config_bits;

/* Static Memory Write Enable Delay Registers */
typedef struct {
  __REG32 WAITWEN   : 4;
  __REG32           :28;
} __emc_s_wait_wen_bits;

/* Static Memory Output Enable Delay Registers */
typedef struct {
  __REG32 WAITOEN   : 4;
  __REG32           :28;
} __emc_s_wait_oen_bits;

/* Static Memory Read Delay Registers */
typedef struct {
  __REG32 WAITRD    : 5;
  __REG32           :27;
} __emc_s_wait_rd_bits;

/* Static Memory Page Mode Read Delay Registers */
typedef struct {
  __REG32 WAITPAGE  : 5;
  __REG32           :27;
} __emc_s_wait_pg_bits;

/* Static Memory Write Delay Registers */
typedef struct {
  __REG32 WAITWR    : 5;
  __REG32           :27;
} __emc_s_wait_wr_bits;

/* Static Memory Turn Round Delay Registers */
typedef struct {
  __REG32 WAITTURN  : 4;
  __REG32           :28;
} __emc_s_wait_turn_bits;


/* USB Device/host capability registers */
typedef struct{
__REG32 CAPLENGTH       : 8;
__REG32 HCIVERSION      :16;
__REG32                 : 8;
} __usb_caplength_reg_bits;

/* USB HCSPARAMS register */
typedef struct{
__REG32 N_PORTS         : 4;
__REG32 PPC             : 1;
__REG32                 : 3;
__REG32 N_PCC           : 4;
__REG32 N_CC            : 4;
__REG32 PI              : 1;
__REG32                 : 3;
__REG32 N_PTT           : 4;
__REG32 N_TT            : 4;
__REG32                 : 4;
} __usb_hcsparams_reg_bits;

/* USB HCCPARAMS register */
typedef struct{
__REG32 ADC             : 1;
__REG32 PFL             : 1;
__REG32 ASP             : 1;
__REG32                 : 1;
__REG32 IST             : 4;
__REG32 EECP            : 8;
__REG32                 :16;
} __usb_hccparams_reg_bits;

/* USB DCIVERSION register */
typedef struct{
__REG32 DCIVERSION      :16;
__REG32                 :16;
} __usb_dciversion_reg_bits;

/* USB DCCPARAMS */
typedef struct{
__REG32 DEN             : 5;
__REG32                 : 2;
__REG32 DC              : 1;
__REG32 HC              : 1;
__REG32                 :23;
} __usb_dccparams_reg_bits;

/* USB Command register */
typedef struct{
__REG32 RS              : 1;
__REG32 RST             : 1;
__REG32 FS0             : 1;
__REG32 FS1             : 1;
__REG32 PSE             : 1;
__REG32 ASE             : 1;
__REG32 IAA             : 1;
__REG32                 : 1;
__REG32 ASP1_0          : 2;
__REG32                 : 1;
__REG32 ASPE            : 1;
__REG32                 : 1;
__REG32 SUTW            : 1;
__REG32 ATDTW           : 1;
__REG32 FS2             : 1;
__REG32 ITC             : 8;
__REG32                 : 8;
} __usb_usbcmd_reg_bits;

/* USB Status register */
typedef struct{
__REG32 UI              : 1;
__REG32 UEI             : 1;
__REG32 PCI             : 1;
__REG32 FRI             : 1;
__REG32                 : 1;
__REG32 AAI             : 1;
__REG32 URI             : 1;
__REG32 SRI             : 1;
__REG32 SLI             : 1;
__REG32                 : 3;
__REG32 HCH             : 1;
__REG32 RCL             : 1;
__REG32 PS              : 1;
__REG32 AS              : 1;
__REG32 NAKI            : 1;
__REG32                 : 1;
__REG32 UAI             : 1;
__REG32 UPI             : 1;
__REG32                 :12;
} __usb_usbsts_reg_bits;

/* USB Interrupt register */
typedef struct{
__REG32 UE              : 1;
__REG32 UEE             : 1;
__REG32 PCE             : 1;
__REG32 FRE             : 1;
__REG32                 : 1;
__REG32 AAE             : 1;
__REG32 URE             : 1;
__REG32 SRE             : 1;
__REG32 SLE             : 1;
__REG32                 : 7;
__REG32 NAKE            : 1;
__REG32                 : 1;
__REG32 UAIE            : 1;
__REG32 UPIA            : 1;
__REG32                 :12;
} __usb_usbintr_reg_bits;

/* USB Frame index register */
typedef struct{
__REG32 FRINDEX2_0      : 3;
__REG32 FRINDEX12_3     :10;
__REG32                 :19;
} __usb_frindex_bits;

/* USB_PERIODICLISTBASE_DEVICEADDR_REG */
typedef union{
  /* USB0_PERIODICLISTBASE */
  /* USB1_PERIODICLISTBASE */
  struct{
  __REG32                 :12;
  __REG32 PERBASE31_12    :20;
  };
  /* USB0_DEVICEADDR */
  /* USB1_DEVICEADDR */
  struct{
  __REG32                 :24;
  __REG32 USBADRA         : 1;
  __REG32 USBADR          : 7;
  };
} __usb_periodiclistbase_reg_bits;

/* USB_ASYNCLISTADDR_ENDPOINTLISTADDR_REG */
typedef union{
  /* USB0_ASYNCLISTADDR */
  /* USB1_ASYNCLISTADDR */
  struct{
  __REG32                 : 5;
  __REG32 ASYBASE31_5     :27;
  };
  /* USB0_ENDPOINTLISTADDR */
  /* USB1_ENDPOINTLISTADDR */
  struct{
  __REG32                 :11;
  __REG32 EPBASE31_11     :21;
  };
} __usb_asynclistaddr_reg_bits;

/* TT Control register (TTCTRL) */
typedef struct{
__REG32                 :24;
__REG32 TTHA            : 7;
__REG32                 : 1;
} __usb_ttctrl_reg_bits;

/* Burst Size register (BURSTSIZE) */
typedef struct{
__REG32 RXPBURST        : 8;
__REG32 TXPBURST        : 8;
__REG32                 :16;
} __usb_burstsize_reg_bits;

/* Transfer buffer Fill Tuning register (TXFILLTUNING)*/
typedef struct{
__REG32 TXSCHOH         : 8;
__REG32 TXSCHEATLTH     : 5;
__REG32                 : 3;
__REG32 TXFIFOTHRES     : 6;
__REG32                 :10;
} __usb_txfilltuning_reg_bits;

/* USB ULPI viewport register */
typedef struct{
__REG32 ULPIDATWR       : 8;
__REG32 ULPIDATRD       : 8;
__REG32 ULPIADDR        : 8;
__REG32 ULPIPORT        : 3;
__REG32 ULPISS          : 1;
__REG32                 : 1;
__REG32 ULPIRW          : 1;
__REG32 ULPIRUN         : 1;
__REG32 ULPIWU          : 1;
} __usb1_ulpiviewport_reg_bits;

/* BINTERVAL register */
typedef struct{
__REG32 BINT            : 4;
__REG32                 :28;
} __usb_binterval_reg_bits;

/* USB Endpoint NAK register (ENDPTNAK) */
typedef union{
  struct{
  __REG32 EP0RN           : 1;
  __REG32 EP1RN           : 1;
  __REG32 EP2RN           : 1;
  __REG32 EP3RN           : 1;
  __REG32 EP4RN           : 1;
  __REG32 EP5RN           : 1;
  __REG32                 :10;
  __REG32 EP0TN           : 1;
  __REG32 EP1TN           : 1;
  __REG32 EP2TN           : 1;
  __REG32 EP3TN           : 1;
  __REG32 EP4TN           : 1;
  __REG32 EP5TN           : 1;
  __REG32                 :10;
  };
  /* USB0_ENDPTNAK */
  struct{
  __REG32 EPRN0           : 1;
  __REG32 EPRN1           : 1;
  __REG32 EPRN2           : 1;
  __REG32 EPRN3           : 1;
  __REG32 EPRN4           : 1;
  __REG32 EPRN5           : 1;
  __REG32                 :10;
  __REG32 EPTN0           : 1;
  __REG32 EPTN1           : 1;
  __REG32 EPTN2           : 1;
  __REG32 EPTN3           : 1;
  __REG32 EPTN4           : 1;
  __REG32 EPTN5           : 1;
  __REG32                 :10;
  };
} __usb_endptnak_reg_bits;

/* USB1 Endpoint NAK register (ENDPTNAK) */
typedef union {
  struct{
  __REG32 EP0RN           : 1;
  __REG32 EP1RN           : 1;
  __REG32 EP2RN           : 1;
  __REG32 EP3RN           : 1;
  __REG32                 :12;
  __REG32 EP0TN           : 1;
  __REG32 EP1TN           : 1;
  __REG32 EP2TN           : 1;
  __REG32 EP3TN           : 1;
  __REG32                 :12;
  };
  /* USB1_ENDPTNAK */
  struct{
  __REG32 EPRN0           : 1;
  __REG32 EPRN1           : 1;
  __REG32 EPRN2           : 1;
  __REG32 EPRN3           : 1;
  __REG32                 :12;
  __REG32 EPTN0           : 1;
  __REG32 EPTN1           : 1;
  __REG32 EPTN2           : 1;
  __REG32 EPTN3           : 1;
  __REG32                 :12;
  };
} __usb1_endptnak_reg_bits;

/* USB Endpoint NAK Enable register (ENDPTNAKEN) */
typedef union {
  struct{
  __REG32 EP0RNE          : 1;
  __REG32 EP1RNE          : 1;
  __REG32 EP2RNE          : 1;
  __REG32 EP3RNE          : 1;
  __REG32 EP4RNE          : 1;
  __REG32 EP5RNE          : 1;
  __REG32                 :10;
  __REG32 EP0TNE          : 1;
  __REG32 EP1TNE          : 1;
  __REG32 EP2TNE          : 1;
  __REG32 EP3TNE          : 1;
  __REG32 EP4TNE          : 1;
  __REG32 EP5TNE          : 1;
  __REG32                 :10;
  };
  /* USB0_ENDPTNAKEN */
  struct{
  __REG32 EPRNE0          : 1;
  __REG32 EPRNE1          : 1;
  __REG32 EPRNE2          : 1;
  __REG32 EPRNE3          : 1;
  __REG32 EPRNE4          : 1;
  __REG32 EPRNE5          : 1;
  __REG32                 :10;
  __REG32 EPTNE0          : 1;
  __REG32 EPTNE1          : 1;
  __REG32 EPTNE2          : 1;
  __REG32 EPTNE3          : 1;
  __REG32 EPTNE4          : 1;
  __REG32 EPTNE5          : 1;
  __REG32                 :10;
  };
} __usb_endptnaken_reg_bits;

/* USB Endpoint NAK Enable register (ENDPTNAKEN) */
typedef union {
  struct{
  __REG32 EP0RNE          : 1;
  __REG32 EP1RNE          : 1;
  __REG32 EP2RNE          : 1;
  __REG32 EP3RNE          : 1;
  __REG32                 :12;
  __REG32 EP0TNE          : 1;
  __REG32 EP1TNE          : 1;
  __REG32 EP2TNE          : 1;
  __REG32 EP3TNE          : 1;
  __REG32                 :12;
  };
  /* USB1_ENDPTNAKEN */
  struct{
  __REG32 EPRNE0          : 1;
  __REG32 EPRNE1          : 1;
  __REG32 EPRNE2          : 1;
  __REG32 EPRNE3          : 1;
  __REG32                 :12;
  __REG32 EPTNE0          : 1;
  __REG32 EPTNE1          : 1;
  __REG32 EPTNE2          : 1;
  __REG32 EPTNE3          : 1;
  __REG32                 :12;
  };
} __usb1_endptnaken_reg_bits;

/* Port Status and Control register (PORTSC1) */
typedef struct{
__REG32 CCS             : 1;
__REG32 CSC             : 1;
__REG32 PE              : 1;
__REG32 PEC             : 1;
__REG32 OCA             : 1;
__REG32 OCC             : 1;
__REG32 FPR             : 1;
__REG32 SUSP            : 1;
__REG32 PR              : 1;
__REG32 HSP             : 1;
__REG32 LS              : 2;
__REG32 PP              : 1;
__REG32                 : 1;
__REG32 PIC1_0          : 2;
__REG32 PTC3_0          : 4;
__REG32 WKCN            : 1;
__REG32 WKDC            : 1;
__REG32 WKOC            : 1;
__REG32 PHCD            : 1;
__REG32 PFSC            : 1;
__REG32                 : 1;
__REG32 PSPD            : 2;
__REG32                 : 4;
} __usb_portsc1_reg_bits;

/* OTG Status and Control register (OTGSC) */
typedef struct{
__REG32 VD              : 1;
__REG32 VC              : 1;
__REG32 HAAR            : 1;
__REG32 OT              : 1;
__REG32 DP              : 1;
__REG32 IDPU            : 1;
__REG32 HADP            : 1;
__REG32 HABA            : 1;
__REG32 ID              : 1;
__REG32 AVV             : 1;
__REG32 ASV             : 1;
__REG32 BSV             : 1;
__REG32 BSE             : 1;
__REG32 MS1T            : 1;
__REG32 DPS             : 1;
__REG32                 : 1;
__REG32 IDIS            : 1;
__REG32 AVVIS           : 1;
__REG32 ASVIS           : 1;
__REG32 BSVIS           : 1;
__REG32 BSEIS           : 1;
__REG32 ms1S            : 1;
__REG32 DPIS            : 1;
__REG32                 : 1;
__REG32 IDIE            : 1;
__REG32 AVVIE           : 1;
__REG32 ASVIE           : 1;
__REG32 BSVIE           : 1;
__REG32 BSEIE           : 1;
__REG32 MS1E            : 1;
__REG32 DPIE            : 1;
__REG32                 : 1;
} __usb_otgsc_reg_bits;

/* USB Mode register (USBMODE) */
typedef struct{
__REG32 CM1_0           : 2;
__REG32 ES              : 1;
__REG32 SLOM            : 1;
__REG32 SDIS            : 1;
__REG32 VBPS            : 1;
__REG32                 :26;
} __usb_usbmode_reg_bits;

/* USB Endpoint Setup Status register (ENDPSETUPSTAT) */
typedef union {
  struct{
  __REG32 ENDPT0SETUPSTAT : 1;
  __REG32 ENDPT1SETUPSTAT : 1;
  __REG32 ENDPT2SETUPSTAT : 1;
  __REG32 ENDPT3SETUPSTAT : 1;
  __REG32 ENDPT4SETUPSTAT : 1;
  __REG32 ENDPT5SETUPSTAT : 1;
  __REG32                 :26;
  };
  /* USB0_ENDPTSETUPSTAT */
  struct{
  __REG32 ENDPTSETUPSTAT0 : 1;
  __REG32 ENDPTSETUPSTAT1 : 1;
  __REG32 ENDPTSETUPSTAT2 : 1;
  __REG32 ENDPTSETUPSTAT3 : 1;
  __REG32 ENDPTSETUPSTAT4 : 1;
  __REG32 ENDPTSETUPSTAT5 : 1;
  __REG32                 :26;
  };
} __usb_endptsetupstat_reg_bits;

/* USB1 Endpoint Setup Status register (ENDPSETUPSTAT) */
typedef struct{
__REG32 ENDPT0SETUPSTAT : 1;
__REG32 ENDPT1SETUPSTAT : 1;
__REG32 ENDPT2SETUPSTAT : 1;
__REG32 ENDPT3SETUPSTAT : 1;
__REG32                 :28;
} __usb1_endptsetupstat_reg_bits;

/* USB Endpoint Prime register (ENDPTPRIME) */
typedef struct{
__REG32 PERB0           : 1;
__REG32 PERB1           : 1;
__REG32 PERB2           : 1;
__REG32 PERB3           : 1;
__REG32 PERB4           : 1;
__REG32 PERB5           : 1;
__REG32                 :10;
__REG32 PETB0           : 1;
__REG32 PETB1           : 1;
__REG32 PETB2           : 1;
__REG32 PETB3           : 1;
__REG32 PETB4           : 1;
__REG32 PETB5           : 1;
__REG32                 :10;
} __usb_endptprime_reg_bits;

/* USB1 Endpoint Prime register (ENDPTPRIME) */
typedef struct{
__REG32 PERB0           : 1;
__REG32 PERB1           : 1;
__REG32 PERB2           : 1;
__REG32 PERB3           : 1;
__REG32                 :12;
__REG32 PETB0           : 1;
__REG32 PETB1           : 1;
__REG32 PETB2           : 1;
__REG32 PETB3           : 1;
__REG32                 :12;
} __usb1_endptprime_reg_bits;

/* USB Endpoint Flush register (ENDPTFLUSH) */
typedef struct{
__REG32 FERB0           : 1;
__REG32 FERB1           : 1;
__REG32 FERB2           : 1;
__REG32 FERB3           : 1;
__REG32 FERB4           : 1;
__REG32 FERB5           : 1;
__REG32                 :10;
__REG32 FETB0           : 1;
__REG32 FETB1           : 1;
__REG32 FETB2           : 1;
__REG32 FETB3           : 1;
__REG32 FETB4           : 1;
__REG32 FETB5           : 1;
__REG32                 :10;
} __usb_endptflush_reg_bits;

/* USB1 Endpoint Flush register (ENDPTFLUSH) */
typedef struct{
__REG32 FERB0           : 1;
__REG32 FERB1           : 1;
__REG32 FERB2           : 1;
__REG32 FERB3           : 1;
__REG32                 :12;
__REG32 FETB0           : 1;
__REG32 FETB1           : 1;
__REG32 FETB2           : 1;
__REG32 FETB3           : 1;
__REG32                 :12;
} __usb1_endptflush_reg_bits;

/* USB Endpoint Status register (ENDPSTAT) */
typedef struct{
__REG32 ERBR0           : 1;
__REG32 ERBR1           : 1;
__REG32 ERBR2           : 1;
__REG32 ERBR3           : 1;
__REG32 ERBR4           : 1;
__REG32 ERBR5           : 1;
__REG32                 :10;
__REG32 ETBR0           : 1;
__REG32 ETBR1           : 1;
__REG32 ETBR2           : 1;
__REG32 ETBR3           : 1;
__REG32 ETBR4           : 1;
__REG32 ETBR5           : 1;
__REG32                 :10;
} __usb_endptstatus_reg_bits;

/* USB1 Endpoint Status register (ENDPSTAT) */
typedef struct{
__REG32 ERBR0           : 1;
__REG32 ERBR1           : 1;
__REG32 ERBR2           : 1;
__REG32 ERBR3           : 1;
__REG32                 :12;
__REG32 ETBR0           : 1;
__REG32 ETBR1           : 1;
__REG32 ETBR2           : 1;
__REG32 ETBR3           : 1;
__REG32                 :12;
} __usb1_endptstatus_reg_bits;

/* USB Endpoint Complete register (ENDPTCOMPLETE) */
typedef struct{
__REG32 ERCE0           : 1;
__REG32 ERCE1           : 1;
__REG32 ERCE2           : 1;
__REG32 ERCE3           : 1;
__REG32 ERCE4           : 1;
__REG32 ERCE5           : 1;
__REG32                 :10;
__REG32 ETCE0           : 1;
__REG32 ETCE1           : 1;
__REG32 ETCE2           : 1;
__REG32 ETCE3           : 1;
__REG32 ETCE4           : 1;
__REG32 ETCE5           : 1;
__REG32                 :10;
} __usb_endptcomplete_reg_bits;

/* USB1 Endpoint Complete register (ENDPTCOMPLETE) */
typedef struct{
__REG32 ERCE0           : 1;
__REG32 ERCE1           : 1;
__REG32 ERCE2           : 1;
__REG32 ERCE3           : 1;
__REG32                 :12;
__REG32 ETCE0           : 1;
__REG32 ETCE1           : 1;
__REG32 ETCE2           : 1;
__REG32 ETCE3           : 1;
__REG32                 :12;
} __usb1_endptcomplete_reg_bits;

/* USB Endpoint 0 Control register (ENDPTCTRL0) */
typedef struct{
__REG32 RXS             : 1;
__REG32                 : 1;
__REG32 RXT1_0          : 2;
__REG32                 : 3;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32                 : 1;
__REG32 TXT1_0          : 2;
__REG32                 : 3;
__REG32 TXE             : 1;
__REG32                 : 8;
} __usb_endptctrl0_reg_bits;

/* Endpoint 1 to 5 control registers */
typedef struct{
__REG32 RXS             : 1;
__REG32                 : 1;
__REG32 RXT             : 2;
__REG32                 : 1;
__REG32 RXI             : 1;
__REG32 RXR             : 1;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32                 : 1;
__REG32 TXT1_0          : 2;
__REG32                 : 1;
__REG32 TXI             : 1;
__REG32 TXR             : 1;
__REG32 TXE             : 1;
__REG32                 : 8;
} __usb_endptctrl_reg_bits;


/* MAC Configuration register */
typedef struct{
__REG32                 : 2;
__REG32 RE              : 1;
__REG32 TE              : 1;
__REG32 DF              : 1;
__REG32 BL              : 2;
__REG32 ACS             : 1;
__REG32                 : 1;
__REG32 DR              : 1;
__REG32 IPC             : 1;
__REG32 DM              : 1;
__REG32 LM              : 1;
__REG32 DO              : 1;
__REG32 FES             : 1;
__REG32 PS              : 1;
__REG32 DCRS            : 1;
__REG32 IFG             : 3;
__REG32 JE              : 1;
__REG32                 : 1;
__REG32 JD              : 1;
__REG32 WD              : 1;
__REG32                 : 8;
} __eth_mac_config_bits;

/* MAC Frame filter register */
typedef struct{
__REG32 PR              : 1;
__REG32                 : 2;
__REG32 DAIF            : 1;
__REG32 PM              : 1;
__REG32 DBF             : 1;
__REG32 PCF             : 2;
__REG32 SAIF            : 1;
__REG32 SAF             : 1;
__REG32                 :21;
__REG32 RA              : 1;
} __eth_mac_frame_filter_bits;

/* MAC GMII Address register */
typedef struct{
__REG32 GB              : 1;
__REG32 W               : 1;
__REG32 CR              : 4;
__REG32 GR              : 5;
__REG32 PA              : 5;
__REG32                 :16;
} __eth_mac_gmii_addr_bits;

/* MAC GMII Data register */
typedef struct{
__REG32 GD              :16;
__REG32                 :16;
} __eth_mac_gmii_data_bits;

/* MAC Flow control register */
typedef struct{
__REG32 FCB             : 1;
__REG32 TFE             : 1;
__REG32 RFE             : 1;
__REG32 UP              : 1;
__REG32 PLT             : 2;
__REG32                 : 1;
__REG32 DZPQ            : 1;
__REG32                 : 8;
__REG32 PT              :16;
} __eth_mac_flow_ctrl_bits;

/* MAC VLAN tag register */
typedef struct{
__REG32 VL              :16;
__REG32 ETV             : 1;
__REG32                 :15;
} __eth_mac_vlan_tag_bits;

/* MAC Debug register */
typedef struct{
__REG32 RXIDLESTAT      : 1;
__REG32 FIFOSTAT0       : 2;
__REG32                 : 1;
__REG32 RXFIFOSTAT1     : 1;
__REG32 RXFIFOSTAT      : 2;
__REG32                 : 1;
__REG32 RXFIFOLVL       : 2;
__REG32                 : 6;
__REG32 TXIDLESTAT      : 1;
__REG32 TXSTAT          : 2;
__REG32 PAUSE           : 1;
__REG32 TXFIFOSTAT      : 2;
__REG32 TXFIFOSTAT1     : 1;
__REG32                 : 1;
__REG32 TXFIFOLVL       : 1;
__REG32 TXFIFOFULL      : 1;
__REG32                 : 6;
} __eth_mac_debug_bits;

/* MAC PMT control and status register */
typedef struct{
__REG32 PD              : 1;
__REG32 MPE             : 1;
__REG32 WFE             : 1;
__REG32                 : 2;
__REG32 MPR             : 1;
__REG32 WFR             : 1;
__REG32                 : 2;
__REG32 GU              : 1;
__REG32                 :21;
__REG32 WFFRPR          : 1;
} __eth_mac_pmt_ctrl_stat_bits;

/* MAC Interrupt mask register */
typedef struct{
__REG32                 : 3;
__REG32 PMTMSK          : 1;
__REG32                 :28;
} __eth_mac_intr_mask_bits;

/* MAC Address 0 high register */
typedef struct{
__REG32 A47_32          :16;
__REG32                 :15;
__REG32 MO              : 1;
} __eth_mac_addr0_high_bits;

/* MAC IEEE1588 time stamp control register */
typedef struct{
__REG32 TSENA           : 1;
__REG32 TSCFUPDT        : 1;
__REG32 TSINIT          : 1;
__REG32 TSUPDT          : 1;
__REG32 TSTRIG          : 1;
__REG32 TSADDREG        : 1;
__REG32                 : 2;
__REG32 TSENALL         : 1;
__REG32 TSCTRLSSR       : 1;
__REG32 TSVER2ENA       : 1;
__REG32 TSIPENA         : 1;
__REG32 TSIPV6ENA       : 1;
__REG32 TSIPV4ENA       : 1;
__REG32 TSEVNTENA       : 1;
__REG32 TSMSTRENA       : 1;
__REG32 TSCLKTYPE       : 2;
__REG32 TSENMACADDR     : 1;
__REG32                 :13;
} __eth_mac_timestp_ctrl_bits;

/* Ethernet sub-second increment register */
typedef struct{
__REG32 SSINC           : 8;
__REG32                 :24;
} __eth_subsecond_incr_bits;

/* Ethernet system time seconds register */
typedef struct{
__REG32 TSS             :32;
} __eth_seconds_bits;

/* Ethernet system time nanoseconds register */
typedef struct{
__REG32 TSSS            :31;
__REG32 PSNT            :1;
} __eth_nanoseconds_bits;

/* Ethernet system time nanoseconds update register */
typedef struct{
__REG32 TSSS            :31;
__REG32 ADDSUB          :1;
} __eth_nanosecondsupdate_bits;

/* Ethernet timestamp addend register */
typedef struct{
__REG32 TSAR            :32;
} __eth_addend_bits;

/* Ethernet target time seconds register */
typedef struct{
__REG32 TSTR            :32;
} __eth_targetseconds_bits;

/* Ethernet target time nanoseconds register */
typedef struct{
__REG32 TSTR            :31;
__REG32                 : 1;
} __eth_targetnanoseconds_bits;

/* Ethernet system timer higher words seconds register */
typedef struct{
__REG32 TSHWR           :16;
__REG32                 :16;
} __eth_highword_bits;

/* Ethernet time stamp status register */
typedef struct{
__REG32 TSSOVF          :1;
__REG32 TSTARGT         :1;
__REG32 AUXSS           :1;
__REG32                 :21;
__REG32 ATSSTM          :1;
__REG32 ATSNS           :3;
__REG32                 :4;
} __eth_timestampstat_bits;

/* Ethernet PPS control register */
typedef struct{
__REG32 PPSCTRL         :4;
__REG32                 :28;
} __eth_ppsctrl_bits;

/* Ethernet auxiliary time stamp nanoseconds register */
typedef struct{
__REG32 AUXNS           :32;
} __eth_auxnanoseconds_bits;

/* Ethernet auxiliary time stamp seconds register */
typedef struct{
__REG32 AUXS            :32;
} __eth_auxseconds_bits;

/* DMA Bus mode register */
typedef struct{
__REG32 SWR             : 1;
__REG32 DA              : 1;
__REG32 DSL             : 5;
__REG32 ATDS            : 1;
__REG32 PBL             : 6;
__REG32 PR              : 2;
__REG32 FB              : 1;
__REG32 RPBL            : 6;
__REG32 USP             : 1;
__REG32 PBL8X           : 1;
__REG32 AAL             : 1;
__REG32 MB              : 1;
__REG32 TXPR            : 1;
__REG32                 : 2;
__REG32                 : 2;
} __eth_dma_bus_mode_bits;

/* DMA Status register */
typedef struct{
__REG32 TI              : 1;
__REG32 TPS             : 1;
__REG32 TU              : 1;
__REG32 TJT             : 1;
__REG32 OVF             : 1;
__REG32 UNF             : 1;
__REG32 RI              : 1;
__REG32 RU              : 1;
__REG32 RPS             : 1;
__REG32 RWT             : 1;
__REG32 ETI             : 1;
__REG32                 : 2;
__REG32 FBI             : 1;
__REG32 ERI             : 1;
__REG32 AIE             : 1;
__REG32 NIS             : 1;
__REG32                 :15;
} __eth_dma_stat_bits;

/* DMA Operation mode register */
typedef struct{
__REG32                 : 1;
__REG32 SR              : 1;
__REG32 OSF             : 1;
__REG32 RTC             : 2;
__REG32                 : 1;
__REG32 FUF             : 1;
__REG32 FEF             : 1;
__REG32                 : 5;
__REG32 ST              : 1;
__REG32 TTC             : 3;
__REG32                 : 3;
__REG32 FTF             : 1;
__REG32 TSF             : 1;
__REG32                 : 2;
__REG32 DFF             : 1;
__REG32 RSF             : 1;
__REG32 DT              : 1;
__REG32                 : 5;
} __eth_dma_op_mode_bits;

/* DMA Interrupt enable register */
typedef struct{
__REG32 TIE             : 1;
__REG32 TSE             : 1;
__REG32 TUE             : 1;
__REG32 TJE             : 1;
__REG32 OVE             : 1;
__REG32 UNE             : 1;
__REG32 RIE             : 1;
__REG32 RUE             : 1;
__REG32 RSE             : 1;
__REG32 RWE             : 1;
__REG32 ETE             : 1;
__REG32                 : 2;
__REG32 FBE             : 1;
__REG32 ERE             : 1;
__REG32 AIE             : 1;
__REG32 NIE             : 1;
__REG32                 :15;
} __eth_dma_int_en_bits;

/* DMA Missed frame and buffer overflow counter register */
typedef struct{
__REG32 FMC             :16;
__REG32 OC              : 1;
__REG32 FMA             :11;
__REG32 OF              : 1;
__REG32                 : 3;
} __eth_dma_mfrm_bufof_bits;

/* DMA Receive interrupt watchdog timer register */
typedef struct{
__REG32 RIWT            : 8;
__REG32                 :24;
} __eth_dma_rec_int_wdt_bits;


/* Horizontal Timing register */
typedef struct{
__REG32                 : 2;
__REG32 PPL             : 6;
__REG32 HSW             : 8;
__REG32 HFP             : 8;
__REG32 HBP             : 8;
}__lcd_timh_bits;

/* Vertical Timing register */
typedef struct{
__REG32 LPP             :10;
__REG32 VSW             : 6;
__REG32 VFP             : 8;
__REG32 VBP             : 8;
}__lcd_timv_bits;

/* Clock and Signal Polarity register */
typedef struct{
__REG32 PCD_LO          : 5;
__REG32 CLKSEL          : 1;
__REG32 ACB             : 5;
__REG32 IVS             : 1;
__REG32 IHS             : 1;
__REG32 IPC             : 1;
__REG32 IOE             : 1;
__REG32                 : 1;
__REG32 CPL             :10;
__REG32 BCD             : 1;
__REG32 PCD_HI          : 5;
}__lcd_pol_bits;

/* Line End Control register */
typedef struct{
__REG32 LED             : 7;
__REG32                 : 9;
__REG32 LEE             : 1;
__REG32                 :15;
}__lcd_le_bits;

/* Upper Panel Frame Base register */
typedef struct{
__REG32                 : 3;
__REG32 LCDUPBASE       :29;
}__lcd_upbase_bits;

/* Lower Panel Frame Base register */
typedef struct{
__REG32                 : 3;
__REG32 LCDLPBASE       :29;
}__lcd_lpbase_bits;

/* LCD Control register */
typedef struct{
__REG32 LCDEN           : 1;
__REG32 LCDBPP          : 3;
__REG32 LCDBW           : 1;
__REG32 LCDTFT          : 1;
__REG32 LCDMONO8        : 1;
__REG32 LCDDUAL         : 1;
__REG32 BGR             : 1;
__REG32 BEBO            : 1;
__REG32 BEPO            : 1;
__REG32 LCDPWR          : 1;
__REG32 LCDVCOMP        : 2;
__REG32                 : 2;
__REG32 WATERMARK       : 1;
__REG32                 :15;
}__lcd_ctrl_bits;

/* Interrupt Mask register */
typedef struct{
__REG32                 : 1;
__REG32 FUFIM           : 1;
__REG32 LNBUIM          : 1;
__REG32 VCOMPIM         : 1;
__REG32 BERIM           : 1;
__REG32                 :27;
}__lcd_intmsk_bits;

/* Raw Interrupt Status register */
typedef struct{
__REG32                 : 1;
__REG32 FUFRIS          : 1;
__REG32 LNBURIS         : 1;
__REG32 VCOMPRIS        : 1;
__REG32 BERRAW          : 1;
__REG32                 :27;
}__lcd_intraw_bits;

/* Masked Interrupt Status register */
typedef struct{
__REG32                 : 1;
__REG32 FUFMIS          : 1;
__REG32 LNBUMIS         : 1;
__REG32 VCOMPMIS        : 1;
__REG32 BERMIS          : 1;
__REG32                 :27;
}__lcd_intstat_bits;

/* Interrupt Clear register */

#define LCD_INTCLR_FUFIC    (0x00000002UL)
#define LCD_INTCLR_LNBUIC   (0x00000004UL)
#define LCD_INTCLR_VCOMPIC  (0x00000008UL)
#define LCD_INTCLR_BERIC    (0x00000010UL)

/* Color Palette register */
typedef struct{
__REG32 R04_0           : 5;
__REG32 G04_0           : 5;
__REG32 B04_0           : 5;
__REG32 I0              : 1;
__REG32 R14_0           : 5;
__REG32 G14_0           : 5;
__REG32 B14_0           : 5;
__REG32 I1              : 1;
}__lcd_pal_bits;

/* Cursor Control register */
typedef struct{
__REG32 CrsrOn          : 1;
__REG32                 : 3;
__REG32 CRSRNUM1_0      : 2;
__REG32                 :26;
}__lcd_crsr_ctrl_bits;

/* Cursor Configuration register */
typedef struct{
__REG32 CrsrSize        : 1;
__REG32 FRAMESYNC       : 1;
__REG32                 :30;
}__lcd_crsr_cfg_bits;

/* Cursor Palette register 0 */
typedef struct{
__REG32 RED             : 8;
__REG32 GREEN           : 8;
__REG32 BLUE            : 8;
__REG32                 : 8;
}__lcd_crsr_pal0_bits;

/* Cursor Palette register 1 */
typedef struct{
__REG32 RED             : 8;
__REG32 GREEN           : 8;
__REG32 BLUE            : 8;
__REG32                 : 8;
}__lcd_crsr_pal1_bits;

/* Cursor XY Position register */
typedef struct{
__REG32 CRSRX           :10;
__REG32                 : 6;
__REG32 CRSRY           :10;
__REG32                 : 6;
}__lcd_crsr_xy_bits;

/* Cursor Clip Position register */
typedef struct{
__REG32 CRSRCLIPX       : 6;
__REG32                 : 2;
__REG32 CRSRCLIPY       : 6;
__REG32                 :18;
}__lcd_crsr_clip_bits;

/* Cursor Interrupt Mask register */
typedef struct{
__REG32 CRSRIM          : 1;
__REG32                 :31;
}__lcd_crsr_intmsk_bits;

/* Cursor Interrupt Clear register */

#define LCD_CRSR_INTCLR_CRSRIC      (0x00000001UL)

/* Cursor Raw Interrupt Status register */
typedef struct{
__REG32 CRSRRIS         : 1;
__REG32                 :31;
}__lcd_crsr_intraw_bits;

/* Cursor Masked Interrupt Status register */
typedef struct{
__REG32 CRSRMIS         : 1;
__REG32                 :31;
}__lcd_crsr_intstat_bits;


/* SCT configuration register */
typedef struct{
__REG32 UNIFY                 : 1;
__REG32 CLKMODE               : 2;
__REG32 CLKSEL                : 4;
__REG32 NORELAODL_NORELOADU   : 1;
__REG32 NORELOADH             : 1;
__REG32 INSYNC0               : 1;
__REG32 INSYNC1               : 1;
__REG32 INSYNC2               : 1;
__REG32 INSYNC3               : 1;
__REG32 INSYNC4               : 1;
__REG32 INSYNC5               : 1;
__REG32 INSYNC6               : 1;
__REG32 INSYNC7               : 1;
__REG32                       :15;
} __sct_config_bits;

/* SCT control register */
typedef union{
  /* SCT_CTRL */
  struct{
  __REG32 DOWN_L                : 1;
  __REG32 STOP_L                : 1;
  __REG32 HALT_L                : 1;
  __REG32 CLRCTR_L              : 1;
  __REG32 BIDIR_L               : 1;
  __REG32 PRE_L                 : 8;
  __REG32                       : 3;
  __REG32 DOWN_H                : 1;
  __REG32 STOP_H                : 1;
  __REG32 HALT_H                : 1;
  __REG32 CLRCTR_H              : 1;
  __REG32 BIDIR_H               : 1;
  __REG32 PRE_H                 : 8;
  __REG32                       : 3;
  };

  struct{
    union{
      /* SCT_CTRL_L */
      struct{
      __REG16 DOWN_L                : 1;
      __REG16 STOP_L                : 1;
      __REG16 HALT_L                : 1;
      __REG16 CLRCTR_L              : 1;
      __REG16 BIDIR_L               : 1;
      __REG16 PRE_L                 : 8;
      __REG16                       : 3;
      } __sct_ctrl_l_bits;
      __REG16 __sct_ctrl_l;
    };
    union{
      /* SCT_CTRL_H */
      struct{
      __REG16 DOWN_H                : 1;
      __REG16 STOP_H                : 1;
      __REG16 HALT_H                : 1;
      __REG16 CLRCTR_H              : 1;
      __REG16 BIDIR_H               : 1;
      __REG16 PRE_H                 : 8;
      __REG16                       : 3;
      } __sct_ctrl_h_bits;
      __REG16 __sct_ctrl_h;
    };
  };
} __sct_ctrl_bits;


/* SCT limit register */
typedef union{
  /* SCT_LIMIT */
  struct{
  __REG32 LIMMSK0_L             : 1;
  __REG32 LIMMSK1_L             : 1;
  __REG32 LIMMSK2_L             : 1;
  __REG32 LIMMSK3_L             : 1;
  __REG32 LIMMSK4_L             : 1;
  __REG32 LIMMSK5_L             : 1;
  __REG32 LIMMSK6_L             : 1;
  __REG32 LIMMSK7_L             : 1;
  __REG32 LIMMSK8_L             : 1;
  __REG32 LIMMSK9_L             : 1;
  __REG32 LIMMSK10_L            : 1;
  __REG32 LIMMSK11_L            : 1;
  __REG32 LIMMSK12_L            : 1;
  __REG32 LIMMSK13_L            : 1;
  __REG32 LIMMSK14_L            : 1;
  __REG32 LIMMSK15_L            : 1;
  __REG32 LIMMSK0_H             : 1;
  __REG32 LIMMSK1_H             : 1;
  __REG32 LIMMSK2_H             : 1;
  __REG32 LIMMSK3_H             : 1;
  __REG32 LIMMSK4_H             : 1;
  __REG32 LIMMSK5_H             : 1;
  __REG32 LIMMSK6_H             : 1;
  __REG32 LIMMSK7_H             : 1;
  __REG32 LIMMSK8_H             : 1;
  __REG32 LIMMSK9_H             : 1;
  __REG32 LIMMSK10_H            : 1;
  __REG32 LIMMSK11_H            : 1;
  __REG32 LIMMSK12_H            : 1;
  __REG32 LIMMSK13_H            : 1;
  __REG32 LIMMSK14_H            : 1;
  __REG32 LIMMSK15_H            : 1;
  };

  struct{
    union{
      /* SCT_LIMIT_L */
      struct{
      __REG16 LIMMSK0_L             : 1;
      __REG16 LIMMSK1_L             : 1;
      __REG16 LIMMSK2_L             : 1;
      __REG16 LIMMSK3_L             : 1;
      __REG16 LIMMSK4_L             : 1;
      __REG16 LIMMSK5_L             : 1;
      __REG16 LIMMSK6_L             : 1;
      __REG16 LIMMSK7_L             : 1;
      __REG16 LIMMSK8_L             : 1;
      __REG16 LIMMSK9_L             : 1;
      __REG16 LIMMSK10_L            : 1;
      __REG16 LIMMSK11_L            : 1;
      __REG16 LIMMSK12_L            : 1;
      __REG16 LIMMSK13_L            : 1;
      __REG16 LIMMSK14_L            : 1;
      __REG16 LIMMSK15_L            : 1;
      } __sct_limit_l_bits;
      __REG16 __sct_limit_l;
    };
    union{
      /* SCT_LIMIT_H */
      struct{
      __REG16 LIMMSK0_H             : 1;
      __REG16 LIMMSK1_H             : 1;
      __REG16 LIMMSK2_H             : 1;
      __REG16 LIMMSK3_H             : 1;
      __REG16 LIMMSK4_H             : 1;
      __REG16 LIMMSK5_H             : 1;
      __REG16 LIMMSK6_H             : 1;
      __REG16 LIMMSK7_H             : 1;
      __REG16 LIMMSK8_H             : 1;
      __REG16 LIMMSK9_H             : 1;
      __REG16 LIMMSK10_H            : 1;
      __REG16 LIMMSK11_H            : 1;
      __REG16 LIMMSK12_H            : 1;
      __REG16 LIMMSK13_H            : 1;
      __REG16 LIMMSK14_H            : 1;
      __REG16 LIMMSK15_H            : 1;
      } __sct_limit_h_bits;
      __REG16 __sct_limit_h;
    };
  };
} __sct_limit_bits;

/* SCT halt condition register */
typedef union{
  /* SCT_HALT */
  struct{
  __REG32 HALTMSK0_L            : 1;
  __REG32 HALTMSK1_L            : 1;
  __REG32 HALTMSK2_L            : 1;
  __REG32 HALTMSK3_L            : 1;
  __REG32 HALTMSK4_L            : 1;
  __REG32 HALTMSK5_L            : 1;
  __REG32 HALTMSK6_L            : 1;
  __REG32 HALTMSK7_L            : 1;
  __REG32 HALTMSK8_L            : 1;
  __REG32 HALTMSK9_L            : 1;
  __REG32 HALTMSK10_L           : 1;
  __REG32 HALTMSK11_L           : 1;
  __REG32 HALTMSK12_L           : 1;
  __REG32 HALTMSK13_L           : 1;
  __REG32 HALTMSK14_L           : 1;
  __REG32 HALTMSK15_L           : 1;
  __REG32 HALTMSK0_H            : 1;
  __REG32 HALTMSK1_H            : 1;
  __REG32 HALTMSK2_H            : 1;
  __REG32 HALTMSK3_H            : 1;
  __REG32 HALTMSK4_H            : 1;
  __REG32 HALTMSK5_H            : 1;
  __REG32 HALTMSK6_H            : 1;
  __REG32 HALTMSK7_H            : 1;
  __REG32 HALTMSK8_H            : 1;
  __REG32 HALTMSK9_H            : 1;
  __REG32 HALTMSK10_H           : 1;
  __REG32 HALTMSK11_H           : 1;
  __REG32 HALTMSK12_H           : 1;
  __REG32 HALTMSK13_H           : 1;
  __REG32 HALTMSK14_H           : 1;
  __REG32 HALTMSK15_H           : 1;
  };

  struct{
    union{
      /* SCT_HALT_L */
      struct{
      __REG16 HALTMSK0_L            : 1;
      __REG16 HALTMSK1_L            : 1;
      __REG16 HALTMSK2_L            : 1;
      __REG16 HALTMSK3_L            : 1;
      __REG16 HALTMSK4_L            : 1;
      __REG16 HALTMSK5_L            : 1;
      __REG16 HALTMSK6_L            : 1;
      __REG16 HALTMSK7_L            : 1;
      __REG16 HALTMSK8_L            : 1;
      __REG16 HALTMSK9_L            : 1;
      __REG16 HALTMSK10_L           : 1;
      __REG16 HALTMSK11_L           : 1;
      __REG16 HALTMSK12_L           : 1;
      __REG16 HALTMSK13_L           : 1;
      __REG16 HALTMSK14_L           : 1;
      __REG16 HALTMSK15_L           : 1;
      } __sct_halt_l_bits;
      __REG16 __sct_halt_l;
    };
    union{
      /* SCT_HALT_H */
      struct{
      __REG16 HALTMSK0_H            : 1;
      __REG16 HALTMSK1_H            : 1;
      __REG16 HALTMSK2_H            : 1;
      __REG16 HALTMSK3_H            : 1;
      __REG16 HALTMSK4_H            : 1;
      __REG16 HALTMSK5_H            : 1;
      __REG16 HALTMSK6_H            : 1;
      __REG16 HALTMSK7_H            : 1;
      __REG16 HALTMSK8_H            : 1;
      __REG16 HALTMSK9_H            : 1;
      __REG16 HALTMSK10_H           : 1;
      __REG16 HALTMSK11_H           : 1;
      __REG16 HALTMSK12_H           : 1;
      __REG16 HALTMSK13_H           : 1;
      __REG16 HALTMSK14_H           : 1;
      __REG16 HALTMSK15_H           : 1;
      } __sct_halt_h_bits;
      __REG16 __sct_halt_h;
    };
  };
} __sct_halt_bits;


/* SCT stop condition register */
typedef union{
  /* SCT_STOP */
  struct{
  __REG32 STOPMSK0_L            : 1;
  __REG32 STOPMSK1_L            : 1;
  __REG32 STOPMSK2_L            : 1;
  __REG32 STOPMSK3_L            : 1;
  __REG32 STOPMSK4_L            : 1;
  __REG32 STOPMSK5_L            : 1;
  __REG32 STOPMSK6_L            : 1;
  __REG32 STOPMSK7_L            : 1;
  __REG32 STOPMSK8_L            : 1;
  __REG32 STOPMSK9_L            : 1;
  __REG32 STOPMSK10_L           : 1;
  __REG32 STOPMSK11_L           : 1;
  __REG32 STOPMSK12_L           : 1;
  __REG32 STOPMSK13_L           : 1;
  __REG32 STOPMSK14_L           : 1;
  __REG32 STOPMSK15_L           : 1;
  __REG32 STOPMSK0_H            : 1;
  __REG32 STOPMSK1_H            : 1;
  __REG32 STOPMSK2_H            : 1;
  __REG32 STOPMSK3_H            : 1;
  __REG32 STOPMSK4_H            : 1;
  __REG32 STOPMSK5_H            : 1;
  __REG32 STOPMSK6_H            : 1;
  __REG32 STOPMSK7_H            : 1;
  __REG32 STOPMSK8_H            : 1;
  __REG32 STOPMSK9_H            : 1;
  __REG32 STOPMSK10_H           : 1;
  __REG32 STOPMSK11_H           : 1;
  __REG32 STOPMSK12_H           : 1;
  __REG32 STOPMSK13_H           : 1;
  __REG32 STOPMSK14_H           : 1;
  __REG32 STOPMSK15_H           : 1;
  };

  struct{
    union{
      /* SCT_STOP_L */
      struct{
        __REG16 STOPMSK0_L            : 1;
        __REG16 STOPMSK1_L            : 1;
        __REG16 STOPMSK2_L            : 1;
        __REG16 STOPMSK3_L            : 1;
        __REG16 STOPMSK4_L            : 1;
        __REG16 STOPMSK5_L            : 1;
        __REG16 STOPMSK6_L            : 1;
        __REG16 STOPMSK7_L            : 1;
        __REG16 STOPMSK8_L            : 1;
        __REG16 STOPMSK9_L            : 1;
        __REG16 STOPMSK10_L           : 1;
        __REG16 STOPMSK11_L           : 1;
        __REG16 STOPMSK12_L           : 1;
        __REG16 STOPMSK13_L           : 1;
        __REG16 STOPMSK14_L           : 1;
        __REG16 STOPMSK15_L           : 1;
      } __sct_stop_l_bits;
      __REG16 __sct_stop_l;
    };
    union{
      /* SCT_STOP_H */
      struct{
      __REG16 STOPMSK0_H            : 1;
      __REG16 STOPMSK1_H            : 1;
      __REG16 STOPMSK2_H            : 1;
      __REG16 STOPMSK3_H            : 1;
      __REG16 STOPMSK4_H            : 1;
      __REG16 STOPMSK5_H            : 1;
      __REG16 STOPMSK6_H            : 1;
      __REG16 STOPMSK7_H            : 1;
      __REG16 STOPMSK8_H            : 1;
      __REG16 STOPMSK9_H            : 1;
      __REG16 STOPMSK10_H           : 1;
      __REG16 STOPMSK11_H           : 1;
      __REG16 STOPMSK12_H           : 1;
      __REG16 STOPMSK13_H           : 1;
      __REG16 STOPMSK14_H           : 1;
      __REG16 STOPMSK15_H           : 1;
      } __sct_stop_h_bits;
      __REG16 __sct_stop_h;
    };
  };
} __sct_stop_bits;

/* SCT start condition register */
typedef union{
  /* SCT_START */
  struct{
  __REG32 STARTMSK0_L           : 1;
  __REG32 STARTMSK1_L           : 1;
  __REG32 STARTMSK2_L           : 1;
  __REG32 STARTMSK3_L           : 1;
  __REG32 STARTMSK4_L           : 1;
  __REG32 STARTMSK5_L           : 1;
  __REG32 STARTMSK6_L           : 1;
  __REG32 STARTMSK7_L           : 1;
  __REG32 STARTMSK8_L           : 1;
  __REG32 STARTMSK9_L           : 1;
  __REG32 STARTMSK10_L          : 1;
  __REG32 STARTMSK11_L          : 1;
  __REG32 STARTMSK12_L          : 1;
  __REG32 STARTMSK13_L          : 1;
  __REG32 STARTMSK14_L          : 1;
  __REG32 STARTMSK15_L          : 1;
  __REG32 STARTMSK0_H           : 1;
  __REG32 STARTMSK1_H           : 1;
  __REG32 STARTMSK2_H           : 1;
  __REG32 STARTMSK3_H           : 1;
  __REG32 STARTMSK4_H           : 1;
  __REG32 STARTMSK5_H           : 1;
  __REG32 STARTMSK6_H           : 1;
  __REG32 STARTMSK7_H           : 1;
  __REG32 STARTMSK8_H           : 1;
  __REG32 STARTMSK9_H           : 1;
  __REG32 STARTMSK10_H          : 1;
  __REG32 STARTMSK11_H          : 1;
  __REG32 STARTMSK12_H          : 1;
  __REG32 STARTMSK13_H          : 1;
  __REG32 STARTMSK14_H          : 1;
  __REG32 STARTMSK15_H          : 1;
  };

  struct{
    union{
      /* SCT_START_L */
      struct{
      __REG16 STARTMSK0_L           : 1;
      __REG16 STARTMSK1_L           : 1;
      __REG16 STARTMSK2_L           : 1;
      __REG16 STARTMSK3_L           : 1;
      __REG16 STARTMSK4_L           : 1;
      __REG16 STARTMSK5_L           : 1;
      __REG16 STARTMSK6_L           : 1;
      __REG16 STARTMSK7_L           : 1;
      __REG16 STARTMSK8_L           : 1;
      __REG16 STARTMSK9_L           : 1;
      __REG16 STARTMSK10_L          : 1;
      __REG16 STARTMSK11_L          : 1;
      __REG16 STARTMSK12_L          : 1;
      __REG16 STARTMSK13_L          : 1;
      __REG16 STARTMSK14_L          : 1;
      __REG16 STARTMSK15_L          : 1;
      } __sct_start_l_bits;
      __REG16 __sct_start_l;
    };
    union{
      /* SCT_START_H */
      struct{
      __REG16 STARTMSK0_H           : 1;
      __REG16 STARTMSK1_H           : 1;
      __REG16 STARTMSK2_H           : 1;
      __REG16 STARTMSK3_H           : 1;
      __REG16 STARTMSK4_H           : 1;
      __REG16 STARTMSK5_H           : 1;
      __REG16 STARTMSK6_H           : 1;
      __REG16 STARTMSK7_H           : 1;
      __REG16 STARTMSK8_H           : 1;
      __REG16 STARTMSK9_H           : 1;
      __REG16 STARTMSK10_H          : 1;
      __REG16 STARTMSK11_H          : 1;
      __REG16 STARTMSK12_H          : 1;
      __REG16 STARTMSK13_H          : 1;
      __REG16 STARTMSK14_H          : 1;
      __REG16 STARTMSK15_H          : 1;
      } __sct_start_h_bits;
      __REG16 __sct_start_h;
    };
  };
} __sct_start_bits;

/* SCT counter register */
typedef union{
  /* SCT_COUNT */
  struct{
  __REG32 CTR_L                 :16;
  __REG32 CTR_H                 :16;
  };
  struct{
    __REG16 __sct_count_l;
    __REG16 __sct_count_h;
  };
} __sct_count_bits;

/* SCT state register */
typedef union{
   /* SCT_STATE */
  struct{
  __REG32 STATE_L               : 5;
  __REG32                       :11;
  __REG32 STATE_H               : 5;
  __REG32                       :11;
  };
  struct{
    union{
      /* SCT_STATE_L */
      struct{
      __REG16 STATE_L               : 5;
      __REG16                       :11;
      } __sct_state_l_bits;
      __REG16 __sct_state_l;
    };
    union{
      /* SCT_STATE_H */
      struct{
      __REG16 STATE_H               : 5;
      __REG16                       :11;
      } __sct_state_h_bits;
      __REG16 __sct_state_h;
    };
  };
} __sct_state_bits;

/* SCT input register */
typedef struct{
__REG32 AIN0                  : 1;
__REG32 AIN1                  : 1;
__REG32 AIN2                  : 1;
__REG32 AIN3                  : 1;
__REG32 AIN4                  : 1;
__REG32 AIN5                  : 1;
__REG32 AIN6                  : 1;
__REG32 AIN7                  : 1;
__REG32                       : 8;
__REG32 SIN0                  : 1;
__REG32 SIN1                  : 1;
__REG32 SIN2                  : 1;
__REG32 SIN3                  : 1;
__REG32 SIN4                  : 1;
__REG32 SIN5                  : 1;
__REG32 SIN6                  : 1;
__REG32 SIN7                  : 1;
__REG32                       : 8;
} __sct_input_bits;

/* SCT match/capture registers mode register */
typedef union{
  /* SCT_REGMODE */
  struct{
  __REG32 REGMOD0_L             : 1;
  __REG32 REGMOD1_L             : 1;
  __REG32 REGMOD2_L             : 1;
  __REG32 REGMOD3_L             : 1;
  __REG32 REGMOD4_L             : 1;
  __REG32 REGMOD5_L             : 1;
  __REG32 REGMOD6_L             : 1;
  __REG32 REGMOD7_L             : 1;
  __REG32 REGMOD8_L             : 1;
  __REG32 REGMOD9_L             : 1;
  __REG32 REGMOD10_L            : 1;
  __REG32 REGMOD11_L            : 1;
  __REG32 REGMOD12_L            : 1;
  __REG32 REGMOD13_L            : 1;
  __REG32 REGMOD14_L            : 1;
  __REG32 REGMOD15_L            : 1;
  __REG32 REGMOD0_H             : 1;
  __REG32 REGMOD1_H             : 1;
  __REG32 REGMOD2_H             : 1;
  __REG32 REGMOD3_H             : 1;
  __REG32 REGMOD4_H             : 1;
  __REG32 REGMOD5_H             : 1;
  __REG32 REGMOD6_H             : 1;
  __REG32 REGMOD7_H             : 1;
  __REG32 REGMOD8_H             : 1;
  __REG32 REGMOD9_H             : 1;
  __REG32 REGMOD10_H            : 1;
  __REG32 REGMOD11_H            : 1;
  __REG32 REGMOD12_H            : 1;
  __REG32 REGMOD13_H            : 1;
  __REG32 REGMOD14_H            : 1;
  __REG32 REGMOD15_H            : 1;
  };

  struct{
    union{
      /* SCT_REGMODE_L */
      struct{
      __REG16 REGMOD0_L             : 1;
      __REG16 REGMOD1_L             : 1;
      __REG16 REGMOD2_L             : 1;
      __REG16 REGMOD3_L             : 1;
      __REG16 REGMOD4_L             : 1;
      __REG16 REGMOD5_L             : 1;
      __REG16 REGMOD6_L             : 1;
      __REG16 REGMOD7_L             : 1;
      __REG16 REGMOD8_L             : 1;
      __REG16 REGMOD9_L             : 1;
      __REG16 REGMOD10_L            : 1;
      __REG16 REGMOD11_L            : 1;
      __REG16 REGMOD12_L            : 1;
      __REG16 REGMOD13_L            : 1;
      __REG16 REGMOD14_L            : 1;
      __REG16 REGMOD15_L            : 1;
      } __sct_regmode_l_bits;
      __REG16 __sct_regmode_l;
    };
    union{
      /* SCT_REGMODE_H */
      struct{
      __REG16 REGMOD0_H             : 1;
      __REG16 REGMOD1_H             : 1;
      __REG16 REGMOD2_H             : 1;
      __REG16 REGMOD3_H             : 1;
      __REG16 REGMOD4_H             : 1;
      __REG16 REGMOD5_H             : 1;
      __REG16 REGMOD6_H             : 1;
      __REG16 REGMOD7_H             : 1;
      __REG16 REGMOD8_H             : 1;
      __REG16 REGMOD9_H             : 1;
      __REG16 REGMOD10_H            : 1;
      __REG16 REGMOD11_H            : 1;
      __REG16 REGMOD12_H            : 1;
      __REG16 REGMOD13_H            : 1;
      __REG16 REGMOD14_H            : 1;
      __REG16 REGMOD15_H            : 1;
      } __sct_regmode_h_bits;
      __REG16 __sct_regmode_h;
    };
  };
} __sct_regmode_bits;

/* SCT output register */
typedef struct{
__REG32 OUT0                  : 1;
__REG32 OUT1                  : 1;
__REG32 OUT2                  : 1;
__REG32 OUT3                  : 1;
__REG32 OUT4                  : 1;
__REG32 OUT5                  : 1;
__REG32 OUT6                  : 1;
__REG32 OUT7                  : 1;
__REG32 OUT8                  : 1;
__REG32 OUT9                  : 1;
__REG32 OUT10                 : 1;
__REG32 OUT11                 : 1;
__REG32 OUT12                 : 1;
__REG32 OUT13                 : 1;
__REG32 OUT14                 : 1;
__REG32 OUT15                 : 1;
__REG32                       :16;
} __sct_output_bits;

/* SCT bidirectional output control register */
typedef struct{
__REG32 SETCLR0               : 2;
__REG32 SETCLR1               : 2;
__REG32 SETCLR2               : 2;
__REG32 SETCLR3               : 2;
__REG32 SETCLR4               : 2;
__REG32 SETCLR5               : 2;
__REG32 SETCLR6               : 2;
__REG32 SETCLR7               : 2;
__REG32 SETCLR8               : 2;
__REG32 SETCLR9               : 2;
__REG32 SETCLR10              : 2;
__REG32 SETCLR11              : 2;
__REG32 SETCLR12              : 2;
__REG32 SETCLR13              : 2;
__REG32 SETCLR14              : 2;
__REG32 SETCLR15              : 2;
} __sct_outputdirctrl_bits;

/* SCT conflict resolution register */
typedef struct{
__REG32 O0RES                 : 2;
__REG32 O1RES                 : 2;
__REG32 O2RES                 : 2;
__REG32 O3RES                 : 2;
__REG32 O4RES                 : 2;
__REG32 O5RES                 : 2;
__REG32 O6RES                 : 2;
__REG32 O7RES                 : 2;
__REG32 O8RES                 : 2;
__REG32 O9RES                 : 2;
__REG32 O10RES                : 2;
__REG32 O11RES                : 2;
__REG32 O12RES                : 2;
__REG32 O13RES                : 2;
__REG32 O14RES                : 2;
__REG32 O15RES                : 2;
} __sct_res_bits;

/* SCT DMA request 0 */
typedef struct{
__REG32 DEV0_0                : 1;
__REG32 DEV1_0                : 1;
__REG32 DEV2_0                : 1;
__REG32 DEV3_0                : 1;
__REG32 DEV4_0                : 1;
__REG32 DEV5_0                : 1;
__REG32 DEV6_0                : 1;
__REG32 DEV7_0                : 1;
__REG32 DEV8_0                : 1;
__REG32 DEV9_0                : 1;
__REG32 DEV10_0               : 1;
__REG32 DEV11_0               : 1;
__REG32 DEV12_0               : 1;
__REG32 DEV13_0               : 1;
__REG32 DEV14_0               : 1;
__REG32 DEV15_0               : 1;
__REG32                       :14;
__REG32 DRL0                  : 1;
__REG32 DRQ0                  : 1;
} __sct_dmareq0_bits;

/* SCT DMA request 1 */
typedef struct{
__REG32 DEV0_1                : 1;
__REG32 DEV1_1                : 1;
__REG32 DEV2_1                : 1;
__REG32 DEV3_1                : 1;
__REG32 DEV4_1                : 1;
__REG32 DEV5_1                : 1;
__REG32 DEV6_1                : 1;
__REG32 DEV7_1                : 1;
__REG32 DEV8_1                : 1;
__REG32 DEV9_1                : 1;
__REG32 DEV10_1               : 1;
__REG32 DEV11_1               : 1;
__REG32 DEV12_1               : 1;
__REG32 DEV13_1               : 1;
__REG32 DEV14_1               : 1;
__REG32 DEV15_1               : 1;
__REG32                       :14;
__REG32 DRL1                  : 1;
__REG32 DRQ1                  : 1;
} __sct_dmareq1_bits;

/* SCT flag enable register */
typedef struct{
__REG32 IEN0                  : 1;
__REG32 IEN1                  : 1;
__REG32 IEN2                  : 1;
__REG32 IEN3                  : 1;
__REG32 IEN4                  : 1;
__REG32 IEN5                  : 1;
__REG32 IEN6                  : 1;
__REG32 IEN7                  : 1;
__REG32 IEN8                  : 1;
__REG32 IEN9                  : 1;
__REG32 IEN10                 : 1;
__REG32 IEN11                 : 1;
__REG32 IEN12                 : 1;
__REG32 IEN13                 : 1;
__REG32 IEN14                 : 1;
__REG32 IEN15                 : 1;
__REG32                       :16;
} __sct_even_bits;

/* SCT event flag register */
typedef struct{
__REG32 FLAG0                 : 1;
__REG32 FLAG1                 : 1;
__REG32 FLAG2                 : 1;
__REG32 FLAG3                 : 1;
__REG32 FLAG4                 : 1;
__REG32 FLAG5                 : 1;
__REG32 FLAG6                 : 1;
__REG32 FLAG7                 : 1;
__REG32 FLAG8                 : 1;
__REG32 FLAG9                 : 1;
__REG32 FLAG10                : 1;
__REG32 FLAG11                : 1;
__REG32 FLAG12                : 1;
__REG32 FLAG13                : 1;
__REG32 FLAG14                : 1;
__REG32 FLAG15                : 1;
__REG32                       :16;
} __sct_evflag_bits;

/* SCT conflict enable register */
typedef struct{
__REG32 NCEN0                 : 1;
__REG32 NCEN1                 : 1;
__REG32 NCEN2                 : 1;
__REG32 NCEN3                 : 1;
__REG32 NCEN4                 : 1;
__REG32 NCEN5                 : 1;
__REG32 NCEN6                 : 1;
__REG32 NCEN7                 : 1;
__REG32 NCEN8                 : 1;
__REG32 NCEN9                 : 1;
__REG32 NCEN10                : 1;
__REG32 NCEN11                : 1;
__REG32 NCEN12                : 1;
__REG32 NCEN13                : 1;
__REG32 NCEN14                : 1;
__REG32 NCEN15                : 1;
__REG32                       :16;
} __sct_conen_bits;

/* SCT conflict flag register */
typedef struct{
__REG32 NCFLAG0               : 1;
__REG32 NCFLAG1               : 1;
__REG32 NCFLAG2               : 1;
__REG32 NCFLAG3               : 1;
__REG32 NCFLAG4               : 1;
__REG32 NCFLAG5               : 1;
__REG32 NCFLAG6               : 1;
__REG32 NCFLAG7               : 1;
__REG32 NCFLAG8               : 1;
__REG32 NCFLAG9               : 1;
__REG32 NCFLAG10              : 1;
__REG32 NCFLAG11              : 1;
__REG32 NCFLAG12              : 1;
__REG32 NCFLAG13              : 1;
__REG32 NCFLAG14              : 1;
__REG32 NCFLAG15              : 1;
__REG32                       :14;
__REG32 BUSERRL               : 1;
__REG32 BUSERRH               : 1;
} __sct_conflag_bits;

/* SCT match and capture registers */
typedef union{
  /* SCT_MATCHx */
  struct {
    __REG32 MATCH_L             :16;
    __REG32 MATCH_H             :16;
  };
  /* SCT_CAPx */
  struct {
    __REG32 CAP_L               :16;
    __REG32 CAP_H               :16;
  };

  struct{
    __REG16 __sct_match_cap_l;
    __REG16 __sct_match_cap_h;
  };

} __sct_match_cap_bits;

/* SCT match reload and capture control registers */
typedef union{
  /* SCT_MATCHRELx */
  struct {
    __REG32 RELOAD_L            :16;
    __REG32 RELOAD_H            :16;
  };

  struct{
    __REG16 __sct_matchrel_l;
    __REG16 __sct_matchrel_h;
  };

  /* SCT_CAPCTRLx */
  struct {
    __REG32 CAPCON0_L           : 1;
    __REG32 CAPCON1_L           : 1;
    __REG32 CAPCON2_L           : 1;
    __REG32 CAPCON3_L           : 1;
    __REG32 CAPCON4_L           : 1;
    __REG32 CAPCON5_L           : 1;
    __REG32 CAPCON6_L           : 1;
    __REG32 CAPCON7_L           : 1;
    __REG32 CAPCON8_L           : 1;
    __REG32 CAPCON9_L           : 1;
    __REG32 CAPCON10_L          : 1;
    __REG32 CAPCON11_L          : 1;
    __REG32 CAPCON12_L          : 1;
    __REG32 CAPCON13_L          : 1;
    __REG32 CAPCON14_L          : 1;
    __REG32 CAPCON15_L          : 1;
    __REG32 CAPCON0_H           : 1;
    __REG32 CAPCON1_H           : 1;
    __REG32 CAPCON2_H           : 1;
    __REG32 CAPCON3_H           : 1;
    __REG32 CAPCON4_H           : 1;
    __REG32 CAPCON5_H           : 1;
    __REG32 CAPCON6_H           : 1;
    __REG32 CAPCON7_H           : 1;
    __REG32 CAPCON8_H           : 1;
    __REG32 CAPCON9_H           : 1;
    __REG32 CAPCON10_H          : 1;
    __REG32 CAPCON11_H          : 1;
    __REG32 CAPCON12_H          : 1;
    __REG32 CAPCON13_H          : 1;
    __REG32 CAPCON14_H          : 1;
    __REG32 CAPCON15_H          : 1;
  };

  struct{
    union{
      /* SCT_CAPCTRLx_L */
      struct{
      __REG16 CAPCON0_L           : 1;
      __REG16 CAPCON1_L           : 1;
      __REG16 CAPCON2_L           : 1;
      __REG16 CAPCON3_L           : 1;
      __REG16 CAPCON4_L           : 1;
      __REG16 CAPCON5_L           : 1;
      __REG16 CAPCON6_L           : 1;
      __REG16 CAPCON7_L           : 1;
      __REG16 CAPCON8_L           : 1;
      __REG16 CAPCON9_L           : 1;
      __REG16 CAPCON10_L          : 1;
      __REG16 CAPCON11_L          : 1;
      __REG16 CAPCON12_L          : 1;
      __REG16 CAPCON13_L          : 1;
      __REG16 CAPCON14_L          : 1;
      __REG16 CAPCON15_L          : 1;
      } __sct_capctrl_l_bits;
      __REG16 __sct_capctrl_l;
    };
    union{
      /* SCT_CAPCTRLx_H */
      struct{
      __REG16 CAPCON0_H           : 1;
      __REG16 CAPCON1_H           : 1;
      __REG16 CAPCON2_H           : 1;
      __REG16 CAPCON3_H           : 1;
      __REG16 CAPCON4_H           : 1;
      __REG16 CAPCON5_H           : 1;
      __REG16 CAPCON6_H           : 1;
      __REG16 CAPCON7_H           : 1;
      __REG16 CAPCON8_H           : 1;
      __REG16 CAPCON9_H           : 1;
      __REG16 CAPCON10_H          : 1;
      __REG16 CAPCON11_H          : 1;
      __REG16 CAPCON12_H          : 1;
      __REG16 CAPCON13_H          : 1;
      __REG16 CAPCON14_H          : 1;
      __REG16 CAPCON15_H          : 1;
      } __sct_capctrl_h_bits;
      __REG16 __sct_capctrl_h;
    };
  };
} __sct_matchrel_capctrl_bits;

/* SCT event state mask registers */
typedef struct{
__REG32 STATEMSK0             : 1;
__REG32 STATEMSK1             : 1;
__REG32 STATEMSK2             : 1;
__REG32 STATEMSK3             : 1;
__REG32 STATEMSK4             : 1;
__REG32 STATEMSK5             : 1;
__REG32 STATEMSK6             : 1;
__REG32 STATEMSK7             : 1;
__REG32 STATEMSK8             : 1;
__REG32 STATEMSK9             : 1;
__REG32 STATEMSK10            : 1;
__REG32 STATEMSK11            : 1;
__REG32 STATEMSK12            : 1;
__REG32 STATEMSK13            : 1;
__REG32 STATEMSK14            : 1;
__REG32 STATEMSK15            : 1;
__REG32 STATEMSK16            : 1;
__REG32 STATEMSK17            : 1;
__REG32 STATEMSK18            : 1;
__REG32 STATEMSK19            : 1;
__REG32 STATEMSK20            : 1;
__REG32 STATEMSK21            : 1;
__REG32 STATEMSK22            : 1;
__REG32 STATEMSK23            : 1;
__REG32 STATEMSK24            : 1;
__REG32 STATEMSK25            : 1;
__REG32 STATEMSK26            : 1;
__REG32 STATEMSK27            : 1;
__REG32 STATEMSK28            : 1;
__REG32 STATEMSK29            : 1;
__REG32 STATEMSK30            : 1;
__REG32 STATEMSK31            : 1;
} __sct_evstatemsk_bits;

/* SCT event control registers */
typedef struct{
__REG32 MATCHSEL              : 4;
__REG32 HEVENT                : 1;
__REG32 OUTSEL                : 1;
__REG32 IOSEL                 : 4;
__REG32 IOCOND                : 2;
__REG32 COMBMODE              : 2;
__REG32 STATELD               : 1;
__REG32 STATEV                : 5;
__REG32                       :12;
} __sct_evctrl_bits;

/* SCT output set registers */
typedef struct{
__REG32 SET0                  : 1;
__REG32 SET1                  : 1;
__REG32 SET2                  : 1;
__REG32 SET3                  : 1;
__REG32 SET4                  : 1;
__REG32 SET5                  : 1;
__REG32 SET6                  : 1;
__REG32 SET7                  : 1;
__REG32 SET8                  : 1;
__REG32 SET9                  : 1;
__REG32 SET10                 : 1;
__REG32 SET11                 : 1;
__REG32 SET12                 : 1;
__REG32 SET13                 : 1;
__REG32 SET14                 : 1;
__REG32 SET15                 : 1;
__REG32                       :16;
} __sct_outputset_bits;

/* SCT output clear registers */
typedef struct{
__REG32 CLR0                  : 1;
__REG32 CLR1                  : 1;
__REG32 CLR2                  : 1;
__REG32 CLR3                  : 1;
__REG32 CLR4                  : 1;
__REG32 CLR5                  : 1;
__REG32 CLR6                  : 1;
__REG32 CLR7                  : 1;
__REG32 CLR8                  : 1;
__REG32 CLR9                  : 1;
__REG32 CLR10                 : 1;
__REG32 CLR11                 : 1;
__REG32 CLR12                 : 1;
__REG32 CLR13                 : 1;
__REG32 CLR14                 : 1;
__REG32 CLR15                 : 1;
__REG32                       :16;
} __sct_outputcl_bits;


/* TIMER interrupt register */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32 CR1INT  : 1;
__REG32 CR2INT  : 1;
__REG32 CR3INT  : 1;
__REG32         :24;
} __tmr_ir_bits;

/* TIMER control register */
typedef struct{
__REG32 CEN  : 1;
__REG32 CRST : 1;
__REG32      :30;
} __tmr_tcr_bits;

/* TIMER count control register */
typedef struct{
__REG32 CTMODE : 2;     /*Counter/Timer Mode*/
__REG32 CINSEL : 2;     /*Count Input Select*/
__REG32        :28;
} __tmr_ctcr_bits;

/* TIMER match control register */
typedef struct{
__REG32 MR0I     : 1;
__REG32 MR0R     : 1;
__REG32 MR0S     : 1;
__REG32 MR1I     : 1;
__REG32 MR1R     : 1;
__REG32 MR1S     : 1;
__REG32 MR2I     : 1;
__REG32 MR2R     : 1;
__REG32 MR2S     : 1;
__REG32 MR3I     : 1;
__REG32 MR3R     : 1;
__REG32 MR3S     : 1;
__REG32          :20;
} __tmr_mcr_bits;

/* TIMER capture control register */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32 CAP1RE   : 1;
__REG32 CAP1FE   : 1;
__REG32 CAP1I    : 1;
__REG32 CAP2RE   : 1;
__REG32 CAP2FE   : 1;
__REG32 CAP2I    : 1;
__REG32 CAP3RE   : 1;
__REG32 CAP3FE   : 1;
__REG32 CAP3I    : 1;
__REG32          :20;
} __tmr_tccr_bits;

/* TIMER external match register */
typedef struct{
__REG32 EM0   : 1;
__REG32 EM1   : 1;
__REG32 EM2   : 1;
__REG32 EM3   : 1;
__REG32 EMC0  : 2;
__REG32 EMC1  : 2;
__REG32 EMC2  : 2;
__REG32 EMC3  : 2;
__REG32       :20;
} __tmr_emr_bits;


/* MCPWM Control read  */
typedef struct{
__REG32 RUN0    : 1;
__REG32 CENTER0 : 1;
__REG32 POLA0   : 1;
__REG32 DTE0    : 1;
__REG32 DISUP0  : 1;
__REG32         : 3;
__REG32 RUN1    : 1;
__REG32 CENTER1 : 1;
__REG32 POLA1   : 1;
__REG32 DTE1    : 1;
__REG32 DISUP1  : 1;
__REG32         : 3;
__REG32 RUN2    : 1;
__REG32 CENTER2 : 1;
__REG32 POLA2   : 1;
__REG32 DTE2    : 1;
__REG32 DISUP2  : 1;
__REG32         : 8;
__REG32 INVBDC  : 1;
__REG32 ACMODE  : 1;
__REG32 DCMODE  : 1;
} __mc_con_bits;

/* MCPWM Control set  */

#define MC_CON_SET_RUN0_SET       (0x00000001UL)
#define MC_CON_SET_CENTER0_SET    (0x00000002UL)
#define MC_CON_SET_POLA0_SET      (0x00000004UL)
#define MC_CON_SET_DTE0_SET       (0x00000008UL)
#define MC_CON_SET_DISUP0_SET     (0x00000010UL)

#define MC_CON_SET_RUN1_SET       (0x00000100UL)
#define MC_CON_SET_CENTER1_SET    (0x00000200UL)
#define MC_CON_SET_POLA1_SET      (0x00000400UL)
#define MC_CON_SET_DTE1_SET       (0x00000800UL)
#define MC_CON_SET_DISUP1_SET     (0x00001000UL)

#define MC_CON_SET_RUN2_SET       (0x00010000UL)
#define MC_CON_SET_CENTER2_SET    (0x00020000UL)
#define MC_CON_SET_POLA2_SET      (0x00040000UL)
#define MC_CON_SET_DTE2_SET       (0x00080000UL)
#define MC_CON_SET_DISUP2_SET     (0x00100000UL)

#define MC_CON_SET_INVBDC_SET     (0x20000000UL)
#define MC_CON_SET_ACMODE_SET     (0x40000000UL)
#define MC_CON_SET_DCMODE_SET     (0x80000000UL)

/* MCPWM Control clear  */

#define MC_CON_CLR_RUN0_CLR       (0x00000001UL)
#define MC_CON_CLR_CENTER0_CLR    (0x00000002UL)
#define MC_CON_CLR_POLA0_CLR      (0x00000004UL)
#define MC_CON_CLR_DTE0_CLR       (0x00000008UL)
#define MC_CON_CLR_DISUP0_CLR     (0x00000010UL)

#define MC_CON_CLR_RUN1_CLR       (0x00000100UL)
#define MC_CON_CLR_CENTER1_CLR    (0x00000200UL)
#define MC_CON_CLR_POLA1_CLR      (0x00000400UL)
#define MC_CON_CLR_DTE1_CLR       (0x00000800UL)
#define MC_CON_CLR_DISUP1_CLR     (0x00001000UL)

#define MC_CON_CLR_RUN2_CLR       (0x00010000UL)
#define MC_CON_CLR_CENTER2_CLR    (0x00020000UL)
#define MC_CON_CLR_POLA2_CLR      (0x00040000UL)
#define MC_CON_CLR_DTE2_CLR       (0x00080000UL)
#define MC_CON_CLR_DISUP2_CLR     (0x00100000UL)

#define MC_CON_CLR_INVBDC_CLR     (0x20000000UL)
#define MC_CON_CLR_ACMODE_CLR     (0x40000000UL)
#define MC_CON_CLR_DCMODE_CLR     (0x80000000UL)

/* MCPWM Capture control register */
typedef struct{
__REG32 CAP0MCI0_RE : 1;
__REG32 CAP0MCI0_FE : 1;
__REG32 CAP0MCI1_RE : 1;
__REG32 CAP0MCI1_FE : 1;
__REG32 CAP0MCI2_RE : 1;
__REG32 CAP0MCI2_FE : 1;
__REG32 CAP1MCI0_RE : 1;
__REG32 CAP1MCI0_FE : 1;
__REG32 CAP1MCI1_RE : 1;
__REG32 CAP1MCI1_FE : 1;
__REG32 CAP1MCI2_RE : 1;
__REG32 CAP1MCI2_FE : 1;
__REG32 CAP2MCI0_RE : 1;
__REG32 CAP2MCI0_FE : 1;
__REG32 CAP2MCI1_RE : 1;
__REG32 CAP2MCI1_FE : 1;
__REG32 CAP2MCI2_RE : 1;
__REG32 CAP2MCI2_FE : 1;
__REG32 RT0         : 1;
__REG32 RT1         : 1;
__REG32 RT2         : 1;
__REG32 HNFCAP0     : 1;
__REG32 HNFCAP1     : 1;
__REG32 HNFCAP2     : 1;
__REG32             : 8;
} __mc_capcon_bits;

/* MCPWM Capture Control set  */

#define MC_CAPCON_SET_CAP0MCI0_RE_SET    (0x00000001UL)
#define MC_CAPCON_SET_CAP0MCI0_FE_SET    (0x00000002UL)
#define MC_CAPCON_SET_CAP0MCI1_RE_SET    (0x00000004UL)
#define MC_CAPCON_SET_CAP0MCI1_FE_SET    (0x00000008UL)
#define MC_CAPCON_SET_CAP0MCI2_RE_SET    (0x00000010UL)
#define MC_CAPCON_SET_CAP0MCI2_FE_SET    (0x00000020UL)
#define MC_CAPCON_SET_CAP1MCI0_RE_SET    (0x00000040UL)
#define MC_CAPCON_SET_CAP1MCI0_FE_SET    (0x00000080UL)
#define MC_CAPCON_SET_CAP1MCI1_RE_SET    (0x00000100UL)
#define MC_CAPCON_SET_CAP1MCI1_FE_SET    (0x00000200UL)
#define MC_CAPCON_SET_CAP1MCI2_RE_SET    (0x00000400UL)
#define MC_CAPCON_SET_CAP1MCI2_FE_SET    (0x00000800UL)
#define MC_CAPCON_SET_CAP2MCI0_RE_SET    (0x00001000UL)
#define MC_CAPCON_SET_CAP2MCI0_FE_SET    (0x00002000UL)
#define MC_CAPCON_SET_CAP2MCI1_RE_SET    (0x00004000UL)
#define MC_CAPCON_SET_CAP2MCI1_FE_SET    (0x00008000UL)
#define MC_CAPCON_SET_CAP2MCI2_RE_SET    (0x00010000UL)
#define MC_CAPCON_SET_CAP2MCI2_FE_SET    (0x00020000UL)
#define MC_CAPCON_SET_RT0_SET            (0x00040000UL)
#define MC_CAPCON_SET_RT1_SET            (0x00080000UL)
#define MC_CAPCON_SET_RT2_SET            (0x00100000UL)
#define MC_CAPCON_SET_HNFCAP0_SET        (0x00200000UL)
#define MC_CAPCON_SET_HNFCAP1_SET        (0x00400000UL)
#define MC_CAPCON_SET_HNFCAP2_SET        (0x00800000UL)

/* MCPWM Capture control clear  */

#define MC_CAPCON_CLR_CAP0MCI0_RE_CLR    (0x00000001UL)
#define MC_CAPCON_CLR_CAP0MCI0_FE_CLR    (0x00000002UL)
#define MC_CAPCON_CLR_CAP0MCI1_RE_CLR    (0x00000004UL)
#define MC_CAPCON_CLR_CAP0MCI1_FE_CLR    (0x00000008UL)
#define MC_CAPCON_CLR_CAP0MCI2_RE_CLR    (0x00000010UL)
#define MC_CAPCON_CLR_CAP0MCI2_FE_CLR    (0x00000020UL)
#define MC_CAPCON_CLR_CAP1MCI0_RE_CLR    (0x00000040UL)
#define MC_CAPCON_CLR_CAP1MCI0_FE_CLR    (0x00000080UL)
#define MC_CAPCON_CLR_CAP1MCI1_RE_CLR    (0x00000100UL)
#define MC_CAPCON_CLR_CAP1MCI1_FE_CLR    (0x00000200UL)
#define MC_CAPCON_CLR_CAP1MCI2_RE_CLR    (0x00000400UL)
#define MC_CAPCON_CLR_CAP1MCI2_FE_CLR    (0x00000800UL)
#define MC_CAPCON_CLR_CAP2MCI0_RE_CLR    (0x00001000UL)
#define MC_CAPCON_CLR_CAP2MCI0_FE_CLR    (0x00002000UL)
#define MC_CAPCON_CLR_CAP2MCI1_RE_CLR    (0x00004000UL)
#define MC_CAPCON_CLR_CAP2MCI1_FE_CLR    (0x00008000UL)
#define MC_CAPCON_CLR_CAP2MCI2_RE_CLR    (0x00010000UL)
#define MC_CAPCON_CLR_CAP2MCI2_FE_CLR    (0x00020000UL)
#define MC_CAPCON_CLR_RT0_CLR            (0x00040000UL)
#define MC_CAPCON_CLR_RT1_CLR            (0x00080000UL)
#define MC_CAPCON_CLR_RT2_CLR            (0x00100000UL)
#define MC_CAPCON_CLR_HNFCAP0_CLR        (0x00200000UL)
#define MC_CAPCON_CLR_HNFCAP1_CLR        (0x00400000UL)
#define MC_CAPCON_CLR_HNFCAP2_CLR        (0x00800000UL)

/* MCPWM interrupt enable register */
typedef struct{
__REG32 ILIM0       : 1;
__REG32 IMAT0       : 1;
__REG32 ICAP0       : 1;
__REG32             : 1;
__REG32 ILIM1       : 1;
__REG32 IMAT1       : 1;
__REG32 ICAP1       : 1;
__REG32             : 1;
__REG32 ILIM2       : 1;
__REG32 IMAT2       : 1;
__REG32 ICAP2       : 1;
__REG32             : 4;
__REG32 ABORT       : 1;
__REG32             :16;
} __mc_inten_bits;

/* MCPWM Interrupt Enable set */

#define MC_INTEN_SET_ILIM0_SET      (0x00000001UL)
#define MC_INTEN_SET_IMAT0_SET      (0x00000002UL)
#define MC_INTEN_SET_ICAP0_SET      (0x00000004UL)
#define MC_INTEN_SET_ILIM1_SET      (0x00000010UL)
#define MC_INTEN_SET_IMAT1_SET      (0x00000020UL)
#define MC_INTEN_SET_ICAP1_SET      (0x00000040UL)
#define MC_INTEN_SET_ILIM2_SET      (0x00000100UL)
#define MC_INTEN_SET_IMAT2_SET      (0x00000200UL)
#define MC_INTEN_SET_ICAP2_SET      (0x00000400UL)
#define MC_INTEN_SET_ABORT_SET      (0x00008000UL)

/* MCPWM Interrupt Enable clear */

#define MC_INTEN_CLR_ILIM0_CLR      (0x00000001UL)
#define MC_INTEN_CLR_IMAT0_CLR      (0x00000002UL)
#define MC_INTEN_CLR_ICAP0_CLR      (0x00000004UL)
#define MC_INTEN_CLR_ILIM1_CLR      (0x00000010UL)
#define MC_INTEN_CLR_IMAT1_CLR      (0x00000020UL)
#define MC_INTEN_CLR_ICAP1_CLR      (0x00000040UL)
#define MC_INTEN_CLR_ILIM2_CLR      (0x00000100UL)
#define MC_INTEN_CLR_IMAT2_CLR      (0x00000200UL)
#define MC_INTEN_CLR_ICAP2_CLR      (0x00000400UL)
#define MC_INTEN_CLR_ABORT_CLR      (0x00008000UL)

/* MCPWM Interrupt Flags */
typedef struct{
__REG32 ILIM0_F       : 1;
__REG32 IMAT0_F       : 1;
__REG32 ICAP0_F       : 1;
__REG32               : 1;
__REG32 ILIM1_F       : 1;
__REG32 IMAT1_F       : 1;
__REG32 ICAP1_F       : 1;
__REG32               : 1;
__REG32 ILIM2_F       : 1;
__REG32 IMAT2_F       : 1;
__REG32 ICAP2_F       : 1;
__REG32               : 4;
__REG32 ABORT_F       : 1;
__REG32               :16;
} __mc_intf_bits;

/* MCPWM Interrupt Flags set */

#define MC_INTEN_SET_ILIM0_F_SET      (0x00000001UL)
#define MC_INTEN_SET_IMAT0_F_SET      (0x00000002UL)
#define MC_INTEN_SET_ICAP0_F_SET      (0x00000004UL)
#define MC_INTEN_SET_ILIM1_F_SET      (0x00000010UL)
#define MC_INTEN_SET_IMAT1_F_SET      (0x00000020UL)
#define MC_INTEN_SET_ICAP1_F_SET      (0x00000040UL)
#define MC_INTEN_SET_ILIM2_F_SET      (0x00000100UL)
#define MC_INTEN_SET_IMAT2_F_SET      (0x00000200UL)
#define MC_INTEN_SET_ICAP2_F_SET      (0x00000400UL)
#define MC_INTEN_SET_ABORT_F_SET      (0x00008000UL)

/* MCPWM Interrupt Flags clear */

#define MC_INTEN_CLR_ILIM0_F_CLR      (0x00000001UL)
#define MC_INTEN_CLR_IMAT0_F_CLR      (0x00000002UL)
#define MC_INTEN_CLR_ICAP0_F_CLR      (0x00000004UL)
#define MC_INTEN_CLR_ILIM1_F_CLR      (0x00000010UL)
#define MC_INTEN_CLR_IMAT1_F_CLR      (0x00000020UL)
#define MC_INTEN_CLR_ICAP1_F_CLR      (0x00000040UL)
#define MC_INTEN_CLR_ILIM2_F_CLR      (0x00000100UL)
#define MC_INTEN_CLR_IMAT2_F_CLR      (0x00000200UL)
#define MC_INTEN_CLR_ICAP2_F_CLR      (0x00000400UL)
#define MC_INTEN_CLR_ABORT_F_CLR      (0x00008000UL)

/* Count control register */
typedef struct{
__REG32 TC0MCI0_RE  : 1;
__REG32 TC0MCI0_FE  : 1;
__REG32 TC0MCI1_RE  : 1;
__REG32 TC0MCI1_FE  : 1;
__REG32 TC0MCI2_RE  : 1;
__REG32 TC0MCI2_FE  : 1;
__REG32 TC1MCI0_RE  : 1;
__REG32 TC1MCI0_FE  : 1;
__REG32 TC1MCI1_RE  : 1;
__REG32 TC1MCI1_FE  : 1;
__REG32 TC1MCI2_RE  : 1;
__REG32 TC1MCI2_FE  : 1;
__REG32 TC2MCI0_RE  : 1;
__REG32 TC2MCI0_FE  : 1;
__REG32 TC2MCI1_RE  : 1;
__REG32 TC2MCI1_FE  : 1;
__REG32 TC2MCI2_RE  : 1;
__REG32 TC2MCI2_FE  : 1;
__REG32             :11;
__REG32 CNTR0       : 1;
__REG32 CNTR1       : 1;
__REG32 CNTR2       : 1;
} __mc_cntcon_bits;

/* Count control register set */

#define MC_CNTCON_SET_TC0MCI0_RE_SET      (0x00000001UL)
#define MC_CNTCON_SET_TC0MCI0_FE_SET      (0x00000002UL)
#define MC_CNTCON_SET_TC0MCI1_RE_SET      (0x00000004UL)
#define MC_CNTCON_SET_TC0MCI1_FE_SET      (0x00000008UL)
#define MC_CNTCON_SET_TC0MCI2_RE_SET      (0x00000010UL)
#define MC_CNTCON_SET_TC0MCI2_FE_SET      (0x00000020UL)
#define MC_CNTCON_SET_TC1MCI0_RE_SET      (0x00000040UL)
#define MC_CNTCON_SET_TC1MCI0_FE_SET      (0x00000080UL)
#define MC_CNTCON_SET_TC1MCI1_RE_SET      (0x00000100UL)
#define MC_CNTCON_SET_TC1MCI1_FE_SET      (0x00000200UL)
#define MC_CNTCON_SET_TC1MCI2_RE_SET      (0x00000400UL)
#define MC_CNTCON_SET_TC1MCI2_FE_SET      (0x00000800UL)
#define MC_CNTCON_SET_TC2MCI0_RE_SET      (0x00001000UL)
#define MC_CNTCON_SET_TC2MCI0_FE_SET      (0x00002000UL)
#define MC_CNTCON_SET_TC2MCI1_RE_SET      (0x00004000UL)
#define MC_CNTCON_SET_TC2MCI1_FE_SET      (0x00008000UL)
#define MC_CNTCON_SET_TC2MCI2_RE_SET      (0x00010000UL)
#define MC_CNTCON_SET_TC2MCI2_FE_SET      (0x00020000UL)
#define MC_CNTCON_SET_CNTR0_SET           (0x20000000UL)
#define MC_CNTCON_SET_CNTR1_SET           (0x40000000UL)
#define MC_CNTCON_SET_CNTR2_SET           (0x80000000UL)

/* Count control register */

#define MC_CNTCON_CLR_TC0MCI0_RE_CLR      (0x00000001UL)
#define MC_CNTCON_CLR_TC0MCI0_FE_CLR      (0x00000002UL)
#define MC_CNTCON_CLR_TC0MCI1_RE_CLR      (0x00000004UL)
#define MC_CNTCON_CLR_TC0MCI1_FE_CLR      (0x00000008UL)
#define MC_CNTCON_CLR_TC0MCI2_RE_CLR      (0x00000010UL)
#define MC_CNTCON_CLR_TC0MCI2_FE_CLR      (0x00000020UL)
#define MC_CNTCON_CLR_TC1MCI0_RE_CLR      (0x00000040UL)
#define MC_CNTCON_CLR_TC1MCI0_FE_CLR      (0x00000080UL)
#define MC_CNTCON_CLR_TC1MCI1_RE_CLR      (0x00000100UL)
#define MC_CNTCON_CLR_TC1MCI1_FE_CLR      (0x00000200UL)
#define MC_CNTCON_CLR_TC1MCI2_RE_CLR      (0x00000400UL)
#define MC_CNTCON_CLR_TC1MCI2_FE_CLR      (0x00000800UL)
#define MC_CNTCON_CLR_TC2MCI0_RE_CLR      (0x00001000UL)
#define MC_CNTCON_CLR_TC2MCI0_FE_CLR      (0x00002000UL)
#define MC_CNTCON_CLR_TC2MCI1_RE_CLR      (0x00004000UL)
#define MC_CNTCON_CLR_TC2MCI1_FE_CLR      (0x00008000UL)
#define MC_CNTCON_CLR_TC2MCI2_RE_CLR      (0x00010000UL)
#define MC_CNTCON_CLR_TC2MCI2_FE_CLR      (0x00020000UL)
#define MC_CNTCON_CLR_CNTR0_CLR           (0x20000000UL)
#define MC_CNTCON_CLR_CNTR1_CLR           (0x40000000UL)
#define MC_CNTCON_CLR_CNTR2_CLR           (0x80000000UL)

/* Dead-time register */
typedef struct{
__REG32 DT0         :10;
__REG32 DT1         :10;
__REG32 DT2         :10;
__REG32             : 2;
} __mc_dt_bits;

/* Current communication pattern register */
typedef struct{
__REG32 CCPA0       : 1;
__REG32 CCPB0       : 1;
__REG32 CCPA1       : 1;
__REG32 CCPB1       : 1;
__REG32 CCPA2       : 1;
__REG32 CCPB2       : 1;
__REG32             :26;
} __mc_ccp_bits;

/* Capture clear register */

#define MC_CAP_CLR_CAP_CLR0     (0x000000001UL)
#define MC_CAP_CLR_CAP_CLR1     (0x000000002UL)
#define MC_CAP_CLR_CAP_CLR2     (0x000000004UL)

/* QEI Control register */

#define QEI_CON_RESP            (0x000000001UL)
#define QEI_CON_RESPI           (0x000000002UL)
#define QEI_CON_RESV            (0x000000004UL)
#define QEI_CON_RESI            (0x000000008UL)

/* QEI Configuration register */
typedef struct{
__REG32 DIRINV      : 1;
__REG32 SIGMODE     : 1;
__REG32 CAPMODE     : 1;
__REG32 INVINX      : 1;
__REG32 CRESPI      : 1;
__REG32             :11;
__REG32 INXGATE     : 4;
__REG32             :12;
} __qei_conf_bits;

/* QEI Status register */
typedef struct{
__REG32 DIR         : 1;
__REG32             :31;
} __qei_stat_bits;


/* QEI Interrupt Clear register */

#define QEI_IEC_INX_EN      (0x00000001UL)
#define QEI_IEC_TIM_EN      (0x00000002UL)
#define QEI_IEC_VELC_EN     (0x00000004UL)
#define QEI_IEC_DIR_EN      (0x00000008UL)
#define QEI_IEC_ERR_EN      (0x00000010UL)
#define QEI_IEC_ENCLK_EN    (0x00000020UL)
#define QEI_IEC_POS0_INT    (0x00000040UL)
#define QEI_IEC_POS1_INT    (0x00000080UL)
#define QEI_IEC_POS2_INT    (0x00000100UL)
#define QEI_IEC_REV0_INT    (0x00000200UL)
#define QEI_IEC_POS0REV_INT (0x00000400UL)
#define QEI_IEC_POS1REV_INT (0x00000800UL)
#define QEI_IEC_POS2REV_INT (0x00001000UL)
#define QEI_IEC_REV1_INT    (0x00002000UL)
#define QEI_IEC_REV2_INT    (0x00004000UL)
#define QEI_IEC_MAXPOS_INT  (0x00008000UL)

/* QEI Interrupt Set register */

#define QEI_IES_INX_EN      (0x00000001UL)
#define QEI_IES_TIM_EN      (0x00000002UL)
#define QEI_IES_VELC_EN     (0x00000004UL)
#define QEI_IES_DIR_EN      (0x00000008UL)
#define QEI_IES_ERR_EN      (0x00000010UL)
#define QEI_IES_ENCLK_EN    (0x00000020UL)
#define QEI_IES_POS0_Int    (0x00000040UL)
#define QEI_IES_POS1_Int    (0x00000080UL)
#define QEI_IES_POS2_Int    (0x00000100UL)
#define QEI_IES_REV0_Int    (0x00000200UL)
#define QEI_IES_POS0REV_Int (0x00000400UL)
#define QEI_IES_POS1REV_Int (0x00000800UL)
#define QEI_IES_POS2REV_Int (0x00001000UL)
#define QEI_IES_REV1_Int    (0x00002000UL)
#define QEI_IES_REV2_Int    (0x00004000UL)
#define QEI_IES_MAXPOS_Int  (0x00008000UL)

/* QEI Interrupt Status register */
/* QEI Interrupt Enable register */
typedef struct{
__REG32 INX_Int     : 1;
__REG32 TIM_Int     : 1;
__REG32 VELC_Int    : 1;
__REG32 DIR_Int     : 1;
__REG32 ERR_Int     : 1;
__REG32 ENCLK_Int   : 1;
__REG32 POS0_Int    : 1;
__REG32 POS1_Int    : 1;
__REG32 POS2_Int    : 1;
__REG32 REV0_Int    : 1;
__REG32 POS0REV_Int : 1;
__REG32 POS1REV_Int : 1;
__REG32 POS2REV_Int : 1;
__REG32 REV1_Int    : 1;
__REG32 REV2_Int    : 1;
__REG32 MAXPOS_Int  : 1;
__REG32             :16;
} __qei_intstat_bits;

/* QEI Interrupt Clear register */

#define QEI_CLR_INX_Int      (0x00000001UL)
#define QEI_CLR_TIM_Int      (0x00000002UL)
#define QEI_CLR_VELC_Int     (0x00000004UL)
#define QEI_CLR_DIR_Int      (0x00000008UL)
#define QEI_CLR_ERR_Int      (0x00000010UL)
#define QEI_CLR_ENCLK_Int    (0x00000020UL)
#define QEI_CLR_POS0_Int     (0x00000040UL)
#define QEI_CLR_POS1_Int     (0x00000080UL)
#define QEI_CLR_POS2_Int     (0x00000100UL)
#define QEI_CLR_REV0_Int     (0x00000200UL)
#define QEI_CLR_POS0REV_Int  (0x00000400UL)
#define QEI_CLR_POS1REV_Int  (0x00000800UL)
#define QEI_CLR_POS2REV_Int  (0x00001000UL)
#define QEI_CLR_REV1_Int     (0x00002000UL)
#define QEI_CLR_REV2_Int     (0x00004000UL)
#define QEI_CLR_MAXPOS_Int   (0x00008000UL)

/* QEI Interrupt Set register */

#define QEI_SET_INX_Int      (0x00000001UL)
#define QEI_SET_TIM_Int      (0x00000002UL)
#define QEI_SET_VELC_Int     (0x00000004UL)
#define QEI_SET_DIR_Int      (0x00000008UL)
#define QEI_SET_ERR_Int      (0x00000010UL)
#define QEI_SET_ENCLK_Int    (0x00000020UL)
#define QEI_SET_POS0_Int     (0x00000040UL)
#define QEI_SET_POS1_Int     (0x00000080UL)
#define QEI_SET_POS2_Int     (0x00000100UL)
#define QEI_SET_REV0_Int     (0x00000200UL)
#define QEI_SET_POS0REV_Int  (0x00000400UL)
#define QEI_SET_POS1REV_Int  (0x00000800UL)
#define QEI_SET_POS2REV_Int  (0x00001000UL)
#define QEI_SET_REV1_Int     (0x00002000UL)
#define QEI_SET_REV2_Int     (0x00004000UL)
#define QEI_SET_MAXPOS_Int   (0x00008000UL)


/* RIT Control register */
typedef struct{
__REG32 RITINT   : 1;
__REG32 RITENCLR : 1;
__REG32 RITENBR  : 1;
__REG32 RITEN    : 1;
__REG32          :28;
} __rit_ctrl_bits;


/* Downcounter register */
typedef struct{
__REG32 CVAL     :16;
__REG32          :16;
} __at_downcounter_bits;

/* Preset value register */
typedef struct{
__REG32 PRESETVAL :16;
__REG32           :16;
} __at_preset_bits;

/* Interrupt clear enable register */

#define AT_CLR_EN_CLR_EN      (0x00000001UL)

/* Interrupt set enable register */

#define AT_SET_EN_SET_EN      (0x00000001UL)

/* Interrupt status register */
typedef struct{
__REG32 STAT     : 1;
__REG32          :31;
} __at_status_bits;

/* Interrupt enable register */
typedef struct{
__REG32 EN       : 1;
__REG32          :31;
} __at_enable_bits;

/* Clear status register */

#define AT_CLR_STAT_CSTAT     (0x00000001UL)

/* Set status register */

#define AT_SET_STAT_SSTAT     (0x00000001UL)

/* Watchdog mode register */
typedef struct{
__REG32 WDEN     : 1;
__REG32 WDRESET  : 1;
__REG32 WDTOF    : 1;
__REG32 WDINT    : 1;
__REG32 WDPROTECT: 1;
__REG32          :27;
} __wwdt_mod_bits;

/* Watchdog Timer Constant register */
typedef struct{
__REG32 WDTC     :24;
__REG32          : 8;
} __wwdt_tc_bits;

/* Watchdog feed register */

#define WWDT_FEED_Feed     (0x000000FFUL)

/* Watchdog Timer Value register */
typedef struct{
__REG32 Count    :24;
__REG32          : 8;
} __wwdt_tv_bits;

/* Watchdog Timer Warning Interrupt register */
typedef struct{
__REG32 WDWARNINT :10;
__REG32           :22;
} __wwdt_warnint_bits;

/* Watchdog Timer Window register */
typedef struct{
__REG32 WDWINDOW  :24;
__REG32           : 8;
} __wwdt_window_bits;


/* RTC interrupt location register */

#define RTC_ILR_RTCCIF     (0x00000001UL)
#define RTC_ILR_RTCALF     (0x00000002UL)

/* RTC clock control register */
typedef struct{
__REG32 CLKEN   : 1;
__REG32 CTCRST  : 1;
__REG32         : 2;
__REG32 CCALEN  : 1;
__REG32         :27;
} __rtc_ccr_bits;

/* RTC counter increment interrupt register */
typedef struct{
__REG32 IMSEC   : 1;
__REG32 IMMIN   : 1;
__REG32 IMHOUR  : 1;
__REG32 IMDOM   : 1;
__REG32 IMDOW   : 1;
__REG32 IMDOY   : 1;
__REG32 IMMON   : 1;
__REG32 IMYEAR  : 1;
__REG32         :24;
} __rtc_ciir_bits;

/* RTC Counter Increment Select Mask Register */
typedef struct{
__REG32 CALVAL    :17;
__REG32 CALDIR    : 1;
__REG32           :14;
} __rtc_calibration_bits;

/* RTC alarm mask register */
typedef struct{
__REG32 AMRSEC   : 1;
__REG32 AMRMIN   : 1;
__REG32 AMRHOUR  : 1;
__REG32 AMRDOM   : 1;
__REG32 AMRDOW   : 1;
__REG32 AMRDOY   : 1;
__REG32 AMRMON   : 1;
__REG32 AMRYEAR  : 1;
__REG32          :24;
} __rtc_amr_bits;

/* RTC consolidated time register 0 */
typedef struct{
__REG32 SECONDS : 6;
__REG32         : 2;
__REG32 MINUTES : 6;
__REG32         : 2;
__REG32 HOURS   : 5;
__REG32         : 3;
__REG32 DOW     : 3;
__REG32         : 5;
} __rtc_ctime0_bits;

/* RTC consolidated time register 1 */
typedef struct{
__REG32 DOM   : 5;
__REG32       : 3;
__REG32 MONTH : 4;
__REG32       : 4;
__REG32 YEAR  :12;
__REG32       : 4;
} __rtc_ctime1_bits;

/* RTC consolidated time register 2 */
typedef struct{
__REG32 DOY  :12;
__REG32      :20;
} __rtc_ctime2_bits;

/* RTC second register */
typedef struct{
__REG32 SECONDS : 6;
__REG32         :26;
} __rtc_sec_bits;

/* RTC minute register */
typedef struct{
__REG32 MINUTES : 6;
__REG32         :26;
} __rtc_min_bits;

/* RTC hour register */
typedef struct{
__REG32 HOURS : 5;
__REG32       :27;
} __rtc_hour_bits;

/* RTC day of month register */
typedef struct{
__REG32 DOM  : 5;
__REG32      :27;
} __rtc_dom_bits;

/* RTC day of week register */
typedef struct{
__REG32 DOW  : 3;
__REG32      :29;
} __rtc_dow_bits;

/* RTC day of year register */
typedef struct{
__REG32 DOY  : 9;
__REG32      :23;
} __rtc_doy_bits;

/* RTC month register */
typedef struct{
__REG32 MONTH : 4;
__REG32       :28;
} __rtc_month_bits;

/* RTC year register */
typedef struct{
__REG32 YEAR :12;
__REG32      :20;
} __rtc_year_bits;


/* UART Receive Buffer Register (RBR) */
/* UART Transmit Holding Register (THR) */
/* UART Divisor Latch Register  Low (DLL) */
typedef union {
  /*UxRBR*/
  struct {
    __REG32 RBR           : 8;
    __REG32               :24;
  };
  /*UxTHR*/
  struct {
    __REG32 THR           : 8;
    __REG32               :24;
  };
  /*UxDLL*/
  struct {
    __REG32 DLLSB         : 8;
    __REG32               :24;
  };
} __uart_rbr_bits;

/* UART interrupt enable register */
typedef union {
  /*UxIER*/
  struct{
    __REG32 RBRIE     : 1;
    __REG32 THREIE    : 1;
    __REG32 RXIE      : 1;
    __REG32           : 5;
    __REG32 ABEOINTEN : 1;
    __REG32 ABTOINTEN : 1;
    __REG32           :22;
  };
  /*UxDLM*/
  struct {
    __REG32 DLMSB         : 8;
    __REG32               :24;
  };
} __uart_ier_bits;

//* UART interrupt enable register */
typedef union {
  /*U1IER*/
  struct{
    __REG32 RBRIE     : 1;
    __REG32 THREIE    : 1;
    __REG32 RXIE      : 1;
    __REG32 MSIE      : 1;
    __REG32           : 3;
    __REG32 CTSIE     : 1;
    __REG32 ABEOIE    : 1;
    __REG32 ABTOIE    : 1;
    __REG32           :22;
  };
  /*U1DLM*/
  struct {
    __REG32 DLMSB         : 8;
    __REG32               :24;
  };
} __uart1_ier_bits;


/* UART Transmit Enable Register */
typedef struct{
  __REG32 TXEN            : 1;
  __REG32                 :31;
} __uart_ter_bits;

/* UART Scratch Pad Register */
typedef struct{
  __REG32 PAD             : 8;
  __REG32                 :24;
} __uart_scr_bits;

/* UART1 Scratch Pad Register */
typedef struct{
  __REG32 Pad             : 8;
  __REG32                 :24;
} __uart1_scr_bits;

/* UART1 Transmit Enable Register */
typedef struct{
  __REG32                 : 7;
  __REG32 TXEN            : 1;
  __REG32                 :24;
} __uart1_ter_bits;

/* UART Line Status Register (LSR) */
typedef struct {
__REG32 RDR             : 1;
__REG32 OE              : 1;
__REG32 PE              : 1;
__REG32 FE              : 1;
__REG32 BI              : 1;
__REG32 THRE            : 1;
__REG32 TEMT            : 1;
__REG32 RXFE            : 1;
__REG32 TXERR           : 1;
__REG32                 :23;
} __uart_lsr_bits;

/* UART1 Line Status Register (LSR) */
typedef struct {
__REG32 RDR             : 1;
__REG32 OE              : 1;
__REG32 PE              : 1;
__REG32 FE              : 1;
__REG32 BI              : 1;
__REG32 THRE            : 1;
__REG32 TEMT            : 1;
__REG32 RXFE            : 1;
__REG32                 :24;
} __uart1_lsr_bits;

/* UART Line Control Register (LCR) */
typedef struct {
__REG32 WLS   : 2;
__REG32 SBS   : 1;
__REG32 PE    : 1;
__REG32 PS    : 2;
__REG32 BC    : 1;
__REG32 DLAB  : 1;
__REG32       :24;
} __uart_lcr_bits;

/* UART interrupt identification register and fifo control register */
typedef union {
  /*UxIIR*/
  struct {
__REG32 INTSTATUS : 1;
__REG32 INTID     : 3;
__REG32           : 2;
__REG32 FIFOENABLE: 2;
__REG32 ABEOINT   : 1;
__REG32 ABTOINT   : 1;
__REG32           :22;
  };
  /*UxFCR*/
  struct {
__REG32 FIFOEN      : 1;
__REG32 RXFIFORES   : 1;
__REG32 TXFIFORES   : 1;
__REG32 DMAMODE     : 1;
__REG32             : 2;
__REG32 RXTRIGLVL   : 2;
__REG32             :24;
  };
} __uart_fcriir_bits;

/* UART modem control register */
typedef struct{
__REG32 DTRCTRL : 1;
__REG32 RTSCTRL : 1;
__REG32         : 2;
__REG32 LMS     : 1;
__REG32         : 1;
__REG32 RTSEN   : 1;
__REG32 CTSEN   : 1;
__REG32         :24;
} __uart1_mcr_bits;

/* UART modem status register */
typedef struct{
__REG32 DCTS  : 1;
__REG32 DDSR  : 1;
__REG32 TERI  : 1;
__REG32 DDCD  : 1;
__REG32 CTS   : 1;
__REG32 DSR   : 1;
__REG32 RI    : 1;
__REG32 DCD   : 1;
__REG32       :24;
} __uart1_msr_bits;

/* UART Auto-baud Control Register */
typedef struct{
__REG32 START        : 1;
__REG32 MODE         : 1;
__REG32 AUTORESTART  : 1;
__REG32              : 5;
__REG32 ABEOINTCLR   : 1;
__REG32 ABTOINTCLR   : 1;
__REG32              :22;
} __uart_acr_bits;

/* IrDA Control Register */
typedef struct{
__REG32 IRDAEN       : 1;
__REG32 IRDAINV      : 1;
__REG32 FIXPULSEEN   : 1;
__REG32 PULSEDIV     : 3;
__REG32              :26;
} __uart_icr_bits;

/* UART Fractional Divider Register */
typedef struct{
__REG32 DIVADDVAL  : 4;
__REG32 MULVAL     : 4;
__REG32            :24;
} __uart_fdr_bits;

/* UART Oversampling Register */
typedef struct{
__REG32            : 1;
__REG32 OSFRAC     : 3;
__REG32 OSINT      : 4;
__REG32 FDINT      : 7;
__REG32            :17;
} __uart_osr_bits;

/* UART RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32            : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __uart_rs485ctrl_bits;

/* UART1 RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32 SEL        : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __uart1_rs485ctrl_bits;

/* UART Half-duplex enable register */
typedef struct{
__REG32 HDEN       : 1;
__REG32            :31;
} __uart_hden_bits;

/* UART Smart card interface control register */
typedef struct{
__REG32 SCIEN      : 1;
__REG32 NACKDIS    : 1;
__REG32 PROTSEL    : 1;
__REG32            : 2;
__REG32 TXRETRY    : 3;
__REG32 GUARDTIME  : 8;
__REG32            :16;
} __uart_scictrl_bits;

/* UART Smart card interface control register */
typedef struct{
__REG32 SYNC              : 1;
__REG32 CSRC              : 1;
__REG32 FES               : 1;
__REG32 TSBYPASS          : 1;
__REG32 CSCEN             : 1;
__REG32 SSSDIS            : 1;
__REG32 CCCLR             : 1;
__REG32                   :25;
} __uart_syncctrl_bits;

/* UART FIFO Level register */
typedef struct{
__REG32 RXFIFILVL  : 4;
__REG32            : 4;
__REG32 TXFIFOLVL  : 4;
__REG32            :20;
} __uart1_fifolvl_bits;

/* UART RS485 Address Match register */
typedef struct{
__REG32 ADRMATCH   : 8;
__REG32            :24;
} __uart_rs485adrmatch_bits;

/* UART RS485 Delay value register */
typedef struct{
__REG32 DLY        : 8;
__REG32            :24;
} __uart_rs485dly_bits;


/* SSP Control Register 0 */
typedef struct{
__REG32 DSS  : 4;
__REG32 FRF  : 2;
__REG32 CPOL : 1;
__REG32 CPHA : 1;
__REG32 SCR  : 8;
__REG32      :16;
} __ssp_cr0_bits;

/* SSP Control Register 1 */
typedef struct{
__REG32 LBM  : 1;
__REG32 SSE  : 1;
__REG32 MS   : 1;
__REG32 SOD  : 1;
__REG32      :28;
} __ssp_cr1_bits;

/* SSP Data Register */
typedef struct{
__REG32 DATA :16;
__REG32      :16;
} __ssp_dr_bits;

/* SSP Status Register */
typedef struct{
__REG32 TFE  : 1;
__REG32 TNF  : 1;
__REG32 RNE  : 1;
__REG32 RFF  : 1;
__REG32 BSY  : 1;
__REG32      :27;
} __ssp_sr_bits;

/* SSP Clock Prescale Register */
typedef struct{
__REG32 CPSDVSR : 8;
__REG32         :24;
} __ssp_cpsr_bits;

/* SSP Interrupt Mask Set/Clear Register */
typedef struct{
__REG32 RORIM  : 1;
__REG32 RTIM   : 1;
__REG32 RXIM   : 1;
__REG32 TXIM   : 1;
__REG32        :28;
} __ssp_imsc_bits;

/* SSP Raw Interrupt Status Register */
typedef struct{
__REG32 RORRIS  : 1;
__REG32 RTRIS   : 1;
__REG32 RXRIS   : 1;
__REG32 TXRIS   : 1;
__REG32         :28;
} __ssp_ris_bits;

/* SSP Masked Interrupt Status Register */
typedef struct{
__REG32 RORMIS  : 1;
__REG32 RTMIS   : 1;
__REG32 RXMIS   : 1;
__REG32 TXMIS   : 1;
__REG32         :28;
} __ssp_mis_bits;

/* SSP Interrupt Clear Register */

#define SSP_IRC_RORIC   (0x00000001UL)
#define SSP_IRC_RTIC    (0x00000002UL)

/* SSP DMA Control Register */
typedef struct{
__REG32 RXDMAE : 1;
__REG32 TXDMAE : 1;
__REG32        :30;
} __ssp_dmacr_bits;


/* CAN control register */
typedef struct{
__REG32 INIT      : 1;
__REG32 IE        : 1;
__REG32 SIE       : 1;
__REG32 EIE       : 1;
__REG32           : 1;
__REG32 DAR       : 1;
__REG32 CCE       : 1;
__REG32 TEST      : 1;
__REG32           :24;
} __can_cntl_bits;

/* CAN status register */
typedef struct{
__REG32 LEC       : 3;
__REG32 TXOK      : 1;
__REG32 RXOK      : 1;
__REG32 EPASS     : 1;
__REG32 EWARN     : 1;
__REG32 BOFF      : 1;
__REG32           :24;
} __can_stat_bits;

/* CAN error counter */
typedef struct{
__REG32 TEC_7_0   : 8;
__REG32 REC_6_0   : 7;
__REG32 RP        : 1;
__REG32           :16;
} __can_ec_bits;

/* CAN bit timing register */
typedef struct{
__REG32 BRP       : 6;
__REG32 SJW       : 2;
__REG32 TSEG1     : 4;
__REG32 TSEG2     : 3;
__REG32           :17;
} __can_bt_bits;

/* CAN interrupt register */
typedef struct{
__REG32 INTID15_0 :16;
__REG32           :16;
} __can_int_bits;

/* CAN test register */
typedef struct{
__REG32           : 2;
__REG32 BASIC     : 1;
__REG32 SILENT    : 1;
__REG32 LBACK     : 1;
__REG32 TX1_0     : 2;
__REG32 RX        : 1;
__REG32           :24;
} __can_test_bits;

/* CAN baud rate prescaler extension register */
typedef struct{
__REG32 BRPE      : 4;
__REG32           :28;
} __can_brpe_bits;

/* CAN message interface command request registers */
typedef struct{
__REG32 Message_Number : 6;
__REG32                : 9;
__REG32 BUSY           : 1;
__REG32                :16;
} __can_ifx_cmdreq_bits;

/* CAN message interface command mask registers */
typedef struct{
__REG32 DATA_B         : 1;
__REG32 DATA_A         : 1;
__REG32 TXRQST_NEWDAT  : 1;
__REG32 CLRINTPND      : 1;
__REG32 CTRL           : 1;
__REG32 ARB            : 1;
__REG32 MASK           : 1;
__REG32 WR_RD          : 1;
__REG32                :24;
} __can_ifx_cmdmsk_bits;

/* CAN message interface command mask 1 registers */
typedef struct{
__REG32 MSK0           : 1;
__REG32 MSK1           : 1;
__REG32 MSK2           : 1;
__REG32 MSK3           : 1;
__REG32 MSK4           : 1;
__REG32 MSK5           : 1;
__REG32 MSK6           : 1;
__REG32 MSK7           : 1;
__REG32 MSK8           : 1;
__REG32 MSK9           : 1;
__REG32 MSK10          : 1;
__REG32 MSK11          : 1;
__REG32 MSK12          : 1;
__REG32 MSK13          : 1;
__REG32 MSK14          : 1;
__REG32 MSK15          : 1;
__REG32                :16;
} __can_ifx_msk1_bits;

/* CAN message interface command mask 1 registers */
typedef struct{
__REG32 MSK16          : 1;
__REG32 MSK17          : 1;
__REG32 MSK18          : 1;
__REG32 MSK19          : 1;
__REG32 MSK20          : 1;
__REG32 MSK21          : 1;
__REG32 MSK22          : 1;
__REG32 MSK23          : 1;
__REG32 MSK24          : 1;
__REG32 MSK25          : 1;
__REG32 MSK26          : 1;
__REG32 MSK27          : 1;
__REG32 MSK28          : 1;
__REG32                : 1;
__REG32 MDIR           : 1;
__REG32 MXTD           : 1;
__REG32                :16;
} __can_ifx_msk2_bits;

/* CAN message interface command arbitration 1 registers */
typedef struct{
__REG32  ID15_0         :16;
__REG32                 :16;
} __can_ifx_arb1_bits;

/* CAN message interface command arbitration 2 registers */
typedef struct{
__REG32  ID28_16        :13;
__REG32  DIR            : 1;
__REG32  XTD            : 1;
__REG32  MSGVAL         : 1;
__REG32                 :16;
} __can_ifx_arb2_bits;

/* CAN message interface message control registers */
typedef struct{
__REG32  DLC3_0         : 4;
__REG32                 : 3;
__REG32  EOB            : 1;
__REG32  TXRQST         : 1;
__REG32  RMTEN          : 1;
__REG32  RXIE           : 1;
__REG32  TXIE           : 1;
__REG32  UMASK          : 1;
__REG32  INTPND         : 1;
__REG32  MSGLST         : 1;
__REG32  NEWDAT         : 1;
__REG32                 :16;
} __can_ifx_mctrl_bits;

/* CAN message interface data A1 registers */
typedef struct{
__REG32  DATA0          : 8;
__REG32  DATA1          : 8;
__REG32                 :16;
} __can_ifx_da1_bits;

/* CAN message interface data A2 registers */
typedef struct{
__REG32  DATA2          : 8;
__REG32  DATA3          : 8;
__REG32                 :16;
} __can_ifx_da2_bits;

/* CAN message interface data B1 registers */
typedef struct{
__REG32  DATA4          : 8;
__REG32  DATA5          : 8;
__REG32                 :16;
} __can_ifx_db1_bits;

/* CAN message interface data B2 registers */
typedef struct{
__REG32  DATA6          : 8;
__REG32  DATA7          : 8;
__REG32                 :16;
} __can_ifx_db2_bits;

/* CAN transmission request 1 register */
typedef struct{
__REG32  TXRQST1        : 1;
__REG32  TXRQST2        : 1;
__REG32  TXRQST3        : 1;
__REG32  TXRQST4        : 1;
__REG32  TXRQST5        : 1;
__REG32  TXRQST6        : 1;
__REG32  TXRQST7        : 1;
__REG32  TXRQST8        : 1;
__REG32  TXRQST9        : 1;
__REG32  TXRQST10       : 1;
__REG32  TXRQST11       : 1;
__REG32  TXRQST12       : 1;
__REG32  TXRQST13       : 1;
__REG32  TXRQST14       : 1;
__REG32  TXRQST15       : 1;
__REG32  TXRQST16       : 1;
__REG32                 :16;
} __can_txreq1_bits;

/* CAN transmission request 2 register */
typedef struct{
__REG32  TXRQST17       : 1;
__REG32  TXRQST18       : 1;
__REG32  TXRQST19       : 1;
__REG32  TXRQST20       : 1;
__REG32  TXRQST21       : 1;
__REG32  TXRQST22       : 1;
__REG32  TXRQST23       : 1;
__REG32  TXRQST24       : 1;
__REG32  TXRQST25       : 1;
__REG32  TXRQST26       : 1;
__REG32  TXRQST27       : 1;
__REG32  TXRQST28       : 1;
__REG32  TXRQST29       : 1;
__REG32  TXRQST30       : 1;
__REG32  TXRQST31       : 1;
__REG32  TXRQST32       : 1;
__REG32                 :16;
} __can_txreq2_bits;

/* CAN new data 1 register */
typedef struct{
__REG32  NEWDAT1        : 1;
__REG32  NEWDAT2        : 1;
__REG32  NEWDAT3        : 1;
__REG32  NEWDAT4        : 1;
__REG32  NEWDAT5        : 1;
__REG32  NEWDAT6        : 1;
__REG32  NEWDAT7        : 1;
__REG32  NEWDAT8        : 1;
__REG32  NEWDAT9        : 1;
__REG32  NEWDAT10       : 1;
__REG32  NEWDAT11       : 1;
__REG32  NEWDAT12       : 1;
__REG32  NEWDAT13       : 1;
__REG32  NEWDAT14       : 1;
__REG32  NEWDAT15       : 1;
__REG32  NEWDAT16       : 1;
__REG32                 :16;
} __can_nd1_bits;

/* CAN new data 2 register */
typedef struct{
__REG32  NEWDAT17       : 1;
__REG32  NEWDAT18       : 1;
__REG32  NEWDAT19       : 1;
__REG32  NEWDAT20       : 1;
__REG32  NEWDAT21       : 1;
__REG32  NEWDAT22       : 1;
__REG32  NEWDAT23       : 1;
__REG32  NEWDAT24       : 1;
__REG32  NEWDAT25       : 1;
__REG32  NEWDAT26       : 1;
__REG32  NEWDAT27       : 1;
__REG32  NEWDAT28       : 1;
__REG32  NEWDAT29       : 1;
__REG32  NEWDAT30       : 1;
__REG32  NEWDAT31       : 1;
__REG32  NEWDAT32       : 1;
__REG32                 :16;
} __can_nd2_bits;

/* CAN interrupt pending 1 register */
typedef struct{
__REG32  INTPND1        : 1;
__REG32  INTPND2        : 1;
__REG32  INTPND3        : 1;
__REG32  INTPND4        : 1;
__REG32  INTPND5        : 1;
__REG32  INTPND6        : 1;
__REG32  INTPND7        : 1;
__REG32  INTPND8        : 1;
__REG32  INTPND9        : 1;
__REG32  INTPND10       : 1;
__REG32  INTPND11       : 1;
__REG32  INTPND12       : 1;
__REG32  INTPND13       : 1;
__REG32  INTPND14       : 1;
__REG32  INTPND15       : 1;
__REG32  INTPND16       : 1;
__REG32                 :16;
} __can_ir1_bits;

/* CAN interrupt pending 2 register */
typedef struct{
__REG32  INTPND17       : 1;
__REG32  INTPND18       : 1;
__REG32  INTPND19       : 1;
__REG32  INTPND20       : 1;
__REG32  INTPND21       : 1;
__REG32  INTPND22       : 1;
__REG32  INTPND23       : 1;
__REG32  INTPND24       : 1;
__REG32  INTPND25       : 1;
__REG32  INTPND26       : 1;
__REG32  INTPND27       : 1;
__REG32  INTPND28       : 1;
__REG32  INTPND29       : 1;
__REG32  INTPND30       : 1;
__REG32  INTPND31       : 1;
__REG32  INTPND32       : 1;
__REG32                 :16;
} __can_ir2_bits;

/* CAN message valid 1 register */
typedef struct{
__REG32  MSGVAL1        : 1;
__REG32  MSGVAL2        : 1;
__REG32  MSGVAL3        : 1;
__REG32  MSGVAL4        : 1;
__REG32  MSGVAL5        : 1;
__REG32  MSGVAL6        : 1;
__REG32  MSGVAL7        : 1;
__REG32  MSGVAL8        : 1;
__REG32  MSGVAL9        : 1;
__REG32  MSGVAL10       : 1;
__REG32  MSGVAL11       : 1;
__REG32  MSGVAL12       : 1;
__REG32  MSGVAL13       : 1;
__REG32  MSGVAL14       : 1;
__REG32  MSGVAL15       : 1;
__REG32  MSGVAL16       : 1;
__REG32                 :16;
} __can_msgv1_bits;

/* CAN message valid 2 register */
typedef struct{
__REG32  MSGVAL17       : 1;
__REG32  MSGVAL18       : 1;
__REG32  MSGVAL19       : 1;
__REG32  MSGVAL20       : 1;
__REG32  MSGVAL21       : 1;
__REG32  MSGVAL22       : 1;
__REG32  MSGVAL23       : 1;
__REG32  MSGVAL24       : 1;
__REG32  MSGVAL25       : 1;
__REG32  MSGVAL26       : 1;
__REG32  MSGVAL27       : 1;
__REG32  MSGVAL28       : 1;
__REG32  MSGVAL29       : 1;
__REG32  MSGVAL30       : 1;
__REG32  MSGVAL31       : 1;
__REG32  MSGVAL32       : 1;
__REG32                 :16;
} __can_msgv2_bits;

/* CAN clock divider register */
typedef struct{
__REG32  CLKDIVVAL      : 3;
__REG32                 :29;
} __can_clkdiv_bits;


/* I2S Digital Audio Output Registes */
typedef struct{
__REG32 WORDWIDTH     : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32 MUTE          : 1;
__REG32               :16;
} __i2s_dao_bits;

/* I2S Digital Audio Input Register */
typedef struct{
__REG32 WORDWIDTH     : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32               :17;
} __i2s_dai_bits;

/* I2S Status Feedback Register */
typedef struct{
__REG32 IRQ           : 1;
__REG32 DMAREQ1       : 1;
__REG32 DMAREQ2       : 1;
__REG32               : 5;
__REG32 RX_LEVEL      : 4;
__REG32               : 4;
__REG32 TX_LEVEL      : 4;
__REG32               :12;
} __i2s_state_bits;

/* I2S DMA1 Configuration Register */
typedef struct{
__REG32 RX_DMA1_ENABLE : 1;
__REG32 TX_DMA1_ENABLE : 1;
__REG32                : 6;
__REG32 RX_DEPTH_DMA1  : 4;
__REG32                : 4;
__REG32 TX_DEPTH_DMA1  : 4;
__REG32                :12;
} __i2s_dma1_bits;

/* I2S DMA2 Configuration Register */
typedef struct{
__REG32 RX_DMA2_ENABLE : 1;
__REG32 TX_DMA2_ENABLE : 1;
__REG32                : 6;
__REG32 RX_DEPTH_DMA2  : 4;
__REG32                : 4;
__REG32 TX_DEPTH_DMA2  : 4;
__REG32                :12;
} __i2s_dma2_bits;

/* I2S Interrupt Request Control register */
typedef struct{
__REG32 RX_IRQ_ENABLE : 1;
__REG32 TX_IRQ_ENABLE : 1;
__REG32               : 6;
__REG32 RX_DEPTH_IRQ  : 4;
__REG32               : 4;
__REG32 TX_DEPTH_IRQ  : 4;
__REG32               :12;
} __i2s_irq_bits;

/* I2S Receive & Transmit Clock Rate Registers */
typedef struct{
__REG32 Y_DIVIDER     : 8;
__REG32 X_DIVIDER     : 8;
__REG32               :16;
} __i2s_rxtxrate_bits;

/* Transmit Clock Rate register */
typedef struct{
__REG32 TX_BITRATE    : 6;
__REG32               :26;
} __i2s_txbitrate_bits;

/* Receive Clock Rate register */
typedef struct{
__REG32 RX_BITRATE    : 6;
__REG32               :26;
} __i2s_rxbitrate_bits;

/* Transmit Mode Control register */
typedef struct{
__REG32 TXCLKSEL      : 2;
__REG32 TX4PIN        : 1;
__REG32 TXMCENA       : 1;
__REG32               :28;
} __i2s_txmode_bits;

/* Receive Mode Control register */
typedef struct{
__REG32 RXCLKSEL      : 2;
__REG32 RX4PIN        : 1;
__REG32 RXMCENA       : 1;
__REG32               :28;
} __i2s_rxmode_bits;


/* I2C control set register */
typedef struct{
__REG32       : 2;
__REG32 AA    : 1;
__REG32 SI    : 1;
__REG32 STO   : 1;
__REG32 STA   : 1;
__REG32 I2EN  : 1;
__REG32       :25;
} __i2c_conset_bits;

/* I2C control clear register */

#define I2C_CONCLR_AAC      (0x00000004UL)
#define I2C_CONCLR_SIC      (0x00000008UL)
#define I2C_CONCLR_STAC     (0x00000020UL)
#define I2C_CONCLR_I2ENC    (0x00000040UL)

/* I2C status register */
typedef struct{
__REG32         : 3;
__REG32 Status  : 5;
__REG32         :24;
} __i2c_stat_bits;

/* I2C data register */
typedef struct{
__REG32 Data  : 8;
__REG32       :24;
} __i2c_dat_bits;

/* I2C Monitor mode control register */
typedef struct{
__REG32 MM_ENA    : 1;
__REG32 ENA_SCL   : 1;
__REG32 MATCH_ALL : 1;
__REG32           :29;
} __i2c_mmctrl_bits;

/* I2C slave  register */
typedef struct{
__REG32 GC      : 1;
__REG32 Address : 7;
__REG32         :24;
} __i2c_adr_bits;

/* I2C Mask registers */
typedef struct{
__REG32       : 1;
__REG32 MASK  : 7;
__REG32       :24;
} __i2c_mask_bits;

/* I2C SCL High Duty Cycle register */
typedef struct{
__REG32 SCLH   :16;
__REG32        :16;
} __i2c_sclh_bits;

/* I2C scl duty cycle register */
typedef struct{
__REG32 SCLL   :16;
__REG32        :16;
} __i2c_scll_bits;


/* A/D Control Register */
typedef struct{
__REG32 SEL     : 8;
__REG32 CLKDIV  : 8;
__REG32 BURST   : 1;
__REG32 CLKS    : 3;
__REG32         : 1;
__REG32 PDN     : 1;
__REG32         : 2;
__REG32 START   : 3;
__REG32 EDGE    : 1;
__REG32         : 4;
} __adc_cr_bits;

/* A/D Global Data Register */
typedef struct{
__REG32         : 6;
__REG32 V_VREF  :10;
__REG32         : 8;
__REG32 CHN     : 3;
__REG32         : 3;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __adc_gdr_bits;

/* A/D Status Register */
typedef struct{
__REG32 DONE0     : 1;
__REG32 DONE1     : 1;
__REG32 DONE2     : 1;
__REG32 DONE3     : 1;
__REG32 DONE4     : 1;
__REG32 DONE5     : 1;
__REG32 DONE6     : 1;
__REG32 DONE7     : 1;
__REG32 OVERUN0  : 1;
__REG32 OVERUN1  : 1;
__REG32 OVERUN2  : 1;
__REG32 OVERUN3  : 1;
__REG32 OVERUN4  : 1;
__REG32 OVERUN5  : 1;
__REG32 OVERUN6  : 1;
__REG32 OVERUN7  : 1;
__REG32 ADINT     : 1;
__REG32           :15;
} __adc_stat_bits;

/* A/D Intrrupt Enable Register */
typedef struct{
__REG32 ADINTEN0  : 1;
__REG32 ADINTEN1  : 1;
__REG32 ADINTEN2  : 1;
__REG32 ADINTEN3  : 1;
__REG32 ADINTEN4  : 1;
__REG32 ADINTEN5  : 1;
__REG32 ADINTEN6  : 1;
__REG32 ADINTEN7  : 1;
__REG32 ADGINTEN  : 1;
__REG32           :23;
} __adc_inten_bits;

/* A/D Data Register */
typedef struct{
__REG32         : 6;
__REG32 V_VREF  :10;
__REG32         :14;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __adc_dr_bits;


/* D/A Converter Register */
typedef struct{
__REG32        : 6;
__REG32 VALUE  :10;
__REG32 BIAS   : 1;
__REG32        :15;
} __dac_cr_bits;

/* D/A Converter Control register */
typedef struct{
__REG32 INT_DMA_REQ : 1;
__REG32 DBLBUF_ENA  : 1;
__REG32 CNT_ENA     : 1;
__REG32 DMA_ENA     : 1;
__REG32             :28;
} __dac_ctrl_bits;

/* D/A Converter Counter Value register */
typedef struct{
__REG32 VALUE       :16;
__REG32             :16;
} __dac_cntval_bits;


/* SPI control register */
typedef struct{
__REG32           : 2;
__REG32 BITENABLE : 1;
__REG32 CPHA      : 1;
__REG32 CPOL      : 1;
__REG32 MSTR      : 1;
__REG32 LSBF      : 1;
__REG32 SPIE      : 1;
__REG32 BITS      : 4;
__REG32           :20;
} __spi_cr_bits;

/* SPI status register */
typedef struct{
__REG32         : 3;
__REG32 ABRT    : 1;
__REG32 MODF    : 1;
__REG32 ROVR    : 1;
__REG32 WCOL    : 1;
__REG32 SPIF    : 1;
__REG32         :24;
} __spi_sr_bits;

/* SPI clock counter register */
typedef struct{
__REG32 COUNTER  : 8;
__REG32          :24;
} __spi_ccr_bits;

/* SPI Data Register */
typedef struct{
__REG32 DATALOW  : 8;
__REG32 DATAHIGH : 8;
__REG32          :16;
} __spi_dr_bits;

/* SPI interrupt register */
typedef struct{
__REG32 SPIIF   : 1;
__REG32         :31;
} __spi_int_bits;

/* SPI Test control register */
typedef struct{
__REG32          : 1;
__REG32  TEST    : 7;
__REG32          :24;
} __spi_tcr_bits;

/* SPI Test Status Register */
typedef struct{
__REG32          : 3;
__REG32  ABRT    : 1;
__REG32  MODF    : 1;
__REG32  ROVR    : 1;
__REG32  WCOL    : 1;
__REG32  SPIF    : 1;
__REG32          :24;
} __spi_tsr_bits;


/* OTP Bank 3  Word 0 */
typedef struct{
__REG32                    :23;
__REG32  USB_ID_ENABLE     : 1;
__REG32                    : 1;
__REG32  BOOT_SRC          : 4;
__REG32  AES_KEY2_LOCK     : 1;
__REG32  AES_KEY1_LOCK     : 1;
__REG32  JTAG_DISABLE      : 1;
} __otp_b3_w0_bits;

/* OTP Bank 3  Word 1 */
typedef struct{
__REG32  USB_VENDOR_ID     :16;
__REG32  USB_PRODUCT_ID    :16;
} __otp_b3_w1_bits;

/* OTP Bank 3  Word 2 */
typedef struct{
__REG32  ETH_MAC           :32;
} __otp_b3_w2_bits;

/* FMC Signature Start register */
typedef struct{
__REG32  START             :17;
__REG32                    :15;
} __fmc_fmsstart_bits;

/* FMC Signature Stop register */
typedef struct{
__REG32  STOP              :17;
__REG32  SIG_START         : 1;
__REG32                    :14;
} __fmc_fmsstop_bits;

/* FMC Signature Generation Result W0 register */
typedef struct{
__REG32  SW0               :32;
} __fmc_fmsw0_bits;

/* FMC Signature Generation Result W1 register */
typedef struct{
__REG32  SW1               :32;
} __fmc_fmsw1_bits;

/* FMC Signature Generation Result W2 register */
typedef struct{
__REG32  SW2               :32;
} __fmc_fmsw2_bits;

/* FMC Signature Generation Result W3 register */
typedef struct{
__REG32  SW3               :32;
} __fmc_fmsw3_bits;

/* FMC Status register */
typedef struct{
__REG32                    : 2;
__REG32  SIG_DONE          : 1;
__REG32                    :29;
} __fmc_fmstat_bits;

/* FMC Status Clear register */
#define FMC_FMSTAT_CLR     0x0000004UL


#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** SCS
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC_ICTR,             0xE000E004,__READ       ,__nvic_ictr_bits);
#define NVIC      NVIC_ICTR
#define NVIC_bit  NVIC_ICTR_bit
__IO_REG32_BIT(ACTLR,                 0xE000E008,__READ_WRITE ,__actlr_bits);
__IO_REG32_BIT(SYSTICK_CSR,           0xE000E010,__READ_WRITE ,__systick_csr_bits);
#define STCSR      SYSTICK_CSR
#define STCSR_bit  SYSTICK_CSR_bit
__IO_REG32_BIT(SYSTICK_RVR,           0xE000E014,__READ_WRITE ,__systick_rvr_bits);
#define STRVR      SYSTICK_RVR
#define STRVR_bit  SYSTICK_RVR_bit
__IO_REG32_BIT(SYSTICK_CVR,           0xE000E018,__READ_WRITE ,__systick_cvr_bits);
#define STCVR      SYSTICK_CVR
#define STCVR_bit  SYSTICK_CVR_bit
__IO_REG32_BIT(SYSTICK_CALVR,         0xE000E01C,__READ       ,__systick_calvr_bits);
#define STCR      SYSTICK_CALVR
#define STCR_bit  SYSTICK_CALVR_bit

__IO_REG32_BIT(NVIC_ISER0,            0xE000E100,__READ_WRITE ,__nvic_iser0_bits);
__IO_REG32_BIT(NVIC_ISER1,            0xE000E104,__READ_WRITE ,__nvic_iser1_bits);

__IO_REG32_BIT(NVIC_ICER0,            0xE000E180,__READ_WRITE ,__nvic_icer0_bits);
__IO_REG32_BIT(NVIC_ICER1,            0xE000E184,__READ_WRITE ,__nvic_icer1_bits);

__IO_REG32_BIT(NVIC_ISPR0,            0xE000E200,__READ_WRITE ,__nvic_ispr0_bits);
__IO_REG32_BIT(NVIC_ISPR1,            0xE000E204,__READ_WRITE ,__nvic_ispr1_bits);

__IO_REG32_BIT(NVIC_ICPR0,            0xE000E280,__READ_WRITE ,__nvic_icpr0_bits);
__IO_REG32_BIT(NVIC_ICPR1,            0xE000E284,__READ_WRITE ,__nvic_icpr1_bits);

__IO_REG32_BIT(NVIC_IABR0,            0xE000E300,__READ       ,__nvic_iabr0_bits);
__IO_REG32_BIT(NVIC_IABR1,            0xE000E304,__READ       ,__nvic_iabr1_bits);

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

__IO_REG32_BIT(CPUID,                 0xE000ED00,__READ       ,__cpuid_bits);
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,                  0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR1,                 0xE000ED18,__READ_WRITE ,__nvic_ipr1_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED1C,__READ_WRITE ,__nvic_ipr2_bits);
__IO_REG32_BIT(SHPR3,                 0xE000ED20,__READ_WRITE ,__nvic_ipr3_bits);
__IO_REG32_BIT(SHCSR,                 0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                  0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                  0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                  0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,                 0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                  0xE000ED38,__READ_WRITE);
__IO_REG32(    AFSR,                  0xE000ED3C,__READ_WRITE);
__IO_REG32(    ID_PFR0,               0xE000ED40,__READ       );
__IO_REG32(    ID_PFR1,               0xE000ED44,__READ       );
__IO_REG32(    ID_DFR0,               0xE000ED48,__READ       );
__IO_REG32(    ID_AFR0,               0xE000ED4C,__READ       );

__IO_REG32(    ID_MMFR0,              0xE000ED50,__READ       );
__IO_REG32(    ID_MMFR1,              0xE000ED54,__READ       );
__IO_REG32(    ID_MMFR2,              0xE000ED58,__READ       );
__IO_REG32(    ID_MMFR3,              0xE000ED5C,__READ       );

__IO_REG32(    ID_ISAR0,              0xE000ED60,__READ       );
__IO_REG32(    ID_ISAR1,              0xE000ED64,__READ       );
__IO_REG32(    ID_ISAR2,              0xE000ED68,__READ       );
__IO_REG32(    ID_ISAR3,              0xE000ED6C,__READ       );
__IO_REG32(    ID_ISAR4,              0xE000ED70,__READ       );

__IO_REG32_BIT(CPACR,                 0xE000ED88,__READ_WRITE ,__cpacr_bits);

__IO_REG32_BIT(DHCSR,                 0xE000EDF0,__READ_WRITE ,__dhcsr_bits);
#define DHCSR_READ       DHCSR
#define DHCSR_READ_bit   DHCSR_bit
#define DHCSR_WRITE      DHCSR
#define DHCSR_WRITE_bit  DHCSR_bit.__dhcsr_write_bits

__IO_REG32(    DCRSR,                 0xE000EDF4,__WRITE      );
__IO_REG32(    DCRDR,                 0xE000EDF8,__READ_WRITE );
__IO_REG32_BIT(DEMCR,                 0xE000EDFC,__READ_WRITE ,__demcr_bits);

__IO_REG32(    STIR,                  0xE000EF00,__WRITE      );

__IO_REG32_BIT(FPCCR,                 0xE000EF34,__READ_WRITE ,__fpccr_bits);
__IO_REG32_BIT(FPCAR,                 0xE000EF38,__READ_WRITE ,__fpcar_bits);
__IO_REG32_BIT(FPDSCR,                0xE000EF3C,__READ_WRITE ,__fpdscr_bits);
__IO_REG32_BIT(MVFR0,                 0xE000EF40,__READ_WRITE ,__mvfr0_bits);
__IO_REG32_BIT(MVFR1,                 0xE000EF44,__READ_WRITE ,__mvfr1_bits);

__IO_REG32(    PID4,                  0xE000EFD0,__READ       );
__IO_REG32(    PID5,                  0xE000EFD4,__READ       );
__IO_REG32(    PID6,                  0xE000EFD8,__READ       );
__IO_REG32(    PID7,                  0xE000EFDC,__READ       );
__IO_REG32(    PID0,                  0xE000EFE0,__READ       );
__IO_REG32(    PID1,                  0xE000EFE4,__READ       );
__IO_REG32(    PID2,                  0xE000EFE8,__READ       );
__IO_REG32(    PID3,                  0xE000EFEC,__READ       );

__IO_REG32(    CID0,                  0xE000EFF0,__READ       );
__IO_REG32(    CID1,                  0xE000EFF4,__READ       );
__IO_REG32(    CID2,                  0xE000EFF8,__READ       );
__IO_REG32(    CID3,                  0xE000EFFC,__READ       );

/***************************************************************************
 **
 ** MPU
 **
 ***************************************************************************/
__IO_REG32_BIT(MPU_TYPE,              0xE000ED90,__READ       ,__mpu_type_bits);
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
 ** OTP
 **
 ***************************************************************************/
__IO_REG32(    OTP_B0W0,              0x40045000,__READ_WRITE );
__IO_REG32(    OTP_B0W1,              0x40045004,__READ_WRITE );
__IO_REG32(    OTP_B0W2,              0x40045008,__READ_WRITE );
__IO_REG32(    OTP_B0W3,              0x4004500C,__READ_WRITE );
__IO_REG32(    OTP_B1W0,              0x40045010,__READ_WRITE );
__IO_REG32(    OTP_B1W1,              0x40045014,__READ_WRITE );
__IO_REG32(    OTP_B1W2,              0x40045018,__READ_WRITE );
__IO_REG32(    OTP_B1W3,              0x4004501C,__READ_WRITE );
__IO_REG32(    OTP_B2W0,              0x40045020,__READ_WRITE );
__IO_REG32(    OTP_B2W1,              0x40045024,__READ_WRITE );
__IO_REG32(    OTP_B2W2,              0x40045028,__READ_WRITE );
__IO_REG32(    OTP_B2W3,              0x4004502C,__READ_WRITE );
__IO_REG32_BIT(OTP_B3W0,              0x40045030,__READ_WRITE ,__otp_b3_w0_bits);
__IO_REG32_BIT(OTP_B3W1,              0x40045034,__READ_WRITE ,__otp_b3_w1_bits);
__IO_REG32_BIT(OTP_B3W2,              0x40045038,__READ_WRITE ,__otp_b3_w2_bits);
__IO_REG32(    OTP_B3W3,              0x4004503C,__READ_WRITE );

/***************************************************************************
 **
 ** ER (Event router)
 **
 ***************************************************************************/
__IO_REG32_BIT(ER_HILO,               0x40044000,__READ_WRITE ,__er_hilo_bits);
__IO_REG32_BIT(ER_EDGE,               0x40044004,__READ_WRITE ,__er_edge_bits);
__IO_REG32(    ER_CLR_EN,             0x40044FD8,__WRITE      );
__IO_REG32(    ER_SET_EN,             0x40044FDC,__WRITE      );
__IO_REG32_BIT(ER_STATUS,             0x40044FE0,__READ       ,__er_status_bits);
__IO_REG32_BIT(ER_ENABLE,             0x40044FE4,__READ       ,__er_enable_bits);
__IO_REG32(    ER_CLR_STAT,           0x40044FE8,__WRITE      );
__IO_REG32(    ER_SET_STAT,           0x40044FEC,__WRITE      );

/***************************************************************************
 **
 ** CREG (Configuration registers)
 **
 ***************************************************************************/
__IO_REG32_BIT(CREG_CREG0,            0x40043004,__READ_WRITE ,__creg_creg0_bits);
/*__IO_REG32_BIT(CREG_PMUCON,           0x40043008,__READ_WRITE ,__creg_pmucon_bits);*/
__IO_REG32_BIT(CREG_M4MEMMAP,         0x40043100,__READ_WRITE ,__creg_m4memmap_bits);
__IO_REG32(    CREG_CREG1,            0x40043108,__READ       );
__IO_REG32(    CREG_CREG2,            0x4004310C,__READ       );
__IO_REG32(    CREG_CREG3,            0x40043110,__READ       );
__IO_REG32(    CREG_CREG4,            0x40043114,__READ       );
__IO_REG32_BIT(CREG_CREG5,            0x40043118,__READ_WRITE ,__creg_creg5_bits);
__IO_REG32_BIT(CREG_DMAMUX,           0x4004311C,__READ_WRITE ,__creg_dmamux_bits);
__IO_REG32_BIT(CREG_ETBCFG,           0x40043128,__READ_WRITE ,__creg_etbcfg_bits);
__IO_REG32_BIT(CREG_CREG6,            0x4004312C,__READ_WRITE ,__creg_creg6_bits);
__IO_REG32_BIT(CREG_M4TXEVENT,        0x40043130,__READ_WRITE ,__creg_m4txevent_bits);
__IO_REG32(    CREG_CHIPID,           0x40043200,__READ       );
__IO_REG32_BIT(CREG_M0TXEVENT,        0x40043400,__READ_WRITE ,__creg_m0txevent_bits);
__IO_REG32_BIT(CREG_M0APPMEMMAP,      0x40043404,__READ_WRITE ,__creg_m0appmemmap_bits);

/***************************************************************************
 **
 ** PMC (Power Management Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PMC_PD0_SLEEP0_HW_ENA,     0x40042000,__READ_WRITE ,__pmc_pd0_sleep0_hw_ena_bits);
__IO_REG32_BIT(PMC_PD0_SLEEP0_MODE,       0x4004201C,__READ_WRITE ,__pmc_pd0_sleep0_mode_bits);

/***************************************************************************
 **
 ** CGU
 **
 ***************************************************************************/
__IO_REG32_BIT(CGU_FREQ_MON,          0x40050014,__READ_WRITE ,__cgu_freq_mon_bits);
__IO_REG32_BIT(CGU_XTAL_OSC_CTRL,     0x40050018,__READ_WRITE ,__cgu_xtal_osc_ctrl_bits);
__IO_REG32_BIT(CGU_PLL0USB_STAT,      0x4005001C,__READ       ,__cgu_pll0usb_stat_bits);
__IO_REG32_BIT(CGU_PLL0USB_CTRL,      0x40050020,__READ_WRITE ,__cgu_pll0usb_ctrl_bits);
__IO_REG32_BIT(CGU_PLL0USB_MDIV,      0x40050024,__READ_WRITE ,__cgu_pll0usb_mdiv_bits);
__IO_REG32_BIT(CGU_PLL0USB_NP_DIV,    0x40050028,__READ_WRITE ,__cgu_pll0usb_np_div_bits);
__IO_REG32_BIT(CGU_PLL0AUDIO_STAT,    0x4005002C,__READ       ,__cgu_pll0audio_stat_bits);
__IO_REG32_BIT(CGU_PLL0AUDIO_CTRL,    0x40050030,__READ_WRITE ,__cgu_pll0audio_ctrl_bits);
__IO_REG32_BIT(CGU_PLL0AUDIO_MDIV,    0x40050034,__READ_WRITE ,__cgu_pll0audio_mdiv_bits);
__IO_REG32_BIT(CGU_PLL0AUDIO_NP_DIV,  0x40050038,__READ_WRITE ,__cgu_pll0audio_np_div_bits);
__IO_REG32_BIT(CGU_PLL0AUDIO_FRAC,    0x4005003C,__READ_WRITE ,__cgu_pll0audio_frac_bits);
__IO_REG32_BIT(CGU_PLL1_STAT,         0x40050040,__READ       ,__cgu_pll1_stat_bits);
__IO_REG32_BIT(CGU_PLL1_CTRL,         0x40050044,__READ_WRITE ,__cgu_pll1_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVA_CTRL,        0x40050048,__READ_WRITE ,__cgu_idiva_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVB_CTRL,        0x4005004C,__READ_WRITE ,__cgu_idivx_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVC_CTRL,        0x40050050,__READ_WRITE ,__cgu_idivx_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVD_CTRL,        0x40050054,__READ_WRITE ,__cgu_idivx_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVE_CTRL,        0x40050058,__READ_WRITE ,__cgu_idive_ctrl_bits);
__IO_REG32_BIT(CGU_BASE_SAFE_CLK,     0x4005005C,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_USB0_CLK,     0x40050060,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_PERIPH_CLK,   0x40050064,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_USB1_CLK,     0x40050068,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_M4_CLK,       0x4005006C,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_SPIFI_CLK,    0x40050070,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_SPI_CLK,      0x40050074,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_PHY_RX_CLK,   0x40050078,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_PHY_TX_CLK,   0x4005007C,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_APB1_CLK,     0x40050080,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_APB3_CLK,     0x40050084,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_LCD_CLK,      0x40050088,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_VADC_CLK,     0x4005008C,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_SDIO_CLK,     0x40050090,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_SSP0_CLK,     0x40050094,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_SSP1_CLK,     0x40050098,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_UART0_CLK,    0x4005009C,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_UART1_CLK,    0x400500A0,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_UART2_CLK,    0x400500A4,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_UART3_CLK,    0x400500A8,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_OUT_CLK,      0x400500AC,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32(    CGU_OUTCLK_21_CTRL,    0x400500B0,__READ_WRITE );
__IO_REG32(    CGU_OUTCLK_22_CTRL,    0x400500B4,__READ_WRITE );
__IO_REG32(    CGU_OUTCLK_23_CTRL,    0x400500B8,__READ_WRITE );
__IO_REG32(    CGU_OUTCLK_24_CTRL,    0x400500BC,__READ_WRITE );
__IO_REG32_BIT(CGU_BASE_APLL_CLK,     0x400500C0,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_CGU_OUT0_CLK, 0x400500C4,__READ_WRITE ,__cgu_base_xxx_clk_bits);
__IO_REG32_BIT(CGU_BASE_CGU_OUT1_CLK, 0x400500C8,__READ_WRITE ,__cgu_base_xxx_clk_bits);

/***************************************************************************
 **
 ** CCU1
 **
 ***************************************************************************/
__IO_REG32_BIT(CCU1_PM,                        0x40051000,__READ_WRITE ,__ccu_pm_bits);
__IO_REG32_BIT(CCU1_BASE_STAT,                 0x40051004,__READ       ,__ccu1_base_stat_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_BUS_CFG,          0x40051100,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_BUS_STAT,         0x40051104,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_I2C1_CFG,         0x40051108,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_I2C1_STAT,        0x4005110C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_DAC_CFG,          0x40051110,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_DAC_STAT,         0x40051114,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC0_CFG,         0x40051118,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC0_STAT,        0x4005111C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC1_CFG,         0x40051120,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC1_STAT,        0x40051124,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_CAN0_CFG,         0x40051128,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_CAN0_STAT,        0x4005112C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_BUS_CFG,          0x40051200,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_BUS_STAT,         0x40051204,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_MOTOCON_PWM_CFG,  0x40051208,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_MOTOCON_PWM_STAT, 0x4005120C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2C0_CFG,         0x40051210,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2C0_STAT,        0x40051214,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2S_CFG,          0x40051218,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2S_STAT,         0x4005121C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_CAN1_CFG,         0x40051220,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_CAN1_STAT,        0x40051224,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPIFI_CFG,             0x40051300,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPIFI_STAT,            0x40051304,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_BUS_CFG,            0x40051400,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_BUS_STAT,           0x40051404,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SPIFI_CFG,          0x40051408,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SPIFI_STAT,         0x4005140C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_GPIO_CFG,           0x40051410,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_GPIO_STAT,          0x40051414,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_LCD_CFG,            0x40051418,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_LCD_STAT,           0x4005141C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_ETHERNET_CFG,       0x40051420,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_ETHERNET_STAT,      0x40051424,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USB0_CFG,           0x40051428,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USB0_STAT,          0x4005142C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_EMC_CFG,            0x40051430,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_EMC_STAT,           0x40051434,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SDIO_CFG,           0x40051438,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SDIO_STAT,          0x4005143C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_DMA_CFG,            0x40051440,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_DMA_STAT,           0x40051444,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_M4CORE_CFG,         0x40051448,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_M4CORE_STAT,        0x4005144C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SCT_CFG,            0x40051468,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SCT_STAT,           0x4005146C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USB1_CFG,           0x40051470,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USB1_STAT,          0x40051474,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_EMCDIV_CFG,         0x40051478,__READ_WRITE ,__ccu1_clk_emcdiv_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_EMCDIV_STAT,        0x4005147C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_M0APP_CFG,          0x40051490,__READ_WRITE ,__ccu1_clk_emcdiv_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_M0APP_STAT,         0x40051494,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_VADC_CFG,           0x40051498,__READ_WRITE ,__ccu1_clk_emcdiv_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_VADC_STAT,          0x4005149C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_WWDT_CFG,           0x40051500,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_WWDT_STAT,          0x40051504,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USART0_CFG,         0x40051508,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USART0_STAT,        0x4005150C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_UART1_CFG,          0x40051510,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_UART1_STAT,         0x40051514,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SSP0_CFG,           0x40051518,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SSP0_STAT,          0x4005151C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER0_CFG,         0x40051520,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER0_STAT,        0x40051524,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER1_CFG,         0x40051528,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER1_STAT,        0x4005152C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SCU_CFG,            0x40051530,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SCU_STAT,           0x40051534,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_CREG_CFG,           0x40051538,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_CREG_STAT,          0x4005153C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_RITIMER_CFG,        0x40051600,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_RITIMER_STAT,       0x40051604,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USART2_CFG,         0x40051608,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USART2_STAT,        0x4005160C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USART3_CFG,         0x40051610,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_USART3_STAT,        0x40051614,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER2_CFG,         0x40051618,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER2_STAT,        0x4005161C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER3_CFG,         0x40051620,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_TIMER3_STAT,        0x40051624,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SSP1_CFG,           0x40051628,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_SSP1_STAT,          0x4005162C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_QEI_CFG,            0x40051630,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M4_QEI_STAT,           0x40051634,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_PERIPH_BUS_CFG,        0x40051700,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_PERIPH_BUS_STAT,       0x40051704,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_PERIPH_CORE_CFG,       0x40051710,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_PERIPH_CORE_STAT,      0x40051714,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_PERIPH_SGPIO_CFG,      0x40051718,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_PERIPH_SGPIO_STAT,     0x4005171C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_USB0_CFG,              0x40051800,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_USB0_STAT,             0x40051804,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_USB1_CFG,              0x40051900,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_USB1_STAT,             0x40051904,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPI_CFG,               0x40051A00,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPI_STAT,              0x40051A04,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_VADC_CFG,              0x40051B00,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_VADC_STAT,             0x40051B04,__READ       ,__ccu_clk_cfg_bits);


/***************************************************************************
 **
 ** CCU2
 **
 ***************************************************************************/
__IO_REG32_BIT(CCU2_PM,                       0x40052000,__READ_WRITE ,__ccu_pm_bits);
__IO_REG32_BIT(CCU2_BASE_STAT,                0x40052004,__READ       ,__ccu2_base_stat_bits);
__IO_REG32_BIT(CCU2_CLK_APLL_CFG,             0x40052100,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APLL_STAT,            0x40052104,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_USART3_CFG,      0x40052200,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_USART3_STAT,     0x40052204,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_USART2_CFG,      0x40052300,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_USART2_STAT,     0x40052304,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART1_CFG,       0x40052400,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART1_STAT,      0x40052404,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_USART0_CFG,      0x40052500,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_USART0_STAT,     0x40052504,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP1_CFG,        0x40052600,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP1_STAT,       0x40052604,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP0_CFG,        0x40052700,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP0_STAT,       0x40052704,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_SDIO_CFG,             0x40052800,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_SDIO_STAT,            0x40052804,__READ       ,__ccu_clk_cfg_bits);

/***************************************************************************
 **
 ** RGU
 **
 ***************************************************************************/
__IO_REG32(    RGU_RESET_CTRL0,               0x40053100,__WRITE      );
__IO_REG32(    RGU_RESET_CTRL1,               0x40053104,__WRITE      );
__IO_REG32_BIT(RGU_RESET_STATUS0,             0x40053110,__READ_WRITE ,__rgu_reset_status0_bits);
__IO_REG32_BIT(RGU_RESET_STATUS1,             0x40053114,__READ_WRITE ,__rgu_reset_status1_bits);
__IO_REG32_BIT(RGU_RESET_STATUS2,             0x40053118,__READ_WRITE ,__rgu_reset_status2_bits);
__IO_REG32_BIT(RGU_RESET_STATUS3,             0x4005311C,__READ_WRITE ,__rgu_reset_status3_bits);
__IO_REG32_BIT(RGU_RESET_ACTIVE_STATUS0,      0x40053150,__READ       ,__rgu_reset_active_status0_bits);
__IO_REG32_BIT(RGU_RESET_ACTIVE_STATUS1,      0x40053154,__READ       ,__rgu_reset_active_status1_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT0,           0x40053400,__READ_WRITE ,__rgu_reset_ext_stat0_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT1,           0x40053404,__READ_WRITE ,__rgu_reset_ext_stat1_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT2,           0x40053408,__READ_WRITE ,__rgu_reset_ext_stat2_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT4,           0x40053410,__READ_WRITE ,__rgu_reset_ext_stat4_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT5,           0x40053414,__READ_WRITE ,__rgu_reset_ext_stat5_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT8,           0x40053420,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT9,           0x40053424,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT13,          0x40053434,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT16,          0x40053440,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT17,          0x40053444,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT18,          0x40053448,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT19,          0x4005344C,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT20,          0x40053450,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT21,          0x40053454,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT22,          0x40053458,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT28,          0x40053470,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT32,          0x40053480,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT33,          0x40053484,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT34,          0x40053488,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT35,          0x4005348C,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT36,          0x40053490,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT37,          0x40053494,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT38,          0x40053498,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT39,          0x4005349C,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT40,          0x400534A0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT41,          0x400534A4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT42,          0x400534A8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT44,          0x400534B0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT45,          0x400534B4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT46,          0x400534B8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT47,          0x400534BC,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT48,          0x400534C0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT49,          0x400534C4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT50,          0x400534C8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT51,          0x400534CC,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT52,          0x400534D0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT53,          0x400534D4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT54,          0x400534D8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT55,          0x400534DC,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT56,          0x400534E0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT57,          0x400534E4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT58,          0x400534E8,__READ_WRITE ,__rgu_peripheral_reset_bits);

/***************************************************************************
 **
 ** System control unit
 **
 ***************************************************************************/
__IO_REG32_BIT(SCU_SFSP0_0,                       0x40086000,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP0_1,                       0x40086004,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_0,                       0x40086080,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_1,                       0x40086084,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_2,                       0x40086088,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_3,                       0x4008608C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_4,                       0x40086090,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_5,                       0x40086094,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_6,                       0x40086098,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_7,                       0x4008609C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_8,                       0x400860A0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_9,                       0x400860A4,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_10,                      0x400860A8,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_11,                      0x400860AC,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_12,                      0x400860B0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_13,                      0x400860B4,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_14,                      0x400860B8,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_15,                      0x400860BC,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_16,                      0x400860C0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_17,                      0x400860C4,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP1_18,                      0x400860C8,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_19,                      0x400860CC,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP1_20,                      0x400860D0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_0,                       0x40086100,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_1,                       0x40086104,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_2,                       0x40086108,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_3,                       0x4008610C,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP2_4,                       0x40086110,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP2_5,                       0x40086114,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP2_6,                       0x40086118,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_7,                       0x4008611C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_8,                       0x40086120,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_9,                       0x40086124,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_10,                      0x40086128,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_11,                      0x4008612C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_12,                      0x40086130,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP2_13,                      0x40086134,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_0,                       0x40086180,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_1,                       0x40086184,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_2,                       0x40086188,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_3,                       0x4008618C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_4,                       0x40086190,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_5,                       0x40086194,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_6,                       0x40086198,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_7,                       0x4008619C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP3_8,                       0x400861A0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_0,                       0x40086200,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_1,                       0x40086204,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_2,                       0x40086208,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_3,                       0x4008620C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_4,                       0x40086210,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_5,                       0x40086214,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_6,                       0x40086218,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_7,                       0x4008621C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_8,                       0x40086220,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_9,                       0x40086224,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP4_10,                      0x40086228,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_0,                       0x40086280,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_1,                       0x40086284,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_2,                       0x40086288,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_3,                       0x4008628C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_4,                       0x40086290,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_5,                       0x40086294,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_6,                       0x40086298,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP5_7,                       0x4008629C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_0,                       0x40086300,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_1,                       0x40086304,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_2,                       0x40086308,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_3,                       0x4008630C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_4,                       0x40086310,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_5,                       0x40086314,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_6,                       0x40086318,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_7,                       0x4008631C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_8,                       0x40086320,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_9,                       0x40086324,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_10,                      0x40086328,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_11,                      0x4008632C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP6_12,                      0x40086330,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_0,                       0x40086380,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_1,                       0x40086384,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_2,                       0x40086388,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_3,                       0x4008638C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_4,                       0x40086390,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_5,                       0x40086394,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_6,                       0x40086398,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP7_7,                       0x4008639C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP8_0,                       0x40086400,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP8_1,                       0x40086404,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP8_2,                       0x40086408,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSP8_3,                       0x4008640C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP8_4,                       0x40086410,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP8_5,                       0x40086414,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP8_6,                       0x40086418,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP8_7,                       0x4008641C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP8_8,                       0x40086420,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_0,                       0x40086480,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_1,                       0x40086484,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_2,                       0x40086488,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_3,                       0x4008648C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_4,                       0x40086490,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_5,                       0x40086494,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSP9_6,                       0x40086498,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPA_0,                       0x40086500,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPA_1,                       0x40086504,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSPA_2,                       0x40086508,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSPA_3,                       0x4008650C,__READ_WRITE ,__scu_sfspx_hidrv_bits);
__IO_REG32_BIT(SCU_SFSPA_4,                       0x40086510,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_0,                       0x40086580,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_1,                       0x40086584,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_2,                       0x40086588,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_3,                       0x4008658C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_4,                       0x40086590,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_5,                       0x40086594,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPB_6,                       0x40086598,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_0,                       0x40086600,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_1,                       0x40086604,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_2,                       0x40086608,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_3,                       0x4008660C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_4,                       0x40086610,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_5,                       0x40086614,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_6,                       0x40086618,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_7,                       0x4008661C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_8,                       0x40086620,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_9,                       0x40086624,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_10,                      0x40086628,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_11,                      0x4008662C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_12,                      0x40086630,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_13,                      0x40086634,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPC_14,                      0x40086638,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_0,                       0x40086680,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_1,                       0x40086684,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_2,                       0x40086688,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_3,                       0x4008668C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_4,                       0x40086690,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_5,                       0x40086694,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_6,                       0x40086698,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_7,                       0x4008669C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_8,                       0x400866A0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_9,                       0x400866A4,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_10,                      0x400866A8,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_11,                      0x400866AC,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_12,                      0x400866B0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_13,                      0x400866B4,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_14,                      0x400866B8,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_15,                      0x400866BC,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPD_16,                      0x400866C0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_0,                       0x40086700,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_1,                       0x40086704,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_2,                       0x40086708,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_3,                       0x4008670C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_4,                       0x40086710,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_5,                       0x40086714,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_6,                       0x40086718,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_7,                       0x4008671C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_8,                       0x40086720,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_9,                       0x40086724,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_10,                      0x40086728,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_11,                      0x4008672C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_12,                      0x40086730,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_13,                      0x40086734,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_14,                      0x40086738,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPE_15,                      0x4008673C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_0,                       0x40086780,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_1,                       0x40086784,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_2,                       0x40086788,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_3,                       0x4008678C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_4,                       0x40086790,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_5,                       0x40086794,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_6,                       0x40086798,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_7,                       0x4008679C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_8,                       0x400867A0,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_9,                       0x400867A4,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_10,                      0x400867A8,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSPF_11,                      0x400867AC,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSCLK0,                       0x40086C00,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSCLK1,                       0x40086C04,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSCLK2,                       0x40086C08,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSCLK3,                       0x40086C0C,__READ_WRITE ,__scu_sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SCU_SFSUSB,                        0x40086C80,__READ_WRITE ,__scu_sfsusb_bits);
__IO_REG32_BIT(SCU_SFSI2C0,                       0x40086C84,__READ_WRITE ,__scu_sfsi2c0_bits);
__IO_REG32_BIT(SCU_ENAIO0,                        0x40086C88,__READ_WRITE ,__scu_enaio0_bits);
__IO_REG32_BIT(SCU_ENAIO1,                        0x40086C8C,__READ_WRITE ,__scu_enaio1_bits);
__IO_REG32_BIT(SCU_ENAIO2,                        0x40086C90,__READ_WRITE ,__scu_enaio2_bits);
__IO_REG32_BIT(SCU_EMCDELAYCLK,                   0x40086D00,__READ_WRITE ,__scu_emcdelayclk_bits);
__IO_REG32_BIT(SCU_PINTSEL0,                      0x40086E00,__READ_WRITE ,__scu_pintsel0_bits);
__IO_REG32_BIT(SCU_PINTSEL1,                      0x40086E04,__READ_WRITE ,__scu_pintsel1_bits);

/***************************************************************************
 **
 ** GIMA
 **
 ***************************************************************************/

__IO_REG32_BIT(GIMA_CAP0_0_IN,                    0x400C7000,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP0_1_IN,                    0x400C7004,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP0_2_IN,                    0x400C7008,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP0_3_IN,                    0x400C700C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP1_0_IN,                    0x400C7010,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP1_1_IN,                    0x400C7014,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP1_2_IN,                    0x400C7018,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP1_3_IN,                    0x400C701C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP2_0_IN,                    0x400C7020,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP2_1_IN,                    0x400C7024,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP2_2_IN,                    0x400C7028,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP2_3_IN,                    0x400C702C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP3_0_IN,                    0x400C7030,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP3_1_IN,                    0x400C7034,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP3_2_IN,                    0x400C7038,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CAP3_3_IN,                    0x400C703C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_0_IN,                    0x400C7040,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_1_IN,                    0x400C7044,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_2_IN,                    0x400C7048,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_3_IN,                    0x400C704C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_4_IN,                    0x400C7050,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_5_IN,                    0x400C7054,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_6_IN,                    0x400C7058,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_CTIN_7_IN,                    0x400C705C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_VADC_TRIGGER_IN,              0x400C7060,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_EVENTROUTER_13_IN,            0x400C7064,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_EVENTROUTER_14_IN,            0x400C7068,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_EVENTROUTER_16_IN,            0x400C706C,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_ADCSTART0_IN,                 0x400C7070,__READ_WRITE ,__gima_in_bits);
__IO_REG32_BIT(GIMA_ADCSTART1_IN,                 0x400C7074,__READ_WRITE ,__gima_in_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/

__IO_REG32_BIT(GPIO_ISEL,         0x40087000,__READ_WRITE,__gpio_isel_bits);
__IO_REG32_BIT(GPIO_IENR,         0x40087004,__READ_WRITE,__gpio_ienr_bits);
__IO_REG32(    GPIO_SIENR,        0x40087008,__WRITE     );
__IO_REG32(    GPIO_CIENR,        0x4008700C,__WRITE     );
__IO_REG32_BIT(GPIO_IENF,         0x40087010,__READ_WRITE,__gpio_ienf_bits);
__IO_REG32(    GPIO_SIENF,        0x40087014,__WRITE     );
__IO_REG32(    GPIO_CIENF,        0x40087018,__WRITE     );
__IO_REG32_BIT(GPIO_RISE,         0x4008701C,__READ_WRITE,__gpio_rise_bits);
__IO_REG32_BIT(GPIO_FALL,         0x40087020,__READ_WRITE,__gpio_fall_bits);
__IO_REG32_BIT(GPIO_IST,          0x40087024,__READ_WRITE,__gpio_ist_bits);

__IO_REG32_BIT(GPIOG0_CTRL,       0x40088000,__READ_WRITE,__gpiogx_ctrl_bits);
__IO_REG32(    GPIOG0_PORT_POL0,  0x40088020,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL1,  0x40088024,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL2,  0x40088028,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL3,  0x4008802C,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL4,  0x40088030,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL5,  0x40088034,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL6,  0x40088038,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_POL7,  0x4008803C,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA0,  0x40088040,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA1,  0x40088044,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA2,  0x40088048,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA3,  0x4008804C,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA4,  0x40088050,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA5,  0x40088054,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA6,  0x40088058,__READ_WRITE);
__IO_REG32(    GPIOG0_PORT_ENA7,  0x4008805C,__READ_WRITE);

__IO_REG32_BIT(GPIOG1_CTRL,       0x40089000,__READ_WRITE,__gpiogx_ctrl_bits);
__IO_REG32(    GPIOG1_PORT_POL0,  0x40089020,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL1,  0x40089024,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL2,  0x40089028,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL3,  0x4008902C,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL4,  0x40089030,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL5,  0x40089034,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL6,  0x40089038,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_POL7,  0x4008903C,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA0,  0x40089040,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA1,  0x40089044,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA2,  0x40089048,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA3,  0x4008904C,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA4,  0x40089050,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA5,  0x40089054,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA6,  0x40089058,__READ_WRITE);
__IO_REG32(    GPIOG1_PORT_ENA7,  0x4008905C,__READ_WRITE);

__IO_REG8(     GPIO_B0,           0x400F4000,__READ_WRITE);
__IO_REG8(     GPIO_B1,           0x400F4001,__READ_WRITE);
__IO_REG8(     GPIO_B2,           0x400F4002,__READ_WRITE);
__IO_REG8(     GPIO_B3,           0x400F4003,__READ_WRITE);
__IO_REG8(     GPIO_B4,           0x400F4004,__READ_WRITE);
__IO_REG8(     GPIO_B5,           0x400F4005,__READ_WRITE);
__IO_REG8(     GPIO_B6,           0x400F4006,__READ_WRITE);
__IO_REG8(     GPIO_B7,           0x400F4007,__READ_WRITE);
__IO_REG8(     GPIO_B8,           0x400F4008,__READ_WRITE);
__IO_REG8(     GPIO_B9,           0x400F4009,__READ_WRITE);
__IO_REG8(     GPIO_B10,          0x400F400A,__READ_WRITE);
__IO_REG8(     GPIO_B11,          0x400F400B,__READ_WRITE);
__IO_REG8(     GPIO_B12,          0x400F400C,__READ_WRITE);
__IO_REG8(     GPIO_B13,          0x400F400D,__READ_WRITE);
__IO_REG8(     GPIO_B14,          0x400F400E,__READ_WRITE);
__IO_REG8(     GPIO_B15,          0x400F400F,__READ_WRITE);

__IO_REG8(     GPIO_B32,          0x400F4020,__READ_WRITE);
__IO_REG8(     GPIO_B33,          0x400F4021,__READ_WRITE);
__IO_REG8(     GPIO_B34,          0x400F4022,__READ_WRITE);
__IO_REG8(     GPIO_B35,          0x400F4023,__READ_WRITE);
__IO_REG8(     GPIO_B36,          0x400F4024,__READ_WRITE);
__IO_REG8(     GPIO_B37,          0x400F4025,__READ_WRITE);
__IO_REG8(     GPIO_B38,          0x400F4026,__READ_WRITE);
__IO_REG8(     GPIO_B39,          0x400F4027,__READ_WRITE);
__IO_REG8(     GPIO_B40,          0x400F4028,__READ_WRITE);
__IO_REG8(     GPIO_B41,          0x400F4029,__READ_WRITE);
__IO_REG8(     GPIO_B42,          0x400F402A,__READ_WRITE);
__IO_REG8(     GPIO_B43,          0x400F402B,__READ_WRITE);
__IO_REG8(     GPIO_B44,          0x400F402C,__READ_WRITE);
__IO_REG8(     GPIO_B45,          0x400F402D,__READ_WRITE);
__IO_REG8(     GPIO_B46,          0x400F402E,__READ_WRITE);
__IO_REG8(     GPIO_B47,          0x400F402F,__READ_WRITE);

__IO_REG8(     GPIO_B64,          0x400F4040,__READ_WRITE);
__IO_REG8(     GPIO_B65,          0x400F4041,__READ_WRITE);
__IO_REG8(     GPIO_B66,          0x400F4042,__READ_WRITE);
__IO_REG8(     GPIO_B67,          0x400F4043,__READ_WRITE);
__IO_REG8(     GPIO_B68,          0x400F4044,__READ_WRITE);
__IO_REG8(     GPIO_B69,          0x400F4045,__READ_WRITE);
__IO_REG8(     GPIO_B70,          0x400F4046,__READ_WRITE);
__IO_REG8(     GPIO_B71,          0x400F4047,__READ_WRITE);
__IO_REG8(     GPIO_B72,          0x400F4048,__READ_WRITE);
__IO_REG8(     GPIO_B73,          0x400F4049,__READ_WRITE);
__IO_REG8(     GPIO_B74,          0x400F404A,__READ_WRITE);
__IO_REG8(     GPIO_B75,          0x400F404B,__READ_WRITE);
__IO_REG8(     GPIO_B76,          0x400F404C,__READ_WRITE);
__IO_REG8(     GPIO_B77,          0x400F404D,__READ_WRITE);
__IO_REG8(     GPIO_B78,          0x400F404E,__READ_WRITE);
__IO_REG8(     GPIO_B79,          0x400F404F,__READ_WRITE);

__IO_REG8(     GPIO_B96,          0x400F4060,__READ_WRITE);
__IO_REG8(     GPIO_B97,          0x400F4061,__READ_WRITE);
__IO_REG8(     GPIO_B98,          0x400F4062,__READ_WRITE);
__IO_REG8(     GPIO_B99,          0x400F4063,__READ_WRITE);
__IO_REG8(     GPIO_B100,         0x400F4064,__READ_WRITE);
__IO_REG8(     GPIO_B101,         0x400F4065,__READ_WRITE);
__IO_REG8(     GPIO_B102,         0x400F4066,__READ_WRITE);
__IO_REG8(     GPIO_B103,         0x400F4067,__READ_WRITE);
__IO_REG8(     GPIO_B104,         0x400F4068,__READ_WRITE);
__IO_REG8(     GPIO_B105,         0x400F4069,__READ_WRITE);
__IO_REG8(     GPIO_B106,         0x400F406A,__READ_WRITE);
__IO_REG8(     GPIO_B107,         0x400F406B,__READ_WRITE);
__IO_REG8(     GPIO_B108,         0x400F406C,__READ_WRITE);
__IO_REG8(     GPIO_B109,         0x400F406D,__READ_WRITE);
__IO_REG8(     GPIO_B110,         0x400F406E,__READ_WRITE);
__IO_REG8(     GPIO_B111,         0x400F406F,__READ_WRITE);

__IO_REG8(     GPIO_B128,         0x400F4080,__READ_WRITE);
__IO_REG8(     GPIO_B129,         0x400F4081,__READ_WRITE);
__IO_REG8(     GPIO_B130,         0x400F4082,__READ_WRITE);
__IO_REG8(     GPIO_B131,         0x400F4083,__READ_WRITE);
__IO_REG8(     GPIO_B132,         0x400F4084,__READ_WRITE);
__IO_REG8(     GPIO_B133,         0x400F4085,__READ_WRITE);
__IO_REG8(     GPIO_B134,         0x400F4086,__READ_WRITE);
__IO_REG8(     GPIO_B135,         0x400F4087,__READ_WRITE);
__IO_REG8(     GPIO_B136,         0x400F4088,__READ_WRITE);
__IO_REG8(     GPIO_B137,         0x400F4089,__READ_WRITE);
__IO_REG8(     GPIO_B138,         0x400F408A,__READ_WRITE);
__IO_REG8(     GPIO_B139,         0x400F408B,__READ_WRITE);
__IO_REG8(     GPIO_B140,         0x400F408C,__READ_WRITE);
__IO_REG8(     GPIO_B141,         0x400F408D,__READ_WRITE);
__IO_REG8(     GPIO_B142,         0x400F408E,__READ_WRITE);
__IO_REG8(     GPIO_B143,         0x400F408F,__READ_WRITE);

__IO_REG8(     GPIO_B160,         0x400F40A0,__READ_WRITE);
__IO_REG8(     GPIO_B161,         0x400F40A1,__READ_WRITE);
__IO_REG8(     GPIO_B162,         0x400F40A2,__READ_WRITE);
__IO_REG8(     GPIO_B163,         0x400F40A3,__READ_WRITE);
__IO_REG8(     GPIO_B164,         0x400F40A4,__READ_WRITE);
__IO_REG8(     GPIO_B165,         0x400F40A5,__READ_WRITE);
__IO_REG8(     GPIO_B166,         0x400F40A6,__READ_WRITE);
__IO_REG8(     GPIO_B167,         0x400F40A7,__READ_WRITE);
__IO_REG8(     GPIO_B168,         0x400F40A8,__READ_WRITE);
__IO_REG8(     GPIO_B169,         0x400F40A9,__READ_WRITE);
__IO_REG8(     GPIO_B170,         0x400F40AA,__READ_WRITE);
__IO_REG8(     GPIO_B171,         0x400F40AB,__READ_WRITE);
__IO_REG8(     GPIO_B172,         0x400F40AC,__READ_WRITE);
__IO_REG8(     GPIO_B173,         0x400F40AD,__READ_WRITE);
__IO_REG8(     GPIO_B174,         0x400F40AE,__READ_WRITE);
__IO_REG8(     GPIO_B175,         0x400F40AF,__READ_WRITE);
__IO_REG8(     GPIO_B176,         0x400F40B0,__READ_WRITE);
__IO_REG8(     GPIO_B177,         0x400F40B1,__READ_WRITE);
__IO_REG8(     GPIO_B178,         0x400F40B2,__READ_WRITE);
__IO_REG8(     GPIO_B179,         0x400F40B3,__READ_WRITE);
__IO_REG8(     GPIO_B180,         0x400F40B4,__READ_WRITE);
__IO_REG8(     GPIO_B181,         0x400F40B5,__READ_WRITE);
__IO_REG8(     GPIO_B182,         0x400F40B6,__READ_WRITE);
__IO_REG8(     GPIO_B183,         0x400F40B7,__READ_WRITE);
__IO_REG8(     GPIO_B184,         0x400F40B8,__READ_WRITE);
__IO_REG8(     GPIO_B185,         0x400F40B9,__READ_WRITE);
__IO_REG8(     GPIO_B186,         0x400F40BA,__READ_WRITE);

__IO_REG8(     GPIO_B192,         0x400F40C0,__READ_WRITE);
__IO_REG8(     GPIO_B193,         0x400F40C1,__READ_WRITE);
__IO_REG8(     GPIO_B194,         0x400F40C2,__READ_WRITE);
__IO_REG8(     GPIO_B195,         0x400F40C3,__READ_WRITE);
__IO_REG8(     GPIO_B196,         0x400F40C4,__READ_WRITE);
__IO_REG8(     GPIO_B197,         0x400F40C5,__READ_WRITE);
__IO_REG8(     GPIO_B198,         0x400F40C6,__READ_WRITE);
__IO_REG8(     GPIO_B199,         0x400F40C7,__READ_WRITE);
__IO_REG8(     GPIO_B200,         0x400F40C8,__READ_WRITE);
__IO_REG8(     GPIO_B201,         0x400F40C9,__READ_WRITE);
__IO_REG8(     GPIO_B202,         0x400F40CA,__READ_WRITE);
__IO_REG8(     GPIO_B203,         0x400F40CB,__READ_WRITE);
__IO_REG8(     GPIO_B204,         0x400F40CC,__READ_WRITE);
__IO_REG8(     GPIO_B205,         0x400F40CD,__READ_WRITE);
__IO_REG8(     GPIO_B206,         0x400F40CE,__READ_WRITE);
__IO_REG8(     GPIO_B207,         0x400F40CF,__READ_WRITE);
__IO_REG8(     GPIO_B208,         0x400F40D0,__READ_WRITE);
__IO_REG8(     GPIO_B209,         0x400F40D1,__READ_WRITE);
__IO_REG8(     GPIO_B210,         0x400F40D2,__READ_WRITE);
__IO_REG8(     GPIO_B211,         0x400F40D3,__READ_WRITE);
__IO_REG8(     GPIO_B212,         0x400F40D4,__READ_WRITE);
__IO_REG8(     GPIO_B213,         0x400F40D5,__READ_WRITE);
__IO_REG8(     GPIO_B214,         0x400F40D6,__READ_WRITE);
__IO_REG8(     GPIO_B215,         0x400F40D7,__READ_WRITE);
__IO_REG8(     GPIO_B216,         0x400F40D8,__READ_WRITE);
__IO_REG8(     GPIO_B217,         0x400F40D9,__READ_WRITE);
__IO_REG8(     GPIO_B218,         0x400F40DA,__READ_WRITE);
__IO_REG8(     GPIO_B219,         0x400F40DB,__READ_WRITE);
__IO_REG8(     GPIO_B220,         0x400F40Dc,__READ_WRITE);
__IO_REG8(     GPIO_B221,         0x400F40DD,__READ_WRITE);
__IO_REG8(     GPIO_B222,         0x400F40DE,__READ_WRITE);

__IO_REG8(     GPIO_B224,         0x400F40E0,__READ_WRITE);
__IO_REG8(     GPIO_B225,         0x400F40E1,__READ_WRITE);
__IO_REG8(     GPIO_B226,         0x400F40E2,__READ_WRITE);
__IO_REG8(     GPIO_B227,         0x400F40E3,__READ_WRITE);
__IO_REG8(     GPIO_B228,         0x400F40E4,__READ_WRITE);
__IO_REG8(     GPIO_B229,         0x400F40E5,__READ_WRITE);
__IO_REG8(     GPIO_B230,         0x400F40E6,__READ_WRITE);
__IO_REG8(     GPIO_B231,         0x400F40E7,__READ_WRITE);
__IO_REG8(     GPIO_B232,         0x400F40E8,__READ_WRITE);
__IO_REG8(     GPIO_B233,         0x400F40E9,__READ_WRITE);
__IO_REG8(     GPIO_B234,         0x400F40EA,__READ_WRITE);
__IO_REG8(     GPIO_B235,         0x400F40EB,__READ_WRITE);
__IO_REG8(     GPIO_B236,         0x400F40EC,__READ_WRITE);
__IO_REG8(     GPIO_B237,         0x400F40ED,__READ_WRITE);
__IO_REG8(     GPIO_B238,         0x400F40EE,__READ_WRITE);
__IO_REG8(     GPIO_B239,         0x400F40EF,__READ_WRITE);
__IO_REG8(     GPIO_B240,         0x400F40F0,__READ_WRITE);
__IO_REG8(     GPIO_B241,         0x400F40F1,__READ_WRITE);
__IO_REG8(     GPIO_B242,         0x400F40F2,__READ_WRITE);
__IO_REG8(     GPIO_B243,         0x400F40F3,__READ_WRITE);
__IO_REG8(     GPIO_B244,         0x400F40F4,__READ_WRITE);
__IO_REG8(     GPIO_B245,         0x400F40F5,__READ_WRITE);
__IO_REG8(     GPIO_B246,         0x400F40F6,__READ_WRITE);
__IO_REG8(     GPIO_B247,         0x400F40F7,__READ_WRITE);
__IO_REG8(     GPIO_B248,         0x400F40F8,__READ_WRITE);
__IO_REG8(     GPIO_B249,         0x400F40F9,__READ_WRITE);


__IO_REG32(    GPIO_W0,           0x400F5000,__READ_WRITE);
__IO_REG32(    GPIO_W1,           0x400F5004,__READ_WRITE);
__IO_REG32(    GPIO_W2,           0x400F5008,__READ_WRITE);
__IO_REG32(    GPIO_W3,           0x400F500C,__READ_WRITE);
__IO_REG32(    GPIO_W4,           0x400F5010,__READ_WRITE);
__IO_REG32(    GPIO_W5,           0x400F5014,__READ_WRITE);
__IO_REG32(    GPIO_W6,           0x400F5018,__READ_WRITE);
__IO_REG32(    GPIO_W7,           0x400F501C,__READ_WRITE);
__IO_REG32(    GPIO_W8,           0x400F5020,__READ_WRITE);
__IO_REG32(    GPIO_W9,           0x400F5024,__READ_WRITE);
__IO_REG32(    GPIO_W10,          0x400F5028,__READ_WRITE);
__IO_REG32(    GPIO_W11,          0x400F502C,__READ_WRITE);
__IO_REG32(    GPIO_W12,          0x400F5030,__READ_WRITE);
__IO_REG32(    GPIO_W13,          0x400F5034,__READ_WRITE);
__IO_REG32(    GPIO_W14,          0x400F5038,__READ_WRITE);
__IO_REG32(    GPIO_W15,          0x400F503C,__READ_WRITE);

__IO_REG32(    GPIO_W32,          0x400F5080,__READ_WRITE);
__IO_REG32(    GPIO_W33,          0x400F5084,__READ_WRITE);
__IO_REG32(    GPIO_W34,          0x400F5088,__READ_WRITE);
__IO_REG32(    GPIO_W35,          0x400F508C,__READ_WRITE);
__IO_REG32(    GPIO_W36,          0x400F5090,__READ_WRITE);
__IO_REG32(    GPIO_W37,          0x400F5094,__READ_WRITE);
__IO_REG32(    GPIO_W38,          0x400F5098,__READ_WRITE);
__IO_REG32(    GPIO_W39,          0x400F509C,__READ_WRITE);
__IO_REG32(    GPIO_W40,          0x400F50A0,__READ_WRITE);
__IO_REG32(    GPIO_W41,          0x400F50A4,__READ_WRITE);
__IO_REG32(    GPIO_W42,          0x400F50A8,__READ_WRITE);
__IO_REG32(    GPIO_W43,          0x400F50AC,__READ_WRITE);
__IO_REG32(    GPIO_W44,          0x400F50B0,__READ_WRITE);
__IO_REG32(    GPIO_W45,          0x400F50B4,__READ_WRITE);
__IO_REG32(    GPIO_W46,          0x400F50B8,__READ_WRITE);
__IO_REG32(    GPIO_W47,          0x400F50BC,__READ_WRITE);

__IO_REG32(    GPIO_W64,          0x400F5100,__READ_WRITE);
__IO_REG32(    GPIO_W65,          0x400F5104,__READ_WRITE);
__IO_REG32(    GPIO_W66,          0x400F5108,__READ_WRITE);
__IO_REG32(    GPIO_W67,          0x400F510C,__READ_WRITE);
__IO_REG32(    GPIO_W68,          0x400F5110,__READ_WRITE);
__IO_REG32(    GPIO_W69,          0x400F5114,__READ_WRITE);
__IO_REG32(    GPIO_W70,          0x400F5118,__READ_WRITE);
__IO_REG32(    GPIO_W71,          0x400F511C,__READ_WRITE);
__IO_REG32(    GPIO_W72,          0x400F5120,__READ_WRITE);
__IO_REG32(    GPIO_W73,          0x400F5124,__READ_WRITE);
__IO_REG32(    GPIO_W74,          0x400F5128,__READ_WRITE);
__IO_REG32(    GPIO_W75,          0x400F512C,__READ_WRITE);
__IO_REG32(    GPIO_W76,          0x400F5130,__READ_WRITE);
__IO_REG32(    GPIO_W77,          0x400F5134,__READ_WRITE);
__IO_REG32(    GPIO_W78,          0x400F5138,__READ_WRITE);
__IO_REG32(    GPIO_W79,          0x400F513C,__READ_WRITE);

__IO_REG32(    GPIO_W96,          0x400F5180,__READ_WRITE);
__IO_REG32(    GPIO_W97,          0x400F5184,__READ_WRITE);
__IO_REG32(    GPIO_W98,          0x400F5188,__READ_WRITE);
__IO_REG32(    GPIO_W99,          0x400F518C,__READ_WRITE);
__IO_REG32(    GPIO_W100,         0x400F5190,__READ_WRITE);
__IO_REG32(    GPIO_W101,         0x400F5194,__READ_WRITE);
__IO_REG32(    GPIO_W102,         0x400F5198,__READ_WRITE);
__IO_REG32(    GPIO_W103,         0x400F519C,__READ_WRITE);
__IO_REG32(    GPIO_W104,         0x400F51A0,__READ_WRITE);
__IO_REG32(    GPIO_W105,         0x400F51A4,__READ_WRITE);
__IO_REG32(    GPIO_W106,         0x400F51A8,__READ_WRITE);
__IO_REG32(    GPIO_W107,         0x400F51AC,__READ_WRITE);
__IO_REG32(    GPIO_W108,         0x400F51B0,__READ_WRITE);
__IO_REG32(    GPIO_W109,         0x400F51B4,__READ_WRITE);
__IO_REG32(    GPIO_W110,         0x400F51B8,__READ_WRITE);
__IO_REG32(    GPIO_W111,         0x400F51BC,__READ_WRITE);

__IO_REG32(    GPIO_W128,         0x400F5200,__READ_WRITE);
__IO_REG32(    GPIO_W129,         0x400F5204,__READ_WRITE);
__IO_REG32(    GPIO_W130,         0x400F5208,__READ_WRITE);
__IO_REG32(    GPIO_W131,         0x400F520C,__READ_WRITE);
__IO_REG32(    GPIO_W132,         0x400F5210,__READ_WRITE);
__IO_REG32(    GPIO_W133,         0x400F5214,__READ_WRITE);
__IO_REG32(    GPIO_W134,         0x400F5218,__READ_WRITE);
__IO_REG32(    GPIO_W135,         0x400F521C,__READ_WRITE);
__IO_REG32(    GPIO_W136,         0x400F5220,__READ_WRITE);
__IO_REG32(    GPIO_W137,         0x400F5224,__READ_WRITE);
__IO_REG32(    GPIO_W138,         0x400F5228,__READ_WRITE);
__IO_REG32(    GPIO_W139,         0x400F522C,__READ_WRITE);
__IO_REG32(    GPIO_W140,         0x400F5230,__READ_WRITE);
__IO_REG32(    GPIO_W141,         0x400F5234,__READ_WRITE);
__IO_REG32(    GPIO_W142,         0x400F5238,__READ_WRITE);
__IO_REG32(    GPIO_W143,         0x400F523C,__READ_WRITE);

__IO_REG32(    GPIO_W160,         0x400F5280,__READ_WRITE);
__IO_REG32(    GPIO_W161,         0x400F5284,__READ_WRITE);
__IO_REG32(    GPIO_W162,         0x400F5288,__READ_WRITE);
__IO_REG32(    GPIO_W163,         0x400F528C,__READ_WRITE);
__IO_REG32(    GPIO_W164,         0x400F5290,__READ_WRITE);
__IO_REG32(    GPIO_W165,         0x400F5294,__READ_WRITE);
__IO_REG32(    GPIO_W166,         0x400F5298,__READ_WRITE);
__IO_REG32(    GPIO_W167,         0x400F529C,__READ_WRITE);
__IO_REG32(    GPIO_W168,         0x400F52A0,__READ_WRITE);
__IO_REG32(    GPIO_W169,         0x400F52A4,__READ_WRITE);
__IO_REG32(    GPIO_W170,         0x400F52A8,__READ_WRITE);
__IO_REG32(    GPIO_W171,         0x400F52AC,__READ_WRITE);
__IO_REG32(    GPIO_W172,         0x400F52B0,__READ_WRITE);
__IO_REG32(    GPIO_W173,         0x400F52B4,__READ_WRITE);
__IO_REG32(    GPIO_W174,         0x400F52B8,__READ_WRITE);
__IO_REG32(    GPIO_W175,         0x400F52BC,__READ_WRITE);
__IO_REG32(    GPIO_W176,         0x400F52C0,__READ_WRITE);
__IO_REG32(    GPIO_W177,         0x400F52C4,__READ_WRITE);
__IO_REG32(    GPIO_W178,         0x400F52C8,__READ_WRITE);
__IO_REG32(    GPIO_W179,         0x400F52CC,__READ_WRITE);
__IO_REG32(    GPIO_W180,         0x400F52D0,__READ_WRITE);
__IO_REG32(    GPIO_W181,         0x400F52D4,__READ_WRITE);
__IO_REG32(    GPIO_W182,         0x400F52D8,__READ_WRITE);
__IO_REG32(    GPIO_W183,         0x400F52DC,__READ_WRITE);
__IO_REG32(    GPIO_W184,         0x400F52E0,__READ_WRITE);
__IO_REG32(    GPIO_W185,         0x400F52E4,__READ_WRITE);
__IO_REG32(    GPIO_W186,         0x400F52E8,__READ_WRITE);

__IO_REG32(    GPIO_W192,         0x400F5300,__READ_WRITE);
__IO_REG32(    GPIO_W193,         0x400F5304,__READ_WRITE);
__IO_REG32(    GPIO_W194,         0x400F5308,__READ_WRITE);
__IO_REG32(    GPIO_W195,         0x400F530C,__READ_WRITE);
__IO_REG32(    GPIO_W196,         0x400F5310,__READ_WRITE);
__IO_REG32(    GPIO_W197,         0x400F5314,__READ_WRITE);
__IO_REG32(    GPIO_W198,         0x400F5318,__READ_WRITE);
__IO_REG32(    GPIO_W199,         0x400F531C,__READ_WRITE);
__IO_REG32(    GPIO_W200,         0x400F5320,__READ_WRITE);
__IO_REG32(    GPIO_W201,         0x400F5324,__READ_WRITE);
__IO_REG32(    GPIO_W202,         0x400F5328,__READ_WRITE);
__IO_REG32(    GPIO_W203,         0x400F532C,__READ_WRITE);
__IO_REG32(    GPIO_W204,         0x400F5330,__READ_WRITE);
__IO_REG32(    GPIO_W205,         0x400F5334,__READ_WRITE);
__IO_REG32(    GPIO_W206,         0x400F5338,__READ_WRITE);
__IO_REG32(    GPIO_W207,         0x400F533C,__READ_WRITE);
__IO_REG32(    GPIO_W208,         0x400F5340,__READ_WRITE);
__IO_REG32(    GPIO_W209,         0x400F5344,__READ_WRITE);
__IO_REG32(    GPIO_W210,         0x400F5348,__READ_WRITE);
__IO_REG32(    GPIO_W211,         0x400F534C,__READ_WRITE);
__IO_REG32(    GPIO_W212,         0x400F5350,__READ_WRITE);
__IO_REG32(    GPIO_W213,         0x400F5354,__READ_WRITE);
__IO_REG32(    GPIO_W214,         0x400F5358,__READ_WRITE);
__IO_REG32(    GPIO_W215,         0x400F535C,__READ_WRITE);
__IO_REG32(    GPIO_W216,         0x400F5360,__READ_WRITE);
__IO_REG32(    GPIO_W217,         0x400F5364,__READ_WRITE);
__IO_REG32(    GPIO_W218,         0x400F5368,__READ_WRITE);
__IO_REG32(    GPIO_W219,         0x400F536C,__READ_WRITE);
__IO_REG32(    GPIO_W220,         0x400F5370,__READ_WRITE);
__IO_REG32(    GPIO_W221,         0x400F5374,__READ_WRITE);
__IO_REG32(    GPIO_W222,         0x400F5378,__READ_WRITE);

__IO_REG32(    GPIO_W224,         0x400F5380,__READ_WRITE);
__IO_REG32(    GPIO_W225,         0x400F5384,__READ_WRITE);
__IO_REG32(    GPIO_W226,         0x400F5388,__READ_WRITE);
__IO_REG32(    GPIO_W227,         0x400F538C,__READ_WRITE);
__IO_REG32(    GPIO_W228,         0x400F5390,__READ_WRITE);
__IO_REG32(    GPIO_W229,         0x400F5394,__READ_WRITE);
__IO_REG32(    GPIO_W230,         0x400F5398,__READ_WRITE);
__IO_REG32(    GPIO_W231,         0x400F539C,__READ_WRITE);
__IO_REG32(    GPIO_W232,         0x400F53A0,__READ_WRITE);
__IO_REG32(    GPIO_W233,         0x400F53A4,__READ_WRITE);
__IO_REG32(    GPIO_W234,         0x400F53A8,__READ_WRITE);
__IO_REG32(    GPIO_W235,         0x400F53AC,__READ_WRITE);
__IO_REG32(    GPIO_W236,         0x400F53B0,__READ_WRITE);
__IO_REG32(    GPIO_W237,         0x400F53B4,__READ_WRITE);
__IO_REG32(    GPIO_W238,         0x400F53B8,__READ_WRITE);
__IO_REG32(    GPIO_W239,         0x400F53BC,__READ_WRITE);
__IO_REG32(    GPIO_W240,         0x400F53C0,__READ_WRITE);
__IO_REG32(    GPIO_W241,         0x400F53C4,__READ_WRITE);
__IO_REG32(    GPIO_W242,         0x400F53C8,__READ_WRITE);
__IO_REG32(    GPIO_W243,         0x400F53CC,__READ_WRITE);
__IO_REG32(    GPIO_W244,         0x400F53D0,__READ_WRITE);
__IO_REG32(    GPIO_W245,         0x400F53D4,__READ_WRITE);
__IO_REG32(    GPIO_W246,         0x400F53D8,__READ_WRITE);
__IO_REG32(    GPIO_W247,         0x400F53DC,__READ_WRITE);
__IO_REG32(    GPIO_W248,         0x400F53E0,__READ_WRITE);
__IO_REG32(    GPIO_W249,         0x400F53E4,__READ_WRITE);

__IO_REG32(    GPIO_DIR0,         0x400F6000,__READ_WRITE);
__IO_REG32(    GPIO_DIR1,         0x400F6004,__READ_WRITE);
__IO_REG32(    GPIO_DIR2,         0x400F6008,__READ_WRITE);
__IO_REG32(    GPIO_DIR3,         0x400F600C,__READ_WRITE);
__IO_REG32(    GPIO_DIR4,         0x400F6010,__READ_WRITE);
__IO_REG32(    GPIO_DIR5,         0x400F6014,__READ_WRITE);
__IO_REG32(    GPIO_DIR6,         0x400F6018,__READ_WRITE);
__IO_REG32(    GPIO_DIR7,         0x400F601C,__READ_WRITE);

__IO_REG32(    GPIO_MASK0,        0x400F6080,__READ_WRITE);
__IO_REG32(    GPIO_MASK1,        0x400F6084,__READ_WRITE);
__IO_REG32(    GPIO_MASK2,        0x400F6088,__READ_WRITE);
__IO_REG32(    GPIO_MASK3,        0x400F608C,__READ_WRITE);
__IO_REG32(    GPIO_MASK4,        0x400F6090,__READ_WRITE);
__IO_REG32(    GPIO_MASK5,        0x400F6094,__READ_WRITE);
__IO_REG32(    GPIO_MASK6,        0x400F6098,__READ_WRITE);
__IO_REG32(    GPIO_MASK7,        0x400F609C,__READ_WRITE);

__IO_REG32(    GPIO_PIN0,         0x400F6100,__READ_WRITE);
__IO_REG32(    GPIO_PIN1,         0x400F6104,__READ_WRITE);
__IO_REG32(    GPIO_PIN2,         0x400F6108,__READ_WRITE);
__IO_REG32(    GPIO_PIN3,         0x400F610C,__READ_WRITE);
__IO_REG32(    GPIO_PIN4,         0x400F6110,__READ_WRITE);
__IO_REG32(    GPIO_PIN5,         0x400F6114,__READ_WRITE);
__IO_REG32(    GPIO_PIN6,         0x400F6118,__READ_WRITE);
__IO_REG32(    GPIO_PIN7,         0x400F611C,__READ_WRITE);

__IO_REG32(    GPIO_MPIN0,        0x400F6180,__READ_WRITE);
__IO_REG32(    GPIO_MPIN1,        0x400F6184,__READ_WRITE);
__IO_REG32(    GPIO_MPIN2,        0x400F6188,__READ_WRITE);
__IO_REG32(    GPIO_MPIN3,        0x400F618C,__READ_WRITE);
__IO_REG32(    GPIO_MPIN4,        0x400F6190,__READ_WRITE);
__IO_REG32(    GPIO_MPIN5,        0x400F6194,__READ_WRITE);
__IO_REG32(    GPIO_MPIN6,        0x400F6198,__READ_WRITE);
__IO_REG32(    GPIO_MPIN7,        0x400F619C,__READ_WRITE);

__IO_REG32(    GPIO_SET0,         0x400F6200,__READ_WRITE);
__IO_REG32(    GPIO_SET1,         0x400F6204,__READ_WRITE);
__IO_REG32(    GPIO_SET2,         0x400F6208,__READ_WRITE);
__IO_REG32(    GPIO_SET3,         0x400F620C,__READ_WRITE);
__IO_REG32(    GPIO_SET4,         0x400F6210,__READ_WRITE);
__IO_REG32(    GPIO_SET5,         0x400F6214,__READ_WRITE);
__IO_REG32(    GPIO_SET6,         0x400F6218,__READ_WRITE);
__IO_REG32(    GPIO_SET7,         0x400F621C,__READ_WRITE);

__IO_REG32(    GPIO_CLR0,         0x400F6280,__WRITE);
__IO_REG32(    GPIO_CLR1,         0x400F6284,__WRITE);
__IO_REG32(    GPIO_CLR2,         0x400F6288,__WRITE);
__IO_REG32(    GPIO_CLR3,         0x400F628C,__WRITE);
__IO_REG32(    GPIO_CLR4,         0x400F6290,__WRITE);
__IO_REG32(    GPIO_CLR5,         0x400F6294,__WRITE);
__IO_REG32(    GPIO_CLR6,         0x400F6298,__WRITE);
__IO_REG32(    GPIO_CLR7,         0x400F629C,__WRITE);

__IO_REG32(    GPIO_NOT0,         0x400F6300,__WRITE);
__IO_REG32(    GPIO_NOT1,         0x400F6304,__WRITE);
__IO_REG32(    GPIO_NOT2,         0x400F6308,__WRITE);
__IO_REG32(    GPIO_NOT3,         0x400F630C,__WRITE);
__IO_REG32(    GPIO_NOT4,         0x400F6310,__WRITE);
__IO_REG32(    GPIO_NOT5,         0x400F6314,__WRITE);
__IO_REG32(    GPIO_NOT6,         0x400F6318,__WRITE);
__IO_REG32(    GPIO_NOT7,         0x400F631C,__WRITE);

/***************************************************************************
 **
 ** SGPIO
 **
 ***************************************************************************/

__IO_REG32_BIT(SGPIO_OUT_MUX_CFG0,    0x40101000,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG1,    0x40101004,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG2,    0x40101008,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG3,    0x4010100C,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG4,    0x40101010,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG5,    0x40101014,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG6,    0x40101018,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG7,    0x4010101C,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG8,    0x40101020,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG9,    0x40101024,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG10,   0x40101028,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG11,   0x4010102C,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG12,   0x40101030,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG13,   0x40101034,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG14,   0x40101038,__READ_WRITE,__sgpio_out_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_OUT_MUX_CFG15,   0x4010103C,__READ_WRITE,__sgpio_out_mux_cfgx_bits);

__IO_REG32_BIT(SGPIO_MUX_CFG0,        0x40101040,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG1,        0x40101044,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG2,        0x40101048,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG3,        0x4010104C,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG4,        0x40101050,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG5,        0x40101054,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG6,        0x40101058,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG7,        0x4010105C,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG8,        0x40101060,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG9,        0x40101064,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG10,       0x40101068,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG11,       0x4010106C,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG12,       0x40101070,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG13,       0x40101074,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG14,       0x40101078,__READ_WRITE,__sgpio_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_MUX_CFG15,       0x4010107C,__READ_WRITE,__sgpio_mux_cfgx_bits);

__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG0,  0x40101080,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG1,  0x40101084,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG2,  0x40101088,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG3,  0x4010108C,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG4,  0x40101090,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG5,  0x40101094,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG6,  0x40101098,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG7,  0x4010109C,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG8,  0x401010A0,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG9,  0x401010A4,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG10, 0x401010A8,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG11, 0x401010AC,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG12, 0x401010B0,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG13, 0x401010B4,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG14, 0x401010B8,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);
__IO_REG32_BIT(SGPIO_SLICE_MUX_CFG15, 0x401010BC,__READ_WRITE,__sgpio_slice_mux_cfgx_bits);

__IO_REG32(    SGPIO_REG0,            0x401010C0,__READ_WRITE);
__IO_REG32(    SGPIO_REG1,            0x401010C4,__READ_WRITE);
__IO_REG32(    SGPIO_REG2,            0x401010C8,__READ_WRITE);
__IO_REG32(    SGPIO_REG3,            0x401010CC,__READ_WRITE);
__IO_REG32(    SGPIO_REG4,            0x401010D0,__READ_WRITE);
__IO_REG32(    SGPIO_REG5,            0x401010D4,__READ_WRITE);
__IO_REG32(    SGPIO_REG6,            0x401010D8,__READ_WRITE);
__IO_REG32(    SGPIO_REG7,            0x401010DC,__READ_WRITE);
__IO_REG32(    SGPIO_REG8,            0x401010E0,__READ_WRITE);
__IO_REG32(    SGPIO_REG9,            0x401010E4,__READ_WRITE);
__IO_REG32(    SGPIO_REG10,           0x401010E8,__READ_WRITE);
__IO_REG32(    SGPIO_REG11,           0x401010EC,__READ_WRITE);
__IO_REG32(    SGPIO_REG12,           0x401010F0,__READ_WRITE);
__IO_REG32(    SGPIO_REG13,           0x401010F4,__READ_WRITE);
__IO_REG32(    SGPIO_REG14,           0x401010F8,__READ_WRITE);
__IO_REG32(    SGPIO_REG15,           0x401010FC,__READ_WRITE);

__IO_REG32(    SGPIO_REG_SS0,         0x40101100,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS1,         0x40101104,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS2,         0x40101108,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS3,         0x4010110C,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS4,         0x40101110,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS5,         0x40101114,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS6,         0x40101118,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS7,         0x4010111C,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS8,         0x40101120,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS9,         0x40101124,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS10,        0x40101128,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS11,        0x4010112C,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS12,        0x40101130,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS13,        0x40101134,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS14,        0x40101138,__READ_WRITE);
__IO_REG32(    SGPIO_REG_SS15,        0x4010113C,__READ_WRITE);

__IO_REG32_BIT(SGPIO_PRESET0,         0x40101140,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET1,         0x40101144,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET2,         0x40101148,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET3,         0x4010114C,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET4,         0x40101150,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET5,         0x40101154,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET6,         0x40101158,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET7,         0x4010115C,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET8,         0x40101160,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET9,         0x40101164,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET10,        0x40101168,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET11,        0x4010116C,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET12,        0x40101170,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET13,        0x40101174,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET14,        0x40101178,__READ_WRITE,__sgpio_presetx_bits);
__IO_REG32_BIT(SGPIO_PRESET15,        0x4010117C,__READ_WRITE,__sgpio_presetx_bits);

__IO_REG32_BIT(SGPIO_COUNT0,          0x40101180,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT1,          0x40101184,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT2,          0x40101188,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT3,          0x4010118C,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT4,          0x40101190,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT5,          0x40101194,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT6,          0x40101198,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT7,          0x4010119C,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT8,          0x401011A0,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT9,          0x401011A4,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT10,         0x401011A8,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT11,         0x401011AC,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT12,         0x401011B0,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT13,         0x401011B4,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT14,         0x401011B8,__READ_WRITE,__sgpio_countx_bits);
__IO_REG32_BIT(SGPIO_COUNT15,         0x401011BC,__READ_WRITE,__sgpio_countx_bits);

__IO_REG32_BIT(SGPIO_POS0,            0x401011C0,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS1,            0x401011C4,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS2,            0x401011C8,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS3,            0x401011CC,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS4,            0x401011D0,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS5,            0x401011D4,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS6,            0x401011D8,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS7,            0x401011DC,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS8,            0x401011E0,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS9,            0x401011E4,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS10,           0x401011E8,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS11,           0x401011EC,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS12,           0x401011F0,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS13,           0x401011F4,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS14,           0x401011F8,__READ_WRITE,__sgpio_posx_bits);
__IO_REG32_BIT(SGPIO_POS15,           0x401011FC,__READ_WRITE,__sgpio_posx_bits);

__IO_REG32(    SGPIO_MASK_A,          0x40101200,__READ_WRITE);
__IO_REG32(    SGPIO_MASK_H,          0x40101204,__READ_WRITE);
__IO_REG32(    SGPIO_MASK_I,          0x40101208,__READ_WRITE);
__IO_REG32(    SGPIO_MASK_P,          0x4010120C,__READ_WRITE);
__IO_REG32_BIT(SGPIO_GPIO_INREG,      0x40101210,__READ      ,__sgpio_gpio_inreg_bits);
__IO_REG32_BIT(SGPIO_GPIO_OUTREG,     0x40101214,__READ_WRITE,__sgpio_gpio_outreg_bits);
__IO_REG32_BIT(SGPIO_GPIO_OENREG,     0x40101218,__READ_WRITE,__sgpio_gpio_oenreg_bits);
__IO_REG32_BIT(SGPIO_CTRL_ENABLED,    0x4010121C,__READ_WRITE,__sgpio_ctrl_enabled_bits);
__IO_REG32_BIT(SGPIO_CTRL_DISABLED,   0x40101220,__READ_WRITE,__sgpio_ctrl_disabled_bits);

__IO_REG32(    SGPIO_CLR_EN_0,        0x40101F00,__WRITE     );
__IO_REG32(    SGPIO_SET_EN_0,        0x40101F04,__WRITE     );
__IO_REG32_BIT(SGPIO_ENABLE_0,        0x40101F08,__READ      ,__sgpio_enable_0_bits);
__IO_REG32_BIT(SGPIO_STATUS_0,        0x40101F0C,__READ      ,__sgpio_status_0_bits);
__IO_REG32(    SGPIO_CTR_STATUS_0,    0x40101F10,__WRITE     );
__IO_REG32(    SGPIO_SET_STATUS_0,    0x40101F14,__WRITE     );

__IO_REG32(    SGPIO_CLR_EN_1,        0x40101F20,__WRITE     );
__IO_REG32(    SGPIO_SET_EN_1,        0x40101F24,__WRITE     );
__IO_REG32_BIT(SGPIO_ENABLE_1,        0x40101F28,__READ      ,__sgpio_enable_1_bits);
__IO_REG32_BIT(SGPIO_STATUS_1,        0x40101F2C,__READ      ,__sgpio_status_1_bits);
__IO_REG32(    SGPIO_CTR_STATUS_1,    0x40101F30,__WRITE     );
__IO_REG32(    SGPIO_SET_STATUS_1,    0x40101F34,__WRITE     );

__IO_REG32(    SGPIO_CLR_EN_2,        0x40101F40,__WRITE     );
__IO_REG32(    SGPIO_SET_EN_2,        0x40101F44,__WRITE     );
__IO_REG32_BIT(SGPIO_ENABLE_2,        0x40101F48,__READ      ,__sgpio_enable_2_bits);
__IO_REG32_BIT(SGPIO_STATUS_2,        0x40101F4C,__READ      ,__sgpio_status_2_bits);
__IO_REG32(    SGPIO_CTR_STATUS_2,    0x40101F50,__WRITE     );
__IO_REG32(    SGPIO_SET_STATUS_2,    0x40101F54,__WRITE     );

__IO_REG32(    SGPIO_CLR_EN_3,        0x40101F60,__WRITE     );
__IO_REG32(    SGPIO_SET_EN_3,        0x40101F64,__WRITE     );
__IO_REG32_BIT(SGPIO_ENABLE_3,        0x40101F68,__READ      ,__sgpio_enable_3_bits);
__IO_REG32_BIT(SGPIO_STATUS_3,        0x40101F6C,__READ      ,__sgpio_status_3_bits);
__IO_REG32(    SGPIO_CTR_STATUS_3,    0x40101F70,__WRITE     );
__IO_REG32(    SGPIO_SET_STATUS_3,    0x40101F74,__WRITE     );



/***************************************************************************
 **
 ** GPDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACINTSTATUS,         0x40002000,__READ      ,__dmacintstatus_bits);
#define GPDMA_INTSTAT      DMACINTSTATUS
#define GPDMA_INTSTAT_bit  DMACINTSTATUS_bit

__IO_REG32_BIT(DMACINTTCSTATUS,       0x40002004,__READ      ,__dmacinttcstatus_bits);
#define GPDMA_INTTCSTAT      DMACINTTCSTATUS
#define GPDMA_INTTCSTAT_bit  DMACINTTCSTATUS_bit

__IO_REG32(   DMACINTTCCLEAR,        0x40002008,__WRITE     );
#define GPDMA_INTTCCLEAR     DMACINTTCCLEAR

__IO_REG32_BIT(DMACINTERRSTAT,        0x4000200C,__READ      ,__dmacinterrstat_bits);
#define GPDMA_INTERRSTAT      DMACINTERRSTAT
#define GPDMA_INTERRSTAT_bit  DMACINTERRSTAT_bit

__IO_REG32(    DMACINTERRCLR,         0x40002010,__WRITE     );
#define GPDMA_INTERRCLR      DMACINTERRCLR

__IO_REG32_BIT(DMACRAWINTTCSTATUS,    0x40002014,__READ      ,__dmacrawinttcstatus_bits);
#define GPDMA_RAWINTTCSTAT       DMACRAWINTTCSTATUS
#define GPDMA_RAWINTTCSTAT_bit   DMACRAWINTTCSTATUS_bit

__IO_REG32_BIT(DMACRAWINTERRORSTATUS, 0x40002018,__READ      ,__dmacrawinterrorstatus_bits);
#define GPDMA_RAWINTERRSTAT      DMACRAWINTERRORSTATUS
#define GPDMA_RAWINTERRSTAT_bit  DMACRAWINTERRORSTATUS_bit

__IO_REG32_BIT(DMACENBLDCHNS,         0x4000201C,__READ      ,__dmacenbldchns_bits);
#define GPDMA_ENBLDCHNS      DMACENBLDCHNS
#define GPDMA_ENBLDCHNS_bit  DMACENBLDCHNS_bit

__IO_REG32_BIT(DMACSOFTBREQ,          0x40002020,__READ_WRITE,__dmacsoftbreq_bits);
#define GPDMA_SOFTBREQ      DMACSOFTBREQ
#define GPDMA_SOFTBREQ_bit  DMACSOFTBREQ_bit

__IO_REG32_BIT(DMACSOFTSREQ,          0x40002024,__READ_WRITE,__dmacsoftsreq_bits);
#define GPDMA_SOFTSREQ      DMACSOFTSREQ
#define GPDMA_SOFTSREQ_bit  DMACSOFTSREQ_bit

__IO_REG32_BIT(DMACSOFTLBREQ,         0x40002028,__READ_WRITE,__dmacsoftlbreq_bits);
#define GPDMA_SOFTLBREQ      DMACSOFTLBREQ
#define GPDMA_SOFTLBREQ_bit  DMACSOFTLBREQ_bit

__IO_REG32_BIT(DMACSOFTLSREQ,         0x4000202C,__READ_WRITE,__dmacsoftlsreq_bits);
#define GPDMA_SOFTLSREQ      DMACSOFTLSREQ
#define GPDMA_SOFTLSREQ_bit  DMACSOFTLSREQ_bit

__IO_REG32_BIT(DMACCONFIGURATION,     0x40002030,__READ_WRITE,__dmacconfig_bits);
#define GPDMA_CONFIG      DMACCONFIGURATION
#define GPDMA_CONFIG_bit  DMACCONFIGURATION_bit

__IO_REG32_BIT(DMACSYNC,              0x40002034,__READ_WRITE,__dmacsync_bits);
#define GPDMA_SYNC      DMACSYNC
#define GPDMA_SYNC_bit  DMACSYNC_bit

__IO_REG32(    DMACC0SRCADDR,         0x40002100,__READ_WRITE);
#define GPDMA_C0SRCADDR      DMACC0SRCADDR

__IO_REG32(    DMACC0DESTADDR,        0x40002104,__READ_WRITE);
#define GPDMA_C0DESTADDR      DMACC0DESTADDR

__IO_REG32_BIT(DMACC0LLI,             0x40002108,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C0LLI      DMACC0LLI
#define GPDMA_C0LLI_bit  DMACC0LLI_bit

__IO_REG32_BIT(DMACC0CONTROL,         0x4000210C,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C0CONTROL      DMACC0CONTROL
#define GPDMA_C0CONTROL_bit  DMACC0CONTROL_bit

__IO_REG32_BIT(DMACC0CONFIGURATION,   0x40002110,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C0CONFIG      DMACC0CONFIGURATION
#define GPDMA_C0CONFIG_bit  DMACC0CONFIGURATION_bit

__IO_REG32(    DMACC1SRCADDR,         0x40002120,__READ_WRITE);
#define GPDMA_C1SRCADDR      DMACC1SRCADDR

__IO_REG32(    DMACC1DESTADDR,        0x40002124,__READ_WRITE);
#define GPDMA_C1DESTADDR      DMACC1DESTADDR

__IO_REG32_BIT(DMACC1LLI,             0x40002128,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C1LLI      DMACC1LLI
#define GPDMA_C1LLI_bit  DMACC1LLI_bit

__IO_REG32_BIT(DMACC1CONTROL,         0x4000212C,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C1CONTROL      DMACC1CONTROL
#define GPDMA_C1CONTROL_bit  DMACC1CONTROL_bit

__IO_REG32_BIT(DMACC1CONFIGURATION,   0x40002130,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C1CONFIG      DMACC1CONFIGURATION
#define GPDMA_C1CONFIG_bit  DMACC1CONFIGURATION_bit

__IO_REG32(    DMACC2SRCADDR,         0x40002140,__READ_WRITE);
#define GPDMA_C2SRCADDR      DMACC2SRCADDR

__IO_REG32(    DMACC2DESTADDR,        0x40002144,__READ_WRITE);
#define GPDMA_C2DESTADDR      DMACC2DESTADDR

__IO_REG32_BIT(DMACC2LLI,             0x40002148,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C2LLI      DMACC2LLI
#define GPDMA_C2LLI_bit  DMACC2LLI_bit

__IO_REG32_BIT(DMACC2CONTROL,         0x4000214C,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C2CONTROL      DMACC2CONTROL
#define GPDMA_C2CONTROL_bit  DMACC2CONTROL_bit

__IO_REG32_BIT(DMACC2CONFIGURATION,   0x40002150,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C2CONFIG      DMACC2CONFIGURATION
#define GPDMA_C2CONFIG_bit  DMACC2CONFIGURATION_bit

__IO_REG32(    DMACC3SRCADDR,         0x40002160,__READ_WRITE);
#define GPDMA_C3SRCADDR      DMACC3SRCADDR

__IO_REG32(    DMACC3DESTADDR,        0x40002164,__READ_WRITE);
#define GPDMA_C3DESTADDR      DMACC3DESTADDR

__IO_REG32_BIT(DMACC3LLI,             0x40002168,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C3LLI      DMACC3LLI
#define GPDMA_C3LLI_bit  DMACC3LLI_bit

__IO_REG32_BIT(DMACC3CONTROL,         0x4000216C,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C3CONTROL      DMACC3CONTROL
#define GPDMA_C3CONTROL_bit  DMACC3CONTROL_bit

__IO_REG32_BIT(DMACC3CONFIGURATION,   0x40002170,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C3CONFIG      DMACC3CONFIGURATION
#define GPDMA_C3CONFIG_bit  DMACC3CONFIGURATION_bit

__IO_REG32(    DMACC4SRCADDR,         0x40002180,__READ_WRITE);
#define GPDMA_C4SRCADDR      DMACC4SRCADDR

__IO_REG32(    DMACC4DESTADDR,        0x40002184,__READ_WRITE);
#define GPDMA_C4DESTADDR      DMACC4DESTADDR

__IO_REG32_BIT(DMACC4LLI,             0x40002188,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C4LLI      DMACC4LLI
#define GPDMA_C4LLI_bit  DMACC4LLI_bit

__IO_REG32_BIT(DMACC4CONTROL,         0x4000218C,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C4CONTROL      DMACC4CONTROL
#define GPDMA_C4CONTROL_bit  DMACC4CONTROL_bit

__IO_REG32_BIT(DMACC4CONFIGURATION,   0x40002190,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C4CONFIG      DMACC4CONFIGURATION
#define GPDMA_C4CONFIG_bit  DMACC4CONFIGURATION_bit

__IO_REG32(    DMACC5SRCADDR,         0x400021A0,__READ_WRITE);
#define GPDMA_C5SRCADDR      DMACC5SRCADDR

__IO_REG32(    DMACC5DESTADDR,        0x400021A4,__READ_WRITE);
#define GPDMA_C5DESTADDR      DMACC5DESTADDR

__IO_REG32_BIT(DMACC5LLI,             0x400021A8,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C5LLI      DMACC5LLI
#define GPDMA_C5LLI_bit  DMACC5LLI_bit

__IO_REG32_BIT(DMACC5CONTROL,         0x400021AC,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C5CONTROL      DMACC5CONTROL
#define GPDMA_C5CONTROL_bit  DMACC5CONTROL_bit

__IO_REG32_BIT(DMACC5CONFIGURATION,   0x400021B0,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C5CONFIG      DMACC5CONFIGURATION
#define GPDMA_C5CONFIG_bit  DMACC5CONFIGURATION_bit

__IO_REG32(    DMACC6SRCADDR,         0x400021C0,__READ_WRITE);
#define GPDMA_C6SRCADDR      DMACC6SRCADDR

__IO_REG32(    DMACC6DESTADDR,        0x400021C4,__READ_WRITE);
#define GPDMA_C6DESTADDR      DMACC6DESTADDR

__IO_REG32_BIT(DMACC6LLI,             0x400021C8,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C6LLI      DMACC6LLI
#define GPDMA_C6LLI_bit  DMACC6LLI_bit

__IO_REG32_BIT(DMACC6CONTROL,         0x400021CC,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C6CONTROL      DMACC6CONTROL
#define GPDMA_C6CONTROL_bit  DMACC6CONTROL_bit

__IO_REG32_BIT(DMACC6CONFIGURATION,   0x400021D0,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C6CONFIG      DMACC6CONFIGURATION
#define GPDMA_C6CONFIG_bit  DMACC6CONFIGURATION_bit

__IO_REG32(    DMACC7SRCADDR,         0x400021E0,__READ_WRITE);
#define GPDMA_C7SRCADDR      DMACC7SRCADDR

__IO_REG32(    DMACC7DESTADDR,        0x400021E4,__READ_WRITE);
#define GPDMA_C7DESTADDR      DMACC7DESTADDR

__IO_REG32_BIT(DMACC7LLI,             0x400021E8,__READ_WRITE,__dma_lli_bits);
#define GPDMA_C7LLI      DMACC7LLI
#define GPDMA_C7LLI_bit  DMACC7LLI_bit

__IO_REG32_BIT(DMACC7CONTROL,         0x400021EC,__READ_WRITE,__dma_ctrl_bits);
#define GPDMA_C7CONTROL      DMACC7CONTROL
#define GPDMA_C7CONTROL_bit  DMACC7CONTROL_bit

__IO_REG32_BIT(DMACC7CONFIGURATION,   0x400021F0,__READ_WRITE,__dma_cfg_bits);
#define GPDMA_C7CONFIG      DMACC7CONFIGURATION
#define GPDMA_C7CONFIG_bit  DMACC7CONFIGURATION_bit

/***************************************************************************
 **
 ** SDIO
 **
 ***************************************************************************/
__IO_REG32_BIT(SDIO_CTRL,             0x40004000,__READ_WRITE ,__sdio_ctrl_bits);
__IO_REG32_BIT(SDIO_PWREN,            0x40004004,__READ_WRITE ,__sdio_pwren_bits);
__IO_REG32_BIT(SDIO_CLKDIV,           0x40004008,__READ_WRITE ,__sdio_clkdiv_bits);
__IO_REG32_BIT(SDIO_CLKSRC,           0x4000400C,__READ_WRITE ,__sdio_clksrc_bits);
__IO_REG32_BIT(SDIO_CLKENA,           0x40004010,__READ_WRITE ,__sdio_clkena_bits);
__IO_REG32_BIT(SDIO_TMOUT,            0x40004014,__READ_WRITE ,__sdio_tmout_bits);
__IO_REG32_BIT(SDIO_CTYPE,            0x40004018,__READ_WRITE ,__sdio_ctype_bits);
__IO_REG32_BIT(SDIO_BLKSIZ,           0x4000401C,__READ_WRITE ,__sdio_blksiz_bits);
__IO_REG32(    SDIO_BYTCNT,           0x40004020,__READ_WRITE );
__IO_REG32_BIT(SDIO_INTMASK,          0x40004024,__READ_WRITE ,__sdio_intmask_bits);
__IO_REG32(    SDIO_CMDARG,           0x40004028,__READ_WRITE );
__IO_REG32_BIT(SDIO_CMD,              0x4000402C,__READ_WRITE ,__sdio_cmd_bits);
__IO_REG32(    SDIO_RESP0,            0x40004030,__READ       );
__IO_REG32(    SDIO_RESP1,            0x40004034,__READ       );
__IO_REG32(    SDIO_RESP2,            0x40004038,__READ       );
__IO_REG32(    SDIO_RESP3,            0x4000403C,__READ       );
__IO_REG32_BIT(SDIO_MINTSTS,          0x40004040,__READ       ,__sdio_mintsts_bits);
__IO_REG32_BIT(SDIO_RINTSTS,          0x40004044,__READ_WRITE ,__sdio_rintsts_bits);
__IO_REG32_BIT(SDIO_STATUS,           0x40004048,__READ       ,__sdio_status_bits);
__IO_REG32_BIT(SDIO_FIFOTH,           0x4000404C,__READ_WRITE ,__sdio_fifoth_bits);
__IO_REG32_BIT(SDIO_CDETECT,          0x40004050,__READ       ,__sdio_cdetect_bits);
__IO_REG32_BIT(SDIO_WRTPRT,           0x40004054,__READ       ,__sdio_wrtprt_bits);
/*__IO_REG32_BIT(SDIO_GPIO,             0x40004058,__READ_WRITE ,__sdio_gpio_bits);*/
__IO_REG32(    SDIO_TCBCNT,           0x4000405C,__READ       );
__IO_REG32(    SDIO_TBBCNT,           0x40004060,__READ       );
__IO_REG32_BIT(SDIO_DEBNCE,           0x40004064,__READ_WRITE ,__sdio_debnce_bits);
/*__IO_REG32(    SDIO_USRID,            0x40004068,__READ_WRITE );*/
/*__IO_REG32(    SDIO_VERID,            0x4000406C,__READ       );*/
__IO_REG32_BIT(SDIO_UHS_REG,          0x40004074,__READ_WRITE ,__sdio_uhs_reg_bits);
__IO_REG32_BIT(SDIO_RST_N,            0x40004078,__READ_WRITE ,__sdio_rst_n_bits);
__IO_REG32_BIT(SDIO_BMOD,             0x40004080,__READ_WRITE ,__sdio_bmod_bits);
__IO_REG32(    SDIO_PLDMND,           0x40004084,__WRITE      );
__IO_REG32(    SDIO_DBADDR,           0x40004088,__READ_WRITE );
__IO_REG32_BIT(SDIO_IDSTS,            0x4000408C,__READ_WRITE ,__sdio_idsts_bits);
__IO_REG32_BIT(SDIO_IDINTEN,          0x40004090,__READ_WRITE ,__sdio_idinten_bits);
__IO_REG32(    SDIO_DSCADDR,          0x40004094,__READ       );
__IO_REG32(    SDIO_BUFADDR,          0x40004098,__READ       );
__IO_REG32(    SDIO_DATA,             0x40004100,__READ_WRITE );

/***************************************************************************
 **
 ** EMC
 **
 ***************************************************************************/
__IO_REG32_BIT(EMC_CONTROL,            0x40005000,__READ_WRITE ,__emc_control_bits);
__IO_REG32_BIT(EMC_STATUS,             0x40005004,__READ       ,__emc_status_bits);
__IO_REG32_BIT(EMC_CONFIG,             0x40005008,__READ_WRITE ,__emc_config_bits);
__IO_REG32_BIT(EMC_DYNAMICCONTROL,     0x40005020,__READ_WRITE ,__emc_dctrl_bits);
__IO_REG32_BIT(EMC_DYNAMICREFRESH,     0x40005024,__READ_WRITE ,__emc_drfr_bits);
__IO_REG32_BIT(EMC_DYNAMICREADCONFIG,  0x40005028,__READ_WRITE ,__emc_drdcfg_bits);
__IO_REG32_BIT(EMC_DYNAMICRP,          0x40005030,__READ_WRITE ,__emc_drp_bits);
__IO_REG32_BIT(EMC_DYNAMICRAS,         0x40005034,__READ_WRITE ,__emc_dras_bits);
__IO_REG32_BIT(EMC_DYNAMICSREX,        0x40005038,__READ_WRITE ,__emc_dsrex_bits);
__IO_REG32_BIT(EMC_DYNAMICAPR,         0x4000503C,__READ_WRITE ,__emc_dapr_bits);
__IO_REG32_BIT(EMC_DYNAMICDAL,         0x40005040,__READ_WRITE ,__emc_ddal_bits);
__IO_REG32_BIT(EMC_DYNAMICWR,          0x40005044,__READ_WRITE ,__emc_dwr_bits);
__IO_REG32_BIT(EMC_DYNAMICRC,          0x40005048,__READ_WRITE ,__emc_drc_bits);
__IO_REG32_BIT(EMC_DYNAMICRFC,         0x4000504C,__READ_WRITE ,__emc_drfc_bits);
__IO_REG32_BIT(EMC_DYNAMICXSR,         0x40005050,__READ_WRITE ,__emc_dxsr_bits);
__IO_REG32_BIT(EMC_DYNAMICRRD,         0x40005054,__READ_WRITE ,__emc_drrd_bits);
__IO_REG32_BIT(EMC_DYNAMICMRD,         0x40005058,__READ_WRITE ,__emc_dmrd_bits);
__IO_REG32_BIT(EMC_STATICEXTENDEDWAIT, 0x40005080,__READ_WRITE ,__emc_s_ext_wait_bits);
__IO_REG32_BIT(EMC_DYNAMICCONFIG0,     0x40005100,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMC_DYNAMICRASCAS0,     0x40005104,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMC_DYNAMICCONFIG1,     0x40005120,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMC_DYNAMICRASCAS1,     0x40005124,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMC_DYNAMICCONFIG2,     0x40005140,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMC_DYNAMICRASCAS2,     0x40005144,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMC_DYNAMICCONFIG3,     0x40005160,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMC_DYNAMICRASCAS3,     0x40005164,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMC_STATICCONFIG0,      0x40005200,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMC_STATICWAITWEN0,     0x40005204,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMC_STATICWAITOEN0,     0x40005208,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMC_STATICWAITRD0,      0x4000520C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMC_STATICWAITPAGE0,    0x40005210,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMC_STATICWAITWR0,      0x40005214,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMC_STATICWAITTURN0,    0x40005218,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMC_STATICCONFIG1,      0x40005220,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMC_STATICWAITWEN1,     0x40005224,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMC_STATICWAITOEN1,     0x40005228,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMC_STATICWAITRD1,      0x4000522C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMC_STATICWAITPAGE1,    0x40005230,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMC_STATICWAITWR1,      0x40005234,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMC_STATICWAITTURN1,    0x40005238,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMC_STATICCONFIG2,      0x40005240,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMC_STATICWAITWEN2,     0x40005244,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMC_STATICWAITOEN2,     0x40005248,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMC_STATICWAITRD2,      0x4000524C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMC_STATICWAITPAGE2,    0x40005250,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMC_STATICWAITWR2,      0x40005254,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMC_STATICWAITTURN2,    0x40005258,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMC_STATICCONFIG3,      0x40005260,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMC_STATICWAITWEN3,     0x40005264,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMC_STATICWAITOEN3,     0x40005268,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMC_STATICWAITRD3,      0x4000526C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMC_STATICWAITPAGE3,    0x40005270,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMC_STATICWAITWR3,      0x40005274,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMC_STATICWAITTURN3,    0x40005278,__READ_WRITE ,__emc_s_wait_turn_bits);

/***************************************************************************
 **
 ** USB0
 **
 ***************************************************************************/
__IO_REG32_BIT(USB0_CAPLENGTH,        0x40006100,__READ      ,__usb_caplength_reg_bits);
__IO_REG32_BIT(USB0_HCSPARAMS,        0x40006104,__READ      ,__usb_hcsparams_reg_bits);
__IO_REG32_BIT(USB0_HCCPARAMS,        0x40006108,__READ      ,__usb_hccparams_reg_bits);
__IO_REG32_BIT(USB0_DCIVERSION,       0x40006120,__READ      ,__usb_dciversion_reg_bits);
__IO_REG32_BIT(USB0_DCCPARAMS,        0x40006124,__READ      ,__usb_dccparams_reg_bits);
__IO_REG32_BIT(USB0_USBCMD,           0x40006140,__READ_WRITE,__usb_usbcmd_reg_bits);
#define USB0_USBCMD_D         USB0_USBCMD
#define USB0_USBCMD_D_bit     USB0_USBCMD_bit
#define USB0_USBCMD_H         USB0_USBCMD
#define USB0_USBCMD_H_bit     USB0_USBCMD_bit
__IO_REG32_BIT(USB0_USBSTS,           0x40006144,__READ_WRITE,__usb_usbsts_reg_bits);
#define USB0_USBSTS_D         USB0_USBSTS
#define USB0_USBSTS_D_bit     USB0_USBSTS_bit
#define USB0_USBSTS_H         USB0_USBSTS
#define USB0_USBSTS_H_bit     USB0_USBSTS_bit
__IO_REG32_BIT(USB0_USBINTR,          0x40006148,__READ_WRITE,__usb_usbintr_reg_bits);
#define USB0_USBINTR_D         USB0_USBINTR
#define USB0_USBINTR_D_bit     USB0_USBINTR_bit
#define USB0_USBINTR_H         USB0_USBINTR
#define USB0_USBINTR_H_bit     USB0_USBINTR_bit
__IO_REG32_BIT(USB0_FRINDEX,          0x4000614C,__READ_WRITE,__usb_frindex_bits);
#define USB0_FRINDEX_D         USB0_FRINDEX
#define USB0_FRINDEX_D_bit     USB0_FRINDEX_bit
#define USB0_FRINDEX_H         USB0_FRINDEX
#define USB0_FRINDEX_H_bit     USB0_FRINDEX_bit
__IO_REG32_BIT(USB0_PERIODICLISTBASE, 0x40006154,__READ_WRITE,__usb_periodiclistbase_reg_bits);
#define USB0_DEVICEADDR             USB0_PERIODICLISTBASE
#define USB0_DEVICEADDR_bit         USB0_PERIODICLISTBASE_bit
__IO_REG32_BIT(USB0_ASYNCLISTADDR,    0x40006158,__READ_WRITE,__usb_asynclistaddr_reg_bits);
#define USB0_ENDPOINTLISTADDR       USB0_ASYNCLISTADDR
#define USB0_ENDPOINTLISTADDR_bit   USB0_ASYNCLISTADDR_bit
__IO_REG32_BIT(USB0_TTCTRL,           0x4000615C,__READ_WRITE,__usb_ttctrl_reg_bits);
__IO_REG32_BIT(USB0_BURSTSIZE,        0x40006160,__READ_WRITE,__usb_burstsize_reg_bits);
__IO_REG32_BIT(USB0_TXFILLTUNING,     0x40006164,__READ_WRITE,__usb_txfilltuning_reg_bits);
__IO_REG32_BIT(USB0_BINTERVAL,        0x40006174,__READ_WRITE,__usb_binterval_reg_bits);
__IO_REG32_BIT(USB0_ENDPTNAK,         0x40006178,__READ_WRITE,__usb_endptnak_reg_bits);
__IO_REG32_BIT(USB0_ENDPTNAKEN,       0x4000617C,__READ_WRITE,__usb_endptnaken_reg_bits);
__IO_REG32_BIT(USB0_PORTSC1,          0x40006184,__READ_WRITE,__usb_portsc1_reg_bits);
#define USB0_PORTSC1_D         USB0_PORTSC1
#define USB0_PORTSC1_D_bit     USB0_PORTSC1_bit
#define USB0_PORTSC1_H         USB0_PORTSC1
#define USB0_PORTSC1_H_bit     USB0_PORTSC1_bit
__IO_REG32_BIT(USB0_OTGSC,            0x400061A4,__READ_WRITE,__usb_otgsc_reg_bits);
__IO_REG32_BIT(USB0_USBMODE,          0x400061A8,__READ_WRITE,__usb_usbmode_reg_bits);
#define USB0_USBMODE_D         USB0_USBMODE
#define USB0_USBMODE_D_bit     USB0_USBMODE_bit
#define USB0_USBMODE_H         USB0_USBMODE
#define USB0_USBMODE_H_bit     USB0_USBMODE_bit
__IO_REG32_BIT(USB0_ENDPTSETUPSTAT,   0x400061AC,__READ_WRITE,__usb_endptsetupstat_reg_bits);
__IO_REG32_BIT(USB0_ENDPTPRIME,       0x400061B0,__READ_WRITE,__usb_endptprime_reg_bits);
__IO_REG32_BIT(USB0_ENDPTFLUSH,       0x400061B4,__READ_WRITE,__usb_endptflush_reg_bits);
__IO_REG32_BIT(USB0_ENDPTSTAT,        0x400061B8,__READ_WRITE,__usb_endptstatus_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCOMPLETE,    0x400061BC,__READ_WRITE,__usb_endptcomplete_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCTRL0,       0x400061C0,__READ_WRITE,__usb_endptctrl0_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCTRL1,       0x400061C4,__READ_WRITE,__usb_endptctrl_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCTRL2,       0x400061C8,__READ_WRITE,__usb_endptctrl_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCTRL3,       0x400061CC,__READ_WRITE,__usb_endptctrl_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCTRL4,       0x400061D0,__READ_WRITE,__usb_endptctrl_reg_bits);
__IO_REG32_BIT(USB0_ENDPTCTRL5,       0x400061D4,__READ_WRITE,__usb_endptctrl_reg_bits);

/***************************************************************************
 **
 ** USB1
 **
 ***************************************************************************/
__IO_REG32_BIT(USB1_CAPLENGTH,        0x40007100,__READ      ,__usb_caplength_reg_bits);
__IO_REG32_BIT(USB1_HCSPARAMS,        0x40007104,__READ      ,__usb_hcsparams_reg_bits);
__IO_REG32_BIT(USB1_HCCPARAMS,        0x40007108,__READ      ,__usb_hccparams_reg_bits);
__IO_REG32_BIT(USB1_DCIVERSION,       0x40007120,__READ      ,__usb_dciversion_reg_bits);
__IO_REG32_BIT(USB1_DCCPARAMS,        0x40007124,__READ      ,__usb_dccparams_reg_bits);
__IO_REG32_BIT(USB1_USBCMD,           0x40007140,__READ_WRITE,__usb_usbcmd_reg_bits);
#define USB1_USBCMD_D         USB0_USBCMD
#define USB1_USBCMD_D_bit     USB0_USBCMD_bit
#define USB1_USBCMD_H         USB0_USBCMD
#define USB1_USBCMD_H_bit     USB0_USBCMD_bit
__IO_REG32_BIT(USB1_USBSTS,           0x40007144,__READ_WRITE,__usb_usbsts_reg_bits);
#define USB1_USBSTS_D         USB0_USBSTS
#define USB1_USBSTS_D_bit     USB0_USBSTS_bit
#define USB1_USBSTS_H         USB0_USBSTS
#define USB1_USBSTS_H_bit     USB0_USBSTS_bit
__IO_REG32_BIT(USB1_USBINTR,          0x40007148,__READ_WRITE,__usb_usbintr_reg_bits);
#define USB1_USBINTR_D         USB0_USBINTR
#define USB1_USBINTR_D_bit     USB0_USBINTR_bit
#define USB1_USBINTR_H         USB0_USBINTR
#define USB1_USBINTR_H_bit     USB0_USBINTR_bit
__IO_REG32(    USB1_FRINDEX,          0x4000714C,__READ_WRITE);
#define USB1_FRINDEX_D         USB0_FRINDEX
#define USB1_FRINDEX_D_bit     USB0_FRINDEX_bit
#define USB1_FRINDEX_H         USB0_FRINDEX
#define USB1_FRINDEX_H_bit     USB0_FRINDEX_bit
__IO_REG32_BIT(USB1_PERIODICLISTBASE, 0x40007154,__READ_WRITE,__usb_periodiclistbase_reg_bits);
#define USB1_DEVICEADDR          USB1_PERIODICLISTBASE
#define USB1_DEVICEADDR_bit      USB1_PERIODICLISTBASE_bit
__IO_REG32_BIT(USB1_ASYNCLISTADDR,    0x40007158,__READ_WRITE,__usb_asynclistaddr_reg_bits);
#define USB1_ENDPOINTLISTADDR       USB1_ASYNCLISTADDR
#define USB1_ENDPOINTLISTADDR_bit   USB1_ASYNCLISTADDR_bit
__IO_REG32_BIT(USB1_TTCTRL,           0x4000715C,__READ_WRITE,__usb_ttctrl_reg_bits);
__IO_REG32_BIT(USB1_BURSTSIZE,        0x40007160,__READ_WRITE,__usb_burstsize_reg_bits);
__IO_REG32_BIT(USB1_TXFILLTUNING,     0x40007164,__READ_WRITE,__usb_txfilltuning_reg_bits);
__IO_REG32_BIT(USB1_ULPIVIEWPORT,     0x40007170,__READ_WRITE,__usb1_ulpiviewport_reg_bits);
__IO_REG32_BIT(USB1_BINTERVAL,        0x40007174,__READ_WRITE,__usb_binterval_reg_bits);
__IO_REG32_BIT(USB1_ENDPTNAK,         0x40007178,__READ_WRITE,__usb1_endptnak_reg_bits);
__IO_REG32_BIT(USB1_ENDPTNAKEN,       0x4000717C,__READ_WRITE,__usb1_endptnaken_reg_bits);
__IO_REG32_BIT(USB1_PORTSC1,          0x40007184,__READ_WRITE,__usb_portsc1_reg_bits);
#define USB1_PORTSC1_D         USB0_PORTSC1
#define USB1_PORTSC1_D_bit     USB0_PORTSC1_bit
#define USB1_PORTSC1_H         USB0_PORTSC1
#define USB1_PORTSC1_H_bit     USB0_PORTSC1_bit
__IO_REG32_BIT(USB1_USBMODE,          0x400071A8,__READ_WRITE,__usb_usbmode_reg_bits);
#define USB1_USBMODE_D         USB0_USBMODE
#define USB1_USBMODE_D_bit     USB0_USBMODE_bit
#define USB1_USBMODE_H         USB0_USBMODE
#define USB1_USBMODE_H_bit     USB0_USBMODE_bit
__IO_REG32_BIT(USB1_ENDPTSETUPSTAT,   0x400071AC,__READ_WRITE,__usb1_endptsetupstat_reg_bits);
__IO_REG32_BIT(USB1_ENDPTPRIME,       0x400071B0,__READ_WRITE,__usb1_endptprime_reg_bits);
__IO_REG32_BIT(USB1_ENDPTFLUSH,       0x400071B4,__READ_WRITE,__usb1_endptflush_reg_bits);
__IO_REG32_BIT(USB1_ENDPTSTAT,        0x400071B8,__READ_WRITE,__usb1_endptstatus_reg_bits);
__IO_REG32_BIT(USB1_ENDPTCOMPLETE,    0x400071BC,__READ_WRITE,__usb1_endptcomplete_reg_bits);
__IO_REG32_BIT(USB1_ENDPTCTRL0,       0x400071C0,__READ_WRITE,__usb_endptctrl0_reg_bits);
__IO_REG32_BIT(USB1_ENDPTCTRL1,       0x400071C4,__READ_WRITE,__usb_endptctrl_reg_bits);
__IO_REG32_BIT(USB1_ENDPTCTRL2,       0x400071C8,__READ_WRITE,__usb_endptctrl_reg_bits);
__IO_REG32_BIT(USB1_ENDPTCTRL3,       0x400071CC,__READ_WRITE,__usb_endptctrl_reg_bits);

/***************************************************************************
 **
 ** Ethernet
 **
 ***************************************************************************/
__IO_REG32_BIT(ETH_MAC_CONFIG,            0x40010000,__READ_WRITE,__eth_mac_config_bits);
__IO_REG32_BIT(ETH_MAC_FRAME_FILTER,      0x40010004,__READ_WRITE,__eth_mac_frame_filter_bits);
__IO_REG32(    ETH_MAC_HASHTABLE_HIGH,    0x40010008,__READ_WRITE);
__IO_REG32(    ETH_MAC_HASHTABLE_LOW,     0x4001000C,__READ_WRITE);
__IO_REG32_BIT(ETH_MAC_MII_ADDR,          0x40010010,__READ_WRITE,__eth_mac_gmii_addr_bits);
__IO_REG32_BIT(ETH_MAC_MII_DATA,          0x40010014,__READ_WRITE,__eth_mac_gmii_data_bits);
__IO_REG32_BIT(ETH_MAC_FLOW_CTRL,         0x40010018,__READ_WRITE,__eth_mac_flow_ctrl_bits);
__IO_REG32_BIT(ETH_MAC_VLAN_TAG,          0x4001001C,__READ_WRITE,__eth_mac_vlan_tag_bits);
__IO_REG32_BIT(ETH_MAC_DEBUG,             0x40010024,__READ      ,__eth_mac_debug_bits);
__IO_REG32(    ETH_MAC_RWAKE_FRFLT,       0x40010028,__READ_WRITE);
__IO_REG32_BIT(ETH_MAC_PMT_CTRL_STAT,     0x4001002C,__READ_WRITE,__eth_mac_pmt_ctrl_stat_bits);
__IO_REG32(    ETH_MAC_INTR,              0x40010038,__READ      );
__IO_REG32_BIT(ETH_MAC_INTR_MASK,         0x4001003C,__READ_WRITE,__eth_mac_intr_mask_bits);
__IO_REG32_BIT(ETH_MAC_ADDR0_HIGH,        0x40010040,__READ_WRITE,__eth_mac_addr0_high_bits);
__IO_REG32(    ETH_MAC_ADDR0_LOW,         0x40010044,__READ_WRITE);
__IO_REG32_BIT(ETH_MAC_TIMESTP_CTRL,      0x40010700,__READ_WRITE,__eth_mac_timestp_ctrl_bits);

__IO_REG32_BIT(ETH_SUBSECOND_INCR,        0x40010704,__READ_WRITE,__eth_subsecond_incr_bits);
__IO_REG32_BIT(ETH_SECONDS,               0x40010708,__READ      ,__eth_seconds_bits);
__IO_REG32_BIT(ETH_NANOSECONDS,           0x4001070C,__READ      ,__eth_nanoseconds_bits);
__IO_REG32_BIT(ETH_SECONDSUPDATE,         0x40010710,__READ_WRITE,__eth_seconds_bits);
__IO_REG32_BIT(ETH_NANOSECONDSUPDATE,     0x40010714,__READ_WRITE,__eth_nanosecondsupdate_bits);
__IO_REG32_BIT(ETH_ADDEND,                0x40010718,__READ_WRITE,__eth_addend_bits);
__IO_REG32_BIT(ETH_TARGETSECONDS,         0x4001071C,__READ_WRITE,__eth_targetseconds_bits);
__IO_REG32_BIT(ETH_TARGETNANOSECONDS,     0x40010720,__READ_WRITE,__eth_targetnanoseconds_bits);
__IO_REG32_BIT(ETH_HIGHWORD,              0x40010724,__READ_WRITE,__eth_highword_bits);
__IO_REG32_BIT(ETH_TIMESTAMPSTAT,         0x40010728,__READ      ,__eth_timestampstat_bits);
__IO_REG32_BIT(ETH_PPSCTRL,               0x4001072C,__READ_WRITE,__eth_ppsctrl_bits);
__IO_REG32_BIT(ETH_AUXNANOSECONDS,        0x40010730,__READ      ,__eth_auxnanoseconds_bits);
__IO_REG32_BIT(ETH_AUXSECONDS,            0x40010734,__READ      ,__eth_auxseconds_bits);

__IO_REG32_BIT(ETH_DMA_BUS_MODE,          0x40011000,__READ_WRITE,__eth_dma_bus_mode_bits);
__IO_REG32(    ETH_DMA_TRANS_POLL_DEMAND, 0x40011004,__READ_WRITE);
__IO_REG32(    ETH_DMA_REC_POLL_DEMAND,   0x40011008,__READ_WRITE);
__IO_REG32(    ETH_DMA_REC_DES_ADDR,      0x4001100C,__READ_WRITE);
__IO_REG32(    ETH_DMA_TRANS_DES_ADDR,    0x40011010,__READ_WRITE);
__IO_REG32_BIT(ETH_DMA_STAT,              0x40011014,__READ_WRITE,__eth_dma_stat_bits);
__IO_REG32_BIT(ETH_DMA_OP_MODE,           0x40011018,__READ_WRITE,__eth_dma_op_mode_bits);
__IO_REG32_BIT(ETH_DMA_INT_EN,            0x4001101C,__READ_WRITE,__eth_dma_int_en_bits);
__IO_REG32_BIT(ETH_DMA_MFRM_BUFOF,        0x40011020,__READ      ,__eth_dma_mfrm_bufof_bits);
__IO_REG32_BIT(ETH_DMA_REC_INT_WDT,       0x40011024,__READ_WRITE,__eth_dma_rec_int_wdt_bits);
__IO_REG32(    ETH_DMA_CURHOST_TRANS_DES, 0x40011048,__READ      );
__IO_REG32(    ETH_DMA_CURHOST_REC_DES,   0x4001104C,__READ      );
__IO_REG32(    ETH_DMA_CURHOST_TRANS_BUF, 0x40011050,__READ      );
__IO_REG32(    ETH_DMA_CURHOST_REC_BUF,   0x40011054,__READ      );

/***************************************************************************
 **
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_TIMH,              0x40008000,__READ_WRITE ,__lcd_timh_bits);
__IO_REG32_BIT(LCD_TIMV,              0x40008004,__READ_WRITE ,__lcd_timv_bits);
__IO_REG32_BIT(LCD_POL,               0x40008008,__READ_WRITE ,__lcd_pol_bits);
__IO_REG32_BIT(LCD_LE,                0x4000800C,__READ_WRITE ,__lcd_le_bits);
__IO_REG32_BIT(LCD_UPBASE,            0x40008010,__READ_WRITE ,__lcd_upbase_bits);
__IO_REG32_BIT(LCD_LPBASE,            0x40008014,__READ_WRITE ,__lcd_lpbase_bits);
__IO_REG32_BIT(LCD_CTRL,              0x40008018,__READ_WRITE ,__lcd_ctrl_bits);
__IO_REG32_BIT(LCD_INTMSK,            0x4000801C,__READ_WRITE ,__lcd_intmsk_bits);
__IO_REG32_BIT(LCD_INTRAW,            0x40008020,__READ       ,__lcd_intraw_bits);
__IO_REG32_BIT(LCD_INTSTAT,           0x40008024,__READ       ,__lcd_intstat_bits);
__IO_REG32(    LCD_INTCLR,            0x40008028,__WRITE      );
__IO_REG32(    LCD_UPCURR,            0x4000802C,__READ       );
__IO_REG32(    LCD_LPCURR,            0x40008030,__READ       );
__IO_REG32_BIT(LCD_PAL,               0x40008200,__READ_WRITE ,__lcd_pal_bits);
__IO_REG32(    LCD_CRSR_IMG,          0x40008800,__READ_WRITE );
__IO_REG32_BIT(LCD_CRSR_CTRL,         0x40008C00,__READ_WRITE ,__lcd_crsr_ctrl_bits);
__IO_REG32_BIT(LCD_CRSR_CFG,          0x40008C04,__READ_WRITE ,__lcd_crsr_cfg_bits);
__IO_REG32_BIT(LCD_CRSR_PAL0,         0x40008C08,__READ_WRITE ,__lcd_crsr_pal0_bits);
__IO_REG32_BIT(LCD_CRSR_PAL1,         0x40008C0C,__READ_WRITE ,__lcd_crsr_pal1_bits);
__IO_REG32_BIT(LCD_CRSR_XY,           0x40008C10,__READ_WRITE ,__lcd_crsr_xy_bits);
__IO_REG32_BIT(LCD_CRSR_CLIP,         0x40008C14,__READ_WRITE ,__lcd_crsr_clip_bits);
__IO_REG32_BIT(LCD_CRSR_INTMSK,       0x40008C20,__READ_WRITE ,__lcd_crsr_intmsk_bits);
__IO_REG32(    LCD_CRSR_INTCLR,       0x40008C24,__WRITE      );
__IO_REG32_BIT(LCD_CRSR_INTRAW,       0x40008C28,__READ       ,__lcd_crsr_intraw_bits);
__IO_REG32_BIT(LCD_CRSR_INTSTAT,      0x40008C2C,__READ       ,__lcd_crsr_intstat_bits);

/***************************************************************************
 **
 ** SCT
 **
 ***************************************************************************/
__IO_REG32_BIT(SCT_CONFIG,             0x40000000,__READ_WRITE ,__sct_config_bits);
__IO_REG32_BIT(SCT_CTRL,               0x40000004,__READ_WRITE ,__sct_ctrl_bits);
#define SCT_CTRL_L       SCT_CTRL_bit.__sct_ctrl_l
#define SCT_CTRL_L_bit   SCT_CTRL_bit.__sct_ctrl_l_bits
#define SCT_CTRL_H       SCT_CTRL_bit.__sct_ctrl_h
#define SCT_CTRL_H_bit   SCT_CTRL_bit.__sct_ctrl_h_bits
__IO_REG32_BIT(SCT_LIMIT,              0x40000008,__READ_WRITE ,__sct_limit_bits);
#define SCT_LIMIT_L       SCT_LIMIT_bit.__sct_limit_l
#define SCT_LIMIT_L_bit   SCT_LIMIT_bit.__sct_limit_l_bits
#define SCT_LIMIT_H       SCT_LIMIT_bit.__sct_limit_h
#define SCT_LIMIT_H_bit   SCT_LIMIT_bit.__sct_limit_h_bits
__IO_REG32_BIT(SCT_HALT,               0x4000000C,__READ_WRITE ,__sct_halt_bits);
#define SCT_HALT_L       SCT_HALT_bit.__sct_halt_l
#define SCT_HALT_L_bit   SCT_HALT_bit.__sct_halt_l_bits
#define SCT_HALT_H       SCT_HALT_bit.__sct_halt_h
#define SCT_HALT_H_bit   SCT_HALT_bit.__sct_halt_h_bits
__IO_REG32_BIT(SCT_STOP,               0x40000010,__READ_WRITE ,__sct_stop_bits);
#define SCT_STOP_L       SCT_STOP_bit.__sct_stop_l
#define SCT_STOP_L_bit   SCT_STOP_bit.__sct_stop_l_bits
#define SCT_STOP_H       SCT_STOP_bit.__sct_stop_h
#define SCT_STOP_H_bit   SCT_STOP_bit.__sct_stop_h_bits
__IO_REG32_BIT(SCT_START,              0x40000014,__READ_WRITE ,__sct_start_bits);
#define SCT_START_L       SCT_START_bit.__sct_start_l
#define SCT_START_L_bit   SCT_START_bit.__sct_start_l_bits
#define SCT_START_H       SCT_START_bit.__sct_start_h
#define SCT_START_H_bit   SCT_START_bit.__sct_start_h_bits
__IO_REG32_BIT(SCT_COUNT,              0x40000040,__READ_WRITE ,__sct_count_bits);
#define SCT_COUNT_L       SCT_COUNT_bit.__sct_count_l
#define SCT_COUNT_H       SCT_COUNT_bit.__sct_count_h
__IO_REG32_BIT(SCT_STATE,              0x40000044,__READ_WRITE ,__sct_state_bits);
#define SCT_STATE_L       SCT_STATE_bit.__sct_state_l
#define SCT_STATE_L_bit   SCT_STATE_bit.__sct_state_l_bits
#define SCT_STATE_H       SCT_STATE_bit.__sct_state_h
#define SCT_STATE_H_bit   SCT_STATE_bit.__sct_state_h_bits
__IO_REG32_BIT(SCT_INPUT,              0x40000048,__READ       ,__sct_input_bits);
__IO_REG32_BIT(SCT_REGMODE,            0x4000004C,__READ_WRITE ,__sct_regmode_bits);
#define SCT_REGMODE_L       SCT_REGMODE_bit.__sct_regmode_l
#define SCT_REGMODE_L_bit   SCT_REGMODE_bit.__sct_regmode_l_bits
#define SCT_REGMODE_H       SCT_REGMODE_bit.__sct_regmode_h
#define SCT_REGMODE_H_bit   SCT_REGMODE_bit.__sct_regmode_h_bits
__IO_REG32_BIT(SCT_OUTPUT,             0x40000050,__READ_WRITE ,__sct_output_bits);
__IO_REG32_BIT(SCT_OUTPUTDIRCTRL,      0x40000054,__READ_WRITE ,__sct_outputdirctrl_bits);
__IO_REG32_BIT(SCT_RES,                0x40000058,__READ_WRITE ,__sct_res_bits);
__IO_REG32_BIT(SCT_DMAREQ0,            0x4000005C,__READ_WRITE ,__sct_dmareq0_bits);
__IO_REG32_BIT(SCT_DMAREQ1,            0x40000060,__READ_WRITE ,__sct_dmareq1_bits);
__IO_REG32_BIT(SCT_EVEN,               0x400000F0,__READ_WRITE ,__sct_even_bits);
__IO_REG32_BIT(SCT_EVFLAG,             0x400000F4,__READ_WRITE ,__sct_evflag_bits);
__IO_REG32_BIT(SCT_CONEN,              0x400000F8,__READ_WRITE ,__sct_conen_bits);
__IO_REG32_BIT(SCT_CONFLAG,            0x400000FC,__READ_WRITE ,__sct_conflag_bits);
__IO_REG32_BIT(SCT_MATCH0,             0x40000100,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP0             SCT_MATCH0
#define SCT_CAP0_bit         SCT_MATCH0_bit
#define SCT_MATCH0_L         SCT_MATCH0_bit.__sct_match_cap_l
#define SCT_MATCH0_H         SCT_MATCH0_bit.__sct_match_cap_h
#define SCT_CAP0_L           SCT_MATCH0_bit.__sct_match_cap_l
#define SCT_CAP0_H           SCT_MATCH0_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH1,             0x40000104,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP1             SCT_MATCH1
#define SCT_CAP1_bit         SCT_MATCH1_bit
#define SCT_MATCH1_L         SCT_MATCH1_bit.__sct_match_cap_l
#define SCT_MATCH1_H         SCT_MATCH1_bit.__sct_match_cap_h
#define SCT_CAP1_L           SCT_MATCH1_bit.__sct_match_cap_l
#define SCT_CAP1_H           SCT_MATCH1_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH2,             0x40000108,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP2             SCT_MATCH2
#define SCT_CAP2_bit         SCT_MATCH2_bit
#define SCT_MATCH2_L         SCT_MATCH2_bit.__sct_match_cap_l
#define SCT_MATCH2_H         SCT_MATCH2_bit.__sct_match_cap_h
#define SCT_CAP2_L           SCT_MATCH2_bit.__sct_match_cap_l
#define SCT_CAP2_H           SCT_MATCH2_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH3,             0x4000010C,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP3             SCT_MATCH3
#define SCT_CAP3_bit         SCT_MATCH3_bit
#define SCT_MATCH3_L         SCT_MATCH3_bit.__sct_match_cap_l
#define SCT_MATCH3_H         SCT_MATCH3_bit.__sct_match_cap_h
#define SCT_CAP3_L           SCT_MATCH3_bit.__sct_match_cap_l
#define SCT_CAP3_H           SCT_MATCH3_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH4,             0x40000110,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP4             SCT_MATCH4
#define SCT_CAP4_bit         SCT_MATCH4_bit
#define SCT_MATCH4_L         SCT_MATCH4_bit.__sct_match_cap_l
#define SCT_MATCH4_H         SCT_MATCH4_bit.__sct_match_cap_h
#define SCT_CAP4_L           SCT_MATCH4_bit.__sct_match_cap_l
#define SCT_CAP4_H           SCT_MATCH4_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH5,             0x40000114,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP5             SCT_MATCH5
#define SCT_CAP5_bit         SCT_MATCH5_bit
#define SCT_MATCH5_L         SCT_MATCH5_bit.__sct_match_cap_l
#define SCT_MATCH5_H         SCT_MATCH5_bit.__sct_match_cap_h
#define SCT_CAP5_L           SCT_MATCH5_bit.__sct_match_cap_l
#define SCT_CAP5_H           SCT_MATCH5_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH6,             0x40000118,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP6             SCT_MATCH6
#define SCT_CAP6_bit         SCT_MATCH6_bit
#define SCT_MATCH6_L         SCT_MATCH6_bit.__sct_match_cap_l
#define SCT_MATCH6_H         SCT_MATCH6_bit.__sct_match_cap_h
#define SCT_CAP6_L           SCT_MATCH6_bit.__sct_match_cap_l
#define SCT_CAP6_H           SCT_MATCH6_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH7,             0x4000011C,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP7             SCT_MATCH7
#define SCT_CAP7_bit         SCT_MATCH7_bit
#define SCT_MATCH7_L         SCT_MATCH7_bit.__sct_match_cap_l
#define SCT_MATCH7_H         SCT_MATCH7_bit.__sct_match_cap_h
#define SCT_CAP7_L           SCT_MATCH7_bit.__sct_match_cap_l
#define SCT_CAP7_H           SCT_MATCH7_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH8,             0x40000120,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP8             SCT_MATCH8
#define SCT_CAP8_bit         SCT_MATCH8_bit
#define SCT_MATCH8_L         SCT_MATCH8_bit.__sct_match_cap_l
#define SCT_MATCH8_H         SCT_MATCH8_bit.__sct_match_cap_h
#define SCT_CAP8_L           SCT_MATCH8_bit.__sct_match_cap_l
#define SCT_CAP8_H           SCT_MATCH8_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH9,             0x40000124,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP9             SCT_MATCH9
#define SCT_CAP9_bit         SCT_MATCH9_bit
#define SCT_MATCH9_L         SCT_MATCH9_bit.__sct_match_cap_l
#define SCT_MATCH9_H         SCT_MATCH9_bit.__sct_match_cap_h
#define SCT_CAP9_L           SCT_MATCH9_bit.__sct_match_cap_l
#define SCT_CAP9_H           SCT_MATCH9_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH10,            0x40000128,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP10            SCT_MATCH10
#define SCT_CAP10_bit        SCT_MATCH10_bit
#define SCT_MATCH10_L        SCT_MATCH10_bit.__sct_match_cap_l
#define SCT_MATCH10_H        SCT_MATCH10_bit.__sct_match_cap_h
#define SCT_CAP10_L          SCT_MATCH10_bit.__sct_match_cap_l
#define SCT_CAP10_H          SCT_MATCH10_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH11,            0x4000012C,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP11            SCT_MATCH11
#define SCT_CAP11_bit        SCT_MATCH11_bit
#define SCT_MATCH11_L        SCT_MATCH11_bit.__sct_match_cap_l
#define SCT_MATCH11_H        SCT_MATCH11_bit.__sct_match_cap_h
#define SCT_CAP11_L          SCT_MATCH11_bit.__sct_match_cap_l
#define SCT_CAP11_H          SCT_MATCH11_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH12,            0x40000130,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP12            SCT_MATCH12
#define SCT_CAP12_bit        SCT_MATCH12_bit
#define SCT_MATCH12_L        SCT_MATCH12_bit.__sct_match_cap_l
#define SCT_MATCH12_H        SCT_MATCH12_bit.__sct_match_cap_h
#define SCT_CAP12_L          SCT_MATCH12_bit.__sct_match_cap_l
#define SCT_CAP12_H          SCT_MATCH12_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH13,            0x40000134,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP13            SCT_MATCH13
#define SCT_CAP13_bit        SCT_MATCH13_bit
#define SCT_MATCH13_L        SCT_MATCH13_bit.__sct_match_cap_l
#define SCT_MATCH13_H        SCT_MATCH13_bit.__sct_match_cap_h
#define SCT_CAP13_L          SCT_MATCH13_bit.__sct_match_cap_l
#define SCT_CAP13_H          SCT_MATCH13_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH14,            0x40000138,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP14            SCT_MATCH14
#define SCT_CAP14_bit        SCT_MATCH14_bit
#define SCT_MATCH14_L        SCT_MATCH14_bit.__sct_match_cap_l
#define SCT_MATCH14_H        SCT_MATCH14_bit.__sct_match_cap_h
#define SCT_CAP14_L          SCT_MATCH14_bit.__sct_match_cap_l
#define SCT_CAP14_H          SCT_MATCH14_bit.__sct_match_cap_h
__IO_REG32_BIT(SCT_MATCH15,            0x4000013C,__READ_WRITE ,__sct_match_cap_bits);
#define SCT_CAP15            SCT_MATCH15
#define SCT_CAP15_bit        SCT_MATCH15_bit
#define SCT_MATCH15_L        SCT_MATCH15_bit.__sct_match_cap_l
#define SCT_MATCH15_H        SCT_MATCH15_bit.__sct_match_cap_h
#define SCT_CAP15_L          SCT_MATCH15_bit.__sct_match_cap_l
#define SCT_CAP15_H          SCT_MATCH15_bit.__sct_match_cap_h

__IO_REG32_BIT(SCT_MATCHREL0,          0x40000200,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL0         SCT_MATCHREL0
#define SCT_CAPCTRL0_bit     SCT_MATCHREL0_bit
#define SCT_MATCHREL0_L      SCT_MATCHREL0_bit.__sct_matchrel_l
#define SCT_MATCHREL0_H      SCT_MATCHREL0_bit.__sct_matchrel_h
#define SCT_CAPCTRL0_L       SCT_MATCHREL0_bit.__sct_capctrl_l
#define SCT_CAPCTRL0_L_bit   SCT_MATCHREL0_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL0_H       SCT_MATCHREL0_bit.__sct_capctrl_h
#define SCT_CAPCTRL0_H_bit   SCT_MATCHREL0_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL1,          0x40000204,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL1         SCT_MATCHREL1
#define SCT_CAPCTRL1_bit     SCT_MATCHREL1_bit
#define SCT_MATCHREL1_L      SCT_MATCHREL1_bit.__sct_matchrel_l
#define SCT_MATCHREL1_H      SCT_MATCHREL1_bit.__sct_matchrel_h
#define SCT_CAPCTRL1_L       SCT_MATCHREL1_bit.__sct_capctrl_l
#define SCT_CAPCTRL1_L_bit   SCT_MATCHREL1_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL1_H       SCT_MATCHREL1_bit.__sct_capctrl_h
#define SCT_CAPCTRL1_H_bit   SCT_MATCHREL1_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL2,          0x40000208,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL2         SCT_MATCHREL2
#define SCT_CAPCTRL2_bit     SCT_MATCHREL2_bit
#define SCT_MATCHREL2_L      SCT_MATCHREL2_bit.__sct_matchrel_l
#define SCT_MATCHREL2_H      SCT_MATCHREL2_bit.__sct_matchrel_h
#define SCT_CAPCTRL2_L       SCT_MATCHREL2_bit.__sct_capctrl_l
#define SCT_CAPCTRL2_L_bit   SCT_MATCHREL2_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL2_H       SCT_MATCHREL2_bit.__sct_capctrl_h
#define SCT_CAPCTRL2_H_bit   SCT_MATCHREL2_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL3,          0x4000020C,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL3         SCT_MATCHREL3
#define SCT_CAPCTRL3_bit     SCT_MATCHREL3_bit
#define SCT_MATCHREL3_L      SCT_MATCHREL3_bit.__sct_matchrel_l
#define SCT_MATCHREL3_H      SCT_MATCHREL3_bit.__sct_matchrel_h
#define SCT_CAPCTRL3_L       SCT_MATCHREL3_bit.__sct_capctrl_l
#define SCT_CAPCTRL3_L_bit   SCT_MATCHREL3_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL3_H       SCT_MATCHREL3_bit.__sct_capctrl_h
#define SCT_CAPCTRL3_H_bit   SCT_MATCHREL3_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL4,          0x40000210,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL4         SCT_MATCHREL4
#define SCT_CAPCTRL4_bit     SCT_MATCHREL4_bit
#define SCT_MATCHREL4_L      SCT_MATCHREL4_bit.__sct_matchrel_l
#define SCT_MATCHREL4_H      SCT_MATCHREL4_bit.__sct_matchrel_h
#define SCT_CAPCTRL4_L       SCT_MATCHREL4_bit.__sct_capctrl_l
#define SCT_CAPCTRL4_L_bit   SCT_MATCHREL4_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL4_H       SCT_MATCHREL4_bit.__sct_capctrl_h
#define SCT_CAPCTRL4_H_bit   SCT_MATCHREL4_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL5,          0x40000214,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL5         SCT_MATCHREL5
#define SCT_CAPCTRL5_bit     SCT_MATCHREL5_bit
#define SCT_MATCHREL5_L      SCT_MATCHREL5_bit.__sct_matchrel_l
#define SCT_MATCHREL5_H      SCT_MATCHREL5_bit.__sct_matchrel_h
#define SCT_CAPCTRL5_L       SCT_MATCHREL5_bit.__sct_capctrl_l
#define SCT_CAPCTRL5_L_bit   SCT_MATCHREL5_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL5_H       SCT_MATCHREL5_bit.__sct_capctrl_h
#define SCT_CAPCTRL5_H_bit   SCT_MATCHREL5_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL6,          0x40000218,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL6         SCT_MATCHREL6
#define SCT_CAPCTRL6_bit     SCT_MATCHREL6_bit
#define SCT_MATCHREL6_L      SCT_MATCHREL6_bit.__sct_matchrel_l
#define SCT_MATCHREL6_H      SCT_MATCHREL6_bit.__sct_matchrel_h
#define SCT_CAPCTRL6_L       SCT_MATCHREL6_bit.__sct_capctrl_l
#define SCT_CAPCTRL6_L_bit   SCT_MATCHREL6_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL6_H       SCT_MATCHREL6_bit.__sct_capctrl_h
#define SCT_CAPCTRL6_H_bit   SCT_MATCHREL6_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL7,          0x4000021C,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL7         SCT_MATCHREL7
#define SCT_CAPCTRL7_bit     SCT_MATCHREL7_bit
#define SCT_MATCHREL7_L      SCT_MATCHREL7_bit.__sct_matchrel_l
#define SCT_MATCHREL7_H      SCT_MATCHREL7_bit.__sct_matchrel_h
#define SCT_CAPCTRL7_L       SCT_MATCHREL7_bit.__sct_capctrl_l
#define SCT_CAPCTRL7_L_bit   SCT_MATCHREL7_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL7_H       SCT_MATCHREL7_bit.__sct_capctrl_h
#define SCT_CAPCTRL7_H_bit   SCT_MATCHREL7_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL8,          0x40000220,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL8         SCT_MATCHREL8
#define SCT_CAPCTRL8_bit     SCT_MATCHREL8_bit
#define SCT_MATCHREL8_L      SCT_MATCHREL8_bit.__sct_matchrel_l
#define SCT_MATCHREL8_H      SCT_MATCHREL8_bit.__sct_matchrel_h
#define SCT_CAPCTRL8_L       SCT_MATCHREL8_bit.__sct_capctrl_l
#define SCT_CAPCTRL8_L_bit   SCT_MATCHREL8_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL8_H       SCT_MATCHREL8_bit.__sct_capctrl_h
#define SCT_CAPCTRL8_H_bit   SCT_MATCHREL8_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL9,          0x40000224,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL9         SCT_MATCHREL9
#define SCT_CAPCTRL9_bit     SCT_MATCHREL9_bit
#define SCT_MATCHREL9_L      SCT_MATCHREL9_bit.__sct_matchrel_l
#define SCT_MATCHREL9_H      SCT_MATCHREL9_bit.__sct_matchrel_h
#define SCT_CAPCTRL9_L       SCT_MATCHREL9_bit.__sct_capctrl_l
#define SCT_CAPCTRL9_L_bit   SCT_MATCHREL9_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL9_H       SCT_MATCHREL9_bit.__sct_capctrl_h
#define SCT_CAPCTRL9_H_bit   SCT_MATCHREL9_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL10,         0x40000228,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL10         SCT_MATCHREL10
#define SCT_CAPCTRL10_bit     SCT_MATCHREL10_bit
#define SCT_MATCHREL10_L      SCT_MATCHREL10_bit.__sct_matchrel_l
#define SCT_MATCHREL10_H      SCT_MATCHREL10_bit.__sct_matchrel_h
#define SCT_CAPCTRL10_L       SCT_MATCHREL10_bit.__sct_capctrl_l
#define SCT_CAPCTRL10_L_bit   SCT_MATCHREL10_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL10_H       SCT_MATCHREL10_bit.__sct_capctrl_h
#define SCT_CAPCTRL10_H_bit   SCT_MATCHREL10_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL11,         0x4000022C,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL11         SCT_MATCHREL11
#define SCT_CAPCTRL11_bit     SCT_MATCHREL11_bit
#define SCT_MATCHREL11_L      SCT_MATCHREL11_bit.__sct_matchrel_l
#define SCT_MATCHREL11_H      SCT_MATCHREL11_bit.__sct_matchrel_h
#define SCT_CAPCTRL11_L       SCT_MATCHREL11_bit.__sct_capctrl_l
#define SCT_CAPCTRL11_L_bit   SCT_MATCHREL11_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL11_H       SCT_MATCHREL11_bit.__sct_capctrl_h
#define SCT_CAPCTRL11_H_bit   SCT_MATCHREL11_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL12,         0x40000230,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL12         SCT_MATCHREL12
#define SCT_CAPCTRL12_bit     SCT_MATCHREL12_bit
#define SCT_MATCHREL12_L      SCT_MATCHREL12_bit.__sct_matchrel_l
#define SCT_MATCHREL12_H      SCT_MATCHREL12_bit.__sct_matchrel_h
#define SCT_CAPCTRL12_L       SCT_MATCHREL12_bit.__sct_capctrl_l
#define SCT_CAPCTRL12_L_bit   SCT_MATCHREL12_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL12_H       SCT_MATCHREL12_bit.__sct_capctrl_h
#define SCT_CAPCTRL12_H_bit   SCT_MATCHREL12_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL13,         0x40000234,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL13         SCT_MATCHREL13
#define SCT_CAPCTRL13_bit     SCT_MATCHREL13_bit
#define SCT_MATCHREL13_L      SCT_MATCHREL13_bit.__sct_matchrel_l
#define SCT_MATCHREL13_H      SCT_MATCHREL13_bit.__sct_matchrel_h
#define SCT_CAPCTRL13_L       SCT_MATCHREL13_bit.__sct_capctrl_l
#define SCT_CAPCTRL13_L_bit   SCT_MATCHREL13_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL13_H       SCT_MATCHREL13_bit.__sct_capctrl_h
#define SCT_CAPCTRL13_H_bit   SCT_MATCHREL13_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL14,         0x40000238,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL14         SCT_MATCHREL14
#define SCT_CAPCTRL14_bit     SCT_MATCHREL14_bit
#define SCT_MATCHREL14_L      SCT_MATCHREL14_bit.__sct_matchrel_l
#define SCT_MATCHREL14_H      SCT_MATCHREL14_bit.__sct_matchrel_h
#define SCT_CAPCTRL14_L       SCT_MATCHREL14_bit.__sct_capctrl_l
#define SCT_CAPCTRL14_L_bit   SCT_MATCHREL14_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL14_H       SCT_MATCHREL14_bit.__sct_capctrl_h
#define SCT_CAPCTRL14_H_bit   SCT_MATCHREL14_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_MATCHREL15,         0x4000023C,__READ_WRITE ,__sct_matchrel_capctrl_bits);
#define SCT_CAPCTRL16         SCT_MATCHREL16
#define SCT_CAPCTRL16_bit     SCT_MATCHREL16_bit
#define SCT_MATCHREL16_L      SCT_MATCHREL16_bit.__sct_matchrel_l
#define SCT_MATCHREL16_H      SCT_MATCHREL16_bit.__sct_matchrel_h
#define SCT_CAPCTRL16_L       SCT_MATCHREL16_bit.__sct_capctrl_l
#define SCT_CAPCTRL16_L_bit   SCT_MATCHREL16_bit.__sct_capctrl_l_bits
#define SCT_CAPCTRL16_H       SCT_MATCHREL16_bit.__sct_capctrl_h
#define SCT_CAPCTRL16_H_bit   SCT_MATCHREL16_bit.__sct_capctrl_h_bits
__IO_REG32_BIT(SCT_EVSTATEMSK0,        0x40000300,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL0,            0x40000304,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK1,        0x40000308,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL1,            0x4000030C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK2,        0x40000310,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL2,            0x40000314,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK3,        0x40000318,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL3,            0x4000031C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK4,        0x40000320,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL4,            0x40000324,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK5,        0x40000328,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL5,            0x4000032C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK6,        0x40000330,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL6,            0x40000334,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK7,        0x40000338,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL7,            0x4000033C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK8,        0x40000340,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL8,            0x40000344,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK9,        0x40000348,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL9,            0x4000034C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK10,       0x40000350,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL10,           0x40000354,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK11,       0x40000358,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL11,           0x4000035C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK12,       0x40000360,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL12,           0x40000364,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK13,       0x40000368,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL13,           0x4000036C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK14,       0x40000370,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL14,           0x40000374,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_EVSTATEMSK15,       0x40000378,__READ_WRITE ,__sct_evstatemsk_bits);
__IO_REG32_BIT(SCT_EVCTRL15,           0x4000037C,__READ_WRITE ,__sct_evctrl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET0,         0x40000500,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL0,          0x40000504,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET1,         0x40000508,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL1,          0x4000050C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET2,         0x40000510,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL2,          0x40000514,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET3,         0x40000518,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL3,          0x4000051C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET4,         0x40000520,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL4,          0x40000524,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET5,         0x40000528,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL5,          0x4000052C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET6,         0x40000530,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL6,          0x40000534,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET7,         0x40000538,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL7,          0x4000053C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET8,         0x40000540,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL8,          0x40000544,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET9,         0x40000548,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL9,          0x4000054C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET10,        0x40000550,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL10,         0x40000554,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET11,        0x40000558,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL11,         0x4000055C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET12,        0x40000560,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL12,         0x40000564,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET13,        0x40000568,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL13,         0x4000056C,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET14,        0x40000570,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL14,         0x40000574,__READ_WRITE ,__sct_outputcl_bits);
__IO_REG32_BIT(SCT_OUTPUTSET15,        0x40000578,__READ_WRITE ,__sct_outputset_bits);
__IO_REG32_BIT(SCT_OUTPUTCL15,         0x4000057C,__READ_WRITE ,__sct_outputcl_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(T0IR,                  0x40084000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T0TCR,                 0x40084004,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    T0TC,                  0x40084008,__READ_WRITE);
__IO_REG32(    T0PR,                  0x4008400C,__READ_WRITE);
__IO_REG32(    T0PC,                  0x40084010,__READ_WRITE);
__IO_REG32_BIT(T0MCR,                 0x40084014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T0MR0,                 0x40084018,__READ_WRITE);
__IO_REG32(    T0MR1,                 0x4008401C,__READ_WRITE);
__IO_REG32(    T0MR2,                 0x40084020,__READ_WRITE);
__IO_REG32(    T0MR3,                 0x40084024,__READ_WRITE);
__IO_REG32_BIT(T0CCR,                 0x40084028,__READ_WRITE ,__tmr_tccr_bits);
__IO_REG32(    T0CR0,                 0x4008402C,__READ);
__IO_REG32(    T0CR1,                 0x40084030,__READ);
__IO_REG32(    T0CR2,                 0x40084034,__READ);
__IO_REG32(    T0CR3,                 0x40084038,__READ);
__IO_REG32_BIT(T0EMR,                 0x4008403C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T0CTCR,                0x40084070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(T1IR,                  0x40085000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T1TCR,                 0x40085004,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    T1TC,                  0x40085008,__READ_WRITE);
__IO_REG32(    T1PR,                  0x4008500C,__READ_WRITE);
__IO_REG32(    T1PC,                  0x40085010,__READ_WRITE);
__IO_REG32_BIT(T1MCR,                 0x40085014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T1MR0,                 0x40085018,__READ_WRITE);
__IO_REG32(    T1MR1,                 0x4008501C,__READ_WRITE);
__IO_REG32(    T1MR2,                 0x40085020,__READ_WRITE);
__IO_REG32(    T1MR3,                 0x40085024,__READ_WRITE);
__IO_REG32_BIT(T1CCR,                 0x40085028,__READ_WRITE ,__tmr_tccr_bits);
__IO_REG32(    T1CR0,                 0x4008502C,__READ);
__IO_REG32(    T1CR1,                 0x40085030,__READ);
__IO_REG32(    T1CR2,                 0x40085034,__READ);
__IO_REG32(    T1CR3,                 0x40085038,__READ);
__IO_REG32_BIT(T1EMR,                 0x4008503C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T1CTCR,                0x40085070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(T2IR,                  0x400C3000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T2TCR,                 0x400C3004,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    T2TC,                  0x400C3008,__READ_WRITE);
__IO_REG32(    T2PR,                  0x400C300C,__READ_WRITE);
__IO_REG32(    T2PC,                  0x400C3010,__READ_WRITE);
__IO_REG32_BIT(T2MCR,                 0x400C3014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T2MR0,                 0x400C3018,__READ_WRITE);
__IO_REG32(    T2MR1,                 0x400C301C,__READ_WRITE);
__IO_REG32(    T2MR2,                 0x400C3020,__READ_WRITE);
__IO_REG32(    T2MR3,                 0x400C3024,__READ_WRITE);
__IO_REG32_BIT(T2CCR,                 0x400C3028,__READ_WRITE ,__tmr_tccr_bits);
__IO_REG32(    T2CR0,                 0x400C302C,__READ);
__IO_REG32(    T2CR1,                 0x400C3030,__READ);
__IO_REG32(    T2CR2,                 0x400C3034,__READ);
__IO_REG32(    T2CR3,                 0x400C3038,__READ);
__IO_REG32_BIT(T2EMR,                 0x400C303C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T2CTCR,                0x400C3070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(T3IR,                  0x400C4000,__READ_WRITE ,__tmr_ir_bits);
__IO_REG32_BIT(T3TCR,                 0x400C4004,__READ_WRITE ,__tmr_tcr_bits);
__IO_REG32(    T3TC,                  0x400C4008,__READ_WRITE);
__IO_REG32(    T3PR,                  0x400C400C,__READ_WRITE);
__IO_REG32(    T3PC,                  0x400C4010,__READ_WRITE);
__IO_REG32_BIT(T3MCR,                 0x400C4014,__READ_WRITE ,__tmr_mcr_bits);
__IO_REG32(    T3MR0,                 0x400C4018,__READ_WRITE);
__IO_REG32(    T3MR1,                 0x400C401C,__READ_WRITE);
__IO_REG32(    T3MR2,                 0x400C4020,__READ_WRITE);
__IO_REG32(    T3MR3,                 0x400C4024,__READ_WRITE);
__IO_REG32_BIT(T3CCR,                 0x400C4028,__READ_WRITE ,__tmr_tccr_bits);
__IO_REG32(    T3CR0,                 0x400C402C,__READ);
__IO_REG32(    T3CR1,                 0x400C4030,__READ);
__IO_REG32(    T3CR2,                 0x400C4034,__READ);
__IO_REG32(    T3CR3,                 0x400C4038,__READ);
__IO_REG32_BIT(T3EMR,                 0x400C403C,__READ_WRITE ,__tmr_emr_bits);
__IO_REG32_BIT(T3CTCR,                0x400C4070,__READ_WRITE ,__tmr_ctcr_bits);

/***************************************************************************
 **
 ** Motor control PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(MC_CON,                 0x400A0000,__READ       ,__mc_con_bits);
__IO_REG32(    MC_CON_SET,             0x400A0004,__WRITE      );
__IO_REG32(    MC_CON_CLR,             0x400A0008,__WRITE      );
__IO_REG32_BIT(MC_CAPCON,              0x400A000C,__READ       ,__mc_capcon_bits);
__IO_REG32(    MC_CAPCON_SET,          0x400A0010,__WRITE      );
__IO_REG32(    MC_CAPCON_CLR,          0x400A0014,__WRITE      );
__IO_REG32(    MC_TC0,                 0x400A0018,__READ_WRITE );
__IO_REG32(    MC_TC1,                 0x400A001C,__READ_WRITE );
__IO_REG32(    MC_TC2,                 0x400A0020,__READ_WRITE );
__IO_REG32(    MC_LIM0,                0x400A0024,__READ_WRITE );
__IO_REG32(    MC_LIM1,                0x400A0028,__READ_WRITE );
__IO_REG32(    MC_LIM2,                0x400A002C,__READ_WRITE );
__IO_REG32(    MC_MAT0,                0x400A0030,__READ_WRITE );
__IO_REG32(    MC_MAT1,                0x400A0034,__READ_WRITE );
__IO_REG32(    MC_MAT2,                0x400A0038,__READ_WRITE );
__IO_REG32_BIT(MC_DT,                  0x400A003C,__READ_WRITE ,__mc_dt_bits);
__IO_REG32_BIT(MC_MCCP,                0x400A0040,__READ_WRITE ,__mc_ccp_bits);
__IO_REG32(    MC_CAP0,                0x400A0044,__READ       );
__IO_REG32(    MC_CAP1,                0x400A0048,__READ       );
__IO_REG32(    MC_CAP2,                0x400A004C,__READ       );
__IO_REG32_BIT(MC_INTEN,               0x400A0050,__READ       ,__mc_inten_bits);
__IO_REG32(    MC_INTEN_SET,           0x400A0054,__WRITE      );
__IO_REG32(    MC_INTEN_CLR,           0x400A0058,__WRITE      );
__IO_REG32_BIT(MC_CNTCON,              0x400A005C,__READ       ,__mc_cntcon_bits);
__IO_REG32(    MC_CNTCON_SET,          0x400A0060,__WRITE      );
__IO_REG32(    MC_CNTCON_CLR,          0x400A0064,__WRITE      );
__IO_REG32_BIT(MC_INTF,                0x400A0068,__READ       ,__mc_intf_bits);
__IO_REG32(    MC_INTF_SET,            0x400A006C,__WRITE      );
__IO_REG32(    MC_INTF_CLR,            0x400A0070,__WRITE      );
__IO_REG32(    MC_CAP_CLR,             0x400A0074,__WRITE      );

/***************************************************************************
 **
 ** Quadrature Encoder Interface
 **
 ***************************************************************************/
__IO_REG32(    QEI_CON,                0x400C6000,__WRITE      );
__IO_REG32_BIT(QEI_STAT,               0x400C6004,__READ       ,__qei_stat_bits);
__IO_REG32_BIT(QEI_CONF,               0x400C6008,__READ_WRITE ,__qei_conf_bits);
__IO_REG32(    QEI_POS,                0x400C600C,__READ       );
__IO_REG32(    QEI_MAXPSOS,            0x400C6010,__READ_WRITE );
__IO_REG32(    QEI_CMPOS0,             0x400C6014,__READ_WRITE );
__IO_REG32(    QEI_CMPOS1,             0x400C6018,__READ_WRITE );
__IO_REG32(    QEI_CMPOS2,             0x400C601C,__READ_WRITE );
__IO_REG32(    QEI_INXCNT,             0x400C6020,__READ       );
__IO_REG32(    QEI_INXCMP0,            0x400C6024,__READ_WRITE );
__IO_REG32(    QEI_LOAD,               0x400C6028,__READ_WRITE );
__IO_REG32(    QEI_TIME,               0x400C602C,__READ       );
__IO_REG32(    QEI_VEL,                0x400C6030,__READ       );
__IO_REG32(    QEI_CAP,                0x400C6034,__READ       );
__IO_REG32(    QEI_VELCOMP,            0x400C6038,__READ_WRITE );
__IO_REG32(    QEI_FILTERPHA,          0x400C603C,__READ_WRITE );
__IO_REG32(    QEI_FILTERPHB,          0x400C6040,__READ_WRITE );
__IO_REG32(    QEI_FILTERINX,          0x400C6044,__READ_WRITE );
__IO_REG32(    QEI_WINDOW,             0x400C6048,__READ_WRITE );
__IO_REG32(    QEI_INXCMP1,            0x400C604C,__READ_WRITE );
__IO_REG32(    QEI_INXCMP2,            0x400C6050,__READ_WRITE );
__IO_REG32(    QEI_IEC,                0x400C6FD8,__WRITE      );
__IO_REG32(    QEI_IES,                0x400C6FDC,__WRITE      );
__IO_REG32_BIT(QEI_INTSTAT,            0x400C6FE0,__READ       ,__qei_intstat_bits);
__IO_REG32_BIT(QEI_IE,                 0x400C6FE4,__READ       ,__qei_intstat_bits);
__IO_REG32(    QEI_CLR,                0x400C6FE8,__WRITE      );
__IO_REG32(    QEI_SET,                0x400C6FEC,__WRITE      );

/***************************************************************************
 **
 ** Repetitive Interrupt Timer
 **
 ***************************************************************************/
__IO_REG32(    RIT_COMPVAL,             0x400C0000,__READ_WRITE );
__IO_REG32(    RIT_MASK,                0x400C0004,__READ_WRITE );
__IO_REG32_BIT(RIT_CTRL,                0x400C0008,__READ_WRITE ,__rit_ctrl_bits);
__IO_REG32(    RIT_COUNTER,             0x400C000C,__READ_WRITE );

/***************************************************************************
 **
 ** Alarm timer
 **
 ***************************************************************************/
__IO_REG32_BIT(AT_DOWNCOUNTER,         0x40040000,__READ_WRITE ,__at_downcounter_bits);
__IO_REG32_BIT(AT_PRESET,              0x40040004,__READ_WRITE ,__at_preset_bits);
__IO_REG32(    AT_CLR_EN,              0x40040FD8,__WRITE      );
__IO_REG32(    AT_SET_EN,              0x40040FDC,__WRITE      );
__IO_REG32_BIT(AT_STATUS,              0x40040FE0,__READ       ,__at_status_bits);
__IO_REG32_BIT(AT_ENABLE,              0x40040FE4,__READ       ,__at_enable_bits);
__IO_REG32(    AT_CLR_STAT,            0x40040FE8,__WRITE      );
__IO_REG32(    AT_SET_STAT,            0x40040FEC,__WRITE      );

/***************************************************************************
 **
 ** Watchdog
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,                  0x40080000,__READ_WRITE ,__wwdt_mod_bits);
#define WWDT_MOD         WDMOD
#define WWDT_MOD_bit     WDMOD_bit
__IO_REG32_BIT(WDTC,                   0x40080004,__READ_WRITE ,__wwdt_tc_bits);
#define WWDT_TC          WDTC
#define WWDT_TC_bit      WDTC_bit
__IO_REG32(    WDFEED,                 0x40080008,__WRITE      );
#define WWDT_FEED        WDFEED
__IO_REG32_BIT(WDTV,                   0x4008000C,__READ       ,__wwdt_tv_bits);
#define WWDT_TV          WDTV
#define WWDT_TV_bit      WDTV_bit
__IO_REG32_BIT(WDWARNINT,              0x40080014,__READ_WRITE ,__wwdt_warnint_bits);
#define WWDT_WARNINT     WDWARNINT
#define WWDT_WARNINT_bit WDWARNINT_bit
__IO_REG32_BIT(WDWINDOW,               0x40080018,__READ_WRITE ,__wwdt_window_bits);
#define WWDT_WINDOW      WDWINDOW
#define WWDT_WINDOW_bit  WDWINDOW_bit

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RTC_ILR,                0x40046000,__WRITE      );
__IO_REG32_BIT(RTC_CCR,                0x40046008,__READ_WRITE ,__rtc_ccr_bits);
__IO_REG32_BIT(RTC_CIIR,               0x4004600C,__READ_WRITE ,__rtc_ciir_bits);
__IO_REG32_BIT(RTC_AMR,                0x40046010,__READ_WRITE ,__rtc_amr_bits);
__IO_REG32_BIT(RTC_CTIME0,             0x40046014,__READ       ,__rtc_ctime0_bits);
__IO_REG32_BIT(RTC_CTIME1,             0x40046018,__READ       ,__rtc_ctime1_bits);
__IO_REG32_BIT(RTC_CTIME2,             0x4004601C,__READ       ,__rtc_ctime2_bits);
__IO_REG32_BIT(RTC_SEC,                0x40046020,__READ_WRITE ,__rtc_sec_bits);
__IO_REG32_BIT(RTC_MIN,                0x40046024,__READ_WRITE ,__rtc_min_bits);
__IO_REG32_BIT(RTC_HRS,                0x40046028,__READ_WRITE ,__rtc_hour_bits);
__IO_REG32_BIT(RTC_DOM,                0x4004602C,__READ_WRITE ,__rtc_dom_bits);
__IO_REG32_BIT(RTC_DOW,                0x40046030,__READ_WRITE ,__rtc_dow_bits);
__IO_REG32_BIT(RTC_DOY,                0x40046034,__READ_WRITE ,__rtc_doy_bits);
__IO_REG32_BIT(RTC_MONTH,              0x40046038,__READ_WRITE ,__rtc_month_bits);
__IO_REG32_BIT(RTC_YEAR,               0x4004603C,__READ_WRITE ,__rtc_year_bits);
__IO_REG32_BIT(RTC_CALIBRATION,        0x40046040,__READ_WRITE ,__rtc_calibration_bits);
__IO_REG32_BIT(RTC_ASEC,               0x40046060,__READ_WRITE ,__rtc_sec_bits);
__IO_REG32_BIT(RTC_AMIN,               0x40046064,__READ_WRITE ,__rtc_min_bits);
__IO_REG32_BIT(RTC_AHRS,               0x40046068,__READ_WRITE ,__rtc_hour_bits);
__IO_REG32_BIT(RTC_ADOM,               0x4004606C,__READ_WRITE ,__rtc_dom_bits);
__IO_REG32_BIT(RTC_ADOW,               0x40046070,__READ_WRITE ,__rtc_dow_bits);
__IO_REG32_BIT(RTC_ADOY,               0x40046074,__READ_WRITE ,__rtc_doy_bits);
__IO_REG32_BIT(RTC_AMON,               0x40046078,__READ_WRITE ,__rtc_month_bits);
__IO_REG32_BIT(RTC_AYRS,               0x4004607C,__READ_WRITE ,__rtc_year_bits);
__IO_REG32(    REGFILE0,               0x40041000,__READ_WRITE );
__IO_REG32(    REGFILE1,               0x40041004,__READ_WRITE );
__IO_REG32(    REGFILE2,               0x40041008,__READ_WRITE );
__IO_REG32(    REGFILE3,               0x4004100C,__READ_WRITE );
__IO_REG32(    REGFILE4,               0x40041010,__READ_WRITE );
__IO_REG32(    REGFILE5,               0x40041014,__READ_WRITE );
__IO_REG32(    REGFILE6,               0x40041018,__READ_WRITE );
__IO_REG32(    REGFILE7,               0x4004101C,__READ_WRITE );
__IO_REG32(    REGFILE8,               0x40041020,__READ_WRITE );
__IO_REG32(    REGFILE9,               0x40041024,__READ_WRITE );
__IO_REG32(    REGFILE10,              0x40041028,__READ_WRITE );
__IO_REG32(    REGFILE11,              0x4004102C,__READ_WRITE );
__IO_REG32(    REGFILE12,              0x40041030,__READ_WRITE );
__IO_REG32(    REGFILE13,              0x40041034,__READ_WRITE );
__IO_REG32(    REGFILE14,              0x40041038,__READ_WRITE );
__IO_REG32(    REGFILE15,              0x4004103C,__READ_WRITE );
__IO_REG32(    REGFILE16,              0x40041040,__READ_WRITE );
__IO_REG32(    REGFILE17,              0x40041044,__READ_WRITE );
__IO_REG32(    REGFILE18,              0x40041048,__READ_WRITE );
__IO_REG32(    REGFILE19,              0x4004104C,__READ_WRITE );
__IO_REG32(    REGFILE20,              0x40041050,__READ_WRITE );
__IO_REG32(    REGFILE21,              0x40041054,__READ_WRITE );
__IO_REG32(    REGFILE22,              0x40041058,__READ_WRITE );
__IO_REG32(    REGFILE23,              0x4004105C,__READ_WRITE );
__IO_REG32(    REGFILE24,              0x40041060,__READ_WRITE );
__IO_REG32(    REGFILE25,              0x40041064,__READ_WRITE );
__IO_REG32(    REGFILE26,              0x40041068,__READ_WRITE );
__IO_REG32(    REGFILE27,              0x4004106C,__READ_WRITE );
__IO_REG32(    REGFILE28,              0x40041070,__READ_WRITE );
__IO_REG32(    REGFILE29,              0x40041074,__READ_WRITE );
__IO_REG32(    REGFILE30,              0x40041078,__READ_WRITE );
__IO_REG32(    REGFILE31,              0x4004107C,__READ_WRITE );
__IO_REG32(    REGFILE32,              0x40041080,__READ_WRITE );
__IO_REG32(    REGFILE33,              0x40041084,__READ_WRITE );
__IO_REG32(    REGFILE34,              0x40041088,__READ_WRITE );
__IO_REG32(    REGFILE35,              0x4004108C,__READ_WRITE );
__IO_REG32(    REGFILE36,              0x40041090,__READ_WRITE );
__IO_REG32(    REGFILE37,              0x40041094,__READ_WRITE );
__IO_REG32(    REGFILE38,              0x40041098,__READ_WRITE );
__IO_REG32(    REGFILE39,              0x4004109C,__READ_WRITE );
__IO_REG32(    REGFILE40,              0x400410A0,__READ_WRITE );
__IO_REG32(    REGFILE41,              0x400410A4,__READ_WRITE );
__IO_REG32(    REGFILE42,              0x400410A8,__READ_WRITE );
__IO_REG32(    REGFILE43,              0x400410AC,__READ_WRITE );
__IO_REG32(    REGFILE44,              0x400410B0,__READ_WRITE );
__IO_REG32(    REGFILE45,              0x400410B4,__READ_WRITE );
__IO_REG32(    REGFILE46,              0x400410B8,__READ_WRITE );
__IO_REG32(    REGFILE47,              0x400410BC,__READ_WRITE );
__IO_REG32(    REGFILE48,              0x400410C0,__READ_WRITE );
__IO_REG32(    REGFILE49,              0x400410C4,__READ_WRITE );
__IO_REG32(    REGFILE50,              0x400410C8,__READ_WRITE );
__IO_REG32(    REGFILE51,              0x400410CC,__READ_WRITE );
__IO_REG32(    REGFILE52,              0x400410D0,__READ_WRITE );
__IO_REG32(    REGFILE53,              0x400410D4,__READ_WRITE );
__IO_REG32(    REGFILE54,              0x400410D8,__READ_WRITE );
__IO_REG32(    REGFILE55,              0x400410DC,__READ_WRITE );
__IO_REG32(    REGFILE56,              0x400410E0,__READ_WRITE );
__IO_REG32(    REGFILE57,              0x400410E4,__READ_WRITE );
__IO_REG32(    REGFILE58,              0x400410E8,__READ_WRITE );
__IO_REG32(    REGFILE59,              0x400410EC,__READ_WRITE );
__IO_REG32(    REGFILE60,              0x400410F0,__READ_WRITE );
__IO_REG32(    REGFILE61,              0x400410F4,__READ_WRITE );
__IO_REG32(    REGFILE62,              0x400410F8,__READ_WRITE );
__IO_REG32(    REGFILE63,              0x400410FC,__READ_WRITE );

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
/* U0DLL, U0RBR and U0THR share the same address */
__IO_REG32_BIT(U0RBR,                 0x40081000,__READ_WRITE  ,__uart_rbr_bits);
#define U0THR       U0RBR
#define U0THR_bit   U0RBR_bit
#define U0DLL       U0RBR
#define U0DLL_bit   U0RBR_bit
/* U0DLM and U0IER share the same address */
__IO_REG32_BIT(U0IER,                 0x40081004,__READ_WRITE ,__uart_ier_bits);
#define U0DLM      U0IER
#define U0DLM_bit  U0IER_bit
/* U0FCR and U0IIR share the same address */
__IO_REG32_BIT(U0FCR,                 0x40081008,__READ_WRITE ,__uart_fcriir_bits);
#define U0IIR      U0FCR
#define U0IIR_bit  U0FCR_bit

__IO_REG32_BIT(U0LCR,                 0x4008100C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(U0LSR,                 0x40081014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(U0SCR,                 0x4008101C,__READ_WRITE ,__uart_scr_bits);
__IO_REG32_BIT(U0ACR,                 0x40081020,__READ_WRITE ,__uart_acr_bits);
__IO_REG32_BIT(U0FDR,                 0x40081028,__READ_WRITE ,__uart_fdr_bits);
__IO_REG32_BIT(U0OSR,                 0x4008102C,__READ_WRITE ,__uart_osr_bits);
__IO_REG32_BIT(U0HDEN,                0x40081040,__READ_WRITE ,__uart_hden_bits);
__IO_REG32_BIT(U0SCICTRL,             0x40081048,__READ_WRITE ,__uart_scictrl_bits);
__IO_REG32_BIT(U0RS485CTRL,           0x4008104C,__READ_WRITE ,__uart_rs485ctrl_bits);
__IO_REG32_BIT(U0RS485ADRMATCH,       0x40081050,__READ_WRITE ,__uart_rs485adrmatch_bits);
__IO_REG32_BIT(U0RS485DLY,            0x40081054,__READ_WRITE ,__uart_rs485dly_bits);
__IO_REG32_BIT(U0SYNCCTRL,            0x40081058,__READ_WRITE ,__uart_syncctrl_bits);
__IO_REG32_BIT(U0TER,                 0x4008105C,__READ_WRITE ,__uart_ter_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
/* U1DLL, U1RBR and U1THR share the same address */
__IO_REG32_BIT(U1RBR,                 0x40082000,__READ_WRITE  ,__uart_rbr_bits);
#define U1THR         U1RBR
#define U1THR_bit     U1RBR_bit
#define U1DLL         U1RBR
#define U1DLL_bit     U1RBR_bit

/* U1DLM and U1IER share the same address */
__IO_REG32_BIT(U1IER,                 0x40082004,__READ_WRITE ,__uart1_ier_bits);
#define U1DLM      U1IER
#define U1DLM_bit  U1IER_bit
/* U1FCR and U1IIR share the same address */
__IO_REG32_BIT(U1FCR,                 0x40082008,__READ_WRITE ,__uart_fcriir_bits);
#define U1IIR      U1FCR
#define U1IIR_bit  U1FCR_bit

__IO_REG32_BIT(U1LCR,                 0x4008200C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(U1MCR,                 0x40082010,__READ_WRITE ,__uart1_mcr_bits);
__IO_REG32_BIT(U1LSR,                 0x40082014,__READ       ,__uart1_lsr_bits);
__IO_REG32_BIT(U1MSR,                 0x40082018,__READ       ,__uart1_msr_bits);
__IO_REG32_BIT(U1SCR,                 0x4008201C,__READ_WRITE ,__uart1_scr_bits);
__IO_REG32_BIT(U1ACR,                 0x40082020,__READ_WRITE ,__uart_acr_bits);
__IO_REG32_BIT(U1FDR,                 0x40082028,__READ_WRITE ,__uart_fdr_bits);
__IO_REG32_BIT(U1TER,                 0x40082030,__READ_WRITE ,__uart1_ter_bits);
__IO_REG32_BIT(U1RS485CTRL,           0x4008204C,__READ_WRITE ,__uart1_rs485ctrl_bits);
__IO_REG32_BIT(U1RS485ADRMATCH,       0x40082050,__READ_WRITE ,__uart_rs485adrmatch_bits);
__IO_REG32_BIT(U1RS485DLY,            0x40082054,__READ_WRITE ,__uart_rs485dly_bits);
__IO_REG32_BIT(U1FIFOLVL,             0x40082058,__READ_WRITE ,__uart1_fifolvl_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
/* U2DLL, U2RBR and U2THR share the same address */
__IO_REG32_BIT(U2RBR,                 0x400C1000,__READ_WRITE  ,__uart_rbr_bits);
#define U2THR         U2RBR
#define U2THR_bit     U2RBR_bit
#define U2DLL         U2RBR
#define U2DLL_bit     U2RBR_bit

/* U2DLM and U2IER share the same address */
__IO_REG32_BIT(U2IER,                 0x400C1004,__READ_WRITE ,__uart_ier_bits);
#define U2DLM      U2IER
#define U2DLM_bit  U2IER_bit
/* U2FCR and U2IIR share the same address */
__IO_REG32_BIT(U2FCR,                 0x400C1008,__READ_WRITE ,__uart_fcriir_bits);
#define U2IIR      U2FCR
#define U2IIR_bit  U2FCR_bit

__IO_REG32_BIT(U2LCR,                 0x400C100C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(U2LSR,                 0x400C1014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(U2SCR,                 0x400C101C,__READ_WRITE ,__uart_scr_bits);
__IO_REG32_BIT(U2ACR,                 0x400C1020,__READ_WRITE ,__uart_acr_bits);
__IO_REG32_BIT(U2FDR,                 0x400C1028,__READ_WRITE ,__uart_fdr_bits);
__IO_REG32_BIT(U2OSR,                 0x400C102C,__READ_WRITE ,__uart_osr_bits);
__IO_REG32_BIT(U2HDEN,                0x400C1040,__READ_WRITE ,__uart_hden_bits);
__IO_REG32_BIT(U2SCICTRL,             0x400C1048,__READ_WRITE ,__uart_scictrl_bits);
__IO_REG32_BIT(U2RS485CTRL,           0x400C104C,__READ_WRITE ,__uart_rs485ctrl_bits);
__IO_REG32_BIT(U2ADRMATCH,            0x400C1050,__READ_WRITE ,__uart_rs485adrmatch_bits);
__IO_REG32_BIT(U2RS485DLY,            0x400C1054,__READ_WRITE ,__uart_rs485dly_bits);
__IO_REG32_BIT(U2SYNCCTRL,            0x400C1058,__READ_WRITE ,__uart_syncctrl_bits);
__IO_REG32_BIT(U2TER,                 0x400C105C,__READ_WRITE ,__uart_ter_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
/* U3DLL, U3RBR and U3THR share the same address */
__IO_REG32_BIT(U3RBR,                 0x400C2000,__READ_WRITE  ,__uart_rbr_bits);
#define U3THR         U3RBR
#define U3THR_bit     U3RBR_bit
#define U3DLL         U3RBR
#define U3DLL_bit     U3RBR_bit

/* U3DLM and U3IER share the same address */
__IO_REG32_BIT(U3IER,                 0x400C2004,__READ_WRITE ,__uart_ier_bits);
#define U3DLM      U3IER
#define U3DLM_bit  U3IER_bit
/* U3FCR and U3IIR share the same address */
__IO_REG32_BIT(U3FCR,                 0x400C2008,__READ_WRITE ,__uart_fcriir_bits);
#define U3IIR      U3FCR
#define U3IIR_bit  U3FCR_bit

__IO_REG32_BIT(U3LCR,                 0x400C200C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(U3LSR,                 0x400C2014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(U3SCR,                 0x400C201C,__READ_WRITE ,__uart_scr_bits);
__IO_REG32_BIT(U3ACR,                 0x400C2020,__READ_WRITE ,__uart_acr_bits);
__IO_REG32_BIT(U3ICR,                 0x400C2024,__READ_WRITE ,__uart_icr_bits);
__IO_REG32_BIT(U3FDR,                 0x400C2028,__READ_WRITE ,__uart_fdr_bits);
__IO_REG32_BIT(U3OSR,                 0x400C202C,__READ_WRITE ,__uart_osr_bits);
__IO_REG32_BIT(U3HDEN,                0x400C2040,__READ_WRITE ,__uart_hden_bits);
__IO_REG32_BIT(U3SCICTRL,             0x400C2048,__READ_WRITE ,__uart_scictrl_bits);
__IO_REG32_BIT(U3RS485CTRL,           0x400C204C,__READ_WRITE ,__uart_rs485ctrl_bits);
__IO_REG32_BIT(U3ADRMATCH,            0x400C2050,__READ_WRITE ,__uart_rs485adrmatch_bits);
__IO_REG32_BIT(U3RS485DLY,            0x400C2054,__READ_WRITE ,__uart_rs485dly_bits);
__IO_REG32_BIT(U3SYNCCTRL,            0x400C2058,__READ_WRITE ,__uart_syncctrl_bits);
__IO_REG32_BIT(U3TER,                 0x400C205C,__READ_WRITE ,__uart_ter_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0_CR0,               0x40083000,__READ_WRITE ,__ssp_cr0_bits);
__IO_REG32_BIT(SSP0_CR1,               0x40083004,__READ_WRITE ,__ssp_cr1_bits);
__IO_REG32_BIT(SSP0_DR,                0x40083008,__READ_WRITE ,__ssp_dr_bits);
__IO_REG32_BIT(SSP0_SR,                0x4008300C,__READ       ,__ssp_sr_bits);
__IO_REG32_BIT(SSP0_CPSR,              0x40083010,__READ_WRITE ,__ssp_cpsr_bits);
__IO_REG32_BIT(SSP0_IMSC,              0x40083014,__READ_WRITE ,__ssp_imsc_bits);
__IO_REG32_BIT(SSP0_RIS,               0x40083018,__READ       ,__ssp_ris_bits);
__IO_REG32_BIT(SSP0_MIS,               0x4008301C,__READ       ,__ssp_mis_bits);
__IO_REG32(    SSP0_ICR,               0x40083020,__WRITE      );
__IO_REG32_BIT(SSP0_DMACR,             0x40083024,__READ_WRITE ,__ssp_dmacr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1_CR0,               0x400C5000,__READ_WRITE ,__ssp_cr0_bits);
__IO_REG32_BIT(SSP1_CR1,               0x400C5004,__READ_WRITE ,__ssp_cr1_bits);
__IO_REG32_BIT(SSP1_DR,                0x400C5008,__READ_WRITE ,__ssp_dr_bits);
__IO_REG32_BIT(SSP1_SR,                0x400C500C,__READ       ,__ssp_sr_bits);
__IO_REG32_BIT(SSP1_CPSR,              0x400C5010,__READ_WRITE ,__ssp_cpsr_bits);
__IO_REG32_BIT(SSP1_IMSC,              0x400C5014,__READ_WRITE ,__ssp_imsc_bits);
__IO_REG32_BIT(SSP1_RIS,               0x400C5018,__READ       ,__ssp_ris_bits);
__IO_REG32_BIT(SSP1_MIS,               0x400C501C,__READ       ,__ssp_mis_bits);
__IO_REG32(    SSP1_ICR,               0x400C5020,__WRITE      );
__IO_REG32_BIT(SSP1_DMACR,             0x400C5024,__READ_WRITE ,__ssp_dmacr_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI_CR,                  0x40100000,__READ_WRITE ,__spi_cr_bits);
__IO_REG32_BIT(SPI_SR,                  0x40100004,__READ       ,__spi_sr_bits);
__IO_REG32_BIT(SPI_DR,                  0x40100008,__READ_WRITE ,__spi_dr_bits);
__IO_REG32_BIT(SPI_CCR,                 0x4010000C,__READ_WRITE ,__spi_ccr_bits);
__IO_REG32_BIT(SPI_TCR,                 0x40100010,__READ_WRITE ,__spi_tcr_bits);
__IO_REG32_BIT(SPI_TSR,                 0x40100014,__READ_WRITE ,__spi_tsr_bits);
__IO_REG32_BIT(SPI_INT,                 0x4010001C,__READ_WRITE ,__spi_int_bits);

/***************************************************************************
 **
 ** CCAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(CCAN0_CNTL,               0x400E2000,__READ_WRITE ,__can_cntl_bits);
__IO_REG32_BIT(CCAN0_STAT,               0x400E2004,__READ_WRITE ,__can_stat_bits);
__IO_REG32_BIT(CCAN0_EC,                 0x400E2008,__READ       ,__can_ec_bits);
__IO_REG32_BIT(CCAN0_BT,                 0x400E200C,__READ_WRITE ,__can_bt_bits);
__IO_REG32_BIT(CCAN0_INT,                0x400E2010,__READ       ,__can_int_bits);
__IO_REG32_BIT(CCAN0_TEST,               0x400E2014,__READ_WRITE ,__can_test_bits);
__IO_REG32_BIT(CCAN0_BRPE,               0x400E2018,__READ_WRITE ,__can_brpe_bits);
__IO_REG32_BIT(CCAN0_IF1_CMDREQ,         0x400E2020,__READ_WRITE ,__can_ifx_cmdreq_bits);
__IO_REG32_BIT(CCAN0_IF1_CMDMSK_W,       0x400E2024,__READ_WRITE ,__can_ifx_cmdmsk_bits);
#define CCAN0_IF1_CMDMSK_R      CCAN0_IF1_CMDMSK_W
#define CCAN0_IF1_CMDMSK_R_bit  CCAN0_IF1_CMDMSK_W_bit
__IO_REG32_BIT(CCAN0_IF1_MSK1,           0x400E2028,__READ_WRITE ,__can_ifx_msk1_bits);
__IO_REG32_BIT(CCAN0_IF1_MSK2,           0x400E202C,__READ_WRITE ,__can_ifx_msk2_bits);
__IO_REG32_BIT(CCAN0_IF1_ARB1,           0x400E2030,__READ_WRITE ,__can_ifx_arb1_bits);
__IO_REG32_BIT(CCAN0_IF1_ARB2,           0x400E2034,__READ_WRITE ,__can_ifx_arb2_bits);
__IO_REG32_BIT(CCAN0_IF1_MCTRL,          0x400E2038,__READ_WRITE ,__can_ifx_mctrl_bits);
__IO_REG32_BIT(CCAN0_IF1_DA1,            0x400E203C,__READ_WRITE ,__can_ifx_da1_bits);
__IO_REG32_BIT(CCAN0_IF1_DA2,            0x400E2040,__READ_WRITE ,__can_ifx_da2_bits);
__IO_REG32_BIT(CCAN0_IF1_DB1,            0x400E2044,__READ_WRITE ,__can_ifx_db1_bits);
__IO_REG32_BIT(CCAN0_IF1_DB2,            0x400E2048,__READ_WRITE ,__can_ifx_db2_bits);
__IO_REG32_BIT(CCAN0_IF2_CMDREQ,         0x400E2080,__READ_WRITE ,__can_ifx_cmdreq_bits);
__IO_REG32_BIT(CCAN0_IF2_CMDMSK_W,         0x400E2084,__READ_WRITE ,__can_ifx_cmdmsk_bits);
#define CCAN0_IF2_CMDMSK_R      CCAN0_IF2_CMDMSK_W
#define CCAN0_IF2_CMDMSK_R_bit  CCAN0_IF2_CMDMSK_W_bit
__IO_REG32_BIT(CCAN0_IF2_MSK1,           0x400E2088,__READ_WRITE ,__can_ifx_msk1_bits);
__IO_REG32_BIT(CCAN0_IF2_MSK2,           0x400E208C,__READ_WRITE ,__can_ifx_msk2_bits);
__IO_REG32_BIT(CCAN0_IF2_ARB1,           0x400E2090,__READ_WRITE ,__can_ifx_arb1_bits);
__IO_REG32_BIT(CCAN0_IF2_ARB2,           0x400E2094,__READ_WRITE ,__can_ifx_arb2_bits);
__IO_REG32_BIT(CCAN0_IF2_MCTRL,          0x400E2098,__READ_WRITE ,__can_ifx_mctrl_bits);
__IO_REG32_BIT(CCAN0_IF2_DA1,            0x400E209C,__READ_WRITE ,__can_ifx_da1_bits);
__IO_REG32_BIT(CCAN0_IF2_DA2,            0x400E20A0,__READ_WRITE ,__can_ifx_da2_bits);
__IO_REG32_BIT(CCAN0_IF2_DB1,            0x400E20A4,__READ_WRITE ,__can_ifx_db1_bits);
__IO_REG32_BIT(CCAN0_IF2_DB2,            0x400E20A8,__READ_WRITE ,__can_ifx_db2_bits);
__IO_REG32_BIT(CCAN0_TXREQ1,             0x400E2100,__READ       ,__can_txreq1_bits);
__IO_REG32_BIT(CCAN0_TXREQ2,             0x400E2104,__READ       ,__can_txreq2_bits);
__IO_REG32_BIT(CCAN0_ND1,                0x400E2120,__READ       ,__can_nd1_bits);
__IO_REG32_BIT(CCAN0_ND2,                0x400E2124,__READ       ,__can_nd2_bits);
__IO_REG32_BIT(CCAN0_IR1,                0x400E2140,__READ       ,__can_ir1_bits);
__IO_REG32_BIT(CCAN0_IR2,                0x400E2144,__READ       ,__can_ir2_bits);
__IO_REG32_BIT(CCAN0_MSGV1,              0x400E2160,__READ       ,__can_msgv1_bits);
__IO_REG32_BIT(CCAN0_MSGV2,              0x400E2164,__READ       ,__can_msgv2_bits);
__IO_REG32_BIT(CCAN0_CLKDIV,             0x400E2180,__READ_WRITE ,__can_clkdiv_bits);

/***************************************************************************
 **
 ** CCAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CCAN1_CNTL,               0x400A4000,__READ_WRITE ,__can_cntl_bits);
__IO_REG32_BIT(CCAN1_STAT,               0x400A4004,__READ_WRITE ,__can_stat_bits);
__IO_REG32_BIT(CCAN1_EC,                 0x400A4008,__READ       ,__can_ec_bits);
__IO_REG32_BIT(CCAN1_BT,                 0x400A400C,__READ_WRITE ,__can_bt_bits);
__IO_REG32_BIT(CCAN1_INT,                0x400A4010,__READ       ,__can_int_bits);
__IO_REG32_BIT(CCAN1_TEST,               0x400A4014,__READ_WRITE ,__can_test_bits);
__IO_REG32_BIT(CCAN1_BRPE,               0x400A4018,__READ_WRITE ,__can_brpe_bits);
__IO_REG32_BIT(CCAN1_IF1_CMDREQ,         0x400A4020,__READ_WRITE ,__can_ifx_cmdreq_bits);
__IO_REG32_BIT(CCAN1_IF1_CMDMSK_W,       0x400A4024,__READ_WRITE ,__can_ifx_cmdmsk_bits);
#define CCAN1_IF1_CMDMSK_R      CCAN1_IF1_CMDMSK_W
#define CCAN1_IF1_CMDMSK_R_bit  CCAN1_IF1_CMDMSK_W_bit
__IO_REG32_BIT(CCAN1_IF1_MSK1,           0x400A4028,__READ_WRITE ,__can_ifx_msk1_bits);
__IO_REG32_BIT(CCAN1_IF1_MSK2,           0x400A402C,__READ_WRITE ,__can_ifx_msk2_bits);
__IO_REG32_BIT(CCAN1_IF1_ARB1,           0x400A4030,__READ_WRITE ,__can_ifx_arb1_bits);
__IO_REG32_BIT(CCAN1_IF1_ARB2,           0x400A4034,__READ_WRITE ,__can_ifx_arb2_bits);
__IO_REG32_BIT(CCAN1_IF1_MCTRL,          0x400A4038,__READ_WRITE ,__can_ifx_mctrl_bits);
__IO_REG32_BIT(CCAN1_IF1_DA1,            0x400A403C,__READ_WRITE ,__can_ifx_da1_bits);
__IO_REG32_BIT(CCAN1_IF1_DA2,            0x400A4040,__READ_WRITE ,__can_ifx_da2_bits);
__IO_REG32_BIT(CCAN1_IF1_DB1,            0x400A4044,__READ_WRITE ,__can_ifx_db1_bits);
__IO_REG32_BIT(CCAN1_IF1_DB2,            0x400A4048,__READ_WRITE ,__can_ifx_db2_bits);
__IO_REG32_BIT(CCAN1_IF2_CMDREQ,         0x400A4080,__READ_WRITE ,__can_ifx_cmdreq_bits);
__IO_REG32_BIT(CCAN1_IF2_CMDMSK_W,         0x400A4084,__READ_WRITE ,__can_ifx_cmdmsk_bits);
#define CCAN1_IF2_CMDMSK_R      CCAN1_IF2_CMDMSK_W
#define CCAN1_IF2_CMDMSK_R_bit  CCAN1_IF2_CMDMSK_W_bit
__IO_REG32_BIT(CCAN1_IF2_MSK1,           0x400A4088,__READ_WRITE ,__can_ifx_msk1_bits);
__IO_REG32_BIT(CCAN1_IF2_MSK2,           0x400A408C,__READ_WRITE ,__can_ifx_msk2_bits);
__IO_REG32_BIT(CCAN1_IF2_ARB1,           0x400A4090,__READ_WRITE ,__can_ifx_arb1_bits);
__IO_REG32_BIT(CCAN1_IF2_ARB2,           0x400A4094,__READ_WRITE ,__can_ifx_arb2_bits);
__IO_REG32_BIT(CCAN1_IF2_MCTRL,          0x400A4098,__READ_WRITE ,__can_ifx_mctrl_bits);
__IO_REG32_BIT(CCAN1_IF2_DA1,            0x400A409C,__READ_WRITE ,__can_ifx_da1_bits);
__IO_REG32_BIT(CCAN1_IF2_DA2,            0x400A40A0,__READ_WRITE ,__can_ifx_da2_bits);
__IO_REG32_BIT(CCAN1_IF2_DB1,            0x400A40A4,__READ_WRITE ,__can_ifx_db1_bits);
__IO_REG32_BIT(CCAN1_IF2_DB2,            0x400A40A8,__READ_WRITE ,__can_ifx_db2_bits);
__IO_REG32_BIT(CCAN1_TXREQ1,             0x400A4100,__READ       ,__can_txreq1_bits);
__IO_REG32_BIT(CCAN1_TXREQ2,             0x400A4104,__READ       ,__can_txreq2_bits);
__IO_REG32_BIT(CCAN1_ND1,                0x400A4120,__READ       ,__can_nd1_bits);
__IO_REG32_BIT(CCAN1_ND2,                0x400A4124,__READ       ,__can_nd2_bits);
__IO_REG32_BIT(CCAN1_IR1,                0x400A4140,__READ       ,__can_ir1_bits);
__IO_REG32_BIT(CCAN1_IR2,                0x400A4144,__READ       ,__can_ir2_bits);
__IO_REG32_BIT(CCAN1_MSGV1,              0x400A4160,__READ       ,__can_msgv1_bits);
__IO_REG32_BIT(CCAN1_MSGV2,              0x400A4164,__READ       ,__can_msgv2_bits);
__IO_REG32_BIT(CCAN1_CLKDIV,             0x400A4180,__READ_WRITE ,__can_clkdiv_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_DAO,                0x400A2000,__READ_WRITE ,__i2s_dao_bits);
__IO_REG32_BIT(I2S_DAI,                0x400A2004,__READ_WRITE ,__i2s_dai_bits);
__IO_REG32(    I2S_TXFIFO,             0x400A2008,__WRITE);
__IO_REG32(    I2S_RXFIFO,             0x400A200C,__READ);
__IO_REG32_BIT(I2S_STATE,              0x400A2010,__READ       ,__i2s_state_bits);
__IO_REG32_BIT(I2S_DMA1,               0x400A2014,__READ_WRITE ,__i2s_dma1_bits);
__IO_REG32_BIT(I2S_DMA2,               0x400A2018,__READ_WRITE ,__i2s_dma2_bits);
__IO_REG32_BIT(I2S_IRQ,                0x400A201C,__READ_WRITE ,__i2s_irq_bits);
__IO_REG32_BIT(I2S_TXRATE,             0x400A2020,__READ_WRITE ,__i2s_rxtxrate_bits);
__IO_REG32_BIT(I2S_RXRATE,             0x400A2024,__READ_WRITE ,__i2s_rxtxrate_bits);
__IO_REG32_BIT(I2S_TXBITRATE,          0x400A2028,__READ_WRITE ,__i2s_txbitrate_bits);
__IO_REG32_BIT(I2S_RXBITRATE,          0x400A202C,__READ_WRITE ,__i2s_rxbitrate_bits);
__IO_REG32_BIT(I2S_TXMODE,             0x400A2030,__READ_WRITE ,__i2s_txmode_bits);
__IO_REG32_BIT(I2S_RXMODE,             0x400A2034,__READ_WRITE ,__i2s_rxmode_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_CONSET,            0x400A1000,__READ_WRITE ,__i2c_conset_bits);
__IO_REG32_BIT(I2C0_STAT,              0x400A1004,__READ       ,__i2c_stat_bits);
__IO_REG32_BIT(I2C0_DAT,               0x400A1008,__READ_WRITE ,__i2c_dat_bits);
__IO_REG32_BIT(I2C0_ADR0,              0x400A100C,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C0_SCLH,              0x400A1010,__READ_WRITE ,__i2c_sclh_bits);
__IO_REG32_BIT(I2C0_SCLL,              0x400A1014,__READ_WRITE ,__i2c_scll_bits);
__IO_REG32(    I2C0_CONCLR,            0x400A1018,__WRITE      );
__IO_REG32_BIT(I2C0_MMCTRL,            0x400A101C,__READ_WRITE ,__i2c_mmctrl_bits);
__IO_REG32_BIT(I2C0_ADR1,              0x400A1020,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C0_ADR2,              0x400A1024,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C0_ADR3,              0x400A1028,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C0_DATA_BUFFER,       0x400A102C,__READ       ,__i2c_dat_bits);
__IO_REG32_BIT(I2C0_MASK0,             0x400A1030,__READ_WRITE ,__i2c_mask_bits);
__IO_REG32_BIT(I2C0_MASK1,             0x400A1034,__READ_WRITE ,__i2c_mask_bits);
__IO_REG32_BIT(I2C0_MASK2,             0x400A1038,__READ_WRITE ,__i2c_mask_bits);
__IO_REG32_BIT(I2C0_MASK3,             0x400A103C,__READ_WRITE ,__i2c_mask_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_CONSET,            0x400E0000,__READ_WRITE ,__i2c_conset_bits);
__IO_REG32_BIT(I2C1_STAT,              0x400E0004,__READ       ,__i2c_stat_bits);
__IO_REG32_BIT(I2C1_DAT,               0x400E0008,__READ_WRITE ,__i2c_dat_bits);
__IO_REG32_BIT(I2C1_ADR0,              0x400E000C,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C1_SCLH,              0x400E0010,__READ_WRITE ,__i2c_sclh_bits);
__IO_REG32_BIT(I2C1_SCLL,              0x400E0014,__READ_WRITE ,__i2c_scll_bits);
__IO_REG32(    I2C1_CONCLR,            0x400E0018,__WRITE      );
__IO_REG32_BIT(I2C1_MMCTRL,            0x400E001C,__READ_WRITE ,__i2c_mmctrl_bits);
__IO_REG32_BIT(I2C1_ADR1,              0x400E0020,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C1_ADR2,              0x400E0024,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C1_ADR3,              0x400E0028,__READ_WRITE ,__i2c_adr_bits);
__IO_REG32_BIT(I2C1_DATA_BUFFER,       0x400E002C,__READ       ,__i2c_dat_bits);
__IO_REG32_BIT(I2C1_MASK0,             0x400E0030,__READ_WRITE ,__i2c_mask_bits);
__IO_REG32_BIT(I2C1_MASK1,             0x400E0034,__READ_WRITE ,__i2c_mask_bits);
__IO_REG32_BIT(I2C1_MASK2,             0x400E0038,__READ_WRITE ,__i2c_mask_bits);
__IO_REG32_BIT(I2C1_MASK3,             0x400E003C,__READ_WRITE ,__i2c_mask_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC0_CR,                 0x400E3000,__READ_WRITE ,__adc_cr_bits);
__IO_REG32_BIT(ADC0_GDR,                0x400E3004,__READ_WRITE ,__adc_gdr_bits);
__IO_REG32_BIT(ADC0_INTEN,              0x400E300C,__READ_WRITE ,__adc_inten_bits);
__IO_REG32_BIT(ADC0_DR0,                0x400E3010,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR1,                0x400E3014,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR2,                0x400E3018,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR3,                0x400E301C,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR4,                0x400E3020,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR5,                0x400E3024,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR6,                0x400E3028,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_DR7,                0x400E302C,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC0_STAT,               0x400E3030,__READ       ,__adc_stat_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1_CR,                 0x400E4000,__READ_WRITE ,__adc_cr_bits);
__IO_REG32_BIT(ADC1_GDR,                0x400E4004,__READ_WRITE ,__adc_gdr_bits);
__IO_REG32_BIT(ADC1_INTEN,              0x400E400C,__READ_WRITE ,__adc_inten_bits);
__IO_REG32_BIT(ADC1_DR0,                0x400E4010,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR1,                0x400E4014,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR2,                0x400E4018,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR3,                0x400E401C,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR4,                0x400E4020,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR5,                0x400E4024,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR6,                0x400E4028,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_DR7,                0x400E402C,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC1_STAT,               0x400E4030,__READ       ,__adc_stat_bits);

/***************************************************************************
 **
 ** D/A Converter
 **
 ***************************************************************************/
__IO_REG32_BIT(DAC_CR,                  0x400E1000,__READ_WRITE ,__dac_cr_bits);
__IO_REG32_BIT(DAC_CTRL,                0x400E1004,__READ_WRITE ,__dac_ctrl_bits);
__IO_REG32_BIT(DAC_CNTVAL,              0x400E1008,__READ_WRITE ,__dac_cntval_bits);

/***************************************************************************
 **
 ** FMC
 **
 ***************************************************************************/
//__IO_REG32_BIT(FMC_FMSSTART,            0x4D084020,__READ_WRITE ,__fmc_fmsstart_bits);
//__IO_REG32_BIT(FMC_FMSSTOP,             0x4D084024,__READ_WRITE ,__fmc_fmsstop_bits);
//__IO_REG32_BIT(FMC_FMSW0,               0x4D08402C,__READ       ,__fmc_fmsw0_bits);
//__IO_REG32_BIT(FMC_FMSW1,               0x4D084030,__READ       ,__fmc_fmsw1_bits);
//__IO_REG32_BIT(FMC_FMSW2,               0x4D084034,__READ       ,__fmc_fmsw2_bits);
//__IO_REG32_BIT(FMC_FMSW3,               0x4D084038,__READ       ,__fmc_fmsw3_bits);
//__IO_REG32_BIT(FMC_FMSTAT,              0x4D084FE0,__READ       ,__fmc_fmsw0_bits);
//__IO_REG32(    FMC_FMSTATCLR,           0x4D084FE8,__WRITE      );


/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  GPDMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define GPDMA_SPIFI             0   /* SPIFI                                   */
#define GPDMA_SCTMATCH2         0   /* SCT Match 2                             */
#define GPDMA_SGPIO14           0   /* SGPIO14                                 */
#define GPDMA_T0MATCH0          1   /* Timer 0 match 0                         */
#define GPDMA_UART0TX           1   /* UART0 transmit                          */
#define GPDMA_AESIN             1   /* AES input                               */
#define GPDMA_T0MATCH1          2   /* Timer 0 match 1                         */
#define GPDMA_UART0RX           2   /* UART0 receive                           */
#define GPDMA_AESOUT            2   /* AES output                              */
#define GPDMA_T1MATCH0          3   /* Timer 1 match 0                         */
#define GPDMA_UART1TX           3   /* UART1 transmit                          */
#define GPDMA_I2S1DMA0          3   /* I2S1 DMA req 0                          */
#define GPDMA_T1MATCH1          4   /* Timer 1 match 1                         */
#define GPDMA_UART1RX           4   /* UART1 receive                           */
#define GPDMA_I2S1DMA1          4   /* I2S1 DMA req 1                          */
#define GPDMA_T2MATCH0          5   /* Timer 2 match 0                         */
#define GPDMA_UART2TX           5   /* UART2 transmit                          */
#define GPDMA_T2MATCH1          6   /* Timer 2 match 1                         */
#define GPDMA_UART2RX           6   /* UART2 receive                           */
#define GPDMA_T3MATCH0          7   /* Timer 3 match 0                         */
#define GPDMA_UART3TX           7   /* UART3 transmit                          */
#define GPDMA_SCTDMA0           7   /* SCT DMA request 0                       */
#define GPDMA_VADCWR            7   /* VADC write                              */
#define GPDMA_T3MATCH1          8   /* Timer 3 match 1                         */
#define GPDMA_UART3RX           8   /* UART3 receive                           */
#define GPDMA_VADCRD            8   /* VADC read                               */
#define GPDMA_SSP0RX            9   /* SSP0 receive                            */
#define GPDMA_I2S0DMA0          9   /* I2S DMA request 0                       */
#define GPDMA_SCTDMA1           9   /* SCT DMA request 1                       */
#define GPDMA_SSP0TX           10   /* SSP0 transmit                           */
#define GPDMA_I2S0DMA1         10   /* I2S channel 1                           */
#define GPDMA_SCTMATCH0        10   /* SCT match 0                             */
#define GPDMA_SSP1RX           11   /* SSP1 receive                            */
#define GPDMA_SSP1TX           12   /* SSP1 transmit                           */
#define GPDMA_SGPIO15          12   /* SGPIO15                                 */
#define GPDMA_ADC0             13   /* ADC0                                    */
#define GPDMA_ADC1             14   /* ADC1                                    */
#define GPDMA_DAC              15   /* DAC                                     */

/***************************************************************************
 **
 **  NVIC M4 Interrupt channels
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
#define NVIC_DAC              16
#define NVIC_M0CORE           17
#define NVIC_DMA              18
#define NVIC_ETHERNET         21
#define NVIC_SDIO             22
#define NVIC_LCD              23
#define NVIC_USB0             24
#define NVIC_USB1             25
#define NVIC_SCT              26
#define NVIC_RITIMER          27
#define NVIC_TIMER0           28
#define NVIC_TIMER1           29
#define NVIC_TIMER2           30
#define NVIC_TIMER3           31
#define NVIC_MCPWM            32
#define NVIC_ADC0             33
#define NVIC_I2C0             34
#define NVIC_I2C1             35
#define NVIC_SPI              36
#define NVIC_ADC1             37
#define NVIC_SSP0             38
#define NVIC_SSP1             39
#define NVIC_USART0           40
#define NVIC_UART1            41
#define NVIC_USART2           42
#define NVIC_USART3           43
#define NVIC_I2S0             44
#define NVIC_I2S1             45
#define NVIC_SPIFI            46
#define NVIC_SGPIO            47
#define NVIC_PIN_INT0         48
#define NVIC_PIN_INT1         49
#define NVIC_PIN_INT2         50
#define NVIC_PIN_INT3         51
#define NVIC_PIN_INT4         52
#define NVIC_PIN_INT5         53
#define NVIC_PIN_INT6         54
#define NVIC_PIN_INT7         55
#define NVIC_GINT0            56
#define NVIC_GINT1            57
#define NVIC_EVENTROUTER      58
#define NVIC_C_CAN1           59
#define NVIC_ATIMER           62
#define NVIC_RTC              63
#define NVIC_WWDT             65
#define NVIC_C_CAN0           67
#define NVIC_QEI              68

/***************************************************************************
 **
 **  EVR Interrupt channels
 **
 ***************************************************************************/
#define EVR_WAKEUP0           0
#define EVR_WAKEUP1           1
#define EVR_WAKEUP2           2
#define EVR_WAKEUP3           3
#define EVR_ATIMER            4
#define EVR_RTC               5
#define EVR_BOD               6
#define EVR_WWDT              7
#define EVR_ETHERNET          8
#define EVR_USB0              9
#define EVR_USB1              10
#define EVR_SDIO              11
#define EVR_C_CAN             12
#define EVR_TIMER2            13
#define EVR_TIMER6            14
#define EVR_QEI               15
#define EVR_TIMER14           16
#define EVR_RESET             19

/***************************************************************************
 **
 **  Power Down Modes
 **
 ***************************************************************************/

#define PWR_STATE_DEEP_SLEEP      0x003000AA
#define PWR_STATE_POWER_DOWN      0x0030FCBA
#define PWR_STATE_DEEP_POWER_DOWN 0x003FFF7F

#endif    /* __IOLPC4357_M4_H */

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
Interrupt9   = DAC            0x40
Interrupt10  = M0CORE         0x44
Interrupt11  = DMA            0x48
Interrupt12  = ETHERNET       0x54
Interrupt13  = SDIO           0x58
Interrupt14  = LCD            0x5C
Interrupt15  = USB0           0x60
Interrupt16  = USB1           0x64
Interrupt17  = SCT            0x68
Interrupt18  = RITIMER        0x6C
Interrupt19  = TIMER0         0x70
Interrupt20  = TIMER1         0x74
Interrupt21  = TIMER2         0x78
Interrupt22  = TIMER3         0x7C
Interrupt23  = MCPWM          0x80
Interrupt24  = ADC0           0x84
Interrupt25  = I2C0           0x88
Interrupt26  = I2C1           0x8C
Interrupt27  = SPI            0x90
Interrupt28  = ADC1           0x94
Interrupt29  = SSP0           0x98
Interrupt30  = SSP1           0x9C
Interrupt31  = USART0         0xA0
Interrupt32  = UART1          0xA4
Interrupt33  = USART2         0xA8
Interrupt34  = USART3         0xAC
Interrupt35  = I2S0           0xB0
Interrupt36  = I2S1           0xB4
Interrupt37  = SPIFI          0xB8
Interrupt38  = SGPIO          0xBC
Interrupt39  = PIN_INT0       0xC0
Interrupt40  = PIN_INT1       0xC4
Interrupt41  = PIN_INT2       0xC8
Interrupt42  = PIN_INT3       0xCC
Interrupt43  = PIN_INT4       0xD0
Interrupt44  = PIN_INT5       0xD4
Interrupt45  = PIN_INT6       0xD8
Interrupt46  = PIN_INT7       0xDC
Interrupt47  = GINT0          0xE0
Interrupt48  = GINT1          0xE4
Interrupt49  = EVENTROUTER    0xE8
Interrupt50  = C_CAN1         0xEC
Interrupt51  = ATIMER         0xF8
Interrupt52  = RTC            0xFC
Interrupt53  = WWDT           0x104
Interrupt54  = C_CAN0         0x10C
Interrupt55  = QEI            0x110
###DDF-INTERRUPT-END###*/
