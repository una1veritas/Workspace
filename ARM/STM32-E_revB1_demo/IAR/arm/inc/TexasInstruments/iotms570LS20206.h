/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments TMS570LS20206
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems and Texas Instruments 2011
 **
 **    $Revision: 46492 $
 **
***************************************************************************/

#ifndef __IOTMS570LS20206_H
#define __IOTMS570LS20206_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMS570LS20206 SPECIAL FUNCTION REGISTERS
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

/* SYS Pin Control Register 1 (SYSPC1) */
typedef struct {
  __REG32 ECPCLKFUN       : 1;
  __REG32                 :31;
} __syspc1_bits;

/* SYS Pin Control Register 2 (SYSPC2) */
typedef struct {
  __REG32 ECPCLKDIR       : 1;
  __REG32                 :31;
} __syspc2_bits;

/* SYS Pin Control Register 3 (SYSPC3) */
typedef struct {
  __REG32 ECPCLKDIN       : 1;
  __REG32                 :31;
} __syspc3_bits;

/* SYS Pin Control Register 4 (SYSPC4) */
typedef struct {
  __REG32 ECPCLKDOUT      : 1;
  __REG32                 :31;
} __syspc4_bits;

/* SYS Pin Control Register 5 (SYSPC5) */
typedef struct {
  __REG32 ECPCLKSET       : 1;
  __REG32                 :31;
} __syspc5_bits;

/* SYS Pin Control Register 6 (SYSPC6) */
typedef struct {
  __REG32 ECPCLKCLR       : 1;
  __REG32                 :31;
} __syspc6_bits;

/* SYS Pin Control Register 7 (SYSPC7) */
typedef struct {
  __REG32 ECPCLKODE       : 1;
  __REG32                 :31;
} __syspc7_bits;

/* SYS Pin Control Register 8 (SYSPC8) */
typedef struct {
  __REG32 ECPCLKPUE       : 1;
  __REG32                 :31;
} __syspc8_bits;

/* SYS Pin Control Register 9 (SYSPC9) */
typedef struct {
  __REG32 ECPCLKPS        : 1;
  __REG32                 :31;
} __syspc9_bits;

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
  __REG32 BWADJ           : 9;
  __REG32                 : 1;
  __REG32 SPREADINGRATE   : 9;
  __REG32 FMENA           : 1;
} __pllctl2_bits;

/* Die Identification Register, Lower Word (DIEIDL) */
typedef struct {
  __REG32 X_WAFER_COORDINATE  : 8;
  __REG32 Y_WAFER_COORDINATE  : 8;
  __REG32 WAFER               : 6;
  __REG32 LOT                 :10;
} __dieidl_bits;

/* Die Identification Register, Upper Word (DIEIDH) */
typedef struct {
  __REG32 LOT                 :14;
  __REG32                     :18;
} __dieidh_bits;

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
  __REG32 ALTLIMPCLOCKENABLE  : 1;
  __REG32                     : 5;
} __clktest_bits;

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
  __REG32                     :13;
  __REG32 RAM_DFT_EN          : 4;
  __REG32                     :12;
} __ramgcr_bits;

/* Bus Matrix Module Control Register 1 (BMMCR) */
typedef struct {
  __REG32 MEMSW               : 4;
  __REG32                     :28;
} __bmmcr_bits;

/* Bus Matrix Module Control Register2 (BMMCR2) */
typedef struct {
  __REG32 PRTY_RAM0           : 1;
  __REG32 PRTY_FLASH          : 1;
  __REG32 PRTY_PRG            : 1;
  __REG32 PRTY_CRC            : 1;
  __REG32 PRTY_RAM2           : 1;
  __REG32 PRTY_RAM3           : 1;
  __REG32 PRTY_HPI            : 1;
  __REG32                     :25;
} __bmmcr2_bits;

/* MMU Global Control Register (MMUGCR) */
typedef struct {
  __REG32 MPMENA              : 1;
  __REG32                     :31;
} __mmugcr_bits;

/* Clock Control Register (CLKCNTRL) */
typedef struct {
  __REG32                     : 8;
  __REG32 PENA                : 1;
  __REG32                     : 7;
  __REG32 VCLKR               : 4;
  __REG32                     : 4;
  __REG32 VCLK2R              : 4;
  __REG32                     : 4;
} __clkcntrl_bits;

/* ECP Control Register (ECPCNTL) */
typedef struct {
  __REG32 ECPDIV              :16;
  __REG32                     : 7;
  __REG32 ECPCOS              : 1;
  __REG32 ECPSSEL             : 1;
  __REG32                     : 7;
} __ecpcntl_bits;

/* System Exception Control Register (SYSECR) */
typedef struct {
  __REG32                     :14;
  __REG32 RESET               : 2;
  __REG32                     :16;
} __sysecr_bits;

/* System Exception Status Register (SYSESR) */
typedef struct {
  __REG32                     : 3;
  __REG32 EXTRST              : 1;
  __REG32 SWRST               : 1;
  __REG32 CPURST              : 1;
  __REG32                     : 7;
  __REG32 WDRST               : 1;
  __REG32 OSCRST              : 1;
  __REG32 PORST               : 1;
  __REG32                     :16;
} __sysesr_bits;

/* Global Status Register (GLBSTAT) */
typedef struct {
  __REG32 OSCFAIL             : 1;
  __REG32                     : 7;
  __REG32 RFSLIP              : 1;
  __REG32 FBSLIP              : 1;
  __REG32                     :22;
} __glbstat_bits;

/* Device Identification Register (DEV) */
typedef struct {
  __REG32 Platform_ID         : 3;
  __REG32 VERSION             : 5;
  __REG32 RECC                : 1;
  __REG32 Program_parity      : 2;
  __REG32 PPAR                : 1;
  __REG32 IO                  : 1;
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

/* PLL Control Register 3 (PLLCTL3) */
typedef struct {
  __REG32 PLL_DIV             : 3;
  __REG32                     : 5;
  __REG32 PLL_MUL             : 4;
  __REG32                     :10;
  __REG32 OSC_DIV             : 1;
  __REG32                     : 9;
} __pllctl3_bits;

/* CPU Logic BIST Clock Prescaler (STCLKDIV) */
typedef struct {
  __REG32                     :24;
  __REG32 CLKDIV              : 3;
  __REG32                     : 5;
} __stcclkdiv_bits;

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

/* PBIST DD0 Data Register (DD0) */
typedef struct {
  __REG32 D0                    :16;
  __REG32 D1                    :16;
} __pbist_dd0_bits;

/* PBIST DDE Data Register (DE0) */
typedef struct {
  __REG32 E0                    :16;
  __REG32 E1                    :16;
} __pbist_de0_bits;

/* PBIST RAM Configuration Register (RAMT) */
typedef struct {
  __REG32 RLS                   : 2;
  __REG32 PLS                   : 4;
  __REG32 SMS                   : 2;
  __REG32 DWR                   : 8;
  __REG32 RDS                   : 8;
  __REG32 RGS                   : 8;
} __pbist_ramt_bits;

/* PBIST Datalogger Register (DLR) */
typedef struct {
  __REG32 DLR0                  : 1;
  __REG32 DLR1                  : 1;
  __REG32 DLR2                  : 1;
  __REG32 DLR3                  : 1;
  __REG32 DLR4                  : 1;
  __REG32 DLR5                  : 1;
  __REG32 DLR6                  : 1;
  __REG32 DLR7                  : 1;
  __REG32 DLR8                  : 1;
  __REG32 DLR9                  : 1;
  __REG32 DLR10                 : 1;
  __REG32                       :21;
} __pbist_dlr_bits;

/* PBIST Clock-Mux Select Register (CMS) */
typedef struct {
  __REG32 CMS                   : 4;
  __REG32                       :28;
} __pbist_cms_bits;

/* PBIST Program Control Register (STR) */
typedef struct {
  __REG32 STR                   : 5;
  __REG32                       :27;
} __pbist_str_bits;

/* PBIST Chip Select Register (CSR) */
typedef struct {
  __REG32 CSR0                  : 8;
  __REG32 CSR1                  : 8;
  __REG32 CSR2                  : 8;
  __REG32 CSR3                  : 8;
} __pbist_csr_bits;

/* PBIST PBIST Activate/ROM Clock Enable Register (PACT) */
typedef struct {
  __REG32 PACT0                 : 1;
  __REG32 PACT1                 : 1;
  __REG32                       :30;
} __pbist_pact_bits;

/* PBIST Override Register (OVER) */
typedef struct {
  __REG32 OVER0                 : 1;
  __REG32 OVER1                 : 1;
  __REG32 OVER2                 : 1;
  __REG32                       :29;
} __pbist_over_bits;

/* PBIST Fail Status Fail Register 0 (FSRF0) */
typedef struct {
  __REG32 FSRF0                 : 1;
  __REG32                       :31;
} __pbist_fsrf0_bits;

/* PBIST Fail Status Fail Register 1 (FSRF1) */
typedef struct {
  __REG32 FSRF1                 : 1;
  __REG32                       :31;
} __pbist_fsrf1_bits;

/* PBIST ROM Mask Register (ROM) */
typedef struct {
  __REG32 ROM                   : 2;
  __REG32                       :30;
} __pbist_rom_bits;

/* PBIST ROM Algorithm Mask Register (ALGO) */
typedef struct {
  __REG32 ALGO0                 : 8;
  __REG32 ALGO1                 : 8;
  __REG32 ALGO2                 : 8;
  __REG32 ALGO3                 : 8;
} __pbist_algo_bits;

/* PBIST RAM Info Mask Lower Register (RINFOL) */
typedef struct {
  __REG32 RINFOL0               : 8;
  __REG32 RINFOL1               : 8;
  __REG32 RINFOL2               : 8;
  __REG32 RINFOL3               : 8;
} __pbist_rinfol_bits;

/* PBIST RAM Info Mask Upper Register (RINFOU) */
typedef struct {
  __REG32 RINFOU0               : 8;
  __REG32 RINFOU1               : 8;
  __REG32 RINFOU2               : 8;
  __REG32 RINFOU3               : 8;
} __pbist_rinfou_bits;

/* STC global control register0 (STCGCR0) */
typedef struct {
  __REG32 RS_CNT                : 1;
  __REG32                       :15;
  __REG32 INTCOUNT              :16;
} __stcgcr0_bits;

/* STC Global Control Register1 (STCGCR1) */
typedef struct {
  __REG32 STC_ENA               : 4;
  __REG32                       :28;
} __stcgcr1_bits;

/* SelfTest Global Status Register (STCGSTAT) */
typedef struct {
  __REG32 TEST_DONE             : 1;
  __REG32 TEST_FAIL             : 1;
  __REG32                       :30;
} __stcgstat_bits;

/* SelfTest Fail Status Register (STCFSTAT) */
typedef struct {
  __REG32 CPU1_FAIL             : 1;
  __REG32 CPU2_FAIL             : 1;
  __REG32 TO_ERR                : 1;
  __REG32                       :29;
} __stcfstat_bits;

/* TCRAM Wrapper Control Register (RAMCTRL) */
typedef struct {
  __REG32 ECC_DETECT_EN         : 4;
  __REG32                       : 4;
  __REG32 ECC_WR_EN             : 1;
  __REG32                       : 7;
  __REG32 ADDR_PARITY_DISABLE   : 4;
  __REG32                       : 4;
  __REG32 ADDR_PARITY_OVERRIDE  : 4;
  __REG32                       : 2;
  __REG32 EMU_TRACE_DIS         : 1;
  __REG32                       : 1;
} __ramctrl_bits;

/* TCRAM Wrapper Single-Bit Error Correction Threshold Register (RAMTHRESHOLD) */
typedef struct {
  __REG32 THRESHOLD             :16;
  __REG32                       :16;
} __ramthreshold_bits;

/* TCRAM Wrapper Single-Bit Error Occurrences Counter Register (RAMOCCUR) */
typedef struct {
  __REG32 SEO                   :16;
  __REG32                       :16;
} __ramoccur_bits;

/* TCRAM Wrapper Interrupt Control Register (RAMINTCTRL) */
typedef struct {
  __REG32 SERREN                : 1;
  __REG32                       :31;
} __ramintctrl_bits;

/* TCRAM Wrapper Error Status Register (RAMERRSTATUS) */
typedef struct {
  __REG32 SERR                  : 1;
  __REG32                       : 1;
  __REG32 ADDRDECFAIL           : 1;
  __REG32                       : 1;
  __REG32 ADDRCOMPLOGICFAIL     : 1;
  __REG32 DERR                  : 1;
  __REG32                       : 2;
  __REG32 RADDRPARFAIL          : 1;
  __REG32 WADDRPARFAIL          : 1;
  __REG32                       :22;
} __ramerrstatus_bits;

/* TCRAM Wrapper Single-Bit Error Address Register (RAMSERRADDR) */
typedef struct {
  __REG32 SEA                   :18;
  __REG32                       :14;
} __ramserraddr_bits;

/* TCRAM Wrapper Uncorrectable Error Address Register (RAMUERRADDR) */
typedef struct {
  __REG32 UEA                   :23;
  __REG32                       : 9;
} __ramuerraddr_bits;

/* TCRAM Wrapper Test Mode Control Register (RAMTEST) */
typedef struct {
  __REG32 TEST_ENABLE           : 4;
  __REG32                       : 2;
  __REG32 TEST_MODE             : 2;
  __REG32 TRIGGER               : 1;
  __REG32                       :23;
} __ramtest_bits;

/* TCRAM Wrapper Test Mode Vector Register (RAMADDRDECVECT) */
typedef struct {
  __REG32 RAM_CHIP_SELECT       :16;
  __REG32                       :10;
  __REG32 ECCSELECT             : 1;
  __REG32                       : 5;
} __ramaddrdecvect_bits;

/* TCRAM Wrapper Parity Error Address Register (RAMPERRADDR) */
typedef struct {
  __REG32 APEA                  :23;
  __REG32                       : 9;
} __ramperraddr_bits;

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
  __REG32                     : 6;
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

/* Error Status Register (FEDACSTATUS) */
typedef struct {
  __REG32 ERRPRFFLG           : 1;
  __REG32 SBEFLG              : 1;
  __REG32                     : 6;
  __REG32 ECCMULERR           : 1;
  __REG32                     : 1;
  __REG32 ADDPARERR           : 1;
  __REG32                     :21;
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
  __REG32 BSE                 :16;
  __REG32                     :16;
} __fbse_bits;

/* Bank Access Control Register (FBAC) */
typedef struct {
  __REG32 VREADST             : 8;
  __REG32 BAGP                : 8;
  __REG32 OTPPROTDIS          : 8;
  __REG32                     : 8;
} __fbac_bits;

/* Bank Fallback Power Register (FBFALLBACK) */
typedef struct {
  __REG32 BANKPWR0            : 2;
  __REG32 BANKPWR1            : 2;
  __REG32 BANKPWR2            : 2;
  __REG32 BANKPWR3            : 2;
  __REG32                     :24;
} __fbfallback_bits;

/* Bank/Pump Ready Register (FBPRDY) */
typedef struct {
  __REG32 BANKRDY0            : 1;
  __REG32 BANKRDY1            : 1;
  __REG32 BANKRDY2            : 1;
  __REG32 BANKRDY3            : 1;
  __REG32                     :11;
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

/* EEPROM Emulation ECC Register (FEMU_ECC) */
typedef struct {
  __REG32 EMU_ECC             : 8;
  __REG32                     : 8;
  __REG32 RD_ECC              : 8;
  __REG32                     : 8;
} __femu_ecc_bits;

/* Parity Override (FPAR_OVR) */
typedef struct {
  __REG32                     : 8;
  __REG32 ADD_INV_PAR         : 1;
  __REG32 PAR_OVR_KEY         : 3;
  __REG32 BUS_PAR_DIS         : 4;
  __REG32                     :16;
} __fpar_ovr_bits;

/* Error Detection Sector Disable (FEDACSDIS2) */
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

/* EMIF Revision Code and Status Register (RCSR) */
typedef struct {
  __REG32 MINOR_REVISION      : 8;
  __REG32 MAJOR_REVISION      : 8;
  __REG32 MODULE_ID           :14;
  __REG32 FR                  : 1;
  __REG32 BE                  : 1;
} __ercsr_bits;

/* EMIF Asynchronous Configuration Registers (A1CR-A4CR) */
typedef struct {
  __REG32 ASIZE               : 2;
  __REG32 TA                  : 2;
  __REG32 R_HOLD              : 3;
  __REG32 R_STROBE            : 6;
  __REG32 R_SETUP             : 4;
  __REG32 W_HOLD              : 3;
  __REG32 W_STROBE            : 6;
  __REG32 W_SETUP             : 4;
  __REG32                     : 1;
  __REG32 SS                  : 1;
} __eacr_bits;

/* EMIF Interrupt Raw Register (EIRR) */
typedef struct {
  __REG32 AT                  : 1;
  __REG32                     :31;
} __eirr_bits;

/* EMIF Interrupt Mask Register (EIMR) */
typedef struct {
  __REG32 ATM                 : 1;
  __REG32                     :31;
} __eimr_bits;

/* EMIF Interrupt Mask Set Register (EIMSR) */
typedef struct {
  __REG32 ATMSET              : 1;
  __REG32                     :31;
} __eimsr_bits;

/* EMIF Interrupt Mask Clear Register (EIMCR) */
typedef struct {
  __REG32 ATMCLR              : 1;
  __REG32                     :31;
} __eimcr_bits;

/* POM Global Control Register (POMGLBCTRL) */
typedef struct {
  __REG32 ON_OFF              : 4;
  __REG32                     :28;
} __pomglbctrl_bits;

/* POM Revision ID (POMREV) */
typedef struct {
  __REG32 MINOR               : 6;
  __REG32 CUSTOM              : 2;
  __REG32 MAJOR               : 3;
  __REG32 RTL                 : 5;
  __REG32 FUNC                :12;
  __REG32                     : 2;
  __REG32 SCHEME              : 2;
} __pomrev_bits;

/* POM Program Region Start Address Register x (POMPROGSTARTx) */
typedef struct {
  __REG32 STARTADDRESS        :22;
  __REG32                     :10;
} __pomprgstartx_bits;

/* POM Overlay Region Start Address Register x (POMOVLSTARTx) */
typedef struct {
  __REG32 STARTADDRESS        :22;
  __REG32                     :10;
} __pomovlstartx_bits;

/* POM Region Size Register x (POMREGSIZEx) */
typedef struct {
  __REG32 SIZE                : 4;
  __REG32                     :28;
} __pomregsizex_bits;

/* POM Claim Set Register (POMCLAIMSET) */
typedef struct {
  __REG32 SET0                : 1;
  __REG32 SET1                : 1;
  __REG32                     :30;
} __pomclaimset_bits;

/* POM Claim Clear Register (POMCLAIMCLR) */
typedef struct {
  __REG32 CLR0                : 1;
  __REG32 CLR2                : 1;
  __REG32                     :30;
} __pomclaimclr_bits;

/* POM Device Type Register (POMDEVTYPE) */
typedef struct {
  __REG32 MajorType           : 4;
  __REG32 SubType             : 4;
  __REG32                     :24;
} __pomdevtype_bits;

/* POM Peripheral ID 4 Register (POMPERIPHERALID4) */
typedef struct {
  __REG32 JEP106_CC           : 4;
  __REG32 _4K_Count           : 4;
  __REG32                     :24;
} __pomperipheralid4_bits;

/* POM Peripheral ID 0 Register (POMPERIPHERALID0) */
typedef struct {
  __REG32 PartNumber          : 8;
  __REG32                     :24;
} __pomperipheralid0_bits;

/* POM Peripheral ID 1 Register (POMPERIPHERALID1) */
typedef struct {
  __REG32 PartNumber          : 4;
  __REG32 JEP106_ID           : 4;
  __REG32                     :24;
} __pomperipheralid1_bits;

/* POM Peripheral ID 2 Register (POMPERIPHERALID2) */
typedef struct {
  __REG32 JEP106_ID           : 3;
  __REG32 JEDEC               : 1;
  __REG32 Revision            : 4;
  __REG32                     :24;
} __pomperipheralid2_bits;

/* POM Component ID 0 Register (POMCOMPONENTID0) */
typedef struct {
  __REG32 Preamble            : 8;
  __REG32                     :24;
} __pomcomponentid0_bits;

/* POM Component ID 1 Register (POMCOMPONENTID1) */
typedef struct {
  __REG32 Preamble            : 4;
  __REG32 ComponentClass      : 4;
  __REG32                     :24;
} __pomcomponentid1_bits;

/* GIO Global Control Register (GIOGCR0) */
typedef struct {
  __REG32 GIOGCR0             : 1;
  __REG32                     :31;
} __giogcr0_bits;

/* GIO Interrupt Detect Register (GIOINTDET) */
typedef struct {
  __REG32 GIOINTDET0          : 1;
  __REG32 GIOINTDET1          : 1;
  __REG32 GIOINTDET2          : 1;
  __REG32 GIOINTDET3          : 1;
  __REG32 GIOINTDET4          : 1;
  __REG32 GIOINTDET5          : 1;
  __REG32 GIOINTDET6          : 1;
  __REG32 GIOINTDET7          : 1;
  __REG32                     :24;
} __giointdet_bits;

/* GIO Interrupt Polarity Register (GIOPOL) */
typedef struct {
  __REG32 GIOPOL0             : 1;
  __REG32 GIOPOL1             : 1;
  __REG32 GIOPOL2             : 1;
  __REG32 GIOPOL3             : 1;
  __REG32 GIOPOL4             : 1;
  __REG32 GIOPOL5             : 1;
  __REG32 GIOPOL6             : 1;
  __REG32 GIOPOL7             : 1;
  __REG32                     :24;
} __giopol_bits;

/* GIO Interrupt Enable Register (GIOENASET) */
typedef struct {
  __REG32 GIOENASET0          : 1;
  __REG32 GIOENASET1          : 1;
  __REG32 GIOENASET2          : 1;
  __REG32 GIOENASET3          : 1;
  __REG32 GIOENASET4          : 1;
  __REG32 GIOENASET5          : 1;
  __REG32 GIOENASET6          : 1;
  __REG32 GIOENASET7          : 1;
  __REG32                     :24;
} __gioenaset_bits;

/* GIO Interrupt Enable Register (GIOENACLR) */
typedef struct {
  __REG32 GIOENACLR0          : 1;
  __REG32 GIOENACLR1          : 1;
  __REG32 GIOENACLR2          : 1;
  __REG32 GIOENACLR3          : 1;
  __REG32 GIOENACLR4          : 1;
  __REG32 GIOENACLR5          : 1;
  __REG32 GIOENACLR6          : 1;
  __REG32 GIOENACLR7          : 1;
  __REG32                     :24;
} __gioenaclr_bits;

/* GIO Interrupt Priority Register (GIOLVSLSET) */
typedef struct {
  __REG32 GIOLVLSET0          : 1;
  __REG32 GIOLVLSET1          : 1;
  __REG32 GIOLVLSET2          : 1;
  __REG32 GIOLVLSET3          : 1;
  __REG32 GIOLVLSET4          : 1;
  __REG32 GIOLVLSET5          : 1;
  __REG32 GIOLVLSET6          : 1;
  __REG32 GIOLVLSET7          : 1;
  __REG32                     :24;
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
  __REG32 GIOFLG0             : 1;
  __REG32 GIOFLG1             : 1;
  __REG32 GIOFLG2             : 1;
  __REG32 GIOFLG3             : 1;
  __REG32 GIOFLG4             : 1;
  __REG32 GIOFLG5             : 1;
  __REG32 GIOFLG6             : 1;
  __REG32 GIOFLG7             : 1;
  __REG32                     :24;
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

/* GIO Data Direction Registers [A-B][7:0] (GIODIR[A-B][7:0]) */
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

/* SPIP Pin Control Register 5 (SPIP�C5) */
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

/* SPIP Pin Control Register 9 (SPIPPC9) */
typedef struct {
  __REG32 SPISCSSRS0          : 1;
  __REG32 SPISCSSRS1          : 1;
  __REG32 SPISCSSRS2          : 1;
  __REG32 SPISCSSRS3          : 1;
  __REG32 SPISCSSRS4          : 1;
  __REG32 SPISCSSRS5          : 1;
  __REG32 SPISCSSRS6          : 1;
  __REG32 SPISCSSRS7          : 1;
  __REG32 SPIENASRS           : 1;
  __REG32 SPICLKSRS           : 1;
  __REG32 SIMOSRS0            : 1;
  __REG32 SOMISRS0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOSRS0           : 1;
  __REG32 SIMOSRS1            : 1;
  __REG32 SIMOSRS2            : 1;
  __REG32 SIMOSRS3            : 1;
  __REG32 SIMOSRS4            : 1;
  __REG32 SIMOSRS5            : 1;
  __REG32 SIMOSRS6            : 1;
  __REG32 SIMOSRS7            : 1;
  __REG32 _SOMISRS0           : 1;
  __REG32 SOMISRS1            : 1;
  __REG32 SOMISRS2            : 1;
  __REG32 SOMISRS3            : 1;
  __REG32 SOMISRS4            : 1;
  __REG32 SOMISRS5            : 1;
  __REG32 SOMISRS6            : 1;
  __REG32 SOMISRS7            : 1;
} __spippc9_bits;

/* SPI Pin Control Register 9 (SPIPC9) */
typedef struct {
  __REG32 SPISCSSRS0          : 1;
  __REG32                     : 7;
  __REG32 SPIENASRS           : 1;
  __REG32 SPICLKSRS           : 1;
  __REG32 SIMOSRS0            : 1;
  __REG32 SOMISRS0            : 1;
  __REG32                     : 4;
  __REG32 _SIMOSRS0           : 1;
  __REG32                     : 7;
  __REG32 _SOMISRS0           : 1;
  __REG32                     : 7;
} __spipc9_bits;

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
  __REG32 OVERRADDR1          :10;
  __REG32                     :22;
} __spiuerraddr1_bits;

/* SPI RXRAM Uncorrectable Parity Error Address Register (UERRADDR0) */
typedef struct {
  __REG32 OVERRADDR0          :10;
  __REG32                     :22;
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

/* ADC Reset Control Register (ADRSTCR) */
typedef struct {
  __REG32 RESET               : 1;
  __REG32                     :31;
} __adrstcr_bits;

/* ADC Operating Mode Control Register (ADOPMODECR) */
typedef struct {
  __REG32 ADC_EN              : 1;
  __REG32                     : 7;
  __REG32 POWERDOWN           : 1;
  __REG32                     : 7;
  __REG32 RAM_TEST_EN         : 1;
  __REG32 CHN_TEST_EN         : 4;
  __REG32                     : 3;
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
  __REG32                     : 2;
  __REG32 OVR_EV_RAM_IGN      : 1;
  __REG32 EV_CHID             : 1;
  __REG32                     : 2;
  __REG32 EV_DATA_FMT         : 2;
  __REG32                     :22;
} __adevmodecr_bits;

/* ADC Group1 Operating Mode Control Register (ADG1MODECR) */
typedef struct {
  __REG32 FRZ_G1              : 1;
  __REG32 G1_MODE             : 1;
  __REG32                     : 1;
  __REG32 G1_GW_TRIG          : 1;
  __REG32 OVR_G1_RAM_IGN      : 1;
  __REG32 G1_CHID             : 1;
  __REG32                     : 2;
  __REG32 G1_DATA_FMT         : 2;
  __REG32                     :22;
} __adg1modecr_bits;

/* ADC Group2 Operating Mode Control Register (ADG2MODECR) */
typedef struct {
  __REG32 FRZ_G2              : 1;
  __REG32 G2_MODE             : 1;
  __REG32                     : 1;
  __REG32 G2_GW_TRIG          : 1;
  __REG32 OVR_G2_RAM_IGN      : 1;
  __REG32 G2_CHID             : 1;
  __REG32                     : 2;
  __REG32 G2_DATA_FMT         : 2;
  __REG32                     :22;
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

/* ADC Event Group DMA Control Register (ADEVDMACR) */
typedef struct {
  __REG32 EV_DMA_EN           : 1;
  __REG32                     : 1;
  __REG32 EV_BLK_XFER         : 1;
  __REG32                     :13;
  __REG32 EVBLOCKS            : 8;
  __REG32                     : 8;
} __adevdmacr_bits;

/* ADC Group1 DMA Control Register (ADG1DMACR) */
typedef struct {
  __REG32 G1_DMA_EN           : 1;
  __REG32                     : 1;
  __REG32 G1_BLK_XFER         : 1;
  __REG32                     :13;
  __REG32 G1BLOCKS            : 8;
  __REG32                     : 8;
} __adg1dmacr_bits;

/* ADC Group2 DMA Control Register (ADG2DMACR) */
typedef struct {
  __REG32 G2_DMA_EN           : 1;
  __REG32                     : 1;
  __REG32 G2_BLK_XFER         : 1;
  __REG32                     :13;
  __REG32 G2BLOCKS            : 8;
  __REG32                     : 8;
} __adg2dmacr_bits;

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
  __REG32 EV_SEL16            : 1;
  __REG32 EV_SEL17            : 1;
  __REG32 EV_SEL18            : 1;
  __REG32 EV_SEL19            : 1;
  __REG32 EV_SEL20            : 1;
  __REG32 EV_SEL21            : 1;
  __REG32 EV_SEL22            : 1;
  __REG32 EV_SEL23            : 1;
  __REG32 EV_SEL24            : 1;
  __REG32 EV_SEL25            : 1;
  __REG32 EV_SEL26            : 1;
  __REG32 EV_SEL27            : 1;
  __REG32 EV_SEL28            : 1;
  __REG32 EV_SEL29            : 1;
  __REG32 EV_SEL30            : 1;
  __REG32 EV_SEL31            : 1;
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
  __REG32 G1_SEL16            : 1;
  __REG32 G1_SEL17            : 1;
  __REG32 G1_SEL18            : 1;
  __REG32 G1_SEL19            : 1;
  __REG32 G1_SEL20            : 1;
  __REG32 G1_SEL21            : 1;
  __REG32 G1_SEL22            : 1;
  __REG32 G1_SEL23            : 1;
  __REG32 G1_SEL24            : 1;
  __REG32 G1_SEL25            : 1;
  __REG32 G1_SEL26            : 1;
  __REG32 G1_SEL27            : 1;
  __REG32 G1_SEL28            : 1;
  __REG32 G1_SEL29            : 1;
  __REG32 G1_SEL30            : 1;
  __REG32 G1_SEL31            : 1;
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
  __REG32 G2_SEL16            : 1;
  __REG32 G2_SEL17            : 1;
  __REG32 G2_SEL18            : 1;
  __REG32 G2_SEL19            : 1;
  __REG32 G2_SEL20            : 1;
  __REG32 G2_SEL21            : 1;
  __REG32 G2_SEL22            : 1;
  __REG32 G2_SEL23            : 1;
  __REG32 G2_SEL24            : 1;
  __REG32 G2_SEL25            : 1;
  __REG32 G2_SEL26            : 1;
  __REG32 G2_SEL27            : 1;
  __REG32 G2_SEL28            : 1;
  __REG32 G2_SEL29            : 1;
  __REG32 G2_SEL30            : 1;
  __REG32 G2_SEL31            : 1;
} __adg2sel_bits;

/* ADC Calibration and Error Offset Correction Register (ADCALR) */
typedef struct {
  __REG32 ADCALR              :12;
  __REG32                     :20;
} __adcalr_bits;

/* ADC State Machine Status Register (ADSMSTATE) */
typedef struct {
  __REG32 SMSTATE             : 4;
  __REG32                     :28;
} __adsmstate_bits;

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
  __REG32 LAST_CONV16         : 1;
  __REG32 LAST_CONV17         : 1;
  __REG32 LAST_CONV18         : 1;
  __REG32 LAST_CONV19         : 1;
  __REG32 LAST_CONV20         : 1;
  __REG32 LAST_CONV21         : 1;
  __REG32 LAST_CONV22         : 1;
  __REG32 LAST_CONV23         : 1;
  __REG32 LAST_CONV24         : 1;
  __REG32 LAST_CONV25         : 1;
  __REG32 LAST_CONV26         : 1;
  __REG32 LAST_CONV27         : 1;
  __REG32 LAST_CONV28         : 1;
  __REG32 LAST_CONV29         : 1;
  __REG32 LAST_CONV30         : 1;
  __REG32 LAST_CONV31         : 1;
} __adlastconv_bits;

/* ADC Event Group Results FIFO (ADEVBUFFER) */
typedef struct {
  __REG32 EV_DR               :12;
  __REG32                     : 4;
  __REG32 EV_CHID             : 5;
  __REG32                     :10;
  __REG32 EV_EMPTY            : 1;
} __adevbuffer_bits;

/* ADC Group1 Results FIFO (ADG1BUFFER) */
typedef struct {
  __REG32 G1_DR               :12;
  __REG32                     : 4;
  __REG32 G1_CHID             : 5;
  __REG32                     :10;
  __REG32 G1_EMPTY            : 1;
} __adg1buffer_bits;

/* ADC Group2 Results FIFO (ADG2BUFFER) */
typedef struct {
  __REG32 G2_DR               :12;
  __REG32                     : 4;
  __REG32 G2_CHID             : 5;
  __REG32                     :10;
  __REG32 G2_EMPTY            : 1;
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
  __REG32 ADEVT_IN            : 1;
  __REG32                     :31;
} __adevtset_bits;

/* ADC ADEVT Pin Clear Register (ADEVTCLR) */
typedef struct {
  __REG32 ADEVT_IN            : 1;
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
  __REG32 MAG_CHID            : 5;
  __REG32                     : 3;
  __REG32 COMP_CHID           : 5;
  __REG32                     : 1;
  __REG32 CMP_GE_LT           : 1;
  __REG32 CHN_THR_COMP        : 1;
  __REG32 MAG_THR             :12;
  __REG32                     : 4;
} __admagintcr_bits;

/* ADC Magnitude Compare Interruptx Mask (ADMAGINTxMASK) */
typedef struct {
  __REG32 MAG_INT_MASK        :12;
  __REG32                     :20;
} __admagintmask_bits;

/* ADC Magnitude Compare Interrupt Enable Set (ADMAGINTENASET) */
typedef struct {
  __REG32 MAG_INT_ENA_SET0    : 1;
  __REG32 MAG_INT_ENA_SET1    : 1;
  __REG32 MAG_INT_ENA_SET2    : 1;
  __REG32                     :29;
} __admagthrintenaset_bits;

/* ADC Magnitude Compare Interrupt Enable Clear (ADMAGINTENACLR) */
typedef struct {
  __REG32 MAG_INT_ENA_CLR0    : 1;
  __REG32 MAG_INT_ENA_CLR1    : 1;
  __REG32 MAG_INT_ENA_CLR2    : 1;
  __REG32                     :29;
} __admagthrintenaclr_bits;

/* ADC Magnitude Compare Interrupt Flag (ADMAGINTFLG) */
typedef struct {
  __REG32 MAG_INT_FLG0        : 1;
  __REG32 MAG_INT_FLG1        : 1;
  __REG32 MAG_INT_FLG2        : 1;
  __REG32                     :29;
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
  __REG32 EV_RAM_ADDR         : 8;
  __REG32                     :24;
} __adevramaddr_bits;

/* ADC Group1 RAM Write Address (ADG1RAMWRADDR) */
typedef struct {
  __REG32 G1_RAM_ADDR         : 8;
  __REG32                     :24;
} __adg1ramaddr_bits;

/* ADC Group2 RAM Write Address (ADG2RAMWRADDR) */
typedef struct {
  __REG32 G2_RAM_ADDR         : 8;
  __REG32                     :24;
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
  __REG32                     : 2;
  __REG32 ADDR                :10;
  __REG32                     :20;
} __adparaddr_bits;

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

/* Output Buffer Command Mask Register (OBCR) */
typedef struct {
  __REG32 OBRS                : 7;
  __REG32                     : 1;
  __REG32 VIEW                : 1;
  __REG32 REQ                 : 1;
  __REG32                     : 5;
  __REG32 OBSYS               : 1;
  __REG32 OBRH                : 7;
  __REG32                     : 9;
} __fobcr_bits;

/* Global Configuration Register (HETGCR) */
typedef struct {
  __REG32 TO                  : 1;
  __REG32                     :15;
  __REG32 CMS                 : 1;
  __REG32 IS                  : 1;
  __REG32 PPF                 : 1;
  __REG32                     : 2;
  __REG32 MP                  : 2;
  __REG32                     : 1;
  __REG32 HET_PIN_ENA         : 1;
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
  __REG32 HETADDR             : 9;
  __REG32                     :23;
} __hetaddr_bits;

/* NHET Offset Index Priority Level 1 Register (HETOFF1) */
typedef struct {
  __REG32 Offset1             : 6;
  __REG32                     :26;
} __hetoff1_bits;

/* NHET Offset Index Priority Level 2 Register (HETOFF2) */
typedef struct {
  __REG32 Offset2             : 6;
  __REG32                     :26;
} __hetoff2_bits;

/* NHET Interrupt Enable Set Register (HETINTENAS) */
typedef struct {
  __REG32 HETINTENAS0         : 1;
  __REG32 HETINTENAS1         : 1;
  __REG32 HETINTENAS2         : 1;
  __REG32 HETINTENAS3         : 1;
  __REG32 HETINTENAS4         : 1;
  __REG32 HETINTENAS5         : 1;
  __REG32 HETINTENAS6         : 1;
  __REG32 HETINTENAS7         : 1;
  __REG32 HETINTENAS8         : 1;
  __REG32 HETINTENAS9         : 1;
  __REG32 HETINTENAS10        : 1;
  __REG32 HETINTENAS11        : 1;
  __REG32 HETINTENAS12        : 1;
  __REG32 HETINTENAS13        : 1;
  __REG32 HETINTENAS14        : 1;
  __REG32 HETINTENAS15        : 1;
  __REG32 HETINTENAS16        : 1;
  __REG32 HETINTENAS17        : 1;
  __REG32 HETINTENAS18        : 1;
  __REG32 HETINTENAS19        : 1;
  __REG32 HETINTENAS20        : 1;
  __REG32 HETINTENAS21        : 1;
  __REG32 HETINTENAS22        : 1;
  __REG32 HETINTENAS23        : 1;
  __REG32 HETINTENAS24        : 1;
  __REG32 HETINTENAS25        : 1;
  __REG32 HETINTENAS26        : 1;
  __REG32 HETINTENAS27        : 1;
  __REG32 HETINTENAS28        : 1;
  __REG32 HETINTENAS29        : 1;
  __REG32 HETINTENAS30        : 1;
  __REG32 HETINTENAS31        : 1;
} __hetintenas_bits;

/* Interrupt Enable Clear Register (HETINTENAC) */
typedef struct {
  __REG32 HETINTENAC0         : 1;
  __REG32 HETINTENAC1         : 1;
  __REG32 HETINTENAC2         : 1;
  __REG32 HETINTENAC3         : 1;
  __REG32 HETINTENAC4         : 1;
  __REG32 HETINTENAC5         : 1;
  __REG32 HETINTENAC6         : 1;
  __REG32 HETINTENAC7         : 1;
  __REG32 HETINTENAC8         : 1;
  __REG32 HETINTENAC9         : 1;
  __REG32 HETINTENAC10        : 1;
  __REG32 HETINTENAC11        : 1;
  __REG32 HETINTENAC12        : 1;
  __REG32 HETINTENAC13        : 1;
  __REG32 HETINTENAC14        : 1;
  __REG32 HETINTENAC15        : 1;
  __REG32 HETINTENAC16        : 1;
  __REG32 HETINTENAC17        : 1;
  __REG32 HETINTENAC18        : 1;
  __REG32 HETINTENAC19        : 1;
  __REG32 HETINTENAC20        : 1;
  __REG32 HETINTENAC21        : 1;
  __REG32 HETINTENAC22        : 1;
  __REG32 HETINTENAC23        : 1;
  __REG32 HETINTENAC24        : 1;
  __REG32 HETINTENAC25        : 1;
  __REG32 HETINTENAC26        : 1;
  __REG32 HETINTENAC27        : 1;
  __REG32 HETINTENAC28        : 1;
  __REG32 HETINTENAC29        : 1;
  __REG32 HETINTENAC30        : 1;
  __REG32 HETINTENAC31        : 1;
} __hetintenac_bits;

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
  __REG32                     : 5;
  __REG32 DebugStatusFlag     : 1;
  __REG32                     :23;
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
  __REG32 HRShare_25_24        : 1;
  __REG32 HRShare_27_26        : 1;
  __REG32 HRShare_29_28        : 1;
  __REG32 HRShare_31_30        : 1;
  __REG32                      :16;
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
  __REG32 HR_XORShare_25_24    : 1;
  __REG32 HR_XORShare_27_26    : 1;
  __REG32 HR_XORShare_29_28    : 1;
  __REG32 HR_XORShare_31_30    : 1;
  __REG32                      :16;
} __hetxor_bits;

/* Request Enable Set Register (HETREQENS) */
typedef struct {
  __REG32 REQ_ENA_0            : 1;
  __REG32 REQ_ENA_1            : 1;
  __REG32 REQ_ENA_2            : 1;
  __REG32 REQ_ENA_3            : 1;
  __REG32 REQ_ENA_4            : 1;
  __REG32 REQ_ENA_5            : 1;
  __REG32 REQ_ENA_6            : 1;
  __REG32 REQ_ENA_7            : 1;
  __REG32                      :24;
} __hetreqens_bits;

/* Request Enable Clear Register (HETREQENC) */
typedef struct {
  __REG32 REQ_DIS_0            : 1;
  __REG32 REQ_DIS_1            : 1;
  __REG32 REQ_DIS_2            : 1;
  __REG32 REQ_DIS_3            : 1;
  __REG32 REQ_DIS_4            : 1;
  __REG32 REQ_DIS_5            : 1;
  __REG32 REQ_DIS_6            : 1;
  __REG32 REQ_DIS_7            : 1;
  __REG32                      :24;
} __hetreqenc_bits;

/* Request Destination Select Register (HETREQDS) */
typedef struct {
  __REG32 TDS_0                : 1;
  __REG32 TDS_1                : 1;
  __REG32 TDS_2                : 1;
  __REG32 TDS_3                : 1;
  __REG32 TDS_4                : 1;
  __REG32 TDS_5                : 1;
  __REG32 TDS_6                : 1;
  __REG32 TDS_7                : 1;
  __REG32                      : 8;
  __REG32 TDBS_0               : 1;
  __REG32 TDBS_1               : 1;
  __REG32 TDBS_2               : 1;
  __REG32 TDBS_3               : 1;
  __REG32 TDBS_4               : 1;
  __REG32 TDBS_5               : 1;
  __REG32 TDBS_6               : 1;
  __REG32 TDBS_7               : 1;
  __REG32                      : 8;
} __hetreqds_bits;

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

/* Parity Control Register (HETPCR) */
typedef struct {
  __REG32 PARITY_ENA           : 4;
  __REG32                      : 4;
  __REG32 TEST                 : 1;
  __REG32                      :23;
} __hetpcr_bits;

/* Parity Address Register (HETPAR) */
typedef struct {
  __REG32 PAOFF                :13;
  __REG32                      :19;
} __hetpar_bits;

/* NHET Parity Pin Register (HETPPR) */
typedef struct {
  __REG32 HETPPR0              : 1;
  __REG32 HETPPR1              : 1;
  __REG32 HETPPR2              : 1;
  __REG32 HETPPR3              : 1;
  __REG32 HETPPR4              : 1;
  __REG32 HETPPR5              : 1;
  __REG32 HETPPR6              : 1;
  __REG32 HETPPR7              : 1;
  __REG32 HETPPR8              : 1;
  __REG32 HETPPR9              : 1;
  __REG32 HETPPR10             : 1;
  __REG32 HETPPR11             : 1;
  __REG32 HETPPR12             : 1;
  __REG32 HETPPR13             : 1;
  __REG32 HETPPR14             : 1;
  __REG32 HETPPR15             : 1;
  __REG32 HETPPR16             : 1;
  __REG32 HETPPR17             : 1;
  __REG32 HETPPR18             : 1;
  __REG32 HETPPR19             : 1;
  __REG32 HETPPR20             : 1;
  __REG32 HETPPR21             : 1;
  __REG32 HETPPR22             : 1;
  __REG32 HETPPR23             : 1;
  __REG32 HETPPR24             : 1;
  __REG32 HETPPR25             : 1;
  __REG32 HETPPR26             : 1;
  __REG32 HETPPR27             : 1;
  __REG32 HETPPR28             : 1;
  __REG32 HETPPR29             : 1;
  __REG32 HETPPR30             : 1;
  __REG32 HETPPR31             : 1;
} __hetppr_bits;

/* NHET Suppression Filter Preload Register (HETSFPRLD) */
typedef struct {
  __REG32 CPRLD                :10;
  __REG32                      : 6;
  __REG32 CCDIV                : 2;
  __REG32                      :14;
} __hetsfprld_bits;

/* NHET Suppression Filter Enable Register (HETSFENA) */
typedef struct {
  __REG32 HETSFENA0              : 1;
  __REG32 HETSFENA1              : 1;
  __REG32 HETSFENA2              : 1;
  __REG32 HETSFENA3              : 1;
  __REG32 HETSFENA4              : 1;
  __REG32 HETSFENA5              : 1;
  __REG32 HETSFENA6              : 1;
  __REG32 HETSFENA7              : 1;
  __REG32 HETSFENA8              : 1;
  __REG32 HETSFENA9              : 1;
  __REG32 HETSFENA10             : 1;
  __REG32 HETSFENA11             : 1;
  __REG32 HETSFENA12             : 1;
  __REG32 HETSFENA13             : 1;
  __REG32 HETSFENA14             : 1;
  __REG32 HETSFENA15             : 1;
  __REG32 HETSFENA16             : 1;
  __REG32 HETSFENA17             : 1;
  __REG32 HETSFENA18             : 1;
  __REG32 HETSFENA19             : 1;
  __REG32 HETSFENA20             : 1;
  __REG32 HETSFENA21             : 1;
  __REG32 HETSFENA22             : 1;
  __REG32 HETSFENA23             : 1;
  __REG32 HETSFENA24             : 1;
  __REG32 HETSFENA25             : 1;
  __REG32 HETSFENA26             : 1;
  __REG32 HETSFENA27             : 1;
  __REG32 HETSFENA28             : 1;
  __REG32 HETSFENA29             : 1;
  __REG32 HETSFENA30             : 1;
  __REG32 HETSFENA31             : 1;
} __hetsfena_bits;

/* NHET Loop Back Pair Select Register (HETLBPSEL) */
typedef struct {
  __REG32 LBP_SEL_1_0             : 1;
  __REG32 LBP_SEL_3_2             : 1;
  __REG32 LBP_SEL_5_4             : 1;
  __REG32 LBP_SEL_7_6             : 1;
  __REG32 LBP_SEL_9_8             : 1;
  __REG32 LBP_SEL_11_10           : 1;
  __REG32 LBP_SEL_13_12           : 1;
  __REG32 LBP_SEL_15_14           : 1;
  __REG32 LBP_SEL_17_16           : 1;
  __REG32 LBP_SEL_19_18           : 1;
  __REG32 LBP_SEL_21_20           : 1;
  __REG32 LBP_SEL_23_22           : 1;
  __REG32 LBP_SEL_25_24           : 1;
  __REG32 LBP_SEL_27_26           : 1;
  __REG32 LBP_SEL_29_28           : 1;
  __REG32 LBP_SEL_31_30           : 1;
  __REG32 LBP_TYPE_1_0            : 1;
  __REG32 LBP_TYPE_3_2            : 1;
  __REG32 LBP_TYPE_5_4            : 1;
  __REG32 LBP_TYPE_7_6            : 1;
  __REG32 LBP_TYPE_9_8            : 1;
  __REG32 LBP_TYPE_11_10          : 1;
  __REG32 LBP_TYPE_13_12          : 1;
  __REG32 LBP_TYPE_15_14          : 1;
  __REG32 LBP_TYPE_17_16          : 1;
  __REG32 LBP_TYPE_19_18          : 1;
  __REG32 LBP_TYPE_21_20          : 1;
  __REG32 LBP_TYPE_23_22          : 1;
  __REG32 LBP_TYPE_25_24          : 1;
  __REG32 LBP_TYPE_27_26          : 1;
  __REG32 LBP_TYPE_29_28          : 1;
  __REG32 LBP_TYPE_31_30          : 1;
} __hetlbpsel_bits;


/* NHET Loop Back Pair Direction Register (HETLBPDIR) */
typedef struct {
  __REG32 LBP_DIR_1_0             : 1;
  __REG32 LBP_DIR_3_2             : 1;
  __REG32 LBP_DIR_5_4             : 1;
  __REG32 LBP_DIR_7_6             : 1;
  __REG32 LBP_DIR_9_8             : 1;
  __REG32 LBP_DIR_11_10           : 1;
  __REG32 LBP_DIR_13_12           : 1;
  __REG32 LBP_DIR_15_14           : 1;
  __REG32 LBP_DIR_17_16           : 1;
  __REG32 LBP_DIR_19_18           : 1;
  __REG32 LBP_DIR_21_20           : 1;
  __REG32 LBP_DIR_23_22           : 1;
  __REG32 LBP_DIR_25_24           : 1;
  __REG32 LBP_DIR_27_26           : 1;
  __REG32 LBP_DIR_29_28           : 1;
  __REG32 LBP_DIR_31_30           : 1;
  __REG32 IODFTENA                : 4;
  __REG32                         :12;
} __hetlbpdir_bits;

/* NHET Pin Disable Register (HETPINDIS) */
typedef struct {
  __REG32 HETPINDIS0              : 1;
  __REG32 HETPINDIS1              : 1;
  __REG32 HETPINDIS2              : 1;
  __REG32 HETPINDIS3              : 1;
  __REG32 HETPINDIS4              : 1;
  __REG32 HETPINDIS5              : 1;
  __REG32 HETPINDIS6              : 1;
  __REG32 HETPINDIS7              : 1;
  __REG32 HETPINDIS8              : 1;
  __REG32 HETPINDIS9              : 1;
  __REG32 HETPINDIS10             : 1;
  __REG32 HETPINDIS11             : 1;
  __REG32 HETPINDIS12             : 1;
  __REG32 HETPINDIS13             : 1;
  __REG32 HETPINDIS14             : 1;
  __REG32 HETPINDIS15             : 1;
  __REG32 HETPINDIS16             : 1;
  __REG32 HETPINDIS17             : 1;
  __REG32 HETPINDIS18             : 1;
  __REG32 HETPINDIS19             : 1;
  __REG32 HETPINDIS20             : 1;
  __REG32 HETPINDIS21             : 1;
  __REG32 HETPINDIS22             : 1;
  __REG32 HETPINDIS23             : 1;
  __REG32 HETPINDIS24             : 1;
  __REG32 HETPINDIS25             : 1;
  __REG32 HETPINDIS26             : 1;
  __REG32 HETPINDIS27             : 1;
  __REG32 HETPINDIS28             : 1;
  __REG32 HETPINDIS29             : 1;
  __REG32 HETPINDIS30             : 1;
  __REG32 HETPINDIS31             : 1;
} __hetpindis_bits;

/* Global Control Register (HTU GC) */
typedef struct {
  __REG32 HTURES               : 1;
  __REG32                      : 7;
  __REG32 DEBM                 : 1;
  __REG32                      : 7;
  __REG32 HTUEN                : 1;
  __REG32                      : 7;
  __REG32 VBUSHOLD             : 1;
  __REG32                      : 7;
} __htugc_bits;

/* Control Packet Enable Register (HTU CPENA) */
typedef struct {
  __REG32 CPENA0               : 1;
  __REG32 CPENA1               : 1;
  __REG32 CPENA2               : 1;
  __REG32 CPENA3               : 1;
  __REG32 CPENA4               : 1;
  __REG32 CPENA5               : 1;
  __REG32 CPENA6               : 1;
  __REG32 CPENA7               : 1;
  __REG32 CPENA8               : 1;
  __REG32 CPENA9               : 1;
  __REG32 CPENA10              : 1;
  __REG32 CPENA11              : 1;
  __REG32 CPENA12              : 1;
  __REG32 CPENA13              : 1;
  __REG32 CPENA14              : 1;
  __REG32 CPENA15              : 1;
  __REG32                      :16;
} __htcpena_bits;

/* Control Packet (CP) Busy Register 0 (HTU BUSY0) */
typedef struct {
  __REG32 BUSY1B               : 1;
  __REG32                      : 7;
  __REG32 BUSY1A               : 1;
  __REG32                      : 7;
  __REG32 BUSY0B               : 1;
  __REG32                      : 7;
  __REG32 BUSY0A               : 1;
  __REG32                      : 7;
} __htubusy0_bits;

/* Control Packet (CP) Busy Register 1 (HTU BUSY1) */
typedef struct {
  __REG32 BUSY3B               : 1;
  __REG32                      : 7;
  __REG32 BUSY3A               : 1;
  __REG32                      : 7;
  __REG32 BUSY2B               : 1;
  __REG32                      : 7;
  __REG32 BUSY2A               : 1;
  __REG32                      : 7;
} __htubusy1_bits;

/* Control Packet (CP) Busy Register 2 (HTU BUSY2) */
typedef struct {
  __REG32 BUSY5B               : 1;
  __REG32                      : 7;
  __REG32 BUSY5A               : 1;
  __REG32                      : 7;
  __REG32 BUSY4B               : 1;
  __REG32                      : 7;
  __REG32 BUSY4A               : 1;
  __REG32                      : 7;
} __htubusy2_bits;

/* Control Packet (CP) Busy Register 3 (HTU BUSY3) */
typedef struct {
  __REG32 BUSY7B               : 1;
  __REG32                      : 7;
  __REG32 BUSY7A               : 1;
  __REG32                      : 7;
  __REG32 BUSY6B               : 1;
  __REG32                      : 7;
  __REG32 BUSY6A               : 1;
  __REG32                      : 7;
} __htubusy3_bits;

/* Active Control Packet and Error Register (HTU ACPE) */
typedef struct {
  __REG32 NACP                 : 4;
  __REG32                      : 4;
  __REG32 CETCOUNT             : 5;
  __REG32                      : 1;
  __REG32 BUSBUSY              : 1;
  __REG32 TIPF                 : 1;
  __REG32 ERRCPN               : 4;
  __REG32                      : 4;
  __REG32 ERRETC               : 5;
  __REG32                      : 2;
  __REG32 ERRF                 : 1;
} __htuacp_bits;

/* Request Lost and Bus Error Control Register (HTU RLBECTRL) */
typedef struct {
  __REG32 RL_INT_ENA           : 1;
  __REG32                      : 7;
  __REG32 CORL                 : 1;
  __REG32                      : 7;
  __REG32 BER_INT_ENA          : 1;
  __REG32                      :15;
} __hturlbectrl_bits;

/* Buffer Full Interrupt Enable Set Register (HTU BFINTS) */
typedef struct {
  __REG32 BFINTENA0            : 1;
  __REG32 BFINTENA1            : 1;
  __REG32 BFINTENA2            : 1;
  __REG32 BFINTENA3            : 1;
  __REG32 BFINTENA4            : 1;
  __REG32 BFINTENA5            : 1;
  __REG32 BFINTENA6            : 1;
  __REG32 BFINTENA7            : 1;
  __REG32 BFINTENA8            : 1;
  __REG32 BFINTENA9            : 1;
  __REG32 BFINTENA10           : 1;
  __REG32 BFINTENA11           : 1;
  __REG32 BFINTENA12           : 1;
  __REG32 BFINTENA13           : 1;
  __REG32 BFINTENA14           : 1;
  __REG32 BFINTENA15           : 1;
  __REG32                      :16;
} __htubfints_bits;

/* Buffer Full Interrupt Enable Clear Register (HTU BFINTC) */
typedef struct {
  __REG32 BFINTDIS0            : 1;
  __REG32 BFINTDIS1            : 1;
  __REG32 BFINTDIS2            : 1;
  __REG32 BFINTDIS3            : 1;
  __REG32 BFINTDIS4            : 1;
  __REG32 BFINTDIS5            : 1;
  __REG32 BFINTDIS6            : 1;
  __REG32 BFINTDIS7            : 1;
  __REG32 BFINTDIS8            : 1;
  __REG32 BFINTDIS9            : 1;
  __REG32 BFINTDIS10           : 1;
  __REG32 BFINTDIS11           : 1;
  __REG32 BFINTDIS12           : 1;
  __REG32 BFINTDIS13           : 1;
  __REG32 BFINTDIS14           : 1;
  __REG32 BFINTDIS15           : 1;
  __REG32                      :16;
} __htubfintc_bits;

/* Interrupt Mapping Register (HTU INTMAP) */
typedef struct {
  __REG32 CPINTMAP0            : 1;
  __REG32 CPINTMAP1            : 1;
  __REG32 CPINTMAP2            : 1;
  __REG32 CPINTMAP3            : 1;
  __REG32 CPINTMAP4            : 1;
  __REG32 CPINTMAP5            : 1;
  __REG32 CPINTMAP6            : 1;
  __REG32 CPINTMAP7            : 1;
  __REG32 CPINTMAP8            : 1;
  __REG32 CPINTMAP9            : 1;
  __REG32 CPINTMAP10           : 1;
  __REG32 CPINTMAP11           : 1;
  __REG32 CPINTMAP12           : 1;
  __REG32 CPINTMAP13           : 1;
  __REG32 CPINTMAP14           : 1;
  __REG32 CPINTMAP15           : 1;
  __REG32 MAPSEL               : 1;
  __REG32                      :15;
} __htuintmap_bits;

/* Interrupt Offset Register 0 (HTU INTOFF0) */
typedef struct {
  __REG32 CPOFF0               : 4;
  __REG32                      : 4;
  __REG32 INTTYPE0             : 2;
  __REG32                      :22;
} __htuintoff0_bits;

/* Interrupt Offset Register 1 (HTU INTOFF1) */
typedef struct {
  __REG32 CPOFF1               : 4;
  __REG32                      : 4;
  __REG32 INTTYPE1             : 2;
  __REG32                      :22;
} __htuintoff1_bits;

/* Buffer Initialization Mode Register (HTU BIM) */
typedef struct {
  __REG32 BIM                  : 8;
  __REG32                      :24;
} __htubim_bits;

/* Request Lost Flag Register (HTU RLOSTFL) */
typedef struct {
  __REG32 CPRLFL0              : 1;
  __REG32 CPRLFL1              : 1;
  __REG32 CPRLFL2              : 1;
  __REG32 CPRLFL3              : 1;
  __REG32 CPRLFL4              : 1;
  __REG32 CPRLFL5              : 1;
  __REG32 CPRLFL6              : 1;
  __REG32 CPRLFL7              : 1;
  __REG32 CPRLFL8              : 1;
  __REG32 CPRLFL9              : 1;
  __REG32 CPRLFL10             : 1;
  __REG32 CPRLFL11             : 1;
  __REG32 CPRLFL12             : 1;
  __REG32 CPRLFL13             : 1;
  __REG32 CPRLFL14             : 1;
  __REG32 CPRLFL15             : 1;
  __REG32                      :16;
} __hturlostfl_bits;

/* Buffer Full Interrupt Flag Register (HTU BFINTFL) */
typedef struct {
  __REG32 BFINTFL0             : 1;
  __REG32 BFINTFL1             : 1;
  __REG32 BFINTFL2             : 1;
  __REG32 BFINTFL3             : 1;
  __REG32 BFINTFL4             : 1;
  __REG32 BFINTFL5             : 1;
  __REG32 BFINTFL6             : 1;
  __REG32 BFINTFL7             : 1;
  __REG32 BFINTFL8             : 1;
  __REG32 BFINTFL9             : 1;
  __REG32 BFINTFL10            : 1;
  __REG32 BFINTFL11            : 1;
  __REG32 BFINTFL12            : 1;
  __REG32 BFINTFL13            : 1;
  __REG32 BFINTFL14            : 1;
  __REG32 BFINTFL15            : 1;
  __REG32                      :16;
} __htubfintfl_bits;

/* BER Interrupt Flag Register (HTU BERINTFL) */
typedef struct {
  __REG32 BERINTFL0            : 1;
  __REG32 BERINTFL1            : 1;
  __REG32 BERINTFL2            : 1;
  __REG32 BERINTFL3            : 1;
  __REG32 BERINTFL4            : 1;
  __REG32 BERINTFL5            : 1;
  __REG32 BERINTFL6            : 1;
  __REG32 BERINTFL7            : 1;
  __REG32 BERINTFL8            : 1;
  __REG32 BERINTFL9            : 1;
  __REG32 BERINTFL10           : 1;
  __REG32 BERINTFL11           : 1;
  __REG32 BERINTFL12           : 1;
  __REG32 BERINTFL13           : 1;
  __REG32 BERINTFL14           : 1;
  __REG32 BERINTFL15           : 1;
  __REG32                      :16;
} __htuberintfl_bits;

/* Debug Control Register (HTU DCTRL) */
typedef struct {
  __REG32 DBREN                : 1;
  __REG32                      :15;
  __REG32 HTUDBGS              : 1;
  __REG32                      : 7;
  __REG32 CPNUM                : 4;
  __REG32                      : 4;
} __htudcrtl_bits;

/* Module Identification Register (HTU ID) */
typedef struct {
  __REG32 REV                  : 8;
  __REG32 TYPE                 : 8;
  __REG32 CLASS                : 8;
  __REG32                      : 8;
} __htuid_bits;

/* Parity Control Register (HTU PCR) */
typedef struct {
  __REG32 PARITY_ENA           : 4;
  __REG32                      : 4;
  __REG32 TEST                 : 1;
  __REG32                      : 7;
  __REG32 COPE                 : 1;
  __REG32                      :15;
} __htupcr_bits;

/* Parity Address Register (HTU PAR) */
typedef struct {
  __REG32 PAOFF                : 8;
  __REG32                      : 8;
  __REG32 PEFT                 : 1;
  __REG32                      :15;
} __htupar_bits;

/* Memory Protection Control and Status Register (HTU MPCS) */
typedef struct {
  __REG32 REG0ENA              : 1;
  __REG32 ACCR0                : 1;
  __REG32 INTENA0              : 1;
  __REG32 REG01ENA             : 1;
  __REG32 ACCR01               : 1;
  __REG32 INTENA01             : 1;
  __REG32                      : 2;
  __REG32 CPNUM1               : 4;
  __REG32                      : 4;
  __REG32 MPEFT0               : 1;
  __REG32 MPEFT1               : 1;
  __REG32                      : 6;
  __REG32 CPNUM0               : 4;
  __REG32                      : 4;
} __htumpcs_bits;

/* Initial NHET Address and Control Register (HTU IHADDRCT) */
typedef struct {
  __REG32 IHADDR                :13;
  __REG32                       : 3;
  __REG32 TMBB                  : 2;
  __REG32 TMBA                  : 2;
  __REG32 ADDMF                 : 1;
  __REG32 ADDMH                 : 1;
  __REG32 SIZE                  : 1;
  __REG32 DIR                   : 1;
  __REG32                       : 8;
} __htudcpihaddrct_bits;

/* Initial Transfer Count Register (HTU ITCOUNT) */
typedef struct {
  __REG32 IFTCOUNT              : 8;
  __REG32                       : 8;
  __REG32 IETCOUNT              : 5;
  __REG32                       :11;
} __htudcpitcount_bits;

/* Current Frame Count Register (HTU CFCOUNT) */
typedef struct {
  __REG32 CFTCTB                : 8;
  __REG32                       : 8;
  __REG32 CFTCTA                : 8;
  __REG32                       : 8;
} __htudcpcfcount_bits;

/* DMA Global Control Register (GCTRL) */
typedef struct {
  __REG32 DMA_RES               : 1;
  __REG32                       : 7;
  __REG32 DEBUG_MODE            : 2;
  __REG32                       : 4;
  __REG32 BUS_BUSY              : 1;
  __REG32                       : 1;
  __REG32 DMA_EN                : 1;
  __REG32                       :15;
} __dmagctrl_bits;

/* DMA Channel Pending Register (PEND) */
typedef struct {
  __REG32 PEND0                 : 1;
  __REG32 PEND1                 : 1;
  __REG32 PEND2                 : 1;
  __REG32 PEND3                 : 1;
  __REG32 PEND4                 : 1;
  __REG32 PEND5                 : 1;
  __REG32 PEND6                 : 1;
  __REG32 PEND7                 : 1;
  __REG32 PEND8                 : 1;
  __REG32 PEND9                 : 1;
  __REG32 PEND10                : 1;
  __REG32 PEND11                : 1;
  __REG32 PEND12                : 1;
  __REG32 PEND13                : 1;
  __REG32 PEND14                : 1;
  __REG32 PEND15                : 1;
  __REG32 PEND16                : 1;
  __REG32 PEND17                : 1;
  __REG32 PEND18                : 1;
  __REG32 PEND19                : 1;
  __REG32 PEND20                : 1;
  __REG32 PEND21                : 1;
  __REG32 PEND22                : 1;
  __REG32 PEND23                : 1;
  __REG32 PEND24                : 1;
  __REG32 PEND25                : 1;
  __REG32 PEND26                : 1;
  __REG32 PEND27                : 1;
  __REG32 PEND28                : 1;
  __REG32 PEND29                : 1;
  __REG32 PEND30                : 1;
  __REG32 PEND31                : 1;
} __dmapend_bits;

/* DMA Status Register (DMASTAT) */
typedef struct {
  __REG32 DMASTAT0              : 1;
  __REG32 DMASTAT1              : 1;
  __REG32 DMASTAT2              : 1;
  __REG32 DMASTAT3              : 1;
  __REG32 DMASTAT4              : 1;
  __REG32 DMASTAT5              : 1;
  __REG32 DMASTAT6              : 1;
  __REG32 DMASTAT7              : 1;
  __REG32 DMASTAT8              : 1;
  __REG32 DMASTAT9              : 1;
  __REG32 DMASTAT10             : 1;
  __REG32 DMASTAT11             : 1;
  __REG32 DMASTAT12             : 1;
  __REG32 DMASTAT13             : 1;
  __REG32 DMASTAT14             : 1;
  __REG32 DMASTAT15             : 1;
  __REG32 DMASTAT16             : 1;
  __REG32 DMASTAT17             : 1;
  __REG32 DMASTAT18             : 1;
  __REG32 DMASTAT19             : 1;
  __REG32 DMASTAT20             : 1;
  __REG32 DMASTAT21             : 1;
  __REG32 DMASTAT22             : 1;
  __REG32 DMASTAT23             : 1;
  __REG32 DMASTAT24             : 1;
  __REG32 DMASTAT25             : 1;
  __REG32 DMASTAT26             : 1;
  __REG32 DMASTAT27             : 1;
  __REG32 DMASTAT28             : 1;
  __REG32 DMASTAT29             : 1;
  __REG32 DMASTAT30             : 1;
  __REG32 DMASTAT31             : 1;
} __dmastat_bits;

/* DMA HW Channel Enable Set and Status Register (HWCHENAS) */
typedef struct {
  __REG32 HWCHENAS0              : 1;
  __REG32 HWCHENAS1              : 1;
  __REG32 HWCHENAS2              : 1;
  __REG32 HWCHENAS3              : 1;
  __REG32 HWCHENAS4              : 1;
  __REG32 HWCHENAS5              : 1;
  __REG32 HWCHENAS6              : 1;
  __REG32 HWCHENAS7              : 1;
  __REG32 HWCHENAS8              : 1;
  __REG32 HWCHENAS9              : 1;
  __REG32 HWCHENAS10             : 1;
  __REG32 HWCHENAS11             : 1;
  __REG32 HWCHENAS12             : 1;
  __REG32 HWCHENAS13             : 1;
  __REG32 HWCHENAS14             : 1;
  __REG32 HWCHENAS15             : 1;
  __REG32 HWCHENAS16             : 1;
  __REG32 HWCHENAS17             : 1;
  __REG32 HWCHENAS18             : 1;
  __REG32 HWCHENAS19             : 1;
  __REG32 HWCHENAS20             : 1;
  __REG32 HWCHENAS21             : 1;
  __REG32 HWCHENAS22             : 1;
  __REG32 HWCHENAS23             : 1;
  __REG32 HWCHENAS24             : 1;
  __REG32 HWCHENAS25             : 1;
  __REG32 HWCHENAS26             : 1;
  __REG32 HWCHENAS27             : 1;
  __REG32 HWCHENAS28             : 1;
  __REG32 HWCHENAS29             : 1;
  __REG32 HWCHENAS30             : 1;
  __REG32 HWCHENAS31             : 1;
} __dmahwchenas_bits;

/* DMA HW Channel Enable Reset and Status Register (HWCHENAR) */
typedef struct {
  __REG32 HWCHENAR0              : 1;
  __REG32 HWCHENAR1              : 1;
  __REG32 HWCHENAR2              : 1;
  __REG32 HWCHENAR3              : 1;
  __REG32 HWCHENAR4              : 1;
  __REG32 HWCHENAR5              : 1;
  __REG32 HWCHENAR6              : 1;
  __REG32 HWCHENAR7              : 1;
  __REG32 HWCHENAR8              : 1;
  __REG32 HWCHENAR9              : 1;
  __REG32 HWCHENAR10             : 1;
  __REG32 HWCHENAR11             : 1;
  __REG32 HWCHENAR12             : 1;
  __REG32 HWCHENAR13             : 1;
  __REG32 HWCHENAR14             : 1;
  __REG32 HWCHENAR15             : 1;
  __REG32 HWCHENAR16             : 1;
  __REG32 HWCHENAR17             : 1;
  __REG32 HWCHENAR18             : 1;
  __REG32 HWCHENAR19             : 1;
  __REG32 HWCHENAR20             : 1;
  __REG32 HWCHENAR21             : 1;
  __REG32 HWCHENAR22             : 1;
  __REG32 HWCHENAR23             : 1;
  __REG32 HWCHENAR24             : 1;
  __REG32 HWCHENAR25             : 1;
  __REG32 HWCHENAR26             : 1;
  __REG32 HWCHENAR27             : 1;
  __REG32 HWCHENAR28             : 1;
  __REG32 HWCHENAR29             : 1;
  __REG32 HWCHENAR30             : 1;
  __REG32 HWCHENAR31             : 1;
} __dmahwchenar_bits;

/* DMA SW Channel Enable Set and Status Register (SWCHENAS) */
typedef struct {
  __REG32 SWCHENA0              : 1;
  __REG32 SWCHENA1              : 1;
  __REG32 SWCHENA2              : 1;
  __REG32 SWCHENA3              : 1;
  __REG32 SWCHENA4              : 1;
  __REG32 SWCHENA5              : 1;
  __REG32 SWCHENA6              : 1;
  __REG32 SWCHENA7              : 1;
  __REG32 SWCHENA8              : 1;
  __REG32 SWCHENA9              : 1;
  __REG32 SWCHENA10             : 1;
  __REG32 SWCHENA11             : 1;
  __REG32 SWCHENA12             : 1;
  __REG32 SWCHENA13             : 1;
  __REG32 SWCHENA14             : 1;
  __REG32 SWCHENA15             : 1;
  __REG32 SWCHENA16             : 1;
  __REG32 SWCHENA17             : 1;
  __REG32 SWCHENA18             : 1;
  __REG32 SWCHENA19             : 1;
  __REG32 SWCHENA20             : 1;
  __REG32 SWCHENA21             : 1;
  __REG32 SWCHENA22             : 1;
  __REG32 SWCHENA23             : 1;
  __REG32 SWCHENA24             : 1;
  __REG32 SWCHENA25             : 1;
  __REG32 SWCHENA26             : 1;
  __REG32 SWCHENA27             : 1;
  __REG32 SWCHENA28             : 1;
  __REG32 SWCHENA29             : 1;
  __REG32 SWCHENA30             : 1;
  __REG32 SWCHENA31             : 1;
} __dmaswchenas_bits;

/* DMA SW Channel Enable Reset and Status Register (SWCHENAR) */
typedef struct {
  __REG32 SWCHDIS0              : 1;
  __REG32 SWCHDIS1              : 1;
  __REG32 SWCHDIS2              : 1;
  __REG32 SWCHDIS3              : 1;
  __REG32 SWCHDIS4              : 1;
  __REG32 SWCHDIS5              : 1;
  __REG32 SWCHDIS6              : 1;
  __REG32 SWCHDIS7              : 1;
  __REG32 SWCHDIS8              : 1;
  __REG32 SWCHDIS9              : 1;
  __REG32 SWCHDIS10             : 1;
  __REG32 SWCHDIS11             : 1;
  __REG32 SWCHDIS12             : 1;
  __REG32 SWCHDIS13             : 1;
  __REG32 SWCHDIS14             : 1;
  __REG32 SWCHDIS15             : 1;
  __REG32 SWCHDIS16             : 1;
  __REG32 SWCHDIS17             : 1;
  __REG32 SWCHDIS18             : 1;
  __REG32 SWCHDIS19             : 1;
  __REG32 SWCHDIS20             : 1;
  __REG32 SWCHDIS21             : 1;
  __REG32 SWCHDIS22             : 1;
  __REG32 SWCHDIS23             : 1;
  __REG32 SWCHDIS24             : 1;
  __REG32 SWCHDIS25             : 1;
  __REG32 SWCHDIS26             : 1;
  __REG32 SWCHDIS27             : 1;
  __REG32 SWCHDIS28             : 1;
  __REG32 SWCHDIS29             : 1;
  __REG32 SWCHDIS30             : 1;
  __REG32 SWCHDIS31             : 1;
} __dmaswchenar_bits;

/* DMA Channel Priority Set Register (CHPRIOS) */
typedef struct {
  __REG32 CPS0              : 1;
  __REG32 CPS1              : 1;
  __REG32 CPS2              : 1;
  __REG32 CPS3              : 1;
  __REG32 CPS4              : 1;
  __REG32 CPS5              : 1;
  __REG32 CPS6              : 1;
  __REG32 CPS7              : 1;
  __REG32 CPS8              : 1;
  __REG32 CPS9              : 1;
  __REG32 CPS10             : 1;
  __REG32 CPS11             : 1;
  __REG32 CPS12             : 1;
  __REG32 CPS13             : 1;
  __REG32 CPS14             : 1;
  __REG32 CPS15             : 1;
  __REG32 CPS16             : 1;
  __REG32 CPS17             : 1;
  __REG32 CPS18             : 1;
  __REG32 CPS19             : 1;
  __REG32 CPS20             : 1;
  __REG32 CPS21             : 1;
  __REG32 CPS22             : 1;
  __REG32 CPS23             : 1;
  __REG32 CPS24             : 1;
  __REG32 CPS25             : 1;
  __REG32 CPS26             : 1;
  __REG32 CPS27             : 1;
  __REG32 CPS28             : 1;
  __REG32 CPS29             : 1;
  __REG32 CPS30             : 1;
  __REG32 CPS31             : 1;
} __dmachprios_bits;

/* DMA Channel Priority Reset Register (CHPRIOR) */
typedef struct {
  __REG32 CPR0              : 1;
  __REG32 CPR1              : 1;
  __REG32 CPR2              : 1;
  __REG32 CPR3              : 1;
  __REG32 CPR4              : 1;
  __REG32 CPR5              : 1;
  __REG32 CPR6              : 1;
  __REG32 CPR7              : 1;
  __REG32 CPR8              : 1;
  __REG32 CPR9              : 1;
  __REG32 CPR10             : 1;
  __REG32 CPR11             : 1;
  __REG32 CPR12             : 1;
  __REG32 CPR13             : 1;
  __REG32 CPR14             : 1;
  __REG32 CPR15             : 1;
  __REG32 CPR16             : 1;
  __REG32 CPR17             : 1;
  __REG32 CPR18             : 1;
  __REG32 CPR19             : 1;
  __REG32 CPR20             : 1;
  __REG32 CPR21             : 1;
  __REG32 CPR22             : 1;
  __REG32 CPR23             : 1;
  __REG32 CPR24             : 1;
  __REG32 CPR25             : 1;
  __REG32 CPR26             : 1;
  __REG32 CPR27             : 1;
  __REG32 CPR28             : 1;
  __REG32 CPR29             : 1;
  __REG32 CPR30             : 1;
  __REG32 CPR31             : 1;
} __dmachprior_bits;

/* DMA Global Channel Interrupt Enable Set Register (GCHIENAS) */
typedef struct {
  __REG32 GCHIE0              : 1;
  __REG32 GCHIE1              : 1;
  __REG32 GCHIE2              : 1;
  __REG32 GCHIE3              : 1;
  __REG32 GCHIE4              : 1;
  __REG32 GCHIE5              : 1;
  __REG32 GCHIE6              : 1;
  __REG32 GCHIE7              : 1;
  __REG32 GCHIE8              : 1;
  __REG32 GCHIE9              : 1;
  __REG32 GCHIE10             : 1;
  __REG32 GCHIE11             : 1;
  __REG32 GCHIE12             : 1;
  __REG32 GCHIE13             : 1;
  __REG32 GCHIE14             : 1;
  __REG32 GCHIE15             : 1;
  __REG32 GCHIE16             : 1;
  __REG32 GCHIE17             : 1;
  __REG32 GCHIE18             : 1;
  __REG32 GCHIE19             : 1;
  __REG32 GCHIE20             : 1;
  __REG32 GCHIE21             : 1;
  __REG32 GCHIE22             : 1;
  __REG32 GCHIE23             : 1;
  __REG32 GCHIE24             : 1;
  __REG32 GCHIE25             : 1;
  __REG32 GCHIE26             : 1;
  __REG32 GCHIE27             : 1;
  __REG32 GCHIE28             : 1;
  __REG32 GCHIE29             : 1;
  __REG32 GCHIE30             : 1;
  __REG32 GCHIE31             : 1;
} __dmagchienas_bits;

/* DMA Global Channel Interrupt Enable Reset Register (GCHIENAR) */
typedef struct {
  __REG32 GCHID0              : 1;
  __REG32 GCHID1              : 1;
  __REG32 GCHID2              : 1;
  __REG32 GCHID3              : 1;
  __REG32 GCHID4              : 1;
  __REG32 GCHID5              : 1;
  __REG32 GCHID6              : 1;
  __REG32 GCHID7              : 1;
  __REG32 GCHID8              : 1;
  __REG32 GCHID9              : 1;
  __REG32 GCHID10             : 1;
  __REG32 GCHID11             : 1;
  __REG32 GCHID12             : 1;
  __REG32 GCHID13             : 1;
  __REG32 GCHID14             : 1;
  __REG32 GCHID15             : 1;
  __REG32 GCHID16             : 1;
  __REG32 GCHID17             : 1;
  __REG32 GCHID18             : 1;
  __REG32 GCHID19             : 1;
  __REG32 GCHID20             : 1;
  __REG32 GCHID21             : 1;
  __REG32 GCHID22             : 1;
  __REG32 GCHID23             : 1;
  __REG32 GCHID24             : 1;
  __REG32 GCHID25             : 1;
  __REG32 GCHID26             : 1;
  __REG32 GCHID27             : 1;
  __REG32 GCHID28             : 1;
  __REG32 GCHID29             : 1;
  __REG32 GCHID30             : 1;
  __REG32 GCHID31             : 1;
} __dmagchienar_bits;

/* DMA Request Assignment Register 0 (DREQASI0) */
typedef struct {
  __REG32 CH3ASI              : 6;
  __REG32                     : 2;
  __REG32 CH2ASI              : 6;
  __REG32                     : 2;
  __REG32 CH1ASI              : 6;
  __REG32                     : 2;
  __REG32 CH0ASI              : 6;
  __REG32                     : 2;
} __dmadreqasi0_bits;

/* DMA Request Assignment Register 1 (DREQASI1) */
typedef struct {
  __REG32 CH7ASI              : 6;
  __REG32                     : 2;
  __REG32 CH6ASI              : 6;
  __REG32                     : 2;
  __REG32 CH5ASI              : 6;
  __REG32                     : 2;
  __REG32 CH4ASI              : 6;
  __REG32                     : 2;
} __dmadreqasi1_bits;

/* DMA Request Assignment Register 2 (DREQASI2) */
typedef struct {
  __REG32 CH11ASI             : 6;
  __REG32                     : 2;
  __REG32 CH10ASI             : 6;
  __REG32                     : 2;
  __REG32 CH9ASI              : 6;
  __REG32                     : 2;
  __REG32 CH8ASI              : 6;
  __REG32                     : 2;
} __dmadreqasi2_bits;

/* DMA Request Assignment Register 3 (DREQASI3) */
typedef struct {
  __REG32 CH15ASI             : 6;
  __REG32                     : 2;
  __REG32 CH14ASI             : 6;
  __REG32                     : 2;
  __REG32 CH13ASI             : 6;
  __REG32                     : 2;
  __REG32 CH12ASI             : 6;
  __REG32                     : 2;
} __dmadreqasi3_bits;

/* DMA Request Assignment Register 4 (DREQASI4) */
typedef struct {
  __REG32 CH19ASI             : 6;
  __REG32                     : 2;
  __REG32 CH18ASI             : 6;
  __REG32                     : 2;
  __REG32 CH17ASI             : 6;
  __REG32                     : 2;
  __REG32 CH16ASI             : 6;
  __REG32                     : 2;
} __dmadreqasi4_bits;

/* DMA Request Assignment Register 5 (DREQASI5) */
typedef struct {
  __REG32 CH23ASI             : 6;
  __REG32                     : 2;
  __REG32 CH22ASI             : 6;
  __REG32                     : 2;
  __REG32 CH21ASI             : 6;
  __REG32                     : 2;
  __REG32 CH20ASI             : 6;
  __REG32                     : 2;
} __dmadreqasi5_bits;

/* DMA Request Assignment Register 6 (DREQASI6) */
typedef struct {
  __REG32 CH27ASI             : 6;
  __REG32                     : 2;
  __REG32 CH26ASI             : 6;
  __REG32                     : 2;
  __REG32 CH25ASI             : 6;
  __REG32                     : 2;
  __REG32 CH24ASI             : 6;
  __REG32                     : 2;
} __dmadreqasi6_bits;

/* DMA Request Assignment Register 7 (DREQASI7) */
typedef struct {
  __REG32 CH31ASI             : 6;
  __REG32                     : 2;
  __REG32 CH30ASI             : 6;
  __REG32                     : 2;
  __REG32 CH29ASI             : 6;
  __REG32                     : 2;
  __REG32 CH28ASI             : 6;
  __REG32                     : 2;
} __dmadreqasi7_bits;

/* Port Assignment Register 0 (PAR0) */
typedef struct {
  __REG32 CH7PA               : 3;
  __REG32                     : 1;
  __REG32 CH6PA               : 3;
  __REG32                     : 1;
  __REG32 CH5PA               : 3;
  __REG32                     : 1;
  __REG32 CH4PA               : 3;
  __REG32                     : 1;
  __REG32 CH3PA               : 3;
  __REG32                     : 1;
  __REG32 CH2PA               : 3;
  __REG32                     : 1;
  __REG32 CH1PA               : 3;
  __REG32                     : 1;
  __REG32 CH0PA               : 3;
  __REG32                     : 1;
} __dmapar0_bits;

/* Port Assignment Register 1 (PAR1) */
typedef struct {
  __REG32 CH15PA              : 3;
  __REG32                     : 1;
  __REG32 CH14PA              : 3;
  __REG32                     : 1;
  __REG32 CH13PA              : 3;
  __REG32                     : 1;
  __REG32 CH12PA              : 3;
  __REG32                     : 1;
  __REG32 CH11PA              : 3;
  __REG32                     : 1;
  __REG32 CH10PA              : 3;
  __REG32                     : 1;
  __REG32 CH9PA               : 3;
  __REG32                     : 1;
  __REG32 CH8PA               : 3;
  __REG32                     : 1;
} __dmapar1_bits;

/* Port Assignment Register 2 (PAR2) */
typedef struct {
  __REG32 CH23PA              : 3;
  __REG32                     : 1;
  __REG32 CH22PA              : 3;
  __REG32                     : 1;
  __REG32 CH21PA              : 3;
  __REG32                     : 1;
  __REG32 CH20PA              : 3;
  __REG32                     : 1;
  __REG32 CH19PA              : 3;
  __REG32                     : 1;
  __REG32 CH18PA              : 3;
  __REG32                     : 1;
  __REG32 CH17PA              : 3;
  __REG32                     : 1;
  __REG32 CH16PA              : 3;
  __REG32                     : 1;
} __dmapar2_bits;

/* Port Assignment Register 3 (PAR3) */
typedef struct {
  __REG32 CH31PA              : 3;
  __REG32                     : 1;
  __REG32 CH30PA              : 3;
  __REG32                     : 1;
  __REG32 CH29PA              : 3;
  __REG32                     : 1;
  __REG32 CH28PA              : 3;
  __REG32                     : 1;
  __REG32 CH27PA              : 3;
  __REG32                     : 1;
  __REG32 CH26PA              : 3;
  __REG32                     : 1;
  __REG32 CH25PA              : 3;
  __REG32                     : 1;
  __REG32 CH24PA              : 3;
  __REG32                     : 1;
} __dmapar3_bits;

/* DMA FTC Interrupt Mapping Register (FTCMAP) */
typedef struct {
  __REG32 FTCAB0              : 1;
  __REG32 FTCAB1              : 1;
  __REG32 FTCAB2              : 1;
  __REG32 FTCAB3              : 1;
  __REG32 FTCAB4              : 1;
  __REG32 FTCAB5              : 1;
  __REG32 FTCAB6              : 1;
  __REG32 FTCAB7              : 1;
  __REG32 FTCAB8              : 1;
  __REG32 FTCAB9              : 1;
  __REG32 FTCAB10             : 1;
  __REG32 FTCAB11             : 1;
  __REG32 FTCAB12             : 1;
  __REG32 FTCAB13             : 1;
  __REG32 FTCAB14             : 1;
  __REG32 FTCAB15             : 1;
  __REG32 FTCAB16             : 1;
  __REG32 FTCAB17             : 1;
  __REG32 FTCAB18             : 1;
  __REG32 FTCAB19             : 1;
  __REG32 FTCAB20             : 1;
  __REG32 FTCAB21             : 1;
  __REG32 FTCAB22             : 1;
  __REG32 FTCAB23             : 1;
  __REG32 FTCAB24             : 1;
  __REG32 FTCAB25             : 1;
  __REG32 FTCAB26             : 1;
  __REG32 FTCAB27             : 1;
  __REG32 FTCAB28             : 1;
  __REG32 FTCAB29             : 1;
  __REG32 FTCAB30             : 1;
  __REG32 FTCAB31             : 1;
} __dmaftcmap_bits;

/* DMA LFS Interrupt Mapping Register (LFSMAP) */
typedef struct {
  __REG32 LFSAB0              : 1;
  __REG32 LFSAB1              : 1;
  __REG32 LFSAB2              : 1;
  __REG32 LFSAB3              : 1;
  __REG32 LFSAB4              : 1;
  __REG32 LFSAB5              : 1;
  __REG32 LFSAB6              : 1;
  __REG32 LFSAB7              : 1;
  __REG32 LFSAB8              : 1;
  __REG32 LFSAB9              : 1;
  __REG32 LFSAB10             : 1;
  __REG32 LFSAB11             : 1;
  __REG32 LFSAB12             : 1;
  __REG32 LFSAB13             : 1;
  __REG32 LFSAB14             : 1;
  __REG32 LFSAB15             : 1;
  __REG32 LFSAB16             : 1;
  __REG32 LFSAB17             : 1;
  __REG32 LFSAB18             : 1;
  __REG32 LFSAB19             : 1;
  __REG32 LFSAB20             : 1;
  __REG32 LFSAB21             : 1;
  __REG32 LFSAB22             : 1;
  __REG32 LFSAB23             : 1;
  __REG32 LFSAB24             : 1;
  __REG32 LFSAB25             : 1;
  __REG32 LFSAB26             : 1;
  __REG32 LFSAB27             : 1;
  __REG32 LFSAB28             : 1;
  __REG32 LFSAB29             : 1;
  __REG32 LFSAB30             : 1;
  __REG32 LFSAB31             : 1;
} __dmalfsmap_bits;

/* DMA HBC Interrupt Mapping Register (HBCMAP) */
typedef struct {
  __REG32 HBCAB0              : 1;
  __REG32 HBCAB1              : 1;
  __REG32 HBCAB2              : 1;
  __REG32 HBCAB3              : 1;
  __REG32 HBCAB4              : 1;
  __REG32 HBCAB5              : 1;
  __REG32 HBCAB6              : 1;
  __REG32 HBCAB7              : 1;
  __REG32 HBCAB8              : 1;
  __REG32 HBCAB9              : 1;
  __REG32 HBCAB10             : 1;
  __REG32 HBCAB11             : 1;
  __REG32 HBCAB12             : 1;
  __REG32 HBCAB13             : 1;
  __REG32 HBCAB14             : 1;
  __REG32 HBCAB15             : 1;
  __REG32 HBCAB16             : 1;
  __REG32 HBCAB17             : 1;
  __REG32 HBCAB18             : 1;
  __REG32 HBCAB19             : 1;
  __REG32 HBCAB20             : 1;
  __REG32 HBCAB21             : 1;
  __REG32 HBCAB22             : 1;
  __REG32 HBCAB23             : 1;
  __REG32 HBCAB24             : 1;
  __REG32 HBCAB25             : 1;
  __REG32 HBCAB26             : 1;
  __REG32 HBCAB27             : 1;
  __REG32 HBCAB28             : 1;
  __REG32 HBCAB29             : 1;
  __REG32 HBCAB30             : 1;
  __REG32 HBCAB31             : 1;
} __dmahbcmap_bits;

/* DMA BTC Interrupt Mapping Register (BTCMAP) */
typedef struct {
  __REG32 BTCAB0              : 1;
  __REG32 BTCAB1              : 1;
  __REG32 BTCAB2              : 1;
  __REG32 BTCAB3              : 1;
  __REG32 BTCAB4              : 1;
  __REG32 BTCAB5              : 1;
  __REG32 BTCAB6              : 1;
  __REG32 BTCAB7              : 1;
  __REG32 BTCAB8              : 1;
  __REG32 BTCAB9              : 1;
  __REG32 BTCAB10             : 1;
  __REG32 BTCAB11             : 1;
  __REG32 BTCAB12             : 1;
  __REG32 BTCAB13             : 1;
  __REG32 BTCAB14             : 1;
  __REG32 BTCAB15             : 1;
  __REG32 BTCAB16             : 1;
  __REG32 BTCAB17             : 1;
  __REG32 BTCAB18             : 1;
  __REG32 BTCAB19             : 1;
  __REG32 BTCAB20             : 1;
  __REG32 BTCAB21             : 1;
  __REG32 BTCAB22             : 1;
  __REG32 BTCAB23             : 1;
  __REG32 BTCAB24             : 1;
  __REG32 BTCAB25             : 1;
  __REG32 BTCAB26             : 1;
  __REG32 BTCAB27             : 1;
  __REG32 BTCAB28             : 1;
  __REG32 BTCAB29             : 1;
  __REG32 BTCAB30             : 1;
  __REG32 BTCAB31             : 1;
} __dmabtcmap_bits;

/* DMA BER Interrupt Mapping Register (BERMAP) */
typedef struct {
  __REG32 BERAB0              : 1;
  __REG32 BERAB1              : 1;
  __REG32 BERAB2              : 1;
  __REG32 BERAB3              : 1;
  __REG32 BERAB4              : 1;
  __REG32 BERAB5              : 1;
  __REG32 BERAB6              : 1;
  __REG32 BERAB7              : 1;
  __REG32 BERAB8              : 1;
  __REG32 BERAB9              : 1;
  __REG32 BERAB10             : 1;
  __REG32 BERAB11             : 1;
  __REG32 BERAB12             : 1;
  __REG32 BERAB13             : 1;
  __REG32 BERAB14             : 1;
  __REG32 BERAB15             : 1;
  __REG32 BERAB16             : 1;
  __REG32 BERAB17             : 1;
  __REG32 BERAB18             : 1;
  __REG32 BERAB19             : 1;
  __REG32 BERAB20             : 1;
  __REG32 BERAB21             : 1;
  __REG32 BERAB22             : 1;
  __REG32 BERAB23             : 1;
  __REG32 BERAB24             : 1;
  __REG32 BERAB25             : 1;
  __REG32 BERAB26             : 1;
  __REG32 BERAB27             : 1;
  __REG32 BERAB28             : 1;
  __REG32 BERAB29             : 1;
  __REG32 BERAB30             : 1;
  __REG32 BERAB31             : 1;
} __dmabermap_bits;

/* DMA FTC Interrupt Enable Set (FTCINTENAS) */
typedef struct {
  __REG32 FTCINTENA0              : 1;
  __REG32 FTCINTENA1              : 1;
  __REG32 FTCINTENA2              : 1;
  __REG32 FTCINTENA3              : 1;
  __REG32 FTCINTENA4              : 1;
  __REG32 FTCINTENA5              : 1;
  __REG32 FTCINTENA6              : 1;
  __REG32 FTCINTENA7              : 1;
  __REG32 FTCINTENA8              : 1;
  __REG32 FTCINTENA9              : 1;
  __REG32 FTCINTENA10             : 1;
  __REG32 FTCINTENA11             : 1;
  __REG32 FTCINTENA12             : 1;
  __REG32 FTCINTENA13             : 1;
  __REG32 FTCINTENA14             : 1;
  __REG32 FTCINTENA15             : 1;
  __REG32 FTCINTENA16             : 1;
  __REG32 FTCINTENA17             : 1;
  __REG32 FTCINTENA18             : 1;
  __REG32 FTCINTENA19             : 1;
  __REG32 FTCINTENA20             : 1;
  __REG32 FTCINTENA21             : 1;
  __REG32 FTCINTENA22             : 1;
  __REG32 FTCINTENA23             : 1;
  __REG32 FTCINTENA24             : 1;
  __REG32 FTCINTENA25             : 1;
  __REG32 FTCINTENA26             : 1;
  __REG32 FTCINTENA27             : 1;
  __REG32 FTCINTENA28             : 1;
  __REG32 FTCINTENA29             : 1;
  __REG32 FTCINTENA30             : 1;
  __REG32 FTCINTENA31             : 1;
} __dmaftcintenas_bits;

/* DMA FTC Interrupt Enable Reset (FTCINTENAR) */
typedef struct {
  __REG32 FTCINTDIS0              : 1;
  __REG32 FTCINTDIS1              : 1;
  __REG32 FTCINTDIS2              : 1;
  __REG32 FTCINTDIS3              : 1;
  __REG32 FTCINTDIS4              : 1;
  __REG32 FTCINTDIS5              : 1;
  __REG32 FTCINTDIS6              : 1;
  __REG32 FTCINTDIS7              : 1;
  __REG32 FTCINTDIS8              : 1;
  __REG32 FTCINTDIS9              : 1;
  __REG32 FTCINTDIS10             : 1;
  __REG32 FTCINTDIS11             : 1;
  __REG32 FTCINTDIS12             : 1;
  __REG32 FTCINTDIS13             : 1;
  __REG32 FTCINTDIS14             : 1;
  __REG32 FTCINTDIS15             : 1;
  __REG32 FTCINTDIS16             : 1;
  __REG32 FTCINTDIS17             : 1;
  __REG32 FTCINTDIS18             : 1;
  __REG32 FTCINTDIS19             : 1;
  __REG32 FTCINTDIS20             : 1;
  __REG32 FTCINTDIS21             : 1;
  __REG32 FTCINTDIS22             : 1;
  __REG32 FTCINTDIS23             : 1;
  __REG32 FTCINTDIS24             : 1;
  __REG32 FTCINTDIS25             : 1;
  __REG32 FTCINTDIS26             : 1;
  __REG32 FTCINTDIS27             : 1;
  __REG32 FTCINTDIS28             : 1;
  __REG32 FTCINTDIS29             : 1;
  __REG32 FTCINTDIS30             : 1;
  __REG32 FTCINTDIS31             : 1;
} __dmaftcintenar_bits;

/* DMA LFS Interrupt Enable Set (LFSINTENAS) */
typedef struct {
  __REG32 LFSINTENA0              : 1;
  __REG32 LFSINTENA1              : 1;
  __REG32 LFSINTENA2              : 1;
  __REG32 LFSINTENA3              : 1;
  __REG32 LFSINTENA4              : 1;
  __REG32 LFSINTENA5              : 1;
  __REG32 LFSINTENA6              : 1;
  __REG32 LFSINTENA7              : 1;
  __REG32 LFSINTENA8              : 1;
  __REG32 LFSINTENA9              : 1;
  __REG32 LFSINTENA10             : 1;
  __REG32 LFSINTENA11             : 1;
  __REG32 LFSINTENA12             : 1;
  __REG32 LFSINTENA13             : 1;
  __REG32 LFSINTENA14             : 1;
  __REG32 LFSINTENA15             : 1;
  __REG32 LFSINTENA16             : 1;
  __REG32 LFSINTENA17             : 1;
  __REG32 LFSINTENA18             : 1;
  __REG32 LFSINTENA19             : 1;
  __REG32 LFSINTENA20             : 1;
  __REG32 LFSINTENA21             : 1;
  __REG32 LFSINTENA22             : 1;
  __REG32 LFSINTENA23             : 1;
  __REG32 LFSINTENA24             : 1;
  __REG32 LFSINTENA25             : 1;
  __REG32 LFSINTENA26             : 1;
  __REG32 LFSINTENA27             : 1;
  __REG32 LFSINTENA28             : 1;
  __REG32 LFSINTENA29             : 1;
  __REG32 LFSINTENA30             : 1;
  __REG32 LFSINTENA31             : 1;
} __dmalfsintenas_bits;

/* DMA LFS Interrupt Enable Reset (LFSINTENAR) */
typedef struct {
  __REG32 LFSINTDIS0              : 1;
  __REG32 LFSINTDIS1              : 1;
  __REG32 LFSINTDIS2              : 1;
  __REG32 LFSINTDIS3              : 1;
  __REG32 LFSINTDIS4              : 1;
  __REG32 LFSINTDIS5              : 1;
  __REG32 LFSINTDIS6              : 1;
  __REG32 LFSINTDIS7              : 1;
  __REG32 LFSINTDIS8              : 1;
  __REG32 LFSINTDIS9              : 1;
  __REG32 LFSINTDIS10             : 1;
  __REG32 LFSINTDIS11             : 1;
  __REG32 LFSINTDIS12             : 1;
  __REG32 LFSINTDIS13             : 1;
  __REG32 LFSINTDIS14             : 1;
  __REG32 LFSINTDIS15             : 1;
  __REG32 LFSINTDIS16             : 1;
  __REG32 LFSINTDIS17             : 1;
  __REG32 LFSINTDIS18             : 1;
  __REG32 LFSINTDIS19             : 1;
  __REG32 LFSINTDIS20             : 1;
  __REG32 LFSINTDIS21             : 1;
  __REG32 LFSINTDIS22             : 1;
  __REG32 LFSINTDIS23             : 1;
  __REG32 LFSINTDIS24             : 1;
  __REG32 LFSINTDIS25             : 1;
  __REG32 LFSINTDIS26             : 1;
  __REG32 LFSINTDIS27             : 1;
  __REG32 LFSINTDIS28             : 1;
  __REG32 LFSINTDIS29             : 1;
  __REG32 LFSINTDIS30             : 1;
  __REG32 LFSINTDIS31             : 1;
} __dmalfsintenar_bits;

/* DMA HBC Interrupt Enable Reset (HBCINTENAS) */
typedef struct {
  __REG32 HBCINTENA0              : 1;
  __REG32 HBCINTENA1              : 1;
  __REG32 HBCINTENA2              : 1;
  __REG32 HBCINTENA3              : 1;
  __REG32 HBCINTENA4              : 1;
  __REG32 HBCINTENA5              : 1;
  __REG32 HBCINTENA6              : 1;
  __REG32 HBCINTENA7              : 1;
  __REG32 HBCINTENA8              : 1;
  __REG32 HBCINTENA9              : 1;
  __REG32 HBCINTENA10             : 1;
  __REG32 HBCINTENA11             : 1;
  __REG32 HBCINTENA12             : 1;
  __REG32 HBCINTENA13             : 1;
  __REG32 HBCINTENA14             : 1;
  __REG32 HBCINTENA15             : 1;
  __REG32 HBCINTENA16             : 1;
  __REG32 HBCINTENA17             : 1;
  __REG32 HBCINTENA18             : 1;
  __REG32 HBCINTENA19             : 1;
  __REG32 HBCINTENA20             : 1;
  __REG32 HBCINTENA21             : 1;
  __REG32 HBCINTENA22             : 1;
  __REG32 HBCINTENA23             : 1;
  __REG32 HBCINTENA24             : 1;
  __REG32 HBCINTENA25             : 1;
  __REG32 HBCINTENA26             : 1;
  __REG32 HBCINTENA27             : 1;
  __REG32 HBCINTENA28             : 1;
  __REG32 HBCINTENA29             : 1;
  __REG32 HBCINTENA30             : 1;
  __REG32 HBCINTENA31             : 1;
} __dmahbcintenas_bits;

/* DMA HBC Interrupt Enable Reset (HBCINTENAR) */
typedef struct {
  __REG32 HBCINTDIS0              : 1;
  __REG32 HBCINTDIS1              : 1;
  __REG32 HBCINTDIS2              : 1;
  __REG32 HBCINTDIS3              : 1;
  __REG32 HBCINTDIS4              : 1;
  __REG32 HBCINTDIS5              : 1;
  __REG32 HBCINTDIS6              : 1;
  __REG32 HBCINTDIS7              : 1;
  __REG32 HBCINTDIS8              : 1;
  __REG32 HBCINTDIS9              : 1;
  __REG32 HBCINTDIS10             : 1;
  __REG32 HBCINTDIS11             : 1;
  __REG32 HBCINTDIS12             : 1;
  __REG32 HBCINTDIS13             : 1;
  __REG32 HBCINTDIS14             : 1;
  __REG32 HBCINTDIS15             : 1;
  __REG32 HBCINTDIS16             : 1;
  __REG32 HBCINTDIS17             : 1;
  __REG32 HBCINTDIS18             : 1;
  __REG32 HBCINTDIS19             : 1;
  __REG32 HBCINTDIS20             : 1;
  __REG32 HBCINTDIS21             : 1;
  __REG32 HBCINTDIS22             : 1;
  __REG32 HBCINTDIS23             : 1;
  __REG32 HBCINTDIS24             : 1;
  __REG32 HBCINTDIS25             : 1;
  __REG32 HBCINTDIS26             : 1;
  __REG32 HBCINTDIS27             : 1;
  __REG32 HBCINTDIS28             : 1;
  __REG32 HBCINTDIS29             : 1;
  __REG32 HBCINTDIS30             : 1;
  __REG32 HBCINTDIS31             : 1;
} __dmahbcintenar_bits;

/* DMA BTC Interrupt Enable Set (BTCINTENAS) */
typedef struct {
  __REG32 BTCINTENA0              : 1;
  __REG32 BTCINTENA1              : 1;
  __REG32 BTCINTENA2              : 1;
  __REG32 BTCINTENA3              : 1;
  __REG32 BTCINTENA4              : 1;
  __REG32 BTCINTENA5              : 1;
  __REG32 BTCINTENA6              : 1;
  __REG32 BTCINTENA7              : 1;
  __REG32 BTCINTENA8              : 1;
  __REG32 BTCINTENA9              : 1;
  __REG32 BTCINTENA10             : 1;
  __REG32 BTCINTENA11             : 1;
  __REG32 BTCINTENA12             : 1;
  __REG32 BTCINTENA13             : 1;
  __REG32 BTCINTENA14             : 1;
  __REG32 BTCINTENA15             : 1;
  __REG32 BTCINTENA16             : 1;
  __REG32 BTCINTENA17             : 1;
  __REG32 BTCINTENA18             : 1;
  __REG32 BTCINTENA19             : 1;
  __REG32 BTCINTENA20             : 1;
  __REG32 BTCINTENA21             : 1;
  __REG32 BTCINTENA22             : 1;
  __REG32 BTCINTENA23             : 1;
  __REG32 BTCINTENA24             : 1;
  __REG32 BTCINTENA25             : 1;
  __REG32 BTCINTENA26             : 1;
  __REG32 BTCINTENA27             : 1;
  __REG32 BTCINTENA28             : 1;
  __REG32 BTCINTENA29             : 1;
  __REG32 BTCINTENA30             : 1;
  __REG32 BTCINTENA31             : 1;
} __dmabtcintenas_bits;

/* DMA BTC Interrupt Enable Reset (BTCINTENAR) */
typedef struct {
  __REG32 BTCINTDIS0              : 1;
  __REG32 BTCINTDIS1              : 1;
  __REG32 BTCINTDIS2              : 1;
  __REG32 BTCINTDIS3              : 1;
  __REG32 BTCINTDIS4              : 1;
  __REG32 BTCINTDIS5              : 1;
  __REG32 BTCINTDIS6              : 1;
  __REG32 BTCINTDIS7              : 1;
  __REG32 BTCINTDIS8              : 1;
  __REG32 BTCINTDIS9              : 1;
  __REG32 BTCINTDIS10             : 1;
  __REG32 BTCINTDIS11             : 1;
  __REG32 BTCINTDIS12             : 1;
  __REG32 BTCINTDIS13             : 1;
  __REG32 BTCINTDIS14             : 1;
  __REG32 BTCINTDIS15             : 1;
  __REG32 BTCINTDIS16             : 1;
  __REG32 BTCINTDIS17             : 1;
  __REG32 BTCINTDIS18             : 1;
  __REG32 BTCINTDIS19             : 1;
  __REG32 BTCINTDIS20             : 1;
  __REG32 BTCINTDIS21             : 1;
  __REG32 BTCINTDIS22             : 1;
  __REG32 BTCINTDIS23             : 1;
  __REG32 BTCINTDIS24             : 1;
  __REG32 BTCINTDIS25             : 1;
  __REG32 BTCINTDIS26             : 1;
  __REG32 BTCINTDIS27             : 1;
  __REG32 BTCINTDIS28             : 1;
  __REG32 BTCINTDIS29             : 1;
  __REG32 BTCINTDIS30             : 1;
  __REG32 BTCINTDIS31             : 1;
} __dmabtcintenar_bits;

/* DMA Global Interrupt Flag Register (GINTFLAG) */
typedef struct {
  __REG32 GINT0              : 1;
  __REG32 GINT1              : 1;
  __REG32 GINT2              : 1;
  __REG32 GINT3              : 1;
  __REG32 GINT4              : 1;
  __REG32 GINT5              : 1;
  __REG32 GINT6              : 1;
  __REG32 GINT7              : 1;
  __REG32 GINT8              : 1;
  __REG32 GINT9              : 1;
  __REG32 GINT10             : 1;
  __REG32 GINT11             : 1;
  __REG32 GINT12             : 1;
  __REG32 GINT13             : 1;
  __REG32 GINT14             : 1;
  __REG32 GINT15             : 1;
  __REG32 GINT16             : 1;
  __REG32 GINT17             : 1;
  __REG32 GINT18             : 1;
  __REG32 GINT19             : 1;
  __REG32 GINT20             : 1;
  __REG32 GINT21             : 1;
  __REG32 GINT22             : 1;
  __REG32 GINT23             : 1;
  __REG32 GINT24             : 1;
  __REG32 GINT25             : 1;
  __REG32 GINT26             : 1;
  __REG32 GINT27             : 1;
  __REG32 GINT28             : 1;
  __REG32 GINT29             : 1;
  __REG32 GINT30             : 1;
  __REG32 GINT31             : 1;
} __dmagintflag_bits;

/* DMA FTC Interrupt Flag Register (FTCFLAG) */
typedef struct {
  __REG32 FTCI0              : 1;
  __REG32 FTCI1              : 1;
  __REG32 FTCI2              : 1;
  __REG32 FTCI3              : 1;
  __REG32 FTCI4              : 1;
  __REG32 FTCI5              : 1;
  __REG32 FTCI6              : 1;
  __REG32 FTCI7              : 1;
  __REG32 FTCI8              : 1;
  __REG32 FTCI9              : 1;
  __REG32 FTCI10             : 1;
  __REG32 FTCI11             : 1;
  __REG32 FTCI12             : 1;
  __REG32 FTCI13             : 1;
  __REG32 FTCI14             : 1;
  __REG32 FTCI15             : 1;
  __REG32 FTCI16             : 1;
  __REG32 FTCI17             : 1;
  __REG32 FTCI18             : 1;
  __REG32 FTCI19             : 1;
  __REG32 FTCI20             : 1;
  __REG32 FTCI21             : 1;
  __REG32 FTCI22             : 1;
  __REG32 FTCI23             : 1;
  __REG32 FTCI24             : 1;
  __REG32 FTCI25             : 1;
  __REG32 FTCI26             : 1;
  __REG32 FTCI27             : 1;
  __REG32 FTCI28             : 1;
  __REG32 FTCI29             : 1;
  __REG32 FTCI30             : 1;
  __REG32 FTCI31             : 1;
} __dmaftcflag_bits;

/* DMA LFS Interrupt Flag Register (LFSFLAG) */
typedef struct {
  __REG32 LFSI0              : 1;
  __REG32 LFSI1              : 1;
  __REG32 LFSI2              : 1;
  __REG32 LFSI3              : 1;
  __REG32 LFSI4              : 1;
  __REG32 LFSI5              : 1;
  __REG32 LFSI6              : 1;
  __REG32 LFSI7              : 1;
  __REG32 LFSI8              : 1;
  __REG32 LFSI9              : 1;
  __REG32 LFSI10             : 1;
  __REG32 LFSI11             : 1;
  __REG32 LFSI12             : 1;
  __REG32 LFSI13             : 1;
  __REG32 LFSI14             : 1;
  __REG32 LFSI15             : 1;
  __REG32 LFSI16             : 1;
  __REG32 LFSI17             : 1;
  __REG32 LFSI18             : 1;
  __REG32 LFSI19             : 1;
  __REG32 LFSI20             : 1;
  __REG32 LFSI21             : 1;
  __REG32 LFSI22             : 1;
  __REG32 LFSI23             : 1;
  __REG32 LFSI24             : 1;
  __REG32 LFSI25             : 1;
  __REG32 LFSI26             : 1;
  __REG32 LFSI27             : 1;
  __REG32 LFSI28             : 1;
  __REG32 LFSI29             : 1;
  __REG32 LFSI30             : 1;
  __REG32 LFSI31             : 1;
} __dmalfsflag_bits;

/* DMA HBC Interrupt Flag Register (HBCFLAG) */
typedef struct {
  __REG32 HBCI0              : 1;
  __REG32 HBCI1              : 1;
  __REG32 HBCI2              : 1;
  __REG32 HBCI3              : 1;
  __REG32 HBCI4              : 1;
  __REG32 HBCI5              : 1;
  __REG32 HBCI6              : 1;
  __REG32 HBCI7              : 1;
  __REG32 HBCI8              : 1;
  __REG32 HBCI9              : 1;
  __REG32 HBCI10             : 1;
  __REG32 HBCI11             : 1;
  __REG32 HBCI12             : 1;
  __REG32 HBCI13             : 1;
  __REG32 HBCI14             : 1;
  __REG32 HBCI15             : 1;
  __REG32 HBCI16             : 1;
  __REG32 HBCI17             : 1;
  __REG32 HBCI18             : 1;
  __REG32 HBCI19             : 1;
  __REG32 HBCI20             : 1;
  __REG32 HBCI21             : 1;
  __REG32 HBCI22             : 1;
  __REG32 HBCI23             : 1;
  __REG32 HBCI24             : 1;
  __REG32 HBCI25             : 1;
  __REG32 HBCI26             : 1;
  __REG32 HBCI27             : 1;
  __REG32 HBCI28             : 1;
  __REG32 HBCI29             : 1;
  __REG32 HBCI30             : 1;
  __REG32 HBCI31             : 1;
} __dmahbcflag_bits;

/* DMA BTC Interrupt Flag Register (BTCFLAG) */
typedef struct {
  __REG32 BTCI0              : 1;
  __REG32 BTCI1              : 1;
  __REG32 BTCI2              : 1;
  __REG32 BTCI3              : 1;
  __REG32 BTCI4              : 1;
  __REG32 BTCI5              : 1;
  __REG32 BTCI6              : 1;
  __REG32 BTCI7              : 1;
  __REG32 BTCI8              : 1;
  __REG32 BTCI9              : 1;
  __REG32 BTCI10             : 1;
  __REG32 BTCI11             : 1;
  __REG32 BTCI12             : 1;
  __REG32 BTCI13             : 1;
  __REG32 BTCI14             : 1;
  __REG32 BTCI15             : 1;
  __REG32 BTCI16             : 1;
  __REG32 BTCI17             : 1;
  __REG32 BTCI18             : 1;
  __REG32 BTCI19             : 1;
  __REG32 BTCI20             : 1;
  __REG32 BTCI21             : 1;
  __REG32 BTCI22             : 1;
  __REG32 BTCI23             : 1;
  __REG32 BTCI24             : 1;
  __REG32 BTCI25             : 1;
  __REG32 BTCI26             : 1;
  __REG32 BTCI27             : 1;
  __REG32 BTCI28             : 1;
  __REG32 BTCI29             : 1;
  __REG32 BTCI30             : 1;
  __REG32 BTCI31             : 1;
} __dmabtcflag_bits;

/* DMA BER Interrupt Flag Register (BERFLAG) */
typedef struct {
  __REG32 BERI0              : 1;
  __REG32 BERI1              : 1;
  __REG32 BERI2              : 1;
  __REG32 BERI3              : 1;
  __REG32 BERI4              : 1;
  __REG32 BERI5              : 1;
  __REG32 BERI6              : 1;
  __REG32 BERI7              : 1;
  __REG32 BERI8              : 1;
  __REG32 BERI9              : 1;
  __REG32 BERI10             : 1;
  __REG32 BERI11             : 1;
  __REG32 BERI12             : 1;
  __REG32 BERI13             : 1;
  __REG32 BERI14             : 1;
  __REG32 BERI15             : 1;
  __REG32 BERI16             : 1;
  __REG32 BERI17             : 1;
  __REG32 BERI18             : 1;
  __REG32 BERI19             : 1;
  __REG32 BERI20             : 1;
  __REG32 BERI21             : 1;
  __REG32 BERI22             : 1;
  __REG32 BERI23             : 1;
  __REG32 BERI24             : 1;
  __REG32 BERI25             : 1;
  __REG32 BERI26             : 1;
  __REG32 BERI27             : 1;
  __REG32 BERI28             : 1;
  __REG32 BERI29             : 1;
  __REG32 BERI30             : 1;
  __REG32 BERI31             : 1;
} __dmaberflag_bits;

/* DMA FTCA Interrupt Channel Offset Register (FTCAOFFSET) */
typedef struct {
  __REG32 FTCA               : 6;
  __REG32                    :26;
} __dmaftcaoffset_bits;

/* DMA LFSA Interrupt Channel Offset Register (LFSAOFFSET) */
typedef struct {
  __REG32 LFSA               : 6;
  __REG32                    :26;
} __dmalfsaoffset_bits;

/* DMA HBCA Interrupt Channel Offset Register (HBCAOFFSET) */
typedef struct {
  __REG32 HBCA               : 6;
  __REG32                    :26;
} __dmahbcaoffset_bits;

/* DMA BTCA Interrupt Channel Offset Register (BTCAOFFSET) */
typedef struct {
  __REG32 BTCA               : 6;
  __REG32                    :26;
} __dmabtcaoffset_bits;

/* DMA BERA Interrupt Channel Offset Register (BERAOFFSET) */
typedef struct {
  __REG32 BERA               : 6;
  __REG32                    :26;
} __dmaberaoffset_bits;

/* DMA FTCB Interrupt Channel Offset Register (FTCBOFFSET) */
typedef struct {
  __REG32 FTCB               : 6;
  __REG32                    :26;
} __dmaftcboffset_bits;

/* DMA FTCB Interrupt Channel Offset Register (FTCBOFFSET) */
typedef struct {
  __REG32 LFSB               : 6;
  __REG32                    :26;
} __dmalfsboffset_bits;

/* DMA HBCB Interrupt Channel Offset Register (HBCBOFFSET) */
typedef struct {
  __REG32 HBCB               : 6;
  __REG32                    :26;
} __dmahbcboffset_bits;


/* DMA BTCB Interrupt Channel Offset Register (BTCBOFFSET) */
typedef struct {
  __REG32 HBCB               : 6;
  __REG32                    :26;
} __dmabtcboffset_bits;

/* DMA BERB Interrupt Channel Offset Register (BERBOFFSET) */
typedef struct {
  __REG32 BERB               : 6;
  __REG32                    :26;
} __dmaberboffset_bits;

/* DMA Port Control Register (PTCRL) */
typedef struct {
  __REG32                    :16;
  __REG32 PSFRLQPB           : 1;
  __REG32 PSFRHQPB           : 1;
  __REG32 BYB                : 1;
  __REG32                    : 5;
  __REG32 PENDB              : 1;
  __REG32                    : 7;
} __dmaptcrl_bits;

/* DMA RAM Test Control (RTCTRL) */
typedef struct {
  __REG32 RTC                : 1;
  __REG32                    :31;
} __dmartctrl_bits;

/* DMA RAM Test Control (RTCTRL) */
typedef struct {
  __REG32 DBGEN              : 1;
  __REG32                    :15;
  __REG32 DMADBGS            : 1;
  __REG32                    : 7;
  __REG32 CHNUM              : 5;
  __REG32                    : 3;
} __dmadctrl_bits;

/* DMA Port B Active Channel Transfer Count Register (PBACTC) */
typedef struct {
  __REG32 PBETCOUNT          :13;
  __REG32                    : 3;
  __REG32 PBFTCOUNT          :13;
  __REG32                    : 3;
} __dmapbactc_bits;

/* DMA Parity Control Register (DMAPCR) */
typedef struct {
  __REG32 PARITY_ENA         : 4;
  __REG32                    : 4;
  __REG32 TEST               : 1;
  __REG32                    : 7;
  __REG32 ERRA               : 1;
  __REG32                    :15;
} __dmapcr_bits;

/* DMA Parity Error Address Register (DMAPAR) */
typedef struct {
  __REG32 ERRORADDRESS       :12;
  __REG32                    :12;
  __REG32 EDFLAG             : 1;
  __REG32                    : 7;
} __dmapar_bits;

/* DMA Memory Protection Control Register (DMAMPCTRL) */
typedef struct {
  __REG32 REG0ENA            : 1;
  __REG32 REG0AP             : 2;
  __REG32 INT0ENA            : 1;
  __REG32 INT0AB             : 1;
  __REG32                    : 3;
  __REG32 REG1ENA            : 1;
  __REG32 REG1AP             : 2;
  __REG32 INT1ENA            : 1;
  __REG32 INT1AB             : 1;
  __REG32                    : 3;
  __REG32 REG2ENA            : 1;
  __REG32 REG2AP             : 2;
  __REG32 INT2ENA            : 1;
  __REG32 INT2AB             : 1;
  __REG32                    : 3;
  __REG32 REG3ENA            : 1;
  __REG32 REG3AP             : 2;
  __REG32 INT3ENA            : 1;
  __REG32 INT3AB             : 1;
  __REG32                    : 3;
} __dmampctrl_bits;

/* DMA Memory Protection Status Register (DMAMPST) */
typedef struct {
  __REG32 REG0FT             : 1;
  __REG32                    : 7;
  __REG32 REG1FT             : 1;
  __REG32                    : 7;
  __REG32 REG2FT             : 1;
  __REG32                    : 7;
  __REG32 REG3FT             : 1;
  __REG32                    : 7;
} __dmampst_bits;

/* DMA Initial Transfer Count Register (ITCOUNT) */
typedef struct {
  __REG32 IETCOUNT           :13;
  __REG32                    : 3;
  __REG32 IFTCOUNT           :13;
  __REG32                    : 3;
} __dmacpitcount_bits;

/* DMA Channel Control Register (CHCTRL) */
typedef struct {
  __REG32 AIM                : 1;
  __REG32 ADDMW              : 2;
  __REG32 ADDMR              : 2;
  __REG32                    : 3;
  __REG32 TTYPE              : 1;
  __REG32                    : 3;
  __REG32 WES                : 2;
  __REG32 RES                : 2;
  __REG32 CHAIN              : 6;
  __REG32                    :10;
} __dmachctrl_bits;

/* DMA Element Index Offset Register (EIOFF) */
typedef struct {
  __REG32 EIDXS              :13;
  __REG32                    : 3;
  __REG32 EIDXD              :13;
  __REG32                    : 3;
} __dmacpeioff_bits;

/* DMA Frame Index Offset Register (FIOFF) */
typedef struct {
  __REG32 EIDXS              :13;
  __REG32                    : 3;
  __REG32 EIDXD              :13;
  __REG32                    : 3;
} __dmacpfioff_bits;

/* DMA Current Transfer Count Register (CTCOUNT) */
typedef struct {
  __REG32 CETCOUNT           :13;
  __REG32                    : 3;
  __REG32 CFTCOUNT           :13;
  __REG32                    : 3;
} __dmacpctcount_bits;

/* RTI Global Control Register (RTIGCTRL) */
typedef struct {
  __REG32 CNT0EN             : 1;
  __REG32 CNT1EN             : 1;
  __REG32                    :13;
  __REG32 COS                : 1;
  __REG32 NTUSEL             : 1;
  __REG32                    :15;
} __rtigctrl_bits;

/* RTI Timebase Control Register (RTITBCTRL) */
typedef struct {
  __REG32 TBEXT               : 1;
  __REG32 INC                 : 1;
  __REG32                     :30;
} __rtitbctrl_bits;

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

/* RTI Set Interrupt Enable Register (RTISETINTENA) */
typedef struct {
  __REG32 SETINT0             : 1;
  __REG32 SETINT1             : 1;
  __REG32 SETINT2             : 1;
  __REG32 SETINT3             : 1;
  __REG32                     : 4;
  __REG32 SETDMA0             : 1;
  __REG32 SETDMA1             : 1;
  __REG32 SETDMA2             : 1;
  __REG32 SETDMA3             : 1;
  __REG32                     : 4;
  __REG32 SETTBINT            : 1;
  __REG32 SETOVL0INT          : 1;
  __REG32 SETOVL1INT          : 1;
  __REG32                     :13;
} __rtisetintena_bits;

/* RTI Clear Interrupt Enable Register (RTICLEARINTENA) */
typedef struct {
  __REG32 CLEARINT0           : 1;
  __REG32 CLEARINT1           : 1;
  __REG32 CLEARINT2           : 1;
  __REG32 CLEARINT3           : 1;
  __REG32                     : 4;
  __REG32 CLEARDMA0           : 1;
  __REG32 CLEARDMA1           : 1;
  __REG32 CLEARDMA2           : 1;
  __REG32 CLEARDMA3           : 1;
  __REG32                     : 4;
  __REG32 SETTBINT            : 1;
  __REG32 SETOVL0INT          : 1;
  __REG32 SETOVL1INT          : 1;
  __REG32                     :13;
} __rticlearintena_bits;

/* RTI Interrupt Flag Register (RTIINTFLAG) */
typedef struct {
  __REG32 INT0                : 1;
  __REG32 INT1                : 1;
  __REG32 INT2                : 1;
  __REG32 INT3                : 1;
  __REG32                     :12;
  __REG32 TBINT               : 1;
  __REG32 OVL0INT             : 1;
  __REG32 OVL1INT             : 1;
  __REG32                     :13;
} __rtiintflag_bits;

/* CRC_CTRL0: CRC Global Control Register 0 */
typedef struct {
  __REG32 CH1_PSA_SWREST      : 1;
  __REG32                     : 7;
  __REG32 CH2_PSA_SWREST      : 1;
  __REG32                     : 7;
  __REG32 CH3_PSA_SWREST      : 1;
  __REG32                     : 7;
  __REG32 CH4_PSA_SWREST      : 1;
  __REG32                     : 7;
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
  __REG32                     : 3;
  __REG32 CH2_MODE            : 2;
  __REG32                     : 6;
  __REG32 CH3_MODE            : 2;
  __REG32                     : 6;
  __REG32 CH4_MODE            : 2;
  __REG32                     : 6;
} __crc_ctrl2_bits;

/* CRC_INTS: CRC Interrupt Enable Set Register */
typedef struct {
  __REG32 CH1_CCITENS         : 1;
  __REG32 CH1_CRCFAILENS      : 1;
  __REG32 CH1_OVERENS         : 1;
  __REG32 CH1_UNDERENS        : 1;
  __REG32 CH1_TIMEOUTENS      : 1;
  __REG32                     : 3;
  __REG32 CH2_CCITENS         : 1;
  __REG32 CH2_CRCFAILENS      : 1;
  __REG32 CH2_OVERENS         : 1;
  __REG32 CH2_UNDERENS        : 1;
  __REG32 CH2_TIMEOUTENS      : 1;
  __REG32                     : 3;
  __REG32 CH3_CCITENS         : 1;
  __REG32 CH3_CRCFAILENS      : 1;
  __REG32 CH3_OVERENS         : 1;
  __REG32 CH3_UNDERENS        : 1;
  __REG32 CH3_TIMEOUTENS      : 1;
  __REG32                     : 3;
  __REG32 CH4_CCITENS         : 1;
  __REG32 CH4_CRCFAILENS      : 1;
  __REG32 CH4_OVERENS         : 1;
  __REG32 CH4_UNDERENS        : 1;
  __REG32 CH4_TIMEOUTENS      : 1;
  __REG32                     : 3;
} __crc_ints_bits;

/* CRC_INTR: CRC Interrupt Enable Reset Register */
typedef struct {
  __REG32 CH1_CCITENR         : 1;
  __REG32 CH1_CRCFAILENR      : 1;
  __REG32 CH1_OVERENR         : 1;
  __REG32 CH1_UNDERENR        : 1;
  __REG32 CH1_TIMEOUTENR      : 1;
  __REG32                     : 3;
  __REG32 CH2_CCITENR         : 1;
  __REG32 CH2_CRCFAILENR      : 1;
  __REG32 CH2_OVERENR         : 1;
  __REG32 CH2_UNDERENR        : 1;
  __REG32 CH2_TIMEOUTENR      : 1;
  __REG32                     : 3;
  __REG32 CH3_CCITENR         : 1;
  __REG32 CH3_CRCFAILENR      : 1;
  __REG32 CH3_OVERENR         : 1;
  __REG32 CH3_UNDERENR        : 1;
  __REG32 CH3_TIMEOUTENR      : 1;
  __REG32                     : 3;
  __REG32 CH4_CCITENR         : 1;
  __REG32 CH4_CRCFAILENR      : 1;
  __REG32 CH4_OVERENR         : 1;
  __REG32 CH4_UNDERENR        : 1;
  __REG32 CH4_TIMEOUTENR      : 1;
  __REG32                     : 3;
} __crc_intr_bits;

/* CRC_STATUS_REG: CRC Interrupt Status Register */
typedef struct {
  __REG32 CH1_CCIT            : 1;
  __REG32 CH1_CRCFAIL         : 1;
  __REG32 CH1_OVER            : 1;
  __REG32 CH1_UNDER           : 1;
  __REG32 CH1_TIMEOUT         : 1;
  __REG32                     : 3;
  __REG32 CH2_CCIT            : 1;
  __REG32 CH2_CRCFAIL         : 1;
  __REG32 CH2_OVER            : 1;
  __REG32 CH2_UNDER           : 1;
  __REG32 CH2_TIMEOUT         : 1;
  __REG32                     : 3;
  __REG32 CH3_CCIT            : 1;
  __REG32 CH3_CRCFAIL         : 1;
  __REG32 CH3_OVER            : 1;
  __REG32 CH3_UNDER           : 1;
  __REG32 CH3_TIMEOUT         : 1;
  __REG32                     : 3;
  __REG32 CH4_CCIT            : 1;
  __REG32 CH4_CRCFAIL         : 1;
  __REG32 CH4_OVER            : 1;
  __REG32 CH4_UNDER           : 1;
  __REG32 CH4_TIMEOUT         : 1;
  __REG32                     : 3;
} __crc_status_bits;

/* CRC_INT_OFFSET_REG: CRC Interrupt Offset Register */
typedef struct {
  __REG32 OFSTREG             : 8;
  __REG32                     :24;
} __crc_int_offset_reg_bits;

/* CRC_BUSY: CRC Busy Register */
typedef struct {
  __REG32 CH1_BUSY            : 1;
  __REG32                     : 7;
  __REG32 CH2_BUSY            : 1;
  __REG32                     : 7;
  __REG32 CH3_BUSY            : 1;
  __REG32                     : 7;
  __REG32 CH4_BUSY            : 1;
  __REG32                     : 7;
} __crc_busy_bits;

/* CRC_PCOUNT_REGx: CRC Pattern Counter Preload Register x */
typedef struct {
  __REG32 CRC_PAT_COUNT       :20;
  __REG32                     :12;
} __crc_pcount_reg_bits;

/* CRC_SCOUNT_REGx: CRC Sector Counter Preload Register x */
typedef struct {
  __REG32 CRC_SEC_COUNT       :16;
  __REG32                     :16;
} __crc_scount_reg_bits;

/* CRC_CURSEC_REGx: CRC Current Sector Register x */
typedef struct {
  __REG32 CRC_CURSEC          :16;
  __REG32                     :16;
} __crc_cursec_reg_bits;

/* CRC_WDTOPLDx: CRC Channel x Watchdog Timeout Preload Register */
typedef struct {
  __REG32 CRC_WDTOPLD         :24;
  __REG32                     : 8;
} __crc_wdtopld_bits;

/* CRC_BCTOPLDx: CRC Channel x Block Complete Timeout Preload Register */
typedef struct {
  __REG32 CRC_BCTOPLD         :24;
  __REG32                     : 8;
} __crc_bctopld_bits;

/* MCRC_TRACE_BUS_SEL: Data bus selection register */
typedef struct {
  __REG32 ITCMEn              : 1;
  __REG32 DTCMEn              : 1;
  __REG32 MEn                 : 1;
  __REG32                     :29;
} __mcrc_bus_sel_bits;

/* CCM-R4F Status Register (CCMSR) */
typedef struct {
  __REG32 STE                 : 1;
  __REG32 STET                : 1;
  __REG32                     : 6;
  __REG32 STC                 : 1;
  __REG32                     : 7;
  __REG32 CPME                : 1;
  __REG32                     :15;
} __ccmsr_bits;

/* CCM-R4F Key Register (CCMKEYR) */
typedef struct {
  __REG32 MKEY                : 4;
  __REG32                     :28;
} __ccmkeyr_bits;

/* Parity Flag Register (PARFLG) */
typedef struct {
  __REG32 PARFLG              : 1;
  __REG32                     :31;
} __parflg_bits;

/* Parity Control Register (PARCTL) */
typedef struct {
  __REG32 PARENA              : 4;
  __REG32                     : 4;
  __REG32 TEST                : 1;
  __REG32                     :23;
} __parctl_bits;

/* IRQ Index Offset Vector Register (IRQINDEX) */
typedef struct {
  __REG32 IRQINDEX            : 8;
  __REG32                     :24;
} __irqindex_bits;

/* FIQ Index Offset Vector Registers (FIQINDEX) */
typedef struct {
  __REG32 FIQINDEX            : 8;
  __REG32                     :24;
} __fiqindex_bits;

/* FIQ/IRQ Program Control Register 0 (FIRQPR0) */
typedef struct {
  __REG32 FIQINDEX0           : 1;
  __REG32 FIQINDEX1           : 1;
  __REG32 FIQINDEX2           : 1;
  __REG32 FIQINDEX3           : 1;
  __REG32 FIQINDEX4           : 1;
  __REG32 FIQINDEX5           : 1;
  __REG32 FIQINDEX6           : 1;
  __REG32 FIQINDEX7           : 1;
  __REG32 FIQINDEX8           : 1;
  __REG32 FIQINDEX9           : 1;
  __REG32 FIQINDEX10          : 1;
  __REG32 FIQINDEX11          : 1;
  __REG32 FIQINDEX12          : 1;
  __REG32 FIQINDEX13          : 1;
  __REG32 FIQINDEX14          : 1;
  __REG32 FIQINDEX15          : 1;
  __REG32 FIQINDEX16          : 1;
  __REG32 FIQINDEX17          : 1;
  __REG32 FIQINDEX18          : 1;
  __REG32 FIQINDEX19          : 1;
  __REG32 FIQINDEX20          : 1;
  __REG32 FIQINDEX21          : 1;
  __REG32 FIQINDEX22          : 1;
  __REG32 FIQINDEX23          : 1;
  __REG32 FIQINDEX24          : 1;
  __REG32 FIQINDEX25          : 1;
  __REG32 FIQINDEX26          : 1;
  __REG32 FIQINDEX27          : 1;
  __REG32 FIQINDEX28          : 1;
  __REG32 FIQINDEX29          : 1;
  __REG32 FIQINDEX30          : 1;
  __REG32 FIQINDEX31          : 1;
} __firqpr0_bits;

/* FIQ/IRQ Program Control Register 1 (FIRQPR1) */
typedef struct {
  __REG32 FIQINDEX32          : 1;
  __REG32 FIQINDEX33          : 1;
  __REG32 FIQINDEX34          : 1;
  __REG32 FIQINDEX35          : 1;
  __REG32 FIQINDEX36          : 1;
  __REG32 FIQINDEX37          : 1;
  __REG32 FIQINDEX38          : 1;
  __REG32 FIQINDEX39          : 1;
  __REG32 FIQINDEX40          : 1;
  __REG32 FIQINDEX41          : 1;
  __REG32 FIQINDEX42          : 1;
  __REG32 FIQINDEX43          : 1;
  __REG32 FIQINDEX44          : 1;
  __REG32 FIQINDEX45          : 1;
  __REG32 FIQINDEX46          : 1;
  __REG32 FIQINDEX47          : 1;
  __REG32 FIQINDEX48          : 1;
  __REG32 FIQINDEX49          : 1;
  __REG32 FIQINDEX50          : 1;
  __REG32 FIQINDEX51          : 1;
  __REG32 FIQINDEX52          : 1;
  __REG32 FIQINDEX53          : 1;
  __REG32 FIQINDEX54          : 1;
  __REG32 FIQINDEX55          : 1;
  __REG32 FIQINDEX56          : 1;
  __REG32 FIQINDEX57          : 1;
  __REG32 FIQINDEX58          : 1;
  __REG32 FIQINDEX59          : 1;
  __REG32 FIQINDEX60          : 1;
  __REG32 FIQINDEX61          : 1;
  __REG32 FIQINDEX62          : 1;
  __REG32 FIQINDEX63          : 1;
} __firqpr1_bits;

/* Pending Interrupt Read Location Register 0 (INTREQ0) */
typedef struct {
  __REG32 INTREQ0           : 1;
  __REG32 INTREQ1           : 1;
  __REG32 INTREQ2           : 1;
  __REG32 INTREQ3           : 1;
  __REG32 INTREQ4           : 1;
  __REG32 INTREQ5           : 1;
  __REG32 INTREQ6           : 1;
  __REG32 INTREQ7           : 1;
  __REG32 INTREQ8           : 1;
  __REG32 INTREQ9           : 1;
  __REG32 INTREQ10          : 1;
  __REG32 INTREQ11          : 1;
  __REG32 INTREQ12          : 1;
  __REG32 INTREQ13          : 1;
  __REG32 INTREQ14          : 1;
  __REG32 INTREQ15          : 1;
  __REG32 INTREQ16          : 1;
  __REG32 INTREQ17          : 1;
  __REG32 INTREQ18          : 1;
  __REG32 INTREQ19          : 1;
  __REG32 INTREQ20          : 1;
  __REG32 INTREQ21          : 1;
  __REG32 INTREQ22          : 1;
  __REG32 INTREQ23          : 1;
  __REG32 INTREQ24          : 1;
  __REG32 INTREQ25          : 1;
  __REG32 INTREQ26          : 1;
  __REG32 INTREQ27          : 1;
  __REG32 INTREQ28          : 1;
  __REG32 INTREQ29          : 1;
  __REG32 INTREQ30          : 1;
  __REG32 INTREQ31          : 1;
} __intreq0_bits;

/* Pending Interrupt Read Location Register 1 (INTREQ1) */
typedef struct {
  __REG32 INTREQ32          : 1;
  __REG32 INTREQ33          : 1;
  __REG32 INTREQ34          : 1;
  __REG32 INTREQ35          : 1;
  __REG32 INTREQ36          : 1;
  __REG32 INTREQ37          : 1;
  __REG32 INTREQ38          : 1;
  __REG32 INTREQ39          : 1;
  __REG32 INTREQ40          : 1;
  __REG32 INTREQ41          : 1;
  __REG32 INTREQ42          : 1;
  __REG32 INTREQ43          : 1;
  __REG32 INTREQ44          : 1;
  __REG32 INTREQ45          : 1;
  __REG32 INTREQ46          : 1;
  __REG32 INTREQ47          : 1;
  __REG32 INTREQ48          : 1;
  __REG32 INTREQ49          : 1;
  __REG32 INTREQ50          : 1;
  __REG32 INTREQ51          : 1;
  __REG32 INTREQ52          : 1;
  __REG32 INTREQ53          : 1;
  __REG32 INTREQ54          : 1;
  __REG32 INTREQ55          : 1;
  __REG32 INTREQ56          : 1;
  __REG32 INTREQ57          : 1;
  __REG32 INTREQ58          : 1;
  __REG32 INTREQ59          : 1;
  __REG32 INTREQ60          : 1;
  __REG32 INTREQ61          : 1;
  __REG32 INTREQ62          : 1;
  __REG32 INTREQ63          : 1;
} __intreq1_bits;

/* Interrupt Enable Set Register 0 (REQENASET0) */
typedef struct {
  __REG32 REQENASET0           : 1;
  __REG32 REQENASET1           : 1;
  __REG32 REQENASET2           : 1;
  __REG32 REQENASET3           : 1;
  __REG32 REQENASET4           : 1;
  __REG32 REQENASET5           : 1;
  __REG32 REQENASET6           : 1;
  __REG32 REQENASET7           : 1;
  __REG32 REQENASET8           : 1;
  __REG32 REQENASET9           : 1;
  __REG32 REQENASET10          : 1;
  __REG32 REQENASET11          : 1;
  __REG32 REQENASET12          : 1;
  __REG32 REQENASET13          : 1;
  __REG32 REQENASET14          : 1;
  __REG32 REQENASET15          : 1;
  __REG32 REQENASET16          : 1;
  __REG32 REQENASET17          : 1;
  __REG32 REQENASET18          : 1;
  __REG32 REQENASET19          : 1;
  __REG32 REQENASET20          : 1;
  __REG32 REQENASET21          : 1;
  __REG32 REQENASET22          : 1;
  __REG32 REQENASET23          : 1;
  __REG32 REQENASET24          : 1;
  __REG32 REQENASET25          : 1;
  __REG32 REQENASET26          : 1;
  __REG32 REQENASET27          : 1;
  __REG32 REQENASET28          : 1;
  __REG32 REQENASET29          : 1;
  __REG32 REQENASET30          : 1;
  __REG32 REQENASET31          : 1;
} __reqenaset0_bits;

/* Interrupt Enable Set Register 1 (REQENASET1) */
typedef struct {
  __REG32 REQENASET32          : 1;
  __REG32 REQENASET33          : 1;
  __REG32 REQENASET34          : 1;
  __REG32 REQENASET35          : 1;
  __REG32 REQENASET36          : 1;
  __REG32 REQENASET37          : 1;
  __REG32 REQENASET38          : 1;
  __REG32 REQENASET39          : 1;
  __REG32 REQENASET40          : 1;
  __REG32 REQENASET41          : 1;
  __REG32 REQENASET42          : 1;
  __REG32 REQENASET43          : 1;
  __REG32 REQENASET44          : 1;
  __REG32 REQENASET45          : 1;
  __REG32 REQENASET46          : 1;
  __REG32 REQENASET47          : 1;
  __REG32 REQENASET48          : 1;
  __REG32 REQENASET49          : 1;
  __REG32 REQENASET50          : 1;
  __REG32 REQENASET51          : 1;
  __REG32 REQENASET52          : 1;
  __REG32 REQENASET53          : 1;
  __REG32 REQENASET54          : 1;
  __REG32 REQENASET55          : 1;
  __REG32 REQENASET56          : 1;
  __REG32 REQENASET57          : 1;
  __REG32 REQENASET58          : 1;
  __REG32 REQENASET59          : 1;
  __REG32 REQENASET60          : 1;
  __REG32 REQENASET61          : 1;
  __REG32 REQENASET62          : 1;
  __REG32 REQENASET63          : 1;
} __reqenaset1_bits;

/* Interrupt Enable Clear Register 0 (REQENACLR0) */
typedef struct {
  __REG32 REQENACLR0           : 1;
  __REG32 REQENACLR1           : 1;
  __REG32 REQENACLR2           : 1;
  __REG32 REQENACLR3           : 1;
  __REG32 REQENACLR4           : 1;
  __REG32 REQENACLR5           : 1;
  __REG32 REQENACLR6           : 1;
  __REG32 REQENACLR7           : 1;
  __REG32 REQENACLR8           : 1;
  __REG32 REQENACLR9           : 1;
  __REG32 REQENACLR10          : 1;
  __REG32 REQENACLR11          : 1;
  __REG32 REQENACLR12          : 1;
  __REG32 REQENACLR13          : 1;
  __REG32 REQENACLR14          : 1;
  __REG32 REQENACLR15          : 1;
  __REG32 REQENACLR16          : 1;
  __REG32 REQENACLR17          : 1;
  __REG32 REQENACLR18          : 1;
  __REG32 REQENACLR19          : 1;
  __REG32 REQENACLR20          : 1;
  __REG32 REQENACLR21          : 1;
  __REG32 REQENACLR22          : 1;
  __REG32 REQENACLR23          : 1;
  __REG32 REQENACLR24          : 1;
  __REG32 REQENACLR25          : 1;
  __REG32 REQENACLR26          : 1;
  __REG32 REQENACLR27          : 1;
  __REG32 REQENACLR28          : 1;
  __REG32 REQENACLR29          : 1;
  __REG32 REQENACLR30          : 1;
  __REG32 REQENACLR31          : 1;
} __reqenaclr0_bits;

/* Interrupt Enable Clear Register 1 (REQENACLR1) */
typedef struct {
  __REG32 REQENACLR32          : 1;
  __REG32 REQENACLR33          : 1;
  __REG32 REQENACLR34          : 1;
  __REG32 REQENACLR35          : 1;
  __REG32 REQENACLR36          : 1;
  __REG32 REQENACLR37          : 1;
  __REG32 REQENACLR38          : 1;
  __REG32 REQENACLR39          : 1;
  __REG32 REQENACLR40          : 1;
  __REG32 REQENACLR41          : 1;
  __REG32 REQENACLR42          : 1;
  __REG32 REQENACLR43          : 1;
  __REG32 REQENACLR44          : 1;
  __REG32 REQENACLR45          : 1;
  __REG32 REQENACLR46          : 1;
  __REG32 REQENACLR47          : 1;
  __REG32 REQENACLR48          : 1;
  __REG32 REQENACLR49          : 1;
  __REG32 REQENACLR50          : 1;
  __REG32 REQENACLR51          : 1;
  __REG32 REQENACLR52          : 1;
  __REG32 REQENACLR53          : 1;
  __REG32 REQENACLR54          : 1;
  __REG32 REQENACLR55          : 1;
  __REG32 REQENACLR56          : 1;
  __REG32 REQENACLR57          : 1;
  __REG32 REQENACLR58          : 1;
  __REG32 REQENACLR59          : 1;
  __REG32 REQENACLR60          : 1;
  __REG32 REQENACLR61          : 1;
  __REG32 REQENACLR62          : 1;
  __REG32 REQENACLR63          : 1;
} __reqenaclr1_bits;

/* Wake-Up Enable Set Register 0 (WAKEENASET0) */
typedef struct {
  __REG32 WAKEENASET0           : 1;
  __REG32 WAKEENASET1           : 1;
  __REG32 WAKEENASET2           : 1;
  __REG32 WAKEENASET3           : 1;
  __REG32 WAKEENASET4           : 1;
  __REG32 WAKEENASET5           : 1;
  __REG32 WAKEENASET6           : 1;
  __REG32 WAKEENASET7           : 1;
  __REG32 WAKEENASET8           : 1;
  __REG32 WAKEENASET9           : 1;
  __REG32 WAKEENASET10          : 1;
  __REG32 WAKEENASET11          : 1;
  __REG32 WAKEENASET12          : 1;
  __REG32 WAKEENASET13          : 1;
  __REG32 WAKEENASET14          : 1;
  __REG32 WAKEENASET15          : 1;
  __REG32 WAKEENASET16          : 1;
  __REG32 WAKEENASET17          : 1;
  __REG32 WAKEENASET18          : 1;
  __REG32 WAKEENASET19          : 1;
  __REG32 WAKEENASET20          : 1;
  __REG32 WAKEENASET21          : 1;
  __REG32 WAKEENASET22          : 1;
  __REG32 WAKEENASET23          : 1;
  __REG32 WAKEENASET24          : 1;
  __REG32 WAKEENASET25          : 1;
  __REG32 WAKEENASET26          : 1;
  __REG32 WAKEENASET27          : 1;
  __REG32 WAKEENASET28          : 1;
  __REG32 WAKEENASET29          : 1;
  __REG32 WAKEENASET30          : 1;
  __REG32 WAKEENASET31          : 1;
} __wakeenaset0_bits;

/* Wake-Up Enable Set Register 1 (WAKEENASET1) */
typedef struct {
  __REG32 WAKEENASET32          : 1;
  __REG32 WAKEENASET33          : 1;
  __REG32 WAKEENASET34          : 1;
  __REG32 WAKEENASET35          : 1;
  __REG32 WAKEENASET36          : 1;
  __REG32 WAKEENASET37          : 1;
  __REG32 WAKEENASET38          : 1;
  __REG32 WAKEENASET39          : 1;
  __REG32 WAKEENASET40          : 1;
  __REG32 WAKEENASET41          : 1;
  __REG32 WAKEENASET42          : 1;
  __REG32 WAKEENASET43          : 1;
  __REG32 WAKEENASET44          : 1;
  __REG32 WAKEENASET45          : 1;
  __REG32 WAKEENASET46          : 1;
  __REG32 WAKEENASET47          : 1;
  __REG32 WAKEENASET48          : 1;
  __REG32 WAKEENASET49          : 1;
  __REG32 WAKEENASET50          : 1;
  __REG32 WAKEENASET51          : 1;
  __REG32 WAKEENASET52          : 1;
  __REG32 WAKEENASET53          : 1;
  __REG32 WAKEENASET54          : 1;
  __REG32 WAKEENASET55          : 1;
  __REG32 WAKEENASET56          : 1;
  __REG32 WAKEENASET57          : 1;
  __REG32 WAKEENASET58          : 1;
  __REG32 WAKEENASET59          : 1;
  __REG32 WAKEENASET60          : 1;
  __REG32 WAKEENASET61          : 1;
  __REG32 WAKEENASET62          : 1;
  __REG32 WAKEENASET63          : 1;
} __wakeenaset1_bits;

/* Wake-Up Enable Clear Register 0 (WAKEENACLR0) */
typedef struct {
  __REG32 WAKEENACLR0           : 1;
  __REG32 WAKEENACLR1           : 1;
  __REG32 WAKEENACLR2           : 1;
  __REG32 WAKEENACLR3           : 1;
  __REG32 WAKEENACLR4           : 1;
  __REG32 WAKEENACLR5           : 1;
  __REG32 WAKEENACLR6           : 1;
  __REG32 WAKEENACLR7           : 1;
  __REG32 WAKEENACLR8           : 1;
  __REG32 WAKEENACLR9           : 1;
  __REG32 WAKEENACLR10          : 1;
  __REG32 WAKEENACLR11          : 1;
  __REG32 WAKEENACLR12          : 1;
  __REG32 WAKEENACLR13          : 1;
  __REG32 WAKEENACLR14          : 1;
  __REG32 WAKEENACLR15          : 1;
  __REG32 WAKEENACLR16          : 1;
  __REG32 WAKEENACLR17          : 1;
  __REG32 WAKEENACLR18          : 1;
  __REG32 WAKEENACLR19          : 1;
  __REG32 WAKEENACLR20          : 1;
  __REG32 WAKEENACLR21          : 1;
  __REG32 WAKEENACLR22          : 1;
  __REG32 WAKEENACLR23          : 1;
  __REG32 WAKEENACLR24          : 1;
  __REG32 WAKEENACLR25          : 1;
  __REG32 WAKEENACLR26          : 1;
  __REG32 WAKEENACLR27          : 1;
  __REG32 WAKEENACLR28          : 1;
  __REG32 WAKEENACLR29          : 1;
  __REG32 WAKEENACLR30          : 1;
  __REG32 WAKEENACLR31          : 1;
} __wakeenaclr0_bits;

/* Wake-Up Enable Clear Register 1 (WAKEENACLR1) */
typedef struct {
  __REG32 WAKEENACLR32          : 1;
  __REG32 WAKEENACLR33          : 1;
  __REG32 WAKEENACLR34          : 1;
  __REG32 WAKEENACLR35          : 1;
  __REG32 WAKEENACLR36          : 1;
  __REG32 WAKEENACLR37          : 1;
  __REG32 WAKEENACLR38          : 1;
  __REG32 WAKEENACLR39          : 1;
  __REG32 WAKEENACLR40          : 1;
  __REG32 WAKEENACLR41          : 1;
  __REG32 WAKEENACLR42          : 1;
  __REG32 WAKEENACLR43          : 1;
  __REG32 WAKEENACLR44          : 1;
  __REG32 WAKEENACLR45          : 1;
  __REG32 WAKEENACLR46          : 1;
  __REG32 WAKEENACLR47          : 1;
  __REG32 WAKEENACLR48          : 1;
  __REG32 WAKEENACLR49          : 1;
  __REG32 WAKEENACLR50          : 1;
  __REG32 WAKEENACLR51          : 1;
  __REG32 WAKEENACLR52          : 1;
  __REG32 WAKEENACLR53          : 1;
  __REG32 WAKEENACLR54          : 1;
  __REG32 WAKEENACLR55          : 1;
  __REG32 WAKEENACLR56          : 1;
  __REG32 WAKEENACLR57          : 1;
  __REG32 WAKEENACLR58          : 1;
  __REG32 WAKEENACLR59          : 1;
  __REG32 WAKEENACLR60          : 1;
  __REG32 WAKEENACLR61          : 1;
  __REG32 WAKEENACLR62          : 1;
  __REG32 WAKEENACLR63          : 1;
} __wakeenaclr1_bits;

/* Capture Event Register (CAPEVT) */
typedef struct {
  __REG32 CAPEVTSRC0            : 7;
  __REG32                       : 9;
  __REG32 CAPEVTSRC1            : 7;
  __REG32                       : 9;
} __capevt_bits;

/* VIM Interrupt Control Register (CHANCTRL0) */
typedef struct {
  __REG32 CHANMAP0              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP1              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP2              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP3              : 7;
  __REG32                       : 1;
} __chanctrl0_bits;

/* VIM Interrupt Control Register (CHANCTRL1) */
typedef struct {
  __REG32 CHANMAP4              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP5              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP6              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP7              : 7;
  __REG32                       : 1;
} __chanctrl1_bits;

/* VIM Interrupt Control Register (CHANCTRL2) */
typedef struct {
  __REG32 CHANMAP8              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP9              : 7;
  __REG32                       : 1;
  __REG32 CHANMAP10             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP11             : 7;
  __REG32                       : 1;
} __chanctrl2_bits;

/* VIM Interrupt Control Register (CHANCTRL3) */
typedef struct {
  __REG32 CHANMAP12             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP13             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP14             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP15             : 7;
  __REG32                       : 1;
} __chanctrl3_bits;

/* VIM Interrupt Control Register (CHANCTRL4) */
typedef struct {
  __REG32 CHANMAP16             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP17             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP18             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP19             : 7;
  __REG32                       : 1;
} __chanctrl4_bits;

/* VIM Interrupt Control Register (CHANCTRL5) */
typedef struct {
  __REG32 CHANMAP20             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP21             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP22             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP23             : 7;
  __REG32                       : 1;
} __chanctrl5_bits;

/* VIM Interrupt Control Register (CHANCTRL6) */
typedef struct {
  __REG32 CHANMAP24             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP25             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP26             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP27             : 7;
  __REG32                       : 1;
} __chanctrl6_bits;

/* VIM Interrupt Control Register (CHANCTRL7) */
typedef struct {
  __REG32 CHANMAP28             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP29             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP30             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP31             : 7;
  __REG32                       : 1;
} __chanctrl7_bits;

/* VIM Interrupt Control Register (CHANCTRL8) */
typedef struct {
  __REG32 CHANMAP32             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP33             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP34             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP35             : 7;
  __REG32                       : 1;
} __chanctrl8_bits;

/* VIM Interrupt Control Register (CHANCTRL9) */
typedef struct {
  __REG32 CHANMAP36             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP37             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP38             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP39             : 7;
  __REG32                       : 1;
} __chanctrl9_bits;

/* VIM Interrupt Control Register (CHANCTRL10) */
typedef struct {
  __REG32 CHANMAP40             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP41             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP42             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP43             : 7;
  __REG32                       : 1;
} __chanctrl10_bits;

/* VIM Interrupt Control Register (CHANCTRL11) */
typedef struct {
  __REG32 CHANMAP44             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP45             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP46             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP47             : 7;
  __REG32                       : 1;
} __chanctrl11_bits;

/* VIM Interrupt Control Register (CHANCTRL12) */
typedef struct {
  __REG32 CHANMAP48             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP49             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP50             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP51             : 7;
  __REG32                       : 1;
} __chanctrl12_bits;

/* VIM Interrupt Control Register (CHANCTRL13) */
typedef struct {
  __REG32 CHANMAP52             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP53             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP54             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP55             : 7;
  __REG32                       : 1;
} __chanctrl13_bits;

/* VIM Interrupt Control Register (CHANCTRL14) */
typedef struct {
  __REG32 CHANMAP56             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP57             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP58             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP59             : 7;
  __REG32                       : 1;
} __chanctrl14_bits;

/* VIM Interrupt Control Register (CHANCTRL15) */
typedef struct {
  __REG32 CHANMAP60             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP61             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP62             : 7;
  __REG32                       : 1;
  __REG32 CHANMAP63             : 7;
  __REG32                       : 1;
} __chanctrl15_bits;

/* ESM Influence Error Pin Set/Status Register 1 (ESMIEPSR1) */
typedef struct {
  __REG32 IEPSET0               : 1;
  __REG32 IEPSET1               : 1;
  __REG32 IEPSET2               : 1;
  __REG32 IEPSET3               : 1;
  __REG32 IEPSET4               : 1;
  __REG32 IEPSET5               : 1;
  __REG32 IEPSET6               : 1;
  __REG32 IEPSET7               : 1;
  __REG32 IEPSET8               : 1;
  __REG32 IEPSET9               : 1;
  __REG32 IEPSET10              : 1;
  __REG32 IEPSET11              : 1;
  __REG32 IEPSET12              : 1;
  __REG32 IEPSET13              : 1;
  __REG32 IEPSET14              : 1;
  __REG32 IEPSET15              : 1;
  __REG32 IEPSET16              : 1;
  __REG32 IEPSET17              : 1;
  __REG32 IEPSET18              : 1;
  __REG32 IEPSET19              : 1;
  __REG32 IEPSET20              : 1;
  __REG32 IEPSET21              : 1;
  __REG32 IEPSET22              : 1;
  __REG32 IEPSET23              : 1;
  __REG32 IEPSET24              : 1;
  __REG32 IEPSET25              : 1;
  __REG32 IEPSET26              : 1;
  __REG32 IEPSET27              : 1;
  __REG32 IEPSET28              : 1;
  __REG32 IEPSET29              : 1;
  __REG32 IEPSET30              : 1;
  __REG32 IEPSET31              : 1;
} __esmiepsr1_bits;

/* ESM Influence Error Pin Clear/Status Register 1 (ESMIEPCR1) */
typedef struct {
  __REG32 IEPCLR0               : 1;
  __REG32 IEPCLR1               : 1;
  __REG32 IEPCLR2               : 1;
  __REG32 IEPCLR3               : 1;
  __REG32 IEPCLR4               : 1;
  __REG32 IEPCLR5               : 1;
  __REG32 IEPCLR6               : 1;
  __REG32 IEPCLR7               : 1;
  __REG32 IEPCLR8               : 1;
  __REG32 IEPCLR9               : 1;
  __REG32 IEPCLR10              : 1;
  __REG32 IEPCLR11              : 1;
  __REG32 IEPCLR12              : 1;
  __REG32 IEPCLR13              : 1;
  __REG32 IEPCLR14              : 1;
  __REG32 IEPCLR15              : 1;
  __REG32 IEPCLR16              : 1;
  __REG32 IEPCLR17              : 1;
  __REG32 IEPCLR18              : 1;
  __REG32 IEPCLR19              : 1;
  __REG32 IEPCLR20              : 1;
  __REG32 IEPCLR21              : 1;
  __REG32 IEPCLR22              : 1;
  __REG32 IEPCLR23              : 1;
  __REG32 IEPCLR24              : 1;
  __REG32 IEPCLR25              : 1;
  __REG32 IEPCLR26              : 1;
  __REG32 IEPCLR27              : 1;
  __REG32 IEPCLR28              : 1;
  __REG32 IEPCLR29              : 1;
  __REG32 IEPCLR30              : 1;
  __REG32 IEPCLR31              : 1;
} __esmiepcr1_bits;

/* ESM Interrupt Enable Set/Status Register 1 (ESMIESR1) */
typedef struct {
  __REG32 INTENSET0               : 1;
  __REG32 INTENSET1               : 1;
  __REG32 INTENSET2               : 1;
  __REG32 INTENSET3               : 1;
  __REG32 INTENSET4               : 1;
  __REG32 INTENSET5               : 1;
  __REG32 INTENSET6               : 1;
  __REG32 INTENSET7               : 1;
  __REG32 INTENSET8               : 1;
  __REG32 INTENSET9               : 1;
  __REG32 INTENSET10              : 1;
  __REG32 INTENSET11              : 1;
  __REG32 INTENSET12              : 1;
  __REG32 INTENSET13              : 1;
  __REG32 INTENSET14              : 1;
  __REG32 INTENSET15              : 1;
  __REG32 INTENSET16              : 1;
  __REG32 INTENSET17              : 1;
  __REG32 INTENSET18              : 1;
  __REG32 INTENSET19              : 1;
  __REG32 INTENSET20              : 1;
  __REG32 INTENSET21              : 1;
  __REG32 INTENSET22              : 1;
  __REG32 INTENSET23              : 1;
  __REG32 INTENSET24              : 1;
  __REG32 INTENSET25              : 1;
  __REG32 INTENSET26              : 1;
  __REG32 INTENSET27              : 1;
  __REG32 INTENSET28              : 1;
  __REG32 INTENSET29              : 1;
  __REG32 INTENSET30              : 1;
  __REG32 INTENSET31              : 1;
} __esmiesr1_bits;

/* ESM Interrupt Enable Clear/Status Register 1 (ESMIECR1) */
typedef struct {
  __REG32 INTENCLR0               : 1;
  __REG32 INTENCLR1               : 1;
  __REG32 INTENCLR2               : 1;
  __REG32 INTENCLR3               : 1;
  __REG32 INTENCLR4               : 1;
  __REG32 INTENCLR5               : 1;
  __REG32 INTENCLR6               : 1;
  __REG32 INTENCLR7               : 1;
  __REG32 INTENCLR8               : 1;
  __REG32 INTENCLR9               : 1;
  __REG32 INTENCLR10              : 1;
  __REG32 INTENCLR11              : 1;
  __REG32 INTENCLR12              : 1;
  __REG32 INTENCLR13              : 1;
  __REG32 INTENCLR14              : 1;
  __REG32 INTENCLR15              : 1;
  __REG32 INTENCLR16              : 1;
  __REG32 INTENCLR17              : 1;
  __REG32 INTENCLR18              : 1;
  __REG32 INTENCLR19              : 1;
  __REG32 INTENCLR20              : 1;
  __REG32 INTENCLR21              : 1;
  __REG32 INTENCLR22              : 1;
  __REG32 INTENCLR23              : 1;
  __REG32 INTENCLR24              : 1;
  __REG32 INTENCLR25              : 1;
  __REG32 INTENCLR26              : 1;
  __REG32 INTENCLR27              : 1;
  __REG32 INTENCLR28              : 1;
  __REG32 INTENCLR29              : 1;
  __REG32 INTENCLR30              : 1;
  __REG32 INTENCLR31              : 1;
} __esmiecr1_bits;

/* ESM Interrupt Level Set/Status Register 1 (ESMILSR1) */
typedef struct {
  __REG32 INTVLVSET0               : 1;
  __REG32 INTVLVSET1               : 1;
  __REG32 INTVLVSET2               : 1;
  __REG32 INTVLVSET3               : 1;
  __REG32 INTVLVSET4               : 1;
  __REG32 INTVLVSET5               : 1;
  __REG32 INTVLVSET6               : 1;
  __REG32 INTVLVSET7               : 1;
  __REG32 INTVLVSET8               : 1;
  __REG32 INTVLVSET9               : 1;
  __REG32 INTVLVSET10              : 1;
  __REG32 INTVLVSET11              : 1;
  __REG32 INTVLVSET12              : 1;
  __REG32 INTVLVSET13              : 1;
  __REG32 INTVLVSET14              : 1;
  __REG32 INTVLVSET15              : 1;
  __REG32 INTVLVSET16              : 1;
  __REG32 INTVLVSET17              : 1;
  __REG32 INTVLVSET18              : 1;
  __REG32 INTVLVSET19              : 1;
  __REG32 INTVLVSET20              : 1;
  __REG32 INTVLVSET21              : 1;
  __REG32 INTVLVSET22              : 1;
  __REG32 INTVLVSET23              : 1;
  __REG32 INTVLVSET24              : 1;
  __REG32 INTVLVSET25              : 1;
  __REG32 INTVLVSET26              : 1;
  __REG32 INTVLVSET27              : 1;
  __REG32 INTVLVSET28              : 1;
  __REG32 INTVLVSET29              : 1;
  __REG32 INTVLVSET30              : 1;
  __REG32 INTVLVSET31              : 1;
} __esmilsr1_bits;

/* ESM Interrupt Level Clear/Status Register 1 (ESMILCR1) */
typedef struct {
  __REG32 INTVLVCLR0               : 1;
  __REG32 INTVLVCLR1               : 1;
  __REG32 INTVLVCLR2               : 1;
  __REG32 INTVLVCLR3               : 1;
  __REG32 INTVLVCLR4               : 1;
  __REG32 INTVLVCLR5               : 1;
  __REG32 INTVLVCLR6               : 1;
  __REG32 INTVLVCLR7               : 1;
  __REG32 INTVLVCLR8               : 1;
  __REG32 INTVLVCLR9               : 1;
  __REG32 INTVLVCLR10              : 1;
  __REG32 INTVLVCLR11              : 1;
  __REG32 INTVLVCLR12              : 1;
  __REG32 INTVLVCLR13              : 1;
  __REG32 INTVLVCLR14              : 1;
  __REG32 INTVLVCLR15              : 1;
  __REG32 INTVLVCLR16              : 1;
  __REG32 INTVLVCLR17              : 1;
  __REG32 INTVLVCLR18              : 1;
  __REG32 INTVLVCLR19              : 1;
  __REG32 INTVLVCLR20              : 1;
  __REG32 INTVLVCLR21              : 1;
  __REG32 INTVLVCLR22              : 1;
  __REG32 INTVLVCLR23              : 1;
  __REG32 INTVLVCLR24              : 1;
  __REG32 INTVLVCLR25              : 1;
  __REG32 INTVLVCLR26              : 1;
  __REG32 INTVLVCLR27              : 1;
  __REG32 INTVLVCLR28              : 1;
  __REG32 INTVLVCLR29              : 1;
  __REG32 INTVLVCLR30              : 1;
  __REG32 INTVLVCLR31              : 1;
} __esmilcr1_bits;

/* ESM Status Register 1 (ESMSR1) */
typedef struct {
  __REG32 ESF0               : 1;
  __REG32 ESF1               : 1;
  __REG32 ESF2               : 1;
  __REG32 ESF3               : 1;
  __REG32 ESF4               : 1;
  __REG32 ESF5               : 1;
  __REG32 ESF6               : 1;
  __REG32 ESF7               : 1;
  __REG32 ESF8               : 1;
  __REG32 ESF9               : 1;
  __REG32 ESF10              : 1;
  __REG32 ESF11              : 1;
  __REG32 ESF12              : 1;
  __REG32 ESF13              : 1;
  __REG32 ESF14              : 1;
  __REG32 ESF15              : 1;
  __REG32 ESF16              : 1;
  __REG32 ESF17              : 1;
  __REG32 ESF18              : 1;
  __REG32 ESF19              : 1;
  __REG32 ESF20              : 1;
  __REG32 ESF21              : 1;
  __REG32 ESF22              : 1;
  __REG32 ESF23              : 1;
  __REG32 ESF24              : 1;
  __REG32 ESF25              : 1;
  __REG32 ESF26              : 1;
  __REG32 ESF27              : 1;
  __REG32 ESF28              : 1;
  __REG32 ESF29              : 1;
  __REG32 ESF30              : 1;
  __REG32 ESF31              : 1;
} __esmsr_bits;

/* ESM Error Pin Status Register (ESMEPSR) */
typedef struct {
  __REG32 EPSF               : 1;
  __REG32                    :31;
} __esmepsr_bits;

/* ESM Interrupt Offset High Register (ESMIOFFHR) */
typedef struct {
  __REG32 INTOFFH            : 7;
  __REG32                    :25;
} __esmioffhr_bits;

/* ESM Interrupt Offset Low Register (ESMIOFFLR) */
typedef struct {
  __REG32 INTOFFL            : 7;
  __REG32                    :25;
} __esmiofflr_bits;

/* ESM Low-Time Counter Register (ESMLTCR) */
typedef struct {
  __REG32 LTC                :16;
  __REG32                    :16;
} __esmltcr_bits;

/* ESM Low-Time Counter Preload Register (ESMLTCPR) */
typedef struct {
  __REG32 LTCP               :16;
  __REG32                    :16;
} __esmltcpr_bits;

/* ESM Error Key Register (ESMEKR) */
typedef struct {
  __REG32 EKEY               : 4;
  __REG32                    :28;
} __esmekr_bits;

/* DMM Global Control Register (DMMGLBCTRL) */
typedef struct {
  __REG32 ON_OFF             : 4;
  __REG32                    : 4;
  __REG32 TM_DDM             : 1;
  __REG32 DDM_WIDTH          : 2;
  __REG32                    : 5;
  __REG32 RESET              : 1;
  __REG32 COS                : 1;
  __REG32 CONTCLK            : 1;
  __REG32                    : 5;
  __REG32 BUSY               : 1;
  __REG32                    : 7;
} __dmmglbctrl_bits;

/* DMM Interrupt Set Register (DMMINTSET) */
typedef struct {
  __REG32 PACKET_ERR_INT      : 1;
  __REG32 DEST0_ERR           : 1;
  __REG32 DEST1_ERR           : 1;
  __REG32 DEST2_ERR           : 1;
  __REG32 DEST3_ERR           : 1;
  __REG32 SRC_OVF             : 1;
  __REG32 BUFF_OVF            : 1;
  __REG32 BUSERROR            : 1;
  __REG32 DEST0REG1           : 1;
  __REG32 DEST0REG2           : 1;
  __REG32 DEST1REG1           : 1;
  __REG32 DEST1REG2           : 1;
  __REG32 DEST2REG1           : 1;
  __REG32 DEST2REG2           : 1;
  __REG32 DEST3REG1           : 1;
  __REG32 DEST3REG2           : 1;
  __REG32 EO_BUFF             : 1;
  __REG32 PROG_BUFF           : 1;
  __REG32                     :14;
} __dmmintset_bits;

/* DMM Interrupt Offset 1 Register (DMMOFF1) */
typedef struct {
  __REG32 OFFSET              : 5;
  __REG32                     :27;
} __dmmoff_bits;

/* DMM Direct Data Mode Blocksize Register (DMMDDMBL) */
typedef struct {
  __REG32 BLOCKSIZE           : 4;
  __REG32                     :28;
} __dmmddmbl_bits;

/* DMM Direct Data Mode Pointer Register (DMMDDMPT) */
typedef struct {
  __REG32 POINTER             :15;
  __REG32                     :17;
} __dmmddmpt_bits;

/* DMM Direct Data Mode Interrupt Pointer Register (DMMINTPT) */
typedef struct {
  __REG32 INTPT               :15;
  __REG32                     :17;
} __dmmintpt_bits;

/* DMM Destination x Region n (DMMDESTxREGn) */
typedef struct {
  __REG32 BLOCKADDR           :18;
  __REG32 BASEADDR            :14;
} __dmmdestreg_bits;

/* DMM Destination x Blocksize n (DMMDESTxBLn) */
typedef struct {
  __REG32 BLOCKSIZE           : 4;
  __REG32                     :28;
} __dmmdestbl_bits;

/* DMM Pin Control 0 (DMMPC0) */
typedef struct {
  __REG32 SYNCFUNC            : 1;
  __REG32 CLKFUNC             : 1;
  __REG32 DATA0FUNC           : 1;
  __REG32 DATA1FUNC           : 1;
  __REG32 DATA2FUNC           : 1;
  __REG32 DATA3FUNC           : 1;
  __REG32 DATA4FUNC           : 1;
  __REG32 DATA5FUNC           : 1;
  __REG32 DATA6FUNC           : 1;
  __REG32 DATA7FUNC           : 1;
  __REG32 DATA8FUNC           : 1;
  __REG32 DATA9FUNC           : 1;
  __REG32 DATA10FUNC          : 1;
  __REG32 DATA11FUNC          : 1;
  __REG32 DATA12FUNC          : 1;
  __REG32 DATA13FUNC          : 1;
  __REG32 DATA14FUNC          : 1;
  __REG32 DATA15FUNC          : 1;
  __REG32 ENAFUNC             : 1;
  __REG32                     :13;
} __dmmpc0_bits;

/* DMM Pin Control 1 (DMMPC1) */
typedef struct {
  __REG32 SYNCDIR             : 1;
  __REG32 CLKDIR              : 1;
  __REG32 DATA0DIR            : 1;
  __REG32 DATA1DIR            : 1;
  __REG32 DATA2DIR            : 1;
  __REG32 DATA3DIR            : 1;
  __REG32 DATA4DIR            : 1;
  __REG32 DATA5DIR            : 1;
  __REG32 DATA6DIR            : 1;
  __REG32 DATA7DIR            : 1;
  __REG32 DATA8DIR            : 1;
  __REG32 DATA9DIR            : 1;
  __REG32 DATA10DIR           : 1;
  __REG32 DATA11DIR           : 1;
  __REG32 DATA12DIR           : 1;
  __REG32 DATA13DIR           : 1;
  __REG32 DATA14DIR           : 1;
  __REG32 DATA15DIR           : 1;
  __REG32 ENADIR              : 1;
  __REG32                     :13;
} __dmmpc1_bits;

/* DMM Pin Control 2 (DMMPC2) */
typedef struct {
  __REG32 SYNCIN             : 1;
  __REG32 CLKIN              : 1;
  __REG32 DATA0IN            : 1;
  __REG32 DATA1IN            : 1;
  __REG32 DATA2IN            : 1;
  __REG32 DATA3IN            : 1;
  __REG32 DATA4IN            : 1;
  __REG32 DATA5IN            : 1;
  __REG32 DATA6IN            : 1;
  __REG32 DATA7IN            : 1;
  __REG32 DATA8IN            : 1;
  __REG32 DATA9IN            : 1;
  __REG32 DATA10IN           : 1;
  __REG32 DATA11IN           : 1;
  __REG32 DATA12IN           : 1;
  __REG32 DATA13IN           : 1;
  __REG32 DATA14IN           : 1;
  __REG32 DATA15IN           : 1;
  __REG32 ENAIN              : 1;
  __REG32                    :13;
} __dmmpc2_bits;

/* DMM Pin Control 3 (DMMPC3) */
typedef struct {
  __REG32 SYNCOUT             : 1;
  __REG32 CLKOUT              : 1;
  __REG32 DATA0OUT            : 1;
  __REG32 DATA1OUT            : 1;
  __REG32 DATA2OUT            : 1;
  __REG32 DATA3OUT            : 1;
  __REG32 DATA4OUT            : 1;
  __REG32 DATA5OUT            : 1;
  __REG32 DATA6OUT            : 1;
  __REG32 DATA7OUT            : 1;
  __REG32 DATA8OUT            : 1;
  __REG32 DATA9OUT            : 1;
  __REG32 DATA10OUT           : 1;
  __REG32 DATA11OUT           : 1;
  __REG32 DATA12OUT           : 1;
  __REG32 DATA13OUT           : 1;
  __REG32 DATA14OUT           : 1;
  __REG32 DATA15OUT           : 1;
  __REG32 ENAOUT              : 1;
  __REG32                     :13;
} __dmmpc3_bits;

/* DMM Pin Control 4 (DMMPC5) */
typedef struct {
  __REG32 SYNCSET             : 1;
  __REG32 CLKSET              : 1;
  __REG32 DATA0SET            : 1;
  __REG32 DATA1SET            : 1;
  __REG32 DATA2SET            : 1;
  __REG32 DATA3SET            : 1;
  __REG32 DATA4SET            : 1;
  __REG32 DATA5SET            : 1;
  __REG32 DATA6SET            : 1;
  __REG32 DATA7SET            : 1;
  __REG32 DATA8SET            : 1;
  __REG32 DATA9SET            : 1;
  __REG32 DATA10SET           : 1;
  __REG32 DATA11SET           : 1;
  __REG32 DATA12SET           : 1;
  __REG32 DATA13SET           : 1;
  __REG32 DATA14SET           : 1;
  __REG32 DATA15SET           : 1;
  __REG32 ENASET              : 1;
  __REG32                     :13;
} __dmmpc4_bits;

/* DMM Pin Control 5 (DMMPC5) */
typedef struct {
  __REG32 SYNCCLR             : 1;
  __REG32 CLKCLR              : 1;
  __REG32 DATA0CLR            : 1;
  __REG32 DATA1CLR            : 1;
  __REG32 DATA2CLR            : 1;
  __REG32 DATA3CLR            : 1;
  __REG32 DATA4CLR            : 1;
  __REG32 DATA5CLR            : 1;
  __REG32 DATA6CLR            : 1;
  __REG32 DATA7CLR            : 1;
  __REG32 DATA8CLR            : 1;
  __REG32 DATA9CLR            : 1;
  __REG32 DATA10CLR           : 1;
  __REG32 DATA11CLR           : 1;
  __REG32 DATA12CLR           : 1;
  __REG32 DATA13CLR           : 1;
  __REG32 DATA14CLR           : 1;
  __REG32 DATA15CLR           : 1;
  __REG32 ENACLR              : 1;
  __REG32                     :13;
} __dmmpc5_bits;

/* DMM Pin Control 6 (DMMPC6) */
typedef struct {
  __REG32 SYNCPDR             : 1;
  __REG32 CLKPDR              : 1;
  __REG32 DATA0PDR            : 1;
  __REG32 DATA1PDR            : 1;
  __REG32 DATA2PDR            : 1;
  __REG32 DATA3PDR            : 1;
  __REG32 DATA4PDR            : 1;
  __REG32 DATA5PDR            : 1;
  __REG32 DATA6PDR            : 1;
  __REG32 DATA7PDR            : 1;
  __REG32 DATA8PDR            : 1;
  __REG32 DATA9PDR            : 1;
  __REG32 DATA10PDR           : 1;
  __REG32 DATA11PDR           : 1;
  __REG32 DATA12PDR           : 1;
  __REG32 DATA13PDR           : 1;
  __REG32 DATA14PDR           : 1;
  __REG32 DATA15PDR           : 1;
  __REG32 ENAPDR              : 1;
  __REG32                     :13;
} __dmmpc6_bits;

/* DMM Pin Control 7 (DMMPC7) */
typedef struct {
  __REG32 SYNCPDIS            : 1;
  __REG32 CLKPDIS             : 1;
  __REG32 DATA0PDIS           : 1;
  __REG32 DATA1PDIS           : 1;
  __REG32 DATA2PDIS           : 1;
  __REG32 DATA3PDIS           : 1;
  __REG32 DATA4PDIS           : 1;
  __REG32 DATA5PDIS           : 1;
  __REG32 DATA6PDIS           : 1;
  __REG32 DATA7PDIS           : 1;
  __REG32 DATA8PDIS           : 1;
  __REG32 DATA9PDIS           : 1;
  __REG32 DATA10PDIS          : 1;
  __REG32 DATA11PDIS          : 1;
  __REG32 DATA12PDIS          : 1;
  __REG32 DATA13PDIS          : 1;
  __REG32 DATA14PDIS          : 1;
  __REG32 DATA15PDIS          : 1;
  __REG32 ENAPDIS             : 1;
  __REG32                     :13;
} __dmmpc7_bits;

/* DMM Pin Control 8 (DMMPC8) */
typedef struct {
  __REG32 SYNCPSEL            : 1;
  __REG32 CLKPSEL             : 1;
  __REG32 DATA0PSEL           : 1;
  __REG32 DATA1PSEL           : 1;
  __REG32 DATA2PSEL           : 1;
  __REG32 DATA3PSEL           : 1;
  __REG32 DATA4PSEL           : 1;
  __REG32 DATA5PSEL           : 1;
  __REG32 DATA6PSEL           : 1;
  __REG32 DATA7PSEL           : 1;
  __REG32 DATA8PSEL           : 1;
  __REG32 DATA9PSEL           : 1;
  __REG32 DATA10PSEL          : 1;
  __REG32 DATA11PSEL          : 1;
  __REG32 DATA12PSEL          : 1;
  __REG32 DATA13PSEL          : 1;
  __REG32 DATA14PSEL          : 1;
  __REG32 DATA15PSEL          : 1;
  __REG32 ENAPSEL             : 1;
  __REG32                     :13;
} __dmmpc8_bits;

/* RTP Global Control Register (RTPGLBCTRL) */
typedef struct {
  __REG32 ON_OFF              : 4;
  __REG32 INV_RGN             : 1;
  __REG32 HOVF                : 1;
  __REG32 CONTCLK             : 1;
  __REG32 RESET               : 1;
  __REG32 PW                  : 2;
  __REG32 TM_DDM              : 1;
  __REG32 DDM_RW              : 1;
  __REG32 DDM_WIDTH           : 2;
  __REG32                     : 2;
  __REG32 PRESCALER           : 3;
  __REG32                     : 5;
  __REG32 TEST                : 1;
  __REG32                     : 7;
} __rtpglbctrl_bits;

/* RTP Trace Enable Register (RTPTRENA) */
typedef struct {
  __REG32 ENA1                : 1;
  __REG32                     : 7;
  __REG32 ENA2                : 1;
  __REG32                     :15;
  __REG32 ENA4                : 1;
  __REG32                     : 7;
} __rtptrena_bits;

/* RTP Global Status Register (RTPGSR) */
typedef struct {
  __REG32 OVF1                : 1;
  __REG32 OVF2                : 1;
  __REG32                     : 1;
  __REG32 OVFPER              : 1;
  __REG32                     : 4;
  __REG32 EMPTY1              : 1;
  __REG32 EMPTY2              : 1;
  __REG32                     : 1;
  __REG32 EMPTYPER            : 1;
  __REG32 EMPTYSER            : 1;
  __REG32                     :19;
} __rtpgsr_bits;

/* RTP RAM 1/2 Trace Region x Register  */
typedef struct {
  __REG32 STARTADDR           :18;
  __REG32                     : 6;
  __REG32 BLOCKSIZE           : 4;
  __REG32 RW                  : 1;
  __REG32 CPU_DMA             : 2;
  __REG32                     : 1;
} __rtpramreg_bits;

/* RTP Peripheral Trace Region [1:2] Registers (RTPPERREG[1:2]) */
typedef struct {
  __REG32 STARTADDR           :24;
  __REG32 BLOCKSIZE           : 4;
  __REG32 RW                  : 1;
  __REG32 CPU_DMA             : 2;
  __REG32                     : 1;
} __rtpperreg_bits;

/* RTP Pin Control 0 Register (RTPPC0) */
typedef struct {
  __REG32 DATA0FUNC           : 1;
  __REG32 DATA1FUNC           : 1;
  __REG32 DATA2FUNC           : 1;
  __REG32 DATA3FUNC           : 1;
  __REG32 DATA4FUNC           : 1;
  __REG32 DATA5FUNC           : 1;
  __REG32 DATA6FUNC           : 1;
  __REG32 DATA7FUNC           : 1;
  __REG32 DATA8FUNC           : 1;
  __REG32 DATA9FUNC           : 1;
  __REG32 DATA10FUNC          : 1;
  __REG32 DATA11FUNC          : 1;
  __REG32 DATA12FUNC          : 1;
  __REG32 DATA13FUNC          : 1;
  __REG32 DATA14FUNC          : 1;
  __REG32 DATA15FUNC          : 1;
  __REG32 SYNCFUNC            : 1;
  __REG32 CLKFUNC             : 1;
  __REG32 ENAFUNC             : 1;
  __REG32                     :13;
} __rtppc0_bits;

/* RTP Pin Control 1 Register (RTPPC1) */
typedef struct {
  __REG32 DATA0DIR           : 1;
  __REG32 DATA1DIR           : 1;
  __REG32 DATA2DIR           : 1;
  __REG32 DATA3DIR           : 1;
  __REG32 DATA4DIR           : 1;
  __REG32 DATA5DIR           : 1;
  __REG32 DATA6DIR           : 1;
  __REG32 DATA7DIR           : 1;
  __REG32 DATA8DIR           : 1;
  __REG32 DATA9DIR           : 1;
  __REG32 DATA10DIR          : 1;
  __REG32 DATA11DIR          : 1;
  __REG32 DATA12DIR          : 1;
  __REG32 DATA13DIR          : 1;
  __REG32 DATA14DIR          : 1;
  __REG32 DATA15DIR          : 1;
  __REG32 SYNCDIR            : 1;
  __REG32 CLKDIR             : 1;
  __REG32 ENADIR             : 1;
  __REG32                    :13;
} __rtppc1_bits;

/* RTP Pin Control 2 Register (RTPPC2) */
typedef struct {
  __REG32 DATA0IN           : 1;
  __REG32 DATA1IN           : 1;
  __REG32 DATA2IN           : 1;
  __REG32 DATA3IN           : 1;
  __REG32 DATA4IN           : 1;
  __REG32 DATA5IN           : 1;
  __REG32 DATA6IN           : 1;
  __REG32 DATA7IN           : 1;
  __REG32 DATA8IN           : 1;
  __REG32 DATA9IN           : 1;
  __REG32 DATA10IN          : 1;
  __REG32 DATA11IN          : 1;
  __REG32 DATA12IN          : 1;
  __REG32 DATA13IN          : 1;
  __REG32 DATA14IN          : 1;
  __REG32 DATA15IN          : 1;
  __REG32 SYNCIN            : 1;
  __REG32 CLKIN             : 1;
  __REG32 ENAIN             : 1;
  __REG32                   :13;
} __rtppc2_bits;

/* RTP Pin Control 3 Register (RTPPC3) */
typedef struct {
  __REG32 DATA0OUT           : 1;
  __REG32 DATA1OUT           : 1;
  __REG32 DATA2OUT           : 1;
  __REG32 DATA3OUT           : 1;
  __REG32 DATA4OUT           : 1;
  __REG32 DATA5OUT           : 1;
  __REG32 DATA6OUT           : 1;
  __REG32 DATA7OUT           : 1;
  __REG32 DATA8OUT           : 1;
  __REG32 DATA9OUT           : 1;
  __REG32 DATA10OUT          : 1;
  __REG32 DATA11OUT          : 1;
  __REG32 DATA12OUT          : 1;
  __REG32 DATA13OUT          : 1;
  __REG32 DATA14OUT          : 1;
  __REG32 DATA15OUT          : 1;
  __REG32 SYNCOUT            : 1;
  __REG32 CLKOUT             : 1;
  __REG32 ENAOUT             : 1;
  __REG32                    :13;
} __rtppc3_bits;

/* RTP Pin Control 4 Register (RTPPC4) */
typedef struct {
  __REG32 DATA0SET           : 1;
  __REG32 DATA1SET           : 1;
  __REG32 DATA2SET           : 1;
  __REG32 DATA3SET           : 1;
  __REG32 DATA4SET           : 1;
  __REG32 DATA5SET           : 1;
  __REG32 DATA6SET           : 1;
  __REG32 DATA7SET           : 1;
  __REG32 DATA8SET           : 1;
  __REG32 DATA9SET           : 1;
  __REG32 DATA10SET          : 1;
  __REG32 DATA11SET          : 1;
  __REG32 DATA12SET          : 1;
  __REG32 DATA13SET          : 1;
  __REG32 DATA14SET          : 1;
  __REG32 DATA15SET          : 1;
  __REG32 SYNCSET            : 1;
  __REG32 CLKSET             : 1;
  __REG32 ENASET             : 1;
  __REG32                    :13;
} __rtppc4_bits;

/* RTP Pin Control 5 Register (RTPPC5) */
typedef struct {
  __REG32 DATA0CLR           : 1;
  __REG32 DATA1CLR           : 1;
  __REG32 DATA2CLR           : 1;
  __REG32 DATA3CLR           : 1;
  __REG32 DATA4CLR           : 1;
  __REG32 DATA5CLR           : 1;
  __REG32 DATA6CLR           : 1;
  __REG32 DATA7CLR           : 1;
  __REG32 DATA8CLR           : 1;
  __REG32 DATA9CLR           : 1;
  __REG32 DATA10CLR          : 1;
  __REG32 DATA11CLR          : 1;
  __REG32 DATA12CLR          : 1;
  __REG32 DATA13CLR          : 1;
  __REG32 DATA14CLR          : 1;
  __REG32 DATA15CLR          : 1;
  __REG32 SYNCCLR            : 1;
  __REG32 CLKCLR             : 1;
  __REG32 ENACLR             : 1;
  __REG32                    :13;
} __rtppc5_bits;

/* RTP Pin Control 6 Register (RTPPC6) */
typedef struct {
  __REG32 DATA0PDR           : 1;
  __REG32 DATA1PDR           : 1;
  __REG32 DATA2PDR           : 1;
  __REG32 DATA3PDR           : 1;
  __REG32 DATA4PDR           : 1;
  __REG32 DATA5PDR           : 1;
  __REG32 DATA6PDR           : 1;
  __REG32 DATA7PDR           : 1;
  __REG32 DATA8PDR           : 1;
  __REG32 DATA9PDR           : 1;
  __REG32 DATA10PDR          : 1;
  __REG32 DATA11PDR          : 1;
  __REG32 DATA12PDR          : 1;
  __REG32 DATA13PDR          : 1;
  __REG32 DATA14PDR          : 1;
  __REG32 DATA15PDR          : 1;
  __REG32 SYNCPDR            : 1;
  __REG32 CLKPDR             : 1;
  __REG32 ENAPDR             : 1;
  __REG32                    :13;
} __rtppc6_bits;

/* RTP Pin Control 7 Register (RTPPC7) */
typedef struct {
  __REG32 DATA0PDIS          : 1;
  __REG32 DATA1PDIS          : 1;
  __REG32 DATA2PDIS          : 1;
  __REG32 DATA3PDIS          : 1;
  __REG32 DATA4PDIS          : 1;
  __REG32 DATA5PDIS          : 1;
  __REG32 DATA6PDIS          : 1;
  __REG32 DATA7PDIS          : 1;
  __REG32 DATA8PDIS          : 1;
  __REG32 DATA9PDIS          : 1;
  __REG32 DATA10PDIS         : 1;
  __REG32 DATA11PDIS         : 1;
  __REG32 DATA12PDIS         : 1;
  __REG32 DATA13PDIS         : 1;
  __REG32 DATA14PDIS         : 1;
  __REG32 DATA15PDIS         : 1;
  __REG32 SYNCPDIS           : 1;
  __REG32 CLKPDIS            : 1;
  __REG32 ENAPDIS            : 1;
  __REG32                    :13;
} __rtppc7_bits;

/* RTP Pin Control 8 Register (RTPPC8) */
typedef struct {
  __REG32 DATA0PSEL          : 1;
  __REG32 DATA1PSEL          : 1;
  __REG32 DATA2PSEL          : 1;
  __REG32 DATA3PSEL          : 1;
  __REG32 DATA4PSEL          : 1;
  __REG32 DATA5PSEL          : 1;
  __REG32 DATA6PSEL          : 1;
  __REG32 DATA7PSEL          : 1;
  __REG32 DATA8PSEL          : 1;
  __REG32 DATA9PSEL          : 1;
  __REG32 DATA10PSEL         : 1;
  __REG32 DATA11PSEL         : 1;
  __REG32 DATA12PSEL         : 1;
  __REG32 DATA13PSEL         : 1;
  __REG32 DATA14PSEL         : 1;
  __REG32 DATA15PSEL         : 1;
  __REG32 SYNCPSEL           : 1;
  __REG32 CLKPSEL            : 1;
  __REG32 ENAPSEL            : 1;
  __REG32                    :13;
} __rtppc8_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler *********************************/

/***************************************************************************
 **
 ** VIM (Vectored Interrupt Manager)
 **
 ***************************************************************************/
__IO_REG32_BIT(IRQINDEX,          0xFFFFFE00,__READ       ,__irqindex_bits);
__IO_REG32_BIT(FIQINDEX,          0xFFFFFE04,__READ       ,__fiqindex_bits);
__IO_REG32_BIT(FIRQPR0,           0xFFFFFE10,__READ_WRITE ,__firqpr0_bits);
__IO_REG32_BIT(FIRQPR1,           0xFFFFFE14,__READ_WRITE ,__firqpr1_bits);
__IO_REG32_BIT(INTREQ0,           0xFFFFFE20,__READ       ,__intreq0_bits);
__IO_REG32_BIT(INTREQ1,           0xFFFFFE24,__READ       ,__intreq1_bits);
__IO_REG32_BIT(REQENASET0,        0xFFFFFE30,__READ_WRITE ,__reqenaset0_bits);
__IO_REG32_BIT(REQENASET1,        0xFFFFFE34,__READ_WRITE ,__reqenaset1_bits);
__IO_REG32_BIT(REQENACLR0,        0xFFFFFE40,__READ_WRITE ,__reqenaclr0_bits);
__IO_REG32_BIT(REQENACLR1,        0xFFFFFE44,__READ_WRITE ,__reqenaclr1_bits);
__IO_REG32_BIT(WAKEENASET0,       0xFFFFFE50,__READ_WRITE ,__wakeenaset0_bits);
__IO_REG32_BIT(WAKEENASET1,       0xFFFFFE54,__READ_WRITE ,__wakeenaset1_bits);
__IO_REG32_BIT(WAKEENACLR0,       0xFFFFFE60,__READ_WRITE ,__wakeenaclr0_bits);
__IO_REG32_BIT(WAKEENACLR1,       0xFFFFFE64,__READ_WRITE ,__wakeenaclr1_bits);
__IO_REG32(    IRQVECREG,         0xFFFFFE70,__READ       );
__IO_REG32(    FIQVECREG,         0xFFFFFE74,__READ       );
__IO_REG32_BIT(CAPEVT,            0xFFFFFE78,__READ_WRITE ,__capevt_bits);
__IO_REG32_BIT(CHANCTRL0,         0xFFFFFE80,__READ_WRITE ,__chanctrl0_bits);
__IO_REG32_BIT(CHANCTRL1,         0xFFFFFE84,__READ_WRITE ,__chanctrl1_bits);
__IO_REG32_BIT(CHANCTRL2,         0xFFFFFE88,__READ_WRITE ,__chanctrl2_bits);
__IO_REG32_BIT(CHANCTRL3,         0xFFFFFE8C,__READ_WRITE ,__chanctrl3_bits);
__IO_REG32_BIT(CHANCTRL4,         0xFFFFFE90,__READ_WRITE ,__chanctrl4_bits);
__IO_REG32_BIT(CHANCTRL5,         0xFFFFFE94,__READ_WRITE ,__chanctrl5_bits);
__IO_REG32_BIT(CHANCTRL6,         0xFFFFFE98,__READ_WRITE ,__chanctrl6_bits);
__IO_REG32_BIT(CHANCTRL7,         0xFFFFFE9C,__READ_WRITE ,__chanctrl7_bits);
__IO_REG32_BIT(CHANCTRL8,         0xFFFFFEA0,__READ_WRITE ,__chanctrl8_bits);
__IO_REG32_BIT(CHANCTRL9,         0xFFFFFEA4,__READ_WRITE ,__chanctrl9_bits);
__IO_REG32_BIT(CHANCTRL10,        0xFFFFFEA8,__READ_WRITE ,__chanctrl10_bits);
__IO_REG32_BIT(CHANCTRL11,        0xFFFFFEAC,__READ_WRITE ,__chanctrl11_bits);
__IO_REG32_BIT(CHANCTRL12,        0xFFFFFEB0,__READ_WRITE ,__chanctrl12_bits);
__IO_REG32_BIT(CHANCTRL13,        0xFFFFFEB4,__READ_WRITE ,__chanctrl13_bits);
__IO_REG32_BIT(CHANCTRL14,        0xFFFFFEB8,__READ_WRITE ,__chanctrl14_bits);
__IO_REG32_BIT(CHANCTRL15,        0xFFFFFEBC,__READ_WRITE ,__chanctrl15_bits);
__IO_REG32_BIT(PARFLG,            0xFFFFFDEC,__READ_WRITE ,__parflg_bits);
__IO_REG32_BIT(PARCTL,            0xFFFFFDF0,__READ_WRITE ,__parctl_bits);
__IO_REG32(    ADDERR,            0xFFFFFDF4,__READ       );
__IO_REG32(    FBPARERR,          0xFFFFFDF8,__READ_WRITE );
__IO_REG32(    VIM_RAM_BASE,      0xFFF82000,__READ_WRITE );

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
__IO_REG32_BIT(CSDIS,             0xFFFFFF30,__READ_WRITE ,__csdis_bits);
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
__IO_REG32_BIT(DIEIDL,            0xFFFFFF7C,__READ       ,__dieidl_bits);
__IO_REG32_BIT(DIEIDH,            0xFFFFFF80,__READ       ,__dieidh_bits);
__IO_REG32_BIT(LPOMONCTL,         0xFFFFFF88,__READ_WRITE ,__lpomonctl_bits);
__IO_REG32_BIT(CLKTEST,           0xFFFFFF8C,__READ_WRITE ,__clktest_bits);
__IO_REG32_BIT(IMPFASTS,          0xFFFFFFA8,__READ       ,__impfasts_bits);
__IO_REG32(    IMPFTADD,          0xFFFFFFAC,__READ       );
__IO_REG32_BIT(SSIR1,             0xFFFFFFB0,__READ_WRITE ,__ssir1_bits);
__IO_REG32_BIT(SSIR2,             0xFFFFFFB4,__READ_WRITE ,__ssir2_bits);
__IO_REG32_BIT(SSIR3,             0xFFFFFFB8,__READ_WRITE ,__ssir3_bits);
__IO_REG32_BIT(SSIR4,             0xFFFFFFBC,__READ_WRITE ,__ssir4_bits);
__IO_REG32_BIT(RAMGCR,            0xFFFFFFC0,__READ_WRITE ,__ramgcr_bits);
__IO_REG32_BIT(BMMCR1,            0xFFFFFFC4,__READ_WRITE ,__bmmcr_bits);
__IO_REG32_BIT(BMMCR2,            0xFFFFFFC8,__READ_WRITE ,__bmmcr_bits);
__IO_REG32_BIT(MMUGCR,            0xFFFFFFCC,__READ_WRITE ,__mmugcr_bits);
__IO_REG32_BIT(CLKCNTRL,          0xFFFFFFD0,__READ_WRITE ,__clkcntrl_bits);
__IO_REG32_BIT(ECPCNTRL,          0xFFFFFFD4,__READ_WRITE ,__ecpcntl_bits);
__IO_REG32_BIT(SYSECR,            0xFFFFFFE0,__READ_WRITE ,__sysecr_bits);
__IO_REG32_BIT(SYSESR,            0xFFFFFFE4,__READ_WRITE ,__sysesr_bits);
__IO_REG32_BIT(GLBSTAT,           0xFFFFFFEC,__READ_WRITE ,__glbstat_bits);
__IO_REG32_BIT(DEVID,             0xFFFFFFF0,__READ       ,__devid_bits);
__IO_REG32_BIT(SSIVEC,            0xFFFFFFF4,__READ       ,__ssivec_bits);
__IO_REG32_BIT(SSIF,              0xFFFFFFF8,__READ_WRITE ,__ssif_bits);
__IO_REG32_BIT(SSIR1_MIRROR,      0xFFFFFFFC,__READ_WRITE ,__ssir1_bits);

/***************************************************************************
 **
 ** SYS2 (Secondary System Control)
 **
 ***************************************************************************/
__IO_REG32_BIT(PLLCTL3,           0xFFFFE100,__READ_WRITE ,__pllctl3_bits);
__IO_REG32_BIT(STCCLKDIV,         0xFFFFE108,__READ_WRITE ,__stcclkdiv_bits);

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
 ** PBIST (Programmable Built-In Self Test)
 **
 ***************************************************************************/
__IO_REG16(    PBIST_A0,          0xFFFFE500,__READ_WRITE );
__IO_REG16(    PBIST_A1,          0xFFFFE504,__READ_WRITE );
__IO_REG16(    PBIST_A2,          0xFFFFE508,__READ_WRITE );
__IO_REG16(    PBIST_A3,          0xFFFFE50C,__READ_WRITE );
__IO_REG16(    PBIST_L0,          0xFFFFE510,__READ_WRITE );
__IO_REG16(    PBIST_L1,          0xFFFFE514,__READ_WRITE );
__IO_REG16(    PBIST_L2,          0xFFFFE518,__READ_WRITE );
__IO_REG16(    PBIST_L3,          0xFFFFE51C,__READ_WRITE );
__IO_REG32_BIT(PBIST_DD0,         0xFFFFE520,__READ_WRITE ,__pbist_dd0_bits);
__IO_REG32_BIT(PBIST_DE0,         0xFFFFE524,__READ_WRITE ,__pbist_de0_bits);
__IO_REG16(    PBIST_CA0,         0xFFFFE530,__READ_WRITE );
__IO_REG16(    PBIST_CA1,         0xFFFFE534,__READ_WRITE );
__IO_REG16(    PBIST_CA2,         0xFFFFE538,__READ_WRITE );
__IO_REG16(    PBIST_CA3,         0xFFFFE53C,__READ_WRITE );
__IO_REG16(    PBIST_CL0,         0xFFFFE540,__READ_WRITE );
__IO_REG16(    PBIST_CL1,         0xFFFFE544,__READ_WRITE );
__IO_REG16(    PBIST_CL2,         0xFFFFE548,__READ_WRITE );
__IO_REG16(    PBIST_CL3,         0xFFFFE54C,__READ_WRITE );
__IO_REG16(    PBIST_I0,          0xFFFFE550,__READ_WRITE );
__IO_REG16(    PBIST_I1,          0xFFFFE554,__READ_WRITE );
__IO_REG16(    PBIST_I2,          0xFFFFE558,__READ_WRITE );
__IO_REG16(    PBIST_I3,          0xFFFFE55C,__READ_WRITE );
__IO_REG32_BIT(PBIST_RAMT,        0xFFFFE560,__READ_WRITE ,__pbist_ramt_bits);
__IO_REG32_BIT(PBIST_DLR,         0xFFFFE564,__READ_WRITE ,__pbist_dlr_bits);
__IO_REG32_BIT(PBIST_CMS,         0xFFFFE568,__READ_WRITE ,__pbist_cms_bits);
__IO_REG32_BIT(PBIST_STR,         0xFFFFE56C,__READ_WRITE ,__pbist_str_bits);
__IO_REG32_BIT(PBIST_CSR,         0xFFFFE578,__READ_WRITE ,__pbist_csr_bits);
__IO_REG8(     PBIST_FDLY,        0xFFFFE57C,__READ_WRITE );
__IO_REG32_BIT(PBIST_PACT,        0xFFFFE580,__READ_WRITE ,__pbist_pact_bits);
__IO_REG8(     PBIST_PBIST_ID,    0xFFFFE584,__READ_WRITE );
__IO_REG32_BIT(PBIST_OVER,        0xFFFFE588,__READ_WRITE ,__pbist_over_bits);
__IO_REG32_BIT(PBIST_FSRF0,       0xFFFFE590,__READ_WRITE ,__pbist_fsrf0_bits);
__IO_REG32_BIT(PBIST_FSRF1,       0xFFFFE594,__READ_WRITE ,__pbist_fsrf1_bits);
__IO_REG8(     PBIST_FSRC0,       0xFFFFE598,__READ       );
__IO_REG8(     PBIST_FSRC1,       0xFFFFE59C,__READ       );
__IO_REG16(    PBIST_FSRA0,       0xFFFFE5A0,__READ       );
__IO_REG16(    PBIST_FSRA1,       0xFFFFE5A4,__READ       );
__IO_REG16(    PBIST_FSRDL0,      0xFFFFE5A8,__READ       );
__IO_REG32(    PBIST_FSRDL1,      0xFFFFE5B0,__READ       );
__IO_REG32_BIT(PBIST_ROM,         0xFFFFE5C0,__READ_WRITE ,__pbist_rom_bits);
__IO_REG32_BIT(PBIST_ALGO,        0xFFFFE5C4,__READ_WRITE ,__pbist_algo_bits);
__IO_REG32_BIT(PBIST_RINFOL,      0xFFFFE5C8,__READ_WRITE ,__pbist_rinfol_bits);
__IO_REG32_BIT(PBIST_RINFOU,      0xFFFFE5CC,__READ_WRITE ,__pbist_rinfou_bits);

/***************************************************************************
 **
 ** STC (CPU Self Test Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(STCGCR0,           0xFFFFE600,__READ_WRITE ,__stcgcr0_bits);
__IO_REG32_BIT(STCGCR1,           0xFFFFE604,__READ_WRITE ,__stcgcr1_bits);
__IO_REG32(    STCTPR,            0xFFFFE608,__READ_WRITE );
__IO_REG32(    STC_CADDR,         0xFFFFE60C,__READ       );
__IO_REG16(    STCCICR,           0xFFFFE610,__READ       );
__IO_REG32_BIT(STCGSTAT,          0xFFFFE614,__READ_WRITE ,__stcgstat_bits);
__IO_REG32_BIT(STCFSTAT,          0xFFFFE618,__READ_WRITE ,__stcfstat_bits);
__IO_REG32(    CPU1_CURMISR3,     0xFFFFE62C,__READ       );
__IO_REG32(    CPU1_CURMISR2,     0xFFFFE630,__READ       );
__IO_REG32(    CPU1_CURMISR1,     0xFFFFE634,__READ       );
__IO_REG32(    CPU1_CURMISR0,     0xFFFFE638,__READ       );
__IO_REG32(    CPU2_CURMISR3,     0xFFFFE63C,__READ       );
__IO_REG32(    CPU2_CURMISR2,     0xFFFFE640,__READ       );
__IO_REG32(    CPU2_CURMISR1,     0xFFFFE644,__READ       );
__IO_REG32(    CPU2_CURMISR0,     0xFFFFE648,__READ       );

/***************************************************************************
 **
 ** TCRAM (Tightly-Coupled RAM)
 **
 ***************************************************************************/
__IO_REG32_BIT(RAMCTRL1,          0xFFFFF800,__READ_WRITE ,__ramctrl_bits);
__IO_REG32_BIT(RAMTHRESHOLD1,     0xFFFFF804,__READ_WRITE ,__ramthreshold_bits);
__IO_REG32_BIT(RAMOCCUR1,         0xFFFFF808,__READ_WRITE ,__ramoccur_bits);
__IO_REG32_BIT(RAMINTCTRL1,       0xFFFFF80C,__READ_WRITE ,__ramintctrl_bits);
__IO_REG32_BIT(RAMERRSTATUS1,     0xFFFFF810,__READ_WRITE ,__ramerrstatus_bits);
__IO_REG32_BIT(RAMSERRADDR1,      0xFFFFF814,__READ       ,__ramserraddr_bits);
__IO_REG32_BIT(RAMUERRADDR1,      0xFFFFF81C,__READ       ,__ramuerraddr_bits);
__IO_REG32_BIT(RAMTEST1,          0xFFFFF830,__READ_WRITE ,__ramtest_bits);
__IO_REG32_BIT(RAMADDRDECVECT1,   0xFFFFF838,__READ_WRITE ,__ramaddrdecvect_bits);
__IO_REG32_BIT(RAMPERRADDR1,      0xFFFFF83C,__READ       ,__ramperraddr_bits);
__IO_REG32_BIT(RAMCTRL2,          0xFFFFF900,__READ_WRITE ,__ramctrl_bits);
__IO_REG32_BIT(RAMTHRESHOLD2,     0xFFFFF904,__READ_WRITE ,__ramthreshold_bits);
__IO_REG32_BIT(RAMOCCUR2,         0xFFFFF908,__READ_WRITE ,__ramoccur_bits);
__IO_REG32_BIT(RAMINTCTRL2,       0xFFFFF90C,__READ_WRITE ,__ramintctrl_bits);
__IO_REG32_BIT(RAMERRSTATUS2,     0xFFFFF910,__READ_WRITE ,__ramerrstatus_bits);
__IO_REG32_BIT(RAMSERRADDR2,      0xFFFFF914,__READ       ,__ramserraddr_bits);
__IO_REG32_BIT(RAMUERRADDR2,      0xFFFFF91C,__READ       ,__ramuerraddr_bits);
__IO_REG32_BIT(RAMTEST2,          0xFFFFF930,__READ_WRITE ,__ramtest_bits);
__IO_REG32_BIT(RAMADDRDECVECT2,   0xFFFFF938,__READ_WRITE ,__ramaddrdecvect_bits);
__IO_REG32_BIT(RAMPERRADDR2,      0xFFFFF93C,__READ       ,__ramperraddr_bits);

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
__IO_REG32_BIT(FEMU_ECC,          0xFFF87060,__READ_WRITE ,__femu_ecc_bits);
__IO_REG32_BIT(FPAR_OVR,          0xFFF8707C,__READ_WRITE ,__fpar_ovr_bits);
__IO_REG32_BIT(FEDACSDIS2,        0xFFF870C0,__READ_WRITE ,__fbprdy_bits);

/***************************************************************************
 **
 ** EMIF (Asynchronous External Memory Interface)
 **
 ***************************************************************************/
__IO_REG32_BIT(ERCSR,             0xFCFFE800,__READ       ,__ercsr_bits);
__IO_REG32_BIT(EA1CR,             0xFCFFE810,__READ_WRITE ,__eacr_bits);
__IO_REG32_BIT(EA2CR,             0xFCFFE814,__READ_WRITE ,__eacr_bits);
__IO_REG32_BIT(EA3CR,             0xFCFFE818,__READ_WRITE ,__eacr_bits);
__IO_REG32_BIT(EA4CR,             0xFCFFE81C,__READ_WRITE ,__eacr_bits);
__IO_REG32_BIT(EIRR,              0xFCFFE840,__READ_WRITE ,__eirr_bits);
__IO_REG32_BIT(EIMR,              0xFCFFE844,__READ_WRITE ,__eimr_bits);
__IO_REG32_BIT(EIMSR,             0xFCFFE848,__READ_WRITE ,__eimsr_bits);
__IO_REG32_BIT(EIMCR,             0xFCFFE84C,__READ_WRITE ,__eimcr_bits);

/***************************************************************************
 **
 ** POM (Parameter Overlay Module)
 **
 ***************************************************************************/
__IO_REG32_BIT(POMGLBCTRL,        0xFFA04000,__READ_WRITE ,__pomglbctrl_bits);
__IO_REG32_BIT(POMREV,            0xFFA04004,__READ       ,__pomrev_bits);
__IO_REG32_BIT(POMPRGSTART0,      0xFFA04200,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART0,      0xFFA04204,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE0,       0xFFA04208,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART1,      0xFFA04210,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART1,      0xFFA04214,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE1,       0xFFA04218,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART2,      0xFFA04220,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART2,      0xFFA04224,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE2,       0xFFA04228,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART3,      0xFFA04230,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART3,      0xFFA04234,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE3,       0xFFA04238,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART4,      0xFFA04240,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART4,      0xFFA04244,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE4,       0xFFA04248,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART5,      0xFFA04250,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART5,      0xFFA04254,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE5,       0xFFA04258,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART6,      0xFFA04260,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART6,      0xFFA04264,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE6,       0xFFA04268,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART7,      0xFFA04270,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART7,      0xFFA04274,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE7,       0xFFA04278,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART8,      0xFFA04280,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART8,      0xFFA04284,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE8,       0xFFA04288,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART9,      0xFFA04290,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART9,      0xFFA04294,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE9,       0xFFA04298,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART10,     0xFFA042A0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART10,     0xFFA042A4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE10,      0xFFA042A8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART11,     0xFFA042B0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART11,     0xFFA042B4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE11,      0xFFA042B8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART12,     0xFFA042C0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART12,     0xFFA042C4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE12,      0xFFA042C8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART13,     0xFFA042D0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART13,     0xFFA042D4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE13,      0xFFA042D8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART14,     0xFFA042E0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART14,     0xFFA042E4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE14,      0xFFA042E8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART15,     0xFFA042F0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART15,     0xFFA042F4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE15,      0xFFA042F8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART16,     0xFFA04300,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART16,     0xFFA04304,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE16,      0xFFA04308,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART17,     0xFFA04310,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART17,     0xFFA04314,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE17,      0xFFA04318,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART18,     0xFFA04320,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART18,     0xFFA04324,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE18,      0xFFA04328,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART19,     0xFFA04330,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART19,     0xFFA04334,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE19,      0xFFA04338,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART20,     0xFFA04340,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART20,     0xFFA04344,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE20,      0xFFA04348,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART21,     0xFFA04350,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART21,     0xFFA04354,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE21,      0xFFA04358,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART22,     0xFFA04360,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART22,     0xFFA04364,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE22,      0xFFA04368,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART23,     0xFFA04370,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART23,     0xFFA04374,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE23,      0xFFA04378,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART24,     0xFFA04380,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART24,     0xFFA04384,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE24,      0xFFA04388,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART25,     0xFFA04390,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART25,     0xFFA04394,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE25,      0xFFA04398,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART26,     0xFFA043A0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART26,     0xFFA043A4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE26,      0xFFA043A8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART27,     0xFFA043B0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART27,     0xFFA043B4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE27,      0xFFA043B8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART28,     0xFFA043C0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART28,     0xFFA043C4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE28,      0xFFA043C8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART29,     0xFFA043D0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART29,     0xFFA043D4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE29,      0xFFA043D8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART30,     0xFFA043E0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART30,     0xFFA043E4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE30,      0xFFA043E8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMPRGSTART31,     0xFFA043F0,__READ_WRITE ,__pomprgstartx_bits);
__IO_REG32_BIT(POMOVLSTART31,     0xFFA043F4,__READ_WRITE ,__pomovlstartx_bits);
__IO_REG32_BIT(POMREGSIZE31,      0xFFA043F8,__READ_WRITE ,__pomregsizex_bits);
__IO_REG32_BIT(POMCLAIMSET,       0xFFA04FA0,__READ_WRITE ,__pomclaimset_bits);
__IO_REG32_BIT(POMCLAIMCLR,       0xFFA04FA4,__READ_WRITE ,__pomclaimclr_bits);
__IO_REG32_BIT(POMDEVTYPE,        0xFFA04FCC,__READ       ,__pomdevtype_bits);
__IO_REG32_BIT(POMPERIPHERALID4,  0xFFA04FD0,__READ       ,__pomperipheralid4_bits);
__IO_REG32_BIT(POMPERIPHERALID0,  0xFFA04FE0,__READ       ,__pomperipheralid0_bits);
__IO_REG32_BIT(POMPERIPHERALID1,  0xFFA04FE4,__READ       ,__pomperipheralid1_bits);
__IO_REG32_BIT(POMPERIPHERALID2,  0xFFA04FE8,__READ       ,__pomperipheralid2_bits);
__IO_REG32_BIT(POMCOMPONENTID0,   0xFFA04FF0,__READ       ,__pomcomponentid0_bits);
__IO_REG32_BIT(POMCOMPONENTID1,   0xFFA04FF4,__READ       ,__pomcomponentid1_bits);
__IO_REG32_BIT(POMCOMPONENTID2,   0xFFA04FF8,__READ       ,__pomcomponentid0_bits);
__IO_REG32_BIT(POMCOMPONENTID3,   0xFFA04FFC,__READ       ,__pomcomponentid0_bits);

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

/***************************************************************************
 **
 ** SCI1/LIN1 (Serial Communication Interface/Local InterconnectNetwork)
 **
 ***************************************************************************/
__IO_REG32_BIT(SCI1GCR0,          0xFFF7E400,__READ_WRITE ,__scigcr0_bits);
__IO_REG32_BIT(SCI1GCR1,          0xFFF7E404,__READ_WRITE ,__scigcr1_bits);
__IO_REG32_BIT(SCI1GCR2,          0xFFF7E408,__READ_WRITE ,__scigcr2_bits);
__IO_REG32_BIT(SCI1SETINT,        0xFFF7E40C,__READ_WRITE ,__scisetint_bits);
__IO_REG32_BIT(SCI1CLEARINT,      0xFFF7E410,__READ_WRITE ,__sciclearint_bits);
__IO_REG32_BIT(SCI1SETINTLVL,     0xFFF7E414,__READ_WRITE ,__scisetintlvl_bits);
__IO_REG32_BIT(SCI1CLEARINTLVL,   0xFFF7E418,__READ_WRITE ,__sciclearintlvl_bits);
__IO_REG32_BIT(SCI1FLR,           0xFFF7E41C,__READ_WRITE ,__sciflr_bits);
__IO_REG32_BIT(SCI1INTVECT0,      0xFFF7E420,__READ       ,__sciintvect0_bits);
__IO_REG32_BIT(SCI1INTVECT1,      0xFFF7E424,__READ       ,__sciintvect1_bits);
__IO_REG32_BIT(SCI1FORMAT,        0xFFF7E428,__READ_WRITE ,__sciformat_bits);
__IO_REG32_BIT(SCI1BRS,           0xFFF7E42C,__READ_WRITE ,__scibrs_bits);
__IO_REG32_BIT(SCI1ED,            0xFFF7E430,__READ       ,__scied_bits);
__IO_REG32_BIT(SCI1RD,            0xFFF7E434,__READ       ,__scird_bits);
__IO_REG32_BIT(SCI1TD,            0xFFF7E438,__READ_WRITE ,__scitd_bits);
__IO_REG32_BIT(SCI1PIO0,          0xFFF7E43C,__READ_WRITE ,__scipio0_bits);
__IO_REG32_BIT(SCI1PIO1,          0xFFF7E440,__READ_WRITE ,__scipio1_bits);
__IO_REG32_BIT(SCI1PIO2,          0xFFF7E444,__READ       ,__scipio2_bits);
__IO_REG32_BIT(SCI1PIO3,          0xFFF7E448,__READ_WRITE ,__scipio3_bits);
__IO_REG32_BIT(SCI1PIO4,          0xFFF7E44C,__READ_WRITE ,__scipio4_bits);
__IO_REG32_BIT(SCI1PIO5,          0xFFF7E450,__READ_WRITE ,__scipio5_bits);
__IO_REG32_BIT(SCI1PIO6,          0xFFF7E454,__READ_WRITE ,__scipio6_bits);
__IO_REG32_BIT(SCI1PIO7,          0xFFF7E458,__READ_WRITE ,__scipio7_bits);
__IO_REG32_BIT(SCI1PIO8,          0xFFF7E45C,__READ_WRITE ,__scipio8_bits);
__IO_REG32_BIT(LIN1COMPARE,       0xFFF7E460,__READ_WRITE ,__lincompare_bits);
__IO_REG32_BIT(LIN1RD0,           0xFFF7E464,__READ       ,__linrd0_bits);
__IO_REG32_BIT(LIN1RD1,           0xFFF7E468,__READ       ,__linrd1_bits);
__IO_REG32_BIT(LIN1MASK,          0xFFF7E46C,__READ_WRITE ,__linmask_bits);
__IO_REG32_BIT(LIN1ID,            0xFFF7E470,__READ_WRITE ,__linid_bits);
__IO_REG32_BIT(LIN1TD0,           0xFFF7E474,__READ_WRITE ,__lintd0_bits);
__IO_REG32_BIT(LIN1TD1,           0xFFF7E478,__READ_WRITE ,__lintd1_bits);
__IO_REG32_BIT(LIN1MBRS,          0xFFF7E47C,__READ_WRITE ,__linmbrs_bits);
__IO_REG32_BIT(IO1DFTCTRL,        0xFFF7E490,__READ_WRITE ,__iodftctrl_bits);

/***************************************************************************
 **
 ** SCI2/LIN2 (Serial Communication Interface/Local InterconnectNetwork)
 **
 ***************************************************************************/
__IO_REG32_BIT(SCI2GCR0,          0xFFF7E500,__READ_WRITE ,__scigcr0_bits);
__IO_REG32_BIT(SCI2GCR1,          0xFFF7E504,__READ_WRITE ,__scigcr1_bits);
__IO_REG32_BIT(SCI2GCR2,          0xFFF7E508,__READ_WRITE ,__scigcr2_bits);
__IO_REG32_BIT(SCI2SETINT,        0xFFF7E50C,__READ_WRITE ,__scisetint_bits);
__IO_REG32_BIT(SCI2CLEARINT,      0xFFF7E510,__READ_WRITE ,__sciclearint_bits);
__IO_REG32_BIT(SCI2SETINTLVL,     0xFFF7E514,__READ_WRITE ,__scisetintlvl_bits);
__IO_REG32_BIT(SCI2CLEARINTLVL,   0xFFF7E518,__READ_WRITE ,__sciclearintlvl_bits);
__IO_REG32_BIT(SCI2FLR,           0xFFF7E51C,__READ_WRITE ,__sciflr_bits);
__IO_REG32_BIT(SCI2INTVECT0,      0xFFF7E520,__READ       ,__sciintvect0_bits);
__IO_REG32_BIT(SCI2INTVECT1,      0xFFF7E524,__READ       ,__sciintvect1_bits);
__IO_REG32_BIT(SCI2FORMAT,        0xFFF7E528,__READ_WRITE ,__sciformat_bits);
__IO_REG32_BIT(SCI2BRS,           0xFFF7E52C,__READ_WRITE ,__scibrs_bits);
__IO_REG32_BIT(SCI2ED,            0xFFF7E530,__READ       ,__scied_bits);
__IO_REG32_BIT(SCI2RD,            0xFFF7E534,__READ       ,__scird_bits);
__IO_REG32_BIT(SCI2TD,            0xFFF7E538,__READ_WRITE ,__scitd_bits);
__IO_REG32_BIT(SCI2PIO0,          0xFFF7E53C,__READ_WRITE ,__scipio0_bits);
__IO_REG32_BIT(SCI2PIO1,          0xFFF7E540,__READ_WRITE ,__scipio1_bits);
__IO_REG32_BIT(SCI2PIO2,          0xFFF7E544,__READ       ,__scipio2_bits);
__IO_REG32_BIT(SCI2PIO3,          0xFFF7E548,__READ_WRITE ,__scipio3_bits);
__IO_REG32_BIT(SCI2PIO4,          0xFFF7E54C,__READ_WRITE ,__scipio4_bits);
__IO_REG32_BIT(SCI2PIO5,          0xFFF7E550,__READ_WRITE ,__scipio5_bits);
__IO_REG32_BIT(SCI2PIO6,          0xFFF7E554,__READ_WRITE ,__scipio6_bits);
__IO_REG32_BIT(SCI2PIO7,          0xFFF7E558,__READ_WRITE ,__scipio7_bits);
__IO_REG32_BIT(SCI2PIO8,          0xFFF7E55C,__READ_WRITE ,__scipio8_bits);
__IO_REG32_BIT(LIN2COMPARE,       0xFFF7E560,__READ_WRITE ,__lincompare_bits);
__IO_REG32_BIT(LIN2RD0,           0xFFF7E564,__READ       ,__linrd0_bits);
__IO_REG32_BIT(LIN2RD1,           0xFFF7E568,__READ       ,__linrd1_bits);
__IO_REG32_BIT(LIN2MASK,          0xFFF7E56C,__READ_WRITE ,__linmask_bits);
__IO_REG32_BIT(LIN2ID,            0xFFF7E570,__READ_WRITE ,__linid_bits);
__IO_REG32_BIT(LIN2TD0,           0xFFF7E574,__READ_WRITE ,__lintd0_bits);
__IO_REG32_BIT(LIN2TD1,           0xFFF7E578,__READ_WRITE ,__lintd1_bits);
__IO_REG32_BIT(LIN2MBRS,          0xFFF7E57C,__READ_WRITE ,__linmbrs_bits);
__IO_REG32_BIT(IO2DFTCTRL,        0xFFF7E590,__READ_WRITE ,__iodftctrl_bits);

/***************************************************************************
 **
 ** MibSPIP5 (Multi-Buffered Serial Peripheral Interface with Parallel Pin)
 **
 ***************************************************************************/
__IO_REG32_BIT(MibSPIP5GCR0,      0xFFF7FC00,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(MibSPIP5GCR1,      0xFFF7FC04,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(MibSPIP5INT0,      0xFFF7FC08,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(MibSPIP5LVL,       0xFFF7FC0C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(MibSPIP5FLG,       0xFFF7FC10,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(MibSPIP5PC0,       0xFFF7FC14,__READ_WRITE ,__spippc0_bits);
__IO_REG32_BIT(MibSPIP5PC1,       0xFFF7FC18,__READ_WRITE ,__spippc1_bits);
__IO_REG32_BIT(MibSPIP5PC2,       0xFFF7FC1C,__READ       ,__spippc2_bits);
__IO_REG32_BIT(MibSPIP5PC3,       0xFFF7FC20,__READ_WRITE ,__spippc3_bits);
__IO_REG32_BIT(MibSPIP5PC4,       0xFFF7FC24,__READ_WRITE ,__spippc4_bits);
__IO_REG32_BIT(MibSPIP5PC5,       0xFFF7FC28,__READ_WRITE ,__spippc5_bits);
__IO_REG32_BIT(MibSPIP5PC6,       0xFFF7FC2C,__READ_WRITE ,__spippc6_bits);
__IO_REG32_BIT(MibSPIP5PC7,       0xFFF7FC30,__READ_WRITE ,__spippc7_bits);
__IO_REG32_BIT(MibSPIP5PC8,       0xFFF7FC34,__READ_WRITE ,__spippc8_bits);
__IO_REG32_BIT(MibSPIP5DAT0,      0xFFF7FC38,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(MibSPIP5DAT1,      0xFFF7FC3C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(MibSPIP5BUF,       0xFFF7FC40,__READ       ,__spibuf_bits);
__IO_REG32_BIT(MibSPIP5EMU,       0xFFF7FC44,__READ       ,__spiemu_bits);
__IO_REG32_BIT(MibSPIP5DELAY,     0xFFF7FC48,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(MibSPIP5DEF,       0xFFF7FC4C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(MibSPIP5FMT0,      0xFFF7FC50,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPIP5FMT1,      0xFFF7FC54,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPIP5FMT2,      0xFFF7FC58,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPIP5FMT3,      0xFFF7FC5C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(TG5INTVECT0,       0xFFF7FC60,__READ       ,__tgintvect0_bits);
__IO_REG32_BIT(TG5NTVECT1,        0xFFF7FC64,__READ       ,__tgintvect1_bits);
__IO_REG32_BIT(MibSPIP5PC9,       0xFFF7FC68,__READ_WRITE ,__spippc9_bits);
__IO_REG32_BIT(MibSPIP5PMCTRL,    0xFFF7FC6C,__READ_WRITE ,__spipmctrl_bits);
__IO_REG32_BIT(MibSPIP5MIBSPIE,   0xFFF7FC70,__READ_WRITE ,__spimibspie_bits);
__IO_REG32_BIT(TG5ITENST,         0xFFF7FC74,__READ_WRITE ,__tgitenst_bits);
__IO_REG32_BIT(TG5ITENCR,         0xFFF7FC78,__READ_WRITE ,__tgitencr_bits);
__IO_REG32_BIT(TG5ITLVST,         0xFFF7FC7C,__READ_WRITE ,__tgitlvst_bits);
__IO_REG32_BIT(TG5ITLVCR,         0xFFF7FC80,__READ_WRITE ,__tgitlvcr_bits);
__IO_REG32_BIT(TG5ITFLG,          0xFFF7FC84,__READ_WRITE ,__tgitflg_bits);
__IO_REG32_BIT(TG5TICKCNT,        0xFFF7FC90,__READ_WRITE ,__tgtickcnt_bits);
__IO_REG32_BIT(TG5LTGPEND,        0xFFF7FC94,__READ_WRITE ,__tgltgpend_bits);
__IO_REG32_BIT(TG5CTRL0,          0xFFF7FC98,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL1,          0xFFF7FC9C,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL2,          0xFFF7FCA0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL3,          0xFFF7FCA4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL4,          0xFFF7FCA8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL5,          0xFFF7FCAC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL6,          0xFFF7FCB0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL7,          0xFFF7FCB4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL8,          0xFFF7FCB8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL9,          0xFFF7FCBC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL10,         0xFFF7FCC0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL11,         0xFFF7FCC4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL12,         0xFFF7FCC8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL13,         0xFFF7FCCC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL14,         0xFFF7FCD0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG5CTRL15,         0xFFF7FCD4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(MibSPIP5DMA0CTRL,  0xFFF7FCD8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA1CTRL,  0xFFF7FCDC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA2CTRL,  0xFFF7FCE0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA3CTRL,  0xFFF7FCE4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA4CTRL,  0xFFF7FCE8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA5CTRL,  0xFFF7FCEC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA6CTRL,  0xFFF7FCF0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA7CTRL,  0xFFF7FCF4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPIP5DMA0COUNT, 0xFFF7FCF8,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA1COUNT, 0xFFF7FCFC,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA2COUNT, 0xFFF7FD00,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA3COUNT, 0xFFF7FD04,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA4COUNT, 0xFFF7FD08,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA5COUNT, 0xFFF7FD0C,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA6COUNT, 0xFFF7FD10,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMA7COUNT, 0xFFF7FD14,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPIP5DMACNTLEN, 0xFFF7FD18,__READ_WRITE ,__spidmacntlen_bits);
__IO_REG32_BIT(MibSPIP5UERRCTRL,  0xFFF7FD20,__READ_WRITE ,__spiuerrctrl_bits);
__IO_REG32_BIT(MibSPIP5UERRSTAT,  0xFFF7FD24,__READ_WRITE ,__spiuerrstat_bits);
__IO_REG32_BIT(MibSPIP5UERRADDR1, 0xFFF7FD28,__READ_WRITE ,__spiuerraddr1_bits);
__IO_REG32_BIT(MibSPIP5UERRADDR0, 0xFFF7FD2C,__READ_WRITE ,__spiuerraddr0_bits);
__IO_REG32_BIT(MibSPIP5RXOVRN_BUF_ADDR,0xFFF7FD30,__READ  ,__spirxovrn_buf_addr_bits);
__IO_REG32_BIT(MibSPIP5IOLPBKTSTCR,0xFFF7FD34,__READ_WRITE,__spiiolpbktstcr_bits);
__IO_REG32(MibSPIP5_BUFER_TX_BASE,0xFF0A0000,__READ_WRITE );
__IO_REG32(MibSPIP5_BUFER_RX_BASE,0xFF0A0200,__READ_WRITE );

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
__IO_REG32_BIT(TG1NTVECT1,        0xFFF7F464,__READ       ,__tgintvect1_bits);
__IO_REG32_BIT(MibSPI1PC9,        0xFFF7F468,__READ_WRITE ,__spipc9_bits);
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
 ** MibSPI3 (Multi-Buffered Serial Peripheral Interface)
 **
 ***************************************************************************/
__IO_REG32_BIT(MibSPI3GCR0,       0xFFF7F800,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(MibSPI3GCR1,       0xFFF7F804,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(MibSPI3INT0,       0xFFF7F808,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(MibSPI3LVL,        0xFFF7F80C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(MibSPI3FLG,        0xFFF7F810,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(MibSPI3PC0,        0xFFF7F814,__READ_WRITE ,__spipc0_bits);
__IO_REG32_BIT(MibSPI3PC1,        0xFFF7F818,__READ_WRITE ,__spipc1_bits);
__IO_REG32_BIT(MibSPI3PC2,        0xFFF7F81C,__READ       ,__spipc2_bits);
__IO_REG32_BIT(MibSPI3PC3,        0xFFF7F820,__READ_WRITE ,__spipc3_bits);
__IO_REG32_BIT(MibSPI3PC4,        0xFFF7F824,__READ_WRITE ,__spipc4_bits);
__IO_REG32_BIT(MibSPI3PC5,        0xFFF7F828,__READ_WRITE ,__spipc5_bits);
__IO_REG32_BIT(MibSPI3PC6,        0xFFF7F82C,__READ_WRITE ,__spipc6_bits);
__IO_REG32_BIT(MibSPI3PC7,        0xFFF7F830,__READ_WRITE ,__spipc7_bits);
__IO_REG32_BIT(MibSPI3PC8,        0xFFF7F834,__READ_WRITE ,__spipc8_bits);
__IO_REG32_BIT(MibSPI3DAT0,       0xFFF7F838,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(MibSPI3DAT1,       0xFFF7F83C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(MibSPI3BUF,        0xFFF7F840,__READ       ,__spibuf_bits);
__IO_REG32_BIT(MibSPI3EMU,        0xFFF7F844,__READ       ,__spiemu_bits);
__IO_REG32_BIT(MibSPI3DELAY,      0xFFF7F848,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(MibSPI3DEF,        0xFFF7F84C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(MibSPI3FMT0,       0xFFF7F850,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPI3FMT1,       0xFFF7F854,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPI3FMT2,       0xFFF7F858,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(MibSPI3FMT3,       0xFFF7F85C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(TG3INTVECT0,       0xFFF7F860,__READ       ,__tgintvect0_bits);
__IO_REG32_BIT(TG3NTVECT1,        0xFFF7F864,__READ       ,__tgintvect1_bits);
__IO_REG32_BIT(MibSPI3PC9,        0xFFF7F868,__READ_WRITE ,__spipc9_bits);
__IO_REG32_BIT(MibSPI3PMCTRL,     0xFFF7F86C,__READ_WRITE ,__spipmctrl_bits);
__IO_REG32_BIT(MibSPI3MIBSPIE,    0xFFF7F870,__READ_WRITE ,__spimibspie_bits);
__IO_REG32_BIT(TG3ITENST,         0xFFF7F874,__READ_WRITE ,__tgitenst_bits);
__IO_REG32_BIT(TG3ITENCR,         0xFFF7F878,__READ_WRITE ,__tgitencr_bits);
__IO_REG32_BIT(TG3ITLVST,         0xFFF7F87C,__READ_WRITE ,__tgitlvst_bits);
__IO_REG32_BIT(TG3ITLVCR,         0xFFF7F880,__READ_WRITE ,__tgitlvcr_bits);
__IO_REG32_BIT(TG3ITFLG,          0xFFF7F884,__READ_WRITE ,__tgitflg_bits);
__IO_REG32_BIT(TG3TICKCNT,        0xFFF7F890,__READ_WRITE ,__tgtickcnt_bits);
__IO_REG32_BIT(TG3LTGPEND,        0xFFF7F894,__READ_WRITE ,__tgltgpend_bits);
__IO_REG32_BIT(TG3CTRL0,          0xFFF7F898,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL1,          0xFFF7F89C,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL2,          0xFFF7F8A0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL3,          0xFFF7F8A4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL4,          0xFFF7F8A8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL5,          0xFFF7F8AC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL6,          0xFFF7F8B0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL7,          0xFFF7F8B4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL8,          0xFFF7F8B8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL9,          0xFFF7F8BC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL10,         0xFFF7F8C0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL11,         0xFFF7F8C4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL12,         0xFFF7F8C8,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL13,         0xFFF7F8CC,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL14,         0xFFF7F8D0,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(TG3CTRL15,         0xFFF7F8D4,__READ_WRITE ,__tgctrl_bits);
__IO_REG32_BIT(MibSPI3DMA0CTRL,   0xFFF7F8D8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA1CTRL,   0xFFF7F8DC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA2CTRL,   0xFFF7F8E0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA3CTRL,   0xFFF7F8E4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA4CTRL,   0xFFF7F8E8,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA5CTRL,   0xFFF7F8EC,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA6CTRL,   0xFFF7F8F0,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA7CTRL,   0xFFF7F8F4,__READ_WRITE ,__spidmactrl_bits);
__IO_REG32_BIT(MibSPI3DMA0COUNT,  0xFFF7F8F8,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA1COUNT,  0xFFF7F8FC,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA2COUNT,  0xFFF7F900,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA3COUNT,  0xFFF7F904,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA4COUNT,  0xFFF7F908,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA5COUNT,  0xFFF7F90C,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA6COUNT,  0xFFF7F910,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMA7COUNT,  0xFFF7F914,__READ_WRITE ,__spidmacount_bits);
__IO_REG32_BIT(MibSPI3DMACNTLEN,  0xFFF7F918,__READ_WRITE ,__spidmacntlen_bits);
__IO_REG32_BIT(MibSPI3UERRCTRL,   0xFFF7F920,__READ_WRITE ,__spiuerrctrl_bits);
__IO_REG32_BIT(MibSPI3UERRSTAT,   0xFFF7F924,__READ_WRITE ,__spiuerrstat_bits);
__IO_REG32_BIT(MibSPI3UERRADDR1,  0xFFF7F928,__READ_WRITE ,__spiuerraddr1_bits);
__IO_REG32_BIT(MibSPI3UERRADDR0,  0xFFF7F92C,__READ_WRITE ,__spiuerraddr0_bits);
__IO_REG32_BIT(MibSPI3RXOVRN_BUF_ADDR,0xFFF7F930,__READ   ,__spirxovrn_buf_addr_bits);
__IO_REG32_BIT(MibSPI3IOLPBKTSTCR,0xFFF7F934,__READ_WRITE ,__spiiolpbktstcr_bits);
__IO_REG32(MibSPI3_BUFER_TX_BASE, 0xFF0C0000,__READ_WRITE );
__IO_REG32(MibSPI3_BUFER_RX_BASE, 0xFF0C0200,__READ_WRITE );

/***************************************************************************
 **
 ** ADC1 (Analog To Digital Converter)
 **
 ***************************************************************************/
__IO_REG32_BIT(AD1RSTCR,          0xFFF7C000,__READ_WRITE ,__adrstcr_bits);
__IO_REG32_BIT(AD1OPMODECR,       0xFFF7C004,__READ_WRITE ,__adopmodecr_bits);
__IO_REG32_BIT(AD1CLOCKCR,        0xFFF7C008,__READ_WRITE ,__adclockcr_bits);
__IO_REG32_BIT(AD1CALCR,          0xFFF7C00C,__READ_WRITE ,__adcalcr_bits);
__IO_REG32_BIT(AD1EVMODECR,       0xFFF7C010,__READ_WRITE ,__adevmodecr_bits);
__IO_REG32_BIT(AD1G1MODECR,       0xFFF7C014,__READ_WRITE ,__adg1modecr_bits);
__IO_REG32_BIT(AD1G2MODECR,       0xFFF7C018,__READ_WRITE ,__adg2modecr_bits);
__IO_REG32_BIT(AD1EVSRC,          0xFFF7C01C,__READ_WRITE ,__adevsrc_bits);
__IO_REG32_BIT(AD1G1SRC,          0xFFF7C020,__READ_WRITE ,__adg1src_bits);
__IO_REG32_BIT(AD1G2SRC,          0xFFF7C024,__READ_WRITE ,__adg2src_bits);
__IO_REG32_BIT(AD1EVINTENA,       0xFFF7C028,__READ_WRITE ,__adevintena_bits);
__IO_REG32_BIT(AD1G1INTENA,       0xFFF7C02C,__READ_WRITE ,__adg1intena_bits);
__IO_REG32_BIT(AD1G2INTENA,       0xFFF7C030,__READ_WRITE ,__adg2intena_bits);
__IO_REG32_BIT(AD1EVINTFLG,       0xFFF7C034,__READ       ,__adevintflg_bits);
__IO_REG32_BIT(AD1G1INTFLG,       0xFFF7C038,__READ       ,__adg1intflg_bits);
__IO_REG32_BIT(AD1G2INTFLG,       0xFFF7C03C,__READ       ,__adg2intflg_bits);
__IO_REG32_BIT(AD1EVINTCR,        0xFFF7C040,__READ_WRITE ,__adevintcr_bits);
__IO_REG32_BIT(AD1G1INTCR,        0xFFF7C044,__READ_WRITE ,__adg1intcr_bits);
__IO_REG32_BIT(AD1G2INTCR,        0xFFF7C048,__READ_WRITE ,__adg2intcr_bits);
__IO_REG32_BIT(AD1EVDMACR,        0xFFF7C04C,__READ_WRITE ,__adevdmacr_bits);
__IO_REG32_BIT(AD1G1DMACR,        0xFFF7C050,__READ_WRITE ,__adg1dmacr_bits);
__IO_REG32_BIT(AD1G2DMACR,        0xFFF7C054,__READ_WRITE ,__adg2dmacr_bits);
__IO_REG32_BIT(AD1BNDCR,          0xFFF7C058,__READ_WRITE ,__adbndcr_bits);
__IO_REG32_BIT(AD1BNDEND,         0xFFF7C05C,__READ_WRITE ,__adbndend_bits);
__IO_REG32_BIT(AD1EVSAMP,         0xFFF7C060,__READ_WRITE ,__adevsamp_bits);
__IO_REG32_BIT(AD1G1SAMP,         0xFFF7C064,__READ_WRITE ,__adg1samp_bits);
__IO_REG32_BIT(AD1G2SAMP,         0xFFF7C068,__READ_WRITE ,__adg2samp_bits);
__IO_REG32_BIT(AD1EVSR,           0xFFF7C06C,__READ       ,__adevsr_bits);
__IO_REG32_BIT(AD1G1SR,           0xFFF7C070,__READ       ,__adg1sr_bits);
__IO_REG32_BIT(AD1G2SR,           0xFFF7C074,__READ       ,__adg2sr_bits);
__IO_REG32_BIT(AD1EVSEL,          0xFFF7C078,__READ_WRITE ,__adevsel_bits);
__IO_REG32_BIT(AD1G1SEL,          0xFFF7C07C,__READ_WRITE ,__adg1sel_bits);
__IO_REG32_BIT(AD1G2SEL,          0xFFF7C080,__READ_WRITE ,__adg2sel_bits);
__IO_REG32_BIT(AD1CALR,           0xFFF7C084,__READ_WRITE ,__adcalr_bits);
__IO_REG32_BIT(AD1SMSTATE,        0xFFF7C088,__READ       ,__adsmstate_bits);
__IO_REG32_BIT(AD1LASTCONV,       0xFFF7C08C,__READ       ,__adlastconv_bits);
__IO_REG32_BIT(AD1EVBUFFER0,      0xFFF7C090,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER1,      0xFFF7C094,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER2,      0xFFF7C098,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER3,      0xFFF7C09C,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER4,      0xFFF7C0A0,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER5,      0xFFF7C0A4,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER6,      0xFFF7C0A8,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1EVBUFFER7,      0xFFF7C0AC,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1G1BUFFER0,      0xFFF7C0B0,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER1,      0xFFF7C0B4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER2,      0xFFF7C0B8,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER3,      0xFFF7C0BC,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER4,      0xFFF7C0C0,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER5,      0xFFF7C0C4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER6,      0xFFF7C0C8,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G1BUFFER7,      0xFFF7C0CC,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER0,      0xFFF7C0D0,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER1,      0xFFF7C0D4,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER2,      0xFFF7C0D8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER3,      0xFFF7C0DC,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER4,      0xFFF7C0E0,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER5,      0xFFF7C0E4,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER6,      0xFFF7C0E8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1G2BUFFER7,      0xFFF7C0EC,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1EVEMUBUFFER,    0xFFF7C0F0,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD1G1EMUBUFFER,    0xFFF7C0F4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD1G2EMUBUFFER,    0xFFF7C0F8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD1EVTDIR,         0xFFF7C0FC,__READ_WRITE ,__adevtdir_bits);
__IO_REG32_BIT(AD1EVTOUT,         0xFFF7C100,__READ_WRITE ,__adevtout_bits);
__IO_REG32_BIT(AD1EVTIN,          0xFFF7C104,__READ       ,__adevtin_bits);
__IO_REG32_BIT(AD1EVTSET,         0xFFF7C108,__READ_WRITE ,__adevtset_bits);
__IO_REG32_BIT(AD1EVTCLR,         0xFFF7C10C,__READ_WRITE ,__adevtclr_bits);
__IO_REG32_BIT(AD1EVTPDR,         0xFFF7C110,__READ_WRITE ,__adevtpdr_bits);
__IO_REG32_BIT(AD1EVTPDIS,        0xFFF7C114,__READ_WRITE ,__adevtpdis_bits);
__IO_REG32_BIT(AD1EVTPSEL,        0xFFF7C118,__READ_WRITE ,__adevtpsel_bits);
__IO_REG32_BIT(AD1EVSAMPDISEN,    0xFFF7C11C,__READ_WRITE ,__adevsampdisen_bits);
__IO_REG32_BIT(AD1G1SAMPDISEN,    0xFFF7C120,__READ_WRITE ,__adg1sampdisen_bits);
__IO_REG32_BIT(AD1G2SAMPDISEN,    0xFFF7C124,__READ_WRITE ,__adg2sampdisen_bits);
__IO_REG32_BIT(AD1MAGINTCR1,      0xFFF7C128,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(AD1MAGINT1MASK,    0xFFF7C12C,__READ_WRITE ,__admagintmask_bits);
__IO_REG32_BIT(AD1MAGINTCR2,      0xFFF7C130,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(AD1MAGINT2MASK,    0xFFF7C134,__READ_WRITE ,__admagintmask_bits);
__IO_REG32_BIT(AD1MAGINTCR3,      0xFFF7C138,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(AD1MAGINT3MASK,    0xFFF7C13C,__READ_WRITE ,__admagintmask_bits);
__IO_REG32_BIT(AD1MAGTHRINTENASET,0xFFF7C158,__READ_WRITE ,__admagthrintenaset_bits);
__IO_REG32_BIT(AD1MAGTHRINTENACLR,0xFFF7C15C,__READ_WRITE ,__admagthrintenaclr_bits);
__IO_REG32_BIT(AD1MAGTHRINTFLG,   0xFFF7C160,__READ_WRITE ,__admagthrintflg_bits);
__IO_REG32_BIT(AD1MAGTHRINTOFFSET,0xFFF7C164,__READ       ,__admagthrintoffset_bits);
__IO_REG32_BIT(AD1EVFIFORESETCR,  0xFFF7C168,__READ_WRITE ,__adevfiforesetcr_bits);
__IO_REG32_BIT(AD1G1FIFORESETCR,  0xFFF7C16C,__READ_WRITE ,__adg1fiforesetcr_bits);
__IO_REG32_BIT(AD1G2FIFORESETCR,  0xFFF7C170,__READ_WRITE ,__adg2fiforesetcr_bits);
__IO_REG32_BIT(AD1EVRAMADDR,      0xFFF7C174,__READ       ,__adevramaddr_bits);
__IO_REG32_BIT(AD1G1RAMADDR,      0xFFF7C178,__READ       ,__adg1ramaddr_bits);
__IO_REG32_BIT(AD1G2RAMADDR,      0xFFF7C17C,__READ       ,__adg2ramaddr_bits);
__IO_REG32_BIT(AD1PARCR,          0xFFF7C180,__READ_WRITE ,__adparcr_bits);
__IO_REG32_BIT(AD1PARADDR,        0xFFF7C184,__READ       ,__adparaddr_bits);
__IO_REG32(    AD1BUFER_BASE,     0xFF3E0000,__READ_WRITE );

/***************************************************************************
 **
 ** ADC2 (Analog To Digital Converter)
 **
 ***************************************************************************/
__IO_REG32_BIT(AD2RSTCR,          0xFFF7C200,__READ_WRITE ,__adrstcr_bits);
__IO_REG32_BIT(AD2OPMODECR,       0xFFF7C204,__READ_WRITE ,__adopmodecr_bits);
__IO_REG32_BIT(AD2CLOCKCR,        0xFFF7C208,__READ_WRITE ,__adclockcr_bits);
__IO_REG32_BIT(AD2CALCR,          0xFFF7C20C,__READ_WRITE ,__adcalcr_bits);
__IO_REG32_BIT(AD2EVMODECR,       0xFFF7C210,__READ_WRITE ,__adevmodecr_bits);
__IO_REG32_BIT(AD2G1MODECR,       0xFFF7C214,__READ_WRITE ,__adg1modecr_bits);
__IO_REG32_BIT(AD2G2MODECR,       0xFFF7C218,__READ_WRITE ,__adg2modecr_bits);
__IO_REG32_BIT(AD2EVSRC,          0xFFF7C21C,__READ_WRITE ,__adevsrc_bits);
__IO_REG32_BIT(AD2G1SRC,          0xFFF7C220,__READ_WRITE ,__adg1src_bits);
__IO_REG32_BIT(AD2G2SRC,          0xFFF7C224,__READ_WRITE ,__adg2src_bits);
__IO_REG32_BIT(AD2EVINTENA,       0xFFF7C228,__READ_WRITE ,__adevintena_bits);
__IO_REG32_BIT(AD2G1INTENA,       0xFFF7C22C,__READ_WRITE ,__adg1intena_bits);
__IO_REG32_BIT(AD2G2INTENA,       0xFFF7C230,__READ_WRITE ,__adg2intena_bits);
__IO_REG32_BIT(AD2EVINTFLG,       0xFFF7C234,__READ       ,__adevintflg_bits);
__IO_REG32_BIT(AD2G1INTFLG,       0xFFF7C238,__READ       ,__adg1intflg_bits);
__IO_REG32_BIT(AD2G2INTFLG,       0xFFF7C23C,__READ       ,__adg2intflg_bits);
__IO_REG32_BIT(AD2EVINTCR,        0xFFF7C240,__READ_WRITE ,__adevintcr_bits);
__IO_REG32_BIT(AD2G1INTCR,        0xFFF7C244,__READ_WRITE ,__adg1intcr_bits);
__IO_REG32_BIT(AD2G2INTCR,        0xFFF7C248,__READ_WRITE ,__adg2intcr_bits);
__IO_REG32_BIT(AD2EVDMACR,        0xFFF7C24C,__READ_WRITE ,__adevdmacr_bits);
__IO_REG32_BIT(AD2G1DMACR,        0xFFF7C250,__READ_WRITE ,__adg1dmacr_bits);
__IO_REG32_BIT(AD2G2DMACR,        0xFFF7C254,__READ_WRITE ,__adg2dmacr_bits);
__IO_REG32_BIT(AD2BNDCR,          0xFFF7C258,__READ_WRITE ,__adbndcr_bits);
__IO_REG32_BIT(AD2BNDEND,         0xFFF7C25C,__READ_WRITE ,__adbndend_bits);
__IO_REG32_BIT(AD2EVSAMP,         0xFFF7C260,__READ_WRITE ,__adevsamp_bits);
__IO_REG32_BIT(AD2G1SAMP,         0xFFF7C264,__READ_WRITE ,__adg1samp_bits);
__IO_REG32_BIT(AD2G2SAMP,         0xFFF7C268,__READ_WRITE ,__adg2samp_bits);
__IO_REG32_BIT(AD2EVSR,           0xFFF7C26C,__READ       ,__adevsr_bits);
__IO_REG32_BIT(AD2G1SR,           0xFFF7C270,__READ       ,__adg1sr_bits);
__IO_REG32_BIT(AD2G2SR,           0xFFF7C274,__READ       ,__adg2sr_bits);
__IO_REG32_BIT(AD2EVSEL,          0xFFF7C278,__READ_WRITE ,__adevsel_bits);
__IO_REG32_BIT(AD2G1SEL,          0xFFF7C27C,__READ_WRITE ,__adg1sel_bits);
__IO_REG32_BIT(AD2G2SEL,          0xFFF7C280,__READ_WRITE ,__adg2sel_bits);
__IO_REG32_BIT(AD2CALR,           0xFFF7C284,__READ_WRITE ,__adcalr_bits);
__IO_REG32_BIT(AD2SMSTATE,        0xFFF7C288,__READ       ,__adsmstate_bits);
__IO_REG32_BIT(AD2LASTCONV,       0xFFF7C28C,__READ       ,__adlastconv_bits);
__IO_REG32_BIT(AD2EVBUFFER0,      0xFFF7C290,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER1,      0xFFF7C294,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER2,      0xFFF7C298,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER3,      0xFFF7C29C,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER4,      0xFFF7C2A0,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER5,      0xFFF7C2A4,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER6,      0xFFF7C2A8,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2EVBUFFER7,      0xFFF7C2AC,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2G1BUFFER0,      0xFFF7C2B0,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER1,      0xFFF7C2B4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER2,      0xFFF7C2B8,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER3,      0xFFF7C2BC,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER4,      0xFFF7C2C0,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER5,      0xFFF7C2C4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER6,      0xFFF7C2C8,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G1BUFFER7,      0xFFF7C2CC,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER0,      0xFFF7C2D0,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER1,      0xFFF7C2D4,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER2,      0xFFF7C2D8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER3,      0xFFF7C2DC,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER4,      0xFFF7C2E0,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER5,      0xFFF7C2E4,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER6,      0xFFF7C2E8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2G2BUFFER7,      0xFFF7C2EC,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2EVEMUBUFFER,    0xFFF7C2F0,__READ_WRITE ,__adevbuffer_bits);
__IO_REG32_BIT(AD2G1EMUBUFFER,    0xFFF7C2F4,__READ_WRITE ,__adg1buffer_bits);
__IO_REG32_BIT(AD2G2EMUBUFFER,    0xFFF7C2F8,__READ_WRITE ,__adg2buffer_bits);
__IO_REG32_BIT(AD2EVTDIR,         0xFFF7C2FC,__READ_WRITE ,__adevtdir_bits);
__IO_REG32_BIT(AD2EVTOUT,         0xFFF7C300,__READ_WRITE ,__adevtout_bits);
__IO_REG32_BIT(AD2EVTIN,          0xFFF7C304,__READ       ,__adevtin_bits);
__IO_REG32_BIT(AD2EVTSET,         0xFFF7C308,__READ_WRITE ,__adevtset_bits);
__IO_REG32_BIT(AD2EVTCLR,         0xFFF7C30C,__READ_WRITE ,__adevtclr_bits);
__IO_REG32_BIT(AD2EVTPDR,         0xFFF7C310,__READ_WRITE ,__adevtpdr_bits);
__IO_REG32_BIT(AD2EVTPDIS,        0xFFF7C314,__READ_WRITE ,__adevtpdis_bits);
__IO_REG32_BIT(AD2EVTPSEL,        0xFFF7C318,__READ_WRITE ,__adevtpsel_bits);
__IO_REG32_BIT(AD2EVSAMPDISEN,    0xFFF7C31C,__READ_WRITE ,__adevsampdisen_bits);
__IO_REG32_BIT(AD2G1SAMPDISEN,    0xFFF7C320,__READ_WRITE ,__adg1sampdisen_bits);
__IO_REG32_BIT(AD2G2SAMPDISEN,    0xFFF7C324,__READ_WRITE ,__adg2sampdisen_bits);
__IO_REG32_BIT(AD2MAGINTCR1,      0xFFF7C328,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(AD2MAGINT1MASK,    0xFFF7C32C,__READ_WRITE ,__admagintmask_bits);
__IO_REG32_BIT(AD2MAGINTCR2,      0xFFF7C330,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(AD2MAGINT2MASK,    0xFFF7C334,__READ_WRITE ,__admagintmask_bits);
__IO_REG32_BIT(AD2MAGINTCR3,      0xFFF7C338,__READ_WRITE ,__admagintcr_bits);
__IO_REG32_BIT(AD2MAGINT3MASK,    0xFFF7C33C,__READ_WRITE ,__admagintmask_bits);
__IO_REG32_BIT(AD2MAGTHRINTENASET,0xFFF7C358,__READ_WRITE ,__admagthrintenaset_bits);
__IO_REG32_BIT(AD2MAGTHRINTENACLR,0xFFF7C35C,__READ_WRITE ,__admagthrintenaclr_bits);
__IO_REG32_BIT(AD2MAGTHRINTFLG,   0xFFF7C360,__READ_WRITE ,__admagthrintflg_bits);
__IO_REG32_BIT(AD2MAGTHRINTOFFSET,0xFFF7C364,__READ       ,__admagthrintoffset_bits);
__IO_REG32_BIT(AD2EVFIFORESETCR,  0xFFF7C368,__READ_WRITE ,__adevfiforesetcr_bits);
__IO_REG32_BIT(AD2G1FIFORESETCR,  0xFFF7C36C,__READ_WRITE ,__adg1fiforesetcr_bits);
__IO_REG32_BIT(AD2G2FIFORESETCR,  0xFFF7C370,__READ_WRITE ,__adg2fiforesetcr_bits);
__IO_REG32_BIT(AD2EVRAMADDR,      0xFFF7C374,__READ       ,__adevramaddr_bits);
__IO_REG32_BIT(AD2G1RAMADDR,      0xFFF7C378,__READ       ,__adg1ramaddr_bits);
__IO_REG32_BIT(AD2G2RAMADDR,      0xFFF7C37C,__READ       ,__adg2ramaddr_bits);
__IO_REG32_BIT(AD2PARCR,          0xFFF7C380,__READ_WRITE ,__adparcr_bits);
__IO_REG32_BIT(AD2PARADDR,        0xFFF7C384,__READ       ,__adparaddr_bits);
__IO_REG32(    AD2BUFER_BASE,     0xFF3A0000,__READ_WRITE );

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
 ** DCAN3 (Controller Area Network)
 **
 ***************************************************************************/
__IO_REG32_BIT(DCAN3CTL,          0xFFF7E000,__READ_WRITE ,__dcanctl_bits);
__IO_REG32_BIT(DCAN3ES,           0xFFF7E004,__READ       ,__dcanes_bits);
__IO_REG32_BIT(DCAN3ERRC,         0xFFF7E008,__READ       ,__dcanerrc_bits);
__IO_REG32_BIT(DCAN3BTR,          0xFFF7E00C,__READ_WRITE ,__dcanbtr_bits);
__IO_REG32_BIT(DCAN3INT,          0xFFF7E010,__READ       ,__dcanint_bits);
__IO_REG32_BIT(DCAN3TEST,         0xFFF7E014,__READ_WRITE ,__dcantest_bits);
__IO_REG32_BIT(DCAN3PERR,         0xFFF7E01C,__READ       ,__dcanperr_bits);
__IO_REG32(    DCAN3ABOTR,        0xFFF7E080,__READ_WRITE );
__IO_REG32_BIT(DCAN3TXRQX,        0xFFF7E084,__READ       ,__dcantxrqx_bits);
__IO_REG32_BIT(DCAN3TXRQ12,       0xFFF7E088,__READ       ,__dcantxrq12_bits);
__IO_REG32_BIT(DCAN3TXRQ34,       0xFFF7E08C,__READ       ,__dcantxrq34_bits);
__IO_REG32_BIT(DCAN3TXRQ56,       0xFFF7E090,__READ       ,__dcantxrq56_bits);
__IO_REG32_BIT(DCAN3TXRQ78,       0xFFF7E094,__READ       ,__dcantxrq78_bits);
__IO_REG32_BIT(DCAN3NWDATX,       0xFFF7E098,__READ       ,__dcannwdatx_bits);
__IO_REG32_BIT(DCAN3NWDAT12,      0xFFF7E09C,__READ       ,__dcannwdat12_bits);
__IO_REG32_BIT(DCAN3NWDAT34,      0xFFF7E0A0,__READ       ,__dcannwdat34_bits);
__IO_REG32_BIT(DCAN3NWDAT56,      0xFFF7E0A4,__READ       ,__dcannwdat56_bits);
__IO_REG32_BIT(DCAN3NWDAT78,      0xFFF7E0A8,__READ       ,__dcannwdat78_bits);
__IO_REG32_BIT(DCAN3INTPND,       0xFFF7E0AC,__READ       ,__dcanintpnd_bits);
__IO_REG32_BIT(DCAN3INTPND12,     0xFFF7E0B0,__READ       ,__dcanintpnd12_bits);
__IO_REG32_BIT(DCAN3INTPND34,     0xFFF7E0B4,__READ       ,__dcanintpnd34_bits);
__IO_REG32_BIT(DCAN3INTPND56,     0xFFF7E0B8,__READ       ,__dcanintpnd56_bits);
__IO_REG32_BIT(DCAN3INTPND78,     0xFFF7E0BC,__READ       ,__dcanintpnd78_bits);
__IO_REG32_BIT(DCAN3MSGVAL,       0xFFF7E0C0,__READ       ,__dcanmsgval_bits);
__IO_REG32_BIT(DCAN3MSGVAL12,     0xFFF7E0C4,__READ       ,__dcanmsgval12_bits);
__IO_REG32_BIT(DCAN3MSGVAL34,     0xFFF7E0C8,__READ       ,__dcanmsgval34_bits);
__IO_REG32_BIT(DCAN3MSGVAL56,     0xFFF7E0CC,__READ       ,__dcanmsgval56_bits);
__IO_REG32_BIT(DCAN3MSGVAL78,     0xFFF7E0D0,__READ       ,__dcanmsgval78_bits);
__IO_REG32_BIT(DCAN3INTMUX12,     0xFFF7E0D8,__READ_WRITE ,__dcanintmux12_bits);
__IO_REG32_BIT(DCAN3INTMUX34,     0xFFF7E0DC,__READ_WRITE ,__dcanintmux34_bits);
__IO_REG32_BIT(DCAN3INTMUX56,     0xFFF7E0E0,__READ_WRITE ,__dcanintmux56_bits);
__IO_REG32_BIT(DCAN3INTMUX78,     0xFFF7E0E4,__READ_WRITE ,__dcanintmux78_bits);
__IO_REG32_BIT(DCAN3IF1CMD,       0xFFF7E100,__READ_WRITE ,__dcanifcmd_bits);
__IO_REG32_BIT(DCAN3IF1MSK,       0xFFF7E104,__READ_WRITE ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN3IF1ARB,       0xFFF7E108,__READ_WRITE ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN3IF1MCTL,      0xFFF7E10C,__READ_WRITE ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN3IF1DATA,      0xFFF7E110,__READ_WRITE ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN3IF1DATB,      0xFFF7E114,__READ_WRITE ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN3IF2CMD,       0xFFF7E120,__READ_WRITE ,__dcanifcmd_bits);
__IO_REG32_BIT(DCAN3IF2MSK,       0xFFF7E124,__READ_WRITE ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN3IF2ARB,       0xFFF7E128,__READ_WRITE ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN3IF2MCTL,      0xFFF7E12C,__READ_WRITE ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN3IF2DATA,      0xFFF7E130,__READ_WRITE ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN3IF2DATB,      0xFFF7E134,__READ_WRITE ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN3IF3OBS,       0xFFF7E140,__READ_WRITE ,__dcanif3obs_bits);
__IO_REG32_BIT(DCAN3IF3MSK,       0xFFF7E144,__READ       ,__dcanifmsk_bits);
__IO_REG32_BIT(DCAN3IF3ARB,       0xFFF7E148,__READ       ,__dcanifarb_bits);
__IO_REG32_BIT(DCAN3IF3MCTL,      0xFFF7E14C,__READ       ,__dcanifmctl_bits);
__IO_REG32_BIT(DCAN3IF3DATA,      0xFFF7E150,__READ       ,__dcanifdata_bits);
__IO_REG32_BIT(DCAN3IF3DATB,      0xFFF7E154,__READ       ,__dcanifdatb_bits);
__IO_REG32_BIT(DCAN3IF3UPD12,     0xFFF7E160,__READ_WRITE ,__dcanif3upd12_bits);
__IO_REG32_BIT(DCAN3IF3UPD34,     0xFFF7E164,__READ_WRITE ,__dcanif3upd34_bits);
__IO_REG32_BIT(DCAN3IF3UPD56,     0xFFF7E168,__READ_WRITE ,__dcanif3upd56_bits);
__IO_REG32_BIT(DCAN3IF3UPD78,     0xFFF7E16C,__READ_WRITE ,__dcanif3upd78_bits);
__IO_REG32_BIT(DCAN3TIOC,         0xFFF7E1E0,__READ_WRITE ,__dcantioc_bits);
__IO_REG32_BIT(DCAN3RIOC,         0xFFF7E1E4,__READ_WRITE ,__dcantioc_bits);
__IO_REG32(    DCAN3RAM_BASE,     0xFF1A0000,__READ_WRITE );

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
__IO_REG32_BIT(HETINTENAS,        0xFFF7B814,__READ_WRITE ,__hetintenas_bits);
__IO_REG32_BIT(HETINTENAC,        0xFFF7B818,__READ_WRITE ,__hetintenac_bits);
__IO_REG32_BIT(HETEXC1,           0xFFF7B81C,__READ_WRITE ,__hetexc1_bits);
__IO_REG32_BIT(HETEXC2,           0xFFF7B820,__READ_WRITE ,__hetexc2_bits);
__IO_REG32_BIT(HETPRY,            0xFFF7B824,__READ_WRITE ,__hetpry_bits);
__IO_REG32_BIT(HETFLG,            0xFFF7B828,__READ_WRITE ,__hetflg_bits);
__IO_REG32_BIT(HETHRSH,           0xFFF7B834,__READ_WRITE ,__hethrsh_bits);
__IO_REG32_BIT(HETXOR,            0xFFF7B838,__READ_WRITE ,__hetxor_bits);
__IO_REG32_BIT(HETREQENS,         0xFFF7B83C,__READ_WRITE ,__hetreqens_bits);
__IO_REG32_BIT(HETREQENC,         0xFFF7B840,__READ_WRITE ,__hetreqenc_bits);
__IO_REG32_BIT(HETREQDS,          0xFFF7B844,__READ_WRITE ,__hetreqds_bits);
__IO_REG32_BIT(HETDIR,            0xFFF7B84C,__READ_WRITE ,__hetdir_bits);
__IO_REG32_BIT(HETDIN,            0xFFF7B850,__READ       ,__hetdin_bits);
__IO_REG32_BIT(HETDOUT,           0xFFF7B854,__READ_WRITE ,__hetdout_bits);
__IO_REG32_BIT(HETDSET,           0xFFF7B858,__READ_WRITE ,__hetdset_bits);
__IO_REG32_BIT(HETDCLR,           0xFFF7B85C,__READ_WRITE ,__hetdclr_bits);
__IO_REG32_BIT(HETPDR,            0xFFF7B860,__READ_WRITE ,__hetpdr_bits);
__IO_REG32_BIT(HETPULDIS,         0xFFF7B864,__READ_WRITE ,__hetpuldis_bits);
__IO_REG32_BIT(HETPSL,            0xFFF7B868,__READ_WRITE ,__hetpsl_bits);
__IO_REG32_BIT(HETPCR,            0xFFF7B874,__READ_WRITE ,__hetpcr_bits);
__IO_REG32_BIT(HETPAR,            0xFFF7B878,__READ       ,__hetpar_bits);
__IO_REG32_BIT(HETPPR,            0xFFF7B87C,__READ_WRITE ,__hetppr_bits);
__IO_REG32_BIT(HETSFPRLD,         0xFFF7B880,__READ_WRITE ,__hetsfprld_bits);
__IO_REG32_BIT(HETSFENA,          0xFFF7B884,__READ_WRITE ,__hetsfena_bits);
__IO_REG32_BIT(HETLBPSEL,         0xFFF7B88C,__READ_WRITE ,__hetlbpsel_bits);
__IO_REG32_BIT(HETLBPDIR,         0xFFF7B890,__READ_WRITE ,__hetlbpdir_bits);
__IO_REG32_BIT(HETPINDIS,         0xFFF7B894,__READ_WRITE ,__hetpindis_bits);
__IO_REG32(    HETP_RAM_BASE,     0xFF460000,__READ_WRITE );

/***************************************************************************
 **
 ** HTU (High End Timer Transfer Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(HTUGC,             0xFFF7A400,__READ_WRITE ,__htugc_bits);
__IO_REG32_BIT(HTUCPENA,          0xFFF7A404,__READ_WRITE ,__htcpena_bits);
__IO_REG32_BIT(HTUBUSY0,          0xFFF7A408,__READ_WRITE ,__htubusy0_bits);
__IO_REG32_BIT(HTUBUSY1,          0xFFF7A40C,__READ_WRITE ,__htubusy1_bits);
__IO_REG32_BIT(HTUBUSY2,          0xFFF7A410,__READ_WRITE ,__htubusy2_bits);
__IO_REG32_BIT(HTUBUSY3,          0xFFF7A414,__READ_WRITE ,__htubusy3_bits);
__IO_REG32_BIT(HTUACP,            0xFFF7A418,__READ_WRITE ,__htuacp_bits);
__IO_REG32_BIT(HTUARLBECTRL,      0xFFF7A420,__READ_WRITE ,__hturlbectrl_bits);
__IO_REG32_BIT(HTUBFINTS,         0xFFF7A424,__READ_WRITE ,__htubfints_bits);
__IO_REG32_BIT(HTUBFINTC,         0xFFF7A428,__READ_WRITE ,__htubfintc_bits);
__IO_REG32_BIT(HTUINTMAP,         0xFFF7A42C,__READ_WRITE ,__htuintmap_bits);
__IO_REG32_BIT(HTUINTOFF0,        0xFFF7A434,__READ       ,__htuintoff0_bits);
__IO_REG32_BIT(HTUINTOFF1,        0xFFF7A438,__READ_WRITE ,__htuintoff1_bits);
__IO_REG32_BIT(HTUBIM,            0xFFF7A43C,__READ_WRITE ,__htubim_bits);
__IO_REG32_BIT(HTURLOSTFL,        0xFFF7A440,__READ_WRITE ,__hturlostfl_bits);
__IO_REG32_BIT(HTUBFINTFL,        0xFFF7A444,__READ_WRITE ,__htubfintfl_bits);
__IO_REG32_BIT(HTUBERINTFL,       0xFFF7A448,__READ_WRITE ,__htuberintfl_bits);
__IO_REG32(    HTUMP1S,           0xFFF7A44C,__READ_WRITE );
__IO_REG32(    HTUMP1E,           0xFFF7A450,__READ_WRITE );
__IO_REG32_BIT(HTUDCRTL,          0xFFF7A454,__READ_WRITE ,__htudcrtl_bits);
__IO_REG32(    HTUWPR,            0xFFF7A458,__READ_WRITE );
__IO_REG32(    HTUWMR,            0xFFF7A45C,__READ_WRITE );
__IO_REG32_BIT(HTUID,             0xFFF7A460,__READ       ,__htuid_bits);
__IO_REG32_BIT(HTUPCR,            0xFFF7A464,__READ_WRITE ,__htupcr_bits);
__IO_REG32_BIT(HTUPAR,            0xFFF7A468,__READ_WRITE ,__htupar_bits);
__IO_REG32_BIT(HTUMPCS,           0xFFF7A470,__READ_WRITE ,__htumpcs_bits);
__IO_REG32(    HTUMP0S,           0xFFF7A474,__READ_WRITE );
__IO_REG32(    HTUMP0E,           0xFFF7A478,__READ_WRITE );
__IO_REG32(    HTUDCP0IFADDRA,    0xFF4E0000,__READ_WRITE );
__IO_REG32(    HTUDCP0IFADDRB,    0xFF4E0004,__READ_WRITE );
__IO_REG32_BIT(HTUDCP0IHADDRCT,   0xFF4E0008,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP0ITCOUNT,    0xFF4E000C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP1IFADDRA,    0xFF4E0010,__READ_WRITE );
__IO_REG32(    HTUDCP1IFADDRB,    0xFF4E0014,__READ_WRITE );
__IO_REG32_BIT(HTUDCP1IHADDRCT,   0xFF4E0018,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP1ITCOUNT,    0xFF4E001C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP2IFADDRA,    0xFF4E0020,__READ_WRITE );
__IO_REG32(    HTUDCP2IFADDRB,    0xFF4E0024,__READ_WRITE );
__IO_REG32_BIT(HTUDCP2IHADDRCT,   0xFF4E0028,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP2ITCOUNT,    0xFF4E002C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP3IFADDRA,    0xFF4E0030,__READ_WRITE );
__IO_REG32(    HTUDCP3IFADDRB,    0xFF4E0034,__READ_WRITE );
__IO_REG32_BIT(HTUDCP3IHADDRCT,   0xFF4E0038,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP3ITCOUNT,    0xFF4E003C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP4IFADDRA,    0xFF4E0040,__READ_WRITE );
__IO_REG32(    HTUDCP4IFADDRB,    0xFF4E0044,__READ_WRITE );
__IO_REG32_BIT(HTUDCP4IHADDRCT,   0xFF4E0048,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP4ITCOUNT,    0xFF4E004C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP5IFADDRA,    0xFF4E0050,__READ_WRITE );
__IO_REG32(    HTUDCP5IFADDRB,    0xFF4E0054,__READ_WRITE );
__IO_REG32_BIT(HTUDCP5IHADDRCT,   0xFF4E0058,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP5ITCOUNT,    0xFF4E005C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP6IFADDRA,    0xFF4E0060,__READ_WRITE );
__IO_REG32(    HTUDCP6IFADDRB,    0xFF4E0064,__READ_WRITE );
__IO_REG32_BIT(HTUDCP6IHADDRCT,   0xFF4E0068,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP6ITCOUNT,    0xFF4E006C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP7IFADDRA,    0xFF4E0070,__READ_WRITE );
__IO_REG32(    HTUDCP7IFADDRB,    0xFF4E0074,__READ_WRITE );
__IO_REG32_BIT(HTUDCP7IHADDRCT,   0xFF4E0078,__READ_WRITE ,__htudcpihaddrct_bits);
__IO_REG32_BIT(HTUDCP7ITCOUNT,    0xFF4E007C,__READ_WRITE ,__htudcpitcount_bits);
__IO_REG32(    HTUDCP0CFADDRA,    0xFF4E0100,__READ_WRITE );
__IO_REG32(    HTUDCP0CFADDRB,    0xFF4E0104,__READ_WRITE );
__IO_REG32_BIT(HTUDCP0CFCOUNT,    0xFF4E0108,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP1CFADDRA,    0xFF4E0110,__READ_WRITE );
__IO_REG32(    HTUDCP1CFADDRB,    0xFF4E0114,__READ_WRITE );
__IO_REG32_BIT(HTUDCP1CFCOUNT,    0xFF4E0118,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP2CFADDRA,    0xFF4E0120,__READ_WRITE );
__IO_REG32(    HTUDCP2CFADDRB,    0xFF4E0124,__READ_WRITE );
__IO_REG32_BIT(HTUDCP2CFCOUNT,    0xFF4E0128,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP3CFADDRA,    0xFF4E0130,__READ_WRITE );
__IO_REG32(    HTUDCP3CFADDRB,    0xFF4E0134,__READ_WRITE );
__IO_REG32_BIT(HTUDCP3CFCOUNT,    0xFF4E0138,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP4CFADDRA,    0xFF4E0140,__READ_WRITE );
__IO_REG32(    HTUDCP4CFADDRB,    0xFF4E0144,__READ_WRITE );
__IO_REG32_BIT(HTUDCP4CFCOUNT,    0xFF4E0148,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP5CFADDRA,    0xFF4E0150,__READ_WRITE );
__IO_REG32(    HTUDCP5CFADDRB,    0xFF4E0154,__READ_WRITE );
__IO_REG32_BIT(HTUDCP5CFCOUNT,    0xFF4E0158,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP6CFADDRA,    0xFF4E0160,__READ_WRITE );
__IO_REG32(    HTUDCP6CFADDRB,    0xFF4E0164,__READ_WRITE );
__IO_REG32_BIT(HTUDCP6CFCOUNT,    0xFF4E0168,__READ_WRITE ,__htudcpcfcount_bits);
__IO_REG32(    HTUDCP7CFADDRA,    0xFF4E0170,__READ_WRITE );
__IO_REG32(    HTUDCP7CFADDRB,    0xFF4E0174,__READ_WRITE );
__IO_REG32_BIT(HTUDCP7CFCOUNT,    0xFF4E0178,__READ_WRITE ,__htudcpcfcount_bits);

/***************************************************************************
 **
 ** DMA (Direct Memory Access Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(DMAGCTRL,          0xFFFFF000,__READ_WRITE ,__dmagctrl_bits);
__IO_REG32_BIT(DMAPEND,           0xFFFFF004,__READ_WRITE ,__dmapend_bits);
__IO_REG32_BIT(DMASTAT,           0xFFFFF00C,__READ       ,__dmastat_bits);
__IO_REG32_BIT(DMAHWCHENAS,       0xFFFFF014,__READ_WRITE ,__dmahwchenas_bits);
__IO_REG32_BIT(DMAHWCHENAR,       0xFFFFF01C,__READ_WRITE ,__dmahwchenar_bits);
__IO_REG32_BIT(DMASWCHENAS,       0xFFFFF024,__READ_WRITE ,__dmaswchenas_bits);
__IO_REG32_BIT(DMASWCHENAR,       0xFFFFF02C,__READ_WRITE ,__dmaswchenar_bits);
__IO_REG32_BIT(DMACHPRIOS,        0xFFFFF034,__READ_WRITE ,__dmachprios_bits);
__IO_REG32_BIT(DMACHPRIOR,        0xFFFFF03C,__READ_WRITE ,__dmachprior_bits);
__IO_REG32_BIT(DMAGCHIENAS,       0xFFFFF044,__READ_WRITE ,__dmagchienas_bits);
__IO_REG32_BIT(DMAGCHIENAR,       0xFFFFF04C,__READ_WRITE ,__dmagchienar_bits);
__IO_REG32_BIT(DMADREQASI0,       0xFFFFF054,__READ_WRITE ,__dmadreqasi0_bits);
__IO_REG32_BIT(DMADREQASI1,       0xFFFFF058,__READ_WRITE ,__dmadreqasi1_bits);
__IO_REG32_BIT(DMADREQASI2,       0xFFFFF05C,__READ_WRITE ,__dmadreqasi2_bits);
__IO_REG32_BIT(DMADREQASI3,       0xFFFFF060,__READ_WRITE ,__dmadreqasi3_bits);
__IO_REG32_BIT(DMADREQASI4,       0xFFFFF064,__READ_WRITE ,__dmadreqasi4_bits);
__IO_REG32_BIT(DMADREQASI5,       0xFFFFF068,__READ_WRITE ,__dmadreqasi5_bits);
__IO_REG32_BIT(DMADREQASI6,       0xFFFFF06C,__READ_WRITE ,__dmadreqasi6_bits);
__IO_REG32_BIT(DMADREQASI7,       0xFFFFF070,__READ_WRITE ,__dmadreqasi7_bits);
__IO_REG32_BIT(DMAPAR0,           0xFFFFF094,__READ_WRITE ,__dmapar0_bits);
__IO_REG32_BIT(DMAPAR1,           0xFFFFF098,__READ_WRITE ,__dmapar1_bits);
__IO_REG32_BIT(DMAPAR2,           0xFFFFF09C,__READ_WRITE ,__dmapar2_bits);
__IO_REG32_BIT(DMAPAR3,           0xFFFFF0A0,__READ_WRITE ,__dmapar3_bits);
__IO_REG32_BIT(DMAFTCMAP,         0xFFFFF0B4,__READ_WRITE ,__dmaftcmap_bits);
__IO_REG32_BIT(DMALFSMAP,         0xFFFFF0BC,__READ_WRITE ,__dmalfsmap_bits);
__IO_REG32_BIT(DMAHBCMAP,         0xFFFFF0C4,__READ_WRITE ,__dmahbcmap_bits);
__IO_REG32_BIT(DMABTCMAP,         0xFFFFF0CC,__READ_WRITE ,__dmabtcmap_bits);
__IO_REG32_BIT(DMABERMAP,         0xFFFFF0D4,__READ_WRITE ,__dmabermap_bits);
__IO_REG32_BIT(DMAFTCINTENAS,     0xFFFFF0DC,__READ_WRITE ,__dmaftcintenas_bits);
__IO_REG32_BIT(DMAFTCINTENAR,     0xFFFFF0E4,__READ_WRITE ,__dmaftcintenar_bits);
__IO_REG32_BIT(DMALFSINTENAS,     0xFFFFF0EC,__READ_WRITE ,__dmalfsintenas_bits);
__IO_REG32_BIT(DMALFSINTENAR,     0xFFFFF0F4,__READ_WRITE ,__dmalfsintenar_bits);
__IO_REG32_BIT(DMAHBCINTENAS,     0xFFFFF0FC,__READ       ,__dmahbcintenas_bits);
__IO_REG32_BIT(DMAHBCINTENAR,     0xFFFFF104,__READ_WRITE ,__dmahbcintenar_bits);
__IO_REG32_BIT(DMABTCINTENAS,     0xFFFFF10C,__READ       ,__dmabtcintenas_bits);
__IO_REG32_BIT(DMABTCINTENAR,     0xFFFFF114,__READ_WRITE ,__dmabtcintenar_bits);
__IO_REG32_BIT(DMAGINTFLAG,       0xFFFFF11C,__READ       ,__dmagintflag_bits);
__IO_REG32_BIT(DMAFTCFLAG,        0xFFFFF124,__READ_WRITE ,__dmaftcflag_bits);
__IO_REG32_BIT(DMALFSFLAG,        0xFFFFF12C,__READ_WRITE ,__dmalfsflag_bits);
__IO_REG32_BIT(DMAHBCFLAG,        0xFFFFF134,__READ_WRITE ,__dmahbcflag_bits);
__IO_REG32_BIT(DMABTCFLAG,        0xFFFFF13C,__READ_WRITE ,__dmabtcflag_bits);
__IO_REG32_BIT(DMABERFLAG,        0xFFFFF144,__READ       ,__dmaberflag_bits);
__IO_REG32_BIT(DMAFTCAOFFSET,     0xFFFFF14C,__READ       ,__dmaftcaoffset_bits);
__IO_REG32_BIT(DMALFSAOFFSET,     0xFFFFF150,__READ       ,__dmalfsaoffset_bits);
__IO_REG32_BIT(DMAHBCAOFFSET,     0xFFFFF154,__READ       ,__dmahbcaoffset_bits);
__IO_REG32_BIT(DMABTCAOFFSET,     0xFFFFF158,__READ       ,__dmabtcaoffset_bits);
__IO_REG32_BIT(DMABERAOFFSET,     0xFFFFF15C,__READ       ,__dmaberaoffset_bits);
__IO_REG32_BIT(DMAFTCBOFFSET,     0xFFFFF160,__READ       ,__dmaftcboffset_bits);
__IO_REG32_BIT(DMALFSBOFFSET,     0xFFFFF164,__READ       ,__dmalfsboffset_bits);
__IO_REG32_BIT(DMAHBCBOFFSET,     0xFFFFF168,__READ       ,__dmahbcboffset_bits);
__IO_REG32_BIT(DMABTCBOFFSET,     0xFFFFF16C,__READ       ,__dmabtcboffset_bits);
__IO_REG32_BIT(DMABERBOFFSET,     0xFFFFF170,__READ       ,__dmaberboffset_bits);
__IO_REG32_BIT(DMAPTCRL,          0xFFFFF178,__READ_WRITE ,__dmaptcrl_bits);
__IO_REG32_BIT(DMARTCTRL,         0xFFFFF17C,__READ_WRITE ,__dmartctrl_bits);
__IO_REG32_BIT(DMADCTRL,          0xFFFFF180,__READ_WRITE ,__dmadctrl_bits);
__IO_REG32(    DMAWPR,            0xFFFFF184,__READ_WRITE );
__IO_REG32(    DMAWMR,            0xFFFFF188,__READ_WRITE );
__IO_REG32(    DMAPBACSADDR,      0xFFFFF198,__READ       );
__IO_REG32(    DMPBACDADDR,       0xFFFFF19C,__READ       );
__IO_REG32_BIT(DMAPBACTC,         0xFFFFF1A0,__READ       ,__dmapbactc_bits);
__IO_REG32_BIT(DMAPCR,            0xFFFFF1A8,__READ_WRITE ,__dmapcr_bits);
__IO_REG32_BIT(DMAPAR,            0xFFFFF1AC,__READ       ,__dmapar_bits);
__IO_REG32_BIT(DMAMPCTRL,         0xFFFFF1B0,__READ_WRITE ,__dmampctrl_bits);
__IO_REG32_BIT(DMAMPST,           0xFFFFF1B4,__READ_WRITE ,__dmampst_bits);
__IO_REG32(    DMAMPROS,          0xFFFFF1B8,__READ_WRITE );
__IO_REG32(    DMAMPR0E,          0xFFFFF1BC,__READ_WRITE );
__IO_REG32(    DMAMPR1S,          0xFFFFF1C0,__READ_WRITE );
__IO_REG32(    DMAMPR1E,          0xFFFFF1C4,__READ_WRITE );
__IO_REG32(    DMAMPR2S,          0xFFFFF1C8,__READ_WRITE );
__IO_REG32(    DMAMPR2E,          0xFFFFF1CC,__READ_WRITE );
__IO_REG32(    DMAMPR3S,          0xFFFFF1D0,__READ_WRITE );
__IO_REG32(    DMAMPR3E,          0xFFFFF1D4,__READ_WRITE );
__IO_REG32(    DMACP0ISADDR,      0xFFF80000,__READ_WRITE );
__IO_REG32(    DMACP0IDADDR,      0xFFF80004,__READ_WRITE );
__IO_REG32_BIT(DMACP0ITCOUNT,     0xFFF80008,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP0CHCTRL,      0xFFF80010,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP0EIOFF,       0xFFF80014,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP0FIOFF,       0xFFF80018,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP1ISADDR,      0xFFF80020,__READ_WRITE );
__IO_REG32(    DMACP1IDADDR,      0xFFF80024,__READ_WRITE );
__IO_REG32_BIT(DMACP1ITCOUNT,     0xFFF80028,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP1CHCTRL,      0xFFF80030,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP1EIOFF,       0xFFF80034,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP1FIOFF,       0xFFF80038,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP2ISADDR,      0xFFF80040,__READ_WRITE );
__IO_REG32(    DMACP2IDADDR,      0xFFF80044,__READ_WRITE );
__IO_REG32_BIT(DMACP2ITCOUNT,     0xFFF80048,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP2CHCTRL,      0xFFF80050,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP2EIOFF,       0xFFF80054,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP2FIOFF,       0xFFF80058,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP3ISADDR,      0xFFF80060,__READ_WRITE );
__IO_REG32(    DMACP3IDADDR,      0xFFF80064,__READ_WRITE );
__IO_REG32_BIT(DMACP3ITCOUNT,     0xFFF80068,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP3CHCTRL,      0xFFF80070,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP3EIOFF,       0xFFF80074,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP3FIOFF,       0xFFF80078,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP4ISADDR,      0xFFF80080,__READ_WRITE );
__IO_REG32(    DMACP4IDADDR,      0xFFF80084,__READ_WRITE );
__IO_REG32_BIT(DMACP4ITCOUNT,     0xFFF80088,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP4CHCTRL,      0xFFF80090,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP4EIOFF,       0xFFF80094,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP4FIOFF,       0xFFF80098,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP5ISADDR,      0xFFF800A0,__READ_WRITE );
__IO_REG32(    DMACP5IDADDR,      0xFFF800A4,__READ_WRITE );
__IO_REG32_BIT(DMACP5ITCOUNT,     0xFFF800A8,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP5CHCTRL,      0xFFF800B0,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP5EIOFF,       0xFFF800B4,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP5FIOFF,       0xFFF800B8,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP6ISADDR,      0xFFF800C0,__READ_WRITE );
__IO_REG32(    DMACP6IDADDR,      0xFFF800C4,__READ_WRITE );
__IO_REG32_BIT(DMACP6ITCOUNT,     0xFFF800C8,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP6CHCTRL,      0xFFF800D0,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP6EIOFF,       0xFFF800D4,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP6FIOFF,       0xFFF800D8,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP7ISADDR,      0xFFF800E0,__READ_WRITE );
__IO_REG32(    DMACP7IDADDR,      0xFFF800E4,__READ_WRITE );
__IO_REG32_BIT(DMACP7ITCOUNT,     0xFFF800E8,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP7CHCTRL,      0xFFF800F0,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP7EIOFF,       0xFFF800F4,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP7FIOFF,       0xFFF800F8,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP8ISADDR,      0xFFF80100,__READ_WRITE );
__IO_REG32(    DMACP8IDADDR,      0xFFF80104,__READ_WRITE );
__IO_REG32_BIT(DMACP8ITCOUNT,     0xFFF80108,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP8CHCTRL,      0xFFF80110,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP8EIOFF,       0xFFF80114,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP8FIOFF,       0xFFF80118,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP9ISADDR,      0xFFF80120,__READ_WRITE );
__IO_REG32(    DMACP9IDADDR,      0xFFF80124,__READ_WRITE );
__IO_REG32_BIT(DMACP9ITCOUNT,     0xFFF80128,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP9CHCTRL,      0xFFF80130,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP9EIOFF,       0xFFF80134,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP9FIOFF,       0xFFF80138,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP10ISADDR,     0xFFF80140,__READ_WRITE );
__IO_REG32(    DMACP10IDADDR,     0xFFF80144,__READ_WRITE );
__IO_REG32_BIT(DMACP10ITCOUNT,    0xFFF80148,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP10CHCTRL,     0xFFF80150,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP10EIOFF,      0xFFF80154,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP10FIOFF,      0xFFF80158,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP11ISADDR,     0xFFF80160,__READ_WRITE );
__IO_REG32(    DMACP11IDADDR,     0xFFF80164,__READ_WRITE );
__IO_REG32_BIT(DMACP11ITCOUNT,    0xFFF80168,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP11CHCTRL,     0xFFF80170,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP11EIOFF,      0xFFF80174,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP11FIOFF,      0xFFF80178,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP12ISADDR,     0xFFF80180,__READ_WRITE );
__IO_REG32(    DMACP12IDADDR,     0xFFF80184,__READ_WRITE );
__IO_REG32_BIT(DMACP12ITCOUNT,    0xFFF80188,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP12CHCTRL,     0xFFF80190,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP12EIOFF,      0xFFF80194,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP12FIOFF,      0xFFF80198,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP13ISADDR,     0xFFF801A0,__READ_WRITE );
__IO_REG32(    DMACP13IDADDR,     0xFFF801A4,__READ_WRITE );
__IO_REG32_BIT(DMACP13ITCOUNT,    0xFFF801A8,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP13CHCTRL,     0xFFF801B0,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP13EIOFF,      0xFFF801B4,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP13FIOFF,      0xFFF801B8,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP14ISADDR,     0xFFF801C0,__READ_WRITE );
__IO_REG32(    DMACP14IDADDR,     0xFFF801C4,__READ_WRITE );
__IO_REG32_BIT(DMACP14ITCOUNT,    0xFFF801C8,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP14CHCTRL,     0xFFF801D0,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP14EIOFF,      0xFFF801D4,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP14FIOFF,      0xFFF801D8,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP15ISADDR,     0xFFF801E0,__READ_WRITE );
__IO_REG32(    DMACP15IDADDR,     0xFFF801E4,__READ_WRITE );
__IO_REG32_BIT(DMACP15ITCOUNT,    0xFFF801E8,__READ_WRITE ,__dmacpitcount_bits);
__IO_REG32_BIT(DMACP15CHCTRL,     0xFFF801F0,__READ_WRITE ,__dmachctrl_bits);
__IO_REG32_BIT(DMACP15EIOFF,      0xFFF801F4,__READ_WRITE ,__dmacpeioff_bits);
__IO_REG32_BIT(DMACP15FIOFF,      0xFFF801F8,__READ_WRITE ,__dmacpfioff_bits);
__IO_REG32(    DMACP0CSADDR,      0xFFF80800,__READ       );
__IO_REG32(    DMACP0CDADDR,      0xFFF80804,__READ       );
__IO_REG32_BIT(DMACP0CTCOUNT,     0xFFF80808,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP1CSADDR,      0xFFF80810,__READ       );
__IO_REG32(    DMACP1CDADDR,      0xFFF80814,__READ       );
__IO_REG32_BIT(DMACP1CTCOUNT,     0xFFF80818,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP2CSADDR,      0xFFF80820,__READ       );
__IO_REG32(    DMACP2CDADDR,      0xFFF80824,__READ       );
__IO_REG32_BIT(DMACP2CTCOUNT,     0xFFF80828,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP3CSADDR,      0xFFF80830,__READ       );
__IO_REG32(    DMACP3CDADDR,      0xFFF80834,__READ       );
__IO_REG32_BIT(DMACP3CTCOUNT,     0xFFF80838,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP4CSADDR,      0xFFF80840,__READ       );
__IO_REG32(    DMACP4CDADDR,      0xFFF80844,__READ       );
__IO_REG32_BIT(DMACP4CTCOUNT,     0xFFF80848,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP5CSADDR,      0xFFF80850,__READ       );
__IO_REG32(    DMACP5CDADDR,      0xFFF80854,__READ       );
__IO_REG32_BIT(DMACP5CTCOUNT,     0xFFF80858,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP6CSADDR,      0xFFF80860,__READ       );
__IO_REG32(    DMACP6CDADDR,      0xFFF80864,__READ       );
__IO_REG32_BIT(DMACP6CTCOUNT,     0xFFF80868,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP7CSADDR,      0xFFF80870,__READ       );
__IO_REG32(    DMACP7CDADDR,      0xFFF80874,__READ       );
__IO_REG32_BIT(DMACP7CTCOUNT,     0xFFF80878,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP8CSADDR,      0xFFF80880,__READ       );
__IO_REG32(    DMACP8CDADDR,      0xFFF80884,__READ       );
__IO_REG32_BIT(DMACP8CTCOUNT,     0xFFF80888,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP9CSADDR,      0xFFF80890,__READ       );
__IO_REG32(    DMACP9CDADDR,      0xFFF80894,__READ       );
__IO_REG32_BIT(DMACP9CTCOUNT,     0xFFF80898,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP10CSADDR,     0xFFF808A0,__READ       );
__IO_REG32(    DMACP10CDADDR,     0xFFF808A4,__READ       );
__IO_REG32_BIT(DMACP10CTCOUNT,    0xFFF808A8,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP11CSADDR,     0xFFF808B0,__READ       );
__IO_REG32(    DMACP11CDADDR,     0xFFF808B4,__READ       );
__IO_REG32_BIT(DMACP11CTCOUNT,    0xFFF808B8,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP12CSADDR,     0xFFF808C0,__READ       );
__IO_REG32(    DMACP12CDADDR,     0xFFF808C4,__READ       );
__IO_REG32_BIT(DMACP12CTCOUNT,    0xFFF808C8,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP13CSADDR,     0xFFF808D0,__READ       );
__IO_REG32(    DMACP13CDADDR,     0xFFF808D4,__READ       );
__IO_REG32_BIT(DMACP13CTCOUNT,    0xFFF808D8,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP14CSADDR,     0xFFF808E0,__READ       );
__IO_REG32(    DMACP14CDADDR,     0xFFF808E4,__READ       );
__IO_REG32_BIT(DMACP14CTCOUNT,    0xFFF808E8,__READ       ,__dmacpctcount_bits);
__IO_REG32(    DMACP15CSADDR,     0xFFF808F0,__READ       );
__IO_REG32(    DMACP15CDADDR,     0xFFF808F4,__READ       );
__IO_REG32_BIT(DMACP15CTCOUNT,    0xFFF808F8,__READ       ,__dmacpctcount_bits);

/***************************************************************************
 **
 ** RTI (Real-Time Interrupt)
 **
 ***************************************************************************/
__IO_REG32_BIT(RTIGCTRL,          0xFFFFFC00,__READ_WRITE ,__rtigctrl_bits);
__IO_REG32_BIT(RTITBCTRL,         0xFFFFFC04,__READ_WRITE ,__rtitbctrl_bits);
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
__IO_REG32(    RTITBLCOMP,        0xFFFFFC70,__READ_WRITE );
__IO_REG32(    RTITBHCOMP,        0xFFFFFC74,__READ_WRITE );
__IO_REG32_BIT(RTISETINTENA,      0xFFFFFC80,__READ_WRITE ,__rtisetintena_bits);
__IO_REG32_BIT(RTICLEARINTENA,    0xFFFFFC84,__READ_WRITE ,__rticlearintena_bits);
__IO_REG32_BIT(RTIINTFLAG,        0xFFFFFC88,__READ_WRITE ,__rtiintflag_bits);

/***************************************************************************
 **
 ** CRC (Cyclic Redundancy Check Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(CRC_CTRL0,         0xFE000000,__READ_WRITE ,__crc_ctrl0_bits);
__IO_REG32_BIT(CRC_CTRL1,         0xFE000008,__READ_WRITE ,__crc_ctrl1_bits);
__IO_REG32_BIT(CRC_CTRL2,         0xFE000010,__READ_WRITE ,__crc_ctrl2_bits);
__IO_REG32_BIT(CRC_INTS,          0xFE000018,__READ_WRITE ,__crc_ints_bits);
__IO_REG32_BIT(CRC_INTR,          0xFE000020,__READ_WRITE ,__crc_intr_bits);
__IO_REG32_BIT(CRC_STATUS,        0xFE000028,__READ_WRITE ,__crc_status_bits);
__IO_REG32_BIT(CRC_INT_OFFSET_REG,0xFE000030,__READ       ,__crc_int_offset_reg_bits);
__IO_REG32_BIT(CRC_BUSY,          0xFE000038,__READ       ,__crc_busy_bits);
__IO_REG32_BIT(CRC_PCOUNT_REG1,   0xFE000040,__READ_WRITE ,__crc_pcount_reg_bits);
__IO_REG32_BIT(CRC_SCOUNT_REG1,   0xFE000044,__READ_WRITE ,__crc_scount_reg_bits);
__IO_REG32_BIT(CRC_CURSEC_REG1,   0xFE000048,__READ       ,__crc_cursec_reg_bits);
__IO_REG32_BIT(CRC_WDTOPLD1,      0xFE00004C,__READ_WRITE ,__crc_wdtopld_bits);
__IO_REG32_BIT(CRC_BCTOPLD1,      0xFE000050,__READ_WRITE ,__crc_bctopld_bits);
__IO_REG32(    PSA_SIGREGL1,      0xFE000060,__READ_WRITE );
__IO_REG32(    PSA_SIGREGH1,      0xFE000064,__READ_WRITE );
__IO_REG32(    CRC_REGL1,         0xFE000068,__READ_WRITE );
__IO_REG32(    CRC_REGH1,         0xFE00006C,__READ_WRITE );
__IO_REG32(    PSA_SECSIGREGL1,   0xFE000070,__READ       );
__IO_REG32(    PSA_SECSIGREGH1,   0xFE000074,__READ       );
__IO_REG32(    RAW_DATAREGL1,     0xFE000078,__READ       );
__IO_REG32(    RAW_DATAREGH1,     0xFE00007C,__READ       );
__IO_REG32_BIT(CRC_PCOUNT_REG2,   0xFE000080,__READ_WRITE ,__crc_pcount_reg_bits);
__IO_REG32_BIT(CRC_SCOUNT_REG2,   0xFE000084,__READ_WRITE ,__crc_scount_reg_bits);
__IO_REG32_BIT(CRC_CURSEC_REG2,   0xFE000088,__READ       ,__crc_cursec_reg_bits);
__IO_REG32_BIT(CRC_WDTOPLD2,      0xFE00008C,__READ_WRITE ,__crc_wdtopld_bits);
__IO_REG32_BIT(CRC_BCTOPLD2,      0xFE000090,__READ_WRITE ,__crc_bctopld_bits);
__IO_REG32(    PSA_SIGREGL2,      0xFE0000A0,__READ_WRITE );
__IO_REG32(    PSA_SIGREGH2,      0xFE0000A4,__READ_WRITE );
__IO_REG32(    CRC_REGL2,         0xFE0000A8,__READ_WRITE );
__IO_REG32(    CRC_REGH2,         0xFE0000AC,__READ_WRITE );
__IO_REG32(    PSA_SECSIGREGL2,   0xFE0000B0,__READ       );
__IO_REG32(    PSA_SECSIGREGH2,   0xFE0000B4,__READ       );
__IO_REG32(    RAW_DATAREGL2,     0xFE0000B8,__READ       );
__IO_REG32(    RAW_DATAREGH2,     0xFE0000BC,__READ       );
__IO_REG32_BIT(CRC_PCOUNT_REG3,   0xFE0000C0,__READ_WRITE ,__crc_pcount_reg_bits);
__IO_REG32_BIT(CRC_SCOUNT_REG3,   0xFE0000C4,__READ_WRITE ,__crc_scount_reg_bits);
__IO_REG32_BIT(CRC_CURSEC_REG3,   0xFE0000C8,__READ       ,__crc_cursec_reg_bits);
__IO_REG32_BIT(CRC_WDTOPLD3,      0xFE0000CC,__READ_WRITE ,__crc_wdtopld_bits);
__IO_REG32_BIT(CRC_BCTOPLD3,      0xFE0000D0,__READ_WRITE ,__crc_bctopld_bits);
__IO_REG32(    PSA_SIGREGL3,      0xFE0000E0,__READ_WRITE );
__IO_REG32(    PSA_SIGREGH3,      0xFE0000E4,__READ_WRITE );
__IO_REG32(    CRC_REGL3,         0xFE0000E8,__READ_WRITE );
__IO_REG32(    CRC_REGH3,         0xFE0000EC,__READ_WRITE );
__IO_REG32(    PSA_SECSIGREGL3,   0xFE0000F0,__READ       );
__IO_REG32(    PSA_SECSIGREGH3,   0xFE0000F4,__READ       );
__IO_REG32(    RAW_DATAREGL3,     0xFE0000F8,__READ       );
__IO_REG32(    RAW_DATAREGH3,     0xFE0000FC,__READ       );
__IO_REG32_BIT(CRC_PCOUNT_REG4,   0xFE000100,__READ_WRITE ,__crc_pcount_reg_bits);
__IO_REG32_BIT(CRC_SCOUNT_REG4,   0xFE000104,__READ_WRITE ,__crc_scount_reg_bits);
__IO_REG32_BIT(CRC_CURSEC_REG4,   0xFE000108,__READ       ,__crc_cursec_reg_bits);
__IO_REG32_BIT(CRC_WDTOPLD4,      0xFE00010C,__READ_WRITE ,__crc_wdtopld_bits);
__IO_REG32_BIT(CRC_BCTOPLD4,      0xFE000110,__READ_WRITE ,__crc_bctopld_bits);
__IO_REG32(    PSA_SIGREGL4,      0xFE000120,__READ_WRITE );
__IO_REG32(    PSA_SIGREGH4,      0xFE000124,__READ_WRITE );
__IO_REG32(    CRC_REGL4,         0xFE000128,__READ_WRITE );
__IO_REG32(    CRC_REGH4,         0xFE00012C,__READ_WRITE );
__IO_REG32(    PSA_SECSIGREGL4,   0xFE000130,__READ       );
__IO_REG32(    PSA_SECSIGREGH4,   0xFE000134,__READ       );
__IO_REG32(    RAW_DATAREGL4,     0xFE000138,__READ       );
__IO_REG32(    RAW_DATAREGH4,     0xFE00013C,__READ       );
__IO_REG32_BIT(MCRC_BUS_SEL,      0xFE000140,__READ_WRITE ,__mcrc_bus_sel_bits);

/***************************************************************************
 **
 ** CCM-R4F (CPU Compare R4F)
 **
 ***************************************************************************/
__IO_REG32_BIT(CCMSR,             0xFFFFF600,__READ_WRITE ,__ccmsr_bits);
__IO_REG32_BIT(CCMKEYR,           0xFFFFF604,__READ_WRITE ,__ccmkeyr_bits);

/***************************************************************************
 **
 ** ESM (Error Signaling Module)
 **
 ***************************************************************************/
__IO_REG32_BIT(ESMIEPSR1,         0xFFFFF500,__READ_WRITE ,__esmiepsr1_bits);
__IO_REG32_BIT(ESMIEPCR1,         0xFFFFF504,__READ_WRITE ,__esmiepcr1_bits);
__IO_REG32_BIT(ESMIESR1,          0xFFFFF508,__READ_WRITE ,__esmiesr1_bits);
__IO_REG32_BIT(ESMIECR1,          0xFFFFF50C,__READ_WRITE ,__esmiecr1_bits);
__IO_REG32_BIT(ESMILSR1,          0xFFFFF510,__READ_WRITE ,__esmilsr1_bits);
__IO_REG32_BIT(ESMILCR1,          0xFFFFF514,__READ_WRITE ,__esmilcr1_bits);
__IO_REG32_BIT(ESMSR1,            0xFFFFF518,__READ_WRITE ,__esmsr_bits);
__IO_REG32_BIT(ESMSR2,            0xFFFFF51C,__READ_WRITE ,__esmsr_bits);
__IO_REG32_BIT(ESMSR3,            0xFFFFF520,__READ_WRITE ,__esmsr_bits);
__IO_REG32_BIT(ESMEPSR,           0xFFFFF524,__READ       ,__esmepsr_bits);
__IO_REG32_BIT(ESMIOFFHR,         0xFFFFF528,__READ       ,__esmioffhr_bits);
__IO_REG32_BIT(ESMIOFFLR,         0xFFFFF52C,__READ       ,__esmiofflr_bits);
__IO_REG32_BIT(ESMLTCR,           0xFFFFF530,__READ       ,__esmltcr_bits);
__IO_REG32_BIT(ESMLTCPR,          0xFFFFF534,__READ_WRITE ,__esmltcpr_bits);
__IO_REG32_BIT(ESMEKR,            0xFFFFF538,__READ_WRITE ,__esmekr_bits);
__IO_REG32_BIT(ESMSSR2,           0xFFFFF53C,__READ_WRITE ,__esmsr_bits);

/***************************************************************************
 **
 ** DMM (Data Modification Module)
 **
 ***************************************************************************/
__IO_REG32_BIT(DMMGLBCTRL,        0xFFFFF700,__READ_WRITE ,__dmmglbctrl_bits);
__IO_REG32_BIT(DMMINTSET,         0xFFFFF704,__READ_WRITE ,__dmmintset_bits);
__IO_REG32_BIT(DMMINTCLR,         0xFFFFF708,__READ_WRITE ,__dmmintset_bits);
__IO_REG32_BIT(DMMINTLVL,         0xFFFFF70C,__READ_WRITE ,__dmmintset_bits);
__IO_REG32_BIT(DMMINTFLG,         0xFFFFF710,__READ_WRITE ,__dmmintset_bits);
__IO_REG32_BIT(DMMOFF1,           0xFFFFF714,__READ       ,__dmmoff_bits);
__IO_REG32_BIT(DMMOFF2,           0xFFFFF718,__READ       ,__dmmoff_bits);
__IO_REG32(    DMMDDMDEST,        0xFFFFF71C,__READ_WRITE );
__IO_REG32_BIT(DMMDDMBL,          0xFFFFF720,__READ_WRITE ,__dmmddmbl_bits);
__IO_REG32_BIT(DMMDDMPT,          0xFFFFF724,__READ       ,__dmmddmpt_bits);
__IO_REG32_BIT(DMMINTPT,          0xFFFFF728,__READ_WRITE ,__dmmintpt_bits);
__IO_REG32_BIT(DMMDEST0REG1,      0xFFFFF72C,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST0BL1,       0xFFFFF730,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST0REG2,      0xFFFFF734,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST0BL2,       0xFFFFF738,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST1REG1,      0xFFFFF73C,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST1BL1,       0xFFFFF740,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST1REG2,      0xFFFFF744,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST1BL2,       0xFFFFF748,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST2REG1,      0xFFFFF74C,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST2BL1,       0xFFFFF750,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST2REG2,      0xFFFFF754,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST2BL2,       0xFFFFF758,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST3REG1,      0xFFFFF75C,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST3BL1,       0xFFFFF760,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMDEST3REG2,      0xFFFFF764,__READ_WRITE ,__dmmdestreg_bits);
__IO_REG32_BIT(DMMDEST3BL2,       0xFFFFF768,__READ_WRITE ,__dmmdestbl_bits);
__IO_REG32_BIT(DMMPC0,            0xFFFFF76C,__READ_WRITE ,__dmmpc0_bits);
__IO_REG32_BIT(DMMPC1,            0xFFFFF770,__READ_WRITE ,__dmmpc1_bits);
__IO_REG32_BIT(DMMPC2,            0xFFFFF774,__READ_WRITE ,__dmmpc2_bits);
__IO_REG32_BIT(DMMPC3,            0xFFFFF778,__READ_WRITE ,__dmmpc3_bits);
__IO_REG32_BIT(DMMPC4,            0xFFFFF77C,__READ_WRITE ,__dmmpc4_bits);
__IO_REG32_BIT(DMMPC5,            0xFFFFF780,__READ_WRITE ,__dmmpc5_bits);
__IO_REG32_BIT(DMMPC6,            0xFFFFF784,__READ_WRITE ,__dmmpc6_bits);
__IO_REG32_BIT(DMMPC7,            0xFFFFF788,__READ_WRITE ,__dmmpc7_bits);
__IO_REG32_BIT(DMMPC8,            0xFFFFF78C,__READ_WRITE ,__dmmpc8_bits);

/***************************************************************************
 **
 ** RTP (RAM Trace Port Module)
 **
 ***************************************************************************/
__IO_REG32_BIT(RTPGLBCTRL,        0xFFFFFA00,__READ_WRITE ,__rtpglbctrl_bits);
__IO_REG32_BIT(RTPTRENA,          0xFFFFFA04,__READ       ,__rtptrena_bits);
__IO_REG32_BIT(RTPGSR,            0xFFFFFA08,__READ_WRITE ,__rtpgsr_bits);
__IO_REG32_BIT(RTPRAM1REG1,       0xFFFFFA0C,__READ_WRITE ,__rtpramreg_bits);
__IO_REG32_BIT(RTPRAM1REG2,       0xFFFFFA10,__READ_WRITE ,__rtpramreg_bits);
__IO_REG32_BIT(RTPRAM2REG1,       0xFFFFFA14,__READ_WRITE ,__rtpramreg_bits);
__IO_REG32_BIT(RTPRAM2REG2,       0xFFFFFA18,__READ_WRITE ,__rtpramreg_bits);
__IO_REG32_BIT(RTPPERREG1,        0xFFFFFA24,__READ_WRITE ,__rtpperreg_bits);
__IO_REG32_BIT(RTPPERREG2,        0xFFFFFA28,__READ_WRITE ,__rtpperreg_bits);
__IO_REG32(    RTPDDMW,           0xFFFFFA2C,__READ_WRITE );
__IO_REG32_BIT(RTPPC0,            0xFFFFFA34,__READ_WRITE ,__rtppc0_bits);
__IO_REG32_BIT(RTPPC1,            0xFFFFFA38,__READ_WRITE ,__rtppc1_bits);
__IO_REG32_BIT(RTPPC2,            0xFFFFFA3C,__READ       ,__rtppc2_bits);
__IO_REG32_BIT(RTPPC3,            0xFFFFFA40,__READ_WRITE ,__rtppc3_bits);
__IO_REG32_BIT(RTPPC4,            0xFFFFFA44,__READ_WRITE ,__rtppc4_bits);
__IO_REG32_BIT(RTPPC5,            0xFFFFFA48,__READ_WRITE ,__rtppc5_bits);
__IO_REG32_BIT(RTPPC6,            0xFFFFFA4C,__READ_WRITE ,__rtppc6_bits);
__IO_REG32_BIT(RTPPC7,            0xFFFFFA50,__READ_WRITE ,__rtppc7_bits);
__IO_REG32_BIT(RTPPC8,            0xFFFFFA54,__READ_WRITE ,__rtppc8_bits);

/* Assembler-specific declarations **********************************************/

#ifdef __IAR_SYSTEMS_ASM__


#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **    TMS570LS20206 INTERRUPT VALUES
 **
***************************************************************************/
/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV  0x00  /* Reset                           */
#define UNDEFV  0x04  /* Undefined instruction           */
#define SWIV    0x08  /* Software interrupt              */
#define PABORTV 0x0c  /* Prefetch abort                  */
#define DABORTV 0x10  /* Data abort                      */
#define IRQV    0x18  /* Normal interrupt                */
#define FIQV    0x1c  /* Fast interrupt                  */

/***************************************************************************
 **
 **  CIM interrupt channels
 **
 ***************************************************************************/
#define CIM_SPI1           0  /* SPI1 end-transfer/overrun          */
#define CIM_COMP2          1  /* COMP2 interrupt                    */
#define CIM_COMP1          2  /* COMP1 interrupt                    */
#define CIM_TAP            3  /* TAP interrupt                      */
#define CIM_SPI2           4  /* SPI2 end-transfer/overrun          */
#define CIM_GIOA           5  /* GIO interrupt A                    */
/*#define CIM_RES          6     --- Reserved ---                     */
#define CIM_HET1           7  /* HET interrupt 1                    */
#define CIM_I2C1           8  /* I2C1 interrupt                     */
#define CIM_SCIRXERR       9  /* SCI1 or SCI2 error interrupt       */
#define CIM_SCI1RX        10  /* SCI1 receive interrupt             */
/*#define CIM_RES         11     --- Reserved ---                     */
#define CIM_I2C2          12  /* I2C2 interrupt                     */
#define CIM_HECC1A        13  /* HECC1 interrupt A                  */
/*#define CIM_SCCA        14     SCC interrupt A                      */
#define CIM_SPI3          15  /* SPI3 end-transfer/overrun          */
#define CIM_MIBADCEE      16  /* MibADC end event conversion        */
#define CIM_SCI2RX        17  /* SCI2 receive interrupt             */
#define CIM_DMA0          18  /* DMA interrupt 0                    */
#define CIM_I2C3          19  /* I2C3 interrupt                     */
#define CIM_SCI1TX        20  /* SCI1 transmit interrupt            */
#define CIM_SSI           21  /* SW interrupt (SSI)                 */
/*#define CIM_RES         22     --- Reserved ---                     */
#define CIM_HET2          23  /* HET interrupt 2                    */
#define CIM_HECC1B        24  /* HECC1 interrupt B                  */
/*#define CIM_SCCB        25     SCC interrupt B                      */
#define CIM_SCI2TX        26  /* SCI2 transmit interrupt            */
#define CIM_MIBADCE1      27  /* MibADC end Group 1 conversion      */
#define CIM_DMA1          28  /* DMA Interrupt 1                    */
#define CIM_GIOB          29  /* GIO interrupt B                    */
#define CIM_MIBADCE2      30  /* MibADC end Group 2 conversion      */
#define CIM_SCI3          31  /* SCI3 error interrupt               */

/***************************************************************************
 **
 **  IEM interrupt channels
 **
 ***************************************************************************/
/*#define IEM_RES          32     --- Reserved ---                     */
/*#define IEM_RES          33     --- Reserved ---                     */
/*#define IEM_RES          34     --- Reserved ---                     */
/*#define IEM_RES          35     --- Reserved ---                     */
/*#define IEM_RES          36     --- Reserved ---                     */
/*#define IEM_RES          37     --- Reserved ---                     */
#define IEM_HECC2A         38  /* HECC1 interrupt B                  */
#define IEM_HECC2B         39  /* HECC1 interrupt B                  */
/*#define IEM_SCI3RX       40     SCI2 receive interrupt               */
/*#define IEM_SCI3TX       41     SCI1 transmit interrupt              */
/*#define IEM_I2C4         42     I2C4 interrupt                       */
/*#define IEM_I2C5         43     I2C5 interrupt                       */

/*#define IEM_RES          44     --- Reserved ---                     */
/*#define IEM_RES          45     --- Reserved ---                     */
/*#define IEM_RES          46     --- Reserved ---                     */
/*#define IEM_RES          47     --- Reserved ---                     */



#endif    /* __IOTMS570LS20206_H */
